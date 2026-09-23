"""Publish a diagnostic comparison without changing any original grades."""
from collections import Counter
import json
from pathlib import Path

from memory_condense.domain._discourse_identity import quote_sha256
from tools.assemble_native_spine_summaries import digest
from tools.audit_native_spine_source_answers import validate_review
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding, bound


ROOT = Path('eval_results/native-spine-source-answer-review-20260915-r1')
# These are agent inspection notes, not human adjudications or replacement grades.
NOTES = {
    9: ('The review used a nonexact quote. The supplied user turns support the answer; retain the invalid review unchanged.', ['served-0-0', 'served-1-0']),
    13: ('The reviewer inserted ellipses into its purported exact quote. Both interpretations appear in the supplied user turn. Retain the invalid review.', ['served-1-0']),
    15: ('The reviewer joined noncontiguous source text with an ellipsis. The bake sale and both promotion methods are explicit. Retain the invalid review.', ['served-5-0']),
    17: ('A concrete qualification error: the answer says the jogs and strength training are being done; the user described those as plans. The needed user evidence was served.', ['served-4-0', 'served-8-0', 'served-10-0']),
    31: ('A concrete attribution error: the user asked whether penalties would exist. Proportionality and monetary fines came from the assistant response. The original grader passed this answer.', ['served-2-0', 'served-10-0']),
    38: ('The reviewer overstates its case: educational game is explicitly stated in a user follow-up, and Flow/NFTs also appear in the same user conversation. Those user turns were served. Whether the answer overstates commitment to the proposed technical implementation remains separate from its supported educational description.', ['served-0-0', 'served-3-0', 'served-5-0', 'served-16-0']),
    42: ('A concrete certainty error: looks like it is from the Victorian era becomes a definite Victorian-era armchair. The additional furniture-style learning goal originated in the assistant response, while the user explicitly asked about upholstery and fabric selection.', ['served-0-0', 'served-2-0', 'served-8-0']),
    53: ('Both concert sets have source support. The question lacks a distinguishing conversation or time. Preserve the ambiguous verdict and original failed grade.', ['reference-user-0', 'reference-user-6', 'served-18-0', 'served-20-0']),
    58: ('The original failed grade contradicts the explicit user statement about having attended open-mic nights. This supports a grading defect, but does not authorize adding a point to the accepted score.', ['served-2-0']),
    59: ('The reviewer overlooks the immediately relevant user statement about reading during the daily commute. Portability is explicit; connecting it specifically to commuting is a plausible inference rather than an explicit stated reason. Do not report this as an unequivocal hallucination.', ['served-0-0', 'served-3-0']),
    61: ('The reviewer inserted an ellipsis into its purported exact quote. Soapstone, birds, details and textures are explicit user requests. Retain the invalid review.', ['served-12-0', 'served-15-0']),
    79: ('The extra routine details are supported user statements from other conversations. The reviewer treats the reference conversation as the exclusive scope, which the broad question does not uniquely establish. Scope is disputed; no revised grade is assigned.', ['served-7-0', 'served-8-0', 'served-9-0', 'served-11-0', 'served-13-0', 'served-15-0', 'served-17-0']),
    81: ('The user requested a tactical/strategic rewrite. The answer additionally attributes the assistant-generated tactical specifics to that request. The year 1970 is present in the preceding story that the user was discussing, so its presence alone is not proof of invented history. Preserve this distinction when reviewing the flag.', ['served-1-0', 'served-3-0', 'served-7-0', 'served-8-0']),
    82: ('The requested drama qualities and named social issues are supported in the user turns. The original grader imposed additional named-show/character requirements from the reference.', ['served-0-0', 'served-4-0']),
    93: ('The history contains more than three supported vintage collections. The question does not uniquely select the reference set. Preserve the ambiguous verdict and original failed grade.', ['reference-user-0', 'served-2-0']),
    95: ('The extra French cinema and French New Wave interests are explicit user statements. A reference omission does not make these claims false.', ['served-1-0', 'served-17-0']),
    96: ('The answer explicitly separates household-shopping apps from other online-shopping cashback use. The source review accepts that distinction; the original failed grade is retained.', []),
}


def run():
    plan = read_sealed_json(ROOT / 'preflight.json')
    review = read_sealed_json(ROOT / 'report.json')
    original = bound(plan.payload['source_report'])
    if review.payload['preflight'] != binding(plan):
        raise ValueError('review is not bound to the source population')
    items, reviews, originals = (plan.payload['items'], review.payload['rows'], original.payload['rows'])
    for population in (items, reviews, originals):
        if [row['ordinal'] for row in population] != list(range(100)):
            raise ValueError('comparison requires the complete ordered population')
    rows = []
    for item, inspected, old in zip(items, reviews, originals, strict=True):
        if (item['question_id'] != inspected['question_id'] or item['question_id'] != old['question_id']
                or quote_sha256(item['prediction']) != old['prediction_sha256']
                or quote_sha256(item['reference']) != old['reference_sha256']):
            raise ValueError('original question, answer or reference changed')
        result = inspected['review']
        if result is not None:
            if validate_review(json.dumps(result), item) != result:
                raise ValueError('source review changed during validation')
        else:
            try:
                validate_review(inspected['completion'], item)
            except (ValueError, TypeError, KeyError) as error:
                if str(error) != inspected['validation_error']:
                    raise ValueError('invalid-review diagnosis changed') from error
            else:
                raise ValueError('previously invalid review became valid')
        note, source_ids = NOTES.get(item['ordinal'], (None, []))
        sources = {s['source_id']: s for s in item['sources']}
        rows.append({'ordinal': item['ordinal'], 'question_id': item['question_id'],
            'original_correct': old['correct'], 'original_verdict': old['verdict'],
            'review_verdict': result['verdict'] if result else 'invalid_review',
            'agent_inspection_note': note, 'inspection_sources': [sources[s] for s in source_ids]})
    counts = Counter((r['original_correct'], r['review_verdict']) for r in rows)
    if sum(r['original_correct'] for r in rows) != original.payload['accuracy']['correct']:
        raise ValueError('original score differs from its full population')
    if dict(Counter(r['review_verdict'] for r in rows)) != review.payload['review_counts']:
        raise ValueError('review counts differ from their full population')
    comparison, _ = publish_sealed_json(ROOT / 'comparison.json', {
        'implementation_sha256': digest(__file__), 'preflight': binding(plan),
        'review': binding(review), 'original_report': binding(original), 'rows': rows,
        'cross_tab': [{'original_correct': original_correct, 'review_verdict': verdict, 'count': count}
                      for (original_correct, verdict), count in sorted(counts.items())],
        'denominator': 100, 'original_accuracy': original.payload['accuracy'],
        'diagnostic_only': True, 'original_scores_changed': False, 'acceptance_gate_changed': False,
        'human_adjudication_performed': False, 'target_completion_claim_permitted': False,
        'new_answer_calls': 0, 'new_review_calls': 0})
    lines = ['# Source review of all 100 saved answers', '',
        'Original score: **94/100**. This is a diagnostic comparison, not a replacement score.', '',
        'The reviewer returned 87 correct, 7 incorrect, 2 ambiguous and 4 invalid reviews. '
        'Agent inspection notes are not human adjudications. No item is excluded and no original grade changes.', '',
        f'Comparison SHA-256: `{comparison.sha256}`', '',
        '| Original grade | Source review | Count |', '|---|---|---:|']
    for (correct, verdict), count in sorted(counts.items()):
        lines.append(f'| {"Pass" if correct else "Fail"} | {verdict} | {count} |')
    for item, inspected, row in zip(items, reviews, rows, strict=True):
        lines += ['', f'## {item["ordinal"]:03d}: {item["question"]}', '',
            '**Saved answer:** ' + item['prediction'], '', '**Locked reference:** ' + item['reference'], '',
            '**Original grade:** ' + row['original_verdict'], '', '**Source review:** ' + row['review_verdict'], '']
        if inspected['review']:
            lines += ['```json', json.dumps(inspected['review'], ensure_ascii=False, indent=2), '```', '']
        else:
            lines += ['Validation error: ' + inspected['validation_error'], '',
                      '```text', inspected['completion'], '```', '']
        if row['agent_inspection_note']:
            lines += ['**Agent source inspection:** ' + row['agent_inspection_note'], '']
        for source in row['inspection_sources']:
            lines += [f'**{source["source_id"]} — {source["role"]}; {source["origin"]}**', '',
                      '```text', source['text'], '```', '']
    rendered = '\n'.join(lines)
    destination = ROOT / 'inspection.md'
    if destination.exists() and destination.read_text(encoding='utf-8') != rendered:
        raise ValueError('refusing to overwrite different inspection text')
    if not destination.exists():
        destination.write_text(rendered, encoding='utf-8', newline='\n')
    print({'comparison_sha256': comparison.sha256, 'inspection_sha256': digest(destination),
           'questions': len(rows), 'original_accuracy': original.payload['accuracy'],
           'new_answer_calls': 0, 'new_review_calls': 0})


if __name__ == '__main__':
    run()
