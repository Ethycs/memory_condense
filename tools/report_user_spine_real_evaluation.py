"""Authenticate and report the completed matched real-source development pilot.

This report replays answer/judge phases without provider clients, checks the
matched population, hierarchy topology and raw hydration, and keeps benchmark
and source-derived development scores separate. It never promotes a router.
"""

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path

from memory_condense.application.section_retrieval import HydratedSection, HydratedSectionSpan
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval._binary_judge_protocol import parse_binary_judge_verdict
from memory_condense.eval._retrieval_qa_prompt import QA_SYSTEM_PROMPT, QA_USER_TEMPLATE, QA_NO_CONTEXT
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from tools.assay_user_spine_hierarchy import _turns
from tools.evaluate_user_spine_real_pilot import answers, judge
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


def require(value, message):
    if not value:
        raise ValueError(message)


def report(matched_root, projection_root):
    roots = (matched_root, projection_root)
    scope = read_sealed_json(matched_root/'answer-judge-approval-request.json')
    resumption = read_sealed_json(matched_root/'answer-judge-resumption.json')
    require(resumption.payload['scope_sha256'] == scope.sha256, 'resumption scope changed')
    questions_digest = source_digest = hierarchy_digest = topology = None
    reference_digests, arms, artifacts, all_predictions = {}, {}, [], []
    total_answer_hits = total_judge_hits = 0
    for root, authorized in zip(roots, scope.payload['populations'], strict=True):
        preflight = read_sealed_json(root/'preflight.json')
        p = preflight.payload
        selection = read_sealed_json(root/'selection.json')
        prediction = read_sealed_json(root/'answers.json')
        judge_input = read_sealed_json(root/'judge-preflight.json')
        judgment = read_sealed_json(root/'judgments.json')
        require(preflight.sha256 == authorized['preflight_sha256'] and
                selection.sha256 == authorized['selection_sha256'], 'authorized inputs changed')
        require(all(file_sha256(Path(n)) == sha for n, sha in p['implementation'].items()), 'implementation changed')
        require(prediction.payload['selection_sha256'] == selection.sha256 and
                judge_input.payload['answers_sha256'] == prediction.sha256 and
                judgment.payload['judge_preflight_sha256'] == judge_input.sha256, 'broken phase binding')
        # These calls authenticate all unique journals, reconstruct logical rows,
        # and must reproduce the existing immutable artifacts without networking.
        with redirect_stdout(io.StringIO()):
            answers(root, False)
            judge(root, False)
        require(read_sealed_json(root/'answers.json').sha256 == prediction.sha256 and
                read_sealed_json(root/'judgments.json').sha256 == judgment.sha256, 'replay changed results')
        answer_hits = len(prediction.payload['response_journal_shas'])
        judge_hits = len(judgment.payload['response_journal_shas'])
        total_answer_hits += answer_hits
        total_judge_hits += judge_hits
        current_questions = identity_sha256(p['questions'])
        source = read_sealed_json(Path(p['source_root'])/'preflight.json')
        hierarchy = read_sealed_json(Path(p['source_root'])/'hierarchy-r4.json')
        require(source.sha256 == p['source_preflight_sha256'] and hierarchy.sha256 == p['hierarchy_sha256'], 'source changed')
        index = SectionSummaryIndex.from_json(p.get('projected_index_json', hierarchy.payload['index_json']))
        current_topology = [(s.section_id, s.source_id, s.child_section_ids, s.spans) for s in index.sections]
        if questions_digest is None:
            questions_digest, source_digest, hierarchy_digest = current_questions, source.sha256, hierarchy.sha256
            topology = current_topology
        require((questions_digest, source_digest, hierarchy_digest, topology) ==
                (current_questions, source.sha256, hierarchy.sha256, current_topology), 'arms have unmatched populations or topology')
        require((p['max_sections'], p['max_context_tokens'], p['max_raw_spans'], p['max_answer_prompt_tokens']) ==
                (3, 4096, 128, 5500), 'matched evidence budget changed')
        turns, _ = _turns(source.payload['binding'])
        by_id = {t.turn_id: t for t in turns}
        questions = {q['id']: q for q in p['questions']}
        expected = [(q['id'], arm) for q in p['questions'] for arm in p['arms']]
        for population in (selection.payload['rows'], prediction.payload['rows'], judge_input.payload['rows'], judgment.payload['rows']):
            require([(r['case_id'], r['arm']) for r in population] == expected, 'case/arm rows missing or reordered')
        for row, answer, judge_request, verdict in zip(selection.payload['rows'], prediction.payload['rows'],
                                                     judge_input.payload['rows'], judgment.payload['rows'], strict=True):
            sections = []
            if row['hydration']:
                require(row['hydration']['context_token_count'] <= 4096, 'raw budget exceeded')
                for section in row['hydration']['sections']:
                    evidence = []
                    for entry in section['evidence']:
                        span = RawSectionSpan(**entry['span'])
                        turn = by_id[span.turn_id]
                        require((span.source_id, span.role, span.created_at, span.turn_text_sha256) ==
                                (turn.source_id, turn.role, turn.created_at.isoformat(), quote_sha256(turn.text)), 'raw turn changed')
                        require(entry['text'] == turn.text[span.start_char:span.end_char], 'raw section text changed')
                        evidence.append(HydratedSectionSpan(span, entry['text']))
                    sections.append(HydratedSection(SectionSummary.from_dict(section['section']), tuple(evidence)))
            context = '\n\n'.join(s.render_raw(f'S{i}') for i, s in enumerate(sections, 1))
            messages = [{'role': 'system', 'content': QA_SYSTEM_PROMPT}, {'role': 'user', 'content': QA_USER_TEMPLATE.format(
                context=context or QA_NO_CONTEXT, question=questions[row['case_id']]['dated_question'])}]
            require(messages == row['messages'] and count_chat_prompt_token_proxy(messages) <= 5500, 'answer prompt changed')
            require(answer['prediction'] == judge_request['prediction'] == verdict['prediction'], 'judged a different prediction')
            require(answer['prediction_sha256'] == quote_sha256(answer['prediction']), 'prediction identity changed')
            require(verdict['correct'] == parse_binary_judge_verdict(verdict['verdict']), 'incorrect verdict aggregation')
            ref = judge_request['reference_sha256']
            require(ref == verdict['reference_sha256'], 'reference binding changed')
            previous = reference_digests.setdefault(row['case_id'], ref)
            require(previous == ref, 'arms used different references')
        for group, population in judgment.payload['aggregates'].items():
            for arm, values in population.items():
                rows = [r for r in judgment.payload['rows'] if r['group'] == group and r['arm'] == arm]
                require(values == {'correct': sum(r['correct'] for r in rows), 'count': len(rows)}, 'score population mismatch')
                arms.setdefault(arm, {})[group] = values
        all_predictions.extend(r for r in judgment.payload['rows'] if r['group'] == 'locked_benchmark_development')
        artifacts.append({'root': str(root), 'preflight_sha256': preflight.sha256, 'selection_sha256': selection.sha256,
            'answers_sha256': prediction.sha256, 'judge_preflight_sha256': judge_input.sha256, 'judgments_sha256': judgment.sha256,
            'answer_replay_hits': answer_hits, 'judge_replay_hits': judge_hits, 'new_replay_calls': 0})
    require(len(arms) == 5 and sum(len(p) for p in arms.values()) == 10, 'comparison arms or groups changed')
    require(total_answer_hits <= scope.payload['maximum_new_answer_calls'] and
            total_judge_hits <= scope.payload['maximum_new_judge_calls'], 'execution exceeded described call limits')
    result, _ = publish_sealed_json(matched_root/'completed-evaluation.json', {
        'implementation_sha256': file_sha256(Path(__file__)), 'resumption_sha256': resumption.sha256,
        'source_preflight_sha256': source_digest, 'hierarchy_sha256': hierarchy_digest,
        'question_population_sha256': questions_digest, 'artifacts': artifacts, 'scores': arms,
        'benchmark_predictions': all_predictions, 'same_question_reference_population': True,
        'same_hierarchy_topology_and_raw_sections': True, 'same_evidence_budget': True,
        'all_answer_packets_reconstructed_from_exact_raw': True,
        'new_answer_calls_this_resumption': total_answer_hits, 'new_judge_calls_this_resumption': total_judge_hits,
        'authenticated_replay_hits': total_answer_hits+total_judge_hits, 'new_replay_calls': 0,
        'raw_qwen_inputs': 0, 'promotion': False,
        'scope': '39 turns from three already selected real conversations; one examined benchmark question and seven source-derived development probes; not full-corpus or untouched validation'})
    print(json.dumps({'completed_evaluation_sha256': result.sha256, 'scores': arms,
        'answer_calls': total_answer_hits, 'judge_calls': total_judge_hits,
        'replay_hits': total_answer_hits+total_judge_hits, 'new_replay_calls': 0}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matched-root', type=Path, required=True)
    parser.add_argument('--projection-root', type=Path, required=True)
    args = parser.parse_args()
    report(args.matched_root, args.projection_root)
