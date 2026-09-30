"""Reconstruct a 2x2 routing/presentation comparison on the same saved history.

No ingestion, embeddings or answers during preparation. The two new arms keep
every hydrated role in transcript order, removing downstream assistant omission
and user-first layout together. The v7 reader, routing and budgets stay fixed.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import closing
import statistics
from pathlib import Path
from types import SimpleNamespace

from memory_condense.application.threaded_section_context import TranscriptOrder, render_threaded_sections
from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import TranscriptStore
from tools import evaluate_native_spine_heuristic_ablation as prior
from tools.matched_eval.artifacts import read_sealed_json
from tools.native_spine_threaded_presentation import hydration_from_payload

ROOT = Path('eval_results/native-spine-downstream-compensation-20260923-r2')
ARMS = ('routing_on', 'routing_off')
current = prior.current


def neutral_messages(saved, order, reader):
    hydrated = hydration_from_payload(saved['hydration'])
    rendered = render_threaded_sections(hydrated, order)
    expected = {e.span.receipt_sha256: e.text for s in hydrated.sections for e in s.evidence}
    if (set(expected) != {sha for sha, _, _ in rendered.placements}
            or any(rendered.text[a:b] != expected[sha] for sha, a, b in rendered.placements)):
        raise ValueError('neutral presentation dropped or changed hydrated evidence')
    messages = current.reader.apply_reader(current.context_policy.messages(saved['question'],
        SimpleNamespace(render_context=lambda: rendered.text)), reader)
    old_text = saved['rendered']['text']
    if not old_text or sum(m['content'].count(old_text) for m in saved['messages']) != 1:
        raise ValueError('expected one unambiguous context block in the saved prompt')
    expected_messages = [{**m, 'content': m['content'].replace(old_text, rendered.text, 1)}
                         for m in saved['messages']]
    if messages != expected_messages:
        raise ValueError('presentation comparison changed the reader or question')
    return messages, rendered.identity_payload()


def covered(quote, turn_id, spans, turns):
    parts = sorted((s['start_char'], s['end_char']) for s in spans if s['turn_id'] == turn_id)
    intervals = []
    for start, end in parts:
        if intervals and start <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
        else:
            intervals.append((start, end))
    return any(quote in turns[turn_id].text[start:end] for start, end in intervals)


def support_stages(saved, reference, turns):
    source = reference['source']['source']
    routed = [s for r in saved['routing']['expanded']['routes'] for s in r['section']['spans']]
    hydrated = [e['span'] for s in saved['hydration']['sections'] for e in s['evidence']]
    served_ids = {p['span_sha256'] for p in saved['rendered']['placements']}
    served = [s for s in hydrated if s['receipt_sha256'] in served_ids]
    results = []
    for support in reference['supports']:
        turn_id = 'native-turn-' + identity_sha256({'occurrence_id': source['occurrence_id'],
            'body_sha256': source['body_sha256'], 'turn_ordinal': support['turn_index']})
        turn = turns[turn_id]
        if turn.role != 'user' or support['quote'] not in turn.text:
            raise ValueError('reference quote is not an exact original user statement')
        found = [covered(support['quote'], turn_id, spans, turns) for spans in (routed, hydrated, served)]
        results.append('present' if all(found) else ('routing', 'hydration', 'projection')[found.index(False)])
    return results


def prepare(root):
    if root.exists():
        raise ValueError('prepare requires a fresh directory')
    previous = read_sealed_json(prior.DEFAULT_ROOT / 'report.json')
    previous_plan = prior.bound(previous.payload['preflight'])
    p = previous_plan.payload
    questions = prior.bound(p['questions'])
    ingested = prior.bound(p['ingestion'])
    reader = current.validate_reader_policy(prior.bound(p['reader_policy']).payload)
    application = Path(p['application'])
    for name, sha in ingested.payload['application_files'].items():
        if prior.digest(application / name) != sha:
            raise ValueError('persisted application changed')
    plan = prior.publish(root / 'preflight.json', {
        'runner_sha256': prior.digest(__file__), 'prior_report': prior.binding(previous),
        'prior_preflight': prior.binding(previous_plan), 'question_count': 100, 'history_count': 1,
        'arms': list(ARMS), 'model': p['model'], 'reader_policy': p['reader_policy'],
        'change': 'keep every hydrated role; chronological interleaving instead of user-first blocks',
        'unchanged': ['exact hydrated spans', 'routing', 'raw budget', 'reader', 'questions', 'grader'],
        'cached_packet_replay': True, 'new_ingestion': False, 'new_qwen_calls': 0,
        'new_embedding_calls': 0, 'measure_warm_retrieval_latency': False,
        'answer_concurrency': 4, 'judge_concurrency': 8, 'max_tokens': 256,
        'historical_cells': {'routing_on_projected': 94, 'routing_off_projected': 73},
        'limitations': ['joint presentation change does not isolate omission from layout',
            'v7 evidence-attribution reader retained in every cell',
            'historical projected cells; same exposed development questions',
            'cached packets and concurrent answers do not measure interactive serving latency']})
    packets, saved_by_arm = [], {arm: [] for arm in ARMS}
    with closing(Database(application / 'memory.db', read_only=True)) as db:
        turns = TranscriptStore(db).get_all()
        order = TranscriptOrder(turns)
        turn_map = {t.turn_id: t for t in turns}
        for i in range(100):
            # Alternate which arm is scheduled first; the corpus never changes.
            for arm in ARMS if i % 2 == 0 else ARMS[::-1]:
                folder = Path(p['baseline_folder']) if arm == 'routing_on' else prior.DEFAULT_ROOT
                saved = read_sealed_json(folder / 'answers' / f'{i:03d}.response.json')
                if saved.payload['question'] != current.frozen.question(questions.payload['questions'][i]):
                    raise ValueError('saved packet belongs to a different question')
                messages, rendered = neutral_messages(saved.payload, order, reader)
                packet = prior.publish(root / 'packets' / f'{i:03d}-{arm}.json', {
                    'source': prior.binding(saved), 'ordinal': i, 'arm': arm,
                    'messages': messages, 'rendered': rendered})
                packets.append(prior.binding(packet))
                saved_by_arm[arm].append(saved.payload)
        population = prior.publish(root / 'packets.json', {'preflight': prior.binding(plan),
            'packets': packets, 'references_opened_during_construction': False,
            'exact_evidence_populations_verified': 200})
        # Diagnostics only after every candidate packet is frozen. No reference
        # or observed outcome participates in either presentation transform.
        refs = prior.bound(questions.payload['references'])
        references = {r['question_id']: r for r in refs.payload['references']}
        rows, summaries = [], {}
        for arm in ARMS:
            arm_rows = []
            for i, saved in enumerate(saved_by_arm[arm]):
                ref = references[questions.payload['questions'][i]['question_id']]
                stages = support_stages(saved, ref, turn_map)
                hydrated = saved['hydration']
                omitted = set(saved['rendered']['omitted_section_ids'])
                wasted = sum(e['span']['token_count'] for s in hydrated['sections']
                             if s['section']['section_id'] in omitted for e in s['evidence'])
                roles = {r['section']['section_id']: r['section']['spans'][0]['role']
                         for r in saved['routing']['expanded']['routes']}
                budget_user = sum(d['reason'] == 'context_budget' and roles[d['section_id']] == 'user'
                                  for d in hydrated['diagnostics'])
                row = {'ordinal': i, 'arm': arm, 'support_stages': stages,
                    'raw_tokens_discarded_after_hydration': wasted,
                    'routed_user_sections_rejected_for_budget': budget_user,
                    'served_context_tokens': saved['rendered']['token_count'],
                    'original_correct': previous.payload['rows'][i]['baseline_correct' if arm == 'routing_on' else 'correct']}
                rows.append(row)
                arm_rows.append(row)
            summaries[arm] = {
                'support_quote_stages': dict(Counter(s for r in arm_rows for s in r['support_stages'])),
                'all_supports_present_before_projection': sum(all(s in ('present', 'projection') for s in r['support_stages']) for r in arm_rows),
                'all_supports_present_after_projection': sum(all(s == 'present' for s in r['support_stages']) for r in arm_rows),
                'packets_with_assistant_omission': sum(r['raw_tokens_discarded_after_hydration'] > 0 for r in arm_rows),
                'mean_raw_tokens_discarded_after_hydration': statistics.fmean(r['raw_tokens_discarded_after_hydration'] for r in arm_rows),
                'packets_with_user_budget_rejection': sum(r['routed_user_sections_rejected_for_budget'] > 0 for r in arm_rows)}
    result = prior.publish(root / 'assessment.json', {'preflight': prior.binding(plan),
        'packets': prior.binding(population), 'references': prior.binding(refs),
        'summaries': summaries, 'rows': rows, 'answer_calls': 0})
    prior.emit(phase='assessment_complete', summaries=summaries, report=str(result.path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    prepare(parser.parse_args().root)
