"""Compare two fixed packet widths over one persisted history, without answers.

References are opened only after all candidate packets are sealed. Support-quote
coverage is a diagnostic on this exposed development set, never answer accuracy.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
import statistics
import time

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.application.threaded_section_context import TranscriptOrder, render_threaded_sections
from memory_condense.domain._discourse_identity import identity_sha256
from tools import evaluate_native_spine_application_reader100 as previous
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding, bound


BASELINE = Path('eval_results/native-spine-app-reader100-v7-20260915-r1')


def support_coverage(hydration, reference, app):
    if not reference['supports']:
        raise ValueError('support coverage requires a recorded quote')
    source = reference['source']['source']
    expected_source = 'native-source-' + source['occurrence_id']
    intervals = defaultdict(list)
    for section in hydration['sections']:
        for evidence in section['evidence']:
            s = evidence['span']
            if s['source_id'] == expected_source:
                intervals[s['turn_id']].append((s['start_char'], s['end_char'], evidence['text']))
    texts = {}
    for turn_id, spans in intervals.items():
        merged = []
        for start, end, text in sorted(spans):
            if merged and start == merged[-1][1]:
                first, _, earlier = merged[-1]
                merged[-1] = (first, end, earlier + text)
            elif merged and start < merged[-1][1]:
                raise ValueError('selected raw spans overlap')
            else:
                merged.append((start, end, text))
        texts[turn_id] = [row[2] for row in merged]
    found = []
    for support in reference['supports']:
        turn_id = 'native-turn-' + identity_sha256({'occurrence_id': source['occurrence_id'],
            'body_sha256': source['body_sha256'], 'turn_ordinal': support['turn_index']})
        raw = app.transcript.get_turn(turn_id)
        if (raw is None or raw.source_id != expected_source or raw.role != 'user'
                or support['quote'] not in raw.text):
            raise ValueError('recorded support differs from the persisted original user turn')
        found.append(any(support['quote'] in text for text in texts.get(turn_id, ())))
    return 'all' if all(found) else 'partial' if any(found) else 'none'


def run(root):
    if root.exists():
        raise ValueError('packet-width assessment requires a fresh root')
    previous.frozen.require_idle()
    baseline = read_sealed_json(BASELINE / 'preflight.json')
    questions, scope = previous.validate_plan(baseline)
    policy = bound(baseline.payload['context_policy'])
    variants = {str(width): {**policy.payload, 'max_direct': width} for width in (16, 8)}
    for config in variants.values():
        previous.context_policy.validate_policy(config)
    verification = bound(baseline.payload['application_lifecycle_verification'])
    ingested, _ = previous.application_admission(verification)
    for name, sha in ingested.payload['application_files'].items():
        if previous.digest(previous.APPLICATION / 'application' / name) != sha:
            raise ValueError('persisted application data changed')
    plan, _ = publish_sealed_json(root / 'preflight.json', {
        'baseline': binding(baseline), 'questions': binding(questions), 'scope': binding(scope),
        'variants': variants, 'implementation': {**previous.implementation(), __file__: previous.digest(__file__)},
        'history_count': 1, 'question_count': 100, 'namespace_load_count': 1,
        'new_answer_calls': 0, 'new_qwen_calls': 0, 'new_ingestions': 0,
        'references_opened_after_packet_seal': True, 'accuracy_claim_permitted': False})
    packets = []
    with closing(previous.frozen.EmbeddingService(device='cuda', batch_size=8)) as encoder:
        with MemoryCondenser(previous.APPLICATION / 'application', embedder=encoder,
                             auto_extract=False, read_only=True) as app:
            if app.native_spine_receipt() != verification.payload['snapshot']:
                raise ValueError('application snapshot changed')
            order = TranscriptOrder(app.transcript.get_all())
            encoder.embed_query('Packet width assessment warmup.')
            for case in previous.baseline.validate_population(questions, scope):
                q = previous.frozen.question(case)
                for width, config in variants.items():
                    start = time.perf_counter()
                    result = app.retrieve_native_spine(q['retrieval_query'], q['prompt_question'], **config)
                    rendered = render_threaded_sections(result.hydration, order)
                    elapsed = time.perf_counter() - start
                    packet, _ = publish_sealed_json(root / 'packets' / width / f'{q["ordinal"]:03d}.json', {
                        'question': q, 'hydration': result.hydration.identity_payload(),
                        'routing': result.routing.identity_payload(), 'rendered': rendered.identity_payload(),
                        'prepare_s': elapsed})
                    packets.append({'width': width, 'ordinal': q['ordinal'], 'packet': binding(packet)})
                if (case['ordinal'] + 1) % 10 == 0:
                    print({'packet_questions_complete': case['ordinal'] + 1, 'variants': 2}, flush=True)
            sealed, _ = publish_sealed_json(root / 'packets-complete.json', {
                'preflight': binding(plan), 'packets': packets, 'packet_count': len(packets)})
            references = bound(questions.payload['references'])
            refs = {r['question_id']: r for r in references.payload['references']}
            rows = []
            for case in questions.payload['questions']:
                i = case['ordinal']
                old = read_sealed_json(BASELINE / 'evidence' / f'{i:03d}.json').payload
                by_width = {'32': {'hydration': old['hydration']['parent_context'],
                                   'rendered': old['rendered']['parent_context']}}
                by_width.update({width: read_sealed_json(root / 'packets' / width / f'{i:03d}.json').payload
                                 for width in variants})
                rows.append({'ordinal': i, 'question_id': case['question_id'], 'widths': {
                    width: {'support_coverage': support_coverage(data['hydration'], refs[case['question_id']], app),
                        'conversations': data['rendered']['conversation_count'],
                        'rendered_tokens': data['rendered']['token_count'],
                        'served_spans': sum(len(s['evidence']) for s in data['hydration']['sections'])}
                    for width, data in by_width.items()}})
    summaries = {width: {
        'recorded_support_coverage': dict(Counter(row['widths'][width]['support_coverage'] for row in rows)),
        'median_conversations': statistics.median(row['widths'][width]['conversations'] for row in rows),
        'median_rendered_tokens': statistics.median(row['widths'][width]['rendered_tokens'] for row in rows),
        'lost_all_support_ordinals': [row['ordinal'] for row in rows
            if row['widths']['32']['support_coverage'] == 'all' and row['widths'][width]['support_coverage'] != 'all'],
        'gained_all_support_ordinals': [row['ordinal'] for row in rows
            if row['widths']['32']['support_coverage'] != 'all' and row['widths'][width]['support_coverage'] == 'all']}
        for width in ('32', '16', '8')}
    report, _ = publish_sealed_json(root / 'report.json', {
        'preflight': binding(plan), 'packets': binding(sealed), 'references': binding(references),
        'rows': rows, 'summaries': summaries, 'history_count': 1, 'question_count': 100,
        'new_answer_calls': 0, 'answer_accuracy_measured': False, 'development_set': True})
    print({'report_sha256': report.sha256, 'summaries': summaries, 'new_answer_calls': 0}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
