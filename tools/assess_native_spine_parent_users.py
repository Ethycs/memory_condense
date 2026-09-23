"""Check append-only parent-user routing on all 100 questions without answer calls.

The existing application is reopened once. Packets seal before references open;
coverage is diagnostic and every candidate packet is independently reconstructed.
"""
import argparse
from collections import Counter
from contextlib import closing
from functools import lru_cache
import json
from pathlib import Path
import sqlite3
import statistics

from tools import evaluate_native_spine_user_layout100 as evaluation
from tools import native_spine_parent_user_presentation as presentation
from tools import assess_native_spine_packet_width as coverage
from memory_condense.application import native_spine_parent_users as application
from memory_condense.search import native_spine_parent_user_routing as routing
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding, bound


BASELINE = Path('eval_results/native-spine-app-userlayout100-20260915-r1')
PARENTS = Path('eval_results/native-spine-parent-users-20260915-r1/report.json')


def run(root, policy_path):
    if root.exists():
        raise ValueError('policy assessment requires a fresh root')
    evaluation.frozen.require_idle()
    baseline = read_sealed_json(BASELINE / 'preflight.json')
    questions, scope = evaluation.validate_plan(baseline)
    policy = read_sealed_json(policy_path)
    evaluation.context_policy.validate_policy(policy.payload)
    if policy.payload != bound(baseline.payload['context_policy']).payload:
        raise ValueError('additive comparison requires the unchanged baseline numeric policy')
    parent_report = read_sealed_json(PARENTS)
    parent_file = parent_report.payload['parent_file']
    if evaluation.digest(parent_file['path']) != parent_file['sha256']:
        raise ValueError('persisted parent summary vectors changed')
    verification = bound(baseline.payload['application_lifecycle_verification'])
    ingested, _ = evaluation.application_admission(verification)
    for name, sha in ingested.payload['application_files'].items():
        if evaluation.digest(evaluation.APPLICATION / 'application' / name) != sha:
            raise ValueError('persisted application data changed')
    plan, _ = publish_sealed_json(root / 'preflight.json', {
        'baseline': binding(baseline), 'policy': binding(policy), 'parent_compilation': binding(parent_report), 'questions': binding(questions),
        'scope': binding(scope), 'application_verification': binding(verification),
        'implementation': {**evaluation.implementation(), coverage.__file__: evaluation.digest(coverage.__file__),
                           __file__: evaluation.digest(__file__),
                           application.__file__: evaluation.digest(application.__file__),
                           routing.__file__: evaluation.digest(routing.__file__),
                           application.store.__file__: evaluation.digest(application.store.__file__),
                           presentation.__file__: evaluation.digest(presentation.__file__),
                           presentation.renderer.__file__: evaluation.digest(presentation.renderer.__file__)},
        'routing_strategy': routing.FORMAT,
        'history_count': 1, 'question_count': 100, 'namespace_load_count': 1,
        'new_ingestions': 0, 'new_answer_calls': 0, 'new_qwen_calls': 0,
        'references_opened_after_packet_seal': True, 'accuracy_claim_permitted': False})
    packets, rows = [], []
    with closing(evaluation.frozen.EmbeddingService(device='cuda', batch_size=8)) as encoder:
        with application.ParentUserMemoryCondenser(evaluation.APPLICATION / 'application', embedder=encoder,
                                       auto_extract=False, read_only=True) as app:
            if app.native_spine_receipt() != verification.payload['snapshot']:
                raise ValueError('application snapshot changed')
            if app.native_parent_user_receipt() != parent_report.payload['parent_snapshot']:
                raise ValueError('reopened parent snapshot changed')
            order = presentation.renderer.TranscriptOrder(app.transcript.get_all())
            encoder.embed_query('Sealed policy diagnostic warmup.')
            cases = evaluation.baseline.validate_population(questions, scope)
            for case in cases:
                q = evaluation.frozen.question(case)
                result = app.retrieve_native_spine(q['retrieval_query'], q['prompt_question'], **policy.payload)
                rendered = presentation.renderer.render_user_spine_sections(result.hydration, order)
                before = read_sealed_json(BASELINE / 'evidence' / f'{q["ordinal"]:03d}.json').payload
                old_route = before['routing']['parent_context']
                new_route = result.routing.identity_payload()
                routing.route_from_payload(new_route)
                preserved = new_route.get('parent_base', new_route)
                old_sections = before['hydration']['parent_context']['sections']
                if (preserved != old_route
                        or result.hydration.identity_payload()['sections'][:len(old_sections)] != old_sections):
                    raise ValueError('additive routing displaced prior selection or hydrated evidence')
                packet, _ = publish_sealed_json(root / 'packets' / f'{q["ordinal"]:03d}.json', {
                    'question': q, 'hydration': result.hydration.identity_payload(),
                    'routing': result.routing.identity_payload(), 'rendered': rendered.identity_payload()})
                packets.append({'ordinal': q['ordinal'], 'packet': binding(packet)})
                if (case['ordinal'] + 1) % 10 == 0:
                    print({'packet_questions_complete': case['ordinal'] + 1, 'required': 100}, flush=True)
            sealed, _ = publish_sealed_json(root / 'packets-complete.json', {
                'preflight': binding(plan), 'packets': packets, 'packet_count': len(packets)})
            references = bound(questions.payload['references'])
            refs = {r['question_id']: r for r in references.payload['references']}
            if (set(refs) != {q['question_id'] for q in cases} or len(refs) != 100
                    or references.payload['scope_sha256'] != scope.sha256
                    or references.payload['ingest_use_permitted'] is not False):
                raise ValueError('reference population or evaluation-only boundary changed')
            for case in cases:
                ordinal = case['ordinal']
                candidate = read_sealed_json(root / 'packets' / f'{ordinal:03d}.json').payload
                old = read_sealed_json(BASELINE / 'evidence' / f'{ordinal:03d}.json').payload
                rows.append({'ordinal': ordinal, 'question_id': case['question_id'], 'arms': {
                    name: {'support_coverage': coverage.support_coverage(hydration, refs[case['question_id']], app),
                           'conversations': rendering['conversation_count'], 'rendered_tokens': rendering['token_count']}
                    for name, hydration, rendering in (
                        ('baseline', old['hydration']['parent_context'], old['rendered']['parent_context']),
                        ('candidate', candidate['hydration'], candidate['rendered']))}})
    bank_path = Path(scope.payload['body_bank']['path'])
    if evaluation.digest(bank_path) != scope.payload['body_bank']['sha256']:
        raise ValueError('original raw source bank changed')
    sessions = bound(scope.payload['namespace']).payload['sessions']
    spans = 0
    with closing(sqlite3.connect(bank_path.as_uri() + '?mode=ro', uri=True)) as raw:
        @lru_cache(maxsize=1024)
        def load_body(sha):
            return json.loads(raw.execute('SELECT body_json FROM bodies WHERE body_sha256=?', (sha,)).fetchone()[0])
        source_order = presentation.source_order(sessions, load_body)
        for entry in packets:
            p = bound(entry['packet']).payload
            routing.route_from_payload(p['routing'])
            _, count = presentation.verify_packet(p['question'], p['hydration'], p['routing'],
                p['rendered'], sessions, load_body, policy.payload, source_order)
            spans += count
    summaries = {arm: {
        'recorded_support_coverage': dict(Counter(r['arms'][arm]['support_coverage'] for r in rows)),
        'median_conversations': statistics.median(r['arms'][arm]['conversations'] for r in rows),
        'median_rendered_tokens': statistics.median(r['arms'][arm]['rendered_tokens'] for r in rows)}
        for arm in ('baseline', 'candidate')}
    lost = [r['ordinal'] for r in rows if r['arms']['baseline']['support_coverage'] == 'all'
            and r['arms']['candidate']['support_coverage'] != 'all']
    gained = [r['ordinal'] for r in rows if r['arms']['baseline']['support_coverage'] != 'all'
              and r['arms']['candidate']['support_coverage'] == 'all']
    report, _ = publish_sealed_json(root / 'report.json', {
        'preflight': binding(plan), 'packets': binding(sealed), 'references': binding(references),
        'rows': rows, 'summaries': summaries, 'lost_all_support_ordinals': lost, 'gained_all_support_ordinals': gained,
        'history_count': 1, 'question_count': 100, 'verified_memory_packets': len(packets),
        'verified_raw_spans': spans, 'new_answer_calls': 0, 'answer_accuracy_measured': False,
        'prior_routes_and_hydrated_evidence_preserved': True, 'routing_strategy': routing.FORMAT,
        'packets_with_addition': sum('parent_base' in bound(entry['packet']).payload['routing'] for entry in packets),
        'development_set': True})
    print({'report_sha256': report.sha256, 'summaries': summaries, 'lost_all_support_ordinals': lost,
           'gained_all_support_ordinals': gained, 'verified_raw_spans': spans, 'new_answer_calls': 0}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.policy)
