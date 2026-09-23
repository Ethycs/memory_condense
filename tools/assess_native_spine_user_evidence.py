"""Independently reconstruct all 100 projected packets before answer calls."""
from contextlib import closing
from functools import lru_cache
import json
from pathlib import Path
import sqlite3
import statistics
from types import SimpleNamespace

from tools import evaluate_native_spine_user_evidence100 as evaluation
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding, bound

ROOT = Path('eval_results/native-spine-user-evidence-admission-20260915-r1')


def run():
    evaluation.frozen.require_idle()
    original = read_sealed_json(evaluation.PACKET_BASELINE / 'preflight.json')
    questions, scope = evaluation.previous.validate_plan(original)
    policy = bound(original.payload['context_policy'])
    reader = bound(original.payload['reader_policy'])
    complete = read_sealed_json(evaluation.PACKET_BASELINE / 'complete.json')
    old_report = bound(complete.payload['joint_report'])
    old_audit = bound(complete.payload['raw_audit'])
    if (old_report.payload['preflight_sha256'] != original.sha256
            or old_audit.payload['joint_report'] != binding(old_report)
            or old_audit.payload['verified_memory_packets'] != 100):
        raise ValueError('projection requires a completed and audited baseline')
    bank_path = Path(scope.payload['body_bank']['path'])
    if evaluation.digest(bank_path) != scope.payload['body_bank']['sha256']:
        raise ValueError('original raw source bank changed')
    sessions = bound(scope.payload['namespace']).payload['sessions']
    packets = []
    with closing(sqlite3.connect(bank_path.as_uri() + '?mode=ro', uri=True)) as raw:
        @lru_cache(maxsize=1024)
        def load_body(sha):
            return json.loads(raw.execute('SELECT body_json FROM bodies WHERE body_sha256=?', (sha,)).fetchone()[0])
        order = evaluation.presentation.source_order(sessions, load_body)
        if order.receipt_sha256 != original.payload['transcript_order_sha256']:
            raise ValueError('independent source order differs from original serving order')
        for case in questions.payload['questions']:
            ordinal = case['ordinal']
            old = read_sealed_json(evaluation.PACKET_BASELINE / 'evidence' / f'{ordinal:03d}.json')
            hydration = old.payload['hydration']['parent_context']
            rendered = evaluation.presentation.renderer.render_user_spine_sections(
                evaluation.presentation.hydration_from_payload(hydration), order)
            q = evaluation.frozen.question(case)
            messages = evaluation.reader.apply_reader(evaluation.context_policy.messages(q,
                SimpleNamespace(render_context=lambda: rendered.text)), evaluation.validate_reader_policy(reader.payload))
            payload = {**old.payload, 'rendered': {'parent_context': rendered.identity_payload()},
                       'messages': {a: messages for a in evaluation.ARMS}}
            evaluation.validate_preservation(SimpleNamespace(payload=payload), old)
            checked, count = evaluation.presentation.verify_packet(q, hydration,
                payload['routing']['parent_context'], rendered.identity_payload(), sessions, load_body, policy.payload, order)
            if evaluation.reader.apply_reader(checked, evaluation.validate_reader_policy(reader.payload)) != messages:
                raise ValueError('projection differs from independent raw reconstruction')
            packet, _ = publish_sealed_json(ROOT / 'packets' / f'{ordinal:03d}.json', payload)
            packets.append({'ordinal': ordinal, 'packet': binding(packet), 'verified_raw_spans': count,
                'served_raw_spans': len(rendered.identity_payload()['placements']),
                'omitted_assistant_only_sections': len(rendered.omitted_section_ids),
                'baseline_rendered_tokens': old.payload['rendered']['parent_context']['token_count'],
                'projected_rendered_tokens': rendered.identity_payload()['token_count']})
    if [p['ordinal'] for p in packets] != list(range(100)):
        raise ValueError('projection admission requires all 100 original questions')
    report, _ = publish_sealed_json(ROOT / 'report.json', {
        'baseline': binding(original), 'baseline_completion': binding(complete),
        'implementation': evaluation.implementation(), 'packets': packets,
        'history_count': 1, 'question_count': 100, 'cached_packet_reconstruction_only': True,
        'live_retrieval_measured': False, 'accuracy_measured': False,
        'references_opened': False, 'new_answer_calls': 0, 'new_qwen_calls': 0,
        'all_user_sections_retained_exactly': True,
        'verified_raw_spans': sum(p['verified_raw_spans'] for p in packets),
        'served_raw_spans': sum(p['served_raw_spans'] for p in packets),
        'omitted_assistant_only_sections': sum(p['omitted_assistant_only_sections'] for p in packets),
        'median_baseline_rendered_tokens': statistics.median(p['baseline_rendered_tokens'] for p in packets),
        'median_projected_rendered_tokens': statistics.median(p['projected_rendered_tokens'] for p in packets)})
    print({k: v for k, v in report.payload.items() if k not in {'implementation', 'packets'}}, flush=True)
    print({'report_sha256': report.sha256}, flush=True)


if __name__ == '__main__':
    run()
