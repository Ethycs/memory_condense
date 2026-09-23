"""Audit all 100 user-first layouts against original bodies without model calls."""
import argparse
from contextlib import closing
from functools import lru_cache
import json
from pathlib import Path
import sqlite3
import statistics

from tools import evaluate_native_spine_additive100 as previous
from tools import native_spine_user_spine_presentation as presentation
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding, bound


BASELINE = Path('eval_results/native-spine-app-additive100-20260915-r2')


def implementation():
    return {**previous.implementation(), __file__: previous.digest(__file__),
            presentation.__file__: previous.digest(presentation.__file__),
            presentation.renderer.__file__: previous.digest(presentation.renderer.__file__)}


def run(root):
    if root.exists():
        raise ValueError('layout assessment requires a fresh root')
    baseline = read_sealed_json(BASELINE / 'preflight.json')
    questions, scope = previous.validate_plan(baseline)
    policy = bound(baseline.payload['context_policy'])
    plan, _ = publish_sealed_json(root / 'preflight.json', {
        'implementation': implementation(), 'baseline': binding(baseline),
        'policy': binding(policy), 'questions': binding(questions), 'scope': binding(scope),
        'presentation': presentation.renderer.FORMAT,
        'history_count': 1, 'question_count': 100, 'new_answer_calls': 0,
        'new_ingestions': 0, 'new_qwen_calls': 0, 'references_opened': False})
    sessions = bound(scope.payload['namespace']).payload['sessions']
    path = Path(scope.payload['body_bank']['path'])
    if previous.digest(path) != scope.payload['body_bank']['sha256']:
        raise ValueError('original body bank changed')
    packets = []
    span_count = 0
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as raw:
        @lru_cache(maxsize=1024)
        def load_body(sha):
            return json.loads(raw.execute('SELECT body_json FROM bodies WHERE body_sha256=?', (sha,)).fetchone()[0])
        order = presentation.source_order(sessions, load_body)
        if order.receipt_sha256 != baseline.payload['transcript_order_sha256']:
            raise ValueError('original raw transcript order changed')
        for case in previous.baseline.validate_population(questions, scope):
            q = previous.frozen.question(case)
            old = read_sealed_json(BASELINE / 'evidence' / f'{q["ordinal"]:03d}.json')
            h, r = (old.payload[k]['parent_context'] for k in ('hydration', 'routing'))
            rendered = presentation.renderer.render_user_spine_sections(
                presentation.hydration_from_payload(h), order).identity_payload()
            _, count = presentation.verify_packet(q, h, r, rendered, sessions, load_body, policy.payload, order)
            packet, _ = publish_sealed_json(root / 'packets' / f'{q["ordinal"]:03d}.json', {
                'question': q, 'baseline_evidence': binding(old), 'hydration': h, 'routing': r, 'rendered': rendered})
            packets.append({'ordinal': q['ordinal'], 'packet': binding(packet), 'tokens': rendered['token_count']})
            span_count += count
    report, _ = publish_sealed_json(root / 'report.json', {
        'preflight': binding(plan), 'packets': packets, 'history_count': 1, 'question_count': 100,
        'verified_memory_packets': len(packets), 'verified_raw_spans': span_count,
        'median_rendered_tokens': statistics.median(p['tokens'] for p in packets),
        'max_rendered_tokens': max(p['tokens'] for p in packets),
        'hydration_and_routing_unchanged': True, 'references_opened': False,
        'new_answer_calls': 0, 'answer_accuracy_measured': False})
    print({'report_sha256': report.sha256, **{k: v for k, v in report.payload.items()
                                            if k not in ('packets', 'preflight')}}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
