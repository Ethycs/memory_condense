"""Bounded four-prompt transport/latency check before a full answer-model run.

The prompts are fixed, evenly spaced ordinals from the existing 100-question
preflight. No references open and no accuracy claim is made from this probe.
"""
import argparse
from contextlib import closing
from pathlib import Path
import statistics

from tools import evaluate_native_spine_application_model100 as evaluation
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding


def run(root, model, enable=False):
    if not enable or root.exists():
        raise ValueError('explicit provider flag and a fresh output root are required')
    evaluation.frozen.require_idle()
    inventory = read_sealed_json(evaluation.MODEL_INVENTORY)
    evaluation.validate_model(model, inventory)
    previous = read_sealed_json(evaluation.PACKET_BASELINE / 'preflight.json')
    evaluation.previous.validate_plan(previous)
    ordinals = (0, 25, 50, 75)
    calls = [c for c in previous.payload['calls']
             if c['arm'] == 'parent_context' and c['question']['ordinal'] in ordinals]
    if tuple(c['question']['ordinal'] for c in calls) != ordinals:
        raise ValueError('fixed evenly spaced probe population changed')
    plan, _ = publish_sealed_json(root / 'preflight.json', {
        'implementation': {**evaluation.implementation(), __file__: evaluation.digest(__file__)},
        'baseline': binding(previous), 'model_inventory': binding(inventory), 'model': model,
        'calls': calls, 'answer_calls': 4, 'new_ingestions': 0, 'new_qwen_calls': 0,
        'raw_evidence_unchanged': True, 'references_opened': False,
        'accuracy_claim_permitted': False, 'automatic_retries': 0})
    rows = []
    with closing(evaluation.frozen._completion_client('LITELLM_KEY', evaluation.frozen.GATEWAY)) as client:
        for i, call in enumerate(calls):
            prefix = root / 'journal' / f'{i:03d}'
            request, _ = publish_sealed_json(prefix.with_suffix('.request.json'), {
                'preflight': binding(plan), 'model': model, 'max_tokens': 256, 'call': call})
            with prefix.with_suffix('.reserved').open('x', encoding='utf-8') as stream:
                stream.write(request.sha256 + '\n')
            measurement = evaluation.frozen.measure_streaming_answer(client=client, model=model,
                prepare_prompt=lambda: call['messages'], max_tokens=256)
            response, _ = publish_sealed_json(prefix.with_suffix('.response.json'), {
                'request': binding(request), 'measurement': measurement})
            rows.append({'ordinal': call['question']['ordinal'], 'response': binding(response),
                         'total_s': measurement['e2e_total_s'], 'finish_reason': measurement['finish_reason'],
                         'response_model': measurement.get('response_model')})
            print(rows[-1], flush=True)
    report, _ = publish_sealed_json(root / 'report.json', {
        'preflight': binding(plan), 'rows': rows,
        'median_total_s': statistics.median(r['total_s'] for r in rows),
        'all_stopped': all(r['finish_reason'] == 'stop' for r in rows),
        'accuracy_measured': False, 'new_answer_calls': len(rows), 'new_judge_calls': 0})
    print({'report_sha256': report.sha256, 'median_total_s': report.payload['median_total_s'],
           'all_stopped': report.payload['all_stopped'], 'accuracy_measured': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', choices=sorted(evaluation.ANSWER_MODELS), required=True)
    parser.add_argument('--enable-provider', action='store_true')
    args = parser.parse_args()
    run(args.root, args.model, args.enable_provider)
