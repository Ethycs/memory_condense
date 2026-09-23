"""Continue an interrupted parent stage without changing the frozen evaluation."""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import psutil

from memory_condense.domain._discourse_identity import identity_sha256
from tools import run_frozen_native_spine_stages as original
from tools import compile_remaining_native_spine_hierarchies as parent
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding, bound


def require_stopped(workers):
    for worker in workers:
        if original.same_process(worker.payload['worker']):
            raise ValueError('original worker is still live; do not start a continuation')


def checkpoint_snapshot(root, plan, scope):
    missing = [row for row in scope.payload['bodies'] if row['parent'] is None]
    size = plan.payload['body_batch_size']
    paths = sorted((root/'batches').glob('*.json'))
    checkpoints, completed = [], 0
    for ordinal, path in enumerate(paths):
        rows = missing[ordinal*size:(ordinal+1)*size]
        checkpoint = read_sealed_json(path)
        p = checkpoint.payload
        if (path.name != f'{ordinal:04d}.json' or not rows
                or p['preflight_sha256'] != plan.sha256
                or p['body_bindings_sha256'] != identity_sha256(rows)
                or set(p['completed']) != {row['body_sha256'] for row in rows}):
            raise ValueError('parent checkpoints are not a complete bound prefix')
        checkpoints.append(binding(checkpoint))
        completed += len(rows)
    # Authentication only: this object cannot load a model or generate text.
    journal = parent.bounded.BoundedJournal(root, plan,
        SimpleNamespace(identity_sha256=plan.payload['backend_sha256']), 0)
    journal.replay()
    transactions, spent = [], 0
    expected_reservations = set()
    for path in sorted((root/'requests').glob('*.json')):
        request = read_sealed_json(path)
        reservation = root/'executions'/f'{request.sha256}.reserved'
        response_path = root/'responses'/f'{request.sha256}.json'
        reserved = reservation.exists()
        if reserved:
            if reservation.read_text(encoding='utf-8').strip() != request.sha256:
                raise ValueError('parent execution reservation changed')
            expected_reservations.add(reservation.name)
            spent += len(request.payload['jobs'])
        if reserved != response_path.exists():
            raise ValueError('parent execution has an incomplete request/response pair')
        response = read_sealed_json(response_path) if reserved else None
        transactions.append([request.sha256, response.sha256 if response else None])
    if {p.name for p in (root/'executions').glob('*.reserved')} != expected_reservations:
        raise ValueError('parent execution has a foreign reservation')
    allowance = plan.payload['maximum_new_local_jobs']
    if spent > allowance:
        raise ValueError('parent execution exceeded its original job budget')
    return {'parent_preflight': binding(plan), 'checkpoints': checkpoints,
        'completed_missing_parent_bodies': completed, 'missing_parent_body_count': len(missing),
        'reserved_local_jobs': spent, 'remaining_original_job_allowance': allowance-spent,
        'authenticated_transactions_sha256': identity_sha256(transactions),
        'authenticated_accepted_summary_keys': len(journal.cache.values)}


def source_state(source_root):
    root = Path(source_root).resolve()
    plan = read_sealed_json(root/'preflight.json')
    p = plan.payload
    if (p['implementation_sha256'] != digest(original.__file__)
            or p['evaluation_implementation'] != original.evaluation.implementation()):
        raise ValueError('original frozen implementation changed')
    candidate = bound(p['candidate'])
    original.evaluation.validate_candidate(candidate.path)
    scope = bound(p['scope'])
    bound(p['vector_preflight'])
    if p['stages'] != original.stages(p['corpus_root'], p['evaluation_root'], candidate.path):
        raise ValueError('original frozen stage commands changed')
    controller = read_sealed_json(root/'worker-started.json')
    child = read_sealed_json(root/'01-parent-run.started.json')
    if (controller.payload['preflight_sha256'] != plan.sha256
            or child.payload['preflight_sha256'] != plan.sha256
            or child.payload['stage'] != p['stages'][1]):
        raise ValueError('original worker receipt changed')
    require_stopped((controller, child))
    if (root/'01-parent-run.exit.json').exists():
        raise ValueError('this continuation requires an interrupted, unrecorded parent exit')
    prepared = read_sealed_json(root/'00-parent-prepare.complete.json')
    if bound(prepared.payload['exit']).payload['exit_code'] != 0:
        raise ValueError('original parent preparation did not succeed')
    parent_plan = bound(prepared.payload['result'])
    parent_root = Path(p['corpus_root'])/'remaining-parents'
    if (parent_plan.path != (parent_root/'preflight.json').resolve()
            or bound(parent_plan.payload['scope']).sha256 != scope.sha256):
        raise ValueError('parent preparation differs from the original scope')
    for stage in p['stages'][1:]:
        if Path(stage['result']).exists():
            raise ValueError('a remaining stage already has a result; inspect it before recovery')
    if Path(p['evaluation_root']).exists():
        raise ValueError('evaluation has already started preparation; do not release it again')
    return {'original_plan': binding(plan), 'original_controller': binding(controller),
        'original_parent_worker': binding(child), 'original_processes_absent': True,
        'original_exit_code': None, 'interruption_cause': 'unknown',
        'parent_snapshot': checkpoint_snapshot(parent_root, parent_plan, scope),
        'remaining_stages': p['stages'][1:]}


def prepare(root, source_root):
    root = Path(root)
    if root.exists():
        raise ValueError('continuation requires a fresh controller root')
    original.evaluation.require_idle()
    state = source_state(source_root)
    plan, _ = publish_sealed_json(root/'preflight.json', {
        'format': 'native-spine-interrupted-parent-continuation-v1',
        'implementation_sha256': digest(__file__), 'source_root': str(Path(source_root).resolve()),
        'source_state': state, 'automatic_retries': 0, 'maximum_answer_calls': 300,
        'logical_judgments': 200, 'concurrent_gpu_models': 1, 'raw_inputs_to_qwen': False})
    print({'continuation_preflight_sha256': plan.sha256,
        **{k: v for k, v in state['parent_snapshot'].items() if k != 'checkpoints'}}, flush=True)
    return plan


def run(root):
    root = Path(root)
    plan = read_sealed_json(root/'preflight.json')
    p = plan.payload
    if p['implementation_sha256'] != digest(__file__):
        raise ValueError('continuation implementation changed')
    original.evaluation.require_idle()
    if source_state(p['source_root']) != p['source_state']:
        raise ValueError('interrupted source state changed after continuation preparation')
    with (root/'execution.reserved').open('x', encoding='utf-8') as stream:
        stream.write(plan.sha256+'\n')
    process = psutil.Process(os.getpid())
    publish_sealed_json(root/'worker-started.json', {'preflight_sha256': plan.sha256,
        'worker': {'pid': process.pid, 'create_time': process.create_time()}, 'at': original.utc_now()})
    original_plan = bound(p['source_state']['original_plan'])
    for ordinal, stage in enumerate(p['source_state']['remaining_stages'], 1):
        if original_plan.payload['evaluation_implementation'] != original.evaluation.implementation():
            raise ValueError('frozen implementation changed between continuation stages')
        result = original.launch_stage(root, plan, ordinal, stage)
    publish_sealed_json(root/'complete.json', {'original_plan': binding(original_plan),
        'joint_report': binding(result), 'target_gate_passed': result.payload['target_gate_passed'],
        'at': original.utc_now()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'run'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path)
    args = parser.parse_args()
    if args.phase == 'prepare':
        if args.source_root is None:
            parser.error('prepare requires --source-root')
        prepare(args.root, args.source_root)
    else:
        run(args.root)
