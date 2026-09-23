"""Restore missing seals only after exact, model-free hierarchy reconstruction."""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from tools import resume_frozen_native_spine_stages as continuation
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding, bound


class ReplayOnly:
    def __init__(self, cache):
        self.cache = cache

    def resolve(self, pending):
        raise ValueError('saved summaries are insufficient for exact reconstruction')


def restore_missing_seal(target, reconstructed):
    """Never rewrite body bytes or replace an existing checksum."""
    target = Path(target)
    expected = read_sealed_json(reconstructed)
    sidecar = target.with_name(target.name + '.sha256')
    if target.is_symlink() or not target.is_file() or sidecar.exists() or sidecar.is_symlink():
        raise ValueError('repair requires a regular body with no checksum file')
    if target.read_bytes() != expected.path.read_bytes():
        raise ValueError('orphan body differs from exact frozen reconstruction')
    with sidecar.open('xb') as stream:
        stream.write(f'{expected.sha256}  {target.name}\n'.encode('ascii'))
        stream.flush()
        os.fsync(stream.fileno())
    return read_sealed_json(target)


def repair(root, failed_controller):
    root, failed = Path(root), Path(failed_controller).resolve()
    if root.exists():
        raise ValueError('seal repair requires a fresh audit root')
    continuation.original.evaluation.require_idle()
    failed_plan = read_sealed_json(failed/'preflight.json')
    fp = failed_plan.payload
    if fp['implementation_sha256'] != digest(continuation.__file__):
        raise ValueError('failed continuation implementation changed')
    workers = [read_sealed_json(failed/name) for name in
        ('worker-started.json', '01-parent-run.started.json')]
    continuation.require_stopped(workers)
    ended = read_sealed_json(failed/'01-parent-run.exit.json')
    if (any(w.payload['preflight_sha256'] != failed_plan.sha256 for w in workers)
            or bound(ended.payload['started']).sha256 != workers[1].sha256
            or ended.payload['exit_code'] != 1
            or ended.payload['log_sha256'] != digest(failed/'01-parent-run.log')):
        raise ValueError('failed parent execution receipt changed')
    state = continuation.source_state(fp['source_root'])
    if state != fp['source_state']:
        raise ValueError('parent state changed since the failed continuation')
    snapshot = state['parent_snapshot']
    plan = bound(snapshot['parent_preflight'])
    parent_root = plan.path.parent
    parent = continuation.parent
    if (plan.payload['producer_implementation'] != parent.implementation()
            or plan.payload['implementation'] != parent.parent.implementation()):
        raise ValueError('frozen parent implementation changed')
    scope, seed = bound(plan.payload['scope']), bound(plan.payload['seed'])
    if (seed.payload['scope_sha256'] != scope.sha256
            or seed.payload['backend_sha256'] != plan.payload['backend_sha256']
            or seed.payload['cache_sha256'] != parent.identity_sha256(seed.payload['values'])):
        raise ValueError('parent reconstruction seed changed')
    missing = [r for r in scope.payload['bodies'] if r['parent'] is None]
    start = snapshot['completed_missing_parent_bodies']
    pending = {r['body_sha256']: r for r in missing[start:start+plan.payload['body_batch_size']]}
    orphans = [p for p in sorted((parent_root/'hierarchies').glob('*.json'))
        if not p.with_name(p.name+'.sha256').exists()]
    if not orphans or any(p.stem not in pending for p in orphans):
        raise ValueError('missing seals must belong to the first incomplete batch')
    intent, _ = publish_sealed_json(root/'intent.json', {
        'implementation_sha256': digest(__file__), 'failed_controller': binding(failed_plan),
        'failed_exit': binding(ended), 'source_state': state,
        'orphans': [{'path': str(p), 'sha256': digest(p)} for p in orphans],
        'new_model_calls': 0, 'body_bytes_replaced': False})
    journal = parent.bounded.BoundedJournal(parent_root, plan,
        SimpleNamespace(identity_sha256=plan.payload['backend_sha256']), 0)
    journal.cache.values.update(seed.payload['values'])
    journal.replay()
    scorer = parent.FrozenAttention(Path(plan.payload['attention_root']))
    if (scorer.result.sha256 != plan.payload['attention_result_sha256']
            or scorer.method.sha256 != plan.payload['attention_method_sha256']):
        raise ValueError('reconstruction attention binding changed')
    groups = {p.stem: parent.preparation.load_body(pending[p.stem]) for p in orphans}
    rebuilt_root = root/'reconstructed'
    done = parent.parent.compile_groups(rebuilt_root, plan, groups, scorer, ReplayOnly(journal.cache))
    if set(done) != set(groups):
        raise ValueError('orphan reconstruction is incomplete')
    # Compare every artifact before restoring any missing seal.
    rebuilt = {p: bound(parent.preparation.scoped_binding(rebuilt_root, done[p.stem], 'hierarchies'))
        for p in orphans}
    if any(p.read_bytes() != a.path.read_bytes() for p, a in rebuilt.items()):
        raise ValueError('orphan body differs from exact frozen reconstruction')
    restored = [restore_missing_seal(p, a.path) for p, a in rebuilt.items()]
    result, _ = publish_sealed_json(root/'result.json', {
        'intent': binding(intent), 'restored': [binding(a) for a in restored],
        'reconstructed': [binding(a) for a in rebuilt.values()],
        'new_model_calls': 0, 'body_bytes_replaced': False,
        'original_parent_preflight_sha256': plan.sha256})
    print({'repair_result_sha256': result.sha256, 'restored_seals': len(restored),
        'new_model_calls': 0, 'body_bytes_replaced': False}, flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--failed-controller', type=Path, required=True)
    args = parser.parse_args()
    repair(args.root, args.failed_controller)
