from types import SimpleNamespace

import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from tools import resume_frozen_native_spine_stages as recovery
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding


def test_live_original_child_prevents_continuation(monkeypatch):
    monkeypatch.setattr(recovery.original, 'same_process', lambda worker: worker['pid'] == 2)
    workers = [SimpleNamespace(payload={'worker': {'pid': pid}}) for pid in (1, 2)]
    with pytest.raises(ValueError, match='still live'):
        recovery.require_stopped(workers)
    recovery.require_stopped(workers[:1])


def snapshot_fixture(root, *, checkpoint_name='0000.json', changed=False):
    rows = [{'body_sha256': str(i), 'parent': None} for i in range(4)]
    plan, _ = publish_sealed_json(root/'preflight.json', {
        'body_batch_size': 2, 'backend_sha256': 'fixture', 'maximum_new_local_jobs': 4096})
    publish_sealed_json(root/'batches'/checkpoint_name, {
        'preflight_sha256': plan.sha256, 'body_bindings_sha256': identity_sha256(rows[:2]),
        'completed': {str(i): {} for i in range(2)}})
    if changed:
        rows[0]['body_sha256'] = 'foreign'
    return plan, SimpleNamespace(payload={'bodies': rows})


def test_snapshot_authenticates_completed_prefix_without_loading_a_model(tmp_path):
    plan, scope = snapshot_fixture(tmp_path)
    snapshot = recovery.checkpoint_snapshot(tmp_path, plan, scope)
    assert snapshot['completed_missing_parent_bodies'] == 2
    assert snapshot['missing_parent_body_count'] == 4
    assert snapshot['reserved_local_jobs'] == 0
    assert snapshot['remaining_original_job_allowance'] == 4096
    assert snapshot['authenticated_accepted_summary_keys'] == 0


@pytest.mark.parametrize('settings', [{'checkpoint_name': '0001.json'}, {'changed': True}])
def test_missing_prefix_or_changed_population_is_rejected(tmp_path, settings):
    plan, scope = snapshot_fixture(tmp_path, **settings)
    with pytest.raises(ValueError, match='complete bound prefix'):
        recovery.checkpoint_snapshot(tmp_path, plan, scope)


def run_fixture(root, monkeypatch):
    original_plan, _ = publish_sealed_json(root/'original.json', {'evaluation_implementation': {}})
    state = {'original_plan': binding(original_plan), 'remaining_stages': [
        {'name': name} for name in ('parent-run', 'vectors', 'evaluation-prepare', 'evaluation-run')]}
    publish_sealed_json(root/'preflight.json', {
        'implementation_sha256': recovery.digest(recovery.__file__),
        'source_root': 'fixture', 'source_state': state})
    monkeypatch.setattr(recovery.original.evaluation, 'require_idle', lambda: None)
    monkeypatch.setattr(recovery.original.evaluation, 'implementation', lambda: {})
    monkeypatch.setattr(recovery, 'source_state', lambda _: state)
    return state


def test_stages_remain_serial_and_failed_parent_cannot_release_answers(tmp_path, monkeypatch):
    run_fixture(tmp_path, monkeypatch)
    calls = []
    def fail(root, plan, ordinal, stage):
        calls.append((ordinal, stage['name']))
        raise RuntimeError('parent failed')
    monkeypatch.setattr(recovery.original, 'launch_stage', fail)
    with pytest.raises(RuntimeError, match='parent failed'):
        recovery.run(tmp_path)
    assert calls == [(1, 'parent-run')]
    assert not (tmp_path/'complete.json').exists()
    with pytest.raises(FileExistsError):
        recovery.run(tmp_path)
    assert calls == [(1, 'parent-run')]


def test_successful_workflow_keeps_failed_target_failed(tmp_path, monkeypatch):
    run_fixture(tmp_path, monkeypatch)
    calls = []
    def finish(root, plan, ordinal, stage):
        calls.append((ordinal, stage['name']))
        return publish_sealed_json(root/f'{ordinal}.json', {'target_gate_passed': False})[0]
    monkeypatch.setattr(recovery.original, 'launch_stage', finish)
    recovery.run(tmp_path)
    assert calls == list(enumerate(
        ('parent-run', 'vectors', 'evaluation-prepare', 'evaluation-run'), 1))
    assert read_sealed_json(tmp_path/'complete.json').payload['target_gate_passed'] is False


def test_changed_source_state_is_rejected_before_any_release(tmp_path, monkeypatch):
    run_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(recovery, 'source_state', lambda _: {'changed': True})
    with pytest.raises(ValueError, match='source state changed'):
        recovery.run(tmp_path)
    assert not (tmp_path/'execution.reserved').exists()
