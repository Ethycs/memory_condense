from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from tools import run_frozen_native_spine_stages as controller
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


def test_same_pid_with_different_creation_is_not_the_dependency(monkeypatch):
    monkeypatch.setattr(controller.psutil, 'Process', lambda _: SimpleNamespace(create_time=lambda: 2.0))
    assert not controller.same_process({'pid': 10, 'create_time': 1.0})
    assert controller.same_process({'pid': 10, 'create_time': 2.0})


def test_stage_order_keeps_models_separate_and_uses_frozen_evaluator(tmp_path):
    stages = controller.stages(tmp_path/'corpus', tmp_path/'answers', tmp_path/'candidate.json')
    assert [s['name'] for s in stages] == [
        'parent-prepare', 'parent-run', 'vectors', 'evaluation-prepare', 'evaluation-run']
    assert stages[1]['command'][-2:] == ['--budget', '4096']
    assert 'tools.evaluate_frozen_native_spine_full100' in stages[-1]['command']
    assert stages[-1]['command'][-1] == '--enable-provider'


@pytest.mark.parametrize('name,payload', [
    ('parent-run', {'complete_native_hierarchies': False}),
    ('vectors', {'complete_prepared_vectors': False}),
    ('evaluation-run', {'accuracy': {'parent_context': {'questions': 8}}}),
])
def test_incomplete_stage_cannot_advance(name, payload):
    with pytest.raises(ValueError):
        controller.validate_stage_result({'name': name}, SimpleNamespace(payload=payload))


def test_failed_child_is_saved_once_and_does_not_retry(tmp_path):
    plan = SimpleNamespace(sha256='fixture')
    counter = tmp_path/'count.txt'
    command = [sys.executable, '-c',
        'from pathlib import Path; import sys; p=Path(sys.argv[1]); p.write_text("one attempt"); sys.exit(7)',
        str(counter)]
    stage = {'name': 'fixture', 'command': command, 'result': str(tmp_path/'missing.json')}
    with pytest.raises(RuntimeError, match='exited 7'):
        controller.launch_stage(tmp_path, plan, 0, stage)
    ended = read_sealed_json(tmp_path/'00-fixture.exit.json')
    assert ended.payload['exit_code'] == 7 and ended.payload['retry_performed'] is False
    assert counter.read_text() == 'one attempt'
    with pytest.raises(FileExistsError):
        controller.launch_stage(tmp_path, plan, 0, stage)
    assert not (tmp_path/'00-fixture.complete.json').exists()
