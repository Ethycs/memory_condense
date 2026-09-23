from types import SimpleNamespace

import pytest

from tools import run_spine_reader_after_timeout as schedule
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


def fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source, output = tmp_path / "source", tmp_path / "successor"
    old_plan, _ = publish_sealed_json(source / "runner-plan.json", {"preparation_sha256": "prepared"})
    terminal, _ = publish_sealed_json(source / "terminal.json", {"terminal": True})
    readiness, _ = publish_sealed_json(tmp_path / "readiness.json", {"ready": True})
    roots, left, calls = list(range(4)), [60] * 4, []
    def run(root, count):
        assert count == left[root] == 60
        calls.append(("run", root, count))
        left[root] = 0
    def judge(root, enable):
        assert left[root] == 0 and enable
        calls.append(("judge", root))
    def report(current_roots):
        assert current_roots == roots and left == [0] * 4
        publish_sealed_json(source / "development-report.json", {"completed": True})
    previous = SimpleNamespace(evaluation=SimpleNamespace(run=run, judge=judge), build_report=report)
    monkeypatch.setattr(schedule, "context", lambda _: (previous, old_plan, terminal, roots))
    monkeypatch.setattr(schedule, "remaining", lambda *_: list(left))
    monkeypatch.setattr(schedule, "require_idle", lambda: None)
    monkeypatch.setattr(schedule, "require_readiness", lambda _: readiness)
    schedule.prepare(source, output)
    return source, output, readiness, left, calls, previous


@pytest.mark.parametrize("condition", ("readiness", "partial", "concurrent"))
def test_unready_partial_or_concurrent_work_cannot_release_calls(tmp_path, monkeypatch, condition):
    _, output, _, left, calls, _ = fixture(tmp_path, monkeypatch)
    def fail(*_):
        raise ValueError("not ready")
    if condition == "readiness":
        monkeypatch.setattr(schedule, "require_readiness", fail)
    elif condition == "partial":
        left[0] = 59
    else:
        monkeypatch.setattr(schedule, "require_idle", fail)
    with pytest.raises(ValueError):
        schedule.run(output, tmp_path / "readiness.json", True)
    assert not calls and not (output / "execution.reserved").exists()


def test_exact_serial_population_with_no_second_release(tmp_path, monkeypatch):
    source, output, _, _, calls, _ = fixture(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in source.glob("*.json")}
    schedule.run(output, tmp_path / "readiness.json", True)
    assert calls == [item for root in range(4) for item in (("run", root, 60), ("judge", root))]
    assert all(p.read_bytes() == content for p, content in before.items())
    completion = read_sealed_json(output / "complete.json")
    assert completion.payload["full100_target_eligible"] is False
    with pytest.raises(ValueError, match="second release"):
        schedule.run(output, tmp_path / "readiness.json", True)
    assert len(calls) == 8


def test_existing_release_stops_duplicate_process(tmp_path, monkeypatch):
    _, output, _, _, calls, _ = fixture(tmp_path, monkeypatch)
    (output / "execution.reserved").write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        schedule.run(output, tmp_path / "readiness.json", True)
    assert not calls


def test_failed_answer_stops_judges_and_preserves_release(tmp_path, monkeypatch):
    _, output, _, left, calls, previous = fixture(tmp_path, monkeypatch)
    def fail(root, count):
        calls.append(("attempt", root))
        left[root] = 59
        raise TimeoutError("fixture failure")
    previous.evaluation.run = fail
    with pytest.raises(TimeoutError):
        schedule.run(output, tmp_path / "readiness.json", True)
    assert calls == [("attempt", 0)] and (output / "execution.reserved").exists()
    assert read_sealed_json(output / "failure.json").payload["automatic_retry_performed"] is False
    with pytest.raises(ValueError, match="second release"):
        schedule.run(output, tmp_path / "readiness.json", True)
    assert len(calls) == 1
