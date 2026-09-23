from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from tools import run_spine_as_of_after_compilation as handoff
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


@pytest.mark.parametrize("state", ["live", "reused", "missing", "access_denied"])
def test_dependency_identity_is_checked_before_timing(monkeypatch, state):
    def process(pid):
        assert pid == 321
        if state == "missing":
            raise psutil.NoSuchProcess(pid)
        if state == "access_denied":
            raise psutil.AccessDenied(pid)
        return SimpleNamespace(is_running=lambda: True, create_time=lambda: 123.0 if state == "live" else 456.0)
    monkeypatch.setattr(handoff.psutil, "Process", process)
    row = {"executor_pid": 321, "executor_process_create_time": 123.0}
    if state == "live":
        with pytest.raises(ValueError, match="still running"):
            handoff.require_terminal(row)
    elif state == "access_denied":
        with pytest.raises(psutil.AccessDenied):
            handoff.require_terminal(row)
    else:
        handoff.require_terminal(row)


def completion_fixture(tmp_path, monkeypatch, *, fault=None):
    bulk_root, compilation = tmp_path / "bulk", tmp_path / "compilation"
    monkeypatch.setattr(handoff, "COMPILATION", compilation)
    monkeypatch.setattr(handoff.runner, "BULK", bulk_root)
    monkeypatch.setattr(handoff, "require_terminal", lambda row: None)
    rows = [{"root": str(root), "preflight_sha256": name}
            for root, name in ((bulk_root, "raw-plan"), (compilation, "compile-plan"))]
    bulk, _ = publish_sealed_json(bulk_root / "complete.json", {"preflight_sha256": "raw-plan"})
    monkeypatch.setattr(handoff.runner, "require_bulk_complete", lambda: bulk)
    workers, finished = [], []
    for offset in (80, 90):
        root = compilation / f"offset-{offset}"
        row = {"root": str(root), "offset": offset, "preflight_sha256": f"plan-{offset}"}
        raw, _ = publish_sealed_json(bulk_root / f"completed-offset-{offset:03d}.json", {"offset": offset})
        result, _ = publish_sealed_json(root / "complete.json", {
            "preflight_sha256": row["preflight_sha256"], "offset": offset,
            "raw_completion_sha256": "wrong" if fault == "raw_binding" and offset == 90 else raw.sha256,
            "answer_calls_sent": 0, "judge_calls_sent": 0})
        workers.append(row)
        finished.append({**row, "completion_sha256": result.sha256})
    publish_sealed_json(compilation / "preflight.json", {"workers": workers})
    complete, _ = publish_sealed_json(compilation / "complete.json", {
        "preflight_sha256": "wrong" if fault == "plan_binding" else "compile-plan",
        "answer_calls_sent": 1 if fault == "started_answers" else 0,
        "judge_calls_sent": 0, "timed_evaluation_requires_separate_idle_release": True,
        "completed_namespaces": finished[:1] if fault == "missing_memory" else finished})
    if fault == "failed_dependency":
        publish_sealed_json(compilation / "failure.json", {"exception_type": "TimeoutError"})
    if fault == "unsealed":
        (compilation / "complete.json.sha256").unlink()
    return rows, bulk, complete


def test_both_completed_memories_bind_to_their_whole_raw_inputs(tmp_path, monkeypatch):
    rows, bulk, complete = completion_fixture(tmp_path, monkeypatch)
    assert [r.sha256 for r in handoff.require_dependencies_complete(rows)] == [bulk.sha256, complete.sha256]


@pytest.mark.parametrize("fault", ["raw_binding", "plan_binding", "started_answers", "missing_memory",
                                    "failed_dependency", "unsealed"])
def test_failed_or_incomplete_dependencies_cannot_release(tmp_path, monkeypatch, fault):
    rows, _, _ = completion_fixture(tmp_path, monkeypatch, fault=fault)
    with pytest.raises(ValueError):
        handoff.require_dependencies_complete(rows)


def run_fixture(tmp_path, monkeypatch, *, failed_preparation=None, readiness_failure=False):
    monkeypatch.chdir(tmp_path)
    root, campaign = tmp_path / "handoff", tmp_path / "campaign"
    monkeypatch.setattr(handoff, "CAMPAIGN", campaign)
    p = {"dependencies": [{"root": "raw"}, {"root": "compile"}]}
    publish_sealed_json(root / "preflight.json", p)
    monkeypatch.setattr(handoff, "payload", lambda root: p)
    monkeypatch.setattr(handoff, "require_unprepared_tail", lambda: None)
    events = []
    def dependencies(rows):
        events.append("dependencies")
        return [SimpleNamespace(sha256="raw-complete"), SimpleNamespace(sha256="compile-complete")]
    monkeypatch.setattr(handoff, "require_dependencies_complete", dependencies)
    monkeypatch.setattr(handoff.runner, "require_idle", lambda: events.append("idle"))
    def prepare(offset):
        events.append(("prepare", offset))
        if offset == failed_preparation:
            raise ValueError("fixture incomplete namespace")
    monkeypatch.setattr(handoff, "prepare_namespace", prepare)
    monkeypatch.setattr(handoff.runner, "prepare", lambda root: events.append("verify500"))
    def probe(root):
        events.append("readiness")
        if readiness_failure:
            raise TimeoutError("fixture unavailable gateway")
    monkeypatch.setattr(handoff.readiness, "run", probe)
    def evaluate(target, report, enable):
        assert target == campaign and report == root / "readiness/report.json" and enable
        events.append("full100")
        publish_sealed_json(campaign / "complete.json", {"target_gate_passed": False})
    monkeypatch.setattr(handoff.runner, "run", evaluate)
    return root, events


def test_final_preparation_then_readiness_then_frozen_full100(tmp_path, monkeypatch):
    root, events = run_fixture(tmp_path, monkeypatch)
    handoff.run(root, True)
    assert events == ["dependencies", "idle", ("prepare", 80), ("prepare", 90),
                      "verify500", "idle", "readiness", "full100"]
    assert read_sealed_json(root / "complete.json").payload["target_gate_passed"] is False


def test_partial_preparation_is_preserved_and_never_retried(tmp_path, monkeypatch):
    root, events = run_fixture(tmp_path, monkeypatch, failed_preparation=90)
    with pytest.raises(ValueError, match="incomplete namespace"):
        handoff.run(root, True)
    assert events == ["dependencies", "idle", ("prepare", 80), ("prepare", 90)]
    assert (root / "execution.reserved").exists()
    failure = read_sealed_json(root / "failure.json")
    assert failure.payload["phase"] == "final namespace preparation"
    assert failure.payload["automatic_retry_performed"] is False
    with pytest.raises(FileExistsError):
        handoff.run(root, True)
    assert events.count(("prepare", 80)) == 1
    assert events.count(("prepare", 90)) == 1


def test_readiness_failure_sends_no_benchmark_answers(tmp_path, monkeypatch):
    root, events = run_fixture(tmp_path, monkeypatch, readiness_failure=True)
    with pytest.raises(TimeoutError):
        handoff.run(root, True)
    assert "full100" not in events
    assert read_sealed_json(root / "failure.json").payload["phase"] == "fresh bounded readiness"


def test_live_dependency_blocks_before_reservation_or_model_work(tmp_path, monkeypatch):
    root, events = run_fixture(tmp_path, monkeypatch)
    def busy(rows):
        raise ValueError("dependency still running")
    monkeypatch.setattr(handoff, "require_dependencies_complete", busy)
    with pytest.raises(ValueError, match="still running"):
        handoff.run(root, True)
    assert not events and not (root / "execution.reserved").exists()


def test_provider_flag_is_required(tmp_path, monkeypatch):
    root, events = run_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="provider flag"):
        handoff.run(root)
    assert not events and not (root / "execution.reserved").exists()


@pytest.mark.parametrize("artifact", ["prepared/offset-080.json", "namespaces/offset-090",
                                       "runner-plan.json", "execution.reserved"])
def test_handoff_rejects_existing_or_partial_tail_preparation(tmp_path, monkeypatch, artifact):
    monkeypatch.setattr(handoff, "CAMPAIGN", tmp_path)
    path = tmp_path / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    with pytest.raises(ValueError, match="already"):
        handoff.require_unprepared_tail()
