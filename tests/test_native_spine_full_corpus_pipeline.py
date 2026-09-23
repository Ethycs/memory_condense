import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_native_spine_full_corpus as pipeline
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json, SealedArtifactError


def dependency(root, *, finished=None):
    policy, _ = publish_sealed_json(root / "policy.json", {"fixture": True})
    started, _ = publish_sealed_json(root / "started.json", {
        "policy_sha256": policy.sha256, "pid": 42, "create_time": 50.0})
    item = {"name": "fixture", "control": str(root), "policy": pipeline.source_completion.binding(policy.path),
            "started": pipeline.source_completion.binding(started.path), "required": {"complete": True}}
    if finished is not None:
        publish_sealed_json(root / "finished.json", {"policy_sha256": policy.sha256, "complete": finished})
    return item


def test_live_predecessor_cannot_be_released_by_stale_terminal_file(tmp_path, monkeypatch):
    item = dependency(tmp_path, finished=True)
    monkeypatch.setattr(pipeline.psutil, "Process", lambda pid: SimpleNamespace(pid=pid, create_time=lambda: 50.0))
    assert pipeline.dependency_state([item]) == ([{"name": "fixture", "pid": 42}], {})
    monkeypatch.setattr(pipeline.psutil, "Process", lambda pid: SimpleNamespace(pid=pid, create_time=lambda: 51.0))
    live, finished = pipeline.dependency_state([item])
    assert not live and set(finished) == {"fixture"}


@pytest.mark.parametrize("finished", [None, False, 1])
def test_exited_predecessor_still_needs_authentic_complete_receipt(tmp_path, monkeypatch, finished):
    item = dependency(tmp_path, finished=finished)
    monkeypatch.setattr(pipeline.psutil, "Process", lambda pid: SimpleNamespace(pid=pid, create_time=lambda: 51.0))
    with pytest.raises((ValueError, SealedArtifactError)):
        pipeline.dependency_state([item])


class Backend:
    def __init__(self):
        self.model = None
        self.calls = []

    def generate(self, jobs, attempt):
        self.calls.append((jobs, attempt))
        self.model = "loaded"


def merge_result(root, name, complete, full=True):
    return publish_sealed_json(root / name, {"done": complete, "complete_source_compilation": full,
                                           "body_count": 2 if complete else 1})[0]


def test_merge_continuation_uses_existing_128_job_budget_and_keeps_final_result(tmp_path):
    backend = Backend()
    zero, final = merge_result(tmp_path, "zero.json", False), merge_result(tmp_path, "final.json", True)
    budgets = []
    def execute(budget):
        budgets.append(budget)
        if budget:
            backend.generate(["a", "b", "c", "d"], 0)
            return final
        return zero
    result = pipeline.complete_merges(tmp_path, backend, execute, limit=256, complete_key="done")
    assert result.sha256 == final.sha256 and budgets == [0, 128]
    assert backend.calls == [(("a", "b", "c", "d"), 0)]
    progress = read_sealed_json(tmp_path / "pipeline-progress" / "0001.json")
    assert progress.payload["new_local_jobs"] == 4 and progress.payload["new_local_batches"] == 1


@pytest.mark.parametrize("problem", ["allowance", "stalled", "partial", "loaded_during_replay"])
def test_merge_stage_cannot_pass_with_incomplete_or_unbounded_execution(tmp_path, problem):
    backend = Backend()
    zero = merge_result(tmp_path, "zero.json", problem == "partial", full=problem != "partial")
    def execute(budget):
        if problem == "loaded_during_replay":
            backend.model = "unexpected"
        if budget and problem == "allowance":
            backend.generate(range(4), 0)
        return zero
    with pytest.raises(ValueError):
        pipeline.complete_merges(tmp_path, backend, execute, limit=3, complete_key="done")
    assert backend.calls == []


def test_partial_actual_store_is_rejected_before_any_compilation_stage(tmp_path, monkeypatch):
    sources, _ = publish_sealed_json(tmp_path / "sources.json", {"body_count": 2})
    manifest = merge_result(tmp_path, "manifest.json", False, full=False)
    done, _ = publish_sealed_json(tmp_path / "finished.json", {"summary_store_sha256": manifest.sha256})
    manifest.payload.update(sources_sha256=sources.sha256, all_prepared_bodies_admitted=False)
    reader = SimpleNamespace(manifest=manifest, close=lambda: None)
    monkeypatch.setattr(pipeline.source_completion.admission, "JsonRecoveredSummaryBodies", lambda root: reader)
    policy = SimpleNamespace(payload={"paths": {"store": str(tmp_path)}, "sources": pipeline.source_completion.binding(sources.path)})
    release = SimpleNamespace(payload={"predecessor_completions": {"source_completion": pipeline.source_completion.binding(done.path)}})
    with pytest.raises(ValueError, match="actual complete source corpus"):
        pipeline.full_store(policy, release)


@pytest.mark.parametrize("fail_stage", ["waiting", "parents", None])
def test_pipeline_stops_on_dependency_or_child_failure_and_preserves_failed_joint_gate(tmp_path, monkeypatch, fail_stage):
    root = tmp_path / "pipeline"
    policy, _ = publish_sealed_json(root / "policy.json", {"fixture": True})
    monkeypatch.setattr(pipeline, "load", lambda root: policy)
    def wait(policy):
        if fail_stage == "waiting":
            raise TimeoutError("still live")
        return {}
    monkeypatch.setattr(pipeline, "wait_dependencies", wait)
    monkeypatch.setattr(pipeline, "full_store", lambda *a: None)
    called = []
    def child(root, policy, release, stage, enable):
        assert enable
        called.append(stage)
        if stage == fail_stage:
            raise RuntimeError("child failed")
        if stage == "evaluation":
            publish_sealed_json(root / "evaluation" / "joint-report.json", {"accuracy": {"hierarchy": .84}, "target_gate_passed": False})
    monkeypatch.setattr(pipeline, "run_child", child)
    if fail_stage:
        with pytest.raises((TimeoutError, RuntimeError)):
            pipeline.run(root, enable_provider=True)
        assert not (root / "finished.json").exists()
        failure = read_sealed_json(root / "failure.json")
        assert failure.payload["stage"] == fail_stage and not failure.payload["automatic_retry"]
        assert called == ([] if fail_stage == "waiting" else list(pipeline.STAGES[:5]))
    else:
        result = pipeline.run(root, enable_provider=True)
        assert called == list(pipeline.STAGES)
        assert result.payload["target_gate_passed"] is False and result.payload["joint_report_replayed"] is True
    with pytest.raises(FileExistsError):
        pipeline.run(root, enable_provider=True)


def test_child_runs_without_shell_and_requires_successful_bound_result(tmp_path, monkeypatch):
    root = tmp_path / "pipeline"
    (root / "stages").mkdir(parents=True)
    policy, _ = publish_sealed_json(root / "policy.json", {"fixture": True})
    release, _ = publish_sealed_json(root / "released.json", {"policy_sha256": policy.sha256})
    def popen(command, **kwargs):
        assert command[:5] == [pipeline.sys.executable, "-X", "utf8", "-m", "tools.run_native_spine_full_corpus"]
        assert command[-1] == "--enable-provider" and "shell" not in kwargs
        assert kwargs["creationflags"] == getattr(pipeline.subprocess, "CREATE_NO_WINDOW", 0)
        report, _ = publish_sealed_json(root / "report.json", {"target_gate_passed": False})
        publish_sealed_json(root / "stages" / "evaluation.result.json", {
            "policy_sha256": policy.sha256, "stage": "evaluation", "artifact": pipeline.source_completion.binding(report.path)})
        return SimpleNamespace(pid=os.getpid(), poll=lambda: 0, returncode=0)
    monkeypatch.setattr(pipeline.subprocess, "Popen", popen)
    result = pipeline.run_child(root, policy, release, "evaluation", True)
    assert result.payload["stage"] == "evaluation"
    with pytest.raises(FileExistsError):
        pipeline.run_child(root, policy, release, "evaluation", True)
