from pathlib import Path
import subprocess
from types import SimpleNamespace

import psutil
import pytest

from tools import schedule_spine_memory_completion_v3 as scheduler
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json, SealedArtifactError


def fixture(tmp_path, monkeypatch):
    root, raw, campaign = tmp_path / "jobs", tmp_path / "raw", tmp_path / "campaign"
    release = SimpleNamespace(sha256="release", payload={"executor_pid": 123,
        "executor_process_create_time": 456.0})
    monkeypatch.setattr(scheduler.worker, "RAW_CAMPAIGN", raw)
    monkeypatch.setattr(scheduler.worker, "CAMPAIGN", campaign)
    monkeypatch.setattr(scheduler, "raw_release", lambda: release)
    monkeypatch.setattr(scheduler, "dependency_alive", lambda _: True)
    monkeypatch.setattr(scheduler, "require_local_capacity", lambda _: None)
    monkeypatch.setattr(scheduler.worker, "plan_payload", lambda offset: {"offset": offset})
    monkeypatch.setattr(scheduler.worker, "prepare", lambda job, offset:
        publish_sealed_json(job / "preflight.json", {"offset": offset}))
    monkeypatch.setattr(scheduler.worker, "inputs", lambda offset: (None, None, {"offset": offset}, None, None))
    monkeypatch.setattr(scheduler.worker, "completed_raw", lambda offset, _: read_sealed_json(raw / f"completed-offset-{offset:03d}.json"))
    scheduler.prepare(root)
    return root, raw, campaign, release


def test_wait_never_releases_on_unfinished_payload(tmp_path, monkeypatch):
    _, raw, _, release = fixture(tmp_path, monkeypatch)
    marker, _ = publish_sealed_json(raw / "completed-offset-080.json", {"offset": 80})
    sidecar = marker.path.with_name(marker.path.name + ".sha256")
    original = sidecar.read_bytes()
    sidecar.unlink()
    waited = []
    def finish_publication(seconds):
        waited.append(seconds)
        sidecar.write_bytes(original)
    monkeypatch.setattr(scheduler.time, "sleep", finish_publication)
    assert scheduler.wait_for_raw(80, {}, release, float("inf")).sha256 == marker.sha256
    assert waited == [10]


@pytest.mark.parametrize("condition,exception", [("failed", ValueError), ("dead", ValueError), ("expired", TimeoutError)])
def test_wait_stops_for_failure_terminal_process_or_deadline(tmp_path, monkeypatch, condition, exception):
    _, raw, _, release = fixture(tmp_path, monkeypatch)
    if condition == "failed":
        publish_sealed_json(raw / "failure.json", {"exception_type": "APITimeoutError"})
    if condition == "dead":
        monkeypatch.setattr(scheduler, "dependency_alive", lambda _: False)
    with pytest.raises(exception):
        scheduler.wait_for_raw(80, {}, release, -1 if condition == "expired" else float("inf"))


def test_corrupt_completed_namespace_is_not_treated_as_pending(tmp_path, monkeypatch):
    _, raw, _, release = fixture(tmp_path, monkeypatch)
    artifact, _ = publish_sealed_json(raw / "completed-offset-080.json", {"offset": 80})
    artifact.path.write_bytes(b"corrupt")
    with pytest.raises(SealedArtifactError):
        scheduler.wait_for_raw(80, {}, release, float("inf"))


def test_process_identity_does_not_accept_reused_pid(monkeypatch):
    release = SimpleNamespace(payload={"executor_pid": 123, "executor_process_create_time": 456.0})
    monkeypatch.setattr(scheduler.psutil, "Process", lambda _: SimpleNamespace(is_running=lambda: True,
        create_time=lambda: 457.0))
    assert scheduler.dependency_alive(release) is False
    def missing(_):
        raise psutil.NoSuchProcess(123)
    monkeypatch.setattr(scheduler.psutil, "Process", missing)
    assert scheduler.dependency_alive(release) is False


@pytest.mark.parametrize("fail_offset", [None, 90])
def test_serial_workers_prepare_full100_only_after_every_completion(tmp_path, monkeypatch, fail_offset):
    root, raw, campaign, _ = fixture(tmp_path, monkeypatch)
    for offset in scheduler.OFFSETS:
        publish_sealed_json(raw / f"completed-offset-{offset:03d}.json", {"offset": offset})
    calls = []
    def command(module, *args):
        calls.append((module, args))
        if module == "tools.finish_spine_memory_namespace_v3":
            job = Path(args[args.index("--output-root") + 1])
            offset = args[args.index("--shard-offset") + 1]
            if offset == fail_offset:
                raise subprocess.CalledProcessError(1, [module])
            plan = read_sealed_json(job / "preflight.json")
            raw_completion = read_sealed_json(raw / f"completed-offset-{offset:03d}.json")
            publish_sealed_json(job / "complete.json", {"offset": offset, "preflight_sha256": plan.sha256,
                "raw_completion_sha256": raw_completion.sha256, "answer_calls_sent": 0, "judge_calls_sent": 0})
        else:
            assert module == "tools.run_spine_semantic_seed_full100_v3"
            assert args[0] == "prepare" and "--enable-provider" not in args
            assert all((root / f"completed-offset-{offset:03d}.json").is_file() for offset in scheduler.OFFSETS)
            publish_sealed_json(campaign / "runner-plan.json", {"maximum_answer_calls": 500,
                "maximum_logical_judgments": 200})
    monkeypatch.setattr(scheduler.worker, "command", command)
    if fail_offset is None:
        scheduler.run(root, True)
        result = read_sealed_json(root / "complete.json")
        assert len(result.payload["completed_namespaces"]) == 2
        assert result.payload["answer_calls_sent"] == result.payload["judge_calls_sent"] == 0
        before = list(calls)
        with pytest.raises(FileExistsError):
            scheduler.run(root, True)
        assert calls == before
    else:
        with pytest.raises(subprocess.CalledProcessError):
            scheduler.run(root, True)
        failure = read_sealed_json(root / "failure.json")
        assert [r["offset"] for r in failure.payload["completed_namespaces"]] == [80]
        assert failure.payload["offset"] == 90 and failure.payload["automatic_retry_performed"] is False
        assert (root / "execution.reserved").is_file()
        assert not (campaign / "runner-plan.json").exists()
        assert len(calls) == 2


