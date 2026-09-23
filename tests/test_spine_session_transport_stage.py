import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import spine_session_transport_stage as staging
from tools import spine_transport_lineage_v3 as lineage
from tools import execute_spine_transport_recovery_v3 as execution
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.contracts import canonical_json_bytes
from tools.stage_spine_transport_recovery import journal_state
from tests.test_spine_transport_recovery_v3 import fixture


def reseal(path, payload):
    content = canonical_json_bytes(payload)
    sha = hashlib.sha256(content).hexdigest()
    path.write_bytes(content)
    path.with_name(path.name + ".sha256").write_bytes(f"{sha}  {path.name}\n".encode("ascii"))
    return read_sealed_json(path)


def terminal_fixture(tmp_path, monkeypatch):
    original, root, successor, requests, readiness, attempts = fixture(tmp_path, monkeypatch)
    corpus = read_sealed_json(original / "preflight.json")
    raw_execution = read_sealed_json(original / "execution.json")
    scheduler = tmp_path / "scheduler"
    scheduler_plan, _ = publish_sealed_json(scheduler / "preflight.json", {"maximum_raw_calls": 3})
    failure, _ = publish_sealed_json(scheduler / "failure.json", {
        "preflight_sha256": scheduler_plan.sha256, "exception_type": "APITimeoutError",
        "automatic_retry_performed": False, "original_reservations_preserved": True})
    release, _ = publish_sealed_json(scheduler / "release.json", {"preflight_sha256": scheduler_plan.sha256})
    rows = [{"batch_index": i, "raw_request_sha256": r.sha256,
        "raw_request_path": f"offset-040/requests/{i:04d}.json",
        **journal_state(original / "offset-040/raw-checkpoints" / r.sha256)} for i, r in enumerate(requests)]
    counts = {"completed": 1, "reserved_without_response": 1, "unstarted": 1}
    observation, _ = publish_sealed_json(scheduler / "terminal.json", {
        "format": "memory-condense-terminal-ingest-session-observation-v1",
        "session_id": 123, "terminal_tool_chunk_id": "fixture-terminal", "ingest_exit_code": 1,
        "original_process_absent": True, "observed_exception": "openai.APITimeoutError caused by httpx.ReadTimeout",
        "reservations_cleared": False, "process_termination_performed": False,
        "scheduler_preflight_sha256": scheduler_plan.sha256, "scheduler_failure_sha256": failure.sha256,
        "scheduler_release_sha256": release.sha256, "corpus_preflight_sha256": corpus.sha256,
        "shard_offset": 40, "full_request_count": 3, "rows": rows, "counts": counts,
        "response_files": 1, "request_reservations": 2, "unacknowledged": [rows[1]]})
    inventory, _ = publish_sealed_json(root / "original-inventory.json", {
        "corpus_preflight_sha256": corpus.sha256, "execution_preflight_sha256": raw_execution.sha256,
        "terminal_observation_sha256": observation.sha256, "shard_offset": 40,
        "full_request_count": 3, "rows": rows, "counts": counts, "new_provider_calls": 0})
    source = root / "stage_from_terminal.py"
    source.write_text("fixture staging implementation\n", encoding="utf-8")
    plan = read_sealed_json(root / "stage-preflight.json")
    work = [{"raw_request_sha256": r["raw_request_sha256"], "raw_request_path": r["raw_request_path"],
        "batch_index": r["batch_index"], "prior_state": r["state"],
        "prior_reservation_journal_shas": r["reservation_journal_shas"], "maximum_additional_attempts": 1}
        for r in rows if r["state"] != "completed"]
    plan = reseal(plan.path, {**plan.payload, "format": "memory-condense-spine-transport-recovery-stage-v3",
        "terminal_identity_kind": "recorded exec session exit plus scheduler failure",
        "implementation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "automatic_retries": 0,
        "provider_execution_enabled": False, "full100_method_verification_required": True,
        "terminal_observation_path": str(observation.path), "terminal_observation_sha256": observation.sha256,
        "original_inventory_sha256": inventory.sha256, "raw_work": work, "retained_completed_raw_responses": 1})
    stage = read_sealed_json(root / "stage.json")
    reseal(stage.path, {**stage.payload, "stage_preflight_sha256": plan.sha256,
        "original_inventory_sha256": inventory.sha256})
    monkeypatch.setattr(staging, "prepare_corpus", lineage.prepare_corpus)
    monkeypatch.setattr(lineage, "validate", staging.validate)
    monkeypatch.setattr(staging.psutil, "process_iter", lambda *args: [])
    return original, root, successor, requests, observation, attempts


def test_terminal_session_recovery_runs_and_replays_with_exact_attempt_accounting(tmp_path, monkeypatch):
    original, root, successor, requests, _, attempts = terminal_fixture(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in original.rglob("*.json")}
    execution.run(root, tmp_path / "readiness.json", True)
    result = lineage.verify_transport_lineage(successor, 40, None)
    assert result["additional_raw_attempts"] == result["first_attempt_raw_requests"] == 1
    assert attempts == [r.sha256 for r in requests[1:]]
    assert all(p.read_bytes() == content for p, content in before.items())
    execution.run(root, enable=False)
    assert len(attempts) == 2


def test_live_original_command_prevents_recovery(tmp_path, monkeypatch):
    _, root, _, _, _, attempts = terminal_fixture(tmp_path, monkeypatch)
    process = SimpleNamespace(info={"name": "python.exe"}, cmdline=lambda: ["python", "run_remaining.py"])
    monkeypatch.setattr(staging.psutil, "process_iter", lambda *args: [process])
    with pytest.raises(ValueError, match="command is live"):
        execution.run(root, tmp_path / "readiness.json", True)
    assert not attempts and not (root / "execution-raw-v3.reserved").exists()


@pytest.mark.parametrize("change", ["missing_work", "extra_attempt", "changed_code", "changed_failure"])
def test_altered_terminal_recovery_cannot_release_requests(tmp_path, monkeypatch, change):
    _, root, _, _, observation, attempts = terminal_fixture(tmp_path, monkeypatch)
    if change == "changed_code":
        (root / "stage_from_terminal.py").write_text("changed staging method", encoding="utf-8")
    elif change == "changed_failure":
        failure = read_sealed_json(observation.path.parent / "failure.json")
        reseal(failure.path, {**failure.payload, "exception_type": "DifferentError"})
    else:
        plan = read_sealed_json(root / "stage-preflight.json")
        payload = plan.payload
        if change == "missing_work":
            payload["raw_work"].pop()
        else:
            payload["raw_work"][0]["maximum_additional_attempts"] = 2
        reseal(plan.path, payload)
    with pytest.raises(ValueError):
        execution.run(root, tmp_path / "readiness.json", True)
    assert not attempts and not (root / "execution-raw-v3.reserved").exists()


def test_timeout_label_without_terminal_exit_is_not_accepted(tmp_path, monkeypatch):
    _, _, _, _, observation, _ = terminal_fixture(tmp_path, monkeypatch)
    changed = SimpleNamespace(path=observation.path, payload={**observation.payload, "ingest_exit_code": None})
    with pytest.raises(ValueError, match="terminal exec timeout"):
        staging.require_terminal(changed)
