from pathlib import Path

import pytest

from memory_condense.eval.fast_completion_runtime import _canonical_bytes, _read_journal, _sealed
from tools import execute_spine_transport_recovery as old_execution
from tools import execute_spine_transport_recovery_v2 as execution
from tools import spine_transport_lineage_v2 as lineage
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.stage_spine_transport_recovery import copy_completed_checkpoint, journal_state
from tests.test_spine_transport_lineage import record
from tests import test_spine_transport_lineage as legacy_fixtures


def fixture(tmp_path, monkeypatch):
    original, root = tmp_path / "original", tmp_path / "stage"
    successor = root / "corpus"
    bindings, requests, copied, work = [], [], [], []
    for i in range(3):
        relative = f"offset-040/requests/{i:04d}.json"
        request, _ = publish_sealed_json(original / relative, {
            "model": "codex_sdk/gpt-5.6-terra", "batch_index": i,
            "messages": [{"role": "user", "content": f"Raw fragment {i}."}]})
        publish_sealed_json(successor / relative, request.payload)
        requests.append(request)
        bindings.append({"path": relative, "sha256": request.sha256})
        old = original / "offset-040/raw-checkpoints" / request.sha256
        new = successor / "offset-040/raw-checkpoints" / request.sha256
        if i < 2:
            record(old, request.payload["messages"], request.payload["model"],
                   {"raw_request_sha256": request.sha256}, fail=i == 1)
        prior = journal_state(old)
        if i == 0:
            copied.append({"offset": 40, "raw_request_sha256": request.sha256,
                "files": copy_completed_checkpoint(old, new, request), **prior})
        else:
            work.append({"raw_request_sha256": request.sha256, "prior_state": prior["state"],
                "prior_reservation_journal_shas": prior["reservation_journal_shas"],
                "maximum_additional_attempts": 1})
    namespace = {"shard_offset": 40, "request_count": 3, "requests": bindings}
    corpus, _ = publish_sealed_json(original / "preflight.json", {"namespaces": [namespace]})
    publish_sealed_json(successor / "preflight.json", corpus.payload)
    raw_execution, _ = publish_sealed_json(original / "execution.json", {"request_shas": [r.sha256 for r in requests]})
    plan, _ = publish_sealed_json(root / "stage-preflight.json", {
        "original_corpus_root": str(original), "shard_offset": 40, "full_request_count": 3,
        "original_inventory_sha256": "inventory", "corpus_preflight_sha256": corpus.sha256,
        "execution_preflight_sha256": raw_execution.sha256, "raw_work": work,
        "maximum_new_provider_calls": 2, "maximum_reissued_raw_requests": 1, "first_attempt_raw_requests": 1})
    publish_sealed_json(root / "stage.json", {
        "format": "memory-condense-spine-transport-recovery-stage-result-v2",
        "stage_preflight_sha256": plan.sha256, "original_inventory_sha256": "inventory",
        "staged_corpus_root": str(successor), "shard_offset": 40,
        "copied_completed_requests": copied, "copied_completed_request_count": 1})
    monkeypatch.setattr(lineage, "validate", lambda _: plan)
    def prepare(directory, offset, limit):
        assert offset == 40 and limit == 3
        current = [read_sealed_json(directory / b["path"]) for b in bindings]
        if [r.sha256 for r in current] != [b["sha256"] for b in bindings]:
            raise ValueError("transport changed a raw request")
        return corpus, namespace, current, raw_execution
    monkeypatch.setattr(lineage, "prepare_corpus", prepare)
    readiness, _ = publish_sealed_json(tmp_path / "readiness.json", {"fixture": True})
    monkeypatch.setattr(execution, "require_readiness", lambda *args, **kwargs: readiness)
    monkeypatch.setattr(old_execution, "require_readiness", lambda *args, **kwargs: readiness)
    attempts = []
    def execute(directory, offset, count, enable):
        assert directory == successor and offset == 40 and count == 3
        for request in requests:
            checkpoint = directory / "offset-040/raw-checkpoints" / request.sha256
            if journal_state(checkpoint)["state"] == "completed":
                continue
            assert enable
            attempts.append(request.sha256)
            record(checkpoint, request.payload["messages"], request.payload["model"],
                   {"raw_request_sha256": request.sha256})
    monkeypatch.setattr(execution, "execute_corpus", execute)
    return original, root, successor, requests, readiness, attempts


def test_one_declared_release_preserves_success_counts_unknown_attempt_and_replays(tmp_path, monkeypatch):
    original, root, successor, requests, _, attempts = fixture(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in (original / "offset-040/raw-checkpoints").glob("*/*.json")}
    execution.run(root, tmp_path / "readiness.json", True)
    result = lineage.verify_transport_lineage(successor, 40, None)
    assert attempts == [r.sha256 for r in requests[1:]]
    assert result["additional_raw_attempts"] == result["first_attempt_raw_requests"] == 1
    assert result["preserved_completed_raw_responses"] == 1
    assert result["additional_compaction_attempts"] == result["new_provider_calls"] == 0
    assert all(p.read_bytes() == content for p, content in before.items())
    completed = read_sealed_json(root / "execution-complete.json")
    execution.run(root, enable=False)
    assert len(attempts) == 2
    assert read_sealed_json(root / "execution-complete.json").sha256 == completed.sha256


def test_readiness_failure_creates_no_release_or_new_request(tmp_path, monkeypatch):
    _, root, successor, requests, _, attempts = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(execution, "require_readiness", lambda *args: (_ for _ in ()).throw(ValueError("not ready")))
    with pytest.raises(ValueError, match="not ready"):
        execution.run(root, tmp_path / "readiness.json", True)
    assert not attempts and not (root / "execution-raw-v2.reserved").exists()
    assert all(journal_state(successor / "offset-040/raw-checkpoints" / r.sha256)["state"] == "unstarted"
               for r in requests[1:])


def test_uncertain_successor_is_not_retried_or_certified(tmp_path, monkeypatch):
    _, root, successor, requests, _, attempts = fixture(tmp_path, monkeypatch)
    request = requests[1]
    record(successor / "offset-040/raw-checkpoints" / request.sha256,
           request.payload["messages"], request.payload["model"], {"raw_request_sha256": request.sha256}, fail=True)
    with pytest.raises(RuntimeError, match="unacknowledged"):
        execution.run(root, tmp_path / "readiness.json", True)
    with pytest.raises(RuntimeError, match="unacknowledged"):
        lineage.verify_transport_lineage(successor, 40, None)
    assert not attempts


def test_executor_failure_keeps_reservation_and_records_failure_without_second_release(tmp_path, monkeypatch):
    _, root, successor, requests, _, attempts = fixture(tmp_path, monkeypatch)
    def fail(*args):
        request = requests[1]
        attempts.append(request.sha256)
        record(successor / "offset-040/raw-checkpoints" / request.sha256,
               request.payload["messages"], request.payload["model"], {"raw_request_sha256": request.sha256}, fail=True)
        raise TimeoutError("fixture transport failure")
    monkeypatch.setattr(execution, "execute_corpus", fail)
    with pytest.raises(TimeoutError):
        execution.run(root, tmp_path / "readiness.json", True)
    failure = read_sealed_json(root / "execution-failure.json").payload
    assert failure["exception_type"] == "TimeoutError"
    assert failure["automatic_retry_performed"] is False
    assert (root / "execution-raw-v2.reserved").exists()
    with pytest.raises(RuntimeError, match="unacknowledged"):
        execution.run(root, tmp_path / "readiness.json", True)
    assert len(attempts) == 1


def test_existing_release_cannot_launch_a_duplicate_executor(tmp_path, monkeypatch):
    _, root, _, _, _, attempts = fixture(tmp_path, monkeypatch)
    preflight, _ = execution.prepare(root)
    (root / "execution-raw-v2.reserved").write_text(preflight.sha256 + "\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        execution.run(root, tmp_path / "readiness.json", True)
    assert not attempts


def test_resealed_replacement_and_foreign_request_cannot_enter_recovery(tmp_path, monkeypatch):
    _, _, successor, requests, _, _ = fixture(tmp_path, monkeypatch)
    path = next((successor / "offset-040/raw-checkpoints" / requests[0].sha256).glob("*.response.json"))
    body, _ = _read_journal(path)
    body.pop("journal_sha256")
    body["completion"] = "Replacement."
    path.write_bytes(_canonical_bytes(_sealed(body)))
    with pytest.raises(ValueError, match="dropped or replaced"):
        lineage.stage_state(successor, 40)
    with pytest.raises(ValueError, match="different corpus or namespace"):
        lineage.stage_state(successor, 50)


def test_completed_responses_without_release_or_complete_population_are_rejected(tmp_path, monkeypatch):
    _, root, successor, requests, _, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="complete successor"):
        lineage.verify_transport_lineage(successor, 40, None)
    for request in requests[1:]:
        record(successor / "offset-040/raw-checkpoints" / request.sha256,
               request.payload["messages"], request.payload["model"], {"raw_request_sha256": request.sha256})
    with pytest.raises(ValueError, match="no authorized release"):
        execution.run(root, enable=False)


@pytest.mark.parametrize("offset", (10, 20))
def test_existing_transport_receipts_remain_identical(tmp_path, monkeypatch, offset):
    _, successor, repair, _ = legacy_fixtures.fixture(tmp_path, monkeypatch, offset=offset)
    repair = repair if offset == 10 else None
    assert lineage.verify_transport_lineage(successor, offset, repair) == lineage.previous.verify_transport_lineage(successor, offset, repair)
