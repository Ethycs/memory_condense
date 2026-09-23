from types import SimpleNamespace
import os

import psutil
import pytest

from tests import test_native_spine_repair as fixture
from tests.test_native_json_batch_repair import malformed_source
from tests.test_native_spine_direct_repairs import SectionClient
from tests.test_native_spine_admission import corpus
from tests.test_native_spine_repair import Client
from tools import complete_native_spine_ingestion as completion
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json, SealedArtifactError


@pytest.mark.parametrize("malformed", [False, True])
def test_real_original_classification_and_repair_preserve_every_original_file(tmp_path, monkeypatch, malformed):
    source = tmp_path / "source"
    (malformed_source if malformed else fixture.original_checkpoint)(source, monkeypatch)
    parent = read_sealed_json(source / "preflight.json")
    original = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    invalid, seen, counts = completion.validation_state(source, parent)
    assert seen == {0} and counts == {"invalid_summary": 1}
    groups = completion.classify(source, parent, invalid)
    kind = "json_repair_roots" if malformed else "direct_roots"
    assert groups[kind] == [0] and sum(map(len, groups.values())) == 1
    client = SectionClient()
    module = completion.malformed if malformed else completion.sections
    monkeypatch.setattr(module, "_completion_client", lambda *a: client)
    target, result, calls = completion.repair_group(source, tmp_path / "cohort", kind, [0],
        policy_sha="fixture", remaining_calls=4, max_stages=3, enable_provider=True)
    assert result.payload["complete_repair_snapshot"] and calls == client.calls == 1
    assert target.name == "stage-00"
    assert all(path.read_bytes() == raw for path, raw in original.items())
    completion.require_complete_requests(parent, seen, invalid, {0}, set())


def test_coordinator_refines_failed_sections_without_regenerating_accepted_sections(tmp_path, monkeypatch):
    source, root = tmp_path / "source", tmp_path / "cohort"
    malformed_source(source, monkeypatch)
    first, second = SectionClient(invalid_first=True), SectionClient()
    clients = iter([first, second])
    monkeypatch.setattr(completion.malformed, "_completion_client", lambda *a: next(clients))
    target, result, calls = completion.repair_group(source, root, "json_repair_roots", [0],
        policy_sha="fixture", remaining_calls=4, max_stages=3, enable_provider=True)
    assert result.payload["complete_repair_snapshot"] and calls == 2 and target.name == "stage-01"
    before = read_sealed_json(root / "stage-01" / "source-snapshot.json")
    assert len(before.payload["ready"]) == 1
    admitted = read_sealed_json(target / "admitted-batches" / "000000.json")
    assert before.payload["ready"][0]["atom"] in admitted.payload["summaries"]


@pytest.mark.parametrize("failure", ["allowance", "transport", "stages"])
def test_failures_never_release_a_second_unplanned_call(tmp_path, monkeypatch, failure):
    source, root = tmp_path / "source", tmp_path / "cohort"
    malformed_source(source, monkeypatch)
    client = SectionClient(fail=failure == "transport", invalid_first=failure == "stages")
    monkeypatch.setattr(completion.malformed, "_completion_client", lambda *a: client)
    with pytest.raises((ValueError, ConnectionError)):
        completion.repair_group(source, root, "json_repair_roots", [0], policy_sha="fixture",
            remaining_calls=0 if failure == "allowance" else 4, max_stages=1, enable_provider=True)
    assert client.calls == (0 if failure == "allowance" else 1)
    with pytest.raises(ValueError, match="already reserved"):
        completion.repair_group(source, root, "json_repair_roots", [0], policy_sha="fixture",
            remaining_calls=4, max_stages=3, enable_provider=True)
    assert client.calls == (0 if failure == "allowance" else 1)


def test_duplicate_and_inflight_validations_cannot_be_counted_as_complete(tmp_path, monkeypatch):
    source = tmp_path / "source"
    original = fixture.original_checkpoint(source, monkeypatch)
    parent = read_sealed_json(source / "preflight.json")
    other = source / "validated" / "alias.json"
    other.write_bytes(original.path.read_bytes())
    invalid, seen, _ = completion.validation_state(source, parent, allow_inflight=True)
    assert seen == {0} and len(invalid) == 1
    with pytest.raises(SealedArtifactError):
        completion.validation_state(source, parent)
    other.with_name(other.name + ".sha256").write_bytes(f"{original.sha256}  {other.name}\n".encode("ascii"))
    with pytest.raises(ValueError, match="escaped"):
        completion.validation_state(source, parent, allow_inflight=True)


@pytest.mark.parametrize("seen,invalid,repaired,recovered", [
    ({0}, [], set(), set()),
    ({0, 1}, [{"ordinal": 1}], set(), set()),
    ({0}, [], set(), {0, 1}),
    ({0, 1}, [], {1}, set()),
])
def test_final_admission_rejects_missing_unrepaired_overlapping_or_stale_state(seen, invalid, repaired, recovered):
    parent = SimpleNamespace(payload={"requests": [{}, {}]})
    with pytest.raises(ValueError, match="full corpus"):
        completion.require_complete_requests(parent, seen, invalid, repaired, recovered)


def test_full_request_gate_accepts_only_complete_unique_repair_and_transport_partition():
    parent = SimpleNamespace(payload={"requests": [{}, {}, {}]})
    completion.require_complete_requests(parent, {0, 1}, [{"ordinal": 1}], {1}, {2})


def test_live_process_wins_over_stale_receipt_and_missing_process_requires_terminal_evidence(tmp_path, monkeypatch):
    started = SimpleNamespace(payload={"pid": 123, "create_time": 45.0})
    monkeypatch.setattr(completion.psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: 45.0))
    assert completion.producer_finished(tmp_path, started, "policy") is None
    monkeypatch.setattr(completion.psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: 46.0))
    with pytest.raises(SealedArtifactError):
        completion.producer_finished(tmp_path, started, "policy")
    terminal, _ = publish_sealed_json(tmp_path / "finished.json", {
        "policy_sha256": "policy", "failed_batches": 0, "not_dispatched_batches": 0, "failures": []})
    assert completion.producer_finished(tmp_path, started, "policy").sha256 == terminal.sha256
    with pytest.raises(ValueError, match="source producer"):
        completion.producer_finished(tmp_path, started, "another-policy")
    def gone(pid):
        raise psutil.NoSuchProcess(pid)
    monkeypatch.setattr(completion.psutil, "Process", gone)
    assert completion.producer_finished(tmp_path, started, "policy").sha256 == terminal.sha256


def test_complete_coordinator_repairs_both_failure_types_and_publishes_only_full_store(tmp_path, monkeypatch, corpus):
    previous = tmp_path / "previous"
    completion.admission.assemble(corpus.source, previous, allow_partial=True)
    bad = Client('{"atoms":[{"label":"T0" "summary":"broken"}]}')
    runner = completion.admission.base.runner
    monkeypatch.setattr(runner, "_completion_client", lambda *a: bad)
    runner.run_one(corpus.source, corpus.plan, completion.sections.MODEL, corpus.bindings[2], True)
    publish_sealed_json(corpus.source / "public-source-provenance.json", {
        "sources_sha256": corpus.plan.payload["sources_sha256"],
        "body_bank_sha256": corpus.plan.payload["body_bank_sha256"],
        "private_workspace_or_current_conversation_text_in_model_inputs": False})
    control = tmp_path / "producer"
    producer_policy, _ = publish_sealed_json(control / "policy.json", {"preflight_sha256": corpus.plan.sha256})
    publish_sealed_json(control / "started.json", {
        "policy_sha256": producer_policy.sha256, "pid": os.getpid(), "create_time": -1})
    publish_sealed_json(control / "finished.json", {
        "policy_sha256": producer_policy.sha256, "failed_batches": 0, "not_dispatched_batches": 0, "failures": []})
    root = tmp_path / "completion"
    completion.prepare({"source_root": str(corpus.source), "producer_control": str(control),
                        "previous_store_root": str(previous)}, root)
    original = {p: p.read_bytes() for p in corpus.source.rglob("*.json")}
    monkeypatch.setattr(completion.sections, "_completion_client", lambda *a: corpus.good)
    monkeypatch.setattr(completion.malformed, "_completion_client", lambda *a: corpus.good)
    before = corpus.good.calls
    result = completion.run(root, enable_provider=True)
    assert result.payload["complete_source_compilation"] and result.payload["body_count"] == 2
    assert result.payload["new_provider_calls"] == corpus.good.calls - before == 2
    store = read_sealed_json(root / "complete-body-store" / "summary-bodies.json")
    assert store.payload["pending_batches"] == store.payload["unrepaired_batches"] == 0
    assert store.payload["repaired_batches"] == store.payload["json_repaired_batches"] == 1
    assert not result.payload["full100_target_passed"]
    assert all(p.read_bytes() == raw for p, raw in original.items())
    with pytest.raises(FileExistsError):
        completion.run(root, enable_provider=True)
    assert corpus.good.calls - before == 2
