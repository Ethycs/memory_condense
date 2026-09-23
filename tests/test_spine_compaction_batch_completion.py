import json

import pytest

from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from tools import admit_spine_corpus_v6 as admission
from tools import repair_spine_summary_budget_v4 as parent
from tools import finish_spine_compaction_batches as completion
from tools import recover_spine_summary_budget_v2 as recovery
from tools import build_spine_corpus_hierarchy_resilient as resilient
from tools import verify_spine_admission_method_v8 as verification
from tools.matched_eval.artifacts import read_sealed_json
from tests import test_spine_summary_budget_multibatch as fixtures
from tests.test_spine_admission_method_verification import Client


def fixture(tmp_path, monkeypatch, *, invalid=True):
    monkeypatch.setattr(fixtures, "repair", parent)
    monkeypatch.setattr(fixtures, "admission", admission)
    monkeypatch.setattr(fixtures, "verification", verification)
    corpus, root, preflight = fixtures.fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(verification.previous, "prepare", admission.prepare)
    monkeypatch.setattr(admission.previous, "prepare", admission.prepare)
    first, second = preflight.payload["batches"]
    text = json.loads(fixtures.output(first))
    long = "User reports " + "several tasks " * 100
    if invalid:
        text["summaries"][0]["summary"] = long
    fixtures.record(json.dumps(text), **fixtures.runtime_kwargs(root, preflight, first))
    return corpus, root, preflight, second, long


@pytest.mark.parametrize("needs_second", [False, True])
def test_finish_unstarted_batch_then_recover_only_invalid_slot_and_verify_all_atoms(tmp_path, monkeypatch, needs_second):
    corpus, parent_root, original, second, long = fixture(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in parent_root.rglob("*.json")}
    finish_root, recovery_root = tmp_path / "finish", tmp_path / "recover"
    plan = completion.prepare(finish_root, parent_root)
    assert plan.payload["maximum_new_calls"] == 1
    captured = []
    class CapturingClient(Client):
        def create(self, **kwargs):
            captured.append(kwargs["messages"])
            return super().create(**kwargs)
    monkeypatch.setattr(completion, "_completion_client", lambda *args: CapturingClient(fixtures.output(second)))
    finished = completion.run(finish_root, True)
    assert len(captured) == 1 and captured[0] == second["messages"]
    assert [r["invalid_slots"] for r in finished.payload["rows"]] == [[0], []]
    assert all(p.read_bytes() == content for p, content in before.items())
    monkeypatch.setattr(completion, "_completion_client", lambda *args: pytest.fail("completed original resent"))
    assert completion.run(finish_root, False).sha256 == finished.sha256
    preflight = recovery.prepare(recovery_root, parent_root)
    assert len(preflight.payload["failures"]) == 1
    assert preflight.payload["maximum_recovery_calls"] == 2
    outputs = ([json.dumps({"summary": long})] if needs_second else []) + [json.dumps({"summary": "User reports item 0."})]
    clients = iter(CapturingClient(text) for text in outputs)
    monkeypatch.setattr(resilient, "_completion_client", lambda *args: next(clients))
    repaired = recovery.run(recovery_root, True, 2)
    assert repaired.payload["recovery_attempts"] == 1 + int(needs_second)
    assert [r["summary"] for r in repaired.payload["rows"]] == [f"User reports item {i}." for i in range(9)]
    assert "RAW_CANARY" not in json.dumps(captured)
    monkeypatch.setattr(resilient, "_completion_client", lambda *args: pytest.fail("replay attempted recovery"))
    assert recovery.run(recovery_root, False, 0).sha256 == repaired.sha256
    admission.admit(corpus, 0, 1, recovery_root)
    atoms_path = corpus / "offset-000/source-bound-atoms-prefix-0001.json"
    atoms_bytes = atoms_path.read_bytes()
    verified = verification.verify(atoms_path, recovery_root)
    assert verified.payload["atom_count"] == 9
    assert verified.payload["compaction_provider_attempts"] == 3 + int(needs_second)
    assert verified.payload["compaction_recovery_attempts"] == 1 + int(needs_second)
    assert verification.load_verified_method(atoms_path, read_sealed_json(atoms_path))[1] == verified.sha256
    assert atoms_path.read_bytes() == atoms_bytes
    # The conditional method remains common when every original slot was valid.
    other, regular, regular_preflight, _, _ = fixture(tmp_path / "regular", monkeypatch, invalid=False)
    last = regular_preflight.payload["batches"][1]
    fixtures.record(fixtures.output(last), **fixtures.runtime_kwargs(regular, regular_preflight, last))
    parent.run(regular, False)
    admission.admit(other, 0, 1, regular)
    ordinary = verification.verify(other / "offset-000/source-bound-atoms-prefix-0001.json", regular)
    assert ordinary.payload["method_sha256"] == verified.payload["method_sha256"]
    assert ordinary.payload["compaction_recovery_attempts"] == 0


def test_unacknowledged_original_cannot_receive_a_completion_release(tmp_path, monkeypatch):
    _, parent_root, original, second, _ = fixture(tmp_path, monkeypatch)
    class FailingClient(Client):
        def create(self, **kwargs):
            raise RuntimeError("simulated unknown original response")
    runtime = FastCompletionRuntime(client=FailingClient(""), **fixtures.runtime_kwargs(parent_root, original, second))
    try:
        with pytest.raises(RuntimeError, match="simulated unknown"):
            runtime.run()
    finally:
        runtime.close()
    before = {p: p.read_bytes() for p in parent_root.rglob("*.json")}
    monkeypatch.setattr(completion, "_completion_client", lambda *args: pytest.fail("unknown original was resent"))
    with pytest.raises((ValueError, RuntimeError)):
        completion.prepare(tmp_path / "finish", parent_root)
    assert all(p.read_bytes() == content for p, content in before.items())
    assert not (tmp_path / "finish/preflight.json").exists()


def test_two_invalid_recoveries_do_not_publish_or_retry_after_exhaustion(tmp_path, monkeypatch):
    _, parent_root, original, second, long = fixture(tmp_path, monkeypatch)
    fixtures.record(fixtures.output(second), **fixtures.runtime_kwargs(parent_root, original, second))
    root = tmp_path / "recovery"
    recovery.prepare(root, parent_root)
    calls = []
    class InvalidClient(Client):
        def create(self, **kwargs):
            calls.append(kwargs["messages"])
            return super().create(**kwargs)
    monkeypatch.setattr(resilient, "_completion_client", lambda *args: InvalidClient(json.dumps({"summary": long})))
    with pytest.raises(ValueError, match="exhausted"):
        recovery.run(root, True, 2)
    assert len(calls) == 2 and not (root / "repairs.json").exists()
    monkeypatch.setattr(resilient, "_completion_client", lambda *args: pytest.fail("exhausted recovery retried"))
    with pytest.raises(ValueError, match="exhausted"):
        recovery.run(root, False, 0)
