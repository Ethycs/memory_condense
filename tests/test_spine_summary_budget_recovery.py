import json

import pytest

from tools import recover_spine_summary_budget as recovery
from tools import admit_spine_corpus_v3 as admission
from tools import build_spine_corpus_hierarchy_resilient as resilient
from tools import verify_spine_admission_method_v4 as verification
from tools import verify_spine_admission_method as original
from tools.matched_eval.artifacts import read_sealed_json
from tests.test_spine_admission_method_verification import Client, fixture as legacy_fixture
from tests.test_spine_summary_budget_multibatch import fixture, output, runtime_kwargs, record
from tools import admit_spine_corpus_v2 as multi_admission


def prepared(tmp_path, monkeypatch):
    corpus, parent, preflight = fixture(tmp_path, monkeypatch)
    first, second = preflight.payload["batches"]
    record(output(first), **runtime_kwargs(parent, preflight, first))
    long = "User reports " + "several tasks " * 100
    record(json.dumps({"summaries": [{"label": "S0", "summary": long}]}),
        **runtime_kwargs(parent, preflight, second))
    root = tmp_path / "recovery"
    recovery.prepare(root, parent)
    monkeypatch.setattr(admission, "prepare", multi_admission.prepare)
    return corpus, parent, root, long


@pytest.mark.parametrize("needs_second", [False, True])
def test_failed_slot_recovers_without_replacing_eight_valid_outputs(tmp_path, monkeypatch, needs_second):
    corpus, parent, root, long = prepared(tmp_path, monkeypatch)
    original_bytes = {p: p.read_bytes() for p in parent.rglob("*.json")}
    captured = []
    contents = ([json.dumps({"summary": long})] if needs_second else []) + [json.dumps({"summary": "User reports item 8."})]

    class CapturingClient(Client):
        def create(self, **kwargs):
            captured.append(kwargs["messages"])
            return super().create(**kwargs)

    clients = iter(CapturingClient(content) for content in contents)
    monkeypatch.setattr(resilient, "_completion_client", lambda *args: next(clients))
    repaired = recovery.run(root, True, 2)
    assert repaired.payload["recovered_summary_count"] == 1
    assert repaired.payload["recovery_attempts"] == 1 + int(needs_second)
    assert [r["summary"] for r in repaired.payload["rows"]] == [f"User reports item {i}." for i in range(9)]
    assert len(captured) == 1 + int(needs_second)
    assert "RAW_CANARY" not in json.dumps(captured)
    assert original_bytes == {p: p.read_bytes() for p in parent.rglob("*.json")}
    monkeypatch.setattr(resilient, "_completion_client", lambda *args: pytest.fail("replay attempted provider"))
    assert recovery.run(root, False, 0).sha256 == repaired.sha256
    admission.admit(corpus, 0, 1, root)
    path = corpus / "offset-000/source-bound-atoms-prefix-0001.json"
    result = verification.verify(path, root)
    assert result.payload["compaction_provider_attempts"] == 3 + int(needs_second)
    assert verification.load_verified_method(path, read_sealed_json(path))[1] == result.sha256
    method = result.payload["method_sha256"]
    for compact in (False, True):
        path, repair_root = legacy_fixture(tmp_path, monkeypatch, compact=compact)
        monkeypatch.setattr(verification.base, "prepare", original.prepare)
        before = path.read_bytes()
        prior = verification.verify(path, repair_root)
        assert prior.payload["method_sha256"] == method
        assert prior.payload["compaction_recovery_attempts"] == 0
        assert before == path.read_bytes()


def test_two_invalid_recoveries_exhaust_budget_without_publishing(tmp_path, monkeypatch):
    _, parent, root, long = prepared(tmp_path, monkeypatch)
    calls = []

    class InvalidClient(Client):
        def create(self, **kwargs):
            calls.append(kwargs["messages"])
            return super().create(**kwargs)

    monkeypatch.setattr(resilient, "_completion_client", lambda *args: InvalidClient(json.dumps({"summary": long})))
    with pytest.raises(ValueError, match="exhausted"):
        recovery.run(root, True, 2)
    assert len(calls) == 2
    assert not (root / "repairs.json").exists()
    assert not (parent / "repairs.json").exists()
    monkeypatch.setattr(resilient, "_completion_client", lambda *args: pytest.fail("exhausted recovery retried"))
    with pytest.raises(ValueError, match="exhausted"):
        recovery.run(root, False, 0)
