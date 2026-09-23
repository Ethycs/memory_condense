from contextlib import closing
from types import SimpleNamespace

import pytest

from memory_condense.search.native_spine_memory import validate_body_summaries
from memory_condense.search.native_spine_summary import body_identity
from tests.test_native_spine_admission import corpus
from tests.test_native_spine_direct_admission import direct_repair, no_calls
from tests.test_native_spine_repair import Client
from tools import assemble_native_spine_json_recovered as admission
from tools import repair_native_json_batches as repair
from tools import run_native_spine_batches as runner
from tools.matched_eval.artifacts import read_sealed_json


def recover_json(tmp_path, monkeypatch, corpus, *, refine=False):
    bad = Client('{"atoms":[{"label":"T0" "summary":"broken"}]}')
    monkeypatch.setattr(runner, "_completion_client", lambda *args: bad)
    runner.run_one(corpus.source, corpus.plan, admission.base.MODEL, corpus.bindings[2], True)
    root = tmp_path / "json-repair"
    repair.prepare(corpus.source, root, [2])
    monkeypatch.setattr(repair, "_completion_client", lambda *args: corpus.bad if refine else corpus.good)
    result = repair.execute(root, True)
    if refine:
        assert not result.payload["complete_repair_snapshot"]
        successor = tmp_path / "json-refined"
        repair.prepare(corpus.source, successor, [2], previous=root)
        monkeypatch.setattr(repair, "_completion_client", lambda *args: corpus.good)
        result = repair.execute(successor, True)
        root = successor
    assert result.payload["complete_repair_snapshot"]
    return root


def forbid(monkeypatch):
    no_calls(monkeypatch)
    monkeypatch.setattr(repair, "_completion_client", lambda *args: pytest.fail("provider during admission"))


@pytest.mark.parametrize("refine", [False, True])
def test_malformed_json_recovery_enters_complete_body_store_with_failure_intact(tmp_path, monkeypatch, corpus, refine):
    repaired = direct_repair(tmp_path, monkeypatch, corpus)
    recovered = recover_json(tmp_path, monkeypatch, corpus, refine=refine)
    original = {p: p.read_bytes() for p in corpus.source.rglob("*.json")}
    forbid(monkeypatch)
    root = tmp_path / "complete"
    result = admission.assemble(corpus.source, root, direct_roots=[repaired], json_repair_roots=[recovered])
    p = result.payload
    assert p["complete_source_compilation"] and p["body_count"] == 2
    assert p["original_fragments_covered"] == 4
    assert p["json_repaired_batches"] == p["repaired_batches"] == 1
    assert p["recovered_batches"] == p["unrepaired_batches"] == p["pending_batches"] == 0
    assert not p["full100_target_passed"] and p["new_model_calls"] == 0
    for reader in (admission.JsonRecoveredSummaryBodies, admission.base.AdmittedSummaryBodies):
        with closing(reader(root)) as store:
            for body in corpus.bodies:
                assert validate_body_summaries(body, store.load(body_identity(body))) == body_identity(body)
    with pytest.raises(ValueError, match="producer changed"):
        admission.previous.RecoveredSummaryBodies(root)
    row = read_sealed_json(root / "admission-snapshot.json").payload["rows"][2]
    assert row["status"] == "json_repaired" and row["original_status"] == "invalid_summary"
    assert row["original_validation_sha256"] != row["validation_sha256"]
    assert admission.assemble(corpus.source, root, direct_roots=[repaired],
                              json_repair_roots=[recovered]).sha256 == result.sha256
    assert all(path.read_bytes() == data for path, data in original.items())


def test_json_recovery_does_not_hide_an_unrepaired_batch_or_promote_partial_store(tmp_path, monkeypatch, corpus):
    recovered = recover_json(tmp_path, monkeypatch, corpus)
    forbid(monkeypatch)
    root = tmp_path / "partial"
    with pytest.raises(ValueError, match="every original batch"):
        admission.assemble(corpus.source, root, json_repair_roots=[recovered])
    result = admission.assemble(corpus.source, root, json_repair_roots=[recovered], allow_partial=True)
    assert result.payload["body_count"] == 1 and result.payload["unrepaired_batches"] == 1
    assert not result.payload["complete_source_compilation"]
    with pytest.raises(ValueError, match="partial snapshot"):
        admission.assemble(corpus.source, root, json_repair_roots=[recovered])


def test_duplicate_json_lineage_and_wrong_source_are_rejected(tmp_path, monkeypatch, corpus):
    recovered = recover_json(tmp_path, monkeypatch, corpus)
    forbid(monkeypatch)
    with pytest.raises(ValueError, match="duplicate JSON recovery"):
        admission.verified_json_repairs(corpus.plan, [recovered, recovered])
    wrong = SimpleNamespace(sha256="wrong", payload=corpus.plan.payload)
    with pytest.raises(ValueError, match="another source compilation"):
        admission.verified_json_repairs(wrong, [recovered])


def test_json_store_reader_rejects_changed_database(tmp_path, monkeypatch, corpus):
    recovered = recover_json(tmp_path, monkeypatch, corpus)
    forbid(monkeypatch)
    root = tmp_path / "store"
    admission.assemble(corpus.source, root, json_repair_roots=[recovered], allow_partial=True)
    with (root / "summary-bodies.sqlite").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="database changed"):
        admission.JsonRecoveredSummaryBodies(root)
