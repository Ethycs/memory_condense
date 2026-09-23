import pytest

from memory_condense.search.native_spine_memory import validate_body_summaries
from memory_condense.search.native_spine_summary import body_identity
from tools import assemble_native_spine_recovered as admission
from tools import recover_native_spine_transport as recovery
from tools import repair_native_recovery_section as sections
from tools import run_native_spine_batches as runner
from tests.test_native_spine_admission import corpus
from tests.test_native_spine_direct_admission import direct_repair, no_calls
from tests.test_native_spine_dispatch import FakeClient


def recover_pending(tmp_path, monkeypatch, corpus, *, invalid=False):
    monkeypatch.setattr(runner, "_completion_client", lambda *args: FakeClient(fail=True))
    report = runner.execute(corpus.source, admission.base.MODEL, True)
    assert report.payload["failures"][0]["ordinal"] == 2
    root = tmp_path/"recovery"
    recovery.prepare(corpus.source, report.path, root, [2])
    monkeypatch.setattr(recovery, "_completion_client", lambda *args: corpus.bad if invalid else corpus.good)
    recovery.execute(root, True)
    return root


def forbid(monkeypatch):
    no_calls(monkeypatch)
    monkeypatch.setattr(recovery, "_completion_client", lambda *args: pytest.fail("provider during admission"))
    monkeypatch.setattr(sections, "_completion_client", lambda *args: pytest.fail("provider during admission"))


def test_recoveries_and_subdivision_repairs_admit_complete_bodies_without_rewriting_originals(tmp_path, monkeypatch, corpus):
    recovered = recover_pending(tmp_path, monkeypatch, corpus)
    repairs = direct_repair(tmp_path, monkeypatch, corpus)
    original = {str(p): p.read_bytes() for p in (corpus.source/"checkpoints").rglob('*.json')}
    forbid(monkeypatch)
    root = tmp_path/"complete"
    result = admission.assemble(corpus.source, root, direct_roots=[repairs], recovery_roots=[recovered])
    assert result.payload["complete_source_compilation"]
    assert result.payload["body_count"] == 2 and result.payload["original_fragments_covered"] == 4
    assert result.payload["recovered_batches"] == result.payload["repaired_batches"] == 1
    assert not result.payload["full100_target_passed"]
    for reader in (admission.RecoveredSummaryBodies, admission.base.AdmittedSummaryBodies):
        store = reader(root)
        try:
            for body in corpus.bodies:
                assert validate_body_summaries(body, store.load(body_identity(body))) == body_identity(body)
        finally:
            store.close()
    assert admission.assemble(corpus.source, root, direct_roots=[repairs], recovery_roots=[recovered]).sha256 == result.sha256
    assert original == {str(p): p.read_bytes() for p in (corpus.source/"checkpoints").rglob('*.json')}


def test_recovery_does_not_hide_unrepaired_body_or_promote_partial_snapshot(tmp_path, monkeypatch, corpus):
    recovered = recover_pending(tmp_path, monkeypatch, corpus)
    forbid(monkeypatch)
    root = tmp_path/"partial"
    with pytest.raises(ValueError, match="every original batch"):
        admission.assemble(corpus.source, root, recovery_roots=[recovered])
    result = admission.assemble(corpus.source, root, recovery_roots=[recovered], allow_partial=True)
    assert result.payload["body_count"] == 1 and result.payload["unrepaired_batches"] == 1
    assert not result.payload["complete_source_compilation"]
    with pytest.raises(ValueError, match="partial snapshot"):
        admission.assemble(corpus.source, root, recovery_roots=[recovered])


def test_invalid_recovery_is_not_an_admitted_batch_and_duplicates_fail(tmp_path, monkeypatch, corpus):
    recovered = recover_pending(tmp_path, monkeypatch, corpus, invalid=True)
    forbid(monkeypatch)
    result = admission.assemble(corpus.source, tmp_path/"empty", recovery_roots=[recovered], allow_partial=True)
    assert result.payload["body_count"] == result.payload["recovered_batches"] == 0
    assert result.payload["pending_batches"] == 1


def test_duplicate_recovery_or_corrupt_recovered_database_is_rejected(tmp_path, monkeypatch, corpus):
    recovered = recover_pending(tmp_path, monkeypatch, corpus)
    forbid(monkeypatch)
    with pytest.raises(ValueError, match="duplicate transport recovery"):
        admission.verified_recoveries(corpus.plan, [recovered, recovered])
    root = tmp_path/"store"
    admission.assemble(corpus.source, root, recovery_roots=[recovered], allow_partial=True)
    with (root/"summary-bodies.sqlite").open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match="database changed"):
        admission.RecoveredSummaryBodies(root)


def test_rejected_recovery_section_can_be_subdivided_then_admitted_with_originals_intact(tmp_path, monkeypatch, corpus):
    recovered = recover_pending(tmp_path, monkeypatch, corpus, invalid=True)
    repaired = tmp_path/"recovery-section"
    sections.prepare(recovered, repaired, 2)
    monkeypatch.setattr(sections, "_completion_client", lambda *args: corpus.good)
    result = sections.execute(repaired, True)
    assert result.payload["complete_original_raw_coverage"]
    assert len(result.payload["summaries"]) > result.payload["original_atom_count"]
    forbid(monkeypatch)
    assert sections.execute(repaired, False).sha256 == result.sha256
    store_root = tmp_path/"section-store"
    admitted = admission.assemble(corpus.source, store_root, recovery_roots=[recovered],
        recovery_section_roots=[repaired], allow_partial=True)
    assert admitted.payload["body_count"] == 1
    assert admitted.payload["recovered_batches"] == admitted.payload["recovered_section_repaired_batches"] == 1
    store = admission.RecoveredSummaryBodies(store_root)
    try:
        body = corpus.bodies[1]
        assert validate_body_summaries(body, store.load(body_identity(body))) == body_identity(body)
    finally:
        store.close()


def test_section_repair_does_not_replace_an_already_valid_recovery(tmp_path, monkeypatch, corpus):
    recovered = recover_pending(tmp_path, monkeypatch, corpus)
    with pytest.raises(ValueError, match="invalid explicit recovery"):
        sections.prepare(recovered, tmp_path/"unnecessary", 2)
