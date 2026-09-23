import pytest

from memory_condense.search.native_spine_memory import validate_body_summaries
from memory_condense.search.native_spine_summary import body_identity
from tools import assemble_native_spine_direct_repairs as admission
from tools import repair_native_spine_sections as repair
from tools import run_native_spine_batches as runner
from tools.matched_eval.artifacts import read_sealed_json
from tests.test_native_spine_admission import corpus, forbid_provider, repair_chain


def direct_repair(tmp_path, monkeypatch, corpus):
    root = tmp_path/"direct-repair"
    repair.prepare(corpus.source, root, [0])
    monkeypatch.setattr(repair, "_completion_client", lambda *args: corpus.good)
    result = repair.execute(root, True)
    assert result.payload["complete_repair_snapshot"]
    return root


def no_calls(monkeypatch):
    forbid_provider(monkeypatch)
    def forbidden(*args):
        raise AssertionError("body admission must not contact a provider")
    monkeypatch.setattr(repair, "_completion_client", forbidden)


def test_direct_repairs_produce_read_compatible_complete_bodies_with_distinct_producer_binding(tmp_path, monkeypatch, corpus):
    repairs = direct_repair(tmp_path, monkeypatch, corpus)
    originals = [read_sealed_json(p).sha256 for p in sorted((corpus.source/"validated").glob('*.json'))]
    no_calls(monkeypatch)
    root = tmp_path/"partial"
    result = admission.assemble(corpus.source, root, direct_roots=[repairs], allow_partial=True)
    assert result.payload["body_count"] == 1 and result.payload["additional_sections"] > 0
    assert not result.payload["complete_source_compilation"]
    assert result.payload["producer_implementation"] == admission.implementation()
    assert result.payload["implementation"] == admission.base.implementation()
    for loader in (admission.DirectRepairSummaryBodies, admission.base.AdmittedSummaryBodies):
        store = loader(root)
        try:
            atoms = store.load(body_identity(corpus.bodies[0]))
            assert validate_body_summaries(corpus.bodies[0], atoms) == body_identity(corpus.bodies[0])
            with pytest.raises(KeyError):
                store.load(body_identity(corpus.bodies[1]))
        finally:
            store.close()
    assert originals == [read_sealed_json(p).sha256 for p in sorted((corpus.source/"validated").glob('*.json'))]
    assert admission.assemble(corpus.source, root, direct_roots=[repairs], allow_partial=True).sha256 == result.sha256
    with pytest.raises(ValueError, match="partial snapshot"):
        admission.assemble(corpus.source, root, direct_roots=[repairs])

    monkeypatch.setattr(runner, "_completion_client", lambda *args: corpus.good)
    runner.run_one(corpus.source, corpus.plan, admission.base.MODEL, corpus.bindings[2], True)
    no_calls(monkeypatch)
    assert admission.assemble(corpus.source, root, direct_roots=[repairs], allow_partial=True).sha256 == result.sha256
    complete = admission.assemble(corpus.source, tmp_path/"complete", direct_roots=[repairs])
    assert complete.payload["complete_source_compilation"] and complete.payload["body_count"] == 2
    assert complete.payload["original_fragments_covered"] == 4
    assert not complete.payload["full100_target_passed"]


def test_legacy_repairs_remain_supported_but_duplicate_lineages_cannot_replace_a_batch_twice(tmp_path, monkeypatch, corpus):
    legacy = repair_chain(tmp_path, monkeypatch, corpus)
    new = direct_repair(tmp_path, monkeypatch, corpus)
    no_calls(monkeypatch)
    with pytest.raises(ValueError, match="duplicate repaired batch"):
        admission.verified_repairs(corpus.plan, [legacy], [new])
    result = admission.assemble(corpus.source, tmp_path/"legacy-store", legacy_roots=[legacy], allow_partial=True)
    assert result.payload["body_count"] == 1 and result.payload["repaired_batches"] == 1


def test_unrepaired_or_pending_fragments_still_exclude_the_whole_body(tmp_path, monkeypatch, corpus):
    no_calls(monkeypatch)
    root = tmp_path/"empty"
    with pytest.raises(ValueError, match="every original batch"):
        admission.assemble(corpus.source, root)
    assert not (root/"admission-snapshot.json").exists()
    result = admission.assemble(corpus.source, root, allow_partial=True)
    assert result.payload["body_count"] == result.payload["atom_count"] == 0
    assert result.payload["unrepaired_batches"] == result.payload["pending_batches"] == 1


def test_repair_database_corruption_is_rejected(tmp_path, monkeypatch, corpus):
    repairs = direct_repair(tmp_path, monkeypatch, corpus)
    no_calls(monkeypatch)
    root = tmp_path/"store"
    admission.assemble(corpus.source, root, direct_roots=[repairs], allow_partial=True)
    with (root/"summary-bodies.sqlite").open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match="database changed"):
        admission.DirectRepairSummaryBodies(root)
