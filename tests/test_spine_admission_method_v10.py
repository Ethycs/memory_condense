import json

import pytest

from tools import admit_spine_corpus_v6 as admission
from tools import build_spine_corpus_hierarchy_resilient as resilient
from tools import recover_spine_summary_budget_v2 as recovery
from tools import repair_spine_summary_budget_v4 as compact
from tools import verify_spine_admission_method_v10 as verification
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json
from tests import test_spine_compaction_batch_completion as completed
from tests import test_spine_compaction_transport as transport_tests
from tests import test_spine_summary_budget_multibatch as batches
from tests.test_spine_admission_method_verification import Client


def test_prior_compaction_transport_and_regular_admission_keep_one_method(tmp_path, monkeypatch):
    monkeypatch.setattr(transport_tests, "verification", verification)
    transport_tests.test_all_originals_preserved_and_additional_attempt_counted_with_common_method(tmp_path, monkeypatch)


def content_fixture(tmp_path, monkeypatch, *, invalid=False):
    corpus, parent, preflight, second, _ = completed.fixture(tmp_path, monkeypatch, invalid=invalid)
    batches.record(batches.output(second), **batches.runtime_kwargs(parent, preflight, second))
    repair_root = parent
    if invalid:
        repair_root = tmp_path / "slot-recovery"
        recovery.prepare(repair_root, parent)
        monkeypatch.setattr(resilient, "_completion_client", lambda *args:
            Client(json.dumps({"summary": "User reports item 0."})))
        recovery.run(repair_root, True, 2)
    else:
        compact.run(parent, False)
    admission.admit(corpus, 0, 1, repair_root)
    atoms = corpus / "offset-000/source-bound-atoms-prefix-0001.json"
    publish_sealed_json(corpus.parent / "stage.json", {
        "format": "memory-condense-spine-transport-recovery-stage-result-v3"})
    monkeypatch.setattr(verification, "prepare", admission.prepare)
    verified_lineage = []
    def verify_lineage(root, offset, repairs):
        assert root == corpus and offset == 0 and repairs == repair_root
        verified_lineage.append((root, offset, repairs))
        return {"recovery_used": True, "additional_raw_attempts": 1, "additional_compaction_attempts": 0}
    # Native raw admission and compaction replay are exercised here. The complete
    # stage/transport verifier is exercised separately with real runtime journals.
    monkeypatch.setattr(verification, "verify_transport_lineage", verify_lineage)
    monkeypatch.setattr(compact, "_completion_client", lambda *args: pytest.fail("verification called Qwen"))
    monkeypatch.setattr(resilient, "_completion_client", lambda *args: pytest.fail("verification called Qwen"))
    return atoms, repair_root, verified_lineage


@pytest.mark.parametrize("invalid", [False, True])
def test_session_admission_replays_all_raw_and_compaction_content(tmp_path, monkeypatch, invalid):
    atoms, repairs, calls = content_fixture(tmp_path, monkeypatch, invalid=invalid)
    before = atoms.read_bytes()
    result = verification.verify(atoms, repairs)
    assert len(calls) == 1
    assert result.payload["atom_count"] == result.payload["budget_compacted_atoms"] == 9
    assert result.payload["compaction_provider_attempts"] == 2 + int(invalid)
    assert result.payload["transport_lineage"]["additional_raw_attempts"] == 1
    assert result.payload["new_provider_calls"] == 0
    assert atoms.read_bytes() == before
    assert verification.load_verified_method(atoms, read_sealed_json(atoms))[1] == result.sha256
    assert len(calls) == 2
    policy = read_sealed_json(atoms.parent / f'source-binding-policy-v{9 if invalid else 8}-prefix-0001.json')
    assert result.payload["method"] == verification.conditional_method(policy.payload)


@pytest.mark.parametrize("failure", ["omitted_repairs", "missing_raw_response", "unverified_lineage"])
def test_session_admission_requires_content_and_transport_to_verify(tmp_path, monkeypatch, failure):
    atoms, repairs, calls = content_fixture(tmp_path, monkeypatch)
    if failure == "omitted_repairs":
        repairs = None
    elif failure == "missing_raw_response":
        next((atoms.parent / "raw-checkpoints").glob("*/*.response.json")).unlink()
    else:
        def reject(*args):
            raise ValueError("fixture incomplete transport")
        monkeypatch.setattr(verification, "verify_transport_lineage", reject)
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        verification.verify(atoms, repairs)
    assert not atoms.with_name("conditional-method-v10-prefix-0001.json").exists()
