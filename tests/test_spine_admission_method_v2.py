from tools import verify_spine_admission_method as original
from tools import verify_spine_admission_method_v2 as verification
from tools.matched_eval.artifacts import read_sealed_json
from tests.test_spine_admission_method_verification import fixture


def test_v2_replays_original_admission_branches_without_changing_old_receipts(tmp_path, monkeypatch):
    methods = []
    for compact in (False, True):
        path, repair_root = fixture(tmp_path, monkeypatch, compact=compact)
        monkeypatch.setattr(verification, "prepare", original.prepare)
        prior = original.verify(path, repair_root)
        before = path.read_bytes(), prior.path.read_bytes()
        result = verification.verify(path, repair_root)
        assert result.payload["transport_lineage"]["recovery_used"] is False
        assert result.payload["method"]["compaction"]["maximum_provider_calls_per_namespace"] == 2
        assert result.payload["method"]["compaction"]["maximum_successful_batches_per_namespace"] == 1
        assert before == (path.read_bytes(), prior.path.read_bytes())
        methods.append(verification.load_verified_method(path, read_sealed_json(path))[0])
    assert methods[0] == methods[1]
