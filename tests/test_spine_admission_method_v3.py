from tools import verify_spine_admission_method as original
from tools import verify_spine_admission_method_v2 as v2
from tools import verify_spine_admission_method_v3 as verification
from tools.matched_eval.artifacts import read_sealed_json
from tests.test_spine_admission_method_verification import fixture
from tests.test_spine_summary_budget_multibatch import fixture as multi_fixture, output, runtime_kwargs, record
from tools import repair_spine_summary_budget_v2 as repair
from tools import admit_spine_corpus_v2 as admission


def test_legacy_and_multibatch_require_exact_replay_before_sharing_method(tmp_path, monkeypatch):
    methods = []
    for compact in (False, True):
        path, repair_root = fixture(tmp_path, monkeypatch, compact=compact)
        monkeypatch.setattr(v2, "prepare", original.prepare)
        monkeypatch.setattr(verification, "prepare", original.prepare)
        prior = v2.verify(path, repair_root)
        before = path.read_bytes(), prior.path.read_bytes()
        result = verification.verify(path, repair_root)
        assert result.payload["required_compaction_batches"] == int(compact)
        assert before == (path.read_bytes(), prior.path.read_bytes())
        methods.append(verification.load_verified_method(path, read_sealed_json(path))[0])
    root, repair_root, preflight = multi_fixture(tmp_path, monkeypatch)
    for batch in preflight.payload["batches"]:
        record(output(batch), **runtime_kwargs(repair_root, preflight, batch))
    repair.run(repair_root)
    admission.admit(root, 0, 1, repair_root)
    path = root / "offset-000/source-bound-atoms-prefix-0001.json"
    result = verification.verify(path, repair_root)
    assert result.payload["required_compaction_batches"] == 2
    methods.append(verification.load_verified_method(path, read_sealed_json(path))[0])
    assert len(set(methods)) == 1
