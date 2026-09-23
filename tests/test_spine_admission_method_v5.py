import json

import pytest

from tools import admit_spine_corpus_v4 as admission
from tools import repair_spine_summary_budget_v3 as repair
from tools import verify_spine_admission_method_v5 as verification
from tools.matched_eval.artifacts import read_sealed_json
from tests import test_spine_summary_budget_multibatch as fixtures


def test_duplicate_support_quote_and_all_oversized_summaries_replay_without_raw_qwen_input(tmp_path, monkeypatch):
    original_record = fixtures.record
    def record(text, **kwargs):
        body = json.loads(text)
        if "atoms" in body:
            body["atoms"][0]["support"] = ['"quoted text"']
            text = json.dumps(body).replace('"support": ["', '"support": [""', 1)
        return original_record(text, **kwargs)
    monkeypatch.setattr(fixtures, "record", record)
    monkeypatch.setattr(fixtures, "repair", repair)
    monkeypatch.setattr(fixtures, "admission", admission)
    monkeypatch.setattr(fixtures, "verification", verification)
    root, repair_root, preflight = fixtures.fixture(tmp_path, monkeypatch)
    assert "RAW_CANARY" not in json.dumps([b["messages"] for b in preflight.payload["batches"]])
    assert [len(b["row_indices"]) for b in preflight.payload["batches"]] == [8, 1]
    for batch in preflight.payload["batches"]:
        original_record(fixtures.output(batch), **fixtures.runtime_kwargs(repair_root, preflight, batch))
    monkeypatch.setattr(repair, "_completion_client", lambda *args: pytest.fail("replay attempted a provider call"))
    repair.run(repair_root, False)
    admission.admit(root, 0, 1, repair_root)
    path = root / "offset-000/source-bound-atoms-prefix-0001.json"
    before = path.read_bytes()
    result = verification.verify(path, repair_root)
    assert result.payload["additional_support_syntax_rule_needed"]
    assert result.payload["required_compaction_batches"] == 2
    assert result.payload["compaction_provider_attempts"] == 2
    assert result.payload["atom_count"] == 9
    assert verification.load_verified_method(path, read_sealed_json(path))[1] == result.sha256
    assert path.read_bytes() == before
    policy = read_sealed_json(root / "offset-000/source-binding-policy-v7-prefix-0001.json").payload
    legacy = {**policy, "format": "memory-condense-spine-source-binding-policy-v5",
        "syntax_repair": verification.LEGACY_SYNTAX_RULE,
        "implementation": verification.base.base.legacy.hashes(verification.base.base.ADMISSION_IMPLEMENTATION)}
    assert verification.conditional_method(policy) == verification.conditional_method(legacy)
    for changed in ({**policy, "syntax_repair": "rewrite summary content"},
                    {**policy, "summary_entailment_verified": True}):
        with pytest.raises(ValueError):
            verification.conditional_method(changed)
