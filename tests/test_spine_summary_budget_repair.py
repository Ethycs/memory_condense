import json

import pytest

from memory_condense.domain._discourse_identity import quote_sha256
from tools.repair_spine_summary_budget import apply_repairs


def fixture():
    long = "Assistant recommends " + "several books " * 100
    body = {"atoms": [{"label": "T0", "summary": "User asks about books.", "support": ["Original exact quote"]},
                      {"label": "T1", "summary": long, "support": ["Diagnostic quote remains"]}]}
    short = "Assistant recommends historical fiction books."
    repair = {"raw_request_sha256": "request", "label": "T1", "original_summary_sha256": quote_sha256(long),
              "summary": short, "summary_sha256": quote_sha256(short), "role": "assistant"}
    return body, repair


def test_compacts_only_bound_over_budget_summary_and_preserves_every_other_field():
    body, repair = fixture()
    result, applied = apply_repairs(json.dumps(body), "request", [repair])
    result = json.loads(result)
    assert result["atoms"][0] == body["atoms"][0]
    assert result["atoms"][1] == {**body["atoms"][1], "summary": repair["summary"]}
    assert applied == [repair]
    unchanged, applied = apply_repairs(json.dumps(body), "other-request", [repair])
    assert json.loads(unchanged) == body and not applied


def test_changed_source_or_over_budget_replacement_is_rejected():
    body, repair = fixture()
    for changed in ({**repair, "original_summary_sha256": "a" * 64},
                    {**repair, "summary": body["atoms"][1]["summary"], "summary_sha256": repair["original_summary_sha256"]}):
        with pytest.raises(ValueError, match="changed"):
            apply_repairs(json.dumps(body), "request", [changed])
    with pytest.raises(ValueError, match="changed"):
        apply_repairs(json.dumps(body), "request", [repair, repair])
