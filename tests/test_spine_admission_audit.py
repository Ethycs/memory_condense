import json

from tools.audit_spine_source_admission import classify_failure


def test_audit_collects_every_budget_failure_and_distinguishes_invalid_attribution():
    rows = [{"label": f"T{i}", "summary": "long summary " * 150, "support": ["diagnostic quote"]} for i in range(2)]
    oversized, schema = classify_failure(json.dumps({"atoms": rows}), 2)
    assert [r["label"] for r in oversized] == ["T0", "T1"] and not schema
    rows[1]["label"] = "T0"
    oversized, schema = classify_failure(json.dumps({"atoms": rows}), 2)
    assert [r["label"] for r in oversized] == ["T0"] and schema
    assert classify_failure(json.dumps({"atoms": rows}), 3) == ([], True)
