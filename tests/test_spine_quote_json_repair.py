import json

import pytest

from memory_condense.search.spine_quote_json_repair import repair_support_list_closures


def broken_response():
    good = json.dumps({"atoms": [{"label": "T0", "summary": 'User asked about a 55" TV; summary contains "support".',
                                 "support": ['"a television"']}]}, separators=(",", ":"))
    return good.replace('\\""', '\\"'), good


def test_repairs_missing_support_terminator_without_rewriting_summary():
    broken, good = broken_response()
    fixed, audit = repair_support_list_closures(broken)
    assert fixed == good
    assert json.loads(fixed)["atoms"][0]["summary"] == json.loads(good)["atoms"][0]["summary"]
    assert audit["summary_texts_unchanged"] and audit["new_provider_calls"] == 0
    assert len(audit["original_insertion_offsets"]) == 1
    assert repair_support_list_closures(good) == (good, None)


def test_other_faults_and_summary_edits_are_rejected():
    broken, _ = broken_response()
    with pytest.raises((ValueError, json.JSONDecodeError)):
        repair_support_list_closures(broken.replace('"T0"', 'T0'))
    with pytest.raises(ValueError):
        repair_support_list_closures('{"atoms":[{"label":"T0","support":[],"summary":"text\\"]}]}')


def test_moves_misplaced_support_terminator_without_changing_summary():
    broken, good = broken_response()
    misplaced = broken.replace(']}', ']"}', 1)
    fixed, audit = repair_support_list_closures(misplaced)
    assert fixed == good
    assert len(audit["removed_misplaced_terminator_offsets"]) == 1
    assert audit["summary_texts_unchanged"]
