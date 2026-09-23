import json

import pytest

from memory_condense.search.spine_quote_json_repair import repair_support_list_closures as legacy
from memory_condense.search.spine_quote_json_repair_v2 import repair_support_list_closures
from tests.test_spine_quote_json_repair import broken_response


def duplicated():
    good = json.dumps({"atoms": [{"label": "T0", "summary": 'User discusses a blog called "First Light".',
        "support": ['"First Light" is a blog name.', 'The second quote is unchanged.']}]}, separators=(",", ":"))
    return good.replace('"support":["', '"support":[""', 1), good


def test_duplicate_opening_quote_is_removed_only_from_support():
    broken, good = duplicated()
    with pytest.raises(ValueError):
        legacy(broken)
    repaired, audit = repair_support_list_closures(broken)
    assert repaired == good
    assert audit["summary_texts_unchanged"] and audit["new_provider_calls"] == 0
    assert len(audit["original_removal_offsets"]) == 1
    assert json.loads(repaired)["atoms"][0]["summary"] == json.loads(good)["atoms"][0]["summary"]


def test_legacy_repair_and_valid_bytes_keep_their_original_receipts():
    broken, good = broken_response()
    for response in (broken, good, broken.replace(']}', ']"}', 1)):
        assert repair_support_list_closures(response) == legacy(response)


@pytest.mark.parametrize("change", [
    lambda text: text.replace('"T0"', 'T0'),
    lambda text: text.replace('"summary":"', '"summary":""'),
    lambda text: text.replace('"summary":', '"extra":"value","summary":'),
    lambda text: text.replace('"atoms":', '"atoms":[],"atoms":'),
    lambda text: text + ' trailing material',
])
def test_other_syntax_changes_and_duplicate_fields_are_rejected(change):
    broken, _ = duplicated()
    with pytest.raises(ValueError):
        repair_support_list_closures(change(broken))
