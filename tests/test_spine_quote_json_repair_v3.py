import json

import pytest

from memory_condense.search.spine_quote_json_repair_v2 import repair_support_list_closures as previous
from memory_condense.search.spine_quote_json_repair_v3 import repair_support_list_closures
from tests.test_spine_quote_json_repair import broken_response
from tests.test_spine_quote_json_repair_v2 import duplicated


def test_duplicated_delimiters_preserve_support_content_and_summary_literals():
    good = '{"atoms":[{"label":"T0","summary":"Assistant discusses garden access.","support":["Garden access is seasonal.","Plants need water."]}]}'
    broken = good.replace('["Garden access is seasonal.","Plants need water."]',
        '[""Garden access is seasonal."",""Plants need water."]')
    with pytest.raises(ValueError):
        previous(broken)
    repaired, audit = repair_support_list_closures(broken)
    assert repaired == good
    assert audit["summary_texts_unchanged"]
    assert audit["quote_edits"] and audit["new_provider_calls"] == 0


def test_combined_duplicate_opening_and_missing_terminator_are_confined_to_support():
    broken, good = broken_response()
    broken = broken.replace('"support":["', '"support":[""', 1)
    repaired, audit = repair_support_list_closures(broken)
    assert repaired == good
    assert audit["nested_terminator_repair"]["summary_texts_unchanged"]
    assert json.loads(repaired)["atoms"][0]["summary"] == json.loads(good)["atoms"][0]["summary"]


def test_previous_valid_and_repaired_responses_keep_exact_receipts():
    for pair in (broken_response(), duplicated()):
        for response in pair:
            assert repair_support_list_closures(response) == previous(response)


@pytest.mark.parametrize("response", [
    '{"atoms":[{"label":"T0","summary":""Broken summary"","support":[]}]}',
    '{"atoms":[{"label":"T0","summary":"Existing summary.","support":[""quote""],"extra":0}]}',
    '{"atoms":[{"label":"T1","summary":"Existing summary.","support":[""quote""]}]}',
])
def test_summary_edits_foreign_fields_and_changed_labels_are_rejected(response):
    with pytest.raises(ValueError):
        repair_support_list_closures(response)
