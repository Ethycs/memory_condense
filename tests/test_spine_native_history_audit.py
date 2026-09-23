from copy import deepcopy

import pytest

from memory_condense.domain._discourse_identity import quote_sha256
from tools.audit_spine_native_history import covered_characters, inspect_case, membership_key, selected_intervals


def fixture():
    date = "2023-05-21T16:24:00+00:00"
    native = {"haystack_session_ids": ["meeting"], "haystack_dates": ["2023/05/21 (Sun) 16:24"],
              "haystack_sessions": [[{"role": "user", "content": "coffee shop", "has_answer": True}]]}
    turns = {"a": {"turn_id": "a", "source_id": "other_owner::meeting", "role": "user",
                    "created_at": date, "text": "coffee shop", "text_sha256": quote_sha256("coffee shop")},
             "b": {"turn_id": "b", "source_id": "native_owner::meeting", "role": "user",
                    "created_at": date, "text": "grocery store", "text_sha256": quote_sha256("grocery store")}}
    return native, turns


def hydration(turns, specs):
    result = []
    for tid, start, end in specs:
        t = turns[tid]
        text = t["text"][start:end]
        result.append({"text": text, "span": {**{k: t[k] for k in ("turn_id", "source_id", "role", "created_at")},
            "start_char": start, "end_char": end, "turn_text_sha256": t["text_sha256"],
            "span_text_sha256": quote_sha256(text)}})
    return {"sections": [{"evidence": result}]}


def test_shared_session_requires_full_content_provenance_not_owner_prefix():
    native, turns = fixture()
    result = inspect_case(native, turns, hydration(turns, [("a", 0, 11), ("b", 0, 13)]))
    assert result["selected_native_user_turn_ids"] == ["a"]
    assert result["selected_foreign_user_turn_ids"] == ["b"]
    assert result["all_annotated_turns_fully_hydrated"]
    assert not result["foreign_means_contradictory"]
    changed = {**turns["a"], "created_at": "2023-05-22T16:24:00+00:00"}
    assert membership_key(changed) != membership_key(turns["a"])


def test_overlap_and_duplicate_source_copies_do_not_inflate_coverage():
    native, turns = fixture()
    turns["c"] = {**turns["a"], "turn_id": "c", "source_id": "third::meeting"}
    result = inspect_case(native, turns, hydration(turns, [("a", 0, 6), ("a", 0, 6), ("c", 3, 7)]))
    assert result["annotated_turns"][0]["covered_characters"] == 7
    assert not result["all_annotated_turns_fully_hydrated"]
    complete = inspect_case(native, turns, hydration(turns, [("a", 0, 6), ("c", 6, 11)]))
    assert complete["all_annotated_turns_fully_hydrated"]


def test_missing_native_turn_is_not_satisfied_by_conflicting_same_session():
    native, turns = fixture()
    del turns["a"]
    result = inspect_case(native, turns, hydration(turns, [("b", 0, 13)]))
    assert result["native_turns_present_in_pool"] == 0
    assert not result["annotated_turns"][0]["present_in_pool"]
    assert not result["all_annotated_turns_fully_hydrated"]


@pytest.mark.parametrize("field,value", [("role", "assistant"), ("source_id", "x::y"),
    ("start_char", -1), ("end_char", 100), ("turn_text_sha256", "0" * 64),
    ("span_text_sha256", "0" * 64)])
def test_tampered_hydration_cannot_supply_coverage(field, value):
    _, turns = fixture()
    packet = hydration(turns, [("a", 0, 11)])
    packet["sections"][0]["evidence"][0]["span"][field] = value
    with pytest.raises(ValueError):
        selected_intervals(packet, turns)


def test_no_annotations_does_not_claim_sufficiency():
    native, turns = fixture()
    native = deepcopy(native)
    native["haystack_sessions"][0][0]["has_answer"] = False
    result = inspect_case(native, turns, hydration(turns, [("a", 0, 11)]))
    assert not result["all_annotated_turns_fully_hydrated"]
    assert covered_characters([(0, 4), (2, 6), (6, 11)], 11) == 11
