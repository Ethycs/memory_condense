from __future__ import annotations

from dataclasses import replace

import pytest

from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.temporal_event_reconciler import (
    TemporalEventReconcilerError,
    reconcile_temporal_events,
)


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _item(handle: str, summary: str, *, when: str | None = None, value=None, unit=None):
    row = {
        "content_coherence": "match",
        "handle_ids": [handle],
        "included": True,
        "kind": "event",
        "status": "completed",
        "summary": summary,
        "value_authority": "explicit",
    }
    if when is not None:
        row["date"] = when
    if value is not None:
        row["numeric_value"] = value
        row["numeric_unit"] = unit
    return row


def _inputs(items, anchors):
    handles = [
        {
            "group_handle": f"G{index}",
            "handle_id": handle,
            "origin": "map",
            "provenance_grade": "exact_citation",
        }
        for index, handle in enumerate(anchors, 1)
    ]
    provider = {"typed_evidence": {"handles": handles, "items": items}}
    contract = {
        "by_handle": {
            handle: {
                "answer_anchor_terms": terms,
                "usable_item_receipt_sha256s": [_sha(f"item-{handle}")],
            }
            for handle, terms in anchors.items()
        }
    }
    return provider, contract


def _resolve(question, candidate, parent, items, anchors):
    provider, contract = _inputs(items, anchors)
    return reconcile_temporal_events(
        dated_question=question,
        candidate_prediction=candidate,
        parent_prediction=parent,
        provider_input=provider,
        validation_contract=contract,
        allowed_handle_ids=tuple(anchors),
        source_receipt_sha256=_sha("source"),
    )


def test_direct_duration_prefers_nearest_question_bound_statement() -> None:
    result = _resolve(
        "[Question asked at 2023/10/15 (Sun) 08:39]\n"
        "How long have I been living in my current apartment in Harajuku?",
        "3 months",
        "6 months",
        [
            _item("HOLD", "I have lived in my Harajuku apartment for 1 month.",
                  when="2023-04-11", value=1, unit="months"),
            _item("HNEW", "I have lived in my Harajuku apartment for 3 months.",
                  when="2023-10-15", value=3, unit="months"),
        ],
        {"HOLD": ["harajuku", "apartment"], "HNEW": ["harajuku", "apartment"]},
    )
    assert result is not None
    assert result.prediction == "3 months"
    assert result.prediction_source == "candidate"
    assert result.operation == "direct_duration"
    assert result.proof_handle_ids == ("HNEW",)


def test_two_named_event_interval_is_subtracted_before_answer_validation() -> None:
    result = _resolve(
        "[Question asked at 2023/05/26 (Fri) 17:49]\n"
        "How long had I been using the new area rug when I rearranged my living room furniture?",
        "7 days",
        "0 days",
        [
            _item("HRUG", "The new area rug entered use in the living room.", when="2023-04-28"),
            _item("HMOVE", "I rearranged the living room furniture.", when="2023-05-05"),
        ],
        {"HRUG": ["area", "rug"], "HMOVE": ["rearrang", "furniture"]},
    )
    assert result is not None
    assert result.prediction == "7 days"
    assert result.operation == "event_interval"
    assert result.proof_handle_ids == ("HRUG", "HMOVE")
    assert result.provider_calls == result.retained_transformer_token_state_bytes == 0


def test_binary_order_binds_both_named_events() -> None:
    result = _resolve(
        "[Question asked at 2023/05/25 (Thu) 11:29]\n"
        "Which device did I set up first, the smart thermostat or the mesh network system?",
        "The smart thermostat was first.",
        "The mesh network was first.",
        [
            _item("HT", "I set up the smart thermostat.", when="2023-04-25"),
            _item("HM", "I set up the mesh network system.", when="2023-05-04"),
        ],
        {"HT": ["smart", "thermostat"], "HM": ["mesh", "network"]},
    )
    assert result is not None
    assert result.prediction_source == "candidate"
    assert result.proof_handle_ids == ("HT", "HM")
    assert result.proof["computed"]["earlier_side"] == "left"


def test_binary_order_uses_consistent_elapsed_offsets_from_question_context() -> None:
    result = _resolve(
        "[Question asked at 2023/05/25 (Thu) 11:29]\n"
        "Which device did I set up first, the smart thermostat or the mesh network system?",
        "I cannot tell.",
        "I cannot tell.",
        [
            _item("HT", "Smart thermostat setup was one month before this conversation."),
            _item("HM", "Mesh network setup was three weeks before the 2023/05/25 question."),
        ],
        {"HT": ["smart", "thermostat"], "HM": ["mesh", "network"]},
    )
    assert result is not None
    assert result.prediction == "the smart thermostat"
    assert result.prediction_source == "computed"
    assert all(row["relative_relation"] for row in result.proof["evidence"])


def test_relative_interval_accepts_answer_inside_conservative_month_range() -> None:
    result = _resolve(
        "[Question asked at 2023/05/26 (Fri) 17:49]\n"
        "How long had I been using the new area rug when I rearranged my living room furniture?",
        "0 days",
        "one week",
        [
            _item("HRUG", "2023/05/26 03:36: the new area rug entered use one month earlier."),
            _item("HMOVE", "2023/05/26 18:55: I rearranged the living room furniture three weeks earlier."),
        ],
        {"HRUG": ["area", "rug"], "HMOVE": ["rearrang", "furniture"]},
    )
    assert result is not None
    assert result.prediction == "one week"
    assert result.prediction_source == "parent"
    assert result.proof["computed"]["duration_days_min"] <= 7
    assert result.proof["computed"]["duration_days_max"] >= 7


def test_conflicting_relative_offsets_for_one_named_event_fail_closed() -> None:
    result = _resolve(
        "[Question asked at 2023/05/25 (Thu) 11:29]\n"
        "Which device did I set up first, the smart thermostat or the mesh network system?",
        "The smart thermostat.",
        "The mesh network.",
        [
            _item("HT1", "Smart thermostat setup was one month before this conversation."),
            _item("HT2", "Smart thermostat setup was one week before this conversation."),
            _item("HM", "Mesh network setup was three weeks before this conversation."),
        ],
        {"HT1": ["smart", "thermostat"], "HT2": ["smart", "thermostat"],
         "HM": ["mesh", "network"]},
    )
    assert result is None


def test_unrelated_same_day_lexical_match_cannot_supply_named_event() -> None:
    # The date and the word "rug" match, but neither the sealed contract nor
    # the item binds the named rearrangement event.
    result = _resolve(
        "[Question asked at 2023/05/26 (Fri) 17:49]\n"
        "How long had I been using the new area rug when I rearranged my living room furniture?",
        "7 days",
        "0 days",
        [
            _item("HRUG", "The new area rug entered use in the living room.", when="2023-04-28"),
            _item("HNOISE", "I cleaned a rug after moving a chair.", when="2023-05-05"),
        ],
        {"HRUG": ["area", "rug"], "HNOISE": ["rug", "chair"]},
    )
    assert result is None


def test_same_day_conflicting_direct_durations_fail_closed() -> None:
    result = _resolve(
        "[Question asked at 2023/10/15 (Sun) 08:39]\n"
        "How long have I been living in my Harajuku apartment?",
        "3 months",
        "6 months",
        [
            _item("HA", "I have lived in my Harajuku apartment for 3 months.",
                  when="2023-10-15", value=3, unit="months"),
            _item("HB", "I have lived in my Harajuku apartment for 6 months.",
                  when="2023-10-15", value=6, unit="months"),
        ],
        {"HA": ["harajuku", "apartment"], "HB": ["harajuku", "apartment"]},
    )
    assert result is None


def test_composite_duration_is_not_mistaken_for_one_component() -> None:
    result = _resolve(
        "[Question asked at 2023/05/24 (Wed) 08:08]\n"
        "How long did I take to finish Book A and Book B combined?",
        "five weeks",
        "five weeks",
        [_item("HB", "I took three weeks to finish Book B.",
               when="2023-05-24", value=3, unit="weeks")],
        {"HB": ["finish", "book"]},
    )
    assert result is None


def test_days_ago_emits_query_minus_named_event_date() -> None:
    result = _resolve(
        "[Question asked at 2022/04/15 (Fri) 18:46]\n"
        "How many days ago did I attend the baking class?",
        "5 days",
        "21 days",
        [_item("HCLASS", "I attended the baking class.", when="2022-03-20")],
        {"HCLASS": ["bake", "class"]},
    )
    assert result is not None
    assert result.prediction == "26 days"
    assert result.prediction_source == "computed"


def test_interval_emits_canonical_computed_answer_when_models_both_miss() -> None:
    result = _resolve(
        "[Question asked at 2023/05/26 (Fri) 17:49]\n"
        "How long had I been using the new area rug when I rearranged my living room furniture?",
        "about a while",
        "0 days",
        [
            _item("HRUG", "The new area rug entered use in the living room.", when="2023-04-28"),
            _item("HMOVE", "I rearranged the living room furniture.", when="2023-05-05"),
        ],
        {"HRUG": ["area", "rug"], "HMOVE": ["rearrang", "furniture"]},
    )
    assert result is not None
    assert result.prediction == "7 days"
    assert result.prediction_source == "computed"


def test_order_emits_exact_named_side_when_models_both_miss() -> None:
    result = _resolve(
        "[Question asked at 2023/05/25 (Thu) 11:29]\n"
        "Which device did I set up first, the smart thermostat or the mesh network system?",
        "I cannot tell.",
        "There is not enough information.",
        [
            _item("HT", "I set up the smart thermostat.", when="2023-04-25"),
            _item("HM", "I set up the mesh network system.", when="2023-05-04"),
        ],
        {"HT": ["smart", "thermostat"], "HM": ["mesh", "network"]},
    )
    assert result is not None
    assert result.prediction == "the smart thermostat"
    assert result.prediction_source == "computed"


def test_resolution_receipt_rejects_resealed_mutation() -> None:
    result = _resolve(
        "[Question asked at 2023/05/25 (Thu) 11:29]\n"
        "Which device did I set up first, the smart thermostat or the mesh network system?",
        "The smart thermostat was first.",
        "The mesh network was first.",
        [
            _item("HT", "I set up the smart thermostat.", when="2023-04-25"),
            _item("HM", "I set up the mesh network system.", when="2023-05-04"),
        ],
        {"HT": ["smart", "thermostat"], "HM": ["mesh", "network"]},
    )
    assert result is not None
    with pytest.raises(TemporalEventReconcilerError, match="resolution changed"):
        replace(result, prediction_source="parent")
