from __future__ import annotations

from dataclasses import replace

import pytest

from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.local_temporal_pair import (
    LocalTemporalPairError,
    resolve_parent_from_local_temporal_pair,
)


QUESTION = (
    "[Question asked at 2023/05/25 (Thu) 09:31]\n"
    "Which event happened first, the purchase of the coffee maker or the "
    "malfunction of the stand mixer?"
)
PARENT = "The stand mixer malfunction happened first."


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _inputs():
    provider_input = {
        "typed_evidence": {
            "handles": [
                {
                    "group_handle": "G001",
                    "handle_id": "H001",
                    "origin": "map",
                    "provenance_grade": "exact_citation",
                },
                {
                    "group_handle": "G003",
                    "handle_id": "H003",
                    "origin": "map",
                    "provenance_grade": "exact_citation",
                },
                {
                    "group_handle": "G900003",
                    "handle_id": "H900003",
                    "origin": "direct_pointer",
                    "provenance_grade": "direct_pointer",
                },
            ],
            "items": [
                {
                    "content_coherence": "match",
                    "handle_ids": ["H003"],
                    "included": True,
                    "kind": "event",
                    "date": "April 2023",
                    "status": "unknown",
                    "summary": (
                        "Date relative to the 2023/05/25 question: repair-shop "
                        "event began during April 2023; event: stand mixer "
                        "malfunction requiring repair."
                    ),
                    "value_authority": "explicit",
                },
                {
                    "content_coherence": "match",
                    "handle_ids": ["H001"],
                    "included": True,
                    "kind": "event",
                    "status": "unknown",
                    "summary": (
                        "Date relative to the 2023/05/25 question: about "
                        "2023/05/04; event: coffee maker purchased; state: "
                        "owned and in use."
                    ),
                    "value_authority": "explicit",
                },
                {
                    "content_coherence": "match",
                    "handle_ids": ["H900003"],
                    "included": True,
                    "kind": "event",
                    "date": "2022-03-21",
                    "status": "completed",
                    "summary": (
                        "I got a coffee maker that I'm giving to my coworker "
                        "Sarah for her birthday."
                    ),
                    "value_authority": "explicit",
                },
            ],
        }
    }
    contract = {
        "temporal_mode": "order",
        "by_handle": {
            "H001": {
                "answer_anchor_terms": ["coffee", "maker", "purchas"],
                "usable_item_receipt_sha256s": [_sha("coffee")],
            },
            "H003": {
                "answer_anchor_terms": ["stand", "mixer", "malfunction"],
                "usable_item_receipt_sha256s": [_sha("mixer")],
            },
            "H900003": {
                "answer_anchor_terms": ["coffee", "maker", "coworker", "sarah"],
                "usable_item_receipt_sha256s": [_sha("gift")],
            },
        },
    }
    return provider_input, contract


def _resolve(*, parent: str = PARENT, provider_input=None, contract=None):
    default_input, default_contract = _inputs()
    return resolve_parent_from_local_temporal_pair(
        dated_question=QUESTION,
        parent_prediction=parent,
        provider_input=provider_input or default_input,
        validation_contract=contract or default_contract,
        allowed_handle_ids=("H003", "H001", "H900003"),
        answer_plan_receipt_sha256=_sha("plan"),
        base_scope_receipt_sha256=_sha("scope"),
        source_completion_sha256=_sha("completion"),
    )


def test_target_local_pair_validates_parent_and_excludes_global_gift() -> None:
    result = _resolve()
    assert result is not None
    assert result.prediction == PARENT
    assert result.earlier_side == "right"
    assert result.proof_handle_ids == ("H003", "H001")
    assert "H900003" not in result.proof_handle_ids
    assert result.provider_calls == result.retained_transformer_token_state_bytes == 0


def test_direct_pointer_is_not_allowed_to_fill_a_missing_local_operand() -> None:
    provider_input, contract = _inputs()
    provider_input["typed_evidence"]["items"] = [
        row
        for row in provider_input["typed_evidence"]["items"]
        if row["handle_ids"] != ["H001"]
    ]
    assert _resolve(provider_input=provider_input, contract=contract) is None


def test_parent_must_name_the_locally_proven_earlier_side() -> None:
    assert _resolve(parent="The coffee maker purchase happened first.") is None


def test_overlapping_month_intervals_fail_closed() -> None:
    provider_input, contract = _inputs()
    coffee = next(
        row
        for row in provider_input["typed_evidence"]["items"]
        if row["handle_ids"] == ["H001"]
    )
    coffee["summary"] = (
        "Date relative to the question: about 2023/04/15; event: coffee maker "
        "purchased."
    )
    assert _resolve(provider_input=provider_input, contract=contract) is None


def test_question_timestamp_is_not_mistaken_for_an_undated_event() -> None:
    provider_input, contract = _inputs()
    coffee = next(
        row
        for row in provider_input["typed_evidence"]["items"]
        if row["handle_ids"] == ["H001"]
    )
    coffee["summary"] = (
        "Date relative to the 2023/05/25 question; event: coffee maker purchased."
    )
    assert _resolve(provider_input=provider_input, contract=contract) is None


def test_resolution_receipt_rejects_resealed_mutation() -> None:
    result = _resolve()
    assert result is not None
    with pytest.raises(LocalTemporalPairError, match="resolution changed"):
        replace(result, earlier_side="left")
