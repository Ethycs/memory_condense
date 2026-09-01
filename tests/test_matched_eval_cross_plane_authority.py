from __future__ import annotations

import copy
import json
from dataclasses import replace
from functools import lru_cache
from typing import Any

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools import run_locked_specialist_final_answer_v2 as answer
from tools.matched_eval.cross_plane_authority import (
    CrossPlaneAuthorityError,
    CrossPlaneAuthorityProtection,
    protect_parent_from_cross_plane_authority,
)


@lru_cache(maxsize=1)
def _sealed_rows() -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    _construction, plans = answer.load_answer_plans(answer.DEFAULT_CONSTRUCTION)
    run = json.loads(
        (answer.DEFAULT_OUTPUT / answer.RUN_NAME).read_text(encoding="utf-8")
    )
    return (
        {row["ordinal"]: row for row in plans},
        {row["ordinal"]: row for row in run["questions"]},
    )


def _guard(
    ordinal: int,
    *,
    plan: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    parent_prediction: str | None = None,
    replacement_prediction: str | None = None,
    replacement_handles: list[str] | None = None,
) -> CrossPlaneAuthorityProtection | None:
    plans, results = _sealed_rows()
    selected_plan = plans[ordinal] if plan is None else plan
    selected_result = results[ordinal] if result is None else result
    parent = (
        selected_plan["parent_prediction"]
        if parent_prediction is None
        else parent_prediction
    )
    return protect_parent_from_cross_plane_authority(
        dated_question=selected_plan["provider_input"]["dated_question"],
        parent_prediction=parent,
        replacement_prediction=(
            selected_result["prediction"]
            if replacement_prediction is None
            else replacement_prediction
        ),
        replacement_used_handle_ids=(
            selected_result["used_handle_ids"]
            if replacement_handles is None
            else replacement_handles
        ),
        replacement_proof_kind=selected_result["proof_kind"],
        provider_input=selected_plan["provider_input"],
        validation_contract=selected_plan["validation_contract"],
        allowed_handle_ids=selected_plan["allowed_handle_ids"],
        answer_plan_receipt_sha256=selected_plan["answer_plan_receipt_sha256"],
        base_scope_receipt_sha256=selected_result[
            "specialist_scope_receipt_sha256"
        ],
        source_completion_sha256=selected_result["completion_receipt_sha256"],
    )


def test_real_exact_total_current_total_and_duration_are_protected() -> None:
    expected = {
        12: ("exact_declared_total", ("H001",)),
        15: ("explicit_duration", ("H001",)),
        44: ("exact_current_total", ("H001",)),
        90: ("exact_current_total", ("H002",)),
    }

    for ordinal, (basis, handles) in expected.items():
        result = _guard(ordinal)
        assert result is not None
        assert result.basis == basis
        assert result.parent_support_handle_ids == handles
        assert result.provider_calls == 0
        assert result.retained_transformer_token_state_bytes == 0
        assert result.proof["provider_calls"] == 0
        assert result.proof["target_terms"]


def test_real_bounded_cardinality_protects_only_a_contradicted_lower_bound() -> None:
    result = _guard(76)
    assert result is not None
    assert result.basis == "bounded_cardinality_lower_bound"
    assert result.parent_support_handle_ids == ("H001", "H002", "H003", "H004")
    assert result.proof["lower_bound"] == 5.0
    assert result.proof["frontier"] == {
        "closed": False,
        "mode": "bounded",
        "truncated": True,
    }

    plans, _results = _sealed_rows()
    lower_plan = copy.deepcopy(plans[76])
    lower_parent = "At least 5 babies."
    lower_plan["parent_prediction"] = lower_parent
    lower_plan["provider_input"]["protected_parent_fallback"] = {
        "label": "fallback_not_evidence",
        "prediction": lower_parent,
        "prediction_sha256": quote_sha256(lower_parent),
    }
    lower = _guard(76, plan=lower_plan, parent_prediction=lower_parent)
    not_contradicted = _guard(76, replacement_prediction="6 babies.")

    assert lower is not None
    assert lower.proof["parent_numeric"]["qualifier"] == "lower_bound"
    assert not_contradicted is None


def test_guard_abstains_when_specialist_may_have_added_valid_operands() -> None:
    # 31 is a valid 50 + 20 -> 70 specialist improvement.  The parent's 50 is
    # one exact local operand, not a total.  34 has a local lower bound of at
    # least three services, which does not contradict a proposed total of four.
    # 40's two sealed acquisitions do not directly support its parent total of
    # three.  Currency/latest-state row 79 is outside this count/duration guard.
    for ordinal in (31, 34, 40, 61, 79):
        assert _guard(ordinal) is None


def test_provenance_must_be_strictly_map_over_direct_pointer() -> None:
    plans, _results = _sealed_rows()

    weak_parent = copy.deepcopy(plans[44])
    parent_inventory = weak_parent["provider_input"]["typed_evidence"]["handles"]
    parent_row = next(row for row in parent_inventory if row["handle_id"] == "H001")
    parent_row["origin"] = "direct_pointer"
    parent_row["provenance_grade"] = "direct_pointer"

    strong_replacement = _guard(44, replacement_handles=["H001"])

    assert _guard(44, plan=weak_parent) is None
    assert strong_replacement is None


def test_target_unit_and_current_value_ambiguity_fail_closed() -> None:
    plans, _results = _sealed_rows()

    wrong_target = copy.deepcopy(plans[44])
    parent_item = next(
        row
        for row in wrong_target["provider_input"]["typed_evidence"]["items"]
        if row["handle_ids"] == ["H001"]
    )
    parent_item["summary"] = "4 coffee makers; current count."
    parent_contract = wrong_target["validation_contract"]["by_handle"]["H001"]
    parent_contract["answer_anchor_terms"] = ["4", "coffee", "maker", "current"]
    parent_contract["semantic_rows"][0]["summary_terms"] = [
        "4",
        "coffee",
        "maker",
        "current",
    ]

    wrong_unit = copy.deepcopy(plans[15])
    duration_item = next(
        row
        for row in wrong_unit["provider_input"]["typed_evidence"]["items"]
        if row["handle_ids"] == ["H001"]
    )
    duration_item["unit"] = "week"
    duration_contract = wrong_unit["validation_contract"]["by_handle"]["H001"]
    duration_contract["numeric_value_rows"][0]["unit"] = "week"
    duration_contract["semantic_rows"][0]["unit"] = "week"

    ambiguous = copy.deepcopy(plans[44])
    typed = ambiguous["provider_input"]["typed_evidence"]
    source_inventory = next(
        row for row in typed["handles"] if row["handle_id"] == "H001"
    )
    conflicting_inventory = copy.deepcopy(source_inventory)
    conflicting_inventory.update({"group_handle": "G099", "handle_id": "H099"})
    typed["handles"].append(conflicting_inventory)
    source_item = next(
        row for row in typed["items"] if row["handle_ids"] == ["H001"]
    )
    conflicting_item = copy.deepcopy(source_item)
    conflicting_item.update(
        {
            "handle_ids": ["H099"],
            "numeric_value": 5.0,
            "summary": "5 bikes; explicit current count.",
        }
    )
    typed["items"].append(conflicting_item)
    conflicting_contract = copy.deepcopy(
        ambiguous["validation_contract"]["by_handle"]["H001"]
    )
    conflicting_contract["answer_anchor_terms"] = ["5", "bike", "current"]
    conflicting_contract["numeric_value_rows"][0]["numeric_value"] = 5.0
    conflicting_contract["semantic_rows"][0]["numeric_value"] = 5.0
    conflicting_contract["semantic_rows"][0]["summary_terms"] = [
        "5",
        "bike",
        "current",
    ]
    ambiguous["validation_contract"]["by_handle"]["H099"] = conflicting_contract
    ambiguous["allowed_handle_ids"].append("H099")

    assert _guard(44, plan=wrong_target) is None
    assert _guard(15, plan=wrong_unit) is None
    assert _guard(44, plan=ambiguous) is None


def test_receipt_is_inspectable_gold_blind_and_tamper_evident() -> None:
    result = _guard(44)
    assert result is not None
    projection = result.projection()
    encoded = json.dumps(projection, sort_keys=True)

    assert result.proof_receipt_sha256
    assert result.receipt_sha256
    assert result.proof["parent_evidence"][0]["handle_id"] == "H001"
    assert result.proof["replacement_evidence"][0]["origin"] == "direct_pointer"
    assert "question_id" not in encoded
    assert "reference" not in encoded
    assert "gold" not in encoded
    assert "ordinal" not in encoded
    with pytest.raises(CrossPlaneAuthorityError, match="authority proof changed"):
        replace(result, proof_receipt_sha256="0" * 64)
