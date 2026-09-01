from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.numeric_evidence_reconciler import (
    NumericEvidenceReconcilerError,
    ReconciliationMode,
    ReconciliationStatus,
)
from tools.matched_eval.numeric_evidence_reconciler_v2 import (
    reconcile_sealed_numeric_evidence_v2,
)


PREFLIGHT = Path(
    "eval_results/matched_eval_100/locked-specialist-final-answer-v2/"
    "locked-specialist-final-answer-preflight-v2.json"
)
PREFLIGHT_SHA256 = (
    "61371cd58b239a07f493ea4c116908a7f72e252cb503c0a5210f30c7f66ad413"
)


def _provider(ordinal: int) -> dict[str, object]:
    raw = PREFLIGHT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PREFLIGHT_SHA256
    source = json.loads(raw)
    row = next(
        value for value in source["physical_prompt_rows"]
        if value["ordinal"] == ordinal
    )
    assert identity_sha256(row["messages"]) == row["messages_sha256"]
    return json.loads(row["messages"][1]["content"])


def _reconcile(provider: dict[str, object]):
    return reconcile_sealed_numeric_evidence_v2(
        provider,
        sealed_provider_input_sha256=identity_sha256(provider),
    )


def _item(provider: dict[str, object], handle_id: str) -> dict[str, object]:
    return next(
        row
        for row in provider["typed_evidence"]["items"]  # type: ignore[index]
        if handle_id in row["handle_ids"]
    )


def test_actual_distinct_summary_schema_projects_three_semantic_contributions() -> None:
    provider = _provider(34)
    assert "numeric_value" not in _item(provider, "H001")

    receipt = _reconcile(provider)

    assert receipt.status is ReconciliationStatus.SUPPORTED
    assert receipt.mode is ReconciliationMode.CARDINALITY_SUM
    assert receipt.numeric_result == 3
    assert receipt.unit == "food delivery service type"
    assert receipt.used_handle_ids == ("H001", "H002", "H003")
    assert len(receipt.contributions) == 3
    assert {row.numeric_value for row in receipt.contributions} == {1}
    assert len({row.semantic_key_sha256 for row in receipt.contributions}) == 3
    assert receipt.provider_prompt_count == 0


def test_distinct_summary_numeric_conflict_and_close_category_fail_closed() -> None:
    conflict = _provider(34)
    _item(conflict, "H002")["numeric_value"] = 2.0

    conflict_receipt = _reconcile(conflict)

    assert conflict_receipt.status is ReconciliationStatus.CONFLICTED
    assert conflict_receipt.reason == "projected_summary_numeric_conflict"
    assert conflict_receipt.numeric_result is None

    close = _provider(34)
    first = _item(close, "H001")
    first["summary"] = (
        "1 distinct food catering service type: Picnic Pro; positive; eligible; "
        "recent-use context (lately)."
    )
    close["typed_evidence"]["items"] = [first]  # type: ignore[index]

    close_receipt = _reconcile(close)

    assert close_receipt.status is ReconciliationStatus.CONFLICTED
    assert close_receipt.reason == "strict_distinct_summary_operands_missing"


def test_generic_h700_cannot_become_a_distinct_summary_operand() -> None:
    provider = _provider(34)
    generic = _item(provider, "H700001")
    generic.update(
        {
            "numeric_qualifier": "exact",
            "numeric_role": "operand",
            "numeric_value": 1.0,
            "summary": (
                "1 distinct food delivery service type: False Friend; positive; "
                "eligible; recent-use context (lately)."
            ),
            "value_authority": "explicit",
        }
    )
    provider["typed_evidence"]["items"] = [generic]  # type: ignore[index]

    receipt = _reconcile(provider)

    assert receipt.status is ReconciliationStatus.CONFLICTED
    assert receipt.numeric_result is None
    assert any(
        row.handle_ids == ("H700001",)
        and row.reason == "generic_lexical_h700_operand"
        for row in receipt.projection_rejections
    )


def test_actual_recurring_schema_adds_two_and_deduplicates_bodypump() -> None:
    receipt = _reconcile(_provider(87))

    assert receipt.status is ReconciliationStatus.SUPPORTED
    assert receipt.mode is ReconciliationMode.RECURRING_PLUS_ADDITIONS
    assert receipt.numeric_result == 5
    assert receipt.unit == "fitness_class/week"
    assert receipt.used_handle_ids == ("H001", "H002", "H003", "H004")
    assert len(receipt.contributions) == 3
    baseline = [row for row in receipt.contributions if row.numeric_role == "baseline"]
    additions = [row for row in receipt.contributions if row.numeric_role == "delta"]
    assert [row.numeric_value for row in baseline] == [3]
    assert [row.numeric_value for row in additions] == [1, 1]
    bodypump = next(row for row in additions if row.handle_ids == ("H003", "H004"))
    assert bodypump.corroborated_duplicate_count == 1
    assert receipt.deduplicated_item_count == 1


def test_recurring_corroboration_without_a_primary_fails_closed() -> None:
    provider = _provider(87)
    third = _item(provider, "H003")
    third["summary"] = str(third["summary"]).replace(
        "eligible weekly-class operand",
        "potentially corroborative weekly-class operand",
    )

    receipt = _reconcile(provider)

    assert receipt.status is ReconciliationStatus.CONFLICTED
    assert receipt.reason == "projected_corroboration_without_primary"
    assert receipt.numeric_result is None


def test_different_wording_deduplicates_only_with_sealed_relation_identity() -> None:
    unbound = _provider(87)
    fourth = _item(unbound, "H004")
    fourth["summary"] = str(fourth["summary"]).replace(
        "Monday schedule", "Monday evening schedule"
    )

    unbound_receipt = _reconcile(unbound)

    assert unbound_receipt.status is ReconciliationStatus.CONFLICTED
    assert unbound_receipt.reason == "projected_corroboration_without_primary"

    bound = copy.deepcopy(unbound)
    _item(bound, "H003")["relation"] = "event_key=bodypump_weekly"
    _item(bound, "H004")["relation"] = "corroborates=bodypump_weekly"

    bound_receipt = _reconcile(bound)

    assert bound_receipt.status is ReconciliationStatus.SUPPORTED
    assert bound_receipt.numeric_result == 5
    bodypump = next(
        row for row in bound_receipt.contributions
        if row.handle_ids == ("H003", "H004")
    )
    assert bodypump.corroborated_duplicate_count == 1


def test_recurring_summary_numeric_conflict_does_not_sum() -> None:
    provider = _provider(87)
    _item(provider, "H004")["numeric_value"] = 2.0

    receipt = _reconcile(provider)

    assert receipt.status is ReconciliationStatus.CONFLICTED
    assert receipt.reason == "projected_summary_numeric_conflict"
    assert receipt.used_handle_ids == ()


def test_v1_decision_is_preserved_and_v2_receipt_is_tamper_evident() -> None:
    receipt = _reconcile(_provider(44))

    assert receipt.status is ReconciliationStatus.SUPPORTED
    assert receipt.projection_rule == "sealed_typed_fields"
    assert receipt.numeric_result == receipt.base_reconciliation.numeric_result == 4
    assert receipt.used_handle_ids == receipt.base_reconciliation.used_handle_ids
    assert receipt.provider_prompt_count == 0
    assert receipt.retained_transformer_token_state_bytes == 0
    assert receipt.gold_loaded is False
    projection = receipt.projection()
    assert "dated_question" not in projection
    assert "question_id" not in repr(projection).casefold()
    with pytest.raises(NumericEvidenceReconcilerError, match="receipt changed"):
        replace(receipt, numeric_result=99)
