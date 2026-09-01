from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.numeric_evidence_reconciler import (
    NumericEvidenceReconcilerError,
    ReconciliationMode,
    ReconciliationStatus,
    reconcile_sealed_numeric_evidence,
)
from tools.matched_eval.typed_memory_final_arm import PROMPT_ROW_FORMAT
from tools.matched_eval.typed_operator_adapter import COMPACT_FINAL_PROVIDER_FORMAT


def _slot(label: str, *, kind: str = "operand") -> dict[str, object]:
    return {
        "kind": kind,
        "label": label,
        "match_terms": [label.casefold()],
        "minimum_match_term_count": 1,
        "relation_constraint": None,
        "requires_numeric": True,
        "slot_id": identity_sha256({"slot": label}),
    }


def _operator(
    slots: list[dict[str, object]],
    *,
    comparison_mode: str = "none",
    temporal_mode: str = "none",
    temporal_window_days: int | None = None,
    query_timestamp: str | None = None,
) -> dict[str, object]:
    return {
        "absence_decision_requires_closed_frontier": False,
        "answer_shape": "boolean" if comparison_mode == "boolean_greater" else "number",
        "cardinality": None,
        "comparison_mode": comparison_mode,
        "include_proposed": False,
        "operation": "reconcile sealed numeric evidence",
        "ordering": None,
        "personalization_required": False,
        "query_timestamp": query_timestamp,
        "required_evidence_role": "explicit numeric fact",
        "required_slots": slots,
        "requires_all_slots": True,
        "requires_complete_frontier": False,
        "specificity_required": False,
        "style": "numeric_reduce",
        "temporal_mode": temporal_mode,
        "temporal_window_days": temporal_window_days,
    }


def _handle(handle_id: str, group_handle: str) -> dict[str, str]:
    return {
        "group_handle": group_handle,
        "handle_id": handle_id,
        "origin": "map",
        "provenance_grade": "exact_citation",
    }


def _item(
    handle_id: str,
    slot_ids: list[str],
    *,
    value: float | int,
    unit: str,
    summary: str,
    role: str = "operand",
    status: str = "completed",
    relation: str | None = None,
    date: str | None = None,
    authority: str = "explicit",
    qualifier: str = "exact",
) -> dict[str, object]:
    value_row: dict[str, object] = {
        "content_coherence": "match",
        "handle_ids": [handle_id],
        "included": True,
        "kind": "operand",
        "numeric_qualifier": qualifier,
        "numeric_role": role,
        "numeric_value": value,
        "status": status,
        "summary": summary,
        "supported_slot_ids": slot_ids,
        "unit": unit,
        "value_authority": authority,
    }
    if relation is not None:
        value_row["relation"] = relation
    if date is not None:
        value_row["date"] = date
    return value_row


def _provider(
    question: str,
    *,
    handles: list[dict[str, str]],
    items: list[dict[str, object]],
    operator: dict[str, object],
) -> dict[str, object]:
    return {
        "dated_question": question,
        "format": PROMPT_ROW_FORMAT,
        "typed_evidence": {
            "conflict_policy": "quarantine",
            "format": COMPACT_FINAL_PROVIDER_FORMAT,
            "frontier": {
                "available_handle_ids": [row["handle_id"] for row in handles],
                "closed": False,
                "mode": "bounded",
                "omitted_handle_ids": [],
                "rejected_item_count": 0,
                "represented_handle_ids": [row["handle_id"] for row in handles],
                "truncated": False,
                "unresolved_slot_ids": [],
            },
            "handles": handles,
            "items": items,
            "operator_spec": operator,
        },
    }


def _reconcile(provider: dict[str, object]):
    return reconcile_sealed_numeric_evidence(
        provider,
        sealed_provider_input_sha256=identity_sha256(provider),
    )


def test_direct_current_total_deduplicates_corroboration() -> None:
    slot = _slot("widgets")
    slot_id = str(slot["slot_id"])
    provider = _provider(
        "How many widgets are there now?",
        handles=[_handle("H001", "G001"), _handle("H002", "G002")],
        items=[
            _item(
                "H001",
                [slot_id],
                value=12,
                unit="widgets",
                summary="The current total is twelve widgets.",
                role="end",
                status="current",
                relation="event_key=current_widget_total",
            ),
            _item(
                "H002",
                [slot_id],
                value=12,
                unit="widgets",
                summary="A second source corroborates the current widget total.",
                role="end",
                status="current",
                relation="corroborates=current_widget_total",
            ),
        ],
        operator=_operator([slot]),
    )

    receipt = _reconcile(provider)

    assert receipt.status is ReconciliationStatus.SUPPORTED
    assert receipt.mode is ReconciliationMode.DIRECT_TOTAL
    assert receipt.numeric_result == 12
    assert receipt.used_handle_ids == ("H001", "H002")
    assert receipt.deduplicated_item_count == 1
    assert len(receipt.contributions) == 1


def test_direct_current_total_fails_on_conflicting_corroboration() -> None:
    slot = _slot("widgets")
    slot_id = str(slot["slot_id"])
    provider = _provider(
        "How many widgets are there now?",
        handles=[_handle("H001", "G001"), _handle("H002", "G002")],
        items=[
            _item(
                "H001", [slot_id], value=11, unit="widgets",
                summary="The current total is eleven.", role="end",
                status="current", relation="event_key=current_total",
            ),
            _item(
                "H002", [slot_id], value=12, unit="widgets",
                summary="The current total is twelve.", role="end",
                status="current", relation="corroborates=current_total",
            ),
        ],
        operator=_operator([slot]),
    )

    receipt = _reconcile(provider)

    assert receipt.status is ReconciliationStatus.CONFLICTED
    assert receipt.reason == "conflicting_current_or_end_totals"
    assert receipt.contributions == ()


def test_cardinality_sums_exact_item_operands_and_rejects_event_mix() -> None:
    slot = _slot("pieces")
    slot_id = str(slot["slot_id"])
    base = _provider(
        "How many total pieces are in the two sets?",
        handles=[_handle("H001", "G001"), _handle("H002", "G002")],
        items=[
            _item(
                "H001", [slot_id], value=2, unit="pieces",
                summary="The first set contains two pieces.",
                relation="event_key=first_set",
            ),
            _item(
                "H002", [slot_id], value=3, unit="items",
                summary="The second set contains three items.",
                relation="event_key=second_set",
            ),
        ],
        operator=_operator([slot]),
    )

    receipt = _reconcile(base)
    assert receipt.status is ReconciliationStatus.SUPPORTED
    assert receipt.mode is ReconciliationMode.CARDINALITY_SUM
    assert receipt.numeric_result == 5
    assert receipt.unit == "piece"

    mixed = copy.deepcopy(base)
    mixed["typed_evidence"]["items"][1]["unit"] = "events"  # type: ignore[index]
    failed = _reconcile(mixed)
    assert failed.status is ReconciliationStatus.INSUFFICIENT
    assert failed.reason == "item_event_unit_conflict"


def test_plural_quantity_uses_explicit_numeric_value_not_lexical_digits() -> None:
    slot = _slot("children")
    slot_id = str(slot["slot_id"])
    provider = _provider(
        "How many children were welcomed?",
        handles=[_handle("H001", "G001")],
        items=[
            _item(
                "H001", [slot_id], value=2, unit="children",
                summary="The family welcomed twins.",
                relation="event_key=birth",
            )
        ],
        operator=_operator([slot]),
    )

    receipt = _reconcile(provider)
    assert receipt.numeric_result == 2
    assert receipt.unit == "child"

    derived = copy.deepcopy(provider)
    derived["typed_evidence"]["items"][0]["value_authority"] = "derived"  # type: ignore[index]
    failed = _reconcile(derived)
    assert failed.status is ReconciliationStatus.INSUFFICIENT
    assert failed.reason == "non_explicit_numeric_value"


def test_non_temporal_cardinality_does_not_confuse_unknown_state_with_unsealed_input() -> None:
    slot = _slot("widgets")
    provider = _provider(
        "How many widgets are in the box?",
        handles=[_handle("H001", "G001")],
        items=[
            _item(
                "H001", [str(slot["slot_id"])], value=3, unit="widgets",
                summary="The box contains three widgets.", status="unknown",
                relation="event_key=box_contents",
            )
        ],
        operator=_operator([slot]),
    )

    receipt = _reconcile(provider)

    assert receipt.status is ReconciliationStatus.SUPPORTED
    assert receipt.numeric_result == 3


def test_recurring_base_adds_distinct_events_and_deduplicates_corroboration() -> None:
    slot = _slot("weekly events")
    slot_id = str(slot["slot_id"])
    provider = _provider(
        "How many events per week are now scheduled?",
        handles=[
            _handle("H001", "G001"),
            _handle("H002", "G002"),
            _handle("H003", "G003"),
            _handle("H004", "G004"),
        ],
        items=[
            _item(
                "H001", [slot_id], value=2, unit="events/week",
                summary="The base schedule has two events per week.",
                role="baseline", relation="event_key=base;recurrence=base",
            ),
            _item(
                "H002", [slot_id], value=1, unit="event",
                summary="A club event was added to every week.", role="delta",
                relation="event_key=club;frequency_addition=true",
            ),
            _item(
                "H003", [slot_id], value=1, unit="events",
                summary="The weekly club event is corroborated.", role="delta",
                relation="corroborates=club;frequency_addition=true",
            ),
            _item(
                "H004", [slot_id], value=1, unit="event",
                summary="A gym event was separately added to every week.", role="delta",
                relation="event_key=gym;frequency_addition=true",
            ),
        ],
        operator=_operator([slot]),
    )

    receipt = _reconcile(provider)

    assert receipt.status is ReconciliationStatus.SUPPORTED
    assert receipt.mode is ReconciliationMode.RECURRING_PLUS_ADDITIONS
    assert receipt.numeric_result == 4
    assert receipt.unit == "event/week"
    assert receipt.deduplicated_item_count == 1
    assert receipt.used_handle_ids == ("H001", "H002", "H003", "H004")


def test_recurring_addition_requires_explicit_recurrence_and_consistent_claim() -> None:
    slot = _slot("weekly events")
    slot_id = str(slot["slot_id"])
    provider = _provider(
        "How many events per week are scheduled?",
        handles=[_handle("H001", "G001"), _handle("H002", "G002")],
        items=[
            _item(
                "H001", [slot_id], value=2, unit="events/week",
                summary="Two events recur every week.", role="baseline",
                relation="event_key=base;recurrence=base",
            ),
            _item(
                "H002", [slot_id], value=1, unit="event",
                summary="One separate event occurred.", role="delta",
                relation="event_key=one_off",
            ),
        ],
        operator=_operator([slot]),
    )

    failed = _reconcile(provider)
    assert failed.status is ReconciliationStatus.CONFLICTED
    assert failed.reason == "ambiguous_one_off_frequency_addition"

    conflicting = copy.deepcopy(provider)
    conflicting["typed_evidence"]["items"][1]["relation"] = (  # type: ignore[index]
        "corroborates=base;frequency_addition=true"
    )
    conflicting["typed_evidence"]["items"][1]["numeric_role"] = "baseline"  # type: ignore[index]
    conflicting["typed_evidence"]["items"][1]["unit"] = "events/week"  # type: ignore[index]
    conflict_receipt = _reconcile(conflicting)
    assert conflict_receipt.status is ReconciliationStatus.CONFLICTED
    assert conflict_receipt.reason == "recurring_base_conflict"


@pytest.mark.parametrize(
    ("comparison_mode", "expected_boolean"),
    [("difference", None), ("boolean_greater", True)],
)
def test_comparison_uses_two_sealed_numeric_sides(
    comparison_mode: str, expected_boolean: bool | None
) -> None:
    left = _slot("left total", kind="comparison_side")
    right = _slot("right total", kind="comparison_side")
    provider = _provider(
        "Is the left total greater, and by how much?",
        handles=[_handle("H001", "G001"), _handle("H002", "G002")],
        items=[
            _item(
                "H001", [str(left["slot_id"])], value=30, unit="dollars",
                summary="The sealed left total is thirty dollars.",
                relation="event_key=left_total",
            ),
            _item(
                "H002", [str(right["slot_id"])], value=20, unit="dollars",
                summary="The sealed right total is twenty dollars.",
                relation="event_key=right_total",
            ),
        ],
        operator=_operator([left, right], comparison_mode=comparison_mode),
    )

    receipt = _reconcile(provider)

    assert receipt.status is ReconciliationStatus.SUPPORTED
    assert receipt.mode is ReconciliationMode.COMPARISON
    assert receipt.numeric_result == 10
    assert receipt.unit == "$"
    assert receipt.comparison_relation == "left_greater"
    assert receipt.boolean_result is expected_boolean


def test_comparison_rejects_cross_side_binding_and_unit_mismatch() -> None:
    left = _slot("left", kind="comparison_side")
    right = _slot("right", kind="comparison_side")
    left_id, right_id = str(left["slot_id"]), str(right["slot_id"])
    provider = _provider(
        "What is the difference between the two totals?",
        handles=[_handle("H001", "G001"), _handle("H002", "G002")],
        items=[
            _item(
                "H001", [left_id], value=3, unit="items",
                summary="The left side has three items.", relation="event_key=left",
            ),
            _item(
                "H002", [right_id], value=2, unit="items",
                summary="The right side has two items.", relation="event_key=right",
            ),
        ],
        operator=_operator([left, right], comparison_mode="difference"),
    )

    both = copy.deepcopy(provider)
    both["typed_evidence"]["items"][0]["supported_slot_ids"] = [  # type: ignore[index]
        left_id, right_id
    ]
    both_receipt = _reconcile(both)
    assert both_receipt.status is ReconciliationStatus.CONFLICTED
    assert both_receipt.reason == "comparison_item_binds_zero_or_multiple_sides"

    units = copy.deepcopy(provider)
    units["typed_evidence"]["items"][1]["unit"] = "events"  # type: ignore[index]
    unit_receipt = _reconcile(units)
    assert unit_receipt.status is ReconciliationStatus.CONFLICTED
    assert unit_receipt.reason == "comparison_side_unit_conflict"


def test_generic_slotless_h700_operand_is_rejected_after_sealed_selection() -> None:
    slot = _slot("widgets")
    slot_id = str(slot["slot_id"])
    provider = _provider(
        "How many widgets are there?",
        handles=[_handle("H700", "G700"), _handle("H001", "G001")],
        items=[
            _item(
                "H700", [], value=700, unit="widgets",
                summary="A lexical scanner exposed a generic number.",
                relation="event_key=lexical_number",
            ),
            _item(
                "H001", [slot_id], value=4, unit="widgets",
                summary="The sealed widget fact gives four.",
                relation="event_key=widget_total",
            ),
        ],
        operator=_operator([slot]),
    )

    receipt = _reconcile(provider)

    assert receipt.numeric_result == 4
    assert receipt.used_handle_ids == ("H001",)
    assert [row.reason for row in receipt.rejected_operands] == [
        "generic_lexical_h700_operand"
    ]

    only_h700 = copy.deepcopy(provider)
    only_h700["typed_evidence"]["handles"] = [  # type: ignore[index]
        only_h700["typed_evidence"]["handles"][0]  # type: ignore[index]
    ]
    only_h700["typed_evidence"]["items"] = [  # type: ignore[index]
        only_h700["typed_evidence"]["items"][0]  # type: ignore[index]
    ]
    failed = _reconcile(only_h700)
    assert failed.status is ReconciliationStatus.INSUFFICIENT
    assert failed.reason == "generic_lexical_h700_operand"


def test_temporal_eligibility_requires_a_sealed_unambiguous_basis() -> None:
    slot = _slot("events")
    slot_id = str(slot["slot_id"])
    operator = _operator(
        [slot],
        temporal_mode="relative_select",
        temporal_window_days=30,
        query_timestamp="2026-08-28",
    )
    provider = _provider(
        "[Question asked at 2026-08-28] How many events were in the last 30 days?",
        handles=[_handle("H001", "G001")],
        items=[
            _item(
                "H001", [slot_id], value=2, unit="events",
                summary="Two events; temporal eligibility is unspecified.",
                date="2026-08-20", relation="event_key=recent_events",
            )
        ],
        operator=operator,
    )

    ambiguous = _reconcile(provider)
    assert ambiguous.status is ReconciliationStatus.INSUFFICIENT
    assert ambiguous.reason == "ambiguous_temporal_eligibility"

    in_window = copy.deepcopy(provider)
    in_window["typed_evidence"]["items"][0]["summary"] = (  # type: ignore[index]
        "Two completed events happened on the sealed date."
    )
    accepted = _reconcile(in_window)
    assert accepted.status is ReconciliationStatus.SUPPORTED
    assert accepted.numeric_result == 2
    assert accepted.contributions[0].temporal_bases == ("dated_closed_window",)

    outside = copy.deepcopy(in_window)
    outside["typed_evidence"]["items"][0]["date"] = "2026-06-01"  # type: ignore[index]
    rejected = _reconcile(outside)
    assert rejected.status is ReconciliationStatus.INSUFFICIENT
    assert rejected.reason == "outside_temporal_window"


@pytest.mark.parametrize(
    "forbidden_key",
    ["ordinal", "question_id", "reference", "gold_answer", "question_ordinal"],
)
def test_input_firewall_rejects_identity_and_gold_fields(forbidden_key: str) -> None:
    slot = _slot("widgets")
    provider = _provider(
        "How many widgets are there?",
        handles=[_handle("H001", "G001")],
        items=[
            _item(
                "H001", [str(slot["slot_id"])], value=1, unit="widget",
                summary="There is one widget.", relation="event_key=widget",
            )
        ],
        operator=_operator([slot]),
    )
    provider[forbidden_key] = "forbidden"

    with pytest.raises((NumericEvidenceReconcilerError, ValueError)):
        _reconcile(provider)


def test_seal_and_receipt_are_tamper_evident_and_provider_free() -> None:
    slot = _slot("widgets")
    provider = _provider(
        "How many widgets are there?",
        handles=[_handle("H001", "G001")],
        items=[
            _item(
                "H001", [str(slot["slot_id"])], value=1, unit="widget",
                summary="There is one widget.", relation="event_key=widget",
            )
        ],
        operator=_operator([slot]),
    )
    sealed_sha = identity_sha256(provider)
    changed = copy.deepcopy(provider)
    changed["typed_evidence"]["items"][0]["numeric_value"] = 9  # type: ignore[index]

    with pytest.raises(NumericEvidenceReconcilerError, match="SHA-256 mismatch"):
        reconcile_sealed_numeric_evidence(
            changed,
            sealed_provider_input_sha256=sealed_sha,
        )

    receipt = _reconcile(provider)
    projection = receipt.projection()
    assert projection["provider_prompt_count"] == 0
    assert projection["retained_transformer_token_state_bytes"] == 0
    assert projection["gold_loaded"] is False
    assert "dated_question" not in projection
    assert "question_id" not in repr(projection).casefold()
    with pytest.raises(NumericEvidenceReconcilerError, match="receipt changed"):
        replace(receipt, numeric_result=99)
