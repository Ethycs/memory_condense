from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tools.matched_eval.contracts import MatchedEvalContractError, identity_sha256
from tools.matched_eval.operator_first_numeric_policy import (
    NumericPolicyMode,
    build_relevant_numeric_frontier,
    execute_operator_first_numeric_policy,
)
from tools.matched_eval.typed_operator_executor import ExecutionStatus


def _item(
    handle: str,
    summary: str,
    *,
    relation: str = "authored_by_user;date_basis=source_created_at",
    status: str = "completed",
    date: str = "2023-05-30T00:00:00-07:00",
    numeric_value: float | None = None,
    unit: str | None = None,
    supported_slot_ids: list[str] | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "content_coherence": "match",
        "date": date,
        "handle_ids": [handle],
        "included": True,
        "kind": "operand" if numeric_value is not None else "direct",
        "relation": relation,
        "status": status,
        "summary": summary,
        "supported_slot_ids": supported_slot_ids or [],
        "value_authority": "explicit",
    }
    if numeric_value is not None:
        value.update(
            {
                "numeric_qualifier": "exact",
                "numeric_role": "operand",
                "numeric_value": numeric_value,
            }
        )
    if unit is not None:
        value["unit"] = unit
    return value


def _provider(
    question: str,
    items: list[dict[str, object]],
    *,
    comparison_mode: str = "none",
    include_proposed: bool = False,
    required_slots: list[dict[str, object]] | None = None,
    global_frontier_closed: bool = False,
    parent_prediction: str = "irrelevant parent",
) -> dict[str, object]:
    handles = tuple(
        dict.fromkeys(
            handle
            for item in items
            for handle in item["handle_ids"]  # type: ignore[index]
        )
    )
    return {
        "dated_question": question,
        # The policy must bind neither this field nor its value.
        "protected_parent_fallback": {
            "label": "fallback_not_evidence",
            "prediction": parent_prediction,
        },
        "typed_evidence": {
            "conflict_policy": "quarantine",
            "format": "test-compact-typed-evidence-v1",
            "frontier": {
                "available_handle_ids": list(handles),
                "closed": global_frontier_closed,
                "mode": "exhaustive" if global_frontier_closed else "open",
                "omitted_handle_ids": [],
                "represented_handle_ids": list(handles),
                "truncated": not global_frontier_closed,
                "unresolved_slot_ids": [],
            },
            "handles": [
                {
                    "group_handle": f"G{index:03d}",
                    "handle_id": handle,
                    "origin": "map",
                    "provenance_grade": "exact_citation",
                }
                for index, handle in enumerate(handles, start=1)
            ],
            "items": items,
            "operator_spec": {
                "answer_shape": (
                    "boolean" if comparison_mode == "boolean_greater" else "number"
                ),
                "comparison_mode": comparison_mode,
                "include_proposed": include_proposed,
                "operation": "count_or_aggregate",
                "query_timestamp": question.split("]", 1)[0].removeprefix(
                    "[Question asked at "
                ),
                "required_slots": required_slots or [],
                "requires_complete_frontier": True,
                "style": "numeric_reduce",
                "temporal_window_days": None,
            },
        },
    }


def _frontier(provider: dict[str, object]):
    return build_relevant_numeric_frontier(
        provider,
        candidate_population_receipt_sha256=identity_sha256(
            {
                "kind": "exhaustive-operator-relevant-fixture-scan",
                "policy_input": provider["dated_question"],
            }
        ),
    )


def test_fixed_scalar_comparison_uses_user_operands_and_ignores_parent_and_assistant() -> None:
    question = (
        "[Question asked at 2023/05/30 (Tue) 21:33]\n"
        "Did I receive a higher percentage discount on my first order from "
        "HelloFresh, compared to my first UberEats order?"
    )
    slots = [
        {
            "kind": "comparison_side",
            "label": "HelloFresh",
            "slot_id": "side_hello_fresh",
        },
        {
            "kind": "comparison_side",
            "label": "UberEats",
            "slot_id": "side_uber_eats",
        },
    ]
    provider = _provider(
        question,
        [
            _item(
                "H001",
                "I got a 40% discount on my first HelloFresh order.",
                numeric_value=40,
                unit="%",
                supported_slot_ids=["side_hello_fresh"],
            ),
            _item(
                "H002",
                "Last week I got 20% off my UberEats order.",
                numeric_value=20,
                unit="%",
                supported_slot_ids=["side_uber_eats"],
            ),
            _item(
                "H003",
                "You told me the user's UberEats discount was 20%.",
                relation="authored_by_assistant;date_basis=source_created_at",
                numeric_value=20,
                unit="%",
                supported_slot_ids=["side_uber_eats"],
            ),
        ],
        comparison_mode="boolean_greater",
        required_slots=slots,
        parent_prediction="I cannot determine that.",
    )

    result = execute_operator_first_numeric_policy(provider)
    changed_parent = execute_operator_first_numeric_policy(
        _provider(
            question,
            provider["typed_evidence"]["items"],  # type: ignore[index]
            comparison_mode="boolean_greater",
            required_slots=slots,
            parent_prediction="No.",
        )
    )

    assert result.status is ExecutionStatus.SUPPORTED
    assert result.mode is NumericPolicyMode.FIXED_SCALAR_COMPARISON
    assert result.prediction == "Yes"
    assert result.numeric_result == 20
    assert result.used_handle_ids == ("H001", "H002")
    assert result.policy_input_sha256 == changed_parent.policy_input_sha256
    assert result.receipt_sha256 == changed_parent.receipt_sha256
    assert any(
        row.reason == "assistant_not_autobiographical_evidence"
        for row in result.exclusions
    )


def test_include_proposed_is_question_controlled_and_distinct_bikes_deduplicate() -> None:
    question = (
        "[Question asked at 2023/03/20 (Mon) 23:57]\n"
        "How many bikes did I service or plan to service in March?"
    )
    items = [
        _item(
            "H001",
            "I got my road bike serviced on March 10th.",
            date="2023-03-10",
            relation="authored_by_user;date_basis=textual_event_date",
        ),
        _item(
            "H002",
            "I was glad my road bike was serviced on March 10th.",
            date="2023-03-10",
            relation="authored_by_user;date_basis=textual_event_date",
        ),
        _item(
            "H003",
            "I need to replace the tire on my commuter bike this month, before April comes.",
            relation=(
                "authored_by_user;date_basis=source_created_at;"
                "event_action=service"
            ),
            status="proposed",
            date="2023-03-20T00:32:00-07:00",
        ),
    ]
    provider = _provider(question, items, include_proposed=True)
    result = execute_operator_first_numeric_policy(
        provider, relevant_frontier=_frontier(provider)
    )

    completed_only_question = (
        "[Question asked at 2023/03/20 (Mon) 23:57]\n"
        "How many bikes did I service in March?"
    )
    completed_only = _provider(completed_only_question, items)
    completed_result = execute_operator_first_numeric_policy(
        completed_only, relevant_frontier=_frontier(completed_only)
    )

    assert result.prediction == "2"
    assert result.used_handle_ids == ("H001", "H003")
    assert completed_result.prediction == "1"
    assert completed_result.used_handle_ids == ("H001",)
    assert any(row.reason == "proposed_not_requested" for row in completed_result.exclusions)


def test_temporal_entity_count_filters_old_and_unrequested_proposed_jewelry() -> None:
    question = (
        "[Question asked at 2023/05/30 (Tue) 15:43]\n"
        "How many pieces of jewelry did I acquire in the last two months?"
    )
    provider = _provider(
        question,
        [
            _item(
                "H001",
                "I got my engagement ring a month ago.",
                relation="authored_by_user;event_action=acquire;date_basis=source_created_at",
            ),
            _item(
                "H002",
                "I got a stunning pair of emerald earrings last weekend.",
                relation="authored_by_user;event_action=acquire;date_basis=source_created_at",
            ),
            _item(
                "H003",
                "I got a silver necklace on the 15th of last month.",
                relation="authored_by_user;event_action=acquire;date_basis=source_created_at",
            ),
            _item(
                "H004",
                "I acquired a gold bracelet on January 10th.",
                relation="authored_by_user;event_action=acquire;date_basis=textual_event_date",
                date="2023-01-10",
            ),
            _item(
                "H005",
                "I plan to acquire a diamond brooch next week.",
                relation="authored_by_user;event_action=acquire;date_basis=relative_event_time",
                status="proposed",
            ),
        ],
    )

    result = execute_operator_first_numeric_policy(
        provider, relevant_frontier=_frontier(provider)
    )

    assert result.status is ExecutionStatus.SUPPORTED
    assert result.prediction == "3"
    assert result.used_handle_ids == ("H001", "H002", "H003")
    assert {row.entity_key for row in result.candidate_atoms} == {
        "engagement_ring",
        "emerald_earring",
        "silver_necklace",
    }


def test_coordinated_purchase_expands_two_plants_before_deduplication() -> None:
    question = (
        "[Question asked at 2023/05/30 (Tue) 21:51]\n"
        "How many plants did I acquire in the last month?"
    )
    provider = _provider(
        question,
        [
            _item(
                "H001",
                "I bought the peace lily and a succulent plant two weeks ago.",
            ),
            _item(
                "H002",
                "My snake plant, which I got from my sister last month, needs repotting.",
                status="unknown",
            ),
        ],
    )

    result = execute_operator_first_numeric_policy(
        provider, relevant_frontier=_frontier(provider)
    )

    assert result.prediction == "3"
    assert result.used_handle_ids == ("H001", "H002")
    assert [row.entity_key for row in result.candidate_atoms].count("peace_lily") == 1
    assert [row.entity_key for row in result.candidate_atoms].count("succulent") == 1
    assert len([row for row in result.candidate_atoms if row.handle_ids == ("H001",)]) == 2


def test_explicit_month_uses_event_text_not_source_timestamp() -> None:
    question = (
        "[Question asked at 2023/03/03 (Fri) 23:25]\n"
        "How many different museums or galleries did I visit in the month of February?"
    )
    provider = _provider(
        question,
        [
            _item(
                "H001",
                "I took my niece to the Natural History Museum on 2/8.",
                date="2023-03-03T07:10:00-08:00",
            ),
            _item(
                "H002",
                "I visited The Art Cube on 2/15.",
                date="2023-03-03T03:38:00-08:00",
            ),
            _item(
                "H003",
                "I recently visited an exhibition at the Local History Museum.",
                date="2023-02-15T01:40:00-08:00",
            ),
        ],
    )

    result = execute_operator_first_numeric_policy(
        provider, relevant_frontier=_frontier(provider)
    )

    assert result.prediction == "2"
    assert result.used_handle_ids == ("H001", "H002")
    assert any(
        row.handle_ids == ("H003",) and row.reason == "event_time_not_in_scope"
        for row in result.exclusions
    )


def test_action_obligations_count_return_and_replacement_pickup_separately() -> None:
    question = (
        "[Question asked at 2023/02/15 (Wed) 23:50]\n"
        "How many items of clothing do I need to pick up or return from a store?"
    )
    provider = _provider(
        question,
        [
            _item(
                "H001",
                "I still need to pick up my dry cleaning for the navy blue blazer.",
                status="unknown",
            ),
            _item(
                "H002",
                "I exchanged a pair of boots and still need to pick up the new pair.",
                status="unknown",
            ),
            _item(
                "H003",
                "I need to return some boots because they were too small, so I exchanged "
                "them for a larger size. I haven't had a chance to pick them up yet.",
                status="unknown",
            ),
        ],
    )

    result = execute_operator_first_numeric_policy(
        provider, relevant_frontier=_frontier(provider)
    )

    assert result.mode is NumericPolicyMode.ACTION_OBLIGATION_COUNT
    assert result.prediction == "3"
    assert result.used_handle_ids == ("H001", "H002", "H003")
    assert {(row.action_key, row.entity_key) for row in result.candidate_atoms} == {
        ("pickup", "navy_blue_blazer"),
        ("pickup", "replacement_boot"),
        ("return", "original_boot"),
    }
    assert any(row.reason == "duplicate_semantic_identity" for row in result.exclusions)


def test_scoped_distinct_set_excludes_non_event_distractor_and_deduplicates() -> None:
    question = (
        "[Question asked at 2023/05/30 (Tue) 16:04]\n"
        "How many different cuisines have I learned to cook or tried out in the past few months?"
    )
    provider = _provider(
        question,
        [
            _item(
                "H001",
                "I recently learned to make a vegan lasagna in a vegan cuisine class.",
                relation="authored_by_user;event_action=learn;date_basis=relative_event_time",
            ),
            _item(
                "H002",
                "I tried a Korean bibimbap recipe last week.",
                relation="authored_by_user;event_action=try;date_basis=relative_event_time",
            ),
            _item(
                "H003",
                "I learned chicken tikka masala in an Indian cuisine class a month ago.",
                relation="authored_by_user;event_action=learn;date_basis=relative_event_time",
            ),
            _item(
                "H004",
                "I tried an Ethiopian restaurant last week.",
                relation="authored_by_user;event_action=try;date_basis=relative_event_time",
            ),
            _item(
                "H005",
                "I am researching Thai cuisine classes.",
                relation=(
                    "authored_by_user;event_action=research;"
                    "numeric_scope=out_of_scope;date_basis=source_created_at"
                ),
                status="unknown",
            ),
            _item(
                "H006",
                "I also recently learned another vegan cuisine recipe.",
                relation="authored_by_user;event_action=learn;date_basis=relative_event_time",
            ),
        ],
    )

    result = execute_operator_first_numeric_policy(
        provider, relevant_frontier=_frontier(provider)
    )

    assert result.mode is NumericPolicyMode.DISTINCT_ENTITY_COUNT
    assert result.prediction == "4"
    assert result.used_handle_ids == ("H001", "H002", "H003", "H004")
    assert any(row.reason == "outside_operator_scope" for row in result.exclusions)
    assert any(row.reason == "duplicate_semantic_identity" for row in result.exclusions)
    encoded = json.dumps(result.projection(), sort_keys=True)
    assert "irrelevant parent" not in encoded
    assert result.provider_prompt_count == 0
    assert result.retained_transformer_token_state_bytes == 0
    with pytest.raises(MatchedEvalContractError, match="decision changed"):
        replace(result, prediction="5")


def test_count_abstains_without_operator_relevant_closure() -> None:
    question = (
        "[Question asked at 2023/05/30 (Tue) 21:51]\n"
        "How many plants did I acquire in the last month?"
    )
    provider = _provider(
        question,
        [_item("H001", "I bought a peace lily two weeks ago.")],
    )

    result = execute_operator_first_numeric_policy(provider)

    assert result.status is ExecutionStatus.INSUFFICIENT
    assert result.decision == "abstain"
    assert result.prediction == ""
    assert result.reason == "relevant_candidate_frontier_not_closed"
