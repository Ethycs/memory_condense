from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from tools.matched_eval.contracts import MatchedEvalContractError
from tools.matched_eval.prompt_tick_contracts import CallBudget, LaneBudget
from tools.matched_eval.typed_lane_allocator import (
    allocate_typed_contribution_lanes,
    fill_typed_lane_surplus,
    lane_content_token_proxy,
)
from tools.matched_eval.typed_operator_adapter import (
    EvidenceHandleBinding,
    EvidenceOrigin,
    FrontierMode,
    ParsedTypedItems,
    ProvenanceGrade,
    TypedEvidenceContribution,
    parse_typed_items,
)
from tools.matched_eval.typed_operator_spec import compile_typed_operator_spec


QUESTION = (
    "[Question asked at 2023/05/30 (Tue) 16:15] "
    "Did I receive a higher percentage discount on my first order from "
    "HelloFresh, compared to my first UberEats order?"
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _contribution(
    mechanism: str,
    start: int,
    summaries: tuple[str, ...],
    *,
    groups: tuple[int, ...] | None = None,
) -> TypedEvidenceContribution:
    spec = compile_typed_operator_spec(QUESTION)
    summaries = tuple(summary.strip() for summary in summaries)
    if groups is None:
        groups = tuple(start + index for index in range(len(summaries)))
    assert len(groups) == len(summaries)
    bindings = tuple(
        EvidenceHandleBinding(
            f"H{start + index:03d}",
            EvidenceOrigin.MAP,
            ProvenanceGrade.EXACT_CITATION,
            f"G{groups[index]:03d}",
            _sha(f"artifact-{mechanism}"),
            _sha(f"parent-{mechanism}"),
            _sha(f"evidence-{mechanism}-{index}"),
            _sha(f"payload-{mechanism}-{index}"),
            _sha(summary),
            len(summary),
            _sha(f"locator-{mechanism}-{index}"),
        )
        for index, summary in enumerate(summaries)
    )
    raw = [
        {
            "handle_ids": [binding.handle_id],
            "kind": "operand",
            "summary": summary,
            "numeric_value": 40 if "HelloFresh" in summary else 20,
            "numeric_role": "operand",
            "unit": "%",
        }
        for binding, summary in zip(bindings, summaries, strict=True)
    ]
    parsed = parse_typed_items(raw, operator_spec=spec, bindings=bindings)
    assert not parsed.rejected_items
    return TypedEvidenceContribution(
        mechanism,
        bindings,
        ParsedTypedItems(
            parsed.accepted_items,
            parsed.rejected_items,
            _sha(f"parse-{mechanism}"),
        ),
        _sha(f"artifact-{mechanism}"),
        FrontierMode.BOUNDED,
        False,
    )


def _budget(lane_id: str, cap: int) -> LaneBudget:
    return LaneBudget(lane_id, cap, CallBudget(8_000, 768, 0))


def test_each_mechanism_lane_survives_an_earlier_verbose_lane() -> None:
    verbose = _contribution(
        "parent_map",
        1,
        tuple(
            f"HelloFresh first order was 40 percent. {'detail ' * 80}{index}"
            for index in range(4)
        ),
    )
    specialist = _contribution(
        "full_store",
        500_001,
        ("UberEats first order was discounted by 20 percent.",),
    )
    one_verbose = lane_content_token_proxy(
        (verbose.parsed.accepted_items[0],), verbose.bindings
    )
    one_specialist = lane_content_token_proxy(
        specialist.parsed.accepted_items, specialist.bindings
    )
    allocation = allocate_typed_contribution_lanes(
        (verbose, specialist),
        lane_budgets=(
            _budget("protected_parent", one_verbose),
            _budget("full_store", one_specialist),
        ),
        lane_by_mechanism={
            "parent_map": "protected_parent",
            "full_store": "full_store",
        },
    )

    retained = {
        row.mechanism_id: len(row.parsed.accepted_items)
        for row in allocation.contributions
    }
    assert retained == {"parent_map": 1, "full_store": 1}
    assert allocation.receipts[0].omitted_item_receipt_sha256s
    assert all(row.non_borrowable for row in allocation.receipts)


def test_unused_lane_capacity_is_not_borrowed() -> None:
    empty = _contribution("empty_parent", 1, ())
    specialist = _contribution(
        "full_store",
        500_001,
        ("UberEats first order was discounted by 20 percent. " + "detail " * 10,),
    )
    required = lane_content_token_proxy(
        specialist.parsed.accepted_items, specialist.bindings
    )
    allocation = allocate_typed_contribution_lanes(
        (empty, specialist),
        lane_budgets=(
            _budget("parent", required * 2),
            _budget("full_store", required - 1),
        ),
        lane_by_mechanism={
            "empty_parent": "parent",
            "full_store": "full_store",
        },
    )

    by_mechanism = {row.mechanism_id: row for row in allocation.contributions}
    assert not by_mechanism["full_store"].parsed.accepted_items
    assert allocation.receipts[0].unspent_content_tokens == required * 2
    assert allocation.receipts[1].omitted_item_receipt_sha256s


def test_shared_lane_round_robins_source_and_pointer_contributions() -> None:
    source = _contribution(
        "tail_source",
        300_001,
        (
            "HelloFresh first order was 40 percent. " + "source " * 10,
            "HelloFresh repeated detail. " + "source " * 10,
        ),
    )
    pointer = _contribution(
        "tail_pointer",
        400_001,
        (
            "UberEats first order was 20 percent. " + "pointer " * 10,
            "UberEats repeated detail. " + "pointer " * 10,
        ),
    )
    first_pair = (
        source.parsed.accepted_items[0],
        pointer.parsed.accepted_items[0],
    )
    all_bindings = tuple((*source.bindings, *pointer.bindings))
    pair_cap = lane_content_token_proxy(first_pair, all_bindings)
    allocation = allocate_typed_contribution_lanes(
        (source, pointer),
        lane_budgets=(_budget("tail", pair_cap),),
        lane_by_mechanism={
            "tail_source": "tail",
            "tail_pointer": "tail",
        },
    )

    retained = {
        row.mechanism_id: tuple(item.summary for item in row.parsed.accepted_items)
        for row in allocation.contributions
    }
    assert len(retained["tail_source"]) == 1
    assert len(retained["tail_pointer"]) == 1
    assert allocation.receipts[0].final_content_token_proxy <= pair_cap


def test_dropped_items_also_drop_their_local_bindings_but_keep_receipts() -> None:
    contribution = _contribution(
        "full_store",
        500_001,
        (
            "HelloFresh first order was 40 percent.",
            "UberEats first order was 20 percent.",
        ),
    )
    one_cap = lane_content_token_proxy(
        (contribution.parsed.accepted_items[0],), contribution.bindings
    )
    allocation = allocate_typed_contribution_lanes(
        (contribution,),
        lane_budgets=(_budget("full_store", one_cap),),
        lane_by_mechanism={"full_store": "full_store"},
    )
    rebuilt = allocation.contributions[0]
    receipt = allocation.receipts[0]

    represented = {
        handle for item in rebuilt.parsed.accepted_items for handle in item.handle_ids
    }
    assert {row.handle_id for row in rebuilt.bindings} == represented
    assert len(receipt.selected_item_receipt_sha256s) == 1
    assert len(receipt.omitted_item_receipt_sha256s) == 1
    assert len(receipt.selected_binding_receipt_sha256s) == 1
    assert len(receipt.omitted_binding_receipt_sha256s) == 1
    assert allocation.projection()["gold_loaded"] is False
    assert allocation.projection()["retained_transformer_token_state_bytes"] == 0


def test_late_third_source_group_survives_three_item_lane_with_local_priority() -> None:
    contribution = _contribution(
        "full_store",
        500_001,
        (
            "HelloFresh first order was 40 percent, early group one.",
            "HelloFresh repeated group-one detail was 40 percent.",
            "HelloFresh another group-one detail was 40 percent.",
            "UberEats first order was 20 percent, group two.",
            "UberEats repeated group-two detail was 20 percent.",
            "HelloFresh and UberEats comparison bridge was 30 percent, late group three.",
        ),
        groups=(501, 501, 501, 502, 502, 503),
    )
    items = contribution.parsed.accepted_items
    # The late bridge has the strongest local scanner axes, but the allocator
    # must still rotate through all three opaque source groups before taking a
    # second item from any group.
    priorities = {
        binding.handle_id: ((10, 1) if index == 5 else (1, 0))
        for index, binding in enumerate(contribution.bindings)
    }
    expected = (items[5], items[0], items[3])
    cap = lane_content_token_proxy(expected, contribution.bindings)
    allocation = allocate_typed_contribution_lanes(
        (contribution,),
        lane_budgets=(_budget("full_store", cap),),
        lane_by_mechanism={"full_store": "full_store"},
        operator_spec=compile_typed_operator_spec(QUESTION),
        local_selection_priority_by_handle=priorities,
    )
    retained = allocation.contributions[0]
    retained_groups = {
        binding.source_group_handle for binding in retained.bindings
    }
    assert retained_groups == {"G501", "G502", "G503"}
    assert len(retained.parsed.accepted_items) == 3
    assert (
        allocation.receipts[0].local_selection_priority_receipt_sha256
        == allocation.receipts[0].projection()[
            "local_selection_priority_receipt_sha256"
        ]
    )


def test_shared_surplus_preserves_minima_and_uses_only_aggregate_slack() -> None:
    empty = _contribution("empty_parent", 1, ())
    specialist = _contribution(
        "full_store",
        500_001,
        ("UberEats first order was discounted by 20 percent.",),
    )
    required = lane_content_token_proxy(
        specialist.parsed.accepted_items,
        specialist.bindings,
    )
    lane_budgets = (
        _budget("parent", required * 2),
        _budget("full_store", required - 1),
    )
    mapping = {
        "empty_parent": "parent",
        "full_store": "full_store",
    }
    minimum = allocate_typed_contribution_lanes(
        (empty, specialist),
        lane_budgets=lane_budgets,
        lane_by_mechanism=mapping,
    )
    assert not minimum.contributions[1].parsed.accepted_items

    expanded, audit = fill_typed_lane_surplus(
        (empty, specialist),
        minimum,
        lane_budgets=lane_budgets,
        lane_by_mechanism=mapping,
    )
    assert expanded[1].parsed.accepted_items == specialist.parsed.accepted_items
    assert audit["added_item_receipt_sha256s"] == [
        specialist.parsed.accepted_items[0].receipt_sha256
    ]
    assert audit["final_content_token_proxy"] <= audit[
        "shared_final_content_token_cap"
    ]
    rows = {row["lane_id"]: row for row in audit["lane_rows"]}
    assert rows["full_store"]["borrowed_content_tokens"] == 1
    assert audit["provider_prompt_count"] == 0
    assert audit["retained_transformer_token_state_bytes"] == 0
    assert audit["gold_loaded"] is False


def test_shared_surplus_never_drops_minima_and_is_deterministic() -> None:
    parent = _contribution(
        "parent_map",
        1,
        ("HelloFresh first order was discounted by 40 percent.",),
    )
    specialist = _contribution(
        "full_store",
        500_001,
        (
            "UberEats first order was discounted by 20 percent.",
            "The two first-order discounts belong to separate services.",
        ),
    )
    parent_cap = lane_content_token_proxy(
        parent.parsed.accepted_items,
        parent.bindings,
    )
    one_specialist_cap = lane_content_token_proxy(
        (specialist.parsed.accepted_items[0],),
        specialist.bindings,
    )
    lane_budgets = (
        _budget("parent", parent_cap),
        _budget("full_store", one_specialist_cap),
    )
    mapping = {"parent_map": "parent", "full_store": "full_store"}
    minimum = allocate_typed_contribution_lanes(
        (parent, specialist),
        lane_budgets=lane_budgets,
        lane_by_mechanism=mapping,
    )
    minimum_receipts = {
        item.receipt_sha256
        for contribution in minimum.contributions
        for item in contribution.parsed.accepted_items
    }

    first = fill_typed_lane_surplus(
        (parent, specialist),
        minimum,
        lane_budgets=lane_budgets,
        lane_by_mechanism=mapping,
    )
    second = fill_typed_lane_surplus(
        (parent, specialist),
        minimum,
        lane_budgets=lane_budgets,
        lane_by_mechanism=mapping,
    )
    final_receipts = {
        item.receipt_sha256
        for contribution in first[0]
        for item in contribution.parsed.accepted_items
    }
    assert minimum_receipts <= final_receipts
    assert first[1] == second[1]
    assert [row.projection() for row in first[0]] == [
        row.projection() for row in second[0]
    ]
    assert first[1]["added_item_receipt_sha256s"] == []
    assert first[1]["budget_omitted_item_receipt_sha256s"]


def test_shared_surplus_rejects_allocation_from_changed_originals() -> None:
    original = _contribution(
        "parent_map",
        1,
        ("HelloFresh first order was discounted by 40 percent.",),
    )
    changed = _contribution(
        "parent_map",
        1,
        ("HelloFresh first order was discounted by 10 percent.",),
    )
    cap = max(
        lane_content_token_proxy(
            original.parsed.accepted_items,
            original.bindings,
        ),
        lane_content_token_proxy(
            changed.parsed.accepted_items,
            changed.bindings,
        ),
    )
    budgets = (_budget("parent", cap),)
    mapping = {"parent_map": "parent"}
    changed_allocation = allocate_typed_contribution_lanes(
        (changed,),
        lane_budgets=budgets,
        lane_by_mechanism=mapping,
    )

    with pytest.raises(
        MatchedEvalContractError,
        match="escaped or changed its original contribution",
    ):
        fill_typed_lane_surplus(
            (original,),
            changed_allocation,
            lane_budgets=budgets,
            lane_by_mechanism=mapping,
        )


def test_shared_surplus_recomputes_claimed_lane_token_proxy() -> None:
    original = _contribution(
        "parent_map",
        1,
        ("HelloFresh first order was discounted by 40 percent.",),
    )
    actual_proxy = lane_content_token_proxy(
        original.parsed.accepted_items,
        original.bindings,
    )
    budgets = (_budget("parent", actual_proxy),)
    mapping = {"parent_map": "parent"}
    allocation = allocate_typed_contribution_lanes(
        (original,),
        lane_budgets=budgets,
        lane_by_mechanism=mapping,
    )
    falsified_lane = replace(
        allocation.receipts[0],
        final_content_token_proxy=0,
        receipt_sha256="",
    )
    falsified_allocation = replace(
        allocation,
        receipts=(falsified_lane,),
        receipt_sha256="",
    )

    with pytest.raises(
        MatchedEvalContractError,
        match="partitions or accounting changed",
    ):
        fill_typed_lane_surplus(
            (original,),
            falsified_allocation,
            lane_budgets=budgets,
            lane_by_mechanism=mapping,
        )
