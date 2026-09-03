from __future__ import annotations

from dataclasses import fields

import hashlib

import pytest

from tools.matched_eval.contracts import MatchedEvalContractError
from tools.matched_eval.prompt_tick_contracts import CallBudget, LaneBudget
from tools.matched_eval.typed_additive_composer import (
    AdditiveTypedComposition,
    FORMAT as LEGACY_COMPOSER_FORMAT,
    LEGACY_COMPOSITION_MODE,
    POST_DEDUP_BACKFILL_COMPOSITION_MODE,
    POST_DEDUP_BACKFILL_FORMAT,
    compose_additive_typed_evidence,
    deduplicate_selected_contributions,
)
from tools.matched_eval.typed_lane_allocator import lane_content_token_proxy
from tools.matched_eval.typed_memory_final_arm import (
    LOCAL_RETENTION_PRIORITY_WIDTH,
    fit_typed_final_prompt,
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
    "[Question asked at 2026/08/27 12:00] "
    "What paper did I select for the cedar lantern?"
)


def test_successor_format_field_preserves_legacy_positional_receipt_slot() -> None:
    names = [field.name for field in fields(AdditiveTypedComposition)]

    assert names[-2:] == ["receipt_sha256", "format_id"]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _contribution(
    mechanism_id: str,
    start: int,
    summaries: tuple[str, ...],
) -> TypedEvidenceContribution:
    summaries = tuple(row.strip() for row in summaries)
    artifact = _sha(f"artifact-{mechanism_id}")
    bindings = tuple(
        EvidenceHandleBinding(
            f"H{start + ordinal:03d}",
            EvidenceOrigin.MAP,
            ProvenanceGrade.EXACT_CITATION,
            f"G{start + ordinal:03d}",
            artifact,
            _sha(f"parent-{mechanism_id}"),
            _sha(f"evidence-{mechanism_id}-{ordinal}"),
            _sha(f"payload-{mechanism_id}-{ordinal}"),
            _sha(summary),
            len(summary),
            _sha(f"locator-{mechanism_id}-{ordinal}"),
        )
        for ordinal, summary in enumerate(summaries)
    )
    parsed = parse_typed_items(
        [
            {
                "handle_ids": [binding.handle_id],
                "status": "completed",
                "summary": summary,
            }
            for binding, summary in zip(bindings, summaries, strict=True)
        ],
        operator_spec=compile_typed_operator_spec(QUESTION),
        bindings=bindings,
    )
    assert not parsed.rejected_items
    return TypedEvidenceContribution(
        mechanism_id,
        bindings,
        ParsedTypedItems(
            parsed.accepted_items,
            parsed.rejected_items,
            _sha(f"parse-{mechanism_id}"),
        ),
        artifact,
        FrontierMode.BOUNDED,
        False,
    )


def _budget(lane_id: str, cap: int) -> LaneBudget:
    return LaneBudget(lane_id, cap, CallBudget(8_000, 768, 0))


def test_post_selection_dedup_requires_equal_nonempty_exact_span_sets() -> None:
    summary = "I selected cobalt paper for the cedar lantern."
    first = _contribution("first", 1, (summary,))
    second = _contribution("second", 100_001, (summary,))
    shared = _sha("span-shared")
    additional = _sha("span-additional-origin")

    retained, audit = deduplicate_selected_contributions(
        (first, second),
        owner_priority_by_mechanism={"first": 1, "second": 2},
        exact_span_keys_by_handle={
            first.bindings[0].handle_id: (shared,),
            second.bindings[0].handle_id: (shared, additional),
        },
    )

    assert not audit["exclusions"]
    assert [len(row.parsed.accepted_items) for row in retained] == [1, 1]
    assert audit["operation_position"] == "after_all_mechanism_selection"
    assert audit["receipt_sha256"]


def test_explicit_parent_priority_owns_a_proven_duplicate() -> None:
    summary = "I selected cobalt paper for the cedar lantern."
    specialist = _contribution("specialist", 100_001, (summary,))
    parent = _contribution("protected_parent", 1, (summary,))
    shared = _sha("same-exact-span")

    retained, audit = deduplicate_selected_contributions(
        (specialist, parent),
        owner_priority_by_mechanism={
            "specialist": 10,
            "protected_parent": 100,
        },
        exact_span_keys_by_handle={
            specialist.bindings[0].handle_id: (shared,),
            parent.bindings[0].handle_id: (shared,),
        },
    )

    assert [len(row.parsed.accepted_items) for row in retained] == [0, 1]
    assert len(audit["exclusions"]) == 1
    exclusion = audit["exclusions"][0]
    assert exclusion["owner_mechanism_id"] == "protected_parent"
    assert exclusion["duplicate_mechanism_id"] == "specialist"
    assert exclusion["shared_exact_span_receipt_sha256s"] == [shared]


def test_compose_preserves_each_nonempty_lane_minimum_and_is_fit_ready() -> None:
    parent = _contribution(
        "protected_parent",
        1,
        (
            "I selected cobalt paper for the cedar lantern. " + "parent detail " * 10,
            "The lantern workshop also discussed bamboo frames. " + "extra " * 10,
        ),
    )
    specialist = _contribution(
        "episodic_specialist",
        100_001,
        ("At the cedar lantern workshop, the selected paper was cobalt.",),
    )
    parent_cap = lane_content_token_proxy(
        (parent.parsed.accepted_items[0],),
        parent.bindings,
    )
    specialist_cap = lane_content_token_proxy(
        specialist.parsed.accepted_items,
        specialist.bindings,
    )

    result = compose_additive_typed_evidence(
        compile_typed_operator_spec(QUESTION),
        (parent, specialist),
        lane_budgets=(
            _budget("parent", parent_cap),
            _budget("specialist", specialist_cap),
        ),
        lane_by_mechanism={
            "protected_parent": "parent",
            "episodic_specialist": "specialist",
        },
        dedup_owner_priority_by_mechanism={
            "protected_parent": 100,
            "episodic_specialist": 10,
        },
    )
    explicit_legacy = compose_additive_typed_evidence(
        compile_typed_operator_spec(QUESTION),
        (parent, specialist),
        lane_budgets=(
            _budget("parent", parent_cap),
            _budget("specialist", specialist_cap),
        ),
        lane_by_mechanism={
            "protected_parent": "parent",
            "episodic_specialist": "specialist",
        },
        dedup_owner_priority_by_mechanism={
            "protected_parent": 100,
            "episodic_specialist": 10,
        },
        composition_mode=LEGACY_COMPOSITION_MODE,
    )

    protected = set(result.protected_item_receipt_sha256s)
    assert result.projection() == explicit_legacy.projection()
    assert result.projection()["format"] == LEGACY_COMPOSER_FORMAT
    assert result.receipt_sha256 == (
        "64404922f943965d40cebbfd59c8e81eab63b3e77172ca7ea53cf7fb0f10a7a3"
    )
    assert "initial_post_selection_dedup_audit" not in (
        result.post_selection_dedup_audit
    )
    assert len(protected) == 2
    assert protected <= {row.receipt_sha256 for row in result.packet.items}
    lane_rows = {
        row.lane_id: row for row in result.minimum_allocation.receipts
    }
    assert len(lane_rows["parent"].selected_item_receipt_sha256s) == 1
    assert len(lane_rows["specialist"].selected_item_receipt_sha256s) == 1
    assert result.surplus_fill_audit["final_content_token_proxy"] <= (
        result.surplus_fill_audit["shared_final_content_token_cap"]
    )
    assert result.projection()["gold_loaded"] is False
    assert result.projection()["provider_prompt_count"] == 0

    fitted = fit_typed_final_prompt(
        dated_question=QUESTION,
        parent_prediction="The earlier answer did not recover the paper color.",
        packet=result.packet,
        mechanism_by_handle=result.mechanism_by_handle,
        local_retention_priority_by_handle=(
            result.retained_local_priority_by_handle
        ),
        minimum_usable_items_per_mechanism=1,
        protected_item_receipt_sha256s=result.protected_item_receipt_sha256s,
        protection_source_receipt_sha256=(
            result.protection_source_receipt_sha256
        ),
    )
    assert protected <= {row.receipt_sha256 for row in fitted.packet.items}
    assert fitted.protection_source_receipt_sha256 == result.receipt_sha256


def test_lane_admission_precedes_dedup_so_a_later_omitted_owner_cannot_erase_class() -> None:
    duplicate = (
        "I selected cobalt paper for the cedar lantern. "
        + "exact supporting detail " * 12
    )
    preferred_parent_item = (
        "The parent map retained the cedar lantern workshop decision. "
        + "protected context " * 8
    )
    parent = _contribution(
        "protected_parent",
        1,
        (preferred_parent_item, duplicate),
    )
    specialist = _contribution(
        "episodic_specialist",
        100_001,
        (duplicate,),
    )
    parent_cap = lane_content_token_proxy(
        (parent.parsed.accepted_items[0],), parent.bindings
    )
    specialist_cap = lane_content_token_proxy(
        specialist.parsed.accepted_items, specialist.bindings
    )
    priority = (1, *((0,) * (LOCAL_RETENTION_PRIORITY_WIDTH - 1)))
    shared_span = _sha("dedup-owner-can-be-lane-omitted")

    result = compose_additive_typed_evidence(
        compile_typed_operator_spec(QUESTION),
        (parent, specialist),
        lane_budgets=(
            _budget("parent", parent_cap),
            _budget("specialist", specialist_cap),
        ),
        lane_by_mechanism={
            "protected_parent": "parent",
            "episodic_specialist": "specialist",
        },
        dedup_owner_priority_by_mechanism={
            "protected_parent": 100,
            "episodic_specialist": 10,
        },
        exact_span_keys_by_handle={
            parent.bindings[1].handle_id: (shared_span,),
            specialist.bindings[0].handle_id: (shared_span,),
        },
        local_selection_priority_by_handle={
            parent.bindings[0].handle_id: priority,
        },
        composition_mode=POST_DEDUP_BACKFILL_COMPOSITION_MODE,
    )

    retained_summaries = {item.summary for item in result.packet.items}
    assert preferred_parent_item.strip() in retained_summaries
    assert duplicate.strip() in retained_summaries
    assert specialist.bindings[0].handle_id in result.mechanism_by_handle
    assert result.post_selection_dedup_audit["exclusions"] == []
    assert result.post_selection_dedup_audit["operation_position"] == (
        "after_independent_lane_admission_and_shared_surplus_fill_"
        "then_dedup_and_freed_capacity_backfill"
    )


def test_exact_dedup_frees_shared_capacity_for_next_unique_item() -> None:
    duplicate = "I selected cobalt paper for the cedar lantern."
    unique = "The lantern frame used ash wood."
    parent = _contribution("protected_parent", 1, (duplicate,))
    specialist = _contribution(
        "episodic_specialist",
        100_001,
        (duplicate, unique),
    )
    parent_cap = lane_content_token_proxy(
        parent.parsed.accepted_items,
        parent.bindings,
    )
    specialist_cap = lane_content_token_proxy(
        (specialist.parsed.accepted_items[0],),
        specialist.bindings,
    )
    shared_span = _sha("dedup-releases-capacity-for-unique-backfill")
    priority = (1, *((0,) * (LOCAL_RETENTION_PRIORITY_WIDTH - 1)))

    result = compose_additive_typed_evidence(
        compile_typed_operator_spec(QUESTION),
        (parent, specialist),
        lane_budgets=(
            _budget("parent", parent_cap),
            _budget("specialist", specialist_cap),
        ),
        lane_by_mechanism={
            "protected_parent": "parent",
            "episodic_specialist": "specialist",
        },
        dedup_owner_priority_by_mechanism={
            "protected_parent": 100,
            "episodic_specialist": 10,
        },
        exact_span_keys_by_handle={
            parent.bindings[0].handle_id: (shared_span,),
            specialist.bindings[0].handle_id: (shared_span,),
        },
        local_selection_priority_by_handle={
            specialist.bindings[0].handle_id: priority,
        },
        composition_mode=POST_DEDUP_BACKFILL_COMPOSITION_MODE,
    )

    assert {item.summary for item in result.packet.items} == {
        duplicate,
        unique,
    }
    assert result.post_selection_dedup_audit[
        "backfilled_item_receipt_sha256s"
    ] == [specialist.parsed.accepted_items[1].receipt_sha256]
    assert result.projection()["format"] == POST_DEDUP_BACKFILL_FORMAT
    backfill = result.post_selection_dedup_audit["backfill_rows"]
    assert [row["disposition"] for row in backfill] == ["admitted_unique"]
    assert (
        result.post_selection_dedup_audit["final_content_token_proxy"]
        <= result.post_selection_dedup_audit["shared_final_content_token_cap"]
    )


def test_selected_duplicate_minima_transfer_protection_to_one_surviving_owner() -> None:
    summary = "I selected cobalt paper for the cedar lantern."
    parent = _contribution("protected_parent", 1, (summary,))
    specialist = _contribution("episodic_specialist", 100_001, (summary,))
    parent_cap = lane_content_token_proxy(
        parent.parsed.accepted_items, parent.bindings
    )
    specialist_cap = lane_content_token_proxy(
        specialist.parsed.accepted_items, specialist.bindings
    )
    shared_span = _sha("both-lane-minima-same-span")

    result = compose_additive_typed_evidence(
        compile_typed_operator_spec(QUESTION),
        (parent, specialist),
        lane_budgets=(
            _budget("parent", parent_cap),
            _budget("specialist", specialist_cap),
        ),
        lane_by_mechanism={
            "protected_parent": "parent",
            "episodic_specialist": "specialist",
        },
        dedup_owner_priority_by_mechanism={
            "protected_parent": 100,
            "episodic_specialist": 10,
        },
        exact_span_keys_by_handle={
            parent.bindings[0].handle_id: (shared_span,),
            specialist.bindings[0].handle_id: (shared_span,),
        },
        composition_mode=POST_DEDUP_BACKFILL_COMPOSITION_MODE,
    )

    assert len(result.packet.items) == 1
    assert result.packet.items[0].summary == summary
    assert result.protected_item_receipt_sha256s == (
        parent.parsed.accepted_items[0].receipt_sha256,
    )
    assert len(result.post_selection_dedup_audit["exclusions"]) == 1
    assert result.fair_merge_audit[
        "original_protected_minimum_item_receipt_sha256s"
    ] == [
        parent.parsed.accepted_items[0].receipt_sha256,
        specialist.parsed.accepted_items[0].receipt_sha256,
    ]
    assert result.fair_merge_audit["protected_minimum_item_transfer_rows"]


def test_compose_fails_if_a_hard_lane_cap_starves_a_nonempty_mechanism() -> None:
    parent = _contribution(
        "protected_parent",
        1,
        ("I selected cobalt paper for the cedar lantern.",),
    )
    required = lane_content_token_proxy(parent.parsed.accepted_items, parent.bindings)

    with pytest.raises(
        MatchedEvalContractError,
        match="starved a nonempty additive mechanism",
    ):
        compose_additive_typed_evidence(
            compile_typed_operator_spec(QUESTION),
            (parent,),
            lane_budgets=(_budget("parent", required - 1),),
            lane_by_mechanism={"protected_parent": "parent"},
            dedup_owner_priority_by_mechanism={"protected_parent": 100},
        )


def test_compose_rejects_cross_mechanism_opaque_range_overlap() -> None:
    first = _contribution("first", 1, ("Cobalt paper was selected.",))
    second = _contribution("second", 2, ("A bamboo frame was selected.",))

    with pytest.raises(
        MatchedEvalContractError,
        match="globally disjoint H/G ranges",
    ):
        compose_additive_typed_evidence(
            compile_typed_operator_spec(QUESTION),
            (first, second),
            lane_budgets=(_budget("shared", 1_000),),
            lane_by_mechanism={"first": "shared", "second": "shared"},
            dedup_owner_priority_by_mechanism={"first": 2, "second": 1},
        )
