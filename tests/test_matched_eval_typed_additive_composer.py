from __future__ import annotations

import hashlib

import pytest

from tools.matched_eval.contracts import MatchedEvalContractError
from tools.matched_eval.prompt_tick_contracts import CallBudget, LaneBudget
from tools.matched_eval.typed_additive_composer import (
    compose_additive_typed_evidence,
    deduplicate_selected_contributions,
)
from tools.matched_eval.typed_lane_allocator import lane_content_token_proxy
from tools.matched_eval.typed_memory_final_arm import fit_typed_final_prompt
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

    protected = set(result.protected_item_receipt_sha256s)
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
