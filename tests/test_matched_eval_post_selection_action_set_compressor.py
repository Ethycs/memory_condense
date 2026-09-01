from __future__ import annotations

import pytest

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256
from tools.matched_eval.post_selection_action_set_compressor import (
    ActionSetCompressionError,
    SelectedExactSnippet,
    compile_action_linked_set_demand,
    compress_action_linked_set_evidence,
    compress_selected_typed_action_set_evidence,
    locate_exact_action_set_fact,
    typed_packet_from_action_linked_set_compression,
)
from tools.matched_eval.typed_downstream_operator import (
    compile_downstream_operator_overlay,
    execute_downstream_typed_operator,
)
from tools.matched_eval.typed_operator_adapter import (
    EvidenceHandleBinding,
    EvidenceOrigin,
    FrontierMode,
    ProvenanceGrade,
    parse_typed_items,
)
from tools.matched_eval.typed_operator_executor import ExecutionStatus
from tools.matched_eval.typed_operator_spec import compile_typed_operator_spec


_QUESTION = (
    "[Question asked at 2023/05/30 (Tue) 19:49]\n"
    "What are the two hobbies that led me to join online communities?"
)
_COOKING = (
    "I've already joined a few online communities related to cooking, which "
    "led me to engage in discussions about recipe techniques and share my "
    "thoughts on food-related posts."
)
_PHOTOGRAPHY = (
    "By the way, I've been really enjoying editing my photos in Lightroom - "
    "the online communities I've joined have been super helpful in learning "
    "new techniques and getting feedback on my work."
)


def _sha(value: str) -> str:
    return quote_sha256(value)


def _spec_overlay_demand():
    spec = compile_typed_operator_spec(_QUESTION)
    overlay = compile_downstream_operator_overlay(_QUESTION, spec)
    demand = compile_action_linked_set_demand(_QUESTION, spec, overlay)
    return spec, overlay, demand


def _binding(
    handle_id: str,
    group_handle: str,
    text: str,
    *,
    artifact: str | None = None,
) -> EvidenceHandleBinding:
    return EvidenceHandleBinding(
        handle_id=handle_id,
        origin=EvidenceOrigin.MAP,
        provenance_grade=ProvenanceGrade.EXACT_CITATION,
        source_group_handle=group_handle,
        sealed_artifact_sha256=artifact or _sha("sealed-q65-selected-artifact"),
        parent_receipt_sha256=_sha(f"parent:{handle_id}"),
        evidence_receipt_sha256=_sha(f"evidence:{handle_id}"),
        payload_sha256=_sha(text),
        citation_sha256=_sha(text),
        citation_char_count=len(text),
        local_source_locator_sha256=_sha(f"source:{handle_id}"),
    )


def _typed_inputs(*rows: tuple[str, str, str]):
    spec, overlay, demand = _spec_overlay_demand()
    bindings = tuple(
        _binding(handle, group, text) for handle, group, text in rows
    )
    parsed = parse_typed_items(
        [
            {
                "handle_ids": [handle],
                "kind": "member",
                "relation": "memory_role:user;date_basis:row_created_at",
                "status": "completed",
                "summary": text,
                "value_authority": "explicit",
            }
            for handle, _group, text in rows
        ],
        operator_spec=spec,
        bindings=bindings,
    )
    assert not parsed.rejected_items
    return spec, overlay, demand, parsed.accepted_items, bindings


def test_q65_exact_typed_snippets_form_scoped_witness_without_global_closure() -> None:
    spec, overlay, demand, items, bindings = _typed_inputs(
        ("H500001", "G500001", _COOKING),
        ("H500013", "G500008", _PHOTOGRAPHY),
    )

    compression = compress_selected_typed_action_set_evidence(
        demand,
        items,
        bindings,
        spec,
        selection_receipt_sha256=_sha("q65-post-selection"),
    )
    packet = typed_packet_from_action_linked_set_compression(spec, compression)
    execution = execute_downstream_typed_operator(spec, packet, overlay)

    assert demand.cardinality == 2
    assert demand.action_concepts == ("join",)
    assert demand.relation_anchor_terms == ("online", "community")
    assert tuple(row.member_text for row in compression.facts) == (
        "cooking",
        "photography",
    )
    assert tuple(row.member_surface_text for row in compression.bound_candidates) == (
        "cooking",
        "photos",
    )
    assert tuple(row.member_derivation for row in compression.bound_candidates) == (
        "exact_span",
        "lexical_normalization",
    )
    assert compression.bindings == bindings
    assert tuple(row.receipt_sha256 for row in compression.bindings) == tuple(
        row.receipt_sha256 for row in bindings
    )
    assert compression.closure.explicit_cardinality_satisfied is True
    assert compression.closure.support_frontier_closed is True
    assert compression.closure.semantic_absence_may_be_inferred is False
    assert compression.contribution.frontier_mode is FrontierMode.BOUNDED
    assert compression.provider_prompt_count == 0
    assert compression.retained_transformer_token_state_bytes == 0
    assert compression.provider_payload_token_proxy < 512
    assert packet.frontier.closed is False
    assert packet.provider_payload_token_proxy < 768
    assert packet.provider_payload_token_proxy + packet.output_token_reserve < 8_000
    assert execution.status is ExecutionStatus.INSUFFICIENT
    assert execution.prediction == ""
    assert execution.used_handle_ids == ()
    provider = compression.provider_projection()
    assert provider["support_frontier"]["closed"] is True
    assert provider["support_frontier"]["scope"] == (
        "selected_action_linked_members_only"
    )
    assert provider["support_frontier"]["generic_frontier_closed"] is False
    assert provider["support_frontier"]["semantic_absence_may_be_inferred"] is False
    assert tuple(row["support"][0]["quote_sha256"] for row in provider["facts"]) == (
        _sha(_COOKING),
        _sha(_PHOTOGRAPHY),
    )


def test_fact_dedup_runs_only_after_all_selected_support_is_bound() -> None:
    duplicate = "I joined online communities related to cooking."
    spec, _overlay, demand, items, bindings = _typed_inputs(
        ("H500001", "G500001", _COOKING),
        ("H500002", "G500002", duplicate),
        ("H500013", "G500008", _PHOTOGRAPHY),
    )

    compression = compress_selected_typed_action_set_evidence(
        demand,
        items,
        bindings,
        spec,
        selection_receipt_sha256=_sha("q65-three-selected-before-dedup"),
    )

    assert compression.closure.selected_snippet_count == 3
    assert compression.closure.candidate_count_before_dedup == 3
    assert compression.closure.distinct_supported_member_count == 2
    assert len(compression.bound_candidates) == 3
    assert len(compression.facts) == 2
    assert compression.facts[0].member_text == "cooking"
    assert compression.facts[0].handle_ids == ("H500001", "H500002")
    assert compression.closure.support_frontier_closed is True


def test_unused_typed_snippet_is_excluded_after_selection_not_before() -> None:
    distractor = "I might join online communities to discuss camera collecting."
    spec, _overlay, demand, items, bindings = _typed_inputs(
        ("H500001", "G500001", _COOKING),
        ("H500009", "G500009", distractor),
        ("H500013", "G500008", _PHOTOGRAPHY),
    )

    compression = compress_selected_typed_action_set_evidence(
        demand,
        items,
        bindings,
        spec,
        selection_receipt_sha256=_sha("q65-distractor-selected"),
    )

    assert len(compression.selected_snippet_receipt_sha256s) == 3
    assert len(compression.post_selection_exclusions) == 1
    exclusion = compression.post_selection_exclusions[0]
    assert exclusion.selected_candidate_id == items[1].item_id
    assert "after_selection" in exclusion.reason
    assert tuple(row.handle_id for row in compression.bindings) == (
        "H500001",
        "H500013",
    )
    assert compression.closure.support_frontier_closed is True
    assert compression.closure.semantic_absence_may_be_inferred is False


def test_missing_member_remains_bounded_and_cannot_execute_complete_set() -> None:
    spec, overlay, demand, items, bindings = _typed_inputs(
        ("H500001", "G500001", _COOKING),
    )
    compression = compress_selected_typed_action_set_evidence(
        demand,
        items,
        bindings,
        spec,
        selection_receipt_sha256=_sha("q65-cooking-only"),
    )
    packet = typed_packet_from_action_linked_set_compression(spec, compression)
    execution = execute_downstream_typed_operator(spec, packet, overlay)

    assert compression.closure.explicit_cardinality_satisfied is False
    assert compression.closure.support_frontier_closed is False
    assert compression.contribution.frontier_mode is FrontierMode.BOUNDED
    assert packet.frontier.closed is False
    assert execution.status is ExecutionStatus.INSUFFICIENT
    assert execution.prediction == ""


def test_exact_span_api_rejects_action_support_without_relation_object() -> None:
    spec, _overlay, demand = _spec_overlay_demand()
    text = "I joined a local club because I enjoy cooking."
    selection_receipt = _sha("selected-unrelated-join")
    snippet = SelectedExactSnippet(
        selection_ordinal=0,
        candidate_id=_sha("unrelated-candidate"),
        source_group_handle="G900001",
        quote=text,
        quote_sha256=_sha(text),
        role="user",
        created_at="2023-05-01",
        evidence_receipt_sha256=_sha("unrelated-evidence"),
        local_binding_receipt_sha256=_sha("unrelated-binding"),
        local_source_locator_sha256=_sha("unrelated-source"),
        selection_receipt_sha256=selection_receipt,
        token_count=count_tokens(text),
    )
    proposal = locate_exact_action_set_fact(
        snippet,
        member_text="cooking",
        support_quote=text,
        action_concept="join",
    )

    with pytest.raises(ActionSetCompressionError, match="relation object"):
        compress_action_linked_set_evidence(
            demand,
            (snippet,),
            (proposal,),
            spec,
            sealed_selection_artifact_sha256=_sha("unrelated-artifact"),
        )


def test_compact_payload_cap_fails_closed_without_fact_truncation() -> None:
    spec, _overlay, demand, items, bindings = _typed_inputs(
        ("H500001", "G500001", _COOKING),
        ("H500013", "G500008", _PHOTOGRAPHY),
    )

    with pytest.raises(ActionSetCompressionError, match="payload cap"):
        compress_selected_typed_action_set_evidence(
            demand,
            items,
            bindings,
            spec,
            selection_receipt_sha256=_sha("q65-impossible-payload-cap"),
            payload_token_cap=1,
        )
