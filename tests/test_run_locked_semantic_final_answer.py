from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools import run_locked_semantic_final_answer as answer
from tools.matched_eval.artifacts import SealedArtifact, publish_sealed_json
from tools.matched_eval.contracts import MatchedEvalContractError, identity_sha256
from tools.matched_eval.typed_memory_final_arm import (
    PROMPT_ROW_FORMAT,
    VALIDATION_CONTRACT_FORMAT,
    judge_row_projection,
    render_final_messages,
)
from tests.test_run_locked_specialist_final_answer import (
    _construction_question as _v1_construction_question,
)


SEMANTIC_ORDINALS = (3, 4)
SEMANTIC_FALLBACK_ORDINAL = 5


def _sha(label: str) -> str:
    return quote_sha256(label)


def _validation_contract(handles: Sequence[str]) -> dict[str, Any]:
    def semantic_body(handle: str) -> dict[str, Any]:
        return {
            "action_concepts": [],
            "completed_action_concepts": [],
            "date": "2026-08-01",
            "entity_terms": ["blue", "mug"],
            "explicit_member_count": None,
            "group_terms": [],
            "item_receipt_sha256": _sha(f"semantic item {handle}"),
            "kind": "direct",
            "numeric_role": "none",
            "numeric_qualifier": "exact",
            "numeric_value": None,
            "participant_count": None,
            "relation_terms": [],
            "semantic_unit_sha256": _sha(f"semantic unit {handle}"),
            "status": "completed",
            "summary_terms": ["bought", "blue", "mug", "market"],
            "supported_slot_ids": [],
            "unit": None,
        }

    return {
        "answer_shape": "direct",
        "by_handle": {
            handle: {
                "answer_anchor_terms": [],
                "numeric_value_rows": [],
                "semantic_rows": [semantic_body(handle)],
                "status_values": ["completed"],
                "supported_slot_ids": [],
                "usable_item_receipt_sha256s": [
                    _sha(f"semantic item {handle}")
                ],
            }
            for handle in handles
        },
        "cardinality": None,
        "comparison_mode": "none",
        "deterministic_execution_advisory": None,
        "format": VALIDATION_CONTRACT_FORMAT,
        "include_proposed": False,
        "operation": "single_supported_fact",
        "operator_spec_receipt_sha256": _sha("semantic operator spec"),
        "packet_receipt_sha256": _sha("semantic packet"),
        "question_action_concepts": [],
        "question_terms": ["what", "feed", "bought", "memory"],
        "required_slot_ids": [],
        "required_slots": [],
        "requires_all_slots": True,
        "scalar_validation_advisory": None,
        "temporal_mode": "none",
    }


def _semantic_search_binding(
    ordinal: int,
    *,
    handle: str,
    exact_text: str,
    activation_reason: str,
) -> dict[str, Any]:
    definitely_no = _sha(f"definitely no leaf {ordinal}")
    overlap = _sha(f"selected overlap leaf {ordinal}")
    retained = _sha(f"retained semantic leaf {ordinal}")
    selected = [overlap, retained]
    full = [definitely_no, *selected]
    overlap_body = {
        "overlap_local_receipt_sha256": _sha(f"local overlap {ordinal}"),
        "selected_leaf_receipt_sha256": overlap,
    }
    body = {
        "activation_reason": activation_reason,
        "dedup_receipt_sha256": _sha(f"selected dedup {ordinal}"),
        "definitely_no_leaf_receipt_sha256s": [definitely_no],
        "excluded_overlap_rows": [
            {**overlap_body, "receipt_sha256": identity_sha256(overlap_body)}
        ],
        "frontier_closed": True,
        "frontier_complete": True,
        "full_leaf_partition_receipt_sha256": identity_sha256(
            {"full_leaf_receipt_sha256s": full}
        ),
        "full_leaf_receipt_sha256s": full,
        "may_survivor_leaf_receipt_sha256s": selected,
        "no_silent_top_k_uncertainty_drop": True,
        "omitted_unknown_leaf_receipt_sha256s": [],
        "retained_leaf_rows": [
            {
                "exact_text": exact_text,
                "exact_text_sha256": _sha(exact_text),
                "handle_id": handle,
                "leaf_receipt_sha256": retained,
            }
        ],
        "search_result_receipt_sha256": _sha(f"semantic search result {ordinal}"),
        "selected_leaf_receipt_sha256s": selected,
        "selection_precedes_dedup": True,
        "specialists_evaluated_first": True,
    }
    return {**body, "receipt_sha256": identity_sha256(body)}


def _classified_closure(
    ordinal: int,
    *,
    allowed: Sequence[str],
    binding: Mapping[str, Any],
    protected_handle: str,
    protected_text: str,
    residual_handle: str,
    residual_text: str,
) -> dict[str, Any]:
    overlap = binding["excluded_overlap_rows"][0]
    retained = binding["retained_leaf_rows"][0]
    protected_binding = _sha(f"protected binding {ordinal}")
    protected_item = _sha(f"protected item {ordinal}")
    residual_binding = _sha(f"residual binding {ordinal}")
    residual_item = _sha(f"residual item {ordinal}")
    body = {
        "all_selected_may_leaves_provider_visible": True,
        "definitely_no_leaf_count": len(
            binding["definitely_no_leaf_receipt_sha256s"]
        ),
        "format": answer.CLASSIFIED_CLOSURE_FORMAT,
        "leaf_partition_receipt_sha256": binding[
            "full_leaf_partition_receipt_sha256"
        ],
        "rows": [
            {
                "cell_id": f"cell-overlap-{ordinal}",
                "cell_receipt_sha256": overlap[
                    "selected_leaf_receipt_sha256"
                ],
                "dedup_exclusion_sha256": overlap["receipt_sha256"],
                "disposition": "protected_visible_exact_duplicate",
                "exact_text_sha256": _sha(protected_text),
                "residual_binding_receipt_sha256": _sha(
                    f"overlap residual binding {ordinal}"
                ),
                "residual_item_receipt_sha256": _sha(
                    f"overlap residual item {ordinal}"
                ),
                "visible_binding_receipt_sha256": protected_binding,
                "visible_handle_id": protected_handle,
                "visible_item_receipt_sha256": protected_item,
            },
            {
                "cell_id": f"cell-retained-{ordinal}",
                "cell_receipt_sha256": retained["leaf_receipt_sha256"],
                "dedup_exclusion_sha256": None,
                "disposition": "residual_visible",
                "exact_text_sha256": _sha(residual_text),
                "residual_binding_receipt_sha256": residual_binding,
                "residual_item_receipt_sha256": residual_item,
                "visible_binding_receipt_sha256": residual_binding,
                "visible_handle_id": residual_handle,
                "visible_item_receipt_sha256": residual_item,
            },
        ],
        "selected_may_leaf_count": len(binding["selected_leaf_receipt_sha256s"]),
        "semantic_search_result_receipt_sha256": binding[
            "search_result_receipt_sha256"
        ],
        "terminal_allowed_handle_ids_sha256": identity_sha256(
            {"allowed_handle_ids": list(allowed)}
        ),
        "typed_frontier_closed": True,
    }
    return {**body, "receipt_sha256": identity_sha256(body)}


def _semantic_question(ordinal: int) -> dict[str, Any]:
    row = _v1_construction_question(ordinal)
    question = f"How much feed was bought in memory {ordinal}?"
    dated_question = f"[Question asked at 2026/08/28] {question}"
    parent = row["parent_source"]["prediction"]
    handle = f"H950{ordinal:03d}"
    group = f"G950{ordinal:03d}"
    protected_handle = f"H940{ordinal:03d}"
    protected_group = f"G940{ordinal:03d}"
    exact_text = "I bought a blue mug at the market."
    protected_text = "I bought a red bowl at the market."
    allowed = [protected_handle, handle]
    story = {"incompatible_group_pairs": []}
    preservation = {"by_handle": {}, "question_required_terms": []}
    validation = _validation_contract(allowed)
    search_binding = _semantic_search_binding(
        ordinal,
        handle=handle,
        exact_text=exact_text,
        activation_reason=(
            "no_applicable_specialist"
            if ordinal == SEMANTIC_ORDINALS[0]
            else "specialist_proofless"
        ),
    )
    provider_input = {
        "dated_question": dated_question,
        "deterministic_execution_advisory": None,
        "format": PROMPT_ROW_FORMAT,
        "protected_parent_fallback": {
            "label": "fallback_not_evidence",
            "prediction": parent,
            "prediction_sha256": _sha(parent),
        },
        "response_schema": {
            "decision": "keep_parent|replace",
            "prediction": "nonempty exact text",
            "used_handle_ids": [handle],
        },
        "scalar_validation_advisory": None,
        "story_coherence": story,
        "typed_evidence": {
            "conflict_policy": "keep_both",
            "format": "synthetic-compact-semantic-evidence-v1",
            "frontier": {
                "available_handle_ids": allowed,
                "closed": True,
                "mode": "exhaustive",
                "omitted_handle_ids": [],
                "rejected_item_count": 0,
                "represented_handle_ids": allowed,
                "truncated": False,
                "unresolved_slot_ids": [],
            },
            "handles": [
                {
                    "group_handle": protected_group,
                    "handle_id": protected_handle,
                    "origin": "parent_protected",
                    "provenance_grade": "exact_citation",
                },
                {
                    "group_handle": group,
                    "handle_id": handle,
                    "origin": "direct_pointer",
                    "provenance_grade": "exact_citation",
                }
            ],
            "items": [
                {
                    "content_coherence": "coherent",
                    "handle_ids": [protected_handle],
                    "included": True,
                    "kind": "direct",
                    "status": "completed",
                    "summary": protected_text,
                    "supported_slot_ids": [],
                    "value_authority": "explicit",
                },
                {
                    "content_coherence": "coherent",
                    "handle_ids": [handle],
                    "included": True,
                    "kind": "direct",
                    "status": "completed",
                    "summary": exact_text,
                    "supported_slot_ids": [],
                    "value_authority": "explicit",
                }
            ],
            "operator_spec": {"operation": "single_supported_fact"},
        },
    }
    messages = render_final_messages(provider_input)
    prompt_tokens = answer.count_chat_prompt_token_proxy(messages)
    fitted_receipt = _sha(f"semantic fitted prompt {ordinal}")
    terminal_receipt_body = {
        "fitted_prompt_receipt_sha256": fitted_receipt,
        "message_renderer_format": answer.SEMANTIC_PROMPT_FORMAT,
        "messages_sha256": identity_sha256(list(messages)),
        "output_token_reserve": answer.OUTPUT_TOKEN_RESERVE,
        "prompt_token_proxy": prompt_tokens,
        "provider_input_sha256": identity_sha256(provider_input),
    }
    activation = (
        "no_applicable_specialist" if ordinal == SEMANTIC_ORDINALS[0]
        else "specialist_proofless"
    )
    body = dict(row)
    body.pop("question_receipt_sha256")
    body.update(
        {
            "applicable_specialist_ids": (
                [] if activation == "no_applicable_specialist" else ["numeric_v1"]
            ),
            "classified_closure": _classified_closure(
                ordinal,
                allowed=allowed,
                binding=search_binding,
                protected_handle=protected_handle,
                protected_text=protected_text,
                residual_handle=handle,
                residual_text=exact_text,
            ),
            "fitted_typed_prompt": {
                "allowed_handle_ids": allowed,
                "handle_group_by_id": {
                    protected_handle: protected_group,
                    handle: group,
                },
                "preservation_requirements": preservation,
                "provider_input": provider_input,
                "receipt_sha256": fitted_receipt,
                "story_coherence": story,
                "validation_contract": validation,
            },
            "mode": answer.SEMANTIC_MODE,
            "semantic_search_binding": search_binding,
            "terminal_prompt": {
                "fitted_prompt_receipt_sha256": fitted_receipt,
                "full_chat_plus_output_tokens": (
                    prompt_tokens + answer.OUTPUT_TOKEN_RESERVE
                ),
                "hard_prompt_token_cap": answer.HARD_COMPLETE_CHAT_TOKEN_CAP,
                "message_renderer_format": answer.SEMANTIC_PROMPT_FORMAT,
                "messages_sha256": identity_sha256(list(messages)),
                "output_token_reserve": answer.OUTPUT_TOKEN_RESERVE,
                "prompt_token_proxy": prompt_tokens,
                "provider_input": provider_input,
                "provider_prompt_count": 0,
                "retained_transformer_token_state_bytes": 0,
                "terminal_prompt_receipt_sha256": identity_sha256(
                    terminal_receipt_body
                ),
            },
        }
    )
    return {**body, "question_receipt_sha256": identity_sha256(body)}


def _canonical_semantic_question(ordinal: int) -> dict[str, Any]:
    base = _v1_construction_question(ordinal)
    question = f"How much feed was bought in memory {ordinal}?"
    dated_question = f"[Question asked at 2026/08/28] {question}"
    parent = base["parent_source"]["prediction"]
    vector_sha = _sha(f"vectors {ordinal}")
    index_sha = _sha(f"residual index {ordinal}")
    protected_handle = f"H940{ordinal:03d}"
    residual_handle = f"H950{ordinal:03d}"
    protected_group = f"G940{ordinal:03d}"
    residual_group = f"G950{ordinal:03d}"
    allowed = [protected_handle, residual_handle]
    overlap_text = "I bought a red bowl at the market."
    residual_text = "I bought a blue mug at the market."
    overlap_segment = _sha(f"overlap segment {ordinal}")
    residual_segment = _sha(f"residual segment {ordinal}")
    overlap_cell = f"cell-overlap-{ordinal}"
    residual_cell = f"cell-residual-{ordinal}"
    overlap_binding = _sha(f"overlap local binding {ordinal}")
    residual_binding = _sha(f"residual local binding {ordinal}")

    query_body = {
        "dated_question": dated_question,
        "format": "synthetic-semantic-residual-query-v1",
        "query_vector_artifact_sha256": vector_sha,
        "residual_index_receipt_sha256": index_sha,
    }
    query = {**query_body, "receipt_sha256": identity_sha256(query_body)}
    core_body = {
        "format": "synthetic-semantic-binary-search-result-v1",
        "gold_loaded": False,
        "provider_calls_performed_by_core": 0,
        "pruned_leaf_cell_ids": [],
        "retained_leaf_cell_ids": [overlap_cell, residual_cell],
        "retained_transformer_token_state_bytes": 0,
    }
    core = {**core_body, "receipt_sha256": identity_sha256(core_body)}
    frontier_body = {
        "all_novel_survivors_protected": True,
        "certified_negative_leaf_cell_ids": [],
        "classified_leaf_count": 2,
        "closed": True,
        "complete_leaf_partition": True,
        "core_result_receipt_sha256": core["receipt_sha256"],
        "format": "memory-condense-semantic-residual-classified-frontier-v1",
        "packed_segment_receipt_sha256s": [overlap_segment, residual_segment],
        "protected_duplicate_audit_receipt_sha256s": [],
        "protected_duplicate_segment_receipt_sha256s": [],
        "residual_index_receipt_sha256": index_sha,
        "retained_leaf_cell_ids": [overlap_cell, residual_cell],
        "retained_segment_receipt_sha256s": [overlap_segment, residual_segment],
        "unresolved_segment_receipt_sha256s": [],
    }
    frontier = {
        **frontier_body,
        "receipt_sha256": identity_sha256(frontier_body),
    }

    def evidence(
        *, cell: str, segment: str, text: str, binding: str, label: str
    ) -> dict[str, Any]:
        body = {
            "candidate_id": _sha(f"candidate {label} {ordinal}"),
            "cell_id": cell,
            "citation_binding_receipt_sha256": binding,
            "contains_numeric_value": False,
            "created_at": "2026-08-01",
            "event_dates": [],
            "format": "memory-condense-semantic-residual-exact-evidence-v1",
            "matched_action_concepts": [],
            "matched_query_terms": ["bought"],
            "packing_protection": "must_include",
            "quote": text,
            "quote_sha256": _sha(text),
            "role": "user",
            "segment_receipt_sha256": segment,
            "source_group_handle": residual_group,
            "token_count": 9,
        }
        return {**body, "receipt_sha256": identity_sha256(body)}

    evidence_rows = [
        evidence(
            cell=overlap_cell,
            segment=overlap_segment,
            text=overlap_text,
            binding=overlap_binding,
            label="overlap",
        ),
        evidence(
            cell=residual_cell,
            segment=residual_segment,
            text=residual_text,
            binding=residual_binding,
            label="residual",
        ),
    ]

    def local_binding(receipt: str, label: str) -> dict[str, Any]:
        body = {"format": "synthetic-local-binding-v1", "label": label}
        assert identity_sha256(body) != receipt
        # The canonical search projection binds this externally; use the
        # actual sealed body receipt in both evidence and audit.
        return {**body, "receipt_sha256": identity_sha256(body)}

    local_rows = [
        local_binding(overlap_binding, f"overlap {ordinal}"),
        local_binding(residual_binding, f"residual {ordinal}"),
    ]
    overlap_binding = local_rows[0]["receipt_sha256"]
    residual_binding = local_rows[1]["receipt_sha256"]
    for row, binding in zip(
        evidence_rows, (overlap_binding, residual_binding), strict=True
    ):
        unsigned = dict(row)
        unsigned.pop("receipt_sha256")
        unsigned["citation_binding_receipt_sha256"] = binding
        row.clear()
        row.update({**unsigned, "receipt_sha256": identity_sha256(unsigned)})
    search_body = {
        "attempted_evidence_count": 2,
        "attempted_provider_payload_tokens": 100,
        "classified_frontier": frontier,
        "core_result": core,
        "decision_audits": [],
        "dedup_after_semantic_selection": True,
        "evidence": evidence_rows,
        "fallback_reason": "none",
        "fallback_required": False,
        "format": "memory-condense-semantic-residual-terminal-result-v1",
        "gold_loaded": False,
        "local_binding_receipt_sha256s": [overlap_binding, residual_binding],
        "new_provider_calls": 0,
        "protected_duplicates": [],
        "protected_evidence_mutated": False,
        "protected_evidence_population_receipt_sha256": _sha(
            f"empty protected population {ordinal}"
        ),
        "provider_payload_tokens": 100,
        "query_receipt_sha256": query["receipt_sha256"],
        "query_vector_artifact_sha256": vector_sha,
        "residual_index_receipt_sha256": index_sha,
        "retained_transformer_token_state_bytes": 0,
        "searched_complete_memory_population": True,
        "terminal_after_specialist_selection": True,
    }
    search = {**search_body, "receipt_sha256": identity_sha256(search_body)}
    search_audit = {
        "classified_frontier": frontier,
        "compact_result_receipt_sha256": search["receipt_sha256"],
        "local_bindings": local_rows,
        "protected_duplicates": [],
        "query": query,
    }

    overlap_item = _sha(f"overlap residual item {ordinal}")
    owner_item = _sha(f"protected owner item {ordinal}")
    residual_item = _sha(f"residual item {ordinal}")
    owner_binding = _sha(f"protected owner binding {ordinal}")
    exclusion = {
        "dedup_proof": "exact_provider_semantic_projection",
        "duplicate_binding_receipt_sha256s": [overlap_binding],
        "duplicate_item_receipt_sha256": overlap_item,
        "duplicate_mechanism_id": "semantic_residual_terminal_branch_and_bound_v1",
        "operation_position": "after_all_mechanism_selection",
        "owner_binding_receipt_sha256s": [owner_binding],
        "owner_item_receipt_sha256": owner_item,
        "owner_mechanism_id": "protected-parent",
    }
    dedup_body = {
        "exclusions": [exclusion],
        "format": "memory-condense-typed-additive-composer-v1-post-selection-dedup-v1",
        "gold_loaded": False,
        "operation_position": "after_all_mechanism_selection",
    }
    dedup = {**dedup_body, "receipt_sha256": identity_sha256(dedup_body)}
    composition_body = {
        "format": "memory-condense-typed-additive-composer-v1",
        "gold_loaded": False,
        "post_selection_dedup_audit_receipt_sha256": dedup["receipt_sha256"],
        "provider_prompt_count": 0,
        "retained_transformer_token_state_bytes": 0,
    }
    composition = {
        **composition_body,
        "receipt_sha256": identity_sha256(composition_body),
    }
    composition_audit = {
        "dropped_binding_projections": [],
        "fair_merge": {},
        "minimum_allocation": {},
        "post_selection_dedup": dedup,
        "shared_lane_surplus_fill": {},
    }
    closure_rows = [
        {
            "cell_id": overlap_cell,
            "dedup_exclusion_sha256": identity_sha256(exclusion),
            "disposition": "protected_visible_exact_duplicate",
            "exact_text_sha256": _sha(overlap_text),
            "residual_binding_receipt_sha256": overlap_binding,
            "residual_evidence_receipt_sha256": evidence_rows[0]["receipt_sha256"],
            "residual_item_receipt_sha256": overlap_item,
            "segment_receipt_sha256": overlap_segment,
            "visible_binding_receipt_sha256s": [owner_binding],
            "visible_handle_ids": [protected_handle],
            "visible_item_receipt_sha256": owner_item,
        },
        {
            "cell_id": residual_cell,
            "dedup_exclusion_sha256": None,
            "disposition": "residual_visible",
            "exact_text_sha256": _sha(residual_text),
            "residual_binding_receipt_sha256": residual_binding,
            "residual_evidence_receipt_sha256": evidence_rows[1]["receipt_sha256"],
            "residual_item_receipt_sha256": residual_item,
            "segment_receipt_sha256": residual_segment,
            "visible_binding_receipt_sha256s": [residual_binding],
            "visible_handle_ids": [residual_handle],
            "visible_item_receipt_sha256": residual_item,
        },
    ]
    protection_body = {
        "classified_frontier_receipt_sha256": frontier["receipt_sha256"],
        "format": f"{answer.CLASSIFIED_CLOSURE_FORMAT}-protection-source-v1",
        "post_selection_dedup_audit_receipt_sha256": dedup["receipt_sha256"],
        "retained_segment_receipt_sha256s": [overlap_segment, residual_segment],
        "rows": closure_rows,
        "semantic_residual_search_receipt_sha256": search["receipt_sha256"],
    }
    protection_sha = identity_sha256(protection_body)

    story = {"incompatible_group_pairs": []}
    preservation = {"by_handle": {}, "question_required_terms": []}
    validation = _validation_contract(allowed)
    provider_input = {
        "dated_question": dated_question,
        "deterministic_execution_advisory": None,
        "format": PROMPT_ROW_FORMAT,
        "protected_parent_fallback": {
            "label": "fallback_not_evidence",
            "prediction": parent,
            "prediction_sha256": _sha(parent),
        },
        "response_schema": {
            "decision": "keep_parent|replace",
            "prediction": "nonempty exact text",
            "used_handle_ids": [residual_handle],
        },
        "scalar_validation_advisory": None,
        "story_coherence": story,
        "typed_evidence": {
            "conflict_policy": "keep_both",
            "format": "synthetic-compact-semantic-evidence-v1",
            "frontier": {
                "available_handle_ids": allowed,
                "closed": True,
                "mode": "exhaustive",
                "omitted_handle_ids": [],
                "rejected_item_count": 0,
                "represented_handle_ids": allowed,
                "truncated": False,
                "unresolved_slot_ids": [],
            },
            "handles": [
                {
                    "group_handle": protected_group,
                    "handle_id": protected_handle,
                    "origin": "parent_protected",
                    "provenance_grade": "exact_citation",
                },
                {
                    "group_handle": residual_group,
                    "handle_id": residual_handle,
                    "origin": "direct_pointer",
                    "provenance_grade": "exact_citation",
                },
            ],
            "items": [
                {
                    "content_coherence": "coherent",
                    "handle_ids": [protected_handle],
                    "included": True,
                    "kind": "direct",
                    "status": "completed",
                    "summary": overlap_text,
                    "supported_slot_ids": [],
                    "value_authority": "explicit",
                },
                {
                    "content_coherence": "coherent",
                    "handle_ids": [residual_handle],
                    "included": True,
                    "kind": "direct",
                    "status": "completed",
                    "summary": residual_text,
                    "supported_slot_ids": [],
                    "value_authority": "explicit",
                },
            ],
            "operator_spec": {"operation": "single_supported_fact"},
        },
    }
    messages = list(render_final_messages(provider_input))
    messages_sha = identity_sha256(messages)
    prompt_tokens = answer.count_chat_prompt_token_proxy(messages)
    fitted_receipt = _sha(f"semantic fitted {ordinal}")
    fitted = {
        "allowed_handle_ids": allowed,
        "dropped_binding_receipt_sha256s": [],
        "dropped_item_receipt_sha256s": [],
        "execution_receipt_sha256": _sha(f"execution {ordinal}"),
        "format": PROMPT_ROW_FORMAT,
        "full_chat_plus_output_tokens": prompt_tokens + answer.OUTPUT_TOKEN_RESERVE,
        "handle_group_by_id": {
            protected_handle: protected_group,
            residual_handle: residual_group,
        },
        "hard_prompt_token_cap": answer.HARD_COMPLETE_CHAT_TOKEN_CAP,
        "local_bindings": [],
        "local_retention_priority_receipt_sha256": _sha(f"priority {ordinal}"),
        "mechanism_by_handle": {
            protected_handle: "protected-parent",
            residual_handle: "semantic-residual",
        },
        "messages_sha256": messages_sha,
        "output_token_reserve": answer.OUTPUT_TOKEN_RESERVE,
        "packet_receipt_sha256": _sha(f"packet {ordinal}"),
        "preservation_requirements": preservation,
        "prompt_token_proxy": prompt_tokens,
        "protected_binding_receipt_sha256s": [owner_binding, residual_binding],
        "protected_item_receipt_sha256s": [owner_item, residual_item],
        "protection_source_receipt_sha256": protection_sha,
        "provider_input": provider_input,
        "receipt_sha256": fitted_receipt,
        "retained_transformer_token_state_bytes": 0,
        "story_coherence": story,
        "story_link_local_bindings": [],
        "validation_contract": validation,
    }
    allowed_body = {
        "format": f"{answer.CLASSIFIED_CLOSURE_FORMAT}-terminal-allowed-handles-v1",
        "terminal_allowed_handle_ids": allowed,
    }
    closure_body = {
        "all_retained_segments_provider_visible": True,
        "classified_frontier_receipt_sha256": frontier["receipt_sha256"],
        "closed": True,
        "complete_leaf_partition": True,
        "fitted_prompt_receipt_sha256": fitted_receipt,
        "format": answer.CLASSIFIED_CLOSURE_FORMAT,
        "post_selection_dedup_audit_receipt_sha256": dedup["receipt_sha256"],
        "protection_source_receipt_sha256": protection_sha,
        "retained_segment_receipt_sha256s": [overlap_segment, residual_segment],
        "rows": closure_rows,
        "semantic_residual_search_receipt_sha256": search["receipt_sha256"],
        "terminal_allowed_handle_ids": allowed,
        "terminal_allowed_handle_ids_sha256": identity_sha256(allowed_body),
    }
    closure = {**closure_body, "receipt_sha256": identity_sha256(closure_body)}
    message_bytes = json.dumps(
        messages,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    terminal_receipt_body = {
        "fitted_prompt_receipt_sha256": fitted_receipt,
        "messages_sha256": messages_sha,
        "output_token_reserve": answer.OUTPUT_TOKEN_RESERVE,
        "prompt_token_proxy": prompt_tokens,
        "provider_input_sha256": identity_sha256(provider_input),
        "rendered_messages_utf8_byte_count": len(message_bytes),
        "rendered_messages_utf8_sha256": __import__("hashlib").sha256(
            message_bytes
        ).hexdigest(),
    }
    terminal = {
        "fitted_prompt_receipt_sha256": fitted_receipt,
        "full_chat_plus_output_tokens": prompt_tokens + answer.OUTPUT_TOKEN_RESERVE,
        "hard_prompt_token_cap": answer.HARD_COMPLETE_CHAT_TOKEN_CAP,
        "messages": messages,
        "messages_sha256": messages_sha,
        "output_token_reserve": answer.OUTPUT_TOKEN_RESERVE,
        "prompt_token_proxy": prompt_tokens,
        "provider_input": provider_input,
        "provider_prompt_count": 0,
        "rendered_messages_utf8_byte_count": len(message_bytes),
        "rendered_messages_utf8_sha256": terminal_receipt_body[
            "rendered_messages_utf8_sha256"
        ],
        "retained_transformer_token_state_bytes": 0,
        "terminal_prompt_receipt_sha256": identity_sha256(terminal_receipt_body),
    }
    body = {
        "additive_composition": composition,
        "additive_composition_local_audit": composition_audit,
        "classified_closure": closure,
        "dated_question_sha256": base["dated_question_sha256"],
        "fallback_reason": "none",
        "fitted_typed_prompt": fitted,
        "mode": answer.SEMANTIC_MODE,
        "namespace_id": base["namespace_id"],
        "new_provider_calls": 0,
        "ordinal": ordinal,
        "parent_source": base["parent_source"],
        "query_vector_artifact_sha256": vector_sha,
        "query_vector_row_receipt_sha256": _sha(f"vector row {ordinal}"),
        "question_id": base["question_id"],
        "question_sha256": base["question_sha256"],
        "retained_transformer_token_state_bytes": 0,
        "semantic_query": query,
        "semantic_residual_index_receipt_sha256": index_sha,
        "semantic_residual_local_audit": search_audit,
        "semantic_residual_search": search,
        "terminal_prompt": terminal,
    }
    return {**body, "question_receipt_sha256": identity_sha256(body)}


def _canonical_semantic_fallback_question(ordinal: int) -> dict[str, Any]:
    row = _canonical_semantic_question(ordinal)
    body = dict(row)
    body.pop("question_receipt_sha256")
    body.update(
        {
            "additive_composition": None,
            "additive_composition_local_audit": None,
            "classified_closure": None,
            "fallback_reason": "protected_semantic_residual_exceeds_terminal_cap",
            "fitted_typed_prompt": None,
            "mode": answer.PARENT_PASSTHROUGH_MODE,
            "terminal_prompt": None,
        }
    )
    return {**body, "question_receipt_sha256": identity_sha256(body)}


def _construction(
    tmp_path: Path,
) -> tuple[SealedArtifact, tuple[dict[str, Any], ...], answer.ConstructionLoader]:
    rows = tuple(
        _canonical_semantic_question(ordinal)
        if ordinal in SEMANTIC_ORDINALS
        else _canonical_semantic_fallback_question(ordinal)
        if ordinal == SEMANTIC_FALLBACK_ORDINAL
        else _v1_construction_question(ordinal)
        for ordinal in range(100)
    )
    payload = {
        "format": "synthetic-locked-semantic-final-construction-v2",
        "gold_loaded": False,
        "question_count": 100,
        "questions_sha256": identity_sha256(list(rows)),
    }
    artifact, created = publish_sealed_json(
        tmp_path / "locked-semantic-final-construction-v2.json",
        payload,
    )
    assert created

    def loader(
        path: Path,
        *,
        expected_sha256: str,
    ) -> tuple[SealedArtifact, Sequence[Mapping[str, Any]]]:
        assert path == artifact.path
        assert expected_sha256 == artifact.sha256
        return artifact, tuple(json.loads(json.dumps(rows, sort_keys=True)))

    return artifact, rows, loader


def _plans(
    tmp_path: Path,
) -> tuple[SealedArtifact, tuple[dict[str, Any], ...]]:
    artifact, _rows, loader = _construction(tmp_path)
    return answer.load_answer_plans(
        artifact.path,
        artifact.sha256,
        construction_loader=loader,
    )


def _preflight(
    tmp_path: Path,
) -> tuple[SealedArtifact, SealedArtifact, tuple[dict[str, Any], ...]]:
    construction, plans = _plans(tmp_path)
    payload = answer._preflight_projection(
        construction,
        plans,
        model=answer.DEFAULT_MODEL,
        gateway_url="https://central-dev.zt:4000/v1",
        max_concurrency=4,
    )
    preflight = SealedArtifact(
        Path("synthetic-semantic-preflight-v2.json"),
        identity_sha256(payload),
        payload,
    )
    return construction, preflight, plans


class _FakeBatch:
    def __init__(
        self,
        physical_rows: Sequence[Mapping[str, Any]],
        completions: Sequence[str] | None = None,
    ) -> None:
        rows = tuple(physical_rows)
        logical = tuple(
            completions
            or (
                json.dumps(
                    {
                        "decision": "keep_parent",
                        "prediction": row["parent_prediction"],
                        "used_handle_ids": [],
                    },
                    sort_keys=True,
                )
                for row in rows
            )
        )
        self.logical_completions = logical
        self.unique_records = tuple(
            SimpleNamespace(
                call_key_sha256=identity_sha256({"call": index}),
                checkpoint_hit=True,
                completion=completion,
                completion_sha256=_sha(completion),
                messages_sha256=row["messages_sha256"],
                physical_call=False,
                request_journal_sha256=identity_sha256({"request": index}),
                response_journal_sha256=identity_sha256({"response": index}),
            )
            for index, (row, completion) in enumerate(
                zip(rows, logical, strict=True)
            )
        )
        count = len(rows)
        self.usage = SimpleNamespace(
            checkpoint_hits=count,
            logical_calls=count,
            physical_calls=0,
            unique_calls=count,
        )

    def model_dump(self) -> dict[str, Any]:
        return {
            "logical_completions": list(self.logical_completions),
            "prompt_population": {},
            "provenance": {},
            "runtime_identity_sha256": identity_sha256({"runtime": "fake-v2"}),
            "unique_records": [vars(row) for row in self.unique_records],
            "usage": vars(self.usage),
        }


def test_v2_paths_and_public_contract_are_distinct() -> None:
    assert answer.DEFAULT_CONSTRUCTION.name == "locked-semantic-final-construction-v2.json"
    assert answer.DEFAULT_OUTPUT.name == "locked-semantic-final-answer-v2"
    assert answer.RUN_NAME == "locked-semantic-final-answer-v2.json"
    assert answer.REPLAY_NAME == "locked-semantic-final-answer-replay-v2.json"
    assert answer.FORMAT == "memory-condense-locked-semantic-final-terra-answer-v2"


def test_preflight_seals_combined_unique_population_and_mode_counts(
    tmp_path: Path,
) -> None:
    construction, plans = _plans(tmp_path)
    payload = answer._preflight_projection(
        construction,
        plans,
        model=answer.DEFAULT_MODEL,
        gateway_url="https://central-dev.zt:4000/v1",
        max_concurrency=4,
    )

    assert len(plans) == 100
    assert payload["specialist_question_count"] == 2
    assert payload["semantic_question_count"] == 2
    assert payload["parent_passthrough_count"] == 96
    assert payload["required_authorized_provider_calls"] == 4
    assert tuple(row["ordinal"] for row in payload["physical_prompt_rows"]) == (
        2,
        3,
        4,
        77,
    )
    assert payload["prompt_population"]["logical_prompt_count"] == 4
    assert payload["prompt_population"]["unique_prompt_count"] == 4
    assert payload["observed_max_complete_envelope_tokens"] <= 8_000
    assert payload["gold_loaded"] is False
    assert payload["provider_calls"] == 0
    assert payload["retained_transformer_token_state_bytes"] == 0
    assert plans[SEMANTIC_FALLBACK_ORDINAL]["fallback_reason"] == (
        "protected_semantic_residual_exceeds_terminal_cap"
    )


@pytest.mark.parametrize("mutation", ["summary", "owner", "open"])
def test_semantic_source_rejects_hidden_or_incomplete_segment_closure(
    tmp_path: Path,
    mutation: str,
) -> None:
    artifact, rows, _loader = _construction(tmp_path)
    tampered = json.loads(json.dumps(rows, sort_keys=True))
    row = tampered[SEMANTIC_ORDINALS[0]]
    closure = row["classified_closure"]
    if mutation == "summary":
        closure["rows"][0]["exact_text_sha256"] = _sha("A lossy summary.")
    elif mutation == "owner":
        closure["rows"][0]["visible_handle_ids"] = ["H999999"]
    else:
        closure["closed"] = False
    closure_body = dict(closure)
    closure_body.pop("receipt_sha256")
    closure["receipt_sha256"] = identity_sha256(closure_body)
    question_body = dict(row)
    question_body.pop("question_receipt_sha256")
    row["question_receipt_sha256"] = identity_sha256(question_body)

    def loader(path: Path, *, expected_sha256: str):
        assert path == artifact.path and expected_sha256 == artifact.sha256
        return artifact, tuple(tampered)

    with pytest.raises(answer.LockedSemanticFinalAnswerError, match="semantic"):
        answer.load_answer_plans(
            artifact.path,
            artifact.sha256,
            construction_loader=loader,
        )


def test_materialization_dispatches_both_parsers_and_preserves_parent(
    tmp_path: Path,
) -> None:
    _construction_artifact, preflight, plans = _preflight(tmp_path)
    physical = tuple(
        row
        for row in plans
        if row["mode"] in {answer.SPECIALIST_MODE, answer.SEMANTIC_MODE}
    )
    completions = tuple(
        (
            json.dumps(
                {
                    "decision": "replace",
                    "prediction": "blue mug",
                    "used_handle_ids": [f"H950{row['ordinal']:03d}"],
                },
                separators=(",", ":"),
            )
            if row["ordinal"] == SEMANTIC_ORDINALS[0]
            else "not valid JSON"
            if row["ordinal"] == SEMANTIC_ORDINALS[1]
            else json.dumps(
                {
                    "decision": "keep_parent",
                    "prediction": row["parent_prediction"],
                    "used_handle_ids": [],
                },
                sort_keys=True,
            )
        )
        for row in physical
    )
    payload = answer._materialization_projection(
        preflight,
        plans,
        _FakeBatch(physical, completions),
    )

    assert payload["question_count"] == 100
    assert payload["required_authorized_provider_calls"] == 4
    assert payload["physical_provider_calls_during_materialization"] == 0
    assert payload["parent_passthrough_count"] == 96
    assert payload["specialist_question_count"] == 2
    assert payload["semantic_question_count"] == 2
    assert payload["validated_replacement_count"] == 1
    assert payload["invalid_completion_parent_fallback_count"] == 1
    assert payload["changed_prediction_count"] == 1

    for row, projected in zip(payload["questions"], payload["judge_rows"], strict=True):
        unsigned = dict(row)
        declared = unsigned.pop("source_row_sha256")
        assert declared == identity_sha256(unsigned)
        assert projected == judge_row_projection(row)

    semantic = payload["questions"][SEMANTIC_ORDINALS[0]]
    assert semantic["prediction"] == "blue mug"
    assert semantic["completion_parser"] == "typed_final_v1"
    assert semantic["changed_from_parent"] is True
    assert semantic["solver_valid"] is True

    invalid = payload["questions"][SEMANTIC_ORDINALS[1]]
    assert invalid["prediction"] == plans[SEMANTIC_ORDINALS[1]]["parent_prediction"]
    assert invalid["changed_from_parent"] is False
    assert invalid["solver_valid"] is False

    passthrough = payload["questions"][0]
    assert passthrough["prediction"] == plans[0]["parent_prediction"]
    assert passthrough["call_key_sha256"] is None
    assert passthrough["completion_parser"] == "none"

    semantic_fallback = payload["questions"][SEMANTIC_FALLBACK_ORDINAL]
    assert semantic_fallback["prediction"] == plans[SEMANTIC_FALLBACK_ORDINAL][
        "parent_prediction"
    ]
    assert semantic_fallback["call_key_sha256"] is None
    assert semantic_fallback["completion_parser"] == "none"


def test_preflight_gold_firewall_and_byte_identical_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construction, plans = _plans(tmp_path)
    preflight_payload = answer._preflight_projection(
        construction,
        plans,
        model=answer.DEFAULT_MODEL,
        gateway_url="https://central-dev.zt:4000/v1",
        max_concurrency=4,
    )
    poisoned = json.loads(json.dumps(preflight_payload))
    poisoned["reference_answer"] = "gold must not enter"
    with pytest.raises(MatchedEvalContractError, match="gold-bearing field"):
        answer._validate_preflight(
            SealedArtifact(Path("poisoned.json"), identity_sha256(poisoned), poisoned)
        )

    preflight, created = publish_sealed_json(
        tmp_path / answer.PREFLIGHT_NAME,
        preflight_payload,
    )
    assert created
    physical = tuple(
        row
        for row in plans
        if row["mode"] in {answer.SPECIALIST_MODE, answer.SEMANTIC_MODE}
    )
    batch = _FakeBatch(physical)
    run_payload = answer._materialization_projection(preflight, plans, batch)
    run, created = publish_sealed_json(tmp_path / answer.RUN_NAME, run_payload)
    assert created

    monkeypatch.setattr(
        answer,
        "load_answer_plans",
        lambda path, expected_sha256: (construction, plans),
    )
    monkeypatch.setattr(answer, "_checkpoint_batch", lambda *args, **kwargs: batch)
    args = SimpleNamespace(
        construction=construction.path,
        expected_construction_sha256=construction.sha256,
        expected_preflight_sha256=preflight.sha256,
        expected_run_sha256=run.sha256,
        gateway_url="https://central-dev.zt:4000/v1",
        max_concurrency=4,
        model=answer.DEFAULT_MODEL,
        output_root=tmp_path,
    )
    result = answer.run_replay(args)

    assert result["byte_identical"] is True
    assert result["physical_provider_calls"] == 0
    assert result["run_sha256"] == result["replay_sha256"] == run.sha256
