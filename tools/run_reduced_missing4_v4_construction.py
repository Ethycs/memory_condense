#!/usr/bin/env python3
"""Build the provider-free reduced missing-four v4 operator stack.

The construction phase is deliberately gold-blind.  It verifies the sealed
official parent and sealed semantic-v3 inputs, then applies one question-only
operator per residual shape:

* conjunctive same-event sufficiency for ordinal 42 (fail closed while its
  support frontier remains open),
* post-selection action-linked SET compression for ordinal 65,
* the complete sealed two-item semantic residual lane for ordinal 74, and
* the temporal specialist, routed from the typed ``latest_state`` contract,
  for ordinal 79.

No provider is called and no transformer token state is retained.  The audit
phase validates the sealed construction before it opens the post-hoc target
registry and reports both selection and terminal source survival.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from memory_condense.persistence.db import Database  # noqa: E402
from tools import run_locked_specialist_final_construction as parent_cli  # noqa: E402
from tools import run_reduced_second_read_retrieval_assay as reduced_cli  # noqa: E402
from tools import run_reduced_semantic_binary_search_assay as semantic_cli  # noqa: E402
from tools import run_reduced_specialist_answer_v2 as answer_v2_cli  # noqa: E402
from tools import run_reduced_specialist_retrieval_assay as specialist_cli  # noqa: E402
from tools.matched_eval import conjunctive_event_sufficiency as event_op  # noqa: E402
from tools.matched_eval import semantic_residual_search as residual  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.full_store_slot_closure import (  # noqa: E402
    build_full_store_window_index,
)
from tools.matched_eval.post_selection_action_set_compressor import (  # noqa: E402
    MECHANISM_ID as ACTION_SET_MECHANISM_ID,
    compile_action_linked_set_demand,
    compress_selected_typed_action_set_evidence,
)
from tools.matched_eval.protected_parent_contribution import (  # noqa: E402
    ProtectedParentContributionSet,
    rehydrate_protected_parent_contributions,
)
from tools.matched_eval.query_guided_scan import (  # noqa: E402
    cache_namespace_partitions,
)
from tools.matched_eval.temporal_insufficiency_specialist import (  # noqa: E402
    MECHANISM_ID as TEMPORAL_MECHANISM_ID,
    TemporalInsufficiencyResult,
)
from tools.matched_eval.typed_action_semantics import (  # noqa: E402
    completed_action_concepts,
)
from tools.matched_eval.typed_downstream_operator import (  # noqa: E402
    compile_downstream_operator_overlay,
    execute_downstream_typed_operator,
)
from tools.matched_eval.typed_memory_final_arm import (  # noqa: E402
    fit_typed_final_prompt,
    render_final_messages,
)
from tools.matched_eval.typed_operator_adapter import (  # noqa: E402
    FrontierMode,
    ParsedTypedItems,
    ProviderPayloadMode,
    TypedEvidenceContribution,
    TypedEvidenceItem,
    build_typed_evidence_packet,
    parse_typed_items,
)
from tools.matched_eval.typed_operator_spec import (  # noqa: E402
    compile_typed_operator_spec,
    normalized_terms,
)


FORMAT = "memory-condense-reduced-missing4-operator-stack-v4"
CONSTRUCTION_FORMAT = f"{FORMAT}-construction-v1"
AUDIT_FORMAT = f"{FORMAT}-posthoc-target-audit-v1"
QUESTION_FORMAT = f"{FORMAT}-question-v1"
METHOD_FORMAT = f"{FORMAT}-selected-method-v1"
SELECTION_FORMAT = f"{FORMAT}-selection-v1"
PROVENANCE_FORMAT = f"{FORMAT}-local-provenance-v1"
INDEX_FORMAT = f"{FORMAT}-resident-index-v1"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TARGET_ORDINALS = (42, 65, 74, 79)
QUESTION_COUNT = len(TARGET_ORDINALS)
HARD_COMPLETE_CHAT_TOKEN_CAP = 8_000
OUTPUT_TOKEN_RESERVE = 768

CONSTRUCTION_NAME = "reduced-missing4-operator-stack-construction-v4.json"
AUDIT_NAME = "reduced-missing4-operator-stack-target-audit-v4.json"
DEFAULT_SEMANTIC_CONSTRUCTION = (
    REPOSITORY_ROOT
    / "eval_results/matched_eval_100/reduced-semantic-binary-search-missing4-v3"
    / semantic_cli.CONSTRUCTION_NAME
)
EXPECTED_SEMANTIC_CONSTRUCTION_SHA256 = (
    "cb6c0e2c66be18039dbb6f246f333d909fd18f40e81231f0fbf167ebc55dfbc8"
)
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/reduced-missing4-operator-stack-v4"
)
DEFAULT_TARGET_PLAN = reduced_cli.DEFAULT_TARGET_PLAN

class ReducedMissing4V4Error(MatchedEvalContractError):
    """Raised when a sealed input, operator, prompt, or audit invariant changes."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ReducedMissing4V4Error(message)


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact list")
    return value  # type: ignore[return-value]


def _with_receipt(body: Mapping[str, Any], key: str = "receipt_sha256") -> dict[str, Any]:
    value = dict(body)
    value[key] = identity_sha256(body)
    return value


def _validate_receipt(value: object, *, label: str, key: str = "receipt_sha256") -> dict[str, Any]:
    row = _exact_dict(value, label)
    body = dict(row)
    declared = require_sha256(body.pop(key, None), label)
    _require(identity_sha256(body) == declared, f"{label} receipt changed")
    return row


def _question_inputs(row: Mapping[str, Any]) -> tuple[str, str, str]:
    provider = _exact_dict(row.get("provider_projection"), "parent provider")
    provider_input = _exact_dict(provider.get("provider_input"), "parent provider input")
    return (
        require_text(provider_input.get("dated_question"), "dated question"),
        require_text(row.get("parent_prediction"), "parent prediction"),
        require_text(row.get("question_id"), "question ID"),
    )


def _ordered_parent_items(parent: ProtectedParentContributionSet) -> tuple[TypedEvidenceItem, ...]:
    by_receipt = {
        item.receipt_sha256: item
        for contribution in parent.contributions
        for item in contribution.parsed.accepted_items
    }
    _require(
        set(by_receipt) == set(parent.audit.compact_item_receipt_order),
        "protected parent item inventory changed",
    )
    return tuple(by_receipt[value] for value in parent.audit.compact_item_receipt_order)


def _exact_parent_lane(
    parent: ProtectedParentContributionSet,
) -> tuple[tuple[TypedEvidenceItem, ...], dict[str, Any]]:
    bindings = {
        binding.handle_id: binding
        for contribution in parent.contributions
        for binding in contribution.bindings
    }
    exact: list[TypedEvidenceItem] = []
    for item in _ordered_parent_items(parent):
        if len(item.handle_ids) != 1:
            continue
        binding = bindings[item.handle_ids[0]]
        digest = quote_sha256(item.summary)
        if (
            binding.payload_sha256 == digest
            and binding.citation_sha256 == digest
            and binding.citation_char_count == len(item.summary)
        ):
            exact.append(item)
    return tuple(exact), bindings


def _is_user_item(item: TypedEvidenceItem) -> bool:
    relation = (item.relation or "").casefold()
    return "memory_role:user" in relation or "authored_by_user" in relation


def _subset_parsed(items: tuple[TypedEvidenceItem, ...], *, label: str) -> ParsedTypedItems:
    _require(bool(items), f"{label} cannot be empty")
    receipt = identity_sha256(
        {
            "accepted_item_receipt_sha256s": [row.receipt_sha256 for row in items],
            "format": f"{FORMAT}-{label}-parsed-subset-v1",
            "rejected_item_receipt_sha256s": [],
        }
    )
    return ParsedTypedItems(items, (), receipt)


def _source_ids_by_parent_handle(
    parent: ProtectedParentContributionSet,
) -> dict[str, tuple[str, ...]]:
    return {
        row.handle_id: row.source_ids for row in parent.audit.source_provenance
    }


def _provenance_row(
    *,
    binding: Any,
    source_ids: Sequence[str],
    local_evidence: Mapping[str, Any] | None = None,
    parent_lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ordered_sources = tuple(dict.fromkeys(source_ids))
    _require(bool(ordered_sources), "local provenance requires a source")
    body: dict[str, Any] = {
        "format": PROVENANCE_FORMAT,
        "handle_id": binding.handle_id,
        "source_ids": list(ordered_sources),
        "typed_binding": binding.projection(),
    }
    if local_evidence is not None:
        body["local_evidence"] = dict(local_evidence)
    if parent_lineage is not None:
        body["parent_lineage"] = dict(parent_lineage)
    return _with_receipt(body)


def _parent_provenance_rows(
    parent: ProtectedParentContributionSet,
    handle_ids: Sequence[str],
) -> list[dict[str, Any]]:
    by_handle = {row.handle_id: row for row in parent.audit.source_provenance}
    rows: list[dict[str, Any]] = []
    for handle in handle_ids:
        lineage = by_handle.get(handle)
        _require(lineage is not None, "selected parent handle lost provenance")
        assert lineage is not None
        rows.append(
            _provenance_row(
                binding=lineage.cloned_binding,
                source_ids=lineage.source_ids,
                parent_lineage=lineage.projection(),
            )
        )
    return rows


def _source_ids_for_handles(
    provenance: Sequence[Mapping[str, Any]],
    handles: Sequence[str],
) -> list[str]:
    by_handle = {
        require_text(row.get("handle_id"), "provenance handle"): tuple(
            require_text(value, "provenance source")
            for value in _exact_list(row.get("source_ids"), "provenance sources")
        )
        for row in provenance
    }
    result: list[str] = []
    for handle in handles:
        _require(handle in by_handle, "handle escaped local provenance")
        for source in by_handle[handle]:
            if source not in result:
                result.append(source)
    return result


def _selection_projection(
    *,
    mode: str,
    mechanism_ids: Sequence[str],
    items: Sequence[TypedEvidenceItem],
    bindings: Sequence[Any],
    provenance: Sequence[Mapping[str, Any]],
    selection_precedes_compression: bool = False,
) -> dict[str, Any]:
    handle_ids = tuple(binding.handle_id for binding in bindings)
    _require(
        bool(handle_ids)
        and len(set(handle_ids)) == len(handle_ids)
        and all(set(item.handle_ids) <= set(handle_ids) for item in items),
        "selection item/binding coverage changed",
    )
    body = {
        "format": SELECTION_FORMAT,
        "mechanism_ids": list(mechanism_ids),
        "mode": mode,
        "selected_binding_receipt_sha256s": [row.receipt_sha256 for row in bindings],
        "selected_handle_ids": list(handle_ids),
        "selected_item_receipt_sha256s": [row.receipt_sha256 for row in items],
        "selected_source_ids": _source_ids_for_handles(provenance, handle_ids),
        "selection_precedes_post_selection_compression": (
            selection_precedes_compression
        ),
    }
    return _with_receipt(body)


def _selected_method_projection(
    *,
    mechanism_id: str,
    items: Sequence[TypedEvidenceItem],
    bindings: Sequence[Any],
    provenance: Sequence[Mapping[str, Any]],
    source_receipt_sha256: str,
) -> dict[str, Any]:
    body = {
        "accepted_typed_item_count": len(items),
        "format": METHOD_FORMAT,
        "local_provenance_receipt_sha256s": [
            row["receipt_sha256"] for row in provenance
        ],
        "mechanism_id": mechanism_id,
        "new_provider_calls": 0,
        "retained_transformer_token_state_bytes": 0,
        "selected_bindings": [row.projection() for row in bindings],
        "selected_items": [row.projection() for row in items],
        "source_receipt_sha256": require_sha256(
            source_receipt_sha256, "selected method source"
        ),
    }
    return _with_receipt(body, "method_receipt_sha256")


def _terminal_projection(
    *,
    fitted: Any,
    specialist_advisories: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Emit the generic answer-v2 terminal schema for every v4 row."""

    advisories = tuple(dict(row) for row in specialist_advisories)
    _require(bool(advisories), "v4 terminal requires a scoped advisory")
    terminal = specialist_cli._terminal_projection(  # noqa: SLF001
        provider_input=fitted.provider_input,
        specialist_advisories=advisories,
        fitted_prompt_receipt_sha256=fitted.receipt_sha256,
        message_renderer=render_final_messages,
    )
    terminal_input = _exact_dict(
        terminal.get("provider_input"), "v4 terminal provider input"
    )
    _require(
        terminal_input
        == {
            **dict(fitted.provider_input),
            "specialist_advisories": list(advisories),
        }
        and terminal.get("provider_prompt_count") == 0
        and terminal.get("retained_transformer_token_state_bytes") == 0
        and terminal.get("hard_prompt_token_cap")
        == HARD_COMPLETE_CHAT_TOKEN_CAP
        and terminal.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
        and terminal.get("full_chat_plus_output_tokens")
        == terminal.get("prompt_token_proxy") + OUTPUT_TOKEN_RESERVE
        and terminal["full_chat_plus_output_tokens"]
        <= HARD_COMPLETE_CHAT_TOKEN_CAP,
        "v4 generic provider terminal escaped its sealed 8k envelope",
    )
    return terminal


def _question_shell(
    *,
    ordinal: int,
    row: Mapping[str, Any],
    namespace_id: str,
    mechanism_plan: Sequence[str],
    selection: Mapping[str, Any],
    methods: Sequence[Mapping[str, Any]],
    operator: Mapping[str, Any],
    provenance: Sequence[Mapping[str, Any]],
    fitted_typed_prompt: Mapping[str, Any],
    terminal_kind: str,
    terminal_prompt: Mapping[str, Any],
) -> dict[str, Any]:
    dated_question, _parent_prediction, question_id = _question_inputs(row)
    body: dict[str, Any] = {
        "dated_question_sha256": quote_sha256(dated_question),
        "fitted_typed_prompt": dict(fitted_typed_prompt),
        "format": QUESTION_FORMAT,
        "local_provenance": [dict(value) for value in provenance],
        "mechanism_plan": list(mechanism_plan),
        "methods": [dict(value) for value in methods],
        "namespace_id": require_sha256(namespace_id, "question namespace"),
        "new_provider_calls": 0,
        "operator": dict(operator),
        "ordinal": ordinal,
        "question_id": question_id,
        "question_sha256": require_sha256(row.get("question_sha256"), "question"),
        "retained_transformer_token_state_bytes": 0,
        "selection": dict(selection),
        "terminal_kind": require_text(terminal_kind, "terminal kind"),
        "terminal_prompt": dict(terminal_prompt),
    }
    assert_gold_blind(body, path=f"reduced_missing4_v4_question_{ordinal}")
    return {**body, "question_receipt_sha256": identity_sha256(body)}


def _q42_question(
    *,
    row: Mapping[str, Any],
    composition_sha256: str,
    namespace_id: str,
) -> dict[str, Any]:
    dated_question, parent_prediction, _question_id = _question_inputs(row)
    spec = compile_typed_operator_spec(dated_question)
    parent = rehydrate_protected_parent_contributions(row, spec, composition_sha256)
    exact_items, binding_by_handle = _exact_parent_lane(parent)
    overlay = event_op.compile_conjunctive_event_obligation_overlay(dated_question)
    _require(overlay is not None, "q42 event program did not compile")
    assert overlay is not None
    obligation_terms = {
        term
        for obligation in overlay.obligations
        if not obligation.answer_variable
        for term in normalized_terms(obligation.required_value or "")
        if term != "user"
    }
    selected_items = tuple(
        item
        for item in exact_items
        if _is_user_item(item)
        and len(obligation_terms & set(normalized_terms(item.summary))) >= 2
    )
    selected_bindings = tuple(
        binding_by_handle[item.handle_ids[0]] for item in selected_items
    )
    _require(
        len(selected_items) >= 2
        and len({row.sealed_artifact_sha256 for row in selected_bindings}) == 1,
        "q42 question-only exact lane changed",
    )
    population_sha = identity_sha256(
        {
            "format": f"{FORMAT}-q42-open-support-population-v1",
            "item_receipt_sha256s": [row.receipt_sha256 for row in selected_items],
            "question_sha256": spec.question_sha256,
        }
    )
    # Packing is closed over the terminal subset, but semantic support remains
    # explicitly open.  This prevents the known target pair from authorizing
    # an outcome-conditioned absence conclusion.
    decision = event_op.decide_typed_conjunctive_event(
        dated_question,
        selected_items,
        selected_bindings,
        population_identity_sha256=population_sha,
        parent_hypothesis=parent_prediction,
        packing_closed=True,
        support_enumerated_handle_ids=(),
    )
    _require(
        decision.disposition is event_op.EventDecisionDisposition.KEEP_PARENT
        and decision.terminal_authorized is False
        and decision.reason
        in {
            event_op.EventDecisionReason.SUPPORT_OPEN_EVENT_UNRESOLVED,
            event_op.EventDecisionReason.SUPPORT_OPEN_EVENT_CONFLICT,
        },
        "q42 open support incorrectly authorized a conclusion",
    )
    parsed = _subset_parsed(selected_items, label="q42-event")
    contribution = TypedEvidenceContribution(
        event_op.MECHANISM_ID,
        selected_bindings,
        parsed,
        selected_bindings[0].sealed_artifact_sha256,
        FrontierMode.BOUNDED,
        False,
    )
    packet = build_typed_evidence_packet(
        spec,
        selected_bindings,
        parsed,
        sealed_input_artifact_sha256s=(contribution.sealed_artifact_sha256,),
        frontier_mode=FrontierMode.BOUNDED,
        truncated=False,
        provider_payload_mode=ProviderPayloadMode.COMPACT_FINAL,
    )
    fitted = fit_typed_final_prompt(
        dated_question=dated_question,
        parent_prediction=parent_prediction,
        packet=packet,
        mechanism_by_handle={
            binding.handle_id: event_op.MECHANISM_ID
            for binding in selected_bindings
        },
        minimum_usable_items_per_mechanism=1,
        protected_item_receipt_sha256s=tuple(
            item.receipt_sha256 for item in selected_items
        ),
        protection_source_receipt_sha256=contribution.receipt_sha256,
    )
    advisory = {
        "conjunctive_event_decision_state": decision.projection(),
        "conjunctive_event_program": overlay.projection(),
        "mechanism_id": event_op.MECHANISM_ID,
        "proof_kind": "same_event_conjunctive_obligation",
        "purpose": (
            "require every question edge on one proven event identity component; "
            "separate compatible memories do not prove the join"
        ),
        "support_frontier": {
            "cross_source_compatibility_proves_identity": False,
            "deterministic_abstention_authorized": False,
            "generic_frontier_closed": False,
            "semantic_absence_may_be_inferred": False,
            "scope": "open_full_store_support",
        },
    }
    terminal_prompt = _terminal_projection(
        fitted=fitted,
        specialist_advisories=(advisory,),
    )
    provenance = _parent_provenance_rows(
        parent, tuple(binding.handle_id for binding in selected_bindings)
    )
    selection = _selection_projection(
        mode="question_only_conjunctive_event_candidate_selection",
        mechanism_ids=(event_op.MECHANISM_ID,),
        items=selected_items,
        bindings=selected_bindings,
        provenance=provenance,
    )
    method = _selected_method_projection(
        mechanism_id=event_op.MECHANISM_ID,
        items=selected_items,
        bindings=selected_bindings,
        provenance=provenance,
        source_receipt_sha256=parent.audit.receipt_sha256,
    )
    operator = _with_receipt(
        {
            "decision": decision.projection(),
            "format": f"{FORMAT}-q42-conjunctive-event-v1",
            "packing_closed": True,
            "population_identity_sha256": population_sha,
            "provider_prompt_count": 0,
            "retained_transformer_token_state_bytes": 0,
            "semantic_absence_may_be_inferred": False,
            "support_frontier_closed": False,
        },
        "operator_receipt_sha256",
    )
    return _question_shell(
        ordinal=42,
        row=row,
        namespace_id=namespace_id,
        mechanism_plan=(event_op.MECHANISM_ID,),
        selection=selection,
        methods=(method,),
        operator=operator,
        provenance=provenance,
        fitted_typed_prompt=fitted.projection(),
        terminal_kind="conjunctive_event_synthesis",
        terminal_prompt=terminal_prompt,
    )


def _q65_question(
    *,
    row: Mapping[str, Any],
    composition_sha256: str,
    namespace_id: str,
) -> dict[str, Any]:
    dated_question, parent_prediction, _question_id = _question_inputs(row)
    spec = compile_typed_operator_spec(dated_question)
    overlay = compile_downstream_operator_overlay(dated_question, spec)
    demand = compile_action_linked_set_demand(dated_question, spec, overlay)
    parent = rehydrate_protected_parent_contributions(row, spec, composition_sha256)
    exact_items, binding_by_handle = _exact_parent_lane(parent)
    selected_items = tuple(
        item
        for item in exact_items
        if _is_user_item(item)
        and set(demand.relation_anchor_terms) <= set(normalized_terms(item.summary))
        and set(demand.action_concepts) & set(completed_action_concepts(item.summary))
    )
    selected_bindings = tuple(
        binding_by_handle[item.handle_ids[0]] for item in selected_items
    )
    selection_receipt = identity_sha256(
        {
            "format": f"{FORMAT}-q65-question-only-selection-v1",
            "item_receipt_sha256s": [row.receipt_sha256 for row in selected_items],
            "question_sha256": spec.question_sha256,
        }
    )
    compression = compress_selected_typed_action_set_evidence(
        demand,
        selected_items,
        selected_bindings,
        spec,
        selection_receipt_sha256=selection_receipt,
    )
    _require(
        len(compression.facts) == demand.cardinality
        and compression.closure.explicit_cardinality_satisfied
        and tuple(binding.handle_id for binding in compression.bindings)
        == tuple(binding.handle_id for binding in selected_bindings),
        "q65 selected-scope action-set witness changed",
    )
    # The compact SET lane replaces the raw selected lane.  Its two exact facts
    # satisfy the selected-scope cardinality demand, but the upstream full-store
    # selector was truncated.  Reconstruct the terminal contribution explicitly
    # as BOUNDED/truncated and never merge the compressor's raw selected lane.
    bounded_contribution = TypedEvidenceContribution(
        ACTION_SET_MECHANISM_ID,
        compression.bindings,
        compression.parsed,
        compression.bindings[0].sealed_artifact_sha256,
        FrontierMode.BOUNDED,
        True,
    )
    packet = build_typed_evidence_packet(
        spec,
        bounded_contribution.bindings,
        bounded_contribution.parsed,
        sealed_input_artifact_sha256s=(
            bounded_contribution.sealed_artifact_sha256,
        ),
        frontier_mode=FrontierMode.BOUNDED,
        truncated=True,
        provider_payload_mode=ProviderPayloadMode.COMPACT_FINAL,
    )
    execution = execute_downstream_typed_operator(spec, packet, overlay)
    _require(
        execution.status.value == "insufficient"
        and not execution.used_handle_ids,
        "q65 bounded generic executor incorrectly upgraded selected scope",
    )
    fitted = fit_typed_final_prompt(
        dated_question=dated_question,
        parent_prediction=parent_prediction,
        packet=packet,
        mechanism_by_handle={
            binding.handle_id: ACTION_SET_MECHANISM_ID
            for binding in bounded_contribution.bindings
        },
        minimum_usable_items_per_mechanism=1,
        protected_item_receipt_sha256s=tuple(
            item.receipt_sha256 for item in bounded_contribution.parsed.accepted_items
        ),
        protection_source_receipt_sha256=bounded_contribution.receipt_sha256,
    )
    fact_handles = tuple(
        dict.fromkeys(
            handle_id
            for fact in compression.facts
            for handle_id in fact.handle_ids
        )
    )
    _require(
        set(fact_handles) == set(fitted.allowed_handle_ids),
        "q65 terminal fit dropped a selected action-set fact",
    )
    scoped_prediction = " and ".join(
        fact.member_text for fact in compression.facts
    )
    provider_facts = compression.provider_projection()["facts"]
    advisory = {
        "cardinality": demand.cardinality,
        "facts": provider_facts,
        "generic_frontier_closed": False,
        "mechanism_id": ACTION_SET_MECHANISM_ID,
        "prediction": scoped_prediction,
        "proof_kind": "selected_scope_action_member_cardinality",
        "purpose": (
            "answer from the exact action-linked members selected before "
            "deduplication; do not infer global absence or completeness"
        ),
        "scope": "selected_action_linked_members_only",
        "selected_scope_cardinality_satisfied": True,
        "semantic_absence_may_be_inferred": False,
        "status": "selected_scope_supported",
        "upstream_frontier_mode": FrontierMode.BOUNDED.value,
        "upstream_truncated": True,
        "used_handle_ids": list(fact_handles),
    }
    terminal_prompt = _terminal_projection(
        fitted=fitted,
        specialist_advisories=(advisory,),
    )
    provenance = _parent_provenance_rows(
        parent, tuple(binding.handle_id for binding in selected_bindings)
    )
    selection = _selection_projection(
        mode="question_only_action_relation_selection_before_compression",
        mechanism_ids=(ACTION_SET_MECHANISM_ID,),
        items=selected_items,
        bindings=selected_bindings,
        provenance=provenance,
        selection_precedes_compression=True,
    )
    method = _selected_method_projection(
        mechanism_id=ACTION_SET_MECHANISM_ID,
        items=compression.parsed.accepted_items,
        bindings=compression.bindings,
        provenance=provenance,
        source_receipt_sha256=compression.receipt_sha256,
    )
    operator = _with_receipt(
        {
            "compression": compression.projection(),
            "downstream_overlay": overlay.projection(),
            "execution": execution.projection(),
            "format": f"{FORMAT}-q65-action-set-v1",
            "generic_frontier_closed": False,
            "provider_prompt_count": 0,
            "raw_selected_lane_merged_after_compression": False,
            "retained_transformer_token_state_bytes": 0,
            "selected_scope_cardinality_satisfied": True,
            "selected_scope_only": True,
            "terminal_typed_contribution": bounded_contribution.projection(),
            "upstream_truncated": True,
        },
        "operator_receipt_sha256",
    )
    return _question_shell(
        ordinal=65,
        row=row,
        namespace_id=namespace_id,
        mechanism_plan=(ACTION_SET_MECHANISM_ID,),
        selection=selection,
        methods=(method,),
        operator=operator,
        provenance=provenance,
        fitted_typed_prompt=fitted.projection(),
        terminal_kind="selected_scope_action_set_synthesis",
        terminal_prompt=terminal_prompt,
    )


def _clean_compact_item(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in raw.items()
        if key not in {"content_coherence", "supported_slot_ids"}
    }


def _q74_question(
    *,
    row: Mapping[str, Any],
    semantic_row: Mapping[str, Any],
    semantic_artifact_sha256: str,
    namespace_id: str,
) -> dict[str, Any]:
    dated_question, parent_prediction, _question_id = _question_inputs(row)
    _require(
        semantic_row.get("ordinal") == 74
        and semantic_row.get("mode") == "semantic_residual"
        and semantic_row.get("namespace_id") == namespace_id,
        "q74 semantic source row changed",
    )
    spec = compile_typed_operator_spec(dated_question)
    terminal = _exact_dict(semantic_row.get("terminal_prompt"), "q74 semantic terminal")
    provider_input = _exact_dict(terminal.get("provider_input"), "q74 semantic provider")
    compact = _exact_dict(provider_input.get("typed_evidence"), "q74 semantic typed evidence")
    raw_items = tuple(
        _exact_dict(value, "q74 compact item")
        for value in _exact_list(compact.get("items"), "q74 compact items")
        if all(str(handle).startswith("H950") for handle in value.get("handle_ids", ()))
    )
    fitted = _exact_dict(semantic_row.get("fitted_typed_prompt"), "q74 fitted source")
    bindings = tuple(
        reduced_cli._rehydrate_handle_binding(value)  # noqa: SLF001
        for value in _exact_list(fitted.get("local_bindings"), "q74 fitted bindings")
        if str(_exact_dict(value, "q74 fitted binding").get("handle_id")).startswith("H950")
    )
    _require(
        tuple(binding.handle_id for binding in bindings) == ("H950001", "H950002")
        and tuple(item.get("handle_ids") for item in raw_items)
        == (["H950001"], ["H950002"])
        and len({binding.sealed_artifact_sha256 for binding in bindings}) == 1,
        "q74 complete residual lane changed order or identity",
    )
    local_audit = _exact_dict(
        semantic_row.get("semantic_residual_local_audit"), "q74 semantic local audit"
    )
    local_bindings = tuple(
        _exact_dict(value, "q74 local citation")
        for value in _exact_list(local_audit.get("local_bindings"), "q74 local citations")
    )
    attempts = {
        require_sha256(value.get("local_binding_receipt_sha256"), "q74 attempt local"): value
        for value in (
            _exact_dict(row, "q74 attempted selection")
            for row in _exact_list(
                local_audit.get("attempted_selection"), "q74 attempted selections"
            )
        )
    }
    _require(len(local_bindings) == len(bindings), "q74 local lane lost a binding")
    provenance: list[dict[str, Any]] = []
    for raw_item, binding, local in zip(raw_items, bindings, local_bindings, strict=True):
        summary = require_text(raw_item.get("summary"), "q74 residual summary")
        attempt = attempts.get(binding.local_source_locator_sha256)
        _require(
            local.get("receipt_sha256") == binding.local_source_locator_sha256
            and local.get("quote_sha256") == binding.citation_sha256
            and binding.citation_sha256 == quote_sha256(summary)
            and binding.citation_char_count == len(summary)
            and attempt is not None
            and attempt.get("evidence_receipt_sha256")
            == binding.evidence_receipt_sha256
            and attempt.get("source_id") == local.get("source_id"),
            "q74 exact item/binding/local lineage changed",
        )
        provenance.append(
            _provenance_row(
                binding=binding,
                source_ids=(require_text(local.get("source_id"), "q74 source"),),
                local_evidence=local,
            )
        )
    parsed = parse_typed_items(
        [_clean_compact_item(value) for value in raw_items],
        operator_spec=spec,
        bindings=bindings,
    )
    _require(
        not parsed.rejected_items and len(parsed.accepted_items) == len(bindings),
        "q74 complete residual lane failed typed rehydration",
    )
    contribution = TypedEvidenceContribution(
        residual.TYPED_ADAPTER_MECHANISM_ID,
        bindings,
        parsed,
        bindings[0].sealed_artifact_sha256,
        FrontierMode.BOUNDED,
        False,
    )
    packet = build_typed_evidence_packet(
        spec,
        bindings,
        parsed,
        sealed_input_artifact_sha256s=(contribution.sealed_artifact_sha256,),
        frontier_mode=FrontierMode.BOUNDED,
        truncated=False,
        provider_payload_mode=ProviderPayloadMode.COMPACT_FINAL,
    )
    final_fit = fit_typed_final_prompt(
        dated_question=dated_question,
        parent_prediction=parent_prediction,
        packet=packet,
        mechanism_by_handle={
            binding.handle_id: residual.TYPED_ADAPTER_MECHANISM_ID
            for binding in bindings
        },
        minimum_usable_items_per_mechanism=1,
        protected_item_receipt_sha256s=tuple(
            item.receipt_sha256 for item in parsed.accepted_items
        ),
        protection_source_receipt_sha256=contribution.receipt_sha256,
    )
    _require(
        final_fit.allowed_handle_ids == ("H950001", "H950002"),
        "q74 terminal fit dropped part of the complete residual lane",
    )
    advisory = {
        "frontier_mode": FrontierMode.BOUNDED.value,
        "global_exhaustiveness_claimed": False,
        "lane_handle_ids": list(final_fit.allowed_handle_ids),
        "mechanism_id": residual.TYPED_ADAPTER_MECHANISM_ID,
        "proof_kind": "sealed_semantic_residual_lane",
        "purpose": (
            "read the complete sealed two-item residual lane as substitute "
            "context, including the question-bearing item and answer-bearing item"
        ),
        "sealed_lane_complete": True,
        "truncated": False,
    }
    terminal_prompt = _terminal_projection(
        fitted=final_fit,
        specialist_advisories=(advisory,),
    )
    selection = _selection_projection(
        mode="sealed_complete_semantic_residual_lane_rehydration",
        mechanism_ids=(residual.TYPED_ADAPTER_MECHANISM_ID,),
        items=parsed.accepted_items,
        bindings=bindings,
        provenance=provenance,
    )
    method_body = {
        "accepted_typed_item_count": len(parsed.accepted_items),
        "format": METHOD_FORMAT,
        "frontier_mode": FrontierMode.BOUNDED.value,
        "local_provenance_receipt_sha256s": [row["receipt_sha256"] for row in provenance],
        "mechanism_id": residual.TYPED_ADAPTER_MECHANISM_ID,
        "new_provider_calls": 0,
        "retained_transformer_token_state_bytes": 0,
        "sealed_semantic_construction_sha256": semantic_artifact_sha256,
        "source_search_receipt_sha256": bindings[0].sealed_artifact_sha256,
        "stored_search_mechanism_id": residual.MECHANISM_ID,
        "truncated": False,
        "typed_contribution": contribution.projection(),
    }
    method = _with_receipt(method_body, "method_receipt_sha256")
    operator = _with_receipt(
        {
            "format": f"{FORMAT}-q74-semantic-bridge-v1",
            "frontier_mode": FrontierMode.BOUNDED.value,
            "provider_prompt_count": 0,
            "retained_transformer_token_state_bytes": 0,
            "semantic_global_exhaustiveness_claimed": False,
            "truncated": False,
            "typed_contribution_receipt_sha256": contribution.receipt_sha256,
        },
        "operator_receipt_sha256",
    )
    return _question_shell(
        ordinal=74,
        row=row,
        namespace_id=namespace_id,
        mechanism_plan=(residual.TYPED_ADAPTER_MECHANISM_ID,),
        selection=selection,
        methods=(method,),
        operator=operator,
        provenance=provenance,
        fitted_typed_prompt=final_fit.projection(),
        terminal_kind="semantic_residual_synthesis",
        terminal_prompt=terminal_prompt,
    )


def _verified_resident_index(
    args: argparse.Namespace,
    *,
    namespace_id: str,
    question_id: str,
    closure: SealedArtifact,
) -> tuple[Any, dict[str, Any]]:
    guided = reduced_cli._guided_args(args)  # noqa: SLF001
    context = reduced_cli._scoped_guided_context(guided, namespace_id)  # noqa: SLF001
    prompt = context.prompt_rows_by_question.get(question_id)
    _require(
        prompt is not None and prompt.namespace.namespace_id == namespace_id,
        "v4 question changed locked namespace ownership",
    )
    with Database(context.store_dir / "memory.db", read_only=True) as database:
        cache = cache_namespace_partitions(
            database,
            context.namespace,
            source_database_sha256=context.database_sha256,
            source_store_receipt_sha256=(
                context.namespace.combined_store_receipt_sha256
            ),
        )
    index = build_full_store_window_index(cache)
    sealed_by_namespace = {
        require_sha256(value.get("namespace_id"), "sealed cache namespace"): value
        for value in (
            _exact_dict(row, "sealed cache receipt")
            for row in _exact_list(closure.payload.get("cache_receipts"), "cache receipts")
        )
    }
    sealed = sealed_by_namespace.get(namespace_id)
    _require(
        sealed is not None
        and sealed.get("cache_receipt_sha256") == cache.cache_receipt_sha256
        and sealed.get("window_index_receipt_sha256") == index.receipt_sha256
        and sealed.get("content_row_count") == cache.content_row_count
        and sealed.get("physical_store_row_count") == cache.physical_store_row_count,
        "v4 resident index differs from sealed parent closure",
    )
    body = {
        "cache_receipt_sha256": cache.cache_receipt_sha256,
        "content_row_count": cache.content_row_count,
        "database_read_passes": 1,
        "format": INDEX_FORMAT,
        "namespace_id": namespace_id,
        "physical_content_token_count": index.physical_content_tokens_indexed,
        "physical_store_row_count": cache.physical_store_row_count,
        "source_database_sha256": context.database_sha256,
        "source_store_receipt_sha256": context.namespace.combined_store_receipt_sha256,
        "window_index_receipt_sha256": index.receipt_sha256,
    }
    return index, _with_receipt(body, "resident_index_receipt_sha256")


def _q79_question(
    *,
    row: Mapping[str, Any],
    namespace_id: str,
    index: Any,
) -> dict[str, Any]:
    dated_question, parent_prediction, _question_id = _question_inputs(row)
    spec = compile_typed_operator_spec(dated_question)
    applicable = specialist_cli.applicable_specialist_ids(dated_question)
    _require(
        applicable == (TEMPORAL_MECHANISM_ID,),
        "q79 latest-state contract did not route only to temporal specialist",
    )
    runs = specialist_cli._run_specialists(index, dated_question)  # noqa: SLF001
    _require(len(runs) == 1 and runs[0].mechanism_id == TEMPORAL_MECHANISM_ID, "q79 specialist run changed")
    run = runs[0]
    _require(type(run.result) is TemporalInsufficiencyResult, "q79 temporal result changed")
    result = run.result
    bundle = result.temporal_bundle
    _require(bundle is not None and bundle.winner_candidate_id is not None, "q79 temporal winner missing")
    candidate_handle = {
        local.candidate_id: binding.handle_id
        for binding, local in zip(
            run.contribution.bindings, run.local_bindings, strict=True
        )
    }
    winner_handle = candidate_handle[bundle.winner_candidate_id]
    predecessor_handle = (
        None
        if bundle.predecessor_candidate_id is None
        else candidate_handle[bundle.predecessor_candidate_id]
    )
    contribution = run.contribution
    _require(
        contribution.frontier_mode is not FrontierMode.EXHAUSTIVE,
        "q79 bounded temporal bundle became globally exhaustive",
    )
    packet = build_typed_evidence_packet(
        spec,
        contribution.bindings,
        contribution.parsed,
        sealed_input_artifact_sha256s=(contribution.sealed_artifact_sha256,),
        frontier_mode=contribution.frontier_mode,
        truncated=contribution.truncated,
        provider_payload_mode=ProviderPayloadMode.COMPACT_FINAL,
    )
    protected_handles = {winner_handle}
    if predecessor_handle is not None:
        protected_handles.add(predecessor_handle)
    protected_items = tuple(
        item.receipt_sha256
        for item in packet.items
        if protected_handles & set(item.handle_ids)
    )
    local_priorities = {
        handle: values
        for handle, values in specialist_cli._specialist_local_priorities(runs).items()  # noqa: SLF001
        if handle in {binding.handle_id for binding in packet.local_bindings}
    }
    fitted = fit_typed_final_prompt(
        dated_question=dated_question,
        parent_prediction=parent_prediction,
        packet=packet,
        mechanism_by_handle={
            binding.handle_id: TEMPORAL_MECHANISM_ID
            for binding in packet.local_bindings
        },
        local_retention_priority_by_handle=local_priorities,
        minimum_usable_items_per_mechanism=1,
        protected_item_receipt_sha256s=protected_items,
        protection_source_receipt_sha256=contribution.receipt_sha256,
    )
    _require(winner_handle in fitted.allowed_handle_ids, "q79 fit dropped temporal winner")
    advisories = specialist_cli._specialist_advisories(runs, fitted.allowed_handle_ids)  # noqa: SLF001
    _require(len(advisories) == 1, "q79 temporal advisory missing")
    advisory_bundle = _exact_dict(advisories[0].get("temporal_bundle"), "q79 terminal bundle")
    _require(
        advisory_bundle.get("winner_handle_id") == winner_handle,
        "q79 terminal advisory changed its winner",
    )
    terminal_prompt = _terminal_projection(
        fitted=fitted,
        specialist_advisories=advisories,
    )
    provenance = [
        _provenance_row(
            binding=binding,
            source_ids=(local.source_id,),
            local_evidence=local.projection(),
        )
        for binding, local in zip(
            contribution.bindings, run.local_bindings, strict=True
        )
    ]
    selection = _selection_projection(
        mode="typed_latest_state_temporal_specialist",
        mechanism_ids=(TEMPORAL_MECHANISM_ID,),
        items=contribution.parsed.accepted_items,
        bindings=contribution.bindings,
        provenance=provenance,
    )
    method = specialist_cli._method_projection(run)  # noqa: SLF001
    operator = _with_receipt(
        {
            "applicable_specialist_ids": list(applicable),
            "bundle_population_count": bundle.population_count,
            "bundle_selected_count": len(bundle.ordered_candidate_ids),
            "bundle_truncated": bundle.truncated,
            "format": f"{FORMAT}-q79-temporal-specialist-v1",
            "global_exhaustiveness_claimed": False,
            "predecessor_handle_id": predecessor_handle,
            "provider_prompt_count": 0,
            "retained_transformer_token_state_bytes": 0,
            "specialist_receipt_sha256": run.specialist_receipt_sha256,
            "winner_handle_id": winner_handle,
        },
        "operator_receipt_sha256",
    )
    return _question_shell(
        ordinal=79,
        row=row,
        namespace_id=namespace_id,
        mechanism_plan=(TEMPORAL_MECHANISM_ID,),
        selection=selection,
        methods=(method,),
        operator=operator,
        provenance=provenance,
        fitted_typed_prompt=fitted.projection(),
        terminal_kind="temporal_specialist_synthesis",
        terminal_prompt=terminal_prompt,
    )


def build_construction(args: argparse.Namespace) -> dict[str, Any]:
    (
        composition,
        closure,
        run,
        replay,
        composition_rows,
        _closure_rows,
        _run_rows,
        _judge_rows,
    ) = parent_cli._load_parent_inputs(Path(args.parent_root))  # noqa: SLF001
    semantic_artifact, semantic_rows = semantic_cli.load_verified_construction(
        Path(args.semantic_construction),
        expected_sha256=args.expected_semantic_construction_sha256,
    )
    semantic_by_ordinal = {int(row["ordinal"]): row for row in semantic_rows}
    _require(
        tuple(sorted(semantic_by_ordinal)) == TARGET_ORDINALS,
        "semantic v3 residual population changed",
    )
    namespace_by_ordinal = {
        ordinal: require_sha256(
            semantic_by_ordinal[ordinal].get("namespace_id"),
            f"semantic namespace {ordinal}",
        )
        for ordinal in TARGET_ORDINALS
    }

    q42 = _q42_question(
        row=composition_rows[42],
        composition_sha256=composition.sha256,
        namespace_id=namespace_by_ordinal[42],
    )
    q65 = _q65_question(
        row=composition_rows[65],
        composition_sha256=composition.sha256,
        namespace_id=namespace_by_ordinal[65],
    )
    q74 = _q74_question(
        row=composition_rows[74],
        semantic_row=semantic_by_ordinal[74],
        semantic_artifact_sha256=semantic_artifact.sha256,
        namespace_id=namespace_by_ordinal[74],
    )
    q79_id = require_text(composition_rows[79].get("question_id"), "q79 question")
    index, lifecycle = _verified_resident_index(
        args,
        namespace_id=namespace_by_ordinal[79],
        question_id=q79_id,
        closure=closure,
    )
    q79 = _q79_question(
        row=composition_rows[79],
        namespace_id=namespace_by_ordinal[79],
        index=index,
    )
    del index
    gc.collect()

    questions = [q42, q65, q74, q79]
    terminal_tokens = [
        int(row["terminal_prompt"]["full_chat_plus_output_tokens"])
        for row in questions
    ]
    structural_body = {
        "all_questions_have_terminal_evidence": all(
            bool(row["fitted_typed_prompt"]["allowed_handle_ids"])
            for row in questions
        ),
        "answer_row_base_validator": (
            "tools.run_reduced_specialist_answer_v2._prompt_plan_row"
        ),
        "deterministic_terminal_count": 0,
        "format": f"{FORMAT}-gold-blind-structural-gate-v1",
        "max_terminal_complete_envelope_tokens": max(terminal_tokens),
        "ordinary_scoped_provider_terminal_count": QUESTION_COUNT,
        "provider_ready_terminal_count": QUESTION_COUNT,
        "q42_support_frontier_closed": False,
        "q65_post_selection_replacement_only": True,
        "q74_complete_residual_lane_retained": True,
        "q79_scoped_temporal_winner_retained": True,
    }
    structural_gate = _with_receipt(structural_body)
    payload: dict[str, Any] = {
        "bindings": {
            "parent_composition_artifact_sha256": composition.sha256,
            "parent_full_store_input_artifact_sha256": closure.sha256,
            "parent_replay_artifact_sha256": replay.sha256,
            "parent_run_artifact_sha256": run.sha256,
            "semantic_v3_construction_artifact_sha256": semantic_artifact.sha256,
        },
        "construction_is_posthoc_outcome_conditioned": False,
        "format": CONSTRUCTION_FORMAT,
        "gold_loaded": False,
        "hard_complete_chat_token_cap": HARD_COMPLETE_CHAT_TOKEN_CAP,
        "max_terminal_complete_envelope_tokens": max(terminal_tokens),
        "new_provider_calls": 0,
        "ordinals": list(TARGET_ORDINALS),
        "question_count": QUESTION_COUNT,
        "questions": questions,
        "resident_index_lifecycle": lifecycle,
        "retained_transformer_token_state_bytes": 0,
        "selection_and_routing_frozen_before_target_plan_load": True,
        "structural_gate": structural_gate,
        "target_labels_loaded": False,
        "target_plan_loaded": False,
    }
    assert_gold_blind(payload, path="reduced_missing4_v4_construction")
    return {**payload, "construction_identity_sha256": identity_sha256(payload)}


def validate_construction(artifact: SealedArtifact) -> tuple[dict[str, Any], ...]:
    payload = artifact.payload
    body = dict(payload)
    declared = require_sha256(
        body.pop("construction_identity_sha256", None), "v4 construction identity"
    )
    _require(identity_sha256(body) == declared, "v4 construction identity changed")
    bindings = _exact_dict(payload.get("bindings"), "v4 bindings")
    questions = tuple(
        _exact_dict(row, "v4 question")
        for row in _exact_list(payload.get("questions"), "v4 questions")
    )
    lifecycle = _validate_receipt(
        payload.get("resident_index_lifecycle"),
        label="v4 resident index",
        key="resident_index_receipt_sha256",
    )
    structural = _validate_receipt(payload.get("structural_gate"), label="v4 structural gate")
    _require(
        payload.get("format") == CONSTRUCTION_FORMAT
        and payload.get("construction_is_posthoc_outcome_conditioned") is False
        and payload.get("gold_loaded") is False
        and payload.get("target_labels_loaded") is False
        and payload.get("target_plan_loaded") is False
        and payload.get("selection_and_routing_frozen_before_target_plan_load") is True
        and payload.get("new_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("hard_complete_chat_token_cap") == HARD_COMPLETE_CHAT_TOKEN_CAP
        and tuple(payload.get("ordinals", ())) == TARGET_ORDINALS
        and payload.get("question_count") == QUESTION_COUNT
        and len(questions) == QUESTION_COUNT
        and tuple(row.get("ordinal") for row in questions) == TARGET_ORDINALS
        and bindings
        == {
            "parent_composition_artifact_sha256": parent_cli.EXPECTED_PARENT_COMPOSITION_SHA256,
            "parent_full_store_input_artifact_sha256": parent_cli.EXPECTED_PARENT_CLOSURE_SHA256,
            "parent_replay_artifact_sha256": parent_cli.EXPECTED_PARENT_REPLAY_SHA256,
            "parent_run_artifact_sha256": parent_cli.EXPECTED_PARENT_RUN_SHA256,
            "semantic_v3_construction_artifact_sha256": EXPECTED_SEMANTIC_CONSTRUCTION_SHA256,
        }
        and lifecycle.get("database_read_passes") == 1
        and lifecycle.get("physical_content_token_count") > 1_000_000,
        "v4 construction boundary changed",
    )
    terminal_tokens: list[int] = []
    for ordinal, row in zip(TARGET_ORDINALS, questions, strict=True):
        unsigned = dict(row)
        row_receipt = require_sha256(
            unsigned.pop("question_receipt_sha256", None), "v4 question"
        )
        _require(
            identity_sha256(unsigned) == row_receipt
            and row.get("format") == QUESTION_FORMAT
            and row.get("new_provider_calls") == 0
            and row.get("retained_transformer_token_state_bytes") == 0,
            f"v4 question seal changed at {ordinal}",
        )
        provenance = tuple(
            _validate_receipt(value, label=f"v4 provenance {ordinal}")
            for value in _exact_list(row.get("local_provenance"), "v4 provenance")
        )
        selection = _validate_receipt(row.get("selection"), label=f"v4 selection {ordinal}")
        operator = _validate_receipt(
            row.get("operator"),
            label=f"v4 operator {ordinal}",
            key="operator_receipt_sha256",
        )
        fitted = _exact_dict(
            row.get("fitted_typed_prompt"), f"v4 fitted prompt {ordinal}"
        )
        terminal = _exact_dict(
            row.get("terminal_prompt"), f"v4 terminal prompt {ordinal}"
        )
        answer_plan = answer_v2_cli._prompt_plan_row(row, ordinal)  # noqa: SLF001
        known_handles = {value["handle_id"] for value in provenance}
        selected_handles = tuple(selection.get("selected_handle_ids", ()))
        terminal_handles = tuple(fitted.get("allowed_handle_ids", ()))
        terminal_input = _exact_dict(
            terminal.get("provider_input"), f"v4 provider input {ordinal}"
        )
        advisories = _exact_list(
            terminal_input.get("specialist_advisories"),
            f"v4 scoped advisories {ordinal}",
        )
        _require(
            bool(selected_handles)
            and bool(terminal_handles)
            and bool(advisories)
            and set(selected_handles) <= known_handles
            and set(terminal_handles) <= known_handles
            and answer_plan.get("allowed_handle_ids") == list(terminal_handles)
            and answer_plan.get("terminal_prompt_receipt_sha256")
            == terminal.get("terminal_prompt_receipt_sha256")
            and terminal.get("provider_prompt_count") == 0
            and terminal.get("retained_transformer_token_state_bytes") == 0
            and terminal.get("hard_prompt_token_cap")
            == HARD_COMPLETE_CHAT_TOKEN_CAP,
            f"v4 selection/terminal provenance changed at {ordinal}",
        )
        full_tokens = terminal.get("full_chat_plus_output_tokens")
        _require(
            type(full_tokens) is int and full_tokens <= HARD_COMPLETE_CHAT_TOKEN_CAP,
            f"v4 terminal cap changed at {ordinal}",
        )
        terminal_tokens.append(full_tokens)
        if ordinal == 42:
            decision = _exact_dict(operator.get("decision"), "q42 decision")
            _require(
                operator.get("support_frontier_closed") is False
                and decision.get("disposition") == "keep_parent"
                and decision.get("terminal_authorized") is False
                and row.get("terminal_kind") == "conjunctive_event_synthesis"
                and advisories[0].get("proof_kind")
                == "same_event_conjunctive_obligation"
                and _exact_dict(
                    advisories[0].get("support_frontier"),
                    "q42 provider support frontier",
                ).get("generic_frontier_closed")
                is False,
                "q42 fail-closed support boundary changed",
            )
        elif ordinal == 65:
            compression = _exact_dict(operator.get("compression"), "q65 compression")
            closure = _exact_dict(compression.get("closure"), "q65 closure")
            execution = _exact_dict(operator.get("execution"), "q65 execution")
            contribution = _exact_dict(
                operator.get("terminal_typed_contribution"),
                "q65 bounded terminal contribution",
            )
            _require(
                operator.get("raw_selected_lane_merged_after_compression") is False
                and closure.get("support_frontier_closed") is True
                and len(compression.get("facts", ())) == 2
                and execution.get("status") == "insufficient"
                and operator.get("generic_frontier_closed") is False
                and operator.get("selected_scope_only") is True
                and operator.get("upstream_truncated") is True
                and contribution.get("frontier_mode") == FrontierMode.BOUNDED.value
                and contribution.get("truncated") is True
                and row.get("terminal_kind")
                == "selected_scope_action_set_synthesis"
                and advisories[0].get("scope")
                == "selected_action_linked_members_only"
                and advisories[0].get("generic_frontier_closed") is False
                and advisories[0].get("upstream_truncated") is True,
                "q65 post-selection replacement contract changed",
            )
        elif ordinal == 74:
            _require(
                row.get("mechanism_plan") == [residual.TYPED_ADAPTER_MECHANISM_ID]
                and operator.get("frontier_mode") == FrontierMode.BOUNDED.value
                and operator.get("truncated") is False
                and selected_handles == ("H950001", "H950002")
                and terminal_handles == ("H950001", "H950002"),
                "q74 complete bounded residual lane changed",
            )
        else:
            _require(
                operator.get("applicable_specialist_ids") == [TEMPORAL_MECHANISM_ID]
                and operator.get("global_exhaustiveness_claimed") is False
                and operator.get("winner_handle_id") in terminal_handles,
                "q79 scoped temporal winner changed",
            )
    _require(
        payload.get("max_terminal_complete_envelope_tokens") == max(terminal_tokens)
        and structural.get("max_terminal_complete_envelope_tokens") == max(terminal_tokens)
        and structural.get("all_questions_have_terminal_evidence") is True
        and structural.get("answer_row_base_validator")
        == "tools.run_reduced_specialist_answer_v2._prompt_plan_row"
        and structural.get("provider_ready_terminal_count") == QUESTION_COUNT
        and structural.get("ordinary_scoped_provider_terminal_count")
        == QUESTION_COUNT
        and structural.get("deterministic_terminal_count") == 0,
        "v4 structural summary changed",
    )
    assert_gold_blind(payload, path="validated_reduced_missing4_v4")
    return questions


def _target_sources(plan: Mapping[str, Any]) -> dict[int, tuple[str, ...]]:
    result: dict[int, list[str]] = {ordinal: [] for ordinal in TARGET_ORDINALS}
    for value in _exact_list(plan.get("desired_targets"), "target plan rows"):
        row = _exact_dict(value, "target plan row")
        ordinal = row.get("ordinal")
        if ordinal in result and row.get("target_kind") == "source_id":
            source = require_text(row.get("target_id"), "target source")
            if source not in result[int(ordinal)]:
                result[int(ordinal)].append(source)
    frozen = {ordinal: tuple(values) for ordinal, values in result.items()}
    _require(
        sum(len(values) for values in frozen.values()) == 6
        and all(frozen.values()),
        "missing-four target source population changed",
    )
    return frozen


def build_target_audit(
    construction: SealedArtifact,
    plan: Mapping[str, Any],
    *,
    target_plan_file_sha256: str,
) -> dict[str, Any]:
    questions = validate_construction(construction)
    expected = _target_sources(plan)
    audited: list[dict[str, Any]] = []
    selected_hits_total = 0
    terminal_hits_total = 0
    for ordinal, row in zip(TARGET_ORDINALS, questions, strict=True):
        provenance = _exact_list(row.get("local_provenance"), "audit provenance")
        selection = _exact_dict(row.get("selection"), "audit selection")
        fitted = _exact_dict(row.get("fitted_typed_prompt"), "audit fitted prompt")
        terminal = _exact_dict(row.get("terminal_prompt"), "audit terminal prompt")
        selected_sources = _source_ids_for_handles(
            provenance, selection.get("selected_handle_ids", ())
        )
        terminal_sources = _source_ids_for_handles(
            provenance, fitted.get("allowed_handle_ids", ())
        )
        selected_aliases = reduced_cli._source_aliases(  # noqa: SLF001
            selected_sources, row["question_id"]
        )
        terminal_aliases = reduced_cli._source_aliases(  # noqa: SLF001
            terminal_sources, row["question_id"]
        )
        selected_hits = [value for value in expected[ordinal] if value in selected_aliases]
        terminal_hits = [value for value in expected[ordinal] if value in terminal_aliases]
        selected_hits_total += len(selected_hits)
        terminal_hits_total += len(terminal_hits)
        operator = _exact_dict(row.get("operator"), "audit operator")
        operator_contract_valid = (
            operator.get("support_frontier_closed") is False
            if ordinal == 42
            else (
                operator.get("raw_selected_lane_merged_after_compression") is False
                and operator.get("generic_frontier_closed") is False
                and operator.get("selected_scope_only") is True
            )
            if ordinal == 65
            else operator.get("frontier_mode") == FrontierMode.BOUNDED.value
            if ordinal == 74
            else operator.get("winner_handle_id")
            in fitted.get("allowed_handle_ids", ())
        )
        audit_body = {
            "expected_source_ids": list(expected[ordinal]),
            "operator_contract_valid": operator_contract_valid,
            "ordinal": ordinal,
            "question_id": row["question_id"],
            "selected_source_ids": selected_sources,
            "selected_source_target_hits": selected_hits,
            "selection_source_set_complete": len(selected_hits) == len(expected[ordinal]),
            "terminal_full_chat_plus_output_tokens": terminal["full_chat_plus_output_tokens"],
            "terminal_source_ids": terminal_sources,
            "terminal_source_set_complete": len(terminal_hits) == len(expected[ordinal]),
            "terminal_source_target_hits": terminal_hits,
        }
        audited.append(_with_receipt(audit_body, "audit_row_receipt_sha256"))
    target_count = sum(len(values) for values in expected.values())
    gate_body = {
        "all_operator_contracts_valid": all(row["operator_contract_valid"] for row in audited),
        "format": f"{AUDIT_FORMAT}-structural-gate-v1",
        "hard_complete_chat_token_cap": HARD_COMPLETE_CHAT_TOKEN_CAP,
        "selection_source_target_count": target_count,
        "selection_source_target_hits": selected_hits_total,
        "structural_gate_passed": (
            selected_hits_total == target_count
            and terminal_hits_total == target_count
            and all(row["operator_contract_valid"] for row in audited)
            and all(
                row["terminal_full_chat_plus_output_tokens"]
                <= HARD_COMPLETE_CHAT_TOKEN_CAP
                for row in audited
            )
        ),
        "terminal_source_target_count": target_count,
        "terminal_source_target_hits": terminal_hits_total,
    }
    gate = _with_receipt(gate_body)
    payload: dict[str, Any] = {
        "audit_is_posthoc_only": True,
        "construction_artifact_sha256": construction.sha256,
        "construction_verified_before_target_plan_load": True,
        "format": AUDIT_FORMAT,
        "gold_loaded": False,
        "new_provider_calls": 0,
        "ordinals": list(TARGET_ORDINALS),
        "question_count": QUESTION_COUNT,
        "questions": audited,
        "retained_transformer_token_state_bytes": 0,
        "runtime_use_forbidden": True,
        "selection_source_target_count": target_count,
        "selection_source_target_hits": selected_hits_total,
        "structural_gate": gate,
        "target_labels_loaded": True,
        "target_plan_file_sha256": require_sha256(
            target_plan_file_sha256, "target plan file"
        ),
        "target_plan_identity_sha256": require_sha256(
            plan.get("plan_sha256"), "target plan identity"
        ),
        "target_plan_loaded": True,
        "terminal_source_target_count": target_count,
        "terminal_source_target_hits": terminal_hits_total,
    }
    return {**payload, "audit_identity_sha256": identity_sha256(payload)}


def run_construct(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_construction(args)
    target = Path(args.output_root) / CONSTRUCTION_NAME
    candidate = SealedArtifact(
        path=target,
        sha256=hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        payload=payload,
    )
    validate_construction(candidate)
    artifact, created = publish_sealed_json(target, payload)
    return {
        "construction_sha256": artifact.sha256,
        "created": created,
        "max_terminal_complete_envelope_tokens": payload[
            "max_terminal_complete_envelope_tokens"
        ],
        "new_provider_calls": 0,
        "question_count": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    construction = read_sealed_json(Path(args.construction))
    _require(
        construction.sha256
        == require_sha256(args.expected_construction_sha256, "expected v4 construction"),
        "v4 construction artifact changed",
    )
    # Verification deliberately precedes the first target-plan read.
    validate_construction(construction)
    plan, plan_file_sha = reduced_cli._read_target_plan(Path(args.target_plan))  # noqa: SLF001
    payload = build_target_audit(
        construction,
        plan,
        target_plan_file_sha256=plan_file_sha,
    )
    artifact, created = publish_sealed_json(Path(args.output), payload)
    gate = payload["structural_gate"]
    return {
        "audit_sha256": artifact.sha256,
        "created": created,
        "new_provider_calls": 0,
        "question_count": QUESTION_COUNT,
        "selection_source_target_score": (
            f"{gate['selection_source_target_hits']}/"
            f"{gate['selection_source_target_count']}"
        ),
        "structural_gate_passed": gate["structural_gate_passed"],
        "terminal_source_target_score": (
            f"{gate['terminal_source_target_hits']}/"
            f"{gate['terminal_source_target_count']}"
        ),
    }


def _add_store_args(parser: argparse.ArgumentParser) -> None:
    guided = reduced_cli.guided_scan_cli
    parser.add_argument("--retrieval", type=Path, default=guided.DEFAULT_RETRIEVAL)
    parser.add_argument("--store-root", type=Path, default=guided.DEFAULT_STORE_ROOT)
    parser.add_argument(
        "--query-parent-output-root", type=Path, default=guided.DEFAULT_PARENT_OUTPUT
    )
    parser.add_argument(
        "--expected-retrieval-sha256", default=guided.EXPECTED_RETRIEVAL_SHA256
    )
    parser.add_argument(
        "--expected-query-parent-preflight-sha256",
        default=guided.EXPECTED_PARENT_PREFLIGHT_SHA256,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    construct = commands.add_parser("construct", help="seal provider-free v4 construction")
    construct.add_argument("--parent-root", type=Path, default=parent_cli.DEFAULT_PARENT_ROOT)
    construct.add_argument(
        "--semantic-construction", type=Path, default=DEFAULT_SEMANTIC_CONSTRUCTION
    )
    construct.add_argument(
        "--expected-semantic-construction-sha256",
        default=EXPECTED_SEMANTIC_CONSTRUCTION_SHA256,
    )
    construct.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    _add_store_args(construct)

    audit = commands.add_parser("audit", help="post-hoc source survival audit")
    audit.add_argument(
        "--construction", type=Path, default=DEFAULT_OUTPUT_ROOT / CONSTRUCTION_NAME
    )
    audit.add_argument("--expected-construction-sha256", required=True)
    audit.add_argument("--target-plan", type=Path, default=DEFAULT_TARGET_PLAN)
    audit.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / AUDIT_NAME)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_construct(args) if args.command == "construct" else run_audit(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_FORMAT",
    "AUDIT_NAME",
    "CONSTRUCTION_FORMAT",
    "CONSTRUCTION_NAME",
    "DEFAULT_OUTPUT_ROOT",
    "EXPECTED_SEMANTIC_CONSTRUCTION_SHA256",
    "FORMAT",
    "HARD_COMPLETE_CHAT_TOKEN_CAP",
    "TARGET_ORDINALS",
    "build_construction",
    "build_parser",
    "build_target_audit",
    "main",
    "run_audit",
    "run_construct",
    "validate_construction",
]
