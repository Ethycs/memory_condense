"""Pure, provider-free helpers for the confirmation semantic planes.

This module contains only the small reusable surfaces that the confirmation
runtime needs from the historical assay CLIs.  Keeping them here prevents the
prediction process from importing validation runners (and, transitively,
their gold and judge entry points) merely to call a deterministic helper.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
)
from memory_condense.domain.discourse import EvidenceSpan, quote_sha256

from . import semantic_residual_search as residual
from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
    require_text,
)
from .full_store_slot_closure import BINDING_FORMAT, LocalCitationBinding
from .protected_parent_contribution import (
    rehydrate_protected_parent_contributions,
)
from .semantic_global_terminal_adapter import (
    BACKFILL_FORMAT,
    EXACT_SPAN_SUPPORT_FORMAT,
    EXACT_SPAN_SUPPORT_POPULATION_FORMAT,
    EXACT_SPAN_SUPPORT_RANKING_POLICY,
    FORMAT as TERMINAL_COMPILATION_FORMAT,
    HARD_PROMPT_TOKEN_CAP,
    LINKED_BACKFILL_FORMAT,
    LINKED_FORMAT,
    OUTPUT_TOKEN_RESERVE,
    PLANE_ORDER,
    ExactSpanSupportAuthority,
    ExactSpanSupportPopulationReceipt,
    SemanticGlobalTerminalPolicy,
    TerminalSealedSources,
    compile_semantic_global_terminal,
    load_selected_protected_owner_evidence,
    replay_semantic_global_terminal,
)
from .semantic_residual_eligibility import SemanticResidualEligibilityPolicy
from .typed_memory_final_arm import render_final_messages
from .typed_operator_spec import compile_typed_operator_spec


_RESIDUAL_TERMINAL_FORMAT = (
    "memory-condense-locked-semantic-residual-v4-separate-synthesis-prompt-v2"
)
_TYPED_BINDING_FORMAT = "memory-condense-local-evidence-binding-v1"
_DEFAULT_PROTECTED_OWNER_TOKEN_CAP = 2_400
_RESIDUAL_SYSTEM_PROMPT = (
    "You are the terminal memory reconciler. Use the current answer as a "
    "protected fallback. residual_evidence is newly selected evidence. "
    "protected_owner_evidence contains exact text reinjected only to preserve "
    "the provider-visible owner of post-selection duplicates. Replace the "
    "answer only when the supplied evidence directly supports a better answer, "
    "cite only supplied evidence handles, and include at least one R handle for "
    "replacement. Return one JSON object matching the response schema."
)

ANSWER_PLAN_FORMAT = (
    "memory-condense-reduced-semantic-global-terminal-assay-v2-answer-plan-v2"
)
TERMINAL_COMPILATION_MODE_V2 = "v2"
TERMINAL_COMPILATION_MODE_V3 = "v3-linked"
TERMINAL_COMPILATION_MODE_V4 = "v4-backfill"
TERMINAL_COMPILATION_MODE_V5 = "v5-linked-backfill"
_TERMINAL_COMPILATION_FORMAT_BY_MODE = {
    TERMINAL_COMPILATION_MODE_V2: TERMINAL_COMPILATION_FORMAT,
    TERMINAL_COMPILATION_MODE_V3: LINKED_FORMAT,
    TERMINAL_COMPILATION_MODE_V4: BACKFILL_FORMAT,
    TERMINAL_COMPILATION_MODE_V5: LINKED_BACKFILL_FORMAT,
}
TERMINAL_COMPILATION_FORMAT_BY_MODE = _TERMINAL_COMPILATION_FORMAT_BY_MODE
_ROUTE_ID = "semantic-global-terminal-terra-answer-v2"
ROUTE_ID = _ROUTE_ID
ANSWER_PLAN_KEYS = frozenset(
    {
        "allowed_handle_ids",
        "answer_plan_receipt_sha256",
        "dated_question",
        "dated_question_sha256",
        "format",
        "handle_group_by_id",
        "hard_prompt_token_cap",
        "messages_sha256",
        "ordinal",
        "output_token_reserve",
        "parent_prediction",
        "parent_prediction_sha256",
        "preservation_requirements",
        "prompt_token_proxy",
        "provider_input",
        "provider_input_sha256",
        "question_id",
        "question_sha256",
        "route_id",
        "source_artifact_bindings",
        "story_coherence",
        "terminal_compilation",
        "terminal_compilation_receipt_sha256",
        "validation_contract",
    }
)
EXACT_SPAN_SUPPORT_AUTHORITY_KEYS = frozenset(
    {
        "authority_candidate_receipt_sha256s",
        "authority_source_planes",
        "exact_relation_support",
        "exact_span_identity_sha256",
        "format",
        "matched_query_actions",
        "past_event_witness",
        "policy",
        "priority_prefix",
        "query_temporal_support",
        "receipt_sha256",
        "role",
        "source_group_supported_kinds",
        "source_group_supported_obligation_ids",
        "supported_obligation_ids",
    }
)
EXACT_SPAN_SUPPORT_POPULATION_KEYS = frozenset(
    {
        "authorities",
        "format",
        "plane_candidate_receipt_sha256s",
        "plane_selection_receipt_sha256s",
        "policy",
        "receipt_sha256",
    }
)


class ConfirmationSemanticHelperError(MatchedEvalContractError):
    """A pure semantic reconstruction or receipt changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationSemanticHelperError(message)


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact array")
    return value  # type: ignore[return-value]


def _exact_int(value: object, label: str) -> int:
    _require(type(value) is int, f"{label} must be an exact integer")
    return value  # type: ignore[return-value]


def _with_receipt(
    body: Mapping[str, Any], key: str = "receipt_sha256"
) -> dict[str, Any]:
    return {**dict(body), key: identity_sha256(body)}


def question_inputs(composition_row: Mapping[str, Any]) -> tuple[str, str, str]:
    """Read the three gold-blind question fields from a typed composition."""

    provider = _exact_dict(
        composition_row.get("provider_projection"), "composition provider"
    )
    provider_input = _exact_dict(provider.get("provider_input"), "provider input")
    return (
        require_text(
            provider_input.get("dated_question"), "specialist dated question"
        ),
        require_text(
            composition_row.get("parent_prediction"),
            "specialist parent prediction",
        ),
        require_text(composition_row.get("question_id"), "specialist question ID"),
    )


def _rehydrate_span(raw: object) -> EvidenceSpan:
    row = _exact_dict(raw, "evidence span")
    return EvidenceSpan(
        chunk_id=require_text(row.get("chunk_id"), "span chunk"),
        start_char=row.get("start_char"),
        end_char=row.get("end_char"),
        quote_sha256=require_sha256(row.get("quote_sha256"), "span quote"),
        ordinal=row.get("ordinal"),
        source_id=row.get("source_id"),
        turn_start_char=row.get("turn_start_char", 0),
        turn_id=row.get("turn_id"),
        role=row.get("role"),
        created_at=row.get("created_at"),
    )


def _rehydrate_local_binding(raw: object) -> LocalCitationBinding:
    row = _exact_dict(raw, "local citation binding")
    binding = LocalCitationBinding(
        candidate_id=require_sha256(row.get("candidate_id"), "local candidate"),
        source_group_handle=require_text(
            row.get("source_group_handle"), "local group"
        ),
        namespace_id=require_sha256(row.get("namespace_id"), "local namespace"),
        cache_receipt_sha256=require_sha256(
            row.get("cache_receipt_sha256"), "local cache"
        ),
        source_database_sha256=require_sha256(
            row.get("source_database_sha256"), "local database"
        ),
        source_store_receipt_sha256=require_sha256(
            row.get("source_store_receipt_sha256"), "local store"
        ),
        source_id=require_text(row.get("source_id"), "local source"),
        partition_id=require_text(row.get("partition_id"), "local partition"),
        span=_rehydrate_span(row.get("span")),
        quote_sha256=require_sha256(row.get("quote_sha256"), "local quote"),
        receipt_sha256=require_sha256(row.get("receipt_sha256"), "local receipt"),
    )
    _require(binding.projection() == row, "local citation projection changed")
    return binding


def _local_binding_projections(value: object) -> tuple[LocalCitationBinding, ...]:
    found: dict[str, LocalCitationBinding] = {}

    def visit(raw: object) -> None:
        if type(raw) is dict:
            if raw.get("format") == BINDING_FORMAT:
                binding = _rehydrate_local_binding(raw)
                previous = found.get(binding.receipt_sha256)
                _require(
                    previous is None
                    or previous.projection() == binding.projection(),
                    "parent local citation receipt changed projection",
                )
                found[binding.receipt_sha256] = binding
                return
            for nested in raw.values():
                visit(nested)
        elif type(raw) is list:
            for nested in raw:
                visit(nested)

    visit(value)
    return tuple(found[key] for key in sorted(found))


def _protected_parent_local_evidence(
    composition_row: Mapping[str, Any],
    parent: Any,
    *,
    namespace_id: str,
) -> tuple[LocalCitationBinding, ...]:
    local_audit = _exact_dict(
        composition_row.get("local_audit"), "parent composition local audit"
    )
    local_by_receipt = {
        row.receipt_sha256: row for row in _local_binding_projections(local_audit)
    }
    item_by_handle = {
        handle: item
        for contribution in parent.contributions
        for item in contribution.parsed.accepted_items
        for handle in item.handle_ids
    }
    provenance_by_locator: dict[str, list[Any]] = defaultdict(list)
    for provenance in parent.audit.source_provenance:
        provenance_by_locator[
            provenance.original_binding.local_source_locator_sha256
        ].append(provenance)
    compact_order = {
        receipt: index
        for index, receipt in enumerate(parent.audit.compact_item_receipt_order)
    }
    cloned_by_handle = {
        value.handle_id: value.cloned_binding
        for value in parent.audit.source_provenance
    }
    protected: list[LocalCitationBinding] = []
    for local_receipt in sorted(set(local_by_receipt) & set(provenance_by_locator)):
        local = local_by_receipt[local_receipt]
        if local.namespace_id != namespace_id:
            continue
        eligible: dict[str, tuple[Any, Any]] = {}
        for provenance in provenance_by_locator[local_receipt]:
            item = item_by_handle.get(provenance.handle_id)
            if item is None:
                continue
            if (
                provenance.original_binding.citation_sha256 != local.quote_sha256
                or quote_sha256(item.summary) != local.quote_sha256
                or provenance.original_binding.citation_char_count
                != len(item.summary)
            ):
                continue
            eligible.setdefault(item.receipt_sha256, (item, provenance))
        if not eligible:
            continue
        owner_receipt = min(
            eligible,
            key=lambda receipt: (compact_order.get(receipt, 1 << 30), receipt),
        )
        item, _provenance = eligible[owner_receipt]
        visible_bindings = tuple(
            cloned_by_handle.get(handle) for handle in item.handle_ids
        )
        _require(
            all(value is not None for value in visible_bindings),
            "protected parent owner lost a typed binding",
        )
        protected.append(local)
    return tuple(protected)


def _walk_dicts(value: object):
    if type(value) is dict:
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif type(value) is list:
        for child in value:
            yield from _walk_dicts(child)


def _visible_specialist_local_evidence(
    construction_row: Mapping[str, Any],
    *,
    namespace_id: str,
) -> tuple[LocalCitationBinding, ...]:
    terminal = _exact_dict(construction_row.get("terminal_prompt"), "terminal")
    provider_input = _exact_dict(terminal.get("provider_input"), "provider input")
    typed = _exact_dict(provider_input.get("typed_evidence"), "typed evidence")
    frontier = _exact_dict(typed.get("frontier"), "typed frontier")
    represented = set(_exact_list(frontier.get("represented_handle_ids"), "handles"))
    locator_receipts: set[str] = set()
    for row in _walk_dicts(construction_row):
        if (
            row.get("format") == _TYPED_BINDING_FORMAT
            and row.get("handle_id") in represented
            and type(row.get("local_source_locator_sha256")) is str
        ):
            locator_receipts.add(row["local_source_locator_sha256"])
    return tuple(
        binding
        for binding in _local_binding_projections(construction_row)
        if binding.namespace_id == namespace_id
        and binding.receipt_sha256 in locator_receipts
    )


def protected_evidence(
    *,
    construction_row: Mapping[str, Any],
    composition_row: Mapping[str, Any],
    composition_sha256: str,
    namespace_id: str,
) -> tuple[LocalCitationBinding, ...]:
    """Rebuild exact provider-visible protected owners without a CLI import."""

    dated_question, _prediction, _question_id = question_inputs(composition_row)
    parent = rehydrate_protected_parent_contributions(
        composition_row,
        compile_typed_operator_spec(dated_question),
        composition_sha256,
    )
    base = _protected_parent_local_evidence(
        composition_row,
        parent,
        namespace_id=namespace_id,
    )
    specialist: tuple[LocalCitationBinding, ...] = ()
    if construction_row.get("terminal_prompt") is not None:
        specialist = _visible_specialist_local_evidence(
            construction_row,
            namespace_id=namespace_id,
        )
    by_receipt = {row.receipt_sha256: row for row in (*base, *specialist)}
    return tuple(by_receipt[key] for key in sorted(by_receipt))


def _terminal_plane_accounting(
    field_name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    token_cap: int,
) -> dict[str, Any]:
    _require(type(token_cap) is int and token_cap > 0, "terminal plane cap changed")
    value = {field_name: [dict(row) for row in rows]}
    encoded = canonical_json_bytes(value)[:-1]
    body = {
        "exact_serialized_field_sha256": hashlib.sha256(encoded).hexdigest(),
        "exact_serialized_utf8_bytes": len(encoded),
        "field_name": field_name,
        "non_borrowable": True,
        "row_count": len(rows),
        "token_cap": token_cap,
        "token_proxy": count_tokens(encoded.decode("utf-8")),
        "within_cap": count_tokens(encoded.decode("utf-8")) <= token_cap,
    }
    return _with_receipt(body)


def _protected_owner_reinjection(
    index: residual.SemanticResidualIndex,
    result: residual.SemanticResidualSearchResult,
    protected: Sequence[LocalCitationBinding],
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    by_binding = {row.receipt_sha256: row for row in protected}
    _require(
        len(by_binding) == len(protected),
        "protected owner population repeated a binding receipt",
    )
    by_cell = index.cell_by_id
    owner_rows: list[dict[str, Any]] = []
    owner_source_ids: list[str] = []
    closure_rows: list[dict[str, Any]] = []
    for ordinal, duplicate in enumerate(result.protected_duplicates, start=1):
        cell = by_cell.get(duplicate.cell_id)
        _require(cell is not None, "protected duplicate lost its semantic cell")
        assert cell is not None
        matches = tuple(
            row
            for row in cell.segments
            if row.receipt_sha256 == duplicate.segment_receipt_sha256
        )
        binding = by_binding.get(duplicate.protected_binding_receipt_sha256)
        _require(
            len(matches) == 1
            and binding is not None
            and binding.candidate_id == duplicate.protected_candidate_id,
            "protected duplicate lost its exact immutable owner",
        )
        segment = matches[0]
        assert binding is not None
        span_sha = identity_sha256(segment.span.identity_payload())
        _require(
            span_sha
            == duplicate.span_identity_sha256
            == identity_sha256(binding.span.identity_payload())
            and segment.quote_sha256 == binding.quote_sha256
            and segment.source_id == binding.source_id
            and segment.partition_id == binding.partition_id,
            "protected duplicate owner does not own the exact selected quote",
        )
        handle = f"P{ordinal:04d}"
        owner_rows.append(
            {
                "created_at": segment.created_at,
                "event_dates": list(segment.event_dates),
                "evidence_handle": handle,
                "owner_binding_receipt_sha256": binding.receipt_sha256,
                "owner_candidate_id": binding.candidate_id,
                "protected_duplicate_receipt_sha256": duplicate.receipt_sha256,
                "quote": segment.quote,
                "quote_sha256": segment.quote_sha256,
                "role": segment.role,
                "segment_receipt_sha256": segment.receipt_sha256,
            }
        )
        owner_source_ids.append(binding.source_id)
        closure_rows.append(
            {
                "evidence_handle": handle,
                "owner_binding_receipt_sha256": binding.receipt_sha256,
                "protected_duplicate_receipt_sha256": duplicate.receipt_sha256,
                "quote_sha256": segment.quote_sha256,
                "segment_receipt_sha256": segment.receipt_sha256,
            }
        )
    closure_body = {
        "every_removed_duplicate_has_exact_provider_visible_owner": True,
        "format": f"{_RESIDUAL_TERMINAL_FORMAT}-lossless-owner-closure-v1",
        "owner_count": len(owner_rows),
        "rows": closure_rows,
        "search_receipt_sha256": result.receipt_sha256,
    }
    return owner_rows, owner_source_ids, _with_receipt(closure_body)


def build_separate_terminal_prompt(
    *,
    dated_question: str,
    current_prediction: str,
    result: residual.SemanticResidualSearchResult,
    residual_index: residual.SemanticResidualIndex,
    protected_evidence: Sequence[LocalCitationBinding],
    policy: SemanticResidualEligibilityPolicy,
    protected_owner_token_cap: int = _DEFAULT_PROTECTED_OWNER_TOKEN_CAP,
) -> tuple[dict[str, Any] | None, str]:
    """Render the historical provider-lossless residual prompt exactly."""

    _require(
        type(result) is residual.SemanticResidualSearchResult
        and not result.fallback_required
        and bool(result.evidence),
        "terminal residual prompt requires packed novel evidence",
    )
    _require(
        result.residual_index_receipt_sha256 == residual_index.receipt_sha256
        and dated_question == result.query.dated_question,
        "terminal residual result escaped its exact index/question",
    )
    _require(
        result.protected_evidence_population_receipt_sha256
        == residual.semantic_residual_protected_evidence_population_receipt(
            residual_index, protected_evidence
        ),
        "terminal protected evidence differs from search-time dedup owners",
    )
    _require(
        result.packed_residual_evidence_tokens
        <= result.residual_evidence_token_cap
        == policy.residual_payload_token_cap,
        "residual search evidence plane exceeded its non-borrowable lane cap",
    )
    _require(
        len(result.evidence) == len(result.local_bindings),
        "novel residual evidence lost local source ownership",
    )
    retained_source_ids = tuple(
        sorted({row.source_id for row in result.attempted_selection})
    )
    group_by_source = residual.semantic_residual_source_group_map(retained_source_ids)
    exact_residual_rows = residual.semantic_residual_terminal_evidence_rows(
        result.evidence
    )
    residual_rows: list[tuple[dict[str, Any], str]] = []
    for row, binding, rendered in zip(
        result.evidence, result.local_bindings, exact_residual_rows, strict=True
    ):
        _require(
            row.candidate_id == binding.candidate_id
            and row.citation_binding_receipt_sha256 == binding.receipt_sha256
            and row.quote_sha256 == binding.quote_sha256
            and row.source_group_handle == group_by_source.get(binding.source_id),
            "novel residual evidence escaped its exact local owner",
        )
        residual_rows.append((dict(rendered), binding.source_id))
    protected_owners, owner_source_ids, owner_closure = _protected_owner_reinjection(
        residual_index, result, protected_evidence
    )
    _require(
        set(owner_source_ids) <= set(retained_source_ids),
        "protected owner escaped the ranked retained source universe",
    )
    evidence = [dict(row) for row, _source_id in residual_rows]
    protected_owners = [
        {**row, "source_group_handle": group_by_source[source_id]}
        for row, source_id in zip(protected_owners, owner_source_ids, strict=True)
    ]
    handles_by_source: dict[str, list[str]] = defaultdict(list)
    for row, source_id in (
        *residual_rows,
        *zip(protected_owners, owner_source_ids, strict=True),
    ):
        handles_by_source[source_id].append(row["evidence_handle"])
    group_mapping_body = {
        "format": f"{_RESIDUAL_TERMINAL_FORMAT}-unified-source-group-map-v1",
        "rows": [
            {
                "evidence_handle_ids": handles_by_source[source_id],
                "source_group_handle": group_by_source[source_id],
                "source_id": source_id,
            }
            for source_id in sorted(handles_by_source)
        ],
        "allocation_algorithm": residual.SOURCE_GROUP_ALLOCATION_FORMAT,
        "retained_source_count": len(retained_source_ids),
        "retained_source_identity_population_sha256": identity_sha256(
            [
                residual.semantic_residual_source_identity_receipt(source_id)
                for source_id in retained_source_ids
            ]
        ),
        "single_opaque_group_namespace": True,
        "visible_source_count": len(handles_by_source),
    }
    group_mapping = _with_receipt(group_mapping_body)
    residual_accounting = _terminal_plane_accounting(
        "residual_evidence", evidence, token_cap=policy.residual_payload_token_cap
    )
    owner_accounting = _terminal_plane_accounting(
        "protected_owner_evidence",
        protected_owners,
        token_cap=protected_owner_token_cap,
    )
    if not residual_accounting["within_cap"]:
        return None, "exact_terminal_residual_evidence_exceeds_cap"
    _require(
        residual_accounting["token_proxy"]
        == result.packed_residual_evidence_tokens
        and residual_accounting["exact_serialized_field_sha256"]
        == result.packed_residual_evidence_sha256,
        "terminal residual plane differs from search-time greedy packing bytes",
    )
    if not owner_accounting["within_cap"]:
        return None, "protected_owner_reinjection_exceeds_cap"
    residual_handles = [row["evidence_handle"] for row in evidence]
    owner_handles = [row["evidence_handle"] for row in protected_owners]
    provider_input = {
        "current_answer": current_prediction,
        "dated_question": dated_question,
        "format": f"{_RESIDUAL_TERMINAL_FORMAT}-provider-input",
        "group_mapping_receipt_sha256": group_mapping["receipt_sha256"],
        "lossless_post_selection_closure": owner_closure,
        "protected_owner_evidence": protected_owners,
        "residual_evidence": evidence,
        "residual_frontier": {
            "all_novel_survivors_protected": (
                result.classified_frontier.all_novel_survivors_protected
            ),
            "packing_closed": result.classified_frontier.closed,
            "complete_leaf_partition": (
                result.classified_frontier.complete_leaf_partition
            ),
            "receipt_sha256": result.classified_frontier.receipt_sha256,
            "support_closure_proven": False,
        },
        "response_schema": {
            "decision": "keep_current|replace",
            "prediction": "nonempty exact text",
            "replacement_requires_at_least_one_residual_handle": residual_handles,
            "used_evidence_handle_ids": [*residual_handles, *owner_handles],
        },
    }
    messages = (
        {"role": "system", "content": _RESIDUAL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                provider_input,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    )
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    complete = prompt_tokens + policy.output_token_reserve
    if complete > policy.hard_complete_chat_token_cap:
        return None, "separate_complete_chat_envelope_exceeds_cap"
    message_list = [dict(row) for row in messages]
    message_bytes = canonical_json_bytes(message_list)
    body = {
        "complete_chat_plus_output_tokens": complete,
        "format": _RESIDUAL_TERMINAL_FORMAT,
        "hard_complete_chat_token_cap": policy.hard_complete_chat_token_cap,
        "messages": message_list,
        "messages_sha256": identity_sha256(message_list),
        "messages_utf8_sha256": hashlib.sha256(message_bytes).hexdigest(),
        "new_provider_calls": 0,
        "non_borrowable_residual_budget": True,
        "owner_reinjection_budget_non_borrowable": True,
        "output_token_reserve": policy.output_token_reserve,
        "parent_prompt_tokens_borrowed": 0,
        "prompt_external_unified_group_mapping": group_mapping,
        "prompt_token_proxy": prompt_tokens,
        "protected_owner_evidence_accounting": owner_accounting,
        "protected_owner_token_cap": protected_owner_token_cap,
        "provider_visible_selected_union_lossless": owner_closure[
            "every_removed_duplicate_has_exact_provider_visible_owner"
        ],
        "provider_input": provider_input,
        "provider_input_sha256": identity_sha256(provider_input),
        "residual_payload_token_cap": policy.residual_payload_token_cap,
        "residual_evidence_accounting": residual_accounting,
        "residual_search_payload_tokens": result.provider_payload_tokens,
        "retained_transformer_token_state_bytes": 0,
        "search_receipt_sha256": result.receipt_sha256,
        "separate_synthesis_call": True,
    }
    assert_gold_blind(body, path="locked_semantic_residual_terminal")
    return _with_receipt(body, "terminal_prompt_receipt_sha256"), "none"


def selected_handle_bindings(
    semantic_index: residual.SemanticResidualIndex,
    search: residual.SemanticResidualSearchResult,
    protected: Sequence[LocalCitationBinding],
    terminal: Mapping[str, Any],
) -> tuple[dict[str, LocalCitationBinding], dict[str, str], tuple[str, ...]]:
    """Rebuild the exact R/P handle owners and authenticated G mapping."""

    protected_by_receipt = {row.receipt_sha256: row for row in protected}
    _require(
        len(protected_by_receipt) == len(protected),
        "R7 protected binding population repeated a receipt",
    )
    handle_bindings: dict[str, LocalCitationBinding] = {}
    for index, binding in enumerate(search.local_bindings, start=1):
        handle_bindings[f"R{index:04d}"] = binding
    for index, duplicate in enumerate(search.protected_duplicates, start=1):
        binding = protected_by_receipt.get(
            duplicate.protected_binding_receipt_sha256
        )
        _require(
            binding is not None
            and binding.candidate_id == duplicate.protected_candidate_id,
            "R7 protected duplicate lost its immutable provider owner",
        )
        assert binding is not None
        handle_bindings[f"P{index:04d}"] = binding
    mapping = _exact_dict(
        terminal.get("prompt_external_unified_group_mapping"),
        "R7 terminal group mapping",
    )
    mapping_body = dict(mapping)
    declared_mapping = require_sha256(
        mapping_body.pop("receipt_sha256", None), "R7 terminal group mapping"
    )
    _require(
        identity_sha256(mapping_body) == declared_mapping,
        "R7 terminal group mapping receipt changed",
    )
    group_by_handle: dict[str, str] = {}
    source_by_handle: dict[str, str] = {}
    for raw in _exact_list(mapping.get("rows"), "R7 terminal group rows"):
        row = _exact_dict(raw, "R7 terminal group row")
        for handle in _exact_list(row.get("evidence_handle_ids"), "R7 mapped handles"):
            _require(handle not in group_by_handle, "R7 handle repeated a G mapping")
            group_by_handle[str(handle)] = str(row["source_group_handle"])
            source_by_handle[str(handle)] = str(row["source_id"])
    _require(
        set(group_by_handle) == set(handle_bindings)
        and all(
            source_by_handle[handle] == binding.source_id
            for handle, binding in handle_bindings.items()
        ),
        "R7 terminal handle population differs from reconstructed exact owners",
    )
    retained_source_ids = tuple(
        sorted({row.source_id for row in search.attempted_selection})
    )
    expected_groups = residual.semantic_residual_source_group_map(retained_source_ids)
    _require(
        all(
            group_by_handle[handle] == expected_groups[binding.source_id]
            for handle, binding in handle_bindings.items()
        )
        and mapping.get("retained_source_count") == len(retained_source_ids),
        "R7 terminal G mapping differs from retained-source allocation",
    )
    return handle_bindings, group_by_handle, retained_source_ids


def ordered_protected_union(
    r7_protected: Sequence[LocalCitationBinding],
    r7_global: Sequence[LocalCitationBinding],
    v6_local: Sequence[LocalCitationBinding],
) -> tuple[LocalCitationBinding, ...]:
    """Preserve P/R/L order while forbidding duplicate exact spans."""

    rows = tuple((*r7_protected, *r7_global, *v6_local))
    span_receipts = tuple(
        identity_sha256(row.span.identity_payload()) for row in rows
    )
    _require(
        len(set(span_receipts)) == len(span_receipts),
        "cumulative R/P/L protected union repeated an exact span",
    )
    return rows


def _terminal_compilation_features(mode: str) -> tuple[bool, bool]:
    _require(mode in _TERMINAL_COMPILATION_FORMAT_BY_MODE, "unsupported terminal mode")
    return (
        mode in {TERMINAL_COMPILATION_MODE_V3, TERMINAL_COMPILATION_MODE_V5},
        mode in {TERMINAL_COMPILATION_MODE_V4, TERMINAL_COMPILATION_MODE_V5},
    )


def compile_answer_plan_core(
    *,
    dated_question: str,
    parent_prediction: str,
    residual_index: Any,
    query: Any,
    protected_owner_universe_bindings: Sequence[Any],
    selected_protected_owner_evidence_rows: Sequence[Mapping[str, Any]],
    residual_result: Any,
    local_result: Any,
    global_result: Any,
    sealed_sources: TerminalSealedSources,
    policy: SemanticGlobalTerminalPolicy,
    terminal_mode: str = TERMINAL_COMPILATION_MODE_V2,
) -> dict[str, Any]:
    """Compile the exact historical answer-plan core from mechanism inputs."""

    enable_links, enable_backfill = _terminal_compilation_features(terminal_mode)
    upstream_projection_sha256s = {
        "global": identity_sha256(global_result.projection()),
        "local": identity_sha256(local_result.projection()),
        "residual": identity_sha256(residual_result.projection()),
    }
    selected_owners = load_selected_protected_owner_evidence(
        selected_protected_owner_evidence_rows
    )
    compilation = compile_semantic_global_terminal(
        dated_question=dated_question,
        parent_prediction=parent_prediction,
        residual_index=residual_index,
        query=query,
        protected_owner_universe_bindings=protected_owner_universe_bindings,
        selected_protected_owner_evidence=selected_owners,
        residual_result=residual_result,
        local_result=local_result,
        global_result=global_result,
        sealed_sources=sealed_sources,
        policy=policy,
        enable_selected_evidence_discourse_links=enable_links,
        enable_post_dedup_backfill=enable_backfill,
    )
    _require(
        compilation.format_id == _TERMINAL_COMPILATION_FORMAT_BY_MODE[terminal_mode],
        "terminal compiler emitted the wrong successor format",
    )
    replayed = replay_semantic_global_terminal(
        dated_question=dated_question,
        parent_prediction=parent_prediction,
        residual_index=residual_index,
        query=query,
        protected_owner_universe_bindings=protected_owner_universe_bindings,
        selected_protected_owner_evidence=selected_owners,
        residual_result=residual_result,
        local_result=local_result,
        global_result=global_result,
        sealed_sources=sealed_sources,
        sealed_compilation=compilation,
        policy=policy,
    )
    _require(
        replayed.projection(include_local=True)
        == compilation.projection(include_local=True),
        "terminal compilation changed during immediate resident replay",
    )
    provider_input = compilation.provider_projection()
    messages = render_final_messages(provider_input)
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    _require(
        tuple(messages) == compilation.fitted.messages
        and identity_sha256(list(messages))
        == compilation.fitted.projection()["messages_sha256"]
        and prompt_tokens == compilation.fitted.prompt_token_proxy
        and prompt_tokens + OUTPUT_TOKEN_RESERVE <= HARD_PROMPT_TOKEN_CAP,
        "terminal answer-plan messages or hard token envelope changed",
    )
    body = {
        "allowed_handle_ids": list(compilation.fitted.allowed_handle_ids),
        "dated_question": dated_question,
        "format": ANSWER_PLAN_FORMAT,
        "handle_group_by_id": dict(compilation.fitted.handle_group_by_id),
        "hard_prompt_token_cap": HARD_PROMPT_TOKEN_CAP,
        "messages_sha256": identity_sha256(list(messages)),
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "parent_prediction": parent_prediction,
        "parent_prediction_sha256": quote_sha256(parent_prediction),
        "preservation_requirements": dict(
            compilation.fitted.preservation_requirements
        ),
        "prompt_token_proxy": prompt_tokens,
        "provider_input": provider_input,
        "provider_input_sha256": identity_sha256(provider_input),
        "route_id": _ROUTE_ID,
        "source_artifact_bindings": sealed_sources.projection(),
        "story_coherence": dict(compilation.fitted.story_coherence),
        "terminal_compilation": compilation.projection(include_local=True),
        "terminal_compilation_receipt_sha256": compilation.receipt_sha256,
        "validation_contract": dict(compilation.fitted.validation_contract),
    }
    _require(
        upstream_projection_sha256s
        == {
            "global": identity_sha256(global_result.projection()),
            "local": identity_sha256(local_result.projection()),
            "residual": identity_sha256(residual_result.projection()),
        },
        "terminal compiler mutated an authenticated upstream retrieval result",
    )
    assert_gold_blind(body, path="semantic_global_terminal_answer_plan_core")
    return body


def _validate_exact_span_support_population(
    value: object,
) -> ExactSpanSupportPopulationReceipt:
    population = _exact_dict(value, "terminal exact-span support population")
    _require(
        set(population) == EXACT_SPAN_SUPPORT_POPULATION_KEYS
        and population.get("format") == EXACT_SPAN_SUPPORT_POPULATION_FORMAT
        and population.get("policy") == EXACT_SPAN_SUPPORT_RANKING_POLICY,
        "terminal exact-span support population schema changed",
    )
    raw_authorities = _exact_list(
        population.get("authorities"),
        "terminal exact-span support authorities",
    )
    authorities: list[ExactSpanSupportAuthority] = []
    for raw in raw_authorities:
        authority = _exact_dict(raw, "terminal exact-span support authority")
        _require(
            set(authority) == EXACT_SPAN_SUPPORT_AUTHORITY_KEYS
            and authority.get("format") == EXACT_SPAN_SUPPORT_FORMAT
            and authority.get("policy") == EXACT_SPAN_SUPPORT_RANKING_POLICY,
            "terminal exact-span support authority schema changed",
        )
        parsed = ExactSpanSupportAuthority(
            exact_span_identity_sha256=require_sha256(
                str(authority.get("exact_span_identity_sha256")),
                "terminal exact-span support identity",
            ),
            authority_candidate_receipt_sha256s=tuple(
                _exact_list(
                    authority.get("authority_candidate_receipt_sha256s"),
                    "terminal exact-span support candidates",
                )
            ),
            authority_source_planes=tuple(
                _exact_list(
                    authority.get("authority_source_planes"),
                    "terminal exact-span support source planes",
                )
            ),
            supported_obligation_ids=tuple(
                _exact_list(
                    authority.get("supported_obligation_ids"),
                    "terminal exact-span direct obligations",
                )
            ),
            source_group_supported_obligation_ids=tuple(
                _exact_list(
                    authority.get("source_group_supported_obligation_ids"),
                    "terminal exact-span source-group obligations",
                )
            ),
            source_group_supported_kinds=tuple(
                _exact_list(
                    authority.get("source_group_supported_kinds"),
                    "terminal exact-span support kinds",
                )
            ),
            matched_query_actions=tuple(
                _exact_list(
                    authority.get("matched_query_actions"),
                    "terminal exact-span query actions",
                )
            ),
            exact_relation_support=authority.get("exact_relation_support"),
            query_temporal_support=authority.get("query_temporal_support"),
            past_event_witness=authority.get("past_event_witness"),
            role=require_text(authority.get("role"), "terminal exact-span role"),
            priority_prefix=tuple(
                _exact_list(
                    authority.get("priority_prefix"),
                    "terminal exact-span priority prefix",
                )
            ),
            receipt_sha256=require_sha256(
                str(authority.get("receipt_sha256")),
                "terminal exact-span support authority",
            ),
        )
        _require(
            parsed.projection() == authority,
            "terminal exact-span support authority projection changed",
        )
        authorities.append(parsed)

    raw_candidates = _exact_dict(
        population.get("plane_candidate_receipt_sha256s"),
        "terminal exact-span support plane candidates",
    )
    _require(
        set(raw_candidates) == set(PLANE_ORDER),
        "terminal exact-span support candidate planes changed",
    )
    parsed_population = ExactSpanSupportPopulationReceipt(
        plane_candidate_receipt_sha256s=tuple(
            (
                plane,
                tuple(
                    _exact_list(
                        raw_candidates.get(plane),
                        f"terminal exact-span support {plane} candidates",
                    )
                ),
            )
            for plane in PLANE_ORDER
        ),
        plane_selection_receipt_sha256s=tuple(
            _exact_list(
                population.get("plane_selection_receipt_sha256s"),
                "terminal exact-span support plane selections",
            )
        ),
        authorities=tuple(authorities),
        receipt_sha256=require_sha256(
            str(population.get("receipt_sha256")),
            "terminal exact-span support population",
        ),
    )
    _require(
        parsed_population.projection() == population,
        "terminal exact-span support population projection changed",
    )
    return parsed_population


def _validate_answer_plan(
    row: Mapping[str, Any],
    question: Mapping[str, Any],
) -> dict[str, Any]:
    plan = _exact_dict(row, "terminal answer plan")
    _require(set(plan) == ANSWER_PLAN_KEYS, "terminal answer-plan schema changed")
    receipt = require_sha256(
        str(plan.get("answer_plan_receipt_sha256")), "terminal answer plan"
    )
    body = {
        key: value
        for key, value in plan.items()
        if key != "answer_plan_receipt_sha256"
    }
    compilation = _exact_dict(
        plan.get("terminal_compilation"), "terminal compilation"
    )
    compilation_format = require_text(
        compilation.get("format"), "terminal compilation format"
    )
    has_backfill = "post_dedup_backfill" in compilation
    backfill_receipt_valid = True
    if has_backfill:
        backfill = _exact_dict(
            compilation.get("post_dedup_backfill"), "terminal post-dedup backfill"
        )
        backfill_body = {
            key: value for key, value in backfill.items() if key != "receipt_sha256"
        }
        backfill_receipt_valid = (
            require_sha256(
                backfill.get("receipt_sha256"), "terminal post-dedup backfill"
            )
            == identity_sha256(backfill_body)
        )
    compilation_body = {
        key: value
        for key, value in compilation.items()
        if key not in {"local_audit", "receipt_sha256"}
    }
    terminal_prompt = _exact_dict(
        compilation.get("terminal_prompt"), "terminal prompt"
    )
    local_audit = _exact_dict(compilation.get("local_audit"), "terminal local audit")
    exact_span_support_population = _validate_exact_span_support_population(
        local_audit.get("exact_span_support_population")
    )
    plane_selections = _exact_list(
        compilation.get("plane_selections"), "terminal plane selections"
    )
    plane_selection_receipts: list[str] = []
    plane_candidate_receipts: list[tuple[str, tuple[str, ...]]] = []
    _require(
        len(plane_selections) == len(PLANE_ORDER),
        "terminal plane selection population changed",
    )
    for expected_plane, raw_selection in zip(
        PLANE_ORDER, plane_selections, strict=True
    ):
        selection = _exact_dict(raw_selection, "terminal plane selection")
        selection_receipt = require_sha256(
            str(selection.get("receipt_sha256")), "terminal plane selection"
        )
        selection_body = {
            key: value
            for key, value in selection.items()
            if key != "receipt_sha256"
        }
        candidate_receipts = tuple(
            require_sha256(str(value), "terminal plane candidate")
            for value in _exact_list(
                selection.get("candidate_receipt_sha256s"),
                "terminal plane candidates",
            )
        )
        _require(
            selection.get("plane") == expected_plane
            and selection_receipt == identity_sha256(selection_body),
            "terminal plane selection receipt/order changed",
        )
        plane_selection_receipts.append(selection_receipt)
        plane_candidate_receipts.append((expected_plane, candidate_receipts))
    local_prompt = _exact_dict(
        local_audit.get("terminal_prompt"), "terminal local prompt"
    )
    global_completion = _exact_dict(
        question.get("global_completion"), "terminal inherited global completion"
    )
    provider_input = _exact_dict(plan.get("provider_input"), "terminal provider input")
    allowed = _exact_list(plan.get("allowed_handle_ids"), "terminal allowed handles")
    handle_groups = _exact_dict(
        plan.get("handle_group_by_id"), "terminal handle groups"
    )
    prompt_tokens = _exact_int(plan.get("prompt_token_proxy"), "terminal prompt tokens")
    output_reserve = _exact_int(
        plan.get("output_token_reserve"), "terminal output reserve"
    )
    hard_cap = _exact_int(plan.get("hard_prompt_token_cap"), "terminal hard cap")
    messages = render_final_messages(provider_input)
    _require(
        receipt == identity_sha256(body)
        and plan.get("format") == ANSWER_PLAN_FORMAT
        and plan.get("route_id") == ROUTE_ID
        and all(
            plan.get(key) == question.get(key)
            for key in (
                "ordinal",
                "question_id",
                "question_sha256",
                "dated_question_sha256",
            )
        )
        and quote_sha256(require_text(plan.get("dated_question"), "dated question"))
        == plan.get("dated_question_sha256")
        and quote_sha256(
            require_text(plan.get("parent_prediction"), "parent prediction")
        )
        == plan.get("parent_prediction_sha256")
        and provider_input.get("dated_question") == plan.get("dated_question")
        and _exact_dict(
            provider_input.get("protected_parent_fallback"),
            "terminal parent fallback",
        ).get("prediction")
        == plan.get("parent_prediction")
        and identity_sha256(provider_input) == plan.get("provider_input_sha256")
        and identity_sha256(list(messages)) == plan.get("messages_sha256")
        and count_chat_prompt_token_proxy(messages) == prompt_tokens
        and prompt_tokens + output_reserve <= hard_cap
        and output_reserve == OUTPUT_TOKEN_RESERVE
        and hard_cap == HARD_PROMPT_TOKEN_CAP
        and len(set(allowed)) == len(allowed)
        and all(type(value) is str and bool(value) for value in allowed)
        and set(handle_groups) == set(allowed)
        and compilation.get("receipt_sha256")
        == plan.get("terminal_compilation_receipt_sha256")
        and identity_sha256(compilation_body)
        == compilation.get("receipt_sha256")
        and compilation_format in set(TERMINAL_COMPILATION_FORMAT_BY_MODE.values())
        and has_backfill
        == (compilation_format in {BACKFILL_FORMAT, LINKED_BACKFILL_FORMAT})
        and backfill_receipt_valid
        and question.get("new_provider_calls") == 0
        and question.get("retained_transformer_token_state_bytes") == 0
        and compilation.get("new_provider_calls") == 0
        and compilation.get("retained_transformer_token_state_bytes") == 0
        and global_completion.get("new_provider_calls") == 0
        and global_completion.get("retained_transformer_token_state_bytes") == 0
        and compilation.get("local_result_receipt_sha256")
        == question.get("v6_result_receipt_sha256")
        and compilation.get("global_result_receipt_sha256")
        == global_completion.get("receipt_sha256")
        and compilation.get("query_receipt_sha256")
        == global_completion.get("query_receipt_sha256")
        and compilation.get("residual_index_receipt_sha256")
        == global_completion.get("residual_index_receipt_sha256")
        and compilation.get("exact_span_support_population_receipt_sha256")
        == exact_span_support_population.receipt_sha256
        and exact_span_support_population.plane_selection_receipt_sha256s
        == tuple(plane_selection_receipts)
        and exact_span_support_population.plane_candidate_receipt_sha256s
        == tuple(plane_candidate_receipts)
        and identity_sha256(
            {
                "format": f"{TERMINAL_COMPILATION_FORMAT}-local-audit-v1",
                "exact_span_support_population": (
                    exact_span_support_population.projection()
                ),
                "local_rows": _exact_list(
                    local_audit.get("local_rows"), "terminal local rows"
                ),
                "mechanism_by_handle": _exact_dict(
                    local_audit.get("mechanism_by_handle"),
                    "terminal local mechanisms",
                ),
            }
        )
        == compilation.get("local_audit_receipt_sha256")
        and terminal_prompt.get("provider_input") == provider_input
        and terminal_prompt.get("messages_sha256") == plan.get("messages_sha256")
        and terminal_prompt.get("prompt_token_proxy") == prompt_tokens
        and terminal_prompt.get("allowed_handle_ids")
        == plan.get("allowed_handle_ids")
        and local_prompt.get("provider_input") == provider_input
        and local_prompt.get("messages_sha256") == plan.get("messages_sha256")
        and local_prompt.get("prompt_token_proxy") == prompt_tokens
        and local_prompt.get("allowed_handle_ids") == plan.get("allowed_handle_ids")
        and local_prompt.get("handle_group_by_id") == plan.get("handle_group_by_id")
        and terminal_prompt.get("story_coherence") == plan.get("story_coherence")
        and local_prompt.get("story_coherence") == plan.get("story_coherence")
        and terminal_prompt.get("preservation_requirements")
        == plan.get("preservation_requirements")
        and local_prompt.get("preservation_requirements")
        == plan.get("preservation_requirements")
        and terminal_prompt.get("validation_contract")
        == plan.get("validation_contract")
        and local_prompt.get("validation_contract")
        == plan.get("validation_contract")
        and compilation.get("sealed_sources")
        == plan.get("source_artifact_bindings"),
        "terminal answer plan failed strict self-authentication",
    )
    return plan




validate_answer_plan = _validate_answer_plan


__all__ = [
    "TERMINAL_COMPILATION_MODE_V5",
    "build_separate_terminal_prompt",
    "compile_answer_plan_core",
    "ordered_protected_union",
    "protected_evidence",
    "question_inputs",
    "selected_handle_bindings",
    "validate_answer_plan",
]
