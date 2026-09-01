"""Current-artifact R7/A1 linked terminal prompt repair.

This module is deliberately separate from the sealed A/B/C answer lifecycle.
It authenticates one existing A1 question and its original R7 terminal plan,
then derives a fourth, provider-safe representation.  Exact source locators and
all provenance/linker receipts remain in ``local_audit``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import EvidenceSpan, quote_sha256

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .selected_evidence_discourse_links import (
    SelectedEvidenceDiscourseLinks,
    SelectedEvidenceLinkInput,
    link_selected_evidence,
)
from .terminal_leaf_metadata import (
    AuthenticatedTerminalLeafMetadata,
    METADATA_AUTHORITY,
    authenticate_selected_leaf_projection,
    compile_terminal_leaf_metadata,
)


FORMAT = "memory-condense-r7-linked-terminal-repair-v1"
PROVIDER_FORMAT = f"{FORMAT}-provider-input-v1"
AUDIT_FORMAT = f"{FORMAT}-local-audit-v1"
REPRESENTATION = "deduplicated_typed_facts_plus_unresolved_raw_linked_metadata"
HARD_TOTAL_TOKEN_CAP = 8_000
OUTPUT_TOKEN_RESERVE = 768
MAX_CHAT_PROMPT_TOKENS = HARD_TOTAL_TOKEN_CAP - OUTPUT_TOKEN_RESERVE

_METADATA_FIELDS = (
    "event_date",
    "source_relation",
    "kind",
    "status",
    "entity_label",
)
_METADATA_TRIM_ORDER = (
    "entity_label",
    "source_relation",
    "status",
    "kind",
    "event_date",
)
_RAW_LOCATOR_KEYS = frozenset(
    {
        "binding",
        "candidate",
        "chunk_id",
        "namespace_id",
        "partition_id",
        "source_id",
        "span",
        "turn_id",
    }
)


class R7LinkedTerminalRepairError(MatchedEvalContractError):
    """The A1/R7 binding or repaired prompt contract changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise R7LinkedTerminalRepairError(message)


def _dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact list")
    return value  # type: ignore[return-value]


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _verify_receipt(row: Mapping[str, Any], label: str) -> str:
    declared = require_sha256(row.get("receipt_sha256"), label)
    body = {key: value for key, value in row.items() if key != "receipt_sha256"}
    _require(declared == identity_sha256(body), f"{label} receipt changed")
    return declared


def _provider_locator_keys(value: object) -> set[str]:
    if type(value) is dict:
        row: dict[str, Any] = value  # type: ignore[assignment]
        return (set(row) & _RAW_LOCATOR_KEYS) | {
            key for child in row.values() for key in _provider_locator_keys(child)
        }
    if type(value) is list:
        return {
            key for child in value for key in _provider_locator_keys(child)
        }
    return set()


def _span(raw: object) -> EvidenceSpan:
    row = _dict(raw, "R7 exact citation span")
    expected = {
        "chunk_id",
        "created_at",
        "end_char",
        "ordinal",
        "quote_sha256",
        "role",
        "source_id",
        "start_char",
        "turn_id",
        "turn_start_char",
    }
    _require(set(row) == expected, "R7 exact citation span schema changed")
    span = EvidenceSpan(
        chunk_id=require_text(row.get("chunk_id"), "R7 span chunk"),
        start_char=row.get("start_char"),  # type: ignore[arg-type]
        end_char=row.get("end_char"),  # type: ignore[arg-type]
        quote_sha256=require_sha256(row.get("quote_sha256"), "R7 span quote"),
        ordinal=row.get("ordinal"),  # type: ignore[arg-type]
        source_id=row.get("source_id"),  # type: ignore[arg-type]
        turn_start_char=row.get("turn_start_char"),  # type: ignore[arg-type]
        turn_id=row.get("turn_id"),  # type: ignore[arg-type]
        role=row.get("role"),  # type: ignore[arg-type]
        created_at=row.get("created_at"),  # type: ignore[arg-type]
    )
    _require(span.identity_payload() == row, "R7 exact citation span changed")
    return span


def _source_terminal(
    source_question: Mapping[str, Any],
    *,
    question_id: str,
    dated_question: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    """Authenticate the source plan and its retained handle/local-row bijection."""

    _require(
        source_question.get("question_id") == question_id,
        "A1/source question IDs differ",
    )
    plan = _dict(source_question.get("terminal_answer_plan"), "R7 terminal plan")
    plan_receipt = require_sha256(
        plan.get("answer_plan_receipt_sha256"), "R7 terminal plan"
    )
    _require(
        plan_receipt
        == identity_sha256(
            {
                key: value
                for key, value in plan.items()
                if key != "answer_plan_receipt_sha256"
            }
        ),
        "R7 terminal plan receipt changed",
    )
    provider = _dict(plan.get("provider_input"), "R7 provider input")
    _require(
        provider.get("dated_question") == dated_question
        and plan.get("dated_question") == dated_question
        and plan.get("question_id") == question_id
        and plan.get("provider_input_sha256") == identity_sha256(provider),
        "A1/source dated question or provider binding changed",
    )
    compilation = _dict(plan.get("terminal_compilation"), "R7 compilation")
    compilation_receipt = require_sha256(
        compilation.get("receipt_sha256"), "R7 compilation"
    )
    _require(
        compilation_receipt == plan.get("terminal_compilation_receipt_sha256")
        and compilation_receipt
        == identity_sha256(
            {
                key: value
                for key, value in compilation.items()
                if key not in {"local_audit", "receipt_sha256"}
            }
        ),
        "R7 compilation receipt changed",
    )
    local = _dict(compilation.get("local_audit"), "R7 compilation local audit")
    local_rows = _list(local.get("local_rows"), "R7 local rows")
    local_receipt_body = {
        "format": f"{compilation.get('format')}-local-audit-v1",
        "exact_span_support_population": local.get("exact_span_support_population"),
        "local_rows": local_rows,
        "mechanism_by_handle": local.get("mechanism_by_handle"),
    }
    _require(
        compilation.get("local_audit_receipt_sha256")
        == identity_sha256(local_receipt_body),
        "R7 local audit receipt changed",
    )

    typed = _dict(provider.get("typed_evidence"), "R7 typed evidence")
    handle_rows = _list(typed.get("handles"), "R7 typed handles")
    item_rows = _list(typed.get("items"), "R7 typed items")
    group_by_handle: dict[str, str] = {}
    for raw in handle_rows:
        row = _dict(raw, "R7 typed handle")
        handle = require_text(row.get("handle_id"), "R7 typed handle ID")
        group = require_text(row.get("group_handle"), "R7 typed group")
        _require(handle not in group_by_handle, "R7 typed handles repeat")
        group_by_handle[handle] = group
    summary_by_handle: dict[str, str] = {}
    for raw in item_rows:
        row = _dict(raw, "R7 typed item")
        handles = _list(row.get("handle_ids"), "R7 typed item handles")
        _require(
            len(handles) == 1 and type(handles[0]) is str,
            "R7 terminal item is not one exact H leaf",
        )
        handle = handles[0]
        _require(
            handle in group_by_handle
            and handle not in summary_by_handle
            and row.get("included") is True,
            "R7 typed item population changed",
        )
        summary_by_handle[handle] = require_text(
            row.get("summary"), "R7 typed summary"
        )
    _require(
        set(summary_by_handle) == set(group_by_handle),
        "R7 typed handles/items are not a bijection",
    )

    binding_by_handle: dict[str, dict[str, Any]] = {}
    for raw in local_rows:
        outer = _dict(raw, "R7 local row")
        typed_row = outer.get("typed_terminal")
        if typed_row is None:
            continue
        terminal = _dict(typed_row, "R7 typed local row")
        if terminal.get("retained_in_final_prompt") is not True:
            continue
        handle = require_text(terminal.get("final_handle_id"), "R7 local handle")
        _require(handle not in binding_by_handle, "R7 local handles repeat")
        outer_binding = _dict(outer.get("binding"), "R7 outer binding")
        outer_candidate = _dict(outer.get("candidate"), "R7 outer candidate")
        binding = _dict(terminal.get("binding"), "R7 terminal binding")
        candidate = _dict(terminal.get("candidate"), "R7 terminal candidate")
        _require(
            binding == outer_binding and candidate == outer_candidate,
            "R7 terminal local row disagrees with its outer binding",
        )
        binding_receipt = _verify_receipt(binding, "R7 local binding")
        candidate_receipt = _verify_receipt(candidate, "R7 local candidate")
        span = _span(binding.get("span"))
        summary = summary_by_handle.get(handle)
        _require(
            summary is not None
            and candidate.get("binding_receipt_sha256") == binding_receipt
            and candidate.get("quote_sha256") == quote_sha256(summary)
            and binding.get("quote_sha256") == quote_sha256(summary)
            and span.quote_sha256 == quote_sha256(summary)
            and binding.get("source_id") == span.source_id,
            "R7 summary did not authenticate to the exact local span",
        )
        binding_by_handle[handle] = {
            "binding": binding,
            "binding_receipt_sha256": binding_receipt,
            "candidate": candidate,
            "candidate_receipt_sha256": candidate_receipt,
            "group_handle": group_by_handle[handle],
            "span": span,
            "span_identity_sha256": identity_sha256(span.identity_payload()),
            "summary": summary,
            "summary_sha256": quote_sha256(summary),
        }
    _require(
        set(binding_by_handle) == set(group_by_handle),
        "R7 local audit does not exactly cover retained provider handles",
    )
    source_audit = {
        "answer_plan_receipt_sha256": plan_receipt,
        "compilation_receipt_sha256": compilation_receipt,
        "local_audit_receipt_sha256": compilation["local_audit_receipt_sha256"],
        "provider_input_sha256": plan["provider_input_sha256"],
        "story_link_local_bindings": deepcopy(
            _dict(local.get("terminal_prompt"), "R7 local terminal prompt").get(
                "story_link_local_bindings", []
            )
        ),
    }
    return provider, source_audit, binding_by_handle


def _edge_kind(relation: str) -> str:
    terms = set(relation.casefold().replace("_", " ").split())
    if terms & {"date", "temporal", "before", "after", "adjacent", "time"}:
        return "temporal"
    if terms & {"event", "episode", "candidate", "comembership", "sequence"}:
        return "event"
    return "entity"


def _edge(left: str, right: str, relation: str) -> dict[str, str]:
    if right < left:
        left, right = right, left
    body = {
        "kind": _edge_kind(relation),
        "left_handle_id": left,
        "relation": relation,
        "right_handle_id": right,
    }
    return {"edge_id": f"E{identity_sha256(body)[:24]}", **body}


def _source_group_edges(
    story: Mapping[str, Any],
    handles_by_group: Mapping[str, tuple[str, ...]],
) -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    for key in ("group_links", "link_overlays"):
        for raw in _list(story.get(key, []), f"R7 story {key}"):
            link = _dict(raw, f"R7 story {key} row")
            if any(name in link for name in ("left_group", "right_group", "basis")):
                groups = (link.get("left_group"), link.get("right_group"))
            else:
                groups = tuple(link.get("group_handles", link.get("groups", ())))
            _require(
                len(groups) >= 2
                and all(type(group) is str and bool(group) for group in groups),
                "R7 source story group link schema changed",
            )
            selected_groups = tuple(dict.fromkeys(groups))
            # A source overlay is useful only when every endpoint survived A1.
            if not all(group in handles_by_group for group in selected_groups):
                continue
            relation = link.get("relation", link.get("basis", "explicit_cross_boundary_link"))
            relation = require_text(relation, "R7 source story relation")
            for left_index, left_group in enumerate(selected_groups):
                for right_group in selected_groups[left_index + 1 :]:
                    for left in handles_by_group[left_group]:
                        for right in handles_by_group[right_group]:
                            rows.append(_edge(left, right, relation))
    return tuple(rows)


def _typed_provider_links(
    links: SelectedEvidenceDiscourseLinks,
) -> tuple[dict[str, Any], ...]:
    """Keep the linker's provider-safe member roles intact.

    Typed discourse is not flattened into pair edges: flattening loses each
    member's relation role, evidence role, and source ordinal.  Exact relation
    IDs and span bindings remain available only in the local audit emitted by
    ``SelectedEvidenceDiscourseLinks``.
    """

    return tuple(link.projection() for link in links.links)


def _metadata_fields(
    metadata: AuthenticatedTerminalLeafMetadata,
    visible: Sequence[str],
) -> dict[str, str]:
    return {
        field: value
        for field in visible
        if (value := getattr(metadata, field)) is not None
    }


def _decorate_memory(
    base_memory: Mapping[str, Any],
    metadata_by_handle: Mapping[str, AuthenticatedTerminalLeafMetadata],
    visible: Sequence[str],
) -> dict[str, Any]:
    memory = deepcopy(dict(base_memory))
    for raw in _list(memory.get("raw_summaries"), "hybrid raw summaries"):
        row = _dict(raw, "hybrid raw summary")
        handle = require_text(row.get("handle_id"), "hybrid raw handle")
        fields = _metadata_fields(metadata_by_handle[handle], visible)
        if fields:
            row["terminal_leaf_metadata"] = fields
    for raw in _list(memory.get("typed_facts"), "hybrid typed facts"):
        fact = _dict(raw, "hybrid typed fact")
        handles = tuple(
            require_text(value, "hybrid fact handle")
            for value in _list(fact.get("handle_ids"), "hybrid fact handles")
        )
        fact_metadata = [
            {"handle_id": handle, **_metadata_fields(metadata_by_handle[handle], visible)}
            for handle in handles
        ]
        if any(len(row) > 1 for row in fact_metadata):
            fact["terminal_leaf_metadata"] = fact_metadata
        for raw_citation in _list(fact.get("citations"), "hybrid citations"):
            citation = _dict(raw_citation, "hybrid citation")
            handle = require_text(citation.get("handle_id"), "hybrid citation handle")
            fields = _metadata_fields(metadata_by_handle[handle], visible)
            if fields:
                citation["terminal_leaf_metadata"] = fields
    if visible:
        memory["terminal_leaf_metadata_authority"] = METADATA_AUTHORITY
    return memory


def compile_r7_linked_terminal_repair(
    a1_question: Mapping[str, Any],
    source_question: Mapping[str, Any],
    *,
    discourse_linker: Callable[
        [Sequence[SelectedEvidenceLinkInput]], SelectedEvidenceDiscourseLinks
    ] = link_selected_evidence,
    hard_total_token_cap: int = HARD_TOTAL_TOKEN_CAP,
    output_token_reserve: int = OUTPUT_TOKEN_RESERVE,
) -> dict[str, Any]:
    """Build one authenticated, budgeted repair prompt without provider calls."""

    _require(
        type(a1_question) is dict and type(source_question) is dict,
        "repair inputs must be exact question objects",
    )
    _require(
        type(hard_total_token_cap) is int
        and type(output_token_reserve) is int
        and hard_total_token_cap > output_token_reserve >= 0,
        "repair prompt envelope changed",
    )
    # Reuse the sealed hybrid compiler as a read-only compatibility oracle.
    # Importing lazily avoids making the sealed CLI depend on this repair arm.
    from tools import run_r7_a1_terminal_answer as sealed_terminal

    _question_row, prompt_rows = sealed_terminal._question_prompt_rows(  # noqa: SLF001
        a1_question
    )
    base_prompt = prompt_rows[2]
    base_messages = _list(base_prompt.get("messages"), "sealed hybrid messages")
    _require(len(base_messages) == 2, "sealed hybrid message population changed")
    base_provider = _dict(
        json.loads(require_text(base_messages[1].get("content"), "hybrid content")),
        "sealed hybrid provider input",
    )
    retained = tuple(
        require_text(value, "A1 retained handle")
        for value in _list(base_prompt.get("presented_handle_ids"), "A1 retained handles")
    )
    _require(
        retained == tuple(base_prompt.get("allowed_handle_ids", ()))
        and len(retained) == len(set(retained)),
        "sealed hybrid retained H population changed",
    )
    question_id = require_text(a1_question.get("question_id"), "A1 question ID")
    dated_question = require_text(a1_question.get("dated_question"), "A1 dated question")
    source_provider, source_receipts, source_bindings = _source_terminal(
        source_question,
        question_id=question_id,
        dated_question=dated_question,
    )

    selection = _dict(a1_question.get("semantic_selection"), "A1 selection")
    leaves_by_handle: dict[str, Any] = {}
    metadata_by_handle: dict[str, AuthenticatedTerminalLeafMetadata] = {}
    for raw in _list(selection.get("leaves"), "A1 selected leaves"):
        leaf = authenticate_selected_leaf_projection(raw)
        leaves_by_handle[leaf.handle_id] = leaf
    for handle in retained:
        _require(handle in leaves_by_handle, "retained H escaped selected A1 leaves")
        leaf = leaves_by_handle[handle]
        source = source_bindings.get(handle)
        _require(
            source is not None
            and leaf.text == source["summary"]
            and leaf.group_handle == source["group_handle"]
            and quote_sha256(leaf.text) == source["span"].quote_sha256,
            "A1 leaf did not bind byte-exactly to its R7 source span",
        )
        metadata_by_handle[handle] = compile_terminal_leaf_metadata(leaf)

    link_inputs: list[SelectedEvidenceLinkInput] = []
    duplicate_link_inputs: list[dict[str, str]] = []
    owner_by_span: dict[str, str] = {}
    for handle in retained:  # deduplication is intentionally after selection
        source = source_bindings[handle]
        span_identity = source["span_identity_sha256"]
        if span_identity in owner_by_span:
            duplicate_link_inputs.append(
                {"excluded_handle_id": handle, "owner_handle_id": owner_by_span[span_identity]}
            )
            continue
        owner_by_span[span_identity] = handle
        link_inputs.append(
            SelectedEvidenceLinkInput(
                handle_id=handle,
                span=source["span"],
                quote=source["summary"],
                source_binding_receipt_sha256=source["binding_receipt_sha256"],
                selected_evidence_receipt_sha256=leaves_by_handle[handle].receipt_sha256,
            )
        )
    discourse = discourse_linker(tuple(link_inputs))
    _require(
        type(discourse) is SelectedEvidenceDiscourseLinks,
        "selected-evidence discourse linker changed output type",
    )

    current_edges = tuple(
        dict(_dict(row, "A1 current graph edge"))
        for row in _list(base_provider.get("graph_links"), "A1 current graph links")
    )
    handles_by_group_mutable: dict[str, list[str]] = {}
    for handle in retained:
        handles_by_group_mutable.setdefault(
            leaves_by_handle[handle].group_handle, []
        ).append(handle)
    handles_by_group = {
        group: tuple(handles) for group, handles in handles_by_group_mutable.items()
    }
    story = _dict(source_provider.get("story_coherence"), "R7 story coherence")
    source_edges = _source_group_edges(story, handles_by_group)
    typed_links = _typed_provider_links(discourse)

    edge_by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}
    provenance_by_edge: dict[str, list[str]] = {}
    for origin, rows in (
        ("current_a1", current_edges),
        ("recovered_source_story", source_edges),
    ):
        for raw in rows:
            row = dict(raw)
            key = (
                str(row["kind"]),
                min(str(row["left_handle_id"]), str(row["right_handle_id"])),
                str(row["relation"]),
                max(str(row["left_handle_id"]), str(row["right_handle_id"])),
            )
            normalized = _edge(key[1], key[3], key[2])
            # Preserve the authenticated current A1 edge byte-for-byte.
            if origin == "current_a1":
                normalized = row  # type: ignore[assignment]
            edge_by_key.setdefault(key, normalized)
            edge_id = str(edge_by_key[key]["edge_id"])
            provenance_by_edge.setdefault(edge_id, []).append(origin)
    current_keys = {
        (
            str(row["kind"]),
            min(str(row["left_handle_id"]), str(row["right_handle_id"])),
            str(row["relation"]),
            max(str(row["left_handle_id"]), str(row["right_handle_id"])),
        )
        for row in current_edges
    }
    extra_keys = sorted(key for key in edge_by_key if key not in current_keys)
    admitted_typed_links = list(typed_links)
    visible_fields = list(_METADATA_FIELDS)

    def provider_for(keys: Sequence[tuple[str, str, str, str]]) -> dict[str, Any]:
        provider = deepcopy(base_provider)
        provider["format"] = PROVIDER_FORMAT
        provider["memory_representation"] = REPRESENTATION
        provider["memory"] = _decorate_memory(
            _dict(base_provider.get("memory"), "sealed hybrid memory"),
            metadata_by_handle,
            visible_fields,
        )
        provider["graph_links"] = [
            *[dict(row) for row in current_edges],
            *[dict(edge_by_key[key]) for key in keys],
        ]
        if admitted_typed_links:
            provider["typed_links"] = [dict(row) for row in admitted_typed_links]
        return provider

    admitted_extra_keys = list(extra_keys)
    system_message = dict(_dict(base_messages[0], "sealed hybrid system message"))

    def render() -> tuple[dict[str, Any], list[dict[str, str]], int]:
        provider = provider_for(admitted_extra_keys)
        messages = [
            system_message,
            {"role": "user", "content": _canonical(provider)},
        ]
        return provider, messages, count_chat_prompt_token_proxy(messages)

    provider_input, messages, prompt_tokens = render()
    trimmed_edge_ids: list[str] = []
    trimmed_typed_link_ids: list[str] = []
    while (
        prompt_tokens + output_token_reserve > hard_total_token_cap
        and admitted_typed_links
    ):
        removed_link = admitted_typed_links.pop()
        trimmed_typed_link_ids.append(str(removed_link["link_id"]))
        provider_input, messages, prompt_tokens = render()
    while prompt_tokens + output_token_reserve > hard_total_token_cap and admitted_extra_keys:
        removed = admitted_extra_keys.pop()
        trimmed_edge_ids.append(str(edge_by_key[removed]["edge_id"]))
        provider_input, messages, prompt_tokens = render()
    trimmed_metadata_fields: list[str] = []
    for field in _METADATA_TRIM_ORDER:
        if prompt_tokens + output_token_reserve <= hard_total_token_cap:
            break
        visible_fields.remove(field)
        trimmed_metadata_fields.append(field)
        provider_input, messages, prompt_tokens = render()
    _require(
        prompt_tokens + output_token_reserve <= hard_total_token_cap,
        "sealed hybrid evidence population cannot fit the requested envelope",
    )
    _require(
        tuple(base_prompt["presented_handle_ids"]) == retained
        and tuple(base_prompt["allowed_handle_ids"]) == retained,
        "repair changed retained H membership/order",
    )
    assert_gold_blind(provider_input, path="r7_linked_terminal_repair.provider")
    _require(
        not _provider_locator_keys(provider_input),
        "repair provider input leaked raw source locators",
    )

    local_binding_rows = []
    for handle in retained:
        source = source_bindings[handle]
        local_binding_rows.append(
            {
                "binding": source["binding"],
                "candidate": source["candidate"],
                "handle_id": handle,
                "leaf_receipt_sha256": leaves_by_handle[handle].receipt_sha256,
                "metadata": metadata_by_handle[handle].projection(),
                "span_identity_sha256": source["span_identity_sha256"],
                "summary_sha256": source["summary_sha256"],
            }
        )
    audit_body = {
        "base_hybrid_local_bindings": {
            "graph_edge_receipt_sha256s": deepcopy(
                base_prompt.get("graph_edge_receipt_sha256s", [])
            ),
            "raw_leaf_bindings": deepcopy(base_prompt.get("raw_leaf_bindings", [])),
            "retained_population_sha256": base_prompt.get(
                "retained_population_sha256"
            ),
            "source_fact_closure_receipt_sha256": base_prompt.get(
                "source_fact_closure_receipt_sha256"
            ),
            "typed_fact_bindings": deepcopy(
                base_prompt.get("typed_fact_bindings", [])
            ),
        },
        "base_hybrid_prompt_row_receipt_sha256": base_prompt[
            "prompt_row_receipt_sha256"
        ],
        "discourse_compilation": discourse.projection(),
        "discourse_local_bindings": [dict(row) for row in discourse.local_bindings],
        "duplicate_post_selection_link_inputs": duplicate_link_inputs,
        "edge_provenance": [
            {"edge_id": edge_id, "origins": list(dict.fromkeys(origins))}
            for edge_id, origins in sorted(provenance_by_edge.items())
        ],
        "format": AUDIT_FORMAT,
        "local_handle_bindings": local_binding_rows,
        "metadata_visible_fields": list(visible_fields),
        "question_receipt_sha256": require_sha256(
            a1_question.get("question_receipt_sha256"), "A1 question"
        ),
        "selection_receipt_sha256": require_sha256(
            selection.get("receipt_sha256"), "A1 selection"
        ),
        "source_receipts": source_receipts,
        "trimmed_extra_edge_ids": trimmed_edge_ids,
        "trimmed_metadata_fields": trimmed_metadata_fields,
        "trimmed_typed_link_ids": trimmed_typed_link_ids,
    }
    local_audit = {**audit_body, "receipt_sha256": identity_sha256(audit_body)}
    body = {
        "allowed_handle_ids": list(retained),
        "format": FORMAT,
        "hard_total_token_cap": hard_total_token_cap,
        "local_audit": local_audit,
        "memory_representation": REPRESENTATION,
        "messages": messages,
        "messages_sha256": identity_sha256(messages),
        "new_provider_calls": 0,
        "output_token_reserve": output_token_reserve,
        "presented_handle_ids": list(retained),
        "prompt_token_proxy": prompt_tokens,
        "provider_input": provider_input,
        "provider_input_sha256": identity_sha256(provider_input),
        "question_id": question_id,
        "question_sha256": require_sha256(a1_question.get("question_sha256"), "A1 question"),
        "retained_transformer_token_state_bytes": 0,
    }
    return {**body, "receipt_sha256": identity_sha256(body)}


__all__ = [
    "AUDIT_FORMAT",
    "FORMAT",
    "HARD_TOTAL_TOKEN_CAP",
    "MAX_CHAT_PROMPT_TOKENS",
    "OUTPUT_TOKEN_RESERVE",
    "PROVIDER_FORMAT",
    "REPRESENTATION",
    "R7LinkedTerminalRepairError",
    "compile_r7_linked_terminal_repair",
]
