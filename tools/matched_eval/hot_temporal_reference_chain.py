"""Bounded mention chronology over an already sealed provider packet.

This index is an address aid, not a latest-state reducer. It introduces no
event dates, entity joins, correction edges, answers, or closure certificates.
Every G label is authenticated against the parent's exact visible bytes.
"""

from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy, count_tokens
from memory_condense.domain.discourse import quote_sha256

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_text,
)
from .typed_operator_spec import normalized_terms


FORMAT = "memory-condense-hot-temporal-reference-chain-v1"
MAX_CONTEXT_TOKENS = 10_000
MAX_WORKSPACE_TOKENS = 11_000
OUTPUT_TOKEN_RESERVE = 256
MAX_INDEX_TOKENS = 192
MAX_REFERENCES = 24
POLICY = (
    "The mention index is a partial reading aid, not event chronology or proof "
    "of latest state. S labels distinguish exact sources, not people. Read the "
    "G text for entity, event time and completed status; ownership or a recap "
    "does not date a purchase. A later contradiction is not a correction "
    "without explicit revision evidence. Respect the question date."
)
_EXTRA_STOP = frozenset(normalized_terms(
    "current currently latest most recently recent final updated revised corrected "
    "type brand use using purchase purchased bought acquire acquired visit visited "
    "many much passed since first last now still say said tell told"
))
_RECOMMEND = re.compile(r"\b(?:recommend\w*|suggest\w*)\b", re.I)
_ROUTES = (
    ("current_state", re.compile(r"\b(?:current(?:ly)?|right now|still use)\b", re.I)),
    ("latest_event", re.compile(
        r"\b(?:most recent(?:ly)?|latest|last (?:time|visit|purchase)|"
        r"since (?:I |we )?last)\b", re.I)),
    ("explicit_revision", re.compile(r"\b(?:corrected|revised|updated)\b", re.I)),
)


class TemporalReferenceChainError(MatchedEvalContractError):
    """A parent packet, reference, or budget is inconsistent."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise TemporalReferenceChainError(message)


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "receipt_sha256": identity_sha256(body)}


def split_packet(arm: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return prefix, exact context, and dated question from the v7 renderer."""
    messages = arm.get("provider_messages")
    _require(type(messages) is list and len(messages) == 2, "expected two messages")
    for role, message in zip(("system", "user"), messages, strict=True):
        _require(
            type(message) is dict and set(message) == {"role", "content"}
            and message["role"] == role and type(message["content"]) is str,
            "parent role/content shape changed",
        )
    user = messages[1]["content"]
    head, marker, tail = user.rpartition("\n\nQuestion: ")
    _require(bool(marker) and tail.endswith("\nShort answer:"), "question framing changed")
    question = tail.removesuffix("\nShort answer:")
    require_text(question, "dated question")
    prefix, marker, context = head.partition("\n<G1>\n")
    _require(bool(marker), "protected global context is missing")
    return prefix + "\n", "<G1>\n" + context, question


def question_route(dated_question: str) -> tuple[str, tuple[str, ...]]:
    """Route by question language only, never by source/benchmark identifiers."""
    body = re.sub(r"^\[Question asked at [^\]]+\]\s*", "", dated_question)
    terms = tuple(term for term in normalized_terms(body) if term not in _EXTRA_STOP)
    if _RECOMMEND.search(body):
        return "not_applicable", terms
    for route, pattern in _ROUTES:
        if pattern.search(body):
            return route, terms
    return "not_applicable", terms


def _global_rows(arm: Mapping[str, Any], context: str) -> list[dict[str, Any]]:
    manifest = arm.get("global_citation_manifest")
    _require(type(manifest) is dict, "global citation manifest is missing")
    body = dict(manifest)
    receipt = body.pop("receipt_sha256", None)
    _require(receipt == identity_sha256(body), "global citation manifest receipt changed")
    entries = body.get("entries")
    _require(type(entries) is list and bool(entries), "global entries are missing")
    _require(
        [row["evidence_id"] for row in entries] == arm.get("rendered_parent_evidence_ids"),
        "global entries escaped the rendered parent population",
    )
    rows = []
    for index, entry in enumerate(entries, 1):
        label = f"G{index}"
        _require(entry.get("citation") == label, "global citation order changed")
        marker = f"<{label}>\n"
        # An apparent nested G marker in raw data is ambiguous: fail closed.
        _require(len(re.findall(r"(?m)^" + re.escape(marker), context)) == 1,
                 "ambiguous global citation marker")
        start = context.index(marker) + len(marker)
        if index < len(entries):
            end = context.find(f"\n\n<G{index + 1}>\n", start)
            _require(end >= start, "global reference boundary changed")
        else:
            match = re.search(r"\n\n<(?:E1>|FACTS |TYPED_REDUCTION_ADVISORY )", context[start:])
            end = start + match.start() if match else len(context)
        header = f"[{entry['created_at']} | {entry['role']}] "
        block = context[start:end]
        _require(block.startswith(header), "global timestamp or role changed")
        raw = block[len(header):]
        _require(quote_sha256(raw) == entry["raw_text_sha256"], "global visible text hash changed")
        require_text(entry["source_id"], "exact source ID")
        rows.append({**entry, "text": raw})
    return rows


def _instant(value: str) -> datetime | None:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return result.astimezone(timezone.utc) if result.tzinfo is not None else None


def _index_text(rows: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    # Exact source identity only. Never split opaque addresses on :: or infer
    # user affinity from a benchmark partition/question prefix.
    source_labels: dict[str, str] = {}
    groups: dict[tuple[str, str], list[str]] = {}
    bindings = []
    for row in rows:
        source = row["source_id"]
        source_label = source_labels.setdefault(source, f"S{len(source_labels) + 1}")
        # UTC calendar date is a compact mention-date label; full instants and
        # original offsets remain bound in the audit and visible G headers.
        mention_date = _instant(row["created_at"]).date().isoformat()
        groups.setdefault((mention_date, source_label), []).append(row["citation"])
        bindings.append({
            key: row[key] for key in (
                "citation", "created_at", "evidence_id", "raw_text_sha256", "role",
                "row_sha256", "source_id",
            )
        } | {"source_label": source_label, "mention_date_utc": mention_date})
    lines = ["Mention index (UTC; partial):"]
    lines.extend(f"{date} {source}:" + ",".join(labels)
                 for (date, source), labels in groups.items())
    return "\n".join(lines), bindings


def compose_temporal_reference_chain(arm: Mapping[str, Any]) -> dict[str, Any]:
    """Add an atomic, capped reference index; retain every parent evidence byte.

    All lexically eligible user G rows must fit, otherwise the complete parent
    prompt is reused. No ranking/truncation can silently discard a later or
    conflicting eligible mention. Episode-only rows remain visible but unindexed.
    """
    assert_gold_blind(arm, path="temporal_chain.parent")
    prefix, context, question = split_packet(arm)
    messages = arm["provider_messages"]
    encoded = canonical_json_bytes({"messages": messages})
    _require(arm.get("provider_payload_sha256") == hashlib.sha256(encoded).hexdigest()
             and arm.get("provider_payload_utf8_bytes") == len(encoded),
             "parent provider payload binding changed")
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    _require(arm.get("context_token_proxy") == count_tokens(context)
             and arm.get("prompt_token_proxy") == prompt_tokens
             and arm.get("prompt_workspace_token_proxy") == prompt_tokens + OUTPUT_TOKEN_RESERVE,
             "parent token accounting changed")
    _require(count_tokens(context) <= MAX_CONTEXT_TOKENS
             and prompt_tokens + OUTPUT_TOKEN_RESERVE <= MAX_WORKSPACE_TOKENS,
             "parent exceeds hard caps")
    route, terms = question_route(question)
    rows = _global_rows(arm, context)
    candidates, undated = [], []
    for row in rows:
        if row["role"] != "user" or not set(terms).intersection(normalized_terms(row["text"])):
            continue
        (candidates if _instant(row["created_at"]) else undated).append(row)
    candidates.sort(key=lambda row: (_instant(row["created_at"]), int(row["citation"][1:])))
    reason = "selected"
    if route == "not_applicable":
        reason = "not_applicable"
    elif undated:
        reason = "eligible_reference_has_no_aware_timestamp"
    elif not terms or not candidates:
        reason = "no_lexical_user_references"
    elif len(candidates) > MAX_REFERENCES:
        reason = "reference_cap_atomic_fallback"
    index_text, bindings = _index_text(candidates) if reason == "selected" else ("", [])
    if count_tokens(index_text) > MAX_INDEX_TOKENS:
        reason = "index_cap_atomic_fallback"
    successor_messages = copy.deepcopy(messages)
    successor_context = context
    if reason == "selected":
        successor_context = context + "\n\n" + index_text
        successor_messages[0]["content"] += "\n\n" + POLICY
        successor_messages[1]["content"] = (
            prefix + successor_context + "\n\nQuestion: " + question + "\nShort answer:"
        )
        if count_tokens(successor_context) > MAX_CONTEXT_TOKENS or (
            count_chat_prompt_token_proxy(successor_messages) + OUTPUT_TOKEN_RESERVE
            > MAX_WORKSPACE_TOKENS
        ):
            reason = "prompt_cap_atomic_fallback"
    if reason != "selected":
        successor_messages = copy.deepcopy(messages)
        successor_context, index_text, bindings = context, "", []
    result = copy.deepcopy(dict(arm))
    payload = canonical_json_bytes({"messages": successor_messages})
    successor_tokens = count_chat_prompt_token_proxy(successor_messages)
    result.update(
        provider_messages=successor_messages,
        provider_payload_sha256=hashlib.sha256(payload).hexdigest(),
        provider_payload_utf8_bytes=len(payload),
        context_token_proxy=count_tokens(successor_context),
        prompt_token_proxy=successor_tokens,
        prompt_workspace_token_proxy=successor_tokens + OUTPUT_TOKEN_RESERVE,
    )
    for metric, value in (("overlay_context_token_delta", "context_token_proxy"),
                          ("overlay_workspace_token_delta", "prompt_workspace_token_proxy")):
        if metric in result:
            result[metric] += result[value] - arm[value]
    result["temporal_reference_chain"] = _seal({
        "format": FORMAT,
        "route": route,
        "status": reason,
        "question_sha256": quote_sha256(question),
        "query_terms": list(terms),
        "parent_provider_payload_sha256": arm["provider_payload_sha256"],
        "parent_context_sha256": quote_sha256(context),
        "parent_evidence_bytes_preserved": True,
        "candidate_citations": [row["citation"] for row in candidates],
        "undated_candidate_citations": [row["citation"] for row in undated],
        "bindings": bindings,
        "provider_index_text": index_text,
        "provider_index_token_count": count_tokens(index_text),
        "provider_index_text_sha256": quote_sha256(index_text),
        "context_token_delta": result["context_token_proxy"] - arm["context_token_proxy"],
        "workspace_token_delta": result["prompt_workspace_token_proxy"] - arm["prompt_workspace_token_proxy"],
        "entity_links_added": 0,
        "revision_edges_added": 0,
        "frontier_closed": False,
        "provider_calls": 0,
        "gold_loaded": False,
    })
    assert_gold_blind(result, path="temporal_chain.successor")
    return result
