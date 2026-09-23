"""Gold-blind source-grouped rendering of the sealed conventional r9 packet."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict
from typing import Any, Mapping

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy, count_tokens
from memory_condense.domain.discourse import quote_sha256
from memory_condense.search.packing.source_grouped_packet import (
    RawPacketExcerpt, SessionPacketBlock, render_source_grouped_packet,
)
from tools import assay_hot_v7_spine_episode_fact_reserved_full100 as parent_renderer
from tools.matched_eval.contracts import assert_gold_blind, canonical_json_bytes, identity_sha256
from tools.matched_eval.hot_temporal_reference_chain import (
    MAX_CONTEXT_TOKENS, MAX_WORKSPACE_TOKENS, OUTPUT_TOKEN_RESERVE,
    TemporalReferenceChainError, _global_rows, split_packet,
)

FORMAT = "memory-condense-hot-source-grouped-packet-v1"
PREFIX = (
    "Retrieved memory. Each <S#> groups excerpts from one exact session; labels "
    "do not establish that different sessions concern the same person or event. "
    "An Excerpt timestamp applies to the following G rows until the next timestamp. "
    "G rows include their speaker; E exchanges retain their own dates and user/assistant "
    "owner labels. <REF G#> reuses that exact excerpt. F facts cite raw backing. "
    "Excerpt timestamps are mention times, not necessarily event times. Evidence "
    "order is not relevance or event chronology.\n"
)


def _require(ok: object, message: str) -> None:
    if not ok:
        raise TemporalReferenceChainError(message)


def compose_source_grouped_packet(arm: Mapping[str, Any]) -> dict[str, Any]:
    assert_gold_blind(arm, path="source_grouped.parent")
    _, context, question = split_packet(arm)
    messages = arm["provider_messages"]
    payload = canonical_json_bytes({"messages": messages})
    _require(arm.get("provider_payload_sha256") == hashlib.sha256(payload).hexdigest()
             and arm.get("provider_payload_utf8_bytes") == len(payload), "parent payload binding changed")
    _require(arm.get("context_token_proxy") == count_tokens(context)
             and arm.get("prompt_token_proxy") == count_chat_prompt_token_proxy(messages)
             and arm.get("prompt_workspace_token_proxy") == count_chat_prompt_token_proxy(messages) + OUTPUT_TOKEN_RESERVE,
             "parent token accounting changed")
    _require(count_tokens(context) <= MAX_CONTEXT_TOKENS
             and arm["prompt_workspace_token_proxy"] <= MAX_WORKSPACE_TOKENS, "parent exceeds hard caps")
    globals_ = _global_rows(arm, context)
    parent_rows = [{**row, "chunk_id": row["evidence_id"]} for row in globals_]
    episodes = arm.get("episode_manifests", [])
    globals_by_id = {row["evidence_id"]: row for row in globals_}
    for episode in episodes:
        body = dict(episode)
        digest = body.pop("manifest_sha256", None)
        _require(digest == identity_sha256(body), "episode manifest hash changed")
        for raw in episode["raw_rows"]:
            _require(raw["source_id"] == episode["source_id"]
                     and quote_sha256(raw["text"]) == raw["text_sha256"], "episode raw source or text changed")
        for ref in episode["global_refs"]:
            global_row = globals_by_id[ref["global_evidence_id"]]
            _require(global_row["source_id"] == episode["source_id"]
                     and global_row["raw_text_sha256"] == ref["global_raw_text_sha256"], "episode reference escaped its source")
    facts = arm.get("fact_ledger", {}).get("selected_facts", [])
    raw_context = parent_renderer._render_context(parent_rows, episodes, [])
    # Authenticate the complete block layout, including raw episode text and
    # collision/ref markers, before splitting at known byte positions.
    fact_text = parent_renderer._render_fact_section(
        facts, parent_renderer._provider_label_index(parent_rows, episodes))
    advisory = arm.get("fact_advisory", {}).get("typed_reducer_audit", {}).get("provider_advisory", {}).get("text", "")
    completion = arm.get("numeric_slot_completion", {}).get("provider_text", "")
    tail = "\n\n".join(text for text in (fact_text, advisory, completion) if text)
    _require(context == raw_context + ("\n\n" + tail if tail else ""), "parent rendered context differs from authenticated manifest")
    global_context = parent_renderer._render_context(parent_rows, [], [])
    episode_blocks: list[SessionPacketBlock] = []
    cursor = len(global_context)
    for i, episode in enumerate(episodes, 1):
        # Rendering successive prefixes determines exact boundaries even when
        # untrusted raw text contains apparent packet delimiters.
        through = parent_renderer._render_context(parent_rows, episodes[:i], [])
        block = through[cursor + 2:]
        _require(block.startswith(f"<E{i}>\n"), "episode block boundary changed")
        episode_blocks.append(SessionPacketBlock(f"E{i}", episode["source_id"], block))
        cursor = len(through)
    raw_inputs = [RawPacketExcerpt(row["citation"], row["source_id"], row["created_at"], row["role"], row["text"]) for row in globals_]
    grouped = render_source_grouped_packet(raw_inputs, episode_blocks, tail=tail)
    original = {row.citation: row.text for row in (*raw_inputs, *episode_blocks)}
    _require(len(grouped.bindings) == len(original), "rendered membership changed")
    for binding in grouped.bindings:
        _require(grouped.context[binding.start:binding.end] == original[binding.citation], "raw evidence bytes changed")
    next_messages = [copy.deepcopy(messages[0]), {"role": "user", "content": PREFIX + grouped.context + "\n\nQuestion: " + question + "\nShort answer:"}]
    new_context = grouped.context
    status = "selected"
    if count_tokens(new_context) > MAX_CONTEXT_TOKENS or count_chat_prompt_token_proxy(next_messages) + OUTPUT_TOKEN_RESERVE > MAX_WORKSPACE_TOKENS:
        next_messages, new_context, status = copy.deepcopy(messages), context, "budget_atomic_fallback"
    result = copy.deepcopy(dict(arm))
    encoded = canonical_json_bytes({"messages": next_messages})
    result.update(provider_messages=next_messages, provider_payload_sha256=hashlib.sha256(encoded).hexdigest(),
                  provider_payload_utf8_bytes=len(encoded), context_token_proxy=count_tokens(new_context),
                  prompt_token_proxy=count_chat_prompt_token_proxy(next_messages),
                  prompt_workspace_token_proxy=count_chat_prompt_token_proxy(next_messages) + OUTPUT_TOKEN_RESERVE)
    for metric, value in (("overlay_context_token_delta", "context_token_proxy"), ("overlay_workspace_token_delta", "prompt_workspace_token_proxy")):
        if metric in result:
            result[metric] += result[value] - arm[value]
    audit = {"format": FORMAT, "status": status, "parent_provider_payload_sha256": arm["provider_payload_sha256"],
             "parent_context_sha256": quote_sha256(context), "context_sha256": quote_sha256(new_context),
             "raw_evidence_preserved": True, "tail_preserved": True, "system_prompt_preserved": True,
             "global_excerpt_count": len(globals_), "episode_block_count": len(episodes),
             "bindings": [{**asdict(b), "raw_text_sha256": quote_sha256(original[b.citation])} for b in grouped.bindings] if status == "selected" else [],
             "context_token_delta": result["context_token_proxy"] - arm["context_token_proxy"],
             "provider_calls": 0, "retrieval_changed": False, "frontier_closed": False, "gold_loaded": False}
    result["source_grouped_packet"] = {**audit, "receipt_sha256": identity_sha256(audit)}
    assert_gold_blind(result, path="source_grouped.successor")
    return result
