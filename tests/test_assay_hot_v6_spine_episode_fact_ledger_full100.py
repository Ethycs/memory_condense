from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256
from tools import assay_hot_retrieval_1m as hot
from tools import assay_hot_v6_spine_episode_fact_ledger_full100 as assay
from tools import assay_hot_v7_spine_episode_fact_reserved_full100 as successor
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.hot_v3_user_led_envelope_shadow import (
    UserLedEnvelopeShadowEnvelope,
    UserLedEnvelopeShadowIndex,
)
from tools.matched_eval.hot_v5_user_spine_prompt import OPERATION_AWARE_SYSTEM_PROMPT
from tools.matched_eval.query_guided_scan import CachedContentRow, _sentence_windows


def _sha(label: str) -> str:
    return quote_sha256(label)


def _row(
    chunk_id: str,
    text: str,
    *,
    source_id: str,
    turn_id: str,
    role: str,
    ordinal: int,
) -> CachedContentRow:
    return CachedContentRow(
        namespace_id="a" * 64,
        partition_id=source_id.split("::", 1)[0],
        chunk_id=chunk_id,
        turn_id=turn_id,
        source_id=source_id,
        role=role,
        created_at=f"2026-01-{ordinal + 1:02d}T00:00:00+00:00",
        ordinal=ordinal,
        turn_start_char=0,
        turn_end_char=len(text),
        text=text,
        text_sha256=quote_sha256(text),
        token_count=count_tokens(text),
        sentence_windows=_sentence_windows(text),
    )


def _envelope(
    envelope_id: str, source_id: str, rows: tuple[CachedContentRow, ...]
) -> UserLedEnvelopeShadowEnvelope:
    return UserLedEnvelopeShadowEnvelope(
        envelope_id=envelope_id,
        source_id=source_id,
        exchange_kind="user_led",
        opener_turn_id=rows[0].turn_id,
        turn_ids=tuple(dict.fromkeys(row.turn_id for row in rows)),
        turn_ordinals=tuple(
            row.ordinal
            for row in rows
            if row.turn_id
            not in {prior.turn_id for prior in rows[: rows.index(row)]}
        ),
        chunk_ids=tuple(row.chunk_id for row in rows),
        chunk_turn_ids=tuple(row.turn_id for row in rows),
        chunk_token_counts=tuple(row.token_count for row in rows),
    )


def _exact_parent(
    row: CachedContentRow, *, raw_text: str | None = None
) -> dict[str, object]:
    raw = row.text if raw_text is None else raw_text
    logical_id = row.chunk_id if raw == row.text else _sha(f"excerpt:{raw}")
    rendered = f"[{row.created_at} | {row.role}] {raw}"
    result: dict[str, object] = {
        "chunk_id": logical_id,
        "created_at": row.created_at,
        "evidence_id": logical_id,
        "raw_text": raw,
        "raw_text_sha256": quote_sha256(raw),
        "rendered_text": rendered,
        "rendered_text_sha256": quote_sha256(rendered),
        "role": row.role,
        "source_id": row.source_id,
        "turn_id": row.turn_id,
    }
    if logical_id != row.chunk_id:
        result["backing_chunk_id"] = row.chunk_id
        result["excerpt_occurrence"] = True
    return result


def _index() -> UserLedEnvelopeShadowIndex:
    source_a = "p0::session-a"
    source_b = "p0::session-b"
    source_c = "p0::session-c"
    rows = (
        _row(
            _sha("u-table"),
            "I bought the oak dining table for $800 yesterday.",
            source_id=source_a,
            turn_id="turn-u-table",
            role="user",
            ordinal=1,
        ),
        _row(
            _sha("a-table"),
            "That table should work well in the dining room.",
            source_id=source_a,
            turn_id="turn-a-table",
            role="assistant",
            ordinal=2,
        ),
        _row(
            _sha("u-book"),
            "I finished reading a history book.",
            source_id=source_a,
            turn_id="turn-u-book",
            role="user",
            ordinal=3,
        ),
        _row(
            _sha("a-book"),
            "You could try another biography next.",
            source_id=source_a,
            turn_id="turn-a-book",
            role="assistant",
            ordinal=4,
        ),
        _row(
            _sha("u-bike"),
            "I replaced my bicycle chain last week.",
            source_id=source_b,
            turn_id="turn-u-bike",
            role="user",
            ordinal=5,
        ),
        _row(
            _sha("a-bike"),
            "The repaired bike is ready to ride.",
            source_id=source_b,
            turn_id="turn-a-bike",
            role="assistant",
            ordinal=6,
        ),
        _row(
            _sha("u-soccer"),
            "I participated in the fundraiser in June.",
            source_id=source_c,
            turn_id="turn-u-soccer",
            role="user",
            ordinal=7,
        ),
        _row(
            _sha("a-soccer"),
            "The charity soccer tournament raised a lot of money.",
            source_id=source_c,
            turn_id="turn-a-soccer",
            role="assistant",
            ordinal=8,
        ),
        _row(
            _sha("u-movie"),
            "I watched a movie in March.",
            source_id=source_c,
            turn_id="turn-u-movie",
            role="user",
            ordinal=9,
        ),
        _row(
            _sha("a-movie"),
            "It was an enjoyable comedy.",
            source_id=source_c,
            turn_id="turn-a-movie",
            role="assistant",
            ordinal=10,
        ),
    )
    envelopes = (
        _envelope("env-table", source_a, rows[0:2]),
        _envelope("env-book", source_a, rows[2:4]),
        _envelope("env-bike", source_b, rows[4:6]),
        _envelope("env-soccer", source_c, rows[6:8]),
        _envelope("env-movie", source_c, rows[8:10]),
    )
    by_id = {row.chunk_id: row for row in rows}
    membership = {
        chunk_id: envelope.envelope_id
        for envelope in envelopes
        for chunk_id in envelope.chunk_ids
    }
    return UserLedEnvelopeShadowIndex(
        namespace_id="a" * 64,
        cache_receipt_sha256="b" * 64,
        envelopes=envelopes,
        row_by_chunk_id=MappingProxyType(by_id),
        envelope_by_chunk_id=MappingProxyType(membership),
    )


def test_selector_ranks_relevant_user_led_exchange_before_distractor() -> None:
    selected = assay.select_spine_episodes(
        _index(),
        active_source_ids=("p0::session-a",),
        dated_question="[Question asked at 2026-02-01] How much did I spend on the dining table?",
    )
    assert selected["selected_groups"][0]["envelope_id"] == "env-table"
    assert selected["selected_groups"][0]["rows"][0]["role"] == "user"


def test_dated_header_does_not_create_temporal_intent() -> None:
    selected = assay.select_spine_episodes(
        _index(),
        active_source_ids=("p0::session-a",),
        dated_question="[Question asked at 2026-02-01] What table did I buy?",
    )
    winner = selected["selected_groups"][0]
    assert winner["envelope_id"] == "env-table"
    assert winner["rank_audit"]["date_hits"] == 0


def test_selector_confines_candidates_to_active_opaque_sources() -> None:
    selected = assay.select_spine_episodes(
        _index(),
        active_source_ids=("p0::session-b",),
        dated_question="[Question asked at 2026-02-01] What did I replace on my bike?",
    )
    assert {group["source_id"] for group in selected["selected_groups"]} == {
        "p0::session-b"
    }


def test_generic_temporal_query_follows_fast_packet_source_anchor() -> None:
    selected = assay.select_spine_episodes(
        _index(),
        active_source_ids=("p0::session-c",),
        dated_question=(
            "[Question asked at 2026-02-01] Which events did I participate in last year?"
        ),
        parent_packed_rows=(
            {
                "source_id": "p0::session-c",
                "raw_text": "Details for the charity soccer tournament fundraiser.",
            },
        ),
    )
    winner = selected["selected_groups"][0]
    assert winner["envelope_id"] == "env-soccer"
    assert winner["rank_audit"]["source_anchor_hits"] > 0
    assert winner["rank_audit"]["date_hits"] == 1


def test_hydrated_assistant_rows_keep_their_user_lead() -> None:
    selected = assay.select_spine_episodes(
        _index(),
        active_source_ids=("p0::session-a",),
        dated_question="[Question asked at 2026-02-01] What table did I buy?",
    )
    for group in selected["selected_groups"]:
        for row in group["rows"]:
            if row["role"] == "assistant":
                assert row["opener_user_chunk_id"] == group["opener_user_chunk_id"]
                assert row["opener_user_turn_id"] == group["opener_user_turn_id"]


def test_fact_compiler_is_downstream_of_selected_episodes_only() -> None:
    selected = assay.select_spine_episodes(
        _index(),
        active_source_ids=("p0::session-a",),
        dated_question="[Question asked at 2026-02-01] How much was the table?",
    )
    # Restrict the test to the winning episode; the non-winning book exchange
    # is intentionally absent from the compiler input.
    fact_rows = assay._episode_fact_rows(selected["selected_groups"][:1])
    from tools.matched_eval.hot_v6_query_fact_ledger import compile_query_fact_ledger

    ledger = compile_query_fact_ledger(
        "[Question asked at 2026-02-01] How much was the table?", fact_rows
    )
    assert {fact.envelope_id for fact in ledger.facts} == {"env-table"}
    assert all(fact.user_lead_evidence_id == _sha("u-table") for fact in ledger.facts)


def test_fact_selection_has_a_non_borrowing_token_cap() -> None:
    selected = assay.select_spine_episodes(
        _index(),
        active_source_ids=("p0::session-a",),
        dated_question="[Question asked at 2026-02-01] How much was the table?",
    )
    from tools.matched_eval.hot_v6_query_fact_ledger import compile_query_fact_ledger

    ledger = compile_query_fact_ledger(
        "[Question asked at 2026-02-01] How much was the table?",
        assay._episode_fact_rows(selected["selected_groups"]),
    )
    compact = assay._compact_fact_selection(ledger)
    assert compact["selected_fact_count"] <= assay.MAX_FACTS
    assert compact["selected_fact_token_count"] <= assay.MAX_FACT_TOKENS
    assert compact["compiled_ledger_receipt_sha256"] == ledger.receipt_sha256


def test_source_conservation_accepts_virtual_excerpt_and_rejects_tampered_quote() -> None:
    index = _index()
    candidates = assay._candidate_rows(index, ("p0::session-a",))
    parent = [
        {
            "backing_chunk_id": _sha("u-table"),
            "chunk_id": _sha("logical excerpt"),
            "evidence_id": _sha("logical excerpt"),
            "raw_text": "oak dining table",
            "raw_text_sha256": quote_sha256("oak dining table"),
            "source_id": "p0::session-a",
        }
    ]
    assert assay._conserves_parent_sources(
        parent,
        candidates=candidates,
        active_source_ids=("p0::session-a",),
    ) == (True, "validated")
    parent[0]["raw_text"] = "not in the backing row"
    assert assay._conserves_parent_sources(
        parent,
        candidates=candidates,
        active_source_ids=("p0::session-a",),
    )[0] is False


def test_exact_operation_a_fallback_changes_only_system_message() -> None:
    evidence = {
        "chunk_id": "chunk-1",
        "evidence_id": "chunk-1",
        "source_id": "p0::session-a",
    }
    source_messages = [
        {"role": "system", "content": "old system"},
        {"role": "user", "content": "exact raw evidence and question"},
    ]
    arm = {
        "context_token_proxy": 10,
        "packed_chunk_ids": ["chunk-1"],
        "packed_evidence": [evidence],
        "provider_messages": source_messages,
    }
    fallback = assay._exact_operation_a_fallback(arm, "test")
    assert fallback["provider_messages"][0]["content"] == OPERATION_AWARE_SYSTEM_PROMPT
    assert fallback["provider_messages"][1] == source_messages[1]
    assert fallback["mode"] == "exact_operation_a_raw_fail_open"


def test_context_keeps_global_and_episode_blocks_separate() -> None:
    group = {
        "rows": [
            {
                "chunk_id": "episode-user",
                "created_at": "2026-01-01",
                "role": "user",
                "source_id": "p0::session-a",
                "text": "episode text",
            }
        ]
    }
    context = assay._render_context(
        [
            {
                "created_at": "2025-01-01",
                "raw_text": "global text",
                "role": "user",
                "source_id": "p0::session-a",
            }
        ],
        [group],
        [],
    )
    assert context.startswith("<G1>\n")
    assert "\n\n<E1 envelope=unknown source=unknown owner_user=unknown>" in context
    assert "<U evidence=episode-user source=p0::session-a" in context


def test_selector_rejects_duplicate_active_source_addresses() -> None:
    with pytest.raises(ValueError, match="active sources changed"):
        assay.select_spine_episodes(
            _index(),
            active_source_ids=("p0::session-a", "p0::session-a"),
            dated_question="[Question asked at 2026-02-01] What table did I buy?",
        )


def test_selector_receipt_is_independent_of_index_envelope_iteration_order() -> None:
    original = _index()
    reversed_index = UserLedEnvelopeShadowIndex(
        namespace_id=original.namespace_id,
        cache_receipt_sha256=original.cache_receipt_sha256,
        envelopes=tuple(reversed(original.envelopes)),
        row_by_chunk_id=original.row_by_chunk_id,
        envelope_by_chunk_id=original.envelope_by_chunk_id,
    )
    kwargs = {
        "active_source_ids": ("p0::session-a", "p0::session-b"),
        "dated_question": (
            "[Question asked at 2026-02-01] What did I buy and replace?"
        ),
    }
    first = assay.select_spine_episodes(original, **kwargs)
    second = assay.select_spine_episodes(reversed_index, **kwargs)
    assert first == second
    assert first["receipt_sha256"] == second["receipt_sha256"]


def test_long_assistant_boilerplate_does_not_evict_complete_user_lead() -> None:
    source = "p0::long-exchange"
    user = _row(
        _sha("short-user-lead"),
        "I bought the cedar desk yesterday.",
        source_id=source,
        turn_id="long-user",
        role="user",
        ordinal=1,
    )
    assistant = _row(
        _sha("long-assistant"),
        " ".join(["boilerplate"] * 600) + ".",
        source_id=source,
        turn_id="long-assistant",
        role="assistant",
        ordinal=2,
    )
    envelope = _envelope("env-long", source, (user, assistant))
    index = UserLedEnvelopeShadowIndex(
        namespace_id="a" * 64,
        cache_receipt_sha256="b" * 64,
        envelopes=(envelope,),
        row_by_chunk_id=MappingProxyType(
            {user.chunk_id: user, assistant.chunk_id: assistant}
        ),
        envelope_by_chunk_id=MappingProxyType(
            {user.chunk_id: envelope.envelope_id, assistant.chunk_id: envelope.envelope_id}
        ),
    )
    selected = assay.select_spine_episodes(
        index,
        active_source_ids=(source,),
        dated_question="[Question asked at 2026-02-01] What desk did I buy?",
    )
    group = selected["selected_groups"][0]
    assert [row["chunk_id"] for row in group["rows"]] == [user.chunk_id]
    assert "lane_token_bound" in group["truncation_reasons"]


@pytest.mark.parametrize(
    ("anchor_position", "question", "expected_position", "direction"),
    (
        (1, "Which charity soccer event did I participate in?", 0, "previous"),
        (0, "What is my current Ford pickup truck model?", 1, "next"),
    ),
)
def test_physical_anchor_lane_surfaces_adjacent_user_episode(
    anchor_position: int,
    question: str,
    expected_position: int,
    direction: str,
) -> None:
    source = "p0::adjacent"
    if direction == "previous":
        texts = (
            "I participated in the annual charity soccer tournament today.",
            "I changed my exercise routine and did interval training today.",
        )
    else:
        texts = (
            "I finished airbrushing the old vehicle model today.",
            "I switched to a Ford F-150 pickup truck today.",
        )
    rows = tuple(
        _row(
            _sha(f"adjacent-{position}"),
            text,
            source_id=source,
            turn_id=f"turn-adjacent-{position}",
            role="user",
            ordinal=position + 1,
        )
        for position, text in enumerate(texts)
    )
    envelopes = tuple(
        _envelope(f"env-adjacent-{position}", source, (row,))
        for position, row in enumerate(rows)
    )
    index = UserLedEnvelopeShadowIndex(
        namespace_id="a" * 64,
        cache_receipt_sha256="b" * 64,
        envelopes=envelopes,
        row_by_chunk_id=MappingProxyType({row.chunk_id: row for row in rows}),
        envelope_by_chunk_id=MappingProxyType(
            {row.chunk_id: envelope.envelope_id for row, envelope in zip(rows, envelopes)}
        ),
    )
    selected = assay.select_spine_episodes(
        index,
        active_source_ids=(source,),
        dated_question=f"[Question asked at 2026-02-01] {question}",
        parent_packed_rows=(_exact_parent(rows[anchor_position]),),
    )
    winner = next(
        group
        for group in selected["selected_groups"]
        if group["envelope_id"] == envelopes[expected_position].envelope_id
    )
    assert winner["envelope_id"] == envelopes[expected_position].envelope_id
    assert winner["rank_audit"]["selected_lanes"][0] == "physical_anchor_transition"
    assert winner["physical_anchor_links"][0]["direction"] == direction


def test_physical_excerpt_anchor_selects_own_full_episode() -> None:
    source = "p0::same-opener"
    user = _row(
        _sha("coffee-mattress"),
        "I bought a coffee table and finally ordered a Casper mattress last week.",
        source_id=source,
        turn_id="turn-coffee-mattress",
        role="user",
        ordinal=1,
    )
    envelope = _envelope("env-coffee-mattress", source, (user,))
    index = UserLedEnvelopeShadowIndex(
        namespace_id="a" * 64,
        cache_receipt_sha256="b" * 64,
        envelopes=(envelope,),
        row_by_chunk_id=MappingProxyType({user.chunk_id: user}),
        envelope_by_chunk_id=MappingProxyType({user.chunk_id: envelope.envelope_id}),
    )
    selected = assay.select_spine_episodes(
        index,
        active_source_ids=(source,),
        dated_question="[Question asked at 2026-02-01] Which furniture did I buy?",
        parent_packed_rows=(_exact_parent(user, raw_text="I bought a coffee table"),),
    )
    winner = selected["selected_groups"][0]
    assert winner["rows"][0]["text"] == user.text
    assert winner["rank_audit"]["selected_lanes"][0] == "physical_anchor_owner"
    assert winner["physical_anchor_links"][0]["direction"] == "owner"


def test_physical_user_lane_excludes_assistant_followers_unless_required() -> None:
    source = "p0::role-gate"
    user = _row(
        _sha("role-user"),
        "I bought a cedar desk yesterday.",
        source_id=source,
        turn_id="turn-role-user",
        role="user",
        ordinal=1,
    )
    assistant = _row(
        _sha("role-assistant"),
        "The desk cost eight hundred dollars.",
        source_id=source,
        turn_id="turn-role-assistant",
        role="assistant",
        ordinal=2,
    )
    envelope = _envelope("env-role", source, (user, assistant))
    index = UserLedEnvelopeShadowIndex(
        namespace_id="a" * 64,
        cache_receipt_sha256="b" * 64,
        envelopes=(envelope,),
        row_by_chunk_id=MappingProxyType(
            {user.chunk_id: user, assistant.chunk_id: assistant}
        ),
        envelope_by_chunk_id=MappingProxyType(
            {
                user.chunk_id: envelope.envelope_id,
                assistant.chunk_id: envelope.envelope_id,
            }
        ),
    )
    parent = (_exact_parent(user, raw_text="I bought a cedar desk"),)
    question = "[Question asked at 2026-02-01] What desk did I buy?"
    user_only = assay.select_spine_episodes(
        index,
        active_source_ids=(source,),
        dated_question=question,
        parent_packed_rows=parent,
    )
    assert [row["role"] for row in user_only["selected_groups"][0]["rows"]] == [
        "user"
    ]
    assistant_spec = replace(
        assay.compile_typed_operator_spec(question),
        required_evidence_role="assistant",
        receipt_sha256="",
    )
    with_assistant = assay.select_spine_episodes(
        index,
        active_source_ids=(source,),
        dated_question=question,
        parent_packed_rows=parent,
        typed_spec=assistant_spec,
    )
    assert [row["role"] for row in with_assistant["selected_groups"][0]["rows"]] == [
        "user",
        "assistant",
    ]
    assert "physical_anchor_owner" in with_assistant["selected_groups"][0][
        "rank_audit"
    ]["selected_lanes"]


def test_assistant_required_backfills_when_ranked_follower_does_not_fit() -> None:
    source = "p0::assistant-backfill"
    large_user = _row(
        _sha("large-user"),
        "I bought the primary cedar desk yesterday.",
        source_id=source,
        turn_id="turn-large-user",
        role="user",
        ordinal=1,
    )
    large_assistant = _row(
        _sha("large-assistant"),
        ("The primary desk cost $900. " * 900).strip(),
        source_id=source,
        turn_id="turn-large-assistant",
        role="assistant",
        ordinal=2,
    )
    small_user = _row(
        _sha("small-user"),
        "I also bought the backup cedar desk yesterday.",
        source_id=source,
        turn_id="turn-small-user",
        role="user",
        ordinal=3,
    )
    small_assistant = _row(
        _sha("small-assistant"),
        "The backup desk cost $700.",
        source_id=source,
        turn_id="turn-small-assistant",
        role="assistant",
        ordinal=4,
    )
    rows = (large_user, large_assistant, small_user, small_assistant)
    envelopes = (
        _envelope("env-too-large", source, rows[:2]),
        _envelope("env-fits", source, rows[2:]),
    )
    index = UserLedEnvelopeShadowIndex(
        namespace_id="a" * 64,
        cache_receipt_sha256="b" * 64,
        envelopes=envelopes,
        row_by_chunk_id=MappingProxyType({row.chunk_id: row for row in rows}),
        envelope_by_chunk_id=MappingProxyType(
            {
                row.chunk_id: envelope.envelope_id
                for envelope in envelopes
                for row in rows
                if row.chunk_id in envelope.chunk_ids
            }
        ),
    )
    question = "[Question asked at 2026-02-01] What did the cedar desk cost?"
    assistant_spec = replace(
        assay.compile_typed_operator_spec(question),
        required_evidence_role="assistant",
        receipt_sha256="",
    )
    selected = assay.select_spine_episodes(
        index,
        active_source_ids=(source,),
        dated_question=question,
        parent_packed_rows=(_exact_parent(large_user), _exact_parent(small_user)),
        typed_spec=assistant_spec,
    )
    assert any(
        decision["envelope_id"] == "env-too-large"
        and decision["decision"] == "rejected_required_assistant_not_hydrated"
        for decision in selected["lane_decisions"]
    )
    assert any(group["envelope_id"] == "env-fits" for group in selected["selected_groups"])
    assert all(
        any(row["role"] == "assistant" for row in group["rows"])
        for group in selected["selected_groups"]
    )


def test_hidden_delta_owner_outranks_duplicate_only_excerpt() -> None:
    rows = (
        _row(
            _sha("duplicate-only"),
            "I got a coffee table.",
            source_id="p0::duplicate",
            turn_id="turn-duplicate",
            role="user",
            ordinal=1,
        ),
        _row(
            _sha("hidden-action"),
            "I got a coffee table. I bought a mattress yesterday.",
            source_id="p0::hidden",
            turn_id="turn-hidden",
            role="user",
            ordinal=2,
        ),
    )
    envelopes = (
        _envelope("env-duplicate", rows[0].source_id, (rows[0],)),
        _envelope("env-hidden", rows[1].source_id, (rows[1],)),
    )
    index = UserLedEnvelopeShadowIndex(
        namespace_id="a" * 64,
        cache_receipt_sha256="b" * 64,
        envelopes=envelopes,
        row_by_chunk_id=MappingProxyType({row.chunk_id: row for row in rows}),
        envelope_by_chunk_id=MappingProxyType(
            {row.chunk_id: envelope.envelope_id for row, envelope in zip(rows, envelopes)}
        ),
    )
    selected = assay.select_spine_episodes(
        index,
        active_source_ids=(rows[0].source_id, rows[1].source_id),
        dated_question=(
            "[Question asked at 2026-02-01] Which furniture did I buy, assemble, sell, or fix?"
        ),
        parent_packed_rows=(
            _exact_parent(rows[0], raw_text="I got a coffee table"),
            _exact_parent(rows[1], raw_text="I got a coffee table."),
        ),
    )
    owner_ranking = selected["physical_anchor_rankings"]["physical_anchor_owner"]
    assert owner_ranking[0]["envelope_id"] == "env-hidden"
    assert owner_ranking[0]["conditioned_score"][1] > 0


def test_atomic_episode_manifest_uses_ref_or_receipted_projection_collision() -> None:
    row = {
        "chunk_id": _sha("episode-row"),
        "created_at": "2026-01-01",
        "role": "user",
        "source_id": "p0::session-a",
        "text": "exact episode bytes",
        "text_sha256": quote_sha256("exact episode bytes"),
    }
    assistant = {
        "chunk_id": _sha("episode-assistant"),
        "created_at": "2026-01-01",
        "role": "assistant",
        "source_id": "p0::session-a",
        "text": "assistant continuation",
        "text_sha256": quote_sha256("assistant continuation"),
    }
    group = {
        "envelope_id": "env-exact",
        "opener_user_chunk_id": row["chunk_id"],
        "rows": [row, assistant],
        "source_id": row["source_id"],
    }
    global_row = {
        "evidence_id": row["chunk_id"],
        "created_at": row["created_at"],
        "role": row["role"],
        "source_id": row["source_id"],
        "raw_text": row["text"],
        "raw_text_sha256": row["text_sha256"],
    }
    manifest, error = assay._episode_manifest(group, {row["chunk_id"]: global_row})
    assert error is None
    assert [value["chunk_id"] for value in manifest["raw_rows"]] == [
        assistant["chunk_id"]
    ]
    assert manifest["global_refs"][0]["episode_chunk_id"] == row["chunk_id"]
    rendered = assay._render_context([global_row], [manifest], [])
    assert rendered.index("<GREF episode_evidence=") < rendered.index(
        f"<A evidence={assistant['chunk_id']}"
    )

    global_row["raw_text"] = "exact episode"
    global_row["raw_text_sha256"] = quote_sha256(global_row["raw_text"])
    manifest, error = assay._episode_manifest(
        group,
        {row["chunk_id"]: global_row},
        global_citations={row["chunk_id"]: "G1"},
    )
    assert error is None
    assert manifest["manifest_rows"][0]["representation"] == "episode_collision_raw"
    collision = manifest["representation_collisions"][0]
    assert collision["mismatch_fields"] == ["text", "text_sha256"]
    assert collision["projection_relationship"] == (
        "whitespace_normalized_contiguous_projection"
    )
    citation = assay._global_citation_manifest([global_row])["entries"][0]
    assert assay._collision_binding_valid(collision, row, citation)
    rendered = assay._render_context([global_row], [manifest], [])
    assert "projection_of=G1 collision=C1" in rendered
    assert "exact episode bytes" in rendered

    assert assay._collision_partition_valid(manifest)
    manifest["representation_collisions"] = [
        {**collision, "episode_evidence_id": assistant["chunk_id"]}
    ]
    assert not assay._collision_partition_valid(manifest)

    global_row["source_id"] = "p0::different-session"
    assert assay._episode_manifest(
        group,
        {row["chunk_id"]: global_row},
        global_citations={row["chunk_id"]: "G1"},
    )[1].startswith(
        "episode_global_exact_id_mismatch:"
    )


def test_compact_global_labels_preserve_parent_text_order_and_bind_full_ids() -> None:
    rows = [
        {
            "created_at": "2026-01-01",
            "evidence_id": _sha(f"global-{number}"),
            "raw_text": f"exact parent text {number}",
            "role": "user",
            "source_id": f"p0::source-{number}",
        }
        for number in (1, 2, 3)
    ]
    context = assay._render_context(rows, [], [])
    assert [context.index(row["raw_text"]) for row in rows] == sorted(
        context.index(row["raw_text"]) for row in rows
    )
    assert [line for line in context.splitlines() if line.startswith("<G")] == [
        "<G1>",
        "<G2>",
        "<G3>",
    ]
    manifest = assay._global_citation_manifest(rows)
    assert [entry["citation"] for entry in manifest["entries"]] == ["G1", "G2", "G3"]
    assert [entry["evidence_id"] for entry in manifest["entries"]] == [
        row["evidence_id"] for row in rows
    ]


def test_empty_fact_selection_receipts_mandatory_coverage_failure() -> None:
    selected = assay.select_spine_episodes(
        _index(),
        active_source_ids=("p0::session-a",),
        dated_question="[Question asked at 2026-02-01] How much was the table?",
    )
    from tools.matched_eval.hot_v6_query_fact_ledger import compile_query_fact_ledger

    ledger = compile_query_fact_ledger(
        "[Question asked at 2026-02-01] How much was the table?",
        assay._episode_fact_rows(selected["selected_groups"]),
    )
    empty = assay._empty_fact_selection(
        ledger, status="mandatory_coverage_did_not_fit"
    )
    assert empty["selected_facts"] == []
    assert empty["selection_status"] == "mandatory_coverage_did_not_fit"
    unsigned = dict(empty)
    assert unsigned.pop("receipt_sha256") == identity_sha256(unsigned)


def test_oversized_first_user_lead_is_rejected_and_receipted() -> None:
    source = "p0::oversized-lead"
    user = _row(
        _sha("oversized-user-lead"),
        ("table " * (assay.EPISODE_LANE_TOKEN_BUDGET + 100)).strip(),
        source_id=source,
        turn_id="turn-oversized",
        role="user",
        ordinal=1,
    )
    envelope = _envelope("env-oversized", source, (user,))
    index = UserLedEnvelopeShadowIndex(
        namespace_id="a" * 64,
        cache_receipt_sha256="b" * 64,
        envelopes=(envelope,),
        row_by_chunk_id=MappingProxyType({user.chunk_id: user}),
        envelope_by_chunk_id=MappingProxyType({user.chunk_id: envelope.envelope_id}),
    )
    selected = assay.select_spine_episodes(
        index,
        active_source_ids=(source,),
        dated_question="[Question asked at 2026-02-01] Which table?",
    )
    assert selected["selected_groups"] == []
    assert any(
        decision["decision"] == "rejected_lane_lead_budget"
        for decision in selected["lane_decisions"]
    )


def test_v7_is_additive_and_keeps_sealed_v6_bytes_unchanged() -> None:
    assert successor.FORMAT != assay.FORMAT
    assert successor.FORMAT.endswith("selection-v6")
    assert successor.DEFAULT_OUTPUT_ROOT != assay.DEFAULT_OUTPUT_ROOT
    assert successor.DEFAULT_OUTPUT_ROOT.name.endswith("20260908-r9")
    assert (
        hashlib.sha256(Path(assay.__file__).read_bytes()).hexdigest()
        == "24dab00ba6e7c81bfa27e443dc6f6f28eaa4aaecc8372207857d71d89926b823"
    )


def test_v7_specialist_selection_is_reserved_and_transition_cap_is_typed() -> None:
    index = _index()
    direct = successor.select_spine_episodes(
        index,
        active_source_ids=("p0::session-a",),
        dated_question="[Question asked at 2026-02-01] What table did I buy?",
        parent_packed_rows=(_exact_parent(index.row_by_chunk_id[_sha("u-table")]),),
    )
    temporal = successor.select_spine_episodes(
        index,
        active_source_ids=("p0::session-c",),
        dated_question=(
            "[Question asked at 2026-02-01] Which happened first, participating "
            "in the fundraiser or watching the movie?"
        ),
        parent_packed_rows=(_exact_parent(index.row_by_chunk_id[_sha("u-movie")]),),
    )
    assert direct["lane_budgets"]["union_policy"] == (
        "specialist_reserved_then_physical_remainder"
    )
    assert set(direct["selection_pass_receipts"]) == {
        "physical_remainder",
        "specialist_reserved",
    }
    assert direct["lane_budgets"]["physical_transition_episode_cap"] == 2
    assert temporal["lane_budgets"]["physical_transition_episode_cap"] == 4
    assert any(
        decision["selection_pass"] == "specialist_reserved"
        and decision["decision"].startswith("selected")
        for decision in direct["lane_decisions"]
    )


def test_v7_compact_provider_labels_keep_full_provenance_out_of_context() -> None:
    row = {
        "chunk_id": _sha("v7-episode-user"),
        "created_at": "2026-01-01",
        "role": "user",
        "source_id": "p0::v7-session",
        "text": "I bought a cedar desk.",
        "text_sha256": quote_sha256("I bought a cedar desk."),
    }
    assistant = {
        "chunk_id": _sha("v7-episode-assistant"),
        "created_at": "2026-01-01",
        "role": "assistant",
        "source_id": "p0::v7-session",
        "text": "The desk cost $700.",
        "text_sha256": quote_sha256("The desk cost $700."),
    }
    group = {
        "envelope_id": "env-v7",
        "opener_user_chunk_id": row["chunk_id"],
        "rows": [row, assistant],
        "source_id": row["source_id"],
    }
    global_row = {
        "evidence_id": row["chunk_id"],
        "created_at": row["created_at"],
        "role": row["role"],
        "source_id": row["source_id"],
        "raw_text": row["text"],
        "raw_text_sha256": row["text_sha256"],
    }
    manifest, error = assay._episode_manifest(group, {row["chunk_id"]: global_row})
    assert error is None
    context = successor._render_context([global_row], [manifest], [])
    assert "<G1>" in context
    assert "<E1>" in context
    assert "<REF G1>" in context
    assert "<A2 owner=G1>" in context
    assert row["chunk_id"] not in context
    assert assistant["chunk_id"] not in context
    assert row["source_id"] not in context
    assert manifest["manifest_rows"][0]["chunk_id"] == row["chunk_id"]


def test_v7_composer_reserves_fact_budget_and_seals_compact_label_mapping() -> None:
    index = _index()
    user = index.row_by_chunk_id[_sha("u-table")]
    parent = _exact_parent(user)
    question = "[Question asked at 2026-02-01] How much was the dining table?"
    old_context = assay._render_context([parent], [], [])
    old_messages, old_context_tokens, _workspace = assay._prompt(question, old_context)
    source_arm = {
        "context_token_proxy": old_context_tokens,
        "packed_chunk_ids": [user.chunk_id],
        "packed_evidence": [parent],
        "provider_messages": old_messages,
    }
    arm = successor._compose_arm(
        source_arm,
        dated_question=question,
        index=index,
        candidate_binding={"namespace_id": index.namespace_id},
    )
    assert arm["mode"] == "spine_indexed_episodic_fact_ledger"
    assert arm["fact_budget_reservation"][
        "reserved_before_optional_episode_expansion"
    ] is True
    assert arm["fact_budget_reservation"]["fact_provider_token_count"] <= (
        successor.MAX_FACT_TOKENS
    )
    assert arm["fact_ledger"]["selected_fact_count"] > 0
    assert arm["fact_input_selection"]["global_input_count"] == 1
    assert arm["fact_input_selection"]["episode_input_count"] > 0
    user_prompt = arm["provider_messages"][1]["content"]
    assert "<FACTS exact_quotes raw_authoritative>" in user_prompt
    typed_audit = arm["fact_advisory"]["typed_reducer_audit"]
    assert typed_audit["reduction"]["status"] == "not_applicable"
    provider_advisory = typed_audit["provider_advisory"]
    assert provider_advisory["emitted"] is False
    assert provider_advisory["text"] == ""
    assert arm["fact_advisory"]["provider_token_count"] == count_tokens(
        provider_advisory["text"]
    )
    assert user.chunk_id not in user_prompt
    assert user.source_id not in user_prompt
    assert user.chunk_id not in provider_advisory["text"]
    assert all(
        fact_id not in provider_advisory["text"]
        for fact_id in arm["rendered_fact_ids"]
    )
    audit = json.dumps(arm["provider_provenance_manifest"], sort_keys=True)
    assert user.chunk_id in audit
    assert user.source_id in audit
    unsigned = dict(arm["provider_provenance_manifest"])
    receipt = unsigned.pop("receipt_sha256")
    assert receipt == identity_sha256(unsigned)


def test_v7_provider_admission_grows_audit_and_backs_off_fact_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_id = "p0::audit-decoupling"
    text = " ".join(
        f"I recorded marker {name} in my notebook."
        for name in (
            "amber",
            "bronze",
            "cobalt",
            "denim",
            "emerald",
            "fuchsia",
            "gold",
            "hazel",
            "indigo",
            "jade",
            "khaki",
            "lilac",
        )
    )
    row = _row(
        _sha("audit-decoupling-row"),
        text,
        source_id=source_id,
        turn_id="turn-audit-decoupling",
        role="user",
        ordinal=1,
    )
    envelope = _envelope("env-audit-decoupling", source_id, (row,))
    index = UserLedEnvelopeShadowIndex(
        namespace_id="a" * 64,
        cache_receipt_sha256="b" * 64,
        envelopes=(envelope,),
        row_by_chunk_id=MappingProxyType({row.chunk_id: row}),
        envelope_by_chunk_id=MappingProxyType(
            {row.chunk_id: envelope.envelope_id}
        ),
    )
    parent = _exact_parent(row)
    question = "[Question asked at 2026-02-01] What marker did I record?"
    old_context = assay._render_context([parent], [], [])
    old_messages, old_context_tokens, _workspace = assay._prompt(
        question, old_context
    )
    source_arm = {
        "context_token_proxy": old_context_tokens,
        "packed_chunk_ids": [row.chunk_id],
        "packed_evidence": [parent],
        "provider_messages": old_messages,
    }
    baseline = successor._compose_arm(
        source_arm,
        dated_question=question,
        index=index,
        candidate_binding={"namespace_id": index.namespace_id},
    )
    baseline_reservation = baseline["fact_budget_reservation"]
    baseline_attempt = baseline_reservation["provider_admission_attempts"][0]
    assert baseline_attempt["decision"] == "admitted"
    assert baseline_attempt["selected_fact_count"] > baseline_reservation[
        "mandatory_fact_count"
    ]

    monkeypatch.setattr(successor, "INITIAL_FACT_SELECTION_AUDIT_TOKENS", 1)
    monkeypatch.setattr(
        successor,
        "MAX_FACT_TOKENS",
        baseline_attempt["provider_overlay_token_count"] - 1,
    )
    arm = successor._compose_arm(
        source_arm,
        dated_question=question,
        index=index,
        candidate_binding={"namespace_id": index.namespace_id},
    )
    reservation = arm["fact_budget_reservation"]
    attempts = reservation["provider_admission_attempts"]
    assert reservation["selection_status"] == "selected_with_mandatory_coverage"
    assert reservation["internal_audit_encoding_controls_provider_admission"] is False
    assert attempts[0]["decision"] == "provider_fact_count_backoff"
    assert attempts[-1]["decision"] == "admitted"
    assert [attempt["requested_fact_count"] for attempt in attempts] == list(
        range(
            attempts[0]["requested_fact_count"],
            attempts[-1]["requested_fact_count"] - 1,
            -1,
        )
    )
    assert attempts[0]["audit_token_budget"] > 1
    assert all(
        later["audit_token_budget"] >= earlier["audit_token_budget"]
        for earlier, later in zip(attempts, attempts[1:], strict=False)
    )
    assert reservation["provider_selected_fact_count"] >= reservation[
        "mandatory_fact_count"
    ]
    assert reservation["provider_selected_fact_count"] < reservation[
        "initial_provider_fact_count"
    ]
    assert reservation["fact_provider_token_count"] <= successor.MAX_FACT_TOKENS

    monkeypatch.setattr(successor, "MAX_FACT_TOKENS", 1)
    minimum_failure = successor._compose_arm(
        source_arm,
        dated_question=question,
        index=index,
        candidate_binding={"namespace_id": index.namespace_id},
    )
    failed_reservation = minimum_failure["fact_budget_reservation"]
    failed_attempts = failed_reservation["provider_admission_attempts"]
    assert failed_reservation["selection_status"] == (
        "mandatory_provider_coverage_did_not_fit"
    )
    assert failed_attempts[-1]["requested_fact_count"] == failed_reservation[
        "mandatory_fact_count"
    ]
    assert failed_attempts[-1]["selected_fact_count"] == failed_reservation[
        "mandatory_fact_count"
    ]
    assert failed_attempts[-1]["decision"] == (
        "mandatory_provider_coverage_did_not_fit"
    )
    assert failed_reservation["provider_selected_fact_count"] == 0
    assert minimum_failure["rendered_fact_ids"] == []


def test_v7_provider_overlay_uses_the_exact_adopted_manifest_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _index()
    question = "[Question asked at 2026-02-01] How much was the dining table?"
    original_accounting = successor._provider_overlay_accounting

    def manifest_sensitive_accounting(*args: object, **kwargs: object) -> dict:
        accounting = original_accounting(*args, **kwargs)
        episode_manifests = args[1]
        facts = args[2]
        if episode_manifests and facts:
            accounting = {
                **accounting,
                "fact_advisory_token_count": successor.MAX_FACT_TOKENS + 1,
                "total_token_count": successor.MAX_FACT_TOKENS + 1,
            }
        return accounting

    def compose(parent: dict[str, object]) -> dict:
        old_context = assay._render_context([parent], [], [])
        old_messages, old_context_tokens, _workspace = assay._prompt(
            question, old_context
        )
        return successor._compose_arm(
            {
                "context_token_proxy": old_context_tokens,
                "packed_chunk_ids": [parent["evidence_id"]],
                "packed_evidence": [parent],
                "provider_messages": old_messages,
            },
            dated_question=question,
            index=index,
            candidate_binding={"namespace_id": index.namespace_id},
        )

    monkeypatch.setattr(successor, "MAX_FACTS", 1)
    monkeypatch.setattr(
        successor, "_provider_overlay_accounting", manifest_sensitive_accounting
    )

    # A candidate-only fact needs an adopted backing manifest.  The exact-set
    # count, rather than a filtered provisional view, must decide admission.
    candidate_only = compose(
        _exact_parent(index.row_by_chunk_id[_sha("u-book")])
    )
    attempt = candidate_only["fact_budget_reservation"][
        "provider_admission_attempts"
    ][-1]
    assert attempt["decision"] == "mandatory_provider_coverage_did_not_fit"
    assert candidate_only["rendered_fact_ids"] == []

    # A globally backed fact needs no manifest.  Optional episode expansion
    # must still rerender/recount the exact trial set and reject an overflow.
    global_backed = compose(
        _exact_parent(index.row_by_chunk_id[_sha("u-table")])
    )
    reservation = global_backed["fact_budget_reservation"]
    assert reservation["selection_status"] == "selected_with_mandatory_coverage"
    assert reservation["admitted_optional_episode_count"] == 0
    assert reservation["provider_total_overlay_token_count"] <= (
        successor.MAX_FACT_TOKENS
    )
    assert global_backed["episode_manifest_rejections"]
    assert {
        row["reason"] for row in global_backed["episode_manifest_rejections"]
    } == {"hard_provider_overlay_cap_after_fact_reserve"}


def test_v7_mandatory_audit_can_exceed_initial_floor_without_provider_overflow() -> None:
    source_id = "p0::" + "-".join(f"opaque{number}" for number in range(1_000))
    row = _row(
        _sha("large-private-audit-row"),
        "I bought a cedar desk yesterday.",
        source_id=source_id,
        turn_id="turn-large-private-audit",
        role="user",
        ordinal=1,
    )
    envelope = _envelope("env-large-private-audit", source_id, (row,))
    index = UserLedEnvelopeShadowIndex(
        namespace_id="a" * 64,
        cache_receipt_sha256="b" * 64,
        envelopes=(envelope,),
        row_by_chunk_id=MappingProxyType({row.chunk_id: row}),
        envelope_by_chunk_id=MappingProxyType({row.chunk_id: envelope.envelope_id}),
    )
    parent = _exact_parent(row)
    question = "[Question asked at 2026-02-01] What desk did I buy?"
    old_context = assay._render_context([parent], [], [])
    old_messages, old_context_tokens, _workspace = assay._prompt(
        question, old_context
    )
    arm = successor._compose_arm(
        {
            "context_token_proxy": old_context_tokens,
            "packed_chunk_ids": [row.chunk_id],
            "packed_evidence": [parent],
            "provider_messages": old_messages,
        },
        dated_question=question,
        index=index,
        candidate_binding={"namespace_id": index.namespace_id},
    )
    reservation = arm["fact_budget_reservation"]
    attempt = reservation["provider_admission_attempts"][-1]
    assert reservation["selection_status"] == "selected_with_mandatory_coverage"
    assert attempt["decision"] == "admitted"
    assert attempt["audit_rendered_token_count"] > 2_048
    assert attempt["audit_token_budget"] > 2_048
    assert attempt["provider_overlay_token_count"] <= successor.MAX_FACT_TOKENS
    assert reservation["provider_selected_fact_count"] >= reservation[
        "mandatory_fact_count"
    ]
    assert source_id not in arm["provider_messages"][1]["content"]


def test_v7_fact_hydration_uses_lead_and_backing_not_whole_chunked_opener() -> None:
    source_id = "p0::chunked-opener"
    turn_id = "turn-chunked-opener"
    lead = _row(
        _sha("chunked-opener-lead"),
        "I started a long note.",
        source_id=source_id,
        turn_id=turn_id,
        role="user",
        ordinal=1,
    )
    filler = _row(
        _sha("chunked-opener-filler"),
        "This middle chunk carries no requested fact.",
        source_id=source_id,
        turn_id=turn_id,
        role="user",
        ordinal=2,
    )
    backing = _row(
        _sha("chunked-opener-backing"),
        "The exact access marker is cobalt.",
        source_id=source_id,
        turn_id=turn_id,
        role="user",
        ordinal=3,
    )
    envelope = _envelope("env-chunked-opener", source_id, (lead, filler, backing))
    index = UserLedEnvelopeShadowIndex(
        namespace_id="a" * 64,
        cache_receipt_sha256="b" * 64,
        envelopes=(envelope,),
        row_by_chunk_id=MappingProxyType(
            {row.chunk_id: row for row in (lead, filler, backing)}
        ),
        envelope_by_chunk_id=MappingProxyType(
            {row.chunk_id: envelope.envelope_id for row in (lead, filler, backing)}
        ),
    )
    groups, hydration = successor._hydrate_fact_backing_groups(
        (),
        [
            {
                "backing_evidence_id": backing.chunk_id,
                "envelope_id": envelope.envelope_id,
                "exact_quote": backing.text,
                "fact_id": _sha("chunked-opener-fact"),
            }
        ],
        index=index,
        global_rows=(),
    )
    assert hydration["unresolved_backing_evidence_ids"] == []
    assert len(groups) == 1
    assert [row["chunk_id"] for row in groups[0]["rows"]] == [
        lead.chunk_id,
        backing.chunk_id,
    ]
    assert filler.chunk_id not in {
        row["chunk_id"] for row in groups[0]["rows"]
    }


def test_v7_unresolved_comparison_slot_keeps_bound_candidate_fact() -> None:
    source_id = "p0::comparison-partial"
    hawaii = _row(
        _sha("comparison-hawaii"),
        "The Hawaii room had an ocean view.",
        source_id=source_id,
        turn_id="turn-comparison-hawaii",
        role="user",
        ordinal=1,
    )
    tokyo = _row(
        _sha("comparison-tokyo"),
        "The Tokyo hotel cost $50 per night.",
        source_id=source_id,
        turn_id="turn-comparison-tokyo",
        role="user",
        ordinal=2,
    )
    question = (
        "[Question asked at 2026-09-07] How much higher were accommodations "
        "in Hawaii compared to Tokyo?"
    )
    ledger = successor.compile_query_fact_ledger(
        question,
        [_exact_parent(hawaii)],
        candidate_rows=[_exact_parent(tokyo)],
    )
    unresolved_labels = {
        slot.label
        for slot in ledger.operator_spec.required_slots
        if slot.slot_id in ledger.unresolved_slot_ids
    }
    assert unresolved_labels == {"Hawaii"}
    assert {binding.slot_label for binding in ledger.slot_bindings} == {"Tokyo"}
    fact_slice, _audit_tokens = successor._select_audit_fact_slice(
        ledger,
        max_facts=successor._mandatory_fact_count(ledger),
    )
    tokyo_fact_ids = {
        binding.fact_id
        for binding in ledger.slot_bindings
        if binding.slot_label == "Tokyo"
    }
    assert tokyo_fact_ids <= set(fact_slice.fact_ids)
    assert set(fact_slice.mandatory_fact_ids) <= set(fact_slice.fact_ids)
    assert "Hawaii:unresolved" in fact_slice.rendered_text
    assert "Tokyo:bound" in fact_slice.rendered_text
    assert any(
        fact.backing_evidence_id == tokyo.chunk_id
        and fact.exact_quote == "The Tokyo hotel cost $50 per night."
        for fact in fact_slice.facts
    )
    provider = successor._render_fact_section(
        [fact.projection() for fact in fact_slice.facts],
        {hawaii.chunk_id: "E1.U1", tokyo.chunk_id: "E2.U1"},
    )
    assert "backs=E2.U1" in provider
    assert "$50 per night" in provider
    assert count_tokens(provider) <= successor.MAX_FACT_TOKENS


def test_v7_unresolved_numeric_completion_is_source_local_raw_and_unbound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_id = "p0::travel-comparison"
    hawaii = _row(
        _sha("completion-hawaii-anchor"),
        "I found a restaurant in Hawaii with an ocean view.",
        source_id=source_id,
        turn_id="turn-completion-hawaii",
        role="user",
        ordinal=1,
    )
    maui = _row(
        _sha("completion-maui-rate"),
        "I booked accommodations in Maui for $310 per night.",
        source_id=source_id,
        turn_id="turn-completion-maui",
        role="user",
        ordinal=2,
    )
    tokyo = _row(
        _sha("completion-tokyo-rate"),
        "The Tokyo hotel cost $50 per night.",
        source_id=source_id,
        turn_id="turn-completion-tokyo",
        role="user",
        ordinal=3,
    )
    restaurant = _row(
        _sha("completion-restaurant-price"),
        "I spent $90 at a restaurant.",
        source_id=source_id,
        turn_id="turn-completion-restaurant",
        role="user",
        ordinal=4,
    )
    envelopes = tuple(
        _envelope(f"env-{position}", source_id, (row,))
        for position, row in enumerate((hawaii, maui, tokyo, restaurant), 1)
    )
    rows = (hawaii, maui, tokyo, restaurant)
    index = UserLedEnvelopeShadowIndex(
        namespace_id="a" * 64,
        cache_receipt_sha256="b" * 64,
        envelopes=envelopes,
        row_by_chunk_id=MappingProxyType({row.chunk_id: row for row in rows}),
        envelope_by_chunk_id=MappingProxyType(
            {
                row.chunk_id: envelope.envelope_id
                for row, envelope in zip(rows, envelopes, strict=True)
            }
        ),
    )
    parent = _exact_parent(hawaii)
    question = (
        "[Question asked at 2026-09-07] How much more did I spend on "
        "accommodations per night in Hawaii compared to Tokyo?"
    )
    old_context = assay._render_context([parent], [], [])
    old_messages, old_context_tokens, _workspace = assay._prompt(
        question, old_context
    )
    arm = successor._compose_arm(
        {
            "context_token_proxy": old_context_tokens,
            "packed_chunk_ids": [hawaii.chunk_id],
            "packed_evidence": [parent],
            "provider_messages": old_messages,
        },
        dated_question=question,
        index=index,
        candidate_binding={"namespace_id": index.namespace_id},
    )
    completion = arm["numeric_slot_completion"]
    selected = [
        decision
        for decision in completion["decisions"]
        if decision["decision"] == "selected_source_local_numeric_raw"
    ]
    assert len(selected) == 1
    assert selected[0]["anchor_evidence_id"] == hawaii.chunk_id
    assert selected[0]["selected_operand_evidence_ids"] == [maui.chunk_id]
    assert completion["frontier_closed"] is False
    assert completion["slot_bindings_added"] == 0
    assert completion["unresolved_slot_ids_before"] == completion[
        "unresolved_slot_ids_after"
    ]
    assert completion["admission"]["decision"] == (
        "admitted_before_optional_episodes"
    )
    assert maui.chunk_id in arm["rendered_raw_chunk_ids"]
    assert tokyo.chunk_id not in completion["operand_evidence_ids"]
    assert restaurant.chunk_id not in completion["operand_evidence_ids"]
    assert "<SOURCE_LOCAL_CANDIDATES unbound>" in completion["provider_text"]
    assert "relation=night" in completion["provider_text"]
    assert maui.chunk_id not in arm["provider_messages"][1]["content"]
    assert arm["fact_budget_reservation"][
        "provider_total_overlay_token_count"
    ] <= successor.MAX_FACT_TOKENS

    baseline_reservation = arm["fact_budget_reservation"]
    assert baseline_reservation["provider_selected_fact_count"] > (
        baseline_reservation["mandatory_fact_count"]
    )
    monkeypatch.setattr(
        successor,
        "MAX_FACT_TOKENS",
        completion["admission"]["provider_overlay_token_count"] - 1,
    )
    backed_off = successor._compose_arm(
        {
            "context_token_proxy": old_context_tokens,
            "packed_chunk_ids": [hawaii.chunk_id],
            "packed_evidence": [parent],
            "provider_messages": old_messages,
        },
        dated_question=question,
        index=index,
        candidate_binding={"namespace_id": index.namespace_id},
    )
    backed_completion = backed_off["numeric_slot_completion"]
    backed_reservation = backed_off["fact_budget_reservation"]
    assert backed_completion["admission"]["decision"] == (
        "admitted_after_optional_fact_backoff"
    )
    assert backed_reservation["provider_selected_fact_count"] < (
        baseline_reservation["provider_selected_fact_count"]
    )
    assert backed_reservation["provider_selected_fact_count"] >= (
        backed_reservation["mandatory_fact_count"]
    )
    assert backed_reservation["provider_admission_attempts"][-1]["decision"] == (
        "admitted"
    )
    assert backed_reservation["provider_total_overlay_token_count"] <= (
        successor.MAX_FACT_TOKENS
    )
    assert maui.chunk_id in backed_off["rendered_raw_chunk_ids"]
    successor._validate_successor_extensions(
        {"questions": [{"arms": {"a3_protected_union": backed_off}}]}
    )


def test_v7_composer_injects_only_a_supported_fact_cited_advisory() -> None:
    source_id = "p0::session-camera"
    camera = _row(
        _sha("u-camera-duration"),
        "I've been collecting vintage cameras for three months now.",
        source_id=source_id,
        turn_id="turn-u-camera-duration",
        role="user",
        ordinal=1,
    )
    envelope = _envelope("env-camera", source_id, (camera,))
    index = UserLedEnvelopeShadowIndex(
        namespace_id="a" * 64,
        cache_receipt_sha256="b" * 64,
        envelopes=(envelope,),
        row_by_chunk_id=MappingProxyType({camera.chunk_id: camera}),
        envelope_by_chunk_id=MappingProxyType(
            {camera.chunk_id: envelope.envelope_id}
        ),
    )
    parent = _exact_parent(camera)
    question = (
        "[Question asked at 2026/02/01] How long have I been collecting "
        "vintage cameras?"
    )
    old_context = assay._render_context([parent], [], [])
    old_messages, old_context_tokens, _workspace = assay._prompt(
        question, old_context
    )
    arm = successor._compose_arm(
        {
            "context_token_proxy": old_context_tokens,
            "packed_chunk_ids": [camera.chunk_id],
            "packed_evidence": [parent],
            "provider_messages": old_messages,
        },
        dated_question=question,
        index=index,
        candidate_binding={"namespace_id": index.namespace_id},
    )

    typed_audit = arm["fact_advisory"]["typed_reducer_audit"]
    assert typed_audit["reduction"]["status"] == "supported"
    provider_advisory = typed_audit["provider_advisory"]
    assert provider_advisory["emitted"] is True
    assert "prediction=" in provider_advisory["text"]
    assert "support=F" in provider_advisory["text"]
    assert provider_advisory["text"] in arm["provider_messages"][1]["content"]
    assert arm["fact_advisory"]["provider_token_count"] == count_tokens(
        provider_advisory["text"]
    )
    for raw_id in (
        *arm["packed_chunk_ids"],
        *arm["rendered_fact_ids"],
    ):
        assert raw_id not in provider_advisory["text"]


def test_v7_candidate_fact_can_hydrate_a_row_missed_by_episode_selection() -> None:
    index = _index()
    book = index.row_by_chunk_id[_sha("u-book")]
    global_rows = [_exact_parent(book)]
    selected_rows, candidate_rows, audit = successor._fact_input_rows(
        global_rows,
        [],
        index=index,
        active_source_ids=("p0::session-a",),
    )
    question = "[Question asked at 2026-02-01] How much was the dining table?"
    from tools.matched_eval.hot_v6_query_fact_ledger import (
        compile_query_fact_ledger,
        select_and_render_query_fact_ledger,
    )

    ledger = compile_query_fact_ledger(
        question,
        selected_rows,
        candidate_rows=candidate_rows,
    )
    fact_slice = select_and_render_query_fact_ledger(
        ledger,
        max_facts=successor.MAX_FACTS,
        max_tokens=successor.MAX_FACT_TOKENS,
    )
    facts = [fact.projection() for fact in fact_slice.facts]
    assert any("$800" in fact["exact_quote"] for fact in facts)
    groups, hydration = successor._hydrate_fact_backing_groups(
        [],
        facts,
        index=index,
        global_rows=global_rows,
    )
    assert audit["absence_authority"] == (
        "none_active_source_scan_does_not_close_frontier"
    )
    assert any(group["envelope_id"] == "env-table" for group in groups)
    assert any(
        binding["backing_evidence_id"] == _sha("u-table")
        for binding in hydration["hydrated_bindings"]
    )


def test_authenticated_loader_requires_semantic_replay_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = "[Question asked at 2026-02-01] What table did I buy?"
    messages, context_tokens, workspace_tokens = assay._prompt(question, "")
    payload = hot._canonical_json_bytes({"messages": messages})
    cache_body = {
        "cache_immutable": True,
        "content_row_count": 1,
        "database_read_passes": 1,
        "format": "cache-test",
        "metadata_row_count": 0,
        "namespace_id": "a" * 64,
        "partitions": [],
        "physical_store_row_count": 1,
        "source_database_sha256": "b" * 64,
        "source_store_receipt_sha256": "c" * 64,
    }
    cache = {**cache_body, "cache_receipt_sha256": identity_sha256(cache_body)}
    index_body = {
        "cache_receipt_sha256": cache["cache_receipt_sha256"],
        "chunk_count": 1,
        "envelope_count": 1,
        "envelopes_sha256": "d" * 64,
        "format": "index-test",
        "gold_loaded": False,
        "mechanism_id": "test",
        "namespace_id": "a" * 64,
        "provider_calls": 0,
    }
    index = {**index_body, "receipt_sha256": identity_sha256(index_body)}
    binding = {
        "cache": cache,
        "cache_build_binding": {},
        "index": index,
        "namespace_id": "a" * 64,
        "store": {
            "database_sha256": "b" * 64,
            "physical_store_row_count": 1,
            "store_receipt_sha256": "c" * 64,
        },
    }
    fact_body = {
        "selected_facts": [],
        "selection_status": "mandatory_coverage_did_not_fit",
    }
    fact_body = {**fact_body, "receipt_sha256": identity_sha256(fact_body)}
    arm = {
        "candidate_universe_binding": binding,
        "context_token_proxy": context_tokens,
        "fact_ledger": fact_body,
        "global_raw_omitted_evidence_ids": [],
        "global_raw_selected_evidence_ids": [],
        "global_citation_manifest": assay._global_citation_manifest([]),
        "mode": "spine_indexed_episodic_fact_ledger",
        "parent_all_rendered": True,
        "parent_rows_protected": True,
        "prompt_token_proxy": hot.count_chat_prompt_token_proxy(messages),
        "prompt_workspace_token_proxy": workspace_tokens,
        "provider_messages": messages,
            "provider_payload_sha256": hashlib.sha256(payload).hexdigest(),
            "provider_payload_utf8_bytes": len(payload),
            "raw_collision_bindings": [],
            "rendered_fact_ids": [],
        "rendered_parent_evidence_ids": [],
    }
    row_body = {
        "arms": {"a3_protected_union": arm},
        "format": assay.ROW_FORMAT,
        "local_ordinal": 0,
        "namespace_id": "a" * 64,
        "ordinal": 0,
        "prompt_question_sha256": quote_sha256(question),
        "question_id": "question-test",
        "shard_offset": 0,
        "source_row_receipt_sha256": "e" * 64,
    }
    row = {**row_body, "row_receipt_sha256": identity_sha256(row_body)}
    selection = {
        "candidate_universe_bindings": [binding],
        "format": assay.FORMAT,
        "gold_fields_present": False,
        "implementation": assay._implementation_identity(),
        "population_identity_sha256": assay.EXPECTED_POPULATION_SHA256,
        "provider_calls": 0,
        "question_count": 1,
        "questions": [row],
        "retrieval_sha256": assay.shadow.EXPECTED_RETRIEVAL_SHA256,
        "source_construction_sha256": assay.source_assay.EXPECTED_SOURCE_CONSTRUCTION_SHA256,
        "source_replay_sha256": assay.source_assay.EXPECTED_SOURCE_REPLAY_SHA256,
        "source_runtime_sha256": assay.source_assay.EXPECTED_SOURCE_RUNTIME_SHA256,
        "status": "sealed_gold_free_spine_indexed_episodic_fact_packets",
    }
    selection_sha = "f" * 64
    replay_body = {
        "canonical_semantic_identity": True,
        "format": f"{assay.FORMAT}-semantic-replay-v1",
        "provider_calls": 0,
        "question_count": 1,
        "row_receipts_sha256": identity_sha256([row["row_receipt_sha256"]]),
        "selection_sha256": selection_sha,
        "source_construction_sha256": selection["source_construction_sha256"],
        "source_replay_sha256": selection["source_replay_sha256"],
        "source_runtime_sha256": selection["source_runtime_sha256"],
    }
    replay = {**replay_body, "replay_receipt_sha256": identity_sha256(replay_body)}
    monkeypatch.setattr(assay, "EXPECTED_QUESTION_COUNT", 1)
    monkeypatch.setattr(
        hot,
        "_read_json_artifact",
        lambda path: (replay, "0" * 64)
        if Path(path).name == assay.REPLAY_NAME
        else (selection, selection_sha),
    )
    loaded, digest = assay._load_selection(Path("unused"))
    assert loaded is selection
    assert digest == selection_sha
    tampered = dict(replay)
    tampered["selection_sha256"] = "0" * 64
    unsigned = dict(tampered)
    unsigned.pop("replay_receipt_sha256")
    tampered["replay_receipt_sha256"] = identity_sha256(unsigned)
    monkeypatch.setattr(
        hot,
        "_read_json_artifact",
        lambda path: (tampered, "0" * 64)
        if Path(path).name == assay.REPLAY_NAME
        else (selection, selection_sha),
    )
    with pytest.raises(ValueError, match="semantic replay binding changed"):
        assay._load_selection(Path("unused"))
