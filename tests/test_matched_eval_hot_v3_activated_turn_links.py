from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from memory_condense.application.discourse_sources import scan_discourse_source_chunks
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.schemas import Chunk
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.search.indexes.lexical import LexicalIndex
from memory_condense.search.source_neighborhood import (
    SourceChunkMetadata,
    SourceNeighborhoodIndex,
)

from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.full_store_slot_closure import build_full_store_window_index
from tools.matched_eval.hot_v3_activated_turn_links import (
    DEFAULT_TOKEN_CAP,
    build_hot_v3_activated_turn_link_index,
    select_hot_v3_activated_turn_links,
)
from tools.matched_eval.query_expansion import FrozenSourceNamespace
from tools.matched_eval.query_guided_scan import cache_namespace_partitions


ASKED_AT = datetime(2026, 9, 6, 12, tzinfo=timezone.utc)


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _add_turn_chunks(
    transcript: TranscriptStore,
    lexical: LexicalIndex,
    *,
    role: str,
    source_id: str,
    text: str,
    chunk_specs: tuple[tuple[str, int, int], ...],
    created_at: datetime,
) -> None:
    turn = transcript.append(
        role,
        text,
        source_id=source_id,
        created_at=created_at,
    )
    lexical.add_chunks(
        [
            Chunk(
                chunk_id=chunk_id,
                turn_id=turn.turn_id,
                text=text[start:end],
                start_char=start,
                end_char=end,
                token_count=count_tokens(text[start:end]),
            )
            for chunk_id, start, end in chunk_specs
        ]
    )


def _index(path: Path):
    database = Database(path)
    transcript = TranscriptStore(database)
    lexical = LexicalIndex(database)
    source_id = "profile-story::screen-preference"
    profile = "I prefer understated science-fiction films and dislike jump scares."
    _add_turn_chunks(
        transcript,
        lexical,
        role="user",
        source_id=source_id,
        text=profile,
        chunk_specs=(("profile-seed", 0, len(profile)),),
        created_at=ASKED_AT - timedelta(days=3),
    )
    response = (
        "That profile suggests thoughtful speculative dramas. "
        "Arrival is one option, and Contact is another."
    )
    second_start = response.index("Arrival")
    _add_turn_chunks(
        transcript,
        lexical,
        role="assistant",
        source_id=source_id,
        text=response,
        chunk_specs=(
            ("assistant-successor-a", 0, second_start - 1),
            ("assistant-successor-b", second_start, len(response)),
        ),
        created_at=ASKED_AT - timedelta(days=3) + timedelta(minutes=1),
    )
    too_far = "This later user turn must not be linked from the profile seed."
    _add_turn_chunks(
        transcript,
        lexical,
        role="user",
        source_id=source_id,
        text=too_far,
        chunk_specs=(("too-far", 0, len(too_far)),),
        created_at=ASKED_AT - timedelta(days=2),
    )
    parent_other = "A separate parent result remains after exact chunk dedup."
    _add_turn_chunks(
        transcript,
        lexical,
        role="user",
        source_id="other-story::retained",
        text=parent_other,
        chunk_specs=(("parent-other", 0, len(parent_other)),),
        created_at=ASKED_AT - timedelta(days=1),
    )
    streams = scan_discourse_source_chunks(database)
    database.close()
    store_receipt = _sha("activated-turn-link-store")
    namespace = FrozenSourceNamespace.from_source_streams(
        snapshot_id=_sha("activated-turn-link-snapshot"),
        combined_store_receipt_sha256=store_receipt,
        source_streams=streams,
    )
    with Database(path, read_only=True) as readonly:
        cache = cache_namespace_partitions(
            readonly,
            namespace,
            source_database_sha256=_sha("activated-turn-link-database"),
            source_store_receipt_sha256=store_receipt,
        )
    full_store = build_full_store_window_index(cache)
    neighborhood = SourceNeighborhoodIndex(
        tuple(
            SourceChunkMetadata(
                chunk_id=row.chunk_id,
                source_id=row.source_id,
                turn_id=row.turn_id,
                ordinal=row.ordinal,
                start_char=row.turn_start_char,
            )
            for row in full_store.rows
        )
    )
    return build_hot_v3_activated_turn_link_index(
        full_store,
        source_neighborhood_index=neighborhood,
    )


def _large_seed_index(path: Path):
    database = Database(path)
    transcript = TranscriptStore(database)
    lexical = LexicalIndex(database)
    for ordinal in range(4):
        source_id = f"large-seed-story::{ordinal}"
        activation = (f"activation-{ordinal} " * 400).strip()
        _add_turn_chunks(
            transcript,
            lexical,
            role="user",
            source_id=source_id,
            text=activation,
            chunk_specs=((f"large-seed-{ordinal}", 0, len(activation)),),
            created_at=ASKED_AT - timedelta(days=10 - ordinal),
        )
        neighbor = f"Small linked neighbor fact {ordinal}."
        _add_turn_chunks(
            transcript,
            lexical,
            role="assistant",
            source_id=source_id,
            text=neighbor,
            chunk_specs=((f"small-neighbor-{ordinal}", 0, len(neighbor)),),
            created_at=ASKED_AT
            - timedelta(days=10 - ordinal)
            + timedelta(minutes=1),
        )
    streams = scan_discourse_source_chunks(database)
    database.close()
    store_receipt = _sha("large-activated-turn-link-store")
    namespace = FrozenSourceNamespace.from_source_streams(
        snapshot_id=_sha("large-activated-turn-link-snapshot"),
        combined_store_receipt_sha256=store_receipt,
        source_streams=streams,
    )
    with Database(path, read_only=True) as readonly:
        cache = cache_namespace_partitions(
            readonly,
            namespace,
            source_database_sha256=_sha("large-activated-turn-link-database"),
            source_store_receipt_sha256=store_receipt,
        )
    return build_hot_v3_activated_turn_link_index(
        build_full_store_window_index(cache)
    )


def test_profile_seed_adds_every_chunk_of_assistant_successor_then_dedups_parent(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path / "activated-turn-links.db")
    parent_ids = (
        "profile-seed",
        "assistant-successor-a",
        "opaque-projection-id",
        "parent-other",
    )

    result = select_hot_v3_activated_turn_links(
        index,
        ("profile-seed",),
        parent_chunk_ids=parent_ids,
    )
    without_parent = select_hot_v3_activated_turn_links(
        index,
        ("profile-seed",),
    )

    selected_ids = tuple(row.chunk_id for row in result.selected_before_dedup)
    assert selected_ids == (
        "assistant-successor-a",
        "assistant-successor-b",
    )
    assert selected_ids == tuple(
        row.chunk_id for row in without_parent.selected_before_dedup
    )
    assert result.selected_before_dedup_ids == selected_ids
    assert tuple(row.role for row in result.selected_before_dedup) == (
        "assistant",
        "assistant",
    )
    assert [(row.linked_chunk_id, row.direction) for row in result.links] == [
        ("assistant-successor-a", "successor_turn"),
        ("assistant-successor-b", "successor_turn"),
    ]
    assert result.receipt.selected_before_dedup_tokens <= DEFAULT_TOKEN_CAP
    assert result.receipt.seed_activation_inputs_charge_tokens is False
    assert result.receipt.seed_activation_inputs_emitted is False
    assert result.receipt.exact_parent_duplicate_ids == ("assistant-successor-a",)
    assert result.receipt.activated_retained_after_dedup_ids == selected_ids
    assert result.receipt.parent_retained_after_dedup_ids == (
        "profile-seed",
        "opaque-projection-id",
        "parent-other",
    )
    assert result.receipt.unmaterialized_parent_retained_ids == (
        "opaque-projection-id",
    )
    assert result.receipt.retained_after_dedup_ids == (
        *selected_ids,
        "profile-seed",
        "opaque-projection-id",
        "parent-other",
    )
    assert result.retained_after_dedup_ids == result.receipt.retained_after_dedup_ids
    assert result.receipt.selection_before_parent_dedup is True
    assert result.receipt.selection_before_global_dedup is True
    assert result.receipt.activated_lane_wins_exact_chunk_collision is True
    assert result.receipt.refill_after_parent_dedup is False
    assert "too-far" not in result.receipt.candidate_population_ids
    assert "too-far" not in result.receipt.retained_after_dedup_ids

    second_successor = result.selected_before_dedup[1]
    assert second_successor.raw_text == "Arrival is one option, and Contact is another."
    assert second_successor.span.start_char == 0
    assert second_successor.span.turn_start_char > 0
    assert second_successor.span_receipt_sha256 == identity_sha256(
        second_successor.span.identity_payload()
    )
    source_row = index.rows_by_chunk_id[second_successor.chunk_id]
    assert second_successor.source_row_receipt_sha256 == identity_sha256(
        source_row.receipt_projection()
    )

    assert result.receipt.gold_loaded is False
    assert result.receipt.new_provider_calls == 0
    assert result.receipt.model_calls == 0
    assert result.receipt.retained_transformer_token_state_bytes == 0
    assert result.audit_projection()["receipt"]["gold_loaded"] is False
    assert result == select_hot_v3_activated_turn_links(
        index,
        ("profile-seed",),
        parent_chunk_ids=parent_ids,
    )


def test_large_many_activation_seeds_cannot_starve_linked_neighbor_budget(
    tmp_path: Path,
) -> None:
    index = _large_seed_index(tmp_path / "large-seed-turn-links.db")
    seed_ids = tuple(f"large-seed-{ordinal}" for ordinal in range(4))
    neighbor_ids = tuple(f"small-neighbor-{ordinal}" for ordinal in range(4))
    parent_ids = (*seed_ids, "small-neighbor-1", "opaque-parent")

    result = select_hot_v3_activated_turn_links(
        index,
        seed_ids,
        parent_chunk_ids=parent_ids,
    )

    assert result.receipt.seed_activation_input_tokens == sum(
        index.rows_by_chunk_id[value].token_count for value in seed_ids
    )
    assert result.receipt.seed_activation_input_tokens > DEFAULT_TOKEN_CAP
    assert result.receipt.seed_activation_inputs_charge_tokens is False
    assert result.receipt.seed_activation_inputs_emitted is False
    assert result.receipt.candidate_population_ids == neighbor_ids
    assert result.receipt.candidate_neighbor_ids_sha256 == identity_sha256(
        list(neighbor_ids)
    )
    assert result.receipt.neighborhood_paths_sha256 == identity_sha256(
        list(result.receipt.neighborhood_link_receipt_sha256s)
    )
    assert result.selected_before_dedup_ids == neighbor_ids
    assert all(row.origin == "neighbor" for row in result.selected_before_dedup)
    assert result.receipt.selected_before_dedup_tokens == sum(
        index.rows_by_chunk_id[value].token_count for value in neighbor_ids
    )
    assert result.receipt.selected_before_dedup_tokens <= DEFAULT_TOKEN_CAP
    assert result.receipt.budget_excluded_ids == ()
    assert result.receipt.exact_parent_duplicate_ids == ("small-neighbor-1",)
    assert result.receipt.parent_retained_after_dedup_ids == (
        *seed_ids,
        "opaque-parent",
    )
    assert result.receipt.retained_after_dedup_ids == (
        *neighbor_ids,
        *seed_ids,
        "opaque-parent",
    )
