"""Matched-eval contracts for the provider-free incremental graph lane."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from memory_condense.application.discourse_sources import scan_discourse_source_chunks
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.schemas import Chunk
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.search.incremental_conversation_graph import (
    ConversationGraphChunk,
    OrderedStorySearchPolicy,
)
from memory_condense.search.indexes.lexical import LexicalIndex
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.full_store_slot_closure import build_full_store_window_index
from tools.matched_eval.hot_incremental_graph_lane import (
    DEFAULT_EVIDENCE_TOKEN_CAP,
    HotIncrementalGraphBudget,
    HotIncrementalGraphLaneError,
    HotIncrementalGraphSeedBinding,
    build_hot_incremental_graph_index,
    query_hot_incremental_graph,
)
from tools.matched_eval.hot_incremental_ordered_story import (
    HotIncrementalOrderedStoryError,
    query_hot_incremental_ordered_story,
)
from tools.matched_eval.query_expansion import FrozenSourceNamespace
from tools.matched_eval.query_guided_scan import cache_namespace_partitions


ASKED_AT = datetime(2026, 9, 6, 12, tzinfo=timezone.utc)
QUESTION = "[Question asked at 2026/09/06 12:00] Where is the cobalt telescope?"


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _binding(
    *seeds: str,
    upstream: tuple[str, ...] | None = None,
) -> HotIncrementalGraphSeedBinding:
    selected = tuple(seeds) if upstream is None else upstream
    return HotIncrementalGraphSeedBinding(
        upstream_retrieval_receipt_sha256=_sha(
            "upstream-retrieval:" + ",".join(selected)
        ),
        upstream_selected_chunk_ids=selected,
        seed_chunk_ids=tuple(seeds),
    )


def _index(
    path: Path,
    rows: tuple[tuple[str, str, str, str], ...],
):
    database = Database(path)
    transcript = TranscriptStore(database)
    lexical = LexicalIndex(database)
    for position, (chunk_id, source_id, role, text) in enumerate(rows):
        turn = transcript.append(
            role,
            text,
            source_id=source_id,
            created_at=ASKED_AT - timedelta(days=len(rows) - position),
        )
        lexical.add_chunks(
            [
                Chunk(
                    chunk_id=chunk_id,
                    turn_id=turn.turn_id,
                    text=text,
                    start_char=0,
                    end_char=len(text),
                    token_count=count_tokens(text),
                )
            ]
        )
    streams = scan_discourse_source_chunks(database)
    database.close()
    store_receipt = _sha(f"{path.name}-store")
    namespace = FrozenSourceNamespace.from_source_streams(
        snapshot_id=_sha(f"{path.name}-snapshot"),
        combined_store_receipt_sha256=store_receipt,
        source_streams=streams,
    )
    with Database(path, read_only=True) as readonly:
        cache = cache_namespace_partitions(
            readonly,
            namespace,
            source_database_sha256=_sha(f"{path.name}-database"),
            source_store_receipt_sha256=store_receipt,
        )
    return build_hot_incremental_graph_index(
        build_full_store_window_index(cache)
    )


def _two_hop_index(path: Path):
    return _index(
        path,
        (
            (
                "local-seed",
                "conversation-a",
                "user",
                "The cobalt telescope is at Cedar Observatory.",
            ),
            (
                "local-link",
                "conversation-a",
                "assistant",
                "Its replacement vendor is Northwind Labs.",
            ),
            (
                "global-fact",
                "document-b",
                "user",
                "Northwind Labs contract carries warranty ZX-9.",
            ),
        ),
    )


def test_cold_build_appends_each_exact_full_store_row_once(tmp_path: Path) -> None:
    index = _two_hop_index(tmp_path / "graph-build.db")

    assert index.graph_stats.chunk_count == len(index.parent.rows) == 3
    assert index.graph_stats.revision == 3
    assert tuple(row.revision for row in index.append_receipts.values()) == (1, 2, 3)
    assert tuple(index.rows_by_chunk_id) == tuple(row.chunk_id for row in index.parent.rows)
    assert all(
        index.graph.chunk(row.chunk_id)
        == ConversationGraphChunk(
            chunk_id=row.chunk_id,
            source_id=row.source_id,
            turn_id=row.turn_id,
            ordinal=row.ordinal,
            role=row.role,
            text=row.text,
            start_char=row.turn_start_char,
            end_char=row.turn_end_char,
            created_at=row.created_at,
            text_sha256=row.text_sha256,
        )
        for row in index.parent.rows
    )
    projection = index.projection()
    assert projection["incremental_append_attempt_count"] == 3
    assert projection["incremental_append_created_count"] == 3
    assert projection["incremental_append_retry_count"] == 0
    assert projection["new_provider_calls"] == projection["model_calls"] == 0
    assert projection["gold_loaded"] is False


def test_two_hop_lane_returns_complete_raw_evidence_and_path_receipts(
    tmp_path: Path,
) -> None:
    index = _two_hop_index(tmp_path / "graph-two-hop.db")

    activation = _binding("local-seed", upstream=("local-seed", "unused-parent"))
    result = query_hot_incremental_graph(index, QUESTION, activation)
    replay = query_hot_incremental_graph(index, QUESTION, activation)

    assert result == replay
    assert result.status == "novel_graph_evidence_selected"
    assert result.selected_before_dedup_ids == ("local-link", "global-fact")
    assert "local-seed" not in result.selected_before_dedup_ids
    assert result.receipt.seed_activation_inputs_emitted is False
    assert result.receipt.seed_activation_inputs_charge_tokens is False
    assert result.receipt.seed_binding == activation
    assert result.receipt.selected_before_dedup_tokens <= DEFAULT_EVIDENCE_TOKEN_CAP

    global_fact = result.selected_before_dedup[1]
    assert global_fact.raw_text == "Northwind Labs contract carries warranty ZX-9."
    assert global_fact.span.start_char == 0
    assert global_fact.span.end_char == len(global_fact.raw_text)
    assert global_fact.graph_hop == 2
    assert [step.relation for step in global_fact.path] == [
        "same_source_sequence",
        "shared_phrase",
    ]
    assert global_fact.path[0].source_chunk_id == "local-seed"
    assert global_fact.supporting_seed_chunk_ids == ("local-seed",)
    assert global_fact.path[-1].target_chunk_id == "global-fact"
    bridge = global_fact.path[-1]
    assert bridge.source_occurrence is not None
    assert bridge.target_occurrence is not None
    assert bridge.source_occurrence.quote == "Northwind Labs"
    assert bridge.target_occurrence.quote == "Northwind Labs"
    assert bridge.source_append_receipt_sha256 == index.append_receipts[
        "local-link"
    ].receipt_sha256
    assert bridge.target_append_receipt_sha256 == index.append_receipts[
        "global-fact"
    ].receipt_sha256
    assert global_fact.append_receipt_sha256 == index.append_receipts[
        "global-fact"
    ].receipt_sha256
    assert global_fact.index_receipt_sha256 == index.receipt_sha256
    assert global_fact.full_store_index_receipt_sha256 == index.parent.receipt_sha256
    assert global_fact.cache_receipt_sha256 == index.parent.cache.cache_receipt_sha256
    audit = result.audit_projection()
    assert audit["receipt"]["selection_before_global_dedup"] is True
    assert audit["receipt"]["new_provider_calls"] == 0
    assert audit["receipt"]["model_calls"] == 0
    assert audit["receipt"]["gold_loaded"] is False

    with pytest.raises(HotIncrementalGraphLaneError, match="path receipts"):
        replace(
            global_fact,
            supporting_seed_chunk_ids=("local-link",),
            receipt_sha256="",
        )


def test_no_seed_and_no_novel_graph_reach_have_distinct_statuses(
    tmp_path: Path,
) -> None:
    index = _index(
        tmp_path / "graph-noops.db",
        (("isolated", "only-source", "user", "An isolated cobalt fact."),),
    )

    no_seed = query_hot_incremental_graph(index, QUESTION, _binding())
    no_novel = query_hot_incremental_graph(
        index, QUESTION, _binding("isolated")
    )

    assert no_seed.status == "no_seed"
    assert no_seed.selected_before_dedup == ()
    assert no_seed.receipt.candidate_population_ids == ()
    assert no_novel.status == "no_novel_graph_evidence"
    assert no_novel.selected_before_dedup == ()
    assert no_novel.receipt.seed_chunk_ids == ("isolated",)
    assert no_novel.receipt.candidate_population_ids == ()


def test_source_round_robin_precedes_second_hit_from_first_source(
    tmp_path: Path,
) -> None:
    index = _index(
        tmp_path / "graph-source-diversity.db",
        (
            ("seed", "seed-source", "user", "Bridgeword anchor."),
            ("a-one", "source-one", "user", "Bridgeword alpha."),
            ("b-one", "source-one", "assistant", "Bridgeword beta."),
            ("z-two", "source-two", "user", "Bridgeword zeta."),
        ),
    )

    result = query_hot_incremental_graph(
        index,
        "[Question asked at 2026/09/06 12:00] Follow the bridge.",
        _binding("seed"),
        budget=HotIncrementalGraphBudget(max_hops=1),
    )

    assert result.receipt.graph_ranked_novel_ids == (
        "a-one",
        "b-one",
        "z-two",
    )
    assert result.receipt.candidate_population_ids == (
        "a-one",
        "z-two",
        "b-one",
    )
    assert result.receipt.candidate_source_ids == (
        "source-one",
        "source-two",
        "source-one",
    )
    assert result.selected_before_dedup_ids == result.receipt.candidate_population_ids


def test_large_seed_is_free_and_overbudget_candidate_does_not_block_later_fact(
    tmp_path: Path,
) -> None:
    large_seed = ("activationword " * 1_300).strip()
    too_large = "Bridgeword " + ("payload " * 1_500).strip()
    index = _index(
        tmp_path / "graph-budget.db",
        (
            ("large-seed", "seed-source", "user", large_seed + " Bridgeword"),
            ("a-too-large", "large-source", "user", too_large),
            ("z-small", "small-source", "user", "Bridgeword concise fact."),
        ),
    )

    result = query_hot_incremental_graph(
        index,
        "[Question asked at 2026/09/06 12:00] Follow the memory bridge.",
        _binding("large-seed"),
        budget=HotIncrementalGraphBudget(max_hops=1),
    )

    assert result.receipt.seed_activation_input_tokens > DEFAULT_EVIDENCE_TOKEN_CAP
    assert result.receipt.seed_activation_inputs_charge_tokens is False
    assert result.receipt.seed_activation_inputs_emitted is False
    assert result.receipt.candidate_population_ids == ("a-too-large", "z-small")
    assert result.receipt.budget_excluded_ids == ("a-too-large",)
    assert result.selected_before_dedup_ids == ("z-small",)
    assert result.receipt.selected_before_dedup_tokens == index.rows_by_chunk_id[
        "z-small"
    ].token_count
    assert result.receipt.selection_truncated is True


def test_all_novel_chunks_over_budget_reports_explicit_status(tmp_path: Path) -> None:
    too_large = "Bridgeword " + ("payload " * 1_500).strip()
    index = _index(
        tmp_path / "graph-all-over.db",
        (
            ("seed", "seed-source", "user", "Bridgeword anchor."),
            ("huge", "huge-source", "user", too_large),
        ),
    )

    result = query_hot_incremental_graph(
        index,
        "[Question asked at 2026/09/06 12:00] Follow the bridge.",
        _binding("seed"),
        budget=HotIncrementalGraphBudget(max_hops=1),
    )

    assert result.status == "novel_graph_evidence_all_over_budget"
    assert result.selected_before_dedup == ()
    assert result.receipt.budget_excluded_ids == ("huge",)


def test_invalid_question_seeds_and_postseal_mutation_fail_closed(
    tmp_path: Path,
) -> None:
    index = _two_hop_index(tmp_path / "graph-invalid.db")

    with pytest.raises(HotIncrementalGraphLaneError, match="dated question"):
        query_hot_incremental_graph(index, "Where is it?", _binding("local-seed"))
    with pytest.raises(HotIncrementalGraphLaneError, match="absent from full store"):
        query_hot_incremental_graph(index, QUESTION, _binding("missing"))
    with pytest.raises(HotIncrementalGraphLaneError, match="sealed upstream"):
        query_hot_incremental_graph(  # type: ignore[arg-type]
            index, QUESTION, ("local-seed",)
        )
    with pytest.raises(HotIncrementalGraphLaneError, match="ordered unique"):
        HotIncrementalGraphSeedBinding(
            upstream_retrieval_receipt_sha256=_sha("duplicate-upstream"),
            upstream_selected_chunk_ids=("local-seed",),
            seed_chunk_ids=("local-seed", "local-seed"),
        )
    with pytest.raises(HotIncrementalGraphLaneError, match="ordered subset"):
        _binding("oracle-only", upstream=("local-seed",))
    with pytest.raises(HotIncrementalGraphLaneError, match="hard bound"):
        too_many = tuple(f"seed-{number}" for number in range(17))
        query_hot_incremental_graph(
            index,
            QUESTION,
            _binding(*too_many),
        )

    extra = ConversationGraphChunk(
        chunk_id="postseal",
        source_id="postseal-source",
        turn_id="postseal-turn",
        ordinal=99,
        role="user",
        text="Postseal graph mutation.",
        start_char=0,
        end_char=24,
        created_at="2026-09-06T12:00:00+00:00",
    )
    index.graph.append_chunk(extra)
    with pytest.raises(HotIncrementalGraphLaneError, match="changed after"):
        query_hot_incremental_graph(index, QUESTION, _binding("local-seed"))


def test_authenticated_ordered_story_uses_only_sealed_upstream_candidates(
    tmp_path: Path,
) -> None:
    index = _index(
        tmp_path / "ordered-story.db",
        (
            (
                "muir-event",
                "muir-session",
                "user",
                "I returned from a day hike to Muir Woods with family.",
            ),
            (
                "muir-context",
                "muir-session",
                "user",
                "Eastern Sierra backpacking requires careful bear safety.",
            ),
            (
                "big-sur-event",
                "big-sur-session",
                "user",
                "I returned from a road trip with friends to Big Sur.",
            ),
            (
                "big-sur-context",
                "big-sur-session",
                "user",
                "Eastern Sierra backpacking emphasizes careful bear safety.",
            ),
            (
                "yosemite-event",
                "yosemite-session",
                "user",
                "I returned from a solo camping trip to Yosemite.",
            ),
            (
                "yosemite-context",
                "yosemite-session",
                "user",
                "Eastern Sierra backpacking made bear safety second nature.",
            ),
            (
                "new-york-event",
                "new-york-session",
                "user",
                "I took a business trip to New York for a conference.",
            ),
            (
                "whistler-event",
                "whistler-session",
                "user",
                "I took a ski trip to Whistler Blackcomb.",
            ),
        ),
    )
    candidate_ids = (
        "new-york-event",
        "whistler-event",
        "muir-event",
        "big-sur-event",
        "yosemite-event",
    )
    binding = _binding(*candidate_ids)

    result = query_hot_incremental_ordered_story(
        index,
        binding,
        requested_count=3,
        policy=OrderedStorySearchPolicy(max_seed_chunks=8),
    )
    replay = query_hot_incremental_ordered_story(
        index,
        binding,
        requested_count=3,
        policy=OrderedStorySearchPolicy(max_seed_chunks=8),
    )

    assert result == replay
    assert result.core.status == "selected"
    assert result.core.selected_source_ids == (
        "muir-session",
        "big-sur-session",
        "yosemite-session",
    )
    assert result.receipt.candidate_chunk_ids == candidate_ids
    audit = result.audit_projection()
    assert audit["receipt"]["seed_binding"] == binding.projection()
    assert audit["receipt"]["new_provider_calls"] == 0
    assert audit["receipt"]["model_calls"] == 0
    assert audit["receipt"]["gold_loaded"] is False

    missing = HotIncrementalGraphSeedBinding(
        upstream_retrieval_receipt_sha256=_sha("missing-candidate"),
        upstream_selected_chunk_ids=("muir-event", "not-in-store"),
        seed_chunk_ids=("muir-event",),
    )
    with pytest.raises(
        HotIncrementalOrderedStoryError,
        match="absent from full store",
    ):
        query_hot_incremental_ordered_story(
            index,
            missing,
            requested_count=2,
        )


def test_authenticated_ordered_story_rejects_empty_seed_binding(
    tmp_path: Path,
) -> None:
    index = _index(
        tmp_path / "ordered-story-empty-seed.db",
        (("candidate", "source", "user", "A dated memory."),),
    )
    binding = _binding(upstream=("candidate",))

    with pytest.raises(
        HotIncrementalOrderedStoryError,
        match="authenticated seed",
    ):
        query_hot_incremental_ordered_story(
            index,
            binding,
            requested_count=1,
        )
