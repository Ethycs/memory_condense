"""Atomic final packing for sealed conversation-envelope expansions."""

from __future__ import annotations

import numpy as np

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.domain.schemas import Chunk
from memory_condense.search.packing.context_packer import ContextBudget


class _Embedder:
    dim = 8

    def embed_query(self, _query: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        vector[0] = 1.0
        return vector

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        vector = self.embed_query("").tolist()
        return [chunk.model_copy(update={"embedding": vector}) for chunk in chunks]


def _condenser(path, budget: ContextBudget) -> MemoryCondenser:
    return MemoryCondenser(
        data_dir=path,
        embedder=_Embedder(),
        budget=budget,
        auto_extract=False,
        chunker_min_tokens=1,
        chunker_max_tokens=250,
    )


def _raw(condenser: MemoryCondenser, chunk_id: str, *, score: float = 0.9):
    result = condenser.retriever.hydrate_chunk(
        chunk_id,
        score=score,
        route="hot_raw",
    )
    assert result is not None
    return result


def _pack(condenser: MemoryCondenser, expansion_results):
    return condenser.build_context(
        "render the exchange",
        recent_turns=0,
        k_memories=0,
        k_expansions=0,
        use_consolidation=False,
        learn_consolidation=False,
        reheat_memories=False,
        expansion_results=expansion_results,
    )


def test_receipt_adds_companion_without_consuming_the_raw_count_slot(tmp_path) -> None:
    budget = ContextBudget(
        expansion_tokens=80,
        max_expansions=1,
        max_expansion_tokens=40,
    )
    with _condenser(tmp_path / "companion-slot", budget) as condenser:
        rows = condenser.ingest_many(
            [
                ("user", "user opens the exchange", "session-a", None, "u1"),
                (
                    "assistant",
                    "assistant supplies the answer",
                    "session-a",
                    None,
                    "a1",
                ),
            ]
        )
        anchor = _raw(condenser, rows[1][1][0].chunk_id)
        expanded = condenser.expand_conversation_envelopes([anchor])

        packed = _pack(condenser, expanded)

        assert packed.expansion_chunk_ids == [
            rows[0][1][0].chunk_id,
            anchor.chunk.chunk_id,
        ]
        assert packed.token_counts["expansions"] <= budget.expansion_tokens


def test_plain_sequence_keeps_normal_priority_while_receipt_falls_back_to_raw(
    tmp_path,
) -> None:
    budget = ContextBudget(
        expansion_tokens=80,
        max_expansions=1,
        max_expansion_tokens=40,
    )
    with _condenser(tmp_path / "sequence-parity", budget) as condenser:
        rows = condenser.ingest_many(
            [
                ("user", "raw user row", "session-b", None, "u2"),
                ("assistant", "raw assistant row", "session-b", None, "a2"),
            ]
        )
        user_hit = _raw(condenser, rows[0][1][0].chunk_id, score=0.8)
        assistant_hit = _raw(condenser, rows[1][1][0].chunk_id, score=0.95)
        expanded = condenser.expand_conversation_envelopes(
            [assistant_hit, user_hit]
        )

        plain = _pack(condenser, expanded.results)
        atomic = _pack(condenser, expanded)

        assert plain.expansion_chunk_ids == [user_hit.chunk.chunk_id]
        assert atomic.expansion_chunk_ids == [assistant_hit.chunk.chunk_id]


def test_budget_aware_policy_readds_a_missing_group_original_atomically(
    tmp_path,
) -> None:
    budget = ContextBudget(
        expansion_tokens=40,
        max_expansions=2,
        max_expansion_tokens=250,
        budget_aware_expansions=True,
    )
    with _condenser(tmp_path / "budget-aware", budget) as condenser:
        rows = condenser.ingest_many(
            [
                (
                    "user",
                    "long opener evidence " * 50,
                    "session-c",
                    None,
                    "u3",
                ),
                ("assistant", "short answer", "session-c", None, "a3"),
            ]
        )
        user_hit = _raw(condenser, rows[0][1][0].chunk_id, score=0.2)
        assistant_hit = _raw(condenser, rows[1][1][0].chunk_id, score=1.0)
        expanded = condenser.expand_conversation_envelopes(
            [assistant_hit, user_hit]
        )

        packed = _pack(condenser, expanded)

        assert packed.expansion_chunk_ids == [
            user_hit.chunk.chunk_id,
            assistant_hit.chunk.chunk_id,
        ]
        assert packed.token_counts["expansions"] <= budget.expansion_tokens


def test_selector_sees_raw_lane_and_atomic_pack_restores_user_first(tmp_path) -> None:
    class AssistantOnlySelector:
        requires_baseline_ranking = True
        last_report = None
        last_candidate_trace = []

        def __init__(self) -> None:
            self.seen: list[str] = []

        def select(self, _query, candidates, **_kwargs):
            self.seen = [result.chunk.turn_id for result in candidates]
            return [
                result
                for result in candidates
                if result.chunk.turn_id == "a4"
            ]

    budget = ContextBudget(
        expansion_tokens=40,
        max_expansions=2,
        max_expansion_tokens=250,
        budget_aware_expansions=True,
    )
    with _condenser(tmp_path / "selector", budget) as condenser:
        rows = condenser.ingest_many(
            [
                (
                    "user",
                    "long selector opener " * 50,
                    "session-d",
                    None,
                    "u4",
                ),
                ("assistant", "selected answer", "session-d", None, "a4"),
            ]
        )
        user_hit = _raw(condenser, rows[0][1][0].chunk_id, score=0.2)
        assistant_hit = _raw(condenser, rows[1][1][0].chunk_id, score=1.0)
        expanded = condenser.expand_conversation_envelopes(
            [assistant_hit, user_hit]
        )
        selector = AssistantOnlySelector()
        condenser.set_context_candidate_selector(selector)

        packed = _pack(condenser, expanded)

        assert selector.seen == ["a4"]
        assert packed.expansion_chunk_ids == [
            user_hit.chunk.chunk_id,
            assistant_hit.chunk.chunk_id,
        ]


def test_heat_weighted_policy_keeps_each_admitted_group_user_first(tmp_path) -> None:
    budget = ContextBudget(
        expansion_tokens=100,
        max_expansions=2,
        max_expansion_tokens=40,
        heat_weighted_expansions=True,
    )
    with _condenser(tmp_path / "heat", budget) as condenser:
        rows = condenser.ingest_many(
            [
                ("user", "heated opener", "session-e", None, "u5"),
                ("assistant", "heated answer", "session-e", None, "a5"),
                ("assistant", "outside evidence", "outside", None, "outside"),
            ]
        )
        anchor = _raw(condenser, rows[1][1][0].chunk_id).model_copy(
            update={"source_heat": 0.2}
        )
        outside = _raw(condenser, rows[2][1][0].chunk_id).model_copy(
            update={"source_heat": 0.8}
        )
        expanded = condenser.expand_conversation_envelopes([anchor, outside])

        packed = _pack(condenser, expanded)

        user_index = packed.expansion_chunk_ids.index(rows[0][1][0].chunk_id)
        anchor_index = packed.expansion_chunk_ids.index(anchor.chunk.chunk_id)
        assert anchor_index == user_index + 1
        assert packed.token_counts["expansions"] <= budget.expansion_tokens


def test_tight_shared_token_cap_rejects_whole_group_and_keeps_anchor(tmp_path) -> None:
    budget = ContextBudget(
        expansion_tokens=12,
        max_expansions=1,
        max_expansion_tokens=40,
    )
    with _condenser(tmp_path / "token-fallback", budget) as condenser:
        rows = condenser.ingest_many(
            [
                ("user", "U", "session-f", None, "u6"),
                ("assistant", "A", "session-f", None, "a6"),
            ]
        )
        anchor = _raw(condenser, rows[1][1][0].chunk_id)
        expanded = condenser.expand_conversation_envelopes([anchor])

        packed = _pack(condenser, expanded)

        assert packed.expansion_chunk_ids == [anchor.chunk.chunk_id]
        assert packed.token_counts["expansions"] <= budget.expansion_tokens


def test_sealed_receipt_skips_frontier_mutating_source_refresh(
    tmp_path,
    monkeypatch,
) -> None:
    budget = ContextBudget(
        expansion_tokens=100,
        max_expansions=1,
        max_expansion_tokens=40,
        source_metadata_expansions=True,
    )
    with _condenser(tmp_path / "source-metadata", budget) as condenser:
        rows = condenser.ingest_many(
            [
                ("user", "dated opener", "session-g", None, "u7"),
                ("assistant", "dated answer", "session-g", None, "a7"),
            ]
        )
        anchor = _raw(condenser, rows[1][1][0].chunk_id)
        expanded = condenser.expand_conversation_envelopes([anchor])

        def forbidden_refresh(*_args, **_kwargs):
            raise AssertionError("sealed envelope rows must not be replaced")

        monkeypatch.setattr(
            condenser,
            "_hydrate_source_metadata_companions",
            forbidden_refresh,
        )

        packed = _pack(condenser, expanded)

        assert packed.expansion_chunk_ids == [
            rows[0][1][0].chunk_id,
            anchor.chunk.chunk_id,
        ]
