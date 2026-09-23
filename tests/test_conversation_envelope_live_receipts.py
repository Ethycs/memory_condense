"""Liveness and receipt regressions for conversation-envelope expansion."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.application.conversation_envelope_retrieval import (
    hydrate_conversation_envelope_plan,
)
from memory_condense.domain.schemas import Chunk, RetrievalResult


class _Embedder:
    dim = 8

    def embed_query(self, _query: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        vector[0] = 1.0
        return vector

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        vector = self.embed_query("").tolist()
        return [chunk.model_copy(update={"embedding": vector}) for chunk in chunks]


def _condenser(path) -> MemoryCondenser:
    return MemoryCondenser(
        data_dir=path,
        embedder=_Embedder(),
        auto_extract=False,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    )


def _raw(condenser: MemoryCondenser, chunk_id: str) -> RetrievalResult:
    result = condenser.retriever.hydrate_chunk(
        chunk_id,
        score=0.9,
        route="hot_raw",
    )
    assert result is not None
    return result


def _exchange(condenser: MemoryCondenser, source_id: str):
    rows = condenser.ingest_many(
        [
            ("user", "durable private lead", source_id, None, f"{source_id}-u"),
            (
                "assistant",
                "durable private answer",
                source_id,
                None,
                f"{source_id}-a",
            ),
        ]
    )
    lead_chunk_id = rows[0][1][0].chunk_id
    anchor = _raw(condenser, rows[1][1][0].chunk_id)
    return lead_chunk_id, anchor


def _plan(condenser: MemoryCondenser, anchor: RetrievalResult):
    return condenser._conversation_envelopes.plan_retrieval_expansion(
        (anchor.chunk.turn_id,),
        original_chunk_token_counts={
            anchor.chunk.chunk_id: anchor.chunk.token_count,
        },
    )


def test_retired_chunk_is_excluded_before_planning(tmp_path) -> None:
    with _condenser(tmp_path / "retired-before-plan") as condenser:
        lead_chunk_id, anchor = _exchange(condenser, "retired-before-plan")
        assert condenser.retriever.delete_chunk(lead_chunk_id)
        assert lead_chunk_id not in condenser._conversation_envelopes.live_chunk_ids(
            (lead_chunk_id,)
        )

        expanded = condenser.expand_conversation_envelopes([anchor])

        assert expanded.results == (anchor,)
        assert expanded.results[0] is anchor
        assert expanded.companion_chunk_ids == ()
        assert expanded.plan is not None and expanded.plan.groups == ()
        assert [row.reason for row in expanded.plan.diagnostics] == [
            "mandatory_turn_has_no_chunks"
        ]


def test_retirement_after_plan_fails_group_open_atomically(tmp_path) -> None:
    with _condenser(tmp_path / "retired-after-plan") as condenser:
        lead_chunk_id, anchor = _exchange(condenser, "retired-after-plan")
        plan = _plan(condenser, anchor)
        assert len(plan.groups) == 1
        assert lead_chunk_id in plan.groups[0].ordered_chunk_ids
        assert condenser.retriever.delete_chunk(lead_chunk_id)

        expanded = hydrate_conversation_envelope_plan(
            [anchor],
            plan=plan,
            hydrate_chunk=condenser.retriever.hydrate_chunk,
            live_chunk_ids=condenser._conversation_envelopes.live_chunk_ids,
            max_companion_chunks=plan.max_companion_chunks,
            max_companion_tokens=plan.max_companion_tokens,
        )

        assert expanded.results == (anchor,)
        assert expanded.results[0] is anchor
        assert expanded.companion_chunk_ids == ()
        assert expanded.unhydrated_chunk_ids == (lead_chunk_id,)
        assert [row.reason for row in expanded.hydration_diagnostics] == [
            "chunk_not_live"
        ]


def test_retirement_during_hydration_discards_whole_group(tmp_path) -> None:
    with _condenser(tmp_path / "retired-during-hydration") as condenser:
        lead_chunk_id, anchor = _exchange(condenser, "retired-during-hydration")
        plan = _plan(condenser, anchor)
        checks = 0

        def retire_before_postcheck(chunk_ids):
            nonlocal checks
            checks += 1
            if checks == 2:
                assert condenser.retriever.delete_chunk(lead_chunk_id)
            return condenser._conversation_envelopes.live_chunk_ids(chunk_ids)

        expanded = hydrate_conversation_envelope_plan(
            [anchor],
            plan=plan,
            hydrate_chunk=condenser.retriever.hydrate_chunk,
            live_chunk_ids=retire_before_postcheck,
            max_companion_chunks=plan.max_companion_chunks,
            max_companion_tokens=plan.max_companion_tokens,
        )

        assert checks == 2
        assert expanded.results == (anchor,)
        assert expanded.companion_chunk_ids == ()
        assert expanded.unhydrated_chunk_ids == (lead_chunk_id,)
        assert [row.reason for row in expanded.hydration_diagnostics] == [
            "chunk_retired_during_hydration"
        ]


def test_receipt_seals_role_source_heat_and_diffusion_heat(tmp_path) -> None:
    with _condenser(tmp_path / "receipt-state") as condenser:
        _lead_chunk_id, anchor = _exchange(condenser, "receipt-state")
        anchor = anchor.model_copy(
            update={"source_heat": 0.25, "diffusion_heat": 0.5}
        )
        expanded = condenser.expand_conversation_envelopes([anchor])
        original_index = next(
            index
            for index, result in enumerate(expanded.results)
            if result.chunk.chunk_id == anchor.chunk.chunk_id
        )
        original = expanded.results[original_index]
        assert original.turn is not None
        mutations = (
            original.model_copy(update={"source_heat": 0.75}),
            original.model_copy(update={"diffusion_heat": 0.875}),
            original.model_copy(
                update={
                    "turn": original.turn.model_copy(update={"role": "system"})
                }
            ),
        )

        for mutation in mutations:
            tampered = list(expanded.results)
            tampered[original_index] = mutation
            with pytest.raises(ValueError, match="result row digest"):
                replace(
                    expanded,
                    results=tuple(tampered),
                    receipt_sha256="",
                )
