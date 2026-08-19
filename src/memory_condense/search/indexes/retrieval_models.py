"""Immutable retrieval rows and pure payload hydration helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass

from memory_condense.domain.schemas import Chunk, RetrievalResult, Turn
from memory_condense.persistence.db import Database


#: Span sizes that :meth:`SimilarityRetriever.span_query` pools over, in
#: **tokens of the pooled span**, not in chunks.
#:
#: Measured in chunks first, which was a bug waiting for a different corpus:
#: 4 and 8 chunks is ~110/~220 tokens on short-turn dialogue but ~900/~1800 on
#: long-form prose, where chunks already run 227 tokens. The same setting would
#: have helped dialogue and quietly wrecked monologue. A token target is
#: corpus-independent — it asks for a span of a given size regardless of how
#: many chunks that takes.
#:
#: 110 and 220 are the measured optima, translated. They are not round numbers:
#: at ~440 tokens (16 chunks) recall *falls* — 20.6% against 21.6% — while
#: costing 2.2x, because a mean vector washes out once a span straddles several
#: topics. Two levels beat one: stratified scored 22.1% at 660 tokens where the
#: single coarse level gave 21.6% at 673.
DEFAULT_SPAN_TOKENS: tuple[int, ...] = (110, 220)


@dataclass(frozen=True, slots=True)
class PartitionContentRow:
    """One exact raw chunk reached by a hierarchical partition scan.

    The row deliberately contains no embedding or derived semantic label.  A
    caller may inspect every row with cheap query-specific logic, retain only
    bounded chunk IDs, and hydrate the winners through the ordinary provenance
    path.  Synthetic source-timestamp rows are excluded; their timestamps are
    provenance metadata rather than answer evidence.
    """

    chunk_id: str
    source_id: str
    role: str
    ordinal: int
    text: str


def load_chunk_payload(db: Database, chunk_id: str) -> Chunk | None:
    """Load a chunk without decoding its unused dense embedding."""
    row = db.execute(
        "SELECT chunk_id, turn_id, text, start_char, end_char, token_count, "
        "lexical_weights FROM chunks WHERE chunk_id = ?",
        (chunk_id,),
    ).fetchone()
    if row is None:
        return None
    lexical_weights = None if row[6] is None else json.loads(row[6])
    return Chunk(
        chunk_id=row[0],
        turn_id=row[1],
        text=row[2],
        start_char=row[3],
        end_char=row[4],
        token_count=row[5],
        embedding=None,
        lexical_weights=lexical_weights,
    )


def load_turn_payload(db: Database, turn_id: str) -> Turn | None:
    row = db.execute(
        "SELECT turn_id, role, text, source_id, created_at FROM turns WHERE turn_id = ?",
        (turn_id,),
    ).fetchone()
    if row is None:
        return None
    return Turn(
        turn_id=row[0],
        role=row[1],
        text=row[2],
        source_id=row[3],
        created_at=row[4],
    )


def hydrate_chunk_result(
    db: Database,
    chunk_id: str,
    *,
    score: float,
    dense_score: float | None = None,
    lexical_score: float | None = None,
    route: str | None = None,
    association_score: float | None = None,
    anchor_chunk_id: str | None = None,
    transition_distance: int | None = None,
    transition_direction: str | None = None,
) -> RetrievalResult | None:
    """Hydrate an external-memory ID without constructing an ANN retriever."""
    chunk = load_chunk_payload(db, chunk_id)
    if chunk is None:
        return None
    return RetrievalResult(
        chunk=chunk,
        score=score,
        turn=load_turn_payload(db, chunk.turn_id),
        dense_score=dense_score,
        lexical_score=lexical_score,
        route=route,
        association_score=association_score,
        anchor_chunk_id=anchor_chunk_id,
        transition_distance=transition_distance,
        transition_direction=transition_direction,
    )


__all__ = [
    "DEFAULT_SPAN_TOKENS",
    "PartitionContentRow",
    "hydrate_chunk_result",
    "load_chunk_payload",
    "load_turn_payload",
]
