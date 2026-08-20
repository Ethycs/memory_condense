"""Deterministic representative selection without retaining feature vectors."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from memory_condense.domain.discourse import (
    Episode,
    EpisodeRepresentative,
    evidence_span_sort_key,
    identity_sha256,
)

from .surprise import dense_cosine, lexical_cosine


def select_episode_representatives(
    episode: Episode,
    *,
    limit: int = 2,
    texts: Mapping[str, str] | None = None,
    embeddings: Mapping[str, Sequence[float]] | None = None,
) -> tuple[EpisodeRepresentative, ...]:
    """Select central episode chunks with exact, stable tie-breaks.

    Centrality is the mean pairwise lexical/embedding similarity to all other
    distinct chunks in the episode.  Ties break by source order and chunk ID.
    Only an identity hash of each selected feature survives the call.
    """
    if int(limit) < 0:
        raise ValueError("representative limit must be non-negative")
    if limit == 0:
        return ()

    source_order_by_chunk: dict[
        str,
        tuple[int, str, int, int, int, str, str],
    ] = {}
    for span in episode.evidence:
        order = evidence_span_sort_key(span)
        current = source_order_by_chunk.get(span.chunk_id)
        if current is None or order < current:
            source_order_by_chunk[span.chunk_id] = order
    chunk_ids = tuple(
        sorted(source_order_by_chunk, key=source_order_by_chunk.__getitem__)
    )
    normalized_vectors = _normalize_vectors(embeddings or {}, chunk_ids)
    scores: dict[str, float] = {}
    for chunk_id in chunk_ids:
        similarities: list[float] = []
        for other_id in chunk_ids:
            if other_id == chunk_id:
                continue
            similarities.append(
                _feature_similarity(
                    chunk_id,
                    other_id,
                    texts=texts,
                    embeddings=normalized_vectors,
                )
            )
        scores[chunk_id] = (
            math.fsum(similarities) / len(similarities) if similarities else 1.0
        )

    ranked = sorted(
        chunk_ids,
        key=lambda item: (-scores[item], *source_order_by_chunk[item]),
    )[: int(limit)]
    representatives: list[EpisodeRepresentative] = []
    for rank, chunk_id in enumerate(ranked):
        vector = normalized_vectors.get(chunk_id)
        if vector is not None:
            feature_identity = {
                "method": "ordinary_embedding",
                "chunk_id": chunk_id,
                "vector": [0.0 if value == 0.0 else value for value in vector],
            }
        elif texts is not None and chunk_id in texts:
            feature_identity = {
                "method": "lexical_control",
                "chunk_id": chunk_id,
                "text_sha256": identity_sha256(str(texts[chunk_id])),
            }
        else:
            feature_identity = {
                "method": "source_order_control",
                "chunk_id": chunk_id,
                "ordinal": source_order_by_chunk[chunk_id][0],
                "turn_start_char": source_order_by_chunk[chunk_id][2],
            }
        representatives.append(
            EpisodeRepresentative(
                episode_id=episode.episode_id,
                chunk_id=chunk_id,
                rank=rank,
                vector_identity_sha256=identity_sha256(feature_identity),
            )
        )
    return tuple(representatives)


def _feature_similarity(
    left_id: str,
    right_id: str,
    *,
    texts: Mapping[str, str] | None,
    embeddings: Mapping[str, tuple[float, ...]],
) -> float:
    signals: list[float] = []
    if texts is not None and left_id in texts and right_id in texts:
        signals.append(lexical_cosine(str(texts[left_id]), str(texts[right_id])))
    if left_id in embeddings and right_id in embeddings:
        signals.append((dense_cosine(embeddings[left_id], embeddings[right_id]) + 1.0) / 2.0)
    return math.fsum(signals) / len(signals) if signals else 0.0


def _normalize_vectors(
    embeddings: Mapping[str, Sequence[float]],
    chunk_ids: Sequence[str],
) -> dict[str, tuple[float, ...]]:
    normalized: dict[str, tuple[float, ...]] = {}
    dimension: int | None = None
    for chunk_id in chunk_ids:
        if chunk_id not in embeddings:
            continue
        vector = tuple(float(value) for value in embeddings[chunk_id])
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("representative embeddings must be finite and non-empty")
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise ValueError("representative embeddings must share one dimension")
        normalized[chunk_id] = vector
    return normalized


__all__ = ["select_episode_representatives"]
