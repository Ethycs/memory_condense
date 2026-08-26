"""Budgeted composition of direct and associative candidates."""

from __future__ import annotations

import re
from typing import Callable, Mapping, Sequence

from memory_condense.associations.head_memory_models import (
    AssociativeComposition,
    AssociativeMemoryCandidate,
)
from memory_condense.domain.schemas import RetrievalResult


def anchor_candidates(
    anchors: Sequence[RetrievalResult],
) -> list[AssociativeMemoryCandidate]:
    """Pack ranked direct anchors as ``hybrid``-route composition candidates."""

    return [
        AssociativeMemoryCandidate(
            episode_id=result.chunk.chunk_id,
            text=result.chunk.text,
            score=result.score,
            route="hybrid",
        )
        for result in anchors
    ]


def hydration_memo(
    hydrate: Callable[..., RetrievalResult | None],
    *,
    seed: Mapping[str, RetrievalResult] | None = None,
) -> Callable[[str], RetrievalResult | None]:
    """Memoize base hydration so one expansion never re-reads a chunk.

    Misses are cached too: a chunk that no longer hydrates stays absent for
    the rest of the call instead of being retried by every route.
    """

    cache: dict[str, RetrievalResult | None] = dict(seed or {})

    def hydrated(chunk_id: str) -> RetrievalResult | None:
        if chunk_id not in cache:
            cache[chunk_id] = hydrate(chunk_id, score=0.0)
        return cache[chunk_id]

    return hydrated


def compose_associative_candidates(
    anchors: Sequence[AssociativeMemoryCandidate],
    *,
    qk_neighbors: Sequence[AssociativeMemoryCandidate] = (),
    residual_candidates: Sequence[AssociativeMemoryCandidate] = (),
    top_k: int = 10,
    qk_reserve: int = 1,
    association_slots: int = 0,
    protected_anchor_ids: Sequence[str] = (),
) -> AssociativeComposition:
    """Keep every unique direct anchor and recycle only redundant slots.

    QK neighbors get first use of the recycled capacity, followed by residual
    candidates. Neither association route can displace unique direct evidence.
    """
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if qk_reserve < 0:
        raise ValueError("qk_reserve must be non-negative")
    if association_slots < 0:
        raise ValueError("association_slots must be non-negative")
    protected = set(protected_anchor_ids)

    unique_anchors: list[AssociativeMemoryCandidate] = []
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    duplicates_removed = 0

    def content_key(candidate: AssociativeMemoryCandidate) -> str:
        return re.sub(r"\s+", " ", candidate.text).strip().casefold()

    for candidate in anchors[:top_k]:
        content = content_key(candidate)
        if candidate.episode_id in seen_ids or content in seen_content:
            duplicates_removed += 1
            continue
        seen_ids.add(candidate.episode_id)
        seen_content.add(content)
        unique_anchors.append(candidate)

    # Only a contiguous unprotected suffix may be displaced. If the weakest
    # anchor carries protected direct evidence, moving the reservation upward
    # would perversely replace an even stronger anchor instead.
    reserved = 0
    for candidate in reversed(unique_anchors):
        if reserved >= association_slots or candidate.episode_id in protected:
            break
        reserved += 1
    selected = list(unique_anchors[: len(unique_anchors) - reserved])
    held_anchors = unique_anchors[len(unique_anchors) - reserved :]
    seen_ids = {candidate.episode_id for candidate in selected}
    seen_content = {content_key(candidate) for candidate in selected}

    def add(candidate: AssociativeMemoryCandidate) -> bool:
        content = content_key(candidate)
        if candidate.episode_id in seen_ids or content in seen_content:
            return False
        seen_ids.add(candidate.episode_id)
        seen_content.add(content)
        selected.append(candidate)
        return True

    # Association routes may consume only slots freed by duplicate direct
    # results unless the caller explicitly reserves a fixed number of slots.
    capacity = min(duplicates_removed + reserved, top_k - len(selected))
    qk_added = 0
    for candidate in qk_neighbors:
        if qk_added >= min(qk_reserve, capacity):
            break
        if add(candidate):
            qk_added += 1

    residual_added = 0
    residual_capacity = capacity - qk_added
    for candidate in residual_candidates:
        if residual_added >= residual_capacity:
            break
        if add(candidate):
            residual_added += 1

    # If association routes cannot use every reserved slot, restore the held
    # direct anchors before returning a short result.
    for candidate in held_anchors:
        if len(selected) >= top_k:
            break
        add(candidate)

    selected_ids = {candidate.episode_id for candidate in selected}
    anchors_displaced = sum(
        candidate.episode_id not in selected_ids for candidate in unique_anchors
    )

    return AssociativeComposition(
        candidates=tuple(selected[:top_k]),
        duplicates_removed=duplicates_removed,
        qk_added=qk_added,
        residual_added=residual_added,
        anchors_displaced=anchors_displaced,
    )
