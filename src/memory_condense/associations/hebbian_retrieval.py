"""Bounded retrieval through a live same-turn Hebbian access graph.

The graph's nodes are source-grounded conceptual chunks.  It persists only
chunk IDs and scalar co-access statistics; query text, token IDs, attention
maps, residuals, and K/V state never enter this layer.
"""

from __future__ import annotations

from typing import Callable, Sequence

from memory_condense.associations.association_store import AssociationStore
from memory_condense.associations.coaccess_graph import rank_discount
from memory_condense.associations.expansion_guards import (
    exceeds_prompt_budget,
    guard_expansion_request,
    require_artifact,
)
from memory_condense.domain.schemas import RetrievalResult


Hydrate = Callable[..., RetrievalResult | None]


def retrieval_concept_activations(
    results: Sequence[RetrievalResult],
    *,
    max_concepts: int = 12,
) -> dict[str, float]:
    """Convert one ranked, exposed result set into bounded concept activity."""
    if max_concepts < 1:
        raise ValueError("max_concepts must be positive")
    activations: dict[str, float] = {}
    for rank, result in enumerate(results, start=1):
        chunk_id = result.chunk.chunk_id
        if chunk_id in activations:
            continue
        # Rank discount is stable across retrievers whose raw score scales are
        # not comparable. Every exposed concept remains active, but an early
        # result contributes more to the learned association.
        activations[chunk_id] = rank_discount(rank)
        if len(activations) >= max_concepts:
            break
    return activations


def expand_hebbian_results(
    anchors: Sequence[RetrievalResult],
    artifact_id: str,
    *,
    store: AssociationStore,
    hydrate: Hydrate,
    now_turn: int,
    k: int | None = None,
    hebbian_slots: int = 1,
    max_candidates: int = 32,
    half_life_turns: float = 200.0,
    min_score: float = 0.05,
    lexical_protection_threshold: float | None = None,
    max_prompt_token_increase: int | None = None,
) -> list[RetrievalResult]:
    """Replace reserved tail slots with learned co-access neighbors.

    Result count never exceeds ``k``. With ``max_prompt_token_increase=0`` the
    learned arm is also rolled back if its replacement chunks contain more
    tokens than the direct retrieval it would displace.
    """
    guards = guard_expansion_request(
        anchors,
        k=k,
        lexical_protection_threshold=lexical_protection_threshold,
        max_prompt_token_increase=max_prompt_token_increase,
    )
    result_cap = guards.result_cap
    if result_cap <= 0:
        return []
    if hebbian_slots < 0:
        raise ValueError("hebbian_slots must be non-negative")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must lie in [0, 1]")
    require_artifact(store, artifact_id)

    bounded_anchors = guards.bounded_anchors
    if not bounded_anchors or hebbian_slots == 0:
        return bounded_anchors
    activations = retrieval_concept_activations(
        bounded_anchors,
        max_concepts=len(bounded_anchors),
    )
    neighbors = store.hebbian_neighbors(
        activations,
        artifact_id,
        top_k=max_candidates,
        exclude=tuple(activations),
        now_turn=now_turn,
        half_life_turns=half_life_turns,
        min_score=min_score,
    )
    if not neighbors:
        return bounded_anchors

    replaceable: list[int] = []
    for index in range(len(bounded_anchors) - 1, -1, -1):
        anchor = bounded_anchors[index]
        protected = (
            lexical_protection_threshold is not None
            and anchor.lexical_score is not None
            and anchor.lexical_score >= lexical_protection_threshold
        )
        if not protected:
            replaceable.append(index)
        if len(replaceable) >= hebbian_slots:
            break
    slot_count = min(len(replaceable), len(neighbors))
    if slot_count == 0:
        return bounded_anchors

    learned: list[RetrievalResult] = []
    for neighbor in neighbors:
        base = hydrate(neighbor.chunk_id, score=0.0)
        if base is None:
            continue
        learned.append(
            base.model_copy(
                update={
                    "score": neighbor.score,
                    "route": "hebbian_coaccess",
                    "association_score": neighbor.score,
                    "anchor_chunk_id": neighbor.anchor_chunk_id,
                    "association_hop": 1,
                    "edge_source_chunk_id": neighbor.anchor_chunk_id,
                    "association_path": (
                        neighbor.anchor_chunk_id,
                        neighbor.chunk_id,
                    ),
                    "association_support": neighbor.support,
                }
            )
        )
        if len(learned) >= slot_count:
            break
    if not learned:
        return bounded_anchors

    removed = set(replaceable[: len(learned)])
    composed = [
        result for index, result in enumerate(bounded_anchors) if index not in removed
    ] + learned
    if exceeds_prompt_budget(
        composed,
        # Denominator: this arm spends against all bounded_anchors while
        # associative windows to [:result_cap] (open author decision).
        direct_anchors=bounded_anchors,
        max_prompt_token_increase=max_prompt_token_increase,
    ):
        return bounded_anchors
    return composed[:result_cap]
