"""Model-free expansion of cached retrieval anchors through compact links."""

from __future__ import annotations

from typing import Callable, Sequence

from memory_condense.associations.association_store import AssociationStore
from memory_condense.associations.associative_composition import (
    compose_associative_candidates,
)
from memory_condense.associations.expansion_guards import (
    exceeds_prompt_budget,
    guard_expansion_request,
    require_artifact,
)
from memory_condense.associations.head_memory_models import (
    AssociativeMemoryCandidate,
)
from memory_condense.domain.schemas import RetrievalResult


Hydrate = Callable[..., RetrievalResult | None]


def expand_associative_results(
    anchors: Sequence[RetrievalResult],
    artifact_id: str,
    *,
    store: AssociationStore,
    hydrate: Hydrate,
    now_turn: int,
    k: int | None = None,
    association_slots: int = 0,
    qk_reserve: int = 1,
    neighbors_per_anchor: int = 4,
    association_hops: int = 1,
    max_association_candidates: int = 64,
    cav_candidates: int = 8,
    lexical_protection_threshold: float | None = None,
    max_prompt_token_increase: int | None = None,
    touch: bool = True,
) -> list[RetrievalResult]:
    """Expand already-ranked anchors without invoking an embedding or LLM."""
    guards = guard_expansion_request(
        anchors,
        k=k,
        lexical_protection_threshold=lexical_protection_threshold,
        max_prompt_token_increase=max_prompt_token_increase,
    )
    result_cap = guards.result_cap
    if result_cap <= 0:
        return []
    if neighbors_per_anchor < 0:
        raise ValueError("neighbors_per_anchor must be non-negative")
    if association_hops < 1:
        raise ValueError("association_hops must be positive")
    if max_association_candidates < 1:
        raise ValueError("max_association_candidates must be positive")
    if cav_candidates < 0:
        raise ValueError("cav_candidates must be non-negative")
    require_artifact(store, artifact_id)
    bounded_anchors = guards.bounded_anchors
    if not bounded_anchors:
        return []

    anchor_ids = [result.chunk.chunk_id for result in bounded_anchors]
    anchor_id_set = set(anchor_ids)
    anchor_candidates = [
        AssociativeMemoryCandidate(
            episode_id=result.chunk.chunk_id,
            text=result.chunk.text,
            score=result.score,
            route="hybrid",
        )
        for result in bounded_anchors
    ]
    hydration_cache: dict[str, RetrievalResult | None] = {}

    def hydrate_base(chunk_id: str) -> RetrievalResult | None:
        if chunk_id not in hydration_cache:
            hydration_cache[chunk_id] = hydrate(chunk_id, score=0.0)
        return hydration_cache[chunk_id]

    # Recursive traversal carries only external IDs and scalar evidence. The
    # retained pool is trimmed after every hop, so neither graph state nor a
    # later transformer workspace grows with corpus size.
    frontier: dict[str, dict] = {
        anchor.chunk.chunk_id: {
            "score": anchor.score,
            "anchor_chunk_id": anchor.chunk.chunk_id,
            "association_path": (anchor.chunk.chunk_id,),
        }
        for anchor in bounded_anchors
    }
    seen = set(anchor_ids)
    qk_states: dict[str, dict] = {}
    for hop in range(1, association_hops + 1):
        if not frontier or neighbors_per_anchor == 0:
            break
        edge_groups = store.neighbors_many(
            list(frontier),
            artifact_id,
            top_k_per_source=neighbors_per_anchor,
            exclude=anchor_ids,
            now_turn=now_turn,
        )
        next_states: dict[str, dict] = {}
        for parent_id, parent in frontier.items():
            for edge in edge_groups[parent_id]:
                destination_id = edge.destination_chunk_id
                if destination_id in anchor_id_set or destination_id in seen:
                    continue
                association_score = edge.utility(now_turn=now_turn) / hop
                state = {
                    "episode_id": destination_id,
                    "score": parent["score"] + association_score,
                    "anchor_chunk_id": parent["anchor_chunk_id"],
                    "edge_source_id": parent_id,
                    "association_score": association_score,
                    "qk_score": edge.qk_score,
                    "ov_transport": edge.ov_transport,
                    "association_hop": hop,
                    "association_path": parent["association_path"]
                    + (destination_id,),
                }
                current = qk_states.get(destination_id)
                if current is None or state["score"] > current["score"]:
                    qk_states[destination_id] = state
                    next_states[destination_id] = state
        seen.update(next_states)
        ranked_states = sorted(
            qk_states.values(),
            key=lambda state: (state["score"], state["episode_id"]),
            reverse=True,
        )[:max_association_candidates]
        qk_states = {state["episode_id"]: state for state in ranked_states}
        frontier = {
            state["episode_id"]: state
            for state in ranked_states
            if state["episode_id"] in next_states
        }

    qk_neighbors: list[AssociativeMemoryCandidate] = []
    for state in sorted(
        qk_states.values(),
        key=lambda item: (item["score"], item["episode_id"]),
        reverse=True,
    ):
        hydrated = hydrate_base(state["episode_id"])
        if hydrated is None:
            continue
        qk_neighbors.append(
            AssociativeMemoryCandidate(
                episode_id=state["episode_id"],
                text=hydrated.chunk.text,
                score=state["score"],
                route="qk",
                metadata={
                    key: value
                    for key, value in state.items()
                    if key not in {"episode_id", "score"}
                },
            )
        )
    qk_ids = {candidate.episode_id for candidate in qk_neighbors}

    cav_neighbors: list[AssociativeMemoryCandidate] = []
    for hit in store.cav_neighbors(
        anchor_ids,
        artifact_id,
        top_k=cav_candidates,
        exclude=tuple(anchor_id_set | qk_ids),
    ):
        hydrated = hydrate_base(hit.chunk_id)
        if hydrated is None:
            continue
        cav_neighbors.append(
            AssociativeMemoryCandidate(
                episode_id=hit.chunk_id,
                text=hydrated.chunk.text,
                score=hit.score,
                route="cav",
                metadata={
                    "association_score": hit.score,
                    "shared_concepts": hit.shared_concepts,
                },
            )
        )

    composition = compose_associative_candidates(
        anchor_candidates,
        qk_neighbors=qk_neighbors,
        residual_candidates=cav_neighbors,
        top_k=result_cap,
        qk_reserve=qk_reserve,
        association_slots=association_slots,
        protected_anchor_ids=(
            ()
            if lexical_protection_threshold is None
            else tuple(
                result.chunk.chunk_id
                for result in bounded_anchors
                if result.lexical_score is not None
                and result.lexical_score >= lexical_protection_threshold
            )
        ),
    )
    direct = {result.chunk.chunk_id: result for result in bounded_anchors}
    results: list[RetrievalResult] = []
    used_edge_pairs: list[tuple[str, str]] = []
    used_cav_ids: list[str] = []
    for candidate in composition.candidates:
        if candidate.route == "hybrid" and candidate.episode_id in direct:
            results.append(direct[candidate.episode_id])
            continue
        metadata = candidate.metadata
        anchor_chunk_id = metadata.get("anchor_chunk_id")
        base = hydrate_base(candidate.episode_id)
        if base is None:
            continue
        result = base.model_copy(
            update={
                "score": candidate.score,
                "route": candidate.route,
                "association_score": metadata.get("association_score"),
                "anchor_chunk_id": anchor_chunk_id,
                "association_hop": metadata.get("association_hop"),
                "edge_source_chunk_id": metadata.get("edge_source_id"),
                "association_path": metadata.get("association_path"),
            }
        )
        results.append(result)
        if candidate.route == "qk" and anchor_chunk_id is not None:
            used_edge_pairs.append(
                (metadata.get("edge_source_id", anchor_chunk_id), candidate.episode_id)
            )
        elif candidate.route == "cav":
            used_cav_ids.append(candidate.episode_id)

    if exceeds_prompt_budget(
        results,
        # Denominator: this arm windows to bounded_anchors[:result_cap] while
        # hebbian spends against all bounded_anchors (open author decision).
        direct_anchors=bounded_anchors[:result_cap],
        max_prompt_token_increase=max_prompt_token_increase,
    ):
        # Admission is decided before touches, so a rejected exploration
        # cannot reinforce the very edge that exceeded the prompt budget.
        return bounded_anchors[:result_cap]

    if touch:
        if used_edge_pairs:
            store.touch_edges(
                artifact_id,
                used_edge_pairs,
                now_turn=now_turn,
            )
        if used_cav_ids:
            store.touch_signatures(
                artifact_id,
                used_cav_ids,
                now_turn=now_turn,
            )
    return results[:result_cap]
