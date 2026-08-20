"""Bounded heat diffusion over compact attention-head association edges.

The transformer is not part of this read path. QK/OV inspection has already
emitted fixed-width edge evidence into :class:`AssociationStore`; this module
turns those local conductivities into a conserved scalar distribution. Only
chunk IDs, scalar heat, and one compact explanatory path cross iterations.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Sequence

from memory_condense.associations.association_store import AssociationStore
from memory_condense.associations.associative_composition import (
    compose_associative_candidates,
)
from memory_condense.associations.associative_retrieval import expand_associative_results
from memory_condense.associations.expansion_guards import (
    exceeds_prompt_budget,
    guard_expansion_request,
    require_artifact,
)
from memory_condense.associations.head_memory_models import (
    AssociativeMemoryCandidate,
)
from memory_condense.domain.ranking import softmax, weighted_fair_order
from memory_condense.domain.schemas import RetrievalResult


Hydrate = Callable[..., RetrievalResult | None]
SourceKey = Callable[[RetrievalResult], str]


@dataclass(frozen=True, slots=True)
class DiffusedHeatNode:
    """One fixed-width entry in the final external heat distribution."""

    chunk_id: str
    heat: float
    hop: int
    best_path: tuple[str, ...]
    supporting_transitions: int


@dataclass(frozen=True, slots=True)
class HeatDiffusion:
    """Diagnostics for a hop- and candidate-capped diffusion run."""

    nodes: tuple[DiffusedHeatNode, ...]
    hops_completed: int
    discarded_heat: float

    @property
    def total_heat(self) -> float:
        return sum(node.heat for node in self.nodes)


@dataclass(slots=True)
class _MutableHeat:
    heat: float = 0.0
    hop: int = 0
    best_path: tuple[str, ...] = ()
    best_contribution: float = -1.0
    supporting_transitions: int = 0


def _seed_distribution(
    anchors: Sequence[RetrievalResult], temperature: float
) -> dict[str, float]:
    scores = [float(anchor.score) for anchor in anchors]
    weights = softmax(scores, temperature=temperature)
    return {
        anchor.chunk.chunk_id: weight
        for anchor, weight in zip(anchors, weights, strict=True)
    }


def diffuse_association_heat(
    anchors: Sequence[RetrievalResult],
    artifact_id: str,
    *,
    store: AssociationStore,
    now_turn: int,
    hops: int = 2,
    neighbors_per_node: int = 3,
    max_nodes: int = 8,
    restart_probability: float = 0.35,
    seed_temperature: float = 1.0,
    edge_temperature: float = 1.0,
) -> HeatDiffusion:
    """Diffuse a unit of heat through bounded, row-normalized QK/OV edges.

    This is a finite personalized-PageRank-style walk. Restart keeps query
    evidence present, row normalization prevents high-degree nodes from
    creating heat, and candidate trimming is followed by renormalization. A
    candidate reached from several parents receives the sum of their heat.
    """

    if hops < 0:
        raise ValueError("hops must be non-negative")
    if neighbors_per_node < 0:
        raise ValueError("neighbors_per_node must be non-negative")
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    if not 0.0 <= restart_probability <= 1.0:
        raise ValueError("restart_probability must lie in [0, 1]")
    if seed_temperature <= 0.0 or edge_temperature <= 0.0:
        raise ValueError("diffusion temperatures must be positive")
    require_artifact(store, artifact_id)

    unique_anchors: dict[str, RetrievalResult] = {}
    for anchor in anchors:
        unique_anchors.setdefault(anchor.chunk.chunk_id, anchor)
    bounded_anchors = list(unique_anchors)
    if not bounded_anchors:
        return HeatDiffusion(nodes=(), hops_completed=0, discarded_heat=0.0)
    if max_nodes < len(bounded_anchors):
        raise ValueError("max_nodes cannot be smaller than the unique anchor count")

    seeds = _seed_distribution(list(unique_anchors.values()), seed_temperature)
    current: dict[str, _MutableHeat] = {
        chunk_id: _MutableHeat(
            heat=heat,
            hop=0,
            best_path=(chunk_id,),
            best_contribution=heat,
        )
        for chunk_id, heat in seeds.items()
    }
    discarded_heat = 0.0
    completed = 0

    for step in range(1, hops + 1):
        if not current:
            break
        edge_groups = store.neighbors_many(
            list(current),
            artifact_id,
            top_k_per_source=neighbors_per_node,
            exclude_frontier=False,
            now_turn=now_turn,
        )
        following: dict[str, _MutableHeat] = {}

        def add(
            chunk_id: str,
            contribution: float,
            *,
            path: tuple[str, ...],
            hop: int,
            transition: bool,
        ) -> None:
            state = following.setdefault(chunk_id, _MutableHeat())
            state.heat += contribution
            if transition:
                state.supporting_transitions += 1
            if contribution > state.best_contribution:
                state.best_contribution = contribution
                state.best_path = path
                state.hop = hop

        for chunk_id, seed_heat in seeds.items():
            contribution = restart_probability * seed_heat
            if contribution:
                add(
                    chunk_id,
                    contribution,
                    path=(chunk_id,),
                    hop=0,
                    transition=False,
                )

        walk_fraction = 1.0 - restart_probability
        for source_id, source in current.items():
            walk_heat = walk_fraction * source.heat
            if walk_heat <= 0.0:
                continue
            edges = edge_groups.get(source_id, ())
            if not edges:
                # Dangling nodes retain their walk mass, keeping the external
                # scalar conserved without inventing a transformer state.
                add(
                    source_id,
                    walk_heat,
                    path=source.best_path,
                    hop=source.hop,
                    transition=False,
                )
                continue
            utilities = [edge.utility(now_turn=now_turn) for edge in edges]
            for edge, probability in zip(
                edges,
                softmax(utilities, temperature=edge_temperature),
                strict=True,
            ):
                contribution = walk_heat * probability
                path = source.best_path + (edge.destination_chunk_id,)
                add(
                    edge.destination_chunk_id,
                    contribution,
                    path=path,
                    hop=max(1, len(path) - 1),
                    transition=True,
                )

        if not following:
            break
        ranked = sorted(
            following.items(),
            key=lambda item: (item[1].heat, item[0]),
            reverse=True,
        )
        kept = dict(ranked[:max_nodes])
        discarded_heat += sum(state.heat for _, state in ranked[max_nodes:])
        retained = sum(state.heat for state in kept.values())
        if retained <= 0.0:
            break
        for state in kept.values():
            state.heat /= retained
            state.best_contribution /= retained
        current = kept
        completed = step

    nodes = tuple(
        DiffusedHeatNode(
            chunk_id=chunk_id,
            heat=state.heat,
            hop=state.hop,
            best_path=state.best_path,
            supporting_transitions=state.supporting_transitions,
        )
        for chunk_id, state in sorted(
            current.items(),
            key=lambda item: (item[1].heat, item[0]),
            reverse=True,
        )
    )
    return HeatDiffusion(
        nodes=nodes,
        hops_completed=completed,
        discarded_heat=discarded_heat,
    )


def _default_source_key(result: RetrievalResult) -> str:
    if result.turn is not None and result.turn.source_id:
        return result.turn.source_id
    return result.chunk.turn_id


def _source_fair_order(
    candidates: Sequence[AssociativeMemoryCandidate],
    *,
    source_heat: dict[str, float],
    total_token_budget: int,
    max_source_token_fraction: float,
) -> list[AssociativeMemoryCandidate]:
    """Weighted-fair order whose early prefix spends tokens by source heat."""

    return weighted_fair_order(
        candidates,
        source_key=lambda candidate: str(candidate.metadata["memory_source_id"]),
        source_weight=source_heat,
        item_cost=lambda candidate: int(candidate.metadata["token_count"]),
        item_priority=lambda candidate: float(candidate.metadata["diffusion_heat"]),
        total_budget=total_token_budget,
        max_source_fraction=max_source_token_fraction,
        # No cost_clip: this arm bills each candidate's full chunk token count.
    )


def expand_heat_diffusion_results(
    anchors: Sequence[RetrievalResult],
    artifact_id: str,
    *,
    store: AssociationStore,
    hydrate: Hydrate,
    now_turn: int,
    k: int | None = None,
    association_slots: int = 0,
    qk_reserve: int = 1,
    ranked_qk_reserve: int = 0,
    neighbors_per_node: int = 3,
    diffusion_hops: int = 2,
    max_diffusion_nodes: int = 8,
    restart_probability: float = 0.35,
    seed_temperature: float = 1.0,
    edge_temperature: float = 1.0,
    lexical_protection_threshold: float | None = None,
    max_prompt_token_increase: int | None = None,
    max_source_token_fraction: float = 1.0,
    source_key: SourceKey | None = None,
    touch: bool = True,
) -> list[RetrievalResult]:
    """Allocate bounded association slots using diffused per-source heat."""

    guards = guard_expansion_request(
        anchors,
        k=k,
        lexical_protection_threshold=lexical_protection_threshold,
        max_prompt_token_increase=max_prompt_token_increase,
    )
    result_cap = guards.result_cap
    if result_cap <= 0:
        return []
    if association_slots < 0 or qk_reserve < 0 or ranked_qk_reserve < 0:
        raise ValueError("association budgets must be non-negative")
    if ranked_qk_reserve > qk_reserve:
        raise ValueError("ranked_qk_reserve cannot exceed qk_reserve")
    if not 0.0 < max_source_token_fraction <= 1.0:
        raise ValueError("max_source_token_fraction must lie in (0, 1]")

    bounded_anchors = guards.bounded_anchors
    if not bounded_anchors:
        return []
    diffusion = diffuse_association_heat(
        bounded_anchors,
        artifact_id,
        store=store,
        now_turn=now_turn,
        hops=diffusion_hops,
        neighbors_per_node=neighbors_per_node,
        max_nodes=max_diffusion_nodes,
        restart_probability=restart_probability,
        seed_temperature=seed_temperature,
        edge_temperature=edge_temperature,
    )
    nodes = {node.chunk_id: node for node in diffusion.nodes}
    anchor_ids = {anchor.chunk.chunk_id for anchor in bounded_anchors}
    hydration_cache: dict[str, RetrievalResult | None] = {
        anchor.chunk.chunk_id: anchor for anchor in bounded_anchors
    }

    def hydrated(chunk_id: str) -> RetrievalResult | None:
        if chunk_id not in hydration_cache:
            hydration_cache[chunk_id] = hydrate(chunk_id, score=0.0)
        return hydration_cache[chunk_id]

    resolver = source_key or _default_source_key
    source_by_chunk: dict[str, str] = {}
    source_heat: dict[str, float] = defaultdict(float)
    for node in diffusion.nodes:
        result = hydrated(node.chunk_id)
        if result is None:
            continue
        source_id = str(resolver(result))
        source_by_chunk[node.chunk_id] = source_id
        source_heat[source_id] += node.heat

    direct_tokens = sum(anchor.chunk.token_count for anchor in bounded_anchors)
    total_source_heat = sum(source_heat.values()) or 1.0

    def candidate_metadata(
        *,
        anchor_chunk_id: str | None,
        edge_source_id: str | None,
        association_hop: int | None,
        association_path: tuple[str, ...] | None,
        diffusion_heat: float,
        association_support: int,
        source_id: str,
        token_count: int,
    ) -> dict[str, object]:
        """The packed-candidate metadata shape shared by both routes."""
        return {
            "anchor_chunk_id": anchor_chunk_id,
            "edge_source_id": edge_source_id,
            "association_hop": association_hop,
            "association_path": association_path,
            "diffusion_heat": diffusion_heat,
            "association_support": association_support,
            "memory_source_id": source_id,
            "source_heat": source_heat[source_id],
            "source_token_budget": round(
                direct_tokens * source_heat[source_id] / total_source_heat
            ),
            "token_count": token_count,
        }

    heat_candidates: list[AssociativeMemoryCandidate] = []
    for node in diffusion.nodes:
        if node.chunk_id in anchor_ids:
            continue
        result = hydrated(node.chunk_id)
        source_id = source_by_chunk.get(node.chunk_id)
        if result is None or source_id is None:
            continue
        heat_candidates.append(
            AssociativeMemoryCandidate(
                episode_id=node.chunk_id,
                text=result.chunk.text,
                score=node.heat,
                route="heat",
                metadata=candidate_metadata(
                    anchor_chunk_id=node.best_path[0],
                    edge_source_id=(
                        node.best_path[-2] if len(node.best_path) > 1 else None
                    ),
                    association_hop=max(1, node.hop),
                    association_path=node.best_path,
                    diffusion_heat=node.heat,
                    association_support=node.supporting_transitions,
                    source_id=source_id,
                    token_count=result.chunk.token_count,
                ),
            )
        )
    heat_candidates.sort(
        key=lambda candidate: (
            float(candidate.metadata["diffusion_heat"])
            / max(1, int(candidate.metadata["token_count"])),
            float(candidate.metadata["diffusion_heat"]),
            candidate.episode_id,
        ),
        reverse=True,
    )
    heat_candidates = _source_fair_order(
        heat_candidates,
        source_heat=source_heat,
        total_token_budget=max(1, direct_tokens),
        max_source_token_fraction=max_source_token_fraction,
    )

    # Diffusion rewards corroboration and token-efficient source exposure, but
    # a rare decisive edge can have little global heat. Optionally preserve a
    # small max-path exploitation channel before heat spends the remaining
    # association slots. This is still model-free and uses the same compact
    # persisted edges.
    ranked_candidates: list[AssociativeMemoryCandidate] = []
    if ranked_qk_reserve:
        ranked_results = expand_associative_results(
            bounded_anchors,
            artifact_id,
            store=store,
            hydrate=hydrate,
            now_turn=now_turn,
            k=result_cap,
            association_slots=min(association_slots, ranked_qk_reserve),
            qk_reserve=ranked_qk_reserve,
            neighbors_per_anchor=neighbors_per_node,
            association_hops=diffusion_hops,
            max_association_candidates=max_diffusion_nodes,
            cav_candidates=0,
            lexical_protection_threshold=lexical_protection_threshold,
            max_prompt_token_increase=max_prompt_token_increase,
            touch=False,
        )
        for result in ranked_results:
            if result.route != "qk":
                continue
            source_id = source_by_chunk.get(result.chunk.chunk_id)
            if source_id is None:
                source_id = str(resolver(result))
                source_by_chunk[result.chunk.chunk_id] = source_id
                source_heat.setdefault(source_id, 0.0)
            node = nodes.get(result.chunk.chunk_id)
            ranked_candidates.append(
                AssociativeMemoryCandidate(
                    episode_id=result.chunk.chunk_id,
                    text=result.chunk.text,
                    score=result.score,
                    route="qk",
                    metadata=candidate_metadata(
                        anchor_chunk_id=result.anchor_chunk_id,
                        edge_source_id=result.edge_source_chunk_id,
                        association_hop=result.association_hop,
                        association_path=result.association_path,
                        diffusion_heat=0.0 if node is None else node.heat,
                        association_support=(
                            0 if node is None else node.supporting_transitions
                        ),
                        source_id=source_id,
                        token_count=result.chunk.token_count,
                    ),
                )
            )
    ranked_ids = {candidate.episode_id for candidate in ranked_candidates}
    heat_candidates = [
        candidate
        for candidate in heat_candidates
        if candidate.episode_id not in ranked_ids
    ]

    anchor_candidates = [
        AssociativeMemoryCandidate(
            episode_id=anchor.chunk.chunk_id,
            text=anchor.chunk.text,
            score=anchor.score,
            route="hybrid",
        )
        for anchor in bounded_anchors
    ]
    protected = (
        ()
        if lexical_protection_threshold is None
        else tuple(
            anchor.chunk.chunk_id
            for anchor in bounded_anchors
            if anchor.lexical_score is not None
            and anchor.lexical_score >= lexical_protection_threshold
        )
    )

    # If the hottest association is too expensive, try the remaining stream
    # rather than rolling the entire heat arm back immediately.
    remaining = list(heat_candidates)
    composition = None
    results: list[RetrievalResult] = []
    while True:
        composition = compose_associative_candidates(
            anchor_candidates,
            qk_neighbors=[*ranked_candidates, *remaining],
            top_k=result_cap,
            qk_reserve=qk_reserve,
            association_slots=association_slots,
            protected_anchor_ids=protected,
        )
        direct = {anchor.chunk.chunk_id: anchor for anchor in bounded_anchors}
        results = []
        for candidate in composition.candidates:
            base = direct.get(candidate.episode_id) or hydrated(candidate.episode_id)
            if base is None:
                continue
            node = nodes.get(candidate.episode_id)
            source_id = source_by_chunk.get(candidate.episode_id)
            update = {
                "diffusion_heat": None if node is None else node.heat,
                "association_support": (
                    None if node is None else node.supporting_transitions
                ),
                "memory_source_id": source_id,
                "source_heat": (
                    None if source_id is None else source_heat[source_id]
                ),
                "source_token_budget": (
                    None
                    if source_id is None
                    else round(
                        direct_tokens * source_heat[source_id] / total_source_heat
                    )
                ),
            }
            if candidate.route in {"qk", "heat"}:
                update.update(
                    {
                        "score": candidate.score,
                        "route": candidate.route,
                        "association_score": candidate.score,
                        "anchor_chunk_id": candidate.metadata["anchor_chunk_id"],
                        "association_hop": candidate.metadata["association_hop"],
                        "edge_source_chunk_id": candidate.metadata["edge_source_id"],
                        "association_path": candidate.metadata["association_path"],
                    }
                )
            results.append(base.model_copy(update=update))

        if not exceeds_prompt_budget(
            results,
            # Denominator: like hebbian, this arm spends against all
            # bounded_anchors; associative windows to [:result_cap].
            direct_anchors=bounded_anchors,
            max_prompt_token_increase=max_prompt_token_increase,
        ):
            break
        selected_heat = {
            candidate.episode_id
            for candidate in composition.candidates
            if candidate.route == "heat"
        }
        if not selected_heat:
            return bounded_anchors
        worst = min(
            (candidate for candidate in remaining if candidate.episode_id in selected_heat),
            key=lambda candidate: (
                float(candidate.metadata["diffusion_heat"])
                / max(1, int(candidate.metadata["token_count"])),
                candidate.episode_id,
            ),
        )
        remaining = [
            candidate
            for candidate in remaining
            if candidate.episode_id != worst.episode_id
        ]
        if not remaining:
            return bounded_anchors

    if touch and composition is not None:
        edge_pairs: list[tuple[str, str]] = []
        for result in results:
            if result.route not in {"qk", "heat"} or result.association_path is None:
                continue
            edge_pairs.extend(zip(result.association_path, result.association_path[1:]))
        if edge_pairs:
            store.touch_edges(artifact_id, edge_pairs, now_turn=now_turn)
    return results[:result_cap]
