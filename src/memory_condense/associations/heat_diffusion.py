"""Bounded heat diffusion over compact attention-head association edges.

The transformer is not part of this read path. QK/OV inspection has already
emitted fixed-width edge evidence into :class:`AssociationStore`; this module
turns those local conductivities into a conserved scalar distribution. Only
chunk IDs, scalar heat, and one compact explanatory path cross iterations.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from memory_condense.associations.association_store import (
    AssociationStore,
    StoredHeadEdge,
)
from memory_condense.associations.associative_composition import (
    anchor_candidates,
    compose_associative_candidates,
    hydration_memo,
)
from memory_condense.associations.associative_retrieval import expand_associative_results
from memory_condense.associations.expansion_guards import (
    exceeds_prompt_budget,
    guard_expansion_request,
    protected_anchor_ids,
    require_artifact,
)
from memory_condense.associations.head_memory_models import (
    AssociativeComposition,
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
    """The final external heat distribution of one capped diffusion run."""

    nodes: tuple[DiffusedHeatNode, ...]

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


@dataclass(slots=True)
class _HeatAccumulator:
    """The next frontier under construction, keyed by chunk id.

    Every seeding strategy of one hop pours into the same accumulator, so a
    node reached by several routes ends up holding the sum of their heat while
    the single largest contribution owns the explanatory path it reports.
    """

    states: dict[str, _MutableHeat] = field(default_factory=dict)

    def add(
        self,
        chunk_id: str,
        contribution: float,
        *,
        path: tuple[str, ...],
        hop: int,
        transition: bool,
    ) -> None:
        state = self.states.setdefault(chunk_id, _MutableHeat())
        state.heat += contribution
        if transition:
            state.supporting_transitions += 1
        if contribution > state.best_contribution:
            state.best_contribution = contribution
            state.best_path = path
            state.hop = hop


def _validate_diffusion_knobs(
    *,
    hops: int,
    neighbors_per_node: int,
    max_nodes: int,
    restart_probability: float,
    seed_temperature: float,
    edge_temperature: float,
) -> None:
    """Reject knob combinations that could not conserve a unit of heat."""

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


def _seed_distribution(
    anchors: Sequence[RetrievalResult], temperature: float
) -> dict[str, float]:
    scores = [float(anchor.score) for anchor in anchors]
    weights = softmax(scores, temperature=temperature)
    return {
        anchor.chunk.chunk_id: weight
        for anchor, weight in zip(anchors, weights, strict=True)
    }


def _seed_restart_mass(
    accumulator: _HeatAccumulator,
    *,
    seeds: Mapping[str, float],
    restart_probability: float,
) -> None:
    """Return the restart share of each hop to the query's own anchors.

    Restart is what keeps the walk personalized: without it a long chain of
    edges would eventually carry the whole unit of heat away from the query.
    """

    for chunk_id, seed_heat in seeds.items():
        contribution = restart_probability * seed_heat
        if contribution:
            accumulator.add(
                chunk_id,
                contribution,
                path=(chunk_id,),
                hop=0,
                transition=False,
            )


def _carry_dangling_mass(
    accumulator: _HeatAccumulator,
    *,
    source_id: str,
    source: _MutableHeat,
    walk_heat: float,
) -> None:
    """Hold an edge-less node's walk mass on the node itself.

    Dangling nodes retain their walk mass, keeping the external scalar
    conserved without inventing a transformer state to leak it into.
    """

    accumulator.add(
        source_id,
        walk_heat,
        path=source.best_path,
        hop=source.hop,
        transition=False,
    )


def _spread_transition_mass(
    accumulator: _HeatAccumulator,
    *,
    source: _MutableHeat,
    edges: Sequence[StoredHeadEdge],
    walk_heat: float,
    now_turn: int,
    edge_temperature: float,
) -> None:
    """Split one node's walk mass across its outgoing edges by utility.

    Row normalization is the point: a high-degree node divides the heat it was
    given rather than manufacturing more for each edge it happens to own.
    """

    utilities = [edge.utility(now_turn=now_turn) for edge in edges]
    for edge, probability in zip(
        edges,
        softmax(utilities, temperature=edge_temperature),
        strict=True,
    ):
        contribution = walk_heat * probability
        path = source.best_path + (edge.destination_chunk_id,)
        accumulator.add(
            edge.destination_chunk_id,
            contribution,
            path=path,
            hop=max(1, len(path) - 1),
            transition=True,
        )


def _walk_frontier_mass(
    accumulator: _HeatAccumulator,
    *,
    current: Mapping[str, _MutableHeat],
    edge_groups: Mapping[str, Sequence[StoredHeadEdge]],
    walk_fraction: float,
    now_turn: int,
    edge_temperature: float,
) -> None:
    """Move every frontier node's non-restart mass one hop outward.

    A node either has persisted edges to spend its mass on, or it is dangling
    and keeps it; nothing evaporates in between.
    """

    for source_id, source in current.items():
        walk_heat = walk_fraction * source.heat
        if walk_heat <= 0.0:
            continue
        edges = edge_groups.get(source_id, ())
        if not edges:
            _carry_dangling_mass(
                accumulator,
                source_id=source_id,
                source=source,
                walk_heat=walk_heat,
            )
            continue
        _spread_transition_mass(
            accumulator,
            source=source,
            edges=edges,
            walk_heat=walk_heat,
            now_turn=now_turn,
            edge_temperature=edge_temperature,
        )


def _trim_to_capacity(
    states: Mapping[str, _MutableHeat], *, max_nodes: int
) -> dict[str, _MutableHeat] | None:
    """Keep the hottest nodes and renormalize them back to one unit of heat.

    ``None`` means nothing survived with positive heat, which ends the walk
    rather than propagating a degenerate distribution.
    """

    ranked = sorted(
        states.items(),
        key=lambda item: (item[1].heat, item[0]),
        reverse=True,
    )
    kept = dict(ranked[:max_nodes])
    retained = sum(state.heat for state in kept.values())
    if retained <= 0.0:
        return None
    for state in kept.values():
        state.heat /= retained
        state.best_contribution /= retained
    return kept


def _emit_ranked_nodes(
    current: Mapping[str, _MutableHeat],
) -> tuple[DiffusedHeatNode, ...]:
    """Freeze the surviving frontier into the hottest-first external result."""

    return tuple(
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

    _validate_diffusion_knobs(
        hops=hops,
        neighbors_per_node=neighbors_per_node,
        max_nodes=max_nodes,
        restart_probability=restart_probability,
        seed_temperature=seed_temperature,
        edge_temperature=edge_temperature,
    )
    require_artifact(store, artifact_id)

    unique_anchors: dict[str, RetrievalResult] = {}
    for anchor in anchors:
        unique_anchors.setdefault(anchor.chunk.chunk_id, anchor)
    bounded_anchors = list(unique_anchors)
    if not bounded_anchors:
        return HeatDiffusion(nodes=())
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
    for _step in range(hops):
        if not current:
            break
        edge_groups = store.neighbors_many(
            list(current),
            artifact_id,
            top_k_per_source=neighbors_per_node,
            exclude_frontier=False,
            now_turn=now_turn,
        )
        accumulator = _HeatAccumulator()
        _seed_restart_mass(
            accumulator,
            seeds=seeds,
            restart_probability=restart_probability,
        )
        _walk_frontier_mass(
            accumulator,
            current=current,
            edge_groups=edge_groups,
            walk_fraction=1.0 - restart_probability,
            now_turn=now_turn,
            edge_temperature=edge_temperature,
        )
        if not accumulator.states:
            break
        kept = _trim_to_capacity(accumulator.states, max_nodes=max_nodes)
        if kept is None:
            break
        current = kept

    return HeatDiffusion(nodes=_emit_ranked_nodes(current))


def _default_source_key(result: RetrievalResult) -> str:
    if result.turn is not None and result.turn.source_id:
        return result.turn.source_id
    return result.chunk.turn_id


@dataclass(frozen=True, slots=True)
class _SourceHeatLedger:
    """Which memory source each diffused chunk belongs to, and how hot it is.

    The ledger is the arm's whole notion of fairness: heat is attributed to
    sources rather than chunks, and a source's share of the direct-retrieval
    token spend follows its share of the diffused heat.
    """

    source_by_chunk: dict[str, str]
    source_heat: dict[str, float]
    direct_tokens: int
    total_source_heat: float

    def token_budget(self, source_id: str) -> int:
        """Tokens this source earned, pro-rated by its share of total heat."""

        return round(
            self.direct_tokens * self.source_heat[source_id] / self.total_source_heat
        )


def _tally_source_heat(
    nodes: Sequence[DiffusedHeatNode],
    *,
    hydrated: Callable[[str], RetrievalResult | None],
    resolver: SourceKey,
    direct_tokens: int,
) -> _SourceHeatLedger:
    """Attribute every diffused node's heat to the memory source it came from.

    A node that no longer hydrates is attributed to nothing, so a vanished
    chunk cannot earn its source either exposure or a token budget.
    """

    source_by_chunk: dict[str, str] = {}
    source_heat: dict[str, float] = defaultdict(float)
    for node in nodes:
        result = hydrated(node.chunk_id)
        if result is None:
            continue
        source_id = str(resolver(result))
        source_by_chunk[node.chunk_id] = source_id
        source_heat[source_id] += node.heat
    return _SourceHeatLedger(
        source_by_chunk=source_by_chunk,
        source_heat=source_heat,
        direct_tokens=direct_tokens,
        total_source_heat=sum(source_heat.values()) or 1.0,
    )


def _packed_metadata(
    ledger: _SourceHeatLedger,
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
    """The packed-candidate metadata shape shared by both candidate routes.

    Every candidate reaching composition already knows its source, so these
    fields are always concrete here — unlike the provenance written back onto
    admitted results, which must describe anchors that were never diffused.
    """

    return {
        "anchor_chunk_id": anchor_chunk_id,
        "edge_source_id": edge_source_id,
        "association_hop": association_hop,
        "association_path": association_path,
        "diffusion_heat": diffusion_heat,
        "association_support": association_support,
        "memory_source_id": source_id,
        "source_heat": ledger.source_heat[source_id],
        "source_token_budget": ledger.token_budget(source_id),
        "token_count": token_count,
    }


def _pack_heat_candidates(
    nodes: Sequence[DiffusedHeatNode],
    *,
    anchor_ids: set[str],
    hydrated: Callable[[str], RetrievalResult | None],
    ledger: _SourceHeatLedger,
) -> list[AssociativeMemoryCandidate]:
    """Turn diffused non-anchor nodes into composable ``heat`` candidates.

    An anchor is already in the prompt, so re-admitting it would spend an
    association slot on evidence the caller retrieved directly.
    """

    candidates: list[AssociativeMemoryCandidate] = []
    for node in nodes:
        if node.chunk_id in anchor_ids:
            continue
        result = hydrated(node.chunk_id)
        source_id = ledger.source_by_chunk.get(node.chunk_id)
        if result is None or source_id is None:
            continue
        candidates.append(
            AssociativeMemoryCandidate(
                episode_id=node.chunk_id,
                text=result.chunk.text,
                score=node.heat,
                route="heat",
                metadata=_packed_metadata(
                    ledger,
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
    return candidates


def _density_first_order(
    candidates: Sequence[AssociativeMemoryCandidate],
) -> list[AssociativeMemoryCandidate]:
    """Order candidates by heat per token, so cheap evidence is spent first."""

    return sorted(
        candidates,
        key=lambda candidate: (
            float(candidate.metadata["diffusion_heat"])
            / max(1, int(candidate.metadata["token_count"])),
            float(candidate.metadata["diffusion_heat"]),
            candidate.episode_id,
        ),
        reverse=True,
    )


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


def _ranked_qk_candidates(
    bounded_anchors: Sequence[RetrievalResult],
    artifact_id: str,
    *,
    store: AssociationStore,
    hydrate: Hydrate,
    now_turn: int,
    nodes: Mapping[str, DiffusedHeatNode],
    ledger: _SourceHeatLedger,
    resolver: SourceKey,
    result_cap: int,
    association_slots: int,
    ranked_qk_reserve: int,
    neighbors_per_node: int,
    diffusion_hops: int,
    max_diffusion_nodes: int,
    lexical_protection_threshold: float | None,
    max_prompt_token_increase: int | None,
) -> list[AssociativeMemoryCandidate]:
    """Reserve a small max-path exploitation channel ahead of the heat stream.

    Diffusion rewards corroboration and token-efficient source exposure, but a
    rare decisive edge can carry little global heat. This sub-arm preserves it
    by running the ranked QK arm over the same compact persisted edges — still
    model-free — and packing its results as ``qk`` candidates that composition
    admits before heat spends the remaining association slots.

    Sources first seen here are recorded in the ledger at zero heat: they were
    never diffused, so they have earned no share of the token budget.
    """

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
    candidates: list[AssociativeMemoryCandidate] = []
    for result in ranked_results:
        if result.route != "qk":
            continue
        source_id = ledger.source_by_chunk.get(result.chunk.chunk_id)
        if source_id is None:
            source_id = str(resolver(result))
            ledger.source_by_chunk[result.chunk.chunk_id] = source_id
            ledger.source_heat.setdefault(source_id, 0.0)
        node = nodes.get(result.chunk.chunk_id)
        candidates.append(
            AssociativeMemoryCandidate(
                episode_id=result.chunk.chunk_id,
                text=result.chunk.text,
                score=result.score,
                route="qk",
                metadata=_packed_metadata(
                    ledger,
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
    return candidates


def _diffusion_provenance(
    ledger: _SourceHeatLedger,
    *,
    node: DiffusedHeatNode | None,
    source_id: str | None,
) -> dict[str, object]:
    """The diffusion fields stamped onto every admitted result.

    Deliberately unlike :func:`_packed_metadata`: this runs over the composed
    set, which includes direct anchors that diffusion never reached, so an
    absent node or source reports ``None`` — "not diffused" — rather than a
    zero that would read as measured cold evidence downstream.
    """

    return {
        "diffusion_heat": None if node is None else node.heat,
        "association_support": (
            None if node is None else node.supporting_transitions
        ),
        "memory_source_id": source_id,
        "source_heat": (
            None if source_id is None else ledger.source_heat[source_id]
        ),
        "source_token_budget": (
            None if source_id is None else ledger.token_budget(source_id)
        ),
    }


def _hydrate_composition(
    composition: AssociativeComposition,
    *,
    direct: Mapping[str, RetrievalResult],
    hydrated: Callable[[str], RetrievalResult | None],
    nodes: Mapping[str, DiffusedHeatNode],
    ledger: _SourceHeatLedger,
) -> list[RetrievalResult]:
    """Rebuild composed candidates as results carrying their diffusion story.

    Direct anchors keep their original score and route and gain only source
    provenance; association routes additionally carry the score, path, and hop
    that earned them their slot.
    """

    results: list[RetrievalResult] = []
    for candidate in composition.candidates:
        base = direct.get(candidate.episode_id) or hydrated(candidate.episode_id)
        if base is None:
            continue
        update = _diffusion_provenance(
            ledger,
            node=nodes.get(candidate.episode_id),
            source_id=ledger.source_by_chunk.get(candidate.episode_id),
        )
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
    return results


def _weakest_admitted_heat(
    remaining: Sequence[AssociativeMemoryCandidate], *, selected: set[str]
) -> AssociativeMemoryCandidate:
    """The admitted heat candidate paying least evidence per token spent."""

    return min(
        (candidate for candidate in remaining if candidate.episode_id in selected),
        key=lambda candidate: (
            float(candidate.metadata["diffusion_heat"])
            / max(1, int(candidate.metadata["token_count"])),
            candidate.episode_id,
        ),
    )


def _admit_candidates(
    packed_anchors: Sequence[AssociativeMemoryCandidate],
    ranked_candidates: Sequence[AssociativeMemoryCandidate],
    heat_candidates: Sequence[AssociativeMemoryCandidate],
    *,
    direct: Mapping[str, RetrievalResult],
    hydrated: Callable[[str], RetrievalResult | None],
    nodes: Mapping[str, DiffusedHeatNode],
    ledger: _SourceHeatLedger,
    budget_anchors: Sequence[RetrievalResult],
    result_cap: int,
    qk_reserve: int,
    association_slots: int,
    protected: Sequence[str],
    max_prompt_token_increase: int | None,
) -> list[RetrievalResult] | None:
    """Compose the widest admissible set that still fits the token budget.

    If the hottest association is too expensive, the arm drops that single
    candidate and re-composes rather than rolling the whole heat arm back:
    the remaining stream may well fit. ``None`` is returned only when nothing
    admissible is left, meaning the caller must fall back to direct retrieval.
    """

    remaining = list(heat_candidates)
    while True:
        composition = compose_associative_candidates(
            packed_anchors,
            qk_neighbors=[*ranked_candidates, *remaining],
            top_k=result_cap,
            qk_reserve=qk_reserve,
            association_slots=association_slots,
            protected_anchor_ids=protected,
        )
        results = _hydrate_composition(
            composition,
            direct=direct,
            hydrated=hydrated,
            nodes=nodes,
            ledger=ledger,
        )
        if not exceeds_prompt_budget(
            results,
            # Denominator: like hebbian, this arm spends against all
            # bounded_anchors; associative windows to [:result_cap].
            direct_anchors=budget_anchors,
            max_prompt_token_increase=max_prompt_token_increase,
        ):
            return results
        selected_heat = {
            candidate.episode_id
            for candidate in composition.candidates
            if candidate.route == "heat"
        }
        if not selected_heat:
            return None
        worst = _weakest_admitted_heat(remaining, selected=selected_heat)
        remaining = [
            candidate
            for candidate in remaining
            if candidate.episode_id != worst.episode_id
        ]
        if not remaining:
            return None


def _touch_traversed_edges(
    results: Sequence[RetrievalResult],
    artifact_id: str,
    *,
    store: AssociationStore,
    now_turn: int,
) -> None:
    """Credit the edges that actually carried an admitted association.

    Only the explanatory path of an admitted association route is reinforced,
    so recency accrues to edges that earned prompt space, not to every edge
    the diffusion happened to sample.
    """

    edge_pairs: list[tuple[str, str]] = []
    for result in results:
        if result.route not in {"qk", "heat"} or result.association_path is None:
            continue
        edge_pairs.extend(zip(result.association_path, result.association_path[1:]))
    if edge_pairs:
        store.touch_edges(artifact_id, edge_pairs, now_turn=now_turn)


def _validate_heat_budgets(
    *,
    association_slots: int,
    qk_reserve: int,
    ranked_qk_reserve: int,
    max_source_token_fraction: float,
) -> None:
    """Reject slot budgets this arm could not honor without overspending."""

    if association_slots < 0 or qk_reserve < 0 or ranked_qk_reserve < 0:
        raise ValueError("association budgets must be non-negative")
    if ranked_qk_reserve > qk_reserve:
        raise ValueError("ranked_qk_reserve cannot exceed qk_reserve")
    if not 0.0 < max_source_token_fraction <= 1.0:
        raise ValueError("max_source_token_fraction must lie in (0, 1]")


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
    """Allocate bounded association slots using diffused per-source heat.

    The arm is a pipeline over the phases named below: diffuse heat, attribute
    it to memory sources, pack and fairly order the candidates it justifies,
    optionally reserve a ranked-QK exploitation slot ahead of them, admit as
    many as the token budget allows, and credit the edges that were used.
    Store reads and edge writes live here; every phase in between is a
    function of its inputs.
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
    _validate_heat_budgets(
        association_slots=association_slots,
        qk_reserve=qk_reserve,
        ranked_qk_reserve=ranked_qk_reserve,
        max_source_token_fraction=max_source_token_fraction,
    )

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
    hydrated = hydration_memo(
        hydrate,
        seed={anchor.chunk.chunk_id: anchor for anchor in bounded_anchors},
    )

    resolver = source_key or _default_source_key
    direct_tokens = sum(anchor.chunk.token_count for anchor in bounded_anchors)
    ledger = _tally_source_heat(
        diffusion.nodes,
        hydrated=hydrated,
        resolver=resolver,
        direct_tokens=direct_tokens,
    )

    heat_candidates = _source_fair_order(
        _density_first_order(
            _pack_heat_candidates(
                diffusion.nodes,
                anchor_ids=anchor_ids,
                hydrated=hydrated,
                ledger=ledger,
            )
        ),
        source_heat=ledger.source_heat,
        total_token_budget=max(1, direct_tokens),
        max_source_token_fraction=max_source_token_fraction,
    )

    ranked_candidates: list[AssociativeMemoryCandidate] = []
    if ranked_qk_reserve:
        ranked_candidates = _ranked_qk_candidates(
            bounded_anchors,
            artifact_id,
            store=store,
            hydrate=hydrate,
            now_turn=now_turn,
            nodes=nodes,
            ledger=ledger,
            resolver=resolver,
            result_cap=result_cap,
            association_slots=association_slots,
            ranked_qk_reserve=ranked_qk_reserve,
            neighbors_per_node=neighbors_per_node,
            diffusion_hops=diffusion_hops,
            max_diffusion_nodes=max_diffusion_nodes,
            lexical_protection_threshold=lexical_protection_threshold,
            max_prompt_token_increase=max_prompt_token_increase,
        )
    ranked_ids = {candidate.episode_id for candidate in ranked_candidates}
    heat_candidates = [
        candidate
        for candidate in heat_candidates
        if candidate.episode_id not in ranked_ids
    ]

    results = _admit_candidates(
        anchor_candidates(bounded_anchors),
        ranked_candidates,
        heat_candidates,
        direct={anchor.chunk.chunk_id: anchor for anchor in bounded_anchors},
        hydrated=hydrated,
        nodes=nodes,
        ledger=ledger,
        budget_anchors=bounded_anchors,
        result_cap=result_cap,
        qk_reserve=qk_reserve,
        association_slots=association_slots,
        protected=protected_anchor_ids(
            bounded_anchors,
            lexical_protection_threshold=lexical_protection_threshold,
        ),
        max_prompt_token_increase=max_prompt_token_increase,
    )
    if results is None:
        return bounded_anchors

    if touch:
        _touch_traversed_edges(
            results,
            artifact_id,
            store=store,
            now_turn=now_turn,
        )
    return results[:result_cap]
