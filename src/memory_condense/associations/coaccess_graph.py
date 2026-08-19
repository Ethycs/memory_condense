"""Shared decay-aware co-access graph arithmetic and traversal.

Both durable co-access graphs — the artifact-scoped Hebbian chunk graph and
the cross-partition live consolidation graph — persist the same statistics:
decayed ``activation**2`` node mass, decayed pairwise activation-product edge
mass, and SHA-256 event receipts.  This module holds the shared math, the
noisy-OR neighbor accumulation, and the prune selection; each store keeps its
own SQLite schema, receipt payload, and result types.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping

from memory_condense.domain.decay import decay_factor


def rank_discount(rank: int) -> float:
    """The shared ``1 / sqrt(rank)`` exposure discount for ranked results."""
    return 1.0 / math.sqrt(rank)


def decayed_mass(
    mass: float,
    last_turn: int,
    now_turn: int,
    half_life_turns: float,
) -> float:
    return max(0.0, float(mass)) * decay_factor(
        last_turn,
        now_turn,
        half_life_turns,
    )


def coaccess_edge_score(
    *,
    edge_mass: float,
    last_reinforced_turn: int,
    left_mass: float,
    left_turn: int,
    right_mass: float,
    right_turn: int,
    now_turn: int,
    half_life_turns: float,
) -> float:
    """Time-decayed cosine association, which discounts frequent hubs.

    With matching exponential updates the ratio is a cosine and therefore at
    most one.  A separate freshness term is intentional: otherwise an isolated
    pair's node and edge masses decay in lockstep and its normalized score
    never cools.
    """
    edge = decayed_mass(
        edge_mass,
        last_reinforced_turn,
        now_turn,
        half_life_turns,
    )
    left = decayed_mass(left_mass, left_turn, now_turn, half_life_turns)
    right = decayed_mass(right_mass, right_turn, now_turn, half_life_turns)
    denominator = math.sqrt(left * right)
    if denominator <= 0.0:
        return 0.0
    normalized = min(1.0, max(0.0, edge / denominator))
    freshness = decay_factor(
        last_reinforced_turn,
        now_turn,
        half_life_turns,
    )
    return normalized * freshness


def validate_observation_params(
    *,
    access_event_id: str,
    learning_rate: float,
    half_life_turns: float,
    max_members_per_event: int,
    max_degree: int,
    min_edge_score: float,
    max_event_history: int,
    member_limit_name: str,
) -> tuple[str, float, float]:
    """Validate one observation's scalar parameters shared by both stores."""
    event_id = str(access_event_id).strip()
    if not event_id:
        raise ValueError("access_event_id must be non-empty")
    if len(event_id) > 256:
        raise ValueError("access_event_id must be at most 256 characters")
    rate = float(learning_rate)
    half_life = float(half_life_turns)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(half_life) or half_life <= 0.0:
        raise ValueError("half_life_turns must be finite and positive")
    if max_members_per_event < 1:
        raise ValueError(f"{member_limit_name} must be positive")
    if max_degree < 0:
        raise ValueError("max_degree must be non-negative")
    if not 0.0 <= min_edge_score <= 1.0:
        raise ValueError("min_edge_score must lie in [0, 1]")
    if max_event_history < 1:
        raise ValueError("max_event_history must be positive")
    return event_id, rate, half_life


@dataclass(slots=True)
class CoaccessNeighborState:
    """Accumulated noisy-OR evidence for one candidate node."""

    score: float = 0.0
    anchors: set[str] = field(default_factory=set)
    anchor_key: str = ""
    best_evidence: float = -1.0
    coaccess_count: int = 0
    causal_count: int = 0
    last_reinforced_turn: int = 0

    @property
    def support(self) -> int:
        return len(self.anchors)


def accumulate_neighbor_evidence(
    edge_rows: Iterable[tuple[str, str, float, int, int, int]],
    *,
    seeds: Mapping[str, float],
    excluded: set[str],
    nodes: Mapping[str, tuple[float, int]],
    default_node: tuple[float, int] | None,
    now_turn: int,
    half_life_turns: float,
    min_score: float,
    min_coaccess_count: int = 0,
    min_causal_count: int = 0,
    candidate_allowed: Callable[[str], bool] | None = None,
) -> dict[str, CoaccessNeighborState]:
    """Fold decayed edge evidence around seed nodes into candidate states.

    Rows are ``(low, high, mass, count, causal_count, last_reinforced_turn)``;
    a store without causal evidence passes zero.  ``default_node`` of ``None``
    requires both endpoints to exist in ``nodes``.  Noisy-OR combines support
    from several anchors without allowing a high-degree candidate to gain an
    unbounded additive score.
    """
    candidates: dict[str, CoaccessNeighborState] = {}
    for low, high, mass, count, causal_count, edge_turn in edge_rows:
        low = str(low)
        high = str(high)
        if (
            int(count) < min_coaccess_count
            and int(causal_count) < min_causal_count
        ):
            continue
        if low in seeds and high not in seeds:
            anchor_key, candidate_key = low, high
        elif high in seeds and low not in seeds:
            anchor_key, candidate_key = high, low
        else:
            continue
        if candidate_key in excluded:
            continue
        if candidate_allowed is not None and not candidate_allowed(candidate_key):
            continue
        left = nodes.get(low, default_node)
        right = nodes.get(high, default_node)
        if left is None or right is None:
            continue
        edge_score = coaccess_edge_score(
            edge_mass=float(mass),
            last_reinforced_turn=int(edge_turn),
            left_mass=left[0],
            left_turn=left[1],
            right_mass=right[0],
            right_turn=right[1],
            now_turn=now_turn,
            half_life_turns=half_life_turns,
        )
        evidence = min(1.0, edge_score * seeds[anchor_key])
        if evidence < min_score:
            continue
        state = candidates.setdefault(candidate_key, CoaccessNeighborState())
        state.score = 1.0 - (1.0 - state.score) * (1.0 - evidence)
        state.anchors.add(anchor_key)
        state.coaccess_count += int(count)
        state.causal_count += int(causal_count)
        state.last_reinforced_turn = max(
            state.last_reinforced_turn, int(edge_turn)
        )
        if evidence > state.best_evidence:
            state.best_evidence = evidence
            state.anchor_key = anchor_key
    return candidates


def ranked_neighbor_states(
    candidates: Mapping[str, CoaccessNeighborState],
) -> list[tuple[str, CoaccessNeighborState]]:
    """Order candidates by score, then support, then stable key."""
    return sorted(
        candidates.items(),
        key=lambda item: (-item[1].score, -item[1].support, item[0]),
    )


def score_coaccess_edges(
    edge_rows: Iterable[tuple[str, str, float, int]],
    nodes: Mapping[str, tuple[float, int]],
    *,
    now_turn: int,
    half_life_turns: float,
) -> dict[tuple[str, str], float]:
    """Score ``(low, high, mass, last_reinforced_turn)`` rows for pruning."""
    scored: dict[tuple[str, str], float] = {}
    for low, high, mass, edge_turn in edge_rows:
        low = str(low)
        high = str(high)
        left = nodes.get(low, (0.0, now_turn))
        right = nodes.get(high, (0.0, now_turn))
        scored[(low, high)] = coaccess_edge_score(
            edge_mass=float(mass),
            last_reinforced_turn=int(edge_turn),
            left_mass=left[0],
            left_turn=left[1],
            right_mass=right[0],
            right_turn=right[1],
            now_turn=now_turn,
            half_life_turns=half_life_turns,
        )
    return scored


def select_prune_victims(
    scored: Mapping[tuple[str, str], float],
    scoped: Iterable[str],
    *,
    max_degree: int,
    min_score: float,
) -> set[tuple[str, str]]:
    """Edges that cooled below ``min_score`` or exceed the degree bound."""
    deletions = {edge for edge, score in scored.items() if score < min_score}
    for node_key in scoped:
        incident = [
            (score, edge)
            for edge, score in scored.items()
            if node_key in edge and edge not in deletions
        ]
        incident.sort(key=lambda item: (-item[0], item[1]))
        deletions.update(edge for _score, edge in incident[max_degree:])
    return deletions
