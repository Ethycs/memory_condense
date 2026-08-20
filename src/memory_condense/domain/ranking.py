"""The deterministic rerank scalar from the design:

    score = wR*relevance + wI*importance + wP*pin_boost
          + wE*energy - wS*superseded_penalty

Pure functions only — no I/O, no model calls, and deliberately **no decay
kernel** — ``energy`` arrives already decayed from :mod:`memory_condense.domain.decay`,
which owns that arithmetic.

The ``wE*energy`` term was ``wT*recency`` until 2026-08-14, computed here from
a second copy of the exponential in ``decay.py``. Two consequences, both
measured: the two copies had drifted to opposite semantics for a non-positive
half-life, and because ``MemoryStore.touch`` restamps ``last_access_at`` on
every retrieve, ``recency`` was 1.0 for every item ever recalled — a constant,
discriminating nothing. Decayed energy carries the same time signal *times* a
stored amplitude that access frequency actually moves.

Note that only memory items carry the full scalar. Chunks are ranked by
``blend_hybrid`` alone — they have no importance, pin, or energy.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Hashable, Iterable, Mapping, Sequence, TypeVar

from memory_condense.domain.schemas import PinState

T = TypeVar("T")


@dataclass(frozen=True)
class RankWeights:
    """Weights for the rerank scalar. Defaults keep relevance dominant."""

    relevance: float = 1.0
    importance: float = 0.3
    pin: float = 0.5
    energy: float = 0.2
    superseded_penalty: float = 1.0


DEFAULT_WEIGHTS = RankWeights()

#: How much a pin contributes before the ``pin`` weight is applied.
_PIN_BOOST = {
    PinState.USER: 1.0,
    PinState.SYSTEM: 0.6,
    PinState.NONE: 0.0,
}


def pin_boost(pin: PinState) -> float:
    return _PIN_BOOST.get(pin, 0.0)


def rank_score(
    relevance: float,
    importance: float = 0.0,
    pin: PinState = PinState.NONE,
    energy: float = 0.0,
    superseded: bool = False,
    weights: RankWeights = DEFAULT_WEIGHTS,
) -> float:
    """The rerank scalar. All component inputs are expected in [0, 1].

    ``energy`` must already be decayed to the scoring instant — pass
    ``decay.item_energy(item, now=...)``, not ``item.energy``, which is the
    stored amplitude and ignores elapsed time.
    """
    score = (
        weights.relevance * relevance
        + weights.importance * importance
        + weights.pin * pin_boost(pin)
        + weights.energy * energy
    )
    if superseded:
        score -= weights.superseded_penalty
    return score


def blend_hybrid(dense: float, lexical: float, alpha: float = 0.65) -> float:
    """Combine a dense and a lexical relevance score.

    ``alpha`` is the dense weight; both inputs must already be on a comparable
    scale (use ``min_max_normalize`` on raw BM25 scores first).
    """
    alpha = max(0.0, min(1.0, alpha))
    return alpha * dense + (1.0 - alpha) * lexical


def min_max_normalize(
    values: Sequence[float], *, flat_value: float = 1.0
) -> list[float]:
    """Scale values into [0, 1]. A flat input carries no signal and maps to
    ``flat_value`` everywhere (1.0 by default; pass 0.5 to treat an invariant
    component as neutral rather than uniformly strong)."""
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-12:
        return [flat_value for _ in values]
    span = hi - lo
    return [(v - lo) / span for v in values]


def softmax(
    values: Sequence[float],
    *,
    temperature: float = 1.0,
    clamp: float | None = None,
) -> list[float]:
    """Max-shifted softmax over raw scores.

    ``temperature`` divides the shifted logits; ``clamp`` (when given) bounds
    each shifted logit to ``[-clamp, clamp]`` for overflow-proof weighting of
    unbounded energies.
    """
    if not values:
        return []
    maximum = max(values)
    shifted = [(value - maximum) / temperature for value in values]
    if clamp is not None:
        shifted = [max(-clamp, min(clamp, value)) for value in shifted]
    exponentials = [math.exp(value) for value in shifted]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def top_k(scored: Iterable[tuple[float, object]], k: int) -> list[tuple[float, object]]:
    """Highest-scoring k pairs, descending. Stable for equal scores."""
    if k <= 0:
        return []
    return sorted(scored, key=lambda pair: pair[0], reverse=True)[:k]


def round_robin_unique(
    groups: Sequence[Sequence[T]],
    limit: int | None = None,
    *,
    key: Callable[[T], Hashable] | None = None,
    seen: set[Hashable] | None = None,
    stop_on_stall: bool = True,
) -> list[T]:
    """Interleave groups position-by-position, one item per group per round.

    Groups keep their caller-supplied order, both across groups and within
    each group. ``limit`` caps the output, stopping mid-round the instant it
    is reached.

    With ``key`` set, an item whose key was already emitted — or pre-seeded in
    ``seen``, which is mutated in place — is skipped, and two stall policies
    diverge: ``stop_on_stall=True`` gives up the first time a full position
    yields nothing new, while ``False`` keeps scanning deeper positions until
    every group is exhausted. Without ``key`` nothing is ever skipped, so a
    stalled position already means every group is exhausted and the flag is
    inert.
    """

    if seen is None:
        seen = set()
    output: list[T] = []
    position = 0
    while groups and (limit is None or len(output) < limit):
        added = False
        for group in groups:
            if position >= len(group):
                continue
            item = group[position]
            if key is not None:
                item_key = key(item)
                if item_key in seen:
                    continue
                seen.add(item_key)
            output.append(item)
            added = True
            if limit is not None and len(output) >= limit:
                break
        if not added and (
            stop_on_stall
            or all(position >= len(group) - 1 for group in groups)
        ):
            break
        position += 1
    return output


def source_rows_with_fallback(
    candidates_by_source: Mapping[str, Sequence[T]],
    *,
    dedup_key: Callable[[T], Hashable] | None = None,
) -> tuple[list[tuple[str, list[T]]], dict[str, T]]:
    """Materialize per-source candidate queues plus a first-row fallback map.

    Source order and each source's local candidate order are preserved; empty
    sources are dropped from both structures. ``dedup_key`` removes repeats
    within one source only — the same item may still appear under two sources.
    The fallback maps every surviving source to its first row, so a selector
    whose scoring backend fails can still return one companion per source.
    """

    source_rows: list[tuple[str, list[T]]] = []
    for source_id, candidates in candidates_by_source.items():
        if dedup_key is None:
            rows = list(candidates)
        else:
            row_keys: set[Hashable] = set()
            rows = []
            for item in candidates:
                item_key = dedup_key(item)
                if item_key in row_keys:
                    continue
                row_keys.add(item_key)
                rows.append(item)
        if rows:
            source_rows.append((str(source_id), rows))
    fallback = {source_id: rows[0] for source_id, rows in source_rows}
    return source_rows, fallback


def weighted_fair_order(
    items: Sequence[T],
    *,
    source_key: Callable[[T], str],
    source_weight: Mapping[str, float],
    item_cost: Callable[[T], int],
    item_priority: Callable[[T], float],
    total_budget: int,
    max_source_fraction: float,
    cost_clip: int | None = None,
) -> list[T]:
    """Deficit round-robin over per-source queues, weighted by purchasing power.

    Every source keeps its items in the caller-supplied order. Each pick serves
    the source whose accumulated spend per unit of ``source_weight`` is lowest,
    so the early prefix of the returned list spends ``item_cost`` tokens in
    proportion to source weight; ties break toward higher ``item_priority``,
    then lexicographic source id. A source whose served cost would exceed
    ``max_source_fraction`` of ``total_budget`` is deferred while any uncapped
    queue can still serve, though every source may always serve its first item.
    ``cost_clip``, when given, caps a single item's accounted cost — callers
    that later truncate items to a fixed token ceiling pass that ceiling so an
    oversized chunk is not billed for tokens it will never render.
    """

    queues: dict[str, deque[T]] = defaultdict(deque)
    for item in items:
        queues[source_key(item)].append(item)

    def accounted_cost(item: T) -> int:
        cost = item_cost(item)
        if cost_clip is not None:
            cost = min(cost, cost_clip)
        return max(1, cost)

    served: dict[str, int] = defaultdict(int)
    ordered: list[T] = []
    source_cap = max(1, math.ceil(total_budget * max_source_fraction))
    while any(queues.values()):
        choices: list[tuple[float, float, str]] = []
        capped_choices: list[tuple[float, float, str]] = []
        for source_id, queue in queues.items():
            if not queue:
                continue
            cost = accounted_cost(queue[0])
            weight = max(source_weight.get(source_id, 0.0), 1e-12)
            choice = (
                (served[source_id] + cost) / weight,
                -item_priority(queue[0]),
                source_id,
            )
            choices.append(choice)
            if served[source_id] == 0 or served[source_id] + cost <= source_cap:
                capped_choices.append(choice)
        _, _, source_id = min(capped_choices or choices)
        item = queues[source_id].popleft()
        served[source_id] += accounted_cost(item)
        ordered.append(item)
    return ordered
