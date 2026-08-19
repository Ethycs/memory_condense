"""Adaptive surprise boundaries and bounded graph-cohesion refinement."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


SimilarityLookup = Callable[[int, int], float]


@runtime_checkable
class BoundaryDetector(Protocol):
    """Pluggable scalar-to-boundary strategy used by :class:`EpisodeBuilder`."""

    requires_surprise_scores: bool
    method: str

    def detect(self, surprises: Sequence[float]) -> tuple[BoundaryProposal, ...]: ...


@dataclass(frozen=True, slots=True)
class BoundaryProposal:
    """A boundary before ``position`` under an adaptive trailing threshold."""

    position: int
    score: float
    threshold: float

    def __post_init__(self) -> None:
        if self.position < 1:
            raise ValueError("a boundary position must be positive")
        if not math.isfinite(float(self.score)):
            raise ValueError("boundary score must be finite")
        if not math.isfinite(float(self.threshold)):
            raise ValueError("boundary threshold must be finite")


@dataclass(frozen=True, slots=True)
class BoundaryRefinement:
    """A source-local refinement with a deterministic graph objective."""

    initial_position: int
    position: int
    score: float
    threshold: float
    cohesion: float | None = None

    def __post_init__(self) -> None:
        if self.initial_position < 1 or self.position < 1:
            raise ValueError("boundary positions must be positive")
        for name in ("score", "threshold", "cohesion"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")


class AdaptiveBoundaryDetector:
    """Detect ``score > moving_mean + gamma * moving_population_std``.

    Only the strictly preceding ``window_size`` values form the baseline.  A
    value that ages out of that trailing window cannot affect later decisions.
    The detector retains configuration only and is safe to reuse across
    unrelated source histories.
    """

    requires_surprise_scores = True
    method = "adaptive_surprise"
    __slots__ = ("window_size", "gamma", "min_history")

    def __init__(
        self,
        *,
        window_size: int = 32,
        gamma: float = 1.0,
        min_history: int = 2,
    ) -> None:
        if int(window_size) < 1:
            raise ValueError("window_size must be positive")
        if int(min_history) < 1 or int(min_history) > int(window_size):
            raise ValueError("min_history must be inside [1, window_size]")
        normalized_gamma = float(gamma)
        if not math.isfinite(normalized_gamma) or normalized_gamma < 0.0:
            raise ValueError("gamma must be finite and non-negative")
        self.window_size = int(window_size)
        self.gamma = normalized_gamma
        self.min_history = int(min_history)

    def detect(self, surprises: Sequence[float]) -> tuple[BoundaryProposal, ...]:
        values = tuple(float(value) for value in surprises)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("surprise scores must all be finite")
        proposals: list[BoundaryProposal] = []
        for position in range(1, len(values)):
            history = values[max(0, position - self.window_size) : position]
            if len(history) < self.min_history:
                continue
            mean = statistics.fmean(history)
            deviation = statistics.pstdev(history)
            threshold = mean + self.gamma * deviation
            if values[position] > threshold:
                proposals.append(
                    BoundaryProposal(
                        position=position,
                        score=values[position],
                        threshold=threshold,
                    )
                )
        return tuple(proposals)


class FixedIntervalBoundaryDetector:
    """Model-free ablation that proposes a boundary every ``interval`` rows."""

    requires_surprise_scores = False
    method = "fixed_interval"
    __slots__ = ("interval",)

    def __init__(self, interval: int = 8) -> None:
        normalized = int(interval)
        if normalized < 1 or float(interval) != float(normalized):
            raise ValueError("interval must be a positive integer")
        self.interval = normalized

    def detect(self, surprises: Sequence[float]) -> tuple[BoundaryProposal, ...]:
        values = tuple(float(value) for value in surprises)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("boundary-control scores must all be finite")
        return tuple(
            BoundaryProposal(position=position, score=1.0, threshold=0.0)
            for position in range(self.interval, len(values), self.interval)
        )


class CohesionBoundaryRefiner:
    """Move each proposal only inside a bounded local similarity graph.

    The local graph contains at most ``max_nodes`` observations.  Each node
    contributes at most ``max_degree`` nearest-neighbor choices; ties break by
    source ordinal.  Candidate cuts maximize mean within-side edge cohesion
    minus crossing-edge cohesion, then minimize displacement, then choose the
    earlier cut.  The implementation never stores the supplied matrix.
    """

    __slots__ = ("window", "max_nodes", "max_degree")

    def __init__(
        self,
        *,
        window: int = 4,
        max_nodes: int = 32,
        max_degree: int = 4,
    ) -> None:
        if int(window) < 0:
            raise ValueError("window must be non-negative")
        if int(max_nodes) < 2:
            raise ValueError("max_nodes must be at least two")
        if int(max_degree) < 1:
            raise ValueError("max_degree must be positive")
        self.window = int(window)
        self.max_nodes = int(max_nodes)
        self.max_degree = int(max_degree)

    def refine(
        self,
        proposals: Sequence[BoundaryProposal],
        *,
        item_count: int,
        similarities: Sequence[Sequence[float]] | SimilarityLookup | None,
    ) -> tuple[BoundaryRefinement, ...]:
        if int(item_count) < 0:
            raise ValueError("item_count must be non-negative")
        ordered = tuple(sorted(proposals, key=lambda item: item.position))
        if any(item.position >= item_count for item in ordered):
            raise ValueError("a boundary cannot fall outside the source stream")
        if len({item.position for item in ordered}) != len(ordered):
            raise ValueError("boundary proposal positions must be unique")
        similarity = _validated_similarity_lookup(similarities, item_count)

        refined: list[BoundaryRefinement] = []
        for index, proposal in enumerate(ordered):
            lower = 1 if index == 0 else ordered[index - 1].position + 1
            upper = (
                item_count - 1
                if index + 1 == len(ordered)
                else ordered[index + 1].position - 1
            )
            lower = max(lower, proposal.position - self.window)
            upper = min(upper, proposal.position + self.window)
            if similarity is None or self.window == 0 or lower > upper:
                refined.append(
                    BoundaryRefinement(
                        initial_position=proposal.position,
                        position=proposal.position,
                        score=proposal.score,
                        threshold=proposal.threshold,
                    )
                )
                continue

            local_start, local_end = _bounded_local_range(
                proposal.position,
                item_count,
                self.max_nodes,
                self.window,
            )
            lower = max(lower, local_start + 1)
            upper = min(upper, local_end - 1)
            edges = _local_graph_edges(
                similarity,
                start=local_start,
                end=local_end,
                max_degree=self.max_degree,
            )
            choices = []
            for position in range(lower, upper + 1):
                cohesion = _cut_cohesion(edges, position)
                choices.append(
                    (
                        -cohesion,
                        abs(position - proposal.position),
                        position,
                        cohesion,
                    )
                )
            if not choices:
                position = proposal.position
                cohesion = None
            else:
                _, _, position, cohesion = min(choices)
            refined.append(
                BoundaryRefinement(
                    initial_position=proposal.position,
                    position=position,
                    score=proposal.score,
                    threshold=proposal.threshold,
                    cohesion=cohesion,
                )
            )
        return _deduplicate_refinements(refined)


def _validated_similarity_lookup(
    similarities: Sequence[Sequence[float]] | SimilarityLookup | None,
    item_count: int,
) -> SimilarityLookup | None:
    if similarities is None:
        return None
    if callable(similarities):
        def lookup(left: int, right: int) -> float:
            value = float(similarities(left, right))
            if not math.isfinite(value):
                raise ValueError("similarity lookup must return only finite values")
            return max(0.0, min(1.0, value))

        return lookup
    matrix = tuple(tuple(float(value) for value in row) for row in similarities)
    if len(matrix) != item_count or any(len(row) != item_count for row in matrix):
        raise ValueError("similarity matrix must be square and match item_count")
    if not all(math.isfinite(value) for row in matrix for value in row):
        raise ValueError("similarity matrix must contain only finite values")
    bounded = tuple(
        tuple(max(0.0, min(1.0, value)) for value in row)
        for row in matrix
    )

    return lambda left, right: bounded[left][right]


def _bounded_local_range(
    position: int,
    item_count: int,
    max_nodes: int,
    window: int,
) -> tuple[int, int]:
    start = max(0, position - window - 1)
    end = min(item_count, position + window + 1)
    while end - start > max_nodes:
        left_room = position - start
        right_room = end - position
        if right_room > left_room:
            end -= 1
        else:
            start += 1
    return start, end


def _local_graph_edges(
    similarity: SimilarityLookup,
    *,
    start: int,
    end: int,
    max_degree: int,
) -> tuple[tuple[int, int, float], ...]:
    edge_weights: dict[tuple[int, int], float] = {}
    for node in range(start, end):
        neighbors = sorted(
            (
                (-similarity(node, other), other)
                for other in range(start, end)
                if other != node
            )
        )[:max_degree]
        for _, other in neighbors:
            left, right = sorted((node, other))
            weight = (similarity(node, other) + similarity(other, node)) / 2.0
            edge_weights[(left, right)] = max(
                edge_weights.get((left, right), 0.0),
                weight,
            )
    return tuple(
        (left, right, edge_weights[(left, right)])
        for left, right in sorted(edge_weights)
    )


def _cut_cohesion(
    edges: Sequence[tuple[int, int, float]],
    position: int,
) -> float:
    left = [weight for a, b, weight in edges if a < position and b < position]
    right = [weight for a, b, weight in edges if a >= position and b >= position]
    crossing = [weight for a, b, weight in edges if a < position <= b]
    within_means = []
    if left:
        within_means.append(statistics.fmean(left))
    if right:
        within_means.append(statistics.fmean(right))
    within = statistics.fmean(within_means) if within_means else 0.0
    across = statistics.fmean(crossing) if crossing else 0.0
    return within - across


def _deduplicate_refinements(
    rows: Sequence[BoundaryRefinement],
) -> tuple[BoundaryRefinement, ...]:
    by_position: dict[int, BoundaryRefinement] = {}
    for row in rows:
        existing = by_position.get(row.position)
        if existing is None or (
            -row.score,
            row.initial_position,
        ) < (
            -existing.score,
            existing.initial_position,
        ):
            by_position[row.position] = row
    return tuple(by_position[position] for position in sorted(by_position))


__all__ = [
    "AdaptiveBoundaryDetector",
    "BoundaryDetector",
    "BoundaryProposal",
    "BoundaryRefinement",
    "CohesionBoundaryRefiner",
    "FixedIntervalBoundaryDetector",
]
