"""The deterministic rerank scalar from the design:

    score = wR*relevance + wI*importance + wP*pin_boost
          + wT*recency - wS*superseded_penalty

Pure functions only — no I/O, no model calls. Both the memory store (ranking
memory items) and the retriever (ranking chunks) score through here so the
weighting lives in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Sequence

from memory_condense.schemas import DEFAULT_HALF_LIFE_S, PinState


@dataclass(frozen=True)
class RankWeights:
    """Weights for the rerank scalar. Defaults keep relevance dominant."""

    relevance: float = 1.0
    importance: float = 0.3
    pin: float = 0.5
    recency: float = 0.2
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


def recency_score(
    timestamp: datetime,
    now: datetime | None = None,
    half_life_s: float = DEFAULT_HALF_LIFE_S,
) -> float:
    """1.0 for something that just happened, decaying to 0 with age."""
    now = now or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    elapsed = (now - timestamp).total_seconds()
    if elapsed <= 0:
        return 1.0
    if half_life_s <= 0:
        return 0.0
    return 0.5 ** (elapsed / half_life_s)


def rank_score(
    relevance: float,
    importance: float = 0.0,
    pin: PinState = PinState.NONE,
    recency: float = 0.0,
    superseded: bool = False,
    weights: RankWeights = DEFAULT_WEIGHTS,
) -> float:
    """The rerank scalar. All component inputs are expected in [0, 1]."""
    score = (
        weights.relevance * relevance
        + weights.importance * importance
        + weights.pin * pin_boost(pin)
        + weights.recency * recency
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


def min_max_normalize(values: Sequence[float]) -> list[float]:
    """Scale values into [0, 1]. A flat input maps to all-1.0 (no signal)."""
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-12:
        return [1.0 for _ in values]
    span = hi - lo
    return [(v - lo) / span for v in values]


def top_k(scored: Iterable[tuple[float, object]], k: int) -> list[tuple[float, object]]:
    """Highest-scoring k pairs, descending. Stable for equal scores."""
    if k <= 0:
        return []
    return sorted(scored, key=lambda pair: pair[0], reverse=True)[:k]
