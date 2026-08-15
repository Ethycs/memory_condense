"""The deterministic rerank scalar from the design:

    score = wR*relevance + wI*importance + wP*pin_boost
          + wE*energy - wS*superseded_penalty

Pure functions only — no I/O, no model calls, and deliberately **no decay
kernel** — ``energy`` arrives already decayed from :mod:`memory_condense.decay`,
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

from dataclasses import dataclass
from typing import Iterable, Sequence

from memory_condense.schemas import PinState


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
