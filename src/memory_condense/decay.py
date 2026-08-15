"""Energy decay and HOT/WARM/COLD tiering for memory items.

Energy decays exponentially with a per-item half-life. Decay is computed
lazily on read — nothing runs on a timer — so ``effective_energy`` is the
single source of truth for "how hot is this item *now*".

Pins override decay entirely: a pinned item holds its stored energy forever.

**This module owns the decay kernel, and it owns it alone.** Until 2026-08-14
``ranking.recency_score`` carried a second copy of the same exponential, and
the two had drifted to opposite semantics for a non-positive half-life. They
were never independent to begin with::

    effective_energy(e, t, now, hl) == e * decay_factor(t, now, hl)

so ``recency`` was this module's energy term with the amplitude forced to 1.0.
The scalar in ``ranking`` now takes decayed energy directly. If you find
yourself writing ``0.5 ** (elapsed / half_life)`` anywhere else, you are
re-opening that bug — call :func:`decay_factor`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from memory_condense.schemas import (
    DEFAULT_HALF_LIFE_S,
    HOT_THRESHOLD,
    WARM_THRESHOLD,
    Heat,
    MemoryItem,
    PinState,
)

#: Fraction of the *remaining* headroom that an access closes ("reheating").
#:
#: Deliberately multiplicative rather than additive. With a flat ``+0.25`` the
#: fixed point of decay-then-reheat clamps at 1.0 for anything touched more
#: often than roughly every three days, so every regularly-used item pins
#: itself at maximum and the energy term stops discriminating — the same
#: failure the old constant ``recency`` term had. Closing a fraction of the
#: headroom instead gives a fixed point that is strictly monotone in access
#: frequency and never reaches 1.0::
#:
#:     e* = b / (1 - d*(1 - b))     where d = 0.5 ** (interval / half_life)
#:
#: hourly ≈ 0.99, daily ≈ 0.78, every 3 days ≈ 0.57, weekly ≈ 0.40,
#: monthly ≈ 0.26, never → 0. Energy becomes a rate estimator, and only a pin
#: can hold the top of the range — which is what pinning is for.
REHEAT_BOOST = 0.25

#: Minimum gap between two reheats of the same item.
#:
#: Without it, ten ``recall`` calls in one working session ratchet an item
#: from 0.5 to 0.97 — "accessed ten times in a minute" is one access, not ten.
#: ``MemoryStore.touch`` still restamps ``last_access_at`` inside the window;
#: only the boost is withheld.
REHEAT_REFRACTORY_S = 300.0

#: Energy assigned to a newly created item the extractor marked important.
HOT_SEED_ENERGY = 0.8

#: Energy assigned to an ordinary new item.
WARM_SEED_ENERGY = 0.5

#: Importance at or above which a new item seeds HOT rather than WARM.
SEED_IMPORTANCE_THRESHOLD = 0.7

#: Most unpinned items that may hold HOT at once (the design's "HOT cap ~20").
#:
#: Applied when *deriving* tiers, never stored — see :func:`heat_map`.
HOT_CAP = 20


def _as_utc(dt: datetime) -> datetime:
    """Treat naive datetimes as UTC (SQLite round-trips can drop tzinfo)."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def decay_factor(
    last_access_at: datetime,
    now: datetime | None = None,
    half_life_s: float = DEFAULT_HALF_LIFE_S,
) -> float:
    """``0.5 ** (elapsed / half_life)`` — the kernel, without the amplitude.

    1.0 for an access in the future or the present, and 1.0 for a non-positive
    half-life. That last case used to differ between this module (which
    returned the stored energy, i.e. never decays) and ``ranking.recency_score``
    (which returned 0.0, i.e. fully stale) for the same input. "Never decays"
    wins: a corrupt half-life that makes memories immortal is visible and
    recoverable, while one that makes them invisible is silent data loss.
    """
    if half_life_s <= 0:
        return 1.0
    elapsed = (_as_utc(now or now_utc()) - _as_utc(last_access_at)).total_seconds()
    if elapsed <= 0:
        return 1.0
    return 0.5 ** (elapsed / half_life_s)


def effective_energy(
    energy: float,
    last_access_at: datetime,
    now: datetime | None = None,
    half_life_s: float = DEFAULT_HALF_LIFE_S,
    pinned: bool = False,
) -> float:
    """Energy decayed forward from ``last_access_at`` to ``now``.

    Exponential with the given half-life: after one half-life the stored
    energy has halved. Pinned items are exempt and return ``energy`` as-is.

    Identically ``energy * decay_factor(...)`` for an unpinned item — a
    property asserted in the tests so the two cannot silently re-diverge.
    """
    if pinned:
        return _clamp(energy)
    return _clamp(energy * decay_factor(last_access_at, now, half_life_s))


def heat_for(energy: float) -> Heat:
    """Tier an energy value using the design thresholds (0.75 / 0.25)."""
    if energy >= HOT_THRESHOLD:
        return Heat.HOT
    if energy >= WARM_THRESHOLD:
        return Heat.WARM
    return Heat.COLD


def item_energy(item: MemoryItem, now: datetime | None = None) -> float:
    """Decayed energy for a stored item, honouring its pin state."""
    return effective_energy(
        energy=item.energy,
        last_access_at=item.last_access_at,
        now=now,
        half_life_s=item.half_life_s,
        pinned=item.is_pinned,
    )


def item_heat(item: MemoryItem, now: datetime | None = None) -> Heat:
    """Decayed heat tier for a stored item."""
    return heat_for(item_energy(item, now=now))


def reheat(energy: float, boost: float = REHEAT_BOOST) -> float:
    """Raise energy on access with diminishing returns.

    Each access closes ``boost`` of the *remaining* headroom, so energy
    approaches 1.0 without reaching it. See :data:`REHEAT_BOOST` for why this
    is not a flat addition.
    """
    return _clamp(energy + boost * (1.0 - _clamp(energy)))


def should_reheat(
    last_access_at: datetime,
    now: datetime | None = None,
    refractory_s: float = REHEAT_REFRACTORY_S,
) -> bool:
    """False if this item was already reheated within the refractory window."""
    if refractory_s <= 0:
        return True
    elapsed = (_as_utc(now or now_utc()) - _as_utc(last_access_at)).total_seconds()
    return elapsed >= refractory_s


def seed_energy(importance: float) -> float:
    """Starting energy for a new item: important items enter HOT, others WARM."""
    return (
        HOT_SEED_ENERGY
        if importance >= SEED_IMPORTANCE_THRESHOLD
        else WARM_SEED_ENERGY
    )


def heat_map(
    items: Iterable[MemoryItem],
    now: datetime | None = None,
    hot_cap: int = HOT_CAP,
) -> dict[str, Heat]:
    """Tier a whole pool, enforcing the design's HOT cap.

    :func:`item_heat` tiers one item against fixed thresholds. That is the
    right answer for a single item and the wrong one for a store: with a
    seven-day half-life and reheating on recall, nearly every active item
    clears ``HOT_THRESHOLD``, so ``Heat`` reports HOT for everything and
    carries no information.

    Here the tier is derived *pool-relative*: threshold first, then unpinned
    items beyond ``hot_cap`` are demoted one tier to WARM, lowest energy
    demoted first. Pins never occupy a slot and are never demoted — that is
    what a pin buys. Nothing is written; the standard's "heat tier — derived,
    never stored" still holds, this simply derives from more than one row.

    Ties break on ``(-energy, mem_id)``, so the result is stable across runs
    and across input order.
    """
    ranked = sorted(items, key=lambda i: (-item_energy(i, now=now), i.mem_id))

    tiers: dict[str, Heat] = {}
    unpinned_hot = 0
    for item in ranked:
        heat = item_heat(item, now=now)
        if heat is Heat.HOT and not item.is_pinned:
            unpinned_hot += 1
            if unpinned_hot > hot_cap:
                heat = Heat.WARM
        tiers[item.mem_id] = heat
    return tiers


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
