"""Energy decay and HOT/WARM/COLD tiering for memory items.

Energy decays exponentially with a per-item half-life. Decay is computed
lazily on read — nothing runs on a timer — so ``effective_energy`` is the
single source of truth for "how hot is this item *now*".

Pins override decay entirely: a pinned item holds its stored energy forever.
"""

from __future__ import annotations

from datetime import datetime, timezone

from memory_condense.schemas import (
    DEFAULT_HALF_LIFE_S,
    HOT_THRESHOLD,
    WARM_THRESHOLD,
    Heat,
    MemoryItem,
    PinState,
)

#: Energy added when an item is retrieved ("access reheating").
REHEAT_BOOST = 0.25

#: Energy assigned to a newly created item the extractor marked important.
HOT_SEED_ENERGY = 0.8

#: Energy assigned to an ordinary new item.
WARM_SEED_ENERGY = 0.5


def _as_utc(dt: datetime) -> datetime:
    """Treat naive datetimes as UTC (SQLite round-trips can drop tzinfo)."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


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
    """
    if pinned:
        return _clamp(energy)
    if half_life_s <= 0:
        return _clamp(energy)

    elapsed = (_as_utc(now or now_utc()) - _as_utc(last_access_at)).total_seconds()
    if elapsed <= 0:
        return _clamp(energy)

    return _clamp(energy * (0.5 ** (elapsed / half_life_s)))


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
    """Raise energy on access, capped at 1.0."""
    return _clamp(energy + boost)


def seed_energy(importance: float) -> float:
    """Starting energy for a new item: important items enter HOT, others WARM."""
    return HOT_SEED_ENERGY if importance >= 0.7 else WARM_SEED_ENERGY


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
