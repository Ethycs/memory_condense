"""Energy decay and HOT/WARM/COLD tiering for memory items.

**The decay coordinate is conversation turns, not wall-clock time.** Energy
decays exponentially in the number of turns since an item was last created or
recalled. Decay is computed lazily on read — nothing runs on a timer — so
``effective_energy`` is the single source of truth for "how hot is this item
*now*".

Turns are the coordinate because the design's claim is that *each subsequent
turn differentially assigns decay*: the conversation itself decides what stays
warm. That mechanism was already wired — ``MemoryStore.retrieve`` reheats
exactly the top-k it returns, so a turn selects its winners for free — but it
was inert while the exponent counted seconds. An ingest runs in minutes, so
``elapsed`` rounded to nothing and every item, retrieved or not, kept a decay
factor of ~1.0. Selection carried no consequence.

That is the same failure as the old ``ranking.recency_score``: a term that
looks like a discriminator and evaluates to a constant. Fixing it twice at two
levels is why this module owns the kernel alone::

    effective_energy(e, t, now, hl) == e * decay_factor(t, now, hl)

If you find yourself writing ``0.5 ** (elapsed / half_life)`` anywhere else,
you are re-opening that bug — call :func:`decay_factor`.

Pins override decay entirely: a pinned item holds its stored energy forever.

``last_access_at`` still exists on :class:`MemoryItem` and is still restamped,
but it is **audit only**. Nothing here reads it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from memory_condense.domain.schemas import (
    DEFAULT_HALF_LIFE_TURNS,
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
#: With a 30-turn half-life: recalled every turn ≈ 0.99, every 5 turns ≈ 0.75,
#: every 15 ≈ 0.57, every 30 ≈ 0.40, every 60 ≈ 0.29, never → 0. Energy becomes
#: a *rate* estimator — how often the conversation reaches for this item — and
#: only a pin can hold the top of the range, which is what pinning is for.
REHEAT_BOOST = 0.25

#: Reheating is once per turn.
#:
#: Without a refractory rule, ten ``recall`` calls while answering one turn
#: ratchet an item from 0.5 to 0.97 — "accessed ten times in one turn" is one
#: access, not ten. The old rule was a 300-second wall-clock window, which was
#: only ever an approximation of "the same turn"; with turns as the coordinate
#: the exact rule is available, so use it. ``MemoryStore.touch`` still restamps
#: within the same turn; only the boost is withheld.
REHEAT_ONCE_PER_TURN = True

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


def now_utc() -> datetime:
    """Wall-clock now, for stamping the **audit** field ``last_access_at``.

    Not part of the decay kernel. The tz-normalising ``_as_utc`` helper that
    used to live beside this is gone with it: decay reads an integer turn, so
    SQLite dropping ``tzinfo`` on a round-trip can no longer perturb an
    energy value.
    """
    return datetime.now(timezone.utc)


def decay_factor(
    last_access_turn: int,
    now_turn: int = 0,
    half_life_turns: float = DEFAULT_HALF_LIFE_TURNS,
) -> float:
    """``0.5 ** (turns_elapsed / half_life_turns)`` — the kernel, no amplitude.

    ``turns_elapsed`` is ``now_turn - last_access_turn``: how many turns the
    conversation has moved on without reaching for this item. This is the
    whole mechanism — an item the conversation keeps recalling has its
    ``last_access_turn`` pushed forward by ``touch`` and stays near 1.0, while
    one it has moved past falls behind and cools. Nothing else differentiates
    items, and nothing needs to: retrieval already selects the winners.

    1.0 for an access in the future or the current turn, and 1.0 for a
    non-positive half-life. That last case used to differ between this module
    (never decays) and ``ranking.recency_score`` (fully stale) for the same
    input. "Never decays" wins: a corrupt half-life that makes memories
    immortal is visible and recoverable, while one that makes them invisible
    is silent data loss.

    Both arguments default to 0 so a bare :class:`MemoryItem` evaluated with no
    explicit turn decays by nothing, which is what callers that do not track
    turns should get.
    """
    if half_life_turns <= 0:
        return 1.0
    elapsed = now_turn - last_access_turn
    if elapsed <= 0:
        return 1.0
    return 0.5 ** (elapsed / half_life_turns)


def effective_energy(
    energy: float,
    last_access_turn: int,
    now_turn: int = 0,
    half_life_turns: float = DEFAULT_HALF_LIFE_TURNS,
    pinned: bool = False,
) -> float:
    """Energy decayed forward from ``last_access_turn`` to ``now_turn``.

    Exponential in turns: after ``half_life_turns`` untouched turns the stored
    energy has halved. Pinned items are exempt and return ``energy`` as-is.

    Identically ``energy * decay_factor(...)`` for an unpinned item — a
    property asserted in the tests so the two cannot silently re-diverge.
    """
    if pinned:
        return _clamp(energy)
    return _clamp(energy * decay_factor(last_access_turn, now_turn, half_life_turns))


def heat_for(energy: float) -> Heat:
    """Tier an energy value using the design thresholds (0.75 / 0.25)."""
    if energy >= HOT_THRESHOLD:
        return Heat.HOT
    if energy >= WARM_THRESHOLD:
        return Heat.WARM
    return Heat.COLD


def item_energy(item: MemoryItem, now_turn: int = 0) -> float:
    """Decayed energy for a stored item, honouring its pin state."""
    return effective_energy(
        energy=item.energy,
        last_access_turn=item.last_access_turn,
        now_turn=now_turn,
        half_life_turns=item.half_life_turns,
        pinned=item.is_pinned,
    )


def item_heat(item: MemoryItem, now_turn: int = 0) -> Heat:
    """Decayed heat tier for a stored item."""
    return heat_for(item_energy(item, now_turn=now_turn))


def reheat(energy: float, boost: float = REHEAT_BOOST) -> float:
    """Raise energy on access with diminishing returns.

    Each access closes ``boost`` of the *remaining* headroom, so energy
    approaches 1.0 without reaching it. See :data:`REHEAT_BOOST` for why this
    is not a flat addition.
    """
    return _clamp(energy + boost * (1.0 - _clamp(energy)))


def should_reheat(
    last_access_turn: int,
    now_turn: int = 0,
    once_per_turn: bool = REHEAT_ONCE_PER_TURN,
) -> bool:
    """False if this item was already reheated on this turn.

    Note the asymmetry with :func:`decay_factor`, which is deliberate: an item
    created *this* turn has ``last_access_turn == now_turn`` and so is already
    within its own window. It decays by nothing and it boosts by nothing —
    creation is the access.
    """
    if not once_per_turn:
        return True
    return now_turn > last_access_turn


def seed_energy(importance: float) -> float:
    """Starting energy for a new item: important items enter HOT, others WARM."""
    return (
        HOT_SEED_ENERGY
        if importance >= SEED_IMPORTANCE_THRESHOLD
        else WARM_SEED_ENERGY
    )


def heat_map(
    items: Iterable[MemoryItem],
    now_turn: int = 0,
    hot_cap: int = HOT_CAP,
) -> dict[str, Heat]:
    """Tier a whole pool, enforcing the design's HOT cap.

    :func:`item_heat` tiers one item against fixed thresholds. That is the
    right answer for a single item and the wrong one for a store: early in a
    conversation few items have fallen behind, so nearly every active item
    clears ``HOT_THRESHOLD`` and ``Heat`` reports HOT for everything, carrying
    no information.

    Here the tier is derived *pool-relative*: threshold first, then unpinned
    items beyond ``hot_cap`` are demoted one tier to WARM, lowest energy
    demoted first. Pins never occupy a slot and are never demoted — that is
    what a pin buys. Nothing is written; the standard's "heat tier — derived,
    never stored" still holds, this simply derives from more than one row.

    Ties break on ``(-energy, mem_id)``, so the result is stable across runs
    and across input order.
    """
    ranked = sorted(items, key=lambda i: (-item_energy(i, now_turn=now_turn), i.mem_id))

    tiers: dict[str, Heat] = {}
    unpinned_hot = 0
    for item in ranked:
        heat = item_heat(item, now_turn=now_turn)
        if heat is Heat.HOT and not item.is_pinned:
            unpinned_hot += 1
            if unpinned_hot > hot_cap:
                heat = Heat.WARM
        tiers[item.mem_id] = heat
    return tiers


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
