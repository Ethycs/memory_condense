from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from memory_condense.decay import (
    HOT_CAP,
    HOT_SEED_ENERGY,
    REHEAT_BOOST,
    REHEAT_REFRACTORY_S,
    WARM_SEED_ENERGY,
    decay_factor,
    effective_energy,
    heat_for,
    heat_map,
    item_energy,
    item_heat,
    reheat,
    seed_energy,
    should_reheat,
)
from memory_condense.schemas import (
    DEFAULT_HALF_LIFE_S,
    Heat,
    MemoryItem,
    MemoryType,
    PinState,
)


def _now() -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc)


class TestEffectiveEnergy:
    def test_no_elapsed_time_returns_energy(self):
        now = _now()
        assert effective_energy(0.8, now, now) == pytest.approx(0.8)

    def test_one_half_life_halves_energy(self):
        now = _now()
        last = now - timedelta(seconds=DEFAULT_HALF_LIFE_S)
        assert effective_energy(0.8, last, now) == pytest.approx(0.4)

    def test_two_half_lives_quarters_energy(self):
        now = _now()
        last = now - timedelta(seconds=2 * DEFAULT_HALF_LIFE_S)
        assert effective_energy(0.8, last, now) == pytest.approx(0.2)

    def test_pinned_items_do_not_decay(self):
        now = _now()
        last = now - timedelta(days=365)
        assert effective_energy(0.8, last, now, pinned=True) == pytest.approx(0.8)

    def test_naive_datetime_treated_as_utc(self):
        now = _now()
        naive = datetime(2026, 1, 1) - timedelta(seconds=DEFAULT_HALF_LIFE_S)
        assert effective_energy(1.0, naive, now) == pytest.approx(0.5)

    def test_result_is_clamped(self):
        now = _now()
        assert effective_energy(5.0, now, now) == 1.0
        assert effective_energy(-3.0, now, now) == 0.0

    def test_future_last_access_does_not_amplify(self):
        now = _now()
        future = now + timedelta(days=10)
        assert effective_energy(0.5, future, now) == pytest.approx(0.5)


class TestHeatFor:
    @pytest.mark.parametrize(
        "energy,expected",
        [
            (1.0, Heat.HOT),
            (0.75, Heat.HOT),
            (0.74, Heat.WARM),
            (0.25, Heat.WARM),
            (0.24, Heat.COLD),
            (0.0, Heat.COLD),
        ],
    )
    def test_thresholds(self, energy, expected):
        assert heat_for(energy) is expected


class TestDecayFactor:
    """Moved from tests/test_ranking.py when `recency_score` was deleted."""

    def test_just_happened_scores_one(self):
        now = _now()
        assert decay_factor(now, now) == pytest.approx(1.0)

    def test_one_half_life_scores_half(self):
        now = _now()
        past = now - timedelta(seconds=DEFAULT_HALF_LIFE_S)
        assert decay_factor(past, now) == pytest.approx(0.5)

    def test_future_timestamp_clamps_to_one(self):
        now = _now()
        assert decay_factor(now + timedelta(days=1), now) == 1.0

    def test_naive_datetime_handled(self):
        now = _now()
        assert decay_factor(datetime(2026, 1, 1), now) == pytest.approx(1.0)

    def test_non_positive_half_life_does_not_decay(self):
        """The semantics the two implementations used to disagree on.

        `decay.effective_energy` returned the stored energy (never decays);
        `ranking.recency_score` returned 0.0 (fully stale). "Never decays"
        won: invisible memories are silent data loss, immortal ones are not.
        """
        now = _now()
        assert decay_factor(now - timedelta(days=365), now, half_life_s=0) == 1.0
        assert decay_factor(now - timedelta(days=365), now, half_life_s=-1) == 1.0

    def test_effective_energy_is_amplitude_times_decay_factor(self):
        """The identity that stops the two kernels ever re-diverging."""
        now = _now()
        past = now - timedelta(days=3)
        for energy in (0.0, 0.25, 0.5, 0.8, 1.0):
            for half_life in (3600.0, DEFAULT_HALF_LIFE_S, 30 * 86400.0):
                assert effective_energy(
                    energy, past, now=now, half_life_s=half_life
                ) == pytest.approx(
                    energy * decay_factor(past, now, half_life_s=half_life)
                )


class TestReheatAndSeed:
    def test_reheat_closes_a_fraction_of_the_headroom(self):
        # 0.5 + 0.25 * (1 - 0.5) = 0.625, not the old flat 0.75.
        assert reheat(0.5) == pytest.approx(0.625)

    def test_reheat_never_reaches_one(self):
        """Saturating, so a frequently-read item cannot pin itself at max."""
        energy = 0.0
        for _ in range(50):
            energy = reheat(energy)
            assert energy < 1.0

    def test_reheat_fixed_point_is_monotone_in_access_frequency(self):
        """The property that makes energy a rate estimator instead of a ratchet.

        With the old additive reheat every interval below ~3 days converged to
        exactly 1.0, so the term was a constant for all regular use.
        """

        def fixed_point(interval_s: float) -> float:
            energy = 0.5
            for _ in range(400):
                energy = reheat(
                    effective_energy(
                        energy,
                        _now(),
                        now=_now() + timedelta(seconds=interval_s),
                    )
                )
            return energy

        hourly = fixed_point(3600)
        daily = fixed_point(86400)
        weekly = fixed_point(7 * 86400)
        monthly = fixed_point(30 * 86400)
        assert 1.0 > hourly > daily > weekly > monthly > 0.0

    def test_refractory_window_suppresses_a_second_boost(self):
        now = _now()
        assert should_reheat(now - timedelta(seconds=REHEAT_REFRACTORY_S + 1), now)
        assert not should_reheat(now - timedelta(seconds=1), now)
        assert not should_reheat(now, now)

    def test_reheat_boost_is_still_the_documented_fraction(self):
        assert reheat(0.0) == pytest.approx(REHEAT_BOOST)

    def test_important_items_seed_hot(self):
        assert seed_energy(0.8) == HOT_SEED_ENERGY
        assert heat_for(seed_energy(0.8)) is Heat.HOT

    def test_ordinary_items_seed_warm(self):
        assert seed_energy(0.5) == WARM_SEED_ENERGY
        assert heat_for(seed_energy(0.5)) is Heat.WARM


class TestItemHelpers:
    def _item(self, **kwargs) -> MemoryItem:
        defaults = dict(
            type=MemoryType.DECISION,
            content="use SQLite",
            energy=0.8,
            last_access_at=_now() - timedelta(seconds=DEFAULT_HALF_LIFE_S),
        )
        defaults.update(kwargs)
        return MemoryItem(**defaults)

    def test_item_energy_decays(self):
        assert item_energy(self._item(), now=_now()) == pytest.approx(0.4)

    def test_pinned_item_energy_holds(self):
        item = self._item(pin=PinState.USER)
        assert item_energy(item, now=_now()) == pytest.approx(0.8)

    def test_item_heat_uses_decayed_energy(self):
        assert item_heat(self._item(), now=_now()) is Heat.WARM
        assert item_heat(self._item(pin=PinState.USER), now=_now()) is Heat.HOT

    def test_stored_heat_property_is_not_the_decayed_tier(self):
        """`MemoryItem.heat` reads *stored* energy and is a live footgun.

        Because `seed_energy` and `reheat` never drive stored energy below
        REHEAT_BOOST, `item.heat` is effectively never COLD. Anything that
        enumerates cold items must use `decay.item_heat`, which applies
        elapsed time — otherwise it silently returns nothing and looks correct.
        """
        item = self._item(energy=0.8, last_access_at=_now() - timedelta(days=60))
        assert item.heat is Heat.HOT  # stored: no time applied
        assert item_heat(item, now=_now()) is Heat.COLD  # decayed: 60 days on


class TestHeatMap:
    def _item(self, energy: float, mem_id: str, **kwargs) -> MemoryItem:
        defaults = dict(
            mem_id=mem_id,
            type=MemoryType.DECISION,
            content=f"item {mem_id}",
            energy=energy,
            last_access_at=_now(),
        )
        defaults.update(kwargs)
        return MemoryItem(**defaults)

    def test_hot_beyond_the_cap_is_demoted_to_warm(self):
        items = [self._item(0.9, f"{i:04d}") for i in range(HOT_CAP + 5)]
        tiers = heat_map(items, now=_now())
        assert sum(1 for h in tiers.values() if h is Heat.HOT) == HOT_CAP
        assert sum(1 for h in tiers.values() if h is Heat.WARM) == 5

    def test_the_lowest_energy_items_are_the_ones_demoted(self):
        items = [
            self._item(0.99 - i * 0.001, f"{i:04d}") for i in range(HOT_CAP + 3)
        ]
        tiers = heat_map(items, now=_now())
        demoted = [m for m, h in tiers.items() if h is Heat.WARM]
        assert sorted(demoted) == [f"{i:04d}" for i in range(HOT_CAP, HOT_CAP + 3)]

    def test_pins_never_consume_a_slot_and_are_never_demoted(self):
        pinned = [
            self._item(0.99, f"p{i:03d}", pin=PinState.USER) for i in range(5)
        ]
        plain = [self._item(0.9, f"u{i:03d}") for i in range(HOT_CAP)]
        tiers = heat_map(pinned + plain, now=_now())
        assert all(tiers[p.mem_id] is Heat.HOT for p in pinned)
        # All 20 unpinned still fit: the 5 pins did not take slots.
        assert all(tiers[u.mem_id] is Heat.HOT for u in plain)

    def test_demotion_is_one_tier_only(self):
        items = [self._item(0.9, f"{i:04d}") for i in range(HOT_CAP * 3)]
        tiers = heat_map(items, now=_now())
        assert Heat.COLD not in tiers.values()

    def test_genuinely_cold_items_are_untouched_by_the_cap(self):
        cold = self._item(
            0.8, "cold", last_access_at=_now() - timedelta(days=60)
        )
        hot = [self._item(0.9, f"{i:04d}") for i in range(HOT_CAP + 2)]
        tiers = heat_map(hot + [cold], now=_now())
        assert tiers["cold"] is Heat.COLD

    def test_result_is_stable_across_input_order(self):
        items = [self._item(0.9, f"{i:04d}") for i in range(HOT_CAP + 4)]
        assert heat_map(items, now=_now()) == heat_map(
            list(reversed(items)), now=_now()
        )

    def test_empty_pool(self):
        assert heat_map([], now=_now()) == {}
