from __future__ import annotations

import pytest

from memory_condense.decay import (
    HOT_CAP,
    HOT_SEED_ENERGY,
    REHEAT_BOOST,
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
    DEFAULT_HALF_LIFE_TURNS,
    Heat,
    MemoryItem,
    MemoryType,
    PinState,
)

HL = DEFAULT_HALF_LIFE_TURNS  # 30 turns


class TestEffectiveEnergy:
    def test_same_turn_returns_energy(self):
        assert effective_energy(0.8, 10, 10) == pytest.approx(0.8)

    def test_one_half_life_halves_energy(self):
        assert effective_energy(0.8, 0, int(HL)) == pytest.approx(0.4)

    def test_two_half_lives_quarters_energy(self):
        assert effective_energy(0.8, 0, int(2 * HL)) == pytest.approx(0.2)

    def test_pinned_items_do_not_decay(self):
        assert effective_energy(0.8, 0, 10_000, pinned=True) == pytest.approx(0.8)

    def test_result_is_clamped(self):
        assert effective_energy(5.0, 3, 3) == 1.0
        assert effective_energy(-3.0, 3, 3) == 0.0

    def test_future_last_access_does_not_amplify(self):
        """An item stamped ahead of the clock decays by nothing, not upward."""
        assert effective_energy(0.5, 100, 40) == pytest.approx(0.5)


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
    def test_same_turn_scores_one(self):
        assert decay_factor(7, 7) == pytest.approx(1.0)

    def test_one_half_life_scores_half(self):
        assert decay_factor(0, int(HL)) == pytest.approx(0.5)

    def test_future_turn_clamps_to_one(self):
        assert decay_factor(50, 10) == 1.0

    def test_non_positive_half_life_does_not_decay(self):
        """The semantics the two implementations used to disagree on.

        `decay.effective_energy` returned the stored energy (never decays);
        the deleted `ranking.recency_score` returned 0.0 (fully stale). "Never
        decays" won: invisible memories are silent data loss, immortal ones
        are not.
        """
        assert decay_factor(0, 10_000, half_life_turns=0) == 1.0
        assert decay_factor(0, 10_000, half_life_turns=-1) == 1.0

    def test_effective_energy_is_amplitude_times_decay_factor(self):
        """The identity that stops the two kernels ever re-diverging."""
        for energy in (0.0, 0.25, 0.5, 0.8, 1.0):
            for half_life in (5.0, HL, 300.0):
                assert effective_energy(
                    energy, 4, now_turn=40, half_life_turns=half_life
                ) == pytest.approx(
                    energy * decay_factor(4, 40, half_life_turns=half_life)
                )

    def test_the_coordinate_is_turns_not_wall_clock(self):
        """The defect schema v4 exists to fix.

        Wall-clock decay could not express "each subsequent turn differentially
        assigns decay": an ingest runs in minutes, so elapsed rounded to
        nothing and every item held a factor of ~1.0 whether the conversation
        touched it or not. Two items stamped at the same *instant* but a
        half-life apart in *turns* must not share a decay factor.
        """
        recalled_recently = decay_factor(100, 100)
        left_behind = decay_factor(100 - int(HL), 100)
        assert recalled_recently == pytest.approx(1.0)
        assert left_behind == pytest.approx(0.5)
        assert recalled_recently != pytest.approx(left_behind)


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

        With the old additive reheat every short interval converged to exactly
        1.0, so the term was a constant for all regular use. Now the fixed
        point strictly orders items by *how often the conversation reaches for
        them*, which is the whole point of the mechanism.
        """

        def fixed_point(interval_turns: int) -> float:
            energy = 0.5
            for _ in range(400):
                energy = reheat(
                    effective_energy(energy, 0, now_turn=interval_turns)
                )
            return energy

        every_turn = fixed_point(1)
        every_5 = fixed_point(5)
        every_30 = fixed_point(30)
        every_60 = fixed_point(60)
        assert 1.0 > every_turn > every_5 > every_30 > every_60 > 0.0

    def test_reheat_is_once_per_turn(self):
        """Ten recalls while answering one turn is one access, not ten."""
        assert should_reheat(9, 10)
        assert not should_reheat(10, 10)

    def test_an_item_is_inside_its_own_window_on_creation(self):
        """Creation is the access, so it neither decays nor boosts."""
        assert not should_reheat(42, 42)
        assert decay_factor(42, 42) == pytest.approx(1.0)

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
            last_access_turn=0,
        )
        defaults.update(kwargs)
        return MemoryItem(**defaults)

    def test_item_energy_decays(self):
        assert item_energy(self._item(), now_turn=int(HL)) == pytest.approx(0.4)

    def test_pinned_item_energy_holds(self):
        item = self._item(pin=PinState.USER)
        assert item_energy(item, now_turn=int(HL)) == pytest.approx(0.8)

    def test_item_heat_uses_decayed_energy(self):
        assert item_heat(self._item(), now_turn=int(HL)) is Heat.WARM
        assert item_heat(self._item(pin=PinState.USER), now_turn=int(HL)) is Heat.HOT

    def test_stored_heat_property_is_not_the_decayed_tier(self):
        """`MemoryItem.heat` reads *stored* energy and is a live footgun.

        Because `seed_energy` and `reheat` never drive stored energy below
        REHEAT_BOOST, `item.heat` is effectively never COLD. Anything that
        enumerates cold items must use `decay.item_heat`, which applies the
        elapsed turns — otherwise it silently returns nothing and looks correct.
        """
        item = self._item(energy=0.8, last_access_turn=0)
        assert item.heat is Heat.HOT  # stored: no turns applied
        assert item_heat(item, now_turn=int(2 * HL)) is Heat.COLD

    def test_an_ordinary_item_reaches_cold_within_one_conversation(self):
        """The property the wall-clock coordinate made unreachable.

        Under a seven-day half-life an item needed 7-11.75 days of no access to
        reach COLD, so no run could produce one and Phase 4's gate was
        unsatisfiable. Both seed levels must now cross inside a normal-length
        conversation.
        """
        ordinary = self._item(energy=WARM_SEED_ENERGY, last_access_turn=0)
        important = self._item(energy=HOT_SEED_ENERGY, last_access_turn=0)
        assert item_heat(ordinary, now_turn=31) is Heat.COLD
        assert item_heat(important, now_turn=31) is Heat.WARM
        assert item_heat(important, now_turn=51) is Heat.COLD


class TestHeatMap:
    def _item(self, energy: float, mem_id: str, **kwargs) -> MemoryItem:
        defaults = dict(
            mem_id=mem_id,
            type=MemoryType.DECISION,
            content=f"item {mem_id}",
            energy=energy,
            last_access_turn=100,
        )
        defaults.update(kwargs)
        return MemoryItem(**defaults)

    def test_hot_beyond_the_cap_is_demoted_to_warm(self):
        items = [self._item(0.9, f"{i:04d}") for i in range(HOT_CAP + 5)]
        tiers = heat_map(items, now_turn=100)
        assert sum(1 for h in tiers.values() if h is Heat.HOT) == HOT_CAP
        assert sum(1 for h in tiers.values() if h is Heat.WARM) == 5

    def test_the_lowest_energy_items_are_the_ones_demoted(self):
        items = [self._item(0.99 - i * 0.001, f"{i:04d}") for i in range(HOT_CAP + 3)]
        tiers = heat_map(items, now_turn=100)
        demoted = [m for m, h in tiers.items() if h is Heat.WARM]
        assert sorted(demoted) == [f"{i:04d}" for i in range(HOT_CAP, HOT_CAP + 3)]

    def test_pins_never_consume_a_slot_and_are_never_demoted(self):
        pinned = [self._item(0.99, f"p{i:03d}", pin=PinState.USER) for i in range(5)]
        plain = [self._item(0.9, f"u{i:03d}") for i in range(HOT_CAP)]
        tiers = heat_map(pinned + plain, now_turn=100)
        assert all(tiers[p.mem_id] is Heat.HOT for p in pinned)
        # All 20 unpinned still fit: the 5 pins did not take slots.
        assert all(tiers[u.mem_id] is Heat.HOT for u in plain)

    def test_demotion_is_one_tier_only(self):
        items = [self._item(0.9, f"{i:04d}") for i in range(HOT_CAP * 3)]
        tiers = heat_map(items, now_turn=100)
        assert Heat.COLD not in tiers.values()

    def test_genuinely_cold_items_are_untouched_by_the_cap(self):
        cold = self._item(0.8, "cold", last_access_turn=0)
        hot = [self._item(0.9, f"{i:04d}") for i in range(HOT_CAP + 2)]
        tiers = heat_map(hot + [cold], now_turn=100)
        assert tiers["cold"] is Heat.COLD

    def test_result_is_stable_across_input_order(self):
        items = [self._item(0.9, f"{i:04d}") for i in range(HOT_CAP + 4)]
        assert heat_map(items, now_turn=100) == heat_map(
            list(reversed(items)), now_turn=100
        )

    def test_empty_pool(self):
        assert heat_map([], now_turn=100) == {}
