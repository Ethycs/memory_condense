from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from memory_condense.decay import (
    HOT_SEED_ENERGY,
    WARM_SEED_ENERGY,
    effective_energy,
    heat_for,
    item_energy,
    item_heat,
    reheat,
    seed_energy,
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


class TestReheatAndSeed:
    def test_reheat_raises_energy(self):
        assert reheat(0.5) == pytest.approx(0.75)

    def test_reheat_caps_at_one(self):
        assert reheat(0.95) == 1.0

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
