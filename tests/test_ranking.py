from __future__ import annotations

import pytest

from memory_condense.ranking import (
    DEFAULT_WEIGHTS,
    RankWeights,
    blend_hybrid,
    min_max_normalize,
    pin_boost,
    rank_score,
    top_k,
)
from memory_condense.schemas import PinState

# The decay kernel used to live here as `recency_score`, a second copy of the
# exponential in decay.py. It now has one home; its tests moved with it to
# tests/test_decay.py::TestDecayFactor.


class TestPinBoost:
    def test_user_pin_outranks_system_pin(self):
        assert pin_boost(PinState.USER) > pin_boost(PinState.SYSTEM)

    def test_unpinned_contributes_nothing(self):
        assert pin_boost(PinState.NONE) == 0.0


class TestRankScore:
    def test_relevance_only_matches_relevance(self):
        assert rank_score(relevance=0.7) == pytest.approx(0.7)

    def test_importance_adds_weighted_contribution(self):
        score = rank_score(relevance=0.5, importance=1.0)
        assert score == pytest.approx(0.5 + DEFAULT_WEIGHTS.importance)

    def test_pin_raises_score(self):
        plain = rank_score(relevance=0.5)
        pinned = rank_score(relevance=0.5, pin=PinState.USER)
        assert pinned > plain

    def test_superseded_penalty_applied(self):
        plain = rank_score(relevance=0.9)
        stale = rank_score(relevance=0.9, superseded=True)
        assert stale == pytest.approx(plain - DEFAULT_WEIGHTS.superseded_penalty)
        assert stale < 0

    def test_custom_weights_respected(self):
        weights = RankWeights(relevance=2.0, importance=0.0)
        assert rank_score(relevance=0.5, importance=1.0, weights=weights) == pytest.approx(1.0)


class TestBlendHybrid:
    def test_alpha_one_is_pure_dense(self):
        assert blend_hybrid(0.8, 0.2, alpha=1.0) == pytest.approx(0.8)

    def test_alpha_zero_is_pure_lexical(self):
        assert blend_hybrid(0.8, 0.2, alpha=0.0) == pytest.approx(0.2)

    def test_default_alpha_favours_dense(self):
        blended = blend_hybrid(1.0, 0.0)
        assert blended > 0.5

    def test_alpha_is_clamped(self):
        assert blend_hybrid(0.8, 0.2, alpha=5.0) == pytest.approx(0.8)


class TestMinMaxNormalize:
    def test_scales_into_unit_range(self):
        assert min_max_normalize([1.0, 3.0, 5.0]) == [0.0, 0.5, 1.0]

    def test_flat_input_has_no_signal(self):
        assert min_max_normalize([2.0, 2.0, 2.0]) == [1.0, 1.0, 1.0]

    def test_empty_input(self):
        assert min_max_normalize([]) == []


class TestTopK:
    def test_returns_highest_scores_descending(self):
        pairs = [(0.1, "a"), (0.9, "b"), (0.5, "c")]
        assert [p[1] for p in top_k(pairs, 2)] == ["b", "c"]

    def test_k_zero_returns_empty(self):
        assert top_k([(1.0, "a")], 0) == []

    def test_k_larger_than_input(self):
        assert len(top_k([(1.0, "a")], 10)) == 1
