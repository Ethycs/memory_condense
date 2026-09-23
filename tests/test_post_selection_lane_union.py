from __future__ import annotations

from dataclasses import dataclass

import pytest

from memory_condense.search.post_selection_lane_union import (
    RankedEvidenceLane,
    post_selection_lane_union,
)


@dataclass(frozen=True, slots=True)
class Candidate:
    evidence_id: str
    label: str


def candidate(evidence_id: str, label: str | None = None) -> Candidate:
    return Candidate(evidence_id, label or evidence_id)


def union(*lanes: RankedEvidenceLane[Candidate]):
    return post_selection_lane_union(
        lanes,
        evidence_id=lambda item: item.evidence_id,
    )


def test_selects_each_lane_before_cross_lane_dedup_and_refills_its_vacancy() -> None:
    dense_shared = candidate("shared", "dense-shared")
    lexical_shared = candidate("shared", "lexical-shared")
    result = union(
        RankedEvidenceLane(
            "dense",
            2,
            (candidate("dense-1"), dense_shared, candidate("dense-tail")),
        ),
        RankedEvidenceLane(
            "lexical",
            2,
            (
                lexical_shared,
                candidate("lexical-1"),
                candidate("lexical-refill"),
            ),
        ),
    )

    assert result.evidence_ids == (
        "dense-1",
        "shared",
        "lexical-1",
        "lexical-refill",
    )
    assert result.items[1] is dense_shared
    lexical = result.lane_selections[1]
    assert lexical.selected_before_dedup == (
        lexical_shared,
        candidate("lexical-1"),
    )
    assert lexical.dedup_excluded_evidence_ids == ("shared",)
    assert tuple(item.evidence_id for item in lexical.refilled) == (
        "lexical-refill",
    )
    assert lexical.unfilled_slots == 0


def test_all_initial_selections_have_priority_over_any_refill() -> None:
    result = union(
        RankedEvidenceLane("protected", 1, (candidate("duplicate"),)),
        RankedEvidenceLane(
            "dense",
            2,
            (
                candidate("duplicate"),
                candidate("dense-initial"),
                candidate("lexical-initial"),
                candidate("dense-refill"),
            ),
        ),
        RankedEvidenceLane(
            "lexical",
            1,
            (candidate("lexical-initial"), candidate("lexical-tail")),
        ),
    )

    assert result.evidence_ids == (
        "duplicate",
        "dense-initial",
        "lexical-initial",
        "dense-refill",
    )
    dense = result.lane_selections[1]
    assert dense.dedup_excluded_evidence_ids == ("duplicate",)
    assert dense.refill_skipped_evidence_ids == ("lexical-initial",)
    assert tuple(item.evidence_id for item in dense.refilled) == ("dense-refill",)


def test_duplicate_within_one_lane_is_deduped_then_refilled() -> None:
    result = union(
        RankedEvidenceLane(
            "episode",
            3,
            (
                candidate("episode-1", "first-copy"),
                candidate("episode-1", "second-copy"),
                candidate("episode-2"),
                candidate("episode-3"),
            ),
        )
    )

    assert result.evidence_ids == ("episode-1", "episode-2", "episode-3")
    selection = result.lane_selections[0]
    assert selection.dedup_excluded_evidence_ids == ("episode-1",)
    assert tuple(item.evidence_id for item in selection.refilled) == ("episode-3",)


def test_budget_is_not_borrowed_from_a_lane_with_unused_capacity() -> None:
    result = union(
        RankedEvidenceLane("empty", 2, (candidate("only-one"),)),
        RankedEvidenceLane(
            "surplus",
            1,
            (candidate("surplus-1"), candidate("surplus-2")),
        ),
    )

    assert result.evidence_ids == ("only-one", "surplus-1")
    assert result.lane_selections[0].unfilled_slots == 1
    assert result.lane_selections[1].unfilled_slots == 0


def test_identity_is_exact_and_output_is_repeatable() -> None:
    lanes = (
        RankedEvidenceLane(
            "exact",
            3,
            (candidate("Fact"), candidate("fact"), candidate("Fact ")),
        ),
    )

    first = union(*lanes)
    second = union(*lanes)

    assert first == second
    assert first.evidence_ids == ("Fact", "fact", "Fact ")


def test_zero_budget_does_not_read_or_emit_candidates() -> None:
    result = post_selection_lane_union(
        (RankedEvidenceLane("off", 0, (object(),)),),
        evidence_id=lambda _item: (_ for _ in ()).throw(AssertionError()),
    )

    assert result.items == ()
    assert result.lane_selections[0].selected_before_dedup == ()


@pytest.mark.parametrize("budget", [-1, True, 1.5])
def test_rejects_invalid_budgets(budget: object) -> None:
    error = TypeError if isinstance(budget, (bool, float)) else ValueError
    with pytest.raises(error):
        RankedEvidenceLane("dense", budget, ())  # type: ignore[arg-type]


def test_rejects_duplicate_lane_ids_and_invalid_evidence_ids() -> None:
    with pytest.raises(ValueError, match="duplicate lane_id"):
        union(
            RankedEvidenceLane("dense", 0, ()),
            RankedEvidenceLane("dense", 0, ()),
        )

    with pytest.raises(ValueError, match="non-empty string"):
        union(RankedEvidenceLane("dense", 1, (candidate(""),)))

    with pytest.raises(TypeError, match="must return a string"):
        post_selection_lane_union(
            (RankedEvidenceLane("dense", 1, (candidate("dense"),)),),
            evidence_id=lambda _item: 1,  # type: ignore[return-value]
        )
