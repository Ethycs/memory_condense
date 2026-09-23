from __future__ import annotations

from dataclasses import dataclass

import pytest

from memory_condense.search.adaptive_source_admission import (
    POLICY_ID,
    RankedSourceLane,
    source_balanced_surplus_admission,
)


@dataclass(frozen=True, slots=True)
class Candidate:
    evidence_id: str
    source_id: str
    route: str


def item(evidence_id: str, source_id: str, route: str = "parent") -> Candidate:
    return Candidate(evidence_id, source_id, route)


def select(
    parents: tuple[Candidate, ...],
    *lanes: RankedSourceLane[Candidate],
    budget: int,
    rrf_constant: int = 60,
    max_hits: int = 4,
):
    return source_balanced_surplus_admission(
        parents,
        lanes,
        evidence_id=lambda row: row.evidence_id,
        source_id=lambda row: row.source_id,
        surplus_budget=budget,
        rrf_constant=rrf_constant,
        max_hits_per_source_per_lane=max_hits,
    )


def test_preserves_parent_exact_order_and_covers_new_sources_before_seconds() -> None:
    parent = (item("p-1", "source-a"), item("p-2", "source-a"))
    result = select(
        parent,
        RankedSourceLane(
            "bm25",
            (
                item("a-tail", "source-a", "bm25"),
                item("b-1", "source-b", "bm25"),
                item("b-2", "source-b", "bm25"),
                item("c-1", "source-c", "bm25"),
            ),
        ),
        budget=2,
    )

    assert result.parent_items == parent
    assert result.items[: len(parent)] == parent
    assert result.evidence_ids[: len(parent)] == ("p-1", "p-2")
    assert {row.source_id for row in result.surplus_items} == {
        "source-b",
        "source-c",
    }
    assert tuple(row.phase for row in result.audit.selected) == (
        "new_source_representative",
        "new_source_representative",
    )
    assert "a-tail" not in result.evidence_ids


def test_duplicate_across_routes_accumulates_support_but_earliest_object_wins() -> None:
    lexical = item("shared", "source-b", "bm25")
    dense = item("shared", "source-b", "dense")
    result = select(
        (),
        RankedSourceLane("bm25", (lexical, item("other", "source-c", "bm25"))),
        RankedSourceLane("dense", (dense,)),
        budget=1,
    )

    assert result.surplus_items == (lexical,)
    assert result.evidence_ids == ("shared",)
    selected = result.audit.selected[0]
    assert selected.candidate_rrf == "2/61"
    assert [(row.lane_id, row.rank) for row in selected.origins] == [
        ("bm25", 1),
        ("dense", 1),
    ]


def test_source_rrf_caps_per_lane_size_bias_but_keeps_cross_lane_support() -> None:
    result = select(
        (),
        RankedSourceLane(
            "bm25",
            (
                item("a-1", "long-source", "bm25"),
                item("a-2", "long-source", "bm25"),
                item("a-3", "long-source", "bm25"),
                item("b-1", "corroborated-source", "bm25"),
            ),
        ),
        RankedSourceLane(
            "dense",
            (item("b-1", "corroborated-source", "dense"),),
        ),
        budget=1,
        max_hits=1,
    )

    assert result.surplus_items[0].source_id == "corroborated-source"
    groups = {row.source_id: row for row in result.audit.source_groups}
    assert groups["long-source"].contributing_hit_count == 1
    assert groups["corroborated-source"].contributing_hit_count == 2
    assert groups["corroborated-source"].contributing_lane_count == 2
    assert result.audit.max_hits_per_source_per_lane == 1


def test_second_pass_starts_only_after_every_new_source_has_a_representative() -> None:
    result = select(
        (item("parent", "source-a"),),
        RankedSourceLane(
            "bm25",
            (
                item("b-1", "source-b", "bm25"),
                item("b-2", "source-b", "bm25"),
                item("c-1", "source-c", "bm25"),
                item("a-2", "source-a", "bm25"),
            ),
        ),
        budget=4,
    )

    phases = tuple(row.phase for row in result.audit.selected)
    assert phases[:2] == (
        "new_source_representative",
        "new_source_representative",
    )
    assert phases[2:] == (
        "source_round_robin_surplus",
        "source_round_robin_surplus",
    )
    assert {row.source_id for row in result.surplus_items[:2]} == {
        "source-b",
        "source-c",
    }


def test_audit_projection_and_selection_are_repeatable() -> None:
    lanes = (
        RankedSourceLane(
            "dense",
            (item("x", "source-x", "dense"), item("y", "source-y", "dense")),
        ),
    )
    first = select((), *lanes, budget=2, rrf_constant=10, max_hits=2)
    second = select((), *lanes, budget=2, rrf_constant=10, max_hits=2)

    assert first == second
    projection = first.audit.projection()
    assert projection["policy_id"] == POLICY_ID
    assert projection["rrf_constant"] == 10
    assert projection["max_hits_per_source_per_lane"] == 2
    assert projection["parent_evidence_ids"] == []
    assert [row["evidence_id"] for row in projection["selected"]] == ["x", "y"]


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("budget", -1, ValueError),
        ("budget", True, TypeError),
        ("rrf_constant", -1, ValueError),
        ("rrf_constant", 1.5, TypeError),
        ("max_hits", 0, ValueError),
        ("max_hits", False, TypeError),
    ],
)
def test_rejects_invalid_controls(field: str, value: object, error: type[Exception]) -> None:
    controls = {"budget": 1, "rrf_constant": 60, "max_hits": 4}
    controls[field] = value
    with pytest.raises(error):
        select(
            (),
            RankedSourceLane("dense", (item("x", "source-x"),)),
            **controls,  # type: ignore[arg-type]
        )


def test_rejects_ambiguous_lanes_and_evidence_provenance() -> None:
    with pytest.raises(ValueError, match="duplicate lane_id"):
        select(
            (),
            RankedSourceLane("dense", ()),
            RankedSourceLane("dense", ()),
            budget=1,
        )
    with pytest.raises(ValueError, match="repeats evidence_id"):
        select(
            (),
            RankedSourceLane(
                "dense",
                (item("x", "source-x"), item("x", "source-x")),
            ),
            budget=1,
        )
    with pytest.raises(ValueError, match="conflicting sources"):
        select(
            (),
            RankedSourceLane("bm25", (item("x", "source-x"),)),
            RankedSourceLane("dense", (item("x", "source-y"),)),
            budget=1,
        )
    with pytest.raises(ValueError, match="duplicate parent"):
        select(
            (item("x", "source-x"), item("x", "source-x")),
            budget=0,
        )


def test_source_identifiers_are_used_as_exact_opaque_keys() -> None:
    result = select(
        (),
        RankedSourceLane(
            "bm25",
            (
                item("a", "prefix::answer_like", "bm25"),
                item("b", "prefix", "bm25"),
                item("c", "PREFIX::ANSWER_LIKE", "bm25"),
            ),
        ),
        budget=3,
    )

    assert {row.source_id for row in result.surplus_items} == {
        "prefix::answer_like",
        "prefix",
        "PREFIX::ANSWER_LIKE",
    }
