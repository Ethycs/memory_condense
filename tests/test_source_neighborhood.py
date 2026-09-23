"""Focused tests for deterministic source-local turn linking."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from memory_condense.search.source_neighborhood import (
    SourceChunkMetadata,
    SourceNeighborhoodIndex,
    source_neighborhood,
)


def _row(
    chunk: str,
    source: str,
    turn: str,
    ordinal: int,
    start_char: int = 0,
) -> SourceChunkMetadata:
    return SourceChunkMetadata(
        chunk_id=chunk,
        source_id=source,
        turn_id=turn,
        ordinal=ordinal,
        start_char=start_char,
    )


def test_collects_all_chunks_from_immediate_neighbor_turns_only() -> None:
    rows = [
        _row("before-b", "s1", "t0", 10),
        _row("before-a", "s1", "t0", 10),
        _row("seed", "s1", "t1", 20),
        _row("same-turn-peer", "s1", "t1", 20),
        _row("after-b", "s1", "t2", 40),
        _row("after-a", "s1", "t2", 40),
        _row("too-far", "s1", "t3", 50),
    ]

    result = source_neighborhood(rows, ["seed"])

    assert result.candidate_chunk_ids == (
        "before-a",
        "before-b",
        "after-a",
        "after-b",
    )
    assert result.source_order == ("s1",)
    assert result.seed_groups[0].seed_chunk_ids == ("seed",)
    assert [(link.linked_chunk_id, link.direction) for link in result.links] == [
        ("before-a", "predecessor_turn"),
        ("before-b", "predecessor_turn"),
        ("after-a", "successor_turn"),
        ("after-b", "successor_turn"),
    ]


def test_round_robins_candidate_depth_in_first_seen_source_order() -> None:
    rows = [
        _row("a-before-1", "source-a", "a0", 1),
        _row("a-before-2", "source-a", "a0", 1),
        _row("a-seed", "source-a", "a1", 3),
        _row("a-after", "source-a", "a2", 8),
        _row("b-before", "source-b", "b0", 2),
        _row("b-seed", "source-b", "b1", 5),
        _row("b-after", "source-b", "b2", 9),
    ]

    result = source_neighborhood(rows, ["b-seed", "a-seed"])

    assert result.source_order == ("source-b", "source-a")
    assert result.candidate_chunk_ids == (
        "b-before",
        "a-before-1",
        "b-after",
        "a-before-2",
        "a-after",
    )
    assert all(
        candidate.source_id
        == next(
            link.source_id
            for link in result.links
            if link.linked_chunk_id == candidate.chunk_id
        )
        for candidate in result.candidates
    )


def test_candidate_is_unique_but_audits_every_seed_link() -> None:
    rows = [
        _row("outside-left", "source", "t0", 0),
        _row("seed-left", "source", "t1", 1),
        _row("middle", "source", "t2", 2),
        _row("seed-right", "source", "t3", 3),
        _row("outside-right", "source", "t4", 4),
    ]

    result = source_neighborhood(rows, ["seed-left", "seed-right"])

    assert result.candidate_chunk_ids == (
        "outside-left",
        "middle",
        "outside-right",
    )
    middle_links = [
        link for link in result.links if link.linked_chunk_id == "middle"
    ]
    assert [
        (link.seed_chunk_id, link.direction) for link in middle_links
    ] == [
        ("seed-left", "successor_turn"),
        ("seed-right", "predecessor_turn"),
    ]
    assert result.seed_groups[0].seed_chunk_ids == (
        "seed-left",
        "seed-right",
    )


def test_seed_chunks_are_excluded_even_when_in_another_seeds_neighbor_turn() -> None:
    rows = [
        _row("seed-a", "source", "t1", 1),
        _row("peer", "source", "t1", 1),
        _row("seed-b", "source", "t2", 2),
        _row("other", "source", "t2", 2),
    ]

    result = source_neighborhood(rows, ["seed-a", "seed-b"])

    assert result.candidate_chunk_ids == ("other", "peer")
    assert not ({"seed-a", "seed-b"} & set(result.candidate_chunk_ids))


def test_linking_never_crosses_sources_even_at_adjacent_global_ordinals() -> None:
    rows = [
        _row("a-before", "a", "a0", 10),
        _row("a-seed", "a", "a1", 30),
        _row("a-after", "a", "a2", 80),
        _row("b-near", "b", "b0", 29),
        _row("b-nearer", "b", "b1", 31),
    ]

    result = source_neighborhood(rows, ["a-seed"])

    assert result.candidate_chunk_ids == ("a-before", "a-after")
    assert {candidate.source_id for candidate in result.candidates} == {"a"}
    assert {link.source_id for link in result.links} == {"a"}


def test_result_is_independent_of_metadata_input_order() -> None:
    rows = [
        _row("a", "s", "t0", 0),
        _row("b", "s", "t0", 0),
        _row("seed", "s", "t1", 1),
        _row("c", "s", "t2", 2),
    ]

    assert source_neighborhood(rows, ["seed"]) == source_neighborhood(
        list(reversed(rows)), ["seed"]
    )


def test_chunks_within_a_neighbor_turn_order_by_start_then_chunk_id() -> None:
    rows = [
        _row("chunk-z", "s", "t0", 0, 20),
        _row("chunk-b", "s", "t0", 0, 5),
        _row("chunk-a", "s", "t0", 0, 5),
        _row("seed", "s", "t1", 1, 0),
    ]

    assert source_neighborhood(rows, ["seed"]).candidate_chunk_ids == (
        "chunk-a",
        "chunk-b",
        "chunk-z",
    )


def test_empty_seeds_produce_an_empty_audited_result() -> None:
    result = source_neighborhood([_row("chunk", "source", "turn", 1)], [])
    assert result.candidates == ()
    assert result.seed_groups == ()
    assert result.links == ()
    assert result.source_order == ()


def test_index_compiles_once_and_is_isolated_from_caller_list_mutation() -> None:
    caller_rows = [
        _row("before", "s", "t0", 0),
        _row("seed", "s", "t1", 1),
        _row("after", "s", "t2", 2),
        _row("other", "other-source", "u0", 3),
    ]
    expected_bytes = sum(
        len(row.chunk_id.encode("utf-8"))
        + len(row.source_id.encode("utf-8"))
        + len(row.turn_id.encode("utf-8"))
        + 16
        for row in caller_rows
    )
    index = SourceNeighborhoodIndex(caller_rows)
    first = index.neighbors(["seed"])

    caller_rows.clear()
    caller_rows.append(_row("forged", "s", "forged-turn", 99))

    assert index.neighbors(["seed"]) == first
    assert first.candidate_chunk_ids == ("before", "after")
    assert index.chunk_count == 4
    assert index.turn_count == 4
    assert index.source_count == 2
    assert index.approx_resident_bytes == expected_bytes
    with pytest.raises(AttributeError, match="immutable"):
        index._source_count = 99


def test_metadata_and_results_are_immutable() -> None:
    seed = _row("seed", "s", "t1", 1)
    result = source_neighborhood(
        [_row("before", "s", "t0", 0), seed],
        ["seed"],
    )

    with pytest.raises(FrozenInstanceError):
        seed.ordinal = 9  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.candidates = ()  # type: ignore[misc]
    assert isinstance(result.candidates, tuple)
    assert isinstance(result.links, tuple)


@pytest.mark.parametrize(
    "seeds, message",
    [
        (["missing"], "absent from metadata"),
        (["seed", "seed"], "must be unique"),
        ([""], "non-empty"),
    ],
)
def test_rejects_invalid_seed_ids(seeds: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        source_neighborhood([_row("seed", "s", "t", 1)], seeds)


def test_rejects_duplicate_chunk_metadata() -> None:
    with pytest.raises(ValueError, match="duplicate metadata chunk_id"):
        source_neighborhood(
            [_row("same", "s", "t", 1), _row("same", "s", "t", 1)],
            [],
        )


def test_rejects_ambiguous_turn_coordinates() -> None:
    with pytest.raises(ValueError, match="conflicting source or ordinal"):
        source_neighborhood(
            [_row("a", "s", "turn", 1), _row("b", "s", "turn", 2)],
            [],
        )
    with pytest.raises(ValueError, match="multiple turns at the same ordinal"):
        source_neighborhood(
            [_row("a", "s", "turn-a", 1), _row("b", "s", "turn-b", 1)],
            [],
        )


def test_rejects_untyped_metadata_rows() -> None:
    with pytest.raises(TypeError, match="SourceChunkMetadata"):
        source_neighborhood([object()], [])  # type: ignore[list-item]
