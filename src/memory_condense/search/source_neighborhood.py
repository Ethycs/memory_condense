"""Deterministic local-turn links from retrieved chunks to their sources."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Sequence


LinkDirection = Literal["predecessor_turn", "successor_turn"]


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class SourceChunkMetadata:
    """The query-independent coordinates needed to link one raw chunk."""

    chunk_id: str
    source_id: str
    turn_id: str
    ordinal: int
    start_char: int

    def __post_init__(self) -> None:
        _nonempty(self.chunk_id, "chunk_id")
        _nonempty(self.source_id, "source_id")
        _nonempty(self.turn_id, "turn_id")
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int):
            raise TypeError("ordinal must be an integer")
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative")
        if isinstance(self.start_char, bool) or not isinstance(
            self.start_char, int
        ):
            raise TypeError("start_char must be an integer")
        if self.start_char < 0:
            raise ValueError("start_char must be non-negative")


@dataclass(frozen=True, slots=True)
class SourceSeedGroup:
    """Seeds from one source, preserving their first-seen query order."""

    source_id: str
    seeds: tuple[SourceChunkMetadata, ...]

    def __post_init__(self) -> None:
        _nonempty(self.source_id, "source_id")
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if not self.seeds:
            raise ValueError("a seed group cannot be empty")
        if any(seed.source_id != self.source_id for seed in self.seeds):
            raise ValueError("every grouped seed must belong to its source")
        if len({seed.chunk_id for seed in self.seeds}) != len(self.seeds):
            raise ValueError("grouped seeds must be unique")

    @property
    def seed_chunk_ids(self) -> tuple[str, ...]:
        return tuple(seed.chunk_id for seed in self.seeds)


@dataclass(frozen=True, slots=True)
class SourceNeighborhoodLink:
    """Audit edge from a seed turn to one same-source neighboring chunk."""

    seed_chunk_id: str
    linked_chunk_id: str
    source_id: str
    direction: LinkDirection

    def __post_init__(self) -> None:
        _nonempty(self.seed_chunk_id, "seed_chunk_id")
        _nonempty(self.linked_chunk_id, "linked_chunk_id")
        _nonempty(self.source_id, "source_id")
        if self.seed_chunk_id == self.linked_chunk_id:
            raise ValueError("a source-neighborhood link cannot point to its seed")
        if self.direction not in ("predecessor_turn", "successor_turn"):
            raise ValueError("direction must name an immediate neighboring turn")


@dataclass(frozen=True, slots=True)
class SourceNeighborhood:
    """Round-robin local candidates plus complete seed/link provenance."""

    candidates: tuple[SourceChunkMetadata, ...]
    seed_groups: tuple[SourceSeedGroup, ...]
    links: tuple[SourceNeighborhoodLink, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "seed_groups", tuple(self.seed_groups))
        object.__setattr__(self, "links", tuple(self.links))
        candidate_ids = tuple(row.chunk_id for row in self.candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("source-neighborhood candidates must be unique")
        source_ids = tuple(group.source_id for group in self.seed_groups)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source seed groups must be unique")

    @property
    def candidate_chunk_ids(self) -> tuple[str, ...]:
        return tuple(row.chunk_id for row in self.candidates)

    @property
    def source_order(self) -> tuple[str, ...]:
        return tuple(group.source_id for group in self.seed_groups)


class SourceNeighborhoodIndex:
    """Immutable source/turn topology compiled once for repeated queries."""

    __slots__ = (
        "_approx_resident_bytes",
        "_by_chunk_id",
        "_chunks_by_turn",
        "_locked",
        "_position_by_turn",
        "_source_count",
        "_turns_by_source",
    )

    def __init__(self, metadata_rows: Sequence[SourceChunkMetadata]) -> None:
        object.__setattr__(self, "_locked", False)
        rows = tuple(metadata_rows)
        if any(not isinstance(row, SourceChunkMetadata) for row in rows):
            raise TypeError(
                "metadata_rows must contain SourceChunkMetadata values"
            )

        by_chunk_id: dict[str, SourceChunkMetadata] = {}
        turn_coordinates: dict[str, tuple[str, int]] = {}
        ordinal_turns: dict[tuple[str, int], str] = {}
        mutable_chunks_by_turn: dict[str, list[SourceChunkMetadata]] = {}
        for row in rows:
            if row.chunk_id in by_chunk_id:
                raise ValueError(f"duplicate metadata chunk_id: {row.chunk_id}")
            by_chunk_id[row.chunk_id] = row
            coordinate = (row.source_id, row.ordinal)
            previous_coordinate = turn_coordinates.setdefault(
                row.turn_id, coordinate
            )
            if previous_coordinate != coordinate:
                raise ValueError(
                    "one turn_id has conflicting source or ordinal metadata"
                )
            previous_turn = ordinal_turns.setdefault(coordinate, row.turn_id)
            if previous_turn != row.turn_id:
                raise ValueError(
                    "one source has multiple turns at the same ordinal"
                )
            mutable_chunks_by_turn.setdefault(row.turn_id, []).append(row)

        mutable_turns_by_source: dict[str, list[str]] = {}
        for turn_id, (source_id, _ordinal) in turn_coordinates.items():
            mutable_turns_by_source.setdefault(source_id, []).append(turn_id)
        for source_id, turn_ids in mutable_turns_by_source.items():
            turn_ids.sort(
                key=lambda turn_id: (turn_coordinates[turn_id][1], turn_id)
            )
        chunks_by_turn = {
            turn_id: tuple(
                sorted(turn_rows, key=lambda row: (row.start_char, row.chunk_id))
            )
            for turn_id, turn_rows in mutable_chunks_by_turn.items()
        }
        turns_by_source = {
            source_id: tuple(turn_ids)
            for source_id, turn_ids in mutable_turns_by_source.items()
        }
        position_by_turn = {
            turn_id: position
            for turn_ids in turns_by_source.values()
            for position, turn_id in enumerate(turn_ids)
        }
        # Approximate immutable payload only. It intentionally excludes Python
        # dict/tuple/object headers, whose size varies by interpreter build.
        approximate_bytes = sum(
            len(row.chunk_id.encode("utf-8"))
            + len(row.source_id.encode("utf-8"))
            + len(row.turn_id.encode("utf-8"))
            + 16
            for row in rows
        )
        object.__setattr__(self, "_by_chunk_id", MappingProxyType(by_chunk_id))
        object.__setattr__(
            self, "_chunks_by_turn", MappingProxyType(chunks_by_turn)
        )
        object.__setattr__(
            self, "_turns_by_source", MappingProxyType(turns_by_source)
        )
        object.__setattr__(
            self, "_position_by_turn", MappingProxyType(position_by_turn)
        )
        object.__setattr__(self, "_source_count", len(turns_by_source))
        object.__setattr__(self, "_approx_resident_bytes", approximate_bytes)
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError(f"{type(self).__name__} is immutable")
        object.__setattr__(self, name, value)

    @property
    def chunk_count(self) -> int:
        return len(self._by_chunk_id)

    @property
    def turn_count(self) -> int:
        return len(self._chunks_by_turn)

    @property
    def source_count(self) -> int:
        return self._source_count

    @property
    def approx_resident_bytes(self) -> int:
        """UTF-8 coordinates plus 64-bit ordinal/start offsets per chunk."""

        return self._approx_resident_bytes

    def neighbors(self, seed_chunk_ids: Sequence[str]) -> SourceNeighborhood:
        """Link ordered seeds to immediate same-source turn neighbors.

        Seeds are grouped by their source's first appearance. Within each
        source, seed order is retained; each seed visits its predecessor turn
        before its successor turn. Candidate IDs are deduplicated within those
        source queues. Finally, queue depth is round-robined in first-seen
        source order, so a large source cannot consume all local coverage
        before another source contributes one chunk.

        Every seed is excluded from the candidate set. ``links`` retains all
        distinct seed-to-candidate relationships, including additional seeds
        that independently discover an already admitted candidate.
        """

        seeds = tuple(seed_chunk_ids)
        for seed_id in seeds:
            _nonempty(seed_id, "seed_chunk_id")
        if len(seeds) != len(set(seeds)):
            raise ValueError("seed chunk IDs must be unique")
        missing = tuple(
            seed_id for seed_id in seeds if seed_id not in self._by_chunk_id
        )
        if missing:
            raise ValueError(
                f"seed chunk ID is absent from metadata: {missing[0]}"
            )

        grouped_seeds: dict[str, list[SourceChunkMetadata]] = {}
        for seed_id in seeds:
            seed = self._by_chunk_id[seed_id]
            grouped_seeds.setdefault(seed.source_id, []).append(seed)
        seed_groups = tuple(
            SourceSeedGroup(source_id=source_id, seeds=tuple(source_seeds))
            for source_id, source_seeds in grouped_seeds.items()
        )
        if not seed_groups:
            return SourceNeighborhood(candidates=(), seed_groups=(), links=())

        seed_id_set = set(seeds)
        candidates_by_source: dict[str, list[SourceChunkMetadata]] = {}
        links_by_candidate: dict[str, list[SourceNeighborhoodLink]] = {}
        for group in seed_groups:
            source_turns = self._turns_by_source[group.source_id]
            source_candidates: list[SourceChunkMetadata] = []
            source_candidate_ids: set[str] = set()
            observed_links: set[tuple[str, str, LinkDirection]] = set()
            for seed in group.seeds:
                seed_position = self._position_by_turn[seed.turn_id]
                neighbors: tuple[tuple[int, LinkDirection], ...] = (
                    (seed_position - 1, "predecessor_turn"),
                    (seed_position + 1, "successor_turn"),
                )
                for neighbor_position, direction in neighbors:
                    if not 0 <= neighbor_position < len(source_turns):
                        continue
                    neighbor_turn = source_turns[neighbor_position]
                    for candidate in self._chunks_by_turn[neighbor_turn]:
                        if candidate.chunk_id in seed_id_set:
                            continue
                        link_key = (
                            seed.chunk_id,
                            candidate.chunk_id,
                            direction,
                        )
                        if link_key not in observed_links:
                            observed_links.add(link_key)
                            links_by_candidate.setdefault(
                                candidate.chunk_id, []
                            ).append(
                                SourceNeighborhoodLink(
                                    seed_chunk_id=seed.chunk_id,
                                    linked_chunk_id=candidate.chunk_id,
                                    source_id=group.source_id,
                                    direction=direction,
                                )
                            )
                        if candidate.chunk_id not in source_candidate_ids:
                            source_candidate_ids.add(candidate.chunk_id)
                            source_candidates.append(candidate)
            candidates_by_source[group.source_id] = source_candidates

        candidates: list[SourceChunkMetadata] = []
        depth = 0
        while True:
            admitted = False
            for group in seed_groups:
                source_candidates = candidates_by_source[group.source_id]
                if depth < len(source_candidates):
                    candidates.append(source_candidates[depth])
                    admitted = True
            if not admitted:
                break
            depth += 1

        links = tuple(
            link
            for candidate in candidates
            for link in links_by_candidate[candidate.chunk_id]
        )
        return SourceNeighborhood(
            candidates=tuple(candidates),
            seed_groups=seed_groups,
            links=links,
        )


def source_neighborhood(
    metadata_rows: Sequence[SourceChunkMetadata],
    seed_chunk_ids: Sequence[str],
) -> SourceNeighborhood:
    """Compile a one-use topology and return its seed neighborhood."""

    return SourceNeighborhoodIndex(metadata_rows).neighbors(seed_chunk_ids)


__all__ = [
    "LinkDirection",
    "SourceChunkMetadata",
    "SourceNeighborhood",
    "SourceNeighborhoodIndex",
    "SourceNeighborhoodLink",
    "SourceSeedGroup",
    "source_neighborhood",
]
