"""Normalize caller-supplied closure programs and episode seeds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from memory_condense.domain.discourse import (
    ClosureScopeWitness,
    DiscourseSnapshot,
    EpisodeSeed,
    QueryProgram,
)
from memory_condense.search.closure.compiler import compile_query_program


@dataclass(frozen=True, slots=True)
class BoundedClosureInputs:
    seeds: tuple[EpisodeSeed, ...]
    direct_chunk_ids: tuple[str, ...]
    scope_witnesses: tuple[ClosureScopeWitness, ...]


def resolve_program(
    query: str | QueryProgram | None,
    explicit: QueryProgram | None,
) -> QueryProgram:
    if explicit is not None:
        if query is not None:
            raise ValueError("supply query or query_program, not both")
        return compile_query_program(explicit)
    if query is None:
        raise ValueError("query or query_program is required")
    return compile_query_program(query)


def normalize_seeds(seeds: Sequence[EpisodeSeed]) -> tuple[EpisodeSeed, ...]:
    values: dict[tuple[str, str], EpisodeSeed] = {}
    for seed in seeds:
        key = (seed.episode_id, seed.anchor_chunk_id)
        existing = values.get(key)
        if existing is None or (-seed.score, seed.route, seed.path) < (
            -existing.score,
            existing.route,
            existing.path,
        ):
            values[key] = seed
    return tuple(
        sorted(
            values.values(),
            key=lambda item: (
                -item.score,
                item.episode_id,
                item.anchor_chunk_id,
                item.route,
                item.path,
            ),
        )
    )


def bound_closure_inputs(
    seeds: Sequence[EpisodeSeed],
    direct_chunk_ids: Sequence[str],
    *,
    limit: int,
) -> BoundedClosureInputs:
    """Normalize caller routes and cap both families before any store read."""

    input_ceiling = limit * 2
    if len(seeds) > input_ceiling or len(direct_chunk_ids) > input_ceiling:
        raise ValueError(
            "caller closure inputs exceed the hard normalization ceiling of "
            f"{input_ceiling} items per route family"
        )
    normalized_seeds = normalize_seeds(seeds)
    normalized_direct = tuple(
        sorted(
            {
                str(chunk_id).strip()
                for chunk_id in direct_chunk_ids
                if str(chunk_id).strip()
            }
        )
    )
    selected_seeds = normalized_seeds[:limit]
    selected_direct = normalized_direct[:limit]
    omitted_seeds = normalized_seeds[limit:]
    omitted_direct = normalized_direct[limit:]
    return BoundedClosureInputs(
        seeds=selected_seeds,
        direct_chunk_ids=selected_direct,
        scope_witnesses=(
            ClosureScopeWitness(
                kind="closure_seed_inputs",
                subject_id="caller_episode_seeds",
                requested_limit=limit,
                returned_count=len(selected_seeds),
                exhaustive=not omitted_seeds,
                detail={
                    "input_count": len(normalized_seeds),
                    "input_ceiling": input_ceiling,
                    "omitted_count": len(omitted_seeds),
                    "omitted_seeds": tuple(
                        {
                            "episode_id": seed.episode_id,
                            "anchor_chunk_id": seed.anchor_chunk_id,
                        }
                        for seed in omitted_seeds
                    ),
                },
            ),
            ClosureScopeWitness(
                kind="closure_direct_inputs",
                subject_id="caller_direct_chunk_ids",
                requested_limit=limit,
                returned_count=len(selected_direct),
                exhaustive=not omitted_direct,
                detail={
                    "input_count": len(normalized_direct),
                    "input_ceiling": input_ceiling,
                    "omitted_count": len(omitted_direct),
                    "omitted_direct_chunk_ids": omitted_direct,
                },
            ),
        ),
    )


def resolve_artifact_id(
    snapshot: DiscourseSnapshot,
    requested: str | None,
    *,
    graph_requested: bool,
    has_explicit_seeds: bool,
) -> str | None:
    if requested is not None:
        value = str(requested).strip()
        if not value:
            raise ValueError("artifact_id must be non-empty when supplied")
        if value not in snapshot.artifact_ids:
            raise ValueError(
                f"artifact_id {value!r} is not present in the closure snapshot"
            )
        return value
    if len(snapshot.artifact_ids) == 1:
        return snapshot.artifact_ids[0]
    if len(snapshot.artifact_ids) > 1 and graph_requested:
        raise ValueError(
            "explicit artifact_id is required when graph closure could read a "
            "snapshot containing multiple artifacts"
        )
    if not snapshot.artifact_ids and has_explicit_seeds:
        raise ValueError("episode seeds require a discourse artifact in the snapshot")
    return None


__all__ = [
    "BoundedClosureInputs",
    "bound_closure_inputs",
    "normalize_seeds",
    "resolve_artifact_id",
    "resolve_program",
]
