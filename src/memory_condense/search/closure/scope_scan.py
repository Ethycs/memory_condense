"""Exhaustive text-free artifact scans with bounded retained frontiers."""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass

from memory_condense.domain.discourse import (
    DiscourseRelation,
    DiscourseUnit,
    EvidenceSpan,
    QueryProgram,
)
from memory_condense.domain.discourse_routing import DiscourseUnitRoute
from memory_condense.search.closure.semantics import (
    unit_obligation_ids,
    unit_priority,
)
from memory_condense.search.closure.store import EvidenceClosureStore

#: Shared invariant message for any store probe returning more than requested.
PROBE_OVERFLOW_MESSAGE = "store returned more results than the requested probe limit"


def validate_requested_spans(
    spans: Sequence[EvidenceSpan],
    chunk_ids: Sequence[str],
) -> None:
    """Reject raw evidence returned for a chunk the caller did not request."""

    requested = set(chunk_ids)
    if any(span.chunk_id not in requested for span in spans):
        raise ValueError("chunk evidence contains an unrequested chunk")


def validate_chunk_scoped_rows(
    rows: Sequence[DiscourseUnit | DiscourseRelation],
    chunk_ids: Sequence[str],
    *,
    kind: str,
) -> None:
    """Require every indexed graph row to intersect the requested chunks."""

    requested = set(chunk_ids)
    if any(
        not any(span.chunk_id in requested for span in row.evidence)
        for row in rows
    ):
        raise ValueError(f"{kind} is not grounded in the requested chunks")


@dataclass(frozen=True, slots=True)
class ArtifactUnitScan:
    units: tuple[DiscourseUnit, ...]
    scanned_count: int
    matched_count: int
    scan_mode: str
    exhaustive: bool
    failure: str | None = None


def scan_artifact_units(
    store: EvidenceClosureStore,
    *,
    artifact_id: str,
    program: QueryProgram,
    max_units: int,
) -> ArtifactUnitScan:
    """Scan all routing units while retaining only the best exact matches."""

    stream_routes = getattr(store, "iter_unit_routes_for_artifact", None)
    streaming = callable(stream_routes)
    probe_limit = max_units + 1
    retained: list[
        tuple[tuple[int, int, float, int, str], DiscourseUnitRoute]
    ] = []
    scanned_count = 0
    matched_count = 0
    try:
        values = (
            stream_routes(artifact_id)
            if streaming
            else (
                DiscourseUnitRoute.from_unit(unit)
                for unit in store.units_for_artifact(
                    artifact_id,
                    limit=probe_limit,
                )
            )
        )
        for route in values:
            scanned_count += 1
            if not streaming and scanned_count > probe_limit:
                raise ValueError(PROBE_OVERFLOW_MESSAGE)
            if route.artifact_id != artifact_id:
                raise ValueError("artifact unit belongs to another artifact")
            if (
                program.as_of_ordinal is not None
                and route.asserted_ordinal > program.as_of_ordinal
            ):
                continue
            obligation_ids = unit_obligation_ids(
                route,
                program,
                connected=False,
            )
            if not obligation_ids:
                continue
            matched_count += 1
            priority = unit_priority(
                route,
                program,
                connected=True,
                route_rank=3,
            )
            bisect.insort(retained, (priority, route), key=lambda item: item[0])
            if len(retained) > max_units:
                retained.pop()
    except Exception as exc:  # fail closed; direct raw evidence remains usable
        return ArtifactUnitScan(
            units=(),
            scanned_count=scanned_count,
            matched_count=matched_count,
            scan_mode=(
                "exhaustive_stream" if streaming else "bounded_legacy_probe"
            ),
            exhaustive=False,
            failure=type(exc).__name__,
        )

    source_exhaustive = streaming or scanned_count <= max_units
    hydrated: list[DiscourseUnit] = []
    try:
        for _priority, route in retained:
            unit = store.get_unit(route.unit_id)
            if unit is None or not route.matches(unit):
                raise ValueError("routed artifact unit changed during hydration")
            hydrated.append(unit)
    except Exception as exc:  # keep the direct raw path, never trust stale routes
        return ArtifactUnitScan(
            units=(),
            scanned_count=scanned_count,
            matched_count=matched_count,
            scan_mode=(
                "exhaustive_stream" if streaming else "bounded_legacy_probe"
            ),
            exhaustive=False,
            failure=type(exc).__name__,
        )

    return ArtifactUnitScan(
        units=tuple(hydrated),
        scanned_count=scanned_count,
        matched_count=matched_count,
        scan_mode=(
            "exhaustive_stream" if streaming else "bounded_legacy_probe"
        ),
        exhaustive=matched_count <= max_units and source_exhaustive,
    )


__all__ = [
    "ArtifactUnitScan",
    "scan_artifact_units",
    "validate_chunk_scoped_rows",
    "validate_requested_spans",
]
