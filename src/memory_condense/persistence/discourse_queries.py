"""Streaming read queries over immutable discourse objects."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from memory_condense.domain.discourse_routing import DiscourseUnitRoute
from memory_condense.persistence.discourse_evidence import (
    _safe_metadata,
    _strict_json_object,
)


class _UnitQueryStore(Protocol):
    _db: object


class DiscourseQueryMixin:
    """Bounded-memory transformations over the discourse repository."""

    def iter_unit_routes_for_artifact(
        self: _UnitQueryStore,
        artifact_id: str,
    ) -> Iterator[DiscourseUnitRoute]:
        """Stream every text-free route in deterministic newest-first order.

        Closure can inspect the complete routing index while retaining only a
        bounded matched frontier. Evidence spans and source text are not read.
        """

        rows = self._db.execute(  # type: ignore[attr-defined]
            "SELECT unit_id, artifact_id, kind, canonical_key, "
            "asserted_ordinal, confidence, metadata "
            "FROM discourse_units WHERE artifact_id = ? "
            "ORDER BY asserted_ordinal DESC, unit_id",
            (artifact_id,),
        )
        for row in rows:
            metadata = _strict_json_object(row[6], label="unit metadata")
            _safe_metadata(metadata, label="unit metadata", owner="unit")
            yield DiscourseUnitRoute(
                unit_id=row[0],
                artifact_id=row[1],
                kind=row[2],
                canonical_key=row[3],
                asserted_ordinal=int(row[4]),
                confidence=float(row[5]),
                metadata=metadata,
            )


__all__ = ["DiscourseQueryMixin"]
