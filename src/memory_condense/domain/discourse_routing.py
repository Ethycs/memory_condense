"""Text-free routing projections for discourse units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from memory_condense.domain._discourse_identity import (
    _confidence,
    _json_mapping,
    _labeled,
    _nonempty,
    _nonnegative,
    normalize_fields,
)
from memory_condense.domain.discourse import DiscourseUnit


@dataclass(frozen=True, slots=True)
class DiscourseUnitRoute:
    """The fields needed to route a unit without hydrating its evidence."""

    unit_id: str
    artifact_id: str
    kind: str
    canonical_key: str
    asserted_ordinal: int
    confidence: float
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        normalize_fields(
            self,
            unit_id=_nonempty,
            artifact_id=_nonempty,
            kind=_nonempty,
            canonical_key=_nonempty,
            asserted_ordinal=_nonnegative,
            confidence=_confidence,
            metadata=_labeled("unit route metadata", _json_mapping),
        )

    @classmethod
    def from_unit(cls, unit: DiscourseUnit) -> "DiscourseUnitRoute":
        return cls(
            unit_id=unit.unit_id,
            artifact_id=unit.artifact_id,
            kind=unit.kind,
            canonical_key=unit.canonical_key,
            asserted_ordinal=unit.asserted_ordinal,
            confidence=unit.confidence,
            metadata=unit.metadata,
        )

    def matches(self, unit: DiscourseUnit) -> bool:
        """Return whether a hydrated unit preserves the routed projection."""

        return self == type(self).from_unit(unit)


__all__ = ["DiscourseUnitRoute"]
