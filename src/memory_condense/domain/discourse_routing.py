"""Text-free routing projections for discourse units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from memory_condense.domain._discourse_identity import (
    _confidence,
    _json_mapping,
    _nonempty,
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
        object.__setattr__(self, "unit_id", _nonempty(self.unit_id, "unit_id"))
        object.__setattr__(
            self,
            "artifact_id",
            _nonempty(self.artifact_id, "artifact_id"),
        )
        object.__setattr__(self, "kind", _nonempty(self.kind, "kind"))
        object.__setattr__(
            self,
            "canonical_key",
            _nonempty(self.canonical_key, "canonical_key"),
        )
        if self.asserted_ordinal < 0:
            raise ValueError("asserted_ordinal must be non-negative")
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(
            self,
            "metadata",
            _json_mapping(self.metadata, "unit route metadata"),
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
