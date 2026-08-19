"""Value objects and stable identities for durable associations."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from memory_condense.domain.decay import decay_factor


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class AssociationArtifact:
    """Version identity for one compatible CAV/QK/OV artifact family."""

    artifact_id: str
    model_id: str
    checkpoint_id: str
    prefix_layers: int
    head_layer: int
    cav_layer: int | None
    concept_names: tuple[str, ...]
    head_count: int
    created_at: str = field(default_factory=_now_iso)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.concept_names)
        metadata = dict(self.metadata)
        if not self.artifact_id.strip():
            raise ValueError("artifact_id must be non-empty")
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if not self.checkpoint_id.strip():
            raise ValueError("checkpoint_id must be non-empty")
        if self.prefix_layers < 1:
            raise ValueError("prefix_layers must be positive")
        if not 0 <= self.head_layer < self.prefix_layers:
            raise ValueError("head_layer must be inside the loaded prefix")
        if self.cav_layer is not None and not 0 <= self.cav_layer < self.prefix_layers:
            raise ValueError("cav_layer must be inside the loaded prefix")
        if self.head_count < 1:
            raise ValueError("head_count must be positive")
        if any(not name.strip() for name in names) or len(set(names)) != len(names):
            raise ValueError("concept_names must be non-empty strings and unique")
        # Fail at registration time rather than halfway through an experiment.
        _canonical_json(metadata)
        object.__setattr__(self, "concept_names", names)
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def create(
        cls,
        *,
        model_id: str,
        checkpoint_id: str,
        prefix_layers: int,
        head_layer: int,
        cav_layer: int | None,
        concept_names: Sequence[str],
        head_count: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> AssociationArtifact:
        """Create a stable ID from every field that changes interpretation."""
        spec = {
            "model_id": model_id,
            "checkpoint_id": checkpoint_id,
            "prefix_layers": int(prefix_layers),
            "head_layer": int(head_layer),
            "cav_layer": None if cav_layer is None else int(cav_layer),
            "concept_names": list(concept_names),
            "head_count": int(head_count),
            "metadata": dict(metadata or {}),
        }
        digest = hashlib.sha256(_canonical_json(spec).encode("utf-8")).hexdigest()
        return cls(artifact_id=f"assoc-{digest[:24]}", **spec)

    def identity(self) -> tuple[Any, ...]:
        """Fields that must match when an artifact ID already exists."""
        return (
            self.model_id,
            self.checkpoint_id,
            self.prefix_layers,
            self.head_layer,
            self.cav_layer,
            self.concept_names,
            self.head_count,
            _canonical_json(dict(self.metadata)),
        )


@dataclass(frozen=True, slots=True)
class StoredCAVSignature:
    chunk_id: str
    artifact_id: str
    values: tuple[float, ...]
    created_turn: int
    access_count: int
    last_access_turn: int


@dataclass(frozen=True, slots=True)
class StoredCAVNeighbor:
    chunk_id: str
    score: float
    shared_concepts: tuple[str, ...]
    source_id: str | None = None


@dataclass(frozen=True, slots=True)
class StoredHeadEdge:
    source_chunk_id: str
    destination_chunk_id: str
    artifact_id: str
    head_weights: tuple[float, ...]
    qk_score: float
    ov_transport: float
    evidence_count: int
    traversal_count: int
    last_access_turn: int
    temporal_forward: bool | None

    def utility(
        self,
        *,
        now_turn: int,
        usage_half_life: float = 100.0,
        usage_weight: float = 0.05,
    ) -> float:
        """QK evidence + transported value + decayed live traversal evidence."""
        live_usage = math.log1p(self.traversal_count) * decay_factor(
            self.last_access_turn,
            now_turn,
            usage_half_life,
        )
        return self.qk_score + math.log1p(self.ov_transport) + usage_weight * live_usage


@dataclass(frozen=True, slots=True)
class HebbianUpdate:
    """Result of one idempotent same-turn co-access observation."""

    event_id: str
    created: bool
    concepts_observed: int
    edges_reinforced: int
    edges_pruned: int


@dataclass(frozen=True, slots=True)
class StoredHebbianNeighbor:
    """A conceptual chunk recalled through decayed co-access evidence."""

    chunk_id: str
    score: float
    support: int
    anchor_chunk_id: str
    coaccess_count: int
    last_reinforced_turn: int
