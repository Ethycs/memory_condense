"""Compatibility facade for durable compact association storage.

The implementation is decomposed by responsibility into artifact, sparse-edge,
Hebbian, model, and repository modules.  Existing imports remain stable here.
"""

from __future__ import annotations

from memory_condense.associations.association_models import (
    AssociationArtifact,
    HebbianUpdate,
    StoredCAVNeighbor,
    StoredCAVSignature,
    StoredHeadEdge,
    StoredHebbianNeighbor,
    _canonical_json,
    _now_iso,
)
from memory_condense.associations.association_repository import AssociationStore


__all__ = [
    "AssociationArtifact",
    "AssociationStore",
    "HebbianUpdate",
    "StoredCAVNeighbor",
    "StoredCAVSignature",
    "StoredHeadEdge",
    "StoredHebbianNeighbor",
]
