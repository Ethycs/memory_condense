"""Shared value objects for transient and compiled head memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class HeadMemoryItem:
    episode_id: str
    text: str
    keys: Any
    values: Any
    cav_signature: Any | None
    residual: Any | None = None
    association_residual: Any | None = None
    importance: float = 0.0
    created_turn: int = 0
    last_access_turn: int = 0
    access_count: int = 0
    qk_attention_mass: float = 0.0
    ov_transport: float = 0.0
    last_head_turn: int = 0
    pinned: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HeadAddress:
    indices: Any
    aggregate_scores: Any
    # Full external-attention probabilities: [query_heads, query_slots, memory_slots].
    head_weights: Any
    # OV input before W_O: [query_slots, query_heads, head_dim].
    mixed_values: Any
    slot_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class LiveMemoryHit:
    episode_id: str
    text: str
    score: float
    access_count: int
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LiveMemoryResult:
    hits: tuple[LiveMemoryHit, ...]
    hop_episode_ids: tuple[tuple[str, ...], ...]
    query_cav_signature: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class AssociativeMemoryCandidate:
    """A retrieval candidate tagged with the route that produced it."""

    episode_id: str
    text: str
    score: float = 0.0
    route: str = "hybrid"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AssociativeComposition:
    """Bounded result of recycling redundant direct-retrieval slots."""

    candidates: tuple[AssociativeMemoryCandidate, ...]
    duplicates_removed: int
    qk_added: int
    residual_added: int
    anchors_displaced: int


@dataclass(frozen=True, slots=True)
class MemoryLinkHit:
    """Compact evidence retained after a transient head-linking pass."""

    episode_id: str
    qk_score: float
    ov_transport: float
    head_weights: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    # Optional query-conditioned OV contribution used only by a live coverage
    # pass. It is a bounded CPU tensor and must never be written to the graph.
    transport_signature: Any | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class MemoryLinkResult:
    hits: tuple[MemoryLinkHit, ...]
    source_cav_signature: tuple[float, ...]
    workspace_candidates: int
    workspace_tokens: int
    passes: int = 1
    total_candidate_inspections: int = 0


@dataclass(frozen=True, slots=True)
class NestedMemoryInspection:
    """Finalists from fresh workspaces; no request-token state crosses hops."""

    hits: tuple[MemoryLinkHit, ...]
    passes: int
    max_workspace_candidates: int
    max_workspace_tokens: int
    total_candidate_inspections: int


@dataclass(slots=True)
class HeadAssociationEdge:
    source_id: str
    destination_id: str
    head_weights: Any
    score: float
    ov_transport: float = 0.0
    evidence_count: int = 1
    temporal_forward: bool | None = None
