"""Private protocols and immutable scan receipts used by the condenser."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from typing import Any, Mapping, Protocol, Sequence

from memory_condense.associations.association_store import AssociationArtifact
from memory_condense.domain.schemas import RetrievalResult


class SourceCandidateReranker(Protocol):
    candidate_pool: int
    last_report: Any
    association_artifact: AssociationArtifact | None

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        top_k: int,
        unique_sources: bool = False,
    ) -> list[RetrievalResult]: ...

    def select(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        top_k: int,
    ) -> list[RetrievalResult]: ...


class SourceCompanionSelector(Protocol):
    """Optional transient chooser for ambiguous metadata-only sources."""

    def select_source_companions(
        self,
        query: str,
        candidates_by_source: Mapping[str, Sequence[RetrievalResult]],
    ) -> Mapping[str, RetrievalResult]: ...


@dataclass(frozen=True, slots=True)
class ActivePartitionHypothesis:
    """One bounded typed occurrence retained from a complete row scan."""

    chunk_id: str
    source_id: str
    timestamp: str | None
    ordinal: int
    surface_score: float
    identity_key: str | None


@dataclass(frozen=True, slots=True)
class ActivePartitionRoutingSnapshot:
    """Immutable audit bound to one selected-partition scan and DB turn."""

    routing_identity: str
    query_sha256: str
    transcript_turn: int
    content_high_watermark: int
    selected_partitions: tuple[str, ...]
    routed_source_ids: tuple[str, ...]
    frontier_chunk_ids: tuple[str, ...]
    frontier_routes: tuple[str, ...]
    active_frontier_rows: tuple[tuple[str, str], ...]
    active_partition_total: int
    active_partition_inspected: int
    active_partition_exhaustive: bool
    active_partition_sources_total: int
    active_partition_structural_rows: int
    active_partition_structural_hypotheses: int
    active_partition_candidates_admitted: int
    active_partition_candidates_already_present: int
    active_partition_candidates_replaced: int
    active_partition_candidates_truncated: int
    active_partition_structural_overflow: int
    active_partition_scan_contract: str
    active_partition_semantically_complete: bool
    partition_scope_kind: str
    partition_inventory_total: int
    selected_partition_count: int
    partition_scope_exhaustive: bool
    selected_scope_structurally_complete: bool
    global_semantic_complete: bool

    def matches(
        self,
        query: str,
        transcript_turn: int,
        content_high_watermark: int,
        candidates: Sequence[RetrievalResult],
    ) -> bool:
        return bool(
            transcript_turn == self.transcript_turn
            and content_high_watermark == self.content_high_watermark
            and hashlib.sha256(query.encode("utf-8")).hexdigest()
            == self.query_sha256
            and tuple(
                sorted(
                    (result.chunk.chunk_id, str(result.route or ""))
                    for result in candidates
                    if str(result.route or "").casefold().startswith(
                        "active_partition_"
                    )
                )
            )
            == self.active_frontier_rows
        )

    @classmethod
    def from_report(
        cls,
        report: Mapping[str, Any],
        **identity: Any,
    ) -> ActivePartitionRoutingSnapshot:
        """Bind one completed scan report to its immutable routing identity.

        The scan counters are read from ``report`` under their own field
        names and coerced to the declared field type; the identity and
        frontier bindings arrive as keyword arguments.
        """
        scan = {
            name: _SCAN_FIELD_TYPES[name](report[name])
            for name in _SCAN_FIELDS
        }
        return cls(**identity, **scan)

    def pack_fields(self) -> dict[str, Any]:
        return {
            "active_partition_total": self.active_partition_total,
            "active_partition_inspected": self.active_partition_inspected,
            "active_partition_scan": {
                field: getattr(self, field) for field in _SCAN_FIELDS
            },
        }


# The scan counters are every snapshot field that is not part of the routing
# identity or frontier binding; deriving the list keeps ``from_report`` and
# ``pack_fields`` in lockstep with the field declarations above.
_IDENTITY_FIELDS = frozenset(
    {
        "routing_identity",
        "query_sha256",
        "transcript_turn",
        "content_high_watermark",
        "selected_partitions",
        "routed_source_ids",
        "frontier_chunk_ids",
        "frontier_routes",
        "active_frontier_rows",
    }
)
_SCAN_FIELDS = tuple(
    item.name
    for item in fields(ActivePartitionRoutingSnapshot)
    if item.name not in _IDENTITY_FIELDS
)
_SCAN_FIELD_TYPES = {
    item.name: {"int": int, "bool": bool, "str": str}[item.type]
    for item in fields(ActivePartitionRoutingSnapshot)
    if item.name in _SCAN_FIELDS
}


__all__ = [
    "ActivePartitionHypothesis",
    "ActivePartitionRoutingSnapshot",
    "SourceCandidateReranker",
    "SourceCompanionSelector",
]
