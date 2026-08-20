"""Validated parameters and mutable state shared by graph-search phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from memory_condense.application.query_routing import role_aware_results
from memory_condense.domain.ranking import round_robin_unique
from memory_condense.domain.schemas import RetrievalResult


def _round_robin_unique(
    groups: Sequence[Sequence[RetrievalResult]],
    limit: int,
    seen: set[str],
    *,
    stop_on_stall: bool,
) -> list[RetrievalResult]:
    """Interleave groups position-by-position, skipping seen chunk IDs.

    ``stop_on_stall=True`` gives up the first time a full position yields
    nothing new; ``False`` keeps scanning until every group is exhausted.
    """

    return round_robin_unique(
        groups,
        limit,
        key=lambda result: result.chunk.chunk_id,
        seen=seen,
        stop_on_stall=stop_on_stall,
    )


@dataclass(frozen=True, slots=True)
class _GraphSearchParams:
    """Validated tunables for one ``search_hybrid_graph`` invocation.

    The public keyword surface stays on the method; this object centralizes
    the cross-parameter validation and travels into the phase helpers so
    they do not each take a dozen loose arguments.
    """

    k: int
    neighbor_radius: int
    neighbor_slots: int
    neighbor_direction: str
    source_slots: int
    source_candidate_pool: int
    source_activation_k: int | None
    query_facet_retrieval: bool
    query_facet_slots: int
    query_facet_max: int
    role_aware_retrieval: bool
    role_user_weight: float
    role_assistant_weight: float
    role_system_weight: float
    multi_fact_source_diversity: bool
    source_tfisf_activation: bool
    source_tfisf_slots: int
    source_hsc_activation: bool
    source_hsc_slots: int
    source_hsc_hops: int
    source_hsc_chunk_slots: int
    source_partition_routing: bool
    source_partition_slots: int
    source_partition_separator: str
    source_local_search: bool
    use_source_reranker: bool
    use_attention_feedback: bool
    feedback_slots: int
    feedback_seed_slots: int
    feedback_evidence_tokens: int
    feedback_query_tokens: int
    ef_search: int
    candidates: int
    alpha: float

    def __post_init__(self) -> None:
        if (
            self.neighbor_radius < 0
            or self.neighbor_slots < 0
            or self.source_slots < 0
        ):
            raise ValueError("graph retrieval bounds must be non-negative")
        if self.use_attention_feedback and (
            self.feedback_slots < 0 or self.feedback_slots > self.source_slots
        ):
            raise ValueError("feedback_slots must lie in [0, source_slots]")
        if self.feedback_seed_slots < 1:
            raise ValueError("feedback_seed_slots must be positive")
        if self.feedback_evidence_tokens < 1 or self.feedback_query_tokens < 1:
            raise ValueError("feedback token caps must be positive")
        if self.neighbor_direction not in {"both", "previous", "next"}:
            raise ValueError("invalid neighbor_direction")
        if self.source_candidate_pool < self.k:
            raise ValueError("source_candidate_pool must be at least k")
        if self.query_facet_slots < 0 or (
            self.query_facet_retrieval
            and self.query_facet_slots > self.source_slots
        ):
            raise ValueError("query_facet_slots must lie in [0, source_slots]")
        if self.query_facet_max < 1:
            raise ValueError("query_facet_max must be positive")
        if min(
            self.role_user_weight,
            self.role_assistant_weight,
            self.role_system_weight,
        ) < 0.0:
            raise ValueError("role weights must be non-negative")
        if self.source_tfisf_slots < 1:
            raise ValueError("source_tfisf_slots must be positive")
        if self.source_hsc_slots < 1 or self.source_hsc_hops < 1:
            raise ValueError("source HSC slots and hops must be positive")
        if self.source_hsc_activation and (
            self.source_hsc_chunk_slots < 1
            or self.source_hsc_chunk_slots > self.source_slots
        ):
            raise ValueError(
                "source_hsc_chunk_slots must lie in [1, source_slots]"
            )
        if (
            self.query_facet_retrieval
            and self.source_hsc_activation
            and self.query_facet_slots + self.source_hsc_chunk_slots
            > self.source_slots
        ):
            raise ValueError("facet and HSC reserves cannot exceed source_slots")
        activation_k = self.activation_k
        if activation_k < self.k or activation_k > self.source_candidate_pool:
            raise ValueError("source_activation_k must be between k and the pool")
        if self.use_source_reranker and not self.source_local_search:
            raise ValueError("source reranking requires source_local_search")
        if self.use_attention_feedback and not self.source_local_search:
            raise ValueError("attention feedback requires source_local_search")
        if self.use_source_reranker and self.use_attention_feedback:
            raise ValueError("reranking and attention feedback are separate arms")
        if self.source_partition_routing and self.source_partition_slots < 1:
            raise ValueError("source_partition_slots must be positive")
        if self.source_partition_routing and not self.source_partition_separator:
            raise ValueError("source_partition_separator must be non-empty")

    @property
    def activation_k(self) -> int:
        return (
            self.k
            if self.source_activation_k is None
            else self.source_activation_k
        )

    @property
    def pool_size(self) -> int:
        return max(self.k, self.source_candidate_pool)

    def role_weighted(
        self,
        text: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Apply role weighting when enabled; identity otherwise."""

        if not self.role_aware_retrieval:
            return results
        return role_aware_results(
            text,
            results,
            user_weight=self.role_user_weight,
            assistant_weight=self.role_assistant_weight,
            system_weight=self.role_system_weight,
        )


@dataclass(slots=True)
class _PartitionRouting:
    """State one partition-routed scan carries to the final snapshot binding."""

    routed_source_ids: list[str] | None = None
    partition_ids: list[str] = field(default_factory=list)
    candidates: list[RetrievalResult] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)
    routing_turn: int | None = None
    content_high_watermark: int | None = None


def _accumulate_source_activation(
    results: Sequence[RetrievalResult],
    anchor_by_source: dict[str, str],
    source_scores: dict[str, float],
) -> None:
    """Record each result's source anchor and best score in place."""

    for result in results:
        if result.turn is None:
            continue
        source_id = result.source_key
        anchor_by_source.setdefault(source_id, result.chunk.chunk_id)
        source_scores[source_id] = max(
            source_scores.get(source_id, 0.0),
            float(result.score),
        )


__all__ = []
