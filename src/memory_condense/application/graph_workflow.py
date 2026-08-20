"""Stateful hybrid graph retrieval, routing, and memory recall workflows."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from memory_condense.application.condenser_contracts import (
    ActivePartitionRoutingSnapshot as _ActivePartitionRoutingSnapshot,
)
from memory_condense.application.query_routing import (
    _retrieval_source_id,
    is_multi_fact_query,
    query_facets,
    rank_concept_members,
    role_aware_results,
    source_diverse_results,
    source_partition_ranking,
)
from memory_condense.domain._tokenizer import count_tokens, truncate_to_tokens
from memory_condense.domain.ranking import (
    DEFAULT_WEIGHTS,
    RankWeights,
    round_robin_unique,
)
from memory_condense.domain.schemas import MemoryResult, RetrievalResult


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
        self, text: str, results: list[RetrievalResult]
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
            source_scores.get(source_id, 0.0), float(result.score)
        )


class GraphWorkflowMixin:
    """Internal workflow methods composed by ``MemoryCondenser``."""

    def search_hybrid_graph(
        self,
        query: str,
        *,
        k: int = 10,
        neighbor_radius: int = 5,
        neighbor_slots: int = 24,
        neighbor_direction: Literal["both", "previous", "next"] = "next",
        source_slots: int = 24,
        source_candidate_pool: int = 200,
        source_activation_k: int | None = None,
        query_facet_retrieval: bool = False,
        query_facet_slots: int = 6,
        query_facet_max: int = 4,
        role_aware_retrieval: bool = False,
        role_user_weight: float = 1.25,
        role_assistant_weight: float = 0.75,
        role_system_weight: float = 0.50,
        multi_fact_source_diversity: bool = False,
        source_tfisf_activation: bool = False,
        source_tfisf_slots: int = 8,
        source_hsc_activation: bool = False,
        source_hsc_slots: int = 8,
        source_hsc_hops: int = 2,
        source_hsc_chunk_slots: int = 8,
        source_partition_routing: bool = False,
        source_partition_slots: int = 3,
        source_partition_separator: str = "::",
        source_local_search: bool = False,
        use_source_reranker: bool = False,
        use_attention_feedback: bool = False,
        feedback_slots: int = 12,
        feedback_seed_slots: int = 6,
        feedback_evidence_tokens: int = 48,
        feedback_query_tokens: int = 384,
        ef_search: int = 50,
        candidates: int = 100,
        alpha: float = 0.65,
    ) -> list[RetrievalResult]:
        """Union transition and source links behind immutable hybrid anchors.

        Results are ordered as anchors, bounded source-local transitions, then
        lower-ranked candidates from activated sources.  The caller's prompt
        cap remains the final hard byte/token boundary.
        """
        self.last_source_rerank_report = {}
        self._active_partition_routing_snapshot = None
        if k <= 0:
            return []
        params = _GraphSearchParams(
            k=k,
            neighbor_radius=neighbor_radius,
            neighbor_slots=neighbor_slots,
            neighbor_direction=neighbor_direction,
            source_slots=source_slots,
            source_candidate_pool=source_candidate_pool,
            source_activation_k=source_activation_k,
            query_facet_retrieval=query_facet_retrieval,
            query_facet_slots=query_facet_slots,
            query_facet_max=query_facet_max,
            role_aware_retrieval=role_aware_retrieval,
            role_user_weight=role_user_weight,
            role_assistant_weight=role_assistant_weight,
            role_system_weight=role_system_weight,
            multi_fact_source_diversity=multi_fact_source_diversity,
            source_tfisf_activation=source_tfisf_activation,
            source_tfisf_slots=source_tfisf_slots,
            source_hsc_activation=source_hsc_activation,
            source_hsc_slots=source_hsc_slots,
            source_hsc_hops=source_hsc_hops,
            source_hsc_chunk_slots=source_hsc_chunk_slots,
            source_partition_routing=source_partition_routing,
            source_partition_slots=source_partition_slots,
            source_partition_separator=source_partition_separator,
            source_local_search=source_local_search,
            use_source_reranker=use_source_reranker,
            use_attention_feedback=use_attention_feedback,
            feedback_slots=feedback_slots,
            feedback_seed_slots=feedback_seed_slots,
            feedback_evidence_tokens=feedback_evidence_tokens,
            feedback_query_tokens=feedback_query_tokens,
            ef_search=ef_search,
            candidates=candidates,
            alpha=alpha,
        )
        if (
            use_attention_feedback
            and self._source_candidate_reranker is None
        ):
            raise RuntimeError(
                "attention feedback requested but no Qwen controller is attached"
            )

        self.last_partition_routing_report = {}
        activation_k = params.activation_k

        query_embedding = self._embedder.embed_query(query)
        pool, anchors, routing = self._route_partitions(
            query, query_embedding, params
        )
        routed_source_ids = routing.routed_source_ids
        partition_ids = routing.partition_ids
        active_partition_report = routing.report
        if role_aware_retrieval:
            pool = params.role_weighted(query, pool)
            anchors = pool[:k]
        use_source_diversity = (
            multi_fact_source_diversity and is_multi_fact_query(query)
        )
        if use_source_diversity:
            pool = source_diverse_results(pool)
            anchors = pool[:k]
        self.last_source_diversity_report = {
            "enabled": multi_fact_source_diversity,
            "applied": use_source_diversity,
            "activated_prefix_sources": len(
                {
                    result.source_key
                    for result in pool[:activation_k]
                    if result.turn is not None
                }
            ),
        }

        (
            concept_results,
            concept_member_results,
            concept_name,
        ) = self._concept_activation(
            query,
            query_embedding,
            params,
            routed_source_ids=routed_source_ids,
            use_source_diversity=use_source_diversity,
        )

        facet_results = self._facet_candidates(
            query,
            params,
            routed_source_ids=routed_source_ids,
            anchors=anchors,
        )
        expanded = self.expand_source_neighbors(
            anchors,
            radius=neighbor_radius,
        )
        facet_ids = {result.chunk.chunk_id for result in facet_results}
        neighbors = [
            result
            for result in expanded[len(anchors) :]
            if result.chunk.chunk_id not in facet_ids
            and (
                neighbor_direction == "both"
                or result.transition_direction == neighbor_direction
            )
        ][:neighbor_slots]

        anchor_by_source: dict[str, str] = {}
        source_scores: dict[str, float] = {}
        first_pool_result_by_source: dict[str, RetrievalResult] = {}
        for result in [*pool, *facet_results, *concept_results]:
            if result.turn is None:
                continue
            first_pool_result_by_source.setdefault(result.source_key, result)
        for group in (pool[:activation_k], facet_results, concept_results):
            _accumulate_source_activation(group, anchor_by_source, source_scores)

        tfisf_ranked = (
            self._retriever.source_tfisf_query(
                query,
                k_sources=source_tfisf_slots,
            )
            if source_tfisf_activation
            else []
        )
        tfisf_admitted: list[str] = []
        for source_id, source_score in tfisf_ranked:
            candidate = first_pool_result_by_source.get(source_id)
            if candidate is None:
                continue
            if source_id not in anchor_by_source:
                tfisf_admitted.append(source_id)
            anchor_by_source.setdefault(source_id, candidate.chunk.chunk_id)
            source_scores[source_id] = max(
                source_scores.get(source_id, 0.0), float(source_score)
            )
        self.last_source_tfisf_report = {
            "enabled": source_tfisf_activation,
            "ranked_sources": [source_id for source_id, _score in tfisf_ranked],
            "admitted_sources": tfisf_admitted,
            "activated_sources": len(anchor_by_source),
        }

        hsc_ranked = (
            self._retriever.source_hsc_expand(
                query_embedding,
                list(anchor_by_source),
                slots=source_hsc_slots,
                hops=source_hsc_hops,
            )
            if source_hsc_activation
            else []
        )
        hsc_source_ids = [source_id for source_id, _score in hsc_ranked]
        hsc_source_scores = dict(hsc_ranked)
        hsc_anchor_by_source = {
            source_id: first_pool_result_by_source[source_id].chunk.chunk_id
            for source_id in hsc_source_ids
            if source_id in first_pool_result_by_source
        }
        self.last_source_hsc_report = {
            "enabled": source_hsc_activation,
            "seed_sources": len(anchor_by_source),
            "expanded_sources": hsc_source_ids,
            "pool_visible_sources": len(hsc_anchor_by_source),
            "reserved_chunk_slots": (
                source_hsc_chunk_slots if source_hsc_activation else 0
            ),
        }

        seen = {
            result.chunk.chunk_id
            for result in [*anchors, *facet_results, *neighbors]
        }
        regular_source_slots = source_slots - len(facet_results) - (
            source_hsc_chunk_slots if source_hsc_activation else 0
        )
        (
            concept_partition_results,
            concept_coverage_reserve,
        ) = self._concept_partition_coverage(
            query,
            params,
            concept_name=concept_name,
            partition_ids=partition_ids,
            regular_source_slots=regular_source_slots,
        )
        source_extras = self._source_local_extras(
            query,
            query_embedding,
            params,
            pool=pool,
            seen=seen,
            anchor_by_source=anchor_by_source,
            source_scores=source_scores,
            regular_source_slots=regular_source_slots,
            use_source_diversity=use_source_diversity,
            concept_member_results=concept_member_results,
            concept_partition_results=concept_partition_results,
            concept_coverage_reserve=concept_coverage_reserve,
        )
        hsc_extras: list[RetrievalResult] = []
        if source_hsc_activation and hsc_source_ids:
            hsc_extras = self._retriever.hybrid_query_sources(
                query,
                query_embedding,
                hsc_source_ids,
                k=source_hsc_chunk_slots,
                candidates_per_source=source_candidate_pool,
                alpha=alpha,
                source_scores=hsc_source_scores,
                anchor_chunk_ids=hsc_anchor_by_source,
                exclude_chunk_ids=tuple(
                    {
                        *seen,
                        *(result.chunk.chunk_id for result in source_extras),
                    }
                ),
            )
            hsc_extras = [
                result.model_copy(update={"route": "hsc_contraction"})
                for result in hsc_extras
            ]
            hsc_extras = params.role_weighted(query, hsc_extras)
            if use_source_diversity:
                hsc_extras = source_diverse_results(hsc_extras)
        source_extras = [*source_extras, *hsc_extras]

        if use_attention_feedback and feedback_slots:
            source_extras = self._attention_feedback(
                query,
                query_embedding,
                params,
                anchors=anchors,
                facet_results=facet_results,
                neighbors=neighbors,
                source_extras=source_extras,
            )
        baseline = [*anchors, *facet_results, *neighbors, *source_extras]
        return self._bind_active_partition(query, baseline, anchors, routing)

    def _concept_partition_coverage(
        self,
        query: str,
        params: _GraphSearchParams,
        *,
        concept_name: str | None,
        partition_ids: list[str],
        regular_source_slots: int,
    ) -> tuple[list[RetrievalResult], int]:
        """Reserve concept-member coverage inside the top two partitions."""
        concept_partition_results: list[RetrievalResult] = []
        concept_coverage_reserve = 0
        if (
            concept_name is not None
            and self._source_concept_artifact_id is not None
            and partition_ids
            and regular_source_slots > 0
        ):
            concept_coverage_reserve = min(
                regular_source_slots,
                max(2, math.ceil(regular_source_slots * 0.55)),
            )
            first_quota = math.ceil(concept_coverage_reserve * 2.0 / 3.0)
            partition_quotas = (
                first_quota,
                concept_coverage_reserve - first_quota,
            )
            for partition_id, quota in zip(
                partition_ids[:2], partition_quotas, strict=False
            ):
                if quota <= 0:
                    continue
                partition_sources = self._retriever.source_ids_in_partitions(
                    [partition_id],
                    separator=params.source_partition_separator,
                )
                hits = self._associations.concept_members(
                    self._source_concept_artifact_id,
                    concept_name,
                    top_k=max(quota, len(partition_sources)),
                    source_ids=partition_sources,
                    unique_sources=True,
                )
                partition_members: list[RetrievalResult] = []
                for hit in hits:
                    hydrated = self._retriever.hydrate_chunk(
                        hit.chunk_id,
                        score=hit.score,
                        route="cav_partition_coverage",
                    )
                    if hydrated is not None:
                        partition_members.append(hydrated)
                concept_partition_results.extend(
                    rank_concept_members(query, partition_members)[:quota]
                )
            self.last_source_diversity_report.update(
                {
                    "concept_coverage_reserve": concept_coverage_reserve,
                    "concept_partition_candidates": len(
                        concept_partition_results
                    ),
                }
            )
        return concept_partition_results, concept_coverage_reserve

    def _source_local_extras(
        self,
        query: str,
        query_embedding,
        params: _GraphSearchParams,
        *,
        pool: list[RetrievalResult],
        seen: set[str],
        anchor_by_source: dict[str, str],
        source_scores: dict[str, float],
        regular_source_slots: int,
        use_source_diversity: bool,
        concept_member_results: list[RetrievalResult],
        concept_partition_results: list[RetrievalResult],
        concept_coverage_reserve: int,
    ) -> list[RetrievalResult]:
        """Fill the regular source slots, optionally reranked.

        The fallback (non-local) branch records its picks in ``seen`` in
        place so the later HSC exclusion set stays complete.
        """
        if params.source_local_search:
            if params.use_source_reranker and self._source_candidate_reranker is None:
                raise RuntimeError("source reranking requested but no reranker is attached")
            result_limit = regular_source_slots
            if params.use_source_reranker:
                result_limit = max(
                    result_limit,
                    self._source_candidate_reranker.candidate_pool,
                )
            source_extras = self._retriever.hybrid_query_sources(
                query,
                query_embedding,
                list(anchor_by_source),
                k=result_limit,
                candidates_per_source=params.source_candidate_pool,
                alpha=params.alpha,
                source_scores=source_scores,
                anchor_chunk_ids=anchor_by_source,
                exclude_chunk_ids=tuple(seen),
            )
            source_extras = params.role_weighted(query, source_extras)
            if use_source_diversity:
                source_extras = source_diverse_results(source_extras)
            if params.use_source_reranker:
                rerank_candidates = source_extras
                if use_source_diversity:
                    # Set queries need a safe scalar control plus attention
                    # exploration. Build the control from the narrower source
                    # frontier that supplies the output budget and auxiliary
                    # source routes; broader activation is visible only to the
                    # Qwen reserve and cannot evict this protected prefix.
                    protected_activation_k = min(
                        params.activation_k,
                        params.source_slots
                        + (params.source_tfisf_slots if params.source_tfisf_activation else 0)
                        + (params.source_hsc_slots if params.source_hsc_activation else 0)
                        + 1,
                    )
                    protected_anchor_by_source: dict[str, str] = {}
                    protected_source_scores: dict[str, float] = {}
                    _accumulate_source_activation(
                        pool[:protected_activation_k],
                        protected_anchor_by_source,
                        protected_source_scores,
                    )
                    protected_extras = self._retriever.hybrid_query_sources(
                        query,
                        query_embedding,
                        list(protected_anchor_by_source),
                        k=regular_source_slots,
                        candidates_per_source=params.source_candidate_pool,
                        alpha=params.alpha,
                        source_scores=protected_source_scores,
                        anchor_chunk_ids=protected_anchor_by_source,
                        exclude_chunk_ids=tuple(seen),
                    )
                    protected_extras = params.role_weighted(query, protected_extras)
                    protected_extras = source_diverse_results(protected_extras)
                    protected_ids = {
                        result.chunk.chunk_id for result in protected_extras
                    }
                    concept_member_ids = {
                        result.chunk.chunk_id for result in concept_member_results
                    }
                    rerank_candidates = [
                        *protected_extras,
                        *concept_member_results,
                        *[
                            result
                            for result in source_extras
                            if result.chunk.chunk_id not in protected_ids
                            and result.chunk.chunk_id not in concept_member_ids
                        ],
                    ]
                    self.last_source_diversity_report.update(
                        {
                            "protected_activation_k": protected_activation_k,
                            "protected_candidates": len(protected_extras),
                            "attention_exploration_candidates": max(
                                0,
                                len(rerank_candidates) - len(protected_extras),
                            ),
                        }
                    )
                source_extras = self._source_candidate_reranker.rerank(
                    query,
                    rerank_candidates,
                    top_k=regular_source_slots,
                    unique_sources=use_source_diversity,
                )
                if use_source_diversity and concept_partition_results:
                    scalar_reserve = max(
                        0,
                        regular_source_slots - concept_coverage_reserve,
                    )
                    combined = [
                        *protected_extras[:scalar_reserve],
                        *concept_partition_results,
                        *source_extras,
                    ]
                    source_unique: list[RetrievalResult] = []
                    seen_sources: set[str] = set()
                    for result in combined:
                        source_id = _retrieval_source_id(result)
                        if source_id in seen_sources:
                            continue
                        seen_sources.add(source_id)
                        source_unique.append(result)
                        if len(source_unique) >= regular_source_slots:
                            break
                    source_extras = source_unique
                report = self._source_candidate_reranker.last_report
                self.last_source_rerank_report = (
                    report.model_dump() if report is not None else {}
                )
        else:
            source_extras = []
            for result in pool:
                if regular_source_slots == 0:
                    break
                if result.turn is None or result.chunk.chunk_id in seen:
                    continue
                source_id = result.source_key
                anchor_id = anchor_by_source.get(source_id)
                if anchor_id is None:
                    continue
                source_extras.append(
                    result.model_copy(
                        update={
                            "route": "hybrid_source",
                            "anchor_chunk_id": anchor_id,
                        }
                    )
                )
                seen.add(result.chunk.chunk_id)
                if len(source_extras) >= regular_source_slots:
                    break
        return source_extras

    def _concept_activation(
        self,
        query: str,
        query_embedding,
        params: _GraphSearchParams,
        *,
        routed_source_ids: list[str] | None,
        use_source_diversity: bool,
    ) -> tuple[list[RetrievalResult], list[RetrievalResult], str | None]:
        """Activate CAV concept members inside the routed sources."""
        concept_results: list[RetrievalResult] = []
        concept_member_results: list[RetrievalResult] = []
        concept_name: str | None = None
        if (
            use_source_diversity
            and routed_source_ids
            and self._source_concept_artifact_id is not None
        ):
            artifact = self._associations.get_artifact(
                self._source_concept_artifact_id
            )
            if artifact is not None and artifact.concept_names:
                concept_name = (
                    "autobiographical_completed_event"
                    if "autobiographical_completed_event" in artifact.concept_names
                    else artifact.concept_names[0]
                )
                member_hits = self._associations.concept_members(
                    artifact.artifact_id,
                    concept_name,
                    top_k=max(params.source_slots * 4, params.source_slots),
                    source_ids=routed_source_ids,
                    unique_sources=True,
                )
                concept_member_results = [
                    hydrated.model_copy(update={"route": "cav_concept_member"})
                    for hit in member_hits
                    if (
                        hydrated := self._retriever.hydrate_chunk(
                            hit.chunk_id,
                            score=hit.score,
                            route="cav_concept_member",
                        )
                    )
                    is not None
                ]
                concept_source_ids = [
                    hit.source_id for hit in member_hits if hit.source_id is not None
                ]
                if concept_source_ids:
                    concept_results = self._retriever.hybrid_query_sources(
                        query,
                        query_embedding,
                        concept_source_ids,
                        k=max(params.source_slots, params.k),
                        candidates_per_source=params.source_candidate_pool,
                        alpha=params.alpha,
                    )
                    concept_results = params.role_weighted(query, concept_results)
                    concept_results = source_diverse_results(concept_results)
        self.last_source_diversity_report.update(
            {
                "concept_activation": concept_name,
                "concept_candidates": len(concept_results),
                "concept_members": len(concept_member_results),
            }
        )
        return concept_results, concept_member_results, concept_name

    def _facet_candidates(
        self,
        query: str,
        params: _GraphSearchParams,
        *,
        routed_source_ids: list[str] | None,
        anchors: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Reserve slots for per-facet retrieval on multi-facet queries."""
        facets = (
            query_facets(query, max_facets=params.query_facet_max)
            if params.query_facet_retrieval and params.query_facet_slots
            else []
        )
        facet_groups: list[list[RetrievalResult]] = []
        if facets:
            per_facet = max(
                2,
                2 * ((params.query_facet_slots + len(facets) - 1) // len(facets)),
            )
            for facet in facets:
                facet_embedding = self._embedder.embed_query(facet)
                if routed_source_ids is not None:
                    facet_pool = self._retriever.hybrid_query_sources(
                        facet,
                        facet_embedding,
                        routed_source_ids,
                        k=per_facet,
                        candidates_per_source=params.source_candidate_pool,
                        alpha=params.alpha,
                    )
                else:
                    facet_pool = self.search_hybrid_from_embedding(
                        facet,
                        facet_embedding,
                        k=per_facet,
                        ef_search=params.ef_search,
                        candidates=max(params.candidates, per_facet),
                        alpha=params.alpha,
                    )
                facet_pool = params.role_weighted(facet, facet_pool)
                facet_groups.append(facet_pool)

        facet_results = [
            result.model_copy(update={"route": "query_facet"})
            for result in _round_robin_unique(
                facet_groups,
                params.query_facet_slots,
                {result.chunk.chunk_id for result in anchors},
                stop_on_stall=False,
            )
        ]
        self.last_query_facet_report = {
            "enabled": params.query_facet_retrieval,
            "facets": len(facets),
            "reserved_slots": params.query_facet_slots if facets else 0,
            "candidates_added": len(facet_results),
        }
        return facet_results

    def _bind_active_partition(
        self,
        query: str,
        baseline: list[RetrievalResult],
        anchors: list[RetrievalResult],
        routing: _PartitionRouting,
    ) -> list[RetrievalResult]:
        """Admit the scanned partition candidates and seal the snapshot.

        Returns the final frontier.  A transcript or content change since
        the routed scan invalidates completeness instead of sealing a
        stale snapshot.
        """
        output = baseline
        report = routing.report
        if report.get("active_partition_scan_status") != "applied":
            return output
        output, admission = self._admit_active_partition_candidates(
            baseline,
            routing.candidates,
            anchor_chunk_ids={result.chunk.chunk_id for result in anchors},
            semantic_complete=bool(
                report.get(
                    "active_partition_semantically_complete",
                    False,
                )
            ),
        )
        scan_truncated = int(
            report.get(
                "active_partition_candidates_truncated",
                0,
            )
        )
        total_truncated = scan_truncated + int(
            admission["active_partition_candidates_truncated"]
        )
        admission["active_partition_candidates_truncated"] = total_truncated
        if total_truncated:
            report[
                "active_partition_semantically_complete"
            ] = False
            report[
                "selected_scope_structurally_complete"
            ] = False
            report["global_semantic_complete"] = False
        report.update(admission)
        self.last_partition_routing_report.update(report)

        frontier_ids = tuple(result.chunk.chunk_id for result in output)
        frontier_routes = tuple(str(result.route or "") for result in output)
        active_frontier_rows = tuple(
            sorted(
                (result.chunk.chunk_id, str(result.route or ""))
                for result in output
                if str(result.route or "").casefold().startswith(
                    "active_partition_"
                )
            )
        )
        transcript_turn = self._transcript.current_turn()
        content_high_watermark = self._content_high_watermark()
        if (
            transcript_turn != routing.routing_turn
            or content_high_watermark
            != routing.content_high_watermark
        ):
            invalidated_reason = (
                "transcript_advanced_during_route"
                if transcript_turn != routing.routing_turn
                else "content_changed_during_route"
            )
            report.update(
                {
                    "active_partition_scan_status": "invalidated",
                    "active_partition_exhaustive": False,
                    "active_partition_semantically_complete": False,
                    "selected_scope_structurally_complete": False,
                    "global_semantic_complete": False,
                    "active_partition_snapshot_invalidated_reason": (
                        invalidated_reason
                    ),
                }
            )
            self.last_partition_routing_report.update(
                {
                    **report,
                    "active_partition_snapshot_validated": False,
                }
            )
            return output

        query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
        source_set_sha256 = hashlib.sha256(
            "\0".join(routing.routed_source_ids or ()).encode("utf-8")
        ).hexdigest()
        identity_payload = "\0".join(
            [
                query_sha256,
                str(transcript_turn),
                str(content_high_watermark),
                source_set_sha256,
                *routing.partition_ids,
                *frontier_ids,
                *frontier_routes,
            ]
        )
        routing_identity = hashlib.sha256(
            identity_payload.encode("utf-8")
        ).hexdigest()
        # ``active_partition_candidates_truncated`` already carries
        # ``total_truncated`` — the admission update above wrote it into
        # the report before this binding.
        self._active_partition_routing_snapshot = (
            _ActivePartitionRoutingSnapshot.from_report(
                report,
                routing_identity=routing_identity,
                query_sha256=query_sha256,
                transcript_turn=transcript_turn,
                content_high_watermark=content_high_watermark,
                selected_partitions=tuple(routing.partition_ids),
                routed_source_ids=tuple(routing.routed_source_ids or ()),
                frontier_chunk_ids=frontier_ids,
                frontier_routes=frontier_routes,
                active_frontier_rows=active_frontier_rows,
            )
        )
        self.last_partition_routing_report.update(
            {
                "active_partition_routing_identity": routing_identity,
                "active_partition_snapshot_validated": False,
            }
        )
        return output

    def _route_partitions(
        self,
        query: str,
        query_embedding,
        params: _GraphSearchParams,
    ) -> tuple[
        list[RetrievalResult], list[RetrievalResult], _PartitionRouting
    ]:
        """Resolve the candidate pool and anchors, optionally partition-routed."""
        routing = _PartitionRouting()
        if not params.source_partition_routing:
            anchors = self.search_hybrid_from_embedding(
                query,
                query_embedding,
                k=params.k,
                ef_search=params.ef_search,
                candidates=params.candidates,
                alpha=params.alpha,
            )
            pool = self.search_hybrid_from_embedding(
                query,
                query_embedding,
                k=params.pool_size,
                ef_search=params.ef_search,
                candidates=max(params.candidates, params.pool_size),
                alpha=params.alpha,
            )
            return pool, anchors, routing

        coarse_pool = self.search_hybrid_from_embedding(
            query,
            query_embedding,
            k=params.pool_size,
            ef_search=params.ef_search,
            candidates=max(params.candidates, params.pool_size),
            alpha=params.alpha,
        )
        coarse_pool = params.role_weighted(query, coarse_pool)
        partition_ranking = source_partition_ranking(
            coarse_pool,
            separator=params.source_partition_separator,
        )
        routing.partition_ids = [
            str(item["partition"])
            for item in partition_ranking[: params.source_partition_slots]
        ]
        # Bind the complete scan to the database generation immediately
        # before resolving its immutable source set.  A concurrent append
        # anywhere in the remainder of the route invalidates completeness
        # instead of allowing a stale scan report to reach the packer.
        routing.routing_turn = self._transcript.current_turn()
        routing.content_high_watermark = self._content_high_watermark()
        routing.routed_source_ids = self._retriever.source_ids_in_partitions(
            routing.partition_ids,
            separator=params.source_partition_separator,
        )
        routing.candidates, routing.report = (
            self._scan_active_partition_frontier(
                query,
                query_embedding,
                routing.partition_ids,
                routing.routed_source_ids,
                separator=params.source_partition_separator,
            )
        )
        pool = self._retriever.hybrid_query_sources(
            query,
            query_embedding,
            routing.routed_source_ids,
            k=params.pool_size,
            candidates_per_source=params.source_candidate_pool,
            alpha=params.alpha,
        )
        self.last_partition_routing_report = {
            "coarse_candidates": len(coarse_pool),
            "selected_partitions": routing.partition_ids,
            "partition_ranking": partition_ranking,
            "routed_sources": len(routing.routed_source_ids),
            "routed_candidates": len(pool),
            **routing.report,
        }
        return pool, pool[: params.k], routing

    def _attention_feedback(
        self,
        query: str,
        query_embedding,
        params: _GraphSearchParams,
        *,
        anchors: list[RetrievalResult],
        facet_results: list[RetrievalResult],
        neighbors: list[RetrievalResult],
        source_extras: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Run the two-round Qwen activation-feedback pass.

        Returns the feedback-blended source extras and publishes the
        merged reranker report on ``last_source_rerank_report``.
        """
        # Sample every first-round route instead of letting the longest
        # route monopolize the Qwen workspace. IDs/scalars cross rounds;
        # request-token state does not.
        attention_candidates = _round_robin_unique(
            [
                list(anchors),
                list(facet_results),
                list(neighbors),
                list(source_extras),
            ],
            self._source_candidate_reranker.candidate_pool,
            set(),
            stop_on_stall=True,
        )

        seeds = self._source_candidate_reranker.select(
            query,
            attention_candidates,
            top_k=params.feedback_seed_slots,
        )
        initial_report = self._source_candidate_reranker.last_report
        feedback_source_ids: list[str] = []
        feedback_source_scores: dict[str, float] = {}
        feedback_anchor_by_source: dict[str, str] = {}
        for seed in seeds:
            if seed.turn is None:
                continue
            source_id = seed.source_key
            if source_id not in feedback_source_scores:
                feedback_source_ids.append(source_id)
                feedback_anchor_by_source[source_id] = seed.chunk.chunk_id
            feedback_source_scores[source_id] = max(
                feedback_source_scores.get(source_id, 0.0),
                float(seed.association_score or 0.0),
            )

        activation_pieces = [
            f"Original question: {truncate_to_tokens(query, 96)}",
            "Attended evidence:",
            *[
                truncate_to_tokens(seed.chunk.text, params.feedback_evidence_tokens)
                for seed in seeds
            ],
        ]
        activation_window = truncate_to_tokens(
            "\n".join(activation_pieces),
            params.feedback_query_tokens,
        )
        feedback_results: list[RetrievalResult] = []
        combined_report = None
        if feedback_source_ids:
            initial_ids = {
                result.chunk.chunk_id
                for result in [*anchors, *neighbors, *source_extras]
            }
            second_pool = self._retriever.hybrid_query_sources(
                query,
                query_embedding,
                feedback_source_ids,
                k=self._source_candidate_reranker.candidate_pool,
                candidates_per_source=params.source_candidate_pool,
                alpha=params.alpha,
                source_scores=feedback_source_scores,
                anchor_chunk_ids=feedback_anchor_by_source,
                exclude_chunk_ids=tuple(initial_ids),
            )
            activation_selected = self._source_candidate_reranker.select(
                activation_window,
                second_pool,
                top_k=params.feedback_slots,
            )
            combined_report = self._source_candidate_reranker.last_report
            activation_ids = {
                result.chunk.chunk_id for result in activation_selected
            }
            feedback_results = [
                result.model_copy(update={"route": "qwen_activation_feedback"})
                for result in activation_selected
            ]
            for result in second_pool:
                if len(feedback_results) >= params.feedback_slots:
                    break
                if result.chunk.chunk_id in activation_ids:
                    continue
                feedback_results.append(
                    result.model_copy(update={"route": "feedback_scalar_fill"})
                )

        protected_source_count = max(0, params.source_slots - params.feedback_slots)
        selected_source = list(source_extras[:protected_source_count])
        selected_ids = {result.chunk.chunk_id for result in selected_source}
        for result in feedback_results:
            if len(selected_source) >= params.source_slots:
                break
            if result.chunk.chunk_id in selected_ids:
                continue
            selected_source.append(result)
            selected_ids.add(result.chunk.chunk_id)
        for result in source_extras[protected_source_count:]:
            if len(selected_source) >= params.source_slots:
                break
            if result.chunk.chunk_id in selected_ids:
                continue
            selected_source.append(result)
            selected_ids.add(result.chunk.chunk_id)
        source_extras = selected_source
        initial_stats = (
            initial_report.model_dump() if initial_report is not None else {}
        )
        combined_stats = (
            combined_report.model_dump() if combined_report is not None else {}
        )
        self.last_source_rerank_report = dict(combined_stats)
        self.last_source_rerank_report.update(
            {
                "passes": int(initial_stats.get("passes", 0))
                + int(combined_stats.get("passes", 0)),
                "total_candidate_inspections": int(
                    initial_stats.get("total_candidate_inspections", 0)
                )
                + int(combined_stats.get("total_candidate_inspections", 0)),
                "max_workspace_candidates": max(
                    int(initial_stats.get("max_workspace_candidates", 0)),
                    int(combined_stats.get("max_workspace_candidates", 0)),
                ),
                "max_workspace_tokens": max(
                    int(initial_stats.get("max_workspace_tokens", 0)),
                    int(combined_stats.get("max_workspace_tokens", 0)),
                ),
                "qwen_candidates_added": int(
                    initial_stats.get("qwen_candidates_added", 0)
                )
                + int(combined_stats.get("qwen_candidates_added", 0)),
            }
        )
        self.last_source_rerank_report.update(
            {
                "feedback_rounds": 1,
                "feedback_seed_sources": len(feedback_source_ids),
                "feedback_candidates_added": sum(
                    result.route in {
                        "qwen_activation_feedback",
                        "feedback_scalar_fill",
                    }
                    for result in source_extras
                ),
                "feedback_activation_candidates": sum(
                    result.route == "qwen_activation_feedback"
                    for result in source_extras
                ),
                "feedback_query_tokens": count_tokens(activation_window),
            }
        )
        return source_extras

    def search_hybrid_neighbors(
        self,
        query: str,
        *,
        k: int = 10,
        radius: int = 1,
        max_neighbors: int | None = None,
        replacement_slots: int = 0,
        ef_search: int = 50,
        candidates: int = 100,
        alpha: float = 0.65,
    ) -> list[RetrievalResult]:
        """Retrieve hybrid anchors, then walk bounded source-local neighbors."""
        anchors = self.search_hybrid(
            query,
            k=k,
            ef_search=ef_search,
            candidates=candidates,
            alpha=alpha,
        )
        expanded = self.expand_source_neighbors(
            anchors,
            radius=radius,
            max_neighbors=max_neighbors,
        )
        if replacement_slots <= 0:
            return expanded
        neighbor_candidates = expanded[len(anchors) :]
        slots = min(replacement_slots, len(anchors), len(neighbor_candidates))
        return list(anchors[: len(anchors) - slots]) + neighbor_candidates[:slots]

    def expand_source_neighbors(
        self,
        anchors: Sequence[RetrievalResult],
        *,
        radius: int = 1,
        max_neighbors: int | None = None,
    ) -> list[RetrievalResult]:
        """Expand cached retrieval results without embedding the query again."""
        return self._retriever.hydrate_source_neighbors(
            anchors,
            radius=radius,
            max_neighbors=max_neighbors,
        )

    def recall_memories(
        self,
        query: str,
        k: int = 10,
        weights: RankWeights = DEFAULT_WEIGHTS,
        now_turn: int | None = None,
        min_energy: float = 0.0,
        reheat: bool = True,
    ) -> list[MemoryResult]:
        """Rank memory items for a query. Retrieved items are reheated.

        ``now_turn`` defaults to the transcript's current position, which is
        what callers want: decay is measured in turns, and the store knows
        where the conversation is. It is forwarded rather than dropped so a
        test or an ablation can evaluate the store as it would look N turns
        from now without appending N turns.
        """
        query_embedding = self._embedder.embed_query(query)
        return self._memory.retrieve(
            query_embedding,
            k=k,
            weights=weights,
            now_turn=now_turn,
            min_energy=min_energy,
            reheat=reheat,
        )


__all__ = ["GraphWorkflowMixin"]
