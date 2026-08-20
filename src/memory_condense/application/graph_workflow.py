"""Stateful hybrid graph retrieval, routing, and memory recall workflows."""

from __future__ import annotations

import hashlib
import math
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
        if neighbor_radius < 0 or neighbor_slots < 0 or source_slots < 0:
            raise ValueError("graph retrieval bounds must be non-negative")
        if use_attention_feedback and (
            feedback_slots < 0 or feedback_slots > source_slots
        ):
            raise ValueError("feedback_slots must lie in [0, source_slots]")
        if feedback_seed_slots < 1:
            raise ValueError("feedback_seed_slots must be positive")
        if feedback_evidence_tokens < 1 or feedback_query_tokens < 1:
            raise ValueError("feedback token caps must be positive")
        if neighbor_direction not in {"both", "previous", "next"}:
            raise ValueError("invalid neighbor_direction")
        if source_candidate_pool < k:
            raise ValueError("source_candidate_pool must be at least k")
        if query_facet_slots < 0 or (
            query_facet_retrieval and query_facet_slots > source_slots
        ):
            raise ValueError("query_facet_slots must lie in [0, source_slots]")
        if query_facet_max < 1:
            raise ValueError("query_facet_max must be positive")
        if min(role_user_weight, role_assistant_weight, role_system_weight) < 0.0:
            raise ValueError("role weights must be non-negative")
        if source_tfisf_slots < 1:
            raise ValueError("source_tfisf_slots must be positive")
        if source_hsc_slots < 1 or source_hsc_hops < 1:
            raise ValueError("source HSC slots and hops must be positive")
        if source_hsc_activation and (
            source_hsc_chunk_slots < 1 or source_hsc_chunk_slots > source_slots
        ):
            raise ValueError("source_hsc_chunk_slots must lie in [1, source_slots]")
        if (
            query_facet_retrieval
            and source_hsc_activation
            and query_facet_slots + source_hsc_chunk_slots > source_slots
        ):
            raise ValueError("facet and HSC reserves cannot exceed source_slots")
        activation_k = k if source_activation_k is None else source_activation_k
        if activation_k < k or activation_k > source_candidate_pool:
            raise ValueError("source_activation_k must be between k and the pool")
        if use_source_reranker and not source_local_search:
            raise ValueError("source reranking requires source_local_search")
        if use_attention_feedback and not source_local_search:
            raise ValueError("attention feedback requires source_local_search")
        if use_source_reranker and use_attention_feedback:
            raise ValueError("reranking and attention feedback are separate arms")
        if (
            use_attention_feedback
            and self._source_candidate_reranker is None
        ):
            raise RuntimeError(
                "attention feedback requested but no Qwen controller is attached"
            )
        if source_partition_routing and source_partition_slots < 1:
            raise ValueError("source_partition_slots must be positive")
        if source_partition_routing and not source_partition_separator:
            raise ValueError("source_partition_separator must be non-empty")

        self.last_partition_routing_report = {}

        def apply_role_weights(
            text: str, results: list[RetrievalResult]
        ) -> list[RetrievalResult]:
            return role_aware_results(
                text,
                results,
                user_weight=role_user_weight,
                assistant_weight=role_assistant_weight,
                system_weight=role_system_weight,
            )

        query_embedding = self._embedder.embed_query(query)
        pool_size = max(k, source_candidate_pool)
        routed_source_ids: list[str] | None = None
        active_partition_routing_turn: int | None = None
        active_partition_content_high_watermark: int | None = None
        partition_ids: list[str] = []
        active_partition_candidates: list[RetrievalResult] = []
        active_partition_report: dict[str, Any] = {}
        if source_partition_routing:
            coarse_pool = self.search_hybrid_from_embedding(
                query,
                query_embedding,
                k=pool_size,
                ef_search=ef_search,
                candidates=max(candidates, pool_size),
                alpha=alpha,
            )
            if role_aware_retrieval:
                coarse_pool = apply_role_weights(query, coarse_pool)
            partition_ranking = source_partition_ranking(
                coarse_pool,
                separator=source_partition_separator,
            )
            partition_ids = [
                str(item["partition"])
                for item in partition_ranking[:source_partition_slots]
            ]
            # Bind the complete scan to the database generation immediately
            # before resolving its immutable source set.  A concurrent append
            # anywhere in the remainder of the route invalidates completeness
            # instead of allowing a stale scan report to reach the packer.
            active_partition_routing_turn = self._transcript.current_turn()
            active_partition_content_high_watermark = (
                self._content_high_watermark()
            )
            routed_source_ids = self._retriever.source_ids_in_partitions(
                partition_ids,
                separator=source_partition_separator,
            )
            (
                active_partition_candidates,
                active_partition_report,
            ) = self._scan_active_partition_frontier(
                query,
                query_embedding,
                partition_ids,
                routed_source_ids,
                separator=source_partition_separator,
            )
            pool = self._retriever.hybrid_query_sources(
                query,
                query_embedding,
                routed_source_ids,
                k=pool_size,
                candidates_per_source=source_candidate_pool,
                alpha=alpha,
            )
            self.last_partition_routing_report = {
                "coarse_candidates": len(coarse_pool),
                "selected_partitions": partition_ids,
                "partition_ranking": partition_ranking,
                "routed_sources": len(routed_source_ids),
                "routed_candidates": len(pool),
                **active_partition_report,
            }
            anchors = pool[:k]
        else:
            anchors = self.search_hybrid_from_embedding(
                query,
                query_embedding,
                k=k,
                ef_search=ef_search,
                candidates=candidates,
                alpha=alpha,
            )
            pool = self.search_hybrid_from_embedding(
                query,
                query_embedding,
                k=pool_size,
                ef_search=ef_search,
                candidates=max(candidates, pool_size),
                alpha=alpha,
            )
        if role_aware_retrieval:
            pool = apply_role_weights(query, pool)
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
                    top_k=max(source_slots * 4, source_slots),
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
                        k=max(source_slots, k),
                        candidates_per_source=source_candidate_pool,
                        alpha=alpha,
                    )
                    if role_aware_retrieval:
                        concept_results = apply_role_weights(query, concept_results)
                    concept_results = source_diverse_results(concept_results)
        self.last_source_diversity_report.update(
            {
                "concept_activation": concept_name,
                "concept_candidates": len(concept_results),
                "concept_members": len(concept_member_results),
            }
        )

        facets = (
            query_facets(query, max_facets=query_facet_max)
            if query_facet_retrieval and query_facet_slots
            else []
        )
        facet_groups: list[list[RetrievalResult]] = []
        if facets:
            per_facet = max(
                2,
                2 * ((query_facet_slots + len(facets) - 1) // len(facets)),
            )
            for facet in facets:
                facet_embedding = self._embedder.embed_query(facet)
                if routed_source_ids is not None:
                    facet_pool = self._retriever.hybrid_query_sources(
                        facet,
                        facet_embedding,
                        routed_source_ids,
                        k=per_facet,
                        candidates_per_source=source_candidate_pool,
                        alpha=alpha,
                    )
                else:
                    facet_pool = self.search_hybrid_from_embedding(
                        facet,
                        facet_embedding,
                        k=per_facet,
                        ef_search=ef_search,
                        candidates=max(candidates, per_facet),
                        alpha=alpha,
                    )
                if role_aware_retrieval:
                    facet_pool = apply_role_weights(facet, facet_pool)
                facet_groups.append(facet_pool)

        facet_results = [
            result.model_copy(update={"route": "query_facet"})
            for result in _round_robin_unique(
                facet_groups,
                query_facet_slots,
                {result.chunk.chunk_id for result in anchors},
                stop_on_stall=False,
            )
        ]
        self.last_query_facet_report = {
            "enabled": query_facet_retrieval,
            "facets": len(facets),
            "reserved_slots": query_facet_slots if facets else 0,
            "candidates_added": len(facet_results),
        }
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
                    separator=source_partition_separator,
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
        if source_local_search:
            if use_source_reranker and self._source_candidate_reranker is None:
                raise RuntimeError("source reranking requested but no reranker is attached")
            result_limit = regular_source_slots
            if use_source_reranker:
                result_limit = max(
                    result_limit,
                    self._source_candidate_reranker.candidate_pool,
                )
            source_extras = self._retriever.hybrid_query_sources(
                query,
                query_embedding,
                list(anchor_by_source),
                k=result_limit,
                candidates_per_source=source_candidate_pool,
                alpha=alpha,
                source_scores=source_scores,
                anchor_chunk_ids=anchor_by_source,
                exclude_chunk_ids=tuple(seen),
            )
            if role_aware_retrieval:
                source_extras = apply_role_weights(query, source_extras)
            if use_source_diversity:
                source_extras = source_diverse_results(source_extras)
            if use_source_reranker:
                rerank_candidates = source_extras
                if use_source_diversity:
                    # Set queries need a safe scalar control plus attention
                    # exploration. Build the control from the narrower source
                    # frontier that supplies the output budget and auxiliary
                    # source routes; broader activation is visible only to the
                    # Qwen reserve and cannot evict this protected prefix.
                    protected_activation_k = min(
                        activation_k,
                        source_slots
                        + (source_tfisf_slots if source_tfisf_activation else 0)
                        + (source_hsc_slots if source_hsc_activation else 0)
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
                        candidates_per_source=source_candidate_pool,
                        alpha=alpha,
                        source_scores=protected_source_scores,
                        anchor_chunk_ids=protected_anchor_by_source,
                        exclude_chunk_ids=tuple(seen),
                    )
                    if role_aware_retrieval:
                        protected_extras = apply_role_weights(query, protected_extras)
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
            if role_aware_retrieval:
                hsc_extras = apply_role_weights(query, hsc_extras)
            if use_source_diversity:
                hsc_extras = source_diverse_results(hsc_extras)
        source_extras = [*source_extras, *hsc_extras]

        if use_attention_feedback and feedback_slots:
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
                top_k=feedback_seed_slots,
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
                    truncate_to_tokens(seed.chunk.text, feedback_evidence_tokens)
                    for seed in seeds
                ],
            ]
            activation_window = truncate_to_tokens(
                "\n".join(activation_pieces),
                feedback_query_tokens,
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
                    candidates_per_source=source_candidate_pool,
                    alpha=alpha,
                    source_scores=feedback_source_scores,
                    anchor_chunk_ids=feedback_anchor_by_source,
                    exclude_chunk_ids=tuple(initial_ids),
                )
                activation_selected = self._source_candidate_reranker.select(
                    activation_window,
                    second_pool,
                    top_k=feedback_slots,
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
                    if len(feedback_results) >= feedback_slots:
                        break
                    if result.chunk.chunk_id in activation_ids:
                        continue
                    feedback_results.append(
                        result.model_copy(update={"route": "feedback_scalar_fill"})
                    )

            protected_source_count = max(0, source_slots - feedback_slots)
            selected_source = list(source_extras[:protected_source_count])
            selected_ids = {result.chunk.chunk_id for result in selected_source}
            for result in feedback_results:
                if len(selected_source) >= source_slots:
                    break
                if result.chunk.chunk_id in selected_ids:
                    continue
                selected_source.append(result)
                selected_ids.add(result.chunk.chunk_id)
            for result in source_extras[protected_source_count:]:
                if len(selected_source) >= source_slots:
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
        baseline = [*anchors, *facet_results, *neighbors, *source_extras]
        output = baseline
        if active_partition_report.get("active_partition_scan_status") == "applied":
            output, admission = self._admit_active_partition_candidates(
                baseline,
                active_partition_candidates,
                anchor_chunk_ids={result.chunk.chunk_id for result in anchors},
                semantic_complete=bool(
                    active_partition_report.get(
                        "active_partition_semantically_complete",
                        False,
                    )
                ),
            )
            scan_truncated = int(
                active_partition_report.get(
                    "active_partition_candidates_truncated",
                    0,
                )
            )
            total_truncated = scan_truncated + int(
                admission["active_partition_candidates_truncated"]
            )
            admission["active_partition_candidates_truncated"] = total_truncated
            if total_truncated:
                active_partition_report[
                    "active_partition_semantically_complete"
                ] = False
                active_partition_report[
                    "selected_scope_structurally_complete"
                ] = False
                active_partition_report["global_semantic_complete"] = False
            active_partition_report.update(admission)
            self.last_partition_routing_report.update(active_partition_report)

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
                transcript_turn != active_partition_routing_turn
                or content_high_watermark
                != active_partition_content_high_watermark
            ):
                invalidated_reason = (
                    "transcript_advanced_during_route"
                    if transcript_turn != active_partition_routing_turn
                    else "content_changed_during_route"
                )
                active_partition_report.update(
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
                        **active_partition_report,
                        "active_partition_snapshot_validated": False,
                    }
                )
                return output

            query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
            source_set_sha256 = hashlib.sha256(
                "\0".join(routed_source_ids or ()).encode("utf-8")
            ).hexdigest()
            identity_payload = "\0".join(
                [
                    query_sha256,
                    str(transcript_turn),
                    str(content_high_watermark),
                    source_set_sha256,
                    *partition_ids,
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
                    active_partition_report,
                    routing_identity=routing_identity,
                    query_sha256=query_sha256,
                    transcript_turn=transcript_turn,
                    content_high_watermark=content_high_watermark,
                    selected_partitions=tuple(partition_ids),
                    routed_source_ids=tuple(routed_source_ids or ()),
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
