"""Stateful base, associative, heat, Hebbian, span, and source retrieval."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from memory_condense.application.query_routing import (
    SAFE_ASSOCIATION_LEXICAL_THRESHOLD,
    SAFE_ASSOCIATION_MAX_TOKEN_INCREASE,
)
from memory_condense.associations.association_store import HebbianUpdate
from memory_condense.associations.associative_retrieval import expand_associative_results
from memory_condense.associations.heat_diffusion import expand_heat_diffusion_results
from memory_condense.associations.hebbian_retrieval import (
    expand_hebbian_results,
    retrieval_concept_activations,
)
from memory_condense.domain.schemas import RetrievalResult
from memory_condense.search.indexes.retrieval import DEFAULT_SPAN_TOKENS


class RetrievalWorkflowMixin:
    """Internal workflow methods composed by ``MemoryCondenser``."""

    def search(
        self, query: str, k: int = 10, ef_search: int = 50
    ) -> list[RetrievalResult]:
        """Dense-only chunk search. This is the baseline retrieval path."""
        query_embedding = self._embedder.embed_query(query)
        return self._retriever.query(query_embedding, k=k, ef_search=ef_search)

    def search_hybrid(
        self,
        query: str,
        k: int = 10,
        ef_search: int = 50,
        candidates: int = 100,
        alpha: float = 0.65,
    ) -> list[RetrievalResult]:
        """Hybrid dense + BM25 chunk search, blended and reranked."""
        query_embedding = self._embedder.embed_query(query)
        return self.search_hybrid_from_embedding(
            query,
            query_embedding,
            k=k,
            ef_search=ef_search,
            candidates=candidates,
            alpha=alpha,
        )

    def search_hybrid_from_embedding(
        self,
        query: str,
        query_embedding: np.ndarray,
        *,
        k: int = 10,
        ef_search: int = 50,
        candidates: int = 100,
        alpha: float = 0.65,
    ) -> list[RetrievalResult]:
        """Run hybrid retrieval from a precomputed query embedding.

        Evaluation can batch query encoding across isolated sample stores,
        then preserve exactly the same hybrid ranking in each store.
        """
        return self._retriever.hybrid_query(
            query_text=query,
            query_embedding=np.asarray(query_embedding, dtype=np.float32),
            k=k,
            ef_search=ef_search,
            candidates=candidates,
            alpha=alpha,
        )

    def search_hybrid_many(
        self,
        queries: Sequence[str],
        *,
        k: int = 10,
        ef_search: int = 50,
        candidates: int = 100,
        alpha: float = 0.65,
    ) -> list[list[RetrievalResult]]:
        """Batch query embedding once, then run deterministic hybrid retrieval."""
        if not queries:
            return []
        embed_many = getattr(self._embedder, "embed_queries", None)
        if embed_many is None:
            embeddings = np.stack(
                [self._embedder.embed_query(query) for query in queries]
            )
        else:
            embeddings = embed_many(queries)
        return [
            self._retriever.hybrid_query(
                query_text=query,
                query_embedding=np.asarray(embedding, dtype=np.float32),
                k=k,
                ef_search=ef_search,
                candidates=candidates,
                alpha=alpha,
            )
            for query, embedding in zip(queries, embeddings, strict=True)
        ]

    def search_associative(
        self,
        query: str,
        artifact_id: str,
        *,
        k: int = 10,
        association_slots: int = 0,
        qk_reserve: int = 1,
        neighbors_per_anchor: int = 4,
        association_hops: int = 1,
        max_association_candidates: int = 64,
        cav_candidates: int = 8,
        lexical_protection_threshold: float | None = (
            SAFE_ASSOCIATION_LEXICAL_THRESHOLD
        ),
        max_prompt_token_increase: int | None = (
            SAFE_ASSOCIATION_MAX_TOKEN_INCREASE
        ),
        ef_search: int = 50,
        candidates: int = 100,
        alpha: float = 0.65,
        touch: bool = True,
    ) -> list[RetrievalResult]:
        """Expand hybrid anchors through external links under the same item cap.

        This method does not run Qwen and cannot grow transformer context.  It
        reads compact links previously emitted by bounded head-inspection
        passes, hydrates only the chosen chunk IDs, and returns at most ``k``
        ordinary retrieval results.  ``association_slots=0`` is conservative:
        association routes may fill only slots freed by duplicate anchors.
        A positive value explicitly trades that many direct slots for graph or
        CAV exploration while keeping the total retrieval budget fixed. By
        default, a near-maximum lexical tail anchor cannot be displaced and a
        composition that increases prompt tokens is rolled back before touch.
        """
        if k <= 0:
            return []
        anchors = self.search_hybrid(
            query,
            k=k,
            ef_search=ef_search,
            candidates=candidates,
            alpha=alpha,
        )
        return self.expand_associative(
            anchors,
            artifact_id,
            k=k,
            association_slots=association_slots,
            qk_reserve=qk_reserve,
            neighbors_per_anchor=neighbors_per_anchor,
            association_hops=association_hops,
            max_association_candidates=max_association_candidates,
            cav_candidates=cav_candidates,
            lexical_protection_threshold=lexical_protection_threshold,
            max_prompt_token_increase=max_prompt_token_increase,
            touch=touch,
        )

    def expand_associative(
        self,
        anchors: Sequence[RetrievalResult],
        artifact_id: str,
        *,
        k: int | None = None,
        association_slots: int = 0,
        qk_reserve: int = 1,
        neighbors_per_anchor: int = 4,
        association_hops: int = 1,
        max_association_candidates: int = 64,
        cav_candidates: int = 8,
        lexical_protection_threshold: float | None = (
            SAFE_ASSOCIATION_LEXICAL_THRESHOLD
        ),
        max_prompt_token_increase: int | None = (
            SAFE_ASSOCIATION_MAX_TOKEN_INCREASE
        ),
        touch: bool = True,
    ) -> list[RetrievalResult]:
        """Fan out cached hybrid anchors without embedding the query again."""
        return expand_associative_results(
            anchors,
            artifact_id,
            store=self._associations,
            hydrate=self._retriever.hydrate_chunk,
            now_turn=self._db.current_turn(),
            k=k,
            association_slots=association_slots,
            qk_reserve=qk_reserve,
            neighbors_per_anchor=neighbors_per_anchor,
            association_hops=association_hops,
            max_association_candidates=max_association_candidates,
            cav_candidates=cav_candidates,
            lexical_protection_threshold=lexical_protection_threshold,
            max_prompt_token_increase=max_prompt_token_increase,
            touch=touch,
        )

    def search_heat_associative(
        self,
        query: str,
        artifact_id: str,
        *,
        k: int = 10,
        association_slots: int = 1,
        qk_reserve: int = 1,
        ranked_qk_reserve: int = 0,
        neighbors_per_node: int = 3,
        diffusion_hops: int = 2,
        max_diffusion_nodes: int = 8,
        restart_probability: float = 0.35,
        seed_temperature: float = 1.0,
        edge_temperature: float = 1.0,
        lexical_protection_threshold: float | None = (
            SAFE_ASSOCIATION_LEXICAL_THRESHOLD
        ),
        max_prompt_token_increase: int | None = (
            SAFE_ASSOCIATION_MAX_TOKEN_INCREASE
        ),
        max_source_token_fraction: float = 1.0,
        ef_search: int = 50,
        candidates: int = 100,
        alpha: float = 0.65,
        touch: bool = True,
    ) -> list[RetrievalResult]:
        """Diffuse compact head evidence, then expose memory by source heat.

        Qwen is not loaded here. The method carries only IDs, normalized scalar
        heat, and one explanatory path through a bounded external graph.
        """
        if k <= 0:
            return []
        anchors = self.search_hybrid(
            query,
            k=k,
            ef_search=ef_search,
            candidates=candidates,
            alpha=alpha,
        )
        return self.expand_heat_associative(
            anchors,
            artifact_id,
            k=k,
            association_slots=association_slots,
            qk_reserve=qk_reserve,
            ranked_qk_reserve=ranked_qk_reserve,
            neighbors_per_node=neighbors_per_node,
            diffusion_hops=diffusion_hops,
            max_diffusion_nodes=max_diffusion_nodes,
            restart_probability=restart_probability,
            seed_temperature=seed_temperature,
            edge_temperature=edge_temperature,
            lexical_protection_threshold=lexical_protection_threshold,
            max_prompt_token_increase=max_prompt_token_increase,
            max_source_token_fraction=max_source_token_fraction,
            touch=touch,
        )

    def expand_heat_associative(
        self,
        anchors: Sequence[RetrievalResult],
        artifact_id: str,
        *,
        k: int | None = None,
        association_slots: int = 1,
        qk_reserve: int = 1,
        ranked_qk_reserve: int = 0,
        neighbors_per_node: int = 3,
        diffusion_hops: int = 2,
        max_diffusion_nodes: int = 8,
        restart_probability: float = 0.35,
        seed_temperature: float = 1.0,
        edge_temperature: float = 1.0,
        lexical_protection_threshold: float | None = (
            SAFE_ASSOCIATION_LEXICAL_THRESHOLD
        ),
        max_prompt_token_increase: int | None = (
            SAFE_ASSOCIATION_MAX_TOKEN_INCREASE
        ),
        max_source_token_fraction: float = 1.0,
        touch: bool = True,
    ) -> list[RetrievalResult]:
        """Diffuse cached anchors without embedding the query or loading Qwen."""
        return expand_heat_diffusion_results(
            anchors,
            artifact_id,
            store=self._associations,
            hydrate=self._retriever.hydrate_chunk,
            now_turn=self._db.current_turn(),
            k=k,
            association_slots=association_slots,
            qk_reserve=qk_reserve,
            ranked_qk_reserve=ranked_qk_reserve,
            neighbors_per_node=neighbors_per_node,
            diffusion_hops=diffusion_hops,
            max_diffusion_nodes=max_diffusion_nodes,
            restart_probability=restart_probability,
            seed_temperature=seed_temperature,
            edge_temperature=edge_temperature,
            lexical_protection_threshold=lexical_protection_threshold,
            max_prompt_token_increase=max_prompt_token_increase,
            max_source_token_fraction=max_source_token_fraction,
            touch=touch,
        )

    def search_hebbian(
        self,
        query: str,
        artifact_id: str,
        *,
        k: int = 10,
        hebbian_slots: int = 1,
        max_candidates: int = 32,
        half_life_turns: float = 200.0,
        min_score: float = 0.05,
        lexical_protection_threshold: float | None = (
            SAFE_ASSOCIATION_LEXICAL_THRESHOLD
        ),
        max_prompt_token_increase: int | None = (
            SAFE_ASSOCIATION_MAX_TOKEN_INCREASE
        ),
        access_event_id: str | None = None,
        max_concepts_per_event: int = 12,
        max_degree: int = 32,
        ef_search: int = 50,
        candidates: int = 100,
        alpha: float = 0.65,
    ) -> list[RetrievalResult]:
        """Use learned same-turn co-access links inside a fixed result budget.

        Supplying ``access_event_id`` records the final returned set as one
        idempotent live access event. Omit it for evaluation reads that must not
        train the graph. The ID should identify the user turn or generation,
        not the query text itself.
        """
        if k <= 0:
            return []
        anchors = self.search_hybrid(
            query,
            k=k,
            ef_search=ef_search,
            candidates=candidates,
            alpha=alpha,
        )
        results = self.expand_hebbian(
            anchors,
            artifact_id,
            k=k,
            hebbian_slots=hebbian_slots,
            max_candidates=max_candidates,
            half_life_turns=half_life_turns,
            min_score=min_score,
            lexical_protection_threshold=lexical_protection_threshold,
            max_prompt_token_increase=max_prompt_token_increase,
        )
        if access_event_id is not None:
            self.observe_retrieval_access(
                results,
                artifact_id,
                access_event_id=access_event_id,
                half_life_turns=half_life_turns,
                max_concepts_per_event=max_concepts_per_event,
                max_degree=max_degree,
            )
        return results

    def expand_hebbian(
        self,
        anchors: Sequence[RetrievalResult],
        artifact_id: str,
        *,
        k: int | None = None,
        hebbian_slots: int = 1,
        max_candidates: int = 32,
        half_life_turns: float = 200.0,
        min_score: float = 0.05,
        lexical_protection_threshold: float | None = (
            SAFE_ASSOCIATION_LEXICAL_THRESHOLD
        ),
        max_prompt_token_increase: int | None = (
            SAFE_ASSOCIATION_MAX_TOKEN_INCREASE
        ),
    ) -> list[RetrievalResult]:
        """Expand cached anchors through live co-access links without an LLM."""
        return expand_hebbian_results(
            anchors,
            artifact_id,
            store=self._associations,
            hydrate=self._retriever.hydrate_chunk,
            now_turn=self._db.current_turn(),
            k=k,
            hebbian_slots=hebbian_slots,
            max_candidates=max_candidates,
            half_life_turns=half_life_turns,
            min_score=min_score,
            lexical_protection_threshold=lexical_protection_threshold,
            max_prompt_token_increase=max_prompt_token_increase,
        )

    def observe_retrieval_access(
        self,
        results: Sequence[RetrievalResult],
        artifact_id: str,
        *,
        access_event_id: str,
        now_turn: int | None = None,
        learning_rate: float = 1.0,
        half_life_turns: float = 200.0,
        max_concepts_per_event: int = 12,
        max_degree: int = 32,
        min_edge_score: float = 0.0,
        max_event_history: int = 4096,
    ) -> HebbianUpdate:
        """Reinforce conceptual chunks actually retrieved in the same turn."""
        activations = retrieval_concept_activations(
            results,
            max_concepts=max_concepts_per_event,
        )
        return self._associations.reinforce_retrieval_coaccess(
            artifact_id,
            access_event_id,
            activations,
            now_turn=now_turn,
            learning_rate=learning_rate,
            half_life_turns=half_life_turns,
            max_concepts_per_event=max_concepts_per_event,
            max_degree=max_degree,
            min_edge_score=min_edge_score,
            max_event_history=max_event_history,
        )

    def search_spans(
        self,
        query: str,
        levels: Sequence[int] = DEFAULT_SPAN_TOKENS,
        k_per_level: int = 2,
    ) -> list[RetrievalResult]:
        """Search pooled spans of contiguous chunks rather than single chunks.

        For short conversational turns this recovers most of what per-chunk
        search loses: a 27-token turn carries too little topical signal to be
        matched, while a ~110-220 token span carries enough. ``levels`` are
        token targets, so the same setting works on corpora whose turns differ
        by an order of magnitude in length. Returns the member chunks of the
        winning spans, so callers handle ordinary ``RetrievalResult``s.

        Not the default. It replicates on four LoCoMo samples (10.3% -> 23.4%
        answer containment, better on every sample), but ``search`` remains the
        baseline the eval ablations compare against, and nothing has yet
        measured this on the long-form corpus the original ablation used.
        """
        query_embedding = self._embedder.embed_query(query)
        return self._retriever.span_query(
            query_embedding, levels=levels, k_per_level=k_per_level
        )

    def search_sources(
        self,
        query: str,
        *,
        k_sources: int = 4,
    ) -> list[RetrievalResult]:
        """Retrieve complete source/session groups by pooled dense similarity."""

        query_embedding = self._embedder.embed_query(query)
        return self._retriever.source_query(query_embedding, k_sources=k_sources)

    def search_anchored_sources(
        self,
        query: str,
        *,
        k: int = 10,
        ef_search: int = 50,
        candidates: int = 100,
        alpha: float = 0.65,
    ) -> list[RetrievalResult]:
        """Select sources with hybrid anchors, then fairly expand each source."""
        anchors = self.search_hybrid(
            query,
            k=k,
            ef_search=ef_search,
            candidates=candidates,
            alpha=alpha,
        )
        source_ids: list[str] = []
        source_scores: dict[str, float] = {}
        for result in anchors:
            if result.turn is None:
                continue
            source_id = result.source_key
            if source_id not in source_scores:
                source_ids.append(source_id)
                source_scores[source_id] = float(result.score)
            else:
                source_scores[source_id] = max(
                    source_scores[source_id], float(result.score)
                )
        return self._retriever.hydrate_sources(
            source_ids,
            source_scores=source_scores,
            interleave=True,
        )

    def search_hybrid_sources(
        self,
        query: str,
        *,
        k: int = 10,
        source_slots: int = 24,
        source_candidate_pool: int = 200,
        source_activation_k: int | None = None,
        source_local_search: bool = False,
        use_source_reranker: bool = False,
        ef_search: int = 50,
        candidates: int = 100,
        alpha: float = 0.65,
    ) -> list[RetrievalResult]:
        """Rerank a bounded pool inside sources activated by a ranked prefix.

        The top-k hybrid results remain the anchors.  Remaining slots can only
        be filled by lower-ranked candidates from a source represented in the
        independently bounded activation prefix. This reaches evidence
        elsewhere in a relevant conversation without hydrating whole sources
        or retaining request-derived model/token state.
        """
        self.last_source_rerank_report = {}
        if k <= 0:
            return []
        if source_slots < 0:
            raise ValueError("source_slots must be non-negative")
        if source_candidate_pool < k:
            raise ValueError("source_candidate_pool must be at least k")
        activation_k = k if source_activation_k is None else source_activation_k
        if activation_k < k or activation_k > source_candidate_pool:
            raise ValueError("source_activation_k must be between k and the pool")
        if use_source_reranker and not source_local_search:
            raise ValueError("source reranking requires source_local_search")

        query_embedding = self._embedder.embed_query(query)
        anchors = self.search_hybrid_from_embedding(
            query,
            query_embedding,
            k=k,
            ef_search=ef_search,
            candidates=candidates,
            alpha=alpha,
        )
        pool_size = max(k, source_candidate_pool)
        pool = self.search_hybrid_from_embedding(
            query,
            query_embedding,
            k=pool_size,
            ef_search=ef_search,
            candidates=max(candidates, pool_size),
            alpha=alpha,
        )
        if source_slots == 0 or not pool:
            return anchors

        anchor_by_source: dict[str, str] = {}
        source_scores: dict[str, float] = {}
        for result in pool[:activation_k]:
            if result.turn is None:
                continue
            source_id = result.source_key
            anchor_by_source.setdefault(source_id, result.chunk.chunk_id)
            source_scores[source_id] = max(
                source_scores.get(source_id, 0.0), float(result.score)
            )

        anchor_ids = {result.chunk.chunk_id for result in anchors}
        if source_local_search:
            if use_source_reranker and self._source_candidate_reranker is None:
                raise RuntimeError("source reranking requested but no reranker is attached")
            result_limit = source_slots
            if use_source_reranker:
                result_limit = max(
                    result_limit,
                    self._source_candidate_reranker.candidate_pool,
                )
            local = self._retriever.hybrid_query_sources(
                query,
                query_embedding,
                list(anchor_by_source),
                k=result_limit,
                candidates_per_source=source_candidate_pool,
                alpha=alpha,
                source_scores=source_scores,
                anchor_chunk_ids=anchor_by_source,
                exclude_chunk_ids=tuple(anchor_ids),
            )
            if use_source_reranker:
                local = self._source_candidate_reranker.rerank(
                    query,
                    local,
                    top_k=source_slots,
                )
                report = self._source_candidate_reranker.last_report
                self.last_source_rerank_report = (
                    report.model_dump() if report is not None else {}
                )
            return [*anchors, *local]

        extras: list[RetrievalResult] = []
        for result in pool:
            if result.turn is None or result.chunk.chunk_id in anchor_ids:
                continue
            source_id = result.source_key
            anchor_id = anchor_by_source.get(source_id)
            if anchor_id is None:
                continue
            extras.append(
                result.model_copy(
                    update={
                        "route": "hybrid_source",
                        "anchor_chunk_id": anchor_id,
                    }
                )
            )
            if len(extras) >= source_slots:
                break
        return list(anchors) + extras


__all__ = ["RetrievalWorkflowMixin"]
