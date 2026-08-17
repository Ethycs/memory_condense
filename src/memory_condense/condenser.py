from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np

from memory_condense.association_store import AssociationStore, HebbianUpdate
from memory_condense.associative_retrieval import expand_associative_results
from memory_condense.heat_diffusion import expand_heat_diffusion_results
from memory_condense.hebbian_retrieval import (
    expand_hebbian_results,
    retrieval_concept_activations,
)
from memory_condense.chunker import Chunker
from memory_condense.context_packer import ContextBudget, ContextPacker
from memory_condense.consolidation import (
    ConsolidationNode,
    ConsolidationUpdate,
    LiveConsolidationStore,
    context_activations,
    expand_context_associations,
    inspect_qwen_context_hyperplane,
)
from memory_condense.db import Database
from memory_condense.embedding import EmbeddingService
from memory_condense.extractor import Extractor, RuleBasedExtractor
from memory_condense.memory_store import MemoryStore
from memory_condense.ranking import DEFAULT_WEIGHTS, RankWeights
from memory_condense.retrieval import DEFAULT_SPAN_TOKENS, SimilarityRetriever
from memory_condense.schemas import (
    Chunk,
    MemoryResult,
    PackedContext,
    RetrievalResult,
    Turn,
)
from memory_condense.transcript_store import TranscriptStore
from memory_condense.validator import Validator


# Public-facade defaults validated by the locked v3 source-family split. The
# low-level expansion primitive keeps ``None`` controls for explicit ablations.
SAFE_ASSOCIATION_LEXICAL_THRESHOLD = 0.9
SAFE_ASSOCIATION_MAX_TOKEN_INCREASE = 0


class MemoryCondenser:
    """High-level facade wiring the full memory pipeline.

    Everything stateful runs locally: chunking, embedding (bge-m3), the ANN and
    lexical indexes, the memory state machine, and context packing. No LLM call
    is made from this class — memory extraction defaults to the offline
    rule-based extractor. To use LLM-proposed memory, inject an
    ``extractor=LLMExtractor(complete=...)`` with your own provider binding.

    Usage::

        mc = MemoryCondenser(data_dir="./data")
        mc.ingest("user", "I prefer dark mode in all my apps.")
        ctx = mc.build_context("What are my UI preferences?")
        print(ctx.messages)
        mc.close()
    """

    def __init__(
        self,
        data_dir: str | Path = "./data",
        model_name: str = "BAAI/bge-m3",
        chunker_min_tokens: int = 120,
        chunker_max_tokens: int = 250,
        device: str | None = None,
        extractor: Extractor | None = None,
        budget: ContextBudget | None = None,
        auto_extract: bool = True,
        embedder: EmbeddingService | None = None,
        persist_index_on_close: bool = True,
    ) -> None:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        self._db = Database(data_dir / "memory.db")
        self._transcript = TranscriptStore(self._db)
        self._chunker = Chunker(
            min_tokens=chunker_min_tokens,
            max_tokens=chunker_max_tokens,
        )
        # Injectable so tests (and alternate backends) can avoid loading bge-m3.
        self._embedder = embedder or EmbeddingService(
            model_name=model_name,
            device=device,
        )
        self._associations = AssociationStore(self._db)
        self._retriever = SimilarityRetriever(
            db=self._db,
            dim=self._embedder.dim,
            index_path=data_dir / "hnsw_index.bin",
            association_store=self._associations,
        )
        self._memory = MemoryStore(self._db, embedder=self._embedder)
        self._consolidation = LiveConsolidationStore(self._db)
        self._validator = Validator(self._db)
        self._extractor = extractor if extractor is not None else RuleBasedExtractor()
        self._packer = ContextPacker(budget)
        self._auto_extract = auto_extract
        self._persist_index_on_close = persist_index_on_close

    # -- ingestion ----------------------------------------------------------

    def ingest(
        self,
        role: str,
        text: str,
        *,
        source_id: str | None = None,
    ) -> tuple[Turn, list[Chunk]]:
        """Ingest a single conversation turn.

        Stores the turn, chunks and embeds the text, indexes the chunks for
        both dense and lexical retrieval, and — when ``auto_extract`` is on —
        proposes memory items, validates their provenance, and applies the
        surviving ops.
        """
        turn = self._transcript.append(role, text, source_id=source_id)
        chunks = self._chunker.chunk_turn(turn.turn_id, text)

        if chunks:
            chunks = self._embedder.embed_chunks(chunks)
            self._retriever.add_chunks(chunks)

        if self._auto_extract:
            self.extract_memory([turn], chunks)

        return turn, chunks

    def ingest_many(
        self,
        turns: Sequence[tuple[str, str, str | None]],
    ) -> list[tuple[Turn, list[Chunk]]]:
        """Ingest a turn batch with one embedding/index update.

        This is the fast path for document and benchmark loading. Transcript
        order and source provenance remain exact, but all chunks are embedded
        together so a 30-turn session does not launch 30 tiny model forwards.

        Automatic memory extraction remains strictly turn-causal and therefore
        uses :meth:`ingest` sequentially. The batched path is used when
        ``auto_extract=False``, which is already the retrieval-evaluation and
        corpus-indexing configuration.
        """
        records = list(turns)
        if self._auto_extract:
            return [
                self.ingest(role, text, source_id=source_id)
                for role, text, source_id in records
            ]

        staged: list[tuple[Turn, list[Chunk]]] = []
        flat_chunks: list[Chunk] = []
        for role, text, source_id in records:
            turn = self._transcript.append(role, text, source_id=source_id)
            chunks = self._chunker.chunk_turn(turn.turn_id, text)
            staged.append((turn, chunks))
            flat_chunks.extend(chunks)

        if not flat_chunks:
            return staged

        embedded = self._embedder.embed_chunks(flat_chunks)
        self._retriever.add_chunks(embedded)
        by_turn: dict[str, list[Chunk]] = {}
        for chunk in embedded:
            by_turn.setdefault(chunk.turn_id, []).append(chunk)
        return [(turn, by_turn.get(turn.turn_id, [])) for turn, _ in staged]

    def extract_memory(
        self, turns: list[Turn], chunks: list[Chunk] | None = None
    ) -> dict[str, int]:
        """Propose, validate, and apply memory ops for the given turns.

        Ops whose provenance cannot be verified against the transcript are
        rejected — an LLM cannot write a memory it did not quote.
        """
        ops = self._extractor.extract(turns, chunks)
        if ops.is_empty():
            return {}
        report = self._validator.validate(ops)
        return self._memory.apply(report)

    # -- retrieval ----------------------------------------------------------

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
            source_id = result.turn.source_id or result.turn.turn_id
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
        ef_search: int = 50,
        candidates: int = 100,
        alpha: float = 0.65,
    ) -> list[RetrievalResult]:
        """Rerank a bounded pool inside sources activated by a ranked prefix.

        The top-k hybrid results remain the anchors.  Remaining slots can only
        be filled by lower-ranked candidates from a source represented in the
        independently bounded activation prefix. This reaches evidence
        elsewhere in a relevant conversation without hydrating whole sources
        or retaining model/token state.
        """
        if k <= 0:
            return []
        if source_slots < 0:
            raise ValueError("source_slots must be non-negative")
        if source_candidate_pool < k:
            raise ValueError("source_candidate_pool must be at least k")
        activation_k = k if source_activation_k is None else source_activation_k
        if activation_k < k or activation_k > source_candidate_pool:
            raise ValueError("source_activation_k must be between k and the pool")

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
        for result in pool[:activation_k]:
            if result.turn is None:
                continue
            source_id = result.turn.source_id or result.turn.turn_id
            anchor_by_source.setdefault(source_id, result.chunk.chunk_id)

        extras: list[RetrievalResult] = []
        anchor_ids = {result.chunk.chunk_id for result in anchors}
        for result in pool:
            if result.turn is None or result.chunk.chunk_id in anchor_ids:
                continue
            source_id = result.turn.source_id or result.turn.turn_id
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
        ef_search: int = 50,
        candidates: int = 100,
        alpha: float = 0.65,
    ) -> list[RetrievalResult]:
        """Union transition and source links behind immutable hybrid anchors.

        Results are ordered as anchors, bounded source-local transitions, then
        lower-ranked candidates from activated sources.  The caller's prompt
        cap remains the final hard byte/token boundary.
        """
        if k <= 0:
            return []
        if neighbor_radius < 0 or neighbor_slots < 0 or source_slots < 0:
            raise ValueError("graph retrieval bounds must be non-negative")
        if neighbor_direction not in {"both", "previous", "next"}:
            raise ValueError("invalid neighbor_direction")
        if source_candidate_pool < k:
            raise ValueError("source_candidate_pool must be at least k")
        activation_k = k if source_activation_k is None else source_activation_k
        if activation_k < k or activation_k > source_candidate_pool:
            raise ValueError("source_activation_k must be between k and the pool")

        query_embedding = self._embedder.embed_query(query)
        anchors = self.search_hybrid_from_embedding(
            query,
            query_embedding,
            k=k,
            ef_search=ef_search,
            candidates=candidates,
            alpha=alpha,
        )
        expanded = self.expand_source_neighbors(
            anchors,
            radius=neighbor_radius,
        )
        neighbors = [
            result
            for result in expanded[len(anchors) :]
            if neighbor_direction == "both"
            or result.transition_direction == neighbor_direction
        ][:neighbor_slots]

        pool_size = max(k, source_candidate_pool)
        pool = self.search_hybrid_from_embedding(
            query,
            query_embedding,
            k=pool_size,
            ef_search=ef_search,
            candidates=max(candidates, pool_size),
            alpha=alpha,
        )
        anchor_by_source: dict[str, str] = {}
        for result in pool[:activation_k]:
            if result.turn is not None:
                source_id = result.turn.source_id or result.turn.turn_id
                anchor_by_source.setdefault(source_id, result.chunk.chunk_id)

        seen = {
            result.chunk.chunk_id for result in [*anchors, *neighbors]
        }
        source_extras: list[RetrievalResult] = []
        for result in pool:
            if result.turn is None or result.chunk.chunk_id in seen:
                continue
            source_id = result.turn.source_id or result.turn.turn_id
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
            if len(source_extras) >= source_slots:
                break
        return [*anchors, *neighbors, *source_extras]

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

    # -- context assembly ---------------------------------------------------

    def build_context(
        self,
        user_text: str,
        system_prompt: str = "",
        recent_turns: int = 8,
        k_memories: int = 8,
        k_expansions: int = 10,
        hybrid: bool = True,
        reheat_memories: bool = True,
        use_consolidation: bool = True,
        learn_consolidation: bool = True,
        consolidation_memory_slots: int = 1,
        consolidation_chunk_slots: int = 1,
        consolidation_min_count: int = 2,
        consolidation_hops: int = 1,
        consolidation_candidates: int = 32,
        consolidation_diffusion_width: int = 32,
        access_event_id: str | None = None,
    ) -> PackedContext:
        """Assemble a token-budgeted prompt for ``user_text``.

        Memory header + recent window + verbatim expansions, each capped
        independently so context cost stays predictable as the conversation
        grows. Established live-consolidation edges may propose bounded
        additive candidates before packing; they never evict a direct result
        merely to reserve a graph slot. Only direct results that actually
        survive packing train the graph afterward, preventing a graph-selected
        result from reinforcing itself merely because the graph selected it.
        """
        # Ranking is a read; only memories that survive the header budget are
        # genuine accesses.  Reheating all top-k candidates here kept dropped
        # rows artificially warm and defeated pruning by access frequency.
        memories = (
            self.recall_memories(user_text, k=k_memories, reheat=False)
            if k_memories
            else []
        )

        expansions: list[RetrievalResult] = []
        query_embedding: np.ndarray | None = None
        if k_expansions:
            query_embedding = self._embedder.embed_query(user_text)
            if hybrid:
                expansions = self.search_hybrid_from_embedding(
                    user_text,
                    query_embedding,
                    k=k_expansions,
                )
            else:
                expansions = self._retriever.query(
                    query_embedding,
                    k=k_expansions,
                )

        if use_consolidation and (memories or expansions):
            memories, expansions = expand_context_associations(
                memories,
                expansions,
                store=self._consolidation,
                get_memory=self._memory.get,
                hydrate_chunk=self._retriever.hydrate_chunk,
                now_turn=self._transcript.current_turn(),
                memory_slots=consolidation_memory_slots,
                chunk_slots=consolidation_chunk_slots,
                max_candidates=consolidation_candidates,
                min_coactivation_count=consolidation_min_count,
                diffusion_hops=consolidation_hops,
                diffusion_width=consolidation_diffusion_width,
                chunk_relevance=(
                    lambda chunk_ids: self._retriever.cosine_scores(
                        query_embedding,
                        chunk_ids,
                    )
                    if query_embedding is not None
                    else {}
                ),
            )

        turns = self._transcript.get_recent(recent_turns) if recent_turns else []
        recent = [(t.role, t.text) for t in turns]

        packed = self._packer.pack(
            system_prompt=system_prompt,
            memories=memories,
            recent_turns=recent,
            expansions=expansions,
            user_text=user_text,
        )
        memory_by_id = {result.item.mem_id: result for result in memories}
        chunk_by_id = {result.chunk.chunk_id: result for result in expansions}
        # Keep the independently retrieved subset explicit for delayed Qwen
        # inspection. The text-free IDs are already part of the packed result;
        # no activation or prompt buffer is retained.
        direct_memory_ids = [
            mem_id
            for mem_id in packed.memory_ids
            if memory_by_id[mem_id].route != "live_consolidation"
        ]
        direct_chunk_ids = [
            chunk_id
            for chunk_id in packed.expansion_chunk_ids
            if chunk_by_id[chunk_id].route != "live_consolidation"
        ]
        event_id = access_event_id or self._context_event_id(
            user_text,
            direct_memory_ids,
            direct_chunk_ids,
        )
        packed = packed.model_copy(
            update={
                "direct_memory_ids": direct_memory_ids,
                "direct_expansion_chunk_ids": direct_chunk_ids,
                "consolidation_event_id": event_id,
            }
        )
        if reheat_memories:
            now_turn = self._transcript.current_turn()
            ranked_by_id = {result.item.mem_id: result.item for result in memories}
            self._memory.touch_many(
                [ranked_by_id[mem_id] for mem_id in packed.memory_ids],
                now_turn=now_turn,
            )
        if learn_consolidation:
            # Learn from independent retrieval evidence only.  A candidate
            # admitted by this graph must later be found directly before it can
            # strengthen the assembly, avoiding a self-confirming feedback loop.
            activations = context_activations(
                direct_memory_ids,
                direct_chunk_ids,
            )
            if activations:
                self._consolidation.observe(event_id, activations)
                packed = packed.model_copy(update={"consolidation_learned": True})
        return packed

    def observe_context_access(
        self,
        memory_ids: Sequence[str],
        chunk_ids: Sequence[str],
        *,
        access_event_id: str,
        now_turn: int | None = None,
        node_activations: Mapping[ConsolidationNode, float] | None = None,
        pair_affinities: Mapping[
            tuple[ConsolidationNode, ConsolidationNode], float
        ]
        | None = None,
        causal_chunk_ids: Sequence[str] = (),
    ) -> ConsolidationUpdate:
        """Explicitly reinforce one externally assembled, bounded context.

        Rank-discounted activity is the provider-free default.  A transient
        Qwen prefix inspection may instead pass CAV-derived node activity and
        bounded QK/OV ``pair_affinities``. ``causal_chunk_ids`` marks newly
        produced response/tool evidence in a completed interaction. Only the
        resulting scalar update is durable; the caller remains responsible for
        discarding the workspace.
        """

        allowed = set(context_activations(memory_ids, chunk_ids))
        if node_activations is None:
            activations = context_activations(memory_ids, chunk_ids)
        else:
            activations = {
                node: float(value)
                for node, value in node_activations.items()
                if node in allowed
            }
        filtered_pairs = {
            pair: float(value)
            for pair, value in (pair_affinities or {}).items()
            if pair[0] in allowed and pair[1] in allowed
        }
        causal_targets = tuple(
            node
            for chunk_id in dict.fromkeys(str(value) for value in causal_chunk_ids)
            if (node := ConsolidationNode.chunk(chunk_id)) in allowed
        )
        return self._consolidation.observe(
            access_event_id,
            activations,
            pair_affinities=filtered_pairs,
            causal_targets=causal_targets,
            now_turn=now_turn,
        )

    def consolidate_context_with_qwen(
        self,
        user_text: str,
        packed: PackedContext,
        linker: object,
        *,
        access_event_id: str | None = None,
        now_turn: int | None = None,
        causal_chunk_ids: Sequence[str] = (),
    ) -> tuple[object, ConsolidationUpdate]:
        """Inspect a completed turn with Qwen and apply one scalar update.

        This is intentionally separate from :meth:`build_context` so callers
        may run the bounded prefix pass after the response or on a background
        queue instead of adding it to answer latency. Build with
        ``learn_consolidation=False`` first; mixing the rank fallback and Qwen
        learning for one context would count that prompt twice.
        """

        if packed.consolidation_learned:
            raise ValueError(
                "context already used rank-based consolidation; build with "
                "learn_consolidation=False before Qwen consolidation"
            )
        memories = [
            item
            for mem_id in packed.direct_memory_ids
            if (item := self._memory.get(mem_id)) is not None
        ]
        chunks = [
            result
            for chunk_id in packed.direct_expansion_chunk_ids
            if (
                result := self._retriever.hydrate_chunk(
                    chunk_id,
                    score=0.0,
                    route="direct_for_consolidation",
                )
            )
            is not None
        ]
        result, activations = inspect_qwen_context_hyperplane(
            linker,
            user_text,
            memories,
            chunks,
        )
        event_id = (
            access_event_id
            or packed.consolidation_event_id
            or self._context_event_id(
                user_text,
                packed.direct_memory_ids,
                packed.direct_expansion_chunk_ids,
            )
        )
        update = self.observe_context_access(
            packed.direct_memory_ids,
            packed.direct_expansion_chunk_ids,
            access_event_id=event_id,
            now_turn=now_turn,
            node_activations=activations,
            causal_chunk_ids=causal_chunk_ids,
        )
        return result, update

    def _context_event_id(
        self,
        user_text: str,
        memory_ids: Sequence[str],
        chunk_ids: Sequence[str],
    ) -> str:
        """Stable retry identity without retaining the prompt or context text."""

        payload = "\x1f".join(
            [
                str(self._transcript.current_turn()),
                user_text,
                *memory_ids,
                *chunk_ids,
            ]
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        return f"context:{self._transcript.current_turn()}:{digest}"

    # -- accessors ----------------------------------------------------------

    @property
    def transcript(self) -> TranscriptStore:
        """Access the transcript store directly."""
        return self._transcript

    @property
    def memory(self) -> MemoryStore:
        """Access the memory store directly."""
        return self._memory

    @property
    def retriever(self) -> SimilarityRetriever:
        """Access the chunk retriever directly."""
        return self._retriever

    @property
    def associations(self) -> AssociationStore:
        """Compact external CAV/QK/OV artifacts; never transformer token state."""
        return self._associations

    @property
    def consolidation(self) -> LiveConsolidationStore:
        """Prompt-driven associations spanning semantic memory and evidence."""

        return self._consolidation

    @property
    def database_path(self) -> Path:
        """SQLite path for independent read-only experiment workers."""
        return self._db.path

    @property
    def validator(self) -> Validator:
        """Access the provenance validator directly."""
        return self._validator

    def heat_counts(self, now_turn: int | None = None) -> dict[str, int]:
        """Current HOT/WARM/COLD distribution of active memory items."""
        return self._memory.heat_counts(now_turn=now_turn)

    def close(self) -> None:
        """Persist index and close database."""
        if self._persist_index_on_close:
            self._retriever.save()
        self._db.close()

    def __enter__(self) -> MemoryCondenser:
        return self

    def __exit__(self, *args) -> None:
        self.close()
