from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence

import numpy as np

from memory_condense._tokenizer import count_tokens, truncate_to_tokens
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


def _source_partition(source_id: str, separator: str) -> str:
    """Top-level durable partition encoded by ``partition::source``."""

    return source_id.split(separator, 1)[0]


def select_source_partitions(
    candidates: Sequence[RetrievalResult],
    *,
    slots: int,
    separator: str = "::",
    max_hits_per_partition: int = 8,
) -> list[str]:
    """Rank coarse partitions by reciprocal-rank heat over chunk hits.

    A single lucky chunk should not route an entire million-token memory.
    Accumulating bounded reciprocal-rank mass rewards partitions that produce
    several independently relevant candidates while remaining insensitive to
    their raw source or transcript length.
    """

    if slots < 1:
        raise ValueError("partition slots must be positive")
    if not separator:
        raise ValueError("partition separator must be non-empty")
    if max_hits_per_partition < 1:
        raise ValueError("max_hits_per_partition must be positive")
    scores: dict[str, float] = {}
    first_rank: dict[str, int] = {}
    hit_counts: dict[str, int] = {}
    for rank, result in enumerate(candidates, start=1):
        if result.turn is None:
            continue
        source_id = str(result.turn.source_id or result.turn.turn_id)
        partition = _source_partition(source_id, separator)
        count = hit_counts.get(partition, 0)
        if count >= max_hits_per_partition:
            continue
        hit_counts[partition] = count + 1
        scores[partition] = scores.get(partition, 0.0) + 1.0 / (60.0 + rank)
        first_rank.setdefault(partition, rank)
    return sorted(
        scores,
        key=lambda partition: (
            -scores[partition],
            first_rank[partition],
            partition,
        ),
    )[:slots]


class SourceCandidateReranker(Protocol):
    candidate_pool: int
    last_report: Any

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        top_k: int,
    ) -> list[RetrievalResult]: ...

    def select(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        top_k: int,
    ) -> list[RetrievalResult]: ...


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
        retriever_max_elements: int = 100_000,
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
            max_elements=retriever_max_elements,
        )
        self._memory = MemoryStore(self._db, embedder=self._embedder)
        self._consolidation = LiveConsolidationStore(self._db)
        self._validator = Validator(self._db)
        self._extractor = extractor if extractor is not None else RuleBasedExtractor()
        self._packer = ContextPacker(budget)
        self._auto_extract = auto_extract
        self._persist_index_on_close = persist_index_on_close
        self._source_candidate_reranker: SourceCandidateReranker | None = None
        self.last_source_rerank_report: dict[str, Any] = {}
        self.last_partition_routing_report: dict[str, Any] = {}

    def set_source_candidate_reranker(
        self,
        reranker: SourceCandidateReranker | None,
    ) -> None:
        """Attach one shared transient reranker; no model state enters SQLite."""

        self._source_candidate_reranker = reranker
        self.last_source_rerank_report = {}

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
        or retaining model/token state.
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
            source_id = result.turn.source_id or result.turn.turn_id
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
        if source_tfisf_slots < 1:
            raise ValueError("source_tfisf_slots must be positive")
        if source_hsc_slots < 1 or source_hsc_hops < 1:
            raise ValueError("source HSC slots and hops must be positive")
        if source_hsc_activation and (
            source_hsc_chunk_slots < 1 or source_hsc_chunk_slots > source_slots
        ):
            raise ValueError("source_hsc_chunk_slots must lie in [1, source_slots]")
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
        query_embedding = self._embedder.embed_query(query)
        pool_size = max(k, source_candidate_pool)
        if source_partition_routing:
            coarse_pool = self.search_hybrid_from_embedding(
                query,
                query_embedding,
                k=pool_size,
                ef_search=ef_search,
                candidates=max(candidates, pool_size),
                alpha=alpha,
            )
            partition_ids = select_source_partitions(
                coarse_pool,
                slots=source_partition_slots,
                separator=source_partition_separator,
            )
            routed_source_ids = self._retriever.source_ids_in_partitions(
                partition_ids,
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
                "routed_sources": len(routed_source_ids),
                "routed_candidates": len(pool),
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

        anchor_by_source: dict[str, str] = {}
        source_scores: dict[str, float] = {}
        first_pool_result_by_source: dict[str, RetrievalResult] = {}
        for result in pool:
            if result.turn is None:
                continue
            source_id = str(result.turn.source_id or result.turn.turn_id)
            first_pool_result_by_source.setdefault(source_id, result)
        for result in pool[:activation_k]:
            if result.turn is not None:
                source_id = str(result.turn.source_id or result.turn.turn_id)
                anchor_by_source.setdefault(source_id, result.chunk.chunk_id)
                source_scores[source_id] = max(
                    source_scores.get(source_id, 0.0), float(result.score)
                )

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
            result.chunk.chunk_id for result in [*anchors, *neighbors]
        }
        regular_source_slots = source_slots - (
            source_hsc_chunk_slots if source_hsc_activation else 0
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
            if use_source_reranker:
                source_extras = self._source_candidate_reranker.rerank(
                    query,
                    source_extras,
                top_k=regular_source_slots,
                )
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
        source_extras = [*source_extras, *hsc_extras]

        if use_attention_feedback and feedback_slots:
            # Sample every first-round route instead of letting the longest
            # route monopolize the Qwen workspace. IDs/scalars cross rounds;
            # token state does not.
            groups = [list(anchors), list(neighbors), list(source_extras)]
            attention_candidates: list[RetrievalResult] = []
            attention_seen: set[str] = set()
            position = 0
            while len(attention_candidates) < self._source_candidate_reranker.candidate_pool:
                added = False
                for group in groups:
                    if position >= len(group):
                        continue
                    result = group[position]
                    if result.chunk.chunk_id not in attention_seen:
                        attention_seen.add(result.chunk.chunk_id)
                        attention_candidates.append(result)
                        added = True
                        if len(attention_candidates) >= self._source_candidate_reranker.candidate_pool:
                            break
                if not added:
                    break
                position += 1

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
                source_id = seed.turn.source_id or seed.turn.turn_id
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
        expansion_results: Sequence[RetrievalResult] | None = None,
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

        expansions: list[RetrievalResult] = list(expansion_results or ())
        query_embedding: np.ndarray | None = None
        if expansion_results is not None:
            if use_consolidation and expansions:
                query_embedding = self._embedder.embed_query(user_text)
        elif k_expansions:
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
        try:
            if self._persist_index_on_close:
                self._retriever.save()
        finally:
            self._retriever.release()
            self._db.close()

    def __enter__(self) -> MemoryCondenser:
        return self

    def __exit__(self, *args) -> None:
        self.close()
