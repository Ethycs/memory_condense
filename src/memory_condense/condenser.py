from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

from memory_condense.chunker import Chunker
from memory_condense.context_packer import ContextBudget, ContextPacker
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
        self._retriever = SimilarityRetriever(
            db=self._db,
            dim=self._embedder.dim,
            index_path=data_dir / "hnsw_index.bin",
        )
        self._memory = MemoryStore(self._db, embedder=self._embedder)
        self._validator = Validator(self._db)
        self._extractor = extractor if extractor is not None else RuleBasedExtractor()
        self._packer = ContextPacker(budget)
        self._auto_extract = auto_extract

    # -- ingestion ----------------------------------------------------------

    def ingest(self, role: str, text: str) -> tuple[Turn, list[Chunk]]:
        """Ingest a single conversation turn.

        Stores the turn, chunks and embeds the text, indexes the chunks for
        both dense and lexical retrieval, and — when ``auto_extract`` is on —
        proposes memory items, validates their provenance, and applies the
        surviving ops.
        """
        turn = self._transcript.append(role, text)
        chunks = self._chunker.chunk_turn(turn.turn_id, text)

        if chunks:
            chunks = self._embedder.embed_chunks(chunks)
            self._retriever.add_chunks(chunks)

        if self._auto_extract:
            self.extract_memory([turn], chunks)

        return turn, chunks

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
        return self._retriever.hybrid_query(
            query_text=query,
            query_embedding=query_embedding,
            k=k,
            ef_search=ef_search,
            candidates=candidates,
            alpha=alpha,
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
        k_expansions: int = 3,
        hybrid: bool = True,
    ) -> PackedContext:
        """Assemble a token-budgeted prompt for ``user_text``.

        Memory header + recent window + verbatim expansions, each capped
        independently so context cost stays predictable as the conversation
        grows.
        """
        memories = self.recall_memories(user_text, k=k_memories) if k_memories else []

        expansions: list[RetrievalResult] = []
        if k_expansions:
            search = self.search_hybrid if hybrid else self.search
            expansions = search(user_text, k=k_expansions)

        turns = self._transcript.get_recent(recent_turns) if recent_turns else []
        recent = [(t.role, t.text) for t in turns]

        return self._packer.pack(
            system_prompt=system_prompt,
            memories=memories,
            recent_turns=recent,
            expansions=expansions,
            user_text=user_text,
        )

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
    def validator(self) -> Validator:
        """Access the provenance validator directly."""
        return self._validator

    def heat_counts(self, now_turn: int | None = None) -> dict[str, int]:
        """Current HOT/WARM/COLD distribution of active memory items."""
        return self._memory.heat_counts(now_turn=now_turn)

    def close(self) -> None:
        """Persist index and close database."""
        self._retriever.save()
        self._db.close()

    def __enter__(self) -> MemoryCondenser:
        return self

    def __exit__(self, *args) -> None:
        self.close()
