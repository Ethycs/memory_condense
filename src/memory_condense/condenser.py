from __future__ import annotations

from pathlib import Path

from memory_condense.chunker import Chunker
from memory_condense.db import Database
from memory_condense.embedding import EmbeddingService
from memory_condense.retrieval import SimilarityRetriever
from memory_condense.schemas import Chunk, RetrievalResult, Turn
from memory_condense.transcript_store import TranscriptStore


class MemoryCondenser:
    """High-level facade that wires together all Phase 0 components.

    Usage::

        mc = MemoryCondenser(data_dir="./data")
        mc.ingest("user", "I prefer dark mode in all my apps.")
        results = mc.search("What are the user's UI preferences?", k=5)
        for r in results:
            print(f"[{r.score:.3f}] {r.chunk.text}")
        mc.close()
    """

    def __init__(
        self,
        data_dir: str | Path = "./data",
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        chunker_min_tokens: int = 120,
        chunker_max_tokens: int = 250,
        device: str | None = None,
    ) -> None:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        self._db = Database(data_dir / "memory.db")
        self._transcript = TranscriptStore(self._db)
        self._chunker = Chunker(
            min_tokens=chunker_min_tokens,
            max_tokens=chunker_max_tokens,
        )
        self._embedder = EmbeddingService(
            model_name=model_name,
            use_fp16=use_fp16,
            device=device,
        )
        self._retriever = SimilarityRetriever(
            db=self._db,
            dim=self._embedder.dim,
            index_path=data_dir / "hnsw_index.bin",
        )

    def ingest(self, role: str, text: str) -> tuple[Turn, list[Chunk]]:
        """Ingest a single conversation turn.

        Stores the turn, chunks the text, embeds the chunks,
        and adds them to the ANN index.
        """
        turn = self._transcript.append(role, text)
        chunks = self._chunker.chunk_turn(turn.turn_id, text)

        if chunks:
            chunks = self._embedder.embed_chunks(chunks)
            self._retriever.add_chunks(chunks)

        return turn, chunks

    def search(self, query: str, k: int = 10) -> list[RetrievalResult]:
        """Search for chunks similar to the query."""
        query_embedding = self._embedder.embed_query(query)
        return self._retriever.query(query_embedding, k=k)

    @property
    def transcript(self) -> TranscriptStore:
        """Access the transcript store directly."""
        return self._transcript

    def close(self) -> None:
        """Persist index and close database."""
        self._retriever.save()
        self._db.close()

    def __enter__(self) -> MemoryCondenser:
        return self

    def __exit__(self, *args) -> None:
        self.close()
