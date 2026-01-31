from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from memory_condense.schemas import Chunk

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Wraps BAAI/bge-m3 via sentence-transformers for dense embeddings.

    The model is loaded lazily on first use to keep imports fast.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            kwargs: dict = {}
            if self._device is not None:
                kwargs["device"] = self._device
            self._model = SentenceTransformer(self._model_name, **kwargs)
        return self._model

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Compute dense embeddings for chunks.

        Returns new Chunk objects with embedding fields populated.
        """
        if not chunks:
            return []

        model = self._load_model()
        texts = [c.text for c in chunks]

        dense_vecs: np.ndarray = model.encode(
            texts, batch_size=self._batch_size, normalize_embeddings=False
        )

        result: list[Chunk] = []
        for i, chunk in enumerate(chunks):
            result.append(
                Chunk(
                    chunk_id=chunk.chunk_id,
                    turn_id=chunk.turn_id,
                    text=chunk.text,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    token_count=chunk.token_count,
                    embedding=dense_vecs[i].tolist(),
                )
            )

        return result

    def embed_query(self, query: str) -> np.ndarray:
        """Compute a dense embedding for a single query string.

        Returns a 1-D numpy array of shape (dim,).
        """
        model = self._load_model()
        return model.encode([query], normalize_embeddings=False)[0]

    @property
    def dim(self) -> int:
        """Embedding dimensionality (1024 for bge-m3)."""
        return 1024
