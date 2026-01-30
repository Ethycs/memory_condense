from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from memory_condense.schemas import Chunk

if TYPE_CHECKING:
    from FlagEmbedding import BGEM3FlagModel


class EmbeddingService:
    """Wraps BAAI/bge-m3 via FlagEmbedding for dense + sparse embeddings.

    The model is loaded lazily on first use to keep imports fast.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._use_fp16 = use_fp16
        self._device = device
        self._batch_size = batch_size
        self._model: BGEM3FlagModel | None = None

    def _load_model(self) -> BGEM3FlagModel:
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            kwargs: dict = {
                "model_name_or_path": self._model_name,
                "use_fp16": self._use_fp16,
            }
            if self._device is not None:
                kwargs["device"] = self._device
            self._model = BGEM3FlagModel(**kwargs)
        return self._model

    def embed_chunks(
        self,
        chunks: list[Chunk],
        return_sparse: bool = False,
    ) -> list[Chunk]:
        """Compute embeddings for chunks.

        Returns new Chunk objects with embedding (and optionally
        lexical_weights) fields populated.
        """
        if not chunks:
            return []

        model = self._load_model()
        texts = [c.text for c in chunks]

        output = model.encode(
            texts,
            batch_size=self._batch_size,
            return_dense=True,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
        )

        dense_vecs: np.ndarray = output["dense_vecs"]

        result: list[Chunk] = []
        for i, chunk in enumerate(chunks):
            embedding = dense_vecs[i].tolist()

            lexical_weights = None
            if return_sparse and "lexical_weights" in output:
                raw_weights = output["lexical_weights"][i]
                # Convert token IDs to strings
                tokenizer = model.tokenizer
                lexical_weights = {
                    tokenizer.decode([int(token_id)]): float(weight)
                    for token_id, weight in raw_weights.items()
                }

            # Create a new Chunk with embedding data
            result.append(
                Chunk(
                    chunk_id=chunk.chunk_id,
                    turn_id=chunk.turn_id,
                    text=chunk.text,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    token_count=chunk.token_count,
                    embedding=embedding,
                    lexical_weights=lexical_weights,
                )
            )

        return result

    def embed_query(self, query: str) -> np.ndarray:
        """Compute a dense embedding for a single query string.

        Returns a 1-D numpy array of shape (dim,).
        """
        model = self._load_model()
        output = model.encode(
            [query],
            batch_size=1,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return output["dense_vecs"][0]

    @property
    def dim(self) -> int:
        """Embedding dimensionality (1024 for bge-m3)."""
        return 1024
