from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from memory_condense.schemas import Chunk

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

#: The model this service is built around, and its known dimensionality.
#: Used only to answer ``dim`` cheaply before the model is loaded.
DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_MODEL_DIM = 1024


class EmbeddingService:
    """Wraps BAAI/bge-m3 via sentence-transformers for dense embeddings.

    The model is loaded lazily on first use to keep imports fast.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None
        self._dim: int | None = None

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
                    lexical_weights=chunk.lexical_weights,
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
    def model_name(self) -> str:
        """The configured sentence-transformers model id."""
        return self._model_name

    @property
    def dim(self) -> int:
        """True embedding dimensionality of the configured model.

        Resolution order:

        1. a value already resolved and cached on this instance;
        2. if the model is not loaded *and* ``model_name`` is the default
           ``BAAI/bge-m3``, the known constant 1024 — this keeps the property
           cheap so ``MemoryCondenser.__init__`` can size the hnswlib index
           without downloading ~2.3 GB of weights. The guess is not cached, so
           once the model is loaded the reported value comes from the model;
        3. otherwise the model is loaded and asked
           (``get_sentence_embedding_dimension()``), and the answer cached.

        Any non-default ``model_name`` therefore reports its real dimension
        instead of silently corrupting an index built for 1024 dimensions.
        """
        if self._dim is not None:
            return self._dim

        if self._model is None and self._model_name == DEFAULT_MODEL_NAME:
            return DEFAULT_MODEL_DIM

        reported = self._load_model().get_sentence_embedding_dimension()
        if reported is None:
            # Some wrappers do not expose it; fall back to encoding a probe.
            reported = len(self.embed_query(""))
        self._dim = int(reported)
        return self._dim
