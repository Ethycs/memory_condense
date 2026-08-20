from __future__ import annotations

import gc
import hmac
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from memory_condense.domain.schemas import Chunk
from memory_condense.modeling.checkpoint_identity import (
    checkpoint_manifest_sha256,
    verify_file_sha256,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

#: The model this service is built around, and its known dimensionality.
#: Used only to answer ``dim`` cheaply before the model is loaded.
DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
DEFAULT_MODEL_DIM = 1024

_EMBEDDING_CHECKPOINT_MANIFEST_FORMAT = (
    "memory-condense-sentence-transformer-checkpoint-v1"
)
BGE_M3_FILE_SHA256: dict[str, str] = {
    "1_Pooling/config.json": (
        "e54c164a07274f2eb45bb724f54a79d1efcc90c41573887cd9a29aeee0597352"
    ),
    "config_sentence_transformers.json": (
        "1eef72430e7194a1e59680e635aed81ffa083f05668dbc5bb1c56c04c0999c38"
    ),
    "config.json": (
        "26159e7ad065073448460117eb24b7a4572f6f4e78eadff65dc0a11c052449fa"
    ),
    "modules.json": (
        "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf"
    ),
    "pytorch_model.bin": (
        "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38"
    ),
    "sentence_bert_config.json": (
        "eb9b44b13c0f52a3b3685c3b1cbdea1ba8b04bea123b98f61610048940776eb1"
    ),
    "sentencepiece.bpe.model": (
        "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865"
    ),
    "special_tokens_map.json": (
        "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835"
    ),
    "tokenizer_config.json": (
        "a62b2b6784f990259fddef5f16388693a8043be4f69179e6a5257eeb3f9abac4"
    ),
    "tokenizer.json": (
        "21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08"
    ),
}


def _embedding_checkpoint_manifest_sha256(
    file_hashes: Mapping[str, str],
    *,
    model_id: str,
    model_revision: str,
) -> str:
    return checkpoint_manifest_sha256(
        file_hashes,
        manifest_format=_EMBEDDING_CHECKPOINT_MANIFEST_FORMAT,
        model_id=model_id,
        model_revision=model_revision,
    )


BGE_M3_CHECKPOINT_SHA256 = _embedding_checkpoint_manifest_sha256(
    BGE_M3_FILE_SHA256,
    model_id=DEFAULT_MODEL_NAME,
    model_revision=DEFAULT_MODEL_REVISION,
)


def verify_bge_m3_checkpoint(
    model_dir: str | Path | None = None,
    *,
    expected_checkpoint_sha256: str = BGE_M3_CHECKPOINT_SHA256,
    expected_file_sha256: Mapping[str, str] = BGE_M3_FILE_SHA256,
) -> str:
    """Verify the exact local BGE-M3 model consumed by evaluation.

    When ``model_dir`` is omitted, Hugging Face resolves only the already
    downloaded pinned snapshot.  ``EmbeddingService`` calls this after its
    normal pinned load, so a first use may download the revision but no later
    evaluation can silently switch weights or tokenizer/config files.
    """

    if model_dir is None:
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(
            repo_id=DEFAULT_MODEL_NAME,
            revision=DEFAULT_MODEL_REVISION,
            local_files_only=True,
        )
    root = Path(model_dir).resolve()
    actual_files: dict[str, str] = {}
    for relative, expected_file_digest in sorted(expected_file_sha256.items()):
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"embedding checkpoint path escapes model directory: {relative}"
            )
        # Hugging Face snapshots commonly symlink these safe relative names to
        # a content-addressed blob directory outside the snapshot.  Hash the
        # target but do not reject that standard cache layout.
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"incomplete BGE-M3 checkpoint under {root}: missing {relative}"
            )
        actual_files[relative] = verify_file_sha256(
            path,
            expected_file_digest,
            name=relative,
            context="BGE-M3",
        )
    actual = _embedding_checkpoint_manifest_sha256(
        actual_files,
        model_id=DEFAULT_MODEL_NAME,
        model_revision=DEFAULT_MODEL_REVISION,
    )
    expected = str(expected_checkpoint_sha256).strip().casefold()
    if expected and not hmac.compare_digest(actual, expected):
        raise ValueError(
            f"unexpected BGE-M3 checkpoint SHA-256: {actual}; expected {expected}"
        )
    return actual


class EmbeddingService:
    """Wraps BAAI/bge-m3 via sentence-transformers for dense embeddings.

    The model is loaded lazily on first use to keep imports fast.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        batch_size: int = 32,
        model_revision: str | None = None,
        verify_checkpoint: bool | None = None,
    ) -> None:
        self._model_name = model_name
        self._model_revision = (
            DEFAULT_MODEL_REVISION
            if model_revision is None and model_name == DEFAULT_MODEL_NAME
            else model_revision
        )
        self._checkpoint_sha256 = (
            BGE_M3_CHECKPOINT_SHA256
            if model_name == DEFAULT_MODEL_NAME
            and self._model_revision == DEFAULT_MODEL_REVISION
            else ""
        )
        pinned_default = bool(self._checkpoint_sha256)
        if bool(verify_checkpoint) and not pinned_default:
            raise ValueError(
                "checkpoint verification is available only for the pinned "
                "default BGE-M3 revision"
            )
        self._verify_checkpoint = (
            pinned_default if verify_checkpoint is None else bool(verify_checkpoint)
        )
        self._device = device
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None
        self._verified_checkpoint_sha256: str | None = None
        self._dim: int | None = None

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            kwargs: dict = {}
            if self._device is not None:
                kwargs["device"] = self._device
            if self._model_revision is not None:
                kwargs["revision"] = self._model_revision
            model = SentenceTransformer(self._model_name, **kwargs)
            verified_checkpoint_sha256: str | None = None
            if self._verify_checkpoint:
                try:
                    actual = verify_bge_m3_checkpoint()
                except BaseException:
                    del model
                    raise
                if self._checkpoint_sha256 and actual != self._checkpoint_sha256:
                    del model
                    raise ValueError("loaded BGE-M3 checkpoint identity mismatch")
                verified_checkpoint_sha256 = actual
            self._verified_checkpoint_sha256 = verified_checkpoint_sha256
            self._model = model
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

        return [
            chunk.model_copy(update={"embedding": dense_vecs[i].tolist()})
            for i, chunk in enumerate(chunks)
        ]

    def embed_query(self, query: str) -> np.ndarray:
        """Compute a dense embedding for a single query string.

        Returns a 1-D numpy array of shape (dim,).
        """
        model = self._load_model()
        return model.encode([query], normalize_embeddings=False)[0]

    def embed_queries(self, queries: Sequence[str]) -> np.ndarray:
        """Embed a query batch in one model call for evaluation fan-out."""
        if not queries:
            return np.zeros((0, self.dim), dtype=np.float32)
        model = self._load_model()
        return np.asarray(
            model.encode(
                list(queries),
                batch_size=self._batch_size,
                normalize_embeddings=False,
            ),
            dtype=np.float32,
        )

    def close(self) -> None:
        """Release lazily loaded model weights before another GPU stage.

        The memory pipeline normally keeps one embedder resident.  Staged
        experiments are different: they batch all retrieval queries first,
        then give the same GPU to a Qwen prefix teacher.  Explicit release
        prevents the two models from competing for accelerator memory.
        """

        model = self._model
        self._model = None
        self._verified_checkpoint_sha256 = None
        if model is None:
            return
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover - torch is an optional import here
            pass

    @property
    def model_name(self) -> str:
        """The configured sentence-transformers model id."""
        return self._model_name

    @property
    def model_revision(self) -> str | None:
        """The exact Hugging Face revision, when the model is pinned."""

        return self._model_revision

    @property
    def checkpoint_sha256(self) -> str:
        """Expected full behavioral checkpoint manifest SHA-256."""

        return self._checkpoint_sha256

    @property
    def execution_identity(self) -> dict[str, str | int | bool]:
        """Cache-relevant execution controls for chunk and query vectors.

        Validation policies provide an explicit device, so its cache identity
        is resolved rather than inherited from whatever accelerator happens to
        be visible.  ``auto`` remains explicit for non-validation callers that
        deliberately leave device selection to sentence-transformers.
        """

        return {
            "backend": "sentence-transformers.encode-v1",
            "device": str(self._device or "auto").casefold(),
            "batch_size": self._batch_size,
            "normalize_embeddings": False,
            "output_dtype": "float32",
        }

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

        model = self._load_model()
        getter = getattr(model, "get_embedding_dimension", None)
        if getter is None:  # sentence-transformers < the rename
            getter = model.get_sentence_embedding_dimension
        reported = getter()
        if reported is None:
            # Some wrappers do not expose it; fall back to encoding a probe.
            reported = len(self.embed_query(""))
        self._dim = int(reported)
        return self._dim
