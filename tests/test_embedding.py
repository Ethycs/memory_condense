"""Tests for EmbeddingService.

The tests marked ``slow`` require the bge-m3 model to be downloaded (~2.3GB).
Everything else runs against a stub model and must stay download-free.
"""

import hashlib
import sys
from types import ModuleType

import numpy as np
import pytest

import memory_condense.modeling.embedding as embedding_module
from memory_condense.modeling.embedding import (
    BGE_M3_CHECKPOINT_SHA256,
    DEFAULT_MODEL_DIM,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    EmbeddingService,
    verify_bge_m3_checkpoint,
)
from memory_condense.domain.schemas import Chunk


class FakeModel:
    """Minimal stand-in for a SentenceTransformer."""

    def __init__(self, dim: int | None = 384) -> None:
        self._dim = dim
        self.dim_calls = 0
        self.encode_calls = 0

    def get_sentence_embedding_dimension(self):
        self.dim_calls += 1
        return self._dim

    def encode(self, texts, **kwargs):
        self.encode_calls += 1
        dim = self._dim if self._dim is not None else 7
        return np.ones((len(texts), dim), dtype=np.float32)


def _service_with(model: FakeModel, model_name: str = "stub/model") -> EmbeddingService:
    svc = EmbeddingService(model_name=model_name)
    svc._model = model  # bypass the lazy loader; no download happens
    return svc


@pytest.fixture(scope="module")
def embedder():
    """Shared embedder instance (model loads once per test session)."""
    return EmbeddingService(model_name=DEFAULT_MODEL_NAME)


@pytest.mark.slow
def test_embed_query(embedder):
    vec = embedder.embed_query("Hello world")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (1024,)
    assert not np.all(vec == 0)


@pytest.mark.slow
def test_embed_chunks(embedder):
    chunks = [
        Chunk(turn_id="t1", text="Hello world", start_char=0, end_char=11, token_count=2),
        Chunk(turn_id="t1", text="Goodbye world", start_char=12, end_char=25, token_count=2),
    ]
    result = embedder.embed_chunks(chunks)
    assert len(result) == 2
    for c in result:
        assert c.embedding is not None
        assert len(c.embedding) == 1024


@pytest.mark.slow
def test_embed_empty_list(embedder):
    assert embedder.embed_chunks([]) == []


# ---------------------------------------------------------------------------
# dim resolution (no model download)
# ---------------------------------------------------------------------------


def test_dim_default_model_is_cheap_and_not_loaded():
    svc = EmbeddingService()
    assert svc.model_name == DEFAULT_MODEL_NAME
    assert svc.model_revision == DEFAULT_MODEL_REVISION
    assert svc.checkpoint_sha256 == BGE_M3_CHECKPOINT_SHA256
    assert svc.dim == DEFAULT_MODEL_DIM == 1024
    # The shortcut must not load the model, nor cache a guess.
    assert svc._model is None
    assert svc._dim is None


def test_dim_reflects_the_loaded_model():
    model = FakeModel(dim=384)
    svc = _service_with(model)
    assert svc.dim == 384


def test_dim_prefers_the_model_over_the_default_guess():
    """Even under the default name, a loaded model is the authority."""
    svc = _service_with(FakeModel(dim=768), model_name=DEFAULT_MODEL_NAME)
    assert svc.dim == 768


def test_dim_is_cached():
    model = FakeModel(dim=512)
    svc = _service_with(model)
    assert svc.dim == 512
    assert svc.dim == 512
    assert model.dim_calls == 1


def test_dim_falls_back_to_probe_when_model_does_not_report():
    model = FakeModel(dim=None)
    svc = _service_with(model)
    assert svc.dim == 7  # length of the probe encoding
    assert model.encode_calls == 1


def test_non_default_model_never_silently_reports_1024():
    svc = _service_with(FakeModel(dim=384), model_name="sentence-transformers/all-MiniLM-L6-v2")
    assert svc.dim != 1024


# ---------------------------------------------------------------------------
# embedding with a stub model
# ---------------------------------------------------------------------------


def test_embed_chunks_with_stub_preserves_metadata():
    svc = _service_with(FakeModel(dim=4))
    chunk = Chunk(
        turn_id="t1",
        text="hello world",
        start_char=0,
        end_char=11,
        token_count=2,
        lexical_weights={"hello": 1.0, "world": 1.0},
    )
    (result,) = svc.embed_chunks([chunk])
    assert result.chunk_id == chunk.chunk_id
    assert result.embedding is not None
    assert len(result.embedding) == 4
    assert result.lexical_weights == {"hello": 1.0, "world": 1.0}


def test_embed_queries_batches_the_model_call():
    model = FakeModel(dim=4)
    svc = _service_with(model)
    values = svc.embed_queries(["one", "two", "three"])
    assert values.shape == (3, 4)
    assert model.encode_calls == 1


def test_embed_empty_list_without_model():
    svc = EmbeddingService()
    assert svc.embed_chunks([]) == []
    assert svc._model is None


def test_default_model_load_is_revision_pinned_and_hash_verified(monkeypatch):
    calls = []
    module = ModuleType("sentence_transformers")

    def fake_sentence_transformer(model_name, **kwargs):
        calls.append((model_name, kwargs))
        return FakeModel(dim=DEFAULT_MODEL_DIM)

    module.SentenceTransformer = fake_sentence_transformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(
        embedding_module,
        "verify_bge_m3_checkpoint",
        lambda: BGE_M3_CHECKPOINT_SHA256,
    )

    svc = EmbeddingService(device="cpu")
    loaded = svc._load_model()

    assert isinstance(loaded, FakeModel)
    assert calls == [
        (
            DEFAULT_MODEL_NAME,
            {
                "device": "cpu",
                "revision": DEFAULT_MODEL_REVISION,
                "local_files_only": True,
            },
        )
    ]
    assert svc._verified_checkpoint_sha256 == BGE_M3_CHECKPOINT_SHA256
    svc.close()
    assert svc._verified_checkpoint_sha256 is None


def test_unpinned_embedding_cannot_claim_pinned_checkpoint_verification():
    svc = EmbeddingService(
        model_name="stub/custom",
        model_revision="revision-1",
    )
    assert svc.model_revision == "revision-1"
    assert svc.checkpoint_sha256 == ""

    with pytest.raises(ValueError, match="only for the pinned default"):
        EmbeddingService(
            model_name="stub/custom",
            model_revision="revision-1",
            verify_checkpoint=True,
        )


def test_embedding_checkpoint_manifest_detects_metadata_drift(tmp_path):
    config = tmp_path / "config.json"
    tokenizer = tmp_path / "tokenizer.json"
    config.write_bytes(b"config")
    tokenizer.write_bytes(b"tokenizer")
    file_hashes = {
        "config.json": hashlib.sha256(b"config").hexdigest(),
        "tokenizer.json": hashlib.sha256(b"tokenizer").hexdigest(),
    }
    expected = verify_bge_m3_checkpoint(
        tmp_path,
        expected_checkpoint_sha256="",
        expected_file_sha256=file_hashes,
    )
    assert len(expected) == 64

    tokenizer.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch for tokenizer.json"):
        verify_bge_m3_checkpoint(
            tmp_path,
            expected_checkpoint_sha256=expected,
            expected_file_sha256=file_hashes,
        )


def test_embedding_checkpoint_manifest_rejects_unsafe_relative_name(tmp_path):
    outside = tmp_path.parent / "outside.json"
    outside.write_bytes(b"outside")
    with pytest.raises(ValueError, match="escapes model directory"):
        verify_bge_m3_checkpoint(
            tmp_path,
            expected_checkpoint_sha256="",
            expected_file_sha256={
                "../outside.json": hashlib.sha256(b"outside").hexdigest()
            },
        )
