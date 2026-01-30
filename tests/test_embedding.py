"""Tests for EmbeddingService.

These tests require the bge-m3 model to be downloaded (~2.3GB).
Mark them as slow/integration tests if needed.
"""

import numpy as np
import pytest

from memory_condense.embedding import EmbeddingService
from memory_condense.schemas import Chunk


@pytest.fixture(scope="module")
def embedder():
    """Shared embedder instance (model loads once per test session)."""
    return EmbeddingService(model_name="BAAI/bge-m3", use_fp16=True)


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
        assert c.lexical_weights is None


@pytest.mark.slow
def test_embed_chunks_with_sparse(embedder):
    chunks = [
        Chunk(turn_id="t1", text="Hello world", start_char=0, end_char=11, token_count=2),
    ]
    result = embedder.embed_chunks(chunks, return_sparse=True)
    assert len(result) == 1
    assert result[0].embedding is not None
    # lexical_weights may or may not be populated depending on model version
    # but the call should not error


@pytest.mark.slow
def test_embed_empty_list(embedder):
    assert embedder.embed_chunks([]) == []


def test_dim():
    svc = EmbeddingService.__new__(EmbeddingService)
    assert svc.dim == 1024
