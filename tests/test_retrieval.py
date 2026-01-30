import numpy as np
import pytest

from memory_condense.retrieval import SimilarityRetriever
from memory_condense.schemas import Chunk
from memory_condense.transcript_store import TranscriptStore


def _make_chunk(turn_id: str, text: str, dim: int = 16) -> Chunk:
    """Create a chunk with a random embedding for testing."""
    rng = np.random.default_rng(hash(text) % (2**32))
    vec = rng.standard_normal(dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)  # normalize for cosine
    return Chunk(
        turn_id=turn_id,
        text=text,
        start_char=0,
        end_char=len(text),
        token_count=len(text.split()),
        embedding=vec.tolist(),
    )


@pytest.fixture
def retriever(db):
    return SimilarityRetriever(db=db, dim=16, max_elements=100)


def test_add_and_query(db, retriever):
    # Insert a turn first (FK constraint)
    store = TranscriptStore(db)
    turn = store.append("user", "hello world")

    chunk = _make_chunk(turn.turn_id, "hello world", dim=16)
    retriever.add_chunks([chunk])

    # Query with the same embedding
    query_vec = np.array(chunk.embedding, dtype=np.float32)
    results = retriever.query(query_vec, k=1)
    assert len(results) == 1
    assert results[0].chunk.chunk_id == chunk.chunk_id
    assert results[0].score > 0.99  # same vector


def test_empty_query(retriever):
    query_vec = np.random.randn(16).astype(np.float32)
    results = retriever.query(query_vec, k=5)
    assert results == []


def test_idempotent_add(db, retriever):
    store = TranscriptStore(db)
    turn = store.append("user", "test")
    chunk = _make_chunk(turn.turn_id, "test text", dim=16)

    retriever.add_chunks([chunk])
    retriever.add_chunks([chunk])  # should be a no-op

    query_vec = np.array(chunk.embedding, dtype=np.float32)
    results = retriever.query(query_vec, k=10)
    assert len(results) == 1


def test_multiple_chunks_ranked(db, retriever):
    store = TranscriptStore(db)
    turn = store.append("user", "multiple test")

    chunks = [_make_chunk(turn.turn_id, f"chunk {i}", dim=16) for i in range(5)]
    retriever.add_chunks(chunks)

    # Query with the first chunk's embedding
    query_vec = np.array(chunks[0].embedding, dtype=np.float32)
    results = retriever.query(query_vec, k=5)
    assert len(results) == 5
    # First result should be the best match
    assert results[0].chunk.chunk_id == chunks[0].chunk_id
    # Scores should be descending
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_save_and_rebuild(db, tmp_dir):
    store = TranscriptStore(db)
    turn = store.append("user", "persistence test")

    index_path = tmp_dir / "test_index.bin"
    retriever = SimilarityRetriever(
        db=db, dim=16, index_path=index_path, max_elements=100
    )

    chunk = _make_chunk(turn.turn_id, "persistent chunk", dim=16)
    retriever.add_chunks([chunk])
    retriever.save()

    # Create a new retriever that loads the saved index
    retriever2 = SimilarityRetriever(
        db=db, dim=16, index_path=index_path, max_elements=100
    )
    query_vec = np.array(chunk.embedding, dtype=np.float32)
    results = retriever2.query(query_vec, k=1)
    assert len(results) == 1
    assert results[0].chunk.chunk_id == chunk.chunk_id
