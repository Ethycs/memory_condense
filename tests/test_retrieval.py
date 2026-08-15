import json

import numpy as np
import pytest

from memory_condense.ranking import blend_hybrid
from memory_condense.retrieval import SimilarityRetriever
from memory_condense.schemas import Chunk
from memory_condense.transcript_store import TranscriptStore

DIM = 16


def _vector(components: dict[int, float], dim: int = DIM) -> list[float]:
    """A unit vector with the given axis weights (deterministic embeddings)."""
    vec = np.zeros(dim, dtype=np.float32)
    for axis, weight in components.items():
        vec[axis] = weight
    norm = float(np.linalg.norm(vec))
    return (vec / norm).tolist()


def _chunk_with(turn_id: str, text: str, embedding: list[float]) -> Chunk:
    return Chunk(
        turn_id=turn_id,
        text=text,
        start_char=0,
        end_char=len(text),
        token_count=len(text.split()),
        embedding=embedding,
    )


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


# ---------------------------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------------------------


@pytest.fixture
def turn_id(db):
    return TranscriptStore(db).append("user", "hybrid fixture turn").turn_id


@pytest.fixture
def rescue_corpus(retriever, turn_id):
    """Three chunks where the exact-keyword hit is the *worst* dense match."""
    close = _chunk_with(
        turn_id,
        "the deployment pipeline runs nightly without supervision",
        _vector({0: 1.0}),
    )
    middling = _chunk_with(
        turn_id,
        "we schedule the builds early in the morning",
        _vector({0: 0.3, 1: 0.954}),
    )
    keyword = _chunk_with(
        turn_id,
        "the rollback token qx7f2 must be quoted verbatim",
        _vector({2: 1.0}),
    )
    retriever.add_chunks([close, middling, keyword])
    return close, middling, keyword


def test_add_chunks_persists_lexical_weights_and_terms(db, retriever, turn_id):
    chunk = _make_chunk(turn_id, "kafka kafka broker retention")
    retriever.add_chunks([chunk])

    raw = db.execute(
        "SELECT lexical_weights, term_count FROM chunks WHERE chunk_id = ?",
        (chunk.chunk_id,),
    ).fetchone()
    assert json.loads(raw[0]) == {"kafka": 2, "broker": 1, "retention": 1}
    assert raw[1] == 4

    terms = db.execute(
        "SELECT COUNT(*) FROM chunk_terms WHERE chunk_id = ?", (chunk.chunk_id,)
    ).fetchone()[0]
    assert terms == 3

    # And the hydrated chunk carries them back out.
    results = retriever.query(np.array(chunk.embedding, dtype=np.float32), k=1)
    assert results[0].chunk.lexical_weights == {
        "kafka": 2.0,
        "broker": 1.0,
        "retention": 1.0,
    }


def test_readd_does_not_duplicate_chunk_terms(db, retriever, turn_id):
    chunk = _make_chunk(turn_id, "idempotent kafka broker retention")
    retriever.add_chunks([chunk])
    retriever.add_chunks([chunk])
    retriever.lexical.add_chunks([chunk])

    count = db.execute(
        "SELECT COUNT(*) FROM chunk_terms WHERE chunk_id = ?", (chunk.chunk_id,)
    ).fetchone()[0]
    assert count == 4  # idempotent, kafka, broker, retention
    assert db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1


def test_hybrid_query_empty_index(retriever):
    results = retriever.hybrid_query("anything", np.zeros(16, dtype=np.float32), k=5)
    assert results == []


def test_hybrid_query_populates_both_component_scores(retriever, rescue_corpus):
    _, _, keyword = rescue_corpus
    results = retriever.hybrid_query(
        "qx7f2 rollback token", _vector_query({0: 1.0}), k=3, alpha=0.65
    )

    assert len(results) == 3
    for r in results:
        assert r.dense_score is not None
        assert r.lexical_score is not None
        assert r.score == pytest.approx(
            blend_hybrid(r.dense_score, r.lexical_score, 0.65)
        )

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)

    # The keyword chunk carries all the lexical signal.
    by_id = {r.chunk.chunk_id: r for r in results}
    assert by_id[keyword.chunk_id].lexical_score == pytest.approx(1.0)


def test_hybrid_rescues_exact_keyword_match(retriever, rescue_corpus):
    close, middling, keyword = rescue_corpus
    query_vec = _vector_query({0: 1.0})

    dense_order = [r.chunk.chunk_id for r in retriever.query(query_vec, k=3)]
    assert dense_order == [close.chunk_id, middling.chunk_id, keyword.chunk_id]

    hybrid_order = [
        r.chunk.chunk_id
        for r in retriever.hybrid_query("qx7f2", query_vec, k=3, alpha=0.65)
    ]
    # The literal token pulls the dense-worst chunk above the middling one.
    assert hybrid_order.index(keyword.chunk_id) < hybrid_order.index(
        middling.chunk_id
    )


def test_hybrid_alpha_zero_is_pure_lexical(retriever, rescue_corpus):
    _, _, keyword = rescue_corpus
    results = retriever.hybrid_query(
        "qx7f2", _vector_query({0: 1.0}), k=3, alpha=0.0
    )
    assert results[0].chunk.chunk_id == keyword.chunk_id
    assert results[0].score == pytest.approx(1.0)


def test_hybrid_alpha_one_reproduces_dense_ordering(db, retriever, turn_id):
    chunks = [
        _chunk_with(
            turn_id,
            f"unrelated sentence number {i} about assorted topics",
            _vector({0: 1.0 - i * 0.15, 1 + i: 0.5}),
        )
        for i in range(5)
    ]
    retriever.add_chunks(chunks)

    query_vec = _vector_query({0: 1.0})
    dense = [r.chunk.chunk_id for r in retriever.query(query_vec, k=5)]
    hybrid = [
        r.chunk.chunk_id
        for r in retriever.hybrid_query("assorted topics", query_vec, k=5, alpha=1.0)
    ]
    assert hybrid == dense


def test_hybrid_respects_k(retriever, rescue_corpus):
    assert retriever.hybrid_query("qx7f2", _vector_query({0: 1.0}), k=1) != []
    assert len(retriever.hybrid_query("qx7f2", _vector_query({0: 1.0}), k=1)) == 1
    assert retriever.hybrid_query("qx7f2", _vector_query({0: 1.0}), k=0) == []


def test_hybrid_lexical_only_candidate_is_reachable(retriever, rescue_corpus):
    """A chunk outside the dense candidate window still surfaces via BM25."""
    _, _, keyword = rescue_corpus
    results = retriever.hybrid_query(
        "qx7f2", _vector_query({0: 1.0}), k=2, candidates=1, alpha=0.4
    )
    ids = [r.chunk.chunk_id for r in results]
    assert keyword.chunk_id in ids


def test_delete_chunk_removes_from_both_indexes(db, retriever, rescue_corpus):
    close, middling, keyword = rescue_corpus

    assert retriever.delete_chunk(keyword.chunk_id) is True
    assert retriever.lexical.search("qx7f2") == []

    query_vec = _vector_query({0: 1.0})
    dense_ids = [r.chunk.chunk_id for r in retriever.query(query_vec, k=5)]
    assert keyword.chunk_id not in dense_ids
    assert set(dense_ids) == {close.chunk_id, middling.chunk_id}

    hybrid_ids = [
        r.chunk.chunk_id for r in retriever.hybrid_query("qx7f2", query_vec, k=5)
    ]
    assert keyword.chunk_id not in hybrid_ids

    row = db.execute(
        "SELECT embedding, hnsw_label, term_count FROM chunks WHERE chunk_id = ?",
        (keyword.chunk_id,),
    ).fetchone()
    assert row == (None, None, None)


def test_delete_unknown_chunk_returns_false(retriever):
    assert retriever.delete_chunk("does-not-exist") is False


def _vector_query(components: dict[int, float]) -> np.ndarray:
    return np.array(_vector(components), dtype=np.float32)
