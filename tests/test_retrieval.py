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


def test_source_query_returns_complete_best_source_in_turn_order(db, retriever):
    store = TranscriptStore(db)
    a1 = store.append("user", "alpha first", source_id="session-a")
    b1 = store.append("user", "beta only", source_id="session-b")
    a2 = store.append("assistant", "alpha second", source_id="session-a")
    chunks = [
        _chunk_with(a1.turn_id, "alpha first", _vector({0: 1.0})),
        _chunk_with(b1.turn_id, "beta only", _vector({1: 1.0})),
        _chunk_with(a2.turn_id, "alpha second", _vector({0: 0.9, 2: 0.1})),
    ]
    retriever.add_chunks(chunks)

    results = retriever.source_query(_vector_query({0: 1.0}), k_sources=1)
    assert [result.chunk.text for result in results] == [
        "alpha first",
        "alpha second",
    ]
    assert all(result.route == "source" for result in results)
    assert all(result.turn.source_id == "session-a" for result in results)


def test_source_query_empty_or_zero(retriever):
    assert retriever.source_query(_vector_query({0: 1.0}), k_sources=1) == []
    assert retriever.source_query(_vector_query({0: 1.0}), k_sources=0) == []


def test_hydrate_sources_round_robins_selected_source_chunks(db, retriever):
    store = TranscriptStore(db)
    a1 = store.append("user", "alpha first", source_id="session-a")
    a2 = store.append("assistant", "alpha second", source_id="session-a")
    b1 = store.append("user", "beta first", source_id="session-b")
    retriever.add_chunks(
        [
            _chunk_with(a1.turn_id, "alpha first", _vector({0: 1.0})),
            _chunk_with(a2.turn_id, "alpha second", _vector({0: 1.0})),
            _chunk_with(b1.turn_id, "beta first", _vector({1: 1.0})),
        ]
    )

    results = retriever.hydrate_sources(
        ["session-a", "session-b"],
        source_scores={"session-a": 0.9, "session-b": 0.8},
    )

    assert [result.chunk.text for result in results] == [
        "alpha first",
        "beta first",
        "alpha second",
    ]
    assert [result.score for result in results] == [0.9, 0.8, 0.9]
    assert all(result.route == "anchored_source" for result in results)


def test_hydrate_source_neighbors_walks_ranked_shells_and_deduplicates(
    db, retriever
):
    store = TranscriptStore(db)
    turns = [
        store.append("user", f"turn {index}", source_id="session-a")
        for index in range(5)
    ]
    chunks = [
        _chunk_with(turn.turn_id, f"chunk {index}", _vector({index: 1.0}))
        for index, turn in enumerate(turns)
    ]
    retriever.add_chunks(chunks)
    anchors = [
        retriever._hydrate(chunks[2].chunk_id, score=0.9),
        retriever._hydrate(chunks[3].chunk_id, score=0.8),
    ]

    results = retriever.hydrate_source_neighbors(anchors, radius=2)

    assert [result.chunk.text for result in results] == [
        "chunk 2",
        "chunk 3",
        "chunk 1",
        "chunk 4",
        "chunk 0",
    ]
    assert [result.route for result in results[2:]] == [
        "hybrid_neighbor",
        "hybrid_neighbor",
        "hybrid_neighbor",
    ]
    assert results[2].anchor_chunk_id == chunks[2].chunk_id
    assert results[2].transition_distance == 1
    assert results[2].transition_direction == "previous"
    assert results[3].transition_direction == "next"


def test_hydrate_source_neighbors_validates_radius(retriever):
    with pytest.raises(ValueError, match="radius"):
        retriever.hydrate_source_neighbors([], radius=-1)


def test_hydrate_source_neighbors_enforces_extra_slot_budget(db, retriever):
    store = TranscriptStore(db)
    turns = [
        store.append("user", f"turn {index}", source_id="session-a")
        for index in range(3)
    ]
    chunks = [
        _chunk_with(turn.turn_id, f"chunk {index}", _vector({index: 1.0}))
        for index, turn in enumerate(turns)
    ]
    retriever.add_chunks(chunks)
    anchor = retriever._hydrate(chunks[1].chunk_id, score=0.9)

    results = retriever.hydrate_source_neighbors(
        [anchor], radius=2, max_neighbors=1
    )

    assert [result.chunk.text for result in results] == ["chunk 1", "chunk 0"]


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


# ---------------------------------------------------------------------------
# Concurrent sessions. Two Claude Code windows on one project open two
# retrievers on the same store; before labels were allocated through the
# database this crashed the second writer with
# "UNIQUE constraint failed: chunks.hnsw_label" on its first ingest.
# ---------------------------------------------------------------------------


def _seed(retriever, db, prefix, n, dim=8):
    """Add n chunks through `retriever`, returning their ids."""
    from memory_condense.schemas import Chunk, Turn

    ids = []
    for i in range(n):
        turn = Turn(role="user", text=f"{prefix} turn {i}")
        db.execute(
            "INSERT OR IGNORE INTO turns (turn_id, role, text, created_at) "
            "VALUES (?, ?, ?, ?)",
            (turn.turn_id, turn.role, turn.text, turn.created_at.isoformat()),
        )
        db.commit()
        vec = [0.0] * dim
        vec[i % dim] = 1.0
        chunk = Chunk(
            turn_id=turn.turn_id,
            text=f"{prefix} chunk {i}",
            start_char=0,
            end_char=10,
            token_count=3,
            embedding=vec,
        )
        retriever.add_chunks([chunk])
        ids.append(chunk.chunk_id)
    return ids


def test_two_retrievers_on_one_store_do_not_collide(tmp_path):
    """The regression: the second writer used to raise IntegrityError."""
    from memory_condense.db import Database
    from memory_condense.retrieval import SimilarityRetriever

    db_path = tmp_path / "shared.db"
    db_a, db_b = Database(db_path), Database(db_path)
    a = SimilarityRetriever(db=db_a, dim=8, index_path=tmp_path / "a.bin")
    b = SimilarityRetriever(db=db_b, dim=8, index_path=tmp_path / "b.bin")

    _seed(a, db_a, "alpha", 3)
    _seed(b, db_b, "beta", 3)  # used to raise here

    labels = [
        r[0]
        for r in db_a.execute(
            "SELECT hnsw_label FROM chunks WHERE hnsw_label IS NOT NULL"
        ).fetchall()
    ]
    assert len(labels) == 6
    assert len(set(labels)) == 6, "labels must be globally unique across processes"

    db_a.close()
    db_b.close()


def test_a_session_adopts_another_sessions_writes(tmp_path):
    """SQLite is the source of truth, so a live session must reconcile to it."""
    from memory_condense.db import Database
    from memory_condense.retrieval import SimilarityRetriever

    db_path = tmp_path / "shared.db"
    db_a, db_b = Database(db_path), Database(db_path)
    a = SimilarityRetriever(db=db_a, dim=8, index_path=tmp_path / "a.bin")
    b = SimilarityRetriever(db=db_b, dim=8, index_path=tmp_path / "b.bin")

    _seed(a, db_a, "alpha", 2)
    _seed(b, db_b, "beta", 2)

    probe = _vector_query({0: 1.0})
    # Each retriever should see all four vectors, not just its own two.
    assert len(a.query(probe, k=10)) == 4
    assert len(b.query(probe, k=10)) == 4

    db_a.close()
    db_b.close()


def test_label_counter_repairs_a_store_written_before_it_existed(tmp_path):
    """Older stores have labels but no counter row; allocation must not reuse them."""
    from memory_condense.db import Database
    from memory_condense.retrieval import SimilarityRetriever

    db = Database(tmp_path / "legacy.db")
    retriever = SimilarityRetriever(db=db, dim=8, index_path=tmp_path / "i.bin")
    _seed(retriever, db, "old", 3)

    # Simulate a pre-fix store: drop the counter, keep the labels.
    db.execute("DELETE FROM meta WHERE key = 'next_hnsw_label'")
    db.commit()

    fresh = SimilarityRetriever(db=db, dim=8, index_path=tmp_path / "i2.bin")
    _seed(fresh, db, "new", 2)

    labels = [
        r[0]
        for r in db.execute(
            "SELECT hnsw_label FROM chunks WHERE hnsw_label IS NOT NULL"
        ).fetchall()
    ]
    assert len(set(labels)) == len(labels) == 5
    db.close()


# ---------------------------------------------------------------------------
# Span retrieval
#
# Short conversational turns produce chunks too small to match against. These
# tests pin the properties that make pooled-span retrieval work, including the
# one that is counter-intuitive enough to be re-broken by a well-meaning
# refactor: retrieval must be stratified per level, never a single mixed pool.
# ---------------------------------------------------------------------------


#: Every span_corpus chunk reports this many tokens, so a token target of
#: SPAN_TOKEN_SIZE * n groups exactly n chunks and the tests stay readable.
CHUNK_TOKENS = 10


@pytest.fixture
def span_corpus(retriever, turn_id):
    """24 chunks in 6 topical runs of 4, contiguous in insertion order."""
    chunks = []
    for i in range(24):
        topic = i // 4
        chunk = _chunk_with(
            turn_id,
            f"turn {i} about topic {topic}",
            _vector({topic: 1.0, 15: 0.15}),
        )
        chunks.append(chunk.model_copy(update={"token_count": CHUNK_TOKENS}))
    retriever.add_chunks(chunks)
    return chunks


def test_span_vectors_pool_contiguous_runs(retriever, span_corpus):
    pooled4, members4 = retriever._span_vectors(4 * CHUNK_TOKENS)
    pooled8, members8 = retriever._span_vectors(8 * CHUNK_TOKENS)

    assert len(pooled4) == 6 and len(members4) == 6
    assert len(pooled8) == 3 and len(members8) == 3
    assert all(len(m) == 4 for m in members4)
    # Members are the chunks themselves, in conversation order.
    assert members4[0] == [c.chunk_id for c in span_corpus[:4]]


def test_span_vectors_are_unit_norm(retriever, span_corpus):
    pooled, _ = retriever._span_vectors(4 * CHUNK_TOKENS)
    assert np.allclose(np.linalg.norm(pooled, axis=1), 1.0, atol=1e-5)


def test_span_query_returns_real_member_chunks(retriever, span_corpus):
    """Scoring is at span granularity; what comes back is ordinary chunks.

    This is what keeps provenance and ContextPacker working unchanged — no
    synthetic chunk is ever invented for a span.
    """
    results = retriever.span_query(_vector({2: 1.0}), levels=(4 * CHUNK_TOKENS,), k_per_level=1)

    known = {c.chunk_id for c in span_corpus}
    assert results
    assert all(r.chunk.chunk_id in known for r in results)
    # The winning span is the topic-2 run.
    assert all("topic 2" in r.chunk.text for r in results)


def test_span_query_stratifies_across_levels(retriever, span_corpus):
    """Each level contributes its own top-k rather than competing in one pool."""
    one = retriever.span_query(_vector({1: 1.0}), levels=(4 * CHUNK_TOKENS,), k_per_level=1)
    both = retriever.span_query(_vector({1: 1.0}), levels=(4 * CHUNK_TOKENS, 8 * CHUNK_TOKENS), k_per_level=1)

    assert len(both) > len(one)


def test_a_chunk_reachable_from_two_levels_keeps_its_best_score(
    retriever, span_corpus
):
    results = retriever.span_query(_vector({0: 1.0}), levels=(4 * CHUNK_TOKENS, 8 * CHUNK_TOKENS), k_per_level=1)
    by_id = {}
    for r in results:
        by_id.setdefault(r.chunk.chunk_id, []).append(r.score)
    assert all(len(v) == 1 for v in by_id.values()), "a chunk was returned twice"


def test_span_cache_is_extended_when_chunks_are_added(retriever, span_corpus, turn_id):
    before, _ = retriever._span_vectors(4 * CHUNK_TOKENS)
    before = before.copy()
    high_water = retriever._span_cached_through_rowid[4 * CHUNK_TOKENS]
    retriever.add_chunks(
        [_chunk_with(turn_id, "a later turn", _vector({7: 1.0}))]
    )
    after, _ = retriever._span_vectors(4 * CHUNK_TOKENS)

    assert len(after) == len(before) + 1
    assert np.array_equal(after[: len(before)], before)
    assert retriever._span_cached_through_rowid[4 * CHUNK_TOKENS] > high_water


def test_span_append_loads_only_rows_after_the_cache_high_water(
    retriever, span_corpus, turn_id, monkeypatch
):
    level = 4 * CHUNK_TOKENS
    retriever._span_vectors(level)
    high_water = retriever._span_cached_through_rowid[level]
    retriever.add_chunks(
        [_chunk_with(turn_id, "incremental tail", _vector({8: 1.0}))]
    )
    calls = []
    original = retriever._load_span_rows

    def tracked(after_rowid=None):
        calls.append(after_rowid)
        return original(after_rowid)

    monkeypatch.setattr(retriever, "_load_span_rows", tracked)
    retriever._span_vectors(level)

    assert calls == [high_water]


@pytest.mark.parametrize("appended", [2, 5])
def test_incremental_span_tail_matches_a_clean_rebuild(
    retriever, turn_id, appended
):
    level = 4 * CHUNK_TOKENS
    retriever.add_chunks(
        [
            _chunk_with(turn_id, f"seed {i}", _vector({i % 3: 1.0})).model_copy(
                update={"token_count": CHUNK_TOKENS}
            )
            for i in range(5)
        ]
    )
    retriever._span_vectors(level)
    retriever.add_chunks(
        [
            _chunk_with(turn_id, f"tail {i}", _vector({5 + i: 1.0})).model_copy(
                update={"token_count": CHUNK_TOKENS}
            )
            for i in range(appended)
        ]
    )

    incremental_vectors, incremental_members = retriever._span_vectors(level)
    incremental_vectors = incremental_vectors.copy()
    incremental_members = [list(group) for group in incremental_members]
    retriever._clear_span_cache()
    rebuilt_vectors, rebuilt_members = retriever._span_vectors(level)

    assert incremental_members == rebuilt_members
    assert np.allclose(incremental_vectors, rebuilt_vectors, atol=1e-6)


def test_span_cache_is_dropped_when_a_chunk_is_deleted(retriever, span_corpus):
    retriever._span_vectors(4 * CHUNK_TOKENS)
    retriever.delete_chunk(span_corpus[0].chunk_id)
    _, members = retriever._span_vectors(4 * CHUNK_TOKENS)

    assert span_corpus[0].chunk_id not in [m for run in members for m in run]


def test_span_query_on_an_empty_store_returns_nothing(retriever):
    assert retriever.span_query(_vector({0: 1.0})) == []


def test_span_query_tolerates_a_short_final_run(retriever, turn_id):
    """23 chunks grouped 4-at-a-time leaves a run of 3 — still searchable."""
    retriever.add_chunks(
        [
            _chunk_with(turn_id, f"turn {i}", _vector({i % 5: 1.0})).model_copy(
                update={"token_count": CHUNK_TOKENS}
            )
            for i in range(23)
        ]
    )
    pooled, members = retriever._span_vectors(4 * CHUNK_TOKENS)

    assert len(pooled) == 6
    assert len(members[-1]) == 3
    assert np.isclose(np.linalg.norm(pooled[-1]), 1.0, atol=1e-5)


def test_span_grouping_follows_tokens_not_chunk_counts(retriever, turn_id):
    """One setting has to work on 27-token turns and 227-token prose alike.

    Counting chunks instead would make `span=4` a ~110-token span on dialogue
    and a ~900-token span on long-form — helping one corpus and wrecking the
    other with the same configuration.
    """
    retriever.add_chunks(
        [
            _chunk_with(turn_id, f"big turn {i}", _vector({i % 4: 1.0})).model_copy(
                update={"token_count": 200}
            )
            for i in range(8)
        ]
    )
    pooled, members = retriever._span_vectors(220)

    # Chunks are already at the target, so each stands alone rather than
    # being merged into a span eight times too coarse.
    assert len(pooled) == 8
    assert all(len(m) == 1 for m in members)


def test_shorter_text_scores_higher_cosine_than_a_span_containing_it(
    retriever, span_corpus
):
    """The measured reason retrieval must stratify rather than pool.

    Cosine is not length-invariant: a single-topic chunk matches a single-topic
    query more strongly than a span that also contains other topics. In one
    mixed-granularity pool the small chunks therefore crowd out every span —
    on LoCoMo that collapsed recall from 21.6% to 6.0%.
    """
    query = np.asarray(_vector({0: 1.0}), dtype=np.float32)

    single = retriever.query(query, k=1)[0].score
    pooled8, _ = retriever._span_vectors(8 * CHUNK_TOKENS)
    best_span = float(max(pooled8 @ query))

    assert single > best_span
