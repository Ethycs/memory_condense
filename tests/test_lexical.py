"""Tests for BM25 lexical retrieval (no model required)."""

from __future__ import annotations

import pytest

from memory_condense.search.indexes.lexical import (
    BM25_B,
    BM25_K1,
    STOPWORDS,
    LexicalIndex,
    term_frequencies,
    tokenize,
)
from memory_condense.domain.schemas import Chunk
from memory_condense.persistence.transcript_store import TranscriptStore


def _make_chunk(turn_id: str, text: str) -> Chunk:
    return Chunk(
        turn_id=turn_id,
        text=text,
        start_char=0,
        end_char=len(text),
        token_count=len(text.split()),
    )


@pytest.fixture
def turn_id(db):
    return TranscriptStore(db).append("user", "lexical fixture turn").turn_id


@pytest.fixture
def index(db):
    return LexicalIndex(db)


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("Hello, World! Postgres-15") == [
        "hello",
        "world",
        "postgres",
        "15",
    ]


def test_tokenize_splits_underscores():
    assert tokenize("snake_case_name") == ["snake", "case", "name"]


def test_tokenize_drops_single_characters():
    assert tokenize("a b cd 1 22") == ["cd", "22"]


def test_tokenize_drops_stopwords():
    assert "the" in STOPWORDS
    assert tokenize("the quick brown fox") == ["quick", "brown", "fox"]


def test_tokenize_is_deterministic():
    text = "Deploy the service to us-east-1 at 03:00 UTC."
    assert tokenize(text) == tokenize(text)


def test_tokenize_keeps_negations():
    # "not"/"no" carry meaning for preferences, so they must survive.
    assert tokenize("do not use tabs") == ["not", "use", "tabs"]


def test_term_frequencies_counts_repeats():
    assert term_frequencies("kafka kafka broker") == {"kafka": 2, "broker": 1}


def test_term_frequencies_empty_text():
    assert term_frequencies("the a of") == {}


# ---------------------------------------------------------------------------
# Index writes
# ---------------------------------------------------------------------------


def test_add_chunks_writes_postings_and_term_count(db, index, turn_id):
    chunk = _make_chunk(turn_id, "kafka kafka broker settings")
    index.add_chunks([chunk])

    rows = db.execute(
        "SELECT term, tf FROM chunk_terms WHERE chunk_id = ? ORDER BY term",
        (chunk.chunk_id,),
    ).fetchall()
    assert rows == [("broker", 1), ("kafka", 2), ("settings", 1)]

    term_count = db.execute(
        "SELECT term_count FROM chunks WHERE chunk_id = ?", (chunk.chunk_id,)
    ).fetchone()[0]
    assert term_count == 4


def test_add_chunks_is_idempotent(db, index, turn_id):
    chunk = _make_chunk(turn_id, "idempotent kafka broker")
    index.add_chunks([chunk])
    index.add_chunks([chunk])
    index.add_chunks([chunk])

    count = db.execute(
        "SELECT COUNT(*) FROM chunk_terms WHERE chunk_id = ?", (chunk.chunk_id,)
    ).fetchone()[0]
    assert count == 3

    term_count = db.execute(
        "SELECT term_count FROM chunks WHERE chunk_id = ?", (chunk.chunk_id,)
    ).fetchone()[0]
    assert term_count == 3


def test_add_chunks_empty_list_is_noop(db, index):
    index.add_chunks([])
    assert db.execute("SELECT COUNT(*) FROM chunk_terms").fetchone()[0] == 0


def test_delete_chunk_removes_postings(db, index, turn_id):
    chunk = _make_chunk(turn_id, "ephemeral kafka broker")
    index.add_chunks([chunk])
    index.delete_chunk(chunk.chunk_id)

    assert (
        db.execute(
            "SELECT COUNT(*) FROM chunk_terms WHERE chunk_id = ?", (chunk.chunk_id,)
        ).fetchone()[0]
        == 0
    )
    assert (
        db.execute(
            "SELECT term_count FROM chunks WHERE chunk_id = ?", (chunk.chunk_id,)
        ).fetchone()[0]
        is None
    )
    assert index.search("kafka") == []


def test_rebuild_from_stored_text(db, index, turn_id):
    chunks = [
        _make_chunk(turn_id, "alpha beta"),
        _make_chunk(turn_id, "beta gamma"),
    ]
    index.add_chunks(chunks)
    db.execute("DELETE FROM chunk_terms")
    db.commit()

    assert index.rebuild() == 2
    assert index.stats()["postings"] == 4
    assert [cid for cid, _ in index.search("alpha")] == [chunks[0].chunk_id]


def test_stats_reports_index_size(index, turn_id):
    index.add_chunks(
        [_make_chunk(turn_id, "alpha beta"), _make_chunk(turn_id, "beta gamma")]
    )
    stats = index.stats()
    assert stats["chunks"] == 2
    assert stats["postings"] == 4
    assert stats["distinct_terms"] == 3
    assert stats["avg_term_count"] == 2.0


# ---------------------------------------------------------------------------
# BM25 search
# ---------------------------------------------------------------------------


def test_search_empty_index(index):
    assert index.search("anything") == []


def test_search_stopword_only_query(index, turn_id):
    index.add_chunks([_make_chunk(turn_id, "kafka broker settings")])
    assert index.search("the of a") == []


def test_search_ranks_exact_keyword_first(index, turn_id):
    """A rare literal token must win, which is exactly what dense misses."""
    target = _make_chunk(
        turn_id, "The retention setting for topic zebra-9917 is seven days."
    )
    distractors = [
        _make_chunk(turn_id, "We discussed message queue retention policies broadly."),
        _make_chunk(turn_id, "Topic configuration is stored in the broker settings."),
        _make_chunk(turn_id, "Seven days felt like a reasonable default to everyone."),
    ]
    index.add_chunks([*distractors, target])

    ranked = index.search("zebra9917 zebra 9917")
    assert ranked
    assert ranked[0][0] == target.chunk_id
    assert ranked[0][1] > 0


def test_search_only_returns_chunks_with_a_query_term(index, turn_id):
    hit = _make_chunk(turn_id, "kafka broker")
    miss = _make_chunk(turn_id, "postgres replica")
    index.add_chunks([hit, miss])

    ranked = index.search("kafka")
    assert [cid for cid, _ in ranked] == [hit.chunk_id]


def test_search_respects_limit(index, turn_id):
    index.add_chunks([_make_chunk(turn_id, f"shared token{i}") for i in range(5)])
    assert len(index.search("shared", limit=2)) == 2
    assert index.search("shared", limit=0) == []


def test_search_scores_are_descending(index, turn_id):
    index.add_chunks(
        [
            _make_chunk(turn_id, "kafka kafka kafka broker"),
            _make_chunk(turn_id, "kafka broker"),
            _make_chunk(turn_id, "kafka " + " ".join(f"filler{i}" for i in range(40))),
        ]
    )
    scores = [s for _, s in index.search("kafka broker")]
    assert scores == sorted(scores, reverse=True)


def test_bm25_matches_reference_formula(index, turn_id):
    """Two documents, one query term: check the closed-form score."""
    from math import log

    hit = _make_chunk(turn_id, "kafka kafka")
    other = _make_chunk(turn_id, "postgres replica")
    index.add_chunks([hit, other])

    n_docs, avgdl = index.corpus_stats()
    assert (n_docs, avgdl) == (2, 2.0)

    df = 1
    idf = log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
    tf, doc_len = 2.0, 2.0
    expected = idf * (
        tf * (BM25_K1 + 1.0) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / avgdl))
    )

    ranked = index.search("kafka")
    assert ranked[0][0] == hit.chunk_id
    assert ranked[0][1] == pytest.approx(expected)


def test_longer_documents_are_penalised(index, turn_id):
    short = _make_chunk(turn_id, "kafka broker")
    long = _make_chunk(
        turn_id, "kafka broker " + " ".join(f"padding{i}" for i in range(60))
    )
    index.add_chunks([short, long])

    ranked = dict(index.search("kafka broker"))
    assert ranked[short.chunk_id] > ranked[long.chunk_id]


def test_search_sources_scans_only_selected_partition_and_bounds_each(db, index):
    transcript = TranscriptStore(db)
    alpha = transcript.append("user", "alpha", source_id="session-alpha")
    beta = transcript.append("user", "beta", source_id="session-beta")
    alpha_target = _make_chunk(alpha.turn_id, "cerulean launch code")
    alpha_other = _make_chunk(alpha.turn_id, "cerulean background note")
    beta_target = _make_chunk(beta.turn_id, "cerulean launch code repeated")
    index.add_chunks([alpha_other, beta_target, alpha_target])

    ranked = index.search_sources(
        "cerulean launch code",
        ["session-alpha"],
        limit_per_source=1,
    )

    assert list(ranked) == ["session-alpha"]
    assert ranked["session-alpha"][0][0] == alpha_target.chunk_id
    assert len(ranked["session-alpha"]) == 1


def test_source_tfisf_promotes_rare_terms_across_live_sources(db, index):
    transcript = TranscriptStore(db)
    alpha = transcript.append("user", "alpha", source_id="session-alpha")
    beta = transcript.append("user", "beta", source_id="session-beta")
    gamma = transcript.append("user", "gamma", source_id="session-gamma")
    index.add_chunks(
        [
            _make_chunk(alpha.turn_id, "shared deployment cerulean cerulean"),
            _make_chunk(beta.turn_id, "shared deployment ordinary"),
            _make_chunk(gamma.turn_id, "shared deployment routine"),
        ]
    )

    assert index.search_source_tfisf("cerulean deployment", limit=2)[0][0] == (
        "session-alpha"
    )


def test_source_tfisf_updates_without_a_rebuild(db, index):
    transcript = TranscriptStore(db)
    first = transcript.append("user", "first", source_id="source-first")
    index.add_chunks([_make_chunk(first.turn_id, "orchid")])
    assert [source for source, _score in index.search_source_tfisf("orchid")] == [
        "source-first"
    ]

    second = transcript.append("user", "second", source_id="source-second")
    index.add_chunks([_make_chunk(second.turn_id, "orchid orchid")])
    assert {source for source, _score in index.search_source_tfisf("orchid")} == {
        "source-first",
        "source-second",
    }
