"""Parity and immutability tests for the resident exact BM25 index."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from memory_condense.domain.schemas import Chunk
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.search.hot_lexical import ResidentBM25Index
from memory_condense.search.indexes.lexical import LexicalIndex


def _chunk(turn_id: str, chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        turn_id=turn_id,
        text=text,
        start_char=0,
        end_char=len(text),
        token_count=len(text.split()),
    )


def _indexed_path(tmp_path: Path) -> Path:
    path = tmp_path / "resident-bm25.db"
    with Database(path) as database:
        turn = TranscriptStore(database).append("user", "resident BM25 fixture")
        LexicalIndex(database).add_chunks(
            [
                _chunk(
                    turn.turn_id,
                    "chunk-a",
                    "kafka kafka broker retention zebra-9917 seven days",
                ),
                _chunk(
                    turn.turn_id,
                    "chunk-b",
                    "kafka broker replication settings",
                ),
                _chunk(
                    turn.turn_id,
                    "chunk-c",
                    "postgres replica retention policy",
                ),
                _chunk(
                    turn.turn_id,
                    "chunk-d",
                    "The quick brown fox discussed seven ordinary days",
                ),
                _chunk(
                    turn.turn_id,
                    "chunk-e",
                    "retention " + " ".join(f"filler{value}" for value in range(30)),
                ),
            ]
        )
    return path


@pytest.mark.parametrize(
    ("query", "limit"),
    [
        ("kafka broker", 100),
        ("zebra9917 zebra-9917", 3),
        ("RETENTION seven", 2),
        ("kafka kafka kafka", 1),
        ("postgres unknown-token", 20),
        ("the of a", 100),
        ("completely absent", 100),
    ],
)
def test_resident_search_has_exact_lexical_parity(
    tmp_path: Path,
    query: str,
    limit: int,
) -> None:
    path = _indexed_path(tmp_path)
    with Database(path, read_only=True) as database:
        durable = LexicalIndex(database, k1=0.9, b=0.4)
        resident = ResidentBM25Index(durable)
        expected = durable.search(query, limit=limit)

    observed = resident.search(query, limit=limit)
    assert [chunk_id for chunk_id, _score in observed] == [
        chunk_id for chunk_id, _score in expected
    ]
    assert [score for _chunk_id, score in observed] == pytest.approx(
        [score for _chunk_id, score in expected],
        rel=1e-14,
        abs=1e-14,
    )


def test_database_constructor_uses_standard_bm25_and_matches_stats(
    tmp_path: Path,
) -> None:
    path = _indexed_path(tmp_path)
    with Database(path, read_only=True) as database:
        durable = LexicalIndex(database)
        resident = ResidentBM25Index(database)
        assert resident.stats() == durable.stats()
        assert resident.search("kafka retention") == pytest.approx(
            durable.search("kafka retention")
        )

    assert resident.chunk_count == 5
    assert resident.resident_bytes > 5 * np.dtype(np.uint32).itemsize


def test_empty_snapshot_and_limit_semantics(tmp_path: Path) -> None:
    path = tmp_path / "empty.db"
    with Database(path):
        pass
    with Database(path, read_only=True) as database:
        resident = ResidentBM25Index(database)

    assert resident.stats() == {
        "chunks": 0.0,
        "avg_term_count": 0.0,
        "postings": 0.0,
        "distinct_terms": 0.0,
    }
    assert resident.resident_bytes == 0
    assert resident.search("anything") == []
    assert resident.search("anything", limit=0) == []
    assert resident.search("anything", limit=-4) == []


def test_identical_score_ties_are_broken_by_chunk_id(tmp_path: Path) -> None:
    path = tmp_path / "ties.db"
    with Database(path) as database:
        turn = TranscriptStore(database).append("user", "tie fixture")
        LexicalIndex(database).add_chunks(
            [
                _chunk(turn.turn_id, "chunk-z", "saffron equal"),
                _chunk(turn.turn_id, "chunk-a", "saffron equal"),
                _chunk(turn.turn_id, "chunk-m", "saffron equal"),
            ]
        )
    with Database(path, read_only=True) as database:
        resident = ResidentBM25Index(database)

    assert [value[0] for value in resident.search("saffron")] == [
        "chunk-a",
        "chunk-m",
        "chunk-z",
    ]
    assert [value[0] for value in resident.search("saffron", limit=True)] == [
        "chunk-a"
    ]


def test_retired_postings_still_contribute_to_document_frequency(
    tmp_path: Path,
) -> None:
    path = tmp_path / "retired-posting.db"
    with Database(path) as database:
        turn = TranscriptStore(database).append("user", "retired posting fixture")
        chunks = [
            _chunk(turn.turn_id, "active-a", "orchid alpha"),
            _chunk(turn.turn_id, "active-b", "orchid beta beta"),
            _chunk(turn.turn_id, "retired", "orchid gamma"),
        ]
        LexicalIndex(database).add_chunks(chunks)
        database.execute(
            "UPDATE chunks SET term_count = NULL WHERE chunk_id = 'retired'"
        )
        database.commit()

    with Database(path, read_only=True) as database:
        durable = LexicalIndex(database)
        resident = ResidentBM25Index(durable)
        expected = durable.search("orchid")

    assert resident.search("orchid") == pytest.approx(expected, rel=1e-14, abs=1e-14)
    assert "retired" not in {chunk_id for chunk_id, _score in expected}
    assert resident.stats()["postings"] == 6.0


def test_snapshot_is_independent_and_immutable_after_compile(tmp_path: Path) -> None:
    path = _indexed_path(tmp_path)
    database = Database(path, read_only=True)
    resident = ResidentBM25Index(LexicalIndex(database))
    expected = resident.search("kafka")
    database.close()

    expected.append(("forged", 999.0))
    assert resident.search("kafka") != expected
    assert not resident._document_lengths.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        resident._document_lengths[0] = 99
    with pytest.raises(AttributeError, match="immutable"):
        resident._avgdl = 99.0


def test_compile_rejects_writable_sources(tmp_path: Path) -> None:
    with Database(tmp_path / "writable.db") as database:
        with pytest.raises(ValueError, match="read-only"):
            ResidentBM25Index(database)
        with pytest.raises(ValueError, match="read-only"):
            ResidentBM25Index(LexicalIndex(database))


def test_compile_rejects_unrelated_source() -> None:
    with pytest.raises(TypeError, match="Database or LexicalIndex"):
        ResidentBM25Index(object())  # type: ignore[arg-type]
