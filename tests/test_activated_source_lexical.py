"""Exact-parity tests for the activated-source resident lexical overlay."""

from __future__ import annotations

from pathlib import Path

import pytest

from memory_condense.domain.schemas import Chunk
from memory_condense.persistence.db import TURN_SOURCE_ID_SQL, Database
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.search.activated_source_lexical import (
    ActivatedSourceLexicalIndex,
)
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
    path = tmp_path / "activated-source-bm25.db"
    with Database(path) as database:
        store = TranscriptStore(database)
        lexical = LexicalIndex(database)
        first = store.append("user", "first source", source_id="source-a")
        second = store.append("user", "second source", source_id="source-b")
        lexical.add_chunks(
            [
                _chunk(first.turn_id, "a-low", "orchid ordinary"),
                _chunk(first.turn_id, "a-high", "orchid orchid rare"),
                _chunk(second.turn_id, "b-low", "orchid plain"),
                _chunk(second.turn_id, "b-high", "orchid rare rare"),
            ]
        )
    return path


def _source_map(database: Database) -> dict[str, str]:
    rows = database.execute(
        "SELECT c.chunk_id, " + TURN_SOURCE_ID_SQL + " "
        "FROM chunks AS c JOIN turns AS t ON t.turn_id = c.turn_id "
        "WHERE c.term_count IS NOT NULL ORDER BY c.chunk_id"
    ).fetchall()
    return {str(chunk_id): str(source_id) for chunk_id, source_id in rows}


def test_search_has_exact_durable_source_parity(tmp_path: Path) -> None:
    path = _indexed_path(tmp_path)
    selected = ["source-b", "source-a", "source-b", "missing"]
    with Database(path, read_only=True) as database:
        durable = LexicalIndex(database, k1=0.9, b=0.4)
        overlay = ActivatedSourceLexicalIndex(
            ResidentBM25Index(durable),
            _source_map(database),
        )
        expected = durable.search_sources(
            "orchid rare",
            selected,
            limit_per_source=2,
        )

    observed = overlay.search_sources(
        "orchid rare",
        selected,
        limit_per_source=2,
    )
    assert list(observed) == ["source-b", "source-a", "missing"]
    assert {
        source_id: [chunk_id for chunk_id, _score in rows]
        for source_id, rows in observed.items()
    } == {
        source_id: [chunk_id for chunk_id, _score in rows]
        for source_id, rows in expected.items()
    }
    for source_id in observed:
        assert [score for _chunk_id, score in observed[source_id]] == pytest.approx(
            [score for _chunk_id, score in expected[source_id]],
            rel=1e-14,
            abs=1e-14,
        )


def test_bounds_empty_sources_and_snapshot_independence(tmp_path: Path) -> None:
    path = _indexed_path(tmp_path)
    with Database(path, read_only=True) as database:
        resident = ResidentBM25Index(database)
        source_map = _source_map(database)
        overlay = ActivatedSourceLexicalIndex(resident, source_map)

    source_map.clear()
    assert overlay.chunk_count == 4
    assert overlay.source_count == 2
    assert overlay.search_sources("orchid", ["source-a"], limit_per_source=0) == {
        "source-a": []
    }
    assert overlay.search_sources("absent", ["source-a", "missing"]) == {
        "source-a": [],
        "missing": [],
    }
    assert overlay.search_sources("orchid", ["missing"]) == {"missing": []}


def test_constructor_fails_closed_on_foreign_or_partial_maps(tmp_path: Path) -> None:
    path = _indexed_path(tmp_path)
    with Database(path, read_only=True) as database:
        resident = ResidentBM25Index(database)
        source_map = _source_map(database)

    with pytest.raises(TypeError, match="ResidentBM25Index"):
        ActivatedSourceLexicalIndex(object(), source_map)  # type: ignore[arg-type]
    source_map.pop("a-low")
    with pytest.raises(ValueError, match="complete resident snapshot"):
        ActivatedSourceLexicalIndex(resident, source_map)
