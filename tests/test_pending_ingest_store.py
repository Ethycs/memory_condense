from __future__ import annotations

import json
import sqlite3

import pytest

from memory_condense.domain.schemas import Chunk, Turn
from memory_condense.persistence.pending_ingest_store import (
    PendingIngestManifest,
    PendingIngestStore,
)
from memory_condense.persistence.transcript_store import TranscriptStore


def test_manifest_is_canonical_text_free_and_reconstructs_exact_chunks() -> None:
    turn = Turn(turn_id="turn-1", role="user", text="alpha beta gamma")
    left = Chunk(
        chunk_id="chunk-left",
        turn_id=turn.turn_id,
        text="alpha beta",
        start_char=0,
        end_char=10,
        token_count=2,
        embedding=[1.0, 0.0],
    )
    right = Chunk(
        chunk_id="chunk-right",
        turn_id=turn.turn_id,
        text="gamma",
        start_char=11,
        end_char=16,
        token_count=1,
        embedding=[0.0, 1.0],
    )

    manifest = PendingIngestManifest.build(turn, [right, left])
    payload = json.loads(manifest.canonical_json)

    assert [row["chunk_id"] for row in payload["chunks"]] == [
        "chunk-left",
        "chunk-right",
    ]
    assert "alpha" not in manifest.canonical_json
    assert "embedding" not in manifest.canonical_json
    assert PendingIngestManifest.from_json(manifest.canonical_json) == manifest
    assert manifest.reconstruct(turn) == [
        left.model_copy(update={"embedding": None}),
        right.model_copy(update={"embedding": None}),
    ]


def test_manifest_rejects_noncanonical_or_tampered_receipts() -> None:
    turn = Turn(turn_id="turn-1", role="user", text="alpha")
    chunk = Chunk(
        chunk_id="chunk-1",
        turn_id=turn.turn_id,
        text="alpha",
        start_char=0,
        end_char=5,
        token_count=1,
    )
    manifest = PendingIngestManifest.build(turn, [chunk])
    noncanonical = json.dumps(json.loads(manifest.canonical_json), indent=2)

    with pytest.raises(ValueError, match="not canonical"):
        PendingIngestManifest.from_json(noncanonical)
    with pytest.raises(ValueError, match="hash does not match"):
        manifest.reconstruct(turn.model_copy(update={"text": "omega"}))


def test_finalizer_rejects_unexpected_chunks_owned_by_manifest_turn(db) -> None:
    turn = TranscriptStore(db).append(
        "user",
        "alpha beta",
        turn_id="turn-with-extra",
    )
    expected = Chunk(
        chunk_id="expected",
        turn_id=turn.turn_id,
        text=turn.text,
        start_char=0,
        end_char=len(turn.text),
        token_count=2,
    )
    manifest = PendingIngestManifest.build(turn, [expected])
    store = PendingIngestStore(db)

    db.connection.execute("BEGIN IMMEDIATE")
    store.claim(manifest)
    for chunk_id in ("expected", "unexpected"):
        db.execute(
            "INSERT INTO chunks "
            "(chunk_id, turn_id, text, start_char, end_char, token_count, "
            "embedding, hnsw_label, term_count) VALUES (?, ?, ?, 0, ?, 2, ?, ?, 2)",
            (
                chunk_id,
                turn.turn_id,
                turn.text,
                len(turn.text),
                b"embedding",
                1 if chunk_id == "expected" else 2,
            ),
        )

    with pytest.raises(RuntimeError, match="durable topology differs"):
        store.finalize([manifest])
    db.connection.rollback()


def test_sqlite_completion_rejects_chunk_reserved_by_another_turn(db) -> None:
    transcript = TranscriptStore(db)
    first = transcript.append("user", "alpha", turn_id="first-owner")
    second = transcript.append("user", "alpha", turn_id="second-owner")
    first_chunk = Chunk(
        chunk_id="first-chunk",
        turn_id=first.turn_id,
        text=first.text,
        start_char=0,
        end_char=len(first.text),
        token_count=1,
    )
    second_chunk = first_chunk.model_copy(
        update={"chunk_id": "second-chunk", "turn_id": second.turn_id}
    )
    first_manifest = PendingIngestManifest.build(first, [first_chunk])
    second_manifest = PendingIngestManifest.build(second, [second_chunk])
    store = PendingIngestStore(db)

    db.connection.execute("BEGIN IMMEDIATE")
    assert store.claim(first_manifest) == "pending"
    assert store.claim(second_manifest) == "pending"
    db.connection.commit()

    try:
        db.connection.execute("BEGIN IMMEDIATE")
        for label, chunk_id in enumerate(
            (first_chunk.chunk_id, second_chunk.chunk_id), start=1
        ):
            # The second ID is globally reserved by `second`, but this raw row
            # falsely assigns it to `first`.  A mere ID join must not let that
            # unrelated reservation prove first's topology complete.
            db.execute(
                "INSERT INTO chunks "
                "(chunk_id, turn_id, text, start_char, end_char, token_count, "
                "embedding, hnsw_label, term_count) "
                "VALUES (?, ?, ?, 0, ?, 1, ?, ?, 1)",
                (
                    chunk_id,
                    first.turn_id,
                    first.text,
                    len(first.text),
                    b"embedding",
                    label,
                ),
            )

        with pytest.raises(
            sqlite3.IntegrityError,
            match="allow only complete pending-to-indexed",
        ):
            db.execute(
                "UPDATE pending_ingests SET status = 'indexed', indexed_at = ? "
                "WHERE turn_id = ?",
                ("2026-09-01T00:00:00+00:00", first.turn_id),
            )
    finally:
        db.connection.rollback()
