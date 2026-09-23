from __future__ import annotations

import re

import pytest

from memory_condense.domain.schemas import Chunk, Turn
from memory_condense.persistence.db import Database
from memory_condense.persistence.pending_enrichment_store import (
    PendingEnrichmentStore,
)
from memory_condense.persistence.pending_ingest_store import (
    PendingIngestManifest,
    PendingIngestStore,
)
from memory_condense.persistence.transcript_store import TranscriptStore


def _manifest(turn: Turn, *, chunk_id: str | None = None) -> PendingIngestManifest:
    chunk = Chunk(
        chunk_id=chunk_id or f"chunk-{turn.turn_id}",
        turn_id=turn.turn_id,
        text=turn.text,
        start_char=0,
        end_char=len(turn.text),
        token_count=1,
    )
    return PendingIngestManifest.build(turn, (chunk,))


def _publish_turn_batch(
    db: Database,
    *,
    prefix: str,
    count: int,
) -> list[Turn]:
    transcript = TranscriptStore(db)
    turns = [
        Turn(
            role="user",
            text=f"value-{index}",
            turn_id=f"{prefix}-{index:04d}",
        )
        for index in range(count)
    ]
    db.connection.execute("BEGIN IMMEDIATE")
    try:
        for turn in turns:
            transcript.publish_turn(turn, commit=False)
        db.connection.commit()
    except BaseException:
        db.connection.rollback()
        raise
    return turns


def test_claim_many_uses_bounded_set_queries_across_parameter_boundary(db) -> None:
    turn_count = 501
    turns = _publish_turn_batch(db, prefix="turn", count=turn_count)
    manifests = [_manifest(turn) for turn in turns]
    statements: list[str] = []

    db.connection.execute("BEGIN IMMEDIATE")
    db.connection.set_trace_callback(statements.append)
    try:
        statuses = PendingIngestStore(db).claim_many(manifests)
    finally:
        db.connection.set_trace_callback(None)
        db.connection.commit()

    assert statuses == {turn.turn_id: "pending" for turn in turns}
    selects = [
        statement
        for statement in statements
        if re.match(r"^\s*SELECT\b", statement, re.IGNORECASE)
    ]
    # Query count may change when a validation read is added, but it must scale
    # with bounded parameter batches rather than with individual turns. 501
    # crosses the conservative 500-parameter IN boundary.
    parameter_batches = (turn_count + 499) // 500
    assert 1 <= len(selects) <= 6 * parameter_batches
    assert all(" IN (" in statement.upper() for statement in selects)
    assert db.execute(
        "SELECT COUNT(*) FROM ingest_chunk_reservations"
    ).fetchone()[0] == turn_count


def test_finalize_many_uses_set_reads_and_updates(db) -> None:
    turns = _publish_turn_batch(db, prefix="finalize", count=100)
    manifests = [_manifest(turn) for turn in turns]
    store = PendingIngestStore(db)
    db.connection.execute("BEGIN IMMEDIATE")
    store.claim_many(manifests)
    db.connection.executemany(
        "INSERT INTO chunks "
        "(chunk_id, turn_id, text, start_char, end_char, token_count, "
        "embedding, hnsw_label, term_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                manifest.chunks[0].chunk_id,
                turn.turn_id,
                turn.text,
                0,
                len(turn.text),
                1,
                b"embedding",
                index,
                0,
            )
            for index, (turn, manifest) in enumerate(zip(turns, manifests))
        ],
    )
    statements: list[str] = []
    db.connection.set_trace_callback(statements.append)
    try:
        store.finalize(manifests)
    finally:
        db.connection.set_trace_callback(None)
        db.connection.commit()

    selects = {
        " ".join(statement.upper().split())
        for statement in statements
        if re.match(r"^\s*SELECT\b", statement, re.IGNORECASE)
    }
    updates = {
        " ".join(statement.upper().split())
        for statement in statements
        if re.match(r"^\s*UPDATE PENDING_INGESTS\b", statement, re.IGNORECASE)
    }
    assert len(selects) <= 3
    assert len(updates) == 1
    assert " IN (VALUES " in next(iter(updates))
    assert db.execute(
        "SELECT COUNT(*) FROM pending_ingests WHERE status = 'indexed'"
    ).fetchone() == (len(turns),)

    membership_statements: list[str] = []
    db.connection.set_trace_callback(membership_statements.append)
    try:
        store.validate_chunk_membership(
            manifests[0].reconstruct(turns[0]),
            allow_indexed_rebuild=True,
        )
    finally:
        db.connection.set_trace_callback(None)
    membership_chunk_reads = [
        " ".join(statement.upper().split())
        for statement in membership_statements
        if re.match(r"^\s*SELECT\b", statement, re.IGNORECASE)
        and "FROM CHUNKS" in statement.upper()
    ]
    assert membership_chunk_reads
    assert all(
        "EMBEDDING IS NOT NULL" in query for query in membership_chunk_reads
    )
    assert all(
        "CHUNK_ID, EMBEDDING, HNSW_LABEL" not in query
        for query in membership_chunk_reads
    )


def test_claim_many_rejects_cross_manifest_chunk_identity_and_rolls_back(db) -> None:
    transcript = TranscriptStore(db)
    first = transcript.append("user", "alpha", turn_id="first")
    second = transcript.append("user", "beta", turn_id="second")
    manifests = [
        _manifest(first, chunk_id="shared-chunk"),
        _manifest(second, chunk_id="shared-chunk"),
    ]

    db.connection.execute("BEGIN IMMEDIATE")
    with pytest.raises(ValueError, match="reuse one chunk identity"):
        PendingIngestStore(db).claim_many(manifests)
    db.connection.rollback()

    assert db.execute("SELECT COUNT(*) FROM pending_ingests").fetchone()[0] == 0
    assert db.execute(
        "SELECT COUNT(*) FROM ingest_chunk_reservations"
    ).fetchone()[0] == 0


def test_claim_many_is_exactly_idempotent_and_rejects_changed_manifest(db) -> None:
    turn = TranscriptStore(db).append(
        "user", "alpha beta", turn_id="stable-turn"
    )
    original = _manifest(turn)
    changed_chunk = Chunk(
        chunk_id="changed-chunk",
        turn_id=turn.turn_id,
        text="alpha",
        start_char=0,
        end_char=5,
        token_count=1,
    )
    changed = PendingIngestManifest.build(turn, (changed_chunk,))
    store = PendingIngestStore(db)

    db.connection.execute("BEGIN IMMEDIATE")
    assert store.claim_many((original, original)) == {turn.turn_id: "pending"}
    db.connection.commit()

    db.connection.execute("BEGIN IMMEDIATE")
    assert store.claim(original) == "pending"
    with pytest.raises(ValueError, match="different pending chunk manifest"):
        store.claim_many((changed,))
    db.connection.rollback()


def test_claim_many_preserves_mixed_statuses_and_is_exactly_idempotent(db) -> None:
    transcript = TranscriptStore(db)
    existing_pending = transcript.append(
        "user", "existing pending", turn_id="existing-pending"
    )
    new_pending = transcript.append("user", "new pending", turn_id="new-pending")
    already_indexed = transcript.append(
        "user", "already indexed", turn_id="already-indexed"
    )
    empty = transcript.append("user", "", turn_id="empty")
    pending_manifest = _manifest(existing_pending)
    new_manifest = _manifest(new_pending)
    indexed_manifest = _manifest(already_indexed)
    empty_manifest = PendingIngestManifest.build(empty, ())
    store = PendingIngestStore(db)

    db.connection.execute("BEGIN IMMEDIATE")
    store.claim(pending_manifest)
    store.claim(indexed_manifest)
    indexed_chunk = indexed_manifest.reconstruct(already_indexed)[0]
    db.execute(
        "INSERT INTO chunks "
        "(chunk_id, turn_id, text, start_char, end_char, token_count, "
        "embedding, hnsw_label, term_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            indexed_chunk.chunk_id,
            indexed_chunk.turn_id,
            indexed_chunk.text,
            indexed_chunk.start_char,
            indexed_chunk.end_char,
            indexed_chunk.token_count,
            b"embedding",
            0,
            0,
        ),
    )
    store.finalize((indexed_manifest,))
    db.connection.commit()

    manifests = (
        pending_manifest,
        new_manifest,
        indexed_manifest,
        empty_manifest,
    )
    expected = {
        existing_pending.turn_id: "pending",
        new_pending.turn_id: "pending",
        already_indexed.turn_id: "indexed",
        empty.turn_id: "indexed",
    }

    db.connection.execute("BEGIN IMMEDIATE")
    assert store.claim_many(manifests) == expected
    db.connection.commit()
    before = db.execute(
        "SELECT turn_id, manifest_sha256, manifest_json, status, created_at, "
        "indexed_at FROM pending_ingests ORDER BY turn_id"
    ).fetchall()

    db.connection.execute("BEGIN IMMEDIATE")
    assert store.claim_many(tuple(reversed(manifests))) == expected
    db.connection.commit()
    after = db.execute(
        "SELECT turn_id, manifest_sha256, manifest_json, status, created_at, "
        "indexed_at FROM pending_ingests ORDER BY turn_id"
    ).fetchall()

    assert after == before
    assert db.execute("SELECT COUNT(*) FROM pending_ingests").fetchone()[0] == 4
    assert db.execute(
        "SELECT COUNT(*) FROM ingest_chunk_reservations"
    ).fetchone()[0] == 3


def test_enrichment_claim_many_is_batched_exact_and_transaction_owned(db) -> None:
    transcript = TranscriptStore(db)
    turns = [
        transcript.append("user", f"value-{index}", turn_id=f"enrich-{index:03d}")
        for index in range(100)
    ]
    manifests = [_manifest(turn) for turn in turns]
    ingest = PendingIngestStore(db)
    enrichment = PendingEnrichmentStore(db)

    with pytest.raises(RuntimeError, match="active caller transaction"):
        enrichment.claim_many(manifests)

    statements: list[str] = []
    db.connection.execute("BEGIN IMMEDIATE")
    ingest.claim_many(manifests)
    db.connection.set_trace_callback(statements.append)
    try:
        statuses = enrichment.claim_many(manifests)
    finally:
        db.connection.set_trace_callback(None)
        db.connection.commit()

    assert statuses == {turn.turn_id: "pending" for turn in turns}
    selects = [
        statement
        for statement in statements
        if re.match(r"^\s*SELECT\b", statement, re.IGNORECASE)
    ]
    assert 1 <= len(selects) <= 2
    assert " IN (" in selects[0].upper()

    db.connection.execute("BEGIN IMMEDIATE")
    assert enrichment.claim(manifests[0]) == "pending"
    db.connection.rollback()


def test_ingest_claim_many_requires_caller_transaction(db) -> None:
    turn = TranscriptStore(db).append("user", "alpha", turn_id="turn")

    with pytest.raises(RuntimeError, match="active caller transaction"):
        PendingIngestStore(db).claim_many((_manifest(turn),))
