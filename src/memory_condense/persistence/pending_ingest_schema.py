"""Schema fragment for the durable turn-to-index publication journal."""

from __future__ import annotations


# Exact duplicate writers share a manifest and may finish one another's work;
# there is deliberately no process lease or owner token to strand after process
# death. Indexed receipts remain durable so a later writer cannot silently
# publish a second, incompatible chunk topology for the same turn. Normalized
# reservations make every chunk ID globally exclusive before its chunk row is
# materialized, without an O(turns) JSON scan on each index insertion.
PENDING_INGEST_SCHEMA_V13 = """
CREATE TABLE IF NOT EXISTS pending_ingests (
    turn_id          TEXT PRIMARY KEY REFERENCES turns(turn_id),
    manifest_sha256  TEXT NOT NULL
                     CHECK(length(manifest_sha256) = 64
                           AND manifest_sha256 NOT GLOB '*[^0-9a-f]*'),
    manifest_json    TEXT NOT NULL CHECK(json_valid(manifest_json)),
    status            TEXT NOT NULL
                      CHECK(status IN ('pending', 'indexed')),
    created_at        TEXT NOT NULL,
    indexed_at        TEXT,
    CHECK((status = 'pending' AND indexed_at IS NULL)
          OR (status = 'indexed' AND indexed_at IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS idx_pending_ingests_status
ON pending_ingests(status, created_at, turn_id);

CREATE TABLE IF NOT EXISTS ingest_chunk_reservations (
    chunk_id      TEXT PRIMARY KEY,
    turn_id       TEXT NOT NULL REFERENCES pending_ingests(turn_id)
                  ON DELETE CASCADE,
    start_char    INTEGER NOT NULL CHECK(start_char >= 0),
    end_char      INTEGER NOT NULL CHECK(end_char > start_char),
    token_count   INTEGER NOT NULL CHECK(token_count >= 0),
    text_sha256   TEXT NOT NULL
                  CHECK(length(text_sha256) = 64
                        AND text_sha256 NOT GLOB '*[^0-9a-f]*')
);

CREATE INDEX IF NOT EXISTS idx_ingest_chunk_reservations_turn
ON ingest_chunk_reservations(turn_id, start_char, end_char, chunk_id);

CREATE TRIGGER IF NOT EXISTS trg_pending_ingests_no_delete
BEFORE DELETE ON pending_ingests
BEGIN
    SELECT RAISE(ABORT, 'ingest receipts are durable');
END;

CREATE TRIGGER IF NOT EXISTS trg_pending_ingests_guard_update
BEFORE UPDATE ON pending_ingests
WHEN NOT (
    OLD.status = 'pending'
    AND NEW.status = 'indexed'
    AND NEW.turn_id = OLD.turn_id
    AND NEW.manifest_sha256 = OLD.manifest_sha256
    AND NEW.manifest_json = OLD.manifest_json
    AND NEW.created_at = OLD.created_at
    AND OLD.indexed_at IS NULL
    AND NEW.indexed_at IS NOT NULL
    AND NOT EXISTS (
        SELECT 1 FROM ingest_chunk_reservations AS r
        LEFT JOIN chunks AS c ON c.chunk_id = r.chunk_id
        WHERE r.turn_id = OLD.turn_id
          AND (c.chunk_id IS NULL OR c.turn_id <> OLD.turn_id
               OR c.embedding IS NULL OR c.hnsw_label IS NULL
               OR c.term_count IS NULL)
    )
    AND NOT EXISTS (
        SELECT 1 FROM chunks AS c
        LEFT JOIN ingest_chunk_reservations AS r
               ON r.chunk_id = c.chunk_id
        WHERE c.turn_id = OLD.turn_id
          AND (r.chunk_id IS NULL OR r.turn_id <> OLD.turn_id)
    )
)
BEGIN
    SELECT RAISE(ABORT, 'ingest receipts allow only complete pending-to-indexed');
END;

CREATE TRIGGER IF NOT EXISTS trg_ingest_chunk_reservations_guard_insert
BEFORE INSERT ON ingest_chunk_reservations
WHEN NOT EXISTS (
    SELECT 1
    FROM pending_ingests AS p,
         json_each(p.manifest_json, '$.chunks') AS member
    WHERE p.turn_id = NEW.turn_id
      AND json_extract(member.value, '$.chunk_id') = NEW.chunk_id
      AND json_extract(member.value, '$.start_char') = NEW.start_char
      AND json_extract(member.value, '$.end_char') = NEW.end_char
      AND json_extract(member.value, '$.token_count') = NEW.token_count
      AND json_extract(member.value, '$.text_sha256') = NEW.text_sha256
)
BEGIN
    SELECT RAISE(ABORT, 'ingest reservation is not declared by its manifest');
END;

CREATE TRIGGER IF NOT EXISTS trg_ingest_chunk_reservations_no_update
BEFORE UPDATE ON ingest_chunk_reservations
BEGIN
    SELECT RAISE(ABORT, 'ingest chunk reservations are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_ingest_chunk_reservations_no_delete
BEFORE DELETE ON ingest_chunk_reservations
BEGIN
    SELECT RAISE(ABORT, 'ingest chunk reservations are durable');
END;
"""


__all__ = ["PENDING_INGEST_SCHEMA_V13"]
