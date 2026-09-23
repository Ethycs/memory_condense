"""Retry and deterministic-replay state for durable ingest work."""

from __future__ import annotations


# Keep mutable operational state beside, rather than inside, the append-only
# T1/T2 receipts.  This leaves their original identity and transition guards
# intact while making failed work observable and fairly schedulable.  A staged
# enrichment result is the validated operation set, not provider prose.
PENDING_WORK_STATE_SCHEMA_V15 = """
CREATE INDEX IF NOT EXISTS idx_memory_content_retirement
ON memory_items(content_hash, status, retired_at_turn);

-- Pre-v15 retirement rows have no source-order coordinate. Record the last
-- transcript ordinal that existed when v15 was installed. This bounds manual
-- retirement bindings; exact pending T2 receipts are quarantined separately
-- below because an old update may have erased identity history entirely.
INSERT OR IGNORE INTO meta(key, value)
SELECT 'v15_legacy_retirement_boundary', CAST(COALESCE(MAX(ordinal), 0) AS TEXT)
FROM turns;

-- Connections running pre-v15 code do not register this function. SQLite
-- resolves trigger functions on the connection executing the write, so an
-- already-open legacy helper is fenced immediately after another process
-- migrates the shared file. This is a deliberate rolling-upgrade stop point.
CREATE TRIGGER IF NOT EXISTS trg_v15_writer_fence_memory_insert
BEFORE INSERT ON memory_items
WHEN memory_condense_writer_schema_version() <> 15
BEGIN
    SELECT RAISE(ABORT, 'memory writer schema version mismatch');
END;

CREATE TRIGGER IF NOT EXISTS trg_v15_writer_fence_memory_update
BEFORE UPDATE ON memory_items
WHEN memory_condense_writer_schema_version() <> 15
BEGIN
    SELECT RAISE(ABORT, 'memory writer schema version mismatch');
END;

CREATE TRIGGER IF NOT EXISTS trg_v15_writer_fence_enrichment_finalize
BEFORE UPDATE ON pending_enrichments
WHEN memory_condense_writer_schema_version() <> 15
BEGIN
    SELECT RAISE(ABORT, 'enrichment writer schema version mismatch');
END;

CREATE TRIGGER IF NOT EXISTS trg_v15_writer_fence_enrichment_insert
BEFORE INSERT ON pending_enrichments
WHEN memory_condense_writer_schema_version() <> 15
BEGIN
    SELECT RAISE(ABORT, 'enrichment writer schema version mismatch');
END;

-- Stop stale ingestion workers at their first durable mutation. In
-- particular, the chunk fence fires before hnswlib.add_items(), preventing a
-- pre-v15 helper from mutating the native index and only later discovering
-- that its receipt/memory writer is obsolete.
CREATE TRIGGER IF NOT EXISTS trg_v15_writer_fence_turn_insert
BEFORE INSERT ON turns
WHEN memory_condense_writer_schema_version() <> 15
BEGIN
    SELECT RAISE(ABORT, 'turn writer schema version mismatch');
END;

CREATE TRIGGER IF NOT EXISTS trg_v15_writer_fence_turn_update
BEFORE UPDATE ON turns
WHEN memory_condense_writer_schema_version() <> 15
BEGIN
    SELECT RAISE(ABORT, 'turn writer schema version mismatch');
END;

CREATE TRIGGER IF NOT EXISTS trg_v15_writer_fence_turn_delete
BEFORE DELETE ON turns
WHEN memory_condense_writer_schema_version() <> 15
BEGIN
    SELECT RAISE(ABORT, 'turn writer schema version mismatch');
END;

CREATE TRIGGER IF NOT EXISTS trg_v15_writer_fence_ingest_insert
BEFORE INSERT ON pending_ingests
WHEN memory_condense_writer_schema_version() <> 15
BEGIN
    SELECT RAISE(ABORT, 'ingest writer schema version mismatch');
END;

CREATE TRIGGER IF NOT EXISTS trg_v15_writer_fence_ingest_update
BEFORE UPDATE ON pending_ingests
WHEN memory_condense_writer_schema_version() <> 15
BEGIN
    SELECT RAISE(ABORT, 'ingest writer schema version mismatch');
END;

CREATE TRIGGER IF NOT EXISTS trg_v15_writer_fence_reservation_insert
BEFORE INSERT ON ingest_chunk_reservations
WHEN memory_condense_writer_schema_version() <> 15
BEGIN
    SELECT RAISE(ABORT, 'ingest writer schema version mismatch');
END;

CREATE TRIGGER IF NOT EXISTS trg_v15_writer_fence_chunk_insert
BEFORE INSERT ON chunks
WHEN memory_condense_writer_schema_version() <> 15
BEGIN
    SELECT RAISE(ABORT, 'chunk writer schema version mismatch');
END;

CREATE TRIGGER IF NOT EXISTS trg_v15_writer_fence_chunk_update
BEFORE UPDATE ON chunks
WHEN memory_condense_writer_schema_version() <> 15
BEGIN
    SELECT RAISE(ABORT, 'chunk writer schema version mismatch');
END;

CREATE TRIGGER IF NOT EXISTS trg_v15_writer_fence_chunk_delete
BEFORE DELETE ON chunks
WHEN memory_condense_writer_schema_version() <> 15
BEGIN
    SELECT RAISE(ABORT, 'chunk writer schema version mismatch');
END;

CREATE TRIGGER IF NOT EXISTS trg_v15_writer_fence_chunk_term_insert
BEFORE INSERT ON chunk_terms
WHEN memory_condense_writer_schema_version() <> 15
BEGIN
    SELECT RAISE(ABORT, 'lexical writer schema version mismatch');
END;

CREATE TRIGGER IF NOT EXISTS trg_v15_writer_fence_chunk_term_update
BEFORE UPDATE ON chunk_terms
WHEN memory_condense_writer_schema_version() <> 15
BEGIN
    SELECT RAISE(ABORT, 'lexical writer schema version mismatch');
END;

CREATE TRIGGER IF NOT EXISTS trg_v15_writer_fence_chunk_term_delete
BEFORE DELETE ON chunk_terms
WHEN memory_condense_writer_schema_version() <> 15
BEGIN
    SELECT RAISE(ABORT, 'lexical writer schema version mismatch');
END;

-- An in-place content update removes the old identity from memory_items. Keep
-- a compact source-order tombstone so delayed automatic extraction cannot
-- recreate that old value. Legacy retirement order is unknowable, so v15 does
-- not fabricate rows here; pre-v15 retired_at_turn values remain NULL.
CREATE TABLE IF NOT EXISTS memory_identity_retirements (
    mem_id          TEXT NOT NULL REFERENCES memory_items(mem_id),
    content_hash    TEXT NOT NULL,
    retired_at_turn INTEGER NOT NULL CHECK(retired_at_turn >= 0),
    reason          TEXT NOT NULL
                    CHECK(reason IN ('updated', 'deleted', 'superseded',
                                     'deduplicated')),
    PRIMARY KEY (mem_id, content_hash, retired_at_turn)
);

CREATE INDEX IF NOT EXISTS idx_memory_identity_retirement_lookup
ON memory_identity_retirements(content_hash, retired_at_turn DESC);

CREATE TRIGGER IF NOT EXISTS trg_memory_identity_retirement_no_delete
BEFORE DELETE ON memory_identity_retirements
BEGIN
    SELECT RAISE(ABORT, 'memory identity retirement history is durable');
END;

CREATE TRIGGER IF NOT EXISTS trg_memory_identity_retirement_no_update
BEFORE UPDATE ON memory_identity_retirements
BEGIN
    SELECT RAISE(ABORT, 'memory identity retirement history is immutable');
END;

-- This is also the rolling-upgrade writer fence. A v14 process connected
-- before migration still sees v15's trigger and cannot move an active identity
-- with its old UPDATE statement, because only v15 writes the required ledger
-- entry in the same transaction first.
CREATE TRIGGER IF NOT EXISTS trg_memory_identity_change_requires_retirement
BEFORE UPDATE OF content_hash ON memory_items
WHEN OLD.status = 'active' AND NEW.status = 'active'
 AND NEW.content_hash IS NOT OLD.content_hash
 AND NOT EXISTS (
     SELECT 1 FROM memory_identity_retirements AS r
     WHERE r.mem_id = OLD.mem_id
       AND r.content_hash = OLD.content_hash
       AND r.reason = 'updated'
       AND r.retired_at_turn = (SELECT COALESCE(MAX(ordinal), 0) FROM turns)
 )
BEGIN
    SELECT RAISE(ABORT, 'memory identity change requires v15 retirement receipt');
END;

CREATE TRIGGER IF NOT EXISTS trg_retired_memory_identity_immutable
BEFORE UPDATE OF content_hash ON memory_items
WHEN OLD.status <> 'active' AND NEW.content_hash IS NOT OLD.content_hash
BEGIN
    SELECT RAISE(ABORT, 'retired memory identity is immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_memory_retirement_guard_insert
BEFORE INSERT ON memory_items
WHEN (NEW.status = 'active' AND NEW.retired_at_turn IS NOT NULL)
  OR (NEW.status <> 'active' AND NEW.retired_at_turn IS NULL)
BEGIN
    SELECT RAISE(ABORT, 'memory status and retirement turn must agree');
END;

CREATE TRIGGER IF NOT EXISTS trg_memory_retirement_guard_update
BEFORE UPDATE OF status, retired_at_turn ON memory_items
WHEN (NEW.status = 'active' AND NEW.retired_at_turn IS NOT NULL)
  OR (OLD.status = 'active' AND NEW.status <> 'active'
      AND NEW.retired_at_turn IS NULL)
BEGIN
    SELECT RAISE(ABORT, 'memory status and retirement turn must agree');
END;

CREATE TRIGGER IF NOT EXISTS trg_memory_terminal_status_immutable
BEFORE UPDATE OF status ON memory_items
WHEN OLD.status <> 'active' AND NEW.status IS NOT OLD.status
BEGIN
    SELECT RAISE(ABORT, 'retired memory status is immutable');
END;

-- A terminal item's source-order coordinate is historical evidence. The sole
-- permitted write is the explicit one-time binding of a pre-v15 NULL value;
-- after that, neither changing nor clearing the coordinate is legal.
CREATE TRIGGER IF NOT EXISTS trg_memory_terminal_retirement_turn_immutable
BEFORE UPDATE OF retired_at_turn ON memory_items
WHEN OLD.status <> 'active'
 AND NEW.status = OLD.status
 AND NEW.retired_at_turn IS NOT OLD.retired_at_turn
 AND NOT (
     OLD.retired_at_turn IS NULL
     AND NEW.status = OLD.status
     AND NEW.retired_at_turn IS NOT NULL
 )
BEGIN
    SELECT RAISE(ABORT, 'retired memory chronology is immutable');
END;

CREATE TABLE IF NOT EXISTS pending_ingest_attempts (
    turn_id TEXT PRIMARY KEY REFERENCES pending_ingests(turn_id),
    attempt_count INTEGER NOT NULL CHECK(attempt_count >= 1),
    last_attempt_at TEXT NOT NULL,
    next_attempt_at TEXT NOT NULL,
    last_error_kind TEXT NOT NULL CHECK(length(last_error_kind) > 0)
);

CREATE INDEX IF NOT EXISTS idx_pending_ingest_attempts_count
ON pending_ingest_attempts(next_attempt_at, attempt_count, turn_id);

CREATE TRIGGER IF NOT EXISTS trg_pending_ingest_attempts_no_delete
BEFORE DELETE ON pending_ingest_attempts
BEGIN
    SELECT RAISE(ABORT, 'ingest attempt history is durable');
END;

CREATE TRIGGER IF NOT EXISTS trg_pending_ingest_attempts_guard_update
BEFORE UPDATE ON pending_ingest_attempts
WHEN NOT (
    NEW.turn_id = OLD.turn_id
    AND NEW.attempt_count = OLD.attempt_count + 1
    AND NEW.last_attempt_at IS NOT NULL
    AND NEW.next_attempt_at IS NOT NULL
    AND length(NEW.last_error_kind) > 0
)
BEGIN
    SELECT RAISE(ABORT, 'ingest attempts may only advance');
END;

CREATE TABLE IF NOT EXISTS pending_enrichment_state (
    turn_id TEXT PRIMARY KEY REFERENCES pending_enrichments(turn_id),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count >= 0),
    last_attempt_at TEXT,
    next_attempt_at TEXT,
    last_error_kind TEXT,
    staged_ops_json TEXT
        CHECK(staged_ops_json IS NULL
              OR (json_valid(staged_ops_json)
                  AND json_type(staged_ops_json) = 'object')),
    staged_chunk_ids_json TEXT
        CHECK(staged_chunk_ids_json IS NULL
              OR (json_valid(staged_chunk_ids_json)
                  AND json_type(staged_chunk_ids_json) = 'array')),
    staged_result_sha256 TEXT
        CHECK(staged_result_sha256 IS NULL
              OR (length(staged_result_sha256) = 64
                  AND staged_result_sha256 NOT GLOB '*[^0-9a-f]*')),
    CHECK((staged_ops_json IS NULL) = (staged_chunk_ids_json IS NULL)),
    CHECK(staged_ops_json IS NULL OR staged_result_sha256 IS NOT NULL),
    CHECK((attempt_count = 0
           AND last_attempt_at IS NULL AND next_attempt_at IS NULL
           AND last_error_kind IS NULL)
          OR (attempt_count > 0
              AND last_attempt_at IS NOT NULL
              AND next_attempt_at IS NOT NULL
              AND length(last_error_kind) > 0))
);

CREATE INDEX IF NOT EXISTS idx_pending_enrichment_state_count
ON pending_enrichment_state(next_attempt_at, attempt_count, turn_id);

-- A legacy quarantine can be explicitly disposed without pretending T2 ran.
-- The base receipt still makes its one legal pending->enriched transition so
-- existing readers stay compatible; this immutable side receipt preserves the
-- materially different terminal outcome.
CREATE TABLE IF NOT EXISTS pending_enrichment_dispositions (
    turn_id TEXT PRIMARY KEY REFERENCES pending_enrichments(turn_id),
    disposition TEXT NOT NULL CHECK(disposition = 'discarded_legacy'),
    decided_at TEXT NOT NULL,
    reason TEXT NOT NULL CHECK(length(reason) > 0)
);

CREATE TRIGGER IF NOT EXISTS trg_pending_enrichment_dispositions_no_delete
BEFORE DELETE ON pending_enrichment_dispositions
BEGIN
    SELECT RAISE(ABORT, 'enrichment disposition receipts are durable');
END;

CREATE TRIGGER IF NOT EXISTS trg_pending_enrichment_dispositions_no_update
BEFORE UPDATE ON pending_enrichment_dispositions
BEGIN
    SELECT RAISE(ABORT, 'enrichment disposition receipts are immutable');
END;

-- T2 receipts that existed when v15 was installed carry pre-v15 replay
-- ambiguity. Any receipt later adopted from an older T1 source does too:
-- pre-v15 in-place updates may have erased identities without any tombstone.
CREATE TABLE IF NOT EXISTS pending_enrichment_legacy_quarantine (
    turn_id TEXT PRIMARY KEY REFERENCES pending_enrichments(turn_id),
    migration_boundary INTEGER NOT NULL CHECK(migration_boundary >= 0)
);

INSERT OR IGNORE INTO pending_enrichment_legacy_quarantine
(turn_id, migration_boundary)
SELECT e.turn_id, CAST(m.value AS INTEGER)
FROM pending_enrichments AS e
JOIN meta AS m ON m.key = 'v15_legacy_retirement_boundary'
WHERE e.status = 'pending';

-- Classify at the storage boundary, not only in one Python claim helper. This
-- also covers direct v15 writers and keeps every exact old-source receipt out
-- of automatic T2 replay.
CREATE TRIGGER IF NOT EXISTS trg_pending_enrichment_quarantine_old_source
AFTER INSERT ON pending_enrichments
WHEN EXISTS (
    SELECT 1
    FROM turns AS t
    JOIN meta AS m ON m.key = 'v15_legacy_retirement_boundary'
    WHERE t.turn_id = NEW.turn_id
      AND t.ordinal <= CAST(m.value AS INTEGER)
)
BEGIN
    INSERT OR IGNORE INTO pending_enrichment_legacy_quarantine
    (turn_id, migration_boundary)
    SELECT NEW.turn_id, CAST(value AS INTEGER)
    FROM meta
    WHERE key = 'v15_legacy_retirement_boundary';
END;

CREATE TRIGGER IF NOT EXISTS trg_pending_enrichment_legacy_quarantine_no_delete
BEFORE DELETE ON pending_enrichment_legacy_quarantine
BEGIN
    SELECT RAISE(ABORT, 'legacy enrichment quarantine is durable');
END;

CREATE TRIGGER IF NOT EXISTS trg_pending_enrichment_legacy_quarantine_no_update
BEFORE UPDATE ON pending_enrichment_legacy_quarantine
BEGIN
    SELECT RAISE(ABORT, 'legacy enrichment quarantine is immutable');
END;

-- Corrections are per-operation work, not failed T2 turns. Safe creates can
-- publish and the enrichment receipt can finish while each grounded reversal
-- waits independently for an operator-reviewed target. The original canonical
-- operation stays as the audit record after resolution or dismissal.
CREATE TABLE IF NOT EXISTS pending_corrections (
    correction_id TEXT PRIMARY KEY
        CHECK(length(correction_id) = 64
              AND correction_id NOT GLOB '*[^0-9a-f]*'),
    turn_id TEXT NOT NULL REFERENCES pending_enrichments(turn_id),
    operation_index INTEGER NOT NULL CHECK(operation_index >= 0),
    operation_json TEXT NOT NULL
        CHECK(json_valid(operation_json)
              AND json_type(operation_json) = 'object'
              AND json_extract(operation_json, '$.type') = 'Correction'),
    operation_sha256 TEXT NOT NULL
        CHECK(length(operation_sha256) = 64
              AND operation_sha256 NOT GLOB '*[^0-9a-f]*'),
    status TEXT NOT NULL CHECK(status IN ('pending', 'resolved', 'dismissed')),
    created_at TEXT NOT NULL,
    decided_at TEXT,
    target_mem_id TEXT REFERENCES memory_items(mem_id),
    successor_mem_id TEXT REFERENCES memory_items(mem_id),
    reason TEXT,
    UNIQUE(turn_id, operation_index),
    CHECK(
        (status = 'pending' AND decided_at IS NULL
         AND target_mem_id IS NULL AND successor_mem_id IS NULL
         AND reason IS NULL)
        OR
        (status = 'resolved' AND decided_at IS NOT NULL
         AND target_mem_id IS NOT NULL AND successor_mem_id IS NOT NULL
         AND length(reason) > 0)
        OR
        (status = 'dismissed' AND decided_at IS NOT NULL
         AND target_mem_id IS NULL AND successor_mem_id IS NULL
         AND length(reason) > 0)
    )
);

CREATE INDEX IF NOT EXISTS idx_pending_corrections_status
ON pending_corrections(status, created_at, correction_id);

CREATE TRIGGER IF NOT EXISTS trg_pending_corrections_no_delete
BEFORE DELETE ON pending_corrections
BEGIN
    SELECT RAISE(ABORT, 'correction receipts are durable');
END;

CREATE TRIGGER IF NOT EXISTS trg_pending_corrections_guard_update
BEFORE UPDATE ON pending_corrections
WHEN NOT (
    OLD.status = 'pending'
    AND NEW.status IN ('resolved', 'dismissed')
    AND NEW.correction_id = OLD.correction_id
    AND NEW.turn_id = OLD.turn_id
    AND NEW.operation_index = OLD.operation_index
    AND NEW.operation_json = OLD.operation_json
    AND NEW.operation_sha256 = OLD.operation_sha256
    AND NEW.created_at = OLD.created_at
    AND NEW.decided_at IS NOT NULL
    AND length(NEW.reason) > 0
    AND (
        (NEW.status = 'resolved' AND NEW.target_mem_id IS NOT NULL
         AND NEW.successor_mem_id IS NOT NULL)
        OR
        (NEW.status = 'dismissed' AND NEW.target_mem_id IS NULL
         AND NEW.successor_mem_id IS NULL)
    )
)
BEGIN
    SELECT RAISE(ABORT, 'corrections allow only one terminal disposition');
END;

CREATE TRIGGER IF NOT EXISTS trg_pending_enrichment_state_no_delete
BEFORE DELETE ON pending_enrichment_state
BEGIN
    SELECT RAISE(ABORT, 'enrichment replay state is durable');
END;

CREATE TRIGGER IF NOT EXISTS trg_pending_enrichment_state_guard_update
BEFORE UPDATE ON pending_enrichment_state
WHEN NOT (
    NEW.turn_id = OLD.turn_id
    AND (
        (
            NEW.attempt_count = 0
            AND NEW.last_attempt_at IS NULL
            AND NEW.next_attempt_at IS NULL
            AND NEW.last_error_kind IS NULL
            AND OLD.staged_ops_json IS NULL
            AND NEW.staged_ops_json IS NOT NULL
            AND OLD.staged_chunk_ids_json IS NULL
            AND NEW.staged_chunk_ids_json IS NOT NULL
            AND OLD.staged_result_sha256 IS NULL
            AND NEW.staged_result_sha256 IS NOT NULL
        )
        OR
        (
            NEW.attempt_count = OLD.attempt_count + 1
            AND NEW.last_attempt_at IS NOT NULL
            AND NEW.next_attempt_at IS NOT NULL
            AND length(NEW.last_error_kind) > 0
            AND NEW.staged_ops_json IS OLD.staged_ops_json
            AND NEW.staged_chunk_ids_json IS OLD.staged_chunk_ids_json
            AND NEW.staged_result_sha256 IS OLD.staged_result_sha256
        )
        OR
        (
            NEW.attempt_count = OLD.attempt_count
            AND NEW.last_attempt_at IS OLD.last_attempt_at
            AND NEW.next_attempt_at IS OLD.next_attempt_at
            AND NEW.last_error_kind IS OLD.last_error_kind
            AND OLD.staged_ops_json IS NOT NULL
            AND NEW.staged_ops_json IS NULL
            AND OLD.staged_chunk_ids_json IS NOT NULL
            AND NEW.staged_chunk_ids_json IS NULL
            AND NEW.staged_result_sha256 = OLD.staged_result_sha256
            AND EXISTS (
                SELECT 1 FROM pending_enrichments AS e
                WHERE e.turn_id = OLD.turn_id AND e.status = 'enriched'
            )
        )
    )
)
BEGIN
    SELECT RAISE(ABORT, 'enrichment replay state may only stage once or advance');
END;

CREATE TABLE IF NOT EXISTS pending_work_schedule (
    stage TEXT PRIMARY KEY CHECK(stage IN ('ingest', 'enrichment')),
    prefer_retry INTEGER NOT NULL CHECK(prefer_retry IN (0, 1))
);

INSERT OR IGNORE INTO pending_work_schedule(stage, prefer_retry)
VALUES ('ingest', 1), ('enrichment', 1);

CREATE TRIGGER IF NOT EXISTS trg_pending_work_schedule_no_delete
BEFORE DELETE ON pending_work_schedule
BEGIN
    SELECT RAISE(ABORT, 'pending work schedule is durable');
END;

CREATE TRIGGER IF NOT EXISTS trg_pending_work_schedule_guard_update
BEFORE UPDATE ON pending_work_schedule
WHEN NOT (
    NEW.stage = OLD.stage
    AND NEW.prefer_retry = 1 - OLD.prefer_retry
)
BEGIN
    SELECT RAISE(ABORT, 'pending work schedule may only alternate');
END;
"""


__all__ = ["PENDING_WORK_STATE_SCHEMA_V15"]
