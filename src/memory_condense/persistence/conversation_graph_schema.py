"""Schema fragment for the append-only conversational association graph."""

from __future__ import annotations


# T0 owns only ``pending_graph_compilations``.  The evidence-derived rows are
# written later, after the corresponding pending-ingest receipt is indexed.
# Every graph append advances a SHA-256 checkpoint chain, so publishing one
# turn hashes only that turn's deltas instead of replaying the corpus.
CONVERSATION_GRAPH_SCHEMA_V16 = """
CREATE TABLE IF NOT EXISTS graph_artifacts (
    artifact_id TEXT PRIMARY KEY
        CHECK(length(artifact_id) = 64
              AND artifact_id NOT GLOB '*[^0-9a-f]*'),
    format TEXT NOT NULL
        CHECK(format = 'memory-condense-conversation-graph-v1'),
    extraction_policy_sha256 TEXT NOT NULL
        CHECK(length(extraction_policy_sha256) = 64
              AND extraction_policy_sha256 NOT GLOB '*[^0-9a-f]*'),
    story_index_policy_sha256 TEXT NOT NULL
        CHECK(length(story_index_policy_sha256) = 64
              AND story_index_policy_sha256 NOT GLOB '*[^0-9a-f]*'),
    persistence_policy_sha256 TEXT NOT NULL
        CHECK(length(persistence_policy_sha256) = 64
              AND persistence_policy_sha256 NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL,
    UNIQUE(format, extraction_policy_sha256, story_index_policy_sha256,
           persistence_policy_sha256)
);

CREATE TABLE IF NOT EXISTS pending_graph_compilations (
    artifact_id TEXT NOT NULL REFERENCES graph_artifacts(artifact_id),
    turn_id TEXT NOT NULL REFERENCES pending_ingests(turn_id),
    ingest_manifest_sha256 TEXT NOT NULL
        CHECK(length(ingest_manifest_sha256) = 64
              AND ingest_manifest_sha256 NOT GLOB '*[^0-9a-f]*'),
    status TEXT NOT NULL CHECK(status IN ('pending', 'ready', 'no_output')),
    created_at TEXT NOT NULL,
    completed_at TEXT,
    first_revision INTEGER CHECK(first_revision IS NULL OR first_revision >= 1),
    last_revision INTEGER CHECK(last_revision IS NULL OR last_revision >= 1),
    chunk_count INTEGER CHECK(chunk_count IS NULL OR chunk_count >= 0),
    occurrence_count INTEGER
        CHECK(occurrence_count IS NULL OR occurrence_count >= 0),
    checkpoint_sha256 TEXT
        CHECK(checkpoint_sha256 IS NULL
              OR (length(checkpoint_sha256) = 64
                  AND checkpoint_sha256 NOT GLOB '*[^0-9a-f]*')),
    receipt_sha256 TEXT
        CHECK(receipt_sha256 IS NULL
              OR (length(receipt_sha256) = 64
                  AND receipt_sha256 NOT GLOB '*[^0-9a-f]*')),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count >= 0),
    last_attempt_at TEXT,
    last_error_kind TEXT,
    PRIMARY KEY (artifact_id, turn_id),
    CHECK(
        (status = 'pending'
         AND completed_at IS NULL
         AND first_revision IS NULL AND last_revision IS NULL
         AND chunk_count IS NULL AND occurrence_count IS NULL
         AND checkpoint_sha256 IS NULL AND receipt_sha256 IS NULL)
        OR
        (status = 'ready'
         AND completed_at IS NOT NULL
         AND first_revision IS NOT NULL AND last_revision IS NOT NULL
         AND last_revision >= first_revision
         AND chunk_count > 0 AND occurrence_count >= 0
         AND checkpoint_sha256 IS NOT NULL AND receipt_sha256 IS NOT NULL)
        OR
        (status = 'no_output'
         AND completed_at IS NOT NULL
         AND first_revision IS NULL AND last_revision IS NULL
         AND chunk_count = 0 AND occurrence_count = 0
         AND checkpoint_sha256 IS NOT NULL AND receipt_sha256 IS NOT NULL)
    ),
    CHECK((attempt_count = 0
           AND last_attempt_at IS NULL AND last_error_kind IS NULL)
          OR (attempt_count > 0
              AND last_attempt_at IS NOT NULL
              AND length(last_error_kind) > 0))
);

CREATE INDEX IF NOT EXISTS idx_pending_graph_compilations_status
ON pending_graph_compilations(
    artifact_id, status, attempt_count, created_at, turn_id
);

CREATE TABLE IF NOT EXISTS conversation_graph_chunks (
    artifact_id TEXT NOT NULL REFERENCES graph_artifacts(artifact_id),
    -- Derived graph coordinates deliberately do not own the reconstructible
    -- chunk row. Existing index-repair paths may remove a corrupt chunk and
    -- then rebuild it from the sealed ingest manifest; resident graph sync
    -- fails closed while that authority is absent.
    chunk_id TEXT NOT NULL,
    turn_id TEXT NOT NULL REFERENCES turns(turn_id),
    source_id TEXT NOT NULL CHECK(length(source_id) > 0),
    turn_ordinal INTEGER NOT NULL CHECK(turn_ordinal >= 0),
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    start_char INTEGER NOT NULL CHECK(start_char >= 0),
    end_char INTEGER NOT NULL CHECK(end_char > start_char),
    created_at TEXT NOT NULL,
    text_sha256 TEXT NOT NULL
        CHECK(length(text_sha256) = 64
              AND text_sha256 NOT GLOB '*[^0-9a-f]*'),
    chunk_identity_sha256 TEXT NOT NULL
        CHECK(length(chunk_identity_sha256) = 64
              AND chunk_identity_sha256 NOT GLOB '*[^0-9a-f]*'),
    append_revision INTEGER NOT NULL CHECK(append_revision >= 1),
    predecessor_chunk_id TEXT,
    successor_chunk_id TEXT,
    occurrence_count INTEGER NOT NULL CHECK(occurrence_count >= 0),
    phrase_key_count INTEGER NOT NULL CHECK(phrase_key_count >= 0),
    new_story_term_membership_count INTEGER NOT NULL
        CHECK(new_story_term_membership_count >= 0),
    story_evidence_chunk_retained INTEGER NOT NULL
        CHECK(story_evidence_chunk_retained IN (0, 1)),
    runtime_receipt_sha256 TEXT NOT NULL
        CHECK(length(runtime_receipt_sha256) = 64
              AND runtime_receipt_sha256 NOT GLOB '*[^0-9a-f]*'),
    delta_sha256 TEXT NOT NULL
        CHECK(length(delta_sha256) = 64
              AND delta_sha256 NOT GLOB '*[^0-9a-f]*'),
    parent_checkpoint_sha256 TEXT NOT NULL
        CHECK(length(parent_checkpoint_sha256) = 64
              AND parent_checkpoint_sha256 NOT GLOB '*[^0-9a-f]*'),
    checkpoint_sha256 TEXT NOT NULL
        CHECK(length(checkpoint_sha256) = 64
              AND checkpoint_sha256 NOT GLOB '*[^0-9a-f]*'),
    PRIMARY KEY (artifact_id, chunk_id),
    UNIQUE (artifact_id, append_revision),
    UNIQUE (artifact_id, source_id, turn_ordinal, start_char),
    FOREIGN KEY (artifact_id, turn_id)
        REFERENCES pending_graph_compilations(artifact_id, turn_id),
    CHECK(predecessor_chunk_id IS NULL OR predecessor_chunk_id <> chunk_id),
    CHECK(successor_chunk_id IS NULL OR successor_chunk_id <> chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_conversation_graph_chunks_source_order
ON conversation_graph_chunks(
    artifact_id, source_id, turn_ordinal, start_char, chunk_id
);

CREATE INDEX IF NOT EXISTS idx_conversation_graph_chunks_turn
ON conversation_graph_chunks(artifact_id, turn_id, append_revision);

CREATE INDEX IF NOT EXISTS idx_conversation_graph_chunks_story_evidence
ON conversation_graph_chunks(
    artifact_id, source_id, story_evidence_chunk_retained, append_revision
);

CREATE TABLE IF NOT EXISTS conversation_phrase_occurrences (
    artifact_id TEXT NOT NULL,
    occurrence_id TEXT NOT NULL
        CHECK(length(occurrence_id) = 64
              AND occurrence_id NOT GLOB '*[^0-9a-f]*'),
    chunk_id TEXT NOT NULL,
    occurrence_ordinal INTEGER NOT NULL CHECK(occurrence_ordinal >= 0),
    canonical_key TEXT NOT NULL CHECK(length(canonical_key) > 0),
    start_char INTEGER NOT NULL CHECK(start_char >= 0),
    end_char INTEGER NOT NULL CHECK(end_char > start_char),
    quote_sha256 TEXT NOT NULL
        CHECK(length(quote_sha256) = 64
              AND quote_sha256 NOT GLOB '*[^0-9a-f]*'),
    token_count INTEGER NOT NULL CHECK(token_count >= 1),
    PRIMARY KEY (artifact_id, occurrence_id),
    UNIQUE (artifact_id, chunk_id, occurrence_ordinal),
    FOREIGN KEY (artifact_id, chunk_id)
        REFERENCES conversation_graph_chunks(artifact_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_conversation_phrase_occurrences_key
ON conversation_phrase_occurrences(
    artifact_id, canonical_key, chunk_id, occurrence_ordinal
);

CREATE TABLE IF NOT EXISTS conversation_story_term_memberships (
    artifact_id TEXT NOT NULL,
    source_id TEXT NOT NULL CHECK(length(source_id) > 0),
    story_term TEXT NOT NULL CHECK(length(story_term) > 0),
    occurrence_id TEXT NOT NULL,
    append_revision INTEGER NOT NULL CHECK(append_revision >= 1),
    membership_ordinal INTEGER NOT NULL CHECK(membership_ordinal >= 0),
    PRIMARY KEY (artifact_id, source_id, story_term),
    UNIQUE (artifact_id, append_revision, membership_ordinal),
    FOREIGN KEY (artifact_id, occurrence_id)
        REFERENCES conversation_phrase_occurrences(artifact_id, occurrence_id)
);

CREATE INDEX IF NOT EXISTS idx_conversation_story_term_memberships_term
ON conversation_story_term_memberships(
    artifact_id, story_term, append_revision, membership_ordinal, source_id
);

CREATE TABLE IF NOT EXISTS conversation_graph_state (
    artifact_id TEXT PRIMARY KEY REFERENCES graph_artifacts(artifact_id),
    revision INTEGER NOT NULL CHECK(revision >= 0),
    chunk_count INTEGER NOT NULL CHECK(chunk_count >= 0),
    occurrence_count INTEGER NOT NULL CHECK(occurrence_count >= 0),
    story_term_membership_count INTEGER NOT NULL
        CHECK(story_term_membership_count >= 0),
    story_evidence_chunk_count INTEGER NOT NULL
        CHECK(story_evidence_chunk_count >= 0),
    ready_turn_count INTEGER NOT NULL CHECK(ready_turn_count >= 0),
    checkpoint_sha256 TEXT NOT NULL
        CHECK(length(checkpoint_sha256) = 64
              AND checkpoint_sha256 NOT GLOB '*[^0-9a-f]*'),
    state_sha256 TEXT NOT NULL
        CHECK(length(state_sha256) = 64
              AND state_sha256 NOT GLOB '*[^0-9a-f]*'),
    updated_at TEXT NOT NULL,
    CHECK(revision = chunk_count)
);

CREATE TRIGGER IF NOT EXISTS trg_graph_artifacts_no_update
BEFORE UPDATE ON graph_artifacts
BEGIN
    SELECT RAISE(ABORT, 'graph artifact identities are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_graph_artifacts_no_delete
BEFORE DELETE ON graph_artifacts
BEGIN
    SELECT RAISE(ABORT, 'graph artifact identities are durable');
END;

CREATE TRIGGER IF NOT EXISTS trg_pending_graph_compilations_no_delete
BEFORE DELETE ON pending_graph_compilations
BEGIN
    SELECT RAISE(ABORT, 'graph compilation receipts are durable');
END;

CREATE TRIGGER IF NOT EXISTS trg_pending_graph_compilations_guard_update
BEFORE UPDATE ON pending_graph_compilations
WHEN NOT (
    NEW.artifact_id = OLD.artifact_id
    AND NEW.turn_id = OLD.turn_id
    AND NEW.ingest_manifest_sha256 = OLD.ingest_manifest_sha256
    AND NEW.created_at = OLD.created_at
    AND (
        (OLD.status = 'pending' AND NEW.status = 'pending'
         AND NEW.completed_at IS NULL
         AND NEW.first_revision IS NULL AND NEW.last_revision IS NULL
         AND NEW.chunk_count IS NULL AND NEW.occurrence_count IS NULL
         AND NEW.checkpoint_sha256 IS NULL AND NEW.receipt_sha256 IS NULL
         AND NEW.attempt_count = OLD.attempt_count + 1
         AND NEW.last_attempt_at IS NOT NULL
         AND length(NEW.last_error_kind) > 0)
        OR
        (OLD.status = 'pending' AND NEW.status IN ('ready', 'no_output')
         AND NEW.completed_at IS NOT NULL
         AND NEW.receipt_sha256 IS NOT NULL
         AND NEW.checkpoint_sha256 IS NOT NULL
         AND NEW.attempt_count = OLD.attempt_count
         AND NEW.last_attempt_at IS OLD.last_attempt_at
         AND NEW.last_error_kind IS OLD.last_error_kind
         AND EXISTS (
             SELECT 1 FROM pending_ingests AS p
             WHERE p.turn_id = OLD.turn_id AND p.status = 'indexed'
         ))
    )
)
BEGIN
    SELECT RAISE(ABORT, 'graph receipts allow only retry or terminal transition');
END;

CREATE TRIGGER IF NOT EXISTS trg_conversation_graph_chunks_no_update
BEFORE UPDATE ON conversation_graph_chunks
BEGIN
    SELECT RAISE(ABORT, 'conversation graph deltas are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_conversation_graph_chunks_no_delete
BEFORE DELETE ON conversation_graph_chunks
BEGIN
    SELECT RAISE(ABORT, 'conversation graph deltas are durable');
END;

CREATE TRIGGER IF NOT EXISTS trg_conversation_phrase_occurrences_no_update
BEFORE UPDATE ON conversation_phrase_occurrences
BEGIN
    SELECT RAISE(ABORT, 'conversation phrase occurrences are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_conversation_phrase_occurrences_no_delete
BEFORE DELETE ON conversation_phrase_occurrences
BEGIN
    SELECT RAISE(ABORT, 'conversation phrase occurrences are durable');
END;

CREATE TRIGGER IF NOT EXISTS trg_conversation_story_memberships_no_update
BEFORE UPDATE ON conversation_story_term_memberships
BEGIN
    SELECT RAISE(ABORT, 'conversation story memberships are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_conversation_story_memberships_no_delete
BEFORE DELETE ON conversation_story_term_memberships
BEGIN
    SELECT RAISE(ABORT, 'conversation story memberships are durable');
END;

CREATE TRIGGER IF NOT EXISTS trg_conversation_graph_state_monotone
BEFORE UPDATE ON conversation_graph_state
WHEN NOT (
    NEW.artifact_id = OLD.artifact_id
    AND NEW.revision >= OLD.revision
    AND NEW.chunk_count >= OLD.chunk_count
    AND NEW.occurrence_count >= OLD.occurrence_count
    AND NEW.story_term_membership_count >= OLD.story_term_membership_count
    AND NEW.story_evidence_chunk_count >= OLD.story_evidence_chunk_count
    AND NEW.ready_turn_count = OLD.ready_turn_count + 1
    AND NEW.updated_at IS NOT NULL
    AND NEW.state_sha256 <> OLD.state_sha256
)
BEGIN
    SELECT RAISE(ABORT, 'conversation graph state must advance monotonically');
END;

CREATE TRIGGER IF NOT EXISTS trg_conversation_graph_state_no_delete
BEFORE DELETE ON conversation_graph_state
BEGIN
    SELECT RAISE(ABORT, 'conversation graph state is durable');
END;

-- Replace v15's rolling-upgrade fences.  Their registered callback compares
-- against the current writer version, so leaving the literal 15 in place
-- would fence the v16 process that just completed this migration.
DROP TRIGGER IF EXISTS trg_v15_writer_fence_memory_insert;
DROP TRIGGER IF EXISTS trg_v15_writer_fence_memory_update;
DROP TRIGGER IF EXISTS trg_v15_writer_fence_enrichment_finalize;
DROP TRIGGER IF EXISTS trg_v15_writer_fence_enrichment_insert;
DROP TRIGGER IF EXISTS trg_v15_writer_fence_turn_insert;
DROP TRIGGER IF EXISTS trg_v15_writer_fence_turn_update;
DROP TRIGGER IF EXISTS trg_v15_writer_fence_turn_delete;
DROP TRIGGER IF EXISTS trg_v15_writer_fence_ingest_insert;
DROP TRIGGER IF EXISTS trg_v15_writer_fence_ingest_update;
DROP TRIGGER IF EXISTS trg_v15_writer_fence_reservation_insert;
DROP TRIGGER IF EXISTS trg_v15_writer_fence_chunk_insert;
DROP TRIGGER IF EXISTS trg_v15_writer_fence_chunk_update;
DROP TRIGGER IF EXISTS trg_v15_writer_fence_chunk_delete;
DROP TRIGGER IF EXISTS trg_v15_writer_fence_chunk_term_insert;
DROP TRIGGER IF EXISTS trg_v15_writer_fence_chunk_term_update;
DROP TRIGGER IF EXISTS trg_v15_writer_fence_chunk_term_delete;

CREATE TRIGGER IF NOT EXISTS trg_v16_writer_fence_memory_insert
BEFORE INSERT ON memory_items
WHEN memory_condense_writer_schema_version() <> 16
BEGIN SELECT RAISE(ABORT, 'memory writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v16_writer_fence_memory_update
BEFORE UPDATE ON memory_items
WHEN memory_condense_writer_schema_version() <> 16
BEGIN SELECT RAISE(ABORT, 'memory writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v16_writer_fence_enrichment_finalize
BEFORE UPDATE ON pending_enrichments
WHEN memory_condense_writer_schema_version() <> 16
BEGIN SELECT RAISE(ABORT, 'enrichment writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v16_writer_fence_enrichment_insert
BEFORE INSERT ON pending_enrichments
WHEN memory_condense_writer_schema_version() <> 16
BEGIN SELECT RAISE(ABORT, 'enrichment writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v16_writer_fence_turn_insert
BEFORE INSERT ON turns
WHEN memory_condense_writer_schema_version() <> 16
BEGIN SELECT RAISE(ABORT, 'turn writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v16_writer_fence_turn_update
BEFORE UPDATE ON turns
WHEN memory_condense_writer_schema_version() <> 16
BEGIN SELECT RAISE(ABORT, 'turn writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v16_writer_fence_turn_delete
BEFORE DELETE ON turns
WHEN memory_condense_writer_schema_version() <> 16
BEGIN SELECT RAISE(ABORT, 'turn writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v16_writer_fence_ingest_insert
BEFORE INSERT ON pending_ingests
WHEN memory_condense_writer_schema_version() <> 16
BEGIN SELECT RAISE(ABORT, 'ingest writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v16_writer_fence_ingest_update
BEFORE UPDATE ON pending_ingests
WHEN memory_condense_writer_schema_version() <> 16
BEGIN SELECT RAISE(ABORT, 'ingest writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v16_writer_fence_reservation_insert
BEFORE INSERT ON ingest_chunk_reservations
WHEN memory_condense_writer_schema_version() <> 16
BEGIN SELECT RAISE(ABORT, 'ingest writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v16_writer_fence_chunk_insert
BEFORE INSERT ON chunks
WHEN memory_condense_writer_schema_version() <> 16
BEGIN SELECT RAISE(ABORT, 'chunk writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v16_writer_fence_chunk_update
BEFORE UPDATE ON chunks
WHEN memory_condense_writer_schema_version() <> 16
BEGIN SELECT RAISE(ABORT, 'chunk writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v16_writer_fence_chunk_delete
BEFORE DELETE ON chunks
WHEN memory_condense_writer_schema_version() <> 16
BEGIN SELECT RAISE(ABORT, 'chunk writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v16_writer_fence_chunk_term_insert
BEFORE INSERT ON chunk_terms
WHEN memory_condense_writer_schema_version() <> 16
BEGIN SELECT RAISE(ABORT, 'lexical writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v16_writer_fence_chunk_term_update
BEFORE UPDATE ON chunk_terms
WHEN memory_condense_writer_schema_version() <> 16
BEGIN SELECT RAISE(ABORT, 'lexical writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v16_writer_fence_chunk_term_delete
BEFORE DELETE ON chunk_terms
WHEN memory_condense_writer_schema_version() <> 16
BEGIN SELECT RAISE(ABORT, 'lexical writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v16_writer_fence_graph_job_insert
BEFORE INSERT ON pending_graph_compilations
WHEN memory_condense_writer_schema_version() <> 16
BEGIN SELECT RAISE(ABORT, 'graph writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v16_writer_fence_graph_job_update
BEFORE UPDATE ON pending_graph_compilations
WHEN memory_condense_writer_schema_version() <> 16
BEGIN SELECT RAISE(ABORT, 'graph writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v16_writer_fence_graph_chunk_insert
BEFORE INSERT ON conversation_graph_chunks
WHEN memory_condense_writer_schema_version() <> 16
BEGIN SELECT RAISE(ABORT, 'graph writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v16_writer_fence_graph_occurrence_insert
BEFORE INSERT ON conversation_phrase_occurrences
WHEN memory_condense_writer_schema_version() <> 16
BEGIN SELECT RAISE(ABORT, 'graph writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v16_writer_fence_graph_story_insert
BEFORE INSERT ON conversation_story_term_memberships
WHEN memory_condense_writer_schema_version() <> 16
BEGIN SELECT RAISE(ABORT, 'graph writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v16_writer_fence_graph_state_update
BEFORE UPDATE ON conversation_graph_state
WHEN memory_condense_writer_schema_version() <> 16
BEGIN SELECT RAISE(ABORT, 'graph writer schema version mismatch'); END;
"""


__all__ = ["CONVERSATION_GRAPH_SCHEMA_V16"]
