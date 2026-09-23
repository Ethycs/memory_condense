"""Schema fragment for durable, append-only conversation envelopes."""

from __future__ import annotations


# T0 records only an immutable assignment input.  Boundary assignment runs in
# a separate bounded worker and publishes one immutable event per terminalized
# ``ready`` turn.  Raw text remains authoritative in ``turns``.
CONVERSATION_ENVELOPE_SCHEMA_V17 = """
CREATE TABLE IF NOT EXISTS conversation_envelope_policies (
    policy_sha256 TEXT PRIMARY KEY
        CHECK(length(policy_sha256) = 64
              AND policy_sha256 NOT GLOB '*[^0-9a-f]*'),
    format TEXT NOT NULL
        CHECK(format = 'memory-condense-conversation-envelope-v1'),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pending_conversation_envelope_assignments (
    policy_sha256 TEXT NOT NULL
        REFERENCES conversation_envelope_policies(policy_sha256),
    turn_id TEXT NOT NULL REFERENCES turns(turn_id),
    source_id TEXT,
    turn_ordinal INTEGER NOT NULL CHECK(turn_ordinal >= 0),
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    actor_kind TEXT NOT NULL
        CHECK(actor_kind IN ('user', 'assistant', 'system', 'tool')),
    authority_kind TEXT NOT NULL
        CHECK(authority_kind IN (
            'user_assertion', 'machine_generated', 'system_instruction',
            'tool_observation'
        )),
    parent_turn_id TEXT REFERENCES turns(turn_id),
    input_sha256 TEXT NOT NULL
        CHECK(length(input_sha256) = 64
              AND input_sha256 NOT GLOB '*[^0-9a-f]*'),
    status TEXT NOT NULL CHECK(status IN ('pending', 'ready', 'no_anchor')),
    created_at TEXT NOT NULL,
    completed_at TEXT,
    terminal_reason TEXT
        CHECK(terminal_reason IS NULL
              OR terminal_reason IN (
                  'missing_source', 'no_prior_user', 'invalid_parent'
              )),
    receipt_sha256 TEXT
        CHECK(receipt_sha256 IS NULL
              OR (length(receipt_sha256) = 64
                  AND receipt_sha256 NOT GLOB '*[^0-9a-f]*')),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count >= 0),
    last_attempt_at TEXT,
    last_error_kind TEXT,
    PRIMARY KEY (policy_sha256, turn_id),
    UNIQUE (policy_sha256, turn_ordinal),
    CHECK(
        (status = 'pending' AND completed_at IS NULL
         AND terminal_reason IS NULL AND receipt_sha256 IS NULL)
        OR
        (status = 'ready' AND completed_at IS NOT NULL
         AND terminal_reason IS NULL AND receipt_sha256 IS NOT NULL)
        OR
        (status = 'no_anchor' AND completed_at IS NOT NULL
         AND terminal_reason IS NOT NULL AND receipt_sha256 IS NOT NULL)
    ),
    CHECK(
        (attempt_count = 0
         AND last_attempt_at IS NULL AND last_error_kind IS NULL)
        OR
        (attempt_count > 0
         AND last_attempt_at IS NOT NULL AND length(last_error_kind) > 0)
    )
);

CREATE INDEX IF NOT EXISTS idx_pending_conversation_envelopes_status
ON pending_conversation_envelope_assignments(
    policy_sha256, status, turn_ordinal, turn_id
);

CREATE INDEX IF NOT EXISTS idx_pending_conversation_envelopes_source
ON pending_conversation_envelope_assignments(
    policy_sha256, source_id, status, turn_ordinal, turn_id
);

CREATE TABLE IF NOT EXISTS conversation_envelope_events (
    policy_sha256 TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    envelope_id TEXT NOT NULL
        CHECK(length(envelope_id) = 64
              AND envelope_id NOT GLOB '*[^0-9a-f]*'),
    opener_turn_id TEXT NOT NULL REFERENCES turns(turn_id),
    predecessor_envelope_id TEXT
        CHECK(predecessor_envelope_id IS NULL
              OR (length(predecessor_envelope_id) = 64
                  AND predecessor_envelope_id NOT GLOB '*[^0-9a-f]*')),
    event_kind TEXT NOT NULL CHECK(event_kind IN ('open', 'member')),
    source_id TEXT NOT NULL CHECK(length(trim(source_id)) > 0),
    turn_ordinal INTEGER NOT NULL CHECK(turn_ordinal >= 0),
    actor_kind TEXT NOT NULL
        CHECK(actor_kind IN ('user', 'assistant', 'system', 'tool')),
    authority_kind TEXT NOT NULL
        CHECK(authority_kind IN (
            'user_assertion', 'machine_generated', 'system_instruction',
            'tool_observation'
        )),
    parent_turn_id TEXT REFERENCES turns(turn_id),
    receipt_sha256 TEXT NOT NULL UNIQUE
        CHECK(length(receipt_sha256) = 64
              AND receipt_sha256 NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL,
    PRIMARY KEY (policy_sha256, turn_id),
    UNIQUE (policy_sha256, envelope_id, turn_ordinal),
    FOREIGN KEY (policy_sha256, turn_id)
        REFERENCES pending_conversation_envelope_assignments(
            policy_sha256, turn_id
        ),
    CHECK(
        (event_kind = 'open' AND opener_turn_id = turn_id
         AND actor_kind = 'user')
        OR
        (event_kind = 'member' AND opener_turn_id <> turn_id
         AND actor_kind IN ('assistant', 'system', 'tool')
         AND predecessor_envelope_id IS NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_conversation_envelope_events_envelope
ON conversation_envelope_events(
    policy_sha256, envelope_id, turn_ordinal, turn_id
);

CREATE INDEX IF NOT EXISTS idx_conversation_envelope_events_source_open
ON conversation_envelope_events(
    policy_sha256, source_id, event_kind, turn_ordinal, turn_id
);

CREATE TRIGGER IF NOT EXISTS trg_conversation_envelope_policies_no_update
BEFORE UPDATE ON conversation_envelope_policies
BEGIN
    SELECT RAISE(ABORT, 'conversation envelope policies are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_conversation_envelope_policies_no_delete
BEFORE DELETE ON conversation_envelope_policies
BEGIN
    SELECT RAISE(ABORT, 'conversation envelope policies are durable');
END;

CREATE TRIGGER IF NOT EXISTS trg_pending_conversation_envelopes_guard_insert
BEFORE INSERT ON pending_conversation_envelope_assignments
WHEN NOT EXISTS (
    SELECT 1 FROM turns AS t
    WHERE t.turn_id = NEW.turn_id
      AND t.source_id IS NEW.source_id
      AND t.ordinal = NEW.turn_ordinal
      AND t.role = NEW.role
)
BEGIN
    SELECT RAISE(ABORT, 'conversation envelope claim does not match its turn');
END;

CREATE TRIGGER IF NOT EXISTS trg_pending_conversation_envelopes_guard_update
BEFORE UPDATE ON pending_conversation_envelope_assignments
WHEN NOT (
    NEW.policy_sha256 = OLD.policy_sha256
    AND NEW.turn_id = OLD.turn_id
    AND NEW.source_id IS OLD.source_id
    AND NEW.turn_ordinal = OLD.turn_ordinal
    AND NEW.role = OLD.role
    AND NEW.actor_kind = OLD.actor_kind
    AND NEW.authority_kind = OLD.authority_kind
    AND NEW.parent_turn_id IS OLD.parent_turn_id
    AND NEW.input_sha256 = OLD.input_sha256
    AND NEW.created_at = OLD.created_at
    AND (
        (OLD.status = 'pending' AND NEW.status = 'pending'
         AND NEW.completed_at IS NULL AND NEW.receipt_sha256 IS NULL
         AND NEW.terminal_reason IS NULL
         AND NEW.attempt_count = OLD.attempt_count + 1
         AND NEW.last_attempt_at IS NOT NULL
         AND length(NEW.last_error_kind) > 0)
        OR
        (OLD.status = 'pending' AND NEW.status IN ('ready', 'no_anchor')
         AND NEW.completed_at IS NOT NULL AND NEW.receipt_sha256 IS NOT NULL
         AND ((NEW.status = 'ready' AND NEW.terminal_reason IS NULL)
              OR (NEW.status = 'no_anchor'
                  AND NEW.terminal_reason IN (
                      'missing_source', 'no_prior_user', 'invalid_parent'
                  )))
         AND NEW.attempt_count = OLD.attempt_count
         AND NEW.last_attempt_at IS OLD.last_attempt_at
         AND NEW.last_error_kind IS OLD.last_error_kind)
    )
)
BEGIN
    SELECT RAISE(ABORT, 'conversation envelope assignment is monotone');
END;

CREATE TRIGGER IF NOT EXISTS trg_pending_conversation_envelopes_no_delete
BEFORE DELETE ON pending_conversation_envelope_assignments
BEGIN
    SELECT RAISE(ABORT, 'conversation envelope assignments are durable');
END;

CREATE TRIGGER IF NOT EXISTS trg_conversation_envelope_events_guard_insert
BEFORE INSERT ON conversation_envelope_events
WHEN NOT (
    EXISTS (
        SELECT 1 FROM pending_conversation_envelope_assignments AS p
        WHERE p.policy_sha256 = NEW.policy_sha256
          AND p.turn_id = NEW.turn_id
          AND p.status = 'pending'
          AND p.source_id = NEW.source_id
          AND p.turn_ordinal = NEW.turn_ordinal
          AND p.actor_kind = NEW.actor_kind
          AND p.authority_kind = NEW.authority_kind
          AND p.parent_turn_id IS NEW.parent_turn_id
    )
    AND (
        (NEW.event_kind = 'open'
         AND (NEW.predecessor_envelope_id IS NULL OR EXISTS (
             SELECT 1 FROM conversation_envelope_events AS previous
             WHERE previous.policy_sha256 = NEW.policy_sha256
               AND previous.envelope_id = NEW.predecessor_envelope_id
               AND previous.event_kind = 'open'
               AND previous.source_id = NEW.source_id
               AND previous.turn_ordinal < NEW.turn_ordinal
         )))
        OR
        (NEW.event_kind = 'member' AND EXISTS (
            SELECT 1 FROM conversation_envelope_events AS opener
            WHERE opener.policy_sha256 = NEW.policy_sha256
              AND opener.turn_id = NEW.opener_turn_id
              AND opener.event_kind = 'open'
              AND opener.envelope_id = NEW.envelope_id
              AND opener.source_id = NEW.source_id
              AND opener.turn_ordinal < NEW.turn_ordinal
        ))
    )
    AND (NEW.parent_turn_id IS NULL OR EXISTS (
        SELECT 1 FROM conversation_envelope_events AS parent
        WHERE parent.policy_sha256 = NEW.policy_sha256
          AND parent.turn_id = NEW.parent_turn_id
          AND parent.source_id = NEW.source_id
          AND parent.turn_ordinal < NEW.turn_ordinal
    ))
)
BEGIN
    SELECT RAISE(ABORT, 'conversation envelope event has no matching claim');
END;

CREATE TRIGGER IF NOT EXISTS trg_conversation_envelope_events_no_update
BEFORE UPDATE ON conversation_envelope_events
BEGIN
    SELECT RAISE(ABORT, 'conversation envelope events are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_conversation_envelope_events_no_delete
BEFORE DELETE ON conversation_envelope_events
BEGIN
    SELECT RAISE(ABORT, 'conversation envelope events are durable');
END;

-- Replace v16 rolling-upgrade fences with the v17 writer identity.
DROP TRIGGER IF EXISTS trg_v16_writer_fence_memory_insert;
DROP TRIGGER IF EXISTS trg_v16_writer_fence_memory_update;
DROP TRIGGER IF EXISTS trg_v16_writer_fence_enrichment_finalize;
DROP TRIGGER IF EXISTS trg_v16_writer_fence_enrichment_insert;
DROP TRIGGER IF EXISTS trg_v16_writer_fence_turn_insert;
DROP TRIGGER IF EXISTS trg_v16_writer_fence_turn_update;
DROP TRIGGER IF EXISTS trg_v16_writer_fence_turn_delete;
DROP TRIGGER IF EXISTS trg_v16_writer_fence_ingest_insert;
DROP TRIGGER IF EXISTS trg_v16_writer_fence_ingest_update;
DROP TRIGGER IF EXISTS trg_v16_writer_fence_reservation_insert;
DROP TRIGGER IF EXISTS trg_v16_writer_fence_chunk_insert;
DROP TRIGGER IF EXISTS trg_v16_writer_fence_chunk_update;
DROP TRIGGER IF EXISTS trg_v16_writer_fence_chunk_delete;
DROP TRIGGER IF EXISTS trg_v16_writer_fence_chunk_term_insert;
DROP TRIGGER IF EXISTS trg_v16_writer_fence_chunk_term_update;
DROP TRIGGER IF EXISTS trg_v16_writer_fence_chunk_term_delete;
DROP TRIGGER IF EXISTS trg_v16_writer_fence_graph_job_insert;
DROP TRIGGER IF EXISTS trg_v16_writer_fence_graph_job_update;
DROP TRIGGER IF EXISTS trg_v16_writer_fence_graph_chunk_insert;
DROP TRIGGER IF EXISTS trg_v16_writer_fence_graph_occurrence_insert;
DROP TRIGGER IF EXISTS trg_v16_writer_fence_graph_story_insert;
DROP TRIGGER IF EXISTS trg_v16_writer_fence_graph_state_update;

CREATE TRIGGER IF NOT EXISTS trg_v17_writer_fence_memory_insert
BEFORE INSERT ON memory_items
WHEN memory_condense_writer_schema_version() <> 17
BEGIN SELECT RAISE(ABORT, 'memory writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v17_writer_fence_memory_update
BEFORE UPDATE ON memory_items
WHEN memory_condense_writer_schema_version() <> 17
BEGIN SELECT RAISE(ABORT, 'memory writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v17_writer_fence_enrichment_finalize
BEFORE UPDATE ON pending_enrichments
WHEN memory_condense_writer_schema_version() <> 17
BEGIN SELECT RAISE(ABORT, 'enrichment writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v17_writer_fence_enrichment_insert
BEFORE INSERT ON pending_enrichments
WHEN memory_condense_writer_schema_version() <> 17
BEGIN SELECT RAISE(ABORT, 'enrichment writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v17_writer_fence_turn_insert
BEFORE INSERT ON turns
WHEN memory_condense_writer_schema_version() <> 17
BEGIN SELECT RAISE(ABORT, 'turn writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v17_writer_fence_turn_update
BEFORE UPDATE ON turns
WHEN memory_condense_writer_schema_version() <> 17
BEGIN SELECT RAISE(ABORT, 'turn writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v17_writer_fence_turn_delete
BEFORE DELETE ON turns
WHEN memory_condense_writer_schema_version() <> 17
BEGIN SELECT RAISE(ABORT, 'turn writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v17_writer_fence_ingest_insert
BEFORE INSERT ON pending_ingests
WHEN memory_condense_writer_schema_version() <> 17
BEGIN SELECT RAISE(ABORT, 'ingest writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v17_writer_fence_ingest_update
BEFORE UPDATE ON pending_ingests
WHEN memory_condense_writer_schema_version() <> 17
BEGIN SELECT RAISE(ABORT, 'ingest writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v17_writer_fence_reservation_insert
BEFORE INSERT ON ingest_chunk_reservations
WHEN memory_condense_writer_schema_version() <> 17
BEGIN SELECT RAISE(ABORT, 'ingest writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v17_writer_fence_chunk_insert
BEFORE INSERT ON chunks
WHEN memory_condense_writer_schema_version() <> 17
BEGIN SELECT RAISE(ABORT, 'chunk writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v17_writer_fence_chunk_update
BEFORE UPDATE ON chunks
WHEN memory_condense_writer_schema_version() <> 17
BEGIN SELECT RAISE(ABORT, 'chunk writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v17_writer_fence_chunk_delete
BEFORE DELETE ON chunks
WHEN memory_condense_writer_schema_version() <> 17
BEGIN SELECT RAISE(ABORT, 'chunk writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v17_writer_fence_chunk_term_insert
BEFORE INSERT ON chunk_terms
WHEN memory_condense_writer_schema_version() <> 17
BEGIN SELECT RAISE(ABORT, 'lexical writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v17_writer_fence_chunk_term_update
BEFORE UPDATE ON chunk_terms
WHEN memory_condense_writer_schema_version() <> 17
BEGIN SELECT RAISE(ABORT, 'lexical writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v17_writer_fence_chunk_term_delete
BEFORE DELETE ON chunk_terms
WHEN memory_condense_writer_schema_version() <> 17
BEGIN SELECT RAISE(ABORT, 'lexical writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v17_writer_fence_graph_job_insert
BEFORE INSERT ON pending_graph_compilations
WHEN memory_condense_writer_schema_version() <> 17
BEGIN SELECT RAISE(ABORT, 'graph writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v17_writer_fence_graph_job_update
BEFORE UPDATE ON pending_graph_compilations
WHEN memory_condense_writer_schema_version() <> 17
BEGIN SELECT RAISE(ABORT, 'graph writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v17_writer_fence_graph_chunk_insert
BEFORE INSERT ON conversation_graph_chunks
WHEN memory_condense_writer_schema_version() <> 17
BEGIN SELECT RAISE(ABORT, 'graph writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v17_writer_fence_graph_occurrence_insert
BEFORE INSERT ON conversation_phrase_occurrences
WHEN memory_condense_writer_schema_version() <> 17
BEGIN SELECT RAISE(ABORT, 'graph writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v17_writer_fence_graph_story_insert
BEFORE INSERT ON conversation_story_term_memberships
WHEN memory_condense_writer_schema_version() <> 17
BEGIN SELECT RAISE(ABORT, 'graph writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v17_writer_fence_graph_state_update
BEFORE UPDATE ON conversation_graph_state
WHEN memory_condense_writer_schema_version() <> 17
BEGIN SELECT RAISE(ABORT, 'graph writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v17_writer_fence_envelope_job_insert
BEFORE INSERT ON pending_conversation_envelope_assignments
WHEN memory_condense_writer_schema_version() <> 17
BEGIN SELECT RAISE(ABORT, 'envelope writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v17_writer_fence_envelope_job_update
BEFORE UPDATE ON pending_conversation_envelope_assignments
WHEN memory_condense_writer_schema_version() <> 17
BEGIN SELECT RAISE(ABORT, 'envelope writer schema version mismatch'); END;
CREATE TRIGGER IF NOT EXISTS trg_v17_writer_fence_envelope_event_insert
BEFORE INSERT ON conversation_envelope_events
WHEN memory_condense_writer_schema_version() <> 17
BEGIN SELECT RAISE(ABORT, 'envelope writer schema version mismatch'); END;
"""


__all__ = ["CONVERSATION_ENVELOPE_SCHEMA_V17"]
