from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable

CURRENT_SCHEMA_VERSION = 11


def _execute_sql_script(
    conn: sqlite3.Connection,
    script: str,
) -> None:
    """Execute a multi-statement script without sqlite3's implicit COMMIT.

    ``Connection.executescript`` commits any open transaction before it runs,
    which would publish DDL and ``schema_version`` before a Python post-hook
    succeeds.  ``complete_statement`` also keeps trigger bodies intact while
    this helper executes each top-level statement on the caller's transaction.
    """

    pending = ""
    for character in script:
        pending += character
        if character == ";" and sqlite3.complete_statement(pending):
            conn.execute(pending)
            pending = ""
    if pending.strip():
        conn.execute(pending)


def _apply_schema_transaction(
    conn: sqlite3.Connection,
    script: str,
    *,
    post: Callable[[sqlite3.Connection], None] | None = None,
) -> None:
    """Atomically apply DDL, its backfill hook, and version publication."""

    conn.execute("BEGIN IMMEDIATE")
    try:
        _execute_sql_script(conn, script)
        if post is not None:
            post(conn)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise

#: DDL introduced by v3, written once and reused by both the fresh-database
#: path and the migration path.
#:
#: ``_SCHEMA_SQL`` and ``_MIGRATIONS`` are otherwise two hand-maintained copies
#: of the same schema, and the standard's own verification block warns that
#: they drift. Anything shareable should be shared. The ``content_hash``
#: *column* cannot be — a fresh database declares it inside CREATE TABLE while
#: an existing one needs ALTER TABLE — so `test_db.py` asserts the two paths
#: converge on the same columns and indexes instead of the same DDL text.
_V3_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_memory_content_hash ON memory_items(content_hash);
"""

#: Compact, external association state introduced by v5.  These tables are
#: intentionally incapable of storing growing request-derived token state:
#: a row contains one fixed-width CAV signature or one sparse episode edge.
#: Source text remains in ``chunks`` and every attention workspace is
#: disposable after it emits these records. Static model/tokenizer assets are
#: reusable machinery and are not represented by this request-state invariant.
_V5_ASSOCIATION_SCHEMA = """
CREATE TABLE IF NOT EXISTS association_artifacts (
    artifact_id      TEXT PRIMARY KEY,
    model_id         TEXT NOT NULL,
    checkpoint_id    TEXT NOT NULL,
    prefix_layers    INTEGER NOT NULL CHECK(prefix_layers > 0),
    head_layer       INTEGER NOT NULL CHECK(head_layer >= 0),
    cav_layer        INTEGER CHECK(cav_layer IS NULL OR cav_layer >= 0),
    concept_names    TEXT NOT NULL,
    head_count       INTEGER NOT NULL CHECK(head_count > 0),
    created_at       TEXT NOT NULL,
    metadata         TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS chunk_cav_signatures (
    chunk_id         TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    artifact_id      TEXT NOT NULL REFERENCES association_artifacts(artifact_id)
                     ON DELETE CASCADE,
    signature        BLOB NOT NULL,
    created_turn     INTEGER NOT NULL DEFAULT 0 CHECK(created_turn >= 0),
    access_count     INTEGER NOT NULL DEFAULT 0 CHECK(access_count >= 0),
    last_access_turn INTEGER NOT NULL DEFAULT 0 CHECK(last_access_turn >= 0),
    PRIMARY KEY (chunk_id, artifact_id)
);

CREATE INDEX IF NOT EXISTS idx_chunk_cav_artifact
ON chunk_cav_signatures(artifact_id, chunk_id);

CREATE TABLE IF NOT EXISTS chunk_head_edges (
    source_chunk_id      TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    destination_chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    artifact_id          TEXT NOT NULL REFERENCES association_artifacts(artifact_id)
                         ON DELETE CASCADE,
    head_weights         BLOB NOT NULL,
    qk_score             REAL NOT NULL CHECK(qk_score >= 0.0),
    ov_transport         REAL NOT NULL DEFAULT 0.0 CHECK(ov_transport >= 0.0),
    evidence_count       INTEGER NOT NULL DEFAULT 1 CHECK(evidence_count > 0),
    traversal_count      INTEGER NOT NULL DEFAULT 0 CHECK(traversal_count >= 0),
    last_access_turn     INTEGER NOT NULL DEFAULT 0 CHECK(last_access_turn >= 0),
    temporal_forward     INTEGER CHECK(temporal_forward IN (0, 1)
                                       OR temporal_forward IS NULL),
    PRIMARY KEY (source_chunk_id, destination_chunk_id, artifact_id),
    CHECK(source_chunk_id <> destination_chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_chunk_head_edges_destination
ON chunk_head_edges(artifact_id, destination_chunk_id);
"""

#: Live access learning introduced by v7.  The graph stores only durable chunk
#: IDs and scalar, turn-decayed co-access statistics.  ``event_fingerprint`` is
#: an idempotency receipt (a SHA-256 digest), not a serialized retrieval set;
#: callers cap receipt history so event bookkeeping cannot grow without bound.
_V7_HEBBIAN_SCHEMA = """
CREATE TABLE IF NOT EXISTS hebbian_access_events (
    artifact_id       TEXT NOT NULL REFERENCES association_artifacts(artifact_id)
                      ON DELETE CASCADE,
    event_id          TEXT NOT NULL,
    observed_turn     INTEGER NOT NULL CHECK(observed_turn >= 0),
    event_fingerprint TEXT NOT NULL,
    member_count      INTEGER NOT NULL CHECK(member_count >= 0),
    PRIMARY KEY (artifact_id, event_id)
);

CREATE INDEX IF NOT EXISTS idx_hebbian_events_turn
ON hebbian_access_events(artifact_id, observed_turn);

CREATE TABLE IF NOT EXISTS hebbian_chunk_nodes (
    artifact_id     TEXT NOT NULL REFERENCES association_artifacts(artifact_id)
                    ON DELETE CASCADE,
    chunk_id        TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    access_mass     REAL NOT NULL DEFAULT 0.0 CHECK(access_mass >= 0.0),
    access_count    INTEGER NOT NULL DEFAULT 0 CHECK(access_count >= 0),
    last_access_turn INTEGER NOT NULL DEFAULT 0 CHECK(last_access_turn >= 0),
    PRIMARY KEY (artifact_id, chunk_id)
);

CREATE TABLE IF NOT EXISTS hebbian_chunk_edges (
    artifact_id         TEXT NOT NULL REFERENCES association_artifacts(artifact_id)
                        ON DELETE CASCADE,
    chunk_low           TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    chunk_high          TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    coaccess_mass       REAL NOT NULL DEFAULT 0.0 CHECK(coaccess_mass >= 0.0),
    coaccess_count      INTEGER NOT NULL DEFAULT 0 CHECK(coaccess_count >= 0),
    last_reinforced_turn INTEGER NOT NULL DEFAULT 0
                         CHECK(last_reinforced_turn >= 0),
    PRIMARY KEY (artifact_id, chunk_low, chunk_high),
    CHECK(chunk_low < chunk_high)
);

CREATE INDEX IF NOT EXISTS idx_hebbian_edges_high
ON hebbian_chunk_edges(artifact_id, chunk_high);
"""

#: Model-independent live consolidation introduced by v8.  Nodes are durable
#: references into the semantic-memory and evidence partitions; edges contain
#: only decayed scalar co-activation statistics.  In particular there is no
#: column capable of retaining a prompt, token stream, activation, or K/V
#: cache.  Repeated later contexts strengthen useful assemblies while the
#: ordinary turn clock weakens associations that stop recurring.
_V8_CONSOLIDATION_SCHEMA = """
CREATE TABLE IF NOT EXISTS consolidation_access_events (
    event_id          TEXT PRIMARY KEY,
    observed_turn     INTEGER NOT NULL CHECK(observed_turn >= 0),
    event_fingerprint TEXT NOT NULL,
    member_count      INTEGER NOT NULL CHECK(member_count >= 0)
);

CREATE INDEX IF NOT EXISTS idx_consolidation_events_turn
ON consolidation_access_events(observed_turn);

CREATE TABLE IF NOT EXISTS consolidation_nodes (
    node_key         TEXT PRIMARY KEY,
    node_kind        TEXT NOT NULL CHECK(node_kind IN ('memory', 'chunk')),
    memory_id        TEXT UNIQUE REFERENCES memory_items(mem_id) ON DELETE CASCADE,
    chunk_id         TEXT UNIQUE REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    access_mass      REAL NOT NULL DEFAULT 0.0 CHECK(access_mass >= 0.0),
    access_count     INTEGER NOT NULL DEFAULT 0 CHECK(access_count >= 0),
    last_access_turn INTEGER NOT NULL DEFAULT 0 CHECK(last_access_turn >= 0),
    CHECK(
        (node_kind = 'memory' AND memory_id IS NOT NULL AND chunk_id IS NULL)
        OR
        (node_kind = 'chunk' AND chunk_id IS NOT NULL AND memory_id IS NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_consolidation_nodes_kind
ON consolidation_nodes(node_kind, node_key);

CREATE TABLE IF NOT EXISTS consolidation_edges (
    node_low             TEXT NOT NULL REFERENCES consolidation_nodes(node_key)
                         ON DELETE CASCADE,
    node_high            TEXT NOT NULL REFERENCES consolidation_nodes(node_key)
                         ON DELETE CASCADE,
    coactivation_mass    REAL NOT NULL DEFAULT 0.0
                         CHECK(coactivation_mass >= 0.0),
    coactivation_count   INTEGER NOT NULL DEFAULT 0
                         CHECK(coactivation_count >= 0),
    last_reinforced_turn INTEGER NOT NULL DEFAULT 0
                         CHECK(last_reinforced_turn >= 0),
    PRIMARY KEY (node_low, node_high),
    CHECK(node_low < node_high)
);

CREATE INDEX IF NOT EXISTS idx_consolidation_edges_high
ON consolidation_edges(node_high);

-- A retired semantic memory must not remain reachable through learned state.
-- The underlying memory row and provenance remain intact; only reconstructible
-- consolidation state is removed.
CREATE TRIGGER IF NOT EXISTS trg_consolidation_retire_memory
AFTER UPDATE OF status ON memory_items
WHEN NEW.status <> 'active'
BEGIN
    DELETE FROM consolidation_nodes WHERE memory_id = NEW.mem_id;
END;

-- Chunk deletion is represented by clearing its retrieval payload while the
-- provenance row survives.  Remove only the learned node and incident edges.
CREATE TRIGGER IF NOT EXISTS trg_consolidation_retire_chunk
AFTER UPDATE OF embedding ON chunks
WHEN NEW.embedding IS NULL
BEGIN
    DELETE FROM consolidation_nodes WHERE chunk_id = NEW.chunk_id;
END;
"""

_V9_CAUSAL_BINDING_SCHEMA = """
ALTER TABLE consolidation_edges
ADD COLUMN causal_count INTEGER NOT NULL DEFAULT 0 CHECK(causal_count >= 0);
"""

#: Source-grounded episodic discourse graph introduced by v10.  The graph is
#: deliberately reference-only: factual text remains in ``turns``/``chunks``;
#: these tables retain exact span coordinates and hashes, scalar routing
#: metadata, and immutable publication receipts.  There is no column for
#: generated evidence text, token IDs, activations, attention maps, or K/V.
_V10_DISCOURSE_SCHEMA = """
CREATE TABLE IF NOT EXISTS discourse_artifacts (
    artifact_id          TEXT PRIMARY KEY,
    kind                 TEXT NOT NULL CHECK(length(trim(kind)) > 0),
    implementation_sha256 TEXT NOT NULL
                          CHECK(length(implementation_sha256) = 64
                                AND implementation_sha256 NOT GLOB '*[^0-9a-f]*'),
    policy_sha256        TEXT NOT NULL
                          CHECK(length(policy_sha256) = 64
                                AND policy_sha256 NOT GLOB '*[^0-9a-f]*'),
    model_id             TEXT,
    model_revision       TEXT,
    checkpoint_sha256    TEXT
                          CHECK(checkpoint_sha256 IS NULL
                                OR (length(checkpoint_sha256) = 64
                                    AND checkpoint_sha256 NOT GLOB '*[^0-9a-f]*')),
    metadata             TEXT NOT NULL DEFAULT '{}'
                          CHECK(json_valid(metadata) AND json_type(metadata) = 'object')
);

CREATE INDEX IF NOT EXISTS idx_discourse_artifacts_kind
ON discourse_artifacts(kind, artifact_id);

CREATE TABLE IF NOT EXISTS episodes (
    episode_id        TEXT PRIMARY KEY,
    artifact_id       TEXT NOT NULL REFERENCES discourse_artifacts(artifact_id)
                      ON DELETE CASCADE,
    source_id         TEXT NOT NULL CHECK(length(trim(source_id)) > 0),
    sequence_no       INTEGER NOT NULL CHECK(sequence_no >= 0),
    first_ordinal     INTEGER NOT NULL CHECK(first_ordinal >= 0),
    last_ordinal      INTEGER NOT NULL CHECK(last_ordinal >= first_ordinal),
    boundary_method   TEXT NOT NULL CHECK(length(trim(boundary_method)) > 0),
    initial_boundary  INTEGER CHECK(initial_boundary IS NULL OR initial_boundary >= 0),
    refined_boundary  INTEGER CHECK(refined_boundary IS NULL OR refined_boundary >= 0),
    boundary_score    REAL,
    boundary_threshold REAL,
    receipt_sha256    TEXT NOT NULL UNIQUE
                      CHECK(length(receipt_sha256) = 64
                            AND receipt_sha256 NOT GLOB '*[^0-9a-f]*'),
    UNIQUE (artifact_id, source_id, sequence_no)
);

CREATE INDEX IF NOT EXISTS idx_episodes_source_order
ON episodes(artifact_id, source_id, sequence_no, episode_id);
CREATE INDEX IF NOT EXISTS idx_episodes_ordinal
ON episodes(artifact_id, first_ordinal, last_ordinal, episode_id);

CREATE TABLE IF NOT EXISTS episode_evidence (
    episode_id      TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE CASCADE,
    evidence_order INTEGER NOT NULL CHECK(evidence_order >= 0),
    chunk_id        TEXT NOT NULL REFERENCES chunks(chunk_id),
    start_char      INTEGER NOT NULL CHECK(start_char >= 0),
    end_char        INTEGER NOT NULL CHECK(end_char > start_char),
    quote_sha256    TEXT NOT NULL
                    CHECK(length(quote_sha256) = 64
                          AND quote_sha256 NOT GLOB '*[^0-9a-f]*'),
    ordinal         INTEGER NOT NULL CHECK(ordinal >= 0),
    source_id       TEXT,
    turn_start_char INTEGER NOT NULL DEFAULT 0 CHECK(turn_start_char >= 0),
    PRIMARY KEY (episode_id, evidence_order)
);

CREATE INDEX IF NOT EXISTS idx_episode_evidence_chunk
ON episode_evidence(chunk_id, episode_id, evidence_order);

CREATE TABLE IF NOT EXISTS episode_representatives (
    episode_id            TEXT NOT NULL REFERENCES episodes(episode_id)
                          ON DELETE CASCADE,
    chunk_id              TEXT NOT NULL REFERENCES chunks(chunk_id),
    rank                  INTEGER NOT NULL CHECK(rank >= 0),
    vector_identity_sha256 TEXT NOT NULL
                           CHECK(length(vector_identity_sha256) = 64
                                 AND vector_identity_sha256 NOT GLOB '*[^0-9a-f]*'),
    PRIMARY KEY (episode_id, rank),
    UNIQUE (episode_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_episode_representatives_chunk
ON episode_representatives(chunk_id, episode_id, rank);

CREATE TABLE IF NOT EXISTS discourse_units (
    unit_id          TEXT PRIMARY KEY,
    artifact_id      TEXT NOT NULL REFERENCES discourse_artifacts(artifact_id)
                     ON DELETE CASCADE,
    kind             TEXT NOT NULL CHECK(length(trim(kind)) > 0),
    canonical_key    TEXT NOT NULL CHECK(length(trim(canonical_key)) > 0),
    asserted_ordinal INTEGER NOT NULL CHECK(asserted_ordinal >= 0),
    confidence       REAL NOT NULL CHECK(confidence >= 0.0 AND confidence <= 1.0),
    metadata         TEXT NOT NULL DEFAULT '{}'
                     CHECK(json_valid(metadata) AND json_type(metadata) = 'object')
);

CREATE INDEX IF NOT EXISTS idx_discourse_units_key
ON discourse_units(artifact_id, kind, canonical_key, asserted_ordinal, unit_id);

CREATE TABLE IF NOT EXISTS discourse_unit_evidence (
    unit_id         TEXT NOT NULL REFERENCES discourse_units(unit_id) ON DELETE CASCADE,
    evidence_order INTEGER NOT NULL CHECK(evidence_order >= 0),
    chunk_id        TEXT NOT NULL REFERENCES chunks(chunk_id),
    start_char      INTEGER NOT NULL CHECK(start_char >= 0),
    end_char        INTEGER NOT NULL CHECK(end_char > start_char),
    quote_sha256    TEXT NOT NULL
                    CHECK(length(quote_sha256) = 64
                          AND quote_sha256 NOT GLOB '*[^0-9a-f]*'),
    ordinal         INTEGER NOT NULL CHECK(ordinal >= 0),
    source_id       TEXT,
    turn_start_char INTEGER NOT NULL DEFAULT 0 CHECK(turn_start_char >= 0),
    PRIMARY KEY (unit_id, evidence_order)
);

CREATE INDEX IF NOT EXISTS idx_discourse_unit_evidence_chunk
ON discourse_unit_evidence(chunk_id, unit_id, evidence_order);

CREATE TABLE IF NOT EXISTS discourse_relations (
    relation_id     TEXT PRIMARY KEY,
    artifact_id     TEXT NOT NULL REFERENCES discourse_artifacts(artifact_id)
                    ON DELETE CASCADE,
    relation_type   TEXT NOT NULL CHECK(length(trim(relation_type)) > 0),
    confidence      REAL NOT NULL CHECK(confidence >= 0.0 AND confidence <= 1.0),
    created_ordinal INTEGER NOT NULL CHECK(created_ordinal >= 0),
    metadata        TEXT NOT NULL DEFAULT '{}'
                    CHECK(json_valid(metadata) AND json_type(metadata) = 'object')
);

CREATE INDEX IF NOT EXISTS idx_discourse_relations_type
ON discourse_relations(artifact_id, relation_type, created_ordinal, relation_id);

CREATE TABLE IF NOT EXISTS discourse_relation_members (
    relation_id  TEXT NOT NULL REFERENCES discourse_relations(relation_id)
                 ON DELETE CASCADE,
    member_order INTEGER NOT NULL CHECK(member_order >= 0),
    unit_id      TEXT NOT NULL REFERENCES discourse_units(unit_id),
    role         TEXT NOT NULL CHECK(length(trim(role)) > 0),
    ordinal      INTEGER NOT NULL CHECK(ordinal >= 0),
    weight       REAL NOT NULL CHECK(weight >= 0.0),
    PRIMARY KEY (relation_id, member_order),
    UNIQUE (relation_id, unit_id, role, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_discourse_relation_members_unit
ON discourse_relation_members(unit_id, relation_id, member_order);

CREATE TABLE IF NOT EXISTS discourse_relation_evidence (
    relation_id    TEXT NOT NULL REFERENCES discourse_relations(relation_id)
                   ON DELETE CASCADE,
    evidence_order INTEGER NOT NULL CHECK(evidence_order >= 0),
    chunk_id       TEXT NOT NULL REFERENCES chunks(chunk_id),
    start_char     INTEGER NOT NULL CHECK(start_char >= 0),
    end_char       INTEGER NOT NULL CHECK(end_char > start_char),
    quote_sha256   TEXT NOT NULL
                   CHECK(length(quote_sha256) = 64
                         AND quote_sha256 NOT GLOB '*[^0-9a-f]*'),
    ordinal        INTEGER NOT NULL CHECK(ordinal >= 0),
    source_id      TEXT,
    turn_start_char INTEGER NOT NULL DEFAULT 0 CHECK(turn_start_char >= 0),
    PRIMARY KEY (relation_id, evidence_order)
);

CREATE INDEX IF NOT EXISTS idx_discourse_relation_evidence_chunk
ON discourse_relation_evidence(chunk_id, relation_id, evidence_order);

CREATE TABLE IF NOT EXISTS discourse_graph_revisions (
    graph_revision   INTEGER PRIMARY KEY CHECK(graph_revision > 0),
    max_turn_ordinal INTEGER NOT NULL CHECK(max_turn_ordinal >= 0),
    chunk_count      INTEGER NOT NULL CHECK(chunk_count >= 0),
    schema_version   INTEGER NOT NULL CHECK(schema_version >= 0),
    artifact_ids     TEXT NOT NULL
                     CHECK(json_valid(artifact_ids) AND json_type(artifact_ids) = 'array'),
    snapshot_sha256  TEXT NOT NULL UNIQUE
                     CHECK(length(snapshot_sha256) = 64
                           AND snapshot_sha256 NOT GLOB '*[^0-9a-f]*')
);

CREATE TRIGGER IF NOT EXISTS trg_discourse_revision_no_update
BEFORE UPDATE ON discourse_graph_revisions
BEGIN
    SELECT RAISE(ABORT, 'discourse graph revision receipts are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_discourse_revision_no_delete
BEFORE DELETE ON discourse_graph_revisions
BEGIN
    SELECT RAISE(ABORT, 'discourse graph revision receipts are immutable');
END;
"""


_IMMUTABLE_DISCOURSE_TABLES = (
    "discourse_artifacts",
    "episodes",
    "episode_evidence",
    "episode_representatives",
    "discourse_units",
    "discourse_unit_evidence",
    "discourse_relations",
    "discourse_relation_members",
    "discourse_relation_evidence",
    "discourse_artifact_coverage",
    "discourse_artifact_coverage_receipts",
)


def _immutable_discourse_triggers() -> str:
    return "\n".join(
        f"""
CREATE TRIGGER IF NOT EXISTS trg_{table}_no_update
BEFORE UPDATE ON {table}
BEGIN
    SELECT RAISE(ABORT, 'published discourse graph rows are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_{table}_no_delete
BEFORE DELETE ON {table}
BEGIN
    SELECT RAISE(ABORT, 'published discourse graph rows are immutable');
END;
"""
        for table in _IMMUTABLE_DISCOURSE_TABLES
    )


def _graph_content_revision_triggers() -> str:
    return "\n".join(
        f"""
CREATE TRIGGER IF NOT EXISTS trg_{table}_content_insert
AFTER INSERT ON {table}
BEGIN
    UPDATE discourse_revision_state
    SET graph_content_revision = graph_content_revision + 1
    WHERE singleton = 1;
END;
"""
        for table in _IMMUTABLE_DISCOURSE_TABLES
    )


# v11 closes snapshot TOCTOU holes.  Source facts and immutable graph content
# each advance a durable monotonic counter.  Derived retrieval/index payloads
# (embeddings, lexical weights, HNSW labels, term counts) intentionally do not
# invalidate a factual closure receipt.
_V11_REVISION_SCHEMA = """
-- ``ordinal`` is the authoritative global transcript clock used by evidence
-- ordering and temporal closure.  A UNIQUE index both rejects concurrent/raw
-- duplicate writes and fails migration closed if a legacy database has an
-- ambiguous clock that must be repaired explicitly.
CREATE UNIQUE INDEX IF NOT EXISTS idx_turns_ordinal_unique ON turns(ordinal);

ALTER TABLE episode_evidence ADD COLUMN turn_id TEXT;
ALTER TABLE episode_evidence ADD COLUMN role TEXT;
ALTER TABLE episode_evidence ADD COLUMN created_at TEXT;

ALTER TABLE discourse_unit_evidence ADD COLUMN turn_id TEXT;
ALTER TABLE discourse_unit_evidence ADD COLUMN role TEXT;
ALTER TABLE discourse_unit_evidence ADD COLUMN created_at TEXT;

ALTER TABLE discourse_relation_evidence ADD COLUMN turn_id TEXT;
ALTER TABLE discourse_relation_evidence ADD COLUMN role TEXT;
ALTER TABLE discourse_relation_evidence ADD COLUMN created_at TEXT;

ALTER TABLE discourse_graph_revisions
ADD COLUMN source_revision INTEGER NOT NULL DEFAULT 0 CHECK(source_revision >= 0);

ALTER TABLE discourse_graph_revisions
ADD COLUMN graph_content_revision INTEGER NOT NULL DEFAULT 0
CHECK(graph_content_revision >= 0);

ALTER TABLE discourse_graph_revisions
ADD COLUMN source_content_sha256 TEXT NOT NULL DEFAULT
'0000000000000000000000000000000000000000000000000000000000000000'
CHECK(length(source_content_sha256) = 64
      AND source_content_sha256 NOT GLOB '*[^0-9a-f]*');

ALTER TABLE discourse_graph_revisions
ADD COLUMN graph_content_sha256 TEXT NOT NULL DEFAULT
'0000000000000000000000000000000000000000000000000000000000000000'
CHECK(length(graph_content_sha256) = 64
      AND graph_content_sha256 NOT GLOB '*[^0-9a-f]*');

CREATE TABLE IF NOT EXISTS discourse_artifact_coverage (
    artifact_id          TEXT NOT NULL REFERENCES discourse_artifacts(artifact_id),
    chunk_id             TEXT NOT NULL REFERENCES chunks(chunk_id),
    coverage_kind        TEXT NOT NULL
                         CHECK(coverage_kind IN ('episode', 'discourse')),
    source_revision      INTEGER NOT NULL CHECK(source_revision >= 0),
    chunk_identity_sha256 TEXT NOT NULL
                          CHECK(length(chunk_identity_sha256) = 64
                                AND chunk_identity_sha256 NOT GLOB '*[^0-9a-f]*'),
    status               TEXT NOT NULL
                         CHECK(status IN ('annotated', 'no_output')),
    receipt_sha256       TEXT NOT NULL UNIQUE
                         CHECK(length(receipt_sha256) = 64
                               AND receipt_sha256 NOT GLOB '*[^0-9a-f]*'),
    PRIMARY KEY (artifact_id, chunk_id, coverage_kind, chunk_identity_sha256)
);

CREATE INDEX IF NOT EXISTS idx_discourse_coverage_chunk
ON discourse_artifact_coverage(chunk_id, artifact_id, coverage_kind);

CREATE TABLE IF NOT EXISTS discourse_artifact_coverage_receipts (
    artifact_id      TEXT NOT NULL REFERENCES discourse_artifacts(artifact_id),
    coverage_kind    TEXT NOT NULL
                     CHECK(coverage_kind IN ('episode', 'discourse')),
    source_revision  INTEGER NOT NULL CHECK(source_revision >= 0),
    chunk_count      INTEGER NOT NULL CHECK(chunk_count >= 0),
    coverage_sha256  TEXT NOT NULL
                     CHECK(length(coverage_sha256) = 64
                           AND coverage_sha256 NOT GLOB '*[^0-9a-f]*'),
    turn_coverage_sha256 TEXT NOT NULL
                         CHECK(length(turn_coverage_sha256) = 64
                               AND turn_coverage_sha256 NOT GLOB '*[^0-9a-f]*'),
    receipt_sha256   TEXT NOT NULL UNIQUE
                     CHECK(length(receipt_sha256) = 64
                           AND receipt_sha256 NOT GLOB '*[^0-9a-f]*'),
    PRIMARY KEY (artifact_id, coverage_kind, source_revision)
);

CREATE TABLE IF NOT EXISTS discourse_revision_state (
    singleton              INTEGER PRIMARY KEY CHECK(singleton = 1),
    source_revision        INTEGER NOT NULL DEFAULT 0 CHECK(source_revision >= 0),
    graph_content_revision INTEGER NOT NULL DEFAULT 0
                           CHECK(graph_content_revision >= 0)
);

INSERT OR IGNORE INTO discourse_revision_state
(singleton, source_revision, graph_content_revision) VALUES (1, 0, 0);

CREATE TRIGGER IF NOT EXISTS trg_discourse_state_no_delete
BEFORE DELETE ON discourse_revision_state
BEGIN
    SELECT RAISE(ABORT, 'discourse revision state is immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_discourse_source_revision_monotonic
BEFORE UPDATE OF source_revision ON discourse_revision_state
WHEN NEW.source_revision <= OLD.source_revision
BEGIN
    SELECT RAISE(ABORT, 'source revision must increase monotonically');
END;

CREATE TRIGGER IF NOT EXISTS trg_discourse_graph_content_revision_monotonic
BEFORE UPDATE OF graph_content_revision ON discourse_revision_state
WHEN NEW.graph_content_revision <= OLD.graph_content_revision
BEGIN
    SELECT RAISE(ABORT, 'graph content revision must increase monotonically');
END;

CREATE TRIGGER IF NOT EXISTS trg_turns_source_insert
AFTER INSERT ON turns
BEGIN
    UPDATE discourse_revision_state
    SET source_revision = source_revision + 1 WHERE singleton = 1;
END;

CREATE TRIGGER IF NOT EXISTS trg_turns_source_delete
AFTER DELETE ON turns
BEGIN
    UPDATE discourse_revision_state
    SET source_revision = source_revision + 1 WHERE singleton = 1;
END;

CREATE TRIGGER IF NOT EXISTS trg_turns_source_update
AFTER UPDATE OF turn_id, role, text, source_id, created_at, ordinal ON turns
WHEN OLD.turn_id IS NOT NEW.turn_id
  OR OLD.role IS NOT NEW.role
  OR OLD.text IS NOT NEW.text
  OR OLD.source_id IS NOT NEW.source_id
  OR OLD.created_at IS NOT NEW.created_at
  OR OLD.ordinal IS NOT NEW.ordinal
BEGIN
    UPDATE discourse_revision_state
    SET source_revision = source_revision + 1 WHERE singleton = 1;
END;

CREATE TRIGGER IF NOT EXISTS trg_chunks_source_insert
AFTER INSERT ON chunks
BEGIN
    UPDATE discourse_revision_state
    SET source_revision = source_revision + 1 WHERE singleton = 1;
END;

CREATE TRIGGER IF NOT EXISTS trg_chunks_source_delete
AFTER DELETE ON chunks
BEGIN
    UPDATE discourse_revision_state
    SET source_revision = source_revision + 1 WHERE singleton = 1;
END;

CREATE TRIGGER IF NOT EXISTS trg_chunks_source_update
AFTER UPDATE OF chunk_id, turn_id, text, start_char, end_char, token_count ON chunks
WHEN OLD.chunk_id IS NOT NEW.chunk_id
  OR OLD.turn_id IS NOT NEW.turn_id
  OR OLD.text IS NOT NEW.text
  OR OLD.start_char IS NOT NEW.start_char
  OR OLD.end_char IS NOT NEW.end_char
  OR OLD.token_count IS NOT NEW.token_count
BEGIN
    UPDATE discourse_revision_state
    SET source_revision = source_revision + 1 WHERE singleton = 1;
END;
""" + _immutable_discourse_triggers() + _graph_content_revision_triggers()

#: Full schema for a freshly created database (already at CURRENT_SCHEMA_VERSION).
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS turns (
    turn_id    TEXT PRIMARY KEY,
    role       TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    text       TEXT NOT NULL,
    source_id  TEXT,
    created_at TEXT NOT NULL,
    ordinal    INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_turns_created ON turns(created_at);
CREATE INDEX IF NOT EXISTS idx_turns_ordinal ON turns(ordinal);
CREATE INDEX IF NOT EXISTS idx_turns_source ON turns(source_id, ordinal);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,
    turn_id         TEXT NOT NULL REFERENCES turns(turn_id),
    text            TEXT NOT NULL,
    start_char      INTEGER NOT NULL,
    end_char        INTEGER NOT NULL,
    token_count     INTEGER NOT NULL,
    embedding       BLOB,
    lexical_weights TEXT,
    hnsw_label      INTEGER UNIQUE,
    term_count      INTEGER
);

CREATE INDEX IF NOT EXISTS idx_chunks_turn ON chunks(turn_id);

-- Inverted index backing BM25 lexical retrieval.
CREATE TABLE IF NOT EXISTS chunk_terms (
    term     TEXT NOT NULL,
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id),
    tf       INTEGER NOT NULL,
    PRIMARY KEY (term, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_chunk_terms_chunk ON chunk_terms(chunk_id);

CREATE TABLE IF NOT EXISTS memory_items (
    mem_id           TEXT PRIMARY KEY,
    type             TEXT NOT NULL,
    content          TEXT NOT NULL,
    details          TEXT,
    status           TEXT NOT NULL DEFAULT 'active'
                     CHECK(status IN ('active', 'superseded', 'deleted')),
    supersedes       TEXT,
    pin              TEXT NOT NULL DEFAULT 'none'
                     CHECK(pin IN ('user_pinned', 'system_pinned', 'none')),
    energy           REAL NOT NULL DEFAULT 0.5,
    half_life_s      REAL NOT NULL DEFAULT 604800.0,
    half_life_turns  REAL NOT NULL DEFAULT 30.0,
    importance       REAL NOT NULL DEFAULT 0.5,
    created_at       TEXT NOT NULL,
    last_access_at   TEXT NOT NULL,
    last_access_turn INTEGER NOT NULL DEFAULT 0,
    embedding        BLOB,
    content_hash     TEXT
);

CREATE INDEX IF NOT EXISTS idx_memory_status ON memory_items(status);
CREATE INDEX IF NOT EXISTS idx_memory_supersedes ON memory_items(supersedes);

-- Mandatory provenance: every memory item points back at the transcript.
CREATE TABLE IF NOT EXISTS memory_provenance (
    mem_id   TEXT NOT NULL REFERENCES memory_items(mem_id) ON DELETE CASCADE,
    turn_id  TEXT NOT NULL,
    chunk_id TEXT,
    quote    TEXT NOT NULL,
    UNIQUE (mem_id, turn_id, quote)
);

CREATE INDEX IF NOT EXISTS idx_provenance_mem ON memory_provenance(mem_id);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_SCHEMA_SQL = (
    _SCHEMA_SQL
    + _V3_INDEXES
    + _V5_ASSOCIATION_SCHEMA
    + _V7_HEBBIAN_SCHEMA
    + _V8_CONSOLIDATION_SCHEMA
    + _V9_CAUSAL_BINDING_SCHEMA
    + _V10_DISCOURSE_SCHEMA
    + _V11_REVISION_SCHEMA
    + f"\nINSERT OR REPLACE INTO meta (key, value)"
    f" VALUES ('schema_version', '{CURRENT_SCHEMA_VERSION}');\n"
)

#: Statements that upgrade a database *into* the keyed version.
_MIGRATIONS: dict[int, str] = {
    2: """
ALTER TABLE chunks ADD COLUMN term_count INTEGER;

CREATE TABLE IF NOT EXISTS chunk_terms (
    term     TEXT NOT NULL,
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id),
    tf       INTEGER NOT NULL,
    PRIMARY KEY (term, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_chunk_terms_chunk ON chunk_terms(chunk_id);

CREATE TABLE IF NOT EXISTS memory_items (
    mem_id         TEXT PRIMARY KEY,
    type           TEXT NOT NULL,
    content        TEXT NOT NULL,
    details        TEXT,
    status         TEXT NOT NULL DEFAULT 'active'
                   CHECK(status IN ('active', 'superseded', 'deleted')),
    supersedes     TEXT,
    pin            TEXT NOT NULL DEFAULT 'none'
                   CHECK(pin IN ('user_pinned', 'system_pinned', 'none')),
    energy         REAL NOT NULL DEFAULT 0.5,
    half_life_s    REAL NOT NULL DEFAULT 604800.0,
    importance     REAL NOT NULL DEFAULT 0.5,
    created_at     TEXT NOT NULL,
    last_access_at TEXT NOT NULL,
    embedding      BLOB
);

CREATE INDEX IF NOT EXISTS idx_memory_status ON memory_items(status);
CREATE INDEX IF NOT EXISTS idx_memory_supersedes ON memory_items(supersedes);

CREATE TABLE IF NOT EXISTS memory_provenance (
    mem_id   TEXT NOT NULL REFERENCES memory_items(mem_id) ON DELETE CASCADE,
    turn_id  TEXT NOT NULL,
    chunk_id TEXT,
    quote    TEXT NOT NULL,
    UNIQUE (mem_id, turn_id, quote)
);

CREATE INDEX IF NOT EXISTS idx_provenance_mem ON memory_provenance(mem_id);

UPDATE meta SET value = '2' WHERE key = 'schema_version';
""",
    3: """
ALTER TABLE memory_items ADD COLUMN content_hash TEXT;
"""
    + _V3_INDEXES
    + """
UPDATE meta SET value = '3' WHERE key = 'schema_version';
""",
    # v4 moves the decay coordinate from wall-clock seconds to conversation
    # turns. Purely additive: `half_life_s` and `last_access_at` stay, because
    # dropping a column in SQLite needs a 12-step table rebuild that clause 10
    # of MC-STD-DATA forbids. `last_access_at` keeps a live purpose as the audit
    # timestamp; `half_life_s` is inert and retained only so a v3 store stays
    # readable by v3 code.
    4: """
ALTER TABLE memory_items ADD COLUMN half_life_turns REAL NOT NULL DEFAULT 30.0;
ALTER TABLE memory_items ADD COLUMN last_access_turn INTEGER NOT NULL DEFAULT 0;
ALTER TABLE turns ADD COLUMN ordinal INTEGER NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_turns_ordinal ON turns(ordinal);

UPDATE meta SET value = '4' WHERE key = 'schema_version';
""",
    5: _V5_ASSOCIATION_SCHEMA
    + """
UPDATE meta SET value = '5' WHERE key = 'schema_version';
""",
    6: """
ALTER TABLE turns ADD COLUMN source_id TEXT;

CREATE INDEX IF NOT EXISTS idx_turns_source ON turns(source_id, ordinal);

UPDATE meta SET value = '6' WHERE key = 'schema_version';
""",
    7: _V7_HEBBIAN_SCHEMA
    + """
UPDATE meta SET value = '7' WHERE key = 'schema_version';
""",
    8: _V8_CONSOLIDATION_SCHEMA
    + """
UPDATE meta SET value = '8' WHERE key = 'schema_version';
""",
    # A completed prompt/response episode is stronger evidence than an
    # incidental retrieved-together event. Keep that provenance as one scalar
    # counter so unique outcomes can be recalled without lowering the repeated
    # co-access guard for every edge in the graph.
    9: _V9_CAUSAL_BINDING_SCHEMA
    + """
UPDATE meta SET value = '9' WHERE key = 'schema_version';
""",
    10: _V10_DISCOURSE_SCHEMA
    + """
UPDATE meta SET value = '10' WHERE key = 'schema_version';
""",
    11: _V11_REVISION_SCHEMA
    + """
UPDATE meta SET value = '11' WHERE key = 'schema_version';
""",
}


def _backfill_content_hash(conn: sqlite3.Connection) -> None:
    """Populate ``content_hash`` for rows that predate v3.

    Python rather than SQL because stock SQLite has neither ``sha256`` nor a
    way to collapse internal whitespace runs.
    """
    from memory_condense.domain.schemas import content_key

    rows = conn.execute(
        "SELECT mem_id, type, content FROM memory_items WHERE content_hash IS NULL"
    ).fetchall()
    if not rows:
        return
    conn.executemany(
        "UPDATE memory_items SET content_hash = ? WHERE mem_id = ?",
        [(content_key(mem_type, content), mem_id) for mem_id, mem_type, content in rows],
    )


def _backfill_turn_ordinals(conn: sqlite3.Connection) -> None:
    """Number pre-v4 turns, and enter pre-v4 memories at the current turn.

    Ordinals come from ``rowid`` order, which is insertion order and therefore
    conversation order — the same assumption ``retrieval._span_vectors`` already
    relies on. Numbering starts at 1 so that 0 keeps its meaning of "never
    stamped".

    Existing memory items are backfilled to the **latest** ordinal, not to 0.
    Entering them fresh costs at most one half-life of over-warmth, which the
    next few turns correct. Entering them at 0 would make every memory in a
    long-lived store instantly COLD on upgrade — silent data loss dressed up as
    decay, and the same "immortal beats invisible" argument that
    :func:`decay.decay_factor` makes for a non-positive half-life.
    """
    rows = conn.execute("SELECT turn_id FROM turns ORDER BY rowid").fetchall()
    if rows:
        conn.executemany(
            "UPDATE turns SET ordinal = ? WHERE turn_id = ?",
            [(i, turn_id) for i, (turn_id,) in enumerate(rows, start=1)],
        )
    conn.execute(
        "UPDATE memory_items SET last_access_turn = ?", (len(rows),)
    )


def _backfill_discourse_snapshot_revisions(conn: sqlite3.Connection) -> None:
    """Re-hash immutable pre-v11 graph receipts with baseline revisions."""

    for table in _IMMUTABLE_DISCOURSE_TABLES:
        conn.execute(f"DROP TRIGGER IF EXISTS trg_{table}_no_update")
        conn.execute(f"DROP TRIGGER IF EXISTS trg_{table}_no_delete")
    conn.execute("DROP TRIGGER IF EXISTS trg_discourse_revision_no_update")
    conn.execute("DROP TRIGGER IF EXISTS trg_discourse_revision_no_delete")
    for table in (
        "episode_evidence",
        "discourse_unit_evidence",
        "discourse_relation_evidence",
    ):
        conn.execute(
            f"UPDATE {table} SET "
            "source_id = (SELECT COALESCE(t.source_id, t.turn_id) "
            "FROM chunks AS c JOIN turns AS t ON t.turn_id = c.turn_id "
            f"WHERE c.chunk_id = {table}.chunk_id), "
            "turn_id = (SELECT c.turn_id FROM chunks AS c "
            f"WHERE c.chunk_id = {table}.chunk_id), "
            "role = (SELECT t.role FROM chunks AS c JOIN turns AS t "
            "ON t.turn_id = c.turn_id "
            f"WHERE c.chunk_id = {table}.chunk_id), "
            "created_at = (SELECT t.created_at FROM chunks AS c JOIN turns AS t "
            "ON t.turn_id = c.turn_id "
            f"WHERE c.chunk_id = {table}.chunk_id)"
        )

    from memory_condense.domain.discourse import (
        DiscourseSnapshot,
        Episode,
        EvidenceSpan,
        canonical_json,
    )
    from memory_condense.persistence.discourse_receipts import (
        discourse_content_digests,
    )

    episode_rows = conn.execute(
        "SELECT episode_id, artifact_id, source_id, sequence_no, first_ordinal, "
        "last_ordinal, boundary_method, initial_boundary, refined_boundary, "
        "boundary_score, boundary_threshold FROM episodes"
    ).fetchall()
    episode_receipts: list[tuple[str, str]] = []
    for episode_row in episode_rows:
        evidence_rows = conn.execute(
            "SELECT chunk_id, start_char, end_char, quote_sha256, ordinal, "
            "source_id, turn_start_char, turn_id, role, created_at "
            "FROM episode_evidence WHERE episode_id = ? ORDER BY evidence_order",
            (episode_row[0],),
        ).fetchall()
        evidence = tuple(
            EvidenceSpan(
                chunk_id=row[0],
                start_char=int(row[1]),
                end_char=int(row[2]),
                quote_sha256=row[3],
                ordinal=int(row[4]),
                source_id=row[5],
                turn_start_char=int(row[6]),
                turn_id=row[7],
                role=row[8],
                created_at=row[9],
            )
            for row in evidence_rows
        )
        episode = Episode(
            episode_id=episode_row[0],
            artifact_id=episode_row[1],
            source_id=episode_row[2],
            sequence_no=int(episode_row[3]),
            first_ordinal=int(episode_row[4]),
            last_ordinal=int(episode_row[5]),
            evidence=evidence,
            boundary_method=episode_row[6],
            initial_boundary=episode_row[7],
            refined_boundary=episode_row[8],
            boundary_score=episode_row[9],
            boundary_threshold=episode_row[10],
        )
        episode_receipts.append((episode.receipt_sha256, episode.episode_id))
    conn.executemany(
        "UPDATE episodes SET receipt_sha256 = ? WHERE episode_id = ?",
        episode_receipts,
    )
    source_baseline = sum(
        int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in ("turns", "chunks")
    )
    graph_baseline = sum(
        int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in _IMMUTABLE_DISCOURSE_TABLES
    )
    if source_baseline:
        conn.execute(
            "UPDATE discourse_revision_state SET source_revision = ? "
            "WHERE singleton = 1",
            (source_baseline,),
        )
    if graph_baseline:
        conn.execute(
            "UPDATE discourse_revision_state SET graph_content_revision = ? "
            "WHERE singleton = 1",
            (graph_baseline,),
        )
    source_content, graph_content = discourse_content_digests(conn)
    legacy_max_revision = int(
        conn.execute(
            "SELECT COALESCE(MAX(graph_revision), 0) "
            "FROM discourse_graph_revisions"
        ).fetchone()[0]
    )
    # v10 rows do not record when each graph entity was introduced, so their
    # historical contents cannot be reconstructed honestly.  Retire all old
    # receipts and publish one explicit v11 baseline over the current content.
    conn.execute("DELETE FROM discourse_graph_revisions")
    if legacy_max_revision or graph_baseline:
        baseline = DiscourseSnapshot(
            max_turn_ordinal=int(
                conn.execute(
                    "SELECT COALESCE(MAX(ordinal), 0) FROM turns"
                ).fetchone()[0]
            ),
            chunk_count=int(
                conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            ),
            graph_revision=legacy_max_revision + 1,
            schema_version=CURRENT_SCHEMA_VERSION,
            artifact_ids=tuple(
                row[0]
                for row in conn.execute(
                    "SELECT artifact_id FROM discourse_artifacts "
                    "ORDER BY artifact_id"
                ).fetchall()
            ),
            source_revision=source_baseline,
            graph_content_revision=graph_baseline,
            source_content_sha256=source_content,
            graph_content_sha256=graph_content,
        )
        conn.execute(
            "INSERT INTO discourse_graph_revisions "
            "(graph_revision, max_turn_ordinal, chunk_count, schema_version, "
            "artifact_ids, snapshot_sha256, source_revision, "
            "graph_content_revision, source_content_sha256, "
            "graph_content_sha256) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                baseline.graph_revision,
                baseline.max_turn_ordinal,
                baseline.chunk_count,
                baseline.schema_version,
                canonical_json(list(baseline.artifact_ids)),
                baseline.snapshot_sha256,
                baseline.source_revision,
                baseline.graph_content_revision,
                baseline.source_content_sha256,
                baseline.graph_content_sha256,
            ),
        )
    _execute_sql_script(
        conn,
        """
CREATE TRIGGER IF NOT EXISTS trg_discourse_revision_no_update
BEFORE UPDATE ON discourse_graph_revisions
BEGIN
    SELECT RAISE(ABORT, 'discourse graph revision receipts are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_discourse_revision_no_delete
BEFORE DELETE ON discourse_graph_revisions
BEGIN
    SELECT RAISE(ABORT, 'discourse graph revision receipts are immutable');
END;
"""
        + _immutable_discourse_triggers()
    )


#: Work that must run *after* the SQL for a version, when SQL alone cannot
#: express it. Keyed by target version, same as :data:`_MIGRATIONS`.
#:
#: Deliberately no UNIQUE index on ``content_hash``: stores written before v3
#: almost certainly already contain duplicates — that is the bug this column
#: exists to stop — and ``CREATE UNIQUE INDEX`` would raise inside
#: ``Database.__init__``, making an existing store permanently unopenable.
#: Uniqueness is enforced in ``MemoryStore.create``; promoting it to a database
#: constraint is a later version, after ``dedupe_existing()`` has been run.
_POST_MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {
    3: _backfill_content_hash,
    4: _backfill_turn_ordinals,
    11: _backfill_discourse_snapshot_revisions,
}


class Database:
    """Manages the SQLite connection, schema creation, and migrations.

    A brand-new file is created directly at ``CURRENT_SCHEMA_VERSION``. An
    existing file is migrated forward one version at a time. The transcript is
    append-only and all derived state (chunks, terms, memory) is reconstructible
    from it — see ``docs/05 - Standards/00 - MC-STD-DATA-v0.md``.

    ``read_only=True`` opens an existing database through SQLite's ``mode=ro``
    URI and enables ``query_only`` as a second guard. The URI also marks the
    closed cache snapshot immutable so SQLite does not create WAL/SHM sidecars
    merely to read it. Read-only connections do not create parent directories,
    initialize an empty schema, or run migrations. This makes immutable
    evaluation caches enforce their contract at the storage boundary while
    preserving the writable default for live memory.
    """

    def __init__(
        self,
        db_path: str | Path = "memory.db",
        *,
        read_only: bool = False,
    ) -> None:
        self._path = Path(db_path)
        self._read_only = bool(read_only)
        if self._read_only:
            connection_target = (
                f"{self._path.resolve().as_uri()}?mode=ro&immutable=1"
            )
        else:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection_target = str(self._path)
        self._conn = sqlite3.connect(
            connection_target,
            check_same_thread=False,
            uri=self._read_only,
        )
        self._conn.execute("PRAGMA foreign_keys=ON")
        if self._read_only:
            self._conn.execute("PRAGMA query_only=ON")
        else:
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._init_schema()
            except BaseException:
                self._conn.close()
                raise

    def _init_schema(self) -> None:
        version = self._read_version()

        if version is None:
            _apply_schema_transaction(self._conn, _SCHEMA_SQL)
        else:
            for target in range(version + 1, CURRENT_SCHEMA_VERSION + 1):
                sql = _MIGRATIONS.get(target)
                if sql is not None:
                    _apply_schema_transaction(
                        self._conn,
                        sql,
                        post=_POST_MIGRATIONS.get(target),
                    )

    def _read_version(self) -> int | None:
        """Current schema version, or None if the database is empty."""
        try:
            cur = self._conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            )
        except sqlite3.OperationalError:
            return None  # meta table does not exist yet — fresh database
        row = cur.fetchone()
        return int(row[0]) if row else None

    @property
    def schema_version(self) -> int:
        version = self._read_version()
        return version if version is not None else 0

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    @property
    def path(self) -> Path:
        """Absolute/relative database path used by independent read workers."""
        return self._path

    @property
    def read_only(self) -> bool:
        """Whether SQLite enforces an immutable connection."""
        return self._read_only

    def current_turn(self) -> int:
        """The conversation's position — **the decay coordinate**.

        Lives here rather than on ``TranscriptStore`` because ``MemoryStore``
        needs it too, and two copies of the query is exactly the drift this
        module's own header warns about. 0 for an empty transcript.

        ``MAX(ordinal)``, not ``COUNT(*)``: the transcript is append-only
        today, but a count would silently renumber the clock backwards if a row
        ever went missing, aging every memory item at once.
        """
        cur = self.execute("SELECT COALESCE(MAX(ordinal), 0) FROM turns")
        return int(cur.fetchone()[0])

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, params)

    def executemany(self, sql: str, params_seq) -> sqlite3.Cursor:
        return self._conn.executemany(sql, params_seq)

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *args) -> None:
        self.close()
