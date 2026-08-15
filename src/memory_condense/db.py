from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable

CURRENT_SCHEMA_VERSION = 3

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

#: Full schema for a freshly created database (already at CURRENT_SCHEMA_VERSION).
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS turns (
    turn_id    TEXT PRIMARY KEY,
    role       TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    text       TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_turns_created ON turns(created_at);

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
    embedding      BLOB,
    content_hash   TEXT
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
}


def _backfill_content_hash(conn: sqlite3.Connection) -> None:
    """Populate ``content_hash`` for rows that predate v3.

    Python rather than SQL because stock SQLite has neither ``sha256`` nor a
    way to collapse internal whitespace runs.
    """
    from memory_condense.schemas import content_key

    rows = conn.execute(
        "SELECT mem_id, type, content FROM memory_items WHERE content_hash IS NULL"
    ).fetchall()
    if not rows:
        return
    conn.executemany(
        "UPDATE memory_items SET content_hash = ? WHERE mem_id = ?",
        [(content_key(mem_type, content), mem_id) for mem_id, mem_type, content in rows],
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
}


class Database:
    """Manages the SQLite connection, schema creation, and migrations.

    A brand-new file is created directly at ``CURRENT_SCHEMA_VERSION``. An
    existing file is migrated forward one version at a time. The transcript is
    append-only and all derived state (chunks, terms, memory) is reconstructible
    from it — see ``docs/05 - Standards/00 - MC-STD-DATA-v0.md``.
    """

    def __init__(self, db_path: str | Path = "memory.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._path),
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        version = self._read_version()

        if version is None:
            self._conn.executescript(_SCHEMA_SQL)
        else:
            for target in range(version + 1, CURRENT_SCHEMA_VERSION + 1):
                sql = _MIGRATIONS.get(target)
                if sql is not None:
                    self._conn.executescript(sql)
                post = _POST_MIGRATIONS.get(target)
                if post is not None:
                    post(self._conn)

        self._conn.commit()

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
