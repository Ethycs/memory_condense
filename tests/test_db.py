from __future__ import annotations

import sqlite3

import pytest

from memory_condense.persistence.db import CURRENT_SCHEMA_VERSION, Database

# The v1 schema exactly as it shipped in cd9f423, used to build a legacy
# database and prove the migration path works on real pre-existing files.
#
# The two indexes matter: this fixture used to omit them, which was harmless
# while nothing compared schemas but would make the fresh-vs-migrated parity
# test below report drift that does not exist. Verify against
# `git show cd9f423:src/memory_condense/db.py` before editing.
_V1_SCHEMA = """
CREATE TABLE turns (
    turn_id    TEXT PRIMARY KEY,
    role       TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    text       TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX idx_turns_created ON turns(created_at);
CREATE TABLE chunks (
    chunk_id        TEXT PRIMARY KEY,
    turn_id         TEXT NOT NULL REFERENCES turns(turn_id),
    text            TEXT NOT NULL,
    start_char      INTEGER NOT NULL,
    end_char        INTEGER NOT NULL,
    token_count     INTEGER NOT NULL,
    embedding       BLOB,
    lexical_weights TEXT,
    hnsw_label      INTEGER UNIQUE
);
CREATE INDEX idx_chunks_turn ON chunks(turn_id);
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT INTO meta (key, value) VALUES ('schema_version', '1');
"""


def _table_names(db: Database) -> set[str]:
    cur = db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return {row[0] for row in cur.fetchall()}


def _column_names(db: Database, table: str) -> set[str]:
    cur = db.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


class TestFreshDatabase:
    def test_created_at_current_version(self, tmp_path):
        with Database(tmp_path / "fresh.db") as db:
            assert db.schema_version == CURRENT_SCHEMA_VERSION

    def test_all_tables_present(self, tmp_path):
        with Database(tmp_path / "fresh.db") as db:
            tables = _table_names(db)
        assert {
            "turns",
            "chunks",
            "chunk_terms",
            "memory_items",
            "memory_provenance",
            "memory_successor_redirects",
            "pending_ingests",
            "ingest_chunk_reservations",
            "association_artifacts",
            "chunk_cav_signatures",
            "chunk_head_edges",
            "hebbian_access_events",
            "hebbian_chunk_nodes",
            "hebbian_chunk_edges",
            "consolidation_access_events",
            "consolidation_nodes",
            "consolidation_edges",
            "discourse_artifacts",
            "episodes",
            "episode_evidence",
            "episode_representatives",
            "discourse_units",
            "discourse_unit_evidence",
            "discourse_relations",
            "discourse_relation_members",
            "discourse_relation_evidence",
            "discourse_graph_revisions",
            "meta",
        } <= tables

    def test_chunks_has_term_count(self, tmp_path):
        with Database(tmp_path / "fresh.db") as db:
            assert "term_count" in _column_names(db, "chunks")

    def test_turns_have_source_identity(self, tmp_path):
        with Database(tmp_path / "source.db") as db:
            assert "source_id" in _column_names(db, "turns")
            assert "idx_turns_source" in {
                row[1] for row in db.execute("PRAGMA index_list(turns)").fetchall()
            }

    def test_turn_ordinal_is_a_unique_global_clock(self, tmp_path):
        with Database(tmp_path / "ordinal.db") as db:
            db.execute(
                "INSERT INTO turns "
                "(turn_id, role, text, created_at, ordinal) "
                "VALUES ('t1', 'user', 'one', '2026-08-18', 1)"
            )
            with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
                db.execute(
                    "INSERT INTO turns "
                    "(turn_id, role, text, created_at, ordinal) "
                    "VALUES ('t2', 'user', 'two', '2026-08-18', 1)"
                )

    def test_consolidation_edges_distinguish_causal_binding(self, tmp_path):
        with Database(tmp_path / "causal.db") as db:
            assert "causal_count" in _column_names(db, "consolidation_edges")

    def test_reopening_is_idempotent(self, tmp_path):
        path = tmp_path / "reopen.db"
        with Database(path) as db:
            # Columns named explicitly: a bare VALUES list breaks on every
            # additive migration, which is the normal kind.
            db.execute(
                "INSERT INTO turns (turn_id, role, text, created_at, ordinal)"
                " VALUES ('t1', 'user', 'hello', '2026-01-01T00:00:00+00:00', 1)"
            )
            db.commit()
        with Database(path) as db:
            assert db.schema_version == CURRENT_SCHEMA_VERSION
            assert db.execute("SELECT COUNT(*) FROM turns").fetchone()[0] == 1


class TestReadOnlyDatabase:
    def test_reads_existing_database_and_rejects_writes(self, tmp_path):
        path = tmp_path / "read-only.db"
        with Database(path) as db:
            db.execute(
                "INSERT INTO turns (turn_id, role, text, created_at, ordinal)"
                " VALUES ('t1', 'user', 'hello', '2026-01-01T00:00:00+00:00', 1)"
            )
            db.commit()

        before = path.read_bytes()
        with Database(path, read_only=True) as db:
            assert db.read_only is True
            assert db.execute("PRAGMA query_only").fetchone()[0] == 1
            assert db.execute("SELECT text FROM turns").fetchone()[0] == "hello"
            with pytest.raises(sqlite3.OperationalError):
                db.execute("UPDATE turns SET text = 'changed' WHERE turn_id = 't1'")

        assert path.read_bytes() == before
        assert not path.with_name(f"{path.name}-wal").exists()
        assert not path.with_name(f"{path.name}-shm").exists()

    def test_does_not_create_a_missing_database_or_parent(self, tmp_path):
        path = tmp_path / "missing-parent" / "missing.db"

        with pytest.raises(sqlite3.OperationalError):
            Database(path, read_only=True)

        assert not path.exists()
        assert not path.parent.exists()

    def test_does_not_migrate_an_older_schema(self, tmp_path):
        path = tmp_path / "legacy-read-only.db"
        connection = sqlite3.connect(path)
        connection.executescript(_V1_SCHEMA)
        connection.commit()
        connection.close()
        before = path.read_bytes()

        with Database(path, read_only=True) as db:
            assert db.schema_version == 1
            assert "ordinal" not in _column_names(db, "turns")

        assert path.read_bytes() == before


class TestMigrationFromV1:
    @pytest.fixture
    def v1_db_path(self, tmp_path):
        path = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(path))
        conn.executescript(_V1_SCHEMA)
        conn.execute(
            "INSERT INTO turns VALUES ('t1', 'user', 'legacy turn', '2026-01-01T00:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO chunks (chunk_id, turn_id, text, start_char, end_char, token_count) "
            "VALUES ('c1', 't1', 'legacy chunk', 0, 12, 3)"
        )
        conn.commit()
        conn.close()
        return path

    def test_version_bumped(self, v1_db_path):
        with Database(v1_db_path) as db:
            assert db.schema_version == CURRENT_SCHEMA_VERSION

    def test_existing_rows_survive(self, v1_db_path):
        with Database(v1_db_path) as db:
            assert db.execute("SELECT text FROM turns").fetchone()[0] == "legacy turn"
            assert db.execute("SELECT text FROM chunks").fetchone()[0] == "legacy chunk"

    def test_new_tables_added(self, v1_db_path):
        with Database(v1_db_path) as db:
            tables = _table_names(db)
        assert {
            "chunk_terms",
            "memory_items",
            "memory_provenance",
            "memory_successor_redirects",
            "pending_ingests",
            "ingest_chunk_reservations",
        } <= tables

    def test_term_count_column_added(self, v1_db_path):
        with Database(v1_db_path) as db:
            assert "term_count" in _column_names(db, "chunks")

    def test_migration_runs_only_once(self, v1_db_path):
        with Database(v1_db_path):
            pass
        # A second open must not re-run the ALTER TABLE (which would error).
        with Database(v1_db_path) as db:
            assert db.schema_version == CURRENT_SCHEMA_VERSION

    def test_content_hash_column_added(self, v1_db_path):
        with Database(v1_db_path) as db:
            assert "content_hash" in _column_names(db, "memory_items")

    def test_source_id_column_added(self, v1_db_path):
        with Database(v1_db_path) as db:
            assert "source_id" in _column_names(db, "turns")


class TestSchemaParity:
    """`_SCHEMA_SQL` and `_MIGRATIONS` are two hand-maintained copies of one
    schema. The standard's own verification block warns that they drift. These
    tests make drift a test failure rather than a support ticket.

    Compares columns, indexes, and triggers rather than raw `sqlite_master.sql` text:
    `ALTER TABLE ADD COLUMN` and `CREATE TABLE` produce different DDL text for
    the same logical column, so a text comparison would fail on every additive
    migration and teach everyone to ignore it.
    """

    @staticmethod
    def _shape(db: Database) -> dict:
        tables = sorted(_table_names(db))
        return {
            "tables": tables,
            "columns": {t: sorted(_column_names(db, t)) for t in tables},
            "indexes": sorted(
                row[0]
                for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' "
                    "AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            ),
            "triggers": sorted(
                row[0]
                for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger'"
                ).fetchall()
            ),
        }

    def _migrated(self, tmp_path, schema_sql: str, name: str) -> dict:
        path = tmp_path / name
        conn = sqlite3.connect(str(path))
        conn.executescript(schema_sql)
        conn.commit()
        conn.close()
        with Database(path) as db:
            assert db.schema_version == CURRENT_SCHEMA_VERSION
            return self._shape(db)

    def test_fresh_matches_migrated_from_v1(self, tmp_path):
        with Database(tmp_path / "fresh.db") as db:
            fresh = self._shape(db)
        assert fresh == self._migrated(tmp_path, _V1_SCHEMA, "from_v1.db")

    def test_fresh_matches_migrated_from_v2(self, tmp_path):
        """v2 is what a store written before this change actually looks like."""
        from memory_condense.persistence.db import _MIGRATIONS

        v2_sql = (
            _V1_SCHEMA.replace(
                "INSERT INTO meta (key, value) VALUES ('schema_version', '1');",
                "",
            )
            + _MIGRATIONS[2]
            + "\nINSERT INTO meta (key, value) VALUES ('schema_version', '2');\n"
        )
        with Database(tmp_path / "fresh2.db") as db:
            fresh = self._shape(db)
        assert fresh == self._migrated(tmp_path, v2_sql, "from_v2.db")

    def test_fresh_matches_migrated_from_v9(self, tmp_path):
        """The immediately preceding production schema reaches exact v10 shape."""
        from memory_condense.persistence.db import _MIGRATIONS

        path = tmp_path / "from_v9.db"
        conn = sqlite3.connect(str(path))
        conn.executescript(_V1_SCHEMA)
        for target in range(2, 10):
            conn.executescript(_MIGRATIONS[target])
            conn.execute(
                "UPDATE meta SET value = ? WHERE key = 'schema_version'",
                (str(target),),
            )
        assert conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()[0] == "9"
        conn.commit()
        conn.close()
        with Database(tmp_path / "fresh9.db") as db:
            fresh = self._shape(db)
        with Database(path) as db:
            assert db.schema_version == CURRENT_SCHEMA_VERSION
            assert self._shape(db) == fresh

    def test_fresh_matches_migrated_from_v12(self, tmp_path):
        """The immediately preceding schema receives the pending journal."""
        from memory_condense.persistence.db import _MIGRATIONS

        path = tmp_path / "from_v12.db"
        conn = sqlite3.connect(str(path))
        conn.executescript(_V1_SCHEMA)
        for target in range(2, 13):
            conn.executescript(_MIGRATIONS[target])
            conn.execute(
                "UPDATE meta SET value = ? WHERE key = 'schema_version'",
                (str(target),),
            )
        conn.commit()
        conn.close()

        with Database(tmp_path / "fresh12.db") as db:
            fresh = self._shape(db)
        with Database(path) as db:
            assert db.schema_version == CURRENT_SCHEMA_VERSION
            assert self._shape(db) == fresh
            assert db.execute("SELECT COUNT(*) FROM pending_ingests").fetchone()[
                0
            ] == 0
            assert db.execute(
                "SELECT COUNT(*) FROM ingest_chunk_reservations"
            ).fetchone()[0] == 0


def test_v10_historical_graph_receipts_are_retired_to_one_v11_baseline(tmp_path):
    from memory_condense.persistence.db import _MIGRATIONS
    from memory_condense.persistence.discourse_store import DiscourseStore

    path = tmp_path / "two-v10-publications.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(_V1_SCHEMA)
    for target in range(2, 11):
        conn.executescript(_MIGRATIONS[target])
        conn.execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'",
            (str(target),),
        )
    conn.execute(
        "INSERT INTO turns "
        "(turn_id, role, text, source_id, created_at, ordinal) "
        "VALUES ('t1', 'user', 'source', 'thread', '2026-08-18', 1)"
    )
    conn.execute(
        "INSERT INTO chunks "
        "(chunk_id, turn_id, text, start_char, end_char, token_count) "
        "VALUES ('c1', 't1', 'source', 0, 6, 1)"
    )
    for index in (1, 2):
        artifact_id = f"artifact-{index}"
        conn.execute(
            "INSERT INTO discourse_artifacts "
            "(artifact_id, kind, implementation_sha256, policy_sha256, metadata) "
            "VALUES (?, 'fixture', ?, ?, '{}')",
            (artifact_id, "a" * 64, "b" * 64),
        )
        conn.execute(
            "INSERT INTO discourse_graph_revisions "
            "(graph_revision, max_turn_ordinal, chunk_count, schema_version, "
            "artifact_ids, snapshot_sha256) VALUES (?, 1, 1, 10, ?, ?)",
            (
                index,
                '["artifact-1"]'
                if index == 1
                else '["artifact-1","artifact-2"]',
                str(index) * 64,
            ),
        )
    conn.commit()
    conn.close()

    with Database(path) as db:
        store = DiscourseStore(db)
        assert db.execute(
            "SELECT graph_revision FROM discourse_graph_revisions"
        ).fetchall() == [(3,)]
        with pytest.raises(KeyError, match="unknown discourse graph revision"):
            store.snapshot(1)
        with pytest.raises(KeyError, match="unknown discourse graph revision"):
            store.snapshot(2)
        baseline = store.snapshot(3)
        assert baseline.schema_version == 11
        assert baseline.artifact_ids == ("artifact-1", "artifact-2")
        assert baseline.source_content_sha256 != "0" * 64
        assert baseline.graph_content_sha256 != "0" * 64


@pytest.mark.parametrize(
    ("target", "table", "column"),
    (
        (3, "memory_items", "content_hash"),
        (4, "turns", "ordinal"),
        (11, "episode_evidence", "turn_id"),
    ),
)
def test_failed_post_migration_rolls_back_ddl_and_version_then_reopens(
    tmp_path,
    monkeypatch,
    target,
    table,
    column,
):
    from memory_condense.persistence import db as db_module

    path = tmp_path / f"failed-post-v{target}.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(_V1_SCHEMA)
    for version in range(2, target):
        conn.executescript(db_module._MIGRATIONS[version])
        conn.execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'",
            (str(version),),
        )
    conn.commit()
    conn.close()

    original = db_module._POST_MIGRATIONS[target]

    def fail_post(_conn):
        original(_conn)
        raise RuntimeError(f"forced v{target} post failure")

    monkeypatch.setitem(db_module._POST_MIGRATIONS, target, fail_post)
    with pytest.raises(RuntimeError, match=f"forced v{target} post failure"):
        Database(path)

    raw = sqlite3.connect(str(path))
    try:
        assert raw.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()[0] == str(target - 1)
        assert column not in {
            row[1] for row in raw.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if target == 11:
            assert raw.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'discourse_revision_state'"
            ).fetchone() is None
    finally:
        raw.close()

    monkeypatch.setitem(db_module._POST_MIGRATIONS, target, original)
    with Database(path) as db:
        assert db.schema_version == CURRENT_SCHEMA_VERSION
        assert column in _column_names(db, table)


class TestV4TurnCoordinateBackfill:
    """v4 moves decay from wall-clock seconds to conversation turns.

    Shape parity is covered above; what it cannot catch is whether the
    *values* the backfill writes are sane. Getting this wrong is silent data
    loss dressed up as decay: a store upgraded with every item stamped at turn
    0 would report its entire memory as COLD the moment it reopened.
    """

    @pytest.fixture
    def v3_db_path(self, tmp_path):
        from memory_condense.persistence.db import _MIGRATIONS

        path = tmp_path / "v3.db"
        sql = (
            _V1_SCHEMA.replace(
                "INSERT INTO meta (key, value) VALUES ('schema_version', '1');", ""
            )
            + _MIGRATIONS[2]
            + _MIGRATIONS[3]
            + "\nINSERT INTO meta (key, value) VALUES ('schema_version', '3');\n"
        )
        conn = sqlite3.connect(str(path))
        conn.executescript(sql)
        for i in range(5):
            conn.execute(
                "INSERT INTO turns (turn_id, role, text, created_at)"
                " VALUES (?, 'user', ?, ?)",
                (f"t{i}", f"turn {i}", f"2026-01-0{i + 1}T00:00:00+00:00"),
            )
        conn.execute(
            "INSERT INTO memory_items (mem_id, type, content, energy, importance,"
            " created_at, last_access_at) VALUES"
            " ('m1', 'Decision', 'use SQLite', 0.8, 0.9,"
            " '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')"
        )
        conn.commit()
        conn.close()
        return path

    def test_turns_are_numbered_in_insertion_order(self, v3_db_path):
        with Database(v3_db_path) as db:
            rows = db.execute(
                "SELECT turn_id, ordinal FROM turns ORDER BY ordinal"
            ).fetchall()
        assert rows == [(f"t{i}", i + 1) for i in range(5)]

    def test_ordinals_start_at_one_so_zero_still_means_never_stamped(
        self, v3_db_path
    ):
        with Database(v3_db_path) as db:
            assert db.execute("SELECT MIN(ordinal) FROM turns").fetchone()[0] == 1

    def test_current_turn_reads_the_backfilled_clock(self, v3_db_path):
        with Database(v3_db_path) as db:
            assert db.current_turn() == 5

    def test_existing_memories_enter_fresh_not_cold(self, v3_db_path):
        """The whole point of backfilling to the latest turn rather than 0."""
        from memory_condense.domain import decay
        from memory_condense.persistence.memory_store import MemoryStore

        with Database(v3_db_path) as db:
            item = MemoryStore(db).get("m1")
            assert item.last_access_turn == 5
            assert decay.item_heat(item, now_turn=db.current_turn()) is not (
                decay.Heat.COLD
            )

    def test_an_empty_transcript_leaves_the_clock_at_zero(self, tmp_path):
        with Database(tmp_path / "empty.db") as db:
            assert db.current_turn() == 0

    def test_appending_advances_the_clock(self, tmp_path):
        from memory_condense.persistence.transcript_store import TranscriptStore

        with Database(tmp_path / "advance.db") as db:
            store = TranscriptStore(db)
            for i in range(3):
                store.append("user", f"hello {i}")
            assert db.current_turn() == 3

    def test_the_clock_is_max_not_count(self, tmp_path):
        """A count would renumber backwards if a row ever went missing, aging
        every memory item at once. MAX only ever moves forward."""
        from memory_condense.persistence.transcript_store import TranscriptStore

        with Database(tmp_path / "gap.db") as db:
            store = TranscriptStore(db)
            for i in range(4):
                store.append("user", f"hello {i}")
            db.execute("DELETE FROM turns WHERE ordinal = 2")
            db.commit()
            assert db.current_turn() == 4


class TestConstraints:
    def test_memory_status_is_constrained(self, tmp_path):
        with Database(tmp_path / "c.db") as db:
            with pytest.raises(sqlite3.IntegrityError):
                db.execute(
                    "INSERT INTO memory_items "
                    "(mem_id, type, content, status, created_at, last_access_at) "
                    "VALUES ('m1', 'Decision', 'x', 'bogus', '2026-01-01', '2026-01-01')"
                )

    def test_pin_state_is_constrained(self, tmp_path):
        with Database(tmp_path / "c.db") as db:
            with pytest.raises(sqlite3.IntegrityError):
                db.execute(
                    "INSERT INTO memory_items "
                    "(mem_id, type, content, pin, created_at, last_access_at) "
                    "VALUES ('m1', 'Decision', 'x', 'bogus', '2026-01-01', '2026-01-01')"
                )

    def test_foreign_keys_enforced(self, tmp_path):
        with Database(tmp_path / "c.db") as db:
            with pytest.raises(sqlite3.IntegrityError):
                db.execute(
                    "INSERT INTO chunks "
                    "(chunk_id, turn_id, text, start_char, end_char, token_count) "
                    "VALUES ('c1', 'missing-turn', 'x', 0, 1, 1)"
                )
