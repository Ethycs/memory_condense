from __future__ import annotations

import sqlite3

import pytest

from memory_condense.db import CURRENT_SCHEMA_VERSION, Database

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
            "meta",
        } <= tables

    def test_chunks_has_term_count(self, tmp_path):
        with Database(tmp_path / "fresh.db") as db:
            assert "term_count" in _column_names(db, "chunks")

    def test_reopening_is_idempotent(self, tmp_path):
        path = tmp_path / "reopen.db"
        with Database(path) as db:
            db.execute(
                "INSERT INTO turns VALUES ('t1', 'user', 'hello', '2026-01-01T00:00:00+00:00')"
            )
            db.commit()
        with Database(path) as db:
            assert db.schema_version == CURRENT_SCHEMA_VERSION
            assert db.execute("SELECT COUNT(*) FROM turns").fetchone()[0] == 1


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
        assert {"chunk_terms", "memory_items", "memory_provenance"} <= tables

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


class TestSchemaParity:
    """`_SCHEMA_SQL` and `_MIGRATIONS` are two hand-maintained copies of one
    schema. The standard's own verification block warns that they drift. These
    tests make drift a test failure rather than a support ticket.

    Compares columns and indexes rather than raw `sqlite_master.sql` text:
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
        from memory_condense.db import _MIGRATIONS

        v2_sql = _V1_SCHEMA.replace(
            "INSERT INTO meta (key, value) VALUES ('schema_version', '1');", ""
        ) + _MIGRATIONS[2].replace(
            "UPDATE meta SET value = '2' WHERE key = 'schema_version';",
            "INSERT INTO meta (key, value) VALUES ('schema_version', '2');",
        )
        with Database(tmp_path / "fresh2.db") as db:
            fresh = self._shape(db)
        assert fresh == self._migrated(tmp_path, v2_sql, "from_v2.db")


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
