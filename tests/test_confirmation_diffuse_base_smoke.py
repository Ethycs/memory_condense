"""Real-store smoke coverage for the confirmation source-ingest boundary."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from memory_condense.persistence.db import CURRENT_SCHEMA_VERSION
from tests import test_diffuse_longmemeval_base as base_fixtures
from tests import test_embedding as embedding_tests


def test_confirmation_base_store_records_one_index_revision(
    tmp_path: Path,
) -> None:
    """The one-batch source build and its strict audit must agree on metadata."""

    with base_fixtures._published(tmp_path / "cache") as published:
        base = published[0]
        database = (
            base.store_path
            / base_fixtures.STORE_DIRECTORY_NAME
            / base_fixtures._DATABASE_NAME
        )
        uri = f"file:{database.as_posix()}?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True) as connection:
            rows = connection.execute(
                "SELECT key, value FROM meta ORDER BY key"
            ).fetchall()
            chunk_count = connection.execute(
                "SELECT COUNT(*) FROM chunks"
            ).fetchone()[0]

    assert rows == [
        ("chunk_index_revision", "1"),
        ("next_hnsw_label", str(chunk_count)),
        ("schema_version", str(CURRENT_SCHEMA_VERSION)),
    ]


def test_confirmation_pinned_embedder_is_forced_offline(monkeypatch) -> None:
    """Reuse the core loader contract inside the sealed confirmation gate."""

    embedding_tests.test_default_model_load_is_revision_pinned_and_hash_verified(
        monkeypatch
    )
