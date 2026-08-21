from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from memory_condense.eval._diffuse_base_contracts import DiffuseBaseArtifactError
from memory_condense.eval._diffuse_base_sqlite_wal_image import (
    deserialize_wal_image,
    serialize_wal_image,
)


def _wal_database(path: Path) -> bytes:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA wal_autocheckpoint=1")
    connection.execute("CREATE TABLE values_table (key TEXT PRIMARY KEY, value TEXT)")
    connection.execute("INSERT INTO values_table VALUES ('fixed', 'alpha')")
    connection.commit()
    connection.close()
    return path.read_bytes()


def test_wal_codec_round_trips_noop_without_changing_any_byte(tmp_path: Path) -> None:
    original = _wal_database(tmp_path / "memory.db")
    connection = sqlite3.connect(":memory:")
    connection.deserialize(deserialize_wal_image(original))
    serialized = connection.serialize()
    connection.close()
    assert serialize_wal_image(original, serialized) == original


def test_wal_codec_matches_legacy_fixed_layout_update(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    original = _wal_database(source)
    legacy = tmp_path / "legacy.db"
    shutil.copyfile(source, legacy)
    legacy_connection = sqlite3.connect(legacy)
    legacy_connection.execute("PRAGMA journal_mode=WAL")
    legacy_connection.execute(
        "UPDATE values_table SET value = 'bravo' WHERE key = 'fixed'"
    )
    legacy_connection.commit()
    legacy_connection.close()

    memory = sqlite3.connect(":memory:")
    memory.deserialize(deserialize_wal_image(original))
    memory.execute("UPDATE values_table SET value = 'bravo' WHERE key = 'fixed'")
    memory.commit()
    encoded = serialize_wal_image(original, memory.serialize())
    memory.close()
    assert encoded == legacy.read_bytes()


def test_wal_codec_allows_growth_but_refuses_schema_changes(tmp_path: Path) -> None:
    original = _wal_database(tmp_path / "memory.db")
    memory = sqlite3.connect(":memory:")
    memory.deserialize(deserialize_wal_image(original))
    memory.execute(
        "INSERT INTO values_table VALUES ('growing', ?)",
        ("x" * 100_000,),
    )
    memory.commit()
    grown = serialize_wal_image(original, memory.serialize())
    assert len(grown) > len(original)
    assert grown[18:20] == b"\x02\x02"
    assert grown[24:28] == original[24:28] == grown[92:96]
    memory.close()
    canonical = tmp_path / "canonical-grown.db"
    canonical.write_bytes(grown)
    read_only = sqlite3.connect(
        f"{canonical.resolve().as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    assert read_only.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    assert read_only.execute("SELECT COUNT(*) FROM values_table").fetchone() == (2,)
    read_only.close()

    memory = sqlite3.connect(":memory:")
    memory.deserialize(deserialize_wal_image(original))
    memory.execute("CREATE TABLE unsupported_layout_change (value BLOB)")
    memory.commit()
    with pytest.raises(DiffuseBaseArtifactError, match="header invariants"):
        serialize_wal_image(original, memory.serialize())
    memory.close()


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value[:18] + b"\x01\x01" + value[20:],
        lambda value: value[:24] + b"\x00\x00\x00\x01" + value[28:],
        lambda value: value[:21] + b"\x3f" + value[22:],
        lambda value: value[:28] + b"\x00\x00\x00\x01" + value[32:],
        lambda value: value[:32] + b"\x00\x00\x00\x01" + value[36:],
        lambda value: value[:72] + b"\x01" + value[73:],
        lambda value: value[:-1],
    ),
)
def test_wal_codec_rejects_unsupported_or_inconsistent_images(
    tmp_path: Path,
    mutation,
) -> None:
    original = _wal_database(tmp_path / "memory.db")
    with pytest.raises(DiffuseBaseArtifactError):
        deserialize_wal_image(mutation(original))
