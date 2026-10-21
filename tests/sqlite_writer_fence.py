"""Raw SQLite connections that can write past the product's writer fence.

The canonical schema installs ``trg_v<N>_writer_fence_*`` triggers over the
transcript and derived tables (``turns``, ``chunks``, ``chunk_terms``,
``memory_items``, ...). Each one aborts the write unless the *connection*
exposes an application-defined ``memory_condense_writer_schema_version()``
returning the schema version the triggers were installed for. Only
``memory_condense.persistence.db.Database`` registers that function, so the
fence rejects any writer that is not the product's own connection.

Tamper-rejection tests must still be able to produce the tampered bytes: they
model an attacker who already owns the store file and can therefore satisfy (or
simply remove) the fence. Such a test opens a raw connection, so it has to
register the same function the product registers — otherwise the tampering
``UPDATE`` aborts during setup, the tamper never lands, and the audit the test
is supposed to exercise is never reached.

The version comes from ``CURRENT_SCHEMA_VERSION`` so the helper tracks the
schema and cannot drift out of step with a future fence bump.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from memory_condense.persistence.db import CURRENT_SCHEMA_VERSION


def register_writer_schema_version(
    connection: sqlite3.Connection,
    *,
    version: int = CURRENT_SCHEMA_VERSION,
) -> sqlite3.Connection:
    """Register the writer-identity function the fence triggers resolve.

    Mirrors ``Database.__init__``. Returns the same connection so callers can
    use it inline.
    """

    connection.create_function(
        "memory_condense_writer_schema_version",
        0,
        lambda: version,
        deterministic=True,
    )
    return connection


def connect_past_writer_fence(
    target: str | Path,
    *,
    uri: bool = False,
    **kwargs: object,
) -> sqlite3.Connection:
    """Open a raw connection allowed to write fenced tables.

    Use this only to stage tampering that an audit is then expected to reject;
    production writes belong to ``Database``.
    """

    connection = sqlite3.connect(
        target if uri else str(target),
        uri=uri,
        **kwargs,  # type: ignore[arg-type]
    )
    try:
        register_writer_schema_version(connection)
    except BaseException:
        connection.close()
        raise
    return connection


__all__ = [
    "connect_past_writer_fence",
    "register_writer_schema_version",
]
