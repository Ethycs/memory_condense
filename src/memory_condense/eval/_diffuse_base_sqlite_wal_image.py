"""Narrow SQLite WAL-image codec for writable diffuse clone workspaces.

Python's bundled SQLite can deserialize rollback-journal database images, but
not a main database whose header advertises WAL read/write versions.  Diffuse
clones preserve their schema and header policy while ordinary derived rows may
grow while leaving the freelist unchanged. This codec canonicalizes that shape;
schema and semantic table invariants are audited on the live connection before
the resulting bytes are authorized for disk write.
"""

from __future__ import annotations

import sqlite3

from memory_condense.eval._diffuse_base_contracts import DiffuseBaseArtifactError
from memory_condense.eval._diffuse_base_publication_guard import freeze_namespace_guard


_SQLITE_MAGIC = b"SQLite format 3\x00"
_SUPPORTED_SQLITE_VERSION = (3, 53, 4)
_HEADER_BYTES = 100
_WAL_VERSIONS = b"\x02\x02"
_ROLLBACK_VERSIONS = b"\x01\x01"
_VOLATILE_HEADER_SLICES = ((18, 20), (24, 28), (92, 96))
_CANONICAL_PAGE_ONE_VARIABLE_SLICES = ((18, 20), (24, 32), (92, 96))
_INVARIANT_HEADER_SLICES = (
    (16, 18),  # page size
    (20, 24),  # reserved byte and payload fractions
    (40, 92),  # schema cookie/format, encoding, versions, vacuum policy
    (96, 100),  # SQLite version that last wrote the file
)


def _runtime_supported() -> None:
    if sqlite3.sqlite_version_info != _SUPPORTED_SQLITE_VERSION or not (
        hasattr(sqlite3.Connection, "deserialize")
        and hasattr(sqlite3.Connection, "serialize")
    ):
        raise DiffuseBaseArtifactError(
            "derived SQLite image runtime is not the audited 3.53.4 backend"
        )


def _page_size(payload: bytes, *, versions: bytes) -> int:
    if type(payload) is not bytes or len(payload) < _HEADER_BYTES:
        raise DiffuseBaseArtifactError("derived SQLite image is truncated")
    if payload[:16] != _SQLITE_MAGIC or payload[18:20] != versions:
        raise DiffuseBaseArtifactError("derived SQLite image has an unsupported header")
    encoded = int.from_bytes(payload[16:18], "big")
    page_size = 65536 if encoded == 1 else encoded
    if (
        page_size < 512
        or page_size > 65536
        or page_size & (page_size - 1)
        or len(payload) % page_size
    ):
        raise DiffuseBaseArtifactError("derived SQLite image has an invalid page layout")
    if payload[20] != 0 or payload[21:24] != bytes((64, 32, 32)):
        raise DiffuseBaseArtifactError("derived SQLite payload fractions are unsupported")
    if int.from_bytes(payload[28:32], "big") != len(payload) // page_size:
        raise DiffuseBaseArtifactError("derived SQLite page count disagrees with its bytes")
    if payload[32:40] != b"\x00" * 8 or payload[72:92] != b"\x00" * 20:
        raise DiffuseBaseArtifactError(
            "derived SQLite freelist or reserved header state is unsupported"
        )
    if int.from_bytes(payload[44:48], "big") not in {1, 2, 3, 4}:
        raise DiffuseBaseArtifactError("derived SQLite schema format is unsupported")
    if int.from_bytes(payload[56:60], "big") not in {1, 2, 3}:
        raise DiffuseBaseArtifactError("derived SQLite encoding is unsupported")
    if payload[24:28] != payload[92:96]:
        raise DiffuseBaseArtifactError("derived SQLite header counters disagree")
    return page_size


def deserialize_wal_image(payload: bytes) -> bytes:
    """Return an audited rollback-header image suitable for deserialize()."""

    _runtime_supported()
    _page_size(payload, versions=_WAL_VERSIONS)
    converted = bytearray(payload)
    converted[18:20] = _ROLLBACK_VERSIONS
    return bytes(converted)


def _canonical_page_one(payload: bytes, page_size: int) -> bytes:
    page = bytearray(payload[:page_size])
    for start, stop in _CANONICAL_PAGE_ONE_VARIABLE_SLICES:
        page[start:stop] = b"\x00" * (stop - start)
    return bytes(page)


def serialize_wal_image(original: bytes, serialized: bytes) -> bytes:
    """Reconstruct exact supported legacy WAL main-file bytes.

    The output is canonical for the derived backend: it advertises WAL mode and
    retains the original equal change/version counters. Page-count growth and
    appended pages remain serialized connection output; the page-count field,
    encoding, application/user-version, and auto-vacuum header policy cannot
    change. Exact sqlite_schema equality is checked by the lifecycle audit.
    """

    _runtime_supported()
    page_size = _page_size(original, versions=_WAL_VERSIONS)
    serialized_page_size = _page_size(serialized, versions=_ROLLBACK_VERSIONS)
    if serialized_page_size != page_size:
        raise DiffuseBaseArtifactError(
            "derived SQLite compilation changed the database page size"
        )
    changed = [
        f"{start}:{stop}"
        for start, stop in _INVARIANT_HEADER_SLICES
        if serialized[start:stop] != original[start:stop]
    ]
    if changed:
        raise DiffuseBaseArtifactError(
            "derived SQLite compilation changed header invariants: "
            + ", ".join(changed)
        )
    if _canonical_page_one(serialized, page_size) != _canonical_page_one(
        original, page_size
    ):
        raise DiffuseBaseArtifactError(
            "derived SQLite compilation changed the sqlite_schema page"
        )
    result = bytearray(serialized)
    for start, stop in _VOLATILE_HEADER_SLICES:
        result[start:stop] = original[start:stop]
    encoded = bytes(result)
    _page_size(encoded, versions=_WAL_VERSIONS)
    return encoded


def _freeze_codec_guard(namespace):
    assert_namespace = freeze_namespace_guard(
        namespace,
        error_type=DiffuseBaseArtifactError,
        label="derived SQLite WAL codec",
        exclude=("_freeze_codec_guard",),
    )
    expected_version = sqlite3.sqlite_version_info
    expected_connect = sqlite3.connect
    expected_deserialize = sqlite3.Connection.deserialize
    expected_serialize = sqlite3.Connection.serialize

    def acquire():
        def assert_intact() -> None:
            assert_namespace()
            if (
                sqlite3.sqlite_version_info != expected_version
                or sqlite3.connect is not expected_connect
                or sqlite3.Connection.deserialize is not expected_deserialize
                or sqlite3.Connection.serialize is not expected_serialize
            ):
                raise DiffuseBaseArtifactError(
                    "derived SQLite WAL runtime was rebound"
                )

        assert_intact()
        return assert_intact

    return acquire


sqlite_wal_image_operation_guard = _freeze_codec_guard(globals())
del _freeze_codec_guard


__all__ = ["deserialize_wal_image", "serialize_wal_image"]
