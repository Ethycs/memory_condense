"""Held read-only snapshots for finalized diffuse derived artifacts."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Iterator

from memory_condense.eval._diffuse_base_contracts import (
    DATABASE_NAME,
    DERIVED_FINALIZATION_NAME,
    DERIVED_LEASE_NAME,
    DERIVED_ORIGIN_NAME,
    INDEX_NAME,
    DiffuseBaseArtifactError,
)
from memory_condense.eval._diffuse_base_derived_lifecycle_filesystem import (
    _hash_entry,
    derived_lifecycle_operation_guard,
)
from memory_condense.eval._diffuse_base_derived_lifecycle_native import (
    MAX_METADATA_BYTES,
    assert_bound,
    read_exact,
    require_one_link,
)
from memory_condense.eval._diffuse_base_publication_guard import (
    freeze_callable_guard,
    freeze_namespace_guard,
)
from memory_condense.eval._diffuse_latent_training_corpus_filesystem import (
    _OpenEntry,
    _entry_names,
    _open_chain,
    _open_child,
    LatentTrainingCorpusError,
)

if os.name == "nt":
    from ctypes import wintypes


_MAX_DATABASE_BYTES = 512 * 1024 * 1024
_FINAL_NAMES = {
    DATABASE_NAME,
    INDEX_NAME,
    DERIVED_ORIGIN_NAME,
    DERIVED_LEASE_NAME,
    DERIVED_FINALIZATION_NAME,
}


@dataclass(frozen=True, slots=True)
class HeldRegularFileSnapshot:
    payload: bytes
    size: int


@dataclass(frozen=True, slots=True)
class HeldFinalizedDerivedSnapshot:
    database_bytes: bytes
    database_size: int
    database_sha256: str
    index_sha256: str
    index_size: int
    origin_bytes: bytes
    lease_bytes: bytes
    finalization_bytes: bytes


def _assert_chain(chain: list[_OpenEntry]) -> None:
    for index in range(1, len(chain)):
        assert_bound(chain[index - 1], chain[index].path.name, chain[index])


def _build_snapshot_closer():
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        raw_close = kernel32.CloseHandle
        raw_close.argtypes = (wintypes.HANDLE,)
        raw_close.restype = wintypes.BOOL

        def close_one(handle: int) -> None:
            if not raw_close(handle):
                raise OSError(ctypes.get_last_error(), "cannot close snapshot handle")

    else:
        raw_close = os.close

        def close_one(handle: int) -> None:
            raw_close(handle)

    def acquire():
        def close_all(entries: tuple[_OpenEntry, ...]) -> None:
            failure: BaseException | None = None
            seen: set[int] = set()
            for entry in reversed(entries):
                if entry.handle in seen:
                    continue
                try:
                    close_one(entry.handle)
                except BaseException as exc:
                    if failure is None:
                        failure = exc
                    else:
                        failure.add_note(f"snapshot handle close also failed: {exc!r}")
                seen.add(entry.handle)
            if failure is not None:
                raise failure

        return close_all

    return acquire


_snapshot_closer = _build_snapshot_closer()
del _build_snapshot_closer


def _close_with_primary(
    entries: tuple[_OpenEntry, ...],
    original: BaseException | None,
    close_all,
) -> None:
    try:
        close_all(entries)
    except BaseException as cleanup_error:
        if original is None:
            raise
        original.add_note(f"held snapshot close also failed: {cleanup_error!r}")


@contextmanager
def _held_regular_file_snapshot(
    path: Path,
    *,
    maximum_bytes: int,
    _sealed_guard,
) -> Iterator[HeldRegularFileSnapshot]:
    """Yield bounded bytes while retaining exact parent/name authority."""

    _sealed_guard()
    close_all = _snapshot_closer()
    assert_lifecycle_intact, _emergency, _registration = (
        derived_lifecycle_operation_guard()
    )
    chain: list[_OpenEntry] = []
    entry: _OpenEntry | None = None
    try:
        chain = _open_chain(path.parent)
        entry = _open_child(chain[-1], path.name, directory=False)
        _assert_chain(chain)
        assert_bound(chain[-1], path.name, entry)
        require_one_link(entry)
        payload = read_exact(entry, limit=maximum_bytes)
        snapshot = HeldRegularFileSnapshot(payload=payload, size=len(payload))
        yield snapshot
        _sealed_guard()
        assert_lifecycle_intact()
        _assert_chain(chain)
        assert_bound(chain[-1], path.name, entry)
        require_one_link(entry)
        if read_exact(entry, limit=maximum_bytes) != payload:
            raise DiffuseBaseArtifactError("held file snapshot changed")
    except BaseException as original:
        normalized = (
            DiffuseBaseArtifactError("cannot snapshot the derived file")
            if isinstance(original, LatentTrainingCorpusError)
            else original
        )
        _close_with_primary(
            (() if entry is None else (entry,)) + tuple(chain),
            normalized,
            close_all,
        )
        if normalized is not original:
            raise normalized from original
        raise
    else:
        _close_with_primary((entry, *chain), None, close_all)


def _read_finalized(
    root: _OpenEntry,
    entries: dict[str, _OpenEntry],
) -> HeldFinalizedDerivedSnapshot:
    for name, entry in entries.items():
        assert_bound(root, name, entry)
        require_one_link(entry)
    database = read_exact(entries[DATABASE_NAME], limit=_MAX_DATABASE_BYTES)
    return HeldFinalizedDerivedSnapshot(
        database_bytes=database,
        database_size=len(database),
        database_sha256=hashlib.sha256(database).hexdigest(),
        index_sha256=_hash_entry(entries[INDEX_NAME]),
        index_size=entries[INDEX_NAME].size,
        origin_bytes=read_exact(
            entries[DERIVED_ORIGIN_NAME], limit=MAX_METADATA_BYTES
        ),
        lease_bytes=read_exact(
            entries[DERIVED_LEASE_NAME], limit=MAX_METADATA_BYTES
        ),
        finalization_bytes=read_exact(
            entries[DERIVED_FINALIZATION_NAME], limit=MAX_METADATA_BYTES
        ),
    )


@contextmanager
def _held_finalized_derived_snapshot(
    path: Path,
    *,
    _sealed_guard,
) -> Iterator[HeldFinalizedDerivedSnapshot]:
    """Hold one exact finalized tree across verification and reconstruction."""

    _sealed_guard()
    close_all = _snapshot_closer()
    assert_lifecycle_intact, _emergency, _registration = (
        derived_lifecycle_operation_guard()
    )
    chain: list[_OpenEntry] = []
    root: _OpenEntry | None = None
    entries: dict[str, _OpenEntry] = {}
    try:
        chain = _open_chain(path.parent)
        root = _open_child(chain[-1], path.name, directory=True)
        _assert_chain(chain)
        assert_bound(chain[-1], path.name, root)
        if set(_entry_names(root, cap=6)) != _FINAL_NAMES:
            raise DiffuseBaseArtifactError("finalized derived tree changed")
        for name in sorted(_FINAL_NAMES):
            entries[name] = _open_child(root, name, directory=False)
        snapshot = _read_finalized(root, entries)
        yield snapshot
        _sealed_guard()
        assert_lifecycle_intact()
        _assert_chain(chain)
        assert_bound(chain[-1], path.name, root)
        if set(_entry_names(root, cap=6)) != _FINAL_NAMES:
            raise DiffuseBaseArtifactError("finalized derived tree changed")
        if _read_finalized(root, entries) != snapshot:
            raise DiffuseBaseArtifactError("finalized derived snapshot changed")
    except BaseException as original:
        normalized = (
            DiffuseBaseArtifactError("cannot snapshot the finalized derived tree")
            if isinstance(original, LatentTrainingCorpusError)
            else original
        )
        _close_with_primary(
            tuple(entries.values())
            + (() if root is None else (root,))
            + tuple(chain),
            normalized,
            close_all,
        )
        if normalized is not original:
            raise normalized from original
        raise
    else:
        _close_with_primary((*entries.values(), root, *chain), None, close_all)


def _seal_snapshot_entrypoints(regular, finalized, guard):
    assert_regular = freeze_callable_guard(
        regular,
        error_type=DiffuseBaseArtifactError,
        label="held regular snapshot implementation",
    )
    assert_finalized = freeze_callable_guard(
        finalized,
        error_type=DiffuseBaseArtifactError,
        label="held finalized snapshot implementation",
    )
    regular_init = HeldRegularFileSnapshot.__init__
    finalized_init = HeldFinalizedDerivedSnapshot.__init__
    expected_sha256 = hashlib.sha256
    assert_regular_init = freeze_callable_guard(
        regular_init,
        error_type=DiffuseBaseArtifactError,
        label="held regular snapshot constructor",
    )
    assert_finalized_init = freeze_callable_guard(
        finalized_init,
        error_type=DiffuseBaseArtifactError,
        label="held finalized snapshot constructor",
    )

    def assert_all() -> None:
        guard()
        if (
            HeldRegularFileSnapshot.__init__ is not regular_init
            or HeldFinalizedDerivedSnapshot.__init__ is not finalized_init
            or hashlib.sha256 is not expected_sha256
        ):
            raise DiffuseBaseArtifactError("held snapshot constructor was rebound")
        assert_regular_init(regular_init)
        assert_finalized_init(finalized_init)

    def held_regular_file_snapshot(
        path: Path,
        *,
        maximum_bytes: int,
    ):
        assert_all()
        assert_regular(regular)
        return regular(path, maximum_bytes=maximum_bytes, _sealed_guard=assert_all)

    def held_finalized_derived_snapshot(path: Path):
        assert_all()
        assert_finalized(finalized)
        return finalized(path, _sealed_guard=assert_all)

    return held_regular_file_snapshot, held_finalized_derived_snapshot


_SNAPSHOT_GUARD_EXCLUDES = (
    "_seal_snapshot_entrypoints",
    "held_regular_file_snapshot",
    "held_finalized_derived_snapshot",
    "_SNAPSHOT_GUARD_EXCLUDES",
    "_sealed_snapshot_guard",
)
_sealed_snapshot_guard = freeze_namespace_guard(
    globals(),
    error_type=DiffuseBaseArtifactError,
    label="held derived snapshot module",
    exclude=_SNAPSHOT_GUARD_EXCLUDES,
)
held_regular_file_snapshot, held_finalized_derived_snapshot = (
    _seal_snapshot_entrypoints(
        _held_regular_file_snapshot,
        _held_finalized_derived_snapshot,
        _sealed_snapshot_guard,
    )
)
del _seal_snapshot_entrypoints, _sealed_snapshot_guard


__all__ = [
    "HeldFinalizedDerivedSnapshot",
    "HeldRegularFileSnapshot",
    "held_finalized_derived_snapshot",
    "held_regular_file_snapshot",
]
