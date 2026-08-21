"""Native held-file operations for derived lifecycle receipts."""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import secrets
import stat

from memory_condense.eval._diffuse_base_contracts import (
    DERIVED_FINALIZATION_NAME,
    DiffuseBaseArtifactError,
)
from memory_condense.eval._diffuse_latent_training_corpus_filesystem import (
    _OpenEntry,
    _assert_named_object,
    _close_entry,
    _flush_open_directory,
    _open_child,
    _posix_flags,
    _posix_identity,
    _posix_rename_noreplace,
    _read_entry,
    _same_object,
    _win_final_path,
    _win_info,
    _win_mark_delete,
    _win_open,
    _win_rename,
)

if os.name == "nt":
    from ctypes import wintypes

    from memory_condense.eval._diffuse_latent_training_corpus_filesystem import (
        _CREATE_NEW,
        _DELETE,
        _FILE_FLAG_OPEN_REPARSE_POINT,
        _FILE_READ_ATTRIBUTES,
        _GENERIC_READ,
        _GENERIC_WRITE,
        _INVALID_HANDLE,
        _kernel32,
        _win_close,
        _win_error,
    )


MAX_METADATA_BYTES = 64 * 1024 * 1024
MAX_DATABASE_BYTES = 512 * 1024 * 1024
_DUPLICATE_SAME_ACCESS = 2


def build_commit_closer():
    """Build an independently owned exhaustive raw-handle closer."""

    if os.name == "nt":
        emergency_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        raw_close = emergency_kernel32.CloseHandle
        raw_close.argtypes = (wintypes.HANDLE,)
        raw_close.restype = wintypes.BOOL

        def close_handle(handle: int) -> bool:
            return bool(raw_close(handle))

    else:
        raw_close = os.close

        def close_handle(handle: int) -> bool:
            try:
                raw_close(handle)
            except OSError:
                return False
            return True

    def close_entries_raw(
        entries: tuple[_OpenEntry, ...],
    ) -> tuple[_OpenEntry, ...]:
        failed: list[_OpenEntry] = []
        seen: set[int] = set()
        for entry in reversed(entries):
            if entry.handle not in seen and not close_handle(entry.handle):
                failed.append(entry)
            seen.add(entry.handle)
        return tuple(failed)

    return close_entries_raw


_COMMIT_CLOSE_ENTRIES = build_commit_closer()


def close_committed_lifecycle(
    root: _OpenEntry,
    finalization: _OpenEntry,
    backup_root: _OpenEntry,
    backup_finalization: _OpenEntry,
    entries: tuple[_OpenEntry, ...],
) -> None:
    """Close originals once, using distinct backups only for rollback."""

    failed = _COMMIT_CLOSE_ENTRIES(entries)
    if failed:
        error = DiffuseBaseArtifactError(
            "derived durable commit could not release all held handles: "
            + ", ".join(str(entry.handle) for entry in failed)
        )
        try:
            if not _same_object(root.identity, backup_root.identity):
                raise DiffuseBaseArtifactError(
                    "derived root backup changed during commit cleanup"
                )
            if not _same_object(
                finalization.identity, backup_finalization.identity
            ):
                raise DiffuseBaseArtifactError(
                    "derived final marker backup changed during commit cleanup"
                )
            discard_created_child(
                backup_root,
                DERIVED_FINALIZATION_NAME,
                backup_finalization,
            )
        except BaseException as cleanup_error:
            error.add_note(
                f"derived commit marker rollback also failed: {cleanup_error!r}"
            )
        if _COMMIT_CLOSE_ENTRIES((backup_root,)):
            error.add_note("derived commit quarantine retained its backup root")
        raise error

    backup_failures = _COMMIT_CLOSE_ENTRIES(
        (backup_root, backup_finalization)
    )
    if backup_failures:
        raise DiffuseBaseArtifactError(
            "derived durable commit could not release backup handles: "
            + ", ".join(str(entry.handle) for entry in backup_failures)
        )


def duplicate_entry(entry: _OpenEntry) -> _OpenEntry:
    if os.name == "nt":
        process = _kernel32.GetCurrentProcess()
        duplicated = wintypes.HANDLE()
        if not _kernel32.DuplicateHandle(
            process,
            entry.handle,
            process,
            ctypes.byref(duplicated),
            0,
            False,
            _DUPLICATE_SAME_ACCESS,
        ):
            raise _win_error("cannot duplicate derived handoff handle")
        raw = int(duplicated.value)
        try:
            if not _same_object(_win_info(raw)[0], entry.identity):
                raise DiffuseBaseArtifactError("derived handoff object changed")
            return _OpenEntry(
                entry.path,
                raw,
                entry.identity,
                entry.size,
                entry.is_directory,
            )
        except BaseException:
            _win_close(raw)
            raise
    descriptor = os.dup(entry.handle)
    try:
        if not _same_object(_posix_identity(os.fstat(descriptor)), entry.identity):
            raise DiffuseBaseArtifactError("derived handoff object changed")
        return _OpenEntry(
            entry.path,
            descriptor,
            entry.identity,
            entry.size,
            entry.is_directory,
        )
    except BaseException:
        os.close(descriptor)
        raise


def open_writable_child(parent: _OpenEntry, name: str) -> _OpenEntry:
    if os.name == "nt":
        return _win_open(parent.path / name, directory=False, write=True)
    descriptor = os.open(
        name,
        _posix_flags(directory=False, write=True),
        dir_fd=parent.handle,
    )
    try:
        value = os.fstat(descriptor)
        if not stat.S_ISREG(value.st_mode):
            raise DiffuseBaseArtifactError("derived database is not a regular file")
        return _OpenEntry(
            parent.path / name,
            descriptor,
            _posix_identity(value),
            int(value.st_size),
            False,
        )
    except BaseException:
        os.close(descriptor)
        raise


def hash_entry(entry: _OpenEntry) -> str:
    digest = hashlib.sha256()
    remaining = entry.size
    if os.name == "nt":
        if not _kernel32.SetFilePointerEx(entry.handle, 0, None, 0):
            raise _win_error("cannot seek held derived file")
        while remaining:
            width = min(remaining, 1024 * 1024)
            buffer = ctypes.create_string_buffer(width)
            read = wintypes.DWORD()
            if not _kernel32.ReadFile(
                entry.handle, buffer, width, ctypes.byref(read), None
            ) or not read.value:
                raise _win_error("cannot read held derived file")
            digest.update(buffer.raw[: read.value])
            remaining -= read.value
    else:
        os.lseek(entry.handle, 0, os.SEEK_SET)
        while remaining:
            chunk = os.read(entry.handle, min(remaining, 1024 * 1024))
            if not chunk:
                raise DiffuseBaseArtifactError("held derived file ended early")
            digest.update(chunk)
            remaining -= len(chunk)
    if current_identity(entry) != entry.identity:
        raise DiffuseBaseArtifactError("held derived file changed while hashing")
    return digest.hexdigest()


def write_exact(entry: _OpenEntry, payload: bytes) -> _OpenEntry:
    if (
        type(payload) is not bytes
        or not payload
        or len(payload) > MAX_DATABASE_BYTES
    ):
        raise DiffuseBaseArtifactError(
            "derived database output violates its byte bound"
        )
    if os.name == "nt":
        if not _kernel32.SetFilePointerEx(entry.handle, 0, None, 0):
            raise _win_error("cannot seek held derived database")
        if not _kernel32.SetEndOfFile(entry.handle):
            raise _win_error("cannot truncate held derived database")
        offset = 0
        while offset < len(payload):
            chunk = payload[offset : offset + 1024 * 1024]
            written = wintypes.DWORD()
            if not _kernel32.WriteFile(
                entry.handle,
                chunk,
                len(chunk),
                ctypes.byref(written),
                None,
            ) or written.value != len(chunk):
                raise _win_error("cannot write held derived database")
            offset += written.value
        if not _kernel32.FlushFileBuffers(entry.handle):
            raise _win_error("cannot flush held derived database")
    else:
        os.lseek(entry.handle, 0, os.SEEK_SET)
        os.ftruncate(entry.handle, 0)
        view = memoryview(payload)
        while view:
            count = os.write(entry.handle, view[: 1024 * 1024])
            if count <= 0:
                raise OSError("short held derived database write")
            view = view[count:]
        os.fsync(entry.handle)
    updated = refreshed(entry)
    if read_exact(updated, limit=MAX_DATABASE_BYTES) != payload:
        raise DiffuseBaseArtifactError("held derived database differs after flush")
    return updated


def current_identity(entry: _OpenEntry) -> tuple[int, ...]:
    if os.name == "nt":
        return _win_info(entry.handle)[0]
    return _posix_identity(os.fstat(entry.handle))


def refreshed(entry: _OpenEntry, *, path: Path | None = None) -> _OpenEntry:
    identity = current_identity(entry)
    return _OpenEntry(path or entry.path, entry.handle, identity, int(identity[3]), False)


def assert_bound(parent: _OpenEntry, name: str, entry: _OpenEntry) -> None:
    if os.name == "nt":
        expected = os.path.normcase(os.path.normpath(str(parent.path / name)))
        if _win_final_path(entry.handle) != expected or not _same_object(
            current_identity(entry), entry.identity
        ):
            raise DiffuseBaseArtifactError("derived lifecycle pathname changed")
        return
    _assert_named_object(parent, name, entry)


def require_one_link(entry: _OpenEntry) -> None:
    identity = current_identity(entry)
    count = identity[-1] if os.name == "nt" else int(os.fstat(entry.handle).st_nlink)
    if count != 1:
        raise DiffuseBaseArtifactError("derived lifecycle forbids hard-linked files")


def close_entries(entries: tuple[_OpenEntry, ...]) -> None:
    seen: set[int] = set()
    failure: BaseException | None = None
    for entry in reversed(entries):
        if entry.handle not in seen:
            try:
                _close_entry(entry)
            except BaseException as exc:
                if failure is None:
                    failure = exc
            seen.add(entry.handle)
    if failure is not None:
        raise failure


def read_exact(entry: _OpenEntry, *, limit: int) -> bytes:
    try:
        return _read_entry(entry, limit)
    except BaseException as exc:
        if isinstance(exc, DiffuseBaseArtifactError):
            raise
        raise DiffuseBaseArtifactError("cannot read held derived file") from exc


def flush_directory(entry: _OpenEntry) -> None:
    if os.name == "nt":
        if not _kernel32.FlushFileBuffers(entry.handle):
            raise _win_error("cannot flush held derived directory")
    else:
        _flush_open_directory(entry)


def _delete_windows_child_after_close(
    parent: _OpenEntry,
    name: str,
    identity: tuple[int, ...],
    entries: tuple[_OpenEntry, ...],
) -> None:
    failure: BaseException | None = None
    try:
        close_entries(entries)
    except BaseException as exc:
        failure = exc
    deleter: _OpenEntry | None = None
    deleted = False
    try:
        deleter = _win_open(
            parent.path / name,
            directory=False,
            delete_access=True,
        )
        if not _same_object(deleter.identity, identity):
            raise DiffuseBaseArtifactError(
                "derived receipt changed before exact deletion"
            )
        _win_mark_delete(deleter)
        deleted = True
    except BaseException as exc:
        if failure is None:
            failure = exc
        else:
            failure.add_note(f"derived receipt delete also failed: {exc!r}")
    if deleter is not None:
        try:
            _close_entry(deleter)
        except BaseException as exc:
            if failure is None:
                failure = exc
            else:
                failure.add_note(f"derived receipt deleter close also failed: {exc!r}")
    if deleted:
        try:
            flush_directory(parent)
        except BaseException as exc:
            if failure is None:
                failure = exc
            else:
                failure.add_note(f"derived receipt directory flush also failed: {exc!r}")
    if failure is not None:
        raise failure


def discard_created_child(
    parent: _OpenEntry,
    name: str,
    entry: _OpenEntry,
) -> None:
    if os.name == "nt":
        binding_failure: BaseException | None = None
        try:
            assert_bound(parent, name, refreshed(entry, path=parent.path / name))
        except BaseException as exc:
            binding_failure = exc
        try:
            _delete_windows_child_after_close(
                parent,
                name,
                entry.identity,
                (entry,),
            )
        except BaseException as cleanup_error:
            if binding_failure is None:
                binding_failure = cleanup_error
            else:
                binding_failure.add_note(
                    f"derived receipt release also failed: {cleanup_error!r}"
                )
        if binding_failure is not None:
            raise binding_failure
        return
    failure: BaseException | None = None
    deleted = False
    try:
        assert_bound(parent, name, refreshed(entry, path=parent.path / name))
        os.unlink(name, dir_fd=parent.handle)
        deleted = True
    except BaseException as exc:
        failure = exc
    try:
        _close_entry(entry)
    except BaseException as exc:
        if failure is None:
            failure = exc
        else:
            failure.add_note(f"held child close also failed: {exc!r}")
    if deleted:
        try:
            flush_directory(parent)
        except BaseException as exc:
            if failure is None:
                failure = exc
            else:
                failure.add_note(f"held child directory flush also failed: {exc!r}")
    if failure is not None:
        raise failure


def _downgrade_created_windows_entry(
    parent: _OpenEntry,
    name: str,
    creator: _OpenEntry,
) -> _OpenEntry:
    bridge: _OpenEntry | None = None
    stable: _OpenEntry | None = None
    creator_live = True
    try:
        bridge = _win_open(
            parent.path / name,
            directory=False,
            share_delete=True,
            share_write=True,
        )
        if not _same_object(bridge.identity, creator.identity):
            raise DiffuseBaseArtifactError("created derived receipt changed")
        _close_entry(creator)
        creator_live = False
        stable = _win_open(parent.path / name, directory=False)
        if not _same_object(stable.identity, creator.identity):
            raise DiffuseBaseArtifactError("created derived receipt changed")
        _close_entry(bridge)
        bridge = None
        return stable
    except BaseException as original:
        cleanup = tuple(entry for entry in (
            stable,
            bridge,
            creator if creator_live else None,
        ) if entry is not None)
        try:
            _delete_windows_child_after_close(
                parent,
                name,
                creator.identity,
                cleanup,
            )
        except BaseException as cleanup_error:
            original.add_note(
                f"derived receipt transition cleanup also failed: {cleanup_error!r}"
            )
        raise


def create_held_child(
    parent: _OpenEntry,
    name: str,
    payload: bytes,
    *,
    delete_access: bool = False,
) -> _OpenEntry:
    if type(payload) is not bytes or not payload or len(payload) > MAX_METADATA_BYTES:
        raise DiffuseBaseArtifactError("derived lifecycle receipt violates its byte bound")
    if os.name == "nt":
        handle = _kernel32.CreateFileW(
            str(parent.path / name),
            _GENERIC_READ | _GENERIC_WRITE | _FILE_READ_ATTRIBUTES | _DELETE,
            0x00000001,
            None,
            _CREATE_NEW,
            _FILE_FLAG_OPEN_REPARSE_POINT,
            None,
        )
        if handle == _INVALID_HANDLE:
            code = ctypes.get_last_error()
            if code in {80, 183}:
                raise FileExistsError(parent.path / name)
            raise _win_error("cannot create held derived receipt")
        raw = int(handle)
        try:
            offset = 0
            while offset < len(payload):
                chunk = payload[offset : offset + 1024 * 1024]
                written = wintypes.DWORD()
                if not _kernel32.WriteFile(
                    raw, chunk, len(chunk), ctypes.byref(written), None
                ) or written.value != len(chunk):
                    raise _win_error("cannot write held derived receipt")
                offset += written.value
            if not _kernel32.FlushFileBuffers(raw):
                raise _win_error("cannot flush held derived receipt")
            identity, size, attributes = _win_info(raw)
            if attributes & 0x00000410:
                raise DiffuseBaseArtifactError("held derived receipt is unsafe")
            entry = _OpenEntry(parent.path / name, raw, identity, size, False)
        except BaseException as original:
            deleted = False
            try:
                _win_mark_delete(_OpenEntry(parent.path / name, raw, (), 0, False))
                deleted = True
            except BaseException as cleanup_error:
                original.add_note(
                    f"partial derived receipt delete also failed: {cleanup_error!r}"
                )
            try:
                _win_close(raw)
            except BaseException as cleanup_error:
                original.add_note(
                    f"partial derived receipt close also failed: {cleanup_error!r}"
                )
            if deleted:
                try:
                    flush_directory(parent)
                except BaseException as cleanup_error:
                    original.add_note(
                        "partial derived receipt directory flush also failed: "
                        f"{cleanup_error!r}"
                    )
            raise
    else:
        descriptor = os.open(
            name,
            _posix_flags(directory=False, create=True),
            0o600,
            dir_fd=parent.handle,
        )
        try:
            view = memoryview(payload)
            while view:
                count = os.write(descriptor, view)
                if count <= 0:
                    raise OSError("short held derived receipt write")
                view = view[count:]
            os.fsync(descriptor)
            value = os.fstat(descriptor)
            entry = _OpenEntry(
                parent.path / name,
                descriptor,
                _posix_identity(value),
                int(value.st_size),
                False,
            )
        except BaseException as original:
            deleted = False
            try:
                os.unlink(name, dir_fd=parent.handle)
                deleted = True
            except BaseException as cleanup_error:
                original.add_note(
                    f"partial derived receipt delete also failed: {cleanup_error!r}"
                )
            try:
                os.close(descriptor)
            except BaseException as cleanup_error:
                original.add_note(
                    f"partial derived receipt close also failed: {cleanup_error!r}"
                )
            if deleted:
                try:
                    flush_directory(parent)
                except BaseException as cleanup_error:
                    original.add_note(
                        "partial derived receipt directory flush also failed: "
                        f"{cleanup_error!r}"
                    )
            raise
    try:
        require_one_link(entry)
        if read_exact(entry, limit=MAX_METADATA_BYTES) != payload:
            raise DiffuseBaseArtifactError(
                "held derived receipt changed after creation"
            )
    except BaseException as original:
        try:
            discard_created_child(parent, name, entry)
        except BaseException as cleanup_error:
            original.add_note(
                f"held derived receipt rollback also failed: {cleanup_error!r}"
            )
        raise
    if os.name == "nt" and not delete_access:
        return _downgrade_created_windows_entry(parent, name, entry)
    return entry


def create_atomic_finalization(parent: _OpenEntry, payload: bytes) -> _OpenEntry:
    temporary_name = f".{DERIVED_FINALIZATION_NAME}.new-{secrets.token_hex(16)}"
    temporary: _OpenEntry | None = None
    promoted = False
    try:
        temporary = create_held_child(
            parent,
            temporary_name,
            payload,
            delete_access=True,
        )
        if os.name == "nt":
            _win_rename(temporary, parent, DERIVED_FINALIZATION_NAME)
        else:
            _posix_rename_noreplace(parent, temporary_name, DERIVED_FINALIZATION_NAME)
        promoted = True
        finalization = refreshed(
            temporary,
            path=parent.path / DERIVED_FINALIZATION_NAME,
        )
        assert_bound(parent, DERIVED_FINALIZATION_NAME, finalization)
        flush_directory(parent)
        if os.name == "nt":
            temporary = None
            finalization = _downgrade_created_windows_entry(
                parent,
                DERIVED_FINALIZATION_NAME,
                finalization,
            )
        else:
            temporary = None
        return finalization
    except BaseException as original:
        if temporary is not None:
            deletion_name = DERIVED_FINALIZATION_NAME if promoted else temporary_name
            try:
                discard_created_child(parent, deletion_name, temporary)
            except BaseException as cleanup_error:
                original.add_note(
                    "derived finalization rollback also failed: "
                    f"{cleanup_error!r}"
                )
        raise


__all__ = [
    "MAX_METADATA_BYTES",
    "assert_bound",
    "close_entries",
    "create_atomic_finalization",
    "create_held_child",
    "current_identity",
    "discard_created_child",
    "flush_directory",
    "read_exact",
    "refreshed",
    "require_one_link",
]
