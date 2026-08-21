"""Capability-authorized publication for diffuse base artifacts.

The three publishers in this module's scope build into a random sibling,
capture a closed role-specific object inventory, and atomically rename that
same directory into place.  A registry-issued capability retains the root
identity from creation through caller verification.  Rollback first detaches
the exact owned object under a random tombstone and then removes only the
captured, bounded inventory; caller-provided paths never authorize deletion.

The low-level handle operations are shared with the audited latent-corpus
filesystem boundary.  Consequently traversal is descriptor-relative and
``O_NOFOLLOW`` on POSIX, while Windows opens reparse points themselves and
rejects them before use.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import secrets
from threading import RLock
from typing import Callable, Literal

from memory_condense.eval._diffuse_base_contracts import (
    DATABASE_NAME,
    DERIVED_ORIGIN_NAME,
    FROZEN_QUERY_INPUTS_NAME,
    INDEX_NAME,
    QUERY_MANIFEST_NAME,
    STORE_DIRECTORY_NAME,
    STORE_MANIFEST_NAME,
    DiffuseBaseArtifactError,
)
from memory_condense.eval._diffuse_base_publication_guard import (
    freeze_operation_guard,
)
from memory_condense.eval._diffuse_latent_training_corpus_filesystem import (
    _OpenEntry,
    _assert_named_object,
    _close_entry,
    _entry_names,
    _flush_open_directory,
    _open_chain,
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
    require_plain_parent,
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
        _write_windows,
    )


PublicationRole = Literal["store", "query", "derived"]
_PublicationPhase = Literal["created", "captured", "promoted", "tombstone"]
_SQLITE_PARTIAL_NAMES = {
    DATABASE_NAME,
    f"{DATABASE_NAME}-journal",
    f"{DATABASE_NAME}-shm",
    f"{DATABASE_NAME}-wal",
}
_MAX_METADATA_BYTES = 64 * 1024 * 1024
_MAX_COPY_BYTES = 4 * 1024 * 1024 * 1024


def _absolute_child(value: str | Path) -> Path:
    path = Path(os.path.abspath(os.fspath(value)))
    if not path.name or path.name in {".", ".."}:
        raise DiffuseBaseArtifactError(
            "publication target requires one bounded child name"
        )
    return path


def _bounded_name(value: str, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise DiffuseBaseArtifactError(f"{label} is not a bounded child name")
    return value


def _current_identity(entry: _OpenEntry) -> tuple[int, ...]:
    if os.name == "nt":
        return _win_info(entry.handle)[0]
    return _posix_identity(os.fstat(entry.handle))


def _refreshed(entry: _OpenEntry, *, path: Path | None = None) -> _OpenEntry:
    identity = _current_identity(entry)
    size = int(identity[3]) if os.name == "nt" else int(identity[3])
    return _OpenEntry(path or entry.path, entry.handle, identity, size, entry.is_directory)


def _assert_bound(parent: _OpenEntry, name: str, entry: _OpenEntry) -> None:
    if os.name == "nt":
        expected = os.path.normcase(os.path.normpath(str(parent.path / name)))
        if _win_final_path(entry.handle) != expected:
            raise DiffuseBaseArtifactError("owned publication pathname changed")
        if not _same_object(_current_identity(entry), entry.identity):
            raise DiffuseBaseArtifactError("owned publication object changed")
        return
    _assert_named_object(parent, name, entry)


def _require_one_link(entry: _OpenEntry) -> None:
    identity = _current_identity(entry)
    count = identity[-1] if os.name == "nt" else int(os.fstat(entry.handle).st_nlink)
    if count != 1:
        raise DiffuseBaseArtifactError("diffuse publications forbid hard-linked files")


def _mkdir_child(
    parent: _OpenEntry,
    name: str,
    *,
    delete_access: bool = False,
    held_parent_write: bool = False,
) -> _OpenEntry:
    name = _bounded_name(name, "publication directory")
    if os.name == "nt":
        (parent.path / name).mkdir(mode=0o700)
        entry = _win_open(parent.path / name, directory=True)
        try:
            _flush_open_directory(entry)
        finally:
            _close_entry(entry)
        entry = _win_open(
            parent.path / name,
            directory=True,
            write=delete_access,
            delete_access=delete_access,
        )
    else:
        os.mkdir(name, mode=0o700, dir_fd=parent.handle)
        entry = _open_child(parent, name, directory=True)
        _flush_open_directory(entry)
    if os.name == "nt" and held_parent_write:
        if not _kernel32.FlushFileBuffers(parent.handle):
            raise _win_error("cannot flush owned publication directory")
    else:
        _flush_open_directory(parent)
    return entry


def _write_child(parent: _OpenEntry, name: str, payload: bytes, *, limit: int) -> _OpenEntry:
    name = _bounded_name(name, "publication file")
    if type(payload) is not bytes or not payload or len(payload) > limit:
        raise DiffuseBaseArtifactError("publication payload violates its byte bound")
    if os.name == "nt":
        _write_windows(parent.path / name, payload)
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
                    raise OSError("short diffuse publication write")
                view = view[count:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    entry = _open_child(parent, name, directory=False)
    try:
        _require_one_link(entry)
        if _read_entry(entry, limit) != payload:
            raise DiffuseBaseArtifactError(
                "publication file differs immediately after creation"
            )
        result = _refreshed(entry)
    except BaseException:
        _close_entry(entry)
        raise
    if os.name == "nt":
        if not _kernel32.FlushFileBuffers(parent.handle):
            raise _win_error("cannot flush owned publication directory")
    else:
        _flush_open_directory(parent)
    return result


def _stream_entry(
    entry: _OpenEntry,
    consume: Callable[[bytes], None] | None = None,
) -> str:
    digest = hashlib.sha256()
    remaining = entry.size
    if os.name == "nt":
        if not _kernel32.SetFilePointerEx(entry.handle, 0, None, 0):
            raise _win_error("cannot seek diffuse publication file")
        while remaining:
            width = min(remaining, 1024 * 1024)
            buffer = ctypes.create_string_buffer(width)
            read = wintypes.DWORD()
            if not _kernel32.ReadFile(
                entry.handle,
                buffer,
                width,
                ctypes.byref(read),
                None,
            ):
                raise _win_error("cannot read diffuse publication file")
            if read.value == 0:
                raise DiffuseBaseArtifactError("publication file ended early")
            chunk = buffer.raw[: read.value]
            digest.update(chunk)
            if consume is not None:
                consume(chunk)
            remaining -= read.value
    else:
        os.lseek(entry.handle, 0, os.SEEK_SET)
        while remaining:
            chunk = os.read(entry.handle, min(remaining, 1024 * 1024))
            if not chunk:
                raise DiffuseBaseArtifactError("publication file ended early")
            digest.update(chunk)
            if consume is not None:
                consume(chunk)
            remaining -= len(chunk)
    if _current_identity(entry) != entry.identity:
        raise DiffuseBaseArtifactError("publication file changed while streaming")
    return digest.hexdigest()


def _copy_entry_to_child(
    parent: _OpenEntry,
    name: str,
    source: _OpenEntry,
) -> tuple[_OpenEntry, str]:
    name = _bounded_name(name, "derived copy")
    if os.name == "nt":
        handle = _kernel32.CreateFileW(
            str(parent.path / name),
            _GENERIC_WRITE,
            0,
            None,
            _CREATE_NEW,
            _FILE_FLAG_OPEN_REPARSE_POINT,
            None,
        )
        if handle == _INVALID_HANDLE:
            raise _win_error("cannot create exclusive derived copy")
        raw = int(handle)

        def consume(chunk: bytes) -> None:
            written = wintypes.DWORD()
            if not _kernel32.WriteFile(
                raw,
                chunk,
                len(chunk),
                ctypes.byref(written),
                None,
            ) or written.value != len(chunk):
                raise _win_error("cannot write complete derived copy")

        try:
            source_digest = _stream_entry(source, consume)
            if not _kernel32.FlushFileBuffers(raw):
                raise _win_error("cannot flush derived copy")
        finally:
            _win_close(raw)
    else:
        descriptor = os.open(
            name,
            _posix_flags(directory=False, create=True),
            0o600,
            dir_fd=parent.handle,
        )

        def consume(chunk: bytes) -> None:
            view = memoryview(chunk)
            while view:
                count = os.write(descriptor, view)
                if count <= 0:
                    raise OSError("short derived copy write")
                view = view[count:]

        try:
            source_digest = _stream_entry(source, consume)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    copied = _open_child(parent, name, directory=False)
    try:
        _require_one_link(copied)
        if copied.size != source.size or _stream_entry(copied) != source_digest:
            raise DiffuseBaseArtifactError("derived byte copy differs from its source")
        result = _refreshed(copied)
    except BaseException:
        _close_entry(copied)
        raise
    if os.name == "nt":
        if not _kernel32.FlushFileBuffers(parent.handle):
            raise _win_error("cannot flush owned publication directory")
    else:
        _flush_open_directory(parent)
    return result, source_digest


def _marker_name(target_name: str) -> str:
    return f".{_bounded_name(target_name, 'publication target')}.publish.lock"


def _create_marker_atomically(parent: _OpenEntry, name: str) -> bool:
    """Publish complete marker bytes; an observer never sees an empty file."""

    temporary_name = f"{name}.new-{secrets.token_hex(16)}"
    temporary: _OpenEntry | None = None
    promoted = False
    try:
        if os.name == "nt":
            handle = _kernel32.CreateFileW(
                str(parent.path / temporary_name),
                _GENERIC_READ | _GENERIC_WRITE | _DELETE | _FILE_READ_ATTRIBUTES,
                0,
                None,
                _CREATE_NEW,
                _FILE_FLAG_OPEN_REPARSE_POINT,
                None,
            )
            if handle == _INVALID_HANDLE:
                raise _win_error("cannot create publication marker staging")
            raw = int(handle)
            try:
                written = wintypes.DWORD()
                if not _kernel32.WriteFile(
                    raw, b"0", 1, ctypes.byref(written), None
                ) or written.value != 1:
                    raise _win_error("cannot write publication marker staging")
                if not _kernel32.FlushFileBuffers(raw):
                    raise _win_error("cannot flush publication marker staging")
                identity, size, _attributes = _win_info(raw)
                temporary = _OpenEntry(
                    parent.path / temporary_name,
                    raw,
                    identity,
                    size,
                    False,
                )
                raw = 0
            finally:
                if raw:
                    _win_close(raw)
        else:
            descriptor = os.open(
                temporary_name,
                _posix_flags(directory=False, create=True),
                0o600,
                dir_fd=parent.handle,
            )
            try:
                if os.write(descriptor, b"0") != 1:
                    raise OSError("short publication marker write")
                os.fsync(descriptor)
                value = os.fstat(descriptor)
                temporary = _OpenEntry(
                    parent.path / temporary_name,
                    descriptor,
                    _posix_identity(value),
                    int(value.st_size),
                    False,
                )
            except BaseException:
                os.close(descriptor)
                raise
        _require_one_link(temporary)
        if _read_entry(temporary, 1) != b"0":
            raise DiffuseBaseArtifactError("publication marker staging changed")
        try:
            if os.name == "nt":
                _win_rename(temporary, parent, name)
            else:
                _posix_rename_noreplace(parent, temporary_name, name)
            promoted = True
        except FileExistsError:
            pass
    finally:
        if temporary is not None:
            if not promoted:
                if os.name == "nt":
                    _win_mark_delete(temporary)
                else:
                    os.unlink(temporary_name, dir_fd=parent.handle)
            _close_entry(temporary)
    _flush_open_directory(parent)
    return promoted


def _open_publication_marker(
    parent: _OpenEntry, target_name: str
) -> tuple[_OpenEntry, bool]:
    name = _marker_name(target_name)
    created = False
    try:
        marker = _open_child(parent, name, directory=False)
    except BaseException as initial:
        # Only a genuinely missing name authorizes an exclusive create.  The
        # open primitives normalize errors, so lexists distinguishes absence
        # from a rejected symlink/reparse/special-file object.
        if os.path.lexists(parent.path / name):
            raise DiffuseBaseArtifactError("unsafe publication lock marker") from initial
        created = _create_marker_atomically(parent, name)
        marker = _open_child(parent, name, directory=False)
    try:
        _validate_marker_entry(parent, target_name, marker)
        return _refreshed(marker), created
    except BaseException:
        _close_entry(marker)
        raise


def _validate_marker_entry(
    parent: _OpenEntry,
    target_name: str,
    marker: _OpenEntry,
) -> None:
    _assert_bound(parent, _marker_name(target_name), marker)
    _require_one_link(marker)
    if _current_identity(marker) != marker.identity or _read_entry(marker, 1) != b"0":
        raise DiffuseBaseArtifactError(
            "publication lock marker must remain the exact file b'0'"
        )


def validate_publication_lock_marker(target: str | Path) -> None:
    """Validate an existing marker without creating or changing any name."""

    target_path = _absolute_child(target)
    parent = require_plain_parent(target_path.parent)
    chain = _open_chain(parent)
    marker: _OpenEntry | None = None
    try:
        marker = _open_child(
            chain[-1], _marker_name(target_path.name), directory=False
        )
        _validate_marker_entry(chain[-1], target_path.name, marker)
    except BaseException as exc:
        raise DiffuseBaseArtifactError(
            "published artifact has no safe lock marker"
        ) from exc
    finally:
        if marker is not None:
            _close_entry(marker)
        _close_entries(tuple(chain))


class OwnedBasePublication:
    """Opaque registry-issued authority over one random publication root."""

    __slots__ = ("_token",)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("publication capabilities cannot be constructed directly")

    def __setattr__(self, _name: str, _value: object) -> None:
        raise TypeError("publication capabilities are immutable")

    def __copy__(self) -> object:
        raise TypeError("publication capabilities cannot be copied")

    def __deepcopy__(self, _memo: object) -> object:
        raise TypeError("publication capabilities cannot be copied")

    def __reduce_ex__(self, _protocol: int) -> object:
        raise TypeError("publication capabilities cannot be serialized")

    def __fspath__(self) -> str:
        return os.fspath(_state(self).path)


@dataclass(frozen=True, slots=True)
class _InventoryEntry:
    relative: tuple[str, ...]
    directory: bool
    identity: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _State:
    owner: OwnedBasePublication
    role: PublicationRole
    phase: _PublicationPhase
    path: Path
    parent: Path
    target_name: str
    prefix: str
    root: _OpenEntry
    parent_chain: tuple[_OpenEntry, ...]
    marker: _OpenEntry | None = None
    marker_owned: bool = False
    store_child: _OpenEntry | None = None
    inventory: tuple[_InventoryEntry, ...] = ()
    held: tuple[_OpenEntry, ...] = ()


def _registry_boundary() -> tuple[Callable[..., object], ...]:
    registry: dict[object, _State] = {}
    lock = RLock()
    capability_type = OwnedBasePublication

    def state(value: OwnedBasePublication) -> _State:
        if type(value) is not capability_type:
            raise TypeError("operation requires an issued publication capability")
        try:
            token = object.__getattribute__(value, "_token")
        except (AttributeError, TypeError) as exc:
            raise TypeError("publication capability is not issued") from exc
        with lock:
            result = registry.get(token)
            if result is None or result.owner is not value:
                raise TypeError("publication capability is not live")
            return result

    def issue(**fields: object) -> OwnedBasePublication:
        owner = object.__new__(OwnedBasePublication)
        token = object()
        object.__setattr__(owner, "_token", token)
        created = _State(owner=owner, **fields)  # type: ignore[arg-type]
        with lock:
            registry[token] = created
        return owner

    def replace(
        owner: OwnedBasePublication,
        expected: _State,
        **changes: object,
    ) -> _State:
        token = object.__getattribute__(owner, "_token")
        with lock:
            current = registry.get(token)
            if current is not expected or current.owner is not owner:
                raise TypeError("publication capability changed during operation")
            fields = {
                name: getattr(current, name)
                for name in _State.__dataclass_fields__
                if name != "owner"
            }
            fields.update(changes)
            updated = _State(owner=owner, **fields)  # type: ignore[arg-type]
            registry[token] = updated
            return updated

    def revoke(owner: OwnedBasePublication, expected: _State) -> None:
        token = object.__getattribute__(owner, "_token")
        with lock:
            current = registry.get(token)
            if current is not expected or current.owner is not owner:
                raise TypeError("publication capability changed before revocation")
            del registry[token]

    return issue, state, replace, revoke


_issue, _state, _replace, _revoke = _registry_boundary()
del _registry_boundary


def publication_path(owner: OwnedBasePublication) -> Path:
    """Return the current path carried by a live capability."""

    return _state(owner).path


def prepare_derived_publication_handoff(
    owner: OwnedBasePublication,
) -> tuple[Path, _OpenEntry, tuple[_OpenEntry, ...], tuple[tuple[str, tuple[int, ...]], ...]]:
    """Release child handles while retaining root/ancestry for handoff."""

    state = _state(owner)
    if state.role != "derived" or state.phase != "promoted":
        raise TypeError("derived lifecycle registration requires promotion")
    assert_publication_unchanged(owner)
    state = _close_captured_children(owner, _state(owner))
    return state.path, state.root, state.parent_chain, tuple(
        (item.relative[0], item.identity) for item in state.inventory
    )


def _close_entries(entries: tuple[_OpenEntry, ...]) -> None:
    seen: set[int] = set()
    for entry in reversed(entries):
        if entry.handle in seen:
            continue
        _close_entry(entry)
        seen.add(entry.handle)


def _assert_parent_chain(state: _State) -> _OpenEntry:
    chain = state.parent_chain
    if not chain or chain[-1].path != state.parent:
        raise DiffuseBaseArtifactError("publication parent capability is malformed")
    for index in range(1, len(chain)):
        _assert_bound(chain[index - 1], chain[index].path.name, chain[index])
    return chain[-1]


def _root_kinds(role: PublicationRole, names: tuple[str, ...], *, final: bool) -> dict[str, bool]:
    values = set(names)
    if len(values) != len(names):
        raise DiffuseBaseArtifactError("publication enumeration repeated an entry")
    if role == "store":
        allowed = {STORE_DIRECTORY_NAME, STORE_MANIFEST_NAME}
        expected = allowed if final else values
        if not values <= allowed or (final and values != expected):
            raise DiffuseBaseArtifactError("store publication root changed schema")
        return {name: name == STORE_DIRECTORY_NAME for name in names}
    if role == "query":
        allowed = {FROZEN_QUERY_INPUTS_NAME, QUERY_MANIFEST_NAME}
    else:
        allowed = {DATABASE_NAME, INDEX_NAME, DERIVED_ORIGIN_NAME}
    if not values <= allowed or (final and values != allowed):
        raise DiffuseBaseArtifactError(f"{role} publication root changed schema")
    return {name: False for name in names}


def _store_kinds(names: tuple[str, ...], *, final: bool) -> dict[str, bool]:
    values = set(names)
    allowed = {*_SQLITE_PARTIAL_NAMES, INDEX_NAME}
    expected = {DATABASE_NAME, INDEX_NAME}
    if (
        len(values) != len(names)
        or not values <= allowed
        or (final and values != expected)
    ):
        raise DiffuseBaseArtifactError("store publication child changed schema")
    return {name: False for name in names}


def _scan(
    state: _State,
    *,
    final: bool,
    keep_handles: bool,
) -> tuple[_OpenEntry, _OpenEntry | None, tuple[_InventoryEntry, ...], tuple[_OpenEntry, ...]]:
    opened: list[_OpenEntry] = []
    new_store: _OpenEntry | None = None
    try:
        parent = _assert_parent_chain(state)
        _assert_bound(parent, state.path.name, state.root)
        root_names = _entry_names(state.root, cap=3)
        root_kinds = _root_kinds(state.role, root_names, final=final)
        inventory: list[_InventoryEntry] = []
        for name in sorted(root_names):
            directory = root_kinds[name]
            entry = _open_child(state.root, name, directory=directory)
            opened.append(entry)
            if directory:
                if state.store_child is not None and not _same_object(
                    entry.identity, state.store_child.identity
                ):
                    raise DiffuseBaseArtifactError(
                        "precreated store child was replaced by the factory"
                    )
                child_names = _entry_names(entry, cap=6)
                child_kinds = _store_kinds(child_names, final=final)
                inventory.append(
                    _InventoryEntry((name,), True, _current_identity(entry))
                )
                for child_name in sorted(child_names):
                    child = _open_child(
                        entry,
                        child_name,
                        directory=child_kinds[child_name],
                    )
                    opened.append(child)
                    _require_one_link(child)
                    inventory.append(
                        _InventoryEntry(
                            (name, child_name),
                            False,
                            _current_identity(child),
                        )
                    )
                new_store = _refreshed(entry)
            else:
                _require_one_link(entry)
                inventory.append(
                    _InventoryEntry((name,), False, _current_identity(entry))
                )
        if state.role == "store" and final and new_store is None:
            raise DiffuseBaseArtifactError("store publication has no store child")
        refreshed_root = _refreshed(state.root)
        kept = tuple(opened) if keep_handles else ()
        if keep_handles:
            opened = []
        return refreshed_root, new_store, tuple(inventory), kept
    finally:
        _close_entries(tuple(opened))


def _same_inventory(
    left: tuple[_InventoryEntry, ...],
    right: tuple[_InventoryEntry, ...],
) -> bool:
    return left == right


def create_publication(
    target: str | Path,
    *,
    role: PublicationRole,
) -> OwnedBasePublication:
    """Create a random sibling and issue its sole live ownership capability."""

    if role not in {"store", "query", "derived"}:
        raise ValueError("publication role must be store, query, or derived")
    target_path = _absolute_child(target)
    parent_path = require_plain_parent(target_path.parent)
    chain = _open_chain(parent_path)
    chain_owned = False
    root: _OpenEntry | None = None
    owner: OwnedBasePublication | None = None
    try:
        parent = chain[-1]
        if os.path.lexists(target_path):
            raise FileExistsError(target_path)
        prefix = f".{target_path.name}.{role}-publish-"
        for _ in range(128):
            name = prefix + secrets.token_hex(16)
            try:
                root = _mkdir_child(parent, name, delete_access=os.name == "nt")
            except FileExistsError:
                continue
            break
        if root is None:
            raise FileExistsError("cannot allocate a unique publication root")
        owner = _issue(
            role=role,
            phase="created",
            path=root.path,
            parent=parent_path,
            target_name=target_path.name,
            prefix=prefix,
            root=root,
            parent_chain=tuple(chain),
        )
        chain_owned = True
        root = None
        if role == "store":
            state = _state(owner)
            child = _mkdir_child(
                state.root,
                STORE_DIRECTORY_NAME,
                held_parent_write=True,
            )
            _replace(owner, state, store_child=child)
        return owner
    except BaseException:
        if owner is not None:
            try:
                rollback_publication(owner)
            except BaseException:
                abandon_publication(owner)
        raise
    finally:
        if root is not None:
            _close_entry(root)
        if not chain_owned:
            _close_entries(tuple(chain))


def write_publication_bytes(
    owner: OwnedBasePublication,
    relative: str,
    payload: bytes,
) -> None:
    """Exclusively write an allowed root file relative to the held capability."""

    state = _state(owner)
    if state.phase != "created":
        raise TypeError("publication writes require the created phase")
    allowed = {
        "store": {STORE_MANIFEST_NAME},
        "query": {FROZEN_QUERY_INPUTS_NAME, QUERY_MANIFEST_NAME},
        "derived": {DERIVED_ORIGIN_NAME},
    }[state.role]
    if relative not in allowed:
        raise DiffuseBaseArtifactError("publication write is outside its role schema")
    entry: _OpenEntry | None = None
    try:
        _assert_bound(_assert_parent_chain(state), state.path.name, state.root)
        entry = _write_child(
            state.root,
            relative,
            payload,
            limit=_MAX_METADATA_BYTES,
        )
    finally:
        if entry is not None:
            _close_entry(entry)


def copy_publication_file(
    owner: OwnedBasePublication,
    source: str | Path,
    relative: str,
) -> None:
    """Copy one held regular source into an exclusive derived-root child."""

    state = _state(owner)
    if state.role != "derived" or state.phase != "created":
        raise TypeError("publication copies require a created derived capability")
    if relative not in {DATABASE_NAME, INDEX_NAME}:
        raise DiffuseBaseArtifactError("derived copy is outside its role schema")
    source_path = _absolute_child(source)
    source_chain = _open_chain(source_path.parent)
    source_entry: _OpenEntry | None = None
    destination: _OpenEntry | None = None
    try:
        source_entry = _open_child(source_chain[-1], source_path.name, directory=False)
        _require_one_link(source_entry)
        _assert_bound(source_chain[-1], source_path.name, source_entry)
        if source_entry.size < 1 or source_entry.size > _MAX_COPY_BYTES:
            raise DiffuseBaseArtifactError("derived copy source violates its byte bound")
        _assert_bound(_assert_parent_chain(state), state.path.name, state.root)
        destination, _source_digest = _copy_entry_to_child(
            state.root, relative, source_entry
        )
        if _same_object(source_entry.identity, destination.identity):
            raise DiffuseBaseArtifactError("derived copy must not hardlink its source")
        if _current_identity(source_entry) != source_entry.identity:
            raise DiffuseBaseArtifactError("derived copy source changed during acquisition")
        _assert_bound(source_chain[-1], source_path.name, source_entry)
    finally:
        if destination is not None:
            _close_entry(destination)
        if source_entry is not None:
            _close_entry(source_entry)
        _close_entries(tuple(source_chain))


def capture_publication(owner: OwnedBasePublication) -> None:
    """Capture the complete role schema and retain every object identity."""

    state = _state(owner)
    if state.phase == "captured":
        assert_publication_unchanged(owner)
        return
    if state.phase != "created":
        raise TypeError("publication capture requires the created phase")
    root, store, inventory, held = _scan(state, final=True, keep_handles=True)
    old_handles = (
        *state.held,
        *((state.store_child,) if state.store_child is not None else ()),
    )
    updated = _replace(
        owner,
        state,
        phase="captured",
        root=root,
        store_child=store,
        inventory=inventory,
        held=held,
    )
    del updated
    _close_entries(old_handles)


def assert_publication_unchanged(owner: OwnedBasePublication) -> None:
    state = _state(owner)
    if state.phase not in {"captured", "promoted"}:
        raise TypeError("publication integrity requires a captured tree")
    root, _store, inventory, _held = _scan(
        state,
        final=True,
        keep_handles=False,
    )
    if root.identity != state.root.identity or not _same_inventory(
        inventory, state.inventory
    ):
        raise DiffuseBaseArtifactError("captured publication changed")
    if state.marker is not None:
        _validate_marker_entry(
            _assert_parent_chain(state), state.target_name, state.marker
        )


def _close_captured_children(owner: OwnedBasePublication, state: _State) -> _State:
    _close_entries(state.held)
    if state.store_child is not None and all(
        state.store_child.handle != item.handle for item in state.held
    ):
        _close_entry(state.store_child)
    return _replace(owner, state, held=(), store_child=None)


def _rename_root(
    owner: OwnedBasePublication,
    state: _State,
    target: Path,
    *,
    phase: _PublicationPhase,
) -> _State:
    if target.parent != state.parent or os.path.lexists(target):
        raise FileExistsError(target)
    renamed = False
    try:
        parent = _assert_parent_chain(state)
        _assert_bound(parent, state.path.name, state.root)
        if os.name == "nt":
            _win_rename(state.root, parent, target.name)
        else:
            _posix_rename_noreplace(parent, state.path.name, target.name)
        renamed = True
        moved_root = _refreshed(state.root, path=target)
        try:
            updated = _replace(
                owner,
                state,
                path=target,
                prefix=target.name,
                phase=phase,
                root=moved_root,
            )
        except BaseException:
            # The registry transition is intended to be infallible, but exact
            # rename-back keeps the old live capability truthful under an
            # injected failure or concurrent misuse.
            if os.name == "nt":
                _win_rename(moved_root, parent, state.path.name)
            else:
                _posix_rename_noreplace(parent, target.name, state.path.name)
            renamed = False
            _flush_open_directory(parent)
            raise
        if os.name == "nt":
            if not _kernel32.FlushFileBuffers(moved_root.handle):
                raise _win_error("cannot flush promoted publication directory")
        else:
            _flush_open_directory(moved_root)
        _flush_open_directory(parent)
        return updated
    except BaseException:
        # After the registry transition, the live state already carries the
        # renamed pathname; its caller can roll that exact object back.  Only
        # pre-transition failures reach here with ``renamed`` false/true.
        if renamed and _state(owner) is state:
            parent = _assert_parent_chain(state)
            if os.name == "nt":
                _win_rename(_refreshed(state.root, path=target), parent, state.path.name)
            else:
                _posix_rename_noreplace(parent, target.name, state.path.name)
            _flush_open_directory(parent)
        raise


def promote_publication(owner: OwnedBasePublication) -> Path:
    """Atomically rename a captured sibling to its no-replace target name."""

    state = _state(owner)
    if state.phase != "captured":
        raise TypeError("publication promotion requires the captured phase")
    assert_publication_unchanged(owner)
    state = _state(owner)
    target = state.parent / state.target_name
    if os.path.lexists(target):
        raise FileExistsError(target)
    expected = state.inventory
    state = _close_captured_children(owner, state)
    promoted = _rename_root(owner, state, target, phase="promoted")
    marker: _OpenEntry | None = None
    try:
        marker, marker_owned = _open_publication_marker(
            _assert_parent_chain(promoted), promoted.target_name
        )
        promoted = _replace(
            owner,
            promoted,
            marker=marker,
            marker_owned=marker_owned,
        )
        marker = None
    finally:
        if marker is not None:
            _close_entry(marker)
    root, store, inventory, held = _scan(
        promoted,
        final=True,
        keep_handles=True,
    )
    if not _same_inventory(inventory, expected):
        _close_entries(held)
        raise DiffuseBaseArtifactError("atomic publication changed its object inventory")
    _replace(
        owner,
        promoted,
        root=root,
        store_child=store,
        inventory=inventory,
        held=held,
    )
    assert_publication_unchanged(owner)
    return target


def commit_publication(owner: OwnedBasePublication) -> Path:
    """Release ownership only after the caller's high-level verification."""

    state = _state(owner)
    if state.phase != "promoted":
        raise TypeError("publication commit requires the promoted phase")
    assert_publication_unchanged(owner)
    state = _state(owner)
    result = state.path
    handles = [*state.held]
    if state.store_child is not None:
        handles.append(state.store_child)
    handles.append(state.root)
    if state.marker is not None:
        handles.append(state.marker)
    handles.extend(state.parent_chain)
    _close_entries(tuple(handles))
    _revoke(owner, state)
    return result


def _inventory_map(state: _State) -> dict[tuple[str, ...], _InventoryEntry]:
    return {item.relative: item for item in state.inventory}


def _delete_file(parent: _OpenEntry, name: str, expected: _InventoryEntry) -> None:
    probe = _open_child(parent, name, directory=False)
    try:
        _require_one_link(probe)
        if _current_identity(probe) != expected.identity:
            raise DiffuseBaseArtifactError("owned cleanup file was replaced")
    finally:
        _close_entry(probe)
    entry = _open_child(
        parent,
        name,
        directory=False,
        delete_access=os.name == "nt",
    )
    try:
        _require_one_link(entry)
        if _current_identity(entry) != expected.identity:
            raise DiffuseBaseArtifactError("owned cleanup file changed before delete")
        if os.name == "nt":
            _win_mark_delete(entry)
        else:
            os.unlink(name, dir_fd=parent.handle)
    finally:
        _close_entry(entry)


def _delete_tombstone(owner: OwnedBasePublication) -> None:
    state = _state(owner)
    inventory = _inventory_map(state)
    root_names = _entry_names(state.root, cap=3)
    _root_kinds(state.role, root_names, final=False)
    if set(root_names) != {path[0] for path in inventory}:
        raise DiffuseBaseArtifactError("cleanup tombstone changed after capture")
    if state.role == "store" and (STORE_DIRECTORY_NAME,) in inventory:
        store_expected = inventory[(STORE_DIRECTORY_NAME,)]
        store = _open_child(
            state.root,
            STORE_DIRECTORY_NAME,
            directory=True,
            delete_access=os.name == "nt",
        )
        try:
            if _current_identity(store) != store_expected.identity:
                raise DiffuseBaseArtifactError("cleanup store child was replaced")
            names = _entry_names(store, cap=6)
            _store_kinds(names, final=False)
            if set(names) != {
                path[1] for path in inventory if len(path) == 2
            }:
                raise DiffuseBaseArtifactError("cleanup store inventory changed")
            for name in names:
                _delete_file(store, name, inventory[(STORE_DIRECTORY_NAME, name)])
            if os.name == "nt":
                _win_mark_delete(store)
            else:
                os.rmdir(STORE_DIRECTORY_NAME, dir_fd=state.root.handle)
        finally:
            _close_entry(store)
    for name in root_names:
        if name == STORE_DIRECTORY_NAME:
            continue
        _delete_file(state.root, name, inventory[(name,)])
    root_closed = False
    try:
        parent = _assert_parent_chain(state)
        _assert_bound(parent, state.path.name, state.root)
        if os.name == "nt":
            _win_mark_delete(state.root)
            _close_entry(state.root)
            root_closed = True
        else:
            os.rmdir(state.path.name, dir_fd=parent.handle)
        _flush_open_directory(parent)
    finally:
        if not root_closed:
            _close_entry(state.root)


def _remove_owned_marker(owner: OwnedBasePublication) -> None:
    state = _state(owner)
    marker = state.marker
    if marker is None or not state.marker_owned:
        return
    parent = _assert_parent_chain(state)
    _validate_marker_entry(parent, state.target_name, marker)
    name = _marker_name(state.target_name)
    expected = marker.identity
    if os.name == "nt":
        _close_entry(marker)
        state = _replace(owner, state, marker=None, marker_owned=False)
        deleting = _open_child(
            parent, name, directory=False, delete_access=True
        )
        try:
            _require_one_link(deleting)
            if _current_identity(deleting) != expected:
                raise DiffuseBaseArtifactError(
                    "owned publication marker changed before cleanup"
                )
            _win_mark_delete(deleting)
        finally:
            _close_entry(deleting)
    else:
        os.unlink(name, dir_fd=parent.handle)
        _close_entry(marker)
        state = _replace(owner, state, marker=None, marker_owned=False)
    _flush_open_directory(parent)


def rollback_publication(owner: OwnedBasePublication) -> None:
    """Detach and delete only the exact capability-owned bounded inventory."""

    state = _state(owner)
    if state.phase == "created":
        old_store = state.store_child
        root, store, inventory, held = _scan(
            state,
            final=False,
            keep_handles=True,
        )
        state = _replace(
            owner,
            state,
            root=root,
            store_child=store,
            inventory=inventory,
            held=held,
        )
        if old_store is not None:
            _close_entry(old_store)
    elif state.phase in {"captured", "promoted"}:
        assert_publication_unchanged(owner)
        state = _state(owner)
    else:
        raise TypeError("publication is already a cleanup tombstone")
    state = _close_captured_children(owner, state)
    tomb = state.parent / (
        f".{state.target_name}.{state.role}-cleanup-{secrets.token_hex(16)}"
    )
    tombstone = _rename_root(owner, state, tomb, phase="tombstone")
    try:
        root, store, inventory, held = _scan(
            tombstone,
            final=False,
            keep_handles=False,
        )
        if not _same_inventory(inventory, tombstone.inventory):
            raise DiffuseBaseArtifactError("cleanup tombstone inventory changed")
        tombstone = _replace(
            owner,
            tombstone,
            root=root,
            store_child=store,
            held=held,
        )
        _delete_tombstone(owner)
    except BaseException:
        raise
    _remove_owned_marker(owner)
    completed = _state(owner)
    if completed.marker is not None:
        _close_entry(completed.marker)
    _close_entries(completed.parent_chain)
    _revoke(owner, completed)


def abandon_publication(owner: OwnedBasePublication) -> Path:
    """Close/revoke a refused cleanup while leaving every byte quarantined."""

    state = _state(owner)
    result = state.path
    handles = [*state.held]
    if state.store_child is not None:
        handles.append(state.store_child)
    handles.append(state.root)
    if state.marker is not None:
        handles.append(state.marker)
    handles.extend(state.parent_chain)
    _close_entries(tuple(handles))
    _revoke(owner, state)
    return result


publication_operation_guard = freeze_operation_guard(
    globals(),
    primitive_namespace=_close_entry.__globals__,
    state_op=_state,
    revoke_op=_revoke,
    raw_close=_kernel32.CloseHandle if os.name == "nt" else os.close,
    windows=os.name == "nt",
    error_type=DiffuseBaseArtifactError,
    attribute_dependencies=(
        (os, "path"), (os, "name"),
        *((os, name) for name in (
            "close", "fsync", "fstat", "lseek", "mkdir", "open",
            "read", "rmdir", "scandir", "stat", "unlink", "write",
            "fsencode", "strerror",
        )),
        *((os.path, name) for name in (
            "abspath", "lexists", "normcase", "normpath",
        )),
        (Path, "mkdir"),
        (hashlib, "sha256"),
        (secrets, "token_hex"),
        *((ctypes, name) for name in (
            "byref", "create_string_buffer", "create_unicode_buffer",
            "get_last_error", "addressof", "cast", "memmove", "POINTER",
            "sizeof",
        ) if hasattr(ctypes, name)),
        *(() if os.name != "nt" else tuple(
            (_kernel32, name) for name in (
                "CloseHandle", "CreateFileW", "FlushFileBuffers", "ReadFile",
                "SetFilePointerEx", "WriteFile", "SetFileInformationByHandle",
                "GetFileInformationByHandle", "GetFinalPathNameByHandleW",
            )
        )),
    ),
)


__all__ = [
    "OwnedBasePublication",
    "PublicationRole",
    "abandon_publication",
    "assert_publication_unchanged",
    "capture_publication",
    "commit_publication",
    "copy_publication_file",
    "create_publication",
    "prepare_derived_publication_handoff",
    "promote_publication",
    "publication_operation_guard",
    "publication_path",
    "rollback_publication",
    "validate_publication_lock_marker",
    "write_publication_bytes",
]
