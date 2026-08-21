"""Held-handle authority for one writable diffuse derived-store lifecycle."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from threading import Event, RLock
from typing import Callable, Literal
import weakref

_EVENT_SET = Event.set
_EVENT_WAIT = Event.wait

from memory_condense.eval._diffuse_base_contracts import (
    DATABASE_NAME,
    DERIVED_FINALIZATION_NAME,
    DERIVED_LEASE_NAME,
    DERIVED_ORIGIN_NAME,
    INDEX_NAME,
    DiffuseBaseArtifactError,
    DiffuseDerivedStore,
)
from memory_condense.eval._diffuse_base_publication_filesystem import (
    OwnedBasePublication,
    prepare_derived_publication_handoff,
)
import memory_condense.eval._diffuse_base_derived_lifecycle_native as _lifecycle_native
from memory_condense.eval._diffuse_base_derived_lifecycle_guard import (
    freeze_derived_lifecycle_guard,
)
from memory_condense.eval._diffuse_base_derived_lifecycle_native import (
    MAX_DATABASE_BYTES as _MAX_DATABASE_BYTES,
    MAX_METADATA_BYTES as _MAX_METADATA_BYTES,
    assert_bound as _assert_bound,
    close_committed_lifecycle as _close_committed_lifecycle,
    close_entries as _close_entries,
    create_atomic_finalization as _create_atomic_finalization,
    create_held_child as _create_held_child,
    current_identity as _current_identity,
    discard_created_child as _discard_created_child,
    duplicate_entry as _duplicate_entry,
    flush_directory as _flush_directory,
    hash_entry as _hash_entry,
    open_writable_child as _open_writable_child,
    read_exact as _read_exact,
    require_one_link as _require_one_link,
    write_exact as _write_exact,
)
from memory_condense.eval._diffuse_latent_training_corpus_filesystem import (
    _OpenEntry,
    _close_entry,
    _entry_names,
    _open_chain,
    _open_child,
    _posix_flags,
    _posix_identity,
    _same_object,
    _win_info,
    _win_open,
)

if os.name == "nt":
    from ctypes import wintypes

    from memory_condense.eval._diffuse_latent_training_corpus_filesystem import (
        _kernel32,
        _win_close,
        _win_error,
    )

    _kernel32.SetEndOfFile.argtypes = (wintypes.HANDLE,)
    _kernel32.SetEndOfFile.restype = wintypes.BOOL
    _kernel32.GetCurrentProcess.argtypes = ()
    _kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    _kernel32.DuplicateHandle.argtypes = (
        wintypes.HANDLE,
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    )
    _kernel32.DuplicateHandle.restype = wintypes.BOOL


_Phase = Literal["claimed", "open", "closed", "finalizing"]
_INITIAL_NAMES = {DATABASE_NAME, INDEX_NAME, DERIVED_ORIGIN_NAME}


class OwnedDerivedLifecycle:
    """Opaque registry-issued authority over one derived writable lifecycle."""

    __slots__ = ("_token",)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("derived lifecycle capabilities cannot be constructed")

    def __setattr__(self, _name: str, _value: object) -> None:
        raise TypeError("derived lifecycle capabilities are immutable")

    def __copy__(self) -> object:
        raise TypeError("derived lifecycle capabilities cannot be copied")

    def __deepcopy__(self, _memo: object) -> object:
        raise TypeError("derived lifecycle capabilities cannot be copied")

    def __reduce_ex__(self, _protocol: int) -> object:
        raise TypeError("derived lifecycle capabilities cannot be serialized")


@dataclass(frozen=True, slots=True)
class DerivedLifecycleFiles:
    database_bytes: bytes
    database_sha256: str
    database_size: int
    index_sha256: str
    index_size: int
    origin_bytes: bytes
    lease_bytes: bytes
    finalization_bytes: bytes | None


@dataclass(frozen=True, slots=True)
class _Registration:
    clone_id: int
    clone_ref: weakref.ReferenceType[DiffuseDerivedStore]
    finalizer: weakref.finalize
    path: Path
    clone_path: Path
    clone_origin: object
    clone_base: object
    parent_chain: tuple[_OpenEntry, ...]
    root: _OpenEntry
    inventory: tuple[tuple[str, tuple[int, ...]], ...]


@dataclass(frozen=True, slots=True)
class _State:
    owner: OwnedDerivedLifecycle
    clone_id: int
    clone_ref: weakref.ReferenceType[DiffuseDerivedStore]
    clone_finalizer: weakref.finalize
    operation_lock: RLock
    phase: _Phase
    path: Path
    clone_path: Path
    clone_origin: object
    clone_base: object
    parent_chain: tuple[_OpenEntry, ...]
    root: _OpenEntry
    database: _OpenEntry
    index: _OpenEntry
    origin: _OpenEntry
    database_bytes: bytes
    index_sha256: str
    origin_bytes: bytes
    lease: _OpenEntry | None = None
    lease_bytes: bytes = b""
    finalization: _OpenEntry | None = None
    finalization_bytes: bytes | None = None
    held: tuple[_OpenEntry, ...] = ()
    store_child: None = None
    marker: None = None


@dataclass(slots=True)
class _Claiming:
    clone_ref: weakref.ReferenceType[DiffuseDerivedStore]
    done: Event
    path: Path
    cancelled: bool = False


def _registry_boundary() -> tuple[Callable[..., object], ...]:
    registry: dict[object, _State] = {}
    by_clone: dict[int, OwnedDerivedLifecycle] = {}
    registrations: dict[int, _Registration] = {}
    claiming: dict[int, _Claiming] = {}
    lock = RLock()
    capability_type = OwnedDerivedLifecycle
    windows = os.name == "nt"
    if windows:
        finalizer_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        stable_raw_close = finalizer_kernel32.CloseHandle
        stable_raw_close.argtypes = (wintypes.HANDLE,)
        stable_raw_close.restype = wintypes.BOOL
    else:
        stable_raw_close = os.close

    def state(value: OwnedDerivedLifecycle) -> _State:
        if type(value) is not capability_type:
            raise TypeError("operation requires an issued derived capability")
        try:
            token = object.__getattribute__(value, "_token")
        except (AttributeError, TypeError) as exc:
            raise TypeError("derived lifecycle capability is not issued") from exc
        with lock:
            result = registry.get(token)
            if result is None or result.owner is not value:
                raise TypeError("derived lifecycle capability is not live")
            return result

    def issue(**fields: object) -> OwnedDerivedLifecycle:
        clone = fields.pop("clone")
        claim = fields.pop("claim")
        assert isinstance(clone, DiffuseDerivedStore)
        assert isinstance(claim, _Claiming)
        owner = object.__new__(OwnedDerivedLifecycle)
        token = object()
        object.__setattr__(owner, "_token", token)

        def release_claimed() -> None:
            with lock:
                initial = registry.get(token)
            if initial is None:
                return
            operation_lock = initial.operation_lock
            operation_lock.acquire()
            try:
                with lock:
                    abandoned = registry.get(token)
                    if abandoned is None or abandoned.owner is not owner:
                        return
                    del registry[token]
                    if by_clone.get(abandoned.clone_id) is owner:
                        del by_clone[abandoned.clone_id]
                seen: set[int] = set()
                for entry in reversed((*abandoned.held, *abandoned.parent_chain)):
                    if entry.handle in seen:
                        continue
                    try:
                        stable_raw_close(entry.handle)
                    except BaseException:
                        pass
                    seen.add(entry.handle)
            finally:
                operation_lock.release()

        finalizer = weakref.finalize(clone, release_claimed)
        created = _State(
            owner=owner,
            clone_id=id(clone),
            clone_ref=weakref.ref(clone),
            clone_finalizer=finalizer,
            operation_lock=fields.pop("operation_lock"),  # type: ignore[arg-type]
            **fields,
        )  # type: ignore[arg-type]
        with lock:
            if (
                claiming.get(id(clone)) is not claim
                or claim.cancelled
                or id(clone) in by_clone
                or id(clone) in registrations
            ):
                raise DiffuseBaseArtifactError("derived clone claim was aborted")
            del claiming[id(clone)]
            registry[token] = created
            by_clone[id(clone)] = owner
        _EVENT_SET(claim.done)
        return owner

    def for_clone(clone: DiffuseDerivedStore) -> OwnedDerivedLifecycle:
        if type(clone) is not DiffuseDerivedStore:
            raise TypeError("clone must be an exact DiffuseDerivedStore")
        with lock:
            owner = by_clone.get(id(clone))
            if owner is None:
                raise DiffuseBaseArtifactError("derived clone has no live ownership")
            current = registry.get(object.__getattribute__(owner, "_token"))
            if current is None or current.clone_ref() is not clone:
                raise DiffuseBaseArtifactError("derived clone ownership was forged")
            return owner

    def register(
        clone: DiffuseDerivedStore,
        path: Path,
        parent_chain: tuple[_OpenEntry, ...],
        root: _OpenEntry,
        inventory: tuple[tuple[str, tuple[int, ...]], ...],
    ) -> None:
        if type(clone) is not DiffuseDerivedStore:
            raise TypeError("clone must be an exact DiffuseDerivedStore")
        with lock:
            if (
                id(clone) in registrations
                or id(clone) in claiming
                or id(clone) in by_clone
            ):
                raise DiffuseBaseArtifactError("derived clone is already registered")
            clone_id = id(clone)

            def release_unclaimed() -> None:
                with lock:
                    abandoned = registrations.pop(clone_id, None)
                if abandoned is not None:
                    seen: set[int] = set()
                    for entry in reversed(
                        (abandoned.root, *abandoned.parent_chain)
                    ):
                        if entry.handle in seen:
                            continue
                        try:
                            stable_raw_close(entry.handle)
                        except BaseException:
                            pass
                        seen.add(entry.handle)

            finalizer = weakref.finalize(clone, release_unclaimed)
            created = _Registration(
                clone_id,
                weakref.ref(clone),
                finalizer,
                path,
                clone.path,
                clone.origin,
                clone.base,
                parent_chain,
                root,
                inventory,
            )
            registrations[id(clone)] = created

    def take(clone: DiffuseDerivedStore) -> tuple[_Registration, _Claiming]:
        if type(clone) is not DiffuseDerivedStore:
            raise TypeError("clone must be an exact DiffuseDerivedStore")
        with lock:
            result = registrations.get(id(clone))
            if result is None or result.clone_ref() is not clone:
                raise DiffuseBaseArtifactError("derived clone has no live ownership")
            del registrations[id(clone)]
            claim = _Claiming(weakref.ref(clone), Event(), result.path)
            claiming[id(clone)] = claim
        return result, claim

    def finish_claim(clone: DiffuseDerivedStore, claim: _Claiming) -> None:
        with lock:
            if claiming.get(id(clone)) is claim:
                del claiming[id(clone)]
        _EVENT_SET(claim.done)

    def discard(clone: DiffuseDerivedStore) -> _Registration:
        if type(clone) is not DiffuseDerivedStore:
            raise TypeError("clone must be an exact DiffuseDerivedStore")
        with lock:
            result = registrations.get(id(clone))
            if result is None or result.clone_ref() is not clone:
                raise DiffuseBaseArtifactError("derived clone has no registration")
            del registrations[id(clone)]
        return result

    def emergency_abandon(owner: OwnedDerivedLifecycle) -> Path:
        if type(owner) is not capability_type:
            raise TypeError("operation requires an issued derived capability")
        try:
            token = object.__getattribute__(owner, "_token")
        except (AttributeError, TypeError) as exc:
            raise TypeError("derived lifecycle capability is not issued") from exc
        with lock:
            initial = registry.get(token)
            if initial is None or initial.owner is not owner:
                raise TypeError("derived lifecycle capability is not live")
            operation_lock = initial.operation_lock
            path = initial.path
            retained_clone = initial.clone_ref()
            if retained_clone is None:
                raise DiffuseBaseArtifactError("derived clone was abandoned")
        operation_lock.acquire()
        failure: BaseException | None = None
        try:
            with lock:
                abandoned = registry.get(token)
                if abandoned is None:
                    return path
                if (
                    abandoned.owner is not owner
                    or abandoned.operation_lock is not operation_lock
                ):
                    raise TypeError("derived lifecycle changed during quarantine")
                del registry[token]
                if by_clone.get(abandoned.clone_id) is owner:
                    del by_clone[abandoned.clone_id]
            seen: set[int] = set()
            for entry in reversed((*abandoned.held, *abandoned.parent_chain)):
                if entry.handle in seen:
                    continue
                try:
                    result = stable_raw_close(entry.handle)
                    if windows and not result:
                        raise DiffuseBaseArtifactError(
                            "cannot close quarantined derived handle"
                        )
                except BaseException as exc:
                    if failure is None:
                        failure = exc
                seen.add(entry.handle)
        finally:
            operation_lock.release()
        if failure is not None:
            raise failure
        del retained_clone
        return path

    def emergency_take_registration(clone: DiffuseDerivedStore) -> Path:
        if type(clone) is not DiffuseDerivedStore:
            raise TypeError("clone must be an exact DiffuseDerivedStore")
        safe_path: Path | None = None
        while True:
            wait_for: Event | None = None
            entries: tuple[_OpenEntry, ...] | None = None
            owner: OwnedDerivedLifecycle | None = None
            token: object | None = None
            operation_lock: RLock | None = None
            with lock:
                active_claim = claiming.get(id(clone))
                if active_claim is not None:
                    if active_claim.clone_ref() is not clone:
                        raise DiffuseBaseArtifactError(
                            "derived clone claim was forged"
                        )
                    active_claim.cancelled = True
                    wait_for = active_claim.done
                    safe_path = active_claim.path
                else:
                    result = registrations.get(id(clone))
                    if result is not None and result.clone_ref() is clone:
                        del registrations[id(clone)]
                        safe_path = result.path
                        entries = (result.root, *result.parent_chain)
                    else:
                        owner = by_clone.get(id(clone))
                    if entries is None and owner is None:
                        if safe_path is not None:
                            return safe_path
                        raise DiffuseBaseArtifactError(
                            "derived clone has no live lifecycle"
                        )
                    if owner is not None:
                        token = object.__getattribute__(owner, "_token")
                        state = registry.get(token)
                        if state is None or state.clone_ref() is not clone:
                            raise DiffuseBaseArtifactError(
                                "derived clone ownership was forged"
                            )
                        safe_path = state.path
                        operation_lock = state.operation_lock
            if wait_for is not None:
                _EVENT_WAIT(wait_for)
                continue
            if entries is not None:
                failure: BaseException | None = None
                seen: set[int] = set()
                for entry in reversed(entries):
                    if entry.handle in seen:
                        continue
                    try:
                        result = stable_raw_close(entry.handle)
                        if windows and not result:
                            raise DiffuseBaseArtifactError(
                                "cannot close quarantined registration handle"
                            )
                    except BaseException as exc:
                        if failure is None:
                            failure = exc
                    seen.add(entry.handle)
                if failure is not None:
                    raise failure
                assert safe_path is not None
                return safe_path
            assert owner is not None
            assert token is not None
            assert operation_lock is not None
            operation_lock.acquire()
            failure = None
            try:
                with lock:
                    abandoned = registry.get(token)
                    if abandoned is None:
                        assert safe_path is not None
                        return safe_path
                    if (
                        abandoned.owner is not owner
                        or abandoned.clone_ref() is not clone
                        or abandoned.operation_lock is not operation_lock
                    ):
                        raise DiffuseBaseArtifactError(
                            "derived clone ownership changed during quarantine"
                        )
                    del registry[token]
                    if by_clone.get(id(clone)) is owner:
                        del by_clone[id(clone)]
                seen = set()
                for entry in reversed((*abandoned.held, *abandoned.parent_chain)):
                    if entry.handle in seen:
                        continue
                    try:
                        result = stable_raw_close(entry.handle)
                        if windows and not result:
                            raise DiffuseBaseArtifactError(
                                "cannot close quarantined derived handle"
                            )
                    except BaseException as exc:
                        if failure is None:
                            failure = exc
                    seen.add(entry.handle)
            finally:
                operation_lock.release()
            if failure is not None:
                raise failure
            assert safe_path is not None
            return safe_path

    def take_for_abort(
        clone: DiffuseDerivedStore,
    ) -> tuple[str, object | None]:
        if type(clone) is not DiffuseDerivedStore:
            raise TypeError("clone must be an exact DiffuseDerivedStore")
        while True:
            wait_for: Event | None = None
            with lock:
                active_claim = claiming.get(id(clone))
                if active_claim is not None:
                    if active_claim.clone_ref() is not clone:
                        raise DiffuseBaseArtifactError(
                            "derived clone claim was forged"
                        )
                    active_claim.cancelled = True
                    wait_for = active_claim.done
                else:
                    owner = by_clone.get(id(clone))
                    if owner is not None:
                        state = registry.get(
                            object.__getattribute__(owner, "_token")
                        )
                        if state is None or state.clone_ref() is not clone:
                            raise DiffuseBaseArtifactError(
                                "derived clone ownership was forged"
                            )
                        return "owner", owner
                    registration = registrations.get(id(clone))
                    if registration is None:
                        return "none", None
                    if registration.clone_ref() is not clone:
                        raise DiffuseBaseArtifactError(
                            "derived clone registration was forged"
                        )
                    del registrations[id(clone)]
                    return "registration", registration
            assert wait_for is not None
            _EVENT_WAIT(wait_for)

    def replace(
        owner: OwnedDerivedLifecycle, expected: _State, **changes: object
    ) -> _State:
        token = object.__getattribute__(owner, "_token")
        with lock:
            current = registry.get(token)
            if current is not expected or current.owner is not owner:
                raise TypeError("derived lifecycle changed during operation")
            fields = {
                name: getattr(current, name)
                for name in _State.__dataclass_fields__
                if name != "owner"
            }
            fields.update(changes)
            updated = _State(owner=owner, **fields)  # type: ignore[arg-type]
            registry[token] = updated
            return updated

    def revoke(owner: OwnedDerivedLifecycle, expected: _State) -> None:
        token = object.__getattribute__(owner, "_token")
        with lock:
            current = registry.get(token)
            if current is not expected or current.owner is not owner:
                raise TypeError("derived lifecycle changed before revocation")
            del registry[token]
            if by_clone.get(current.clone_id) is owner:
                del by_clone[current.clone_id]

    return (
        issue,
        state,
        for_clone,
        register,
        take,
        discard,
        emergency_take_registration,
        emergency_abandon,
        take_for_abort,
        finish_claim,
        replace,
        revoke,
    )


(
    _issue,
    _state,
    _for_clone,
    _register,
    _take_registration,
    _discard_registration,
    _emergency_take_registration,
    _emergency_abandon_owner,
    _take_for_abort,
    _finish_claim,
    _replace,
    _revoke,
) = _registry_boundary()
del _registry_boundary


def _parent(state: _State) -> _OpenEntry:
    chain = state.parent_chain
    if not chain or chain[-1].path != state.path.parent:
        raise DiffuseBaseArtifactError("derived parent capability is malformed")
    for index in range(1, len(chain)):
        _assert_bound(chain[index - 1], chain[index].path.name, chain[index])
    return chain[-1]


def _expected_names(state: _State) -> set[str]:
    names = set(_INITIAL_NAMES)
    if state.lease is not None:
        names.add(DERIVED_LEASE_NAME)
    if state.finalization is not None:
        names.add(DERIVED_FINALIZATION_NAME)
    return names


def _assert_clone_fields(
    clone: DiffuseDerivedStore,
    path: Path,
    origin: object,
    base: object,
) -> None:
    if clone.path is not path or clone.origin is not origin or clone.base is not base:
        raise DiffuseBaseArtifactError("derived clone fields changed after issuance")


def _assert_unchanged(state: _State) -> None:
    clone = state.clone_ref()
    if clone is None:
        raise DiffuseBaseArtifactError("derived clone was abandoned")
    _assert_clone_fields(
        clone, state.clone_path, state.clone_origin, state.clone_base
    )
    parent = _parent(state)
    _assert_bound(parent, state.path.name, state.root)
    if set(_entry_names(state.root, cap=6)) != _expected_names(state):
        raise DiffuseBaseArtifactError("derived lifecycle tree changed")
    values = (
        (DATABASE_NAME, state.database, state.database_bytes, _MAX_DATABASE_BYTES),
        (DERIVED_ORIGIN_NAME, state.origin, state.origin_bytes, _MAX_METADATA_BYTES),
    )
    for name, entry, payload, limit in values:
        _assert_bound(state.root, name, entry)
        _require_one_link(entry)
        if _read_exact(entry, limit=limit) != payload:
            raise DiffuseBaseArtifactError(f"held derived {name} changed")
    _assert_bound(state.root, INDEX_NAME, state.index)
    _require_one_link(state.index)
    if _hash_entry(state.index) != state.index_sha256:
        raise DiffuseBaseArtifactError("held derived index changed")
    if state.lease is not None:
        _assert_bound(state.root, DERIVED_LEASE_NAME, state.lease)
        _require_one_link(state.lease)
        if _read_exact(state.lease, limit=_MAX_METADATA_BYTES) != state.lease_bytes:
            raise DiffuseBaseArtifactError("held derived lease changed")
    if state.finalization is not None:
        _assert_bound(state.root, DERIVED_FINALIZATION_NAME, state.finalization)
        _require_one_link(state.finalization)
        if _read_exact(
            state.finalization, limit=_MAX_METADATA_BYTES
        ) != state.finalization_bytes:
            raise DiffuseBaseArtifactError("held derived finalization changed")


def _terminal(
    owner: OwnedDerivedLifecycle,
    state: _State,
    *,
    discard_finalization: bool = False,
) -> None:
    failure: BaseException | None = None
    excluded_handle: int | None = None
    if discard_finalization and state.finalization is not None:
        excluded_handle = state.finalization.handle
        try:
            _discard_created_child(
                state.root,
                DERIVED_FINALIZATION_NAME,
                state.finalization,
            )
        except BaseException as exc:
            failure = exc
    terminal_entries = tuple(
        entry
        for entry in (*state.held, *state.parent_chain)
        if entry.handle != excluded_handle
    )
    try:
        _close_entries(terminal_entries)
    except BaseException as exc:
        if failure is None:
            failure = exc
        else:
            failure.add_note(f"derived handle close also failed: {exc!r}")
    try:
        _revoke(owner, state)
    except BaseException as exc:
        if failure is None:
            failure = exc
        else:
            failure.add_note(f"derived lifecycle revoke also failed: {exc!r}")
    if failure is not None:
        raise failure


def register_derived_publication(
    clone: DiffuseDerivedStore,
    publication: OwnedBasePublication,
) -> None:
    """Retain the exact promoted root before publication ownership commits."""

    path, borrowed_root, borrowed_chain, inventory = (
        prepare_derived_publication_handoff(publication)
    )
    if path != clone.path or set(dict(inventory)) != _INITIAL_NAMES:
        raise DiffuseBaseArtifactError("derived publication identity is malformed")
    chain: list[_OpenEntry] = []
    root: _OpenEntry | None = None
    try:
        for entry in borrowed_chain:
            chain.append(_duplicate_entry(entry))
        root = _duplicate_entry(borrowed_root)
        for index in range(1, len(chain)):
            _assert_bound(chain[index - 1], chain[index].path.name, chain[index])
        _assert_bound(chain[-1], path.name, root)
        _register(clone, path, tuple(chain), root, inventory)
        chain = []
        root = None
    except BaseException as original:
        leftovers = (() if root is None else (root,)) + tuple(chain)
        if leftovers:
            try:
                _close_entries(leftovers)
            except BaseException as cleanup_error:
                original.add_note(
                    f"derived handoff cleanup also failed: {cleanup_error!r}"
                )
        raise


def discard_derived_registration(clone: DiffuseDerivedStore) -> None:
    registration = _discard_registration(clone)
    _close_entries((registration.root, *registration.parent_chain))


def _abort_derived_lifecycle_for_clone(clone: DiffuseDerivedStore) -> None:
    """Atomically take and terminate any current-process clone lifecycle."""

    kind, value = _take_for_abort(clone)
    if kind == "none":
        return
    if kind == "owner":
        try:
            abort_derived_lifecycle(value)  # type: ignore[arg-type]
        except TypeError as exc:
            if str(exc) != "derived lifecycle capability is not live":
                raise
        return
    registration = value
    assert isinstance(registration, _Registration)
    _close_entries((registration.root, *registration.parent_chain))


@contextmanager
def _serialized_derived_lifecycle(owner: OwnedDerivedLifecycle):
    """Serialize every operation that can use or release owned handles."""

    release = _acquire_derived_lifecycle_operation(owner)
    try:
        yield
    finally:
        release()


def _acquire_derived_lifecycle_operation(
    owner: OwnedDerivedLifecycle,
) -> Callable[[], None]:
    initial = _state(owner)
    operation_lock = initial.operation_lock
    retained_clone = initial.clone_ref()
    if retained_clone is None:
        raise DiffuseBaseArtifactError("derived clone was abandoned")
    operation_lock.acquire()
    try:
        current = _state(owner)
        if current.operation_lock is not operation_lock:
            raise TypeError("derived lifecycle operation lock changed")
    except BaseException:
        operation_lock.release()
        raise
    def release() -> None:
        nonlocal retained_clone
        try:
            operation_lock.release()
        finally:
            retained_clone = None

    return release


def _release_claim_leftovers(
    root: _OpenEntry | None,
    chain: list[_OpenEntry],
    opened: list[_OpenEntry],
    lease: _OpenEntry | None,
) -> None:
    failure: BaseException | None = None
    lease_handle: int | None = None
    if lease is not None and root is not None:
        lease_handle = lease.handle
        try:
            _discard_created_child(root, DERIVED_LEASE_NAME, lease)
        except BaseException as exc:
            failure = exc
    leftovers = tuple(
        entry
        for entry in (
            *opened,
            *((() if root is None else (root,))),
            *chain,
        )
        if entry.handle != lease_handle
    )
    try:
        _close_entries(leftovers)
    except BaseException as exc:
        if failure is None:
            failure = exc
        else:
            failure.add_note(f"derived claim handle close also failed: {exc!r}")
    if failure is not None:
        raise failure


def claim_derived_lifecycle(
    clone: DiffuseDerivedStore,
    lease_payload: bytes,
) -> OwnedDerivedLifecycle:
    registration, claim = _take_registration(clone)
    expected = dict(registration.inventory)
    chain = list(registration.parent_chain)
    root: _OpenEntry | None = registration.root
    opened: list[_OpenEntry] = []
    lease: _OpenEntry | None = None
    owner: OwnedDerivedLifecycle | None = None
    operation_lock = RLock()
    operation_lock.acquire()
    try:
        _assert_clone_fields(
            clone,
            registration.clone_path,
            registration.clone_origin,
            registration.clone_base,
        )
        for index in range(1, len(chain)):
            _assert_bound(chain[index - 1], chain[index].path.name, chain[index])
        _assert_bound(chain[-1], registration.path.name, root)
        if set(_entry_names(root, cap=4)) != _INITIAL_NAMES:
            raise DiffuseBaseArtifactError("derived publication schema changed")
        database = _open_writable_child(root, DATABASE_NAME)
        opened.append(database)
        index = _open_child(root, INDEX_NAME, directory=False)
        opened.append(index)
        origin = _open_child(root, DERIVED_ORIGIN_NAME, directory=False)
        opened.append(origin)
        for name, entry in (
            (DATABASE_NAME, database),
            (INDEX_NAME, index),
            (DERIVED_ORIGIN_NAME, origin),
        ):
            _require_one_link(entry)
            if _current_identity(entry) != expected[name]:
                raise DiffuseBaseArtifactError(
                    "derived publication child changed before claim"
                )
        database_bytes = _read_exact(database, limit=_MAX_DATABASE_BYTES)
        origin_bytes = _read_exact(origin, limit=_MAX_METADATA_BYTES)
        index_sha256 = _hash_entry(index)
        lease = _create_held_child(root, DERIVED_LEASE_NAME, lease_payload)
        _flush_directory(root)
        held = (root, database, index, origin, lease)
        owner = _issue(
            clone=clone,
            claim=claim,
            operation_lock=operation_lock,
            phase="claimed",
            path=registration.path,
            clone_path=registration.clone_path,
            clone_origin=registration.clone_origin,
            clone_base=registration.clone_base,
            parent_chain=tuple(chain),
            root=root,
            database=database,
            index=index,
            origin=origin,
            database_bytes=database_bytes,
            index_sha256=index_sha256,
            origin_bytes=origin_bytes,
            lease=lease,
            lease_bytes=lease_payload,
            held=held,
        )
        chain = []
        root = None
        opened = []
        lease = None
        _assert_unchanged(_state(owner))
        return owner
    except FileExistsError as exc:
        error = DiffuseBaseArtifactError(
            "derived store has already been claimed for writable use"
        )
        try:
            _release_claim_leftovers(root, chain, opened, lease)
        except BaseException as cleanup_error:
            error.add_note(
                f"derived claim cleanup also failed: {cleanup_error!r}"
            )
        raise error from exc
    except BaseException as original:
        if owner is not None:
            try:
                _terminal(owner, _state(owner))
            except BaseException as cleanup_error:
                original.add_note(
                    f"derived claim quarantine also failed: {cleanup_error!r}"
                )
        else:
            try:
                _release_claim_leftovers(root, chain, opened, lease)
            except BaseException as cleanup_error:
                original.add_note(
                    f"derived claim cleanup also failed: {cleanup_error!r}"
                )
        raise
    finally:
        _finish_claim(clone, claim)
        operation_lock.release()


def _owned_database_image_locked(owner: OwnedDerivedLifecycle) -> bytes:
    state = _state(owner)
    if state.phase != "claimed":
        raise TypeError("derived database acquisition requires a claimed owner")
    _assert_unchanged(state)
    return state.database_bytes


def owned_database_image(owner: OwnedDerivedLifecycle) -> bytes:
    with _serialized_derived_lifecycle(owner):
        return _owned_database_image_locked(owner)


def owned_derived_lifecycle_for_clone(
    clone: DiffuseDerivedStore,
) -> OwnedDerivedLifecycle:
    """Return the live owner issued for this exact current-process clone."""

    return _for_clone(clone)


def _owned_index_load_path_locked(owner: OwnedDerivedLifecycle) -> Path:
    state = _state(owner)
    if state.phase != "claimed":
        raise TypeError("derived index acquisition requires a claimed owner")
    _assert_unchanged(state)
    if os.name == "nt":
        return state.index.path
    candidate = Path("/proc/self/fd") / str(state.index.handle)
    try:
        value = os.stat(candidate)
    except OSError:
        value = None
    if value is not None and _same_object(
        _posix_identity(value), state.index.identity
    ):
        return candidate
    raise DiffuseBaseArtifactError(
        "descriptor-bound HNSW loading is unavailable on this platform"
    )


def owned_index_load_path(owner: OwnedDerivedLifecycle) -> Path:
    with _serialized_derived_lifecycle(owner):
        return _owned_index_load_path_locked(owner)


def _mark_derived_open_locked(owner: OwnedDerivedLifecycle) -> None:
    state = _state(owner)
    if state.phase != "claimed":
        raise TypeError("derived open requires a claimed owner")
    _assert_unchanged(state)
    _replace(owner, state, phase="open")


def mark_derived_open(owner: OwnedDerivedLifecycle) -> None:
    with _serialized_derived_lifecycle(owner):
        _mark_derived_open_locked(owner)


def _close_derived_lifecycle_locked(
    owner: OwnedDerivedLifecycle,
    serialized_database: bytes,
) -> None:
    state = _state(owner)
    if state.phase != "open":
        raise TypeError("derived close requires an open owner")
    try:
        _assert_unchanged(state)
        database = state.database
        if serialized_database != state.database_bytes:
            database = _write_exact(database, serialized_database)
        updated_held = tuple(
            database if item.handle == state.database.handle else item
            for item in state.held
        )
        closed = _replace(
            owner,
            state,
            phase="closed",
            database=database,
            database_bytes=serialized_database,
            held=updated_held,
        )
        _assert_unchanged(closed)
    except BaseException as original:
        try:
            _terminal(owner, _state(owner))
        except BaseException as cleanup_error:
            original.add_note(
                f"derived close quarantine also failed: {cleanup_error!r}"
            )
        raise


def close_derived_lifecycle(
    owner: OwnedDerivedLifecycle,
    serialized_database: bytes,
) -> None:
    with _serialized_derived_lifecycle(owner):
        _close_derived_lifecycle_locked(owner, serialized_database)


def _derived_lifecycle_files_locked(
    owner: OwnedDerivedLifecycle,
) -> DerivedLifecycleFiles:
    state = _state(owner)
    if state.phase not in {"claimed", "open", "closed", "finalizing"}:
        raise TypeError("derived files require a live lifecycle")
    _assert_unchanged(state)
    return DerivedLifecycleFiles(
        database_bytes=state.database_bytes,
        database_sha256=hashlib.sha256(state.database_bytes).hexdigest(),
        database_size=len(state.database_bytes),
        index_sha256=state.index_sha256,
        index_size=state.index.size,
        origin_bytes=state.origin_bytes,
        lease_bytes=state.lease_bytes,
        finalization_bytes=state.finalization_bytes,
    )


def derived_lifecycle_files(owner: OwnedDerivedLifecycle) -> DerivedLifecycleFiles:
    with _serialized_derived_lifecycle(owner):
        return _derived_lifecycle_files_locked(owner)


def _write_derived_finalization_locked(
    owner: OwnedDerivedLifecycle,
    payload: bytes,
) -> None:
    state = _state(owner)
    if state.phase != "closed":
        raise TypeError("derived finalization requires a closed lifecycle")
    finalization: _OpenEntry | None = None
    try:
        _assert_unchanged(state)
        finalization = _create_atomic_finalization(
            state.root,
            payload,
        )
        finalizing = _replace(
            owner,
            state,
            phase="finalizing",
            finalization=finalization,
            finalization_bytes=payload,
            held=(*state.held, finalization),
        )
        finalization = None
        _assert_unchanged(finalizing)
    except BaseException as original:
        if finalization is not None:
            try:
                _discard_created_child(
                    state.root,
                    DERIVED_FINALIZATION_NAME,
                    finalization,
                )
            except BaseException as cleanup_error:
                original.add_note(
                    f"derived final rollback also failed: {cleanup_error!r}"
                )
        try:
            current = _state(owner)
            _terminal(
                owner,
                current,
                discard_finalization=current.finalization is not None,
            )
        except BaseException as cleanup_error:
            original.add_note(
                f"derived final quarantine also failed: {cleanup_error!r}"
            )
        raise


def write_derived_finalization(
    owner: OwnedDerivedLifecycle,
    payload: bytes,
) -> None:
    with _serialized_derived_lifecycle(owner):
        _write_derived_finalization_locked(owner, payload)


def _commit_derived_lifecycle_locked(owner: OwnedDerivedLifecycle) -> None:
    state = _state(owner)
    if state.phase != "finalizing":
        raise TypeError("derived lifecycle commit requires final self-verification")
    try:
        _assert_unchanged(state)
    except BaseException as original:
        try:
            _terminal(owner, _state(owner), discard_finalization=True)
        except BaseException as cleanup_error:
            original.add_note(
                f"derived commit quarantine also failed: {cleanup_error!r}"
            )
        raise
    finalization = state.finalization
    assert finalization is not None
    backup_root: _OpenEntry | None = None
    backup_finalization: _OpenEntry | None = None
    try:
        backup_root = _duplicate_entry(state.root)
        backup_finalization = _duplicate_entry(finalization)
    except BaseException as original:
        backups = tuple(
            entry
            for entry in (backup_root, backup_finalization)
            if entry is not None
        )
        if backups:
            try:
                _close_entries(backups)
            except BaseException as cleanup_error:
                original.add_note(
                    f"derived commit backup cleanup also failed: {cleanup_error!r}"
                )
        raise
    # Terminal ownership is revoked before the first raw close.  That makes
    # every descriptor eligible for exactly one staged close pass: a partial
    # native failure can never leave already-closed integers in a live state
    # that an abort/finalizer might replay after descriptor reuse.
    try:
        _revoke(owner, state)
    except BaseException as original:
        try:
            _close_entries((backup_root, backup_finalization))
        except BaseException as cleanup_error:
            original.add_note(
                f"derived commit backup cleanup also failed: {cleanup_error!r}"
            )
        try:
            _terminal(owner, _state(owner), discard_finalization=True)
        except BaseException as cleanup_error:
            original.add_note(
                f"derived commit revoke rollback also failed: {cleanup_error!r}"
            )
        raise

    _close_committed_lifecycle(
        state.root,
        finalization,
        backup_root,
        backup_finalization,
        (*state.held, *state.parent_chain),
    )


def commit_derived_lifecycle(owner: OwnedDerivedLifecycle) -> None:
    with _serialized_derived_lifecycle(owner):
        _commit_derived_lifecycle_locked(owner)


def abort_derived_lifecycle(owner: OwnedDerivedLifecycle) -> Path:
    with _serialized_derived_lifecycle(owner):
        state = _state(owner)
        result = state.path
        _terminal(
            owner,
            state,
            discard_finalization=state.finalization is not None,
        )
        return result


derived_lifecycle_operation_guard = freeze_derived_lifecycle_guard(
    globals(),
    primitive_namespace=_close_entry.__globals__,
    native_namespace=_lifecycle_native.__dict__,
    state_op=_state,
    revoke_op=_revoke,
    raw_close=_kernel32.CloseHandle if os.name == "nt" else os.close,
    kernel32=_kernel32 if os.name == "nt" else None,
    error_type=DiffuseBaseArtifactError,
    registration_state_op=None,
    emergency_abandon_op=_emergency_abandon_owner,
    emergency_registration_op=_emergency_take_registration,
)


__all__ = [
    "DerivedLifecycleFiles",
    "OwnedDerivedLifecycle",
    "abort_derived_lifecycle",
    "claim_derived_lifecycle",
    "close_derived_lifecycle",
    "commit_derived_lifecycle",
    "derived_lifecycle_files",
    "derived_lifecycle_operation_guard",
    "discard_derived_registration",
    "mark_derived_open",
    "owned_database_image",
    "owned_derived_lifecycle_for_clone",
    "owned_index_load_path",
    "register_derived_publication",
    "write_derived_finalization",
]
