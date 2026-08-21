"""Held-handle publication and verification for false-only corpus candidates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import secrets
import stat
from threading import RLock
from typing import Any, Callable, Literal

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.eval._diffuse_latent_training_corpus_codec import (
    decode_latent_training_payload,
)
from memory_condense.eval._diffuse_latent_training_corpus_filesystem import (
    CorpusTreeSnapshot,
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
    _write_windows,
    require_plain_parent,
)
from memory_condense.eval._diffuse_latent_training_corpus_io import (
    _PARTITION_KEYS,
    _ROW_KEYS,
    _decode_partition as _decode_partition_model,
    _decode_row,
    _loads as _load_generic_json,
    _mapping as _generic_mapping,
)
from memory_condense.eval._diffuse_latent_training_corpus_models import (
    MAX_METADATA_FILE_BYTES,
    MAX_PAYLOAD_SHARD_BYTES,
    ROOT_MANIFEST_NAME,
    DecodedLatentTrainingCorpusRow,
    LatentTrainingFileIdentity,
)
from memory_condense.eval._diffuse_latent_training_corpus_route import (
    live_route_v2_implementation_sha256,
)
from memory_condense.eval.diffuse_latent_training_corpus import (
    validate_structural_latent_training_partition_rows,
    verify_structural_latent_training_corpus,
)

from tools._diffuse_latent_training_corpus_authority_codec import (
    MAX_CANDIDATE_RECEIPT_BYTES,
    decode_candidate_publication,
    decode_phase_candidate,
    decode_production_candidate,
    encode_candidate_publication,
    encode_phase_candidate,
    encode_production_candidate,
)
from tools._diffuse_latent_training_corpus_authority_models import (
    CANDIDATE_RECEIPT_NAME,
    PHASE_CANDIDATE_NAME,
    PRODUCTION_CANDIDATE_NAME,
    ProductionCandidatePublicationReceipt,
    ProductionCorpusCandidateReceipt,
    ProductionLatentTrainingCorpusError,
    ProductionPhaseCandidateReceipt,
    VerifiedLatentTrainingCorpusCandidate,
    VerifiedLatentTrainingPhaseCandidate,
    inventory_sha256,
    locked_production_external_lock,
)
from tools._diffuse_latent_training_corpus_candidate_verification import (
    GenericCorpusBinding as _BoundedGenericCorpusBinding,
    generic_binding_from_snapshot as _bounded_generic_binding,
    inspect_generic_corpus_binding as _bounded_inspect_generic,
    verify_latent_training_corpus_candidate as _bounded_verify_candidate,
    verify_latent_training_fit_candidate as _bounded_verify_fit,
    verify_latent_training_validation_candidate as _bounded_verify_validation,
)


_ROW_NAME = re.compile(r"[0-9]{6}\.json\Z")
_PAYLOAD_NAME = re.compile(r"[0-9a-f]{64}\.json\Z")
_ROOT_NAMES = {
    CANDIDATE_RECEIPT_NAME,
    PRODUCTION_CANDIDATE_NAME,
    "fit",
    "generic",
    "validation",
}
_PHASE_NAMES = {
    PHASE_CANDIDATE_NAME,
    PRODUCTION_CANDIDATE_NAME,
    "partition.json",
    "payloads",
    "rows",
}
_MAX_PHASE_FILES = 3 + 300 + 300
_MAX_PHASE_BYTES = (
    3 * MAX_CANDIDATE_RECEIPT_BYTES
    + 300 * MAX_METADATA_FILE_BYTES
    + 300 * MAX_PAYLOAD_SHARD_BYTES
)


def _absolute_child(value: str | Path) -> Path:
    path = Path(os.path.abspath(os.fspath(value)))
    if not path.name or path.name in {".", ".."}:
        raise ProductionLatentTrainingCorpusError(
            "candidate path requires one bounded child name"
        )
    return path


def _current_identity(entry: _OpenEntry) -> tuple[int, ...]:
    if os.name == "nt":
        return _win_info(entry.handle)[0]
    return _posix_identity(os.fstat(entry.handle))


def _assert_current(entry: _OpenEntry) -> None:
    if _current_identity(entry) != entry.identity:
        raise ProductionLatentTrainingCorpusError(
            "candidate filesystem object changed while held"
        )


def _assert_bound(parent: _OpenEntry, name: str, entry: _OpenEntry) -> None:
    if os.name == "nt":
        expected = os.path.normcase(os.path.normpath(str(parent.path / name)))
        if _win_final_path(entry.handle) != expected:
            raise ProductionLatentTrainingCorpusError(
                "candidate filesystem name changed while held"
            )
        if not _same_object(_current_identity(entry), entry.identity):
            raise ProductionLatentTrainingCorpusError(
                "candidate filesystem object changed while held"
            )
        return
    _assert_named_object(parent, name, entry)


def _require_one_link(entry: _OpenEntry) -> None:
    count = (
        _current_identity(entry)[-1]
        if os.name == "nt"
        else int(os.fstat(entry.handle).st_nlink)
    )
    if count != 1:
        raise ProductionLatentTrainingCorpusError(
            "candidate packages forbid hard-linked files"
        )
    _assert_current(entry)


def _mkdir_child(
    parent: _OpenEntry,
    name: str,
    *,
    owner: OwnedCandidateStaging | None = None,
    relative: tuple[str, ...] | None = None,
) -> _OpenEntry:
    if type(name) is not str or not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("candidate directory name is not a bounded child")
    if os.name == "nt":
        (parent.path / name).mkdir(mode=0o700)
        try:
            os.chmod(parent.path / name, 0o700)
        except OSError:
            pass
    else:
        os.mkdir(name, mode=0o700, dir_fd=parent.handle)
    child = _open_child(parent, name, directory=True)
    if owner is not None:
        if relative is None:
            _close_entry(child)
            raise ValueError("owned candidate directory requires its relative path")
        _record_owned(owner, relative, child, directory=True)
    _flush_open_directory(child)
    _flush_open_directory(parent)
    return child


def _write_child(
    parent: _OpenEntry,
    name: str,
    payload: bytes,
    *,
    limit: int,
    owner: OwnedCandidateStaging | None = None,
    relative: tuple[str, ...] | None = None,
) -> LatentTrainingFileIdentity:
    if type(payload) is not bytes or not payload or len(payload) > limit:
        raise ValueError("candidate file payload violates its byte bounds")
    if type(name) is not str or not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("candidate file name is not a bounded child")
    if os.name == "nt":
        _write_windows(parent.path / name, payload)
        try:
            os.chmod(parent.path / name, 0o600)
        except OSError:
            pass
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
                    raise OSError("short candidate file write")
                view = view[count:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    entry = _open_child(parent, name, directory=False)
    try:
        _require_one_link(entry)
        if _read_entry(entry, limit) != payload:
            raise ProductionLatentTrainingCorpusError(
                "candidate file differs immediately after creation"
            )
        if owner is not None:
            if relative is None:
                raise ValueError("owned candidate file requires its relative path")
            _record_owned(owner, relative, entry, directory=False)
    finally:
        _close_entry(entry)
    _flush_open_directory(parent)
    return LatentTrainingFileIdentity(
        name, hashlib.sha256(payload).hexdigest(), len(payload)
    )


class OwnedCandidateStaging:
    """Opaque, registry-issued ownership of one random staging root."""

    __slots__ = ("_token",)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("candidate staging capabilities cannot be constructed directly")

    def __setattr__(self, _name: str, _value: object) -> None:
        raise TypeError("candidate staging capabilities are immutable")

    def __fspath__(self) -> str:
        return os.fspath(_owned_state(self).path)


@dataclass(frozen=True, slots=True)
class _OwnedState:
    owner: OwnedCandidateStaging
    path: Path
    parent: Path
    target_name: str
    prefix: str
    identity: tuple[int, ...]
    objects: tuple[tuple[tuple[str, ...], bool, tuple[int, ...]], ...] = ()
    promoted: bool = False


def _ownership_boundary() -> tuple[Callable[..., OwnedCandidateStaging], ...]:
    registry: dict[object, _OwnedState] = {}
    lock = RLock()

    def state(value: OwnedCandidateStaging) -> _OwnedState:
        if type(value) is not OwnedCandidateStaging:
            raise TypeError("operation requires an issued candidate staging capability")
        try:
            token = object.__getattribute__(value, "_token")
        except (AttributeError, TypeError) as exc:
            raise TypeError("candidate staging capability is not issued") from exc
        with lock:
            result = registry.get(token)
            if result is None or result.owner is not value:
                raise TypeError("candidate staging capability is not live")
            return result

    def issue(
        path: Path,
        parent: Path,
        target_name: str,
        prefix: str,
        identity: tuple[int, ...],
    ) -> OwnedCandidateStaging:
        value = object.__new__(OwnedCandidateStaging)
        token = object()
        object.__setattr__(value, "_token", token)
        created = _OwnedState(value, path, parent, target_name, prefix, identity)
        with lock:
            registry[token] = created
        return value

    def replace(value: OwnedCandidateStaging, expected: _OwnedState, **changes: object) -> _OwnedState:
        token = object.__getattribute__(value, "_token")
        with lock:
            current = registry.get(token)
            if current is not expected or current.owner is not value:
                raise TypeError("candidate staging capability changed during operation")
            fields = {
                "owner": current.owner,
                "path": current.path,
                "parent": current.parent,
                "target_name": current.target_name,
                "prefix": current.prefix,
                "identity": current.identity,
                "objects": current.objects,
                "promoted": current.promoted,
            }
            fields.update(changes)
            updated = _OwnedState(**fields)
            registry[token] = updated
            return updated

    def revoke(value: OwnedCandidateStaging, expected: _OwnedState) -> None:
        token = object.__getattribute__(value, "_token")
        with lock:
            current = registry.get(token)
            if current is not expected or current.owner is not value:
                raise TypeError("candidate staging capability changed before revocation")
            del registry[token]

    return issue, state, replace, revoke


_issue_owned, _owned_state, _replace_owned, _revoke_owned = _ownership_boundary()
del _ownership_boundary


def candidate_staging_path(value: OwnedCandidateStaging) -> Path:
    """Return the live private staging path for the generic inner publisher."""

    return _owned_state(value).path


def _record_owned(
    value: OwnedCandidateStaging,
    relative: tuple[str, ...],
    entry: _OpenEntry,
    *,
    directory: bool,
) -> None:
    if not relative or any(not part or part in {".", ".."} for part in relative):
        raise ValueError("owned candidate entry has an invalid relative path")
    current = _owned_state(value)
    values = {path: (kind, identity) for path, kind, identity in current.objects}
    if relative in values:
        raise ProductionLatentTrainingCorpusError("owned candidate entry was repeated")
    values[relative] = (directory, entry.identity)
    ordered = tuple(
        (path, kind, identity)
        for path, (kind, identity) in sorted(values.items())
    )
    _replace_owned(value, current, objects=ordered)


def create_candidate_staging(destination: str | Path) -> OwnedCandidateStaging:
    target = _absolute_child(destination)
    parent = require_plain_parent(target.parent)
    chain = _open_chain(parent)
    parent_entry = chain[-1]
    created: _OpenEntry | None = None
    try:
        if os.path.lexists(target):
            raise FileExistsError(target)
        prefix = f".{target.name}.candidate-staging-"
        for _ in range(128):
            name = prefix + secrets.token_hex(16)
            try:
                created = _mkdir_child(parent_entry, name)
            except FileExistsError:
                continue
            break
        if created is None:
            raise FileExistsError("cannot allocate a unique candidate staging root")
        return _issue_owned(
            created.path, parent, target.name, prefix, created.identity
        )
    finally:
        if created is not None:
            _close_entry(created)
        for entry in reversed(chain):
            _close_entry(entry)


def _open_owned(value: OwnedCandidateStaging) -> tuple[list[_OpenEntry], _OpenEntry]:
    state = _owned_state(value)
    valid_name = (
        state.path.name == state.target_name
        if state.promoted
        else state.path.name.startswith(state.prefix)
    )
    if state.path.parent != state.parent or not valid_name:
        raise TypeError("issued candidate staging state is malformed")
    chain = _open_chain(state.path)
    root = chain[-1]
    if not _same_object(root.identity, state.identity):
        for entry in reversed(chain):
            _close_entry(entry)
        raise ProductionLatentTrainingCorpusError("owned staging root was replaced")
    _assert_named_object(chain[-2], state.path.name, root)
    return chain, root


def _rename_owned(
    value: OwnedCandidateStaging,
    target: Path,
    *,
    promoted: bool,
) -> _OwnedState:
    state = _owned_state(value)
    if target.parent != state.parent or os.path.lexists(target):
        raise FileExistsError(target)
    chain = _open_chain(state.parent)
    promoter: _OpenEntry | None = None
    try:
        parent = chain[-1]
        if os.name == "nt":
            promoter = _win_open(
                state.path,
                directory=True,
                delete_access=True,
            )
            if not _same_object(promoter.identity, state.identity):
                raise ProductionLatentTrainingCorpusError(
                    "owned candidate changed before rename"
                )
            _win_rename(promoter, parent, target.name)
        else:
            promoter = _open_child(parent, state.path.name, directory=True)
            if not _same_object(promoter.identity, state.identity):
                raise ProductionLatentTrainingCorpusError(
                    "owned candidate changed before rename"
                )
            _posix_rename_noreplace(parent, state.path.name, target.name)
        # This registry transition is the first operation after the atomic
        # rename and precedes every fallible flush/reopen.
        updated = _replace_owned(
            value,
            state,
            path=target,
            prefix=target.name,
            promoted=promoted,
        )
    finally:
        if promoter is not None:
            _close_entry(promoter)
        for entry in reversed(chain):
            _close_entry(entry)
    rebound_chain, rebound = _open_owned(value)
    try:
        if not _same_object(rebound.identity, updated.identity):
            raise ProductionLatentTrainingCorpusError(
                "owned candidate rename changed its object"
            )
        _flush_open_directory(rebound_chain[-2])
    finally:
        for entry in reversed(rebound_chain):
            _close_entry(entry)
    return updated


@dataclass(slots=True)
class _DeleteNode:
    entry: _OpenEntry | None
    parent: _OpenEntry
    name: str
    children: tuple["_DeleteNode", ...]


def _schema(path: tuple[str, ...], names: tuple[str, ...]) -> dict[str, bool]:
    values = set(names)
    if len(values) != len(names):
        raise ProductionLatentTrainingCorpusError("candidate tree repeated an entry")
    if not path:
        allowed_files = {PRODUCTION_CANDIDATE_NAME, CANDIDATE_RECEIPT_NAME}
        allowed_dirs = {"generic", "fit", "validation"}
        if not values <= allowed_files | allowed_dirs:
            raise ProductionLatentTrainingCorpusError("owned candidate root changed schema")
        return {name: name in allowed_dirs for name in names}
    if path == ("generic",):
        allowed_files = {ROOT_MANIFEST_NAME}
        allowed_dirs = {"partitions", "payloads", "rows"}
        if not values <= allowed_files | allowed_dirs:
            raise ProductionLatentTrainingCorpusError("owned generic tree changed schema")
        return {name: name in allowed_dirs for name in names}
    if path == ("generic", "partitions"):
        if not values <= {"fit.json", "validation.json"}:
            raise ProductionLatentTrainingCorpusError("generic partitions changed schema")
        return {name: False for name in names}
    if path == ("generic", "rows"):
        if any(_ROW_NAME.fullmatch(name) is None for name in names):
            raise ProductionLatentTrainingCorpusError("generic rows changed schema")
        return {name: False for name in names}
    if path == ("generic", "payloads"):
        if any(_PAYLOAD_NAME.fullmatch(name) is None for name in names):
            raise ProductionLatentTrainingCorpusError("generic payloads changed schema")
        return {name: False for name in names}
    if len(path) == 1 and path[0] in {"fit", "validation"}:
        allowed_files = {
            PRODUCTION_CANDIDATE_NAME, PHASE_CANDIDATE_NAME, "partition.json"
        }
        allowed_dirs = {"rows", "payloads"}
        if not values <= allowed_files | allowed_dirs:
            raise ProductionLatentTrainingCorpusError("phase candidate changed schema")
        return {name: name in allowed_dirs for name in names}
    if len(path) == 2 and path[0] in {"fit", "validation"}:
        pattern = _ROW_NAME if path[1] == "rows" else _PAYLOAD_NAME
        if path[1] not in {"rows", "payloads"} or any(
            pattern.fullmatch(name) is None for name in names
        ):
            raise ProductionLatentTrainingCorpusError("phase child changed schema")
        return {name: False for name in names}
    raise ProductionLatentTrainingCorpusError("candidate cleanup exceeded schema depth")


def _tree_cap(path: tuple[str, ...]) -> int:
    if path in {("generic", "rows"), ("generic", "payloads")} or (
        len(path) == 2 and path[0] in {"fit", "validation"}
    ):
        return 301
    return 8


def _preflight_delete(
    parent: _OpenEntry,
    name: str,
    path: tuple[str, ...],
    state: _OwnedState,
) -> _DeleteNode:
    entry = _open_child(parent, name, directory=True, delete_access=os.name == "nt")
    children: list[_DeleteNode] = []
    try:
        objects = {
            relative: (directory, identity)
            for relative, directory, identity in state.objects
        }
        expected_identity = state.identity if not path else objects.get(path, (None, None))[1]
        if expected_identity is None or not _same_object(entry.identity, expected_identity):
            raise ProductionLatentTrainingCorpusError("owned cleanup root was replaced")
        names = _entry_names(entry, cap=_tree_cap(path))
        kinds = _schema(path, names)
        expected_names = {
            relative[len(path)]
            for relative in objects
            if len(relative) == len(path) + 1 and relative[: len(path)] == path
        }
        if set(names) != expected_names:
            raise ProductionLatentTrainingCorpusError(
                "owned candidate tree differs from its issued object inventory"
            )
        for child_name in names:
            child_path = (*path, child_name)
            expected = objects.get(child_path)
            if expected is None or expected[0] is not kinds[child_name]:
                raise ProductionLatentTrainingCorpusError(
                    "owned candidate entry kind changed"
                )
            if kinds[child_name]:
                children.append(
                    _preflight_delete(entry, child_name, child_path, state)
                )
            else:
                if os.name == "nt":
                    probe = _open_child(entry, child_name, directory=False)
                    try:
                        _require_one_link(probe)
                        if not _same_object(probe.identity, expected[1]):
                            raise ProductionLatentTrainingCorpusError(
                                "owned candidate file was replaced"
                            )
                    finally:
                        _close_entry(probe)
                child = _open_child(
                    entry,
                    child_name,
                    directory=False,
                    delete_access=os.name == "nt",
                )
                try:
                    if os.name != "nt":
                        _require_one_link(child)
                    if not _same_object(child.identity, expected[1]):
                        raise ProductionLatentTrainingCorpusError(
                            "owned candidate file was replaced"
                        )
                except BaseException:
                    _close_entry(child)
                    raise
                children.append(_DeleteNode(child, entry, child_name, ()))
        return _DeleteNode(entry, parent, name, tuple(children))
    except BaseException as original:
        for child in reversed(children):
            _close_delete_tree(child)
        _close_entry(entry)
        raise


def _close_delete_tree(node: _DeleteNode) -> None:
    for child in node.children:
        _close_delete_tree(child)
    if node.entry is not None:
        _close_entry(node.entry)
        node.entry = None


def _delete_preflighted(node: _DeleteNode) -> None:
    for child in node.children:
        _delete_preflighted(child)
    entry = node.entry
    if entry is None:
        raise ProductionLatentTrainingCorpusError("owned delete handle closed early")
    _assert_bound(node.parent, node.name, entry)
    if os.name == "nt":
        _win_mark_delete(entry)
        _close_entry(entry)
        node.entry = None
    elif entry.is_directory:
        os.rmdir(node.name, dir_fd=node.parent.handle)
    else:
        os.unlink(node.name, dir_fd=node.parent.handle)


def _validate_owned_root(root: _OpenEntry, state: _OwnedState) -> tuple[str, ...]:
    objects = {
        relative: (directory, identity)
        for relative, directory, identity in state.objects
    }
    names = _entry_names(root, cap=_tree_cap(()))
    _schema((), names)
    expected_names = {
        relative[0] for relative in objects if len(relative) == 1
    }
    if set(names) != expected_names:
        raise ProductionLatentTrainingCorpusError(
            "owned candidate root differs from its issued object inventory"
        )
    for name in names:
        directory, expected_identity = objects[(name,)]
        entry = _open_child(
            root,
            name,
            directory=directory,
        )
        try:
            if not _same_object(entry.identity, expected_identity):
                raise ProductionLatentTrainingCorpusError(
                    "owned candidate direct child was replaced"
                )
            if not directory:
                _require_one_link(entry)
        finally:
            _close_entry(entry)
    return names


def _drop_owned_prefix(
    value: OwnedCandidateStaging,
    prefix: tuple[str, ...],
) -> _OwnedState:
    state = _owned_state(value)
    remaining = tuple(
        item for item in state.objects if item[0][: len(prefix)] != prefix
    )
    return _replace_owned(value, state, objects=remaining)


def cleanup_candidate_staging(value: OwnedCandidateStaging) -> None:
    state = _owned_state(value)
    if not os.path.lexists(state.path):
        _revoke_owned(value, state)
        return
    chain, root = _open_owned(value)
    try:
        _validate_owned_root(root, state)
        tomb = state.parent / (
            f".{state.target_name}.candidate-cleanup-{secrets.token_hex(16)}"
        )
        if os.path.lexists(tomb):
            raise FileExistsError(tomb)
    finally:
        for entry in reversed(chain):
            _close_entry(entry)
    _rename_owned(value, tomb, promoted=False)

    # The random tombstone is detached from every caller-chosen name.  Keep
    # its exact root capability live, but preflight/delete one bounded subtree
    # at a time so the generic + fit + validation duplicates never exceed the
    # process descriptor limit together.
    chain, root = _open_owned(value)
    root_closed = False
    try:
        parent = chain[-2]
        for name in tuple(_entry_names(root, cap=_tree_cap(()))):
            state = _owned_state(value)
            direct = {
                relative: (directory, identity)
                for relative, directory, identity in state.objects
                if len(relative) == 1
            }
            expected = direct.get((name,))
            if expected is None:
                raise ProductionLatentTrainingCorpusError(
                    "cleanup tombstone gained an unowned child"
                )
            if expected[0]:
                node = _preflight_delete(root, name, (name,), state)
                try:
                    _delete_preflighted(node)
                finally:
                    _close_delete_tree(node)
            else:
                entry = _open_child(
                    root, name, directory=False
                )
                try:
                    _require_one_link(entry)
                    if not _same_object(entry.identity, expected[1]):
                        raise ProductionLatentTrainingCorpusError(
                            "cleanup tombstone file was replaced"
                        )
                    if os.name == "nt":
                        expected_identity = entry.identity
                        _close_entry(entry)
                        entry = _open_child(
                            root,
                            name,
                            directory=False,
                            delete_access=True,
                        )
                        if not _same_object(entry.identity, expected_identity):
                            raise ProductionLatentTrainingCorpusError(
                                "cleanup tombstone file changed before delete"
                            )
                        _assert_bound(root, name, entry)
                        _win_mark_delete(entry)
                        _close_entry(entry)
                        entry = None
                    else:
                        _assert_bound(root, name, entry)
                        os.unlink(name, dir_fd=root.handle)
                finally:
                    if entry is not None:
                        _close_entry(entry)
            _drop_owned_prefix(value, (name,))
            _validate_owned_root(root, _owned_state(value))
        if _owned_state(value).objects:
            raise ProductionLatentTrainingCorpusError(
                "cleanup tombstone retained owned entries"
            )
        _assert_bound(parent, _owned_state(value).path.name, root)
        if os.name == "nt":
            root_identity = root.identity
            _close_entry(root)
            root_closed = True
            delete_root = _win_open(
                _owned_state(value).path,
                directory=True,
                delete_access=True,
            )
            try:
                if not _same_object(delete_root.identity, root_identity):
                    raise ProductionLatentTrainingCorpusError(
                        "cleanup tombstone root changed before delete"
                    )
                _assert_bound(
                    parent, _owned_state(value).path.name, delete_root
                )
                _win_mark_delete(delete_root)
            finally:
                _close_entry(delete_root)
        else:
            os.rmdir(_owned_state(value).path.name, dir_fd=parent.handle)
        _flush_open_directory(parent)
        completed = _owned_state(value)
        _revoke_owned(value, completed)
    finally:
        for entry in reversed(chain):
            if entry is root and root_closed:
                continue
            _close_entry(entry)


def _record_generic_snapshot(
    owner: OwnedCandidateStaging,
    root: _OpenEntry,
    snapshot: CorpusTreeSnapshot,
) -> None:
    state = _owned_state(owner)
    generic = _open_child(root, "generic", directory=True)
    try:
        snapshot_root = snapshot._directory_entries[""]
        if not _same_object(generic.identity, snapshot_root.identity):
            raise ProductionLatentTrainingCorpusError(
                "verified generic root differs from staged child"
            )
        captured: list[tuple[tuple[str, ...], bool, tuple[int, ...]]] = [
            (("generic",), True, generic.identity)
        ]
        for name, entry in snapshot._directory_entries.items():
            if name:
                captured.append((("generic", name), True, entry.identity))
        for relative, item in snapshot.files.items():
            captured.append(
                (("generic", *relative.split("/")), False, item._entry.identity)
            )
        snapshot.assert_unchanged()
        captured_objects = tuple(sorted(captured, key=lambda item: item[0]))
        if state.objects:
            if state.objects != captured_objects:
                raise ProductionLatentTrainingCorpusError(
                    "captured generic candidate ownership changed"
                )
            return
        _replace_owned(
            owner,
            state,
            objects=captured_objects,
        )
    finally:
        _close_entry(generic)


def recheck_captured_candidate_generic(
    owner: OwnedCandidateStaging,
) -> _BoundedGenericCorpusBinding:
    """Reopen the registered generic child without changing ownership state."""

    chain, root = _open_owned(owner)
    try:
        state = _owned_state(owner)
        _validate_owned_root(root, state)
        generic_path = state.path / "generic"
        verified = verify_structural_latent_training_corpus(generic_path)
        with CorpusTreeSnapshot(generic_path) as snapshot:
            binding = _bounded_generic_binding(verified, snapshot)
            generic = _open_child(root, "generic", directory=True)
            try:
                snapshot_root = snapshot._directory_entries[""]
                if not _same_object(generic.identity, snapshot_root.identity):
                    raise ProductionLatentTrainingCorpusError(
                        "captured generic root differs from its staged child"
                    )
                observed: list[
                    tuple[tuple[str, ...], bool, tuple[int, ...]]
                ] = [(('generic',), True, generic.identity)]
                observed.extend(
                    (("generic", name), True, entry.identity)
                    for name, entry in snapshot._directory_entries.items()
                    if name
                )
                observed.extend(
                    (
                        ("generic", *relative.split("/")),
                        False,
                        item._entry.identity,
                    )
                    for relative, item in snapshot.files.items()
                )
                expected = tuple(
                    item for item in state.objects if item[0][0] == "generic"
                )
                if tuple(sorted(observed, key=lambda item: item[0])) != expected:
                    raise ProductionLatentTrainingCorpusError(
                        "captured generic object inventory changed"
                    )
            finally:
                _close_entry(generic)
            snapshot.assert_unchanged()
        _validate_owned_root(root, _owned_state(owner))
        return binding
    finally:
        for entry in reversed(chain):
            _close_entry(entry)


def capture_candidate_generic(
    owner: OwnedCandidateStaging,
) -> _BoundedGenericCorpusBinding:
    """Verify and bind the completed generic child to an issued staging root.

    The generic publisher writes through its own atomic boundary.  Capturing
    immediately afterward lets later source-aware checks roll back that exact
    tree without granting cleanup authority over an unknown replacement.
    """

    chain, root = _open_owned(owner)
    try:
        if set(_entry_names(root, cap=1)) != {"generic"}:
            raise ProductionLatentTrainingCorpusError(
                "generic ownership capture requires exactly one staged child"
            )
        generic_path = _owned_state(owner).path / "generic"
        verified = verify_structural_latent_training_corpus(generic_path)
        with CorpusTreeSnapshot(generic_path) as snapshot:
            binding = _bounded_generic_binding(verified, snapshot)
            _record_generic_snapshot(owner, root, snapshot)
            snapshot.assert_unchanged()
        _validate_owned_root(root, _owned_state(owner))
        return binding
    finally:
        for entry in reversed(chain):
            _close_entry(entry)


# Bind the cohesive bounded verifier as this publication module's read surface.
GenericCorpusBinding = _BoundedGenericCorpusBinding
inspect_generic_corpus_binding = _bounded_inspect_generic
verify_latent_training_corpus_candidate = _bounded_verify_candidate
verify_latent_training_fit_candidate = _bounded_verify_fit
verify_latent_training_validation_candidate = _bounded_verify_validation


def _publish_phase(
    root: _OpenEntry,
    generic_snapshot: CorpusTreeSnapshot,
    view: Any,
    role: Literal["fit", "validation"],
    candidate: ProductionCorpusCandidateReceipt,
    candidate_bytes: bytes,
    owner: OwnedCandidateStaging,
) -> ProductionPhaseCandidateReceipt:
    phase_root = _mkdir_child(root, role, owner=owner, relative=(role,))
    rows = payloads = None
    try:
        rows = _mkdir_child(
            phase_root, "rows", owner=owner, relative=(role, "rows")
        )
        payloads = _mkdir_child(
            phase_root, "payloads", owner=owner, relative=(role, "payloads")
        )
        inventory: list[LatentTrainingFileIdentity] = []

        def write(relative: str, payload: bytes, *, limit: int) -> None:
            parent = phase_root
            name = relative
            if "/" in relative:
                directory, name = relative.split("/", 1)
                parent = rows if directory == "rows" else payloads
            identity = _write_child(
                parent,
                name,
                payload,
                limit=limit,
                owner=owner,
                relative=(role, *relative.split("/")),
            )
            inventory.append(
                LatentTrainingFileIdentity(relative, identity.sha256, identity.bytes)
            )

        write(
            PRODUCTION_CANDIDATE_NAME,
            candidate_bytes,
            limit=MAX_CANDIDATE_RECEIPT_BYTES,
        )
        write(
            "partition.json",
            generic_snapshot.read(f"partitions/{role}.json"),
            limit=MAX_METADATA_FILE_BYTES,
        )
        copied_payloads: set[str] = set()
        for item in view.rows:
            row_path = f"rows/{item.manifest.ordinal:06d}.json"
            write(
                row_path,
                generic_snapshot.read(row_path),
                limit=MAX_METADATA_FILE_BYTES,
            )
            payload_path = item.manifest.payload_relative_path
            if payload_path not in copied_payloads:
                write(
                    payload_path,
                    generic_snapshot.read(payload_path),
                    limit=MAX_PAYLOAD_SHARD_BYTES,
                )
                copied_payloads.add(payload_path)
        inventory.sort(key=lambda item: item.relative_path)
        values = tuple(inventory)
        candidate_file = next(
            item for item in values if item.relative_path == PRODUCTION_CANDIDATE_NAME
        )
        partition_file = next(item for item in values if item.relative_path == "partition.json")
        phase = ProductionPhaseCandidateReceipt(
            phase=role,
            generic_corpus_sha256=candidate.generic_corpus_sha256,
            generic_root_manifest_sha256=candidate.generic_root_manifest_sha256,
            production_candidate_sha256=candidate.candidate_sha256,
            production_candidate_file_sha256=candidate_file.sha256,
            production_candidate_file_bytes=candidate_file.bytes,
            partition_sha256=view.partition.partition_sha256,
            partition_file_sha256=partition_file.sha256,
            partition_file_bytes=partition_file.bytes,
            row_count=view.partition.row_count,
            ordered_question_ids_sha256=view.partition.ordered_question_ids_sha256,
            inventory=values,
            inventory_sha256=inventory_sha256(values),
        )
        _write_child(
            phase_root,
            PHASE_CANDIDATE_NAME,
            encode_phase_candidate(phase),
            limit=MAX_CANDIDATE_RECEIPT_BYTES,
            owner=owner,
            relative=(role, PHASE_CANDIDATE_NAME),
        )
        _flush_open_directory(phase_root)
        return phase
    finally:
        if payloads is not None:
            _close_entry(payloads)
        if rows is not None:
            _close_entry(rows)
        _close_entry(phase_root)


def publish_candidate_root(
    staging: OwnedCandidateStaging,
    destination: str | Path,
    *,
    candidate: ProductionCorpusCandidateReceipt,
    final_guard: Callable[[], None],
) -> VerifiedLatentTrainingCorpusCandidate:
    if type(candidate) is not ProductionCorpusCandidateReceipt:
        raise TypeError("candidate publication requires the exact false-only type")
    if not callable(final_guard):
        raise TypeError("candidate publication requires a final drift guard")
    candidate.__post_init__()
    target = _absolute_child(destination)
    initial = _owned_state(staging)
    if target.parent != initial.parent or target.name != initial.target_name:
        raise ProductionLatentTrainingCorpusError("staging belongs to another target")
    try:
        chain, root = _open_owned(staging)
        try:
            if os.path.lexists(target):
                raise FileExistsError(target)
            if set(_entry_names(root, cap=1)) != {"generic"}:
                raise ProductionLatentTrainingCorpusError(
                    "candidate staging must contain only the verified generic package"
                )
            state = _owned_state(staging)
            generic_path = state.path / "generic"
            generic = verify_structural_latent_training_corpus(generic_path)
            with CorpusTreeSnapshot(generic_path) as snapshot:
                binding = _bounded_generic_binding(generic, snapshot)
                _record_generic_snapshot(staging, root, snapshot)
                if (
                    binding.root_manifest_sha256
                    != candidate.generic_root_manifest_sha256
                    or binding.root_manifest_bytes
                    != candidate.generic_root_manifest_bytes
                    or binding.corpus_sha256 != candidate.generic_corpus_sha256
                    or binding.inventory_sha256 != candidate.generic_inventory_sha256
                    or binding.population_projection_sha256
                    != candidate.generic_population_projection_sha256
                    or binding.implementation_sha256
                    != candidate.generic_implementation_sha256
                    or binding.fit_partition_sha256
                    != candidate.generic_fit_partition_sha256
                    or binding.fit_manifest_file_sha256
                    != candidate.generic_fit_manifest_file_sha256
                    or binding.fit_manifest_file_bytes
                    != candidate.generic_fit_manifest_file_bytes
                    or binding.validation_partition_sha256
                    != candidate.generic_validation_partition_sha256
                    or binding.validation_manifest_file_sha256
                    != candidate.generic_validation_manifest_file_sha256
                    or binding.validation_manifest_file_bytes
                    != candidate.generic_validation_manifest_file_bytes
                ):
                    raise ProductionLatentTrainingCorpusError(
                        "candidate belongs to another generic corpus"
                    )
                candidate_bytes = encode_production_candidate(candidate)
                candidate_file = _write_child(
                    root,
                    PRODUCTION_CANDIDATE_NAME,
                    candidate_bytes,
                    limit=MAX_CANDIDATE_RECEIPT_BYTES,
                    owner=staging,
                    relative=(PRODUCTION_CANDIDATE_NAME,),
                )
                fit = _publish_phase(
                    root, snapshot, generic.fit, "fit", candidate,
                    candidate_bytes, staging,
                )
                validation = _publish_phase(
                    root, snapshot, generic.validation, "validation", candidate,
                    candidate_bytes, staging,
                )
                snapshot.assert_unchanged()
            current_path = _owned_state(staging).path
            verify_latent_training_fit_candidate(current_path / "fit")
            verify_latent_training_validation_candidate(current_path / "validation")
            final_guard()
            receipt = ProductionCandidatePublicationReceipt(
                generic_corpus_sha256=candidate.generic_corpus_sha256,
                generic_root_manifest_sha256=candidate.generic_root_manifest_sha256,
                production_candidate_sha256=candidate.candidate_sha256,
                production_candidate_file_sha256=candidate_file.sha256,
                production_candidate_file_bytes=candidate_file.bytes,
                fit_phase_candidate_sha256=fit.phase_candidate_sha256,
                validation_phase_candidate_sha256=validation.phase_candidate_sha256,
                source_commit=candidate.declared_execution.source_commit,
            )
            _write_child(
                root,
                CANDIDATE_RECEIPT_NAME,
                encode_candidate_publication(receipt),
                limit=MAX_CANDIDATE_RECEIPT_BYTES,
                owner=staging,
                relative=(CANDIDATE_RECEIPT_NAME,),
            )
            _flush_open_directory(root)
            verify_latent_training_corpus_candidate(current_path)
            final_guard()
            _assert_named_object(chain[-2], current_path.name, root)
            if os.path.lexists(target):
                raise FileExistsError(target)
            _flush_open_directory(root)
        finally:
            for entry in reversed(chain):
                _close_entry(entry)

        _rename_owned(staging, target, promoted=True)
        result = verify_latent_training_corpus_candidate(target)
        completed = _owned_state(staging)
        _revoke_owned(staging, completed)
        return result
    except BaseException as original:
        try:
            cleanup = _owned_state(staging)
        except TypeError:
            cleanup = None
        if cleanup is not None and os.path.lexists(cleanup.path):
            try:
                cleanup_candidate_staging(staging)
            except BaseException as cleanup_error:
                original.add_note(
                    f"exact owned rollback was refused: {cleanup_error!r}"
                )
        raise


__all__ = [
    "GenericCorpusBinding", "inspect_generic_corpus_binding",
    "verify_latent_training_corpus_candidate",
    "verify_latent_training_fit_candidate",
    "verify_latent_training_validation_candidate",
]
