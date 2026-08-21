"""Opaque, held-identity scratch trees for the production candidate runner.

Scratch data contains treatment text and model-derived stores.  It therefore
uses the same no-follow, tombstone-first deletion discipline as candidate
publication instead of pathname-recursive temporary-directory cleanup.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import secrets
from threading import RLock
from typing import Literal

from memory_condense.eval._diffuse_latent_training_corpus_filesystem import (
    _OpenEntry,
    _assert_named_object,
    _close_entry,
    _entry_names,
    _flush_open_directory,
    _open_chain,
    _open_child,
    _posix_rename_noreplace,
    _same_object,
    _win_mark_delete,
    _win_open,
    _win_rename,
    require_plain_parent,
)
from tools._diffuse_latent_training_corpus_authority_filesystem import (
    _assert_bound,
    _require_one_link,
)
from tools._diffuse_latent_training_corpus_authority_models import (
    ProductionLatentTrainingCorpusError,
)


_WorkspaceKind = Literal["execution", "row"]
_MAX_DEPTH = 8
_MAX_DIRECTORY_ENTRIES = 1024
_MAX_TOTAL_ENTRIES = 12_000


class OwnedCandidateWorkspace:
    """Registry-issued ownership of one random, private scratch root."""

    __slots__ = ("_token",)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("candidate workspace capabilities cannot be constructed")

    def __setattr__(self, _name: str, _value: object) -> None:
        raise TypeError("candidate workspace capabilities are immutable")

    def __copy__(self) -> object:
        raise TypeError("candidate workspace capabilities cannot be copied")

    def __deepcopy__(self, _memo: object) -> object:
        raise TypeError("candidate workspace capabilities cannot be copied")

    def __reduce_ex__(self, _protocol: int) -> object:
        raise TypeError("candidate workspace capabilities cannot be serialized")


@dataclass(frozen=True, slots=True)
class _WorkspaceState:
    owner: OwnedCandidateWorkspace
    path: Path
    parent: Path
    prefix: str
    identity: tuple[int, ...]
    kind: _WorkspaceKind
    held: _OpenEntry | None
    captured: bool = False
    inventory: tuple[tuple[tuple[str, ...], bool, tuple[int, ...]], ...] = ()


def _workspace_boundary():
    lock = RLock()
    states: dict[object, _WorkspaceState] = {}

    def issue(
        path: Path,
        parent: Path,
        prefix: str,
        identity: tuple[int, ...],
        kind: _WorkspaceKind,
        held: _OpenEntry,
    ) -> OwnedCandidateWorkspace:
        owner = object.__new__(OwnedCandidateWorkspace)
        token = object()
        object.__setattr__(owner, "_token", token)
        state = _WorkspaceState(owner, path, parent, prefix, identity, kind, held)
        with lock:
            states[token] = state
        return owner

    def get(owner: OwnedCandidateWorkspace) -> _WorkspaceState:
        if type(owner) is not OwnedCandidateWorkspace:
            raise TypeError("candidate workspace requires an exact capability")
        token = object.__getattribute__(owner, "_token")
        with lock:
            state = states.get(token)
        if state is None or state.owner is not owner:
            raise TypeError("candidate workspace capability is not live")
        return state

    def replace(
        owner: OwnedCandidateWorkspace,
        expected: _WorkspaceState,
        **changes: object,
    ) -> _WorkspaceState:
        token = object.__getattribute__(owner, "_token")
        with lock:
            if states.get(token) is not expected or expected.owner is not owner:
                raise TypeError("candidate workspace state changed")
            values = {
                name: getattr(expected, name)
                for name in _WorkspaceState.__dataclass_fields__
            }
            values.update(changes)
            updated = _WorkspaceState(**values)
            states[token] = updated
            return updated

    def revoke(owner: OwnedCandidateWorkspace, expected: _WorkspaceState) -> None:
        token = object.__getattribute__(owner, "_token")
        with lock:
            if states.get(token) is not expected or expected.owner is not owner:
                raise TypeError("candidate workspace state changed")
            del states[token]

    return issue, get, replace, revoke


_issue, _state, _replace, _revoke = _workspace_boundary()
del _workspace_boundary


def _absolute_child(value: str | Path) -> Path:
    path = Path(os.path.abspath(os.fspath(value)))
    if not path.name or path.name in {".", ".."}:
        raise ValueError("candidate workspace target needs a bounded child name")
    return path


def create_candidate_workspace(
    destination: str | Path,
    *,
    kind: _WorkspaceKind,
) -> OwnedCandidateWorkspace:
    if kind not in {"execution", "row"}:
        raise ValueError("candidate workspace has an unsupported kind")
    target = _absolute_child(destination)
    parent = require_plain_parent(target.parent)
    chain = _open_chain(parent)
    created: _OpenEntry | None = None
    try:
        parent_entry = chain[-1]
        prefix = f".{target.name}.{kind}-work-"
        for _ in range(128):
            name = prefix + secrets.token_hex(16)
            try:
                if os.name == "nt":
                    (parent / name).mkdir(mode=0o700)
                    try:
                        os.chmod(parent / name, 0o700)
                    except OSError:
                        pass
                else:
                    os.mkdir(name, mode=0o700, dir_fd=parent_entry.handle)
            except FileExistsError:
                continue
            created = (
                _win_open(parent / name, directory=True, share_delete=True)
                if os.name == "nt"
                else _open_child(parent_entry, name, directory=True)
            )
            _flush_open_directory(created)
            _flush_open_directory(parent_entry)
            owner = _issue(
                created.path, parent, prefix, created.identity, kind, created
            )
            created = None
            return owner
        raise FileExistsError("cannot allocate a unique candidate workspace")
    finally:
        if created is not None:
            _close_entry(created)
        for entry in reversed(chain):
            _close_entry(entry)


def candidate_workspace_path(owner: OwnedCandidateWorkspace) -> Path:
    return _state(owner).path


def _open_owned(
    owner: OwnedCandidateWorkspace,
) -> tuple[list[_OpenEntry], _OpenEntry, _OpenEntry, _WorkspaceState]:
    state = _state(owner)
    if state.path.parent != state.parent or not state.path.name.startswith(
        state.prefix
    ):
        raise TypeError("candidate workspace registry state is malformed")
    if state.held is None:
        raise TypeError("candidate workspace root handle is not live")
    chain = _open_chain(state.parent)
    parent = chain[-1]
    root = state.held
    try:
        if not _same_object(root.identity, state.identity):
            raise ProductionLatentTrainingCorpusError(
                "candidate workspace root was replaced"
            )
        _assert_named_object(parent, state.path.name, root)
        return chain, parent, root, state
    except BaseException:
        for entry in reversed(chain):
            _close_entry(entry)
        raise


_HASH_NAME = re.compile(r"[0-9a-f]{64}\Z")


def _schema(kind: _WorkspaceKind, path: tuple[str, ...], names: tuple[str, ...]) -> None:
    values = set(names)
    if len(values) != len(names):
        raise ProductionLatentTrainingCorpusError("workspace repeated an entry")
    if kind == "row":
        expected = (
            {"derived", ".derived.publish.lock"}
            if not path
            else {
                "memory.db", "hnsw_index.bin", "derived-origin.json",
                "derived-open.claim", "derived-final.json",
            }
            if path == ("derived",)
            else None
        )
        if expected is None or values != expected:
            raise ProductionLatentTrainingCorpusError(
                "row workspace changed its exact schema"
            )
        return
    if not path:
        expected = {"cache", "rows"}
    elif path == ("rows",):
        expected = set()
    elif path == ("cache",):
        expected = {"stores", "query-inputs"}
    elif path in {("cache", "stores"), ("cache", "query-inputs")}:
        digests = {name for name in values if _HASH_NAME.fullmatch(name)}
        locks = {
            name[1:-len(".publish.lock")]
            for name in values
            if name.startswith(".") and name.endswith(".publish.lock")
        }
        if not digests or len(digests) > _EXPECTED_WORKSPACE_ROWS or digests != locks:
            raise ProductionLatentTrainingCorpusError(
                "execution cache keys/locks changed schema"
            )
        return
    elif len(path) == 3 and path[:2] == ("cache", "stores"):
        expected = {"base-manifest.json", "store"}
    elif len(path) == 4 and path[:2] == ("cache", "stores") and path[3] == "store":
        expected = {"memory.db", "hnsw_index.bin"}
    elif len(path) == 3 and path[:2] == ("cache", "query-inputs"):
        expected = {"frozen-query-inputs.json", "query-manifest.json"}
    else:
        raise ProductionLatentTrainingCorpusError(
            "execution workspace exceeded its exact schema"
        )
    if values != expected:
        raise ProductionLatentTrainingCorpusError(
            "execution workspace changed its exact schema"
        )


_EXPECTED_WORKSPACE_ROWS = 300


def _directory_expected(
    kind: _WorkspaceKind, path: tuple[str, ...], name: str
) -> bool:
    if kind == "row":
        return not path and name == "derived"
    if not path:
        return True
    if path == ("cache",):
        return True
    if path == ("rows",):
        raise ProductionLatentTrainingCorpusError("rows workspace must be empty")
    if path in {("cache", "stores"), ("cache", "query-inputs")}:
        return _HASH_NAME.fullmatch(name) is not None
    if len(path) == 3 and path[:2] == ("cache", "stores"):
        return name == "store"
    return False


def _snapshot_tree(
    directory: _OpenEntry,
    path: tuple[str, ...],
    kind: _WorkspaceKind,
    values: list[tuple[tuple[str, ...], bool, tuple[int, ...]]],
) -> None:
    if len(path) >= _MAX_DEPTH:
        raise ProductionLatentTrainingCorpusError(
            "candidate workspace exceeded its depth cap"
        )
    names = _entry_names(directory, cap=_MAX_DIRECTORY_ENTRIES)
    _schema(kind, path, names)
    for name in names:
        expected_directory = _directory_expected(kind, path, name)
        child = _open_child(
            directory, name, directory=expected_directory
        )
        child_path = (*path, name)
        try:
            if len(values) >= _MAX_TOTAL_ENTRIES:
                raise ProductionLatentTrainingCorpusError(
                    "candidate workspace exceeded its object cap"
                )
            if not child.is_directory:
                _require_one_link(child)
            values.append((child_path, child.is_directory, child.identity))
            if child.is_directory:
                _snapshot_tree(child, child_path, kind, values)
            _assert_bound(directory, name, child)
        finally:
            _close_entry(child)


def _observe(owner: OwnedCandidateWorkspace) -> tuple[
    _WorkspaceState,
    tuple[tuple[tuple[str, ...], bool, tuple[int, ...]], ...],
]:
    chain, parent, root, state = _open_owned(owner)
    try:
        values: list[tuple[tuple[str, ...], bool, tuple[int, ...]]] = []
        _snapshot_tree(root, (), state.kind, values)
        _assert_named_object(parent, state.path.name, root)
        inventory = tuple(sorted(values, key=lambda item: item[0]))
        return state, inventory
    finally:
        for entry in reversed(chain):
            _close_entry(entry)


def _capture(owner: OwnedCandidateWorkspace, kind: _WorkspaceKind) -> None:
    state, inventory = _observe(owner)
    if state.kind != kind:
        raise TypeError("candidate workspace has another capture role")
    if state.captured:
        if state.inventory != inventory:
            raise ProductionLatentTrainingCorpusError(
                "candidate workspace changed after capture"
            )
        return
    _replace(owner, state, captured=True, inventory=inventory)


def capture_candidate_row_workspace(owner: OwnedCandidateWorkspace) -> None:
    _capture(owner, "row")


def capture_candidate_execution_workspace(owner: OwnedCandidateWorkspace) -> None:
    _capture(owner, "execution")


def _rename_to_tombstone(
    owner: OwnedCandidateWorkspace,
    state: _WorkspaceState,
) -> _WorkspaceState:
    tomb = state.parent / (
        f".{state.path.name}.candidate-cleanup-{secrets.token_hex(16)}"
    )
    if os.path.lexists(tomb):
        raise FileExistsError(tomb)
    if not state.captured or state.held is None:
        raise ProductionLatentTrainingCorpusError(
            "uncaptured candidate workspace cannot be deleted"
        )
    chain = _open_chain(state.parent)
    promoter: _OpenEntry | None = None
    try:
        parent = chain[-1]
        if os.name == "nt":
            _assert_named_object(parent, state.path.name, state.held)
            _close_entry(state.held)
            state = _replace(owner, state, held=None)
            promoter = _win_open(
                state.path, directory=True, delete_access=True
            )
            if not _same_object(promoter.identity, state.identity):
                raise ProductionLatentTrainingCorpusError(
                    "candidate workspace changed before cleanup rename"
                )
            _win_rename(promoter, parent, tomb.name)
        else:
            promoter = _open_child(parent, state.path.name, directory=True)
            if not _same_object(promoter.identity, state.identity):
                raise ProductionLatentTrainingCorpusError(
                    "candidate workspace changed before cleanup rename"
                )
            _posix_rename_noreplace(parent, state.path.name, tomb.name)
        # Registry transition is deliberately the first fallible operation
        # after rename.
        updated = _replace(
            owner, state, path=tomb, prefix=tomb.name, held=None
        )
        if os.name != "nt" and state.held is not None:
            _close_entry(state.held)
        return updated
    finally:
        if promoter is not None:
            _close_entry(promoter)
        for entry in reversed(chain):
            _close_entry(entry)


def _expected_children(
    inventory: tuple[tuple[tuple[str, ...], bool, tuple[int, ...]], ...],
    path: tuple[str, ...],
) -> dict[str, tuple[bool, tuple[int, ...]]]:
    return {
        relative[-1]: (directory, identity)
        for relative, directory, identity in inventory
        if len(relative) == len(path) + 1 and relative[:-1] == path
    }


def _delete_directory(
    parent: _OpenEntry,
    name: str,
    path: tuple[str, ...],
    expected_identity: tuple[int, ...],
    inventory: tuple[tuple[tuple[str, ...], bool, tuple[int, ...]], ...],
) -> None:
    directory = _open_child(parent, name, directory=True)
    try:
        if not _same_object(directory.identity, expected_identity):
            raise ProductionLatentTrainingCorpusError(
                "candidate workspace directory was replaced"
            )
        expected = _expected_children(inventory, path)
        names = _entry_names(directory, cap=_MAX_DIRECTORY_ENTRIES)
        if set(names) != set(expected):
            raise ProductionLatentTrainingCorpusError(
                "candidate workspace changed after inventory capture"
            )
        for child_name in names:
            is_directory, identity = expected[child_name]
            child_path = (*path, child_name)
            if is_directory:
                _delete_directory(
                    directory, child_name, child_path, identity, inventory
                )
                continue
            child = _open_child(directory, child_name, directory=False)
            try:
                _require_one_link(child)
                if not _same_object(child.identity, identity):
                    raise ProductionLatentTrainingCorpusError(
                        "candidate workspace file was replaced"
                    )
                if os.name == "nt":
                    _close_entry(child)
                    child = _open_child(
                        directory,
                        child_name,
                        directory=False,
                        delete_access=True,
                    )
                    if not _same_object(child.identity, identity):
                        raise ProductionLatentTrainingCorpusError(
                            "candidate workspace file changed before delete"
                        )
                    _assert_bound(directory, child_name, child)
                    _win_mark_delete(child)
                else:
                    _assert_bound(directory, child_name, child)
                    os.unlink(child_name, dir_fd=directory.handle)
            finally:
                _close_entry(child)
        _assert_bound(parent, name, directory)
    finally:
        _close_entry(directory)
    if os.name == "nt":
        delete = _open_child(parent, name, directory=True, delete_access=True)
        try:
            if not _same_object(delete.identity, expected_identity):
                raise ProductionLatentTrainingCorpusError(
                    "candidate workspace directory changed before delete"
                )
            _assert_bound(parent, name, delete)
            _win_mark_delete(delete)
        finally:
            _close_entry(delete)
    else:
        os.rmdir(name, dir_fd=parent.handle)


def cleanup_candidate_workspace(owner: OwnedCandidateWorkspace) -> None:
    state = _state(owner)
    if not state.captured:
        raise ProductionLatentTrainingCorpusError(
            "uncaptured candidate workspace cleanup is refused"
        )
    observed_state, inventory = _observe(owner)
    if observed_state is not state or inventory != state.inventory:
        raise ProductionLatentTrainingCorpusError(
            "candidate workspace changed after capture"
        )
    state = _rename_to_tombstone(owner, state)
    chain = _open_chain(state.parent)
    try:
        parent = chain[-1]
        _delete_directory(
            parent,
            state.path.name,
            (),
            state.identity,
            state.inventory,
        )
        _flush_open_directory(parent)
        _revoke(owner, _state(owner))
    finally:
        for entry in reversed(chain):
            _close_entry(entry)


__all__: list[str] = []
