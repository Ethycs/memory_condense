"""Store snapshots and the swappable live store behind the proxy's sink.

Everything here shares one invariant: a store's files are only ever copied,
replaced, or cleared while no connection is open on them.

Observe mode has one store file, ``<data-dir>/memory.db``.
:class:`SnapshotStore` owns ``<data-dir>/snapshots`` for it. ``save`` is
SQLite's own ``VACUUM INTO`` (a consistent copy even under WAL writers) into a
temporary name, then an ``os.replace`` so a reader never sees a torn snapshot.

Augment mode has no single file: each conversation owns a store tree under
``<data-dir>/sessions/<namespace>``. :class:`SessionSnapshotStore` snapshots
that whole tree into one directory per snapshot -- ``VACUUM INTO`` per database
file, a byte copy for everything else, a ``manifest.json`` naming the schema,
the namespaces, and every file in it -- again staged under a temporary name and
``os.replace``d into place. The two stores share ``<data-dir>/snapshots`` and
never collide: one writes ``*.sqlite`` files, the other directories, and each
``list`` ignores what the other wrote.

:class:`CondenserHandle` owns the live condenser the capture sink writes to.
The sink, the completion callback, and ``swap`` all run under one re-entrant
lock: a swap waits for the in-flight batch to finish, closes the condenser,
mutates the files, and reopens, while later batches wait on the lock rather
than touching a half-swapped store. The hnswlib image is disposable (the
retriever reconciles from SQLite when it is missing), so a swap deletes it.
``MemoryProxy.swap`` is the augment-mode counterpart with the same contract.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Callable

from memory_condense.control.status import now_iso

logger = logging.getLogger(__name__)

SNAPSHOT_DIRNAME = "snapshots"
SNAPSHOT_SUFFIX = ".sqlite"
DB_FILENAME = "memory.db"
INDEX_FILENAME = "hnsw_index.bin"

#: Augment mode's store tree, relative to the data directory.
SESSIONS_DIRNAME = "sessions"
MANIFEST_FILENAME = "manifest.json"
#: Bumped whenever the snapshot directory layout changes incompatibly.
SESSION_SNAPSHOT_SCHEMA = 1

#: Files inside a session tree that ``VACUUM INTO`` owns rather than ``copy``.
_DB_SUFFIXES = (".sqlite", ".sqlite3", ".db")
#: Sidecars belong to the live database, not to a snapshot of it; the owner
#: lock file is recreated by whoever opens the session next.
_SKIP_SUFFIXES = ("-wal", "-shm", "-journal", ".lock")

#: One path component: no separators, no leading dot, nothing SQLite or a
#: shell would misread. ``..`` is rejected separately so it can never appear.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_snapshot_name(name: str) -> str:
    """Return ``name`` without its ``.sqlite`` suffix, or raise ``ValueError``."""
    if not isinstance(name, str):
        raise ValueError("snapshot name must be a string")
    name = name.strip()
    if name.endswith(SNAPSHOT_SUFFIX):
        name = name[: -len(SNAPSHOT_SUFFIX)]
    if not name or ".." in name or not _NAME_RE.match(name):
        raise ValueError(
            "snapshot name must be 1-128 characters of letters, digits, '.', '_' "
            "or '-', start with a letter or digit, and contain no path separators"
        )
    return name


def timestamp_name(now: float | None = None) -> str:
    """``2026-10-13T08-40-00``: sortable, and legal on every filesystem."""
    return time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime(now))


class SnapshotStore:
    """``<data-dir>/snapshots/*.sqlite`` and the file operations on the store."""

    def __init__(
        self,
        data_dir: str | os.PathLike[str],
        *,
        db_path: str | os.PathLike[str] | None = None,
        index_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.snapshot_dir = self.data_dir / SNAPSHOT_DIRNAME
        self.db_path = Path(db_path) if db_path is not None else self.data_dir / DB_FILENAME
        self.index_path = (
            Path(index_path) if index_path is not None else self.data_dir / INDEX_FILENAME
        )

    # -- naming ------------------------------------------------------------
    def path_for(self, name: str) -> Path:
        """The snapshot file for ``name``; validates but does not check existence."""
        return self.snapshot_dir / f"{validate_snapshot_name(name)}{SNAPSHOT_SUFFIX}"

    def _fresh_name(self) -> str:
        base = timestamp_name()
        name = base
        counter = 1
        while (self.snapshot_dir / f"{name}{SNAPSHOT_SUFFIX}").exists():
            name = f"{base}-{counter}"
            counter += 1
        return name

    def _record(self, path: Path) -> dict[str, Any]:
        stat = path.stat()
        return {
            "name": path.name[: -len(SNAPSHOT_SUFFIX)],
            "file": path.name,
            "path": str(path),
            "bytes": stat.st_size,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
        }

    # -- queries -----------------------------------------------------------
    def has_store(self) -> bool:
        """Is there anything to snapshot? (``reset`` backs up only if so.)"""
        return self.db_path.is_file()

    def has_snapshot(self, name: str) -> bool:
        """Does ``name`` name a snapshot ``restore`` would accept?"""
        return self.path_for(name).is_file()

    def list(self) -> list[dict[str, Any]]:
        """Snapshot records, oldest first (timestamp names sort chronologically)."""
        if not self.snapshot_dir.is_dir():
            return []
        out = []
        for path in sorted(self.snapshot_dir.iterdir(), key=lambda p: (p.stat().st_mtime, p.name)):
            if path.is_file() and path.suffix == SNAPSHOT_SUFFIX and not path.name.startswith("."):
                out.append(self._record(path))
        return out

    def latest(self) -> dict[str, Any] | None:
        records = self.list()
        return records[-1] if records else None

    # -- mutations ---------------------------------------------------------
    def save(self, name: str | None = None) -> dict[str, Any]:
        """``VACUUM INTO`` a new snapshot and return its record.

        Safe while the store is open: SQLite's VACUUM INTO reads one
        consistent snapshot even with concurrent WAL writers. Refuses to
        overwrite an existing name.
        """
        if not self.db_path.is_file():
            raise FileNotFoundError(f"no store to snapshot at {self.db_path}")
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        name = self._fresh_name() if name is None else validate_snapshot_name(name)
        target = self.snapshot_dir / f"{name}{SNAPSHOT_SUFFIX}"
        if target.exists():
            raise FileExistsError(f"snapshot {name!r} already exists")
        tmp = self.snapshot_dir / f".{name}.{os.getpid()}.tmp"
        if tmp.exists():
            tmp.unlink()
        conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        try:
            conn.execute("VACUUM INTO ?", (str(tmp),))
        except BaseException:
            conn.close()
            if tmp.exists():
                tmp.unlink()
            raise
        conn.close()
        os.replace(tmp, target)
        logger.info("store snapshot written: %s", target)
        return self._record(target)

    def _sidecars(self) -> list[Path]:
        return [
            Path(f"{self.db_path}-wal"),
            Path(f"{self.db_path}-shm"),
            Path(f"{self.db_path}-journal"),
            self.index_path,
        ]

    def _remove_sidecars(self) -> None:
        for path in self._sidecars():
            if path.exists():
                path.unlink()

    def restore(self, name: str) -> Path:
        """Replace the store file with snapshot ``name``.

        Only call while nothing holds the store open (see
        :meth:`CondenserHandle.swap`). The WAL/SHM sidecars belong to the old
        file and are removed; the ANN image is dropped so the retriever
        rebuilds its mapping from the restored rows.
        """
        source = self.path_for(name)
        if not source.is_file():
            raise FileNotFoundError(f"snapshot {name!r} does not exist")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.db_path.with_name(f".{self.db_path.name}.{os.getpid()}.restore")
        shutil.copyfile(source, tmp)
        self._remove_sidecars()
        os.replace(tmp, self.db_path)
        logger.info("store restored from snapshot %s", source)
        return self.db_path

    def clear(self) -> None:
        """Delete the store file and its sidecars. Only call while closed."""
        self._remove_sidecars()
        if self.db_path.exists():
            self.db_path.unlink()
        logger.info("store cleared: %s", self.db_path)


class _TreeStat:
    """Just enough of ``os.stat_result`` for the status publisher's accounting."""

    __slots__ = ("st_size", "st_mtime")

    def __init__(self, st_size: int, st_mtime: float) -> None:
        self.st_size, self.st_mtime = st_size, st_mtime


class TreePath(os.PathLike):
    """A directory that answers ``stat()`` with the size of everything under it.

    The status publisher renders one store path and sums ``stat().st_size``
    over it and its ``-wal`` sidecar. Augment mode's store *is* a directory
    tree, so reporting the directory inode's own size would publish a
    confident, wrong number; this reports the tree's total instead. Every
    other attribute delegates to the wrapped :class:`~pathlib.Path`.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def __fspath__(self) -> str:
        return str(self.path)

    def __str__(self) -> str:
        return str(self.path)

    def __repr__(self) -> str:
        return f"TreePath({str(self.path)!r})"

    def __eq__(self, other: object) -> bool:
        return self.path == (other.path if isinstance(other, TreePath) else other)

    def __hash__(self) -> int:
        return hash(self.path)

    def __truediv__(self, other: Any) -> Path:
        return self.path / other

    def __getattr__(self, name: str) -> Any:
        if name == "path":  # only reachable before __init__ ran; never recurse
            raise AttributeError(name)
        return getattr(self.path, name)

    def stat(self, **_kwargs: Any) -> _TreeStat:
        return _TreeStat(tree_bytes(self.path), self._mtime())

    def _mtime(self) -> float:
        try:
            return self.path.stat().st_mtime
        except OSError:
            return 0.0


def tree_bytes(root: str | os.PathLike[str]) -> int:
    """Total bytes of every regular file under ``root``; 0 if it is not there."""
    root = Path(root)
    total = 0
    if not root.is_dir():
        try:
            return root.stat().st_size
        except OSError:
            return 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def _skip_in_tree(path: Path) -> bool:
    name = path.name
    return name.startswith(".") or any(name.endswith(suffix) for suffix in _SKIP_SUFFIXES)


def _copy_database(source: Path, target: Path) -> str:
    """``VACUUM INTO`` ``source`` to ``target``, byte-copying what SQLite rejects.

    ``VACUUM INTO`` is the only copy that is consistent while another
    connection is writing in WAL mode. A file that happens to carry a database
    suffix but is not one (an empty placeholder, a foreign format) is still
    part of the tree, so it is copied verbatim rather than failing the save.
    """
    try:
        with closing(sqlite3.connect(str(source), isolation_level=None)) as conn:
            conn.execute("VACUUM INTO ?", (str(target),))
        return "vacuum"
    except sqlite3.Error:
        logger.debug("not a vacuumable database, copying bytes: %s", source, exc_info=True)
        if target.exists():
            target.unlink()
        shutil.copyfile(source, target)
        return "copy"


class SessionSnapshotStore:
    """``<data-dir>/snapshots/<name>/`` over augment mode's ``sessions/`` tree.

    Duck-types :class:`SnapshotStore` for the control routes and the status
    publisher: ``path_for``, ``has_store``, ``list``, ``latest``, ``save``,
    ``restore``, ``clear``, ``snapshot_dir``, and a ``db_path`` that points at
    the tree (see :class:`TreePath`) because that is what "the store" means
    here. Names are validated and generated by the same helpers, so a
    snapshot name means the same thing in either mode.

    A snapshot directory holds ``manifest.json`` plus a ``sessions/`` mirror.
    ``save`` is safe while the sessions are open; ``restore`` and ``clear``
    are only ever called from :meth:`MemoryProxy.swap`, with every session
    closed.
    """

    def __init__(
        self,
        data_dir: str | os.PathLike[str],
        *,
        sessions_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.snapshot_dir = self.data_dir / SNAPSHOT_DIRNAME
        self.sessions_dir = (
            Path(sessions_dir) if sessions_dir is not None else self.data_dir / SESSIONS_DIRNAME
        )

    # -- naming ------------------------------------------------------------
    @property
    def db_path(self) -> TreePath:
        """"The store" for status: the session tree, sized recursively."""
        return TreePath(self.sessions_dir)

    def path_for(self, name: str) -> Path:
        """The snapshot directory for ``name``; validates, does not check existence."""
        return self.snapshot_dir / validate_snapshot_name(name)

    def _fresh_name(self) -> str:
        base = timestamp_name()
        name, counter = base, 1
        while (self.snapshot_dir / name).exists():
            name = f"{base}-{counter}"
            counter += 1
        return name

    def _record(self, path: Path) -> dict[str, Any]:
        count = sum(1 for child in path.rglob("*") if child.is_file())
        stat = path.stat()
        return {
            "name": path.name,
            # The publisher lists snapshots by their "file"; for a tree the
            # directory name is the snapshot, so the two are the same string.
            "file": path.name,
            "path": str(path),
            "bytes": tree_bytes(path),
            "file_count": count,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
        }

    # -- queries -----------------------------------------------------------
    def has_store(self) -> bool:
        return self.sessions_dir.is_dir() and any(self.sessions_dir.iterdir())

    def has_snapshot(self, name: str) -> bool:
        return (self.path_for(name) / MANIFEST_FILENAME).is_file()

    def manifest(self, name: str) -> dict[str, Any]:
        path = self.path_for(name) / MANIFEST_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"snapshot {name!r} does not exist")
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict[str, Any]]:
        """Snapshot records, oldest first. A directory without a manifest is
        a half-written or foreign entry and is not a snapshot."""
        if not self.snapshot_dir.is_dir():
            return []
        out = []
        for path in sorted(self.snapshot_dir.iterdir(), key=lambda p: (p.stat().st_mtime, p.name)):
            if path.is_dir() and not path.name.startswith(".") and (path / MANIFEST_FILENAME).is_file():
                out.append(self._record(path))
        return out

    def latest(self) -> dict[str, Any] | None:
        records = self.list()
        return records[-1] if records else None

    # -- mutations ---------------------------------------------------------
    def save(self, name: str | None = None) -> dict[str, Any]:
        """Snapshot the whole session tree into one directory; return its record.

        Safe while the sessions are open: every database is copied with
        ``VACUUM INTO``, which reads one consistent image even under
        concurrent WAL writers. Callers that want the *tree* consistent
        quiesce first (``MemoryProxy.quiesce``). The snapshot is assembled
        under a dotted temporary name and ``os.replace``d into place, so a
        concurrent ``list`` never sees a partial snapshot. Refuses to
        overwrite an existing name.
        """
        source = self.sessions_dir
        if not source.is_dir():
            raise FileNotFoundError(f"no session stores to snapshot at {source}")
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        name = self._fresh_name() if name is None else validate_snapshot_name(name)
        target = self.snapshot_dir / name
        if target.exists():
            raise FileExistsError(f"snapshot {name!r} already exists")
        tmp = self.snapshot_dir / f".{name}.{os.getpid()}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        files: list[dict[str, Any]] = []
        try:
            (tmp / SESSIONS_DIRNAME).mkdir(parents=True)
            for path in sorted(source.rglob("*")):
                if not path.is_file() or _skip_in_tree(path):
                    continue
                relative = path.relative_to(source)
                destination = tmp / SESSIONS_DIRNAME / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix.lower() in _DB_SUFFIXES:
                    kind = _copy_database(path, destination)
                else:
                    shutil.copyfile(path, destination)
                    kind = "copy"
                files.append(
                    {"path": relative.as_posix(), "kind": kind, "bytes": destination.stat().st_size}
                )
            manifest = {
                "schema": SESSION_SNAPSHOT_SCHEMA,
                "kind": "sessions",
                "name": name,
                "created_at": now_iso(),
                "sessions_dir": str(source),
                "namespaces": sorted(p.name for p in source.iterdir() if p.is_dir()),
                "files": files,
            }
            (tmp / MANIFEST_FILENAME).write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
            os.replace(tmp, target)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        logger.info("session snapshot written: %s (%d files)", target, len(files))
        return self._record(target)

    def restore(self, name: str) -> Path:
        """Replace the session tree with snapshot ``name``.

        Only call while every session store is closed (see
        :meth:`MemoryProxy.swap`). The replacement is staged beside the tree
        and swapped in by rename, and the old tree is only deleted once the
        new one is in place; a rename that fails puts the old tree back.
        """
        source = self.path_for(name)
        if not (source / MANIFEST_FILENAME).is_file():
            raise FileNotFoundError(f"snapshot {name!r} does not exist")
        tree = source / SESSIONS_DIRNAME
        parent = self.sessions_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        staged = parent / f".{self.sessions_dir.name}.{os.getpid()}.restore"
        retired = parent / f".{self.sessions_dir.name}.{os.getpid()}.retired"
        shutil.rmtree(staged, ignore_errors=True)
        shutil.rmtree(retired, ignore_errors=True)
        if tree.is_dir():
            shutil.copytree(tree, staged)
        else:  # a snapshot taken of an empty tree
            staged.mkdir(parents=True)
        moved = False
        try:
            if self.sessions_dir.exists():
                os.replace(self.sessions_dir, retired)
                moved = True
            os.replace(staged, self.sessions_dir)
        except BaseException:
            if moved and not self.sessions_dir.exists():
                os.replace(retired, self.sessions_dir)
            shutil.rmtree(staged, ignore_errors=True)
            raise
        shutil.rmtree(retired, ignore_errors=True)
        logger.info("session stores restored from snapshot %s", source)
        return self.sessions_dir

    def clear(self) -> None:
        """Delete the whole session tree. Only call while closed.

        The tree is renamed aside first, so the directory is gone from the
        live path in one step even if deleting its contents takes a while or
        partially fails.
        """
        if not self.sessions_dir.exists():
            return
        retired = self.sessions_dir.parent / f".{self.sessions_dir.name}.{os.getpid()}.cleared"
        shutil.rmtree(retired, ignore_errors=True)
        os.replace(self.sessions_dir, retired)
        shutil.rmtree(retired, ignore_errors=True)
        logger.info("session stores cleared: %s", self.sessions_dir)


class CondenserHandle:
    """The live condenser behind the proxy's sink, swappable under one lock.

    ``factory`` builds a fresh condenser over ``data_dir`` each time the
    handle opens. Anything that touches the condenser (the capture sink, the
    completion callback, a swap) runs under :attr:`lock`.
    """

    def __init__(
        self,
        factory: Callable[[], Any],
        *,
        data_dir: str | os.PathLike[str],
    ) -> None:
        self._factory = factory
        self.data_dir = Path(data_dir)
        self._lock = threading.RLock()
        self._condenser: Any | None = None
        self.generation = 0
        self.last_swap_at: str | None = None

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    @property
    def closed(self) -> bool:
        return self._condenser is None

    @property
    def condenser(self) -> Any:
        condenser = self._condenser
        if condenser is None:
            raise RuntimeError("memory store is closed")
        return condenser

    @property
    def db_path(self) -> Path:
        condenser = self._condenser
        path = getattr(condenser, "database_path", None) if condenser is not None else None
        return Path(path) if path is not None else self.data_dir / DB_FILENAME

    def open(self) -> Any:
        with self._lock:
            if self._condenser is None:
                self._condenser = self._factory()
                self.generation += 1
            return self._condenser

    def close(self) -> None:
        with self._lock:
            condenser = self._condenser
            self._condenser = None
            if condenser is not None:
                condenser.close()

    def swap(self, mutate: Callable[[], Any]) -> Any:
        """Close, run ``mutate`` on the files, reopen; all under the lock.

        Returns ``mutate``'s result.

        A ``mutate`` that raises is reopened over. Its common failures happen
        *before* anything on disk changes -- a restore whose ``os.replace``
        hits a sharing violation, a pre-reset backup that cannot be written
        -- so the files are intact and the only thing a permanently closed
        handle would buy is a dead proxy: every later capture raising
        ``memory store is closed`` with no recovery short of a restart. The
        error still propagates, so the caller reports the failure.

        If reopening fails the handle stays closed and the error propagates;
        the sink then fails loudly instead of writing into a store of unknown
        state. That is the genuinely unrecoverable case. A reopen that fails
        on the way out of a failed ``mutate`` is logged, not raised, so the
        operator sees the mutation error that caused it.
        """
        with self._lock:
            self.close()
            try:
                result = mutate()
            except BaseException:
                try:
                    self.open()
                except Exception:
                    logger.exception("reopen after a failed swap mutation also failed")
                raise
            self.open()
            self.last_swap_at = now_iso()
            return result

    def capture_sink(self) -> Callable[[Any], bool]:
        from memory_condense.interfaces.proxy_scheduler import condenser_capture_sink

        def sink(capture: Any) -> bool:
            with self._lock:
                return bool(condenser_capture_sink(self.condenser)(capture))

        return sink

    def completion_callback(self, **kwargs: Any) -> Callable[[], Any]:
        from memory_condense.interfaces.proxy_scheduler import condenser_completion_callback

        def complete() -> Any:
            with self._lock:
                return condenser_completion_callback(self.condenser, **kwargs)()

        return complete


__all__ = [
    "CondenserHandle",
    "DB_FILENAME",
    "INDEX_FILENAME",
    "MANIFEST_FILENAME",
    "SESSIONS_DIRNAME",
    "SESSION_SNAPSHOT_SCHEMA",
    "SNAPSHOT_DIRNAME",
    "SessionSnapshotStore",
    "SnapshotStore",
    "TreePath",
    "timestamp_name",
    "tree_bytes",
    "validate_snapshot_name",
]
