"""Store snapshots and the swappable live store behind the proxy's sink.

Two things live here because they share one invariant: the SQLite file under
``<data-dir>/memory.db`` is only ever copied, replaced, or cleared while no
connection is open on it.

:class:`SnapshotStore` owns ``<data-dir>/snapshots``. ``save`` is SQLite's own
``VACUUM INTO`` (a consistent copy even under WAL writers) into a temporary
name, then an ``os.replace`` so a reader never sees a torn snapshot.

:class:`CondenserHandle` owns the live condenser the capture sink writes to.
The sink, the completion callback, and ``swap`` all run under one re-entrant
lock: a swap waits for the in-flight batch to finish, closes the condenser,
mutates the files, and reopens, while later batches wait on the lock rather
than touching a half-swapped store. The hnswlib image is disposable (the
retriever reconciles from SQLite when it is missing), so a swap deletes it.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable

from memory_condense.control.status import now_iso

logger = logging.getLogger(__name__)

SNAPSHOT_DIRNAME = "snapshots"
SNAPSHOT_SUFFIX = ".sqlite"
DB_FILENAME = "memory.db"
INDEX_FILENAME = "hnsw_index.bin"

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

        Returns ``mutate``'s result. If reopening fails the handle stays
        closed and the error propagates; the sink then fails loudly instead of
        writing into a store of unknown state.
        """
        with self._lock:
            self.close()
            result = mutate()
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
    "SNAPSHOT_DIRNAME",
    "SnapshotStore",
    "timestamp_name",
    "validate_snapshot_name",
]
