"""Serve virtualized views of live sessions.

Two layers:

* :class:`VirtualSession` -- one session, mapped read-only, rendered through a
  record transform (identity by default). This is the in-process view.
* :class:`Mirror` -- a directory that holds a rendered file per *live* session
  and is kept in sync with the tracker: files appear when an agent owns a
  session (an open handle for Codex; a process launched in the project cwd
  for Claude, which never holds one), are refreshed as it appends, and are
  retired when the owner goes away. Point any consumer at the mirror instead of at
  ``~/.claude``; it sees the virtualized transcript, the agent's own file is
  never touched.

OS-level virtualization (a projected filesystem that intercepts opens of the
real paths) is the extension point above ``Mirror``; :func:`projfs_available`
reports whether Windows ProjFS is present for that. It is not required for
the read-side design and is left unimplemented on purpose.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator

from .discovery import SessionKind, SessionRef, discover_sessions
from .handles import HandleTracker
from .mmap_view import MappedSession, Record

RecordTransform = Callable[[list[dict]], list[dict]]


def identity(records: list[dict]) -> list[dict]:
    return records


def _dumps(record: dict) -> bytes:
    return json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class VirtualSession:
    """A live session seen through a transform."""

    def __init__(self, ref: SessionRef, transform: RecordTransform | None = None) -> None:
        self.ref = ref
        self.transform = transform or identity
        self._mapped: MappedSession | None = None

    def mapped(self) -> MappedSession:
        if self._mapped is None:
            self._mapped = MappedSession(self.ref.path)
        else:
            self._mapped.refresh()
        return self._mapped

    def raw_records(self, start: int = 0) -> Iterator[Record]:
        return self.mapped().records(start)

    def records(self) -> list[dict]:
        out: list[dict] = []
        for rec in self.raw_records():
            try:
                out.append(rec.json())
            except json.JSONDecodeError:
                continue
        return self.transform(out)

    def render(self) -> bytes:
        if self.transform is identity:
            # Passthrough is byte-exact: raw lines, not a re-serialization, so
            # offsets and hashes taken against the mirror match the source.
            return b"".join(rec.raw + b"\n" for rec in self.raw_records())
        return b"".join(_dumps(r) + b"\n" for r in self.records())

    def materialize(self, dest: Path | str) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_bytes(self.render())
        tmp.replace(dest)  # atomic swap so a reader never sees a torn file
        return dest

    def close(self) -> None:
        if self._mapped is not None:
            self._mapped.close()
            self._mapped = None

    def __enter__(self) -> "VirtualSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class Mirror:
    """A directory of rendered live sessions, driven by handle tracking."""

    def __init__(
        self,
        root: Path | str,
        transform: RecordTransform | None = None,
        *,
        kinds: Iterable[SessionKind] | None = None,
        include_recent: bool = False,
    ) -> None:
        self.root = Path(root)
        self.transform = transform or identity
        self.kinds = tuple(kinds) if kinds else tuple(SessionKind)
        self.include_recent = include_recent
        self.tracker = HandleTracker(
            under=tuple(r for _, r in _roots(self.kinds)),
        )
        self.served: dict[str, Path] = {}  # session key -> mirror file

    def target_for(self, ref: SessionRef) -> Path:
        return self.root / ref.kind.value / (ref.project or "_") / f"{ref.session_id}.jsonl"

    def select(self) -> list[SessionRef]:
        """Sessions the mirror should serve right now."""
        self.tracker.poll()
        refs = discover_sessions(self.kinds, snap=self.tracker.current)
        wanted = {"handle", "process"}
        if self.include_recent:
            wanted.add("recent")
        return [r for r in refs if r.liveness in wanted]

    def sync(self) -> dict[str, list[str]]:
        """Render live sessions, retire dead ones. Returns what changed."""
        live = self.select()
        live_keys = {r.key for r in live}
        written: list[str] = []
        for ref in live:
            with VirtualSession(ref, self.transform) as vs:
                path = vs.materialize(self.target_for(ref))
            self.served[ref.key] = path
            written.append(str(path))
        retired: list[str] = []
        for key in list(self.served):
            if key not in live_keys:
                path = self.served.pop(key)
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                retired.append(str(path))
        return {"written": written, "retired": retired}

    def serve(self, interval: float = 2.0, *, stop: Callable[[], bool] | None = None) -> None:
        while True:
            self.sync()
            if stop is not None and stop():
                return
            time.sleep(interval)


def _roots(kinds: Iterable[SessionKind]):
    from .discovery import expanded_roots

    return expanded_roots(kinds)


def projfs_available() -> bool:
    """True when Windows Projected File System is installed (for a future OS-level layer)."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        ctypes.WinDLL("ProjectedFSLib.dll")
        return True
    except OSError:
        return False
