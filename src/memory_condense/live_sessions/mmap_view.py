"""Seamless read-only memory-mapping of live agent session transcripts.

A running Claude or Codex session keeps its transcript ``.jsonl`` open for
append. :class:`MappedSession` maps that file read-only, with shared access, so
the writer is never perturbed, and remaps on growth so a caller can follow an
in-flight session as bytes land. Only complete, newline-terminated records are
surfaced -- a half-written trailing line stays invisible until its ``\\n``
arrives, so a consumer never parses a torn record.

This is the read-side interposition point the whole package is built on: it is
how you "virtualize" a live session without touching the process that owns it.
"""

from __future__ import annotations

import json
import mmap
import os
import time
from dataclasses import dataclass
from typing import Callable, Iterator

_O_BINARY = getattr(os, "O_BINARY", 0)  # no-op on POSIX


@dataclass(slots=True)
class Record:
    """One complete JSONL line, with its byte offset in the mapped file."""

    offset: int
    raw: bytes

    def json(self) -> dict:
        return json.loads(self.raw)


class MappedSession:
    """A read-only, growth-following mmap over an append-only JSONL file.

    On Windows a mapping cannot extend past end-of-file, so growth is handled
    by remapping rather than by over-mapping. The mapping is opened with the C
    runtime's default shared mode, which permits reading a file another process
    holds open for writing.
    """

    def __init__(self, path: os.PathLike[str] | str) -> None:
        self.path = os.fspath(path)
        self._fd = -1
        self._mm: mmap.mmap | None = None
        self._mapped_size = 0
        self._scan_end = 0
        self.open()

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> None:
        self._fd = os.open(self.path, os.O_RDONLY | _O_BINARY)
        self._remap()

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> "MappedSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- mapping -----------------------------------------------------------
    def _file_size(self) -> int:
        return os.fstat(self._fd).st_size

    def _remap(self) -> bool:
        """(Re)map the current extent. Returns True if the visible size changed."""
        size = self._file_size()
        if self._mm is not None and size == self._mapped_size:
            return False
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if size == 0:
            # Windows refuses to mmap an empty file; treat as an empty view.
            changed = self._mapped_size != 0
            self._mapped_size = 0
            return changed
        self._mm = mmap.mmap(self._fd, size, access=mmap.ACCESS_READ)
        self._mapped_size = size
        return True

    def refresh(self) -> bool:
        """Remap if the file has grown. True when new bytes are now visible.

        Any :class:`memoryview` handed out by :meth:`view` before a refresh is
        invalidated by the remap; re-fetch it after calling this.
        """
        return self._remap()

    @property
    def size(self) -> int:
        return self._mapped_size

    def view(self) -> memoryview:
        """Zero-copy view of the mapped bytes. Empty when the file is empty."""
        if self._mm is None:
            return memoryview(b"")
        return memoryview(self._mm)

    def read(self, start: int = 0, end: int | None = None) -> bytes:
        if self._mm is None:
            return b""
        end = self._mapped_size if end is None else end
        return self._mm[start:end]

    # -- records -----------------------------------------------------------
    def records(self, start: int = 0) -> Iterator[Record]:
        """Yield complete records from ``start``; skip a torn trailing line.

        Refreshes the mapping first. After iterating, :meth:`last_offset`
        gives the byte offset just past the final complete record -- pass it
        back as ``start`` to resume without re-reading.
        """
        self.refresh()
        if self._mm is None:
            self._scan_end = start
            return
        data = self._mm
        size = self._mapped_size
        pos = min(start, size)
        last = pos
        while True:
            nl = data.find(b"\n", pos, size)
            if nl == -1:
                break
            line = data[pos:nl]
            last = nl + 1
            # Advance before yielding so a consumer that stops early can
            # still resume from last_offset() without re-reading.
            self._scan_end = last
            if line.strip():
                yield Record(offset=pos, raw=bytes(line))
            pos = nl + 1
        self._scan_end = last

    def last_offset(self) -> int:
        return self._scan_end

    def follow(
        self,
        start: int = 0,
        *,
        poll: float = 0.25,
        stop: Callable[[], bool] | None = None,
    ) -> Iterator[Record]:
        """Yield records as they are appended, forever (or until ``stop``)."""
        offset = start
        while True:
            for rec in self.records(offset):
                yield rec
            offset = self._scan_end
            if stop is not None and stop():
                return
            time.sleep(poll)
