"""Memory-mapped access to on-disk chat transcripts that keep changing.

A vendor export is read once to find the byte range of every top-level
conversation, then conversations are decoded one at a time from the mapping.
A multi-hundred-megabyte export therefore never enters the process as one
Python object, and re-reading a single conversation costs a slice rather than
a re-parse of the file.

Mutation is handled by re-indexing, not by writing through the mapping.  The
file is opened read-only on purpose: this system pins ``sha256`` on every
corpus source and ``quote_sha256`` on every evidence span, so editing bytes
underneath stored spans would silently invalidate provenance for everything
already ingested from that file.  Live transcripts are instead handled as
append-mostly — :meth:`TranscriptFile.refresh` detects growth, re-indexes only
the tail, and reports which conversations are new or changed so a caller can
ingest just those.
"""

from __future__ import annotations

import hashlib
import json
import mmap
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from memory_condense.ingest.transcripts import (
    TranscriptMessage,
    parse_conversation_record,
    parse_transcript_jsonl,
)


_READ_BLOCK = 8 * 1024 * 1024

_WHITESPACE = b" \t\n\r"
_OPENERS = b"{["
_CLOSERS = b"}]"


@dataclass(frozen=True, slots=True)
class ConversationSpan:
    """Byte range of one top-level conversation inside its source file."""

    index: int
    start: int
    end: int
    sha256: str

    @property
    def byte_size(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class TranscriptDelta:
    """What changed between two indexes of the same path."""

    status: str
    added: tuple[ConversationSpan, ...] = ()
    changed: tuple[ConversationSpan, ...] = ()
    removed: tuple[int, ...] = ()

    @property
    def is_unchanged(self) -> bool:
        return self.status == "unchanged"

    @property
    def pending(self) -> tuple[ConversationSpan, ...]:
        """Conversations a caller still needs to ingest."""
        return self.added + self.changed


@dataclass(frozen=True, slots=True)
class TranscriptIndex:
    """Immutable index of one transcript file at one observed size."""

    path: Path
    byte_size: int
    sha256: str
    layout: str
    spans: tuple[ConversationSpan, ...] = field(default=())

    def span_digests(self) -> Mapping[int, str]:
        return {span.index: span.sha256 for span in self.spans}

    def diff(self, previous: "TranscriptIndex | None") -> TranscriptDelta:
        """Classify this index against an earlier one for the same path."""
        if previous is None:
            return TranscriptDelta(status="new", added=self.spans)
        if previous.sha256 == self.sha256 and previous.byte_size == self.byte_size:
            return TranscriptDelta(status="unchanged")
        before = previous.span_digests()
        now = self.span_digests()
        added = tuple(
            span for span in self.spans if span.index not in before
        )
        changed = tuple(
            span
            for span in self.spans
            if span.index in before and before[span.index] != span.sha256
        )
        removed = tuple(sorted(set(before) - set(now)))
        status = "appended" if not changed and not removed else "rewritten"
        return TranscriptDelta(
            status=status,
            added=added,
            changed=changed,
            removed=removed,
        )


def _digest_stream(data: mmap.mmap, start: int, end: int) -> str:
    digest = hashlib.sha256()
    position = start
    while position < end:
        stop = min(position + _READ_BLOCK, end)
        digest.update(data[position:stop])
        position = stop
    return digest.hexdigest()


def _skip_whitespace(data: mmap.mmap, position: int, end: int) -> int:
    while position < end and data[position] in _WHITESPACE:
        position += 1
    return position


def _scan_value(data: mmap.mmap, position: int, end: int) -> int:
    """Return the byte just past the JSON value starting at ``position``.

    Tracks string state and escapes so braces inside quoted text never move
    the structural depth.  Scalars end at the first structural byte.
    """
    depth = 0
    in_string = False
    escaped = False
    while position < end:
        byte = data[position]
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:  # backslash
                escaped = True
            elif byte == 0x22:  # quote
                in_string = False
        elif byte == 0x22:
            in_string = True
        elif byte in _OPENERS:
            depth += 1
        elif byte in _CLOSERS:
            if depth == 0:
                return position
            depth -= 1
            if depth == 0:
                return position + 1
        elif depth == 0 and byte in b",]}":
            return position
        position += 1
    return end


def index_array_spans(data: mmap.mmap, size: int) -> tuple[ConversationSpan, ...]:
    """Index the top-level elements of a JSON array without decoding them."""
    position = _skip_whitespace(data, 0, size)
    if position >= size or data[position] != 0x5B:  # '['
        return ()
    position += 1
    spans: list[ConversationSpan] = []
    while True:
        position = _skip_whitespace(data, position, size)
        if position >= size or data[position] == 0x5D:  # ']'
            break
        start = position
        position = _scan_value(data, position, size)
        if position <= start:
            break
        spans.append(
            ConversationSpan(
                index=len(spans),
                start=start,
                end=position,
                sha256=_digest_stream(data, start, position),
            )
        )
        position = _skip_whitespace(data, position, size)
        if position < size and data[position] == 0x2C:  # ','
            position += 1
    return tuple(spans)


def index_jsonl_spans(data: mmap.mmap, size: int) -> tuple[ConversationSpan, ...]:
    """Index one JSONL record per non-blank line."""
    spans: list[ConversationSpan] = []
    position = 0
    while position < size:
        newline = data.find(b"\n", position)
        stop = size if newline == -1 else newline
        if data[position:stop].strip():
            spans.append(
                ConversationSpan(
                    index=len(spans),
                    start=position,
                    end=stop,
                    sha256=_digest_stream(data, position, stop),
                )
            )
        if newline == -1:
            break
        position = newline + 1
    return tuple(spans)


def detect_layout(data: mmap.mmap, size: int) -> str:
    """Name the container shape: ``array``, ``object``, or ``jsonl``."""
    position = _skip_whitespace(data, 0, size)
    if position >= size:
        return "empty"
    first = data[position]
    if first == 0x5B:
        return "array"
    if first != 0x7B:  # '{'
        return "unknown"
    # A single object is JSONL only when a second record follows it.
    after = _skip_whitespace(data, _scan_value(data, position, size), size)
    return "jsonl" if after < size else "object"


@contextmanager
def map_file(path: Path) -> Iterator[tuple[mmap.mmap, int]]:
    """Map ``path`` read-only, closing the mapping before its handle.

    Windows keeps the file locked until the mapping is closed, so the order
    here is load-bearing rather than stylistic.
    """
    handle = path.open("rb")
    try:
        size = path.stat().st_size
        if size == 0:
            yield mmap.mmap(-1, 1), 0
            return
        data = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            yield data, size
        finally:
            data.close()
    finally:
        handle.close()


class TranscriptFile:
    """A transcript on disk, indexed by byte range and re-readable in place.

    The mapping is opened per operation rather than held open for the object's
    lifetime: a long-lived mapping would pin the file against the writer that
    is still appending to it.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._index: TranscriptIndex | None = None

    @property
    def index(self) -> TranscriptIndex | None:
        return self._index

    def build_index(self) -> TranscriptIndex:
        """Index every top-level conversation without decoding any of them."""
        with map_file(self.path) as (data, size):
            layout = detect_layout(data, size)
            if layout == "array":
                spans = index_array_spans(data, size)
            elif layout == "jsonl":
                spans = index_jsonl_spans(data, size)
            elif layout == "object":
                spans = (
                    ConversationSpan(
                        index=0,
                        start=0,
                        end=size,
                        sha256=_digest_stream(data, 0, size),
                    ),
                )
            else:
                spans = ()
            digest = _digest_stream(data, 0, size)
        return TranscriptIndex(
            path=self.path,
            byte_size=size,
            sha256=digest,
            layout=layout,
            spans=spans,
        )

    def refresh(self) -> TranscriptDelta:
        """Re-index and report what changed since the last :meth:`refresh`."""
        current = self.build_index()
        delta = current.diff(self._index)
        self._index = current
        return delta

    def read_span(self, span: ConversationSpan) -> bytes:
        """Return one conversation's exact bytes from the mapping."""
        with map_file(self.path) as (data, size):
            if span.end > size:
                raise ValueError(
                    "conversation span extends past the current file size; "
                    "call refresh() after the transcript changes"
                )
            return bytes(data[span.start : span.end])

    def decode_span(self, span: ConversationSpan) -> Any:
        """Decode exactly one conversation from the mapping."""
        return json.loads(self.read_span(span).decode("utf-8"))

    def messages_for(self, span: ConversationSpan) -> list[TranscriptMessage]:
        """Parse one conversation's messages straight from its byte range."""
        label = f"{self.path.stem}:{span.index}"
        if self._index is not None and self._index.layout == "jsonl":
            raw = self.read_span(span).decode("utf-8", errors="replace")
            _layout, parsed = parse_transcript_jsonl([raw], source_label=label)
            return parsed
        _format, parsed = parse_conversation_record(
            self.decode_span(span),
            fallback_id=label,
        )
        return parsed

    def iter_messages(
        self,
        spans: tuple[ConversationSpan, ...] | None = None,
    ) -> Iterator[TranscriptMessage]:
        """Stream messages for ``spans``, defaulting to the whole file."""
        if self._index is None:
            self.refresh()
        assert self._index is not None
        for span in spans if spans is not None else self._index.spans:
            yield from self.messages_for(span)

    def pending_messages(self) -> list[TranscriptMessage]:
        """Refresh, then return messages only from new or changed records."""
        delta = self.refresh()
        if delta.is_unchanged:
            return []
        return list(self.iter_messages(delta.pending))


__all__ = [
    "ConversationSpan",
    "TranscriptDelta",
    "TranscriptFile",
    "TranscriptIndex",
    "detect_layout",
    "index_array_spans",
    "index_jsonl_spans",
    "map_file",
]
