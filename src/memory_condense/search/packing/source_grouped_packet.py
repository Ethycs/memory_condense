"""Render already-selected raw evidence with explicit session ownership.

This renderer does not retrieve, summarize, infer source identity, or discard
evidence. Offsets identify the exact raw strings inside the returned context.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence


@dataclass(frozen=True, slots=True)
class RawPacketExcerpt:
    citation: str
    source_id: str
    created_at: str
    role: str
    text: str


@dataclass(frozen=True, slots=True)
class SessionPacketBlock:
    """An existing, fully rendered exchange; its local citations stay intact."""

    citation: str
    source_id: str
    text: str


@dataclass(frozen=True, slots=True)
class PacketTextBinding:
    citation: str
    source_id: str
    session_label: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class GroupedRawPacket:
    context: str
    bindings: tuple[PacketTextBinding, ...]


def render_source_grouped_packet(
    excerpts: Sequence[RawPacketExcerpt],
    exchanges: Sequence[SessionPacketBlock] = (),
    *,
    tail: str = "",
) -> GroupedRawPacket:
    """Group exact source IDs in first-encounter order, retaining all members.

    A timestamp header applies only to the following raw excerpt group. Existing
    exchanges keep their own timestamp and owner headers. Within a timestamp,
    preserve upstream order rather than inventing unavailable turn ordinals.
    """
    sources: OrderedDict[str, tuple[list[RawPacketExcerpt], list[SessionPacketBlock]]] = OrderedDict()
    citations: set[str] = set()
    for row in (*excerpts, *exchanges):
        if any(type(value) is not str or not value for value in (row.citation, row.source_id, row.text)):
            raise ValueError("packet citation, source and raw text must be nonempty strings")
        if row.citation in citations:
            raise ValueError("packet citations must be unique")
        citations.add(row.citation)
        if not row.citation.isascii() or not row.citation.isalnum():
            raise ValueError("packet citations must be ASCII alphanumeric labels")
        group = sources.setdefault(row.source_id, ([], []))
        if isinstance(row, RawPacketExcerpt):
            if row.role not in {"user", "assistant", "system", "tool"}:
                raise ValueError("unsupported evidence role")
            instant = datetime.fromisoformat(row.created_at.replace("Z", "+00:00"))
            if instant.tzinfo is None:
                raise ValueError("evidence timestamp must include its timezone")
            group[0].append(row)
        else:
            group[1].append(row)
    pieces: list[str] = []
    bindings: list[PacketTextBinding] = []
    length = 0

    def append(text: str) -> None:
        nonlocal length
        pieces.append(text)
        length += len(text)

    def raw(row: RawPacketExcerpt | SessionPacketBlock, label: str) -> None:
        start = length
        append(row.text)
        bindings.append(PacketTextBinding(row.citation, row.source_id, label, start, length))

    for number, (source, (rows, blocks)) in enumerate(sources.items(), 1):
        label = f"S{number}"
        if pieces:
            append("\n\n")
        append(f"<{label}>\n")
        dates: OrderedDict[str, list[RawPacketExcerpt]] = OrderedDict()
        for row in rows:
            dates.setdefault(row.created_at, []).append(row)
        for date, members in dates.items():
            append(f"[Excerpt timestamp: {date}]\n")
            for row in members:
                append(f"<{row.citation} {row.role}>\n")
                raw(row, label)
                append("\n\n")
        for block in blocks:
            raw(block, label)
            append("\n\n")
        append(f"</{label}>")
    if tail:
        append("\n\n" if pieces else "")
        append(tail)
    return GroupedRawPacket("".join(pieces), tuple(bindings))
