"""Context policies: decide which units the model sees and what replaces the rest.

A policy never edits inside a unit. It keeps the head (task framing), keeps the
newest units up to a token budget, and replaces the elided middle with one
memory block produced by a ``condense`` callable. The default condenser is a
deterministic extractive digest; the memory_condense pipeline plugs in through
the same callable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .units import Format, Unit, head_length, tool_histogram


@dataclass
class CondenseRequest:
    """Everything a memory system gets when asked for the block that replaces
    the elided span. ``session`` is the on-disk original (a
    ``live_sessions.SessionRef``) when the bridge could locate it -- open it
    read-only via ``live_sessions.MappedSession``; the proxy never writes it.
    """

    fmt: Format
    dropped: list[Unit]
    kept: list[Unit]
    head_text: str
    fingerprint: str  # stable id for the conversation: sha256 of the task framing
    session: Any | None = None


Condenser = Callable[[CondenseRequest], str]


def fingerprint(head_text: str) -> str:
    return hashlib.sha256(" ".join(head_text.split()).encode("utf-8")).hexdigest()[:16]


@dataclass
class Plan:
    keep: list[bool]
    memory_text: str | None = None
    insert_after: int = -1  # unit index after which the memory block goes
    dropped: list[Unit] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return not all(self.keep) or self.memory_text is not None


class ContextPolicy(Protocol):
    name: str

    def plan(self, units: list[Unit], fmt: Format, session: Any | None = None) -> Plan: ...


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def extractive_digest(req: CondenseRequest, *, per_item: int = 240, max_chars: int = 6000) -> str:
    """What happened in the elided span, without inventing anything."""
    dropped = req.dropped
    lines = [
        f"[Context condensed by memory_condense proxy: {len(dropped)} earlier exchanges "
        "elided. The full transcript remains available to the user; ask if a detail "
        "from this span is needed.]"
    ]
    for u in dropped:
        if u.user_text:
            lines.append(f"- User asked: {_truncate(u.user_text, per_item)}")
    tools = tool_histogram(dropped)
    if tools:
        lines.append("- Tools used in the elided span: " + ", ".join(f"{n}×{c}" for n, c in tools.most_common()))
    out = "\n".join(lines)
    return out if len(out) <= max_chars else out[: max_chars - 1] + "…"


@dataclass
class PassthroughPolicy:
    """Forward everything unchanged; useful to validate plumbing and to measure."""

    name: str = "passthrough"

    def plan(self, units: list[Unit], fmt: Format = "anthropic", session: Any | None = None) -> Plan:
        return Plan(keep=[True] * len(units))


@dataclass
class RecencyWindowPolicy:
    """Keep head + newest units within ``budget_tokens``; digest the middle."""

    budget_tokens: int = 60_000
    condense: Condenser = extractive_digest
    min_dropped: int = 2  # trimming fewer units than this is not worth a cache miss
    name: str = "recency-window"

    def plan(self, units: list[Unit], fmt: Format = "anthropic", session: Any | None = None) -> Plan:
        n = len(units)
        keep = [False] * n
        head = head_length(units)
        for i in range(head):
            keep[i] = True
        remaining = self.budget_tokens - sum(units[i].tokens for i in range(head))
        i = n - 1
        while i >= head:
            cost = units[i].tokens
            if remaining - cost < 0 and i != n - 1:  # the newest unit is always kept
                break
            keep[i] = True
            remaining -= cost
            i -= 1
        dropped = [u for u, k in zip(units, keep) if not k]
        if len(dropped) < self.min_dropped:
            return Plan(keep=[True] * n)
        head_text = "\n".join(u.user_text for u in units[:head] if u.user_text)
        req = CondenseRequest(
            fmt=fmt,
            dropped=dropped,
            kept=[u for u, k in zip(units, keep) if k],
            head_text=head_text,
            fingerprint=fingerprint(head_text),
            session=session,
        )
        return Plan(keep=keep, memory_text=self.condense(req), insert_after=head - 1, dropped=dropped)


def memory_item(fmt: Format, text: str) -> dict:
    if fmt == "anthropic":
        return {"role": "user", "content": [{"type": "text", "text": text}]}
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def apply(fmt: Format, units: list[Unit], plan: Plan) -> list[dict]:
    out: list[dict] = []
    for i, (unit, keep) in enumerate(zip(units, plan.keep)):
        if keep:
            out.extend(unit.items)
        if plan.memory_text is not None and i == plan.insert_after:
            out.append(memory_item(fmt, plan.memory_text))
    return out
