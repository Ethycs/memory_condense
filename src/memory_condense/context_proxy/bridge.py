"""Join a model request to the on-disk transcript it came from.

Requests carry no session id we can rely on, but the task framing (the first
user message) is identical in the request and in the transcript's first
``user`` record. The locator fingerprints that text and matches it against
the live sessions found by ``live_sessions``, so a memory system asked to
condense a request can open the original transcript read-only.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from memory_condense.live_sessions import MappedSession, SessionRef, live_sessions

from .policy import fingerprint


def _first_user_text(path: str, *, max_bytes: int = 4_000_000) -> str:
    with MappedSession(path) as m:
        for rec in m.records():
            if rec.offset > max_bytes:
                break
            try:
                obj = rec.json()
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "user":
                content = (obj.get("message") or {}).get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return "\n".join(
                        c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"
                    )
            if obj.get("type") == "response_item":  # Codex rollout
                p = obj.get("payload") or {}
                if p.get("type") == "message" and p.get("role") == "user":
                    c = p.get("content")
                    if isinstance(c, list):
                        return "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
    return ""


@dataclass
class SessionLocator:
    """Map request fingerprints to live on-disk sessions, with a short cache."""

    ttl: float = 15.0
    _index: dict[str, SessionRef] = field(default_factory=dict)
    _built_at: float = 0.0
    _seen: dict[str, str] = field(default_factory=dict)  # session key -> fingerprint

    def refresh(self) -> None:
        now = time.time()
        if now - self._built_at < self.ttl:
            return
        for ref in live_sessions():
            if ref.key in self._seen:
                continue
            try:
                fp = fingerprint(_first_user_text(ref.path))
            except OSError:
                continue
            self._seen[ref.key] = fp
            self._index.setdefault(fp, ref)
        self._built_at = now

    def locate(self, head_text: str) -> SessionRef | None:
        self.refresh()
        return self._index.get(fingerprint(head_text))


def locate_for_units(locator: SessionLocator, units: list[Any]) -> SessionRef | None:
    from .units import head_length

    head = head_length(units)
    head_text = "\n".join(u.user_text for u in units[:head] if u.user_text)
    return locator.locate(head_text) if head_text else None
