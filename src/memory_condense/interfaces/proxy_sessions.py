"""Attribute a proxied request to a conversation without requiring a header.

Claude Code and Codex send no ``x-memory-conversation-id``; before this module
observe mode made every turn its own conversation and augment mode answered
400. Resolution order, first hit wins:

1. explicit header (``x-memory-conversation-id`` / ``x-conversation-id``)
2. the client's own id: Codex sends a ``session_id`` request header; Claude
   Code's ``metadata.user_id`` may carry ``session_<uuid>``
3. the on-disk transcript whose first user message matches this request's
   (located through ``live_sessions``; the file is only ever read)
4. the request digest -- no threading, which is correct but memoryless

Locating never blocks inference: the transcript index refreshes on a
background thread and a stale or empty index simply falls through.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
_SESSION_IN_USER_ID = re.compile(r"session_(" + _UUID + r")")
_BARE_UUID = re.compile(r"\A" + _UUID + r"\Z")
_EXPLICIT = ("x-memory-conversation-id", "x-conversation-id")
_CLIENT_SESSION = ("session_id", "x-session-id", "x-codex-session-id")
_MAX_ID = 512

SOURCE_HEADER = "header"
SOURCE_CLIENT = "client"
SOURCE_TRANSCRIPT = "transcript"
SOURCE_DIGEST = "digest"


def _lower(headers: Mapping[str, str]) -> dict[str, str]:
    return {k.lower(): v for k, v in headers.items()}


def explicit_conversation_id(headers: Mapping[str, str]) -> str | None:
    low = _lower(headers)
    for key in _EXPLICIT:
        value = low.get(key)
        if value and value.strip():
            return value.strip()[:_MAX_ID]
    return None


def client_native_id(headers: Mapping[str, str], payload: Mapping[str, Any] | None) -> str | None:
    low = _lower(headers)
    for key in _CLIENT_SESSION:
        value = low.get(key)
        if value and value.strip():
            ident = value.strip()[:128]
            # The ``metadata.user_id`` path below casefolds; a client that
            # spells the same uuid both ways (Claude Code emits upper case)
            # would otherwise get two ids for one session. Only bare uuids
            # are folded -- an opaque id may be case-sensitive, and folding
            # that could merge two distinct sessions.
            return "session:" + (ident.lower() if _BARE_UUID.match(ident) else ident)
    if payload:
        metadata = payload.get("metadata")
        user_id = metadata.get("user_id") if isinstance(metadata, dict) else None
        if isinstance(user_id, str):
            match = _SESSION_IN_USER_ID.search(user_id)
            if match:
                return "session:" + match.group(1).lower()
    return None


def _text_blocks(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") in ("text", "input_text")
        )
    return ""


def head_text(payload: Mapping[str, Any] | None) -> str:
    """First user message of the request, whichever wire format carries it."""
    if not payload:
        return ""
    items = payload.get("messages")
    if not isinstance(items, list):
        items = payload.get("input")
    if not isinstance(items, list):
        return ""
    for item in items:
        if isinstance(item, dict) and item.get("role") == "user":
            text = _text_blocks(item.get("content")).strip()
            if text:
                return text
    return ""


def fingerprint(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()[:16]


def first_user_text_of_transcript(path: str, *, max_bytes: int = 4_000_000) -> str:
    """First user message in a Claude Code or Codex transcript (read-only mmap)."""
    from memory_condense.live_sessions import MappedSession

    with MappedSession(path) as mapped:
        for rec in mapped.records():
            if rec.offset > max_bytes:
                break
            try:
                obj = rec.json()
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "user":
                text = _text_blocks((obj.get("message") or {}).get("content")).strip()
                if text:
                    return text
            elif obj.get("type") == "response_item":
                item = obj.get("payload") or {}
                if item.get("type") == "message" and item.get("role") == "user":
                    text = _text_blocks(item.get("content")).strip()
                    if text:
                        return text
    return ""


@dataclass
class TranscriptLocator:
    """fingerprint(first user message) -> live on-disk transcript.

    ``refresh`` scans live sessions (a psutil handle walk, ~1 s) on a
    background thread; ``locate`` only reads the current index, plus one
    ``os.stat`` per candidate.

    What matters is freshness, not size: an index entry is evidence that a
    session was live *at scan time* and nothing more. So each scan rebuilds
    the index from the current live set -- ``live_sessions`` already drops
    sessions that have exited, which evicts a finished transcript (and its
    frozen ``liveness``/``mtime``) within ``ttl`` -- and ``locate`` re-stats
    the files it is choosing between instead of trusting the scan's mtime.
    Past ``max_age`` a scan's liveness tier stops counting as evidence at
    all: such an entry only survives if its transcript was appended to
    within ``stale_after``, and the tiebreak is then recency alone, so a
    cached ``handle`` from a session that has since exited cannot outrank a
    session that is demonstrably still being written.
    """

    ttl: float = 15.0  # minimum seconds between scans
    stale_after: float = 120.0  # unwritten this long = not live (cf. DEFAULT_LIVE_WINDOW_S)
    max_age: float = 60.0  # past this, a scan's liveness tier is no longer evidence
    _index: dict[str, list[Any]] = field(default_factory=dict)
    _seen: dict[str, str] = field(default_factory=dict)
    _built_at: float = 0.0  # last scan attempt: throttles refresh
    _scanned_at: float = 0.0  # last scan that installed an index: ages its entries
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _refreshing: bool = False

    def _scan(self) -> None:
        """Rebuild the index from the live set; cache first-message reads."""
        from memory_condense.live_sessions import live_sessions

        refs = live_sessions()
        with self._lock:
            cached = dict(self._seen)  # path key -> fingerprint, from earlier scans
        seen: dict[str, str] = {}
        index: dict[str, list[Any]] = {}
        for ref in refs:
            key = getattr(ref, "key", None) or getattr(ref, "path", "")
            fp = cached.get(key)
            if fp is None:
                try:
                    text = first_user_text_of_transcript(ref.path)
                except Exception:  # unreadable/torn transcript: skip this one only
                    continue
                if not text:
                    # No user message yet (a session that just started). Leave it
                    # uncached so the next scan sees the real first message.
                    continue
                fp = fingerprint(text)
            seen[key] = fp
            index.setdefault(fp, []).append(ref)
        now = time.time()
        with self._lock:
            self._index = index  # installed whole: locate never sees a partial index
            self._seen = seen  # pruned to the live set, so it cannot grow without bound
            self._scanned_at = now

    def refresh(self, *, wait: bool = False) -> None:
        with self._lock:
            if self._refreshing or time.time() - self._built_at < self.ttl:
                return
            self._refreshing = True

        def run() -> None:
            try:
                self._scan()
            except Exception:  # attribution is best-effort by contract
                pass
            finally:
                with self._lock:
                    self._built_at = time.time()
                    self._refreshing = False

        if wait:
            run()
        else:
            threading.Thread(target=run, name="transcript-locator", daemon=True).start()

    def _mtime(self, ref: Any) -> float | None:
        """Current mtime of a candidate's transcript, or ``None`` if it is gone."""
        path = getattr(ref, "path", None)
        if not isinstance(path, str):
            return None
        try:
            return os.stat(path).st_mtime
        except OSError:  # deleted, renamed or unreadable -> not a candidate
            return None

    def locate(self, text: str) -> Any | None:
        if not text:
            return None
        self.refresh()
        with self._lock:
            candidates = list(self._index.get(fingerprint(text), ()))
            scanned_at = self._scanned_at
        if not candidates:
            return None
        now = time.time()
        trusted = now - scanned_at < self.max_age  # strict: max_age=0 trusts nothing
        # One os.stat each: drops transcripts that have gone away and gives the
        # recency tiebreak a measurement rather than the scan's frozen mtime.
        pool: list[tuple[float, Any]] = []
        for ref in candidates:
            mtime = self._mtime(ref)
            if mtime is None:
                continue
            if not trusted and now - mtime > self.stale_after:
                continue  # stale scan over a cold file: nothing says it is still live
            pool.append((mtime, ref))
        if not pool:
            return None
        # Tiebreak: from a fresh scan an owned session (handle/process) beats
        # recency-only; then the clearly most recent file. Still ambiguous -> no
        # attribution, and the caller falls through to the request digest.
        if trusted:
            owned = [p for p in pool if getattr(p[1], "liveness", "") in ("handle", "process")]
            pool = owned or pool
        pool.sort(key=lambda p: p[0], reverse=True)
        if len(pool) == 1 or pool[0][0] - pool[1][0] > 5.0:
            return pool[0][1]
        return None


def resolve_conversation_id(
    headers: Mapping[str, str],
    body: bytes,
    *,
    body_sha: str | None = None,
    locator: TranscriptLocator | None = None,
) -> tuple[str, str]:
    """``(conversation_id, source)``; never raises, never blocks on locating."""
    explicit = explicit_conversation_id(headers)
    if explicit:
        return explicit, SOURCE_HEADER
    payload: Mapping[str, Any] | None = None
    try:
        parsed = json.loads(body) if body else None
        payload = parsed if isinstance(parsed, dict) else None
    except ValueError:
        payload = None
    native = client_native_id(headers, payload)
    if native:
        return native, SOURCE_CLIENT
    if locator is not None:
        try:
            ref = locator.locate(head_text(payload))
        except Exception:
            ref = None
        if ref is not None:
            return f"transcript:{getattr(ref, 'session_id', '')}", SOURCE_TRANSCRIPT
    sha = body_sha or hashlib.sha256(body).hexdigest()
    return f"proxy:{sha[:16]}", SOURCE_DIGEST
