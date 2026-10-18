"""Bulk transcript loading: a job, not a stream of captures.

A :class:`BulkLoader` job takes a transcript path or a session id (resolved
read-only through ``live_sessions``), knows ``bytes_total`` up front and
``records_total`` after one index pass, and feeds each user/assistant
exchange through the proxy's own :class:`CaptureQueue` so queue counters stay
coherent with live traffic. Jobs run FIFO on one worker thread.

Progress and ETA come from a **recent window** (``window_s`` seconds, a ring
buffer of ``(t, tokens, records)`` samples), never a lifetime mean: ingest
speed varies with record size, and a lifetime mean lies for the first minute
and the last.

Two token figures, deliberately distinct. ``bytes_done``/``bytes_total``,
``tokens_done`` and the rate window all track the **file**: every JSONL
record, tool calls and meta included, which is what a percentage and an ETA
have to be measured against. ``tokens_ingested`` tracks what the emitted
captures actually carry (:func:`ingested_tokens`), which is the one unit the
status document's ``totals.tokens_ingested`` reports for both bulk jobs and
live traffic.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from memory_condense.control.status import now_iso
from memory_condense.interfaces.proxy_capture import ANTHROPIC, OPENAI, ExchangeCapture
from memory_condense.interfaces.proxy_scheduler import (
    CaptureQueue,
    CaptureQueueClosedError,
    CaptureQueueSaturatedError,
)
from memory_condense.ingest.transcripts import TranscriptMessage, normalize_role

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_S = 30.0
_PROGRESS_NOTIFY_S = 0.2
_ROOM_POLL_S = 0.02

JOB_STATES = ("queued", "indexing", "running", "done", "cancelled", "failed")

#: Codex rollout files announce themselves with these record types.
_CODEX_RECORD_TYPES = frozenset({"session_meta", "response_item", "turn_context", "event_msg"})


def estimate_tokens(raw: bytes | str) -> int:
    """``len(bytes) // 4`` — the same cheap proxy the capture path uses.

    Applied to a raw JSONL record this measures *file* progress, not ingest:
    a transcript line may be a tool call, a thinking block or a meta record
    that carries nothing ingestable. Use :func:`ingested_tokens` for the
    token total the status document publishes.
    """
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    return len(raw) // 4


def ingested_tokens(capture: ExchangeCapture) -> int:
    """Tokens of the text one capture actually hands to the sink.

    The single definition of ``totals.tokens_ingested`` for both lanes: the
    live proxy path and bulk jobs call this on the same
    :class:`ExchangeCapture` objects, so a transcript loaded in bulk and the
    same conversation arriving live contribute the same number. Counting the
    raw file instead would include tool calls, thinking blocks and meta
    records that are never ingested.
    """
    return sum(len(record[1]) for record in capture.ingest_records()) // 4


# -- record parsing ----------------------------------------------------------

def _text_blocks(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            if item.get("type") in (None, "text", "input_text", "output_text"):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "\n".join(part for part in parts if part).strip()


def exchange_text(obj: Any) -> tuple[str, str] | None:
    """``(role, text)`` for a Claude Code or Codex transcript record, else None.

    Tool calls, tool results, thinking, and meta records carry no ingestable
    prose and yield nothing; the caller still counts their bytes as progress.
    """
    if not isinstance(obj, dict):
        return None
    kind = obj.get("type")
    if kind in ("user", "assistant"):
        if obj.get("isMeta"):
            return None
        message = obj.get("message")
        if not isinstance(message, dict):
            return None
        role = normalize_role(message.get("role")) or kind
        text = _text_blocks(message.get("content"))
    elif kind == "response_item":
        payload = obj.get("payload")
        if not isinstance(payload, dict) or payload.get("type") != "message":
            return None
        role = normalize_role(payload.get("role"))
        text = _text_blocks(payload.get("content"))
    elif kind == "message" and "role" in obj:
        role = normalize_role(obj.get("role"))
        text = _text_blocks(obj.get("content"))
    else:
        return None
    if role not in ("user", "assistant") or not text:
        return None
    return role, text


def sniff_kind(path: str | os.PathLike[str]) -> str:
    """``"codex"`` or ``"claude"`` from the first records of a transcript."""
    try:
        with open(path, "rb") as fh:
            for _ in range(8):
                line = fh.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict) and obj.get("type") in _CODEX_RECORD_TYPES:
                    return "codex"
                return "claude"
    except OSError:
        pass
    return "claude"


# -- jobs --------------------------------------------------------------------

@dataclass
class BulkJob:
    id: str
    session: str
    path: str
    provider: str
    kind: str = "transcript"
    state: str = "queued"
    bytes_total: int = 0
    records_total: int = 0
    records_done: int = 0
    bytes_done: int = 0
    #: File progress, in ``estimate_tokens`` over every raw JSONL record.
    #: Paired with ``bytes_done``/``bytes_total`` and what the rate window and
    #: the ETA are computed from. Not the ingest total.
    tokens_done: int = 0
    #: What the emitted captures actually carry, by ``ingested_tokens`` -- the
    #: same unit the live proxy path reports. This is what reaches
    #: ``totals.tokens_ingested`` in the status document.
    tokens_ingested: int = 0
    captures_emitted: int = 0
    #: Jobs ahead of this one (a running job counts) plus one, at submit time.
    position_at_submit: int = 1
    error: str | None = None
    queued_at: str = field(default_factory=now_iso)
    started_at: str | None = None
    finished_at: str | None = None

    @property
    def active(self) -> bool:
        return self.state in ("indexing", "running")

    @property
    def finished(self) -> bool:
        return self.state in ("done", "cancelled", "failed")

    def queue_doc(self, position: int) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "session": self.session,
            "position": position,
            "bytes": self.bytes_total,
            "state": self.state,
        }

    def current_doc(self, eta_s: int | None) -> dict[str, Any]:
        return {
            "job": self.id,
            "kind": self.kind,
            "session": self.session,
            "path": self.path,
            "state": self.state,
            "records_done": self.records_done,
            "records_total": self.records_total,
            "bytes_done": self.bytes_done,
            "bytes_total": self.bytes_total,
            "tokens_done": self.tokens_done,
            "tokens_ingested": self.tokens_ingested,
            "captures_emitted": self.captures_emitted,
            "started_at": self.started_at,
            "eta_s": eta_s,
        }

    def summary(self) -> dict[str, Any]:
        doc = self.current_doc(None)
        doc.update(error=self.error, finished_at=self.finished_at, queued_at=self.queued_at)
        return doc


class RateWindow:
    """Ring buffer of ``(t, tokens, records)``; rates over the last ``window_s``."""

    def __init__(self, window_s: float = DEFAULT_WINDOW_S) -> None:
        self.window_s = float(window_s)
        self._samples: deque[tuple[float, int, int]] = deque()

    def clear(self) -> None:
        self._samples.clear()

    def add(self, now: float, tokens: int, records: int) -> None:
        self._samples.append((now, tokens, records))
        self._prune(now)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def rates(self, now: float | None = None) -> dict[str, Any]:
        now = time.monotonic() if now is None else now
        self._prune(now)
        if not self._samples:
            return {
                "tokens_per_s_recent": 0.0,
                "records_per_s_recent": 0.0,
                "avg_s_per_1k_tokens": None,
                "window_s": self.window_s,
            }
        span = max(now - self._samples[0][0], 1.0)
        tokens = sum(sample[1] for sample in self._samples)
        records = sum(sample[2] for sample in self._samples)
        tokens_per_s = tokens / span
        return {
            "tokens_per_s_recent": round(tokens_per_s, 1),
            "records_per_s_recent": round(records / span, 2),
            "avg_s_per_1k_tokens": round(1000.0 / tokens_per_s, 3) if tokens_per_s > 0 else None,
            "window_s": self.window_s,
        }

    def eta_s(self, remaining_tokens: int, now: float | None = None) -> int | None:
        rate = self.rates(now)["tokens_per_s_recent"]
        if rate <= 0:
            return None
        return int(max(0, remaining_tokens) / rate)


class _Cancelled(Exception):
    pass


def resolve_transcript(ident: str) -> tuple[str, str, str]:
    """``(path, session_id, kind)`` for a path or a session id prefix.

    A path is used as given (never written). Anything else goes through
    ``live_sessions.find_session``, which scans the Claude Code and Codex
    session roots.
    """
    from memory_condense.live_sessions.discovery import SessionKind, find_session, session_id_of

    if os.path.isfile(ident):
        # An explicit file needs no discovery (which would scan process
        # handle tables); its kind is sniffed from the first records.
        path = os.path.abspath(ident)
        kind = sniff_kind(path)
        return path, session_id_of(path, SessionKind(kind)), kind
    ref = find_session(ident, use_handles=False)
    if ref is None:
        raise FileNotFoundError(f"no transcript file or unique session id matches {ident!r}")
    return ref.path, ref.session_id, ref.kind.value


class BulkLoader:
    """FIFO bulk transcript jobs over the proxy's capture queue.

    ``captures`` is the proxy's :class:`CaptureQueue`; ``capacity`` its
    ``maxsize`` (the queue does not expose it). Captures are offered on the
    proxy's event loop (``bind_loop``) only when the queue has room, so bulk
    work throttles to the ingest rate instead of tripping admission loss.
    ``listener(kind)`` is called with ``"state"`` on job transitions and
    ``"progress"`` (rate-limited) while a job runs.
    """

    def __init__(
        self,
        captures: CaptureQueue,
        *,
        capacity: int,
        window_s: float = DEFAULT_WINDOW_S,
        listener: Callable[[str], None] | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._captures = captures
        self._capacity = int(capacity)
        self._loop = loop
        self._listener = listener
        self._lock = threading.RLock()
        self._jobs: queue.Queue[BulkJob | None] = queue.Queue()
        self._queued: list[BulkJob] = []
        self._current: BulkJob | None = None
        self._history: deque[BulkJob] = deque(maxlen=16)
        self._cancel = threading.Event()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._window = RateWindow(window_s)
        self._last_progress_notify = 0.0
        self._ids: dict[str, int] = {}
        # Lifetime totals across finished jobs plus the running one.
        self._tokens_finished = 0
        self._records_finished = 0
        self._sessions_finished = 0

    # -- wiring ------------------------------------------------------------
    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def set_listener(self, listener: Callable[[str], None] | None) -> None:
        self._listener = listener

    def _notify(self, kind: str) -> None:
        listener = self._listener
        if listener is None:
            return
        try:
            listener(kind)
        except Exception:
            logger.exception("bulk loader listener failed")

    def _notify_progress(self, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self._last_progress_notify >= _PROGRESS_NOTIFY_S:
            self._last_progress_notify = now
            self._notify("progress")

    # -- job control -------------------------------------------------------
    def submit(self, ident: str) -> BulkJob:
        """Resolve ``ident`` and enqueue a job; raises ``FileNotFoundError``."""
        path, session_id, kind = resolve_transcript(ident)
        provider = OPENAI if kind == "codex" else ANTHROPIC
        with self._lock:
            base = f"bulk-{session_id[:8]}"
            n = self._ids.get(base, 0)
            self._ids[base] = n + 1
            job_id = base if n == 0 else f"{base}-{n}"
            job = BulkJob(id=job_id, session=session_id, path=path, provider=provider)
            try:
                job.bytes_total = os.path.getsize(path)
            except OSError:
                job.bytes_total = 0
            current = self._current
            job.position_at_submit = len(self._queued) + 1 + int(
                current is not None and current.active
            )
            self._queued.append(job)
            self._jobs.put(job)
            self._ensure_worker()
        self._notify("state")
        return job

    def cancel(self, job_id: str) -> bool:
        # The listener is notified after this lock is released, never under
        # it: the publisher's listener publishes, and publishing reads back
        # through `snapshot()`. Calling out while holding the job lock would
        # invert the lock order against a concurrent publish.
        notify = False
        try:
            with self._lock:
                for job in self._queued:
                    if job.id == job_id:
                        job.state = "cancelled"
                        job.finished_at = now_iso()
                        self._queued.remove(job)
                        self._history.append(job)
                        notify = True
                        return True
                current = self._current
                if current is not None and current.id == job_id:
                    self._cancel.set()
                    return True
            return False
        finally:
            if notify:
                self._notify("state")

    def start(self) -> None:
        with self._lock:
            self._ensure_worker()

    def _ensure_worker(self) -> None:
        if self._stopping.is_set():
            raise RuntimeError("bulk loader is stopping")
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name="memory-bulk-loader", daemon=True)
            self._thread.start()

    def request_stop(self) -> None:
        """Cancel the running job and drop queued ones; never blocks.

        Safe to call from the event loop before the capture queue closes, so
        the worker stops offering before admission is cancelled under it.
        """
        with self._lock:
            self._stopping.set()
            self._cancel.set()
            for job in self._queued:
                job.state = "cancelled"
                job.finished_at = now_iso()
                self._history.append(job)
            self._queued.clear()
        self._jobs.put(None)

    def join(self, timeout: float = 10.0) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout)

    def stop(self, timeout: float = 10.0) -> None:
        """Cancel the running job, drop queued ones, and join the worker."""
        self.request_stop()
        self.join(timeout)

    # -- observation -------------------------------------------------------
    @property
    def current(self) -> BulkJob | None:
        return self._current

    @property
    def active(self) -> bool:
        current = self._current
        return current is not None and current.active

    def jobs(self) -> list[BulkJob]:
        with self._lock:
            return list(self._queued)

    def history(self) -> list[BulkJob]:
        with self._lock:
            return list(self._history)

    def job(self, job_id: str) -> BulkJob | None:
        with self._lock:
            for job in [self._current, *self._queued, *self._history]:
                if job is not None and job.id == job_id:
                    return job
        return None

    def snapshot(self) -> dict[str, Any]:
        """The ``current``/``rates``/``queue.jobs``/totals slice of the status doc."""
        with self._lock:
            current = self._current
            queued = list(self._queued)
            now = time.monotonic()
            rates = self._window.rates(now)
            if current is not None and current.active:
                # The ETA is file progress against a file-progress rate: both
                # sides are bytes//4, which is why the window is fed
                # `estimate_tokens(record.raw)` and not the ingest total.
                remaining = max(0, current.bytes_total - current.bytes_done) // 4
                current_doc = current.current_doc(self._window.eta_s(remaining, now))
                running_tokens = current.tokens_ingested
                running_records = current.records_done
            else:
                current_doc = None
                running_tokens = running_records = 0
            return {
                "state": "ingesting" if current_doc is not None else "idle",
                "current": current_doc,
                "rates": rates,
                "jobs": [job.queue_doc(i + 1) for i, job in enumerate(queued)],
                "totals": {
                    # Ingested text, in the live path's unit; see
                    # `ingested_tokens`.
                    "tokens_ingested": self._tokens_finished + running_tokens,
                    "records_ingested": self._records_finished + running_records,
                    "sessions": self._sessions_finished,
                },
                "last_job": self._history[-1].summary() if self._history else None,
            }

    # -- worker ------------------------------------------------------------
    def _run(self) -> None:
        while not self._stopping.is_set():
            job = self._jobs.get()
            if job is None:
                return
            with self._lock:
                if job not in self._queued:  # cancelled while queued
                    continue
                self._queued.remove(job)
                self._current = job
                self._cancel.clear()
                self._window.clear()
            try:
                self._run_job(job)
            except _Cancelled:
                job.state = "cancelled"
            except BaseException as exc:
                # An offer cancelled by queue shutdown surfaces as
                # CancelledError (a BaseException); that is a cancellation,
                # not a failure of the transcript.
                if self._cancel.is_set() or self._stopping.is_set():
                    job.state = "cancelled"
                else:
                    job.state = "failed"
                    job.error = f"{type(exc).__name__}: {exc}"
                    logger.exception("bulk job %s failed", job.id)
            else:
                if job.state != "cancelled":
                    job.state = "done"
            finally:
                job.finished_at = now_iso()
                with self._lock:
                    self._tokens_finished += job.tokens_ingested
                    self._records_finished += job.records_done
                    if job.state == "done":
                        self._sessions_finished += 1
                    self._history.append(job)
                    self._current = None
                self._notify("state")

    def _check_cancel(self) -> None:
        if self._cancel.is_set() or self._stopping.is_set():
            raise _Cancelled()

    def _run_job(self, job: BulkJob) -> None:
        from memory_condense.live_sessions import MappedSession

        job.state = "indexing"
        job.started_at = now_iso()
        self._notify("state")
        with MappedSession(job.path) as mapped:
            total = 0
            for _ in mapped.records():
                total += 1
                if total % 10_000 == 0:
                    self._check_cancel()
            job.records_total = total
            job.bytes_total = mapped.last_offset() or mapped.size
            job.state = "running"
            self._notify("state")

            conversation_id = f"transcript:{job.session}"
            pending_user: list[str] = []
            pending_user_offset: int | None = None
            pending_assistant: list[str] = []
            ordinal = 0

            def flush() -> None:
                nonlocal pending_user, pending_user_offset, pending_assistant, ordinal
                if not pending_user and not pending_assistant:
                    return
                user_text = "\n\n".join(pending_user)
                reply_text = "\n\n".join(pending_assistant)
                sha = hashlib.sha256(
                    f"{job.session}:{pending_user_offset}:".encode("utf-8")
                    + user_text.encode("utf-8")
                ).hexdigest()
                prompt: tuple[TranscriptMessage, ...] = ()
                if user_text:
                    prompt = (
                        TranscriptMessage(
                            role="user",
                            text=user_text,
                            conversation_id=conversation_id,
                            message_id=f"{conversation_id}:{pending_user_offset}",
                            ordinal=ordinal,
                        ),
                    )
                capture = ExchangeCapture(
                    provider=job.provider,
                    conversation_id=conversation_id,
                    model=None,
                    prompt=prompt,
                    reply_text=reply_text,
                    streamed=False,
                    request_sha256=sha,
                    prompt_tokens_estimate=estimate_tokens(user_text),
                )
                ordinal += 1
                pending_user, pending_user_offset, pending_assistant = [], None, []
                self._offer(capture)
                job.captures_emitted += 1
                # Counted on admission, from the capture itself: the same
                # number the live proxy path records for the same exchange.
                job.tokens_ingested += ingested_tokens(capture)

            for record in self._bounded_records(mapped, total):
                self._check_cancel()
                parsed = None
                try:
                    parsed = exchange_text(record.json())
                except ValueError:
                    parsed = None
                if parsed is not None:
                    role, text = parsed
                    if role == "user":
                        if pending_assistant:
                            flush()
                        if pending_user_offset is None:
                            pending_user_offset = record.offset
                        pending_user.append(text)
                    else:
                        pending_assistant.append(text)
                tokens = estimate_tokens(record.raw)
                job.records_done += 1
                job.bytes_done = min(job.bytes_total, record.offset + len(record.raw) + 1)
                job.tokens_done += tokens
                self._window.add(time.monotonic(), tokens, 1)
                self._notify_progress()
            flush()
            job.bytes_done = job.bytes_total
            self._notify_progress(force=True)

    @staticmethod
    def _bounded_records(mapped: Any, total: int) -> Iterator[Any]:
        """The first ``total`` records: a live file may grow during the pass."""
        seen = 0
        for record in mapped.records():
            if seen >= total:
                return
            seen += 1
            yield record

    def _has_room(self, capture: ExchangeCapture) -> bool:
        captures = self._captures
        if captures.state != "running":
            return False
        if captures.waiting_producers:
            return False
        if captures.depth >= self._capacity:
            return False
        return captures.retained_bytes + capture.retained_bytes <= captures.max_retained_bytes

    def _offer(self, capture: ExchangeCapture) -> None:
        """Hand one capture to the queue; wait for room; never drop silently."""
        loop = self._loop
        if loop is None:
            raise RuntimeError("bulk loader has no event loop bound")
        while True:
            self._check_cancel()
            if self._captures.state in ("closing", "closed", "failed"):
                raise CaptureQueueClosedError(
                    f"capture queue is not accepting work ({self._captures.state})"
                )
            if not self._has_room(capture):
                time.sleep(_ROOM_POLL_S)
                continue
            future = asyncio.run_coroutine_threadsafe(self._captures.offer_async(capture), loop)
            try:
                future.result()
                return
            except CaptureQueueSaturatedError:
                # Lost a race with live traffic; the queue counted it. Wait
                # for room and try the same item again.
                time.sleep(_ROOM_POLL_S)


__all__ = [
    "BulkJob",
    "BulkLoader",
    "DEFAULT_WINDOW_S",
    "JOB_STATES",
    "RateWindow",
    "estimate_tokens",
    "exchange_text",
    "ingested_tokens",
    "resolve_transcript",
    "sniff_kind",
]
