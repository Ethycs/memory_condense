"""The single writer of ``status.json``.

:class:`StatusPublisher` assembles the whole document from the parts that
own the numbers — the :class:`CaptureQueue` counters, the proxy's ``/health``
reasoning (passed in as callables so the thresholds live in one place), the
memory service in augment mode, the :class:`BulkLoader`, per-request serving
receipts, and the :class:`SnapshotStore` — and writes it atomically on every
state change and on a one-second ticker while a job runs. Nothing else
computes these numbers; every surface renders this document.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Mapping

from memory_condense.control.status import empty_status, status_path_for, write_status

logger = logging.getLogger(__name__)

HealthFn = Callable[[], tuple[str, list[str]]]
MemoryHealthFn = Callable[[Mapping[str, Any]], tuple[str, list[str]]]

_MAX_TRACKED_CONVERSATIONS = 10_000

#: Stages a native session passes through before its memory can answer. Any
#: other stage value is treated as pre-ready too: an unknown stage is startup
#: work we cannot name, which is still not an idle service.
_READY_STAGES = frozenset({"ready", None})


def _native_progress(memory: Mapping[str, Any] | None) -> tuple[bool, bool]:
    """``(starting, busy)`` for the native memory service.

    Read from ``MemoryProxy.status()``. ``starting`` means at least one session
    has not finished opening its models; ``busy`` means durable work is
    outstanding. Either one makes ``idle`` a false report.
    """
    if not isinstance(memory, Mapping) or "error" in memory:
        return False, False
    startup = memory.get("startup")
    stages = (
        [record.get("stage") for record in startup.values() if isinstance(record, Mapping)]
        if isinstance(startup, Mapping)
        else []
    )
    starting = bool(memory.get("opening_sessions")) or any(stage not in _READY_STAGES for stage in stages)
    return starting, bool(memory.get("busy"))


class StatusPublisher:
    """Assemble and atomically publish the status document.

    ``health`` returns ``(status, reasons)`` for observe mode from the queue;
    ``memory_health`` maps ``memory.status()`` to the same pair in augment
    mode. Both are the proxy's own ``/health`` logic, shared not copied.
    Thread-safe: request handlers, the bulk worker, and the ticker all call
    in. Idle cost is one lock and one sleep per second.
    """

    def __init__(
        self,
        *,
        mode: str,
        captures: Any,
        capacity: int,
        health: HealthFn,
        data_dir: str | os.PathLike[str] | None = None,
        memory: Any | None = None,
        memory_health: MemoryHealthFn | None = None,
        bulk: Any | None = None,
        snapshots: Any | None = None,
        store_path: Callable[[], Path | None] | None = None,
        tick_s: float = 1.0,
        pid: int | None = None,
    ) -> None:
        self.mode = mode
        self._captures = captures
        self._capacity = int(capacity)
        self._health = health
        self._memory = memory
        self._memory_health = memory_health
        self._bulk = bulk
        self._snapshots = snapshots
        self._store_path = store_path
        self._tick_s = float(tick_s)
        self._pid = pid
        self.path: Path | None = status_path_for(data_dir) if data_dir is not None else None

        # Two locks, with a fixed order: `_publish_lock` is always taken
        # before `_lock`, never the reverse. `_lock` guards the small mutable
        # counters the request path touches; `_publish_lock` serializes a
        # whole publish (assembly *and* write) so documents cannot go
        # backwards. See `publish` for why they are not the same lock.
        self._publish_lock = threading.RLock()
        self._lock = threading.RLock()
        self._state_override: str | None = None
        self._error: str | None = None
        self._dirty = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_document: dict[str, Any] | None = None
        self.writes = 0
        self.write_failures = 0

        self._serving = {
            "requests": 0,
            "rewritten": 0,
            "static_tokens_last": None,
            "served_tokens_last": None,
            "saved_tokens_total": 0,
            "levels": {"local": 0, "handoff": 0, "passthrough": 0},
            "passthrough_forced": 0,
            "passthrough_degraded": 0,
            "index_deferred": 0,
        }
        self._live_tokens = 0
        self._conversations: set[str] = set()
        self._conversations_overflow = 0

        if bulk is not None:
            bulk.set_listener(self._on_bulk)

    # -- receipts ----------------------------------------------------------
    def record_serving(self, receipt: Any) -> None:
        """One inference request, described by a serving receipt.

        ``receipt`` is an ``interfaces.proxy_receipts.ServingReceipt``, taken
        duck-typed so the control plane keeps no dependency on the proxy: it
        needs ``effective_static_tokens``, ``effective_served_tokens``,
        ``saved_tokens`` and ``rewritten``. The receipt has decided every one
        -- which estimator produced them, whether the upstream's exact
        ``usage`` beats the estimate, and whether a passthrough counts as a
        rewrite. Nothing is recomputed here; this only accumulates.
        """
        with self._lock:
            serving = self._serving
            serving["requests"] += 1
            serving["rewritten"] += int(bool(receipt.rewritten))
            serving["static_tokens_last"] = receipt.effective_static_tokens
            serving["served_tokens_last"] = receipt.effective_served_tokens
            serving["saved_tokens_total"] += int(receipt.saved_tokens)
            # One counter per service level; observe mode has no level. A
            # forced and a degraded passthrough mean opposite things, so they
            # are never added together.
            level = getattr(receipt, "level", None)
            if level in serving["levels"]:
                serving["levels"][level] += 1
                if level == "passthrough":
                    key = "passthrough_degraded" if getattr(receipt, "degraded", False) else "passthrough_forced"
                    serving[key] += 1
            serving["index_deferred"] += int(bool(getattr(receipt, "index_deferred", False)))
            self._dirty = True

    def record_capture(self, tokens: int, conversation_id: str | None = None) -> None:
        """One live capture admitted to the queue (tokens by the capture path's estimate)."""
        with self._lock:
            self._live_tokens += max(0, int(tokens))
            if conversation_id is not None and conversation_id not in self._conversations:
                if len(self._conversations) < _MAX_TRACKED_CONVERSATIONS:
                    self._conversations.add(conversation_id)
                else:
                    self._conversations_overflow += 1
            self._dirty = True

    def set_state(self, state: str | None, *, error: str | None = None) -> dict[str, Any]:
        """Force ``swapping``/``error`` (or clear with None) and publish."""
        with self._lock:
            self._state_override = state
            self._error = error if state == "error" else None
        return self.publish("state")

    @property
    def state_override(self) -> str | None:
        return self._state_override

    # -- document ----------------------------------------------------------
    def _store_bytes(self, path: Path | None) -> int:
        if path is None:
            return 0
        total = 0
        for candidate in (path, Path(f"{path}-wal")):
            try:
                total += candidate.stat().st_size
            except OSError:
                pass
        return total

    def document(self) -> dict[str, Any]:
        """Assemble a complete document without writing it."""
        captures = self._captures
        doc = empty_status(mode=self.mode, pid=self._pid)

        memory_snapshot: dict[str, Any] | None = None
        if self._memory is not None:
            try:
                memory_snapshot = dict(self._memory.status())
            except Exception as exc:
                logger.debug("memory status unavailable: %s", exc)
                memory_snapshot = {"error": f"{type(exc).__name__}: {exc}"}

        try:
            if memory_snapshot is not None and self._memory_health is not None and "error" not in memory_snapshot:
                status, reasons = self._memory_health(memory_snapshot)
            else:
                status, reasons = self._health()
        except Exception as exc:
            status, reasons = "degraded", [f"health_unavailable:{type(exc).__name__}"]
        doc["health"] = {"status": status, "reasons": list(reasons)}

        queue = doc["queue"]
        queue["capacity"] = self._capacity
        queue["in_flight"] = int(getattr(captures, "in_flight", 0))
        queue["depth"] = int(getattr(captures, "depth", 0))
        queue["retained_bytes"] = int(getattr(captures, "retained_bytes", 0))
        queue["rejected_saturated"] = int(getattr(captures, "rejected_saturated", 0))
        queue["terminal_failures"] = int(getattr(captures, "capture_terminal_failures", 0))
        queue["state"] = getattr(captures, "state", None)

        bulk = self._bulk.snapshot() if self._bulk is not None else None
        if bulk is not None:
            queue["jobs"] = bulk["jobs"]
            doc["current"] = bulk["current"]
            doc["rates"] = bulk["rates"]
            doc["last_job"] = bulk["last_job"]
        else:
            doc["last_job"] = None

        with self._lock:
            serving = dict(self._serving, levels=dict(self._serving["levels"]))
            live_tokens = self._live_tokens
            live_sessions = len(self._conversations) + self._conversations_overflow
            override = self._state_override
            error = self._error
        doc["serving"] = serving

        totals = doc["totals"]
        totals["records_ingested"] = int(getattr(captures, "ingested", 0))
        # One unit on both lanes: tokens of the text actually handed to the
        # sink, by `bulk.ingested_tokens`. The bulk job's `tokens_done` is a
        # *file* progress figure (every JSONL record, tool calls and meta
        # included) and drives the rate window and the ETA; adding it here
        # would inflate the headline against the same content arriving live.
        totals["tokens_ingested"] = live_tokens + (bulk["totals"]["tokens_ingested"] if bulk else 0)
        totals["sessions"] = live_sessions + (bulk["totals"]["sessions"] if bulk else 0)

        store_path = None
        if self._store_path is not None:
            try:
                store_path = self._store_path()
            except Exception:
                store_path = None
        if store_path is None and self._snapshots is not None:
            store_path = self._snapshots.db_path
        totals["store_bytes"] = self._store_bytes(store_path)
        store = doc["store"]
        store["path"] = str(store_path) if store_path is not None else None
        if self._snapshots is not None:
            store["snapshot_dir"] = str(self._snapshots.snapshot_dir)
            try:
                records = self._snapshots.list()
            except OSError:
                records = []
            store["snapshots"] = [record["file"] for record in records]
            store["last_snapshot"] = records[-1]["file"] if records else None

        native_starting, native_busy = _native_progress(memory_snapshot)
        if override is not None:
            doc["state"] = override
        elif status == "failed":
            doc["state"] = "error"
        elif bulk is not None and bulk["state"] == "ingesting":
            doc["state"] = "ingesting"
        elif queue["in_flight"] or queue["depth"]:
            doc["state"] = "ingesting"
        elif native_starting:
            # The capture queue is an observe-mode structure; in augment mode it
            # is empty while the native session is still loading models. Without
            # this branch a cold start publishes a confident, wrong "idle".
            doc["state"] = "starting"
        elif native_busy:
            doc["state"] = "ingesting"
        else:
            doc["state"] = "idle"
        doc["error"] = error
        doc["memory"] = memory_snapshot
        return doc

    def publish(self, reason: str = "change") -> dict[str, Any]:
        """Assemble and write (when a path is configured). Never raises.

        Assembly and the write happen together under ``_publish_lock``, so
        publishes are totally ordered and an older document can never land on
        top of a newer one -- which is what happened when ``document()`` was
        built outside the lock the writer then took: a request handler and the
        one-second ticker could interleave and leave the file stale until the
        next tick.

        That lock is deliberately not ``_lock``. ``document()`` stats files,
        lists the snapshot directory, and calls ``memory.status()``; holding
        the counter lock across all of that would make ``record_serving`` and
        ``record_capture`` -- which run on the inference path -- queue behind
        slow status work, and ``runtime/sessions.py`` is explicit that status
        reporting must never do that. Nothing on the inference path publishes:
        the ticker, the bulk worker, the control routes and a swap do, and
        serializing those costs the proxy nothing.
        """
        with self._publish_lock:
            doc = self.document()
            with self._lock:
                self._dirty = False
                self.last_document = doc
            if self.path is not None:
                try:
                    write_status(self.path, doc)
                    self.writes += 1
                except OSError:
                    self.write_failures += 1
                    logger.warning("status write failed (%s)", reason, exc_info=True)
        return doc

    # -- ticker ------------------------------------------------------------
    def _on_bulk(self, kind: str) -> None:
        if kind == "state":
            self.publish("bulk")
        else:
            with self._lock:
                self._dirty = True

    def _job_running(self) -> bool:
        bulk = self._bulk
        return bool(bulk is not None and bulk.active)

    def _tick(self) -> None:
        while not self._stop.wait(self._tick_s):
            with self._lock:
                dirty = self._dirty
            if dirty or self._job_running():
                self.publish("tick")

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._tick, name="memory-status-publisher", daemon=True)
            self._thread.start()
        self.publish("start")

    def stop(self, *, final: bool = True) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(self._tick_s * 2 + 1.0)
        self._thread = None
        if final:
            self.publish("stop")


__all__ = ["StatusPublisher"]
