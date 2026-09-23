"""Serialized durable-capture scheduling for the provider proxy."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from memory_condense.interfaces.proxy_capture import ExchangeCapture


logger = logging.getLogger(__name__)

CaptureSink = Callable[[ExchangeCapture], bool | None]


@dataclass(frozen=True, slots=True)
class PendingWorkSnapshot:
    """Immutable durable-backlog projection sampled by the worker thread."""

    sampled_at_monotonic: float
    ingest_manifest_count: int
    ingest_chunk_count: int
    ingest_token_count: int
    ingest_oldest_age_seconds: float | None
    enrichment_turn_count: int
    enrichment_ready_count: int
    enrichment_oldest_age_seconds: float | None
    ingest_failed_count: int = 0
    ingest_oldest_error_kind: str | None = None
    enrichment_failed_count: int = 0
    enrichment_oldest_error_kind: str | None = None
    enrichment_deferred_correction_count: int = 0
    enrichment_discarded_legacy_count: int = 0

    @property
    def has_pending(self) -> bool:
        return bool(self.ingest_manifest_count or self.enrichment_turn_count)

    def age_at(self, now: float) -> float:
        return max(0.0, now - self.sampled_at_monotonic)

    def ingest_oldest_age_at(self, now: float) -> float | None:
        if self.ingest_oldest_age_seconds is None:
            return None
        return self.ingest_oldest_age_seconds + self.age_at(now)

    def enrichment_oldest_age_at(self, now: float) -> float | None:
        if self.enrichment_oldest_age_seconds is None:
            return None
        return self.enrichment_oldest_age_seconds + self.age_at(now)


@dataclass(frozen=True, slots=True)
class CompletionOutcome:
    """One bounded T1/T2 maintenance tick and its resulting backlog."""

    indexed_manifests: int = 0
    enriched_turns: int = 0
    t1_failed: bool = False
    t2_failed: bool = False
    snapshot_failed: bool = False
    snapshot: PendingWorkSnapshot | None = None

    @property
    def made_progress(self) -> bool:
        return bool(self.indexed_manifests or self.enriched_turns)

    @property
    def failed(self) -> bool:
        return self.t1_failed or self.t2_failed or self.snapshot_failed

    @property
    def should_continue(self) -> bool:
        snapshot = self.snapshot
        if snapshot is None or self.snapshot_failed or self.t1_failed:
            return False
        if snapshot.ingest_manifest_count:
            # T1 progress has priority. A T2 failure must never strand the
            # remaining base-search backlog.
            return self.indexed_manifests > 0
        if self.t2_failed:
            return False
        return bool(snapshot.enrichment_turn_count and self.enriched_turns)


CompletionCallback = Callable[[], CompletionOutcome | None]


class CaptureQueueRejectedError(RuntimeError):
    """A capture could not enter the bounded in-memory handoff."""


class CaptureQueueSaturatedError(CaptureQueueRejectedError):
    """Queue admission exceeded its waiter or time bound."""


class CaptureQueueClosedError(CaptureQueueRejectedError):
    """Queue admission was attempted after shutdown began."""


class CaptureQueueShutdownTimeoutError(TimeoutError):
    """Graceful capture drain exceeded its configured shutdown bound."""


class CaptureSinkNegativeAcknowledgementError(RuntimeError):
    """A sink explicitly reported that it did not capture the exchange."""


@dataclass(frozen=True, slots=True)
class _QueuedCapture:
    capture: ExchangeCapture
    enqueued_at: float
    retained_bytes: int


_STOP = object()


class CaptureQueue:
    """Bounded, serialized capture and durable-completion scheduler.

    Count and retained-payload-byte limits cover waiting producers, queued
    captures, in-flight batches, and sink retries. Sink failures retry the
    same FIFO item with capped exponential backoff, then enter an explicit
    terminal-failure count so one poison item cannot wedge later work.
    """

    def __init__(
        self,
        sink: CaptureSink,
        *,
        maxsize: int = 256,
        max_waiters: int | None = None,
        max_retained_bytes: int = 16 * 1024 * 1024,
        compact_captures: bool = False,
        offer_timeout_seconds: float = 1.0,
        shutdown_timeout_seconds: float = 30.0,
        batch_size: int = 16,
        maintenance_batch_quota: int = 4,
        maintenance_poll_seconds: float = 5.0,
        maintenance_retry_initial_seconds: float = 0.05,
        maintenance_retry_max_seconds: float = 5.0,
        sink_retry_initial_seconds: float = 0.05,
        sink_retry_max_seconds: float = 5.0,
        sink_max_retries: int = 3,
        completion: CompletionCallback | None = None,
    ) -> None:
        for name, value in (
            ("maxsize", maxsize),
            ("max_retained_bytes", max_retained_bytes),
            ("batch_size", batch_size),
            ("maintenance_batch_quota", maintenance_batch_quota),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max_waiters is not None and (
            type(max_waiters) is not int or max_waiters < 0
        ):
            raise ValueError("max_waiters must be non-negative")
        if type(sink_max_retries) is not int or sink_max_retries < 0:
            raise ValueError("sink_max_retries must be non-negative")
        if type(compact_captures) is not bool:
            raise ValueError("compact_captures must be a boolean")
        for name, value in (
            ("offer_timeout_seconds", offer_timeout_seconds),
            ("shutdown_timeout_seconds", shutdown_timeout_seconds),
            (
                "maintenance_poll_seconds",
                maintenance_poll_seconds,
            ),
            (
                "maintenance_retry_initial_seconds",
                maintenance_retry_initial_seconds,
            ),
            ("maintenance_retry_max_seconds", maintenance_retry_max_seconds),
            ("sink_retry_initial_seconds", sink_retry_initial_seconds),
            ("sink_retry_max_seconds", sink_retry_max_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number")
        if maintenance_retry_max_seconds < maintenance_retry_initial_seconds:
            raise ValueError(
                "maintenance_retry_max_seconds must be at least the initial delay"
            )
        if sink_retry_max_seconds < sink_retry_initial_seconds:
            raise ValueError(
                "sink_retry_max_seconds must be at least the initial delay"
            )

        self._sink = sink
        self._maxsize = maxsize
        self._queue: asyncio.Queue[_QueuedCapture | object] = asyncio.Queue(
            maxsize=maxsize
        )
        self._max_waiters = maxsize if max_waiters is None else max_waiters
        self._max_retained_bytes = max_retained_bytes
        self._compact_captures = compact_captures
        self._offer_timeout_seconds = float(offer_timeout_seconds)
        self._shutdown_timeout_seconds = float(shutdown_timeout_seconds)
        self._batch_size = batch_size
        self._maintenance_batch_quota = maintenance_batch_quota
        self._maintenance_poll_seconds = float(maintenance_poll_seconds)
        self._maintenance_retry_initial_seconds = float(
            maintenance_retry_initial_seconds
        )
        self._maintenance_retry_max_seconds = float(maintenance_retry_max_seconds)
        self._sink_retry_initial_seconds = float(sink_retry_initial_seconds)
        self._sink_retry_max_seconds = float(sink_retry_max_seconds)
        self._sink_max_retries = sink_max_retries
        self._completion = completion

        self._task: asyncio.Task[None] | None = None
        self._stop_task: asyncio.Task[None] | None = None
        self._state = "new"
        self._admission_lock = asyncio.Lock()
        self._put_waiters: set[asyncio.Task[None]] = set()
        self._outstanding: dict[int, tuple[float, int]] = {}
        self._retained_bytes = 0
        self._in_flight_enqueued_at: tuple[float, ...] = ()
        self._pending_snapshot: PendingWorkSnapshot | None = None
        self._maintenance_poll_at: float | None = None
        self._maintenance_retry_at: float | None = None
        self._next_maintenance_retry_delay = (
            self._maintenance_retry_initial_seconds
        )
        self._active_sink_retry_delay = 0.0
        self._capture_batches_since_maintenance = 0

        self.accepted = 0
        self.backpressured = 0
        self.rejected = 0
        self.rejected_saturated = 0
        self.rejected_closed = 0
        self.rejected_retained_bytes = 0
        self.offer_timeouts = 0
        self.offer_cancellations = 0
        self.sink_attempts = 0
        self.sink_completed = 0
        self.durably_captured = 0
        self.capture_failures = 0
        self.capture_retries = 0
        self.capture_retry_recoveries = 0
        self.capture_terminal_failures = 0
        self.completion_failures = 0
        self.completion_idle_polls = 0
        self.completion_retries = 0
        self.completion_ticks = 0
        self.t1_indexed_manifests = 0
        self.t1_failures = 0
        self.t2_enriched_turns = 0
        self.t2_failures = 0
        self.completion_snapshot_failures = 0
        self.worker_batches = 0
        self.max_observed_batch_size = 0
        self.peak_retained_bytes = 0
        self.shutdown_timeouts = 0

    def start(self) -> None:
        if self._state == "running":
            if self._task is None or self._task.done():
                self._state = "failed"
                raise RuntimeError("capture queue worker stopped unexpectedly")
            return
        if self._state == "closed":
            # Starlette applications can enter sequential lifespan contexts,
            # sometimes on different event loops. A fresh asyncio.Queue avoids
            # retaining the first loop while cumulative metrics remain intact.
            self._queue = asyncio.Queue(maxsize=self._maxsize)
            self._admission_lock = asyncio.Lock()
            self._task = None
            self._stop_task = None
            self._pending_snapshot = None
            self._maintenance_poll_at = None
            self._maintenance_retry_at = None
            self._next_maintenance_retry_delay = (
                self._maintenance_retry_initial_seconds
            )
            self._capture_batches_since_maintenance = 0
            self._state = "new"
        if self._state != "new":
            raise RuntimeError("capture queue cannot restart after shutdown")
        self._state = "running"
        self._task = asyncio.create_task(self._drain())
        self._task.add_done_callback(self._worker_finished)

    def _worker_finished(self, task: asyncio.Task[None]) -> None:
        if task is not self._task or self._state != "running":
            return
        self._state = "failed"
        if task.cancelled():
            logger.error("capture queue worker was cancelled unexpectedly")
            return
        error = task.exception()
        if error is None:
            logger.error("capture queue worker stopped unexpectedly")
        else:
            logger.error(
                "capture queue worker failed unexpectedly",
                exc_info=(type(error), error, error.__traceback__),
            )

    async def stop(self) -> None:
        if self._state == "new":
            while not self._queue.empty():
                queued = self._queue.get_nowait()
                if isinstance(queued, _QueuedCapture):
                    self._release(queued)
                self._queue.task_done()
            self._state = "closed"
            return
        if self._state == "closed":
            return
        if self._stop_task is None:
            self._state = "closing"
            for waiter in tuple(self._put_waiters):
                waiter.cancel()
            self._stop_task = asyncio.create_task(self._finish_stop())
        await asyncio.shield(self._stop_task)

    async def _finish_stop(self) -> None:
        shutdown_runner = asyncio.create_task(self._graceful_shutdown())
        try:
            await asyncio.wait_for(
                asyncio.shield(shutdown_runner),
                timeout=self._shutdown_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            self.shutdown_timeouts += 1
            self._state = "failed"
            raise CaptureQueueShutdownTimeoutError(
                "capture queue did not drain before the shutdown timeout"
            ) from exc
        except BaseException:
            self._state = "failed"
            raise
        else:
            self._state = "closed"

    async def _graceful_shutdown(self) -> None:
        waiters = tuple(self._put_waiters)
        if waiters:
            await asyncio.gather(*waiters, return_exceptions=True)
        await self._queue.put(_STOP)
        task = self._task
        if task is not None:
            await asyncio.shield(task)

    def offer(self, capture: ExchangeCapture) -> bool:
        """Immediately offer a capture without awaiting or raising.

        This preserves the original fail-open API. It never waits behind a
        full queue or existing asynchronous producer; callers needing bounded
        backpressure use :meth:`offer_async`.
        """

        try:
            queued = self._reserve(capture, allow_new=True)
        except CaptureQueueRejectedError:
            return False
        if self._put_waiters:
            self._release(queued)
            self._record_saturation("capture queue has waiting producers")
            return False
        try:
            self._queue.put_nowait(queued)
        except asyncio.QueueFull:
            self._release(queued)
            self._record_saturation("capture queue is full")
            return False
        self.accepted += 1
        return True

    async def offer_async(self, capture: ExchangeCapture) -> None:
        """Offer with bounded backpressure, raising on rejected admission."""

        queued = self._reserve(capture)
        admitted = False
        try:
            must_wait = bool(self._put_waiters)
            if not must_wait:
                try:
                    self._queue.put_nowait(queued)
                    admitted = True
                except asyncio.QueueFull:
                    must_wait = True
            if must_wait:
                if len(self._put_waiters) >= self._max_waiters:
                    raise self._record_saturation(
                        "capture queue and producer waiter budget are full"
                    )
                self.backpressured += 1
                logger.warning("capture queue full; applying producer backpressure")
                waiter = asyncio.create_task(self._put_in_admission_order(queued))
                self._put_waiters.add(waiter)
                try:
                    await asyncio.wait_for(
                        waiter,
                        timeout=self._offer_timeout_seconds,
                    )
                    admitted = True
                except asyncio.TimeoutError as exc:
                    self.offer_timeouts += 1
                    raise self._record_saturation(
                        "capture queue admission timed out"
                    ) from exc
                except asyncio.CancelledError as exc:
                    if waiter.done() and not waiter.cancelled():
                        waiter.result()
                        admitted = True
                        self.accepted += 1
                        self.offer_cancellations += 1
                        raise
                    if self._state != "running":
                        self.offer_cancellations += 1
                        raise self._record_closed(
                            "capture queue closed during admission"
                        ) from exc
                    self.offer_cancellations += 1
                    raise
                finally:
                    self._put_waiters.discard(waiter)
        except BaseException:
            if not admitted:
                self._release(queued)
            raise
        self.accepted += 1

    async def _put_in_admission_order(self, queued: _QueuedCapture) -> None:
        """Serialize blocked producers so a later runnable task cannot barge."""
        async with self._admission_lock:
            await self._queue.put(queued)

    def _reserve(
        self, capture: ExchangeCapture, *, allow_new: bool = False
    ) -> _QueuedCapture:
        self._ensure_running(allow_new=allow_new)
        prepared = capture.compact_for_ingest() if self._compact_captures else capture
        retained_bytes = prepared.retained_bytes
        if self._retained_bytes + retained_bytes > self._max_retained_bytes:
            self.rejected += 1
            self.rejected_saturated += 1
            self.rejected_retained_bytes += 1
            raise CaptureQueueSaturatedError(
                "capture retained-byte budget is exhausted"
            )
        queued = _QueuedCapture(
            capture=prepared,
            enqueued_at=time.monotonic(),
            retained_bytes=retained_bytes,
        )
        self._outstanding[id(queued)] = (queued.enqueued_at, retained_bytes)
        self._retained_bytes += retained_bytes
        self.peak_retained_bytes = max(
            self.peak_retained_bytes,
            self._retained_bytes,
        )
        return queued

    def _ensure_running(self, *, allow_new: bool = False) -> None:
        if allow_new and self._state == "new":
            return
        if self._state == "running" and (
            self._task is None or self._task.done()
        ):
            self._state = "failed"
        if self._state != "running":
            raise self._record_closed(
                f"capture queue is not accepting work ({self._state})"
            )

    def _record_saturation(self, message: str) -> CaptureQueueSaturatedError:
        self.rejected += 1
        self.rejected_saturated += 1
        return CaptureQueueSaturatedError(message)

    def _record_closed(self, message: str) -> CaptureQueueClosedError:
        self.rejected += 1
        self.rejected_closed += 1
        return CaptureQueueClosedError(message)

    def _release(self, queued: _QueuedCapture) -> None:
        tracked = self._outstanding.pop(id(queued), None)
        if tracked is not None:
            self._retained_bytes -= tracked[1]

    @property
    def state(self) -> str:
        return self._state

    @property
    def waiting_producers(self) -> int:
        return len(self._put_waiters)

    @property
    def max_waiters(self) -> int:
        return self._max_waiters

    @property
    def max_retained_bytes(self) -> int:
        return self._max_retained_bytes

    @property
    def retained_bytes(self) -> int:
        return self._retained_bytes

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    @property
    def in_flight(self) -> int:
        return len(self._in_flight_enqueued_at)

    @property
    def worker_quiesced(self) -> bool:
        return self._task is None or self._task.done()

    @property
    def pending_snapshot(self) -> PendingWorkSnapshot | None:
        return self._pending_snapshot

    @property
    def maintenance_retry_delay_seconds(self) -> float:
        if self._maintenance_retry_at is None:
            return 0.0
        return max(0.0, self._maintenance_retry_at - time.monotonic())

    @property
    def maintenance_poll_delay_seconds(self) -> float:
        if self._maintenance_poll_at is None:
            return 0.0
        return max(0.0, self._maintenance_poll_at - time.monotonic())

    @property
    def sink_retry_delay_seconds(self) -> float:
        return self._active_sink_retry_delay

    @property
    def capture_batches_since_maintenance(self) -> int:
        return self._capture_batches_since_maintenance

    @property
    def ingested(self) -> int:
        return self.sink_completed

    @property
    def failed(self) -> int:
        return self.capture_failures + self.completion_failures

    @property
    def dropped(self) -> int:
        return self.rejected

    @property
    def oldest_age_seconds(self) -> float:
        if not self._outstanding:
            return 0.0
        oldest = min(enqueued_at for enqueued_at, _size in self._outstanding.values())
        return max(0.0, time.monotonic() - oldest)

    async def _drain(self) -> None:
        maintenance_needed = self._completion is not None
        while True:
            poll_due = bool(
                self._completion is not None
                and not maintenance_needed
                and self._maintenance_poll_at is not None
                and self.maintenance_poll_delay_seconds <= 0
            )
            if poll_due:
                maintenance_needed = True
                self._maintenance_poll_at = None
                self.completion_idle_polls += 1

            retry_delay = self.maintenance_retry_delay_seconds
            quota_due = (
                self._capture_batches_since_maintenance
                >= self._maintenance_batch_quota
            )
            if (
                maintenance_needed
                and retry_delay <= 0
                and (self._queue.empty() or quota_due)
            ):
                should_continue, retry = await self._run_completion_tick()
                self._capture_batches_since_maintenance = 0
                if retry:
                    self._maintenance_poll_at = None
                    self._schedule_maintenance_retry()
                    maintenance_needed = True
                else:
                    self._clear_maintenance_retry()
                    maintenance_needed = should_continue
                    if maintenance_needed:
                        self._maintenance_poll_at = None
                    else:
                        self._schedule_maintenance_poll()
                continue

            wait_delay: float | None = None
            if self._queue.empty():
                if maintenance_needed and retry_delay > 0:
                    wait_delay = retry_delay
                elif not maintenance_needed and self._maintenance_poll_at is not None:
                    wait_delay = self.maintenance_poll_delay_seconds
            if wait_delay is not None:
                try:
                    first = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=wait_delay,
                    )
                except asyncio.TimeoutError:
                    continue
            else:
                first = await self._queue.get()

            if first is _STOP:
                self._queue.task_done()
                await self._finish_completion_ticks()
                return
            if not isinstance(first, _QueuedCapture):
                self._queue.task_done()
                raise RuntimeError("capture queue received an unknown work item")

            batch = [first]
            stop_after_batch = False
            while len(batch) < self._batch_size:
                try:
                    queued = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if queued is _STOP:
                    stop_after_batch = True
                    break
                if not isinstance(queued, _QueuedCapture):
                    self._queue.task_done()
                    raise RuntimeError(
                        "capture queue received an unknown work item"
                    )
                batch.append(queued)

            self._in_flight_enqueued_at = tuple(
                queued.enqueued_at for queued in batch
            )
            self.worker_batches += 1
            self.max_observed_batch_size = max(
                self.max_observed_batch_size,
                len(batch),
            )
            try:
                await self._await_sync(self._run_capture_batch, batch)
            finally:
                self._in_flight_enqueued_at = ()
                for queued in batch:
                    self._release(queued)
                    self._queue.task_done()
                if stop_after_batch:
                    self._queue.task_done()

            self._capture_batches_since_maintenance += 1
            maintenance_needed = self._completion is not None
            if stop_after_batch:
                await self._finish_completion_ticks()
                return

    async def _finish_completion_ticks(self) -> None:
        """Force exactly one bounded final tick, ignoring active backoff."""

        if self._completion is None:
            return
        await self._run_completion_tick()

    async def _run_completion_tick(self) -> tuple[bool, bool]:
        callback = self._completion
        if callback is None:
            return False, False
        self.completion_ticks += 1
        try:
            outcome = await self._await_sync(callback)
        except asyncio.CancelledError:
            raise
        except BaseException:
            self.completion_failures += 1
            logger.exception("capture completion callback failed")
            return False, True

        if outcome is None:
            return False, False
        self.t1_indexed_manifests += outcome.indexed_manifests
        self.t2_enriched_turns += outcome.enriched_turns
        self.t1_failures += int(outcome.t1_failed)
        self.t2_failures += int(outcome.t2_failed)
        self.completion_snapshot_failures += int(outcome.snapshot_failed)
        if outcome.failed:
            self.completion_failures += 1
        if outcome.snapshot is not None:
            self._pending_snapshot = outcome.snapshot
        stalled = bool(
            outcome.snapshot is not None
            and outcome.snapshot.has_pending
            and not outcome.made_progress
        )
        return outcome.should_continue, outcome.failed or stalled

    def _schedule_maintenance_retry(self) -> None:
        delay = self._next_maintenance_retry_delay
        self._maintenance_retry_at = time.monotonic() + delay
        self._next_maintenance_retry_delay = min(
            self._maintenance_retry_max_seconds,
            delay * 2,
        )
        self.completion_retries += 1

    def _schedule_maintenance_poll(self) -> None:
        if self._completion is not None:
            self._maintenance_poll_at = (
                time.monotonic() + self._maintenance_poll_seconds
            )

    def _clear_maintenance_retry(self) -> None:
        self._maintenance_retry_at = None
        self._next_maintenance_retry_delay = (
            self._maintenance_retry_initial_seconds
        )

    async def _await_sync(
        self,
        callback: Callable[..., Any],
        *args: Any,
    ) -> Any:
        work = asyncio.create_task(asyncio.to_thread(callback, *args))
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(work)
            except BaseException:
                logger.exception("synchronous capture work failed during cancellation")
            raise

    def _run_capture_batch(self, batch: Sequence[_QueuedCapture]) -> None:
        for queued in batch:
            delay = self._sink_retry_initial_seconds
            failures = 0
            while True:
                self.sink_attempts += 1
                try:
                    durable = self._sink(queued.capture)
                    if durable is False:
                        raise CaptureSinkNegativeAcknowledgementError(
                            "capture sink returned False"
                        )
                except BaseException:
                    failures += 1
                    self.capture_failures += 1
                    if failures > self._sink_max_retries:
                        self.capture_terminal_failures += 1
                        self._active_sink_retry_delay = 0.0
                        logger.exception(
                            "capture sink exhausted retries; releasing FIFO item"
                        )
                        break
                    self.capture_retries += 1
                    self._active_sink_retry_delay = delay
                    logger.exception("capture sink failed; retrying same FIFO item")
                    time.sleep(delay)
                    delay = min(self._sink_retry_max_seconds, delay * 2)
                    continue

                self._active_sink_retry_delay = 0.0
                self.sink_completed += 1
                self.durably_captured += int(durable is True)
                self.capture_retry_recoveries += int(failures > 0)
                break


def condenser_sink(condenser: Any) -> CaptureSink:
    """Synchronously ingest one exchange; retained for API compatibility."""

    def sink(capture: ExchangeCapture) -> bool:
        records = capture.ingest_records()
        if records:
            condenser.ingest_many(records)
        return True

    return sink


def condenser_capture_sink(condenser: Any) -> CaptureSink:
    """Durably publish one exchange without embedding or indexing it."""

    def sink(capture: ExchangeCapture) -> bool:
        records = capture.ingest_records()
        if records:
            condenser.capture_many(records)
        # An exchange with no ingestible records is a successful durable no-op,
        # not a negative acknowledgement that should spin through sink retries.
        return True

    return sink


def condenser_completion_callback(
    condenser: Any,
    *,
    max_manifests: int,
    max_chunks: int,
    max_tokens: int,
    max_enrichment_turns: int = 1,
) -> CompletionCallback:
    """Build a bounded T1/T2 tick with a worker-owned durable snapshot."""

    def complete() -> CompletionOutcome:
        indexed_manifests = 0
        enriched_turns = 0
        t1_failed = False
        t2_failed = False
        try:
            indexed_manifests = len(
                condenser.drain_pending_ingests(
                    max_manifests=max_manifests,
                    max_chunks=max_chunks,
                    max_tokens=max_tokens,
                    enrich=False,
                )
            )
        except BaseException:
            t1_failed = True
            logger.exception("pending-ingest completion failed")

        try:
            ingest = condenser.pending_ingest_stats()
        except BaseException:
            ingest = None
            logger.exception("pending-ingest snapshot failed")

        # T2 is distinct and never competes with a healthy T1 backlog. It
        # still gets one bounded chance when T1 itself fails, so already-
        # indexed enrichment receipts are independently recoverable.
        if t1_failed or indexed_manifests == 0 or (
            ingest is not None and int(ingest["manifest_count"]) == 0
        ):
            try:
                enriched_turns = len(
                    condenser.drain_pending_enrichments(
                        max_turns=max_enrichment_turns
                    )
                )
            except BaseException:
                t2_failed = True
                logger.exception("pending-enrichment completion failed")

        try:
            enrichment = condenser.pending_enrichment_stats()
            if ingest is None:
                raise RuntimeError("pending-ingest snapshot is unavailable")
            snapshot = PendingWorkSnapshot(
                sampled_at_monotonic=time.monotonic(),
                ingest_manifest_count=int(ingest["manifest_count"]),
                ingest_chunk_count=int(ingest["chunk_count"]),
                ingest_token_count=int(ingest["token_count"]),
                ingest_oldest_age_seconds=(
                    None
                    if ingest["oldest_age_seconds"] is None
                    else float(ingest["oldest_age_seconds"])
                ),
                enrichment_turn_count=int(enrichment["turn_count"]),
                enrichment_ready_count=int(enrichment["ready_count"]),
                enrichment_oldest_age_seconds=(
                    None
                    if enrichment["oldest_age_seconds"] is None
                    else float(enrichment["oldest_age_seconds"])
                ),
                ingest_failed_count=int(ingest.get("failed_count", 0)),
                ingest_oldest_error_kind=(
                    None
                    if ingest.get("oldest_error_kind") is None
                    else str(ingest["oldest_error_kind"])
                ),
                enrichment_failed_count=int(enrichment.get("failed_count", 0)),
                enrichment_oldest_error_kind=(
                    None
                    if enrichment.get("oldest_error_kind") is None
                    else str(enrichment["oldest_error_kind"])
                ),
                enrichment_deferred_correction_count=int(
                    enrichment.get("deferred_correction_count", 0)
                ),
                enrichment_discarded_legacy_count=int(
                    enrichment.get("discarded_legacy_count", 0)
                ),
            )
        except BaseException:
            logger.exception("pending-work snapshot failed")
            return CompletionOutcome(
                indexed_manifests=indexed_manifests,
                enriched_turns=enriched_turns,
                t1_failed=t1_failed,
                t2_failed=t2_failed,
                snapshot_failed=True,
            )
        return CompletionOutcome(
            indexed_manifests=indexed_manifests,
            enriched_turns=enriched_turns,
            t1_failed=t1_failed,
            t2_failed=t2_failed,
            snapshot=snapshot,
        )

    return complete


__all__ = [
    "CaptureQueue",
    "CaptureQueueClosedError",
    "CaptureQueueRejectedError",
    "CaptureQueueSaturatedError",
    "CaptureQueueShutdownTimeoutError",
    "CaptureSink",
    "CompletionCallback",
    "CompletionOutcome",
    "PendingWorkSnapshot",
    "condenser_capture_sink",
    "condenser_completion_callback",
    "condenser_sink",
]
