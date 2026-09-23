"""Admission, scheduling, and durable-completion tests for the provider proxy."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import replace

import pytest

from memory_condense.interfaces.proxy_capture import ANTHROPIC, ExchangeCapture
from memory_condense.interfaces.proxy_server import (
    CaptureQueue,
    CaptureQueueClosedError,
    CaptureQueueSaturatedError,
    CaptureQueueShutdownTimeoutError,
    CompletionOutcome,
    PendingWorkSnapshot,
    ProxyConfig,
    condenser_capture_sink,
    condenser_completion_callback,
    main,
)


def _queue_capture(index: int = 0) -> ExchangeCapture:
    return ExchangeCapture(
        provider=ANTHROPIC,
        conversation_id="conv",
        model="model",
        prompt=(),
        reply_text=f"answer-{index}",
        streamed=False,
        request_sha256=f"{index:064x}",
        prompt_tokens_estimate=0,
    )


def _pending_snapshot(
    *,
    ingest: int = 0,
    enrichment: int = 0,
    sampled_at: float = 0.0,
) -> PendingWorkSnapshot:
    return PendingWorkSnapshot(
        sampled_at_monotonic=sampled_at,
        ingest_manifest_count=ingest,
        ingest_chunk_count=ingest,
        ingest_token_count=ingest,
        ingest_oldest_age_seconds=(1.0 if ingest else None),
        enrichment_turn_count=enrichment,
        enrichment_ready_count=enrichment,
        enrichment_oldest_age_seconds=(2.0 if enrichment else None),
    )


def test_capture_sink_acknowledges_empty_exchange_as_durable_noop() -> None:
    class FakeCondenser:
        def capture_many(self, _records) -> None:
            raise AssertionError("empty exchange must not call capture_many")

    capture = ExchangeCapture(
        provider=ANTHROPIC,
        conversation_id="empty",
        model="model",
        prompt=(),
        reply_text="",
        streamed=False,
        request_sha256="0" * 64,
        prompt_tokens_estimate=0,
    )

    assert condenser_capture_sink(FakeCondenser())(capture) is True


class TestCaptureQueueLifecycle:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"capture_queue_size": True},
            {"capture_queue_max_waiters": 1.5},
            {"capture_offer_timeout_seconds": float("nan")},
            {"capture_shutdown_timeout_seconds": float("inf")},
            {"capture_batch_size": 0},
            {"capture_retained_byte_budget": True},
            {"capture_maintenance_batch_quota": 0},
            {"capture_maintenance_poll_seconds": 0},
            {"capture_maintenance_retry_initial_seconds": float("nan")},
            {
                "capture_maintenance_retry_initial_seconds": 2.0,
                "capture_maintenance_retry_max_seconds": 1.0,
            },
            {"capture_sink_retry_initial_seconds": float("inf")},
            {"capture_sink_max_retries": -1},
            {
                "capture_sink_retry_initial_seconds": 2.0,
                "capture_sink_retry_max_seconds": 1.0,
            },
            {"capture_drain_max_manifests": True},
            {"capture_drain_max_chunks": 1.5},
            {"capture_drain_max_tokens": -1},
            {"capture_enrichment_max_turns": 0},
        ],
    )
    def test_config_rejects_non_finite_or_non_integral_bounds(self, kwargs) -> None:
        with pytest.raises(ValueError):
            ProxyConfig(**kwargs)

    def test_waiter_budget_rejects_excess_without_retaining_it(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def sink(_capture: ExchangeCapture) -> None:
            entered.set()
            release.wait(timeout=5)

        async def scenario() -> None:
            queue = CaptureQueue(
                sink,
                maxsize=1,
                max_waiters=1,
                offer_timeout_seconds=5,
            )
            queue.start()
            capture = _queue_capture()
            await queue.offer_async(capture)
            assert await asyncio.to_thread(entered.wait, 1)
            await queue.offer_async(capture)
            waiting = asyncio.create_task(queue.offer_async(capture))
            for _ in range(10):
                await asyncio.sleep(0)
                if queue.waiting_producers:
                    break
            assert queue.waiting_producers == 1

            with pytest.raises(CaptureQueueSaturatedError, match="waiter budget"):
                await queue.offer_async(capture)

            assert queue.depth == 1
            assert queue.rejected == 1
            assert queue.rejected_saturated == 1
            assert queue.dropped == 1
            release.set()
            await asyncio.wait_for(waiting, timeout=1)
            await queue.stop()
            assert queue.accepted == 3
            assert queue.ingested == 3

        asyncio.run(scenario())

    def test_sync_offer_is_immediate_nonblocking_and_fail_open(self) -> None:
        captured: list[str] = []

        def sink(capture: ExchangeCapture) -> None:
            captured.append(capture.reply_text)

        async def scenario() -> None:
            queue = CaptureQueue(sink, maxsize=1)
            queue.start()
            assert queue.offer(_queue_capture(0)) is True
            assert queue.offer(_queue_capture(1)) is False
            assert queue.rejected_saturated == 1
            await queue.stop()
            assert captured == ["answer-0"]
            assert queue.offer(_queue_capture(2)) is False
            assert queue.rejected_closed == 1

        asyncio.run(scenario())

    def test_sync_offer_before_start_preserves_original_queueing_contract(self) -> None:
        captured: list[str] = []

        async def scenario() -> None:
            queue = CaptureQueue(
                lambda capture: captured.append(capture.reply_text),
                maxsize=1,
            )
            assert queue.offer(_queue_capture(7)) is True
            assert queue.depth == 1
            assert queue.retained_bytes > 0

            queue.start()
            await queue.stop()

            assert captured == ["answer-7"]
            assert queue.ingested == 1
            assert queue.retained_bytes == 0

        asyncio.run(scenario())

    def test_queue_restarts_cleanly_on_a_new_event_loop(self) -> None:
        captured: list[str] = []
        queue = CaptureQueue(
            lambda capture: captured.append(capture.reply_text),
            maxsize=1,
        )

        async def cycle(index: int) -> None:
            queue.start()
            await queue.offer_async(_queue_capture(index))
            await queue.stop()
            assert queue.state == "closed"

        asyncio.run(cycle(1))
        asyncio.run(cycle(2))

        assert captured == ["answer-1", "answer-2"]
        assert queue.accepted == 2
        assert queue.ingested == 2

    def test_retained_byte_budget_covers_waiter_queue_and_in_flight(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        captures = [_queue_capture(index) for index in range(4)]
        retained_size = captures[0].retained_bytes

        def sink(_capture: ExchangeCapture) -> None:
            entered.set()
            release.wait(timeout=5)

        async def scenario() -> None:
            queue = CaptureQueue(
                sink,
                maxsize=1,
                max_waiters=2,
                max_retained_bytes=retained_size * 3,
                offer_timeout_seconds=1,
            )
            queue.start()
            await queue.offer_async(captures[0])
            assert await asyncio.to_thread(entered.wait, 1)
            await queue.offer_async(captures[1])
            waiting = asyncio.create_task(queue.offer_async(captures[2]))
            for _ in range(20):
                await asyncio.sleep(0)
                if queue.waiting_producers:
                    break

            assert queue.waiting_producers == 1
            assert queue.retained_bytes == retained_size * 3
            with pytest.raises(CaptureQueueSaturatedError, match="retained-byte"):
                await queue.offer_async(captures[3])
            assert queue.rejected_retained_bytes == 1
            assert queue.retained_bytes == retained_size * 3

            release.set()
            await waiting
            await queue.stop()
            assert queue.retained_bytes == 0
            assert queue.peak_retained_bytes == retained_size * 3

        asyncio.run(scenario())

    def test_retained_budget_counts_utf8_and_rejects_single_oversize(self) -> None:
        capture = replace(_queue_capture(), reply_text="🙂" * 10)
        assert capture.retained_bytes >= 40

        async def scenario() -> None:
            queue = CaptureQueue(
                lambda _capture: True,
                max_retained_bytes=capture.retained_bytes - 1,
            )
            queue.start()
            assert queue.offer(capture) is False
            assert queue.rejected_retained_bytes == 1
            assert queue.retained_bytes == 0
            await queue.stop()

        asyncio.run(scenario())

    def test_greedy_batches_preserve_fifo_without_a_flush_sleep(self) -> None:
        captured: list[int] = []

        def sink(capture: ExchangeCapture) -> None:
            captured.append(int(capture.reply_text.removeprefix("answer-")))

        async def scenario() -> None:
            queue = CaptureQueue(sink, maxsize=8, batch_size=3)
            queue.start()
            for index in range(5):
                await queue.offer_async(_queue_capture(index))
            await queue.stop()

            assert captured == [0, 1, 2, 3, 4]
            assert queue.worker_batches == 2
            assert queue.max_observed_batch_size == 3
            assert queue.sink_completed == 5
            assert queue.durably_captured == 0

        asyncio.run(scenario())

    def test_blocked_async_producers_cannot_be_overtaken(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        captured: list[int] = []

        def sink(capture: ExchangeCapture) -> None:
            captured.append(int(capture.reply_text.removeprefix("answer-")))
            if len(captured) == 1:
                entered.set()
                release.wait(timeout=5)

        async def scenario() -> None:
            queue = CaptureQueue(
                sink,
                maxsize=1,
                max_waiters=2,
                batch_size=1,
                offer_timeout_seconds=2,
            )
            queue.start()
            await queue.offer_async(_queue_capture(0))
            assert await asyncio.to_thread(entered.wait, 1)
            await queue.offer_async(_queue_capture(1))
            older = asyncio.create_task(queue.offer_async(_queue_capture(2)))
            for _ in range(20):
                await asyncio.sleep(0)
                if queue.waiting_producers == 1:
                    break
            later = asyncio.create_task(queue.offer_async(_queue_capture(3)))
            for _ in range(20):
                await asyncio.sleep(0)
                if queue.waiting_producers == 2:
                    break

            release.set()
            await asyncio.gather(older, later)
            await queue.stop()

            assert captured == [0, 1, 2, 3]

        asyncio.run(scenario())

    def test_startup_completion_repeats_only_while_it_makes_progress(self) -> None:
        calls = 0
        finished = threading.Event()

        def complete() -> CompletionOutcome:
            nonlocal calls
            calls += 1
            remaining = max(0, 3 - calls)
            if remaining == 0:
                finished.set()
            return CompletionOutcome(
                indexed_manifests=int(calls <= 3),
                snapshot=_pending_snapshot(
                    ingest=remaining,
                    sampled_at=time.monotonic(),
                ),
            )

        async def scenario() -> None:
            queue = CaptureQueue(lambda _capture: True, completion=complete)
            queue.start()
            assert await asyncio.to_thread(finished.wait, 1)
            await queue.stop()

            assert calls == 4  # three startup drains plus the forced STOP tick
            assert queue.completion_ticks == 4
            assert queue.t1_indexed_manifests == 3
            assert queue.worker_batches == 0
            assert queue.worker_quiesced is True

        asyncio.run(scenario())

    def test_waiting_captures_run_before_each_completion_tick(self) -> None:
        events: list[str] = []

        def sink(capture: ExchangeCapture) -> bool:
            events.append(capture.reply_text)
            return True

        def complete() -> CompletionOutcome:
            events.append("completion")
            return CompletionOutcome(
                snapshot=_pending_snapshot(sampled_at=time.monotonic())
            )

        async def scenario() -> None:
            queue = CaptureQueue(
                sink,
                maxsize=8,
                batch_size=1,
                completion=complete,
            )
            queue.start()
            for index in range(3):
                await queue.offer_async(_queue_capture(index))
            await queue.stop()

            assert events == ["answer-0", "answer-1", "answer-2", "completion"]
            assert queue.durably_captured == 3
            assert queue.completion_ticks == 1

        asyncio.run(scenario())

    def test_stop_forces_one_bounded_completion_tick(self) -> None:
        remaining = 3

        def complete() -> CompletionOutcome:
            nonlocal remaining
            remaining -= 1
            return CompletionOutcome(
                indexed_manifests=1,
                snapshot=_pending_snapshot(
                    ingest=remaining,
                    sampled_at=time.monotonic(),
                ),
            )

        async def scenario() -> None:
            queue = CaptureQueue(
                lambda _capture: True,
                batch_size=1,
                completion=complete,
            )
            queue.start()
            await queue.offer_async(_queue_capture())
            await queue.stop()

            assert remaining == 2
            assert queue.completion_ticks == 1
            assert queue.pending_snapshot is not None
            assert queue.pending_snapshot.has_pending is True
            assert queue.state == "closed"

        asyncio.run(scenario())

    @pytest.mark.parametrize("failure_kind", ["callback", "t1", "t2", "snapshot"])
    def test_transient_completion_failure_retries_without_new_capture(
        self, failure_kind: str
    ) -> None:
        recovered = threading.Event()
        calls = 0
        call_times: list[float] = []

        def complete() -> CompletionOutcome:
            nonlocal calls
            calls += 1
            call_times.append(time.monotonic())
            if calls == 1:
                if failure_kind == "callback":
                    raise RuntimeError("transient callback outage")
                return CompletionOutcome(
                    t1_failed=failure_kind == "t1",
                    t2_failed=failure_kind == "t2",
                    snapshot_failed=failure_kind == "snapshot",
                    snapshot=_pending_snapshot(
                        ingest=1,
                        sampled_at=time.monotonic(),
                    ),
                )
            recovered.set()
            return CompletionOutcome(
                indexed_manifests=1,
                snapshot=_pending_snapshot(sampled_at=time.monotonic()),
            )

        async def scenario() -> None:
            queue = CaptureQueue(
                lambda _capture: True,
                completion=complete,
                maintenance_retry_initial_seconds=0.02,
                maintenance_retry_max_seconds=0.02,
            )
            queue.start()
            assert await asyncio.to_thread(recovered.wait, 1)
            assert call_times[1] - call_times[0] >= 0.01
            await queue.stop()

            assert calls == 3  # retry recovery plus the forced STOP tick
            assert queue.completion_retries == 1
            assert queue.completion_failures == 1

        asyncio.run(scenario())

    def test_idle_poll_discovers_work_committed_by_another_process(self) -> None:
        initial_tick = threading.Event()
        external_receipt = threading.Event()
        discovered = threading.Event()
        calls = 0

        def complete() -> CompletionOutcome:
            nonlocal calls
            calls += 1
            if calls == 1:
                initial_tick.set()
                return CompletionOutcome(
                    snapshot=_pending_snapshot(sampled_at=time.monotonic())
                )
            if external_receipt.is_set():
                external_receipt.clear()
                discovered.set()
                return CompletionOutcome(
                    indexed_manifests=1,
                    snapshot=_pending_snapshot(sampled_at=time.monotonic()),
                )
            return CompletionOutcome(
                snapshot=_pending_snapshot(sampled_at=time.monotonic())
            )

        async def scenario() -> None:
            queue = CaptureQueue(
                lambda _capture: True,
                completion=complete,
                maintenance_poll_seconds=0.02,
            )
            queue.start()
            assert await asyncio.to_thread(initial_tick.wait, 1)
            external_receipt.set()
            assert await asyncio.to_thread(discovered.wait, 1)
            for _ in range(100):
                if queue.t1_indexed_manifests == 1:
                    break
                await asyncio.sleep(0.001)
            assert queue.completion_idle_polls >= 1
            assert queue.t1_indexed_manifests == 1
            assert queue.worker_batches == 0
            await queue.stop()

        asyncio.run(scenario())

    def test_capture_and_stop_interrupt_the_idle_poll_wait(self) -> None:
        initial_tick = threading.Event()
        captured = threading.Event()
        events: list[str] = []

        def sink(capture: ExchangeCapture) -> bool:
            events.append(capture.reply_text)
            captured.set()
            return True

        def complete() -> CompletionOutcome:
            events.append("maintenance")
            initial_tick.set()
            return CompletionOutcome(
                snapshot=_pending_snapshot(sampled_at=time.monotonic())
            )

        async def scenario() -> None:
            queue = CaptureQueue(
                sink,
                completion=complete,
                maintenance_poll_seconds=5,
            )
            queue.start()
            assert await asyncio.to_thread(initial_tick.wait, 1)

            await queue.offer_async(_queue_capture(11))
            assert await asyncio.to_thread(captured.wait, 1)
            assert events[:2] == ["maintenance", "answer-11"]

            started = time.monotonic()
            await queue.stop()
            assert time.monotonic() - started < 1
            assert queue.completion_idle_polls == 0

        asyncio.run(scenario())

    def test_capture_arriving_during_maintenance_backoff_runs_first(self) -> None:
        events: list[str] = []
        failed = threading.Event()
        captured = threading.Event()
        recovered = threading.Event()
        completion_calls = 0

        def sink(capture: ExchangeCapture) -> bool:
            events.append(capture.reply_text)
            captured.set()
            return True

        def complete() -> CompletionOutcome:
            nonlocal completion_calls
            completion_calls += 1
            if completion_calls == 1:
                events.append("failed-maintenance")
                failed.set()
                return CompletionOutcome(
                    t1_failed=True,
                    snapshot=_pending_snapshot(
                        ingest=1,
                        sampled_at=time.monotonic(),
                    ),
                )
            events.append("maintenance-retry")
            recovered.set()
            return CompletionOutcome(
                indexed_manifests=1,
                snapshot=_pending_snapshot(sampled_at=time.monotonic()),
            )

        async def scenario() -> None:
            queue = CaptureQueue(
                sink,
                completion=complete,
                maintenance_retry_initial_seconds=0.1,
                maintenance_retry_max_seconds=0.1,
            )
            queue.start()
            assert await asyncio.to_thread(failed.wait, 1)
            await queue.offer_async(_queue_capture(7))
            assert await asyncio.to_thread(captured.wait, 1)
            assert events[:2] == ["failed-maintenance", "answer-7"]
            assert await asyncio.to_thread(recovered.wait, 1)
            await queue.stop()

        asyncio.run(scenario())

    def test_stop_forces_final_maintenance_tick_during_backoff(self) -> None:
        first_failed = threading.Event()
        calls = 0

        def complete() -> CompletionOutcome:
            nonlocal calls
            calls += 1
            if calls == 1:
                first_failed.set()
                return CompletionOutcome(
                    snapshot_failed=True,
                    snapshot=_pending_snapshot(
                        ingest=1,
                        sampled_at=time.monotonic(),
                    ),
                )
            return CompletionOutcome(
                indexed_manifests=1,
                snapshot=_pending_snapshot(sampled_at=time.monotonic()),
            )

        async def scenario() -> None:
            queue = CaptureQueue(
                lambda _capture: True,
                completion=complete,
                maintenance_retry_initial_seconds=5,
                maintenance_retry_max_seconds=5,
            )
            queue.start()
            assert await asyncio.to_thread(first_failed.wait, 1)
            await queue.stop()
            assert calls == 2
            assert queue.completion_retries == 1
            assert queue.state == "closed"

        asyncio.run(scenario())

    def test_capture_batch_quota_prevents_continuous_arrival_starvation(
        self,
    ) -> None:
        events: list[str] = []

        def sink(capture: ExchangeCapture) -> bool:
            events.append(capture.reply_text)
            return True

        def complete() -> CompletionOutcome:
            events.append("maintenance")
            return CompletionOutcome(
                snapshot=_pending_snapshot(sampled_at=time.monotonic())
            )

        async def scenario() -> None:
            queue = CaptureQueue(
                sink,
                maxsize=8,
                batch_size=1,
                maintenance_batch_quota=2,
                completion=complete,
            )
            queue.start()
            for index in range(6):
                assert queue.offer(_queue_capture(index)) is True
            await queue.stop()

            assert events[:3] == ["answer-0", "answer-1", "maintenance"]
            maintenance_positions = [
                index for index, event in enumerate(events) if event == "maintenance"
            ]
            assert maintenance_positions[:3] == [2, 5, 8]
            assert queue.sink_completed == 6

        asyncio.run(scenario())

    def test_batch_capture_failure_is_isolated_per_item(self) -> None:
        attempted: list[int] = []
        attempts_by_index: dict[int, int] = {}

        def sink(capture: ExchangeCapture) -> bool:
            index = int(capture.reply_text.removeprefix("answer-"))
            attempted.append(index)
            attempts_by_index[index] = attempts_by_index.get(index, 0) + 1
            if index in {1, 3} and attempts_by_index[index] == 1:
                raise RuntimeError(f"capture {index} failed")
            return True

        async def scenario() -> None:
            queue = CaptureQueue(
                sink,
                maxsize=8,
                batch_size=8,
                sink_retry_initial_seconds=0.001,
                sink_retry_max_seconds=0.001,
            )
            queue.start()
            for index in range(4):
                await queue.offer_async(_queue_capture(index))
            await queue.stop()

            assert attempted == [0, 1, 1, 2, 3, 3]
            assert queue.sink_completed == 4
            assert queue.durably_captured == 4
            assert queue.capture_failures == 2
            assert queue.capture_retries == 2
            assert queue.capture_retry_recoveries == 2
            assert queue.worker_batches == 1
            assert queue.state == "closed"

        asyncio.run(scenario())

    def test_t2_failure_cannot_strand_multitick_t1_backlog(self) -> None:
        t2_attempted = threading.Event()

        class FakeCondenser:
            def __init__(self) -> None:
                self.pending = 2
                self.enrichment_pending = True
                self.t1_calls = 0
                self.t2_calls = 0

            def drain_pending_ingests(self, **kwargs):
                assert kwargs["enrich"] is False
                self.t1_calls += 1
                if self.pending:
                    self.pending -= 1
                    return [object()]
                return []

            def drain_pending_enrichments(self, **kwargs):
                assert kwargs == {"max_turns": 1}
                self.t2_calls += 1
                if self.t2_calls == 1:
                    t2_attempted.set()
                    raise RuntimeError("synthetic T2 outage")
                if self.enrichment_pending:
                    self.enrichment_pending = False
                    return [object()]
                return []

            def pending_ingest_stats(self):
                return {
                    "manifest_count": self.pending,
                    "chunk_count": self.pending,
                    "token_count": self.pending,
                    "oldest_age_seconds": 1.0 if self.pending else None,
                }

            def pending_enrichment_stats(self):
                return {
                    "turn_count": int(self.enrichment_pending),
                    "ready_count": int(self.enrichment_pending),
                    "oldest_age_seconds": (
                        2.0 if self.enrichment_pending else None
                    ),
                }

        condenser = FakeCondenser()
        complete = condenser_completion_callback(
            condenser,
            max_manifests=1,
            max_chunks=1,
            max_tokens=1,
        )

        async def scenario() -> None:
            queue = CaptureQueue(
                lambda _capture: True,
                completion=complete,
                maintenance_retry_initial_seconds=0.001,
                maintenance_retry_max_seconds=0.001,
            )
            queue.start()
            assert await asyncio.to_thread(t2_attempted.wait, 1)
            for _ in range(100):
                if not condenser.enrichment_pending:
                    break
                await asyncio.sleep(0.005)
            await queue.stop()

            assert condenser.t1_calls >= 3
            assert condenser.t2_calls >= 2
            assert queue.t1_indexed_manifests == 2
            assert queue.t2_failures == 1
            assert queue.t2_enriched_turns == 1
            assert queue.completion_retries == 1
            assert queue.pending_snapshot is not None
            assert queue.pending_snapshot.ingest_manifest_count == 0
            assert queue.pending_snapshot.enrichment_turn_count == 0

        asyncio.run(scenario())

    def test_main_wires_capture_only_then_bounded_drain(
        self, monkeypatch, tmp_path
    ) -> None:
        import uvicorn

        import memory_condense.application.condenser as condenser_module

        calls: list[tuple[str, object]] = []

        class FakeCondenser:
            def __init__(self, data_dir) -> None:
                calls.append(("open", data_dir))

            def capture_many(self, records) -> None:
                calls.append(("capture", records))

            def drain_pending_ingests(self, **kwargs):
                calls.append(("drain", kwargs))
                return [object()]

            def drain_pending_enrichments(self, **kwargs):
                calls.append(("enrich", kwargs))
                return []

            def pending_ingest_stats(self):
                calls.append(("ingest-stats", None))
                return {
                    "manifest_count": 0,
                    "chunk_count": 0,
                    "token_count": 0,
                    "oldest_age_seconds": None,
                }

            def pending_enrichment_stats(self):
                calls.append(("enrichment-stats", None))
                return {
                    "turn_count": 0,
                    "ready_count": 0,
                    "oldest_age_seconds": None,
                }

            def close(self) -> None:
                calls.append(("close", None))

        def run(app, **_kwargs) -> None:
            async def scenario() -> None:
                assert app.state.config.capture_compact_prompts is True
                async with app.router.lifespan_context(app):
                    await app.state.captures.offer_async(_queue_capture(9))

            asyncio.run(scenario())

        monkeypatch.setattr(condenser_module, "MemoryCondenser", FakeCondenser)
        monkeypatch.setattr(uvicorn, "run", run)

        assert main(
            [
                "--data-dir",
                str(tmp_path),
                "--capture-batch-size",
                "1",
                "--capture-drain-max-manifests",
                "2",
                "--capture-drain-max-chunks",
                "3",
                "--capture-drain-max-tokens",
                "4",
            ]
        ) == 0

        assert [name for name, _value in calls] == [
            "open",
            "capture",
            "drain",
            "ingest-stats",
            "enrich",
            "enrichment-stats",
            "close",
        ]
        assert calls[2] == (
            "drain",
            {
                "max_manifests": 2,
                "max_chunks": 3,
                "max_tokens": 4,
                "enrich": False,
            },
        )

    def test_offer_timeout_rejects_instead_of_blocking_forever(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def sink(_capture: ExchangeCapture) -> None:
            entered.set()
            release.wait(timeout=5)

        async def scenario() -> None:
            queue = CaptureQueue(
                sink,
                maxsize=1,
                max_waiters=1,
                offer_timeout_seconds=0.01,
            )
            queue.start()
            capture = _queue_capture()
            await queue.offer_async(capture)
            assert await asyncio.to_thread(entered.wait, 1)
            await queue.offer_async(capture)

            with pytest.raises(CaptureQueueSaturatedError, match="timed out"):
                await queue.offer_async(capture)

            assert queue.offer_timeouts == 1
            assert queue.rejected_saturated == 1
            assert queue.waiting_producers == 0
            release.set()
            await queue.stop()

        asyncio.run(scenario())

    def test_stop_closes_admission_atomically_and_is_idempotent(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def sink(_capture: ExchangeCapture) -> None:
            entered.set()
            release.wait(timeout=5)

        async def scenario() -> None:
            queue = CaptureQueue(sink, maxsize=1, shutdown_timeout_seconds=1)
            queue.start()
            await queue.offer_async(_queue_capture())
            assert await asyncio.to_thread(entered.wait, 1)
            stopping = asyncio.create_task(queue.stop())
            await asyncio.sleep(0)
            assert queue.state == "closing"
            with pytest.raises(CaptureQueueClosedError):
                await queue.offer_async(_queue_capture())
            release.set()
            await stopping
            assert queue.state == "closed"
            await queue.stop()
            with pytest.raises(CaptureQueueClosedError):
                await queue.offer_async(_queue_capture())
            assert queue.rejected_closed == 2

        asyncio.run(scenario())

    def test_stop_rejects_producers_already_waiting_for_admission(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def sink(_capture: ExchangeCapture) -> None:
            entered.set()
            release.wait(timeout=5)

        async def scenario() -> None:
            queue = CaptureQueue(
                sink,
                maxsize=1,
                max_waiters=1,
                offer_timeout_seconds=5,
            )
            queue.start()
            await queue.offer_async(_queue_capture())
            assert await asyncio.to_thread(entered.wait, 1)
            await queue.offer_async(_queue_capture())
            waiting = asyncio.create_task(queue.offer_async(_queue_capture()))
            for _ in range(10):
                await asyncio.sleep(0)
                if queue.waiting_producers:
                    break
            assert queue.waiting_producers == 1

            stopping = asyncio.create_task(queue.stop())
            with pytest.raises(CaptureQueueClosedError, match="closed"):
                await waiting
            assert queue.rejected_closed == 1
            release.set()
            await stopping
            assert queue.accepted == 2
            assert queue.ingested == 2

        asyncio.run(scenario())

    def test_shutdown_timeout_is_bounded_and_sticky(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def sink(_capture: ExchangeCapture) -> None:
            entered.set()
            release.wait(timeout=5)

        async def scenario() -> None:
            queue = CaptureQueue(
                sink,
                maxsize=1,
                shutdown_timeout_seconds=0.01,
            )
            queue.start()
            await queue.offer_async(_queue_capture())
            assert await asyncio.to_thread(entered.wait, 1)
            with pytest.raises(CaptureQueueShutdownTimeoutError):
                await queue.stop()
            assert queue.state == "failed"
            assert queue.shutdown_timeouts == 1
            assert queue.worker_quiesced is False
            assert queue.in_flight == 1
            with pytest.raises(CaptureQueueClosedError):
                await queue.offer_async(_queue_capture())
            with pytest.raises(CaptureQueueShutdownTimeoutError):
                await queue.stop()
            assert queue.shutdown_timeouts == 1
            release.set()
            for _ in range(100):
                if queue.worker_quiesced:
                    break
                await asyncio.sleep(0.01)
            assert queue.worker_quiesced is True
            assert queue.in_flight == 0
            assert queue.sink_completed == 1

        asyncio.run(scenario())

    def test_sink_base_exception_retries_without_losing_the_capture(self) -> None:
        class SyntheticFatalSink(BaseException):
            pass

        calls = 0

        def sink(_capture: ExchangeCapture) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise SyntheticFatalSink("synthetic fatal sink failure")
            if calls == 2:
                raise asyncio.CancelledError("synthetic sink cancellation")

        async def scenario() -> None:
            queue = CaptureQueue(
                sink,
                maxsize=2,
                sink_retry_initial_seconds=0.001,
                sink_retry_max_seconds=0.001,
            )
            queue.start()
            await queue.offer_async(_queue_capture())
            await queue.offer_async(_queue_capture())
            await queue.offer_async(_queue_capture())
            await queue.stop()
            assert queue.state == "closed"
            assert queue.failed == 2
            assert queue.ingested == 3
            assert queue.capture_retries == 2
            assert queue.capture_retry_recoveries == 1
            assert queue.capture_terminal_failures == 0

        asyncio.run(scenario())

    def test_poison_capture_is_terminal_then_later_fifo_work_proceeds(self) -> None:
        attempts: list[int] = []

        def sink(capture: ExchangeCapture) -> bool:
            index = int(capture.reply_text.removeprefix("answer-"))
            attempts.append(index)
            if index == 0:
                raise RuntimeError("permanent poison capture")
            return True

        async def scenario() -> None:
            queue = CaptureQueue(
                sink,
                maxsize=2,
                sink_max_retries=1,
                sink_retry_initial_seconds=0.001,
                sink_retry_max_seconds=0.001,
            )
            queue.start()
            await queue.offer_async(_queue_capture(0))
            await queue.offer_async(_queue_capture(1))
            await queue.stop()

            assert attempts == [0, 0, 1]
            assert queue.capture_terminal_failures == 1
            assert queue.capture_failures == 2
            assert queue.capture_retries == 1
            assert queue.sink_completed == 1
            assert queue.durably_captured == 1
            assert queue.retained_bytes == 0
            assert queue.state == "closed"

        asyncio.run(scenario())

    def test_false_sink_acknowledgement_retries_before_releasing_capture(self) -> None:
        attempts = 0

        def sink(_capture: ExchangeCapture) -> bool:
            nonlocal attempts
            attempts += 1
            return attempts > 1

        async def scenario() -> None:
            queue = CaptureQueue(
                sink,
                sink_max_retries=1,
                sink_retry_initial_seconds=0.001,
                sink_retry_max_seconds=0.001,
            )
            queue.start()
            await queue.offer_async(_queue_capture())
            await queue.stop()

            assert attempts == 2
            assert queue.capture_failures == 1
            assert queue.capture_retries == 1
            assert queue.capture_retry_recoveries == 1
            assert queue.capture_terminal_failures == 0
            assert queue.durably_captured == 1

        asyncio.run(scenario())

    def test_t1_backoff_does_not_hide_ready_t2_work(self) -> None:
        class FakeCondenser:
            def __init__(self) -> None:
                self.t2_calls = 0

            def drain_pending_ingests(self, **kwargs):
                assert kwargs["enrich"] is False
                return []

            def drain_pending_enrichments(self, **kwargs):
                self.t2_calls += 1
                return [object()]

            def pending_ingest_stats(self):
                return {
                    "manifest_count": 1,
                    "chunk_count": 1,
                    "token_count": 10,
                    "oldest_age_seconds": 1.0,
                    "failed_count": 1,
                    "oldest_error_kind": "RuntimeError",
                }

            def pending_enrichment_stats(self):
                return {
                    "turn_count": 0,
                    "ready_count": 0,
                    "oldest_age_seconds": None,
                    "failed_count": 0,
                    "oldest_error_kind": None,
                }

        condenser = FakeCondenser()
        outcome = condenser_completion_callback(
            condenser,
            max_manifests=1,
            max_chunks=1,
            max_tokens=10,
        )()

        assert condenser.t2_calls == 1
        assert outcome.indexed_manifests == 0
        assert outcome.enriched_turns == 1
        assert outcome.snapshot.ingest_failed_count == 1

    def test_retry_keeps_capture_inside_retained_byte_budget_until_ack(self) -> None:
        failed_once = threading.Event()
        recovered = threading.Event()
        attempts = 0
        capture = _queue_capture(8)

        def sink(_capture: ExchangeCapture) -> bool:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                failed_once.set()
                raise RuntimeError("transient sink outage")
            recovered.set()
            return True

        async def scenario() -> None:
            queue = CaptureQueue(
                sink,
                max_retained_bytes=capture.retained_bytes,
                sink_retry_initial_seconds=0.1,
                sink_retry_max_seconds=0.1,
            )
            queue.start()
            await queue.offer_async(capture)
            assert await asyncio.to_thread(failed_once.wait, 1)
            assert queue.in_flight == 1
            assert queue.retained_bytes == capture.retained_bytes
            assert await asyncio.to_thread(recovered.wait, 1)
            await queue.stop()
            assert queue.retained_bytes == 0
            assert queue.capture_retry_recoveries == 1

        asyncio.run(scenario())

    def test_capture_queue_backpressures_without_dropping(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def sink(_capture: ExchangeCapture) -> None:
            entered.set()
            release.wait(timeout=5)

        capture = _queue_capture()

        async def scenario() -> None:
            queue = CaptureQueue(sink, maxsize=1)
            queue.start()
            await queue.offer_async(capture)
            await asyncio.to_thread(entered.wait, 5)
            await queue.offer_async(capture)
            blocked = asyncio.create_task(queue.offer_async(capture))
            await asyncio.sleep(0)
            assert not blocked.done()
            assert queue.depth == 1
            assert queue.in_flight == 1
            assert queue.oldest_age_seconds >= 0.0
            release.set()
            await asyncio.wait_for(blocked, timeout=5)
            await queue.stop()
            assert queue.accepted == 3
            assert queue.backpressured == 1
            assert queue.ingested == 3
            assert queue.failed == 0

        asyncio.run(scenario())
