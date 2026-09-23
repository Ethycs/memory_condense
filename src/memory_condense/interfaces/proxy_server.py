"""A transparent proxy that sees provider traffic before the provider does.

Point a client's base URL at this server and it forwards every request to the
real upstream unchanged, while capturing the conversation on the way past.
That makes memory a property of the transport rather than something each
client has to integrate.

Two rules govern the design:

**Capture admission is bounded.**  Every capture path is wrapped and ingest
errors are recorded. A full queue applies bounded backpressure, then rejects
the capture on waiter saturation or timeout without changing the upstream
response. The sink's source-journal commit—not this in-memory handoff—is the
crash-durability boundary.

**Observe before augment.**  The default mode forwards request bytes exactly
as received, so installing the proxy cannot change any answer.  Rewriting the
prompt to swap bulk history for a retrieved packet is the point of the
project, but it is a separate, opt-in mode: a proxy that silently edits
prompts would be blamed for every quality regression downstream.

Credentials are forwarded and never stored: the client's own auth headers are
copied to the upstream request and excluded from every receipt.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Sequence

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from memory_condense.interfaces.proxy_capture import (
    ANTHROPIC,
    OPENAI,
    ExchangeCapture,
    StreamAccumulator,
    estimate_prompt_tokens,
    is_streaming_request,
    provider_for_path,
    request_messages,
    request_model,
    response_text,
)
from memory_condense.interfaces.proxy_scheduler import (
    CaptureQueue,
    CaptureQueueClosedError,
    CaptureQueueRejectedError,
    CaptureQueueSaturatedError,
    CaptureQueueShutdownTimeoutError,
    CaptureSink,
    CompletionCallback,
    CompletionOutcome,
    PendingWorkSnapshot,
    condenser_capture_sink,
    condenser_completion_callback,
    condenser_sink,
)


logger = logging.getLogger(__name__)

#: Never forwarded upstream: hop-by-hop or recomputed by httpx.
_DROPPED_REQUEST_HEADERS = frozenset(
    {"host", "content-length", "connection", "transfer-encoding", "accept-encoding"}
)
#: Never returned to the client: httpx already decoded the body.
_DROPPED_RESPONSE_HEADERS = frozenset(
    {"content-length", "content-encoding", "transfer-encoding", "connection"}
)
#: Redacted from receipts.  The proxy forwards these but must not retain them.
_SECRET_HEADERS = frozenset(
    {"authorization", "x-api-key", "api-key", "openai-api-key", "cookie"}
)

DEFAULT_UPSTREAMS = {
    ANTHROPIC: "https://api.anthropic.com",
    OPENAI: "https://api.openai.com",
}


@dataclass(frozen=True, slots=True)
class ProxyConfig:
    """Where to forward, and how much liberty the proxy has with requests."""

    upstreams: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_UPSTREAMS)
    )
    default_provider: str = ANTHROPIC
    #: ``observe`` forwards request bytes verbatim.  Only this mode exists
    #: today; ``augment`` is reserved for prompt rewriting and is rejected
    #: rather than silently behaving like ``observe``.
    mode: str = "observe"
    timeout_seconds: float = 600.0
    capture_queue_size: int = 256
    capture_queue_max_waiters: int | None = None
    capture_offer_timeout_seconds: float = 1.0
    capture_shutdown_timeout_seconds: float = 30.0
    capture_batch_size: int = 16
    capture_retained_byte_budget: int = 16 * 1024 * 1024
    capture_compact_prompts: bool = False
    capture_maintenance_batch_quota: int = 4
    capture_maintenance_poll_seconds: float = 5.0
    capture_maintenance_retry_initial_seconds: float = 0.05
    capture_maintenance_retry_max_seconds: float = 5.0
    capture_sink_retry_initial_seconds: float = 0.05
    capture_sink_retry_max_seconds: float = 5.0
    capture_sink_max_retries: int = 3
    capture_drain_max_manifests: int = 32
    capture_drain_max_chunks: int = 128
    capture_drain_max_tokens: int = 32_000
    capture_enrichment_max_turns: int = 1

    def __post_init__(self) -> None:
        if self.mode != "observe":
            raise ValueError(
                "only 'observe' mode is implemented; prompt rewriting is not "
                "enabled yet"
            )
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a finite positive number")
        if type(self.capture_queue_size) is not int or self.capture_queue_size < 1:
            raise ValueError("capture_queue_size must be a positive integer")
        if (
            self.capture_queue_max_waiters is not None
            and (
                type(self.capture_queue_max_waiters) is not int
                or self.capture_queue_max_waiters < 0
            )
        ):
            raise ValueError("capture_queue_max_waiters must be non-negative")
        if type(self.capture_compact_prompts) is not bool:
            raise ValueError("capture_compact_prompts must be a boolean")
        for name, value in (
            ("capture_batch_size", self.capture_batch_size),
            ("capture_retained_byte_budget", self.capture_retained_byte_budget),
            (
                "capture_maintenance_batch_quota",
                self.capture_maintenance_batch_quota,
            ),
            ("capture_drain_max_manifests", self.capture_drain_max_manifests),
            ("capture_drain_max_chunks", self.capture_drain_max_chunks),
            ("capture_drain_max_tokens", self.capture_drain_max_tokens),
            ("capture_enrichment_max_turns", self.capture_enrichment_max_turns),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            type(self.capture_sink_max_retries) is not int
            or self.capture_sink_max_retries < 0
        ):
            raise ValueError("capture_sink_max_retries must be non-negative")
        for name, value in (
            ("capture_offer_timeout_seconds", self.capture_offer_timeout_seconds),
            (
                "capture_shutdown_timeout_seconds",
                self.capture_shutdown_timeout_seconds,
            ),
            (
                "capture_maintenance_poll_seconds",
                self.capture_maintenance_poll_seconds,
            ),
            (
                "capture_maintenance_retry_initial_seconds",
                self.capture_maintenance_retry_initial_seconds,
            ),
            (
                "capture_maintenance_retry_max_seconds",
                self.capture_maintenance_retry_max_seconds,
            ),
            (
                "capture_sink_retry_initial_seconds",
                self.capture_sink_retry_initial_seconds,
            ),
            (
                "capture_sink_retry_max_seconds",
                self.capture_sink_retry_max_seconds,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number")
        if (
            self.capture_maintenance_retry_max_seconds
            < self.capture_maintenance_retry_initial_seconds
        ):
            raise ValueError(
                "capture maintenance retry maximum must be at least its initial delay"
            )
        if (
            self.capture_sink_retry_max_seconds
            < self.capture_sink_retry_initial_seconds
        ):
            raise ValueError(
                "capture sink retry maximum must be at least its initial delay"
            )


def _forward_headers(headers: Sequence[tuple[bytes, bytes]]) -> dict[str, str]:
    return {
        key.decode("latin-1"): value.decode("latin-1")
        for key, value in headers
        if key.decode("latin-1").lower() not in _DROPPED_REQUEST_HEADERS
    }


def _response_headers(headers: httpx.Headers) -> list[tuple[str, str]]:
    return [
        (key, value)
        for key, value in headers.multi_items()
        if key.lower() not in _DROPPED_RESPONSE_HEADERS
    ]


def redacted_headers(headers: dict[str, str]) -> dict[str, str]:
    """Header view safe to log: secrets replaced, not merely truncated."""
    return {
        key: ("<redacted>" if key.lower() in _SECRET_HEADERS else value)
        for key, value in headers.items()
    }


def conversation_id_for(headers: dict[str, str], body_sha: str) -> str:
    """Prefer an explicit client-supplied thread ID, else the request digest.

    A client that sets ``x-memory-conversation-id`` gets its turns grouped into
    one source; without it each exchange stands alone, which is correct but
    loses threading.
    """
    for key, value in headers.items():
        if key.lower() in {"x-memory-conversation-id", "x-conversation-id"}:
            if value.strip():
                return value.strip()
    return f"proxy:{body_sha[:16]}"


def build_app(
    *,
    config: ProxyConfig | None = None,
    sink: CaptureSink | None = None,
    completion: CompletionCallback | None = None,
    client: httpx.AsyncClient | None = None,
) -> Starlette:
    """Build the proxy application.

    ``sink`` receives one :class:`ExchangeCapture` at a time, in FIFO order,
    through the queue's single serialized scheduler. Returning exactly
    ``True`` acknowledges durable capture; ``None`` preserves compatibility
    with observational sinks. The ``completion`` callback drives bounded
    startup, idle, and final maintenance from the same serialized scheduler.
    ``client`` is injectable so tests can serve a transport without a socket.
    """

    settings = config or ProxyConfig()
    captures = CaptureQueue(
        sink or (lambda capture: None),
        maxsize=settings.capture_queue_size,
        max_waiters=settings.capture_queue_max_waiters,
        offer_timeout_seconds=settings.capture_offer_timeout_seconds,
        shutdown_timeout_seconds=settings.capture_shutdown_timeout_seconds,
        batch_size=settings.capture_batch_size,
        max_retained_bytes=settings.capture_retained_byte_budget,
        compact_captures=settings.capture_compact_prompts,
        maintenance_batch_quota=settings.capture_maintenance_batch_quota,
        maintenance_poll_seconds=settings.capture_maintenance_poll_seconds,
        maintenance_retry_initial_seconds=(
            settings.capture_maintenance_retry_initial_seconds
        ),
        maintenance_retry_max_seconds=(
            settings.capture_maintenance_retry_max_seconds
        ),
        sink_retry_initial_seconds=settings.capture_sink_retry_initial_seconds,
        sink_retry_max_seconds=settings.capture_sink_retry_max_seconds,
        sink_max_retries=settings.capture_sink_max_retries,
        completion=completion,
    )
    owns_client = client is None
    upstream = client or httpx.AsyncClient(timeout=settings.timeout_seconds)

    def upstream_url(provider: str | None, path: str, query: str) -> str:
        base = settings.upstreams.get(
            provider or settings.default_provider,
            settings.upstreams[settings.default_provider],
        ).rstrip("/")
        return f"{base}{path}{'?' + query if query else ''}"

    async def handle(request: Request) -> Response:
        path = "/" + request.path_params.get("path", "").lstrip("/")
        body = await request.body()
        provider = provider_for_path(path)
        headers = _forward_headers(request.headers.raw)
        target = upstream_url(provider, path, request.url.query)

        upstream_request = upstream.build_request(
            request.method,
            target,
            headers=headers,
            content=body,
        )
        try:
            response = await upstream.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            logger.warning("upstream request failed: %s", exc)
            return JSONResponse(
                {
                    "error": {
                        "type": "upstream_error",
                        "message": f"proxy could not reach upstream: {exc}",
                    }
                },
                status_code=502,
            )

        # Nothing to learn from non-chat traffic or a failed call; stream it
        # straight through so the proxy stays transparent for every endpoint.
        if provider is None or response.status_code >= 400:
            return StreamingResponse(
                response.aiter_bytes(),
                status_code=response.status_code,
                headers=dict(_response_headers(response.headers)),
                background=BackgroundTask(response.aclose),
            )

        body_sha = hashlib.sha256(body).hexdigest()
        conversation_id = conversation_id_for(headers, body_sha)
        full_prompt = tuple(
            request_messages(body, conversation_id=conversation_id)
        )
        prompt_tokens_estimate = estimate_prompt_tokens(full_prompt)
        if settings.capture_compact_prompts:
            last_user = next(
                (
                    message
                    for message in reversed(full_prompt)
                    if message.role == "user"
                ),
                None,
            )
            prompt = () if last_user is None else (last_user,)
        else:
            prompt = full_prompt
        model = request_model(body)
        streaming = is_streaming_request(body)

        async def publish(reply_text: str, streamed: bool) -> None:
            if not prompt and not reply_text:
                return
            capture = ExchangeCapture(
                provider=provider,
                conversation_id=conversation_id,
                model=model,
                prompt=prompt,
                reply_text=reply_text,
                streamed=streamed,
                request_sha256=body_sha,
                prompt_tokens_estimate=prompt_tokens_estimate,
            )
            if settings.capture_compact_prompts:
                capture = capture.compact_for_ingest()
            await captures.offer_async(capture)

        if not streaming:
            payload = await response.aread()
            await response.aclose()
            try:
                await publish(response_text(provider, payload), False)
            except CaptureQueueRejectedError as exc:
                logger.warning("response capture rejected: %s", exc)
            except Exception:
                logger.exception("response capture failed")
            return Response(
                content=payload,
                status_code=response.status_code,
                headers=dict(_response_headers(response.headers)),
            )

        accumulator = StreamAccumulator(provider=provider)

        async def tee() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_bytes():
                    try:
                        accumulator.feed(chunk)
                    except Exception:  # never interrupt the client's stream
                        logger.exception("stream capture failed")
                    yield chunk
            finally:
                await response.aclose()
                try:
                    accumulator.close()
                    await publish(accumulator.text, True)
                except CaptureQueueRejectedError as exc:
                    logger.warning("stream response capture rejected: %s", exc)
                except Exception:
                    logger.exception("stream capture finalization failed")

        return StreamingResponse(
            tee(),
            status_code=response.status_code,
            headers=dict(_response_headers(response.headers)),
        )

    async def health(_: Request) -> Response:
        # Pending-ingest/enrichment projections use the condenser's SQLite
        # connection. The serialized worker publishes an immutable snapshot;
        # the event loop reads only that cache and never shares the connection.
        snapshot = captures.pending_snapshot
        sampled_now = time.monotonic()
        health_reasons: list[str] = []
        worker_failed = captures.state == "failed" or (
            captures.state == "running" and captures.worker_quiesced
        )
        if worker_failed:
            health_reasons.append("capture_worker_failed")
        if captures.rejected_saturated:
            health_reasons.append("capture_admission_loss")
        if captures.capture_terminal_failures:
            health_reasons.append("terminal_capture_loss")
        status = (
            "failed"
            if "capture_worker_failed" in health_reasons
            else "degraded" if health_reasons else "ok"
        )
        return JSONResponse(
            {
                "status": status,
                "health_reasons": health_reasons,
                "mode": settings.mode,
                "capture_queue_capacity": settings.capture_queue_size,
                "capture_queue_max_waiters": captures.max_waiters,
                "capture_offer_timeout_seconds": (
                    settings.capture_offer_timeout_seconds
                ),
                "capture_shutdown_timeout_seconds": (
                    settings.capture_shutdown_timeout_seconds
                ),
                "capture_batch_size": settings.capture_batch_size,
                "capture_retained_byte_budget": (
                    settings.capture_retained_byte_budget
                ),
                "capture_compact_prompts": settings.capture_compact_prompts,
                "capture_maintenance_batch_quota": (
                    settings.capture_maintenance_batch_quota
                ),
                "capture_maintenance_poll_seconds": (
                    settings.capture_maintenance_poll_seconds
                ),
                "capture_maintenance_retry_initial_seconds": (
                    settings.capture_maintenance_retry_initial_seconds
                ),
                "capture_maintenance_retry_max_seconds": (
                    settings.capture_maintenance_retry_max_seconds
                ),
                "capture_sink_retry_initial_seconds": (
                    settings.capture_sink_retry_initial_seconds
                ),
                "capture_sink_retry_max_seconds": (
                    settings.capture_sink_retry_max_seconds
                ),
                "capture_sink_max_retries": settings.capture_sink_max_retries,
                "capture_drain_max_manifests": (
                    settings.capture_drain_max_manifests
                ),
                "capture_drain_max_chunks": settings.capture_drain_max_chunks,
                "capture_drain_max_tokens": settings.capture_drain_max_tokens,
                "capture_enrichment_max_turns": (
                    settings.capture_enrichment_max_turns
                ),
                "capture_completion_enabled": completion is not None,
                "capture_queue_depth": captures.depth,
                "capture_queue_in_flight": captures.in_flight,
                "capture_queue_waiting_producers": captures.waiting_producers,
                "capture_queue_oldest_age_seconds": captures.oldest_age_seconds,
                "capture_queue_state": captures.state,
                "capture_worker_quiesced": captures.worker_quiesced,
                "captures_accepted": captures.accepted,
                "captures_backpressured": captures.backpressured,
                "captures_rejected": captures.rejected,
                "captures_rejected_saturated": captures.rejected_saturated,
                "captures_rejected_closed": captures.rejected_closed,
                "capture_offer_timeouts": captures.offer_timeouts,
                "capture_offer_cancellations": captures.offer_cancellations,
                "capture_queue_retained_bytes": captures.retained_bytes,
                "capture_queue_peak_retained_bytes": (
                    captures.peak_retained_bytes
                ),
                "captures_rejected_retained_bytes": (
                    captures.rejected_retained_bytes
                ),
                "capture_sink_attempts": captures.sink_attempts,
                "captures_sink_completed": captures.sink_completed,
                "captures_ingested": captures.ingested,
                "captures_durably_captured": captures.durably_captured,
                "capture_sink_failures": captures.capture_failures,
                "capture_sink_retries": captures.capture_retries,
                "capture_sink_retry_recoveries": (
                    captures.capture_retry_recoveries
                ),
                "captures_terminal_failures": (
                    captures.capture_terminal_failures
                ),
                "capture_sink_retry_delay_seconds": (
                    captures.sink_retry_delay_seconds
                ),
                "capture_completion_failures": captures.completion_failures,
                "capture_completion_idle_polls": captures.completion_idle_polls,
                "capture_completion_poll_delay_seconds": (
                    captures.maintenance_poll_delay_seconds
                ),
                "capture_completion_retries": captures.completion_retries,
                "capture_completion_retry_delay_seconds": (
                    captures.maintenance_retry_delay_seconds
                ),
                "capture_completion_ticks": captures.completion_ticks,
                "capture_t1_indexed_manifests": captures.t1_indexed_manifests,
                "capture_t1_failures": captures.t1_failures,
                "capture_t2_enriched_turns": captures.t2_enriched_turns,
                "capture_t2_failures": captures.t2_failures,
                "capture_completion_snapshot_failures": (
                    captures.completion_snapshot_failures
                ),
                "pending_snapshot_known": snapshot is not None,
                "pending_snapshot_age_seconds": (
                    snapshot.age_at(sampled_now) if snapshot is not None else None
                ),
                "pending_ingest_manifest_count": (
                    snapshot.ingest_manifest_count
                    if snapshot is not None
                    else None
                ),
                "pending_ingest_chunk_count": (
                    snapshot.ingest_chunk_count if snapshot is not None else None
                ),
                "pending_ingest_token_count": (
                    snapshot.ingest_token_count if snapshot is not None else None
                ),
                "pending_ingest_oldest_age_seconds": (
                    snapshot.ingest_oldest_age_at(sampled_now)
                    if snapshot is not None
                    else None
                ),
                "pending_ingest_failed_count": (
                    snapshot.ingest_failed_count if snapshot is not None else None
                ),
                "pending_ingest_oldest_error_kind": (
                    snapshot.ingest_oldest_error_kind
                    if snapshot is not None
                    else None
                ),
                "pending_enrichment_turn_count": (
                    snapshot.enrichment_turn_count
                    if snapshot is not None
                    else None
                ),
                "pending_enrichment_ready_count": (
                    snapshot.enrichment_ready_count
                    if snapshot is not None
                    else None
                ),
                "pending_enrichment_oldest_age_seconds": (
                    snapshot.enrichment_oldest_age_at(sampled_now)
                    if snapshot is not None
                    else None
                ),
                "pending_enrichment_failed_count": (
                    snapshot.enrichment_failed_count
                    if snapshot is not None
                    else None
                ),
                "pending_enrichment_oldest_error_kind": (
                    snapshot.enrichment_oldest_error_kind
                    if snapshot is not None
                    else None
                ),
                "pending_enrichment_deferred_correction_count": (
                    snapshot.enrichment_deferred_correction_count
                    if snapshot is not None
                    else None
                ),
                "pending_enrichment_discarded_legacy_count": (
                    snapshot.enrichment_discarded_legacy_count
                    if snapshot is not None
                    else None
                ),
                "capture_worker_batches": captures.worker_batches,
                "capture_max_observed_batch_size": (
                    captures.max_observed_batch_size
                ),
                "capture_batches_since_maintenance": (
                    captures.capture_batches_since_maintenance
                ),
                "captures_dropped": captures.dropped,
                "captures_failed": captures.failed,
                "capture_shutdown_timeouts": captures.shutdown_timeouts,
            },
            status_code=503 if status == "failed" else 200,
        )

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        captures.start()
        try:
            yield
        finally:
            try:
                await captures.stop()
            finally:
                if owns_client:
                    await upstream.aclose()

    app = Starlette(
        routes=[
            Route("/_memory/health", health, methods=["GET"]),
            Route(
                "/{path:path}",
                handle,
                methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
            ),
        ],
        lifespan=lifespan,
    )
    app.state.captures = captures
    app.state.config = settings
    return app


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    """Run the proxy with a condenser attached."""
    import argparse

    import uvicorn

    from memory_condense.application.condenser import MemoryCondenser

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--data-dir", default=os.environ.get("MEMORY_DATA_DIR", "data"))
    parser.add_argument("--capture-queue-size", type=int, default=256)
    parser.add_argument("--capture-queue-max-waiters", type=int)
    parser.add_argument("--capture-offer-timeout-seconds", type=float, default=1.0)
    parser.add_argument(
        "--capture-shutdown-timeout-seconds", type=float, default=30.0
    )
    parser.add_argument("--capture-batch-size", type=int, default=16)
    parser.add_argument(
        "--capture-retained-byte-budget", type=int, default=16 * 1024 * 1024
    )
    parser.add_argument(
        "--retain-full-capture-prompt",
        action="store_true",
        help="retain resent history for custom sinks (higher queue memory use)",
    )
    parser.add_argument("--capture-maintenance-batch-quota", type=int, default=4)
    parser.add_argument(
        "--capture-maintenance-poll-seconds", type=float, default=5.0
    )
    parser.add_argument(
        "--capture-maintenance-retry-initial-seconds", type=float, default=0.05
    )
    parser.add_argument(
        "--capture-maintenance-retry-max-seconds", type=float, default=5.0
    )
    parser.add_argument(
        "--capture-sink-retry-initial-seconds", type=float, default=0.05
    )
    parser.add_argument(
        "--capture-sink-retry-max-seconds", type=float, default=5.0
    )
    parser.add_argument("--capture-sink-max-retries", type=int, default=3)
    parser.add_argument("--capture-drain-max-manifests", type=int, default=32)
    parser.add_argument("--capture-drain-max-chunks", type=int, default=128)
    parser.add_argument("--capture-drain-max-tokens", type=int, default=32_000)
    parser.add_argument("--capture-enrichment-max-turns", type=int, default=1)
    parser.add_argument(
        "--anthropic-base-url",
        default=os.environ.get("ANTHROPIC_BASE_URL", DEFAULT_UPSTREAMS[ANTHROPIC]),
    )
    parser.add_argument(
        "--openai-base-url",
        default=os.environ.get("OPENAI_BASE_URL", DEFAULT_UPSTREAMS[OPENAI]),
    )
    args = parser.parse_args(argv)

    config = ProxyConfig(
        upstreams={
            ANTHROPIC: args.anthropic_base_url,
            OPENAI: args.openai_base_url,
        },
        capture_queue_size=args.capture_queue_size,
        capture_queue_max_waiters=args.capture_queue_max_waiters,
        capture_offer_timeout_seconds=args.capture_offer_timeout_seconds,
        capture_shutdown_timeout_seconds=args.capture_shutdown_timeout_seconds,
        capture_batch_size=args.capture_batch_size,
        capture_retained_byte_budget=args.capture_retained_byte_budget,
        capture_compact_prompts=not args.retain_full_capture_prompt,
        capture_maintenance_batch_quota=args.capture_maintenance_batch_quota,
        capture_maintenance_poll_seconds=args.capture_maintenance_poll_seconds,
        capture_maintenance_retry_initial_seconds=(
            args.capture_maintenance_retry_initial_seconds
        ),
        capture_maintenance_retry_max_seconds=(
            args.capture_maintenance_retry_max_seconds
        ),
        capture_sink_retry_initial_seconds=args.capture_sink_retry_initial_seconds,
        capture_sink_retry_max_seconds=args.capture_sink_retry_max_seconds,
        capture_sink_max_retries=args.capture_sink_max_retries,
        capture_drain_max_manifests=args.capture_drain_max_manifests,
        capture_drain_max_chunks=args.capture_drain_max_chunks,
        capture_drain_max_tokens=args.capture_drain_max_tokens,
        capture_enrichment_max_turns=args.capture_enrichment_max_turns,
    )
    condenser = MemoryCondenser(data_dir=args.data_dir)
    app: Starlette | None = None
    try:
        app = build_app(
            config=config,
            sink=condenser_capture_sink(condenser),
            completion=condenser_completion_callback(
                condenser,
                max_manifests=config.capture_drain_max_manifests,
                max_chunks=config.capture_drain_max_chunks,
                max_tokens=config.capture_drain_max_tokens,
                max_enrichment_turns=config.capture_enrichment_max_turns,
            ),
        )
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        captures = app.state.captures if app is not None else None
        if captures is None or captures.worker_quiesced:
            condenser.close()
        else:
            # An arbitrary Python/C worker cannot be killed safely. The
            # durable journal will recover on restart, and deliberately not
            # closing here avoids racing a still-live SQLite/embedder call.
            logger.critical(
                "capture worker is not quiescent; deferring condenser close"
            )
    return 0


__all__ = [
    "CompletionOutcome",
    "CaptureQueueClosedError",
    "CaptureQueueRejectedError",
    "CaptureQueueSaturatedError",
    "CaptureQueueShutdownTimeoutError",
    "CaptureQueue",
    "PendingWorkSnapshot",
    "ProxyConfig",
    "build_app",
    "condenser_capture_sink",
    "condenser_completion_callback",
    "condenser_sink",
    "conversation_id_for",
    "main",
    "redacted_headers",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
