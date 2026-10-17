"""Provider proxy with transparent capture and opt-in native memory.

Observe mode forwards requests unchanged and captures conversations. Augment
mode uses the evaluated ChatSession lifecycle: recall, bounded recent context,
durable I/O, inline summaries when compatible, and background ingestion.

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

**One receipt per served request.**  Neither mode computes serving numbers
inline.  Both build an ``interfaces.proxy_receipts`` receipt once the request
has actually been served, from one estimator over the client's body and the
body that went upstream, preferring the upstream's own ``usage`` count where
the response carries it.  That is the only thing the status document's
``serving`` block is assembled from.

Credentials are forwarded and never stored: the client's own auth headers are
copied to the upstream request and excluded from every receipt.
"""

from __future__ import annotations

import hashlib
import asyncio
import gzip
import json
import logging
import math
import os
import ssl
import time
import zlib
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Sequence

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.concurrency import run_in_threadpool

from memory_condense.control.bulk import BulkLoader, ingested_tokens
from memory_condense.control.publisher import StatusPublisher
from memory_condense.control.snapshots import (
    SESSIONS_DIRNAME,
    CondenserHandle,
    SessionSnapshotStore,
    SnapshotStore,
)
from memory_condense.interfaces.proxy_sessions import (
    SOURCE_HEADER,
    TranscriptLocator,
    is_conversation_turn,
    resolve_conversation_id,
)
from memory_condense.interfaces.proxy_capture import (
    ANTHROPIC,
    OPENAI,
    ExchangeCapture,
    StreamAccumulator,
    estimate_prompt_tokens,
    format_for_path,
    is_streaming_request,
    provider_for_path,
    request_messages,
    request_model,
    response_text,
)
from memory_condense.interfaces import proxy_receipts
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
    #: ``observe`` forwards verbatim; ``augment`` uses a configured ChatSession
    #: service to replace older dialogue with authenticated recalled evidence.
    mode: str = "observe"
    timeout_seconds: float = 600.0
    codex_chatgpt: bool = False
    codex_base_url: str = 'https://chatgpt.com/backend-api/codex'
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
    #: Where the control plane writes ``control/status.json`` and keeps
    #: ``snapshots/``; ``None`` keeps the status document in memory only.
    data_dir: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("observe", "augment"):
            raise ValueError(
                "mode must be 'observe' or 'augment'"
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


def capture_health(captures: CaptureQueue) -> tuple[str, list[str]]:
    """``(status, reasons)`` for ``/health`` in observe mode; the status
    document mirrors this exact reasoning rather than recomputing it."""
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
    return status, health_reasons


def memory_health(snapshot) -> tuple[str, list[str]]:
    """``(status, reasons)`` for ``/health`` in augment mode."""
    if snapshot.get('failed_sessions'):
        return 'degraded', ['memory_sessions_failed']
    return 'ok', []


def default_data_dir() -> str:
    """Where a proxy run with no ``--data-dir`` publishes and keeps its store.

    Delegates to :func:`memory_condense.runtime.project.project_data_dir`, the
    one resolver the MCP server and ``control/status.py`` also use, so the
    writer and every reader land in the same place: an explicit
    ``$MEMORY_CONDENSE_DATA_DIR``, else ``<project root>/.memory_condense``,
    else the session's home. The old default was the relative ``"data"``,
    which is in nobody's search path.

    ``MEMORY_DATA_DIR`` is still honoured, after the canonical name, so an
    existing launcher keeps working -- but it is only visible to a reader that
    also sets ``MEMORY_CONDENSE_STATUS`` or the canonical variable.
    """
    legacy = os.environ.get("MEMORY_DATA_DIR")
    if legacy and legacy.strip() and not (os.environ.get("MEMORY_CONDENSE_DATA_DIR") or "").strip():
        return str(Path(legacy).expanduser())
    from memory_condense.runtime.project import project_data_dir

    return str(project_data_dir())


def build_app(
    *,
    config: ProxyConfig | None = None,
    sink: CaptureSink | None = None,
    completion: CompletionCallback | None = None,
    client: httpx.AsyncClient | None = None,
    memory=None,
    data_dir: str | os.PathLike[str] | None = None,
    store: CondenserHandle | None = None,
    locator: TranscriptLocator | None = None,
) -> Starlette:
    """Build the proxy application.

    ``sink`` receives one :class:`ExchangeCapture` at a time, in FIFO order,
    through the queue's single serialized scheduler. Returning exactly
    ``True`` acknowledges durable capture; ``None`` preserves compatibility
    with observational sinks. The ``completion`` callback drives bounded
    startup, idle, and final maintenance from the same serialized scheduler.
    ``client`` is injectable so tests can serve a transport without a socket.

    ``data_dir`` enables the control plane's on-disk status document and
    snapshot directory. ``save``/``load``/``reset`` then work in either mode:
    in observe mode through ``store`` (a :class:`CondenserHandle` whose sink
    and completion callback are the ones passed above), which closes, swaps,
    and reopens the condenser under the same lock the sink uses; in augment
    mode through ``memory.swap``, which closes every session store, mutates
    the ``sessions/`` tree, and lets the next request re-admit. Without a
    ``data_dir`` -- or, in observe mode, without ``store`` -- the control
    routes still answer but the mutating ones return 501. ``load-transcript``
    is observe-mode only in any case: it feeds the capture queue.
    """

    settings = config or ProxyConfig()
    if (settings.mode == 'augment') != (memory is not None):
        raise ValueError('Augment mode requires a memory service; observe mode does not use one')
    if data_dir is None:
        data_dir = settings.data_dir if settings.data_dir is not None else (
            store.data_dir if store is not None else None
        )
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
    # Control plane: one publisher owns status.json; bulk jobs ride the same
    # capture queue as live traffic; snapshots live beside the store.
    # Each mode gets the snapshot store that matches what its store *is*:
    # one SQLite file in observe mode, a tree of per-conversation stores
    # under sessions/<ns> in augment mode. Both duck-type the same surface,
    # so the control routes and the status publisher do not branch on mode.
    if data_dir is None:
        snapshots = None
    elif memory is not None:
        snapshots = SessionSnapshotStore(
            data_dir,
            sessions_dir=Path(getattr(memory, 'directory', data_dir)) / SESSIONS_DIRNAME,
        )
    else:
        snapshots = SnapshotStore(data_dir, db_path=store.db_path if store is not None else None)
    # Bulk loading rides the observe-mode capture queue, which augment mode
    # does not use at all; there is nothing to generalize here.
    bulk = (
        BulkLoader(captures, capacity=settings.capture_queue_size)
        if memory is None
        else None
    )
    # The thing a swap quiesces: the condenser handle in observe mode, the
    # session registry in augment mode. Both expose `swap(mutate)`,
    # `generation`, and a `closed` that is False while they are serving.
    swappable = store if store is not None else memory
    publisher = StatusPublisher(
        mode=settings.mode,
        captures=captures,
        capacity=settings.capture_queue_size,
        health=lambda: capture_health(captures),
        data_dir=data_dir,
        memory=memory,
        memory_health=memory_health,
        bulk=bulk,
        snapshots=snapshots,
        store_path=(lambda: store.db_path) if store is not None else None,
    )
    owns_client = client is None
    if client is None:
        import truststore
        upstream = httpx.AsyncClient(timeout=settings.timeout_seconds,
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT))
    else:
        upstream = client

    def upstream_url(provider: str | None, path: str, query: str) -> str:
        base = settings.upstreams.get(
            provider or settings.default_provider,
            settings.upstreams[settings.default_provider],
        ).rstrip("/")
        # Accept either the provider root or the familiar .../v1 base URL.
        # The ChatGPT Codex backend mounts the Responses API at
        # .../backend-api/codex/responses with no /v1 segment.
        if path.startswith('/v1/') and (base.endswith('/v1') or base.endswith('/backend-api/codex')):
            path = path[3:]
        return f"{base}{path}{'?' + query if query else ''}"

    def record_receipt(receipt) -> None:
        """Publish one serving receipt; never let bookkeeping break a reply.

        Every serving number in the status document comes through here, in
        both modes, so ``static_tokens`` and ``served_tokens`` are produced by
        one estimator and are always comparable. Call this *after* the request
        has been served: a receipt describes what happened, so a request that
        failed before reaching the upstream must not increment ``requests``
        or ``rewritten``.
        """
        try:
            proxy_receipts.publish(receipt, publisher)
        except Exception:  # pragma: no cover - defensive
            logger.exception("serving receipt failed")

    async def handle(request: Request) -> Response:
        path = "/" + request.path_params.get("path", "").lstrip("/")
        body = await request.body()
        provider = provider_for_path(path)
        fmt = format_for_path(path)
        headers = _forward_headers(request.headers.raw)
        target = upstream_url(provider, path, request.url.query)
        body_sha = hashlib.sha256(body).hexdigest()
        if provider is not None:
            conversation_id, conversation_source = resolve_conversation_id(
                headers, body, body_sha=body_sha, locator=locator
            )
        else:
            conversation_id, conversation_source = f"proxy:{body_sha[:16]}", "digest"

        if settings.codex_chatgpt and not path.startswith('/v1/'):
            # Codex uses a backend root without /v1 and sends other endpoints
            # (models, compaction) through the same configured provider.
            target = settings.codex_base_url.rstrip('/') + path
            if request.url.query:
                target += '?' + request.url.query
            if path.rstrip('/') == '/responses' and memory is not None:
                from memory_condense.interfaces.proxy_codex import conversation_headers
                from memory_condense.interfaces.proxy_memory import MemoryProxyError
                try:
                    encoding = headers.get('content-encoding', '').lower()
                    decoded = (gzip.decompress(body) if encoding == 'gzip' else
                               zlib.decompress(body) if encoding == 'deflate' else body)
                    if encoding not in ('', 'identity', 'gzip', 'deflate'):
                        raise MemoryProxyError('Unsupported Codex request compression: ' + encoding)
                    adapted = conversation_headers(json.loads(decoded), headers)
                    if adapted is not None:
                        headers = adapted
                        headers.pop('content-encoding', None)
                        body, provider, fmt = decoded, OPENAI, 'responses'
                except (MemoryProxyError, ValueError, OSError, zlib.error) as exc:
                    return JSONResponse({'error': {'message': str(exc)}}, status_code=400)

        if memory is not None and provider is not None:
            from memory_condense.interfaces.proxy_memory import MemoryProxyError, WireResponse
            if request.method != 'POST':
                return JSONResponse({'error':{'message':'Chat requests require POST'}},status_code=405)
            method = request.method
            if conversation_source != SOURCE_HEADER:
                # Clients such as Claude Code and Codex send no conversation
                # header; hand the memory service the resolved id the same way
                # an explicit header would. send_memory strips it upstream.
                headers = {**headers, 'x-memory-conversation-id': conversation_id}
            # What the memory service actually put on the wire for the
            # visible answer, and the upstream's own count of it. The service
            # never returns the served request, but it hands every byte of it
            # to this very callback, so there is nothing to guess at. Summary
            # generations are separate calls, not what the client asked for,
            # and are excluded; a retry overwrites with the same `sent`.
            served_body: bytes | None = None
            served_usage: int | None = None
            async def send_memory(payload, summary=False):
                nonlocal served_body, served_usage
                forwarded = {k:v for k,v in headers.items()
                             if not k.lower().startswith('x-memory-') and k.lower() != 'x-conversation-id'}
                if summary:
                    # Internal generations are distinct calls, not retries of
                    # the visible answer. Never reuse its idempotency identity.
                    forwarded = {k:v for k,v in forwarded.items() if k.lower() not in
                        ('idempotency-key','x-idempotency-key','x-request-id','request-id','x-client-request-id')}
                reply = await upstream.send(upstream.build_request(method,target,
                    headers=forwarded,content=payload),stream=True)
                try:
                    value = await reply.aread()
                    if not summary:
                        sent = payload.encode('utf-8') if isinstance(payload, str) else payload
                        if isinstance(sent, (bytes, bytearray)):
                            served_body = bytes(sent)
                        # The body is already fully buffered here, streaming
                        # or not, so the exact prompt count is read off it
                        # without keeping a second copy of the stream.
                        served_usage = proxy_receipts.usage_from_response(fmt, value)
                    return WireResponse(reply.status_code,dict(_response_headers(reply.headers)),value)
                finally:
                    await reply.aclose()
            loop = asyncio.get_running_loop()
            def send_from_worker(value, summary=False):
                # Compiler workers are ordinary threads and may outlive this
                # request. The shared HTTP client stays open through final drain.
                return asyncio.run_coroutine_threadsafe(send_memory(value,summary),loop).result()
            try:
                reply = await run_in_threadpool(memory.exchange,body,provider,headers,
                    send_from_worker, lambda value:send_from_worker(value,True),fmt)
            except MemoryProxyError as exc:
                return JSONResponse({'error':{'type':'memory_request_error','message':str(exc)}},status_code=exc.status)
            except Exception:
                logger.exception('memory request failed before provider completion')
                return JSONResponse({'error':{'type':'memory_error','message':'Memory request failed; inspect local service logs'}},status_code=503)
            # The request was served, so there is something to describe. A
            # served body identical to the client's is the pre-send guard's
            # sanctioned passthrough, and `receipt_for_augment` scores that as
            # not rewritten. A served body of None means nothing went upstream
            # at all -- an idempotent replay of an already-complete request --
            # so the served side stays unknown rather than being invented.
            record_receipt(proxy_receipts.receipt_for_augment(
                fmt, body, served_body, usage=served_usage
            ))
            if reply.headers.get('content-type','').startswith('text/event-stream'):
                # Inline summaries are validated before any visible bytes leave
                # the proxy. Native tool streams retain their original framing.
                return StreamingResponse(iter([reply.body]),status_code=reply.status,headers=reply.headers)
            return Response(reply.body,status_code=reply.status,headers=reply.headers)

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

        full_prompt = tuple(
            request_messages(body, conversation_id=conversation_id, fmt=fmt)
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
            # Codex sends title generation and compaction to the same endpoint
            # as real turns; those are not user exchanges and must not be
            # ingested as ones. Everything else is a turn by default.
            if not is_conversation_turn(headers, body):
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
            # Tokens of the text handed to the sink -- the same function bulk
            # jobs count with, so `totals.tokens_ingested` means one thing.
            publisher.record_capture(ingested_tokens(capture), conversation_id)

        if not streaming:
            payload = await response.aread()
            await response.aclose()
            # Observe mode forwards verbatim, so served equals static and
            # nothing is rewritten; the upstream's own prompt count is right
            # here in the body it just returned, and beats the estimate.
            record_receipt(proxy_receipts.receipt_for_observe(
                fmt, body, usage=proxy_receipts.usage_from_response(fmt, payload)
            ))
            try:
                await publish(response_text(provider, payload, fmt), False)
            except CaptureQueueRejectedError as exc:
                logger.warning("response capture rejected: %s", exc)
            except Exception:
                logger.exception("response capture failed")
            return Response(
                content=payload,
                status_code=response.status_code,
                headers=dict(_response_headers(response.headers)),
            )

        accumulator = StreamAccumulator(provider=provider, fmt=fmt or "")
        # Same chunks, same single pass: the usage accumulator shares the
        # stream the capture accumulator already sees, so reading the
        # upstream's exact prompt count costs no second copy of the body.
        usage = proxy_receipts.UsageAccumulator(fmt=fmt or "")

        async def tee() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_bytes():
                    try:
                        accumulator.feed(chunk)
                        usage.feed(chunk)
                    except Exception:  # never interrupt the client's stream
                        logger.exception("stream capture failed")
                    yield chunk
            finally:
                await response.aclose()
                try:
                    usage.close()
                except Exception:
                    logger.exception("stream usage finalization failed")
                # After the stream, not before it: the receipt describes a
                # request that was actually served.
                record_receipt(proxy_receipts.receipt_for_observe(
                    fmt, body, usage=usage.input_tokens
                ))
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
        if memory is not None:
            snapshot = await run_in_threadpool(memory.status)
            return JSONResponse(dict(mode='augment',status=memory_health(snapshot)[0],**snapshot))
        # Pending-ingest/enrichment projections use the condenser's SQLite
        # connection. The serialized worker publishes an immutable snapshot;
        # the event loop reads only that cache and never shares the connection.
        snapshot = captures.pending_snapshot
        sampled_now = time.monotonic()
        status, health_reasons = capture_health(captures)
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

    # -- control plane -----------------------------------------------------
    async def _json_body(request: Request) -> dict:
        try:
            value = await request.json()
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}

    def _not_implemented(message: str) -> Response:
        return JSONResponse({'error': {'type': 'not_implemented', 'message': message}}, status_code=501)

    async def control_status(_: Request) -> Response:
        return JSONResponse(await run_in_threadpool(publisher.publish, 'status'))

    async def control_save(request: Request) -> Response:
        if snapshots is None:
            return _not_implemented('snapshots need the proxy started with a --data-dir')
        body = await _json_body(request)
        name = body.get('name')
        if name is not None and not isinstance(name, str):
            return JSONResponse({'error': {'message': 'name must be a string'}}, status_code=400)

        def save():
            if store is not None:
                with store.lock:  # a quiescent point: no capture batch in flight
                    return snapshots.save(name)
            if memory is not None:
                # Same idea one level up: VACUUM INTO is already consistent
                # per database, and quiescing makes the whole tree one point
                # in time. Sessions stay open, so this costs no model reload.
                with memory.quiesce():
                    return snapshots.save(name)
            return snapshots.save(name)

        try:
            record = await run_in_threadpool(save)
        except ValueError as exc:
            return JSONResponse({'error': {'message': str(exc)}}, status_code=400)
        except FileExistsError as exc:
            return JSONResponse({'error': {'message': str(exc)}}, status_code=409)
        except FileNotFoundError as exc:
            return JSONResponse({'error': {'message': str(exc)}}, status_code=404)
        except Exception as exc:
            status = getattr(exc, 'status', None)
            if not isinstance(status, int):
                raise
            return JSONResponse({'error': {'message': str(exc)}}, status_code=status)
        logger.info('control: snapshot saved %s', record['file'])
        await run_in_threadpool(publisher.publish, 'snapshot')
        return JSONResponse(record)

    def _store_live() -> bool:
        """Is the store serving again? ``closed`` is False while either the
        condenser handle or the memory proxy is able to take work."""
        return not bool(getattr(swappable, 'closed', False))

    def _swap(mutate):
        """Quiesce, mutate the files, resume; the state is visible throughout.

        ``swappable.swap`` is the observe-mode :class:`CondenserHandle` or the
        augment-mode ``MemoryProxy``; both close every open store, run
        ``mutate``, and come back. A mutation that failed before touching the
        files leaves the store serving again, and a status document left
        wedged in ``error`` would then be a standing lie with no recovery
        route short of restarting the proxy. So only a store that did *not*
        come back is published as an error; the failure itself still
        propagates to the caller, which answers 500.
        """
        publisher.set_state('swapping')
        try:
            result = swappable.swap(mutate)
        except Exception as exc:
            if _store_live():
                publisher.set_state(None)
            else:
                publisher.set_state('error', error=f'{type(exc).__name__}: {exc}')
            raise
        publisher.set_state(None)
        return result

    def _no_swappable_store() -> Response:
        return _not_implemented(
            'this needs a store to swap: start the proxy with a --data-dir (observe '
            'mode also needs the condenser handle the installed command supplies)'
        )

    async def control_load(request: Request) -> Response:
        if snapshots is None or swappable is None:
            return _no_swappable_store()
        body = await _json_body(request)
        name = body.get('name')
        if not isinstance(name, str) or not name.strip():
            return JSONResponse({'error': {'message': 'name is required'}}, status_code=400)
        try:
            path = snapshots.path_for(name)
        except ValueError as exc:
            return JSONResponse({'error': {'message': str(exc)}}, status_code=400)
        if not snapshots.has_snapshot(name):
            return JSONResponse({'error': {'message': f'snapshot {name!r} does not exist'}}, status_code=404)
        try:
            await run_in_threadpool(_swap, lambda: snapshots.restore(name))
        except Exception as exc:
            logger.exception('control: load failed')
            return JSONResponse({'error': {'type': 'swap_failed', 'message': f'{type(exc).__name__}: {exc}',
                                           'store_live': _store_live()}}, status_code=500)
        logger.info('control: store loaded from snapshot %s', path.name)
        return JSONResponse({'loaded': path.name, 'store': str(snapshots.db_path),
                             'generation': swappable.generation})

    async def control_reset(request: Request) -> Response:
        body = await _json_body(request)
        if body.get('confirm') is not True:
            return JSONResponse({'error': {'message': 'reset requires {"confirm": true}'}}, status_code=400)
        if snapshots is None or swappable is None:
            return _no_swappable_store()

        def backup_then_clear():
            record = snapshots.save() if snapshots.has_store() else None
            snapshots.clear()
            return record

        try:
            record = await run_in_threadpool(_swap, backup_then_clear)
        except Exception as exc:
            logger.exception('control: reset failed')
            return JSONResponse({'error': {'type': 'swap_failed', 'message': f'{type(exc).__name__}: {exc}',
                                           'store_live': _store_live()}}, status_code=500)
        logger.info('control: store reset (backup %s)', record['file'] if record else 'none')
        return JSONResponse({'reset': True, 'backup': record, 'store': str(snapshots.db_path),
                             'generation': swappable.generation})

    async def control_load_transcript(request: Request) -> Response:
        if bulk is None:
            return _not_implemented(
                'bulk transcript loading is observe-mode only: it feeds the observe-mode '
                'capture queue, which augment mode does not run. In augment mode a '
                'transcript enters memory by being replayed through the proxy.'
            )
        body = await _json_body(request)
        session = body.get('session')
        if not isinstance(session, str) or not session.strip():
            return JSONResponse({'error': {'message': 'session (id or transcript path) is required'}}, status_code=400)
        try:
            job = await run_in_threadpool(bulk.submit, session.strip())
        except FileNotFoundError as exc:
            return JSONResponse({'error': {'message': str(exc)}}, status_code=404)
        except RuntimeError as exc:
            return JSONResponse({'error': {'message': str(exc)}}, status_code=503)
        logger.info('control: bulk job %s queued for %s', job.id, job.path)
        return JSONResponse({'job': job.queue_doc(job.position_at_submit), 'path': job.path}, status_code=202)

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        if memory is None:
            captures.start()
        if bulk is not None:
            bulk.bind_loop(asyncio.get_running_loop())
        publisher.start()
        try:
            yield
        finally:
            # Signal bulk work synchronously so no await lands between the
            # lifespan exit and the queue's own shutdown sequencing.
            if bulk is not None:
                bulk.request_stop()
            try:
                if memory is not None:
                    await run_in_threadpool(memory.close)
                else:
                    await captures.stop()
            finally:
                try:
                    if bulk is not None:
                        await run_in_threadpool(bulk.join)
                    await run_in_threadpool(publisher.stop)
                finally:
                    if owns_client:
                        await upstream.aclose()

    app = Starlette(
        routes=[
            Route("/_memory/health", health, methods=["GET"]),
            Route("/_control/status", control_status, methods=["GET"]),
            Route("/_control/save", control_save, methods=["POST"]),
            Route("/_control/load", control_load, methods=["POST"]),
            Route("/_control/reset", control_reset, methods=["POST"]),
            Route("/_control/load-transcript", control_load_transcript, methods=["POST"]),
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
    app.state.memory = memory
    app.state.publisher = publisher
    app.state.bulk = bulk
    app.state.snapshots = snapshots
    app.state.store = store
    app.state.swappable = swappable
    if memory is not None:
        async def flush(request):
            from memory_condense.interfaces.proxy_memory import MemoryProxyError
            provider = request.headers.get('x-memory-provider','openai')
            if provider not in (OPENAI,ANTHROPIC):
                return JSONResponse({'error':{'message':'Unknown memory provider'}},status_code=400)
            try:
                result = await run_in_threadpool(memory.flush,provider,dict(request.headers))
                return JSONResponse(result)
            except MemoryProxyError as exc:
                return JSONResponse({'error':{'message':str(exc)}},status_code=exc.status)
        app.router.routes.insert(0,Route('/_memory/flush',flush,methods=['POST']))
    return app


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    """Run the proxy with a condenser attached."""
    import argparse

    import uvicorn

    from memory_condense.application.condenser import MemoryCondenser

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--data-dir",
        default=default_data_dir(),
        help="control/status.json and snapshots/ live here (default: %(default)s)",
    )
    parser.add_argument('--mode',choices=('observe','augment'),default='observe')
    parser.add_argument('--inline-memory',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--memory-max-sessions',type=int,default=8)
    parser.add_argument('--memory-recent-exchanges',type=int,default=12)
    parser.add_argument('--memory-recent-token-budget',type=int,default=8192)
    parser.add_argument('--memory-retrieval-policy', choices=('auto','cap8','dialogue','hierarchy_neighborhoods','decision_history'), default='auto',
                        help='Default: BGE selects focused, overview, or decision/fix history recall; explicit modes override classification')
    parser.add_argument('--memory-dialogue-config', metavar='JSON',
                        help='Versioned neighborhood tuning profile for auto or explicit dialogue retrieval')
    parser.add_argument('--memory-experiment-config', metavar='JSON',
                        help='Opt-in bounded native experiments; absent means all experimental modules are off')
    parser.add_argument('--memory-empty-response-retries',type=int,choices=(0,1,2),default=2)
    parser.add_argument('--memory-summary-api-fallback',action=argparse.BooleanOptionalAction,default=True,
                        help='Escalate failed local summaries to the conversation provider/model')
    parser.add_argument('--assets-dir',help='Local model/runtime assets (otherwise saved setup configuration)')
    parser.add_argument('--codex-chatgpt', action='store_true',
                        help='Use Codex ChatGPT backend routing and stable account/thread identity')
    parser.add_argument('--codex-base-url', default='https://chatgpt.com/backend-api/codex',
                        help='Codex transport upstream; standard /v1 routes use their own provider URLs')
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
    from memory_condense.runtime.config import dialogue_policy, experiment_policy
    try:
        retrieval = dialogue_policy(args.memory_retrieval_policy, args.memory_dialogue_config)
        experiments = experiment_policy(args.memory_experiment_config)
        if experiments is not None and args.mode != 'augment':
            raise ValueError('--memory-experiment-config requires augment mode')
        if args.mode != 'augment' and (args.memory_retrieval_policy not in ('auto','cap8')
                                       or args.memory_dialogue_config is not None):
            raise ValueError('Dialogue retrieval requires --mode augment')
        if args.mode != 'augment':
            retrieval = None
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    config = ProxyConfig(
        mode=args.mode,
        codex_chatgpt=args.codex_chatgpt,
        codex_base_url=args.codex_base_url,
        default_provider=OPENAI if args.codex_chatgpt else ANTHROPIC,
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
        data_dir=args.data_dir,
    )
    if args.mode == 'augment':
        from memory_condense.interfaces.proxy_memory import MemoryProxy
        from memory_condense.runtime.config import RuntimeAssets
        from memory_condense.runtime.sessions import NativeProxySessions
        assets = RuntimeAssets.resolve(args.assets_dir)
        assets.require_files(include_summarizer=not args.memory_summary_api_fallback)
        factory = NativeProxySessions(args.data_dir,recent_exchanges=args.memory_recent_exchanges,
            recent_token_budget=args.memory_recent_token_budget,assets=assets,
            summary_api_fallback=args.memory_summary_api_fallback, dialogue_policy=retrieval,
            retrieval_policy=args.memory_retrieval_policy,experiment_policy=experiments)
        memory_class = MemoryProxy
        if args.codex_chatgpt:
            from memory_condense.interfaces.proxy_codex import CodexMemoryProxy
            memory_class = CodexMemoryProxy
        memory = memory_class(args.data_dir,factory,inline_memory=args.inline_memory,
            empty_response_retries=args.memory_empty_response_retries,max_sessions=args.memory_max_sessions,
            recent_token_budget=args.memory_recent_token_budget,summary_api_fallback=args.memory_summary_api_fallback)
        try:
            app = build_app(config=config,memory=memory,locator=TranscriptLocator())
            uvicorn.run(app,host=args.host,port=args.port,log_level='info')
        finally:
            if not memory.closed:
                memory.close()
        return 0
    # The handle owns the condenser so the control plane can close, swap,
    # and reopen it under the same lock the capture sink takes.
    store = CondenserHandle(
        lambda: MemoryCondenser(data_dir=args.data_dir), data_dir=args.data_dir
    )
    store.open()
    app: Starlette | None = None
    try:
        app = build_app(
            config=config,
            sink=store.capture_sink(),
            completion=store.completion_callback(
                max_manifests=config.capture_drain_max_manifests,
                max_chunks=config.capture_drain_max_chunks,
                max_tokens=config.capture_drain_max_tokens,
                max_enrichment_turns=config.capture_enrichment_max_turns,
            ),
            store=store,
            locator=TranscriptLocator(),
        )
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        captures = app.state.captures if app is not None else None
        if captures is None or captures.worker_quiesced:
            store.close()
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
    "capture_health",
    "condenser_capture_sink",
    "condenser_completion_callback",
    "condenser_sink",
    "conversation_id_for",
    "default_data_dir",
    "main",
    "redacted_headers",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
