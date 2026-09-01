"""A transparent proxy that sees provider traffic before the provider does.

Point a client's base URL at this server and it forwards every request to the
real upstream unchanged, while capturing the conversation on the way past.
That makes memory a property of the transport rather than something each
client has to integrate.

Two rules govern the design:

**Capture never breaks the call.**  Every capture path is wrapped; a parse
failure, a full queue, or an ingest error is recorded and dropped.  A proxy in
the critical path must fail open, always.

**Observe before augment.**  The default mode forwards request bytes exactly
as received, so installing the proxy cannot change any answer.  Rewriting the
prompt to swap bulk history for a retrieved packet is the point of the
project, but it is a separate, opt-in mode: a proxy that silently edits
prompts would be blamed for every quality regression downstream.

Credentials are forwarded and never stored: the client's own auth headers are
copied to the upstream request and excluded from every receipt.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Sequence

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

    def __post_init__(self) -> None:
        if self.mode != "observe":
            raise ValueError(
                "only 'observe' mode is implemented; prompt rewriting is not "
                "enabled yet"
            )
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.capture_queue_size < 1:
            raise ValueError("capture_queue_size must be positive")


CaptureSink = Callable[[ExchangeCapture], None]


class CaptureQueue:
    """Bounded hand-off from the request path to a background consumer.

    Ingest work (chunking, embedding, indexing) must not sit inside the
    response path, and it must not grow without bound if the consumer stalls.
    A full queue drops the oldest capture and counts it.
    """

    def __init__(self, sink: CaptureSink, *, maxsize: int = 256) -> None:
        self._sink = sink
        self._queue: asyncio.Queue[ExchangeCapture] = asyncio.Queue(maxsize=maxsize)
        self._task: asyncio.Task[None] | None = None
        self.dropped = 0
        self.ingested = 0
        self.failed = 0

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._drain())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def offer(self, capture: ExchangeCapture) -> None:
        try:
            self._queue.put_nowait(capture)
        except asyncio.QueueFull:
            self.dropped += 1
            logger.warning("capture queue full; dropped one exchange")

    async def _drain(self) -> None:
        while True:
            capture = await self._queue.get()
            try:
                await asyncio.to_thread(self._sink, capture)
                self.ingested += 1
            except Exception:  # capture must never take down the proxy
                self.failed += 1
                logger.exception("capture sink failed")
            finally:
                self._queue.task_done()


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
    client: httpx.AsyncClient | None = None,
) -> Starlette:
    """Build the proxy application.

    ``sink`` receives one :class:`ExchangeCapture` per completed exchange on a
    worker thread.  ``client`` is injectable so tests can serve a transport
    without a socket.
    """

    settings = config or ProxyConfig()
    captures = CaptureQueue(
        sink or (lambda capture: None),
        maxsize=settings.capture_queue_size,
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
        prompt = tuple(request_messages(body, conversation_id=conversation_id))
        model = request_model(body)
        streaming = is_streaming_request(body)

        def publish(reply_text: str, streamed: bool) -> None:
            if not prompt and not reply_text:
                return
            captures.offer(
                ExchangeCapture(
                    provider=provider,
                    conversation_id=conversation_id,
                    model=model,
                    prompt=prompt,
                    reply_text=reply_text,
                    streamed=streamed,
                    request_sha256=body_sha,
                    prompt_tokens_estimate=estimate_prompt_tokens(prompt),
                )
            )

        if not streaming:
            payload = await response.aread()
            await response.aclose()
            try:
                publish(response_text(provider, payload), False)
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
                    publish(accumulator.text, True)
                except Exception:
                    logger.exception("stream capture finalization failed")

        return StreamingResponse(
            tee(),
            status_code=response.status_code,
            headers=dict(_response_headers(response.headers)),
        )

    async def health(_: Request) -> Response:
        return JSONResponse(
            {
                "status": "ok",
                "mode": settings.mode,
                "captures_ingested": captures.ingested,
                "captures_dropped": captures.dropped,
                "captures_failed": captures.failed,
            }
        )

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        captures.start()
        try:
            yield
        finally:
            await captures.stop()
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


def condenser_sink(condenser: Any) -> CaptureSink:
    """Ingest captured exchanges into a live condenser."""

    def sink(capture: ExchangeCapture) -> None:
        records = capture.ingest_records()
        if records:
            condenser.ingest_many(records)

    return sink


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    """Run the proxy with a condenser attached."""
    import argparse

    import uvicorn

    from memory_condense.application.condenser import MemoryCondenser

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--data-dir", default=os.environ.get("MEMORY_DATA_DIR", "data"))
    parser.add_argument(
        "--anthropic-base-url",
        default=os.environ.get("ANTHROPIC_BASE_URL", DEFAULT_UPSTREAMS[ANTHROPIC]),
    )
    parser.add_argument(
        "--openai-base-url",
        default=os.environ.get("OPENAI_BASE_URL", DEFAULT_UPSTREAMS[OPENAI]),
    )
    args = parser.parse_args(argv)

    condenser = MemoryCondenser(data_dir=args.data_dir)
    config = ProxyConfig(
        upstreams={
            ANTHROPIC: args.anthropic_base_url,
            OPENAI: args.openai_base_url,
        }
    )
    app = build_app(config=config, sink=condenser_sink(condenser))
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        condenser.close()
    return 0


__all__ = [
    "CaptureQueue",
    "ProxyConfig",
    "build_app",
    "condenser_sink",
    "conversation_id_for",
    "main",
    "redacted_headers",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
