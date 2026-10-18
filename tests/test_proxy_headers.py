"""End-to-end header and body fidelity through the observe-mode proxy.

The existing suite checks ``x-api-key`` only (``test_proxy.py``).  Claude
Code authenticates with an OAuth bearer token and announces capabilities in
``anthropic-beta``; a proxy that drops either breaks every request.  These
tests pin that the three auth/capability headers, the ``system`` array, and
an upstream rate-limit header all cross the proxy verbatim.
"""

from __future__ import annotations

import json
import time

import httpx
import pytest
from starlette.testclient import TestClient

from memory_condense.interfaces.proxy_capture import ExchangeCapture
from memory_condense.interfaces.proxy_server import ProxyConfig, build_app


CLIENT_HEADERS = {
    "content-type": "application/json",
    "anthropic-beta": "oauth-2025-04-20,context-1m-2025-08-07",
    "anthropic-version": "2023-06-01",
    "Authorization": "Bearer sk-ant-oat01-secret",
}

SYSTEM_BLOCKS = [
    {"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."},
    {
        "type": "text",
        "text": "Project instructions: keep answers terse.",
        "cache_control": {"type": "ephemeral"},
    },
]

ANTHROPIC_SSE = (
    b'event: message_start\ndata: {"type":"message_start","message":{"id":"m1",'
    b'"usage":{"input_tokens":3,"cache_read_input_tokens":1200}}}\n\n'
    b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
    b'"content_block":{"type":"text","text":""}}\n\n'
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"text_delta","text":"Hello"}}\n\n'
    b'event: message_delta\ndata: {"type":"message_delta",'
    b'"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}\n\n'
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)


def _body(stream: bool) -> bytes:
    return json.dumps(
        {
            "model": "claude-opus-5",
            "max_tokens": 64,
            "stream": stream,
            "system": SYSTEM_BLOCKS,
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                {"role": "user", "content": [{"type": "text", "text": "second"}]},
            ],
        }
    ).encode()


@pytest.fixture
def stack():
    """Observe-mode app over a mock upstream that records what it received."""
    received: list[httpx.Request] = []
    captured: list[ExchangeCapture] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        streaming = json.loads(request.content).get("stream")
        headers = {"anthropic-ratelimit-unified-status": "allowed", "request-id": "req_1"}
        if streaming:
            return httpx.Response(
                200,
                content=ANTHROPIC_SSE,
                headers={"content-type": "text/event-stream", **headers},
            )
        return httpx.Response(
            200,
            json={
                "id": "m1",
                "content": [{"type": "text", "text": "Hello"}],
                "usage": {"input_tokens": 3, "cache_read_input_tokens": 1200},
            },
            headers=headers,
        )

    app = build_app(
        config=ProxyConfig(mode="observe"),
        sink=captured.append,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with TestClient(app) as client:
        yield client, received, captured


@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
def test_oauth_headers_and_system_array_reach_upstream_verbatim(stack, stream):
    client, received, captured = stack
    body = _body(stream)

    response = client.post("/v1/messages", content=body, headers=CLIENT_HEADERS)

    assert response.status_code == 200
    assert len(received) == 1
    upstream = received[0]
    assert str(upstream.url) == "https://api.anthropic.com/v1/messages"

    # The three headers Claude Code depends on, byte for byte.
    assert upstream.headers["anthropic-beta"] == "oauth-2025-04-20,context-1m-2025-08-07"
    assert upstream.headers["anthropic-version"] == "2023-06-01"
    assert upstream.headers["authorization"] == "Bearer sk-ant-oat01-secret"
    assert "x-api-key" not in upstream.headers, "the proxy must not invent credentials"

    # Observe mode forwards the request bytes exactly as received, so the
    # system array (two blocks, cache_control intact) arrives unchanged.
    assert upstream.content == body
    sent = json.loads(upstream.content)
    assert sent["system"] == SYSTEM_BLOCKS
    assert [block["text"] for block in sent["system"]] == [block["text"] for block in SYSTEM_BLOCKS]
    assert sent["messages"] == json.loads(body)["messages"]

    # Upstream rate-limit telemetry reaches the client.
    assert response.headers["anthropic-ratelimit-unified-status"] == "allowed"
    assert response.headers["request-id"] == "req_1"
    if stream:
        assert response.content == ANTHROPIC_SSE
    else:
        assert response.json()["content"][0]["text"] == "Hello"

    # Capture is handed to the sink by the queue worker; give it a moment.
    deadline = time.monotonic() + 5.0
    while not captured and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(captured) == 1 and captured[0].reply_text == "Hello"
    assert captured[0].streamed is stream


def test_hop_by_hop_headers_are_recomputed_not_forwarded(stack):
    client, received, _ = stack
    body = _body(False)

    client.post(
        "/v1/messages",
        content=body,
        headers={**CLIENT_HEADERS, "accept-encoding": "br", "content-length": str(len(body))},
    )

    upstream = received[0]
    assert upstream.headers.get("accept-encoding") != "br"
    # httpx sets content-length itself from the forwarded body.
    assert upstream.headers["content-length"] == str(len(body))
    assert upstream.headers["host"] == "api.anthropic.com"
