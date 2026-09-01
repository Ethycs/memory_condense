"""Transparent provider proxy: capture fidelity and forwarding safety."""

from __future__ import annotations

import json

import httpx
import pytest
from starlette.testclient import TestClient

from memory_condense.interfaces.proxy_capture import (
    ANTHROPIC,
    OPENAI,
    ExchangeCapture,
    StreamAccumulator,
    is_streaming_request,
    provider_for_path,
    request_messages,
    request_model,
    response_text,
)
from memory_condense.interfaces.proxy_server import (
    ProxyConfig,
    build_app,
    conversation_id_for,
    redacted_headers,
)


ANTHROPIC_SSE = (
    b'event: message_start\ndata: {"type":"message_start","message":{"id":"m1"}}\n\n'
    b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
    b'"content_block":{"type":"text","text":""}}\n\n'
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"text_delta","text":"Hello"}}\n\n'
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"text_delta","text":" world"}}\n\n'
    b'event: message_delta\ndata: {"type":"message_delta",'
    b'"delta":{"stop_reason":"end_turn"}}\n\n'
    b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
)

OPENAI_SSE = (
    b'data: {"choices":[{"delta":{"role":"assistant"},"index":0}]}\n\n'
    b'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n'
    b'data: {"choices":[{"delta":{"content":" world"},"index":0}]}\n\n'
    b'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}\n\n'
    b"data: [DONE]\n\n"
)


class TestCapture:
    def test_provider_routing(self):
        assert provider_for_path("/v1/messages") == ANTHROPIC
        assert provider_for_path("/v1/chat/completions") == OPENAI
        assert provider_for_path("/v1/models") is None

    def test_request_messages_reads_both_wire_formats(self):
        anthropic = json.dumps(
            {
                "model": "claude-opus-5",
                "system": "Be terse",
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                    {"role": "user", "content": "second"},
                ],
            }
        ).encode()
        parsed = request_messages(anthropic, conversation_id="c")
        assert [(m.role, m.text) for m in parsed] == [
            ("system", "Be terse"),
            ("user", "first"),
            ("assistant", "ok"),
            ("user", "second"),
        ]
        assert request_model(anthropic) == "claude-opus-5"

        openai = json.dumps(
            {
                "model": "gpt-5",
                "messages": [
                    {"role": "system", "content": "Be terse"},
                    {"role": "user", "content": "hi"},
                ],
            }
        ).encode()
        parsed = request_messages(openai, conversation_id="c")
        assert [(m.role, m.text) for m in parsed] == [
            ("system", "Be terse"),
            ("user", "hi"),
        ]

    def test_malformed_body_is_captured_as_nothing(self):
        assert request_messages(b"not json", conversation_id="c") == []
        assert request_model(b"not json") is None
        assert is_streaming_request(b"not json") is False

    def test_non_streaming_response_text(self):
        anthropic = json.dumps(
            {"content": [{"type": "text", "text": "answer"}, {"type": "tool_use"}]}
        ).encode()
        assert response_text(ANTHROPIC, anthropic) == "answer"

        openai = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": "answer"}}]}
        ).encode()
        assert response_text(OPENAI, openai) == "answer"

    def test_stream_accumulators_reassemble_text(self):
        anthropic = StreamAccumulator(provider=ANTHROPIC)
        anthropic.feed(ANTHROPIC_SSE)
        anthropic.close()
        assert anthropic.text == "Hello world"
        assert anthropic.stop_reason == "end_turn"

        openai = StreamAccumulator(provider=OPENAI)
        openai.feed(OPENAI_SSE)
        openai.close()
        assert openai.text == "Hello world"
        assert openai.stop_reason == "stop"

    def test_accumulator_handles_chunk_boundaries_mid_frame(self):
        accumulator = StreamAccumulator(provider=ANTHROPIC)
        for index in range(0, len(ANTHROPIC_SSE), 7):
            accumulator.feed(ANTHROPIC_SSE[index : index + 7])
        accumulator.close()
        assert accumulator.text == "Hello world"

    def test_ingest_records_take_last_user_turn_and_reply(self):
        prompt = tuple(
            request_messages(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "old"},
                            {"role": "assistant", "content": "old reply"},
                            {"role": "user", "content": "new question"},
                        ]
                    }
                ).encode(),
                conversation_id="conv",
            )
        )
        capture = ExchangeCapture(
            provider=ANTHROPIC,
            conversation_id="conv",
            model="claude-opus-5",
            prompt=prompt,
            reply_text="new answer",
            streamed=True,
            request_sha256="a" * 64,
            prompt_tokens_estimate=4,
        )
        records = capture.ingest_records()
        assert [(role, text) for role, text, *_ in records] == [
            ("user", "new question"),
            ("assistant", "new answer"),
        ]
        assert all(record[2] == "conv" for record in records)


class TestProxyServer:
    def _app(self, handler, captured: list[ExchangeCapture], **kwargs):
        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        return build_app(
            config=ProxyConfig(**kwargs),
            sink=captured.append,
            client=client,
        )

    def test_non_streaming_exchange_is_forwarded_and_captured(self):
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("x-api-key")
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": "upstream answer"}]},
            )

        captured: list[ExchangeCapture] = []
        with TestClient(self._app(handler, captured)) as client:
            response = client.post(
                "/v1/messages",
                json={"model": "claude-opus-5", "messages": [{"role": "user", "content": "hi"}]},
                headers={"x-api-key": "secret-key", "anthropic-version": "2023-06-01"},
            )

        assert response.status_code == 200
        assert response.json()["content"][0]["text"] == "upstream answer"
        # Request reached the real endpoint path, with the caller's key intact.
        assert seen["url"] == "https://api.anthropic.com/v1/messages"
        assert seen["auth"] == "secret-key"
        assert seen["body"]["messages"] == [{"role": "user", "content": "hi"}]

        assert len(captured) == 1
        assert captured[0].reply_text == "upstream answer"
        assert captured[0].model == "claude-opus-5"
        assert captured[0].streamed is False

    def test_streaming_bytes_are_forwarded_verbatim_while_captured(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=ANTHROPIC_SSE,
                headers={"content-type": "text/event-stream"},
            )

        captured: list[ExchangeCapture] = []
        with TestClient(self._app(handler, captured)) as client:
            response = client.post(
                "/v1/messages",
                json={
                    "model": "claude-opus-5",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        assert response.content == ANTHROPIC_SSE
        assert len(captured) == 1
        assert captured[0].reply_text == "Hello world"
        assert captured[0].streamed is True

    def test_openai_streaming_is_captured(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=OPENAI_SSE,
                headers={"content-type": "text/event-stream"},
            )

        captured: list[ExchangeCapture] = []
        with TestClient(self._app(handler, captured)) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-5",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        assert response.content == OPENAI_SSE
        assert captured[0].provider == OPENAI
        assert captured[0].reply_text == "Hello world"

    def test_uncaptured_paths_pass_through_untouched(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": ["model-a"]})

        captured: list[ExchangeCapture] = []
        with TestClient(self._app(handler, captured)) as client:
            response = client.get("/v1/models")

        assert response.status_code == 200
        assert response.json() == {"data": ["model-a"]}
        assert captured == []

    def test_upstream_error_is_relayed_and_not_captured(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                json={"error": {"type": "rate_limit_error"}},
            )

        captured: list[ExchangeCapture] = []
        with TestClient(self._app(handler, captured)) as client:
            response = client.post(
                "/v1/messages",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

        assert response.status_code == 429
        assert response.json()["error"]["type"] == "rate_limit_error"
        assert captured == []

    def test_unreachable_upstream_returns_502(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route", request=request)

        captured: list[ExchangeCapture] = []
        with TestClient(self._app(handler, captured)) as client:
            response = client.post(
                "/v1/messages",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

        assert response.status_code == 502
        assert response.json()["error"]["type"] == "upstream_error"

    def test_failing_sink_does_not_break_the_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})

        def explode(_capture: ExchangeCapture) -> None:
            raise RuntimeError("ingest exploded")

        transport = httpx.MockTransport(handler)
        app = build_app(
            config=ProxyConfig(),
            sink=explode,
            client=httpx.AsyncClient(transport=transport),
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/messages",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            assert response.status_code == 200
            health = client.get("/_memory/health").json()
        assert health["status"] == "ok"

    def test_custom_upstream_base_url_is_honored(self):
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            return httpx.Response(200, json={"content": []})

        captured: list[ExchangeCapture] = []
        app = self._app(
            handler,
            captured,
            upstreams={ANTHROPIC: "https://gateway.internal", OPENAI: "https://api.openai.com"},
        )
        with TestClient(app) as client:
            client.post("/v1/messages", json={"messages": []})
        assert seen["url"] == "https://gateway.internal/v1/messages"


class TestSafety:
    def test_secrets_are_redacted_from_receipts(self):
        view = redacted_headers(
            {
                "x-api-key": "sk-live-123",
                "authorization": "Bearer sk-live-456",
                "anthropic-version": "2023-06-01",
            }
        )
        assert view["x-api-key"] == "<redacted>"
        assert view["authorization"] == "<redacted>"
        assert view["anthropic-version"] == "2023-06-01"

    def test_conversation_id_prefers_client_thread_header(self):
        assert (
            conversation_id_for({"X-Memory-Conversation-Id": "thread-7"}, "ab" * 32)
            == "thread-7"
        )
        assert conversation_id_for({}, "ab" * 32).startswith("proxy:")

    def test_rewriting_mode_is_refused_until_implemented(self):
        with pytest.raises(ValueError, match="not enabled yet"):
            ProxyConfig(mode="augment")
