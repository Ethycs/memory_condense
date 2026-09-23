"""Transparent provider proxy: capture fidelity and forwarding safety."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace

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
    CaptureQueueClosedError,
    CaptureQueueSaturatedError,
    CompletionOutcome,
    PendingWorkSnapshot,
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

    def test_compact_capture_ids_are_retry_stable_and_content_distinct(self):
        prompt = tuple(
            request_messages(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "older"},
                            {"role": "assistant", "content": "middle"},
                            {"role": "user", "content": "same slot"},
                        ]
                    }
                ).encode(),
                conversation_id="conv",
            )
        )
        first = ExchangeCapture(
            provider=ANTHROPIC,
            conversation_id="conv",
            model="model",
            prompt=prompt,
            reply_text="first reply",
            streamed=False,
            request_sha256="a" * 64,
            prompt_tokens_estimate=2,
        )
        another_request = replace(first, request_sha256="b" * 64)
        another_reply = replace(first, reply_text="different stochastic reply")

        first_records = first.ingest_records()
        assert first_records == first.ingest_records()
        assert len(first.prompt) == 3
        assert len(first.compact_for_ingest().prompt) == 1
        assert first_records[0][4] == f"conv:user:{'a' * 64}"
        assert (
            another_request.ingest_records()[0][4]
            == f"conv:user:{'b' * 64}"
        )
        reply_sha256 = hashlib.sha256(b"first reply").hexdigest()
        assert (
            first_records[-1][4]
            == f"conv:reply:{'a' * 64}:{reply_sha256}"
        )
        assert first_records[-1][4] != another_reply.ingest_records()[-1][4]


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

    def test_custom_sink_retains_full_prompt_by_default(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": "answer"}]},
            )

        captured: list[ExchangeCapture] = []
        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "middle"},
            {"role": "user", "content": "latest"},
        ]
        with TestClient(self._app(handler, captured)) as client:
            client.post("/v1/messages", json={"messages": history})

        assert [message.text for message in captured[0].prompt] == [
            "first",
            "middle",
            "latest",
        ]

    def test_production_compaction_drops_large_history_before_stream_queue(self):
        request_hash: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            request_hash.append(hashlib.sha256(request.content).hexdigest())
            return httpx.Response(
                200,
                content=ANTHROPIC_SSE,
                headers={"content-type": "text/event-stream"},
            )

        history = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"history-{index}-" + ("x" * 100_000),
            }
            for index in range(40)
        ]
        history.append({"role": "user", "content": "latest needle"})
        expected_tokens = sum(len(message["content"]) for message in history) // 4
        app = self._app(
            handler,
            [],
            capture_compact_prompts=True,
        )
        captured: list[ExchangeCapture] = []

        async def inspect(capture: ExchangeCapture) -> None:
            captured.append(capture)

        with TestClient(app) as client:
            app.state.captures.offer_async = inspect
            response = client.post(
                "/v1/messages",
                json={"stream": True, "messages": history},
            )

        assert response.content == ANTHROPIC_SSE
        assert [message.text for message in captured[0].prompt] == ["latest needle"]
        assert captured[0].prompt_tokens_estimate == expected_tokens
        assert captured[0].prompt_tokens_estimate >= 1_000_000
        assert captured[0].request_sha256 == request_hash[0]
        assert captured[0].retained_bytes < 512

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
        assert health["status"] in {"ok", "degraded"}

    def test_health_exposes_bounded_queue_observability(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"content": [{"type": "text", "text": "ok"}]}
            )

        captured: list[ExchangeCapture] = []
        with TestClient(self._app(handler, captured, capture_queue_size=3)) as client:
            client.post(
                "/v1/messages",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            health = client.get("/_memory/health").json()

        assert health["capture_queue_capacity"] == 3
        assert health["capture_queue_max_waiters"] == 3
        assert health["capture_offer_timeout_seconds"] == 1.0
        assert health["capture_shutdown_timeout_seconds"] == 30.0
        assert health["capture_batch_size"] == 16
        assert health["capture_retained_byte_budget"] == 16 * 1024 * 1024
        assert health["capture_compact_prompts"] is False
        assert health["capture_maintenance_batch_quota"] == 4
        assert health["capture_maintenance_poll_seconds"] == 5.0
        assert health["capture_maintenance_retry_initial_seconds"] == 0.05
        assert health["capture_maintenance_retry_max_seconds"] == 5.0
        assert health["capture_sink_retry_initial_seconds"] == 0.05
        assert health["capture_sink_retry_max_seconds"] == 5.0
        assert health["capture_sink_max_retries"] == 3
        assert health["capture_drain_max_manifests"] == 32
        assert health["capture_drain_max_chunks"] == 128
        assert health["capture_drain_max_tokens"] == 32_000
        assert health["capture_enrichment_max_turns"] == 1
        assert health["capture_completion_enabled"] is False
        assert health["status"] == "ok"
        assert health["health_reasons"] == []
        assert health["capture_queue_state"] == "running"
        assert health["capture_worker_quiesced"] is False
        assert health["captures_accepted"] == 1
        assert health["captures_dropped"] == 0
        assert health["captures_backpressured"] == 0
        assert health["captures_rejected"] == 0
        assert health["captures_rejected_saturated"] == 0
        assert health["captures_rejected_closed"] == 0
        assert health["capture_offer_timeouts"] == 0
        assert health["capture_offer_cancellations"] == 0
        assert 0 <= health["capture_queue_retained_bytes"]
        assert (
            health["capture_queue_retained_bytes"]
            <= health["capture_queue_peak_retained_bytes"]
            <= health["capture_retained_byte_budget"]
        )
        assert health["captures_rejected_retained_bytes"] == 0
        assert health["capture_sink_attempts"] in {0, 1}
        assert health["capture_shutdown_timeouts"] == 0
        assert health["capture_queue_depth"] >= 0
        assert health["capture_queue_in_flight"] in {0, 1}
        assert health["capture_queue_waiting_producers"] == 0
        assert health["capture_queue_oldest_age_seconds"] >= 0.0
        assert health["captures_sink_completed"] in {0, 1}
        assert health["captures_ingested"] == health["captures_sink_completed"]
        assert health["captures_durably_captured"] == 0
        assert health["capture_sink_failures"] == 0
        assert health["capture_sink_retries"] == 0
        assert health["capture_sink_retry_recoveries"] == 0
        assert health["captures_terminal_failures"] == 0
        assert health["capture_sink_retry_delay_seconds"] == 0.0
        assert health["capture_completion_failures"] == 0
        assert health["capture_completion_idle_polls"] == 0
        assert health["capture_completion_poll_delay_seconds"] == 0.0
        assert health["capture_completion_retries"] == 0
        assert health["capture_completion_retry_delay_seconds"] == 0.0
        assert health["capture_completion_ticks"] == 0
        assert health["capture_t1_indexed_manifests"] == 0
        assert health["capture_t1_failures"] == 0
        assert health["capture_t2_enriched_turns"] == 0
        assert health["capture_t2_failures"] == 0
        assert health["pending_snapshot_known"] is False
        assert health["pending_ingest_manifest_count"] is None
        assert health["pending_enrichment_turn_count"] is None
        assert health["capture_worker_batches"] in {0, 1}
        assert health["capture_max_observed_batch_size"] in {0, 1}
        assert health["capture_batches_since_maintenance"] in {0, 1}

    def test_health_degrades_after_terminal_capture_loss(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": "ok"}]},
            )

        def explode(_capture: ExchangeCapture) -> None:
            raise RuntimeError("permanent sink failure")

        app = build_app(
            config=ProxyConfig(
                capture_sink_max_retries=0,
                capture_sink_retry_initial_seconds=0.001,
                capture_sink_retry_max_seconds=0.001,
            ),
            sink=explode,
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/messages",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            assert response.status_code == 200
            for _ in range(100):
                health_response = client.get("/_memory/health")
                health = health_response.json()
                if health["captures_terminal_failures"]:
                    break
                time.sleep(0.005)

        assert health_response.status_code == 200
        assert health["status"] == "degraded"
        assert health["health_reasons"] == ["terminal_capture_loss"]
        assert health["captures_terminal_failures"] == 1

    def test_health_degrades_after_capture_admission_loss(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": "upstream ok"}]},
            )

        app = self._app(handler, [], capture_retained_byte_budget=1)
        with TestClient(app) as client:
            response = client.post(
                "/v1/messages",
                json={"messages": [{"role": "user", "content": "too large"}]},
            )
            health = client.get("/_memory/health").json()

        assert response.status_code == 200
        assert health["status"] == "degraded"
        assert health["health_reasons"] == ["capture_admission_loss"]
        assert health["captures_rejected_saturated"] == 1
        assert health["captures_rejected_retained_bytes"] == 1

    def test_health_fails_when_capture_worker_state_failed(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": []})

        app = self._app(handler, [])
        with TestClient(app) as client:
            captures = app.state.captures
            captures._state = "failed"
            try:
                health_response = client.get("/_memory/health")
                health = health_response.json()
            finally:
                captures._state = "running"

        assert health_response.status_code == 503
        assert health["status"] == "failed"
        assert health["health_reasons"] == ["capture_worker_failed"]

    def test_health_uses_worker_owned_age_aware_pending_snapshot(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": []})

        def complete() -> CompletionOutcome:
            return CompletionOutcome(
                snapshot=_pending_snapshot(
                    ingest=2,
                    enrichment=1,
                    sampled_at=time.monotonic(),
                )
            )

        app = build_app(
            config=ProxyConfig(),
            sink=lambda _capture: True,
            completion=complete,
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        with TestClient(app) as client:
            for _ in range(100):
                health = client.get("/_memory/health").json()
                if health["pending_snapshot_known"]:
                    break
                time.sleep(0.005)

            assert health["pending_ingest_manifest_count"] == 2
            assert health["pending_enrichment_turn_count"] == 1
            assert health["pending_ingest_oldest_age_seconds"] >= 1.0
            assert health["pending_enrichment_oldest_age_seconds"] >= 2.0
            assert health["pending_snapshot_age_seconds"] >= 0.0

    @pytest.mark.parametrize(
        "error",
        [
            CaptureQueueSaturatedError("synthetic saturation"),
            CaptureQueueClosedError("synthetic close"),
        ],
    )
    def test_capture_rejection_never_changes_upstream_response(self, error):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": "upstream ok"}]},
            )

        captured: list[ExchangeCapture] = []
        app = self._app(handler, captured)

        async def reject(_capture: ExchangeCapture) -> None:
            raise error

        with TestClient(app) as client:
            app.state.captures.offer_async = reject
            response = client.post(
                "/v1/messages",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

        assert response.status_code == 200
        assert response.json()["content"][0]["text"] == "upstream ok"
        assert captured == []

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
