"""Serving receipts: static vs. served token pairs for the status document."""

from __future__ import annotations

import json

import pytest

from memory_condense.interfaces.proxy_capture import (
    estimate_prompt_tokens,
    request_messages,
)
from memory_condense.interfaces.proxy_receipts import (
    ServingReceipt,
    UsageAccumulator,
    prompt_tokens,
    publish,
    receipt_for_augment,
    receipt_for_observe,
    usage_from_response,
    usage_input_tokens,
)


ANTHROPIC_BODY = json.dumps(
    {
        "model": "claude-opus-5",
        "stream": True,
        "system": "You are terse.",
        "messages": [
            {"role": "user", "content": "first question " * 40},
            {"role": "assistant", "content": [{"type": "text", "text": "first answer " * 40}]},
            {"role": "user", "content": [{"type": "text", "text": "latest question"}]},
        ],
    }
).encode()

RESPONSES_BODY = json.dumps(
    {
        "model": "gpt-5",
        "instructions": "Be brief.",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "first " * 50}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "reply " * 50}]},
            {"role": "user", "content": [{"type": "input_text", "text": "latest"}]},
        ],
    }
).encode()

ANTHROPIC_SSE_WITH_USAGE = (
    b'event: message_start\ndata: {"type":"message_start","message":{"id":"m1",'
    b'"usage":{"input_tokens":3,"cache_creation_input_tokens":40,'
    b'"cache_read_input_tokens":1200,"output_tokens":1}}}\n\n'
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"text_delta","text":"Hello"}}\n\n'
    b'event: message_delta\ndata: {"type":"message_delta",'
    b'"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}\n\n'
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)

RESPONSES_SSE_WITH_USAGE = (
    b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"Hi"}\n\n'
    b'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_1",'
    b'"status":"completed","usage":{"input_tokens":512,"output_tokens":9,"total_tokens":521}}}\n\n'
)

CHAT_SSE_WITH_USAGE = (
    b'data: {"choices":[{"delta":{"content":"Hi"},"index":0}]}\n\n'
    b'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}\n\n'
    b'data: {"choices":[],"usage":{"prompt_tokens":321,"completion_tokens":2,"total_tokens":323}}\n\n'
    b"data: [DONE]\n\n"
)


def _expected(fmt: str, body: bytes) -> int:
    return estimate_prompt_tokens(request_messages(body, conversation_id="x", fmt=fmt))


class _FakePublisher:
    def __init__(self) -> None:
        self.calls: list[tuple[int | None, int | None, bool]] = []

    def record_serving(self, static_tokens, served_tokens, rewritten) -> None:
        self.calls.append((static_tokens, served_tokens, rewritten))


class TestEstimates:
    def test_anthropic_body_uses_the_capture_estimator(self):
        expected = _expected("anthropic", ANTHROPIC_BODY)
        assert expected > 0
        assert prompt_tokens("anthropic", ANTHROPIC_BODY) == expected
        # The estimator is char/4 over every (stripped) text turn, system included.
        payload = json.loads(ANTHROPIC_BODY)
        chars = len(payload["system"].strip()) + len(payload["messages"][0]["content"].strip())
        chars += len(payload["messages"][1]["content"][0]["text"].strip())
        chars += len(payload["messages"][2]["content"][0]["text"].strip())
        assert expected == chars // 4

    def test_responses_body_uses_the_capture_estimator(self):
        expected = _expected("responses", RESPONSES_BODY)
        assert expected > 0
        assert prompt_tokens("responses", RESPONSES_BODY) == expected
        payload = json.loads(RESPONSES_BODY)
        chars = len(payload["instructions"]) + sum(
            len(item["content"][0]["text"]) for item in payload["input"]
        )
        assert expected == chars // 4

    def test_unreadable_body_is_unknown_not_zero(self):
        assert prompt_tokens("anthropic", b"not json") is None
        assert prompt_tokens("anthropic", b"[1,2]") is None
        receipt = receipt_for_observe("anthropic", b"not json")
        assert receipt.static_tokens is None and receipt.served_tokens is None
        assert receipt.saved_tokens == 0

    def test_observe_receipt_serves_exactly_what_was_sent(self):
        receipt = receipt_for_observe("anthropic", ANTHROPIC_BODY)
        assert receipt.mode == "observe" and receipt.format == "anthropic"
        assert receipt.rewritten is False
        assert receipt.static_tokens == receipt.served_tokens == _expected("anthropic", ANTHROPIC_BODY)
        assert receipt.usage_input_tokens is None
        assert receipt.saved_tokens == 0


class TestUsage:
    def test_anthropic_usage_counts_cached_prompt_tokens(self):
        usage = {"input_tokens": 3, "cache_creation_input_tokens": 40, "cache_read_input_tokens": 1200}
        assert usage_input_tokens("anthropic", usage) == 1243
        assert usage_input_tokens("anthropic", {"input_tokens": 50, "output_tokens": 9}) == 50
        assert usage_input_tokens("anthropic", {"output_tokens": 9}) is None

    def test_openai_usage_keys(self):
        assert usage_input_tokens("chat", {"prompt_tokens": 321}) == 321
        assert usage_input_tokens("responses", {"input_tokens": 512}) == 512
        assert usage_input_tokens("chat", {"input_tokens": 7}) == 7
        assert usage_input_tokens("chat", {"prompt_tokens": True}) is None
        assert usage_input_tokens("chat", None) is None

    @pytest.mark.parametrize(
        ("fmt", "stream", "expected"),
        [
            ("anthropic", ANTHROPIC_SSE_WITH_USAGE, 1243),
            ("responses", RESPONSES_SSE_WITH_USAGE, 512),
            ("chat", CHAT_SSE_WITH_USAGE, 321),
        ],
    )
    def test_stream_usage_survives_chunk_boundaries(self, fmt, stream, expected):
        accumulator = UsageAccumulator(fmt=fmt)
        for index in range(0, len(stream), 7):
            accumulator.feed(stream[index : index + 7])
        accumulator.close()
        assert accumulator.input_tokens == expected
        assert usage_from_response(fmt, stream) == expected

    def test_anthropic_stream_merges_message_delta_usage(self):
        accumulator = UsageAccumulator(fmt="anthropic")
        accumulator.feed(ANTHROPIC_SSE_WITH_USAGE)
        accumulator.close()
        assert accumulator.usage["output_tokens"] == 7
        assert accumulator.usage["input_tokens"] == 3

    def test_non_streaming_bodies_carry_usage(self):
        anthropic = json.dumps({"content": [], "usage": {"input_tokens": 20, "cache_read_input_tokens": 5}}).encode()
        assert usage_from_response("anthropic", anthropic) == 25
        chat = json.dumps({"choices": [], "usage": {"prompt_tokens": 11}}).encode()
        assert usage_from_response("chat", chat) == 11
        responses = json.dumps({"output": [], "usage": {"input_tokens": 13}}).encode()
        assert usage_from_response("responses", responses) == 13
        assert usage_from_response("anthropic", b"") is None
        assert usage_from_response("anthropic", b"garbage") is None
        assert usage_from_response("anthropic", b'{"content": []}') is None

    def test_usage_is_preferred_over_the_estimate(self):
        estimate = _expected("anthropic", ANTHROPIC_BODY)
        receipt = receipt_for_observe("anthropic", ANTHROPIC_BODY, usage=1243)
        assert receipt.served_tokens == estimate
        assert receipt.usage_input_tokens == 1243
        assert receipt.effective_served_tokens == 1243

        from_mapping = receipt_for_observe(
            "anthropic", ANTHROPIC_BODY, usage={"input_tokens": 3, "cache_read_input_tokens": 1240}
        )
        assert from_mapping.usage_input_tokens == 1243

        without = receipt_for_observe("anthropic", ANTHROPIC_BODY, usage=None)
        assert without.effective_served_tokens == estimate


class TestSavedTokens:
    def test_augment_receipt_measures_the_rewrite(self):
        payload = json.loads(ANTHROPIC_BODY)
        served_payload = dict(payload, messages=payload["messages"][-1:])
        served = json.dumps(served_payload, separators=(",", ":")).encode()

        receipt = receipt_for_augment("anthropic", ANTHROPIC_BODY, served, reason="recall")
        assert receipt.mode == "augment" and receipt.rewritten is True
        assert receipt.reason == "recall"
        assert receipt.static_tokens == _expected("anthropic", ANTHROPIC_BODY)
        assert receipt.served_tokens == _expected("anthropic", served)
        assert receipt.static_tokens > receipt.served_tokens
        assert receipt.saved_tokens == receipt.static_tokens - receipt.served_tokens

    def test_saved_prefers_usage_and_never_goes_negative(self):
        served = json.dumps({"messages": [{"role": "user", "content": "latest question"}]}).encode()
        receipt = receipt_for_augment("anthropic", ANTHROPIC_BODY, served, usage=10, reason="recall")
        assert receipt.saved_tokens == receipt.static_tokens - 10

        overcount = receipt_for_augment(
            "anthropic", ANTHROPIC_BODY, served, usage=receipt.static_tokens + 500, reason="recall"
        )
        assert overcount.saved_tokens == 0

    def test_passthrough_reencoding_is_not_a_rewrite(self):
        reencoded = json.dumps(json.loads(ANTHROPIC_BODY), separators=(",", ":")).encode()
        assert reencoded != ANTHROPIC_BODY
        receipt = receipt_for_augment("anthropic", ANTHROPIC_BODY, reencoded, reason="invalid_served_body")
        assert receipt.rewritten is False
        assert receipt.static_tokens == receipt.served_tokens
        assert receipt.saved_tokens == 0

    def test_unserved_request_has_unknown_served_side(self):
        receipt = receipt_for_augment("anthropic", ANTHROPIC_BODY, None, reason="rejected")
        assert receipt.rewritten is False
        assert receipt.static_tokens is not None and receipt.served_tokens is None
        assert receipt.saved_tokens == 0

    def test_unknown_sides_yield_zero_savings(self):
        assert ServingReceipt("anthropic", "observe", False, "", None, 5, None).saved_tokens == 0
        assert ServingReceipt("anthropic", "observe", False, "", 5, None, None).saved_tokens == 0
        assert ServingReceipt("anthropic", "augment", True, "", 50, 20, None).saved_tokens == 30
        assert ServingReceipt("anthropic", "augment", True, "", 50, 20, 45).saved_tokens == 5

    def test_as_dict_is_ledger_shaped(self):
        receipt = ServingReceipt("anthropic", "augment", True, "elided 2 units", 50, 20, None)
        assert receipt.as_dict() == {
            "format": "anthropic", "mode": "augment", "rewritten": True,
            "reason": "elided 2 units", "static_tokens": 50, "served_tokens": 20,
            "usage_input_tokens": None, "saved_tokens": 30,
        }


class TestPublish:
    def test_publish_drives_record_serving_with_the_effective_served_count(self):
        publisher = _FakePublisher()
        estimate = _expected("anthropic", ANTHROPIC_BODY)

        publish(receipt_for_observe("anthropic", ANTHROPIC_BODY), publisher)
        publish(receipt_for_observe("anthropic", ANTHROPIC_BODY, usage=1243), publisher)
        served = json.dumps({"messages": [{"role": "user", "content": "latest question"}]}).encode()
        publish(receipt_for_augment("anthropic", ANTHROPIC_BODY, served, reason="recall"), publisher)
        publish(receipt_for_augment("anthropic", b"nope", None, reason="rejected"), publisher)

        assert publisher.calls == [
            (estimate, estimate, False),
            (estimate, 1243, False),
            (estimate, _expected("anthropic", served), True),
            (None, None, False),
        ]

    def test_publish_matches_the_real_publisher_arithmetic(self):
        from memory_condense.control.publisher import StatusPublisher

        class _Counters:
            in_flight = depth = retained_bytes = rejected_saturated = 0
            capture_terminal_failures = ingested = 0
            state = "running"

        publisher = StatusPublisher(
            mode="augment", captures=_Counters(), capacity=8, health=lambda: ("ok", [])
        )
        served = json.dumps({"messages": [{"role": "user", "content": "latest question"}]}).encode()
        receipt = receipt_for_augment("anthropic", ANTHROPIC_BODY, served, usage=12, reason="recall")
        publish(receipt, publisher)
        serving = publisher.document()["serving"]
        assert serving["requests"] == 1 and serving["rewritten"] == 1
        assert serving["static_tokens_last"] == receipt.static_tokens
        assert serving["served_tokens_last"] == 12
        assert serving["saved_tokens_total"] == receipt.saved_tokens
