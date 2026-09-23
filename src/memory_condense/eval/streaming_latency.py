"""Measure visible streaming latency, including an optional live prompt builder.

No dataset, scores, cached completions, retries, or provider construction live
here. A role-only delta or hidden reasoning delta is not a visible first token.
The caller owns cold setup and must describe whether prepare_prompt retrieves
live evidence or merely returns a previously constructed packet.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def measure_streaming_answer(
    *, client: Any, model: str,
    prepare_prompt: Callable[[], Sequence[Mapping[str, str]]],
    max_tokens: int = 256, clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Time a fresh request; require visible text and an explicit finish event.

    Preparation is inside the end-to-end clock. Stream creation is inside the
    API clock (some gateways buffer there). Stream exhaustion is inside total
    time, including a trailing usage event. A truncated answer is recorded with
    finish_reason=length; it is never silently treated as an untruncated answer.
    """
    if not model or type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("model and a positive output cap are required")
    started = clock()
    messages = [dict(message) for message in prepare_prompt()]
    if not messages or any(
        set(m) != {"role", "content"} or m["role"] not in {"system", "user", "assistant"}
        or not isinstance(m["content"], str) for m in messages
    ):
        raise ValueError("expected nonempty plain role/content messages")
    api_started = clock()
    stream = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, stream=True,
        stream_options={"include_usage": True}, timeout=180.0,
    )
    first_event = first_visible = last_visible = None
    text_parts: list[str] = []
    visible_events = events = 0
    finish_reason = None
    usage = None
    response_model = None
    try:
        for event in stream:
            now = clock()
            events += 1
            if first_event is None:
                first_event = now
            response_model = _field(event, "model") or response_model
            reported_usage = _field(event, "usage")
            if reported_usage is not None:
                usage = {k: _field(reported_usage, k) for k in (
                    "prompt_tokens", "completion_tokens", "total_tokens",
                )}
            for choice in _field(event, "choices", []):
                if _field(choice, "index", 0) != 0:
                    raise ValueError("multiple streamed answer choices are unsupported")
                delta = _field(choice, "delta")
                content = _field(delta, "content")
                if content:
                    if not isinstance(content, str) or finish_reason is not None:
                        raise ValueError("invalid content or text after stream finish")
                    if first_visible is None:
                        first_visible = now
                    last_visible = now
                    visible_events += 1
                    text_parts.append(content)
                reason = _field(choice, "finish_reason")
                if reason is not None:
                    if finish_reason is not None:
                        raise ValueError("duplicate stream finish event")
                    finish_reason = reason
        finished = clock()
    finally:
        stream.close()
    if first_visible is None or finish_reason not in {"stop", "length", "content_filter"}:
        raise ValueError("stream has no visible answer or no supported finish event")
    times = [started, api_started, first_event, first_visible, last_visible, finished]
    if any(not isinstance(t, (int, float)) or not math.isfinite(t) for t in times) or times != sorted(times):
        raise ValueError("stream clock must be finite and monotonic")
    prediction = "".join(text_parts)
    return {
        "model": model, "response_model": response_model,
        "max_tokens": max_tokens, "prediction": prediction,
        "prediction_sha256": hashlib.sha256(prediction.encode()).hexdigest(),
        "messages_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True,
            ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
        "prepare_s": api_started - started,
        "first_api_event_s": first_event - api_started,
        "api_ttft_s": first_visible - api_started,
        "e2e_ttft_s": first_visible - started,
        "api_total_s": finished - api_started, "e2e_total_s": finished - started,
        "visible_generation_s": last_visible - first_visible,
        "event_count": events, "visible_event_count": visible_events,
        "finish_reason": finish_reason, "usage": usage,
    }


def latency_distribution(values: Sequence[float]) -> dict[str, float]:
    """Nearest-rank p95; median interpolates the two middle observations."""
    import statistics
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("latencies must be finite nonnegative observations")
    ordered = sorted(values)
    return {"median_s": statistics.median(ordered),
            "p95_s": ordered[math.ceil(.95 * len(ordered)) - 1],
            "mean_s": statistics.fmean(ordered)}
