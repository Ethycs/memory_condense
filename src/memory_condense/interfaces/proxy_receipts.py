"""Per-request serving receipts: what the client sent vs. what went upstream.

One receipt per inference request, in both proxy modes, is what feeds the
``serving`` section of the status document (``control/status.py``).  The
two estimates share one estimator — :func:`estimate_prompt_tokens` from
``proxy_capture`` — so ``static_tokens`` and ``served_tokens`` are always
comparable.  When the upstream reports its own ``usage`` the exact input
count is preferred for the served side; the estimate is the fallback.

Nothing here touches a socket.  The proxy hands this module request bytes
and, when it has them, the upstream's usage; the module hands back numbers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from memory_condense.interfaces.proxy_capture import (
    estimate_prompt_tokens,
    request_messages,
)

#: ``request_messages`` wants a conversation id for message ids; receipts
#: never keep the messages, so any constant will do.
_RECEIPT_CONVERSATION = "receipt"


@dataclass(frozen=True, slots=True)
class ServingReceipt:
    """What one request cost the client's prompt and what went upstream.

    ``static_tokens`` estimates the prompt the client sent; ``served_tokens``
    estimates the prompt the proxy forwarded (equal in observe mode).  Both
    use the same character-based estimator.  ``usage_input_tokens`` is the
    upstream's own count of the forwarded prompt when the response carried
    one; it is exact and wins over ``served_tokens`` wherever a served figure
    is published.  ``None`` means "unknown", never zero.
    """

    format: str | None
    mode: str
    rewritten: bool
    reason: str
    static_tokens: int | None
    served_tokens: int | None
    usage_input_tokens: int | None

    @property
    def effective_served_tokens(self) -> int | None:
        """The served figure to publish: exact usage when known, else estimate."""
        if self.usage_input_tokens is not None:
            return self.usage_input_tokens
        return self.served_tokens

    @property
    def saved_tokens(self) -> int:
        """``static - served`` when both are known, else 0; never negative.

        Uses :attr:`effective_served_tokens` so the figure matches what the
        publisher accumulates into ``saved_tokens_total``.  Clamped because a
        char/4 estimate can undercount what the upstream then reports.
        """
        served = self.effective_served_tokens
        if self.static_tokens is None or served is None:
            return 0
        return max(0, int(self.static_tokens) - int(served))

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["saved_tokens"] = self.saved_tokens
        return value


# -- estimates ---------------------------------------------------------------

def _payload(body: bytes | None) -> dict[str, Any] | None:
    if body is None:
        return None
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        return None
    return value if isinstance(value, dict) else None


def prompt_tokens(fmt: str | None, body: bytes | None) -> int | None:
    """Estimate the prompt tokens in ``body`` with the capture path's estimator.

    ``None`` when ``body`` is not a JSON object: a receipt must never guess
    at a prompt it cannot read.
    """
    if _payload(body) is None:
        return None
    assert body is not None
    return estimate_prompt_tokens(
        request_messages(body, conversation_id=_RECEIPT_CONVERSATION, fmt=fmt)
    )


def _same_payload(original: bytes, served: bytes) -> bool:
    """Whether two bodies carry the same JSON, ignoring encoding differences."""
    left, right = _payload(original), _payload(served)
    if left is None or right is None:
        return original == served
    return left == right


# -- usage -------------------------------------------------------------------

def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def usage_input_tokens(fmt: str | None, usage: Mapping[str, Any] | None) -> int | None:
    """The upstream's count of the prompt it processed, from a ``usage`` object.

    Anthropic splits the prompt across ``input_tokens`` (uncached) and the
    ``cache_creation_input_tokens``/``cache_read_input_tokens`` pair; the
    served prompt is their sum, which is also what OpenAI's ``prompt_tokens``
    (chat) and ``input_tokens`` (Responses) already report as one number.
    """
    if not isinstance(usage, Mapping):
        return None
    if fmt == "anthropic":
        base = _int_or_none(usage.get("input_tokens"))
        if base is None:
            return None
        cached = sum(
            _int_or_none(usage.get(key)) or 0
            for key in ("cache_creation_input_tokens", "cache_read_input_tokens")
        )
        return base + cached
    if fmt == "responses":
        return _int_or_none(usage.get("input_tokens"))
    value = _int_or_none(usage.get("prompt_tokens"))
    return value if value is not None else _int_or_none(usage.get("input_tokens"))


@dataclass
class UsageAccumulator:
    """Collect the upstream ``usage`` object from a server-sent-event stream.

    Framing mirrors :class:`proxy_capture.StreamAccumulator` (``data:`` lines,
    partial frames tolerated) but only the usage objects are kept, so the
    proxy can tee response bytes into both without re-parsing.  Forgiving by
    design: a malformed stream yields ``None``, never an exception.
    """

    fmt: str
    _buffer: bytes = b""
    _usage: dict[str, Any] | None = None

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._buffer += chunk
        while b"\n" in self._buffer:
            line, _, rest = self._buffer.partition(b"\n")
            self._buffer = rest
            self._consume_line(line.strip())

    def close(self) -> None:
        if self._buffer.strip():
            self._consume_line(self._buffer.strip())
        self._buffer = b""

    @property
    def usage(self) -> dict[str, Any] | None:
        return None if self._usage is None else dict(self._usage)

    @property
    def input_tokens(self) -> int | None:
        return usage_input_tokens(self.fmt, self._usage)

    def _merge(self, usage: Any) -> None:
        if isinstance(usage, dict):
            if self._usage is None:
                self._usage = {}
            self._usage.update(usage)

    def _consume_line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        payload = line[len(b"data:") :].strip()
        if not payload or payload == b"[DONE]":
            return
        try:
            event = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(event, dict):
            return
        if self.fmt == "anthropic":
            kind = event.get("type")
            if kind == "message_start":
                message = event.get("message")
                if isinstance(message, dict):
                    self._merge(message.get("usage"))
            elif kind == "message_delta":
                self._merge(event.get("usage"))
        elif self.fmt == "responses":
            if event.get("type") in (
                "response.completed",
                "response.incomplete",
                "response.failed",
            ):
                response = event.get("response")
                if isinstance(response, dict):
                    self._merge(response.get("usage"))
        else:
            self._merge(event.get("usage"))


def usage_from_response(fmt: str | None, body: bytes | None) -> int | None:
    """Input tokens from a complete response body, JSON or buffered SSE."""
    if not body:
        return None
    payload = _payload(body)
    if payload is not None:
        return usage_input_tokens(fmt, payload.get("usage"))
    accumulator = UsageAccumulator(fmt=fmt or "")
    accumulator.feed(body)
    accumulator.close()
    return accumulator.input_tokens


def _usage_count(fmt: str | None, usage: Any) -> int | None:
    """Accept either an already-extracted count or a raw ``usage`` mapping."""
    if isinstance(usage, Mapping):
        return usage_input_tokens(fmt, usage)
    return _int_or_none(usage)


# -- builders ----------------------------------------------------------------

def receipt_for_observe(
    fmt: str | None,
    body_bytes: bytes,
    usage: int | Mapping[str, Any] | None = None,
) -> ServingReceipt:
    """Observe mode: the request went upstream verbatim, so served == static."""
    estimate = prompt_tokens(fmt, body_bytes)
    return ServingReceipt(
        format=fmt,
        mode="observe",
        rewritten=False,
        reason="forwarded verbatim",
        static_tokens=estimate,
        served_tokens=estimate,
        usage_input_tokens=_usage_count(fmt, usage),
    )


def receipt_for_augment(
    fmt: str | None,
    original_bytes: bytes,
    served_bytes: bytes | None,
    usage: int | Mapping[str, Any] | None = None,
    reason: str = "",
) -> ServingReceipt:
    """Augment mode: the memory service may have replaced the prompt.

    ``served_bytes`` is the body that actually went upstream, or ``None``
    when the request never reached the point of being served (the served
    side is then unknown).  ``rewritten`` is true only when the served JSON
    differs from the original: a passthrough re-encoding is not a rewrite.
    """
    static = prompt_tokens(fmt, original_bytes)
    served = prompt_tokens(fmt, served_bytes) if served_bytes is not None else None
    rewritten = served_bytes is not None and not _same_payload(original_bytes, served_bytes)
    return ServingReceipt(
        format=fmt,
        mode="augment",
        rewritten=rewritten,
        reason=reason,
        static_tokens=static,
        served_tokens=served,
        usage_input_tokens=_usage_count(fmt, usage),
    )


def publish(receipt: ServingReceipt, publisher: Any) -> None:
    """Hand one receipt to ``StatusPublisher.record_serving``.

    Served is the upstream's exact count when the receipt has one, else the
    estimate; the publisher derives ``saved_tokens_total`` from that pair.
    """
    publisher.record_serving(
        receipt.static_tokens,
        receipt.effective_served_tokens,
        receipt.rewritten,
    )


__all__ = [
    "ServingReceipt",
    "UsageAccumulator",
    "prompt_tokens",
    "publish",
    "receipt_for_augment",
    "receipt_for_observe",
    "usage_from_response",
    "usage_input_tokens",
]
