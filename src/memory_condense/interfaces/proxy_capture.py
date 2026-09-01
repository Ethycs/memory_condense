"""Turn extraction from live provider traffic.

Everything here is pure: bytes in, turns out.  The socket work lives in
:mod:`memory_condense.interfaces.proxy_server`, which hands this module the
request body it is about to forward and the response bytes it is streaming
back.  Splitting them keeps wire-format handling testable without binding a
port, and keeps the proxy's hot path free of retrieval imports.

Both vendors accept the same request shape for our purposes — a ``messages``
array of ``{role, content}`` with content either a string or typed blocks — so
:func:`request_messages` delegates to the transcript parser already used for
account exports.  Responses differ and are handled per provider.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from memory_condense.ingest.transcripts import (
    TranscriptMessage,
    parse_anthropic_messages,
)


ANTHROPIC = "anthropic"
OPENAI = "openai"

#: Response paths worth intercepting.  Anything else is proxied untouched.
CAPTURED_PATHS = {
    "/v1/messages": ANTHROPIC,
    "/v1/chat/completions": OPENAI,
}


def provider_for_path(path: str) -> str | None:
    """Return the provider whose wire format ``path`` uses, if captured."""
    return CAPTURED_PATHS.get(path.rstrip("/") or "/")


def request_messages(
    body: bytes,
    *,
    conversation_id: str,
) -> list[TranscriptMessage]:
    """Extract the prompt turns from a request body.

    Returns an empty list rather than raising: a proxy must forward traffic it
    cannot understand, and capture is best-effort by design.
    """
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    return parse_anthropic_messages(payload, fallback_id=conversation_id)


def request_model(body: bytes) -> str | None:
    """Read the requested model name, for receipts and routing."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    model = payload.get("model") if isinstance(payload, dict) else None
    return model if isinstance(model, str) and model.strip() else None


def is_streaming_request(body: bytes) -> bool:
    """Report whether the client asked for an incremental response."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return bool(isinstance(payload, dict) and payload.get("stream"))


def _blocks_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for block in value:
        if isinstance(block, dict) and block.get("type") in (None, "text"):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def response_text(provider: str, body: bytes) -> str:
    """Extract assistant prose from a complete (non-streaming) response."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    if provider == ANTHROPIC:
        return _blocks_text(payload.get("content")).strip()
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict):
        return ""
    return _blocks_text(message.get("content")).strip()


@dataclass
class StreamAccumulator:
    """Reassemble assistant text from a server-sent-event stream.

    The proxy forwards every byte untouched and feeds a copy here.  Parsing is
    deliberately forgiving: a malformed or truncated stream yields whatever
    text arrived rather than raising into the response path.
    """

    provider: str
    _buffer: bytes = b""
    _parts: list[str] = field(default_factory=list)
    _stop_reason: str | None = None

    def feed(self, chunk: bytes) -> None:
        """Consume one forwarded chunk; safe to call with partial frames."""
        if not chunk:
            return
        self._buffer += chunk
        while b"\n" in self._buffer:
            line, _, rest = self._buffer.partition(b"\n")
            self._buffer = rest
            self._consume_line(line.strip())

    def close(self) -> None:
        """Flush a trailing line that arrived without a newline."""
        if self._buffer.strip():
            self._consume_line(self._buffer.strip())
        self._buffer = b""

    @property
    def text(self) -> str:
        return "".join(self._parts).strip()

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

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
        if self.provider == ANTHROPIC:
            self._consume_anthropic(event)
        else:
            self._consume_openai(event)

    def _consume_anthropic(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "content_block_delta":
            delta = event.get("delta")
            if isinstance(delta, dict) and delta.get("type") == "text_delta":
                text = delta.get("text")
                if isinstance(text, str):
                    self._parts.append(text)
        elif kind == "content_block_start":
            block = event.get("content_block")
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text:
                    self._parts.append(text)
        elif kind == "message_delta":
            delta = event.get("delta")
            if isinstance(delta, dict):
                reason = delta.get("stop_reason")
                if isinstance(reason, str):
                    self._stop_reason = reason

    def _consume_openai(self, event: dict[str, Any]) -> None:
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            return
        first = choices[0]
        if not isinstance(first, dict):
            return
        delta = first.get("delta")
        if isinstance(delta, dict):
            text = delta.get("content")
            if isinstance(text, str):
                self._parts.append(text)
        reason = first.get("finish_reason")
        if isinstance(reason, str):
            self._stop_reason = reason


@dataclass(frozen=True, slots=True)
class ExchangeCapture:
    """One proxied request/response pair, ready for ingest."""

    provider: str
    conversation_id: str
    model: str | None
    prompt: tuple[TranscriptMessage, ...]
    reply_text: str
    streamed: bool
    request_sha256: str
    prompt_tokens_estimate: int

    def ingest_records(
        self,
    ) -> list[tuple[str, str, str, Any, str]]:
        """Return ``ingest_many`` records for the reply and any new prompt turns.

        The final user turn and the assistant reply are what a live exchange
        adds; earlier prompt turns are already-known history that the client
        resent, so re-ingesting them would duplicate the conversation.
        """
        records: list[tuple[str, str, str, Any, str]] = []
        last_user = next(
            (m for m in reversed(self.prompt) if m.role == "user"),
            None,
        )
        if last_user is not None:
            records.append(last_user.as_ingest_record())
        if self.reply_text:
            records.append(
                (
                    "assistant",
                    self.reply_text,
                    self.conversation_id,
                    None,
                    f"{self.conversation_id}:reply:{self.request_sha256[:16]}",
                )
            )
        return records


def estimate_prompt_tokens(messages: Iterable[TranscriptMessage]) -> int:
    """Cheap character-based proxy; the real counter needs a tokenizer."""
    return sum(len(message.text) for message in messages) // 4


__all__ = [
    "ANTHROPIC",
    "CAPTURED_PATHS",
    "OPENAI",
    "ExchangeCapture",
    "StreamAccumulator",
    "estimate_prompt_tokens",
    "is_streaming_request",
    "provider_for_path",
    "request_messages",
    "request_model",
    "response_text",
]
