"""Turn extraction from live provider traffic.

Everything here is pure: bytes in, turns out.  The socket work lives in
:mod:`memory_condense.interfaces.proxy_server`, which hands this module the
request body it is about to forward and the response bytes it is streaming
back.  Splitting them keeps wire-format handling testable without binding a
port, and keeps the proxy's hot path free of retrieval imports.

A path selects two independent things: the provider, which binds credentials
and the upstream host, and the wire *format*.  OpenAI speaks two formats, so
the axes do not collapse.  ``chat`` and ``anthropic`` both carry a ``messages``
array of ``{role, content}``, which is why :func:`request_messages` can
delegate both to the transcript parser used for account exports; ``responses``
carries a typed ``input`` item array instead and is parsed here.

Note that "responses" is overloaded.  Where this module says *response* with
no qualifier it means an HTTP response body, which every format has.  The
``responses`` format is the OpenAI Responses API specifically.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from memory_condense.ingest.transcripts import (
    TranscriptMessage,
    normalize_role,
    parse_anthropic_messages,
)


ANTHROPIC = "anthropic"
OPENAI = "openai"

#: Chat paths worth intercepting, as ``(provider, format)``.  Anything else is
#: proxied untouched.
CAPTURED_PATHS = {
    "/v1/messages": (ANTHROPIC, "anthropic"),
    "/v1/chat/completions": (OPENAI, "chat"),
    "/v1/responses": (OPENAI, "responses"),
}


def _captured(path: str) -> tuple[str, str] | None:
    return CAPTURED_PATHS.get(path.rstrip("/") or "/")


def provider_for_path(path: str) -> str | None:
    """Return the provider whose credentials and host ``path`` uses."""
    entry = _captured(path)
    return None if entry is None else entry[0]


def format_for_path(path: str) -> str | None:
    """Return the wire format ``path`` carries, if captured."""
    entry = _captured(path)
    return None if entry is None else entry[1]


def request_messages(
    body: bytes,
    *,
    conversation_id: str,
    fmt: str | None = None,
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
    if fmt == "responses":
        return _responses_messages(payload, conversation_id)
    return parse_anthropic_messages(payload, fallback_id=conversation_id)


def _responses_messages(
    payload: dict[str, Any],
    conversation_id: str,
) -> list[TranscriptMessage]:
    """Parse a Responses request: ``instructions`` plus the typed ``input``."""
    parsed: list[TranscriptMessage] = []
    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        parsed.append(
            TranscriptMessage(
                role="system",
                text=instructions,
                conversation_id=conversation_id,
                message_id=f"{conversation_id}:instructions",
                ordinal=0,
            )
        )
    items = payload.get("input")
    if isinstance(items, str):
        items = [{"role": "user", "content": items}]
    if not isinstance(items, list):
        return parsed
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        role = normalize_role(item.get("role"))
        if role is None:
            continue
        text = _blocks_text(item.get("content"))
        if not text.strip():
            continue
        parsed.append(
            TranscriptMessage(
                role=role,
                text=text,
                conversation_id=conversation_id,
                message_id=f"{conversation_id}:{index}",
                ordinal=len(parsed),
            )
        )
    return parsed


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
        if isinstance(block, dict) and block.get("type") in (
            None,
            "text",
            "input_text",
            "output_text",
        ):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def response_text(provider: str, body: bytes, fmt: str | None = None) -> str:
    """Extract assistant prose from a complete (non-streaming) response."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    if fmt == "responses":
        output = payload.get("output")
        if not isinstance(output, list):
            return ""
        return "".join(
            _blocks_text(item.get("content"))
            for item in output
            if isinstance(item, dict) and item.get("type") in (None, "message")
        ).strip()
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
    fmt: str = ""
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
        kind = self.fmt or ("anthropic" if self.provider == ANTHROPIC else "chat")
        if kind == "anthropic":
            self._consume_anthropic(event)
        elif kind == "responses":
            self._consume_responses(event)
        else:
            self._consume_openai(event)

    def _consume_responses(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                self._parts.append(delta)
        elif kind in (
            "response.completed",
            "response.failed",
            "response.incomplete",
        ):
            response = event.get("response")
            status = response.get("status") if isinstance(response, dict) else None
            self._stop_reason = (
                status if isinstance(status, str) else kind.rpartition(".")[2]
            )

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

    def compact_for_ingest(self) -> ExchangeCapture:
        """Retain only prompt state that :meth:`ingest_records` can emit."""

        last_user = next(
            (message for message in reversed(self.prompt) if message.role == "user"),
            None,
        )
        prompt = (
            ()
            if last_user is None
            else (
                replace(
                    last_user,
                    message_id=(
                        f"{self.conversation_id}:user:"
                        f"{self.request_sha256}"
                    ),
                ),
            )
        )
        if prompt == self.prompt:
            return self
        return replace(self, prompt=prompt)

    @property
    def retained_bytes(self) -> int:
        """Conservative UTF-8 payload bytes held by this capture.

        This intentionally measures variable-size payloads, not interpreter
        object headers. Repeated identifiers are counted repeatedly, so the
        admission budget is deterministic and conservative.
        """

        values = [
            self.provider,
            self.conversation_id,
            self.model or "",
            self.reply_text,
            self.request_sha256,
        ]
        for message in self.prompt:
            values.extend(
                (
                    message.role,
                    message.text,
                    message.conversation_id,
                    message.message_id,
                )
            )
        return sum(len(value.encode("utf-8")) for value in values)

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
            records.append(
                replace(
                    last_user,
                    message_id=(
                        f"{self.conversation_id}:user:"
                        f"{self.request_sha256}"
                    ),
                ).as_ingest_record()
            )
        if self.reply_text:
            reply_sha256 = hashlib.sha256(
                self.reply_text.encode("utf-8")
            ).hexdigest()
            records.append(
                (
                    "assistant",
                    self.reply_text,
                    self.conversation_id,
                    None,
                    (
                        f"{self.conversation_id}:reply:"
                        f"{self.request_sha256}:{reply_sha256}"
                    ),
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
    "format_for_path",
    "is_streaming_request",
    "provider_for_path",
    "request_messages",
    "request_model",
    "response_text",
]
