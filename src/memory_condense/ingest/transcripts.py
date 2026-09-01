"""Vendor chat-transcript formats parsed into ingestable turns.

Covers the JSON shapes people actually have on disk: the ChatGPT account
export (an array of mapping-tree conversations), the Claude account export (an
array of ``chat_messages`` conversations), the Anthropic Messages API request
shape, and JSONL variants of each.

Parsing here is pure: functions take already-decoded JSON and return
:class:`TranscriptMessage` values.  Byte offsets, memory mapping, and change
detection belong to :mod:`memory_condense.ingest.transcript_source`, which
reads the file once and hands slices to these parsers.  Keeping the split
means a caller can re-parse a conversation from a mapped span without holding
the whole export in memory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator, Sequence

from memory_condense.domain._discourse_identity import _nonempty, normalize_fields


CHATGPT_EXPORT = "chatgpt-export"
CHATGPT_CONVERSATION = "chatgpt-json"
CLAUDE_EXPORT = "claude-export"
ANTHROPIC_MESSAGES = "anthropic-messages"

#: Roles the ingest path accepts.  Vendor "human"/"model" spellings are mapped
#: onto these before a message is emitted.
_ROLES = {"user", "assistant", "system"}

_ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "model": "assistant",
    "ai": "assistant",
    "claude": "assistant",
    "chatgpt": "assistant",
    "system": "system",
}


@dataclass(frozen=True, slots=True)
class TranscriptMessage:
    """One conversation turn recovered from a vendor transcript."""

    role: str
    text: str
    conversation_id: str
    message_id: str
    ordinal: int
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        normalize_fields(
            self,
            role=_nonempty,
            conversation_id=_nonempty,
            message_id=_nonempty,
        )
        if self.role not in _ROLES:
            raise ValueError("transcript role must be user, assistant, or system")
        if not self.text.strip():
            raise ValueError("transcript message text must be non-empty")
        if self.ordinal < 0:
            raise ValueError("transcript ordinal must be non-negative")

    def as_ingest_record(self) -> tuple[str, str, str, datetime | None, str]:
        """Return the tuple ``IngestWorkflowMixin.ingest_many`` accepts."""
        return (
            self.role,
            self.text,
            self.conversation_id,
            self.created_at,
            self.message_id,
        )


def normalize_role(raw: Any) -> str | None:
    """Map a vendor role spelling onto an accepted role, or ``None``."""
    if not isinstance(raw, str):
        return None
    return _ROLE_ALIASES.get(raw.strip().lower())


def parse_timestamp(value: Any) -> datetime | None:
    """Read the epoch floats and ISO strings the two exports mix freely."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # Both exports emit RFC 3339; only ChatGPT uses a trailing "Z".
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _joined_text(value: Any) -> str:
    """Flatten the several content shapes both vendors use into plain text.

    Accepts a bare string, a list of strings, or a list of typed content
    blocks.  Non-text blocks (images, tool calls, thinking) carry no ingestable
    prose and are skipped rather than rendered as placeholders.
    """
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            if item.get("type") not in (None, "text"):
                continue
            text = item.get("text") or item.get("content")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(part for part in parts if part).strip()


def _conversation_id(record: Any, fallback: str) -> str:
    for key in ("uuid", "id", "conversation_id"):
        value = record.get(key) if isinstance(record, dict) else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def parse_claude_conversation(
    record: Any,
    *,
    fallback_id: str = "claude-conversation",
) -> list[TranscriptMessage]:
    """Parse one conversation from a Claude account export."""
    if not isinstance(record, dict):
        return []
    messages = record.get("chat_messages")
    if not isinstance(messages, list):
        return []
    conversation_id = _conversation_id(record, fallback_id)
    parsed: list[TranscriptMessage] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = normalize_role(message.get("sender") or message.get("role"))
        if role is None:
            continue
        # Newer exports carry structured `content`; older ones only `text`.
        text = _joined_text(message.get("content")) or _joined_text(
            message.get("text")
        )
        if not text:
            continue
        raw_id = message.get("uuid") or message.get("id")
        message_id = (
            raw_id.strip()
            if isinstance(raw_id, str) and raw_id.strip()
            else f"{conversation_id}:{index}"
        )
        parsed.append(
            TranscriptMessage(
                role=role,
                text=text,
                conversation_id=conversation_id,
                message_id=message_id,
                ordinal=len(parsed),
                created_at=parse_timestamp(message.get("created_at")),
            )
        )
    return parsed


def _chatgpt_active_path(mapping: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return the longest root-to-leaf branch of a ChatGPT mapping tree.

    ChatGPT stores edits and regenerations as siblings, so the file is a tree
    rather than a list.  The longest branch is the conventional reconstruction
    of "the conversation as last seen".
    """
    leaves = [
        node_id
        for node_id, node in mapping.items()
        if isinstance(node, dict) and not node.get("children")
    ]
    best: list[tuple[str, dict[str, Any]]] = []
    for leaf in leaves:
        path: list[tuple[str, dict[str, Any]]] = []
        seen: set[str] = set()
        node_id: str | None = leaf
        while isinstance(node_id, str) and node_id not in seen:
            seen.add(node_id)
            node = mapping.get(node_id)
            if not isinstance(node, dict):
                break
            path.append((node_id, node))
            parent = node.get("parent")
            node_id = parent if isinstance(parent, str) else None
        path.reverse()
        if len(path) > len(best):
            best = path
    return best


def parse_chatgpt_conversation(
    record: Any,
    *,
    fallback_id: str = "chatgpt-conversation",
) -> list[TranscriptMessage]:
    """Parse one mapping-tree conversation from a ChatGPT export."""
    if not isinstance(record, dict) or not isinstance(record.get("mapping"), dict):
        return []
    conversation_id = _conversation_id(record, fallback_id)
    parsed: list[TranscriptMessage] = []
    for node_id, node in _chatgpt_active_path(record["mapping"]):
        message = node.get("message")
        if not isinstance(message, dict):
            continue
        author = message.get("author")
        role = normalize_role(
            author.get("role") if isinstance(author, dict) else None
        )
        if role is None:
            continue
        content = message.get("content")
        text = ""
        if isinstance(content, dict):
            text = _joined_text(content.get("parts"))
        if not text:
            text = _joined_text(content)
        if not text:
            continue
        parsed.append(
            TranscriptMessage(
                role=role,
                text=text,
                conversation_id=conversation_id,
                message_id=str(message.get("id") or node_id),
                ordinal=len(parsed),
                created_at=parse_timestamp(message.get("create_time")),
            )
        )
    return parsed


def parse_anthropic_messages(
    record: Any,
    *,
    fallback_id: str = "anthropic-messages",
) -> list[TranscriptMessage]:
    """Parse an Anthropic Messages API request/response body."""
    if not isinstance(record, dict):
        return []
    messages = record.get("messages")
    if not isinstance(messages, list):
        return []
    conversation_id = _conversation_id(record, fallback_id)
    parsed: list[TranscriptMessage] = []
    system_text = _joined_text(record.get("system"))
    if system_text:
        parsed.append(
            TranscriptMessage(
                role="system",
                text=system_text,
                conversation_id=conversation_id,
                message_id=f"{conversation_id}:system",
                ordinal=0,
            )
        )
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = normalize_role(message.get("role"))
        if role is None:
            continue
        text = _joined_text(message.get("content"))
        if not text:
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


def detect_conversation_format(record: Any) -> str | None:
    """Name the vendor shape of one decoded conversation record."""
    if not isinstance(record, dict):
        return None
    if isinstance(record.get("mapping"), dict):
        return CHATGPT_CONVERSATION
    if isinstance(record.get("chat_messages"), list):
        return CLAUDE_EXPORT
    if isinstance(record.get("messages"), list):
        return ANTHROPIC_MESSAGES
    return None


def parse_conversation_record(
    record: Any,
    *,
    fallback_id: str,
) -> tuple[str, list[TranscriptMessage]]:
    """Dispatch one decoded record onto the parser its shape implies."""
    detected = detect_conversation_format(record)
    if detected == CHATGPT_CONVERSATION:
        return detected, parse_chatgpt_conversation(record, fallback_id=fallback_id)
    if detected == CLAUDE_EXPORT:
        return detected, parse_claude_conversation(record, fallback_id=fallback_id)
    if detected == ANTHROPIC_MESSAGES:
        return detected, parse_anthropic_messages(record, fallback_id=fallback_id)
    return "unknown", []


def iter_conversation_records(data: Any) -> Iterator[tuple[int, Any]]:
    """Yield ``(index, record)`` for a single conversation or an export array."""
    if isinstance(data, list):
        for index, record in enumerate(data):
            yield index, record
        return
    if isinstance(data, dict):
        yield 0, data


def parse_transcript_payload(
    data: Any,
    *,
    source_label: str = "transcript",
) -> tuple[str, list[TranscriptMessage]]:
    """Parse a whole decoded export into one ordered message list.

    Returns the export-level format name and every message across every
    conversation, in file order.  Conversation identity is preserved per
    message, so a multi-conversation export stays separable downstream.
    """
    messages: list[TranscriptMessage] = []
    formats: set[str] = set()
    for index, record in iter_conversation_records(data):
        detected, parsed = parse_conversation_record(
            record,
            fallback_id=f"{source_label}:{index}",
        )
        if parsed:
            formats.add(detected)
            messages.extend(parsed)
    if not messages:
        return "unknown", []
    if formats == {CHATGPT_CONVERSATION} and isinstance(data, list):
        return CHATGPT_EXPORT, messages
    if len(formats) == 1:
        return next(iter(formats)), messages
    return "mixed", messages


def parse_transcript_jsonl(
    lines: Sequence[str],
    *,
    source_label: str = "transcript",
) -> tuple[str, list[TranscriptMessage]]:
    """Parse JSONL where each line is one conversation or one message."""
    messages: list[TranscriptMessage] = []
    formats: set[str] = set()
    loose: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        text = line.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            continue
        detected = detect_conversation_format(record)
        if detected is None:
            # A bare ``{"role": ..., "content": ...}`` message line.
            if isinstance(record, dict) and normalize_role(record.get("role")):
                loose.append(record)
            continue
        _name, parsed = parse_conversation_record(
            record,
            fallback_id=f"{source_label}:{index}",
        )
        if parsed:
            formats.add(detected)
            messages.extend(parsed)
    if loose:
        parsed = parse_anthropic_messages(
            {"messages": loose},
            fallback_id=source_label,
        )
        if parsed:
            formats.add(ANTHROPIC_MESSAGES)
            messages.extend(parsed)
    if not messages:
        return "unknown", []
    if len(formats) == 1:
        return next(iter(formats)), messages
    return "mixed", messages


__all__ = [
    "ANTHROPIC_MESSAGES",
    "CHATGPT_CONVERSATION",
    "CHATGPT_EXPORT",
    "CLAUDE_EXPORT",
    "TranscriptMessage",
    "detect_conversation_format",
    "iter_conversation_records",
    "normalize_role",
    "parse_anthropic_messages",
    "parse_chatgpt_conversation",
    "parse_claude_conversation",
    "parse_conversation_record",
    "parse_timestamp",
    "parse_transcript_jsonl",
    "parse_transcript_payload",
]
