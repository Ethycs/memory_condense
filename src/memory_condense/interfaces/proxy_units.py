"""Group a model request's conversation into atomic units and validate pairing.

Every wire format the proxy serves carries structure that must never be split:

* Anthropic Messages: a ``tool_result`` block (in a ``user`` message) answers a
  ``tool_use`` block in the preceding ``assistant`` message; ``thinking`` blocks
  are signed to the assistant turn they came from.
* OpenAI chat completions: a ``role: "tool"`` message answers one entry of the
  preceding assistant message's ``tool_calls`` by ``tool_call_id``.
* Responses (Codex): ``function_call_output`` answers ``function_call`` by
  ``call_id``; a ``reasoning`` item belongs with the item that follows it.

A :class:`Unit` is the smallest run of messages/items that keeps every such
pair together. Policies decide which units to keep; they never edit inside one.
:func:`check` reports pairs the API would reject, and :func:`guard_served_body`
is the pre-send guard the augment path calls on the body it is about to send.

Format names match :mod:`memory_condense.interfaces.proxy_capture`
(``format_for_path``): ``"anthropic"``, ``"chat"``, ``"responses"``. The
provider name ``"openai"`` is accepted as an alias for ``"chat"``, its default
format (``proxy_memory_wire.format_for``).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

ANTHROPIC = "anthropic"
CHAT = "chat"
RESPONSES = "responses"
OPENAI = "openai"

Format = Literal["anthropic", "chat", "responses"]

_ALIASES = {OPENAI: CHAT}

#: Top-level request key holding the conversation, per format.
CONVERSATION_KEY = {ANTHROPIC: "messages", CHAT: "messages", RESPONSES: "input"}


def _canonical(fmt: Any) -> str | None:
    if not isinstance(fmt, str):
        return None
    fmt = _ALIASES.get(fmt, fmt)
    return fmt if fmt in CONVERSATION_KEY else None


def estimate_tokens(obj: Any) -> int:
    """Cheap, tokenizer-free estimate (~4 chars/token on JSON)."""
    return max(1, len(json.dumps(obj, ensure_ascii=False)) // 4)


@dataclass
class Unit:
    kind: str  # "user" | "context" | "assistant" | "exchange" | "other"
    items: list[Any] = field(default_factory=list)
    user_text: str = ""
    tool_names: list[str] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.items)


# -- Anthropic Messages ----------------------------------------------------------

def _blocks(msg: dict) -> list[dict]:
    content = msg.get("content")
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return []


def _has_block(msg: dict, kind: str) -> bool:
    return any(b.get("type") == kind for b in _blocks(msg))


def _text_of(msg: dict) -> str:
    return "\n".join(b.get("text", "") for b in _blocks(msg) if b.get("type") == "text").strip()


def group_anthropic(messages: list[dict]) -> list[Unit]:
    units: list[Unit] = []
    for msg in messages:
        role = msg.get("role")
        if (
            role == "user"
            and _has_block(msg, "tool_result")
            and units
            and units[-1].items
            and units[-1].items[-1].get("role") == "assistant"
        ):
            units[-1].items.append(msg)
            units[-1].kind = "exchange"
            continue
        unit = Unit(kind="user" if role == "user" else "assistant", items=[msg])
        if role == "user":
            unit.user_text = _text_of(msg)
        else:
            unit.tool_names = [b.get("name", "?") for b in _blocks(msg) if b.get("type") == "tool_use"]
        units.append(unit)
    return units


def check_anthropic(messages: list[dict]) -> list[str]:
    """Structural problems the API would reject. Empty list means valid."""
    errors: list[str] = []
    if not messages:
        return ["no messages"]
    if messages[0].get("role") != "user":
        errors.append("first message is not from user")
    open_ids: set[str] = set()
    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role == "assistant":
            open_ids = {b.get("id") for b in _blocks(msg) if b.get("type") == "tool_use"}
        elif role == "user":
            for b in _blocks(msg):
                if b.get("type") == "tool_result":
                    if b.get("tool_use_id") not in open_ids:
                        errors.append(f"message {i}: tool_result {b.get('tool_use_id')} has no preceding tool_use")
            open_ids = set()
    return errors


# -- OpenAI chat completions -------------------------------------------------------

_CHAT_CONTEXT_ROLES = ("system", "developer")


def _chat_text(msg: dict) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"
        ).strip()
    return ""


def _tool_calls(msg: dict) -> list[dict]:
    calls = msg.get("tool_calls")
    return [c for c in calls if isinstance(c, dict)] if isinstance(calls, list) else []


def _tool_call_name(call: dict) -> str:
    fn = call.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        return fn["name"]
    return call.get("name") or call.get("type") or "?"


def group_chat(messages: list[dict]) -> list[Unit]:
    units: list[Unit] = []
    open_ids: set[str] = set()
    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            if units and units[-1].kind == "exchange" and msg.get("tool_call_id") in open_ids:
                units[-1].items.append(msg)
                continue
            units.append(Unit(kind="other", items=[msg]))
            open_ids = set()
            continue
        if role == "assistant":
            calls = _tool_calls(msg)
            unit = Unit(kind="exchange" if calls else "assistant", items=[msg],
                        tool_names=[_tool_call_name(c) for c in calls])
            open_ids = {c.get("id") for c in calls}
            units.append(unit)
            continue
        open_ids = set()
        if role in _CHAT_CONTEXT_ROLES:
            units.append(Unit(kind="context", items=[msg]))
        elif role == "user":
            units.append(Unit(kind="user", items=[msg], user_text=_chat_text(msg)))
        else:
            units.append(Unit(kind="other", items=[msg]))
    return units


def check_chat(messages: list[dict]) -> list[str]:
    """Structural problems the chat completions API would reject."""
    errors: list[str] = []
    if not messages:
        return ["no messages"]
    first = messages[0].get("role")
    if first not in _CHAT_CONTEXT_ROLES and first != "user":
        errors.append(f"first message is {first!r}, not system/developer or user")
    open_ids: set[str] = set()
    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role == "assistant":
            open_ids = {c.get("id") for c in _tool_calls(msg)}
        elif role == "tool":
            if msg.get("tool_call_id") not in open_ids:
                errors.append(f"message {i}: tool message {msg.get('tool_call_id')} has no preceding tool_calls entry")
        else:
            open_ids = set()
    return errors


# -- Responses (Codex) -------------------------------------------------------------

_CALL_TYPES = {"function_call", "custom_tool_call", "local_shell_call", "computer_call", "shell_call"}


def _item_type(item: dict) -> str:
    t = item.get("type")
    if t:
        return t
    return "message" if "role" in item else "other"


def _input_text(item: dict) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") in ("input_text", "output_text", "text")
        ).strip()
    return ""


def group_responses(items: list[dict]) -> list[Unit]:
    units: list[Unit] = []
    pending: list[dict] = []  # reasoning items waiting for the item they precede
    by_call: dict[str, Unit] = {}
    for item in items:
        t = _item_type(item)
        if t == "reasoning":
            pending.append(item)
            continue
        if t in _CALL_TYPES:
            unit = Unit(kind="exchange", items=pending + [item], tool_names=[item.get("name") or t])
            pending = []
            units.append(unit)
            if item.get("call_id"):
                by_call[item["call_id"]] = unit
            continue
        if t.endswith("_output"):
            target = by_call.get(item.get("call_id", "")) or (units[-1] if units else None)
            if target is None:
                target = Unit(kind="exchange")
                units.append(target)
            target.items.append(item)
            continue
        if t == "message":
            role = item.get("role")
            if role == "assistant":
                units.append(Unit(kind="assistant", items=pending + [item]))
                pending = []
            else:
                units.append(Unit(kind="user" if role == "user" else "context", items=[item], user_text=_input_text(item)))
            continue
        units.append(Unit(kind="other", items=pending + [item]))
        pending = []
    if pending:
        if units:
            units[-1].items.extend(pending)
        else:
            units.append(Unit(kind="other", items=pending))
    return units


def check_responses(items: list[dict]) -> list[str]:
    errors: list[str] = []
    calls: set[str] = set()
    for i, item in enumerate(items):
        t = _item_type(item)
        if t in _CALL_TYPES and item.get("call_id"):
            calls.add(item["call_id"])
        elif t.endswith("_output"):
            if item.get("call_id") not in calls:
                errors.append(f"item {i}: {t} {item.get('call_id')} has no preceding call")
    return errors


# -- shared -------------------------------------------------------------------------

_GROUPERS = {ANTHROPIC: group_anthropic, CHAT: group_chat, RESPONSES: group_responses}
_CHECKERS = {ANTHROPIC: check_anthropic, CHAT: check_chat, RESPONSES: check_responses}


def group(fmt: Format, conversation: list[dict]) -> list[Unit]:
    name = _canonical(fmt)
    if name is None:
        raise ValueError(f"unknown wire format: {fmt!r}")
    return _GROUPERS[name](conversation)


def check(fmt: Format, conversation: list[dict]) -> list[str]:
    name = _canonical(fmt)
    if name is None:
        raise ValueError(f"unknown wire format: {fmt!r}")
    return _CHECKERS[name](conversation)


def guard_served_body(fmt: Any, body: Any) -> list[str]:
    """Problems in the conversation a request body is about to carry upstream.

    Returns :func:`check` problems for the body's conversation key (``messages``
    for anthropic and chat, ``input`` for responses). Returns ``[]`` for an
    unknown format, a non-dict body, or a body without a list under that key:
    the guard never blocks a body it does not understand.
    """
    name = _canonical(fmt)
    if name is None or not isinstance(body, dict):
        return []
    conversation = body.get(CONVERSATION_KEY[name])
    if not isinstance(conversation, list):
        return []
    return _CHECKERS[name](conversation)


def head_length(units: list[Unit]) -> int:
    """Leading run of user/context units: the task framing and injected context."""
    n = 0
    for u in units:
        if u.kind in ("user", "context"):
            n += 1
        else:
            break
    return max(1, min(n, len(units)))


def tool_histogram(units: list[Unit]) -> Counter:
    return Counter(name for u in units for name in u.tool_names)


__all__ = [
    "ANTHROPIC",
    "CHAT",
    "CONVERSATION_KEY",
    "Format",
    "OPENAI",
    "RESPONSES",
    "Unit",
    "check",
    "check_anthropic",
    "check_chat",
    "check_responses",
    "estimate_tokens",
    "group",
    "group_anthropic",
    "group_chat",
    "group_responses",
    "guard_served_body",
    "head_length",
    "tool_histogram",
]
