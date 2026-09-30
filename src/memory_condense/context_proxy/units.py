"""Group a model request's conversation into atomic units that are safe to drop.

Both wire formats carry structure that must never be split:

* Anthropic Messages: a ``tool_result`` block (in a ``user`` message) answers a
  ``tool_use`` block in the preceding ``assistant`` message; ``thinking`` blocks
  are signed to the assistant turn they came from.
* Responses (Codex): ``function_call_output`` answers ``function_call`` by
  ``call_id``; a ``reasoning`` item belongs with the item that follows it.

A :class:`Unit` is the smallest run of messages/items that keeps every such
pair together. Policies decide which units to keep; they never edit inside one.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

Format = Literal["anthropic", "responses"]


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

def group(fmt: Format, conversation: list[dict]) -> list[Unit]:
    return group_anthropic(conversation) if fmt == "anthropic" else group_responses(conversation)


def check(fmt: Format, conversation: list[dict]) -> list[str]:
    return check_anthropic(conversation) if fmt == "anthropic" else check_responses(conversation)


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
