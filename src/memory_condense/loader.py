"""Parse LLM conversation exports from .txt and .md files into Turn objects."""

from __future__ import annotations

import re
from pathlib import Path

from memory_condense.schemas import Turn

# .txt format: "User:\n<text>" / "Claude:\n <text>"
_TXT_TURN_RE = re.compile(
    r"^(User|Claude):\s*\n(.*?)(?=^(?:User|Claude):\s*\n|\Z)",
    re.MULTILINE | re.DOTALL,
)

# .md format: "**User:**\n<text>" / "**Assistant:**\n<text>"
# Note: colon is inside the bold markers: **User:** not **User**:
_MD_TURN_RE = re.compile(
    r"^\*\*(User|Assistant):\*\*\s*\n(.*?)(?=^\*\*(?:User|Assistant):\*\*\s*\n|\Z)",
    re.MULTILINE | re.DOTALL,
)

_ROLE_MAP = {
    "User": "user",
    "Claude": "assistant",
    "Assistant": "assistant",
}


def parse_txt(text: str) -> list[tuple[str, str]]:
    """Parse a .txt conversation export.

    Returns a list of (role, text) tuples.
    """
    turns: list[tuple[str, str]] = []
    for match in _TXT_TURN_RE.finditer(text):
        raw_role = match.group(1)
        body = match.group(2).strip()
        if body:
            role = _ROLE_MAP.get(raw_role, raw_role.lower())
            turns.append((role, body))
    return turns


def parse_md(text: str) -> list[tuple[str, str]]:
    """Parse a .md conversation export.

    Returns a list of (role, text) tuples.
    """
    turns: list[tuple[str, str]] = []
    for match in _MD_TURN_RE.finditer(text):
        raw_role = match.group(1)
        body = match.group(2).strip()
        if body:
            role = _ROLE_MAP.get(raw_role, raw_role.lower())
            turns.append((role, body))
    return turns


def load_conversation(path: str | Path) -> list[tuple[str, str]]:
    """Load a conversation from a .txt or .md file.

    Auto-detects format based on file extension.
    Returns a list of (role, text) tuples.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    if path.suffix == ".md":
        return parse_md(text)
    else:
        return parse_txt(text)


def load_directory(
    directory: str | Path,
    extensions: tuple[str, ...] = (".txt", ".md"),
) -> dict[str, list[tuple[str, str]]]:
    """Load all conversation files from a directory.

    Returns a dict mapping filename -> list of (role, text) tuples.
    Skips files that yield no turns.
    """
    directory = Path(directory)
    conversations: dict[str, list[tuple[str, str]]] = {}

    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix in extensions:
            turns = load_conversation(path)
            if turns:
                conversations[path.name] = turns

    return conversations
