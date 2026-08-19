"""Optional Claude Code `UserPromptSubmit` hook: inject ambient memory context.

This is the passive half of the integration. The MCP server gives the model
tools it chooses to call; this hook pushes a small, always-on reminder of the
pinned and still-hot facts before every prompt.

**Why it does no semantic search.** A hook runs as a fresh process on every
prompt. Loading bge-m3 each time would cost seconds per turn and make the
terminal unusable, so this deliberately ranks by pin state and decayed energy
only — pure SQLite, no model, no embedding, typically single-digit
milliseconds. Query-specific recall is the MCP `recall` tool's job.

**It fails open.** Any error at all exits 0 with no output, so a broken or
missing memory store can never block the user's prompt.

Enable it by adding to `.claude/settings.json` (see
`docs/02 - Implementation/02 - MCP Integration.md`):

    {
      "hooks": {
        "UserPromptSubmit": [
          {
            "matcher": "",
            "hooks": [
              {
                "type": "command",
                "command": "pixi run python examples/claude_hooks/memory_context_hook.py"
              }
            ]
          }
        ]
      }
    }
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

#: How many facts to inject. Keep this small — it is prepended to every turn.
MAX_ITEMS = 6


def _data_dir() -> Path:
    explicit = os.environ.get("MEMORY_CONDENSE_DATA_DIR")
    if explicit:
        return Path(explicit)
    project = os.environ.get("CLAUDE_PROJECT_DIR")
    base = Path(project) if project else Path.cwd()
    return base / ".memory_condense"


def _collect() -> list[str]:
    """Pinned first, then hottest. No embedding, no reheating."""
    from memory_condense.persistence.db import Database
    from memory_condense.domain.decay import item_energy
    from memory_condense.persistence.memory_store import MemoryStore

    db_path = _data_dir() / "memory.db"
    if not db_path.exists():
        return []

    with Database(db_path) as db:
        items = MemoryStore(db).list_items()

    # Rank without touching anything: the hook is a passive reader, so it must
    # not reheat items and quietly defeat the decay model.
    ranked = sorted(
        items,
        key=lambda i: (i.is_pinned, item_energy(i)),
        reverse=True,
    )[:MAX_ITEMS]

    lines = []
    for item in ranked:
        marker = "*" if item.is_pinned else "-"
        lines.append(f"{marker} [{item.type.value}] {item.content}")
    return lines


def main() -> int:
    try:
        json.load(sys.stdin)  # consume the hook payload; the prompt is unused
    except Exception:
        pass

    try:
        lines = _collect()
    except Exception:
        return 0  # fail open — never block a prompt over memory

    if not lines:
        return 0

    context = (
        "Project memory (pinned marked *; from memory_condense). "
        "Use the memory_condense MCP tools to search or update it.\n"
        + "\n".join(lines)
    )
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": context,
            }
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
