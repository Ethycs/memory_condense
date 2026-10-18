"""Claude Code ``statusLine`` renderer for the memory system.

Claude Code pipes a JSON document on stdin (``cwd``, ``workspace.current_dir``,
``session_id``, ``model.display_name``, ``context_window.used_percentage``,
...; see https://code.claude.com/docs/en/statusline) and shows whatever this
script prints. We resolve ``status.json`` from the session's working directory
(``memory_condense.control.status.locate_status_file``) and print one line:

    mem ▸ ingest 52% (ETA 1m38s) · 1.21M tok · q:1 · serving 48k of 1.10M · ctx 8%

``mem ▸ off`` when no status document exists, ``mem ▸ ?`` on any failure.
This script only reads; it never writes anything. Exit status is always 0:
a statusline must never break the client. Imports are stdlib plus
``memory_condense.control.status`` only, because the whole budget is a few
milliseconds beyond interpreter startup.

Run with ``python -S`` when possible: skipping ``site`` saves a measurable
fraction of the startup cost and this script does not need site-packages
(the guarded ``sys.path`` insert below finds the package from a checkout).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OFF = "mem ▸ off"
UNKNOWN = "mem ▸ ?"


def _ensure_importable() -> None:
    """Fallback for a plain interpreter: ``<repo>/src`` relative to this file."""
    src = Path(__file__).resolve().parent.parent / "src"
    if (src / "memory_condense").is_dir() and str(src) not in sys.path:
        sys.path.append(str(src))


def _emit(line: str) -> None:
    # Claude Code captures a pipe, whose default encoding on Windows cannot
    # carry "▸"; write UTF-8 bytes directly rather than trusting the locale.
    data = (line + "\n").encode("utf-8", "replace")
    try:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    except Exception:  # noqa: BLE001 - last resort, ASCII only
        sys.stdout.write(line.encode("ascii", "replace").decode("ascii") + "\n")


def _context_suffix(payload: dict) -> str:
    cw = payload.get("context_window")
    pct = cw.get("used_percentage") if isinstance(cw, dict) else None
    if isinstance(pct, (int, float)) and not isinstance(pct, bool):
        return f" · ctx {pct:.0f}%"
    return ""


def render(payload: dict, env=None) -> str:
    from memory_condense.control.status import locate_status_file, one_line, read_status

    workspace = payload.get("workspace")
    cwd = (workspace.get("current_dir") if isinstance(workspace, dict) else None) or payload.get("cwd")
    path = locate_status_file(env, Path(cwd) if cwd else None)
    if path is None:
        line = OFF
    else:
        doc = read_status(path)
        line = UNKNOWN if doc is None else one_line(doc)
    return line + _context_suffix(payload)


def main() -> int:
    try:
        _ensure_importable()
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        if not isinstance(payload, dict):
            payload = {}
        _emit(render(payload))
    except Exception:  # noqa: BLE001 - never raise out of a statusline
        _emit(UNKNOWN)
    return 0


if __name__ == "__main__":
    sys.exit(main())
