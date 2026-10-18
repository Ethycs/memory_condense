"""``UserPromptSubmit`` hook: a one-line note to the user when memory state changed.

Claude Code (and Codex) pipe the hook JSON on stdin (``session_id``, ``cwd``,
``hook_event_name``, ...; see https://code.claude.com/docs/en/hooks). We load
``status.json`` (``memory_condense.control.status.locate_status_file``), reduce
it to a small state tuple, and compare it with the tuple we saw on the previous
prompt of this session, kept at
``<tempdir>/memory_condense_hook_<session_id>.json``.

Only when the tuple changed do we print

    {"systemMessage": "memory: ingest started (bulk-01a08337)"}

``systemMessage`` is shown to the user in the transcript; it is not
``additionalContext``, so nothing here enters the model's prompt. Otherwise
the hook prints nothing. The exit status is always 0: plain-text stdout from a
``UserPromptSubmit`` hook would be added to the model's context and a non-zero
exit would show an error, so any failure is swallowed silently.

The hook only reads ``status.json``; its sole write is its own last-seen file.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _ensure_importable() -> None:
    """Fallback for a plain interpreter: ``<repo>/src`` relative to this file."""
    src = Path(__file__).resolve().parents[2] / "src"
    if (src / "memory_condense").is_dir() and str(src) not in sys.path:
        sys.path.append(str(src))


def last_seen_path(session_id: str) -> Path:
    safe = _SAFE.sub("_", str(session_id))[:120] or "unknown"
    return Path(tempfile.gettempdir()) / f"memory_condense_hook_{safe}.json"


def state_tuple(doc: dict | None) -> dict:
    """The few fields whose change is worth a note. ``present`` is False when no document exists."""
    if doc is None:
        return {"present": False, "state": None, "snapshot": None, "sessions": None, "job": None,
                "health": None, "stage": None}
    cur = doc.get("current") or {}
    return {
        "present": True,
        "state": doc.get("state"),
        "snapshot": (doc.get("store") or {}).get("last_snapshot"),
        "sessions": (doc.get("totals") or {}).get("sessions"),
        "job": cur.get("job") if isinstance(cur, dict) else None,
        "health": (doc.get("health") or {}).get("status"),
        # A cold start can sit in one state for many minutes. Tracking the
        # stage is what makes its progress visible turn to turn.
        "stage": startup_stage(doc),
    }


def startup_stage(doc: dict) -> str | None:
    """The stage of the least-advanced native session, or None when all are ready."""
    from memory_condense.control.status import startup_records

    pending = [r for r in startup_records(doc).values() if r.get("stage") != "ready"]
    if not pending:
        return None
    return str(max(pending, key=lambda r: r.get("elapsed_s") or 0.0).get("stage") or "unknown")


def describe_change(before: dict | None, now: dict, doc: dict | None) -> str | None:
    """One line saying what changed; None when nothing did or there is nothing to say."""
    from memory_condense.control.status import one_line

    if before is None:
        # First prompt of this session: say hello only if the proxy is publishing.
        return one_line(doc) if now["present"] and doc is not None else None
    if before == now:
        return None
    if not now["present"]:
        return "status file gone (proxy stopped?)"
    if not before["present"]:
        return "proxy online: " + one_line(doc)

    notes: list[str] = []
    if before["state"] != now["state"]:
        if now["state"] == "starting":
            notes.append("memory starting" + (f" ({now['stage'].replace('_', ' ')})" if now["stage"] else ""))
        elif now["state"] == "ingesting":
            notes.append(f"ingest started ({now['job']})" if now["job"] else "ingest started")
        elif before["state"] == "starting":
            notes.append("memory ready" if now["state"] == "idle" else f"memory ready, now {now['state']}")
        elif before["state"] == "ingesting":
            notes.append("ingest finished" + (f", now {now['state']}" if now["state"] != "idle" else ""))
        elif now["state"] == "swapping":
            notes.append("swapping store (serving passthrough)")
        elif now["state"] == "error":
            notes.append("ERROR state")
        else:
            notes.append(f"{before['state']} -> {now['state']}")
    elif before["stage"] != now["stage"] and now["stage"]:
        # Same state, further along: the only progress signal during a long,
        # total-less model load.
        notes.append("startup: " + now["stage"].replace("_", " "))
    elif before["job"] != now["job"] and now["job"]:
        notes.append(f"job now {now['job']}")
    if before["snapshot"] != now["snapshot"] and now["snapshot"]:
        notes.append(f"snapshot taken {now['snapshot']}")
    if before["sessions"] != now["sessions"] and now["sessions"] is not None:
        if before["sessions"] is not None and now["sessions"] < before["sessions"]:
            notes.append(f"store reset (sessions {before['sessions']} -> {now['sessions']})")
        else:
            notes.append(f"sessions {before['sessions']} -> {now['sessions']}")
    if before["health"] != now["health"]:
        notes.append(f"health {before['health']} -> {now['health']}")
    return "; ".join(notes) if notes else None


def _read_last_seen(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and "present" in data else None


def _write_last_seen(path: Path, now: dict) -> None:
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(now), encoding="utf-8")
    os.replace(tmp, path)


def run(payload: dict, env=None) -> str | None:
    from memory_condense.control.status import locate_status_file, read_status

    session_id = payload.get("session_id") or "unknown"
    cwd = payload.get("cwd")
    path = locate_status_file(env, Path(cwd) if cwd else None)
    doc = read_status(path) if path is not None else None
    now = state_tuple(doc)
    seen_at = last_seen_path(session_id)
    before = _read_last_seen(seen_at)
    if before is None and not now["present"]:
        return None  # nothing to report and nothing worth remembering
    message = describe_change(before, now, doc)
    if before != now:
        _write_last_seen(seen_at, now)
    return message


def main() -> int:
    try:
        _ensure_importable()
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        if not isinstance(payload, dict):
            return 0
        message = run(payload)
        if message:
            out = json.dumps({"systemMessage": f"memory: {message}"}, ensure_ascii=False)
            sys.stdout.buffer.write((out + "\n").encode("utf-8", "replace"))
            sys.stdout.buffer.flush()
    except Exception:  # noqa: BLE001 - a hook must never block the prompt
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
