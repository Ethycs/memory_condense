"""Client surfaces (work package C): the statusline script and the prompt hook.

Both are run as real subprocesses, the way Claude Code runs them, against
fixture status documents written with ``write_status``. They only read.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from memory_condense.control.status import empty_status, write_status

REPO = Path(__file__).resolve().parents[1]
STATUSLINE = REPO / "tools" / "memory_statusline.py"
HOOK = REPO / ".claude" / "hooks" / "memory_state_note.py"


def _env(tmp_path: Path, status_file: Path) -> dict[str, str]:
    """Pin every resolution step of ``locate_status_file`` and ``gettempdir`` into tmp_path."""
    env = dict(os.environ)
    env["MEMORY_CONDENSE_STATUS"] = str(status_file)
    env.pop("MEMORY_CONDENSE_DATA_DIR", None)
    env["CLAUDE_PROJECT_DIR"] = str(tmp_path)
    env["LOCALAPPDATA"] = str(tmp_path)
    env["XDG_DATA_HOME"] = str(tmp_path)
    for key in ("TMPDIR", "TEMP", "TMP"):
        env[key] = str(tmp_path)
    return env


def _run(script: Path, stdin: str, env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, str(script)],
        input=stdin.encode("utf-8"),
        capture_output=True,
        env=env,
        cwd=str(cwd),
        timeout=60,
    )


def _ingesting_doc() -> dict:
    doc = empty_status(mode="augment")
    doc["state"] = "ingesting"
    doc["current"] = {
        "job": "bulk-01a08337", "kind": "transcript",
        "records_done": 2944, "records_total": 5677,
        "bytes_done": 134217728, "bytes_total": 258895902,
        "tokens_done": 612000, "started_at": "2026-10-13T08:39:00Z", "eta_s": 98,
    }
    doc["queue"]["jobs"] = [{"id": "bulk-01a08337", "kind": "transcript", "session": "01a08337",
                             "position": 1, "bytes": 258895902}]
    doc["totals"] = {"tokens_ingested": 1210000, "records_ingested": 18231, "sessions": 3, "store_bytes": 48100000}
    doc["serving"] = {"requests": 41, "rewritten": 37, "static_tokens_last": 1104000,
                      "served_tokens_last": 48200, "saved_tokens_total": 38900000}
    return doc


def _statusline_input(tmp_path: Path, **extra) -> str:
    payload = {
        "cwd": str(tmp_path),
        "session_id": "sess-statusline",
        "model": {"id": "claude-opus-5-5", "display_name": "Opus"},
        "workspace": {"current_dir": str(tmp_path), "project_dir": str(tmp_path)},
    }
    payload.update(extra)
    return json.dumps(payload)


def _hook_input(tmp_path: Path, session_id: str) -> str:
    return json.dumps({
        "session_id": session_id,
        "transcript_path": str(tmp_path / "t.jsonl"),
        "cwd": str(tmp_path),
        "hook_event_name": "UserPromptSubmit",
        "prompt": "hello",
    })


# --- statusline ------------------------------------------------------------

def test_statusline_renders_ingest_progress_and_context(tmp_path: Path) -> None:
    status_file = tmp_path / "control" / "status.json"
    write_status(status_file, _ingesting_doc())
    stdin = _statusline_input(tmp_path, context_window={"used_percentage": 41.7, "context_window_size": 200000})

    proc = _run(STATUSLINE, stdin, _env(tmp_path, status_file), tmp_path)

    assert proc.returncode == 0
    line = proc.stdout.decode("utf-8").strip()
    assert line.startswith("mem ▸ ingest 52% (ETA 1m38s)")
    assert "1.21M tok" in line
    assert "q:1" in line
    assert "serving 48k of 1.10M" in line
    assert line.endswith("· ctx 42%")
    assert "\n" not in line


def test_statusline_off_when_no_status_file(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere" / "status.json"
    proc = _run(STATUSLINE, _statusline_input(tmp_path), _env(tmp_path, missing), tmp_path)

    assert proc.returncode == 0
    assert proc.stdout.decode("utf-8").strip() == "mem ▸ off"


def test_statusline_never_raises_on_garbage(tmp_path: Path) -> None:
    proc = _run(STATUSLINE, "{not json", _env(tmp_path, tmp_path / "x.json"), tmp_path)

    assert proc.returncode == 0
    assert proc.stdout.decode("utf-8").strip() == "mem ▸ ?"
    assert proc.stderr == b""


def test_statusline_unreadable_document_is_unknown_not_off(tmp_path: Path) -> None:
    status_file = tmp_path / "control" / "status.json"
    status_file.parent.mkdir(parents=True)
    status_file.write_text("{torn", encoding="utf-8")

    proc = _run(STATUSLINE, _statusline_input(tmp_path), _env(tmp_path, status_file), tmp_path)

    assert proc.returncode == 0
    assert proc.stdout.decode("utf-8").strip() == "mem ▸ ?"


# --- hook ------------------------------------------------------------------

def _hook_message(proc: subprocess.CompletedProcess[bytes]) -> str | None:
    assert proc.returncode == 0
    assert proc.stderr == b""
    out = proc.stdout.decode("utf-8").strip()
    if not out:
        return None
    data = json.loads(out)
    assert set(data) == {"systemMessage"}, "only systemMessage may be emitted; nothing enters model context"
    return data["systemMessage"]


def test_hook_notes_first_sight_change_and_silence_on_repeat(tmp_path: Path) -> None:
    status_file = tmp_path / "control" / "status.json"
    env = _env(tmp_path, status_file)
    stdin = _hook_input(tmp_path, "sess-hook-1")

    idle = empty_status(mode="augment")
    idle["totals"]["sessions"] = 3
    write_status(status_file, idle)

    first = _hook_message(_run(HOOK, stdin, env, tmp_path))
    assert first is not None and first.startswith("memory: mem ▸ ")
    assert (tmp_path / "memory_condense_hook_sess-hook-1.json").is_file()

    assert _hook_message(_run(HOOK, stdin, env, tmp_path)) is None  # unchanged: silent

    write_status(status_file, _ingesting_doc())
    started = _hook_message(_run(HOOK, stdin, env, tmp_path))
    assert started == "memory: ingest started (bulk-01a08337)"

    assert _hook_message(_run(HOOK, stdin, env, tmp_path)) is None  # still ingesting: silent

    done = empty_status(mode="augment")
    done["totals"]["sessions"] = 4
    done["store"]["last_snapshot"] = "2026-10-13T08-40-00.sqlite"
    write_status(status_file, done)
    finished = _hook_message(_run(HOOK, stdin, env, tmp_path))
    assert finished is not None
    assert "ingest finished" in finished
    assert "snapshot taken 2026-10-13T08-40-00.sqlite" in finished
    assert "sessions 3 -> 4" in finished

    reset = empty_status(mode="augment")
    reset["store"]["last_snapshot"] = "2026-10-13T08-40-00.sqlite"
    write_status(status_file, reset)
    assert _hook_message(_run(HOOK, stdin, env, tmp_path)) == "memory: store reset (sessions 4 -> 0)"


def _starting_doc(stage: str, elapsed: float) -> dict:
    doc = empty_status(mode="augment")
    doc["state"] = "starting"
    doc["memory"] = {"startup": {"alpha": dict(stage=stage, error=None,
                                               elapsed_s=elapsed, stage_elapsed_s=elapsed)}}
    return doc


def test_hook_follows_a_cold_start_through_its_stages(tmp_path: Path) -> None:
    """A 20-minute model load must not go silent after its first turn."""
    status_file = tmp_path / "control" / "status.json"
    env = _env(tmp_path, status_file)
    stdin = _hook_input(tmp_path, "sess-cold")

    write_status(status_file, _starting_doc("awaiting_model_start", 3.0))
    first = _hook_message(_run(HOOK, stdin, env, tmp_path))
    assert first is not None and "starting awaiting model start" in first

    assert _hook_message(_run(HOOK, stdin, env, tmp_path)) is None  # same stage: silent

    write_status(status_file, _starting_doc("opening_models", 240.0))
    assert _hook_message(_run(HOOK, stdin, env, tmp_path)) == "memory: startup: opening models"

    ready = empty_status(mode="augment")
    ready["memory"] = {"startup": {"alpha": dict(stage="ready", error=None,
                                                 elapsed_s=1500.0, stage_elapsed_s=1.0)}}
    write_status(status_file, ready)
    assert _hook_message(_run(HOOK, stdin, env, tmp_path)) == "memory: memory ready"


def test_hook_sessions_are_independent(tmp_path: Path) -> None:
    status_file = tmp_path / "control" / "status.json"
    env = _env(tmp_path, status_file)
    write_status(status_file, empty_status())

    assert _hook_message(_run(HOOK, _hook_input(tmp_path, "sess-a"), env, tmp_path)) is not None
    assert _hook_message(_run(HOOK, _hook_input(tmp_path, "sess-a"), env, tmp_path)) is None
    assert _hook_message(_run(HOOK, _hook_input(tmp_path, "sess-b"), env, tmp_path)) is not None


def test_hook_silent_when_proxy_never_published(tmp_path: Path) -> None:
    env = _env(tmp_path, tmp_path / "nowhere" / "status.json")
    proc = _run(HOOK, _hook_input(tmp_path, "sess-off"), env, tmp_path)

    assert _hook_message(proc) is None
    assert not (tmp_path / "memory_condense_hook_sess-off.json").exists()


def test_hook_reports_status_file_gone_after_being_seen(tmp_path: Path) -> None:
    status_file = tmp_path / "control" / "status.json"
    env = _env(tmp_path, status_file)
    stdin = _hook_input(tmp_path, "sess-gone")
    write_status(status_file, empty_status())
    assert _hook_message(_run(HOOK, stdin, env, tmp_path)) is not None

    status_file.unlink()
    assert _hook_message(_run(HOOK, stdin, env, tmp_path)) == "memory: status file gone (proxy stopped?)"
    assert _hook_message(_run(HOOK, stdin, env, tmp_path)) is None


@pytest.mark.parametrize("stdin", ["", "{not json", "[1, 2, 3]", '{"session_id": null}'])
def test_hook_exits_zero_silently_on_garbage(tmp_path: Path, stdin: str) -> None:
    env = _env(tmp_path, tmp_path / "nowhere" / "status.json")
    proc = _run(HOOK, stdin, env, tmp_path)

    assert proc.returncode == 0
    assert proc.stdout == b""
    assert proc.stderr == b""
