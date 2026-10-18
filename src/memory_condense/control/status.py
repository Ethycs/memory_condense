"""The status document: where it lives, its shape, and atomic read/write.

This is the contract every surface shares. The proxy writes it (see
``publisher``); the statusline, hooks, CLI, and MCP tools only read it.
Schema: ``docs/01 - Design/03 - System Controls.md``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

SCHEMA_VERSION = 1
STATUS_RELPATH = Path("control") / "status.json"

# "starting" was added for augment mode: a native session whose models are
# still loading is neither idle nor ingesting records. It is additive to
# schema 1 — renderers that do not know it fall through to showing the name.
STATES = ("idle", "starting", "ingesting", "condensing", "swapping", "error")


def user_data_dir() -> Path:
    from memory_condense.runtime.config import user_directory

    return user_directory() / "data"


def _iter_candidates(env: Mapping[str, str], cwd: Path) -> Iterator[Path]:
    # Lazy on purpose: the statusline runs this every refresh, and the last
    # candidate imports runtime.config (~40 ms); it must not be paid when an
    # earlier candidate resolves.
    if env.get("MEMORY_CONDENSE_STATUS"):
        yield Path(env["MEMORY_CONDENSE_STATUS"]).expanduser()
    if env.get("MEMORY_CONDENSE_DATA_DIR"):
        yield Path(env["MEMORY_CONDENSE_DATA_DIR"]).expanduser() / STATUS_RELPATH
    # runtime.project is stdlib-only and costs nothing extra here; it is the
    # same resolver the proxy writes with, so the walk up to the project root
    # cannot drift from where the store actually is. cwd/.memory_condense is
    # kept as a candidate for a store that predates the project-root walk.
    from memory_condense.runtime.project import DIRNAME, project_root

    root = project_root(cwd, env)
    if root is not None:
        yield root / DIRNAME / STATUS_RELPATH
    yield cwd / DIRNAME / STATUS_RELPATH
    yield user_data_dir() / STATUS_RELPATH


def candidate_status_files(env: Mapping[str, str] | None = None, cwd: Path | None = None) -> list[Path]:
    """Resolution order from the design note; first existing wins."""
    env = os.environ if env is None else env
    cwd = Path.cwd() if cwd is None else cwd
    return list(_iter_candidates(env, cwd))


def locate_status_file(env: Mapping[str, str] | None = None, cwd: Path | None = None) -> Path | None:
    env = os.environ if env is None else env
    cwd = Path.cwd() if cwd is None else cwd
    for path in _iter_candidates(env, cwd):
        if path.is_file():
            return path
    return None


def status_path_for(data_dir: Path | str) -> Path:
    return Path(data_dir) / STATUS_RELPATH


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def empty_status(*, mode: str = "observe", pid: int | None = None) -> dict[str, Any]:
    """A complete, idle document. Writers start from this and fill fields in."""
    return {
        "schema": SCHEMA_VERSION,
        "written_at": now_iso(),
        "pid": os.getpid() if pid is None else pid,
        "mode": mode,
        "state": "idle",
        "health": {"status": "ok", "reasons": []},
        "queue": {
            "capacity": 0, "in_flight": 0, "retained_bytes": 0,
            "rejected_saturated": 0, "terminal_failures": 0, "jobs": [],
        },
        "current": None,
        "rates": {"tokens_per_s_recent": 0.0, "records_per_s_recent": 0.0,
                  "avg_s_per_1k_tokens": None, "window_s": 30},
        "totals": {"tokens_ingested": 0, "records_ingested": 0, "sessions": 0, "store_bytes": 0},
        "serving": {"requests": 0, "rewritten": 0, "static_tokens_last": None,
                    "served_tokens_last": None, "saved_tokens_total": 0},
        "store": {"path": None, "snapshot_dir": None, "last_snapshot": None, "snapshots": []},
    }


def write_status(path: Path | str, doc: Mapping[str, Any]) -> Path:
    """Atomic: a reader never sees a torn document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(doc)
    payload["schema"] = SCHEMA_VERSION
    payload["written_at"] = now_iso()
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    # Pre-rendered one-liner beside the document: a statusline can `type` it
    # in ~10 ms, where any Python process costs 50-100 ms on a slow machine.
    line = path.with_name("status.line")
    line_tmp = line.with_suffix(f".line.{os.getpid()}.tmp")
    line_tmp.write_text(one_line(payload) + "\n", encoding="utf-8")
    os.replace(line_tmp, line)
    return path


def read_status(path: Path | str) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and doc.get("schema") == SCHEMA_VERSION else None


def age_seconds(doc: Mapping[str, Any]) -> float | None:
    """Seconds since the writer last published; stale documents mean a dead proxy."""
    try:
        written = time.strptime(doc["written_at"], "%Y-%m-%dT%H:%M:%SZ")
    except (KeyError, ValueError, TypeError):
        return None
    import calendar

    return max(0.0, time.time() - calendar.timegm(written))


def format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds}s"


def format_tokens(n: int | float | None) -> str:
    if n is None:
        return "--"
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}k"
    return f"{int(n)}"


def startup_records(doc: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-session startup stages from the native memory service, if any."""
    memory = doc.get("memory")
    startup = memory.get("startup") if isinstance(memory, Mapping) else None
    if not isinstance(startup, Mapping):
        return {}
    return {name: dict(record) for name, record in startup.items() if isinstance(record, Mapping)}


def describe_startup(doc: Mapping[str, Any]) -> list[str]:
    """Words for the slowest pre-ready session: its stage and elapsed time."""
    pending = [record for record in startup_records(doc).values() if record.get("stage") != "ready"]
    if not pending:
        opening = (doc.get("memory") or {}).get("opening_sessions") or {}
        waited = max(opening.values(), default=None) if isinstance(opening, Mapping) else None
        return ["opening", format_eta(waited)] if waited is not None else []
    slowest = max(pending, key=lambda record: record.get("elapsed_s") or 0.0)
    stage = str(slowest.get("stage") or "unknown").replace("_", " ")
    words = [stage, format_eta(slowest.get("elapsed_s"))]
    if len(pending) > 1:
        words.append(f"(+{len(pending) - 1})")
    return words


def one_line(doc: Mapping[str, Any]) -> str:
    """The statusline rendering; every surface that wants a summary uses this."""
    parts = ["mem"]
    state = doc.get("state", "idle")
    cur = doc.get("current") or {}
    if state == "ingesting" and cur:
        total = cur.get("records_total") or 0
        done = cur.get("records_done") or 0
        pct = f"{100 * done / total:.0f}%" if total else f"{done} rec"
        parts.append(f"ingest {pct} (ETA {format_eta(cur.get('eta_s'))})")
    elif state == "starting":
        # No total and no ETA during a cold model load. Naming the stage and
        # how long it has run is what tells a waiting user it is not wedged.
        parts.append(" ".join(["starting", *describe_startup(doc)]))
    elif state != "idle":
        parts.append(state)
    totals = doc.get("totals") or {}
    parts.append(f"{format_tokens(totals.get('tokens_ingested'))} tok")
    q = doc.get("queue") or {}
    jobs = len(q.get("jobs") or [])
    if jobs or q.get("in_flight"):
        parts.append(f"q:{jobs}" + (f"+{q['in_flight']}" if q.get("in_flight") else ""))
    serving = doc.get("serving") or {}
    if serving.get("served_tokens_last") is not None and serving.get("static_tokens_last"):
        parts.append(f"serving {format_tokens(serving['served_tokens_last'])} of {format_tokens(serving['static_tokens_last'])}")
    health = (doc.get("health") or {}).get("status")
    if health and health != "ok":
        parts.append(health.upper())
    age = age_seconds(doc)
    if age is not None and age > 10:
        parts.append(f"stale {format_eta(age)}")
    return " ▸ ".join(parts[:1] + [" · ".join(parts[1:])]) if len(parts) > 1 else parts[0]
