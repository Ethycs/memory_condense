"""``memory-condense ctl``: the control plane from the terminal.

``status`` renders ``status.json`` (no proxy needed); ``save``, ``load``,
``reset`` and ``load-transcript`` POST to the proxy's local control routes.
Only ``control.status`` and the routes in
``docs/01 - Design/03 - System Controls.md`` are relied on here — this module
never computes a number of its own.

Exit codes: 0 ok · 1 proxy unreachable or the route refused · 2 usage or a
refused destructive command · 3 no status document.

The HTTP transport (:func:`post_control`) is also what the MCP tools use, so
the two surfaces cannot drift in how they talk to the proxy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from memory_condense.control.status import (
    age_seconds,
    describe_startup,
    format_eta,
    format_tokens,
    locate_status_file,
    one_line,
    read_status,
    startup_records,
)

DEFAULT_PROXY_URL = "http://127.0.0.1:8787"
PROXY_ENV = "MEMORY_CONDENSE_PROXY"
STALE_AFTER_S = 10.0

EXIT_OK = 0
EXIT_UNREACHABLE = 1
EXIT_REFUSED = 2
EXIT_NO_STATUS = 3

NO_STATUS_MESSAGE = "no status document found: proxy not running or no store here"


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


class ControlUnavailable(RuntimeError):
    """The proxy could not be reached at all (nothing listening, DNS, timeout)."""


class ControlRefused(RuntimeError):
    """The proxy answered with an error status; ``body`` is its decoded reply."""

    def __init__(self, status: int, body: Any):
        super().__init__(f"HTTP {status}")
        self.status = status
        self.body = body


def default_proxy_url(env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    return (env.get(PROXY_ENV) or DEFAULT_PROXY_URL).rstrip("/")


def _decode(raw: bytes) -> Any:
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text) if text.strip() else {}
    except ValueError:
        return {"raw": text}


def post_control(base_url: str, route: str, payload: Mapping[str, Any], *, timeout: float = 120.0) -> Any:
    """POST ``payload`` as JSON to ``<base_url>/_control/<route>``; return the decoded reply.

    Raises :class:`ControlUnavailable` when nothing answers and
    :class:`ControlRefused` when the proxy answers with a non-2xx status.
    """
    url = f"{base_url.rstrip('/')}/_control/{route.lstrip('/')}"
    body = json.dumps(dict(payload)).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return _decode(response.read())
    except urllib.error.HTTPError as exc:
        raise ControlRefused(exc.code, _decode(exc.read())) from exc
    except (urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise ControlUnavailable(f"{url}: {reason}") from exc


def unreachable_hint(base_url: str) -> str:
    return (
        f"memory proxy not reachable at {base_url} — start it with `memory-condense proxy`"
        f" or point {PROXY_ENV} at the running one"
    )


# ---------------------------------------------------------------------------
# Status rendering
# ---------------------------------------------------------------------------


def format_bytes(n: int | float | None) -> str:
    if n is None:
        return "--"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"  # pragma: no cover - loop always returns


def _num(x: Any, digits: int = 1) -> str:
    if x is None:
        return "--"
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


def is_stale(doc: Mapping[str, Any]) -> bool:
    age = age_seconds(doc)
    return age is not None and age > STALE_AFTER_S


def render_table(doc: Mapping[str, Any]) -> str:
    """The on-demand view: one summary line, then one labelled row per section."""
    rows: list[tuple[str, str]] = []
    age = age_seconds(doc)
    stale = is_stale(doc)

    state = f"{doc.get('state', '?')} ({doc.get('mode', '?')}, pid {doc.get('pid', '?')})"
    if stale:
        state += f"  STALE — last written {format_eta(age)} ago; the proxy may be gone"
    rows.append(("state", state))

    health = doc.get("health") or {}
    reasons = health.get("reasons") or []
    rows.append(("health", str(health.get("status", "?")) + (f" — {'; '.join(map(str, reasons))}" if reasons else "")))

    q = doc.get("queue") or {}
    jobs = q.get("jobs") or []
    queue = (
        f"capacity {q.get('capacity', '--')} · in-flight {q.get('in_flight', 0)}"
        f" · retained {format_bytes(q.get('retained_bytes'))}"
        f" · rejected {q.get('rejected_saturated', 0)} · failed {q.get('terminal_failures', 0)}"
        f" · jobs {len(jobs)}"
    )
    rows.append(("queue", queue))
    for job in jobs:
        rows.append((
            "",
            f"#{job.get('position', '?')} {job.get('id', '?')} {job.get('kind', '')}"
            f" {job.get('session', '')} {format_bytes(job.get('bytes'))}".rstrip(),
        ))

    cur = doc.get("current")
    if cur:
        rec_total = cur.get("records_total")
        pct = ""
        if rec_total:
            pct = f" ({100 * (cur.get('records_done') or 0) / rec_total:.0f}%)"
        current = (
            f"{cur.get('job', '?')} {cur.get('kind', '')}: "
            f"{cur.get('records_done', 0)}/{rec_total if rec_total is not None else '?'} records{pct}"
            f" · {format_bytes(cur.get('bytes_done'))}/{format_bytes(cur.get('bytes_total'))}"
            f" · {format_tokens(cur.get('tokens_done'))} tok"
            f" · ETA {format_eta(cur.get('eta_s'))}"
        )
        if cur.get("started_at"):
            current += f" · started {cur['started_at']}"
    elif doc.get("state") == "starting":
        # No bulk job, but the native service is not idle either. Saying "idle"
        # here is the exact report this view exists to avoid.
        current = " ".join(["starting:", *describe_startup(doc)])
    else:
        current = "idle"
    rows.append(("current", current))

    # Per-session identity: aggregation must never hide which session is stuck.
    memory = doc.get("memory") if isinstance(doc.get("memory"), Mapping) else {}
    for name, record in sorted(startup_records(doc).items()):
        detail = f"{record.get('stage', '?')} · {format_eta(record.get('stage_elapsed_s'))}"
        detail += f" (session {format_eta(record.get('elapsed_s'))})"
        if record.get("error"):
            detail += f" · ERROR {record['error']}"
        rows.append(("session " + name[:12], detail))
    for name, waited in sorted((memory.get("opening_sessions") or {}).items()):
        rows.append(("session " + name[:12], f"opening · {format_eta(waited)}"))
    if memory.get("error"):
        rows.append(("memory", f"status unavailable — {memory['error']}"))

    rates = doc.get("rates") or {}
    rows.append((
        "rates",
        f"{format_tokens(rates.get('tokens_per_s_recent'))} tok/s"
        f" · {_num(rates.get('records_per_s_recent'))} rec/s"
        f" · {_num(rates.get('avg_s_per_1k_tokens'), 2)} s/1k tok"
        f" (last {rates.get('window_s', '--')}s)",
    ))

    totals = doc.get("totals") or {}
    rows.append((
        "totals",
        f"{format_tokens(totals.get('tokens_ingested'))} tok"
        f" · {totals.get('records_ingested', 0)} records"
        f" · {totals.get('sessions', 0)} sessions"
        f" · store {format_bytes(totals.get('store_bytes'))}",
    ))

    serving = doc.get("serving") or {}
    rows.append((
        "serving",
        f"{serving.get('requests', 0)} requests · {serving.get('rewritten', 0)} rewritten"
        f" · last {format_tokens(serving.get('served_tokens_last'))} of {format_tokens(serving.get('static_tokens_last'))}"
        f" · saved {format_tokens(serving.get('saved_tokens_total'))} tok",
    ))

    store = doc.get("store") or {}
    rows.append(("store", str(store.get("path") or "--")))
    snapshots = store.get("snapshots") or []
    last = store.get("last_snapshot")
    if snapshots or last:
        shown = ", ".join(str(s) for s in snapshots[:5])
        if len(snapshots) > 5:
            shown += f", … (+{len(snapshots) - 5})"
        rows.append(("snapshots", f"{len(snapshots)} in {store.get('snapshot_dir') or '--'}; last {last or '--'}"))
        if shown:
            rows.append(("", shown))
    else:
        rows.append(("snapshots", "none"))

    rows.append(("written", f"{doc.get('written_at', '?')} ({format_eta(age)} ago)" if age is not None else str(doc.get("written_at", "?"))))

    width = max(len(label) for label, _ in rows)
    lines = [one_line(doc)]
    lines += [f"{label:<{width}}  {value}" if label else f"{'':<{width}}    {value}" for label, value in rows]
    return "\n".join(lines)


def _load_status(explicit: str | None) -> tuple[Path | None, dict[str, Any] | None]:
    path = Path(explicit).expanduser() if explicit else locate_status_file()
    if path is None or not path.is_file():
        return path, None
    return path, read_status(path)


def _print_status(explicit: str | None, as_json: bool) -> int:
    path, doc = _load_status(explicit)
    if doc is None:
        if path is not None and path.exists():
            print(f"status document at {path} is unreadable or has an unsupported schema", file=sys.stderr)
        else:
            print(NO_STATUS_MESSAGE, file=sys.stderr)
        return EXIT_NO_STATUS
    if as_json:
        out = dict(doc)
        out["stale"] = is_stale(doc)
        out["age_seconds"] = age_seconds(doc)
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(render_table(doc))
    return EXIT_OK


def cmd_status(args: argparse.Namespace) -> int:
    if not args.watch:
        return _print_status(args.status, args.json)
    interval = max(0.2, float(args.watch))
    try:
        while True:
            if not args.json and sys.stdout.isatty():
                print("\x1b[2J\x1b[H", end="")
            code = _print_status(args.status, args.json)
            if code != EXIT_OK:
                return code
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        return EXIT_OK


# ---------------------------------------------------------------------------
# Mutating commands
# ---------------------------------------------------------------------------


def _call(args: argparse.Namespace, route: str, payload: Mapping[str, Any]) -> int:
    base = (args.proxy or default_proxy_url()).rstrip("/")
    try:
        reply = post_control(base, route, payload)
    except ControlUnavailable:
        print(unreachable_hint(base), file=sys.stderr)
        return EXIT_UNREACHABLE
    except ControlRefused as exc:
        print(f"{route}: proxy refused ({exc})", file=sys.stderr)
        print(json.dumps(exc.body, indent=2, ensure_ascii=False))
        return EXIT_UNREACHABLE
    print(json.dumps(reply, indent=2, ensure_ascii=False))
    return EXIT_OK


def cmd_save(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {}
    if args.name:
        payload["name"] = args.name
    return _call(args, "save", payload)


def cmd_load(args: argparse.Namespace) -> int:
    return _call(args, "load", {"name": args.name})


def cmd_reset(args: argparse.Namespace) -> int:
    if not args.yes:
        print(
            "reset refused: this replaces the store with an empty one (a snapshot is taken first)."
            " Re-run with --yes to confirm.",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    return _call(args, "reset", {"confirm": True})


def cmd_load_transcript(args: argparse.Namespace) -> int:
    return _call(args, "load-transcript", {"session": args.session})


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _add_proxy(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--proxy",
        default=None,
        help=f"proxy base URL (default: ${PROXY_ENV} or {DEFAULT_PROXY_URL})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memory-condense ctl",
        description="Inspect and control the memory proxy: status, snapshots, reset, bulk transcript loads.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("status", help="Render status.json (no proxy connection needed)")
    p.add_argument("--json", action="store_true", help="print the raw document (plus stale/age_seconds)")
    p.add_argument("--watch", type=float, metavar="SECONDS", help="re-render every SECONDS until Ctrl-C")
    p.add_argument("--status", metavar="PATH", help="explicit status.json path (default: resolved per the design note)")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("save", help="Snapshot the store (VACUUM INTO <data-dir>/snapshots/)")
    p.add_argument("name", nargs="?", help="snapshot name (default: timestamp)")
    _add_proxy(p)
    p.set_defaults(func=cmd_save)

    p = sub.add_parser("load", help="Swap the store to a snapshot")
    p.add_argument("name", help="snapshot name as listed by `ctl status`")
    _add_proxy(p)
    p.set_defaults(func=cmd_load)

    p = sub.add_parser("reset", help="Snapshot, then re-initialize an empty store (requires --yes)")
    p.add_argument("--yes", action="store_true", help="confirm the reset")
    _add_proxy(p)
    p.set_defaults(func=cmd_reset)

    p = sub.add_parser("load-transcript", help="Enqueue a bulk transcript ingest job")
    p.add_argument("session", help="session id or transcript path")
    _add_proxy(p)
    p.set_defaults(func=cmd_load_transcript)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    # The summary line contains "▸"; a cp1252 console must not crash on it.
    try:
        sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
