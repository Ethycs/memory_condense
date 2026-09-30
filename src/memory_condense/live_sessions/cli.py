"""``python -m memory_condense.live_sessions`` -- discover, handles, watch, serve."""

from __future__ import annotations

import argparse
import json
import sys
import time

from .discovery import SessionKind, discover_sessions, find_session
from .handles import HandleTracker, snapshot
from .mmap_view import MappedSession
from .virtual_fs import Mirror, projfs_available


def _kinds(arg: str | None) -> list[SessionKind] | None:
    if not arg or arg == "all":
        return None
    return [SessionKind(k) for k in arg.split(",")]


def cmd_discover(args: argparse.Namespace) -> int:
    refs = discover_sessions(_kinds(args.kind), use_handles=not args.no_handles)
    if args.live:
        refs = [r for r in refs if r.live]
    if args.json:
        for r in refs:
            print(json.dumps({**r.__dict__, "kind": r.kind.value}))
        return 0
    print(f"{'liveness':8} {'kind':6} {'pids':12} {'size':>10} {'age':>8}  session  project")
    for r in refs[: args.limit]:
        pids = ",".join(map(str, r.pids)) or "-"
        print(
            f"{r.liveness:8} {r.kind.value:6} {pids:12} {r.size:>10} {r.age_seconds:>7.0f}s  "
            f"{r.session_id[:8]}  {r.project or ''}"
        )
    return 0


def cmd_handles(args: argparse.Namespace) -> int:
    if args.watch:
        tracker = HandleTracker()
        print("watching agent handles (ctrl-c to stop)")
        try:
            for ev in tracker.watch(args.interval):
                print(f"{time.strftime('%H:%M:%S', time.localtime(ev.at))} {ev.kind:6} "
                      f"pid={ev.handle.pid} {ev.handle.process}  {ev.handle.path}")
        except KeyboardInterrupt:
            return 0
    snap = snapshot()
    for h in sorted(snap.handles, key=lambda h: (h.pid, h.path)):
        print(f"pid={h.pid:<7} {h.process:12} {h.path}")
    if snap.unreadable_pids:
        print(f"(handle tables unreadable for pids: {snap.unreadable_pids})", file=sys.stderr)
    if not snap.handles:
        print("no agent process holds a session file open", file=sys.stderr)
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    ref = find_session(args.session)
    if ref is None:
        print(f"no unique session matches {args.session!r}", file=sys.stderr)
        return 2
    print(f"following {ref.path} ({ref.liveness}, pids={ref.pids})", file=sys.stderr)
    with MappedSession(ref.path) as mapped:
        start = 0 if args.from_start else mapped.size
        try:
            for rec in mapped.follow(start, poll=args.interval):
                if args.raw:
                    sys.stdout.buffer.write(rec.raw + b"\n")
                    sys.stdout.flush()
                else:
                    try:
                        obj = rec.json()
                    except json.JSONDecodeError:
                        continue
                    print(f"@{rec.offset:<10} {obj.get('type', '?')}")
        except KeyboardInterrupt:
            return 0
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    mirror = Mirror(args.mirror, kinds=_kinds(args.kind), include_recent=args.include_recent)
    if args.once:
        result = mirror.sync()
        print(json.dumps(result, indent=2))
        return 0
    print(f"serving live sessions into {mirror.root} every {args.interval}s (ctrl-c to stop)")
    try:
        while True:
            result = mirror.sync()
            if result["written"] or result["retired"]:
                stamp = time.strftime("%H:%M:%S")
                print(f"{stamp} written={len(result['written'])} retired={len(result['retired'])}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="memory_condense.live_sessions")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover", help="list session files with liveness")
    d.add_argument("--kind", default="all", help="claude, codex, or all")
    d.add_argument("--live", action="store_true", help="only live sessions")
    d.add_argument("--no-handles", action="store_true", help="skip handle enumeration")
    d.add_argument("--json", action="store_true")
    d.add_argument("--limit", type=int, default=30)
    d.set_defaults(fn=cmd_discover)

    h = sub.add_parser("handles", help="show which agent processes hold session files open")
    h.add_argument("--watch", action="store_true", help="stream open/close events")
    h.add_argument("--interval", type=float, default=2.0)
    h.set_defaults(fn=cmd_handles)

    w = sub.add_parser("watch", help="follow one session via mmap")
    w.add_argument("session", help="path, session id, or unique id prefix")
    w.add_argument("--from-start", action="store_true")
    w.add_argument("--raw", action="store_true", help="emit raw JSONL lines")
    w.add_argument("--interval", type=float, default=0.25)
    w.set_defaults(fn=cmd_watch)

    s = sub.add_parser("serve", help="keep a mirror directory in sync with live sessions")
    s.add_argument("mirror")
    s.add_argument("--kind", default="all")
    s.add_argument("--interval", type=float, default=2.0)
    s.add_argument("--once", action="store_true")
    s.add_argument("--include-recent", action="store_true",
                   help="also serve recently-modified sessions with no open handle")
    s.set_defaults(fn=cmd_serve)

    i = sub.add_parser("info", help="environment capabilities")
    i.set_defaults(fn=lambda a: print(json.dumps({"projfs": projfs_available()})) or 0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)
