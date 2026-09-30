"""Track which agent processes hold which session files open.

An open handle is the ground truth for "this session is live": a Claude or
Codex process keeps its transcript ``.jsonl`` open for append for the life of
the session, and closes it when the session ends. :class:`HandleTracker` walks
the handle tables of agent processes (via psutil -- a read-only system query,
no injection) and turns successive snapshots into open/close events. The live
set it maintains is what the virtual layer serves.

What this module deliberately does not do: rewrite, duplicate, or hook the
handles inside the owning process. That would be DLL injection into a process
mid-append to the file we care about; the failure mode is a corrupted live
session. Read-side interposition (``mmap_view``) plus this tracker gives the
same information without that risk.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Iterable, Iterator

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a declared dependency
    psutil = None  # type: ignore[assignment]

AGENT_EXECUTABLES = frozenset({"claude", "claude.exe", "codex", "codex.exe"})
HOST_RUNTIMES = frozenset({"node", "node.exe", "bun", "bun.exe", "deno", "deno.exe"})
AGENT_MARKERS = ("claude", "codex")
SESSION_SUFFIXES = (".jsonl",)


def normalize(path: str) -> str:
    """Canonical key for a path so handle paths and glob paths compare equal."""
    return os.path.normcase(os.path.realpath(path))


@dataclass(frozen=True, slots=True)
class OpenHandle:
    pid: int
    process: str
    path: str  # normalized


@dataclass(frozen=True, slots=True)
class AgentProcess:
    """A running agent, with the cwd it was launched in (None if unreadable).

    Claude Code does not keep its transcript open between appends, so its
    sessions cannot be found through handles at all; the cwd is the link
    instead (see ``discovery.encode_project``).
    """

    pid: int
    name: str
    cwd: str | None


@dataclass(frozen=True, slots=True)
class HandleEvent:
    kind: str  # "opened" | "closed"
    handle: OpenHandle
    at: float


@dataclass(slots=True)
class HandleSnapshot:
    at: float
    handles: frozenset[OpenHandle]
    # Processes whose handle table could not be read (access denied etc.).
    unreadable_pids: tuple[int, ...] = ()
    # Every agent process seen, whether or not it holds a session file.
    processes: tuple[AgentProcess, ...] = ()

    def by_path(self) -> dict[str, set[int]]:
        out: dict[str, set[int]] = {}
        for h in self.handles:
            out.setdefault(h.path, set()).add(h.pid)
        return out

    def paths(self) -> frozenset[str]:
        return frozenset(h.path for h in self.handles)


def is_agent_process(proc: "psutil.Process") -> bool:
    """True for claude/codex binaries, or a JS runtime whose cmdline names one."""
    try:
        name = (proc.name() or "").lower()
    except (psutil.Error, OSError):
        return False
    if name in AGENT_EXECUTABLES:
        return True
    if name in HOST_RUNTIMES:
        try:
            cmdline = " ".join(proc.cmdline()).lower()
        except (psutil.Error, OSError):
            return False
        return any(marker in cmdline for marker in AGENT_MARKERS)
    return False


def iter_agent_processes() -> Iterator["psutil.Process"]:
    if psutil is None:
        return
    for proc in psutil.process_iter(["pid", "name"]):
        if is_agent_process(proc):
            yield proc


def snapshot(
    *,
    suffixes: Iterable[str] = SESSION_SUFFIXES,
    under: Iterable[str] = (),
) -> HandleSnapshot:
    """Enumerate session-file handles held by agent processes right now.

    ``suffixes`` filters by extension; ``under`` (normalized directory
    prefixes) restricts to known session roots when given.
    """
    suffixes = tuple(s.lower() for s in suffixes)
    roots = tuple(normalize(u) for u in under)
    found: set[OpenHandle] = set()
    unreadable: list[int] = []
    processes: list[AgentProcess] = []
    for proc in iter_agent_processes():
        try:
            pname = proc.name() or "?"
        except (psutil.Error, OSError):
            continue
        try:
            cwd: str | None = proc.cwd()
        except (psutil.Error, OSError):
            cwd = None
        processes.append(AgentProcess(pid=proc.pid, name=pname, cwd=cwd))
        try:
            for of in proc.open_files():
                low = of.path.lower()
                if suffixes and not low.endswith(suffixes):
                    continue
                path = normalize(of.path)
                if roots and not path.startswith(roots):
                    continue
                found.add(OpenHandle(pid=proc.pid, process=pname, path=path))
        except (psutil.AccessDenied, psutil.ZombieProcess):
            unreadable.append(proc.pid)
        except (psutil.Error, OSError):
            continue
    return HandleSnapshot(
        at=time.time(),
        handles=frozenset(found),
        unreadable_pids=tuple(unreadable),
        processes=tuple(processes),
    )


def writers_of(path: str) -> list[OpenHandle]:
    """Handles any agent process holds on one specific file."""
    target = normalize(path)
    return sorted(
        (h for h in snapshot().handles if h.path == target),
        key=lambda h: h.pid,
    )


@dataclass(slots=True)
class HandleTracker:
    """Diff successive handle snapshots into open/close events.

    Typical use::

        tracker = HandleTracker(under=[claude_root, codex_root])
        for event in tracker.poll():        # call on a timer
            ...
        tracker.live_paths()                # what the mirror should serve
    """

    under: tuple[str, ...] = ()
    suffixes: tuple[str, ...] = SESSION_SUFFIXES
    current: HandleSnapshot = field(
        default_factory=lambda: HandleSnapshot(at=0.0, handles=frozenset())
    )
    history: list[HandleEvent] = field(default_factory=list)

    def poll(self) -> list[HandleEvent]:
        new = snapshot(suffixes=self.suffixes, under=self.under)
        opened = new.handles - self.current.handles
        closed = self.current.handles - new.handles
        events = [HandleEvent("opened", h, new.at) for h in sorted(opened, key=_key)]
        events += [HandleEvent("closed", h, new.at) for h in sorted(closed, key=_key)]
        self.current = new
        self.history.extend(events)
        return events

    def live_paths(self) -> frozenset[str]:
        return self.current.paths()

    def owners(self, path: str) -> set[int]:
        return self.current.by_path().get(normalize(path), set())

    def watch(self, interval: float = 2.0) -> Iterator[HandleEvent]:
        """Yield events forever, polling every ``interval`` seconds."""
        while True:
            yield from self.poll()
            time.sleep(interval)


def _key(h: OpenHandle) -> tuple[int, str]:
    return (h.pid, h.path)
