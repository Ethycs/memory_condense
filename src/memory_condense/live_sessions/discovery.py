"""Find Claude and Codex session transcripts and decide which are live.

Liveness has three sources, in order of trust:

``handle``   an agent process holds the file open (Codex keeps its rollout
             open for the session's life; this is ground truth).
``process``  a live agent process was launched in the cwd this transcript's
             project folder encodes, and the file was appended recently.
             Claude Code appends open->write->close and never holds a handle,
             so this is the strongest attribution available for it.
``recent``   the file was appended within the live window; no owner found.

Each :class:`SessionRef` records which one applied and the owning pids, so a
consumer can serve the attributed tiers with confidence and treat
recency-only as the fallback for when process inspection is denied.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from .handles import HandleSnapshot, normalize, snapshot


class SessionKind(str, Enum):
    CLAUDE = "claude"
    CODEX = "codex"


SESSION_ROOTS: dict[SessionKind, tuple[str, ...]] = {
    SessionKind.CLAUDE: ("~/.claude/projects",),
    SessionKind.CODEX: ("~/.codex/sessions", "~/.codex/archived_sessions"),
}

DEFAULT_LIVE_WINDOW_S = 120.0

_NON_ALNUM = re.compile(r"[^A-Za-z0-9]")


def encode_project(cwd: str) -> str:
    """Claude Code's project-folder name for a working directory.

    ``F:\\Keytone\\GitHub\\memory_condense`` -> ``F--Keytone-GitHub-memory-condense``
    (every non-alphanumeric becomes ``-``; compare case-insensitively).
    """
    return _NON_ALNUM.sub("-", cwd)


@dataclass(slots=True)
class SessionRef:
    kind: SessionKind
    path: str
    key: str  # normalized path, joins with handle snapshots
    session_id: str
    project: str | None
    size: int
    mtime: float
    pids: tuple[int, ...] = ()
    liveness: str = "idle"  # "handle" | "process" | "recent" | "idle"

    @property
    def live(self) -> bool:
        return self.liveness != "idle"

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.mtime)


def expanded_roots(kinds: Iterable[SessionKind] | None = None) -> list[tuple[SessionKind, str]]:
    kinds = tuple(kinds) if kinds else tuple(SessionKind)
    out = []
    for kind in kinds:
        for root in SESSION_ROOTS[kind]:
            path = os.path.expanduser(root)
            if os.path.isdir(path):
                out.append((kind, path))
    return out


def session_id_of(path: str, kind: SessionKind) -> str:
    stem = Path(path).stem
    if kind is SessionKind.CODEX and stem.startswith("rollout-"):
        parts = stem.split("-")
        if len(parts) >= 5:  # trailing uuid after the timestamp
            return "-".join(parts[-5:])
    return stem


def project_of(path: str, kind: SessionKind, root: str) -> str | None:
    if kind is SessionKind.CLAUDE:
        rel = Path(path).relative_to(root)
        return rel.parts[0] if len(rel.parts) > 1 else None
    return None


def discover_sessions(
    kinds: Iterable[SessionKind] | None = None,
    *,
    snap: HandleSnapshot | None = None,
    use_handles: bool = True,
    live_window: float = DEFAULT_LIVE_WINDOW_S,
) -> list[SessionRef]:
    """All known session files, newest first, with liveness resolved."""
    roots = expanded_roots(kinds)
    if snap is None and use_handles:
        snap = snapshot(under=[r for _, r in roots])
    open_by_path = snap.by_path() if snap is not None else {}
    # Claude project folder (casefolded) -> pids of agents launched in that cwd.
    pids_by_project: dict[str, list[int]] = {}
    if snap is not None:
        for proc in snap.processes:
            if proc.cwd:
                pids_by_project.setdefault(encode_project(proc.cwd).casefold(), []).append(proc.pid)
    now = time.time()
    seen: set[str] = set()
    refs: list[SessionRef] = []
    for kind, root in roots:
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                if not name.endswith(".jsonl"):
                    continue
                path = os.path.join(dirpath, name)
                key = normalize(path)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                project = project_of(path, kind, root)
                fresh = now - st.st_mtime <= live_window
                pids = tuple(sorted(open_by_path.get(key, ())))
                if pids:
                    liveness = "handle"
                elif fresh and project and pids_by_project.get(project.casefold()):
                    liveness = "process"
                    pids = tuple(sorted(pids_by_project[project.casefold()]))
                elif fresh:
                    liveness = "recent"
                else:
                    liveness = "idle"
                refs.append(
                    SessionRef(
                        kind=kind,
                        path=path,
                        key=key,
                        session_id=session_id_of(path, kind),
                        project=project,
                        size=st.st_size,
                        mtime=st.st_mtime,
                        pids=pids,
                        liveness=liveness,
                    )
                )
    refs.sort(key=lambda r: r.mtime, reverse=True)
    return refs


def live_sessions(**kwargs) -> list[SessionRef]:
    return [r for r in discover_sessions(**kwargs) if r.live]


def find_session(ident: str, **kwargs) -> SessionRef | None:
    """Resolve a path, full session id, or unique id prefix to a session."""
    if os.path.isfile(ident):
        key = normalize(ident)
        for ref in discover_sessions(**kwargs):
            if ref.key == key:
                return ref
        return None
    matches = [r for r in discover_sessions(**kwargs) if r.session_id.startswith(ident)]
    return matches[0] if len(matches) == 1 else None
