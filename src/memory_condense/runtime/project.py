"""Where one project's memory lives.

The proxy, the MCP server and the status resolver each used to answer this
question for themselves, and they disagreed: the MCP server kept a
per-project store at ``<cwd>/.memory_condense`` while the proxy kept one
global store at ``user_directory()/data``, so ``memory_reset`` erased a
different store than ``recall`` read. This module is the single answer.

Resolution order (:func:`project_data_dir`):

1. ``$MEMORY_CONDENSE_DATA_DIR`` -- an explicit override always wins.
2. ``<project root>/.memory_condense`` -- per project, which is the default.
   The root is ``$CLAUDE_PROJECT_DIR`` when the client sets it, else the
   nearest ancestor of the working directory holding a project marker.
3. ``user_directory()/data`` -- the session's home, when there is no project
   at all (a bare shell in a temp directory, say).

Finding the root by **walking up** rather than trusting the working
directory matters: the previous ``Path.cwd()/".memory_condense"`` meant a
process launched from ``src/`` silently used a different store than one
launched from the repository root, splitting one project's memory across
however many subdirectories it was started from.

``.git`` is preferred over a packaging file so a monorepo subpackage shares
its repository's memory instead of starting its own; ``pyproject.toml`` is
the fallback for a project that is not a git checkout.

**Stdlib only, and no ``runtime.config`` import at module scope.** The
statusline imports this on every refresh, where ``runtime.config`` costs a
measured ~23 ms extra (it pulls ``platform``, and ``_wmi`` on Windows).
:func:`home_data_dir` imports it lazily, the same discipline
``control/status.py`` already follows for its last candidate.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

#: The per-project directory name. Already in this repository's ``.gitignore``.
DIRNAME = ".memory_condense"

#: Strongest first: a repository root beats a packaging file (see module docs).
PROJECT_MARKERS = (".git", "pyproject.toml")

DATA_DIR_ENV = "MEMORY_CONDENSE_DATA_DIR"
PROJECT_DIR_ENV = "CLAUDE_PROJECT_DIR"


def _ancestors(start: Path) -> tuple[Path, ...]:
    return (start, *start.parents)


def project_root(cwd: Path | str | None = None, env: Mapping[str, str] | None = None) -> Path | None:
    """The project a working directory belongs to, or ``None`` if there is none.

    ``$CLAUDE_PROJECT_DIR`` wins when the client sets it — it is the client
    telling us which project this session is, which beats any inference.
    Otherwise each marker is tried over the whole ancestor chain in
    :data:`PROJECT_MARKERS` order, so ``.git`` anywhere above wins over a
    ``pyproject.toml`` closer to hand.
    """
    env = os.environ if env is None else env
    explicit = env.get(PROJECT_DIR_ENV)
    if explicit and explicit.strip():
        return Path(explicit).expanduser()
    try:
        here = (Path.cwd() if cwd is None else Path(cwd)).expanduser().resolve()
    except OSError:  # a deleted or unreachable working directory
        return None
    chain = _ancestors(here)
    for marker in PROJECT_MARKERS:
        for directory in chain:
            # .git is a directory in a normal checkout and a file in a
            # worktree, so each worktree correctly gets its own store.
            if (directory / marker).exists():
                return directory
    return None


def home_data_dir() -> Path:
    """The no-project fallback: the session's own user data directory."""
    from memory_condense.runtime.config import user_directory

    return user_directory() / "data"


def _existing_local(cwd: Path | str | None) -> Path | None:
    """A populated ``./.memory_condense`` from before the project-root rule.

    The MCP server's documented default used to be ``<cwd>/.memory_condense``
    with no project check, so a store can legitimately exist in a directory
    that has no marker. Adopting it beats silently opening an empty store in
    the session home and looking like data loss. Deliberately narrow: only
    when no project was found *and* the directory is already there, never as
    a way of creating one.
    """
    try:
        local = (Path.cwd() if cwd is None else Path(cwd)).expanduser() / DIRNAME
    except OSError:
        return None
    return local if local.is_dir() else None


def project_data_dir(env: Mapping[str, str] | None = None, cwd: Path | str | None = None) -> Path:
    """The data directory for this project; see the module docstring."""
    env = os.environ if env is None else env
    explicit = env.get(DATA_DIR_ENV)
    if explicit and explicit.strip():
        return Path(explicit).expanduser()
    root = project_root(cwd, env)
    if root is not None:
        # One project, one store, always at the root -- never conditional on
        # what a subdirectory happens to contain. A store left in a
        # subdirectory by the old ``Path.cwd()`` rule is therefore not found
        # from the root, and that migration is called out in the release
        # notes rather than papered over: adopting it would make the answer
        # depend on the filesystem, so neither a user nor this code could say
        # which store a project uses without going and looking.
        return root / DIRNAME
    # No root to prefer, so adopting is the only way to find anything at all,
    # and there is nothing for it to be ambiguous against.
    return _existing_local(cwd) or home_data_dir()


def describe(env: Mapping[str, str] | None = None, cwd: Path | str | None = None) -> str:
    """One line saying which rule chose the directory, for logs and ``--print-env``."""
    env = os.environ if env is None else env
    if (env.get(DATA_DIR_ENV) or "").strip():
        return f"{project_data_dir(env, cwd)} (from ${DATA_DIR_ENV})"
    root = project_root(cwd, env)
    if root is None:
        local = _existing_local(cwd)
        if local is not None:
            return f"{local} (no project found; adopting an existing local store)"
        return f"{home_data_dir()} (no project found; session home)"
    if (env.get(PROJECT_DIR_ENV) or "").strip():
        return f"{root / DIRNAME} (project from ${PROJECT_DIR_ENV})"
    return f"{root / DIRNAME} (project root of {Path(cwd).expanduser() if cwd else Path.cwd()})"


__all__ = [
    "DATA_DIR_ENV",
    "DIRNAME",
    "PROJECT_DIR_ENV",
    "PROJECT_MARKERS",
    "describe",
    "home_data_dir",
    "project_data_dir",
    "project_root",
]
