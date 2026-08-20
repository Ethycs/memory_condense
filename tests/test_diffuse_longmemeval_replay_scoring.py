from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from memory_condense.eval.diffuse_longmemeval_replay import ReplayExecutionIdentity
from tools.score_diffuse_longmemeval_shared_base_replay import (
    _certify_historical_launcher,
)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_historical_launcher_certification_allows_unrelated_descendant_head(
    tmp_path: Path,
) -> None:
    root = tmp_path / "historical-launcher-repo"
    root.mkdir()
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "scorer-test@example.invalid")
    _git(root, "config", "user.name", "Scorer Test")
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "config", "commit.gpgsign", "false")
    launcher = root / "tools" / "frozen_launcher.py"
    launcher.parent.mkdir()
    launcher_bytes = b'"""Frozen launcher fixture."""\nVALUE = 1\n'
    launcher.write_bytes(launcher_bytes)
    _git(root, "add", "--", "tools/frozen_launcher.py")
    _git(root, "commit", "--quiet", "-m", "add frozen launcher")
    source_commit = _git(root, "rev-parse", "HEAD")

    unrelated = root / "notes.md"
    unrelated.write_bytes(b"later unrelated commit\n")
    _git(root, "add", "--", "notes.md")
    _git(root, "commit", "--quiet", "-m", "unrelated descendant")
    assert _git(root, "rev-parse", "HEAD") != source_commit

    expected = ReplayExecutionIdentity(
        launcher_sha256=hashlib.sha256(launcher_bytes).hexdigest(),
        source_commit=source_commit,
        tracked_worktree_clean=True,
    )
    assert _certify_historical_launcher(expected, launcher) == expected

    launcher.write_bytes(launcher_bytes + b"# drift\n")
    with pytest.raises(RuntimeError, match="current launcher bytes differ"):
        _certify_historical_launcher(expected, launcher)
