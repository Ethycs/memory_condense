"""Provider-free source-closure tests for the production-candidate launcher."""

from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tools import run_diffuse_latent_training_corpus as launcher


def _source_tree(root: Path) -> Path:
    candidate = root / "tools" / "run_diffuse_latent_training_corpus.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("# launcher\n", encoding="utf-8")
    package = root / "src" / "memory_condense"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# package\n", encoding="utf-8")
    return candidate


def _git_runner(root: Path, tracked: tuple[str, ...]):
    def run(arguments, **kwargs):
        command = tuple(arguments)
        if command[1:3] == ("rev-parse", "--show-toplevel"):
            return SimpleNamespace(returncode=0, stdout=f"{root}\n")
        if command[1:3] == ("ls-files", "-z"):
            assert kwargs["text"] is False
            payload = b"\0".join(value.encode("utf-8") for value in tracked) + b"\0"
            return SimpleNamespace(returncode=0, stdout=payload)
        if command[1:3] == ("status", "--porcelain=v1"):
            return SimpleNamespace(returncode=0, stdout="")
        raise AssertionError(f"unexpected git command: {command!r}")

    return run


def test_source_certification_requires_exact_tracked_python_membership(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = _source_tree(tmp_path)
    execution = object()
    tracked = (
        "src/memory_condense/__init__.py",
        "tools/run_diffuse_latent_training_corpus.py",
    )
    monkeypatch.setattr(subprocess, "run", _git_runner(tmp_path, tracked))

    assert launcher._certify_source_execution(
        candidate,
        certifier=lambda _path: execution,
    ) is execution

    (tmp_path / "src" / "memory_condense" / "ignored.py").write_text(
        "raise RuntimeError('ignored code entered the package')\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="including ignored code"):
        launcher._certify_source_execution(
            candidate,
            certifier=lambda _path: execution,
        )


def test_source_certification_rejects_a_tracked_python_membership_gap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = _source_tree(tmp_path)
    tracked = (
        "src/memory_condense/__init__.py",
        "src/memory_condense/missing.py",
        "tools/run_diffuse_latent_training_corpus.py",
    )
    monkeypatch.setattr(subprocess, "run", _git_runner(tmp_path, tracked))

    with pytest.raises(RuntimeError, match="tracked files"):
        launcher._certify_source_execution(candidate, certifier=lambda _path: object())


def test_source_certification_rejects_ignored_uppercase_python(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = _source_tree(tmp_path)
    (tmp_path / "tools" / "ignored.PY").write_text(
        "raise RuntimeError('uppercase ignored code entered the launcher')\n",
        encoding="utf-8",
    )
    tracked = (
        "src/memory_condense/__init__.py",
        "tools/run_diffuse_latent_training_corpus.py",
    )
    monkeypatch.setattr(subprocess, "run", _git_runner(tmp_path, tracked))

    with pytest.raises(RuntimeError, match="including ignored code"):
        launcher._certify_source_execution(candidate, certifier=lambda _path: object())


def test_source_certification_rejects_a_linked_source_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = _source_tree(tmp_path)
    linked_target = tmp_path / "outside-source"
    linked_target.mkdir()
    (linked_target / "ignored.py").write_text("# linked code\n", encoding="utf-8")
    linked = tmp_path / "tools" / "linked"
    try:
        linked.symlink_to(linked_target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    tracked = (
        "src/memory_condense/__init__.py",
        "tools/run_diffuse_latent_training_corpus.py",
    )
    monkeypatch.setattr(subprocess, "run", _git_runner(tmp_path, tracked))

    with pytest.raises(RuntimeError, match="linked directory"):
        launcher._certify_source_execution(candidate, certifier=lambda _path: object())


def test_source_certification_rejects_an_unreadable_source_subtree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = _source_tree(tmp_path)
    tracked = (
        "src/memory_condense/__init__.py",
        "tools/run_diffuse_latent_training_corpus.py",
    )
    monkeypatch.setattr(subprocess, "run", _git_runner(tmp_path, tracked))

    def unreadable_walk(*_args, onerror, **_kwargs):
        onerror(PermissionError("injected unreadable source directory"))
        return ()

    monkeypatch.setattr(launcher.os, "walk", unreadable_walk)

    with pytest.raises(RuntimeError, match="cannot be enumerated"):
        launcher._certify_source_execution(candidate, certifier=lambda _path: object())
