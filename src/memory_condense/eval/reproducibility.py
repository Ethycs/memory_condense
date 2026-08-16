"""Stable fingerprints for benchmark code, environment, and artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def implementation_sha256(root: str | Path | None = None) -> str:
    """Hash every Python source file with its package-relative path."""
    package = (
        Path(root).resolve()
        if root is not None
        else project_root() / "src" / "memory_condense"
    )
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py"), key=lambda item: item.as_posix()):
        relative = path.relative_to(package).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def environment_lock_sha256(root: str | Path | None = None) -> str:
    base = Path(root).resolve() if root is not None else project_root()
    return file_sha256(base / "pixi.lock")
