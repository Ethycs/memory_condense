"""Small strict artifact boundary for the matched-eval tool package."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import MatchedEvalContractError, canonical_json_bytes


class SealedArtifactError(MatchedEvalContractError):
    """Raised when a canonical artifact or digest sidecar is not exact."""


@dataclass(frozen=True, slots=True)
class SealedArtifact:
    path: Path
    sha256: str
    payload: dict[str, Any]


def _sidecar_bytes(path: Path, sha256: str) -> bytes:
    return f"{sha256}  {path.name}\n".encode("ascii")


def read_sealed_json(path: str | Path) -> SealedArtifact:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise SealedArtifactError(f"artifact must be a regular file: {target}")
    raw = target.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SealedArtifactError(f"artifact is not strict JSON: {target}") from exc
    if type(payload) is not dict or raw != canonical_json_bytes(payload):
        raise SealedArtifactError(f"artifact is not canonical JSON: {target}")
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = target.with_name(target.name + ".sha256")
    if (
        sidecar.is_symlink()
        or not sidecar.is_file()
        or sidecar.read_bytes() != _sidecar_bytes(target, digest)
    ):
        raise SealedArtifactError(f"artifact digest sidecar is invalid: {sidecar}")
    return SealedArtifact(path=target, sha256=digest, payload=payload)


def publish_sealed_json(
    path: str | Path,
    payload: dict[str, Any],
) -> tuple[SealedArtifact, bool]:
    """Publish once, or reuse an already byte-identical sealed artifact.

    A different existing payload is never overwritten.  The boolean is true
    only when this call created the artifact.
    """

    target = Path(path)
    raw = canonical_json_bytes(payload)
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = target.with_name(target.name + ".sha256")
    if target.exists() or sidecar.exists():
        existing = read_sealed_json(target)
        if existing.sha256 != digest:
            raise SealedArtifactError(
                f"refusing to replace a different sealed artifact: {target}"
            )
        return existing, False

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    try:
        for destination, content in (
            (target, raw),
            (sidecar, _sidecar_bytes(target, digest)),
        ):
            handle, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=target.parent
            )
            temporary = Path(temporary_name)
            temporary_paths.append(temporary)
            with os.fdopen(handle, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            temporary_paths.remove(temporary)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)

    return read_sealed_json(target), True
