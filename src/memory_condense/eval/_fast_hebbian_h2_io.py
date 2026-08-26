"""Strict canonical input helpers for the provider-free Hebbian H2 layer."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class FastHebbianH2ValidationError(ValueError):
    """Raised when an H2 input or receipt cannot prove its exact identity."""


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def read_canonical_json(path: str | Path) -> tuple[dict[str, Any], str, Path]:
    candidate = Path(path)
    if candidate.is_symlink():
        raise FastHebbianH2ValidationError("artifact path must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FastHebbianH2ValidationError("artifact path does not exist") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise FastHebbianH2ValidationError("artifact path must be a regular file")
    raw = resolved.read_bytes()

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value {value}")

    try:
        payload = json.loads(raw, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FastHebbianH2ValidationError("artifact is not strict JSON") from exc
    if type(payload) is not dict or raw != _canonical_json_bytes(payload):
        raise FastHebbianH2ValidationError("artifact is not canonical JSON")
    return payload, hashlib.sha256(raw).hexdigest(), resolved


def verify_digest_anchor(
    path: Path,
    digest: str,
    *,
    expected_sha256: str | None,
    verify_sidecar: bool,
) -> None:
    if expected_sha256 is not None:
        if (
            type(expected_sha256) is not str
            or _DIGEST_RE.fullmatch(expected_sha256) is None
        ):
            raise FastHebbianH2ValidationError(
                "expected_sha256 must be a lowercase SHA-256 digest"
            )
        if expected_sha256 != digest:
            raise FastHebbianH2ValidationError("artifact digest changed")
    if not verify_sidecar:
        if expected_sha256 is None:
            raise FastHebbianH2ValidationError(
                "disabling sidecar verification requires an explicit digest"
            )
        return
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.is_symlink() or not sidecar.is_file():
        raise FastHebbianH2ValidationError("artifact digest sidecar is missing")
    expected = f"{digest}  {path.name}\n".encode("ascii")
    if sidecar.read_bytes() != expected:
        raise FastHebbianH2ValidationError("artifact digest sidecar changed")
