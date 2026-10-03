"""Small strict artifact boundary for the matched-eval tool package."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict
from memory_condense.domain._discourse_identity import identity_sha256
from dataclasses import dataclass
from pathlib import Path
from typing import Any

class MatchedEvalContractError(ValueError):
    """Raised when an evaluation-spine invariant is violated."""

def canonical_json_bytes(value: object) -> bytes:
    """Return the stable JSON representation used by the tool-only spine."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


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


def save(path, value):
    return publish_sealed_json(path, value)[0]

def read(path):
    return read_sealed_json(path).payload

def emit(**value):
    print(json.dumps(value), flush=True)

class Gateway:
    def __init__(self, root):
        self.root = Path(root)

    def call(self, kind, messages, *, scope, max_tokens=4096, typed_request=None, summary_attempt=0):
        job = dict(kind=kind, messages=messages, scope=scope, max_tokens=max_tokens,
                   typed_request=asdict(typed_request) if typed_request is not None else None)
        if summary_attempt:
            job['summary_attempt'] = summary_attempt
        key = identity_sha256(job)
        folder = self.root / 'gateway'
        request = save(folder / f'{key}.request.json', job)
        response = folder / f'{key}.response.json'
        (folder / f'{key}.ready').touch()
        deadline = time.monotonic() + 300
        while not response.with_suffix('.json.sha256').exists():
            if (self.root / 'STOP').exists():
                raise RuntimeError('Run stopped; request retained')
            if time.monotonic() > deadline:
                raise TimeoutError('Unacknowledged generation; never resend a reserved request')
            time.sleep(.025)
        result = read(response)
        if result['request_sha256'] != request.sha256:
            raise ValueError('Mismatched generation receipt')
        if result.get('error_type'):
            raise RuntimeError(f"Gateway {result['error_type']} (status {result.get('http_status')})")
        if result['finish_reason'] != 'stop':
            raise ValueError('Incomplete generation: ' + str(result['finish_reason']))
        return result
