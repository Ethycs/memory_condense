"""Small, dependency-free primitives for fail-closed artifact verification."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class FirebreakError(ValueError):
    """An artifact cannot support the evaluator-firebreak claim."""


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    path: Path
    payload: bytes
    sha256: str
    size: int
    identity: tuple[int, int, int, int]


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FirebreakError("value is not canonical JSON") from exc


def canonical_sha256(value: Any) -> str:
    return bytes_sha256(canonical_json_bytes(value))


def read_snapshot(path: str | Path, label: str) -> FileSnapshot:
    target = Path(path).resolve()
    try:
        before = target.stat()
        payload = target.read_bytes()
        after = target.stat()
    except OSError as exc:
        raise FirebreakError(f"cannot snapshot {label}") from exc
    before_id = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
    )
    after_id = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
    )
    if before_id != after_id or len(payload) != int(after.st_size):
        raise FirebreakError(f"{label} changed while being snapshotted")
    return FileSnapshot(
        path=target,
        payload=payload,
        sha256=bytes_sha256(payload),
        size=len(payload),
        identity=after_id,
    )


def assert_snapshot_unchanged(snapshot: FileSnapshot, label: str) -> None:
    try:
        stat = snapshot.path.stat()
        current_id = (
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
        )
        digest = hashlib.sha256()
        with snapshot.path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise FirebreakError(f"cannot recheck {label}") from exc
    if current_id != snapshot.identity or digest.hexdigest() != snapshot.sha256:
        raise FirebreakError(f"{label} changed during verification")


def parse_json_bytes(payload: bytes, label: str) -> Any:
    try:
        return json.loads(
            payload,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FirebreakError(f"cannot parse {label} as strict JSON") from exc


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise FirebreakError(f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise FirebreakError(f"{label} must be a JSON array")
    return value


def require_text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise FirebreakError(f"{label} must be text")
    return value


def require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FirebreakError(f"{label} must be an integer >= {minimum}")
    return value


def require_sha256(value: Any, label: str) -> str:
    text = require_text(value, label)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise FirebreakError(f"{label} must be a lowercase SHA-256")
    return text


def exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise FirebreakError(f"{label} has a non-closed schema")


def package_sha256(root: str | Path) -> str:
    package = Path(root).resolve()
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py"), key=lambda item: item.as_posix()):
        relative = path.relative_to(package).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def publish_no_clobber(path: str | Path, payload: bytes) -> None:
    target = Path(path).resolve()
    try:
        with target.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise FirebreakError("refusing to overwrite an existing receipt") from exc
    except OSError as exc:
        raise FirebreakError("cannot publish receipt") from exc


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key")
        value[key] = item
    return value
