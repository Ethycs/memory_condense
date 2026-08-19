"""Canonical encodings and race-detecting file snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class AuditError(ValueError):
    """An input cannot support a fail-closed audit receipt."""


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    """One exact byte image used for both parsing and identity."""

    path: Path
    payload: bytes
    sha256: str


def read_file_snapshot(path: str | Path, label: str) -> FileSnapshot:
    """Read one file once and bind all later parsing to those exact bytes."""

    target = Path(path).resolve()
    try:
        before = target.stat()
        payload = target.read_bytes()
        after = target.stat()
    except OSError as exc:
        raise AuditError(f"cannot snapshot {label} {target}: {exc}") from exc
    before_identity = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
    )
    after_identity = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
    )
    if before_identity != after_identity or len(payload) != int(after.st_size):
        raise AuditError(f"{label} changed while it was being snapshotted: {target}")
    return FileSnapshot(target, payload, bytes_sha256(payload))


def assert_file_snapshot_unchanged(snapshot: FileSnapshot, label: str) -> None:
    if file_sha256(snapshot.path) != snapshot.sha256:
        raise AuditError(f"{label} changed after its byte snapshot: {snapshot.path}")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON using the repository's stable hash representation."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AuditError(f"value is not canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise AuditError(f"cannot hash {Path(path)}: {exc}") from exc
    return digest.hexdigest()


def parse_json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AuditError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"{label} must be a JSON object")
    return value


def load_json_object(path: str | Path, label: str) -> tuple[dict[str, Any], bytes]:
    snapshot = read_file_snapshot(path, label)
    return parse_json_object(snapshot.payload, label), snapshot.payload


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is forbidden")


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise AuditError(f"{label} must be a lowercase SHA-256 digest")
    if len(value) != 64 or value != value.casefold() or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise AuditError(f"{label} must be a lowercase SHA-256 digest")
    return value


def require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AuditError(f"{label} must be an integer >= {minimum}")
    return value


def require_number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditError(f"{label} must be a finite number")
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        raise AuditError(f"{label} must be a finite number")
    if minimum is not None and result < minimum:
        raise AuditError(f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise AuditError(f"{label} must be <= {maximum}")
    return result


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise AuditError(f"{label} must be a JSON object with string keys")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise AuditError(f"{label} must be a JSON array")
    return value


def require_text(value: Any, label: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        suffix = "non-empty " if nonempty else ""
        raise AuditError(f"{label} must be a {suffix}string")
    return value


def length_prefixed_digest(rows: Iterable[Iterable[Any]]) -> str:
    """Hash typed rows without materializing a large canonical JSON array."""

    digest = hashlib.sha256()
    for row in rows:
        payload = canonical_json_bytes(list(row))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def tree_snapshot(root: str | Path) -> dict[str, dict[str, int | str]]:
    """Hash every ordinary file below ``root`` and reject link indirection."""

    base = Path(root).resolve()
    if not base.is_dir():
        raise AuditError(f"cache root is not a directory: {base}")
    _reject_hidden_filesystem_state(base)
    result: dict[str, dict[str, int | str]] = {}
    try:
        for directory, directory_names, file_names in os.walk(base):
            directory_path = Path(directory)
            for name in [*directory_names, *file_names]:
                candidate = directory_path / name
                if _is_link_or_reparse_point(candidate):
                    raise AuditError(
                        f"cache roots may not contain links or reparse points: {candidate}"
                    )
                _reject_hidden_filesystem_state(candidate)
            for name in sorted(file_names):
                candidate = directory_path / name
                if not candidate.is_file():
                    raise AuditError(f"cache entry is not an ordinary file: {candidate}")
                relative = candidate.relative_to(base).as_posix()
                stat = candidate.stat()
                result[relative] = {
                    "size": int(stat.st_size),
                    "sha256": file_sha256(candidate),
                }
    except OSError as exc:
        raise AuditError(f"cannot snapshot cache root {base}: {exc}") from exc
    return dict(sorted(result.items()))


def _is_link_or_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(os.path, "isjunction", None)
    if callable(is_junction) and is_junction(path):
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError as exc:
        raise AuditError(f"cannot inspect cache entry {path}: {exc}") from exc
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse_flag and attributes & reparse_flag)


def _reject_hidden_filesystem_state(path: Path) -> None:
    """Reject xattrs and non-default NTFS streams from the closed-world scope."""

    listxattr = getattr(os, "listxattr", None)
    if callable(listxattr):
        try:
            attributes = listxattr(path, follow_symlinks=False)
        except (OSError, TypeError) as exc:
            raise AuditError(f"cannot enumerate extended attributes for {path}: {exc}") from exc
        if attributes:
            raise AuditError(f"cache entry has unapproved extended attributes: {path}")
    if os.name == "nt":
        streams = _windows_alternate_streams(path)
        extras = [stream for stream in streams if stream != "::$DATA"]
        if extras:
            raise AuditError(
                f"cache entry has unapproved NTFS alternate data streams: {path}"
            )


def _windows_alternate_streams(path: Path) -> tuple[str, ...]:
    """Enumerate NTFS streams without shelling out (Windows only)."""

    if os.name != "nt":
        return ()
    import ctypes
    from ctypes import wintypes

    class _StreamData(ctypes.Structure):
        _fields_ = [
            ("StreamSize", ctypes.c_longlong),
            ("cStreamName", wintypes.WCHAR * 296),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    find_first = kernel32.FindFirstStreamW
    find_first.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
    find_first.restype = wintypes.HANDLE
    find_next = kernel32.FindNextStreamW
    find_next.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    find_next.restype = wintypes.BOOL
    find_close = kernel32.FindClose
    find_close.argtypes = [wintypes.HANDLE]
    find_close.restype = wintypes.BOOL

    data = _StreamData()
    handle = find_first(str(path), 0, ctypes.byref(data), 0)
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid:
        error = ctypes.get_last_error()
        if error in {2, 38}:  # file not found / no more files
            return ()
        raise AuditError(f"cannot enumerate NTFS streams for {path}: winerror {error}")
    streams: list[str] = []
    try:
        streams.append(str(data.cStreamName))
        while find_next(handle, ctypes.byref(data)):
            streams.append(str(data.cStreamName))
        error = ctypes.get_last_error()
        if error not in {0, 38}:
            raise AuditError(
                f"cannot finish NTFS stream enumeration for {path}: winerror {error}"
            )
    finally:
        find_close(handle)
    return tuple(streams)


def snapshot_receipt(
    root: str | Path, snapshot: dict[str, dict[str, int | str]]
) -> dict[str, Any]:
    return {
        "root": str(Path(root).resolve()),
        "file_count": len(snapshot),
        "total_bytes": sum(int(row["size"]) for row in snapshot.values()),
        "tree_sha256": canonical_sha256(snapshot),
    }


def package_sha256(root: str | Path) -> str:
    """Hash Python files with path and length framing, like frozen source."""

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


def validate_output_location(
    output: str | Path,
    *,
    protected_roots: Iterable[str | Path],
    protected_files: Iterable[str | Path],
) -> Path:
    """Resolve an intended receipt path and keep it outside audited state."""

    raw = Path(output)
    if os.name == "nt" and ":" in raw.name:
        raise AuditError("receipt output may not use an NTFS alternate stream")
    target = raw.resolve(strict=False)
    for file_path in protected_files:
        if target == Path(file_path).resolve():
            raise AuditError(f"receipt output would replace an audited input: {target}")
    for root_path in protected_roots:
        root = Path(root_path).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            continue
        raise AuditError(f"receipt output must be outside protected root {root}")
    return target
