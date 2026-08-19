"""Strict canonical JSON and path primitives for locked-v3 recall."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

if __package__:
    from .merge_locked_v3_recall_schema import RecallCampaignError, _SHA256_RE
else:  # Support direct execution of the facade script.
    from merge_locked_v3_recall_schema import RecallCampaignError, _SHA256_RE

def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RecallCampaignError(f"value is not canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_json(payload: bytes | str, label: str) -> Any:
    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        value = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RecallCampaignError(f"invalid {label}: {exc}") from exc
    _assert_finite_json(value, label)
    return value


def _assert_finite_json(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RecallCampaignError(f"{label} contains a non-finite number")
    if isinstance(value, list):
        for item in value:
            _assert_finite_json(item, label)
    elif isinstance(value, dict):
        for item in value.values():
            _assert_finite_json(item, label)


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RecallCampaignError(f"{label} must be a JSON object")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RecallCampaignError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _safe_relative_file(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RecallCampaignError(f"{label} must be a repository-relative file")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RecallCampaignError(f"{label} must be a safe repository-relative file")
    resolved_root = root.resolve()
    candidate = (resolved_root / relative).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise RecallCampaignError(f"{label} escapes the frozen repository") from exc
    if not candidate.is_file():
        raise RecallCampaignError(f"{label} does not exist in the frozen repository")
    return candidate
