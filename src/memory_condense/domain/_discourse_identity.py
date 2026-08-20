"""Canonical JSON, immutable metadata, and digest primitives for discourse."""

from __future__ import annotations

import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class _IdentityPayload(Protocol):
    def identity_payload(self) -> Mapping[str, Any]: ...


def _plain_json(value: Any) -> Any:
    """Return detached JSON containers from recursively frozen values."""

    if isinstance(value, Mapping):
        return {str(key): _plain_json(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(child) for child in value]
    return value


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(child) for key, child in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json(child) for child in value)
    return value


def canonical_json(value: Any) -> str:
    """Return the single JSON encoding used by every closure identity."""

    return json.dumps(
        _plain_json(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def identity_sha256(value: Any) -> str:
    """Hash a strict canonical JSON value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def quote_sha256(text: str) -> str:
    """Hash one exact source span without normalizing its bytes."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _nonempty(value: str, label: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{label} must be non-empty")
    return normalized


def _sha256(value: str, label: str) -> str:
    normalized = str(value).lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def _nonnegative(value: Any, label: str) -> Any:
    """Reject values below zero; the value itself is returned untouched."""

    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _positive(value: Any, label: str) -> Any:
    """Reject values below one; the value itself is returned untouched."""

    if value < 1:
        raise ValueError(f"{label} must be positive")
    return value


def _finite(value: Any, label: str) -> Any:
    """Reject non-finite numbers; the value itself is returned untouched."""

    if not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    return value


def _strict_int(value: Any, label: str) -> Any:
    """Require a true ``int`` (bools rejected); no coercion is applied."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _choice(
    allowed: frozenset[str],
    message: str,
) -> Callable[[Any, str], str]:
    """Return a validator for a non-empty value from a closed vocabulary.

    ``message`` is raised verbatim on values outside ``allowed`` so every
    call site keeps its exact pinned error text.
    """

    def normalize(value: Any, label: str) -> str:
        normalized = _nonempty(value, label)
        if normalized not in allowed:
            raise ValueError(message)
        return normalized

    return normalize


def _labeled(
    label: str,
    normalizer: Callable[[Any, str], Any],
) -> Callable[[Any, str], Any]:
    """Bind a fixed error label, for fields whose label is not the field name."""

    def normalize(value: Any, _field_name: str) -> Any:
        return normalizer(value, label)

    return normalize


def _as_tuple(values: Any, label: str) -> tuple[Any, ...]:
    """Freeze a sequence field into a tuple without validating its items."""

    return tuple(values)


def _unique_nonempty(values: Any, label: str) -> tuple[str, ...]:
    """Deduplicate non-empty strings, preserving first-seen order."""

    return tuple(dict.fromkeys(_nonempty(value, label) for value in values))


def _sorted_unique(values: Any, label: str) -> tuple[Any, ...]:
    """Deduplicate values and return them in sorted order."""

    return tuple(sorted(dict.fromkeys(values)))


def _sorted_unique_nonempty(values: Any, label: str) -> tuple[str, ...]:
    """Deduplicate non-empty strings and return them in sorted order."""

    return tuple(sorted({_nonempty(value, label) for value in values}))


def exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    """Validate an exactly-integral number, optionally bounded below.

    The single spelling behind the former ``_exact_int`` / ``_positive_int`` /
    ``_nonnegative_int`` / ``_positive_integer`` / ``_exact_nonnegative_int``
    copies.  Booleans are rejected; floats must be exactly integral.
    """

    if minimum is None:
        message = f"{label} must be an integer"
    elif minimum == 0:
        message = f"{label} must be a non-negative integer"
    elif minimum == 1:
        message = f"{label} must be a positive integer"
    else:
        message = f"{label} must be at least {minimum}"
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        normalized = int(value)
        exact = math.isfinite(float(value)) and float(value) == normalized
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not exact or (minimum is not None and normalized < minimum):
        raise ValueError(message)
    return normalized


def _confidence(value: float, label: str = "confidence") -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{label} must be finite and inside [0, 1]")
    return normalized


def _optional(
    normalizer: Callable[[Any, str], Any],
) -> Callable[[Any, str], Any]:
    """Wrap a normalizer so ``None`` passes through untouched."""

    def normalize(value: Any, label: str) -> Any:
        return None if value is None else normalizer(value, label)

    return normalize


def normalize_fields(obj: Any, **normalizers: Callable[[Any, str], Any]) -> None:
    """Rebind each named field to ``normalizer(value, field_name)`` in place.

    Only for fields whose error label is the field name itself; labeled or
    cross-field validation stays hand-written at the call site.
    """

    for name, normalizer in normalizers.items():
        object.__setattr__(obj, name, normalizer(getattr(obj, name), name))


def _json_mapping(value: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    try:
        encoded = canonical_json(dict(value))
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be strict JSON") from exc
    if not isinstance(decoded, dict):  # pragma: no cover - guarded by dict()
        raise ValueError(f"{label} must be a JSON object")
    return _freeze_json(decoded)


def make_episode_id(
    *,
    artifact_id: str,
    source_id: str,
    sequence_no: int,
    evidence: Sequence[_IdentityPayload],
) -> str:
    body = {
        "artifact_id": artifact_id,
        "source_id": source_id,
        "sequence_no": int(sequence_no),
        "evidence": [item.identity_payload() for item in evidence],
    }
    return f"episode-{identity_sha256(body)[:24]}"


def make_atom_id(span: _IdentityPayload) -> str:
    return f"atom-{identity_sha256(span.identity_payload())[:24]}"


def make_bundle_id(
    *,
    atom_ids: Sequence[str],
    obligation_ids: Sequence[str],
    unit_ids: Sequence[str] = (),
    relation_ids: Sequence[str] = (),
) -> str:
    body = {
        "atom_ids": list(dict.fromkeys(atom_ids)),
        "obligation_ids": list(dict.fromkeys(obligation_ids)),
        "unit_ids": list(dict.fromkeys(unit_ids)),
        "relation_ids": list(dict.fromkeys(relation_ids)),
    }
    return f"bundle-{identity_sha256(body)[:24]}"


__all__ = [
    "canonical_json",
    "exact_int",
    "identity_sha256",
    "make_atom_id",
    "make_bundle_id",
    "make_episode_id",
    "quote_sha256",
]
