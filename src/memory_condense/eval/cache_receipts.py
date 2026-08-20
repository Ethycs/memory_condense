"""Strict, text-free identities for blind-prepared benchmark caches."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from memory_condense.eval._identity import sha256_digest


COMPILED_RECEIPT_FIELDS = (
    "manifest_sha256",
    "cache_key",
    "sample_sha256",
    "database_sha256",
    "index_sha256",
    "embedding_execution_sha256",
    "implementation_sha256",
    "environment_lock_sha256",
    "turn_count",
    "chunk_count",
)
CAUSAL_RECEIPT_FIELDS = (
    "manifest_sha256",
    "cache_key",
    "sample_sha256",
    "compiled_cache_key",
    "database_sha256",
    "index_sha256",
    "build_protocol_sha256",
    "embedding_execution_sha256",
    "implementation_sha256",
    "environment_lock_sha256",
)
_RECEIPT_FIELDS = {
    "compiled": COMPILED_RECEIPT_FIELDS,
    "causal": CAUSAL_RECEIPT_FIELDS,
}
_INTEGER_FIELDS = {"turn_count", "chunk_count"}


def canonical_sha256(value: Any) -> str:
    """Hash one JSON-compatible value with the repository's stable encoding."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validated_cache_receipts(
    value: object,
    *,
    expected_sample_sha256: str | None = None,
    expected_implementation_sha256: str | None = None,
    expected_environment_lock_sha256: str | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Return an exact defensive copy of one compiled+causal receipt pair.

    The function intentionally rejects unknown fields.  A validation campaign
    therefore cannot silently accept a newer, weaker, or partially populated
    attestation schema under the old report contract.
    """

    if not isinstance(value, Mapping) or set(value) != set(_RECEIPT_FIELDS):
        raise ValueError(
            "cache receipts must contain exactly one compiled and one causal entry"
        )

    result: dict[str, list[dict[str, object]]] = {}
    for cache_kind, fields in _RECEIPT_FIELDS.items():
        entries = value.get(cache_kind)
        if not isinstance(entries, list) or len(entries) != 1:
            raise ValueError(
                "cache receipts must contain exactly one compiled and one causal entry"
            )
        source = entries[0]
        if not isinstance(source, Mapping) or set(source) != set(fields):
            raise ValueError(f"{cache_kind} cache receipt has an unexpected shape")
        receipt: dict[str, object] = {}
        for field in fields:
            raw = source[field]
            if field in _INTEGER_FIELDS:
                if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                    raise ValueError(
                        f"{cache_kind} cache receipt has invalid {field}"
                    )
                receipt[field] = raw
            else:
                receipt[field] = sha256_digest(
                    raw,
                    f"{cache_kind} cache receipt {field}",
                )
        result[cache_kind] = [receipt]

    compiled = result["compiled"][0]
    causal = result["causal"][0]
    if causal["compiled_cache_key"] != compiled["cache_key"]:
        raise ValueError("causal cache receipt does not bind the compiled cache key")
    for field in (
        "sample_sha256",
        "embedding_execution_sha256",
        "implementation_sha256",
        "environment_lock_sha256",
    ):
        if causal[field] != compiled[field]:
            raise ValueError(f"compiled and causal cache receipts disagree on {field}")

    expected = {
        "sample_sha256": expected_sample_sha256,
        "implementation_sha256": expected_implementation_sha256,
        "environment_lock_sha256": expected_environment_lock_sha256,
    }
    for field, raw_expected in expected.items():
        if raw_expected is None:
            continue
        expected_digest = sha256_digest(raw_expected, f"expected {field}")
        if compiled[field] != expected_digest:
            raise ValueError(f"cache receipts do not match expected {field}")
    return result


def cache_receipts_sha256(value: object) -> str:
    """Hash one already validated receipt pair."""

    return canonical_sha256(validated_cache_receipts(value))
