"""Stable content identity for one parsed benchmark sample.

This module is deliberately dependency-light so both the benchmark reporter
and the compiled-store cache can bind to the exact same in-memory sample
without importing one another.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol


class ModelDumpLike(Protocol):
    """Structural Pydantic projection needed for stable content identity."""

    def model_dump(self, *, mode: str) -> Any: ...


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sample_sha256(sample: ModelDumpLike) -> str:
    """Hash all haystack, source, question, answer, and date fields."""

    return canonical_sha256(sample.model_dump(mode="json"))
