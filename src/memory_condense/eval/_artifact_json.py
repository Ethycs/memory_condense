"""Strict canonical JSON bytes shared by sealed evaluation artifacts."""

from __future__ import annotations

import json
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    """Encode one finite JSON value as sorted compact UTF-8 plus newline."""

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


__all__ = ["canonical_json_bytes"]
