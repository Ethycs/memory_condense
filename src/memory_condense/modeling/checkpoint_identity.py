"""Shared checkpoint-verification protocol for pinned model identities.

The local checkpoint verifiers (the Qwen prefix loader and the BGE-M3
embedding service) bind a model to content the same way: every consumed file
is streamed through SHA-256, compared in constant time against a pinned
digest, and folded into one canonical manifest digest over the verified
per-file hashes plus the model id and revision.  The manifest ``format``
string is the only caller-specific input, so two different checkpoint kinds
can never collide on the same manifest digest.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Mapping

from memory_condense.domain.integrity import file_sha256

__all__ = ["checkpoint_manifest_sha256", "file_sha256", "verify_file_sha256"]


def checkpoint_manifest_sha256(
    file_hashes: Mapping[str, str],
    *,
    manifest_format: str,
    model_id: str,
    model_revision: str,
) -> str:
    """Canonical digest binding verified file hashes to one model identity."""

    payload = {
        "format": str(manifest_format),
        "model_id": str(model_id),
        "model_revision": str(model_revision),
        "files": {
            str(name): str(digest).casefold()
            for name, digest in sorted(file_hashes.items())
        },
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_file_sha256(
    path: Path,
    expected: str,
    *,
    name: str,
    context: str,
) -> str:
    """Hash one file and fail closed unless it matches its pinned digest."""

    actual = file_sha256(path)
    if not hmac.compare_digest(actual, str(expected).casefold()):
        raise ValueError(
            f"{context} SHA-256 mismatch for {name}: "
            f"expected {expected}, got {actual}"
        )
    return actual
