"""Content-identity primitives shared by provenance and verification code."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    """SHA-256 hexdigest of a file's exact bytes, streamed in 8 MiB blocks.

    The chunked read never changes the digest; it only bounds memory while
    hashing multi-gigabyte checkpoint shards and databases.
    """

    digest = hashlib.sha256()
    with Path(path).open("rb", buffering=0) as source:
        buffer = bytearray(8 * 1024 * 1024)
        view = memoryview(buffer)
        while size := source.readinto(buffer):
            digest.update(view[:size])
    return digest.hexdigest()
