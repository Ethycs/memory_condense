"""Fetch the pinned public native M dataset; verify size and SHA before use."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import time
from urllib.request import urlopen

from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json

METADATA_SHA = "b3c62b94f93c7d397c2d0de3b909fe78d48812e9d5b67b54bc51b7f234d4f622"


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def download(metadata_path, cache_root, report_root):
    metadata = read_sealed_json(metadata_path)
    if metadata.sha256 != METADATA_SHA:
        raise ValueError("public file metadata changed")
    revision = metadata.payload["revision"]
    spec = next(f for f in metadata.payload["files"] if f["rfilename"] == "longmemeval_m_cleaned.json")
    directory = cache_root / revision
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / spec["rfilename"]
    partial = target.with_suffix(target.suffix + ".partial")
    url = f"https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/{revision}/{spec['rfilename']}"
    if not target.exists():
        # Preserve failures for explicit recovery. Opening exclusively prevents
        # a second invocation from duplicating an unfinished download.
        with partial.open("xb") as handle:
            received = 0
            last_print = time.monotonic()
            with urlopen(url, timeout=45) as response:
                if response.status != 200:
                    raise ValueError("expected complete public file response")
                declared = response.headers.get("Content-Length")
                if declared is not None and int(declared) != spec["size"]:
                    raise ValueError("public response size differs from pinned metadata")
                while chunk := response.read(8 * 1024 * 1024):
                    received += len(chunk)
                    if received > spec["size"]:
                        raise ValueError("download exceeds pinned size")
                    handle.write(chunk)
                    if time.monotonic() - last_print >= 15:
                        print({"received_bytes": received, "expected_bytes": spec["size"]}, flush=True)
                        last_print = time.monotonic()
            handle.flush()
            os.fsync(handle.fileno())
        if partial.stat().st_size != spec["size"] or digest(partial) != spec["lfs"]["sha256"]:
            raise ValueError("download size/hash verification failed; partial preserved")
        partial.rename(target)
    if target.stat().st_size != spec["size"] or digest(target) != spec["lfs"]["sha256"]:
        raise ValueError("native dataset no longer matches the pinned public file")
    artifact, _ = publish_sealed_json(report_root / "download.json", {
        "format": "memory-condense-native-longmemeval-download-v1", "metadata_sha256": metadata.sha256,
        "source_url": url, "dataset_path": str(target.resolve()), "dataset_sha256": spec["lfs"]["sha256"],
        "dataset_size_bytes": spec["size"], "question_records_inspected": False, "new_model_calls": 0,
        "implementation_sha256": digest(Path(__file__)), "benchmark_replacement_executed": False})
    print({"download_sha256": artifact.sha256, "path": str(target.resolve()), "verified": True}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    download(args.metadata, args.cache_root, args.report_root)
