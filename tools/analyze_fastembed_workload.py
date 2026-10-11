"""Characterize production summary lengths and potential padding waste."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

from memory_condense.runtime.artifacts import save
from memory_condense.runtime.config import RuntimeAssets


def summaries(cache: Path, limit: int | None = None) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for folder in (cache / "atoms", cache / "merges"):
        for path in sorted(folder.glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            text = value.get("summary")
            if isinstance(text, str) and text.strip() and text not in seen:
                seen.add(text)
                values.append(text)
                if limit is not None and len(values) >= limit:
                    return values
    if not values:
        raise ValueError("No cached summaries found")
    return values


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def padding_cost(lengths: list[int], batch_size: int) -> dict[str, float | int]:
    linear_actual = sum(lengths)
    quadratic_actual = sum(value * value for value in lengths)
    linear_padded = 0
    quadratic_padded = 0
    for start in range(0, len(lengths), batch_size):
        batch = lengths[start : start + batch_size]
        width = max(batch)
        linear_padded += len(batch) * width
        quadratic_padded += len(batch) * width * width
    return {
        "batch_size": batch_size,
        "linear_padding_ratio": linear_padded / linear_actual,
        "quadratic_attention_padding_ratio": quadratic_padded / quadratic_actual,
        "linear_padded_tokens": linear_padded,
        "quadratic_padded_units": quadratic_padded,
    }


def main(cache: Path, output: Path, assets: Path | None, limit: int | None) -> None:
    runtime_assets = RuntimeAssets.resolve(assets)
    if runtime_assets.fastembed_package is not None:
        sys.path.insert(0, str(runtime_assets.fastembed_package))
    from fastembed.common.preprocessor_utils import load_tokenizer

    tokenizer, _ = load_tokenizer(model_dir=runtime_assets.bge)
    tokenizer.no_padding()
    texts = summaries(cache, limit)
    lengths: list[int] = []
    for start in range(0, len(texts), 512):
        lengths.extend(len(value.ids) for value in tokenizer.encode_batch(texts[start : start + 512]))
    ordered = sorted(lengths)
    batch_sizes = (2, 4, 8, 16)
    payload = {
        "format": "fastembed-production-workload-v1",
        "cache": str(cache),
        "text_count": len(texts),
        "tokens": {
            "minimum": min(lengths),
            "mean": statistics.fmean(lengths),
            "median": statistics.median(lengths),
            "p75": percentile(lengths, 0.75),
            "p90": percentile(lengths, 0.90),
            "p95": percentile(lengths, 0.95),
            "p99": percentile(lengths, 0.99),
            "maximum": max(lengths),
        },
        "arrival_order": [padding_cost(lengths, size) for size in batch_sizes],
        "length_sorted": [padding_cost(ordered, size) for size in batch_sizes],
    }
    receipt = save(output, payload)
    print(json.dumps({"output": str(receipt.path), "sha256": receipt.sha256, **payload}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    main(args.cache, args.output, args.assets, args.limit)
