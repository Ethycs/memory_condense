"""Measure batch-size speed and retrieval fidelity on production summary text."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from memory_condense.runtime.artifacts import save
from memory_condense.runtime.embedding import FastEmbedBGE


def load_texts(cache: Path, limit: int) -> list[str]:
    texts: list[str] = []
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
                texts.append(text)
                if len(texts) == limit:
                    return texts
    raise ValueError(f"Only found {len(texts)} unique cached summaries; expected {limit}")


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(norms <= 0):
        raise ValueError("Embedding assay received invalid vectors")
    return vectors / norms


def topk(matrix: np.ndarray, k: int) -> list[set[int]]:
    similarities = matrix @ matrix.T
    np.fill_diagonal(similarities, -np.inf)
    return [set(np.argpartition(row, -k)[-k:].tolist()) for row in similarities]


def main(cache: Path, output: Path, limit: int, batch_sizes: tuple[int, ...], order: str) -> None:
    texts = load_texts(cache, limit)
    encoder = FastEmbedBGE()
    model = encoder.model
    results: dict[int | str, tuple[np.ndarray, float]] = {}
    try:
        # Warm native code and allocator state outside the measurements.
        list(model.embed(texts[:2], batch_size=1))
        if order == "length":
            tokenizer = model.model.tokenizer
            texts.sort(key=lambda value: len(tokenizer.encode(value).ids))
        started = time.perf_counter()
        separate = [next(iter(model.embed([value], batch_size=1))) for value in texts]
        results["separate_calls"] = (np.asarray(separate, dtype=np.float32), time.perf_counter() - started)
        for batch_size in batch_sizes:
            started = time.perf_counter()
            vectors = np.asarray(list(model.embed(texts, batch_size=batch_size)), dtype=np.float32)
            results[batch_size] = (vectors, time.perf_counter() - started)
    finally:
        encoder.close()

    baseline = normalize(results["separate_calls"][0])
    baseline_top = topk(baseline, 10)
    rows = []
    baseline_elapsed = results["separate_calls"][1]
    for batch_size in ("separate_calls", *batch_sizes):
        vectors, elapsed = results[batch_size]
        normalized = normalize(vectors)
        cosine = np.sum(baseline * normalized, axis=1)
        candidate_top = topk(normalized, 10)
        overlaps = [len(left & right) / 10 for left, right in zip(baseline_top, candidate_top, strict=True)]
        rows.append({
            "batch_size": batch_size,
            "elapsed_s": elapsed,
            "texts_per_s": limit / elapsed,
            "speedup_vs_separate_calls": baseline_elapsed / elapsed,
            "max_abs_vector_delta": float(np.max(np.abs(results["separate_calls"][0] - vectors))),
            "minimum_same_text_cosine": float(np.min(cosine)),
            "mean_same_text_cosine": float(np.mean(cosine)),
            "top10_exact_query_fraction": sum(left == right for left, right in zip(baseline_top, candidate_top, strict=True)) / limit,
            "mean_top10_overlap": statistics.fmean(overlaps),
            "minimum_top10_overlap": min(overlaps),
        })
    receipt = save(output, {
        "format": "fastembed-production-batching-assay-v2",
        "cache": str(cache),
        "text_count": limit,
        "order": order,
        "embedding_identity": encoder.execution_identity,
        "rows": rows,
    })
    print({"output": str(receipt.path), "sha256": receipt.sha256, "rows": rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--batch-sizes", default="1,8,16,32")
    parser.add_argument("--order", choices=("arrival", "length"), default="arrival")
    args = parser.parse_args()
    batch_sizes = tuple(int(value) for value in args.batch_sizes.split(","))
    if not batch_sizes or any(value < 1 for value in batch_sizes) or len(set(batch_sizes)) != len(batch_sizes):
        parser.error("--batch-sizes must contain unique positive integers")
    main(args.cache, args.output, args.limit, batch_sizes, args.order)
