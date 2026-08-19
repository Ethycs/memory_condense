"""Build a verified development trace from reusable LongMemEval stores."""

from __future__ import annotations

import argparse
from pathlib import Path

from memory_condense.modeling.embedding import EmbeddingService
from memory_condense.eval.compiled_cache import compiled_store_ingest_fn
from memory_condense.eval.locked_split import (
    file_sha256,
    load_split_manifest,
    select_locked_split,
)
from memory_condense.eval.schemas import EvalConfig, RetrievalConfig
from memory_condense.eval.transition_trace import (
    build_transition_trace,
    save_transition_trace,
)
from memory_condense.ingest.loader import load_benchmark


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--benchmark-file", type=Path, required=True)
parser.add_argument("--split-manifest", type=Path, required=True)
parser.add_argument("--split", default="development")
parser.add_argument("--compiled-store-cache", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--alpha", type=float, default=0.65)
parser.add_argument("--candidates", type=int, default=100)
parser.add_argument("--max-radius", type=int, default=6)
parser.add_argument("--embedding-device", choices=("cpu", "cuda"), default=None)
parser.add_argument(
    "--max-samples",
    type=int,
    help="Deterministic prefix of the selected split for cold-build preflights",
)
args = parser.parse_args()

if args.max_samples is not None and args.max_samples < 1:
    parser.error("--max-samples must be positive")

samples = load_benchmark(args.benchmark_file)
manifest = load_split_manifest(args.split_manifest)
samples = select_locked_split(
    samples,
    dataset_path=args.benchmark_file,
    manifest=manifest,
    split=args.split,
)
if args.max_samples is not None:
    samples = samples[: args.max_samples]
config = EvalConfig(
    retrieval=RetrievalConfig(
        mode="hybrid_neighbor",
        k=args.k,
        alpha=args.alpha,
        candidates=args.candidates,
    ),
    embedding_device=args.embedding_device,
    max_prompt_tokens=8000,
)
embedder = EmbeddingService(device=args.embedding_device)
ingest = compiled_store_ingest_fn(
    args.compiled_store_cache,
    device=args.embedding_device,
    embedder=embedder,
)
pack = build_transition_trace(
    samples,
    config,
    ingest_fn=ingest,
    embedder=embedder,
    dataset_sha256=file_sha256(args.benchmark_file),
    split_manifest_sha256=file_sha256(args.split_manifest),
    split=args.split,
    max_radius=args.max_radius,
)
path = save_transition_trace(pack, args.output)
print(
    f"trace: {path}\n"
    f"questions: {len(pack.questions)}\n"
    f"sha256: {pack.trace_sha256}"
)
