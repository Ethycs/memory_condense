"""Locate gold spans relative to hybrid anchors inside activated sources."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.modeling.embedding import EmbeddingService
from memory_condense.eval.benchmark import ingest_sample
from memory_condense.eval.locked_split import load_split_manifest, select_locked_split
from memory_condense.eval.recall import contains_answer
from memory_condense.eval.schemas import EvalConfig, RetrievalConfig
from memory_condense.ingest.loader import load_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-file", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", default="development")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-radius", type=int, default=6)
    parser.add_argument("--embedding-device", choices=("cpu", "cuda"), default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    samples = select_locked_split(
        load_benchmark(args.benchmark_file),
        dataset_path=args.benchmark_file,
        manifest=load_split_manifest(args.split_manifest),
        split=args.split,
    )
    requested = set(args.sample_id)
    if requested:
        samples = [sample for sample in samples if sample.sample_id in requested]
        missing = requested - {sample.sample_id for sample in samples}
        if missing:
            raise ValueError(f"sample IDs are not in split {args.split!r}: {sorted(missing)}")
    config = EvalConfig(
        retrieval=RetrievalConfig(mode="hybrid_neighbor", k=args.k)
    )
    embedder = EmbeddingService(device=args.embedding_device)

    for sample in samples:
        question = sample.questions[0]
        with tempfile.TemporaryDirectory(prefix="mc-neighbor-diagnostic-") as data_dir:
            mc = ingest_sample(sample, config, data_dir, embedder=embedder)
            try:
                anchors = mc.search_hybrid(question.dated_question, k=args.k)
                expanded = mc.expand_source_neighbors(
                    anchors, radius=args.max_radius
                )
                minimum_radius = None
                for radius in range(args.max_radius + 1):
                    candidates = mc.expand_source_neighbors(anchors, radius=radius)
                    if contains_answer(
                        [result.chunk.text for result in candidates], question.answer
                    ):
                        minimum_radius = radius
                        break
            finally:
                mc.close()

        answer_results = [
            (index, result)
            for index, result in enumerate(expanded)
            if contains_answer([result.chunk.text], question.answer)
        ]
        first_rank = answer_results[0][0] + 1 if answer_results else None
        first_neighbor_slot = (
            answer_results[0][0] - len(anchors) + 1
            if answer_results and answer_results[0][0] >= len(anchors)
            else None
        )
        first_route = answer_results[0][1].route if answer_results else None
        first_anchor = (
            answer_results[0][1].anchor_chunk_id if answer_results else None
        )
        print(
            {
                "sample_id": sample.sample_id,
                "category": question.category,
                "minimum_radius": minimum_radius,
                "first_result_rank": first_rank,
                "first_neighbor_slot": first_neighbor_slot,
                "route": first_route,
                "anchor_chunk_id": first_anchor,
                "anchor_tokens": sum(count_tokens(r.chunk.text) for r in anchors),
                "expanded_tokens": sum(count_tokens(r.chunk.text) for r in expanded),
            }
        )


if __name__ == "__main__":
    main()
