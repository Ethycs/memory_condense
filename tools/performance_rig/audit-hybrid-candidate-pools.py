"""Measure gold/evidence ranks as the hybrid candidate boundary expands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory_condense.modeling.embedding import EmbeddingService
from memory_condense.eval.compiled_cache import compiled_store_ingest_fn
from memory_condense.eval.locked_split import load_split_manifest, select_locked_split
from memory_condense.eval.recall import contains_answer
from memory_condense.eval.schemas import EvalConfig, RetrievalConfig
from memory_condense.ingest.loader import load_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-file", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", default="development")
    parser.add_argument("--compiled-store-cache", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--pools", default="100,200,400,1000")
    parser.add_argument("--embedding-device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()

    pools = tuple(int(value) for value in args.pools.split(","))
    samples = select_locked_split(
        load_benchmark(args.benchmark_file),
        dataset_path=args.benchmark_file,
        manifest=load_split_manifest(args.split_manifest),
        split=args.split,
    )
    requested = set(args.sample_id)
    if requested:
        samples = [sample for sample in samples if sample.sample_id in requested]

    config = EvalConfig(
        retrieval=RetrievalConfig(mode="hybrid", k=10),
        embedding_device=args.embedding_device,
    )
    embedder = EmbeddingService(device=args.embedding_device)
    ingest = compiled_store_ingest_fn(
        args.compiled_store_cache,
        device=args.embedding_device,
        embedder=embedder,
    )
    for sample in samples:
        question = sample.questions[0]
        mc = ingest(sample, config, Path("unused"))
        try:
            query_embedding = embedder.embed_query(question.dated_question)
            baseline = mc.search_hybrid_from_embedding(
                question.dated_question,
                query_embedding,
                k=10,
                candidates=100,
            )
            baseline_sources = [
                result.turn.source_id or result.turn.turn_id
                for result in baseline
                if result.turn is not None
            ]
            measurements = []
            for pool in pools:
                results = mc.search_hybrid_from_embedding(
                    question.dated_question,
                    query_embedding,
                    k=pool,
                    candidates=pool,
                    ef_search=max(50, pool),
                )
                gold_ranks = [
                    rank
                    for rank, result in enumerate(results, start=1)
                    if contains_answer([result.chunk.text], question.answer)
                ]
                evidence_ranks = [
                    rank
                    for rank, result in enumerate(results, start=1)
                    if result.turn is not None
                    and (result.turn.source_id or result.turn.turn_id)
                    in question.evidence_sources
                ]
                evidence_source_ranks = {
                    source_id: min(
                        (
                            rank
                            for rank, result in enumerate(results, start=1)
                            if result.turn is not None
                            and (result.turn.source_id or result.turn.turn_id)
                            == source_id
                        ),
                        default=None,
                    )
                    for source_id in question.evidence_sources
                }
                measurements.append(
                    {
                        "pool": pool,
                        "returned": len(results),
                        "best_gold_rank": min(gold_ranks, default=None),
                        "best_evidence_rank": min(evidence_ranks, default=None),
                        "evidence_source_ranks": evidence_source_ranks,
                    }
                )
        finally:
            mc.close()
        print(
            json.dumps(
                {
                    "sample_id": sample.sample_id,
                    "answer": question.answer,
                    "evidence_sources": question.evidence_sources,
                    "baseline_sources": baseline_sources,
                    "measurements": measurements,
                }
            )
        )


if __name__ == "__main__":
    main()
