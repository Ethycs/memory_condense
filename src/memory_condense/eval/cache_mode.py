"""Stateful workflow for blind compiled/causal cache preparation."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from memory_condense.eval.cache_receipts import validated_cache_receipts
from memory_condense.eval.runtime import prepare_samples, run_provenance
from memory_condense.eval.schemas import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_RESPONDER_MODEL,
    EvalConfig,
)

def _validate_prepare_cache_args(
    args: argparse.Namespace,
    config: EvalConfig,
) -> None:
    """Reject options that could turn blind cache preparation into an eval."""

    if args.policy_manifest is None:
        raise ValueError("--prepare-cache-only requires --policy-manifest")
    if args.compiled_store_cache is None or args.causal_store_cache is None:
        raise ValueError(
            "--prepare-cache-only requires --compiled-store-cache and "
            "--causal-store-cache"
        )
    if config.retrieval.mode not in {"causal_consolidation", "causal_graph"}:
        raise ValueError(
            "--prepare-cache-only requires --mode causal_consolidation or "
            "causal_graph"
        )

    responder_requested = bool(
        getattr(args, "_responder_model_explicit", False)
        or args.responder_model != DEFAULT_RESPONDER_MODEL
        or args.local_qwen_model_dir
    )
    judge_requested = bool(
        getattr(args, "_judge_model_explicit", False)
        or args.judge_model != DEFAULT_JUDGE_MODEL
        or args.use_judge
    )
    provider_requested = bool(args.max_provider_calls or args.provider_retries)
    if responder_requested or judge_requested or provider_requested:
        raise ValueError(
            "--prepare-cache-only rejects responder, judge, --use-judge, "
            "and remote-provider options"
        )


def _validated_blind_cache_receipts(store) -> dict[str, list[dict[str, object]]]:
    """Copy the two exact, text-free receipts attached by the cache builder."""

    try:
        return validated_cache_receipts(
            getattr(store, "blind_cache_receipts", None)
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def _causal_count_timing_metadata(store) -> dict[str, int | float]:
    """Whitelisted scalar causal-build diagnostics, with no event text/IDs."""

    stats = getattr(store, "causal_consolidation_stats", {})
    staging = stats.get("staging", {}) if isinstance(stats, dict) else {}
    learning = stats.get("learning", {}) if isinstance(stats, dict) else {}
    result: dict[str, int | float] = {}
    for source, fields, prefix in (
        (
            staging,
            (
                "source_turns",
                "events",
                "completed_episodes",
                "outcome_chunks_bound",
                "skipped_large_prompt",
                "skipped_insufficient_candidates",
                "elapsed_s",
            ),
            "staging_",
        ),
        (
            learning,
            ("events_offered", "events_applied", "elapsed_s"),
            "learning_",
        ),
    ):
        if not isinstance(source, dict):
            continue
        for field in fields:
            value = source.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[f"{prefix}{field}"] = value
    return result


def run_prepare_cache_only(
    args: argparse.Namespace, *, runtime
) -> dict[str, object]:
    """Build locked compiled+causal caches without observing QA probes.

    Provenance is checked before parsing or ingesting the dataset. The only
    emitted record contains one-way hashes, aggregate counts, and timings; it
    deliberately has no sample IDs, paths, questions, answers, evidence, or
    retrieved content.
    """

    runtime._reserve_embedding_device_for_transient_models(args)
    config = runtime.config_from_args(args)
    runtime._validate_prepare_cache_args(args, config)

    provenance = run_provenance(
        args,
        args.benchmark_file,
        config,
        runtime=runtime,
        prepare_only=True,
    )

    prepared = prepare_samples(
        args,
        args.benchmark_file,
        runtime=runtime,
        verbose=False,
        stress=True,
        shard_stress_questions=True,
    )
    if prepared is None:
        raise ValueError("no benchmark samples found")
    samples = prepared.samples

    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError("cache-preparation shard contains no samples")

    ingest_fn = runtime._benchmark_ingest_fn(args, config, prepare_only=True)
    started = time.perf_counter()
    sample_rows: list[dict[str, object]] = []
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            for index, sample in enumerate(samples):
                sample_started = time.perf_counter()
                store = ingest_fn(
                    sample,
                    config,
                    Path(tmpdir) / f"sample_{index}",
                )
                try:
                    digest = runtime.sample_sha256(sample)
                    receipts = runtime._validated_blind_cache_receipts(store)
                    database_path = store.database_path
                    index_path = database_path.with_name("hnsw_index.bin")
                    database_digest = runtime.file_sha256(database_path)
                    index_digest = runtime.file_sha256(index_path)
                    causal_receipt = receipts["causal"][0]
                    if (
                        causal_receipt["database_sha256"] != database_digest
                        or causal_receipt["index_sha256"] != index_digest
                    ):
                        raise RuntimeError(
                            "active store hashes do not match its causal cache receipt"
                        )
                    row: dict[str, object] = {
                        "sample_sha256": digest,
                        "turn_count": len(sample.turns),
                        "source_count": len(
                            {
                                source_id
                                for source_id in sample.turn_source_ids
                                if source_id is not None
                            }
                        ),
                        "database_sha256": database_digest,
                        "index_sha256": index_digest,
                    }
                    row.update(runtime._causal_count_timing_metadata(store))
                finally:
                    store.close()
                row["elapsed_s"] = time.perf_counter() - sample_started
                row["compiled_cache_entries"] = receipts["compiled"]
                row["causal_cache_entries"] = receipts["causal"]
                sample_rows.append(row)
    finally:
        release_embedder = getattr(ingest_fn, "release_embedder", None)
        if callable(release_embedder):
            release_embedder()

    runtime._assert_implementation_unchanged(provenance.implementation_sha256)
    report: dict[str, object] = {
        "dataset_sha256": provenance.dataset_sha256,
        "split_manifest_sha256": provenance.split_manifest_sha256,
        "policy_manifest_sha256": provenance.policy_manifest_sha256,
        "implementation_sha256": provenance.implementation_sha256,
        "environment_lock_sha256": provenance.environment_lock_sha256,
        "sample_count": len(sample_rows),
        "turn_count": sum(int(row["turn_count"]) for row in sample_rows),
        "source_count": sum(int(row["source_count"]) for row in sample_rows),
        "stress_context_tokens": prepared.stress_tokens,
        "samples": sample_rows,
        "elapsed_s": time.perf_counter() - started,
    }
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return report
