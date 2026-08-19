"""Stateful workflow for paid or local benchmark execution."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

def run_benchmark_mode(args: argparse.Namespace, *, runtime) -> None:
    print(f"Loading benchmark from {args.benchmark_file}...")
    samples = runtime.load_benchmark(args.benchmark_file, args.benchmark_format)
    if not samples:
        print("No benchmark samples found.")
        sys.exit(1)

    samples = runtime._apply_sample_offset(args, runtime._apply_locked_split(args, samples))
    stress_tokens = getattr(args, "stress_context_tokens", None)
    if stress_tokens is not None:
        from memory_condense.eval.context_stress import (
            compose_context_stress_sample,
            transcript_tokens,
        )

        samples = [
            compose_context_stress_sample(
                samples,
                target_tokens=stress_tokens,
                max_questions=getattr(args, "stress_questions", 10),
                question_offset=getattr(args, "stress_question_offset", 0),
            )
        ]
        actual_tokens = transcript_tokens(samples[0])
        print(
            f"Context stress memory: {actual_tokens:,} tokens, "
            f"{len(samples[0].turns):,} turns, "
            f"{len(samples[0].questions)} questions"
        )
    questions = sum(len(s.questions) for s in samples)
    print(f"Loaded {len(samples)} samples / {questions} questions")

    planned_provider_calls = runtime._planned_provider_calls(
        samples,
        max_samples=args.max_samples,
        local_answerer=bool(args.local_qwen_model_dir),
        use_judge=args.use_judge,
        provider_retries=args.provider_retries,
    )
    if planned_provider_calls > args.max_provider_calls:
        raise ValueError(
            f"planned remote provider calls ({planned_provider_calls}) exceed "
            f"--max-provider-calls ({args.max_provider_calls}); explicit "
            "authorization is required"
        )

    if args.max_samples is None:
        print(
            "\nWARNING: no --max-samples set. A full run ingests every haystack "
            "through bge-m3 and makes one LLM call per question"
            f"{' (doubled by --use-judge)' if args.use_judge else ''}. "
            "Start with --max-samples 10.\n"
        )

    local_answerer = None
    if (
        args.coverage_selector_local_model_dir
        or args.coverage_selector_qwen_prefix_model_dir
        or args.coverage_selector_cross_encoder_model_dir
    ) and args.local_qwen_model_dir:
        raise ValueError(
            "the transient coverage selector and local responder cannot share "
            "this process"
        )
    runtime._reserve_embedding_device_for_transient_models(args)
    if args.local_qwen_model_dir:
        from memory_condense.eval.local_qwen import LocalQwenAnswerer

        # Keep the 8 GiB GPU available for the offloaded full responder. BGE
        # remains functional on CPU and is unloaded with the benchmark run.
        if args.embedding_device is None:
            args.embedding_device = "cpu"
        print(f"Loading local responder from {args.local_qwen_model_dir}...")
        local_answerer = LocalQwenAnswerer(
            args.local_qwen_model_dir,
            max_new_tokens=args.local_qwen_max_new_tokens,
            gpu_memory=args.local_qwen_gpu_memory,
            cpu_memory=args.local_qwen_cpu_memory,
            dtype=args.local_qwen_dtype,
        )
        args.responder_model = (
            f"local/{args.local_qwen_model_dir.name}:{local_answerer.dtype_name}"
        )

    config = runtime.config_from_args(args)
    dataset_hash = runtime.file_sha256(args.benchmark_file)
    split_manifest_hash = (
        runtime.file_sha256(args.benchmark_split_manifest)
        if args.benchmark_split_manifest
        else ""
    )
    implementation_hash = runtime.implementation_sha256()
    environment_lock_hash = runtime.environment_lock_sha256()
    evaluation_identity = runtime._benchmark_evaluation_identity(args, config)
    policy_hash = runtime._verified_policy_sha256(
        args.policy_manifest,
        config=config,
        dataset_sha256=dataset_hash,
        split_manifest=args.benchmark_split_manifest,
        active_split=args.benchmark_split,
        active_implementation_sha256=implementation_hash,
        active_environment_lock_sha256=environment_lock_hash,
        evaluation_identity=evaluation_identity,
    )
    reranker = None
    selector = None
    try:
        reranker = runtime._load_candidate_reranker(args, config)
        selector = runtime._load_coverage_selector(args, config)
        result = runtime.run_benchmark(
            samples,
            config,
            answer_fn=(
                local_answerer
                or runtime._make_answer_fn(
                    args.responder_model,
                    retries=args.provider_retries,
                )
            ),
            judge_fn=(
                runtime._make_judge_fn(
                    args.judge_model,
                    retries=args.provider_retries,
                )
                if args.use_judge
                else None
            ),
            max_samples=args.max_samples,
            # Label the run with the dataset, not the --benchmark-format flag, which
            # defaults to "auto" and would name every report benchmark_auto_*.json.
            benchmark=Path(args.benchmark_file).stem,
            ingest_fn=runtime._attach_runtime_controls(
                runtime._benchmark_ingest_fn(args, config),
                reranker=reranker,
                selector=selector,
            ),
            verbose=True,
            dataset_sha256=dataset_hash,
            split_manifest_sha256=split_manifest_hash,
            benchmark_split=args.benchmark_split or "",
            implementation_sha256=implementation_hash,
            environment_lock_sha256=environment_lock_hash,
            policy_manifest_sha256=policy_hash,
            evaluation_protocol=evaluation_identity,
        )
    finally:
        if reranker is not None:
            reranker.close()
        if selector is not None:
            selector.close()
        if local_answerer is not None:
            print(
                f"Local responder: {local_answerer.calls} calls in "
                f"{local_answerer.elapsed_s:.1f}s"
            )
            local_answerer.close()
    runtime._assert_implementation_unchanged(implementation_hash)
    runtime.print_benchmark_summary(result)
    path = runtime.save_benchmark_report(result, args.results_dir)
    print(f"\nBenchmark report saved to {path}")
