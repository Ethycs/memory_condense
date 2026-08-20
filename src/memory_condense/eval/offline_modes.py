"""Offline evaluation workflows: recall, sufficiency, and comparison."""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
from pathlib import Path

from memory_condense.eval.recall_models import QuestionRecall
from memory_condense.eval.runtime import (
    prepare_samples,
    transient_runtime_controls,
)
from memory_condense.eval.schemas import UsageStats

#: The frozen locked-v3 recall campaign predates the model-driven writer;
#: readers of archived shards key on this historical column name.
_LEGACY_RECALL_HEADERS = {"in_memory_header": "in_header"}


def _compact_json(value) -> str:
    return json.dumps(value, separators=(",", ":"))


#: Fields whose serialization is not derivable from their runtime type.
_RECALL_FIELD_FORMATTERS = {
    "answer_value_component_hit_mask": lambda mask: "|".join(
        "1" if hit else "0" for hit in mask
    ),
    "partition_ranking": _compact_json,
    "coverage_candidate_trace": _compact_json,
    "survives_horizon": lambda value: json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ),
}


def _recall_cell(value):
    """Match the historical hand-written recall column formatting exactly."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return "|".join(value)
    return value


def _recall_questions_csv(report) -> str:
    """Render every ``QuestionRecall`` field as one CSV column.

    Driving the columns from the model eliminates the transcription drift the
    old hand-written table accumulated (``evidence_source_hit`` and
    ``survives_horizon`` were silently absent).
    """

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            _LEGACY_RECALL_HEADERS.get(name, name)
            for name in QuestionRecall.model_fields
        ],
        lineterminator="\n",
    )
    writer.writeheader()
    for question in report.questions:
        writer.writerow(
            {
                _LEGACY_RECALL_HEADERS.get(name, name): (
                    _RECALL_FIELD_FORMATTERS.get(name, _recall_cell)(value)
                )
                for name, value in question.model_dump().items()
            }
        )
    return output.getvalue()


def run_compare(args: argparse.Namespace, *, runtime) -> None:
    baseline_path, treatment_path = args.compare
    baseline = runtime.load_run(baseline_path)
    treatment = runtime.load_run(treatment_path)
    report = runtime.compare_runs(baseline, treatment)
    runtime.print_comparison(report)

    if args.csv:
        Path(args.csv).write_text(runtime.to_csv(treatment), encoding="utf-8")
        print(f"\nPer-turn CSV written to {args.csv}")


def run_answer_recall(args: argparse.Namespace, *, runtime) -> None:
    """Offline: is the gold answer even reachable from the assembled context?

    Free, keyless, and the cheap predictor of the paid comparison — if the
    memory arm's context holds the answer less often than the dense arm's, no
    responder can recover the difference.
    """
    print(f"Loading benchmark from {args.answer_recall}...")
    prepared = prepare_samples(
        args,
        args.answer_recall,
        runtime=runtime,
        stress=True,
    )
    if prepared is None:
        print("No samples parsed. Check --benchmark-format.")
        return
    samples = prepared.samples
    runtime._reserve_embedding_device_for_transient_models(args)
    config = runtime.config_from_args(args)
    print(
        f"{len(samples)} sample(s); measuring "
        f"{args.max_samples or len(samples)} in {config.retrieval.mode} mode. "
        "No API calls will be made."
    )
    with transient_runtime_controls(args, config, runtime=runtime) as (
        reranker,
        selector,
    ):
        report = runtime.run_recall(
            samples,
            config,
            benchmark=Path(args.answer_recall).stem,
            max_samples=1 if prepared.stress_composed else args.max_samples,
            ingest_fn=runtime._attach_runtime_controls(
                runtime._benchmark_ingest_fn(args, config),
                reranker=reranker,
                selector=selector,
            ),
            question_offset=prepared.stress_question_offset,
            max_questions=prepared.stress_question_count,
        )
    runtime.print_recall_report(report)

    if args.csv:
        Path(args.csv).write_text(
            _recall_questions_csv(report), encoding="utf-8"
        )
        print(f"Per-question CSV written to {args.csv}")


def run_sufficiency_mode(args: argparse.Namespace, *, runtime) -> None:
    """Audit retrieval separately from whether labelled evidence is answerable."""

    print(f"Loading benchmark from {args.sufficiency_audit}...")
    prepared = prepare_samples(args, args.sufficiency_audit, runtime=runtime)
    if prepared is None:
        print("No samples parsed. Check --benchmark-format.")
        return
    samples = prepared.samples
    selected = samples[: args.max_samples] if args.max_samples is not None else samples
    labeled_questions = sum(
        bool(question.evidence_sources)
        for sample in selected
        for question in sample.questions
    )
    if args.provider_retries < 0:
        raise ValueError("--provider-retries must be non-negative")
    planned_calls = (
        2 * labeled_questions * (args.provider_retries + 1)
        if args.use_judge
        else 0
    )
    remote_calls = 0 if args.local_qwen_model_dir else planned_calls
    if remote_calls > args.max_provider_calls:
        raise ValueError(
            f"planned remote provider calls ({remote_calls}) exceed "
            f"--max-provider-calls ({args.max_provider_calls}); explicit "
            "authorization is required"
        )
    if args.qwen_rerank_model_dir and args.local_qwen_model_dir and args.use_judge:
        raise ValueError(
            "the full local judge and prefix reranker cannot share this GPU in "
            "one process; run the deterministic retrieval audit first"
        )
    if (
        (
            args.coverage_selector_local_model_dir
            or args.coverage_selector_qwen_prefix_model_dir
            or args.coverage_selector_cross_encoder_model_dir
        )
        and args.local_qwen_model_dir
        and args.use_judge
    ):
        raise ValueError(
            "the transient coverage selector and local sufficiency judge "
            "cannot share this process"
        )
    runtime._reserve_embedding_device_for_transient_models(args)
    config = runtime.config_from_args(args)
    policy_hash = runtime._verified_policy_sha256(
        args.policy_manifest,
        config=config,
        dataset_sha256=runtime.file_sha256(args.sufficiency_audit),
        split_manifest=args.benchmark_split_manifest,
        active_split=args.benchmark_split,
    )
    if policy_hash:
        print(f"Verified retrieval policy sha256 {policy_hash[:12]}...")

    local_judge = None
    sufficiency_fn = None
    if args.use_judge and args.local_qwen_model_dir:
        from memory_condense.eval.local_qwen import LocalQwenAnswerer

        local_judge = LocalQwenAnswerer(
            args.local_qwen_model_dir,
            max_new_tokens=args.local_qwen_max_new_tokens,
            gpu_memory=args.local_qwen_gpu_memory,
            cpu_memory=args.local_qwen_cpu_memory,
            dtype=args.local_qwen_dtype,
        )

        from memory_condense.eval.sufficiency import build_sufficiency_prompt

        def local_sufficiency(question, gold, context):
            started = time.perf_counter()
            verdict = local_judge(
                build_sufficiency_prompt(question, gold, context)
            )
            return (
                verdict.upper().startswith("SUFFICIENT"),
                verdict,
                UsageStats(calls=1, elapsed_s=time.perf_counter() - started),
            )

        sufficiency_fn = local_sufficiency
    elif args.use_judge:
        sufficiency_fn = runtime._make_sufficiency_fn(
            args.judge_model,
            retries=args.provider_retries,
        )

    try:
        with transient_runtime_controls(args, config, runtime=runtime) as (
            reranker,
            selector,
        ):
            report = runtime.run_sufficiency_audit(
                samples,
                config,
                benchmark=Path(args.sufficiency_audit).stem,
                max_samples=args.max_samples,
                ingest_fn=runtime._attach_runtime_controls(
                    runtime._benchmark_ingest_fn(args, config),
                    reranker=reranker,
                    selector=selector,
                ),
                sufficiency_fn=sufficiency_fn,
            )
    finally:
        if local_judge is not None:
            local_judge.close()

    runtime.print_sufficiency_report(report)
    if args.csv:
        rows = [row.model_dump(mode="json") for row in report.questions]
        output = io.StringIO()
        if rows:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                row["expected_source_ids"] = "|".join(row["expected_source_ids"])
                row["retrieved_source_ids"] = "|".join(row["retrieved_source_ids"])
                row["judge_usage"] = json.dumps(row["judge_usage"], sort_keys=True)
                writer.writerow(row)
        Path(args.csv).write_text(output.getvalue(), encoding="utf-8")
        print(f"Per-question sufficiency CSV written to {args.csv}")
