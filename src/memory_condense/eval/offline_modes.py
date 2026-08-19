"""Offline evaluation workflows: recall, sufficiency, and comparison."""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
from pathlib import Path

from memory_condense.eval.schemas import UsageStats

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
    samples = runtime.load_benchmark(args.answer_recall, args.benchmark_format)
    if not samples:
        print("No samples parsed. Check --benchmark-format.")
        return

    samples = runtime._apply_sample_offset(args, runtime._apply_locked_split(args, samples))
    stress_tokens = getattr(args, "stress_context_tokens", None)
    stress_question_offset = 0
    stress_question_count = None
    if stress_tokens is not None:
        from memory_condense.eval.context_stress import (
            compose_context_stress_sample,
            transcript_tokens,
        )

        stress_question_offset = int(
            getattr(args, "stress_question_offset", 0)
        )
        stress_question_count = int(getattr(args, "stress_questions", 10))
        if stress_question_offset < 0:
            raise ValueError("--stress-question-offset must be non-negative")
        if stress_question_count < 1:
            raise ValueError("--stress-questions must be positive")
        # Keep the canonical ten-question stress sample as the causal-store
        # cache identity. Question sharding happens only after that immutable
        # store is opened; held-out questions do not change its learned graph.
        stress_question_pool = max(
            10,
            stress_question_offset + stress_question_count,
        )
        samples = [
            compose_context_stress_sample(
                samples,
                target_tokens=stress_tokens,
                max_questions=stress_question_pool,
            )
        ]
        actual_tokens = transcript_tokens(samples[0])
        print(
            f"Context stress memory: {actual_tokens:,} tokens, "
            f"{len(samples[0].turns):,} turns, "
            f"{len(samples[0].questions)} questions"
        )
    runtime._reserve_embedding_device_for_transient_models(args)
    config = runtime.config_from_args(args)
    print(
        f"{len(samples)} sample(s); measuring "
        f"{args.max_samples or len(samples)} in {config.retrieval.mode} mode. "
        "No API calls will be made."
    )
    reranker = runtime._load_candidate_reranker(args, config)
    selector = runtime._load_coverage_selector(args, config)
    try:
        report = runtime.run_recall(
            samples,
            config,
            benchmark=Path(args.answer_recall).stem,
            max_samples=1 if stress_tokens is not None else args.max_samples,
            ingest_fn=runtime._attach_runtime_controls(
                runtime._benchmark_ingest_fn(args, config),
                reranker=reranker,
                selector=selector,
            ),
            question_offset=stress_question_offset,
            max_questions=stress_question_count,
        )
    finally:
        if reranker is not None:
            reranker.close()
        if selector is not None:
            selector.close()
    runtime.print_recall_report(report)

    if args.csv:
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(
            [
                "question_id",
                "category",
                "in_haystack",
                "in_context",
                "best_f1",
                "in_header",
                "in_expansions",
                "context_tokens",
                "evidence_source_recall",
                "all_evidence_sources",
                "retrieved_source_ids",
                "raw_evidence_source_recall",
                "raw_all_evidence_sources",
                "raw_retrieved_source_ids",
                "answer_value_components_expected",
                "answer_value_components_found",
                "answer_value_component_recall",
                "all_answer_value_components",
                "answer_value_component_hit_mask",
                "answer_value_metric_kind",
                "source_companion_requested",
                "source_companion_hydrated",
                "source_companion_orphans",
                "source_companion_direct_date_retained",
                "source_companion_candidates_before",
                "source_companion_candidates_after",
                "selected_partitions",
                "partition_ranking",
                "direct_chunks",
                "consolidation_chunks",
                "causal_events",
                "causal_graph_edges",
                "causal_write_s",
                "qwen_rerank_passes",
                "qwen_candidate_inspections",
                "qwen_max_workspace_candidates",
                "qwen_max_workspace_tokens",
                "qwen_candidates_added",
                "qwen_feedback_rounds",
                "qwen_feedback_seed_sources",
                "qwen_feedback_candidates_added",
                "qwen_feedback_activation_candidates",
                "qwen_feedback_query_tokens",
                "coverage_selector_inspected",
                "coverage_selector_classified",
                "coverage_selector_clusters",
                "coverage_selector_null",
                "coverage_selector_uncertain",
                "coverage_selector_output",
                "coverage_selector_representatives",
                "coverage_selector_workspace_tokens",
                "coverage_selector_elapsed_s",
                "coverage_selector_operator",
                "coverage_selector_cardinality",
                "coverage_selector_quantifier",
                "coverage_selector_ordering",
                "coverage_selector_query_timestamp",
                "coverage_selector_temporal_window_days",
                "coverage_selector_posterior_kind",
                "coverage_selector_semantic_score_kind",
                "coverage_selector_answerability_score_kind",
                "coverage_selector_frontier_candidates",
                "coverage_selector_frontier_attempted",
                "coverage_selector_frontier_uninspected",
                "coverage_selector_frontier_exhaustive",
                "coverage_selector_frontier_batches",
                "coverage_selector_routed_frontier_exhaustive",
                "coverage_selector_active_partition_total",
                "coverage_selector_active_partition_inspected",
                "coverage_selector_active_partition_exhaustive",
                "coverage_selector_active_partition_sources_total",
                "coverage_selector_active_partition_structural_rows",
                "coverage_selector_active_partition_structural_hypotheses",
                "coverage_selector_active_partition_candidates_admitted",
                "coverage_selector_active_partition_candidates_already_present",
                "coverage_selector_active_partition_candidates_replaced",
                "coverage_selector_active_partition_candidates_truncated",
                "coverage_selector_active_partition_structural_overflow",
                "coverage_selector_active_partition_scan_contract",
                "coverage_selector_active_partition_semantically_complete",
                "coverage_selector_partition_scope_kind",
                "coverage_selector_partition_inventory_total",
                "coverage_selector_selected_partition_count",
                "coverage_selector_partition_scope_exhaustive",
                "coverage_selector_selected_scope_structurally_complete",
                "coverage_selector_global_semantic_complete",
                "coverage_selector_allow_selected_scope_fixed_k_closure",
                "closure_applied",
                "closure_scope",
                "closure_global_recall_guaranteed",
                "coverage_selector_cardinality_deficit",
                "coverage_selector_credible_clusters",
                "coverage_selector_reserved_representatives",
                "coverage_selector_structural_eligible_clusters",
                "coverage_selector_structural_reserved_representatives",
                "coverage_selector_score_provider_fallback",
                "coverage_selector_score_provider_model_id",
                "coverage_selector_score_provider_model_revision",
                "coverage_selector_score_provider_checkpoint_sha256",
                "coverage_selector_score_provider_device",
                "coverage_selector_score_provider_dtype",
                "coverage_selector_score_provider_forward_passes",
                "coverage_selector_score_provider_peak_workspace_tokens",
                "coverage_selector_score_provider_total_workspace_tokens",
                "coverage_selector_score_provider_elapsed_s",
                "coverage_selector_score_provider_retained_state_bytes",
                "coverage_selector_prefix_model_id",
                "coverage_selector_prefix_model_revision",
                "coverage_selector_prefix_checkpoint_sha256",
                "coverage_selector_prefix_device",
                "coverage_selector_prefix_dtype",
                "coverage_selector_prefix_layers",
                "coverage_selector_prefix_attention_layer",
                "coverage_selector_model_id",
                "coverage_selector_model_revision",
                "coverage_selector_checkpoint_sha256",
                "coverage_selector_semantic_inspected",
                "coverage_selector_semantic_workspace_tokens",
                "coverage_selector_semantic_elapsed_s",
                "coverage_selector_retained_state_bytes",
                "coverage_selector_status",
                "coverage_selector_bypass_reason",
                "coverage_selector_fallback_reason",
                "coverage_candidate_trace",
            ]
        )
        for question in report.questions:
            writer.writerow(
                [
                    question.question_id,
                    question.category,
                    int(question.in_haystack),
                    int(question.in_context),
                    f"{question.best_f1:.4f}",
                    int(question.in_memory_header),
                    int(question.in_expansions),
                    question.context_tokens,
                    "" if question.evidence_source_recall is None else (
                        f"{question.evidence_source_recall:.4f}"
                    ),
                    "" if question.all_evidence_sources is None else (
                        int(question.all_evidence_sources)
                    ),
                    "|".join(question.retrieved_source_ids),
                    "" if question.raw_evidence_source_recall is None else (
                        f"{question.raw_evidence_source_recall:.4f}"
                    ),
                    "" if question.raw_all_evidence_sources is None else (
                        int(question.raw_all_evidence_sources)
                    ),
                    "|".join(question.raw_retrieved_source_ids),
                    (
                        ""
                        if question.answer_value_components_expected is None
                        else question.answer_value_components_expected
                    ),
                    (
                        ""
                        if question.answer_value_components_found is None
                        else question.answer_value_components_found
                    ),
                    (
                        ""
                        if question.answer_value_component_recall is None
                        else f"{question.answer_value_component_recall:.4f}"
                    ),
                    (
                        ""
                        if question.all_answer_value_components is None
                        else int(question.all_answer_value_components)
                    ),
                    "|".join(
                        "1" if hit else "0"
                        for hit in question.answer_value_component_hit_mask
                    ),
                    question.answer_value_metric_kind,
                    "|".join(question.source_companion_requested),
                    "|".join(question.source_companion_hydrated),
                    "|".join(question.source_companion_orphans),
                    question.source_companion_direct_date_retained,
                    question.source_companion_candidates_before,
                    question.source_companion_candidates_after,
                    "|".join(question.selected_partitions),
                    json.dumps(question.partition_ranking, separators=(",", ":")),
                    question.direct_chunks,
                    question.consolidation_chunks,
                    question.causal_events,
                    question.causal_graph_edges,
                    f"{question.causal_write_s:.4f}",
                    question.qwen_rerank_passes,
                    question.qwen_candidate_inspections,
                    question.qwen_max_workspace_candidates,
                    question.qwen_max_workspace_tokens,
                    question.qwen_candidates_added,
                    question.qwen_feedback_rounds,
                    question.qwen_feedback_seed_sources,
                    question.qwen_feedback_candidates_added,
                    question.qwen_feedback_activation_candidates,
                    question.qwen_feedback_query_tokens,
                    question.coverage_selector_inspected,
                    question.coverage_selector_classified,
                    question.coverage_selector_clusters,
                    question.coverage_selector_null,
                    question.coverage_selector_uncertain,
                    question.coverage_selector_output,
                    question.coverage_selector_representatives,
                    question.coverage_selector_workspace_tokens,
                    f"{question.coverage_selector_elapsed_s:.4f}",
                    question.coverage_selector_operator,
                    (
                        ""
                        if question.coverage_selector_cardinality is None
                        else question.coverage_selector_cardinality
                    ),
                    question.coverage_selector_quantifier,
                    question.coverage_selector_ordering,
                    question.coverage_selector_query_timestamp or "",
                    (
                        ""
                        if question.coverage_selector_temporal_window_days is None
                        else question.coverage_selector_temporal_window_days
                    ),
                    question.coverage_selector_posterior_kind,
                    question.coverage_selector_semantic_score_kind,
                    question.coverage_selector_answerability_score_kind,
                    question.coverage_selector_frontier_candidates,
                    question.coverage_selector_frontier_attempted,
                    question.coverage_selector_frontier_uninspected,
                    int(question.coverage_selector_frontier_exhaustive),
                    question.coverage_selector_frontier_batches,
                    (
                        ""
                        if question.coverage_selector_routed_frontier_exhaustive
                        is None
                        else int(
                            question.coverage_selector_routed_frontier_exhaustive
                        )
                    ),
                    (
                        ""
                        if question.coverage_selector_active_partition_total is None
                        else question.coverage_selector_active_partition_total
                    ),
                    (
                        ""
                        if question.coverage_selector_active_partition_inspected
                        is None
                        else question.coverage_selector_active_partition_inspected
                    ),
                    (
                        ""
                        if question.coverage_selector_active_partition_exhaustive
                        is None
                        else int(
                            question.coverage_selector_active_partition_exhaustive
                        )
                    ),
                    (
                        ""
                        if question.coverage_selector_active_partition_sources_total
                        is None
                        else question.coverage_selector_active_partition_sources_total
                    ),
                    question.coverage_selector_active_partition_structural_rows,
                    question.coverage_selector_active_partition_structural_hypotheses,
                    question.coverage_selector_active_partition_candidates_admitted,
                    (
                        question.coverage_selector_active_partition_candidates_already_present
                    ),
                    question.coverage_selector_active_partition_candidates_replaced,
                    question.coverage_selector_active_partition_candidates_truncated,
                    question.coverage_selector_active_partition_structural_overflow,
                    question.coverage_selector_active_partition_scan_contract,
                    (
                        ""
                        if question.coverage_selector_active_partition_semantically_complete
                        is None
                        else int(
                            question.coverage_selector_active_partition_semantically_complete
                        )
                    ),
                    question.coverage_selector_partition_scope_kind,
                    (
                        ""
                        if question.coverage_selector_partition_inventory_total
                        is None
                        else question.coverage_selector_partition_inventory_total
                    ),
                    (
                        ""
                        if question.coverage_selector_selected_partition_count
                        is None
                        else question.coverage_selector_selected_partition_count
                    ),
                    (
                        ""
                        if question.coverage_selector_partition_scope_exhaustive
                        is None
                        else int(
                            question.coverage_selector_partition_scope_exhaustive
                        )
                    ),
                    (
                        ""
                        if question.coverage_selector_selected_scope_structurally_complete
                        is None
                        else int(
                            question.coverage_selector_selected_scope_structurally_complete
                        )
                    ),
                    (
                        ""
                        if question.coverage_selector_global_semantic_complete
                        is None
                        else int(
                            question.coverage_selector_global_semantic_complete
                        )
                    ),
                    int(
                        question.coverage_selector_allow_selected_scope_fixed_k_closure
                    ),
                    int(question.closure_applied),
                    question.closure_scope,
                    (
                        ""
                        if question.closure_global_recall_guaranteed is None
                        else int(question.closure_global_recall_guaranteed)
                    ),
                    question.coverage_selector_cardinality_deficit,
                    question.coverage_selector_credible_clusters,
                    question.coverage_selector_reserved_representatives,
                    question.coverage_selector_structural_eligible_clusters,
                    (
                        question.coverage_selector_structural_reserved_representatives
                    ),
                    question.coverage_selector_score_provider_fallback,
                    question.coverage_selector_score_provider_model_id,
                    question.coverage_selector_score_provider_model_revision,
                    question.coverage_selector_score_provider_checkpoint_sha256,
                    question.coverage_selector_score_provider_device,
                    question.coverage_selector_score_provider_dtype,
                    question.coverage_selector_score_provider_forward_passes,
                    (
                        question.coverage_selector_score_provider_peak_workspace_tokens
                    ),
                    (
                        question.coverage_selector_score_provider_total_workspace_tokens
                    ),
                    f"{question.coverage_selector_score_provider_elapsed_s:.4f}",
                    (
                        question.coverage_selector_score_provider_retained_state_bytes
                    ),
                    question.coverage_selector_prefix_model_id,
                    question.coverage_selector_prefix_model_revision,
                    question.coverage_selector_prefix_checkpoint_sha256,
                    question.coverage_selector_prefix_device,
                    question.coverage_selector_prefix_dtype,
                    question.coverage_selector_prefix_layers,
                    question.coverage_selector_prefix_attention_layer,
                    question.coverage_selector_model_id,
                    question.coverage_selector_model_revision,
                    question.coverage_selector_checkpoint_sha256,
                    question.coverage_selector_semantic_inspected,
                    question.coverage_selector_semantic_workspace_tokens,
                    f"{question.coverage_selector_semantic_elapsed_s:.4f}",
                    question.coverage_selector_retained_state_bytes,
                    question.coverage_selector_status,
                    question.coverage_selector_bypass_reason,
                    question.coverage_selector_fallback_reason,
                    json.dumps(
                        question.coverage_candidate_trace,
                        separators=(",", ":"),
                    ),
                ]
            )
        Path(args.csv).write_text(output.getvalue(), encoding="utf-8")
        print(f"Per-question CSV written to {args.csv}")


def run_sufficiency_mode(args: argparse.Namespace, *, runtime) -> None:
    """Audit retrieval separately from whether labelled evidence is answerable."""

    print(f"Loading benchmark from {args.sufficiency_audit}...")
    samples = runtime.load_benchmark(args.sufficiency_audit, args.benchmark_format)
    if not samples:
        print("No samples parsed. Check --benchmark-format.")
        return
    samples = runtime._apply_sample_offset(args, runtime._apply_locked_split(args, samples))
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

    reranker = None
    selector = None
    try:
        reranker = runtime._load_candidate_reranker(args, config)
        selector = runtime._load_coverage_selector(args, config)
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
        if reranker is not None:
            reranker.close()
        if selector is not None:
            selector.close()
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
