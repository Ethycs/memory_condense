"""Stateful per-sample measurement and benchmark recall orchestration."""

from __future__ import annotations

import gc
import tempfile
from pathlib import Path

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.eval.answer_value_coverage import (
    answer_value_component_coverage,
    best_f1,
    contains_answer,
)
from memory_condense.eval.benchmark import (
    IngestFn,
    cap_context_to_prompt_budget,
    ingest_sample,
    shared_embedding_ingest_fn,
)
from memory_condense.eval.recall_assembly import _assemble, _survival
from memory_condense.eval.recall_models import (
    DEFAULT_HORIZONS_TURNS,
    QuestionRecall,
    RecallReport,
)
from memory_condense.eval.schemas import EvalConfig
from memory_condense.ingest.loader import BenchmarkSample
from memory_condense.search.packing.context_packer import is_source_metadata_text

def measure_sample(
    sample: BenchmarkSample,
    config: EvalConfig,
    data_dir: Path,
    ingest_fn: IngestFn = ingest_sample,
    horizons_turns=DEFAULT_HORIZONS_TURNS,
    question_offset: int = 0,
    max_questions: int | None = None,
) -> list[QuestionRecall]:
    """Ingest one sample and measure answer reachability for its questions."""
    mc = ingest_fn(sample, config, data_dir)
    try:
        out: list[QuestionRecall] = []
        haystack_texts = [text for _role, text in sample.turns]
        stop = (
            None
            if max_questions is None
            else question_offset + max_questions
        )
        for question in sample.questions[question_offset:stop]:
            query_text = question.dated_question
            header, body, body_sources, body_is_consolidation = _assemble(
                mc, query_text, config
            )
            header_count = len(header)
            capped = cap_context_to_prompt_budget(
                query_text,
                header + body,
                config.max_prompt_tokens,
            )
            header = capped[:header_count]
            body = capped[header_count:]
            body_sources = body_sources[: len(body)]
            body_is_consolidation = body_is_consolidation[: len(body)]
            everything = header + body
            # Gold answer decomposition begins only here: retrieval, selector
            # execution, context packing, and the hard prompt cap have already
            # completed. Score the exact raw body the responder receives,
            # never memory summaries or provenance-only timestamp rows.
            packed_raw_body = [
                excerpt
                for excerpt in body
                if not is_source_metadata_text(excerpt)
            ]
            answer_value_coverage = answer_value_component_coverage(
                question.answer,
                len(question.evidence_sources),
                packed_raw_body,
            )
            expected_sources = set(question.evidence_sources)
            retrieved_sources = {source for source in body_sources if source}
            raw_retrieved_source_ids = list(
                getattr(mc, "last_raw_graph_source_ids", [])
            )
            raw_retrieved_sources = set(raw_retrieved_source_ids)
            evidence_coverage = (
                len(expected_sources & retrieved_sources) / len(expected_sources)
                if expected_sources
                else None
            )
            raw_evidence_coverage = (
                len(expected_sources & raw_retrieved_sources) / len(expected_sources)
                if expected_sources and raw_retrieved_source_ids
                else None
            )
            causal_stats = getattr(mc, "causal_consolidation_stats", {})
            staging_stats = causal_stats.get("staging", {})
            learning_stats = causal_stats.get("learning", {})
            qwen_stats = getattr(mc, "last_source_rerank_report", {})
            coverage_stats = getattr(
                mc,
                "last_coverage_selection_report",
                {},
            )
            score_provider_stats = coverage_stats.get(
                "score_provider_report",
                {},
            ) or {}
            coverage_candidate_trace = []
            for row in getattr(mc, "last_coverage_candidate_trace", ()):
                diagnostic = dict(row)
                source_id = str(diagnostic.get("source_id") or "")
                diagnostic["required_source"] = (
                    source_id in expected_sources if expected_sources else None
                )
                coverage_candidate_trace.append(diagnostic)
            closure_rows = [
                row
                for row in coverage_candidate_trace
                if row.get("post_coverage_closure_applied") is True
            ]
            closure_applied = bool(closure_rows)
            closure_scopes = {
                str(row.get("closure_scope") or "")
                for row in closure_rows
            }
            closure_guarantees = {
                row.get("closure_global_recall_guaranteed")
                for row in closure_rows
            }
            closure_scope = (
                next(iter(closure_scopes))
                if len(closure_scopes) == 1
                else ("inconsistent" if closure_scopes else "")
            )
            closure_guarantee = (
                next(iter(closure_guarantees))
                if len(closure_guarantees) == 1
                else None
            )
            closure_global_recall_guaranteed = (
                closure_guarantee
                if closure_applied
                and isinstance(closure_guarantee, bool)
                else None
            )
            companion_stats = getattr(mc, "last_source_companion_report", {})
            partition_stats = getattr(mc, "last_partition_routing_report", {})
            out.append(
                QuestionRecall(
                    question_id=question.question_id,
                    category=question.category or "",
                    in_haystack=contains_answer(haystack_texts, question.answer),
                    in_context=contains_answer(everything, question.answer),
                    best_f1=best_f1(everything, question.answer),
                    in_memory_header=contains_answer(header, question.answer),
                    in_expansions=contains_answer(body, question.answer),
                    context_tokens=sum(count_tokens(t) for t in everything),
                    evidence_source_hit=(
                        bool(expected_sources & retrieved_sources)
                        if expected_sources
                        else None
                    ),
                    evidence_source_recall=evidence_coverage,
                    all_evidence_sources=(
                        evidence_coverage == 1.0
                        if evidence_coverage is not None
                        else None
                    ),
                    retrieved_source_ids=list(
                        dict.fromkeys(source for source in body_sources if source)
                    ),
                    raw_evidence_source_recall=raw_evidence_coverage,
                    raw_all_evidence_sources=(
                        raw_evidence_coverage == 1.0
                        if raw_evidence_coverage is not None
                        else None
                    ),
                    raw_retrieved_source_ids=raw_retrieved_source_ids,
                    answer_value_components_expected=(
                        answer_value_coverage.expected
                        if answer_value_coverage is not None
                        else None
                    ),
                    answer_value_components_found=(
                        answer_value_coverage.found
                        if answer_value_coverage is not None
                        else None
                    ),
                    answer_value_component_recall=(
                        answer_value_coverage.recall
                        if answer_value_coverage is not None
                        else None
                    ),
                    all_answer_value_components=(
                        answer_value_coverage.all_components
                        if answer_value_coverage is not None
                        else None
                    ),
                    answer_value_component_hit_mask=(
                        list(answer_value_coverage.hit_mask)
                        if answer_value_coverage is not None
                        else []
                    ),
                    answer_value_metric_kind=(
                        answer_value_coverage.metric_kind
                        if answer_value_coverage is not None
                        else ""
                    ),
                    source_companion_requested=list(
                        companion_stats.get("requested_sources", [])
                    ),
                    source_companion_hydrated=list(
                        companion_stats.get("hydrated_sources", [])
                    ),
                    source_companion_orphans=list(
                        companion_stats.get("orphan_sources", [])
                    ),
                    source_companion_direct_date_retained=int(
                        companion_stats.get("direct_date_retained", 0)
                    ),
                    source_companion_candidates_before=int(
                        companion_stats.get("candidate_count_before", 0)
                    ),
                    source_companion_candidates_after=int(
                        companion_stats.get("candidate_count_after", 0)
                    ),
                    selected_partitions=list(
                        partition_stats.get("selected_partitions", [])
                    ),
                    partition_ranking=list(
                        partition_stats.get("partition_ranking", [])
                    ),
                    direct_chunks=sum(not value for value in body_is_consolidation),
                    consolidation_chunks=sum(body_is_consolidation),
                    causal_events=int(staging_stats.get("events", 0)),
                    causal_graph_edges=int(
                        learning_stats.get("graph", {}).get("edges", 0)
                    ),
                    causal_write_s=float(staging_stats.get("elapsed_s", 0.0))
                    + float(learning_stats.get("elapsed_s", 0.0)),
                    qwen_rerank_passes=int(qwen_stats.get("passes", 0)),
                    qwen_candidate_inspections=int(
                        qwen_stats.get("total_candidate_inspections", 0)
                    ),
                    qwen_max_workspace_candidates=int(
                        qwen_stats.get("max_workspace_candidates", 0)
                    ),
                    qwen_max_workspace_tokens=int(
                        qwen_stats.get("max_workspace_tokens", 0)
                    ),
                    qwen_candidates_added=int(
                        qwen_stats.get("qwen_candidates_added", 0)
                    ),
                    qwen_feedback_rounds=int(
                        qwen_stats.get("feedback_rounds", 0)
                    ),
                    qwen_feedback_seed_sources=int(
                        qwen_stats.get("feedback_seed_sources", 0)
                    ),
                    qwen_feedback_candidates_added=int(
                        qwen_stats.get("feedback_candidates_added", 0)
                    ),
                    qwen_feedback_activation_candidates=int(
                        qwen_stats.get("feedback_activation_candidates", 0)
                    ),
                    qwen_feedback_query_tokens=int(
                        qwen_stats.get("feedback_query_tokens", 0)
                    ),
                    coverage_selector_inspected=int(
                        coverage_stats.get("inspected_candidates", 0)
                    ),
                    coverage_selector_classified=int(
                        coverage_stats.get("classified_candidates", 0)
                    ),
                    coverage_selector_clusters=int(
                        coverage_stats.get("event_clusters", 0)
                    ),
                    coverage_selector_null=int(
                        coverage_stats.get("null_assignments", 0)
                    ),
                    coverage_selector_uncertain=int(
                        coverage_stats.get("uncertain_assignments", 0)
                    ),
                    coverage_selector_output=int(
                        coverage_stats.get("output_candidates", 0)
                    ),
                    coverage_selector_representatives=int(
                        coverage_stats.get("representatives", 0)
                    ),
                    coverage_selector_workspace_tokens=int(
                        coverage_stats.get("workspace_tokens", 0)
                    ),
                    coverage_selector_elapsed_s=float(
                        coverage_stats.get("elapsed_s", 0.0)
                    ),
                    coverage_selector_operator=str(
                        coverage_stats.get("operator", "")
                    ),
                    coverage_selector_cardinality=coverage_stats.get(
                        "cardinality"
                    ),
                    coverage_selector_quantifier=str(
                        coverage_stats.get("quantifier", "")
                    ),
                    coverage_selector_ordering=str(
                        coverage_stats.get("ordering", "")
                    ),
                    coverage_selector_query_timestamp=coverage_stats.get(
                        "query_timestamp"
                    ),
                    coverage_selector_temporal_window_days=coverage_stats.get(
                        "temporal_window_days"
                    ),
                    coverage_selector_posterior_kind=str(
                        coverage_stats.get("posterior_kind", "")
                    ),
                    coverage_selector_semantic_score_kind=str(
                        coverage_stats.get("semantic_score_kind", "")
                    ),
                    coverage_selector_answerability_score_kind=str(
                        coverage_stats.get("answerability_score_kind", "")
                    ),
                    coverage_selector_frontier_candidates=int(
                        coverage_stats.get("frontier_candidates", 0)
                    ),
                    coverage_selector_frontier_attempted=int(
                        coverage_stats.get("frontier_attempted", 0)
                    ),
                    coverage_selector_frontier_uninspected=int(
                        coverage_stats.get("frontier_uninspected", 0)
                    ),
                    coverage_selector_frontier_exhaustive=bool(
                        coverage_stats.get("frontier_exhaustive", False)
                    ),
                    coverage_selector_frontier_batches=int(
                        coverage_stats.get("frontier_batches", 0)
                    ),
                    coverage_selector_routed_frontier_exhaustive=(
                        coverage_stats.get("routed_frontier_exhaustive")
                    ),
                    coverage_selector_active_partition_total=(
                        coverage_stats.get("active_partition_total")
                    ),
                    coverage_selector_active_partition_inspected=(
                        coverage_stats.get("active_partition_inspected")
                    ),
                    coverage_selector_active_partition_exhaustive=(
                        coverage_stats.get("active_partition_exhaustive")
                    ),
                    coverage_selector_active_partition_sources_total=(
                        coverage_stats.get("active_partition_sources_total")
                    ),
                    coverage_selector_active_partition_structural_rows=int(
                        coverage_stats.get(
                            "active_partition_structural_rows", 0
                        )
                        or 0
                    ),
                    coverage_selector_active_partition_structural_hypotheses=int(
                        coverage_stats.get(
                            "active_partition_structural_hypotheses", 0
                        )
                        or 0
                    ),
                    coverage_selector_active_partition_candidates_admitted=int(
                        coverage_stats.get(
                            "active_partition_candidates_admitted", 0
                        )
                        or 0
                    ),
                    coverage_selector_active_partition_candidates_already_present=int(
                        coverage_stats.get(
                            "active_partition_candidates_already_present", 0
                        )
                        or 0
                    ),
                    coverage_selector_active_partition_candidates_replaced=int(
                        coverage_stats.get(
                            "active_partition_candidates_replaced", 0
                        )
                        or 0
                    ),
                    coverage_selector_active_partition_candidates_truncated=int(
                        coverage_stats.get(
                            "active_partition_candidates_truncated", 0
                        )
                        or 0
                    ),
                    coverage_selector_active_partition_structural_overflow=int(
                        coverage_stats.get(
                            "active_partition_structural_overflow", 0
                        )
                        or 0
                    ),
                    coverage_selector_active_partition_scan_contract=str(
                        coverage_stats.get("active_partition_scan_contract", "")
                        or ""
                    ),
                    coverage_selector_active_partition_semantically_complete=(
                        coverage_stats.get(
                            "active_partition_semantically_complete"
                        )
                    ),
                    coverage_selector_partition_scope_kind=str(
                        coverage_stats.get("partition_scope_kind", "") or ""
                    ),
                    coverage_selector_partition_inventory_total=(
                        coverage_stats.get("partition_inventory_total")
                    ),
                    coverage_selector_selected_partition_count=(
                        coverage_stats.get("selected_partition_count")
                    ),
                    coverage_selector_partition_scope_exhaustive=(
                        coverage_stats.get("partition_scope_exhaustive")
                    ),
                    coverage_selector_selected_scope_structurally_complete=(
                        coverage_stats.get(
                            "selected_scope_structurally_complete"
                        )
                    ),
                    coverage_selector_global_semantic_complete=(
                        coverage_stats.get("global_semantic_complete")
                    ),
                    coverage_selector_allow_selected_scope_fixed_k_closure=bool(
                        coverage_stats.get(
                            "allow_selected_scope_fixed_k_closure",
                            False,
                        )
                    ),
                    closure_applied=closure_applied,
                    closure_scope=closure_scope,
                    closure_global_recall_guaranteed=(
                        closure_global_recall_guaranteed
                    ),
                    coverage_selector_cardinality_deficit=int(
                        coverage_stats.get("cardinality_deficit", 0) or 0
                    ),
                    coverage_selector_credible_clusters=int(
                        coverage_stats.get("credible_clusters", 0)
                    ),
                    coverage_selector_reserved_representatives=int(
                        coverage_stats.get("reserved_representatives", 0)
                    ),
                    coverage_selector_structural_eligible_clusters=int(
                        coverage_stats.get("structural_eligible_clusters", 0)
                    ),
                    coverage_selector_structural_reserved_representatives=int(
                        coverage_stats.get(
                            "structural_reserved_representatives",
                            0,
                        )
                    ),
                    coverage_selector_score_provider_fallback=str(
                        coverage_stats.get("score_provider_fallback", "")
                    ),
                    coverage_selector_score_provider_model_id=str(
                        score_provider_stats.get("model_id", "")
                    ),
                    coverage_selector_score_provider_model_revision=str(
                        score_provider_stats.get("model_revision", "")
                    ),
                    coverage_selector_score_provider_checkpoint_sha256=str(
                        score_provider_stats.get("checkpoint_sha256", "")
                    ),
                    coverage_selector_score_provider_device=str(
                        score_provider_stats.get("device", "")
                    ),
                    coverage_selector_score_provider_dtype=str(
                        score_provider_stats.get("dtype", "")
                    ),
                    coverage_selector_score_provider_forward_passes=int(
                        score_provider_stats.get("forward_passes", 0)
                    ),
                    coverage_selector_score_provider_peak_workspace_tokens=int(
                        score_provider_stats.get(
                            "peak_workspace_tokens",
                            score_provider_stats.get("workspace_tokens", 0),
                        )
                    ),
                    coverage_selector_score_provider_total_workspace_tokens=int(
                        score_provider_stats.get(
                            "total_workspace_tokens",
                            score_provider_stats.get("total_sequence_tokens", 0),
                        )
                    ),
                    coverage_selector_score_provider_elapsed_s=float(
                        score_provider_stats.get("elapsed_s", 0.0)
                    ),
                    coverage_selector_score_provider_retained_state_bytes=int(
                        score_provider_stats.get(
                            "retained_transformer_state_bytes",
                            0,
                        )
                    ),
                    coverage_selector_prefix_model_id=str(
                        coverage_stats.get("prefix_model_id", "")
                    ),
                    coverage_selector_prefix_model_revision=str(
                        coverage_stats.get("prefix_model_revision", "")
                    ),
                    coverage_selector_prefix_checkpoint_sha256=str(
                        coverage_stats.get("prefix_checkpoint_sha256", "")
                    ),
                    coverage_selector_prefix_device=str(
                        coverage_stats.get("prefix_device", "")
                    ),
                    coverage_selector_prefix_dtype=str(
                        coverage_stats.get("prefix_dtype", "")
                    ),
                    coverage_selector_prefix_layers=int(
                        coverage_stats.get("prefix_layers", 0) or 0
                    ),
                    coverage_selector_prefix_attention_layer=int(
                        coverage_stats.get("prefix_attention_layer", -1)
                    ),
                    coverage_selector_model_id=str(
                        coverage_stats.get("semantic_model_id", "")
                    ),
                    coverage_selector_model_revision=str(
                        coverage_stats.get("semantic_model_revision", "")
                    ),
                    coverage_selector_checkpoint_sha256=str(
                        coverage_stats.get("semantic_checkpoint_sha256", "")
                    ),
                    coverage_selector_semantic_inspected=int(
                        coverage_stats.get("semantic_inspected_candidates", 0)
                    ),
                    coverage_selector_semantic_workspace_tokens=int(
                        coverage_stats.get("semantic_workspace_tokens", 0)
                    ),
                    coverage_selector_semantic_elapsed_s=float(
                        coverage_stats.get("semantic_elapsed_s", 0.0)
                    ),
                    coverage_selector_retained_state_bytes=int(
                        coverage_stats.get("retained_transformer_state_bytes", 0)
                    ),
                    coverage_selector_status=str(
                        coverage_stats.get("selection_status", "")
                    ),
                    coverage_selector_bypass_reason=str(
                        coverage_stats.get("bypass_reason", "")
                    ),
                    coverage_selector_fallback_reason=str(
                        coverage_stats.get("fallback_reason", "")
                    ),
                    coverage_candidate_trace=coverage_candidate_trace,
                    survives_horizon=_survival(mc, question.answer, horizons_turns),
                )
            )
        return out
    finally:
        mc.close()


def run_recall(
    samples: list[BenchmarkSample],
    config: EvalConfig,
    benchmark: str = "",
    max_samples: int | None = None,
    ingest_fn: IngestFn = ingest_sample,
    horizons_turns=DEFAULT_HORIZONS_TURNS,
    question_offset: int = 0,
    max_questions: int | None = None,
) -> RecallReport:
    """Measure answer reachability across samples. Zero API calls."""
    if question_offset < 0:
        raise ValueError("question_offset must be non-negative")
    if max_questions is not None and max_questions < 1:
        raise ValueError("max_questions must be positive when supplied")
    selected = samples[:max_samples] if max_samples else samples
    effective_ingest_fn = (
        shared_embedding_ingest_fn(config.embedding_device)
        if ingest_fn is ingest_sample
        else ingest_fn
    )

    results: list[QuestionRecall] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, sample in enumerate(selected):
            print(f"  [{i + 1}/{len(selected)}] {sample.sample_id} "
                  f"({len(sample.turns)} turns, {len(sample.questions)} questions)...")
            results.extend(
                measure_sample(
                    sample,
                    config,
                    Path(tmpdir) / f"sample_{i}",
                    ingest_fn=effective_ingest_fn,
                    horizons_turns=horizons_turns,
                    question_offset=question_offset,
                    max_questions=max_questions,
                )
            )
            if config.retrieval.mode in {"causal_consolidation", "causal_graph"}:
                # hnswlib and Pydantic chunk graphs may contain native/cyclic
                # allocations that CPython does not reclaim at frame exit.
                gc.collect()

    n = len(results)
    by_category: dict[str, list[bool]] = {}
    for r in results:
        by_category.setdefault(r.category or "uncategorized", []).append(r.in_context)

    source_any_scored = [
        result.evidence_source_hit
        for result in results
        if result.evidence_source_hit is not None
    ]
    source_coverage_scored = [
        result.evidence_source_recall
        for result in results
        if result.evidence_source_recall is not None
    ]
    source_all_scored = [
        result.all_evidence_sources
        for result in results
        if result.all_evidence_sources is not None
    ]
    raw_source_coverage_scored = [
        result.raw_evidence_source_recall
        for result in results
        if result.raw_evidence_source_recall is not None
    ]
    raw_source_all_scored = [
        result.raw_all_evidence_sources
        for result in results
        if result.raw_all_evidence_sources is not None
    ]
    answer_value_coverage_scored = [
        result.answer_value_component_recall
        for result in results
        if result.answer_value_component_recall is not None
    ]
    answer_value_all_scored = [
        result.all_answer_value_components
        for result in results
        if result.all_answer_value_components is not None
    ]
    return RecallReport(
        benchmark=benchmark,
        mode=config.retrieval.mode,
        k=config.retrieval.k,
        n_questions=n,
        haystack_recall=_frac(r.in_haystack for r in results),
        recall=_frac(r.in_context for r in results),
        mean_best_f1=(sum(r.best_f1 for r in results) / n) if n else 0.0,
        header_recall=_frac(r.in_memory_header for r in results),
        expansion_recall=_frac(r.in_expansions for r in results),
        mean_context_tokens=(sum(r.context_tokens for r in results) / n) if n else 0.0,
        evidence_source_recall=(
            sum(source_coverage_scored) / len(source_coverage_scored)
            if source_coverage_scored
            else None
        ),
        evidence_any_source_recall=(
            _frac(source_any_scored) if source_any_scored else None
        ),
        evidence_all_source_recall=(
            _frac(source_all_scored) if source_all_scored else None
        ),
        raw_evidence_source_recall=(
            sum(raw_source_coverage_scored) / len(raw_source_coverage_scored)
            if raw_source_coverage_scored
            else None
        ),
        raw_evidence_all_source_recall=(
            _frac(raw_source_all_scored) if raw_source_all_scored else None
        ),
        answer_value_component_recall=(
            sum(answer_value_coverage_scored)
            / len(answer_value_coverage_scored)
            if answer_value_coverage_scored
            else None
        ),
        answer_value_all_component_recall=(
            _frac(answer_value_all_scored) if answer_value_all_scored else None
        ),
        answer_value_scored_questions=len(answer_value_coverage_scored),
        survival_by_horizon={
            days: _frac(r.survives_horizon.get(days, False) for r in results)
            for days in horizons_turns
        },
        by_category={k: _frac(v) for k, v in sorted(by_category.items())},
        questions=results,
    )


def _frac(flags) -> float:
    flags = list(flags)
    return (sum(1 for f in flags if f) / len(flags)) if flags else 0.0
