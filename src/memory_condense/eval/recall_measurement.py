"""Stateful per-sample measurement and benchmark recall orchestration."""

from __future__ import annotations

import gc
import tempfile
from pathlib import Path
from typing import Callable

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
from memory_condense.ingest.loader import BenchmarkQuestion, BenchmarkSample
from memory_condense.search.packing.context_packer import is_source_metadata_text

# --- coverage-selector report transcription -------------------------------
#
# Each row maps one stats-dict entry onto one ``QuestionRecall`` field via a
# coercer, replacing a hand-written field-by-field transcription. A ``None``
# key means the stats key equals the field suffix (prefix-identity); renames
# carry the stats key explicitly.

_Coercer = Callable[[dict, str], object]


def _int(stats: dict, key: str) -> int:
    return int(stats.get(key, 0))


def _int_or_zero(stats: dict, key: str) -> int:
    return int(stats.get(key, 0) or 0)


def _int_neg_one(stats: dict, key: str) -> int:
    return int(stats.get(key, -1))


def _float(stats: dict, key: str) -> float:
    return float(stats.get(key, 0.0))


def _str(stats: dict, key: str) -> str:
    return str(stats.get(key, ""))


def _str_or_empty(stats: dict, key: str) -> str:
    return str(stats.get(key, "") or "")


def _bool(stats: dict, key: str) -> bool:
    return bool(stats.get(key, False))


def _raw(stats: dict, key: str):
    return stats.get(key)


#: ``coverage_selector_<suffix>`` <- coverage-selection report entries.
_COVERAGE_SELECTOR_FIELDS: tuple[tuple[str, str | None, _Coercer], ...] = (
    ("inspected", "inspected_candidates", _int),
    ("classified", "classified_candidates", _int),
    ("clusters", "event_clusters", _int),
    ("null", "null_assignments", _int),
    ("uncertain", "uncertain_assignments", _int),
    ("output", "output_candidates", _int),
    ("representatives", None, _int),
    ("workspace_tokens", None, _int),
    ("elapsed_s", None, _float),
    ("operator", None, _str),
    ("cardinality", None, _raw),
    ("quantifier", None, _str),
    ("ordering", None, _str),
    ("query_timestamp", None, _raw),
    ("temporal_window_days", None, _raw),
    ("posterior_kind", None, _str),
    ("semantic_score_kind", None, _str),
    ("answerability_score_kind", None, _str),
    ("frontier_candidates", None, _int),
    ("frontier_attempted", None, _int),
    ("frontier_uninspected", None, _int),
    ("frontier_exhaustive", None, _bool),
    ("frontier_batches", None, _int),
    ("routed_frontier_exhaustive", None, _raw),
    ("active_partition_total", None, _raw),
    ("active_partition_inspected", None, _raw),
    ("active_partition_exhaustive", None, _raw),
    ("active_partition_sources_total", None, _raw),
    ("active_partition_structural_rows", None, _int_or_zero),
    ("active_partition_structural_hypotheses", None, _int_or_zero),
    ("active_partition_candidates_admitted", None, _int_or_zero),
    ("active_partition_candidates_already_present", None, _int_or_zero),
    ("active_partition_candidates_replaced", None, _int_or_zero),
    ("active_partition_candidates_truncated", None, _int_or_zero),
    ("active_partition_structural_overflow", None, _int_or_zero),
    ("active_partition_scan_contract", None, _str_or_empty),
    ("active_partition_semantically_complete", None, _raw),
    ("partition_scope_kind", None, _str_or_empty),
    ("partition_inventory_total", None, _raw),
    ("selected_partition_count", None, _raw),
    ("partition_scope_exhaustive", None, _raw),
    ("selected_scope_structurally_complete", None, _raw),
    ("global_semantic_complete", None, _raw),
    ("allow_selected_scope_fixed_k_closure", None, _bool),
    ("cardinality_deficit", None, _int_or_zero),
    ("credible_clusters", None, _int),
    ("reserved_representatives", None, _int),
    ("structural_eligible_clusters", None, _int),
    ("structural_reserved_representatives", None, _int),
    ("score_provider_fallback", None, _str),
    ("prefix_model_id", None, _str),
    ("prefix_model_revision", None, _str),
    ("prefix_checkpoint_sha256", None, _str),
    ("prefix_device", None, _str),
    ("prefix_dtype", None, _str),
    ("prefix_layers", None, _int_or_zero),
    ("prefix_attention_layer", None, _int_neg_one),
    ("model_id", "semantic_model_id", _str),
    ("model_revision", "semantic_model_revision", _str),
    ("checkpoint_sha256", "semantic_checkpoint_sha256", _str),
    ("semantic_inspected", "semantic_inspected_candidates", _int),
    ("semantic_workspace_tokens", None, _int),
    ("semantic_elapsed_s", None, _float),
    ("retained_state_bytes", "retained_transformer_state_bytes", _int),
    ("status", "selection_status", _str),
    ("bypass_reason", None, _str),
    ("fallback_reason", None, _str),
)

#: ``coverage_selector_score_provider_<suffix>`` <- nested provider report.
_SCORE_PROVIDER_FIELDS: tuple[tuple[str, str | None, _Coercer], ...] = (
    ("model_id", None, _str),
    ("model_revision", None, _str),
    ("checkpoint_sha256", None, _str),
    ("device", None, _str),
    ("dtype", None, _str),
    ("forward_passes", None, _int),
    ("elapsed_s", None, _float),
    ("retained_state_bytes", "retained_transformer_state_bytes", _int),
)


def _coverage_selector_fields(coverage_stats: dict) -> dict:
    """Transcribe the coverage-selection report into ``QuestionRecall`` kwargs."""
    score_provider_stats = coverage_stats.get(
        "score_provider_report",
        {},
    ) or {}
    fields = {
        f"coverage_selector_{suffix}": coerce(coverage_stats, key or suffix)
        for suffix, key, coerce in _COVERAGE_SELECTOR_FIELDS
    }
    fields.update(
        {
            f"coverage_selector_score_provider_{suffix}": coerce(
                score_provider_stats, key or suffix
            )
            for suffix, key, coerce in _SCORE_PROVIDER_FIELDS
        }
    )
    # Dual-key legacy fallbacks do not fit the single-key table pattern.
    fields["coverage_selector_score_provider_peak_workspace_tokens"] = int(
        score_provider_stats.get(
            "peak_workspace_tokens",
            score_provider_stats.get("workspace_tokens", 0),
        )
    )
    fields["coverage_selector_score_provider_total_workspace_tokens"] = int(
        score_provider_stats.get(
            "total_workspace_tokens",
            score_provider_stats.get("total_sequence_tokens", 0),
        )
    )
    return fields


# --- per-question measurement phases --------------------------------------


def _capped_context(
    mc, query_text: str, config: EvalConfig
) -> tuple[list[str], list[str], list[str], list[bool]]:
    """Assemble the context and apply the hard prompt-budget cap."""
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
    return (
        header,
        body,
        body_sources[: len(body)],
        body_is_consolidation[: len(body)],
    )


def _answer_scoring_fields(
    mc,
    question: BenchmarkQuestion,
    haystack_texts: list[str],
    header: list[str],
    body: list[str],
    body_sources: list[str],
    expected_sources: set[str],
) -> dict:
    """Score answer reachability and evidence-source coverage."""
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
    scored = answer_value_coverage is not None
    return {
        "in_haystack": contains_answer(haystack_texts, question.answer),
        "in_context": contains_answer(everything, question.answer),
        "best_f1": best_f1(everything, question.answer),
        "in_memory_header": contains_answer(header, question.answer),
        "in_expansions": contains_answer(body, question.answer),
        "context_tokens": sum(count_tokens(t) for t in everything),
        "evidence_source_hit": (
            bool(expected_sources & retrieved_sources)
            if expected_sources
            else None
        ),
        "evidence_source_recall": evidence_coverage,
        "all_evidence_sources": (
            evidence_coverage == 1.0
            if evidence_coverage is not None
            else None
        ),
        "retrieved_source_ids": list(
            dict.fromkeys(source for source in body_sources if source)
        ),
        "raw_evidence_source_recall": raw_evidence_coverage,
        "raw_all_evidence_sources": (
            raw_evidence_coverage == 1.0
            if raw_evidence_coverage is not None
            else None
        ),
        "raw_retrieved_source_ids": raw_retrieved_source_ids,
        "answer_value_components_expected": (
            answer_value_coverage.expected if scored else None
        ),
        "answer_value_components_found": (
            answer_value_coverage.found if scored else None
        ),
        "answer_value_component_recall": (
            answer_value_coverage.recall if scored else None
        ),
        "all_answer_value_components": (
            answer_value_coverage.all_components if scored else None
        ),
        "answer_value_component_hit_mask": (
            list(answer_value_coverage.hit_mask) if scored else []
        ),
        "answer_value_metric_kind": (
            answer_value_coverage.metric_kind if scored else ""
        ),
    }


def _trace_and_closure_fields(mc, expected_sources: set[str]) -> dict:
    """Join gold sources onto the candidate trace and summarize closure rows."""
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
    return {
        "coverage_candidate_trace": coverage_candidate_trace,
        "closure_applied": closure_applied,
        "closure_scope": closure_scope,
        "closure_global_recall_guaranteed": (
            closure_guarantee
            if closure_applied and isinstance(closure_guarantee, bool)
            else None
        ),
    }


def _engine_diagnostic_fields(mc) -> dict:
    """Transcribe the engine's per-question diagnostic reports."""
    causal_stats = getattr(mc, "causal_consolidation_stats", {})
    staging_stats = causal_stats.get("staging", {})
    learning_stats = causal_stats.get("learning", {})
    qwen_stats = getattr(mc, "last_source_rerank_report", {})
    companion_stats = getattr(mc, "last_source_companion_report", {})
    partition_stats = getattr(mc, "last_partition_routing_report", {})
    coverage_stats = getattr(
        mc,
        "last_coverage_selection_report",
        {},
    )
    return {
        "source_companion_requested": list(
            companion_stats.get("requested_sources", [])
        ),
        "source_companion_hydrated": list(
            companion_stats.get("hydrated_sources", [])
        ),
        "source_companion_orphans": list(
            companion_stats.get("orphan_sources", [])
        ),
        "source_companion_direct_date_retained": int(
            companion_stats.get("direct_date_retained", 0)
        ),
        "source_companion_candidates_before": int(
            companion_stats.get("candidate_count_before", 0)
        ),
        "source_companion_candidates_after": int(
            companion_stats.get("candidate_count_after", 0)
        ),
        "selected_partitions": list(
            partition_stats.get("selected_partitions", [])
        ),
        "partition_ranking": list(
            partition_stats.get("partition_ranking", [])
        ),
        "causal_events": int(staging_stats.get("events", 0)),
        "causal_graph_edges": int(
            learning_stats.get("graph", {}).get("edges", 0)
        ),
        "causal_write_s": float(staging_stats.get("elapsed_s", 0.0))
        + float(learning_stats.get("elapsed_s", 0.0)),
        "qwen_rerank_passes": int(qwen_stats.get("passes", 0)),
        "qwen_candidate_inspections": int(
            qwen_stats.get("total_candidate_inspections", 0)
        ),
        "qwen_max_workspace_candidates": int(
            qwen_stats.get("max_workspace_candidates", 0)
        ),
        "qwen_max_workspace_tokens": int(
            qwen_stats.get("max_workspace_tokens", 0)
        ),
        "qwen_candidates_added": int(
            qwen_stats.get("qwen_candidates_added", 0)
        ),
        "qwen_feedback_rounds": int(
            qwen_stats.get("feedback_rounds", 0)
        ),
        "qwen_feedback_seed_sources": int(
            qwen_stats.get("feedback_seed_sources", 0)
        ),
        "qwen_feedback_candidates_added": int(
            qwen_stats.get("feedback_candidates_added", 0)
        ),
        "qwen_feedback_activation_candidates": int(
            qwen_stats.get("feedback_activation_candidates", 0)
        ),
        "qwen_feedback_query_tokens": int(
            qwen_stats.get("feedback_query_tokens", 0)
        ),
        **_coverage_selector_fields(coverage_stats),
    }


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
            header, body, body_sources, body_is_consolidation = (
                _capped_context(mc, question.dated_question, config)
            )
            expected_sources = set(question.evidence_sources)
            out.append(
                QuestionRecall(
                    question_id=question.question_id,
                    category=question.category or "",
                    direct_chunks=sum(
                        not value for value in body_is_consolidation
                    ),
                    consolidation_chunks=sum(body_is_consolidation),
                    survives_horizon=_survival(
                        mc, question.answer, horizons_turns
                    ),
                    **_answer_scoring_fields(
                        mc,
                        question,
                        haystack_texts,
                        header,
                        body,
                        body_sources,
                        expected_sources,
                    ),
                    **_trace_and_closure_fields(mc, expected_sources),
                    **_engine_diagnostic_fields(mc),
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
