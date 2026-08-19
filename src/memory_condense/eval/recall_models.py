"""Recall result contracts and their derived aggregate invariants."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

DEFAULT_HORIZONS_TURNS = (0, 15, 30, 45)


@dataclass(frozen=True)
class AnswerValueCoverage:
    """Post-packing coverage of a high-confidence multi-value gold answer."""

    expected: int
    found: int
    recall: float
    all_components: bool
    hit_mask: tuple[bool, ...]
    metric_kind: str


class QuestionRecall(BaseModel):
    """Whether one question's answer was reachable, from where, and at what cost."""

    question_id: str
    category: str = ""
    in_haystack: bool = False
    in_context: bool = False
    best_f1: float = 0.0
    in_memory_header: bool = False
    in_expansions: bool = False
    #: tiktoken count of the assembled context. Load-bearing: condensation's
    #: claim is *the same answer for fewer tokens*, so recall alone cannot
    #: show its benefit — and can make a system that spends 10x look better.
    context_tokens: int = 0
    #: Source-level diagnostics are scored only when the benchmark supplies
    #: gold evidence source IDs. They measure retrieval, not answer wording.
    evidence_source_hit: bool | None = None
    evidence_source_recall: float | None = None
    all_evidence_sources: bool | None = None
    retrieved_source_ids: list[str] = Field(default_factory=list)
    raw_evidence_source_recall: float | None = None
    raw_all_evidence_sources: bool | None = None
    raw_retrieved_source_ids: list[str] = Field(default_factory=list)
    #: Evaluation-only value coverage over final packed raw expansions. Gold
    #: is parsed only after retrieval and packing, and never reaches a live
    #: retriever or selector. ``None`` means the answer was not a safely
    #: parseable multi-value list and is excluded from aggregate denominators.
    answer_value_components_expected: int | None = None
    answer_value_components_found: int | None = None
    answer_value_component_recall: float | None = None
    all_answer_value_components: bool | None = None
    answer_value_component_hit_mask: list[bool] = Field(default_factory=list)
    answer_value_metric_kind: str = ""
    source_companion_requested: list[str] = Field(default_factory=list)
    source_companion_hydrated: list[str] = Field(default_factory=list)
    source_companion_orphans: list[str] = Field(default_factory=list)
    source_companion_direct_date_retained: int = 0
    source_companion_candidates_before: int = 0
    source_companion_candidates_after: int = 0
    selected_partitions: list[str] = Field(default_factory=list)
    partition_ranking: list[dict[str, str | int | float]] = Field(
        default_factory=list
    )
    direct_chunks: int = 0
    consolidation_chunks: int = 0
    causal_events: int = 0
    causal_graph_edges: int = 0
    causal_write_s: float = 0.0
    qwen_rerank_passes: int = 0
    qwen_candidate_inspections: int = 0
    qwen_max_workspace_candidates: int = 0
    qwen_max_workspace_tokens: int = 0
    qwen_candidates_added: int = 0
    qwen_feedback_rounds: int = 0
    qwen_feedback_seed_sources: int = 0
    qwen_feedback_candidates_added: int = 0
    qwen_feedback_activation_candidates: int = 0
    qwen_feedback_query_tokens: int = 0
    coverage_selector_inspected: int = 0
    coverage_selector_classified: int = 0
    coverage_selector_clusters: int = 0
    coverage_selector_null: int = 0
    coverage_selector_uncertain: int = 0
    coverage_selector_output: int = 0
    coverage_selector_representatives: int = 0
    coverage_selector_workspace_tokens: int = 0
    coverage_selector_elapsed_s: float = 0.0
    coverage_selector_operator: str = ""
    coverage_selector_cardinality: int | None = None
    coverage_selector_quantifier: str = ""
    coverage_selector_ordering: str = ""
    coverage_selector_query_timestamp: str | None = None
    coverage_selector_temporal_window_days: int | None = None
    coverage_selector_posterior_kind: str = ""
    coverage_selector_semantic_score_kind: str = ""
    coverage_selector_answerability_score_kind: str = ""
    coverage_selector_frontier_candidates: int = 0
    coverage_selector_frontier_attempted: int = 0
    coverage_selector_frontier_uninspected: int = 0
    coverage_selector_frontier_exhaustive: bool = False
    coverage_selector_frontier_batches: int = 0
    coverage_selector_routed_frontier_exhaustive: bool | None = None
    coverage_selector_active_partition_total: int | None = None
    coverage_selector_active_partition_inspected: int | None = None
    coverage_selector_active_partition_exhaustive: bool | None = None
    coverage_selector_active_partition_sources_total: int | None = None
    coverage_selector_active_partition_structural_rows: int = 0
    coverage_selector_active_partition_structural_hypotheses: int = 0
    coverage_selector_active_partition_candidates_admitted: int = 0
    coverage_selector_active_partition_candidates_already_present: int = 0
    coverage_selector_active_partition_candidates_replaced: int = 0
    coverage_selector_active_partition_candidates_truncated: int = 0
    coverage_selector_active_partition_structural_overflow: int = 0
    coverage_selector_active_partition_scan_contract: str = ""
    coverage_selector_active_partition_semantically_complete: bool | None = None
    coverage_selector_partition_scope_kind: str = ""
    coverage_selector_partition_inventory_total: int | None = None
    coverage_selector_selected_partition_count: int | None = None
    coverage_selector_partition_scope_exhaustive: bool | None = None
    coverage_selector_selected_scope_structurally_complete: bool | None = None
    coverage_selector_global_semantic_complete: bool | None = None
    coverage_selector_allow_selected_scope_fixed_k_closure: bool = False
    closure_applied: bool = False
    closure_scope: str = ""
    closure_global_recall_guaranteed: bool | None = None
    coverage_selector_cardinality_deficit: int = 0
    coverage_selector_credible_clusters: int = 0
    coverage_selector_reserved_representatives: int = 0
    coverage_selector_structural_eligible_clusters: int = 0
    coverage_selector_structural_reserved_representatives: int = 0
    coverage_selector_score_provider_fallback: str = ""
    coverage_selector_score_provider_model_id: str = ""
    coverage_selector_score_provider_model_revision: str = ""
    coverage_selector_score_provider_checkpoint_sha256: str = ""
    coverage_selector_score_provider_device: str = ""
    coverage_selector_score_provider_dtype: str = ""
    coverage_selector_score_provider_forward_passes: int = 0
    coverage_selector_score_provider_peak_workspace_tokens: int = 0
    coverage_selector_score_provider_total_workspace_tokens: int = 0
    coverage_selector_score_provider_elapsed_s: float = 0.0
    coverage_selector_score_provider_retained_state_bytes: int = 0
    coverage_selector_prefix_model_id: str = ""
    coverage_selector_prefix_model_revision: str = ""
    coverage_selector_prefix_checkpoint_sha256: str = ""
    coverage_selector_prefix_device: str = ""
    coverage_selector_prefix_dtype: str = ""
    coverage_selector_prefix_layers: int = 0
    coverage_selector_prefix_attention_layer: int = -1
    coverage_selector_model_id: str = ""
    coverage_selector_model_revision: str = ""
    coverage_selector_checkpoint_sha256: str = ""
    coverage_selector_semantic_inspected: int = 0
    coverage_selector_semantic_workspace_tokens: int = 0
    coverage_selector_semantic_elapsed_s: float = 0.0
    coverage_selector_retained_state_bytes: int = 0
    coverage_selector_status: str = ""
    coverage_selector_bypass_reason: str = ""
    coverage_selector_fallback_reason: str = ""
    #: Text-free per-candidate loss ledger. ``required_source`` is joined by
    #: the evaluator only after retrieval, selection, and packing have run;
    #: the live selector never receives benchmark gold.
    coverage_candidate_trace: list[
        dict[str, str | int | float | bool | None]
    ] = Field(default_factory=list)
    #: ``{horizon_turns: answer_still_in_a_non_cold_item}``
    survives_horizon: dict[int, bool] = Field(default_factory=dict)


class RecallReport(BaseModel):
    """Aggregate answer-reachability for one config over one benchmark.

    Reports **recall and cost together**. Condensation's claim is not "finds
    the answer more often" but "finds it as often for fewer tokens", so a
    recall-only comparison structurally cannot show its benefit, and rewards
    whichever arm simply sends more text. ``recall_per_1k_tokens`` is the
    efficiency figure; ``mean_context_tokens`` is what it is normalised by.
    """

    benchmark: str = ""
    mode: str = "dense"
    k: int = 10
    n_questions: int = 0
    haystack_recall: float = 0.0
    recall: float = 0.0
    mean_best_f1: float = 0.0
    header_recall: float = 0.0
    expansion_recall: float = 0.0
    mean_context_tokens: float = 0.0
    #: Mean fraction of each question's required evidence sources retrieved.
    evidence_source_recall: float | None = None
    evidence_any_source_recall: float | None = None
    evidence_all_source_recall: float | None = None
    raw_evidence_source_recall: float | None = None
    raw_evidence_all_source_recall: float | None = None
    #: Macro mean over safely parsed multi-value questions only. This is
    #: intentionally separate from source-ID coverage: equivalent answer
    #: evidence may be packed from another source, while an expected source
    #: can be present without carrying the requested value.
    answer_value_component_recall: float | None = None
    answer_value_all_component_recall: float | None = None
    answer_value_scored_questions: int = 0
    #: Derived, mutually interpretable coverage-control counts. Selector and
    #: score-provider failures are separate; ``degraded`` is their per-call
    #: union, not their sum.
    coverage_selector_calls: int = 0
    coverage_selector_bypasses: int = 0
    coverage_selector_fallbacks: int = 0
    coverage_score_provider_fallbacks: int = 0
    coverage_degraded_calls: int = 0
    coverage_routed_frontier_audited_calls: int = 0
    coverage_routed_frontier_exhaustive_calls: int = 0
    coverage_routed_frontier_non_exhaustive_calls: int = 0
    coverage_active_partition_audited_calls: int = 0
    coverage_active_partition_exhaustive_calls: int = 0
    coverage_active_partition_non_exhaustive_calls: int = 0
    coverage_active_partition_semantically_complete_calls: int = 0
    coverage_active_partition_semantically_incomplete_calls: int = 0
    coverage_active_partition_candidates_admitted_total: int = 0
    coverage_active_partition_structural_overflow_total: int = 0
    coverage_selected_scope_structurally_complete_calls: int = 0
    coverage_global_semantic_complete_calls: int = 0
    coverage_closure_calls: int = 0
    coverage_selected_scope_policy_closure_calls: int = 0
    coverage_global_recall_guaranteed_closure_calls: int = 0
    coverage_cardinality_deficit_calls: int = 0
    coverage_cardinality_deficit_total: int = 0
    survival_by_horizon: dict[int, float] = Field(default_factory=dict)
    by_category: dict[str, float] = Field(default_factory=dict)
    questions: list[QuestionRecall] = Field(default_factory=list)

    def model_post_init(self, __context) -> None:
        """Keep summary counters derived from their per-question provenance."""

        if not self.questions:
            return
        calls = [
            question
            for question in self.questions
            if (
                question.coverage_selector_inspected > 0
                or question.coverage_selector_frontier_candidates > 0
                or bool(question.coverage_selector_operator)
                or bool(question.coverage_selector_status)
                or bool(question.coverage_selector_bypass_reason)
                or bool(question.coverage_selector_fallback_reason)
                or bool(question.coverage_selector_score_provider_fallback)
            )
        ]

        def is_bypassed(question: QuestionRecall) -> bool:
            return (
                question.coverage_selector_status == "bypassed"
                or bool(question.coverage_selector_bypass_reason)
            )

        bypassed = [question for question in calls if is_bypassed(question)]
        selector_fallbacks = sum(
            bool(question.coverage_selector_fallback_reason)
            and not is_bypassed(question)
            for question in calls
        )
        provider_fallbacks = sum(
            bool(question.coverage_selector_score_provider_fallback)
            for question in calls
        )
        degraded = sum(
            bool(
                (
                    question.coverage_selector_fallback_reason
                    and not is_bypassed(question)
                )
                or question.coverage_selector_score_provider_fallback
            )
            for question in calls
        )
        routed_audited = [
            question
            for question in calls
            if not is_bypassed(question)
            and question.coverage_selector_routed_frontier_exhaustive is not None
        ]
        active_audited = [
            question
            for question in calls
            if not is_bypassed(question)
            and question.coverage_selector_active_partition_exhaustive is not None
        ]
        self.coverage_selector_calls = len(calls)
        self.coverage_selector_bypasses = len(bypassed)
        self.coverage_selector_fallbacks = selector_fallbacks
        self.coverage_score_provider_fallbacks = provider_fallbacks
        self.coverage_degraded_calls = degraded
        self.coverage_routed_frontier_audited_calls = len(routed_audited)
        self.coverage_routed_frontier_exhaustive_calls = sum(
            question.coverage_selector_routed_frontier_exhaustive is True
            for question in routed_audited
        )
        self.coverage_routed_frontier_non_exhaustive_calls = sum(
            question.coverage_selector_routed_frontier_exhaustive is False
            for question in routed_audited
        )
        self.coverage_active_partition_audited_calls = len(active_audited)
        self.coverage_active_partition_exhaustive_calls = sum(
            question.coverage_selector_active_partition_exhaustive is True
            for question in active_audited
        )
        self.coverage_active_partition_non_exhaustive_calls = sum(
            question.coverage_selector_active_partition_exhaustive is False
            for question in active_audited
        )
        self.coverage_active_partition_semantically_complete_calls = sum(
            question.coverage_selector_active_partition_semantically_complete is True
            for question in active_audited
        )
        self.coverage_active_partition_semantically_incomplete_calls = sum(
            question.coverage_selector_active_partition_semantically_complete is False
            for question in active_audited
        )
        self.coverage_active_partition_candidates_admitted_total = sum(
            question.coverage_selector_active_partition_candidates_admitted
            for question in active_audited
        )
        self.coverage_active_partition_structural_overflow_total = sum(
            question.coverage_selector_active_partition_structural_overflow
            for question in active_audited
        )
        self.coverage_selected_scope_structurally_complete_calls = sum(
            question.coverage_selector_selected_scope_structurally_complete
            is True
            for question in calls
        )
        self.coverage_global_semantic_complete_calls = sum(
            question.coverage_selector_global_semantic_complete is True
            for question in calls
        )
        self.coverage_closure_calls = sum(
            question.closure_applied for question in calls
        )
        self.coverage_selected_scope_policy_closure_calls = sum(
            question.closure_scope == "selected_scope_policy"
            for question in calls
        )
        self.coverage_global_recall_guaranteed_closure_calls = sum(
            question.closure_applied
            and question.closure_global_recall_guaranteed is True
            for question in calls
        )
        self.coverage_cardinality_deficit_calls = sum(
            question.coverage_selector_cardinality_deficit > 0
            for question in calls
        )
        self.coverage_cardinality_deficit_total = sum(
            question.coverage_selector_cardinality_deficit
            for question in calls
        )

    @property
    def recall_per_1k_tokens(self) -> float:
        """Recall points earned per 1,000 tokens of context spent."""
        if not self.mean_context_tokens:
            return 0.0
        return self.recall * 100.0 / (self.mean_context_tokens / 1000.0)
