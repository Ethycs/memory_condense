"""Retrieval-recall: can the gold answer even reach the prompt? No API calls.

Every benchmark question ships a gold answer string, so retrieval quality can
be measured **without generating anything**: ingest the haystack, assemble the
context the responder would see, and ask a purely local question — is the
answer recoverable from it?

This matters because it is the cheap predictor of the expensive run. If the
memory arm's context contains the answer *less often* than the dense arm's,
no responder can recover the difference, and the paid comparison's outcome is
knowable in advance for zero dollars.

It also answers the question that gated design Phase 4: **do COLD items hold
answers nothing else holds?** Before schema v4 that question was unanswerable
— decay counted wall-clock seconds, so an item needed 7–11.75 days of no
access to reach COLD while a run lasted minutes, and the horizons this module
projected to were mostly incapable of returning anything but zero. Decay now
counts *turns*, so a run advances the coordinate on its own and the tiers
populate for real. The forward projection below is consequently a genuine
extrapolation of a live signal rather than a workaround for a dead one.

Deliberately no LLM, no key, no network. It is a fifth CLI mode beside
``--compare``, and both are free.
"""

from __future__ import annotations

import gc
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from memory_condense import decay
from memory_condense._tokenizer import count_tokens
from memory_condense.context_packer import is_source_metadata_text
from memory_condense.eval.benchmark import (
    IngestFn,
    cap_context_to_prompt_budget,
    ingest_sample,
    shared_embedding_ingest_fn,
    normalize_answer,
    f1_score,
)
from memory_condense.eval.schemas import EvalConfig
from memory_condense.loader import BenchmarkSample

#: Simulated ages, **in turns**, at which to re-tier the items and ask whether
#: the answer survives. 0 is "now" (the transcript's current position).
#:
#: Chosen to straddle both crossing points at the default 30-turn half-life:
#: an ordinary item (seed 0.5) crosses into COLD at 30 untouched turns and an
#: important one (seed 0.8) at ~50. So 15 is inside WARM for both, 30 is the
#: first divider, and 45 separates them.
#:
#: 60 is deliberately **excluded**. Energy is clamped to ``<= 1.0`` and COLD
#: begins below ``0.25 = 1.0 * 0.5**2``, so two half-lives is the theoretical
#: ceiling for *any* unpinned item and that horizon can only ever report 0.0%.
#: The previous day-based set had exactly that defect in two of its four
#: entries; a horizon that cannot vary is not a measurement.
DEFAULT_HORIZONS_TURNS = (0, 15, 30, 45)

_NUMBERED_ANSWER_COMPONENT_RE = re.compile(
    r"(?<!\w)(?P<number>[1-9]\d*)[.)]\s+"
)
_ANSWER_VALUE_TOKEN_RECALL_NUMERATOR = 4
_ANSWER_VALUE_TOKEN_RECALL_DENOMINATOR = 5
_ANSWER_VALUE_MIN_FALLBACK_TOKENS = 4


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


def contains_answer(texts: list[str], gold: str) -> bool:
    """Whether ``gold`` appears in any of ``texts`` under SQuAD normalization.

    Containment, not equality: the context is a passage and the answer is a
    span inside it. Normalization (lowercase, strip articles and punctuation)
    is the same one the benchmark grades with, so this measures the same notion
    of "the answer is there" that F1 does.
    """
    needle = normalize_answer(gold)
    if not needle:
        return False
    return any(needle in normalize_answer(t) for t in texts)


def best_f1(texts: list[str], gold: str) -> float:
    """Highest token-F1 between the gold answer and any single context piece.

    A softer signal than containment: it still scores when the answer is
    present but reworded, which containment misses entirely.
    """
    return max((f1_score(t, gold) for t in texts), default=0.0)


def _parse_answer_value_components(
    gold: str,
    expected_count: int,
) -> tuple[list[str], str] | None:
    """Parse only list shapes whose cardinality is independently known.

    LongMemEval stores aggregate multi-answer golds as either numbered lists
    (whose items may themselves contain commas) or plain comma-separated
    lists. The benchmark's evidence-source count supplies an independent
    cardinality check. Derived numeric answers and ambiguous prose return
    ``None`` instead of being counted as misses.
    """

    if expected_count < 2 or not gold.strip():
        return None

    markers = list(_NUMBERED_ANSWER_COMPONENT_RE.finditer(gold))
    if markers:
        numbers = [int(marker.group("number")) for marker in markers]
        if len(markers) != expected_count or numbers != list(
            range(1, expected_count + 1)
        ):
            return None
        components = [
            gold[
                marker.end() : (
                    markers[index + 1].start()
                    if index + 1 < len(markers)
                    else len(gold)
                )
            ].strip(" \t\r\n,;")
            for index, marker in enumerate(markers)
        ]
        parse_kind = "numbered_list"
    else:
        components = [part.strip() for part in gold.split(",")]
        if len(components) != expected_count:
            return None
        parse_kind = "comma_list"

    normalized = [normalize_answer(component) for component in components]
    if (
        any(not component for component in normalized)
        or len(set(normalized)) != expected_count
        or any(not any(character.isalpha() for character in component) for component in components)
    ):
        return None
    return components, parse_kind


def _answer_value_component_in_excerpt(component: str, excerpt: str) -> bool:
    """Match one value within one excerpt using transparent lexical rules."""

    normalized_component = normalize_answer(component)
    normalized_excerpt = normalize_answer(excerpt)
    if not normalized_component or not normalized_excerpt:
        return False
    if normalized_component in normalized_excerpt:
        return True

    component_tokens = normalized_component.split()
    if len(component_tokens) < _ANSWER_VALUE_MIN_FALLBACK_TOKENS:
        return False
    # Preserve token order for the paraphrase fallback. Bag overlap falsely
    # treated "contemporary art ... museum of modern art" as evidence for the
    # distinct venue "Museum of Contemporary Art". An LCS still accepts mild
    # paraphrases such as "Queen live with Adam Lambert ..." while requiring
    # the identifying words to occur in a compatible sequence in one excerpt.
    excerpt_tokens = normalized_excerpt.split()
    previous = [0] * (len(excerpt_tokens) + 1)
    for component_token in component_tokens:
        current = [0]
        for index, excerpt_token in enumerate(excerpt_tokens, start=1):
            if component_token == excerpt_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    overlap = previous[-1]
    return (
        overlap * _ANSWER_VALUE_TOKEN_RECALL_DENOMINATOR
        >= len(component_tokens) * _ANSWER_VALUE_TOKEN_RECALL_NUMERATOR
    )


def answer_value_component_coverage(
    gold: str,
    evidence_source_count: int,
    packed_raw_excerpts: list[str],
) -> AnswerValueCoverage | None:
    """Measure explicit answer values across final packed raw excerpts.

    This intentionally ignores source and chunk identity: equivalent raw
    evidence can operationally provide the same answer value. Each component
    must occur within one excerpt; tokens are never assembled across chunks.
    The caller must supply the post-budget body with metadata rows removed.
    """

    parsed = _parse_answer_value_components(gold, evidence_source_count)
    if parsed is None:
        return None
    components, parse_kind = parsed
    hit_mask = tuple(
        any(
            _answer_value_component_in_excerpt(component, excerpt)
            for excerpt in packed_raw_excerpts
        )
        for component in components
    )
    found = sum(hit_mask)
    return AnswerValueCoverage(
        expected=len(components),
        found=found,
        recall=found / len(components),
        all_components=found == len(components),
        hit_mask=hit_mask,
        metric_kind=(
            f"{parse_kind}:normalized_literal_or_80pct_ordered_token_recall_same_excerpt"
        ),
    )


def _assemble(
    mc, question: str, config: EvalConfig
) -> tuple[list[str], list[str], list[str | None], list[bool]]:
    """Return header, body, and the body items' durable source IDs.

    ``reheat`` is off throughout: this is a measurement, and an item must not
    become hotter merely because a measurement looked at it.
    """
    mc.last_raw_graph_source_ids = []
    if config.retrieval.mode in {
        "memory",
        "causal_consolidation",
        "causal_graph",
    }:
        causal = config.retrieval.mode in {
            "causal_consolidation",
            "causal_graph",
        }
        graph_results = (
            mc.search_hybrid_graph(
                question,
                k=config.retrieval.k,
                neighbor_radius=config.retrieval.neighbor_radius,
                neighbor_slots=config.retrieval.neighbor_slots,
                neighbor_direction=config.retrieval.neighbor_direction,
                source_slots=config.retrieval.source_slots,
                source_candidate_pool=config.retrieval.source_candidate_pool,
                source_activation_k=config.retrieval.source_activation_k,
                query_facet_retrieval=config.retrieval.query_facet_retrieval,
                query_facet_slots=config.retrieval.query_facet_slots,
                query_facet_max=config.retrieval.query_facet_max,
                role_aware_retrieval=config.retrieval.role_aware_retrieval,
                role_user_weight=config.retrieval.role_user_weight,
                role_assistant_weight=config.retrieval.role_assistant_weight,
                role_system_weight=config.retrieval.role_system_weight,
                multi_fact_source_diversity=(
                    config.retrieval.multi_fact_source_diversity
                ),
                source_tfisf_activation=(
                    config.retrieval.source_tfisf_activation
                ),
                source_tfisf_slots=config.retrieval.source_tfisf_slots,
                source_hsc_activation=config.retrieval.source_hsc_activation,
                source_hsc_slots=config.retrieval.source_hsc_slots,
                source_hsc_hops=config.retrieval.source_hsc_hops,
                source_hsc_chunk_slots=(
                    config.retrieval.source_hsc_chunk_slots
                ),
                source_partition_routing=(
                    config.retrieval.source_partition_routing
                ),
                source_partition_slots=config.retrieval.source_partition_slots,
                source_partition_separator=(
                    config.retrieval.source_partition_separator
                ),
                source_local_search=config.retrieval.source_local_search,
                use_source_reranker=config.retrieval.qwen_rerank,
                use_attention_feedback=config.retrieval.qwen_feedback,
                feedback_slots=config.retrieval.qwen_feedback_slots,
                feedback_seed_slots=config.retrieval.qwen_feedback_seed_slots,
                feedback_evidence_tokens=(
                    config.retrieval.qwen_feedback_evidence_tokens
                ),
                feedback_query_tokens=config.retrieval.qwen_feedback_query_tokens,
                ef_search=config.retrieval.ef_search,
                candidates=config.retrieval.candidates,
                alpha=config.retrieval.alpha,
            )
            if config.retrieval.mode == "causal_graph"
            else None
        )
        packed = mc.build_context(
            question,
            recent_turns=0,
            k_memories=0 if causal else config.retrieval.k_memories,
            k_expansions=(0 if graph_results is not None else config.retrieval.k),
            # Hybrid is the production facade's default expansion retriever
            # and B0's strongest in-regime arm.  Memory mode should not
            # silently override it back to dense.
            hybrid=True,
            reheat_memories=False,
            use_consolidation=causal,
            learn_consolidation=False,
            consolidation_memory_slots=0 if causal else 1,
            consolidation_chunk_slots=(
                config.retrieval.consolidation_chunk_slots if causal else 1
            ),
            consolidation_min_count=config.retrieval.consolidation_min_count,
            consolidation_hops=config.retrieval.consolidation_hops,
            consolidation_candidates=config.retrieval.consolidation_candidates,
            consolidation_diffusion_width=(
                config.retrieval.consolidation_diffusion_width
            ),
            expansion_results=graph_results,
        )
        if graph_results is not None:
            mc.last_raw_graph_source_ids = list(
                dict.fromkeys(
                    str(
                        result.memory_source_id
                        or (
                            result.turn.source_id
                            if result.turn is not None
                            else None
                        )
                        or result.chunk.turn_id
                    )
                    for result in graph_results
                    if not is_source_metadata_text(result.chunk.text)
                )
            )
        header = [packed.memory_header] if packed.memory_header else []
        sources: list[str | None] = []
        if causal:
            for chunk_id in packed.expansion_chunk_ids:
                hydrated = mc.retriever.hydrate_chunk(
                    chunk_id,
                    score=0.0,
                    route="source_diagnostic",
                )
                sources.append(
                    getattr(getattr(hydrated, "turn", None), "source_id", None)
                )
        direct = set(packed.direct_expansion_chunk_ids)
        return (
            header,
            list(packed.expansions),
            sources,
            [chunk_id not in direct for chunk_id in packed.expansion_chunk_ids],
        )

    if config.retrieval.mode == "span":
        results = mc.search_spans(
            question,
            levels=config.retrieval.span_levels,
            k_per_level=config.retrieval.k_per_level,
        )
    elif config.retrieval.mode == "source":
        results = mc.search_sources(
            question,
            k_sources=config.retrieval.k_sources,
        )
    elif config.retrieval.mode == "anchored_source":
        results = mc.search_anchored_sources(
            question,
            k=config.retrieval.k,
            ef_search=config.retrieval.ef_search,
            candidates=config.retrieval.candidates,
            alpha=config.retrieval.alpha,
        )
    elif config.retrieval.mode == "hybrid_source":
        results = mc.search_hybrid_sources(
            question,
            k=config.retrieval.k,
            source_slots=config.retrieval.source_slots,
            source_candidate_pool=config.retrieval.source_candidate_pool,
            source_activation_k=config.retrieval.source_activation_k,
            query_facet_retrieval=config.retrieval.query_facet_retrieval,
            query_facet_slots=config.retrieval.query_facet_slots,
            query_facet_max=config.retrieval.query_facet_max,
            role_aware_retrieval=config.retrieval.role_aware_retrieval,
            role_user_weight=config.retrieval.role_user_weight,
            role_assistant_weight=config.retrieval.role_assistant_weight,
            role_system_weight=config.retrieval.role_system_weight,
            multi_fact_source_diversity=(
                config.retrieval.multi_fact_source_diversity
            ),
            source_partition_routing=config.retrieval.source_partition_routing,
            source_partition_slots=config.retrieval.source_partition_slots,
            source_partition_separator=(
                config.retrieval.source_partition_separator
            ),
            source_local_search=config.retrieval.source_local_search,
            use_source_reranker=config.retrieval.qwen_rerank,
            ef_search=config.retrieval.ef_search,
            candidates=config.retrieval.candidates,
            alpha=config.retrieval.alpha,
        )
    elif config.retrieval.mode == "hybrid_graph":
        results = mc.search_hybrid_graph(
            question,
            k=config.retrieval.k,
            neighbor_radius=config.retrieval.neighbor_radius,
            neighbor_slots=config.retrieval.neighbor_slots,
            neighbor_direction=config.retrieval.neighbor_direction,
            source_slots=config.retrieval.source_slots,
            source_candidate_pool=config.retrieval.source_candidate_pool,
            source_activation_k=config.retrieval.source_activation_k,
            source_tfisf_activation=config.retrieval.source_tfisf_activation,
            source_tfisf_slots=config.retrieval.source_tfisf_slots,
            source_hsc_activation=config.retrieval.source_hsc_activation,
            source_hsc_slots=config.retrieval.source_hsc_slots,
            source_hsc_hops=config.retrieval.source_hsc_hops,
            source_hsc_chunk_slots=config.retrieval.source_hsc_chunk_slots,
            source_local_search=config.retrieval.source_local_search,
            use_source_reranker=config.retrieval.qwen_rerank,
            use_attention_feedback=config.retrieval.qwen_feedback,
            feedback_slots=config.retrieval.qwen_feedback_slots,
            feedback_seed_slots=config.retrieval.qwen_feedback_seed_slots,
            feedback_evidence_tokens=(
                config.retrieval.qwen_feedback_evidence_tokens
            ),
            feedback_query_tokens=config.retrieval.qwen_feedback_query_tokens,
            ef_search=config.retrieval.ef_search,
            candidates=config.retrieval.candidates,
            alpha=config.retrieval.alpha,
        )
    elif config.retrieval.mode == "hybrid_neighbor":
        results = mc.search_hybrid_neighbors(
            question,
            k=config.retrieval.k,
            radius=config.retrieval.neighbor_radius,
            max_neighbors=config.retrieval.neighbor_slots,
            replacement_slots=config.retrieval.neighbor_replacement_slots,
            ef_search=config.retrieval.ef_search,
            candidates=config.retrieval.candidates,
            alpha=config.retrieval.alpha,
        )
    elif config.retrieval.effective_hybrid:
        results = mc.search_hybrid(
            question,
            k=config.retrieval.k,
            ef_search=config.retrieval.ef_search,
            candidates=config.retrieval.candidates,
            alpha=config.retrieval.alpha,
        )
    else:
        results = mc.search(
            question, k=config.retrieval.k, ef_search=config.retrieval.ef_search
        )
    return (
        [],
        [r.chunk.text for r in results],
        [getattr(getattr(r, "turn", None), "source_id", None) for r in results],
        [False] * len(results),
    )


def _survival(mc, gold: str, horizons_turns) -> dict[int, bool]:
    """Would the answer still sit in a non-COLD memory item N turns from now?

    Projects :func:`decay.effective_energy` forward over the stored items from
    the transcript's current position. Horizon 0 is not a projection at all —
    it is the store as it stands, which is now a real reading because turns
    have actually elapsed during the run.

    An empty memory store (the chunk arms, where nothing is extracted) yields
    ``False`` at every horizon — correctly: there is no memory item holding
    the answer.
    """
    items = mc.memory.list_items()
    now_turn = mc.transcript.current_turn()
    out: dict[int, bool] = {}
    for turns in horizons_turns:
        alive = [
            f"{i.content} {i.details or ''}"
            for i in items
            if decay.item_heat(i, now_turn=now_turn + turns) is not decay.Heat.COLD
        ]
        out[turns] = contains_answer(alive, gold)
    return out


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


def print_recall_report(report: RecallReport) -> None:
    """Human-readable summary. No API calls were made to produce any of this."""
    print()
    print("=" * 72)
    print("ANSWER REACHABILITY (offline — no API calls)")
    print(f"  benchmark: {report.benchmark or '(unnamed)'}")
    print(f"  mode     : {report.mode}  k={report.k}")
    print(f"  questions: {report.n_questions}")
    print("=" * 72)
    print(f"{'answer present anywhere in haystack':<34}{report.haystack_recall:>8.1%}")
    print(f"{'answer present in context':<34}{report.recall:>8.1%}")
    print(f"{'mean best token-F1':<34}{report.mean_best_f1:>8.3f}")
    if report.mode == "memory":
        print(f"{'  ...via the memory header':<34}{report.header_recall:>8.1%}")
        print(f"{'  ...via verbatim expansions':<34}{report.expansion_recall:>8.1%}")
    print()
    print(f"{'mean context tokens':<34}{report.mean_context_tokens:>8.0f}")
    print(f"{'recall per 1k tokens':<34}{report.recall_per_1k_tokens:>8.2f}")
    print("  (condensation wins by costing less, not only by recalling more —")
    print("   compare arms on this row as well as the one above)")

    if report.evidence_source_recall is not None:
        print()
        print(f"{'mean evidence-source coverage':<34}{report.evidence_source_recall:>8.1%}")
        print(f"{'questions with any evidence':<34}{report.evidence_any_source_recall:>8.1%}")
        print(f"{'questions with all evidence':<34}{report.evidence_all_source_recall:>8.1%}")
        if report.raw_evidence_source_recall is not None:
            print(
                f"{'raw graph source coverage':<34}"
                f"{report.raw_evidence_source_recall:>8.1%}"
            )
            print(
                f"{'raw graph with all evidence':<34}"
                f"{report.raw_evidence_all_source_recall:>8.1%}"
            )

    if report.answer_value_component_recall is not None:
        print()
        print(
            f"{'packed answer-value coverage':<34}"
            f"{report.answer_value_component_recall:>8.1%}"
        )
        print(
            f"{'questions with all answer values':<34}"
            f"{report.answer_value_all_component_recall:>8.1%}"
        )
        print(
            f"{'answer-value questions scored':<34}"
            f"{report.answer_value_scored_questions:>8}"
        )

    if report.survival_by_horizon:
        print()
        print("Answer still held by a non-COLD memory item, by turns ahead:")
        for turns, frac in sorted(report.survival_by_horizon.items()):
            label = "now" if turns == 0 else f"+{turns}t"
            print(f"  {label:>5} {frac:>7.1%}")

    selector_calls = [
        question
        for question in report.questions
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
    if report.coverage_selector_calls:
        print()
        print(
            f"{'coverage-selector calls':<34}"
            f"{report.coverage_selector_calls:>8}"
        )
        if selector_calls:
            print(
                f"{'mean selector latency (s)':<34}"
                f"{sum(q.coverage_selector_elapsed_s for q in selector_calls) / len(selector_calls):>8.2f}"
            )
        print(
            f"{'selector bypasses':<34}"
            f"{report.coverage_selector_bypasses:>8}"
        )
        print(
            f"{'selector fallbacks':<34}"
            f"{report.coverage_selector_fallbacks:>8}"
        )
        print(
            f"{'score-provider fallbacks':<34}"
            f"{report.coverage_score_provider_fallbacks:>8}"
        )
        print(
            f"{'degraded/fallback calls':<34}"
            f"{report.coverage_degraded_calls:>8}"
        )
        if report.coverage_routed_frontier_audited_calls:
            print(
                f"{'routed frontier exhaustive':<34}"
                f"{report.coverage_routed_frontier_exhaustive_calls:>4}/"
                f"{report.coverage_routed_frontier_audited_calls:<3}"
            )
        if report.coverage_active_partition_audited_calls:
            print(
                f"{'active partition exhaustive':<34}"
                f"{report.coverage_active_partition_exhaustive_calls:>4}/"
                f"{report.coverage_active_partition_audited_calls:<3}"
            )
        print(
            f"{'selected scope structurally complete':<34}"
            f"{report.coverage_selected_scope_structurally_complete_calls:>8}"
        )
        print(
            f"{'global semantic completeness':<34}"
            f"{report.coverage_global_semantic_complete_calls:>8}"
        )
        print(
            f"{'post-coverage closures':<34}"
            f"{report.coverage_closure_calls:>8}"
        )
        if report.coverage_closure_calls:
            print(
                f"{'  selected-scope policy':<34}"
                f"{report.coverage_selected_scope_policy_closure_calls:>8}"
            )
            print(
                f"{'  globally recall-guaranteed':<34}"
                f"{report.coverage_global_recall_guaranteed_closure_calls:>8}"
            )
        if report.coverage_cardinality_deficit_calls:
            print(
                f"{'cardinality deficit calls':<34}"
                f"{report.coverage_cardinality_deficit_calls:>8}"
            )
            print(
                f"{'cardinality deficit total':<34}"
                f"{report.coverage_cardinality_deficit_total:>8}"
            )

    if report.by_category:
        print()
        print(f"{'category':<40}{'recall':>8}")
        print("-" * 48)
        for category, frac in report.by_category.items():
            print(f"{category[:40]:<40}{frac:>8.1%}")
    print()
