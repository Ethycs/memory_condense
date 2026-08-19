"""Frozen identities, wire schema, and value types for recall merging."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

REPORT_FORMAT = "memory-condense-locked-v3-recall-campaign-v1"
FROZEN_OFFSETS = tuple(range(0, 100, 10))
# The frozen recall wire format includes a text-free JSON candidate trace.
# Real traces are larger than Python's platform-default 128 KiB CSV field
# limit, so parsing uses an explicit bounded ceiling rather than an ambient
# process setting.
MAX_CSV_FIELD_CHARS = 16 * 1024 * 1024


class RecallCampaignError(ValueError):
    """A source artifact or recall shard violates the locked-v3 contract."""


@dataclass(frozen=True, slots=True)
class FrozenV3Anchors:
    dataset_sha256: str
    split_manifest_sha256: str
    policy_manifest_sha256: str
    implementation_sha256: str
    environment_lock_sha256: str
    selection_artifact_sha256: str


FROZEN_V3_ANCHORS = FrozenV3Anchors(
    dataset_sha256="d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
    split_manifest_sha256="8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4",
    policy_manifest_sha256="5263d5afd15298ec4088db9d6381ae243ddb685e9a3cf4d9892fc84e14fb9883",
    implementation_sha256="452be3bfa7524bb81676c7abcb032529a32a480311d24d1e17f8513c783ecd83",
    environment_lock_sha256="058083871240979257ada7ca4c71dd816fee64792b275ef11e4857c9f5ebba33",
    selection_artifact_sha256="a82a3ffb2880121e3952f0e581c2affe199e48e2a3d0cdddf2fe09492b6e4a3e",
)


@dataclass(frozen=True, slots=True)
class RecallProtocol:
    split: str = "validation"
    sample_offsets: tuple[int, ...] = FROZEN_OFFSETS
    questions_per_shard: int = 10
    population_questions: int = 100
    stress_context_tokens: int = 1_000_000


FROZEN_V3_PROTOCOL = RecallProtocol()


@dataclass(frozen=True, slots=True)
class ExpectedRecallQuestion:
    question_id: str
    category: str
    evidence_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExpectedRecallShard:
    sample_offset: int
    questions: tuple[ExpectedRecallQuestion, ...]


@dataclass(frozen=True, slots=True)
class SourceRecheck:
    label: str
    path: Path
    sha256: str
    kind: str = "file"


@dataclass(frozen=True, slots=True)
class LockedV3RecallPlan:
    dataset_sha256: str
    split_manifest_sha256: str
    policy_manifest_sha256: str
    implementation_sha256: str
    environment_lock_sha256: str
    selection_artifact_sha256: str
    retrieval_identity_sha256: str
    retrieval: Mapping[str, Any]
    protocol: RecallProtocol
    shards: tuple[ExpectedRecallShard, ...]
    source_rechecks: tuple[SourceRecheck, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class RecallCsvShard:
    sample_offset: int
    name: str
    payload: bytes


CSV_SCHEMA = (
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
)


_REQUIRED_BINARY_FIELDS = {
    "in_haystack",
    "in_context",
    "in_header",
    "in_expansions",
    "coverage_selector_frontier_exhaustive",
    "coverage_selector_allow_selected_scope_fixed_k_closure",
    "closure_applied",
}
_OPTIONAL_BINARY_FIELDS = {
    "all_evidence_sources",
    "raw_all_evidence_sources",
    "all_answer_value_components",
    "coverage_selector_routed_frontier_exhaustive",
    "coverage_selector_active_partition_exhaustive",
    "coverage_selector_active_partition_semantically_complete",
    "coverage_selector_partition_scope_exhaustive",
    "coverage_selector_selected_scope_structurally_complete",
    "coverage_selector_global_semantic_complete",
    "closure_global_recall_guaranteed",
}
_REQUIRED_INT_FIELDS = {
    "context_tokens",
    "source_companion_direct_date_retained",
    "source_companion_candidates_before",
    "source_companion_candidates_after",
    "direct_chunks",
    "consolidation_chunks",
    "causal_events",
    "causal_graph_edges",
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
    "coverage_selector_frontier_candidates",
    "coverage_selector_frontier_attempted",
    "coverage_selector_frontier_uninspected",
    "coverage_selector_frontier_batches",
    "coverage_selector_active_partition_structural_rows",
    "coverage_selector_active_partition_structural_hypotheses",
    "coverage_selector_active_partition_candidates_admitted",
    "coverage_selector_active_partition_candidates_already_present",
    "coverage_selector_active_partition_candidates_replaced",
    "coverage_selector_active_partition_candidates_truncated",
    "coverage_selector_active_partition_structural_overflow",
    "coverage_selector_cardinality_deficit",
    "coverage_selector_credible_clusters",
    "coverage_selector_reserved_representatives",
    "coverage_selector_structural_eligible_clusters",
    "coverage_selector_structural_reserved_representatives",
    "coverage_selector_score_provider_forward_passes",
    "coverage_selector_score_provider_peak_workspace_tokens",
    "coverage_selector_score_provider_total_workspace_tokens",
    "coverage_selector_score_provider_retained_state_bytes",
    "coverage_selector_prefix_layers",
    "coverage_selector_prefix_attention_layer",
    "coverage_selector_semantic_inspected",
    "coverage_selector_semantic_workspace_tokens",
    "coverage_selector_retained_state_bytes",
}
_OPTIONAL_INT_FIELDS = {
    "answer_value_components_expected",
    "answer_value_components_found",
    "coverage_selector_cardinality",
    "coverage_selector_temporal_window_days",
    "coverage_selector_active_partition_total",
    "coverage_selector_active_partition_inspected",
    "coverage_selector_active_partition_sources_total",
    "coverage_selector_partition_inventory_total",
    "coverage_selector_selected_partition_count",
}
_REQUIRED_FIXED_FLOAT_FIELDS = {
    "best_f1",
    "causal_write_s",
    "coverage_selector_elapsed_s",
    "coverage_selector_score_provider_elapsed_s",
    "coverage_selector_semantic_elapsed_s",
}
_OPTIONAL_FIXED_FLOAT_FIELDS = {
    "evidence_source_recall",
    "raw_evidence_source_recall",
    "answer_value_component_recall",
}
_JSON_FIELDS = {"partition_ranking", "coverage_candidate_trace"}
_PIPE_FIELDS = {
    "retrieved_source_ids",
    "raw_retrieved_source_ids",
    "answer_value_component_hit_mask",
    "source_companion_requested",
    "source_companion_hydrated",
    "source_companion_orphans",
    "selected_partitions",
}
_TEXT_FIELDS = set(CSV_SCHEMA) - (
    _REQUIRED_BINARY_FIELDS
    | _OPTIONAL_BINARY_FIELDS
    | _REQUIRED_INT_FIELDS
    | _OPTIONAL_INT_FIELDS
    | _REQUIRED_FIXED_FLOAT_FIELDS
    | _OPTIONAL_FIXED_FLOAT_FIELDS
    | _JSON_FIELDS
    | _PIPE_FIELDS
)

if len(CSV_SCHEMA) != len(set(CSV_SCHEMA)):
    raise RuntimeError("locked recall CSV schema contains duplicate columns")
if (
    _TEXT_FIELDS
    | _REQUIRED_BINARY_FIELDS
    | _OPTIONAL_BINARY_FIELDS
    | _REQUIRED_INT_FIELDS
    | _OPTIONAL_INT_FIELDS
    | _REQUIRED_FIXED_FLOAT_FIELDS
    | _OPTIONAL_FIXED_FLOAT_FIELDS
    | _JSON_FIELDS
    | _PIPE_FIELDS
) != set(CSV_SCHEMA):
    raise RuntimeError("locked recall CSV field classification is incomplete")


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NONNEGATIVE_INT_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")
_FIXED_FLOAT_RE = re.compile(r"^(?:0|[1-9][0-9]*)\.[0-9]{4}$")
