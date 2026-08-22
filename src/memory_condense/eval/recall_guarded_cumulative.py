"""Recall-guarded composition of frozen-v3 and diffuse retrieval.

The immutable receipts, validated result model, and retrieval operations live
in cohesive private modules.  This facade preserves the original import
surface for callers and archived experiment code.
"""

from memory_condense.eval._recall_guarded_cumulative_contracts import (
    CAUSAL_COVERAGE_PREDECESSOR_FORMAT,
    CUMULATIVE_LADDER_FORMAT,
    CUMULATIVE_STAGE_FORMAT,
    NOVEL_CLOSURE_PROJECTION_FORMAT,
    RECALL_GUARDED_CUMULATIVE_FORMAT,
    _CUMULATIVE_STAGE_IDS,
    _atom_evidence_id,
    _expected_predecessor_budget,
    _freeze_messages,
    _nonempty,
    _numbered_context,
    _ordered_unique,
    _protected_evidence_id,
    _unique_ids,
    causal_graph_context_budget,
    CausalCoveragePredecessor,
    CausalCoveragePredecessorReceipt,
    CumulativeRetrievalLadder,
    CumulativeRetrievalStageReceipt,
    NovelClosureProjection,
    NovelClosureProjectionReceipt,
    ProtectedExcerpt,
    RecallGuardedCumulativeReceipt,
)
from memory_condense.eval._recall_guarded_cumulative_ops import (
    _close_cumulative_method_plan,
    _combine_episode_seeds,
    _episode_seed_payload,
    _pack_additions,
    _require_causal_coverage,
    _validate_coverage_runtime_binding,
    _widen_direct_episode_policy,
    measure_recall_guarded_cumulative_packet,
    retrieve_causal_coverage_predecessor,
    retrieve_recall_guarded_cumulative_packet,
)
from memory_condense.eval._recall_guarded_cumulative_result import (
    _addition_prompt_prefix,
    _novel_closure_projection,
    _stage_evidence_projection_sha256,
    RecallGuardedCumulativeMetrics,
    RecallGuardedCumulativeRetrieval,
    RecallGuardedCumulativeStageMetrics,
)


__all__ = [
    "CAUSAL_COVERAGE_PREDECESSOR_FORMAT",
    "CUMULATIVE_LADDER_FORMAT",
    "CUMULATIVE_STAGE_FORMAT",
    "NOVEL_CLOSURE_PROJECTION_FORMAT",
    "RECALL_GUARDED_CUMULATIVE_FORMAT",
    "CausalCoveragePredecessor",
    "CausalCoveragePredecessorReceipt",
    "causal_graph_context_budget",
    "CumulativeRetrievalLadder",
    "CumulativeRetrievalStageReceipt",
    "NovelClosureProjection",
    "NovelClosureProjectionReceipt",
    "ProtectedExcerpt",
    "RecallGuardedCumulativeMetrics",
    "RecallGuardedCumulativeStageMetrics",
    "RecallGuardedCumulativeReceipt",
    "RecallGuardedCumulativeRetrieval",
    "measure_recall_guarded_cumulative_packet",
    "retrieve_causal_coverage_predecessor",
    "retrieve_recall_guarded_cumulative_packet",
]
