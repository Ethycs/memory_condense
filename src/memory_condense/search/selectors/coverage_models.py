"""Shared contracts and text-free coverage-selection reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from pydantic import BaseModel, Field

from memory_condense.domain.schemas import RetrievalResult

class _RawAssignment(BaseModel):
    """One validated row emitted by the injected set classifier."""

    id: int = Field(ge=0)
    event_key: str | None = None
    answer_value: str | None = ""
    timestamp: str | None = None
    p_existing: float = Field(default=0.0, ge=0.0, le=1.0)
    p_new: float = Field(default=0.0, ge=0.0, le=1.0)
    p_null: float = Field(default=0.0, ge=0.0, le=1.0)
    answerability: float = Field(default=0.5, ge=0.0, le=1.0)


@dataclass(frozen=True, slots=True)
class CandidateAssignment:
    """Normalized existing/new/null assignment used by the selector."""

    candidate_id: int
    event_key: str | None
    answer_value: str
    timestamp: str | None
    p_existing: float
    p_new: float
    p_null: float
    answerability: float
    entropy: float

    @property
    def member_probability(self) -> float:
        return self.p_existing + self.p_new


@dataclass(frozen=True, slots=True)
class CoverageSelectionReport:
    """Text-free diagnostics for one transient set-selection pass."""

    operator: str
    cardinality: int | None
    requires_completeness: bool
    input_candidates: int
    inspected_candidates: int
    classified_candidates: int
    event_clusters: int
    new_assignments: int
    existing_assignments: int
    null_assignments: int
    uncertain_assignments: int
    output_candidates: int
    representatives: int
    supporting_candidates: int
    workspace_tokens: int
    elapsed_s: float
    # ``bypassed`` is an intentional query-dependent no-op, not a degraded
    # selector call.  In particular, singleton questions do not require the
    # complete-set coverage operator at all.  ``fallback`` is reserved for an
    # applicable pass that failed open.
    selection_status: str = "applied"
    bypass_reason: str = ""
    retained_transformer_state_bytes: int = 0
    fallback_reason: str = ""
    quantifier: str = ""
    ordering: str = ""
    posterior_kind: str = ""
    semantic_score_kind: str = ""
    frontier_candidates: int = 0
    frontier_attempted: int = 0
    frontier_uninspected: int = 0
    # ``routed_frontier_exhaustive`` means every row received from upstream
    # routing was inspected.  It must not be confused with complete coverage
    # of the active durable partition, whose size is optional metadata below.
    routed_frontier_exhaustive: bool | None = False
    frontier_exhaustive: bool = False
    frontier_batches: int = 0
    active_partition_total: int | None = None
    active_partition_inspected: int | None = None
    active_partition_exhaustive: bool | None = None
    # A cheap typed scan may inspect every durable row without running every
    # row through Qwen.  Keep its physical coverage and semantic conclusion
    # separate from the bounded model-frontier counters above.
    active_partition_sources_total: int | None = None
    active_partition_structural_rows: int = 0
    active_partition_structural_hypotheses: int = 0
    active_partition_candidates_admitted: int = 0
    active_partition_candidates_already_present: int = 0
    active_partition_candidates_replaced: int = 0
    active_partition_candidates_truncated: int = 0
    active_partition_structural_overflow: int = 0
    active_partition_scan_contract: str = ""
    active_partition_semantically_complete: bool | None = None
    # Partition selection and partition scanning prove different scopes.  A
    # structurally complete scan of four approximately selected partitions is
    # not evidence that a fifth relevant partition does not exist.
    partition_scope_kind: str = "approximate_top_k"
    partition_inventory_total: int | None = None
    selected_partition_count: int | None = None
    partition_scope_exhaustive: bool | None = None
    selected_scope_structurally_complete: bool | None = None
    global_semantic_complete: bool | None = None
    allow_selected_scope_fixed_k_closure: bool = False
    credible_clusters: int = 0
    reserved_representatives: int = 0
    structural_eligible_clusters: int = 0
    structural_reserved_representatives: int = 0
    cardinality_deficit: int = 0
    answerability_score_kind: str = ""
    score_provider_fallback: str = ""
    score_provider_report: Mapping[
        str,
        str | int | float | bool | None,
    ] | None = None
    prefix_model_id: str = ""
    prefix_model_revision: str = ""
    prefix_checkpoint_sha256: str = ""
    prefix_device: str = ""
    prefix_dtype: str = ""
    prefix_layers: int = 0
    prefix_attention_layer: int = -1
    required_evidence_role: str | None = None
    required_evidence_role_basis: str | None = None
    query_timestamp: str | None = None
    temporal_window_days: int | None = None

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


CompletionFn = Callable[[list[dict[str, str]]], Any]


class CoverageScoreProvider(Protocol):
    """Optional non-generative scorer that can feed the prefix posterior.

    Implementations may keep model weights loaded, but each call must return
    only text-free scalar evidence and must not retain KV caches/activations.
    """

    def score_candidates(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
    ) -> Mapping[str, Any]: ...

    def select_source_companions(
        self,
        query: str,
        candidates_by_source: Mapping[str, Sequence[RetrievalResult]],
    ) -> Mapping[str, RetrievalResult]: ...
