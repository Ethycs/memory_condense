"""Shared contracts and text-free coverage-selection reports."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from memory_condense.domain.schemas import RetrievalResult
from memory_condense.search.selectors.evidence_features import (
    _normalized_event_key,
)
from memory_condense.search.selectors.set_program import SetProgram


class ReportDumpMixin:
    """Uniform ``model_dump`` seam for frozen report dataclasses."""

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class CandidateAssignment(BaseModel):
    """One validated classifier row, normalized into selector evidence.

    Field constraints validate the raw injected-classifier values; the
    posterior triple is then renormalized to sum to one on construction.
    A zero-mass row is a classifier fault and raises ``ValueError``.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    candidate_id: int = Field(ge=0, validation_alias="id")
    event_key: str | None = None
    answer_value: str | None = ""
    timestamp: str | None = None
    p_existing: float = Field(default=0.0, ge=0.0, le=1.0)
    p_new: float = Field(default=0.0, ge=0.0, le=1.0)
    p_null: float = Field(default=0.0, ge=0.0, le=1.0)
    answerability: float = Field(default=0.5, ge=0.0, le=1.0)

    def model_post_init(self, __context: Any) -> None:
        values = [float(self.p_existing), float(self.p_new), float(self.p_null)]
        total = sum(values)
        if total <= 0.0:
            raise ValueError(
                f"candidate {self.candidate_id} has zero posterior mass"
            )
        existing, new, null = (value / total for value in values)
        # ``frozen`` guards outside mutation; normalization is part of
        # construction, so write through the instance dict directly.
        self.__dict__.update(
            event_key=_normalized_event_key(self.event_key),
            answer_value=(self.answer_value or "").strip(),
            timestamp=self.timestamp.strip() if self.timestamp else None,
            p_existing=existing,
            p_new=new,
            p_null=null,
            answerability=float(self.answerability),
        )

    @property
    def entropy(self) -> float:
        return -sum(
            probability * math.log(probability)
            for probability in (self.p_existing, self.p_new, self.p_null)
            if probability > 0.0
        )

    @property
    def member_probability(self) -> float:
        return self.p_existing + self.p_new


@dataclass(frozen=True, slots=True)
class CoverageSelectionReport(ReportDumpMixin):
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

    @classmethod
    def uninspected(
        cls,
        program: SetProgram,
        *,
        started: float,
        input_candidates: int,
        selection_status: str,
        **overrides: Any,
    ) -> "CoverageSelectionReport":
        """Report a pass that classified nothing, with standard zero counters.

        Every candidate is treated as its own uncertain cluster and returned
        unchanged.  Sites that deviate from the standard shape pass explicit
        ``overrides``.
        """

        values: dict[str, Any] = {
            "operator": program.operator.value,
            "cardinality": program.cardinality,
            "requires_completeness": program.requires_completeness,
            "input_candidates": input_candidates,
            "inspected_candidates": 0,
            "classified_candidates": 0,
            "event_clusters": 0,
            "new_assignments": 0,
            "existing_assignments": 0,
            "null_assignments": 0,
            "uncertain_assignments": input_candidates,
            "output_candidates": input_candidates,
            "representatives": 0,
            "supporting_candidates": 0,
            "workspace_tokens": 0,
            "elapsed_s": time.perf_counter() - started,
            "selection_status": selection_status,
        }
        values.update(overrides)
        return cls(**values)


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
