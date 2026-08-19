"""Transient data structures passed through the prefix coverage stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from memory_condense.domain.schemas import RetrievalResult

@dataclass(slots=True)
class _PrefixEventCluster:
    """Transient query-conditioned event group; never leaves one call."""

    prototype: np.ndarray
    vectors: list[np.ndarray]
    members: list["_PrefixAssignment"]
    source_ids: set[str]
    timestamps: set[str]
    answer_object_keys: set[str]


@dataclass(slots=True)
class _PrefixAssignment:
    """One transient uncalibrated existing/new/null decision."""

    index: int
    result: RetrievalResult
    quality: float
    value_evidence: float
    membership_score: float | None
    vector: np.ndarray
    p_existing: float
    p_new: float
    p_null: float
    existing_energy: float | None
    new_energy: float
    null_energy: float
    temporal_in_scope: bool | None
    entropy: float
    semantic_surprisal: float
    hypothesis: str
    existing_cluster: int | None
    merge_similarity: float | None
    merge_threshold: float | None


@dataclass(slots=True)
class _PreparedCoverage:
    """Validated request and frontier metadata for one transient selector call."""

    started: float
    query: str
    program: Any
    unique: list[RetrievalResult]
    max_results: int | None
    timestamps: Mapping[str, str]
    semantic_scores: Mapping[str, float | None] | None
    active_partition_total: int | None
    active_partition_inspected: int | None
    active_partition_exhaustive: bool | None
    normalized_scan_fields: dict[str, Any]
    score_provider_fallback: str
    score_provider_report: Mapping[str, str | int | float | bool | None] | None
    typed_performance_frontier: bool
    performance_event_keys_by_id: dict[str, str]
    performance_primary_ids: set[str]
    effective_answerability: Mapping[str, Any] | None
    effective_membership: Mapping[str, Any] | None


@dataclass(slots=True)
class _ScoredCoverage:
    """Prefix scores, event assignments, and counters awaiting reservation."""

    prepared: _PreparedCoverage
    attempted_candidates: int
    inspected_candidates: int
    frontier_batches: int
    max_workspace_tokens: int
    hits: dict[str, Any]
    semantic_kind_by_id: dict[str, str]
    answerability_by_id: dict[str, float | None]
    membership_by_id: dict[str, float | None]
    score_by_id: dict[str, float]
    value_by_id: dict[str, float]
    semantic_raw_by_id: dict[str, float]
    canonical_answer_object_keys_by_id: dict[str, str | None]
    answer_object_keys_by_id: dict[str, str | None]
    clusters: list[_PrefixEventCluster]
    uncertain: list[tuple[int, RetrievalResult]]
    posterior_uncertain_rows: list[_PrefixAssignment]
    null_rows: list[_PrefixAssignment]
    existing_count: int
    new_count: int
