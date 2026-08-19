"""Orchestration for the bounded, transient prefix coverage stages."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from memory_condense.domain.schemas import RetrievalResult
from memory_condense.search.selectors.prefix_admission import (
    prepare_prefix_coverage,
)
from memory_condense.search.selectors.prefix_models import (
    _PreparedCoverage,
    _ScoredCoverage,
)
from memory_condense.search.selectors.prefix_reservation import (
    reserve_prefix_coverage,
)
from memory_condense.search.selectors.prefix_scoring import score_prefix_coverage


def select_prefix_coverage(
    selector: Any,
    query: str,
    candidates: Sequence[RetrievalResult],
    *,
    max_results: int | None = None,
    source_timestamps: Mapping[str, str] | None = None,
    semantic_scores: Mapping[str, float | None] | None = None,
    answerability_scores: Mapping[str, Any] | None = None,
    membership_scores: Mapping[str, Any] | None = None,
    active_partition_total: int | None = None,
    active_partition_inspected: int | None = None,
    active_partition_scan: Mapping[str, Any] | None = None,
) -> list[RetrievalResult]:
    """Validate, score, and reserve without retaining transformer state."""

    prepared = prepare_prefix_coverage(
        selector,
        query,
        candidates,
        max_results=max_results,
        source_timestamps=source_timestamps,
        semantic_scores=semantic_scores,
        answerability_scores=answerability_scores,
        membership_scores=membership_scores,
        active_partition_total=active_partition_total,
        active_partition_inspected=active_partition_inspected,
        active_partition_scan=active_partition_scan,
    )
    if not isinstance(prepared, _PreparedCoverage):
        return prepared

    scored = score_prefix_coverage(selector, prepared)
    if not isinstance(scored, _ScoredCoverage):
        return scored
    return reserve_prefix_coverage(selector, scored)
