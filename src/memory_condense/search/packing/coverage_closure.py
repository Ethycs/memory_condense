"""Fail-closed proof for destructive post-coverage tail closure."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from memory_condense.domain.schemas import RetrievalResult
from memory_condense.search.packing.source_provenance import (
    provenance_timestamp_key as _provenance_timestamp_key,
)


# Post-coverage closure is intentionally narrower than ordinary coverage
# reservation. These bases are emitted only for deterministic typed frontiers.
_POST_COVERAGE_SCAN_CONTRACT_BASES = {
    "canonical_venue_episode_aligned_v1": "canonical_fixed_frontier",
    "direct_performance_source_occurrence_v1": "direct_performance_frontier",
}


def _report_value(report: Any, field: str) -> Any:
    """Read one diagnostic field without coupling to a report class."""

    if isinstance(report, Mapping):
        return report.get(field)
    return getattr(report, field, None)


def _exact_report_int(report: Any, field: str) -> int | None:
    """Return a report integer, rejecting bools and coercion-friendly values."""

    value = _report_value(report, field)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


class _CoverageClosureMixin:
    def _post_coverage_closure_ids(
        self,
        *,
        selector_report: Any,
        selector_report_is_current: bool,
        selector_output_rejected: bool,
        selector_input: Sequence[RetrievalResult],
        returned_ids: set[str],
        selector_trace_rows: Sequence[Mapping[str, Any]],
        requested_reserved_rows: Sequence[RetrievalResult],
        active_reserved_ids: set[str],
        reservation_bodies: Mapping[str, str],
        reservation_snippets: Mapping[str, str],
        source_timestamps: Mapping[str, str],
    ) -> tuple[tuple[str, ...], str, bool] | None:
        """Prove when a typed FIXED frontier can safely close the prompt tail.

        Coverage ranking normally fails open: unselected and uncertain rows
        remain available after the reserved representatives.  Closure is the
        narrow exception for a fully inspected, structurally typed, ordered
        FIXED-K result whose exact raw bodies have all been preflighted.  Any
        absent, stale, malformed, contradictory, truncated, or rejected
        diagnostic returns ``None`` and preserves the ordinary fail-open path.
        A proof returns ``(closed chunk_ids, closure scope,
        global_recall_guaranteed)``.
        """

        if (
            self.expansion_selector is None
            or not selector_report_is_current
            or selector_report is None
            or selector_output_rejected
        ):
            return None
        if (
            _report_value(selector_report, "selection_status") != "applied"
            or _report_value(selector_report, "fallback_reason") not in (None, "")
            or _report_value(selector_report, "bypass_reason") not in (None, "")
            or _report_value(selector_report, "score_provider_fallback")
            not in (None, "")
            or _report_value(selector_report, "operator") != "fixed_cardinality"
            or _report_value(selector_report, "quantifier") != "fixed_cardinality"
            or _report_value(selector_report, "requires_completeness") is not True
        ):
            return None

        cardinality = _exact_report_int(selector_report, "cardinality")
        if cardinality is None or cardinality < 1:
            return None
        active_partition_exhaustive = _report_value(
            selector_report,
            "active_partition_exhaustive",
        )
        selected_scope_complete = _report_value(
            selector_report,
            "selected_scope_structurally_complete",
        )
        legacy_selected_scope_complete = _report_value(
            selector_report,
            "active_partition_semantically_complete",
        )
        if (
            selected_scope_complete is not True
            or legacy_selected_scope_complete is not True
        ):
            return None
        scope_kind = _report_value(selector_report, "partition_scope_kind")
        if scope_kind not in {
            "approximate_top_k",
            "global",
            "authoritative",
        }:
            return None
        partition_scope_exhaustive = _report_value(
            selector_report,
            "partition_scope_exhaustive",
        )
        if (
            partition_scope_exhaustive is not None
            and not isinstance(partition_scope_exhaustive, bool)
        ):
            return None
        inventory_total_value = _report_value(
            selector_report,
            "partition_inventory_total",
        )
        selected_partition_value = _report_value(
            selector_report,
            "selected_partition_count",
        )
        inventory_total = (
            None
            if inventory_total_value is None
            else _exact_report_int(selector_report, "partition_inventory_total")
        )
        selected_partition_count = (
            None
            if selected_partition_value is None
            else _exact_report_int(selector_report, "selected_partition_count")
        )
        if (
            (inventory_total_value is not None and inventory_total is None)
            or (
                selected_partition_value is not None
                and selected_partition_count is None
            )
            or (inventory_total is not None and inventory_total < 0)
            or (
                selected_partition_count is not None
                and selected_partition_count < 0
            )
            or (
                inventory_total is not None
                and selected_partition_count is not None
                and selected_partition_count > inventory_total
            )
            or inventory_total is None
            or selected_partition_count is None
            or inventory_total < 1
            or selected_partition_count < 1
            or partition_scope_exhaustive is None
        ):
            return None
        if inventory_total is not None and selected_partition_count is not None:
            if partition_scope_exhaustive is not (
                inventory_total == selected_partition_count
            ):
                return None
        global_semantic_complete = _report_value(
            selector_report,
            "global_semantic_complete",
        )
        if (
            global_semantic_complete is not None
            and not isinstance(global_semantic_complete, bool)
        ):
            return None
        global_proof = bool(
            global_semantic_complete is True
            and scope_kind in {"global", "authoritative"}
            and (
                scope_kind == "authoritative"
                or partition_scope_exhaustive is True
            )
        )
        if (
            (scope_kind == "global" and partition_scope_exhaustive is not True)
            or (global_semantic_complete is True and not global_proof)
        ):
            return None
        selected_scope_policy = bool(
            not global_proof
            and scope_kind == "approximate_top_k"
            and partition_scope_exhaustive is False
            and global_semantic_complete is False
            and _report_value(
                selector_report,
                "allow_selected_scope_fixed_k_closure",
            )
            is True
            and getattr(
                self.expansion_selector,
                "allow_selected_scope_fixed_k_closure",
                False,
            )
            is True
        )
        if not global_proof and not selected_scope_policy:
            return None
        scan_contract = _report_value(
            selector_report,
            "active_partition_scan_contract",
        )
        required_reservation_basis = _POST_COVERAGE_SCAN_CONTRACT_BASES.get(
            scan_contract
        )
        structural_hypotheses = _exact_report_int(
            selector_report,
            "active_partition_structural_hypotheses",
        )
        structural_rows = _exact_report_int(
            selector_report,
            "active_partition_structural_rows",
        )
        active_sources = _exact_report_int(
            selector_report,
            "active_partition_sources_total",
        )
        if (
            required_reservation_basis is None
            or structural_hypotheses != cardinality
            or structural_rows is None
            or structural_rows < cardinality
            or active_sources is None
            or active_sources < 1
        ):
            return None
        if (
            _report_value(selector_report, "routed_frontier_exhaustive") is not True
            or _exact_report_int(selector_report, "frontier_uninspected") != 0
            # Closing the tail is destructive.  Scoring every row that happened
            # to reach the bounded route union is not proof that the active
            # durable partition was searched.  Require the typed structural
            # scan to state physical and semantic completeness explicitly.
            or active_partition_exhaustive is not True
            or _exact_report_int(selector_report, "cardinality_deficit") != 0
            or _exact_report_int(
                selector_report,
                "structural_eligible_clusters",
            )
            != cardinality
            or _exact_report_int(
                selector_report,
                "structural_reserved_representatives",
            )
            != cardinality
            or _exact_report_int(selector_report, "reserved_representatives")
            != cardinality
        ):
            return None

        for field in (
            "active_partition_candidates_truncated",
            "active_partition_structural_overflow",
        ):
            if _exact_report_int(selector_report, field) != 0:
                return None

        input_ids = [result.chunk.chunk_id for result in selector_input]
        if len(input_ids) != len(set(input_ids)):
            return None
        input_count = len(input_ids)
        for field in (
            "input_candidates",
            "inspected_candidates",
            "classified_candidates",
            "frontier_candidates",
            "frontier_attempted",
        ):
            if _exact_report_int(selector_report, field) != input_count:
                return None
        if _exact_report_int(selector_report, "output_candidates") != len(
            returned_ids
        ):
            return None

        active_total = _report_value(selector_report, "active_partition_total")
        active_inspected = _report_value(
            selector_report,
            "active_partition_inspected",
        )
        if active_total is not None or active_inspected is not None:
            if (
                isinstance(active_total, bool)
                or not isinstance(active_total, int)
                or isinstance(active_inspected, bool)
                or not isinstance(active_inspected, int)
                or active_total < 0
                or active_inspected != active_total
            ):
                return None

        trace_ids = [
            row.get("chunk_id")
            for row in selector_trace_rows
            if isinstance(row.get("chunk_id"), str)
        ]
        if (
            len(trace_ids) != len(selector_trace_rows)
            or len(trace_ids) != len(set(trace_ids))
            or set(trace_ids) != set(input_ids)
        ):
            return None
        structural_rows = [
            row
            for row in selector_trace_rows
            if row.get("coverage_reserved") is True
            and row.get("group_role") == "representative"
        ]
        if len(structural_rows) != cardinality:
            return None
        structural_ids = tuple(str(row["chunk_id"]) for row in structural_rows)
        if (
            len(set(structural_ids)) != cardinality
            or not set(structural_ids).issubset(returned_ids)
            or any(
                not isinstance(row.get("reservation_basis"), str)
                or row.get("reservation_basis")
                != required_reservation_basis
                or row.get("group_id") is None
                or row.get("role_match") is False
                or (
                    row.get("temporal_in_scope") is not None
                    and row.get("temporal_in_scope") is not True
                )
                for row in structural_rows
            )
            or len({str(row["group_id"]) for row in structural_rows})
            != cardinality
        ):
            return None

        requested_ids = tuple(
            result.chunk.chunk_id for result in requested_reserved_rows
        )
        if set(requested_ids) != set(structural_ids):
            return None
        if set(requested_ids) != active_reserved_ids:
            return None
        if any(
            not reservation_bodies.get(chunk_id)
            or reservation_snippets.get(chunk_id)
            != reservation_bodies.get(chunk_id)
            for chunk_id in requested_ids
        ):
            return None

        ordering = _report_value(selector_report, "ordering")
        if ordering not in ("ascending", "descending"):
            return None
        timestamps = [
            _provenance_timestamp_key(
                source_timestamps.get(self._result_source_id(result))
            )
            for result in requested_reserved_rows
        ]
        if any(timestamp is None for timestamp in timestamps):
            return None
        ordered_values = [float(timestamp) for timestamp in timestamps]
        if ordering == "ascending":
            ordered = all(
                left < right
                for left, right in zip(
                    ordered_values,
                    ordered_values[1:],
                    strict=False,
                )
            )
        else:
            ordered = all(
                left > right
                for left, right in zip(
                    ordered_values,
                    ordered_values[1:],
                    strict=False,
                )
            )
        if not ordered:
            return None
        return (
            requested_ids,
            "global_semantic" if global_proof else "selected_scope_policy",
            global_proof,
        )
