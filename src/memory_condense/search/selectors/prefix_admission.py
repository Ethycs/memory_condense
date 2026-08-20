"""Validation, structural admission, and provider setup for prefix coverage."""

from __future__ import annotations

import inspect
import time
from collections import defaultdict
from typing import Any, Mapping, Sequence

from memory_condense.domain.schemas import RetrievalResult
from memory_condense.search.packing.performance_events import (
    is_direct_past_performance,
    is_performance_query,
    performance_event_key,
)
from memory_condense.search.selectors.prefix_models import _PreparedCoverage
from memory_condense.search.selectors.set_program import (
    SetOrdering,
    SetQuantifier,
    compile_set_program,
)

def prepare_prefix_coverage(
    self,
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
) -> _PreparedCoverage | list[RetrievalResult]:
    started = time.perf_counter()
    program = compile_set_program(query)
    unique: list[RetrievalResult] = []
    seen: set[str] = set()
    for result in candidates:
        if result.chunk.chunk_id in seen:
            continue
        seen.add(result.chunk.chunk_id)
        unique.append(result)
    self.last_candidate_trace = self._uninspected_trace(unique, program)
    if max_results is not None and max_results < 1:
        raise ValueError("max_results must be positive when supplied")
    scan_fields = dict(active_partition_scan or {})
    scan_total = scan_fields.pop("active_partition_total", None)
    scan_inspected = scan_fields.pop("active_partition_inspected", None)
    scan_exhaustive = scan_fields.pop("active_partition_exhaustive", None)
    if active_partition_total is None:
        active_partition_total = scan_total
    elif scan_total is not None and scan_total != active_partition_total:
        raise ValueError("active partition total disagrees with scan report")
    if active_partition_inspected is None:
        active_partition_inspected = scan_inspected
    elif scan_inspected is not None and scan_inspected != active_partition_inspected:
        raise ValueError("active partition inspected disagrees with scan report")
    allowed_scan_fields = {
        "active_partition_sources_total",
        "active_partition_structural_rows",
        "active_partition_structural_hypotheses",
        "active_partition_candidates_admitted",
        "active_partition_candidates_already_present",
        "active_partition_candidates_replaced",
        "active_partition_candidates_truncated",
        "active_partition_structural_overflow",
        "active_partition_scan_contract",
        "active_partition_semantically_complete",
        "partition_scope_kind",
        "partition_inventory_total",
        "selected_partition_count",
        "partition_scope_exhaustive",
        "selected_scope_structurally_complete",
        "global_semantic_complete",
    }
    unknown_scan_fields = set(scan_fields) - allowed_scan_fields
    if unknown_scan_fields:
        raise ValueError(
            "unsupported active partition scan fields: "
            + ", ".join(sorted(unknown_scan_fields))
        )
    for field in allowed_scan_fields - {
        "active_partition_scan_contract",
        "active_partition_semantically_complete",
        "partition_scope_kind",
        "partition_scope_exhaustive",
        "selected_scope_structurally_complete",
        "global_semantic_complete",
    }:
        value = scan_fields.get(field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"{field} must be a non-negative integer")
    scan_contract = scan_fields.get("active_partition_scan_contract")
    if scan_contract is not None and not isinstance(scan_contract, str):
        raise ValueError("active_partition_scan_contract must be text")
    semantic_complete = scan_fields.get(
        "active_partition_semantically_complete"
    )
    if semantic_complete is not None and not isinstance(semantic_complete, bool):
        raise ValueError(
            "active_partition_semantically_complete must be boolean or null"
        )
    scope_kind = scan_fields.get(
        "partition_scope_kind",
        "approximate_top_k",
    )
    if scope_kind not in {
        "approximate_top_k",
        "global",
        "authoritative",
    }:
        raise ValueError(
            "partition_scope_kind must be approximate_top_k, global, or "
            "authoritative"
        )
    for field in (
        "partition_scope_exhaustive",
        "selected_scope_structurally_complete",
        "global_semantic_complete",
    ):
        value = scan_fields.get(field)
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"{field} must be boolean or null")
    selected_scope_complete = scan_fields.get(
        "selected_scope_structurally_complete"
    )
    if selected_scope_complete is None:
        # The legacy flag proved only the currently selected active
        # partitions.  Preserve that meaning while refusing to promote it
        # to a global proof.
        selected_scope_complete = semantic_complete
    elif semantic_complete is None:
        semantic_complete = selected_scope_complete
    elif selected_scope_complete is not semantic_complete:
        raise ValueError(
            "selected-scope structural completeness disagrees with the "
            "legacy active-partition semantic flag"
        )
    inventory_total = scan_fields.get("partition_inventory_total")
    selected_partition_count = scan_fields.get("selected_partition_count")
    if (
        inventory_total is not None
        and selected_partition_count is not None
        and selected_partition_count > inventory_total
    ):
        raise ValueError(
            "selected_partition_count cannot exceed partition_inventory_total"
        )
    partition_scope_exhaustive = scan_fields.get(
        "partition_scope_exhaustive"
    )
    if inventory_total is not None and selected_partition_count is not None:
        count_exhaustive = selected_partition_count == inventory_total
        if (
            partition_scope_exhaustive is not None
            and partition_scope_exhaustive is not count_exhaustive
        ):
            raise ValueError(
                "partition_scope_exhaustive disagrees with partition counts"
            )
        partition_scope_exhaustive = count_exhaustive
    global_semantic_complete = scan_fields.get("global_semantic_complete")
    if scope_kind == "global" and partition_scope_exhaustive is not True:
        raise ValueError("global partition scope must be exhaustive")
    if scope_kind == "global" and (
        inventory_total is None or selected_partition_count is None
    ):
        raise ValueError(
            "global partition scope requires explicit inventory counts"
        )
    if global_semantic_complete is True:
        if selected_scope_complete is not True:
            raise ValueError(
                "global semantic completeness requires selected-scope "
                "structural completeness"
            )
        if scope_kind == "approximate_top_k":
            raise ValueError(
                "approximate top-k partition scope cannot claim global "
                "semantic completeness"
            )
    if active_partition_total is not None and (
        isinstance(active_partition_total, bool)
        or not isinstance(active_partition_total, int)
        or active_partition_total < 0
    ):
        raise ValueError("active_partition_total must be non-negative")
    if (
        active_partition_inspected is not None
        and (
            isinstance(active_partition_inspected, bool)
            or not isinstance(active_partition_inspected, int)
            or active_partition_inspected < 0
        )
    ):
        raise ValueError("active_partition_inspected must be non-negative")
    if (
        active_partition_total is not None
        and active_partition_inspected is not None
        and active_partition_inspected > active_partition_total
    ):
        raise ValueError(
            "active_partition_inspected cannot exceed active_partition_total"
        )
    active_partition_exhaustive = (
        active_partition_inspected >= active_partition_total
        if active_partition_total is not None
        and active_partition_inspected is not None
        else None
    )
    if (
        scan_exhaustive is not None
        and scan_exhaustive is not active_partition_exhaustive
    ):
        raise ValueError("active partition exhaustive flag disagrees with counts")
    normalized_scan_fields = {
        "active_partition_sources_total": scan_fields.get(
            "active_partition_sources_total"
        ),
        "active_partition_structural_rows": int(
            scan_fields.get("active_partition_structural_rows", 0) or 0
        ),
        "active_partition_structural_hypotheses": int(
            scan_fields.get("active_partition_structural_hypotheses", 0) or 0
        ),
        "active_partition_candidates_admitted": int(
            scan_fields.get("active_partition_candidates_admitted", 0) or 0
        ),
        "active_partition_candidates_already_present": int(
            scan_fields.get(
                "active_partition_candidates_already_present", 0
            )
            or 0
        ),
        "active_partition_candidates_replaced": int(
            scan_fields.get("active_partition_candidates_replaced", 0) or 0
        ),
        "active_partition_candidates_truncated": int(
            scan_fields.get("active_partition_candidates_truncated", 0) or 0
        ),
        "active_partition_structural_overflow": int(
            scan_fields.get("active_partition_structural_overflow", 0) or 0
        ),
        "active_partition_scan_contract": str(scan_contract or ""),
        "active_partition_semantically_complete": semantic_complete,
        "partition_scope_kind": str(scope_kind),
        "partition_inventory_total": inventory_total,
        "selected_partition_count": selected_partition_count,
        "partition_scope_exhaustive": partition_scope_exhaustive,
        "selected_scope_structurally_complete": selected_scope_complete,
        "global_semantic_complete": global_semantic_complete,
    }
    if not unique:
        return self._fail_open(
            unique,
            program,
            started=started,
            reason="empty candidates",
            active_partition_total=active_partition_total,
            active_partition_inspected=active_partition_inspected,
            active_partition_scan=normalized_scan_fields,
        )
    if not program.requires_completeness:
        return self._fail_open(
            unique,
            program,
            started=started,
            reason="",
            selection_status="bypassed",
            bypass_reason="not a set query",
            active_partition_total=active_partition_total,
            active_partition_inspected=active_partition_inspected,
            active_partition_scan=normalized_scan_fields,
        )

    score_provider_fallback = ""
    score_provider_report: dict[
        str,
        str | int | float | bool | None,
    ] | None = None
    timestamps = source_timestamps or {}
    # A complete ordered/fixed performance request has one conservative
    # structural signal that is stronger than the uncalibrated choice
    # head: an explicitly required-role row directly states completed
    # first-person attendance.  Select the earliest row per
    # high-confidence event key;
    # distinct keys in one source survive, while exact keyed recaps across
    # sources contract.  This mapping is local to this call; neither keys
    # nor transformer state are persisted.
    typed_performance_frontier = bool(
        is_performance_query(query)
        and (
            program.quantifier
            in {SetQuantifier.ALL, SetQuantifier.FIXED}
            or program.ordering is not SetOrdering.NONE
        )
    )
    direct_performance_scan_contract = bool(
        scan_contract == "direct_performance_source_occurrence_v1"
        and active_partition_exhaustive is True
    )

    def raw_occurrence_order(
        item: tuple[int, RetrievalResult],
    ) -> tuple[bool, float, int]:
        index, result = item
        created_at = (
            result.turn.created_at if result.turn is not None else None
        )
        try:
            value = float(created_at.timestamp()) if created_at else 0.0
        except (AttributeError, OSError, OverflowError, ValueError):
            value = 0.0
            created_at = None
        return created_at is None, value, index

    direct_performance_by_key: dict[
        str,
        list[tuple[int, RetrievalResult]],
    ] = defaultdict(list)
    direct_performance_rows: list[
        tuple[int, RetrievalResult, str | None]
    ] = []
    if typed_performance_frontier:
        for index, result in enumerate(unique):
            role = (
                result.turn.role.casefold()
                if result.turn is not None
                else ""
            )
            source_id = result.durable_source_id
            if (
                (
                    program.required_evidence_role is not None
                    and role != program.required_evidence_role
                )
                or self._timestamp_in_scope(
                    program,
                    timestamps.get(source_id),
                )
                is False
                or not is_direct_past_performance(
                    query,
                    result.chunk.text,
                )
            ):
                continue
            event_key = performance_event_key(query, result.chunk.text)
            direct_performance_rows.append((index, result, event_key))
            if event_key is not None:
                direct_performance_by_key[event_key].append((index, result))
    if direct_performance_scan_contract:
        # The validated scanner already bounded and provenance-checked
        # each occurrence.  Preserve one earliest structural row per
        # audited transient identity.  Baseline recaps carrying the same
        # key can still merge below, but cannot create another slot.
        performance_primary_items = [
            (
                event_key,
                min(structural_rows, key=raw_occurrence_order)[1],
            )
            for event_key, rows in direct_performance_by_key.items()
            for structural_rows in [
                [
                    item
                    for item in rows
                    if item[1].route == "active_partition_structural"
                ]
            ]
            if structural_rows
        ]
    else:
        performance_primary_items = [
            (event_key, min(rows, key=raw_occurrence_order)[1])
            for event_key, rows in direct_performance_by_key.items()
            if rows
        ]
    performance_event_keys_by_id = {
        result.chunk.chunk_id: event_key
        for _index, result, event_key in direct_performance_rows
        if event_key is not None
    }
    performance_primary_ids = {
        result.chunk.chunk_id
        for _event_id, result in performance_primary_items
    }
    effective_answerability = answerability_scores
    if effective_answerability is None and self.score_provider is not None:
        try:
            score_candidates = self.score_provider.score_candidates
            parameters = inspect.signature(score_candidates).parameters
            supports_timestamps = "source_timestamps" in parameters or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            provided = score_candidates(
                query,
                unique,
                **(
                    {"source_timestamps": timestamps}
                    if supports_timestamps
                    else {}
                ),
            )
            if not isinstance(provided, Mapping):
                raise TypeError("score provider did not return a mapping")
            effective_answerability = provided
            raw_provider_report = getattr(
                self.score_provider,
                "last_report",
                None,
            )
            dump_provider_report = getattr(
                raw_provider_report,
                "model_dump",
                None,
            )
            if callable(dump_provider_report):
                raw_provider_report = dump_provider_report()
            if isinstance(raw_provider_report, Mapping):
                allowed = {
                    "model_id",
                    "model_revision",
                    "checkpoint_sha256",
                    "device",
                    "dtype",
                    "runtime",
                    "input_candidates",
                    "inspected_candidates",
                    "output_candidates",
                    "forward_passes",
                    "peak_workspace_tokens",
                    "total_workspace_tokens",
                    "workspace_tokens",
                    "total_sequence_tokens",
                    "elapsed_s",
                    "retained_transformer_state_bytes",
                    "fallback_reason",
                }
                score_provider_report = {
                    str(key): value
                    for key, value in raw_provider_report.items()
                    if key in allowed
                    and (
                        value is None
                        or isinstance(value, (str, int, float, bool))
                    )
                }
                if int(
                    score_provider_report.get(
                        "retained_transformer_state_bytes",
                        0,
                    )
                    or 0
                ):
                    raise RuntimeError("score provider retained transformer state")
                provider_reason = str(
                    score_provider_report.get("fallback_reason") or ""
                )
                provider_input = int(
                    score_provider_report.get("input_candidates") or 0
                )
                provider_inspected = int(
                    score_provider_report.get("inspected_candidates") or 0
                )
                if provider_reason:
                    score_provider_fallback = provider_reason
                elif provider_input and provider_inspected < provider_input:
                    score_provider_fallback = (
                        "non_exhaustive_score_provider:"
                        f"{provider_inspected}/{provider_input}"
                    )
        except Exception as exc:
            # The neural value head is optional. Its failure cannot erase
            # the deterministic surface-value and QK/OV path.
            if self.strict:
                raise
            score_provider_fallback = f"{type(exc).__name__}: {exc}"
            effective_answerability = None
            score_provider_report = None
    # The forced-choice question jointly asks whether a row directly
    # proves a requested member *and* states its value. Until a separately
    # trained membership head is supplied, reuse that one explicitly
    # shared, uncalibrated signal for both energies.
    effective_membership = (
        membership_scores
        if membership_scores is not None
        else effective_answerability
    )

    return _PreparedCoverage(
        started=started,
        query=query,
        program=program,
        unique=unique,
        max_results=max_results,
        timestamps=timestamps,
        semantic_scores=semantic_scores,
        active_partition_total=active_partition_total,
        active_partition_inspected=active_partition_inspected,
        active_partition_exhaustive=active_partition_exhaustive,
        normalized_scan_fields=normalized_scan_fields,
        score_provider_fallback=score_provider_fallback,
        score_provider_report=score_provider_report,
        typed_performance_frontier=typed_performance_frontier,
        performance_event_keys_by_id=performance_event_keys_by_id,
        performance_primary_ids=performance_primary_ids,
        effective_answerability=effective_answerability,
        effective_membership=effective_membership,
    )
