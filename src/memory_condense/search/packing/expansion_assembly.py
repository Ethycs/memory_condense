"""Expansion packet assembly over deterministic policy helpers."""

from __future__ import annotations

import inspect
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping

from memory_condense.domain._tokenizer import count_tokens, truncate_to_tokens
from memory_condense.domain.schemas import RetrievalResult
from memory_condense.search.packing.coverage_closure import (
    _CoverageClosureMixin,
    _report_value,
)
from memory_condense.search.packing.derived_scalar import (
    filter_conflicting_approximate_duration_recaps,
)
from memory_condense.search.packing.expansion_ordering import (
    _ExpansionOrderingMixin,
)
from memory_condense.search.packing.packing_contracts import (
    EXPANSION_PREFIX,
    AtomicExpansionContract,
)

# Sentinel for _trace_row: rows for selector-injected results historically
# omit the ``selector_output_rejection`` key entirely instead of carrying it
# as ``None``.  Preserve that shape difference exactly.
_OMIT = object()

_DETAIL_FIELDS = (
    "cross_encoder_input_rank",
    "cross_encoder_score",
    "cross_encoder_rank",
    "group_id",
    "group_role",
    "representative_chunk_id",
    "merge_similarity",
    "merge_threshold",
    "qk_score",
    "ov_transport",
    "prefix_utility",
    "semantic_score",
    "answer_object_key_present",
    "semantic_score_kind",
    "answerability_score",
    "answerability_score_kind",
    "membership_score",
    "preferred_evidence_role",
    "role_match",
    "value_evidence",
    "assignment_hypothesis",
    "p_existing",
    "p_new",
    "p_null",
    "existing_energy",
    "new_energy",
    "null_energy",
    "temporal_in_scope",
    "posterior_entropy",
    "posterior_kind",
    "semantic_surprisal",
    "posterior_uncertain",
    "credible_cluster",
    "coverage_reserved",
    "reservation_basis",
)


@dataclass
class _ExpansionPass:
    """Cross-phase state for a single ``_build_expansions`` invocation."""

    query: str
    original: list[RetrievalResult]
    source_metadata: dict[str, str] | None
    active_partition_total: int | None
    active_partition_inspected: int | None
    active_partition_scan: Mapping[str, Any] | None
    # Phase 1: candidate ordering.
    original_rank: dict[str, int] = field(default_factory=dict)
    ranked: list[RetrievalResult] = field(default_factory=list)
    source_timestamps: dict[str, str] = field(default_factory=dict)
    metadata_rank: dict[str, int] = field(default_factory=dict)
    # Phase 2: selector negotiation.
    selector_input: list[RetrievalResult] = field(default_factory=list)
    rejected_selector_rows: list[tuple[RetrievalResult, str, int]] = field(
        default_factory=list
    )
    selector_report: Any = None
    selector_report_is_current: bool = False
    selector_output_rejected: bool = False
    returned_ids: set[str] = field(default_factory=set)
    # Phase 3: diagnostic trace construction.
    trace_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    selector_trace_rows: list[Mapping[str, Any]] = field(default_factory=list)
    reserved_ids: set[str] = field(default_factory=set)
    packing_ranked: list[RetrievalResult] = field(default_factory=list)
    # Phase 4: coverage reservation planning.
    used_tokens: int = 0
    requested_reserved_rows: list[RetrievalResult] = field(default_factory=list)
    reservation_bodies: dict[str, str] = field(default_factory=dict)
    reservation_snippets: dict[str, str] = field(default_factory=dict)
    active_reserved_rows: list[RetrievalResult] = field(default_factory=list)
    active_reserved_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class _PackedExpansionRow:
    """One exact excerpt body retained across atomic envelope insertion."""

    result: RetrievalResult
    snippet: str


class _ExpansionPackingMixin(
    _CoverageClosureMixin,
    _ExpansionOrderingMixin,
):
    def _build_expansions(
        self,
        expansions: list[RetrievalResult],
        *,
        query: str = "",
        source_metadata: dict[str, str] | None = None,
        active_partition_total: int | None = None,
        active_partition_inspected: int | None = None,
        active_partition_scan: Mapping[str, Any] | None = None,
        atomic_contract: AtomicExpansionContract | None = None,
    ) -> tuple[list[str], list[str], int, int, dict[str, int]]:
        """Verbatim excerpts, each capped, and capped again in aggregate.

        The final excerpt is shortened to the remaining aggregate budget.  The
        old implementation dropped it wholesale, often leaving a material
        fraction of the fixed budget unused even though more ranked evidence
        was available.
        """
        if atomic_contract is not None:
            return self._build_atomic_expansions(
                expansions,
                query=query,
                source_metadata=source_metadata,
                active_partition_total=active_partition_total,
                active_partition_inspected=active_partition_inspected,
                active_partition_scan=active_partition_scan,
                contract=atomic_contract,
            )
        self.last_expansion_trace = []
        self._last_packed_expansion_rows: list[_PackedExpansionRow] = []
        self._last_expansion_source_timestamps: dict[str, str] = {}
        self.last_closure_report = {
            "applied": False,
            "closure_scope": "",
            "closure_global_recall_guaranteed": False,
        }
        state = _ExpansionPass(
            query=query,
            original=list(expansions),
            source_metadata=source_metadata,
            active_partition_total=active_partition_total,
            active_partition_inspected=active_partition_inspected,
            active_partition_scan=active_partition_scan,
        )
        self._order_expansion_candidates(state, expansions)
        self._negotiate_expansion_selector(state)
        self._build_expansion_trace(state)
        self._plan_coverage_reservations(state)
        self._apply_post_coverage_closure(state)
        return self._pack_expansion_rows(state, len(expansions))

    def _build_atomic_expansions(
        self,
        expansions: list[RetrievalResult],
        *,
        query: str,
        source_metadata: dict[str, str] | None,
        active_partition_total: int | None,
        active_partition_inspected: int | None,
        active_partition_scan: Mapping[str, Any] | None,
        contract: AtomicExpansionContract,
    ) -> tuple[list[str], list[str], int, int, dict[str, int]]:
        """Pack originals normally, then admit complete envelope groups only."""

        expansion_by_id: dict[str, RetrievalResult] = {}
        for result in expansions:
            chunk_id = result.chunk.chunk_id
            if chunk_id in expansion_by_id:
                raise ValueError("atomic expansion inputs must have unique chunk IDs")
            expansion_by_id[chunk_id] = result
        contract_ids = set(
            (*contract.original_chunk_ids, *contract.companion_chunk_ids)
        )
        if not contract_ids.issubset(expansion_by_id):
            raise ValueError("atomic expansion contract references an absent chunk")

        companion_ids = set(contract.companion_chunk_ids)
        original_ids = set(contract.original_chunk_ids)
        # Hydration renders groups chronologically, but the receipt separately
        # retains the caller's raw retrieval order. Apply every ordinary
        # ranking/selection policy to that raw order before any group rewrite.
        baseline_candidates = [
            expansion_by_id[chunk_id]
            for chunk_id in contract.original_chunk_ids
        ]
        baseline_candidates.extend(
            result
            for result in expansions
            if result.chunk.chunk_id not in companion_ids
            and result.chunk.chunk_id not in original_ids
        )
        baseline = self._build_expansions(
            baseline_candidates,
            query=query,
            source_metadata=source_metadata,
            active_partition_total=active_partition_total,
            active_partition_inspected=active_partition_inspected,
            active_partition_scan=active_partition_scan,
        )
        (
            baseline_texts,
            baseline_chunk_ids,
            baseline_tokens,
            _baseline_dropped,
            baseline_source_tokens,
        ) = baseline
        if not contract.groups:
            return (
                baseline_texts,
                baseline_chunk_ids,
                baseline_tokens,
                len(expansions) - len(baseline_chunk_ids),
                baseline_source_tokens,
            )

        baseline_rows = list(self._last_packed_expansion_rows)
        baseline_snippets = {
            row.result.chunk.chunk_id: row.snippet for row in baseline_rows
        }
        baseline_ids = set(baseline_snippets)
        baseline_trace = [dict(row) for row in self.last_expansion_trace]
        source_timestamps = dict(self._last_expansion_source_timestamps)
        current_rows = baseline_rows
        rendered: tuple[
            list[_PackedExpansionRow],
            list[str],
            list[str],
            int,
            dict[str, int],
        ] | None = None
        admitted_group_indexes: set[int] = set()

        for group_index, group in enumerate(contract.groups):
            group_original_ids = set(group.original_chunk_ids)
            surviving_group_originals = group_original_ids.intersection(
                baseline_ids
            )
            if not surviving_group_originals:
                continue
            current_by_id = {
                row.result.chunk.chunk_id: row.result for row in current_rows
            }
            insertion_index = min(
                index
                for index, row in enumerate(current_rows)
                if row.result.chunk.chunk_id in surviving_group_originals
            )
            remaining = [
                row.result
                for row in current_rows
                if row.result.chunk.chunk_id not in group_original_ids
            ]
            ordered_group = [
                current_by_id.get(chunk_id, expansion_by_id[chunk_id])
                for chunk_id in group.ordered_chunk_ids
            ]
            proposed = [
                *remaining[:insertion_index],
                *ordered_group,
                *remaining[insertion_index:],
            ]
            proposed_companion_count = sum(
                result.chunk.chunk_id in companion_ids for result in proposed
            )
            if proposed_companion_count > contract.max_companion_chunks:
                continue
            proposed_direct_count = sum(
                result.chunk.chunk_id not in companion_ids
                and result.route != "live_consolidation"
                for result in proposed
            )
            proposed_consolidation_count = sum(
                result.chunk.chunk_id not in companion_ids
                and result.route == "live_consolidation"
                for result in proposed
            )
            if (
                proposed_direct_count > self.budget.max_expansions
                or proposed_consolidation_count
                > self.budget.max_consolidation_expansions
            ):
                continue
            trial = self._fit_atomic_expansion_rows(
                proposed,
                query=query,
                source_timestamps=source_timestamps,
                baseline_snippets=baseline_snippets,
                additive_ids=contract_ids - baseline_ids,
            )
            if trial is None:
                continue
            current_rows = trial[0]
            rendered = trial
            admitted_group_indexes.add(group_index)

        if rendered is None:
            self._finalize_atomic_expansion_trace(
                expansions,
                baseline_rows,
                baseline_trace=baseline_trace,
                source_timestamps=source_timestamps,
                contract=contract,
                admitted_group_indexes=admitted_group_indexes,
            )
            return (
                baseline_texts,
                baseline_chunk_ids,
                baseline_tokens,
                len(expansions) - len(baseline_chunk_ids),
                baseline_source_tokens,
            )

        rows, texts, chunk_ids, used, source_tokens = rendered
        self._last_packed_expansion_rows = list(rows)
        self._finalize_atomic_expansion_trace(
            expansions,
            rows,
            baseline_trace=baseline_trace,
            source_timestamps=source_timestamps,
            contract=contract,
            admitted_group_indexes=admitted_group_indexes,
        )
        return (
            texts,
            chunk_ids,
            used,
            len(expansions) - len(chunk_ids),
            source_tokens,
        )

    def _fit_atomic_expansion_rows(
        self,
        rows: list[RetrievalResult],
        *,
        query: str,
        source_timestamps: Mapping[str, str],
        baseline_snippets: Mapping[str, str],
        additive_ids: set[str],
    ) -> tuple[
        list[_PackedExpansionRow],
        list[str],
        list[str],
        int,
        dict[str, int],
    ] | None:
        """Fit all rows under one token ceiling without shrinking originals."""

        label_state = _ExpansionPass(
            query=query,
            original=[],
            source_metadata=None,
            active_partition_total=None,
            active_partition_inspected=None,
            active_partition_scan=None,
        )
        label_state.source_timestamps = dict(source_timestamps)
        prepared_additions = {
            result.chunk.chunk_id: self._prepare_expansion_text(
                result.chunk.text,
                query,
            )
            for result in rows
            if result.chunk.chunk_id in additive_ids
        }

        def render(
            companion_content_cap: int,
        ) -> tuple[
            list[_PackedExpansionRow],
            list[str],
            list[str],
            int,
            dict[str, int],
        ] | None:
            packed_rows: list[_PackedExpansionRow] = []
            texts: list[str] = []
            chunk_ids: list[str] = []
            source_tokens: dict[str, int] = defaultdict(int)
            used = count_tokens(EXPANSION_PREFIX)
            for ordinal, result in enumerate(rows, start=1):
                chunk_id = result.chunk.chunk_id
                if chunk_id in baseline_snippets:
                    snippet = baseline_snippets[chunk_id]
                else:
                    prepared = prepared_additions.get(chunk_id)
                    if prepared is None:
                        raise ValueError(
                            "atomic expansion row is neither baseline nor companion"
                        )
                    snippet = truncate_to_tokens(
                        prepared,
                        companion_content_cap,
                    )
                if not snippet:
                    return None
                label = self._expansion_label(label_state, result, ordinal)
                entry = label + snippet
                cost = count_tokens(entry) + 1
                if used + cost > self.budget.expansion_tokens:
                    return None
                packed_rows.append(_PackedExpansionRow(result, snippet))
                texts.append(entry)
                chunk_ids.append(chunk_id)
                used += cost
                source_tokens[self._result_source_id(result)] += count_tokens(
                    snippet
                )
            return packed_rows, texts, chunk_ids, used, dict(source_tokens)

        maximum = self.budget.max_expansion_tokens
        fitted = render(maximum)
        if fitted is not None:
            return fitted
        if not prepared_additions or maximum < 1:
            return None
        lower = 1
        upper = maximum - 1
        best = None
        while lower <= upper:
            midpoint = (lower + upper) // 2
            candidate = render(midpoint)
            if candidate is None:
                upper = midpoint - 1
            else:
                best = candidate
                lower = midpoint + 1
        return best

    def _finalize_atomic_expansion_trace(
        self,
        expansions: list[RetrievalResult],
        packed_rows: list[_PackedExpansionRow],
        *,
        baseline_trace: list[dict[str, Any]],
        source_timestamps: Mapping[str, str],
        contract: AtomicExpansionContract,
        admitted_group_indexes: set[int],
    ) -> None:
        """Expose text-free atomic decisions while retaining selector detail."""

        trace_by_id = {
            str(row["chunk_id"]): dict(row)
            for row in baseline_trace
            if isinstance(row.get("chunk_id"), str)
        }
        group_index_by_id = {
            chunk_id: group_index
            for group_index, group in enumerate(contract.groups)
            for chunk_id in group.ordered_chunk_ids
        }
        input_rank = {
            result.chunk.chunk_id: rank
            for rank, result in enumerate(expansions, start=1)
        }
        for result in expansions:
            chunk_id = result.chunk.chunk_id
            diagnostic = trace_by_id.get(chunk_id)
            if diagnostic is None:
                diagnostic = self._trace_row(
                    result,
                    original_rank=input_rank[chunk_id],
                    selector_input_rank=None,
                    post_selector_rank=None,
                    cutoff_reason="atomic_group_no_fit",
                    selector_output_rejection=None,
                )
                trace_by_id[chunk_id] = diagnostic
            group_index = group_index_by_id.get(chunk_id)
            if group_index is not None:
                diagnostic["atomic_expansion_group"] = group_index
                diagnostic["atomic_expansion_admitted"] = (
                    group_index in admitted_group_indexes
                )

        label_state = _ExpansionPass(
            query="",
            original=[],
            source_metadata=None,
            active_partition_total=None,
            active_partition_inspected=None,
            active_partition_scan=None,
        )
        label_state.source_timestamps = dict(source_timestamps)
        used = count_tokens(EXPANSION_PREFIX)
        for ordinal, row in enumerate(packed_rows, start=1):
            chunk_id = row.result.chunk.chunk_id
            entry = self._expansion_label(
                label_state,
                row.result,
                ordinal,
            ) + row.snippet
            cost = count_tokens(entry) + 1
            used += cost
            trace_by_id[chunk_id].update(
                {
                    "packed_rank": ordinal,
                    "cutoff_reason": "packed",
                    "content_tokens": count_tokens(row.snippet),
                    "rendered_tokens": cost,
                    "cumulative_tokens": used,
                }
            )
        self.last_expansion_trace = list(trace_by_id.values())

    def _order_expansion_candidates(
        self,
        state: _ExpansionPass,
        expansions: list[RetrievalResult],
    ) -> None:
        """Phase 1: baseline order, provenance binding, and rank indexes."""
        for rank, result in enumerate(state.original, start=1):
            state.original_rank.setdefault(result.chunk.chunk_id, rank)
        ranked = (
            self._heat_weighted_order(expansions)
            if self.budget.heat_weighted_expansions
            else expansions
        )
        if self.budget.source_metadata_expansions:
            # Provenance is not evidence. Resolve synthetic timestamp rows
            # before estimating information gain so their unique source IDs
            # and date numbers cannot crowd real conversational content out
            # of the packet. The timestamp remains attached to every emitted
            # excerpt from that source below.
            state.source_timestamps, ranked = self._bind_source_metadata(
                ranked,
                candidate_pool=ranked,
                source_metadata=state.source_metadata,
            )
        state.metadata_rank = {
            result.chunk.chunk_id: rank
            for rank, result in enumerate(ranked, start=1)
        }
        state.ranked = ranked

    def _negotiate_expansion_selector(self, state: _ExpansionPass) -> None:
        """Phase 2: run the selector and validate its output fail-open."""
        query = state.query
        ranked = state.ranked
        source_timestamps = state.source_timestamps
        active_partition_total = state.active_partition_total
        active_partition_inspected = state.active_partition_inspected
        active_partition_scan = state.active_partition_scan
        selector_report: Any = None
        selector_report_is_current = False
        selector_output_rejected = False
        returned_ids: set[str] = set()
        if self.expansion_selector is not None:
            selector_report_before = getattr(
                self.expansion_selector,
                "last_report",
                None,
            )
            complete_frontier_for = getattr(
                self.expansion_selector,
                "requires_complete_frontier_for",
                None,
            )
            requires_complete_frontier = (
                bool(complete_frontier_for(query))
                if callable(complete_frontier_for)
                else bool(
                    getattr(
                        self.expansion_selector,
                        "requires_complete_frontier",
                        False,
                    )
                )
            )
            # The frozen-prefix arm is deliberately monotonic: preserve the
            # measured baseline ranker and only demote likely duplicate
            # support. A semantic selector may instead replace that ranking.
            if getattr(
                self.expansion_selector,
                "requires_baseline_ranking",
                False,
            ) and not requires_complete_frontier:
                if self.budget.information_gain_expansions:
                    ranked = self._information_gain_order(ranked, query=query)
                elif self.budget.budget_aware_expansions:
                    ranked = self._budget_aware_order(ranked, query=query)
            selector_input = list(ranked)
            select_kwargs: dict[str, Any] = {
                "source_timestamps": source_timestamps,
            }
            scan_fields = dict(active_partition_scan or {})
            scan_total = scan_fields.get("active_partition_total")
            scan_inspected = scan_fields.get("active_partition_inspected")
            if active_partition_total is None and scan_total is not None:
                active_partition_total = scan_total
            if active_partition_inspected is None and scan_inspected is not None:
                active_partition_inspected = scan_inspected
            if (
                active_partition_total is not None
                or active_partition_inspected is not None
                or scan_fields
            ):
                try:
                    selector_parameters = inspect.signature(
                        self.expansion_selector.select
                    ).parameters.values()
                except (TypeError, ValueError):
                    selector_parameters = ()
                accepts_partition = {
                    parameter.name for parameter in selector_parameters
                }
                accepts_kwargs = any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in selector_parameters
                )
                if accepts_kwargs or "active_partition_total" in accepts_partition:
                    select_kwargs["active_partition_total"] = (
                        active_partition_total
                    )
                if (
                    accepts_kwargs
                    or "active_partition_inspected" in accepts_partition
                ):
                    select_kwargs["active_partition_inspected"] = (
                        active_partition_inspected
                    )
                if accepts_kwargs or "active_partition_scan" in accepts_partition:
                    select_kwargs["active_partition_scan"] = scan_fields
            returned = self.expansion_selector.select(
                query,
                ranked,
                **select_kwargs,
            )
            allowed_by_id: dict[str, RetrievalResult] = {}
            for result in selector_input:
                allowed_by_id.setdefault(result.chunk.chunk_id, result)
            ranked = []
            rejected_selector_rows: list[tuple[RetrievalResult, str, int]] = []
            for returned_rank, result in enumerate(returned, start=1):
                if not isinstance(result, RetrievalResult):
                    selector_output_rejected = True
                    if bool(getattr(self.expansion_selector, "strict", False)):
                        raise TypeError("selector returned a non-RetrievalResult row")
                    continue
                chunk_id = result.chunk.chunk_id
                expected = allowed_by_id.get(chunk_id)
                if expected is None:
                    selector_output_rejected = True
                    rejected_selector_rows.append(
                        (result, "selector_injected_rejected", returned_rank)
                    )
                    continue
                if expected is not result:
                    selector_output_rejected = True
                    rejected_selector_rows.append(
                        (result, "selector_replacement_rejected", returned_rank)
                    )
                    continue
                if chunk_id in returned_ids:
                    selector_output_rejected = True
                    rejected_selector_rows.append(
                        (result, "selector_duplicate_rejected", returned_rank)
                    )
                    continue
                returned_ids.add(chunk_id)
                ranked.append(result)
            if rejected_selector_rows and bool(
                getattr(self.expansion_selector, "strict", False)
            ):
                reasons = ", ".join(
                    reason for _result, reason, _rank in rejected_selector_rows
                )
                raise ValueError(f"unsafe selector output: {reasons}")
            # A selector may prioritize exact inputs but cannot silently erase
            # an omitted row. Preserve every unreturned original at the tail;
            # downstream budget accounting, not a malformed/partial selector,
            # remains the only destructive cutoff.
            ranked.extend(
                result
                for result in selector_input
                if result.chunk.chunk_id not in returned_ids
            )
            selector_report = getattr(
                self.expansion_selector,
                "last_report",
                None,
            )
            # Closure may delete evidence, so a stale report from an earlier
            # query is never sufficient. The production selector replaces its
            # frozen report on every call; mutable/reused reports fail open.
            selector_report_is_current = bool(
                selector_report is not None
                and selector_report is not selector_report_before
            )
        elif self.budget.information_gain_expansions:
            ranked = self._information_gain_order(ranked, query=query)
            selector_input = list(ranked)
            rejected_selector_rows = []
        elif self.budget.budget_aware_expansions:
            ranked = self._budget_aware_order(ranked, query=query)
            selector_input = list(ranked)
            rejected_selector_rows = []
        else:
            selector_input = list(ranked)
            rejected_selector_rows = []
        state.ranked = ranked
        state.selector_input = selector_input
        state.rejected_selector_rows = rejected_selector_rows
        state.selector_report = selector_report
        state.selector_report_is_current = selector_report_is_current
        state.selector_output_rejected = selector_output_rejected
        state.returned_ids = returned_ids

    def _trace_row(
        self,
        result: RetrievalResult,
        *,
        original_rank: int | None,
        selector_input_rank: int | None,
        post_selector_rank: int | None,
        cutoff_reason: str,
        selector_output_rejection: Any = _OMIT,
    ) -> dict[str, Any]:
        """Text-free diagnostic row shared by every trace entry.

        ``selector_output_rejection`` is only present when the caller passes
        it: rows for results that entered through the original ranking always
        carry the key (possibly ``None``), while rows for selector-injected
        results omit it and record the rejection as their ``cutoff_reason``.
        """
        row: dict[str, Any] = {
            "chunk_id": result.chunk.chunk_id,
            "source_id": self._result_source_id(result),
            "route": result.route or "",
            "anchor_chunk_id": result.anchor_chunk_id,
            "original_rank": original_rank,
            "selector_input_rank": selector_input_rank,
            "post_selector_rank": post_selector_rank,
            "packed_rank": None,
            "cutoff_reason": cutoff_reason,
            "chunk_tokens": int(result.chunk.token_count),
            "content_tokens": None,
            "rendered_tokens": None,
            "cumulative_tokens": None,
        }
        if selector_output_rejection is not _OMIT:
            row["selector_output_rejection"] = selector_output_rejection
        return row

    def _build_expansion_trace(self, state: _ExpansionPass) -> None:
        """Phase 3: temporal conflict filter plus the per-candidate trace."""
        query = state.query
        original_rank = state.original_rank

        # A derived duration query needs two explicit temporal boundaries, not
        # exhaustive set coverage.  The normal IG path above restores those
        # operands; remove only a separately proven, approximate recap that
        # conflicts with their provenance dates.  The helper is fail-open and
        # returns the exact surviving RetrievalResult objects unchanged.
        ranked, temporal_conflicts = (
            filter_conflicting_approximate_duration_recaps(
                state.ranked,
                query=query,
                source_timestamps=state.source_timestamps,
            )
        )
        state.ranked = ranked

        selector_input_rank: dict[str, int] = {}
        for result in state.selector_input:
            selector_input_rank.setdefault(
                result.chunk.chunk_id,
                len(selector_input_rank) + 1,
            )
        post_selector_rank: dict[str, int] = {}
        for result in ranked:
            post_selector_rank.setdefault(
                result.chunk.chunk_id,
                len(post_selector_rank) + 1,
            )
        selector_details: dict[str, Mapping[str, Any]] = {}
        selector_trace_rows: list[Mapping[str, Any]] = []
        if self.expansion_selector is not None:
            for row in getattr(
                self.expansion_selector,
                "last_candidate_trace",
                (),
            ):
                if not isinstance(row, Mapping):
                    continue
                selector_trace_rows.append(row)
                chunk_id = row.get("chunk_id")
                if isinstance(chunk_id, str):
                    selector_details[chunk_id] = row
        rejected_input_reason = {
            result.chunk.chunk_id: reason
            for result, reason, _rank in state.rejected_selector_rows
            if result.chunk.chunk_id in original_rank
        }

        trace_by_id: dict[str, dict[str, Any]] = {}
        for result in state.original:
            chunk_id = result.chunk.chunk_id
            if chunk_id in trace_by_id:
                continue
            if chunk_id not in state.metadata_rank:
                reason = "source_metadata_filtered"
            elif chunk_id not in selector_input_rank:
                if self.budget.information_gain_expansions:
                    reason = "preselector_information_gain_filtered"
                elif self.budget.budget_aware_expansions:
                    reason = "preselector_budget_filtered"
                else:
                    reason = "preselector_filtered"
            elif chunk_id not in post_selector_rank:
                reason = (
                    "temporal_conflict_suppressed"
                    if chunk_id in temporal_conflicts
                    else rejected_input_reason.get(
                        chunk_id,
                        "selector_filtered",
                    )
                )
            else:
                reason = "pending"
            row = self._trace_row(
                result,
                original_rank=original_rank[chunk_id],
                selector_input_rank=selector_input_rank.get(chunk_id),
                post_selector_rank=post_selector_rank.get(chunk_id),
                cutoff_reason=reason,
                selector_output_rejection=rejected_input_reason.get(chunk_id),
            )
            details = selector_details.get(chunk_id, {})
            for field_name in _DETAIL_FIELDS:
                value = details.get(field_name)
                if value is None or isinstance(value, (str, int, float, bool)):
                    row[field_name] = value
            conflict = temporal_conflicts.get(chunk_id)
            if conflict is not None:
                row.update(
                    {
                        "temporal_conflict_action": "suppressed",
                        "temporal_conflict_basis": conflict.reason,
                        "temporal_onset_chunk_id": conflict.onset_chunk_id,
                        "temporal_endpoint_chunk_id": conflict.endpoint_chunk_id,
                    }
                )
            trace_by_id[chunk_id] = row

        # A selector is an ordering policy, not an evidence source. Retain a
        # text-free diagnostic for rejected fabrications without ever letting
        # their payload reach the prompt.
        for result, rejection_reason, returned_rank in state.rejected_selector_rows:
            chunk_id = result.chunk.chunk_id
            if chunk_id in trace_by_id:
                continue
            trace_by_id[chunk_id] = self._trace_row(
                result,
                original_rank=None,
                selector_input_rank=None,
                post_selector_rank=returned_rank,
                cutoff_reason=rejection_reason,
            )
        reserved_ids = {
            result.chunk.chunk_id
            for result in ranked
            if bool(
                selector_details.get(result.chunk.chunk_id, {}).get(
                    "coverage_reserved",
                    False,
                )
            )
            and selector_details.get(result.chunk.chunk_id, {}).get("group_role")
            == "representative"
        }
        # Enforce the selector's coverage contract even if a future composite
        # interleaves support rows. The objects themselves remain untouched,
        # preserving exact chunk/source provenance.
        packing_ranked = [
            result for result in ranked if result.chunk.chunk_id in reserved_ids
        ]
        packing_ranked.extend(
            result for result in ranked if result.chunk.chunk_id not in reserved_ids
        )
        for rank, result in enumerate(packing_ranked, start=1):
            trace_by_id[result.chunk.chunk_id]["coverage_pack_rank"] = rank
        state.trace_by_id = trace_by_id
        state.selector_trace_rows = selector_trace_rows
        state.reserved_ids = reserved_ids
        state.packing_ranked = packing_ranked

    def _expansion_label(
        self,
        state: _ExpansionPass,
        result: RetrievalResult,
        ordinal: int,
    ) -> str:
        source_id = self._result_source_id(result)
        timestamp = state.source_timestamps.get(source_id)
        role = (
            result.turn.role.strip().lower()
            if result.turn is not None and result.turn.role.strip()
            else ""
        )
        provenance = ""
        if timestamp:
            provenance += f" @ {timestamp}"
        if role:
            provenance += f" | {role}"
        return f"[{ordinal}{provenance}] "

    def _plan_coverage_reservations(self, state: _ExpansionPass) -> None:
        """Phase 4: admit a feasible reserved prefix and water-fill its caps."""
        used = count_tokens(EXPANSION_PREFIX)
        state.used_tokens = used
        requested_reserved_rows = [
            result
            for result in state.packing_ranked
            if result.chunk.chunk_id in state.reserved_ids
        ]
        # Query-aware sentence packing is a lossy optimization.  A coverage
        # representative has already been selected as the one row that must
        # carry its event, so applying that optimization here can turn a
        # 60-token evidence body into a 16-token sentence and then incorrectly
        # declare the 24-token reservation fulfilled.  Reservations therefore
        # allocate from the raw chunk body; ordinary evidence remains free to
        # use query-aware sentence packing below.
        reservation_bodies = {
            result.chunk.chunk_id: result.chunk.text.strip()
            for result in requested_reserved_rows
        }
        minimum_content = min(
            self.budget.min_coverage_expansion_tokens,
            self.budget.max_expansion_tokens,
        )
        active_reserved_rows: list[RetrievalResult] = []
        reservation_minimums: dict[str, int] = {}
        reservation_snippets: dict[str, str] = {}
        projected = used
        # Admit only a deterministic prefix that can carry labels plus a
        # useful excerpt from every event.  This prevents a large ALL set from
        # dividing the fair share to zero and aborting before any evidence is
        # emitted.
        for result in requested_reserved_rows:
            body = reservation_bodies[result.chunk.chunk_id]
            body_tokens = count_tokens(body)
            required_content = min(minimum_content, body_tokens)
            ordinal = len(active_reserved_rows) + 1
            minimum_snippet = truncate_to_tokens(body, required_content)
            minimum_snippet_tokens = count_tokens(minimum_snippet)
            required_cost = count_tokens(
                self._expansion_label(state, result, ordinal) + minimum_snippet
            ) + 1
            if minimum_snippet_tokens < required_content:
                # ``truncate_to_tokens`` is expected to round-trip token
                # prefixes, but keep the reservation invariant explicit if a
                # tokenizer implementation ever changes.
                break
            if required_content < 1 or (
                projected + required_cost > self.budget.expansion_tokens
            ):
                break
            active_reserved_rows.append(result)
            reservation_minimums[result.chunk.chunk_id] = required_content
            projected += required_cost

        active_reserved_ids = {
            result.chunk.chunk_id for result in active_reserved_rows
        }
        reservation_content_cap: int | None = None
        if active_reserved_rows:
            lower = max(reservation_minimums.values())
            upper = self.budget.max_expansion_tokens
            # Equal-cap water filling gives every active event its raw-body
            # minimum, while short rows return unused capacity to longer rows.
            # Cost the exact rendered label+body pair: estimating labels and
            # bodies independently lets BPE boundary differences accumulate
            # and can starve the final representative.
            while lower <= upper:
                midpoint = (lower + upper) // 2
                candidate_snippets = {
                    result.chunk.chunk_id: truncate_to_tokens(
                        reservation_bodies[result.chunk.chunk_id],
                        midpoint,
                    )
                    for result in active_reserved_rows
                }
                rendered_cost = used + sum(
                    count_tokens(
                        self._expansion_label(state, result, index)
                        + candidate_snippets[result.chunk.chunk_id]
                    )
                    + 1
                    for index, result in enumerate(
                        active_reserved_rows,
                        start=1,
                    )
                )
                if rendered_cost <= self.budget.expansion_tokens:
                    reservation_content_cap = midpoint
                    reservation_snippets = candidate_snippets
                    lower = midpoint + 1
                else:
                    upper = midpoint - 1
            if reservation_content_cap is None:
                # The prefix admission calculation proves the minimum fits;
                # this guard keeps that invariant explicit if token accounting
                # changes later.
                reservation_content_cap = max(reservation_minimums.values())
                reservation_snippets = {
                    result.chunk.chunk_id: truncate_to_tokens(
                        reservation_bodies[result.chunk.chunk_id],
                        reservation_content_cap,
                    )
                    for result in active_reserved_rows
                }

        for result in requested_reserved_rows:
            chunk_id = result.chunk.chunk_id
            active = chunk_id in active_reserved_ids
            state.trace_by_id[chunk_id].update(
                {
                    "coverage_reservation_requested": True,
                    "coverage_reservation_active": active,
                    "coverage_reservation_degraded": not active,
                    "coverage_reservation_feasible": active,
                    "coverage_content_cap": (
                        count_tokens(
                            reservation_snippets.get(chunk_id, "")
                        )
                        if active
                        else None
                    ),
                }
            )
        state.requested_reserved_rows = requested_reserved_rows
        state.reservation_bodies = reservation_bodies
        state.reservation_snippets = reservation_snippets
        state.active_reserved_rows = active_reserved_rows
        state.active_reserved_ids = active_reserved_ids

    def _apply_post_coverage_closure(self, state: _ExpansionPass) -> None:
        """Phase 5: stamp closure provenance and suppress non-members."""
        selector_report = state.selector_report
        selector_report_is_current = state.selector_report_is_current
        closure = self._post_coverage_closure_ids(
            selector_report=selector_report,
            selector_report_is_current=selector_report_is_current,
            selector_output_rejected=state.selector_output_rejected,
            selector_input=state.selector_input,
            returned_ids=state.returned_ids,
            selector_trace_rows=state.selector_trace_rows,
            requested_reserved_rows=state.requested_reserved_rows,
            active_reserved_ids=state.active_reserved_ids,
            reservation_bodies=state.reservation_bodies,
            reservation_snippets=state.reservation_snippets,
            source_timestamps=state.source_timestamps,
        )
        closure_chunk_ids, closure_scope, closure_global_recall_guaranteed = (
            closure if closure is not None else ((), "", False)
        )
        closure_id_set = set(closure_chunk_ids)
        closure_applied = closure is not None
        scope_provenance = {
            "partition_scope_kind": (
                _report_value(selector_report, "partition_scope_kind")
                if selector_report_is_current
                else None
            ),
            "partition_inventory_total": (
                _report_value(selector_report, "partition_inventory_total")
                if selector_report_is_current
                else None
            ),
            "selected_partition_count": (
                _report_value(selector_report, "selected_partition_count")
                if selector_report_is_current
                else None
            ),
            "partition_scope_exhaustive": (
                _report_value(selector_report, "partition_scope_exhaustive")
                if selector_report_is_current
                else None
            ),
            "selected_scope_structurally_complete": (
                _report_value(
                    selector_report,
                    "selected_scope_structurally_complete",
                )
                if selector_report_is_current
                else None
            ),
            "global_semantic_complete": (
                _report_value(selector_report, "global_semantic_complete")
                if selector_report_is_current
                else None
            ),
        }
        self.last_closure_report = {
            "applied": closure_applied,
            "closure_scope": closure_scope,
            "closure_global_recall_guaranteed": (
                closure_global_recall_guaranteed
            ),
            **scope_provenance,
        }
        for diagnostic in state.trace_by_id.values():
            diagnostic["post_coverage_closure_applied"] = closure_applied
            diagnostic["post_coverage_closed"] = False
            diagnostic["closure_scope"] = closure_scope
            diagnostic["closure_global_recall_guaranteed"] = (
                closure_global_recall_guaranteed
            )
            diagnostic.update(scope_provenance)
        if closure_applied:
            # The proof above includes full raw-body preflight for every
            # member, so no ordinary alternative is needed to complete this
            # exact FIXED answer set. Keep the exact trusted objects and mark
            # every suppressed row explicitly in the text-free trace.
            for chunk_id, diagnostic in state.trace_by_id.items():
                if chunk_id not in closure_id_set:
                    diagnostic["cutoff_reason"] = "post_coverage_closed"
                    diagnostic["post_coverage_closed"] = True
            state.packing_ranked = [
                result
                for result in state.active_reserved_rows
                if result.chunk.chunk_id in closure_id_set
            ]

    def _pack_expansion_rows(
        self,
        state: _ExpansionPass,
        expansion_count: int,
    ) -> tuple[list[str], list[str], int, int, dict[str, int]]:
        """Phase 6: emit budgeted excerpts and finalize the trace."""
        trace_by_id = state.trace_by_id
        self._last_expansion_source_timestamps = dict(state.source_timestamps)
        texts: list[str] = []
        chunk_ids: list[str] = []
        used = state.used_tokens
        source_tokens: dict[str, int] = defaultdict(int)
        direct_kept = 0
        consolidation_kept = 0
        token_cutoff = False
        for result in state.packing_ranked:
            diagnostic = trace_by_id[result.chunk.chunk_id]
            is_reserved = result.chunk.chunk_id in state.active_reserved_ids
            is_consolidation = result.route == "live_consolidation"
            if is_consolidation:
                if (
                    not is_reserved
                    and consolidation_kept
                    >= self.budget.max_consolidation_expansions
                ):
                    diagnostic["cutoff_reason"] = "consolidation_count_cap"
                    continue
            elif not is_reserved and direct_kept >= self.budget.max_expansions:
                diagnostic["cutoff_reason"] = "direct_count_cap"
                continue
            remaining = self.budget.expansion_tokens - used
            source_id = self._result_source_id(result)
            label = self._expansion_label(state, result, len(texts) + 1)
            if is_reserved:
                # This exact raw-body snippet was preflighted together with
                # every other active representative.  Never shrink it based
                # on what earlier rows happened to consume.
                snippet = state.reservation_snippets[result.chunk.chunk_id]
                content_budget = count_tokens(snippet)
            else:
                # Reserve the label and newline accounted for by this packer.
                content_budget = min(
                    self.budget.max_expansion_tokens,
                    remaining - count_tokens(label) - 1,
                )
                if content_budget <= 0:
                    diagnostic["cutoff_reason"] = "token_budget_exhausted"
                    token_cutoff = True
                    break
                prepared = self._prepare_expansion_text(
                    result.chunk.text, state.query
                )
                snippet = truncate_to_tokens(prepared, content_budget)
            if not snippet:
                diagnostic["cutoff_reason"] = "empty_after_prepare"
                continue
            entry = label + snippet
            cost = count_tokens(entry) + 1
            # Token boundaries can shift where the label meets the excerpt.
            # Tighten by the exact overage so the hard ceiling remains exact.
            if not is_reserved and used + cost > self.budget.expansion_tokens:
                snippet = truncate_to_tokens(
                    snippet, max(0, content_budget - (used + cost - self.budget.expansion_tokens))
                )
                entry = label + snippet
                cost = count_tokens(entry) + 1
            if not snippet or used + cost > self.budget.expansion_tokens:
                diagnostic["cutoff_reason"] = "token_budget_no_fit"
                token_cutoff = True
                break
            texts.append(entry)
            chunk_ids.append(result.chunk.chunk_id)
            self._last_packed_expansion_rows.append(
                _PackedExpansionRow(result, snippet)
            )
            if is_consolidation:
                consolidation_kept += 1
            else:
                direct_kept += 1
            used += cost
            source_tokens[source_id] += count_tokens(snippet)
            diagnostic.update(
                {
                    "packed_rank": len(texts),
                    "cutoff_reason": "packed",
                    "content_tokens": count_tokens(snippet),
                    "rendered_tokens": cost,
                    "cumulative_tokens": used,
                }
            )

        for diagnostic in trace_by_id.values():
            if diagnostic["cutoff_reason"] == "pending":
                diagnostic["cutoff_reason"] = (
                    "after_token_cutoff" if token_cutoff else "not_packed"
                )
        self.last_expansion_trace = list(trace_by_id.values())

        if not texts:
            return [], [], 0, expansion_count, {}

        return (
            texts,
            chunk_ids,
            used,
            expansion_count - len(texts),
            dict(source_tokens),
        )
