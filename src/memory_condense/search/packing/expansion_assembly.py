"""Expansion packet assembly over deterministic policy helpers."""

from __future__ import annotations

import inspect
from collections import defaultdict
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
from memory_condense.search.packing.packing_contracts import EXPANSION_PREFIX


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
    ) -> tuple[list[str], list[str], int, int, dict[str, int]]:
        """Verbatim excerpts, each capped, and capped again in aggregate.

        The final excerpt is shortened to the remaining aggregate budget.  The
        old implementation dropped it wholesale, often leaving a material
        fraction of the fixed budget unused even though more ranked evidence
        was available.
        """
        self.last_expansion_trace = []
        self.last_closure_report = {
            "applied": False,
            "closure_scope": "",
            "closure_global_recall_guaranteed": False,
        }
        original = list(expansions)
        original_rank: dict[str, int] = {}
        for rank, result in enumerate(original, start=1):
            original_rank.setdefault(result.chunk.chunk_id, rank)
        ranked = (
            self._heat_weighted_order(expansions)
            if self.budget.heat_weighted_expansions
            else expansions
        )
        source_timestamps: dict[str, str] = {}
        if self.budget.source_metadata_expansions:
            # Provenance is not evidence. Resolve synthetic timestamp rows
            # before estimating information gain so their unique source IDs
            # and date numbers cannot crowd real conversational content out
            # of the packet. The timestamp remains attached to every emitted
            # excerpt from that source below.
            source_timestamps, ranked = self._bind_source_metadata(
                ranked,
                candidate_pool=ranked,
                source_metadata=source_metadata,
            )
        metadata_rank = {
            result.chunk.chunk_id: rank
            for rank, result in enumerate(ranked, start=1)
        }
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

        # A derived duration query needs two explicit temporal boundaries, not
        # exhaustive set coverage.  The normal IG path above restores those
        # operands; remove only a separately proven, approximate recap that
        # conflicts with their provenance dates.  The helper is fail-open and
        # returns the exact surviving RetrievalResult objects unchanged.
        ranked, temporal_conflicts = (
            filter_conflicting_approximate_duration_recaps(
                ranked,
                query=query,
                source_timestamps=source_timestamps,
            )
        )

        selector_input_rank: dict[str, int] = {}
        for result in selector_input:
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
            for result, reason, _rank in rejected_selector_rows
            if result.chunk.chunk_id in original_rank
        }

        detail_fields = (
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
        trace_by_id: dict[str, dict[str, Any]] = {}
        for result in original:
            chunk_id = result.chunk.chunk_id
            if chunk_id in trace_by_id:
                continue
            if chunk_id not in metadata_rank:
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
            row: dict[str, Any] = {
                "chunk_id": chunk_id,
                "source_id": self._result_source_id(result),
                "route": result.route or "",
                "anchor_chunk_id": result.anchor_chunk_id,
                "original_rank": original_rank[chunk_id],
                "selector_input_rank": selector_input_rank.get(chunk_id),
                "post_selector_rank": post_selector_rank.get(chunk_id),
                "packed_rank": None,
                "cutoff_reason": reason,
                "chunk_tokens": int(result.chunk.token_count),
                "content_tokens": None,
                "rendered_tokens": None,
                "cumulative_tokens": None,
                "selector_output_rejection": rejected_input_reason.get(chunk_id),
            }
            details = selector_details.get(chunk_id, {})
            for field in detail_fields:
                value = details.get(field)
                if value is None or isinstance(value, (str, int, float, bool)):
                    row[field] = value
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
        for result, rejection_reason, returned_rank in rejected_selector_rows:
            chunk_id = result.chunk.chunk_id
            if chunk_id in trace_by_id:
                continue
            trace_by_id[chunk_id] = {
                "chunk_id": chunk_id,
                "source_id": self._result_source_id(result),
                "route": result.route or "",
                "anchor_chunk_id": result.anchor_chunk_id,
                "original_rank": None,
                "selector_input_rank": None,
                "post_selector_rank": returned_rank,
                "packed_rank": None,
                "cutoff_reason": rejection_reason,
                "chunk_tokens": int(result.chunk.token_count),
                "content_tokens": None,
                "rendered_tokens": None,
                "cumulative_tokens": None,
            }
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

        def expansion_label(result: RetrievalResult, ordinal: int) -> str:
            source_id = self._result_source_id(result)
            timestamp = source_timestamps.get(source_id)
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

        texts: list[str] = []
        chunk_ids: list[str] = []
        used = count_tokens(EXPANSION_PREFIX)
        source_tokens: dict[str, int] = defaultdict(int)
        direct_kept = 0
        consolidation_kept = 0
        token_cutoff = False
        requested_reserved_rows = [
            result
            for result in packing_ranked
            if result.chunk.chunk_id in reserved_ids
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
                expansion_label(result, ordinal) + minimum_snippet
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
                        expansion_label(result, index)
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
            trace_by_id[chunk_id].update(
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

        closure = self._post_coverage_closure_ids(
            selector_report=selector_report,
            selector_report_is_current=selector_report_is_current,
            selector_output_rejected=selector_output_rejected,
            selector_input=selector_input,
            returned_ids=returned_ids,
            selector_trace_rows=selector_trace_rows,
            requested_reserved_rows=requested_reserved_rows,
            active_reserved_ids=active_reserved_ids,
            reservation_bodies=reservation_bodies,
            reservation_snippets=reservation_snippets,
            source_timestamps=source_timestamps,
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
        for diagnostic in trace_by_id.values():
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
            for chunk_id, diagnostic in trace_by_id.items():
                if chunk_id not in closure_id_set:
                    diagnostic["cutoff_reason"] = "post_coverage_closed"
                    diagnostic["post_coverage_closed"] = True
            packing_ranked = [
                result
                for result in active_reserved_rows
                if result.chunk.chunk_id in closure_id_set
            ]

        for result in packing_ranked:
            diagnostic = trace_by_id[result.chunk.chunk_id]
            is_reserved = result.chunk.chunk_id in active_reserved_ids
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
            label = expansion_label(result, len(texts) + 1)
            if is_reserved:
                # This exact raw-body snippet was preflighted together with
                # every other active representative.  Never shrink it based
                # on what earlier rows happened to consume.
                snippet = reservation_snippets[result.chunk.chunk_id]
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
                prepared = self._prepare_expansion_text(result.chunk.text, query)
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
            return [], [], 0, len(expansions), {}

        return (
            texts,
            chunk_ids,
            used,
            len(expansions) - len(texts),
            dict(source_tokens),
        )
