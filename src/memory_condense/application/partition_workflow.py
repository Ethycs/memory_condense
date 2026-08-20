"""Stateful typed active-partition scanning and admission workflows."""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from datetime import datetime
from typing import Any, Sequence

import numpy as np

from memory_condense.application.condenser_contracts import (
    ActivePartitionHypothesis as _ActivePartitionHypothesis,
)
from memory_condense.persistence.transcript_store import parse_source_metadata
from memory_condense.search.indexes.lexical import tokenize
from memory_condense.search.packing.performance_events import (
    is_direct_past_performance,
    is_performance_query,
    performance_event_key,
)
from memory_condense.domain.schemas import RetrievalResult


_ACTIVE_PARTITION_HYPOTHESIS_CAP = 128


class PartitionWorkflowMixin:
    """Internal workflow methods composed by ``MemoryCondenser``."""

    @staticmethod
    def _active_partition_surface_score(query: str, text: str) -> float:
        query_terms = set(tokenize(query))
        text_terms = set(tokenize(text))
        if not query_terms:
            return 0.0
        return len(query_terms & text_terms) / len(query_terms)

    @staticmethod
    def _first_person_completed_venue_occurrence(text: str) -> bool:
        """Conservative direct-event gate for canonical venue mentions."""

        first_person = re.search(r"\b(?:I|we|my|our)\b", text, re.IGNORECASE)
        completed = re.search(
            r"\b(?:visited|attended|participated|went|saw|took|returned|"
            r"came\s+back|got\s+back)\b",
            text,
            re.IGNORECASE,
        )
        return first_person is not None and completed is not None

    @staticmethod
    def _venue_episode_alignment(
        text: str,
        source_timestamp: str | None,
    ) -> bool | None:
        """Whether a stated venue occurrence belongs to its source episode.

        Immediate relative language is aligned by construction.  An explicit
        month/day is compared with the durable source date; a disagreement is
        a proved retrospective recap.  Rows without either signal remain
        ambiguous and are kept as alternatives rather than promoted to a
        structural primary.
        """

        relative_alignment = re.search(
            r"\b(?:today|tonight|yesterday|this\s+(?:morning|afternoon|evening)|"
            r"just\s+(?:came|got|returned)(?:\s+back)?)\b",
            text,
            re.IGNORECASE,
        ) is not None
        month_numbers = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }
        explicit = re.search(
            r"\b(?P<month>" + "|".join(month_numbers) + r")\s+"
            r"(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(?P<year>\d{4}))?\b",
            text,
            re.IGNORECASE,
        )
        if explicit is None:
            return True if relative_alignment else None
        from memory_condense.search.selectors.coverage_selector import _timestamp_key

        source_value = _timestamp_key(source_timestamp)
        if source_value is None:
            return None
        source_date = datetime.fromtimestamp(source_value)
        month = month_numbers[explicit.group("month").casefold()]
        day = int(explicit.group("day"))
        year = int(explicit.group("year") or source_date.year)
        try:
            event_date = datetime(year, month, day)
        except ValueError:
            return None
        return event_date.date() == source_date.date()

    def _active_partition_timestamps(
        self,
        source_ids: Sequence[str],
    ) -> dict[str, str]:
        timestamps: dict[str, str] = {}
        for start in range(0, len(source_ids), 400):
            metadata = self._transcript.source_metadata(
                list(source_ids[start : start + 400])
            )
            for source_id, text in metadata.items():
                parsed = parse_source_metadata(text)
                if parsed is not None:
                    timestamps[source_id] = parsed[1]
        return timestamps

    def _content_high_watermark(self) -> int:
        """Return the committed chunk generation used by scan snapshots."""

        row = self._db.execute(
            "SELECT COALESCE(MAX(rowid), 0) FROM chunks"
        ).fetchone()
        return int(row[0] if row is not None else 0)

    def _scan_active_partition_frontier(
        self,
        query: str,
        query_embedding: np.ndarray,
        partition_ids: Sequence[str],
        routed_source_ids: Sequence[str],
        *,
        separator: str,
    ) -> tuple[list[RetrievalResult], dict[str, Any]]:
        """Reduce a complete selected-partition row scan to bounded IDs."""

        from memory_condense.search.selectors.coverage_selector import (
            SetQuantifier,
            QwenPrefixCoverageSelector,
            _canonical_answer_object_key,
            _timestamp_key,
            compile_set_program,
        )

        program = compile_set_program(query)
        venue_program = bool(
            program.requires_completeness
            and re.search(
                r"\b(?:museum|museums|gallery|galleries)\b",
                query,
                re.IGNORECASE,
            )
        )
        performance_program = bool(
            program.requires_completeness and is_performance_query(query)
        )
        partition_inventory = self._retriever.source_partition_ids(
            separator=separator,
        )
        selected_partition_ids = list(
            dict.fromkeys(str(value) for value in partition_ids if str(value))
        )
        partition_scope_exhaustive = bool(
            partition_inventory
            and set(selected_partition_ids) == set(partition_inventory)
        )
        base_report: dict[str, Any] = {
            "active_partition_scan_status": "bypassed",
            "active_partition_total": 0,
            "active_partition_inspected": 0,
            "active_partition_exhaustive": None,
            "active_partition_sources_total": len(routed_source_ids),
            "active_partition_sources_inspected": 0,
            "active_partition_structural_rows": 0,
            "active_partition_structural_hypotheses": 0,
            "active_partition_alternative_hypotheses": 0,
            "active_partition_ambiguous_structural_rows": 0,
            "active_partition_recap_conflict_rows": 0,
            "active_partition_performance_multirow_sources": 0,
            "active_partition_role_rejected_rows": 0,
            "active_partition_time_rejected_rows": 0,
            "active_partition_unknown_timestamp_rows": 0,
            "active_partition_candidates_admitted": 0,
            "active_partition_candidates_already_present": 0,
            "active_partition_candidates_replaced": 0,
            "active_partition_candidates_truncated": 0,
            "active_partition_structural_overflow": 0,
            "active_partition_scan_contract": "",
            "active_partition_semantically_complete": False,
            "partition_scope_kind": (
                "global" if partition_scope_exhaustive else "approximate_top_k"
            ),
            "partition_inventory_total": len(partition_inventory),
            "selected_partition_count": len(selected_partition_ids),
            "partition_scope_exhaustive": partition_scope_exhaustive,
            "selected_scope_structurally_complete": False,
            "global_semantic_complete": False,
            "active_partition_scan_elapsed_s": 0.0,
        }
        if not partition_ids or not (venue_program or performance_program):
            return [], base_report

        contract = (
            "canonical_venue_episode_aligned_v1"
            if venue_program
            else "direct_performance_source_occurrence_v1"
        )
        started = time.perf_counter()
        timestamps = self._active_partition_timestamps(routed_source_ids)
        # A source may contain more than one completed occurrence.  Transient
        # venue/performance keys separate those occurrences without becoming
        # stored state, and globally contract exact keyed recaps even when a
        # recap leaked into another routed source.
        primary_by_occurrence: dict[
            tuple[str, str], _ActivePartitionHypothesis
        ] = {}
        alternative_by_occurrence: dict[
            tuple[str, str], _ActivePartitionHypothesis
        ] = {}
        ambiguous_occurrences: set[tuple[str, str]] = set()
        inspected_sources: set[str] = set()
        total_rows = 0
        structural_rows = 0
        role_rejected = 0
        time_rejected = 0
        unknown_timestamp = 0
        ambiguous_rows = 0
        recap_conflicts = 0
        performance_occurrence_counts: Counter[str] = Counter()
        try:
            for row in self._retriever.iter_source_content_rows(
                routed_source_ids,
            ):
                total_rows += 1
                inspected_sources.add(row.source_id)
                canonical_key = (
                    _canonical_answer_object_key(query, row.text)
                    if venue_program
                    else performance_event_key(query, row.text)
                )
                typed_match = (
                    canonical_key is not None
                    and self._first_person_completed_venue_occurrence(row.text)
                    if venue_program
                    else is_direct_past_performance(query, row.text)
                )
                if not typed_match:
                    continue
                if (
                    program.preferred_evidence_role is not None
                    and row.role.casefold() != program.preferred_evidence_role
                ):
                    role_rejected += 1
                    continue
                timestamp = timestamps.get(row.source_id)
                temporal_in_scope = QwenPrefixCoverageSelector._timestamp_in_scope(
                    program,
                    timestamp,
                )
                if temporal_in_scope is False:
                    time_rejected += 1
                    continue
                if temporal_in_scope is None and (
                    program.query_timestamp is not None
                    or program.temporal_window_days is not None
                ):
                    unknown_timestamp += 1
                hypothesis = _ActivePartitionHypothesis(
                    chunk_id=row.chunk_id,
                    source_id=row.source_id,
                    timestamp=timestamp,
                    ordinal=row.ordinal,
                    surface_score=self._active_partition_surface_score(
                        query,
                        row.text,
                    ),
                    identity_key=(
                        str(canonical_key) if canonical_key is not None else None
                    ),
                )
                occurrence_key = (
                    (row.source_id, str(canonical_key))
                    if venue_program
                    else (
                        ("performance", str(canonical_key))
                        if canonical_key is not None
                        else (row.source_id, row.chunk_id)
                    )
                )
                if venue_program:
                    alignment = self._venue_episode_alignment(row.text, timestamp)
                    if alignment is False:
                        recap_conflicts += 1
                        alternative_by_occurrence.setdefault(
                            occurrence_key,
                            hypothesis,
                        )
                        continue
                    if alignment is None:
                        ambiguous_rows += 1
                        ambiguous_occurrences.add(occurrence_key)
                        alternative_by_occurrence.setdefault(
                            occurrence_key,
                            hypothesis,
                        )
                        continue
                else:
                    performance_occurrence_counts[row.source_id] += 1
                    if canonical_key is None:
                        ambiguous_rows += 1
                        ambiguous_occurrences.add(occurrence_key)
                        alternative_by_occurrence.setdefault(
                            occurrence_key,
                            hypothesis,
                        )
                        continue
                structural_rows += 1
                existing = primary_by_occurrence.get(occurrence_key)
                if existing is None:
                    primary_by_occurrence[occurrence_key] = hypothesis
                else:
                    existing_timestamp = _timestamp_key(existing.timestamp)
                    hypothesis_timestamp = _timestamp_key(hypothesis.timestamp)
                    existing_order = (
                        existing_timestamp is None,
                        existing_timestamp or 0.0,
                        existing.ordinal,
                        existing.chunk_id,
                    )
                    hypothesis_order = (
                        hypothesis_timestamp is None,
                        hypothesis_timestamp or 0.0,
                        hypothesis.ordinal,
                        hypothesis.chunk_id,
                    )
                    if hypothesis_order < existing_order:
                        primary_by_occurrence[occurrence_key] = hypothesis
        except Exception as exc:
            base_report.update(
                {
                    "active_partition_scan_status": "failed",
                    "active_partition_total": total_rows,
                    "active_partition_inspected": total_rows,
                    "active_partition_exhaustive": False,
                    "active_partition_sources_inspected": len(inspected_sources),
                    "active_partition_scan_contract": contract,
                    "active_partition_scan_error": type(exc).__name__,
                    "active_partition_scan_elapsed_s": (
                        time.perf_counter() - started
                    ),
                }
            )
            return [], base_report

        primary_occurrences = list(primary_by_occurrence.values())
        alternative_occurrences = [
            hypothesis
            for occurrence_key, hypothesis in alternative_by_occurrence.items()
            if occurrence_key not in primary_by_occurrence
        ]
        if venue_program:
            def venue_occurrence_order(
                hypothesis: _ActivePartitionHypothesis,
            ) -> tuple[bool, float, int, str]:
                timestamp_key = _timestamp_key(hypothesis.timestamp)
                return (
                    timestamp_key is None,
                    timestamp_key or 0.0,
                    hypothesis.ordinal,
                    hypothesis.chunk_id,
                )

            primary_by_identity: dict[str, _ActivePartitionHypothesis] = {}
            for hypothesis in sorted(
                primary_occurrences,
                key=venue_occurrence_order,
            ):
                if hypothesis.identity_key is not None:
                    primary_by_identity.setdefault(
                        hypothesis.identity_key,
                        hypothesis,
                    )
            primary_hypotheses = list(primary_by_identity.values())
            primary_identity_keys = set(primary_by_identity)
            alternative_by_identity: dict[
                str, _ActivePartitionHypothesis
            ] = {}
            for hypothesis in sorted(
                alternative_occurrences,
                key=venue_occurrence_order,
            ):
                identity_key = hypothesis.identity_key
                if (
                    identity_key is not None
                    and identity_key not in primary_identity_keys
                ):
                    alternative_by_identity.setdefault(identity_key, hypothesis)
            alternatives = list(alternative_by_identity.values())
            ambiguous_identity_keys = {
                identity_key
                for _source_id, identity_key in ambiguous_occurrences
                if identity_key not in primary_identity_keys
            }
            ambiguous_hypotheses = len(ambiguous_identity_keys)
        else:
            primary_hypotheses = primary_occurrences
            alternatives = alternative_occurrences
            ambiguous_hypotheses = len(ambiguous_occurrences)
        hypothesis_count = len(primary_hypotheses)
        overflow = (
            max(0, hypothesis_count - int(program.cardinality or 0))
            if program.quantifier is SetQuantifier.FIXED
            else 0
        )
        def hypothesis_rank(
            item: _ActivePartitionHypothesis,
        ) -> tuple[float, int, str]:
            return (
                -item.surface_score,
                item.ordinal,
                item.chunk_id,
            )

        pre_ranked = [
            *sorted(primary_hypotheses, key=hypothesis_rank),
            *sorted(alternatives, key=hypothesis_rank),
        ]
        cap_truncated = max(
            0,
            len(pre_ranked) - _ACTIVE_PARTITION_HYPOTHESIS_CAP,
        )
        retained = pre_ranked[:_ACTIVE_PARTITION_HYPOTHESIS_CAP]
        dense_scores = self._retriever.cosine_scores(
            query_embedding,
            [item.chunk_id for item in retained],
        )
        primary_ids = {item.chunk_id for item in primary_hypotheses}
        retained.sort(
            key=lambda item: (
                item.chunk_id not in primary_ids,
                -dense_scores.get(item.chunk_id, float("-inf")),
                *hypothesis_rank(item),
            )
        )
        candidates: list[RetrievalResult] = []
        for hypothesis in retained:
            score = dense_scores.get(hypothesis.chunk_id, 0.0)
            hydrated = self._retriever.hydrate_chunk(
                hypothesis.chunk_id,
                score=score,
                route=(
                    "active_partition_structural"
                    if hypothesis.chunk_id in primary_ids
                    else "active_partition_alternative"
                ),
                anchor_chunk_id=hypothesis.chunk_id,
            )
            if hydrated is None:
                cap_truncated += 1
                continue
            candidates.append(
                hydrated.model_copy(
                    update={
                        "memory_source_id": hypothesis.source_id,
                        "source_heat": max(0.0, score),
                    }
                )
            )

        fixed_complete = bool(
            program.quantifier is SetQuantifier.FIXED
            and program.cardinality is not None
            and hypothesis_count == program.cardinality
        )
        exhaustive_set_complete = bool(
            performance_program
            and program.quantifier in {SetQuantifier.ALL, SetQuantifier.COUNT}
        )
        performance_multirow_sources = sum(
            count > 1 for count in performance_occurrence_counts.values()
        )
        semantic_complete = bool(
            cap_truncated == 0
            and unknown_timestamp == 0
            and ambiguous_hypotheses == 0
            and (fixed_complete or exhaustive_set_complete)
        )
        base_report.update(
            {
                "active_partition_scan_status": "applied",
                "active_partition_total": total_rows,
                "active_partition_inspected": total_rows,
                "active_partition_exhaustive": True,
                "active_partition_sources_inspected": len(routed_source_ids),
                "active_partition_structural_rows": structural_rows,
                "active_partition_structural_hypotheses": hypothesis_count,
                "active_partition_alternative_hypotheses": len(alternatives),
                "active_partition_ambiguous_structural_rows": ambiguous_rows,
                "active_partition_recap_conflict_rows": recap_conflicts,
                "active_partition_performance_multirow_sources": (
                    performance_multirow_sources
                ),
                "active_partition_role_rejected_rows": role_rejected,
                "active_partition_time_rejected_rows": time_rejected,
                "active_partition_unknown_timestamp_rows": unknown_timestamp,
                "active_partition_candidates_truncated": cap_truncated,
                "active_partition_structural_overflow": overflow,
                "active_partition_scan_contract": contract,
                "active_partition_semantically_complete": semantic_complete,
                "selected_scope_structurally_complete": semantic_complete,
                "global_semantic_complete": bool(
                    partition_scope_exhaustive and semantic_complete
                ),
                "active_partition_scan_elapsed_s": time.perf_counter() - started,
            }
        )
        return candidates, base_report

    @staticmethod
    def _admit_active_partition_candidates(
        baseline: Sequence[RetrievalResult],
        candidates: Sequence[RetrievalResult],
        *,
        anchor_chunk_ids: set[str],
        semantic_complete: bool = True,
    ) -> tuple[list[RetrievalResult], dict[str, Any]]:
        """Force typed hypotheses into a fixed-count frontier.

        A proved-complete typed scan may consume the full frontier.  An
        ambiguous, overflowing, or truncated scan remains fail-open: direct
        anchors are immutable and at least one quarter of the baseline is
        reserved before bounded typed additions are considered.
        """

        capacity = len(baseline)
        baseline_ids = {result.chunk.chunk_id for result in baseline}
        unique_candidates: list[RetrievalResult] = []
        candidate_ids: set[str] = set()
        for result in candidates:
            chunk_id = result.chunk.chunk_id
            if chunk_id in candidate_ids:
                continue
            candidate_ids.add(chunk_id)
            unique_candidates.append(result)

        protected_ids: set[str] = set()
        if not semantic_complete and capacity:
            protected_ids.update(baseline_ids & anchor_chunk_ids)
            reserve = max(1, math.ceil(capacity / 4))
            for result in baseline:
                if len(protected_ids) >= reserve:
                    break
                protected_ids.add(result.chunk.chunk_id)
        evictable_ids = baseline_ids - protected_ids
        existing_candidate_ids = baseline_ids & candidate_ids
        new_candidate_budget = (
            capacity
            if semantic_complete
            else len(evictable_ids - existing_candidate_ids)
        )
        retained: list[RetrievalResult] = []
        admitted_count = 0
        for result in unique_candidates:
            chunk_id = result.chunk.chunk_id
            if chunk_id in baseline_ids:
                retained.append(result)
            elif admitted_count < new_candidate_budget:
                retained.append(result)
                admitted_count += 1
            if len(retained) >= capacity:
                break
        capacity_truncated = max(0, len(unique_candidates) - len(retained))
        retained_ids = {result.chunk.chunk_id for result in retained}
        already_present = len(retained_ids & baseline_ids)
        admitted = len(retained_ids - baseline_ids)
        ordinary = [
            (index, result)
            for index, result in enumerate(baseline)
            if result.chunk.chunk_id not in retained_ids
        ]

        def eviction_key(
            item: tuple[int, RetrievalResult],
        ) -> tuple[int, float, int, str]:
            index, result = item
            route = (result.route or "").casefold()
            if route == "hsc_contraction":
                tier = 0
            elif route == "live_consolidation":
                tier = 1
            elif "neighbor" in route:
                tier = 2
            elif result.chunk.chunk_id not in anchor_chunk_ids:
                tier = 3
            else:
                tier = 4
            return tier, float(result.score), -index, result.chunk.chunk_id

        evict_count = min(admitted, len(ordinary))
        evicted_ids = {
            result.chunk.chunk_id
            for _index, result in sorted(
                (
                    item
                    for item in ordinary
                    if item[1].chunk.chunk_id not in protected_ids
                ),
                key=eviction_key,
            )[:evict_count]
        }
        survivors = [
            result
            for _index, result in ordinary
            if result.chunk.chunk_id not in evicted_ids
        ]
        output = [*retained, *survivors]
        if len(output) > capacity:
            output = output[:capacity]
        return output, {
            "active_partition_candidates_admitted": admitted,
            "active_partition_candidates_already_present": already_present,
            "active_partition_candidates_replaced": len(evicted_ids),
            "active_partition_candidates_truncated": capacity_truncated,
            "active_partition_baseline_protected": len(protected_ids),
            "active_partition_candidate_count_before": capacity,
            "active_partition_candidate_count_after": len(output),
        }


__all__ = ["PartitionWorkflowMixin"]
