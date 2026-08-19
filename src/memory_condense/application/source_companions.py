"""Stateful source companion discovery and metadata hydration workflows."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np

from memory_condense.application.query_routing import _retrieval_source_id
from memory_condense.domain.schemas import RetrievalResult
from memory_condense.persistence.transcript_store import parse_source_metadata
from memory_condense.search.packing.performance_events import (
    is_direct_past_performance,
    is_performance_query,
)


_SOURCE_COMPANION_MAX_PER_SOURCE = 4
_SOURCE_CANONICAL_COMPANIONS_PER_SOURCE = 4
_SOURCE_PERFORMANCE_COMPANIONS_PER_SOURCE = 1


def default_source_companion_report(**overrides: Any) -> dict[str, Any]:
    """Return the one authoritative ``last_source_companion_report`` shape.

    Every producer starts from these defaults so the key set cannot drift
    between the build-context preamble, the empty-report path, and the full
    companion pass.  Unknown keys are rejected to catch typos at the caller.
    """
    report: dict[str, Any] = {
        "requested_sources": [],
        "hydrated_sources": [],
        "refreshed_sources": [],
        "already_present_sources": [],
        "orphan_sources": [],
        "orphan_count": 0,
        "direct_date_retained": 0,
        "candidate_count_before": 0,
        "candidate_count_after": 0,
        "max_candidates_per_source": 1,
        "companion_candidate_count": 0,
        "selector_used": False,
        "selector_fallback_sources": [],
        "selector_fallback_reason": "",
        "semantic_selector_report": {},
        "selected_chunk_ids": {},
        "refresh_all_activated_sources": False,
        "choice_diagnostics": [],
    }
    unknown = set(overrides) - set(report)
    if unknown:
        raise ValueError(
            f"unknown source companion report fields: {sorted(unknown)}"
        )
    report.update(overrides)
    return report


class SourceCompanionWorkflowMixin:
    """Internal workflow methods composed by ``MemoryCondenser``."""

    def _canonical_source_companion_candidates(
        self,
        query: str,
        source_ids: Sequence[str],
        *,
        preferred_role: str | None,
        excluded_chunk_ids: Sequence[str],
        source_scores: Mapping[str, float],
        anchor_chunk_ids: Mapping[str, str],
    ) -> list[RetrievalResult]:
        """Stream a bounded, query-head-specific raw-source shortlist.

        Dense source-local rank can miss the first answer-bearing turn when a
        later recap repeats more query words.  For the one conservative
        identity relation currently available (museum/gallery venue), retain
        the earliest preferred-role row for each unambiguous canonical key.
        The scan retains only chunk IDs and keys for at most four rows/source;
        raw text, keys, and activations are not persisted.
        """

        if not source_ids or re.search(
            r"\b(?:museum|museums|gallery|galleries)\b",
            query,
            re.IGNORECASE,
        ) is None:
            return []
        from memory_condense.search.selectors.coverage_selector import (
            _canonical_answer_object_key,
        )

        selected_sources = list(
            dict.fromkeys(str(source_id) for source_id in source_ids if source_id)
        )
        placeholders = ",".join("?" for _ in selected_sources)
        source_expr = "COALESCE(t.source_id, t.turn_id)"
        excluded = {str(chunk_id) for chunk_id in excluded_chunk_ids}
        selected_ids: dict[str, list[str]] = {
            source_id: [] for source_id in selected_sources
        }
        seen_keys: dict[str, set[str]] = {
            source_id: set() for source_id in selected_sources
        }
        rows = self._db.execute(
            "SELECT c.chunk_id, c.text, " + source_expr + ", t.role "
            "FROM chunks AS c JOIN turns AS t ON t.turn_id = c.turn_id "
            f"WHERE {source_expr} IN ({placeholders}) "
            "ORDER BY t.ordinal, c.rowid",
            tuple(selected_sources),
        )
        for chunk_id, text, source_id, role in rows:
            source_key = str(source_id)
            if (
                source_key not in selected_ids
                or str(chunk_id) in excluded
                or len(selected_ids[source_key])
                >= _SOURCE_CANONICAL_COMPANIONS_PER_SOURCE
                or (
                    preferred_role is not None
                    and str(role).casefold() != preferred_role
                )
            ):
                continue
            answer_key = _canonical_answer_object_key(query, str(text))
            if answer_key is None or answer_key in seen_keys[source_key]:
                continue
            seen_keys[source_key].add(answer_key)
            selected_ids[source_key].append(str(chunk_id))

        results: list[RetrievalResult] = []
        for source_id in selected_sources:
            for chunk_id in selected_ids[source_id]:
                hydrated = self._retriever.hydrate_chunk(
                    chunk_id,
                    score=float(source_scores.get(source_id, 0.0)),
                    route="source_canonical_companion",
                    anchor_chunk_id=anchor_chunk_ids.get(source_id),
                )
                if hydrated is not None:
                    results.append(
                        hydrated.model_copy(
                            update={
                                "memory_source_id": source_id,
                                "source_heat": max(
                                    0.0,
                                    float(source_scores.get(source_id, 0.0)),
                                ),
                            }
                        )
                    )
        return results

    def _performance_source_companion_candidates(
        self,
        query: str,
        source_ids: Sequence[str],
        *,
        preferred_role: str | None,
        excluded_chunk_ids: Sequence[str],
        source_scores: Mapping[str, float],
        anchor_chunk_ids: Mapping[str, str],
    ) -> list[RetrievalResult]:
        """Retain the first direct performance occurrence in each source.

        Source-local dense rank often favors generic playlists, future plans,
        or later summaries over a short artist-and-venue fact.  A single
        streaming pass over already activated sources supplies that missing
        primary event without growing the route union or retaining row text.
        """

        if not source_ids or not is_performance_query(query):
            return []
        selected_sources = list(
            dict.fromkeys(str(source_id) for source_id in source_ids if source_id)
        )
        placeholders = ",".join("?" for _ in selected_sources)
        source_expr = "COALESCE(t.source_id, t.turn_id)"
        excluded = {str(chunk_id) for chunk_id in excluded_chunk_ids}
        selected_ids: dict[str, str] = {}
        rows = self._db.execute(
            "SELECT c.chunk_id, c.text, " + source_expr + ", t.role "
            "FROM chunks AS c JOIN turns AS t ON t.turn_id = c.turn_id "
            f"WHERE {source_expr} IN ({placeholders}) "
            "ORDER BY t.ordinal, c.rowid",
            tuple(selected_sources),
        )
        for chunk_id, text, source_id, role in rows:
            source_key = str(source_id)
            if (
                source_key in selected_ids
                or source_key not in selected_sources
                or str(chunk_id) in excluded
                or (
                    preferred_role is not None
                    and str(role).casefold() != preferred_role
                )
                or not is_direct_past_performance(query, str(text))
            ):
                continue
            selected_ids[source_key] = str(chunk_id)

        results: list[RetrievalResult] = []
        for source_id in selected_sources:
            chunk_id = selected_ids.get(source_id)
            if chunk_id is None:
                continue
            hydrated = self._retriever.hydrate_chunk(
                chunk_id,
                score=float(source_scores.get(source_id, 0.0)),
                route="source_performance_companion",
                anchor_chunk_id=anchor_chunk_ids.get(source_id),
            )
            if hydrated is not None:
                results.append(
                    hydrated.model_copy(
                        update={
                            "memory_source_id": source_id,
                            "source_heat": max(
                                0.0,
                                float(source_scores.get(source_id, 0.0)),
                            ),
                        }
                    )
                )
        return results

    def _hydrate_source_metadata_companions(
        self,
        user_text: str,
        expansions: Sequence[RetrievalResult],
        query_embedding: np.ndarray | None,
    ) -> tuple[list[RetrievalResult], set[str]]:
        """Ensure routed sources carry one bounded, query-selected raw payload.

        Metadata-only routes retain the historical one-row hydration.  For an
        explicit complete-set query, the same bounded source-local chooser is
        run for every source already activated in the final route union.  A
        selected raw row that is absent replaces one deterministic low-value
        row from its own source, so neither source reachability nor candidate
        count can grow.  No answer labels or benchmark categories are read.
        """

        from memory_condense.search.selectors.coverage_selector import compile_set_program

        output = list(expansions)
        source_rows: dict[str, list[tuple[int, RetrievalResult]]] = {}
        metadata_rows: dict[str, list[tuple[int, RetrievalResult]]] = {}
        content_sources: set[str] = set()
        metadata_chunk_ids: list[str] = []
        source_order: list[str] = []
        for index, result in enumerate(output):
            source_id = _retrieval_source_id(result)
            if source_id not in source_rows:
                source_order.append(source_id)
                source_rows[source_id] = []
            source_rows[source_id].append((index, result))
            if parse_source_metadata(result.chunk.text) is None:
                content_sources.add(source_id)
            else:
                metadata_rows.setdefault(source_id, []).append((index, result))
                metadata_chunk_ids.append(result.chunk.chunk_id)

        choose_companions = getattr(
            self._context_candidate_selector,
            "select_source_companions",
            None,
        )
        program = compile_set_program(user_text)
        refresh_all_sources = bool(
            program.requires_completeness and callable(choose_companions)
        )
        requested = (
            list(source_order)
            if refresh_all_sources
            else [
                source_id
                for source_id in source_order
                if source_id in metadata_rows and source_id not in content_sources
            ]
        )

        def empty_report(
            *,
            fallback_reason: str = "",
            fallback_sources: Sequence[str] = (),
        ) -> dict[str, Any]:
            return default_source_companion_report(
                requested_sources=requested,
                candidate_count_before=len(output),
                candidate_count_after=len(output),
                selector_fallback_sources=list(fallback_sources),
                selector_fallback_reason=fallback_reason,
                refresh_all_activated_sources=refresh_all_sources,
            )

        if not requested:
            self.last_source_companion_report = empty_report()
            return output, set()

        vector = (
            np.asarray(query_embedding, dtype=np.float32)
            if query_embedding is not None
            else np.asarray(self._embedder.embed_query(user_text), dtype=np.float32)
        )
        source_scores = {
            source_id: max(
                float(result.score) for _index, result in source_rows[source_id]
            )
            for source_id in requested
        }
        anchor_chunk_ids = {
            source_id: (
                source_rows[source_id][0][1].anchor_chunk_id
                or source_rows[source_id][0][1].chunk.chunk_id
            )
            for source_id in requested
        }
        max_per_source = (
            _SOURCE_COMPANION_MAX_PER_SOURCE
            if callable(choose_companions)
            else 1
        )
        try:
            hybrid_companions = self._retriever.hybrid_query_source_companions(
                user_text,
                vector,
                requested,
                metadata_chunk_ids=metadata_chunk_ids,
                max_sources=len(requested),
                max_per_source=max_per_source,
                candidates_per_source=64,
                source_scores=source_scores,
                anchor_chunk_ids=anchor_chunk_ids,
            )
            canonical_companions = (
                self._canonical_source_companion_candidates(
                    user_text,
                    requested,
                    preferred_role=program.preferred_evidence_role,
                    excluded_chunk_ids=metadata_chunk_ids,
                    source_scores=source_scores,
                    anchor_chunk_ids=anchor_chunk_ids,
                )
                if refresh_all_sources
                else []
            )
            performance_companions = (
                self._performance_source_companion_candidates(
                    user_text,
                    requested,
                    preferred_role=program.preferred_evidence_role,
                    excluded_chunk_ids=metadata_chunk_ids,
                    source_scores=source_scores,
                    anchor_chunk_ids=anchor_chunk_ids,
                )
                if refresh_all_sources
                else []
            )
        except Exception as exc:
            if bool(getattr(self._context_candidate_selector, "strict", False)):
                raise
            metadata_orphans = [
                source_id
                for source_id in requested
                if source_id not in content_sources
            ]
            report = empty_report(
                fallback_reason=type(exc).__name__,
                fallback_sources=requested,
            )
            report["orphan_sources"] = metadata_orphans
            report["orphan_count"] = len(metadata_orphans)
            self.last_source_companion_report = report
            return output, set(metadata_orphans)

        candidates_by_source: dict[str, list[RetrievalResult]] = {
            source_id: [] for source_id in requested
        }
        canonical_by_source: dict[str, list[RetrievalResult]] = {
            source_id: [] for source_id in requested
        }
        for result in canonical_companions:
            source_id = _retrieval_source_id(result)
            if source_id in canonical_by_source:
                canonical_by_source[source_id].append(result)
        performance_by_source: dict[str, list[RetrievalResult]] = {
            source_id: [] for source_id in requested
        }
        for result in performance_companions:
            source_id = _retrieval_source_id(result)
            if source_id in performance_by_source:
                performance_by_source[source_id].append(result)
        canonical_primary_rule = bool(
            refresh_all_sources
            and re.search(
                r"\b(?:museum|museums|gallery|galleries)\b",
                user_text,
                re.IGNORECASE,
            )
            and (
                program.quantifier.value == "fixed_cardinality"
                or program.ordering.value != "none"
            )
        )
        # In an ordered/fixed venue set, the earliest preferred-role canonical
        # occurrence is the source's primary event anchor.  Do not let a later
        # high-overlap recap with no venue identity compete it away.  Sources
        # without a conservative canonical parse keep the generic top-N path.
        canonical_primary_sources = {
            source_id
            for source_id, candidates in canonical_by_source.items()
            if canonical_primary_rule and candidates
        }
        performance_primary_rule = bool(
            refresh_all_sources
            and is_performance_query(user_text)
            and (
                program.quantifier.value in {"all", "fixed_cardinality"}
                or program.ordering.value != "none"
            )
        )
        performance_primary_sources = {
            source_id
            for source_id, candidates in performance_by_source.items()
            if performance_primary_rule and candidates
        }
        primary_sources = canonical_primary_sources | performance_primary_sources
        for source_id in canonical_primary_sources:
            candidates_by_source[source_id].append(
                canonical_by_source[source_id][0]
            )
        for source_id in performance_primary_sources - canonical_primary_sources:
            candidates_by_source[source_id].append(
                performance_by_source[source_id][0]
            )
        for result in [
            *canonical_companions,
            *performance_companions,
            *hybrid_companions,
        ]:
            source_id = _retrieval_source_id(result)
            if source_id not in candidates_by_source:
                continue
            if source_id in primary_sources:
                continue
            if any(
                prior.chunk.chunk_id == result.chunk.chunk_id
                for prior in candidates_by_source[source_id]
            ):
                continue
            candidates_by_source[source_id].append(result)

        selectable = {
            source_id: tuple(candidates)
            for source_id, candidates in candidates_by_source.items()
            if candidates
            and (refresh_all_sources or len(candidates) > 1)
        }
        semantic_choices: dict[str, RetrievalResult] = {}
        selector_used = False
        selector_fallback_sources: list[str] = []
        selector_fallback_reasons: list[str] = []
        semantic_selector_report: dict[str, Any] = {}
        selector_reports: list[dict[str, Any]] = []

        def dump_report(raw_report: Any) -> dict[str, Any]:
            dumped = getattr(raw_report, "model_dump", None)
            if callable(dumped):
                return dict(dumped())
            if isinstance(raw_report, Mapping):
                return dict(raw_report)
            return {}

        def nested_value(report: Mapping[str, Any], key: str) -> Any:
            if key in report:
                return report[key]
            for nested_key in ("provider_report", "score_report"):
                nested = report.get(nested_key)
                if isinstance(nested, Mapping):
                    value = nested_value(nested, key)
                    if value is not None:
                        return value
            return None

        selection_batches: list[dict[str, tuple[RetrievalResult, ...]]] = []
        if selectable:
            if refresh_all_sources:
                current: dict[str, tuple[RetrievalResult, ...]] = {}
                current_count = 0
                for source_id, candidates in selectable.items():
                    if current and current_count + len(candidates) > 128:
                        selection_batches.append(current)
                        current = {}
                        current_count = 0
                    current[source_id] = candidates
                    current_count += len(candidates)
                if current:
                    selection_batches.append(current)
            else:
                selection_batches.append(selectable)

        for batch in selection_batches:
            selector_used = True
            batch_unavailable = False
            try:
                proposed = choose_companions(user_text, batch)
            except Exception as exc:
                if bool(getattr(self._context_candidate_selector, "strict", False)):
                    raise
                proposed = {}
                selector_fallback_sources.extend(
                    source_id
                    for source_id in batch
                    if source_id not in selector_fallback_sources
                )
                selector_fallback_reasons.append(type(exc).__name__)
                batch_unavailable = True
            raw_report = getattr(
                self._context_candidate_selector,
                "last_source_companion_report",
                None,
            )
            batch_report = dump_report(raw_report)
            if batch_report:
                selector_reports.append(batch_report)
            if not isinstance(proposed, Mapping):
                proposed = {}
                selector_fallback_sources.extend(
                    source_id
                    for source_id in batch
                    if source_id not in selector_fallback_sources
                )
                selector_fallback_reasons.append("invalid_selection_mapping")
                batch_unavailable = True

            selected_ids = nested_value(batch_report, "selected_chunk_ids")
            selected_ids = selected_ids if isinstance(selected_ids, Mapping) else {}
            membership_scores = nested_value(
                batch_report,
                "selected_membership_scores",
            )
            membership_scores = (
                membership_scores
                if isinstance(membership_scores, Mapping)
                else {}
            )
            input_count = int(nested_value(batch_report, "input_candidates") or 0)
            inspected_count = int(
                nested_value(batch_report, "inspected_candidates") or 0
            )
            fallback_reason = str(
                nested_value(batch_report, "fallback_reason") or ""
            )
            all_inspected = bool(
                input_count >= sum(len(rows) for rows in batch.values())
                and inspected_count >= input_count
                and not fallback_reason
            )
            for source_id, candidates in batch.items():
                if batch_unavailable:
                    continue
                proposed_result = proposed.get(source_id)
                # Exact object provenance matters: a provider may reorder the
                # supplied raw rows, but it may not fabricate a replacement
                # that merely reuses one of their IDs.
                selected = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate is proposed_result
                    ),
                    None,
                )
                inspected_winner = bool(
                    source_id in membership_scores
                    or (
                        all_inspected
                        and selected is not None
                        and str(selected_ids.get(source_id, ""))
                        == selected.chunk.chunk_id
                    )
                )
                if selected is None or (refresh_all_sources and not inspected_winner):
                    if source_id not in selector_fallback_sources:
                        selector_fallback_sources.append(source_id)
                    selector_fallback_reasons.append(
                        "invalid_selection"
                        if selected is None
                        else "uninspected_selection"
                    )
                    continue
                semantic_choices[source_id] = selected

        if selector_reports:
            if refresh_all_sources:
                semantic_selector_report = {
                    "batch_count": len(selector_reports),
                    "input_sources": sum(
                        int(nested_value(report, "input_sources") or 0)
                        for report in selector_reports
                    ),
                    "input_candidates": sum(
                        int(nested_value(report, "input_candidates") or 0)
                        for report in selector_reports
                    ),
                    "inspected_candidates": sum(
                        int(nested_value(report, "inspected_candidates") or 0)
                        for report in selector_reports
                    ),
                    "selected_chunk_ids": {
                        source_id: result.chunk.chunk_id
                        for source_id, result in semantic_choices.items()
                    },
                    "retained_transformer_state_bytes": max(
                        (
                            int(
                                nested_value(
                                    report,
                                    "retained_transformer_state_bytes",
                                )
                                or 0
                            )
                            for report in selector_reports
                        ),
                        default=0,
                    ),
                    "fallback_reasons": list(
                        dict.fromkeys(
                            reason
                            for reason in selector_fallback_reasons
                            if reason
                        )
                    ),
                }
            else:
                semantic_selector_report = selector_reports[-1]

        companion_by_source: dict[str, RetrievalResult] = {}
        choice_diagnostics: list[dict[str, Any]] = []
        for source_id in requested:
            candidates = candidates_by_source[source_id]
            if not candidates:
                continue
            companion = semantic_choices.get(source_id)
            if companion is None and not refresh_all_sources:
                companion = candidates[0]
            if companion is None:
                continue
            companion_by_source[source_id] = companion
            choice_diagnostics.append(
                {
                    "source_id": source_id,
                    "candidate_count": len(candidates),
                    "candidate_chunk_ids": [
                        candidate.chunk.chunk_id for candidate in candidates
                    ],
                    "selected_chunk_id": companion.chunk.chunk_id,
                    "selected_local_rank": next(
                        rank
                        for rank, candidate in enumerate(candidates, start=1)
                        if candidate is companion
                    ),
                    "selected_by": (
                        "semantic"
                        if source_id in semantic_choices
                        else "retrieval"
                    ),
                }
            )

        hydrated_sources: list[str] = []
        refreshed_sources: list[str] = []
        already_present_sources: list[str] = []
        active_partition_protected_sources: list[str] = []
        for source_id in requested:
            companion = companion_by_source.get(source_id)
            if companion is None:
                continue
            if any(
                result.chunk.chunk_id == companion.chunk.chunk_id
                for _index, result in source_rows[source_id]
            ):
                already_present_sources.append(source_id)
                continue

            def replacement_key(
                row: tuple[int, RetrievalResult],
            ) -> tuple[int, float, int]:
                index, activation = row
                if parse_source_metadata(activation.chunk.text) is not None:
                    tier = 0
                    # Synthetic timestamps carry no answer payload; preserve
                    # the first routed anchor rather than letting an arbitrary
                    # score difference between duplicate metadata rows choose
                    # provenance.
                    value = 0.0
                    tie_break = index
                else:
                    role = (
                        activation.turn.role.casefold()
                        if activation.turn is not None
                        else ""
                    )
                    route = (activation.route or "").casefold()
                    tier = (
                        1
                        if (
                            (
                                program.preferred_evidence_role is not None
                                and role != program.preferred_evidence_role
                            )
                            or "support" in route
                        )
                        else 2
                    )
                    value = float(activation.score)
                    tie_break = index
                return tier, value, tie_break

            replaceable_rows = [
                row
                for row in source_rows[source_id]
                if not str(row[1].route or "").casefold().startswith(
                    "active_partition_"
                )
            ]
            if not replaceable_rows:
                active_partition_protected_sources.append(source_id)
                continue
            index, activation = min(
                replaceable_rows,
                key=replacement_key,
            )
            activation_route = str(activation.route or "")
            copied_route = (
                companion.route
                if activation_route.casefold().startswith("active_partition_")
                else activation.route
            )
            output[index] = companion.model_copy(
                update={
                    "score": float(activation.score),
                    "route": copied_route,
                    "anchor_chunk_id": (
                        activation.anchor_chunk_id or activation.chunk.chunk_id
                    ),
                    "memory_source_id": source_id,
                    "source_heat": activation.source_heat,
                    "source_token_budget": activation.source_token_budget,
                }
            )
            hydrated_sources.append(source_id)
            if refresh_all_sources:
                refreshed_sources.append(source_id)

        orphan_sources = [
            source_id
            for source_id in requested
            if source_id not in content_sources
            and source_id not in companion_by_source
        ]
        selector_fallback_reason = ";".join(
            dict.fromkeys(
                reason for reason in selector_fallback_reasons if reason
            )
        )
        companion_report = default_source_companion_report(
            requested_sources=requested,
            hydrated_sources=hydrated_sources,
            refreshed_sources=refreshed_sources,
            already_present_sources=already_present_sources,
            orphan_sources=orphan_sources,
            orphan_count=len(orphan_sources),
            candidate_count_before=len(expansions),
            candidate_count_after=len(output),
            max_candidates_per_source=(
                max_per_source
                + (
                    _SOURCE_CANONICAL_COMPANIONS_PER_SOURCE
                    if canonical_companions
                    else 0
                )
                + (
                    _SOURCE_PERFORMANCE_COMPANIONS_PER_SOURCE
                    if performance_companions
                    else 0
                )
            ),
            companion_candidate_count=sum(
                len(candidates) for candidates in candidates_by_source.values()
            ),
            selector_used=selector_used,
            selector_fallback_sources=selector_fallback_sources,
            selector_fallback_reason=selector_fallback_reason,
            semantic_selector_report=semantic_selector_report,
            selected_chunk_ids={
                source_id: result.chunk.chunk_id
                for source_id, result in companion_by_source.items()
            },
            refresh_all_activated_sources=refresh_all_sources,
            choice_diagnostics=choice_diagnostics,
        )
        if active_partition_protected_sources:
            companion_report["active_partition_protected_sources"] = (
                active_partition_protected_sources
            )
        self.last_source_companion_report = companion_report
        return output, set(orphan_sources)

__all__ = [
    "SourceCompanionWorkflowMixin",
    "default_source_companion_report",
]
