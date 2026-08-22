"""Stateful shell for the non-generative Qwen prefix coverage selector."""

from __future__ import annotations

import gc
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np

from memory_condense.domain.schemas import RetrievalResult
from memory_condense.search.selectors.coverage_models import (
    CoverageScoreProvider,
    CoverageSelectionReport,
)
from memory_condense.search.selectors.evidence_features import (
    _energy_softmax,
    _timestamp_key,
)
from memory_condense.search.selectors.prefix_models import (
    _PrefixAssignment,
    _PrefixEventCluster,
)
from memory_condense.search.selectors.prefix_pipeline import select_prefix_coverage
from memory_condense.search.selectors.set_program import SetProgram, compile_set_program

class QwenPrefixCoverageSelector:
    """Non-generative coverage loop over a bounded Qwen layer prefix.

    The active frontier is streamed through independent QK/OV rows in bounded
    batches.  Candidate vectors exist only for this call.  A deterministic
    energy model combines transient semantic logits (when supplied), QK, OV,
    surface value evidence, and source-time metadata into explicit
    EXISTING/NEW/NULL scores.  The normalized scores are *not calibrated
    probabilities*; the report and trace label them accordingly.
    """

    requires_baseline_ranking = True
    # Coverage/posterior decisions require the complete routed union. A rate-
    # distortion filter may run after grouping, but must not erase a possible
    # event before the selector can reserve its representative.
    requires_complete_frontier = True

    @staticmethod
    def requires_complete_frontier_for(query: str) -> bool:
        """Only set/temporal reducers need admission before rate filtering."""

        return compile_set_program(query).requires_completeness

    def __init__(
        self,
        linker: Any,
        *,
        score_provider: CoverageScoreProvider | None = None,
        candidate_pool: int = 64,
        candidate_tokens: int = 64,
        query_tokens: int = 96,
        merge_similarity: float = 0.985,
        same_source_merge_similarity: float = 0.90,
        posterior_temperature: float = 0.08,
        null_threshold: float = 0.90,
        credible_member_threshold: float = 0.20,
        explicit_membership_threshold: float = 0.50,
        uncertainty_entropy: float = 0.95,
        allow_selected_scope_fixed_k_closure: bool = False,
        strict: bool = False,
    ) -> None:
        if candidate_pool < 1:
            raise ValueError("candidate_pool must be positive")
        if min(candidate_tokens, query_tokens) < 1:
            raise ValueError("token caps must be positive")
        if not 0.0 <= same_source_merge_similarity <= merge_similarity <= 1.0:
            raise ValueError(
                "merge thresholds must satisfy 0 <= same_source <= cross_source <= 1"
            )
        if posterior_temperature <= 0.0:
            raise ValueError("posterior_temperature must be positive")
        if not 0.0 <= null_threshold <= 1.0:
            raise ValueError("null_threshold must lie in [0, 1]")
        if not 0.0 <= credible_member_threshold <= 1.0:
            raise ValueError("credible_member_threshold must lie in [0, 1]")
        if not 0.0 <= explicit_membership_threshold <= 1.0:
            raise ValueError("explicit_membership_threshold must lie in [0, 1]")
        if not 0.0 <= uncertainty_entropy <= 1.0:
            raise ValueError("uncertainty_entropy must lie in [0, 1]")
        self.linker = linker
        self.score_provider = score_provider
        self.last_source_companion_report: dict[str, Any] | None = None
        self.candidate_pool = int(candidate_pool)
        self.candidate_tokens = int(candidate_tokens)
        self.query_tokens = int(query_tokens)
        self.merge_similarity = float(merge_similarity)
        self.same_source_merge_similarity = float(same_source_merge_similarity)
        self.posterior_temperature = float(posterior_temperature)
        self.null_threshold = float(null_threshold)
        self.credible_member_threshold = float(credible_member_threshold)
        self.explicit_membership_threshold = float(explicit_membership_threshold)
        self.uncertainty_entropy = float(uncertainty_entropy)
        self.allow_selected_scope_fixed_k_closure = bool(
            allow_selected_scope_fixed_k_closure
        )
        self.strict = bool(strict)
        self.last_report: CoverageSelectionReport | None = None
        # Text-free, gold-free diagnostics for joining selector decisions to
        # the downstream pack cutoff.  No candidate content or activation is
        # retained here; transport vectors remain local to ``select``.
        self.last_candidate_trace: list[dict[str, Any]] = []

    def _prefix_report_fields(self) -> dict[str, str | int]:
        """Read immutable checkpoint/runtime identity from the live linker."""

        encoder = getattr(self.linker, "encoder", None)
        identity = getattr(encoder, "checkpoint_identity", None)
        device = getattr(encoder, "device", "")
        dtype_name = getattr(encoder, "dtype_name", "")
        return {
            "prefix_model_id": str(getattr(identity, "model_id", "")),
            "prefix_model_revision": str(
                getattr(identity, "model_revision", "")
            ),
            "prefix_checkpoint_sha256": str(
                getattr(identity, "checkpoint_sha256", "")
            ),
            "prefix_device": str(device),
            "prefix_dtype": str(dtype_name),
            "prefix_layers": int(getattr(encoder, "layers", 0) or 0),
            "prefix_attention_layer": int(
                getattr(self.linker, "layer", -1)
            ),
        }

    def _score_provider_identity_fields(
        self,
    ) -> dict[str, str | int] | None:
        """Read checkpoint identity even when a scalar query bypasses scoring.

        A non-set query intentionally never invokes the optional score provider,
        so it has no per-call ``last_report``.  Certified runtimes still need to
        attest which already-loaded provider was bound to the selector.  These
        fields are immutable construction-time identity, not fabricated scoring
        counters.
        """

        provider = self.score_provider
        if provider is None:
            return None
        return {
            "model_id": str(getattr(provider, "model_id", "")),
            "model_revision": str(getattr(provider, "model_revision", "")),
            "checkpoint_sha256": str(
                getattr(provider, "checkpoint_sha256", "")
            ),
            "device": str(getattr(provider, "device", "")),
            "dtype": str(getattr(provider, "dtype_name", "")),
            "runtime": (
                f"{type(provider).__module__}.{type(provider).__name__}"
            ),
            "retained_transformer_state_bytes": 0,
        }

    def select_source_companions(
        self,
        query: str,
        candidates_by_source: Mapping[str, Sequence[RetrievalResult]],
    ) -> Mapping[str, RetrievalResult]:
        """Delegate source hydration to the optional scalar-score provider."""

        rows = {
            str(source_id): list(candidates)
            for source_id, candidates in candidates_by_source.items()
            if candidates
        }
        fallback = {source_id: candidates[0] for source_id, candidates in rows.items()}
        provider = self.score_provider
        selected: dict[str, RetrievalResult] = dict(fallback)
        fallback_reason = ""
        provider_report: dict[str, Any] = {}
        if provider is None:
            fallback_reason = "no_score_provider"
        else:
            try:
                proposed = provider.select_source_companions(query, rows)
                if not isinstance(proposed, Mapping):
                    raise TypeError("score provider did not return a mapping")
                for source_id, candidates in rows.items():
                    proposed_result = proposed.get(source_id)
                    proposed_id = (
                        proposed_result.chunk.chunk_id
                        if isinstance(proposed_result, RetrievalResult)
                        else None
                    )
                    match = next(
                        (
                            candidate
                            for candidate in candidates
                            if candidate.chunk.chunk_id == proposed_id
                        ),
                        None,
                    )
                    if match is not None:
                        selected[source_id] = match
                raw_report = getattr(provider, "last_source_companion_report", None)
                dump = getattr(raw_report, "model_dump", None)
                if callable(dump):
                    provider_report = dict(dump())
                elif isinstance(raw_report, Mapping):
                    provider_report = dict(raw_report)
                if int(provider_report.get("retained_transformer_state_bytes", 0)):
                    raise RuntimeError("score provider retained transformer state")
                nested_score_report = provider_report.get("score_report")
                nested_report = (
                    nested_score_report
                    if isinstance(nested_score_report, Mapping)
                    else {}
                )
                provider_reason = str(
                    provider_report.get("fallback_reason")
                    or nested_report.get("fallback_reason")
                    or ""
                )
                provider_input = int(
                    provider_report.get("input_candidates")
                    or nested_report.get("input_candidates")
                    or 0
                )
                provider_inspected = int(
                    provider_report.get("inspected_candidates")
                    or nested_report.get("inspected_candidates")
                    or 0
                )
                if provider_reason:
                    fallback_reason = provider_reason
                elif provider_input and provider_inspected < provider_input:
                    fallback_reason = (
                        "non_exhaustive_score_provider:"
                        f"{provider_inspected}/{provider_input}"
                    )
            except Exception as exc:
                if self.strict:
                    raise
                selected = fallback
                fallback_reason = f"{type(exc).__name__}: {exc}"
                provider_report = {}
        self.last_source_companion_report = {
            "input_sources": len(rows),
            "input_candidates": sum(len(candidates) for candidates in rows.values()),
            "selected_sources": len(selected),
            "selected_chunk_ids": {
                source_id: result.chunk.chunk_id
                for source_id, result in selected.items()
            },
            "provider": type(provider).__name__ if provider is not None else "",
            "provider_report": provider_report,
            "retained_transformer_state_bytes": 0,
            "fallback_reason": fallback_reason,
        }
        return selected

    @staticmethod
    def _uninspected_trace(
        candidates: Sequence[RetrievalResult],
        program: SetProgram | None = None,
    ) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": result.chunk.chunk_id,
                "source_id": result.durable_source_id,
                "selector_input_rank": index + 1,
                "group_id": None,
                "group_role": "uninspected",
                "qk_score": None,
                "ov_transport": None,
                "prefix_utility": None,
                "representative_chunk_id": None,
                "merge_similarity": None,
                "merge_threshold": None,
                "semantic_score": None,
                "answer_object_key_present": None,
                "semantic_score_kind": None,
                "answerability_score": None,
                "answerability_score_kind": None,
                "membership_score": None,
                "preferred_evidence_role": None,
                "role_match": None,
                "required_evidence_role": (
                    program.required_evidence_role if program is not None else None
                ),
                "required_evidence_role_basis": (
                    program.required_evidence_role_basis
                    if program is not None
                    else None
                ),
                "required_role_match": None,
                "value_evidence": None,
                "assignment_hypothesis": None,
                "p_existing": None,
                "p_new": None,
                "p_null": None,
                "existing_energy": None,
                "new_energy": None,
                "null_energy": None,
                "temporal_in_scope": None,
                "posterior_entropy": None,
                "posterior_kind": "uncalibrated_energy_softmax",
                "semantic_surprisal": None,
                "posterior_uncertain": None,
                "credible_cluster": False,
                "coverage_reserved": False,
                "reservation_basis": None,
            }
            for index, result in enumerate(candidates)
        ]

    def close(self) -> None:
        linker = getattr(self, "linker", None)
        torch = getattr(getattr(linker, "encoder", None), "_torch", None)
        score_provider = getattr(self, "score_provider", None)
        close_score_provider = getattr(score_provider, "close", None)
        if callable(close_score_provider):
            close_score_provider()
        self.score_provider = None
        if linker is not None:
            self.linker = None
        self.last_source_companion_report = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _forbid_occurrence_merge(
        program: SetProgram,
        timestamp: str | None,
        cluster: _PrefixEventCluster,
    ) -> bool:
        occurrence_identity = "occurrence" in program.identity_rule.casefold()
        return bool(
            occurrence_identity
            and timestamp
            and cluster.timestamps
            and timestamp not in cluster.timestamps
        )

    @staticmethod
    def _timestamp_in_scope(
        program: SetProgram,
        timestamp: str | None,
    ) -> bool | None:
        asked_at = _timestamp_key(program.query_timestamp)
        event_at = _timestamp_key(timestamp)
        if asked_at is None or event_at is None:
            return None
        age_s = asked_at - event_at
        # Evidence from after the question cannot describe a completed past
        # event, even when the wording omits an explicit lookback window.
        if age_s < 0.0:
            return False
        if program.temporal_window_days is None:
            return True
        return age_s <= program.temporal_window_days * 86_400.0

    def _fail_open(
        self,
        candidates: Sequence[RetrievalResult],
        program: SetProgram,
        *,
        started: float,
        reason: str,
        attempted: int = 0,
        inspected: int = 0,
        batches: int = 0,
        workspace_tokens: int = 0,
        active_partition_total: int | None = None,
        active_partition_inspected: int | None = None,
        active_partition_scan: Mapping[str, Any] | None = None,
        selection_status: str = "fallback",
        bypass_reason: str = "",
    ) -> list[RetrievalResult]:
        """Return the exact input objects and record an honest partial frontier."""

        output = list(candidates)
        self.last_candidate_trace = self._uninspected_trace(output, program)
        self.last_report = CoverageSelectionReport.uninspected(
            program,
            started=started,
            input_candidates=len(output),
            selection_status=selection_status,
            inspected_candidates=inspected,
            workspace_tokens=workspace_tokens,
            bypass_reason=bypass_reason,
            fallback_reason=reason,
            quantifier=program.quantifier.value,
            ordering=program.ordering.value,
            posterior_kind="uncalibrated_energy_softmax",
            frontier_candidates=len(output),
            frontier_attempted=attempted,
            frontier_uninspected=max(0, len(output) - attempted),
            routed_frontier_exhaustive=(
                None
                if selection_status == "bypassed"
                else attempted == len(output)
            ),
            frontier_batches=batches,
            active_partition_total=active_partition_total,
            active_partition_inspected=active_partition_inspected,
            active_partition_exhaustive=(
                None
                if selection_status == "bypassed"
                else (
                    active_partition_inspected >= active_partition_total
                    if active_partition_total is not None
                    and active_partition_inspected is not None
                    else None
                )
            ),
            **dict(active_partition_scan or {}),
            allow_selected_scope_fixed_k_closure=(
                self.allow_selected_scope_fixed_k_closure
            ),
            score_provider_report=self._score_provider_identity_fields(),
            **self._prefix_report_fields(),
            required_evidence_role=program.required_evidence_role,
            required_evidence_role_basis=program.required_evidence_role_basis,
            query_timestamp=program.query_timestamp,
            temporal_window_days=program.temporal_window_days,
        )
        return output

    def _cluster_posterior(
        self,
        *,
        quality: float,
        vector: np.ndarray,
        source_id: str,
        timestamp: str | None,
        program: SetProgram,
        clusters: Sequence[_PrefixEventCluster],
        temporal_in_scope: bool | None,
        answer_object_key: str | None,
    ) -> tuple[
        float,
        float,
        float,
        float | None,
        float,
        float,
        int | None,
        float | None,
        float | None,
    ]:
        """Return uncalibrated posterior-shaped scores and a compatible slot."""

        effective_member = 0.05 + 0.90 * max(0.0, min(1.0, quality))
        member_energy = math.log(effective_member / (1.0 - effective_member))
        if temporal_in_scope is False:
            # A deterministic query/date contradiction is stronger NULL
            # evidence than any uncalibrated semantic magnitude.
            member_energy -= 8.0
        existing_energies: list[float] = []
        compatibility: list[bool] = []
        similarities: list[float | None] = []
        thresholds: list[float | None] = []
        cluster_prior = math.log(max(1, len(clusters)))

        for cluster in clusters:
            member_similarities = [
                float(np.dot(vector, member_vector))
                for member_vector in cluster.vectors
            ]
            member_thresholds = [
                self.same_source_merge_similarity
                if source_id == member.result.durable_source_id
                else self.merge_similarity
                for member in cluster.members
            ]
            similarity = min(member_similarities)
            threshold = max(member_thresholds)
            occurrence_forbidden = self._forbid_occurrence_merge(
                program,
                timestamp,
                cluster,
            )
            identity_equal = bool(
                answer_object_key
                and cluster.answer_object_keys == {answer_object_key}
            )
            identity_conflict = bool(
                answer_object_key
                and cluster.answer_object_keys
                and answer_object_key not in cluster.answer_object_keys
            )
            forbidden = occurrence_forbidden or identity_conflict
            compatible = (not forbidden) and (
                identity_equal
                or all(
                    value >= limit
                    for value, limit in zip(
                        member_similarities,
                        member_thresholds,
                        strict=True,
                    )
                )
            )
            margin = (
                12.0
                if identity_equal and not forbidden
                else (similarity - threshold) / self.posterior_temperature
            )
            if forbidden:
                margin = -12.0
            elif not compatible:
                # Keep the hypothesis explicit without allowing a just-below
                # threshold vector to merge through posterior mass alone.
                margin = min(-2.5, margin)
            metadata_bonus = 0.0
            if source_id in cluster.source_ids:
                metadata_bonus += 0.25
            if timestamp and timestamp in cluster.timestamps:
                metadata_bonus += 0.20
            existing_energies.append(
                member_energy
                + 0.50
                + max(-12.0, min(12.0, margin))
                + metadata_bonus
                + 0.08 * math.log1p(len(cluster.members))
                - cluster_prior
            )
            compatibility.append(compatible)
            similarities.append(similarity)
            thresholds.append(threshold)

        new_energy = member_energy + 0.35
        null_energy = -member_energy - 0.35
        normalized = _energy_softmax(
            [*existing_energies, new_energy, null_energy]
        )
        existing_probabilities = normalized[: len(existing_energies)]
        p_new = normalized[-2]
        p_null = normalized[-1]
        p_existing = sum(existing_probabilities)
        aggregate_existing_energy: float | None = None
        if existing_energies:
            peak_existing = max(existing_energies)
            aggregate_existing_energy = peak_existing + math.log(
                sum(
                    math.exp(value - peak_existing)
                    for value in existing_energies
                )
            )
        best_cluster: int | None = None
        best_similarity: float | None = None
        best_threshold: float | None = None
        if existing_probabilities:
            diagnostic_slot = max(
                range(len(existing_probabilities)),
                key=lambda index: existing_probabilities[index],
            )
            compatible_slots = [
                index for index, value in enumerate(compatibility) if value
            ]
            proposed = (
                max(
                    compatible_slots,
                    key=lambda index: existing_probabilities[index],
                )
                if compatible_slots
                else diagnostic_slot
            )
            best_similarity = similarities[proposed]
            best_threshold = thresholds[proposed]
            # The global posterior divides existing-slot prior mass across K
            # clusters so aggregate EXISTING mass remains well behaved. Slot
            # identity, however, is a conditional comparison between the best
            # compatible slot and NEW/NULL. Undo only that K prior here: adding
            # unrelated clusters must not turn an exact duplicate into NEW.
            conditional_existing_energy = (
                existing_energies[proposed] + cluster_prior
            )
            conditional = _energy_softmax(
                [conditional_existing_energy, new_energy, null_energy]
            )
            if compatibility[proposed] and conditional[0] >= max(
                conditional[1],
                conditional[2],
            ):
                best_cluster = proposed
        return (
            p_existing,
            p_new,
            p_null,
            aggregate_existing_energy,
            new_energy,
            null_energy,
            best_cluster,
            best_similarity,
            best_threshold,
        )

    def select(
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
    ) -> list[RetrievalResult]:
        """Run the bounded prefix pipeline while keeping runtime state local."""

        return select_prefix_coverage(
            self,
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
