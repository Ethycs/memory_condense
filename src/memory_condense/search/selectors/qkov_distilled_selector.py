"""Conservative MiniLM shadow/apply adapter for a QK/OV teacher.

The student is only a ranking surrogate.  Its scalar scores are never exposed
as QK attention or OV transport, and raw-memory hydration remains the caller's
responsibility.  Any uncertainty delegates the complete, unchanged candidate
groups to the teacher.  If that is impossible, the adapter returns every
distinct candidate in original order and explicitly requests raw fallback.
"""

from __future__ import annotations

import hmac
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from memory_condense.associations.head_memory_models import (
    AssociativeMemoryCandidate,
    MemoryLinkHit,
    NestedMemoryInspection,
)
from memory_condense.domain._discourse_identity import identity_sha256


QKOVDistilledMode = Literal["shadow", "apply"]
QKOVDistilledDecision = Literal["student", "teacher", "fail_open"]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCORE_SEMANTICS = "distilled_ranking_scalar_not_qk_or_ov"


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class QKOVStudentIdentity:
    """Complete behavioral identity of one distilled scalar scorer."""

    model_id: str
    model_revision: str
    checkpoint_sha256: str
    tokenizer_sha256: str
    contract_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.model_id) is not str
            or type(self.model_revision) is not str
            or not self.model_id.strip()
            or not self.model_revision.strip()
        ):
            raise ValueError("student model identity must be non-empty")
        _sha256(self.checkpoint_sha256, "student checkpoint_sha256")
        _sha256(self.tokenizer_sha256, "student tokenizer_sha256")
        _sha256(self.contract_sha256, "student contract_sha256")

    def identity_payload(self) -> dict[str, str]:
        return {
            "format": "memory-condense-qkov-student-identity-v1",
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "checkpoint_sha256": self.checkpoint_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "contract_sha256": self.contract_sha256,
        }

    @property
    def identity_sha256(self) -> str:
        return identity_sha256(self.identity_payload())


@dataclass(frozen=True, slots=True)
class QKOVDistilledPolicy:
    """Sealed bounds and acceptance threshold for one student deployment."""

    expected_student_identity_sha256: str
    mode: QKOVDistilledMode = "shadow"
    min_margin: float = 0.0
    max_candidates: int = 8
    max_student_candidates: int = 64
    batch_size: int = 32

    def __post_init__(self) -> None:
        _sha256(
            self.expected_student_identity_sha256,
            "expected_student_identity_sha256",
        )
        if self.mode not in {"shadow", "apply"}:
            raise ValueError("mode must be 'shadow' or 'apply'")
        try:
            margin = float(self.min_margin)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("min_margin must be finite and non-negative") from exc
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError("min_margin must be finite and non-negative")
        object.__setattr__(self, "min_margin", 0.0 if margin == 0.0 else margin)
        _positive_integer(self.max_candidates, "max_candidates")
        _positive_integer(self.max_student_candidates, "max_student_candidates")
        _positive_integer(self.batch_size, "batch_size")

    def identity_payload(self) -> dict[str, object]:
        return {
            "format": "memory-condense-qkov-distilled-policy-v1",
            "expected_student_identity_sha256": (
                self.expected_student_identity_sha256
            ),
            "mode": self.mode,
            "min_margin": self.min_margin,
            "max_candidates": self.max_candidates,
            "max_student_candidates": self.max_student_candidates,
            "batch_size": self.batch_size,
        }

    @property
    def policy_sha256(self) -> str:
        return identity_sha256(self.identity_payload())


@dataclass(frozen=True, slots=True)
class QKOVDistilledReport:
    """Text-free semantic receipt for one student/teacher decision."""

    mode: QKOVDistilledMode
    decision: QKOVDistilledDecision
    reason: str
    scope_complete: bool
    requires_completeness: bool
    candidate_count_truncated: bool
    input_candidates: int
    effective_top_k: int
    selection_exhaustive: bool
    student_scored_candidates: int
    returned_candidates: int
    student_accepted: bool
    observed_margin: float | None
    student_candidate_ids: tuple[str, ...]
    returned_candidate_ids: tuple[str, ...]
    teacher_called: bool
    requires_raw_fallback: bool
    student_identity_sha256: str
    expected_student_identity_sha256: str
    policy_sha256: str


@dataclass(frozen=True, slots=True)
class DistilledSelectionHit:
    """One student ranking score, with no QK/OV fields by construction."""

    episode_id: str
    score: float
    rank: int
    score_kind: Literal["distilled_ranking_scalar"] = "distilled_ranking_scalar"


@dataclass(frozen=True, slots=True)
class DistilledSelection:
    """Native result of the student scorer before linker compatibility."""

    hits: tuple[DistilledSelectionHit, ...]
    input_candidates: int
    effective_top_k: int
    selection_exhaustive: bool
    observed_margin: float | None
    student_identity_sha256: str


class _StudentUncertain(Exception):
    def __init__(
        self,
        reason: str,
        *,
        scored: int = 0,
        margin: float | None = None,
        proposal_ids: tuple[str, ...] = (),
    ) -> None:
        self.reason = reason
        self.scored = scored
        self.margin = margin
        self.proposal_ids = proposal_ids


def _flat_prediction_scores(predicted: Any, expected: int) -> tuple[float, ...]:
    values = predicted.tolist() if hasattr(predicted, "tolist") else predicted
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        rows = list(values)
    else:
        rows = [values]
    flattened: list[object] = []
    for row in rows:
        if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
            children = list(row)
            if len(children) != 1:
                raise _StudentUncertain("score_shape_mismatch")
            flattened.append(children[0])
        else:
            flattened.append(row)
    if len(flattened) != expected:
        raise _StudentUncertain("score_count_mismatch")
    scores: list[float] = []
    for value in flattened:
        if isinstance(value, bool):
            raise _StudentUncertain("nonfinite_student_score")
        try:
            score = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise _StudentUncertain("nonfinite_student_score") from exc
        if not math.isfinite(score):
            raise _StudentUncertain("nonfinite_student_score")
        scores.append(0.0 if score == 0.0 else score)
    return tuple(scores)


class QKOVDistilledSelector:
    """Expose a conservative distilled scorer through the nested-linker API.

    ``scope_complete`` and ``requires_completeness`` are deliberately explicit
    constructor inputs.  The current contextual-card caller does not pass them
    through ``inspect_nested``; binding them when constructing this query-local
    adapter prevents an omitted value from silently authorizing the student.
    """

    def __init__(
        self,
        student: Any,
        *,
        identity: QKOVStudentIdentity,
        policy: QKOVDistilledPolicy,
        scope_complete: bool,
        requires_completeness: bool,
        candidate_count_truncated: bool,
        teacher: Any | None = None,
    ) -> None:
        if type(scope_complete) is not bool:
            raise TypeError("scope_complete must be bool")
        if type(requires_completeness) is not bool:
            raise TypeError("requires_completeness must be bool")
        if type(candidate_count_truncated) is not bool:
            raise TypeError("candidate_count_truncated must be bool")
        self.student = student
        self.identity = identity
        self.policy = policy
        self.scope_complete = scope_complete
        self.requires_completeness = requires_completeness
        self.candidate_count_truncated = candidate_count_truncated
        self.teacher = teacher
        teacher_limit = getattr(teacher, "max_candidates", policy.max_candidates)
        if isinstance(teacher_limit, bool) or not isinstance(teacher_limit, int):
            teacher_limit = policy.max_candidates
        self.max_candidates = min(policy.max_candidates, max(1, teacher_limit))
        self.last_report: QKOVDistilledReport | None = None
        self.last_distilled_selection: DistilledSelection | None = None

    @property
    def requires_raw_fallback(self) -> bool:
        return self.last_report is None or self.last_report.requires_raw_fallback

    @staticmethod
    def _flatten(
        groups: Sequence[Sequence[AssociativeMemoryCandidate]],
    ) -> tuple[AssociativeMemoryCandidate, ...]:
        return tuple(candidate for group in groups for candidate in group)

    @staticmethod
    def _finalist_capacity(
        groups: Sequence[Sequence[AssociativeMemoryCandidate]],
        *,
        beam_per_group: int,
        max_candidates: int,
    ) -> int:
        """Mirror the nested linker's structural result-cardinality ceiling."""

        finalists = sum(min(len(group), beam_per_group) for group in groups)
        while finalists > max_candidates:
            full_groups, remainder = divmod(finalists, max_candidates)
            finalists = full_groups * beam_per_group + min(
                remainder,
                beam_per_group,
            )
        return finalists

    def _report(
        self,
        *,
        decision: QKOVDistilledDecision,
        reason: str,
        input_count: int,
        effective_top_k: int,
        selection_exhaustive: bool,
        scored: int,
        returned_ids: tuple[str, ...],
        student_accepted: bool,
        margin: float | None,
        proposal_ids: tuple[str, ...],
        teacher_called: bool,
        requires_raw_fallback: bool,
    ) -> None:
        self.last_report = QKOVDistilledReport(
            mode=self.policy.mode,
            decision=decision,
            reason=reason,
            scope_complete=self.scope_complete,
            requires_completeness=self.requires_completeness,
            candidate_count_truncated=self.candidate_count_truncated,
            input_candidates=input_count,
            effective_top_k=effective_top_k,
            selection_exhaustive=selection_exhaustive,
            student_scored_candidates=scored,
            returned_candidates=len(returned_ids),
            student_accepted=student_accepted,
            observed_margin=margin,
            student_candidate_ids=proposal_ids,
            returned_candidate_ids=returned_ids,
            teacher_called=teacher_called,
            requires_raw_fallback=requires_raw_fallback,
            student_identity_sha256=self.identity.identity_sha256,
            expected_student_identity_sha256=(
                self.policy.expected_student_identity_sha256
            ),
            policy_sha256=self.policy.policy_sha256,
        )

    @staticmethod
    def _original_order_hits(
        candidates: Sequence[AssociativeMemoryCandidate],
    ) -> tuple[MemoryLinkHit, ...]:
        hits: list[MemoryLinkHit] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.episode_id in seen:
                continue
            seen.add(candidate.episode_id)
            hits.append(
                MemoryLinkHit(
                    episode_id=candidate.episode_id,
                    qk_score=0.0,
                    ov_transport=0.0,
                    head_weights=(),
                    metadata={
                        **candidate.metadata,
                        "selection_backend": "qkov_distilled_fail_open",
                        "score_kind": "unscored_raw_fallback",
                        "score_semantics": "none",
                        "qk_score_available": False,
                        "ov_transport_available": False,
                        "compatibility_qk_score_is_placeholder": True,
                        "compatibility_ov_transport_is_placeholder": True,
                        "requires_raw_fallback": True,
                    },
                )
            )
        return tuple(hits)

    def _teacher_result_is_valid(
        self,
        result: Any,
        candidate_ids: set[str],
        *,
        effective_top_k: int,
        input_candidates: int,
    ) -> bool:
        def counter(name: str) -> int | None:
            value = getattr(result, name, None)
            if isinstance(value, bool) or not isinstance(value, int):
                return None
            return value

        hits = getattr(result, "hits", None)
        if (
            not isinstance(hits, tuple)
            or not hits
            or len(hits) > effective_top_k
        ):
            return False
        ids = tuple(getattr(hit, "episode_id", None) for hit in hits)
        if not (
            all(type(episode_id) is str for episode_id in ids)
            and len(ids) == len(set(ids))
            and set(ids).issubset(candidate_ids)
        ):
            return False
        for hit in hits:
            for name in ("qk_score", "ov_transport"):
                value = getattr(hit, name, None)
                if isinstance(value, bool):
                    return False
                try:
                    score = float(value)
                except (TypeError, ValueError, OverflowError):
                    return False
                if not math.isfinite(score):
                    return False

        passes = counter("passes")
        workspace_candidates = counter("max_workspace_candidates")
        workspace_tokens = counter("max_workspace_tokens")
        inspections = counter("total_candidate_inspections")
        if None in (passes, workspace_candidates, workspace_tokens, inspections):
            return False
        assert passes is not None
        assert workspace_candidates is not None
        assert workspace_tokens is not None
        assert inspections is not None
        return (
            passes >= 1
            and len(hits) <= workspace_candidates <= self.max_candidates
            and workspace_candidates <= input_candidates
            and workspace_tokens >= 0
            and inspections >= input_candidates
            and passes <= inspections
            and inspections <= passes * workspace_candidates
        )

    def _teacher_or_fail_open(
        self,
        source_text: str,
        original_groups: tuple[tuple[AssociativeMemoryCandidate, ...], ...],
        candidates: tuple[AssociativeMemoryCandidate, ...],
        *,
        beam_per_group: int,
        effective_top_k: int,
        selection_exhaustive: bool,
        score_mode: Literal["qk", "qk_ov"],
        reason: str,
        scored: int,
        margin: float | None,
        proposal_ids: tuple[str, ...],
        student_accepted: bool,
    ) -> NestedMemoryInspection:
        teacher = self.teacher
        teacher_called = teacher is not None
        if teacher is not None:
            try:
                result = teacher.inspect_nested(
                    source_text,
                    original_groups,
                    beam_per_group=beam_per_group,
                    top_k=effective_top_k,
                    score_mode=score_mode,
                )
                if not self._teacher_result_is_valid(
                    result,
                    {candidate.episode_id for candidate in candidates},
                    effective_top_k=effective_top_k,
                    input_candidates=len(candidates),
                ):
                    raise ValueError("teacher returned invalid candidate hits")
                returned_ids = tuple(hit.episode_id for hit in result.hits)
                self._report(
                    decision="teacher",
                    reason=reason,
                    input_count=len(candidates),
                    effective_top_k=effective_top_k,
                    selection_exhaustive=selection_exhaustive,
                    scored=scored,
                    returned_ids=returned_ids,
                    student_accepted=student_accepted,
                    margin=margin,
                    proposal_ids=proposal_ids,
                    teacher_called=True,
                    requires_raw_fallback=(
                        not self.scope_complete
                        or self.requires_completeness
                        or self.candidate_count_truncated
                    ),
                )
                return result
            except Exception as exc:
                reason = f"teacher_failed:{type(exc).__name__}"

        hits = self._original_order_hits(candidates)
        returned_ids = tuple(hit.episode_id for hit in hits)
        self._report(
            decision="fail_open",
            reason=(reason if teacher_called else f"teacher_unavailable:{reason}"),
            input_count=len(candidates),
            effective_top_k=effective_top_k,
            selection_exhaustive=selection_exhaustive,
            scored=scored,
            returned_ids=returned_ids,
            student_accepted=student_accepted,
            margin=margin,
            proposal_ids=proposal_ids,
            teacher_called=teacher_called,
            requires_raw_fallback=True,
        )
        return NestedMemoryInspection(
            hits=hits,
            passes=0,
            max_workspace_candidates=0,
            max_workspace_tokens=0,
            total_candidate_inspections=0,
        )

    def _student_selection(
        self,
        source_text: str,
        candidates: tuple[AssociativeMemoryCandidate, ...],
        *,
        effective_top_k: int,
        selection_exhaustive: bool,
    ) -> DistilledSelection:
        if not self.scope_complete:
            raise _StudentUncertain("scope_incomplete")
        if self.requires_completeness:
            raise _StudentUncertain("query_requires_completeness")
        if self.candidate_count_truncated:
            raise _StudentUncertain("candidate_count_truncated")
        if not hmac.compare_digest(
            self.identity.identity_sha256,
            self.policy.expected_student_identity_sha256,
        ):
            raise _StudentUncertain("student_identity_mismatch")
        if len(candidates) > self.policy.max_student_candidates:
            raise _StudentUncertain("student_candidate_limit_exceeded")
        candidate_ids = tuple(candidate.episode_id for candidate in candidates)
        if any(not episode_id for episode_id in candidate_ids) or len(
            candidate_ids
        ) != len(set(candidate_ids)):
            raise _StudentUncertain("duplicate_or_empty_candidate_id")
        predict = getattr(self.student, "predict", None)
        if not callable(predict):
            raise _StudentUncertain("student_predict_unavailable")
        pairs = [(source_text, candidate.text) for candidate in candidates]
        try:
            predicted = predict(
                pairs,
                batch_size=self.policy.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise _StudentUncertain("student_exception") from exc
        scores = _flat_prediction_scores(predicted, len(candidates))
        order = sorted(
            range(len(candidates)),
            key=lambda index: (-scores[index], index),
        )
        if len(order) < 2:
            selected_indices = order
        elif scores[order[0]] == scores[order[1]]:
            raise _StudentUncertain("student_top_score_tie", scored=len(scores))
        else:
            selected_indices = order[:effective_top_k]
        proposal = tuple(
            candidates[index].episode_id
            for index in selected_indices
        )
        margin: float | None = None
        if not selection_exhaustive:
            margin = (
                scores[order[effective_top_k - 1]]
                - scores[order[effective_top_k]]
            )
            if not margin > self.policy.min_margin:
                raise _StudentUncertain(
                    "student_margin_not_strictly_above_threshold",
                    scored=len(scores),
                    margin=margin,
                    proposal_ids=proposal,
                )
        hits = tuple(
            DistilledSelectionHit(
                episode_id=candidates[index].episode_id,
                score=scores[index],
                rank=rank,
            )
            for rank, index in enumerate(selected_indices, start=1)
        )
        return DistilledSelection(
            hits=hits,
            input_candidates=len(candidates),
            effective_top_k=effective_top_k,
            selection_exhaustive=selection_exhaustive,
            observed_margin=margin,
            student_identity_sha256=self.identity.identity_sha256,
        )

    @staticmethod
    def _compatibility_inspection(
        selection: DistilledSelection,
        candidates: Sequence[AssociativeMemoryCandidate],
    ) -> NestedMemoryInspection:
        """Adapt native scalar hits without representing them as QK or OV."""

        candidate_by_id = {
            candidate.episode_id: candidate for candidate in candidates
        }
        return NestedMemoryInspection(
            hits=tuple(
                MemoryLinkHit(
                    episode_id=hit.episode_id,
                    # Required placeholders in the legacy MemoryLinkHit shape.
                    qk_score=0.0,
                    ov_transport=0.0,
                    head_weights=(),
                    metadata={
                        **candidate_by_id[hit.episode_id].metadata,
                        "selection_backend": "qkov_distilled_student_adapter",
                        "score_kind": hit.score_kind,
                        "score_semantics": _SCORE_SEMANTICS,
                        "distilled_student_score": hit.score,
                        "distilled_student_rank": hit.rank,
                        "effective_top_k": selection.effective_top_k,
                        "selection_exhaustive": selection.selection_exhaustive,
                        "student_identity_sha256": (
                            selection.student_identity_sha256
                        ),
                        "qk_score_available": False,
                        "ov_transport_available": False,
                        "compatibility_qk_score_is_placeholder": True,
                        "compatibility_ov_transport_is_placeholder": True,
                    },
                )
                for hit in selection.hits
            ),
            passes=1,
            max_workspace_candidates=selection.input_candidates,
            max_workspace_tokens=0,
            total_candidate_inspections=selection.input_candidates,
        )

    def inspect_nested(
        self,
        source_text: str,
        candidate_groups: Sequence[Sequence[AssociativeMemoryCandidate]],
        *,
        beam_per_group: int = 2,
        top_k: int = 4,
        score_mode: Literal["qk", "qk_ov"] = "qk",
    ) -> NestedMemoryInspection:
        """Score every offered candidate or delegate the unchanged frontier."""

        normalized = str(source_text).strip()
        if not normalized:
            raise ValueError("source_text must be non-empty")
        beam_per_group = _positive_integer(beam_per_group, "beam_per_group")
        top_k = _positive_integer(top_k, "top_k")
        if beam_per_group >= self.max_candidates:
            raise ValueError("beam_per_group must be smaller than max_candidates")
        if score_mode not in {"qk", "qk_ov"}:
            raise ValueError("score_mode must be 'qk' or 'qk_ov'")
        original_groups = tuple(tuple(group) for group in candidate_groups)
        groups = tuple(group for group in original_groups if group)
        if not groups:
            raise ValueError("at least one non-empty candidate group is required")
        if any(len(group) > self.max_candidates for group in groups):
            raise ValueError("a candidate group exceeds max_candidates")
        candidates = self._flatten(groups)
        finalist_capacity = self._finalist_capacity(
            groups,
            beam_per_group=beam_per_group,
            max_candidates=self.max_candidates,
        )
        effective_top_k = min(top_k, finalist_capacity)
        selection_exhaustive = effective_top_k == len(candidates)
        self.last_distilled_selection = None

        try:
            selection = self._student_selection(
                normalized,
                candidates,
                effective_top_k=effective_top_k,
                selection_exhaustive=selection_exhaustive,
            )
        except _StudentUncertain as uncertain:
            return self._teacher_or_fail_open(
                normalized,
                original_groups,
                candidates,
                beam_per_group=beam_per_group,
                effective_top_k=effective_top_k,
                selection_exhaustive=selection_exhaustive,
                score_mode=score_mode,
                reason=uncertain.reason,
                scored=uncertain.scored,
                margin=uncertain.margin,
                proposal_ids=uncertain.proposal_ids,
                student_accepted=False,
            )

        proposal_ids = tuple(hit.episode_id for hit in selection.hits)
        self.last_distilled_selection = selection
        if self.policy.mode == "shadow":
            return self._teacher_or_fail_open(
                normalized,
                original_groups,
                candidates,
                beam_per_group=beam_per_group,
                effective_top_k=effective_top_k,
                selection_exhaustive=selection_exhaustive,
                score_mode=score_mode,
                reason="shadow_mode",
                scored=len(candidates),
                margin=selection.observed_margin,
                proposal_ids=proposal_ids,
                student_accepted=True,
            )

        inspection = self._compatibility_inspection(selection, candidates)
        returned_ids = tuple(hit.episode_id for hit in inspection.hits)
        self._report(
            decision="student",
            reason=(
                "student_exhaustive_accepted"
                if selection_exhaustive
                else "student_cutoff_margin_accepted"
            ),
            input_count=len(candidates),
            effective_top_k=effective_top_k,
            selection_exhaustive=selection_exhaustive,
            scored=len(candidates),
            returned_ids=returned_ids,
            student_accepted=True,
            margin=selection.observed_margin,
            proposal_ids=proposal_ids,
            teacher_called=False,
            requires_raw_fallback=False,
        )
        return inspection


__all__ = [
    "DistilledSelection",
    "DistilledSelectionHit",
    "QKOVDistilledDecision",
    "QKOVDistilledMode",
    "QKOVDistilledPolicy",
    "QKOVDistilledReport",
    "QKOVDistilledSelector",
    "QKOVStudentIdentity",
]
