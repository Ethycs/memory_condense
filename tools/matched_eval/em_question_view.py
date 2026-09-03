"""Pure cumulative-evidence question carriers used by EM prediction stages.

The historical locked-EM adapter also contains artifact and validation
loaders.  Keeping these two structural carriers here lets confirmation
prediction code share the exact types without importing that loader surface.
"""

from __future__ import annotations

from dataclasses import dataclass

from memory_condense.eval.recall_guarded_cumulative_fast_artifact import FastEvidence


@dataclass(frozen=True, slots=True)
class LockedEMStageView:
    """Minimal cumulative stage surface consumed by EM v2."""

    stage_id: str
    stage_receipt_sha256: str
    evidence_projection_sha256: str
    evidence: tuple[FastEvidence, ...]

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(row.evidence_id for row in self.evidence)


@dataclass(frozen=True, slots=True)
class LockedEMQuestionView:
    """Minimal retrieval-question-compatible cumulative projection."""

    ordinal: int
    question_id: str
    question_sha256: str
    dated_question_sha256: str
    retrieval_question_part_sha256: str
    dated_question: str
    stages: tuple[LockedEMStageView, LockedEMStageView]

    @property
    def stage_ids(self) -> tuple[str, ...]:
        return tuple(stage.stage_id for stage in self.stages)

    def stage(self, stage_id: str) -> LockedEMStageView:
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        raise KeyError(stage_id)


__all__ = ["LockedEMQuestionView", "LockedEMStageView"]
