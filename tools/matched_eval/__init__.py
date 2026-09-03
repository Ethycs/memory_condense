"""Versioned, gold-blind evaluation spine for matched retrieval experiments.

The package intentionally lives under :mod:`tools` during the legacy migration.
Historical LongMemEval artifacts bind the digest of every Python file under
``src/memory_condense``; changing that tree before import would invalidate the
artifacts that establish the 57/60/53 migration checkpoint.

The package facade is intentionally inert.  Prediction code imports focused
submodules such as :mod:`tools.matched_eval.live`; importing one of those
submodules must not also import (or retain a callable route to) the
renderer/runner evaluation surface and its benchmark loaders.  Runtime,
renderer, and ledger names therefore live only in their explicit submodules.
"""

from __future__ import annotations

from .contracts import (
    AnswerOperatorDelta,
    ArmPlan,
    ArtifactRef,
    EvaluationMemorySnapshot,
    EvidenceItem,
    FactItem,
    LinkingDelta,
    LinkItem,
    MembershipDelta,
    MemoryPacket,
    ObservationDelta,
    PlanMode,
    RepresentationDelta,
    StageBudget,
    StageDisposition,
    StagePlan,
    StageTrace,
)

__all__ = [
    "AnswerOperatorDelta",
    "ArmPlan",
    "ArtifactRef",
    "EvaluationMemorySnapshot",
    "EvidenceItem",
    "FactItem",
    "LinkingDelta",
    "LinkItem",
    "MembershipDelta",
    "MemoryPacket",
    "ObservationDelta",
    "PlanMode",
    "RepresentationDelta",
    "StageBudget",
    "StageDisposition",
    "StagePlan",
    "StageTrace",
]
