"""Versioned, gold-blind evaluation spine for matched retrieval experiments.

The package intentionally lives under :mod:`tools` during the legacy migration.
Historical LongMemEval artifacts bind the digest of every Python file under
``src/memory_condense``; changing that tree before import would invalidate the
artifacts that establish the 57/60/53 migration checkpoint.
"""

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
from .ledger import (
    RuntimeLedgerEntry,
    ScoreLedgerEntry,
    build_runtime_ledger,
    build_score_ledger,
    runtime_entry_from_stage_run,
)
from .renderer import RENDERER_ID, RenderedPrompt, render_memory_packet
from .runner import (
    ArmRunResult,
    MatchedEvalRunner,
    StageRunReceipt,
    StageRunResult,
    run_arm,
)

__all__ = [
    "AnswerOperatorDelta",
    "ArmPlan",
    "ArmRunResult",
    "ArtifactRef",
    "EvaluationMemorySnapshot",
    "EvidenceItem",
    "FactItem",
    "LinkingDelta",
    "LinkItem",
    "MembershipDelta",
    "MemoryPacket",
    "MatchedEvalRunner",
    "ObservationDelta",
    "PlanMode",
    "RepresentationDelta",
    "RENDERER_ID",
    "RenderedPrompt",
    "RuntimeLedgerEntry",
    "ScoreLedgerEntry",
    "StageBudget",
    "StageDisposition",
    "StagePlan",
    "StageRunReceipt",
    "StageRunResult",
    "StageTrace",
    "build_runtime_ledger",
    "build_score_ledger",
    "render_memory_packet",
    "run_arm",
    "runtime_entry_from_stage_run",
]
