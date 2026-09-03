"""Prediction-safe carriers for authenticated query-expansion artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from tools.matched_eval.artifacts import SealedArtifact


@dataclass(frozen=True, slots=True)
class VerifiedQueryExpansionArtifacts:
    """Exact replayed query-expansion artifacts accepted downstream."""

    preflight: SealedArtifact
    run: SealedArtifact
    run_replay: SealedArtifact
    runtime_ledger: SealedArtifact
    runtime_ledger_replay: SealedArtifact


__all__ = ["VerifiedQueryExpansionArtifacts"]
