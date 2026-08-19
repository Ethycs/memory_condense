"""Read-only store boundary used by the discourse-closure engine."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from memory_condense.domain.discourse import (
    ArtifactCoverageReceipt,
    DiscourseRelation,
    DiscourseSnapshot,
    DiscourseUnit,
    Episode,
    EvidenceSpan,
)


@runtime_checkable
class EvidenceClosureStore(Protocol):
    """Minimum source-grounded graph API required by closure.

    Implementations return immutable domain values and hydrate raw text only
    for the duration of the request.  The protocol contains no method for
    reading or writing model tokens, activations, attention, or K/V state.
    """

    def snapshot(self, graph_revision: int | None = None) -> DiscourseSnapshot:
        """Return one immutable content/graph high-water receipt."""

    def get_episode(self, episode_id: str) -> Episode | None:
        """Return an episode by exact identity."""

    def episode_ids_for_chunks(
        self,
        chunk_ids: Sequence[str],
        *,
        artifact_id: str | None = None,
    ) -> dict[str, str]:
        """Map raw hits to deterministic episode identities when available."""

    def adjacent_episodes(
        self,
        episode_id: str,
        *,
        radius: int = 1,
        include_self: bool = False,
    ) -> tuple[Episode, ...]:
        """Return source-local neighbors in source order."""

    def coverage_for_chunks(
        self,
        artifact_id: str,
        chunk_ids: Sequence[str],
        *,
        coverage_kind: str = "discourse",
    ) -> dict[str, str]:
        """Return fresh annotation/no-output receipts for requested chunks."""

    def artifact_coverage(
        self,
        artifact_id: str,
        coverage_kind: str = "discourse",
    ) -> ArtifactCoverageReceipt | None:
        """Return finalized whole-corpus coverage at the current source revision."""

    def units_for_artifact(
        self,
        artifact_id: str,
        *,
        limit: int | None = None,
    ) -> tuple[DiscourseUnit, ...]:
        """Return a deterministic bounded artifact-wide unit scan."""

    def units_for_chunks(
        self,
        chunk_ids: Sequence[str],
        *,
        artifact_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[DiscourseUnit, ...]:
        """Return units evidenced by any supplied raw chunk."""

    def relations_for_chunks(
        self,
        chunk_ids: Sequence[str],
        *,
        artifact_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[DiscourseRelation, ...]:
        """Return relations directly evidenced by supplied raw chunks."""

    def get_unit(self, unit_id: str) -> DiscourseUnit | None:
        """Return a unit by exact identity."""

    def get_relation(self, relation_id: str) -> DiscourseRelation | None:
        """Return a relation by exact identity."""

    def incident_relations(
        self,
        unit_ids: Sequence[str],
        *,
        artifact_id: str | None = None,
        max_degree: int,
    ) -> dict[str, tuple[DiscourseRelation, ...]]:
        """Return bounded incident edges for every supplied unit."""

    def evidence_for_chunks(
        self,
        chunk_ids: Sequence[str],
    ) -> tuple[EvidenceSpan, ...]:
        """Return verified full raw spans in first-input order."""

    def hydrate_span(self, span: EvidenceSpan) -> str:
        """Hydrate and verify exact source text transiently."""


__all__ = ["EvidenceClosureStore"]
