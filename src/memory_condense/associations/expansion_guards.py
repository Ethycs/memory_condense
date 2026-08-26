"""Shared admission preamble and rollback predicate for expansion arms.

Every association expansion arm (QK ranked, Hebbian co-access, heat diffusion)
opens with the same guard block — cap the result count, validate the shared
budget knobs, bound the anchor window — and closes with the same admission
test: roll the arm back to direct retrieval when the composed prompt would
exceed the token increase budget. This module owns both so the arms cannot
drift on the shared parts; arm-specific validations stay at each call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from memory_condense.domain.schemas import RetrievalResult

if TYPE_CHECKING:
    from memory_condense.associations.association_store import AssociationStore


@dataclass(frozen=True, slots=True)
class ExpansionGuards:
    """The validated common preamble of one expansion call."""

    result_cap: int
    bounded_anchors: list[RetrievalResult]


def guard_expansion_request(
    anchors: Sequence[RetrievalResult],
    *,
    k: int | None,
    lexical_protection_threshold: float | None,
    max_prompt_token_increase: int | None,
) -> ExpansionGuards:
    """Validate the shared expansion knobs and bound the anchor window.

    A non-positive ``result_cap`` short-circuits before any validation, which
    preserves the historical behavior of every arm: ``k=0`` returns empty
    rather than raising on an unrelated invalid knob.
    """

    result_cap = len(anchors) if k is None else int(k)
    if result_cap <= 0:
        return ExpansionGuards(result_cap=result_cap, bounded_anchors=[])
    if lexical_protection_threshold is not None and not (
        0.0 <= lexical_protection_threshold <= 1.0
    ):
        raise ValueError("lexical_protection_threshold must lie in [0, 1]")
    if max_prompt_token_increase is not None and max_prompt_token_increase < 0:
        raise ValueError("max_prompt_token_increase must be non-negative")
    return ExpansionGuards(
        result_cap=result_cap,
        bounded_anchors=list(anchors[:result_cap]),
    )


def require_artifact(store: AssociationStore, artifact_id: str) -> None:
    """Raise the shared ``KeyError`` for an unknown association artifact.

    The store owns the check; the arms call it through this free function so
    they never depend on the store's private surface directly.
    """

    store._require_artifact(artifact_id)


def protected_anchor_ids(
    anchors: Sequence[RetrievalResult],
    *,
    lexical_protection_threshold: float | None,
) -> tuple[str, ...]:
    """Anchors whose lexical evidence may not be displaced by an association.

    ``None`` disables protection entirely, and an anchor with no lexical score
    was never lexically evidenced, so it is not protected either.
    """

    if lexical_protection_threshold is None:
        return ()
    return tuple(
        result.chunk.chunk_id
        for result in anchors
        if result.lexical_score is not None
        and result.lexical_score >= lexical_protection_threshold
    )


def exceeds_prompt_budget(
    composed: Sequence[RetrievalResult],
    *,
    # Denominator anchors: the arms currently pass different windows over
    # their bounded anchors (open author decision) — see each call site.
    direct_anchors: Sequence[RetrievalResult],
    max_prompt_token_increase: int | None,
) -> bool:
    """True when the composed set overspends the direct-retrieval baseline."""

    if max_prompt_token_increase is None:
        return False
    direct_tokens = sum(result.chunk.token_count for result in direct_anchors)
    composed_tokens = sum(result.chunk.token_count for result in composed)
    return composed_tokens > direct_tokens + max_prompt_token_increase
