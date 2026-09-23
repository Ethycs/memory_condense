"""Deterministic post-selection union for independently budgeted evidence lanes.

Each lane selects against its own fixed budget before cross-lane deduplication.
Duplicate exact evidence IDs then leave vacancies in their original lanes, and
those vacancies are refilled only from that lane's remaining ranked candidates.
No lane can borrow another lane's unused capacity.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RankedEvidenceLane(Generic[T]):
    """One ranked candidate stream with a non-borrowable item budget."""

    lane_id: str
    budget: int
    candidates: Sequence[T]

    def __post_init__(self) -> None:
        if not isinstance(self.lane_id, str) or not self.lane_id:
            raise ValueError("lane_id must be a non-empty string")
        if isinstance(self.budget, bool) or not isinstance(self.budget, int):
            raise TypeError("budget must be an integer")
        if self.budget < 0:
            raise ValueError("budget must be non-negative")
        object.__setattr__(self, "candidates", tuple(self.candidates))


@dataclass(frozen=True, slots=True)
class EvidenceLaneSelection(Generic[T]):
    """Audit of one lane's independent selection, dedup, and refill."""

    lane_id: str
    budget: int
    selected_before_dedup: tuple[T, ...]
    retained_after_dedup: tuple[T, ...]
    dedup_excluded_evidence_ids: tuple[str, ...]
    refilled: tuple[T, ...]
    refill_skipped_evidence_ids: tuple[str, ...]

    @property
    def selected_after_refill(self) -> tuple[T, ...]:
        """Return this lane's final retained rows in lane-local rank order."""

        return self.retained_after_dedup + self.refilled

    @property
    def unfilled_slots(self) -> int:
        """Return capacity that this lane's own ranked stream could not fill."""

        return self.budget - len(self.selected_after_refill)


@dataclass(frozen=True, slots=True)
class EvidenceLaneUnion(Generic[T]):
    """Globally unique evidence plus an ordered audit for every input lane."""

    items: tuple[T, ...]
    evidence_ids: tuple[str, ...]
    lane_selections: tuple[EvidenceLaneSelection[T], ...]


def post_selection_lane_union(
    lanes: Sequence[RankedEvidenceLane[T]],
    *,
    evidence_id: Callable[[T], str],
) -> EvidenceLaneUnion[T]:
    """Union independently selected lanes by exact evidence identity.

    Input lane order is the deterministic conflict priority. The algorithm has
    two phases:

    1. Select ``candidates[:budget]`` independently for every lane, then dedup
       those selections by exact ID. All initial selections are considered
       before any lower-ranked refill candidate.
    2. For each underfilled lane in input order, scan only that lane's tail and
       admit the first globally novel IDs until its original budget is full.

    The returned global item order is retained initial selections followed by
    admitted refills. The helper neither reranks candidates nor normalizes IDs.
    """

    ordered_lanes = tuple(lanes)
    seen_lane_ids: set[str] = set()
    for lane in ordered_lanes:
        if lane.lane_id in seen_lane_ids:
            raise ValueError(f"duplicate lane_id: {lane.lane_id}")
        seen_lane_ids.add(lane.lane_id)

    def checked_id(item: T) -> str:
        value = evidence_id(item)
        if not isinstance(value, str):
            raise TypeError("evidence_id must return a string")
        if not value:
            raise ValueError("evidence_id must return a non-empty string")
        return value

    initial_by_lane = tuple(
        tuple(lane.candidates[: lane.budget]) for lane in ordered_lanes
    )
    retained_by_lane: list[list[T]] = [[] for _lane in ordered_lanes]
    excluded_by_lane: list[list[str]] = [[] for _lane in ordered_lanes]
    refilled_by_lane: list[list[T]] = [[] for _lane in ordered_lanes]
    refill_skips_by_lane: list[list[str]] = [[] for _lane in ordered_lanes]

    claimed_ids: set[str] = set()
    union_items: list[T] = []
    union_ids: list[str] = []

    # Claim the complete independently selected frontier before allowing a
    # lower-ranked refill from any lane to compete for an evidence identity.
    for lane_index, selected in enumerate(initial_by_lane):
        for item in selected:
            item_id = checked_id(item)
            if item_id in claimed_ids:
                excluded_by_lane[lane_index].append(item_id)
                continue
            claimed_ids.add(item_id)
            retained_by_lane[lane_index].append(item)
            union_items.append(item)
            union_ids.append(item_id)

    # Vacancies remain owned by the lane that lost them. Candidate tails are
    # never shared, so an exhausted lane stays underfilled rather than taking
    # another method's surplus.
    for lane_index, lane in enumerate(ordered_lanes):
        retained_count = len(retained_by_lane[lane_index])
        needed = lane.budget - retained_count
        if needed <= 0:
            continue
        for item in lane.candidates[lane.budget :]:
            item_id = checked_id(item)
            if item_id in claimed_ids:
                refill_skips_by_lane[lane_index].append(item_id)
                continue
            claimed_ids.add(item_id)
            refilled_by_lane[lane_index].append(item)
            union_items.append(item)
            union_ids.append(item_id)
            needed -= 1
            if needed == 0:
                break

    selections = tuple(
        EvidenceLaneSelection(
            lane_id=lane.lane_id,
            budget=lane.budget,
            selected_before_dedup=initial_by_lane[lane_index],
            retained_after_dedup=tuple(retained_by_lane[lane_index]),
            dedup_excluded_evidence_ids=tuple(excluded_by_lane[lane_index]),
            refilled=tuple(refilled_by_lane[lane_index]),
            refill_skipped_evidence_ids=tuple(
                refill_skips_by_lane[lane_index]
            ),
        )
        for lane_index, lane in enumerate(ordered_lanes)
    )
    return EvidenceLaneUnion(
        items=tuple(union_items),
        evidence_ids=tuple(union_ids),
        lane_selections=selections,
    )


__all__ = [
    "EvidenceLaneSelection",
    "EvidenceLaneUnion",
    "RankedEvidenceLane",
    "post_selection_lane_union",
]
