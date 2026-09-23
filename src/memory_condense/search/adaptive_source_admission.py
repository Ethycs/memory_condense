"""Deterministic source-balanced surplus admission over ranked evidence lanes.

The selector is deliberately additive.  A caller supplies an already accepted
parent packet, whose exact evidence identities and order are immutable, plus
ranked candidate lanes.  Source identifiers are opaque equality keys: they are
never parsed or compared with question identifiers.

Only ranks affect priority.  Raw route scores are intentionally ignored because
scores from lexical, dense, and specialist lanes do not share a scale.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Generic, Literal, TypeVar


T = TypeVar("T")

POLICY_ID = "parent-protected-opaque-source-bounded-rrf-v1"


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _positive_int(value: object, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


@dataclass(frozen=True, slots=True)
class RankedSourceLane(Generic[T]):
    """One ordered candidate stream; position is its only strength signal."""

    lane_id: str
    candidates: Sequence[T]

    def __post_init__(self) -> None:
        _nonempty(self.lane_id, "lane_id")
        object.__setattr__(self, "candidates", tuple(self.candidates))


@dataclass(frozen=True, slots=True)
class CandidateOrigin:
    """One exact route/rank occurrence contributing to a candidate."""

    lane_id: str
    rank: int


@dataclass(frozen=True, slots=True)
class EvidenceAdmissionAudit:
    """Stable identity, provenance, and phase for one surplus item."""

    evidence_id: str
    source_id: str
    phase: Literal["new_source_representative", "source_round_robin_surplus"]
    candidate_rrf: str
    origins: tuple[CandidateOrigin, ...]


@dataclass(frozen=True, slots=True)
class SourceGroupAudit:
    """Bounded source-level support and the exact selected partition."""

    source_id: str
    source_rrf: str
    contributing_hit_count: int
    contributing_lane_count: int
    candidate_evidence_ids: tuple[str, ...]
    selected_evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceSurplusAudit:
    """Complete deterministic receipt for one source-balanced selection."""

    policy_id: str
    surplus_budget: int
    rrf_constant: int
    max_hits_per_source_per_lane: int
    parent_evidence_ids: tuple[str, ...]
    parent_source_ids: tuple[str, ...]
    ranked_new_source_ids: tuple[str, ...]
    selected: tuple[EvidenceAdmissionAudit, ...]
    source_groups: tuple[SourceGroupAudit, ...]
    unfilled_slots: int

    def projection(self) -> dict[str, Any]:
        """Return a JSON-compatible audit without depending on object reprs."""

        return {
            "policy_id": self.policy_id,
            "surplus_budget": self.surplus_budget,
            "rrf_constant": self.rrf_constant,
            "max_hits_per_source_per_lane": self.max_hits_per_source_per_lane,
            "parent_evidence_ids": list(self.parent_evidence_ids),
            "parent_source_ids": list(self.parent_source_ids),
            "ranked_new_source_ids": list(self.ranked_new_source_ids),
            "selected": [
                {
                    "evidence_id": row.evidence_id,
                    "source_id": row.source_id,
                    "phase": row.phase,
                    "candidate_rrf": row.candidate_rrf,
                    "origins": [
                        {"lane_id": origin.lane_id, "rank": origin.rank}
                        for origin in row.origins
                    ],
                }
                for row in self.selected
            ],
            "source_groups": [
                {
                    "source_id": row.source_id,
                    "source_rrf": row.source_rrf,
                    "contributing_hit_count": row.contributing_hit_count,
                    "contributing_lane_count": row.contributing_lane_count,
                    "candidate_evidence_ids": list(row.candidate_evidence_ids),
                    "selected_evidence_ids": list(row.selected_evidence_ids),
                }
                for row in self.source_groups
            ],
            "unfilled_slots": self.unfilled_slots,
        }


@dataclass(frozen=True, slots=True)
class SourceSurplusSelection(Generic[T]):
    """Protected parent items followed by the bounded admitted surplus."""

    items: tuple[T, ...]
    evidence_ids: tuple[str, ...]
    parent_items: tuple[T, ...]
    surplus_items: tuple[T, ...]
    audit: SourceSurplusAudit


@dataclass(slots=True)
class _Candidate(Generic[T]):
    item: T
    evidence_id: str
    source_id: str
    origins: list[CandidateOrigin]
    rrf: Fraction


@dataclass(slots=True)
class _SourceGroup(Generic[T]):
    source_id: str
    candidates: list[_Candidate[T]]
    rrf: Fraction
    contributing_hits: int
    contributing_lanes: set[str]
    best_rank: int
    first_lane_index: int
    first_rank: int


def source_balanced_surplus_admission(
    parent_items: Sequence[T],
    lanes: Sequence[RankedSourceLane[T]],
    *,
    evidence_id: Callable[[T], str],
    source_id: Callable[[T], str],
    surplus_budget: int,
    rrf_constant: int = 60,
    max_hits_per_source_per_lane: int = 4,
) -> SourceSurplusSelection[T]:
    """Add rank-fused source coverage without changing the parent packet.

    Source-group RRF receives at most ``max_hits_per_source_per_lane``
    contributions from any one lane.  This prevents a long source from winning
    solely because it produced many chunks.  Duplicate exact evidence across
    different lanes accumulates candidate support, but the first lane/rank
    occurrence owns the returned object and the evidence is emitted once.

    One representative is admitted from every reachable source absent from the
    parent before any source receives another surplus item.  If the item budget
    cannot cover all new sources, the highest bounded source-RRF groups win.
    Remaining capacity is filled by deterministic round robin over all groups.
    """

    budget = _nonnegative_int(surplus_budget, "surplus_budget")
    constant = _nonnegative_int(rrf_constant, "rrf_constant")
    per_source_lane_cap = _positive_int(
        max_hits_per_source_per_lane,
        "max_hits_per_source_per_lane",
    )
    ordered_lanes = tuple(lanes)
    lane_ids: set[str] = set()
    for lane in ordered_lanes:
        if not isinstance(lane, RankedSourceLane):
            raise TypeError("lanes must contain RankedSourceLane values")
        if lane.lane_id in lane_ids:
            raise ValueError(f"duplicate lane_id: {lane.lane_id}")
        lane_ids.add(lane.lane_id)

    parents = tuple(parent_items)
    parent_ids: list[str] = []
    parent_sources: list[str] = []
    source_by_evidence: dict[str, str] = {}
    for item in parents:
        item_id = _nonempty(evidence_id(item), "evidence_id")
        item_source = _nonempty(source_id(item), "source_id")
        if item_id in source_by_evidence:
            raise ValueError(f"duplicate parent evidence_id: {item_id}")
        source_by_evidence[item_id] = item_source
        parent_ids.append(item_id)
        if item_source not in parent_sources:
            parent_sources.append(item_source)

    candidates: dict[str, _Candidate[T]] = {}
    source_groups: dict[str, _SourceGroup[T]] = {}
    per_lane_source_hits: dict[tuple[str, str], int] = {}
    for lane_index, lane in enumerate(ordered_lanes):
        seen_in_lane: set[str] = set()
        for rank, item in enumerate(lane.candidates, 1):
            item_id = _nonempty(evidence_id(item), "evidence_id")
            item_source = _nonempty(source_id(item), "source_id")
            if item_id in seen_in_lane:
                raise ValueError(
                    f"lane {lane.lane_id!r} repeats evidence_id: {item_id}"
                )
            seen_in_lane.add(item_id)
            known_source = source_by_evidence.setdefault(item_id, item_source)
            if known_source != item_source:
                raise ValueError(
                    f"evidence_id {item_id!r} maps to conflicting sources"
                )
            origin = CandidateOrigin(lane.lane_id, rank)
            contribution = Fraction(1, constant + rank)
            candidate = candidates.get(item_id)
            if candidate is None:
                candidate = _Candidate(
                    item=item,
                    evidence_id=item_id,
                    source_id=item_source,
                    origins=[],
                    rrf=Fraction(),
                )
                candidates[item_id] = candidate
            candidate.origins.append(origin)
            candidate.rrf += contribution

            group = source_groups.get(item_source)
            if group is None:
                group = _SourceGroup(
                    source_id=item_source,
                    candidates=[],
                    rrf=Fraction(),
                    contributing_hits=0,
                    contributing_lanes=set(),
                    best_rank=rank,
                    first_lane_index=lane_index,
                    first_rank=rank,
                )
                source_groups[item_source] = group
            group.best_rank = min(group.best_rank, rank)
            group_key = (lane.lane_id, item_source)
            counted = per_lane_source_hits.get(group_key, 0)
            if counted < per_source_lane_cap:
                group.rrf += contribution
                group.contributing_hits += 1
                group.contributing_lanes.add(lane.lane_id)
                per_lane_source_hits[group_key] = counted + 1

    for candidate in candidates.values():
        source_groups[candidate.source_id].candidates.append(candidate)

    lane_order = {
        lane.lane_id: lane_index for lane_index, lane in enumerate(ordered_lanes)
    }

    def candidate_key(row: _Candidate[T]) -> tuple[Any, ...]:
        first = row.origins[0]
        return (
            -row.rrf,
            min(origin.rank for origin in row.origins),
            lane_order[first.lane_id],
            first.rank,
            row.evidence_id,
        )

    for group in source_groups.values():
        group.candidates.sort(key=candidate_key)

    def source_key(group: _SourceGroup[T]) -> tuple[Any, ...]:
        return (
            -group.rrf,
            -len(group.contributing_lanes),
            group.best_rank,
            group.first_lane_index,
            group.first_rank,
            group.source_id,
        )

    ranked_groups = sorted(source_groups.values(), key=source_key)
    parent_id_set = set(parent_ids)
    represented_sources = set(parent_sources)
    queues = {
        group.source_id: [
            candidate
            for candidate in group.candidates
            if candidate.evidence_id not in parent_id_set
        ]
        for group in ranked_groups
    }
    ranked_new_groups = [
        group
        for group in ranked_groups
        if group.source_id not in represented_sources and queues[group.source_id]
    ]

    admitted: list[_Candidate[T]] = []
    phases: list[
        Literal["new_source_representative", "source_round_robin_surplus"]
    ] = []
    selected_ids = set(parent_ids)
    for group in ranked_new_groups:
        if len(admitted) >= budget:
            break
        candidate = queues[group.source_id].pop(0)
        admitted.append(candidate)
        phases.append("new_source_representative")
        selected_ids.add(candidate.evidence_id)

    # A second item from any source is considered only after every reachable
    # parent-absent source has received its representative.  If the first pass
    # stopped on budget, no second-pass candidate is eligible.
    if len(admitted) < budget and len(admitted) == len(ranked_new_groups):
        while len(admitted) < budget:
            advanced = False
            for group in ranked_groups:
                queue = queues[group.source_id]
                while queue and queue[0].evidence_id in selected_ids:
                    queue.pop(0)
                if not queue:
                    continue
                candidate = queue.pop(0)
                admitted.append(candidate)
                phases.append("source_round_robin_surplus")
                selected_ids.add(candidate.evidence_id)
                advanced = True
                if len(admitted) >= budget:
                    break
            if not advanced:
                break

    admission_rows = tuple(
        EvidenceAdmissionAudit(
            evidence_id=candidate.evidence_id,
            source_id=candidate.source_id,
            phase=phase,
            candidate_rrf=_fraction_text(candidate.rrf),
            origins=tuple(candidate.origins),
        )
        for candidate, phase in zip(admitted, phases, strict=True)
    )
    selected_by_source: dict[str, list[str]] = {}
    for candidate in admitted:
        selected_by_source.setdefault(candidate.source_id, []).append(
            candidate.evidence_id
        )
    group_rows = tuple(
        SourceGroupAudit(
            source_id=group.source_id,
            source_rrf=_fraction_text(group.rrf),
            contributing_hit_count=group.contributing_hits,
            contributing_lane_count=len(group.contributing_lanes),
            candidate_evidence_ids=tuple(
                candidate.evidence_id for candidate in group.candidates
            ),
            selected_evidence_ids=tuple(
                selected_by_source.get(group.source_id, ())
            ),
        )
        for group in ranked_groups
    )
    audit = SourceSurplusAudit(
        policy_id=POLICY_ID,
        surplus_budget=budget,
        rrf_constant=constant,
        max_hits_per_source_per_lane=per_source_lane_cap,
        parent_evidence_ids=tuple(parent_ids),
        parent_source_ids=tuple(parent_sources),
        ranked_new_source_ids=tuple(group.source_id for group in ranked_new_groups),
        selected=admission_rows,
        source_groups=group_rows,
        unfilled_slots=budget - len(admitted),
    )
    surplus = tuple(candidate.item for candidate in admitted)
    return SourceSurplusSelection(
        items=parents + surplus,
        evidence_ids=tuple(parent_ids) + tuple(row.evidence_id for row in admitted),
        parent_items=parents,
        surplus_items=surplus,
        audit=audit,
    )


__all__ = [
    "CandidateOrigin",
    "EvidenceAdmissionAudit",
    "POLICY_ID",
    "RankedSourceLane",
    "SourceGroupAudit",
    "SourceSurplusAudit",
    "SourceSurplusSelection",
    "source_balanced_surplus_admission",
]
