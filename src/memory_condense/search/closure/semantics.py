"""Pure matching and graph-semantics transformations for evidence closure."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

from memory_condense.domain.discourse import (
    DiscourseRelation,
    DiscourseUnit,
    EvidenceObligation,
    QueryProgram,
    canonical_json,
)
from memory_condense.domain.discourse_routing import DiscourseUnitRoute


_RoutableUnit = DiscourseUnit | DiscourseUnitRoute


REVISION_RELATIONS = frozenset(
    {
        "revises",
        "supersedes",
        "retracts",
        "amends",
        "replaces",
        "corrects",
        "updates",
    }
)
CONFLICT_RELATIONS = frozenset(
    {"contradicts", "conflicts_with", "opposes", "inconsistent_with"}
)
RESOLUTION_RELATIONS = frozenset(
    {"resolves", "accepts", "rejects", "qualifies", "settles", "chooses"}
)
DEPENDENCY_RELATIONS = frozenset(
    {"depends_on", "requires", "blocked_by", "prerequisite_for"}
)
TEST_RESULT_RELATIONS = frozenset(
    {
        "tests",
        "validates",
        "evaluates",
        "checks",
        "produces",
        "results_in",
        "measures",
        "observes",
        "causes",
        "implements",
        "addresses",
    }
)
SUPPORT_RELATIONS = frozenset({"supports", "qualifies"})
CORE_CLOSURE_RELATIONS = (
    REVISION_RELATIONS
    | CONFLICT_RELATIONS
    | RESOLUTION_RELATIONS
    | DEPENDENCY_RELATIONS
    | TEST_RESULT_RELATIONS
    | SUPPORT_RELATIONS
)

_OLD_ROLES = frozenset(
    {"old", "previous", "prior", "original", "revised", "superseded", "retracted", "target", "predecessor"}
)
_NEW_ROLES = frozenset(
    {"new", "current", "replacement", "revision", "reviser", "successor", "amendment", "source"}
)
_TERM_RE = re.compile(r"[^\W_]+", re.UNICODE)


def normalize_label(value: str) -> str:
    """Normalize open-vocabulary type labels without changing their ontology."""

    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def unit_obligation_ids(
    unit: _RoutableUnit,
    program: QueryProgram,
    *,
    connected: bool,
) -> tuple[str, ...]:
    """Return obligations a unit can discharge under one evidenced route."""

    if program.as_of_ordinal is not None and unit.asserted_ordinal > program.as_of_ordinal:
        return ()
    matches: list[str] = []
    for obligation in program.obligations:
        allowed = {normalize_label(value) for value in obligation.unit_kinds}
        if not allowed:
            continue
        if "any" not in allowed and "*" not in allowed:
            if normalize_label(unit.kind) not in allowed:
                continue
        if not connected and not subject_matches(
            unit,
            obligation,
            default_terms=program.subject_terms,
        ):
            continue
        matches.append(obligation.obligation_id)
    return tuple(matches)


def relation_obligation_ids(
    relation: DiscourseRelation,
    program: QueryProgram,
) -> tuple[str, ...]:
    """Return obligations named by a typed, evidenced relation."""

    if program.as_of_ordinal is not None and relation.created_ordinal > program.as_of_ordinal:
        return ()
    relation_type = normalize_label(relation.relation_type)
    return tuple(
        obligation.obligation_id
        for obligation in program.obligations
        if relation_type
        in {normalize_label(value) for value in obligation.relation_types}
    )


def subject_matches(
    unit: _RoutableUnit,
    obligation: EvidenceObligation,
    *,
    default_terms: Sequence[str] = (),
) -> bool:
    """Conservatively match explicit subject terms against routing metadata."""

    terms = obligation.subject_terms or tuple(default_terms)
    if not terms:
        return True
    haystack = " ".join(
        (
            unit.canonical_key.casefold(),
            unit.kind.casefold(),
            canonical_json(unit.metadata).casefold(),
        )
    )
    hay_tokens = set(_TERM_RE.findall(haystack))
    for subject in terms:
        phrase = str(subject).casefold().strip()
        if phrase and phrase in haystack:
            return True
        tokens = [token for token in _TERM_RE.findall(phrase) if len(token) >= 2]
        if tokens and all(token in hay_tokens for token in tokens):
            return True
    return False


def relation_is_useful(
    relation: DiscourseRelation,
    program: QueryProgram,
    units: Mapping[str, DiscourseUnit],
    *,
    connected: bool,
) -> bool:
    """Whether an edge can close or disambiguate a live obligation."""

    relation_type = normalize_label(relation.relation_type)
    if relation_type in CORE_CLOSURE_RELATIONS:
        return True
    if relation_obligation_ids(relation, program):
        return True
    return any(
        unit_obligation_ids(units[member.unit_id], program, connected=connected)
        for member in relation.members
        if member.unit_id in units
    )


def relation_confers_subject_connection(
    relation: DiscourseRelation,
    program: QueryProgram,
    *,
    min_confidence: float,
) -> bool:
    """Whether an edge is strong and semantically anchors its peer units.

    Pure contiguity edges such as ``sequence`` and ``reply_to`` remain routing
    hints.  They may widen the walk, but they cannot make an otherwise
    unrelated unit satisfy an obligation merely by being nearby.
    """

    if relation.confidence < min_confidence:
        return False
    relation_type = normalize_label(relation.relation_type)
    return (
        relation_type in CORE_CLOSURE_RELATIONS
        or bool(relation_obligation_ids(relation, program))
    )


def relation_priority(
    relation: DiscourseRelation,
    program: QueryProgram,
    units: Mapping[str, DiscourseUnit],
    *,
    min_confidence: float,
) -> tuple[int, int, int, float, int, str]:
    """Deterministic best-first relation priority; lower tuples win."""

    obligation_gain = set(relation_obligation_ids(relation, program))
    for member in relation.members:
        unit = units.get(member.unit_id)
        if unit is not None:
            obligation_gain.update(unit_obligation_ids(unit, program, connected=True))
    semantic = normalize_label(relation.relation_type) in CORE_CLOSURE_RELATIONS
    weak = relation.confidence < min_confidence
    return (
        -len(obligation_gain),
        0 if semantic else 1,
        1 if weak else 0,
        -float(relation.confidence),
        -int(relation.created_ordinal),
        relation.relation_id,
    )


def unit_priority(
    unit: _RoutableUnit,
    program: QueryProgram,
    *,
    connected: bool,
    route_rank: int,
) -> tuple[int, int, float, int, str]:
    """Deterministic best-first unit priority; lower tuples win."""

    gain = len(unit_obligation_ids(unit, program, connected=connected))
    return (
        route_rank,
        -gain,
        -float(unit.confidence),
        -int(unit.asserted_ordinal),
        unit.unit_id,
    )


def revision_successors(
    relations: Iterable[DiscourseRelation],
    units: Mapping[str, DiscourseUnit],
) -> dict[str, tuple[str, ...]]:
    """Orient revision chains from superseded units to their replacements."""

    successors: dict[str, set[str]] = {}
    for relation in relations:
        if normalize_label(relation.relation_type) not in REVISION_RELATIONS:
            continue
        members = [member for member in relation.members if member.unit_id in units]
        old = [member.unit_id for member in members if normalize_label(member.role) in _OLD_ROLES]
        new = [member.unit_id for member in members if normalize_label(member.role) in _NEW_ROLES]
        if not old or not new:
            ordered = sorted(
                {member.unit_id for member in members},
                key=lambda unit_id: (units[unit_id].asserted_ordinal, unit_id),
            )
            if len(ordered) >= 2:
                old = ordered[:-1]
                new = ordered[-1:]
        for predecessor in old:
            successors.setdefault(predecessor, set()).update(
                successor for successor in new if successor != predecessor
            )
    return {
        unit_id: tuple(sorted(values))
        for unit_id, values in sorted(successors.items())
    }


def terminal_unit_ids(
    unit_ids: Sequence[str],
    successors: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    """Keep only revision terminals; never resurrect a known predecessor."""

    terminal = [
        unit_id
        for unit_id in unit_ids
        if not successors.get(unit_id)
    ]
    return tuple(terminal)


def relation_resolves_conflict(
    candidate: DiscourseRelation,
    conflict: DiscourseRelation,
) -> bool:
    """Require an explicit link or evidence covering every conflict side."""

    if normalize_label(candidate.relation_type) not in RESOLUTION_RELATIONS:
        return False
    if candidate.created_ordinal < conflict.created_ordinal:
        return False
    linked = candidate.metadata.get("resolved_relation_id")
    if isinstance(linked, str) and linked == conflict.relation_id:
        return True
    linked_many = candidate.metadata.get("resolved_relation_ids", ())
    if (
        isinstance(linked_many, Sequence)
        and not isinstance(linked_many, (str, bytes))
        and conflict.relation_id in linked_many
    ):
        return True
    conflict_members = {member.unit_id for member in conflict.members}
    resolution_members = {member.unit_id for member in candidate.members}
    return bool(conflict_members) and conflict_members <= resolution_members


def unresolved_conflicts(
    relations: Sequence[DiscourseRelation],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return conflict edges lacking a later, overlapping resolution edge."""

    resolutions = [
        relation
        for relation in relations
        if normalize_label(relation.relation_type) in RESOLUTION_RELATIONS
    ]
    unresolved: list[tuple[str, tuple[str, ...]]] = []
    for relation in relations:
        if normalize_label(relation.relation_type) not in CONFLICT_RELATIONS:
            continue
        members = tuple(sorted({member.unit_id for member in relation.members}))
        resolved = any(
            relation_resolves_conflict(candidate, relation)
            for candidate in resolutions
        )
        if not resolved:
            unresolved.append((relation.relation_id, members))
    return tuple(sorted(unresolved))


def relation_components(
    relations: Sequence[DiscourseRelation],
) -> tuple[tuple[DiscourseRelation, ...], ...]:
    """Group overlapping n-ary edges into deterministic atomic closures."""

    remaining = {relation.relation_id: relation for relation in relations}
    components: list[tuple[DiscourseRelation, ...]] = []
    while remaining:
        first_id = min(remaining)
        queue = [remaining.pop(first_id)]
        component: list[DiscourseRelation] = []
        member_ids: set[str] = set()
        while queue:
            relation = queue.pop(0)
            component.append(relation)
            member_ids.update(member.unit_id for member in relation.members)
            touching = sorted(
                relation_id
                for relation_id, candidate in remaining.items()
                if member_ids & {member.unit_id for member in candidate.members}
            )
            for relation_id in touching:
                queue.append(remaining.pop(relation_id))
        components.append(tuple(sorted(component, key=lambda item: item.relation_id)))
    return tuple(components)


__all__ = [
    "CONFLICT_RELATIONS",
    "CORE_CLOSURE_RELATIONS",
    "DEPENDENCY_RELATIONS",
    "RESOLUTION_RELATIONS",
    "REVISION_RELATIONS",
    "SUPPORT_RELATIONS",
    "TEST_RESULT_RELATIONS",
    "normalize_label",
    "relation_components",
    "relation_confers_subject_connection",
    "relation_is_useful",
    "relation_obligation_ids",
    "relation_priority",
    "relation_resolves_conflict",
    "revision_successors",
    "subject_matches",
    "terminal_unit_ids",
    "unit_obligation_ids",
    "unit_priority",
    "unresolved_conflicts",
]
