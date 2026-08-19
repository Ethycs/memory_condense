"""Atomic, exact-span evidence-bundle assembly."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from memory_condense.domain.discourse import (
    ClosurePolicy,
    DiscourseRelation,
    DiscourseUnit,
    EvidenceAtom,
    EvidenceBundle,
    EvidenceSpan,
    QueryProgram,
    evidence_span_sort_key,
    identity_sha256,
    make_atom_id,
    make_bundle_id,
)
from memory_condense.search.closure.semantics import (
    CONFLICT_RELATIONS,
    DEPENDENCY_RELATIONS,
    RESOLUTION_RELATIONS,
    REVISION_RELATIONS,
    TEST_RESULT_RELATIONS,
    normalize_label,
    relation_components,
    relation_obligation_ids,
    relation_resolves_conflict,
)
from memory_condense.search.closure.store import EvidenceClosureStore


@dataclass(frozen=True, slots=True)
class BundleAssembly:
    atoms: tuple[EvidenceAtom, ...]
    bundles: tuple[EvidenceBundle, ...]
    unit_bundle_ids: Mapping[str, tuple[str, ...]]
    relation_bundle_ids: Mapping[str, tuple[str, ...]]
    truncated: bool
    omitted_direct_chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _Candidate:
    category: int
    bundle: EvidenceBundle


class _AtomRegistry:
    __slots__ = ("_atoms", "_store")

    def __init__(self, store: EvidenceClosureStore) -> None:
        self._store = store
        self._atoms: dict[str, EvidenceAtom] = {}

    def add(self, span: EvidenceSpan, *, label: str) -> str:
        atom_id = make_atom_id(span)
        if atom_id not in self._atoms:
            self._atoms[atom_id] = EvidenceAtom(
                atom_id=atom_id,
                span=span,
                text=self._store.hydrate_span(span),
                label=label,
                role=span.role,
                created_at=span.created_at,
            )
        return atom_id

    def selected(self, atom_ids: set[str]) -> tuple[EvidenceAtom, ...]:
        return tuple(
            sorted(
                (self._atoms[atom_id] for atom_id in atom_ids),
                key=lambda atom: _span_key(atom.span) + (atom.atom_id,),
            )
        )


def assemble_bundles(
    store: EvidenceClosureStore,
    *,
    program: QueryProgram,
    policy: ClosurePolicy,
    raw_spans: Sequence[EvidenceSpan],
    units: Mapping[str, DiscourseUnit],
    relations: Mapping[str, DiscourseRelation],
    unit_obligations: Mapping[str, tuple[str, ...]],
    credited_relation_ids: set[str],
) -> BundleAssembly:
    """Build and cap atomic bundles, always prioritizing direct raw hits."""

    registry = _AtomRegistry(store)
    required_ids = {
        obligation.obligation_id
        for obligation in program.obligations
        if obligation.required
    }
    weights = {
        obligation.obligation_id: obligation.weight
        for obligation in program.obligations
    }
    candidates: list[_Candidate] = []

    units_by_chunk: dict[str, list[DiscourseUnit]] = defaultdict(list)
    for unit in units.values():
        for chunk_id in {span.chunk_id for span in unit.evidence}:
            units_by_chunk[chunk_id].append(unit)
    # Raw hits are never removed merely because graph annotation is absent or
    # weak. Direct bundles sort first; any hard-cap omission is explicit in the
    # assembly receipt returned below.  A direct bundle may credit a unit only
    # when it contains every exact evidence span owned by that unit.  Relations
    # remain in their dedicated atomic groups: a relation citation alone is not
    # proof of all member claims, revision sides, or contradiction sides.
    raw_by_chunk: dict[str, list[EvidenceSpan]] = defaultdict(list)
    direct_chunk_by_bundle_id: dict[str, str] = {}
    for span in raw_spans:
        raw_by_chunk[span.chunk_id].append(span)
    for chunk_id in sorted(raw_by_chunk):
        direct_spans = tuple(sorted(raw_by_chunk[chunk_id], key=_span_key))
        atom_ids = tuple(
            registry.add(span, label=f"direct:{chunk_id}")
            for span in direct_spans
        )
        direct_span_ids = {
            identity_sha256(span.identity_payload()) for span in direct_spans
        }
        chunk_units = sorted(
            (
                unit
                for unit in units_by_chunk.get(chunk_id, ())
                if all(
                    identity_sha256(span.identity_payload()) in direct_span_ids
                    for span in unit.evidence
                )
            ),
            key=lambda item: item.unit_id,
        )
        obligation_ids = _ordered_obligations(
            program,
            {
                obligation_id
                for unit in chunk_units
                for obligation_id in unit_obligations.get(unit.unit_id, ())
            },
        )
        bundle = _bundle(
            atom_ids=atom_ids,
            obligation_ids=obligation_ids,
            unit_ids=tuple(item.unit_id for item in chunk_units),
            relation_ids=(),
            required=bool(required_ids & set(obligation_ids)),
            utility=_utility(obligation_ids, weights, chunk_units, ()),
        )
        candidates.append(_Candidate(category=0, bundle=bundle))
        direct_chunk_by_bundle_id[bundle.bundle_id] = chunk_id

    covered_units: set[str] = set()
    proof_relations = tuple(
        relation
        for relation_id, relation in relations.items()
        if relation_id in credited_relation_ids
    )
    for component in _atomic_relation_groups(proof_relations):
        member_ids = tuple(
            sorted(
                {
                    member.unit_id
                    for relation in component
                    for member in relation.members
                    if member.unit_id in units
                }
            )
        )
        component_units = [units[unit_id] for unit_id in member_ids]
        obligation_ids = _ordered_obligations(
            program,
            {
                obligation_id
                for unit_id in member_ids
                for obligation_id in unit_obligations.get(unit_id, ())
            }
            | {
                obligation_id
                for relation in component
                for obligation_id in relation_obligation_ids(relation, program)
            },
        )
        spans = _unique_spans(
            span
            for unit in component_units
            for span in unit.evidence
        ) + _unique_spans(
            span
            for relation in component
            for span in relation.evidence
        )
        atom_ids = tuple(
            registry.add(span, label=_span_label(span, component_units, component))
            for span in _unique_spans(spans)
        )
        if not atom_ids:
            continue
        relation_ids = tuple(relation.relation_id for relation in component)
        bundle = _bundle(
            atom_ids=atom_ids,
            obligation_ids=obligation_ids,
            unit_ids=member_ids,
            relation_ids=relation_ids,
            required=bool(required_ids & set(obligation_ids)),
            utility=_utility(obligation_ids, weights, component_units, component),
        )
        candidates.append(_Candidate(category=1, bundle=bundle))
        covered_units.update(member_ids)

    # Weak edges are routing hypotheses only.  Hydrate their exact spans for
    # inspection, but do not expose relation/unit IDs or obligation credit to
    # the answer packet.  Negative utility keeps an ordinary packer from
    # selecting this diagnostic bundle over grounded answer evidence.
    for relation_id in sorted(set(relations) - credited_relation_ids):
        relation = relations[relation_id]
        member_units = [
            units[member.unit_id]
            for member in relation.members
            if member.unit_id in units
        ]
        spans = _unique_spans(
            span for unit in member_units for span in unit.evidence
        ) + _unique_spans(relation.evidence)
        atom_ids = tuple(
            registry.add(
                span,
                label=f"routing-hypothesis:{relation.relation_type}:{relation_id}",
            )
            for span in _unique_spans(spans)
        )
        if not atom_ids:
            continue
        candidates.append(
            _Candidate(
                category=3,
                bundle=_bundle(
                    atom_ids=atom_ids,
                    obligation_ids=(),
                    unit_ids=(),
                    relation_ids=(),
                    required=False,
                    utility=-1.0,
                ),
            )
        )

    for unit_id, unit in sorted(units.items()):
        obligation_ids = _ordered_obligations(
            program, set(unit_obligations.get(unit_id, ()))
        )
        if unit_id in covered_units or not obligation_ids:
            continue
        atom_ids = tuple(
            registry.add(span, label=f"unit:{unit.kind}:{unit.unit_id}")
            for span in sorted(unit.evidence, key=_span_key)
        )
        bundle = _bundle(
            atom_ids=atom_ids,
            obligation_ids=obligation_ids,
            unit_ids=(unit_id,),
            relation_ids=(),
            required=bool(required_ids & set(obligation_ids)),
            utility=_utility(obligation_ids, weights, (unit,), ()),
        )
        candidates.append(_Candidate(category=2, bundle=bundle))

    deduplicated: dict[str, _Candidate] = {}
    for candidate in candidates:
        existing = deduplicated.get(candidate.bundle.bundle_id)
        if existing is None or _candidate_key(candidate) < _candidate_key(existing):
            deduplicated[candidate.bundle.bundle_id] = candidate
    ordered = sorted(deduplicated.values(), key=_candidate_key)
    truncated = len(ordered) > policy.max_bundles
    selected = ordered[: policy.max_bundles]
    bundles = tuple(candidate.bundle for candidate in selected)
    selected_bundle_ids = {bundle.bundle_id for bundle in bundles}
    omitted_direct_chunk_ids = tuple(
        sorted(
            chunk_id
            for bundle_id, chunk_id in direct_chunk_by_bundle_id.items()
            if bundle_id not in selected_bundle_ids
        )
    )

    selected_atom_ids = {
        atom_id for bundle in bundles for atom_id in bundle.atom_ids
    }
    unit_bundle_ids: dict[str, list[str]] = defaultdict(list)
    relation_bundle_ids: dict[str, list[str]] = defaultdict(list)
    for bundle in bundles:
        for unit_id in bundle.unit_ids:
            unit_bundle_ids[unit_id].append(bundle.bundle_id)
        for relation_id in bundle.relation_ids:
            relation_bundle_ids[relation_id].append(bundle.bundle_id)
    return BundleAssembly(
        atoms=registry.selected(selected_atom_ids),
        bundles=bundles,
        unit_bundle_ids={
            key: tuple(values) for key, values in sorted(unit_bundle_ids.items())
        },
        relation_bundle_ids={
            key: tuple(values) for key, values in sorted(relation_bundle_ids.items())
        },
        truncated=truncated,
        omitted_direct_chunk_ids=omitted_direct_chunk_ids,
    )


def _bundle(
    *,
    atom_ids: Sequence[str],
    obligation_ids: Sequence[str],
    unit_ids: Sequence[str],
    relation_ids: Sequence[str],
    required: bool,
    utility: float,
) -> EvidenceBundle:
    return EvidenceBundle(
        bundle_id=make_bundle_id(
            atom_ids=atom_ids,
            obligation_ids=obligation_ids,
            unit_ids=unit_ids,
            relation_ids=relation_ids,
        ),
        atom_ids=tuple(atom_ids),
        obligation_ids=tuple(obligation_ids),
        unit_ids=tuple(unit_ids),
        relation_ids=tuple(relation_ids),
        required=required,
        utility=utility,
    )


def _utility(
    obligation_ids: Sequence[str],
    weights: Mapping[str, float],
    units: Sequence[DiscourseUnit],
    relations: Sequence[DiscourseRelation],
) -> float:
    confidence = sum(unit.confidence for unit in units) + sum(
        relation.confidence for relation in relations
    )
    denominator = len(units) + len(relations)
    return float(sum(weights.get(item, 0.0) for item in obligation_ids)) + (
        confidence / denominator if denominator else 0.0
    )


def _candidate_key(candidate: _Candidate) -> tuple[int, int, float, str]:
    bundle = candidate.bundle
    return (
        candidate.category,
        0 if bundle.required else 1,
        -bundle.utility,
        bundle.bundle_id,
    )


def _ordered_obligations(
    program: QueryProgram,
    selected: set[str],
) -> tuple[str, ...]:
    return tuple(
        obligation.obligation_id
        for obligation in program.obligations
        if obligation.obligation_id in selected
    )


def _unique_spans(spans: Iterable[EvidenceSpan]) -> tuple[EvidenceSpan, ...]:
    values = list(spans)
    unique: dict[str, EvidenceSpan] = {}
    for span in values:
        key = identity_sha256(span.identity_payload())
        unique.setdefault(key, span)
    return tuple(sorted(unique.values(), key=_span_key))


def _span_key(span: EvidenceSpan) -> tuple[object, ...]:
    return evidence_span_sort_key(span)


def _span_label(
    span: EvidenceSpan,
    units: Sequence[DiscourseUnit],
    relations: Sequence[DiscourseRelation],
) -> str:
    for unit in units:
        if span in unit.evidence:
            return f"unit:{unit.kind}:{unit.unit_id}"
    for relation in relations:
        if span in relation.evidence:
            return f"relation:{relation.relation_type}:{relation.relation_id}"
    return f"evidence:{span.chunk_id}"


def _atomic_relation_groups(
    relations: Sequence[DiscourseRelation],
) -> tuple[tuple[DiscourseRelation, ...], ...]:
    """Keep claim closures atomic instead of merging a whole graph component.

    Revision chains, contradiction-plus-resolution sets, and causal/test sets
    each have distinct interpretation requirements.  A shared decision node
    must not fuse all three into one prompt-sized mega-bundle.
    """

    by_id = {relation.relation_id: relation for relation in relations}
    assigned: set[str] = set()
    groups: list[tuple[DiscourseRelation, ...]] = []

    revisions = tuple(
        relation
        for relation in by_id.values()
        if normalize_label(relation.relation_type) in REVISION_RELATIONS
    )
    for component in relation_components(revisions):
        groups.append(component)
        assigned.update(relation.relation_id for relation in component)

    conflicts = sorted(
        (
            relation
            for relation in by_id.values()
            if normalize_label(relation.relation_type) in CONFLICT_RELATIONS
        ),
        key=lambda item: item.relation_id,
    )
    resolutions = tuple(
        relation
        for relation in by_id.values()
        if normalize_label(relation.relation_type) in RESOLUTION_RELATIONS
    )
    used_resolution_ids: set[str] = set()
    for conflict in conflicts:
        if conflict.relation_id in assigned:
            continue
        group = [conflict]
        group.extend(
            relation
            for relation in resolutions
            if relation_resolves_conflict(relation, conflict)
        )
        ordered = tuple(sorted(group, key=lambda item: item.relation_id))
        groups.append(ordered)
        assigned.add(conflict.relation_id)
        used_resolution_ids.update(
            relation.relation_id for relation in ordered if relation is not conflict
        )
    assigned.update(used_resolution_ids)

    causal_types = DEPENDENCY_RELATIONS | TEST_RESULT_RELATIONS
    causal = tuple(
        relation
        for relation in by_id.values()
        if relation.relation_id not in assigned
        and normalize_label(relation.relation_type) in causal_types
    )
    for component in relation_components(causal):
        groups.append(component)
        assigned.update(relation.relation_id for relation in component)

    for relation_id in sorted(set(by_id) - assigned):
        groups.append((by_id[relation_id],))
    return tuple(
        sorted(
            groups,
            key=lambda group: tuple(relation.relation_id for relation in group),
        )
    )


__all__ = ["BundleAssembly", "assemble_bundles"]
