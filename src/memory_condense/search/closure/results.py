"""Transform visited closure evidence into obligation and completion receipts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from memory_condense.domain.discourse import (
    ClosureScopeWitness,
    DiscourseRelation,
    DiscourseUnit,
    EvidenceObligation,
    ObligationResult,
    QueryProgram,
)
from memory_condense.search.closure.bundles import BundleAssembly
from memory_condense.search.closure.semantics import (
    relation_obligation_ids,
    revision_successors,
    terminal_unit_ids,
    unresolved_conflicts,
)


def obligation_results(
    program: QueryProgram,
    *,
    units: Mapping[str, DiscourseUnit],
    relations: Mapping[str, DiscourseRelation],
    unit_obligations: Mapping[str, tuple[str, ...]],
    assembly: BundleAssembly,
    min_relation_confidence: float,
    credited_relation_ids: set[str],
) -> tuple[ObligationResult, ...]:
    """Build one honest result for every compiled evidence obligation."""

    credited_relations = {
        relation_id: relation
        for relation_id, relation in relations.items()
        if relation_id in credited_relation_ids
        and relation.confidence >= min_relation_confidence
    }
    successors = revision_successors(credited_relations.values(), units)
    unresolved = unresolved_conflicts(tuple(credited_relations.values()))
    unresolved_relation_ids = {relation_id for relation_id, _ in unresolved}
    unresolved_member_ids = {
        unit_id for _, member_ids in unresolved for unit_id in member_ids
    }
    bundle_by_id = {bundle.bundle_id: bundle for bundle in assembly.bundles}
    preliminary: dict[str, ObligationResult] = {}

    for obligation in program.obligations:
        unit_ids = [
            unit_id
            for unit_id in units
            if obligation.obligation_id in unit_obligations.get(unit_id, ())
        ]
        unit_ids = _select_units(
            unit_ids,
            units,
            obligation,
            successors,
            ordering=program.ordering,
        )
        relation_ids = [
            relation_id
            for relation_id, relation in credited_relations.items()
            if obligation.obligation_id in relation_obligation_ids(relation, program)
        ]
        relation_ids.sort(
            key=lambda relation_id: (
                relations[relation_id].created_ordinal,
                relation_id,
            )
        )
        if obligation.max_count is not None:
            remaining = max(0, obligation.max_count - len(unit_ids))
            relation_ids = relation_ids[:remaining]

        bundle_ids = _evidencing_bundles(obligation.obligation_id, assembly)
        evidenced_units = [
            unit_id
            for unit_id in unit_ids
            if _has_obligation_bundle(
                assembly.unit_bundle_ids.get(unit_id, ()),
                obligation.obligation_id,
                bundle_by_id,
            )
        ]
        evidenced_relations = [
            relation_id
            for relation_id in relation_ids
            if _has_obligation_bundle(
                assembly.relation_bundle_ids.get(relation_id, ()),
                obligation.obligation_id,
                bundle_by_id,
            )
        ]
        # A typed relation explains membership/causality; it is not an extra
        # answer member. Count evidenced units first and use relations only for
        # genuinely relation-shaped obligations.
        claim_count = (
            len(evidenced_units) if evidenced_units else len(evidenced_relations)
        )
        conflict_affects = bool(
            unresolved_relation_ids & set(relation_ids)
            or unresolved_member_ids & set(unit_ids)
        )
        if obligation.obligation_id == "revisions_conflicts" and unresolved:
            conflict_affects = True

        if conflict_affects:
            status = "conflicted"
            reason = (
                "both contradiction sides were found without a later "
                "evidenced resolution"
            )
        elif claim_count >= obligation.min_count and bundle_ids:
            status = "satisfied"
            reason = None
        elif (unit_ids or relation_ids) and not bundle_ids:
            status = "budget_impossible"
            reason = (
                "matching evidence was found but no atomic bundle survived "
                "the workspace cap"
            )
        elif claim_count:
            status = "not_found"
            reason = (
                f"found {claim_count} evidenced claim(s), below the required "
                f"minimum of {obligation.min_count}"
            )
        else:
            status = "not_found"
            reason = "no evidenced unit or relation discharged the obligation"
        preliminary[obligation.obligation_id] = ObligationResult(
            obligation_id=obligation.obligation_id,
            status=status,
            unit_ids=tuple(evidenced_units),
            relation_ids=tuple(evidenced_relations),
            bundle_ids=bundle_ids,
            reason=reason,
        )

    results: list[ObligationResult] = []
    for obligation in program.obligations:
        result = preliminary[obligation.obligation_id]
        failed_dependencies = tuple(
            dependency
            for dependency in obligation.dependencies
            if preliminary[dependency].status != "satisfied"
        )
        if failed_dependencies and result.status == "satisfied":
            result = ObligationResult(
                obligation_id=result.obligation_id,
                status="not_found",
                unit_ids=result.unit_ids,
                relation_ids=result.relation_ids,
                bundle_ids=result.bundle_ids,
                reason=(
                    "unsatisfied obligation dependencies: "
                    + ", ".join(failed_dependencies)
                ),
            )
        results.append(result)
    return tuple(results)


def _select_units(
    unit_ids: Sequence[str],
    units: Mapping[str, DiscourseUnit],
    obligation: EvidenceObligation,
    successors: Mapping[str, Sequence[str]],
    *,
    ordering: str,
) -> list[str]:
    values = list(dict.fromkeys(unit_ids))
    if obligation.temporal_stance == "terminal":
        values = list(terminal_unit_ids(values, successors))
    elif obligation.temporal_stance == "latest" and values:
        latest = max(units[unit_id].asserted_ordinal for unit_id in values)
        values = [
            unit_id
            for unit_id in values
            if units[unit_id].asserted_ordinal == latest
        ]
    if obligation.temporal_stance == "ordered":
        reverse = ordering == "descending"
    elif obligation.temporal_stance == "ascending":
        reverse = False
    else:
        reverse = True
    values.sort(
        key=lambda unit_id: (units[unit_id].asserted_ordinal, unit_id),
        reverse=reverse,
    )
    if obligation.max_count is not None:
        values = values[: obligation.max_count]
    return values


def _evidencing_bundles(
    obligation_id: str,
    assembly: BundleAssembly,
) -> tuple[str, ...]:
    # A result owns every bundle labelled for its obligation, including
    # superseded/conflicting context that explains why a terminal claim won.
    return tuple(
        bundle.bundle_id
        for bundle in assembly.bundles
        if obligation_id in bundle.obligation_ids
    )


def _has_obligation_bundle(
    bundle_ids: Iterable[str],
    obligation_id: str,
    bundles: Mapping[str, object],
) -> bool:
    return any(
        obligation_id in getattr(bundles[bundle_id], "obligation_ids")
        for bundle_id in bundle_ids
        if bundle_id in bundles
    )


def completion(
    results: Sequence[ObligationResult],
    program: QueryProgram,
    *,
    scope_witnesses: Sequence[ClosureScopeWitness],
) -> tuple[str, bool]:
    """Choose the plan stop reason without overstating bounded scope."""

    result_by_id = {result.obligation_id: result for result in results}
    required = [
        result_by_id[obligation.obligation_id]
        for obligation in program.obligations
        if obligation.required
    ]
    if any(result.status == "conflicted" for result in required):
        return "conflicted", False
    if not scope_witnesses or any(
        not witness.exhaustive for witness in scope_witnesses
    ):
        return "workspace_cap", False
    if all(result.status == "satisfied" for result in required):
        return "complete", True
    if required and all(result.status == "not_found" for result in required):
        return "not_found", False
    return "frontier_exhausted", False


__all__ = ["completion", "obligation_results"]
