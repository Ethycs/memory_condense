"""Coverage reservation, trace assembly, and final selector reporting."""

from __future__ import annotations

import time
from typing import Any

from memory_condense.domain.schemas import RetrievalResult
from memory_condense.search.selectors.coverage_models import CoverageSelectionReport
from memory_condense.search.selectors.evidence_features import (
    _VENUE_QUERY_RE,
    _source_id,
    _timestamp_key,
)
from memory_condense.search.selectors.prefix_models import (
    _PrefixAssignment,
    _PrefixEventCluster,
    _ScoredCoverage,
)
from memory_condense.search.selectors.set_program import SetOrdering, SetQuantifier


def _raw_occurrence_order(
    item: tuple[int, RetrievalResult],
) -> tuple[bool, float, int]:
    """Return deterministic raw-turn order for one event candidate."""

    index, result = item
    created_at = result.turn.created_at if result.turn is not None else None
    try:
        value = float(created_at.timestamp()) if created_at else 0.0
    except (AttributeError, OSError, OverflowError, ValueError):
        value = 0.0
        created_at = None
    return created_at is None, value, index


def reserve_prefix_coverage(
    self: Any,
    scored: _ScoredCoverage,
) -> list[RetrievalResult]:
    """Reserve credible event representatives, trace decisions, and report."""

    prepared = scored.prepared
    started = prepared.started
    query = prepared.query
    program = prepared.program
    unique = prepared.unique
    max_results = prepared.max_results
    timestamps = prepared.timestamps
    active_partition_total = prepared.active_partition_total
    active_partition_inspected = prepared.active_partition_inspected
    active_partition_exhaustive = prepared.active_partition_exhaustive
    normalized_scan_fields = prepared.normalized_scan_fields
    score_provider_fallback = prepared.score_provider_fallback
    score_provider_report = prepared.score_provider_report
    typed_performance_frontier = prepared.typed_performance_frontier
    performance_primary_ids = prepared.performance_primary_ids

    attempted_candidates = scored.attempted_candidates
    inspected_candidates = scored.inspected_candidates
    frontier_batches = scored.frontier_batches
    max_workspace_tokens = scored.max_workspace_tokens
    hits = scored.hits
    semantic_kind_by_id = scored.semantic_kind_by_id
    answerability_by_id = scored.answerability_by_id
    membership_by_id = scored.membership_by_id
    score_by_id = scored.score_by_id
    value_by_id = scored.value_by_id
    semantic_raw_by_id = scored.semantic_raw_by_id
    canonical_answer_object_keys_by_id = (
        scored.canonical_answer_object_keys_by_id
    )
    answer_object_keys_by_id = scored.answer_object_keys_by_id
    clusters = scored.clusters
    uncertain = scored.uncertain
    posterior_uncertain_rows = scored.posterior_uncertain_rows
    null_rows = scored.null_rows
    existing_count = scored.existing_count
    new_count = scored.new_count

    cluster_rows: list[
        tuple[int, _PrefixAssignment, float, float | None, _PrefixEventCluster]
    ] = []
    supporting: list[_PrefixAssignment] = []
    trace_by_id = {
        row["chunk_id"]: row
        for row in self._uninspected_trace(unique, program)
    }

    def role_match(member: _PrefixAssignment) -> float:
        preferred = program.preferred_evidence_role
        if preferred is None:
            return 0.5
        return float(
            member.result.turn is not None
            and member.result.turn.role.casefold() == preferred
        )

    def required_role_match(member: _PrefixAssignment) -> bool:
        required = program.required_evidence_role
        return bool(
            required is not None
            and member.result.turn is not None
            and member.result.turn.role.casefold() == required
        )

    def member_is_credible(member: _PrefixAssignment) -> bool:
        if member.membership_score is not None:
            return (
                member.membership_score
                >= self.explicit_membership_threshold
            )
        return (1.0 - member.p_null) >= self.credible_member_threshold

    def representative_score(member: _PrefixAssignment) -> tuple[float, int]:
        return (
            0.40 * member.value_evidence
            + 0.30 * role_match(member)
            + 0.20 * (1.0 - member.p_null)
            + 0.10 * member.quality,
            -member.index,
        )

    typed_fixed_identity_frontier = bool(
        program.quantifier is SetQuantifier.FIXED
        and _VENUE_QUERY_RE.search(query)
        and any(canonical_answer_object_keys_by_id.values())
    )
    active_structural_contract = bool(
        normalized_scan_fields["active_partition_scan_contract"]
    )

    def is_active_structural_primary(member: _PrefixAssignment) -> bool:
        """Honor an exhaustive scanner's primary/alternative boundary.

        Without a scan, canonical extraction over the bounded route union
        remains the historical fallback.  Once a typed scan is present,
        however, a retrospective recap deliberately admitted as an
        alternative must not become a seventh structural member merely
        because it also names a venue.
        """

        if not active_structural_contract:
            return True
        return member.result.route == "active_partition_structural"

    for cluster_index, cluster in enumerate(clusters, start=1):
        # Prefer a row that actually states a recoverable value.  This
        # keeps a generic anaphoric follow-up from replacing the first
        # answer-bearing occurrence merely because CE ranked it higher.
        required_role_pool = [
            member
            for member in cluster.members
            if program.quantifier is SetQuantifier.FIXED
            and required_role_match(member)
        ]
        representative_pool = required_role_pool or list(cluster.members)
        representative_timestamp: str | None = None
        performance_structural_pool = [
            member
            for member in cluster.members
            if member.result.chunk.chunk_id in performance_primary_ids
        ]
        structural_pool = [
            member
            for member in cluster.members
            if typed_fixed_identity_frontier
            and is_active_structural_primary(member)
            and canonical_answer_object_keys_by_id.get(
                member.result.chunk.chunk_id
            )
            and (
                program.required_evidence_role is None
                or required_role_match(member)
            )
            and member.temporal_in_scope is not False
        ]
        if performance_structural_pool:
            # A later, more query-similar recap from the same source must
            # not replace the first raw occurrence.  Source timestamp is
            # still the event timestamp used for cross-source ordering.
            representative = min(
                performance_structural_pool,
                key=lambda member: _raw_occurrence_order(
                    (member.index, member.result)
                ),
            )
            representative_pool = performance_structural_pool
            representative_timestamp = timestamps.get(
                _source_id(representative.result)
            )
        elif structural_pool:
            # Canonical identity can merge a direct occurrence and later
            # recaps. The event time is the earliest in-scope source time;
            # query direction only orders clusters after this choice.
            # Stable route index resolves equal or missing timestamps.
            representative_pool = structural_pool
            representative = min(
                representative_pool,
                key=lambda member: (
                    _timestamp_key(
                        timestamps.get(_source_id(member.result))
                    )
                    is None,
                    _timestamp_key(
                        timestamps.get(_source_id(member.result))
                    )
                    or 0.0,
                    member.index,
                ),
            )
            representative_timestamp = timestamps.get(
                _source_id(representative.result)
            )
        elif program.ordering is not SetOrdering.NONE:
            timed = [
                (
                    member,
                    timestamps.get(_source_id(member.result)),
                    _timestamp_key(timestamps.get(_source_id(member.result))),
                )
                for member in representative_pool
            ]
            credible_preferred = [
                row
                for row in timed
                if row[2] is not None
                and member_is_credible(row[0])
                and (
                    program.preferred_evidence_role is None
                    or role_match(row[0]) == 1.0
                )
            ]
            credible_any_role = [
                row
                for row in timed
                if row[2] is not None and member_is_credible(row[0])
            ]
            preferred_any_strength = [
                row
                for row in timed
                if row[2] is not None
                and (
                    program.preferred_evidence_role is None
                    or role_match(row[0]) == 1.0
                )
            ]
            occurrence_rows = (
                credible_preferred
                or credible_any_role
                or preferred_any_strength
                or [row for row in timed if row[2] is not None]
            )
            if occurrence_rows:
                # Venue identity denotes the first evidenced visit. Query
                # direction reverses cluster order; it must never replace
                # occurrence time with a later recap timestamp.
                occurrence_key = min(float(row[2]) for row in occurrence_rows)
                occurrence_members = [
                    row
                    for row in occurrence_rows
                    if float(row[2]) == occurrence_key
                ]
                representative_pool = [row[0] for row in occurrence_members]
                representative_timestamp = str(occurrence_members[0][1])
            representative = max(
                representative_pool,
                key=representative_score,
            )
        else:
            representative = max(
                representative_pool,
                key=representative_score,
            )
        representative_id = representative.result.chunk.chunk_id
        if representative_timestamp is None:
            representative_timestamp = timestamps.get(
                _source_id(representative.result)
            )
        priority = (
            0.35 * representative.quality
            + 0.30 * representative.value_evidence
            + 0.20 * min(1.0, representative.semantic_surprisal)
            + 0.15 * role_match(representative)
        )
        cluster_rows.append(
            (
                cluster_index,
                representative,
                priority,
                _timestamp_key(representative_timestamp),
                cluster,
            )
        )
        supporting.extend(
            member
            for member in cluster.members
            if member.result.chunk.chunk_id != representative_id
        )

    def is_credible(
        item: tuple[
            int,
            _PrefixAssignment,
            float,
            float | None,
            _PrefixEventCluster,
        ],
    ) -> bool:
        explicit_membership = [
            member.membership_score
            for member in item[4].members
            if member.membership_score is not None
        ]
        if explicit_membership:
            return max(explicit_membership) >= (
                self.explicit_membership_threshold
            )
        return max(1.0 - member.p_null for member in item[4].members) >= (
            self.credible_member_threshold
        )

    credible_cluster_ids = {
        item[0] for item in cluster_rows if is_credible(item)
    }
    canonical_structural_rows = [
        item
        for item in cluster_rows
        if typed_fixed_identity_frontier
        and is_active_structural_primary(item[1])
        and bool(item[4].answer_object_keys)
        and bool(
            canonical_answer_object_keys_by_id.get(
                item[1].result.chunk.chunk_id
            )
        )
        and (
            program.required_evidence_role is None
            or required_role_match(item[1])
        )
    ]
    performance_structural_rows = [
        item
        for item in cluster_rows
        if typed_performance_frontier
        and item[1].result.chunk.chunk_id in performance_primary_ids
        and item[1].temporal_in_scope is not False
        and (
            program.required_evidence_role is None
            or required_role_match(item[1])
        )
    ]
    structural_eligible_rows = list(canonical_structural_rows)
    structural_eligible_rows.extend(
        item
        for item in performance_structural_rows
        if item[0] not in {row[0] for row in structural_eligible_rows}
    )
    structural_eligible_cluster_ids = {
        item[0] for item in structural_eligible_rows
    }
    performance_structural_cluster_ids = {
        item[0] for item in performance_structural_rows
    }
    structural_reserved_cluster_ids: set[int] = set()
    role_aligned_reserved_cluster_ids: set[int] = set()
    cardinality_deficit = 0
    if program.quantifier is SetQuantifier.FIXED:
        requested_cardinality = program.cardinality or 0
        if typed_fixed_identity_frontier or typed_performance_frontier:
            # The typed route frontier is a deterministic structural
            # hypothesis: stable upstream order establishes which K
            # distinct keyed events were activated. QK/OV utility is only
            # a tie-break, never permission for an untyped false positive
            # to consume one of those slots.
            reservation_rows = sorted(
                structural_eligible_rows,
                key=lambda item: (
                    item[1].index,
                    -item[2],
                    item[0],
                ),
            )[:requested_cardinality]
            structural_reserved_cluster_ids = {
                item[0] for item in reservation_rows
            }
            reserved_cluster_ids = set(structural_reserved_cluster_ids)
            reservation_count = len(reservation_rows)
        else:
            credible_rows = [
                item
                for item in cluster_rows
                if item[0] in credible_cluster_ids
            ]
            if program.required_evidence_role is None:
                reservation_count = min(
                    requested_cardinality,
                    len(credible_rows),
                )
                reserved_cluster_ids = {
                    item[0]
                    for item in sorted(
                        credible_rows,
                        key=lambda item: (
                            -role_match(item[1]),
                            -item[2],
                            item[1].index,
                        ),
                    )[:reservation_count]
                }
            else:
                # A high-confidence retrospective role supplies a bounded
                # FIXED-K frontier without converting the broad preferred
                # role prior into a hard filter.  Fill matching credible
                # clusters first, then stable matching-role hypotheses,
                # and only then credible clusters authored by another
                # role.  Every unreserved row remains in the fail-open
                # tail below.
                matching_credible_rows = [
                    item
                    for item in cluster_rows
                    if any(
                        required_role_match(member)
                        and member_is_credible(member)
                        for member in item[4].members
                    )
                ]
                matching_stable_rows = [
                    item
                    for item in cluster_rows
                    if any(
                        required_role_match(member)
                        for member in item[4].members
                    )
                ]
                cross_role_credible_rows = [
                    item
                    for item in credible_rows
                    if not any(
                        required_role_match(member)
                        for member in item[4].members
                    )
                ]

                reserved_cluster_ids = set()
                for tier, rows in (
                    ("matching_credible", matching_credible_rows),
                    ("matching_stable", matching_stable_rows),
                    ("cross_role_credible", cross_role_credible_rows),
                ):
                    for item in sorted(
                        rows,
                        key=lambda value: (
                            -value[2],
                            value[1].index,
                            value[0],
                        ),
                    ):
                        cluster_id = item[0]
                        if cluster_id in reserved_cluster_ids:
                            continue
                        if len(reserved_cluster_ids) >= requested_cardinality:
                            break
                        reserved_cluster_ids.add(cluster_id)
                        if tier == "matching_stable":
                            role_aligned_reserved_cluster_ids.add(cluster_id)
                    if len(reserved_cluster_ids) >= requested_cardinality:
                        break
                reservation_count = len(reserved_cluster_ids)
        cardinality_deficit = max(
            0,
            requested_cardinality - reservation_count,
        )
    elif program.quantifier is SetQuantifier.SINGLE:
        candidates_for_one = (
            list(performance_structural_rows)
            if performance_structural_rows
            else [item for item in cluster_rows if is_credible(item)]
        )
        if program.ordering is SetOrdering.ASCENDING:
            candidates_for_one.sort(
                key=lambda item: (
                    item[3] is None,
                    item[3] or 0.0,
                    -item[2],
                    item[1].index,
                )
            )
        elif program.ordering is SetOrdering.DESCENDING:
            candidates_for_one.sort(
                key=lambda item: (
                    item[3] is None,
                    -(item[3] or 0.0),
                    -item[2],
                    item[1].index,
                )
            )
        else:
            candidates_for_one.sort(
                key=lambda item: (-item[2], item[1].index)
            )
        reserved_cluster_ids = (
            {candidates_for_one[0][0]} if candidates_for_one else set()
        )
        structural_reserved_cluster_ids = {
            cluster_id
            for cluster_id in reserved_cluster_ids
            if cluster_id in structural_eligible_cluster_ids
        }
    else:
        # ALL and COUNT expose every credible event hypothesis. Weak rows
        # remain fail-open alternatives after the reserved coverage pass.
        # When the typed performance frontier exists, only its direct raw
        # occurrences receive hard prompt reservations. Neural rows such
        # as plans, playlists, and recaps remain fail-open alternatives,
        # but cannot consume the useful-content floor ahead of evidence.
        if performance_structural_rows:
            structural_reserved_cluster_ids = set(
                performance_structural_cluster_ids
            )
            reserved_cluster_ids = set(structural_reserved_cluster_ids)
        else:
            reserved_cluster_ids = {
                item[0]
                for item in cluster_rows
                if item[0] in credible_cluster_ids
            }

    def coverage_order(
        item: tuple[
            int,
            _PrefixAssignment,
            float,
            float | None,
            _PrefixEventCluster,
        ],
    ) -> tuple[Any, ...]:
        structural_tier = (
            0
            if item[0] in structural_reserved_cluster_ids
            else 1
            if typed_performance_frontier
            else 0
        )
        if program.ordering is SetOrdering.ASCENDING:
            return (
                structural_tier,
                item[3] is None,
                item[3] or 0.0,
                -item[2],
                item[1].index,
            )
        if program.ordering is SetOrdering.DESCENDING:
            return (
                structural_tier,
                item[3] is None,
                -(item[3] or 0.0),
                -item[2],
                item[1].index,
            )
        # Semantic surprisal/utility supplies deterministic coverage order
        # when the query does not request a temporal reduction.
        return (
            structural_tier,
            -item[2],
            -item[1].semantic_surprisal,
            item[1].index,
        )

    reserved_rows = sorted(
        [item for item in cluster_rows if item[0] in reserved_cluster_ids],
        key=coverage_order,
    )
    alternative_rows = sorted(
        [item for item in cluster_rows if item[0] not in reserved_cluster_ids],
        key=lambda item: (-item[2], -item[1].semantic_surprisal, item[1].index),
    )
    representative_by_cluster = {
        item[0]: item[1].result.chunk.chunk_id for item in cluster_rows
    }
    for cluster_index, _representative, _priority, _timestamp, cluster in cluster_rows:
        representative_id = representative_by_cluster[cluster_index]
        credible = cluster_index in credible_cluster_ids
        reserved = cluster_index in reserved_cluster_ids
        for member in cluster.members:
            chunk_id = member.result.chunk.chunk_id
            hit = hits[chunk_id]
            trace_by_id[chunk_id].update(
                {
                    "group_id": f"event-{cluster_index}",
                    "group_role": (
                        "representative"
                        if chunk_id == representative_id
                        else "support"
                    ),
                    "qk_score": float(hit.qk_score),
                    "ov_transport": float(hit.ov_transport),
                    "prefix_utility": float(score_by_id.get(chunk_id, 0.0)),
                    "representative_chunk_id": (
                        None if chunk_id == representative_id else representative_id
                    ),
                    "merge_similarity": member.merge_similarity,
                    "merge_threshold": member.merge_threshold,
                    "semantic_score": semantic_raw_by_id.get(chunk_id),
                    "answer_object_key_present": bool(
                        answer_object_keys_by_id.get(chunk_id)
                    ),
                    "semantic_score_kind": semantic_kind_by_id.get(chunk_id),
                    "answerability_score": answerability_by_id.get(chunk_id),
                    "answerability_score_kind": (
                        "forced_choice_explicit_probability"
                        if answerability_by_id.get(chunk_id) is not None
                        else "surface_value_heuristic"
                    ),
                    "membership_score": membership_by_id.get(chunk_id),
                    "preferred_evidence_role": program.preferred_evidence_role,
                    "role_match": (
                        None
                        if program.preferred_evidence_role is None
                        else bool(
                            member.result.turn is not None
                            and member.result.turn.role.casefold()
                            == program.preferred_evidence_role
                        )
                    ),
                    "required_evidence_role": (
                        program.required_evidence_role
                    ),
                    "required_evidence_role_basis": (
                        program.required_evidence_role_basis
                    ),
                    "required_role_match": (
                        None
                        if program.required_evidence_role is None
                        else required_role_match(member)
                    ),
                    "value_evidence": member.value_evidence,
                    "assignment_hypothesis": member.hypothesis,
                    "p_existing": member.p_existing,
                    "p_new": member.p_new,
                    "p_null": member.p_null,
                    "existing_energy": member.existing_energy,
                    "new_energy": member.new_energy,
                    "null_energy": member.null_energy,
                    "temporal_in_scope": member.temporal_in_scope,
                    "posterior_entropy": member.entropy,
                    "semantic_surprisal": member.semantic_surprisal,
                    "posterior_uncertain": (
                        member.entropy >= self.uncertainty_entropy
                    ),
                    "credible_cluster": credible,
                    "coverage_reserved": (
                        reserved and chunk_id == representative_id
                    ),
                    "reservation_basis": (
                        (
                            "direct_performance_frontier"
                            if cluster_index
                            in performance_structural_cluster_ids
                            else "canonical_fixed_frontier"
                        )
                        if (
                            chunk_id == representative_id
                            and cluster_index
                            in structural_reserved_cluster_ids
                        )
                        else (
                            "role_aligned_fixed_frontier"
                            if (
                                chunk_id == representative_id
                                and cluster_index
                                in role_aligned_reserved_cluster_ids
                            )
                            else (
                                "neural_credible"
                                if reserved
                                and chunk_id == representative_id
                                else None
                            )
                        )
                    ),
                }
            )

    for _index, result in uncertain:
        chunk_id = result.chunk.chunk_id
        hit = hits.get(chunk_id)
        trace_by_id[chunk_id].update(
            {
                "group_role": "uncertain",
                "qk_score": (
                    float(hit.qk_score) if hit is not None else None
                ),
                "ov_transport": (
                    float(hit.ov_transport) if hit is not None else None
                ),
                "prefix_utility": score_by_id.get(chunk_id),
                "semantic_score": semantic_raw_by_id.get(chunk_id),
                "answer_object_key_present": bool(
                    answer_object_keys_by_id.get(chunk_id)
                ),
                "semantic_score_kind": semantic_kind_by_id.get(chunk_id),
                "answerability_score": answerability_by_id.get(chunk_id),
                "answerability_score_kind": (
                    "forced_choice_explicit_probability"
                    if answerability_by_id.get(chunk_id) is not None
                    else "surface_value_heuristic"
                ),
                "membership_score": membership_by_id.get(chunk_id),
                "preferred_evidence_role": program.preferred_evidence_role,
                "required_evidence_role": program.required_evidence_role,
                "required_evidence_role_basis": (
                    program.required_evidence_role_basis
                ),
                "required_role_match": (
                    None
                    if program.required_evidence_role is None
                    else bool(
                        result.turn is not None
                        and result.turn.role.casefold()
                        == program.required_evidence_role
                    )
                ),
                "value_evidence": value_by_id.get(chunk_id),
            }
        )
    for member in posterior_uncertain_rows:
        chunk_id = member.result.chunk.chunk_id
        hit = hits[chunk_id]
        trace_by_id[chunk_id].update(
            {
                "group_role": "uncertain",
                "qk_score": float(hit.qk_score),
                "ov_transport": float(hit.ov_transport),
                "prefix_utility": float(score_by_id.get(chunk_id, 0.0)),
                "semantic_score": semantic_raw_by_id.get(chunk_id),
                "answer_object_key_present": bool(
                    answer_object_keys_by_id.get(chunk_id)
                ),
                "semantic_score_kind": semantic_kind_by_id.get(chunk_id),
                "answerability_score": answerability_by_id.get(chunk_id),
                "answerability_score_kind": (
                    "forced_choice_explicit_probability"
                    if answerability_by_id.get(chunk_id) is not None
                    else "surface_value_heuristic"
                ),
                "membership_score": membership_by_id.get(chunk_id),
                "preferred_evidence_role": program.preferred_evidence_role,
                "role_match": (
                    None
                    if program.preferred_evidence_role is None
                    else bool(role_match(member))
                ),
                "required_evidence_role": program.required_evidence_role,
                "required_evidence_role_basis": (
                    program.required_evidence_role_basis
                ),
                "required_role_match": (
                    None
                    if program.required_evidence_role is None
                    else required_role_match(member)
                ),
                "value_evidence": member.value_evidence,
                "assignment_hypothesis": "uncertain",
                "p_existing": member.p_existing,
                "p_new": member.p_new,
                "p_null": member.p_null,
                "existing_energy": member.existing_energy,
                "new_energy": member.new_energy,
                "null_energy": member.null_energy,
                "temporal_in_scope": member.temporal_in_scope,
                "posterior_entropy": member.entropy,
                "semantic_surprisal": member.semantic_surprisal,
                "posterior_uncertain": True,
                "credible_cluster": False,
                "coverage_reserved": False,
            }
        )
    for member in null_rows:
        chunk_id = member.result.chunk.chunk_id
        hit = hits[chunk_id]
        trace_by_id[chunk_id].update(
            {
                "group_role": "null",
                "qk_score": float(hit.qk_score),
                "ov_transport": float(hit.ov_transport),
                "prefix_utility": float(score_by_id.get(chunk_id, 0.0)),
                "semantic_score": semantic_raw_by_id.get(chunk_id),
                "answer_object_key_present": bool(
                    answer_object_keys_by_id.get(chunk_id)
                ),
                "semantic_score_kind": semantic_kind_by_id.get(chunk_id),
                "answerability_score": answerability_by_id.get(chunk_id),
                "answerability_score_kind": (
                    "forced_choice_explicit_probability"
                    if answerability_by_id.get(chunk_id) is not None
                    else "surface_value_heuristic"
                ),
                "membership_score": membership_by_id.get(chunk_id),
                "preferred_evidence_role": program.preferred_evidence_role,
                "role_match": (
                    None
                    if program.preferred_evidence_role is None
                    else bool(
                        member.result.turn is not None
                        and member.result.turn.role.casefold()
                        == program.preferred_evidence_role
                    )
                ),
                "required_evidence_role": program.required_evidence_role,
                "required_evidence_role_basis": (
                    program.required_evidence_role_basis
                ),
                "required_role_match": (
                    None
                    if program.required_evidence_role is None
                    else required_role_match(member)
                ),
                "value_evidence": member.value_evidence,
                "assignment_hypothesis": "null",
                "p_existing": member.p_existing,
                "p_new": member.p_new,
                "p_null": member.p_null,
                "existing_energy": member.existing_energy,
                "new_energy": member.new_energy,
                "null_energy": member.null_energy,
                "temporal_in_scope": member.temporal_in_scope,
                "posterior_entropy": member.entropy,
                "semantic_surprisal": member.semantic_surprisal,
                "posterior_uncertain": (
                    member.entropy >= self.uncertainty_entropy
                ),
            }
        )
    self.last_candidate_trace = [
        trace_by_id[result.chunk.chunk_id] for result in unique
    ]

    selected = [item[1].result for item in reserved_rows]
    unresolved = list(uncertain)
    unresolved.extend(
        (member.index, member.result) for member in posterior_uncertain_rows
    )
    unresolved.sort(key=lambda item: item[0])
    selected.extend(result for _index, result in unresolved)
    selected.extend(item[1].result for item in alternative_rows)
    supporting.sort(key=lambda item: item.index)
    selected.extend(item.result for item in supporting)
    # NULL is a posterior hypothesis, not permission to destroy evidence.
    # Retain it at the end so any downstream fail-open or larger budget can
    # still inspect the exact raw row.
    null_rows.sort(key=lambda item: item.index)
    selected.extend(item.result for item in null_rows)
    if max_results is not None:
        selected = selected[:max_results]
    score_kinds = set(semantic_kind_by_id.values())
    semantic_score_kind = "+".join(sorted(score_kinds))
    self.last_report = CoverageSelectionReport(
        operator=program.operator.value,
        cardinality=program.cardinality,
        requires_completeness=True,
        input_candidates=len(unique),
        inspected_candidates=inspected_candidates,
        classified_candidates=len(hits),
        event_clusters=len(clusters),
        new_assignments=new_count,
        existing_assignments=existing_count,
        null_assignments=len(null_rows),
        uncertain_assignments=len(unresolved),
        output_candidates=len(selected),
        representatives=len(cluster_rows) + len(unresolved),
        supporting_candidates=len(supporting),
        workspace_tokens=max_workspace_tokens,
        elapsed_s=time.perf_counter() - started,
        quantifier=program.quantifier.value,
        ordering=program.ordering.value,
        posterior_kind="uncalibrated_energy_softmax",
        semantic_score_kind=semantic_score_kind,
        frontier_candidates=len(unique),
        frontier_attempted=attempted_candidates,
        frontier_uninspected=max(0, len(unique) - attempted_candidates),
        routed_frontier_exhaustive=(attempted_candidates == len(unique)),
        frontier_exhaustive=bool(active_partition_exhaustive),
        frontier_batches=frontier_batches,
        active_partition_total=active_partition_total,
        active_partition_inspected=active_partition_inspected,
        active_partition_exhaustive=active_partition_exhaustive,
        **normalized_scan_fields,
        allow_selected_scope_fixed_k_closure=(
            self.allow_selected_scope_fixed_k_closure
        ),
        credible_clusters=len(credible_cluster_ids),
        reserved_representatives=len(reserved_rows),
        structural_eligible_clusters=len(structural_eligible_cluster_ids),
        structural_reserved_representatives=len(
            structural_reserved_cluster_ids
        ),
        cardinality_deficit=cardinality_deficit,
        answerability_score_kind=(
            "forced_choice_explicit_probability"
            if any(value is not None for value in answerability_by_id.values())
            else "surface_value_heuristic"
        ),
        score_provider_fallback=score_provider_fallback,
        score_provider_report=score_provider_report,
        **self._prefix_report_fields(),
        required_evidence_role=program.required_evidence_role,
        required_evidence_role_basis=program.required_evidence_role_basis,
        query_timestamp=program.query_timestamp,
        temporal_window_days=program.temporal_window_days,
    )
    return selected
