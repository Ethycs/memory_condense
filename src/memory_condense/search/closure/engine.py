"""Bounded, deterministic discourse-obligation closure."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from memory_condense.domain.discourse import (
    ClosurePlan,
    ClosurePolicy,
    ClosureScopeWitness,
    DiscourseRelation,
    DiscourseSnapshot,
    DiscourseUnit,
    Episode,
    EpisodeSeed,
    EvidenceSpan,
    QueryProgram,
)
from memory_condense.search.closure.bundles import BundleAssembly, assemble_bundles
from memory_condense.search.closure.request import (
    bound_closure_inputs,
    resolve_artifact_id,
    resolve_program,
)
from memory_condense.search.closure.results import completion, obligation_results
from memory_condense.search.closure.scope_scan import scan_artifact_units
from memory_condense.search.closure.semantics import (
    relation_confers_subject_connection,
    relation_is_useful,
    relation_obligation_ids,
    relation_priority,
    unit_obligation_ids,
    unit_priority,
)
from memory_condense.search.closure.store import EvidenceClosureStore


class _AnnotationLookupFailure(RuntimeError):
    """Recoverable graph-annotation failure; raw evidence remains authoritative."""


@dataclass(frozen=True, slots=True)
class _UnitRoute:
    unit: DiscourseUnit
    hop: int
    route_rank: int
    connected: bool


@dataclass(slots=True)
class _Walk:
    episodes: dict[str, Episode]
    episode_order: list[str]
    units: dict[str, DiscourseUnit]
    unit_order: list[str]
    relations: dict[str, DiscourseRelation]
    relation_order: list[str]
    connected_unit_ids: set[str]
    connected_relation_ids: set[str] = field(default_factory=set)
    scope_witnesses: list[ClosureScopeWitness] = field(default_factory=list)
    episode_budget_exhaustive: bool = True
    unit_budget_exhaustive: bool = True
    relation_budget_exhaustive: bool = True
    hop_budget_exhaustive: bool = True
    max_visited_hop: int = 0
    capped: bool = False


class EvidenceClosureEngine:
    """Stateful composition root around a read-only closure store.

    The instance retains only the store reference and scalar policy.  Query
    text, hydrated atoms, graph frontiers, and evidence packets are local to a
    single call and are released when that call returns.
    """

    __slots__ = ("policy", "store")

    def __init__(
        self,
        store: EvidenceClosureStore,
        *,
        policy: ClosurePolicy | None = None,
    ) -> None:
        self.store = store
        self.policy = policy or ClosurePolicy()

    def close(
        self,
        query: str | QueryProgram | None = None,
        *,
        query_program: QueryProgram | None = None,
        seeds: Sequence[EpisodeSeed] = (),
        direct_chunk_ids: Sequence[str] = (),
        artifact_id: str | None = None,
        expansion_receipt_sha256: str | None = None,
        expansion_exhaustive: bool | None = None,
    ) -> ClosurePlan:
        """Close the query's obligations over bounded episodic graph routes."""

        program = resolve_program(query, query_program)
        bounded_inputs = bound_closure_inputs(
            seeds,
            direct_chunk_ids,
            limit=self.policy.max_frontier,
        )
        normalized_seeds = bounded_inputs.seeds
        normalized_direct_chunk_ids = bounded_inputs.direct_chunk_ids
        raw_chunk_ids = tuple(
            sorted(
                set(normalized_direct_chunk_ids)
                | {seed.anchor_chunk_id for seed in normalized_seeds}
            )
        )
        snapshot = self.store.snapshot()
        selected_artifact_id = resolve_artifact_id(
            snapshot,
            artifact_id,
            graph_requested=bool(raw_chunk_ids or normalized_seeds),
            has_explicit_seeds=bool(normalized_seeds),
        )
        if expansion_exhaustive is not None and expansion_receipt_sha256 is None:
            raise ValueError(
                "expansion_exhaustive requires an expansion_receipt_sha256"
            )
        raw_spans = tuple(
            span
            for span in self.store.evidence_for_chunks(raw_chunk_ids)
            if program.as_of_ordinal is None or span.ordinal <= program.as_of_ordinal
        )
        if selected_artifact_id is None:
            walk = _Walk({}, [], {}, [], {}, [], set())
            unit_routes: dict[str, _UnitRoute] = {}
        else:
            walk, unit_routes = self._walk(
                program,
                artifact_id=selected_artifact_id,
                snapshot=snapshot,
                seeds=normalized_seeds,
                raw_chunk_ids=raw_chunk_ids,
            )
        walk.scope_witnesses.extend(bounded_inputs.scope_witnesses)
        if expansion_receipt_sha256 is not None:
            _record_witness(
                walk,
                kind="episode_expansion",
                subject_id=expansion_receipt_sha256,
                requested_limit=None,
                returned_count=len(normalized_seeds) + len(normalized_direct_chunk_ids),
                exhaustive=expansion_exhaustive is True,
                detail={
                    "seed_count": len(normalized_seeds),
                    "direct_chunk_count": len(normalized_direct_chunk_ids),
                    "receipt_sha256": expansion_receipt_sha256,
                    "exhaustiveness_attested": expansion_exhaustive is not None,
                },
            )

        proof_relation_ids = {
            relation_id
            for relation_id, relation in walk.relations.items()
            if relation_id in walk.connected_relation_ids
            and relation_confers_subject_connection(
                relation,
                program,
                min_confidence=self.policy.min_relation_confidence,
            )
        }
        unit_obligations = {
            unit_id: unit_obligation_ids(
                unit,
                program,
                connected=(
                    unit_id in walk.connected_unit_ids
                    or unit_routes.get(unit_id, _UnitRoute(unit, 0, 3, False)).connected
                ),
            )
            for unit_id, unit in walk.units.items()
        }
        assembly = assemble_bundles(
            self.store,
            program=program,
            policy=self.policy,
            raw_spans=raw_spans,
            units=walk.units,
            relations=walk.relations,
            unit_obligations=unit_obligations,
            credited_relation_ids=proof_relation_ids,
        )
        _record_witness(
            walk,
            kind="bundle_budget",
            subject_id=program.program_sha256,
            requested_limit=self.policy.max_bundles,
            returned_count=len(assembly.bundles),
            exhaustive=not assembly.truncated,
            detail={
                "candidate_selection": "atomic_evidence_bundles",
                "configured_limit": self.policy.max_bundles,
                "omitted_direct_chunk_ids": assembly.omitted_direct_chunk_ids,
            },
        )
        results = obligation_results(
            program,
            units=walk.units,
            relations=walk.relations,
            unit_obligations=unit_obligations,
            assembly=assembly,
            min_relation_confidence=self.policy.min_relation_confidence,
            credited_relation_ids=proof_relation_ids,
        )
        stopping_reason, complete = completion(
            results,
            program,
            scope_witnesses=walk.scope_witnesses,
        )

        end_snapshot = self.store.snapshot()
        if end_snapshot != snapshot:
            raise RuntimeError("discourse snapshot changed during evidence closure")
        return ClosurePlan(
            query_program=program,
            policy=self.policy,
            snapshot=snapshot,
            seeds=normalized_seeds,
            atoms=assembly.atoms,
            bundles=assembly.bundles,
            obligation_results=results,
            visited_episode_ids=tuple(walk.episode_order),
            visited_unit_ids=tuple(walk.unit_order),
            visited_relation_ids=tuple(walk.relation_order),
            stopping_reason=stopping_reason,
            complete_claimed=complete,
            scope_witnesses=tuple(walk.scope_witnesses),
            direct_chunk_ids=normalized_direct_chunk_ids,
            expansion_receipt_sha256=expansion_receipt_sha256,
            artifact_id=selected_artifact_id,
        )

    def _walk(
        self,
        program: QueryProgram,
        *,
        artifact_id: str,
        snapshot: DiscourseSnapshot,
        seeds: Sequence[EpisodeSeed],
        raw_chunk_ids: Sequence[str],
    ) -> tuple[_Walk, dict[str, _UnitRoute]]:
        walk = _Walk({}, [], {}, [], {}, [], set())
        global_units = self._artifact_scope(
            walk,
            program,
            artifact_id=artifact_id,
            snapshot=snapshot,
        )
        trusted_episode_ids = self._seed_episodes(
            walk,
            artifact_id=artifact_id,
            seeds=seeds,
            raw_chunk_ids=raw_chunk_ids,
        )
        if trusted_episode_ids:
            episode_chunks = tuple(
                sorted(
                    {
                        span.chunk_id
                        for episode_id in trusted_episode_ids
                        for span in walk.episodes[episode_id].evidence
                    }
                )
            )
            self._annotation_coverage(
                walk,
                artifact_id=artifact_id,
                chunks=episode_chunks,
                coverage_kind="episode",
                witness_kind="episode_coverage",
                completion_critical=False,
            )
        neighbor_ids = self._temporal_neighbors(
            walk,
            artifact_id=artifact_id,
            trusted_episode_ids=trusted_episode_ids,
        )
        _record_witness(
            walk,
            kind="episode_budget",
            subject_id=artifact_id,
            requested_limit=self.policy.max_frontier,
            returned_count=len(walk.episodes),
            exhaustive=True,
            detail={
                "seed_count": len(seeds),
                "neighbor_radius": self.policy.max_episode_neighbors,
                "route_truncated": not walk.episode_budget_exhaustive,
            },
        )
        trusted_chunks = tuple(
            sorted(
                set(raw_chunk_ids)
                | {
                    span.chunk_id
                    for episode_id in trusted_episode_ids
                    if episode_id in walk.episodes
                    for span in walk.episodes[episode_id].evidence
                }
            )
        )
        neighbor_chunks = tuple(
            sorted(
                {
                    span.chunk_id
                    for episode_id in neighbor_ids
                    if episode_id in walk.episodes
                    for span in walk.episodes[episode_id].evidence
                }
                - set(trusted_chunks)
            )
        )
        graph_chunks = tuple(sorted(set(trusted_chunks) | set(neighbor_chunks)))
        self._annotation_coverage(
            walk,
            artifact_id=artifact_id,
            chunks=graph_chunks,
            coverage_kind="discourse",
            witness_kind="annotation_coverage",
        )

        unit_routes: dict[str, _UnitRoute] = {}
        for unit in global_units:
            matched = bool(unit_obligation_ids(unit, program, connected=False))
            if matched:
                _offer_route(
                    unit_routes,
                    unit,
                    hop=0,
                    route_rank=3,
                    connected=True,
                )
        candidate_limit = self.policy.max_units + 1
        trusted_units = self._units_for_chunks(
            walk,
            program,
            artifact_id=artifact_id,
            chunks=trusted_chunks,
            route_rank=0,
            witness_subject="trusted_chunks",
            limit=candidate_limit,
        )
        for unit in trusted_units:
            _offer_route(
                unit_routes,
                unit,
                hop=0,
                route_rank=0,
                connected=bool(
                    unit_obligation_ids(unit, program, connected=False)
                ),
            )
        if neighbor_chunks:
            temporal_units = self._units_for_chunks(
                walk,
                program,
                artifact_id=artifact_id,
                chunks=neighbor_chunks,
                route_rank=2,
                witness_subject="temporal_chunks",
                limit=candidate_limit,
            )
            for unit in temporal_units:
                _offer_route(unit_routes, unit, hop=0, route_rank=2, connected=False)

        relation_probe_limit = self.policy.max_relations + 1
        try:
            probed_direct_relations = self.store.relations_for_chunks(
                trusted_chunks,
                artifact_id=artifact_id,
                limit=relation_probe_limit,
            )
        except Exception as exc:
            probed_direct_relations = ()
            _record_witness(
                walk,
                kind="chunk_relation_lookup",
                subject_id="trusted_chunks",
                requested_limit=self.policy.max_relations,
                returned_count=0,
                exhaustive=False,
                detail={"failure": type(exc).__name__},
            )
        else:
            if len(probed_direct_relations) > relation_probe_limit:
                raise ValueError(
                    "store returned more relations than the requested probe limit"
                )
            _validate_artifacts(probed_direct_relations, artifact_id, "relation")

        direct_probe_count = len(probed_direct_relations)
        direct_member_units = dict(unit_routes)
        priority_units = {
            unit_id: route.unit for unit_id, route in direct_member_units.items()
        }
        valid_direct_relations: list[DiscourseRelation] = []
        for relation in probed_direct_relations:
            try:
                for member in relation.members:
                    if member.unit_id not in priority_units:
                        priority_units[member.unit_id] = _get_member_unit(
                            self.store,
                            member.unit_id,
                            artifact_id=artifact_id,
                        )
            except _AnnotationLookupFailure as failure:
                _record_member_failure(walk, relation.relation_id, failure)
                continue
            valid_direct_relations.append(relation)
        probed_direct_relations = tuple(valid_direct_relations)
        direct_relations = tuple(
            sorted(
                probed_direct_relations,
                key=lambda relation: relation_priority(
                    relation,
                    program,
                    priority_units,
                    min_confidence=self.policy.min_relation_confidence,
                ),
            )[: self.policy.max_relations]
        )
        if not any(
            witness.kind == "chunk_relation_lookup"
            and witness.subject_id == "trusted_chunks"
            for witness in walk.scope_witnesses
        ):
            _record_witness(
                walk,
                kind="chunk_relation_lookup",
                subject_id="trusted_chunks",
                requested_limit=self.policy.max_relations,
                returned_count=len(direct_relations),
                exhaustive=direct_probe_count <= self.policy.max_relations,
                detail={
                    "artifact_id": artifact_id,
                    "chunk_count": len(trusted_chunks),
                    "probe_limit": relation_probe_limit,
                    "probe_count": direct_probe_count,
                },
            )
        # Relation members are valid starting points even when a unit's own
        # evidence is outside the seed chunk.
        for relation in direct_relations:
            if not _as_of_relation(relation, program):
                continue
            member_units = tuple(
                priority_units[member.unit_id] for member in relation.members
            )
            for unit in member_units:
                directly_matches = bool(
                    unit_obligation_ids(unit, program, connected=False)
                )
                _offer_route(
                    unit_routes,
                    unit,
                    hop=0,
                    route_rank=1,
                    # The relation may still lose a bounded priority cut.  It
                    # confers semantic credit only once it is actually visited.
                    connected=directly_matches,
                )

        pending_direct = {
            relation.relation_id: relation
            for relation in direct_relations
            if _as_of_relation(relation, program)
        }
        for hop in range(self.policy.max_hops + 1):
            current = [
                route
                for route in unit_routes.values()
                if route.hop == hop and route.unit.unit_id not in walk.units
                and _as_of_unit(route.unit, program)
            ]
            current.sort(
                key=lambda route: unit_priority(
                    route.unit,
                    program,
                    connected=route.connected,
                    route_rank=route.route_rank,
                )
            )
            frontier_limit = min(self.policy.beam_width, self.policy.max_frontier)
            frontier_exhaustive = len(current) <= frontier_limit
            current = current[:frontier_limit]
            if current:
                _record_witness(
                    walk,
                    kind="unit_frontier",
                    subject_id=f"{artifact_id}:hop:{hop}",
                    requested_limit=frontier_limit,
                    returned_count=len(current),
                    exhaustive=frontier_exhaustive,
                    detail={
                        "beam_width": self.policy.beam_width,
                        "max_frontier": self.policy.max_frontier,
                    },
                )
            remaining_units = self.policy.max_units - len(walk.units)
            if len(current) > remaining_units:
                current = current[: max(0, remaining_units)]
                walk.unit_budget_exhaustive = False
            if not current and not (hop == 0 and pending_direct):
                continue

            for route in current:
                unit_id = route.unit.unit_id
                walk.units[unit_id] = route.unit
                walk.unit_order.append(unit_id)
                walk.max_visited_hop = max(walk.max_visited_hop, hop)
                if route.connected:
                    walk.connected_unit_ids.add(unit_id)

            requested_unit_ids = tuple(route.unit.unit_id for route in current)
            incident: Mapping[str, Sequence[DiscourseRelation]] = {}
            if requested_unit_ids:
                try:
                    incident = self.store.incident_relations(
                        requested_unit_ids,
                        artifact_id=artifact_id,
                        max_degree=self.policy.max_degree + 1,
                    )
                except Exception as exc:
                    incident = {unit_id: () for unit_id in requested_unit_ids}
                    for unit_id in requested_unit_ids:
                        _record_witness(
                            walk,
                            kind="incident_relations",
                            subject_id=unit_id,
                            requested_limit=self.policy.max_degree,
                            returned_count=0,
                            exhaustive=False,
                            detail={"failure": type(exc).__name__},
                        )
                else:
                    _validate_incident(
                        incident,
                        requested_unit_ids,
                        artifact_id=artifact_id,
                    )
                    for unit_id in requested_unit_ids:
                        probed = tuple(incident[unit_id])
                        if len(probed) > self.policy.max_degree + 1:
                            raise ValueError(
                                "store returned more incident relations than the probe limit"
                            )
                        _record_witness(
                            walk,
                            kind="incident_relations",
                            subject_id=unit_id,
                            requested_limit=self.policy.max_degree,
                            returned_count=min(len(probed), self.policy.max_degree),
                            exhaustive=len(probed) <= self.policy.max_degree,
                            detail={
                                "artifact_id": artifact_id,
                                "probe_limit": self.policy.max_degree + 1,
                                "probe_count": len(probed),
                            },
                        )

            probed_candidates: dict[str, DiscourseRelation] = {}
            if hop == 0:
                probed_candidates.update(pending_direct)
            for unit_id in sorted(incident):
                for relation in incident[unit_id]:
                    probed_candidates.setdefault(relation.relation_id, relation)

            known_units = dict(walk.units)
            invalid_relation_ids: set[str] = set()
            for relation in probed_candidates.values():
                try:
                    for member in relation.members:
                        if member.unit_id not in known_units:
                            known_units[member.unit_id] = _get_member_unit(
                                self.store,
                                member.unit_id,
                                artifact_id=artifact_id,
                            )
                except _AnnotationLookupFailure as failure:
                    invalid_relation_ids.add(relation.relation_id)
                    _record_member_failure(walk, relation.relation_id, failure)
            candidates: dict[str, DiscourseRelation] = {}
            if hop == 0:
                candidates.update(
                    (relation_id, relation)
                    for relation_id, relation in pending_direct.items()
                    if relation_id not in invalid_relation_ids
                )
            for unit_id in sorted(incident):
                ordered_incident = sorted(
                    (
                        relation
                        for relation in incident[unit_id]
                        if relation.relation_id not in invalid_relation_ids
                    ),
                    key=lambda relation: relation_priority(
                        relation,
                        program,
                        known_units,
                        min_confidence=self.policy.min_relation_confidence,
                    ),
                )[: self.policy.max_degree]
                for relation in ordered_incident:
                    candidates.setdefault(relation.relation_id, relation)
            ordered_relations = sorted(
                (
                    relation
                    for relation in candidates.values()
                    if relation.relation_id not in walk.relations
                    and _as_of_relation(relation, program)
                    and relation_is_useful(
                        relation,
                        program,
                        known_units,
                        connected=True,
                    )
                ),
                key=lambda relation: relation_priority(
                    relation,
                    program,
                    known_units,
                    min_confidence=self.policy.min_relation_confidence,
                ),
            )
            remaining_relations = self.policy.max_relations - len(walk.relations)
            if len(ordered_relations) > remaining_relations:
                ordered_relations = ordered_relations[: max(0, remaining_relations)]
                walk.relation_budget_exhaustive = False
            for relation in ordered_relations:
                walk.relations[relation.relation_id] = relation
                walk.relation_order.append(relation.relation_id)
                member_units = [
                    known_units[member.unit_id]
                    for member in relation.members
                    if member.unit_id in known_units
                ]
                anchor_connected = any(
                    unit.unit_id in walk.connected_unit_ids
                    or unit_routes.get(
                        unit.unit_id,
                        _UnitRoute(unit, hop, 3, False),
                    ).connected
                    or bool(unit_obligation_ids(unit, program, connected=False))
                    for unit in member_units
                )
                edge_connects = anchor_connected and relation_confers_subject_connection(
                    relation,
                    program,
                    min_confidence=self.policy.min_relation_confidence,
                )
                if edge_connects:
                    walk.connected_relation_ids.add(relation.relation_id)
                for member in relation.members:
                    unit = known_units.get(member.unit_id)
                    if unit is None or not _as_of_unit(unit, program):
                        continue
                    directly_matches = bool(
                        unit_obligation_ids(unit, program, connected=False)
                    )
                    member_connects = edge_connects or directly_matches
                    if member_connects:
                        walk.connected_unit_ids.add(unit.unit_id)
                    if unit.unit_id in walk.units:
                        continue
                    if hop >= self.policy.max_hops:
                        walk.hop_budget_exhaustive = False
                        continue
                    _offer_route(
                        unit_routes,
                        unit,
                        hop=hop + 1,
                        route_rank=1,
                        connected=member_connects,
                    )

        if any(
            route.unit.unit_id not in walk.units and _as_of_unit(route.unit, program)
            for route in unit_routes.values()
        ):
            walk.unit_budget_exhaustive = False
        _record_witness(
            walk,
            kind="unit_budget",
            subject_id=artifact_id,
            requested_limit=self.policy.max_units,
            returned_count=len(walk.units),
            exhaustive=walk.unit_budget_exhaustive,
        )
        _record_witness(
            walk,
            kind="relation_budget",
            subject_id=artifact_id,
            requested_limit=self.policy.max_relations,
            returned_count=len(walk.relations),
            exhaustive=walk.relation_budget_exhaustive,
        )
        _record_witness(
            walk,
            kind="hop_budget",
            subject_id=artifact_id,
            requested_limit=self.policy.max_hops,
            returned_count=min(walk.max_visited_hop, self.policy.max_hops),
            exhaustive=walk.hop_budget_exhaustive,
        )
        return walk, unit_routes

    def _artifact_scope(
        self,
        walk: _Walk,
        program: QueryProgram,
        *,
        artifact_id: str,
        snapshot: DiscourseSnapshot,
    ) -> tuple[DiscourseUnit, ...]:
        try:
            coverage = self.store.artifact_coverage(
                artifact_id,
                coverage_kind="discourse",
            )
        except Exception as exc:
            coverage = None
            coverage_failure = type(exc).__name__
        else:
            coverage_failure = None
        if coverage is not None:
            _validate_artifact(coverage, artifact_id, "artifact coverage receipt")
            if coverage.coverage_kind != "discourse":
                raise ValueError("artifact coverage receipt has the wrong kind")
        coverage_current = bool(
            coverage is not None
            and coverage.source_revision == snapshot.source_revision
            and coverage.chunk_count == snapshot.chunk_count
        )
        _record_witness(
            walk,
            kind="artifact_coverage",
            subject_id=artifact_id,
            requested_limit=None,
            returned_count=0 if coverage is None else coverage.chunk_count,
            exhaustive=coverage_current,
            detail={
                "coverage_kind": "discourse",
                "failure": coverage_failure,
                "snapshot_source_revision": snapshot.source_revision,
                "snapshot_chunk_count": snapshot.chunk_count,
                "receipt_sha256": (
                    None if coverage is None else coverage.receipt_sha256
                ),
                "coverage_sha256": (
                    None if coverage is None else coverage.coverage_sha256
                ),
            },
        )

        probe_limit = self.policy.max_units + 1
        scan = scan_artifact_units(
            self.store,
            artifact_id=artifact_id,
            program=program,
            max_units=self.policy.max_units,
        )
        _record_witness(
            walk,
            kind="artifact_unit_scan",
            subject_id=artifact_id,
            requested_limit=self.policy.max_units,
            returned_count=len(scan.units),
            exhaustive=scan.exhaustive,
            detail={
                "probe_limit": probe_limit,
                "probe_count": scan.scanned_count,
                "matched_count": scan.matched_count,
                "scan_mode": scan.scan_mode,
                "failure": scan.failure,
            },
        )
        return scan.units

    def _annotation_coverage(
        self,
        walk: _Walk,
        *,
        artifact_id: str,
        chunks: Sequence[str],
        coverage_kind: str,
        witness_kind: str,
        completion_critical: bool = True,
    ) -> None:
        try:
            coverage = self.store.coverage_for_chunks(
                artifact_id,
                chunks,
                coverage_kind=coverage_kind,
            )
        except Exception as exc:
            _record_witness(
                walk,
                kind=witness_kind,
                subject_id=artifact_id,
                requested_limit=None,
                returned_count=0,
                exhaustive=not completion_critical,
                detail={
                    "chunk_count": len(chunks),
                    "failure": type(exc).__name__,
                    "coverage_complete": False,
                    "completion_critical": completion_critical,
                },
            )
            return
        requested = set(chunks)
        if any(chunk_id not in requested for chunk_id in coverage):
            raise ValueError("coverage result contains an unrequested chunk")
        if any(status not in {"annotated", "no_output"} for status in coverage.values()):
            raise ValueError("coverage result contains an unsupported status")
        missing = tuple(sorted(requested - set(coverage)))
        _record_witness(
            walk,
            kind=witness_kind,
            subject_id=artifact_id,
            requested_limit=None,
            returned_count=len(coverage),
            exhaustive=(not missing) or not completion_critical,
            detail={
                "coverage_kind": coverage_kind,
                "chunk_count": len(chunks),
                "missing_chunk_ids": missing,
                "annotated_count": sum(
                    status == "annotated" for status in coverage.values()
                ),
                "no_output_count": sum(
                    status == "no_output" for status in coverage.values()
                ),
                "coverage_complete": not missing,
                "completion_critical": completion_critical,
            },
        )

    def _units_for_chunks(
        self,
        walk: _Walk,
        program: QueryProgram,
        *,
        artifact_id: str,
        chunks: Sequence[str],
        route_rank: int,
        witness_subject: str,
        limit: int,
    ) -> tuple[DiscourseUnit, ...]:
        try:
            probed = self.store.units_for_chunks(
                chunks,
                artifact_id=artifact_id,
                limit=limit,
            )
        except Exception as exc:
            _record_witness(
                walk,
                kind="chunk_unit_lookup",
                subject_id=witness_subject,
                requested_limit=self.policy.max_units,
                returned_count=0,
                exhaustive=False,
                detail={"failure": type(exc).__name__},
            )
            return ()
        if len(probed) > limit:
            raise ValueError("store returned more units than the requested probe limit")
        _validate_artifacts(probed, artifact_id, "unit")
        ordered = sorted(
            probed,
            key=lambda unit: unit_priority(
                unit,
                program,
                connected=bool(unit_obligation_ids(unit, program, connected=False)),
                route_rank=route_rank,
            ),
        )
        exhaustive = len(ordered) <= self.policy.max_units
        admitted = tuple(ordered[: self.policy.max_units])
        _record_witness(
            walk,
            kind="chunk_unit_lookup",
            subject_id=witness_subject,
            requested_limit=self.policy.max_units,
            returned_count=len(admitted),
            exhaustive=exhaustive,
            detail={
                "artifact_id": artifact_id,
                "chunk_count": len(chunks),
                "probe_limit": limit,
                "probe_count": len(probed),
            },
        )
        return admitted

    def _seed_episodes(
        self,
        walk: _Walk,
        *,
        artifact_id: str,
        seeds: Sequence[EpisodeSeed],
        raw_chunk_ids: Sequence[str],
    ) -> tuple[str, ...]:
        candidates: dict[str, Episode] = {}
        for seed in seeds:
            episode = self.store.get_episode(seed.episode_id)
            if episode is None:
                raise ValueError(f"episode seed {seed.episode_id!r} does not exist")
            _validate_artifact(episode, artifact_id, "episode seed")
            if seed.anchor_chunk_id not in {
                span.chunk_id for span in episode.evidence
            }:
                raise ValueError(
                    f"episode seed {seed.episode_id!r} does not contain anchor "
                    f"chunk {seed.anchor_chunk_id!r}"
                )
            candidates.setdefault(seed.episode_id, episode)

        mapping_exhaustive = True
        mapping_failures: list[str] = []
        try:
            mapped = self.store.episode_ids_for_chunks(
                raw_chunk_ids,
                artifact_id=artifact_id,
            )
        except Exception as exc:
            mapped = {}
            mapping_exhaustive = False
            mapping_failures.append(type(exc).__name__)
        if any(chunk_id not in set(raw_chunk_ids) for chunk_id in mapped):
            raise ValueError("store returned an episode mapping for an unrequested chunk")
        valid_mapping_count = 0
        for chunk_id in sorted(mapped):
            episode_id = mapped[chunk_id]
            try:
                episode = self.store.get_episode(episode_id)
            except Exception as exc:
                mapping_exhaustive = False
                mapping_failures.append(type(exc).__name__)
                continue
            if episode is None:
                mapping_exhaustive = False
                mapping_failures.append("missing_episode")
                continue
            _validate_artifact(episode, artifact_id, "episode")
            if chunk_id not in {span.chunk_id for span in episode.evidence}:
                raise ValueError("episode mapping does not contain its mapped chunk")
            candidates.setdefault(episode_id, episode)
            valid_mapping_count += 1
        _record_witness(
            walk,
            kind="episode_mapping",
            subject_id=artifact_id,
            requested_limit=None,
            returned_count=valid_mapping_count,
            exhaustive=mapping_exhaustive,
            detail={
                "chunk_count": len(raw_chunk_ids),
                "failures": mapping_failures,
            },
        )

        episode_ids: list[str] = [seed.episode_id for seed in seeds]
        episode_ids.extend(mapped[chunk_id] for chunk_id in sorted(mapped))
        ordered_ids = tuple(dict.fromkeys(episode_ids))
        if len(ordered_ids) > self.policy.max_frontier:
            ordered_ids = ordered_ids[: self.policy.max_frontier]
            walk.episode_budget_exhaustive = False
        for episode_id in ordered_ids:
            episode = candidates.get(episode_id)
            if episode is None:
                # A failed optional annotation lookup is recorded above; the
                # verified raw chunk remains available to bundle assembly.
                continue
            walk.episodes[episode_id] = episode
            walk.episode_order.append(episode_id)
        return tuple(walk.episode_order)

    def _temporal_neighbors(
        self,
        walk: _Walk,
        trusted_episode_ids: Sequence[str],
        *,
        artifact_id: str,
    ) -> tuple[str, ...]:
        candidates: dict[str, Episode] = {}
        allowed_radius = self.policy.max_episode_neighbors
        probe_radius = allowed_radius + 1
        for episode_id in trusted_episode_ids:
            seed = walk.episodes.get(episode_id)
            if seed is None:
                continue
            try:
                neighbors = self.store.adjacent_episodes(
                    episode_id,
                    radius=probe_radius,
                )
            except Exception as exc:
                _record_witness(
                    walk,
                    kind="temporal_neighbors",
                    subject_id=episode_id,
                    requested_limit=None,
                    returned_count=0,
                    exhaustive=False,
                    detail={
                        "radius": allowed_radius,
                        "probe_radius": probe_radius,
                        "failure": type(exc).__name__,
                    },
                )
                continue
            prior: list[Episode] = []
            following: list[Episode] = []
            seen_neighbor_ids: set[str] = set()
            for neighbor in neighbors:
                if neighbor.source_id != seed.source_id or neighbor.artifact_id != seed.artifact_id:
                    raise ValueError("temporal closure crossed a source or artifact boundary")
                _validate_artifact(neighbor, artifact_id, "episode neighbor")
                if neighbor.episode_id == episode_id:
                    raise ValueError("store returned the seed as its own adjacent episode")
                if neighbor.episode_id in seen_neighbor_ids:
                    raise ValueError("store returned a duplicate adjacent episode")
                seen_neighbor_ids.add(neighbor.episode_id)
                if neighbor.sequence_no < seed.sequence_no:
                    prior.append(neighbor)
                elif neighbor.sequence_no > seed.sequence_no:
                    following.append(neighbor)
                else:
                    raise ValueError(
                        "adjacent episode reused the seed sequence coordinate"
                    )
            if len(prior) > probe_radius or len(following) > probe_radius:
                raise ValueError("store returned more episodes than the per-side probe radius")
            admitted = tuple(
                sorted(prior, key=lambda item: (-item.sequence_no, item.episode_id))[
                    :allowed_radius
                ]
                + sorted(
                    following,
                    key=lambda item: (item.sequence_no, item.episode_id),
                )[:allowed_radius]
            )
            outside_radius = (
                len(prior) > allowed_radius or len(following) > allowed_radius
            )
            for neighbor in admitted:
                if neighbor.episode_id not in walk.episodes:
                    candidates.setdefault(neighbor.episode_id, neighbor)
            _record_witness(
                walk,
                kind="temporal_neighbors",
                subject_id=episode_id,
                requested_limit=None,
                returned_count=len(admitted),
                # This receipt is exhaustive for the requested local radius.
                # A corpus-wide unit scan is the global completeness proof;
                # farther episodes are diagnostic, not a hidden corpus tail.
                exhaustive=True,
                detail={
                    "artifact_id": artifact_id,
                    "source_id": seed.source_id,
                    "radius": allowed_radius,
                    "probe_radius": probe_radius,
                    "probe_count": len(neighbors),
                    "outside_requested_radius": outside_radius,
                },
            )
        ordered = sorted(
            candidates.values(),
            key=lambda item: (item.source_id, item.sequence_no, item.episode_id),
        )
        available = max(0, self.policy.max_frontier - len(walk.episodes))
        if len(ordered) > available:
            ordered = ordered[:available]
            walk.episode_budget_exhaustive = False
        for episode in ordered:
            walk.episodes[episode.episode_id] = episode
            walk.episode_order.append(episode.episode_id)
        return tuple(episode.episode_id for episode in ordered)


def close_evidence(
    store: EvidenceClosureStore,
    query: str | QueryProgram | None = None,
    *,
    query_program: QueryProgram | None = None,
    seeds: Sequence[EpisodeSeed] = (),
    direct_chunk_ids: Sequence[str] = (),
    artifact_id: str | None = None,
    expansion_receipt_sha256: str | None = None,
    expansion_exhaustive: bool | None = None,
    policy: ClosurePolicy | None = None,
) -> ClosurePlan:
    """Functional facade for one bounded closure request."""

    return EvidenceClosureEngine(store, policy=policy).close(
        query,
        query_program=query_program,
        seeds=seeds,
        direct_chunk_ids=direct_chunk_ids,
        artifact_id=artifact_id,
        expansion_receipt_sha256=expansion_receipt_sha256,
        expansion_exhaustive=expansion_exhaustive,
    )


def _validate_artifact(value: object, artifact_id: str, label: str) -> None:
    actual = getattr(value, "artifact_id", None)
    if actual != artifact_id:
        raise ValueError(
            f"{label} crossed artifact scope: expected {artifact_id!r}, got {actual!r}"
        )


def _validate_artifacts(
    values: Iterable[object],
    artifact_id: str,
    label: str,
) -> None:
    for value in values:
        _validate_artifact(value, artifact_id, label)


def _get_member_unit(
    store: EvidenceClosureStore,
    unit_id: str,
    *,
    artifact_id: str,
) -> DiscourseUnit:
    try:
        unit = store.get_unit(unit_id)
    except Exception as exc:
        raise _AnnotationLookupFailure(type(exc).__name__) from exc
    if unit is None:
        raise _AnnotationLookupFailure("missing_unit")
    _validate_artifact(unit, artifact_id, "relation member unit")
    return unit


def _record_member_failure(
    walk: _Walk,
    relation_id: str,
    failure: _AnnotationLookupFailure,
) -> None:
    if any(
        witness.kind == "relation_member_lookup"
        and witness.subject_id == relation_id
        for witness in walk.scope_witnesses
    ):
        return
    _record_witness(
        walk,
        kind="relation_member_lookup",
        subject_id=relation_id,
        requested_limit=None,
        returned_count=0,
        exhaustive=False,
        detail={"failure": str(failure)},
    )


def _validate_incident(
    incident: Mapping[str, Sequence[DiscourseRelation]],
    unit_ids: Sequence[str],
    *,
    artifact_id: str,
) -> None:
    expected = set(unit_ids)
    if set(incident) != expected:
        raise ValueError("incident relation result does not cover exactly the requested units")
    for unit_id in sorted(incident):
        for relation in incident[unit_id]:
            _validate_artifact(relation, artifact_id, "incident relation")
            if unit_id not in {member.unit_id for member in relation.members}:
                raise ValueError("incident relation does not contain its requested unit")


def _record_witness(
    walk: _Walk,
    *,
    kind: str,
    subject_id: str,
    requested_limit: int | None,
    returned_count: int,
    exhaustive: bool,
    detail: Mapping[str, object] | None = None,
) -> None:
    witness = ClosureScopeWitness(
        kind=kind,
        subject_id=subject_id,
        requested_limit=requested_limit,
        returned_count=returned_count,
        exhaustive=exhaustive,
        detail={} if detail is None else detail,
    )
    if any(
        existing.kind == witness.kind and existing.subject_id == witness.subject_id
        for existing in walk.scope_witnesses
    ):
        raise RuntimeError(
            f"duplicate closure scope witness for {kind!r}/{subject_id!r}"
        )
    walk.scope_witnesses.append(witness)
    if not exhaustive:
        walk.capped = True


def _offer_route(
    routes: dict[str, _UnitRoute],
    unit: DiscourseUnit,
    *,
    hop: int,
    route_rank: int,
    connected: bool,
) -> None:
    candidate = _UnitRoute(unit, hop, route_rank, connected)
    existing = routes.get(unit.unit_id)
    if existing is None:
        routes[unit.unit_id] = candidate
        return
    if (hop, route_rank, unit.unit_id) < (
        existing.hop,
        existing.route_rank,
        existing.unit.unit_id,
    ) or (connected and not existing.connected):
        routes[unit.unit_id] = _UnitRoute(
            unit,
            min(hop, existing.hop),
            min(route_rank, existing.route_rank),
            connected or existing.connected,
        )


def _as_of_unit(unit: DiscourseUnit, program: QueryProgram) -> bool:
    return program.as_of_ordinal is None or unit.asserted_ordinal <= program.as_of_ordinal


def _as_of_relation(relation: DiscourseRelation, program: QueryProgram) -> bool:
    return program.as_of_ordinal is None or relation.created_ordinal <= program.as_of_ordinal


__all__ = ["EvidenceClosureEngine", "close_evidence"]
