"""Bounded, deterministic discourse-obligation closure."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
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
    QueryProgram,
)
from memory_condense.search.closure.bundles import assemble_bundles
from memory_condense.search.closure.request import (
    ClosureRoutingScope,
    bound_closure_inputs,
    closure_routing_scope,
    closure_routing_witness,
    resolve_artifact_id,
    resolve_program,
)
from memory_condense.search.closure.results import completion, obligation_results
from memory_condense.search.closure.scope_scan import (
    PROBE_OVERFLOW_MESSAGE,
    scan_artifact_units,
    validate_chunk_scoped_rows,
    validate_requested_spans,
)
from memory_condense.search.closure.semantics import (
    relation_confers_subject_connection,
    relation_is_useful,
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


#: One scope witness to record: (subject_id, returned_count, exhaustive, detail).
_WitnessRow = tuple[str, int, bool, Mapping[str, object]]
_WitnessRows = Sequence[_WitnessRow]


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
        routing_scope: ClosureRoutingScope = "artifact_global",
    ) -> ClosurePlan:
        """Close the query's obligations over bounded episodic graph routes."""

        active_routing_scope = closure_routing_scope(routing_scope)
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
        validate_requested_spans(raw_spans, raw_chunk_ids)
        if selected_artifact_id is None:
            walk = _Walk({}, [], {}, [], {}, [], set())
            unit_routes: dict[str, _UnitRoute] = {}
        else:
            walk, unit_routes = _ClosureWalk(
                self.store,
                policy=self.policy,
                program=program,
                artifact_id=selected_artifact_id,
                snapshot=snapshot,
                routing_scope=active_routing_scope,
            ).run(seeds=normalized_seeds, raw_chunk_ids=raw_chunk_ids)
        walk.scope_witnesses.extend(bounded_inputs.scope_witnesses)
        routing_witness = closure_routing_witness(
            active_routing_scope,
            normalized_seeds,
            normalized_direct_chunk_ids,
            limit=self.policy.max_frontier,
        )
        if routing_witness is not None:
            walk.scope_witnesses.append(routing_witness)
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
                    or ((route := unit_routes.get(unit_id)) is not None and route.connected)
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


class _ClosureWalk:
    """Single-use owner of one bounded graph walk over an artifact scope.

    Phases, in receipt order: scan the artifact-wide scope, seed episodes and
    hop-0 unit routes, expand relations hop by hop, then emit the walk-wide
    budget receipts.  Every store probe funnels through :func:`_probe`, so no
    phase can skip the scope witnesses that ``completion()`` relies on.
    """

    __slots__ = (
        "artifact_id", "policy", "program", "routes", "routing_scope",
        "snapshot", "store", "walk",
    )

    def __init__(
        self,
        store: EvidenceClosureStore,
        *,
        policy: ClosurePolicy,
        program: QueryProgram,
        artifact_id: str,
        snapshot: DiscourseSnapshot,
        routing_scope: ClosureRoutingScope,
    ) -> None:
        self.store = store
        self.policy = policy
        self.program = program
        self.artifact_id = artifact_id
        self.snapshot = snapshot
        self.routing_scope = routing_scope
        self.walk = _Walk({}, [], {}, [], {}, [], set())
        self.routes: dict[str, _UnitRoute] = {}

    def run(
        self,
        *,
        seeds: Sequence[EpisodeSeed],
        raw_chunk_ids: Sequence[str],
    ) -> tuple[_Walk, dict[str, _UnitRoute]]:
        """Execute the walk once and return its state plus the offered routes."""

        global_units = (
            self._scan_artifact_scope()
            if self.routing_scope == "artifact_global"
            else ()
        )
        trusted_chunks, neighbor_chunks = self._seed(seeds, raw_chunk_ids)
        self._seed_unit_routes(global_units, trusted_chunks, neighbor_chunks)
        pending_direct = self._direct_relations(trusted_chunks)
        self._expand(pending_direct)
        self._budget_receipts()
        return self.walk, self.routes

    def _witness(self, **kwargs) -> None:
        _record_witness(self.walk, **kwargs)

    def _probe(self, **kwargs) -> object:
        return _probe(self.walk, **kwargs)

    def _scan_artifact_scope(self) -> tuple[DiscourseUnit, ...]:
        """Witness artifact coverage and scan the artifact-wide unit scope."""

        def _coverage_outcome(coverage, failure_name):
            if coverage is not None:
                _validate_artifact(
                    coverage, self.artifact_id, "artifact coverage receipt"
                )
                if coverage.coverage_kind != "discourse":
                    raise ValueError("artifact coverage receipt has the wrong kind")
            coverage_current = bool(
                coverage is not None
                and coverage.source_revision == self.snapshot.source_revision
                and coverage.chunk_count == self.snapshot.chunk_count
            )
            detail = {
                "coverage_kind": "discourse",
                "failure": failure_name,
                "snapshot_source_revision": self.snapshot.source_revision,
                "snapshot_chunk_count": self.snapshot.chunk_count,
                "receipt_sha256": None if coverage is None else coverage.receipt_sha256,
                "coverage_sha256": None if coverage is None else coverage.coverage_sha256,
            }
            returned = 0 if coverage is None else coverage.chunk_count
            return coverage, [(self.artifact_id, returned, coverage_current, detail)]

        self._probe(
            kind="artifact_coverage",
            requested_limit=None,
            fetch=lambda: self.store.artifact_coverage(
                self.artifact_id,
                coverage_kind="discourse",
            ),
            admit=lambda coverage: _coverage_outcome(coverage, None),
            failure=lambda name: _coverage_outcome(None, name),
        )

        probe_limit = self.policy.max_units + 1
        scan = scan_artifact_units(
            self.store,
            artifact_id=self.artifact_id,
            program=self.program,
            max_units=self.policy.max_units,
        )
        self._witness(
            kind="artifact_unit_scan",
            subject_id=self.artifact_id,
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

    def _seed(
        self,
        seeds: Sequence[EpisodeSeed],
        raw_chunk_ids: Sequence[str],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Admit seed episodes plus temporal neighbors; witness their coverage."""

        trusted_episode_ids = self._seed_episodes(seeds, raw_chunk_ids)
        if trusted_episode_ids:
            episode_chunks = tuple(
                sorted(_episode_chunk_ids(self.walk, trusted_episode_ids))
            )
            self._annotation_coverage(
                chunks=episode_chunks,
                coverage_kind="episode",
                witness_kind="episode_coverage",
                completion_critical=False,
            )
        neighbor_ids = self._temporal_neighbors(trusted_episode_ids)
        self._witness(
            kind="episode_budget",
            subject_id=self.artifact_id,
            requested_limit=self.policy.max_frontier,
            returned_count=len(self.walk.episodes),
            exhaustive=True,
            detail={
                "seed_count": len(seeds),
                "neighbor_radius": self.policy.max_episode_neighbors,
                "route_truncated": not self.walk.episode_budget_exhaustive,
            },
        )
        trusted_chunks = tuple(
            sorted(
                set(raw_chunk_ids) | _episode_chunk_ids(self.walk, trusted_episode_ids)
            )
        )
        neighbor_chunks = tuple(
            sorted(_episode_chunk_ids(self.walk, neighbor_ids) - set(trusted_chunks))
        )
        graph_chunks = tuple(sorted(set(trusted_chunks) | set(neighbor_chunks)))
        self._annotation_coverage(
            chunks=graph_chunks,
            coverage_kind="discourse",
            witness_kind="annotation_coverage",
        )
        return trusted_chunks, neighbor_chunks

    def _seed_episodes(
        self,
        seeds: Sequence[EpisodeSeed],
        raw_chunk_ids: Sequence[str],
    ) -> tuple[str, ...]:
        candidates: dict[str, Episode] = {}
        for seed in seeds:
            episode = self.store.get_episode(seed.episode_id)
            if episode is None:
                raise ValueError(f"episode seed {seed.episode_id!r} does not exist")
            _validate_artifact(episode, self.artifact_id, "episode seed")
            if seed.anchor_chunk_id not in {
                span.chunk_id for span in episode.evidence
            }:
                raise ValueError(
                    f"episode seed {seed.episode_id!r} does not contain anchor "
                    f"chunk {seed.anchor_chunk_id!r}"
                )
            candidates.setdefault(seed.episode_id, episode)

        def _admit_mapping(
            mapped: Mapping[str, str],
        ) -> tuple[Mapping[str, str], _WitnessRows]:
            if any(chunk_id not in set(raw_chunk_ids) for chunk_id in mapped):
                raise ValueError(
                    "store returned an episode mapping for an unrequested chunk"
                )
            exhaustive = True
            failures: list[str] = []
            valid_count = 0
            for chunk_id in sorted(mapped):
                episode_id = mapped[chunk_id]
                try:
                    episode = self.store.get_episode(episode_id)
                except Exception as exc:
                    exhaustive = False
                    failures.append(type(exc).__name__)
                    continue
                if episode is None:
                    exhaustive = False
                    failures.append("missing_episode")
                    continue
                _validate_artifact(episode, self.artifact_id, "episode")
                if chunk_id not in {span.chunk_id for span in episode.evidence}:
                    raise ValueError("episode mapping does not contain its mapped chunk")
                candidates.setdefault(episode_id, episode)
                valid_count += 1
            detail = {"chunk_count": len(raw_chunk_ids), "failures": failures}
            return mapped, [(self.artifact_id, valid_count, exhaustive, detail)]

        mapped = self._probe(
            kind="episode_mapping",
            requested_limit=None,
            fetch=lambda: self.store.episode_ids_for_chunks(
                raw_chunk_ids,
                artifact_id=self.artifact_id,
            ),
            admit=_admit_mapping,
            failure=lambda name: ({}, [(self.artifact_id, 0, False, {
                "chunk_count": len(raw_chunk_ids),
                "failures": [name],
            })]),
        )

        episode_ids: list[str] = [seed.episode_id for seed in seeds]
        episode_ids.extend(mapped[chunk_id] for chunk_id in sorted(mapped))
        ordered_ids = tuple(dict.fromkeys(episode_ids))
        if len(ordered_ids) > self.policy.max_frontier:
            ordered_ids = ordered_ids[: self.policy.max_frontier]
            self.walk.episode_budget_exhaustive = False
        for episode_id in ordered_ids:
            episode = candidates.get(episode_id)
            if episode is None:
                # A failed optional annotation lookup is recorded above; the
                # verified raw chunk remains available to bundle assembly.
                continue
            self.walk.episodes[episode_id] = episode
            self.walk.episode_order.append(episode_id)
        return tuple(self.walk.episode_order)

    def _temporal_neighbors(
        self,
        trusted_episode_ids: Sequence[str],
    ) -> tuple[str, ...]:
        candidates: dict[str, Episode] = {}
        allowed_radius = self.policy.max_episode_neighbors
        probe_radius = allowed_radius + 1
        for episode_id in trusted_episode_ids:
            seed = self.walk.episodes.get(episode_id)
            if seed is None:
                continue

            def _admit_neighbors(
                neighbors: Sequence[Episode],
                *,
                seed: Episode = seed,
                episode_id: str = episode_id,
            ) -> tuple[tuple[Episode, ...], _WitnessRows]:
                prior: list[Episode] = []
                following: list[Episode] = []
                seen_neighbor_ids: set[str] = set()
                for neighbor in neighbors:
                    if neighbor.source_id != seed.source_id or neighbor.artifact_id != seed.artifact_id:
                        raise ValueError("temporal closure crossed a source or artifact boundary")
                    _validate_artifact(neighbor, self.artifact_id, "episode neighbor")
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
                detail = {
                    "artifact_id": self.artifact_id,
                    "source_id": seed.source_id,
                    "radius": allowed_radius,
                    "probe_radius": probe_radius,
                    "probe_count": len(neighbors),
                    "outside_requested_radius": outside_radius,
                }
                # This receipt is exhaustive for the requested local radius.
                # A corpus-wide unit scan is the global completeness proof;
                # farther episodes are diagnostic, not a hidden corpus tail.
                return admitted, [(episode_id, len(admitted), True, detail)]

            admitted = self._probe(
                kind="temporal_neighbors",
                requested_limit=None,
                fetch=lambda episode_id=episode_id: self.store.adjacent_episodes(
                    episode_id,
                    radius=probe_radius,
                ),
                admit=_admit_neighbors,
                failure=lambda name, episode_id=episode_id: ((), [(episode_id, 0, False, {
                    "radius": allowed_radius,
                    "probe_radius": probe_radius,
                    "failure": name,
                })]),
            )
            for neighbor in admitted:
                if neighbor.episode_id not in self.walk.episodes:
                    candidates.setdefault(neighbor.episode_id, neighbor)
        ordered = sorted(
            candidates.values(),
            key=lambda item: (item.source_id, item.sequence_no, item.episode_id),
        )
        available = max(0, self.policy.max_frontier - len(self.walk.episodes))
        if len(ordered) > available:
            ordered = ordered[:available]
            self.walk.episode_budget_exhaustive = False
        for episode in ordered:
            self.walk.episodes[episode.episode_id] = episode
            self.walk.episode_order.append(episode.episode_id)
        return tuple(episode.episode_id for episode in ordered)

    def _annotation_coverage(
        self,
        *,
        chunks: Sequence[str],
        coverage_kind: str,
        witness_kind: str,
        completion_critical: bool = True,
    ) -> None:
        def _admit_coverage(
            coverage: Mapping[str, str],
        ) -> tuple[None, _WitnessRows]:
            requested = set(chunks)
            if any(chunk_id not in requested for chunk_id in coverage):
                raise ValueError("coverage result contains an unrequested chunk")
            if any(status not in {"annotated", "no_output"} for status in coverage.values()):
                raise ValueError("coverage result contains an unsupported status")
            missing = tuple(sorted(requested - set(coverage)))
            detail = {
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
            }
            exhaustive = (not missing) or not completion_critical
            return None, [(self.artifact_id, len(coverage), exhaustive, detail)]

        self._probe(
            kind=witness_kind,
            requested_limit=None,
            fetch=lambda: self.store.coverage_for_chunks(
                self.artifact_id,
                chunks,
                coverage_kind=coverage_kind,
            ),
            admit=_admit_coverage,
            failure=lambda name: (None, [(self.artifact_id, 0, not completion_critical, {
                "chunk_count": len(chunks),
                "failure": name,
                "coverage_complete": False,
                "completion_critical": completion_critical,
            })]),
        )

    def _seed_unit_routes(
        self,
        global_units: Sequence[DiscourseUnit],
        trusted_chunks: Sequence[str],
        neighbor_chunks: Sequence[str],
    ) -> None:
        """Offer hop-0 routes from scope matches and seed-evidenced chunks."""

        if self.routing_scope == "artifact_global":
            for unit in global_units:
                if unit_obligation_ids(unit, self.program, connected=False):
                    _offer_route(self.routes, unit, hop=0, route_rank=3, connected=True)
        candidate_limit = self.policy.max_units + 1
        for unit in self._units_for_chunks(
            trusted_chunks,
            route_rank=0,
            witness_subject="trusted_chunks",
            limit=candidate_limit,
        ):
            matched = bool(unit_obligation_ids(unit, self.program, connected=False))
            _offer_route(self.routes, unit, hop=0, route_rank=0, connected=matched)
        if neighbor_chunks:
            for unit in self._units_for_chunks(
                neighbor_chunks,
                route_rank=2,
                witness_subject="temporal_chunks",
                limit=candidate_limit,
            ):
                _offer_route(self.routes, unit, hop=0, route_rank=2, connected=False)

    def _units_for_chunks(
        self,
        chunks: Sequence[str],
        *,
        route_rank: int,
        witness_subject: str,
        limit: int,
    ) -> tuple[DiscourseUnit, ...]:
        def _admit_units(
            probed: Sequence[DiscourseUnit],
        ) -> tuple[tuple[DiscourseUnit, ...], _WitnessRows]:
            if len(probed) > limit:
                raise ValueError(PROBE_OVERFLOW_MESSAGE)
            _validate_artifacts(probed, self.artifact_id, "unit")
            validate_chunk_scoped_rows(probed, chunks, kind="unit")
            ordered = sorted(
                probed,
                key=lambda unit: unit_priority(
                    unit,
                    self.program,
                    connected=bool(
                        unit_obligation_ids(unit, self.program, connected=False)
                    ),
                    route_rank=route_rank,
                ),
            )
            admitted = tuple(ordered[: self.policy.max_units])
            detail = {
                "artifact_id": self.artifact_id,
                "chunk_count": len(chunks),
                "probe_limit": limit,
                "probe_count": len(probed),
            }
            exhaustive = len(ordered) <= self.policy.max_units
            return admitted, [(witness_subject, len(admitted), exhaustive, detail)]

        return self._probe(
            kind="chunk_unit_lookup",
            requested_limit=self.policy.max_units,
            fetch=lambda: self.store.units_for_chunks(
                chunks,
                artifact_id=self.artifact_id,
                limit=limit,
            ),
            admit=_admit_units,
            failure=lambda name: ((), [(witness_subject, 0, False, {"failure": name})]),
        )

    def _direct_relations(
        self,
        trusted_chunks: Sequence[str],
    ) -> dict[str, DiscourseRelation]:
        """Probe relations anchored on trusted chunks and offer member routes."""

        relation_probe_limit = self.policy.max_relations + 1
        priority_units = {
            unit_id: route.unit for unit_id, route in self.routes.items()
        }

        def _admit_direct_relations(
            probed: Sequence[DiscourseRelation],
        ) -> tuple[tuple[DiscourseRelation, ...], _WitnessRows]:
            if len(probed) > relation_probe_limit:
                raise ValueError(PROBE_OVERFLOW_MESSAGE)
            _validate_artifacts(probed, self.artifact_id, "relation")
            validate_chunk_scoped_rows(probed, trusted_chunks, kind="relation")
            valid: list[DiscourseRelation] = []
            for relation in probed:
                try:
                    for member in relation.members:
                        if member.unit_id not in priority_units:
                            priority_units[member.unit_id] = _get_member_unit(
                                self.store,
                                member.unit_id,
                                artifact_id=self.artifact_id,
                            )
                except _AnnotationLookupFailure as failure:
                    _record_member_failure(self.walk, relation.relation_id, failure)
                    continue
                valid.append(relation)
            admitted = tuple(
                sorted(
                    valid,
                    key=lambda relation: relation_priority(
                        relation,
                        self.program,
                        priority_units,
                        min_confidence=self.policy.min_relation_confidence,
                    ),
                )[: self.policy.max_relations]
            )
            detail = {
                "artifact_id": self.artifact_id,
                "chunk_count": len(trusted_chunks),
                "probe_limit": relation_probe_limit,
                "probe_count": len(probed),
            }
            exhaustive = len(probed) <= self.policy.max_relations
            return admitted, [("trusted_chunks", len(admitted), exhaustive, detail)]

        direct_relations = self._probe(
            kind="chunk_relation_lookup",
            requested_limit=self.policy.max_relations,
            fetch=lambda: self.store.relations_for_chunks(
                trusted_chunks,
                artifact_id=self.artifact_id,
                limit=relation_probe_limit,
            ),
            admit=_admit_direct_relations,
            failure=lambda name: ((), [("trusted_chunks", 0, False, {"failure": name})]),
        )
        # Relation members are valid starting points even when a unit's own
        # evidence is outside the seed chunk.
        for relation in direct_relations:
            if not _as_of_relation(relation, self.program):
                continue
            for member in relation.members:
                unit = priority_units[member.unit_id]
                # The relation may still lose a bounded priority cut.  It
                # confers semantic credit only once it is actually visited.
                matched = bool(unit_obligation_ids(unit, self.program, connected=False))
                _offer_route(self.routes, unit, hop=0, route_rank=1, connected=matched)
        return {
            relation.relation_id: relation
            for relation in direct_relations
            if _as_of_relation(relation, self.program)
        }

    def _expand(self, pending_direct: dict[str, DiscourseRelation]) -> None:
        """Visit frontier units hop by hop, admitting bounded relation cuts."""

        for hop in range(self.policy.max_hops + 1):
            current = self._frontier(hop)
            if not current and not (hop == 0 and pending_direct):
                continue
            for route in current:
                unit_id = route.unit.unit_id
                self.walk.units[unit_id] = route.unit
                self.walk.unit_order.append(unit_id)
                self.walk.max_visited_hop = max(self.walk.max_visited_hop, hop)
                if route.connected:
                    self.walk.connected_unit_ids.add(unit_id)
            incident = self._incident_relations(
                tuple(route.unit.unit_id for route in current)
            )
            self._admit_relations(
                hop,
                incident,
                pending_direct=pending_direct if hop == 0 else {},
            )
        if any(
            route.unit.unit_id not in self.walk.units
            and _as_of_unit(route.unit, self.program)
            for route in self.routes.values()
        ):
            self.walk.unit_budget_exhaustive = False

    def _frontier(self, hop: int) -> list[_UnitRoute]:
        """Select this hop's bounded, priority-ordered unit frontier."""

        current = [
            route
            for route in self.routes.values()
            if route.hop == hop
            and route.unit.unit_id not in self.walk.units
            and _as_of_unit(route.unit, self.program)
        ]
        current.sort(
            key=lambda route: unit_priority(
                route.unit,
                self.program,
                connected=route.connected,
                route_rank=route.route_rank,
            )
        )
        frontier_limit = min(self.policy.beam_width, self.policy.max_frontier)
        frontier_exhaustive = len(current) <= frontier_limit
        current = current[:frontier_limit]
        if current:
            self._witness(
                kind="unit_frontier",
                subject_id=f"{self.artifact_id}:hop:{hop}",
                requested_limit=frontier_limit,
                returned_count=len(current),
                exhaustive=frontier_exhaustive,
                detail={
                    "beam_width": self.policy.beam_width,
                    "max_frontier": self.policy.max_frontier,
                },
            )
        remaining_units = self.policy.max_units - len(self.walk.units)
        if len(current) > remaining_units:
            current = current[: max(0, remaining_units)]
            self.walk.unit_budget_exhaustive = False
        return current

    def _incident_relations(
        self,
        requested_unit_ids: Sequence[str],
    ) -> Mapping[str, Sequence[DiscourseRelation]]:
        """Probe bounded incident relations for the just-visited units."""

        if not requested_unit_ids:
            return {}

        def _admit_incident(
            probed: Mapping[str, Sequence[DiscourseRelation]],
        ) -> tuple[Mapping[str, Sequence[DiscourseRelation]], _WitnessRows]:
            _validate_incident(probed, requested_unit_ids, artifact_id=self.artifact_id)
            rows: list[_WitnessRow] = []
            for unit_id in requested_unit_ids:
                count = len(probed[unit_id])
                if count > self.policy.max_degree + 1:
                    raise ValueError(PROBE_OVERFLOW_MESSAGE)
                rows.append((
                    unit_id,
                    min(count, self.policy.max_degree),
                    count <= self.policy.max_degree,
                    {
                        "artifact_id": self.artifact_id,
                        "probe_limit": self.policy.max_degree + 1,
                        "probe_count": count,
                    },
                ))
            return probed, rows

        return self._probe(
            kind="incident_relations",
            requested_limit=self.policy.max_degree,
            fetch=lambda: self.store.incident_relations(
                requested_unit_ids,
                artifact_id=self.artifact_id,
                max_degree=self.policy.max_degree + 1,
            ),
            admit=_admit_incident,
            failure=lambda name: (
                {unit_id: () for unit_id in requested_unit_ids},
                [(unit_id, 0, False, {"failure": name}) for unit_id in requested_unit_ids],
            ),
        )

    def _admit_relations(
        self,
        hop: int,
        incident: Mapping[str, Sequence[DiscourseRelation]],
        *,
        pending_direct: Mapping[str, DiscourseRelation],
    ) -> None:
        """Admit this hop's relation budget cut and offer next-hop routes."""

        probed_candidates: dict[str, DiscourseRelation] = dict(pending_direct)
        for unit_id in sorted(incident):
            for relation in incident[unit_id]:
                probed_candidates.setdefault(relation.relation_id, relation)

        known_units = dict(self.walk.units)
        invalid_relation_ids: set[str] = set()
        for relation in probed_candidates.values():
            try:
                for member in relation.members:
                    if member.unit_id not in known_units:
                        known_units[member.unit_id] = _get_member_unit(
                            self.store,
                            member.unit_id,
                            artifact_id=self.artifact_id,
                        )
            except _AnnotationLookupFailure as failure:
                invalid_relation_ids.add(relation.relation_id)
                _record_member_failure(self.walk, relation.relation_id, failure)
        candidates: dict[str, DiscourseRelation] = {
            relation_id: relation
            for relation_id, relation in pending_direct.items()
            if relation_id not in invalid_relation_ids
        }
        for unit_id in sorted(incident):
            ordered_incident = sorted(
                (
                    relation
                    for relation in incident[unit_id]
                    if relation.relation_id not in invalid_relation_ids
                ),
                key=lambda relation: relation_priority(
                    relation,
                    self.program,
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
                if relation.relation_id not in self.walk.relations
                and _as_of_relation(relation, self.program)
                and relation_is_useful(
                    relation,
                    self.program,
                    known_units,
                    connected=True,
                )
            ),
            key=lambda relation: relation_priority(
                relation,
                self.program,
                known_units,
                min_confidence=self.policy.min_relation_confidence,
            ),
        )
        remaining_relations = self.policy.max_relations - len(self.walk.relations)
        if len(ordered_relations) > remaining_relations:
            ordered_relations = ordered_relations[: max(0, remaining_relations)]
            self.walk.relation_budget_exhaustive = False
        for relation in ordered_relations:
            self._visit_relation(relation, hop=hop, known_units=known_units)

    def _visit_relation(
        self,
        relation: DiscourseRelation,
        *,
        hop: int,
        known_units: Mapping[str, DiscourseUnit],
    ) -> None:
        """Record one admitted relation and propagate member connectivity."""

        self.walk.relations[relation.relation_id] = relation
        self.walk.relation_order.append(relation.relation_id)
        member_units = [
            known_units[member.unit_id]
            for member in relation.members
            if member.unit_id in known_units
        ]
        anchor_connected = any(
            unit.unit_id in self.walk.connected_unit_ids
            or ((route := self.routes.get(unit.unit_id)) is not None and route.connected)
            or bool(unit_obligation_ids(unit, self.program, connected=False))
            for unit in member_units
        )
        edge_connects = anchor_connected and relation_confers_subject_connection(
            relation,
            self.program,
            min_confidence=self.policy.min_relation_confidence,
        )
        if edge_connects:
            self.walk.connected_relation_ids.add(relation.relation_id)
        for member in relation.members:
            unit = known_units.get(member.unit_id)
            if unit is None or not _as_of_unit(unit, self.program):
                continue
            member_connects = edge_connects or bool(
                unit_obligation_ids(unit, self.program, connected=False)
            )
            if member_connects:
                self.walk.connected_unit_ids.add(unit.unit_id)
            if unit.unit_id in self.walk.units:
                continue
            if hop >= self.policy.max_hops:
                self.walk.hop_budget_exhaustive = False
                continue
            _offer_route(
                self.routes,
                unit,
                hop=hop + 1,
                route_rank=1,
                connected=member_connects,
            )

    def _budget_receipts(self) -> None:
        """Emit the walk-wide unit, relation, and hop budget witnesses."""

        self._witness(
            kind="unit_budget",
            subject_id=self.artifact_id,
            requested_limit=self.policy.max_units,
            returned_count=len(self.walk.units),
            exhaustive=self.walk.unit_budget_exhaustive,
        )
        self._witness(
            kind="relation_budget",
            subject_id=self.artifact_id,
            requested_limit=self.policy.max_relations,
            returned_count=len(self.walk.relations),
            exhaustive=self.walk.relation_budget_exhaustive,
        )
        self._witness(
            kind="hop_budget",
            subject_id=self.artifact_id,
            requested_limit=self.policy.max_hops,
            returned_count=min(self.walk.max_visited_hop, self.policy.max_hops),
            exhaustive=self.walk.hop_budget_exhaustive,
        )


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
    routing_scope: ClosureRoutingScope = "artifact_global",
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
        routing_scope=routing_scope,
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


def _probe(
    walk: _Walk,
    *,
    kind: str,
    requested_limit: int | None,
    fetch: Callable[[], object],
    admit: Callable[..., tuple[object, _WitnessRows]],
    failure: Callable[..., tuple[object, _WitnessRows]],
) -> object:
    """Run one bounded store probe; every outcome ends in recorded witnesses.

    ``admit`` validates the probed value (raising on invariant violations) and
    returns the admitted value plus its witness rows; ``failure`` maps a store
    failure name to a fallback value plus its witness rows.  Because both
    branches must yield rows that are recorded here, a probe site cannot
    forget the scope witness that ``completion()`` relies on.
    """

    try:
        probed = fetch()
    except Exception as exc:
        value, rows = failure(type(exc).__name__)
    else:
        value, rows = admit(probed)
    for subject_id, returned_count, exhaustive, detail in rows:
        _record_witness(
            walk,
            kind=kind,
            subject_id=subject_id,
            requested_limit=requested_limit,
            returned_count=returned_count,
            exhaustive=exhaustive,
            detail=detail,
        )
    return value


def _episode_chunk_ids(walk: _Walk, episode_ids: Iterable[str]) -> set[str]:
    """Chunk ids evidenced by the given already-visited episodes."""

    return {
        span.chunk_id
        for episode_id in episode_ids
        if episode_id in walk.episodes
        for span in walk.episodes[episode_id].evidence
    }


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


__all__ = ["ClosureRoutingScope", "EvidenceClosureEngine", "close_evidence"]
