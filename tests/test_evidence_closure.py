"""Synthetic adversaries for domain-neutral episodic discourse closure."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import (
    ArtifactCoverageReceipt,
    ClosurePolicy,
    DiscourseRelation,
    DiscourseSnapshot,
    DiscourseUnit,
    Episode,
    EpisodeSeed,
    EvidenceObligation,
    EvidenceSpan,
    QueryProgram,
    RelationMember,
    make_bundle_id,
    quote_sha256,
)
from memory_condense.domain.discourse_routing import DiscourseUnitRoute
from memory_condense.search.closure import (
    EvidenceClosureStore,
    close_evidence,
    compile_query_program,
    infer_intent,
)
from memory_condense.search.packing.evidence_packet import (
    pack_evidence_plan,
    render_evidence_context,
)


@dataclass
class _MemoryGraph:
    reverse_reads: bool = False
    chunks: dict[str, tuple[str, int, str, int]] = field(default_factory=dict)
    episodes: dict[str, Episode] = field(default_factory=dict)
    units: dict[str, DiscourseUnit] = field(default_factory=dict)
    relations: dict[str, DiscourseRelation] = field(default_factory=dict)
    mutate_snapshot: bool = False
    artifact_ids: tuple[str, ...] = ("artifact-a",)
    _snapshot_calls: int = 0

    def add_chunk(
        self,
        chunk_id: str,
        text: str,
        ordinal: int,
        source_id: str = "source-a",
        *,
        turn_start_char: int = 0,
    ) -> EvidenceSpan:
        self.chunks[chunk_id] = (text, ordinal, source_id, turn_start_char)
        return EvidenceSpan(
            chunk_id=chunk_id,
            start_char=0,
            end_char=len(text),
            quote_sha256=quote_sha256(text),
            ordinal=ordinal,
            source_id=source_id,
            turn_start_char=turn_start_char,
        )

    def add_episode(
        self,
        episode_id: str,
        sequence_no: int,
        spans: Sequence[EvidenceSpan],
        *,
        source_id: str = "source-a",
        artifact_id: str = "artifact-a",
    ) -> Episode:
        episode = Episode(
            episode_id=episode_id,
            artifact_id=artifact_id,
            source_id=source_id,
            sequence_no=sequence_no,
            first_ordinal=spans[0].ordinal,
            last_ordinal=spans[-1].ordinal,
            evidence=tuple(spans),
            boundary_method="fixture",
        )
        self.episodes[episode_id] = episode
        return episode

    def add_unit(
        self,
        unit_id: str,
        kind: str,
        canonical_key: str,
        span: EvidenceSpan,
        *,
        confidence: float = 1.0,
        artifact_id: str = "artifact-a",
    ) -> DiscourseUnit:
        unit = DiscourseUnit(
            unit_id=unit_id,
            artifact_id=artifact_id,
            kind=kind,
            canonical_key=canonical_key,
            asserted_ordinal=span.ordinal,
            confidence=confidence,
            evidence=(span,),
        )
        self.units[unit_id] = unit
        return unit

    def add_relation(
        self,
        relation_id: str,
        relation_type: str,
        members: Sequence[tuple[str, str]],
        evidence: EvidenceSpan,
        *,
        confidence: float = 1.0,
        created_ordinal: int | None = None,
        artifact_id: str = "artifact-a",
    ) -> DiscourseRelation:
        relation = DiscourseRelation(
            relation_id=relation_id,
            artifact_id=artifact_id,
            relation_type=relation_type,
            members=tuple(
                RelationMember(unit_id=unit_id, role=role, ordinal=index)
                for index, (unit_id, role) in enumerate(members)
            ),
            evidence=(evidence,),
            confidence=confidence,
            created_ordinal=evidence.ordinal if created_ordinal is None else created_ordinal,
        )
        self.relations[relation_id] = relation
        return relation

    def snapshot(self, graph_revision: int | None = None) -> DiscourseSnapshot:
        self._snapshot_calls += 1
        revision = graph_revision or 1
        if self.mutate_snapshot and self._snapshot_calls > 1:
            revision += 1
        return DiscourseSnapshot(
            max_turn_ordinal=max((row[1] for row in self.chunks.values()), default=0),
            chunk_count=len(self.chunks),
            graph_revision=revision,
            schema_version=10,
            artifact_ids=self.artifact_ids,
            source_content_sha256="1" * 64,
            graph_content_sha256="2" * 64,
        )

    def get_episode(self, episode_id: str) -> Episode | None:
        return self.episodes.get(episode_id)

    def episode_ids_for_chunks(
        self,
        chunk_ids: Sequence[str],
        *,
        artifact_id: str | None = None,
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        for chunk_id in chunk_ids:
            matches = sorted(
                episode.episode_id
                for episode in self.episodes.values()
                if any(span.chunk_id == chunk_id for span in episode.evidence)
                and artifact_id in (None, episode.artifact_id)
            )
            if matches:
                result[chunk_id] = matches[0]
        return result

    def adjacent_episodes(
        self,
        episode_id: str,
        *,
        radius: int = 1,
        include_self: bool = False,
    ) -> tuple[Episode, ...]:
        seed = self.episodes[episode_id]
        source_values = [
            episode
            for episode in self.episodes.values()
            if episode.artifact_id == seed.artifact_id
            and episode.source_id == seed.source_id
        ]
        prior = sorted(
            (
                episode
                for episode in source_values
                if episode.sequence_no < seed.sequence_no
            ),
            key=lambda item: (-item.sequence_no, item.episode_id),
        )[:radius]
        following = sorted(
            (
                episode
                for episode in source_values
                if episode.sequence_no > seed.sequence_no
            ),
            key=lambda item: (item.sequence_no, item.episode_id),
        )[:radius]
        values = sorted(
            (*prior, *((seed,) if include_self else ()), *following),
            key=lambda item: (item.sequence_no, item.episode_id),
        )
        if self.reverse_reads:
            values.reverse()
        return tuple(values)

    def coverage_for_chunks(
        self,
        artifact_id: str,
        chunk_ids: Sequence[str],
        *,
        coverage_kind: str = "discourse",
    ) -> dict[str, str]:
        assert artifact_id in self.artifact_ids
        assert coverage_kind in {"episode", "discourse"}
        owners = (
            tuple(self.episodes.values())
            if coverage_kind == "episode"
            else (*self.units.values(), *self.relations.values())
        )
        annotated = {
            span.chunk_id
            for owner in owners
            if owner.artifact_id == artifact_id
            for span in owner.evidence
        }
        return {
            chunk_id: "annotated" if chunk_id in annotated else "no_output"
            for chunk_id in dict.fromkeys(chunk_ids)
        }

    def artifact_coverage(
        self,
        artifact_id: str,
        coverage_kind: str = "discourse",
    ) -> ArtifactCoverageReceipt | None:
        assert artifact_id in self.artifact_ids
        return ArtifactCoverageReceipt(
            artifact_id=artifact_id,
            coverage_kind=coverage_kind,
            source_revision=0,
            chunk_count=len(self.chunks),
            coverage_sha256="a" * 64,
        )

    def units_for_artifact(
        self,
        artifact_id: str,
        *,
        limit: int | None = None,
    ) -> tuple[DiscourseUnit, ...]:
        values = [
            unit for unit in self.units.values() if unit.artifact_id == artifact_id
        ]
        values.sort(key=lambda item: (-item.asserted_ordinal, item.unit_id))
        if self.reverse_reads:
            values.reverse()
        return tuple(values if limit is None else values[:limit])

    def iter_unit_routes_for_artifact(
        self,
        artifact_id: str,
    ):
        values = list(self.units_for_artifact(artifact_id))
        yield from (DiscourseUnitRoute.from_unit(unit) for unit in values)

    def units_for_chunks(
        self,
        chunk_ids: Sequence[str],
        *,
        artifact_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[DiscourseUnit, ...]:
        selected = set(chunk_ids)
        values = [
            unit
            for unit in self.units.values()
            if any(span.chunk_id in selected for span in unit.evidence)
            and artifact_id in (None, unit.artifact_id)
        ]
        values.sort(key=lambda item: (-item.asserted_ordinal, item.unit_id))
        if self.reverse_reads:
            values.reverse()
        return tuple(values if limit is None else values[:limit])

    def relations_for_chunks(
        self,
        chunk_ids: Sequence[str],
        *,
        artifact_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[DiscourseRelation, ...]:
        selected = set(chunk_ids)
        values = [
            relation
            for relation in self.relations.values()
            if any(span.chunk_id in selected for span in relation.evidence)
            and artifact_id in (None, relation.artifact_id)
        ]
        values.sort(key=lambda item: (-item.confidence, -item.created_ordinal, item.relation_id))
        if self.reverse_reads:
            values.reverse()
        return tuple(values if limit is None else values[:limit])

    def get_unit(self, unit_id: str) -> DiscourseUnit | None:
        return self.units.get(unit_id)

    def get_relation(self, relation_id: str) -> DiscourseRelation | None:
        return self.relations.get(relation_id)

    def incident_relations(
        self,
        unit_ids: Sequence[str],
        *,
        artifact_id: str | None = None,
        max_degree: int,
    ) -> dict[str, tuple[DiscourseRelation, ...]]:
        result: dict[str, tuple[DiscourseRelation, ...]] = {}
        for unit_id in unit_ids:
            values = [
                relation
                for relation in self.relations.values()
                if any(member.unit_id == unit_id for member in relation.members)
                and artifact_id in (None, relation.artifact_id)
            ]
            values.sort(key=lambda item: (-item.confidence, -item.created_ordinal, item.relation_id))
            if self.reverse_reads:
                values.reverse()
            result[unit_id] = tuple(values[:max_degree])
        return result

    def evidence_for_chunks(self, chunk_ids: Sequence[str]) -> tuple[EvidenceSpan, ...]:
        return tuple(self._span(chunk_id) for chunk_id in dict.fromkeys(chunk_ids))

    def hydrate_span(self, span: EvidenceSpan) -> str:
        text, ordinal, source_id, turn_start_char = self.chunks[span.chunk_id]
        assert ordinal == span.ordinal
        assert span.source_id in (None, source_id)
        assert span.turn_start_char == turn_start_char
        value = text[span.start_char : span.end_char]
        assert quote_sha256(value) == span.quote_sha256
        return value

    def _span(self, chunk_id: str) -> EvidenceSpan:
        text, ordinal, source_id, turn_start_char = self.chunks[chunk_id]
        return EvidenceSpan(
            chunk_id=chunk_id,
            start_char=0,
            end_char=len(text),
            quote_sha256=quote_sha256(text),
            ordinal=ordinal,
            source_id=source_id,
            turn_start_char=turn_start_char,
        )


def _connected_graph(*, reverse_reads: bool = False) -> tuple[_MemoryGraph, EpisodeSeed]:
    graph = _MemoryGraph(reverse_reads=reverse_reads)
    state = graph.add_chunk("c-state", "Request throughput is presently below target.", 10)
    noise = graph.add_chunk("c-noise", "An unrelated social event changed its menu.", 11)
    objective = graph.add_chunk("c-objective", "Success means stable throughput under load.", 30)
    constraint = graph.add_chunk("c-constraint", "The memory ceiling is binding.", 60)
    old = graph.add_chunk("c-old", "The first choice used a broad cache.", 90)
    new = graph.add_chunk("c-new", "The choice was revised to a bounded cache.", 120)
    observation = graph.add_chunk("c-observation", "The bounded run reduced tail delay.", 150)
    failure = graph.add_chunk("c-failure", "A competing run exhausted memory.", 180)
    dependency = graph.add_chunk("c-dependency", "The change depends on stable input ordering.", 210)
    issue = graph.add_chunk("c-issue", "The effect under burst traffic remains unresolved.", 240)
    resolution = graph.add_chunk("c-resolution", "The bounded result resolves the competing claims.", 270)

    graph.add_episode("ep-state", 0, (state,))
    graph.add_episode("ep-noise", 1, (noise,))
    graph.add_unit("u-state", "status", "request throughput baseline", state)
    graph.add_unit("u-noise", "constraint", "unrelated menu choice", noise)
    graph.add_unit("u-objective", "goal", "stable request throughput", objective)
    graph.add_unit("u-constraint", "limitation", "bounded memory ceiling", constraint)
    graph.add_unit("u-old", "decision", "broad cache choice", old)
    graph.add_unit("u-new", "decision", "bounded cache choice", new)
    graph.add_unit("u-observation", "result", "tail delay measurement", observation)
    graph.add_unit("u-failure", "failure", "memory exhaustion", failure)
    graph.add_unit("u-dependency", "dependency", "stable input ordering", dependency)
    graph.add_unit("u-issue", "open_question", "burst traffic effect", issue)
    graph.add_unit("u-resolution", "resolution", "bounded result accepted", resolution)

    graph.add_relation("r-01", "supports", (("u-state", "evidence"), ("u-objective", "claim")), state, confidence=0.2)
    graph.add_relation("r-02", "requires", (("u-objective", "dependent"), ("u-constraint", "requirement")), objective)
    graph.add_relation("r-03", "requires", (("u-constraint", "requirement"), ("u-old", "dependent")), constraint)
    graph.add_relation("r-04", "revises", (("u-old", "old"), ("u-new", "new")), new)
    graph.add_relation("r-05", "tests", (("u-new", "tested"), ("u-observation", "result")), observation)
    graph.add_relation("r-06", "depends_on", (("u-new", "dependent"), ("u-dependency", "requirement")), dependency)
    graph.add_relation("r-07", "addresses", (("u-new", "action"), ("u-issue", "issue")), issue)
    graph.add_relation("r-08", "contradicts", (("u-observation", "side_a"), ("u-failure", "side_b")), failure)
    graph.add_relation(
        "r-09",
        "resolves",
        (("u-observation", "side_a"), ("u-failure", "side_b"), ("u-resolution", "resolution")),
        resolution,
    )
    return graph, EpisodeSeed(
        episode_id="ep-state",
        anchor_chunk_id="c-state",
        score=0.91,
        route="lexical",
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What value was recorded?", "lookup"),
        ("List all recorded entries.", "enumerate"),
        ("Compare the two alternatives.", "compare"),
        ("Explain how the outcome changed.", "explain"),
        ("Diagnose the root cause of the failure.", "diagnose"),
        ("What should we improve next?", "recommend"),
        ("Create an action plan for the transition.", "plan"),
        ("What is the current status of the transition?", "status"),
    ],
)
def test_query_intent_grammar_is_domain_neutral(query: str, expected: str):
    assert infer_intent(query) == expected
    assert compile_query_program(query).intent == expected


def test_recommendation_program_has_the_full_conservative_obligation_set():
    first = compile_query_program("How should we improve the response process?")
    second = compile_query_program("How should we improve the response process?")
    assert first == second
    assert first.program_sha256 == second.program_sha256
    assert [item.obligation_id for item in first.obligations] == [
        "objective",
        "current_state",
        "constraints",
        "decisions",
        "observations",
        "failures",
        "dependencies",
        "unresolved_issues",
        "revisions_conflicts",
    ]
    assert all(item.required for item in first.obligations)


def test_open_enumeration_requires_scope_evidence_but_fixed_count_does_not():
    open_program = compile_query_program("List all recorded entries.")
    fixed_program = compile_query_program("List three recorded entries.")
    assert [item.obligation_id for item in open_program.obligations] == [
        "members",
        "enumeration_scope",
    ]
    assert fixed_program.cardinality == 3
    assert [item.obligation_id for item in fixed_program.obligations] == ["members"]
    assert fixed_program.obligations[0].min_count == 3


@pytest.mark.parametrize(
    "query",
    (
        "List HTTP/2 features.",
        "List issues after 2024.",
        "Identify Python 3.12 compatibility issues.",
    ),
)
def test_enumeration_cardinality_ignores_versions_dates_and_identifiers(query):
    assert compile_query_program(query).cardinality is None


@pytest.mark.parametrize(
    ("query", "expected"),
    (
        ("List the three retrieval failures.", 3),
        ("Give me five options.", 5),
        ("Show the top 10 results.", 10),
        ("Which are the four retrieval options?", 4),
    ),
)
def test_enumeration_cardinality_requires_explicit_count_grammar(query, expected):
    assert compile_query_program(query).cardinality == expected


def test_manual_query_program_is_used_without_recompilation():
    manual = QueryProgram(
        query="Return the verified record.",
        intent="lookup",
        subject_terms=("record",),
        obligations=(
            EvidenceObligation(
                obligation_id="manual-proof",
                kind="manual",
                required=True,
                weight=2.0,
                unit_kinds=("custom_kind",),
            ),
        ),
    )
    assert compile_query_program(manual) is manual
    with pytest.raises(ValueError, match="cannot be partially overridden"):
        compile_query_program(manual, intent="lookup")


def test_engineering_discourse_closes_diffuse_evidence_and_terminal_revision():
    graph, seed = _connected_graph()
    assert isinstance(graph, EvidenceClosureStore)
    plan = close_evidence(
        graph,
        "How should we improve request throughput?",
        seeds=(seed,),
        policy=ClosurePolicy(max_hops=6, min_relation_confidence=0.8),
    )

    assert plan.complete_claimed is True
    assert plan.stopping_reason == "complete"
    assert {result.status for result in plan.obligation_results} == {"satisfied"}
    assert "ep-noise" in plan.visited_episode_ids
    decisions = next(
        result for result in plan.obligation_results if result.obligation_id == "decisions"
    )
    assert "u-new" in decisions.unit_ids
    assert "u-old" not in decisions.unit_ids
    revision = next(
        result
        for result in plan.obligation_results
        if result.obligation_id == "revisions_conflicts"
    )
    assert {"r-04", "r-08", "r-09"} <= set(revision.relation_ids)
    relation_sets = {frozenset(bundle.relation_ids) for bundle in plan.bundles}
    assert frozenset({"r-04"}) in relation_sets
    assert frozenset({"r-08", "r-09"}) in relation_sets
    assert not any(
        {"r-04", "r-08", "r-09"} <= set(bundle.relation_ids)
        for bundle in plan.bundles
    )
    assert any("c-state" == atom.span.chunk_id for atom in plan.atoms)
    assert not any("c-noise" == atom.span.chunk_id for atom in plan.atoms)
    packet = pack_evidence_plan(plan, max_context_tokens=2_000)
    assert packet.receipt.complete_claimed is True
    assert packet.receipt.context_token_proxy <= 2_000
    assert packet.receipt.retained_request_token_state_bytes == 0


def test_store_return_order_cannot_change_the_closure_receipt():
    first, seed = _connected_graph(reverse_reads=False)
    second, other_seed = _connected_graph(reverse_reads=True)
    policy = ClosurePolicy(max_hops=6, max_degree=16)
    left = close_evidence(first, "How should we improve request throughput?", seeds=(seed,), policy=policy)
    right = close_evidence(second, "How should we improve request throughput?", seeds=(other_seed,), policy=policy)
    assert left.identity_payload() == right.identity_payload()
    assert left.plan_sha256 == right.plan_sha256


def test_same_turn_atoms_follow_authoritative_character_order_not_chunk_id():
    graph = _MemoryGraph()
    graph.add_chunk(
        "z-first",
        "First source slice.",
        7,
        turn_start_char=0,
    )
    graph.add_chunk(
        "a-second",
        "Second source slice.",
        7,
        turn_start_char=20,
    )
    plan = close_evidence(
        graph,
        "What was recorded?",
        direct_chunk_ids=("a-second", "z-first"),
    )
    assert [atom.span.chunk_id for atom in plan.atoms] == ["z-first", "a-second"]
    assert [atom.span.turn_start_char for atom in plan.atoms] == [0, 20]


def test_atom_order_uses_global_ordinal_before_source_label():
    graph = _MemoryGraph()
    graph.add_chunk("early", "Earlier slice.", 1, "source-z")
    graph.add_chunk("late", "Later slice.", 2, "source-a")
    plan = close_evidence(
        graph,
        "What was recorded?",
        direct_chunk_ids=("late", "early"),
    )
    assert [atom.span.chunk_id for atom in plan.atoms] == ["early", "late"]


def test_low_confidence_edge_can_widen_but_cannot_displace_direct_raw_hit():
    graph, seed = _connected_graph()
    plan = close_evidence(
        graph,
        "How should we improve request throughput?",
        seeds=(seed,),
        policy=ClosurePolicy(
            max_hops=6,
            max_bundles=1,
            min_relation_confidence=0.95,
        ),
    )
    assert "r-01" in plan.visited_relation_ids
    assert "u-objective" in plan.visited_unit_ids
    assert len(plan.bundles) == 1
    assert plan.bundles[0].atom_ids == tuple(
        atom.atom_id for atom in plan.atoms if atom.span.chunk_id == "c-state"
    )
    assert {atom.span.chunk_id for atom in plan.atoms} == {"c-state"}
    assert plan.stopping_reason == "workspace_cap"
    assert plan.complete_claimed is False


@pytest.mark.parametrize(
    ("relation_type", "confidence", "should_connect"),
    [
        ("supports", 0.2, False),
        ("sequence", 0.99, False),
        ("supports", 0.99, True),
    ],
)
def test_only_strong_semantic_edges_confer_subject_closure_credit(
    relation_type: str,
    confidence: float,
    should_connect: bool,
):
    graph = _MemoryGraph()
    anchor = graph.add_chunk("anchor", "The release status is under review.", 1)
    unrelated = graph.add_chunk("unrelated", "A distant measurement concerns another subject.", 8)
    graph.add_unit("status", "status", "release current state", anchor)
    graph.add_unit("unrelated-measurement", "measurement", "different subject", unrelated)
    graph.add_relation(
        "edge",
        relation_type,
        (("status", "anchor"), ("unrelated-measurement", "neighbor")),
        anchor,
        confidence=confidence,
    )
    plan = close_evidence(
        graph,
        "What is the current status of the release plan?",
        direct_chunk_ids=("anchor",),
        policy=ClosurePolicy(min_relation_confidence=0.8),
    )
    observation = next(
        result
        for result in plan.obligation_results
        if result.obligation_id == "observations"
    )
    assert "unrelated-measurement" in plan.visited_unit_ids
    assert "edge" in plan.visited_relation_ids
    assert (observation.status == "satisfied") is should_connect
    assert plan.complete_claimed is should_connect
    if not should_connect:
        assert observation.relation_ids == ()
        assert any(
            atom.span.chunk_id == "unrelated"
            and atom.label.startswith("routing-hypothesis:")
            for atom in plan.atoms
        )
        assert not any(
            bundle.obligation_ids and "edge" in bundle.relation_ids
            for bundle in plan.bundles
        )


def test_weak_revision_widens_but_cannot_supersede_direct_current_evidence():
    graph = _MemoryGraph()
    old = graph.add_chunk("old", "The selected mode was bounded.", 2)
    unrelated = graph.add_chunk("new", "A different topic selected another mode.", 9)
    graph.add_unit("old-decision", "decision", "selected bounded mode", old)
    graph.add_unit("new-decision", "decision", "different topic", unrelated)
    graph.add_relation(
        "weak-revision",
        "revises",
        (("old-decision", "old"), ("new-decision", "new")),
        old,
        confidence=0.1,
    )
    manual = QueryProgram(
        query="What mode was selected?",
        intent="lookup",
        subject_terms=("selected mode",),
        obligations=(
            EvidenceObligation(
                obligation_id="decision",
                kind="decision",
                required=True,
                weight=1.0,
                unit_kinds=("decision",),
                relation_types=("revises",),
                subject_terms=("selected mode",),
                temporal_stance="terminal",
            ),
        ),
    )
    plan = close_evidence(
        graph,
        query_program=manual,
        direct_chunk_ids=("old",),
        policy=ClosurePolicy(min_relation_confidence=0.8),
    )
    result = plan.obligation_results[0]
    assert "new-decision" in plan.visited_unit_ids
    assert result.unit_ids == ("old-decision",)
    assert result.relation_ids == ()
    assert result.status == "satisfied"
    assert plan.complete_claimed is True


def test_raw_hit_without_episode_or_annotation_fails_open_as_exact_evidence():
    graph = _MemoryGraph()
    graph.add_chunk("raw", "The source sentence remains directly retrievable.", 4)
    plan = close_evidence(graph, "What was recorded?", direct_chunk_ids=("raw",))
    assert [atom.text for atom in plan.atoms] == [
        "The source sentence remains directly retrievable."
    ]
    assert len(plan.bundles) == 1
    assert plan.obligation_results[0].status == "not_found"
    assert plan.stopping_reason == "not_found"
    assert plan.complete_claimed is False


def test_direct_raw_hit_is_preserved_without_waiving_unit_subject_matching():
    graph = _MemoryGraph()
    span = graph.add_chunk(
        "distractor",
        "A measurement for a different subject was recorded.",
        4,
    )
    graph.add_unit(
        "distractor-unit",
        "measurement",
        "different subject",
        span,
    )
    plan = close_evidence(
        graph,
        "What is the current status of the release plan?",
        direct_chunk_ids=("distractor",),
    )
    observations = next(
        result
        for result in plan.obligation_results
        if result.obligation_id == "observations"
    )
    assert {atom.span.chunk_id for atom in plan.atoms} == {"distractor"}
    assert "distractor-unit" in plan.visited_unit_ids
    assert observations.status == "not_found"
    assert plan.complete_claimed is False


def test_as_of_program_excludes_later_raw_and_graph_evidence():
    graph = _MemoryGraph()
    old = graph.add_chunk("old", "The recorded value was five.", 3)
    future = graph.add_chunk("future", "The recorded value later became nine.", 9)
    graph.add_unit("old-unit", "value", "recorded value", old)
    graph.add_unit("future-unit", "value", "recorded value", future)
    manual = QueryProgram(
        query="What was the recorded value then?",
        intent="lookup",
        subject_terms=("recorded value",),
        obligations=(
            EvidenceObligation(
                obligation_id="value",
                kind="value",
                required=True,
                weight=1.0,
                unit_kinds=("value",),
                temporal_stance="latest",
            ),
        ),
        as_of_ordinal=5,
    )
    plan = close_evidence(
        graph,
        query_program=manual,
        direct_chunk_ids=("future", "old"),
    )
    assert {atom.span.chunk_id for atom in plan.atoms} == {"old"}
    assert plan.obligation_results[0].unit_ids == ("old-unit",)
    assert plan.complete_claimed is True


def test_relation_does_not_inflate_fixed_cardinality_member_count():
    graph = _MemoryGraph()
    first = graph.add_chunk("first", "One recorded entry.", 1)
    second = graph.add_chunk("second", "A second recorded entry.", 2)
    graph.add_unit("first-unit", "item", "recorded entry one", first)
    graph.add_unit("second-unit", "item", "recorded entry two", second)
    graph.add_relation(
        "sequence",
        "sequence",
        (("first-unit", "first"), ("second-unit", "second")),
        second,
    )
    plan = close_evidence(
        graph,
        "List three recorded entries.",
        direct_chunk_ids=("first", "second"),
    )
    result = plan.obligation_results[0]
    assert result.status == "not_found"
    assert len(result.unit_ids) == 2
    assert "below the required minimum of 3" in (result.reason or "")
    assert plan.complete_claimed is False


def test_unresolved_contradiction_reports_conflicted_instead_of_complete():
    graph, seed = _connected_graph()
    del graph.relations["r-09"]
    plan = close_evidence(
        graph,
        "How should we improve request throughput?",
        seeds=(seed,),
        policy=ClosurePolicy(max_hops=6),
    )
    result = next(
        row
        for row in plan.obligation_results
        if row.obligation_id == "revisions_conflicts"
    )
    assert result.status == "conflicted"
    assert plan.stopping_reason == "conflicted"
    assert plan.complete_claimed is False


def test_medical_history_uses_the_same_generic_diagnosis_closure():
    graph = _MemoryGraph()
    observation = graph.add_chunk("m-observation", "The symptoms returned after the intervention.", 1, "patient-a")
    failure = graph.add_chunk("m-failure", "The intervention did not sustain its effect.", 20, "patient-a")
    cause = graph.add_chunk("m-cause", "The likely cause was incomplete absorption.", 40, "patient-a")
    dependency = graph.add_chunk("m-dependency", "Absorption depends on consistent timing.", 60, "patient-a")
    test = graph.add_chunk("m-test", "A level check was ordered.", 80, "patient-a")
    result = graph.add_chunk("m-result", "The check found a low level.", 100, "patient-a")
    graph.add_episode("m-episode", 0, (observation,), source_id="patient-a")
    graph.add_unit("m-u-observation", "observation", "symptoms after intervention", observation)
    graph.add_unit("m-u-failure", "failure", "intervention effect", failure)
    graph.add_unit("m-u-cause", "cause", "incomplete absorption", cause)
    graph.add_unit("m-u-dependency", "dependency", "consistent timing", dependency)
    graph.add_unit("m-u-test", "test", "level check", test)
    graph.add_unit("m-u-result", "result", "low level", result)
    graph.add_relation("m-r1", "supports", (("m-u-observation", "evidence"), ("m-u-failure", "claim")), observation)
    graph.add_relation("m-r2", "causes", (("m-u-cause", "cause"), ("m-u-failure", "effect")), cause)
    graph.add_relation("m-r3", "depends_on", (("m-u-cause", "dependent"), ("m-u-dependency", "requirement")), dependency)
    graph.add_relation("m-r4", "supports", (("m-u-dependency", "context"), ("m-u-test", "test")), test)
    graph.add_relation("m-r5", "produces", (("m-u-test", "test"), ("m-u-result", "result")), result)
    plan = close_evidence(
        graph,
        "Diagnose why the symptoms returned after the intervention.",
        seeds=(EpisodeSeed("m-episode", "m-observation", 1.0, "direct"),),
        policy=ClosurePolicy(max_hops=5),
    )
    assert plan.query_program.intent == "diagnose"
    assert plan.complete_claimed is True
    assert {row.status for row in plan.obligation_results if row.obligation_id != "unresolved_issues"} == {"satisfied"}


def test_project_status_can_complete_while_optional_issues_are_absent():
    graph = _MemoryGraph()
    state = graph.add_chunk("p-state", "The release is currently in final review.", 2, "project-a")
    observation = graph.add_chunk("p-observation", "Seven of eight checks have passed.", 8, "project-a")
    graph.add_episode("p-episode", 0, (state,), source_id="project-a")
    graph.add_unit("p-u-state", "status", "release current state", state)
    graph.add_unit("p-u-observation", "measurement", "release checks", observation)
    graph.add_relation("p-r1", "supports", (("p-u-observation", "evidence"), ("p-u-state", "claim")), observation)
    plan = close_evidence(
        graph,
        "What is the current status of the release plan?",
        seeds=(EpisodeSeed("p-episode", "p-state", 1.0, "direct"),),
    )
    assert plan.query_program.intent == "status"
    assert plan.complete_claimed is True
    statuses = {row.obligation_id: row.status for row in plan.obligation_results}
    assert statuses["current_state"] == "satisfied"
    assert statuses["observations"] == "satisfied"
    assert statuses["unresolved_issues"] == "not_found"


def test_source_local_temporal_contract_is_checked_not_trusted_blindly():
    class _BrokenAdjacent(_MemoryGraph):
        def adjacent_episodes(self, episode_id: str, *, radius: int = 1, include_self: bool = False) -> tuple[Episode, ...]:
            del episode_id, radius, include_self
            return (self.episodes["other"],)

    graph = _BrokenAdjacent()
    anchor = graph.add_chunk("anchor", "A current state was recorded.", 1, "source-a")
    other = graph.add_chunk("other", "A different history exists.", 2, "source-b")
    graph.add_episode("anchor", 0, (anchor,), source_id="source-a")
    graph.add_episode("other", 1, (other,), source_id="source-b")
    graph.add_unit("anchor-unit", "state", "current record", anchor)
    with pytest.raises(ValueError, match="source or artifact boundary"):
        close_evidence(graph, "What is the current record?", direct_chunk_ids=("anchor",))


def test_snapshot_change_during_closure_fails_closed():
    graph = _MemoryGraph(mutate_snapshot=True)
    graph.add_chunk("raw", "One exact fact.", 1)
    with pytest.raises(RuntimeError, match="snapshot changed"):
        close_evidence(graph, "What exact fact?", direct_chunk_ids=("raw",))


def test_multi_artifact_graph_requires_explicit_scope_and_rejects_frankenstein_reads():
    class _IgnoresArtifact(_MemoryGraph):
        def units_for_chunks(
            self,
            chunk_ids: Sequence[str],
            *,
            artifact_id: str | None = None,
            limit: int | None = None,
        ) -> tuple[DiscourseUnit, ...]:
            del artifact_id
            return super().units_for_chunks(chunk_ids, limit=limit)

    graph = _IgnoresArtifact(artifact_ids=("artifact-a", "artifact-b"))
    span = graph.add_chunk("shared", "The release state was recorded.", 1)
    graph.add_unit("state-a", "state", "release state", span, artifact_id="artifact-a")
    graph.add_unit("state-b", "state", "release state", span, artifact_id="artifact-b")

    with pytest.raises(ValueError, match="explicit artifact_id"):
        close_evidence(graph, "What is the release state?", direct_chunk_ids=("shared",))
    with pytest.raises(ValueError, match="crossed artifact scope"):
        close_evidence(
            graph,
            "What is the release state?",
            direct_chunk_ids=("shared",),
            artifact_id="artifact-a",
        )


def test_episode_seed_must_contain_its_claimed_anchor_chunk():
    graph = _MemoryGraph()
    episode_span = graph.add_chunk("episode", "One episode fact.", 1)
    graph.add_chunk("claimed", "A different raw fact.", 2)
    graph.add_episode("episode-id", 0, (episode_span,))
    with pytest.raises(ValueError, match="does not contain anchor chunk"):
        close_evidence(
            graph,
            "What fact was recorded?",
            seeds=(EpisodeSeed("episode-id", "claimed", 1.0, "manual"),),
        )


def test_degree_probe_exposes_a_hidden_revision_successor_and_blocks_completion():
    graph = _MemoryGraph()
    old = graph.add_chunk("old", "The selected mode was bounded.", 1)
    first = graph.add_chunk("first", "One supporting note.", 2)
    second = graph.add_chunk("second", "Another supporting note.", 3)
    replacement = graph.add_chunk("replacement", "The mode was replaced by streaming.", 4)
    graph.add_unit("old", "decision", "selected mode", old)
    graph.add_unit("first", "fact", "supporting note", first)
    graph.add_unit("second", "fact", "supporting note", second)
    graph.add_unit("replacement", "decision", "selected mode", replacement)
    graph.add_relation("support-1", "supports", (("old", "claim"), ("first", "evidence")), first)
    graph.add_relation("support-2", "supports", (("old", "claim"), ("second", "evidence")), second)
    graph.add_relation(
        "hidden-revision",
        "revises",
        (("old", "old"), ("replacement", "new")),
        replacement,
        confidence=0.9,
    )
    manual = QueryProgram(
        query="What mode was selected?",
        intent="lookup",
        subject_terms=("selected mode",),
        obligations=(
            EvidenceObligation(
                obligation_id="decision",
                kind="decision",
                required=True,
                weight=1.0,
                unit_kinds=("decision",),
                temporal_stance="terminal",
            ),
        ),
    )
    plan = close_evidence(
        graph,
        query_program=manual,
        direct_chunk_ids=("old",),
        policy=ClosurePolicy(max_degree=1),
    )
    assert plan.obligation_results[0].status == "satisfied"
    assert plan.complete_claimed is False
    assert plan.stopping_reason == "workspace_cap"
    degree = next(
        witness
        for witness in plan.scope_witnesses
        if witness.kind == "incident_relations" and witness.subject_id == "old"
    )
    assert degree.exhaustive is False
    assert degree.detail["probe_count"] == 2


def test_temporal_radius_is_auxiliary_when_global_artifact_scope_is_exhaustive():
    graph = _MemoryGraph()
    now = graph.add_chunk("now", "The project state is ready.", 1)
    later = graph.add_chunk("later", "A later project state exists.", 2)
    graph.add_episode("episode-now", 0, (now,))
    graph.add_episode("episode-later", 100, (later,))
    graph.add_unit("state", "state", "project state", now)
    plan = close_evidence(
        graph,
        "What is the project state?",
        direct_chunk_ids=("now",),
        policy=ClosurePolicy(max_episode_neighbors=0),
    )
    assert plan.obligation_results[0].status == "satisfied"
    assert plan.complete_claimed is True
    temporal = next(
        witness for witness in plan.scope_witnesses if witness.kind == "temporal_neighbors"
    )
    assert temporal.exhaustive is True
    assert temporal.detail["probe_radius"] == 1
    assert temporal.detail["outside_requested_radius"] is True


def test_global_artifact_scan_finds_a_disconnected_newer_latest_unit():
    graph = _MemoryGraph()
    old = graph.add_chunk("old", "The account state was pending.", 1)
    new = graph.add_chunk("new", "The account state is active.", 9)
    graph.add_unit("old", "state", "account state", old)
    graph.add_unit("new", "state", "account state", new)
    plan = close_evidence(
        graph,
        "What is the current account state?",
        direct_chunk_ids=("old",),
    )
    assert plan.artifact_id == "artifact-a"
    assert plan.obligation_results[0].unit_ids == ("new",)
    assert plan.complete_claimed is True
    scan = next(
        witness for witness in plan.scope_witnesses if witness.kind == "artifact_unit_scan"
    )
    assert scan.exhaustive is True


def test_seed_local_match_without_global_coverage_is_never_corpus_complete():
    class _NoFinalCoverage(_MemoryGraph):
        def artifact_coverage(
            self,
            artifact_id: str,
            coverage_kind: str = "discourse",
        ) -> ArtifactCoverageReceipt | None:
            del artifact_id, coverage_kind
            return None

    graph = _NoFinalCoverage()
    span = graph.add_chunk("state", "The account state is active.", 1)
    graph.add_unit("state", "state", "account state", span)
    plan = close_evidence(
        graph,
        "What is the current account state?",
        direct_chunk_ids=("state",),
    )
    assert plan.obligation_results[0].status == "satisfied"
    assert plan.complete_claimed is False
    coverage = next(
        witness for witness in plan.scope_witnesses if witness.kind == "artifact_coverage"
    )
    assert coverage.exhaustive is False


def test_artifact_wide_unit_probe_reports_truncation_instead_of_global_completion():
    graph = _MemoryGraph()
    for ordinal in (1, 2, 3):
        span = graph.add_chunk(str(ordinal), f"Account state {ordinal}.", ordinal)
        graph.add_unit(str(ordinal), "state", f"account state {ordinal}", span)
    plan = close_evidence(
        graph,
        "What is the current account state?",
        direct_chunk_ids=("1",),
        policy=ClosurePolicy(max_units=2),
    )
    scan = next(
        witness for witness in plan.scope_witnesses if witness.kind == "artifact_unit_scan"
    )
    assert scan.exhaustive is False
    assert scan.detail["probe_count"] == 3
    assert plan.complete_claimed is False


def test_artifact_stream_finds_old_match_beyond_newest_unit_prefix():
    graph = _MemoryGraph()
    relevant = graph.add_chunk(
        "relevant",
        "The account state is active.",
        1,
    )
    graph.add_unit("relevant", "state", "account state", relevant)
    for ordinal in range(2, 14):
        chunk_id = f"noise-{ordinal}"
        span = graph.add_chunk(
            chunk_id,
            f"Unrelated deployment note {ordinal}.",
            ordinal,
        )
        graph.add_unit(chunk_id, "claim", "deployment note", span)

    plan = close_evidence(
        graph,
        "What is the current account state?",
        direct_chunk_ids=("noise-13",),
        policy=ClosurePolicy(max_units=2),
    )

    assert plan.obligation_results[0].unit_ids == ("relevant",)
    assert plan.complete_claimed is True
    scan = next(
        witness
        for witness in plan.scope_witnesses
        if witness.kind == "artifact_unit_scan"
    )
    assert scan.exhaustive is True
    assert scan.detail["scan_mode"] == "exhaustive_stream"
    assert scan.detail["probe_count"] == 13
    assert scan.detail["matched_count"] == 1


def test_optional_episode_mapping_failure_preserves_verified_raw_evidence():
    class _BrokenMapping(_MemoryGraph):
        def episode_ids_for_chunks(
            self,
            chunk_ids: Sequence[str],
            *,
            artifact_id: str | None = None,
        ) -> dict[str, str]:
            del chunk_ids, artifact_id
            raise RuntimeError("corrupt optional annotation")

    graph = _BrokenMapping()
    graph.add_chunk("raw", "The exact raw sentence remains available.", 1)
    plan = close_evidence(graph, "What sentence was recorded?", direct_chunk_ids=("raw",))
    assert [atom.text for atom in plan.atoms] == ["The exact raw sentence remains available."]
    mapping = next(
        witness for witness in plan.scope_witnesses if witness.kind == "episode_mapping"
    )
    assert mapping.exhaustive is False
    assert mapping.detail["failures"] == ("RuntimeError",)


def test_missing_relation_member_annotation_does_not_abort_raw_fail_open():
    graph, seed = _connected_graph()
    original_get_unit = graph.get_unit

    def broken_get_unit(unit_id: str) -> DiscourseUnit | None:
        if unit_id == "u-objective":
            raise RuntimeError("corrupt unit annotation")
        return original_get_unit(unit_id)

    def broken_global_scan(
        artifact_id: str,
        *,
        limit: int | None = None,
    ) -> tuple[DiscourseUnit, ...]:
        del artifact_id, limit
        raise RuntimeError("global scan unavailable")

    graph.get_unit = broken_get_unit  # type: ignore[method-assign]
    graph.units_for_artifact = broken_global_scan  # type: ignore[method-assign]
    graph.iter_unit_routes_for_artifact = broken_global_scan  # type: ignore[method-assign]
    plan = close_evidence(
        graph,
        "How should we improve request throughput?",
        seeds=(seed,),
        policy=ClosurePolicy(max_hops=6),
    )
    assert any(atom.span.chunk_id == "c-state" for atom in plan.atoms)
    member_failure = next(
        witness
        for witness in plan.scope_witnesses
        if witness.kind == "relation_member_lookup"
    )
    assert member_failure.exhaustive is False
    assert plan.complete_claimed is False


def test_missing_annotation_coverage_blocks_a_global_completion_claim():
    class _MissingCoverage(_MemoryGraph):
        def coverage_for_chunks(
            self,
            artifact_id: str,
            chunk_ids: Sequence[str],
            *,
            coverage_kind: str = "discourse",
        ) -> dict[str, str]:
            if coverage_kind == "discourse":
                return {}
            return super().coverage_for_chunks(
                artifact_id,
                chunk_ids,
                coverage_kind=coverage_kind,
            )

    graph = _MissingCoverage()
    span = graph.add_chunk("state", "The release state is ready.", 1)
    graph.add_unit("state", "state", "release state", span)
    plan = close_evidence(
        graph,
        "What is the release state?",
        direct_chunk_ids=("state",),
    )
    assert plan.obligation_results[0].status == "satisfied"
    assert plan.complete_claimed is False
    coverage = next(
        witness
        for witness in plan.scope_witnesses
        if witness.kind == "annotation_coverage"
    )
    assert coverage.exhaustive is False
    assert coverage.detail["missing_chunk_ids"] == ("state",)


def test_terminal_selection_never_resurrects_a_predecessor_with_other_kind_successor():
    graph = _MemoryGraph()
    old = graph.add_chunk("old", "The feature decision selected option A.", 1)
    successor = graph.add_chunk("new", "The feature is now cancelled.", 2)
    graph.add_unit("old", "decision", "feature decision", old)
    graph.add_unit("new", "status", "feature decision", successor)
    graph.add_relation("revision", "revises", (("old", "old"), ("new", "new")), successor)
    manual = QueryProgram(
        query="What is the terminal feature decision?",
        intent="lookup",
        subject_terms=("feature decision",),
        obligations=(
            EvidenceObligation(
                obligation_id="decision",
                kind="decision",
                required=True,
                weight=1.0,
                unit_kinds=("decision",),
                temporal_stance="terminal",
            ),
        ),
    )
    plan = close_evidence(graph, query_program=manual, direct_chunk_ids=("old",))
    assert plan.obligation_results[0].unit_ids == ()
    assert plan.obligation_results[0].status == "not_found"
    assert plan.complete_claimed is False


def test_one_sided_unrelated_resolution_does_not_erase_a_conflict():
    graph = _MemoryGraph()
    left = graph.add_chunk("left", "The rate measurement was five.", 1)
    right = graph.add_chunk("right", "The rate measurement was nine.", 2)
    unrelated = graph.add_chunk("unrelated", "A separate issue was resolved.", 3)
    graph.add_unit("left", "observation", "rate measurement", left)
    graph.add_unit("right", "observation", "rate measurement", right)
    graph.add_unit("unrelated", "resolution", "separate issue", unrelated)
    graph.add_relation("conflict", "contradicts", (("left", "side_a"), ("right", "side_b")), right)
    graph.add_relation(
        "unrelated-resolution",
        "resolves",
        (("left", "context"), ("unrelated", "resolution")),
        unrelated,
    )
    manual = QueryProgram(
        query="What rate measurement was observed?",
        intent="lookup",
        subject_terms=("rate measurement",),
        obligations=(
            EvidenceObligation(
                obligation_id="observation",
                kind="observation",
                required=True,
                weight=1.0,
                unit_kinds=("observation",),
                relation_types=("contradicts", "resolves"),
            ),
        ),
    )
    plan = close_evidence(graph, query_program=manual, direct_chunk_ids=("left",))
    assert plan.obligation_results[0].status == "conflicted"
    assert plan.stopping_reason == "conflicted"
    assert not any(
        {"conflict", "unrelated-resolution"} <= set(bundle.relation_ids)
        for bundle in plan.bundles
    )


def test_typed_relation_without_a_subject_member_is_routing_only():
    graph = _MemoryGraph()
    raw = graph.add_chunk("noise", "An unrelated dependency was recorded.", 1)
    other = graph.add_chunk("other", "Another unrelated constraint.", 2)
    graph.add_unit("noise", "constraint", "garden schedule", raw)
    graph.add_unit("other", "dependency", "garden supplies", other)
    graph.add_relation("typed", "requires", (("noise", "dependent"), ("other", "requirement")), raw)
    manual = QueryProgram(
        query="What constrains the release?",
        intent="lookup",
        subject_terms=("release",),
        obligations=(
            EvidenceObligation(
                obligation_id="constraint",
                kind="constraint",
                required=True,
                weight=1.0,
                unit_kinds=("constraint",),
                relation_types=("requires",),
            ),
        ),
    )
    plan = close_evidence(graph, query_program=manual, direct_chunk_ids=("noise",))
    assert "typed" in plan.visited_relation_ids
    assert plan.obligation_results[0].status == "not_found"
    assert plan.obligation_results[0].relation_ids == ()
    assert plan.complete_claimed is False


def test_enumeration_honors_compiled_reverse_chronological_order():
    graph = _MemoryGraph()
    for ordinal in (1, 2, 3):
        span = graph.add_chunk(str(ordinal), f"Recorded entry {ordinal}.", ordinal)
        graph.add_unit(str(ordinal), "item", f"recorded entry {ordinal}", span)
    plan = close_evidence(
        graph,
        "List two recorded entries in reverse chronological order.",
        direct_chunk_ids=("1", "2", "3"),
    )
    assert plan.query_program.ordering == "descending"
    assert plan.obligation_results[0].unit_ids == ("3", "2")


def test_bundle_identity_includes_its_owning_unit_ids():
    common = {
        "atom_ids": ("atom",),
        "obligation_ids": ("obligation",),
        "relation_ids": ("relation",),
    }
    assert make_bundle_id(**common, unit_ids=("unit-a",)) != make_bundle_id(
        **common,
        unit_ids=("unit-b",),
    )


def test_direct_one_span_bundle_cannot_claim_a_two_sided_conflict():
    graph = _MemoryGraph()
    left = graph.add_chunk("left", "The observed rate was five.", 1)
    right = graph.add_chunk("right", "The observed rate was nine, then resolved.", 2)
    graph.add_unit("left", "observation", "observed rate", left)
    graph.add_unit("right", "observation", "observed rate", right)
    graph.add_relation(
        "conflict",
        "contradicts",
        (("left", "side_a"), ("right", "side_b")),
        left,
    )
    graph.add_relation(
        "resolution",
        "resolves",
        (("left", "side_a"), ("right", "side_b")),
        right,
    )
    manual = QueryProgram(
        query="Was the observed rate conflict resolved?",
        intent="lookup",
        subject_terms=("observed rate",),
        obligations=(
            EvidenceObligation(
                obligation_id="subject",
                kind="observation",
                required=False,
                weight=0.5,
                unit_kinds=("observation",),
            ),
            EvidenceObligation(
                obligation_id="conflict",
                kind="conflict",
                required=True,
                weight=1.0,
                relation_types=("contradicts",),
            ),
        ),
    )
    plan = close_evidence(
        graph,
        query_program=manual,
        direct_chunk_ids=("left",),
    )
    assert plan.complete_claimed is True, (
        plan.stopping_reason,
        plan.obligation_results,
        plan.scope_witnesses,
    )
    atom_by_id = {atom.atom_id: atom for atom in plan.atoms}
    one_span = next(
        bundle
        for bundle in plan.bundles
        if {atom_by_id[item].span.chunk_id for item in bundle.atom_ids}
        == {"left"}
    )
    assert one_span.relation_ids == ()
    assert "conflict" not in one_span.obligation_ids
    atomic = next(
        bundle for bundle in plan.bundles if "conflict" in bundle.relation_ids
    )
    assert {
        atom_by_id[item].span.chunk_id for item in atomic.atom_ids
    } == {"left", "right"}

    raw_atoms = tuple(atom_by_id[item] for item in one_span.atom_ids)
    raw_budget = count_tokens(render_evidence_context(raw_atoms, (one_span,)))
    packet = pack_evidence_plan(plan, max_context_tokens=raw_budget)
    assert packet.receipt.complete_claimed is False
    assert packet.receipt.stopping_reason == "budget_impossible"


def test_bundle_cap_receipts_exactly_which_direct_raw_hit_was_omitted():
    graph = _MemoryGraph()
    graph.add_chunk("raw-a", "First direct raw fact.", 1)
    graph.add_chunk("raw-b", "Second direct raw fact.", 2)
    plan = close_evidence(
        graph,
        "What fact was recorded?",
        direct_chunk_ids=("raw-a", "raw-b"),
        policy=ClosurePolicy(max_bundles=1),
    )
    retained = {atom.span.chunk_id for atom in plan.atoms}
    omitted = {"raw-a", "raw-b"} - retained
    budget = next(
        witness for witness in plan.scope_witnesses if witness.kind == "bundle_budget"
    )
    assert budget.exhaustive is False
    assert set(budget.detail["omitted_direct_chunk_ids"]) == omitted
    assert len(omitted) == 1


def test_span_deduplication_uses_full_role_and_time_provenance_identity():
    graph = _MemoryGraph()
    base = graph.add_chunk("shared", "A shared evidenced claim.", 1)
    peer = graph.add_chunk("peer", "A supporting peer.", 2)
    alternate = EvidenceSpan(
        chunk_id=base.chunk_id,
        start_char=base.start_char,
        end_char=base.end_char,
        quote_sha256=base.quote_sha256,
        ordinal=base.ordinal,
        source_id=base.source_id,
        turn_start_char=base.turn_start_char,
        turn_id=base.turn_id,
        role="assistant",
        created_at=base.created_at,
    )
    graph.add_unit("claim", "state", "shared claim", base)
    graph.add_unit("peer", "observation", "shared claim", peer)
    graph.add_relation(
        "support",
        "supports",
        (("claim", "claim"), ("peer", "evidence")),
        alternate,
    )
    plan = close_evidence(
        graph,
        "What is the shared claim?",
        direct_chunk_ids=("shared",),
    )
    shared_atoms = [atom for atom in plan.atoms if atom.span.chunk_id == "shared"]
    assert {atom.span.role for atom in shared_atoms} == {None, "assistant"}
    assert len({atom.atom_id for atom in shared_atoms}) == 2


def test_episode_expansion_receipt_requires_an_exhaustiveness_attestation():
    graph = _MemoryGraph()
    span = graph.add_chunk("state", "The release state is ready.", 1)
    graph.add_unit("state", "state", "release state", span)
    plan = close_evidence(
        graph,
        "What is the release state?",
        direct_chunk_ids=("state",),
        expansion_receipt_sha256="b" * 64,
        expansion_exhaustive=False,
    )
    expansion = next(
        witness for witness in plan.scope_witnesses if witness.kind == "episode_expansion"
    )
    assert expansion.subject_id == "b" * 64
    assert expansion.exhaustive is False
    assert plan.complete_claimed is False


def test_public_closure_caps_caller_routes_before_store_reads():
    class _TrackingGraph(_MemoryGraph):
        evidence_requests: list[tuple[str, ...]]

        def evidence_for_chunks(
            self,
            chunk_ids: Sequence[str],
        ) -> tuple[EvidenceSpan, ...]:
            self.evidence_requests.append(tuple(chunk_ids))
            return super().evidence_for_chunks(chunk_ids)

    graph = _TrackingGraph()
    graph.evidence_requests = []
    seeds = []
    for index, score in ((1, 3.0), (2, 2.0), (3, 1.0)):
        span = graph.add_chunk(
            f"seed-{index}",
            f"Episode seed {index}.",
            index,
        )
        episode_id = f"episode-{index}"
        graph.add_episode(episode_id, index, (span,))
        seeds.append(
            EpisodeSeed(episode_id, span.chunk_id, score, "caller")
        )
    for index in (1, 2, 3):
        graph.add_chunk(
            f"direct-{index}",
            f"Direct input {index}.",
            10 + index,
        )

    plan = close_evidence(
        graph,
        "What evidence was recorded?",
        seeds=tuple(seeds),
        direct_chunk_ids=("direct-3", "direct-1", "direct-2"),
        policy=ClosurePolicy(max_frontier=2),
    )
    assert tuple(seed.episode_id for seed in plan.seeds) == (
        "episode-1",
        "episode-2",
    )
    assert plan.direct_chunk_ids == ("direct-1", "direct-2")
    assert graph.evidence_requests == [
        ("direct-1", "direct-2", "seed-1", "seed-2")
    ]
    witness_by_kind = {item.kind: item for item in plan.scope_witnesses}
    seed_witness = witness_by_kind["closure_seed_inputs"]
    direct_witness = witness_by_kind["closure_direct_inputs"]
    assert seed_witness.exhaustive is False
    assert seed_witness.detail["omitted_count"] == 1
    assert seed_witness.detail["omitted_seeds"][0]["episode_id"] == "episode-3"
    assert direct_witness.exhaustive is False
    assert direct_witness.detail["omitted_direct_chunk_ids"] == ("direct-3",)
    assert plan.complete_claimed is False

    with pytest.raises(ValueError, match="hard normalization ceiling"):
        close_evidence(
            graph,
            "What evidence was recorded?",
            direct_chunk_ids=tuple(f"overflow-{index}" for index in range(5)),
            policy=ClosurePolicy(max_frontier=2),
        )
    assert len(graph.evidence_requests) == 1


def test_zero_artifact_snapshot_still_allows_raw_only_fail_open():
    graph = _MemoryGraph(artifact_ids=())
    graph.add_chunk("raw", "Raw evidence survives without annotations.", 1)
    plan = close_evidence(graph, "What evidence survives?", direct_chunk_ids=("raw",))
    assert [atom.span.chunk_id for atom in plan.atoms] == ["raw"]
    assert plan.visited_unit_ids == ()
