from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import pytest

from memory_condense.associations.head_memory import (
    MemoryLinkHit,
    NestedMemoryInspection,
)
from memory_condense.domain.discourse import (
    ClosurePolicy,
    ClosureScopeWitness,
    DiscourseUnit,
    Episode,
    EpisodeRepresentative,
    EvidenceSpan,
    identity_sha256,
    make_episode_id,
    quote_sha256,
)
from memory_condense.domain.schemas import Chunk, RetrievalResult, Turn
from memory_condense.search.closure.bundles import assemble_bundles
from memory_condense.search.closure.compiler import compile_query_program
from memory_condense.search.closure.results import completion, obligation_results
from memory_condense.search.closure.semantics import unit_obligation_ids
from memory_condense.search.episodes import (
    EpisodeRepresentativeRetrievalPolicy,
    EpisodeSourceCandidate,
    EpisodeSourceCandidateScope,
    episode_source_candidates_from_results,
    retrieve_episode_representatives,
)


ARTIFACT = "representative-artifact"


def _episode(index: int, source_id: str) -> tuple[Episode, str]:
    text = f"private evidence {source_id} {index}"
    span = EvidenceSpan(
        chunk_id=f"chunk-{source_id}-{index}",
        start_char=0,
        end_char=len(text),
        quote_sha256=quote_sha256(text),
        ordinal=index,
        source_id=source_id,
    )
    episode_id = make_episode_id(
        artifact_id=ARTIFACT,
        source_id=source_id,
        sequence_no=index,
        evidence=(span,),
    )
    return (
        Episode(
            episode_id=episode_id,
            artifact_id=ARTIFACT,
            source_id=source_id,
            sequence_no=index,
            first_ordinal=index,
            last_ordinal=index,
            evidence=(span,),
            boundary_method="fixture",
        ),
        text,
    )


class _Store:
    def __init__(self, rows: Sequence[tuple[Episode, str]]) -> None:
        self.episodes = {episode.episode_id: episode for episode, _ in rows}
        self.texts = {
            episode.evidence[0].chunk_id: text for episode, text in rows
        }
        self.representatives = {
            episode.episode_id: (
                EpisodeRepresentative(
                    episode_id=episode.episode_id,
                    chunk_id=episode.evidence[0].chunk_id,
                    rank=0,
                    vector_identity_sha256=(f"{index + 1:x}" * 64)[:64],
                ),
            )
            for index, (episode, _) in enumerate(rows)
        }

    def episodes_for_source(
        self,
        artifact_id: str,
        source_id: str,
        *,
        start_sequence: int | None = None,
        end_sequence: int | None = None,
        limit: int | None = None,
    ) -> tuple[Episode, ...]:
        rows = sorted(
            (
                episode
                for episode in self.episodes.values()
                if episode.artifact_id == artifact_id
                and episode.source_id == source_id
                and (
                    start_sequence is None
                    or episode.sequence_no >= start_sequence
                )
                and (
                    end_sequence is None
                    or episode.sequence_no <= end_sequence
                )
            ),
            key=lambda item: (item.sequence_no, item.episode_id),
        )
        return tuple(rows if limit is None else rows[:limit])

    def get_representatives(
        self,
        episode_id: str,
    ) -> tuple[EpisodeRepresentative, ...]:
        return self.representatives.get(episode_id, ())

    def hydrate(self, chunk_id: str, *, score: float, route: str) -> RetrievalResult | None:
        text = self.texts.get(chunk_id)
        if text is None:
            return None
        source_id = chunk_id.removeprefix("chunk-").rsplit("-", 1)[0]
        turn = Turn(
            turn_id=f"turn-{chunk_id}",
            role="user",
            text=text,
            source_id=source_id,
        )
        return RetrievalResult(
            chunk=Chunk(
                chunk_id=chunk_id,
                turn_id=turn.turn_id,
                text=text,
                start_char=0,
                end_char=len(text),
                token_count=4,
            ),
            turn=turn,
            score=score,
            route=route,
        )


class _Linker:
    max_candidates = 8

    def __init__(self, selected_episode_id: str | None) -> None:
        self.selected_episode_id = selected_episode_id
        self.calls: list[tuple[str, list[list[object]]]] = []

    def inspect_nested(
        self,
        query,
        groups,
        *,
        beam_per_group,
        top_k,
        score_mode,
    ) -> NestedMemoryInspection:
        copied = [list(group) for group in groups]
        self.calls.append((query, copied))
        assert beam_per_group == 2
        assert score_mode == "qk_ov"
        hits = ()
        if self.selected_episode_id is not None:
            hits = (
                MemoryLinkHit(
                    episode_id=self.selected_episode_id,
                    qk_score=0.5,
                    ov_transport=1.0,
                    head_weights=(0.25,),
                    # A live tensor-like value must not escape in the plan.
                    transport_signature=object(),
                ),
            )
        return NestedMemoryInspection(
            hits=hits[:top_k],
            passes=3,
            max_workspace_candidates=max(len(group) for group in copied),
            max_workspace_tokens=123,
            total_candidate_inspections=sum(len(group) for group in copied) + 1,
        )


def test_query_discovers_episode_absent_from_direct_chunk_hits() -> None:
    first = _episode(0, "source-a")
    target = _episode(0, "source-b")
    store = _Store((first, target))
    linker = _Linker(target[0].episode_id)
    policy = EpisodeRepresentativeRetrievalPolicy(
        artifact_id=ARTIFACT,
        group_size=2,
        top_k=1,
        query_tokens=4,
    )

    query = "Which source contains the private evidence about source B?"
    candidates = (
        EpisodeSourceCandidate("source-b", 0.8, "bounded_source_scan"),
        EpisodeSourceCandidate("source-a", 0.9, "bounded_source_scan"),
    )
    scope = EpisodeSourceCandidateScope(
        artifact_id=ARTIFACT,
        snapshot_sha256="a" * 64,
        source_revision=2,
        source_content_sha256="b" * 64,
        query_sha256=identity_sha256({"query": query}),
        router_policy_sha256="c" * 64,
        universe_source_ids=("source-a", "source-b"),
        candidates=candidates,
        truncated_source_ids=(),
        universe_enumerated=True,
    )
    plan = retrieve_episode_representatives(
        query,
        candidates,
        store,
        store.hydrate,
        linker,
        policy=policy,
        source_scope=scope,
    )

    assert [item.episode_id for item in plan.seeds] == [target[0].episode_id]
    assert plan.seeds[0].anchor_chunk_id == target[0].evidence[0].chunk_id
    assert plan.seeds[0].route == "episode_representative_qwen"
    assert plan.returned_plan_transformer_state_bytes == 0
    assert plan.runtime_binding_certified is False
    assert plan.passes == 3
    assert plan.max_workspace_tokens == 123
    assert plan.candidate_scope_exhaustive is True
    assert plan.source_scope_receipt_sha256 == scope.receipt_sha256
    assert len(linker.calls) == 1
    _, groups = linker.calls[0]
    assert all(
        len({candidate.metadata["source_id"] for candidate in group}) == 1
        for group in groups
    )
    serialized = str(plan.identity_payload())
    assert "private evidence" not in serialized
    with pytest.raises(ValueError, match="does not match"):
        replace(plan, runtime_binding_certified=True)


def test_full_query_identity_does_not_collapse_at_the_model_token_cap() -> None:
    row = _episode(0, "source-a")
    store = _Store((row,))
    policy = EpisodeRepresentativeRetrievalPolicy(
        artifact_id=ARTIFACT,
        query_tokens=2,
        top_k=1,
    )

    first = retrieve_episode_representatives(
        "find evidence alpha tail",
        (EpisodeSourceCandidate("source-a", 1.0),),
        store,
        store.hydrate,
        _Linker(row[0].episode_id),
        policy=policy,
    )
    second = retrieve_episode_representatives(
        "find evidence beta tail",
        (EpisodeSourceCandidate("source-a", 1.0),),
        store,
        store.hydrate,
        _Linker(row[0].episode_id),
        policy=policy,
    )

    assert first.query_input_sha256 == second.query_input_sha256
    assert first.query_sha256 != second.query_sha256


def test_source_candidate_reducer_rejects_conflicting_provenance() -> None:
    row = _episode(0, "source-a")
    store = _Store((row,))
    result = store.hydrate(
        row[0].evidence[0].chunk_id,
        score=1.0,
        route="hybrid",
    )
    assert result is not None
    conflicting = result.model_copy(update={"memory_source_id": "source-b"})

    with pytest.raises(ValueError, match="source identities disagree"):
        episode_source_candidates_from_results((conflicting,), max_sources=1)


def test_empty_unattested_source_input_is_never_globally_exhaustive() -> None:
    policy = EpisodeRepresentativeRetrievalPolicy(artifact_id=ARTIFACT)
    plan = retrieve_episode_representatives(
        "Where is the evidence?",
        (),
        _Store(()),
        _Store(()).hydrate,
        _Linker(None),
        policy=policy,
    )

    assert plan.source_scope_receipt_sha256 is None
    assert plan.source_universe_exhaustive is False
    assert plan.candidate_scope_exhaustive is False


def test_source_and_episode_caps_are_explicit_and_order_deterministic() -> None:
    source_a = (_episode(0, "source-a"), _episode(1, "source-a"))
    source_b = (_episode(0, "source-b"),)
    source_c = (_episode(0, "source-c"),)
    rows = (*source_a, *source_b, *source_c)
    store = _Store(rows)
    sources = (
        EpisodeSourceCandidate("source-c", 0.1),
        EpisodeSourceCandidate("source-a", 0.9),
        EpisodeSourceCandidate("source-b", 0.8),
    )
    policy = EpisodeRepresentativeRetrievalPolicy(
        artifact_id=ARTIFACT,
        max_source_groups=2,
        max_episodes_per_source=1,
        max_total_episodes=2,
        top_k=1,
        group_size=2,
    )

    def run(ordered_sources):
        linker = _Linker(source_b[0][0].episode_id)
        return retrieve_episode_representatives(
            "find source B",
            ordered_sources,
            store,
            store.hydrate,
            linker,
            policy=policy,
        )

    first = run(sources)
    replay = run(tuple(reversed(sources)))

    assert first.receipt_sha256 == replay.receipt_sha256
    assert first.truncated_source_ids == ("source-c",)
    assert first.truncated_episode_ids == (source_a[1][0].episode_id,)
    assert first.candidate_scope_exhaustive is False
    assert first.source_scans[0].observed_count == 2
    assert first.source_scans[0].exhaustive is False


def test_source_candidates_reduce_ranked_rows_without_gold() -> None:
    first = _episode(0, "source-a")
    second = _episode(0, "source-b")
    store = _Store((first, second))
    a_low = store.hydrate(
        first[0].evidence[0].chunk_id,
        score=0.2,
        route="source_dense",
    )
    a_high = store.hydrate(
        first[0].evidence[0].chunk_id,
        score=0.9,
        route="source_hybrid",
    )
    b = store.hydrate(
        second[0].evidence[0].chunk_id,
        score=0.7,
        route="source_dense",
    )

    candidates = episode_source_candidates_from_results(
        (a_low, b, a_high),  # type: ignore[arg-type]
        max_sources=2,
    )

    assert candidates == (
        EpisodeSourceCandidate("source-a", 0.9, "source_hybrid"),
        EpisodeSourceCandidate("source-b", 0.7, "source_dense"),
    )


def test_unverifiable_representative_fails_open_without_invoking_linker() -> None:
    row = _episode(0, "source-a")
    store = _Store((row,))
    store.texts[row[0].evidence[0].chunk_id] = "tampered evidence"
    linker = _Linker(row[0].episode_id)

    plan = retrieve_episode_representatives(
        "find evidence",
        (EpisodeSourceCandidate("source-a", 1.0),),
        store,
        store.hydrate,
        linker,
        policy=EpisodeRepresentativeRetrievalPolicy(artifact_id=ARTIFACT),
    )

    assert plan.seeds == ()
    assert plan.unavailable_episode_ids == (row[0].episode_id,)
    assert plan.candidate_scope_exhaustive is False
    assert linker.calls == []


def test_linker_cannot_fabricate_an_episode_identity() -> None:
    row = _episode(0, "source-a")
    store = _Store((row,))
    linker = _Linker("not-a-candidate")

    with pytest.raises(ValueError, match="outside the candidate set"):
        retrieve_episode_representatives(
            "find evidence",
            (EpisodeSourceCandidate("source-a", 1.0),),
            store,
            store.hydrate,
            linker,
            policy=EpisodeRepresentativeRetrievalPolicy(artifact_id=ARTIFACT),
        )


class _SpanHydrator:
    def __init__(self, text: str) -> None:
        self.text = text

    def hydrate_span(self, span: EvidenceSpan) -> str:
        return self.text[span.start_char : span.end_char]


def test_lookup_routes_claim_and_question_context_without_claiming_answer() -> None:
    text = "The launch badge was amber."
    span = EvidenceSpan(
        chunk_id="claim-chunk",
        start_char=0,
        end_char=len(text),
        quote_sha256=quote_sha256(text),
        ordinal=1,
        source_id="source-a",
    )
    claim = DiscourseUnit(
        unit_id="claim-unit",
        artifact_id=ARTIFACT,
        kind="claim",
        canonical_key="launch badge amber",
        asserted_ordinal=1,
        confidence=0.8,
        evidence=(span,),
    )
    question = DiscourseUnit(
        unit_id="question-unit",
        artifact_id=ARTIFACT,
        kind="question",
        canonical_key="launch badge",
        asserted_ordinal=1,
        confidence=0.8,
        evidence=(span,),
    )
    program = compile_query_program("What color was the launch badge?")
    required_ids = {
        item.obligation_id for item in program.obligations if item.required
    }

    assert required_ids == {"answer_fact"}
    assert unit_obligation_ids(claim, program, connected=False) == (
        "lookup_context",
    )
    assert unit_obligation_ids(question, program, connected=False) == (
        "lookup_context",
    )

    unit_matches = {claim.unit_id: ("lookup_context",)}
    assembly = assemble_bundles(
        _SpanHydrator(text),
        program=program,
        policy=ClosurePolicy(),
        raw_spans=(),
        units={claim.unit_id: claim},
        relations={},
        unit_obligations=unit_matches,
        credited_relation_ids=set(),
    )
    results = obligation_results(
        program,
        units={claim.unit_id: claim},
        relations={},
        unit_obligations=unit_matches,
        assembly=assembly,
        min_relation_confidence=0.5,
        credited_relation_ids=set(),
    )
    statuses = {item.obligation_id: item.status for item in results}
    stopping_reason, complete_claimed = completion(
        results,
        program,
        scope_witnesses=(
            ClosureScopeWitness(
                kind="test_scope",
                subject_id="fixture",
                requested_limit=1,
                returned_count=1,
                exhaustive=True,
            ),
        ),
    )

    assert assembly.bundles[0].obligation_ids == ("lookup_context",)
    assert assembly.bundles[0].required is False
    assert statuses == {"answer_fact": "not_found", "lookup_context": "satisfied"}
    assert stopping_reason == "not_found"
    assert complete_claimed is False
