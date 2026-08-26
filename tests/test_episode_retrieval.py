from __future__ import annotations

from dataclasses import fields
from typing import Sequence

import pytest

import memory_condense.search.episodes.builder as episode_builder_module

from memory_condense.domain.discourse import (
    Episode,
    EpisodeSeed,
    EvidenceSpan,
    make_episode_id,
    quote_sha256,
)
from memory_condense.domain.schemas import Chunk, RetrievalResult, Turn
from memory_condense.search.episodes import (
    AdaptiveBoundaryDetector,
    BoundaryProposal,
    CohesionBoundaryRefiner,
    EpisodeBuilder,
    EpisodeRetrievalPolicy,
    FixedIntervalBoundaryDetector,
    LexicalEmbeddingChangeScorer,
    SurpriseScorer,
    combine_episode_seeds,
    episode_seed_payload,
    expand_episode_seeds,
    score_surprise_sequence,
    select_episode_representatives,
)


ARTIFACT = "disc-test-artifact"


def test_episode_seed_helpers_preserve_exact_payload_and_deterministic_order() -> None:
    direct_winner = EpisodeSeed(
        episode_id="episode-b",
        anchor_chunk_id="chunk-b",
        score=0.8,
        route="episode_direct",
        path=("direct-b", "episode-b"),
    )
    representative_loser = EpisodeSeed(
        episode_id="episode-b",
        anchor_chunk_id="chunk-z",
        score=0.8,
        route="representative",
        path=("representative-b", "episode-b"),
    )
    representative_first = EpisodeSeed(
        episode_id="episode-a",
        anchor_chunk_id="chunk-a",
        score=0.9,
        route="representative",
        path=("representative-a", "episode-a"),
    )

    combined = combine_episode_seeds(
        (direct_winner,),
        (representative_loser, representative_first),
    )

    assert combined == (representative_first, direct_winner)
    assert [episode_seed_payload(seed) for seed in combined] == [
        {
            "episode_id": "episode-a",
            "anchor_chunk_id": "chunk-a",
            "score": 0.9,
            "route": "representative",
            "path": ["representative-a", "episode-a"],
        },
        {
            "episode_id": "episode-b",
            "anchor_chunk_id": "chunk-b",
            "score": 0.8,
            "route": "episode_direct",
            "path": ["direct-b", "episode-b"],
        },
    ]
    assert episode_seed_payload(direct_winner) == direct_winner.identity_payload()


def _span(index: int, *, source_id: str = "source-a", text: str | None = None) -> EvidenceSpan:
    content = text if text is not None else f"evidence {index}"
    return EvidenceSpan(
        chunk_id=f"chunk-{source_id}-{index}",
        start_char=0,
        end_char=len(content),
        quote_sha256=quote_sha256(content),
        ordinal=index,
        source_id=source_id,
    )


def _episode(
    sequence_no: int,
    *,
    source_id: str = "source-a",
    artifact_id: str = ARTIFACT,
    chunk_id: str | None = None,
) -> Episode:
    text = f"episode evidence {source_id} {sequence_no}"
    span = EvidenceSpan(
        chunk_id=chunk_id or f"chunk-{source_id}-{sequence_no}",
        start_char=0,
        end_char=len(text),
        quote_sha256=quote_sha256(text),
        ordinal=sequence_no,
        source_id=source_id,
    )
    episode_id = make_episode_id(
        artifact_id=artifact_id,
        source_id=source_id,
        sequence_no=sequence_no,
        evidence=(span,),
    )
    return Episode(
        episode_id=episode_id,
        artifact_id=artifact_id,
        source_id=source_id,
        sequence_no=sequence_no,
        first_ordinal=sequence_no,
        last_ordinal=sequence_no,
        evidence=(span,),
        boundary_method="fixture",
    )


def _result(
    chunk_id: str,
    score: float,
    *,
    source_id: str | None = "source-a",
    route: str = "hybrid",
) -> RetrievalResult:
    text = f"retrieved {chunk_id}"
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
            token_count=3,
            # Deliberately present in input: it must not escape into the plan.
            embedding=[123.0, 456.0],
        ),
        turn=turn,
        score=score,
        route=route,
    )


class _Store:
    def __init__(
        self,
        episodes: Sequence[Episode],
        mapping: dict[str, str],
        *,
        adjacency: dict[str, Sequence[Episode]] | None = None,
    ) -> None:
        self.episodes = {item.episode_id: item for item in episodes}
        self.mapping = dict(mapping)
        self.adjacency = {
            key: tuple(value) for key, value in (adjacency or {}).items()
        }

    def episode_ids_for_chunks(
        self,
        chunk_ids: Sequence[str],
        *,
        artifact_id: str | None = None,
    ) -> dict[str, str]:
        return {
            chunk_id: self.mapping[chunk_id]
            for chunk_id in chunk_ids
            if chunk_id in self.mapping
            and (
                artifact_id is None
                or self.episodes[self.mapping[chunk_id]].artifact_id == artifact_id
            )
        }

    def get_episode(self, episode_id: str) -> Episode | None:
        return self.episodes.get(episode_id)

    def adjacent_episodes(
        self,
        episode_id: str,
        *,
        radius: int = 1,
        include_self: bool = False,
    ) -> tuple[Episode, ...]:
        rows = list(self.adjacency.get(episode_id, ()))
        if include_self and episode_id in self.episodes:
            rows.append(self.episodes[episode_id])
        # A hostile fixture can return too many rows; production code owns caps.
        return tuple(rows)

    def episodes_for_source(
        self,
        artifact_id: str,
        source_id: str,
        *,
        limit: int | None = None,
    ) -> tuple[Episode, ...]:
        rows = sorted(
            (
                item
                for item in self.episodes.values()
                if item.artifact_id == artifact_id and item.source_id == source_id
            ),
            key=lambda item: (item.sequence_no, item.episode_id),
        )
        return tuple(rows if limit is None else rows[:limit])


def test_surprise_protocol_and_control_are_stateless_and_deterministic() -> None:
    scorer = LexicalEmbeddingChangeScorer()
    assert isinstance(scorer, SurpriseScorer)
    first = score_surprise_sequence(
        scorer,
        ("alpha beta", "alpha beta", "zeta"),
        embeddings=((1.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
    )
    second = score_surprise_sequence(
        scorer,
        ("alpha beta", "alpha beta", "zeta"),
        embeddings=((1.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
    )
    assert first == second
    assert first[0] == 0.0
    assert first[1] == 0.0
    assert first[2] > first[1]
    assert not hasattr(scorer, "__dict__")


def test_adaptive_detector_uses_only_the_bounded_trailing_window() -> None:
    detector = AdaptiveBoundaryDetector(window_size=2, gamma=0.0, min_history=2)
    # The outlier at zero is outside the baseline when position three is read.
    proposals = detector.detect((100.0, 0.0, 0.0, 1.0))
    assert proposals == (BoundaryProposal(position=3, score=1.0, threshold=0.0),)
    assert not hasattr(detector, "__dict__")


def test_graph_refinement_moves_to_cohesive_cut_and_respects_window() -> None:
    # Three mutually similar A nodes followed by three mutually similar B nodes.
    matrix = tuple(
        tuple(
            1.0 if (left < 3) == (right < 3) else 0.05
            for right in range(6)
        )
        for left in range(6)
    )
    proposal = BoundaryProposal(position=2, score=9.0, threshold=1.0)
    refined = CohesionBoundaryRefiner(window=2, max_nodes=6, max_degree=5).refine(
        (proposal,),
        item_count=6,
        similarities=matrix,
    )
    assert refined[0].position == 3
    assert refined[0].cohesion is not None

    bounded = CohesionBoundaryRefiner(window=0, max_nodes=6, max_degree=5).refine(
        (proposal,),
        item_count=6,
        similarities=matrix,
    )
    assert bounded[0].position == 2


def test_builder_is_source_local_size_bounded_and_uses_injected_scores() -> None:
    spans = tuple(_span(index) for index in range(10))
    builder = EpisodeBuilder(
        min_size=2,
        max_size=4,
        detector=AdaptiveBoundaryDetector(window_size=2, gamma=0.0, min_history=2),
    )
    outcome = builder.build(
        source_id="source-a",
        artifact_id=ARTIFACT,
        spans=spans,
        surprise_scores=(0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    assert [len(item.evidence) for item in outcome.episodes] == [2, 4, 4]
    assert outcome.forced_boundaries == (6,)
    assert outcome.episodes[1].initial_boundary == 2
    assert all(item.source_id == "source-a" for item in outcome.episodes)

    mismatched = list(spans)
    mismatched[5] = _span(5, source_id="source-b")
    with pytest.raises(ValueError, match="cross source"):
        builder.build(
            source_id="source-a",
            artifact_id=ARTIFACT,
            spans=mismatched,
            surprise_scores=(0.0,) * 10,
        )


def test_fixed_interval_control_needs_no_model_text_vectors_or_scores() -> None:
    builder = EpisodeBuilder(
        min_size=2,
        max_size=4,
        detector=FixedIntervalBoundaryDetector(interval=2),
    )
    outcome = builder.build(
        source_id="source-a",
        artifact_id=ARTIFACT,
        spans=tuple(_span(index) for index in range(6)),
    )
    assert [len(item.evidence) for item in outcome.episodes] == [2, 2, 2]
    assert [item.boundary_method for item in outcome.episodes] == [
        "stream_start",
        "fixed_interval",
        "fixed_interval",
    ]


def test_builder_without_refinement_never_constructs_pairwise_similarity(
    monkeypatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("pairwise similarity was requested without a refiner")

    monkeypatch.setattr(episode_builder_module, "_similarity_lookup", forbidden)
    outcome = EpisodeBuilder(
        min_size=2,
        max_size=4,
        detector=FixedIntervalBoundaryDetector(interval=2),
    ).build(
        source_id="source-a",
        artifact_id=ARTIFACT,
        spans=tuple(_span(index) for index in range(100)),
    )

    assert len(outcome.episodes) == 50


def test_cohesion_refinement_reads_only_a_bounded_local_similarity_window() -> None:
    calls = 0

    def similarity(left: int, right: int) -> float:
        nonlocal calls
        calls += 1
        return 1.0 if (left < 5_000) == (right < 5_000) else 0.0

    refined = CohesionBoundaryRefiner(
        window=4,
        max_nodes=8,
        max_degree=2,
    ).refine(
        (BoundaryProposal(position=5_000, score=2.0, threshold=1.0),),
        item_count=10_000,
        similarities=similarity,
    )

    assert refined[0].position == 5_000
    assert calls < 256


def test_builder_orders_same_turn_chunks_by_authoritative_turn_start() -> None:
    """Opaque chunk IDs must never replace actual within-turn source order."""
    first_text = "first source chunk"
    second_text = "second source chunk"
    first = EvidenceSpan(
        chunk_id="z-sorts-last",
        start_char=0,
        end_char=len(first_text),
        quote_sha256=quote_sha256(first_text),
        ordinal=7,
        source_id="source-a",
        turn_start_char=0,
    )
    second = EvidenceSpan(
        chunk_id="a-sorts-first",
        start_char=0,
        end_char=len(second_text),
        quote_sha256=quote_sha256(second_text),
        ordinal=7,
        source_id="source-a",
        turn_start_char=40,
    )

    outcome = EpisodeBuilder(
        min_size=1,
        max_size=2,
        detector=FixedIntervalBoundaryDetector(interval=2),
    ).build(
        source_id="source-a",
        artifact_id=ARTIFACT,
        spans=(first, second),
    )

    assert outcome.episodes[0].evidence == (first, second)
    with pytest.raises(ValueError, match="source order"):
        EpisodeBuilder(min_size=1, max_size=2).build(
            source_id="source-a",
            artifact_id=ARTIFACT,
            spans=(second, first),
            surprise_scores=(0.0, 0.0),
        )


def test_builder_validates_exact_text_hashes_and_is_reproducible() -> None:
    texts = tuple(f"exact {index}" for index in range(4))
    spans = tuple(_span(index, text=text) for index, text in enumerate(texts))
    builder = EpisodeBuilder(
        min_size=1,
        max_size=4,
        detector=AdaptiveBoundaryDetector(window_size=1, gamma=0.0, min_history=1),
    )
    first = builder.build(
        source_id="source-a",
        artifact_id=ARTIFACT,
        spans=spans,
        texts=texts,
    )
    second = builder.build(
        source_id="source-a",
        artifact_id=ARTIFACT,
        spans=spans,
        texts=texts,
    )
    assert first == second
    assert [item.receipt_sha256 for item in first.episodes] == [
        item.receipt_sha256 for item in second.episodes
    ]
    with pytest.raises(ValueError, match="evidence hash"):
        builder.build(
            source_id="source-a",
            artifact_id=ARTIFACT,
            spans=spans,
            texts=("tampered", *texts[1:]),
        )


def test_representatives_use_deterministic_source_order_tie_breaks() -> None:
    spans = tuple(_span(index) for index in range(3))
    episode = Episode(
        episode_id=make_episode_id(
            artifact_id=ARTIFACT,
            source_id="source-a",
            sequence_no=0,
            evidence=spans,
        ),
        artifact_id=ARTIFACT,
        source_id="source-a",
        sequence_no=0,
        first_ordinal=0,
        last_ordinal=2,
        evidence=spans,
        boundary_method="fixture",
    )
    embeddings = {span.chunk_id: (1.0, 0.0) for span in spans}
    representatives = select_episode_representatives(
        episode,
        limit=2,
        embeddings=embeddings,
    )
    assert [item.chunk_id for item in representatives] == [
        spans[0].chunk_id,
        spans[1].chunk_id,
    ]
    assert [item.rank for item in representatives] == [0, 1]
    assert all(len(item.vector_identity_sha256) == 64 for item in representatives)


def test_representatives_use_same_turn_offsets_before_reversed_chunk_ids() -> None:
    early_text = "early chunk"
    late_text = "late chunk"
    early = EvidenceSpan(
        chunk_id="z-reversed-id",
        start_char=0,
        end_char=len(early_text),
        quote_sha256=quote_sha256(early_text),
        ordinal=5,
        source_id="source-a",
        turn_start_char=0,
    )
    late = EvidenceSpan(
        chunk_id="a-reversed-id",
        start_char=0,
        end_char=len(late_text),
        quote_sha256=quote_sha256(late_text),
        ordinal=5,
        source_id="source-a",
        turn_start_char=100,
    )
    evidence = (early, late)
    episode = Episode(
        episode_id=make_episode_id(
            artifact_id=ARTIFACT,
            source_id="source-a",
            sequence_no=0,
            evidence=evidence,
        ),
        artifact_id=ARTIFACT,
        source_id="source-a",
        sequence_no=0,
        first_ordinal=5,
        last_ordinal=5,
        evidence=evidence,
        boundary_method="fixture",
    )
    representatives = select_episode_representatives(
        episode,
        limit=2,
        embeddings={early.chunk_id: (1.0,), late.chunk_id: (1.0,)},
    )

    assert [item.chunk_id for item in representatives] == [
        early.chunk_id,
        late.chunk_id,
    ]


def test_episode_retrieval_is_source_isolated_with_exact_paths() -> None:
    previous = _episode(0)
    anchor = _episode(1)
    following = _episode(2)
    hostile_cross_source = _episode(0, source_id="source-b")
    store = _Store(
        (previous, anchor, following, hostile_cross_source),
        {anchor.evidence[0].chunk_id: anchor.episode_id},
        adjacency={
            anchor.episode_id: (
                following,
                hostile_cross_source,
                previous,
            )
        },
    )
    plan = expand_episode_seeds(
        (_result(anchor.evidence[0].chunk_id, 1.0),),
        store,
        policy=EpisodeRetrievalPolicy(
            artifact_id=ARTIFACT,
            previous_episodes=1,
            next_episodes=1,
            max_episode_seeds=3,
        ),
    )
    assert [item.episode_id for item in plan.seeds] == [
        anchor.episode_id,
        previous.episode_id,
        following.episode_id,
    ]
    assert plan.seeds[0].path == (
        anchor.evidence[0].chunk_id,
        anchor.episode_id,
        "retrieval_route:hybrid",
    )
    assert plan.seeds[1].path == (
        anchor.evidence[0].chunk_id,
        anchor.episode_id,
        previous.episode_id,
        "retrieval_route:hybrid",
    )
    assert all(
        seed.anchor_chunk_id
        in {
            span.chunk_id
            for span in store.get_episode(seed.episode_id).evidence
        }
        for seed in plan.seeds
    )
    assert hostile_cross_source.episode_id not in {
        item.episode_id for item in plan.seeds
    }


def test_missing_or_invalid_annotations_fail_open_to_direct_chunk_ids() -> None:
    wrong_source = _episode(0, source_id="source-b", chunk_id="wrong-source")
    store = _Store(
        (wrong_source,),
        {"wrong-source": wrong_source.episode_id},
    )
    plan = expand_episode_seeds(
        (
            _result("unannotated", 0.8),
            _result("wrong-source", 0.9, source_id="source-a"),
        ),
        store,
        policy=EpisodeRetrievalPolicy(artifact_id=ARTIFACT),
    )
    assert plan.seeds == ()
    assert plan.direct_chunk_ids == ("wrong-source", "unannotated")
    assert [item.path for item in plan.direct_fallbacks] == [
        (
            "wrong-source",
            "retrieval_route:hybrid",
            "episode_failure:identity_mismatch",
        ),
        (
            "unannotated",
            "retrieval_route:hybrid",
            "episode_failure:not_annotated",
        ),
    ]
    assert [item.failure_code for item in plan.direct_fallbacks] == [
        "identity_mismatch",
        "not_annotated",
    ]


def test_all_direct_hits_survive_when_multiple_chunks_map_to_one_episode() -> None:
    spans = tuple(
        EvidenceSpan(
            chunk_id=chunk_id,
            start_char=0,
            end_char=len(chunk_id),
            quote_sha256=quote_sha256(chunk_id),
            ordinal=index,
            source_id="source-a",
        )
        for index, chunk_id in enumerate(("chunk-one", "chunk-two"), start=1)
    )
    episode = Episode(
        episode_id=make_episode_id(
            artifact_id=ARTIFACT,
            source_id="source-a",
            sequence_no=0,
            evidence=spans,
        ),
        artifact_id=ARTIFACT,
        source_id="source-a",
        sequence_no=0,
        first_ordinal=1,
        last_ordinal=2,
        evidence=spans,
        boundary_method="fixture",
    )
    plan = expand_episode_seeds(
        (_result("chunk-one", 0.9), _result("chunk-two", 0.8)),
        _Store(
            (episode,),
            {"chunk-one": episode.episode_id, "chunk-two": episode.episode_id},
        ),
        policy=EpisodeRetrievalPolicy(artifact_id=ARTIFACT),
    )
    assert len(plan.seeds) == 1
    assert plan.direct_chunk_ids == ("chunk-one", "chunk-two")
    assert [item.failure_code for item in plan.direct_fallbacks] == [
        "episode_mapped",
        "episode_mapped",
    ]


def test_episode_lookup_errors_and_missing_artifact_stay_distinct() -> None:
    class _BrokenStore(_Store):
        def episode_ids_for_chunks(self, chunk_ids, *, artifact_id=None):
            raise RuntimeError("corrupt episode lookup")

    result = _result("raw-chunk", 1.0, route="lexical")
    broken = expand_episode_seeds(
        (result,),
        _BrokenStore((), {}),
        policy=EpisodeRetrievalPolicy(artifact_id=ARTIFACT),
    )
    unscoped = expand_episode_seeds((result,), _Store((), {}))

    assert broken.direct_fallbacks[0].route == "lexical"
    assert broken.direct_fallbacks[0].failure_code == "lookup_error"
    assert unscoped.direct_fallbacks[0].failure_code == "artifact_not_selected"


def test_hard_caps_and_tie_breaks_are_exact_and_order_independent() -> None:
    anchors = tuple(_episode(index * 10) for index in range(3))
    neighbors = tuple(_episode(index) for index in (8, 9, 11, 12))
    mapping = {item.evidence[0].chunk_id: item.episode_id for item in anchors}
    adjacency = {
        anchors[0].episode_id: tuple(reversed(neighbors)),
        anchors[1].episode_id: tuple(neighbors),
        anchors[2].episode_id: tuple(neighbors),
    }
    store = _Store((*anchors, *neighbors), mapping, adjacency=adjacency)
    inputs = tuple(
        _result(item.evidence[0].chunk_id, 1.0)
        for item in reversed(anchors)
    )
    policy = EpisodeRetrievalPolicy(
        artifact_id=ARTIFACT,
        max_anchor_episodes=2,
        previous_episodes=2,
        next_episodes=2,
        max_episode_seeds=3,
        max_direct_fallbacks=1,
    )
    first = expand_episode_seeds(inputs, store, policy=policy)
    second = expand_episode_seeds(tuple(reversed(inputs)), store, policy=policy)
    assert first == second
    assert len(first.seeds) == 3
    assert [item.episode_id for item in first.seeds[:2]] == [
        anchors[0].episode_id,
        anchors[1].episode_id,
    ]
    assert anchors[2].episode_id in first.truncated_episode_ids
    assert first.receipt_sha256 == second.receipt_sha256


def test_direct_fallback_cap_is_reported_not_silent() -> None:
    store = _Store((), {})
    plan = expand_episode_seeds(
        (
            _result("missing-c", 0.7),
            _result("missing-a", 0.9),
            _result("missing-b", 0.8),
        ),
        store,
        policy=EpisodeRetrievalPolicy(
            artifact_id=ARTIFACT,
            max_direct_fallbacks=2,
        ),
    )
    assert plan.direct_chunk_ids == ("missing-a", "missing-b")
    assert plan.truncated_direct_chunk_ids == ("missing-c",)


def test_outputs_retain_no_input_text_vectors_or_transformer_state() -> None:
    anchor = _episode(1)
    store = _Store(
        (anchor,),
        {anchor.evidence[0].chunk_id: anchor.episode_id},
    )
    result = _result(anchor.evidence[0].chunk_id, 1.0)
    plan = expand_episode_seeds(
        (result,),
        store,
        policy=EpisodeRetrievalPolicy(artifact_id=ARTIFACT),
    )
    payload = plan.identity_payload()
    serialized = repr(payload)
    assert result.chunk.text not in serialized
    assert "123.0" not in serialized
    assert "456.0" not in serialized
    assert not any(
        forbidden in field.name.casefold()
        for field in fields(type(plan))
        for forbidden in ("token", "activation", "attention", "embedding", "kv_cache")
    )

    builder = EpisodeBuilder(
        min_size=1,
        max_size=2,
        detector=FixedIntervalBoundaryDetector(interval=1),
    )
    built = builder.build(
        source_id="source-a",
        artifact_id=ARTIFACT,
        spans=(_span(0), _span(1)),
        embeddings=((789.125, 0.0), (0.0, 789.125)),
    )
    assert "789.125" not in repr(built)
    assert not hasattr(builder, "__dict__")
    assert not hasattr(builder.detector, "__dict__")
