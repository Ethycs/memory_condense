"""Focused contracts for the provider-free incremental phrase graph."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.domain.schemas import Chunk, Turn
from memory_condense.search.incremental_conversation_graph import (
    ConversationGraphAppendDelta,
    ConversationGraphChunk,
    GraphTraversalPolicy,
    IncrementalConversationGraph,
    OrderedStorySearchPolicy,
    PhraseExtractionPolicy,
    StoryAffinityIndexPolicy,
    _story_term,
    extract_canonical_phrases,
)


def _chunk(
    chunk_id: str,
    text: str,
    *,
    source_id: str = "source-a",
    turn_id: str | None = None,
    ordinal: int = 1,
    start_char: int = 0,
    role: str = "user",
    created_at: str = "2026-09-06T12:00:00+00:00",
) -> ConversationGraphChunk:
    return ConversationGraphChunk(
        chunk_id=chunk_id,
        source_id=source_id,
        turn_id=turn_id or f"turn-{chunk_id}",
        ordinal=ordinal,
        role=role,
        text=text,
        start_char=start_char,
        end_char=start_char + len(text),
        created_at=created_at,
    )


def _targets_by_relation(graph: IncrementalConversationGraph, chunk_id: str):
    return {
        (edge.target_chunk_id, edge.relation)
        for edge in graph.transitions(
            chunk_id,
            policy=GraphTraversalPolicy(max_degree=20),
        )
    }


def test_phrase_extraction_is_conservative_canonical_and_exact() -> None:
    text = "We discussed the Cedar-Clinic launch; ZX-9 arrived."
    spans = extract_canonical_phrases(text)
    by_key = {span.key: span for span in spans}

    assert by_key["cedar clinic"].quote == "Cedar-Clinic"
    assert text[
        by_key["cedar clinic"].start_offset : by_key["cedar clinic"].end_offset
    ] == "Cedar-Clinic"
    # A two-character non-numeric token is deliberately too weak alone.
    assert "zx" not in by_key
    assert by_key["zx 9"].quote == "ZX-9"
    assert by_key["9 arrived"].quote == "9 arrived"
    assert "zx arrived" not in by_key
    assert "we" not in by_key
    assert all(span.token_count <= 3 for span in spans)


def test_append_retains_exact_occurrence_and_chunk_provenance() -> None:
    graph = IncrementalConversationGraph()
    chunk = _chunk(
        "c1",
        "Cedar-Clinic signed the Northwind Labs contract.",
        source_id="session-17",
        turn_id="turn-88",
        ordinal=7,
        start_char=100,
        role="assistant",
    )
    result = graph.append_chunk(chunk)
    occurrence = graph.occurrences("cedar clinic")[0]

    assert result.created is True
    assert result.receipt.chunk_identity_sha256
    assert occurrence.chunk_id == "c1"
    assert occurrence.source_id == "session-17"
    assert occurrence.turn_id == "turn-88"
    assert occurrence.quote == "Cedar-Clinic"
    assert occurrence.start_char == 100
    assert occurrence.end_char == 112
    assert occurrence.quote_sha256 == quote_sha256("Cedar-Clinic")
    assert graph.chunk("c1") is chunk


def test_append_is_incremental_idempotent_and_does_not_replace_occurrences() -> None:
    graph = IncrementalConversationGraph()
    first = _chunk("first", "Project Atlas selected cobalt lenses.")
    second = _chunk(
        "second",
        "Project Borealis selected quartz lenses.",
        ordinal=2,
    )
    first_result = graph.append_chunk(first)
    atlas_before = graph.occurrences("atlas")[0]
    graph.append_chunk(second)
    atlas_after = graph.occurrences("atlas")[0]
    retry = graph.append_chunk(first)

    assert atlas_after is atlas_before
    assert retry.created is False
    assert retry.receipt is first_result.receipt
    assert graph.stats().revision == 2
    assert graph.stats().chunk_count == 2


def test_backfill_splices_only_local_same_source_sequence_edges() -> None:
    graph = IncrementalConversationGraph()
    graph.append_chunk(_chunk("one", "alphaword", ordinal=1))
    graph.append_chunk(_chunk("three", "gammaword", ordinal=3))
    assert _targets_by_relation(graph, "one") == {
        ("three", "same_source_sequence")
    }

    graph.append_chunk(_chunk("two", "betaword", ordinal=2))

    assert _targets_by_relation(graph, "one") == {
        ("two", "same_source_sequence")
    }
    assert _targets_by_relation(graph, "two") == {
        ("one", "same_source_sequence"),
        ("three", "same_source_sequence"),
    }
    assert _targets_by_relation(graph, "three") == {
        ("two", "same_source_sequence")
    }
    assert graph.stats().sequence_edge_count == 2


def test_shared_phrase_edges_keep_exact_occurrence_receipts() -> None:
    graph = IncrementalConversationGraph()
    graph.append_chunk(
        _chunk("a1", "Project Juniper launched.", source_id="source-a", ordinal=1)
    )
    graph.append_chunk(
        _chunk("a2", "Project Juniper retrospective.", source_id="source-a", ordinal=2)
    )
    graph.append_chunk(
        _chunk("b1", "Project Juniper budget approved.", source_id="source-b")
    )

    edges = graph.transitions(
        "a1",
        question="What happened to Project Juniper?",
        policy=GraphTraversalPolicy(max_degree=20),
    )
    by_target = {edge.target_chunk_id: edge for edge in edges}

    assert by_target["a2"].relation == "same_source_sequence"
    assert by_target["b1"].relation == "shared_phrase"
    assert "project juniper" in by_target["b1"].shared_phrases
    assert by_target["b1"].source_occurrence is not None
    assert by_target["b1"].target_occurrence is not None
    assert by_target["b1"].source_occurrence.chunk_id == "a1"
    assert by_target["b1"].target_occurrence.chunk_id == "b1"


def test_question_seeded_walk_connects_local_window_to_global_memory_in_two_hops() -> None:
    graph = IncrementalConversationGraph()
    graph.append_chunk(
        _chunk(
            "local-fact",
            "The cobalt telescope is at Cedar Observatory.",
            source_id="conversation-a",
            ordinal=1,
        )
    )
    graph.append_chunk(
        _chunk(
            "local-link",
            "Its replacement vendor is Northwind Labs.",
            source_id="conversation-a",
            ordinal=2,
        )
    )
    graph.append_chunk(
        _chunk(
            "global-fact",
            "Northwind Labs contract carries warranty ZX-9.",
            source_id="document-b",
            ordinal=1,
        )
    )

    result = graph.search(
        "Where is the cobalt telescope?",
        policy=GraphTraversalPolicy(
            max_hops=2,
            max_degree=8,
            max_frontier=8,
            max_results=8,
        ),
    )
    global_evidence = next(
        row for row in result.evidence if row.chunk_id == "global-fact"
    )

    assert result.seed_chunk_ids == ("local-fact",)
    assert global_evidence.hop == 2
    assert [step.relation for step in global_evidence.path] == [
        "same_source_sequence",
        "shared_phrase",
    ]
    assert global_evidence.path[-1].shared_phrases[0] == "northwind labs"
    assert global_evidence.chunk.text.endswith("warranty ZX-9.")


def test_degree_frontier_and_hop_bounds_are_deterministic() -> None:
    graph = IncrementalConversationGraph()
    graph.append_chunk(_chunk("anchor", "Project Helios status.", source_id="s0"))
    for number in range(1, 7):
        graph.append_chunk(
            _chunk(
                f"target-{number}",
                f"Project Helios detail code-{number}.",
                source_id=f"s{number}",
            )
        )
    policy = GraphTraversalPolicy(
        max_hops=2,
        max_degree=2,
        max_frontier=1,
        max_results=3,
        max_seed_chunks=1,
    )

    first_edges = graph.transitions("anchor", policy=policy)
    second_edges = graph.transitions("anchor", policy=policy)
    first = graph.search("", seed_chunk_ids=("anchor",), policy=policy)
    second = graph.search("", seed_chunk_ids=("anchor",), policy=policy)

    assert first_edges == second_edges
    assert len(first_edges) == 2
    assert first == second
    assert len(first.evidence) <= policy.max_results + 1
    assert max(row.hop for row in first.evidence) <= 2
    assert sum(row.hop == 1 for row in first.evidence) <= 1


def test_saturated_phrase_hub_keeps_its_first_bounded_cohort_monotonically() -> None:
    graph = IncrementalConversationGraph()
    graph.append_chunk(
        _chunk("c0", "Project Helios update.", source_id="source-0")
    )
    graph.append_chunk(
        _chunk("c1", "Project Helios update.", source_id="source-1")
    )
    policy = GraphTraversalPolicy(max_phrase_postings=2, max_degree=10)
    before = graph.transitions("c0", policy=policy)
    graph.append_chunk(
        _chunk("c2", "Project Helios update.", source_id="source-2")
    )

    after = graph.transitions("c0", policy=policy)
    late = graph.transitions("c2", policy=policy)

    assert before == after
    assert {edge.target_chunk_id for edge in after} == {"c1"}
    assert late == ()


def test_from_chunk_requires_the_exact_cited_turn_substring() -> None:
    turn = Turn(turn_id="turn", role="user", text="prefix actual suffix")
    mismatched = Chunk(
        chunk_id="chunk",
        turn_id="turn",
        text="xxxxxx",
        start_char=7,
        end_char=13,
        token_count=1,
    )

    with pytest.raises(ValueError, match="exact cited turn substring"):
        ConversationGraphChunk.from_chunk(mismatched, turn, ordinal=1)


def test_shared_phrase_can_jump_between_nonadjacent_turns_in_one_source() -> None:
    graph = IncrementalConversationGraph()
    graph.append_chunk(
        _chunk("turn-1", "Project Juniper began.", ordinal=1)
    )
    for ordinal, word in ((2, "amber"), (3, "cobalt"), (4, "violet")):
        graph.append_chunk(
            _chunk(f"turn-{ordinal}", f"Unrelated {word} note.", ordinal=ordinal)
        )
    graph.append_chunk(
        _chunk("turn-5", "Juniper reached its final milestone.", ordinal=5)
    )

    edges = graph.transitions(
        "turn-1",
        policy=GraphTraversalPolicy(max_degree=8),
    )
    result = graph.search(
        "What happened to Juniper?",
        seed_chunk_ids=("turn-1",),
        derive_question_seeds=False,
    )

    assert any(
        edge.target_chunk_id == "turn-5" and edge.relation == "shared_phrase"
        for edge in edges
    )
    reached = next(row for row in result.evidence if row.chunk_id == "turn-5")
    assert reached.hop <= 2


def test_sequence_neighbors_are_reserved_outside_phrase_degree() -> None:
    graph = IncrementalConversationGraph()
    graph.append_chunk(_chunk("previous", "Previous local fact.", ordinal=1))
    graph.append_chunk(
        _chunk("seed", "Bridgeword central fact.", ordinal=2)
    )
    graph.append_chunk(_chunk("next", "Next local fact.", ordinal=3))
    for number in range(4):
        graph.append_chunk(
            _chunk(
                f"phrase-{number}",
                f"Bridgeword remote {number}.",
                source_id=f"remote-{number}",
            )
        )
    policy = GraphTraversalPolicy(max_hops=1, max_degree=1, max_results=8)

    edges = graph.transitions("seed", policy=policy)
    result = graph.search(
        "Follow Bridgeword",
        seed_chunk_ids=("seed",),
        derive_question_seeds=False,
        policy=policy,
    )

    sequence_targets = {
        edge.target_chunk_id
        for edge in edges
        if edge.relation == "same_source_sequence"
    }
    phrase_targets = {
        edge.target_chunk_id for edge in edges if edge.relation == "shared_phrase"
    }
    assert sequence_targets == {"previous", "next"}
    assert len(phrase_targets) == 1
    assert {"previous", "next"} <= set(result.chunk_ids)


def test_two_hop_stronger_path_relaxes_a_weaker_direct_arrival() -> None:
    graph = IncrementalConversationGraph()
    graph.append_chunk(_chunk("seed", "Commonterm seed.", ordinal=1))
    graph.append_chunk(
        _chunk("bridge", "Rare bridge phrase.", ordinal=2)
    )
    graph.append_chunk(
        _chunk(
            "target",
            "Commonterm rare bridge phrase target.",
            source_id="target-source",
        )
    )
    for number in range(7):
        graph.append_chunk(
            _chunk(
                f"common-{number}",
                f"Commonterm distractor {number}.",
                source_id=f"common-source-{number}",
            )
        )
    result = graph.search(
        "Trace the memory",
        seed_chunk_ids=("seed",),
        derive_question_seeds=False,
        policy=GraphTraversalPolicy(
            max_hops=2,
            max_degree=16,
            max_frontier=32,
            max_results=32,
        ),
    )

    target = next(row for row in result.evidence if row.chunk_id == "target")
    assert target.hop == 2
    assert [step.target_chunk_id for step in target.path] == ["bridge", "target"]
    assert target.supporting_seed_chunk_ids == ("seed",)


def test_explicit_seeds_do_not_consume_max_results_and_keep_query_matches() -> None:
    graph = IncrementalConversationGraph()
    graph.append_chunk(
        _chunk("seed", "Cobalt telescope anchor.", source_id="source-a")
    )
    graph.append_chunk(
        _chunk("novel", "Cobalt telescope detail.", source_id="source-b")
    )
    result = graph.search(
        "What about the cobalt telescope?",
        seed_chunk_ids=("seed",),
        derive_question_seeds=False,
        policy=GraphTraversalPolicy(max_hops=1, max_results=1),
    )

    assert result.chunk_ids == ("seed", "novel")
    seed = result.evidence[0]
    assert "cobalt telescope" in seed.matched_query_phrases
    assert seed.supporting_seed_chunk_ids == ("seed",)


def test_source_ordinal_cannot_name_two_different_turns() -> None:
    graph = IncrementalConversationGraph()
    graph.append_chunk(
        _chunk("first", "First fragment.", turn_id="turn-a", ordinal=4)
    )

    with pytest.raises(ValueError, match="source/ordinal"):
        graph.append_chunk(
            _chunk(
                "second",
                "Second fragment.",
                turn_id="turn-b",
                ordinal=4,
                start_char=20,
            )
        )


def test_single_digit_is_retained_and_stopwords_do_not_fabricate_ngrams() -> None:
    keys = {span.key for span in extract_canonical_phrases("Model 7 in Bay")}

    assert {"7", "model 7"} <= keys
    assert "7 bay" not in keys
    assert "model 7 bay" not in keys


def test_contract_adapter_and_collision_validation() -> None:
    turn = Turn(
        turn_id="turn-1",
        source_id="conversation-1",
        role="user",
        text="Exact source text",
        created_at=datetime(2026, 9, 6, tzinfo=timezone.utc),
    )
    chunk = Chunk(
        chunk_id="chunk-1",
        turn_id="turn-1",
        text="Exact source text",
        start_char=0,
        end_char=17,
        token_count=3,
    )
    adapted = ConversationGraphChunk.from_chunk(chunk, turn, ordinal=4)
    graph = IncrementalConversationGraph()
    graph.append_chunk(adapted)

    assert adapted.source_id == "conversation-1"
    assert adapted.ordinal == 4
    with pytest.raises(ValueError, match="chunk_id"):
        graph.append_chunk(
            _chunk(
                "chunk-1",
                "Different evidence",
                source_id="conversation-1",
                ordinal=4,
            )
        )
    with pytest.raises(ValueError, match="ordinal/start_char"):
        graph.append_chunk(
            _chunk(
                "chunk-2",
                "Coordinate collision",
                source_id="conversation-1",
                ordinal=4,
            )
        )


def test_invalid_limits_and_unknown_seed_fail_closed() -> None:
    with pytest.raises(ValueError, match="cannot exceed 2"):
        GraphTraversalPolicy(max_hops=3)
    with pytest.raises(ValueError, match="exactly cover"):
        ConversationGraphChunk(
            chunk_id="bad",
            source_id="source",
            turn_id="turn",
            ordinal=0,
            role="user",
            text="text",
            start_char=0,
            end_char=99,
        )
    graph = IncrementalConversationGraph(
        extraction_policy=PhraseExtractionPolicy(max_phrases_per_chunk=8)
    )
    with pytest.raises(KeyError, match="unknown graph seed"):
        graph.search("", seed_chunk_ids=("missing",))


def test_question_can_steer_explicit_seeds_without_creating_implicit_seeds() -> None:
    graph = IncrementalConversationGraph()
    graph.append_chunk(_chunk("a", "Cobalt telescope fact.", source_id="a"))
    graph.append_chunk(_chunk("b", "Cobalt telescope detail.", source_id="b"))

    result = graph.search(
        "What about the cobalt telescope?",
        derive_question_seeds=False,
    )

    assert result.question_seed_derivation_enabled is False
    assert result.seed_chunk_ids == ()
    assert result.evidence == ()


def _append_q86_story(graph: IncrementalConversationGraph) -> None:
    rows = (
        (
            "muir-event",
            "muir-session",
            1,
            "2023-03-10T23:32:00+00:00",
            "I just got back from a day hike to Muir Woods with my family.",
        ),
        (
            "muir-context",
            "muir-session",
            2,
            "2023-03-10T23:32:00+00:00",
            "Eastern Sierra backpacking requires careful bear safety practice.",
        ),
        (
            "big-sur-event",
            "big-sur-session",
            1,
            "2023-04-20T16:29:00+00:00",
            "I just returned from a road trip with friends to Big Sur.",
        ),
        (
            "big-sur-context",
            "big-sur-session",
            2,
            "2023-04-20T16:29:00+00:00",
            "My Eastern Sierra backpacking checklist emphasizes bear safety.",
        ),
        (
            "yosemite-event",
            "yosemite-session",
            1,
            "2023-05-15T06:30:00+00:00",
            "I just returned from a solo camping trip to Yosemite.",
        ),
        (
            "yosemite-context",
            "yosemite-session",
            2,
            "2023-05-15T06:30:00+00:00",
            "Eastern Sierra backpacking made bear safety second nature.",
        ),
        (
            "new-york-event",
            "new-york-session",
            1,
            "2023-05-29T10:11:00+00:00",
            "I took a quick business trip to New York for a conference.",
        ),
        (
            "yellowstone-event",
            "yellowstone-session",
            1,
            "2023-05-25T17:38:00+00:00",
            "I took a seven day road trip to Yellowstone.",
        ),
        (
            "whistler-event",
            "whistler-session",
            1,
            "2023-03-22T15:40:00+00:00",
            "We went on a ski trip to Whistler Blackcomb.",
        ),
    )
    for chunk_id, source_id, ordinal, created_at, text in rows:
        graph.append_chunk(
            ConversationGraphChunk(
                chunk_id=chunk_id,
                source_id=source_id,
                turn_id=f"turn-{chunk_id}",
                ordinal=ordinal,
                role="user",
                text=text,
                start_char=0,
                end_char=len(text),
                created_at=created_at,
            )
        )


def test_ordered_story_selects_exact_k_coherent_sources_not_ranked_trip_noise() -> None:
    graph = IncrementalConversationGraph()
    _append_q86_story(graph)
    candidates = (
        "new-york-event",
        "yellowstone-event",
        "whistler-event",
        "muir-event",
        "big-sur-event",
        "yosemite-event",
    )

    result = graph.search_ordered_story(
        seed_chunk_ids=("muir-event",),
        requested_count=3,
        candidate_chunk_ids=candidates,
    )
    replay = graph.search_ordered_story(
        seed_chunk_ids=("muir-event",),
        requested_count=3,
        candidate_chunk_ids=candidates,
    )

    assert result == replay
    assert result.status == "selected"
    assert result.candidate_derivation == "explicit_candidates"
    assert result.selected_source_ids == (
        "muir-session",
        "big-sur-session",
        "yosemite-session",
    )
    assert result.selected_representative_chunk_ids == (
        "muir-event",
        "big-sur-event",
        "yosemite-event",
    )
    assert len(result.selected_sources) == result.requested_count == 3
    assert all(row.score > 0.0 and row.terms for row in result.selected_pair_affinities)
    assert {
        term.term
        for pair in result.selected_pair_affinities
        for term in pair.terms
    } >= {"eastern", "sierra", "backpack", "safety"}
    assert all(
        term.left_occurrence.source_id == pair.left_source_id
        and term.right_occurrence.source_id == pair.right_source_id
        for pair in result.selected_pair_affinities
        for term in pair.terms
    )
    assert len(result.receipt_sha256) == 64


def test_seed_source_expansion_finds_the_same_local_to_global_story() -> None:
    graph = IncrementalConversationGraph()
    _append_q86_story(graph)

    result = graph.search_ordered_story(
        seed_chunk_ids=("muir-event",),
        requested_count=3,
    )

    assert result.status == "selected"
    assert result.candidate_derivation == "seed_source_expansion"
    assert result.selected_source_ids == (
        "muir-session",
        "big-sur-session",
        "yosemite-session",
    )
    evidence_ids = {
        chunk.chunk_id
        for source in result.selected_sources
        for chunk in source.evidence_chunks
    }
    assert {
        "muir-event",
        "big-sur-event",
        "yosemite-event",
        "muir-context",
        "big-sur-context",
        "yosemite-context",
    } <= evidence_ids


def test_story_term_source_saturation_is_first_seen_and_monotone() -> None:
    graph = IncrementalConversationGraph(
        story_index_policy=StoryAffinityIndexPolicy(
            max_sources_per_term=2,
        )
    )
    first = graph.append_chunk(
        _chunk("first", "Rarebridge", source_id="source-a")
    )
    second = graph.append_chunk(
        _chunk("second", "Rarebridge", source_id="source-b")
    )
    before = graph.story_term_sources("Rarebridges")
    third = graph.append_chunk(
        _chunk("third", "Rarebridge", source_id="source-c")
    )

    assert first.receipt.new_story_term_membership_count == 1
    assert second.receipt.new_story_term_membership_count == 1
    assert third.receipt.new_story_term_membership_count == 0
    assert before == graph.story_term_sources("Rarebridges") == (
        "source-a",
        "source-b",
    )
    assert graph.story_source_terms("source-c") == ()
    stats = graph.stats()
    assert stats.story_term_count == 1
    assert stats.story_term_membership_count == 2
    assert stats.story_evidence_chunk_count == 3


@pytest.mark.parametrize(
    ("surface", "expected"),
    [
        ("closed", "closed"),
        ("raised", "raised"),
        ("stories", "story"),
        ("planning", "plan"),
        ("Rarebridges", "rarebridge"),
    ],
)
def test_story_term_normalization_emits_only_fixed_points(
    surface: str,
    expected: str,
) -> None:
    assert _story_term(surface) == expected
    assert _story_term(expected) == expected


def test_unstable_legacy_suffix_term_appends_and_restores_directly() -> None:
    source = IncrementalConversationGraph()
    chunk = _chunk(
        "closed-event",
        "Cedar Archive closed after the gates were raised.",
        source_id="closed-session",
    )

    result = source.append_chunk(chunk)
    delta = source.append_delta(chunk.chunk_id)
    restored = IncrementalConversationGraph()
    restored_result = restored.restore_append_delta(delta)

    assert result.created is True
    assert restored_result.created is True
    assert {"closed", "raised"} <= set(
        source.story_source_terms("closed-session")
    )
    assert restored.story_source_terms("closed-session") == (
        source.story_source_terms("closed-session")
    )


def test_authenticated_append_delta_round_trip_restores_story_index() -> None:
    source = IncrementalConversationGraph()
    chunks = (
        _chunk(
            "alpha-one",
            "Eastern Sierra backpack planning.",
            source_id="source-alpha",
            ordinal=1,
        ),
        _chunk(
            "beta-one",
            "Eastern Sierra campsite safety.",
            source_id="source-beta",
            ordinal=1,
            created_at="2026-09-07T12:00:00+00:00",
        ),
        _chunk(
            "alpha-two",
            "Bear canister safety follow-up.",
            source_id="source-alpha",
            ordinal=2,
            created_at="2026-09-08T12:00:00+00:00",
        ),
    )
    deltas = []
    for chunk in chunks:
        source.append_chunk(chunk)
        deltas.append(source.append_delta(chunk.chunk_id))

    restored = IncrementalConversationGraph()
    for delta in deltas:
        result = restored.restore_append_delta(delta)
        assert result.created is True
        assert restored.append_delta(delta.chunk.chunk_id) == delta
        retry = restored.restore_append_delta(delta)
        assert retry.created is False
        assert retry.receipt == delta.receipt

    assert restored.stats() == source.stats()
    assert restored.story_source_terms("source-alpha") == source.story_source_terms(
        "source-alpha"
    )
    assert restored.story_term_sources("Sierra") == (
        "source-alpha",
        "source-beta",
    )
    result = restored.search_ordered_story(
        seed_chunk_ids=("alpha-one",),
        requested_count=2,
        candidate_chunk_ids=("alpha-one", "beta-one"),
    )
    assert result.status == "selected"
    assert result.selected_source_ids == ("source-alpha", "source-beta")


def test_append_delta_restore_fails_before_mutation_on_tampering() -> None:
    source = IncrementalConversationGraph()
    source.append_chunk(_chunk("first", "Rarebridge alpha evidence."))
    source.append_chunk(
        _chunk(
            "second",
            "Rarebridge beta evidence.",
            source_id="source-b",
            ordinal=2,
        )
    )
    first = source.append_delta("first")
    second = source.append_delta("second")

    target = IncrementalConversationGraph()
    with pytest.raises(ValueError, match="revision is not contiguous"):
        target.restore_append_delta(second)
    assert target.stats().revision == 0

    broken_occurrence = replace(first.occurrences[0], quote="forged")
    forged = ConversationGraphAppendDelta(
        chunk=first.chunk,
        occurrences=(broken_occurrence, *first.occurrences[1:]),
        new_story_term_memberships=first.new_story_term_memberships,
        story_evidence_chunk_retained=first.story_evidence_chunk_retained,
        receipt=first.receipt,
    )
    with pytest.raises(ValueError, match="exact chunk provenance"):
        target.restore_append_delta(forged)
    assert target.stats().revision == 0

    missing_memberships = replace(first, new_story_term_memberships=())
    with pytest.raises(ValueError, match="counts disagree"):
        target.restore_append_delta(missing_memberships)
    assert target.stats().revision == 0

    wrong_policy = IncrementalConversationGraph(
        story_index_policy=StoryAffinityIndexPolicy(max_sources_per_term=2)
    )
    with pytest.raises(ValueError, match="story policy does not match"):
        wrong_policy.restore_append_delta(first)
    assert wrong_policy.stats().revision == 0


def test_ordered_story_fails_closed_on_unbound_or_unprovable_inputs() -> None:
    graph = IncrementalConversationGraph()
    graph.append_chunk(
        _chunk(
            "alpha",
            "Cedarbridge alpha memory.",
            source_id="source-a",
        )
    )
    graph.append_chunk(
        _chunk(
            "beta",
            "Quartzbridge beta memory.",
            source_id="source-b",
            ordinal=2,
            created_at="2026-09-07T12:00:00+00:00",
        )
    )

    with pytest.raises(ValueError, match="requires explicit seed"):
        graph.search_ordered_story(seed_chunk_ids=(), requested_count=2)
    with pytest.raises(KeyError, match="unknown ordered story"):
        graph.search_ordered_story(
            seed_chunk_ids=("missing",),
            requested_count=2,
        )
    with pytest.raises(ValueError, match="ordered subset"):
        graph.search_ordered_story(
            seed_chunk_ids=("alpha",),
            requested_count=2,
            candidate_chunk_ids=("beta",),
        )
    with pytest.raises(ValueError, match="ordered subset"):
        graph.search_ordered_story(
            seed_chunk_ids=("beta", "alpha"),
            requested_count=2,
            candidate_chunk_ids=("alpha", "beta"),
        )
    with pytest.raises(ValueError, match="candidate population"):
        graph.search_ordered_story(
            seed_chunk_ids=("alpha",),
            requested_count=2,
            candidate_chunk_ids=("alpha", "beta"),
            policy=OrderedStorySearchPolicy(max_candidate_chunks=1),
        )

    disconnected = graph.search_ordered_story(
        seed_chunk_ids=("alpha",),
        requested_count=2,
        candidate_chunk_ids=("alpha", "beta"),
    )
    assert disconnected.status == "no_connected_story"
    assert disconnected.selected_sources == ()
    assert disconnected.selected_pair_affinities == ()
