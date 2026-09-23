from __future__ import annotations

import json

import pytest

from memory_condense.associations.head_memory_models import (
    MemoryLinkHit,
    NestedMemoryInspection,
)
from memory_condense.search.context_card_retrieval import (
    attention_search_context_cards,
    lexical_route_context_card_sources,
)
from memory_condense.search.context_cards import (
    ContextCardPolicy,
    ContextMemory,
    build_context_window,
    make_context_card_request,
    materialize_context_card,
)


class _Linker:
    max_candidates = 8

    def __init__(self) -> None:
        self.groups = None

    def inspect_nested(
        self,
        _query,
        groups,
        *,
        beam_per_group,
        top_k,
        score_mode,
    ):
        self.groups = groups
        candidates = [candidate for group in groups for candidate in group]
        hits = tuple(
            MemoryLinkHit(
                episode_id=candidate.episode_id,
                qk_score=float(len(candidates) - index),
                ov_transport=0.5,
                head_weights=(1.0,),
                metadata=dict(candidate.metadata),
            )
            for index, candidate in enumerate(candidates[:top_k])
        )
        return NestedMemoryInspection(hits, 1, len(candidates), 20, len(candidates))


class _OverflowLinker:
    max_candidates = 8

    def inspect_nested(self, *_args, **_kwargs):
        raise MemoryError("fixture workspace overflow")


class _FallbackSignalingLinker(_Linker):
    requires_raw_fallback = True


class _MalformedFallbackSignalingLinker(_Linker):
    requires_raw_fallback = "yes"


class _DistilledLinker(_Linker):
    requires_raw_fallback = False

    def inspect_nested(
        self,
        _query,
        groups,
        *,
        beam_per_group,
        top_k,
        score_mode,
    ):
        candidates = [candidate for group in groups for candidate in group]
        hit = MemoryLinkHit(
            episode_id=candidates[0].episode_id,
            qk_score=0.0,
            ov_transport=0.0,
            head_weights=(),
            metadata={
                **candidates[0].metadata,
                "selection_backend": "qkov_distilled_student",
                "score_kind": "distilled_ranking_scalar",
                "distilled_student_score": 2.5,
                "qk_ov_measured": False,
            },
        )
        return NestedMemoryInspection((hit,), 1, len(candidates), 0, len(candidates))


def _fixture():
    memories = [
        ContextMemory("old", "source", 1, "user", "The codename is Cobalt."),
        ContextMemory("target", "source", 2, "user", "Use it for the launch."),
    ]
    policy = ContextCardPolicy(previous_memories=1)
    request = make_context_card_request(
        build_context_window(memories, "target", policy=policy), policy=policy
    )
    completion = json.dumps(
        {
            "statement": "Use Cobalt for the launch.",
            "target_quote": "Use it for the launch.",
            "context_alias": "M1",
            "context_quote": "codename is Cobalt",
            "topics": ["launch codename"],
            "entities": ["Cobalt"],
        }
    )
    card = materialize_context_card(
        completion,
        request,
        generator_identity={"provider": "fixture", "model": "extract"},
    )
    return memories, card


def test_attention_uses_cards_then_excludes_and_dedupes_raw_chunks() -> None:
    memories, card = _fixture()
    linker = _Linker()

    result = attention_search_context_cards(
        "What codename should the launch use?",
        [card],
        memories,
        linker=linker,
        source_order=["source"],
        eligible_target_memory_ids=["target"],
        source_scope_complete=True,
        exclude_memory_ids=["target"],
    )

    assert result.selected_card_ids == (card.card_id,)
    assert not result.requires_raw_fallback
    assert [evidence.memory_id for evidence in result.evidence] == ["old"]
    assert result.eligible_source_ids == ("source",)
    assert result.covered_source_ids == ("source",)
    assert result.uncovered_source_ids == ()
    assert result.eligible_target_memory_ids == ("target",)
    assert result.covered_target_memory_ids == ("target",)
    assert result.uncovered_target_memory_ids == ()
    assert linker.groups[0][0].text == card.routing_text
    assert "The codename is Cobalt." not in linker.groups[0][0].metadata.values()


def test_partial_card_population_requires_additive_raw_fallback() -> None:
    memories, card = _fixture()

    result = attention_search_context_cards(
        "What codename should the launch use?",
        [card],
        memories,
        linker=_Linker(),
        source_order=["source"],
    )

    assert result.eligible_target_memory_ids == ("old", "target")
    assert result.covered_target_memory_ids == ("target",)
    assert result.uncovered_target_memory_ids == ("old",)
    assert result.covered_source_ids == ()
    assert result.uncovered_source_ids == ("source",)
    assert result.requires_raw_fallback


def test_hydration_rejects_changed_raw_memory() -> None:
    memories, card = _fixture()
    changed = [
        ContextMemory("old", "source", 1, "user", "The codename is Amber."),
        memories[1],
    ]

    with pytest.raises(RuntimeError, match="changed"):
        attention_search_context_cards(
            "What codename should the launch use?",
            [card],
            changed,
            linker=_Linker(),
            source_order=["source"],
        )


def test_no_search_occurs_when_only_valid_empty_cards_exist() -> None:
    memories, card = _fixture()
    empty_request = make_context_card_request(
        build_context_window(memories, "target"),
    )
    empty = materialize_context_card(
        '{"statement":null,"target_quote":null,"context_alias":null,'
        '"context_quote":null,"topics":[],"entities":[]}',
        empty_request,
        generator_identity={"provider": "fixture"},
    )

    result = attention_search_context_cards(
        "question",
        [empty],
        memories,
        linker=_Linker(),
        source_order=["source"],
    )

    assert result.inspection is None
    assert result.evidence == ()
    assert result.covered_target_memory_ids == ()
    assert result.uncovered_target_memory_ids == ("old", "target")
    assert result.uncovered_source_ids == ("source",)
    assert result.requires_raw_fallback


def _three_memory_fixture():
    memories = [
        ContextMemory("old", "source", 1, "user", "The codename is Cobalt."),
        ContextMemory("middle", "source", 2, "assistant", "Bring a spare cable."),
        ContextMemory("target", "source", 3, "user", "Use it for the launch."),
    ]
    policy = ContextCardPolicy(previous_memories=2)
    request = make_context_card_request(
        build_context_window(memories, "target", policy=policy), policy=policy
    )
    card = materialize_context_card(
        json.dumps(
            {
                "statement": "Use Cobalt for the launch.",
                "target_quote": "Use it for the launch.",
                "context_alias": "M1",
                "context_quote": "codename is Cobalt",
                "topics": ["launch codename"],
                "entities": ["Cobalt"],
            }
        ),
        request,
        generator_identity={"provider": "fixture", "model": "extract"},
    )
    return memories, card


def test_hydration_preserves_target_and_cited_memory_before_neighborhood() -> None:
    memories, card = _three_memory_fixture()

    result = attention_search_context_cards(
        "What codename should the launch use?",
        [card],
        memories,
        linker=_Linker(),
        source_order=["source"],
        eligible_target_memory_ids=["target"],
        source_scope_complete=True,
        max_raw_chunks=2,
    )

    assert [evidence.memory_id for evidence in result.evidence] == ["target", "old"]
    assert result.omitted_memory_ids == ("middle",)
    assert result.requires_raw_fallback


def test_hydration_does_not_partially_admit_an_over_budget_atomic_set() -> None:
    memories, card = _three_memory_fixture()

    result = attention_search_context_cards(
        "What codename should the launch use?",
        [card],
        memories,
        linker=_Linker(),
        source_order=["source"],
        eligible_target_memory_ids=["target"],
        source_scope_complete=True,
        max_raw_chunks=1,
    )

    assert result.evidence == ()
    assert result.omitted_memory_ids == ("target", "old", "middle")
    assert result.requires_raw_fallback


def test_attention_workspace_overflow_fails_open() -> None:
    memories, card = _fixture()

    result = attention_search_context_cards(
        "What codename should the launch use?",
        [card],
        memories,
        linker=_OverflowLinker(),
        source_order=["source"],
        eligible_target_memory_ids=["target"],
        source_scope_complete=True,
    )

    assert result.inspection is None
    assert result.evidence == ()
    assert result.attention_overflow
    assert result.requires_raw_fallback


def test_linker_fallback_signal_propagates_through_raw_hydration() -> None:
    memories, card = _fixture()

    result = attention_search_context_cards(
        "What codename should the launch use?",
        [card],
        memories,
        linker=_FallbackSignalingLinker(),
        source_order=["source"],
        eligible_target_memory_ids=["target"],
        source_scope_complete=True,
    )

    assert result.selected_card_ids == (card.card_id,)
    assert result.evidence
    assert result.requires_raw_fallback


def test_distilled_selection_semantics_survive_raw_hydration() -> None:
    memories, card = _fixture()

    result = attention_search_context_cards(
        "What codename should the launch use?",
        [card],
        memories,
        linker=_DistilledLinker(),
        source_order=["source"],
        eligible_target_memory_ids=["target"],
        source_scope_complete=True,
    )

    evidence = result.evidence[0]
    assert evidence.qk_score == 0.0
    assert evidence.ov_transport == 0.0
    assert evidence.selection_score == pytest.approx(2.5)
    assert evidence.selection_score_kind == "distilled_ranking_scalar"
    assert evidence.selection_backend == "qkov_distilled_student"
    assert not evidence.qk_ov_measured


def test_non_boolean_linker_fallback_signal_is_rejected() -> None:
    memories, card = _fixture()

    with pytest.raises(TypeError, match="requires_raw_fallback must be bool"):
        attention_search_context_cards(
            "What codename should the launch use?",
            [card],
            memories,
            linker=_MalformedFallbackSignalingLinker(),
            source_order=["source"],
            eligible_target_memory_ids=["target"],
            source_scope_complete=True,
        )


def test_lexical_source_gate_routes_matching_cards_and_fails_open() -> None:
    memories, launch = _fixture()
    garden_memory = ContextMemory(
        "garden", "garden-source", 1, "user", "Water rosemary on Friday."
    )
    garden_request = make_context_card_request(
        build_context_window([garden_memory], "garden")
    )
    garden = materialize_context_card(
        json.dumps(
            {
                "statement": "Water rosemary on Friday.",
                "target_quote": "Water rosemary on Friday.",
                "context_alias": None,
                "context_quote": None,
                "topics": ["garden watering"],
                "entities": ["rosemary"],
            }
        ),
        garden_request,
        generator_identity={"provider": "fixture"},
    )

    matched = lexical_route_context_card_sources(
        "What gets watered on Friday?", [launch, garden], max_sources=1
    )
    fallback = lexical_route_context_card_sources(
        "unrepresented vocabulary",
        [launch, garden],
        max_sources=1,
        eligible_source_ids=["source", "garden-source", "raw-only"],
    )

    assert [route.source_id for route in matched] == ["garden-source"]
    assert {route.source_id for route in fallback} == {
        "source",
        "garden-source",
        "raw-only",
    }
