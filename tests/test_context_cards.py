from __future__ import annotations

import copy
import json

import pytest

from memory_condense.search.context_cards import (
    ContextCardPolicy,
    ContextCardValidationError,
    ContextMemory,
    ContextWindowUnavailableError,
    build_context_window,
    make_context_card_request,
    materialize_context_card,
)


GENERATOR = {"provider": "fixture", "model_id": "extract-test", "revision": "1"}


def _memories() -> list[ContextMemory]:
    return [
        ContextMemory(
            "a1",
            "alpha",
            1,
            "user",
            "Mira owns the Atlas deployment.",
            "2026-09-01T00:00:00Z",
        ),
        ContextMemory("b1", "beta", 1, "user", "The garden has rosemary."),
        ContextMemory(
            "a2",
            "alpha",
            2,
            "assistant",
            "Atlas was scheduled for Monday.",
        ),
        ContextMemory("b2", "beta", 2, "user", "Water it on Friday."),
        ContextMemory(
            "a3",
            "alpha",
            3,
            "user",
            "Actually, move it to Tuesday.",
        ),
        ContextMemory("a4", "alpha", 4, "user", "Keep the rollback window."),
    ]


def _request():
    policy = ContextCardPolicy(previous_memories=2)
    window = build_context_window(_memories(), "a3", policy=policy)
    return make_context_card_request(window, policy=policy)


def _completion() -> dict[str, object]:
    return {
        "statement": "The Atlas deployment moved from Monday to Tuesday.",
        "target_quote": "move it to Tuesday",
        "context_alias": "M2",
        "context_quote": "Atlas was scheduled for Monday.",
        "topics": ["Atlas deployment", "scheduling"],
        "entities": ["Atlas"],
    }


def test_window_is_last_n_predecessors_from_only_the_target_source() -> None:
    policy = ContextCardPolicy(previous_memories=2)
    window = build_context_window(_memories(), "a4", policy=policy)

    assert [memory.memory_id for memory in window.memories] == ["a2", "a3", "a4"]
    assert window.target_alias == "M3"
    assert all(memory.source_id == "alpha" for memory in window.memories)


def test_window_orders_chunks_within_the_same_turn() -> None:
    memories = [
        ContextMemory(
            "turn-late",
            "source",
            7,
            "user",
            "Second chunk.",
            turn_start_char=20,
        ),
        ContextMemory(
            "turn-early",
            "source",
            7,
            "user",
            "First chunk.",
            turn_start_char=0,
        ),
    ]

    window = build_context_window(memories, "turn-late")

    assert [memory.memory_id for memory in window.memories] == [
        "turn-early",
        "turn-late",
    ]


def test_window_drops_oldest_context_but_never_target_to_meet_cap() -> None:
    target = ContextMemory("target", "s", 2, "user", "The answer is cobalt.")
    long_old = ContextMemory("old", "s", 1, "user", "noise " * 1000)
    target_only = build_context_window(
        [target],
        "target",
        policy=ContextCardPolicy(previous_memories=1),
    )
    policy = ContextCardPolicy(
        previous_memories=1,
        max_window_tokens=target_only.token_count,
    )

    bounded = build_context_window([long_old, target], "target", policy=policy)

    assert bounded.memories == (target,)


def test_window_fails_open_when_target_alone_exceeds_cap() -> None:
    target = ContextMemory("target", "s", 1, "user", "large " * 100)
    with pytest.raises(ContextWindowUnavailableError):
        build_context_window(
            [target],
            "target",
            policy=ContextCardPolicy(max_window_tokens=8),
        )


def test_materialization_resolves_exact_quotes_and_is_deterministic() -> None:
    request = _request()
    raw = json.dumps(_completion(), separators=(",", ":"))

    first = materialize_context_card(raw, request, generator_identity=GENERATOR)
    second = materialize_context_card(raw, request, generator_identity=GENERATOR)

    assert first == second
    assert first.card_id.startswith("context-card-")
    assert first.target_memory_id == "a3"
    assert "Atlas deployment" in first.routing_text
    assert [citation.memory_id for citation in first.facts[0].citations] == [
        "a2",
        "a3",
    ]
    target_citation = first.facts[0].citations[1]
    assert request.window.target.text[
        target_citation.start_char : target_citation.end_char
    ] == target_citation.quote


def test_valid_empty_card_is_distinct_from_invalid_or_unavailable() -> None:
    request = _request()
    card = materialize_context_card(
        '{"statement":null,"target_quote":null,"context_alias":null,'
        '"context_quote":null,"topics":[],"entities":[]}',
        request,
        generator_identity=GENERATOR,
    )

    assert card.is_empty
    assert card.routing_text == ""
    with pytest.raises(ContextCardValidationError):
        materialize_context_card("", request, generator_identity=GENERATOR)


def test_empty_statement_cannot_create_ungrounded_routing_labels() -> None:
    request = _request()

    with pytest.raises(ContextCardValidationError, match="empty topics and entities"):
        materialize_context_card(
            '{"statement":null,"target_quote":null,"context_alias":null,'
            '"context_quote":null,"topics":["hallucinated"],"entities":[]}',
            request,
            generator_identity=GENERATOR,
        )


@pytest.mark.parametrize(
    "mutation",
    ["extra_root", "unknown_alias", "inexact_quote", "missing_target", "extra_fact"],
)
def test_untrusted_completion_fails_closed(mutation: str) -> None:
    request = _request()
    payload = copy.deepcopy(_completion())
    if mutation == "extra_root":
        payload["commentary"] = "no"
    elif mutation == "unknown_alias":
        payload["context_alias"] = "M99"
    elif mutation == "inexact_quote":
        payload["target_quote"] = "Move it to Wednesday"
    elif mutation == "missing_target":
        payload["target_quote"] = None
    elif mutation == "extra_fact":
        payload["confidence"] = 1.0

    with pytest.raises((ContextCardValidationError, ValueError)):
        materialize_context_card(
            json.dumps(payload), request, generator_identity=GENERATOR
        )


def test_prompt_contains_no_query_or_durable_memory_ids() -> None:
    request = _request()

    assert "question" not in request.user_prompt.casefold()
    assert "a1" not in request.user_prompt
    assert "a2" not in request.user_prompt
    assert "a3" not in request.user_prompt
    assert "<<< TARGET M3 >>>" in request.user_prompt
