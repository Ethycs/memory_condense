from __future__ import annotations

from enum import Enum
from types import SimpleNamespace

import pytest

from memory_condense.eval.recall_guarded_cumulative_fast_artifact import FastEvidence
from tools._routed_repair_prompts import (
    DEFAULT_MEASURED_ARM,
    MAX_ROUTED_PROMPT_TOKENS,
    RoutedRepairPromptError,
    build_routed_answer_prompt,
    build_routed_fact_compression_prompt,
    numeric_facts_are_quote_grounded,
    normalize_repair_style,
)
from tools._routed_repair_routing import RoutedRepairStyle, route_question


def _evidence(identity: str, text: str, source: str) -> FastEvidence:
    return FastEvidence(evidence_id=identity, source_id=source, text=text)


def _question(
    *,
    noisy: bool = False,
    question_text: str = "How many parts are in the inventory in total?",
):
    root = _evidence(
        "root",
        "The initial inventory contained 3 blue parts.",
        "source-root",
    )
    root_copy = _evidence("root-copy", root.text, root.source_id)
    addition = _evidence(
        "addition",
        "On Tuesday, 4 red parts were added to the inventory.",
        "source-addition",
    )
    noise = _evidence(
        "noise",
        "Unrelated workshop note. " * (1_000 if noisy else 1),
        "source-noise",
    )
    return SimpleNamespace(
        question_id="q-routed",
        dated_question=(
            "[Question asked at 2024/01/10]\n"
            + question_text
        ),
        stages=(
            SimpleNamespace(
                stage_id="causal_graph_coverage_predecessor",
                evidence=(root,),
            ),
            SimpleNamespace(
                stage_id="direct_episode_additions",
                evidence=(root, root_copy, addition, noise),
            ),
        ),
    )


def _valid_response() -> str:
    return (
        '{"facts":[{"text":"4 red parts were added Tuesday.",'
        '"citations":[{"evidence_alias":"E001",'
        '"quote":"4 red parts were added"}]}]}'
    )


class _ForeignStyle(Enum):
    VALUE = "numeric_reduce"


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("numeric_reduce", RoutedRepairStyle.NUMERIC_REDUCE),
        ("numeric-aggregate-compare", RoutedRepairStyle.NUMERIC_REDUCE),
        (_ForeignStyle.VALUE, RoutedRepairStyle.NUMERIC_REDUCE),
        (RoutedRepairStyle.TIMELINE, RoutedRepairStyle.TIMELINE),
        ("direct_lookup", RoutedRepairStyle.EXTRACT),
    ),
)
def test_normalizes_string_enum_and_analysis_style_aliases(value, expected) -> None:
    assert normalize_repair_style(value) is expected


def test_accepts_question_route_receipt_without_exposing_gold_inputs() -> None:
    question = _question()
    receipt = route_question(question.dated_question)

    prompt = build_routed_fact_compression_prompt(question, receipt)

    assert prompt.style is receipt.style
    assert prompt.receipt_sha256
    assert prompt.prompt_token_proxy <= MAX_ROUTED_PROMPT_TOKENS
    joined = "\n".join(row.content for row in prompt.messages)
    assert "reference answer" in joined
    assert "do not use a reference answer" in joined.casefold()


@pytest.mark.parametrize(
    ("style", "compression_marker", "answer_marker"),
    (
        ("numeric_reduce", "every potentially relevant operand", "requested count"),
        ("temporal_timeline", "one event per cited fact", "dated event sequence"),
        ("state_chain", "every relevant dated state", "completed supersessions"),
        ("set_join", "one candidate member per cited fact", "deduplicate only"),
        ("synthesize", "each relevant attributed claim", "reconcile the cited claims"),
        ("direct_extract", "exact candidate value", "direct extract"),
    ),
)
def test_each_style_adds_explicit_gold_blind_operator_guidance(
    style: str,
    compression_marker: str,
    answer_marker: str,
) -> None:
    bodies = {
        "numeric_reduce": "How many parts are in the inventory in total?",
        "temporal_timeline": "What is the order of the two inventory updates?",
        "state_chain": "What was the previous inventory before I updated it?",
        "set_join": "What are the two colors in the inventory?",
        "synthesize": "Can you suggest an inventory organization approach?",
        "direct_extract": "What color were the added parts?",
    }
    question = _question(question_text=bodies[style])
    route = route_question(question.dated_question)
    assert route.style.value == style
    compression = build_routed_fact_compression_prompt(question, route)
    answer = build_routed_answer_prompt(question, _valid_response(), route)

    assert compression_marker in compression.messages[0].content
    assert answer_marker in answer.messages[-1].content
    assert "source-completeness label" in compression.messages[0].content
    assert "source-completeness label" in answer.messages[-1].content
    assert answer.prompt.prompt_token_proxy + answer.prompt.responder_output_token_reserve <= 8_000


def test_default_measured_arm_is_facts_only_and_hash_bound() -> None:
    question = _question()

    result = build_routed_answer_prompt(
        question,
        _valid_response(),
        route_question(question.dated_question),
    )

    assert DEFAULT_MEASURED_ARM == "facts"
    assert result.requested_arm == "facts"
    assert result.effective_arm == "facts"
    assert not result.used_raw_s1_fallback
    assert result.fallback_reason is None
    assert result.compression_receipt_sha256
    memory = result.messages[1].content
    assert "4 red parts were added Tuesday." in memory
    assert "Unrelated workshop note." not in memory
    assert result.messages_sha256 == result.prompt.messages_sha256
    assert result.receipt_sha256

    with pytest.raises(RoutedRepairPromptError, match="receipt does not match"):
        type(result)(
            question_id=result.question_id,
            style=result.style,
            route_receipt_sha256=result.route_receipt_sha256,
            requested_arm=result.requested_arm,
            effective_arm=result.effective_arm,
            used_raw_s1_fallback=result.used_raw_s1_fallback,
            fallback_reason=result.fallback_reason,
            compression_response_sha256=result.compression_response_sha256,
            compression_receipt_sha256=result.compression_receipt_sha256,
            prompt=result.prompt,
            receipt_sha256="0" * 64,
        )


def test_numeric_fact_rejects_an_operand_absent_from_its_exact_quotes() -> None:
    question = _question()
    invented = (
        '{"facts":[{"text":"5 red parts were added Tuesday.",'
        '"citations":[{"evidence_alias":"E001",'
        '"quote":"4 red parts were added"}]}]}'
    )

    result = build_routed_answer_prompt(
        question, invented, route_question(question.dated_question)
    )

    assert result.fallback_reason == "unsupported_numeric_fact"
    assert result.compression_receipt_sha256 is None


def test_numeric_literal_grounding_is_specific_to_numeric_route() -> None:
    question = _question()
    invented = (
        '{"facts":[{"text":"5 red parts were added Tuesday.",'
        '"citations":[{"evidence_alias":"E001",'
        '"quote":"4 red parts were added"}]}]}'
    )

    direct = _question(question_text="What color were the added parts?")
    result = build_routed_answer_prompt(
        direct, invented, route_question(direct.dated_question)
    )

    assert result.fallback_reason is None
    assert result.effective_arm == "facts"


@pytest.mark.parametrize(
    ("fact_text", "quote", "expected"),
    (
        ("The package weighed 999kg.", "The package weighed 2kg.", False),
        ("There were two widgets.", "There were three widgets.", False),
        ("The package weighed 2kg.", "There were 2 widgets.", False),
        ("The adjustment was -5kg.", "The adjustment was 5kg.", False),
        ("The adjustment was -5kg.", "The adjustment was -5kg.", True),
        ("The discount was $20.", "The discount was 20 dollars.", False),
        ("The discount was $20.", "The discount was $20.", True),
        ("The rate was 20%.", "The rate was 20 percent.", False),
        ("The rate was 20%.", "The rate was 20%.", True),
        ("It finished in 2nd place.", "It finished in 2nd place.", True),
    ),
)
def test_numeric_grounding_requires_exact_value_and_unit_span(
    fact_text: str,
    quote: str,
    expected: bool,
) -> None:
    fact = SimpleNamespace(
        text=fact_text,
        citations=(SimpleNamespace(quote=quote),),
    )

    assert numeric_facts_are_quote_grounded((fact,)) is expected


@pytest.mark.parametrize(
    ("response", "reason"),
    (
        ("not json", "invalid_compression"),
        ('{"facts":[]}', "empty_compression"),
        (
            '{"facts":[{"text":"Invented.","citations":['
            '{"evidence_alias":"E999","quote":"invented"}]}]}',
            "invalid_compression",
        ),
    ),
)
def test_invalid_or_empty_compression_requires_sealed_baseline_fallback(
    response: str,
    reason: str,
) -> None:
    question = _question()

    route = route_question(question.dated_question)
    first = build_routed_answer_prompt(question, response, route)
    replay = build_routed_answer_prompt(question, response, route)

    assert not first.used_raw_s1_fallback
    assert first.fallback_reason == reason
    assert first.requested_arm == "facts"
    assert first.effective_arm == "facts"
    assert first.prompt.arm == "facts"
    assert first.receipt_sha256 == replay.receipt_sha256
    assert first.messages_sha256 == replay.messages_sha256
    memory = first.messages[1].content
    assert "The initial inventory contained 3 blue parts." in memory
    assert "4 red parts were added" not in memory
    assert "Unrelated workshop note." not in memory
    assert "Episodic neighborhood payload" not in memory
    assert all(
        row.content.strip().casefold().rstrip(".") != "i don't know"
        for row in first.messages
    )


def test_baseline_fallback_diagnostic_is_deterministic_and_bounded() -> None:
    question = _question(noisy=True)

    first = build_routed_answer_prompt(
        question,
        '{"facts":[]}',
        route_question(question.dated_question),
        max_prompt_tokens=900,
        responder_output_token_reserve=64,
    )
    replay = build_routed_answer_prompt(
        question,
        '{"facts":[]}',
        route_question(question.dated_question),
        max_prompt_tokens=900,
        responder_output_token_reserve=64,
    )

    assert first.prompt.prompt_token_proxy + 64 <= 900
    assert first.messages_sha256 == replay.messages_sha256
    assert first.prompt.selected_neighborhood_evidence_ids == ()
    assert first.prompt.dropped_neighborhood_evidence_ids == (
        "addition",
        "noise",
    )


def test_rejects_unknown_styles_and_caps_above_8000() -> None:
    question = _question()
    with pytest.raises(RoutedRepairPromptError, match="question-bound"):
        build_routed_fact_compression_prompt(question, "oracle_route")
    with pytest.raises(RoutedRepairPromptError, match="1 through 8000"):
        build_routed_fact_compression_prompt(
            question,
            route_question(question.dated_question),
            max_prompt_tokens=8_001,
        )


def test_provider_prompt_rejects_a_route_receipt_from_another_question() -> None:
    question = _question()
    other = _question(question_text="What color were the added parts?")

    with pytest.raises(RoutedRepairPromptError, match="another question"):
        build_routed_fact_compression_prompt(
            question,
            route_question(other.dated_question),
        )
