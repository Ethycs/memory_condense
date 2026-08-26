from __future__ import annotations

import hashlib
import inspect

import pytest

from tools._routed_repair_routing import (
    ROUTED_REPAIR_ROUTING_FORMAT,
    RoutedRepairReason,
    RoutedRepairReceipt,
    RoutedRepairStyle,
    route_question,
)


@pytest.mark.parametrize(
    ("question", "style", "reason"),
    (
        (
            "How many weeks passed between the day I bought my racket and the day I received it?",
            RoutedRepairStyle.TIMELINE,
            RoutedRepairReason.TEMPORAL_INTERVAL,
        ),
        (
            "How many days ago did I go whitewater rafting?",
            RoutedRepairStyle.TIMELINE,
            RoutedRepairReason.TEMPORAL_INTERVAL,
        ),
        (
            "What is the order of the three trips, from earliest to latest?",
            RoutedRepairStyle.TIMELINE,
            RoutedRepairReason.TEMPORAL_ORDER,
        ),
        (
            "Which device did I set up first, the thermostat or the mesh network?",
            RoutedRepairStyle.TIMELINE,
            RoutedRepairReason.TEMPORAL_ORDER,
        ),
        (
            "What is the artist that I started listening to last Friday?",
            RoutedRepairStyle.TIMELINE,
            RoutedRepairReason.RELATIVE_TIME_LOOKUP,
        ),
        (
            "How many total pieces have I written since I started again three weeks ago?",
            RoutedRepairStyle.NUMERIC_REDUCE,
            RoutedRepairReason.NUMERIC_AGGREGATE,
        ),
        (
            "How much more did I spend in Hawaii compared to Tokyo?",
            RoutedRepairStyle.NUMERIC_REDUCE,
            RoutedRepairReason.NUMERIC_COMPARISON,
        ),
        (
            "Which platform did I gain the most followers on?",
            RoutedRepairStyle.NUMERIC_REDUCE,
            RoutedRepairReason.NUMERIC_COMPARISON,
        ),
        (
            "What are the two hobbies that led me to join communities?",
            RoutedRepairStyle.SET_JOIN,
            RoutedRepairReason.EXPLICIT_SET,
        ),
        (
            "What was my previous goal before I updated it?",
            RoutedRepairStyle.STATE_CHAIN,
            RoutedRepairReason.STATE_RESOLUTION,
        ),
        (
            "Can you suggest accessories for my current photography setup?",
            RoutedRepairStyle.SYNTHESIZE,
            RoutedRepairReason.CURRENT_SYNTHESIS_REQUEST,
        ),
        (
            "What type of bulb did I replace?",
            RoutedRepairStyle.EXTRACT,
            RoutedRepairReason.DIRECT_FALLBACK,
        ),
    ),
)
def test_question_only_styles_have_deterministic_precedence(question, style, reason):
    receipt = route_question(question)

    assert receipt.style is style
    assert receipt.reason is reason


def test_temporal_interval_precedes_count_and_aggregate_precedes_relative_phrase():
    interval = route_question("How many days passed between the two events?")
    aggregate = route_question(
        "How many total stories did I write since restarting three weeks ago?"
    )

    assert interval.reason is RoutedRepairReason.TEMPORAL_INTERVAL
    assert aggregate.reason is RoutedRepairReason.NUMERIC_AGGREGATE


def test_current_synthesis_precedes_state_but_retrospective_recommendation_does_not():
    current = route_question("Can you suggest accessories for my current setup?")
    retrospective = route_question(
        "Can you remind me what type of beer you specifically recommended?"
    )

    assert current.style is RoutedRepairStyle.SYNTHESIZE
    assert retrospective.style is RoutedRepairStyle.EXTRACT
    assert retrospective.modifiers.retrospective is True
    assert retrospective.modifiers.required_evidence_role == "assistant"
    assert retrospective.modifiers.required_evidence_role_basis == (
        "explicit_retrospective_assistant_attribution"
    )


@pytest.mark.parametrize(
    "question",
    (
        "How much RAM did I upgrade my laptop to?",
        "How much did I spend on a designer handbag?",
    ),
)
def test_direct_how_much_values_are_not_mistaken_for_arithmetic(question):
    assert route_question(question).style is RoutedRepairStyle.EXTRACT


def test_current_first_person_possession_uses_state_not_set_count():
    receipt = route_question("How many followers do I currently have on Instagram now?")

    assert receipt.style is RoutedRepairStyle.STATE_CHAIN
    assert receipt.modifiers.requires_temporal_metadata is True
    assert receipt.modifiers.requires_complete_frontier is False


def test_explicit_set_and_ordinal_modifiers_are_retained():
    fixed = route_question("Name six museums I visited")
    ordinal = route_question(
        "Can you remind me what was the 7th job in the list you provided?"
    )

    assert fixed.style is RoutedRepairStyle.SET_JOIN
    assert fixed.modifiers.cardinality == 6
    assert fixed.modifiers.requires_complete_frontier is True
    assert ordinal.style is RoutedRepairStyle.EXTRACT
    assert ordinal.modifiers.ordinal == 7
    assert ordinal.modifiers.required_evidence_role == "assistant"


def test_dated_wrapper_changes_identity_not_style_and_retains_timestamp():
    body = "How much more did I spend in Hawaii compared to Tokyo?"
    dated = f"[Question asked at 2026/08/26 12:00] {body}"
    plain_receipt = route_question(body)
    dated_receipt = route_question(dated)

    assert dated_receipt.style is plain_receipt.style
    assert dated_receipt.reason is plain_receipt.reason
    assert dated_receipt.question_sha256 != plain_receipt.question_sha256
    assert dated_receipt.modifiers.query_timestamp == "2026/08/26 12:00"


@pytest.mark.parametrize(
    ("question", "style"),
    (
        ("How often do I see Dr. Johnson?", RoutedRepairStyle.EXTRACT),
        (
            "At which university did I present a poster for my undergrad project?",
            RoutedRepairStyle.EXTRACT,
        ),
        (
            "How many plants did I initially plant for tomatoes and chili peppers?",
            RoutedRepairStyle.NUMERIC_REDUCE,
        ),
    ),
)
def test_answer_insufficiency_is_never_inferred_from_question_form(question, style):
    receipt = route_question(question)

    assert receipt.style is style
    assert "insufficient" not in receipt.style.value
    assert "insufficient" not in receipt.reason.value


def test_public_api_accepts_exactly_one_positional_question():
    signature = inspect.signature(route_question)

    assert tuple(signature.parameters) == ("question",)
    assert signature.parameters["question"].kind is inspect.Parameter.POSITIONAL_ONLY
    with pytest.raises(TypeError):
        route_question("Where?", category="single-session-user")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        route_question(question="Where?")  # type: ignore[call-arg]


def test_receipt_is_content_addressed_and_deterministic():
    question = "What brand of shampoo do I currently use?"
    first = route_question(question)
    second = route_question(question)

    assert first == second
    assert first.question_sha256 == hashlib.sha256(question.encode()).hexdigest()
    assert first.identity_payload()["format"] == ROUTED_REPAIR_ROUTING_FORMAT
    assert len(first.receipt_sha256) == 64
    assert first.style is RoutedRepairStyle.STATE_CHAIN
    with pytest.raises(ValueError, match="digest does not match"):
        RoutedRepairReceipt(
            question_sha256=first.question_sha256,
            style=first.style,
            reason=first.reason,
            modifiers=first.modifiers,
            receipt_sha256="0" * 64,
        )


@pytest.mark.parametrize("question", ("", "   ", "[Question asked at now]   "))
def test_empty_question_or_body_is_rejected(question):
    with pytest.raises(ValueError):
        route_question(question)


def test_non_string_question_is_rejected():
    with pytest.raises(TypeError):
        route_question({"question": "Where?"})  # type: ignore[arg-type]
