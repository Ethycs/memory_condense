from __future__ import annotations

import inspect

from tools.matched_eval.typed_operator_spec import (
    AnswerShape,
    SlotKind,
    TemporalMode,
    compile_typed_operator_spec,
)


Q16 = """[Question asked at 2023/10/15 (Sun) 08:39]
How long have I been living in my current apartment in Harajuku?"""
Q50 = """[Question asked at 2023/05/30 (Tue) 22:43]
Which social media platform did I gain the most followers on over the past month?"""
Q72 = """[Question asked at 2023/05/30 (Tue) 14:24]
How many plants did I initially plant for tomatoes and chili peppers?"""
Q77 = """[Question asked at 2023/03/25 (Sat) 17:18]
How many months have passed since I last visited a museum with a friend?"""
Q79 = """[Question asked at 2023/05/30 (Tue) 23:20]
How much did I spend on a designer handbag?"""
Q75 = """[Question asked at 2023/05/25 (Thu) 15:52]
How much more did I spend on accommodations per night in Hawaii compared to Tokyo?"""
Q97 = """[Question asked at 2023/05/30 (Tue) 16:15]
Did I receive a higher percentage discount on my first order from HelloFresh, compared to my first UberEats order?"""


def test_question_only_api_cannot_accept_posthoc_fields() -> None:
    signature = inspect.signature(compile_typed_operator_spec)
    assert tuple(signature.parameters) == ("question",)
    assert signature.parameters["question"].kind is inspect.Parameter.POSITIONAL_ONLY


def test_q72_has_two_numeric_operand_slots_and_generic_subject_pair_variant() -> None:
    spec = compile_typed_operator_spec(Q72)
    operands = tuple(row for row in spec.required_slots if row.kind is SlotKind.OPERAND)
    assert tuple(row.match_terms for row in operands) == (("tomato",), ("chili", "pepper"))
    assert all(row.requires_numeric for row in operands)

    variant = compile_typed_operator_spec(
        "[Question asked at 2023/05/30 (Tue) 14:24]\n"
        "How many tomato and chili plants did I initially plant?"
    )
    variant_operands = tuple(
        row for row in variant.required_slots if row.kind is SlotKind.OPERAND
    )
    assert tuple(row.match_terms for row in variant_operands) == (
        ("tomato",),
        ("chili",),
    )


def test_q77_has_event_boundary_participant_and_implicit_query_time_end() -> None:
    spec = compile_typed_operator_spec(Q77)
    assert spec.temporal_mode is TemporalMode.INTERVAL
    assert spec.query_timestamp == "2023/03/25 (Sat) 17:18"
    assert any(
        row.kind is SlotKind.TEMPORAL_BOUNDARY
        and row.relation_constraint == "implicit_query_time_end"
        and {"visit", "museum"} <= set(row.match_terms)
        for row in spec.required_slots
    )
    assert any(
        row.kind is SlotKind.PARTICIPANT
        and row.relation_constraint == "participant_singular"
        for row in spec.required_slots
    )


def test_q16_has_residence_state_and_interval_boundary_slots() -> None:
    spec = compile_typed_operator_spec(Q16)
    assert spec.temporal_mode is TemporalMode.INTERVAL
    assert any(
        row.kind is SlotKind.PREDICATE
        and row.relation_constraint == "state_entity"
        and {"apartment", "harajuku"} <= set(row.match_terms)
        for row in spec.required_slots
    )
    assert any(row.kind is SlotKind.TEMPORAL_BOUNDARY for row in spec.required_slots)


def test_q50_and_q79_compile_entity_and_latest_state_shapes() -> None:
    q50 = compile_typed_operator_spec(Q50)
    assert q50.answer_shape is AnswerShape.DIRECT
    assert q50.comparison_mode.value == "max_entity"

    q79 = compile_typed_operator_spec(Q79)
    assert q79.answer_shape is AnswerShape.DIRECT
    assert q79.temporal_mode is TemporalMode.LATEST_STATE
    assert q79.requires_complete_frontier is True
    assert any(
        row.relation_constraint == "latest_completed_transaction"
        and row.requires_numeric
        for row in q79.required_slots
    )


def test_q75_and_q97_comparison_sides_use_nearest_named_entities() -> None:
    q75 = tuple(
        row
        for row in compile_typed_operator_spec(Q75).required_slots
        if row.kind is SlotKind.COMPARISON_SIDE
    )
    assert tuple(row.label for row in q75) == ("Hawaii", "Tokyo")
    assert tuple(row.match_terms for row in q75) == (("hawaii",), ("tokyo",))

    q97 = tuple(
        row
        for row in compile_typed_operator_spec(Q97).required_slots
        if row.kind is SlotKind.COMPARISON_SIDE
    )
    assert tuple(row.label for row in q97) == ("HelloFresh", "UberEats")
    assert tuple(row.match_terms for row in q97) == (
        ("hellofresh",),
        ("ubereat",),
    )


def test_projection_is_zero_state_and_contains_no_posthoc_target_fields() -> None:
    projection = compile_typed_operator_spec(Q72).projection()
    assert projection["retained_transformer_token_state_bytes"] == 0
    rendered = repr(projection).casefold()
    for forbidden in ("reference", "gold", "correct", "target_owner"):
        assert forbidden not in rendered
