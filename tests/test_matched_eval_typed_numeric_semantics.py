from __future__ import annotations

from tools.matched_eval.typed_numeric_semantics import (
    NumericDimension,
    NumericQualifier,
    expected_numeric_dimension,
    numeric_mentions,
)
from tools.matched_eval.typed_operator_spec import compile_typed_operator_spec


Q28 = """[Question asked at 2023/03/20 (Mon) 23:57]
How many bikes did I service or plan to service in March?"""
Q53 = """[Question asked at 2023/05/30 (Tue) 21:51]
How many plants did I acquire in the last month?"""
Q75 = """[Question asked at 2023/05/30 (Tue) 22:16]
How much more did I spend on accommodations per night in Hawaii compared to Tokyo?"""
Q97 = """[Question asked at 2023/05/30 (Tue) 16:15]
Did I receive a higher percentage discount on my first order from HelloFresh, compared to my first UberEats order?"""


def _mentions(question: str, evidence: str):
    return numeric_mentions(
        evidence,
        operator_spec=compile_typed_operator_spec(question),
        question=question,
    )


def test_q28_calendar_days_are_not_bike_count_operands() -> None:
    assert _mentions(Q28, "I serviced my road bike on March 2.") == ()
    assert _mentions(Q28, "I planned to service my mountain bike on March 10.") == ()
    assert [row.value for row in _mentions(
        Q28, "I serviced 1 road bike on March 2."
    )] == [1]
    assert [row.value for row in _mentions(
        Q28, "I planned to service 1 mountain bike on March 10."
    )] == [1]


def test_q53_duration_is_not_a_plant_count_operand() -> None:
    assert _mentions(
        Q53, "The aquarium plants need a 31-day treatment window."
    ) == ()


def test_q75_rank_is_rejected_and_price_qualifiers_are_preserved() -> None:
    assert expected_numeric_dimension(
        operator_spec=compile_typed_operator_spec(Q75), question=Q75
    ) is NumericDimension.CURRENCY
    assert _mentions(Q75, "Top 5 Hawaii experiences.") == ()

    hawaii = _mentions(Q75, "I spent over $300 per night in Hawaii.")
    tokyo = _mentions(
        Q75,
        "I stayed in a hostel in Tokyo that cost around $30 per night.",
    )
    assert [
        (row.value, row.dimension, row.qualifier, row.unit)
        for row in (*hawaii, *tokyo)
    ] == [
        (300, NumericDimension.CURRENCY, NumericQualifier.LOWER_BOUND, "$"),
        (30, NumericDimension.CURRENCY, NumericQualifier.APPROXIMATE, "$"),
    ]


def test_q31_q72_q97_compatible_numeric_dimensions_survive() -> None:
    q31 = """[Question asked at 2023/03/20 (Mon) 23:57]
How many pounds of pet food did I buy?"""
    q72 = """[Question asked at 2023/05/30 (Tue) 14:24]
How many plants did I initially plant for tomatoes and chili peppers?"""

    pounds = _mentions(q31, "I bought a 50-lb bag and 20 pounds of feed.")
    assert [(row.value, row.dimension, row.unit) for row in pounds] == [
        (50, NumericDimension.MEASURE, "lb"),
        (20, NumericDimension.MEASURE, "lb"),
    ]
    assert [row.value for row in _mentions(q72, "I initially planted 6 tomatoes.")] == [6]
    percent = _mentions(Q97, "My first HelloFresh order was 40 percent off.")
    assert [(row.value, row.dimension, row.unit) for row in percent] == [
        (40, NumericDimension.PERCENTAGE, "%")
    ]
