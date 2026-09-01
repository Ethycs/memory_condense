from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.matched_eval import full_store_typed_adapter as adapter
from tools.matched_eval.typed_operator_spec import compile_typed_operator_spec


QUESTION = (
    "[Question asked at 2023/03/20 (Mon) 23:57]\n"
    "How many plants did I initially plant?"
)


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("I had 6 plants on 2023-03-01.", 6.0),
        ("On 2023-03-01 I had 6 plants.", 6.0),
        ("The note was recorded on 2023-03-01.", None),
        ("I serviced my road bike on March 2.", None),
        ("The aquarium plants need a 31-day treatment window.", None),
        ("Top 5 Hawaii experiences.", None),
    ),
)
def test_full_store_numeric_value_ignores_iso_date_fragments(
    text: str, expected: float | None
) -> None:
    assert adapter._number(text, numeric_slot_supported=True) == expected  # noqa: SLF001


@pytest.mark.parametrize(
    ("basis", "expected_authority"),
    (
        ("explicit_text_date", "explicit"),
        ("row_created_at", "derived"),
    ),
)
def test_full_store_date_authority_preserves_scanner_basis(
    basis: str, expected_authority: str
) -> None:
    spec = compile_typed_operator_spec(QUESTION)
    supported_slot = spec.required_slots[0]
    candidate = SimpleNamespace(
        quote="I initially planted 6 plants.",
        supported_slot_ids=(supported_slot.slot_id,),
        event_date="2023-03-01",
        event_date_basis=basis,
        role="support",
    )
    raw = adapter._raw_item(spec, candidate, "H500001")  # noqa: SLF001
    assert raw["date"] == "2023-03-01"
    assert raw["value_authority"] == expected_authority
    assert f"date_basis:{basis}" in raw["relation"]


def test_generic_predicate_slot_label_is_never_promoted_to_entity_or_group() -> None:
    question = (
        "[Question asked at 2023/03/20 (Mon) 23:57]\n"
        "How many clothing items do I need to pick up or return?"
    )
    spec = compile_typed_operator_spec(question)
    slot = spec.required_slots[0]
    assert "predicate" in slot.label.casefold()
    candidate = SimpleNamespace(
        quote="I still need to pick up my dry cleaning for a navy blue blazer.",
        supported_slot_ids=(slot.slot_id,),
        event_date="2023-03-18",
        event_date_basis="row_created_at",
        role="user",
    )
    raw = adapter._raw_item(spec, candidate, "H500001")  # noqa: SLF001
    assert "entity_key" not in raw
    assert "group_key" not in raw
    assert slot.slot_id in candidate.supported_slot_ids


def test_specific_comparison_side_label_can_bind_its_named_entity() -> None:
    question = (
        "[Question asked at 2023/05/30 (Tue) 16:15]\n"
        "Did I receive a higher percentage discount on my first order from "
        "HelloFresh, compared to my first UberEats order?"
    )
    spec = compile_typed_operator_spec(question)
    slot = next(row for row in spec.required_slots if row.label == "HelloFresh")
    candidate = SimpleNamespace(
        quote="My first HelloFresh order had a 40 percent discount.",
        supported_slot_ids=(slot.slot_id,),
        event_date=None,
        event_date_basis=None,
        role="user",
    )
    raw = adapter._raw_item(spec, candidate, "H500001")  # noqa: SLF001
    assert raw["entity_key"] == "HelloFresh"
    assert raw["group_key"] == "HelloFresh"


def test_q75_bound_adapter_rejects_rank_and_preserves_price_qualifiers() -> None:
    question = (
        "[Question asked at 2023/05/30 (Tue) 22:16]\n"
        "How much more did I spend on accommodations per night in Hawaii "
        "compared to Tokyo?"
    )
    spec = compile_typed_operator_spec(question)
    hawaii_slot = next(row for row in spec.required_slots if row.label == "Hawaii")

    def candidate(quote: str) -> SimpleNamespace:
        return SimpleNamespace(
            quote=quote,
            supported_slot_ids=(hawaii_slot.slot_id,),
            event_date=None,
            event_date_basis=None,
            role="user",
        )

    ranked = adapter._raw_item(  # noqa: SLF001
        spec,
        candidate("Top 5 Hawaii experiences."),
        "H500001",
        dated_question=question,
    )
    bounded = adapter._raw_item(  # noqa: SLF001
        spec,
        candidate("I spent over $300 per night in Hawaii."),
        "H500002",
        dated_question=question,
    )

    assert "numeric_value" not in ranked
    assert bounded["numeric_value"] == 300
    assert bounded["numeric_qualifier"] == "lower_bound"
    assert bounded["unit"] == "$"
