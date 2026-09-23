from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256
from tools.matched_eval.contracts import assert_gold_blind
from tools.matched_eval.hot_v6_query_fact_ledger import (
    DEDUP_POLICY,
    DEFAULT_MAX_RENDERED_FACTS,
    DEFAULT_MAX_RENDERED_TOKENS,
    FactLedgerSlice,
    FactOrigin,
    FactStatus,
    QueryFactLedgerError,
    compile_query_fact_ledger,
    render_query_fact_ledger,
    select_and_render_query_fact_ledger,
)


def _row(
    evidence_id: str,
    text: str,
    *,
    source_id: str = "source-a",
    role: str = "user",
    created_at: str = "2026-09-01T12:00:00Z",
    exchange_id: str | None = None,
    envelope_id: str | None = None,
    user_lead_evidence_id: str | None = None,
) -> dict[str, Any]:
    return {
        "chunk_id": evidence_id,
        "created_at": created_at,
        "envelope_id": envelope_id,
        "evidence_id": evidence_id,
        "exchange_id": exchange_id,
        "raw_text": text,
        "raw_text_sha256": quote_sha256(text),
        "role": role,
        "source_id": source_id,
        "user_lead_evidence_id": user_lead_evidence_id,
    }


def test_compiles_question_typed_numeric_slots_with_exact_citations() -> None:
    question = (
        "[Question asked at 2026-09-07] How much higher were accommodations "
        "in Hawaii compared to Tokyo?"
    )
    selected = _row(
        "lead",
        "Compare the lodging costs for my trips.",
        exchange_id="exchange-lead",
    )
    hawaii = _row(
        "hawaii",
        "The Hawaii room was $320 per night.",
        exchange_id="exchange-hawaii",
        user_lead_evidence_id="hawaii",
    )
    tokyo = _row(
        "tokyo",
        "The Tokyo hotel cost $50 per night.",
        exchange_id="exchange-tokyo",
        user_lead_evidence_id="tokyo",
    )

    ledger = compile_query_fact_ledger(
        question,
        [selected],
        candidate_rows=[hawaii, tokyo],
    )

    by_evidence = {row.backing_evidence_id: row for row in ledger.facts}
    assert by_evidence["hawaii"].exact_quote == hawaii["raw_text"]
    assert by_evidence["tokyo"].exact_quote == tokyo["raw_text"]
    assert by_evidence["hawaii"].fact_id != "hawaii"
    assert by_evidence["tokyo"].fact_id != "tokyo"
    assert [row.value for row in by_evidence["hawaii"].numeric_operands] == [
        320.0
    ]
    assert [row.value for row in by_evidence["tokyo"].numeric_operands] == [50.0]
    assert {row.slot_label for row in ledger.slot_bindings} == {"Hawaii", "Tokyo"}
    assert ledger.unresolved_slot_ids == ()
    assert ledger.raw_rows_embedded is False
    assert ledger.dedup_policy == DEDUP_POLICY
    assert ledger.provider_calls == 0
    assert_gold_blind(ledger.projection())

    fact_slice = select_and_render_query_fact_ledger(ledger)
    rendered = render_query_fact_ledger(ledger)
    assert type(fact_slice) is FactLedgerSlice
    assert fact_slice.text == fact_slice.rendered_text == rendered
    assert fact_slice.fact_ids == tuple(row.fact_id for row in fact_slice.facts)
    assert fact_slice.ledger_fact_ids == tuple(row.fact_id for row in ledger.facts)
    assert fact_slice.ledger_receipt_sha256 == ledger.receipt_sha256
    assert fact_slice.rendered_token_count == count_tokens(rendered)
    assert fact_slice.rendered_text_sha256 == quote_sha256(rendered)
    assert fact_slice.provider_calls == 0
    assert_gold_blind(fact_slice.projection())
    assert "The Hawaii room was $320 per night." in rendered
    assert "backs=hawaii" in rendered
    assert "Hawaii:bound" in rendered
    assert "raw backing rows remain authoritative" in rendered


def test_candidate_seam_is_activated_source_scoped_and_user_spine_attached() -> None:
    question = "[Question asked at 2026-09-07] What lens was recommended?"
    lead = _row(
        "user-lead",
        "Can you recommend a camera lens?",
        exchange_id="exchange-camera",
        envelope_id="envelope-camera",
    )
    answer = _row(
        "assistant-answer",
        "I recommend the 70-200mm zoom lens.",
        role="assistant",
        exchange_id="exchange-camera",
        envelope_id="envelope-camera",
    )

    ledger = compile_query_fact_ledger(
        question,
        [lead],
        candidate_rows=[answer],
    )
    fact = next(
        row for row in ledger.facts if row.backing_evidence_id == "assistant-answer"
    )
    assert fact.source_role == "assistant"
    assert fact.user_lead_evidence_id == "user-lead"
    assert fact.exchange_id == "exchange-camera"
    assert fact.envelope_id == "envelope-camera"
    assert fact.selected_spine_affinity is True
    assert fact.origins == (FactOrigin.CANDIDATE,)

    outside = _row(
        "outside",
        "I recommend a prime lens.",
        source_id="source-not-activated",
        role="assistant",
        exchange_id="exchange-camera",
    )
    with pytest.raises(QueryFactLedgerError, match="sources activated"):
        compile_query_fact_ledger(question, [lead], candidate_rows=[outside])


def test_exact_id_dedup_happens_after_both_lanes_and_never_content_dedups() -> None:
    question = "[Question asked at 2026-09-07] What camera did I buy?"
    shared = _row(
        "same-id",
        "I bought a blue camera.",
        exchange_id="exchange-camera",
    )
    different_id = _row(
        "different-id",
        "I bought a blue camera.",
        exchange_id="exchange-camera",
        user_lead_evidence_id="same-id",
    )

    ledger = compile_query_fact_ledger(
        question,
        [shared, dict(shared)],
        candidate_rows=[dict(shared), different_id],
    )

    assert ledger.selected_evidence_ids == ("same-id", "same-id")
    assert ledger.candidate_evidence_ids == ("same-id", "different-id")
    assert ledger.selected_lane_fact_count == 2
    assert ledger.candidate_lane_fact_count == 2
    assert len(ledger.facts) == 2
    assert ledger.duplicate_fact_count == 2
    assert {row.backing_evidence_id for row in ledger.facts} == {
        "same-id",
        "different-id",
    }
    same = next(row for row in ledger.facts if row.backing_evidence_id == "same-id")
    assert same.origins == (FactOrigin.SELECTED, FactOrigin.CANDIDATE)


def test_status_time_and_role_are_metadata_not_silent_filters() -> None:
    question = (
        "[Question asked at 2026-09-07] How many clothing items did I pick up "
        "and return?"
    )
    rows = [
        _row(
            "completed",
            "I picked up one blazer yesterday.",
            exchange_id="exchange-completed",
        ),
        _row(
            "planned",
            "I plan to return two shirts tomorrow.",
            exchange_id="exchange-planned",
        ),
        _row(
            "failed",
            "I failed to pick up three coats last Friday.",
            exchange_id="exchange-failed",
        ),
    ]

    ledger = compile_query_fact_ledger(question, rows)
    by_evidence = {row.backing_evidence_id: row for row in ledger.facts}
    assert by_evidence["completed"].status is FactStatus.COMPLETED
    assert by_evidence["completed"].time_mentions == ("yesterday",)
    assert by_evidence["planned"].status is FactStatus.PLANNED
    assert by_evidence["planned"].time_mentions == ("tomorrow",)
    assert by_evidence["failed"].status is FactStatus.FAILED
    assert by_evidence["failed"].time_mentions == ("last Friday",)
    assert set(by_evidence) == {"completed", "planned", "failed"}


def test_receipts_and_runtime_firewalls_fail_closed() -> None:
    question = "[Question asked at 2026-09-07] What did I buy?"
    row = _row("purchase", "I bought a bicycle.")
    ledger = compile_query_fact_ledger(question, [row])

    with pytest.raises(QueryFactLedgerError):
        replace(ledger.facts[0], relevance_score=999)

    with_ordinal = {**row, "ordinal": 9}
    with pytest.raises(QueryFactLedgerError, match="ordinals"):
        compile_query_fact_ledger(question, [with_ordinal])

    with_category = {**row, "category": "lookup"}
    with pytest.raises(QueryFactLedgerError, match="gold-bearing"):
        compile_query_fact_ledger(question, [with_category])


def test_required_numeric_slot_stays_unresolved_without_a_numeric_operand() -> None:
    question = (
        "[Question asked at 2026-09-07] How much higher were accommodations "
        "in Hawaii compared to Tokyo?"
    )
    rows = [
        _row("hawaii", "The Hawaii room had an ocean view."),
        _row("tokyo", "The Tokyo hotel cost $50 per night."),
    ]

    ledger = compile_query_fact_ledger(question, rows)
    labels = {
        slot.slot_id: slot.label for slot in ledger.operator_spec.required_slots
    }
    assert {labels[slot_id] for slot_id in ledger.unresolved_slot_ids} == {"Hawaii"}
    assert {row.slot_label for row in ledger.slot_bindings} == {"Tokyo"}


def test_renderer_is_bounded_and_preserves_origin_and_slot_coverage() -> None:
    question = (
        "[Question asked at 2026-09-07] How much higher were accommodations "
        "in Hawaii compared to Tokyo?"
    )
    hawaii = _row(
        "hawaii",
        "The Hawaii room was $320 per night.",
        exchange_id="exchange-hawaii",
    )
    tokyo = _row(
        "tokyo",
        "The Tokyo hotel cost $50 per night.",
        exchange_id="exchange-tokyo",
    )
    noise = [
        _row(
            f"noise-{index}",
            f"Background note {index} about a separate topic with enough text.",
            exchange_id=f"exchange-noise-{index}",
        )
        for index in range(60)
    ]
    ledger = compile_query_fact_ledger(
        question,
        [hawaii],
        candidate_rows=[tokyo, *noise],
    )

    fact_slice = select_and_render_query_fact_ledger(ledger)
    rendered = render_query_fact_ledger(ledger)
    assert fact_slice.text == rendered
    emitted = [line for line in rendered.splitlines() if line.startswith("[F")]
    assert len(emitted) <= DEFAULT_MAX_RENDERED_FACTS
    assert count_tokens(rendered) <= DEFAULT_MAX_RENDERED_TOKENS
    assert "backs=hawaii" in rendered
    assert "backs=tokyo" in rendered
    assert "Hawaii:bound" in rendered and "Tokyo:bound" in rendered
    assert f"emitted_facts={len(emitted)}/{len(ledger.facts)}" in rendered
    assert fact_slice.fact_ids == tuple(row.fact_id for row in fact_slice.facts)
    assert set(fact_slice.mandatory_fact_ids) <= set(fact_slice.fact_ids)
    assert not (set(fact_slice.fact_ids) & set(fact_slice.omitted_fact_ids))
    assert (
        len(fact_slice.fact_ids) + len(fact_slice.omitted_fact_ids)
        == len(ledger.facts)
    )
    assert fact_slice.omitted_fact_ids
    assert fact_slice == select_and_render_query_fact_ledger(ledger)
    with pytest.raises(FrozenInstanceError):
        fact_slice.max_facts = 1  # type: ignore[misc]
    with pytest.raises(QueryFactLedgerError, match="rendered text digest"):
        replace(
            fact_slice,
            rendered_text="X" + fact_slice.rendered_text[1:],
        )
    with pytest.raises(QueryFactLedgerError, match="selected/omitted accounting"):
        replace(
            fact_slice,
            omitted_fact_ids=fact_slice.omitted_fact_ids[:-1],
        )
    with pytest.raises(QueryFactLedgerError, match="mandatory coverage became empty"):
        replace(fact_slice, mandatory_fact_ids=())
    with pytest.raises(QueryFactLedgerError, match="called a provider"):
        replace(fact_slice, provider_calls=False)

    exact_two = compile_query_fact_ledger(
        question,
        [hawaii],
        candidate_rows=[tokyo],
    )
    bounded_slice = select_and_render_query_fact_ledger(
        exact_two, max_facts=2, max_tokens=1_024
    )
    bounded = render_query_fact_ledger(exact_two, max_facts=2, max_tokens=1_024)
    assert bounded_slice.text == bounded
    assert bounded_slice.fact_ids == tuple(row.fact_id for row in exact_two.facts)
    assert bounded_slice.mandatory_fact_ids == bounded_slice.fact_ids
    assert bounded_slice.omitted_fact_ids == ()
    assert sum(line.startswith("[F") for line in bounded.splitlines()) == 2
    with pytest.raises(QueryFactLedgerError, match="cannot preserve"):
        select_and_render_query_fact_ledger(
            exact_two, max_facts=1, max_tokens=1_024
        )
    with pytest.raises(QueryFactLedgerError, match="cannot preserve"):
        select_and_render_query_fact_ledger(
            exact_two, max_facts=2, max_tokens=1
        )
    with pytest.raises(QueryFactLedgerError, match="cannot preserve"):
        render_query_fact_ledger(exact_two, max_facts=1, max_tokens=1_024)


def test_structured_slice_has_a_non_circular_rendering_golden() -> None:
    question = "[Question asked at 2026-09-07] What did I buy?"
    ledger = compile_query_fact_ledger(
        question,
        [_row("purchase", "I bought a bicycle.")],
    )

    fact_slice = select_and_render_query_fact_ledger(
        ledger,
        max_facts=1,
        max_tokens=1_024,
    )

    expected = "\n".join(
        (
            "<QUERY_FACT_LEDGER>",
            "operation=single_supported_fact; shape=direct; temporal=none; comparison=none",
            (
                "authority=raw backing rows remain authoritative and separately "
                "selectable; this selected/activated-source ledger cannot prove absence"
            ),
            "emitted_facts=1/1; bounds=max_facts:1,max_tokens:1024",
            (
                "[F1 fact=4a9ad3e7ff6a8c765e15377bcccdd295408475fc7f144d7c2c84c5275d9fb2c7 "
                "backs=purchase source=source-a] role=user; status=completed; "
                "time_basis=source_created_at_fallback; "
                "source_time=2026-09-01T12:00:00Z; actions=acquire; "
                'quote="I bought a bicycle."'
            ),
            (
                "ledger_receipt="
                "7fcdc4d008241e8ecbf0a608a6e059582069f3010f956719ffc6f5d57ea79cf6"
            ),
            "</QUERY_FACT_LEDGER>",
        )
    )
    assert fact_slice.text == expected
    assert (
        fact_slice.rendered_text_sha256
        == "2440b3f6a1874e3f3df4b810f6afdead9cae9fb9285da71cda96d53851b7660b"
    )
