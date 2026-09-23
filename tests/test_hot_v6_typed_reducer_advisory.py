from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools.matched_eval.contracts import assert_gold_blind, identity_sha256
from tools.matched_eval.hot_v6_query_fact_ledger import (
    compile_query_fact_ledger,
    select_and_render_query_fact_ledger,
)
from tools.matched_eval.hot_v6_typed_reducer import (
    HotV6TypedReducerError,
    build_hot_v6_relevant_frontier,
    build_hot_v6_typed_reducer_input,
)
from tools.matched_eval.hot_v6_typed_reducer_advisory import (
    HotV6TypedReducerAdvisoryError,
    compile_hot_v6_typed_reducer_advisory,
)


def _row(
    evidence_id: str,
    text: str,
    *,
    created_at: str = "2023-05-30T12:00:00Z",
) -> dict[str, Any]:
    return {
        "chunk_id": evidence_id,
        "created_at": created_at,
        "evidence_id": evidence_id,
        "raw_text": text,
        "raw_text_sha256": quote_sha256(text),
        "role": "user",
        "source_id": "source-a",
    }


def _ledger_slice(question: str, rows: list[dict[str, Any]]):
    ledger = compile_query_fact_ledger(question, rows)
    fact_slice = select_and_render_query_fact_ledger(ledger)
    return ledger, fact_slice


def test_supported_reduction_emits_only_prediction_and_slice_local_facts() -> None:
    question = (
        "[Question asked at 2023/05/25 (Thu) 11:29]\n"
        "Which device did I set up first, the smart thermostat or the mesh "
        "network system?"
    )
    rows = [
        _row("raw-thermostat", "I set up the smart thermostat a month ago."),
        _row("raw-mesh", "I set up the mesh network system three weeks ago."),
    ]
    ledger, fact_slice = _ledger_slice(question, rows)

    result = compile_hot_v6_typed_reducer_advisory(
        question,
        ledger,
        fact_slice,
        represented_backing_evidence_ids=[row["evidence_id"] for row in rows],
    )

    assert result.reduction.status == "supported"
    assert result.emitted is True
    assert 'prediction="the smart thermostat"' in result.provider_advisory_text
    assert result.support_fact_labels == ("F1", "F2")
    assert "support=F1,F2" in result.provider_advisory_text
    assert tuple(row.local_label for row in result.local_fact_bindings) == (
        "F1",
        "F2",
    )
    for raw_id in (
        *[row["evidence_id"] for row in rows],
        *[fact.fact_id for fact in fact_slice.facts],
    ):
        assert raw_id not in result.provider_advisory_text
    assert result.provider_calls == 0
    assert_gold_blind(result.audit_projection)


def test_q25_bounded_interval_without_candidate_emits_no_advisory() -> None:
    question = (
        "[Question asked at 2023/05/26 (Fri) 17:49]\n"
        "How long had I been using the new area rug when I rearranged my "
        "living room furniture?"
    )
    rows = [
        _row(
            "raw-rug",
            "I got a new area rug for my living room a month ago.",
        ),
        _row(
            "raw-furniture",
            "I rearranged my living room furniture three weeks ago.",
        ),
    ]
    ledger, fact_slice = _ledger_slice(question, rows)

    result = compile_hot_v6_typed_reducer_advisory(
        question,
        ledger,
        fact_slice,
        represented_backing_evidence_ids=[row["evidence_id"] for row in rows],
    )

    assert result.reduction.status == "insufficient"
    assert result.provider_advisory_text == ""
    assert result.support_fact_labels == ()
    assert result.emitted is False


def test_count_remains_silent_without_independent_closed_frontier() -> None:
    question = (
        "[Question asked at 2023/03/03 (Fri) 23:25]\n"
        "How many different museums or galleries did I visit in February?"
    )
    rows = [
        _row(
            "raw-gallery",
            "I visited The Art Cube gallery on February 15, 2023.",
            created_at="2023-02-15T12:00:00Z",
        ),
        _row(
            "raw-museum",
            "I visited the Natural History Museum on February 8, 2023.",
            created_at="2023-02-08T12:00:00Z",
        ),
    ]
    represented = [row["evidence_id"] for row in rows]
    ledger, fact_slice = _ledger_slice(question, rows)

    open_result = compile_hot_v6_typed_reducer_advisory(
        question,
        ledger,
        fact_slice,
        represented_backing_evidence_ids=represented,
    )
    assert open_result.reduction.status == "insufficient"
    assert open_result.provider_advisory_text == ""

    independently_built_input = build_hot_v6_typed_reducer_input(
        question,
        ledger,
        represented_backing_evidence_ids=represented,
        fact_slice=fact_slice,
    )
    frontier = build_hot_v6_relevant_frontier(
        independently_built_input,
        candidate_population_receipt_sha256=identity_sha256(
            {"scope": "complete February museum section", "rows": 2}
        ),
    )
    closed_result = compile_hot_v6_typed_reducer_advisory(
        question,
        ledger,
        fact_slice,
        represented_backing_evidence_ids=represented,
        relevant_frontier=frontier,
    )

    assert frontier.closed is True
    assert closed_result.reduction.status == "supported"
    assert 'prediction="2"' in closed_result.provider_advisory_text
    assert closed_result.support_fact_labels == ("F1", "F2")


def test_missing_final_raw_backing_fails_before_advisory_compilation() -> None:
    question = "[Question asked at 2023/05/30] How much did the table cost?"
    rows = [
        _row("raw-table", "The table cost $800."),
        _row("raw-delivery", "Delivery cost $50."),
    ]
    ledger, fact_slice = _ledger_slice(question, rows)

    with pytest.raises(
        HotV6TypedReducerError,
        match="ledger fact backing evidence is not rendered: raw-delivery",
    ):
        compile_hot_v6_typed_reducer_advisory(
            question,
            ledger,
            fact_slice,
            represented_backing_evidence_ids=["raw-table"],
        )


def test_omitted_ledger_candidates_are_outside_the_final_slice_advisory() -> None:
    question = "[Question asked at 2023/05/30] What did I buy?"
    rows = [
        _row("raw-kept", "I bought a bicycle."),
        _row("raw-omitted-one", "I bought a helmet."),
        _row("raw-omitted-two", "I bought a bell."),
    ]
    ledger = compile_query_fact_ledger(question, rows)
    fact_slice = select_and_render_query_fact_ledger(
        ledger,
        max_facts=1,
        max_tokens=2_048,
    )
    represented = [fact_slice.facts[0].backing_evidence_id]

    result = compile_hot_v6_typed_reducer_advisory(
        question,
        ledger,
        fact_slice,
        represented_backing_evidence_ids=represented,
    )

    assert len(result.local_fact_bindings) == 1
    assert result.represented_backing_evidence_ids == tuple(represented)
    assert set(result.reducer_input.represented_backing_evidence_ids) == set(
        represented
    )
    assert {
        "raw-omitted-one",
        "raw-omitted-two",
    }.isdisjoint(
        row.backing_evidence_id for row in result.local_fact_bindings
    )


def test_advisory_receipt_rejects_text_tampering() -> None:
    question = (
        "[Question asked at 2023/05/30]\n"
        "How long have I been collecting vintage cameras?"
    )
    rows = [_row("raw-camera", "I have collected cameras for three months.")]
    ledger, fact_slice = _ledger_slice(question, rows)
    result = compile_hot_v6_typed_reducer_advisory(
        question,
        ledger,
        fact_slice,
        represented_backing_evidence_ids=["raw-camera"],
    )
    assert result.reduction.status == "supported"

    with pytest.raises(
        HotV6TypedReducerAdvisoryError,
        match="provider text digest changed",
    ):
        replace(result, provider_advisory_text="prediction=wrong")
