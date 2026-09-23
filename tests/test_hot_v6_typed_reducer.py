from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools.matched_eval import hot_v6_typed_reducer as reducer_module
from tools.matched_eval.contracts import assert_gold_blind, identity_sha256
from tools.matched_eval.hot_v6_query_fact_ledger import (
    compile_query_fact_ledger,
    select_and_render_query_fact_ledger,
)
from tools.matched_eval.hot_v6_typed_reducer import (
    HotV6TypedReducerError,
    build_hot_v6_relevant_frontier,
    build_hot_v6_typed_reducer_input,
    reduce_hot_v6_typed_ledger,
)
from tools.matched_eval.numeric_evidence_reconciler import (
    NumericEvidenceReconcilerError,
)


def _row(
    evidence_id: str,
    text: str,
    *,
    source_id: str = "source-a",
    created_at: str = "2023-05-30T12:00:00Z",
    role: str = "user",
) -> dict[str, Any]:
    return {
        "chunk_id": evidence_id,
        "created_at": created_at,
        "evidence_id": evidence_id,
        "raw_text": text,
        "raw_text_sha256": quote_sha256(text),
        "role": role,
        "source_id": source_id,
    }


def _input(question: str, rows: list[dict[str, Any]]):
    ledger = compile_query_fact_ledger(question, rows)
    reducer_input = build_hot_v6_typed_reducer_input(
        question,
        ledger,
        represented_backing_evidence_ids=[row["evidence_id"] for row in rows],
    )
    return ledger, reducer_input


def test_adapter_is_provenance_agnostic_and_requires_every_raw_backing_row() -> None:
    question = "[Question asked at 2023/05/30] What did I buy?"
    global_row = _row("G-raw", "I bought a bicycle.", source_id="global-source")
    episode_row = _row("E-raw", "I bought a helmet.", source_id="episode-source")
    ledger = compile_query_fact_ledger(question, [global_row, episode_row])

    with pytest.raises(
        HotV6TypedReducerError,
        match="ledger fact backing evidence is not rendered: E-raw",
    ):
        build_hot_v6_typed_reducer_input(
            question,
            ledger,
            represented_backing_evidence_ids=["G-raw"],
        )

    adapted = build_hot_v6_typed_reducer_input(
        question,
        ledger,
        represented_backing_evidence_ids=["G-raw", "E-raw"],
    )
    assert {row.backing_evidence_id for row in adapted.handle_bindings} == {
        "G-raw",
        "E-raw",
    }
    assert adapted.provider_input["typed_evidence"]["frontier"]["closed"] is False
    assert "G-raw" not in adapted.provider_input_json
    assert "E-raw" not in adapted.provider_input_json
    assert adapted.provider_calls == 0
    assert_gold_blind(adapted.projection())


def test_adapter_uses_only_final_rendered_fact_slice_not_omitted_candidates() -> None:
    question = "[Question asked at 2023/05/30] What did I buy?"
    rows = [
        _row("G-raw", "I bought a bicycle.", source_id="global-source"),
        _row("E-kept", "I bought a helmet.", source_id="episode-source"),
        _row("E-omitted", "I bought a bell.", source_id="episode-source"),
    ]
    ledger = compile_query_fact_ledger(question, rows)
    fact_slice = select_and_render_query_fact_ledger(
        ledger,
        max_facts=1,
        max_tokens=2_048,
    )
    assert len(fact_slice.facts) == 1
    represented = [fact_slice.facts[0].backing_evidence_id]

    adapted = build_hot_v6_typed_reducer_input(
        question,
        ledger,
        represented_backing_evidence_ids=represented,
        fact_slice=fact_slice,
    )

    assert adapted.fact_population_receipt_sha256 == fact_slice.receipt_sha256
    assert {row.backing_evidence_id for row in adapted.handle_bindings} == set(
        represented
    )
    assert not (
        {"G-raw", "E-kept", "E-omitted"} - set(represented)
    ) & {row.backing_evidence_id for row in adapted.handle_bindings}
    assert len(adapted.item_bindings) == 1


def test_direct_duration_uses_exact_statement_instead_of_aging_it() -> None:
    question = (
        "[Question asked at 2023/05/30 (Tue) 21:43]\n"
        "How long have I been collecting vintage cameras?"
    )
    rows = [
        _row(
            "camera-duration",
            "I've been collecting vintage cameras for three months now.",
            created_at="2023-05-21T10:00:00Z",
        )
    ]
    _ledger, adapted = _input(question, rows)

    result = reduce_hot_v6_typed_ledger(adapted)

    assert result.status == "supported"
    assert result.prediction == "3 months"
    assert result.reason == "direct_duration"
    assert result.used_backing_evidence_ids == ("camera-duration",)


def test_relative_offsets_compute_binary_order_without_gold() -> None:
    question = (
        "[Question asked at 2023/05/25 (Thu) 11:29]\n"
        "Which device did I set up first, the smart thermostat or the mesh "
        "network system?"
    )
    rows = [
        _row(
            "thermostat",
            "I set up the smart thermostat a month ago.",
            created_at="2023-05-25T09:00:00Z",
        ),
        _row(
            "mesh",
            "I set up the mesh network system three weeks ago.",
            created_at="2023-05-25T09:05:00Z",
        ),
    ]
    _ledger, adapted = _input(question, rows)

    result = reduce_hot_v6_typed_ledger(adapted)

    assert result.status == "supported"
    assert result.prediction == "the smart thermostat"
    assert result.reason == "event_order"
    assert set(result.used_backing_evidence_ids) == {"thermostat", "mesh"}


def test_interval_range_validates_one_week_candidate() -> None:
    question = (
        "[Question asked at 2023/05/26 (Fri) 17:49]\n"
        "How long had I been using the new area rug when I rearranged my "
        "living room furniture?"
    )
    rows = [
        _row(
            "rug",
            "I got a new area rug for my living room a month ago.",
            created_at="2023-05-26T10:00:00Z",
        ),
        _row(
            "furniture",
            "I rearranged my living room furniture three weeks ago.",
            created_at="2023-05-26T10:05:00Z",
        ),
    ]
    _ledger, adapted = _input(question, rows)

    result = reduce_hot_v6_typed_ledger(
        adapted,
        candidate_prediction="About one week.",
    )

    assert result.status == "supported"
    assert result.prediction == "About one week."
    assert result.reason == "event_interval"
    assert result.proof["proof"]["computed"]["duration_days_min"] <= 7
    assert result.proof["proof"]["computed"]["duration_days_max"] >= 7


def test_fixed_two_side_percentage_comparison_needs_no_global_frontier() -> None:
    question = (
        "[Question asked at 2023/05/30 (Tue) 21:33]\n"
        "Did I receive a higher percentage discount on my first order from "
        "HelloFresh, compared to my first UberEats order?"
    )
    rows = [
        _row("hello-fresh", "My first HelloFresh order had a 40% discount."),
        _row("uber-eats", "My first UberEats order had a 20% discount."),
    ]
    _ledger, adapted = _input(question, rows)

    result = reduce_hot_v6_typed_ledger(adapted)

    assert result.status == "supported"
    assert result.prediction == "Yes"
    assert result.reason == "fixed_arity_named_scalar_operands"
    assert set(result.used_backing_evidence_ids) == {"hello-fresh", "uber-eats"}


def test_set_count_abstains_until_independent_frontier_is_closed() -> None:
    question = (
        "[Question asked at 2023/03/03 (Fri) 23:25]\n"
        "How many different museums or galleries did I visit in the month of "
        "February?"
    )
    rows = [
        _row(
            "art-cube",
            "I visited The Art Cube on February 15, 2023.",
            created_at="2023-02-15T12:00:00Z",
        ),
        _row(
            "museum",
            "I visited the Natural History Museum on February 8, 2023.",
            created_at="2023-02-08T12:00:00Z",
        ),
    ]
    _ledger, adapted = _input(question, rows)

    open_result = reduce_hot_v6_typed_ledger(adapted)
    assert open_result.status == "insufficient"
    assert "relevant_candidate_frontier_not_closed" in open_result.reason

    frontier = build_hot_v6_relevant_frontier(
        adapted,
        candidate_population_receipt_sha256=identity_sha256(
            {"scope": "complete February museum section", "rows": 2}
        ),
    )
    closed_result = reduce_hot_v6_typed_ledger(
        adapted,
        relevant_frontier=frontier,
    )

    assert closed_result.status == "supported"
    assert closed_result.prediction == "2"
    assert closed_result.reason == "operator_relevant_candidate_reduction"
    assert set(closed_result.used_backing_evidence_ids) == {"art-cube", "museum"}


def test_open_frontier_certificate_cannot_authorize_a_count() -> None:
    question = (
        "[Question asked at 2023/05/30]\n"
        "How many plants did I acquire in the last month?"
    )
    rows = [
        _row("lily", "I acquired a peace lily last week."),
        _row("succulent", "I bought a succulent yesterday."),
    ]
    _ledger, adapted = _input(question, rows)
    frontier = build_hot_v6_relevant_frontier(
        adapted,
        candidate_population_receipt_sha256=identity_sha256(
            {"scope": "plant section", "rows": 2}
        ),
        unresolved_candidate_keys=["unscanned:plant-section-2"],
    )

    result = reduce_hot_v6_typed_ledger(
        adapted,
        relevant_frontier=frontier,
    )

    assert frontier.closed is False
    assert result.status == "insufficient"
    assert "relevant_candidate_frontier_not_closed" in result.reason


def test_multi_slot_operator_is_normalized_for_legacy_numeric_reconciler() -> None:
    question = (
        "[Question asked at 2023/05/30]\n"
        "How many plants and flowers did I buy?"
    )
    rows = [_row("plants", "I bought two plants last week.")]
    _ledger, adapted = _input(question, rows)

    slots = adapted.provider_input["typed_evidence"]["operator_spec"][
        "required_slots"
    ]
    assert len(slots) == 2
    assert all(
        set(slot)
        == {
            "kind",
            "label",
            "match_terms",
            "minimum_match_term_count",
            "relation_constraint",
            "requires_numeric",
            "slot_id",
        }
        for slot in slots
    )

    result = reduce_hot_v6_typed_ledger(adapted)

    assert result.status == "insufficient"
    assert "numeric_reconciler_contract_non_applicable" not in result.reason


def test_known_numeric_contract_refusal_becomes_sealed_insufficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = (
        "[Question asked at 2023/05/30]\n"
        "How many plants and flowers did I buy?"
    )
    _ledger, adapted = _input(
        question,
        [_row("plants", "I bought two plants last week.")],
    )

    def reject_known_schema(*_args: object, **_kwargs: object):
        raise NumericEvidenceReconcilerError("unsupported numeric domain schema")

    monkeypatch.setattr(
        reducer_module,
        "reconcile_sealed_numeric_evidence_v2",
        reject_known_schema,
    )
    result = reduce_hot_v6_typed_ledger(adapted)

    assert result.status == "insufficient"
    assert "numeric_reconciler_contract_non_applicable" in result.reason
    refusal = result.proof["numeric_reconciliation"]
    assert refusal["contract_error_type"] == "NumericEvidenceReconcilerError"
    assert refusal["contract_error_message_sha256"] == quote_sha256(
        "unsupported numeric domain schema"
    )
    assert_gold_blind(result.projection())


def test_arbitrary_numeric_failure_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = (
        "[Question asked at 2023/05/30]\n"
        "How many plants and flowers did I buy?"
    )
    _ledger, adapted = _input(
        question,
        [_row("plants", "I bought two plants last week.")],
    )

    def fail_arbitrarily(*_args: object, **_kwargs: object):
        raise RuntimeError("programming failure")

    monkeypatch.setattr(
        reducer_module,
        "reconcile_sealed_numeric_evidence_v2",
        fail_arbitrarily,
    )
    with pytest.raises(RuntimeError, match="programming failure"):
        reduce_hot_v6_typed_ledger(adapted)


def test_reducer_receipts_are_tamper_evident() -> None:
    question = (
        "[Question asked at 2023/05/30]\n"
        "How much higher was Hawaii compared to Tokyo?"
    )
    rows = [
        _row("hawaii", "Hawaii cost $320."),
        _row("tokyo", "Tokyo cost $50."),
    ]
    _ledger, adapted = _input(question, rows)
    result = reduce_hot_v6_typed_ledger(adapted)
    assert result.status == "supported"

    with pytest.raises(HotV6TypedReducerError, match="typed reduction changed"):
        replace(result, prediction="$999")
