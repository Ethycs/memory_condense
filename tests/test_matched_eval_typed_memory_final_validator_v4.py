from __future__ import annotations

import hashlib
import json

from tools.matched_eval.typed_memory_final_arm import (
    fit_typed_final_prompt,
    parse_typed_final_completion,
)
from tools.matched_eval.typed_memory_final_validator_v4 import (
    parse_typed_final_completion_v4,
    upgrade_completion_validation_contract_v4,
)
from tools.matched_eval.typed_operator_adapter import (
    EvidenceHandleBinding,
    EvidenceOrigin,
    FrontierMode,
    ProvenanceGrade,
    build_typed_evidence_packet,
    parse_typed_items,
)
from tools.matched_eval.typed_operator_spec import compile_typed_operator_spec


Q40 = (
    "[Question asked at 2023/05/30 (Tue) 15:43]\n"
    "How many pieces of jewelry did I acquire in the last two months?"
)
Q54 = (
    "[Question asked at 2023/03/25 (Sat) 18:26]\n"
    "What kitchen appliance did I buy 10 days ago?"
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fitted(
    question: str,
    raw_items: list[dict[str, object]],
    *,
    parent: str,
    frontier: FrontierMode = FrontierMode.BOUNDED,
):
    spec = compile_typed_operator_spec(question)
    handles = tuple(
        dict.fromkeys(
            handle
            for row in raw_items
            for handle in row["handle_ids"]  # type: ignore[index]
        )
    )
    bindings = tuple(
        EvidenceHandleBinding(
            handle,
            EvidenceOrigin.MAP,
            ProvenanceGrade.EXACT_CITATION,
            f"G{index:03d}",
            _sha("sealed-source"),
            _sha("parent"),
            _sha(f"evidence-{index}"),
            _sha(f"payload-{index}"),
            _sha(f"citation-{index}"),
            index,
            _sha(f"private-source-{index}"),
        )
        for index, handle in enumerate(handles, start=1)
    )
    parsed = parse_typed_items(
        raw_items, operator_spec=spec, bindings=bindings
    )
    assert not parsed.rejected_items
    packet = build_typed_evidence_packet(
        spec,
        bindings,
        parsed,
        sealed_input_artifact_sha256s=(_sha("sealed-source"),),
        frontier_mode=frontier,
        output_token_reserve=1,
    )
    return fit_typed_final_prompt(
        dated_question=question,
        parent_prediction=parent,
        packet=packet,
        mechanism_by_handle={handle: "test_map" for handle in handles},
    )


def _completion(prediction: str, handles: tuple[str, ...]) -> str:
    return json.dumps(
        {
            "decision": "replace",
            "prediction": prediction,
            "used_handle_ids": list(handles),
        },
        separators=(",", ":"),
    )


def _parse_v3(fitted, completion: str, contract: dict[str, object]):
    return parse_typed_final_completion(
        completion,
        parent_prediction=fitted.provider_input["protected_parent_fallback"][
            "prediction"
        ],
        allowed_handle_ids=fitted.allowed_handle_ids,
        handle_group_by_id=fitted.handle_group_by_id,
        story_coherence=fitted.story_coherence,
        preservation_requirements=fitted.preservation_requirements,
        validation_contract=contract,
    )


def _parse_v4(fitted, question: str, completion: str, legacy: dict[str, object]):
    contract = upgrade_completion_validation_contract_v4(
        legacy, dated_question=question
    )
    return parse_typed_final_completion_v4(
        completion,
        dated_question=question,
        parent_prediction=fitted.provider_input["protected_parent_fallback"][
            "prediction"
        ],
        allowed_handle_ids=fitted.allowed_handle_ids,
        handle_group_by_id=fitted.handle_group_by_id,
        story_coherence=fitted.story_coherence,
        preservation_requirements=fitted.preservation_requirements,
        validation_contract=contract,
    )


def _q40_fitted():
    return _fitted(
        Q40,
        [
            {
                "handle_ids": ["H001"],
                "kind": "direct",
                "summary": "I acquired a silver necklace for my jewelry collection.",
                "date": "2023-05-15",
                "status": "completed",
            },
            {
                "handle_ids": ["H002"],
                "kind": "direct",
                "summary": "I acquired a gold engagement ring for my jewelry collection.",
                "date": "2023-05-20",
                "status": "completed",
            },
            {
                "handle_ids": ["H003"],
                "kind": "direct",
                "summary": "I acquired emerald earrings for my jewelry collection.",
                "date": "2023-05-21",
                "status": "completed",
            },
            {
                "handle_ids": ["H004"],
                "kind": "direct",
                "summary": "I acquired a laptop for work.",
                "date": "2023-05-25",
                "status": "completed",
            },
            {
                "handle_ids": ["H005"],
                "kind": "direct",
                "summary": (
                    "I acquired my grandmother's pearl earrings on February "
                    "20th and discussed jewelry again later."
                ),
                # The source turn is in-window, but the event stated inside it
                # is outside the question's two-calendar-month lower bound.
                "date": "2023-05-22",
                "status": "completed",
            },
            {
                "handle_ids": ["H006"],
                "kind": "direct",
                "summary": "General assistant advice about photographing jewelry.",
                "date": "2023-05-28",
                "status": "unknown",
            },
            {
                "handle_ids": ["H007"],
                "kind": "direct",
                "summary": "I acquired a wooden jewelry box as a gift.",
                "date": "2023-10-14",
                "status": "completed",
            },
        ],
        parent="1 piece.",
    )


def test_v4_question_topic_scope_repairs_action_wide_aggregate_rejection() -> None:
    fitted = _q40_fitted()
    legacy = dict(fitted.validation_contract)
    candidate = _completion("3 pieces.", ("H001", "H002", "H003"))

    historical = _parse_v3(fitted, candidate, legacy)
    assert historical.valid is False
    assert historical.error_code == "aggregate_scope_incomplete"

    repaired = _parse_v4(fitted, Q40, candidate, legacy)
    assert repaired.valid is True
    assert repaired.prediction == "3 pieces."
    assert repaired.validation_basis == "model_attested_question_topic_complete_v4"
    assert len(repaired.receipt_sha256) == 64
    assert repaired.receipt_sha256 != historical.receipt_sha256


def test_v4_topic_scope_remains_fail_closed_for_omissions_and_distractors() -> None:
    fitted = _q40_fitted()
    legacy = dict(fitted.validation_contract)

    omitted = _parse_v4(
        fitted,
        Q40,
        _completion("2 pieces.", ("H001", "H002")),
        legacy,
    )
    assert omitted.valid is False
    assert omitted.error_code == "aggregate_scope_incomplete"

    candidate_only_topic = _parse_v4(
        fitted,
        Q40,
        _completion("1 laptop.", ("H004",)),
        legacy,
    )
    assert candidate_only_topic.valid is False
    assert candidate_only_topic.error_code == "aggregate_scope_topic_unresolved"

    mixed = _parse_v4(
        fitted,
        Q40,
        _completion("4 pieces.", ("H001", "H002", "H003", "H004")),
        legacy,
    )
    assert mixed.valid is False
    assert mixed.error_code == "aggregate_scope_topic_unresolved"


def _historical_q54_contract():
    fitted = _fitted(
        Q54,
        [
            {
                "handle_ids": ["H001"],
                "kind": "event",
                "summary": "I bought a smoker today for barbecue experiments.",
                "date": "2023-03-15T04:56:00-07:00",
                "status": "completed",
            },
            {
                "handle_ids": ["H002"],
                "kind": "event",
                "summary": "I bought a toaster on sale for breakfast.",
                "date": "2023-03-18T10:00:00-07:00",
                "status": "completed",
            },
            {
                "handle_ids": ["H003"],
                "kind": "event",
                "summary": "I acquired a backpack for my commute.",
                "date": "2023-03-21T10:00:00-07:00",
                "status": "completed",
            },
        ],
        parent="I don't know.",
    )
    legacy = dict(fitted.validation_contract)
    legacy["deterministic_execution_advisory"] = {
        "advisory_only": True,
        "executor": "time",
        "prediction": (
            "I bought a smoker today for barbecue experiments. → "
            "I bought a toaster on sale for breakfast. → "
            "I acquired a backpack for my commute."
        ),
        "receipt_sha256": _sha("historical-bad-relative-advisory"),
        "status": "supported",
        "used_handle_ids": ["H001", "H002", "H003"],
    }
    return fitted, legacy


def test_v4_direct_semantic_shape_demotes_bad_relative_timeline_advisory() -> None:
    fitted, legacy = _historical_q54_contract()
    candidate = _completion("A smoker.", ("H001",))

    historical = _parse_v3(fitted, candidate, legacy)
    assert historical.valid is False
    assert historical.error_code == "deterministic_advisory_disagreement"

    contract = upgrade_completion_validation_contract_v4(
        legacy, dated_question=Q54
    )
    assert contract["deterministic_advisory_policy"]["status"] == "ineligible"
    assert (
        contract["deterministic_advisory_policy"]["reason"]
        == "semantic_answer_shape_requires_candidate_arbiter"
    )
    assert contract["temporal_validation"]["target_date"] == "2023-03-15"

    repaired = _parse_v4(fitted, Q54, candidate, legacy)
    assert repaired.valid is True
    assert repaired.prediction == "A smoker."
    assert repaired.used_handle_ids == ("H001",)
    assert repaired.validation_basis == "model_attested_relative_exact_day_v4"


def test_v4_relative_replacement_must_cite_the_exact_question_derived_day() -> None:
    fitted, legacy = _historical_q54_contract()

    wrong_day = _parse_v4(
        fitted,
        Q54,
        _completion("A toaster.", ("H002",)),
        legacy,
    )
    assert wrong_day.valid is False
    assert wrong_day.error_code == "relative_temporal_target_mismatch"

    mixed_days = _parse_v4(
        fitted,
        Q54,
        _completion("A smoker and a toaster.", ("H001", "H002")),
        legacy,
    )
    assert mixed_days.valid is False
    assert mixed_days.error_code == "relative_temporal_target_mismatch"


def test_v4_keeps_executable_numeric_advisory_exclusive() -> None:
    question = (
        "[Question asked at 2026/08/30 12:00]\n"
        "How many apples did I acquire?"
    )
    fitted = _fitted(
        question,
        [
            {
                "handle_ids": ["H001"],
                "kind": "operand",
                "summary": "I acquired one apple.",
                "numeric_value": 1,
                "numeric_role": "operand",
                "status": "completed",
            },
            {
                "handle_ids": ["H002"],
                "kind": "operand",
                "summary": "I acquired two apples.",
                "numeric_value": 2,
                "numeric_role": "operand",
                "status": "completed",
            },
        ],
        parent="2",
        frontier=FrontierMode.EXHAUSTIVE,
    )
    legacy = dict(fitted.validation_contract)
    legacy["deterministic_execution_advisory"] = {
        "advisory_only": True,
        "executor": "numeric",
        "prediction": "3",
        "receipt_sha256": _sha("executable-numeric-advisory"),
        "status": "supported",
        "used_handle_ids": ["H001", "H002"],
    }
    contract = upgrade_completion_validation_contract_v4(
        legacy, dated_question=question
    )
    assert contract["deterministic_advisory_policy"]["binding_required"] is True

    disagreement = _parse_v4(
        fitted,
        question,
        _completion("1", ("H001",)),
        legacy,
    )
    assert disagreement.valid is False
    assert disagreement.error_code == "deterministic_advisory_disagreement"

    advisory = legacy["deterministic_execution_advisory"]
    exact = _parse_v4(
        fitted,
        question,
        _completion(
            advisory["prediction"], tuple(advisory["used_handle_ids"])
        ),
        legacy,
    )
    assert exact.valid is True
    assert exact.validation_basis == "deterministic_execution_agreement"
