from __future__ import annotations

import hashlib
import json

from memory_condense.domain.discourse import quote_sha256
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.typed_memory_final_arm import fit_typed_final_prompt
from tools.matched_eval.typed_memory_final_validator_v5 import (
    build_parent_defect_certificate,
    evaluate_typed_final_replacement_policy_v5,
    parse_typed_final_completion_v5,
    upgrade_completion_validation_contract_v5,
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


Q5 = (
    "[Question asked at 2023/05/30 (Tue) 16:40]\n"
    "Can you suggest some accessories that would complement my current "
    "photography setup?"
)
Q36 = (
    "[Question asked at 2023/05/30 (Tue) 23:43]\n"
    "Can you recommend a show or movie for me to watch tonight?"
)
Q54 = (
    "[Question asked at 2023/03/25 (Sat) 18:26]\n"
    "What kitchen appliance did I buy 10 days ago?"
)
Q_CORRECTION = (
    "[Question asked at 2023/05/30 (Tue) 16:40]\n"
    "What camera body do I use?"
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fitted(
    question: str,
    raw_items: list[dict[str, object]],
    *,
    parent: str,
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
    parsed = parse_typed_items(raw_items, operator_spec=spec, bindings=bindings)
    assert not parsed.rejected_items
    packet = build_typed_evidence_packet(
        spec,
        bindings,
        parsed,
        sealed_input_artifact_sha256s=(_sha("sealed-source"),),
        frontier_mode=FrontierMode.BOUNDED,
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


def _parse(
    fitted,
    question: str,
    completion: str,
    *,
    story=None,
    parent_defect_certificate=None,
):
    parent = fitted.provider_input["protected_parent_fallback"]["prediction"]
    contract = upgrade_completion_validation_contract_v5(
        fitted.validation_contract,
        dated_question=question,
        parent_prediction=parent,
        parent_defect_certificate=parent_defect_certificate,
    )
    return parse_typed_final_completion_v5(
        completion,
        dated_question=question,
        parent_prediction=parent,
        allowed_handle_ids=fitted.allowed_handle_ids,
        handle_group_by_id=fitted.handle_group_by_id,
        story_coherence=fitted.story_coherence if story is None else story,
        preservation_requirements=fitted.preservation_requirements,
        validation_contract=contract,
    )


def test_q5_touched_contradiction_requires_the_whole_neighborhood() -> None:
    fitted = _fitted(
        Q5,
        [
            {
                "handle_ids": ["H001"],
                "kind": "event",
                "relation": "authored_by_user",
                "summary": "I use a Sony A7R IV camera body.",
                "status": "unknown",
            },
            {
                "handle_ids": ["H002"],
                "kind": "event",
                "relation": "authored_by_user",
                "summary": "I used my Nikon D750 camera body on my trip.",
                "status": "unknown",
            },
        ],
        parent="My current camera body is a Sony A7R IV.",
    )
    story = {
        **fitted.story_coherence,
        "typed_links": [
            {
                "link_id": "D-camera-conflict",
                "members": [
                    {"handle_id": "H001"},
                    {"handle_id": "H002"},
                ],
                "relation": "contradicts",
            }
        ],
    }
    parsed = _parse(
        fitted,
        Q5,
        _completion("My current camera body is a Nikon D750.", ("H002",)),
        story=story,
    )
    assert parsed.valid is False
    assert parsed.error_code == "conflict_neighborhood_incomplete"


def test_q5_one_matching_fact_cannot_authorize_unsupported_accessories() -> None:
    fitted = _fitted(
        Q5,
        [
            {
                "handle_ids": ["H001"],
                "kind": "event",
                "relation": "authored_by_user",
                "summary": (
                    "I used my Nikon D750 with a Canon 24-70mm f/2.8 lens "
                    "on my recent trip."
                ),
                "status": "unknown",
            }
        ],
        parent="I don't know.",
    )
    parsed = _parse(
        fitted,
        Q5,
        _completion(
            "For my Nikon D750 and Canon 24-70mm lens, add a sturdy tripod, "
            "speedlight, extra batteries, and a weather cover.",
            ("H001",),
        ),
    )
    assert parsed.valid is False
    assert parsed.error_code == "unsupported_material_claim"


def test_q36_incidental_consumption_is_not_preference_evidence() -> None:
    fitted = _fitted(
        Q36,
        [
            {
                "handle_ids": ["H001"],
                "kind": "event",
                "relation": "authored_by_user",
                    "summary": (
                        "I watched Chicago and Moulin Rouge after Evita and "
                        "thought the dancing was amazing. I compared movie "
                        "musical adaptations with seeing them on stage."
                    ),
                "status": "unknown",
            }
        ],
        parent="I don't know.",
    )
    parsed = _parse(
        fitted,
        Q36,
        _completion("Watch a movie musical tonight.", ("H001",)),
    )
    assert parsed.valid is False
    assert parsed.error_code == "preference_evidence_missing"


def test_q36_explicit_user_preference_can_fill_an_abstention() -> None:
    fitted = _fitted(
        Q36,
        [
            {
                "handle_ids": ["H001"],
                "kind": "event",
                "relation": "authored_by_user",
                "summary": (
                    "I love storytelling-focused stand-up comedy specials."
                ),
                "status": "unknown",
            }
        ],
        parent="I don't know.",
    )
    parsed = _parse(
        fitted,
        Q36,
        _completion(
            "Watch a storytelling-focused stand-up comedy special tonight.",
            ("H001",),
        ),
    )
    assert parsed.valid is True
    assert parsed.decision == "replace"
    assert parsed.validation_basis == "abstention_fill_direct_v5"


def test_nonabstaining_parent_requires_a_complete_defect_certificate() -> None:
    fitted = _fitted(
        Q_CORRECTION,
        [
            {
                "handle_ids": ["H001"],
                "kind": "event",
                "relation": "authored_by_user",
                "summary": "I previously used a Sony A7R IV camera body.",
                "status": "unknown",
            },
            {
                "handle_ids": ["H002"],
                "kind": "event",
                "relation": "authored_by_user",
                "summary": "I now use a Nikon D750 camera body.",
                "status": "unknown",
            },
        ],
        parent="A Sony A7R IV camera body.",
    )
    story = {
        **fitted.story_coherence,
        "typed_links": [
            {
                "link_id": "D-camera-revision",
                "members": [
                    {"handle_id": "H001"},
                    {"handle_id": "H002"},
                ],
                "relation": "revises",
            }
        ],
    }
    completion = _completion(
        "A Nikon D750 camera body.",
        ("H001", "H002"),
    )
    missing = _parse(
        fitted,
        Q_CORRECTION,
        completion,
        story=story,
    )
    assert missing.valid is False
    assert missing.error_code == "parent_defect_certificate_missing"

    certificate = build_parent_defect_certificate(
        parent_prediction="A Sony A7R IV camera body.",
        dated_question=Q_CORRECTION,
        challenged_parent_terms=("Sony", "A7R", "IV"),
        supporting_link_ids=("D-camera-revision",),
        used_handle_ids=("H001", "H002"),
    )
    corrected = _parse(
        fitted,
        Q_CORRECTION,
        completion,
        story=story,
        parent_defect_certificate=certificate,
    )
    assert corrected.valid is True
    assert corrected.validation_basis == "certified_parent_correction_v5"


def _q54_fitted(*, evidence_date: str = "2023-03-15"):
    return _fitted(
        Q54,
        [
            {
                "handle_ids": ["H001"],
                "kind": "event",
                "relation": "authored_by_user",
                "summary": "I just got a smoker today for barbecue experiments.",
                "date": evidence_date,
                "status": "completed",
            }
        ],
        parent="I don't know.",
    )


def test_q54_exact_day_direct_evidence_fills_parent_abstention() -> None:
    fitted = _q54_fitted()
    parsed = _parse(
        fitted,
        Q54,
        _completion("You bought a smoker.", ("H001",)),
    )
    assert parsed.valid is True
    assert parsed.prediction == "You bought a smoker."
    assert parsed.validation_basis == "abstention_fill_relative_exact_day_v5"


def test_q54_wrong_day_still_fails_under_inherited_v4_temporal_policy() -> None:
    fitted = _q54_fitted(evidence_date="2023-03-18")
    parsed = _parse(
        fitted,
        Q54,
        _completion("You bought a smoker.", ("H001",)),
    )
    assert parsed.valid is False
    assert parsed.error_code == "relative_temporal_target_mismatch"


def test_real_preflight_row_api_returns_a_gold_blind_fail_closed_proof() -> None:
    fitted = _q54_fitted()
    parent = fitted.provider_input["protected_parent_fallback"]["prediction"]
    body = {
        "allowed_handle_ids": list(fitted.allowed_handle_ids),
        "dated_question_sha256": quote_sha256(Q54),
        "handle_group_by_id": dict(fitted.handle_group_by_id),
        "messages": list(fitted.messages),
        "parent_prediction": parent,
        "preservation_requirements": dict(fitted.preservation_requirements),
        "story_coherence": dict(fitted.story_coherence),
        "validation_contract": dict(fitted.validation_contract),
    }
    plan = {**body, "prompt_row_receipt_sha256": identity_sha256(body)}
    completion = _completion("You bought a smoker.", ("H001",))
    proof = evaluate_typed_final_replacement_policy_v5(plan, completion)
    assert proof["accepted_replacement"] is True
    assert proof["final_prediction"] == "You bought a smoker."
    assert proof["gold_loaded"] is False
    assert proof["physical_provider_calls"] == 0
    assert proof["retained_transformer_token_state_bytes"] == 0
    unsigned = dict(proof)
    receipt = unsigned.pop("policy_proof_receipt_sha256")
    assert receipt == identity_sha256(unsigned)
