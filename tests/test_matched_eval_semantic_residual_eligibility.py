from __future__ import annotations

import copy

import pytest

from tools.matched_eval.contracts import MatchedEvalContractError
from tools.matched_eval.semantic_residual_eligibility import (
    SemanticResidualEligibilityPolicy,
    evaluate_semantic_residual_eligibility,
    replay_semantic_residual_eligibility,
)


def _answer(
    prediction: str = "A confident answer",
    *,
    decision: str = "replace",
    solver_valid: bool | None = True,
    used: tuple[str, ...] = ("H001",),
) -> dict[str, object]:
    return {
        "decision": decision,
        "gold_loaded": False,
        "parse_error_code": None,
        "prediction": prediction,
        "solver_valid": solver_valid,
        "used_handle_ids": list(used),
    }


def _construction(
    style: str = "direct_extract",
    *,
    mode: str = "specialist",
    frontier_mode: str = "bounded",
    truncated: bool = True,
    closed: bool = False,
    unresolved: tuple[str, ...] = (),
    applicable: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "applicable_specialist_ids": list(applicable),
        "gold_loaded": False,
        "methods": [
            {
                "mechanism_id": "protected_parent",
                "typed_contributions": [
                    {
                        "frontier_mode": frontier_mode,
                        "truncated": truncated,
                    }
                ],
            }
        ],
        "mode": mode,
        "route": {"style": style},
        "terminal_prompt": {
            "provider_input": {
                "typed_evidence": {
                    "frontier": {
                        "closed": closed,
                        "mode": frontier_mode,
                        "truncated": truncated,
                        "unresolved_slot_ids": list(unresolved),
                    },
                    "operator_spec": {"style": style},
                }
            }
        },
    }


def test_bounded_frontier_alone_does_not_admit_confident_direct_answer() -> None:
    decision = evaluate_semantic_residual_eligibility(
        _answer(), _construction()
    )

    assert decision.eligible is False
    assert decision.reasons == ()
    assert decision.signals.bounded_or_truncated_frontier is True
    assert decision.signals.route_specialized is False


def test_numeric_keep_parent_with_only_bounded_frontier_is_not_admitted() -> None:
    decision = evaluate_semantic_residual_eligibility(
        _answer(decision="keep_parent", used=()),
        _construction(style="numeric_reduce"),
    )

    assert decision.eligible is False
    assert decision.reasons == ()
    assert decision.signals.answer_weak is True


def test_specialized_parent_passthrough_is_a_route_gap() -> None:
    construction = _construction(
        style="synthesize",
        mode="parent_passthrough",
        applicable=(),
    )
    construction["methods"] = []
    decision = evaluate_semantic_residual_eligibility(
        _answer(decision="parent_passthrough", solver_valid=None, used=()),
        construction,
    )

    assert decision.eligible is True
    assert "specialist_route_gap" in decision.reasons


@pytest.mark.parametrize(
    "prediction",
    (
        "I don't know.",
        "Unable to determine from memory.",
        "Insufficient memory evidence for the requested count.",
    ),
)
def test_abstention_admits_even_a_direct_route(prediction: str) -> None:
    decision = evaluate_semantic_residual_eligibility(
        _answer(
            prediction,
            decision="parent_passthrough",
            solver_valid=None,
            used=(),
        ),
        _construction(style="direct_extract", mode="parent_passthrough"),
    )

    assert decision.eligible is True
    assert decision.reasons[0] == "answer_abstained"


def test_invalid_solver_admits_without_a_specialized_route() -> None:
    decision = evaluate_semantic_residual_eligibility(
        _answer(decision="invalid_keep_parent", solver_valid=False, used=()),
        _construction(style="direct_extract"),
    )

    assert decision.eligible is True
    assert "answer_invalid" in decision.reasons


def test_reconciliation_unresolved_is_generic_and_replayable() -> None:
    answer = _answer()
    construction = _construction()
    reconciliation = {
        "decision_scope": "combined",
        "gold_loaded": False,
        "resolution_status": "unresolved",
    }
    policy = SemanticResidualEligibilityPolicy(residual_payload_token_cap=1_900)
    sealed = evaluate_semantic_residual_eligibility(
        answer,
        construction,
        reconciliation_row=reconciliation,
        policy=policy,
    )

    replayed = replay_semantic_residual_eligibility(
        answer,
        construction,
        sealed,
        reconciliation_row=reconciliation,
        policy=policy,
    )

    assert replayed.projection() == sealed.projection()
    assert sealed.reasons == ("reconciliation_unresolved",)
    assert policy.projection()["non_borrowable_residual_budget"] is True


def test_unresolved_slots_are_preserved_as_sealed_signals() -> None:
    decision = evaluate_semantic_residual_eligibility(
        _answer(decision="keep_parent", used=()),
        _construction(
            style="temporal_timeline",
            unresolved=("event_start", "event_end"),
        ),
    )

    assert decision.signals.unresolved_slot_ids == ("event_start", "event_end")
    assert "specialized_sufficiency_unresolved" in decision.reasons


def test_per_lane_insufficiency_cannot_open_global_residual_gate() -> None:
    reconciliation = {
        "gold_loaded": False,
        "resolution_status": "unresolved",
        "scope": "numeric_lane",
    }
    decision = evaluate_semantic_residual_eligibility(
        _answer(decision="keep_parent", used=()),
        _construction(style="numeric_reduce"),
        reconciliation_row=reconciliation,
    )

    assert decision.eligible is False
    assert decision.signals.reconciliation_unresolved is False


def test_uncited_weak_state_chain_is_an_explicit_profile_need() -> None:
    decision = evaluate_semantic_residual_eligibility(
        _answer(decision="keep_parent", used=()),
        _construction(style="state_chain"),
    )

    assert decision.eligible is True
    assert decision.reasons == ("aggregation_frontier_open",)


def test_v3_combined_resolution_is_terminal_even_when_v2_route_was_weak() -> None:
    answer = {
        "decision_lane": "question_bound_temporal",
        "format": (
            "memory-condense-locked-specialist-final-reconciliation-v3-"
            "result-row-v1"
        ),
        "gold_loaded": False,
        "prediction": "3 months",
        "prediction_source": "locked_v3_temporal_computed",
        "reconciliation": {"operation": "direct_duration"},
    }
    construction = _construction(
        style="temporal_timeline",
        mode="parent_passthrough",
        unresolved=("event_end",),
    )
    construction["methods"] = []

    decision = evaluate_semantic_residual_eligibility(
        answer,
        construction,
        prior_answer_row=_answer(decision="keep_parent", used=()),
    )

    assert decision.eligible is False
    assert decision.signals.combined_resolution_applied is True
    assert decision.signals.combined_decision_lane == "question_bound_temporal"
    assert decision.signals.specialist_route_gap is False


@pytest.mark.parametrize(
    "answer_update, expected_reason",
    (
        (
            {
                "solver_valid": False,
                "parse_error_code": "specialist_temporal_interval_disagreement",
            },
            "answer_invalid",
        ),
        (
            {
                "solver_valid": False,
                "prediction": "Unable to determine from memory.",
            },
            "answer_abstained",
        ),
    ),
)
def test_v3_attempt_is_not_terminal_when_invalid_or_abstaining(
    answer_update: dict[str, object], expected_reason: str
) -> None:
    answer = {
        "decision_lane": "question_bound_temporal",
        "format": (
            "memory-condense-locked-specialist-final-reconciliation-v3-"
            "result-row-v1"
        ),
        "gold_loaded": False,
        "prediction": "3 months",
        "prediction_source": "locked_v3_temporal_computed",
        "reconciliation": {"operation": "direct_duration"},
        **answer_update,
    }
    construction = _construction(
        style="temporal_timeline",
        mode="parent_passthrough",
        unresolved=("event_end",),
    )
    construction["methods"] = []

    decision = evaluate_semantic_residual_eligibility(
        answer,
        construction,
        prior_answer_row=_answer(decision="keep_parent", used=()),
    )

    assert decision.eligible is True
    assert expected_reason in decision.reasons
    assert decision.signals.combined_resolution_applied is False
    assert decision.signals.specialist_route_gap is True


def test_v3_fallback_missing_v2_fields_is_not_weak_numeric_evidence() -> None:
    answer = {
        "decision_lane": "v2_fallback",
        "format": (
            "memory-condense-locked-specialist-final-reconciliation-v3-"
            "result-row-v1"
        ),
        "gold_loaded": False,
        "prediction": "15",
        "prediction_source": "locked_v3_v2_fallback",
        "reconciliation": None,
    }

    decision = evaluate_semantic_residual_eligibility(
        answer,
        _construction(style="numeric_reduce"),
        prior_answer_row=_answer(decision="replace", used=("H001",)),
    )

    assert decision.eligible is False
    assert decision.signals.answer_weak is False
    assert decision.signals.combined_resolution_applied is False


def test_v3_fallback_preserves_uncited_state_chain_need() -> None:
    answer = {
        "decision_lane": "v2_fallback",
        "format": (
            "memory-condense-locked-specialist-final-reconciliation-v3-"
            "result-row-v1"
        ),
        "gold_loaded": False,
        "prediction": "chain and cassette",
        "prediction_source": "locked_v3_v2_fallback",
        "reconciliation": None,
    }

    decision = evaluate_semantic_residual_eligibility(
        answer,
        _construction(style="state_chain"),
        prior_answer_row=_answer(decision="keep_parent", used=()),
    )

    assert decision.eligible is True
    assert decision.reasons == ("aggregation_frontier_open",)


def test_v3_fallback_global_aggregation_does_not_require_weak_inference() -> None:
    answer = {
        "decision_lane": "v2_fallback",
        "format": (
            "memory-condense-locked-specialist-final-reconciliation-v3-"
            "result-row-v1"
        ),
        "gold_loaded": False,
        "prediction": "chain and cassette",
        "prediction_source": "locked_v3_v2_fallback",
        "reconciliation": None,
    }

    decision = evaluate_semantic_residual_eligibility(
        answer, _construction(style="state_chain")
    )

    assert decision.eligible is True
    assert decision.reasons == ("aggregation_frontier_open",)
    assert decision.signals.answer_weak is False
    assert decision.signals.prior_answer_bound is False


@pytest.mark.parametrize("style", ("set_list", "set_join", "profile", "synthesize"))
def test_closed_world_aggregation_requires_frontier_closure(style: str) -> None:
    decision = evaluate_semantic_residual_eligibility(
        _answer(decision="replace", used=("H001",)),
        _construction(style=style),
    )

    assert decision.eligible is True
    assert decision.reasons == ("aggregation_frontier_open",)
    assert decision.signals.answer_weak is False


def test_gold_or_reference_fields_are_rejected_before_gating() -> None:
    answer = _answer()
    answer["reference_answer"] = "secret"

    with pytest.raises(MatchedEvalContractError, match="gold-bearing field"):
        evaluate_semantic_residual_eligibility(answer, _construction())


def test_tampered_replay_fails_closed() -> None:
    answer = _answer(decision="keep_parent", used=())
    construction = _construction(style="numeric_reduce")
    sealed = evaluate_semantic_residual_eligibility(answer, construction)
    changed = copy.deepcopy(answer)
    changed["prediction"] = "A different answer"

    with pytest.raises(MatchedEvalContractError, match="replay changed"):
        replay_semantic_residual_eligibility(changed, construction, sealed)
