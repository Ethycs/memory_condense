from __future__ import annotations

import copy

import pytest

from tools.matched_eval.fast_v3_selective_escalation_gate import (
    AnswerState,
    FastV3EscalationError,
    NextAction,
    OperatorState,
    evaluate_fast_v3_escalation,
    seal_fast_v3_escalation_state,
)
from tools.matched_eval.typed_operator_spec import compile_typed_operator_spec


DIRECT = (
    "[Question asked at 2023/05/30 (Tue) 14:18]\n"
    "What type of bulb did I replace in my bedside lamp?"
)
COUNT = (
    "[Question asked at 2023/05/30 (Tue) 14:18]\n"
    "How many different cuisines did I try?"
)
TIMELINE = (
    "[Question asked at 2023/05/30 (Tue) 14:18]\n"
    "In what order did I visit Paris and Rome?"
)
SYNTHESIS = (
    "[Question asked at 2023/05/30 (Tue) 14:18]\n"
    "Why do I prefer warm lighting?"
)
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _state(
    question: str,
    *,
    answer_state: AnswerState = AnswerState.NOT_RUN,
    operator_state: OperatorState = OperatorState.NOT_RUN,
    frontier_closed: bool = False,
    frontier_truncated: bool = True,
    unresolved_slot_ids: tuple[str, ...] = (),
    unresolved_obligation_ids: tuple[str, ...] = (),
    completed: tuple[NextAction, ...] = (),
    available: tuple[NextAction, ...] = (),
    anchor: bool = False,
) -> dict:
    spec = compile_typed_operator_spec(question)
    return seal_fast_v3_escalation_state(
        provider_packet_sha256=SHA_A,
        parent_prediction_sha256=(
            None if answer_state is AnswerState.NOT_RUN else SHA_B
        ),
        answer_state=answer_state,
        typed_spec_receipt_sha256=spec.receipt_sha256,
        typed_frontier_receipt_sha256=SHA_C,
        frontier_closed=frontier_closed,
        frontier_truncated=frontier_truncated,
        unresolved_slot_ids=unresolved_slot_ids,
        unresolved_obligation_ids=unresolved_obligation_ids,
        operator_execution_status=operator_state,
        completed_actions=completed,
        available_actions=available,
        authenticated_anchor_available=anchor,
    )


def test_useful_direct_answer_stops_despite_bounded_frontier() -> None:
    decision = evaluate_fast_v3_escalation(
        DIRECT,
        _state(DIRECT, answer_state=AnswerState.USEFUL),
    )
    assert decision.next_action is NextAction.NONE
    assert decision.reasons[0] == "useful_answer_and_obligations_closed"


def test_deterministic_route_tries_packed_operator_first() -> None:
    decision = evaluate_fast_v3_escalation(
        COUNT,
        _state(
            COUNT,
            available=(
                NextAction.SOURCE_LOCAL_EPISODE,
                NextAction.NUMERIC_FULL_STORE,
                NextAction.SEMANTIC_GLOBAL,
            ),
            anchor=True,
            unresolved_obligation_ids=("count-frontier",),
        ),
    )
    assert decision.next_action is NextAction.PACKED_OPERATOR
    assert decision.applicable_specialist_actions == (
        NextAction.NUMERIC_FULL_STORE,
    )


def test_concrete_obligation_uses_authenticated_local_expansion_second() -> None:
    decision = evaluate_fast_v3_escalation(
        COUNT,
        _state(
            COUNT,
            operator_state=OperatorState.INSUFFICIENT,
            completed=(NextAction.PACKED_OPERATOR,),
            available=(
                NextAction.SOURCE_LOCAL_EPISODE,
                NextAction.NUMERIC_FULL_STORE,
                NextAction.SEMANTIC_GLOBAL,
            ),
            anchor=True,
            unresolved_obligation_ids=("count-frontier",),
        ),
    )
    assert decision.next_action is NextAction.SOURCE_LOCAL_EPISODE


def test_applicable_specialist_precedes_semantic_global() -> None:
    decision = evaluate_fast_v3_escalation(
        COUNT,
        _state(
            COUNT,
            operator_state=OperatorState.INSUFFICIENT,
            completed=(
                NextAction.PACKED_OPERATOR,
                NextAction.SOURCE_LOCAL_EPISODE,
            ),
            available=(
                NextAction.NUMERIC_FULL_STORE,
                NextAction.SEMANTIC_GLOBAL,
            ),
            unresolved_obligation_ids=("count-frontier",),
        ),
    )
    assert decision.next_action is NextAction.NUMERIC_FULL_STORE


def test_temporal_and_profile_routes_choose_their_own_specialists() -> None:
    temporal = evaluate_fast_v3_escalation(
        TIMELINE,
        _state(
            TIMELINE,
            operator_state=OperatorState.INSUFFICIENT,
            completed=(NextAction.PACKED_OPERATOR,),
            available=(
                NextAction.TEMPORAL_FULL_STORE,
                NextAction.SEMANTIC_GLOBAL,
            ),
        ),
    )
    profile = evaluate_fast_v3_escalation(
        SYNTHESIS,
        _state(
            SYNTHESIS,
            answer_state=AnswerState.ABSTAINED,
            available=(
                NextAction.PROFILE_FULL_STORE,
                NextAction.SEMANTIC_GLOBAL,
            ),
        ),
    )
    assert temporal.next_action is NextAction.TEMPORAL_FULL_STORE
    assert profile.next_action is NextAction.PROFILE_FULL_STORE


def test_bad_direct_answer_without_specialist_routes_semantic_global() -> None:
    decision = evaluate_fast_v3_escalation(
        DIRECT,
        _state(
            DIRECT,
            answer_state=AnswerState.ABSTAINED,
            available=(NextAction.SEMANTIC_GLOBAL,),
        ),
    )
    assert decision.next_action is NextAction.SEMANTIC_GLOBAL
    assert "answer_abstained" in decision.reasons


def test_re_evaluation_advances_one_action_at_a_time() -> None:
    first = _state(
        COUNT,
        available=(NextAction.NUMERIC_FULL_STORE, NextAction.SEMANTIC_GLOBAL),
    )
    assert (
        evaluate_fast_v3_escalation(COUNT, first).next_action
        is NextAction.PACKED_OPERATOR
    )
    second = _state(
        COUNT,
        operator_state=OperatorState.INSUFFICIENT,
        completed=(NextAction.PACKED_OPERATOR,),
        available=(NextAction.NUMERIC_FULL_STORE, NextAction.SEMANTIC_GLOBAL),
    )
    assert (
        evaluate_fast_v3_escalation(COUNT, second).next_action
        is NextAction.NUMERIC_FULL_STORE
    )
    third = _state(
        COUNT,
        operator_state=OperatorState.INSUFFICIENT,
        completed=(NextAction.PACKED_OPERATOR, NextAction.NUMERIC_FULL_STORE),
        available=(NextAction.NUMERIC_FULL_STORE, NextAction.SEMANTIC_GLOBAL),
    )
    assert (
        evaluate_fast_v3_escalation(COUNT, third).next_action
        is NextAction.SEMANTIC_GLOBAL
    )


def test_supported_closed_world_operator_requires_closed_frontier() -> None:
    state = _state(
        COUNT,
        operator_state=OperatorState.SUPPORTED,
        completed=(NextAction.PACKED_OPERATOR,),
    )
    with pytest.raises(FastV3EscalationError, match="open frontier"):
        evaluate_fast_v3_escalation(COUNT, state)


def test_state_schema_rejects_runtime_targeting_fields() -> None:
    state = _state(DIRECT)
    state["ordinal"] = 14
    with pytest.raises(FastV3EscalationError, match="schema"):
        evaluate_fast_v3_escalation(DIRECT, state)


def test_state_receipt_and_question_binding_fail_closed() -> None:
    state = _state(DIRECT)
    changed = copy.deepcopy(state)
    changed["frontier_closed"] = True
    with pytest.raises(FastV3EscalationError, match="receipt changed"):
        evaluate_fast_v3_escalation(DIRECT, changed)
    with pytest.raises(FastV3EscalationError, match="another typed specification"):
        evaluate_fast_v3_escalation(
            "[Question asked at 2023/05/30 (Tue) 14:18]\nWhere did I put it?",
            state,
        )


def test_identical_inputs_replay_byte_identically() -> None:
    state = _state(
        DIRECT,
        answer_state=AnswerState.INVALID,
        available=(NextAction.SEMANTIC_GLOBAL,),
    )
    first = evaluate_fast_v3_escalation(DIRECT, state)
    second = evaluate_fast_v3_escalation(DIRECT, copy.deepcopy(state))
    assert first.projection() == second.projection()
    assert first.receipt_sha256 == second.receipt_sha256

