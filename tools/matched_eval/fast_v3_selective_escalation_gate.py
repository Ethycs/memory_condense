"""Gold-blind, one-step escalation policy for the fast-v3 memory path.

The gate does not retrieve evidence and does not answer a question.  It
chooses the next permitted memory operation from a sealed question-local
state.  Callers must re-evaluate it after every operation, which keeps the
online path adaptive without turning a first observation into a static plan.

Only content hashes, typed-closure state, availability flags, and question
text cross this boundary.  Benchmark identifiers, references, verdicts, and
population statistics are deliberately outside the schema.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from tools._routed_repair_routing import RoutedRepairStyle

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
)
from .typed_operator_spec import (
    AnswerShape,
    ComparisonMode,
    SlotKind,
    TemporalMode,
    TypedOperatorSpec,
    compile_typed_operator_spec,
)


MECHANISM_ID = "fast_v3_selective_escalation_gate_v1"
STATE_FORMAT = "memory-condense-fast-v3-escalation-state-v1"
POLICY_FORMAT = "memory-condense-fast-v3-escalation-policy-v1"
DECISION_FORMAT = "memory-condense-fast-v3-escalation-decision-v1"


class FastV3EscalationError(MatchedEvalContractError):
    """A sealed state or escalation invariant changed."""


class AnswerState(str, Enum):
    NOT_RUN = "not_run"
    USEFUL = "useful"
    ABSTAINED = "abstained"
    INVALID = "invalid"


class OperatorState(str, Enum):
    NOT_RUN = "not_run"
    SUPPORTED = "supported"
    INSUFFICIENT = "insufficient"
    CONFLICTED = "conflicted"
    NON_DETERMINISTIC = "non_deterministic"


class NextAction(str, Enum):
    NONE = "none"
    PACKED_OPERATOR = "packed_operator"
    SOURCE_LOCAL_EPISODE = "source_local_episode"
    NUMERIC_FULL_STORE = "numeric_full_store"
    PROFILE_FULL_STORE = "profile_full_store"
    TEMPORAL_FULL_STORE = "temporal_full_store"
    SEMANTIC_GLOBAL = "semantic_global"


_EXECUTABLE_ACTIONS = tuple(action for action in NextAction if action is not NextAction.NONE)
_EXECUTABLE_VALUES = frozenset(action.value for action in _EXECUTABLE_ACTIONS)
_STATE_KEYS = frozenset(
    {
        "answer_state",
        "authenticated_anchor_available",
        "available_actions",
        "completed_actions",
        "format",
        "frontier_closed",
        "frontier_truncated",
        "gold_loaded",
        "new_provider_calls",
        "operator_execution_status",
        "parent_prediction_sha256",
        "provider_packet_sha256",
        "receipt_sha256",
        "retained_transformer_token_state_bytes",
        "typed_frontier_receipt_sha256",
        "typed_spec_receipt_sha256",
        "unresolved_obligation_ids",
        "unresolved_slot_ids",
    }
)


def _require(ok: object, message: str) -> None:
    if not ok:
        raise FastV3EscalationError(message)


def _ordered_actions(value: object, label: str) -> tuple[NextAction, ...]:
    _require(
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray)),
        f"{label} must be an ordered action sequence",
    )
    try:
        actions = tuple(NextAction(row) for row in value)
    except (TypeError, ValueError) as exc:
        raise FastV3EscalationError(f"{label} contains an unknown action") from exc
    _require(
        all(action is not NextAction.NONE for action in actions)
        and len(actions) == len(set(actions)),
        f"{label} must contain unique executable actions",
    )
    return actions


def _ordered_ids(value: object, label: str) -> tuple[str, ...]:
    _require(
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray)),
        f"{label} must be an ordered sequence",
    )
    rows = tuple(value)
    _require(
        all(type(row) is str and bool(row) for row in rows)
        and len(rows) == len(set(rows)),
        f"{label} must contain unique exact text",
    )
    return rows


@dataclass(frozen=True, slots=True)
class FastV3EscalationPolicy:
    """Versioned action order; budgets live with the selected mechanisms."""

    specialist_order: tuple[NextAction, ...] = (
        NextAction.NUMERIC_FULL_STORE,
        NextAction.PROFILE_FULL_STORE,
        NextAction.TEMPORAL_FULL_STORE,
    )
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            type(self.specialist_order) is tuple
            and set(self.specialist_order)
            == {
                NextAction.NUMERIC_FULL_STORE,
                NextAction.PROFILE_FULL_STORE,
                NextAction.TEMPORAL_FULL_STORE,
            }
            and len(self.specialist_order) == 3,
            "specialist order must be an exact permutation",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "escalation policy receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="fast_v3_escalation_policy")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "format": POLICY_FORMAT,
            "gold_loaded": False,
            "mechanism_id": MECHANISM_ID,
            "new_provider_calls": 0,
            "one_action_per_evaluation": True,
            "retained_transformer_token_state_bytes": 0,
            "specialist_order": [row.value for row in self.specialist_order],
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class _ValidatedState:
    projection: Mapping[str, Any]
    receipt_sha256: str
    answer_state: AnswerState
    operator_state: OperatorState
    parent_prediction_sha256: str | None
    frontier_closed: bool
    frontier_truncated: bool
    unresolved_slot_ids: tuple[str, ...]
    unresolved_obligation_ids: tuple[str, ...]
    completed_actions: tuple[NextAction, ...]
    available_actions: tuple[NextAction, ...]
    authenticated_anchor_available: bool


@dataclass(frozen=True, slots=True)
class FastV3EscalationDecision:
    question_sha256: str
    policy_receipt_sha256: str
    state_receipt_sha256: str
    typed_spec_receipt_sha256: str
    applicable_specialist_actions: tuple[NextAction, ...]
    next_action: NextAction
    reasons: tuple[str, ...]
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    gold_loaded: Literal[False] = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.question_sha256, "escalation question"),
            (self.policy_receipt_sha256, "escalation policy"),
            (self.state_receipt_sha256, "escalation state"),
            (self.typed_spec_receipt_sha256, "escalation typed spec"),
        ):
            require_sha256(value, label)
        _require(type(self.next_action) is NextAction, "next action must be canonical")
        _require(
            type(self.applicable_specialist_actions) is tuple
            and all(
                row
                in {
                    NextAction.NUMERIC_FULL_STORE,
                    NextAction.PROFILE_FULL_STORE,
                    NextAction.TEMPORAL_FULL_STORE,
                }
                for row in self.applicable_specialist_actions
            )
            and len(self.applicable_specialist_actions)
            == len(set(self.applicable_specialist_actions)),
            "applicable specialist actions changed",
        )
        _require(
            type(self.reasons) is tuple
            and bool(self.reasons)
            and all(type(row) is str and bool(row) for row in self.reasons)
            and len(self.reasons) == len(set(self.reasons)),
            "escalation reasons must be ordered unique text",
        )
        _require(
            self.provider_prompt_count == 0
            and self.retained_transformer_token_state_bytes == 0
            and self.gold_loaded is False,
            "escalation decision must remain provider-free and gold-blind",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "escalation decision receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="fast_v3_escalation_decision")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "applicable_specialist_actions": [
                row.value for row in self.applicable_specialist_actions
            ],
            "format": DECISION_FORMAT,
            "gold_loaded": False,
            "next_action": self.next_action.value,
            "policy_receipt_sha256": self.policy_receipt_sha256,
            "provider_prompt_count": 0,
            "question_sha256": self.question_sha256,
            "reasons": list(self.reasons),
            "retained_transformer_token_state_bytes": 0,
            "state_receipt_sha256": self.state_receipt_sha256,
            "typed_spec_receipt_sha256": self.typed_spec_receipt_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def seal_fast_v3_escalation_state(
    *,
    provider_packet_sha256: str,
    parent_prediction_sha256: str | None,
    answer_state: AnswerState,
    typed_spec_receipt_sha256: str,
    typed_frontier_receipt_sha256: str,
    frontier_closed: bool,
    frontier_truncated: bool,
    unresolved_slot_ids: tuple[str, ...] = (),
    unresolved_obligation_ids: tuple[str, ...] = (),
    operator_execution_status: OperatorState = OperatorState.NOT_RUN,
    completed_actions: tuple[NextAction, ...] = (),
    available_actions: tuple[NextAction, ...] = (),
    authenticated_anchor_available: bool = False,
) -> dict[str, Any]:
    """Create the exact sealed mapping accepted by the gate."""

    body: dict[str, Any] = {
        "answer_state": answer_state.value,
        "authenticated_anchor_available": authenticated_anchor_available,
        "available_actions": [row.value for row in available_actions],
        "completed_actions": [row.value for row in completed_actions],
        "format": STATE_FORMAT,
        "frontier_closed": frontier_closed,
        "frontier_truncated": frontier_truncated,
        "gold_loaded": False,
        "new_provider_calls": 0,
        "operator_execution_status": operator_execution_status.value,
        "parent_prediction_sha256": parent_prediction_sha256,
        "provider_packet_sha256": provider_packet_sha256,
        "retained_transformer_token_state_bytes": 0,
        "typed_frontier_receipt_sha256": typed_frontier_receipt_sha256,
        "typed_spec_receipt_sha256": typed_spec_receipt_sha256,
        "unresolved_obligation_ids": list(unresolved_obligation_ids),
        "unresolved_slot_ids": list(unresolved_slot_ids),
    }
    body["receipt_sha256"] = identity_sha256(body)
    _validate_state(body)
    return body


def _validate_state(value: Mapping[str, Any]) -> _ValidatedState:
    _require(type(value) is dict, "escalation state must be an exact object")
    _require(set(value) == _STATE_KEYS, "escalation state schema changed")
    _require(value.get("format") == STATE_FORMAT, "escalation state format changed")
    _require(
        value.get("gold_loaded") is False
        and value.get("new_provider_calls") == 0
        and value.get("retained_transformer_token_state_bytes") == 0,
        "escalation state crossed the gold/provider boundary",
    )
    assert_gold_blind(value, path="fast_v3_escalation_state")
    declared = require_sha256(value.get("receipt_sha256"), "escalation state receipt")
    body = dict(value)
    body.pop("receipt_sha256")
    _require(declared == identity_sha256(body), "escalation state receipt changed")
    for key in (
        "provider_packet_sha256",
        "typed_frontier_receipt_sha256",
        "typed_spec_receipt_sha256",
    ):
        require_sha256(value.get(key), key.replace("_", " "))
    try:
        answer = AnswerState(value.get("answer_state"))
        operator = OperatorState(value.get("operator_execution_status"))
    except (TypeError, ValueError) as exc:
        raise FastV3EscalationError("escalation state enum changed") from exc
    parent = value.get("parent_prediction_sha256")
    if answer is AnswerState.NOT_RUN:
        _require(parent is None, "not-run answer cannot carry a prediction")
    else:
        require_sha256(parent, "parent prediction")
    for key in (
        "authenticated_anchor_available",
        "frontier_closed",
        "frontier_truncated",
    ):
        _require(type(value.get(key)) is bool, f"{key} must be exact")
    slots = _ordered_ids(value.get("unresolved_slot_ids"), "unresolved slots")
    obligations = _ordered_ids(
        value.get("unresolved_obligation_ids"), "unresolved obligations"
    )
    completed = _ordered_actions(value.get("completed_actions"), "completed actions")
    available = _ordered_actions(value.get("available_actions"), "available actions")
    operator_completed = NextAction.PACKED_OPERATOR in completed
    _require(
        operator_completed == (operator is not OperatorState.NOT_RUN),
        "packed-operator completion and execution state disagree",
    )
    _require(
        not (value["frontier_closed"] and value["frontier_truncated"]),
        "a truncated frontier cannot be closed",
    )
    return _ValidatedState(
        projection=dict(value),
        receipt_sha256=declared,
        answer_state=answer,
        operator_state=operator,
        parent_prediction_sha256=parent,
        frontier_closed=value["frontier_closed"],
        frontier_truncated=value["frontier_truncated"],
        unresolved_slot_ids=slots,
        unresolved_obligation_ids=obligations,
        completed_actions=completed,
        available_actions=available,
        authenticated_anchor_available=value["authenticated_anchor_available"],
    )


def _deterministic_operator_applicable(spec: TypedOperatorSpec) -> bool:
    return bool(
        spec.temporal_mode is not TemporalMode.NONE
        or spec.answer_shape
        in {AnswerShape.NUMBER, AnswerShape.BOOLEAN, AnswerShape.SET_LIST}
        or spec.comparison_mode is not ComparisonMode.NONE
    )


def _applicable_specialists(spec: TypedOperatorSpec) -> tuple[NextAction, ...]:
    result: list[NextAction] = []
    if spec.style is RoutedRepairStyle.NUMERIC_REDUCE:
        result.append(NextAction.NUMERIC_FULL_STORE)
    if spec.style is RoutedRepairStyle.SYNTHESIZE:
        result.append(NextAction.PROFILE_FULL_STORE)
    numeric_required = sum(slot.requires_numeric for slot in spec.required_slots)
    if (
        spec.style is RoutedRepairStyle.TIMELINE
        or spec.temporal_mode is not TemporalMode.NONE
        or numeric_required >= 2
        or any(slot.kind is SlotKind.TEMPORAL_BOUNDARY for slot in spec.required_slots)
    ):
        result.append(NextAction.TEMPORAL_FULL_STORE)
    return tuple(dict.fromkeys(result))


def evaluate_fast_v3_escalation(
    dated_question: str,
    sealed_state: Mapping[str, Any],
    /,
    *,
    policy: FastV3EscalationPolicy | None = None,
) -> FastV3EscalationDecision:
    """Select exactly one next action from question-local sealed state."""

    if type(dated_question) is not str or not dated_question or dated_question.strip() != dated_question:
        raise FastV3EscalationError("dated question must be non-empty exact text")
    state = _validate_state(sealed_state)
    spec = compile_typed_operator_spec(dated_question)
    _require(
        hashlib.sha256(dated_question.encode("utf-8")).hexdigest()
        == spec.question_sha256,
        "compiled question identity changed",
    )
    _require(
        state.projection["typed_spec_receipt_sha256"] == spec.receipt_sha256,
        "escalation state belongs to another typed specification",
    )
    required_slot_ids = {row.slot_id for row in spec.required_slots}
    _require(
        set(state.unresolved_slot_ids) <= required_slot_ids,
        "unresolved slot escaped the typed specification",
    )
    if state.operator_state is OperatorState.SUPPORTED:
        _require(not state.unresolved_slot_ids, "supported operator retained missing slots")
        _require(
            not spec.requires_complete_frontier or state.frontier_closed,
            "supported closed-world operator has an open frontier",
        )

    active_policy = policy or FastV3EscalationPolicy()
    applicable = _applicable_specialists(spec)
    completed = set(state.completed_actions)
    available = set(state.available_actions)
    bad_answer = state.answer_state in {AnswerState.ABSTAINED, AnswerState.INVALID}
    concrete_unresolved = bool(
        state.unresolved_slot_ids or state.unresolved_obligation_ids
    )
    closure_required = bool(
        spec.requires_complete_frontier
        or (bad_answer and spec.absence_decision_requires_closed_frontier)
    )
    frontier_unresolved = closure_required and not state.frontier_closed
    operator_unresolved = state.operator_state in {
        OperatorState.INSUFFICIENT,
        OperatorState.CONFLICTED,
    }
    needs_resolution = bool(
        concrete_unresolved or frontier_unresolved or operator_unresolved or bad_answer
    )

    action = NextAction.NONE
    reasons: list[str] = []
    if (
        _deterministic_operator_applicable(spec)
        and NextAction.PACKED_OPERATOR not in completed
    ):
        action = NextAction.PACKED_OPERATOR
        reasons.append("deterministic_packed_operator_not_attempted")
    elif (
        concrete_unresolved
        and state.authenticated_anchor_available
        and NextAction.SOURCE_LOCAL_EPISODE in available
        and NextAction.SOURCE_LOCAL_EPISODE not in completed
    ):
        action = NextAction.SOURCE_LOCAL_EPISODE
        reasons.append("concrete_obligation_has_authenticated_local_anchor")
    elif needs_resolution:
        for candidate in active_policy.specialist_order:
            if (
                candidate in applicable
                and candidate in available
                and candidate not in completed
            ):
                action = candidate
                reasons.append("applicable_specialist_remains_untried")
                break
        if (
            action is NextAction.NONE
            and NextAction.SEMANTIC_GLOBAL in available
            and NextAction.SEMANTIC_GLOBAL not in completed
        ):
            action = NextAction.SEMANTIC_GLOBAL
            reasons.append("unresolved_after_cheaper_available_actions")

    if action is NextAction.NONE:
        if state.answer_state is AnswerState.USEFUL and not needs_resolution:
            reasons.append("useful_answer_and_obligations_closed")
        elif not needs_resolution:
            reasons.append("no_memory_escalation_required")
        else:
            reasons.append("no_untried_applicable_action_available")
    if concrete_unresolved:
        reasons.append("typed_obligation_unresolved")
    if frontier_unresolved:
        reasons.append("required_frontier_open")
    if bad_answer:
        reasons.append(f"answer_{state.answer_state.value}")

    return FastV3EscalationDecision(
        question_sha256=spec.question_sha256,
        policy_receipt_sha256=active_policy.receipt_sha256,
        state_receipt_sha256=state.receipt_sha256,
        typed_spec_receipt_sha256=spec.receipt_sha256,
        applicable_specialist_actions=applicable,
        next_action=action,
        reasons=tuple(dict.fromkeys(reasons)),
    )


__all__ = [
    "AnswerState",
    "DECISION_FORMAT",
    "FastV3EscalationDecision",
    "FastV3EscalationError",
    "FastV3EscalationPolicy",
    "MECHANISM_ID",
    "NextAction",
    "OperatorState",
    "POLICY_FORMAT",
    "STATE_FORMAT",
    "evaluate_fast_v3_escalation",
    "seal_fast_v3_escalation_state",
]
