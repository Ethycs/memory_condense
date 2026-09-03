"""Pure provider-free policy-v5 numeric and typed arbitration helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from memory_condense.domain.discourse import quote_sha256
from tools.matched_eval.artifacts import SealedArtifact
from tools.matched_eval.contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.operator_first_numeric_policy import (
    DECISION_FORMAT as NUMERIC_POLICY_FORMAT,
    OperatorFirstNumericDecision,
    RelevantNumericFrontier,
    execute_operator_first_numeric_policy,
)
from tools.matched_eval.typed_memory_final_arm import render_final_messages
from tools.matched_eval.typed_memory_final_validator_v5 import (
    FORMAT as VALIDATOR_V5_FORMAT,
    VALIDATOR_POLICY_FORMAT,
    evaluate_typed_final_replacement_policy_v5,
)
from tools.matched_eval.typed_operator_executor import ExecutionStatus

_V5_PROOF_FORMAT = f"{VALIDATOR_V5_FORMAT}-policy-proof-v1"
_NUMERIC_STATUSES = frozenset(status.value for status in ExecutionStatus)


class LockedSemanticGlobalTerminalFull100PolicyV5Error(MatchedEvalContractError):
    """A provider-free policy-v5 proof changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedSemanticGlobalTerminalFull100PolicyV5Error(message)

def _receipt_row(
    raw: Mapping[str, Any], *, key: str, label: str
) -> dict[str, Any]:
    _require(type(raw) is dict, f"{label} changed type")
    row = dict(raw)
    declared = require_sha256(row.pop(key, None), label)
    _require(identity_sha256(row) == declared, f"{label} receipt changed")
    row[key] = declared
    return row


def _strict_json_object(text: str, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise ValueError(f"duplicate key: {key}")
            value[key] = child
        return value

    try:
        value = json.loads(text, object_pairs_hook=reject_duplicates)
    except (json.JSONDecodeError, ValueError) as exc:
        raise LockedSemanticGlobalTerminalFull100PolicyV5Error(
            f"{label} is not strict JSON"
        ) from exc
    _require(type(value) is dict, f"{label} must be a JSON object")
    return value


def authenticated_provider_input(plan_row: Mapping[str, Any]) -> dict[str, Any]:
    """Recover and authenticate the provider input sealed in a preflight row."""

    plan = _receipt_row(
        plan_row,
        key="prompt_row_receipt_sha256",
        label="policy-v5 prompt row",
    )
    messages = plan.get("messages")
    _require(
        type(messages) is list
        and len(messages) == 2
        and all(
            type(row) is dict
            and set(row) == {"role", "content"}
            and type(row.get("content")) is str
            for row in messages
        )
        and messages[0].get("role") == "system"
        and messages[1].get("role") == "user",
        "policy-v5 prompt messages changed",
    )
    provider_input = _strict_json_object(
        str(messages[1]["content"]), "policy-v5 provider input"
    )
    try:
        rendered = [dict(row) for row in render_final_messages(provider_input)]
        provider_input_sha256 = identity_sha256(provider_input)
    except (MatchedEvalContractError, TypeError, ValueError) as exc:
        raise LockedSemanticGlobalTerminalFull100PolicyV5Error(
            "policy-v5 provider input is not canonical"
        ) from exc
    dated_question = require_text(
        provider_input.get("dated_question"), "policy-v5 dated question"
    )
    _require(
        rendered == messages
        and identity_sha256(messages) == plan.get("messages_sha256")
        and provider_input_sha256 == plan.get("provider_input_sha256")
        and quote_sha256(dated_question) == plan.get("dated_question_sha256"),
        "policy-v5 provider input differs from its sealed prompt",
    )
    assert_gold_blind(provider_input, path="full100_policy_v5_provider_input")
    return provider_input


def _validated_completion_records(
    source_run: SealedArtifact,
    plans: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    """Bind every logical completion to its checkpoint-only response record."""

    count = len(plans)
    batch = source_run.payload.get("completion_batch")
    _require(type(batch) is dict, "policy-v5 source completion batch changed type")
    completions = batch.get("logical_completions")
    records = batch.get("unique_records")
    usage = batch.get("usage")
    _require(
        type(completions) is list
        and len(completions) == count
        and all(type(value) is str and bool(value) for value in completions)
        and type(records) is list
        and len(records) == count
        and type(usage) is dict
        and usage.get("logical_calls") == count
        and usage.get("unique_calls") == count
        and usage.get("checkpoint_hits") == count
        and usage.get("physical_calls") == 0,
        "policy-v5 source completion batch is not checkpoint-only",
    )
    by_messages: dict[str, Mapping[str, Any]] = {}
    for raw in records:
        _require(type(raw) is dict, "policy-v5 completion record changed type")
        record = dict(raw)
        messages_sha = require_sha256(
            record.get("messages_sha256"), "policy-v5 response messages"
        )
        completion = record.get("completion")
        _require(
            type(completion) is str
            and bool(completion)
            and record.get("completion_sha256") == quote_sha256(completion)
            and record.get("checkpoint_hit") is True
            and record.get("physical_call") is False
            and messages_sha not in by_messages,
            "policy-v5 completion record changed",
        )
        for key in (
            "call_key_sha256",
            "request_journal_sha256",
            "response_journal_sha256",
        ):
            require_sha256(record.get(key), f"policy-v5 source {key}")
        by_messages[messages_sha] = record

    result: list[tuple[str, Mapping[str, Any]]] = []
    for plan, completion in zip(plans, completions, strict=True):
        messages_sha = require_sha256(
            plan.get("messages_sha256"), "policy-v5 plan messages"
        )
        record = by_messages.get(messages_sha)
        _require(
            record is not None and record.get("completion") == completion,
            "policy-v5 completion differs from its authenticated prompt journal",
        )
        assert record is not None
        result.append((completion, record))
    return tuple(result)


def _validated_source_questions(
    source_run: SealedArtifact,
    expected_ordinals: Sequence[int],
) -> dict[int, dict[str, Any]]:
    raw = source_run.payload.get("questions")
    _require(
        type(raw) is list and len(raw) == len(expected_ordinals),
        "policy-v5 source question population changed",
    )
    by_ordinal: dict[int, dict[str, Any]] = {}
    for value in raw:
        _require(type(value) is dict, "policy-v5 source question changed type")
        row = dict(value)
        unsigned = dict(row)
        declared = require_sha256(
            unsigned.pop("source_row_sha256", None),
            "policy-v5 source answer row",
        )
        prediction = require_text(
            row.get("prediction"), "policy-v5 source prediction"
        )
        ordinal = row.get("ordinal")
        _require(
            type(ordinal) is int
            and ordinal not in by_ordinal
            and identity_sha256(unsigned) == declared
            and row.get("prediction_sha256") == quote_sha256(prediction),
            "policy-v5 source answer row changed",
        )
        require_text(row.get("question_id"), "policy-v5 source question ID")
        for key in (
            "dated_question_sha256",
            "parent_prediction_sha256",
            "question_sha256",
        ):
            require_sha256(row.get(key), f"policy-v5 source {key}")
        by_ordinal[ordinal] = row
    _require(
        tuple(sorted(by_ordinal)) == tuple(expected_ordinals),
        "policy-v5 source answer ordinals changed",
    )
    return by_ordinal


def _numeric_policy_projection(
    provider_input: Mapping[str, Any],
    relevant_frontier: RelevantNumericFrontier | None = None,
) -> dict[str, Any]:
    if relevant_frontier is not None:
        _require(
            type(relevant_frontier) is RelevantNumericFrontier,
            "numeric policy frontier changed type",
        )
    decision = execute_operator_first_numeric_policy(
        provider_input, relevant_frontier=relevant_frontier
    )
    _require(
        type(decision) is OperatorFirstNumericDecision,
        "operator-first numeric policy returned an unknown decision",
    )
    return decision.projection()


def _replacement_policy_proof(
    plan_row: Mapping[str, Any], completion: str
) -> dict[str, Any]:
    return evaluate_typed_final_replacement_policy_v5(plan_row, completion)


def _validate_numeric_projection(
    raw: Mapping[str, Any], *, allowed_handle_ids: Sequence[str]
) -> dict[str, Any]:
    _require(type(raw) is dict, "numeric policy projection changed type")
    value = dict(raw)
    unsigned = dict(value)
    declared = require_sha256(
        unsigned.pop("receipt_sha256", None), "numeric policy decision"
    )
    status = value.get("status")
    used = value.get("used_handle_ids")
    _require(
        identity_sha256(unsigned) == declared
        and value.get("format") == NUMERIC_POLICY_FORMAT
        and status in _NUMERIC_STATUSES
        and value.get("provider_prompt_count") == 0
        and value.get("retained_transformer_token_state_bytes") == 0
        and type(used) is list
        and len(used) == len(set(used))
        and all(type(handle) is str and bool(handle) for handle in used)
        and set(used) <= set(allowed_handle_ids),
        "numeric policy proof changed",
    )
    if status == ExecutionStatus.SUPPORTED.value:
        _require(
            value.get("decision") == "replace"
            and bool(require_text(value.get("prediction"), "numeric prediction"))
            and bool(used),
            "supported numeric policy result is incomplete",
        )
    else:
        _require(
            value.get("decision") == "abstain"
            and value.get("prediction") == ""
            and used == [],
            "unsupported numeric policy result attempted replacement",
        )
    assert_gold_blind(value, path="full100_policy_v5_numeric_proof")
    return value


def _validate_v5_proof(
    raw: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    completion: str,
) -> dict[str, Any]:
    _require(type(raw) is dict, "validator-v5 proof changed type")
    value = dict(raw)
    unsigned = dict(value)
    declared = require_sha256(
        unsigned.pop("policy_proof_receipt_sha256", None),
        "validator-v5 policy proof",
    )
    parent = require_text(plan.get("parent_prediction"), "policy-v5 parent")
    final = require_text(value.get("final_prediction"), "validator-v5 prediction")
    used = value.get("used_handle_ids")
    accepted = value.get("accepted_replacement")
    _require(
        identity_sha256(unsigned) == declared
        and value.get("format") == _V5_PROOF_FORMAT
        and value.get("gold_loaded") is False
        and value.get("physical_provider_calls") == 0
        and value.get("retained_transformer_token_state_bytes") == 0
        and value.get("prompt_row_receipt_sha256")
        == plan.get("prompt_row_receipt_sha256")
        and value.get("completion_sha256") == quote_sha256(completion)
        and value.get("parent_prediction_sha256") == quote_sha256(parent)
        and value.get("final_prediction_sha256") == quote_sha256(final)
        and value.get("validator_policy_format") == VALIDATOR_POLICY_FORMAT
        and type(accepted) is bool
        and type(used) is list
        and len(used) == len(set(used))
        and all(type(handle) is str and bool(handle) for handle in used)
        and set(used) <= set(plan.get("allowed_handle_ids", ())),
        "validator-v5 policy proof changed",
    )
    _require(
        (
            accepted
            and value.get("decision") == "replace"
            and final != parent
            and bool(used)
        )
        or (
            not accepted
            and value.get("decision") == "keep_parent"
            and final == parent
            and used == []
        ),
        "validator-v5 replacement semantics changed",
    )
    assert_gold_blind(value, path="full100_policy_v5_replacement_proof")
    return value




numeric_policy_projection = _numeric_policy_projection
replacement_policy_proof = _replacement_policy_proof
validate_numeric_projection = _validate_numeric_projection
validate_v5_proof = _validate_v5_proof
validated_completion_records = _validated_completion_records

__all__ = [
    "VALIDATOR_POLICY_FORMAT",
    "authenticated_provider_input",
    "numeric_policy_projection",
    "replacement_policy_proof",
    "validate_numeric_projection",
    "validate_v5_proof",
    "validated_completion_records",
]
