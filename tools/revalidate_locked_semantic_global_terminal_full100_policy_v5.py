#!/usr/bin/env python3
"""Provider-free policy-v5 overlay for the sealed full100 answer run.

The source Terra completions are authenticated through the historical full100
preflight/run/replay loader.  Every terminal completion is then evaluated by
both the deterministic operator-first numeric policy and the asymmetric typed
final validator v5.  A supported numeric result has priority; otherwise v5
may replace only under its own sealed proof.  Passthrough rows remain exact
parents.  This lifecycle has no provider command, no ordinal selector, and no
gold-loading path.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from tools import (  # noqa: E402
    run_locked_full100_numeric_frontier as numeric_frontier_cli,
    run_locked_semantic_global_terminal_full100_answer as answer_cli,
)
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.operator_first_numeric_policy import (  # noqa: E402
    DECISION_FORMAT as NUMERIC_POLICY_FORMAT,
    OperatorFirstNumericDecision,
    RelevantNumericFrontier,
    execute_operator_first_numeric_policy,
)
from tools.matched_eval.typed_memory_final_arm import (  # noqa: E402
    judge_row_projection,
    render_final_messages,
)
from tools.matched_eval.typed_memory_final_validator_v5 import (  # noqa: E402
    FORMAT as VALIDATOR_V5_FORMAT,
    VALIDATOR_POLICY_FORMAT,
    evaluate_typed_final_replacement_policy_v5,
)
from tools.matched_eval.typed_operator_executor import ExecutionStatus  # noqa: E402


FORMAT = "memory-condense-locked-semantic-global-terminal-full100-policy-v5"
RUN_FORMAT = f"{FORMAT}-run-v1"
REPLAY_FORMAT = f"{FORMAT}-replay-v1"
RESULT_ROW_FORMAT = f"{FORMAT}-result-row-v1"
NUMERIC_FRONTIER_BINDING_FORMAT = f"{FORMAT}-numeric-frontier-binding-v1"
RUN_NAME = "semantic-global-terminal-full100-policy-v5.json"
REPLAY_NAME = "semantic-global-terminal-full100-policy-v5-replay.json"
DEFAULT_OUTPUT_ROOT = answer_cli.DEFAULT_OUTPUT_ROOT / "policy-v5"

QUESTION_COUNT = answer_cli.QUESTION_COUNT
TERMINAL_COUNT = answer_cli.ELIGIBLE_COUNT
PASSTHROUGH_COUNT = answer_cli.PASSTHROUGH_COUNT
ALL_ORDINALS = tuple(range(QUESTION_COUNT))

_V5_PROOF_FORMAT = f"{VALIDATOR_V5_FORMAT}-policy-proof-v1"
_NUMERIC_STATUSES = frozenset(status.value for status in ExecutionStatus)


class LockedSemanticGlobalTerminalFull100PolicyV5Error(MatchedEvalContractError):
    """A sealed source or provider-free policy-v5 projection changed."""


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


def _numeric_frontier_population_sha256(
    frontier_by_ordinal: Mapping[int, RelevantNumericFrontier],
) -> str:
    return identity_sha256(
        [
            {"frontier": frontier.projection(), "ordinal": ordinal}
            for ordinal, frontier in sorted(frontier_by_ordinal.items())
        ]
    )


def _numeric_frontier_binding(
    materialization: SealedArtifact,
    replay: SealedArtifact,
    frontier_by_ordinal: Mapping[int, RelevantNumericFrontier],
    *,
    preflight: SealedArtifact,
    source_run: SealedArtifact,
    source_replay: SealedArtifact,
) -> dict[str, Any]:
    """Bind an authenticated frontier lifecycle to this answer generation."""

    payload = materialization.payload
    artifact_format = payload.get("format")
    allowed_artifact_formats = {
        numeric_frontier_cli.FORMAT,
        numeric_frontier_cli.V3_FORMAT,
    }
    frontiers = dict(frontier_by_ordinal)
    _require(
        materialization.sha256 == replay.sha256
        and materialization.payload == replay.payload
        and artifact_format in allowed_artifact_formats
        and payload.get("answer_preflight_artifact_sha256") == preflight.sha256
        and payload.get("answer_run_artifact_sha256") == source_run.sha256
        and payload.get("answer_replay_artifact_sha256") == source_replay.sha256
        and all(
            type(ordinal) is int and type(frontier) is RelevantNumericFrontier
            for ordinal, frontier in frontiers.items()
        ),
        "numeric frontier lifecycle escaped its authenticated answer lineage",
    )
    body = {
        "answer_preflight_artifact_sha256": preflight.sha256,
        "answer_replay_artifact_sha256": source_replay.sha256,
        "answer_run_artifact_sha256": source_run.sha256,
        "artifact_format": artifact_format,
        "format": NUMERIC_FRONTIER_BINDING_FORMAT,
        "frontier_count": len(frontiers),
        "frontier_ordinals": sorted(frontiers),
        "frontier_population_sha256": _numeric_frontier_population_sha256(
            frontiers
        ),
        "lifecycle_identity_sha256": require_sha256(
            payload.get("identity_sha256"), "numeric frontier lifecycle identity"
        ),
        "materialization_artifact_sha256": materialization.sha256,
        "replay_artifact_sha256": replay.sha256,
    }
    return {**body, "receipt_sha256": identity_sha256(body)}


def _validate_numeric_frontier_binding(
    raw: Mapping[str, Any] | None,
    frontier_by_ordinal: Mapping[int, RelevantNumericFrontier],
    *,
    preflight: SealedArtifact,
    source_run: SealedArtifact,
    source_replay: SealedArtifact,
) -> dict[str, Any] | None:
    """Validate a supplied binding without requiring artifacts in unit builders."""

    if raw is None:
        # Direct unit-level callers may still exercise an injected frontier without
        # constructing a filesystem lifecycle.  Every CLI path supplies a binding.
        return None
    _require(type(raw) is dict, "numeric frontier binding changed type")
    value = dict(raw)
    unsigned = dict(value)
    declared = require_sha256(
        unsigned.pop("receipt_sha256", None), "numeric frontier binding"
    )
    frontiers = dict(frontier_by_ordinal)
    _require(
        set(value)
        == {
            "answer_preflight_artifact_sha256",
            "answer_replay_artifact_sha256",
            "answer_run_artifact_sha256",
            "artifact_format",
            "format",
            "frontier_count",
            "frontier_ordinals",
            "frontier_population_sha256",
            "lifecycle_identity_sha256",
            "materialization_artifact_sha256",
            "receipt_sha256",
            "replay_artifact_sha256",
        }
        and identity_sha256(unsigned) == declared
        and value.get("format") == NUMERIC_FRONTIER_BINDING_FORMAT
        and value.get("artifact_format")
        in {numeric_frontier_cli.FORMAT, numeric_frontier_cli.V3_FORMAT}
        and value.get("answer_preflight_artifact_sha256") == preflight.sha256
        and value.get("answer_run_artifact_sha256") == source_run.sha256
        and value.get("answer_replay_artifact_sha256") == source_replay.sha256
        and value.get("frontier_count") == len(frontiers)
        and value.get("frontier_ordinals") == sorted(frontiers)
        and value.get("frontier_population_sha256")
        == _numeric_frontier_population_sha256(frontiers),
        "numeric frontier binding changed",
    )
    for key in (
        "frontier_population_sha256",
        "lifecycle_identity_sha256",
        "materialization_artifact_sha256",
        "replay_artifact_sha256",
    ):
        require_sha256(value.get(key), f"numeric frontier binding {key}")
    assert_gold_blind(value, path="full100_policy_v5_numeric_frontier_binding")
    return value


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


def _terminal_result(
    plan_row: Mapping[str, Any],
    completion: str,
    record: Mapping[str, Any],
    source_answer_row: Mapping[str, Any],
    *,
    relevant_frontier: RelevantNumericFrontier | None = None,
) -> dict[str, Any]:
    plan = _receipt_row(
        plan_row,
        key="prompt_row_receipt_sha256",
        label="policy-v5 terminal prompt row",
    )
    provider_input = authenticated_provider_input(plan)
    numeric = _validate_numeric_projection(
        _numeric_policy_projection(provider_input, relevant_frontier),
        allowed_handle_ids=tuple(plan.get("allowed_handle_ids", ())),
    )
    v5 = _validate_v5_proof(
        _replacement_policy_proof(plan, completion),
        plan=plan,
        completion=completion,
    )
    parent = require_text(plan.get("parent_prediction"), "policy-v5 parent")
    numeric_supported = numeric["status"] == ExecutionStatus.SUPPORTED.value
    if numeric_supported:
        prediction = str(numeric["prediction"])
        used = list(numeric["used_handle_ids"])
        selected_policy = "operator_first_numeric"
        source = "operator_first_numeric_supported_v1"
    elif v5["accepted_replacement"]:
        prediction = str(v5["final_prediction"])
        used = list(v5["used_handle_ids"])
        selected_policy = "typed_final_validator_v5"
        source = "typed_final_validator_v5_accepted_replacement_v1"
    else:
        prediction = parent
        used = []
        selected_policy = "protected_parent"
        source = "typed_final_validator_v5_keep_parent_v1"

    source_prediction = require_text(
        source_answer_row.get("prediction"), "policy-v5 source answer prediction"
    )
    _require(
        plan.get("ordinal") == source_answer_row.get("ordinal")
        and plan.get("question_id") == source_answer_row.get("question_id")
        and plan.get("question_sha256") == source_answer_row.get("question_sha256")
        and plan.get("dated_question_sha256")
        == source_answer_row.get("dated_question_sha256")
        and plan.get("parent_prediction_sha256")
        == source_answer_row.get("parent_prediction_sha256")
        == quote_sha256(parent),
        "policy-v5 terminal source binding changed",
    )
    body = {
        "answer_mode": plan.get("answer_mode"),
        "changed_from_parent": prediction != parent,
        "changed_from_source_answer": prediction != source_prediction,
        "completion_sha256": quote_sha256(completion),
        "dated_question_sha256": plan["dated_question_sha256"],
        "decision": "replace" if prediction != parent else "keep_parent",
        "format": RESULT_ROW_FORMAT,
        "gold_loaded": False,
        "numeric_policy_proof": numeric,
        "ordinal": plan["ordinal"],
        "parent_prediction_sha256": quote_sha256(parent),
        "physical_provider_calls": 0,
        "policy_v5_proof": v5,
        "prediction": prediction,
        "prediction_sha256": quote_sha256(prediction),
        "prediction_source": source,
        "prompt_row_receipt_sha256": plan["prompt_row_receipt_sha256"],
        "question_id": plan["question_id"],
        "question_sha256": plan["question_sha256"],
        "retained_transformer_token_state_bytes": 0,
        "route_id": plan["route_id"],
        "selected_policy": selected_policy,
        "source_answer_prediction_sha256": quote_sha256(source_prediction),
        "source_answer_row_sha256": source_answer_row["source_row_sha256"],
        "source_completion_record_sha256": identity_sha256(dict(record)),
        "used_handle_ids": used,
        "validator_policy_format": VALIDATOR_POLICY_FORMAT,
    }
    assert_gold_blind(body, path="full100_policy_v5_terminal_result")
    return {**body, "source_row_sha256": identity_sha256(body)}


def _passthrough_result(
    plan_row: Mapping[str, Any], source_answer_row: Mapping[str, Any]
) -> dict[str, Any]:
    plan = _receipt_row(
        plan_row,
        key="passthrough_plan_receipt_sha256",
        label="policy-v5 passthrough plan row",
    )
    parent = require_text(plan.get("parent_prediction"), "policy-v5 passthrough")
    source_prediction = require_text(
        source_answer_row.get("prediction"), "policy-v5 passthrough source"
    )
    _require(
        plan.get("ordinal") == source_answer_row.get("ordinal")
        and plan.get("question_id") == source_answer_row.get("question_id")
        and plan.get("question_sha256") == source_answer_row.get("question_sha256")
        and plan.get("dated_question_sha256")
        == source_answer_row.get("dated_question_sha256")
        and plan.get("prediction") == parent == source_prediction
        and plan.get("prediction_sha256")
        == plan.get("parent_prediction_sha256")
        == source_answer_row.get("prediction_sha256")
        == source_answer_row.get("parent_prediction_sha256")
        == quote_sha256(parent),
        "policy-v5 passthrough is not an exact parent",
    )
    body = {
        "answer_mode": plan.get("answer_mode"),
        "changed_from_parent": False,
        "changed_from_source_answer": False,
        "completion_sha256": None,
        "dated_question_sha256": plan["dated_question_sha256"],
        "decision": "passthrough",
        "format": RESULT_ROW_FORMAT,
        "gold_loaded": False,
        "numeric_policy_proof": None,
        "ordinal": plan["ordinal"],
        "parent_prediction_sha256": quote_sha256(parent),
        "physical_provider_calls": 0,
        "policy_v5_proof": None,
        "prediction": parent,
        "prediction_sha256": quote_sha256(parent),
        "prediction_source": "sealed_v3_byte_exact_passthrough_v1",
        "prompt_row_receipt_sha256": None,
        "question_id": plan["question_id"],
        "question_sha256": plan["question_sha256"],
        "retained_transformer_token_state_bytes": 0,
        "route_id": plan["route_id"],
        "selected_policy": "passthrough",
        "source_answer_prediction_sha256": quote_sha256(source_prediction),
        "source_answer_row_sha256": source_answer_row["source_row_sha256"],
        "source_completion_record_sha256": None,
        "used_handle_ids": [],
        "validator_policy_format": VALIDATOR_POLICY_FORMAT,
    }
    assert_gold_blind(body, path="full100_policy_v5_passthrough_result")
    return {**body, "source_row_sha256": identity_sha256(body)}


def _build_overlay_rows(
    source_run: SealedArtifact,
    plans: Sequence[Mapping[str, Any]],
    passthroughs: Sequence[Mapping[str, Any]],
    *,
    expected_ordinals: Sequence[int],
    frontier_by_ordinal: Mapping[int, RelevantNumericFrontier] | None = None,
) -> tuple[dict[str, Any], ...]:
    expected = tuple(expected_ordinals)
    _require(
        len(expected) == len(set(expected)) and tuple(sorted(expected)) == expected,
        "policy-v5 expected ordinal population changed",
    )
    plan_ordinals = tuple(plan.get("ordinal") for plan in plans)
    passthrough_ordinals = tuple(plan.get("ordinal") for plan in passthroughs)
    _require(
        len(plan_ordinals) == len(set(plan_ordinals))
        and len(passthrough_ordinals) == len(set(passthrough_ordinals))
        and set(plan_ordinals).isdisjoint(passthrough_ordinals)
        and set(plan_ordinals).union(passthrough_ordinals) == set(expected),
        "policy-v5 terminal/passthrough population changed",
    )
    frontiers = {} if frontier_by_ordinal is None else dict(frontier_by_ordinal)
    _require(
        all(
            type(ordinal) is int
            and ordinal in set(plan_ordinals)
            and type(frontier) is RelevantNumericFrontier
            for ordinal, frontier in frontiers.items()
        ),
        "policy-v5 numeric frontier population changed",
    )
    source_by_ordinal = _validated_source_questions(source_run, expected)
    completions = _validated_completion_records(source_run, plans)
    results: dict[int, dict[str, Any]] = {}
    for plan, (completion, record) in zip(plans, completions, strict=True):
        ordinal = int(plan["ordinal"])
        results[ordinal] = _terminal_result(
            plan,
            completion,
            record,
            source_by_ordinal[ordinal],
            relevant_frontier=frontiers.get(ordinal),
        )
    for plan in passthroughs:
        ordinal = int(plan["ordinal"])
        results[ordinal] = _passthrough_result(plan, source_by_ordinal[ordinal])
    _require(set(results) == set(expected), "policy-v5 result population is incomplete")
    return tuple(results[ordinal] for ordinal in expected)


def build_materialization_payload(
    preflight: SealedArtifact,
    source_run: SealedArtifact,
    source_replay: SealedArtifact,
    plans: Sequence[Mapping[str, Any]],
    passthroughs: Sequence[Mapping[str, Any]],
    *,
    frontier_by_ordinal: Mapping[int, RelevantNumericFrontier] | None = None,
    frontier_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exact zero-provider full100 policy-v5 overlay."""

    eligible = tuple(plan.get("ordinal") for plan in plans)
    bypassed = tuple(plan.get("ordinal") for plan in passthroughs)
    _require(
        len(plans) == TERMINAL_COUNT
        and len(passthroughs) == PASSTHROUGH_COUNT
        and tuple(preflight.payload.get("eligible_ordinals", ())) == eligible
        and tuple(preflight.payload.get("passthrough_ordinals", ())) == bypassed
        and set(eligible).isdisjoint(bypassed)
        and set(eligible).union(bypassed) == set(ALL_ORDINALS)
        and source_run.payload.get("preflight_artifact_sha256") == preflight.sha256
        and source_replay.payload.get("expected_run_sha256") == source_run.sha256
        and source_replay.payload.get("replayed_run_sha256") == source_run.sha256,
        "policy-v5 full100 source bindings changed",
    )
    frontiers = {} if frontier_by_ordinal is None else dict(frontier_by_ordinal)
    bound_frontier = _validate_numeric_frontier_binding(
        frontier_binding,
        frontiers,
        preflight=preflight,
        source_run=source_run,
        source_replay=source_replay,
    )
    results = _build_overlay_rows(
        source_run,
        plans,
        passthroughs,
        expected_ordinals=ALL_ORDINALS,
        frontier_by_ordinal=frontiers,
    )
    judge_rows = tuple(judge_row_projection(row) for row in results)
    _require(
        len(results) == len(judge_rows) == QUESTION_COUNT
        and tuple(row["ordinal"] for row in results) == ALL_ORDINALS,
        "policy-v5 judge population/order changed",
    )
    payload = {
        "caller_ordinal_routing_available": False,
        "changed_from_parent_count": sum(
            bool(row["changed_from_parent"]) for row in results
        ),
        "changed_from_source_count": sum(
            bool(row["changed_from_source_answer"]) for row in results
        ),
        "changed_prediction_count": sum(
            bool(row["changed_from_parent"]) for row in results
        ),
        "changed_prediction_count_basis": "protected_parent",
        "format": RUN_FORMAT,
        "gold_loaded": False,
        "judge_rows": list(judge_rows),
        "numeric_frontier_binding": bound_frontier,
        "numeric_policy_format": NUMERIC_POLICY_FORMAT,
        "numeric_supported_count": sum(
            row["selected_policy"] == "operator_first_numeric" for row in results
        ),
        "passthrough_count": PASSTHROUGH_COUNT,
        "physical_provider_calls_during_revalidation": 0,
        "provider_execution_command_available": False,
        "question_count": QUESTION_COUNT,
        "questions": list(results),
        "retained_transformer_token_state_bytes": 0,
        "source_answer_preflight_artifact_sha256": preflight.sha256,
        "source_answer_replay_artifact_sha256": source_replay.sha256,
        "source_answer_run_artifact_sha256": source_run.sha256,
        "source_completion_batch_sha256": identity_sha256(
            source_run.payload["completion_batch"]
        ),
        "source_question_population_sha256": require_sha256(
            preflight.payload.get("source_question_population_sha256"),
            "policy-v5 source question population",
        ),
        "terminal_count": TERMINAL_COUNT,
        "typed_final_v5_replacement_count": sum(
            row["selected_policy"] == "typed_final_validator_v5" for row in results
        ),
        "validator_policy_format": VALIDATOR_POLICY_FORMAT,
    }
    assert_gold_blind(payload, path="full100_policy_v5_run")
    return payload


def _validate_run(
    artifact: SealedArtifact, *, expected_payload: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    payload = artifact.payload
    questions = payload.get("questions")
    judge_rows = payload.get("judge_rows")
    _require(
        payload == dict(expected_payload)
        and payload.get("format") == RUN_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("physical_provider_calls_during_revalidation") == 0
        and payload.get("provider_execution_command_available") is False
        and payload.get("caller_ordinal_routing_available") is False
        and payload.get("changed_prediction_count_basis") == "protected_parent"
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("terminal_count") == TERMINAL_COUNT
        and payload.get("passthrough_count") == PASSTHROUGH_COUNT
        and type(questions) is list
        and type(judge_rows) is list
        and len(questions) == len(judge_rows) == QUESTION_COUNT,
        "policy-v5 run envelope changed",
    )
    validated: list[dict[str, Any]] = []
    for ordinal, row, projected in zip(
        ALL_ORDINALS, questions, judge_rows, strict=True
    ):
        _require(
            type(row) is dict and type(projected) is dict,
            "policy-v5 result row changed type",
        )
        unsigned = dict(row)
        declared = require_sha256(
            unsigned.pop("source_row_sha256", None), "policy-v5 result row"
        )
        prediction = require_text(row.get("prediction"), "policy-v5 prediction")
        _require(
            row.get("format") == RESULT_ROW_FORMAT
            and row.get("ordinal") == ordinal
            and identity_sha256(unsigned) == declared
            and row.get("prediction_sha256") == quote_sha256(prediction)
            and type(row.get("changed_from_source_answer")) is bool
            and row.get("changed_from_source_answer")
            == (
                row.get("prediction_sha256")
                != row.get("source_answer_prediction_sha256")
            )
            and row.get("physical_provider_calls") == 0
            and row.get("gold_loaded") is False
            and judge_row_projection(row) == projected,
            f"policy-v5 result row {ordinal} changed",
        )
        validated.append(dict(projected))
    _require(
        payload.get("changed_prediction_count")
        == sum(bool(row["changed_from_parent"]) for row in questions)
        and payload.get("changed_from_parent_count")
        == payload.get("changed_prediction_count")
        and payload.get("changed_from_source_count")
        == sum(bool(row["changed_from_source_answer"]) for row in questions)
        and payload.get("numeric_supported_count")
        == sum(
            row["selected_policy"] == "operator_first_numeric" for row in questions
        )
        and payload.get("typed_final_v5_replacement_count")
        == sum(
            row["selected_policy"] == "typed_final_validator_v5"
            for row in questions
        ),
        "policy-v5 aggregate counts changed",
    )
    assert_gold_blind(payload, path="full100_policy_v5_validated_run")
    return tuple(validated)


def _read_source(
    args: argparse.Namespace,
) -> tuple[
    SealedArtifact,
    SealedArtifact,
    SealedArtifact,
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    root = Path(args.answer_root)
    preflight, _prompts, plans, passthroughs = answer_cli._read_preflight(  # noqa: SLF001
        root, str(args.expected_answer_preflight_sha256)
    )
    source_run, source_replay, judge_rows = answer_cli.load_verified_answer_run(
        root,
        expected_preflight_sha256=str(args.expected_answer_preflight_sha256),
        expected_run_sha256=str(args.expected_answer_run_sha256),
        expected_replay_sha256=str(args.expected_answer_replay_sha256),
        postseal_audit=args.postseal_audit,
        expected_postseal_audit_sha256=str(
            args.expected_postseal_audit_sha256
        ),
    )
    _require(
        source_run.payload.get("preflight_artifact_sha256") == preflight.sha256
        and len(judge_rows) == QUESTION_COUNT,
        "policy-v5 source loader population changed",
    )
    return preflight, source_run, source_replay, plans, passthroughs


def _read_numeric_frontiers(
    args: argparse.Namespace,
    *,
    preflight: SealedArtifact,
    source_run: SealedArtifact,
    source_replay: SealedArtifact,
) -> tuple[Mapping[int, RelevantNumericFrontier], dict[str, Any]]:
    profile = str(
        getattr(
            args,
            "numeric_frontier_profile",
            numeric_frontier_cli.STRICT_PROFILE,
        )
    )
    if profile == numeric_frontier_cli.STRICT_PROFILE:
        materialization, replay, frontiers = (
            numeric_frontier_cli.load_verified_numeric_frontiers(
                Path(args.numeric_frontier_root),
                str(args.expected_numeric_frontier_materialization_sha256),
                str(args.expected_numeric_frontier_replay_sha256),
            )
        )
    else:
        materialization, replay, frontiers = (
            numeric_frontier_cli.load_verified_numeric_frontiers(
                Path(args.numeric_frontier_root),
                str(args.expected_numeric_frontier_materialization_sha256),
                str(args.expected_numeric_frontier_replay_sha256),
                policy_profile=profile,
            )
        )
    binding = _numeric_frontier_binding(
        materialization,
        replay,
        frontiers,
        preflight=preflight,
        source_run=source_run,
        source_replay=source_replay,
    )
    return frontiers, binding


def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    preflight, source_run, source_replay, plans, passthroughs = _read_source(args)
    frontiers, frontier_binding = _read_numeric_frontiers(
        args,
        preflight=preflight,
        source_run=source_run,
        source_replay=source_replay,
    )
    payload = build_materialization_payload(
        preflight,
        source_run,
        source_replay,
        plans,
        passthroughs,
        frontier_by_ordinal=frontiers,
        frontier_binding=frontier_binding,
    )
    artifact, created = publish_sealed_json(Path(args.output_root) / RUN_NAME, payload)
    _validate_run(artifact, expected_payload=payload)
    return {
        "changed_from_parent_count": payload["changed_from_parent_count"],
        "changed_from_source_count": payload["changed_from_source_count"],
        "changed_prediction_count": payload["changed_prediction_count"],
        "changed_prediction_count_basis": payload[
            "changed_prediction_count_basis"
        ],
        "created": created,
        "gold_loaded": False,
        "numeric_frontier_binding_receipt_sha256": frontier_binding[
            "receipt_sha256"
        ],
        "numeric_frontier_materialization_sha256": frontier_binding[
            "materialization_artifact_sha256"
        ],
        "numeric_frontier_replay_sha256": frontier_binding[
            "replay_artifact_sha256"
        ],
        "numeric_supported_count": payload["numeric_supported_count"],
        "physical_provider_calls": 0,
        "run_sha256": artifact.sha256,
        "source_answer_run_sha256": source_run.sha256,
        "typed_final_v5_replacement_count": payload[
            "typed_final_v5_replacement_count"
        ],
    }


def _replay_payload(run: SealedArtifact, expected: Mapping[str, Any]) -> dict[str, Any]:
    frontier_binding = expected.get("numeric_frontier_binding")
    _require(
        frontier_binding is None or type(frontier_binding) is dict,
        "policy-v5 replay numeric frontier binding changed type",
    )
    body = {
        "byte_identical": run.payload == dict(expected),
        "expected_run_sha256": run.sha256,
        "format": REPLAY_FORMAT,
        "gold_loaded": False,
        "numeric_frontier_binding": frontier_binding,
        "numeric_policy_format": NUMERIC_POLICY_FORMAT,
        "physical_provider_calls": 0,
        "replayed_run_sha256": run.sha256,
        "retained_transformer_token_state_bytes": 0,
        "source_answer_preflight_artifact_sha256": expected[
            "source_answer_preflight_artifact_sha256"
        ],
        "source_answer_replay_artifact_sha256": expected[
            "source_answer_replay_artifact_sha256"
        ],
        "source_answer_run_artifact_sha256": expected[
            "source_answer_run_artifact_sha256"
        ],
        "source_completion_batch_sha256": expected[
            "source_completion_batch_sha256"
        ],
        "validator_policy_format": VALIDATOR_POLICY_FORMAT,
    }
    assert_gold_blind(body, path="full100_policy_v5_replay")
    return body


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    preflight, source_run, source_replay, plans, passthroughs = _read_source(args)
    frontiers, frontier_binding = _read_numeric_frontiers(
        args,
        preflight=preflight,
        source_run=source_run,
        source_replay=source_replay,
    )
    expected = build_materialization_payload(
        preflight,
        source_run,
        source_replay,
        plans,
        passthroughs,
        frontier_by_ordinal=frontiers,
        frontier_binding=frontier_binding,
    )
    run = read_sealed_json(Path(args.output_root) / RUN_NAME)
    _require(
        run.sha256
        == require_sha256(args.expected_policy_run_sha256, "policy-v5 answer run"),
        "policy-v5 run artifact changed",
    )
    _validate_run(run, expected_payload=expected)
    replay_payload = _replay_payload(run, expected)
    _require(replay_payload["byte_identical"] is True, "policy-v5 replay is not exact")
    replay, created = publish_sealed_json(
        Path(args.output_root) / REPLAY_NAME, replay_payload
    )
    return {
        "byte_identical": True,
        "created": created,
        "numeric_frontier_binding_receipt_sha256": frontier_binding[
            "receipt_sha256"
        ],
        "physical_provider_calls": 0,
        "replay_sha256": replay.sha256,
        "run_sha256": run.sha256,
    }


def load_verified_policy_v5_run(
    output_root: str | Path,
    *,
    answer_root: str | Path,
    expected_answer_preflight_sha256: str,
    expected_answer_run_sha256: str,
    expected_answer_replay_sha256: str,
    postseal_audit: str | Path,
    expected_postseal_audit_sha256: str,
    expected_policy_run_sha256: str,
    expected_policy_replay_sha256: str,
    numeric_frontier_root: str | Path,
    expected_numeric_frontier_materialization_sha256: str,
    expected_numeric_frontier_replay_sha256: str,
    numeric_frontier_profile: str = numeric_frontier_cli.STRICT_PROFILE,
) -> tuple[SealedArtifact, SealedArtifact, tuple[dict[str, Any], ...]]:
    """Authenticate both generations and return the stable 100-row judge seam."""

    args = argparse.Namespace(
        answer_root=Path(answer_root),
        expected_answer_preflight_sha256=expected_answer_preflight_sha256,
        expected_answer_run_sha256=expected_answer_run_sha256,
        expected_answer_replay_sha256=expected_answer_replay_sha256,
        postseal_audit=Path(postseal_audit),
        expected_postseal_audit_sha256=expected_postseal_audit_sha256,
        numeric_frontier_root=Path(numeric_frontier_root),
        expected_numeric_frontier_materialization_sha256=(
            expected_numeric_frontier_materialization_sha256
        ),
        expected_numeric_frontier_replay_sha256=(
            expected_numeric_frontier_replay_sha256
        ),
        numeric_frontier_profile=numeric_frontier_profile,
    )
    preflight, source_run, source_replay, plans, passthroughs = _read_source(args)
    frontiers, frontier_binding = _read_numeric_frontiers(
        args,
        preflight=preflight,
        source_run=source_run,
        source_replay=source_replay,
    )
    expected = build_materialization_payload(
        preflight,
        source_run,
        source_replay,
        plans,
        passthroughs,
        frontier_by_ordinal=frontiers,
        frontier_binding=frontier_binding,
    )
    root = Path(output_root)
    run = read_sealed_json(root / RUN_NAME)
    _require(
        run.sha256
        == require_sha256(expected_policy_run_sha256, "policy-v5 answer run"),
        "policy-v5 run artifact changed",
    )
    judge_rows = _validate_run(run, expected_payload=expected)
    replay = read_sealed_json(root / REPLAY_NAME)
    replay_expected = _replay_payload(run, expected)
    _require(
        replay.sha256
        == require_sha256(expected_policy_replay_sha256, "policy-v5 answer replay")
        and replay.payload == replay_expected
        and replay.payload.get("byte_identical") is True,
        "policy-v5 replay artifact changed",
    )
    return run, replay, judge_rows


def _add_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--answer-root", type=Path, required=True)
    parser.add_argument("--expected-answer-preflight-sha256", required=True)
    parser.add_argument("--expected-answer-run-sha256", required=True)
    parser.add_argument("--expected-answer-replay-sha256", required=True)
    parser.add_argument("--postseal-audit", type=Path, required=True)
    parser.add_argument("--expected-postseal-audit-sha256", required=True)
    parser.add_argument("--numeric-frontier-root", type=Path, required=True)
    parser.add_argument(
        "--expected-numeric-frontier-materialization-sha256", required=True
    )
    parser.add_argument("--expected-numeric-frontier-replay-sha256", required=True)
    parser.add_argument(
        "--numeric-frontier-profile",
        choices=(
            numeric_frontier_cli.STRICT_PROFILE,
            numeric_frontier_cli.OPERATOR_MATERIAL_PROFILE,
        ),
        default=numeric_frontier_cli.STRICT_PROFILE,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    materialize = commands.add_parser("materialize")
    _add_sources(materialize)
    replay = commands.add_parser("replay")
    _add_sources(replay)
    replay.add_argument("--expected-policy-run-sha256", required=True)
    return parser


def _canonical_output(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_materialize(args) if args.command == "materialize" else run_replay(args)
    print(_canonical_output(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
