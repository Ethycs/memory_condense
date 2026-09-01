#!/usr/bin/env python3
"""Run the locked full-100 Terra answer lifecycle for the P/R/L/G terminal.

The provider population is not selectable by the operator.  It is derived by
the strict full100 construction reader as exactly 68 terminal plans and 32
byte-exact V3 passthroughs.  Promotion authority comes from the independently
sealed exact11 semantic-atom audit, and transfers only after the eleven exact
provider-plan projections are byte-identical inside the full100 population.

No phase loads benchmark gold.  Preflight and release require fresh checkpoint
state; provider-run accepts exactly the remaining number of calls; materialize
and replay are checkpoint-only.  The final artifact always contains one row
for each ordinal 0..99 and exposes a stable gold-free judge seam.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from dotenv import load_dotenv  # noqa: E402

from memory_condense.domain._tokenizer import (  # noqa: E402
    count_chat_prompt_token_proxy,
)
from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from memory_condense.eval.fast_completion_runtime import (  # noqa: E402
    FastCompletionBatch,
    FastCompletionRuntime,
    preflight_fast_completion_prompts,
)
from tools import audit_locked_semantic_global_terminal_postseal as postseal_cli  # noqa: E402
from tools import run_locked_semantic_global_terminal_answer as exact_answer_cli  # noqa: E402
from tools import (  # noqa: E402
    run_locked_semantic_global_terminal_full100_construction as full100_cli,
)
from tools import run_reduced_semantic_global_terminal_assay as terminal_cli  # noqa: E402
from tools.matched_eval import live  # noqa: E402
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
from tools.matched_eval.typed_memory_final_arm import (  # noqa: E402
    HARD_PROMPT_TOKEN_CAP,
    MAX_CHAT_PROMPT_TOKENS,
    OUTPUT_TOKEN_RESERVE,
    RESULT_ROW_FORMAT,
    VALIDATOR_POLICY_FORMAT,
    judge_row_projection,
    materialize_typed_final_result_row,
    render_final_messages,
)


FORMAT = "memory-condense-locked-semantic-global-terminal-full100-terra-answer-v1"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight-v1"
RELEASE_FORMAT = f"{FORMAT}-provider-release-v1"
RUN_FORMAT = f"{FORMAT}-run-v1"
REPLAY_FORMAT = f"{FORMAT}-replay-v1"
PROMPT_ROW_FORMAT = f"{FORMAT}-prompt-row-v1"
PASSTHROUGH_ROW_FORMAT = f"{FORMAT}-passthrough-plan-row-v1"

PREFLIGHT_NAME = "semantic-global-terminal-full100-terra-answer-preflight-v1.json"
RELEASE_NAME = "semantic-global-terminal-full100-terra-answer-provider-release-v1.json"
RUN_NAME = "semantic-global-terminal-full100-terra-answer-v1.json"
REPLAY_NAME = "semantic-global-terminal-full100-terra-answer-replay-v1.json"
CHECKPOINT_DIR_NAME = "terra-semantic-global-terminal-full100-v1-calls"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/"
    "locked-semantic-global-terminal-full100-terra-answer-v1"
)
DEFAULT_MODEL = live.DEFAULT_TERRA_GATEWAY_MODEL
DEFAULT_GATEWAY_URL = live.DEFAULT_GATEWAY_URL
DEFAULT_MAX_CONCURRENCY = 4

QUESTION_COUNT = full100_cli.QUESTION_COUNT
ELIGIBLE_COUNT = full100_cli.ELIGIBLE_COUNT
PASSTHROUGH_COUNT = full100_cli.PASSTHROUGH_COUNT
ALL_ORDINALS = tuple(range(QUESTION_COUNT))
EXACT_ORDINALS = terminal_cli.EXACT_ORDINALS
TERMINAL_MODE = full100_cli.TERMINAL_MODE
PASSTHROUGH_MODE = full100_cli.PASSTHROUGH_MODE
TERMINAL_ROUTE_ID = exact_answer_cli.ROUTE_ID
PASSTHROUGH_ROUTE_ID = "semantic-global-terminal-full100-v3-passthrough-v1"
_JOURNAL_FILENAME_RE = re.compile(
    r"^(?P<key>[0-9a-f]{64})\.(?P<kind>request|response)\.json$"
)

POSTSEAL_BINDING_KEYS = exact_answer_cli.POSTSEAL_BINDING_KEYS
SOURCE_BINDING_KEYS = (
    "answer_plan_population_sha256",
    "full100_construction_artifact_sha256",
    "full100_replay_artifact_sha256",
    "promotion_terminal_construction_artifact_sha256",
    "promotion_terminal_replay_artifact_sha256",
    "prompt_population_sha256",
    "source_question_population_sha256",
    "passthrough_population_sha256",
    *POSTSEAL_BINDING_KEYS,
)
PREFLIGHT_KEYS = {
    "answer_plan_population_sha256",
    "eligible_count",
    "eligible_ordinals",
    "format",
    "full100_construction_artifact_sha256",
    "full100_replay_artifact_sha256",
    "gateway_url",
    "gold_loaded",
    "hard_prompt_token_cap",
    "logical_answer_count",
    "max_chat_prompt_tokens",
    "max_concurrency",
    "model",
    "observed_max_complete_envelope_tokens",
    "ordinal_cli_routing_available",
    "output_token_reserve",
    "passthrough_count",
    "passthrough_ordinals",
    "passthrough_plan_rows",
    "passthrough_population_sha256",
    "physical_prompt_rows",
    "production_ordinal_routing_enabled",
    "prompt_population",
    "prompt_population_sha256",
    "promotion_exact_ordinals",
    "promotion_terminal_construction_artifact_sha256",
    "promotion_terminal_replay_artifact_sha256",
    "provider_calls",
    "question_count",
    "required_authorized_provider_calls",
    "retained_transformer_token_state_bytes",
    "source_question_population_sha256",
}.union(POSTSEAL_BINDING_KEYS)
RELEASE_KEYS = {
    "answer_output_root",
    "answer_output_root_sha256",
    "approval_opt_in",
    "eligible_count",
    "format",
    "full100_terminal_root",
    "full100_terminal_root_sha256",
    "gateway_url",
    "gold_loaded",
    "max_concurrency",
    "model",
    "passthrough_count",
    "preflight_artifact_sha256",
    "promotion_terminal_root",
    "promotion_terminal_root_sha256",
    "provider_calls_during_release",
    "question_count",
    "release_identity_sha256",
    "release_status",
    "required_authorized_provider_calls",
    "retained_transformer_token_state_bytes",
}.union(SOURCE_BINDING_KEYS)
RUN_KEYS = {
    "changed_prediction_count",
    "completion_batch",
    "eligible_count",
    "eligible_ordinals",
    "format",
    "gold_loaded",
    "invalid_completion_parent_fallback_count",
    "judge_rows",
    "passthrough_count",
    "passthrough_ordinals",
    "physical_provider_calls_during_materialization",
    "preflight_artifact_sha256",
    "question_count",
    "questions",
    "release_authorization_artifact_sha256",
    "required_authorized_provider_calls",
    "retained_transformer_token_state_bytes",
    "validator_policy_format",
}.union(SOURCE_BINDING_KEYS)
REPLAY_KEYS = {
    "byte_identical",
    "expected_run_sha256",
    "format",
    "gold_loaded",
    "physical_provider_calls",
    "preflight_artifact_sha256",
    "replayed_run_sha256",
    "release_authorization_artifact_sha256",
    "retained_transformer_token_state_bytes",
}.union(SOURCE_BINDING_KEYS)
RESULT_KEYS = {
    "answer_mode",
    "call_key_sha256",
    "changed_from_parent",
    "completion_receipt_sha256",
    "dated_question_sha256",
    "decision",
    "format",
    "full100_question_construction_receipt_sha256",
    "ordinal",
    "parent_answer_row_sha256",
    "parent_prediction_sha256",
    "parse_error_code",
    "parse_receipt_sha256",
    "prediction",
    "prediction_sha256",
    "prediction_source",
    "prompt_row_receipt_sha256",
    "question_id",
    "question_sha256",
    "request_journal_sha256",
    "response_journal_sha256",
    "retained_transformer_token_state_bytes",
    "route_id",
    "solver_valid",
    "source_row_sha256",
    "terminal_answer_plan_receipt_sha256",
    "used_handle_ids",
    "validation_basis",
    "validator_policy_format",
}
TERMINAL_REPLACEMENT_SOURCES = {
    "typed_final_deterministic_validated_replacement_v1",
    "typed_final_model_attested_replacement_v1",
    "typed_final_scalar_validated_replacement_v1",
}


class LockedSemanticGlobalTerminalFull100AnswerError(MatchedEvalContractError):
    """A population, promotion gate, prompt, checkpoint, or answer changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedSemanticGlobalTerminalFull100AnswerError(message)


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact array")
    return value  # type: ignore[return-value]


def _canonical_root(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve(strict=False)))


def _validate_receipt(value: object, *, key: str, label: str) -> dict[str, Any]:
    row = _exact_dict(value, label)
    declared = require_sha256(row.get(key), label)
    unsigned = {name: child for name, child in row.items() if name != key}
    _require(declared == identity_sha256(unsigned), f"{label} receipt changed")
    return row


def _promotion_binding(artifact: SealedArtifact) -> dict[str, Any]:
    try:
        return exact_answer_cli._postseal_promotion_binding(artifact)  # noqa: SLF001
    except MatchedEvalContractError as exc:
        raise LockedSemanticGlobalTerminalFull100AnswerError(
            "full100 semantic-atom promotion binding changed"
        ) from exc


def _read_promotion_audit(
    path: str | Path,
    expected_sha256: str,
    *,
    construction_sha256: str,
    replay_sha256: str,
) -> SealedArtifact:
    try:
        artifact = postseal_cli.load_verified_promotion_audit(
            path,
            expected_sha256,
            expected_terminal_construction_sha256=construction_sha256,
            expected_terminal_replay_sha256=replay_sha256,
        )
    except MatchedEvalContractError as exc:
        raise LockedSemanticGlobalTerminalFull100AnswerError(
            "full100 promotion audit is absent, invalid, or not promoted"
        ) from exc
    _promotion_binding(artifact)
    return artifact


@dataclass(frozen=True, slots=True)
class _VerifiedSources:
    full100_construction: SealedArtifact
    full100_replay: SealedArtifact
    provider_plans: tuple[dict[str, Any], ...]
    passthroughs: tuple[dict[str, Any], ...]
    promotion_construction: SealedArtifact
    promotion_replay: SealedArtifact
    promotion_plans: tuple[dict[str, Any], ...]
    promotion_audit: SealedArtifact


def _load_verified_sources(args: argparse.Namespace) -> _VerifiedSources:
    construction, replay, provider_plans, passthroughs = (
        full100_cli.load_verified_full100_construction(
            args.full100_terminal_root,
            str(args.expected_full100_construction_sha256),
            str(args.expected_full100_replay_sha256),
        )
    )
    promotion_construction, promotion_replay, promotion_plans = (
        terminal_cli.load_verified_terminal_assay(
            args.promotion_terminal_root,
            str(args.expected_promotion_terminal_construction_sha256),
            str(args.expected_promotion_terminal_replay_sha256),
        )
    )
    promotion_audit = _read_promotion_audit(
        args.postseal_audit,
        str(args.expected_postseal_audit_sha256),
        construction_sha256=promotion_construction.sha256,
        replay_sha256=promotion_replay.sha256,
    )
    return _VerifiedSources(
        full100_construction=construction,
        full100_replay=replay,
        provider_plans=provider_plans,
        passthroughs=passthroughs,
        promotion_construction=promotion_construction,
        promotion_replay=promotion_replay,
        promotion_plans=promotion_plans,
        promotion_audit=promotion_audit,
    )


def _plain_messages(value: object, label: str) -> tuple[dict[str, str], ...]:
    rows = _exact_list(value, label)
    messages: list[dict[str, str]] = []
    for raw in rows:
        _require(
            type(raw) is dict
            and set(raw) == {"role", "content"}
            and raw.get("role") in {"system", "user", "assistant"}
            and type(raw.get("content")) is str,
            f"{label} changed message schema",
        )
        messages.append({"role": raw["role"], "content": raw["content"]})
    return tuple(messages)


def _validate_provider_plan(
    raw: Mapping[str, Any], source_row: Mapping[str, Any]
) -> dict[str, Any]:
    _require(type(raw) is dict, "full100 provider plan changed type")
    plan = dict(raw)
    provider_input = _exact_dict(plan.get("provider_input"), "provider input")
    allowed = _exact_list(plan.get("allowed_handle_ids"), "allowed handles")
    groups = _exact_dict(plan.get("handle_group_by_id"), "handle groups")
    bindings = _exact_dict(
        plan.get("source_artifact_bindings"), "provider source bindings"
    )
    dated_question = require_text(plan.get("dated_question"), "dated question")
    parent = require_text(plan.get("parent_prediction"), "parent prediction")
    messages = render_final_messages(provider_input)
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    _require(
        require_sha256(plan.get("answer_plan_receipt_sha256"), "answer plan")
        and require_sha256(
            plan.get("terminal_compilation_receipt_sha256"),
            "terminal compilation",
        )
        and plan.get("ordinal") == source_row.get("ordinal")
        and plan.get("question_id") == source_row.get("question_id")
        and plan.get("question_sha256") == source_row.get("question_sha256")
        and plan.get("dated_question_sha256")
        == source_row.get("dated_question_sha256")
        and quote_sha256(dated_question) == plan.get("dated_question_sha256")
        and plan.get("parent_prediction") == source_row.get("parent_prediction")
        and plan.get("parent_prediction_sha256")
        == source_row.get("parent_prediction_sha256")
        == quote_sha256(parent)
        and identity_sha256(provider_input) == plan.get("provider_input_sha256")
        and identity_sha256(list(messages)) == plan.get("messages_sha256")
        and prompt_tokens == plan.get("prompt_token_proxy")
        and plan.get("hard_prompt_token_cap") == HARD_PROMPT_TOKEN_CAP
        and plan.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
        and prompt_tokens <= MAX_CHAT_PROMPT_TOKENS
        and prompt_tokens + OUTPUT_TOKEN_RESERVE <= HARD_PROMPT_TOKEN_CAP
        and plan.get("route_id") == TERMINAL_ROUTE_ID
        and len(allowed) == len(set(allowed))
        and all(type(value) is str and bool(value) for value in allowed)
        and set(groups) == set(allowed)
        and all(type(value) is str and bool(value) for value in groups.values())
        and bool(bindings)
        and all(
            type(key) is str
            and bool(key)
            and type(value) is str
            and bool(value)
            for key, value in bindings.items()
        )
        and type(plan.get("story_coherence")) is dict
        and type(plan.get("preservation_requirements")) is dict
        and type(plan.get("validation_contract")) is dict,
        f"full100 provider plan {source_row.get('ordinal')} changed mirrors",
    )
    assert_gold_blind(provider_input, path="full100_terminal_provider_input")
    return plan


def _validate_source_populations(
    construction: SealedArtifact,
    raw_provider_plans: Sequence[Mapping[str, Any]],
    raw_passthroughs: Sequence[Mapping[str, Any]],
) -> tuple[
    tuple[tuple[dict[str, Any], dict[str, Any]], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    payload = construction.payload
    raw_questions = _exact_list(payload.get("questions"), "full100 questions")
    provider_by_ordinal = {
        int(row.get("ordinal")): dict(row) for row in raw_provider_plans
    }
    passthrough_by_ordinal = {
        int(row.get("ordinal")): dict(row) for row in raw_passthroughs
    }
    _require(
        payload.get("format") == full100_cli.FORMAT
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("eligible_count") == ELIGIBLE_COUNT
        and payload.get("passthrough_count") == PASSTHROUGH_COUNT
        and payload.get("new_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("ordinal_cli_routing_available") is False
        and payload.get("production_ordinal_routing_enabled") is False
        and len(raw_questions) == QUESTION_COUNT
        and len(provider_by_ordinal) == len(raw_provider_plans) == ELIGIBLE_COUNT
        and len(passthrough_by_ordinal)
        == len(raw_passthroughs)
        == PASSTHROUGH_COUNT
        and set(provider_by_ordinal).isdisjoint(passthrough_by_ordinal)
        and set(provider_by_ordinal).union(passthrough_by_ordinal)
        == set(ALL_ORDINALS),
        "full100 construction answer population changed",
    )
    provider_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    passthrough_rows: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(raw_questions):
        row = _validate_receipt(
            raw,
            key="question_construction_receipt_sha256",
            label=f"full100 construction question {ordinal}",
        )
        _require(
            row.get("ordinal") == ordinal
            and row.get("new_provider_calls") == 0
            and row.get("retained_transformer_token_state_bytes") == 0,
            f"full100 construction row {ordinal} changed",
        )
        if ordinal in provider_by_ordinal:
            compact = _validate_receipt(
                row.get("terminal_answer_plan"),
                key="compact_plan_receipt_sha256",
                label=f"full100 compact plan {ordinal}",
            )
            provider = _exact_dict(
                compact.get("provider_plan"), f"full100 provider plan {ordinal}"
            )
            _require(
                row.get("mode") == TERMINAL_MODE
                and row.get("passthrough_prediction") is None
                and compact.get("provider_plan_sha256")
                == identity_sha256(provider)
                and compact.get("full_answer_plan_receipt_sha256")
                == provider.get("answer_plan_receipt_sha256")
                and compact.get("terminal_compilation_receipt_sha256")
                == provider.get("terminal_compilation_receipt_sha256")
                and provider == provider_by_ordinal[ordinal],
                f"full100 compact provider binding {ordinal} changed",
            )
            plan = _validate_provider_plan(provider, row)
            provider_rows.append((row, plan))
        else:
            expected = passthrough_by_ordinal[ordinal]
            parent = require_text(row.get("parent_prediction"), "V3 parent")
            _require(
                row == expected
                and row.get("mode") == PASSTHROUGH_MODE
                and row.get("terminal_answer_plan") is None
                and row.get("terminal_question_receipt_sha256") is None
                and row.get("terminal_sidecar_sha256") is None
                and row.get("passthrough_prediction") == parent
                and row.get("parent_prediction_sha256") == quote_sha256(parent),
                f"full100 V3 passthrough {ordinal} changed",
            )
            passthrough_rows.append(row)
        questions.append(row)
    return tuple(provider_rows), tuple(passthrough_rows), tuple(questions)


def _promotion_plan_projection(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plan.items() if key != "terminal_compilation"}


def _validate_promotion_transfer(
    provider_rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    promotion_plans: Sequence[Mapping[str, Any]],
) -> None:
    try:
        exact_plans = exact_answer_cli._validated_answer_plans(  # noqa: SLF001
            promotion_plans
        )
    except MatchedEvalContractError as exc:
        raise LockedSemanticGlobalTerminalFull100AnswerError(
            "promoted exact11 answer-plan population changed"
        ) from exc
    full_by_ordinal = {int(plan["ordinal"]): dict(plan) for _, plan in provider_rows}
    _require(
        set(EXACT_ORDINALS).issubset(full_by_ordinal)
        and all(
            full_by_ordinal[int(plan["ordinal"])]
            == _promotion_plan_projection(plan)
            for plan in exact_plans
        ),
        "promoted exact11 plans are not byte-identical inside full100",
    )


def _prompt_row(
    source_row: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    construction_sha256: str,
    replay_sha256: str,
) -> dict[str, Any]:
    provider_input = _exact_dict(plan.get("provider_input"), "provider input")
    messages = render_final_messages(provider_input)
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    body = {
        "allowed_handle_ids": list(plan["allowed_handle_ids"]),
        "answer_mode": TERMINAL_MODE,
        "dated_question_sha256": plan["dated_question_sha256"],
        "eligibility_receipt_sha256": source_row["eligibility_receipt_sha256"],
        "format": PROMPT_ROW_FORMAT,
        "full100_construction_artifact_sha256": construction_sha256,
        "full100_question_construction_receipt_sha256": source_row[
            "question_construction_receipt_sha256"
        ],
        "full100_replay_artifact_sha256": replay_sha256,
        "gate_row_receipt_sha256": source_row["gate_row_receipt_sha256"],
        "handle_group_by_id": dict(plan["handle_group_by_id"]),
        "messages": list(messages),
        "messages_sha256": identity_sha256(list(messages)),
        "ordinal": plan["ordinal"],
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "parent_answer_row_sha256": source_row["parent_answer_row_sha256"],
        "parent_prediction": plan["parent_prediction"],
        "parent_prediction_sha256": plan["parent_prediction_sha256"],
        "preservation_requirements": dict(plan["preservation_requirements"]),
        "prompt_token_proxy": prompt_tokens,
        "provider_input_sha256": plan["provider_input_sha256"],
        "question_id": plan["question_id"],
        "question_sha256": plan["question_sha256"],
        "route_id": plan["route_id"],
        "source_artifact_bindings": dict(plan["source_artifact_bindings"]),
        "story_coherence": dict(plan["story_coherence"]),
        "terminal_answer_plan_receipt_sha256": plan[
            "answer_plan_receipt_sha256"
        ],
        "terminal_compilation_receipt_sha256": plan[
            "terminal_compilation_receipt_sha256"
        ],
        "terminal_question_receipt_sha256": source_row[
            "terminal_question_receipt_sha256"
        ],
        "terminal_sidecar_sha256": source_row["terminal_sidecar_sha256"],
        "validation_contract": dict(plan["validation_contract"]),
    }
    _require(
        body["messages_sha256"] == plan.get("messages_sha256")
        and body["prompt_token_proxy"] == plan.get("prompt_token_proxy")
        and prompt_tokens <= MAX_CHAT_PROMPT_TOKENS
        and prompt_tokens + OUTPUT_TOKEN_RESERVE <= HARD_PROMPT_TOKEN_CAP,
        "full100 prompt differs from fitted terminal provider bytes",
    )
    row = {**body, "prompt_row_receipt_sha256": identity_sha256(body)}
    assert_gold_blind(row, path="full100_terminal_prompt_row")
    return row


def _passthrough_plan_row(source_row: Mapping[str, Any]) -> dict[str, Any]:
    parent = require_text(source_row.get("parent_prediction"), "V3 passthrough")
    body = {
        "answer_mode": PASSTHROUGH_MODE,
        "dated_question_sha256": source_row["dated_question_sha256"],
        "eligibility_receipt_sha256": source_row["eligibility_receipt_sha256"],
        "format": PASSTHROUGH_ROW_FORMAT,
        "full100_question_construction_receipt_sha256": source_row[
            "question_construction_receipt_sha256"
        ],
        "gate_row_receipt_sha256": source_row["gate_row_receipt_sha256"],
        "ordinal": source_row["ordinal"],
        "parent_answer_row_sha256": source_row["parent_answer_row_sha256"],
        "parent_prediction": parent,
        "parent_prediction_sha256": quote_sha256(parent),
        "prediction": parent,
        "prediction_sha256": quote_sha256(parent),
        "question_id": source_row["question_id"],
        "question_sha256": source_row["question_sha256"],
        "route_id": PASSTHROUGH_ROUTE_ID,
    }
    row = {**body, "passthrough_plan_receipt_sha256": identity_sha256(body)}
    assert_gold_blind(row, path="full100_terminal_passthrough_plan")
    return row


def build_preflight_payload(
    construction: SealedArtifact,
    replay: SealedArtifact,
    raw_provider_plans: Sequence[Mapping[str, Any]],
    raw_passthroughs: Sequence[Mapping[str, Any]],
    *,
    promotion_construction: SealedArtifact,
    promotion_replay: SealedArtifact,
    promotion_plans: Sequence[Mapping[str, Any]],
    promotion_audit: SealedArtifact,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> tuple[dict[str, Any], tuple[tuple[dict[str, str], ...], ...]]:
    require_text(model, "full100 answer model")
    require_text(gateway_url, "full100 answer gateway")
    _require(
        construction.sha256 == replay.sha256
        and construction.payload == replay.payload
        and promotion_construction.sha256 == promotion_replay.sha256
        and promotion_construction.payload == promotion_replay.payload
        and promotion_audit.payload.get("terminal_construction_sha256")
        == promotion_construction.sha256
        and promotion_audit.payload.get("terminal_replay_sha256")
        == promotion_replay.sha256
        and model == DEFAULT_MODEL
        and gateway_url == DEFAULT_GATEWAY_URL
        and type(max_concurrency) is int
        and max_concurrency > 0,
        "full100 answer source or runtime policy changed",
    )
    provider_rows, passthrough_sources, questions = _validate_source_populations(
        construction, raw_provider_plans, raw_passthroughs
    )
    _validate_promotion_transfer(provider_rows, promotion_plans)
    promotion = _promotion_binding(promotion_audit)
    prompt_rows = tuple(
        _prompt_row(
            source_row,
            plan,
            construction_sha256=construction.sha256,
            replay_sha256=replay.sha256,
        )
        for source_row, plan in provider_rows
    )
    passthrough_rows = tuple(
        _passthrough_plan_row(source) for source in passthrough_sources
    )
    prompts = tuple(
        tuple(dict(message) for message in row["messages"])
        for row in prompt_rows
    )
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS
    )
    eligible_ordinals = tuple(row["ordinal"] for row in prompt_rows)
    passthrough_ordinals = tuple(row["ordinal"] for row in passthrough_rows)
    _require(
        population.logical_prompt_count
        == population.unique_prompt_count
        == ELIGIBLE_COUNT
        and len(passthrough_rows) == PASSTHROUGH_COUNT
        and set(eligible_ordinals).isdisjoint(passthrough_ordinals)
        and set(eligible_ordinals).union(passthrough_ordinals)
        == set(ALL_ORDINALS),
        "full100 answer requires exactly 68 distinct prompts and 32 passthroughs",
    )
    payload = {
        "answer_plan_population_sha256": identity_sha256(
            [row["terminal_answer_plan_receipt_sha256"] for row in prompt_rows]
        ),
        "eligible_count": ELIGIBLE_COUNT,
        "eligible_ordinals": list(eligible_ordinals),
        "format": PREFLIGHT_FORMAT,
        "full100_construction_artifact_sha256": construction.sha256,
        "full100_replay_artifact_sha256": replay.sha256,
        "gateway_url": gateway_url,
        "gold_loaded": False,
        "hard_prompt_token_cap": HARD_PROMPT_TOKEN_CAP,
        "logical_answer_count": QUESTION_COUNT,
        "max_chat_prompt_tokens": MAX_CHAT_PROMPT_TOKENS,
        "max_concurrency": max_concurrency,
        "model": model,
        "observed_max_complete_envelope_tokens": max(
            row["prompt_token_proxy"] + OUTPUT_TOKEN_RESERVE
            for row in prompt_rows
        ),
        "ordinal_cli_routing_available": False,
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "passthrough_count": PASSTHROUGH_COUNT,
        "passthrough_ordinals": list(passthrough_ordinals),
        "passthrough_plan_rows": list(passthrough_rows),
        "passthrough_population_sha256": identity_sha256(
            [row["passthrough_plan_receipt_sha256"] for row in passthrough_rows]
        ),
        "physical_prompt_rows": list(prompt_rows),
        **promotion,
        "production_ordinal_routing_enabled": False,
        "prompt_population": population.model_dump(),
        "prompt_population_sha256": population.prompt_population_sha256,
        "promotion_exact_ordinals": list(EXACT_ORDINALS),
        "promotion_terminal_construction_artifact_sha256": (
            promotion_construction.sha256
        ),
        "promotion_terminal_replay_artifact_sha256": promotion_replay.sha256,
        "provider_calls": 0,
        "question_count": QUESTION_COUNT,
        "required_authorized_provider_calls": ELIGIBLE_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "source_question_population_sha256": identity_sha256(
            [row["question_construction_receipt_sha256"] for row in questions]
        ),
    }
    assert_gold_blind(payload, path="full100_terminal_answer_preflight")
    return payload, prompts


def _validate_preflight(
    artifact: SealedArtifact,
) -> tuple[
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    payload = artifact.payload
    for key in SOURCE_BINDING_KEYS:
        if key.endswith("_sha256"):
            require_sha256(payload.get(key), f"full100 preflight {key}")
    raw_prompt_rows = _exact_list(
        payload.get("physical_prompt_rows"), "full100 prompt rows"
    )
    raw_passthrough_rows = _exact_list(
        payload.get("passthrough_plan_rows"), "full100 passthrough rows"
    )
    _require(
        set(payload) == PREFLIGHT_KEYS
        and payload.get("format") == PREFLIGHT_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("hard_prompt_token_cap") == HARD_PROMPT_TOKEN_CAP
        and payload.get("max_chat_prompt_tokens") == MAX_CHAT_PROMPT_TOKENS
        and payload.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
        and payload.get("model") == DEFAULT_MODEL
        and payload.get("gateway_url") == DEFAULT_GATEWAY_URL
        and type(payload.get("max_concurrency")) is int
        and int(payload["max_concurrency"]) > 0
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("logical_answer_count") == QUESTION_COUNT
        and payload.get("eligible_count") == ELIGIBLE_COUNT
        and payload.get("passthrough_count") == PASSTHROUGH_COUNT
        and payload.get("required_authorized_provider_calls") == ELIGIBLE_COUNT
        and payload.get("ordinal_cli_routing_available") is False
        and payload.get("production_ordinal_routing_enabled") is False
        and payload.get("promotion_exact_ordinals") == list(EXACT_ORDINALS)
        and payload.get("postseal_semantic_atom_count")
        == payload.get("postseal_semantic_atom_final_usable_count")
        == postseal_cli.SEMANTIC_ATOM_COUNT
        and payload.get("postseal_semantic_atom_manifest_artifact_sha256")
        == postseal_cli.DEFAULT_SEMANTIC_ATOM_MANIFEST_SHA256
        and payload.get("postseal_semantic_atom_manifest_identity_sha256")
        == postseal_cli.DEFAULT_SEMANTIC_ATOM_MANIFEST_IDENTITY_SHA256
        and payload.get("postseal_semantic_atom_population_sha256")
        == postseal_cli.DEFAULT_SEMANTIC_ATOM_POPULATION_SHA256
        and len(raw_prompt_rows) == ELIGIBLE_COUNT
        and len(raw_passthrough_rows) == PASSTHROUGH_COUNT,
        "full100 answer sealed preflight firewall/population changed",
    )
    construct_sha = require_sha256(
        payload.get("full100_construction_artifact_sha256"),
        "full100 construction",
    )
    replay_sha = require_sha256(
        payload.get("full100_replay_artifact_sha256"), "full100 replay"
    )
    prompts: list[tuple[dict[str, str], ...]] = []
    prompt_rows: list[dict[str, Any]] = []
    plan_receipts: list[str] = []
    question_ids: list[str] = []
    eligible_ordinals: list[int] = []
    for raw in raw_prompt_rows:
        row = _validate_receipt(
            raw, key="prompt_row_receipt_sha256", label="full100 prompt row"
        )
        messages = _plain_messages(row.get("messages"), "full100 prompt messages")
        ordinal = row.get("ordinal")
        _require(
            type(ordinal) is int
            and 0 <= ordinal < QUESTION_COUNT
            and row.get("format") == PROMPT_ROW_FORMAT
            and row.get("answer_mode") == TERMINAL_MODE
            and row.get("route_id") == TERMINAL_ROUTE_ID
            and row.get("full100_construction_artifact_sha256") == construct_sha
            and row.get("full100_replay_artifact_sha256") == replay_sha
            and identity_sha256(list(messages)) == row.get("messages_sha256")
            and count_chat_prompt_token_proxy(messages)
            == row.get("prompt_token_proxy")
            and int(row["prompt_token_proxy"]) <= MAX_CHAT_PROMPT_TOKENS
            and int(row["prompt_token_proxy"]) + OUTPUT_TOKEN_RESERVE
            <= HARD_PROMPT_TOKEN_CAP
            and row.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
            and quote_sha256(
                require_text(row.get("parent_prediction"), "full100 parent")
            )
            == row.get("parent_prediction_sha256")
            and set(_exact_dict(row.get("handle_group_by_id"), "handle groups"))
            == set(_exact_list(row.get("allowed_handle_ids"), "allowed handles")),
            f"full100 prompt row {ordinal} changed",
        )
        for key in (
            "eligibility_receipt_sha256",
            "full100_question_construction_receipt_sha256",
            "gate_row_receipt_sha256",
            "parent_answer_row_sha256",
            "terminal_answer_plan_receipt_sha256",
            "terminal_compilation_receipt_sha256",
            "terminal_question_receipt_sha256",
            "terminal_sidecar_sha256",
        ):
            require_sha256(row.get(key), f"full100 prompt {key}")
        assert_gold_blind(messages, path=f"full100_terminal_prompt_{ordinal}")
        prompts.append(messages)
        prompt_rows.append(row)
        eligible_ordinals.append(ordinal)
        question_ids.append(require_text(row.get("question_id"), "question ID"))
        plan_receipts.append(row["terminal_answer_plan_receipt_sha256"])
    passthrough_rows: list[dict[str, Any]] = []
    passthrough_ordinals: list[int] = []
    passthrough_receipts: list[str] = []
    for raw in raw_passthrough_rows:
        row = _validate_receipt(
            raw,
            key="passthrough_plan_receipt_sha256",
            label="full100 passthrough plan",
        )
        ordinal = row.get("ordinal")
        parent = require_text(row.get("parent_prediction"), "passthrough parent")
        _require(
            type(ordinal) is int
            and 0 <= ordinal < QUESTION_COUNT
            and row.get("format") == PASSTHROUGH_ROW_FORMAT
            and row.get("answer_mode") == PASSTHROUGH_MODE
            and row.get("route_id") == PASSTHROUGH_ROUTE_ID
            and row.get("prediction") == parent
            and row.get("prediction_sha256")
            == row.get("parent_prediction_sha256")
            == quote_sha256(parent),
            f"full100 passthrough plan {ordinal} changed",
        )
        for key in (
            "dated_question_sha256",
            "eligibility_receipt_sha256",
            "full100_question_construction_receipt_sha256",
            "gate_row_receipt_sha256",
            "parent_answer_row_sha256",
            "question_sha256",
        ):
            require_sha256(row.get(key), f"passthrough {key}")
        passthrough_rows.append(row)
        passthrough_ordinals.append(ordinal)
        question_ids.append(require_text(row.get("question_id"), "question ID"))
        passthrough_receipts.append(row["passthrough_plan_receipt_sha256"])
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS
    )
    source_receipts_by_ordinal = {
        row["ordinal"]: row["full100_question_construction_receipt_sha256"]
        for row in (*prompt_rows, *passthrough_rows)
    }
    _require(
        tuple(eligible_ordinals) == tuple(payload.get("eligible_ordinals", ()))
        and tuple(passthrough_ordinals)
        == tuple(payload.get("passthrough_ordinals", ()))
        and len(set(eligible_ordinals)) == ELIGIBLE_COUNT
        and len(set(passthrough_ordinals)) == PASSTHROUGH_COUNT
        and set(eligible_ordinals).isdisjoint(passthrough_ordinals)
        and set(eligible_ordinals).union(passthrough_ordinals)
        == set(ALL_ORDINALS)
        and len(set(question_ids)) == QUESTION_COUNT
        and len(set(plan_receipts)) == ELIGIBLE_COUNT
        and len(set(passthrough_receipts)) == PASSTHROUGH_COUNT
        and len(source_receipts_by_ordinal) == QUESTION_COUNT
        and len(set(source_receipts_by_ordinal.values())) == QUESTION_COUNT
        and identity_sha256(plan_receipts)
        == payload.get("answer_plan_population_sha256")
        and identity_sha256(passthrough_receipts)
        == payload.get("passthrough_population_sha256")
        and identity_sha256(
            [source_receipts_by_ordinal[ordinal] for ordinal in ALL_ORDINALS]
        )
        == payload.get("source_question_population_sha256")
        and construct_sha == replay_sha
        and payload.get("promotion_terminal_construction_artifact_sha256")
        == payload.get("promotion_terminal_replay_artifact_sha256")
        and population.model_dump() == payload.get("prompt_population")
        and population.prompt_population_sha256
        == payload.get("prompt_population_sha256")
        and population.logical_prompt_count
        == population.unique_prompt_count
        == ELIGIBLE_COUNT
        and payload.get("observed_max_complete_envelope_tokens")
        == max(row["prompt_token_proxy"] + OUTPUT_TOKEN_RESERVE for row in prompt_rows)
        and int(payload["observed_max_complete_envelope_tokens"])
        <= HARD_PROMPT_TOKEN_CAP,
        "full100 sealed prompt/passthrough population changed",
    )
    assert_gold_blind(payload, path="full100_terminal_answer_preflight")
    return tuple(prompts), tuple(prompt_rows), tuple(passthrough_rows)


def _read_preflight(
    output_root: str | Path, expected_sha256: str
) -> tuple[
    SealedArtifact,
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    artifact = read_sealed_json(Path(output_root) / PREFLIGHT_NAME)
    _require(
        artifact.sha256
        == require_sha256(expected_sha256, "full100 answer preflight"),
        "full100 answer preflight changed",
    )
    prompts, prompt_rows, passthrough_rows = _validate_preflight(artifact)
    return artifact, prompts, prompt_rows, passthrough_rows


def _assert_preflight_promotion_binding(
    preflight: SealedArtifact, promotion_audit: SealedArtifact
) -> None:
    binding = _promotion_binding(promotion_audit)
    _require(
        all(preflight.payload.get(key) == value for key, value in binding.items())
        and preflight.payload.get("promotion_terminal_construction_artifact_sha256")
        == promotion_audit.payload.get("terminal_construction_sha256")
        and preflight.payload.get("promotion_terminal_replay_artifact_sha256")
        == promotion_audit.payload.get("terminal_replay_sha256"),
        "full100 preflight differs from promoted semantic-atom audit",
    )


def _assert_preflight_source_binding(
    preflight: SealedArtifact, sources: _VerifiedSources
) -> None:
    rebuilt, _ = build_preflight_payload(
        sources.full100_construction,
        sources.full100_replay,
        sources.provider_plans,
        sources.passthroughs,
        promotion_construction=sources.promotion_construction,
        promotion_replay=sources.promotion_replay,
        promotion_plans=sources.promotion_plans,
        promotion_audit=sources.promotion_audit,
        model=str(preflight.payload["model"]),
        gateway_url=str(preflight.payload["gateway_url"]),
        max_concurrency=int(preflight.payload["max_concurrency"]),
    )
    _require(
        rebuilt == preflight.payload,
        "full100 preflight differs from authenticated construction sources",
    )


def _release_payload(
    *,
    preflight: SealedArtifact,
    full100_terminal_root: str | Path,
    promotion_terminal_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    full_root = _canonical_root(full100_terminal_root)
    promotion_root = _canonical_root(promotion_terminal_root)
    answer_root = _canonical_root(output_root)
    body = {
        "answer_output_root": answer_root,
        "answer_output_root_sha256": identity_sha256({"canonical_root": answer_root}),
        "approval_opt_in": True,
        "eligible_count": ELIGIBLE_COUNT,
        "format": RELEASE_FORMAT,
        "full100_terminal_root": full_root,
        "full100_terminal_root_sha256": identity_sha256(
            {"canonical_root": full_root}
        ),
        "gateway_url": preflight.payload["gateway_url"],
        "gold_loaded": False,
        "max_concurrency": preflight.payload["max_concurrency"],
        "model": preflight.payload["model"],
        "passthrough_count": PASSTHROUGH_COUNT,
        "preflight_artifact_sha256": preflight.sha256,
        "promotion_terminal_root": promotion_root,
        "promotion_terminal_root_sha256": identity_sha256(
            {"canonical_root": promotion_root}
        ),
        "provider_calls_during_release": 0,
        "question_count": QUESTION_COUNT,
        "release_status": "approved_for_provider_execution",
        "required_authorized_provider_calls": ELIGIBLE_COUNT,
        "retained_transformer_token_state_bytes": 0,
        **{key: preflight.payload[key] for key in SOURCE_BINDING_KEYS},
    }
    assert_gold_blind(body, path="full100_terminal_answer_release")
    return {**body, "release_identity_sha256": identity_sha256(body)}


def _validate_release(
    artifact: SealedArtifact,
    *,
    preflight: SealedArtifact,
    output_root: str | Path,
) -> dict[str, Any]:
    payload = artifact.payload
    body = {key: value for key, value in payload.items() if key != "release_identity_sha256"}
    full_root = require_text(payload.get("full100_terminal_root"), "full100 root")
    promotion_root = require_text(
        payload.get("promotion_terminal_root"), "promotion terminal root"
    )
    answer_root = _canonical_root(output_root)
    _require(
        set(payload) == RELEASE_KEYS
        and require_sha256(payload.get("release_identity_sha256"), "provider release")
        == identity_sha256(body)
        and payload.get("format") == RELEASE_FORMAT
        and payload.get("release_status") == "approved_for_provider_execution"
        and payload.get("approval_opt_in") is True
        and payload.get("gold_loaded") is False
        and payload.get("provider_calls_during_release") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("eligible_count") == ELIGIBLE_COUNT
        and payload.get("passthrough_count") == PASSTHROUGH_COUNT
        and payload.get("required_authorized_provider_calls") == ELIGIBLE_COUNT
        and payload.get("preflight_artifact_sha256") == preflight.sha256
        and payload.get("model") == preflight.payload.get("model")
        and payload.get("gateway_url") == preflight.payload.get("gateway_url")
        and payload.get("max_concurrency") == preflight.payload.get("max_concurrency")
        and all(
            payload.get(key) == preflight.payload.get(key)
            for key in SOURCE_BINDING_KEYS
        )
        and payload.get("answer_output_root") == answer_root
        and payload.get("answer_output_root_sha256")
        == identity_sha256({"canonical_root": answer_root})
        and payload.get("full100_terminal_root_sha256")
        == identity_sha256({"canonical_root": full_root})
        and payload.get("promotion_terminal_root_sha256")
        == identity_sha256({"canonical_root": promotion_root}),
        "full100 answer provider release changed",
    )
    assert_gold_blind(payload, path="full100_terminal_answer_release")
    return payload


def _read_release(
    output_root: str | Path,
    expected_sha256: str,
    *,
    preflight: SealedArtifact,
) -> SealedArtifact:
    try:
        artifact = read_sealed_json(Path(output_root) / RELEASE_NAME)
    except MatchedEvalContractError as exc:
        raise LockedSemanticGlobalTerminalFull100AnswerError(
            "full100 provider release is absent or invalid"
        ) from exc
    _require(
        artifact.sha256 == require_sha256(expected_sha256, "provider release"),
        "full100 provider release artifact changed",
    )
    _validate_release(artifact, preflight=preflight, output_root=output_root)
    return artifact


def _read_bound_promotion_audit(
    args: argparse.Namespace, preflight: SealedArtifact
) -> SealedArtifact:
    audit = _read_promotion_audit(
        args.postseal_audit,
        str(args.expected_postseal_audit_sha256),
        construction_sha256=str(
            preflight.payload["promotion_terminal_construction_artifact_sha256"]
        ),
        replay_sha256=str(
            preflight.payload["promotion_terminal_replay_artifact_sha256"]
        ),
    )
    _assert_preflight_promotion_binding(preflight, audit)
    return audit


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    _require(
        not (output_root / CHECKPOINT_DIR_NAME).exists(),
        "full100 preflight requires a fresh absent checkpoint root",
    )
    sources = _load_verified_sources(args)
    payload, _ = build_preflight_payload(
        sources.full100_construction,
        sources.full100_replay,
        sources.provider_plans,
        sources.passthroughs,
        promotion_construction=sources.promotion_construction,
        promotion_replay=sources.promotion_replay,
        promotion_plans=sources.promotion_plans,
        promotion_audit=sources.promotion_audit,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    artifact, created = publish_sealed_json(output_root / PREFLIGHT_NAME, payload)
    return {
        "created": created,
        "eligible_count": ELIGIBLE_COUNT,
        "maximum_complete_prompt_envelope": payload[
            "observed_max_complete_envelope_tokens"
        ],
        "passthrough_count": PASSTHROUGH_COUNT,
        "physical_provider_calls": 0,
        "postseal_promotion_audit_sha256": sources.promotion_audit.sha256,
        "preflight_sha256": artifact.sha256,
        "question_count": QUESTION_COUNT,
        "required_authorized_provider_calls": ELIGIBLE_COUNT,
        "retained_transformer_token_state_bytes": 0,
    }


def run_approve_release(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    _require(
        args.approve_provider_release is True,
        "full100 release requires explicit provider-release approval",
    )
    _require(
        not (output_root / CHECKPOINT_DIR_NAME).exists(),
        "full100 release requires an absent checkpoint root",
    )
    sources = _load_verified_sources(args)
    preflight, _, _, _ = _read_preflight(
        output_root, str(args.expected_preflight_sha256)
    )
    _assert_preflight_source_binding(preflight, sources)
    _assert_preflight_promotion_binding(preflight, sources.promotion_audit)
    payload = _release_payload(
        preflight=preflight,
        full100_terminal_root=args.full100_terminal_root,
        promotion_terminal_root=args.promotion_terminal_root,
        output_root=output_root,
    )
    artifact, created = publish_sealed_json(output_root / RELEASE_NAME, payload)
    return {
        "created": created,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "postseal_promotion_audit_sha256": sources.promotion_audit.sha256,
        "preflight_sha256": preflight.sha256,
        "release_sha256": artifact.sha256,
        "required_authorized_provider_calls": ELIGIBLE_COUNT,
        "retained_transformer_token_state_bytes": 0,
    }


def _runtime(
    preflight: SealedArtifact,
    release: SealedArtifact,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    args: argparse.Namespace,
    client: Any | None,
) -> FastCompletionRuntime:
    _require(
        str(args.model) == preflight.payload.get("model") == DEFAULT_MODEL
        and str(args.gateway_url)
        == preflight.payload.get("gateway_url")
        == DEFAULT_GATEWAY_URL
        and int(args.max_concurrency) == preflight.payload.get("max_concurrency")
        and release.payload.get("preflight_artifact_sha256") == preflight.sha256
        and release.payload.get("release_status")
        == "approved_for_provider_execution"
        and len(prompts) == ELIGIBLE_COUNT,
        "full100 answer runtime differs from sealed preflight",
    )
    return FastCompletionRuntime(
        checkpoint_dir=Path(args.output_root) / CHECKPOINT_DIR_NAME,
        prompt_population=prompts,
        model=DEFAULT_MODEL,
        client=client,
        max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS,
        max_new_tokens=OUTPUT_TOKEN_RESERVE,
        max_concurrency=int(args.max_concurrency),
        retries=0,
        benchmark_provenance={
            "arm": FORMAT,
            "authorized_unique_calls": ELIGIBLE_COUNT,
            "experiment_format": RUN_FORMAT,
            "gateway_url": DEFAULT_GATEWAY_URL,
            "gold_loaded": False,
            "preflight_artifact_sha256": preflight.sha256,
            "release_authorization_artifact_sha256": release.sha256,
            **{key: preflight.payload[key] for key in SOURCE_BINDING_KEYS},
        },
    )


def _checkpoint_batch(
    preflight: SealedArtifact,
    release: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    args: argparse.Namespace,
    client: Any | None,
) -> FastCompletionBatch:
    runtime = _runtime(preflight, release, prompts, args=args, client=client)
    try:
        return runtime.run()
    finally:
        runtime.close()


def _validated_checkpoint_hits(
    preflight: SealedArtifact,
    release: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    args: argparse.Namespace,
) -> int:
    checkpoint_root = Path(args.output_root) / CHECKPOINT_DIR_NAME
    if not checkpoint_root.exists():
        return 0
    runtime = _runtime(preflight, release, prompts, args=args, client=None)
    try:
        with runtime._journal_guard():  # noqa: SLF001 - journal authority
            records = runtime._load_all_records()  # noqa: SLF001
    finally:
        runtime.close()
    _require(
        len(records) <= ELIGIBLE_COUNT,
        "full100 checkpoint population escaped the 68 derived prompts",
    )
    return len(records)


def _read_only_checkpoint_count(output_root: str | Path) -> int:
    """Count structurally complete journal pairs without creating lock state."""

    root = Path(output_root) / CHECKPOINT_DIR_NAME
    if not root.exists():
        return 0
    _require(
        not root.is_symlink() and root.is_dir(),
        "full100 checkpoint root must be a regular directory",
    )
    requests: set[str] = set()
    responses: set[str] = set()
    for path in root.glob("*.json"):
        match = _JOURNAL_FILENAME_RE.fullmatch(path.name)
        _require(match is not None, "full100 checkpoint root contains foreign JSON")
        assert match is not None
        target = requests if match.group("kind") == "request" else responses
        target.add(match.group("key"))
    _require(
        requests == responses,
        "full100 checkpoint journal pair is incomplete; unsafe retry forbidden",
    )
    _require(
        len(requests) <= ELIGIBLE_COUNT,
        "full100 checkpoint population exceeds the 68 derived prompts",
    )
    return len(requests)


def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    preflight, prompts, _, _ = _read_preflight(
        args.output_root, str(args.expected_preflight_sha256)
    )
    audit = _read_bound_promotion_audit(args, preflight)
    release = _read_release(
        args.output_root,
        str(args.expected_release_sha256),
        preflight=preflight,
    )
    _require(
        args.enable_provider is True
        and type(args.authorized_provider_calls) is int
        and 0 <= args.authorized_provider_calls <= ELIGIBLE_COUNT,
        "full100 provider requires a bounded Terra call authorization",
    )
    candidate_hits = _read_only_checkpoint_count(args.output_root)
    remaining = ELIGIBLE_COUNT - candidate_hits
    _require(
        args.authorized_provider_calls == remaining,
        "full100 authorization must exactly equal remaining Terra calls",
    )
    checkpoint_hits = _validated_checkpoint_hits(
        preflight, release, prompts, args=args
    )
    _require(
        checkpoint_hits == candidate_hits,
        "full100 authenticated checkpoint count changed after authorization",
    )
    if remaining == 0:
        batch = _checkpoint_batch(
            preflight, release, prompts, args=args, client=None
        )
        _require(
            batch.usage.logical_calls
            == batch.usage.unique_calls
            == batch.usage.checkpoint_hits
            == ELIGIBLE_COUNT
            and batch.usage.physical_calls == 0,
            "full100 completed checkpoint replay changed",
        )
        return {
            "authorized_remaining_provider_calls": 0,
            "checkpoint_hits": ELIGIBLE_COUNT,
            "gold_loaded": False,
            "physical_provider_calls": 0,
            "postseal_promotion_audit_sha256": audit.sha256,
            "preflight_sha256": preflight.sha256,
            "release_sha256": release.sha256,
            "required_authorized_provider_calls": 0,
            "retained_transformer_token_state_bytes": 0,
        }
    load_dotenv()
    api_key = os.environ.get(str(args.api_key_env), "").strip()
    _require(bool(api_key), f"provider API key is empty: {args.api_key_env}")
    client = live._make_provider_client(api_key, str(args.gateway_url))  # noqa: SLF001
    try:
        batch = _checkpoint_batch(
            preflight, release, prompts, args=args, client=client
        )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    _require(
        batch.usage.logical_calls == batch.usage.unique_calls == ELIGIBLE_COUNT
        and batch.usage.physical_calls + batch.usage.checkpoint_hits
        == ELIGIBLE_COUNT
        and batch.usage.physical_calls <= args.authorized_provider_calls
        and batch.usage.checkpoint_hits >= checkpoint_hits,
        "full100 provider population changed",
    )
    return {
        "authorized_remaining_provider_calls": remaining,
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "gold_loaded": False,
        "physical_provider_calls": batch.usage.physical_calls,
        "postseal_promotion_audit_sha256": audit.sha256,
        "preflight_sha256": preflight.sha256,
        "release_sha256": release.sha256,
        "required_authorized_provider_calls": remaining,
        "retained_transformer_token_state_bytes": 0,
    }


def _terminal_result(
    prompt_row: Mapping[str, Any],
    completion: str,
    record: Any,
) -> dict[str, Any]:
    base = materialize_typed_final_result_row(
        prompt_row,
        completion,
        completion_receipt_sha256=record.completion_sha256,
        call_key_sha256=record.call_key_sha256,
        request_journal_sha256=record.request_journal_sha256,
        response_journal_sha256=record.response_journal_sha256,
    )
    body = {key: value for key, value in base.items() if key != "source_row_sha256"}
    body.update(
        {
            "answer_mode": TERMINAL_MODE,
            "full100_question_construction_receipt_sha256": prompt_row[
                "full100_question_construction_receipt_sha256"
            ],
            "parent_answer_row_sha256": prompt_row["parent_answer_row_sha256"],
            "terminal_answer_plan_receipt_sha256": prompt_row[
                "terminal_answer_plan_receipt_sha256"
            ],
        }
    )
    return {**body, "source_row_sha256": identity_sha256(body)}


def _passthrough_result(plan: Mapping[str, Any]) -> dict[str, Any]:
    parent = require_text(plan.get("parent_prediction"), "passthrough parent")
    body = {
        "answer_mode": PASSTHROUGH_MODE,
        "call_key_sha256": None,
        "changed_from_parent": False,
        "completion_receipt_sha256": None,
        "dated_question_sha256": plan["dated_question_sha256"],
        "decision": PASSTHROUGH_MODE,
        "format": RESULT_ROW_FORMAT,
        "full100_question_construction_receipt_sha256": plan[
            "full100_question_construction_receipt_sha256"
        ],
        "ordinal": plan["ordinal"],
        "parent_answer_row_sha256": plan["parent_answer_row_sha256"],
        "parent_prediction_sha256": quote_sha256(parent),
        "parse_error_code": None,
        "parse_receipt_sha256": None,
        "prediction": parent,
        "prediction_sha256": quote_sha256(parent),
        "prediction_source": "sealed_v3_byte_exact_passthrough_v1",
        "prompt_row_receipt_sha256": None,
        "question_id": plan["question_id"],
        "question_sha256": plan["question_sha256"],
        "request_journal_sha256": None,
        "response_journal_sha256": None,
        "retained_transformer_token_state_bytes": 0,
        "route_id": PASSTHROUGH_ROUTE_ID,
        "solver_valid": None,
        "terminal_answer_plan_receipt_sha256": None,
        "used_handle_ids": [],
        "validation_basis": "sealed_v3_passthrough_no_provider_call",
        "validator_policy_format": VALIDATOR_POLICY_FORMAT,
    }
    return {**body, "source_row_sha256": identity_sha256(body)}


def _materialization_payload(
    preflight: SealedArtifact,
    release: SealedArtifact,
    prompt_rows: tuple[dict[str, Any], ...],
    passthrough_rows: tuple[dict[str, Any], ...],
    batch: FastCompletionBatch,
) -> dict[str, Any]:
    _require(
        batch.usage.logical_calls
        == batch.usage.unique_calls
        == batch.usage.checkpoint_hits
        == ELIGIBLE_COUNT
        and batch.usage.physical_calls == 0
        and len(batch.logical_completions) == ELIGIBLE_COUNT
        and len(batch.unique_records) == ELIGIBLE_COUNT,
        "full100 materialization requires all 68 checkpoints and no provider calls",
    )
    records = {row.messages_sha256: row for row in batch.unique_records}
    _require(len(records) == ELIGIBLE_COUNT, "full100 completion identities repeat")
    results_by_ordinal: dict[int, dict[str, Any]] = {}
    for plan, completion in zip(
        prompt_rows, batch.logical_completions, strict=True
    ):
        record = records.get(plan["messages_sha256"])
        _require(
            record is not None
            and record.completion == completion
            and record.checkpoint_hit is True
            and record.physical_call is False,
            f"full100 checkpoint record {plan['ordinal']} changed",
        )
        assert record is not None
        results_by_ordinal[plan["ordinal"]] = _terminal_result(
            plan, completion, record
        )
    for plan in passthrough_rows:
        ordinal = int(plan["ordinal"])
        _require(ordinal not in results_by_ordinal, "full100 answer ordinal repeated")
        results_by_ordinal[ordinal] = _passthrough_result(plan)
    _require(
        set(results_by_ordinal) == set(ALL_ORDINALS),
        "full100 merged answer population is incomplete",
    )
    results = [results_by_ordinal[ordinal] for ordinal in ALL_ORDINALS]
    judge_rows = [judge_row_projection(row) for row in results]
    payload = {
        "changed_prediction_count": sum(
            bool(row["changed_from_parent"]) for row in results
        ),
        "completion_batch": batch.model_dump(),
        "eligible_count": ELIGIBLE_COUNT,
        "eligible_ordinals": list(preflight.payload["eligible_ordinals"]),
        "format": RUN_FORMAT,
        "gold_loaded": False,
        "invalid_completion_parent_fallback_count": sum(
            row.get("prediction_source") == "typed_final_invalid_keep_parent_v1"
            for row in results
        ),
        "judge_rows": judge_rows,
        "passthrough_count": PASSTHROUGH_COUNT,
        "passthrough_ordinals": list(preflight.payload["passthrough_ordinals"]),
        "physical_provider_calls_during_materialization": 0,
        "preflight_artifact_sha256": preflight.sha256,
        "question_count": QUESTION_COUNT,
        "questions": results,
        "release_authorization_artifact_sha256": release.sha256,
        "required_authorized_provider_calls": ELIGIBLE_COUNT,
        "retained_transformer_token_state_bytes": 0,
        **{key: preflight.payload[key] for key in SOURCE_BINDING_KEYS},
        "validator_policy_format": VALIDATOR_POLICY_FORMAT,
    }
    assert_gold_blind(payload, path="full100_terminal_answer_run")
    return payload


def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    preflight, prompts, prompt_rows, passthrough_rows = _read_preflight(
        args.output_root, str(args.expected_preflight_sha256)
    )
    audit = _read_bound_promotion_audit(args, preflight)
    release = _read_release(
        args.output_root,
        str(args.expected_release_sha256),
        preflight=preflight,
    )
    batch = _checkpoint_batch(
        preflight, release, prompts, args=args, client=None
    )
    payload = _materialization_payload(
        preflight, release, prompt_rows, passthrough_rows, batch
    )
    artifact, created = publish_sealed_json(Path(args.output_root) / RUN_NAME, payload)
    return {
        "changed_prediction_count": payload["changed_prediction_count"],
        "checkpoint_hits": ELIGIBLE_COUNT,
        "created": created,
        "eligible_count": ELIGIBLE_COUNT,
        "gold_loaded": False,
        "passthrough_count": PASSTHROUGH_COUNT,
        "physical_provider_calls": 0,
        "postseal_promotion_audit_sha256": audit.sha256,
        "release_sha256": release.sha256,
        "run_sha256": artifact.sha256,
    }


def _validate_completion_batch(
    raw: object,
    *,
    preflight: SealedArtifact,
    prompt_rows: Sequence[Mapping[str, Any]],
    expected_batch: FastCompletionBatch,
) -> dict[str, Any]:
    """Bind the nested batch to replayed journals and the sealed runtime.

    ``expected_batch`` must come from the checkpoint-only runtime constructed
    from this preflight and release.  That runtime authenticates the canonical
    request/response journals before this projection is compared.
    """

    observed = _exact_dict(raw, "full100 completion batch")
    expected = expected_batch.model_dump()
    usage = expected_batch.usage
    provenance = expected_batch.provenance
    population = expected_batch.prompt_population
    records = expected_batch.unique_records
    _require(
        observed == expected
        and set(observed)
        == {
            "logical_completions",
            "prompt_population",
            "provenance",
            "runtime_identity_sha256",
            "unique_records",
            "usage",
        }
        and usage.logical_calls
        == usage.unique_calls
        == usage.checkpoint_hits
        == ELIGIBLE_COUNT
        and usage.physical_calls == 0
        and usage.deduplicated_logical_calls == 0
        and len(expected_batch.logical_completions)
        == len(records)
        == len(prompt_rows)
        == ELIGIBLE_COUNT
        and provenance.model == preflight.payload.get("model") == DEFAULT_MODEL
        and provenance.max_new_tokens == OUTPUT_TOKEN_RESERVE
        and provenance.max_prompt_token_proxy == MAX_CHAT_PROMPT_TOKENS
        and provenance.max_concurrency == preflight.payload.get("max_concurrency")
        and provenance.retries == 0
        and dict(provenance.request_options) == {}
        and provenance.prompt_population_sha256
        == preflight.payload.get("prompt_population_sha256")
        and provenance.persisted_transformer_token_state is False
        and provenance.retained_transformer_token_state_bytes == 0
        and provenance.external_provider_persistence_certified is False
        and dict(provenance.benchmark_provenance).get("authorized_unique_calls")
        == ELIGIBLE_COUNT
        and dict(provenance.benchmark_provenance).get("gold_loaded") is False
        and all(
            dict(provenance.benchmark_provenance).get(key)
            == preflight.payload.get(key)
            for key in SOURCE_BINDING_KEYS
        )
        and expected_batch.runtime_identity_sha256
        == identity_sha256(provenance.model_dump())
        and population.model_dump() == preflight.payload.get("prompt_population")
        and population.prompt_population_sha256
        == preflight.payload.get("prompt_population_sha256"),
        "full100 completion batch runtime/provenance changed",
    )
    seen_calls: set[str] = set()
    for plan, completion, record in zip(
        prompt_rows,
        expected_batch.logical_completions,
        records,
        strict=True,
    ):
        _require(
            type(completion) is str
            and bool(completion)
            and record.messages_sha256 == plan.get("messages_sha256")
            and record.completion == completion
            and record.completion_sha256 == quote_sha256(completion)
            and record.requested_model == DEFAULT_MODEL
            and record.finish_reason == "stop"
            and record.prompt_token_proxy == plan.get("prompt_token_proxy")
            and record.checkpoint_hit is True
            and record.physical_call is False,
            f"full100 completion record {plan.get('ordinal')} changed",
        )
        for value, label in (
            (record.call_key_sha256, "call key"),
            (record.request_journal_sha256, "request journal"),
            (record.response_journal_sha256, "response journal"),
            (record.messages_sha256, "messages"),
            (record.completion_sha256, "completion"),
        ):
            require_sha256(value, f"full100 completion {label}")
        seen_calls.add(record.call_key_sha256)
    _require(
        len(seen_calls) == ELIGIBLE_COUNT,
        "full100 completion call identities repeat",
    )
    return observed


def _validate_run(
    artifact: SealedArtifact,
    *,
    preflight: SealedArtifact,
    expected_release_sha256: str,
    expected_batch: FastCompletionBatch,
) -> tuple[dict[str, Any], ...]:
    payload = artifact.payload
    questions = _exact_list(payload.get("questions"), "full100 answer questions")
    judge_rows = _exact_list(payload.get("judge_rows"), "full100 judge rows")
    prompt_by_ordinal = {
        row["ordinal"]: row
        for row in _exact_list(
            preflight.payload.get("physical_prompt_rows"), "preflight prompt rows"
        )
    }
    pass_by_ordinal = {
        row["ordinal"]: row
        for row in _exact_list(
            preflight.payload.get("passthrough_plan_rows"), "preflight passthrough rows"
        )
    }
    _validate_completion_batch(
        payload.get("completion_batch"),
        preflight=preflight,
        prompt_rows=tuple(prompt_by_ordinal.values()),
        expected_batch=expected_batch,
    )
    records_by_messages = {
        record.messages_sha256: record for record in expected_batch.unique_records
    }
    _require(
        set(payload) == RUN_KEYS
        and payload.get("format") == RUN_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("physical_provider_calls_during_materialization") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("eligible_count") == ELIGIBLE_COUNT
        and payload.get("passthrough_count") == PASSTHROUGH_COUNT
        and payload.get("eligible_ordinals")
        == preflight.payload.get("eligible_ordinals")
        and payload.get("passthrough_ordinals")
        == preflight.payload.get("passthrough_ordinals")
        and payload.get("required_authorized_provider_calls") == ELIGIBLE_COUNT
        and payload.get("validator_policy_format") == VALIDATOR_POLICY_FORMAT
        and payload.get("preflight_artifact_sha256") == preflight.sha256
        and payload.get("release_authorization_artifact_sha256")
        == require_sha256(expected_release_sha256, "full100 provider release")
        and all(
            payload.get(key) == preflight.payload.get(key)
            for key in SOURCE_BINDING_KEYS
        )
        and len(questions) == len(judge_rows) == QUESTION_COUNT,
        "full100 answer run envelope changed",
    )
    validated: list[dict[str, Any]] = []
    question_ids: list[str] = []
    for ordinal, raw, projected in zip(ALL_ORDINALS, questions, judge_rows, strict=True):
        row = _exact_dict(raw, f"full100 result row {ordinal}")
        unsigned = {key: value for key, value in row.items() if key != "source_row_sha256"}
        prediction = require_text(row.get("prediction"), "full100 prediction")
        _require(
            set(row) == RESULT_KEYS
            and row.get("format") == RESULT_ROW_FORMAT
            and row.get("ordinal") == ordinal
            and require_sha256(row.get("source_row_sha256"), "result row")
            == identity_sha256(unsigned)
            and row.get("prediction_sha256") == quote_sha256(prediction)
            and row.get("retained_transformer_token_state_bytes") == 0
            and judge_row_projection(row) == projected,
            f"full100 result row {ordinal} changed",
        )
        if ordinal in prompt_by_ordinal:
            source = prompt_by_ordinal[ordinal]
            record = records_by_messages.get(source.get("messages_sha256"))
            parent = require_text(
                source.get("parent_prediction"), "terminal source parent"
            )
            used = _exact_list(row.get("used_handle_ids"), "terminal used handles")
            allowed = _exact_list(
                source.get("allowed_handle_ids"), "terminal allowed handles"
            )
            decision = row.get("decision")
            prediction_source = row.get("prediction_source")
            _require(
                record is not None
                and row.get("answer_mode") == TERMINAL_MODE
                and row.get("question_id") == source.get("question_id")
                and row.get("question_sha256") == source.get("question_sha256")
                and row.get("dated_question_sha256")
                == source.get("dated_question_sha256")
                and row.get("parent_prediction_sha256")
                == source.get("parent_prediction_sha256")
                and row.get("route_id") == source.get("route_id")
                and row.get("prompt_row_receipt_sha256")
                == source.get("prompt_row_receipt_sha256")
                and row.get("full100_question_construction_receipt_sha256")
                == source.get("full100_question_construction_receipt_sha256")
                and row.get("parent_answer_row_sha256")
                == source.get("parent_answer_row_sha256")
                and row.get("terminal_answer_plan_receipt_sha256")
                == source.get("terminal_answer_plan_receipt_sha256")
                and row.get("call_key_sha256") == record.call_key_sha256
                and row.get("completion_receipt_sha256")
                == record.completion_sha256
                and row.get("request_journal_sha256")
                == record.request_journal_sha256
                and row.get("response_journal_sha256")
                == record.response_journal_sha256
                and row.get("changed_from_parent") == (prediction != parent)
                and row.get("validator_policy_format") == VALIDATOR_POLICY_FORMAT
                and type(row.get("parse_error_code")) is str
                and bool(row.get("parse_error_code"))
                and type(row.get("validation_basis")) is str
                and bool(row.get("validation_basis"))
                and len(used) == len(set(used))
                and all(type(value) is str and bool(value) for value in used)
                and set(used) <= set(allowed)
                and (
                    (
                        decision == "replace"
                        and prediction_source in TERMINAL_REPLACEMENT_SOURCES
                        and row.get("solver_valid") is True
                        and row.get("changed_from_parent") is True
                        and bool(used)
                    )
                    or (
                        decision == "keep_parent"
                        and prediction_source
                        == "typed_final_validated_keep_parent_v1"
                        and row.get("solver_valid") is True
                        and row.get("changed_from_parent") is False
                        and prediction == parent
                        and used == []
                    )
                    or (
                        decision == "invalid_keep_parent"
                        and prediction_source == "typed_final_invalid_keep_parent_v1"
                        and row.get("solver_valid") is False
                        and row.get("changed_from_parent") is False
                        and prediction == parent
                        and used == []
                    )
                ),
                f"full100 terminal result provenance {ordinal} changed",
            )
            for key in (
                "call_key_sha256",
                "completion_receipt_sha256",
                "parse_receipt_sha256",
                "request_journal_sha256",
                "response_journal_sha256",
            ):
                require_sha256(row.get(key), f"terminal result {key}")
        else:
            source = pass_by_ordinal.get(ordinal)
            parent = require_text(row.get("prediction"), "passthrough prediction")
            _require(
                source is not None
                and row.get("answer_mode") == PASSTHROUGH_MODE
                and row.get("decision") == PASSTHROUGH_MODE
                and row.get("prediction_source")
                == "sealed_v3_byte_exact_passthrough_v1"
                and row.get("changed_from_parent") is False
                and row.get("prediction") == source.get("prediction")
                and row.get("prediction_sha256")
                == row.get("parent_prediction_sha256")
                == source.get("parent_prediction_sha256")
                == quote_sha256(parent)
                and row.get("question_id") == source.get("question_id")
                and row.get("question_sha256") == source.get("question_sha256")
                and row.get("dated_question_sha256")
                == source.get("dated_question_sha256")
                and row.get("route_id") == PASSTHROUGH_ROUTE_ID
                and row.get("full100_question_construction_receipt_sha256")
                == source.get("full100_question_construction_receipt_sha256")
                and row.get("parent_answer_row_sha256")
                == source.get("parent_answer_row_sha256")
                and row.get("terminal_answer_plan_receipt_sha256") is None
                and row.get("solver_valid") is None
                and row.get("used_handle_ids") == []
                and row.get("validation_basis")
                == "sealed_v3_passthrough_no_provider_call"
                and row.get("validator_policy_format") == VALIDATOR_POLICY_FORMAT
                and all(
                    row.get(key) is None
                    for key in (
                        "call_key_sha256",
                        "completion_receipt_sha256",
                        "parse_receipt_sha256",
                        "prompt_row_receipt_sha256",
                        "request_journal_sha256",
                        "response_journal_sha256",
                    )
                ),
                f"full100 passthrough result provenance {ordinal} changed",
            )
        question_ids.append(require_text(row.get("question_id"), "question ID"))
        validated.append(dict(projected))
    _require(
        len(set(question_ids)) == QUESTION_COUNT,
        "full100 answer result question identities repeat",
    )
    _require(
        payload.get("changed_prediction_count")
        == sum(bool(row.get("changed_from_parent")) for row in questions)
        and payload.get("invalid_completion_parent_fallback_count")
        == sum(
            row.get("prediction_source") == "typed_final_invalid_keep_parent_v1"
            for row in questions
        ),
        "full100 answer aggregate counts changed",
    )
    assert_gold_blind(payload, path="full100_terminal_answer_run")
    return tuple(validated)


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    sources = _load_verified_sources(args)
    preflight, prompts, prompt_rows, passthrough_rows = _read_preflight(
        args.output_root, str(args.expected_preflight_sha256)
    )
    _assert_preflight_source_binding(preflight, sources)
    _assert_preflight_promotion_binding(preflight, sources.promotion_audit)
    release = _read_release(
        args.output_root,
        str(args.expected_release_sha256),
        preflight=preflight,
    )
    batch = _checkpoint_batch(
        preflight, release, prompts, args=args, client=None
    )
    rebuilt = _materialization_payload(
        preflight, release, prompt_rows, passthrough_rows, batch
    )
    run = read_sealed_json(Path(args.output_root) / RUN_NAME)
    _require(
        run.sha256 == require_sha256(args.expected_run_sha256, "full100 answer run")
        and run.payload == rebuilt,
        "full100 answer run differs from checkpoint-only replay",
    )
    _validate_run(
        run,
        preflight=preflight,
        expected_release_sha256=release.sha256,
        expected_batch=batch,
    )
    replay_payload = {
        "byte_identical": True,
        "expected_run_sha256": run.sha256,
        "format": REPLAY_FORMAT,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "preflight_artifact_sha256": preflight.sha256,
        "replayed_run_sha256": run.sha256,
        "release_authorization_artifact_sha256": release.sha256,
        "retained_transformer_token_state_bytes": 0,
        **{key: preflight.payload[key] for key in SOURCE_BINDING_KEYS},
    }
    assert_gold_blind(replay_payload, path="full100_terminal_answer_replay")
    replay, _ = publish_sealed_json(Path(args.output_root) / REPLAY_NAME, replay_payload)
    return {
        "byte_identical": True,
        "physical_provider_calls": 0,
        "postseal_promotion_audit_sha256": sources.promotion_audit.sha256,
        "release_sha256": release.sha256,
        "replay_sha256": replay.sha256,
        "run_sha256": run.sha256,
    }


def load_verified_answer_run(
    output_root: str | Path,
    *,
    expected_preflight_sha256: str,
    expected_run_sha256: str,
    expected_replay_sha256: str,
    postseal_audit: str | Path,
    expected_postseal_audit_sha256: str,
) -> tuple[SealedArtifact, SealedArtifact, tuple[dict[str, Any], ...]]:
    root = Path(output_root)
    preflight, prompts, prompt_rows, passthrough_rows = _read_preflight(
        root, expected_preflight_sha256
    )
    audit = _read_promotion_audit(
        postseal_audit,
        expected_postseal_audit_sha256,
        construction_sha256=str(
            preflight.payload["promotion_terminal_construction_artifact_sha256"]
        ),
        replay_sha256=str(
            preflight.payload["promotion_terminal_replay_artifact_sha256"]
        ),
    )
    _assert_preflight_promotion_binding(preflight, audit)
    run = read_sealed_json(root / RUN_NAME)
    _require(
        run.sha256 == require_sha256(expected_run_sha256, "full100 answer run"),
        "full100 answer run artifact changed",
    )
    release_sha = require_sha256(
        run.payload.get("release_authorization_artifact_sha256"),
        "full100 provider release",
    )
    release = _read_release(root, release_sha, preflight=preflight)
    runtime_args = argparse.Namespace(
        gateway_url=preflight.payload["gateway_url"],
        max_concurrency=preflight.payload["max_concurrency"],
        model=preflight.payload["model"],
        output_root=root,
    )
    batch = _checkpoint_batch(
        preflight, release, prompts, args=runtime_args, client=None
    )
    rebuilt = _materialization_payload(
        preflight, release, prompt_rows, passthrough_rows, batch
    )
    _require(
        run.payload == rebuilt,
        "full100 public judge seam differs from checkpoint-only rebuild",
    )
    judge_rows = _validate_run(
        run,
        preflight=preflight,
        expected_release_sha256=release.sha256,
        expected_batch=batch,
    )
    replay = read_sealed_json(root / REPLAY_NAME)
    payload = replay.payload
    _require(
        set(payload) == REPLAY_KEYS
        and replay.sha256
        == require_sha256(expected_replay_sha256, "full100 answer replay")
        and payload.get("format") == REPLAY_FORMAT
        and payload.get("byte_identical") is True
        and payload.get("gold_loaded") is False
        and payload.get("physical_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("preflight_artifact_sha256") == preflight.sha256
        and payload.get("expected_run_sha256") == run.sha256
        and payload.get("replayed_run_sha256") == run.sha256
        and payload.get("release_authorization_artifact_sha256") == release.sha256
        and all(
            payload.get(key) == preflight.payload.get(key)
            for key in SOURCE_BINDING_KEYS
        ),
        "full100 answer source is not exact replay-verified",
    )
    return run, replay, judge_rows


def _add_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument(
        "--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY
    )


def _add_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--full100-terminal-root", type=Path, default=full100_cli.DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument("--expected-full100-construction-sha256", required=True)
    parser.add_argument("--expected-full100-replay-sha256", required=True)
    parser.add_argument(
        "--promotion-terminal-root", type=Path, default=terminal_cli.DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument(
        "--expected-promotion-terminal-construction-sha256", required=True
    )
    parser.add_argument("--expected-promotion-terminal-replay-sha256", required=True)
    _add_postseal(parser)


def _add_postseal(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--postseal-audit", type=Path, required=True)
    parser.add_argument("--expected-postseal-audit-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    _add_runtime(preflight)
    _add_sources(preflight)
    approve = commands.add_parser("approve-release")
    _add_runtime(approve)
    _add_sources(approve)
    approve.add_argument("--expected-preflight-sha256", required=True)
    approve.add_argument("--approve-provider-release", action="store_true")
    provider = commands.add_parser("provider-run")
    _add_runtime(provider)
    _add_postseal(provider)
    provider.add_argument("--expected-preflight-sha256", required=True)
    provider.add_argument("--expected-release-sha256", required=True)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, required=True)
    provider.add_argument("--api-key-env", default=live.DEFAULT_API_KEY_ENV)
    materialize = commands.add_parser("materialize")
    _add_runtime(materialize)
    _add_postseal(materialize)
    materialize.add_argument("--expected-preflight-sha256", required=True)
    materialize.add_argument("--expected-release-sha256", required=True)
    replay = commands.add_parser("replay")
    _add_runtime(replay)
    _add_sources(replay)
    replay.add_argument("--expected-preflight-sha256", required=True)
    replay.add_argument("--expected-release-sha256", required=True)
    replay.add_argument("--expected-run-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        result = run_preflight(args)
    elif args.command == "approve-release":
        result = run_approve_release(args)
    elif args.command == "provider-run":
        result = run_provider(args)
    elif args.command == "materialize":
        result = run_materialize(args)
    else:
        result = run_replay(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ALL_ORDINALS",
    "CHECKPOINT_DIR_NAME",
    "DEFAULT_OUTPUT_ROOT",
    "ELIGIBLE_COUNT",
    "FORMAT",
    "LockedSemanticGlobalTerminalFull100AnswerError",
    "PASSTHROUGH_COUNT",
    "PREFLIGHT_FORMAT",
    "PREFLIGHT_NAME",
    "QUESTION_COUNT",
    "RELEASE_FORMAT",
    "RELEASE_NAME",
    "REPLAY_FORMAT",
    "REPLAY_NAME",
    "RUN_FORMAT",
    "RUN_NAME",
    "build_parser",
    "build_preflight_payload",
    "load_verified_answer_run",
    "main",
    "run_approve_release",
    "run_materialize",
    "run_preflight",
    "run_provider",
    "run_replay",
]
