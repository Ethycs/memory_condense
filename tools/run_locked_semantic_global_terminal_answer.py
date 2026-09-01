#!/usr/bin/env python3
"""Run the thin exact-11 Terra lifecycle over the sealed P/R/L/G terminal.

The provider-free terminal assay owns retrieval, evidence compilation, prompt
fitting, and exact replay.  This file only authenticates its public answer-plan
rows, seals the resulting Terra prompt population, executes an exactly
authorized checkpointed batch, and delegates completion validation to the
shared typed-final implementation.

No phase loads benchmark gold.  ``provider-run`` reads only the sealed
preflight.  ``materialize`` reads only that preflight and immutable completion
checkpoints.  ``replay`` reauthenticates the terminal construction/replay and
requires byte-identical answer materialization using checkpoint hits only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
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


FORMAT = "memory-condense-locked-semantic-global-terminal-terra-answer-v2"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight-v1"
RUN_FORMAT = f"{FORMAT}-run-v1"
REPLAY_FORMAT = f"{FORMAT}-replay-v1"
RELEASE_FORMAT = f"{FORMAT}-provider-release-v1"
SUPERSESSION_FORMAT = (
    "memory-condense-locked-semantic-global-terminal-terra-answer-v1-"
    "supersession-v1"
)

PREFLIGHT_NAME = "semantic-global-terminal-terra-answer-preflight-v2.json"
RUN_NAME = "semantic-global-terminal-terra-answer-v2.json"
REPLAY_NAME = "semantic-global-terminal-terra-answer-replay-v2.json"
RELEASE_NAME = "semantic-global-terminal-terra-answer-provider-release-v2.json"
SUPERSESSION_NAME = "semantic-global-terminal-terra-answer-supersession-v1.json"
CHECKPOINT_DIR_NAME = "terra-semantic-global-terminal-exact11-v2-calls"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/"
    "locked-semantic-global-terminal-terra-answer-v2"
)
DEFAULT_MODEL = live.DEFAULT_TERRA_GATEWAY_MODEL
DEFAULT_GATEWAY_URL = live.DEFAULT_GATEWAY_URL
DEFAULT_MAX_CONCURRENCY = 4
EXACT_ORDINALS = terminal_cli.EXACT_ORDINALS
QUESTION_COUNT = len(EXACT_ORDINALS)
ROUTE_ID = "semantic-global-terminal-terra-answer-v2"
POSTSEAL_BINDING_KEYS = (
    "postseal_promotion_audit_artifact_sha256",
    "postseal_promotion_audit_identity_sha256",
    "postseal_semantic_atom_count",
    "postseal_semantic_atom_final_usable_count",
    "postseal_semantic_atom_manifest_artifact_sha256",
    "postseal_semantic_atom_manifest_identity_sha256",
    "postseal_semantic_atom_population_sha256",
    "postseal_source_final_usable_count",
    "postseal_source_target_count",
    "postseal_target_plan_artifact_sha256",
    "postseal_target_plan_identity_sha256",
    "postseal_witness_final_usable_count",
    "postseal_witness_manifest_artifact_sha256",
    "postseal_witness_manifest_identity_sha256",
    "postseal_witness_positive_count",
)

SUPERSEDED_TERMINAL_SHA256 = (
    "589b598af2365b719cc83465ff97816ec5d4808a9ef7136a3df80598578adeff"
)
SUPERSEDED_PREFLIGHT_SHA256 = (
    "ec6e8949da27f38ec6f258a46640b9ad8b312cee077ae009de717b806eef4148"
)


class LockedSemanticGlobalTerminalAnswerError(MatchedEvalContractError):
    """A terminal source, prompt, checkpoint, answer, or replay changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedSemanticGlobalTerminalAnswerError(message)


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact array")
    return value  # type: ignore[return-value]


def _canonical_root(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve(strict=False)))


def supersession_marker_payload() -> dict[str, Any]:
    body = {
        "format": SUPERSESSION_FORMAT,
        "preflight_artifact_sha256": SUPERSEDED_PREFLIGHT_SHA256,
        "provider_execution_allowed": False,
        "reason": "post_seal_target_fact_visibility_inadequate_20_of_26",
        "replacement_required": True,
        "status": "superseded_no_provider_execution",
        "terminal_construction_artifact_sha256": SUPERSEDED_TERMINAL_SHA256,
        "terminal_replay_artifact_sha256": SUPERSEDED_TERMINAL_SHA256,
    }
    return {**body, "supersession_identity_sha256": identity_sha256(body)}


def _validate_supersession_marker(artifact: SealedArtifact) -> dict[str, Any]:
    payload = artifact.payload
    body = {
        key: value
        for key, value in payload.items()
        if key != "supersession_identity_sha256"
    }
    _require(
        payload == supersession_marker_payload()
        and require_sha256(
            payload.get("supersession_identity_sha256"),
            "terminal answer supersession marker",
        )
        == identity_sha256(body),
        "terminal answer supersession marker changed",
    )
    return payload


def _assert_output_root_not_retired(output_root: str | Path) -> None:
    root = Path(output_root)
    marker_path = root / SUPERSESSION_NAME
    marker_sidecar = marker_path.with_name(marker_path.name + ".sha256")
    if not marker_path.exists() and not marker_sidecar.exists():
        return
    marker = read_sealed_json(marker_path)
    _validate_supersession_marker(marker)
    raise LockedSemanticGlobalTerminalAnswerError(
        "terminal answer root is explicitly superseded; provider execution forbidden"
    )


def _assert_preflight_not_superseded(preflight: SealedArtifact) -> None:
    payload = preflight.payload
    _require(
        preflight.sha256 != SUPERSEDED_PREFLIGHT_SHA256
        and payload.get("terminal_construction_artifact_sha256")
        != SUPERSEDED_TERMINAL_SHA256
        and payload.get("terminal_replay_artifact_sha256")
        != SUPERSEDED_TERMINAL_SHA256,
        "terminal answer preflight/terminal is superseded; provider execution forbidden",
    )


def _postseal_promotion_binding(artifact: SealedArtifact) -> dict[str, Any]:
    payload = artifact.payload
    totals = _exact_dict(payload.get("totals"), "post-seal promotion totals")
    binding = {
        "postseal_promotion_audit_artifact_sha256": artifact.sha256,
        "postseal_promotion_audit_identity_sha256": require_sha256(
            payload.get("audit_identity_sha256"), "post-seal promotion audit"
        ),
        "postseal_semantic_atom_count": totals.get("semantic_atom_count"),
        "postseal_semantic_atom_final_usable_count": totals.get(
            "semantic_atom_final_usable_count"
        ),
        "postseal_semantic_atom_manifest_artifact_sha256": require_sha256(
            payload.get("semantic_atom_manifest_artifact_sha256"),
            "post-seal semantic atom manifest",
        ),
        "postseal_semantic_atom_manifest_identity_sha256": require_sha256(
            payload.get("semantic_atom_manifest_identity_sha256"),
            "post-seal semantic atom manifest identity",
        ),
        "postseal_semantic_atom_population_sha256": require_sha256(
            payload.get("semantic_atom_population_sha256"),
            "post-seal semantic atom population",
        ),
        "postseal_source_final_usable_count": totals.get(
            "source_final_usable_count"
        ),
        "postseal_source_target_count": totals.get("source_target_count"),
        "postseal_target_plan_artifact_sha256": require_sha256(
            payload.get("target_plan_artifact_sha256"), "post-seal target plan"
        ),
        "postseal_target_plan_identity_sha256": require_sha256(
            payload.get("target_plan_identity_sha256"),
            "post-seal target plan identity",
        ),
        "postseal_witness_final_usable_count": totals.get(
            "raw_witness_final_usable_count",
            totals.get("fact_final_usable_count"),
        ),
        "postseal_witness_manifest_artifact_sha256": require_sha256(
            payload.get("witness_manifest_artifact_sha256"),
            "post-seal witness manifest",
        ),
        "postseal_witness_manifest_identity_sha256": require_sha256(
            payload.get("witness_manifest_identity_sha256"),
            "post-seal witness manifest identity",
        ),
        "postseal_witness_positive_count": totals.get("positive_witness_count"),
    }
    _require(
        payload.get("promotion_gate_passed") is True
        and binding["postseal_semantic_atom_count"]
        == binding["postseal_semantic_atom_final_usable_count"]
        == postseal_cli.SEMANTIC_ATOM_COUNT
        and binding["postseal_semantic_atom_manifest_artifact_sha256"]
        == postseal_cli.DEFAULT_SEMANTIC_ATOM_MANIFEST_SHA256
        and binding["postseal_semantic_atom_manifest_identity_sha256"]
        == postseal_cli.DEFAULT_SEMANTIC_ATOM_MANIFEST_IDENTITY_SHA256
        and binding["postseal_semantic_atom_population_sha256"]
        == postseal_cli.DEFAULT_SEMANTIC_ATOM_POPULATION_SHA256
        and binding["postseal_source_target_count"]
        == postseal_cli.SOURCE_TARGET_COUNT
        and type(binding["postseal_source_final_usable_count"]) is int
        and 0
        <= binding["postseal_source_final_usable_count"]
        <= postseal_cli.SOURCE_TARGET_COUNT
        and binding["postseal_witness_positive_count"]
        == postseal_cli.POSITIVE_WITNESS_COUNT
        and type(binding["postseal_witness_final_usable_count"]) is int
        and 0
        <= binding["postseal_witness_final_usable_count"]
        <= postseal_cli.POSITIVE_WITNESS_COUNT
        and binding["postseal_target_plan_artifact_sha256"]
        == postseal_cli.DEFAULT_TARGET_PLAN_SHA256
        and binding["postseal_target_plan_identity_sha256"]
        == postseal_cli.DEFAULT_TARGET_PLAN_IDENTITY_SHA256
        and binding["postseal_witness_manifest_artifact_sha256"]
        == postseal_cli.DEFAULT_WITNESS_MANIFEST_SHA256
        and binding["postseal_witness_manifest_identity_sha256"]
        == postseal_cli.DEFAULT_WITNESS_MANIFEST_IDENTITY_SHA256,
        "post-seal promotion binding changed",
    )
    return binding


def _read_postseal_promotion_audit(
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
        raise LockedSemanticGlobalTerminalAnswerError(
            "post-seal promotion audit is absent, invalid, or not promoted"
        ) from exc
    _postseal_promotion_binding(artifact)
    return artifact


def _assert_preflight_postseal_binding(
    preflight: SealedArtifact,
    promotion_audit: SealedArtifact,
) -> None:
    binding = _postseal_promotion_binding(promotion_audit)
    _require(
        all(preflight.payload.get(key) == value for key, value in binding.items()),
        "terminal answer preflight differs from promoted post-seal audit",
    )


def _validated_answer_plans(
    raw_plans: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    plans = tuple(raw_plans)
    _require(
        len(plans) == QUESTION_COUNT
        and tuple(row.get("ordinal") for row in plans) == EXACT_ORDINALS,
        "terminal answer-plan population/order changed",
    )
    output: list[dict[str, Any]] = []
    question_ids: list[str] = []
    plan_receipts: list[str] = []
    for ordinal, raw in zip(EXACT_ORDINALS, plans, strict=True):
        _require(type(raw) is dict, "terminal answer plan changed type")
        plan = dict(raw)
        declared = require_sha256(
            plan.get("answer_plan_receipt_sha256"), "terminal answer plan"
        )
        unsigned = dict(plan)
        unsigned.pop("answer_plan_receipt_sha256")
        provider_input = _exact_dict(
            plan.get("provider_input"), "terminal provider input"
        )
        terminal_compilation = _exact_dict(
            plan.get("terminal_compilation"), "terminal compilation"
        )
        allowed = _exact_list(
            plan.get("allowed_handle_ids"), "terminal allowed handles"
        )
        handle_groups = _exact_dict(
            plan.get("handle_group_by_id"), "terminal handle groups"
        )
        source_bindings = _exact_dict(
            plan.get("source_artifact_bindings"), "terminal source bindings"
        )
        dated_question = require_text(
            plan.get("dated_question"), "terminal dated question"
        )
        parent_prediction = require_text(
            plan.get("parent_prediction"), "terminal parent prediction"
        )
        question_id = require_text(
            plan.get("question_id"), "terminal question ID"
        )
        messages = render_final_messages(provider_input)
        prompt_tokens = count_chat_prompt_token_proxy(messages)
        _require(
            identity_sha256(unsigned) == declared
            and type(plan.get("format")) is str
            and bool(plan["format"])
            and plan.get("ordinal") == ordinal
            and quote_sha256(dated_question)
            == require_sha256(
                plan.get("dated_question_sha256"), "terminal dated question"
            )
            and quote_sha256(parent_prediction)
            == require_sha256(
                plan.get("parent_prediction_sha256"),
                "terminal parent prediction",
            )
            and require_sha256(
                plan.get("question_sha256"), "terminal question"
            )
            and terminal_compilation.get("receipt_sha256")
            == require_sha256(
                plan.get("terminal_compilation_receipt_sha256"),
                "terminal compilation",
            )
            and identity_sha256(provider_input)
            == require_sha256(
                plan.get("provider_input_sha256"), "terminal provider input"
            )
            and len(allowed) == len(set(allowed))
            and all(type(value) is str and bool(value) for value in allowed)
            and set(handle_groups) == set(allowed)
            and all(
                type(value) is str and bool(value)
                for value in handle_groups.values()
            )
            and type(plan.get("story_coherence")) is dict
            and type(plan.get("preservation_requirements")) is dict
            and type(plan.get("validation_contract")) is dict
            and bool(source_bindings)
            and all(
                type(key) is str
                and bool(key)
                and type(value) is str
                and bool(value)
                for key, value in source_bindings.items()
            )
            and identity_sha256(list(messages))
            == require_sha256(plan.get("messages_sha256"), "terminal messages")
            and prompt_tokens == plan.get("prompt_token_proxy")
            and plan.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
            and plan.get("hard_prompt_token_cap") == HARD_PROMPT_TOKEN_CAP
            and prompt_tokens <= MAX_CHAT_PROMPT_TOKENS
            and plan.get("route_id") == ROUTE_ID,
            f"terminal answer plan {ordinal} changed authenticated mirrors",
        )
        assert_gold_blind(
            provider_input,
            path=f"semantic_global_terminal_answer_provider_{ordinal}",
        )
        output.append(plan)
        question_ids.append(question_id)
        plan_receipts.append(declared)
    _require(
        len(set(question_ids)) == QUESTION_COUNT
        and len(set(plan_receipts)) == QUESTION_COUNT,
        "terminal answer plans repeat question/receipt identities",
    )
    return tuple(output)


def _prompt_plan_row(
    plan: Mapping[str, Any],
    *,
    construction_sha256: str,
    replay_sha256: str,
) -> dict[str, Any]:
    provider_input = _exact_dict(
        plan.get("provider_input"), "terminal prompt provider input"
    )
    messages = render_final_messages(provider_input)
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    body = {
        "allowed_handle_ids": list(plan["allowed_handle_ids"]),
        "dated_question_sha256": plan["dated_question_sha256"],
        "handle_group_by_id": dict(plan["handle_group_by_id"]),
        "messages": list(messages),
        "messages_sha256": identity_sha256(list(messages)),
        "ordinal": plan["ordinal"],
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
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
        "terminal_construction_artifact_sha256": construction_sha256,
        "terminal_replay_artifact_sha256": replay_sha256,
        "validation_contract": dict(plan["validation_contract"]),
    }
    _require(
        body["messages_sha256"] == plan["messages_sha256"]
        and body["prompt_token_proxy"] == plan["prompt_token_proxy"]
        and prompt_tokens <= MAX_CHAT_PROMPT_TOKENS,
        "terminal answer prompt differs from fitted provider bytes",
    )
    body["prompt_row_receipt_sha256"] = identity_sha256(body)
    assert_gold_blind(body, path="semantic_global_terminal_answer_prompt_row")
    return body


def build_preflight_payload(
    construction: SealedArtifact,
    replay: SealedArtifact,
    raw_plans: Sequence[Mapping[str, Any]],
    *,
    promotion_audit: SealedArtifact,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> tuple[dict[str, Any], tuple[tuple[dict[str, str], ...], ...]]:
    require_text(model, "terminal answer model")
    require_text(gateway_url, "terminal answer gateway")
    _require(
        model == DEFAULT_MODEL
        and gateway_url == DEFAULT_GATEWAY_URL
        and type(max_concurrency) is int
        and max_concurrency > 0,
        "terminal answer runtime policy changed",
    )
    promotion_binding = _postseal_promotion_binding(promotion_audit)
    _require(
        promotion_audit.payload.get("terminal_construction_sha256")
        == construction.sha256
        and promotion_audit.payload.get("terminal_replay_sha256") == replay.sha256,
        "post-seal promotion audit differs from terminal answer source",
    )
    plans = _validated_answer_plans(raw_plans)
    rows = tuple(
        _prompt_plan_row(
            plan,
            construction_sha256=construction.sha256,
            replay_sha256=replay.sha256,
        )
        for plan in plans
    )
    prompts = tuple(tuple(dict(message) for message in row["messages"]) for row in rows)
    population = preflight_fast_completion_prompts(
        prompts,
        max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS,
    )
    _require(
        population.logical_prompt_count
        == population.unique_prompt_count
        == QUESTION_COUNT,
        "terminal answer requires 11 distinct physical prompts",
    )
    plan_receipts = [
        row["terminal_answer_plan_receipt_sha256"] for row in rows
    ]
    payload = {
        "answer_plan_population_sha256": identity_sha256(plan_receipts),
        "exact_ordinals": list(EXACT_ORDINALS),
        "format": PREFLIGHT_FORMAT,
        "gateway_url": gateway_url,
        "gold_loaded": False,
        "hard_prompt_token_cap": HARD_PROMPT_TOKEN_CAP,
        "max_chat_prompt_tokens": MAX_CHAT_PROMPT_TOKENS,
        "max_concurrency": max_concurrency,
        "model": model,
        "observed_max_complete_envelope_tokens": max(
            row["prompt_token_proxy"] + OUTPUT_TOKEN_RESERVE for row in rows
        ),
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "physical_prompt_rows": list(rows),
        **promotion_binding,
        "prompt_population": population.model_dump(),
        "prompt_population_sha256": population.prompt_population_sha256,
        "provider_calls": 0,
        "question_count": QUESTION_COUNT,
        "required_authorized_provider_calls": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "terminal_construction_artifact_sha256": construction.sha256,
        "terminal_replay_artifact_sha256": replay.sha256,
    }
    assert_gold_blind(payload, path="semantic_global_terminal_answer_preflight")
    return payload, prompts


def _validate_preflight(
    artifact: SealedArtifact,
) -> tuple[tuple[tuple[dict[str, str], ...], ...], tuple[dict[str, Any], ...]]:
    payload = artifact.payload
    raw_rows = payload.get("physical_prompt_rows")
    for key in (
        "postseal_promotion_audit_artifact_sha256",
        "postseal_promotion_audit_identity_sha256",
        "postseal_semantic_atom_manifest_artifact_sha256",
        "postseal_semantic_atom_manifest_identity_sha256",
        "postseal_semantic_atom_population_sha256",
    ):
        require_sha256(payload.get(key), f"terminal answer {key}")
    _require(
        payload.get("format") == PREFLIGHT_FORMAT
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
        and payload.get("required_authorized_provider_calls") == QUESTION_COUNT
        and payload.get("exact_ordinals") == list(EXACT_ORDINALS)
        and payload.get("postseal_semantic_atom_count")
        == payload.get("postseal_semantic_atom_final_usable_count")
        == postseal_cli.SEMANTIC_ATOM_COUNT
        and payload.get("postseal_semantic_atom_manifest_artifact_sha256")
        == postseal_cli.DEFAULT_SEMANTIC_ATOM_MANIFEST_SHA256
        and payload.get("postseal_semantic_atom_manifest_identity_sha256")
        == postseal_cli.DEFAULT_SEMANTIC_ATOM_MANIFEST_IDENTITY_SHA256
        and payload.get("postseal_semantic_atom_population_sha256")
        == postseal_cli.DEFAULT_SEMANTIC_ATOM_POPULATION_SHA256
        and payload.get("postseal_source_target_count")
        == postseal_cli.SOURCE_TARGET_COUNT
        and type(payload.get("postseal_source_final_usable_count")) is int
        and 0
        <= payload["postseal_source_final_usable_count"]
        <= postseal_cli.SOURCE_TARGET_COUNT
        and payload.get("postseal_witness_positive_count")
        == postseal_cli.POSITIVE_WITNESS_COUNT
        and type(payload.get("postseal_witness_final_usable_count")) is int
        and 0
        <= payload["postseal_witness_final_usable_count"]
        <= postseal_cli.POSITIVE_WITNESS_COUNT
        and payload.get("postseal_target_plan_artifact_sha256")
        == postseal_cli.DEFAULT_TARGET_PLAN_SHA256
        and payload.get("postseal_target_plan_identity_sha256")
        == postseal_cli.DEFAULT_TARGET_PLAN_IDENTITY_SHA256
        and payload.get("postseal_witness_manifest_artifact_sha256")
        == postseal_cli.DEFAULT_WITNESS_MANIFEST_SHA256
        and payload.get("postseal_witness_manifest_identity_sha256")
        == postseal_cli.DEFAULT_WITNESS_MANIFEST_IDENTITY_SHA256
        and type(raw_rows) is list
        and len(raw_rows) == QUESTION_COUNT,
        "terminal answer sealed preflight firewall/population changed",
    )
    construct_sha = require_sha256(
        payload.get("terminal_construction_artifact_sha256"),
        "terminal answer construction",
    )
    replay_sha = require_sha256(
        payload.get("terminal_replay_artifact_sha256"),
        "terminal answer source replay",
    )
    prompts: list[tuple[dict[str, str], ...]] = []
    rows: list[dict[str, Any]] = []
    question_ids: list[str] = []
    plan_receipts: list[str] = []
    for ordinal, raw in zip(EXACT_ORDINALS, raw_rows, strict=True):
        _require(type(raw) is dict, "terminal answer prompt row changed type")
        row = dict(raw)
        declared = require_sha256(
            row.get("prompt_row_receipt_sha256"), "terminal answer prompt row"
        )
        unsigned = dict(row)
        unsigned.pop("prompt_row_receipt_sha256")
        raw_messages = _exact_list(
            row.get("messages"), "terminal answer prompt messages"
        )
        plain = tuple(
            {"role": message["role"], "content": message["content"]}
            for message in raw_messages
            if type(message) is dict
            and set(message) == {"role", "content"}
            and message.get("role") in {"system", "user", "assistant"}
            and type(message.get("content")) is str
        )
        _require(
            len(plain) == len(raw_messages)
            and identity_sha256(unsigned) == declared
            and row.get("ordinal") == ordinal
            and row.get("route_id") == ROUTE_ID
            and row.get("terminal_construction_artifact_sha256")
            == construct_sha
            and row.get("terminal_replay_artifact_sha256") == replay_sha
            and identity_sha256(list(plain)) == row.get("messages_sha256")
            and count_chat_prompt_token_proxy(plain)
            == row.get("prompt_token_proxy")
            and int(row["prompt_token_proxy"]) <= MAX_CHAT_PROMPT_TOKENS
            and row.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
            and quote_sha256(
                require_text(row.get("parent_prediction"), "terminal parent")
            )
            == row.get("parent_prediction_sha256")
            and set(_exact_dict(row.get("handle_group_by_id"), "handle groups"))
            == set(_exact_list(row.get("allowed_handle_ids"), "allowed handles")),
            f"terminal answer prompt row {ordinal} changed",
        )
        assert_gold_blind(
            list(plain), path=f"semantic_global_terminal_answer_prompt_{ordinal}"
        )
        prompts.append(plain)
        rows.append(row)
        question_ids.append(
            require_text(row.get("question_id"), "terminal prompt question")
        )
        plan_receipts.append(
            require_sha256(
                row.get("terminal_answer_plan_receipt_sha256"),
                "terminal answer plan",
            )
        )
    population = preflight_fast_completion_prompts(
        prompts,
        max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS,
    )
    _require(
        len(set(question_ids)) == QUESTION_COUNT
        and len(set(plan_receipts)) == QUESTION_COUNT
        and identity_sha256(plan_receipts)
        == payload.get("answer_plan_population_sha256")
        and population.model_dump() == payload.get("prompt_population")
        and population.prompt_population_sha256
        == payload.get("prompt_population_sha256")
        and population.unique_prompt_count == QUESTION_COUNT,
        "terminal answer sealed prompt population changed",
    )
    assert_gold_blind(payload, path="semantic_global_terminal_answer_preflight")
    return tuple(prompts), tuple(rows)


def _read_preflight(
    output_root: Path,
    expected_sha256: str,
) -> tuple[
    SealedArtifact,
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
]:
    artifact = read_sealed_json(output_root / PREFLIGHT_NAME)
    _require(
        artifact.sha256
        == require_sha256(expected_sha256, "terminal answer preflight"),
        "terminal answer preflight changed",
    )
    prompts, rows = _validate_preflight(artifact)
    return artifact, prompts, rows


def _assert_preflight_source_binding(
    preflight: SealedArtifact,
    construction: SealedArtifact,
    replay: SealedArtifact,
    plans: Sequence[Mapping[str, Any]],
    promotion_audit: SealedArtifact,
) -> None:
    rebuilt, _ = build_preflight_payload(
        construction,
        replay,
        plans,
        promotion_audit=promotion_audit,
        model=str(preflight.payload["model"]),
        gateway_url=str(preflight.payload["gateway_url"]),
        max_concurrency=int(preflight.payload["max_concurrency"]),
    )
    _require(
        rebuilt == preflight.payload,
        "terminal answer preflight differs from authenticated source plans",
    )


def _release_payload(
    *,
    preflight: SealedArtifact,
    terminal_construction_sha256: str,
    terminal_replay_sha256: str,
    terminal_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    terminal_root_value = _canonical_root(terminal_root)
    output_root_value = _canonical_root(output_root)
    payload = preflight.payload
    _require(
        payload.get("terminal_construction_artifact_sha256")
        == terminal_construction_sha256
        and payload.get("terminal_replay_artifact_sha256")
        == terminal_replay_sha256,
        "terminal answer release escaped its authenticated terminal source",
    )
    body = {
        "answer_output_root": output_root_value,
        "answer_output_root_sha256": identity_sha256(
            {"canonical_root": output_root_value}
        ),
        "answer_plan_population_sha256": payload[
            "answer_plan_population_sha256"
        ],
        "approval_opt_in": True,
        "exact_ordinals": list(EXACT_ORDINALS),
        "format": RELEASE_FORMAT,
        "gateway_url": payload["gateway_url"],
        "gold_loaded": False,
        "max_concurrency": payload["max_concurrency"],
        "model": payload["model"],
        "preflight_artifact_sha256": preflight.sha256,
        "prompt_population_sha256": payload["prompt_population_sha256"],
        **{key: payload[key] for key in POSTSEAL_BINDING_KEYS},
        "provider_calls_during_release": 0,
        "question_count": QUESTION_COUNT,
        "release_status": "approved_for_provider_execution",
        "required_authorized_provider_calls": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "supersession_checked": True,
        "terminal_construction_artifact_sha256": terminal_construction_sha256,
        "terminal_replay_artifact_sha256": terminal_replay_sha256,
        "terminal_root": terminal_root_value,
        "terminal_root_sha256": identity_sha256(
            {"canonical_root": terminal_root_value}
        ),
    }
    assert_gold_blind(body, path="semantic_global_terminal_answer_release")
    return {**body, "release_identity_sha256": identity_sha256(body)}


def _validate_release(
    artifact: SealedArtifact,
    *,
    preflight: SealedArtifact,
    output_root: str | Path,
) -> dict[str, Any]:
    payload = artifact.payload
    body = {
        key: value
        for key, value in payload.items()
        if key != "release_identity_sha256"
    }
    terminal_root = require_text(payload.get("terminal_root"), "terminal root")
    output_root_value = _canonical_root(output_root)
    _require(
        set(payload)
        == {
            "answer_output_root",
            "answer_output_root_sha256",
            "answer_plan_population_sha256",
            "approval_opt_in",
            "exact_ordinals",
            "format",
            "gateway_url",
            "gold_loaded",
            "max_concurrency",
            "model",
            "preflight_artifact_sha256",
            "prompt_population_sha256",
            "provider_calls_during_release",
            "question_count",
            "release_identity_sha256",
            "release_status",
            "required_authorized_provider_calls",
            "retained_transformer_token_state_bytes",
            "supersession_checked",
            "terminal_construction_artifact_sha256",
            "terminal_replay_artifact_sha256",
            "terminal_root",
            "terminal_root_sha256",
        }.union(POSTSEAL_BINDING_KEYS)
        and require_sha256(
            payload.get("release_identity_sha256"), "terminal answer release"
        )
        == identity_sha256(body)
        and payload.get("format") == RELEASE_FORMAT
        and payload.get("release_status") == "approved_for_provider_execution"
        and payload.get("approval_opt_in") is True
        and payload.get("supersession_checked") is True
        and payload.get("gold_loaded") is False
        and payload.get("provider_calls_during_release") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("required_authorized_provider_calls") == QUESTION_COUNT
        and payload.get("exact_ordinals") == list(EXACT_ORDINALS)
        and payload.get("preflight_artifact_sha256") == preflight.sha256
        and payload.get("answer_plan_population_sha256")
        == preflight.payload.get("answer_plan_population_sha256")
        and payload.get("prompt_population_sha256")
        == preflight.payload.get("prompt_population_sha256")
        and all(
            payload.get(key) == preflight.payload.get(key)
            for key in POSTSEAL_BINDING_KEYS
        )
        and payload.get("terminal_construction_artifact_sha256")
        == preflight.payload.get("terminal_construction_artifact_sha256")
        and payload.get("terminal_replay_artifact_sha256")
        == preflight.payload.get("terminal_replay_artifact_sha256")
        and payload.get("model") == preflight.payload.get("model")
        and payload.get("gateway_url") == preflight.payload.get("gateway_url")
        and payload.get("max_concurrency")
        == preflight.payload.get("max_concurrency")
        and payload.get("answer_output_root") == output_root_value
        and payload.get("answer_output_root_sha256")
        == identity_sha256({"canonical_root": output_root_value})
        and payload.get("terminal_root_sha256")
        == identity_sha256({"canonical_root": terminal_root}),
        "terminal answer provider release changed",
    )
    assert_gold_blind(payload, path="semantic_global_terminal_answer_release")
    return payload


def _read_release(
    output_root: str | Path,
    expected_sha256: str,
    *,
    preflight: SealedArtifact,
) -> SealedArtifact:
    _assert_output_root_not_retired(output_root)
    _assert_preflight_not_superseded(preflight)
    try:
        artifact = read_sealed_json(Path(output_root) / RELEASE_NAME)
    except MatchedEvalContractError as exc:
        raise LockedSemanticGlobalTerminalAnswerError(
            "terminal answer provider release is absent or invalid"
        ) from exc
    _require(
        artifact.sha256
        == require_sha256(expected_sha256, "terminal answer provider release"),
        "terminal answer provider release artifact changed",
    )
    _validate_release(artifact, preflight=preflight, output_root=output_root)
    return artifact


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    _assert_output_root_not_retired(output_root)
    _require(
        str(args.expected_terminal_construction_sha256)
        != SUPERSEDED_TERMINAL_SHA256
        and str(args.expected_terminal_replay_sha256)
        != SUPERSEDED_TERMINAL_SHA256
        and
        not (output_root / CHECKPOINT_DIR_NAME).exists(),
        "terminal answer preflight requires a non-superseded source and fresh absent checkpoint root",
    )
    construction, replay, plans = terminal_cli.load_verified_terminal_assay(
        args.terminal_root,
        str(args.expected_terminal_construction_sha256),
        str(args.expected_terminal_replay_sha256),
    )
    promotion_audit = _read_postseal_promotion_audit(
        args.postseal_audit,
        str(args.expected_postseal_audit_sha256),
        construction_sha256=construction.sha256,
        replay_sha256=replay.sha256,
    )
    payload, _ = build_preflight_payload(
        construction,
        replay,
        plans,
        promotion_audit=promotion_audit,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    artifact, created = publish_sealed_json(output_root / PREFLIGHT_NAME, payload)
    return {
        "created": created,
        "maximum_complete_prompt_envelope": payload[
            "observed_max_complete_envelope_tokens"
        ],
        "physical_provider_calls": 0,
        "postseal_promotion_audit_sha256": promotion_audit.sha256,
        "preflight_sha256": artifact.sha256,
        "question_count": QUESTION_COUNT,
        "required_authorized_provider_calls": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "terminal_construction_sha256": construction.sha256,
        "terminal_replay_sha256": replay.sha256,
    }


def run_approve_release(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    _require(
        args.approve_provider_release is True,
        "terminal answer release requires explicit provider-release approval",
    )
    _assert_output_root_not_retired(output_root)
    _require(
        not (output_root / CHECKPOINT_DIR_NAME).exists(),
        "terminal answer release requires an absent checkpoint root",
    )
    construction, replay, plans = terminal_cli.load_verified_terminal_assay(
        args.terminal_root,
        str(args.expected_terminal_construction_sha256),
        str(args.expected_terminal_replay_sha256),
    )
    promotion_audit = _read_postseal_promotion_audit(
        args.postseal_audit,
        str(args.expected_postseal_audit_sha256),
        construction_sha256=construction.sha256,
        replay_sha256=replay.sha256,
    )
    preflight, _, _ = _read_preflight(
        output_root, str(args.expected_preflight_sha256)
    )
    _assert_preflight_source_binding(
        preflight, construction, replay, plans, promotion_audit
    )
    _assert_preflight_postseal_binding(preflight, promotion_audit)
    _assert_preflight_not_superseded(preflight)
    payload = _release_payload(
        preflight=preflight,
        terminal_construction_sha256=construction.sha256,
        terminal_replay_sha256=replay.sha256,
        terminal_root=args.terminal_root,
        output_root=output_root,
    )
    artifact, created = publish_sealed_json(output_root / RELEASE_NAME, payload)
    return {
        "created": created,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "postseal_promotion_audit_sha256": promotion_audit.sha256,
        "preflight_sha256": preflight.sha256,
        "release_sha256": artifact.sha256,
        "retained_transformer_token_state_bytes": 0,
        "terminal_construction_sha256": construction.sha256,
        "terminal_replay_sha256": replay.sha256,
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
        and len(prompts) == QUESTION_COUNT,
        "terminal answer runtime differs from sealed preflight",
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
            "arm": ROUTE_ID,
            "authorized_unique_calls": QUESTION_COUNT,
            "experiment_format": RUN_FORMAT,
            "gateway_url": DEFAULT_GATEWAY_URL,
            "gold_loaded": False,
            "preflight_artifact_sha256": preflight.sha256,
            **{
                key: preflight.payload[key]
                for key in POSTSEAL_BINDING_KEYS
            },
            "release_authorization_artifact_sha256": release.sha256,
            "terminal_construction_artifact_sha256": preflight.payload[
                "terminal_construction_artifact_sha256"
            ],
            "terminal_replay_artifact_sha256": preflight.payload[
                "terminal_replay_artifact_sha256"
            ],
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
    """Authenticate every existing dedicated-root journal before dispatch.

    ``FastCompletionRuntime`` owns journal names, canonical bytes, receipts,
    request/runtime provenance, response bindings, and the unsafe incomplete-
    reservation rule.  Reuse that implementation rather than maintaining a
    second checkpoint parser in this thin adapter.
    """

    runtime = _runtime(preflight, release, prompts, args=args, client=None)
    try:
        with runtime._journal_guard():  # noqa: SLF001 - sole journal authority
            records = runtime._load_all_records()  # noqa: SLF001
    finally:
        runtime.close()
    _require(
        len(records) <= QUESTION_COUNT,
        "terminal answer checkpoint population escaped exact11",
    )
    return len(records)


def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    preflight, prompts, _ = _read_preflight(
        Path(args.output_root), str(args.expected_preflight_sha256)
    )
    promotion_audit = _read_postseal_promotion_audit(
        args.postseal_audit,
        str(args.expected_postseal_audit_sha256),
        construction_sha256=str(
            preflight.payload["terminal_construction_artifact_sha256"]
        ),
        replay_sha256=str(preflight.payload["terminal_replay_artifact_sha256"]),
    )
    _assert_preflight_postseal_binding(preflight, promotion_audit)
    release = _read_release(
        args.output_root,
        str(args.expected_release_sha256),
        preflight=preflight,
    )
    _require(
        args.enable_provider is True
        and type(args.authorized_provider_calls) is int
        and 0 <= args.authorized_provider_calls <= QUESTION_COUNT,
        "terminal answer provider requires a bounded Terra call authorization",
    )
    checkpoint_hits = _validated_checkpoint_hits(
        preflight, release, prompts, args=args
    )
    remaining = QUESTION_COUNT - checkpoint_hits
    _require(
        args.authorized_provider_calls == remaining,
        "terminal answer authorization must exactly equal remaining calls",
    )
    if remaining == 0:
        batch = _checkpoint_batch(
            preflight, release, prompts, args=args, client=None
        )
        _require(
            batch.usage.logical_calls
            == batch.usage.unique_calls
            == batch.usage.checkpoint_hits
            == QUESTION_COUNT
            and batch.usage.physical_calls == 0,
            "terminal answer completed checkpoint replay changed",
        )
        return {
            "authorized_remaining_provider_calls": 0,
            "checkpoint_hits": QUESTION_COUNT,
            "gold_loaded": False,
            "physical_provider_calls": 0,
            "postseal_promotion_audit_sha256": promotion_audit.sha256,
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
        batch.usage.logical_calls
        == batch.usage.unique_calls
        == QUESTION_COUNT
        and batch.usage.physical_calls + batch.usage.checkpoint_hits
        == QUESTION_COUNT
        and batch.usage.physical_calls <= args.authorized_provider_calls
        and batch.usage.checkpoint_hits >= checkpoint_hits,
        "terminal answer provider population changed",
    )
    return {
        "authorized_remaining_provider_calls": remaining,
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "gold_loaded": False,
        "physical_provider_calls": batch.usage.physical_calls,
        "postseal_promotion_audit_sha256": promotion_audit.sha256,
        "preflight_sha256": preflight.sha256,
        "release_sha256": release.sha256,
        "required_authorized_provider_calls": remaining,
        "retained_transformer_token_state_bytes": 0,
    }


def _materialization_payload(
    preflight: SealedArtifact,
    release: SealedArtifact,
    prompt_rows: tuple[dict[str, Any], ...],
    batch: FastCompletionBatch,
) -> dict[str, Any]:
    _require(
        batch.usage.logical_calls
        == batch.usage.unique_calls
        == batch.usage.checkpoint_hits
        == QUESTION_COUNT
        and batch.usage.physical_calls == 0
        and len(batch.logical_completions) == QUESTION_COUNT
        and len(batch.unique_records) == QUESTION_COUNT,
        "terminal answer materialization requires 11 checkpoint hits",
    )
    records = {row.messages_sha256: row for row in batch.unique_records}
    _require(
        len(records) == QUESTION_COUNT,
        "terminal answer completion identities repeat",
    )
    results: list[dict[str, Any]] = []
    for plan, completion in zip(
        prompt_rows, batch.logical_completions, strict=True
    ):
        record = records.get(plan["messages_sha256"])
        _require(
            record is not None
            and record.completion == completion
            and record.checkpoint_hit is True
            and record.physical_call is False,
            "terminal answer checkpoint record changed",
        )
        assert record is not None
        results.append(
            materialize_typed_final_result_row(
                plan,
                completion,
                completion_receipt_sha256=record.completion_sha256,
                call_key_sha256=record.call_key_sha256,
                request_journal_sha256=record.request_journal_sha256,
                response_journal_sha256=record.response_journal_sha256,
            )
        )
    judge_rows = [judge_row_projection(row) for row in results]
    _require(
        tuple(row["ordinal"] for row in results) == EXACT_ORDINALS
        and tuple(row["question_id"] for row in results)
        == tuple(row["question_id"] for row in judge_rows),
        "terminal answer judge seam changed population/order",
    )
    payload = {
        "changed_prediction_count": sum(
            bool(row["changed_from_parent"]) for row in results
        ),
        "completion_batch": batch.model_dump(),
        "exact_ordinals": list(EXACT_ORDINALS),
        "format": RUN_FORMAT,
        "gold_loaded": False,
        "invalid_completion_parent_fallback_count": sum(
            row["prediction_source"] == "typed_final_invalid_keep_parent_v1"
            for row in results
        ),
        "judge_rows": judge_rows,
        "physical_provider_calls_during_materialization": 0,
        **{key: preflight.payload[key] for key in POSTSEAL_BINDING_KEYS},
        "preflight_artifact_sha256": preflight.sha256,
        "question_count": QUESTION_COUNT,
        "questions": results,
        "release_authorization_artifact_sha256": release.sha256,
        "required_authorized_provider_calls": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "terminal_construction_artifact_sha256": preflight.payload[
            "terminal_construction_artifact_sha256"
        ],
        "terminal_replay_artifact_sha256": preflight.payload[
            "terminal_replay_artifact_sha256"
        ],
        "validator_policy_format": VALIDATOR_POLICY_FORMAT,
    }
    assert_gold_blind(payload, path="semantic_global_terminal_answer_run")
    return payload


def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    preflight, prompts, rows = _read_preflight(
        Path(args.output_root), str(args.expected_preflight_sha256)
    )
    promotion_audit = _read_postseal_promotion_audit(
        args.postseal_audit,
        str(args.expected_postseal_audit_sha256),
        construction_sha256=str(
            preflight.payload["terminal_construction_artifact_sha256"]
        ),
        replay_sha256=str(preflight.payload["terminal_replay_artifact_sha256"]),
    )
    _assert_preflight_postseal_binding(preflight, promotion_audit)
    release = _read_release(
        args.output_root,
        str(args.expected_release_sha256),
        preflight=preflight,
    )
    batch = _checkpoint_batch(
        preflight, release, prompts, args=args, client=None
    )
    payload = _materialization_payload(preflight, release, rows, batch)
    artifact, created = publish_sealed_json(Path(args.output_root) / RUN_NAME, payload)
    return {
        "changed_prediction_count": payload["changed_prediction_count"],
        "checkpoint_hits": QUESTION_COUNT,
        "created": created,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "postseal_promotion_audit_sha256": promotion_audit.sha256,
        "release_sha256": release.sha256,
        "run_sha256": artifact.sha256,
    }


def _validate_run(
    artifact: SealedArtifact,
    *,
    preflight: SealedArtifact,
    expected_release_sha256: str,
) -> tuple[dict[str, Any], ...]:
    payload = artifact.payload
    questions = payload.get("questions")
    judge_rows = payload.get("judge_rows")
    _require(
        payload.get("format") == RUN_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("physical_provider_calls_during_materialization") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("required_authorized_provider_calls") == QUESTION_COUNT
        and payload.get("exact_ordinals") == list(EXACT_ORDINALS)
        and payload.get("preflight_artifact_sha256") == preflight.sha256
        and all(
            payload.get(key) == preflight.payload.get(key)
            for key in POSTSEAL_BINDING_KEYS
        )
        and payload.get("release_authorization_artifact_sha256")
        == require_sha256(expected_release_sha256, "terminal answer provider release")
        and type(questions) is list
        and type(judge_rows) is list
        and len(questions) == len(judge_rows) == QUESTION_COUNT,
        "terminal answer run envelope changed",
    )
    validated: list[dict[str, Any]] = []
    question_ids: list[str] = []
    for ordinal, source, projected in zip(
        EXACT_ORDINALS, questions, judge_rows, strict=True
    ):
        _require(
            type(source) is dict and type(projected) is dict,
            "terminal answer result row changed type",
        )
        unsigned = dict(source)
        declared = unsigned.pop("source_row_sha256", None)
        prediction = require_text(
            source.get("prediction"), "terminal answer prediction"
        )
        _require(
            source.get("format") == RESULT_ROW_FORMAT
            and source.get("ordinal") == ordinal
            and declared == identity_sha256(unsigned)
            and source.get("prediction_sha256") == quote_sha256(prediction)
            and judge_row_projection(source) == projected,
            f"terminal answer result row {ordinal} changed",
        )
        question_ids.append(
            require_text(source.get("question_id"), "terminal answer question")
        )
        validated.append(dict(projected))
    _require(
        len(set(question_ids)) == QUESTION_COUNT,
        "terminal answer result question identities repeat",
    )
    assert_gold_blind(payload, path="semantic_global_terminal_answer_run")
    return tuple(validated)


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    construction, terminal_replay, plans = (
        terminal_cli.load_verified_terminal_assay(
            args.terminal_root,
            str(args.expected_terminal_construction_sha256),
            str(args.expected_terminal_replay_sha256),
        )
    )
    promotion_audit = _read_postseal_promotion_audit(
        args.postseal_audit,
        str(args.expected_postseal_audit_sha256),
        construction_sha256=construction.sha256,
        replay_sha256=terminal_replay.sha256,
    )
    preflight, prompts, rows = _read_preflight(
        Path(args.output_root), str(args.expected_preflight_sha256)
    )
    release = _read_release(
        args.output_root,
        str(args.expected_release_sha256),
        preflight=preflight,
    )
    _assert_preflight_source_binding(
        preflight, construction, terminal_replay, plans, promotion_audit
    )
    _assert_preflight_postseal_binding(preflight, promotion_audit)
    batch = _checkpoint_batch(
        preflight, release, prompts, args=args, client=None
    )
    rebuilt = _materialization_payload(preflight, release, rows, batch)
    expected_run = require_sha256(args.expected_run_sha256, "terminal answer run")
    run = read_sealed_json(Path(args.output_root) / RUN_NAME)
    _require(
        run.sha256 == expected_run and run.payload == rebuilt,
        "terminal answer run differs from checkpoint-only replay",
    )
    _validate_run(
        run,
        preflight=preflight,
        expected_release_sha256=release.sha256,
    )
    replay_payload = {
        "byte_identical": True,
        "expected_run_sha256": run.sha256,
        "format": REPLAY_FORMAT,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        **{key: preflight.payload[key] for key in POSTSEAL_BINDING_KEYS},
        "preflight_artifact_sha256": preflight.sha256,
        "replayed_run_sha256": run.sha256,
        "release_authorization_artifact_sha256": release.sha256,
        "retained_transformer_token_state_bytes": 0,
        "terminal_construction_artifact_sha256": construction.sha256,
        "terminal_replay_artifact_sha256": terminal_replay.sha256,
    }
    assert_gold_blind(replay_payload, path="semantic_global_terminal_answer_replay")
    replay, _ = publish_sealed_json(
        Path(args.output_root) / REPLAY_NAME, replay_payload
    )
    return {
        "byte_identical": True,
        "physical_provider_calls": 0,
        "postseal_promotion_audit_sha256": promotion_audit.sha256,
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
    """Return the stable gold-free judge seam after exact answer replay."""

    root = Path(output_root)
    preflight, _, _ = _read_preflight(root, expected_preflight_sha256)
    promotion_audit = _read_postseal_promotion_audit(
        postseal_audit,
        expected_postseal_audit_sha256,
        construction_sha256=str(
            preflight.payload["terminal_construction_artifact_sha256"]
        ),
        replay_sha256=str(preflight.payload["terminal_replay_artifact_sha256"]),
    )
    _assert_preflight_postseal_binding(preflight, promotion_audit)
    run = read_sealed_json(root / RUN_NAME)
    _require(
        run.sha256 == require_sha256(expected_run_sha256, "terminal answer run"),
        "terminal answer run artifact changed",
    )
    release_sha = require_sha256(
        run.payload.get("release_authorization_artifact_sha256"),
        "terminal answer provider release",
    )
    release = _read_release(
        root, release_sha, preflight=preflight
    )
    judge_rows = _validate_run(
        run,
        preflight=preflight,
        expected_release_sha256=release.sha256,
    )
    replay = read_sealed_json(root / REPLAY_NAME)
    payload = replay.payload
    _require(
        replay.sha256
        == require_sha256(expected_replay_sha256, "terminal answer replay")
        and payload.get("format") == REPLAY_FORMAT
        and payload.get("byte_identical") is True
        and payload.get("gold_loaded") is False
        and payload.get("physical_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and all(
            payload.get(key) == preflight.payload.get(key)
            for key in POSTSEAL_BINDING_KEYS
        )
        and payload.get("preflight_artifact_sha256") == preflight.sha256
        and payload.get("expected_run_sha256") == run.sha256
        and payload.get("replayed_run_sha256") == run.sha256
        and payload.get("release_authorization_artifact_sha256")
        == release.sha256
        and payload.get("terminal_construction_artifact_sha256")
        == preflight.payload.get("terminal_construction_artifact_sha256")
        and payload.get("terminal_replay_artifact_sha256")
        == preflight.payload.get("terminal_replay_artifact_sha256"),
        "terminal answer source is not exact replay-verified",
    )
    return run, replay, judge_rows


def _add_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument(
        "--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY
    )


def _add_terminal_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--terminal-root", type=Path, default=terminal_cli.DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument("--expected-terminal-construction-sha256", required=True)
    parser.add_argument("--expected-terminal-replay-sha256", required=True)


def _add_postseal_source(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--postseal-audit", type=Path, required=True)
    parser.add_argument("--expected-postseal-audit-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    _add_runtime(preflight)
    _add_terminal_sources(preflight)
    _add_postseal_source(preflight)
    approve = commands.add_parser("approve-release")
    _add_runtime(approve)
    _add_terminal_sources(approve)
    _add_postseal_source(approve)
    approve.add_argument("--expected-preflight-sha256", required=True)
    approve.add_argument("--approve-provider-release", action="store_true")
    provider = commands.add_parser("provider-run")
    _add_runtime(provider)
    _add_postseal_source(provider)
    provider.add_argument("--expected-preflight-sha256", required=True)
    provider.add_argument("--expected-release-sha256", required=True)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, required=True)
    provider.add_argument("--api-key-env", default=live.DEFAULT_API_KEY_ENV)
    materialize = commands.add_parser("materialize")
    _add_runtime(materialize)
    _add_postseal_source(materialize)
    materialize.add_argument("--expected-preflight-sha256", required=True)
    materialize.add_argument("--expected-release-sha256", required=True)
    replay = commands.add_parser("replay")
    _add_runtime(replay)
    _add_terminal_sources(replay)
    _add_postseal_source(replay)
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
    "CHECKPOINT_DIR_NAME",
    "DEFAULT_OUTPUT_ROOT",
    "EXACT_ORDINALS",
    "FORMAT",
    "LockedSemanticGlobalTerminalAnswerError",
    "PREFLIGHT_FORMAT",
    "PREFLIGHT_NAME",
    "POSTSEAL_BINDING_KEYS",
    "QUESTION_COUNT",
    "REPLAY_FORMAT",
    "REPLAY_NAME",
    "RELEASE_FORMAT",
    "RELEASE_NAME",
    "RUN_FORMAT",
    "RUN_NAME",
    "SUPERSESSION_FORMAT",
    "SUPERSESSION_NAME",
    "SUPERSEDED_PREFLIGHT_SHA256",
    "SUPERSEDED_TERMINAL_SHA256",
    "build_parser",
    "build_preflight_payload",
    "load_verified_answer_run",
    "main",
    "run_materialize",
    "run_approve_release",
    "run_preflight",
    "run_provider",
    "run_replay",
    "supersession_marker_payload",
]
