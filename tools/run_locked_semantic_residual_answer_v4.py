#!/usr/bin/env python3
"""Checkpointed Terra lifecycle for the locked V4 semantic-residual layer.

The provider population is defined only by the sealed construction: rows with
an independently bounded ``residual_synthesis`` prompt consume one completion;
all ineligible, overflow, or no-novel-evidence rows preserve the sealed V3
answer byte-for-byte.  Preflight is provider-free.  Materialization and replay
are checkpoint-only and fail closed to V3 on malformed or ungrounded output.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

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
from tools import run_locked_semantic_residual_construction_v4 as construction_v4  # noqa: E402
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
    judge_row_projection,
)
from tools.matched_eval.typed_operator_spec import normalize_term  # noqa: E402


FORMAT = "memory-condense-locked-semantic-residual-terra-answer-v4"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight-v1"
PLAN_FORMAT = f"{FORMAT}-answer-plan-v1"
PARSE_FORMAT = f"{FORMAT}-parsed-decision-v1"
RESULT_ROW_FORMAT = f"{FORMAT}-result-row-v1"

PREFLIGHT_NAME = "locked-semantic-residual-answer-preflight-v4.json"
RUN_NAME = "locked-semantic-residual-answer-v4.json"
REPLAY_NAME = "locked-semantic-residual-answer-replay-v4.json"
CHECKPOINT_DIR_NAME = "locked-semantic-residual-answer-checkpoints-v4"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONSTRUCTION = construction_v4.DEFAULT_OUTPUT_ROOT / construction_v4.CONSTRUCTION_NAME
DEFAULT_CONSTRUCTION_REPLAY = (
    construction_v4.DEFAULT_OUTPUT_ROOT / construction_v4.REPLAY_NAME
)
DEFAULT_GATE = construction_v4.DEFAULT_GATE
DEFAULT_ANSWER = construction_v4.DEFAULT_ANSWER
DEFAULT_OUTPUT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-semantic-residual-answer-v4-r7"
)
DEFAULT_MODEL = "codex_sdk/gpt-5.6-terra"

QUESTION_COUNT = 100
HARD_COMPLETE_CHAT_TOKEN_CAP = 8_000
OUTPUT_TOKEN_RESERVE = 768
MAX_CHAT_PROMPT_TOKENS = HARD_COMPLETE_CHAT_TOKEN_CAP - OUTPUT_TOKEN_RESERVE

RESIDUAL_MODE = "residual_synthesis"
PASSTHROUGH_MODE = "v3_passthrough"
_EVIDENCE_HANDLE_RE = re.compile(r"^[RP][0-9]{4}$")
_RESIDUAL_HANDLE_RE = re.compile(r"^R[0-9]{4}$")
_PROTECTED_HANDLE_RE = re.compile(r"^P[0-9]{4}$")
_LEXEME_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*|[0-9]+(?:\.[0-9]+)?", re.UNICODE)
_NUMERIC_ANCHOR_RE = re.compile(
    r"(?<![\w.])(?P<currency>[$€£])?\s*(?P<number>[+-]?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(?P<percent>%|percent\b)?",
    re.IGNORECASE,
)
_PREDICTION_BOILERPLATE = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "being",
        "both", "but", "by", "did", "do", "does", "for", "from", "had",
        "has", "have", "he", "her", "hers", "him", "his", "i", "in",
        "is", "it", "its", "me", "mine", "my", "of", "on", "or", "our",
        "ours", "she", "so", "that", "the", "their", "theirs", "them",
        "they", "there", "this", "those", "to", "was", "were", "with", "you",
        "your", "yours",
    }
)
_ANCHOR_ALIASES = {
    "chose": "choose",
    "chosen": "choose",
    "choose": "choose",
    "select": "choose",
    "selected": "choose",
    "colour": "color",
}


class LockedSemanticResidualAnswerError(MatchedEvalContractError):
    """A residual prompt, V3 fallback, completion, or checkpoint changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedSemanticResidualAnswerError(message)


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact array")
    return value  # type: ignore[return-value]


def _with_receipt(body: Mapping[str, Any], key: str = "receipt_sha256") -> dict[str, Any]:
    return {**dict(body), key: identity_sha256(body)}


def _verified(path: Path, expected_sha256: str, label: str) -> SealedArtifact:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == require_sha256(expected_sha256, label),
        f"{label} artifact changed",
    )
    assert_gold_blind(artifact.payload, path=f"semantic_residual_answer.{label}")
    return artifact


def _verified_construction_replay(
    path: Path,
    expected_sha256: str,
    construction: SealedArtifact,
) -> SealedArtifact:
    _require(
        path.is_file() and not path.is_symlink(),
        "construction replay is absent or not an exact file",
    )
    replay = _verified(path, expected_sha256, "construction replay")
    _require(
        replay.sha256 == construction.sha256
        and replay.payload == construction.payload,
        "construction replay is not byte-identical",
    )
    return replay


def _self_hashed(row: Mapping[str, Any], key: str, label: str) -> None:
    body = dict(row)
    declared = require_sha256(body.pop(key, None), label)
    _require(identity_sha256(body) == declared, f"{label} changed")


def _source_rows(
    construction: SealedArtifact,
    gate: SealedArtifact,
    answer: SealedArtifact,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    construction_rows = [
        _exact_dict(row, "residual construction row")
        for row in _exact_list(construction.payload.get("questions"), "construction rows")
    ]
    gate_rows = [
        _exact_dict(row, "residual gate row")
        for row in _exact_list(gate.payload.get("questions"), "gate rows")
    ]
    answer_rows = [
        _exact_dict(row, "V3 answer row")
        for row in _exact_list(answer.payload.get("questions"), "V3 answer rows")
    ]
    _require(
        len(construction_rows) == len(gate_rows) == len(answer_rows) == QUESTION_COUNT
        and construction.payload.get("format") == construction_v4.CONSTRUCTION_FORMAT
        and construction.payload.get("bindings", {}).get("gate_artifact_sha256")
        == gate.sha256
        and gate.payload.get("bindings", {}).get("answer_artifact_sha256")
        == answer.sha256
        and construction.payload.get("gold_loaded") is False
        and gate.payload.get("gold_loaded") is False
        and answer.payload.get("gold_loaded") is False,
        "residual construction/gate/V3 population binding changed",
    )
    construction_bindings = _exact_dict(
        construction.payload.get("bindings"), "residual construction bindings"
    )
    gate_replay = _exact_dict(
        construction.payload.get("gate_exact_rebuild_replay"),
        "construction gate replay",
    )
    _self_hashed(gate_replay, "receipt_sha256", "construction gate replay")
    _require(
        construction.payload.get("separate_exact_construction_replay_required")
        is True
        and construction_bindings.get("query_vector_replay_artifact_sha256")
        == construction_bindings.get("query_vector_artifact_sha256")
        and construction_bindings.get(
            "gate_exact_rebuild_replay_receipt_sha256"
        )
        == gate_replay.get("receipt_sha256")
        and gate_replay.get("byte_identical") is True
        and gate_replay.get("gate_artifact_sha256") == gate.sha256,
        "construction gate/vector replay binding changed",
    )
    for ordinal, (built, gated, current) in enumerate(
        zip(construction_rows, gate_rows, answer_rows, strict=True)
    ):
        _self_hashed(built, "question_receipt_sha256", "construction question receipt")
        _self_hashed(gated, "gate_row_receipt_sha256", "gate question receipt")
        _self_hashed(current, "source_row_sha256", "V3 answer source row")
        _require(
            built.get("ordinal") == gated.get("ordinal") == current.get("ordinal") == ordinal
            and built.get("question_id") == gated.get("question_id") == current.get("question_id")
            and built.get("question_sha256") == gated.get("question_sha256")
            == current.get("question_sha256")
            and built.get("dated_question_sha256") == gated.get("dated_question_sha256")
            == current.get("dated_question_sha256")
            and gated.get("current_prediction") == current.get("prediction")
            and gated.get("current_prediction_sha256") == current.get("prediction_sha256"),
            f"residual answer sources escaped question binding at ordinal {ordinal}",
        )
    return construction_rows, gate_rows, answer_rows


def _load_sources(
    *,
    construction_path: Path,
    construction_sha256: str,
    construction_replay_path: Path,
    construction_replay_sha256: str,
    gate_path: Path,
    gate_sha256: str,
    answer_path: Path,
    answer_sha256: str,
) -> tuple[
    SealedArtifact,
    SealedArtifact,
    SealedArtifact,
    SealedArtifact,
    tuple[dict[str, Any], ...],
]:
    construction = _verified(construction_path, construction_sha256, "construction")
    construction_replay = _verified_construction_replay(
        construction_replay_path,
        construction_replay_sha256,
        construction,
    )
    gate = _verified(gate_path, gate_sha256, "gate")
    answer = _verified(answer_path, answer_sha256, "V3 answer")
    construction_rows, gate_rows, answer_rows = _source_rows(construction, gate, answer)
    plans = tuple(
        _answer_plan(built, gated, current)
        for built, gated, current in zip(
            construction_rows, gate_rows, answer_rows, strict=True
        )
    )
    return construction, construction_replay, gate, answer, plans


def _answer_plan(
    built: Mapping[str, Any],
    gated: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    ordinal = int(built["ordinal"])
    terminal = built.get("terminal_prompt")
    mode = built.get("mode")
    common: dict[str, Any] = {
        "construction_question_receipt_sha256": built["question_receipt_sha256"],
        "current_prediction": current["prediction"],
        "current_prediction_sha256": current["prediction_sha256"],
        "dated_question_sha256": current["dated_question_sha256"],
        "format": PLAN_FORMAT,
        "gate_row_receipt_sha256": gated["gate_row_receipt_sha256"],
        "ordinal": ordinal,
        "question_id": current["question_id"],
        "question_sha256": current["question_sha256"],
        "route_id": current["route_id"],
        "source_v3_answer_row_sha256": current["source_row_sha256"],
    }
    if mode != RESIDUAL_MODE:
        _require(
            terminal is None and mode in {"not_eligible", "residual_unavailable"},
            f"non-residual row carries a prompt at ordinal {ordinal}",
        )
        return _with_receipt(
            {
                **common,
                "fallback_reason": built.get("fallback_reason"),
                "mode": PASSTHROUGH_MODE,
                "source_construction_mode": mode,
            },
            "answer_plan_receipt_sha256",
        )

    prompt = _exact_dict(terminal, f"terminal prompt {ordinal}")
    messages = _exact_list(prompt.get("messages"), f"terminal messages {ordinal}")
    provider = _exact_dict(prompt.get("provider_input"), f"provider input {ordinal}")
    residual_evidence = _exact_list(
        provider.get("residual_evidence"), f"residual evidence {ordinal}"
    )
    owner_evidence = _exact_list(
        provider.get("protected_owner_evidence"),
        f"protected owner evidence {ordinal}",
    )
    residual_handles: list[str] = []
    owner_handles: list[str] = []
    evidence_rows: list[dict[str, Any]] = []
    for handle_class, raw_rows, handle_re, handles in (
        ("residual", residual_evidence, _RESIDUAL_HANDLE_RE, residual_handles),
        ("protected_owner", owner_evidence, _PROTECTED_HANDLE_RE, owner_handles),
    ):
        for raw in raw_rows:
            row = _exact_dict(raw, f"{handle_class} evidence item {ordinal}")
            handle = row.get("evidence_handle")
            quote = row.get("quote")
            _require(
                type(handle) is str
                and handle_re.fullmatch(handle) is not None
                and handle not in handles
                and type(quote) is str
                and bool(quote),
                f"{handle_class} provider evidence changed at ordinal {ordinal}",
            )
            handles.append(handle)
            evidence_rows.append(
                {
                    "evidence_handle": handle,
                    "handle_class": handle_class,
                    "quote": quote,
                    "quote_sha256": quote_sha256(quote),
                    "source_group_handle": row.get("source_group_handle"),
                }
            )
    handles = [*residual_handles, *owner_handles]
    closure = _exact_dict(
        provider.get("lossless_post_selection_closure"),
        f"lossless owner closure {ordinal}",
    )
    closure_rows = _exact_list(
        closure.get("rows"), f"lossless owner closure rows {ordinal}"
    )
    response_schema = _exact_dict(
        provider.get("response_schema"), f"response schema {ordinal}"
    )
    message_rows = [_exact_dict(row, f"message {ordinal}") for row in messages]
    prompt_tokens = count_chat_prompt_token_proxy(message_rows)
    _require(
        bool(residual_handles)
        and len(handles) == len(set(handles))
        and provider.get("current_answer") == current.get("prediction")
        and provider.get("dated_question") == gated.get("dated_question")
        and response_schema.get("used_evidence_handle_ids") == handles
        and response_schema.get("replacement_requires_at_least_one_residual_handle")
        == residual_handles
        and closure.get("every_removed_duplicate_has_exact_provider_visible_owner")
        is True
        and closure.get("owner_count") == len(owner_handles) == len(closure_rows)
        and [row.get("evidence_handle") for row in closure_rows] == owner_handles
        and prompt.get("provider_visible_selected_union_lossless") is True
        and prompt.get("residual_evidence_accounting", {}).get("within_cap") is True
        and prompt.get("protected_owner_evidence_accounting", {}).get("within_cap")
        is True
        and prompt.get("messages_sha256") == identity_sha256(message_rows)
        and prompt.get("provider_input_sha256") == identity_sha256(provider)
        and prompt.get("prompt_token_proxy") == prompt_tokens
        and prompt.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
        and prompt.get("hard_complete_chat_token_cap")
        == HARD_COMPLETE_CHAT_TOKEN_CAP
        and prompt.get("complete_chat_plus_output_tokens")
        == prompt_tokens + OUTPUT_TOKEN_RESERVE
        <= HARD_COMPLETE_CHAT_TOKEN_CAP
        and prompt.get("non_borrowable_residual_budget") is True
        and prompt.get("parent_prompt_tokens_borrowed") == 0
        and prompt.get("new_provider_calls") == 0
        and prompt.get("retained_transformer_token_state_bytes") == 0,
        f"residual terminal prompt changed at ordinal {ordinal}",
    )
    return _with_receipt(
        {
            **common,
            "allowed_evidence_handle_ids": handles,
            "evidence_grounding_rows": evidence_rows,
            "messages": message_rows,
            "messages_sha256": prompt["messages_sha256"],
            "mode": RESIDUAL_MODE,
            "prompt_token_proxy": prompt_tokens,
            "provider_input_sha256": prompt["provider_input_sha256"],
            "required_residual_handle_ids": residual_handles,
            "source_construction_mode": mode,
            "terminal_prompt_receipt_sha256": prompt[
                "terminal_prompt_receipt_sha256"
            ],
        },
        "answer_plan_receipt_sha256",
    )


def _validate_plan(raw: object) -> dict[str, Any]:
    plan = _exact_dict(raw, "residual answer plan")
    body = dict(plan)
    declared = require_sha256(
        body.pop("answer_plan_receipt_sha256", None), "residual answer plan"
    )
    _require(
        plan.get("format") == PLAN_FORMAT
        and identity_sha256(body) == declared
        and plan.get("mode") in {RESIDUAL_MODE, PASSTHROUGH_MODE}
        and type(plan.get("ordinal")) is int
        and 0 <= plan["ordinal"] < QUESTION_COUNT,
        "sealed residual answer plan changed",
    )
    if plan["mode"] == RESIDUAL_MODE:
        messages = _exact_list(plan.get("messages"), "residual answer messages")
        handles = _exact_list(
            plan.get("allowed_evidence_handle_ids"), "allowed evidence handles"
        )
        residual_handles = _exact_list(
            plan.get("required_residual_handle_ids"), "required residual handles"
        )
        evidence = _exact_list(
            plan.get("evidence_grounding_rows"), "residual grounding rows"
        )
        _require(
            bool(handles)
            and bool(residual_handles)
            and len(handles) == len(set(handles)) == len(evidence)
            and set(residual_handles) <= set(handles)
            and all(
                type(value) is str
                and _EVIDENCE_HANDLE_RE.fullmatch(value) is not None
                for value in handles
            )
            and all(
                type(value) is str
                and _RESIDUAL_HANDLE_RE.fullmatch(value) is not None
                for value in residual_handles
            )
            and [row.get("evidence_handle") for row in evidence] == handles
            and plan.get("messages_sha256") == identity_sha256(messages)
            and plan.get("prompt_token_proxy")
            == count_chat_prompt_token_proxy(messages)
            and plan["prompt_token_proxy"] + OUTPUT_TOKEN_RESERVE
            <= HARD_COMPLETE_CHAT_TOKEN_CAP,
            "sealed residual provider plan changed",
        )
    else:
        _require(
            "messages" not in plan and "allowed_evidence_handle_ids" not in plan,
            "V3 passthrough acquired provider material",
        )
    assert_gold_blind(plan, path="semantic_residual_answer.plan")
    return plan


def build_preflight_payload(
    construction: SealedArtifact,
    construction_replay: SealedArtifact,
    gate: SealedArtifact,
    answer: SealedArtifact,
    plans: Sequence[Mapping[str, Any]],
    *,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> dict[str, Any]:
    _require(
        model == DEFAULT_MODEL
        and type(max_concurrency) is int
        and max_concurrency > 0,
        "residual Terra runtime settings changed",
    )
    require_text(gateway_url, "residual Terra gateway")
    validated = tuple(_validate_plan(dict(row)) for row in plans)
    _require(
        len(validated) == QUESTION_COUNT
        and tuple(row["ordinal"] for row in validated) == tuple(range(QUESTION_COUNT)),
        "residual answer plan population changed",
    )
    physical = tuple(row for row in validated if row["mode"] == RESIDUAL_MODE)
    passthrough = tuple(row for row in validated if row["mode"] == PASSTHROUGH_MODE)
    _require(bool(physical), "residual Terra prompt population is empty")
    prompts = tuple(tuple(dict(message) for message in row["messages"]) for row in physical)
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS
    )
    observed_max = max(
        row["prompt_token_proxy"] + OUTPUT_TOKEN_RESERVE for row in physical
    )
    _require(
        population.logical_prompt_count
        == population.unique_prompt_count
        == len(physical)
        and observed_max <= HARD_COMPLETE_CHAT_TOKEN_CAP,
        "residual Terra prompt population duplicated or exceeded 8k",
    )
    payload = {
        "answer_artifact_sha256": answer.sha256,
        "answer_plan_population_sha256": identity_sha256(
            [row["answer_plan_receipt_sha256"] for row in validated]
        ),
        "construction_artifact_sha256": construction.sha256,
        "construction_replay_artifact_sha256": construction_replay.sha256,
        "format": PREFLIGHT_FORMAT,
        "gateway_url": gateway_url,
        "gate_artifact_sha256": gate.sha256,
        "gold_loaded": False,
        "hard_complete_chat_token_cap": HARD_COMPLETE_CHAT_TOKEN_CAP,
        "max_chat_prompt_tokens": MAX_CHAT_PROMPT_TOKENS,
        "max_concurrency": max_concurrency,
        "model": model,
        "observed_max_complete_envelope_tokens": observed_max,
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "passthrough_count": len(passthrough),
        "passthrough_rows": list(passthrough),
        "physical_prompt_rows": list(physical),
        "prompt_population": population.model_dump(),
        "prompt_population_sha256": population.prompt_population_sha256,
        "provider_calls": 0,
        "question_count": QUESTION_COUNT,
        "required_authorized_provider_calls": len(physical),
        "retained_transformer_token_state_bytes": 0,
        "selection_and_routing_frozen_before_provider": True,
    }
    assert_gold_blind(payload, path="semantic_residual_answer.preflight")
    return payload


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    construction, construction_replay, gate, answer, plans = _load_sources(
        construction_path=Path(args.construction),
        construction_sha256=str(args.expected_construction_sha256),
        construction_replay_path=Path(args.construction_replay),
        construction_replay_sha256=str(
            args.expected_construction_replay_sha256
        ),
        gate_path=Path(args.gate),
        gate_sha256=str(args.expected_gate_sha256),
        answer_path=Path(args.answer),
        answer_sha256=str(args.expected_answer_sha256),
    )
    payload = build_preflight_payload(
        construction,
        construction_replay,
        gate,
        answer,
        plans,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    artifact, created = publish_sealed_json(
        Path(args.output_root) / PREFLIGHT_NAME, payload
    )
    return {
        "created": created,
        "maximum_complete_prompt_envelope": payload[
            "observed_max_complete_envelope_tokens"
        ],
        "passthrough_count": payload["passthrough_count"],
        "physical_provider_calls": 0,
        "preflight_sha256": artifact.sha256,
        "question_count": QUESTION_COUNT,
        "required_authorized_provider_calls": payload[
            "required_authorized_provider_calls"
        ],
    }


def _validate_preflight(
    artifact: SealedArtifact,
) -> tuple[tuple[tuple[dict[str, str], ...], ...], tuple[dict[str, Any], ...]]:
    payload = artifact.payload
    assert_gold_blind(payload, path="semantic_residual_answer.loaded_preflight")
    physical = _exact_list(payload.get("physical_prompt_rows"), "physical prompt rows")
    passthrough = _exact_list(payload.get("passthrough_rows"), "passthrough rows")
    plans = tuple(
        sorted(
            (_validate_plan(row) for row in (*physical, *passthrough)),
            key=lambda row: row["ordinal"],
        )
    )
    _require(
        payload.get("format") == PREFLIGHT_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("provider_calls") == 0
        and payload.get("model") == DEFAULT_MODEL
        and payload.get("hard_complete_chat_token_cap")
        == HARD_COMPLETE_CHAT_TOKEN_CAP
        and payload.get("max_chat_prompt_tokens") == MAX_CHAT_PROMPT_TOKENS
        and payload.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
        and len(physical) == payload.get("required_authorized_provider_calls")
        and len(passthrough) == payload.get("passthrough_count")
        and len(plans) == payload.get("question_count") == QUESTION_COUNT
        and tuple(row["ordinal"] for row in plans) == tuple(range(QUESTION_COUNT))
        and payload.get("answer_plan_population_sha256")
        == identity_sha256([row["answer_plan_receipt_sha256"] for row in plans])
        and payload.get("construction_replay_artifact_sha256")
        == payload.get("construction_artifact_sha256")
        and payload.get("retained_transformer_token_state_bytes") == 0,
        "sealed residual answer preflight changed",
    )
    prompts = tuple(
        tuple(dict(message) for message in row["messages"])
        for row in plans
        if row["mode"] == RESIDUAL_MODE
    )
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS
    )
    observed = max(
        row["prompt_token_proxy"] + OUTPUT_TOKEN_RESERVE
        for row in plans
        if row["mode"] == RESIDUAL_MODE
    )
    _require(
        population.model_dump() == payload.get("prompt_population")
        and population.prompt_population_sha256
        == payload.get("prompt_population_sha256")
        and population.logical_prompt_count
        == population.unique_prompt_count
        == len(physical)
        and payload.get("observed_max_complete_envelope_tokens") == observed
        <= HARD_COMPLETE_CHAT_TOKEN_CAP,
        "sealed residual prompt population changed",
    )
    return prompts, plans


def _read_preflight(
    output_root: Path, expected_sha256: str
) -> tuple[SealedArtifact, tuple[tuple[dict[str, str], ...], ...], tuple[dict[str, Any], ...]]:
    artifact = read_sealed_json(output_root / PREFLIGHT_NAME)
    _require(
        artifact.sha256 == require_sha256(expected_sha256, "residual preflight"),
        "residual answer preflight digest changed",
    )
    prompts, plans = _validate_preflight(artifact)
    return artifact, prompts, plans


def _assert_preflight_source_binding(
    preflight: SealedArtifact,
    construction: SealedArtifact,
    construction_replay: SealedArtifact,
    gate: SealedArtifact,
    answer: SealedArtifact,
    plans: Sequence[Mapping[str, Any]],
) -> None:
    _require(
        preflight.payload.get("construction_artifact_sha256")
        == construction.sha256
        and preflight.payload.get("construction_replay_artifact_sha256")
        == construction_replay.sha256
        and construction_replay.sha256 == construction.sha256
        and preflight.payload.get("gate_artifact_sha256") == gate.sha256
        and preflight.payload.get("answer_artifact_sha256") == answer.sha256
        and tuple(plans)
        == tuple(
            sorted(
                (
                    *preflight.payload["physical_prompt_rows"],
                    *preflight.payload["passthrough_rows"],
                ),
                key=lambda row: row["ordinal"],
            )
        ),
        "residual construction replay/preflight binding changed",
    )


def _runtime(
    preflight: SealedArtifact,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    args: argparse.Namespace,
    client: Any | None,
) -> FastCompletionRuntime:
    required = preflight.payload["required_authorized_provider_calls"]
    _require(
        str(args.model) == preflight.payload.get("model") == DEFAULT_MODEL
        and str(args.gateway_url) == preflight.payload.get("gateway_url")
        and int(args.max_concurrency) == preflight.payload.get("max_concurrency")
        and len(prompts) == required,
        "residual answer runtime differs from preflight",
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
            "arm": "locked_semantic_residual_terra_answer_v4",
            "authorized_unique_calls": required,
            "construction_artifact_sha256": preflight.payload[
                "construction_artifact_sha256"
            ],
            "experiment_format": FORMAT,
            "gateway_url": str(args.gateway_url),
            "gold_loaded": False,
            "preflight_artifact_sha256": preflight.sha256,
        },
    )


def _checkpoint_batch(
    preflight: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    args: argparse.Namespace,
    client: Any | None,
) -> FastCompletionBatch:
    runtime = _runtime(preflight, prompts, args=args, client=client)
    try:
        return runtime.run()
    finally:
        runtime.close()


def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    preflight, prompts, _plans = _read_preflight(
        Path(args.output_root), str(args.expected_preflight_sha256)
    )
    construction, construction_replay, gate, answer, source_plans = _load_sources(
        construction_path=Path(args.construction),
        construction_sha256=str(args.expected_construction_sha256),
        construction_replay_path=Path(args.construction_replay),
        construction_replay_sha256=str(
            args.expected_construction_replay_sha256
        ),
        gate_path=Path(args.gate),
        gate_sha256=str(args.expected_gate_sha256),
        answer_path=Path(args.answer),
        answer_sha256=str(args.expected_answer_sha256),
    )
    _assert_preflight_source_binding(
        preflight,
        construction,
        construction_replay,
        gate,
        answer,
        source_plans,
    )
    required = preflight.payload["required_authorized_provider_calls"]
    _require(
        args.enable_provider is True and args.authorized_provider_calls == required,
        f"provider-run requires exact authorization for {required} Terra calls",
    )
    load_dotenv()
    api_key = os.environ.get(str(args.api_key_env), "").strip()
    _require(bool(api_key), f"provider API key is empty: {args.api_key_env}")
    client = live._make_provider_client(api_key, str(args.gateway_url))  # noqa: SLF001
    try:
        batch = _checkpoint_batch(preflight, prompts, args=args, client=client)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    _require(
        batch.usage.logical_calls == batch.usage.unique_calls == required,
        "residual Terra provider population changed",
    )
    return {
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "gold_loaded": False,
        "physical_provider_calls": batch.usage.physical_calls,
        "preflight_sha256": preflight.sha256,
        "required_authorized_provider_calls": required,
    }


def _lexemes(value: str) -> frozenset[str]:
    return frozenset(
        match.group(0).casefold().replace("’", "'")
        for match in _LEXEME_RE.finditer(value)
    )


def _numeric_anchors(value: str) -> frozenset[str]:
    anchors: set[str] = set()
    for match in _NUMERIC_ANCHOR_RE.finditer(value):
        number = match.group("number").replace(",", "")
        currency = match.group("currency") or ""
        percent = "%" if match.group("percent") else ""
        anchors.add(f"{currency}{number}{percent}")
    return frozenset(anchors)


def _material_anchors(value: str) -> frozenset[str]:
    result: set[str] = set()
    for raw in _LEXEME_RE.findall(value):
        if any(character.isdigit() for character in raw):
            continue
        normalized = normalize_term(raw)
        normalized = _ANCHOR_ALIASES.get(normalized, normalized)
        if normalized and normalized not in _PREDICTION_BOILERPLATE:
            result.add(normalized)
    return frozenset(result)


def parse_residual_completion(
    completion: str,
    *,
    current_prediction: str,
    allowed_evidence: Sequence[Mapping[str, Any]],
    required_residual_handle_ids: Sequence[str],
    answer_plan_receipt_sha256: str,
) -> dict[str, Any]:
    """Parse strict keep/replace JSON and require cited lexical grounding."""

    if type(completion) is not str:
        raise TypeError("residual completion must be exact text")
    require_text(current_prediction, "current residual prediction")
    source_receipt = require_sha256(
        answer_plan_receipt_sha256, "residual parser plan"
    )
    evidence = tuple(_exact_dict(dict(row), "residual parser evidence") for row in allowed_evidence)
    assert_gold_blind(evidence, path="semantic_residual_answer.parser_evidence")
    by_handle = {row.get("evidence_handle"): row for row in evidence}
    residual_handles = tuple(required_residual_handle_ids)
    _require(
        bool(evidence)
        and len(by_handle) == len(evidence)
        and bool(residual_handles)
        and len(residual_handles) == len(set(residual_handles))
        and set(residual_handles) <= set(by_handle)
        and all(
            type(handle) is str
            and _RESIDUAL_HANDLE_RE.fullmatch(handle) is not None
            for handle in residual_handles
        )
        and all(
            type(handle) is str
            and _EVIDENCE_HANDLE_RE.fullmatch(handle) is not None
            and type(row.get("quote")) is str
            and quote_sha256(row["quote"]) == row.get("quote_sha256")
            and row.get("handle_class")
            == ("residual" if handle.startswith("R") else "protected_owner")
            for handle, row in by_handle.items()
        ),
        "residual parser evidence universe changed",
    )

    def parsed(
        *,
        valid: bool,
        decision: str,
        prediction: str,
        used: Sequence[str],
        error_code: str,
        grounding_terms: Sequence[str] = (),
        grounding_numeric_anchors: Sequence[str] = (),
    ) -> dict[str, Any]:
        body = {
            "answer_plan_receipt_sha256": source_receipt,
            "decision": decision,
            "error_code": error_code,
            "evidence_grounded": (
                bool(grounding_terms or grounding_numeric_anchors)
                if decision == "replace"
                else valid
            ),
            "format": PARSE_FORMAT,
            "gold_loaded": False,
            "grounding_terms": list(grounding_terms),
            "grounding_numeric_anchors": list(grounding_numeric_anchors),
            "new_provider_calls": 0,
            "prediction": prediction,
            "prediction_sha256": quote_sha256(prediction),
            "retained_transformer_token_state_bytes": 0,
            "used_evidence_handle_ids": list(used),
            "used_protected_owner_handle_ids": [
                value for value in used if value.startswith("P")
            ],
            "used_residual_handle_ids": [
                value for value in used if value.startswith("R")
            ],
            "valid": valid,
        }
        assert_gold_blind(body, path="semantic_residual_answer.parse")
        return _with_receipt(body)

    try:
        raw = json.loads(
            completion,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (json.JSONDecodeError, ValueError):
        return parsed(
            valid=False,
            decision="invalid",
            prediction="",
            used=(),
            error_code="invalid_json",
        )
    if type(raw) is not dict or set(raw) != {
        "decision",
        "prediction",
        "used_evidence_handle_ids",
    }:
        return parsed(valid=False, decision="invalid", prediction="", used=(), error_code="root_schema")
    decision = raw["decision"]
    prediction = raw["prediction"]
    used_raw = raw["used_evidence_handle_ids"]
    if (
        type(decision) is not str
        or type(prediction) is not str
        or type(used_raw) is not list
        or any(type(value) is not str for value in used_raw)
        or len(used_raw) != len(set(used_raw))
    ):
        return parsed(valid=False, decision="invalid", prediction="", used=(), error_code="value_schema")
    used = tuple(used_raw)
    if decision == "keep_current":
        if prediction != current_prediction or used:
            return parsed(valid=False, decision="invalid", prediction="", used=(), error_code="keep_current_contract")
        return parsed(
            valid=True,
            decision="keep_current",
            prediction=current_prediction,
            used=(),
            error_code="none",
        )
    if decision != "replace":
        return parsed(valid=False, decision="invalid", prediction="", used=(), error_code="decision")
    if not prediction or prediction.strip() != prediction or not used:
        return parsed(valid=False, decision="invalid", prediction="", used=(), error_code="replace_contract")
    if not set(used) <= set(by_handle):
        return parsed(valid=False, decision="invalid", prediction="", used=(), error_code="unknown_handle")
    if not set(used) & set(residual_handles):
        return parsed(
            valid=False,
            decision="invalid",
            prediction="",
            used=(),
            error_code="owner_only_replacement",
        )
    cited_text = "\n".join(by_handle[handle]["quote"] for handle in used)
    prediction_terms = _material_anchors(prediction)
    cited_terms = _material_anchors(cited_text)
    prediction_numbers = _numeric_anchors(prediction)
    cited_numbers = _numeric_anchors(cited_text)
    if (
        not (prediction_terms or prediction_numbers)
        or not prediction_terms <= cited_terms
        or not prediction_numbers <= cited_numbers
    ):
        return parsed(
            valid=False,
            decision="invalid",
            prediction="",
            used=(),
            error_code="unsupported_prediction_anchor",
        )
    return parsed(
        valid=True,
        decision="replace",
        prediction=prediction,
        used=used,
        error_code="none",
        grounding_terms=tuple(sorted(prediction_terms)),
        grounding_numeric_anchors=tuple(sorted(prediction_numbers)),
    )


def _stable_batch(batch: FastCompletionBatch) -> dict[str, Any]:
    value = batch.model_dump()
    return {
        "logical_completions": value["logical_completions"],
        "prompt_population": value["prompt_population"],
        "provenance": value["provenance"],
        "runtime_identity_sha256": value["runtime_identity_sha256"],
        "unique_records": [
            {key: child for key, child in row.items() if key not in {"checkpoint_hit", "physical_call"}}
            for row in value["unique_records"]
        ],
        "usage": {
            key: child
            for key, child in value["usage"].items()
            if key not in {"checkpoint_hits", "physical_calls"}
        },
    }


def _result_row(
    plan: Mapping[str, Any],
    *,
    prediction: str,
    prediction_source: str,
    decision: str,
    record: Any | None = None,
    parsed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    body = {
        "answer_plan_receipt_sha256": plan["answer_plan_receipt_sha256"],
        "call_key_sha256": None if record is None else record.call_key_sha256,
        # Preserve the common judge seam: for this additive layer the frozen
        # V3 prediction is the immediate parent prediction.
        "changed_from_parent": prediction != plan["current_prediction"],
        "changed_from_v3": prediction != plan["current_prediction"],
        "completion_receipt_sha256": None if record is None else record.completion_sha256,
        "construction_question_receipt_sha256": plan[
            "construction_question_receipt_sha256"
        ],
        "dated_question_sha256": plan["dated_question_sha256"],
        "decision": decision,
        "format": RESULT_ROW_FORMAT,
        "gold_loaded": False,
        "ordinal": plan["ordinal"],
        "parent_prediction_sha256": plan["current_prediction_sha256"],
        "parse_error_code": None if parsed is None else parsed["error_code"],
        "parse_receipt_sha256": None if parsed is None else parsed["receipt_sha256"],
        "physical_provider_calls": 0,
        "prediction": prediction,
        "prediction_sha256": quote_sha256(prediction),
        "prediction_source": prediction_source,
        "question_id": plan["question_id"],
        "question_sha256": plan["question_sha256"],
        "request_journal_sha256": None if record is None else record.request_journal_sha256,
        "response_journal_sha256": None if record is None else record.response_journal_sha256,
        "retained_transformer_token_state_bytes": 0,
        "route_id": plan["route_id"],
        "source_v3_answer_row_sha256": plan["source_v3_answer_row_sha256"],
        "used_evidence_handle_ids": [] if parsed is None else parsed["used_evidence_handle_ids"],
        "used_protected_owner_handle_ids": (
            [] if parsed is None else parsed["used_protected_owner_handle_ids"]
        ),
        "used_residual_handle_ids": [] if parsed is None else parsed["used_residual_handle_ids"],
    }
    assert_gold_blind(body, path="semantic_residual_answer.result")
    return _with_receipt(body, "source_row_sha256")


def _materialization_payload(
    preflight: SealedArtifact,
    plans: Sequence[Mapping[str, Any]],
    batch: FastCompletionBatch,
) -> dict[str, Any]:
    required = preflight.payload["required_authorized_provider_calls"]
    _require(
        batch.usage.logical_calls
        == batch.usage.unique_calls
        == batch.usage.checkpoint_hits
        == required
        and batch.usage.physical_calls == 0
        and len(batch.logical_completions) == required
        and len(batch.unique_records) == required,
        "residual materialization requires complete checkpoints and zero calls",
    )
    physical = tuple(row for row in plans if row["mode"] == RESIDUAL_MODE)
    completions = {
        plan["ordinal"]: completion
        for plan, completion in zip(physical, batch.logical_completions, strict=True)
    }
    records = {row.messages_sha256: row for row in batch.unique_records}
    results: list[dict[str, Any]] = []
    for plan in plans:
        if plan["mode"] == PASSTHROUGH_MODE:
            results.append(
                _result_row(
                    plan,
                    prediction=plan["current_prediction"],
                    prediction_source="locked_residual_v3_passthrough_v4",
                    decision="v3_passthrough",
                )
            )
            continue
        completion = completions[plan["ordinal"]]
        record = records.get(plan["messages_sha256"])
        _require(
            record is not None
            and record.completion == completion
            and record.checkpoint_hit is True
            and record.physical_call is False,
            f"residual checkpoint changed at ordinal {plan['ordinal']}",
        )
        parsed = parse_residual_completion(
            completion,
            current_prediction=plan["current_prediction"],
            allowed_evidence=plan["evidence_grounding_rows"],
            required_residual_handle_ids=plan["required_residual_handle_ids"],
            answer_plan_receipt_sha256=plan["answer_plan_receipt_sha256"],
        )
        if parsed["valid"] and parsed["decision"] == "replace":
            prediction = parsed["prediction"]
            source = "locked_residual_grounded_replacement_v4"
            decision = "replace"
        elif parsed["valid"]:
            prediction = plan["current_prediction"]
            source = "locked_residual_validated_keep_current_v4"
            decision = "keep_current"
        else:
            prediction = plan["current_prediction"]
            source = "locked_residual_invalid_keep_current_v4"
            decision = "invalid_keep_current"
        results.append(
            _result_row(
                plan,
                prediction=prediction,
                prediction_source=source,
                decision=decision,
                record=record,
                parsed=parsed,
            )
        )
    _require(
        tuple(row["ordinal"] for row in results) == tuple(range(QUESTION_COUNT)),
        "residual result order changed",
    )
    payload = {
        "answer_artifact_sha256": preflight.payload["answer_artifact_sha256"],
        "changed_from_v3_count": sum(row["changed_from_v3"] for row in results),
        "completion_batch": _stable_batch(batch),
        "construction_artifact_sha256": preflight.payload[
            "construction_artifact_sha256"
        ],
        "format": FORMAT,
        "gold_loaded": False,
        "gate_artifact_sha256": preflight.payload["gate_artifact_sha256"],
        "invalid_completion_v3_fallback_count": sum(
            row["decision"] == "invalid_keep_current" for row in results
        ),
        "judge_rows": [judge_row_projection(row) for row in results],
        "model": DEFAULT_MODEL,
        "passthrough_count": sum(row["decision"] == "v3_passthrough" for row in results),
        "physical_provider_calls_during_materialization": 0,
        "preflight_artifact_sha256": preflight.sha256,
        "question_count": QUESTION_COUNT,
        "questions": results,
        "required_authorized_provider_calls": required,
        "retained_transformer_token_state_bytes": 0,
        "validated_keep_current_count": sum(row["decision"] == "keep_current" for row in results),
        "validated_replacement_count": sum(row["decision"] == "replace" for row in results),
    }
    assert_gold_blind(payload, path="semantic_residual_answer.materialization")
    return payload


def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    preflight, prompts, plans = _read_preflight(
        Path(args.output_root), str(args.expected_preflight_sha256)
    )
    batch = _checkpoint_batch(preflight, prompts, args=args, client=None)
    payload = _materialization_payload(preflight, plans, batch)
    artifact, created = publish_sealed_json(Path(args.output_root) / RUN_NAME, payload)
    return {
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "changed_from_v3_count": payload["changed_from_v3_count"],
        "physical_provider_calls": 0,
        "run_sha256": artifact.sha256,
        "terminal_run_replayed": not created,
        "validated_replacement_count": payload["validated_replacement_count"],
    }


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    construction, construction_replay, gate, answer, source_plans = _load_sources(
        construction_path=Path(args.construction),
        construction_sha256=str(args.expected_construction_sha256),
        construction_replay_path=Path(args.construction_replay),
        construction_replay_sha256=str(
            args.expected_construction_replay_sha256
        ),
        gate_path=Path(args.gate),
        gate_sha256=str(args.expected_gate_sha256),
        answer_path=Path(args.answer),
        answer_sha256=str(args.expected_answer_sha256),
    )
    preflight, prompts, plans = _read_preflight(
        Path(args.output_root), str(args.expected_preflight_sha256)
    )
    _assert_preflight_source_binding(
        preflight,
        construction,
        construction_replay,
        gate,
        answer,
        source_plans,
    )
    _require(plans == source_plans, "residual preflight plan population changed")
    batch = _checkpoint_batch(preflight, prompts, args=args, client=None)
    rebuilt = _materialization_payload(preflight, plans, batch)
    terminal = read_sealed_json(Path(args.output_root) / RUN_NAME)
    _require(
        terminal.sha256 == require_sha256(args.expected_run_sha256, "residual answer run")
        and terminal.payload == rebuilt,
        "residual answer differs from checkpoint-only replay",
    )
    replay, _created = publish_sealed_json(
        Path(args.output_root) / REPLAY_NAME, terminal.payload
    )
    _require(replay.sha256 == terminal.sha256, "residual answer replay changed bytes")
    return {
        "byte_identical": True,
        "physical_provider_calls": 0,
        "replay_sha256": replay.sha256,
        "run_sha256": terminal.sha256,
    }


def _add_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gateway-url", default=live.DEFAULT_GATEWAY_URL)
    parser.add_argument("--max-concurrency", type=int, default=4)


def _add_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--construction", type=Path, default=DEFAULT_CONSTRUCTION)
    parser.add_argument("--expected-construction-sha256", required=True)
    parser.add_argument(
        "--construction-replay", type=Path, default=DEFAULT_CONSTRUCTION_REPLAY
    )
    parser.add_argument("--expected-construction-replay-sha256", required=True)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--expected-gate-sha256", required=True)
    parser.add_argument("--answer", type=Path, default=DEFAULT_ANSWER)
    parser.add_argument(
        "--expected-answer-sha256", default=construction_v4.EXPECTED_ANSWER_SHA256
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    _add_runtime(preflight)
    _add_sources(preflight)
    provider = commands.add_parser("provider-run")
    _add_runtime(provider)
    _add_sources(provider)
    provider.add_argument("--expected-preflight-sha256", required=True)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, default=0)
    provider.add_argument("--api-key-env", default=live.DEFAULT_API_KEY_ENV)
    materialize = commands.add_parser("materialize")
    _add_runtime(materialize)
    materialize.add_argument("--expected-preflight-sha256", required=True)
    replay = commands.add_parser("replay")
    _add_runtime(replay)
    _add_sources(replay)
    replay.add_argument("--expected-preflight-sha256", required=True)
    replay.add_argument("--expected-run-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        result = run_preflight(args)
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
    "DEFAULT_MODEL",
    "DEFAULT_OUTPUT",
    "FORMAT",
    "LockedSemanticResidualAnswerError",
    "PREFLIGHT_NAME",
    "REPLAY_NAME",
    "RUN_NAME",
    "build_parser",
    "build_preflight_payload",
    "main",
    "parse_residual_completion",
    "run_materialize",
    "run_preflight",
    "run_provider",
    "run_replay",
]
