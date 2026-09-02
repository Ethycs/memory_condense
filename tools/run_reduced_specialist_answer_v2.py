#!/usr/bin/env python3
"""Checkpointed Terra answer stage for the sealed reduced specialist v2 arm.

The source artifact already contains the authoritative terminal provider input
for the ten known-miss questions.  This runner never retrieves or opens gold:
it only verifies, renders, seals, executes, materializes, and replays those ten
prompts.

Lifecycle::

    preflight -> provider-run -> materialize -> replay
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    LEGACY_SYSTEM_PROMPT_V1,
    RESOURCE_PRESERVING_SYSTEM_PROMPT_V2,
    VALIDATOR_POLICY_FORMAT,
    parse_typed_final_completion,
    render_final_messages,
)


SOURCE_FORMAT = "memory-condense-reduced-specialist-retrieval-assay-v2-construction"
FORMAT = "memory-condense-reduced-specialist-terra-answer-v2"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight"

PREFLIGHT_NAME = "reduced-specialist-answer-preflight-v2.json"
RUN_NAME = "reduced-specialist-answer-v2.json"
REPLAY_NAME = "reduced-specialist-answer-replay-v2.json"
CHECKPOINT_DIR_NAME = "reduced-specialist-answer-checkpoints-v2"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONSTRUCTION = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/reduced-specialist-missing10-v2/"
    "reduced-specialist-construction-v2.json"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/reduced-specialist-answer-v2"
)
EXPECTED_CONSTRUCTION_SHA256 = (
    "fd179de8fc383cb6c051f704d5f0d25a37c93e3cb086ada794b3200ca89ada05"
)
EXPECTED_ORDINALS = (7, 31, 36, 43, 61, 72, 77, 81, 86, 93)
EXPECTED_PROVIDER_CALLS = len(EXPECTED_ORDINALS)
DEFAULT_MODEL = "codex_sdk/gpt-5.6-terra"

HARD_COMPLETE_CHAT_TOKEN_CAP = 8_000
OUTPUT_TOKEN_RESERVE = 768
MAX_CHAT_PROMPT_TOKENS = HARD_COMPLETE_CHAT_TOKEN_CAP - OUTPUT_TOKEN_RESERVE


class ReducedSpecialistAnswerV2Error(MatchedEvalContractError):
    """Raised when a sealed source, prompt, checkpoint, or replay diverges."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ReducedSpecialistAnswerV2Error(message)


def _plain_messages(
    messages: Sequence[Mapping[str, str]],
) -> tuple[dict[str, str], ...]:
    rows = tuple(dict(row) for row in messages)
    _require(
        bool(rows)
        and all(
            set(row) == {"role", "content"}
            and row["role"] in {"system", "user", "assistant"}
            and type(row["content"]) is str
            for row in rows
        ),
        "terminal prompt message schema changed",
    )
    return rows


def _messages_bound_to_terminal(
    provider_input: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> tuple[tuple[dict[str, str], ...], str, int]:
    """Reconstruct the one supported prompt version sealed by ``terminal``."""

    expected_sha256 = require_sha256(
        terminal.get("messages_sha256"), "specialist terminal messages"
    )
    expected_tokens = terminal.get("prompt_token_proxy")
    _require(
        type(expected_tokens) is int and expected_tokens >= 0,
        "sealed specialist prompt token proxy changed type",
    )
    matches: list[tuple[tuple[dict[str, str], ...], str, int]] = []
    for system_prompt in dict.fromkeys(
        (LEGACY_SYSTEM_PROMPT_V1, RESOURCE_PRESERVING_SYSTEM_PROMPT_V2)
    ):
        messages = _plain_messages(
            render_final_messages(provider_input, system_prompt=system_prompt)
        )
        messages_sha256 = identity_sha256(list(messages))
        if messages_sha256 != expected_sha256:
            continue
        prompt_tokens = count_chat_prompt_token_proxy(messages)
        if prompt_tokens == expected_tokens:
            matches.append((messages, messages_sha256, prompt_tokens))
    _require(
        len(matches) == 1,
        "sealed specialist terminal prompt does not bind one supported renderer",
    )
    return matches[0]


def _handle_groups(
    provider_input: Mapping[str, Any],
    allowed_handle_ids: Sequence[str],
) -> dict[str, str]:
    typed = provider_input.get("typed_evidence")
    _require(type(typed) is dict, "terminal typed evidence is missing")
    assert type(typed) is dict
    handles = typed.get("handles")
    _require(type(handles) is list, "terminal typed handles changed type")
    groups: dict[str, str] = {}
    for raw in handles:
        _require(
            type(raw) is dict
            and set(raw) >= {"handle_id", "group_handle"}
            and type(raw.get("handle_id")) is str
            and bool(raw["handle_id"])
            and type(raw.get("group_handle")) is str
            and bool(raw["group_handle"]),
            "terminal handle/group row changed schema",
        )
        assert type(raw) is dict
        handle = str(raw["handle_id"])
        _require(handle not in groups, "terminal handle/group rows repeat")
        groups[handle] = str(raw["group_handle"])
    _require(
        tuple(allowed_handle_ids)
        and len(tuple(allowed_handle_ids)) == len(set(allowed_handle_ids))
        and set(groups) == set(allowed_handle_ids),
        "fitted allowed handles differ from the terminal handle/group bindings",
    )
    return groups


def _prompt_plan_row(raw: Mapping[str, Any], expected_ordinal: int) -> dict[str, Any]:
    body = dict(raw)
    declared_question_receipt = body.pop("question_receipt_sha256", None)
    _require(
        raw.get("ordinal") == expected_ordinal
        and declared_question_receipt == identity_sha256(body),
        f"specialist construction row seal/order changed at ordinal {expected_ordinal}",
    )
    terminal = raw.get("terminal_prompt")
    fitted = raw.get("fitted_typed_prompt")
    _require(
        type(terminal) is dict and type(fitted) is dict,
        "specialist terminal/fitted prompt is missing",
    )
    assert type(terminal) is dict and type(fitted) is dict
    provider_input = terminal.get("provider_input")
    fitted_input = fitted.get("provider_input")
    _require(
        type(provider_input) is dict and type(fitted_input) is dict,
        "specialist provider input is missing",
    )
    assert type(provider_input) is dict and type(fitted_input) is dict
    assert_gold_blind(provider_input, path=f"reduced_specialist_prompt_{expected_ordinal}")

    advisories = provider_input.get("specialist_advisories")
    _require(
        type(advisories) is list
        and provider_input
        == {**dict(fitted_input), "specialist_advisories": advisories},
        "terminal provider input is not the fitted prompt plus sealed advisories",
    )
    messages, messages_sha, prompt_tokens = _messages_bound_to_terminal(
        provider_input,
        terminal,
    )
    fitted_receipt = require_sha256(
        fitted.get("receipt_sha256"), "fitted specialist prompt"
    )
    terminal_receipt_body = {
        "fitted_prompt_receipt_sha256": fitted_receipt,
        "messages_sha256": messages_sha,
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "prompt_token_proxy": prompt_tokens,
        "provider_input_sha256": identity_sha256(provider_input),
        "specialist_advisories_sha256": identity_sha256(advisories),
    }
    _require(
        terminal.get("fitted_prompt_receipt_sha256") == fitted_receipt
        and terminal.get("messages_sha256") == messages_sha
        and terminal.get("prompt_token_proxy") == prompt_tokens
        and terminal.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
        and terminal.get("full_chat_plus_output_tokens")
        == prompt_tokens + OUTPUT_TOKEN_RESERVE
        and terminal.get("hard_prompt_token_cap") == HARD_COMPLETE_CHAT_TOKEN_CAP
        and terminal.get("provider_prompt_count") == 0
        and terminal.get("retained_transformer_token_state_bytes") == 0
        and terminal.get("specialist_advisories_sha256")
        == identity_sha256(advisories)
        and terminal.get("terminal_prompt_receipt_sha256")
        == identity_sha256(terminal_receipt_body)
        and prompt_tokens <= MAX_CHAT_PROMPT_TOKENS,
        "authoritative specialist terminal prompt seal or 7,232-token cap changed",
    )

    parent = provider_input.get("protected_parent_fallback")
    allowed = fitted.get("allowed_handle_ids")
    _require(
        type(parent) is dict
        and type(parent.get("prediction")) is str
        and bool(parent["prediction"])
        and parent.get("prediction_sha256") == quote_sha256(parent["prediction"])
        and type(allowed) is list,
        "specialist parent fallback or fitted handles changed",
    )
    assert type(parent) is dict and type(allowed) is list
    groups = _handle_groups(provider_input, tuple(allowed))
    story = fitted.get("story_coherence")
    preservation = fitted.get("preservation_requirements")
    validation = fitted.get("validation_contract")
    _require(
        type(story) is dict
        and type(preservation) is dict
        and type(validation) is dict
        and provider_input.get("story_coherence") == story,
        "fitted parser contracts changed",
    )
    plan = {
        "allowed_handle_ids": list(allowed),
        "construction_question_receipt_sha256": require_sha256(
            declared_question_receipt, "specialist construction question"
        ),
        "dated_question_sha256": require_sha256(
            raw.get("dated_question_sha256"), "specialist dated question"
        ),
        "handle_group_by_id": groups,
        "messages": list(messages),
        "messages_sha256": messages_sha,
        "ordinal": expected_ordinal,
        "parent_prediction": parent["prediction"],
        "parent_prediction_sha256": parent["prediction_sha256"],
        "preservation_requirements": dict(preservation),
        "prompt_token_proxy": prompt_tokens,
        "question_id": require_text(raw.get("question_id"), "specialist question ID"),
        "question_sha256": require_sha256(
            raw.get("question_sha256"), "specialist question"
        ),
        "story_coherence": dict(story),
        "terminal_prompt_receipt_sha256": require_sha256(
            terminal.get("terminal_prompt_receipt_sha256"),
            "specialist terminal prompt",
        ),
        "validation_contract": dict(validation),
    }
    plan["prompt_row_receipt_sha256"] = identity_sha256(plan)
    assert_gold_blind(plan, path=f"reduced_specialist_plan_{expected_ordinal}")
    return plan


def _read_construction(
    path: Path,
    expected_sha256: str,
) -> tuple[SealedArtifact, tuple[dict[str, Any], ...]]:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256
        == require_sha256(expected_sha256, "expected specialist construction"),
        "specialist v2 construction digest changed",
    )
    payload = artifact.payload
    rows = payload.get("questions")
    _require(
        payload.get("format") == SOURCE_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("new_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("target_labels_loaded") is False
        and payload.get("target_plan_loaded") is False
        and tuple(payload.get("ordinals", ())) == EXPECTED_ORDINALS
        and payload.get("question_count") == EXPECTED_PROVIDER_CALLS
        and type(rows) is list
        and len(rows) == EXPECTED_PROVIDER_CALLS,
        "specialist v2 construction firewall or population changed",
    )
    plans = tuple(
        _prompt_plan_row(row, ordinal)
        for row, ordinal in zip(rows, EXPECTED_ORDINALS, strict=True)
        if type(row) is dict
    )
    _require(
        len(plans) == EXPECTED_PROVIDER_CALLS
        and len({row["question_id"] for row in plans}) == EXPECTED_PROVIDER_CALLS
        and len({row["messages_sha256"] for row in plans})
        == EXPECTED_PROVIDER_CALLS,
        "specialist v2 requires ten distinct questions and physical prompts",
    )
    return artifact, plans


def _preflight_projection(
    construction: SealedArtifact,
    rows: tuple[dict[str, Any], ...],
    *,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> dict[str, Any]:
    _require(model == DEFAULT_MODEL, "reduced specialist v2 model must be Terra")
    require_text(gateway_url, "specialist Terra gateway")
    _require(
        type(max_concurrency) is int and max_concurrency > 0,
        "specialist Terra concurrency must be positive",
    )
    prompts = tuple(_plain_messages(row["messages"]) for row in rows)
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS
    )
    _require(
        population.logical_prompt_count
        == population.unique_prompt_count
        == EXPECTED_PROVIDER_CALLS,
        "specialist Terra answer population must contain ten unique prompts",
    )
    payload = {
        "construction_artifact_sha256": construction.sha256,
        "format": PREFLIGHT_FORMAT,
        "gateway_url": gateway_url,
        "gold_loaded": False,
        "hard_complete_chat_token_cap": HARD_COMPLETE_CHAT_TOKEN_CAP,
        "max_chat_prompt_tokens": MAX_CHAT_PROMPT_TOKENS,
        "max_concurrency": max_concurrency,
        "model": model,
        "observed_max_complete_envelope_tokens": max(
            row["prompt_token_proxy"] + OUTPUT_TOKEN_RESERVE for row in rows
        ),
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "physical_prompt_rows": list(rows),
        "prompt_population": population.model_dump(),
        "prompt_population_sha256": population.prompt_population_sha256,
        "provider_calls": 0,
        "question_count": EXPECTED_PROVIDER_CALLS,
        "required_authorized_provider_calls": EXPECTED_PROVIDER_CALLS,
        "retained_transformer_token_state_bytes": 0,
    }
    assert_gold_blind(payload, path="reduced_specialist_answer_preflight_v2")
    return payload


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    construction, rows = _read_construction(
        Path(args.construction), str(args.expected_construction_sha256)
    )
    payload = _preflight_projection(
        construction,
        rows,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    artifact, created = publish_sealed_json(
        Path(args.output_root) / PREFLIGHT_NAME, payload
    )
    return {
        "construction_sha256": construction.sha256,
        "created": created,
        "maximum_complete_prompt_envelope": payload[
            "observed_max_complete_envelope_tokens"
        ],
        "physical_provider_calls": 0,
        "preflight_sha256": artifact.sha256,
        "question_count": EXPECTED_PROVIDER_CALLS,
        "required_authorized_provider_calls": EXPECTED_PROVIDER_CALLS,
    }


def _validate_preflight(
    artifact: SealedArtifact,
) -> tuple[tuple[tuple[dict[str, str], ...], ...], tuple[dict[str, Any], ...]]:
    payload = artifact.payload
    rows = payload.get("physical_prompt_rows")
    _require(
        payload.get("format") == PREFLIGHT_FORMAT
        and payload.get("construction_artifact_sha256")
        == EXPECTED_CONSTRUCTION_SHA256
        and payload.get("model") == DEFAULT_MODEL
        and payload.get("gold_loaded") is False
        and payload.get("provider_calls") == 0
        and payload.get("max_chat_prompt_tokens") == MAX_CHAT_PROMPT_TOKENS
        and payload.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
        and payload.get("question_count") == EXPECTED_PROVIDER_CALLS
        and payload.get("required_authorized_provider_calls")
        == EXPECTED_PROVIDER_CALLS
        and payload.get("retained_transformer_token_state_bytes") == 0
        and type(rows) is list
        and len(rows) == EXPECTED_PROVIDER_CALLS,
        "sealed specialist Terra preflight changed",
    )
    prompts: list[tuple[dict[str, str], ...]] = []
    validated: list[dict[str, Any]] = []
    for ordinal, raw in zip(EXPECTED_ORDINALS, rows, strict=True):
        _require(type(raw) is dict, "specialist preflight row changed type")
        assert type(raw) is dict
        body = dict(raw)
        declared = body.pop("prompt_row_receipt_sha256", None)
        messages = _plain_messages(raw.get("messages", ()))
        _require(
            raw.get("ordinal") == ordinal
            and declared == identity_sha256(body)
            and identity_sha256(list(messages)) == raw.get("messages_sha256")
            and count_chat_prompt_token_proxy(messages)
            == raw.get("prompt_token_proxy")
            and int(raw["prompt_token_proxy"]) <= MAX_CHAT_PROMPT_TOKENS,
            f"specialist preflight prompt changed at ordinal {ordinal}",
        )
        prompts.append(messages)
        validated.append(dict(raw))
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS
    )
    _require(
        population.model_dump() == payload.get("prompt_population")
        and population.prompt_population_sha256
        == payload.get("prompt_population_sha256")
        and population.unique_prompt_count == EXPECTED_PROVIDER_CALLS,
        "sealed specialist Terra prompt population changed",
    )
    return tuple(prompts), tuple(validated)


def _read_preflight(
    output_root: Path, expected_sha256: str
) -> tuple[
    SealedArtifact,
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
]:
    artifact = read_sealed_json(output_root / PREFLIGHT_NAME)
    _require(
        artifact.sha256
        == require_sha256(expected_sha256, "expected specialist answer preflight"),
        "specialist Terra preflight digest changed",
    )
    prompts, rows = _validate_preflight(artifact)
    return artifact, prompts, rows


def _runtime(
    artifact: SealedArtifact,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    output_root: Path,
    model: str,
    gateway_url: str,
    max_concurrency: int,
    client: Any | None,
) -> FastCompletionRuntime:
    _require(
        model == DEFAULT_MODEL == artifact.payload.get("model")
        and gateway_url == artifact.payload.get("gateway_url")
        and max_concurrency == artifact.payload.get("max_concurrency"),
        "runtime settings differ from the sealed specialist Terra preflight",
    )
    return FastCompletionRuntime(
        checkpoint_dir=output_root / CHECKPOINT_DIR_NAME,
        prompt_population=prompts,
        model=model,
        client=client,
        max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS,
        max_new_tokens=OUTPUT_TOKEN_RESERVE,
        max_concurrency=max_concurrency,
        retries=0,
        benchmark_provenance={
            "arm": "reduced_specialist_terra_answer_v2",
            "authorized_unique_calls": EXPECTED_PROVIDER_CALLS,
            "construction_artifact_sha256": EXPECTED_CONSTRUCTION_SHA256,
            "experiment_format": FORMAT,
            "gateway_url": gateway_url,
            "gold_loaded": False,
            "preflight_artifact_sha256": artifact.sha256,
        },
    )


def _checkpoint_batch(
    artifact: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    args: argparse.Namespace,
    client: Any | None,
) -> FastCompletionBatch:
    runtime = _runtime(
        artifact,
        prompts,
        output_root=Path(args.output_root),
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
        client=client,
    )
    try:
        return runtime.run()
    finally:
        runtime.close()


def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    preflight, prompts, _rows = _read_preflight(
        Path(args.output_root), str(args.expected_preflight_sha256)
    )
    _require(
        args.enable_provider is True
        and args.authorized_provider_calls == EXPECTED_PROVIDER_CALLS,
        "provider-run requires exact authorization for 10 Terra calls",
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
        batch.usage.logical_calls
        == batch.usage.unique_calls
        == EXPECTED_PROVIDER_CALLS,
        "specialist Terra physical population changed",
    )
    return {
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "gold_loaded": False,
        "physical_provider_calls": batch.usage.physical_calls,
        "preflight_sha256": preflight.sha256,
        "required_authorized_provider_calls": EXPECTED_PROVIDER_CALLS,
    }


def _stable_batch(batch: FastCompletionBatch) -> dict[str, Any]:
    value = batch.model_dump()
    return {
        "logical_completions": value["logical_completions"],
        "prompt_population": value["prompt_population"],
        "provenance": value["provenance"],
        "runtime_identity_sha256": value["runtime_identity_sha256"],
        "unique_records": [
            {
                key: child
                for key, child in row.items()
                if key not in {"checkpoint_hit", "physical_call"}
            }
            for row in value["unique_records"]
        ],
        "usage": {
            key: child
            for key, child in value["usage"].items()
            if key not in {"checkpoint_hits", "physical_calls"}
        },
    }


def _materialization_projection(
    preflight: SealedArtifact,
    prompt_rows: tuple[dict[str, Any], ...],
    batch: FastCompletionBatch,
) -> dict[str, Any]:
    _require(
        batch.usage.logical_calls
        == batch.usage.unique_calls
        == batch.usage.checkpoint_hits
        == EXPECTED_PROVIDER_CALLS
        and batch.usage.physical_calls == 0
        and len(batch.logical_completions) == EXPECTED_PROVIDER_CALLS
        and len(batch.unique_records) == EXPECTED_PROVIDER_CALLS,
        "materialization requires ten checkpoint-only Terra completions",
    )
    records = {row.messages_sha256: row for row in batch.unique_records}
    _require(len(records) == EXPECTED_PROVIDER_CALLS, "completion identities repeat")
    results: list[dict[str, Any]] = []
    judge_rows: list[dict[str, Any]] = []
    for plan, completion in zip(
        prompt_rows, batch.logical_completions, strict=True
    ):
        record = records.get(plan["messages_sha256"])
        _require(
            record is not None
            and record.completion == completion
            and record.checkpoint_hit is True
            and record.physical_call is False,
            "specialist Terra checkpoint record changed",
        )
        assert record is not None
        parent = plan["parent_prediction"]
        parsed = parse_typed_final_completion(
            completion,
            parent_prediction=parent,
            allowed_handle_ids=tuple(plan["allowed_handle_ids"]),
            handle_group_by_id=dict(plan["handle_group_by_id"]),
            story_coherence=dict(plan["story_coherence"]),
            preservation_requirements=dict(plan["preservation_requirements"]),
            validation_contract=dict(plan["validation_contract"]),
        )
        valid_replace = parsed.valid and parsed.decision == "replace"
        prediction = parsed.prediction if valid_replace else parent
        if valid_replace:
            source = (
                "typed_final_deterministic_validated_replacement_v1"
                if parsed.validation_basis == "deterministic_execution_agreement"
                else "typed_final_scalar_validated_replacement_v1"
                if parsed.validation_basis == "bounded_positive_scalar_agreement"
                else "typed_final_model_attested_replacement_v1"
            )
            decision = "replace"
            used_handles = list(parsed.used_handle_ids)
        elif parsed.valid:
            source = "typed_final_validated_keep_parent_v1"
            decision = "keep_parent"
            used_handles = []
        else:
            source = "typed_final_invalid_keep_parent_v1"
            decision = "invalid_keep_parent"
            used_handles = []
        body = {
            "call_key_sha256": record.call_key_sha256,
            "changed_from_parent": prediction != parent,
            "completion_receipt_sha256": record.completion_sha256,
            "dated_question_sha256": plan["dated_question_sha256"],
            "decision": decision,
            "ordinal": plan["ordinal"],
            "parent_prediction_sha256": plan["parent_prediction_sha256"],
            "parse_error_code": parsed.error_code,
            "parse_receipt_sha256": parsed.receipt_sha256,
            "prediction": prediction,
            "prediction_sha256": quote_sha256(prediction),
            "prediction_source": source,
            "prompt_row_receipt_sha256": plan["prompt_row_receipt_sha256"],
            "question_id": plan["question_id"],
            "question_sha256": plan["question_sha256"],
            "request_journal_sha256": record.request_journal_sha256,
            "response_journal_sha256": record.response_journal_sha256,
            "solver_valid": parsed.valid,
            "used_handle_ids": used_handles,
            "validation_basis": parsed.validation_basis,
            "validator_policy_format": VALIDATOR_POLICY_FORMAT,
        }
        row = {**body, "source_row_sha256": identity_sha256(body)}
        results.append(row)
        seam = {
            "dated_question_sha256": row["dated_question_sha256"],
            "ordinal": row["ordinal"],
            "prediction": row["prediction"],
            "prediction_sha256": row["prediction_sha256"],
            "question_id": row["question_id"],
            "question_sha256": row["question_sha256"],
        }
        judge_rows.append({**seam, "answer_row_sha256": identity_sha256(seam)})
    payload = {
        "changed_prediction_count": sum(row["changed_from_parent"] for row in results),
        "completion_batch": _stable_batch(batch),
        "construction_artifact_sha256": EXPECTED_CONSTRUCTION_SHA256,
        "format": FORMAT,
        "gold_loaded": False,
        "invalid_completion_parent_fallback_count": sum(
            row["prediction_source"] == "typed_final_invalid_keep_parent_v1"
            for row in results
        ),
        "judge_rows": judge_rows,
        "model": DEFAULT_MODEL,
        "physical_provider_calls_during_materialization": 0,
        "preflight_artifact_sha256": preflight.sha256,
        "question_count": EXPECTED_PROVIDER_CALLS,
        "questions": results,
        "required_authorized_provider_calls": EXPECTED_PROVIDER_CALLS,
        "retained_transformer_token_state_bytes": 0,
    }
    assert_gold_blind(payload, path="reduced_specialist_terra_answer_v2")
    return payload


def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    preflight, prompts, rows = _read_preflight(
        Path(args.output_root), str(args.expected_preflight_sha256)
    )
    batch = _checkpoint_batch(preflight, prompts, args=args, client=None)
    payload = _materialization_projection(preflight, rows, batch)
    artifact, created = publish_sealed_json(
        Path(args.output_root) / RUN_NAME, payload
    )
    return {
        "changed_prediction_count": payload["changed_prediction_count"],
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "physical_provider_calls": 0,
        "run_sha256": artifact.sha256,
        "terminal_run_replayed": not created,
    }


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    construction, _plans = _read_construction(
        Path(args.construction), str(args.expected_construction_sha256)
    )
    preflight, prompts, rows = _read_preflight(
        Path(args.output_root), str(args.expected_preflight_sha256)
    )
    _require(
        preflight.payload.get("construction_artifact_sha256")
        == construction.sha256,
        "specialist answer construction/preflight binding changed",
    )
    batch = _checkpoint_batch(preflight, prompts, args=args, client=None)
    rebuilt = _materialization_projection(preflight, rows, batch)
    terminal = read_sealed_json(Path(args.output_root) / RUN_NAME)
    _require(
        terminal.sha256
        == require_sha256(args.expected_run_sha256, "expected specialist answer run")
        and terminal.payload == rebuilt,
        "specialist answer run differs from checkpoint-only replay",
    )
    replay, _created = publish_sealed_json(
        Path(args.output_root) / REPLAY_NAME, terminal.payload
    )
    _require(
        replay.sha256 == terminal.sha256,
        "specialist answer replay is not byte-identical",
    )
    return {
        "byte_identical": True,
        "physical_provider_calls": 0,
        "replay_sha256": replay.sha256,
        "run_sha256": terminal.sha256,
    }


def _add_runtime_settings(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gateway-url", default=live.DEFAULT_GATEWAY_URL)
    parser.add_argument("--max-concurrency", type=int, default=4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight", help="seal ten terminal prompts")
    _add_runtime_settings(preflight)
    preflight.add_argument("--construction", type=Path, default=DEFAULT_CONSTRUCTION)
    preflight.add_argument(
        "--expected-construction-sha256", default=EXPECTED_CONSTRUCTION_SHA256
    )

    provider = commands.add_parser("provider-run", help="execute sealed Terra prompts")
    _add_runtime_settings(provider)
    provider.add_argument("--expected-preflight-sha256", required=True)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, default=0)
    provider.add_argument("--api-key-env", default=live.DEFAULT_API_KEY_ENV)

    materialize = commands.add_parser(
        "materialize", help="materialize checkpoint-only predictions"
    )
    _add_runtime_settings(materialize)
    materialize.add_argument("--expected-preflight-sha256", required=True)

    replay = commands.add_parser("replay", help="prove byte-identical replay")
    _add_runtime_settings(replay)
    replay.add_argument("--construction", type=Path, default=DEFAULT_CONSTRUCTION)
    replay.add_argument(
        "--expected-construction-sha256", default=EXPECTED_CONSTRUCTION_SHA256
    )
    replay.add_argument("--expected-preflight-sha256", required=True)
    replay.add_argument("--expected-run-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        result = run_preflight(args)
    elif args.command == "provider-run":
        result = run_provider(args)
    elif args.command == "materialize":
        result = run_materialize(args)
    elif args.command == "replay":
        result = run_replay(args)
    else:  # pragma: no cover
        raise AssertionError("unreachable command")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_DIR_NAME",
    "DEFAULT_CONSTRUCTION",
    "DEFAULT_OUTPUT",
    "EXPECTED_CONSTRUCTION_SHA256",
    "EXPECTED_ORDINALS",
    "FORMAT",
    "PREFLIGHT_NAME",
    "REPLAY_NAME",
    "RUN_NAME",
    "ReducedSpecialistAnswerV2Error",
    "build_parser",
    "main",
    "run_materialize",
    "run_preflight",
    "run_provider",
    "run_replay",
]
