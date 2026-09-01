#!/usr/bin/env python3
"""Checkpointed Terra answers for a caller-sealed specialist v3 construction.

The v3 construction owns retrieval and the authoritative provider inputs.  At
preflight this runner verifies its caller-supplied digest, rerenders every
specialist-scoped prompt, compiles each advisory-local validation scope, and
seals ten unique physical prompts.  Provider execution is the only mutating
network phase; materialization and replay consume verified checkpoints only.
Gold is never opened by this module.
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
from tools import run_reduced_specialist_retrieval_assay as retrieval_base  # noqa: E402
from tools import run_reduced_specialist_retrieval_assay_v3 as retrieval_v3  # noqa: E402
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
from tools.matched_eval.specialist_scoped_completion import (  # noqa: E402
    FORMAT as SCOPED_COMPLETION_FORMAT,
    HARD_COMPLETE_CHAT_TOKEN_CAP,
    MAX_CHAT_PROMPT_TOKENS,
    OUTPUT_TOKEN_RESERVE,
    PROMPT_FORMAT,
    SpecialistPromptEnvelope,
    SpecialistValidationScope,
    compile_specialist_validation_scope,
    parse_specialist_scoped_completion,
    render_specialist_scoped_prompt,
)


FORMAT = "memory-condense-reduced-specialist-terra-answer-v3"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight"

PREFLIGHT_NAME = "reduced-specialist-answer-preflight-v3.json"
RUN_NAME = "reduced-specialist-answer-v3.json"
REPLAY_NAME = "reduced-specialist-answer-replay-v3.json"
CHECKPOINT_DIR_NAME = "reduced-specialist-answer-checkpoints-v3"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONSTRUCTION = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/reduced-specialist-missing10-v3/"
    "reduced-specialist-construction-v3.json"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/reduced-specialist-answer-v3"
)
EXPECTED_ORDINALS = tuple(retrieval_base.TARGET_ORDINALS)
EXPECTED_PROVIDER_CALLS = len(EXPECTED_ORDINALS)
DEFAULT_MODEL = "codex_sdk/gpt-5.6-terra"

_PREFLIGHT_FIELDS = frozenset(
    {
        "construction_artifact_sha256",
        "format",
        "gateway_url",
        "gold_loaded",
        "hard_complete_chat_token_cap",
        "max_chat_prompt_tokens",
        "max_concurrency",
        "model",
        "observed_max_complete_envelope_tokens",
        "output_token_reserve",
        "physical_prompt_rows",
        "prompt_population",
        "prompt_population_sha256",
        "provider_calls",
        "question_count",
        "required_authorized_provider_calls",
        "retained_transformer_token_state_bytes",
        "scoped_completion_format",
    }
)


class ReducedSpecialistAnswerV3Error(MatchedEvalContractError):
    """Raised when a source, scoped prompt, checkpoint, or replay diverges."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ReducedSpecialistAnswerV3Error(message)


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
        "specialist v3 messages changed schema",
    )
    return rows


def _handle_groups(
    provider_input: Mapping[str, Any],
    allowed_handle_ids: Sequence[str],
) -> dict[str, str]:
    typed = provider_input.get("typed_evidence")
    _require(type(typed) is dict, "specialist v3 typed evidence is missing")
    assert type(typed) is dict
    handles = typed.get("handles")
    _require(type(handles) is list, "specialist v3 typed handles changed type")
    groups: dict[str, str] = {}
    for raw in handles:
        _require(
            type(raw) is dict
            and type(raw.get("handle_id")) is str
            and bool(raw["handle_id"])
            and type(raw.get("group_handle")) is str
            and bool(raw["group_handle"]),
            "specialist v3 handle/group row changed",
        )
        assert type(raw) is dict
        handle_id = str(raw["handle_id"])
        _require(handle_id not in groups, "specialist v3 handles repeat")
        groups[handle_id] = str(raw["group_handle"])
    allowed = tuple(allowed_handle_ids)
    _require(
        bool(allowed)
        and len(allowed) == len(set(allowed))
        and set(groups) == set(allowed),
        "specialist v3 allowed handles differ from terminal groups",
    )
    return groups


def _compile_scope(
    *,
    provider_input: Mapping[str, Any],
    declared_advisories_sha256: str,
    sealed_source_receipt_sha256: str,
    allowed_handle_ids: Sequence[str],
    handle_group_by_id: Mapping[str, str],
    validation_contract: Mapping[str, Any],
    prompt: SpecialistPromptEnvelope,
) -> SpecialistValidationScope:
    advisories = provider_input.get("specialist_advisories")
    _require(type(advisories) is list, "specialist v3 advisories changed type")
    assert type(advisories) is list
    return compile_specialist_validation_scope(
        specialist_advisories=advisories,
        declared_specialist_advisories_sha256=require_sha256(
            declared_advisories_sha256, "declared specialist v3 advisories"
        ),
        sealed_source_receipt_sha256=require_sha256(
            sealed_source_receipt_sha256, "sealed specialist v3 source"
        ),
        terminal_allowed_handle_ids=tuple(allowed_handle_ids),
        handle_group_by_id=dict(handle_group_by_id),
        validation_contract=dict(validation_contract),
        prompt_envelope=prompt,
    )


def _prompt_plan_row(raw: Mapping[str, Any], expected_ordinal: int) -> dict[str, Any]:
    source_body = dict(raw)
    declared_question_receipt = source_body.pop("question_receipt_sha256", None)
    _require(
        raw.get("ordinal") == expected_ordinal
        and declared_question_receipt == identity_sha256(source_body),
        f"specialist v3 question seal/order changed at {expected_ordinal}",
    )
    terminal = raw.get("terminal_prompt")
    fitted = raw.get("fitted_typed_prompt")
    _require(
        type(terminal) is dict and type(fitted) is dict,
        "specialist v3 terminal/fitted prompt is missing",
    )
    assert type(terminal) is dict and type(fitted) is dict
    provider_input = terminal.get("provider_input")
    fitted_input = fitted.get("provider_input")
    allowed = fitted.get("allowed_handle_ids")
    validation = fitted.get("validation_contract")
    _require(
        type(provider_input) is dict
        and type(fitted_input) is dict
        and type(allowed) is list
        and type(validation) is dict,
        "specialist v3 provider/parser inputs changed type",
    )
    assert (
        type(provider_input) is dict
        and type(fitted_input) is dict
        and type(allowed) is list
        and type(validation) is dict
    )
    advisories = provider_input.get("specialist_advisories")
    _require(
        type(advisories) is list
        and bool(advisories)
        and provider_input
        == {**dict(fitted_input), "specialist_advisories": advisories},
        "specialist v3 terminal input is not fitted evidence plus advisories",
    )
    assert_gold_blind(provider_input, path=f"specialist_v3_provider_{expected_ordinal}")

    prompt = render_specialist_scoped_prompt(provider_input)
    messages = _plain_messages(prompt.messages)
    messages_sha256 = identity_sha256(list(messages))
    fitted_receipt = require_sha256(
        fitted.get("receipt_sha256"), "fitted specialist v3 prompt"
    )
    terminal_receipt_body = {
        "fitted_prompt_receipt_sha256": fitted_receipt,
        "message_renderer_format": PROMPT_FORMAT,
        "messages_sha256": messages_sha256,
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "prompt_token_proxy": prompt.prompt_token_proxy,
        "provider_input_sha256": identity_sha256(provider_input),
        "specialist_advisories_sha256": prompt.specialist_advisories_sha256,
        "specialist_prompt_envelope_receipt_sha256": prompt.receipt_sha256,
    }
    terminal_source_receipt = require_sha256(
        terminal.get("terminal_prompt_receipt_sha256"),
        "specialist v3 terminal source",
    )
    _require(
        terminal.get("fitted_prompt_receipt_sha256") == fitted_receipt
        and terminal.get("message_renderer_format") == PROMPT_FORMAT
        and terminal.get("specialist_prompt_envelope_receipt_sha256")
        == prompt.receipt_sha256
        and terminal.get("messages_sha256") == messages_sha256
        and terminal.get("prompt_token_proxy") == prompt.prompt_token_proxy
        and count_chat_prompt_token_proxy(messages) == prompt.prompt_token_proxy
        and terminal.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
        and terminal.get("hard_prompt_token_cap") == HARD_COMPLETE_CHAT_TOKEN_CAP
        and terminal.get("full_chat_plus_output_tokens")
        == prompt.prompt_token_proxy + OUTPUT_TOKEN_RESERVE
        and terminal.get("specialist_advisories_sha256")
        == prompt.specialist_advisories_sha256
        and terminal_source_receipt == identity_sha256(terminal_receipt_body)
        and prompt.prompt_token_proxy <= MAX_CHAT_PROMPT_TOKENS
        and terminal.get("provider_prompt_count") == 0
        and terminal.get("retained_transformer_token_state_bytes") == 0,
        "specialist v3 scoped prompt seal or hard budget changed",
    )

    parent = provider_input.get("protected_parent_fallback")
    _require(
        type(parent) is dict
        and type(parent.get("prediction")) is str
        and bool(parent["prediction"])
        and parent.get("prediction_sha256") == quote_sha256(parent["prediction"]),
        "specialist v3 protected parent changed",
    )
    assert type(parent) is dict
    groups = _handle_groups(provider_input, tuple(allowed))
    scope = _compile_scope(
        provider_input=provider_input,
        declared_advisories_sha256=prompt.specialist_advisories_sha256,
        sealed_source_receipt_sha256=terminal_source_receipt,
        allowed_handle_ids=tuple(allowed),
        handle_group_by_id=groups,
        validation_contract=validation,
        prompt=prompt,
    )
    dated_question = require_text(
        provider_input.get("dated_question"), "specialist v3 dated question"
    )
    plan = {
        "allowed_handle_ids": list(allowed),
        "construction_question_receipt_sha256": require_sha256(
            declared_question_receipt, "specialist v3 construction question"
        ),
        "dated_question_sha256": require_sha256(
            raw.get("dated_question_sha256"), "specialist v3 dated question"
        ),
        "handle_group_by_id": groups,
        "messages": list(messages),
        "messages_sha256": messages_sha256,
        "ordinal": expected_ordinal,
        "parent_prediction": parent["prediction"],
        "parent_prediction_sha256": parent["prediction_sha256"],
        "prompt_token_proxy": prompt.prompt_token_proxy,
        "provider_input": dict(provider_input),
        "question_id": require_text(raw.get("question_id"), "specialist v3 question ID"),
        "question_sha256": require_sha256(
            raw.get("question_sha256"), "specialist v3 question"
        ),
        "specialist_advisories_sha256": prompt.specialist_advisories_sha256,
        "specialist_prompt_envelope_receipt_sha256": prompt.receipt_sha256,
        "specialist_scope_receipt_sha256": scope.receipt_sha256,
        "specialist_scope_projection_sha256": identity_sha256(scope.projection()),
        "terminal_prompt_receipt_sha256": terminal_source_receipt,
        "validation_contract": dict(validation),
    }
    _require(
        plan["dated_question_sha256"] == quote_sha256(dated_question),
        "specialist v3 dated-question binding changed",
    )
    plan["prompt_row_receipt_sha256"] = identity_sha256(plan)
    assert_gold_blind(plan, path=f"specialist_v3_plan_{expected_ordinal}")
    return plan


def _read_construction(
    path: Path,
    expected_sha256: str,
) -> tuple[SealedArtifact, tuple[dict[str, Any], ...]]:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256
        == require_sha256(expected_sha256, "expected specialist v3 construction"),
        "specialist v3 construction digest changed",
    )
    rows = retrieval_base._validate_construction(  # noqa: SLF001
        artifact,
        construction_format=retrieval_v3.CONSTRUCTION_FORMAT,
    )
    plans = tuple(
        _prompt_plan_row(row, ordinal)
        for row, ordinal in zip(rows, EXPECTED_ORDINALS, strict=True)
    )
    _require(
        len(plans) == EXPECTED_PROVIDER_CALLS
        and len({row["question_id"] for row in plans}) == EXPECTED_PROVIDER_CALLS
        and len({row["messages_sha256"] for row in plans})
        == EXPECTED_PROVIDER_CALLS,
        "specialist v3 requires ten distinct questions and prompts",
    )
    return artifact, plans


def _scope_from_plan(
    raw: Mapping[str, Any],
) -> tuple[SpecialistPromptEnvelope, SpecialistValidationScope]:
    provider_input = raw.get("provider_input")
    _require(type(provider_input) is dict, "specialist v3 plan lost provider input")
    assert type(provider_input) is dict
    prompt = render_specialist_scoped_prompt(provider_input)
    groups = _handle_groups(provider_input, tuple(raw.get("allowed_handle_ids", ())))
    _require(
        groups == raw.get("handle_group_by_id")
        and prompt.receipt_sha256
        == raw.get("specialist_prompt_envelope_receipt_sha256")
        and prompt.specialist_advisories_sha256
        == raw.get("specialist_advisories_sha256"),
        "specialist v3 prompt plan changed its scoped renderer bindings",
    )
    scope = _compile_scope(
        provider_input=provider_input,
        declared_advisories_sha256=str(raw["specialist_advisories_sha256"]),
        sealed_source_receipt_sha256=str(raw["terminal_prompt_receipt_sha256"]),
        allowed_handle_ids=tuple(raw["allowed_handle_ids"]),
        handle_group_by_id=groups,
        validation_contract=dict(raw.get("validation_contract", {})),
        prompt=prompt,
    )
    _require(
        scope.receipt_sha256 == raw.get("specialist_scope_receipt_sha256")
        and identity_sha256(scope.projection())
        == raw.get("specialist_scope_projection_sha256"),
        "specialist v3 compiled validation scope changed",
    )
    return prompt, scope


def _preflight_projection(
    construction: SealedArtifact,
    rows: tuple[dict[str, Any], ...],
    *,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> dict[str, Any]:
    _require(model == DEFAULT_MODEL, "specialist v3 answer model must be Terra")
    require_text(gateway_url, "specialist v3 Terra gateway")
    _require(
        type(max_concurrency) is int and max_concurrency > 0,
        "specialist v3 concurrency must be positive",
    )
    prompts = tuple(_plain_messages(row["messages"]) for row in rows)
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS
    )
    _require(
        population.logical_prompt_count
        == population.unique_prompt_count
        == EXPECTED_PROVIDER_CALLS,
        "specialist v3 answer population must contain ten unique prompts",
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
        "scoped_completion_format": require_text(
            SCOPED_COMPLETION_FORMAT,
            "specialist v3 scoped completion format",
        ),
    }
    assert_gold_blind(payload, path="specialist_v3_answer_preflight")
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
    assert_gold_blind(payload, path="loaded_specialist_v3_answer_preflight")
    rows = payload.get("physical_prompt_rows")
    _require(
        set(payload) == _PREFLIGHT_FIELDS
        and payload.get("format") == PREFLIGHT_FORMAT
        and payload.get("model") == DEFAULT_MODEL
        and payload.get("gold_loaded") is False
        and payload.get("provider_calls") == 0
        and payload.get("hard_complete_chat_token_cap")
        == HARD_COMPLETE_CHAT_TOKEN_CAP
        and payload.get("max_chat_prompt_tokens") == MAX_CHAT_PROMPT_TOKENS
        and payload.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
        and type(payload.get("observed_max_complete_envelope_tokens")) is int
        and payload["observed_max_complete_envelope_tokens"]
        <= HARD_COMPLETE_CHAT_TOKEN_CAP
        and payload.get("question_count") == EXPECTED_PROVIDER_CALLS
        and payload.get("required_authorized_provider_calls")
        == EXPECTED_PROVIDER_CALLS
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("scoped_completion_format")
        == SCOPED_COMPLETION_FORMAT
        and type(rows) is list
        and len(rows) == EXPECTED_PROVIDER_CALLS,
        "sealed specialist v3 preflight changed",
    )
    require_sha256(
        payload.get("construction_artifact_sha256"),
        "specialist v3 preflight construction",
    )
    prompts: list[tuple[dict[str, str], ...]] = []
    validated: list[dict[str, Any]] = []
    for ordinal, raw in zip(EXPECTED_ORDINALS, rows, strict=True):
        _require(type(raw) is dict, "specialist v3 prompt row changed type")
        assert type(raw) is dict
        body = dict(raw)
        declared = body.pop("prompt_row_receipt_sha256", None)
        prompt, _scope = _scope_from_plan(raw)
        messages = _plain_messages(prompt.messages)
        _require(
            raw.get("ordinal") == ordinal
            and declared == identity_sha256(body)
            and raw.get("messages") == list(messages)
            and raw.get("messages_sha256") == identity_sha256(list(messages))
            and raw.get("prompt_token_proxy") == prompt.prompt_token_proxy
            and prompt.prompt_token_proxy <= MAX_CHAT_PROMPT_TOKENS,
            f"specialist v3 preflight prompt changed at {ordinal}",
        )
        prompts.append(messages)
        validated.append(dict(raw))
    observed_max_complete_envelope_tokens = max(
        row["prompt_token_proxy"] + OUTPUT_TOKEN_RESERVE for row in validated
    )
    _require(
        payload.get("observed_max_complete_envelope_tokens")
        == observed_max_complete_envelope_tokens,
        "sealed specialist v3 observed complete-envelope maximum changed",
    )
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS
    )
    _require(
        population.model_dump() == payload.get("prompt_population")
        and population.prompt_population_sha256
        == payload.get("prompt_population_sha256")
        and population.unique_prompt_count == EXPECTED_PROVIDER_CALLS,
        "sealed specialist v3 prompt population changed",
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
        == require_sha256(expected_sha256, "expected specialist v3 preflight"),
        "specialist v3 preflight digest changed",
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
        "runtime settings differ from sealed specialist v3 preflight",
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
            "arm": "reduced_specialist_scoped_terra_answer_v3",
            "authorized_unique_calls": EXPECTED_PROVIDER_CALLS,
            "construction_artifact_sha256": artifact.payload[
                "construction_artifact_sha256"
            ],
            "experiment_format": FORMAT,
            "gateway_url": gateway_url,
            "gold_loaded": False,
            "preflight_artifact_sha256": artifact.sha256,
            "scoped_completion_format": SCOPED_COMPLETION_FORMAT,
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
        "specialist v3 Terra population changed",
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
        "materialization requires ten checkpoint-only v3 completions",
    )
    records = {row.messages_sha256: row for row in batch.unique_records}
    _require(len(records) == EXPECTED_PROVIDER_CALLS, "v3 completions repeat")
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
            "specialist v3 checkpoint record changed",
        )
        assert record is not None
        _prompt, scope = _scope_from_plan(plan)
        parent = plan["parent_prediction"]
        parsed = parse_specialist_scoped_completion(
            completion,
            parent_prediction=parent,
            scope=scope,
        )
        valid_replace = parsed.valid and parsed.decision == "replace"
        prediction = parsed.prediction if valid_replace else parent
        if valid_replace:
            source = "specialist_scoped_validated_replacement_v1"
            decision = "replace"
            used_handles = list(parsed.used_handle_ids)
        elif parsed.valid:
            source = "specialist_scoped_validated_keep_parent_v1"
            decision = "keep_parent"
            used_handles = []
        else:
            source = "specialist_scoped_invalid_keep_parent_v1"
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
            "proof_kind": parsed.proof_kind,
            "proof_receipt_sha256": parsed.proof_receipt_sha256,
            "prompt_row_receipt_sha256": plan["prompt_row_receipt_sha256"],
            "question_id": plan["question_id"],
            "question_sha256": plan["question_sha256"],
            "request_journal_sha256": record.request_journal_sha256,
            "response_journal_sha256": record.response_journal_sha256,
            "solver_valid": parsed.valid,
            "specialist_scope_receipt_sha256": parsed.scope_receipt_sha256,
            "used_handle_ids": used_handles,
            "validation_basis": parsed.validation_basis,
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
        "construction_artifact_sha256": preflight.payload[
            "construction_artifact_sha256"
        ],
        "format": FORMAT,
        "gold_loaded": False,
        "invalid_completion_parent_fallback_count": sum(
            row["prediction_source"] == "specialist_scoped_invalid_keep_parent_v1"
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
        "scoped_completion_format": SCOPED_COMPLETION_FORMAT,
    }
    assert_gold_blind(payload, path="specialist_scoped_terra_answer_v3")
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
        "specialist v3 construction/preflight binding changed",
    )
    batch = _checkpoint_batch(preflight, prompts, args=args, client=None)
    rebuilt = _materialization_projection(preflight, rows, batch)
    terminal = read_sealed_json(Path(args.output_root) / RUN_NAME)
    _require(
        terminal.sha256
        == require_sha256(args.expected_run_sha256, "expected specialist v3 run")
        and terminal.payload == rebuilt,
        "specialist v3 run differs from checkpoint-only replay",
    )
    replay, _created = publish_sealed_json(
        Path(args.output_root) / REPLAY_NAME, terminal.payload
    )
    _require(
        replay.sha256 == terminal.sha256,
        "specialist v3 replay is not byte-identical",
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

    preflight = commands.add_parser("preflight", help="seal ten scoped prompts")
    _add_runtime_settings(preflight)
    preflight.add_argument("--construction", type=Path, default=DEFAULT_CONSTRUCTION)
    preflight.add_argument("--expected-construction-sha256", required=True)

    provider = commands.add_parser("provider-run", help="execute scoped Terra prompts")
    _add_runtime_settings(provider)
    provider.add_argument("--expected-preflight-sha256", required=True)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, default=0)
    provider.add_argument("--api-key-env", default=live.DEFAULT_API_KEY_ENV)

    materialize = commands.add_parser(
        "materialize", help="materialize checkpoint-only scoped predictions"
    )
    _add_runtime_settings(materialize)
    materialize.add_argument("--expected-preflight-sha256", required=True)

    replay = commands.add_parser("replay", help="prove byte-identical v3 replay")
    _add_runtime_settings(replay)
    replay.add_argument("--construction", type=Path, default=DEFAULT_CONSTRUCTION)
    replay.add_argument("--expected-construction-sha256", required=True)
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
    "EXPECTED_ORDINALS",
    "FORMAT",
    "PREFLIGHT_NAME",
    "REPLAY_NAME",
    "RUN_NAME",
    "ReducedSpecialistAnswerV3Error",
    "build_parser",
    "main",
    "run_materialize",
    "run_preflight",
    "run_provider",
    "run_replay",
]
