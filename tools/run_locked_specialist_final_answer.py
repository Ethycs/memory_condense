#!/usr/bin/env python3
"""Selective checkpointed Terra answers for the locked specialist final arm.

The construction stage owns retrieval and seals one hundred gold-blind question
rows.  Rows in ``specialist`` mode carry a scoped prompt and consume one Terra
completion; rows in ``parent_passthrough`` mode carry only a replay-verified
parent binding and never enter the provider population.  Preflight rerenders
and recompiles every specialist prompt, provider execution is the only network
phase, and materialization/replay consume checkpoints only.

The construction module is imported lazily so this lifecycle has a small,
testable adapter boundary and cannot accidentally make construction a runtime
side effect.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from dotenv import load_dotenv  # noqa: E402

from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from memory_condense.eval.fast_completion_runtime import (  # noqa: E402
    FastCompletionBatch,
    FastCompletionRuntime,
    preflight_fast_completion_prompts,
)
from tools import run_reduced_specialist_answer_v3 as reduced_answer  # noqa: E402
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
from tools.matched_eval.population import EXPECTED_QUESTION_COUNT  # noqa: E402
from tools.matched_eval.specialist_scoped_completion import (  # noqa: E402
    FORMAT as SCOPED_COMPLETION_FORMAT,
    HARD_COMPLETE_CHAT_TOKEN_CAP,
    MAX_CHAT_PROMPT_TOKENS,
    OUTPUT_TOKEN_RESERVE,
    parse_specialist_scoped_completion,
)
from tools.matched_eval.typed_memory_final_arm import (  # noqa: E402
    judge_row_projection,
)


FORMAT = "memory-condense-locked-specialist-final-terra-answer-v1"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight"
RESULT_ROW_FORMAT = f"{FORMAT}-result-row"

PREFLIGHT_NAME = "locked-specialist-final-answer-preflight-v1.json"
RUN_NAME = "locked-specialist-final-answer-v1.json"
REPLAY_NAME = "locked-specialist-final-answer-replay-v1.json"
CHECKPOINT_DIR_NAME = "locked-specialist-final-answer-checkpoints-v1"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONSTRUCTION = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-specialist-final-v1/"
    "locked-specialist-final-construction-v1.json"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-specialist-final-answer-v1"
)
DEFAULT_MODEL = "codex_sdk/gpt-5.6-terra"

SPECIALIST_MODE = "specialist"
PARENT_PASSTHROUGH_MODE = "parent_passthrough"
_MODES = frozenset({SPECIALIST_MODE, PARENT_PASSTHROUGH_MODE})

_COMMON_PLAN_FIELDS = frozenset(
    {
        "answer_plan_receipt_sha256",
        "construction_question_receipt_sha256",
        "dated_question_sha256",
        "mode",
        "ordinal",
        "parent_judge_row_sha256",
        "parent_prediction",
        "parent_prediction_sha256",
        "parent_prediction_source",
        "parent_replay_artifact_sha256",
        "parent_run_artifact_sha256",
        "parent_source_receipt_sha256",
        "parent_source_row_sha256",
        "question_id",
        "question_sha256",
        "route_id",
    }
)
_SPECIALIST_PLAN_FIELDS = _COMMON_PLAN_FIELDS | frozenset(
    {
        "allowed_handle_ids",
        "handle_group_by_id",
        "messages",
        "messages_sha256",
        "prompt_token_proxy",
        "provider_input",
        "specialist_advisories_sha256",
        "specialist_prompt_envelope_receipt_sha256",
        "specialist_scope_projection_sha256",
        "specialist_scope_receipt_sha256",
        "terminal_prompt_receipt_sha256",
        "validation_contract",
    }
)

_PREFLIGHT_FIELDS = frozenset(
    {
        "answer_plan_population_sha256",
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
        "parent_passthrough_count",
        "parent_passthrough_rows",
        "physical_prompt_rows",
        "prompt_population",
        "prompt_population_sha256",
        "provider_calls",
        "question_count",
        "required_authorized_provider_calls",
        "retained_transformer_token_state_bytes",
        "scoped_completion_format",
        "specialist_question_count",
    }
)

ConstructionLoader = Callable[..., tuple[SealedArtifact, Sequence[Mapping[str, Any]]]]


class LockedSpecialistFinalAnswerError(MatchedEvalContractError):
    """Raised when the construction, prompt, checkpoint, or replay diverges."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedSpecialistFinalAnswerError(message)


def _default_construction_loader(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[SealedArtifact, Sequence[Mapping[str, Any]]]:
    module = importlib.import_module("tools.run_locked_specialist_final_construction")
    loader = getattr(module, "load_verified_construction", None)
    _require(callable(loader), "locked specialist construction loader is unavailable")
    return loader(path, expected_sha256=expected_sha256)


def _verified_parent(raw: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
    parent = raw.get("parent_source")
    _require(type(parent) is dict, f"parent source is missing at ordinal {ordinal}")
    assert type(parent) is dict
    _require(
        set(parent)
        == {
            "parent_judge_row",
            "parent_judge_row_sha256",
            "prediction",
            "prediction_sha256",
            "receipt_sha256",
            "replay_artifact_sha256",
            "run_artifact_sha256",
            "source_row_sha256",
        },
        f"parent source schema changed at ordinal {ordinal}",
    )
    body = dict(parent)
    declared_receipt = body.pop("receipt_sha256", None)
    judge = parent.get("parent_judge_row")
    _require(
        type(judge) is dict
        and declared_receipt == identity_sha256(body)
        and parent.get("parent_judge_row_sha256") == identity_sha256(judge)
        and parent.get("prediction_sha256")
        == quote_sha256(require_text(parent.get("prediction"), "parent prediction"))
        and parent.get("source_row_sha256") == judge.get("source_row_sha256")
        and parent.get("prediction") == judge.get("prediction")
        and parent.get("prediction_sha256") == judge.get("prediction_sha256"),
        f"parent source seal changed at ordinal {ordinal}",
    )
    assert type(judge) is dict
    _require(
        judge_row_projection(judge) == judge
        and judge.get("ordinal") == ordinal
        and judge.get("question_id") == raw.get("question_id")
        and judge.get("question_sha256") == raw.get("question_sha256")
        and judge.get("dated_question_sha256") == raw.get("dated_question_sha256"),
        f"parent judge seam changed at ordinal {ordinal}",
    )
    require_sha256(parent.get("run_artifact_sha256"), "parent run artifact")
    require_sha256(parent.get("replay_artifact_sha256"), "parent replay artifact")
    require_sha256(parent.get("source_row_sha256"), "parent source row")
    require_sha256(parent.get("parent_judge_row_sha256"), "parent judge row")
    require_sha256(declared_receipt, "parent source receipt")
    return dict(parent)


def _common_plan(
    raw: Mapping[str, Any],
    ordinal: int,
    *,
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    judge = parent["parent_judge_row"]
    assert type(judge) is dict
    mode = raw.get("mode")
    _require(mode in _MODES, f"answer mode changed at ordinal {ordinal}")
    return {
        "construction_question_receipt_sha256": require_sha256(
            raw.get("question_receipt_sha256"), "construction question receipt"
        ),
        "dated_question_sha256": require_sha256(
            raw.get("dated_question_sha256"), "dated question"
        ),
        "mode": mode,
        "ordinal": ordinal,
        "parent_judge_row_sha256": require_sha256(
            parent.get("parent_judge_row_sha256"), "parent judge row"
        ),
        "parent_prediction": require_text(parent.get("prediction"), "parent prediction"),
        "parent_prediction_sha256": require_sha256(
            parent.get("prediction_sha256"), "parent prediction"
        ),
        "parent_prediction_source": require_text(
            judge.get("prediction_source"), "parent prediction source"
        ),
        "parent_replay_artifact_sha256": require_sha256(
            parent.get("replay_artifact_sha256"), "parent replay artifact"
        ),
        "parent_run_artifact_sha256": require_sha256(
            parent.get("run_artifact_sha256"), "parent run artifact"
        ),
        "parent_source_receipt_sha256": require_sha256(
            parent.get("receipt_sha256"), "parent source receipt"
        ),
        "parent_source_row_sha256": require_sha256(
            parent.get("source_row_sha256"), "parent source row"
        ),
        "question_id": require_text(raw.get("question_id"), "question ID"),
        "question_sha256": require_sha256(raw.get("question_sha256"), "question"),
        "route_id": require_text(judge.get("route_id"), "parent route"),
    }


def _source_plan(raw: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
    source_body = dict(raw)
    declared_question_receipt = source_body.pop("question_receipt_sha256", None)
    _require(
        raw.get("ordinal") == ordinal
        and declared_question_receipt == identity_sha256(source_body),
        f"construction question seal/order changed at ordinal {ordinal}",
    )
    parent = _verified_parent(raw, ordinal)
    common = _common_plan(raw, ordinal, parent=parent)
    if raw.get("mode") == PARENT_PASSTHROUGH_MODE:
        _require(
            raw.get("terminal_prompt") is None
            and raw.get("methods") == []
            and raw.get("fitted_typed_prompt") is None,
            f"parent passthrough carries a provider prompt at ordinal {ordinal}",
        )
        plan = common
    else:
        try:
            scoped = reduced_answer._prompt_plan_row(raw, ordinal)  # noqa: SLF001
        except MatchedEvalContractError as exc:
            raise LockedSpecialistFinalAnswerError(
                f"specialist scoped construction changed at ordinal {ordinal}: {exc}"
            ) from exc
        _require(
            scoped["parent_prediction"] == common["parent_prediction"]
            and scoped["parent_prediction_sha256"]
            == common["parent_prediction_sha256"]
            and scoped["question_id"] == common["question_id"]
            and scoped["question_sha256"] == common["question_sha256"]
            and scoped["dated_question_sha256"] == common["dated_question_sha256"]
            and scoped["construction_question_receipt_sha256"]
            == common["construction_question_receipt_sha256"],
            f"specialist prompt escaped its sealed parent at ordinal {ordinal}",
        )
        plan = {
            **common,
            **{
                key: value
                for key, value in scoped.items()
                if key
                not in {
                    "construction_question_receipt_sha256",
                    "dated_question_sha256",
                    "ordinal",
                    "parent_prediction",
                    "parent_prediction_sha256",
                    "prompt_row_receipt_sha256",
                    "question_id",
                    "question_sha256",
                }
            },
        }
    result = {**plan, "answer_plan_receipt_sha256": identity_sha256(plan)}
    _validate_stored_plan(result)
    assert_gold_blind(result, path=f"locked_specialist_answer_plan_{ordinal}")
    return result


def _scoped_prompt_and_scope(raw: Mapping[str, Any]) -> tuple[Any, Any]:
    try:
        return reduced_answer._scope_from_plan(raw)  # noqa: SLF001
    except MatchedEvalContractError as exc:
        raise LockedSpecialistFinalAnswerError(
            f"specialist scoped answer plan changed at ordinal {raw.get('ordinal')}: {exc}"
        ) from exc


def _validate_stored_plan(raw: Mapping[str, Any]) -> dict[str, Any]:
    mode = raw.get("mode")
    expected_fields = (
        _SPECIALIST_PLAN_FIELDS if mode == SPECIALIST_MODE else _COMMON_PLAN_FIELDS
    )
    _require(
        mode in _MODES and set(raw) == expected_fields,
        f"sealed answer plan schema changed at ordinal {raw.get('ordinal')}",
    )
    body = dict(raw)
    declared = body.pop("answer_plan_receipt_sha256", None)
    ordinal = raw.get("ordinal")
    _require(
        type(ordinal) is int
        and 0 <= ordinal < EXPECTED_QUESTION_COUNT
        and declared == identity_sha256(body)
        and raw.get("parent_prediction_sha256")
        == quote_sha256(require_text(raw.get("parent_prediction"), "parent prediction")),
        f"sealed answer plan receipt changed at ordinal {ordinal}",
    )
    for key in (
        "construction_question_receipt_sha256",
        "dated_question_sha256",
        "parent_judge_row_sha256",
        "parent_prediction_sha256",
        "parent_replay_artifact_sha256",
        "parent_run_artifact_sha256",
        "parent_source_receipt_sha256",
        "parent_source_row_sha256",
        "question_sha256",
    ):
        require_sha256(raw.get(key), f"answer plan {key}")
    require_text(raw.get("question_id"), "answer plan question ID")
    require_text(raw.get("parent_prediction_source"), "parent prediction source")
    require_text(raw.get("route_id"), "answer plan route")
    if mode == SPECIALIST_MODE:
        prompt, _scope = _scoped_prompt_and_scope(raw)
        messages = reduced_answer._plain_messages(prompt.messages)  # noqa: SLF001
        _require(
            raw.get("messages") == list(messages)
            and raw.get("messages_sha256") == identity_sha256(list(messages))
            and raw.get("prompt_token_proxy") == prompt.prompt_token_proxy
            and prompt.prompt_token_proxy <= MAX_CHAT_PROMPT_TOKENS
            and prompt.prompt_token_proxy + OUTPUT_TOKEN_RESERVE
            <= HARD_COMPLETE_CHAT_TOKEN_CAP,
            f"sealed specialist prompt changed at ordinal {ordinal}",
        )
    assert_gold_blind(raw, path=f"loaded_locked_specialist_answer_plan_{ordinal}")
    return dict(raw)


def load_answer_plans(
    path: str | Path,
    expected_sha256: str,
    *,
    construction_loader: ConstructionLoader | None = None,
) -> tuple[SealedArtifact, tuple[dict[str, Any], ...]]:
    """Load a sealed construction through the lazy adapter and derive plans."""

    expected = require_sha256(expected_sha256, "expected specialist construction")
    loader = construction_loader or _default_construction_loader
    artifact, raw_rows = loader(Path(path), expected_sha256=expected)
    _require(
        isinstance(artifact, SealedArtifact)
        and artifact.sha256 == expected
        and isinstance(raw_rows, (tuple, list))
        and len(raw_rows) == EXPECTED_QUESTION_COUNT,
        "locked specialist construction digest or population changed",
    )
    plans = tuple(
        _source_plan(raw, ordinal)
        for ordinal, raw in enumerate(raw_rows)
        if isinstance(raw, Mapping)
    )
    _require(
        len(plans) == EXPECTED_QUESTION_COUNT
        and tuple(row["ordinal"] for row in plans)
        == tuple(range(EXPECTED_QUESTION_COUNT))
        and len({row["question_id"] for row in plans}) == EXPECTED_QUESTION_COUNT,
        "locked specialist construction question identities changed",
    )
    return artifact, plans


def _preflight_projection(
    construction: SealedArtifact,
    plans: tuple[dict[str, Any], ...],
    *,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> dict[str, Any]:
    _require(model == DEFAULT_MODEL, "locked specialist answer model must be Terra")
    require_text(gateway_url, "locked specialist Terra gateway")
    _require(
        type(max_concurrency) is int and max_concurrency > 0,
        "locked specialist concurrency must be positive",
    )
    validated = tuple(_validate_stored_plan(row) for row in plans)
    _require(
        len(validated) == EXPECTED_QUESTION_COUNT
        and tuple(row["ordinal"] for row in validated)
        == tuple(range(EXPECTED_QUESTION_COUNT)),
        "locked specialist answer plan population changed",
    )
    specialist = tuple(row for row in validated if row["mode"] == SPECIALIST_MODE)
    passthrough = tuple(
        row for row in validated if row["mode"] == PARENT_PASSTHROUGH_MODE
    )
    _require(bool(specialist), "locked specialist provider population is empty")
    prompts = tuple(
        reduced_answer._plain_messages(row["messages"])  # noqa: SLF001
        for row in specialist
    )
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS
    )
    _require(
        population.logical_prompt_count
        == population.unique_prompt_count
        == len(specialist),
        "locked specialist prompts must be unique",
    )
    observed_max = max(
        row["prompt_token_proxy"] + OUTPUT_TOKEN_RESERVE for row in specialist
    )
    _require(
        observed_max <= HARD_COMPLETE_CHAT_TOKEN_CAP,
        "locked specialist complete prompt envelope exceeds the hard cap",
    )
    payload = {
        "answer_plan_population_sha256": identity_sha256(
            [row["answer_plan_receipt_sha256"] for row in validated]
        ),
        "construction_artifact_sha256": construction.sha256,
        "format": PREFLIGHT_FORMAT,
        "gateway_url": gateway_url,
        "gold_loaded": False,
        "hard_complete_chat_token_cap": HARD_COMPLETE_CHAT_TOKEN_CAP,
        "max_chat_prompt_tokens": MAX_CHAT_PROMPT_TOKENS,
        "max_concurrency": max_concurrency,
        "model": model,
        "observed_max_complete_envelope_tokens": observed_max,
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "parent_passthrough_count": len(passthrough),
        "parent_passthrough_rows": list(passthrough),
        "physical_prompt_rows": list(specialist),
        "prompt_population": population.model_dump(),
        "prompt_population_sha256": population.prompt_population_sha256,
        "provider_calls": 0,
        "question_count": EXPECTED_QUESTION_COUNT,
        "required_authorized_provider_calls": len(specialist),
        "retained_transformer_token_state_bytes": 0,
        "scoped_completion_format": SCOPED_COMPLETION_FORMAT,
        "specialist_question_count": len(specialist),
    }
    assert_gold_blind(payload, path="locked_specialist_answer_preflight")
    return payload


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    construction, plans = load_answer_plans(
        Path(args.construction), str(args.expected_construction_sha256)
    )
    payload = _preflight_projection(
        construction,
        plans,
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
        "parent_passthrough_count": payload["parent_passthrough_count"],
        "physical_provider_calls": 0,
        "preflight_sha256": artifact.sha256,
        "question_count": EXPECTED_QUESTION_COUNT,
        "required_authorized_provider_calls": payload[
            "required_authorized_provider_calls"
        ],
        "specialist_question_count": payload["specialist_question_count"],
    }


def _validate_preflight(
    artifact: SealedArtifact,
) -> tuple[
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
]:
    payload = artifact.payload
    assert_gold_blind(payload, path="loaded_locked_specialist_answer_preflight")
    specialist = payload.get("physical_prompt_rows")
    passthrough = payload.get("parent_passthrough_rows")
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
        and type(payload.get("max_concurrency")) is int
        and payload["max_concurrency"] > 0
        and type(payload.get("observed_max_complete_envelope_tokens")) is int
        and payload["observed_max_complete_envelope_tokens"]
        <= HARD_COMPLETE_CHAT_TOKEN_CAP
        and payload.get("question_count") == EXPECTED_QUESTION_COUNT
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("scoped_completion_format") == SCOPED_COMPLETION_FORMAT
        and type(specialist) is list
        and bool(specialist)
        and type(passthrough) is list
        and len(specialist) == payload.get("specialist_question_count")
        == payload.get("required_authorized_provider_calls")
        and len(passthrough) == payload.get("parent_passthrough_count")
        and len(specialist) + len(passthrough) == EXPECTED_QUESTION_COUNT,
        "sealed locked specialist answer preflight changed",
    )
    require_sha256(
        payload.get("construction_artifact_sha256"),
        "locked specialist preflight construction",
    )
    validated_specialist = tuple(_validate_stored_plan(row) for row in specialist)
    validated_passthrough = tuple(_validate_stored_plan(row) for row in passthrough)
    _require(
        all(row["mode"] == SPECIALIST_MODE for row in validated_specialist)
        and all(
            row["mode"] == PARENT_PASSTHROUGH_MODE
            for row in validated_passthrough
        ),
        "locked specialist preflight modes changed",
    )
    ordered = tuple(
        sorted((*validated_specialist, *validated_passthrough), key=lambda row: row["ordinal"])
    )
    _require(
        tuple(row["ordinal"] for row in ordered)
        == tuple(range(EXPECTED_QUESTION_COUNT))
        and len({row["question_id"] for row in ordered}) == EXPECTED_QUESTION_COUNT
        and payload.get("answer_plan_population_sha256")
        == identity_sha256(
            [row["answer_plan_receipt_sha256"] for row in ordered]
        ),
        "sealed locked specialist answer plan population changed",
    )
    prompts = tuple(
        reduced_answer._plain_messages(row["messages"])  # noqa: SLF001
        for row in validated_specialist
    )
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS
    )
    observed_max = max(
        row["prompt_token_proxy"] + OUTPUT_TOKEN_RESERVE
        for row in validated_specialist
    )
    _require(
        population.model_dump() == payload.get("prompt_population")
        and population.prompt_population_sha256
        == payload.get("prompt_population_sha256")
        and population.logical_prompt_count
        == population.unique_prompt_count
        == len(validated_specialist)
        and payload.get("observed_max_complete_envelope_tokens") == observed_max,
        "sealed locked specialist prompt population changed",
    )
    return prompts, ordered


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
        == require_sha256(expected_sha256, "expected locked specialist preflight"),
        "locked specialist preflight digest changed",
    )
    prompts, plans = _validate_preflight(artifact)
    return artifact, prompts, plans


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
    required = artifact.payload["required_authorized_provider_calls"]
    _require(
        model == DEFAULT_MODEL == artifact.payload.get("model")
        and gateway_url == artifact.payload.get("gateway_url")
        and max_concurrency == artifact.payload.get("max_concurrency")
        and len(prompts) == required,
        "runtime settings differ from sealed locked specialist preflight",
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
            "arm": "locked_specialist_final_terra_answer_v1",
            "authorized_unique_calls": required,
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
    preflight, prompts, _plans = _read_preflight(
        Path(args.output_root), str(args.expected_preflight_sha256)
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
        "locked specialist Terra population changed",
    )
    return {
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "gold_loaded": False,
        "physical_provider_calls": batch.usage.physical_calls,
        "preflight_sha256": preflight.sha256,
        "required_authorized_provider_calls": required,
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


def _result_body(
    plan: Mapping[str, Any],
    *,
    answer_mode: str,
    prediction: str,
    prediction_source: str,
    decision: str,
    call_key_sha256: str | None = None,
    completion_receipt_sha256: str | None = None,
    parse_error_code: str | None = None,
    parse_receipt_sha256: str | None = None,
    proof_kind: str | None = None,
    proof_receipt_sha256: str | None = None,
    prompt_row_receipt_sha256: str | None = None,
    request_journal_sha256: str | None = None,
    response_journal_sha256: str | None = None,
    solver_valid: bool | None = None,
    specialist_scope_receipt_sha256: str | None = None,
    used_handle_ids: Sequence[str] = (),
    validation_basis: str | None = None,
) -> dict[str, Any]:
    parent = plan["parent_prediction"]
    return {
        "answer_mode": answer_mode,
        "call_key_sha256": call_key_sha256,
        "changed_from_parent": prediction != parent,
        "completion_receipt_sha256": completion_receipt_sha256,
        "construction_question_receipt_sha256": plan[
            "construction_question_receipt_sha256"
        ],
        "dated_question_sha256": plan["dated_question_sha256"],
        "decision": decision,
        "format": RESULT_ROW_FORMAT,
        "ordinal": plan["ordinal"],
        "parent_judge_row_sha256": plan["parent_judge_row_sha256"],
        "parent_prediction_sha256": plan["parent_prediction_sha256"],
        "parent_prediction_source": plan["parent_prediction_source"],
        "parent_replay_artifact_sha256": plan[
            "parent_replay_artifact_sha256"
        ],
        "parent_run_artifact_sha256": plan["parent_run_artifact_sha256"],
        "parent_source_receipt_sha256": plan["parent_source_receipt_sha256"],
        "parent_source_row_sha256": plan["parent_source_row_sha256"],
        "parse_error_code": parse_error_code,
        "parse_receipt_sha256": parse_receipt_sha256,
        "prediction": prediction,
        "prediction_sha256": quote_sha256(prediction),
        "prediction_source": prediction_source,
        "proof_kind": proof_kind,
        "proof_receipt_sha256": proof_receipt_sha256,
        "prompt_row_receipt_sha256": prompt_row_receipt_sha256,
        "question_id": plan["question_id"],
        "question_sha256": plan["question_sha256"],
        "request_journal_sha256": request_journal_sha256,
        "response_journal_sha256": response_journal_sha256,
        "retained_transformer_token_state_bytes": 0,
        "route_id": plan["route_id"],
        "solver_valid": solver_valid,
        "specialist_scope_receipt_sha256": specialist_scope_receipt_sha256,
        "used_handle_ids": list(used_handle_ids),
        "validation_basis": validation_basis,
    }


def _materialization_projection(
    preflight: SealedArtifact,
    plans: tuple[dict[str, Any], ...],
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
        "materialization requires every specialist checkpoint and no provider calls",
    )
    specialist = tuple(row for row in plans if row["mode"] == SPECIALIST_MODE)
    _require(
        len(plans) == EXPECTED_QUESTION_COUNT
        and len(specialist) == required,
        "materialization answer plan population changed",
    )
    records = {row.messages_sha256: row for row in batch.unique_records}
    completions = {
        plan["ordinal"]: completion
        for plan, completion in zip(
            specialist, batch.logical_completions, strict=True
        )
    }
    _require(len(records) == required, "locked specialist completions repeat")
    results: list[dict[str, Any]] = []
    for plan in plans:
        parent = plan["parent_prediction"]
        if plan["mode"] == PARENT_PASSTHROUGH_MODE:
            body = _result_body(
                plan,
                answer_mode=PARENT_PASSTHROUGH_MODE,
                prediction=parent,
                prediction_source="locked_specialist_parent_passthrough_v1",
                decision=PARENT_PASSTHROUGH_MODE,
            )
        else:
            completion = completions[plan["ordinal"]]
            record = records.get(plan["messages_sha256"])
            _require(
                record is not None
                and record.completion == completion
                and record.checkpoint_hit is True
                and record.physical_call is False,
                f"specialist checkpoint changed at ordinal {plan['ordinal']}",
            )
            assert record is not None
            _prompt, scope = _scoped_prompt_and_scope(plan)
            parsed = parse_specialist_scoped_completion(
                completion,
                parent_prediction=parent,
                scope=scope,
            )
            valid_replace = parsed.valid and parsed.decision == "replace"
            prediction = parsed.prediction if valid_replace else parent
            if valid_replace:
                source = "locked_specialist_scoped_validated_replacement_v1"
                decision = "replace"
                used_handles = parsed.used_handle_ids
            elif parsed.valid:
                source = "locked_specialist_scoped_validated_keep_parent_v1"
                decision = "keep_parent"
                used_handles = ()
            else:
                source = "locked_specialist_scoped_invalid_keep_parent_v1"
                decision = "invalid_keep_parent"
                used_handles = ()
            body = _result_body(
                plan,
                answer_mode=SPECIALIST_MODE,
                prediction=prediction,
                prediction_source=source,
                decision=decision,
                call_key_sha256=record.call_key_sha256,
                completion_receipt_sha256=record.completion_sha256,
                parse_error_code=parsed.error_code,
                parse_receipt_sha256=parsed.receipt_sha256,
                proof_kind=parsed.proof_kind,
                proof_receipt_sha256=parsed.proof_receipt_sha256,
                prompt_row_receipt_sha256=plan["answer_plan_receipt_sha256"],
                request_journal_sha256=record.request_journal_sha256,
                response_journal_sha256=record.response_journal_sha256,
                solver_valid=parsed.valid,
                specialist_scope_receipt_sha256=parsed.scope_receipt_sha256,
                used_handle_ids=used_handles,
                validation_basis=parsed.validation_basis,
            )
        row = {**body, "source_row_sha256": identity_sha256(body)}
        results.append(row)
    _require(
        tuple(row["ordinal"] for row in results)
        == tuple(range(EXPECTED_QUESTION_COUNT)),
        "locked specialist result order changed",
    )
    judge_rows = [judge_row_projection(row) for row in results]
    payload = {
        "changed_prediction_count": sum(row["changed_from_parent"] for row in results),
        "completion_batch": _stable_batch(batch),
        "construction_artifact_sha256": preflight.payload[
            "construction_artifact_sha256"
        ],
        "format": FORMAT,
        "gold_loaded": False,
        "invalid_completion_parent_fallback_count": sum(
            row["prediction_source"]
            == "locked_specialist_scoped_invalid_keep_parent_v1"
            for row in results
        ),
        "judge_rows": judge_rows,
        "model": DEFAULT_MODEL,
        "parent_passthrough_count": sum(
            row["answer_mode"] == PARENT_PASSTHROUGH_MODE for row in results
        ),
        "physical_provider_calls_during_materialization": 0,
        "preflight_artifact_sha256": preflight.sha256,
        "question_count": EXPECTED_QUESTION_COUNT,
        "questions": results,
        "required_authorized_provider_calls": required,
        "retained_transformer_token_state_bytes": 0,
        "scoped_completion_format": SCOPED_COMPLETION_FORMAT,
        "specialist_question_count": required,
        "validated_keep_parent_count": sum(
            row["prediction_source"]
            == "locked_specialist_scoped_validated_keep_parent_v1"
            for row in results
        ),
        "validated_replacement_count": sum(
            row["prediction_source"]
            == "locked_specialist_scoped_validated_replacement_v1"
            for row in results
        ),
    }
    assert_gold_blind(payload, path="locked_specialist_final_terra_answer_v1")
    return payload


def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    preflight, prompts, plans = _read_preflight(
        Path(args.output_root), str(args.expected_preflight_sha256)
    )
    batch = _checkpoint_batch(preflight, prompts, args=args, client=None)
    payload = _materialization_projection(preflight, plans, batch)
    artifact, created = publish_sealed_json(Path(args.output_root) / RUN_NAME, payload)
    return {
        "changed_prediction_count": payload["changed_prediction_count"],
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "parent_passthrough_count": payload["parent_passthrough_count"],
        "physical_provider_calls": 0,
        "run_sha256": artifact.sha256,
        "terminal_run_replayed": not created,
        "validated_replacement_count": payload["validated_replacement_count"],
    }


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    construction, source_plans = load_answer_plans(
        Path(args.construction), str(args.expected_construction_sha256)
    )
    preflight, prompts, plans = _read_preflight(
        Path(args.output_root), str(args.expected_preflight_sha256)
    )
    _require(
        preflight.payload.get("construction_artifact_sha256") == construction.sha256
        and source_plans == plans,
        "locked specialist construction/preflight binding changed",
    )
    batch = _checkpoint_batch(preflight, prompts, args=args, client=None)
    rebuilt = _materialization_projection(preflight, plans, batch)
    terminal = read_sealed_json(Path(args.output_root) / RUN_NAME)
    _require(
        terminal.sha256
        == require_sha256(args.expected_run_sha256, "expected locked specialist run")
        and terminal.payload == rebuilt,
        "locked specialist run differs from checkpoint-only replay",
    )
    replay, _created = publish_sealed_json(
        Path(args.output_root) / REPLAY_NAME, terminal.payload
    )
    _require(replay.sha256 == terminal.sha256, "locked specialist replay is not byte-identical")
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

    preflight = commands.add_parser("preflight", help="seal specialist-only prompts")
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
        "materialize", help="materialize all 100 rows from specialist checkpoints"
    )
    _add_runtime_settings(materialize)
    materialize.add_argument("--expected-preflight-sha256", required=True)

    replay = commands.add_parser("replay", help="prove byte-identical final replay")
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
    "DEFAULT_MODEL",
    "DEFAULT_OUTPUT",
    "FORMAT",
    "LockedSpecialistFinalAnswerError",
    "PREFLIGHT_FORMAT",
    "PREFLIGHT_NAME",
    "REPLAY_NAME",
    "RESULT_ROW_FORMAT",
    "RUN_NAME",
    "build_parser",
    "load_answer_plans",
    "main",
    "run_materialize",
    "run_preflight",
    "run_provider",
    "run_replay",
]
