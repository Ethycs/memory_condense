#!/usr/bin/env python3
"""Run the sealed remaining-24 typed-fact compiler and answer treatment.

The treatment has two independent completion planes.  ``compiler-*`` turns
the already-selected v3 shared-surplus evidence into source-cited atomic
facts.  Only after that plane has materialized and replayed byte-identically
may ``answer-preflight`` render a second, facts-first Terra population.  The
answer plane retains the original protected parent and validator contract.

No benchmark reference or verdict is opened by either plane.  ``judge-
preflight`` is the sole gold-bearing seam: it first verifies the sealed answer
run/replay, then delegates to the common selected-subset Sol judge protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
from tools import run_locked_typed_memory_final_arm as typed_cli  # noqa: E402
from tools.matched_eval import judging, live  # noqa: E402
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
from tools.matched_eval.typed_fact_compiler import (  # noqa: E402
    ANSWER_OUTPUT_TOKEN_RESERVE,
    COMPILER_OUTPUT_TOKEN_RESERVE,
    build_answer_messages,
    build_typed_fact_compiler_messages,
    parse_typed_fact_compiler_response,
)
from tools.matched_eval.typed_memory_final_arm import (  # noqa: E402
    MAX_CHAT_PROMPT_TOKENS,
    VALIDATOR_POLICY_FORMAT,
    judge_row_projection,
    materialize_typed_final_result_row,
)
from tools.matched_eval.typed_memory_final_judging import (  # noqa: E402
    PREFLIGHT_NAME as JUDGE_PREFLIGHT_NAME,
    load_locked_typed_final_gold,
    preflight_projection as judge_preflight_projection,
)
from tools.run_locked_query_answer_judge import DEFAULT_DATASET  # noqa: E402
from tools.run_matched_eval_spine import DEFAULT_SPLIT  # noqa: E402


FORMAT = "memory-condense-locked-typed-memory-fact-compiler-v2"
COMPILER_PREFLIGHT_FORMAT = f"{FORMAT}-compiler-preflight-v1"
COMPILER_RUN_FORMAT = f"{FORMAT}-compiler-run-v1"
COMPILER_REPLAY_FORMAT = f"{FORMAT}-compiler-replay-v1"
ANSWER_PREFLIGHT_FORMAT = f"{FORMAT}-answer-preflight-v1"
ANSWER_RUN_FORMAT = f"{FORMAT}-answer-run-v1"
ANSWER_REPLAY_FORMAT = f"{FORMAT}-answer-replay-v1"

COMPILER_PREFLIGHT_NAME = "typed-fact-compiler-preflight-v2.json"
COMPILER_RUN_NAME = "typed-fact-compiler-run-v2.json"
COMPILER_REPLAY_NAME = "typed-fact-compiler-replay-v2.json"
ANSWER_PREFLIGHT_NAME = "typed-fact-answer-preflight-v2.json"
ANSWER_RUN_NAME = "typed-fact-answer-run-v2.json"
ANSWER_REPLAY_NAME = "typed-fact-answer-replay-v2.json"
COMPILER_CHECKPOINT_DIR_NAME = "terra-typed-fact-compiler-v2-calls"
ANSWER_CHECKPOINT_DIR_NAME = "terra-typed-fact-answer-v2-calls"

HARD_PROMPT_TOKEN_CAP = 8_000
REMAINING_ORDINALS = (
    6,
    7,
    14,
    16,
    28,
    31,
    36,
    42,
    43,
    49,
    53,
    54,
    61,
    65,
    67,
    69,
    72,
    77,
    79,
    81,
    86,
    93,
    94,
    97,
)
SUBSET_QUESTION_COUNT = len(REMAINING_ORDINALS)

EXPECTED_SOURCE_COMPOSITION_SHA256 = (
    "730a437e242174d188ae67484d9414d87c74d8ed926d9e4cdc726c7d5260317f"
)
EXPECTED_SOURCE_PREFLIGHT_SHA256 = (
    "c74874b4ff13189afd31902cd77f812cc67accf51797a5e6f5022e9fa1f961d0"
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = (
    REPOSITORY_ROOT
    / "eval_results"
    / "matched_eval_100"
    / "typed-memory-final-v3-shared-surplus"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "eval_results"
    / "matched_eval_100"
    / "typed-memory-final-v3-fact-compiler-remaining24-v2"
)
DEFAULT_JUDGE_OUTPUT = DEFAULT_OUTPUT / "sol-judge-v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class LockedTypedFactCompilerError(MatchedEvalContractError):
    """A source, phase boundary, completion journal, or validator changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedTypedFactCompilerError(message)


def _sha256_argument(value: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return value


def _plain_messages(value: object, *, label: str) -> tuple[dict[str, str], ...]:
    _require(type(value) in {list, tuple}, f"{label} messages changed type")
    assert isinstance(value, (list, tuple))
    rows = tuple(
        {"role": row["role"], "content": row["content"]}
        for row in value
        if type(row) is dict
        and set(row) == {"role", "content"}
        and row.get("role") in {"system", "user", "assistant"}
        and type(row.get("content")) is str
    )
    _require(len(rows) == len(value), f"{label} messages changed shape")
    return rows


def _projection(value: object, *, label: str) -> dict[str, Any]:
    for name in ("projection", "provider_projection", "identity_payload", "model_dump"):
        method = getattr(value, name, None)
        if callable(method):
            projected = method()
            _require(type(projected) is dict, f"{label} projection changed type")
            assert type(projected) is dict
            return dict(projected)
    _require(False, f"{label} has no stable projection")
    raise AssertionError("unreachable")


def _contains_key(value: object, forbidden: frozenset[str]) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in forbidden
            or _contains_key(child, forbidden)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_key(child, forbidden) for child in value)
    return False


def _decoded_user_payloads(
    messages: Sequence[Mapping[str, str]], *, label: str
) -> tuple[object, ...]:
    decoded: list[object] = []
    for message in messages:
        if message["role"] != "user":
            continue
        try:
            decoded.append(json.loads(message["content"]))
        except json.JSONDecodeError as exc:
            raise LockedTypedFactCompilerError(
                f"{label} user message is not sealed JSON"
            ) from exc
    _require(bool(decoded), f"{label} has no JSON user payload")
    assert_gold_blind(decoded, path=f"{label}_decoded_user_payloads")
    return tuple(decoded)


def _read_source_population(
    source_root: Path,
    *,
    expected_composition_sha256: str,
    expected_preflight_sha256: str,
) -> tuple[
    SealedArtifact,
    SealedArtifact,
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    composition = typed_cli._read_composition(  # noqa: SLF001
        source_root,
        require_sha256(expected_composition_sha256, "expected source composition"),
    )
    preflight = read_sealed_json(source_root / typed_cli.PREFLIGHT_NAME)
    _require(
        preflight.sha256
        == require_sha256(expected_preflight_sha256, "expected source preflight")
        and preflight.payload.get("composition_artifact_sha256")
        == composition.sha256,
        "source v3 preflight/composition binding changed",
    )
    _prompts, prompt_rows = typed_cli._validate_preflight(preflight)  # noqa: SLF001
    composition_rows = composition.payload.get("questions")
    _require(
        type(composition_rows) is list and len(composition_rows) == 100,
        "source v3 composition population changed",
    )
    selected_composition: list[dict[str, Any]] = []
    selected_prompts: list[dict[str, Any]] = []
    for ordinal in REMAINING_ORDINALS:
        raw = composition_rows[ordinal]
        _require(type(raw) is dict, "source composition row changed type")
        assert type(raw) is dict
        expected_plan = typed_cli._prompt_plan_row(raw)  # noqa: SLF001
        _require(
            expected_plan == prompt_rows[ordinal],
            f"source composition/preflight row changed at ordinal {ordinal}",
        )
        selected_composition.append(dict(raw))
        selected_prompts.append(dict(prompt_rows[ordinal]))
    return (
        composition,
        preflight,
        tuple(selected_composition),
        tuple(selected_prompts),
    )


def _prompt_population(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_reserve: int,
    label: str,
) -> tuple[tuple[tuple[dict[str, str], ...], ...], Any]:
    _require(
        type(output_reserve) is int and 0 < output_reserve < HARD_PROMPT_TOKEN_CAP,
        f"{label} output reserve changed",
    )
    prompts: list[tuple[dict[str, str], ...]] = []
    for ordinal, raw in zip(REMAINING_ORDINALS, rows, strict=True):
        declared = require_sha256(
            raw.get("prompt_row_receipt_sha256"), f"{label} prompt row"
        )
        body = dict(raw)
        body.pop("prompt_row_receipt_sha256")
        messages = _plain_messages(raw.get("messages"), label=label)
        _require(
            identity_sha256(body) == declared
            and raw.get("ordinal") == ordinal
            and identity_sha256(list(messages)) == raw.get("messages_sha256")
            and count_chat_prompt_token_proxy(messages)
            == raw.get("prompt_token_proxy")
            and int(raw["prompt_token_proxy"]) + output_reserve
            <= HARD_PROMPT_TOKEN_CAP,
            f"{label} prompt seal/order/envelope changed at ordinal {ordinal}",
        )
        _decoded_user_payloads(messages, label=label)
        prompts.append(messages)
    population = preflight_fast_completion_prompts(
        prompts,
        max_prompt_tokens=HARD_PROMPT_TOKEN_CAP - output_reserve,
    )
    _require(
        population.logical_prompt_count
        == population.unique_prompt_count
        == SUBSET_QUESTION_COUNT,
        f"{label} requires 24 distinct prompts",
    )
    return tuple(prompts), population


def _compiler_prompt_row(
    composition_row: Mapping[str, Any],
    source_plan: Mapping[str, Any],
) -> dict[str, Any]:
    provider = composition_row.get("provider_projection")
    _require(type(provider) is dict, "source provider projection missing")
    assert type(provider) is dict
    provider_input = provider.get("provider_input")
    _require(type(provider_input) is dict, "source provider input missing")
    assert type(provider_input) is dict
    messages = _plain_messages(
        build_typed_fact_compiler_messages(composition_row),
        label="compiler",
    )
    decoded = _decoded_user_payloads(messages, label="compiler")
    _require(
        not _contains_key(
            decoded,
            frozenset(
                {
                    "parent_prediction",
                    "protected_parent_fallback",
                    "prediction_sha256",
                }
            ),
        ),
        "compiler provider plane contains the protected parent",
    )
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    _require(
        prompt_tokens + COMPILER_OUTPUT_TOKEN_RESERVE
        <= HARD_PROMPT_TOKEN_CAP,
        "compiler complete prompt envelope exceeds 8k",
    )
    body = {
        "composition_row_sha256": composition_row["composition_row_sha256"],
        "dated_question_sha256": source_plan["dated_question_sha256"],
        "messages": list(messages),
        "messages_sha256": identity_sha256(list(messages)),
        "ordinal": source_plan["ordinal"],
        "prompt_token_proxy": prompt_tokens,
        "question_id": source_plan["question_id"],
        "question_sha256": source_plan["question_sha256"],
        "route_id": source_plan["route_id"],
        "source_prompt_plan": dict(source_plan),
        "source_prompt_row_receipt_sha256": source_plan[
            "prompt_row_receipt_sha256"
        ],
        "source_provider_input": dict(provider_input),
        "source_provider_input_sha256": identity_sha256(provider_input),
    }
    body["prompt_row_receipt_sha256"] = identity_sha256(body)
    assert_gold_blind(body, path="typed_fact_compiler_prompt_row")
    return body


def _compiler_preflight_projection(
    composition: SealedArtifact,
    source_preflight: SealedArtifact,
    composition_rows: Sequence[Mapping[str, Any]],
    source_plans: Sequence[Mapping[str, Any]],
    *,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> tuple[dict[str, Any], tuple[tuple[dict[str, str], ...], ...]]:
    require_text(model, "compiler model")
    require_text(gateway_url, "compiler gateway")
    _require(
        type(max_concurrency) is int and max_concurrency > 0,
        "compiler concurrency changed",
    )
    rows = tuple(
        _compiler_prompt_row(composition_row, source_plan)
        for composition_row, source_plan in zip(
            composition_rows, source_plans, strict=True
        )
    )
    prompts, population = _prompt_population(
        rows,
        output_reserve=COMPILER_OUTPUT_TOKEN_RESERVE,
        label="compiler",
    )
    payload = {
        "format": COMPILER_PREFLIGHT_FORMAT,
        "gateway_url": gateway_url,
        "gold_loaded": False,
        "hard_prompt_token_cap": HARD_PROMPT_TOKEN_CAP,
        "max_chat_prompt_tokens": (
            HARD_PROMPT_TOKEN_CAP - COMPILER_OUTPUT_TOKEN_RESERVE
        ),
        "max_concurrency": max_concurrency,
        "model": model,
        "observed_max_complete_envelope_tokens": max(
            row["prompt_token_proxy"] + COMPILER_OUTPUT_TOKEN_RESERVE
            for row in rows
        ),
        "original_ordinals": list(REMAINING_ORDINALS),
        "output_token_reserve": COMPILER_OUTPUT_TOKEN_RESERVE,
        "physical_prompt_rows": list(rows),
        "prompt_population": population.model_dump(),
        "prompt_population_sha256": population.prompt_population_sha256,
        "provider_calls": 0,
        "question_count": SUBSET_QUESTION_COUNT,
        "required_authorized_provider_calls": SUBSET_QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "selection_authority_loaded_at_runtime": False,
        "selection_is_posthoc_outcome_conditioned": True,
        "selection_rule": (
            "fixed unresolved ordinals after the sealed v3 miss27 treatment"
        ),
        "source_composition_artifact_sha256": composition.sha256,
        "source_preflight_artifact_sha256": source_preflight.sha256,
        "source_population_question_count": 100,
    }
    assert_gold_blind(payload, path="typed_fact_compiler_preflight")
    return payload, prompts


def _compiler_preflight(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    source_root = Path(args.source_root)
    _require(
        output_root.resolve() != source_root.resolve(),
        "fact compiler output root must differ from source root",
    )
    composition, source_preflight, composition_rows, source_plans = (
        _read_source_population(
            source_root,
            expected_composition_sha256=args.expected_source_composition_sha256,
            expected_preflight_sha256=args.expected_source_preflight_sha256,
        )
    )
    payload, _prompts = _compiler_preflight_projection(
        composition,
        source_preflight,
        composition_rows,
        source_plans,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    artifact, created = publish_sealed_json(
        output_root / COMPILER_PREFLIGHT_NAME,
        payload,
    )
    return {
        "compiler_preflight_sha256": artifact.sha256,
        "created": created,
        "gold_loaded": False,
        "maximum_complete_prompt_envelope": payload[
            "observed_max_complete_envelope_tokens"
        ],
        "original_ordinals": list(REMAINING_ORDINALS),
        "physical_provider_calls": 0,
        "question_count": SUBSET_QUESTION_COUNT,
        "required_authorized_provider_calls": SUBSET_QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "source_composition_sha256": composition.sha256,
        "source_preflight_sha256": source_preflight.sha256,
    }


def _validate_compiler_preflight(
    artifact: SealedArtifact,
) -> tuple[tuple[tuple[dict[str, str], ...], ...], tuple[dict[str, Any], ...]]:
    payload = artifact.payload
    assert_gold_blind(payload, path="typed_fact_compiler_runtime_preflight")
    rows = payload.get("physical_prompt_rows")
    _require(
        payload.get("format") == COMPILER_PREFLIGHT_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("hard_prompt_token_cap") == HARD_PROMPT_TOKEN_CAP
        and payload.get("max_chat_prompt_tokens")
        == HARD_PROMPT_TOKEN_CAP - COMPILER_OUTPUT_TOKEN_RESERVE
        and payload.get("output_token_reserve")
        == COMPILER_OUTPUT_TOKEN_RESERVE
        and payload.get("question_count") == SUBSET_QUESTION_COUNT
        and payload.get("required_authorized_provider_calls")
        == SUBSET_QUESTION_COUNT
        and payload.get("original_ordinals") == list(REMAINING_ORDINALS)
        and payload.get("selection_authority_loaded_at_runtime") is False
        and payload.get("selection_is_posthoc_outcome_conditioned") is True
        and type(rows) is list
        and len(rows) == SUBSET_QUESTION_COUNT
        and type(payload.get("model")) is str
        and bool(payload.get("model"))
        and type(payload.get("gateway_url")) is str
        and bool(payload.get("gateway_url"))
        and type(payload.get("max_concurrency")) is int
        and int(payload["max_concurrency"]) > 0,
        "compiler preflight firewall/population changed",
    )
    require_sha256(
        payload.get("source_composition_artifact_sha256"),
        "compiler source composition",
    )
    require_sha256(
        payload.get("source_preflight_artifact_sha256"),
        "compiler source preflight",
    )
    assert type(rows) is list
    prompts, population = _prompt_population(
        rows,
        output_reserve=COMPILER_OUTPUT_TOKEN_RESERVE,
        label="compiler",
    )
    _require(
        population.prompt_population_sha256
        == payload.get("prompt_population_sha256")
        and population.model_dump() == payload.get("prompt_population"),
        "compiler prompt population changed",
    )
    question_ids: list[str] = []
    validated: list[dict[str, Any]] = []
    for ordinal, raw in zip(REMAINING_ORDINALS, rows, strict=True):
        source_plan = raw.get("source_prompt_plan")
        provider_input = raw.get("source_provider_input")
        _require(
            type(source_plan) is dict
            and type(provider_input) is dict
            and source_plan.get("ordinal") == ordinal
            and source_plan.get("question_id") == raw.get("question_id")
            and source_plan.get("question_sha256") == raw.get("question_sha256")
            and source_plan.get("dated_question_sha256")
            == raw.get("dated_question_sha256")
            and source_plan.get("route_id") == raw.get("route_id")
            and source_plan.get("prompt_row_receipt_sha256")
            == raw.get("source_prompt_row_receipt_sha256")
            and identity_sha256(provider_input)
            == raw.get("source_provider_input_sha256"),
            f"compiler source bindings changed at ordinal {ordinal}",
        )
        assert type(source_plan) is dict
        source_body = dict(source_plan)
        source_receipt = source_body.pop("prompt_row_receipt_sha256", None)
        source_messages = _plain_messages(
            source_plan.get("messages"), label="compiler source"
        )
        _require(
            source_receipt == identity_sha256(source_body)
            and source_plan.get("messages_sha256")
            == identity_sha256(list(source_messages)),
            f"compiler original prompt binding changed at ordinal {ordinal}",
        )
        question_ids.append(require_text(raw.get("question_id"), "compiler question"))
        validated.append(dict(raw))
    _require(
        len(set(question_ids)) == SUBSET_QUESTION_COUNT,
        "compiler question identities repeat",
    )
    return prompts, tuple(validated)


def _read_phase_preflight(
    output_root: Path,
    *,
    name: str,
    expected_sha256: str,
    validator: Callable[
        [SealedArtifact],
        tuple[tuple[tuple[dict[str, str], ...], ...], tuple[dict[str, Any], ...]],
    ],
) -> tuple[
    SealedArtifact,
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
]:
    artifact = read_sealed_json(output_root / name)
    _require(
        artifact.sha256 == require_sha256(expected_sha256, f"expected {name}"),
        f"sealed preflight changed: {name}",
    )
    prompts, rows = validator(artifact)
    return artifact, prompts, rows


def _runtime(
    artifact: SealedArtifact,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    output_root: Path,
    checkpoint_dir_name: str,
    run_format: str,
    phase: str,
    client: Any | None,
) -> FastCompletionRuntime:
    payload = artifact.payload
    return FastCompletionRuntime(
        checkpoint_dir=output_root / checkpoint_dir_name,
        prompt_population=prompts,
        model=payload["model"],
        client=client,
        max_prompt_tokens=payload["max_chat_prompt_tokens"],
        max_new_tokens=payload["output_token_reserve"],
        max_concurrency=payload["max_concurrency"],
        retries=0,
        benchmark_provenance={
            "arm": "locked_typed_fact_compiler_remaining24_v2",
            "authorized_unique_calls": SUBSET_QUESTION_COUNT,
            "experiment_format": run_format,
            "gateway_url": payload["gateway_url"],
            "gold_loaded": False,
            "phase": phase,
            "preflight_artifact_sha256": artifact.sha256,
            "source_composition_artifact_sha256": payload[
                "source_composition_artifact_sha256"
            ],
            "source_preflight_artifact_sha256": payload[
                "source_preflight_artifact_sha256"
            ],
        },
    )


def _checkpoint_batch(
    artifact: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    output_root: Path,
    checkpoint_dir_name: str,
    run_format: str,
    phase: str,
    client: Any | None,
) -> FastCompletionBatch:
    runtime = _runtime(
        artifact,
        prompts,
        output_root=output_root,
        checkpoint_dir_name=checkpoint_dir_name,
        run_format=run_format,
        phase=phase,
        client=client,
    )
    try:
        return runtime.run()
    finally:
        runtime.close()


def _provider_phase(
    args: argparse.Namespace,
    *,
    preflight_name: str,
    validator: Callable[
        [SealedArtifact],
        tuple[tuple[tuple[dict[str, str], ...], ...], tuple[dict[str, Any], ...]],
    ],
    checkpoint_dir_name: str,
    run_format: str,
    phase: str,
) -> dict[str, Any]:
    output_root = Path(args.output_root)
    artifact, prompts, _rows = _read_phase_preflight(
        output_root,
        name=preflight_name,
        expected_sha256=args.expected_preflight_sha256,
        validator=validator,
    )
    _require(
        args.enable_provider is True
        and args.authorized_provider_calls == SUBSET_QUESTION_COUNT,
        f"{phase} provider-run requires exact authorization for 24 calls",
    )
    # Exact authorization and the immutable prompt population are checked
    # before environment access, client construction, or journal mutation.
    load_dotenv()
    api_key = os.environ.get(str(args.api_key_env), "").strip()
    _require(bool(api_key), f"provider API key is empty: {args.api_key_env}")
    client = live._make_provider_client(  # noqa: SLF001
        api_key,
        artifact.payload["gateway_url"],
    )
    try:
        batch = _checkpoint_batch(
            artifact,
            prompts,
            output_root=output_root,
            checkpoint_dir_name=checkpoint_dir_name,
            run_format=run_format,
            phase=phase,
            client=client,
        )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    _require(
        batch.usage.logical_calls
        == batch.usage.unique_calls
        == SUBSET_QUESTION_COUNT,
        f"{phase} provider population changed",
    )
    return {
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "gold_loaded": False,
        "phase": phase,
        "physical_provider_calls": batch.usage.physical_calls,
        "preflight_sha256": artifact.sha256,
        "required_authorized_provider_calls": SUBSET_QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
    }


def _compiler_provider(args: argparse.Namespace) -> dict[str, Any]:
    return _provider_phase(
        args,
        preflight_name=COMPILER_PREFLIGHT_NAME,
        validator=_validate_compiler_preflight,
        checkpoint_dir_name=COMPILER_CHECKPOINT_DIR_NAME,
        run_format=COMPILER_RUN_FORMAT,
        phase="compiler",
    )


def _require_checkpoint_only_batch(
    batch: FastCompletionBatch,
    *,
    label: str,
) -> None:
    _require(
        batch.usage.logical_calls
        == batch.usage.unique_calls
        == batch.usage.checkpoint_hits
        == SUBSET_QUESTION_COUNT
        and batch.usage.physical_calls == 0
        and len(batch.logical_completions) == SUBSET_QUESTION_COUNT
        and len(batch.unique_records) == SUBSET_QUESTION_COUNT,
        f"{label} requires 24 immutable checkpoint hits",
    )


def _record_by_messages(
    batch: FastCompletionBatch, *, label: str
) -> dict[str, Any]:
    records = {row.messages_sha256: row for row in batch.unique_records}
    _require(
        len(records) == SUBSET_QUESTION_COUNT,
        f"{label} completion identities repeat",
    )
    return records


def _parse_compilation(
    prompt_row: Mapping[str, Any], completion: str
) -> tuple[object, dict[str, Any], object, dict[str, Any]]:
    source_input = prompt_row.get("source_provider_input")
    _require(type(source_input) is dict, "compiler source input changed")
    assert type(source_input) is dict
    compilation = parse_typed_fact_compiler_response(source_input, completion)
    compilation_projection = _projection(compilation, label="fact compilation")
    packet = getattr(compilation, "packet", None)
    _require(packet is not None, "fact compilation packet missing")
    packet_projection = _projection(packet, label="fact packet")
    assert_gold_blind(
        compilation_projection,
        path="typed_fact_compilation_projection",
    )
    assert_gold_blind(packet_projection, path="typed_fact_packet_projection")
    return compilation, compilation_projection, packet, packet_projection


def _compiler_materialization_projection(
    preflight: SealedArtifact,
    prompt_rows: tuple[dict[str, Any], ...],
    batch: FastCompletionBatch,
) -> dict[str, Any]:
    _require_checkpoint_only_batch(batch, label="compiler materialization")
    records = _record_by_messages(batch, label="compiler")
    results: list[dict[str, Any]] = []
    for ordinal, prompt, completion in zip(
        REMAINING_ORDINALS,
        prompt_rows,
        batch.logical_completions,
        strict=True,
    ):
        record = records.get(prompt["messages_sha256"])
        _require(
            record is not None
            and record.completion == completion
            and record.checkpoint_hit is True
            and record.physical_call is False,
            f"compiler checkpoint record changed at ordinal {ordinal}",
        )
        assert record is not None
        _compilation, compilation_projection, _packet, packet_projection = (
            _parse_compilation(prompt, completion)
        )
        body = {
            "call_key_sha256": record.call_key_sha256,
            "compiler_completion": completion,
            "compiler_completion_sha256": record.completion_sha256,
            "compiler_prompt_row_receipt_sha256": prompt[
                "prompt_row_receipt_sha256"
            ],
            "compilation": compilation_projection,
            "composition_row_sha256": prompt["composition_row_sha256"],
            "dated_question_sha256": prompt["dated_question_sha256"],
            "fact_packet": packet_projection,
            "fact_packet_sha256": identity_sha256(packet_projection),
            "ordinal": ordinal,
            "question_id": prompt["question_id"],
            "question_sha256": prompt["question_sha256"],
            "request_journal_sha256": record.request_journal_sha256,
            "response_journal_sha256": record.response_journal_sha256,
            "retained_transformer_token_state_bytes": 0,
            "route_id": prompt["route_id"],
            "source_prompt_row_receipt_sha256": prompt[
                "source_prompt_row_receipt_sha256"
            ],
        }
        body["compiler_result_row_sha256"] = identity_sha256(body)
        results.append(body)
    source = preflight.payload
    payload = {
        "completion_batch": batch.model_dump(),
        "format": COMPILER_RUN_FORMAT,
        "gold_loaded": False,
        "original_ordinals": list(REMAINING_ORDINALS),
        "physical_provider_calls_during_materialization": 0,
        "questions": results,
        "question_count": SUBSET_QUESTION_COUNT,
        "required_authorized_provider_calls": SUBSET_QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "selection_is_posthoc_outcome_conditioned": True,
        "source_composition_artifact_sha256": source[
            "source_composition_artifact_sha256"
        ],
        "source_preflight_artifact_sha256": source[
            "source_preflight_artifact_sha256"
        ],
        "compiler_preflight_artifact_sha256": preflight.sha256,
    }
    assert_gold_blind(payload, path="typed_fact_compiler_run")
    return payload


def _compiler_materialize(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    preflight, prompts, rows = _read_phase_preflight(
        output_root,
        name=COMPILER_PREFLIGHT_NAME,
        expected_sha256=args.expected_preflight_sha256,
        validator=_validate_compiler_preflight,
    )
    batch = _checkpoint_batch(
        preflight,
        prompts,
        output_root=output_root,
        checkpoint_dir_name=COMPILER_CHECKPOINT_DIR_NAME,
        run_format=COMPILER_RUN_FORMAT,
        phase="compiler",
        client=None,
    )
    payload = _compiler_materialization_projection(preflight, rows, batch)
    artifact, created = publish_sealed_json(
        output_root / COMPILER_RUN_NAME,
        payload,
    )
    return {
        "compiler_run_sha256": artifact.sha256,
        "created": created,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "question_count": SUBSET_QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
    }


def _validate_compiler_run(
    artifact: SealedArtifact,
    preflight: SealedArtifact,
    prompt_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    payload = artifact.payload
    assert_gold_blind(payload, path="verified_typed_fact_compiler_run")
    questions = payload.get("questions")
    _require(
        payload.get("format") == COMPILER_RUN_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("physical_provider_calls_during_materialization") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("question_count") == SUBSET_QUESTION_COUNT
        and payload.get("original_ordinals") == list(REMAINING_ORDINALS)
        and payload.get("selection_is_posthoc_outcome_conditioned") is True
        and payload.get("compiler_preflight_artifact_sha256") == preflight.sha256
        and payload.get("source_composition_artifact_sha256")
        == preflight.payload["source_composition_artifact_sha256"]
        and payload.get("source_preflight_artifact_sha256")
        == preflight.payload["source_preflight_artifact_sha256"]
        and type(questions) is list
        and len(questions) == SUBSET_QUESTION_COUNT,
        "compiler run envelope changed",
    )
    verified: list[dict[str, Any]] = []
    assert type(questions) is list
    for ordinal, raw, prompt in zip(
        REMAINING_ORDINALS,
        questions,
        prompt_rows,
        strict=True,
    ):
        _require(type(raw) is dict, "compiler result row changed type")
        assert type(raw) is dict
        unsigned = dict(raw)
        declared = unsigned.pop("compiler_result_row_sha256", None)
        completion = raw.get("compiler_completion")
        _require(
            declared == identity_sha256(unsigned)
            and raw.get("ordinal") == ordinal
            and raw.get("question_id") == prompt.get("question_id")
            and raw.get("question_sha256") == prompt.get("question_sha256")
            and raw.get("dated_question_sha256")
            == prompt.get("dated_question_sha256")
            and raw.get("composition_row_sha256")
            == prompt.get("composition_row_sha256")
            and raw.get("compiler_prompt_row_receipt_sha256")
            == prompt.get("prompt_row_receipt_sha256")
            and raw.get("source_prompt_row_receipt_sha256")
            == prompt.get("source_prompt_row_receipt_sha256")
            and type(completion) is str
            and bool(completion)
            and quote_sha256(completion)
            == raw.get("compiler_completion_sha256"),
            f"compiler result binding changed at ordinal {ordinal}",
        )
        for key in (
            "call_key_sha256",
            "compiler_completion_sha256",
            "compiler_prompt_row_receipt_sha256",
            "compiler_result_row_sha256",
            "composition_row_sha256",
            "dated_question_sha256",
            "fact_packet_sha256",
            "question_sha256",
            "request_journal_sha256",
            "response_journal_sha256",
            "source_prompt_row_receipt_sha256",
        ):
            require_sha256(raw.get(key), f"compiler result {key}")
        _compilation, compilation_projection, _packet, packet_projection = (
            _parse_compilation(prompt, completion)
        )
        _require(
            raw.get("compilation") == compilation_projection
            and raw.get("fact_packet") == packet_projection
            and raw.get("fact_packet_sha256")
            == identity_sha256(packet_projection),
            f"compiler parsed packet changed at ordinal {ordinal}",
        )
        verified.append(dict(raw))
    return tuple(verified)


def _compiler_replay(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    preflight, prompts, rows = _read_phase_preflight(
        output_root,
        name=COMPILER_PREFLIGHT_NAME,
        expected_sha256=args.expected_preflight_sha256,
        validator=_validate_compiler_preflight,
    )
    batch = _checkpoint_batch(
        preflight,
        prompts,
        output_root=output_root,
        checkpoint_dir_name=COMPILER_CHECKPOINT_DIR_NAME,
        run_format=COMPILER_RUN_FORMAT,
        phase="compiler",
        client=None,
    )
    replayed = _compiler_materialization_projection(preflight, rows, batch)
    run = read_sealed_json(output_root / COMPILER_RUN_NAME)
    _require(
        run.sha256
        == require_sha256(args.expected_run_sha256, "expected compiler run")
        and run.payload == replayed,
        "compiler run differs from checkpoint-only replay",
    )
    _validate_compiler_run(run, preflight, rows)
    payload = {
        "byte_identical": True,
        "compiler_preflight_artifact_sha256": preflight.sha256,
        "expected_run_sha256": run.sha256,
        "format": COMPILER_REPLAY_FORMAT,
        "gold_loaded": False,
        "original_ordinals": list(REMAINING_ORDINALS),
        "physical_provider_calls": 0,
        "question_count": SUBSET_QUESTION_COUNT,
        "replayed_run_sha256": run.sha256,
        "retained_transformer_token_state_bytes": 0,
    }
    assert_gold_blind(payload, path="typed_fact_compiler_replay")
    artifact, created = publish_sealed_json(
        output_root / COMPILER_REPLAY_NAME,
        payload,
    )
    return {
        "byte_identical": True,
        "compiler_replay_sha256": artifact.sha256,
        "compiler_run_sha256": run.sha256,
        "created": created,
        "physical_provider_calls": 0,
    }


def _read_verified_compiler_run(
    output_root: Path,
    *,
    expected_preflight_sha256: str,
    expected_run_sha256: str,
    expected_replay_sha256: str,
) -> tuple[
    SealedArtifact,
    SealedArtifact,
    SealedArtifact,
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    preflight, _prompts, prompt_rows = _read_phase_preflight(
        output_root,
        name=COMPILER_PREFLIGHT_NAME,
        expected_sha256=expected_preflight_sha256,
        validator=_validate_compiler_preflight,
    )
    run = read_sealed_json(output_root / COMPILER_RUN_NAME)
    _require(
        run.sha256 == require_sha256(expected_run_sha256, "expected compiler run"),
        "compiler run SHA-256 changed",
    )
    result_rows = _validate_compiler_run(run, preflight, prompt_rows)
    replay = read_sealed_json(output_root / COMPILER_REPLAY_NAME)
    replay_payload = replay.payload
    _require(
        replay.sha256
        == require_sha256(expected_replay_sha256, "expected compiler replay")
        and replay_payload.get("format") == COMPILER_REPLAY_FORMAT
        and replay_payload.get("byte_identical") is True
        and replay_payload.get("gold_loaded") is False
        and replay_payload.get("physical_provider_calls") == 0
        and replay_payload.get("retained_transformer_token_state_bytes") == 0
        and replay_payload.get("question_count") == SUBSET_QUESTION_COUNT
        and replay_payload.get("original_ordinals") == list(REMAINING_ORDINALS)
        and replay_payload.get("compiler_preflight_artifact_sha256")
        == preflight.sha256
        and replay_payload.get("expected_run_sha256") == run.sha256
        and replay_payload.get("replayed_run_sha256") == run.sha256,
        "compiler replay binding changed",
    )
    assert_gold_blind(replay_payload, path="verified_typed_fact_compiler_replay")
    return preflight, run, replay, prompt_rows, result_rows


def _narrow_answer_authority(
    source_plan: Mapping[str, Any],
    source_input: Mapping[str, Any],
    retained_handle_ids: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Restrict one valid answer plane to handles retained by its fact packet."""

    _require(
        type(retained_handle_ids) is tuple
        and bool(retained_handle_ids)
        and len(set(retained_handle_ids)) == len(retained_handle_ids)
        and all(type(value) is str and bool(value) for value in retained_handle_ids),
        "compiled retained handle authority changed",
    )
    original_allowed = source_plan.get("allowed_handle_ids")
    original_groups = source_plan.get("handle_group_by_id")
    preservation = source_plan.get("preservation_requirements")
    validation = source_plan.get("validation_contract")
    response_schema = source_input.get("response_schema")
    _require(
        type(original_allowed) is list
        and len(set(original_allowed)) == len(original_allowed)
        and type(original_groups) is dict
        and set(original_groups) == set(original_allowed)
        and set(retained_handle_ids) <= set(original_allowed)
        and type(preservation) is dict
        and type(preservation.get("by_handle")) is dict
        and set(preservation["by_handle"]) <= set(original_allowed)
        and type(validation) is dict
        and type(validation.get("by_handle")) is dict
        and set(validation["by_handle"]) == set(original_allowed)
        and type(response_schema) is dict,
        "original answer handle authority changed",
    )
    assert (
        type(original_groups) is dict
        and type(preservation) is dict
        and type(validation) is dict
        and type(response_schema) is dict
    )
    retained = set(retained_handle_ids)

    narrowed_preservation = dict(preservation)
    narrowed_preservation["by_handle"] = {
        handle: dict(value)
        for handle, value in preservation["by_handle"].items()
        if handle in retained
    }
    narrowed_validation = dict(validation)
    narrowed_validation["by_handle"] = {
        handle: dict(validation["by_handle"][handle])
        for handle in retained_handle_ids
    }
    # A deterministic/scalar advisory cannot be partially filtered without
    # changing its proof.  Retain it only when its complete cited authority is
    # present in the compiled packet; otherwise fall back to ordinary typed
    # entailment over the narrowed by-handle contract.
    for key in (
        "deterministic_execution_advisory",
        "scalar_validation_advisory",
    ):
        advisory = narrowed_validation.get(key)
        if advisory is None:
            continue
        used = advisory.get("used_handle_ids") if type(advisory) is dict else None
        _require(
            type(used) is list
            and len(set(used)) == len(used)
            and all(type(value) is str and bool(value) for value in used),
            f"original {key} handle authority changed",
        )
        if not set(used) <= retained:
            narrowed_validation[key] = None

    narrowed_plan = dict(source_plan)
    narrowed_plan.update(
        {
            "allowed_handle_ids": list(retained_handle_ids),
            "handle_group_by_id": {
                handle: original_groups[handle] for handle in retained_handle_ids
            },
            "preservation_requirements": narrowed_preservation,
            "validation_contract": narrowed_validation,
        }
    )
    narrowed_schema = dict(response_schema)
    narrowed_schema["used_handle_ids"] = list(retained_handle_ids)
    narrowed_input = dict(source_input)
    narrowed_input["response_schema"] = narrowed_schema
    return narrowed_plan, narrowed_input


def _answer_prompt_row(
    compiler_prompt: Mapping[str, Any],
    compiler_result: Mapping[str, Any],
) -> dict[str, Any]:
    completion = compiler_result.get("compiler_completion")
    _require(type(completion) is str, "compiler completion changed type")
    assert type(completion) is str
    compilation, compilation_projection, packet, packet_projection = (
        _parse_compilation(compiler_prompt, completion)
    )
    _require(
        compilation_projection == compiler_result.get("compilation")
        and packet_projection == compiler_result.get("fact_packet")
        and identity_sha256(packet_projection)
        == compiler_result.get("fact_packet_sha256"),
        "answer preflight compiler packet changed",
    )
    source_plan = compiler_prompt.get("source_prompt_plan")
    source_input = compiler_prompt.get("source_provider_input")
    _require(
        type(source_plan) is dict and type(source_input) is dict,
        "answer source plan/input changed",
    )
    assert type(source_plan) is dict and type(source_input) is dict
    packet_valid = getattr(packet, "valid", None)
    facts = getattr(packet, "facts", None)
    _require(
        type(packet_valid) is bool
        and type(facts) is tuple
        and (not packet_valid or bool(facts)),
        "fact packet validity/facts changed",
    )
    if packet_valid:
        retained_handle_ids = getattr(packet, "retained_handle_ids", None)
        _require(
            type(retained_handle_ids) is tuple,
            "fact packet retained handle authority changed",
        )
        assert type(retained_handle_ids) is tuple
        answer_plan, answer_input = _narrow_answer_authority(
            source_plan,
            source_input,
            retained_handle_ids,
        )
        messages = _plain_messages(
            build_answer_messages(answer_input, packet),
            label="fact answer",
        )
        decoded = _decoded_user_payloads(messages, label="fact answer")
        decoded_schema = (
            decoded[0].get("response_schema")
            if len(decoded) == 1 and type(decoded[0]) is dict
            else None
        )
        _require(
            len(decoded) == 1
            and type(decoded[0]) is dict
            and type(decoded_schema) is dict
            and decoded_schema.get("used_handle_ids")
            == list(retained_handle_ids),
            "fact answer provider schema escaped compiled handle authority",
        )
        byte_identical_fallback = False
    else:
        retained_handle_ids = ()
        answer_plan = dict(source_plan)
        messages = _plain_messages(
            source_plan.get("messages"),
            label="fact answer fallback",
        )
        # Invalid or empty compilation must be exactly the already-sealed v3
        # provider request, not a newly rendered approximation.
        _require(
            list(messages) == source_plan.get("messages")
            and identity_sha256(list(messages))
            == source_plan.get("messages_sha256"),
            "invalid fact compilation fallback is not byte-identical",
        )
        byte_identical_fallback = True
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    _require(
        prompt_tokens + ANSWER_OUTPUT_TOKEN_RESERVE <= HARD_PROMPT_TOKEN_CAP,
        "fact answer complete prompt envelope exceeds 8k",
    )
    body = dict(answer_plan)
    body.pop("prompt_row_receipt_sha256", None)
    body.update(
        {
            "byte_identical_source_fallback": byte_identical_fallback,
            "compiled_handle_authority_enforced": packet_valid,
            "compiled_retained_handle_ids": list(retained_handle_ids),
            "compiler_preflight_prompt_row_receipt_sha256": compiler_prompt[
                "prompt_row_receipt_sha256"
            ],
            "compiler_result_row_sha256": compiler_result[
                "compiler_result_row_sha256"
            ],
            "compilation_receipt_sha256": getattr(
                compilation, "receipt_sha256"
            ),
            "fact_count": len(facts),
            "fact_packet": packet_projection,
            "fact_packet_receipt_sha256": getattr(packet, "receipt_sha256"),
            "fact_packet_valid": packet_valid,
            "messages": list(messages),
            "messages_sha256": identity_sha256(list(messages)),
            "prompt_token_proxy": prompt_tokens,
            "source_prompt_plan": dict(source_plan),
            "source_prompt_row_receipt_sha256": source_plan[
                "prompt_row_receipt_sha256"
            ],
        }
    )
    body["prompt_row_receipt_sha256"] = identity_sha256(body)
    assert_gold_blind(body, path="typed_fact_answer_prompt_row")
    return body


def _answer_preflight_projection(
    compiler_preflight: SealedArtifact,
    compiler_run: SealedArtifact,
    compiler_replay: SealedArtifact,
    compiler_prompt_rows: Sequence[Mapping[str, Any]],
    compiler_result_rows: Sequence[Mapping[str, Any]],
    *,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> tuple[dict[str, Any], tuple[tuple[dict[str, str], ...], ...]]:
    require_text(model, "fact answer model")
    require_text(gateway_url, "fact answer gateway")
    _require(
        type(max_concurrency) is int and max_concurrency > 0,
        "fact answer concurrency changed",
    )
    rows = tuple(
        _answer_prompt_row(prompt, result)
        for prompt, result in zip(
            compiler_prompt_rows,
            compiler_result_rows,
            strict=True,
        )
    )
    prompts, population = _prompt_population(
        rows,
        output_reserve=ANSWER_OUTPUT_TOKEN_RESERVE,
        label="fact answer",
    )
    payload = {
        "byte_identical_invalid_fallback_count": sum(
            bool(row["byte_identical_source_fallback"]) for row in rows
        ),
        "compiler_preflight_artifact_sha256": compiler_preflight.sha256,
        "compiler_replay_artifact_sha256": compiler_replay.sha256,
        "compiler_run_artifact_sha256": compiler_run.sha256,
        "format": ANSWER_PREFLIGHT_FORMAT,
        "gateway_url": gateway_url,
        "gold_loaded": False,
        "hard_prompt_token_cap": HARD_PROMPT_TOKEN_CAP,
        "max_chat_prompt_tokens": HARD_PROMPT_TOKEN_CAP - ANSWER_OUTPUT_TOKEN_RESERVE,
        "max_concurrency": max_concurrency,
        "model": model,
        "observed_max_complete_envelope_tokens": max(
            row["prompt_token_proxy"] + ANSWER_OUTPUT_TOKEN_RESERVE
            for row in rows
        ),
        "original_ordinals": list(REMAINING_ORDINALS),
        "output_token_reserve": ANSWER_OUTPUT_TOKEN_RESERVE,
        "physical_prompt_rows": list(rows),
        "prompt_population": population.model_dump(),
        "prompt_population_sha256": population.prompt_population_sha256,
        "provider_calls": 0,
        "question_count": SUBSET_QUESTION_COUNT,
        "required_authorized_provider_calls": SUBSET_QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "selection_is_posthoc_outcome_conditioned": True,
        "source_composition_artifact_sha256": compiler_preflight.payload[
            "source_composition_artifact_sha256"
        ],
        "source_preflight_artifact_sha256": compiler_preflight.payload[
            "source_preflight_artifact_sha256"
        ],
    }
    assert_gold_blind(payload, path="typed_fact_answer_preflight")
    return payload, prompts


def _answer_preflight(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    compiler_preflight, compiler_run, compiler_replay, prompt_rows, result_rows = (
        _read_verified_compiler_run(
            output_root,
            expected_preflight_sha256=args.expected_compiler_preflight_sha256,
            expected_run_sha256=args.expected_compiler_run_sha256,
            expected_replay_sha256=args.expected_compiler_replay_sha256,
        )
    )
    payload, _prompts = _answer_preflight_projection(
        compiler_preflight,
        compiler_run,
        compiler_replay,
        prompt_rows,
        result_rows,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    artifact, created = publish_sealed_json(
        output_root / ANSWER_PREFLIGHT_NAME,
        payload,
    )
    return {
        "answer_preflight_sha256": artifact.sha256,
        "byte_identical_invalid_fallback_count": payload[
            "byte_identical_invalid_fallback_count"
        ],
        "compiler_run_sha256": compiler_run.sha256,
        "created": created,
        "gold_loaded": False,
        "maximum_complete_prompt_envelope": payload[
            "observed_max_complete_envelope_tokens"
        ],
        "original_ordinals": list(REMAINING_ORDINALS),
        "physical_provider_calls": 0,
        "question_count": SUBSET_QUESTION_COUNT,
        "required_authorized_provider_calls": SUBSET_QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
    }


def _validate_answer_preflight(
    artifact: SealedArtifact,
) -> tuple[tuple[tuple[dict[str, str], ...], ...], tuple[dict[str, Any], ...]]:
    payload = artifact.payload
    assert_gold_blind(payload, path="typed_fact_answer_runtime_preflight")
    rows = payload.get("physical_prompt_rows")
    _require(
        payload.get("format") == ANSWER_PREFLIGHT_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("hard_prompt_token_cap") == HARD_PROMPT_TOKEN_CAP
        and payload.get("max_chat_prompt_tokens")
        == HARD_PROMPT_TOKEN_CAP - ANSWER_OUTPUT_TOKEN_RESERVE
        and payload.get("output_token_reserve") == ANSWER_OUTPUT_TOKEN_RESERVE
        and payload.get("question_count") == SUBSET_QUESTION_COUNT
        and payload.get("required_authorized_provider_calls")
        == SUBSET_QUESTION_COUNT
        and payload.get("original_ordinals") == list(REMAINING_ORDINALS)
        and payload.get("selection_is_posthoc_outcome_conditioned") is True
        and type(rows) is list
        and len(rows) == SUBSET_QUESTION_COUNT
        and type(payload.get("model")) is str
        and bool(payload.get("model"))
        and type(payload.get("gateway_url")) is str
        and bool(payload.get("gateway_url"))
        and type(payload.get("max_concurrency")) is int
        and int(payload["max_concurrency"]) > 0,
        "fact answer preflight firewall/population changed",
    )
    for key in (
        "compiler_preflight_artifact_sha256",
        "compiler_replay_artifact_sha256",
        "compiler_run_artifact_sha256",
        "source_composition_artifact_sha256",
        "source_preflight_artifact_sha256",
    ):
        require_sha256(payload.get(key), f"fact answer {key}")
    assert type(rows) is list
    prompts, population = _prompt_population(
        rows,
        output_reserve=ANSWER_OUTPUT_TOKEN_RESERVE,
        label="fact answer",
    )
    _require(
        population.prompt_population_sha256
        == payload.get("prompt_population_sha256")
        and population.model_dump() == payload.get("prompt_population"),
        "fact answer prompt population changed",
    )
    validated: list[dict[str, Any]] = []
    fallback_count = 0
    for ordinal, raw in zip(REMAINING_ORDINALS, rows, strict=True):
        source_plan = raw.get("source_prompt_plan")
        packet = raw.get("fact_packet")
        _require(
            type(source_plan) is dict
            and type(packet) is dict
            and raw.get("ordinal") == ordinal
            and source_plan.get("ordinal") == ordinal
            and raw.get("source_prompt_row_receipt_sha256")
            == source_plan.get("prompt_row_receipt_sha256")
            and raw.get("fact_packet_receipt_sha256")
            == packet.get("receipt_sha256")
            and raw.get("fact_packet_valid") == packet.get("valid")
            and raw.get("fact_count")
            == len(packet.get("typed_evidence", {}).get("items", [])),
            f"fact answer source/packet binding changed at ordinal {ordinal}",
        )
        if raw.get("fact_packet_valid") is False:
            fallback_count += 1
            _require(
                raw.get("byte_identical_source_fallback") is True
                and raw.get("compiled_handle_authority_enforced") is False
                and raw.get("compiled_retained_handle_ids") == []
                and raw.get("messages") == source_plan.get("messages")
                and raw.get("messages_sha256")
                == source_plan.get("messages_sha256"),
                f"invalid compiler fallback changed at ordinal {ordinal}",
            )
        else:
            retained = packet.get("retained_handle_ids")
            handle_groups = raw.get("handle_group_by_id")
            preservation = raw.get("preservation_requirements")
            validation = raw.get("validation_contract")
            typed_evidence = packet.get("typed_evidence")
            packet_handles = (
                typed_evidence.get("handles")
                if type(typed_evidence) is dict
                else None
            )
            decoded = _decoded_user_payloads(
                _plain_messages(raw.get("messages"), label="fact answer"),
                label="fact answer",
            )
            decoded_schema = (
                decoded[0].get("response_schema")
                if len(decoded) == 1 and type(decoded[0]) is dict
                else None
            )
            _require(
                raw.get("fact_packet_valid") is True
                and raw.get("byte_identical_source_fallback") is False
                and raw.get("compiled_handle_authority_enforced") is True
                and type(retained) is list
                and bool(retained)
                and len(set(retained)) == len(retained)
                and raw.get("compiled_retained_handle_ids") == retained
                and raw.get("allowed_handle_ids") == retained
                and type(handle_groups) is dict
                and set(handle_groups) == set(retained)
                and type(preservation) is dict
                and type(preservation.get("by_handle")) is dict
                and set(preservation["by_handle"]) <= set(retained)
                and type(validation) is dict
                and type(validation.get("by_handle")) is dict
                and set(validation["by_handle"]) == set(retained)
                and type(packet_handles) is list
                and {row.get("handle_id") for row in packet_handles}
                == set(retained)
                and type(decoded_schema) is dict
                and decoded_schema.get("used_handle_ids") == retained,
                f"valid compiler answer marker changed at ordinal {ordinal}",
            )
        validated.append(dict(raw))
    _require(
        fallback_count == payload.get("byte_identical_invalid_fallback_count"),
        "fact answer invalid fallback accounting changed",
    )
    return prompts, tuple(validated)


def _answer_provider(args: argparse.Namespace) -> dict[str, Any]:
    return _provider_phase(
        args,
        preflight_name=ANSWER_PREFLIGHT_NAME,
        validator=_validate_answer_preflight,
        checkpoint_dir_name=ANSWER_CHECKPOINT_DIR_NAME,
        run_format=ANSWER_RUN_FORMAT,
        phase="answer",
    )


def _declared_completion_handles(completion: str) -> tuple[str, ...] | None:
    try:
        raw = json.loads(
            completion,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (json.JSONDecodeError, ValueError):
        return None
    if type(raw) is not dict or type(raw.get("used_handle_ids")) is not list:
        return None
    values = raw["used_handle_ids"]
    if any(type(value) is not str for value in values):
        return None
    return tuple(values)


def _answer_materialization_projection(
    preflight: SealedArtifact,
    prompt_rows: tuple[dict[str, Any], ...],
    batch: FastCompletionBatch,
) -> dict[str, Any]:
    _require_checkpoint_only_batch(batch, label="fact answer materialization")
    records = _record_by_messages(batch, label="fact answer")
    results: list[dict[str, Any]] = []
    authority_rejections = 0
    for ordinal, plan, completion in zip(
        REMAINING_ORDINALS,
        prompt_rows,
        batch.logical_completions,
        strict=True,
    ):
        record = records.get(plan["messages_sha256"])
        _require(
            record is not None
            and record.completion == completion
            and record.checkpoint_hit is True
            and record.physical_call is False,
            f"fact answer checkpoint record changed at ordinal {ordinal}",
        )
        assert record is not None
        result = materialize_typed_final_result_row(
            plan,
            completion,
            completion_receipt_sha256=record.completion_sha256,
            call_key_sha256=record.call_key_sha256,
            request_journal_sha256=record.request_journal_sha256,
            response_journal_sha256=record.response_journal_sha256,
        )
        if plan.get("compiled_handle_authority_enforced") is True:
            retained = plan.get("compiled_retained_handle_ids")
            _require(
                type(retained) is list
                and bool(retained)
                and plan.get("allowed_handle_ids") == retained
                and set(result.get("used_handle_ids", ())) <= set(retained),
                f"materialized answer escaped compiled handles at ordinal {ordinal}",
            )
            declared = _declared_completion_handles(completion)
            if declared is not None and not set(declared) <= set(retained):
                authority_rejections += 1
                _require(
                    result.get("decision") == "invalid_keep_parent"
                    and result.get("parse_error_code") == "unknown_handle"
                    and result.get("used_handle_ids") == [],
                    f"outside-packet handle was not rejected at ordinal {ordinal}",
                )
        results.append(result)
    judge_rows = [judge_row_projection(row) for row in results]
    _require(
        tuple(row["ordinal"] for row in results) == REMAINING_ORDINALS
        and tuple(row["ordinal"] for row in judge_rows) == REMAINING_ORDINALS,
        "fact answer result ordinals changed",
    )
    source = preflight.payload
    payload = {
        "changed_prediction_count": sum(
            bool(row["changed_from_parent"]) for row in results
        ),
        "compiled_handle_authority_rejection_count": authority_rejections,
        "completion_batch": batch.model_dump(),
        "compiler_preflight_artifact_sha256": source[
            "compiler_preflight_artifact_sha256"
        ],
        "compiler_replay_artifact_sha256": source[
            "compiler_replay_artifact_sha256"
        ],
        "compiler_run_artifact_sha256": source["compiler_run_artifact_sha256"],
        "format": ANSWER_RUN_FORMAT,
        "gold_loaded": False,
        "invalid_completion_parent_fallback_count": sum(
            row["prediction_source"] == "typed_final_invalid_keep_parent_v1"
            for row in results
        ),
        "invalid_fact_packet_source_fallback_count": source[
            "byte_identical_invalid_fallback_count"
        ],
        "judge_rows": judge_rows,
        "original_ordinals": list(REMAINING_ORDINALS),
        "physical_provider_calls_during_materialization": 0,
        "questions": results,
        "question_count": SUBSET_QUESTION_COUNT,
        "required_authorized_provider_calls": SUBSET_QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "selection_is_posthoc_outcome_conditioned": True,
        "source_composition_artifact_sha256": source[
            "source_composition_artifact_sha256"
        ],
        "source_preflight_artifact_sha256": source[
            "source_preflight_artifact_sha256"
        ],
        "answer_preflight_artifact_sha256": preflight.sha256,
        "validator_policy_format": VALIDATOR_POLICY_FORMAT,
    }
    assert_gold_blind(payload, path="typed_fact_answer_run")
    return payload


def _answer_materialize(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    preflight, prompts, rows = _read_phase_preflight(
        output_root,
        name=ANSWER_PREFLIGHT_NAME,
        expected_sha256=args.expected_preflight_sha256,
        validator=_validate_answer_preflight,
    )
    batch = _checkpoint_batch(
        preflight,
        prompts,
        output_root=output_root,
        checkpoint_dir_name=ANSWER_CHECKPOINT_DIR_NAME,
        run_format=ANSWER_RUN_FORMAT,
        phase="answer",
        client=None,
    )
    payload = _answer_materialization_projection(preflight, rows, batch)
    artifact, created = publish_sealed_json(
        output_root / ANSWER_RUN_NAME,
        payload,
    )
    return {
        "answer_run_sha256": artifact.sha256,
        "changed_prediction_count": payload["changed_prediction_count"],
        "created": created,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "question_count": SUBSET_QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
    }


def _validate_answer_run(
    artifact: SealedArtifact,
    preflight: SealedArtifact,
    prompt_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    payload = artifact.payload
    assert_gold_blind(payload, path="verified_typed_fact_answer_run")
    questions = payload.get("questions")
    projected = payload.get("judge_rows")
    completion_batch = payload.get("completion_batch")
    logical_completions = (
        completion_batch.get("logical_completions")
        if type(completion_batch) is dict
        else None
    )
    _require(
        payload.get("format") == ANSWER_RUN_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("physical_provider_calls_during_materialization") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("question_count") == SUBSET_QUESTION_COUNT
        and payload.get("original_ordinals") == list(REMAINING_ORDINALS)
        and payload.get("selection_is_posthoc_outcome_conditioned") is True
        and payload.get("answer_preflight_artifact_sha256") == preflight.sha256
        and payload.get("compiler_preflight_artifact_sha256")
        == preflight.payload["compiler_preflight_artifact_sha256"]
        and payload.get("compiler_run_artifact_sha256")
        == preflight.payload["compiler_run_artifact_sha256"]
        and payload.get("compiler_replay_artifact_sha256")
        == preflight.payload["compiler_replay_artifact_sha256"]
        and payload.get("source_composition_artifact_sha256")
        == preflight.payload["source_composition_artifact_sha256"]
        and payload.get("source_preflight_artifact_sha256")
        == preflight.payload["source_preflight_artifact_sha256"]
        and type(questions) is list
        and type(projected) is list
        and type(logical_completions) is list
        and len(questions) == len(projected) == SUBSET_QUESTION_COUNT,
        "fact answer run envelope changed",
    )
    verified: list[dict[str, Any]] = []
    authority_rejections = 0
    assert (
        type(questions) is list
        and type(projected) is list
        and type(logical_completions) is list
    )
    for ordinal, source, judge, prompt, completion in zip(
        REMAINING_ORDINALS,
        questions,
        projected,
        prompt_rows,
        logical_completions,
        strict=True,
    ):
        _require(
            type(source) is dict and type(judge) is dict,
            "fact answer result row changed type",
        )
        assert type(source) is dict and type(judge) is dict
        unsigned = dict(source)
        declared = unsigned.pop("source_row_sha256", None)
        used = source.get("used_handle_ids")
        _require(
            declared == identity_sha256(unsigned)
            and source.get("ordinal") == ordinal
            and prompt.get("ordinal") == ordinal
            and source.get("question_id") == prompt.get("question_id")
            and source.get("question_sha256") == prompt.get("question_sha256")
            and source.get("prompt_row_receipt_sha256")
            == prompt.get("prompt_row_receipt_sha256")
            and type(used) is list
            and set(used) <= set(prompt.get("allowed_handle_ids", []))
            and (source.get("decision") == "replace" or not used)
            and judge_row_projection(source) == judge
            and judge.get("ordinal") == ordinal,
            f"fact answer result binding changed at ordinal {ordinal}",
        )
        if prompt.get("compiled_handle_authority_enforced") is True:
            retained = prompt.get("compiled_retained_handle_ids")
            _require(
                type(retained) is list
                and bool(retained)
                and prompt.get("allowed_handle_ids") == retained
                and set(source.get("used_handle_ids", ())) <= set(retained),
                f"verified answer escaped compiled handles at ordinal {ordinal}",
            )
            _require(
                type(completion) is str,
                f"verified answer completion changed at ordinal {ordinal}",
            )
            assert type(completion) is str
            declared = _declared_completion_handles(completion)
            if declared is not None and not set(declared) <= set(retained):
                authority_rejections += 1
                _require(
                    source.get("decision") == "invalid_keep_parent"
                    and source.get("parse_error_code") == "unknown_handle"
                    and source.get("used_handle_ids") == [],
                    f"verified outside-packet handle was accepted at ordinal {ordinal}",
                )
        verified.append(dict(judge))
    _require(
        payload.get("compiled_handle_authority_rejection_count")
        == authority_rejections,
        "compiled handle authority rejection accounting changed",
    )
    return tuple(verified)


def _answer_replay(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    preflight, prompts, rows = _read_phase_preflight(
        output_root,
        name=ANSWER_PREFLIGHT_NAME,
        expected_sha256=args.expected_preflight_sha256,
        validator=_validate_answer_preflight,
    )
    batch = _checkpoint_batch(
        preflight,
        prompts,
        output_root=output_root,
        checkpoint_dir_name=ANSWER_CHECKPOINT_DIR_NAME,
        run_format=ANSWER_RUN_FORMAT,
        phase="answer",
        client=None,
    )
    replayed = _answer_materialization_projection(preflight, rows, batch)
    run = read_sealed_json(output_root / ANSWER_RUN_NAME)
    _require(
        run.sha256 == require_sha256(args.expected_run_sha256, "expected answer run")
        and run.payload == replayed,
        "fact answer run differs from checkpoint-only replay",
    )
    _validate_answer_run(run, preflight, rows)
    payload = {
        "answer_preflight_artifact_sha256": preflight.sha256,
        "byte_identical": True,
        "expected_run_sha256": run.sha256,
        "format": ANSWER_REPLAY_FORMAT,
        "gold_loaded": False,
        "original_ordinals": list(REMAINING_ORDINALS),
        "physical_provider_calls": 0,
        "question_count": SUBSET_QUESTION_COUNT,
        "replayed_run_sha256": run.sha256,
        "retained_transformer_token_state_bytes": 0,
    }
    assert_gold_blind(payload, path="typed_fact_answer_replay")
    artifact, created = publish_sealed_json(
        output_root / ANSWER_REPLAY_NAME,
        payload,
    )
    return {
        "answer_replay_sha256": artifact.sha256,
        "answer_run_sha256": run.sha256,
        "byte_identical": True,
        "created": created,
        "physical_provider_calls": 0,
    }


def read_verified_fact_answer_run(
    output_root: str | Path,
    *,
    expected_preflight_sha256: str,
    expected_run_sha256: str,
    expected_replay_sha256: str,
) -> tuple[SealedArtifact, SealedArtifact, tuple[dict[str, Any], ...]]:
    """Return the replay-verified answer and its stable common judge rows."""

    root = Path(output_root)
    preflight, _prompts, rows = _read_phase_preflight(
        root,
        name=ANSWER_PREFLIGHT_NAME,
        expected_sha256=expected_preflight_sha256,
        validator=_validate_answer_preflight,
    )
    run = read_sealed_json(root / ANSWER_RUN_NAME)
    _require(
        run.sha256 == require_sha256(expected_run_sha256, "expected answer run"),
        "fact answer run SHA-256 changed",
    )
    judge_rows = _validate_answer_run(run, preflight, rows)
    replay = read_sealed_json(root / ANSWER_REPLAY_NAME)
    replay_payload = replay.payload
    _require(
        replay.sha256
        == require_sha256(expected_replay_sha256, "expected answer replay")
        and replay_payload.get("format") == ANSWER_REPLAY_FORMAT
        and replay_payload.get("byte_identical") is True
        and replay_payload.get("gold_loaded") is False
        and replay_payload.get("physical_provider_calls") == 0
        and replay_payload.get("retained_transformer_token_state_bytes") == 0
        and replay_payload.get("question_count") == SUBSET_QUESTION_COUNT
        and replay_payload.get("original_ordinals") == list(REMAINING_ORDINALS)
        and replay_payload.get("answer_preflight_artifact_sha256")
        == preflight.sha256
        and replay_payload.get("expected_run_sha256") == run.sha256
        and replay_payload.get("replayed_run_sha256") == run.sha256,
        "fact answer replay binding changed",
    )
    assert_gold_blind(replay_payload, path="verified_typed_fact_answer_replay")
    return run, replay, judge_rows


def _judge_preflight(args: argparse.Namespace) -> dict[str, Any]:
    run, replay, source_rows = read_verified_fact_answer_run(
        args.output_root,
        expected_preflight_sha256=args.expected_answer_preflight_sha256,
        expected_run_sha256=args.expected_answer_run_sha256,
        expected_replay_sha256=args.expected_answer_replay_sha256,
    )
    gold_rows, gold_sha = load_locked_typed_final_gold(
        dataset_path=args.dataset,
        split_path=args.split,
        source_rows=source_rows,
        allow_subset=True,
    )
    payload, _prompts = judge_preflight_projection(
        run_artifact=run,
        replay_artifact_sha256=replay.sha256,
        source_rows=source_rows,
        gold_rows=gold_rows,
        gold_population_sha256=gold_sha,
        mode="selected_subset",
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    _require(
        payload.get("selected_question_count") == SUBSET_QUESTION_COUNT
        and tuple(row.get("ordinal") for row in payload.get("prompt_rows", ()))
        == REMAINING_ORDINALS,
        "fact answer judge prompt projection changed",
    )
    artifact, created = publish_sealed_json(
        Path(args.judge_output_root) / JUDGE_PREFLIGHT_NAME,
        payload,
    )
    return {
        "answer_replay_sha256": replay.sha256,
        "answer_run_sha256": run.sha256,
        "created": created,
        "gold_loaded": True,
        "judge_mode": "selected_subset",
        "judge_preflight_sha256": artifact.sha256,
        "physical_provider_calls": 0,
        "required_authorized_provider_calls": SUBSET_QUESTION_COUNT,
        "selected_ordinals": list(REMAINING_ORDINALS),
        "selected_question_count": SUBSET_QUESTION_COUNT,
    }


def _add_output_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)


def _add_terra_settings(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=live.DEFAULT_TERRA_GATEWAY_MODEL)
    parser.add_argument("--gateway-url", default=live.DEFAULT_GATEWAY_URL)
    parser.add_argument("--max-concurrency", type=int, default=4)


def _add_provider_args(parser: argparse.ArgumentParser) -> None:
    _add_output_root(parser)
    parser.add_argument(
        "--expected-preflight-sha256",
        type=_sha256_argument,
        required=True,
    )
    parser.add_argument("--enable-provider", action="store_true")
    parser.add_argument("--authorized-provider-calls", type=int, default=0)
    parser.add_argument("--api-key-env", default=live.DEFAULT_API_KEY_ENV)


def _add_materialize_args(parser: argparse.ArgumentParser) -> None:
    _add_output_root(parser)
    parser.add_argument(
        "--expected-preflight-sha256",
        type=_sha256_argument,
        required=True,
    )


def _add_replay_args(parser: argparse.ArgumentParser) -> None:
    _add_materialize_args(parser)
    parser.add_argument(
        "--expected-run-sha256",
        type=_sha256_argument,
        required=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    compiler_preflight = commands.add_parser(
        "compiler-preflight",
        help="seal 24 parent-free fact compiler prompts from exact v3 rows",
    )
    _add_output_root(compiler_preflight)
    _add_terra_settings(compiler_preflight)
    compiler_preflight.add_argument(
        "--source-root", type=Path, default=DEFAULT_SOURCE_ROOT
    )
    compiler_preflight.add_argument(
        "--expected-source-composition-sha256",
        type=_sha256_argument,
        default=EXPECTED_SOURCE_COMPOSITION_SHA256,
    )
    compiler_preflight.add_argument(
        "--expected-source-preflight-sha256",
        type=_sha256_argument,
        default=EXPECTED_SOURCE_PREFLIGHT_SHA256,
    )

    compiler_provider = commands.add_parser(
        "compiler-provider-run",
        help="execute exactly the sealed 24-call compiler population",
    )
    _add_provider_args(compiler_provider)
    compiler_materialize = commands.add_parser(
        "compiler-materialize",
        help="validate compiler checkpoints into cited fact packets",
    )
    _add_materialize_args(compiler_materialize)
    compiler_replay = commands.add_parser(
        "compiler-replay",
        help="prove compiler materialization is checkpoint-byte-identical",
    )
    _add_replay_args(compiler_replay)

    answer_preflight = commands.add_parser(
        "answer-preflight",
        help="seal facts-first answers after verified compiler replay",
    )
    _add_output_root(answer_preflight)
    _add_terra_settings(answer_preflight)
    answer_preflight.add_argument(
        "--expected-compiler-preflight-sha256",
        type=_sha256_argument,
        required=True,
    )
    answer_preflight.add_argument(
        "--expected-compiler-run-sha256",
        type=_sha256_argument,
        required=True,
    )
    answer_preflight.add_argument(
        "--expected-compiler-replay-sha256",
        type=_sha256_argument,
        required=True,
    )
    answer_provider = commands.add_parser(
        "answer-provider-run",
        help="execute exactly the sealed 24-call facts-first answer population",
    )
    _add_provider_args(answer_provider)
    answer_materialize = commands.add_parser(
        "answer-materialize",
        help="materialize answers with the original typed-final validator",
    )
    _add_materialize_args(answer_materialize)
    answer_replay = commands.add_parser(
        "answer-replay",
        help="prove answer materialization is checkpoint-byte-identical",
    )
    _add_replay_args(answer_replay)

    judge_preflight = commands.add_parser(
        "judge-preflight",
        help="open locked gold only after answer replay and seal 24 Sol prompts",
    )
    _add_output_root(judge_preflight)
    judge_preflight.add_argument(
        "--judge-output-root", type=Path, default=DEFAULT_JUDGE_OUTPUT
    )
    judge_preflight.add_argument(
        "--expected-answer-preflight-sha256",
        type=_sha256_argument,
        required=True,
    )
    judge_preflight.add_argument(
        "--expected-answer-run-sha256",
        type=_sha256_argument,
        required=True,
    )
    judge_preflight.add_argument(
        "--expected-answer-replay-sha256",
        type=_sha256_argument,
        required=True,
    )
    judge_preflight.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    judge_preflight.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    judge_preflight.add_argument(
        "--model", default=judging.DEFAULT_SOL_GATEWAY_MODEL
    )
    judge_preflight.add_argument("--gateway-url", default=live.DEFAULT_GATEWAY_URL)
    judge_preflight.add_argument("--max-concurrency", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    commands: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
        "compiler-preflight": _compiler_preflight,
        "compiler-provider-run": _compiler_provider,
        "compiler-materialize": _compiler_materialize,
        "compiler-replay": _compiler_replay,
        "answer-preflight": _answer_preflight,
        "answer-provider-run": _answer_provider,
        "answer-materialize": _answer_materialize,
        "answer-replay": _answer_replay,
        "judge-preflight": _judge_preflight,
    }
    result = commands[args.command](args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANSWER_CHECKPOINT_DIR_NAME",
    "ANSWER_PREFLIGHT_NAME",
    "ANSWER_REPLAY_NAME",
    "ANSWER_RUN_NAME",
    "COMPILER_CHECKPOINT_DIR_NAME",
    "COMPILER_PREFLIGHT_NAME",
    "COMPILER_REPLAY_NAME",
    "COMPILER_RUN_NAME",
    "DEFAULT_OUTPUT",
    "DEFAULT_SOURCE_ROOT",
    "EXPECTED_SOURCE_COMPOSITION_SHA256",
    "EXPECTED_SOURCE_PREFLIGHT_SHA256",
    "FORMAT",
    "LockedTypedFactCompilerError",
    "REMAINING_ORDINALS",
    "SUBSET_QUESTION_COUNT",
    "main",
    "read_verified_fact_answer_run",
]
