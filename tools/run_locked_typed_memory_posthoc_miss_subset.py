#!/usr/bin/env python3
"""Run the sealed 27-question posthoc typed-memory miss subset.

``preflight`` is the only phase allowed to read the outcome-conditioned Sol
judgment and score ledger.  It verifies the fixed miss set, publishes a sealed
selection plan, and copies the exact corresponding rows from the locked
shared-surplus typed-final preflight into a separate gold-free preflight.

``provider-run`` reads only that subset preflight and requires exact authority
for 27 calls.  ``materialize`` and ``replay`` read only the subset preflight,
its immutable completion journals, and (for replay) the sealed subset run.
They never reopen the selection judgment, score ledger, source composition, or
source preflight.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
from memory_condense.eval.fast_completion_runtime import (  # noqa: E402
    FastCompletionBatch,
    FastCompletionRuntime,
    preflight_fast_completion_prompts,
)
from tools import run_locked_typed_memory_final_arm as typed_cli  # noqa: E402
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
    COMPOSITION_FORMAT,
    EXPECTED_QUESTION_COUNT,
    MAX_CHAT_PROMPT_TOKENS,
    OUTPUT_TOKEN_RESERVE,
    VALIDATOR_POLICY_FORMAT,
    judge_row_projection,
    materialize_typed_final_result_row,
)
from tools.matched_eval.typed_memory_final_judging import (  # noqa: E402
    JUDGE_FORMAT,
    JUDGE_NAME,
    SCORE_FORMAT,
    SCORE_NAME,
)


FORMAT = "memory-condense-locked-typed-memory-posthoc-miss-subset-v1"
SELECTION_PLAN_FORMAT = f"{FORMAT}-selection-plan-v1"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight-v1"
RUN_FORMAT = f"{FORMAT}-run-v1"
REPLAY_FORMAT = f"{FORMAT}-replay-v1"

SELECTION_PLAN_NAME = "typed-memory-posthoc-miss27-selection-plan-v1.json"
PREFLIGHT_NAME = "typed-memory-posthoc-miss27-preflight-v1.json"
RUN_NAME = "typed-memory-posthoc-miss27-run-v1.json"
REPLAY_NAME = "typed-memory-posthoc-miss27-replay-v1.json"
CHECKPOINT_DIR_NAME = "terra-typed-memory-posthoc-miss27-v1-calls"

MISS_ORDINALS = (
    6,
    7,
    14,
    16,
    17,
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
    74,
    77,
    79,
    81,
    86,
    87,
    93,
    94,
    97,
)
SUBSET_QUESTION_COUNT = len(MISS_ORDINALS)

EXPECTED_SOURCE_PREFLIGHT_SHA256 = (
    "c74874b4ff13189afd31902cd77f812cc67accf51797a5e6f5022e9fa1f961d0"
)
EXPECTED_SELECTION_JUDGE_SHA256 = (
    "7ddbfe25e1f048e44524fb948d29463d9393c6a8b0fdee6c62cd0bc965f295e0"
)
EXPECTED_SELECTION_SCORE_SHA256 = (
    "34a1cfff13acf00170c101db9e37490d3c3ef3b607698a89021519362f1f2b1a"
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = (
    REPOSITORY_ROOT
    / "eval_results"
    / "matched_eval_100"
    / "typed-memory-final-v3-shared-surplus"
)
DEFAULT_SOURCE_PREFLIGHT = DEFAULT_SOURCE_ROOT / typed_cli.PREFLIGHT_NAME
DEFAULT_SOURCE_COMPOSITION = DEFAULT_SOURCE_ROOT / typed_cli.COMPOSITION_NAME
DEFAULT_SELECTION_ROOT = (
    REPOSITORY_ROOT
    / "eval_results"
    / "matched_eval_100"
    / "typed-memory-final-v2-compact-budget"
    / "sol-judge-v1"
)
DEFAULT_SELECTION_JUDGE = DEFAULT_SELECTION_ROOT / JUDGE_NAME
DEFAULT_SELECTION_SCORE = DEFAULT_SELECTION_ROOT / SCORE_NAME
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "eval_results"
    / "matched_eval_100"
    / "typed-memory-final-v3-posthoc-miss27"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class LockedTypedMemoryPosthocSubsetError(MatchedEvalContractError):
    """A sealed subset, authority, prompt, journal, or output changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedTypedMemoryPosthocSubsetError(message)


def _sha256_argument(value: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return value


def _validate_prompt_rows(
    raw_rows: object,
    *,
    expected_ordinals: Sequence[int],
    label: str,
) -> tuple[
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
]:
    expected = tuple(expected_ordinals)
    _require(
        type(raw_rows) is list and len(raw_rows) == len(expected),
        f"{label} prompt row population changed",
    )
    prompts: list[tuple[dict[str, str], ...]] = []
    rows: list[dict[str, Any]] = []
    question_ids: list[str] = []
    for source_ordinal, raw in zip(expected, raw_rows, strict=True):
        _require(type(raw) is dict, f"{label} prompt row changed type")
        assert type(raw) is dict
        declared = require_sha256(
            raw.get("prompt_row_receipt_sha256"), f"{label} prompt row"
        )
        body = dict(raw)
        body.pop("prompt_row_receipt_sha256")
        messages = raw.get("messages")
        _require(
            identity_sha256(body) == declared
            and raw.get("ordinal") == source_ordinal
            and type(messages) is list,
            f"{label} prompt row seal/order changed",
        )
        plain = tuple(
            {"role": row["role"], "content": row["content"]}
            for row in messages
            if type(row) is dict
            and set(row) == {"role", "content"}
            and row.get("role") in {"system", "user", "assistant"}
            and type(row.get("content")) is str
        )
        _require(
            len(plain) == len(messages)
            and identity_sha256(list(plain)) == raw.get("messages_sha256")
            and count_chat_prompt_token_proxy(plain)
            == raw.get("prompt_token_proxy")
            and int(raw["prompt_token_proxy"]) + OUTPUT_TOKEN_RESERVE <= 8_000,
            f"{label} prompt messages/budget changed",
        )
        assert_gold_blind(list(plain), path=f"{label}_provider_messages")
        user_payloads: list[object] = []
        for message in plain:
            if message["role"] != "user":
                continue
            try:
                user_payloads.append(json.loads(message["content"]))
            except json.JSONDecodeError as exc:
                raise LockedTypedMemoryPosthocSubsetError(
                    f"{label} provider user message is not sealed JSON"
                ) from exc
        _require(
            bool(user_payloads),
            f"{label} provider prompt has no user payload",
        )
        assert_gold_blind(
            user_payloads,
            path=f"{label}_decoded_provider_user_payloads",
        )
        prompts.append(plain)
        rows.append(dict(raw))
        question_ids.append(
            require_text(raw.get("question_id"), f"{label} question ID")
        )
    _require(
        len(set(question_ids)) == len(expected),
        f"{label} question identities repeat",
    )
    return tuple(prompts), tuple(rows)


def _validate_source_preflight(
    artifact: SealedArtifact,
) -> tuple[
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
]:
    payload = artifact.payload
    assert_gold_blind(payload, path="posthoc_subset_source_preflight")
    _require(
        payload.get("format") == typed_cli.PREFLIGHT_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("hard_prompt_token_cap") == 8_000
        and payload.get("max_chat_prompt_tokens") == MAX_CHAT_PROMPT_TOKENS
        and payload.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
        and payload.get("question_count") == EXPECTED_QUESTION_COUNT
        and payload.get("required_authorized_provider_calls")
        == EXPECTED_QUESTION_COUNT
        and type(payload.get("model")) is str
        and bool(payload.get("model"))
        and type(payload.get("gateway_url")) is str
        and bool(payload.get("gateway_url"))
        and type(payload.get("max_concurrency")) is int
        and int(payload["max_concurrency"]) > 0,
        "source typed-final preflight changed",
    )
    prompts, rows = _validate_prompt_rows(
        payload.get("physical_prompt_rows"),
        expected_ordinals=range(EXPECTED_QUESTION_COUNT),
        label="source typed-final",
    )
    population = preflight_fast_completion_prompts(
        prompts,
        max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS,
    )
    _require(
        population.logical_prompt_count
        == population.unique_prompt_count
        == EXPECTED_QUESTION_COUNT
        and population.prompt_population_sha256
        == payload.get("prompt_population_sha256")
        and population.model_dump() == payload.get("prompt_population"),
        "source typed-final prompt population changed",
    )
    require_sha256(
        payload.get("composition_artifact_sha256"),
        "source composition artifact",
    )
    _require(
        type(payload.get("source_hash_bindings")) is dict,
        "source typed-final hash bindings changed",
    )
    return prompts, rows


def _validated_selection_rows(
    judge: SealedArtifact,
    score: SealedArtifact,
) -> tuple[dict[str, Any], ...]:
    judge_payload = judge.payload
    score_payload = score.payload
    questions = judge_payload.get("questions")
    aggregate = judge_payload.get("aggregate")
    _require(
        judge_payload.get("format") == JUDGE_FORMAT
        and judge_payload.get("gold_loaded") is True
        and judge_payload.get("question_count") == EXPECTED_QUESTION_COUNT
        and judge_payload.get("selected_question_count")
        == EXPECTED_QUESTION_COUNT
        and judge_payload.get("judge_mode") == "full100"
        and type(aggregate) is dict
        and aggregate.get("question_count") == EXPECTED_QUESTION_COUNT
        and aggregate.get("correct") == EXPECTED_QUESTION_COUNT - SUBSET_QUESTION_COUNT
        and type(questions) is list
        and len(questions) == EXPECTED_QUESTION_COUNT,
        "selection judgment population changed",
    )
    validated: list[dict[str, Any]] = []
    wrong: list[int] = []
    for ordinal, raw in enumerate(questions):
        _require(type(raw) is dict, "selection judgment row changed type")
        assert type(raw) is dict
        declared = require_sha256(
            raw.get("judge_row_sha256"), "selection judgment row"
        )
        body = dict(raw)
        body.pop("judge_row_sha256")
        _require(
            identity_sha256(body) == declared
            and raw.get("ordinal") == ordinal
            and type(raw.get("correct")) is bool,
            "selection judgment row seal/order changed",
        )
        if raw["correct"] is False:
            wrong.append(ordinal)
        validated.append(dict(raw))
    _require(
        tuple(wrong) == MISS_ORDINALS,
        "selection judgment miss ordinals changed",
    )
    _require(
        score_payload.get("format") == SCORE_FORMAT
        and score_payload.get("judge_mode") == "full100"
        and score_payload.get("question_count") == EXPECTED_QUESTION_COUNT
        and score_payload.get("selected_question_count")
        == EXPECTED_QUESTION_COUNT
        and score_payload.get("correct")
        == EXPECTED_QUESTION_COUNT - SUBSET_QUESTION_COUNT
        and score_payload.get("selected_accuracy")
        == (EXPECTED_QUESTION_COUNT - SUBSET_QUESTION_COUNT)
        / EXPECTED_QUESTION_COUNT
        and score_payload.get("typed_final_run_sha256")
        == judge_payload.get("typed_final_run_sha256"),
        "selection score ledger changed",
    )
    return tuple(validated)


def _selection_plan_projection(
    source_preflight: SealedArtifact,
    source_composition: SealedArtifact,
    source_rows: Sequence[Mapping[str, Any]],
    selection_judge: SealedArtifact,
    selection_score: SealedArtifact,
) -> dict[str, Any]:
    judge_rows = _validated_selection_rows(selection_judge, selection_score)
    selected_rows: list[dict[str, Any]] = []
    for ordinal in MISS_ORDINALS:
        source = source_rows[ordinal]
        judged = judge_rows[ordinal]
        _require(
            judged.get("question_id") == source.get("question_id")
            and judged.get("question_sha256") == source.get("question_sha256")
            and judged.get("dated_question_sha256")
            == source.get("dated_question_sha256")
            and judged.get("route_id") == source.get("route_id"),
            f"selection/source question binding changed at ordinal {ordinal}",
        )
        body = {
            "ordinal": ordinal,
            "question_id": source["question_id"],
            "question_sha256": source["question_sha256"],
            "selection_judge_row_sha256": judged["judge_row_sha256"],
            "source_messages_sha256": source["messages_sha256"],
            "source_prompt_row_receipt_sha256": source[
                "prompt_row_receipt_sha256"
            ],
        }
        selected_rows.append(
            {**body, "selection_row_sha256": identity_sha256(body)}
        )
    return {
        "format": SELECTION_PLAN_FORMAT,
        "gold_bearing_selection_authority_only": True,
        "gold_or_reference_copied_into_provider_messages": False,
        "parent_judgment_artifact_sha256": selection_judge.sha256,
        "parent_score_artifact_sha256": selection_score.sha256,
        "provider_and_materialization_read_this_artifact": False,
        "question_count": SUBSET_QUESTION_COUNT,
        "selected_ordinals": list(MISS_ORDINALS),
        "selected_rows": selected_rows,
        "selection_is_posthoc_outcome_conditioned": True,
        "selection_rule": "exact false verdicts in sealed compact-v2 full100 Sol judgment",
        "source_composition_artifact_sha256": source_composition.sha256,
        "source_preflight_artifact_sha256": source_preflight.sha256,
    }


def _subset_preflight_projection(
    source_preflight: SealedArtifact,
    source_rows: Sequence[Mapping[str, Any]],
    selection_plan: SealedArtifact,
) -> tuple[dict[str, Any], tuple[tuple[dict[str, str], ...], ...]]:
    selected_rows = tuple(dict(source_rows[ordinal]) for ordinal in MISS_ORDINALS)
    prompts = tuple(
        tuple(dict(message) for message in row["messages"])
        for row in selected_rows
    )
    population = preflight_fast_completion_prompts(
        prompts,
        max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS,
    )
    _require(
        population.logical_prompt_count
        == population.unique_prompt_count
        == SUBSET_QUESTION_COUNT,
        "posthoc subset requires 27 distinct prompt identities",
    )
    source_payload = source_preflight.payload
    payload = {
        "format": PREFLIGHT_FORMAT,
        "gateway_url": source_payload["gateway_url"],
        "gold_loaded": False,
        "hard_prompt_token_cap": 8_000,
        "max_chat_prompt_tokens": MAX_CHAT_PROMPT_TOKENS,
        "max_concurrency": source_payload["max_concurrency"],
        "model": source_payload["model"],
        "observed_max_complete_envelope_tokens": max(
            row["prompt_token_proxy"] + OUTPUT_TOKEN_RESERVE
            for row in selected_rows
        ),
        "original_ordinals": list(MISS_ORDINALS),
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "physical_prompt_rows": list(selected_rows),
        "prompt_population": population.model_dump(),
        "prompt_population_sha256": population.prompt_population_sha256,
        "provider_calls": 0,
        "question_count": SUBSET_QUESTION_COUNT,
        "required_authorized_provider_calls": SUBSET_QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "selection_authority_artifacts_required_at_runtime": False,
        "selection_is_posthoc_outcome_conditioned": True,
        "selection_plan_artifact_sha256": selection_plan.sha256,
        "source_composition_artifact_sha256": source_payload[
            "composition_artifact_sha256"
        ],
        "source_hash_bindings": dict(source_payload["source_hash_bindings"]),
        "source_messages_sha256s": [
            row["messages_sha256"] for row in selected_rows
        ],
        "source_preflight_artifact_sha256": source_preflight.sha256,
        "source_prompt_row_receipt_sha256s": [
            row["prompt_row_receipt_sha256"] for row in selected_rows
        ],
    }
    assert_gold_blind(payload, path="posthoc_miss_subset_preflight")
    return payload, prompts


def _require_distinct_output(
    output_root: Path,
    source_preflight: Path,
    selection_judge: Path,
) -> None:
    output = output_root.resolve()
    forbidden = {
        source_preflight.parent.resolve(),
        selection_judge.parent.resolve(),
    }
    _require(
        output not in forbidden,
        "posthoc subset output root must be distinct from source/selection roots",
    )


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    source_preflight_path = Path(args.source_preflight)
    selection_judge_path = Path(args.selection_judge)
    _require_distinct_output(
        output_root,
        source_preflight_path,
        selection_judge_path,
    )
    source_preflight = read_sealed_json(source_preflight_path)
    _require(
        source_preflight.sha256
        == require_sha256(
            args.expected_source_preflight_sha256,
            "expected source preflight",
        ),
        "source typed-final preflight SHA-256 changed",
    )
    _source_prompts, source_rows = _validate_source_preflight(source_preflight)

    source_composition = read_sealed_json(Path(args.source_composition))
    _require(
        source_composition.sha256
        == source_preflight.payload["composition_artifact_sha256"]
        and source_composition.payload.get("format") == COMPOSITION_FORMAT,
        "source typed-final composition binding changed",
    )
    selection_judge = read_sealed_json(selection_judge_path)
    selection_score = read_sealed_json(Path(args.selection_score))
    _require(
        selection_judge.sha256
        == require_sha256(
            args.expected_selection_judge_sha256,
            "expected selection judgment",
        )
        and selection_score.sha256
        == require_sha256(
            args.expected_selection_score_sha256,
            "expected selection score",
        ),
        "posthoc selection authority SHA-256 changed",
    )
    plan_payload = _selection_plan_projection(
        source_preflight,
        source_composition,
        source_rows,
        selection_judge,
        selection_score,
    )
    plan, plan_created = publish_sealed_json(
        output_root / SELECTION_PLAN_NAME,
        plan_payload,
    )
    preflight_payload, _prompts = _subset_preflight_projection(
        source_preflight,
        source_rows,
        plan,
    )
    preflight, preflight_created = publish_sealed_json(
        output_root / PREFLIGHT_NAME,
        preflight_payload,
    )
    return {
        "gold_loaded_into_provider_messages": False,
        "original_ordinals": list(MISS_ORDINALS),
        "physical_provider_calls": 0,
        "preflight": preflight.path.as_posix(),
        "preflight_created": preflight_created,
        "preflight_sha256": preflight.sha256,
        "question_count": SUBSET_QUESTION_COUNT,
        "required_authorized_provider_calls": SUBSET_QUESTION_COUNT,
        "selection_plan": plan.path.as_posix(),
        "selection_plan_created": plan_created,
        "selection_plan_sha256": plan.sha256,
        "source_composition_sha256": source_composition.sha256,
        "source_preflight_sha256": source_preflight.sha256,
    }


def _validate_subset_preflight(
    artifact: SealedArtifact,
) -> tuple[
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
]:
    payload = artifact.payload
    assert_gold_blind(payload, path="posthoc_subset_runtime_preflight")
    _require(
        payload.get("format") == PREFLIGHT_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("hard_prompt_token_cap") == 8_000
        and payload.get("max_chat_prompt_tokens") == MAX_CHAT_PROMPT_TOKENS
        and payload.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
        and payload.get("question_count") == SUBSET_QUESTION_COUNT
        and payload.get("required_authorized_provider_calls")
        == SUBSET_QUESTION_COUNT
        and payload.get("original_ordinals") == list(MISS_ORDINALS)
        and payload.get("selection_is_posthoc_outcome_conditioned") is True
        and payload.get("selection_authority_artifacts_required_at_runtime")
        is False
        and type(payload.get("model")) is str
        and bool(payload.get("model"))
        and type(payload.get("gateway_url")) is str
        and bool(payload.get("gateway_url"))
        and type(payload.get("max_concurrency")) is int
        and int(payload["max_concurrency"]) > 0,
        "posthoc subset preflight firewall/population changed",
    )
    for key in (
        "selection_plan_artifact_sha256",
        "source_composition_artifact_sha256",
        "source_preflight_artifact_sha256",
    ):
        require_sha256(payload.get(key), f"posthoc subset {key}")
    _require(
        type(payload.get("source_hash_bindings")) is dict,
        "posthoc subset source hash bindings changed",
    )
    prompts, rows = _validate_prompt_rows(
        payload.get("physical_prompt_rows"),
        expected_ordinals=MISS_ORDINALS,
        label="posthoc subset",
    )
    population = preflight_fast_completion_prompts(
        prompts,
        max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS,
    )
    _require(
        population.logical_prompt_count
        == population.unique_prompt_count
        == SUBSET_QUESTION_COUNT
        and population.prompt_population_sha256
        == payload.get("prompt_population_sha256")
        and population.model_dump() == payload.get("prompt_population")
        and payload.get("source_messages_sha256s")
        == [row["messages_sha256"] for row in rows]
        and payload.get("source_prompt_row_receipt_sha256s")
        == [row["prompt_row_receipt_sha256"] for row in rows],
        "posthoc subset prompt population/source view changed",
    )
    return prompts, rows


def _read_subset_preflight(
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
        == require_sha256(expected_sha256, "expected subset preflight"),
        "posthoc subset preflight SHA-256 changed",
    )
    prompts, rows = _validate_subset_preflight(artifact)
    return artifact, prompts, rows


def _runtime(
    artifact: SealedArtifact,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    output_root: Path,
    client: Any | None,
) -> FastCompletionRuntime:
    payload = artifact.payload
    return FastCompletionRuntime(
        checkpoint_dir=output_root / CHECKPOINT_DIR_NAME,
        prompt_population=prompts,
        model=payload["model"],
        client=client,
        max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS,
        max_new_tokens=OUTPUT_TOKEN_RESERVE,
        max_concurrency=payload["max_concurrency"],
        retries=0,
        benchmark_provenance={
            "arm": "locked_typed_memory_posthoc_miss27_v1",
            "authorized_unique_calls": SUBSET_QUESTION_COUNT,
            "experiment_format": RUN_FORMAT,
            "gateway_url": payload["gateway_url"],
            "gold_loaded": False,
            "selection_plan_artifact_sha256": payload[
                "selection_plan_artifact_sha256"
            ],
            "source_composition_artifact_sha256": payload[
                "source_composition_artifact_sha256"
            ],
            "source_preflight_artifact_sha256": payload[
                "source_preflight_artifact_sha256"
            ],
            "subset_preflight_artifact_sha256": artifact.sha256,
        },
    )


def _checkpoint_batch(
    artifact: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    output_root: Path,
    client: Any | None,
) -> FastCompletionBatch:
    runtime = _runtime(
        artifact,
        prompts,
        output_root=output_root,
        client=client,
    )
    try:
        return runtime.run()
    finally:
        runtime.close()


def _provider(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    artifact, prompts, _rows = _read_subset_preflight(
        output_root,
        args.expected_preflight_sha256,
    )
    _require(
        args.enable_provider is True
        and args.authorized_provider_calls == SUBSET_QUESTION_COUNT,
        "provider-run requires exact authorization for 27 calls",
    )
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
        "posthoc subset provider population changed",
    )
    return {
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "gold_loaded": False,
        "physical_provider_calls": batch.usage.physical_calls,
        "preflight_sha256": artifact.sha256,
        "required_authorized_provider_calls": SUBSET_QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
    }


def _materialization_projection(
    preflight: SealedArtifact,
    prompt_rows: tuple[dict[str, Any], ...],
    batch: FastCompletionBatch,
) -> dict[str, Any]:
    _require(
        batch.usage.logical_calls
        == batch.usage.unique_calls
        == SUBSET_QUESTION_COUNT
        and batch.usage.checkpoint_hits == SUBSET_QUESTION_COUNT
        and batch.usage.physical_calls == 0
        and len(batch.logical_completions) == SUBSET_QUESTION_COUNT
        and len(batch.unique_records) == SUBSET_QUESTION_COUNT,
        "posthoc subset materialization requires 27 checkpoint hits",
    )
    record_by_messages = {
        row.messages_sha256: row for row in batch.unique_records
    }
    _require(
        len(record_by_messages) == SUBSET_QUESTION_COUNT,
        "posthoc subset completion identities repeat",
    )
    results: list[dict[str, Any]] = []
    for plan, completion in zip(
        prompt_rows,
        batch.logical_completions,
        strict=True,
    ):
        record = record_by_messages.get(plan["messages_sha256"])
        _require(
            record is not None
            and record.completion == completion
            and record.checkpoint_hit is True
            and record.physical_call is False,
            "posthoc subset checkpoint record changed",
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
        tuple(row["ordinal"] for row in results) == MISS_ORDINALS
        and tuple(row["ordinal"] for row in judge_rows) == MISS_ORDINALS,
        "posthoc subset result ordinals changed",
    )
    source = preflight.payload
    payload = {
        "changed_prediction_count": sum(
            bool(row["changed_from_parent"]) for row in results
        ),
        "completion_batch": batch.model_dump(),
        "format": RUN_FORMAT,
        "gold_loaded": False,
        "invalid_completion_parent_fallback_count": sum(
            row["prediction_source"]
            == "typed_final_invalid_keep_parent_v1"
            for row in results
        ),
        "judge_rows": judge_rows,
        "original_ordinals": list(MISS_ORDINALS),
        "physical_provider_calls_during_materialization": 0,
        "posthoc_selection": True,
        "questions": results,
        "question_count": SUBSET_QUESTION_COUNT,
        "required_authorized_provider_calls": SUBSET_QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "selection_plan_artifact_sha256": source[
            "selection_plan_artifact_sha256"
        ],
        "source_composition_artifact_sha256": source[
            "source_composition_artifact_sha256"
        ],
        "source_hash_bindings": dict(source["source_hash_bindings"]),
        "source_preflight_artifact_sha256": source[
            "source_preflight_artifact_sha256"
        ],
        "source_population_question_count": EXPECTED_QUESTION_COUNT,
        "subset_preflight_artifact_sha256": preflight.sha256,
        "validator_policy_format": VALIDATOR_POLICY_FORMAT,
    }
    assert_gold_blind(payload, path="posthoc_miss_subset_run")
    return payload


def _materialize(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    preflight, prompts, rows = _read_subset_preflight(
        output_root,
        args.expected_preflight_sha256,
    )
    batch = _checkpoint_batch(
        preflight,
        prompts,
        output_root=output_root,
        client=None,
    )
    payload = _materialization_projection(preflight, rows, batch)
    artifact, created = publish_sealed_json(output_root / RUN_NAME, payload)
    return {
        "artifact": artifact.path.as_posix(),
        "created": created,
        "gold_loaded": False,
        "original_ordinals": list(MISS_ORDINALS),
        "physical_provider_calls": 0,
        "question_count": SUBSET_QUESTION_COUNT,
        "run_sha256": artifact.sha256,
    }


def _validate_subset_run(
    artifact: SealedArtifact,
    preflight: SealedArtifact,
    prompt_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    payload = artifact.payload
    assert_gold_blind(payload, path="verified_posthoc_subset_run")
    questions = payload.get("questions")
    projected = payload.get("judge_rows")
    _require(
        payload.get("format") == RUN_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("physical_provider_calls_during_materialization") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("question_count") == SUBSET_QUESTION_COUNT
        and payload.get("original_ordinals") == list(MISS_ORDINALS)
        and payload.get("subset_preflight_artifact_sha256") == preflight.sha256
        and payload.get("selection_plan_artifact_sha256")
        == preflight.payload["selection_plan_artifact_sha256"]
        and payload.get("source_preflight_artifact_sha256")
        == preflight.payload["source_preflight_artifact_sha256"]
        and payload.get("source_composition_artifact_sha256")
        == preflight.payload["source_composition_artifact_sha256"]
        and type(questions) is list
        and type(projected) is list
        and len(questions) == len(projected) == SUBSET_QUESTION_COUNT,
        "posthoc subset run envelope changed",
    )
    verified: list[dict[str, Any]] = []
    for ordinal, source, judge, prompt in zip(
        MISS_ORDINALS,
        questions,
        projected,
        prompt_rows,
        strict=True,
    ):
        _require(
            type(source) is dict and type(judge) is dict,
            "posthoc subset result row changed type",
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
            f"posthoc subset result binding changed at ordinal {ordinal}",
        )
        verified.append(dict(judge))
    return tuple(verified)


def _replay(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    preflight, prompts, rows = _read_subset_preflight(
        output_root,
        args.expected_preflight_sha256,
    )
    batch = _checkpoint_batch(
        preflight,
        prompts,
        output_root=output_root,
        client=None,
    )
    replayed = _materialization_projection(preflight, rows, batch)
    run = read_sealed_json(output_root / RUN_NAME)
    _require(
        run.sha256 == require_sha256(args.expected_run_sha256, "expected subset run")
        and run.payload == replayed,
        "posthoc subset run differs from checkpoint-only replay",
    )
    _validate_subset_run(run, preflight, rows)
    payload = {
        "byte_identical": True,
        "expected_run_sha256": run.sha256,
        "format": REPLAY_FORMAT,
        "gold_loaded": False,
        "original_ordinals": list(MISS_ORDINALS),
        "physical_provider_calls": 0,
        "question_count": SUBSET_QUESTION_COUNT,
        "replayed_run_sha256": run.sha256,
        "subset_preflight_artifact_sha256": preflight.sha256,
    }
    assert_gold_blind(payload, path="posthoc_miss_subset_replay")
    artifact, created = publish_sealed_json(output_root / REPLAY_NAME, payload)
    return {
        "artifact": artifact.path.as_posix(),
        "byte_identical": True,
        "created": created,
        "physical_provider_calls": 0,
        "replay_sha256": artifact.sha256,
        "run_sha256": run.sha256,
    }


def read_verified_subset_run(
    output_root: str | Path,
    *,
    expected_preflight_sha256: str,
    expected_run_sha256: str,
    expected_replay_sha256: str,
) -> tuple[SealedArtifact, SealedArtifact, tuple[dict[str, Any], ...]]:
    """Return the verified run/replay pair and exact 27 typed judge rows.

    This reader intentionally does not open the posthoc selection plan or its
    parent judgment/score authority.  The answer plane is bound through the
    subset preflight and byte-identical replay receipts alone.
    """

    root = Path(output_root)
    preflight, _prompts, rows = _read_subset_preflight(
        root,
        expected_preflight_sha256,
    )
    run = read_sealed_json(root / RUN_NAME)
    _require(
        run.sha256 == require_sha256(expected_run_sha256, "expected subset run"),
        "posthoc subset run SHA-256 changed",
    )
    judge_rows = _validate_subset_run(run, preflight, rows)
    replay = read_sealed_json(root / REPLAY_NAME)
    replay_payload = replay.payload
    _require(
        replay.sha256
        == require_sha256(expected_replay_sha256, "expected subset replay")
        and replay_payload.get("format") == REPLAY_FORMAT
        and replay_payload.get("byte_identical") is True
        and replay_payload.get("gold_loaded") is False
        and replay_payload.get("physical_provider_calls") == 0
        and replay_payload.get("question_count") == SUBSET_QUESTION_COUNT
        and replay_payload.get("original_ordinals") == list(MISS_ORDINALS)
        and replay_payload.get("expected_run_sha256") == run.sha256
        and replay_payload.get("replayed_run_sha256") == run.sha256
        and replay_payload.get("subset_preflight_artifact_sha256")
        == preflight.sha256,
        "posthoc subset replay binding changed",
    )
    assert_gold_blind(replay_payload, path="verified_posthoc_subset_replay")
    return run, replay, judge_rows


def _add_output_root(command: argparse.ArgumentParser) -> None:
    command.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser(
        "preflight",
        help="seal the fixed posthoc plan and exact 27-row prompt view",
    )
    _add_output_root(preflight)
    preflight.add_argument(
        "--source-preflight", type=Path, default=DEFAULT_SOURCE_PREFLIGHT
    )
    preflight.add_argument(
        "--source-composition", type=Path, default=DEFAULT_SOURCE_COMPOSITION
    )
    preflight.add_argument(
        "--selection-judge", type=Path, default=DEFAULT_SELECTION_JUDGE
    )
    preflight.add_argument(
        "--selection-score", type=Path, default=DEFAULT_SELECTION_SCORE
    )
    preflight.add_argument(
        "--expected-source-preflight-sha256",
        type=_sha256_argument,
        default=EXPECTED_SOURCE_PREFLIGHT_SHA256,
    )
    preflight.add_argument(
        "--expected-selection-judge-sha256",
        type=_sha256_argument,
        default=EXPECTED_SELECTION_JUDGE_SHA256,
    )
    preflight.add_argument(
        "--expected-selection-score-sha256",
        type=_sha256_argument,
        default=EXPECTED_SELECTION_SCORE_SHA256,
    )

    provider = commands.add_parser(
        "provider-run",
        help="execute only the sealed 27-prompt population",
    )
    _add_output_root(provider)
    provider.add_argument(
        "--expected-preflight-sha256", type=_sha256_argument, required=True
    )
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, default=0)
    provider.add_argument("--api-key-env", default=live.DEFAULT_API_KEY_ENV)

    materialize = commands.add_parser(
        "materialize",
        help="consume the 27 immutable completion checkpoints only",
    )
    _add_output_root(materialize)
    materialize.add_argument(
        "--expected-preflight-sha256", type=_sha256_argument, required=True
    )

    replay = commands.add_parser(
        "replay",
        help="prove byte-identical subset materialization from checkpoints",
    )
    _add_output_root(replay)
    replay.add_argument(
        "--expected-preflight-sha256", type=_sha256_argument, required=True
    )
    replay.add_argument(
        "--expected-run-sha256", type=_sha256_argument, required=True
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = _preflight(args)
    elif args.command == "provider-run":
        result = _provider(args)
    elif args.command == "materialize":
        result = _materialize(args)
    elif args.command == "replay":
        result = _replay(args)
    else:  # pragma: no cover
        raise AssertionError("unreachable command")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_DIR_NAME",
    "DEFAULT_OUTPUT",
    "EXPECTED_SELECTION_JUDGE_SHA256",
    "EXPECTED_SELECTION_SCORE_SHA256",
    "EXPECTED_SOURCE_PREFLIGHT_SHA256",
    "FORMAT",
    "LockedTypedMemoryPosthocSubsetError",
    "MISS_ORDINALS",
    "PREFLIGHT_NAME",
    "REPLAY_NAME",
    "RUN_NAME",
    "SELECTION_PLAN_NAME",
    "SUBSET_QUESTION_COUNT",
    "main",
    "read_verified_subset_run",
]
