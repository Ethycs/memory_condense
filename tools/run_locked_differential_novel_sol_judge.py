#!/usr/bin/env python3
"""Execute only the sealed novel rows of a differential Sol judge plan.

The provider-free differential planner remains the sole selection authority.
This lifecycle copies its ``novel_prompt_rows`` byte-for-byte into a sealed
preflight, requires a separate provider-release artifact, and owns a distinct
zero-retry checkpoint namespace.  Provider authorization must equal the exact
number of missing request/response pairs.  Materialization and replay are
checkpoint-only, and the resulting preflight/run/replay triplet is accepted by
``authenticate_prior_judge_run`` for the final differential merge.
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
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from dotenv import load_dotenv  # noqa: E402

from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from memory_condense.eval._binary_judge_protocol import (  # noqa: E402
    JUDGE_MAX_TOKENS,
    parse_binary_judge_verdict,
)
from memory_condense.eval.benchmark import build_judge_prompt  # noqa: E402
from memory_condense.eval.fast_completion_runtime import (  # noqa: E402
    FastCompletionBatch,
    FastCompletionRuntime,
    preflight_fast_completion_prompts,
)
from tools import plan_provider_free_differential_judge as plan_cli  # noqa: E402
from tools.matched_eval import judging, live  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    identity_sha256,
    require_sha256,
    require_text,
)


FORMAT = "memory-condense-locked-differential-novel-sol-judge-v1"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight-v1"
RELEASE_FORMAT = f"{FORMAT}-provider-release-v1"
JUDGE_FORMAT = f"{FORMAT}-run-v1"
JUDGE_ROW_FORMAT = f"{FORMAT}-row-v1"
JOURNAL_OWNER_FORMAT = f"{FORMAT}-journal-owner-v1"

PREFLIGHT_NAME = "differential-novel-sol-judge-preflight-v1.json"
RELEASE_NAME = "differential-novel-sol-judge-provider-release-v1.json"
JUDGE_NAME = "differential-novel-sol-judge-run-v1.json"
JUDGE_REPLAY_NAME = "differential-novel-sol-judge-replay-v1.json"
CHECKPOINT_DIR_NAME = "differential-novel-sol-judge-v1-calls"

DEFAULT_OUTPUT_ROOT = plan_cli.DEFAULT_OUTPUT_ROOT / "novel-sol-execution-v1"
DEFAULT_MODEL = plan_cli.DEFAULT_JUDGE_MODEL
DEFAULT_GATEWAY_URL = live.DEFAULT_GATEWAY_URL
DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_MAX_PROMPT_TOKENS = judging.DEFAULT_MAX_JUDGE_PROMPT_TOKENS

_JOURNAL_FILENAME_RE = re.compile(
    r"^(?P<key>[0-9a-f]{64})\.(?P<kind>request|response)\.json$"
)

PROMPT_ROW_KEYS = {
    "format",
    "judge_input_receipt_sha256",
    "judge_reuse_key_sha256",
    "messages",
    "messages_sha256",
    "ordinal",
    "prediction",
    "prediction_sha256",
    "prompt_row_receipt_sha256",
    "prompt_token_proxy",
    "question",
    "question_id",
    "question_sha256",
    "reference",
    "reference_sha256",
    "source_policy_row_sha256",
}
PREFLIGHT_KEYS = {
    "answer_policy_gold_loaded",
    "caller_ordinal_routing_available",
    "differential_plan_artifact_sha256",
    "format",
    "gateway_url",
    "gold_loaded",
    "judge_contract_sha256",
    "judge_input_population_sha256",
    "judge_model_identity_sha256",
    "max_concurrency",
    "max_new_tokens",
    "max_prompt_tokens",
    "model",
    "novel_prompt_count",
    "physical_provider_calls",
    "production_ordinal_routing_enabled",
    "prompt_population",
    "prompt_population_sha256",
    "prompt_row_population_sha256",
    "prompt_rows",
    "required_authorized_provider_calls",
    "retained_transformer_token_state_bytes",
    "retry_count",
    "selected_ordinals",
    "source_policy_replay_artifact_sha256",
    "source_policy_run_artifact_sha256",
    "target_question_count",
}
RELEASE_KEYS = {
    "answer_policy_gold_loaded",
    "approval_opt_in",
    "caller_ordinal_routing_available",
    "checkpoint_root",
    "checkpoint_root_sha256",
    "differential_plan_artifact_sha256",
    "format",
    "gateway_url",
    "gold_loaded",
    "journal_owner_format",
    "journal_owner_identity_sha256",
    "judge_contract_sha256",
    "judge_output_root",
    "judge_output_root_sha256",
    "max_concurrency",
    "model",
    "preflight_artifact_sha256",
    "production_ordinal_routing_enabled",
    "prompt_population_sha256",
    "provider_calls_during_release",
    "release_identity_sha256",
    "release_status",
    "required_authorized_provider_calls",
    "retained_transformer_token_state_bytes",
    "retry_count",
    "selected_ordinals",
    "unsafe_retry_policy",
}
JUDGE_ROW_KEYS = {
    "call_key_sha256",
    "correct",
    "format",
    "judge_input_receipt_sha256",
    "judge_output",
    "judge_output_sha256",
    "judge_reuse_key_sha256",
    "judge_row_sha256",
    "messages_sha256",
    "ordinal",
    "prediction_sha256",
    "prompt_row_receipt_sha256",
    "question_id",
    "question_sha256",
    "reference_sha256",
    "request_journal_sha256",
    "response_journal_sha256",
    "source_policy_row_sha256",
}
JUDGE_KEYS = {
    "aggregate",
    "answer_policy_gold_loaded",
    "completion_batch",
    "differential_plan_artifact_sha256",
    "format",
    "gold_loaded",
    "journal_owner_identity_sha256",
    "judge_contract_sha256",
    "judge_model",
    "physical_provider_calls_during_materialization",
    "preflight_artifact_sha256",
    "prompt_population_sha256",
    "questions",
    "release_authorization_artifact_sha256",
    "retained_transformer_token_state_bytes",
    "selected_ordinals",
    "selected_question_count",
    "source_policy_replay_artifact_sha256",
    "source_policy_run_artifact_sha256",
}


class LockedDifferentialNovelSolJudgeError(MatchedEvalContractError):
    """A plan, release, journal, verdict, or replay escaped its seal."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedDifferentialNovelSolJudgeError(message)


def _canonical_root(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve(strict=False)))


def _self_hashed(
    raw: Mapping[str, Any], *, receipt_key: str, label: str
) -> dict[str, Any]:
    _require(type(raw) is dict, f"{label} changed type")
    body = dict(raw)
    declared = body.pop(receipt_key, None)
    _require(
        require_sha256(declared, label) == identity_sha256(body),
        f"{label} receipt changed",
    )
    return dict(raw)


def _plain_messages(raw: object, label: str) -> tuple[dict[str, str], ...]:
    _require(type(raw) is list and bool(raw), f"{label} changed type")
    result: list[dict[str, str]] = []
    for value in raw:
        _require(
            type(value) is dict
            and set(value) == {"role", "content"}
            and value.get("role") in {"system", "user", "assistant"}
            and type(value.get("content")) is str,
            f"{label} changed schema",
        )
        result.append({"role": value["role"], "content": value["content"]})
    return tuple(result)


def _load_plan(path: str | Path, expected_sha256: str) -> SealedArtifact:
    return plan_cli.load_verified_differential_judge_plan(path, expected_sha256)


def _validate_prompt_row(
    raw: Mapping[str, Any], *, model: str
) -> tuple[dict[str, Any], tuple[dict[str, str], ...]]:
    row = _self_hashed(
        raw,
        receipt_key="prompt_row_receipt_sha256",
        label="differential novel prompt row",
    )
    ordinal = row.get("ordinal")
    question = require_text(row.get("question"), "novel judge question")
    reference = require_text(row.get("reference"), "novel judge reference")
    prediction = require_text(row.get("prediction"), "novel judge prediction")
    messages = _plain_messages(row.get("messages"), "novel judge messages")
    expected_messages = tuple(
        dict(value) for value in build_judge_prompt(question, reference, prediction)
    )
    input_body = {
        "judge_contract_sha256": plan_cli.JUDGE_CONTRACT_SHA256,
        "judge_model": model,
        "ordinal": ordinal,
        "prediction_sha256": row.get("prediction_sha256"),
        "question_sha256": row.get("question_sha256"),
        "reference_sha256": row.get("reference_sha256"),
    }
    _require(
        set(row) == PROMPT_ROW_KEYS
        and row.get("format") == plan_cli.NOVEL_PROMPT_FORMAT
        and type(ordinal) is int
        and 0 <= ordinal < plan_cli.QUESTION_COUNT
        and row.get("question_sha256") == quote_sha256(question)
        and row.get("reference_sha256") == quote_sha256(reference)
        and row.get("prediction_sha256") == quote_sha256(prediction)
        and messages == expected_messages
        and row.get("messages_sha256") == identity_sha256(list(messages))
        and row.get("prompt_token_proxy")
        == plan_cli.count_chat_prompt_token_proxy(messages)
        and row.get("judge_input_receipt_sha256") == identity_sha256(input_body),
        f"differential novel prompt {ordinal} changed",
    )
    for key in (
        "judge_input_receipt_sha256",
        "judge_reuse_key_sha256",
        "messages_sha256",
        "prediction_sha256",
        "question_sha256",
        "reference_sha256",
        "source_policy_row_sha256",
    ):
        require_sha256(row.get(key), f"novel prompt {key}")
    return row, messages


def build_preflight_payload(
    plan: SealedArtifact,
    *,
    model: str = DEFAULT_MODEL,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> dict[str, Any]:
    """Project the plan's exact novel rows into a locked execution preflight."""

    plan_cli.validate_differential_judge_plan(plan.payload)
    rows = tuple(dict(row) for row in plan.payload["novel_prompt_rows"])
    _require(
        rows
        and model == plan.payload.get("judge_model") == DEFAULT_MODEL
        and gateway_url == DEFAULT_GATEWAY_URL
        and type(max_concurrency) is int
        and max_concurrency > 0,
        "differential novel preflight population or runtime policy changed",
    )
    validated = tuple(_validate_prompt_row(row, model=model) for row in rows)
    prompt_rows = tuple(row for row, _messages in validated)
    prompts = tuple(messages for _row, messages in validated)
    ordinals = tuple(int(row["ordinal"]) for row in prompt_rows)
    _require(
        ordinals == tuple(sorted(set(ordinals))),
        "differential novel prompt order changed",
    )
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=DEFAULT_MAX_PROMPT_TOKENS
    )
    _require(
        population.logical_prompt_count
        == population.unique_prompt_count
        == len(prompt_rows)
        == plan.payload.get("novel_prompt_count")
        and all(
            receipt.messages_sha256 == row["messages_sha256"]
            and receipt.prompt_token_proxy == row["prompt_token_proxy"]
            for receipt, row in zip(
                population.ordered_rows, prompt_rows, strict=True
            )
        ),
        "differential novel prompts are not one unique sealed call per row",
    )
    return {
        "answer_policy_gold_loaded": False,
        "caller_ordinal_routing_available": False,
        "differential_plan_artifact_sha256": plan.sha256,
        "format": PREFLIGHT_FORMAT,
        "gateway_url": gateway_url,
        "gold_loaded": True,
        "judge_contract_sha256": plan_cli.JUDGE_CONTRACT_SHA256,
        "judge_input_population_sha256": plan.payload[
            "judge_input_population_sha256"
        ],
        "judge_model_identity_sha256": plan.payload[
            "judge_model_identity_sha256"
        ],
        "max_concurrency": max_concurrency,
        "max_new_tokens": JUDGE_MAX_TOKENS,
        "max_prompt_tokens": DEFAULT_MAX_PROMPT_TOKENS,
        "model": model,
        "novel_prompt_count": len(prompt_rows),
        "physical_provider_calls": 0,
        "production_ordinal_routing_enabled": False,
        "prompt_population": population.model_dump(),
        "prompt_population_sha256": population.prompt_population_sha256,
        "prompt_row_population_sha256": identity_sha256(
            [row["prompt_row_receipt_sha256"] for row in prompt_rows]
        ),
        "prompt_rows": list(prompt_rows),
        "required_authorized_provider_calls": len(prompt_rows),
        "retained_transformer_token_state_bytes": 0,
        "retry_count": 0,
        "selected_ordinals": list(ordinals),
        "source_policy_replay_artifact_sha256": plan.payload[
            "source_policy_replay_artifact_sha256"
        ],
        "source_policy_run_artifact_sha256": plan.payload[
            "source_policy_run_artifact_sha256"
        ],
        "target_question_count": plan_cli.QUESTION_COUNT,
    }


def validate_preflight_artifact(
    artifact: SealedArtifact, *, plan: SealedArtifact | None = None
) -> tuple[tuple[tuple[dict[str, str], ...], ...], tuple[dict[str, Any], ...]]:
    payload = artifact.payload
    raw_rows = payload.get("prompt_rows")
    count = payload.get("novel_prompt_count")
    _require(
        set(payload) == PREFLIGHT_KEYS
        and payload.get("format") == PREFLIGHT_FORMAT
        and payload.get("gold_loaded") is True
        and payload.get("answer_policy_gold_loaded") is False
        and payload.get("caller_ordinal_routing_available") is False
        and payload.get("production_ordinal_routing_enabled") is False
        and payload.get("physical_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("retry_count") == 0
        and payload.get("model") == DEFAULT_MODEL
        and payload.get("judge_contract_sha256")
        == plan_cli.JUDGE_CONTRACT_SHA256
        and payload.get("judge_model_identity_sha256")
        == identity_sha256({"model": DEFAULT_MODEL})
        and payload.get("gateway_url") == DEFAULT_GATEWAY_URL
        and type(payload.get("max_concurrency")) is int
        and int(payload["max_concurrency"]) > 0
        and payload.get("max_new_tokens") == JUDGE_MAX_TOKENS
        and payload.get("max_prompt_tokens") == DEFAULT_MAX_PROMPT_TOKENS
        and payload.get("target_question_count") == plan_cli.QUESTION_COUNT
        and type(count) is int
        and 0 < count <= plan_cli.QUESTION_COUNT
        and payload.get("required_authorized_provider_calls") == count
        and type(raw_rows) is list
        and len(raw_rows) == count,
        "differential novel sealed preflight changed",
    )
    for key in (
        "differential_plan_artifact_sha256",
        "judge_input_population_sha256",
        "judge_model_identity_sha256",
        "prompt_population_sha256",
        "prompt_row_population_sha256",
        "source_policy_replay_artifact_sha256",
        "source_policy_run_artifact_sha256",
    ):
        require_sha256(payload.get(key), f"differential preflight {key}")
    validated = tuple(
        _validate_prompt_row(row, model=DEFAULT_MODEL) for row in raw_rows
    )
    rows = tuple(row for row, _messages in validated)
    prompts = tuple(messages for _row, messages in validated)
    ordinals = tuple(int(row["ordinal"]) for row in rows)
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=DEFAULT_MAX_PROMPT_TOKENS
    )
    _require(
        ordinals == tuple(sorted(set(ordinals)))
        and payload.get("selected_ordinals") == list(ordinals)
        and payload.get("prompt_row_population_sha256")
        == identity_sha256([row["prompt_row_receipt_sha256"] for row in rows])
        and population.logical_prompt_count
        == population.unique_prompt_count
        == count
        and payload.get("prompt_population") == population.model_dump()
        and payload.get("prompt_population_sha256")
        == population.prompt_population_sha256,
        "differential novel preflight prompt population changed",
    )
    if plan is not None:
        plan_cli.validate_differential_judge_plan(plan.payload)
        _require(
            payload.get("differential_plan_artifact_sha256") == plan.sha256
            and payload.get("source_policy_run_artifact_sha256")
            == plan.payload.get("source_policy_run_artifact_sha256")
            and payload.get("source_policy_replay_artifact_sha256")
            == plan.payload.get("source_policy_replay_artifact_sha256")
            and payload.get("judge_input_population_sha256")
            == plan.payload.get("judge_input_population_sha256")
            and payload.get("judge_model_identity_sha256")
            == plan.payload.get("judge_model_identity_sha256")
            and list(rows) == plan.payload.get("novel_prompt_rows"),
            "differential novel preflight differs from its sealed plan",
        )
    return prompts, rows


def _read_preflight(
    output_root: str | Path, expected_sha256: str, *, plan: SealedArtifact | None = None
) -> tuple[SealedArtifact, tuple[tuple[dict[str, str], ...], ...], tuple[dict[str, Any], ...]]:
    artifact = read_sealed_json(Path(output_root) / PREFLIGHT_NAME)
    _require(
        artifact.sha256 == require_sha256(expected_sha256, "novel judge preflight"),
        "differential novel preflight artifact changed",
    )
    prompts, rows = validate_preflight_artifact(artifact, plan=plan)
    return artifact, prompts, rows


def _journal_owner_body(
    preflight: SealedArtifact, *, output_root: str | Path
) -> dict[str, Any]:
    judge_root = _canonical_root(output_root)
    checkpoint_root = _canonical_root(Path(output_root) / CHECKPOINT_DIR_NAME)
    return {
        "checkpoint_root": checkpoint_root,
        "checkpoint_root_sha256": identity_sha256(
            {"canonical_root": checkpoint_root}
        ),
        "differential_plan_artifact_sha256": preflight.payload[
            "differential_plan_artifact_sha256"
        ],
        "format": JOURNAL_OWNER_FORMAT,
        "judge_output_root": judge_root,
        "judge_output_root_sha256": identity_sha256({"canonical_root": judge_root}),
        "model": preflight.payload["model"],
        "preflight_artifact_sha256": preflight.sha256,
        "prompt_population_sha256": preflight.payload[
            "prompt_population_sha256"
        ],
        "required_authorized_provider_calls": preflight.payload[
            "required_authorized_provider_calls"
        ],
    }


def _release_payload(
    preflight: SealedArtifact, *, output_root: str | Path
) -> dict[str, Any]:
    owner = _journal_owner_body(preflight, output_root=output_root)
    body = {
        "answer_policy_gold_loaded": False,
        "approval_opt_in": True,
        "caller_ordinal_routing_available": False,
        "checkpoint_root": owner["checkpoint_root"],
        "checkpoint_root_sha256": owner["checkpoint_root_sha256"],
        "differential_plan_artifact_sha256": preflight.payload[
            "differential_plan_artifact_sha256"
        ],
        "format": RELEASE_FORMAT,
        "gateway_url": preflight.payload["gateway_url"],
        "gold_loaded": True,
        "journal_owner_format": JOURNAL_OWNER_FORMAT,
        "journal_owner_identity_sha256": identity_sha256(owner),
        "judge_contract_sha256": plan_cli.JUDGE_CONTRACT_SHA256,
        "judge_output_root": owner["judge_output_root"],
        "judge_output_root_sha256": owner["judge_output_root_sha256"],
        "max_concurrency": preflight.payload["max_concurrency"],
        "model": preflight.payload["model"],
        "preflight_artifact_sha256": preflight.sha256,
        "production_ordinal_routing_enabled": False,
        "prompt_population_sha256": preflight.payload[
            "prompt_population_sha256"
        ],
        "provider_calls_during_release": 0,
        "release_status": "approved_for_provider_execution",
        "required_authorized_provider_calls": preflight.payload[
            "required_authorized_provider_calls"
        ],
        "retained_transformer_token_state_bytes": 0,
        "retry_count": 0,
        "selected_ordinals": list(preflight.payload["selected_ordinals"]),
        "unsafe_retry_policy": "refuse_incomplete_request_response_pair",
    }
    return {**body, "release_identity_sha256": identity_sha256(body)}


def _validate_release(
    artifact: SealedArtifact,
    *,
    preflight: SealedArtifact,
    output_root: str | Path,
) -> dict[str, Any]:
    payload = artifact.payload
    body = dict(payload)
    declared = body.pop("release_identity_sha256", None)
    owner = _journal_owner_body(preflight, output_root=output_root)
    _require(
        set(payload) == RELEASE_KEYS
        and require_sha256(declared, "novel judge release")
        == identity_sha256(body)
        and payload.get("format") == RELEASE_FORMAT
        and payload.get("release_status") == "approved_for_provider_execution"
        and payload.get("approval_opt_in") is True
        and payload.get("gold_loaded") is True
        and payload.get("answer_policy_gold_loaded") is False
        and payload.get("provider_calls_during_release") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("retry_count") == 0
        and payload.get("unsafe_retry_policy")
        == "refuse_incomplete_request_response_pair"
        and payload.get("caller_ordinal_routing_available") is False
        and payload.get("production_ordinal_routing_enabled") is False
        and payload.get("preflight_artifact_sha256") == preflight.sha256
        and payload.get("differential_plan_artifact_sha256")
        == preflight.payload.get("differential_plan_artifact_sha256")
        and payload.get("model") == preflight.payload.get("model") == DEFAULT_MODEL
        and payload.get("gateway_url")
        == preflight.payload.get("gateway_url")
        == DEFAULT_GATEWAY_URL
        and payload.get("max_concurrency")
        == preflight.payload.get("max_concurrency")
        and payload.get("prompt_population_sha256")
        == preflight.payload.get("prompt_population_sha256")
        and payload.get("required_authorized_provider_calls")
        == preflight.payload.get("required_authorized_provider_calls")
        and payload.get("selected_ordinals")
        == preflight.payload.get("selected_ordinals")
        and payload.get("judge_contract_sha256")
        == plan_cli.JUDGE_CONTRACT_SHA256
        and payload.get("journal_owner_format") == JOURNAL_OWNER_FORMAT
        and all(
            payload.get(key) == value
            for key, value in owner.items()
            if key != "format"
        )
        and payload.get("journal_owner_identity_sha256")
        == identity_sha256(owner),
        "differential novel provider release changed",
    )
    return payload


def _read_release(
    output_root: str | Path,
    expected_sha256: str,
    *,
    preflight: SealedArtifact,
) -> SealedArtifact:
    artifact = read_sealed_json(Path(output_root) / RELEASE_NAME)
    _require(
        artifact.sha256 == require_sha256(expected_sha256, "novel judge release"),
        "differential novel release artifact changed",
    )
    _validate_release(artifact, preflight=preflight, output_root=output_root)
    return artifact


def _runtime(
    preflight: SealedArtifact,
    release: SealedArtifact,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    output_root: str | Path,
    client: Any | None,
) -> FastCompletionRuntime:
    _validate_release(release, preflight=preflight, output_root=output_root)
    count = int(preflight.payload["required_authorized_provider_calls"])
    _require(
        len(prompts) == count,
        "differential novel runtime prompt population changed",
    )
    return FastCompletionRuntime(
        checkpoint_dir=Path(output_root) / CHECKPOINT_DIR_NAME,
        prompt_population=prompts,
        model=DEFAULT_MODEL,
        client=client,
        max_prompt_tokens=DEFAULT_MAX_PROMPT_TOKENS,
        max_new_tokens=JUDGE_MAX_TOKENS,
        max_concurrency=int(preflight.payload["max_concurrency"]),
        retries=0,
        benchmark_provenance={
            "arm": FORMAT,
            "authorized_unique_calls": count,
            "differential_plan_artifact_sha256": preflight.payload[
                "differential_plan_artifact_sha256"
            ],
            "experiment_format": JUDGE_FORMAT,
            "journal_owner_identity_sha256": release.payload[
                "journal_owner_identity_sha256"
            ],
            "judge_contract_sha256": plan_cli.JUDGE_CONTRACT_SHA256,
            "preflight_artifact_sha256": preflight.sha256,
            "release_authorization_artifact_sha256": release.sha256,
        },
    )


def _checkpoint_batch(
    preflight: SealedArtifact,
    release: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    output_root: str | Path,
    client: Any | None,
) -> FastCompletionBatch:
    runtime = _runtime(
        preflight, release, prompts, output_root=output_root, client=client
    )
    try:
        return runtime.run()
    finally:
        runtime.close()


def _read_only_checkpoint_count(
    output_root: str | Path, *, maximum: int
) -> int:
    root = Path(output_root) / CHECKPOINT_DIR_NAME
    if not root.exists():
        return 0
    _require(
        root.is_dir() and not root.is_symlink(),
        "differential novel checkpoint root must be a regular directory",
    )
    requests: set[str] = set()
    responses: set[str] = set()
    for path in root.iterdir():
        _require(
            path.is_file() and not path.is_symlink(),
            "differential novel checkpoint root contains foreign state",
        )
        if path.name == ".fast-completion-journal.lock":
            continue
        match = _JOURNAL_FILENAME_RE.fullmatch(path.name)
        _require(
            match is not None,
            "differential novel checkpoint root contains foreign journal state",
        )
        assert match is not None
        target = requests if match.group("kind") == "request" else responses
        target.add(match.group("key"))
    _require(
        requests == responses,
        "differential novel checkpoint pair is incomplete; unsafe retry forbidden",
    )
    _require(
        len(requests) <= maximum,
        "differential novel checkpoint population exceeds the sealed plan",
    )
    return len(requests)


def _validated_checkpoint_hits(
    preflight: SealedArtifact,
    release: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    output_root: str | Path,
) -> int:
    root = Path(output_root) / CHECKPOINT_DIR_NAME
    if not root.exists():
        return 0
    runtime = _runtime(
        preflight, release, prompts, output_root=output_root, client=None
    )
    try:
        with runtime._journal_guard():  # noqa: SLF001 - runtime owns journals
            records = runtime._load_all_records()  # noqa: SLF001
    finally:
        runtime.close()
    return len(records)


def _complete_checkpoint_batch(
    preflight: SealedArtifact,
    release: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    output_root: str | Path,
) -> FastCompletionBatch:
    count = int(preflight.payload["required_authorized_provider_calls"])
    _require(
        _read_only_checkpoint_count(output_root, maximum=count) == count,
        "differential novel materialization requires complete checkpoints",
    )
    return _checkpoint_batch(
        preflight, release, prompts, output_root=output_root, client=None
    )


def _judge_payload(
    preflight: SealedArtifact,
    release: SealedArtifact,
    prompt_rows: tuple[dict[str, Any], ...],
    batch: FastCompletionBatch,
) -> dict[str, Any]:
    count = len(prompt_rows)
    _require(
        batch.usage.logical_calls
        == batch.usage.unique_calls
        == batch.usage.checkpoint_hits
        == count
        and batch.usage.physical_calls == 0
        and len(batch.logical_completions) == count
        and len(batch.unique_records) == count,
        "differential novel materialization requires checkpoint-only completion",
    )
    records = {record.messages_sha256: record for record in batch.unique_records}
    _require(len(records) == count, "differential novel completion IDs repeat")
    rows: list[dict[str, Any]] = []
    for prompt, completion in zip(
        prompt_rows, batch.logical_completions, strict=True
    ):
        record = records.get(str(prompt["messages_sha256"]))
        _require(
            record is not None
            and record.completion == completion
            and record.checkpoint_hit is True
            and record.physical_call is False
            and record.requested_model == DEFAULT_MODEL,
            "differential novel checkpoint record changed",
        )
        try:
            correct = parse_binary_judge_verdict(completion)
        except RuntimeError as exc:
            raise LockedDifferentialNovelSolJudgeError(
                "differential novel Sol returned an invalid binary verdict"
            ) from exc
        assert record is not None
        body = {
            "call_key_sha256": record.call_key_sha256,
            "correct": correct,
            "format": JUDGE_ROW_FORMAT,
            "judge_input_receipt_sha256": prompt[
                "judge_input_receipt_sha256"
            ],
            "judge_output": completion,
            "judge_output_sha256": quote_sha256(completion),
            "judge_reuse_key_sha256": prompt["judge_reuse_key_sha256"],
            "messages_sha256": prompt["messages_sha256"],
            "ordinal": prompt["ordinal"],
            "prediction_sha256": prompt["prediction_sha256"],
            "prompt_row_receipt_sha256": prompt[
                "prompt_row_receipt_sha256"
            ],
            "question_id": prompt["question_id"],
            "question_sha256": prompt["question_sha256"],
            "reference_sha256": prompt["reference_sha256"],
            "request_journal_sha256": record.request_journal_sha256,
            "response_journal_sha256": record.response_journal_sha256,
            "source_policy_row_sha256": prompt["source_policy_row_sha256"],
        }
        rows.append({**body, "judge_row_sha256": identity_sha256(body)})
    correct_count = sum(bool(row["correct"]) for row in rows)
    return {
        "aggregate": {
            "accuracy": correct_count / count,
            "correct": correct_count,
            "question_count": count,
        },
        "answer_policy_gold_loaded": False,
        "completion_batch": judging._stable_batch(batch),  # noqa: SLF001
        "differential_plan_artifact_sha256": preflight.payload[
            "differential_plan_artifact_sha256"
        ],
        "format": JUDGE_FORMAT,
        "gold_loaded": True,
        "journal_owner_identity_sha256": release.payload[
            "journal_owner_identity_sha256"
        ],
        "judge_contract_sha256": plan_cli.JUDGE_CONTRACT_SHA256,
        "judge_model": DEFAULT_MODEL,
        "physical_provider_calls_during_materialization": 0,
        "preflight_artifact_sha256": preflight.sha256,
        "prompt_population_sha256": preflight.payload[
            "prompt_population_sha256"
        ],
        "questions": rows,
        "release_authorization_artifact_sha256": release.sha256,
        "retained_transformer_token_state_bytes": 0,
        "selected_ordinals": list(preflight.payload["selected_ordinals"]),
        "selected_question_count": count,
        "source_policy_replay_artifact_sha256": preflight.payload[
            "source_policy_replay_artifact_sha256"
        ],
        "source_policy_run_artifact_sha256": preflight.payload[
            "source_policy_run_artifact_sha256"
        ],
    }


def _validate_judge(
    preflight: SealedArtifact,
    release: SealedArtifact,
    payload: Mapping[str, Any],
    *,
    expected_batch: FastCompletionBatch | None = None,
) -> tuple[dict[str, Any], ...]:
    raw_rows = payload.get("questions")
    aggregate = payload.get("aggregate")
    count = int(preflight.payload["required_authorized_provider_calls"])
    _require(
        set(payload) == JUDGE_KEYS
        and payload.get("format") == JUDGE_FORMAT
        and payload.get("gold_loaded") is True
        and payload.get("answer_policy_gold_loaded") is False
        and payload.get("physical_provider_calls_during_materialization") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("preflight_artifact_sha256") == preflight.sha256
        and payload.get("release_authorization_artifact_sha256") == release.sha256
        and payload.get("differential_plan_artifact_sha256")
        == preflight.payload.get("differential_plan_artifact_sha256")
        and payload.get("source_policy_run_artifact_sha256")
        == preflight.payload.get("source_policy_run_artifact_sha256")
        and payload.get("source_policy_replay_artifact_sha256")
        == preflight.payload.get("source_policy_replay_artifact_sha256")
        and payload.get("journal_owner_identity_sha256")
        == release.payload.get("journal_owner_identity_sha256")
        and payload.get("judge_contract_sha256")
        == plan_cli.JUDGE_CONTRACT_SHA256
        and payload.get("judge_model") == DEFAULT_MODEL
        and payload.get("prompt_population_sha256")
        == preflight.payload.get("prompt_population_sha256")
        and payload.get("selected_ordinals")
        == preflight.payload.get("selected_ordinals")
        and payload.get("selected_question_count") == count
        and type(raw_rows) is list
        and len(raw_rows) == count
        and type(aggregate) is dict
        and aggregate.get("question_count") == count,
        "differential novel judge envelope changed",
    )
    if expected_batch is not None:
        _require(
            payload.get("completion_batch") == judging._stable_batch(expected_batch),  # noqa: SLF001
            "differential novel completion batch changed",
        )
    prompt_by_ordinal = {
        int(row["ordinal"]): row for row in preflight.payload["prompt_rows"]
    }
    rows: list[dict[str, Any]] = []
    call_keys: list[str] = []
    for raw in raw_rows:
        row = _self_hashed(
            raw,
            receipt_key="judge_row_sha256",
            label="differential novel judge row",
        )
        ordinal = row.get("ordinal")
        prompt = prompt_by_ordinal.get(ordinal)
        output = row.get("judge_output")
        try:
            parsed = parse_binary_judge_verdict(
                require_text(output, "differential novel judge output")
            )
        except RuntimeError as exc:
            raise LockedDifferentialNovelSolJudgeError(
                "differential novel judge output is not binary"
            ) from exc
        _require(
            set(row) == JUDGE_ROW_KEYS
            and prompt is not None
            and type(row.get("correct")) is bool
            and parsed is row.get("correct")
            and row.get("judge_output_sha256") == quote_sha256(str(output))
            and all(
                row.get(key) == prompt.get(key)
                for key in (
                    "judge_input_receipt_sha256",
                    "judge_reuse_key_sha256",
                    "messages_sha256",
                    "ordinal",
                    "prediction_sha256",
                    "prompt_row_receipt_sha256",
                    "question_id",
                    "question_sha256",
                    "reference_sha256",
                    "source_policy_row_sha256",
                )
            ),
            f"differential novel verdict {ordinal} changed",
        )
        for key in (
            "call_key_sha256",
            "judge_input_receipt_sha256",
            "judge_output_sha256",
            "judge_reuse_key_sha256",
            "messages_sha256",
            "prediction_sha256",
            "prompt_row_receipt_sha256",
            "question_sha256",
            "reference_sha256",
            "request_journal_sha256",
            "response_journal_sha256",
            "source_policy_row_sha256",
        ):
            require_sha256(row.get(key), f"differential verdict {key}")
        call_keys.append(str(row["call_key_sha256"]))
        rows.append(row)
    correct_count = sum(bool(row["correct"]) for row in rows)
    _require(
        tuple(row["ordinal"] for row in rows)
        == tuple(preflight.payload["selected_ordinals"])
        and len(set(call_keys)) == count
        and aggregate.get("correct") == correct_count
        and aggregate.get("accuracy") == correct_count / count,
        "differential novel judge population or arithmetic changed",
    )
    return tuple(rows)


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    _require(
        not (output_root / CHECKPOINT_DIR_NAME).exists(),
        "differential novel preflight requires a fresh absent checkpoint root",
    )
    plan = _load_plan(args.plan, str(args.expected_plan_sha256))
    payload = build_preflight_payload(
        plan,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    artifact, created = publish_sealed_json(output_root / PREFLIGHT_NAME, payload)
    return {
        "created": created,
        "differential_plan_sha256": plan.sha256,
        "physical_provider_calls": 0,
        "preflight_sha256": artifact.sha256,
        "required_authorized_provider_calls": payload[
            "required_authorized_provider_calls"
        ],
        "selected_ordinals": payload["selected_ordinals"],
    }


def run_approve_release(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    _require(
        args.approve_provider_release is True,
        "differential novel release requires explicit provider approval",
    )
    _require(
        not (output_root / CHECKPOINT_DIR_NAME).exists(),
        "differential novel release requires an absent checkpoint root",
    )
    plan = _load_plan(args.plan, str(args.expected_plan_sha256))
    preflight, _prompts, _rows = _read_preflight(
        output_root, str(args.expected_preflight_sha256), plan=plan
    )
    _require(
        str(args.model) == preflight.payload.get("model") == DEFAULT_MODEL
        and str(args.gateway_url)
        == preflight.payload.get("gateway_url")
        == DEFAULT_GATEWAY_URL
        and int(args.max_concurrency) == preflight.payload.get("max_concurrency"),
        "differential novel release runtime differs from preflight",
    )
    payload = _release_payload(preflight, output_root=output_root)
    artifact, created = publish_sealed_json(output_root / RELEASE_NAME, payload)
    return {
        "created": created,
        "journal_owner_identity_sha256": payload[
            "journal_owner_identity_sha256"
        ],
        "physical_provider_calls": 0,
        "preflight_sha256": preflight.sha256,
        "release_sha256": artifact.sha256,
        "required_authorized_provider_calls": payload[
            "required_authorized_provider_calls"
        ],
    }


def _runtime_args_match(args: argparse.Namespace, preflight: SealedArtifact) -> None:
    _require(
        str(args.model) == preflight.payload.get("model") == DEFAULT_MODEL
        and str(args.gateway_url)
        == preflight.payload.get("gateway_url")
        == DEFAULT_GATEWAY_URL
        and int(args.max_concurrency) == preflight.payload.get("max_concurrency"),
        "differential novel runtime settings differ from preflight",
    )


def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    preflight, prompts, _rows = _read_preflight(
        output_root, str(args.expected_preflight_sha256)
    )
    release = _read_release(
        output_root, str(args.expected_release_sha256), preflight=preflight
    )
    _runtime_args_match(args, preflight)
    count = int(preflight.payload["required_authorized_provider_calls"])
    _require(
        args.enable_provider is True
        and type(args.authorized_provider_calls) is int
        and 0 <= args.authorized_provider_calls <= count,
        "differential novel provider requires bounded Sol authorization",
    )
    candidate_hits = _read_only_checkpoint_count(output_root, maximum=count)
    remaining = count - candidate_hits
    _require(
        args.authorized_provider_calls == remaining,
        "differential novel authorization must exactly equal remaining calls",
    )
    checkpoint_hits = _validated_checkpoint_hits(
        preflight, release, prompts, output_root=output_root
    )
    _require(
        checkpoint_hits == candidate_hits,
        "differential novel checkpoint count changed after authorization",
    )
    if remaining == 0:
        batch = _checkpoint_batch(
            preflight, release, prompts, output_root=output_root, client=None
        )
    else:
        load_dotenv()
        api_key = os.environ.get(str(args.api_key_env), "").strip()
        _require(bool(api_key), f"provider API key is empty: {args.api_key_env}")
        client = judging._make_provider_client(  # noqa: SLF001
            api_key, str(args.gateway_url)
        )
        try:
            batch = _checkpoint_batch(
                preflight,
                release,
                prompts,
                output_root=output_root,
                client=client,
            )
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
    _require(
        batch.usage.logical_calls == batch.usage.unique_calls == count
        and batch.usage.physical_calls + batch.usage.checkpoint_hits == count
        and batch.usage.physical_calls <= remaining
        and batch.usage.checkpoint_hits >= checkpoint_hits,
        "differential novel provider population changed",
    )
    return {
        "authorized_remaining_provider_calls": remaining,
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "physical_provider_calls": batch.usage.physical_calls,
        "preflight_sha256": preflight.sha256,
        "release_sha256": release.sha256,
        "required_authorized_provider_calls": remaining,
    }


def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    preflight, prompts, rows = _read_preflight(
        output_root, str(args.expected_preflight_sha256)
    )
    release = _read_release(
        output_root, str(args.expected_release_sha256), preflight=preflight
    )
    _runtime_args_match(args, preflight)
    batch = _complete_checkpoint_batch(
        preflight, release, prompts, output_root=output_root
    )
    payload = _judge_payload(preflight, release, rows, batch)
    _validate_judge(preflight, release, payload, expected_batch=batch)
    artifact, created = publish_sealed_json(output_root / JUDGE_NAME, payload)
    return {
        "checkpoint_hits": len(rows),
        "correct": payload["aggregate"]["correct"],
        "created": created,
        "judge_sha256": artifact.sha256,
        "physical_provider_calls": 0,
        "selected_question_count": len(rows),
    }


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    preflight, prompts, rows = _read_preflight(
        output_root, str(args.expected_preflight_sha256)
    )
    release = _read_release(
        output_root, str(args.expected_release_sha256), preflight=preflight
    )
    _runtime_args_match(args, preflight)
    batch = _complete_checkpoint_batch(
        preflight, release, prompts, output_root=output_root
    )
    rebuilt = _judge_payload(preflight, release, rows, batch)
    _validate_judge(preflight, release, rebuilt, expected_batch=batch)
    judge = read_sealed_json(output_root / JUDGE_NAME)
    _require(
        judge.sha256 == require_sha256(args.expected_judge_sha256, "novel judge")
        and judge.payload == rebuilt,
        "differential novel judge differs from checkpoint-only replay",
    )
    replay, _created = publish_sealed_json(
        output_root / JUDGE_REPLAY_NAME, rebuilt
    )
    _require(
        replay.sha256 == judge.sha256,
        "differential novel judge replay is not byte-identical",
    )
    return {
        "byte_identical": True,
        "judge_replay_sha256": replay.sha256,
        "judge_sha256": judge.sha256,
        "physical_provider_calls": 0,
    }


def load_verified_novel_judge_run(
    output_root: str | Path,
    *,
    plan_path: str | Path,
    expected_plan_sha256: str,
    expected_preflight_sha256: str,
    expected_release_sha256: str,
    expected_judge_sha256: str,
    expected_replay_sha256: str,
) -> plan_cli.AuthenticatedJudgeRun:
    """Authenticate the complete lifecycle and return the differential seam."""

    root = Path(output_root)
    plan = _load_plan(plan_path, expected_plan_sha256)
    preflight, prompts, prompt_rows = _read_preflight(
        root, expected_preflight_sha256, plan=plan
    )
    release = _read_release(root, expected_release_sha256, preflight=preflight)
    judge = read_sealed_json(root / JUDGE_NAME)
    replay = read_sealed_json(root / JUDGE_REPLAY_NAME)
    _require(
        judge.sha256 == require_sha256(expected_judge_sha256, "novel judge")
        and replay.sha256
        == require_sha256(expected_replay_sha256, "novel judge replay")
        == judge.sha256
        and replay.payload == judge.payload,
        "differential novel judge/replay artifacts changed",
    )
    batch = _complete_checkpoint_batch(
        preflight, release, prompts, output_root=root
    )
    rebuilt = _judge_payload(preflight, release, prompt_rows, batch)
    _require(
        rebuilt == judge.payload,
        "differential novel judge differs from authenticated journals",
    )
    rows = _validate_judge(preflight, release, judge.payload, expected_batch=batch)
    authenticated = plan_cli.authenticate_prior_judge_run(
        preflight, judge, replay
    )
    expected_prompts = tuple(plan.payload["novel_prompt_rows"])
    _require(
        authenticated.model == DEFAULT_MODEL
        and tuple(row["ordinal"] for row in authenticated.entries)
        == tuple(row["ordinal"] for row in expected_prompts)
        == tuple(row["ordinal"] for row in rows)
        and all(
            all(entry[key] == prompt[key] for key in (
                "prediction_sha256",
                "question_id",
                "question_sha256",
                "reference_sha256",
            ))
            for entry, prompt in zip(
                authenticated.entries, expected_prompts, strict=True
            )
        ),
        "differential novel authenticated seam differs from its plan",
    )
    return authenticated


def _add_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument(
        "--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY
    )


def _add_plan(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight")
    _add_runtime(preflight)
    _add_plan(preflight)

    approve = commands.add_parser("approve-release")
    _add_runtime(approve)
    _add_plan(approve)
    approve.add_argument("--expected-preflight-sha256", required=True)
    approve.add_argument("--approve-provider-release", action="store_true")

    provider = commands.add_parser("provider-run")
    _add_runtime(provider)
    provider.add_argument("--expected-preflight-sha256", required=True)
    provider.add_argument("--expected-release-sha256", required=True)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, required=True)
    provider.add_argument("--api-key-env", default=live.DEFAULT_API_KEY_ENV)

    materialize = commands.add_parser("materialize")
    _add_runtime(materialize)
    materialize.add_argument("--expected-preflight-sha256", required=True)
    materialize.add_argument("--expected-release-sha256", required=True)

    replay = commands.add_parser("replay")
    _add_runtime(replay)
    replay.add_argument("--expected-preflight-sha256", required=True)
    replay.add_argument("--expected-release-sha256", required=True)
    replay.add_argument("--expected-judge-sha256", required=True)
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
    "DEFAULT_GATEWAY_URL",
    "DEFAULT_MODEL",
    "DEFAULT_OUTPUT_ROOT",
    "FORMAT",
    "JUDGE_FORMAT",
    "JUDGE_NAME",
    "JUDGE_REPLAY_NAME",
    "LockedDifferentialNovelSolJudgeError",
    "PREFLIGHT_FORMAT",
    "PREFLIGHT_NAME",
    "RELEASE_FORMAT",
    "RELEASE_NAME",
    "build_parser",
    "build_preflight_payload",
    "load_verified_novel_judge_run",
    "main",
    "run_approve_release",
    "run_materialize",
    "run_preflight",
    "run_provider",
    "run_replay",
    "validate_preflight_artifact",
]
