#!/usr/bin/env python3
"""Run the lean exact-11 current-artifact linked terminal repair.

The preflight authenticates the original R7 construction/replay and the
compiled A1 construction/replay, joins their eleven questions by question ID,
and seals one repaired provider prompt per question plus a byte-identical
preflight replay.  The preflight SHA and an authorization exactly equal to the
remaining complete checkpoints are the release gate; there is intentionally
no separate release artifact.

Provider journals are immutable and zero-retry.  Materialization and replay
are checkpoint-only and accept one strict JSON object containing exactly
``response_text`` and ``used_handle_ids``.  Gold, references, caller-selected
ordinals, and retained transformer token state are absent from this lifecycle.
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

from memory_condense.domain._tokenizer import (  # noqa: E402
    count_chat_prompt_token_proxy,
)
from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from memory_condense.eval.fast_completion_runtime import (  # noqa: E402
    FastCompletionBatch,
    FastCompletionRuntime,
    preflight_fast_completion_prompts,
)
from tools import run_r7_after_union_a1 as a1_cli  # noqa: E402
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
from tools.matched_eval.r7_linked_terminal_repair import (  # noqa: E402
    FORMAT as REPAIR_FORMAT,
    HARD_TOTAL_TOKEN_CAP,
    MAX_CHAT_PROMPT_TOKENS,
    OUTPUT_TOKEN_RESERVE,
    compile_r7_linked_terminal_repair,
)


FORMAT = "memory-condense-r7-linked-terminal-repair-lifecycle-v1"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight-v1"
RUN_FORMAT = f"{FORMAT}-run-v1"
RESULT_ROW_FORMAT = f"{FORMAT}-result-row-v1"
JUDGE_ROW_FORMAT = f"{FORMAT}-judge-row-v1"

PREFLIGHT_NAME = "r7-linked-terminal-repair-preflight-v1.json"
PREFLIGHT_REPLAY_NAME = "r7-linked-terminal-repair-preflight-replay-v1.json"
RUN_NAME = "r7-linked-terminal-repair-run-v1.json"
REPLAY_NAME = "r7-linked-terminal-repair-replay-v1.json"
CHECKPOINT_DIR_NAME = "terra-r7-linked-terminal-repair-v1-calls"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-semantic-global-terminal-v2-r7"
)
DEFAULT_A1_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/"
    "locked-r7-after-union-a1-compiled-temporal-effective-v1"
)
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-r7-linked-terminal-repair-v1"
)
SOURCE_NAME = "reduced-semantic-global-terminal-assay-v2.json"
SOURCE_REPLAY_NAME = "reduced-semantic-global-terminal-assay-replay-v2.json"
A1_NAME = a1_cli.CONSTRUCTION_NAME
A1_REPLAY_NAME = a1_cli.REPLAY_NAME

EXPECTED_SOURCE_SHA256 = (
    "120199b9f4cf912b0a2c2d0b56b228813393a533da010909a92b4cd6268406a5"
)
EXPECTED_A1_SHA256 = (
    "0da8ae97dd4931f90e4617b9dc09fb7cf99bbf3278e8e9e210f373c73ff52585"
)
SOURCE_FORMAT = "memory-condense-reduced-semantic-global-terminal-assay-v2"
A1_FORMAT = "memory-condense-r7-after-union-a1-preflight-v2"
DEFAULT_MODEL = live.DEFAULT_TERRA_GATEWAY_MODEL
DEFAULT_GATEWAY_URL = live.DEFAULT_GATEWAY_URL
DEFAULT_API_KEY_ENV = live.DEFAULT_API_KEY_ENV
DEFAULT_MAX_CONCURRENCY = 4
QUESTION_COUNT = 11
EXPECTED_RETAINED_LEAF_COUNT = 123

REPAIR_ROW_KEYS = {
    "allowed_handle_ids",
    "format",
    "hard_total_token_cap",
    "local_audit",
    "memory_representation",
    "messages",
    "messages_sha256",
    "new_provider_calls",
    "output_token_reserve",
    "presented_handle_ids",
    "prompt_token_proxy",
    "provider_input",
    "provider_input_sha256",
    "question_id",
    "question_sha256",
    "receipt_sha256",
    "retained_transformer_token_state_bytes",
}
PREFLIGHT_KEYS = {
    "a1_construction_artifact_sha256",
    "a1_replay_artifact_sha256",
    "construction_identity_sha256",
    "format",
    "gateway_url",
    "gold_loaded",
    "hard_total_token_cap",
    "max_concurrency",
    "model",
    "new_provider_calls",
    "ordinal_cli_routing_available",
    "output_token_reserve",
    "prompt_population",
    "prompt_population_sha256",
    "question_count",
    "question_population_sha256",
    "questions",
    "required_authorized_provider_calls",
    "retained_leaf_count",
    "retained_population_sha256",
    "retained_transformer_token_state_bytes",
    "source_construction_artifact_sha256",
    "source_replay_artifact_sha256",
}
RESULT_ROW_KEYS = {
    "call_key_sha256",
    "completion_sha256",
    "format",
    "messages_sha256",
    "parse_receipt_sha256",
    "prediction",
    "prediction_sha256",
    "preflight_question_receipt_sha256",
    "question_id",
    "question_sha256",
    "request_journal_sha256",
    "response_journal_sha256",
    "retained_transformer_token_state_bytes",
    "source_row_sha256",
    "used_handle_ids",
}
JUDGE_ROW_KEYS = {
    "dated_question_sha256",
    "format",
    "prediction",
    "prediction_sha256",
    "question_id",
    "question_sha256",
    "source_row_sha256",
}
RUN_KEYS = {
    "a1_construction_artifact_sha256",
    "a1_replay_artifact_sha256",
    "completion_batch",
    "format",
    "gold_loaded",
    "judge_row_population_sha256",
    "judge_rows",
    "physical_provider_calls_during_materialization",
    "preflight_construction_artifact_sha256",
    "preflight_replay_artifact_sha256",
    "prompt_population_sha256",
    "question_count",
    "result_count",
    "result_population_sha256",
    "results",
    "retained_leaf_count",
    "retained_population_sha256",
    "retained_transformer_token_state_bytes",
    "run_identity_sha256",
    "source_construction_artifact_sha256",
    "source_replay_artifact_sha256",
}

_JOURNAL_FILENAME_RE = re.compile(
    r"^(?P<key>[0-9a-f]{64})\.(?P<kind>request|response)\.json$"
)


class R7LinkedTerminalRepairRunnerError(MatchedEvalContractError):
    """The source pair, repair population, journal, or response changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise R7LinkedTerminalRepairRunnerError(message)


def _without(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    return {name: child for name, child in value.items() if name != key}


def _read_expected(path: str | Path, expected: str, label: str) -> SealedArtifact:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == require_sha256(expected, label),
        f"{label} artifact changed",
    )
    return artifact


def _load_source_pairs(
    *,
    source_root: str | Path,
    a1_root: str | Path,
    expected_source_sha256: str,
    expected_a1_sha256: str,
) -> tuple[SealedArtifact, SealedArtifact, SealedArtifact, SealedArtifact]:
    source_root = Path(source_root)
    a1_root = Path(a1_root)
    source = _read_expected(
        source_root / SOURCE_NAME,
        expected_source_sha256,
        "original R7 construction",
    )
    source_replay = _read_expected(
        source_root / SOURCE_REPLAY_NAME,
        expected_source_sha256,
        "original R7 replay",
    )
    a1 = _read_expected(
        a1_root / A1_NAME, expected_a1_sha256, "compiled A1 construction"
    )
    a1_replay = _read_expected(
        a1_root / A1_REPLAY_NAME, expected_a1_sha256, "compiled A1 replay"
    )
    _require(
        source.sha256 == source_replay.sha256
        and source.payload == source_replay.payload
        and a1.sha256 == a1_replay.sha256
        and a1.payload == a1_replay.payload,
        "repair source construction/replay pair is not byte-identical",
    )
    return source, source_replay, a1, a1_replay


def _validate_source_envelopes(
    source: SealedArtifact,
    source_replay: SealedArtifact,
    a1: SealedArtifact,
    a1_replay: SealedArtifact,
) -> tuple[tuple[dict[str, Any], ...], dict[str, dict[str, Any]]]:
    source_questions = source.payload.get("questions")
    a1_questions = a1.payload.get("questions")
    _require(
        source.sha256 == source_replay.sha256 == EXPECTED_SOURCE_SHA256
        and source.payload == source_replay.payload
        and a1.sha256 == a1_replay.sha256 == EXPECTED_A1_SHA256
        and a1.payload == a1_replay.payload
        and source.payload.get("format") == SOURCE_FORMAT
        and a1.payload.get("format") == A1_FORMAT
        and source.payload.get("gold_loaded") is False
        and a1.payload.get("gold_loaded") is False
        and source.payload.get("question_count") == QUESTION_COUNT
        and source.payload.get("terminal_answer_plan_count") == QUESTION_COUNT
        and a1.payload.get("question_count") == QUESTION_COUNT
        and a1.payload.get("expected_question_count") == QUESTION_COUNT
        and source.payload.get("new_provider_calls") == 0
        and a1.payload.get("provider_calls_performed_by_core") == 0
        and source.payload.get("retained_transformer_token_state_bytes") == 0
        and a1.payload.get("retained_transformer_token_state_bytes") == 0
        and a1.payload.get("source_artifact_sha256") == source.sha256
        and a1.payload.get("source_replay_artifact_sha256")
        == source_replay.sha256
        and type(source_questions) is list
        and type(a1_questions) is list
        and len(source_questions) == len(a1_questions) == QUESTION_COUNT
        and all(type(row) is dict for row in source_questions)
        and all(type(row) is dict for row in a1_questions),
        "repair source envelope changed",
    )
    ordered_a1 = tuple(dict(row) for row in a1_questions)
    source_by_id: dict[str, dict[str, Any]] = {}
    for raw in source_questions:
        question_id = require_text(raw.get("question_id"), "R7 question ID")
        _require(question_id not in source_by_id, "R7 question IDs repeat")
        source_by_id[question_id] = dict(raw)
    a1_ids = [
        require_text(row.get("question_id"), "A1 question ID") for row in ordered_a1
    ]
    _require(
        len(set(a1_ids)) == QUESTION_COUNT
        and set(a1_ids) == set(source_by_id),
        "A1/R7 question-ID join population changed",
    )
    return ordered_a1, source_by_id


def _validate_repair_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(raw)
    messages = row.get("messages")
    provider = row.get("provider_input")
    handles = row.get("allowed_handle_ids")
    presented = row.get("presented_handle_ids")
    _require(
        set(row) == REPAIR_ROW_KEYS
        and row.get("format") == REPAIR_FORMAT
        and row.get("receipt_sha256")
        == identity_sha256(_without(row, "receipt_sha256"))
        and row.get("hard_total_token_cap") == HARD_TOTAL_TOKEN_CAP
        and row.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
        and row.get("new_provider_calls") == 0
        and row.get("retained_transformer_token_state_bytes") == 0
        and type(messages) is list
        and len(messages) == 2
        and all(
            type(message) is dict
            and set(message) == {"role", "content"}
            and type(message.get("role")) is str
            and type(message.get("content")) is str
            for message in messages
        )
        and row.get("messages_sha256") == identity_sha256(messages)
        and type(provider) is dict
        and row.get("provider_input_sha256") == identity_sha256(provider)
        and messages[1] == {
            "role": "user",
            "content": json.dumps(
                provider,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        and type(handles) is list
        and type(presented) is list
        and handles == presented
        and bool(handles)
        and len(handles) == len(set(handles))
        and all(type(value) is str and bool(value) for value in handles)
        and row.get("prompt_token_proxy")
        == count_chat_prompt_token_proxy(messages)
        and type(row.get("prompt_token_proxy")) is int
        and int(row["prompt_token_proxy"]) + OUTPUT_TOKEN_RESERVE
        <= HARD_TOTAL_TOKEN_CAP,
        "sealed linked repair row changed",
    )
    require_text(row.get("question_id"), "repair question ID")
    for key in (
        "messages_sha256",
        "provider_input_sha256",
        "question_sha256",
        "receipt_sha256",
    ):
        require_sha256(row.get(key), f"repair row {key}")
    assert_gold_blind(messages, path="r7_linked_terminal_runner.messages")
    assert_gold_blind(provider, path="r7_linked_terminal_runner.provider")
    return row


def build_preflight_payload(
    source: SealedArtifact,
    source_replay: SealedArtifact,
    a1: SealedArtifact,
    a1_replay: SealedArtifact,
    *,
    model: str = DEFAULT_MODEL,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> tuple[dict[str, Any], tuple[tuple[dict[str, str], ...], ...]]:
    ordered_a1, source_by_id = _validate_source_envelopes(
        source, source_replay, a1, a1_replay
    )
    _require(
        model == DEFAULT_MODEL
        and gateway_url == DEFAULT_GATEWAY_URL
        and type(max_concurrency) is int
        and max_concurrency > 0,
        "linked repair runtime policy changed",
    )
    questions: list[dict[str, Any]] = []
    prompts: list[tuple[dict[str, str], ...]] = []
    question_receipts: list[dict[str, str]] = []
    retained_receipts: list[dict[str, Any]] = []
    for a1_question in ordered_a1:
        question_id = require_text(a1_question.get("question_id"), "A1 question ID")
        source_question = source_by_id[question_id]
        row = _validate_repair_row(
            compile_r7_linked_terminal_repair(a1_question, source_question)
        )
        _require(
            row["question_id"] == question_id
            and row["question_sha256"] == a1_question.get("question_sha256"),
            "linked repair question projection changed",
        )
        questions.append(row)
        prompts.append(tuple(dict(message) for message in row["messages"]))
        question_receipts.append(
            {
                "question_id": question_id,
                "question_sha256": str(row["question_sha256"]),
                "repair_receipt_sha256": str(row["receipt_sha256"]),
            }
        )
        retained_receipts.append(
            {
                "allowed_handle_ids": list(row["allowed_handle_ids"]),
                "question_id": question_id,
            }
        )
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS
    )
    retained_count = sum(len(row["allowed_handle_ids"]) for row in questions)
    _require(
        len(questions) == len(prompts) == QUESTION_COUNT
        and len({row["question_id"] for row in questions}) == QUESTION_COUNT
        and population.logical_prompt_count
        == population.unique_prompt_count
        == QUESTION_COUNT
        and retained_count == EXPECTED_RETAINED_LEAF_COUNT,
        "linked repair exact-11 or retained population changed",
    )
    body = {
        "a1_construction_artifact_sha256": a1.sha256,
        "a1_replay_artifact_sha256": a1_replay.sha256,
        "format": PREFLIGHT_FORMAT,
        "gateway_url": gateway_url,
        "gold_loaded": False,
        "hard_total_token_cap": HARD_TOTAL_TOKEN_CAP,
        "max_concurrency": max_concurrency,
        "model": model,
        "new_provider_calls": 0,
        "ordinal_cli_routing_available": False,
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "prompt_population": population.model_dump(),
        "prompt_population_sha256": population.prompt_population_sha256,
        "question_count": QUESTION_COUNT,
        "question_population_sha256": identity_sha256(question_receipts),
        "questions": questions,
        "required_authorized_provider_calls": QUESTION_COUNT,
        "retained_leaf_count": retained_count,
        "retained_population_sha256": identity_sha256(retained_receipts),
        "retained_transformer_token_state_bytes": 0,
        "source_construction_artifact_sha256": source.sha256,
        "source_replay_artifact_sha256": source_replay.sha256,
    }
    payload = {**body, "construction_identity_sha256": identity_sha256(body)}
    _require(set(payload) == PREFLIGHT_KEYS, "linked repair preflight shape changed")
    return payload, tuple(prompts)


def _validate_preflight(
    construction: SealedArtifact,
    replay: SealedArtifact | None = None,
) -> tuple[tuple[tuple[dict[str, str], ...], ...], tuple[dict[str, Any], ...]]:
    payload = construction.payload
    raw_questions = payload.get("questions")
    _require(
        set(payload) == PREFLIGHT_KEYS
        and payload.get("format") == PREFLIGHT_FORMAT
        and payload.get("construction_identity_sha256")
        == identity_sha256(_without(payload, "construction_identity_sha256"))
        and payload.get("source_construction_artifact_sha256")
        == EXPECTED_SOURCE_SHA256
        and payload.get("source_replay_artifact_sha256")
        == EXPECTED_SOURCE_SHA256
        and payload.get("a1_construction_artifact_sha256")
        == EXPECTED_A1_SHA256
        and payload.get("a1_replay_artifact_sha256") == EXPECTED_A1_SHA256
        and payload.get("gold_loaded") is False
        and payload.get("new_provider_calls") == 0
        and payload.get("ordinal_cli_routing_available") is False
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("hard_total_token_cap") == HARD_TOTAL_TOKEN_CAP
        and payload.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
        and payload.get("model") == DEFAULT_MODEL
        and payload.get("gateway_url") == DEFAULT_GATEWAY_URL
        and type(payload.get("max_concurrency")) is int
        and int(payload["max_concurrency"]) > 0
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("required_authorized_provider_calls") == QUESTION_COUNT
        and payload.get("retained_leaf_count") == EXPECTED_RETAINED_LEAF_COUNT
        and type(raw_questions) is list
        and len(raw_questions) == QUESTION_COUNT,
        "sealed linked repair preflight changed",
    )
    if replay is not None:
        _require(
            construction.sha256 == replay.sha256
            and construction.payload == replay.payload,
            "linked repair preflight replay is not byte-identical",
        )
    questions: list[dict[str, Any]] = []
    prompts: list[tuple[dict[str, str], ...]] = []
    question_receipts: list[dict[str, str]] = []
    retained_receipts: list[dict[str, Any]] = []
    question_ids: list[str] = []
    for raw in raw_questions:
        _require(type(raw) is dict, "repair preflight question changed type")
        row = _validate_repair_row(raw)
        question_id = str(row["question_id"])
        questions.append(row)
        prompts.append(tuple(dict(message) for message in row["messages"]))
        question_ids.append(question_id)
        question_receipts.append(
            {
                "question_id": question_id,
                "question_sha256": str(row["question_sha256"]),
                "repair_receipt_sha256": str(row["receipt_sha256"]),
            }
        )
        retained_receipts.append(
            {
                "allowed_handle_ids": list(row["allowed_handle_ids"]),
                "question_id": question_id,
            }
        )
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS
    )
    _require(
        len(set(question_ids)) == QUESTION_COUNT
        and sum(len(row["allowed_handle_ids"]) for row in questions)
        == EXPECTED_RETAINED_LEAF_COUNT
        and population.logical_prompt_count
        == population.unique_prompt_count
        == QUESTION_COUNT
        and population.model_dump() == payload.get("prompt_population")
        and population.prompt_population_sha256
        == payload.get("prompt_population_sha256")
        and payload.get("question_population_sha256")
        == identity_sha256(question_receipts)
        and payload.get("retained_population_sha256")
        == identity_sha256(retained_receipts),
        "sealed linked repair population changed",
    )
    return tuple(prompts), tuple(questions)


def _read_preflight(
    output_root: str | Path,
    *,
    expected_construction_sha256: str,
    expected_replay_sha256: str,
) -> tuple[
    SealedArtifact,
    SealedArtifact,
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
]:
    root = Path(output_root)
    construction = _read_expected(
        root / PREFLIGHT_NAME,
        expected_construction_sha256,
        "linked repair preflight construction",
    )
    replay = _read_expected(
        root / PREFLIGHT_REPLAY_NAME,
        expected_replay_sha256,
        "linked repair preflight replay",
    )
    prompts, questions = _validate_preflight(construction, replay)
    return construction, replay, prompts, questions


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    _require(
        not (output_root / CHECKPOINT_DIR_NAME).exists(),
        "linked repair preflight requires an absent checkpoint root",
    )
    source, source_replay, a1, a1_replay = _load_source_pairs(
        source_root=args.source_root,
        a1_root=args.a1_root,
        expected_source_sha256=str(args.expected_source_sha256),
        expected_a1_sha256=str(args.expected_a1_sha256),
    )
    payload, prompts = build_preflight_payload(
        source,
        source_replay,
        a1,
        a1_replay,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    # Rebuild independently before publishing the replay identity.
    replayed, replay_prompts = build_preflight_payload(
        source,
        source_replay,
        a1,
        a1_replay,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    _require(
        replayed == payload and replay_prompts == prompts,
        "linked repair preflight rebuild is not deterministic",
    )
    construction, created = publish_sealed_json(
        output_root / PREFLIGHT_NAME, payload
    )
    replay, replay_created = publish_sealed_json(
        output_root / PREFLIGHT_REPLAY_NAME, replayed
    )
    _validate_preflight(construction, replay)
    return {
        "construction_created": created,
        "preflight_construction_sha256": construction.sha256,
        "preflight_replay_created": replay_created,
        "preflight_replay_sha256": replay.sha256,
        "question_count": QUESTION_COUNT,
        "required_authorized_provider_calls": QUESTION_COUNT,
        "retained_leaf_count": EXPECTED_RETAINED_LEAF_COUNT,
        "retained_transformer_token_state_bytes": 0,
    }


def _runtime(
    preflight: SealedArtifact,
    replay: SealedArtifact,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    output_root: str | Path,
    model: str,
    gateway_url: str,
    max_concurrency: int,
    client: Any | None,
) -> FastCompletionRuntime:
    _require(
        preflight.sha256 == replay.sha256
        and preflight.payload == replay.payload
        and model == preflight.payload.get("model") == DEFAULT_MODEL
        and gateway_url
        == preflight.payload.get("gateway_url")
        == DEFAULT_GATEWAY_URL
        and max_concurrency == preflight.payload.get("max_concurrency")
        and len(prompts) == QUESTION_COUNT,
        "linked repair runtime differs from sealed preflight",
    )
    return FastCompletionRuntime(
        checkpoint_dir=Path(output_root) / CHECKPOINT_DIR_NAME,
        prompt_population=prompts,
        model=DEFAULT_MODEL,
        client=client,
        max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS,
        max_new_tokens=OUTPUT_TOKEN_RESERVE,
        max_concurrency=max_concurrency,
        retries=0,
        benchmark_provenance={
            "a1_construction_artifact_sha256": preflight.payload[
                "a1_construction_artifact_sha256"
            ],
            "arm": FORMAT,
            "authorized_unique_calls": QUESTION_COUNT,
            "experiment_format": RUN_FORMAT,
            "gateway_url": DEFAULT_GATEWAY_URL,
            "gold_loaded": False,
            "preflight_construction_artifact_sha256": preflight.sha256,
            "preflight_replay_artifact_sha256": replay.sha256,
            "source_construction_artifact_sha256": preflight.payload[
                "source_construction_artifact_sha256"
            ],
        },
    )


def _checkpoint_batch(
    preflight: SealedArtifact,
    replay: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    output_root: str | Path,
    model: str,
    gateway_url: str,
    max_concurrency: int,
    client: Any | None,
) -> FastCompletionBatch:
    runtime = _runtime(
        preflight,
        replay,
        prompts,
        output_root=output_root,
        model=model,
        gateway_url=gateway_url,
        max_concurrency=max_concurrency,
        client=client,
    )
    try:
        return runtime.run()
    finally:
        runtime.close()


def _read_only_checkpoint_count(output_root: str | Path) -> int:
    root = Path(output_root) / CHECKPOINT_DIR_NAME
    if not root.exists():
        return 0
    _require(
        root.is_dir() and not root.is_symlink(),
        "linked repair checkpoint root must be a regular directory",
    )
    requests: set[str] = set()
    responses: set[str] = set()
    for path in root.iterdir():
        _require(
            path.is_file() and not path.is_symlink(),
            "linked repair checkpoint root contains foreign state",
        )
        if path.name == ".fast-completion-journal.lock":
            continue
        match = _JOURNAL_FILENAME_RE.fullmatch(path.name)
        _require(match is not None, "linked repair journal filename changed")
        assert match is not None
        target = requests if match.group("kind") == "request" else responses
        target.add(match.group("key"))
    _require(
        requests == responses,
        "linked repair request is incomplete; unsafe retry forbidden",
    )
    _require(
        len(requests) <= QUESTION_COUNT,
        "linked repair checkpoint population exceeds eleven calls",
    )
    return len(requests)


def _validated_checkpoint_hits(
    preflight: SealedArtifact,
    replay: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    output_root: str | Path,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> int:
    if not (Path(output_root) / CHECKPOINT_DIR_NAME).exists():
        return 0
    runtime = _runtime(
        preflight,
        replay,
        prompts,
        output_root=output_root,
        model=model,
        gateway_url=gateway_url,
        max_concurrency=max_concurrency,
        client=None,
    )
    try:
        with runtime._journal_guard():  # noqa: SLF001 - runtime owns journals
            records = runtime._load_all_records()  # noqa: SLF001
    finally:
        runtime.close()
    _require(
        len(records) <= QUESTION_COUNT,
        "linked repair journals escaped the sealed prompt population",
    )
    return len(records)


def _make_provider_client(api_key: str, gateway_url: str) -> Any:
    return live._make_provider_client(api_key, gateway_url)  # noqa: SLF001


def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    preflight, replay, prompts, _questions = _read_preflight(
        args.output_root,
        expected_construction_sha256=str(
            args.expected_preflight_construction_sha256
        ),
        expected_replay_sha256=str(args.expected_preflight_replay_sha256),
    )
    _require(
        args.enable_provider is True
        and type(args.authorized_provider_calls) is int
        and 0 <= args.authorized_provider_calls <= QUESTION_COUNT,
        "linked repair provider requires bounded Terra authorization",
    )
    candidate_hits = _read_only_checkpoint_count(args.output_root)
    remaining = QUESTION_COUNT - candidate_hits
    _require(
        args.authorized_provider_calls == remaining,
        "linked repair authorization must exactly equal remaining calls",
    )
    checkpoint_hits = _validated_checkpoint_hits(
        preflight,
        replay,
        prompts,
        output_root=args.output_root,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    _require(
        checkpoint_hits == candidate_hits,
        "linked repair checkpoint count changed after authorization",
    )
    if remaining == 0:
        batch = _checkpoint_batch(
            preflight,
            replay,
            prompts,
            output_root=args.output_root,
            model=str(args.model),
            gateway_url=str(args.gateway_url),
            max_concurrency=int(args.max_concurrency),
            client=None,
        )
    else:
        load_dotenv()
        api_key = os.environ.get(str(args.api_key_env), "").strip()
        _require(bool(api_key), f"provider API key is empty: {args.api_key_env}")
        client = _make_provider_client(api_key, str(args.gateway_url))
        try:
            batch = _checkpoint_batch(
                preflight,
                replay,
                prompts,
                output_root=args.output_root,
                model=str(args.model),
                gateway_url=str(args.gateway_url),
                max_concurrency=int(args.max_concurrency),
                client=client,
            )
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
    _require(
        batch.usage.logical_calls == batch.usage.unique_calls == QUESTION_COUNT
        and batch.usage.physical_calls + batch.usage.checkpoint_hits
        == QUESTION_COUNT
        and batch.usage.physical_calls <= args.authorized_provider_calls
        and batch.usage.checkpoint_hits >= checkpoint_hits,
        "linked repair provider population changed",
    )
    return {
        "authorized_remaining_provider_calls": remaining,
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "physical_provider_calls": batch.usage.physical_calls,
        "preflight_sha256": preflight.sha256,
        "question_count": QUESTION_COUNT,
        "required_authorized_provider_calls": remaining,
        "retained_transformer_token_state_bytes": 0,
    }


def _strict_json_object(text: str) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token: {token}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise R7LinkedTerminalRepairRunnerError(
            "linked repair completion is not strict JSON"
        ) from exc
    _require(type(value) is dict, "linked repair completion must be one object")
    return value


def _parse_completion(
    completion: str, allowed_handle_ids: Sequence[str]
) -> tuple[str, tuple[str, ...], str]:
    parsed = _strict_json_object(completion)
    _require(
        set(parsed) == {"response_text", "used_handle_ids"},
        "linked repair response schema changed",
    )
    response = require_text(parsed.get("response_text"), "repair response text")
    used_raw = parsed.get("used_handle_ids")
    _require(
        response == response.strip()
        and type(used_raw) is list
        and bool(used_raw)
        and all(type(value) is str and bool(value) for value in used_raw),
        "linked repair response values changed",
    )
    used = tuple(used_raw)
    _require(
        len(used) == len(set(used))
        and set(used) <= set(allowed_handle_ids),
        "linked repair response cites a foreign or repeated handle",
    )
    normalized = {"response_text": response, "used_handle_ids": list(used)}
    return response, used, identity_sha256(normalized)


def _complete_checkpoint_batch(
    preflight: SealedArtifact,
    replay: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    output_root: str | Path,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> FastCompletionBatch:
    _require(
        _read_only_checkpoint_count(output_root) == QUESTION_COUNT,
        "linked repair materialization requires eleven complete checkpoints",
    )
    batch = _checkpoint_batch(
        preflight,
        replay,
        prompts,
        output_root=output_root,
        model=model,
        gateway_url=gateway_url,
        max_concurrency=max_concurrency,
        client=None,
    )
    _require(
        batch.usage.logical_calls
        == batch.usage.unique_calls
        == batch.usage.checkpoint_hits
        == QUESTION_COUNT
        and batch.usage.physical_calls == 0,
        "linked repair checkpoint-only batch changed",
    )
    return batch


def _materialize_payload(
    preflight: SealedArtifact,
    replay: SealedArtifact,
    questions: Sequence[Mapping[str, Any]],
    batch: FastCompletionBatch,
) -> dict[str, Any]:
    _require(
        len(questions)
        == len(batch.logical_completions)
        == len(batch.unique_records)
        == QUESTION_COUNT,
        "linked repair materialization population changed",
    )
    records = {record.messages_sha256: record for record in batch.unique_records}
    _require(len(records) == QUESTION_COUNT, "linked repair records repeat")
    results: list[dict[str, Any]] = []
    judge_rows: list[dict[str, Any]] = []
    for question, completion in zip(
        questions, batch.logical_completions, strict=True
    ):
        record = records.get(str(question["messages_sha256"]))
        _require(
            record is not None
            and record.completion == completion
            and record.checkpoint_hit is True
            and record.physical_call is False
            and record.requested_model == DEFAULT_MODEL
            and record.finish_reason == "stop",
            "linked repair checkpoint record changed",
        )
        assert record is not None
        prediction, used, parse_receipt = _parse_completion(
            completion, tuple(question["allowed_handle_ids"])
        )
        body = {
            "call_key_sha256": record.call_key_sha256,
            "completion_sha256": quote_sha256(completion),
            "format": RESULT_ROW_FORMAT,
            "messages_sha256": question["messages_sha256"],
            "parse_receipt_sha256": parse_receipt,
            "prediction": prediction,
            "prediction_sha256": quote_sha256(prediction),
            "preflight_question_receipt_sha256": question["receipt_sha256"],
            "question_id": question["question_id"],
            "question_sha256": question["question_sha256"],
            "request_journal_sha256": record.request_journal_sha256,
            "response_journal_sha256": record.response_journal_sha256,
            "retained_transformer_token_state_bytes": 0,
            "used_handle_ids": list(used),
        }
        result = {**body, "source_row_sha256": identity_sha256(body)}
        results.append(result)
        provider = question["provider_input"]
        dated_question = require_text(
            provider.get("dated_question"), "repair dated question"
        )
        judge_rows.append(
            {
                "dated_question_sha256": quote_sha256(dated_question),
                "format": JUDGE_ROW_FORMAT,
                "prediction": prediction,
                "prediction_sha256": quote_sha256(prediction),
                "question_id": question["question_id"],
                "question_sha256": question["question_sha256"],
                "source_row_sha256": result["source_row_sha256"],
            }
        )
    body = {
        "a1_construction_artifact_sha256": preflight.payload[
            "a1_construction_artifact_sha256"
        ],
        "a1_replay_artifact_sha256": preflight.payload[
            "a1_replay_artifact_sha256"
        ],
        "completion_batch": batch.model_dump(),
        "format": RUN_FORMAT,
        "gold_loaded": False,
        "judge_row_population_sha256": identity_sha256(judge_rows),
        "judge_rows": judge_rows,
        "physical_provider_calls_during_materialization": 0,
        "preflight_construction_artifact_sha256": preflight.sha256,
        "preflight_replay_artifact_sha256": replay.sha256,
        "prompt_population_sha256": preflight.payload[
            "prompt_population_sha256"
        ],
        "question_count": QUESTION_COUNT,
        "result_count": QUESTION_COUNT,
        "result_population_sha256": identity_sha256(
            [row["source_row_sha256"] for row in results]
        ),
        "results": results,
        "retained_leaf_count": preflight.payload["retained_leaf_count"],
        "retained_population_sha256": preflight.payload[
            "retained_population_sha256"
        ],
        "retained_transformer_token_state_bytes": 0,
        "source_construction_artifact_sha256": preflight.payload[
            "source_construction_artifact_sha256"
        ],
        "source_replay_artifact_sha256": preflight.payload[
            "source_replay_artifact_sha256"
        ],
    }
    return {**body, "run_identity_sha256": identity_sha256(body)}


def _validate_run(
    artifact: SealedArtifact,
    *,
    preflight: SealedArtifact,
    replay: SealedArtifact,
    questions: Sequence[Mapping[str, Any]],
    expected_batch: FastCompletionBatch | None = None,
) -> tuple[dict[str, Any], ...]:
    payload = artifact.payload
    raw_results = payload.get("results")
    raw_judge = payload.get("judge_rows")
    completion_batch = payload.get("completion_batch")
    _require(
        set(payload) == RUN_KEYS
        and payload.get("run_identity_sha256")
        == identity_sha256(_without(payload, "run_identity_sha256"))
        and payload.get("format") == RUN_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("physical_provider_calls_during_materialization") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("preflight_construction_artifact_sha256")
        == preflight.sha256
        and payload.get("preflight_replay_artifact_sha256") == replay.sha256
        and payload.get("source_construction_artifact_sha256")
        == preflight.payload.get("source_construction_artifact_sha256")
        and payload.get("source_replay_artifact_sha256")
        == preflight.payload.get("source_replay_artifact_sha256")
        and payload.get("a1_construction_artifact_sha256")
        == preflight.payload.get("a1_construction_artifact_sha256")
        and payload.get("a1_replay_artifact_sha256")
        == preflight.payload.get("a1_replay_artifact_sha256")
        and payload.get("prompt_population_sha256")
        == preflight.payload.get("prompt_population_sha256")
        and payload.get("retained_leaf_count")
        == preflight.payload.get("retained_leaf_count")
        == EXPECTED_RETAINED_LEAF_COUNT
        and payload.get("retained_population_sha256")
        == preflight.payload.get("retained_population_sha256")
        and payload.get("question_count")
        == payload.get("result_count")
        == QUESTION_COUNT
        and type(raw_results) is list
        and type(raw_judge) is list
        and len(raw_results) == len(raw_judge) == len(questions) == QUESTION_COUNT,
        "sealed linked repair run changed",
    )
    _require(
        type(completion_batch) is dict,
        "linked repair completion batch changed type",
    )
    logical = completion_batch.get("logical_completions")
    records_raw = completion_batch.get("unique_records")
    usage = completion_batch.get("usage")
    provenance = completion_batch.get("provenance")
    _require(
        type(logical) is list
        and len(logical) == QUESTION_COUNT
        and all(type(value) is str and bool(value) for value in logical)
        and type(records_raw) is list
        and len(records_raw) == QUESTION_COUNT
        and all(type(value) is dict for value in records_raw)
        and type(usage) is dict
        and usage.get("logical_calls")
        == usage.get("unique_calls")
        == usage.get("checkpoint_hits")
        == QUESTION_COUNT
        and usage.get("deduplicated_logical_calls") == 0
        and usage.get("physical_calls") == 0
        and type(provenance) is dict
        and provenance.get("model") == DEFAULT_MODEL
        and provenance.get("retries") == 0
        and provenance.get("max_new_tokens") == OUTPUT_TOKEN_RESERVE
        and provenance.get("max_prompt_token_proxy") == MAX_CHAT_PROMPT_TOKENS
        and provenance.get("prompt_population_sha256")
        == preflight.payload.get("prompt_population_sha256")
        and provenance.get("persisted_transformer_token_state") is False
        and provenance.get("retained_transformer_token_state_bytes") == 0
        and completion_batch.get("prompt_population")
        == preflight.payload.get("prompt_population")
        and provenance.get("benchmark_provenance")
        == {
            "a1_construction_artifact_sha256": preflight.payload[
                "a1_construction_artifact_sha256"
            ],
            "arm": FORMAT,
            "authorized_unique_calls": QUESTION_COUNT,
            "experiment_format": RUN_FORMAT,
            "gateway_url": DEFAULT_GATEWAY_URL,
            "gold_loaded": False,
            "preflight_construction_artifact_sha256": preflight.sha256,
            "preflight_replay_artifact_sha256": replay.sha256,
            "source_construction_artifact_sha256": preflight.payload[
                "source_construction_artifact_sha256"
            ],
        },
        "linked repair completion batch/provenance changed",
    )
    if expected_batch is not None:
        _require(
            payload.get("completion_batch") == expected_batch.model_dump(),
            "linked repair completion batch changed",
        )
    results: list[dict[str, Any]] = []
    records = {
        record.get("messages_sha256"): record for record in records_raw
    }
    _require(len(records) == QUESTION_COUNT, "linked repair batch records repeat")
    for question, completion, raw, judge_row in zip(
        questions, logical, raw_results, raw_judge, strict=True
    ):
        record = records.get(question.get("messages_sha256"))
        parsed_prediction, parsed_used, parsed_receipt = _parse_completion(
            completion, tuple(question.get("allowed_handle_ids", ()))
        )
        _require(
            type(raw) is dict
            and set(raw) == RESULT_ROW_KEYS
            and raw.get("source_row_sha256")
            == identity_sha256(_without(raw, "source_row_sha256"))
            and raw.get("format") == RESULT_ROW_FORMAT
            and raw.get("question_id") == question.get("question_id")
            and raw.get("question_sha256") == question.get("question_sha256")
            and raw.get("messages_sha256") == question.get("messages_sha256")
            and raw.get("preflight_question_receipt_sha256")
            == question.get("receipt_sha256")
            and raw.get("retained_transformer_token_state_bytes") == 0
            and raw.get("prediction_sha256")
            == quote_sha256(require_text(raw.get("prediction"), "prediction"))
            and raw.get("prediction") == parsed_prediction
            and tuple(raw.get("used_handle_ids", ())) == parsed_used
            and raw.get("parse_receipt_sha256") == parsed_receipt
            and raw.get("completion_sha256") == quote_sha256(completion)
            and type(raw.get("used_handle_ids")) is list
            and bool(raw["used_handle_ids"])
            and len(raw["used_handle_ids"]) == len(set(raw["used_handle_ids"]))
            and set(raw["used_handle_ids"])
            <= set(question.get("allowed_handle_ids", ()))
            and type(record) is dict
            and record.get("completion") == completion
            and record.get("completion_sha256") == quote_sha256(completion)
            and record.get("messages_sha256") == question.get("messages_sha256")
            and record.get("call_key_sha256") == raw.get("call_key_sha256")
            and record.get("request_journal_sha256")
            == raw.get("request_journal_sha256")
            and record.get("response_journal_sha256")
            == raw.get("response_journal_sha256")
            and record.get("requested_model") == DEFAULT_MODEL
            and record.get("finish_reason") == "stop"
            and record.get("checkpoint_hit") is True
            and record.get("physical_call") is False,
            "linked repair result row changed",
        )
        for key in (
            "call_key_sha256",
            "completion_sha256",
            "messages_sha256",
            "parse_receipt_sha256",
            "prediction_sha256",
            "preflight_question_receipt_sha256",
            "question_sha256",
            "request_journal_sha256",
            "response_journal_sha256",
            "source_row_sha256",
        ):
            require_sha256(raw.get(key), f"linked repair result {key}")
        _require(
            type(judge_row) is dict
            and set(judge_row) == JUDGE_ROW_KEYS
            and judge_row.get("format") == JUDGE_ROW_FORMAT
            and judge_row.get("question_id") == raw.get("question_id")
            and judge_row.get("question_sha256") == raw.get("question_sha256")
            and judge_row.get("prediction") == raw.get("prediction")
            and judge_row.get("prediction_sha256")
            == raw.get("prediction_sha256")
            and judge_row.get("source_row_sha256")
            == raw.get("source_row_sha256"),
            "linked repair judge projection changed",
        )
        require_sha256(
            judge_row.get("dated_question_sha256"), "repair dated question"
        )
        results.append(dict(raw))
    _require(
        len({row["question_id"] for row in results}) == QUESTION_COUNT
        and payload.get("result_population_sha256")
        == identity_sha256([row["source_row_sha256"] for row in results])
        and payload.get("judge_row_population_sha256")
        == identity_sha256(raw_judge),
        "linked repair result population changed",
    )
    return tuple(results)


def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    preflight, replay, prompts, questions = _read_preflight(
        args.output_root,
        expected_construction_sha256=str(
            args.expected_preflight_construction_sha256
        ),
        expected_replay_sha256=str(args.expected_preflight_replay_sha256),
    )
    batch = _complete_checkpoint_batch(
        preflight,
        replay,
        prompts,
        output_root=args.output_root,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    payload = _materialize_payload(preflight, replay, questions, batch)
    artifact, created = publish_sealed_json(Path(args.output_root) / RUN_NAME, payload)
    _validate_run(
        artifact,
        preflight=preflight,
        replay=replay,
        questions=questions,
        expected_batch=batch,
    )
    return {
        "checkpoint_hits": QUESTION_COUNT,
        "created": created,
        "physical_provider_calls": 0,
        "question_count": QUESTION_COUNT,
        "run_sha256": artifact.sha256,
        "retained_transformer_token_state_bytes": 0,
    }


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    preflight, replay, prompts, questions = _read_preflight(
        args.output_root,
        expected_construction_sha256=str(
            args.expected_preflight_construction_sha256
        ),
        expected_replay_sha256=str(args.expected_preflight_replay_sha256),
    )
    batch = _complete_checkpoint_batch(
        preflight,
        replay,
        prompts,
        output_root=args.output_root,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    rebuilt = _materialize_payload(preflight, replay, questions, batch)
    run = _read_expected(
        Path(args.output_root) / RUN_NAME,
        str(args.expected_run_sha256),
        "linked repair run",
    )
    _require(run.payload == rebuilt, "linked repair run differs from replay")
    _validate_run(
        run,
        preflight=preflight,
        replay=replay,
        questions=questions,
        expected_batch=batch,
    )
    replay_artifact, _created = publish_sealed_json(
        Path(args.output_root) / REPLAY_NAME, rebuilt
    )
    _require(
        replay_artifact.sha256 == run.sha256,
        "linked repair answer replay is not byte-identical",
    )
    return {
        "byte_identical": True,
        "physical_provider_calls": 0,
        "replay_sha256": replay_artifact.sha256,
        "retained_transformer_token_state_bytes": 0,
        "run_sha256": run.sha256,
    }


def load_verified_answer_run(
    output_root: str | Path,
    *,
    expected_preflight_construction_sha256: str,
    expected_preflight_replay_sha256: str,
    expected_run_sha256: str,
    expected_replay_sha256: str,
) -> tuple[SealedArtifact, SealedArtifact, tuple[dict[str, Any], ...]]:
    preflight, replay, _prompts, questions = _read_preflight(
        output_root,
        expected_construction_sha256=expected_preflight_construction_sha256,
        expected_replay_sha256=expected_preflight_replay_sha256,
    )
    run = _read_expected(
        Path(output_root) / RUN_NAME, expected_run_sha256, "linked repair run"
    )
    answer_replay = _read_expected(
        Path(output_root) / REPLAY_NAME,
        expected_replay_sha256,
        "linked repair answer replay",
    )
    _require(
        run.sha256 == answer_replay.sha256 and run.payload == answer_replay.payload,
        "linked repair answer run/replay is not byte-identical",
    )
    rows = _validate_run(
        run, preflight=preflight, replay=replay, questions=questions
    )
    return run, answer_replay, rows


def _add_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument(
        "--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY
    )


def _add_preflight_binding(parser: argparse.ArgumentParser) -> None:
    _add_runtime(parser)
    parser.add_argument(
        "--expected-preflight-construction-sha256", required=True
    )
    parser.add_argument("--expected-preflight-replay-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight")
    _add_runtime(preflight)
    preflight.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    preflight.add_argument("--a1-root", type=Path, default=DEFAULT_A1_ROOT)
    preflight.add_argument(
        "--expected-source-sha256", default=EXPECTED_SOURCE_SHA256
    )
    preflight.add_argument("--expected-a1-sha256", default=EXPECTED_A1_SHA256)

    provider = commands.add_parser("provider-run")
    _add_preflight_binding(provider)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, required=True)
    provider.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)

    materialize = commands.add_parser("materialize")
    _add_preflight_binding(materialize)

    replay = commands.add_parser("replay")
    _add_preflight_binding(replay)
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
    "A1_REPLAY_NAME",
    "CHECKPOINT_DIR_NAME",
    "DEFAULT_A1_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_SOURCE_ROOT",
    "EXPECTED_A1_SHA256",
    "EXPECTED_SOURCE_SHA256",
    "FORMAT",
    "PREFLIGHT_NAME",
    "PREFLIGHT_REPLAY_NAME",
    "QUESTION_COUNT",
    "REPLAY_NAME",
    "RUN_NAME",
    "R7LinkedTerminalRepairRunnerError",
    "build_parser",
    "build_preflight_payload",
    "load_verified_answer_run",
    "main",
    "run_materialize",
    "run_preflight",
    "run_provider",
    "run_replay",
]
