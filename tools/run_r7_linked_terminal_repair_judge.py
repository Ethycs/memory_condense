#!/usr/bin/env python3
"""Judge the sealed exact-11 linked-terminal repair with Sol.

The linked-repair answer run and its byte-identical replay are authenticated
before the locked benchmark is opened.  Each provider prompt contains exactly
one dated question, its reference answer, and the sealed prediction.  Answer
evidence, handles, facts, source locators, and caller-selected ordinals never
cross the judge boundary.

The lifecycle deliberately has no separate release artifact: the sealed
preflight pair plus an authorization exactly equal to the remaining complete
checkpoints is the release gate.  Calls are resumable and zero-retry;
materialization and replay are checkpoint-only.
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

from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from memory_condense.domain.integrity import file_sha256  # noqa: E402
from memory_condense.eval._binary_judge_protocol import (  # noqa: E402
    JUDGE_MAX_TOKENS,
    parse_binary_judge_verdict,
)
from memory_condense.eval.benchmark import (  # noqa: E402
    build_judge_prompt,
    exact_match,
    f1_score,
)
from memory_condense.eval.fast_completion_runtime import (  # noqa: E402
    FastCompletionBatch,
    FastCompletionRuntime,
    preflight_fast_completion_prompts,
)
from memory_condense.eval.locked_split import (  # noqa: E402
    load_split_manifest,
    select_locked_split,
)
from memory_condense.eval.recall_guarded_cumulative_population import (  # noqa: E402
    LOCKED_LONGMEMEVAL_DATASET_SHA256,
    LOCKED_LONGMEMEVAL_SPLIT_MANIFEST_SHA256,
)
from memory_condense.ingest.loader import load_benchmark  # noqa: E402
from tools import run_r7_linked_terminal_repair as answer_cli  # noqa: E402
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
from tools.run_locked_query_answer_judge import DEFAULT_DATASET  # noqa: E402
from tools.run_matched_eval_spine import DEFAULT_SPLIT  # noqa: E402


FORMAT = "memory-condense-r7-linked-terminal-repair-sol-judge-lifecycle-v1"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight-v1"
JUDGE_FORMAT = f"{FORMAT}-judge-v1"
SCORE_FORMAT = f"{FORMAT}-score-v1"
PROMPT_ROW_FORMAT = f"{FORMAT}-prompt-row-v1"
JUDGE_ROW_FORMAT = f"{FORMAT}-judge-row-v1"
SCORE_ROW_FORMAT = f"{FORMAT}-score-row-v1"

PREFLIGHT_NAME = "r7-linked-terminal-repair-sol-judge-preflight-v1.json"
PREFLIGHT_REPLAY_NAME = (
    "r7-linked-terminal-repair-sol-judge-preflight-replay-v1.json"
)
JUDGE_NAME = "r7-linked-terminal-repair-sol-judge-v1.json"
JUDGE_REPLAY_NAME = "r7-linked-terminal-repair-sol-judge-replay-v1.json"
SCORE_NAME = "r7-linked-terminal-repair-sol-score-v1.json"
SCORE_REPLAY_NAME = "r7-linked-terminal-repair-sol-score-replay-v1.json"
CHECKPOINT_DIR_NAME = "sol-r7-linked-terminal-repair-judge-v1-calls"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANSWER_ROOT = answer_cli.DEFAULT_OUTPUT_ROOT
DEFAULT_JUDGE_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/"
    "locked-r7-linked-terminal-repair-sol-judge-v1"
)
DEFAULT_MODEL = judging.DEFAULT_SOL_GATEWAY_MODEL
DEFAULT_GATEWAY_URL = live.DEFAULT_GATEWAY_URL
DEFAULT_API_KEY_ENV = live.DEFAULT_API_KEY_ENV
DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_MAX_PROMPT_TOKENS = judging.DEFAULT_MAX_JUDGE_PROMPT_TOKENS
QUESTION_COUNT = answer_cli.QUESTION_COUNT

EXPECTED_ANSWER_PREFLIGHT_SHA256 = (
    "b20e88b435f58bdadb6cadb0366301be8b1fd19905bec9e5a88da8e8c27e3144"
)
EXPECTED_ANSWER_RUN_SHA256 = (
    "27b5a4e2cd693e066c4eea56233ae1d73e61be065c525876ebcb54a74b9447f9"
)

ANSWER_BINDING_KEYS = {
    "answer_a1_construction_artifact_sha256",
    "answer_a1_replay_artifact_sha256",
    "answer_judge_row_population_sha256",
    "answer_preflight_construction_artifact_sha256",
    "answer_preflight_replay_artifact_sha256",
    "answer_prompt_population_sha256",
    "answer_replay_artifact_sha256",
    "answer_retained_population_sha256",
    "answer_run_artifact_sha256",
    "answer_source_construction_artifact_sha256",
    "answer_source_replay_artifact_sha256",
    "answer_source_row_population_sha256",
}
PROMPT_ROW_KEYS = {
    "answer_source_row_sha256",
    "category",
    "dated_question",
    "dated_question_sha256",
    "format",
    "judge_input_receipt_sha256",
    "messages",
    "messages_sha256",
    "prediction",
    "prediction_sha256",
    "prompt_row_receipt_sha256",
    "prompt_token_proxy",
    "question_id",
    "question_sha256",
    "reference",
    "reference_sha256",
}
PREFLIGHT_KEYS = {
    "construction_identity_sha256",
    "dataset_file_sha256",
    "format",
    "gateway_url",
    "gold_loaded",
    "gold_population_sha256",
    "judge_input_population_sha256",
    "max_concurrency",
    "model",
    "ordinal_cli_routing_available",
    "physical_provider_calls",
    "prompt_population",
    "prompt_population_sha256",
    "prompt_rows",
    "question_count",
    "question_population_sha256",
    "required_authorized_provider_calls",
    "retained_transformer_token_state_bytes",
    "source_population_sha256",
    "split_file_sha256",
}.union(ANSWER_BINDING_KEYS)
JUDGE_ROW_KEYS = {
    "answer_source_row_sha256",
    "call_key_sha256",
    "category",
    "correct",
    "dated_question_sha256",
    "format",
    "judge_output",
    "judge_output_sha256",
    "judge_row_receipt_sha256",
    "messages_sha256",
    "normalized_exact_match",
    "normalized_f1",
    "prediction_sha256",
    "prompt_row_receipt_sha256",
    "question_id",
    "question_sha256",
    "reference_sha256",
    "request_journal_sha256",
    "response_journal_sha256",
}
JUDGE_KEYS = {
    "aggregate",
    "answer_binding_sha256",
    "completion_batch",
    "format",
    "gold_loaded",
    "judge_identity_sha256",
    "physical_provider_calls_during_materialization",
    "preflight_construction_artifact_sha256",
    "preflight_replay_artifact_sha256",
    "prompt_population_sha256",
    "question_count",
    "question_population_sha256",
    "questions",
    "retained_transformer_token_state_bytes",
    "source_population_sha256",
}.union(ANSWER_BINDING_KEYS)
SCORE_ROW_KEYS = {
    "correct",
    "format",
    "judge_row_receipt_sha256",
    "question_id",
    "question_sha256",
    "score_row_receipt_sha256",
}
SCORE_KEYS = {
    "accuracy",
    "answer_binding_sha256",
    "correct",
    "format",
    "gold_loaded",
    "judge_artifact_sha256",
    "physical_provider_calls_during_scoring",
    "preflight_construction_artifact_sha256",
    "preflight_replay_artifact_sha256",
    "question_count",
    "retained_transformer_token_state_bytes",
    "score_identity_sha256",
    "score_population_sha256",
    "score_rows",
}.union(ANSWER_BINDING_KEYS)

_JOURNAL_FILENAME_RE = re.compile(
    r"^(?P<key>[0-9a-f]{64})\.(?P<kind>request|response)\.json$"
)


class R7LinkedTerminalRepairJudgeError(MatchedEvalContractError):
    """The answer, gold binding, judge prompt, journal, or score changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise R7LinkedTerminalRepairJudgeError(message)


def _without(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    return {name: child for name, child in value.items() if name != key}


def _read_expected(path: str | Path, expected: str, label: str) -> SealedArtifact:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == require_sha256(expected, label),
        f"{label} artifact changed",
    )
    return artifact


@dataclass(frozen=True, slots=True)
class _GoldRow:
    question_id: str
    question_sha256: str
    dated_question: str
    dated_question_sha256: str
    reference: str
    reference_sha256: str
    category: str


def _validate_answer_rows(
    run: SealedArtifact,
    results: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    raw_rows = run.payload.get("judge_rows")
    _require(
        run.payload.get("format") == answer_cli.RUN_FORMAT
        and run.payload.get("gold_loaded") is False
        and run.payload.get("physical_provider_calls_during_materialization") == 0
        and run.payload.get("retained_transformer_token_state_bytes") == 0
        and run.payload.get("question_count") == QUESTION_COUNT
        and run.payload.get("result_count") == QUESTION_COUNT
        and type(raw_rows) is list
        and len(raw_rows) == len(results) == QUESTION_COUNT
        and run.payload.get("judge_row_population_sha256")
        == identity_sha256(raw_rows),
        "linked-repair answer judge population changed",
    )
    by_id = {row.get("question_id"): row for row in results}
    _require(len(by_id) == QUESTION_COUNT, "linked-repair result identities repeat")
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        _require(type(raw) is dict, "linked-repair judge source changed type")
        prediction = require_text(raw.get("prediction"), "sealed prediction")
        question_id = require_text(raw.get("question_id"), "sealed question ID")
        result = by_id.get(question_id)
        _require(
            set(raw) == answer_cli.JUDGE_ROW_KEYS
            and raw.get("format") == answer_cli.JUDGE_ROW_FORMAT
            and raw.get("prediction_sha256") == quote_sha256(prediction)
            and raw.get("question_sha256") == raw.get("dated_question_sha256")
            and result is not None
            and result.get("source_row_sha256") == raw.get("source_row_sha256")
            and result.get("prediction") == prediction
            and result.get("prediction_sha256") == raw.get("prediction_sha256"),
            f"linked-repair judge source changed for {question_id}",
        )
        for key in (
            "dated_question_sha256",
            "prediction_sha256",
            "question_sha256",
            "source_row_sha256",
        ):
            require_sha256(raw.get(key), f"linked-repair source {key}")
        rows.append(dict(raw))
    _require(
        len({row["question_id"] for row in rows}) == QUESTION_COUNT,
        "linked-repair judge question identities repeat",
    )
    return tuple(rows)


def _answer_binding(
    run: SealedArtifact,
    replay: SealedArtifact,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    _require(
        run.sha256 == replay.sha256 and run.payload == replay.payload,
        "linked-repair answer run/replay is not byte-identical",
    )
    binding = {
        "answer_a1_construction_artifact_sha256": require_sha256(
            run.payload.get("a1_construction_artifact_sha256"), "answer A1 construction"
        ),
        "answer_a1_replay_artifact_sha256": require_sha256(
            run.payload.get("a1_replay_artifact_sha256"), "answer A1 replay"
        ),
        "answer_judge_row_population_sha256": require_sha256(
            run.payload.get("judge_row_population_sha256"), "answer judge population"
        ),
        "answer_preflight_construction_artifact_sha256": require_sha256(
            run.payload.get("preflight_construction_artifact_sha256"),
            "answer preflight construction",
        ),
        "answer_preflight_replay_artifact_sha256": require_sha256(
            run.payload.get("preflight_replay_artifact_sha256"),
            "answer preflight replay",
        ),
        "answer_prompt_population_sha256": require_sha256(
            run.payload.get("prompt_population_sha256"), "answer prompt population"
        ),
        "answer_replay_artifact_sha256": replay.sha256,
        "answer_retained_population_sha256": require_sha256(
            run.payload.get("retained_population_sha256"),
            "answer retained population",
        ),
        "answer_run_artifact_sha256": run.sha256,
        "answer_source_construction_artifact_sha256": require_sha256(
            run.payload.get("source_construction_artifact_sha256"),
            "answer source construction",
        ),
        "answer_source_replay_artifact_sha256": require_sha256(
            run.payload.get("source_replay_artifact_sha256"),
            "answer source replay",
        ),
        "answer_source_row_population_sha256": identity_sha256(
            [row["source_row_sha256"] for row in rows]
        ),
    }
    _require(set(binding) == ANSWER_BINDING_KEYS, "answer binding shape changed")
    return binding


def _load_answer_source(
    *,
    answer_root: str | Path,
    expected_preflight_construction_sha256: str,
    expected_preflight_replay_sha256: str,
    expected_run_sha256: str,
    expected_replay_sha256: str,
) -> tuple[SealedArtifact, SealedArtifact, tuple[dict[str, Any], ...]]:
    run, replay, results = answer_cli.load_verified_answer_run(
        answer_root,
        expected_preflight_construction_sha256=(
            expected_preflight_construction_sha256
        ),
        expected_preflight_replay_sha256=expected_preflight_replay_sha256,
        expected_run_sha256=expected_run_sha256,
        expected_replay_sha256=expected_replay_sha256,
    )
    rows = _validate_answer_rows(run, results)
    _answer_binding(run, replay, rows)
    return run, replay, rows


def _load_locked_gold(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    dataset_path: str | Path,
    split_path: str | Path,
) -> tuple[tuple[_GoldRow, ...], str]:
    dataset = Path(dataset_path)
    split = Path(split_path)
    _require(
        file_sha256(dataset) == LOCKED_LONGMEMEVAL_DATASET_SHA256
        and file_sha256(split) == LOCKED_LONGMEMEVAL_SPLIT_MANIFEST_SHA256,
        "linked-repair judge locked dataset/split changed",
    )
    selected = select_locked_split(
        load_benchmark(dataset, "longmemeval"),
        dataset_path=dataset,
        manifest=load_split_manifest(split),
        split="validation",
    )
    questions = tuple(question for sample in selected for question in sample.questions)
    by_id = {question.question_id: question for question in questions}
    _require(
        len(selected) == len(questions) == len(by_id) == 100
        and len(source_rows) == QUESTION_COUNT,
        "linked-repair judge validation population changed",
    )
    rows: list[_GoldRow] = []
    receipts: list[dict[str, str]] = []
    for source in source_rows:
        question_id = require_text(source.get("question_id"), "judge question ID")
        question = by_id.get(question_id)
        _require(question is not None, "linked-repair answer question is not validation")
        assert question is not None
        dated_sha = quote_sha256(question.dated_question)
        reference_sha = quote_sha256(question.answer)
        _require(
            source.get("question_sha256") == dated_sha
            and source.get("dated_question_sha256") == dated_sha,
            f"linked-repair answer/gold join changed for {question_id}",
        )
        category = str(question.category or "uncategorized")
        row = _GoldRow(
            question_id=question_id,
            question_sha256=dated_sha,
            dated_question=question.dated_question,
            dated_question_sha256=dated_sha,
            reference=question.answer,
            reference_sha256=reference_sha,
            category=category,
        )
        rows.append(row)
        receipts.append(
            {
                "category": category,
                "dated_question_sha256": dated_sha,
                "question_id": question_id,
                "question_sha256": dated_sha,
                "reference_sha256": reference_sha,
            }
        )
    _require(
        len({row.question_id for row in rows}) == QUESTION_COUNT,
        "linked-repair gold identities repeat",
    )
    return tuple(rows), identity_sha256(receipts)


def _judge_input_body(
    row: Mapping[str, Any], messages: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    return {
        "dated_question": row["dated_question"],
        "dated_question_sha256": row["dated_question_sha256"],
        "format": f"{FORMAT}-provider-input-v1",
        "messages_sha256": identity_sha256([dict(value) for value in messages]),
        "prediction": row["prediction"],
        "prediction_sha256": row["prediction_sha256"],
        "question_id": row["question_id"],
        "question_sha256": row["question_sha256"],
        "reference": row["reference"],
        "reference_sha256": row["reference_sha256"],
    }


def build_preflight_payload(
    run: SealedArtifact,
    replay: SealedArtifact,
    source_rows: Sequence[Mapping[str, Any]],
    gold_rows: Sequence[_GoldRow],
    *,
    gold_population_sha256: str,
    model: str = DEFAULT_MODEL,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> tuple[dict[str, Any], tuple[tuple[dict[str, str], ...], ...]]:
    source = _validate_answer_rows(run, source_rows)
    gold = tuple(gold_rows)
    binding = _answer_binding(run, replay, source)
    _require(
        len(source) == len(gold) == QUESTION_COUNT
        and model == DEFAULT_MODEL
        and gateway_url == DEFAULT_GATEWAY_URL
        and type(max_concurrency) is int
        and max_concurrency > 0,
        "linked-repair judge preflight policy changed",
    )
    pending: list[dict[str, Any]] = []
    prompts: list[tuple[dict[str, str], ...]] = []
    input_receipts: list[str] = []
    question_receipts: list[dict[str, str]] = []
    for answer, reference in zip(source, gold, strict=True):
        _require(
            answer["question_id"] == reference.question_id
            and answer["question_sha256"] == reference.question_sha256
            and answer["dated_question_sha256"]
            == reference.dated_question_sha256,
            "linked-repair judge answer/reference order changed",
        )
        messages = tuple(
            dict(value)
            for value in build_judge_prompt(
                reference.dated_question,
                reference.reference,
                str(answer["prediction"]),
            )
        )
        body = {
            "answer_source_row_sha256": answer["source_row_sha256"],
            "category": reference.category,
            "dated_question": reference.dated_question,
            "dated_question_sha256": reference.dated_question_sha256,
            "format": PROMPT_ROW_FORMAT,
            "messages": list(messages),
            "messages_sha256": identity_sha256(list(messages)),
            "prediction": answer["prediction"],
            "prediction_sha256": answer["prediction_sha256"],
            "question_id": answer["question_id"],
            "question_sha256": answer["question_sha256"],
            "reference": reference.reference,
            "reference_sha256": reference.reference_sha256,
        }
        input_receipt = identity_sha256(_judge_input_body(body, messages))
        pending.append({**body, "judge_input_receipt_sha256": input_receipt})
        prompts.append(messages)
        input_receipts.append(input_receipt)
        question_receipts.append(
            {
                "dated_question_sha256": reference.dated_question_sha256,
                "question_id": reference.question_id,
                "question_sha256": reference.question_sha256,
            }
        )
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=DEFAULT_MAX_PROMPT_TOKENS
    )
    _require(
        population.logical_prompt_count
        == population.unique_prompt_count
        == QUESTION_COUNT,
        "linked-repair judge prompts must be eleven unique calls",
    )
    rows: list[dict[str, Any]] = []
    for raw, receipt in zip(pending, population.ordered_rows, strict=True):
        _require(
            raw["messages_sha256"] == receipt.messages_sha256,
            "linked-repair judge prompt receipt changed",
        )
        body = {**raw, "prompt_token_proxy": receipt.prompt_token_proxy}
        rows.append({**body, "prompt_row_receipt_sha256": identity_sha256(body)})
    body = {
        **binding,
        "dataset_file_sha256": LOCKED_LONGMEMEVAL_DATASET_SHA256,
        "format": PREFLIGHT_FORMAT,
        "gateway_url": gateway_url,
        "gold_loaded": True,
        "gold_population_sha256": require_sha256(
            gold_population_sha256, "judge gold population"
        ),
        "judge_input_population_sha256": identity_sha256(input_receipts),
        "max_concurrency": max_concurrency,
        "model": model,
        "ordinal_cli_routing_available": False,
        "physical_provider_calls": 0,
        "prompt_population": population.model_dump(),
        "prompt_population_sha256": population.prompt_population_sha256,
        "prompt_rows": rows,
        "question_count": QUESTION_COUNT,
        "question_population_sha256": identity_sha256(question_receipts),
        "required_authorized_provider_calls": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "source_population_sha256": identity_sha256(
            [row["source_row_sha256"] for row in source]
        ),
        "split_file_sha256": LOCKED_LONGMEMEVAL_SPLIT_MANIFEST_SHA256,
    }
    payload = {**body, "construction_identity_sha256": identity_sha256(body)}
    _require(set(payload) == PREFLIGHT_KEYS, "judge preflight shape changed")
    return payload, tuple(prompts)


def _validate_preflight(
    construction: SealedArtifact,
    replay: SealedArtifact | None = None,
) -> tuple[tuple[tuple[dict[str, str], ...], ...], tuple[dict[str, Any], ...]]:
    payload = construction.payload
    raw_rows = payload.get("prompt_rows")
    _require(
        set(payload) == PREFLIGHT_KEYS
        and payload.get("construction_identity_sha256")
        == identity_sha256(_without(payload, "construction_identity_sha256"))
        and payload.get("format") == PREFLIGHT_FORMAT
        and payload.get("dataset_file_sha256")
        == LOCKED_LONGMEMEVAL_DATASET_SHA256
        and payload.get("split_file_sha256")
        == LOCKED_LONGMEMEVAL_SPLIT_MANIFEST_SHA256
        and payload.get("gold_loaded") is True
        and payload.get("physical_provider_calls") == 0
        and payload.get("ordinal_cli_routing_available") is False
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("model") == DEFAULT_MODEL
        and payload.get("gateway_url") == DEFAULT_GATEWAY_URL
        and type(payload.get("max_concurrency")) is int
        and int(payload["max_concurrency"]) > 0
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("required_authorized_provider_calls") == QUESTION_COUNT
        and type(raw_rows) is list
        and len(raw_rows) == QUESTION_COUNT,
        "sealed linked-repair judge preflight changed",
    )
    if replay is not None:
        _require(
            construction.sha256 == replay.sha256
            and construction.payload == replay.payload,
            "judge preflight replay is not byte-identical",
        )
    for key in ANSWER_BINDING_KEYS:
        require_sha256(payload.get(key), f"judge preflight {key}")
    prompts: list[tuple[dict[str, str], ...]] = []
    rows: list[dict[str, Any]] = []
    input_receipts: list[str] = []
    question_receipts: list[dict[str, str]] = []
    source_receipts: list[str] = []
    for raw in raw_rows:
        _require(type(raw) is dict, "judge preflight row changed type")
        row = dict(raw)
        messages_raw = row.get("messages")
        _require(
            set(row) == PROMPT_ROW_KEYS
            and type(messages_raw) is list
            and row.get("prompt_row_receipt_sha256")
            == identity_sha256(_without(row, "prompt_row_receipt_sha256")),
            "judge prompt row seal changed",
        )
        messages = tuple(dict(value) for value in messages_raw)
        expected_messages = tuple(
            dict(value)
            for value in build_judge_prompt(
                require_text(row.get("dated_question"), "judge dated question"),
                require_text(row.get("reference"), "judge reference"),
                require_text(row.get("prediction"), "judge prediction"),
            )
        )
        _require(
            messages == expected_messages
            and row.get("messages_sha256") == identity_sha256(list(messages))
            and row.get("dated_question_sha256")
            == quote_sha256(str(row["dated_question"]))
            and row.get("question_sha256") == row.get("dated_question_sha256")
            and row.get("reference_sha256") == quote_sha256(str(row["reference"]))
            and row.get("prediction_sha256") == quote_sha256(str(row["prediction"]))
            and row.get("judge_input_receipt_sha256")
            == identity_sha256(_judge_input_body(row, messages))
            and "ordinal" not in row,
            "judge prompt row contains non-contract input",
        )
        for key in (
            "answer_source_row_sha256",
            "dated_question_sha256",
            "judge_input_receipt_sha256",
            "messages_sha256",
            "prediction_sha256",
            "prompt_row_receipt_sha256",
            "question_sha256",
            "reference_sha256",
        ):
            require_sha256(row.get(key), f"judge prompt {key}")
        prompts.append(messages)
        rows.append(row)
        input_receipts.append(str(row["judge_input_receipt_sha256"]))
        source_receipts.append(str(row["answer_source_row_sha256"]))
        question_receipts.append(
            {
                "dated_question_sha256": str(row["dated_question_sha256"]),
                "question_id": require_text(row.get("question_id"), "question ID"),
                "question_sha256": str(row["question_sha256"]),
            }
        )
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=DEFAULT_MAX_PROMPT_TOKENS
    )
    _require(
        len({row["question_id"] for row in rows}) == QUESTION_COUNT
        and population.logical_prompt_count
        == population.unique_prompt_count
        == QUESTION_COUNT
        and population.model_dump() == payload.get("prompt_population")
        and population.prompt_population_sha256
        == payload.get("prompt_population_sha256")
        and payload.get("judge_input_population_sha256")
        == identity_sha256(input_receipts)
        and payload.get("question_population_sha256")
        == identity_sha256(question_receipts)
        and payload.get("source_population_sha256")
        == identity_sha256(source_receipts)
        and payload.get("answer_source_row_population_sha256")
        == payload.get("source_population_sha256"),
        "sealed linked-repair judge population changed",
    )
    return tuple(prompts), tuple(rows)


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
        "linked-repair judge preflight construction",
    )
    replay = _read_expected(
        root / PREFLIGHT_REPLAY_NAME,
        expected_replay_sha256,
        "linked-repair judge preflight replay",
    )
    prompts, rows = _validate_preflight(construction, replay)
    return construction, replay, prompts, rows


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.judge_output_root)
    _require(
        not (output_root / CHECKPOINT_DIR_NAME).exists(),
        "judge preflight requires an absent checkpoint root",
    )
    # Authenticate the gold-free answer pair before opening the benchmark.
    run, replay, source = _load_answer_source(
        answer_root=args.answer_root,
        expected_preflight_construction_sha256=str(
            args.expected_answer_preflight_construction_sha256
        ),
        expected_preflight_replay_sha256=str(
            args.expected_answer_preflight_replay_sha256
        ),
        expected_run_sha256=str(args.expected_answer_run_sha256),
        expected_replay_sha256=str(args.expected_answer_replay_sha256),
    )
    gold, gold_sha = _load_locked_gold(
        source, dataset_path=args.dataset, split_path=args.split
    )
    payload, prompts = build_preflight_payload(
        run,
        replay,
        source,
        gold,
        gold_population_sha256=gold_sha,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    rebuilt, rebuilt_prompts = build_preflight_payload(
        run,
        replay,
        source,
        gold,
        gold_population_sha256=gold_sha,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    _require(
        rebuilt == payload and rebuilt_prompts == prompts,
        "judge preflight rebuild is not deterministic",
    )
    construction, created = publish_sealed_json(
        output_root / PREFLIGHT_NAME, payload
    )
    preflight_replay, replay_created = publish_sealed_json(
        output_root / PREFLIGHT_REPLAY_NAME, rebuilt
    )
    _validate_preflight(construction, preflight_replay)
    return {
        "answer_run_sha256": run.sha256,
        "construction_created": created,
        "preflight_construction_sha256": construction.sha256,
        "preflight_replay_created": replay_created,
        "preflight_replay_sha256": preflight_replay.sha256,
        "question_count": QUESTION_COUNT,
        "required_authorized_provider_calls": QUESTION_COUNT,
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
        "linked-repair judge runtime differs from sealed preflight",
    )
    return FastCompletionRuntime(
        checkpoint_dir=Path(output_root) / CHECKPOINT_DIR_NAME,
        prompt_population=prompts,
        model=DEFAULT_MODEL,
        client=client,
        max_prompt_tokens=DEFAULT_MAX_PROMPT_TOKENS,
        max_new_tokens=JUDGE_MAX_TOKENS,
        max_concurrency=max_concurrency,
        retries=0,
        benchmark_provenance={
            "answer_run_artifact_sha256": preflight.payload[
                "answer_run_artifact_sha256"
            ],
            "arm": FORMAT,
            "authorized_unique_calls": QUESTION_COUNT,
            "experiment_format": JUDGE_FORMAT,
            "gold_population_sha256": preflight.payload["gold_population_sha256"],
            "preflight_construction_artifact_sha256": preflight.sha256,
            "preflight_replay_artifact_sha256": replay.sha256,
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
        "judge checkpoint root must be a regular directory",
    )
    requests: set[str] = set()
    responses: set[str] = set()
    for path in root.iterdir():
        _require(
            path.is_file() and not path.is_symlink(),
            "judge checkpoint root contains foreign state",
        )
        if path.name == ".fast-completion-journal.lock":
            continue
        match = _JOURNAL_FILENAME_RE.fullmatch(path.name)
        _require(match is not None, "judge journal filename changed")
        assert match is not None
        target = requests if match.group("kind") == "request" else responses
        target.add(match.group("key"))
    _require(
        requests == responses,
        "judge request is incomplete; unsafe retry forbidden",
    )
    _require(
        len(requests) <= QUESTION_COUNT,
        "judge checkpoint population exceeds eleven calls",
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
        "judge journals escaped the sealed prompt population",
    )
    return len(records)


def _make_provider_client(api_key: str, gateway_url: str) -> Any:
    return judging._make_provider_client(api_key, gateway_url)  # noqa: SLF001


def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    preflight, replay, prompts, _rows = _read_preflight(
        args.judge_output_root,
        expected_construction_sha256=str(
            args.expected_judge_preflight_construction_sha256
        ),
        expected_replay_sha256=str(args.expected_judge_preflight_replay_sha256),
    )
    _require(
        args.enable_provider is True
        and type(args.authorized_provider_calls) is int
        and 0 <= args.authorized_provider_calls <= QUESTION_COUNT,
        "judge provider requires bounded Sol authorization",
    )
    candidate_hits = _read_only_checkpoint_count(args.judge_output_root)
    remaining = QUESTION_COUNT - candidate_hits
    _require(
        args.authorized_provider_calls == remaining,
        "judge authorization must exactly equal remaining calls",
    )
    checkpoint_hits = _validated_checkpoint_hits(
        preflight,
        replay,
        prompts,
        output_root=args.judge_output_root,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    _require(
        checkpoint_hits == candidate_hits,
        "judge checkpoint count changed after authorization",
    )
    if remaining == 0:
        batch = _checkpoint_batch(
            preflight,
            replay,
            prompts,
            output_root=args.judge_output_root,
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
                output_root=args.judge_output_root,
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
        "judge provider population changed",
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
        "judge materialization requires eleven complete checkpoints",
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
        "judge checkpoint-only batch changed",
    )
    return batch


def _parse_verdict(output: str, question_id: str) -> bool:
    try:
        return parse_binary_judge_verdict(output)
    except RuntimeError as exc:
        raise R7LinkedTerminalRepairJudgeError(
            f"invalid Sol verdict for {question_id}"
        ) from exc


def _judge_payload(
    preflight: SealedArtifact,
    replay: SealedArtifact,
    prompt_rows: Sequence[Mapping[str, Any]],
    batch: FastCompletionBatch,
) -> dict[str, Any]:
    _require(
        len(prompt_rows)
        == len(batch.logical_completions)
        == len(batch.unique_records)
        == QUESTION_COUNT,
        "judge materialization population changed",
    )
    records = {record.messages_sha256: record for record in batch.unique_records}
    _require(len(records) == QUESTION_COUNT, "judge completion identities repeat")
    rows: list[dict[str, Any]] = []
    for prompt, output in zip(prompt_rows, batch.logical_completions, strict=True):
        record = records.get(str(prompt["messages_sha256"]))
        _require(
            record is not None
            and record.completion == output
            and record.checkpoint_hit is True
            and record.physical_call is False
            and record.requested_model == DEFAULT_MODEL
            and record.finish_reason == "stop",
            "judge checkpoint record changed",
        )
        assert record is not None
        verdict = _parse_verdict(output, str(prompt["question_id"]))
        body = {
            "answer_source_row_sha256": prompt["answer_source_row_sha256"],
            "call_key_sha256": record.call_key_sha256,
            "category": prompt["category"],
            "correct": verdict,
            "dated_question_sha256": prompt["dated_question_sha256"],
            "format": JUDGE_ROW_FORMAT,
            "judge_output": output,
            "judge_output_sha256": quote_sha256(output),
            "messages_sha256": prompt["messages_sha256"],
            "normalized_exact_match": exact_match(
                str(prompt["prediction"]), str(prompt["reference"])
            ),
            "normalized_f1": f1_score(
                str(prompt["prediction"]), str(prompt["reference"])
            ),
            "prediction_sha256": prompt["prediction_sha256"],
            "prompt_row_receipt_sha256": prompt["prompt_row_receipt_sha256"],
            "question_id": prompt["question_id"],
            "question_sha256": prompt["question_sha256"],
            "reference_sha256": prompt["reference_sha256"],
            "request_journal_sha256": record.request_journal_sha256,
            "response_journal_sha256": record.response_journal_sha256,
        }
        rows.append({**body, "judge_row_receipt_sha256": identity_sha256(body)})
    correct = sum(bool(row["correct"]) for row in rows)
    binding = {key: preflight.payload[key] for key in ANSWER_BINDING_KEYS}
    body = {
        **binding,
        "aggregate": {
            "accuracy": correct / QUESTION_COUNT,
            "correct": correct,
            "question_count": QUESTION_COUNT,
        },
        "answer_binding_sha256": identity_sha256(binding),
        "completion_batch": batch.model_dump(),
        "format": JUDGE_FORMAT,
        "gold_loaded": True,
        "physical_provider_calls_during_materialization": 0,
        "preflight_construction_artifact_sha256": preflight.sha256,
        "preflight_replay_artifact_sha256": replay.sha256,
        "prompt_population_sha256": preflight.payload["prompt_population_sha256"],
        "question_count": QUESTION_COUNT,
        "question_population_sha256": preflight.payload[
            "question_population_sha256"
        ],
        "questions": rows,
        "retained_transformer_token_state_bytes": 0,
        "source_population_sha256": preflight.payload["source_population_sha256"],
    }
    return {**body, "judge_identity_sha256": identity_sha256(body)}


def _validate_judge(
    preflight: SealedArtifact,
    replay: SealedArtifact,
    payload: Mapping[str, Any],
    prompt_rows: Sequence[Mapping[str, Any]],
    *,
    expected_batch: FastCompletionBatch | None = None,
) -> tuple[dict[str, Any], ...]:
    raw_rows = payload.get("questions")
    aggregate = payload.get("aggregate")
    binding = {key: preflight.payload[key] for key in ANSWER_BINDING_KEYS}
    _require(
        set(payload) == JUDGE_KEYS
        and payload.get("judge_identity_sha256")
        == identity_sha256(_without(payload, "judge_identity_sha256"))
        and payload.get("format") == JUDGE_FORMAT
        and payload.get("gold_loaded") is True
        and payload.get("physical_provider_calls_during_materialization") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("preflight_construction_artifact_sha256") == preflight.sha256
        and payload.get("preflight_replay_artifact_sha256") == replay.sha256
        and payload.get("answer_binding_sha256") == identity_sha256(binding)
        and all(payload.get(key) == value for key, value in binding.items())
        and payload.get("question_count") == QUESTION_COUNT
        and type(raw_rows) is list
        and len(raw_rows) == len(prompt_rows) == QUESTION_COUNT
        and type(aggregate) is dict
        and aggregate.get("question_count") == QUESTION_COUNT,
        "sealed linked-repair judge changed",
    )
    if expected_batch is not None:
        _require(
            payload.get("completion_batch") == expected_batch.model_dump(),
            "judge completion batch changed",
        )
    rows: list[dict[str, Any]] = []
    for raw, prompt in zip(raw_rows, prompt_rows, strict=True):
        _require(type(raw) is dict, "judge verdict row changed type")
        output = require_text(raw.get("judge_output"), "Sol judge output")
        verdict = _parse_verdict(output, str(prompt["question_id"]))
        _require(
            set(raw) == JUDGE_ROW_KEYS
            and raw.get("judge_row_receipt_sha256")
            == identity_sha256(_without(raw, "judge_row_receipt_sha256"))
            and raw.get("format") == JUDGE_ROW_FORMAT
            and raw.get("correct") is verdict
            and raw.get("judge_output_sha256") == quote_sha256(output)
            and raw.get("question_id") == prompt.get("question_id")
            and raw.get("question_sha256") == prompt.get("question_sha256")
            and raw.get("dated_question_sha256")
            == prompt.get("dated_question_sha256")
            and raw.get("messages_sha256") == prompt.get("messages_sha256")
            and raw.get("prediction_sha256") == prompt.get("prediction_sha256")
            and raw.get("reference_sha256") == prompt.get("reference_sha256")
            and raw.get("answer_source_row_sha256")
            == prompt.get("answer_source_row_sha256")
            and raw.get("prompt_row_receipt_sha256")
            == prompt.get("prompt_row_receipt_sha256"),
            f"judge verdict row changed for {prompt.get('question_id')}",
        )
        for key in (
            "answer_source_row_sha256",
            "call_key_sha256",
            "dated_question_sha256",
            "judge_output_sha256",
            "judge_row_receipt_sha256",
            "messages_sha256",
            "prediction_sha256",
            "prompt_row_receipt_sha256",
            "question_sha256",
            "reference_sha256",
            "request_journal_sha256",
            "response_journal_sha256",
        ):
            require_sha256(raw.get(key), f"judge verdict {key}")
        rows.append(dict(raw))
    correct = sum(bool(row["correct"]) for row in rows)
    _require(
        len({row["question_id"] for row in rows}) == QUESTION_COUNT
        and aggregate.get("correct") == correct
        and aggregate.get("accuracy") == correct / QUESTION_COUNT,
        "judge aggregate changed",
    )
    return tuple(rows)


def _score_payload(
    preflight: SealedArtifact,
    replay: SealedArtifact,
    judge: SealedArtifact,
    judge_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    score_rows: list[dict[str, Any]] = []
    for row in judge_rows:
        body = {
            "correct": row["correct"],
            "format": SCORE_ROW_FORMAT,
            "judge_row_receipt_sha256": row["judge_row_receipt_sha256"],
            "question_id": row["question_id"],
            "question_sha256": row["question_sha256"],
        }
        score_rows.append({**body, "score_row_receipt_sha256": identity_sha256(body)})
    correct = sum(bool(row["correct"]) for row in score_rows)
    binding = {key: preflight.payload[key] for key in ANSWER_BINDING_KEYS}
    body = {
        **binding,
        "accuracy": correct / QUESTION_COUNT,
        "answer_binding_sha256": identity_sha256(binding),
        "correct": correct,
        "format": SCORE_FORMAT,
        "gold_loaded": True,
        "judge_artifact_sha256": judge.sha256,
        "physical_provider_calls_during_scoring": 0,
        "preflight_construction_artifact_sha256": preflight.sha256,
        "preflight_replay_artifact_sha256": replay.sha256,
        "question_count": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "score_population_sha256": identity_sha256(
            [row["score_row_receipt_sha256"] for row in score_rows]
        ),
        "score_rows": score_rows,
    }
    return {**body, "score_identity_sha256": identity_sha256(body)}


def _validate_score(
    preflight: SealedArtifact,
    replay: SealedArtifact,
    judge: SealedArtifact,
    payload: Mapping[str, Any],
    judge_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    expected = _score_payload(preflight, replay, judge, judge_rows)
    raw_rows = payload.get("score_rows")
    _require(
        set(payload) == SCORE_KEYS
        and dict(payload) == expected
        and payload.get("score_identity_sha256")
        == identity_sha256(_without(payload, "score_identity_sha256"))
        and type(raw_rows) is list
        and len(raw_rows) == QUESTION_COUNT,
        "sealed linked-repair score changed",
    )
    return tuple(dict(row) for row in raw_rows)


def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    preflight, replay, prompts, rows = _read_preflight(
        args.judge_output_root,
        expected_construction_sha256=str(
            args.expected_judge_preflight_construction_sha256
        ),
        expected_replay_sha256=str(args.expected_judge_preflight_replay_sha256),
    )
    batch = _complete_checkpoint_batch(
        preflight,
        replay,
        prompts,
        output_root=args.judge_output_root,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    judge_payload = _judge_payload(preflight, replay, rows, batch)
    verdicts = _validate_judge(
        preflight, replay, judge_payload, rows, expected_batch=batch
    )
    root = Path(args.judge_output_root)
    judge, judge_created = publish_sealed_json(root / JUDGE_NAME, judge_payload)
    score_payload = _score_payload(preflight, replay, judge, verdicts)
    _validate_score(preflight, replay, judge, score_payload, verdicts)
    score, score_created = publish_sealed_json(root / SCORE_NAME, score_payload)
    return {
        "accuracy": score_payload["accuracy"],
        "checkpoint_hits": QUESTION_COUNT,
        "correct": score_payload["correct"],
        "judge_created": judge_created,
        "judge_sha256": judge.sha256,
        "physical_provider_calls": 0,
        "question_count": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "score_created": score_created,
        "score_sha256": score.sha256,
    }


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    preflight, replay, prompts, rows = _read_preflight(
        args.judge_output_root,
        expected_construction_sha256=str(
            args.expected_judge_preflight_construction_sha256
        ),
        expected_replay_sha256=str(args.expected_judge_preflight_replay_sha256),
    )
    batch = _complete_checkpoint_batch(
        preflight,
        replay,
        prompts,
        output_root=args.judge_output_root,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    root = Path(args.judge_output_root)
    rebuilt = _judge_payload(preflight, replay, rows, batch)
    judge = _read_expected(
        root / JUDGE_NAME, str(args.expected_judge_sha256), "linked-repair Sol judge"
    )
    _require(judge.payload == rebuilt, "judge differs from checkpoint-only replay")
    verdicts = _validate_judge(preflight, replay, judge.payload, rows, expected_batch=batch)
    score = _read_expected(
        root / SCORE_NAME, str(args.expected_score_sha256), "linked-repair Sol score"
    )
    rebuilt_score = _score_payload(preflight, replay, judge, verdicts)
    _require(score.payload == rebuilt_score, "score differs from deterministic replay")
    _validate_score(preflight, replay, judge, score.payload, verdicts)
    judge_replay, _ = publish_sealed_json(root / JUDGE_REPLAY_NAME, rebuilt)
    score_replay, _ = publish_sealed_json(root / SCORE_REPLAY_NAME, rebuilt_score)
    _require(
        judge_replay.sha256 == judge.sha256 and score_replay.sha256 == score.sha256,
        "judge/score replay is not byte-identical",
    )
    return {
        "byte_identical": True,
        "judge_replay_sha256": judge_replay.sha256,
        "judge_sha256": judge.sha256,
        "physical_provider_calls": 0,
        "retained_transformer_token_state_bytes": 0,
        "score_replay_sha256": score_replay.sha256,
        "score_sha256": score.sha256,
    }


def load_verified_judge_run(
    output_root: str | Path,
    *,
    expected_preflight_construction_sha256: str,
    expected_preflight_replay_sha256: str,
    expected_judge_sha256: str,
    expected_judge_replay_sha256: str,
    expected_score_sha256: str,
    expected_score_replay_sha256: str,
) -> tuple[SealedArtifact, SealedArtifact, tuple[dict[str, Any], ...]]:
    root = Path(output_root)
    preflight, replay, _prompts, rows = _read_preflight(
        root,
        expected_construction_sha256=expected_preflight_construction_sha256,
        expected_replay_sha256=expected_preflight_replay_sha256,
    )
    judge = _read_expected(root / JUDGE_NAME, expected_judge_sha256, "Sol judge")
    judge_replay = _read_expected(
        root / JUDGE_REPLAY_NAME, expected_judge_replay_sha256, "Sol judge replay"
    )
    _require(
        judge.sha256 == judge_replay.sha256 and judge.payload == judge_replay.payload,
        "judge run/replay is not byte-identical",
    )
    verdicts = _validate_judge(preflight, replay, judge.payload, rows)
    score = _read_expected(root / SCORE_NAME, expected_score_sha256, "Sol score")
    score_replay = _read_expected(
        root / SCORE_REPLAY_NAME, expected_score_replay_sha256, "Sol score replay"
    )
    _require(
        score.sha256 == score_replay.sha256 and score.payload == score_replay.payload,
        "score run/replay is not byte-identical",
    )
    _validate_score(preflight, replay, judge, score.payload, verdicts)
    return judge, score, verdicts


def _add_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--judge-output-root", type=Path, default=DEFAULT_JUDGE_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument(
        "--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY
    )


def _add_preflight_binding(parser: argparse.ArgumentParser) -> None:
    _add_runtime(parser)
    parser.add_argument(
        "--expected-judge-preflight-construction-sha256", required=True
    )
    parser.add_argument("--expected-judge-preflight-replay-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight")
    _add_runtime(preflight)
    preflight.add_argument("--answer-root", type=Path, default=DEFAULT_ANSWER_ROOT)
    preflight.add_argument(
        "--expected-answer-preflight-construction-sha256",
        default=EXPECTED_ANSWER_PREFLIGHT_SHA256,
    )
    preflight.add_argument(
        "--expected-answer-preflight-replay-sha256",
        default=EXPECTED_ANSWER_PREFLIGHT_SHA256,
    )
    preflight.add_argument(
        "--expected-answer-run-sha256", default=EXPECTED_ANSWER_RUN_SHA256
    )
    preflight.add_argument(
        "--expected-answer-replay-sha256", default=EXPECTED_ANSWER_RUN_SHA256
    )
    preflight.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    preflight.add_argument("--split", type=Path, default=DEFAULT_SPLIT)

    provider = commands.add_parser("provider-run")
    _add_preflight_binding(provider)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, required=True)
    provider.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)

    materialize = commands.add_parser("materialize")
    _add_preflight_binding(materialize)

    replay = commands.add_parser("replay")
    _add_preflight_binding(replay)
    replay.add_argument("--expected-judge-sha256", required=True)
    replay.add_argument("--expected-score-sha256", required=True)
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
    "DEFAULT_ANSWER_ROOT",
    "DEFAULT_JUDGE_ROOT",
    "DEFAULT_MODEL",
    "EXPECTED_ANSWER_PREFLIGHT_SHA256",
    "EXPECTED_ANSWER_RUN_SHA256",
    "FORMAT",
    "JUDGE_NAME",
    "JUDGE_REPLAY_NAME",
    "PREFLIGHT_NAME",
    "PREFLIGHT_REPLAY_NAME",
    "QUESTION_COUNT",
    "R7LinkedTerminalRepairJudgeError",
    "SCORE_NAME",
    "SCORE_REPLAY_NAME",
    "build_parser",
    "build_preflight_payload",
    "load_verified_judge_run",
    "main",
    "run_materialize",
    "run_preflight",
    "run_provider",
    "run_replay",
]
