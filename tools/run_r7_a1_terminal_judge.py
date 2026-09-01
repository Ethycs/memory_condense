#!/usr/bin/env python3
"""Judge one sealed R7 A1 terminal-answer arm with exactly eleven Sol calls.

The answer run and its byte-identical replay are authenticated before benchmark
references are opened.  A preflight selects one complete arm by its sealed arm
label and joins its eleven predictions to the locked validation population by
question ID.  Provider messages are then reconstructed from exactly the dated
question, reference answer, and that one sealed prediction.  Evidence, facts,
handles, answer prompts, and caller-selected ordinals never cross the judge
boundary.

Each arm owns a distinct preflight, release, checkpoint directory, judge, and
score replay.  Provider execution has zero retries and is permitted only after
an explicit release; an incomplete request/response pair permanently fails
closed rather than risking a duplicate call.
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
from tools import run_r7_a1_terminal_answer as answer_cli  # noqa: E402
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


FORMAT = "memory-condense-r7-a1-terminal-sol-judge-lifecycle-v1"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight-v1"
RELEASE_FORMAT = f"{FORMAT}-provider-release-v1"
JUDGE_FORMAT = f"{FORMAT}-judge-v1"
SCORE_FORMAT = f"{FORMAT}-score-v1"
PROMPT_ROW_FORMAT = f"{FORMAT}-prompt-row-v1"
JUDGE_ROW_FORMAT = f"{FORMAT}-judge-row-v1"
SCORE_ROW_FORMAT = f"{FORMAT}-score-row-v1"
JOURNAL_OWNER_FORMAT = f"{FORMAT}-journal-owner-v1"

PREFLIGHT_NAME = "r7-a1-terminal-sol-judge-preflight-v1.json"
RELEASE_NAME = "r7-a1-terminal-sol-judge-provider-release-v1.json"
JUDGE_NAME = "r7-a1-terminal-sol-judge-v1.json"
JUDGE_REPLAY_NAME = "r7-a1-terminal-sol-judge-replay-v1.json"
SCORE_NAME = "r7-a1-terminal-sol-score-v1.json"
SCORE_REPLAY_NAME = "r7-a1-terminal-sol-score-replay-v1.json"
CHECKPOINT_DIR_NAME = "sol-r7-a1-terminal-judge-v1-calls"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANSWER_ROOT = answer_cli.DEFAULT_OUTPUT_ROOT
DEFAULT_MODEL = judging.DEFAULT_SOL_GATEWAY_MODEL
DEFAULT_GATEWAY_URL = live.DEFAULT_GATEWAY_URL
DEFAULT_API_KEY_ENV = live.DEFAULT_API_KEY_ENV
DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_MAX_PROMPT_TOKENS = judging.DEFAULT_MAX_JUDGE_PROMPT_TOKENS
QUESTION_COUNT = 11

ANSWER_JUDGE_ROW_KEYS = {
    "arm",
    "dated_question_sha256",
    "format",
    "prediction",
    "prediction_sha256",
    "question_id",
    "question_sha256",
    "source_row_sha256",
}
ANSWER_SOURCE_SHA_KEYS = (
    "compiler_outputs_artifact_sha256",
    "compiler_outputs_replay_artifact_sha256",
    "source_a1_construction_artifact_sha256",
    "source_a1_replay_artifact_sha256",
)
ANSWER_BINDING_KEYS = (
    "answer_preflight_construction_artifact_sha256",
    "answer_preflight_replay_artifact_sha256",
    "answer_release_authorization_artifact_sha256",
    "answer_run_artifact_sha256",
    "answer_replay_artifact_sha256",
    "answer_arm_prediction_population_sha256",
    "answer_arm_source_population_sha256",
    *tuple(f"answer_{key}" for key in ANSWER_SOURCE_SHA_KEYS),
)

PROMPT_ROW_KEYS = {
    "answer_source_row_sha256",
    "arm",
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
    "answer_arm",
    "answer_binding_sha256",
    "answer_root",
    "answer_root_sha256",
    "caller_ordinal_routing_available",
    "dataset_file_sha256",
    "fixed_population_derivation",
    "format",
    "gateway_url",
    "gold_loaded",
    "gold_population_sha256",
    "judge_input_population_sha256",
    "max_concurrency",
    "model",
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
RELEASE_KEYS = {
    "answer_arm",
    "answer_binding_sha256",
    "approval_opt_in",
    "checkpoint_root",
    "checkpoint_root_sha256",
    "format",
    "gateway_url",
    "gold_loaded",
    "gold_population_sha256",
    "journal_owner_format",
    "journal_owner_identity_sha256",
    "judge_output_root",
    "judge_output_root_sha256",
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
    "retry_count",
    "unsafe_retry_policy",
}.union(ANSWER_BINDING_KEYS)
JUDGE_ROW_KEYS = {
    "answer_source_row_sha256",
    "arm",
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
    "answer_arm",
    "answer_binding_sha256",
    "completion_batch",
    "format",
    "gold_loaded",
    "gold_population_sha256",
    "journal_owner_identity_sha256",
    "physical_provider_calls_during_materialization",
    "preflight_artifact_sha256",
    "prompt_population_sha256",
    "question_count",
    "question_population_sha256",
    "questions",
    "release_authorization_artifact_sha256",
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
    "answer_arm",
    "answer_binding_sha256",
    "correct",
    "format",
    "gold_loaded",
    "gold_population_sha256",
    "judge_artifact_sha256",
    "judge_replay_artifact_sha256",
    "physical_provider_calls_during_scoring",
    "preflight_artifact_sha256",
    "prompt_population_sha256",
    "question_count",
    "question_population_sha256",
    "release_authorization_artifact_sha256",
    "retained_transformer_token_state_bytes",
    "score_population_sha256",
    "score_rows",
    "source_population_sha256",
}.union(ANSWER_BINDING_KEYS)

_JOURNAL_FILENAME_RE = re.compile(
    r"^(?P<key>[0-9a-f]{64})\.(?P<kind>request|response)\.json$"
)


class R7A1TerminalJudgeError(MatchedEvalContractError):
    """An answer, reference, prompt, release, journal, or score changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise R7A1TerminalJudgeError(message)


def _canonical_root(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve(strict=False)))


def _arm(value: object) -> str:
    arm = require_text(value, "A1 terminal judge arm")
    labels = tuple(answer_cli.ARM_LABELS)
    _require(
        len(labels) == 3 and len(set(labels)) == 3 and arm in labels,
        "A1 terminal judge arm is outside the sealed three-arm factorial",
    )
    return arm


def _output_root(args: argparse.Namespace) -> Path:
    explicit = getattr(args, "judge_output_root", None)
    if explicit is not None:
        return Path(explicit)
    return Path(args.answer_root) / f"sol-judge-v1-{_arm(args.answer_arm)}"


def _read_expected(path: str | Path, expected: str, label: str) -> SealedArtifact:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == require_sha256(expected, label),
        f"{label} artifact changed",
    )
    return artifact


def _without_receipt(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    return {name: child for name, child in value.items() if name != key}


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
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
) -> tuple[dict[str, Any], ...]:
    selected = tuple(dict(row) for row in rows if row.get("arm") == arm)
    sealed_rows_raw = run.payload.get("judge_rows")
    _require(
        type(sealed_rows_raw) is list
        and len(sealed_rows_raw)
        == QUESTION_COUNT * len(tuple(answer_cli.ARM_LABELS))
        and all(type(row) is dict for row in sealed_rows_raw),
        "A1 terminal answer sealed judge rows changed",
    )
    sealed_rows = tuple(dict(row) for row in sealed_rows_raw)
    sealed_selected = tuple(row for row in sealed_rows if row.get("arm") == arm)
    _require(
        len(selected) == QUESTION_COUNT
        and len(rows)
        in {
            QUESTION_COUNT,
            QUESTION_COUNT * len(tuple(answer_cli.ARM_LABELS)),
        }
        and (len(rows) != QUESTION_COUNT or len(selected) == len(rows))
        and (
            tuple(dict(row) for row in rows) == sealed_rows
            if len(rows) != QUESTION_COUNT
            else selected == sealed_selected
        )
        and run.payload.get("judge_row_population_sha256")
        == identity_sha256(list(sealed_rows))
        and len({row.get("question_id") for row in selected}) == QUESTION_COUNT
        and len({row.get("question_sha256") for row in selected}) == QUESTION_COUNT,
        "A1 terminal judge source population changed",
    )
    source_receipts: list[str] = []
    prediction_receipts: list[str] = []
    for row in selected:
        prediction = require_text(row.get("prediction"), "sealed prediction")
        _require(
            set(row) == ANSWER_JUDGE_ROW_KEYS
            and row.get("format") == answer_cli.JUDGE_ROW_FORMAT
            and row.get("arm") == arm
            and "ordinal" not in row
            and row.get("prediction_sha256") == quote_sha256(prediction),
            "A1 terminal judge source row changed",
        )
        for key in (
            "dated_question_sha256",
            "prediction_sha256",
            "question_sha256",
            "source_row_sha256",
        ):
            require_sha256(row.get(key), f"A1 terminal source {key}")
        require_text(row.get("question_id"), "A1 terminal source question ID")
        source_receipts.append(str(row["source_row_sha256"]))
        prediction_receipts.append(str(row["prediction_sha256"]))
    populations = run.payload.get("arm_prediction_population_sha256s")
    _require(
        type(populations) is dict
        and set(populations) == set(answer_cli.ARM_LABELS)
        and populations.get(arm) == identity_sha256(prediction_receipts),
        "A1 terminal answer arm prediction population changed",
    )
    return selected


def _answer_binding(
    run: SealedArtifact,
    replay: SealedArtifact,
    source_rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
) -> dict[str, str]:
    _require(
        run.sha256 == replay.sha256 and run.payload == replay.payload,
        "A1 terminal answer run/replay is not byte-identical",
    )
    source = _validate_answer_rows(run, source_rows, arm=arm)
    prediction_population = identity_sha256(
        [row["prediction_sha256"] for row in source]
    )
    binding: dict[str, str] = {
        "answer_preflight_construction_artifact_sha256": require_sha256(
            run.payload.get("preflight_construction_artifact_sha256"),
            "A1 terminal answer preflight construction",
        ),
        "answer_preflight_replay_artifact_sha256": require_sha256(
            run.payload.get("preflight_replay_artifact_sha256"),
            "A1 terminal answer preflight replay",
        ),
        "answer_release_authorization_artifact_sha256": require_sha256(
            run.payload.get("release_authorization_artifact_sha256"),
            "A1 terminal answer release",
        ),
        "answer_run_artifact_sha256": run.sha256,
        "answer_replay_artifact_sha256": replay.sha256,
        "answer_arm_prediction_population_sha256": prediction_population,
        "answer_arm_source_population_sha256": identity_sha256(
            [row["source_row_sha256"] for row in source]
        ),
    }
    for key in ANSWER_SOURCE_SHA_KEYS:
        binding[f"answer_{key}"] = require_sha256(
            run.payload.get(key), f"A1 terminal answer {key}"
        )
    _require(
        set(binding) == set(ANSWER_BINDING_KEYS),
        "A1 terminal answer binding shape changed",
    )
    return binding


def _load_answer_source(
    *,
    answer_root: str | Path,
    expected_preflight_construction_sha256: str,
    expected_preflight_replay_sha256: str,
    expected_release_sha256: str,
    expected_run_sha256: str,
    expected_replay_sha256: str,
    arm: str,
) -> tuple[
    SealedArtifact,
    SealedArtifact,
    tuple[dict[str, Any], ...],
    dict[str, str],
]:
    run, replay, rows = answer_cli.load_verified_answer_run(
        answer_root,
        expected_preflight_construction_sha256=(
            expected_preflight_construction_sha256
        ),
        expected_preflight_replay_sha256=expected_preflight_replay_sha256,
        expected_release_sha256=expected_release_sha256,
        expected_run_sha256=expected_run_sha256,
        expected_replay_sha256=expected_replay_sha256,
    )
    selected = _validate_answer_rows(run, rows, arm=arm)
    binding = _answer_binding(run, replay, rows, arm=arm)
    return run, replay, selected, binding


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
        "A1 terminal judge locked dataset/split changed",
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
        len(selected) == len(questions) == 100
        and len(by_id) == 100
        and len(source_rows) == QUESTION_COUNT,
        "A1 terminal judge validation population changed",
    )
    result: list[_GoldRow] = []
    receipts: list[dict[str, str]] = []
    for source in source_rows:
        question_id = require_text(source.get("question_id"), "judge question ID")
        question = by_id.get(question_id)
        _require(question is not None, "A1 terminal answer question is not validation")
        assert question is not None
        dated_sha = quote_sha256(question.dated_question)
        # The sealed A1 answer boundary hashes the exact provider-facing dated
        # question into both question_sha256 fields.  The undated benchmark
        # wording never entered that answer population, so joining it here
        # would reject an otherwise exact question-ID/date binding.
        question_sha = dated_sha
        reference_sha = quote_sha256(question.answer)
        _require(
            source.get("question_sha256") == question_sha
            and source.get("dated_question_sha256") == dated_sha,
            "A1 terminal answer question/reference join changed",
        )
        category = str(question.category or "uncategorized")
        result.append(
            _GoldRow(
                question_id=question_id,
                question_sha256=question_sha,
                dated_question=question.dated_question,
                dated_question_sha256=dated_sha,
                reference=question.answer,
                reference_sha256=reference_sha,
                category=category,
            )
        )
        receipts.append(
            {
                "category": category,
                "dated_question_sha256": dated_sha,
                "question_id": question_id,
                "question_sha256": question_sha,
                "reference_sha256": reference_sha,
            }
        )
    _require(
        len({row.question_id for row in result}) == QUESTION_COUNT,
        "A1 terminal judge reference identities repeat",
    )
    return tuple(result), identity_sha256(receipts)


def _judge_input_body(
    row: Mapping[str, Any], messages: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    return {
        "arm": row["arm"],
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
    answer_root: str | Path,
    arm: str,
    gold_population_sha256: str,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> tuple[dict[str, Any], tuple[tuple[dict[str, str], ...], ...]]:
    arm = _arm(arm)
    source = _validate_answer_rows(run, source_rows, arm=arm)
    gold = tuple(gold_rows)
    binding = _answer_binding(run, replay, source_rows, arm=arm)
    _require(
        len(source) == len(gold) == QUESTION_COUNT
        and model == DEFAULT_MODEL
        and gateway_url == DEFAULT_GATEWAY_URL
        and type(max_concurrency) is int
        and max_concurrency > 0,
        "A1 terminal judge preflight policy changed",
    )
    pending: list[dict[str, Any]] = []
    prompts: list[tuple[dict[str, str], ...]] = []
    question_receipts: list[dict[str, str]] = []
    input_receipts: list[str] = []
    for answer, reference in zip(source, gold, strict=True):
        _require(
            answer["question_id"] == reference.question_id
            and answer["question_sha256"] == reference.question_sha256
            and answer["dated_question_sha256"]
            == reference.dated_question_sha256,
            "A1 terminal judge answer/reference order changed",
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
            "arm": arm,
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
        "A1 terminal judge requires eleven unique prompts",
    )
    rows: list[dict[str, Any]] = []
    for body, receipt in zip(pending, population.ordered_rows, strict=True):
        _require(
            body["messages_sha256"] == receipt.messages_sha256,
            "A1 terminal judge prompt receipt changed",
        )
        unsigned = {**body, "prompt_token_proxy": receipt.prompt_token_proxy}
        rows.append(
            {**unsigned, "prompt_row_receipt_sha256": identity_sha256(unsigned)}
        )
    canonical_answer_root = _canonical_root(answer_root)
    payload = {
        **binding,
        "answer_arm": arm,
        "answer_binding_sha256": identity_sha256(binding),
        "answer_root": canonical_answer_root,
        "answer_root_sha256": identity_sha256(
            {"canonical_root": canonical_answer_root}
        ),
        "caller_ordinal_routing_available": False,
        "dataset_file_sha256": LOCKED_LONGMEMEVAL_DATASET_SHA256,
        "fixed_population_derivation": (
            "sealed_answer_arm_question_id_join_to_locked_validation"
        ),
        "format": PREFLIGHT_FORMAT,
        "gateway_url": gateway_url,
        "gold_loaded": True,
        "gold_population_sha256": require_sha256(
            gold_population_sha256, "A1 terminal judge gold population"
        ),
        "judge_input_population_sha256": identity_sha256(input_receipts),
        "max_concurrency": max_concurrency,
        "model": model,
        "physical_provider_calls": 0,
        "prompt_population": population.model_dump(),
        "prompt_population_sha256": population.prompt_population_sha256,
        "prompt_rows": rows,
        "question_count": QUESTION_COUNT,
        "question_population_sha256": identity_sha256(question_receipts),
        "required_authorized_provider_calls": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "source_population_sha256": binding[
            "answer_arm_source_population_sha256"
        ],
        "split_file_sha256": LOCKED_LONGMEMEVAL_SPLIT_MANIFEST_SHA256,
    }
    _require(set(payload) == PREFLIGHT_KEYS, "judge preflight shape changed")
    return payload, tuple(prompts)


def _validate_preflight(
    artifact: SealedArtifact,
    *,
    expected_arm: str | None = None,
) -> tuple[tuple[tuple[dict[str, str], ...], ...], tuple[dict[str, Any], ...]]:
    payload = artifact.payload
    arm = _arm(payload.get("answer_arm"))
    if expected_arm is not None:
        _require(arm == _arm(expected_arm), "sealed judge arm changed")
    binding = {key: payload.get(key) for key in ANSWER_BINDING_KEYS}
    for key, value in binding.items():
        require_sha256(value, f"sealed judge {key}")
    raw_rows = payload.get("prompt_rows")
    _require(
        set(payload) == PREFLIGHT_KEYS
        and payload.get("format") == PREFLIGHT_FORMAT
        and payload.get("answer_binding_sha256") == identity_sha256(binding)
        and payload.get("answer_root_sha256")
        == identity_sha256(
            {
                "canonical_root": require_text(
                    payload.get("answer_root"), "sealed answer root"
                )
            }
        )
        and payload.get("caller_ordinal_routing_available") is False
        and payload.get("fixed_population_derivation")
        == "sealed_answer_arm_question_id_join_to_locked_validation"
        and payload.get("dataset_file_sha256")
        == LOCKED_LONGMEMEVAL_DATASET_SHA256
        and payload.get("split_file_sha256")
        == LOCKED_LONGMEMEVAL_SPLIT_MANIFEST_SHA256
        and payload.get("gold_loaded") is True
        and payload.get("physical_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("model") == DEFAULT_MODEL
        and payload.get("gateway_url") == DEFAULT_GATEWAY_URL
        and type(payload.get("max_concurrency")) is int
        and int(payload["max_concurrency"]) > 0
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("required_authorized_provider_calls") == QUESTION_COUNT
        and payload.get("source_population_sha256")
        == binding["answer_arm_source_population_sha256"]
        and type(raw_rows) is list
        and len(raw_rows) == QUESTION_COUNT,
        "sealed A1 terminal judge preflight changed",
    )
    for key in (
        "gold_population_sha256",
        "judge_input_population_sha256",
        "prompt_population_sha256",
        "question_population_sha256",
        "source_population_sha256",
    ):
        require_sha256(payload.get(key), f"sealed judge {key}")

    prompts: list[tuple[dict[str, str], ...]] = []
    rows: list[dict[str, Any]] = []
    input_receipts: list[str] = []
    question_receipts: list[dict[str, str]] = []
    source_receipts: list[str] = []
    prediction_receipts: list[str] = []
    question_ids: list[str] = []
    for raw in raw_rows:
        _require(type(raw) is dict, "sealed judge prompt row changed type")
        row = dict(raw)
        declared = row.pop("prompt_row_receipt_sha256", None)
        dated_question = require_text(
            raw.get("dated_question"), "sealed dated question"
        )
        reference = require_text(raw.get("reference"), "sealed reference")
        prediction = require_text(raw.get("prediction"), "sealed prediction")
        expected_messages = tuple(
            dict(value)
            for value in build_judge_prompt(
                dated_question,
                reference,
                prediction,
            )
        )
        messages = raw.get("messages")
        _require(
            set(raw) == PROMPT_ROW_KEYS
            and declared == identity_sha256(row)
            and raw.get("format") == PROMPT_ROW_FORMAT
            and raw.get("arm") == arm
            and "ordinal" not in raw
            and raw.get("dated_question_sha256")
            == quote_sha256(dated_question)
            and raw.get("reference_sha256") == quote_sha256(reference)
            and raw.get("prediction_sha256") == quote_sha256(prediction)
            and type(messages) is list
            and tuple(dict(value) for value in messages) == expected_messages
            and raw.get("messages_sha256")
            == identity_sha256(list(expected_messages))
            and raw.get("judge_input_receipt_sha256")
            == identity_sha256(_judge_input_body(raw, expected_messages)),
            "sealed judge prompt contains non-contract input",
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
            require_sha256(raw.get(key), f"sealed judge prompt {key}")
        question_id = require_text(raw.get("question_id"), "judge question ID")
        question_ids.append(question_id)
        input_receipts.append(str(raw["judge_input_receipt_sha256"]))
        source_receipts.append(str(raw["answer_source_row_sha256"]))
        prediction_receipts.append(str(raw["prediction_sha256"]))
        question_receipts.append(
            {
                "dated_question_sha256": str(raw["dated_question_sha256"]),
                "question_id": question_id,
                "question_sha256": str(raw["question_sha256"]),
            }
        )
        prompts.append(expected_messages)
        rows.append(dict(raw))
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=DEFAULT_MAX_PROMPT_TOKENS
    )
    _require(
        len(set(question_ids)) == QUESTION_COUNT
        and len(set(input_receipts)) == QUESTION_COUNT
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
        and binding["answer_arm_prediction_population_sha256"]
        == identity_sha256(prediction_receipts),
        "sealed judge prompt population changed",
    )
    return tuple(prompts), tuple(rows)


def _read_preflight(
    output_root: str | Path,
    expected_sha256: str,
    *,
    expected_arm: str,
) -> tuple[
    SealedArtifact,
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
]:
    artifact = _read_expected(
        Path(output_root) / PREFLIGHT_NAME,
        expected_sha256,
        "A1 terminal judge preflight",
    )
    prompts, rows = _validate_preflight(artifact, expected_arm=expected_arm)
    return artifact, prompts, rows


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    arm = _arm(args.answer_arm)
    output_root = _output_root(args)
    _require(
        _canonical_root(output_root) != _canonical_root(args.answer_root),
        "judge output root must differ from the answer root",
    )
    _require(
        not (output_root / CHECKPOINT_DIR_NAME).exists(),
        "judge preflight requires a fresh absent checkpoint root",
    )
    # The gold-free answer authority is deliberately verified before references.
    run, replay, source, _binding = _load_answer_source(
        answer_root=args.answer_root,
        expected_preflight_construction_sha256=str(
            args.expected_answer_preflight_construction_sha256
        ),
        expected_preflight_replay_sha256=str(
            args.expected_answer_preflight_replay_sha256
        ),
        expected_release_sha256=str(args.expected_answer_release_sha256),
        expected_run_sha256=str(args.expected_answer_run_sha256),
        expected_replay_sha256=str(args.expected_answer_replay_sha256),
        arm=arm,
    )
    gold, gold_sha = _load_locked_gold(
        source, dataset_path=args.dataset, split_path=args.split
    )
    payload, _prompts = build_preflight_payload(
        run,
        replay,
        source,
        gold,
        answer_root=args.answer_root,
        arm=arm,
        gold_population_sha256=gold_sha,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    artifact, created = publish_sealed_json(output_root / PREFLIGHT_NAME, payload)
    _validate_preflight(artifact, expected_arm=arm)
    return {
        "answer_arm": arm,
        "answer_run_sha256": run.sha256,
        "created": created,
        "physical_provider_calls": 0,
        "preflight_sha256": artifact.sha256,
        "question_count": QUESTION_COUNT,
        "required_authorized_provider_calls": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
    }


def _journal_owner_body(
    preflight: SealedArtifact,
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    root = _canonical_root(output_root)
    checkpoint = _canonical_root(Path(output_root) / CHECKPOINT_DIR_NAME)
    return {
        "answer_arm": preflight.payload["answer_arm"],
        "answer_run_artifact_sha256": preflight.payload[
            "answer_run_artifact_sha256"
        ],
        "checkpoint_root": checkpoint,
        "checkpoint_root_sha256": identity_sha256(
            {"canonical_root": checkpoint}
        ),
        "format": JOURNAL_OWNER_FORMAT,
        "judge_output_root": root,
        "judge_output_root_sha256": identity_sha256({"canonical_root": root}),
        "model": preflight.payload["model"],
        "preflight_artifact_sha256": preflight.sha256,
        "prompt_population_sha256": preflight.payload[
            "prompt_population_sha256"
        ],
        "question_count": QUESTION_COUNT,
    }


def _release_payload(
    preflight: SealedArtifact,
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    owner = _journal_owner_body(preflight, output_root=output_root)
    body = {
        **{key: preflight.payload[key] for key in ANSWER_BINDING_KEYS},
        "answer_arm": preflight.payload["answer_arm"],
        "answer_binding_sha256": preflight.payload["answer_binding_sha256"],
        "approval_opt_in": True,
        "checkpoint_root": owner["checkpoint_root"],
        "checkpoint_root_sha256": owner["checkpoint_root_sha256"],
        "format": RELEASE_FORMAT,
        "gateway_url": preflight.payload["gateway_url"],
        "gold_loaded": True,
        "gold_population_sha256": preflight.payload["gold_population_sha256"],
        "journal_owner_format": JOURNAL_OWNER_FORMAT,
        "journal_owner_identity_sha256": identity_sha256(owner),
        "judge_output_root": owner["judge_output_root"],
        "judge_output_root_sha256": owner["judge_output_root_sha256"],
        "max_concurrency": preflight.payload["max_concurrency"],
        "model": preflight.payload["model"],
        "preflight_artifact_sha256": preflight.sha256,
        "prompt_population_sha256": preflight.payload[
            "prompt_population_sha256"
        ],
        "provider_calls_during_release": 0,
        "question_count": QUESTION_COUNT,
        "release_status": "approved_for_provider_execution",
        "required_authorized_provider_calls": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "retry_count": 0,
        "unsafe_retry_policy": "refuse_incomplete_request_response_pair",
    }
    return {**body, "release_identity_sha256": identity_sha256(body)}


def _validate_release(
    artifact: SealedArtifact,
    *,
    preflight: SealedArtifact,
    output_root: str | Path,
) -> None:
    payload = artifact.payload
    body = _without_receipt(payload, "release_identity_sha256")
    owner = _journal_owner_body(preflight, output_root=output_root)
    _require(
        set(payload) == RELEASE_KEYS
        and payload.get("release_identity_sha256") == identity_sha256(body)
        and payload.get("format") == RELEASE_FORMAT
        and payload.get("answer_arm") == preflight.payload.get("answer_arm")
        and payload.get("answer_binding_sha256")
        == preflight.payload.get("answer_binding_sha256")
        and payload.get("approval_opt_in") is True
        and payload.get("release_status") == "approved_for_provider_execution"
        and payload.get("gold_loaded") is True
        and payload.get("provider_calls_during_release") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("retry_count") == 0
        and payload.get("unsafe_retry_policy")
        == "refuse_incomplete_request_response_pair"
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("required_authorized_provider_calls") == QUESTION_COUNT
        and payload.get("preflight_artifact_sha256") == preflight.sha256
        and payload.get("model") == preflight.payload.get("model")
        and payload.get("gateway_url") == preflight.payload.get("gateway_url")
        and payload.get("max_concurrency")
        == preflight.payload.get("max_concurrency")
        and payload.get("gold_population_sha256")
        == preflight.payload.get("gold_population_sha256")
        and payload.get("prompt_population_sha256")
        == preflight.payload.get("prompt_population_sha256")
        and payload.get("journal_owner_format") == JOURNAL_OWNER_FORMAT
        and payload.get("journal_owner_identity_sha256")
        == identity_sha256(owner)
        and all(
            payload.get(key) == value
            for key, value in owner.items()
            if key != "format"
        )
        and all(
            payload.get(key) == preflight.payload.get(key)
            for key in ANSWER_BINDING_KEYS
        ),
        "A1 terminal judge provider release changed",
    )


def _read_release(
    output_root: str | Path,
    expected_sha256: str,
    *,
    preflight: SealedArtifact,
) -> SealedArtifact:
    artifact = _read_expected(
        Path(output_root) / RELEASE_NAME,
        expected_sha256,
        "A1 terminal judge release",
    )
    _validate_release(artifact, preflight=preflight, output_root=output_root)
    return artifact


def run_approve_release(args: argparse.Namespace) -> dict[str, Any]:
    arm = _arm(args.answer_arm)
    output_root = _output_root(args)
    _require(
        args.approve_provider_release is True,
        "judge release requires explicit provider approval",
    )
    _require(
        not (output_root / CHECKPOINT_DIR_NAME).exists(),
        "judge release requires an absent checkpoint root",
    )
    preflight, _prompts, _rows = _read_preflight(
        output_root,
        str(args.expected_judge_preflight_sha256),
        expected_arm=arm,
    )
    payload = _release_payload(preflight, output_root=output_root)
    artifact, created = publish_sealed_json(output_root / RELEASE_NAME, payload)
    _validate_release(artifact, preflight=preflight, output_root=output_root)
    return {
        "answer_arm": arm,
        "created": created,
        "journal_owner_identity_sha256": payload[
            "journal_owner_identity_sha256"
        ],
        "physical_provider_calls": 0,
        "preflight_sha256": preflight.sha256,
        "release_sha256": artifact.sha256,
        "required_authorized_provider_calls": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
    }


def _runtime(
    preflight: SealedArtifact,
    release: SealedArtifact,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    output_root: str | Path,
    model: str,
    gateway_url: str,
    max_concurrency: int,
    client: Any | None,
) -> FastCompletionRuntime:
    _require(
        model == preflight.payload.get("model") == DEFAULT_MODEL
        and gateway_url
        == preflight.payload.get("gateway_url")
        == DEFAULT_GATEWAY_URL
        and max_concurrency == preflight.payload.get("max_concurrency")
        and release.payload.get("preflight_artifact_sha256") == preflight.sha256
        and release.payload.get("release_status")
        == "approved_for_provider_execution"
        and len(prompts) == QUESTION_COUNT,
        "A1 terminal judge runtime differs from the sealed release",
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
            "answer_arm": preflight.payload["answer_arm"],
            "answer_binding_sha256": preflight.payload[
                "answer_binding_sha256"
            ],
            "answer_run_artifact_sha256": preflight.payload[
                "answer_run_artifact_sha256"
            ],
            "arm": FORMAT,
            "authorized_unique_calls": QUESTION_COUNT,
            "experiment_format": JUDGE_FORMAT,
            "gold_population_sha256": preflight.payload[
                "gold_population_sha256"
            ],
            "journal_owner_identity_sha256": release.payload[
                "journal_owner_identity_sha256"
            ],
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
    model: str,
    gateway_url: str,
    max_concurrency: int,
    client: Any | None,
) -> FastCompletionBatch:
    runtime = _runtime(
        preflight,
        release,
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
        _require(
            match is not None,
            "judge checkpoint root contains foreign journal state",
        )
        assert match is not None
        target = requests if match.group("kind") == "request" else responses
        target.add(match.group("key"))
    _require(
        requests == responses,
        "judge checkpoint pair is incomplete; unsafe retry forbidden",
    )
    _require(
        len(requests) <= QUESTION_COUNT,
        "judge checkpoint population exceeds eleven calls",
    )
    return len(requests)


def _validated_checkpoint_hits(
    preflight: SealedArtifact,
    release: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    output_root: str | Path,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> int:
    root = Path(output_root) / CHECKPOINT_DIR_NAME
    if not root.exists():
        return 0
    runtime = _runtime(
        preflight,
        release,
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
        "judge checkpoints escaped the sealed prompt population",
    )
    return len(records)


def _make_provider_client(api_key: str, gateway_url: str) -> Any:
    return judging._make_provider_client(api_key, gateway_url)  # noqa: SLF001


def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    arm = _arm(args.answer_arm)
    output_root = _output_root(args)
    preflight, prompts, _rows = _read_preflight(
        output_root,
        str(args.expected_judge_preflight_sha256),
        expected_arm=arm,
    )
    release = _read_release(
        output_root,
        str(args.expected_release_sha256),
        preflight=preflight,
    )
    _require(
        args.enable_provider is True
        and type(args.authorized_provider_calls) is int
        and 0 <= args.authorized_provider_calls <= QUESTION_COUNT,
        "judge provider requires bounded Sol authorization",
    )
    candidate_hits = _read_only_checkpoint_count(output_root)
    remaining = QUESTION_COUNT - candidate_hits
    _require(
        args.authorized_provider_calls == remaining,
        "judge authorization must exactly equal remaining calls",
    )
    checkpoint_hits = _validated_checkpoint_hits(
        preflight,
        release,
        prompts,
        output_root=output_root,
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
            release,
            prompts,
            output_root=output_root,
            model=str(args.model),
            gateway_url=str(args.gateway_url),
            max_concurrency=int(args.max_concurrency),
            client=None,
        )
        _require(
            batch.usage.logical_calls
            == batch.usage.unique_calls
            == batch.usage.checkpoint_hits
            == QUESTION_COUNT
            and batch.usage.physical_calls == 0,
            "judge complete checkpoint replay changed",
        )
    else:
        load_dotenv()
        api_key = os.environ.get(str(args.api_key_env), "").strip()
        _require(bool(api_key), f"provider API key is empty: {args.api_key_env}")
        client = _make_provider_client(api_key, str(args.gateway_url))
        try:
            batch = _checkpoint_batch(
                preflight,
                release,
                prompts,
                output_root=output_root,
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
            batch.usage.logical_calls
            == batch.usage.unique_calls
            == QUESTION_COUNT
            and batch.usage.physical_calls + batch.usage.checkpoint_hits
            == QUESTION_COUNT
            and batch.usage.physical_calls <= args.authorized_provider_calls
            and batch.usage.checkpoint_hits >= checkpoint_hits,
            "judge provider population changed",
        )
    return {
        "answer_arm": arm,
        "authorized_remaining_provider_calls": remaining,
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "physical_provider_calls": batch.usage.physical_calls,
        "preflight_sha256": preflight.sha256,
        "release_sha256": release.sha256,
        "required_authorized_provider_calls": remaining,
        "retained_transformer_token_state_bytes": 0,
    }


def _judge_payload(
    preflight: SealedArtifact,
    release: SealedArtifact,
    prompt_rows: Sequence[Mapping[str, Any]],
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
        "judge materialization requires eleven complete checkpoint hits",
    )
    records = {record.messages_sha256: record for record in batch.unique_records}
    _require(len(records) == QUESTION_COUNT, "judge completion identities repeat")
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
            "judge checkpoint record changed",
        )
        assert record is not None
        try:
            correct = parse_binary_judge_verdict(completion)
        except RuntimeError as exc:
            raise R7A1TerminalJudgeError(
                f"invalid Sol verdict for {prompt['question_id']}"
            ) from exc
        body = {
            "answer_source_row_sha256": prompt["answer_source_row_sha256"],
            "arm": preflight.payload["answer_arm"],
            "call_key_sha256": record.call_key_sha256,
            "category": prompt["category"],
            "correct": correct,
            "dated_question_sha256": prompt["dated_question_sha256"],
            "format": JUDGE_ROW_FORMAT,
            "judge_output": completion,
            "judge_output_sha256": quote_sha256(completion),
            "messages_sha256": prompt["messages_sha256"],
            "normalized_exact_match": exact_match(
                str(prompt["prediction"]), str(prompt["reference"])
            ),
            "normalized_f1": f1_score(
                str(prompt["prediction"]), str(prompt["reference"])
            ),
            "prediction_sha256": prompt["prediction_sha256"],
            "prompt_row_receipt_sha256": prompt[
                "prompt_row_receipt_sha256"
            ],
            "question_id": prompt["question_id"],
            "question_sha256": prompt["question_sha256"],
            "reference_sha256": prompt["reference_sha256"],
            "request_journal_sha256": record.request_journal_sha256,
            "response_journal_sha256": record.response_journal_sha256,
        }
        rows.append(
            {**body, "judge_row_receipt_sha256": identity_sha256(body)}
        )
    correct_count = sum(bool(row["correct"]) for row in rows)
    return {
        **{key: preflight.payload[key] for key in ANSWER_BINDING_KEYS},
        "aggregate": {
            "accuracy": correct_count / QUESTION_COUNT,
            "correct": correct_count,
            "question_count": QUESTION_COUNT,
        },
        "answer_arm": preflight.payload["answer_arm"],
        "answer_binding_sha256": preflight.payload["answer_binding_sha256"],
        "completion_batch": judging._stable_batch(batch),  # noqa: SLF001
        "format": JUDGE_FORMAT,
        "gold_loaded": True,
        "gold_population_sha256": preflight.payload[
            "gold_population_sha256"
        ],
        "journal_owner_identity_sha256": release.payload[
            "journal_owner_identity_sha256"
        ],
        "physical_provider_calls_during_materialization": 0,
        "preflight_artifact_sha256": preflight.sha256,
        "prompt_population_sha256": preflight.payload[
            "prompt_population_sha256"
        ],
        "question_count": QUESTION_COUNT,
        "question_population_sha256": preflight.payload[
            "question_population_sha256"
        ],
        "questions": rows,
        "release_authorization_artifact_sha256": release.sha256,
        "retained_transformer_token_state_bytes": 0,
        "source_population_sha256": preflight.payload[
            "source_population_sha256"
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
    _require(
        set(payload) == JUDGE_KEYS
        and payload.get("format") == JUDGE_FORMAT
        and payload.get("answer_arm") == preflight.payload.get("answer_arm")
        and payload.get("answer_binding_sha256")
        == preflight.payload.get("answer_binding_sha256")
        and payload.get("gold_loaded") is True
        and payload.get("physical_provider_calls_during_materialization") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("preflight_artifact_sha256") == preflight.sha256
        and payload.get("release_authorization_artifact_sha256")
        == release.sha256
        and payload.get("journal_owner_identity_sha256")
        == release.payload.get("journal_owner_identity_sha256")
        and payload.get("prompt_population_sha256")
        == preflight.payload.get("prompt_population_sha256")
        and payload.get("gold_population_sha256")
        == preflight.payload.get("gold_population_sha256")
        and payload.get("question_population_sha256")
        == preflight.payload.get("question_population_sha256")
        and payload.get("source_population_sha256")
        == preflight.payload.get("source_population_sha256")
        and all(
            payload.get(key) == preflight.payload.get(key)
            for key in ANSWER_BINDING_KEYS
        )
        and type(raw_rows) is list
        and len(raw_rows) == QUESTION_COUNT
        and type(aggregate) is dict
        and aggregate.get("question_count") == QUESTION_COUNT,
        "A1 terminal judge materialization envelope changed",
    )
    if expected_batch is not None:
        _require(
            payload.get("completion_batch")
            == judging._stable_batch(expected_batch)  # noqa: SLF001
            and expected_batch.usage.logical_calls
            == expected_batch.usage.unique_calls
            == expected_batch.usage.checkpoint_hits
            == QUESTION_COUNT
            and expected_batch.usage.physical_calls == 0,
            "A1 terminal judge completion batch changed",
        )
    prompt_by_question = {
        row["question_id"]: row for row in preflight.payload["prompt_rows"]
    }
    rows: list[dict[str, Any]] = []
    call_keys: list[str] = []
    question_ids: list[str] = []
    for raw in raw_rows:
        _require(type(raw) is dict, "judge verdict row changed type")
        body = _without_receipt(raw, "judge_row_receipt_sha256")
        question_id = require_text(raw.get("question_id"), "verdict question ID")
        prompt = prompt_by_question.get(question_id)
        output = require_text(raw.get("judge_output"), "Sol judge output")
        try:
            parsed = parse_binary_judge_verdict(output)
        except RuntimeError as exc:
            raise R7A1TerminalJudgeError(
                f"stored Sol verdict is malformed for {question_id}"
            ) from exc
        _require(
            set(raw) == JUDGE_ROW_KEYS
            and raw.get("judge_row_receipt_sha256") == identity_sha256(body)
            and raw.get("format") == JUDGE_ROW_FORMAT
            and raw.get("arm") == preflight.payload.get("answer_arm")
            and type(raw.get("correct")) is bool
            and parsed is raw.get("correct")
            and raw.get("judge_output_sha256") == quote_sha256(output)
            and prompt is not None
            and raw.get("messages_sha256") == prompt.get("messages_sha256")
            and raw.get("prediction_sha256")
            == prompt.get("prediction_sha256")
            and raw.get("reference_sha256") == prompt.get("reference_sha256")
            and raw.get("answer_source_row_sha256")
            == prompt.get("answer_source_row_sha256")
            and raw.get("prompt_row_receipt_sha256")
            == prompt.get("prompt_row_receipt_sha256"),
            f"judge verdict row changed for {question_id}",
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
        call_keys.append(str(raw["call_key_sha256"]))
        question_ids.append(question_id)
        rows.append(dict(raw))
    correct = sum(bool(row["correct"]) for row in rows)
    _require(
        len(set(call_keys)) == len(set(question_ids)) == QUESTION_COUNT
        and aggregate.get("correct") == correct
        and aggregate.get("accuracy") == correct / QUESTION_COUNT,
        "judge aggregate or row identities changed",
    )
    return tuple(rows)


def _complete_checkpoint_batch(
    preflight: SealedArtifact,
    release: SealedArtifact,
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
    return _checkpoint_batch(
        preflight,
        release,
        prompts,
        output_root=output_root,
        model=model,
        gateway_url=gateway_url,
        max_concurrency=max_concurrency,
        client=None,
    )


def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    arm = _arm(args.answer_arm)
    output_root = _output_root(args)
    preflight, prompts, rows = _read_preflight(
        output_root,
        str(args.expected_judge_preflight_sha256),
        expected_arm=arm,
    )
    release = _read_release(
        output_root,
        str(args.expected_release_sha256),
        preflight=preflight,
    )
    batch = _complete_checkpoint_batch(
        preflight,
        release,
        prompts,
        output_root=output_root,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    payload = _judge_payload(preflight, release, rows, batch)
    verdicts = _validate_judge(
        preflight, release, payload, expected_batch=batch
    )
    artifact, created = publish_sealed_json(output_root / JUDGE_NAME, payload)
    return {
        "accuracy": payload["aggregate"]["accuracy"],
        "answer_arm": arm,
        "checkpoint_hits": QUESTION_COUNT,
        "correct": sum(bool(row["correct"]) for row in verdicts),
        "created": created,
        "judge_sha256": artifact.sha256,
        "physical_provider_calls": 0,
        "question_count": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
    }


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    arm = _arm(args.answer_arm)
    output_root = _output_root(args)
    preflight, prompts, rows = _read_preflight(
        output_root,
        str(args.expected_judge_preflight_sha256),
        expected_arm=arm,
    )
    release = _read_release(
        output_root,
        str(args.expected_release_sha256),
        preflight=preflight,
    )
    batch = _complete_checkpoint_batch(
        preflight,
        release,
        prompts,
        output_root=output_root,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    rebuilt = _judge_payload(preflight, release, rows, batch)
    _validate_judge(preflight, release, rebuilt, expected_batch=batch)
    artifact = _read_expected(
        output_root / JUDGE_NAME,
        str(args.expected_judge_sha256),
        "A1 terminal Sol judge",
    )
    _require(
        artifact.payload == rebuilt,
        "judge differs from checkpoint-only replay",
    )
    replay, _created = publish_sealed_json(output_root / JUDGE_REPLAY_NAME, rebuilt)
    _require(replay.sha256 == artifact.sha256, "judge replay is not byte-identical")
    return {
        "answer_arm": arm,
        "byte_identical": True,
        "judge_replay_sha256": replay.sha256,
        "judge_sha256": artifact.sha256,
        "physical_provider_calls": 0,
        "retained_transformer_token_state_bytes": 0,
    }


def _load_verified_judge_replay(
    output_root: str | Path,
    *,
    arm: str,
    expected_preflight_sha256: str,
    expected_release_sha256: str,
    expected_judge_sha256: str,
    expected_judge_replay_sha256: str,
) -> tuple[
    SealedArtifact,
    SealedArtifact,
    SealedArtifact,
    SealedArtifact,
    tuple[dict[str, Any], ...],
]:
    preflight, _prompts, _prompt_rows = _read_preflight(
        output_root, expected_preflight_sha256, expected_arm=arm
    )
    release = _read_release(
        output_root, expected_release_sha256, preflight=preflight
    )
    judge = _read_expected(
        Path(output_root) / JUDGE_NAME,
        expected_judge_sha256,
        "A1 terminal Sol judge",
    )
    replay = _read_expected(
        Path(output_root) / JUDGE_REPLAY_NAME,
        expected_judge_replay_sha256,
        "A1 terminal Sol judge replay",
    )
    _require(
        judge.sha256 == replay.sha256 and judge.payload == replay.payload,
        "judge/replay is not byte-identical",
    )
    rows = _validate_judge(preflight, release, judge.payload)
    return preflight, release, judge, replay, rows


def _score_payload(
    preflight: SealedArtifact,
    release: SealedArtifact,
    judge: SealedArtifact,
    replay: SealedArtifact,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    score_rows: list[dict[str, Any]] = []
    for row in rows:
        body = {
            "correct": row["correct"],
            "format": SCORE_ROW_FORMAT,
            "judge_row_receipt_sha256": row[
                "judge_row_receipt_sha256"
            ],
            "question_id": row["question_id"],
            "question_sha256": row["question_sha256"],
        }
        score_rows.append(
            {**body, "score_row_receipt_sha256": identity_sha256(body)}
        )
    correct = sum(bool(row["correct"]) for row in score_rows)
    return {
        **{key: preflight.payload[key] for key in ANSWER_BINDING_KEYS},
        "accuracy": correct / QUESTION_COUNT,
        "answer_arm": preflight.payload["answer_arm"],
        "answer_binding_sha256": preflight.payload["answer_binding_sha256"],
        "correct": correct,
        "format": SCORE_FORMAT,
        "gold_loaded": True,
        "gold_population_sha256": preflight.payload[
            "gold_population_sha256"
        ],
        "judge_artifact_sha256": judge.sha256,
        "judge_replay_artifact_sha256": replay.sha256,
        "physical_provider_calls_during_scoring": 0,
        "preflight_artifact_sha256": preflight.sha256,
        "prompt_population_sha256": preflight.payload[
            "prompt_population_sha256"
        ],
        "question_count": QUESTION_COUNT,
        "question_population_sha256": preflight.payload[
            "question_population_sha256"
        ],
        "release_authorization_artifact_sha256": release.sha256,
        "retained_transformer_token_state_bytes": 0,
        "score_population_sha256": identity_sha256(
            [row["score_row_receipt_sha256"] for row in score_rows]
        ),
        "score_rows": score_rows,
        "source_population_sha256": preflight.payload[
            "source_population_sha256"
        ],
    }


def _validate_score(
    preflight: SealedArtifact,
    release: SealedArtifact,
    judge: SealedArtifact,
    replay: SealedArtifact,
    payload: Mapping[str, Any],
    judge_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    expected = _score_payload(
        preflight, release, judge, replay, judge_rows
    )
    raw_rows = payload.get("score_rows")
    _require(
        set(payload) == SCORE_KEYS
        and dict(payload) == expected
        and payload.get("format") == SCORE_FORMAT
        and payload.get("gold_loaded") is True
        and payload.get("physical_provider_calls_during_scoring") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("question_count") == QUESTION_COUNT
        and type(raw_rows) is list
        and len(raw_rows) == QUESTION_COUNT,
        "A1 terminal judge score changed",
    )
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        _require(type(raw) is dict, "score row changed type")
        body = _without_receipt(raw, "score_row_receipt_sha256")
        _require(
            set(raw) == SCORE_ROW_KEYS
            and raw.get("format") == SCORE_ROW_FORMAT
            and type(raw.get("correct")) is bool
            and raw.get("score_row_receipt_sha256") == identity_sha256(body),
            "score row changed",
        )
        for key in (
            "judge_row_receipt_sha256",
            "question_sha256",
            "score_row_receipt_sha256",
        ):
            require_sha256(raw.get(key), f"score row {key}")
        require_text(raw.get("question_id"), "score row question ID")
        rows.append(dict(raw))
    _require(
        len({row["question_id"] for row in rows}) == QUESTION_COUNT
        and payload.get("correct") == sum(bool(row["correct"]) for row in rows)
        and payload.get("accuracy")
        == sum(bool(row["correct"]) for row in rows) / QUESTION_COUNT
        and payload.get("score_population_sha256")
        == identity_sha256([row["score_row_receipt_sha256"] for row in rows]),
        "score arithmetic or population changed",
    )
    return tuple(rows)


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    arm = _arm(args.answer_arm)
    output_root = _output_root(args)
    preflight, release, judge, replay, rows = _load_verified_judge_replay(
        output_root,
        arm=arm,
        expected_preflight_sha256=str(args.expected_judge_preflight_sha256),
        expected_release_sha256=str(args.expected_release_sha256),
        expected_judge_sha256=str(args.expected_judge_sha256),
        expected_judge_replay_sha256=str(args.expected_judge_replay_sha256),
    )
    payload = _score_payload(preflight, release, judge, replay, rows)
    _validate_score(preflight, release, judge, replay, payload, rows)
    artifact, created = publish_sealed_json(output_root / SCORE_NAME, payload)
    return {
        "accuracy": payload["accuracy"],
        "answer_arm": arm,
        "correct": payload["correct"],
        "created": created,
        "physical_provider_calls": 0,
        "question_count": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "score_sha256": artifact.sha256,
    }


def run_score_replay(args: argparse.Namespace) -> dict[str, Any]:
    arm = _arm(args.answer_arm)
    output_root = _output_root(args)
    preflight, release, judge, replay, rows = _load_verified_judge_replay(
        output_root,
        arm=arm,
        expected_preflight_sha256=str(args.expected_judge_preflight_sha256),
        expected_release_sha256=str(args.expected_release_sha256),
        expected_judge_sha256=str(args.expected_judge_sha256),
        expected_judge_replay_sha256=str(args.expected_judge_replay_sha256),
    )
    rebuilt = _score_payload(preflight, release, judge, replay, rows)
    score = _read_expected(
        output_root / SCORE_NAME,
        str(args.expected_score_sha256),
        "A1 terminal Sol score",
    )
    _validate_score(preflight, release, judge, replay, score.payload, rows)
    _require(score.payload == rebuilt, "score differs from deterministic replay")
    score_replay, _created = publish_sealed_json(
        output_root / SCORE_REPLAY_NAME, rebuilt
    )
    _require(
        score_replay.sha256 == score.sha256,
        "score replay is not byte-identical",
    )
    return {
        "answer_arm": arm,
        "byte_identical": True,
        "physical_provider_calls": 0,
        "retained_transformer_token_state_bytes": 0,
        "score_replay_sha256": score_replay.sha256,
        "score_sha256": score.sha256,
    }


def load_verified_judge_score(
    output_root: str | Path,
    *,
    arm: str,
    expected_preflight_sha256: str,
    expected_release_sha256: str,
    expected_judge_sha256: str,
    expected_judge_replay_sha256: str,
    expected_score_sha256: str,
    expected_score_replay_sha256: str,
) -> tuple[SealedArtifact, SealedArtifact, tuple[dict[str, Any], ...]]:
    """Return one arm's score only after judge and score replay verification."""

    arm = _arm(arm)
    preflight, release, judge, replay, judge_rows = (
        _load_verified_judge_replay(
            output_root,
            arm=arm,
            expected_preflight_sha256=expected_preflight_sha256,
            expected_release_sha256=expected_release_sha256,
            expected_judge_sha256=expected_judge_sha256,
            expected_judge_replay_sha256=expected_judge_replay_sha256,
        )
    )
    score = _read_expected(
        Path(output_root) / SCORE_NAME,
        expected_score_sha256,
        "A1 terminal Sol score",
    )
    score_replay = _read_expected(
        Path(output_root) / SCORE_REPLAY_NAME,
        expected_score_replay_sha256,
        "A1 terminal Sol score replay",
    )
    _require(
        score.sha256 == score_replay.sha256
        and score.payload == score_replay.payload,
        "score/replay is not byte-identical",
    )
    score_rows = _validate_score(
        preflight, release, judge, replay, score.payload, judge_rows
    )
    return judge, score, score_rows


def _add_arm_and_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--answer-arm", required=True, choices=tuple(answer_cli.ARM_LABELS)
    )
    parser.add_argument(
        "--answer-root", type=Path, default=DEFAULT_ANSWER_ROOT
    )
    parser.add_argument(
        "--judge-output-root",
        type=Path,
        help="Defaults to ANSWER_ROOT/sol-judge-v1-ANSWER_ARM.",
    )


def _add_runtime(parser: argparse.ArgumentParser) -> None:
    _add_arm_and_root(parser)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument(
        "--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY
    )


def _add_sealed_judge(parser: argparse.ArgumentParser) -> None:
    _add_runtime(parser)
    parser.add_argument("--expected-judge-preflight-sha256", required=True)
    parser.add_argument("--expected-release-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight")
    _add_runtime(preflight)
    preflight.add_argument(
        "--expected-answer-preflight-construction-sha256", required=True
    )
    preflight.add_argument(
        "--expected-answer-preflight-replay-sha256", required=True
    )
    preflight.add_argument("--expected-answer-release-sha256", required=True)
    preflight.add_argument("--expected-answer-run-sha256", required=True)
    preflight.add_argument("--expected-answer-replay-sha256", required=True)
    preflight.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    preflight.add_argument("--split", type=Path, default=DEFAULT_SPLIT)

    release = commands.add_parser("approve-release")
    _add_runtime(release)
    release.add_argument("--expected-judge-preflight-sha256", required=True)
    release.add_argument("--approve-provider-release", action="store_true")

    provider = commands.add_parser("provider-run")
    _add_sealed_judge(provider)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, required=True)
    provider.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)

    materialize = commands.add_parser("materialize")
    _add_sealed_judge(materialize)

    replay = commands.add_parser("replay")
    _add_sealed_judge(replay)
    replay.add_argument("--expected-judge-sha256", required=True)

    score = commands.add_parser("score")
    _add_arm_and_root(score)
    score.add_argument("--expected-judge-preflight-sha256", required=True)
    score.add_argument("--expected-release-sha256", required=True)
    score.add_argument("--expected-judge-sha256", required=True)
    score.add_argument("--expected-judge-replay-sha256", required=True)

    score_replay = commands.add_parser("score-replay")
    _add_arm_and_root(score_replay)
    score_replay.add_argument("--expected-judge-preflight-sha256", required=True)
    score_replay.add_argument("--expected-release-sha256", required=True)
    score_replay.add_argument("--expected-judge-sha256", required=True)
    score_replay.add_argument("--expected-judge-replay-sha256", required=True)
    score_replay.add_argument("--expected-score-sha256", required=True)
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
    elif args.command == "replay":
        result = run_replay(args)
    elif args.command == "score":
        result = run_score(args)
    else:
        result = run_score_replay(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_DIR_NAME",
    "DEFAULT_ANSWER_ROOT",
    "DEFAULT_GATEWAY_URL",
    "DEFAULT_MODEL",
    "FORMAT",
    "JUDGE_NAME",
    "JUDGE_REPLAY_NAME",
    "PREFLIGHT_NAME",
    "QUESTION_COUNT",
    "R7A1TerminalJudgeError",
    "RELEASE_NAME",
    "SCORE_NAME",
    "SCORE_REPLAY_NAME",
    "build_parser",
    "build_preflight_payload",
    "load_verified_judge_score",
    "main",
    "run_approve_release",
    "run_materialize",
    "run_preflight",
    "run_provider",
    "run_replay",
    "run_score",
    "run_score_replay",
]
