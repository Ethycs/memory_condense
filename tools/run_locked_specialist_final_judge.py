#!/usr/bin/env python3
"""Judge the replay-verified locked specialist-final answer population.

The answer run and its byte-identical replay are verified before locked
LongMemEval gold is opened.  Preflight then seals exactly 100 unique standard
Sol judge prompts.  Provider execution is journal-only; materialization and
replay require complete checkpoint hits and retain no transformer token state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from dotenv import load_dotenv  # noqa: E402

from memory_condense.domain.discourse import quote_sha256  # noqa: E402
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
from tools._routed_repair_routing import route_question  # noqa: E402
from tools.matched_eval import judging, live  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.population import EXPECTED_QUESTION_COUNT  # noqa: E402
from tools.matched_eval.typed_memory_final_arm import (  # noqa: E402
    judge_row_projection,
)
from tools.matched_eval.typed_memory_final_judging import (  # noqa: E402
    TypedFinalJudgeGoldRow,
    load_locked_typed_final_gold,
)
from tools.run_locked_query_answer_judge import DEFAULT_DATASET  # noqa: E402
from tools.run_matched_eval_spine import DEFAULT_SPLIT  # noqa: E402


QUESTION_COUNT = EXPECTED_QUESTION_COUNT
EXACT_ORDINALS = tuple(range(QUESTION_COUNT))

ANSWER_RUN_FORMAT = "memory-condense-locked-specialist-final-terra-answer-v1"
PREFLIGHT_FORMAT = (
    "memory-condense-locked-specialist-final-sol-judge-preflight-v1"
)
JUDGE_FORMAT = "memory-condense-locked-specialist-final-sol-judge-v1"
SCORE_FORMAT = "memory-condense-locked-specialist-final-sol-score-v1"

ANSWER_RUN_NAME = "locked-specialist-final-answer-v1.json"
ANSWER_REPLAY_NAME = "locked-specialist-final-answer-replay-v1.json"
PREFLIGHT_NAME = "locked-specialist-final-sol-judge-preflight-v1.json"
JUDGE_NAME = "locked-specialist-final-semantic-judge-sol-v1.json"
JUDGE_REPLAY_NAME = (
    "locked-specialist-final-semantic-judge-sol-replay-v1.json"
)
SCORE_NAME = "locked-specialist-final-score-v1.json"
SCORE_REPLAY_NAME = "locked-specialist-final-score-replay-v1.json"
CHECKPOINT_DIR_NAME = "sol-locked-specialist-final-judge-calls-v1"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANSWER_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-specialist-final-answer-v1"
)
DEFAULT_JUDGE_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-specialist-final-sol-judge-v1"
)
DEFAULT_SOL_MODEL = "codex_sdk/gpt-5.6-sol"
DEFAULT_CALLER_MODEL = "openai/codex_sdk/gpt-5.6-sol"
DEFAULT_MAX_PROMPT_TOKENS = judging.DEFAULT_MAX_JUDGE_PROMPT_TOKENS
TARGET_ACCURACY = 0.95

# Version adapters temporarily install their immutable artifact contract into
# this shared implementation.  One re-entrant lock must guard both adapter
# overrides and direct calls into the base module, otherwise two versions can
# observe a mixed set of formats, paths, and runtime provenance.
_VERSION_CONTRACT_LOCK = RLock()


def _version_contract_guard(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        with _VERSION_CONTRACT_LOCK:
            return function(*args, **kwargs)

    return guarded


class LockedSpecialistFinalJudgeError(MatchedEvalContractError):
    """Raised when the locked specialist-final judge contract changes."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedSpecialistFinalJudgeError(message)


@_version_contract_guard
def validate_answer_run_artifact(
    artifact: SealedArtifact,
) -> tuple[dict[str, Any], ...]:
    """Validate the complete gold-free, typed judge seam."""

    payload = artifact.payload
    questions = payload.get("questions")
    judge_rows = payload.get("judge_rows")
    _require(
        payload.get("format") == ANSWER_RUN_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("physical_provider_calls_during_materialization") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("question_count") == QUESTION_COUNT
        and type(questions) is list
        and type(judge_rows) is list
        and len(questions) == len(judge_rows) == QUESTION_COUNT,
        "specialist-final answer judge-source envelope changed",
    )
    assert_gold_blind(payload, path="locked_specialist_final_answer")
    validated: list[dict[str, Any]] = []
    question_ids: list[str] = []
    for ordinal, (source, projected) in enumerate(
        zip(questions, judge_rows, strict=True)
    ):
        _require(
            type(source) is dict and type(projected) is dict,
            f"specialist-final answer row changed type at ordinal {ordinal}",
        )
        unsigned = dict(source)
        declared = unsigned.pop("source_row_sha256", None)
        prediction = source.get("prediction")
        _require(
            source.get("ordinal") == ordinal
            and declared == identity_sha256(unsigned)
            and type(prediction) is str
            and bool(prediction.strip())
            and prediction == prediction.strip()
            and source.get("prediction_sha256") == quote_sha256(prediction)
            and judge_row_projection(source) == projected,
            f"specialist-final answer row changed at ordinal {ordinal}",
        )
        question_ids.append(
            require_text(source.get("question_id"), "specialist-final question ID")
        )
        validated.append(dict(projected))
    _require(
        len(set(question_ids)) == QUESTION_COUNT,
        "specialist-final answer question identities repeat",
    )
    return tuple(validated)


@_version_contract_guard
def load_verified_answer_judge_source(
    *,
    answer_run_path: str | Path,
    answer_replay_path: str | Path,
    expected_answer_run_sha256: str,
    expected_answer_replay_sha256: str,
) -> tuple[SealedArtifact, SealedArtifact, tuple[dict[str, Any], ...]]:
    """Verify caller digests and byte identity before exposing the judge seam."""

    run = read_sealed_json(answer_run_path)
    replay = read_sealed_json(answer_replay_path)
    _require(
        run.sha256
        == require_sha256(expected_answer_run_sha256, "specialist answer run")
        and replay.sha256
        == require_sha256(
            expected_answer_replay_sha256, "specialist answer replay"
        )
        and run.sha256 == replay.sha256
        and canonical_json_bytes(run.payload) == canonical_json_bytes(replay.payload),
        "specialist-final answer run/replay are not byte-identical",
    )
    return run, replay, validate_answer_run_artifact(run)


def _gold_source_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "dated_question_sha256": row["dated_question_sha256"],
            "ordinal": row["ordinal"],
            "question_id": row["question_id"],
            "question_sha256": row["question_sha256"],
        }
        for row in rows
    )


@_version_contract_guard
def build_preflight_payload(
    *,
    run: SealedArtifact,
    replay: SealedArtifact,
    source_rows: Sequence[Mapping[str, Any]],
    gold_rows: Sequence[TypedFinalJudgeGoldRow],
    gold_population_sha256: str,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> tuple[dict[str, Any], tuple[tuple[dict[str, str], ...], ...]]:
    """Seal exactly one unique standard Sol prompt per locked question."""

    _require(
        run.sha256 == replay.sha256
        and canonical_json_bytes(run.payload) == canonical_json_bytes(replay.payload)
        and len(source_rows) == len(gold_rows) == QUESTION_COUNT
        and tuple(row.get("ordinal") for row in source_rows) == EXACT_ORDINALS
        and tuple(row.ordinal for row in gold_rows) == EXACT_ORDINALS
        and model == DEFAULT_SOL_MODEL
        and type(max_concurrency) is int
        and max_concurrency > 0,
        "specialist-final judge preflight population/settings changed",
    )
    require_text(model, "specialist-final judge model")
    require_text(gateway_url, "specialist-final judge gateway")
    pending: list[dict[str, Any]] = []
    prompts: list[tuple[dict[str, str], ...]] = []
    for source, gold in zip(source_rows, gold_rows, strict=True):
        _require(
            source.get("ordinal") == gold.ordinal
            and source.get("question_id") == gold.question_id
            and source.get("question_sha256") == gold.question_sha256
            and source.get("dated_question_sha256")
            == gold.dated_question_sha256,
            f"specialist-final judge gold binding changed at ordinal {gold.ordinal}",
        )
        messages = tuple(
            dict(message)
            for message in build_judge_prompt(
                gold.question,
                gold.reference,
                require_text(source.get("prediction"), "judge prediction"),
            )
        )
        pending.append(
            {
                "category": gold.category,
                "changed_from_parent": source["changed_from_parent"],
                "dated_question_sha256": gold.dated_question_sha256,
                "demand_class": route_question(gold.dated_question).style.value,
                "messages": list(messages),
                "messages_sha256": identity_sha256(list(messages)),
                "ordinal": gold.ordinal,
                "parent_prediction_sha256": source["parent_prediction_sha256"],
                "prediction": source["prediction"],
                "prediction_sha256": source["prediction_sha256"],
                "prediction_source": source["prediction_source"],
                "question_id": gold.question_id,
                "question_sha256": gold.question_sha256,
                "reference": gold.reference,
                "reference_sha256": gold.reference_sha256,
                "route_id": source["route_id"],
                "source_row_sha256": source["source_row_sha256"],
            }
        )
        prompts.append(messages)
    population = preflight_fast_completion_prompts(
        prompts,
        max_prompt_tokens=DEFAULT_MAX_PROMPT_TOKENS,
    )
    _require(
        population.logical_prompt_count
        == population.unique_prompt_count
        == QUESTION_COUNT,
        "specialist-final judge prompts must be 100 unique calls",
    )
    prompt_rows: list[dict[str, Any]] = []
    for raw, receipt in zip(pending, population.ordered_rows, strict=True):
        _require(
            raw["messages_sha256"] == receipt.messages_sha256,
            "specialist-final judge prompt order changed",
        )
        body = {**raw, "prompt_token_proxy": receipt.prompt_token_proxy}
        prompt_rows.append(
            {**body, "prompt_row_receipt_sha256": identity_sha256(body)}
        )
    payload = {
        "answer_replay_sha256": replay.sha256,
        "answer_run_format": ANSWER_RUN_FORMAT,
        "answer_run_sha256": run.sha256,
        "caller_model": DEFAULT_CALLER_MODEL,
        "format": PREFLIGHT_FORMAT,
        "gateway_url": gateway_url,
        "gold_loaded": True,
        "gold_population_sha256": require_sha256(
            gold_population_sha256, "specialist-final gold population"
        ),
        "max_concurrency": max_concurrency,
        "model": model,
        "physical_provider_calls": 0,
        "prompt_population": population.model_dump(),
        "prompt_population_sha256": population.prompt_population_sha256,
        "prompt_rows": prompt_rows,
        "question_count": QUESTION_COUNT,
        "required_authorized_provider_calls": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "selected_ordinals": list(EXACT_ORDINALS),
        "selected_question_count": QUESTION_COUNT,
    }
    return payload, tuple(prompts)


@_version_contract_guard
def validate_preflight_artifact(
    artifact: SealedArtifact,
) -> tuple[tuple[tuple[dict[str, str], ...], ...], tuple[dict[str, Any], ...]]:
    payload = artifact.payload
    raw_rows = payload.get("prompt_rows")
    _require(
        payload.get("format") == PREFLIGHT_FORMAT
        and payload.get("answer_run_format") == ANSWER_RUN_FORMAT
        and payload.get("model") == DEFAULT_SOL_MODEL
        and payload.get("gold_loaded") is True
        and payload.get("physical_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("selected_question_count") == QUESTION_COUNT
        and payload.get("required_authorized_provider_calls") == QUESTION_COUNT
        and payload.get("selected_ordinals") == list(EXACT_ORDINALS)
        and type(raw_rows) is list
        and len(raw_rows) == QUESTION_COUNT,
        "sealed specialist-final judge preflight changed",
    )
    prompts: list[tuple[dict[str, str], ...]] = []
    rows: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(raw_rows):
        _require(
            type(raw) is dict,
            f"sealed specialist-final judge row changed type at ordinal {ordinal}",
        )
        body = dict(raw)
        declared = body.pop("prompt_row_receipt_sha256", None)
        messages = raw.get("messages")
        _require(
            raw.get("ordinal") == ordinal
            and declared == identity_sha256(body)
            and type(messages) is list
            and identity_sha256(messages) == raw.get("messages_sha256"),
            f"sealed specialist-final judge row changed at ordinal {ordinal}",
        )
        prompts.append(tuple(dict(message) for message in messages))
        rows.append(dict(raw))
    population = preflight_fast_completion_prompts(
        prompts,
        max_prompt_tokens=DEFAULT_MAX_PROMPT_TOKENS,
    )
    _require(
        population.model_dump() == payload.get("prompt_population")
        and population.prompt_population_sha256
        == payload.get("prompt_population_sha256")
        and population.logical_prompt_count
        == population.unique_prompt_count
        == QUESTION_COUNT,
        "sealed specialist-final judge prompt population changed",
    )
    return tuple(prompts), tuple(rows)


@_version_contract_guard
def build_runtime(
    artifact: SealedArtifact,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    output_root: str | Path,
    model: str,
    gateway_url: str,
    max_concurrency: int,
    client: Any | None,
) -> FastCompletionRuntime:
    _require(
        artifact.payload.get("model") == model == DEFAULT_SOL_MODEL
        and artifact.payload.get("gateway_url") == gateway_url
        and artifact.payload.get("max_concurrency") == max_concurrency
        and len(prompts) == QUESTION_COUNT,
        "specialist-final judge runtime differs from sealed preflight",
    )
    return FastCompletionRuntime(
        checkpoint_dir=Path(output_root) / CHECKPOINT_DIR_NAME,
        prompt_population=prompts,
        model=model,
        client=client,
        max_prompt_tokens=DEFAULT_MAX_PROMPT_TOKENS,
        max_new_tokens=JUDGE_MAX_TOKENS,
        max_concurrency=max_concurrency,
        retries=0,
        benchmark_provenance={
            "answer_run_sha256": artifact.payload["answer_run_sha256"],
            "arm": "locked_specialist_final_sol_judge_v1",
            "authorized_unique_calls": QUESTION_COUNT,
            "experiment_format": JUDGE_FORMAT,
            "preflight_artifact_sha256": artifact.sha256,
        },
    )


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


@_version_contract_guard
def materialization_payloads(
    preflight: SealedArtifact,
    prompt_rows: Sequence[Mapping[str, Any]],
    batch: FastCompletionBatch,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(
        len(prompt_rows) == QUESTION_COUNT
        and batch.usage.logical_calls
        == batch.usage.unique_calls
        == batch.usage.checkpoint_hits
        == QUESTION_COUNT
        and batch.usage.physical_calls == 0
        and len(batch.logical_completions) == QUESTION_COUNT
        and len(batch.unique_records) == QUESTION_COUNT,
        "specialist-final judge materialization requires 100 checkpoint hits",
    )
    records = {record.messages_sha256: record for record in batch.unique_records}
    _require(
        len(records) == QUESTION_COUNT,
        "specialist-final judge completions repeat",
    )
    rows: list[dict[str, Any]] = []
    for prompt, completion in zip(
        prompt_rows, batch.logical_completions, strict=True
    ):
        record = records.get(str(prompt["messages_sha256"]))
        _require(
            record is not None
            and record.completion == completion
            and record.checkpoint_hit is True
            and record.physical_call is False,
            "specialist-final judge checkpoint record changed",
        )
        correct = parse_binary_judge_verdict(completion)
        _require(
            type(correct) is bool,
            "specialist-final judge returned an invalid verdict",
        )
        assert record is not None
        body = {
            "call_key_sha256": record.call_key_sha256,
            "category": prompt["category"],
            "changed_from_parent": prompt["changed_from_parent"],
            "correct": correct,
            "dated_question_sha256": prompt["dated_question_sha256"],
            "demand_class": prompt["demand_class"],
            "judge_output": completion,
            "judge_output_sha256": quote_sha256(completion),
            "messages_sha256": prompt["messages_sha256"],
            "normalized_exact_match": exact_match(
                str(prompt["prediction"]), str(prompt["reference"])
            ),
            "normalized_f1": f1_score(
                str(prompt["prediction"]), str(prompt["reference"])
            ),
            "ordinal": prompt["ordinal"],
            "parent_prediction_sha256": prompt["parent_prediction_sha256"],
            "prediction_sha256": prompt["prediction_sha256"],
            "prediction_source": prompt["prediction_source"],
            "prompt_row_receipt_sha256": prompt[
                "prompt_row_receipt_sha256"
            ],
            "question_id": prompt["question_id"],
            "question_sha256": prompt["question_sha256"],
            "reference_sha256": prompt["reference_sha256"],
            "request_journal_sha256": record.request_journal_sha256,
            "response_journal_sha256": record.response_journal_sha256,
            "route_id": prompt["route_id"],
            "source_row_sha256": prompt["source_row_sha256"],
        }
        rows.append({**body, "judge_row_sha256": identity_sha256(body)})
    correct_count = sum(bool(row["correct"]) for row in rows)
    exact_count = sum(bool(row["normalized_exact_match"]) for row in rows)
    mean_f1 = sum(float(row["normalized_f1"]) for row in rows) / QUESTION_COUNT
    aggregate = {
        "accuracy": correct_count / QUESTION_COUNT,
        "correct": correct_count,
        "gate_passed": correct_count / QUESTION_COUNT >= TARGET_ACCURACY,
        "incorrect": QUESTION_COUNT - correct_count,
        "mean_f1": mean_f1,
        "normalized_exact_match": exact_count,
        "questions": QUESTION_COUNT,
        "target_accuracy": TARGET_ACCURACY,
    }
    judge = {
        "aggregate": aggregate,
        "answer_replay_sha256": preflight.payload["answer_replay_sha256"],
        "answer_run_sha256": preflight.payload["answer_run_sha256"],
        "completion_batch": _stable_batch(batch),
        "format": JUDGE_FORMAT,
        "gold_loaded": True,
        "gold_population_sha256": preflight.payload["gold_population_sha256"],
        "judge_completions_may_echo_gold": True,
        "judge_model": DEFAULT_CALLER_MODEL,
        "physical_provider_calls_during_materialization": 0,
        "preflight_artifact_sha256": preflight.sha256,
        "question_count": QUESTION_COUNT,
        "questions": rows,
        "retained_transformer_token_state_bytes": 0,
        "selected_ordinals": list(EXACT_ORDINALS),
    }
    score = {
        **aggregate,
        "answer_replay_sha256": preflight.payload["answer_replay_sha256"],
        "answer_run_sha256": preflight.payload["answer_run_sha256"],
        "format": SCORE_FORMAT,
        "judge_preflight_sha256": preflight.sha256,
        "question_count": QUESTION_COUNT,
        "questions": [
            {
                "changed_from_parent": row["changed_from_parent"],
                "correct": row["correct"],
                "judge_row_sha256": row["judge_row_sha256"],
                "normalized_exact_match": row["normalized_exact_match"],
                "normalized_f1": row["normalized_f1"],
                "ordinal": row["ordinal"],
                "prediction_sha256": row["prediction_sha256"],
                "prediction_source": row["prediction_source"],
                "question_id": row["question_id"],
                "reference_sha256": row["reference_sha256"],
                "route_id": row["route_id"],
            }
            for row in rows
        ],
        "selected_ordinals": list(EXACT_ORDINALS),
    }
    return judge, score


def _answer_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    root = Path(args.answer_root)
    return (
        Path(args.answer_run or root / ANSWER_RUN_NAME),
        Path(args.answer_replay or root / ANSWER_REPLAY_NAME),
    )


@_version_contract_guard
def run_preflight(
    args: argparse.Namespace,
    *,
    source_loader: Callable[..., tuple[
        SealedArtifact, SealedArtifact, tuple[dict[str, Any], ...]
    ]] = load_verified_answer_judge_source,
    gold_loader: Callable[..., tuple[
        tuple[TypedFinalJudgeGoldRow, ...], str
    ]] = load_locked_typed_final_gold,
) -> dict[str, Any]:
    run_path, replay_path = _answer_paths(args)
    # Source verification deliberately completes before either locked file opens.
    run, replay, source_rows = source_loader(
        answer_run_path=run_path,
        answer_replay_path=replay_path,
        expected_answer_run_sha256=args.expected_answer_run_sha256,
        expected_answer_replay_sha256=args.expected_answer_replay_sha256,
    )
    gold_rows, gold_sha = gold_loader(
        dataset_path=args.dataset,
        split_path=args.split,
        source_rows=_gold_source_rows(source_rows),
    )
    payload, _prompts = build_preflight_payload(
        run=run,
        replay=replay,
        source_rows=source_rows,
        gold_rows=gold_rows,
        gold_population_sha256=gold_sha,
        model=args.model,
        gateway_url=args.gateway_url,
        max_concurrency=args.max_concurrency,
    )
    artifact, created = publish_sealed_json(
        Path(args.judge_output_root) / PREFLIGHT_NAME,
        payload,
    )
    return {
        "answer_run_sha256": run.sha256,
        "created": created,
        "physical_provider_calls": 0,
        "preflight_sha256": artifact.sha256,
        "required_authorized_provider_calls": QUESTION_COUNT,
        "selected_ordinals": list(EXACT_ORDINALS),
    }


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
        == require_sha256(expected_sha256, "expected specialist judge preflight"),
        "specialist-final judge preflight SHA-256 changed",
    )
    prompts, rows = validate_preflight_artifact(artifact)
    return artifact, prompts, rows


def _run_batch(
    artifact: SealedArtifact,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    args: argparse.Namespace,
    client: Any | None,
) -> FastCompletionBatch:
    runtime = build_runtime(
        artifact,
        prompts,
        output_root=args.judge_output_root,
        model=args.model,
        gateway_url=args.gateway_url,
        max_concurrency=args.max_concurrency,
        client=client,
    )
    try:
        return runtime.run()
    finally:
        runtime.close()


@_version_contract_guard
def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    artifact, prompts, _rows = _read_preflight(
        Path(args.judge_output_root), args.expected_judge_preflight_sha256
    )
    _require(
        args.enable_provider is True
        and args.authorized_provider_calls == QUESTION_COUNT
        and artifact.payload.get("required_authorized_provider_calls")
        == QUESTION_COUNT,
        "specialist-final Sol judge requires exact authorization for 100 calls",
    )
    _require(
        artifact.payload.get("model") == args.model == DEFAULT_SOL_MODEL
        and artifact.payload.get("gateway_url") == args.gateway_url
        and artifact.payload.get("max_concurrency") == args.max_concurrency,
        "specialist-final Sol runtime differs from sealed preflight",
    )
    load_dotenv()
    api_key = os.environ.get(args.api_key_env, "").strip()
    _require(bool(api_key), f"provider API key is empty: {args.api_key_env}")
    client = judging._make_provider_client(api_key, args.gateway_url)  # noqa: SLF001
    try:
        batch = _run_batch(artifact, prompts, args=args, client=client)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    _require(
        batch.usage.logical_calls == batch.usage.unique_calls == QUESTION_COUNT
        and batch.usage.physical_calls + batch.usage.checkpoint_hits
        == QUESTION_COUNT,
        "specialist-final Sol journal population changed after authorization",
    )
    return {
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "physical_provider_calls": batch.usage.physical_calls,
        "preflight_sha256": artifact.sha256,
        "required_authorized_provider_calls": QUESTION_COUNT,
    }


@_version_contract_guard
def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    artifact, prompts, rows = _read_preflight(
        Path(args.judge_output_root), args.expected_judge_preflight_sha256
    )
    batch = _run_batch(artifact, prompts, args=args, client=None)
    judge_payload, score_payload = materialization_payloads(artifact, rows, batch)
    root = Path(args.judge_output_root)
    judge_artifact, judge_created = publish_sealed_json(
        root / JUDGE_NAME, judge_payload
    )
    score_artifact, score_created = publish_sealed_json(
        root / SCORE_NAME, score_payload
    )
    return {
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "correct": score_payload["correct"],
        "judge_created": judge_created,
        "judge_sha256": judge_artifact.sha256,
        "physical_provider_calls": 0,
        "score_created": score_created,
        "score_sha256": score_artifact.sha256,
    }


@_version_contract_guard
def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    artifact, prompts, rows = _read_preflight(
        Path(args.judge_output_root), args.expected_judge_preflight_sha256
    )
    batch = _run_batch(artifact, prompts, args=args, client=None)
    expected_judge, expected_score = materialization_payloads(
        artifact, rows, batch
    )
    root = Path(args.judge_output_root)
    judge = read_sealed_json(root / JUDGE_NAME)
    score = read_sealed_json(root / SCORE_NAME)
    _require(
        judge.sha256
        == require_sha256(args.expected_judge_sha256, "expected specialist judge")
        and score.sha256
        == require_sha256(args.expected_score_sha256, "expected specialist score")
        and canonical_json_bytes(judge.payload)
        == canonical_json_bytes(expected_judge)
        and canonical_json_bytes(score.payload)
        == canonical_json_bytes(expected_score),
        "specialist-final judge materialization differs from checkpoint replay",
    )
    judge_replay, _ = publish_sealed_json(root / JUDGE_REPLAY_NAME, expected_judge)
    score_replay, _ = publish_sealed_json(root / SCORE_REPLAY_NAME, expected_score)
    _require(
        judge_replay.sha256 == judge.sha256
        and score_replay.sha256 == score.sha256,
        "specialist-final judge replay is not byte-identical",
    )
    return {
        "byte_identical": True,
        "judge_replay_sha256": judge_replay.sha256,
        "physical_provider_calls": 0,
        "score_replay_sha256": score_replay.sha256,
    }


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--judge-output-root", type=Path, default=DEFAULT_JUDGE_ROOT)
    parser.add_argument("--model", default=DEFAULT_SOL_MODEL)
    parser.add_argument("--gateway-url", default=live.DEFAULT_GATEWAY_URL)
    parser.add_argument("--max-concurrency", type=int, default=4)


@_version_contract_guard
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    _add_runtime_args(preflight)
    preflight.add_argument("--answer-root", type=Path, default=DEFAULT_ANSWER_ROOT)
    preflight.add_argument("--answer-run", type=Path)
    preflight.add_argument("--answer-replay", type=Path)
    preflight.add_argument("--expected-answer-run-sha256", required=True)
    preflight.add_argument("--expected-answer-replay-sha256", required=True)
    preflight.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    preflight.add_argument("--split", type=Path, default=DEFAULT_SPLIT)

    provider = subparsers.add_parser("provider-run")
    _add_runtime_args(provider)
    provider.add_argument("--expected-judge-preflight-sha256", required=True)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, required=True)
    provider.add_argument("--api-key-env", default=live.DEFAULT_API_KEY_ENV)

    materialize = subparsers.add_parser("materialize")
    _add_runtime_args(materialize)
    materialize.add_argument("--expected-judge-preflight-sha256", required=True)

    replay = subparsers.add_parser("replay")
    _add_runtime_args(replay)
    replay.add_argument("--expected-judge-preflight-sha256", required=True)
    replay.add_argument("--expected-judge-sha256", required=True)
    replay.add_argument("--expected-score-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        result = run_preflight(args)
    elif args.command == "provider-run":
        result = run_provider(args)
    elif args.command == "materialize":
        result = run_materialize(args)
    else:
        result = run_replay(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ANSWER_REPLAY_NAME",
    "ANSWER_RUN_FORMAT",
    "ANSWER_RUN_NAME",
    "CHECKPOINT_DIR_NAME",
    "DEFAULT_ANSWER_ROOT",
    "DEFAULT_JUDGE_ROOT",
    "DEFAULT_SOL_MODEL",
    "EXACT_ORDINALS",
    "JUDGE_FORMAT",
    "JUDGE_NAME",
    "JUDGE_REPLAY_NAME",
    "LockedSpecialistFinalJudgeError",
    "PREFLIGHT_FORMAT",
    "PREFLIGHT_NAME",
    "QUESTION_COUNT",
    "SCORE_FORMAT",
    "SCORE_NAME",
    "SCORE_REPLAY_NAME",
    "build_parser",
    "build_preflight_payload",
    "build_runtime",
    "load_verified_answer_judge_source",
    "main",
    "materialization_payloads",
    "run_materialize",
    "run_preflight",
    "run_provider",
    "run_replay",
    "validate_answer_run_artifact",
    "validate_preflight_artifact",
]
