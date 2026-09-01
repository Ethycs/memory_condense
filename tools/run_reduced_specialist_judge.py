#!/usr/bin/env python3
"""Judge replay-verified reduced-specialist Terra answers with ten Sol calls.

The answer run and replay are validated before locked gold is opened.  The
preflight seals ten standard benchmark judge prompts containing the dated
question, reference, and sealed prediction.  Provider execution writes only
completion journals; materialization and replay are checkpoint-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from tools import run_locked_query_answer_judge as locked_judge_cli  # noqa: E402
from tools import run_matched_eval_spine as spine_cli  # noqa: E402
from tools.matched_eval import judging, live  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.typed_memory_final_judging import (  # noqa: E402
    TypedFinalJudgeGoldRow,
    load_locked_typed_final_gold,
)


EXACT_ORDINALS = (7, 31, 36, 43, 61, 72, 77, 81, 86, 93)
QUESTION_COUNT = len(EXACT_ORDINALS)

ANSWER_RUN_FORMAT = "memory-condense-reduced-specialist-terra-answer-v2"
PREFLIGHT_FORMAT = "memory-condense-reduced-specialist-sol-judge-preflight-v2"
JUDGE_FORMAT = "memory-condense-reduced-specialist-sol-judge-v2"
SCORE_FORMAT = "memory-condense-reduced-specialist-sol-score-v2"

ANSWER_RUN_NAME = "reduced-specialist-answer-v2.json"
ANSWER_REPLAY_NAME = "reduced-specialist-answer-replay-v2.json"
PREFLIGHT_NAME = "reduced-specialist-sol-judge-preflight-v2.json"
JUDGE_NAME = "reduced-specialist-semantic-judge-sol-v2.json"
JUDGE_REPLAY_NAME = "reduced-specialist-semantic-judge-sol-replay-v2.json"
SCORE_NAME = "reduced-specialist-score-v2.json"
SCORE_REPLAY_NAME = "reduced-specialist-score-replay-v2.json"
CHECKPOINT_DIR_NAME = "sol-reduced-specialist-judge-calls-v2"

DEFAULT_ANSWER_ROOT = Path(
    "eval_results/matched_eval_100/reduced-specialist-answer-v2"
)
DEFAULT_JUDGE_ROOT = Path(
    "eval_results/matched_eval_100/reduced-specialist-sol-judge-v2"
)
DEFAULT_SOL_MODEL = "codex_sdk/gpt-5.6-sol"
DEFAULT_CALLER_MODEL = "openai/codex_sdk/gpt-5.6-sol"
DEFAULT_MAX_PROMPT_TOKENS = judging.DEFAULT_MAX_JUDGE_PROMPT_TOKENS

_ANSWER_ROW_KEYS = frozenset(
    {
        "answer_row_sha256",
        "dated_question_sha256",
        "ordinal",
        "prediction",
        "prediction_sha256",
        "question_id",
        "question_sha256",
    }
)
_FORBIDDEN_ANSWER_KEYS = frozenset(
    {"answer", "gold", "gold_answer", "reference", "reference_answer"}
)


class ReducedSpecialistJudgeError(MatchedEvalContractError):
    """Raised when the reduced-specialist judge boundary is not exact."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ReducedSpecialistJudgeError(message)


def _has_forbidden_answer_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in _FORBIDDEN_ANSWER_KEYS
            or _has_forbidden_answer_key(child)
            for key, child in value.items()
        )
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and any(
        _has_forbidden_answer_key(child) for child in value
    )


@dataclass(frozen=True, slots=True)
class AnswerSeamRow:
    ordinal: int
    question_id: str
    question_sha256: str
    dated_question_sha256: str
    prediction: str
    prediction_sha256: str
    answer_row_sha256: str

    def projection(self) -> dict[str, Any]:
        return {
            "answer_row_sha256": self.answer_row_sha256,
            "dated_question_sha256": self.dated_question_sha256,
            "ordinal": self.ordinal,
            "prediction": self.prediction,
            "prediction_sha256": self.prediction_sha256,
            "question_id": self.question_id,
            "question_sha256": self.question_sha256,
        }


def _answer_rows(payload: Mapping[str, Any]) -> tuple[AnswerSeamRow, ...]:
    """Read the sole supported specialist-answer seam.

    Envelope metadata may grow, but format, gold-free invariants, and the
    signed ``judge_rows`` schema are deliberately exact.  This avoids silently
    guessing at a concurrently evolving answer artifact.
    """

    raw_rows = payload.get("judge_rows")
    _require(
        payload.get("format") == ANSWER_RUN_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("physical_provider_calls_during_materialization") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and not _has_forbidden_answer_key(payload)
        and type(raw_rows) is list
        and len(raw_rows) == QUESTION_COUNT,
        "specialist answer judge-source envelope changed",
    )
    result: list[AnswerSeamRow] = []
    question_ids: list[str] = []
    for expected_ordinal, raw in zip(EXACT_ORDINALS, raw_rows, strict=True):
        _require(
            type(raw) is dict and frozenset(raw) == _ANSWER_ROW_KEYS,
            f"specialist answer judge row schema changed at ordinal {expected_ordinal}",
        )
        ordinal = raw.get("ordinal")
        question_id = raw.get("question_id")
        question_sha = raw.get("question_sha256")
        dated_sha = raw.get("dated_question_sha256")
        prediction = raw.get("prediction")
        prediction_sha = raw.get("prediction_sha256")
        unsigned = dict(raw)
        declared = unsigned.pop("answer_row_sha256", None)
        _require(
            ordinal == expected_ordinal
            and type(question_id) is str
            and bool(question_id)
            and type(question_sha) is str
            and require_sha256(question_sha, "question SHA-256") == question_sha
            and type(dated_sha) is str
            and require_sha256(dated_sha, "dated-question SHA-256") == dated_sha
            and type(prediction) is str
            and bool(prediction.strip())
            and prediction == prediction.strip()
            and prediction_sha == quote_sha256(prediction)
            and declared == identity_sha256(unsigned),
            f"specialist answer judge row changed at ordinal {expected_ordinal}",
        )
        assert isinstance(question_id, str)
        assert isinstance(question_sha, str)
        assert isinstance(dated_sha, str)
        assert isinstance(prediction, str)
        assert isinstance(prediction_sha, str)
        assert isinstance(declared, str)
        question_ids.append(question_id)
        result.append(
            AnswerSeamRow(
                ordinal=expected_ordinal,
                question_id=question_id,
                question_sha256=question_sha,
                dated_question_sha256=dated_sha,
                prediction=prediction,
                prediction_sha256=prediction_sha,
                answer_row_sha256=declared,
            )
        )
    _require(
        len(set(question_ids)) == QUESTION_COUNT,
        "specialist answer question IDs repeat",
    )
    return tuple(result)


def load_verified_answer_seam(
    *,
    answer_run_path: str | Path,
    answer_replay_path: str | Path,
    expected_answer_run_sha256: str,
    expected_answer_replay_sha256: str,
) -> tuple[SealedArtifact, SealedArtifact, tuple[AnswerSeamRow, ...]]:
    """Verify the gold-free answer run and byte-identical replay."""

    run = read_sealed_json(answer_run_path)
    replay = read_sealed_json(answer_replay_path)
    expected_run = require_sha256(expected_answer_run_sha256, "answer run SHA-256")
    expected_replay = require_sha256(
        expected_answer_replay_sha256, "answer replay SHA-256"
    )
    _require(
        run.sha256 == expected_run
        and replay.sha256 == expected_replay
        and run.sha256 == replay.sha256
        and canonical_json_bytes(run.payload) == canonical_json_bytes(replay.payload),
        "specialist answer run/replay are not byte-identical",
    )
    return run, replay, _answer_rows(run.payload)


def _gold_source_rows(rows: Sequence[AnswerSeamRow]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "dated_question_sha256": row.dated_question_sha256,
            "ordinal": row.ordinal,
            "question_id": row.question_id,
            "question_sha256": row.question_sha256,
        }
        for row in rows
    )


def build_preflight_payload(
    *,
    run: SealedArtifact,
    replay: SealedArtifact,
    answer_rows: Sequence[AnswerSeamRow],
    gold_rows: Sequence[TypedFinalJudgeGoldRow],
    gold_population_sha256: str,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> tuple[dict[str, Any], tuple[tuple[dict[str, str], ...], ...]]:
    _require(
        run.sha256 == replay.sha256
        and canonical_json_bytes(run.payload) == canonical_json_bytes(replay.payload)
        and tuple(row.ordinal for row in answer_rows) == EXACT_ORDINALS
        and tuple(row.ordinal for row in gold_rows) == EXACT_ORDINALS
        and len(answer_rows) == len(gold_rows) == QUESTION_COUNT
        and model == DEFAULT_SOL_MODEL
        and type(max_concurrency) is int
        and max_concurrency > 0,
        "specialist judge preflight population/settings changed",
    )
    require_text(model, "judge model")
    require_text(gateway_url, "judge gateway URL")
    pending: list[dict[str, Any]] = []
    prompts: list[tuple[dict[str, str], ...]] = []
    for source, gold in zip(answer_rows, gold_rows, strict=True):
        _require(
            source.ordinal == gold.ordinal
            and source.question_id == gold.question_id
            and source.question_sha256 == gold.question_sha256
            and source.dated_question_sha256 == gold.dated_question_sha256,
            f"specialist judge gold binding changed at ordinal {source.ordinal}",
        )
        # The dated question is intentional: this is the same benchmark judge
        # protocol, with temporal query context retained explicitly.
        messages = tuple(
            dict(message)
            for message in build_judge_prompt(
                gold.dated_question,
                gold.reference,
                source.prediction,
            )
        )
        messages_sha = identity_sha256(list(messages))
        pending.append(
            {
                "answer_row_sha256": source.answer_row_sha256,
                "category": gold.category,
                "dated_question_sha256": source.dated_question_sha256,
                "messages": list(messages),
                "messages_sha256": messages_sha,
                "ordinal": source.ordinal,
                "prediction": source.prediction,
                "prediction_sha256": source.prediction_sha256,
                "question_id": source.question_id,
                "question_sha256": source.question_sha256,
                "reference": gold.reference,
                "reference_sha256": gold.reference_sha256,
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
        "specialist judge prompts must be ten unique calls",
    )
    prompt_rows: list[dict[str, Any]] = []
    for raw, receipt in zip(pending, population.ordered_rows, strict=True):
        _require(
            raw["messages_sha256"] == receipt.messages_sha256,
            "specialist judge prompt order changed",
        )
        body = {**raw, "prompt_token_proxy": receipt.prompt_token_proxy}
        prompt_rows.append({**body, "prompt_row_sha256": identity_sha256(body)})
    payload = {
        "answer_replay_sha256": replay.sha256,
        "answer_run_sha256": run.sha256,
        "answer_run_format": ANSWER_RUN_FORMAT,
        "caller_model": DEFAULT_CALLER_MODEL,
        "format": PREFLIGHT_FORMAT,
        "gateway_url": gateway_url,
        "gold_loaded_posthoc": True,
        "gold_population_sha256": require_sha256(
            gold_population_sha256, "gold population SHA-256"
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
    }
    return payload, tuple(prompts)


def validate_preflight_artifact(
    artifact: SealedArtifact,
) -> tuple[tuple[tuple[dict[str, str], ...], ...], tuple[dict[str, Any], ...]]:
    payload = artifact.payload
    raw_rows = payload.get("prompt_rows")
    _require(
        payload.get("format") == PREFLIGHT_FORMAT
        and payload.get("answer_run_format") == ANSWER_RUN_FORMAT
        and payload.get("gold_loaded_posthoc") is True
        and payload.get("physical_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("required_authorized_provider_calls") == QUESTION_COUNT
        and payload.get("selected_ordinals") == list(EXACT_ORDINALS)
        and type(raw_rows) is list
        and len(raw_rows) == QUESTION_COUNT,
        "sealed specialist judge preflight changed",
    )
    prompts: list[tuple[dict[str, str], ...]] = []
    rows: list[dict[str, Any]] = []
    for expected_ordinal, raw in zip(EXACT_ORDINALS, raw_rows, strict=True):
        _require(type(raw) is dict, "sealed specialist judge row changed type")
        body = dict(raw)
        declared = body.pop("prompt_row_sha256", None)
        messages = raw.get("messages")
        _require(
            raw.get("ordinal") == expected_ordinal
            and declared == identity_sha256(body)
            and type(messages) is list
            and identity_sha256(messages) == raw.get("messages_sha256"),
            f"sealed specialist judge row changed at ordinal {expected_ordinal}",
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
        and population.unique_prompt_count == QUESTION_COUNT,
        "sealed specialist judge prompt population changed",
    )
    return tuple(prompts), tuple(rows)


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
        "specialist judge runtime differs from sealed preflight",
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
            "arm": "reduced_specialist_sol_judge_v2",
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


def materialization_payloads(
    preflight: SealedArtifact,
    prompt_rows: Sequence[Mapping[str, Any]],
    batch: FastCompletionBatch,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(
        len(prompt_rows) == QUESTION_COUNT
        and batch.usage.logical_calls == batch.usage.unique_calls == QUESTION_COUNT
        and batch.usage.checkpoint_hits == QUESTION_COUNT
        and batch.usage.physical_calls == 0
        and len(batch.logical_completions) == QUESTION_COUNT
        and len(batch.unique_records) == QUESTION_COUNT,
        "specialist judge materialization requires ten checkpoint hits",
    )
    records = {record.messages_sha256: record for record in batch.unique_records}
    _require(len(records) == QUESTION_COUNT, "specialist judge completions repeat")
    rows: list[dict[str, Any]] = []
    for prompt, completion in zip(
        prompt_rows,
        batch.logical_completions,
        strict=True,
    ):
        record = records.get(str(prompt["messages_sha256"]))
        _require(
            record is not None
            and record.completion == completion
            and record.checkpoint_hit is True
            and record.physical_call is False,
            "specialist judge checkpoint record changed",
        )
        correct = parse_binary_judge_verdict(completion)
        _require(type(correct) is bool, "specialist judge returned invalid verdict")
        assert record is not None
        body = {
            "answer_row_sha256": prompt["answer_row_sha256"],
            "call_key_sha256": record.call_key_sha256,
            "category": prompt["category"],
            "correct": correct,
            "dated_question_sha256": prompt["dated_question_sha256"],
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
            "prediction_sha256": prompt["prediction_sha256"],
            "prompt_row_sha256": prompt["prompt_row_sha256"],
            "question_id": prompt["question_id"],
            "question_sha256": prompt["question_sha256"],
            "reference_sha256": prompt["reference_sha256"],
            "request_journal_sha256": record.request_journal_sha256,
            "response_journal_sha256": record.response_journal_sha256,
        }
        rows.append({**body, "judge_row_sha256": identity_sha256(body)})
    correct_count = sum(bool(row["correct"]) for row in rows)
    exact_count = sum(bool(row["normalized_exact_match"]) for row in rows)
    mean_f1 = sum(float(row["normalized_f1"]) for row in rows) / QUESTION_COUNT
    aggregate = {
        "accuracy": correct_count / QUESTION_COUNT,
        "correct": correct_count,
        "incorrect": QUESTION_COUNT - correct_count,
        "mean_f1": mean_f1,
        "normalized_exact_match": exact_count,
        "questions": QUESTION_COUNT,
    }
    judge = {
        "aggregate": aggregate,
        "answer_replay_sha256": preflight.payload["answer_replay_sha256"],
        "answer_run_sha256": preflight.payload["answer_run_sha256"],
        "completion_batch": _stable_batch(batch),
        "format": JUDGE_FORMAT,
        "gold_loaded_posthoc": True,
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
                "correct": row["correct"],
                "judge_row_sha256": row["judge_row_sha256"],
                "normalized_exact_match": row["normalized_exact_match"],
                "normalized_f1": row["normalized_f1"],
                "ordinal": row["ordinal"],
                "prediction_sha256": row["prediction_sha256"],
                "question_id": row["question_id"],
                "reference_sha256": row["reference_sha256"],
            }
            for row in rows
        ],
        "selected_ordinals": list(EXACT_ORDINALS),
    }
    return judge, score


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
        == require_sha256(expected_sha256, "expected judge preflight SHA-256"),
        "specialist judge preflight SHA-256 changed",
    )
    prompts, rows = validate_preflight_artifact(artifact)
    return artifact, prompts, rows


def _answer_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    root = Path(args.answer_root)
    return (
        Path(args.answer_run or root / ANSWER_RUN_NAME),
        Path(args.answer_replay or root / ANSWER_REPLAY_NAME),
    )


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    run_path, replay_path = _answer_paths(args)
    # This call completes before the first locked dataset/split read.
    run, replay, answer_rows = load_verified_answer_seam(
        answer_run_path=run_path,
        answer_replay_path=replay_path,
        expected_answer_run_sha256=args.expected_answer_run_sha256,
        expected_answer_replay_sha256=args.expected_answer_replay_sha256,
    )
    gold_rows, gold_sha = load_locked_typed_final_gold(
        dataset_path=args.dataset,
        split_path=args.split,
        source_rows=_gold_source_rows(answer_rows),
        allow_subset=True,
    )
    payload, _prompts = build_preflight_payload(
        run=run,
        replay=replay,
        answer_rows=answer_rows,
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


def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    artifact, prompts, _rows = _read_preflight(
        Path(args.judge_output_root),
        args.expected_judge_preflight_sha256,
    )
    _require(
        args.enable_provider is True
        and args.authorized_provider_calls == QUESTION_COUNT
        and artifact.payload.get("required_authorized_provider_calls")
        == QUESTION_COUNT,
        "specialist Sol judge requires exact authorization for 10 calls",
    )
    _require(
        artifact.payload.get("model") == args.model == DEFAULT_SOL_MODEL
        and artifact.payload.get("gateway_url") == args.gateway_url
        and artifact.payload.get("max_concurrency") == args.max_concurrency,
        "specialist Sol judge runtime differs from sealed preflight",
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
        batch.usage.physical_calls + batch.usage.checkpoint_hits == QUESTION_COUNT,
        "specialist Sol journal population changed after authorization",
    )
    return {
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "physical_provider_calls": batch.usage.physical_calls,
        "preflight_sha256": artifact.sha256,
        "required_authorized_provider_calls": QUESTION_COUNT,
    }


def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    artifact, prompts, rows = _read_preflight(
        Path(args.judge_output_root),
        args.expected_judge_preflight_sha256,
    )
    batch = _run_batch(artifact, prompts, args=args, client=None)
    judge, score = materialization_payloads(artifact, rows, batch)
    root = Path(args.judge_output_root)
    judge_artifact, judge_created = publish_sealed_json(root / JUDGE_NAME, judge)
    score_artifact, score_created = publish_sealed_json(root / SCORE_NAME, score)
    return {
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "correct": score["correct"],
        "judge_created": judge_created,
        "judge_sha256": judge_artifact.sha256,
        "physical_provider_calls": 0,
        "score_created": score_created,
        "score_sha256": score_artifact.sha256,
    }


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    artifact, prompts, rows = _read_preflight(
        Path(args.judge_output_root),
        args.expected_judge_preflight_sha256,
    )
    batch = _run_batch(artifact, prompts, args=args, client=None)
    expected_judge, expected_score = materialization_payloads(artifact, rows, batch)
    root = Path(args.judge_output_root)
    judge = read_sealed_json(root / JUDGE_NAME)
    score = read_sealed_json(root / SCORE_NAME)
    _require(
        judge.sha256
        == require_sha256(args.expected_judge_sha256, "expected judge SHA-256")
        and score.sha256
        == require_sha256(args.expected_score_sha256, "expected score SHA-256")
        and canonical_json_bytes(judge.payload) == canonical_json_bytes(expected_judge)
        and canonical_json_bytes(score.payload) == canonical_json_bytes(expected_score),
        "specialist judge materialization differs from checkpoint replay",
    )
    judge_replay, _ = publish_sealed_json(root / JUDGE_REPLAY_NAME, expected_judge)
    score_replay, _ = publish_sealed_json(root / SCORE_REPLAY_NAME, expected_score)
    _require(
        judge_replay.sha256 == judge.sha256 and score_replay.sha256 == score.sha256,
        "specialist judge replay is not byte-identical",
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
    preflight.add_argument(
        "--dataset", type=Path, default=locked_judge_cli.DEFAULT_DATASET
    )
    preflight.add_argument("--split", type=Path, default=spine_cli.DEFAULT_SPLIT)

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
    "AnswerSeamRow",
    "CHECKPOINT_DIR_NAME",
    "DEFAULT_ANSWER_ROOT",
    "DEFAULT_JUDGE_ROOT",
    "DEFAULT_SOL_MODEL",
    "EXACT_ORDINALS",
    "JUDGE_FORMAT",
    "JUDGE_NAME",
    "JUDGE_REPLAY_NAME",
    "PREFLIGHT_FORMAT",
    "PREFLIGHT_NAME",
    "QUESTION_COUNT",
    "ReducedSpecialistJudgeError",
    "SCORE_FORMAT",
    "SCORE_NAME",
    "SCORE_REPLAY_NAME",
    "build_parser",
    "build_preflight_payload",
    "build_runtime",
    "load_verified_answer_seam",
    "main",
    "materialization_payloads",
    "run_materialize",
    "run_preflight",
    "run_provider",
    "run_replay",
    "validate_preflight_artifact",
]
