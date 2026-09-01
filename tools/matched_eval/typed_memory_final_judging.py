"""Locked Sol judge lifecycle for the common typed-memory final arm.

The responder is verified and replay-bound before this module opens benchmark
gold. ``full100`` and ``changed_only`` consume the complete stable judge seam;
``selected_subset`` consumes an explicitly sealed, outcome-conditioned
diagnostic projection. Provider execution accepts only the sealed judge-prompt
population; materialization and replay are checkpoint-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval._binary_judge_protocol import (
    JUDGE_MAX_TOKENS,
    parse_binary_judge_verdict,
)
from memory_condense.eval.benchmark import build_judge_prompt, exact_match, f1_score
from memory_condense.eval.fast_completion_runtime import (
    FastCompletionBatch,
    FastCompletionRuntime,
    preflight_fast_completion_prompts,
)
from memory_condense.eval.locked_split import load_split_manifest, select_locked_split
from memory_condense.eval.recall_guarded_cumulative_population import (
    LOCKED_LONGMEMEVAL_DATASET_SHA256,
    LOCKED_LONGMEMEVAL_SPLIT_MANIFEST_SHA256,
)
from memory_condense.ingest.loader import load_benchmark
from tools._routed_repair_routing import route_question

from . import judging
from .artifacts import SealedArtifact, read_sealed_json
from .contracts import (
    MatchedEvalContractError,
    identity_sha256,
    require_sha256,
    require_text,
)
from .population import EXPECTED_QUESTION_COUNT
from .typed_memory_final_arm import judge_row_projection


PREFLIGHT_FORMAT = "memory-condense-typed-memory-final-sol-judge-preflight-v1"
JUDGE_FORMAT = "memory-condense-typed-memory-final-sol-judge-v1"
SCORE_FORMAT = "memory-condense-typed-memory-final-sol-score-v1"
REPLAY_FORMAT = "memory-condense-typed-memory-final-sol-judge-replay-v1"
PREFLIGHT_NAME = "typed-final-judge-preflight-v1.json"
JUDGE_NAME = "typed-final-semantic-judge-sol-v1.json"
SCORE_NAME = "typed-final-score-ledger-v1.json"
REPLAY_NAME = "typed-final-semantic-judge-sol-replay-v1.json"
SCORE_REPLAY_NAME = "typed-final-score-ledger-replay-v1.json"
CHECKPOINT_DIR_NAME = "sol-typed-final-judge-calls-v1"
DEFAULT_MAX_PROMPT_TOKENS = judging.DEFAULT_MAX_JUDGE_PROMPT_TOKENS
JudgeMode = Literal["full100", "changed_only", "selected_subset"]


class TypedMemoryFinalJudgeError(MatchedEvalContractError):
    pass


def _require(ok: object, message: str) -> None:
    if not ok:
        raise TypedMemoryFinalJudgeError(message)


@dataclass(frozen=True, slots=True)
class TypedFinalJudgeGoldRow:
    ordinal: int
    question_id: str
    question: str
    question_sha256: str
    dated_question: str
    dated_question_sha256: str
    reference: str
    reference_sha256: str
    category: str


def validate_typed_final_run_artifact(
    artifact: SealedArtifact,
) -> tuple[dict[str, Any], ...]:
    """Validate the complete gold-free judge seam before dataset access."""

    payload = artifact.payload
    questions = payload.get("questions")
    judge_rows = payload.get("judge_rows")
    _require(
        payload.get("format")
        == "memory-condense-locked-typed-memory-final-arm-v1-run-v1"
        and payload.get("gold_loaded") is False
        and payload.get("physical_provider_calls_during_materialization") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("question_count") == EXPECTED_QUESTION_COUNT
        and type(questions) is list
        and type(judge_rows) is list
        and len(questions) == len(judge_rows) == EXPECTED_QUESTION_COUNT,
        "typed final judge source envelope changed",
    )
    validated: list[dict[str, Any]] = []
    question_ids: list[str] = []
    for ordinal, (source, projected) in enumerate(
        zip(questions, judge_rows, strict=True)
    ):
        _require(
            type(source) is dict and type(projected) is dict,
            "typed final judge source row changed type",
        )
        unsigned = dict(source)
        declared = unsigned.pop("source_row_sha256", None)
        _require(
            declared == identity_sha256(unsigned)
            and source.get("ordinal") == ordinal
            and judge_row_projection(source) == projected
            and source.get("prediction_sha256")
            == quote_sha256(require_text(source.get("prediction"), "judge prediction")),
            f"typed final judge source row changed at ordinal {ordinal}",
        )
        question_ids.append(
            require_text(source.get("question_id"), "judge source question")
        )
        validated.append(dict(projected))
    _require(
        len(set(question_ids)) == EXPECTED_QUESTION_COUNT,
        "typed final judge source question identities repeat",
    )
    return tuple(validated)


def load_verified_typed_final_judge_source(
    output_root: str | Path,
    *,
    expected_run_sha256: str,
    expected_replay_sha256: str,
) -> tuple[SealedArtifact, SealedArtifact, tuple[dict[str, Any], ...]]:
    root = Path(output_root)
    run = read_sealed_json(root / "typed-memory-final-run-v1.json")
    replay = read_sealed_json(root / "typed-memory-final-replay-v1.json")
    _require(
        run.sha256 == require_sha256(expected_run_sha256, "typed judge run")
        and replay.sha256
        == require_sha256(expected_replay_sha256, "typed judge replay")
        and replay.payload.get("format")
        == "memory-condense-locked-typed-memory-final-arm-v1-replay-v1"
        and replay.payload.get("byte_identical") is True
        and replay.payload.get("expected_run_sha256") == run.sha256
        and replay.payload.get("replayed_run_sha256") == run.sha256
        and replay.payload.get("physical_provider_calls") == 0
        and replay.payload.get("gold_loaded") is False,
        "typed final judge source is not replay-verified",
    )
    return run, replay, validate_typed_final_run_artifact(run)


def load_locked_typed_final_gold(
    *,
    dataset_path: str | Path,
    split_path: str | Path,
    source_rows: Sequence[Mapping[str, Any]],
    allow_subset: bool = False,
) -> tuple[tuple[TypedFinalJudgeGoldRow, ...], str]:
    dataset = Path(dataset_path)
    split = Path(split_path)
    _require(
        file_sha256(dataset) == LOCKED_LONGMEMEVAL_DATASET_SHA256
        and file_sha256(split) == LOCKED_LONGMEMEVAL_SPLIT_MANIFEST_SHA256,
        "locked typed-final judge dataset/split changed",
    )
    selected = select_locked_split(
        load_benchmark(dataset, "longmemeval"),
        dataset_path=dataset,
        manifest=load_split_manifest(split),
        split="validation",
    )
    questions = tuple(row for sample in selected for row in sample.questions)
    _require(
        len(selected) == len(questions) == EXPECTED_QUESTION_COUNT
        and type(allow_subset) is bool
        and (
            len(source_rows) == EXPECTED_QUESTION_COUNT
            if not allow_subset
            else 0 < len(source_rows) <= EXPECTED_QUESTION_COUNT
        ),
        "typed-final judge gold population changed",
    )
    source_ordinals = tuple(source.get("ordinal") for source in source_rows)
    _require(
        all(type(value) is int for value in source_ordinals)
        and (
            source_ordinals == tuple(range(EXPECTED_QUESTION_COUNT))
            if not allow_subset
            else source_ordinals == tuple(sorted(set(source_ordinals)))
            and all(0 <= value < EXPECTED_QUESTION_COUNT for value in source_ordinals)
        ),
        "typed-final judge source ordinal projection changed",
    )
    result: list[TypedFinalJudgeGoldRow] = []
    projection: list[dict[str, Any]] = []
    for source in source_rows:
        ordinal = source.get("ordinal")
        assert type(ordinal) is int
        question = questions[ordinal]
        question_sha = quote_sha256(question.question)
        dated_sha = quote_sha256(question.dated_question)
        _require(
            type(ordinal) is int
            and question.question_id == source.get("question_id")
            and question_sha == source.get("question_sha256")
            and dated_sha == source.get("dated_question_sha256"),
            f"typed-final judge gold binding changed at ordinal {ordinal}",
        )
        reference_sha = quote_sha256(question.answer)
        category = str(question.category or "uncategorized")
        row = TypedFinalJudgeGoldRow(
            ordinal,
            question.question_id,
            question.question,
            question_sha,
            question.dated_question,
            dated_sha,
            question.answer,
            reference_sha,
            category,
        )
        result.append(row)
        projection.append(
            {
                "category": category,
                "dated_question_sha256": dated_sha,
                "ordinal": ordinal,
                "question_id": question.question_id,
                "question_sha256": question_sha,
                "reference_sha256": reference_sha,
            }
        )
    return tuple(result), identity_sha256(projection)


def preflight_projection(
    *,
    run_artifact: SealedArtifact,
    replay_artifact_sha256: str,
    source_rows: tuple[dict[str, Any], ...],
    gold_rows: tuple[TypedFinalJudgeGoldRow, ...],
    gold_population_sha256: str,
    mode: JudgeMode,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> tuple[dict[str, Any], tuple[tuple[dict[str, str], ...], ...]]:
    _require(
        mode in {"full100", "changed_only", "selected_subset"},
        "typed judge mode changed",
    )
    require_text(model, "typed judge model")
    require_text(gateway_url, "typed judge gateway")
    _require(
        len(source_rows) == len(gold_rows)
        and bool(source_rows)
        and (
            len(source_rows) == EXPECTED_QUESTION_COUNT
            if mode != "selected_subset"
            else len(source_rows) <= EXPECTED_QUESTION_COUNT
        )
        and type(max_concurrency) is int
        and max_concurrency > 0,
        "typed judge preflight population/settings changed",
    )
    pending: list[dict[str, Any]] = []
    prompts: list[tuple[dict[str, str], ...]] = []
    for source, gold in zip(source_rows, gold_rows, strict=True):
        _require(
            source["ordinal"] == gold.ordinal
            and source["question_id"] == gold.question_id
            and source["question_sha256"] == gold.question_sha256
            and source["dated_question_sha256"] == gold.dated_question_sha256,
            "typed judge source/gold order changed",
        )
        if mode == "changed_only" and not source["changed_from_parent"]:
            continue
        messages = tuple(
            dict(row)
            for row in build_judge_prompt(
                gold.question,
                gold.reference,
                source["prediction"],
            )
        )
        pending.append(
            {
                "category": gold.category,
                "dated_question_sha256": gold.dated_question_sha256,
                "demand_class": route_question(gold.dated_question).style.value,
                "messages": list(messages),
                "messages_sha256": identity_sha256(list(messages)),
                "ordinal": gold.ordinal,
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
    _require(bool(prompts), "typed judge selected population is empty")
    population = preflight_fast_completion_prompts(
        prompts,
        max_prompt_tokens=DEFAULT_MAX_PROMPT_TOKENS,
    )
    _require(
        population.logical_prompt_count
        == population.unique_prompt_count
        == len(prompts),
        "typed judge prompts must be one unique call per selected row",
    )
    rows: list[dict[str, Any]] = []
    for raw, receipt in zip(pending, population.ordered_rows, strict=True):
        _require(
            raw["messages_sha256"] == receipt.messages_sha256,
            "typed judge preflight messages changed",
        )
        body = {**raw, "prompt_token_proxy": receipt.prompt_token_proxy}
        rows.append({**body, "prompt_row_receipt_sha256": identity_sha256(body)})
    payload = {
        "format": PREFLIGHT_FORMAT,
        "gateway_url": gateway_url,
        "gold_loaded": True,
        "gold_population_sha256": require_sha256(
            gold_population_sha256, "typed judge gold population"
        ),
        "judge_mode": mode,
        "max_concurrency": max_concurrency,
        "model": model,
        "physical_provider_calls": 0,
        "prompt_population": population.model_dump(),
        "prompt_population_sha256": population.prompt_population_sha256,
        "prompt_rows": rows,
        "question_count": EXPECTED_QUESTION_COUNT,
        "required_authorized_provider_calls": len(rows),
        "retained_transformer_token_state_bytes": 0,
        "selected_question_count": len(rows),
        "typed_final_replay_sha256": require_sha256(
            replay_artifact_sha256, "typed judge source replay"
        ),
        "typed_final_run_sha256": run_artifact.sha256,
    }
    return payload, tuple(prompts)


def validate_preflight_artifact(
    artifact: SealedArtifact,
) -> tuple[tuple[tuple[dict[str, str], ...], ...], tuple[dict[str, Any], ...]]:
    payload = artifact.payload
    rows = payload.get("prompt_rows")
    _require(
        payload.get("format") == PREFLIGHT_FORMAT
        and payload.get("gold_loaded") is True
        and payload.get("physical_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("judge_mode")
        in {"full100", "changed_only", "selected_subset"}
        and payload.get("question_count") == EXPECTED_QUESTION_COUNT
        and type(rows) is list
        and bool(rows)
        and len(rows) == payload.get("selected_question_count")
        == payload.get("required_authorized_provider_calls"),
        "typed judge sealed preflight changed",
    )
    prompts: list[tuple[dict[str, str], ...]] = []
    validated: list[dict[str, Any]] = []
    for raw in rows:
        _require(type(raw) is dict, "typed judge prompt row changed type")
        body = dict(raw)
        declared = body.pop("prompt_row_receipt_sha256", None)
        messages = raw.get("messages")
        _require(
            declared == identity_sha256(body)
            and type(messages) is list
            and identity_sha256(messages) == raw.get("messages_sha256"),
            "typed judge prompt row seal changed",
        )
        prompts.append(tuple(dict(row) for row in messages))
        validated.append(dict(raw))
    population = preflight_fast_completion_prompts(
        prompts,
        max_prompt_tokens=DEFAULT_MAX_PROMPT_TOKENS,
    )
    _require(
        population.model_dump() == payload.get("prompt_population")
        and population.prompt_population_sha256
        == payload.get("prompt_population_sha256")
        and population.unique_prompt_count == len(rows),
        "typed judge sealed prompt population changed",
    )
    return tuple(prompts), tuple(validated)


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
    payload = artifact.payload
    _require(
        payload.get("model") == model
        and payload.get("gateway_url") == gateway_url
        and payload.get("max_concurrency") == max_concurrency,
        "typed judge runtime settings differ from preflight",
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
            "arm": "locked_common_typed_memory_final_sol_judge_v1",
            "authorized_unique_calls": len(prompts),
            "experiment_format": JUDGE_FORMAT,
            "judge_mode": payload["judge_mode"],
            "preflight_artifact_sha256": artifact.sha256,
            "typed_final_run_sha256": payload["typed_final_run_sha256"],
        },
    )


def materialization_projection(
    preflight: SealedArtifact,
    prompt_rows: tuple[dict[str, Any], ...],
    batch: FastCompletionBatch,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = len(prompt_rows)
    _require(
        batch.usage.logical_calls == batch.usage.unique_calls == selected
        and batch.usage.checkpoint_hits == selected
        and batch.usage.physical_calls == 0
        and len(batch.logical_completions) == selected
        and len(batch.unique_records) == selected,
        "typed judge materialization requires complete checkpoint hits",
    )
    records = {row.messages_sha256: row for row in batch.unique_records}
    _require(len(records) == selected, "typed judge completion identities repeat")
    rows: list[dict[str, Any]] = []
    for prompt, completion in zip(
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
            "typed judge checkpoint record changed",
        )
        assert record is not None
        verdict = parse_binary_judge_verdict(completion)
        _require(type(verdict) is bool, "typed judge returned an invalid verdict")
        body = {
            "call_key_sha256": record.call_key_sha256,
            "category": prompt["category"],
            "correct": verdict,
            "dated_question_sha256": prompt["dated_question_sha256"],
            "demand_class": prompt["demand_class"],
            "judge_output": completion,
            "judge_output_sha256": quote_sha256(completion),
            "messages_sha256": prompt["messages_sha256"],
            "normalized_exact_match": exact_match(
                prompt["prediction"], prompt["reference"]
            ),
            "normalized_f1": f1_score(
                prompt["prediction"], prompt["reference"]
            ),
            "ordinal": prompt["ordinal"],
            "prediction_sha256": prompt["prediction_sha256"],
            "prediction_source": prompt["prediction_source"],
            "prompt_row_receipt_sha256": prompt["prompt_row_receipt_sha256"],
            "question_id": prompt["question_id"],
            "question_sha256": prompt["question_sha256"],
            "reference_sha256": prompt["reference_sha256"],
            "request_journal_sha256": record.request_journal_sha256,
            "response_journal_sha256": record.response_journal_sha256,
            "route_id": prompt["route_id"],
            "source_row_sha256": prompt["source_row_sha256"],
        }
        rows.append({**body, "judge_row_sha256": identity_sha256(body)})
    correct = sum(row["correct"] for row in rows)
    judge = {
        "aggregate": {
            "accuracy": correct / selected,
            "correct": correct,
            "question_count": selected,
        },
        "completion_batch": judging._stable_batch(batch),  # noqa: SLF001
        "format": JUDGE_FORMAT,
        "gold_loaded": True,
        "gold_population_sha256": preflight.payload["gold_population_sha256"],
        "judge_mode": preflight.payload["judge_mode"],
        "physical_provider_calls_during_materialization": 0,
        "preflight_artifact_sha256": preflight.sha256,
        "question_count": EXPECTED_QUESTION_COUNT,
        "questions": rows,
        "retained_transformer_token_state_bytes": 0,
        "selected_question_count": selected,
        "typed_final_run_sha256": preflight.payload["typed_final_run_sha256"],
    }
    score = {
        "correct": correct,
        "format": SCORE_FORMAT,
        "judge_mode": preflight.payload["judge_mode"],
        "question_count": EXPECTED_QUESTION_COUNT,
        "selected_accuracy": correct / selected,
        "selected_question_count": selected,
        "typed_final_run_sha256": preflight.payload["typed_final_run_sha256"],
    }
    return judge, score


__all__ = [
    "CHECKPOINT_DIR_NAME",
    "JUDGE_FORMAT",
    "JUDGE_NAME",
    "JudgeMode",
    "PREFLIGHT_FORMAT",
    "PREFLIGHT_NAME",
    "REPLAY_FORMAT",
    "REPLAY_NAME",
    "SCORE_FORMAT",
    "SCORE_NAME",
    "SCORE_REPLAY_NAME",
    "TypedFinalJudgeGoldRow",
    "TypedMemoryFinalJudgeError",
    "build_runtime",
    "load_locked_typed_final_gold",
    "load_verified_typed_final_judge_source",
    "materialization_projection",
    "preflight_projection",
    "validate_preflight_artifact",
    "validate_typed_final_run_artifact",
]
