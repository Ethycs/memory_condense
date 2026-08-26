"""Answer, replay, and score helpers for the streamlined Hebbian runner.

The history and prompt-population phases live in ``run_fast_1m_hebbian``.
This module owns the provider runtime boundary and the immutable answer/replay
artifact algebra so the command runner remains a small orchestration facade.
"""

from __future__ import annotations

import argparse
import ssl
import statistics
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval.benchmark import exact_match, f1_score
from memory_condense.eval.fast_completion_runtime import (
    FAST_COMPLETION_RUNTIME_FORMAT,
    FastCompletionRuntime,
    preflight_fast_completion_prompts,
)
from memory_condense.eval.fast_hebbian_prompts import ARM_IDS, S0_STAGE_ID


ANSWER_MANIFEST_FORMAT = "memory-condense-fast-1m-hebbian-answers-v1"
SCORE_MANIFEST_FORMAT = "memory-condense-fast-1m-hebbian-scores-v1"
ANSWER_BINDING_FORMAT = "memory-condense-fast-1m-hebbian-answer-binding-v1"
ZERO_STATE_CONTRACT = "tensor-free-fast-1m-hebbian-phase-boundary-v1"

_ANSWER_FIELDS = frozenset(
    {
        "format",
        "mode",
        "experiment_binding",
        "prompt_population",
        "completion_batch",
        "question_count",
        "logical_answer_count",
        "unique_completion_count",
        "answers",
        "gold_fields_present",
        "zero_state",
    }
)
_ANSWER_ROW_FIELDS = frozenset(
    {
        "logical_ordinal",
        "question_ordinal",
        "question_id",
        "stage_id",
        "arm_id",
        "arm_prompt_sha256",
        "messages_sha256",
        "unique_prompt_ordinal",
        "prompt_token_proxy",
        "hard_prompt_token_cap",
        "chunk_ids",
        "alias_order",
        "prediction",
        "prediction_sha256",
    }
)
_COMPLETION_BATCH_FIELDS = frozenset(
    {
        "logical_completions",
        "unique_records",
        "usage",
        "provenance",
        "runtime_identity_sha256",
        "prompt_population",
    }
)
_COMPLETION_PROVENANCE_FIELDS = frozenset(
    {
        "format",
        "model",
        "max_new_tokens",
        "max_prompt_token_proxy",
        "max_concurrency",
        "retries",
        "request_options",
        "prompt_population_sha256",
        "prompt_token_proxy_identity",
        "benchmark_provenance",
        "persisted_transformer_token_state",
        "retained_transformer_token_state_bytes",
        "external_provider_persistence_certified",
    }
)
_COMPLETION_RECORD_FIELDS = frozenset(
    {
        "call_key_sha256",
        "request_journal_sha256",
        "response_journal_sha256",
        "messages_sha256",
        "completion",
        "completion_sha256",
        "requested_model",
        "response_id",
        "response_model",
        "finish_reason",
        "prompt_token_proxy",
        "completion_token_proxy",
        "reported_prompt_tokens",
        "reported_completion_tokens",
        "reported_total_tokens",
        "provider_elapsed_s",
        "checkpoint_hit",
        "physical_call",
    }
)
_COMPLETION_USAGE_FIELDS = frozenset(
    {
        "logical_calls",
        "unique_calls",
        "deduplicated_logical_calls",
        "physical_calls",
        "checkpoint_hits",
        "prompt_token_proxy",
        "completion_token_proxy",
        "recorded_reported_prompt_tokens",
        "recorded_reported_completion_tokens",
        "recorded_reported_total_tokens",
        "reported_prompt_tokens_complete",
        "reported_completion_tokens_complete",
        "reported_total_tokens_complete",
        "recorded_provider_elapsed_s",
    }
)
_RUNTIME_POPULATION_FIELDS = frozenset(
    {
        "format",
        "logical_prompt_count",
        "unique_prompt_count",
        "ordered_rows",
        "prompt_population_sha256",
        "max_prompt_token_proxy",
        "prompt_token_proxy_identity",
    }
)
_FORBIDDEN_SECRET_FIELDS = frozenset(
    {"api_key", "api-key", "authorization", "litellm_key"}
)
_RECORD_DISPOSITION_FIELDS = frozenset({"checkpoint_hit", "physical_call"})
_USAGE_DISPOSITION_FIELDS = frozenset({"physical_calls", "checkpoint_hits"})


def _contains_forbidden_secret_field(value: object) -> bool:
    if type(value) is dict:
        return any(
            str(key).casefold() in _FORBIDDEN_SECRET_FIELDS
            or _contains_forbidden_secret_field(item)
            for key, item in value.items()
        )
    if type(value) is list:
        return any(_contains_forbidden_secret_field(item) for item in value)
    return False


def make_provider_client(api_key: str, gateway_url: str) -> Any:
    """Create the zero-retry OpenAI-compatible client used by central LiteLLM."""

    import httpx
    import truststore
    from openai import OpenAI

    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return OpenAI(
        api_key=api_key,
        base_url=gateway_url,
        http_client=httpx.Client(verify=context),
        max_retries=0,
    )


def experiment_binding(experiment: Any, prompts: Any) -> dict[str, Any]:
    return {
        "format": ANSWER_BINDING_FORMAT,
        "retrieval_sha256": experiment.artifact.raw_sha256,
        "population_identity_sha256": (
            experiment.artifact.population_identity_sha256
        ),
        "original_source_store_receipt_sha256": (
            experiment.artifact.source_store_receipt_sha256
        ),
        "combined_store_receipt_sha256": experiment.source.receipt.receipt_sha256,
        "retrieval_implementation_sha256": (
            experiment.artifact.retrieval_implementation_sha256
        ),
        "retrieval_policy_sha256": experiment.artifact.retrieval_policy_sha256,
        "source_manifest_sha256": experiment.source.manifest_sha256,
        "source_database_sha256": experiment.source.receipt.target_database_sha256,
        "source_index_sha256": experiment.source.receipt.target_index_sha256,
        "history_file_sha256": experiment.history_file_sha256,
        "history_artifact_sha256": experiment.history.artifact_sha256,
        "history_receipt_sha256": experiment.history.receipt.receipt_sha256,
        "direct_capture_sha256": experiment.history.receipt.direct_capture_sha256,
        "capture_policy_sha256": experiment.history.receipt.capture_policy_sha256,
        "derived_manifest_sha256": experiment.derived_manifest_sha256,
        "derived_store_receipt_sha256": experiment.derived.receipt_sha256,
        "derived_database_sha256": experiment.derived.derived_database_sha256,
        "derived_index_sha256": experiment.derived.derived_index_sha256,
        "learning_policy_sha256": experiment.derived.learning_policy_sha256,
        "association_artifact_id": experiment.derived.association_artifact_id,
        "association_artifact_sha256": (
            experiment.derived.association_artifact_sha256
        ),
        "implementation_sha256": experiment.history.receipt.implementation_sha256,
        "environment_lock_sha256": (
            experiment.history.receipt.environment_lock_sha256
        ),
        "prompt_population_sha256": prompts.prompt_population_sha256,
        "stage_id": S0_STAGE_ID,
        "logical_prompt_count": prompts.logical_prompt_count,
        "unique_prompt_count": prompts.unique_prompt_count,
        "retained_request_token_state_bytes": 0,
    }


def benchmark_provenance(
    binding: Mapping[str, Any],
    *,
    caller_model: str,
    gateway_url: str,
) -> dict[str, Any]:
    return {
        "format": ANSWER_BINDING_FORMAT,
        "experiment_binding_sha256": identity_sha256(dict(binding)),
        "retrieval_sha256": binding["retrieval_sha256"],
        "combined_store_receipt_sha256": binding[
            "combined_store_receipt_sha256"
        ],
        "source_manifest_sha256": binding["source_manifest_sha256"],
        "history_artifact_sha256": binding["history_artifact_sha256"],
        "history_receipt_sha256": binding["history_receipt_sha256"],
        "derived_store_receipt_sha256": binding[
            "derived_store_receipt_sha256"
        ],
        "association_artifact_id": binding["association_artifact_id"],
        "prompt_population_sha256": binding["prompt_population_sha256"],
        "authorized_unique_calls": binding["unique_prompt_count"],
        "caller_model_alias": caller_model,
        "gateway_url": gateway_url,
        "gold_blind": True,
        "retained_request_token_state_bytes": 0,
    }


def _answer_rows(prompts: Any, completions: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt, prediction in zip(
        prompts.logical_prompts, completions, strict=True
    ):
        rows.append(
            {
                "logical_ordinal": prompt.logical_ordinal,
                "question_ordinal": prompt.question_ordinal,
                "question_id": prompt.question_id,
                "stage_id": prompt.stage_id,
                "arm_id": prompt.arm_id,
                "arm_prompt_sha256": prompt.arm_prompt_sha256,
                "messages_sha256": prompt.messages_sha256,
                "unique_prompt_ordinal": prompt.unique_prompt_ordinal,
                "prompt_token_proxy": prompt.prompt_token_proxy,
                "hard_prompt_token_cap": prompt.hard_prompt_token_cap,
                "chunk_ids": list(prompt.chunk_ids),
                "alias_order": list(prompt.alias_order),
                "prediction": prediction,
                "prediction_sha256": quote_sha256(prediction),
            }
        )
    return rows


def answer_artifact(
    *,
    mode: str,
    experiment: Any,
    prompts: Any,
    completion_batch: Any,
) -> dict[str, Any]:
    if mode not in {"answer", "replay"}:
        raise ValueError("answer artifact mode must be answer or replay")
    answers = _answer_rows(prompts, completion_batch.logical_completions)
    return {
        "format": ANSWER_MANIFEST_FORMAT,
        "mode": mode,
        "experiment_binding": experiment_binding(experiment, prompts),
        "prompt_population": prompts.identity_payload(),
        "completion_batch": completion_batch.model_dump(),
        "question_count": experiment.artifact.question_count,
        "logical_answer_count": len(answers),
        "unique_completion_count": len(completion_batch.unique_records),
        "answers": answers,
        "gold_fields_present": False,
        "zero_state": {
            "contract": ZERO_STATE_CONTRACT,
            "persisted_transformer_token_state": False,
            "retained_transformer_token_state_bytes": 0,
            "external_provider_persistence_certified": False,
        },
    }


def answers_path(args: argparse.Namespace) -> Path:
    return Path(args.answers or Path(args.output_root) / "answers.json")


def replay_path(args: argparse.Namespace) -> Path:
    return Path(args.replay or Path(args.output_root) / "replay.json")


def checkpoint_path(args: argparse.Namespace) -> Path:
    return answers_path(args).parent / "completion-calls"


def _expected_answer_projection(prompt: Any) -> dict[str, Any]:
    return {
        "logical_ordinal": prompt.logical_ordinal,
        "question_ordinal": prompt.question_ordinal,
        "question_id": prompt.question_id,
        "stage_id": prompt.stage_id,
        "arm_id": prompt.arm_id,
        "arm_prompt_sha256": prompt.arm_prompt_sha256,
        "messages_sha256": prompt.messages_sha256,
        "unique_prompt_ordinal": prompt.unique_prompt_ordinal,
        "prompt_token_proxy": prompt.prompt_token_proxy,
        "hard_prompt_token_cap": prompt.hard_prompt_token_cap,
        "chunk_ids": list(prompt.chunk_ids),
        "alias_order": list(prompt.alias_order),
    }


def stable_completion_batch_projection(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Project away only the cache disposition that necessarily changes on replay."""

    return {
        "logical_completions": batch["logical_completions"],
        "unique_records": [
            {
                name: value
                for name, value in record.items()
                if name not in _RECORD_DISPOSITION_FIELDS
            }
            for record in batch["unique_records"]
        ],
        "usage": {
            name: value
            for name, value in batch["usage"].items()
            if name not in _USAGE_DISPOSITION_FIELDS
        },
        "provenance": batch["provenance"],
        "runtime_identity_sha256": batch["runtime_identity_sha256"],
        "prompt_population": batch["prompt_population"],
    }


def read_and_validate_answers(
    experiment: Any,
    prompts: Any,
    path: Path,
    *,
    expected_mode: str,
    read_canonical_json: Callable[[Path], tuple[dict[str, Any], str]],
    is_digest: Callable[[object], bool],
) -> tuple[dict[str, Any], str]:
    payload, digest = read_canonical_json(path)
    if set(payload) != _ANSWER_FIELDS:
        raise ValueError("answer manifest has a noncanonical shape")
    if (
        payload["format"] != ANSWER_MANIFEST_FORMAT
        or payload["mode"] != expected_mode
        or payload["gold_fields_present"] is not False
    ):
        raise ValueError("answer manifest format, mode, or gold boundary changed")
    if payload["experiment_binding"] != experiment_binding(experiment, prompts):
        raise ValueError("answer manifest changed its upstream experiment binding")
    if payload["prompt_population"] != prompts.identity_payload():
        raise ValueError("answer prompt population does not verify")
    zero_state = payload["zero_state"]
    if zero_state != {
        "contract": ZERO_STATE_CONTRACT,
        "persisted_transformer_token_state": False,
        "retained_transformer_token_state_bytes": 0,
        "external_provider_persistence_certified": False,
    }:
        raise ValueError("answer manifest changed the zero-state boundary")
    if (
        type(payload["question_count"]) is not int
        or payload["question_count"] != experiment.artifact.question_count
        or type(payload["logical_answer_count"]) is not int
        or payload["logical_answer_count"] != prompts.logical_prompt_count
        or type(payload["unique_completion_count"]) is not int
        or payload["unique_completion_count"] != prompts.unique_prompt_count
    ):
        raise ValueError("answer manifest changed population cardinality")

    raw_answers = payload["answers"]
    if type(raw_answers) is not list or len(raw_answers) != prompts.logical_prompt_count:
        raise ValueError("answer row population changed")
    predictions: list[str] = []
    for raw, prompt in zip(raw_answers, prompts.logical_prompts, strict=True):
        if type(raw) is not dict or set(raw) != _ANSWER_ROW_FIELDS:
            raise ValueError("answer row has a noncanonical shape")
        expected = _expected_answer_projection(prompt)
        if any(raw[name] != value for name, value in expected.items()):
            raise ValueError("answer row changed prompt provenance")
        prediction = raw["prediction"]
        if (
            type(prediction) is not str
            or not prediction
            or raw["prediction_sha256"] != quote_sha256(prediction)
        ):
            raise ValueError("answer prediction does not verify")
        predictions.append(prediction)

    batch = payload["completion_batch"]
    if type(batch) is not dict or set(batch) != _COMPLETION_BATCH_FIELDS:
        raise ValueError("answer completion batch has a noncanonical shape")
    if batch["logical_completions"] != predictions:
        raise ValueError("answer rows disagree with the completion batch")
    provenance = batch["provenance"]
    if type(provenance) is not dict or set(provenance) != (
        _COMPLETION_PROVENANCE_FIELDS
    ):
        raise ValueError("answer completion provenance has a noncanonical shape")
    if (
        provenance["retries"] != 0
        or provenance["persisted_transformer_token_state"] is not False
        or provenance["retained_transformer_token_state_bytes"] != 0
        or provenance["external_provider_persistence_certified"] is not False
    ):
        raise ValueError("answer runtime changed its retry or zero-state policy")
    if (
        provenance["format"] != FAST_COMPLETION_RUNTIME_FORMAT
        or type(provenance["model"]) is not str
        or not provenance["model"]
        or type(provenance["max_new_tokens"]) is not int
        or provenance["max_new_tokens"] < 1
        or type(provenance["max_concurrency"]) is not int
        or provenance["max_concurrency"] < 1
        or provenance["request_options"] != {}
        or type(provenance["prompt_token_proxy_identity"]) is not dict
    ):
        raise ValueError("answer runtime configuration is invalid")
    max_prompt_tokens = provenance["max_prompt_token_proxy"]
    if type(max_prompt_tokens) is not int or not 1 <= max_prompt_tokens <= 8_000:
        raise ValueError("answer runtime prompt cap is invalid")
    expected_runtime_population = preflight_fast_completion_prompts(
        prompts.logical_message_population,
        max_prompt_tokens=max_prompt_tokens,
    ).model_dump()
    runtime_population = batch["prompt_population"]
    if (
        type(runtime_population) is not dict
        or set(runtime_population) != _RUNTIME_POPULATION_FIELDS
        or runtime_population != expected_runtime_population
    ):
        raise ValueError("answer runtime prompt population changed")
    if provenance["prompt_population_sha256"] != expected_runtime_population[
        "prompt_population_sha256"
    ]:
        raise ValueError("answer runtime changed its prompt population seal")

    benchmark = provenance["benchmark_provenance"]
    binding = experiment_binding(experiment, prompts)
    if type(benchmark) is not dict:
        raise ValueError("answer benchmark provenance is missing")
    caller = benchmark.get("caller_model_alias")
    gateway = benchmark.get("gateway_url")
    if (
        type(caller) is not str
        or not caller
        or type(gateway) is not str
        or not gateway
        or benchmark
        != benchmark_provenance(binding, caller_model=caller, gateway_url=gateway)
    ):
        raise ValueError("answer benchmark provenance changed")
    if _contains_forbidden_secret_field(payload):
        raise ValueError("answer artifact serialized provider credentials")

    unique_records = batch["unique_records"]
    if type(unique_records) is not list or len(unique_records) != (
        prompts.unique_prompt_count
    ):
        raise ValueError("answer unique completion population changed")
    completions_by_messages: dict[str, str] = {}
    for record in unique_records:
        if type(record) is not dict or set(record) != _COMPLETION_RECORD_FIELDS:
            raise ValueError("answer completion record has a noncanonical shape")
        messages_sha = record["messages_sha256"]
        completion = record["completion"]
        if (
            not is_digest(messages_sha)
            or type(completion) is not str
            or not completion
            or record["completion_sha256"] != quote_sha256(completion)
            or record["finish_reason"] != "stop"
            or messages_sha in completions_by_messages
            or record["requested_model"] != provenance["model"]
            or type(record["checkpoint_hit"]) is not bool
            or type(record["physical_call"]) is not bool
            or record["checkpoint_hit"] == record["physical_call"]
        ):
            raise ValueError("answer completion record does not verify")
        for name in (
            "call_key_sha256",
            "request_journal_sha256",
            "response_journal_sha256",
        ):
            if not is_digest(record[name]):
                raise ValueError("answer completion journal seal is invalid")
        completions_by_messages[messages_sha] = completion
    expected_unique_order = tuple(
        dict.fromkeys(prompt.messages_sha256 for prompt in prompts.logical_prompts)
    )
    if tuple(record["messages_sha256"] for record in unique_records) != (
        expected_unique_order
    ):
        raise ValueError("answer unique completion order changed")
    for prompt, prediction in zip(
        prompts.logical_prompts, predictions, strict=True
    ):
        if completions_by_messages.get(prompt.messages_sha256) != prediction:
            raise ValueError("answer prediction is not its journaled completion")

    usage = batch["usage"]
    if (
        type(usage) is not dict
        or set(usage) != _COMPLETION_USAGE_FIELDS
        or usage["logical_calls"] != prompts.logical_prompt_count
        or usage["unique_calls"] != prompts.unique_prompt_count
        or usage["deduplicated_logical_calls"]
        != prompts.logical_prompt_count - prompts.unique_prompt_count
        or type(usage["physical_calls"]) is not int
        or type(usage["checkpoint_hits"]) is not int
        or usage["physical_calls"] + usage["checkpoint_hits"]
        != prompts.unique_prompt_count
        or usage["physical_calls"]
        != sum(bool(record["physical_call"]) for record in unique_records)
        or usage["checkpoint_hits"]
        != sum(bool(record["checkpoint_hit"]) for record in unique_records)
    ):
        raise ValueError("answer completion usage changed")
    runtime_identity = batch["runtime_identity_sha256"]
    if (
        not is_digest(runtime_identity)
        or runtime_identity != identity_sha256(provenance)
    ):
        raise ValueError("answer runtime identity does not verify")
    return payload, digest


def replay_answer_journals(
    *,
    answers: Mapping[str, Any],
    prompts: Any,
    checkpoint_dir: Path,
) -> Any:
    """Reopen and verify every immutable provider request/response journal."""

    batch = answers["completion_batch"]
    provenance = batch["provenance"]
    runtime = FastCompletionRuntime(
        checkpoint_dir=checkpoint_dir,
        prompt_population=prompts.logical_message_population,
        model=provenance["model"],
        client=None,
        max_prompt_tokens=provenance["max_prompt_token_proxy"],
        max_new_tokens=provenance["max_new_tokens"],
        max_concurrency=provenance["max_concurrency"],
        retries=provenance["retries"],
        request_options=provenance["request_options"],
        benchmark_provenance=provenance["benchmark_provenance"],
    )
    with runtime:
        replay = runtime.run()
    expected_predictions = tuple(row["prediction"] for row in answers["answers"])
    if (
        replay.logical_completions != expected_predictions
        or replay.runtime_identity_sha256 != batch["runtime_identity_sha256"]
        or tuple(row.response_journal_sha256 for row in replay.unique_records)
        != tuple(row["response_journal_sha256"] for row in batch["unique_records"])
    ):
        raise ValueError("answer manifest disagrees with immutable provider journals")
    if stable_completion_batch_projection(
        batch
    ) != stable_completion_batch_projection(replay.model_dump()):
        raise ValueError(
            "answer completion records or aggregate usage disagree with "
            "immutable provider journals"
        )
    return replay


def validate_answer_replay_pair(
    answers: Mapping[str, Any], replay: Mapping[str, Any]
) -> None:
    if answers["experiment_binding"] != replay["experiment_binding"] or answers[
        "prompt_population"
    ] != replay["prompt_population"]:
        raise ValueError("answer and replay artifacts bind different experiments")
    if tuple(row["prediction"] for row in answers["answers"]) != tuple(
        row["prediction"] for row in replay["answers"]
    ):
        raise ValueError("answer and replay predictions differ")
    answer_batch = answers["completion_batch"]
    replay_batch = replay["completion_batch"]
    if (
        answer_batch["runtime_identity_sha256"]
        != replay_batch["runtime_identity_sha256"]
        or tuple(
            row["response_journal_sha256"]
            for row in answer_batch["unique_records"]
        )
        != tuple(
            row["response_journal_sha256"]
            for row in replay_batch["unique_records"]
        )
    ):
        raise ValueError("answer and replay journal populations differ")
    if stable_completion_batch_projection(
        answer_batch
    ) != stable_completion_batch_projection(replay_batch):
        raise ValueError(
            "answer and replay completion records or aggregate usage differ"
        )
    replay_usage = replay_batch["usage"]
    if replay_usage["physical_calls"] != 0 or replay_usage["checkpoint_hits"] != replay[
        "unique_completion_count"
    ]:
        raise ValueError("replay artifact was not a provider-free journal replay")


def load_gold_population(dataset: Path, split: Path) -> Any:
    # Intentionally lazy: no other phase can import or reach gold answers.
    from memory_condense.eval.recall_guarded_cumulative_1m import (
        load_original_population,
    )

    return load_original_population(dataset, split)


def score_artifact(
    *,
    experiment: Any,
    answers: Mapping[str, Any],
    answer_sha256: str,
    replay_sha256: str,
    gold_population: Any,
) -> dict[str, Any]:
    gold_by_id = {
        question.question_id: question for question in gold_population.questions
    }
    expected_ids = tuple(
        question.question_id for question in experiment.artifact.questions
    )
    if (
        len(gold_by_id) != len(gold_population.questions)
        or tuple(gold_by_id) != expected_ids
    ):
        raise RuntimeError("post-hoc gold population changed question order")

    scored_rows: list[dict[str, Any]] = []
    for row in answers["answers"]:
        gold = gold_by_id[row["question_id"]]
        prediction = row["prediction"]
        scored_rows.append(
            {
                "logical_ordinal": row["logical_ordinal"],
                "question_ordinal": row["question_ordinal"],
                "question_id": row["question_id"],
                "stage_id": row["stage_id"],
                "arm_id": row["arm_id"],
                "category": gold.category,
                "prediction_sha256": row["prediction_sha256"],
                "gold_answer_sha256": quote_sha256(gold.answer),
                "exact_match": exact_match(prediction, gold.answer),
                "f1": f1_score(prediction, gold.answer),
            }
        )
    aggregates: list[dict[str, Any]] = []
    for arm_id in ARM_IDS:
        rows = [row for row in scored_rows if row["arm_id"] == arm_id]
        if not rows:
            raise RuntimeError(f"score population omitted {arm_id}")
        aggregates.append(
            {
                "stage_id": S0_STAGE_ID,
                "arm_id": arm_id,
                "questions": len(rows),
                "exact_matches": sum(bool(row["exact_match"]) for row in rows),
                "exact_match_rate": statistics.fmean(
                    float(row["exact_match"]) for row in rows
                ),
                "mean_f1": statistics.fmean(float(row["f1"]) for row in rows),
            }
        )
    return {
        "format": SCORE_MANIFEST_FORMAT,
        "experiment_binding": answers["experiment_binding"],
        "answer_manifest_sha256": answer_sha256,
        "replay_manifest_sha256": replay_sha256,
        "gold_loaded_posthoc": True,
        "question_count": experiment.artifact.question_count,
        "logical_score_count": len(scored_rows),
        "aggregates": aggregates,
        "rows": scored_rows,
        "retained_request_token_state_bytes": 0,
    }


__all__ = [
    "ANSWER_BINDING_FORMAT",
    "ANSWER_MANIFEST_FORMAT",
    "SCORE_MANIFEST_FORMAT",
    "ZERO_STATE_CONTRACT",
    "answer_artifact",
    "answers_path",
    "benchmark_provenance",
    "checkpoint_path",
    "experiment_binding",
    "load_gold_population",
    "make_provider_client",
    "read_and_validate_answers",
    "replay_answer_journals",
    "replay_path",
    "score_artifact",
    "stable_completion_batch_projection",
    "validate_answer_replay_pair",
]
