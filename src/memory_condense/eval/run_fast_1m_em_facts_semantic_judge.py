"""Run an independent Sol semantic judge over sealed EM-fact answers.

The upstream EM run and both of its completion-journal populations are
replayed before benchmark gold is reachable.  Arm identity is kept in local
bookkeeping and is never added to the judge messages.  Provider work is
limited to the exact preflighted population and can be replayed without a
client.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval import run_fast_1m_em_facts as em_runner
from memory_condense.eval._binary_judge_protocol import (
    JUDGE_MAX_TOKENS,
    parse_binary_judge_verdict,
)
from memory_condense.eval.benchmark import build_judge_prompt
from memory_condense.eval.fast_completion_runtime import (
    FastCompletionBatch,
    FastCompletionRuntime,
    FastPromptPopulation,
    preflight_fast_completion_prompts,
)
from memory_condense.eval.fast_em_fact_memory import (
    DEFAULT_EM_STAGE_ID,
    EM_FACT_ARMS,
    EMFactArm,
    EMFactPolicy,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    ORIGINAL_1M_RETRIEVAL_SHA256,
    FastRetrievalArtifact,
)


PREFLIGHT_FORMAT = "memory-condense-fast-1m-em-facts-sol-judge-preflight-v1"
JUDGE_MANIFEST_FORMAT = "memory-condense-fast-1m-em-facts-sol-judge-v1"
JUDGE_BINDING_FORMAT = "memory-condense-fast-1m-em-facts-sol-judge-binding-v1"
JUDGE_POLICY_FORMAT = "memory-condense-fast-1m-em-facts-sol-judge-policy-v1"
ZERO_STATE_CONTRACT = "stateless-fast-1m-em-facts-sol-judge-boundary-v1"

DEFAULT_UPSTREAM_ROOT = em_runner.DEFAULT_OUTPUT_ROOT
DEFAULT_OUTPUT_ROOT = DEFAULT_UPSTREAM_ROOT
DEFAULT_RETRIEVAL = em_runner.DEFAULT_RETRIEVAL
DEFAULT_SPLIT = em_runner.DEFAULT_SPLIT
DEFAULT_GATEWAY_URL = em_runner.DEFAULT_GATEWAY_URL
DEFAULT_JUDGE_GATEWAY_MODEL = "codex_sdk/gpt-5.6-sol"
DEFAULT_JUDGE_CALLER_MODEL = "openai/codex_sdk/gpt-5.6-sol"
LOCKED_ANSWER_GATEWAY_MODEL = em_runner.DEFAULT_MODEL
JUDGE_MAX_PROMPT_TOKENS = 8_000
DEFAULT_EXPECTED_QUESTION_COUNT = em_runner.DEFAULT_EXPECTED_QUESTION_COUNT

def _judge_policy(
    memory_policy: EMFactPolicy,
    arms: Sequence[EMFactArm],
) -> dict[str, Any]:
    return {
        "format": JUDGE_POLICY_FORMAT,
        "protocol": "binary-semantic-equivalence-v1",
        "judge_prompt_builder": "memory_condense.eval.benchmark.build_judge_prompt",
        "verdict_parser": (
            "memory_condense.eval._binary_judge_protocol.parse_binary_judge_verdict"
        ),
        "answer_model": LOCKED_ANSWER_GATEWAY_MODEL,
        "judge_model": DEFAULT_JUDGE_CALLER_MODEL,
        "memory_policy": memory_policy,
        "matched_arms": list(arms),
        "arm_labels_exposed_to_judge": False,
        "explicit_gold_answer_field_persisted": False,
        "judge_completions_may_echo_gold": True,
        "prompt_token_proxy_cap": JUDGE_MAX_PROMPT_TOKENS,
        "completion_token_cap": JUDGE_MAX_TOKENS,
        "retries": 0,
    }


JUDGE_POLICY = _judge_policy("v1", EM_FACT_ARMS)
JUDGE_POLICY_SHA256 = identity_sha256(JUDGE_POLICY)

_MANIFEST_FIELDS = frozenset(
    {
        "format",
        "mode",
        "campaign_binding",
        "judge_prompt_population",
        "completion_batch",
        "question_count",
        "logical_judgment_count",
        "unique_judge_completion_count",
        "judgments",
        "arm_aggregates",
        "gold_loaded_post_upstream_verification",
        "explicit_gold_answer_field_persisted",
        "judge_completions_may_echo_gold",
        "arm_labels_exposed_to_judge",
        "zero_state",
    }
)
_RECORD_DISPOSITION_FIELDS = frozenset({"checkpoint_hit", "physical_call"})
_USAGE_DISPOSITION_FIELDS = frozenset({"physical_calls", "checkpoint_hits"})
_FORBIDDEN_SECRET_FIELDS = frozenset(
    {"api_key", "api-key", "authorization", "litellm_key"}
)


@dataclass(frozen=True, slots=True)
class _VerifiedUpstream:
    artifact: FastRetrievalArtifact
    run: Mapping[str, Any]
    run_sha256: str
    memory_policy: EMFactPolicy
    arms: tuple[EMFactArm, ...]
    predictions: tuple[Mapping[str, Any], ...]
    answer_response_journal_sha256s: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PlannedJudgment:
    logical_ordinal: int
    question_ordinal: int
    question_id: str
    arm: str
    category: str
    question_sha256: str
    dated_question_sha256: str
    gold_answer_sha256: str
    prediction_sha256: str
    answer_response_journal_sha256: str
    messages: tuple[dict[str, str], ...]
    messages_sha256: str


@dataclass(frozen=True, slots=True)
class _JudgePlan:
    upstream: _VerifiedUpstream
    rows: tuple[_PlannedJudgment, ...]
    preflight: FastPromptPopulation
    gold_population_sha256: str


def _upstream_run_path(args: argparse.Namespace) -> Path:
    return Path(args.run_artifact or Path(args.upstream_root) / "run.json")


def _judgments_path(args: argparse.Namespace) -> Path:
    return Path(
        args.judgments
        or Path(args.output_root) / "em-facts-semantic-judge-sol.json"
    )


def _judge_replay_path(args: argparse.Namespace) -> Path:
    return Path(
        args.judge_replay
        or Path(args.output_root) / "em-facts-semantic-judge-sol-replay.json"
    )


def _judge_checkpoints_path(args: argparse.Namespace) -> Path:
    return Path(args.output_root) / "em-facts-semantic-judge-sol-calls"


def _upstream_args(
    args: argparse.Namespace,
    *,
    memory_policy: EMFactPolicy,
    arms: Sequence[EMFactArm],
) -> argparse.Namespace:
    """Project judge CLI arguments onto the upstream replay interface."""

    return argparse.Namespace(
        retrieval=Path(args.retrieval),
        expected_retrieval_sha256=str(args.expected_retrieval_sha256),
        expected_question_count=args.expected_question_count,
        source_stage_id=str(args.source_stage_id),
        memory_policy=memory_policy,
        answer_arms=list(arms),
        run_artifact=_upstream_run_path(args),
        output_root=Path(args.upstream_root),
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.gateway_url != DEFAULT_GATEWAY_URL:
        raise ValueError("semantic judge requires the locked central-dev gateway")
    if args.gateway_model != DEFAULT_JUDGE_GATEWAY_MODEL or (
        args.caller_model != DEFAULT_JUDGE_CALLER_MODEL
    ):
        raise ValueError("semantic judge requires the locked Sol route")
    if args.source_stage_id != DEFAULT_EM_STAGE_ID:
        raise ValueError("semantic judge requires the sealed S1 EM source stage")
    if type(args.max_concurrency) is not int or args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be a positive integer")
    if type(args.expected_question_count) is not int or (
        args.expected_question_count < 1
    ):
        raise ValueError("--expected-question-count must be positive")
    if args.dataset is None:
        raise ValueError("semantic judging requires --dataset")


def _load_verified_upstream(args: argparse.Namespace) -> _VerifiedUpstream:
    """Verify the EM run plus compression and answer journals without gold."""

    _validate_args(args)
    run, run_sha256 = em_runner._read(_upstream_run_path(args))
    settings = run.get("settings")
    if (
        not isinstance(settings, Mapping)
        or settings.get("gateway_url") != DEFAULT_GATEWAY_URL
        or settings.get("model") != LOCKED_ANSWER_GATEWAY_MODEL
        or settings.get("stage_id") != DEFAULT_EM_STAGE_ID
    ):
        raise ValueError("upstream answers are not the locked Terra EM-fact run")
    memory_policy, arms = em_runner._stored_policy_and_arms(settings)
    upstream_args = _upstream_args(
        args,
        memory_policy=memory_policy,
        arms=arms,
    )
    artifact = em_runner._load(upstream_args)
    predictions = tuple(
        em_runner._verified_predictions(upstream_args, artifact, run)
    )
    expected_order = tuple(
        (question.question_id, arm)
        for question in artifact.questions
        for arm in arms
    )
    observed_order = tuple(
        (str(row["question_id"]), str(row["arm"])) for row in predictions
    )
    if observed_order != expected_order:
        raise ValueError("upstream predictions changed matched-arm order")
    answer_runtime = run.get("answers", {}).get("runtime", {})
    journals = tuple(answer_runtime.get("response_journal_sha256s", ()))
    if len(journals) != len(predictions):
        raise ValueError("upstream answer journal population changed")
    return _VerifiedUpstream(
        artifact=artifact,
        run=run,
        run_sha256=run_sha256,
        memory_policy=memory_policy,
        arms=arms,
        predictions=predictions,
        answer_response_journal_sha256s=journals,
    )


def _load_gold_population(dataset: Path, split: Path) -> Any:
    from memory_condense.eval.recall_guarded_cumulative_1m import (
        load_original_population,
    )

    return load_original_population(dataset, split)


def _plan_judgments(
    upstream: _VerifiedUpstream,
    gold_population: Any,
) -> _JudgePlan:
    retrieval_questions = upstream.artifact.questions
    gold_questions = tuple(gold_population.questions)
    if len(gold_questions) != len(retrieval_questions):
        raise ValueError("gold and verified answer populations differ")
    for retrieval, gold in zip(retrieval_questions, gold_questions, strict=True):
        if (
            gold.question_id != retrieval.question_id
            or quote_sha256(gold.question) != retrieval.question_sha256
            or quote_sha256(gold.dated_question) != retrieval.dated_question_sha256
        ):
            raise ValueError("gold question order or exact text changed")

    gold_projection = [
        {
            "question_id": gold.question_id,
            "question_sha256": quote_sha256(gold.question),
            "dated_question_sha256": quote_sha256(gold.dated_question),
            "gold_answer_sha256": quote_sha256(gold.answer),
            "category": str(gold.category or "unknown"),
        }
        for gold in gold_questions
    ]
    gold_by_id = {gold.question_id: gold for gold in gold_questions}
    question_ordinals = {
        question.question_id: question.ordinal for question in retrieval_questions
    }
    planned: list[_PlannedJudgment] = []
    for logical_ordinal, (source, answer_journal) in enumerate(
        zip(
            upstream.predictions,
            upstream.answer_response_journal_sha256s,
            strict=True,
        )
    ):
        question_id = str(source["question_id"])
        gold = gold_by_id[question_id]
        prediction = str(source["completion"])
        # Deliberately pass no arm metadata to the established judge builder.
        messages = tuple(
            dict(message)
            for message in build_judge_prompt(
                gold.question,
                gold.answer,
                prediction,
            )
        )
        planned.append(
            _PlannedJudgment(
                logical_ordinal=logical_ordinal,
                question_ordinal=question_ordinals[question_id],
                question_id=question_id,
                arm=str(source["arm"]),
                category=str(gold.category or "unknown"),
                question_sha256=quote_sha256(gold.question),
                dated_question_sha256=quote_sha256(gold.dated_question),
                gold_answer_sha256=quote_sha256(gold.answer),
                prediction_sha256=quote_sha256(prediction),
                answer_response_journal_sha256=str(answer_journal),
                messages=messages,
                messages_sha256=identity_sha256(list(messages)),
            )
        )
    preflight = preflight_fast_completion_prompts(
        [row.messages for row in planned],
        max_prompt_tokens=JUDGE_MAX_PROMPT_TOKENS,
    )
    if tuple(row.messages_sha256 for row in planned) != tuple(
        row.messages_sha256 for row in preflight.ordered_rows
    ):
        raise RuntimeError("judge preflight changed the planned prompt population")
    return _JudgePlan(
        upstream=upstream,
        rows=tuple(planned),
        preflight=preflight,
        gold_population_sha256=identity_sha256(gold_projection),
    )


def _load_plan(args: argparse.Namespace) -> _JudgePlan:
    upstream = _load_verified_upstream(args)
    # Gold is first reachable after run provenance and both journal sets replay.
    gold = _load_gold_population(Path(args.dataset), Path(args.split))
    return _plan_judgments(upstream, gold)


def _campaign_binding(plan: _JudgePlan) -> dict[str, Any]:
    run = plan.upstream.run
    compression = run["compression"]
    answers = run["answers"]["runtime"]
    judge_policy = _judge_policy(
        plan.upstream.memory_policy,
        plan.upstream.arms,
    )
    return {
        "format": JUDGE_BINDING_FORMAT,
        "upstream_run_sha256": plan.upstream.run_sha256,
        "retrieval_binding": run["retrieval_binding"],
        "compression_runtime_identity_sha256": compression[
            "runtime_identity_sha256"
        ],
        "compression_journal_population_sha256": identity_sha256(
            compression["response_journal_sha256s"]
        ),
        "answer_runtime_identity_sha256": answers["runtime_identity_sha256"],
        "answer_journal_population_sha256": identity_sha256(
            answers["response_journal_sha256s"]
        ),
        "prediction_population_sha256": identity_sha256(
            [
                {
                    "question_id": row.question_id,
                    "arm": row.arm,
                    "prediction_sha256": row.prediction_sha256,
                    "answer_response_journal_sha256": (
                        row.answer_response_journal_sha256
                    ),
                }
                for row in plan.rows
            ]
        ),
        "gold_population_sha256": plan.gold_population_sha256,
        "judge_prompt_population_sha256": plan.preflight.prompt_population_sha256,
        "judge_policy_sha256": identity_sha256(judge_policy),
        "answer_model": LOCKED_ANSWER_GATEWAY_MODEL,
        "judge_model": DEFAULT_JUDGE_CALLER_MODEL,
        "memory_policy": plan.upstream.memory_policy,
        "question_count": plan.upstream.artifact.question_count,
        "logical_judgment_count": len(plan.rows),
        "unique_judge_call_count": plan.preflight.unique_prompt_count,
        "arms": list(plan.upstream.arms),
        "gold_loaded_post_upstream_verification": True,
        "arm_labels_exposed_to_judge": False,
        "retained_request_token_state_bytes": 0,
    }


def _benchmark_provenance(
    plan: _JudgePlan,
    *,
    max_concurrency: int,
) -> dict[str, Any]:
    binding = _campaign_binding(plan)
    return {
        "format": JUDGE_BINDING_FORMAT,
        "campaign_binding_sha256": identity_sha256(binding),
        "upstream_run_sha256": plan.upstream.run_sha256,
        "judge_prompt_population_sha256": plan.preflight.prompt_population_sha256,
        "gold_population_sha256": plan.gold_population_sha256,
        "memory_policy": plan.upstream.memory_policy,
        "answer_arms": list(plan.upstream.arms),
        "judge_policy_sha256": binding["judge_policy_sha256"],
        "gateway_url": DEFAULT_GATEWAY_URL,
        "gateway_model": DEFAULT_JUDGE_GATEWAY_MODEL,
        "caller_model_alias": DEFAULT_JUDGE_CALLER_MODEL,
        "authorized_unique_calls": plan.preflight.unique_prompt_count,
        "logical_prompt_count": plan.preflight.logical_prompt_count,
        "arm_labels_exposed_to_judge": False,
        "max_prompt_tokens": JUDGE_MAX_PROMPT_TOKENS,
        "max_completion_tokens": JUDGE_MAX_TOKENS,
        "max_concurrency": max_concurrency,
        "retries": 0,
        "retained_request_token_state_bytes": 0,
    }


def _runtime(
    plan: _JudgePlan,
    args: argparse.Namespace,
    *,
    client: Any | None,
) -> FastCompletionRuntime:
    return FastCompletionRuntime(
        checkpoint_dir=_judge_checkpoints_path(args),
        prompt_population=[row.messages for row in plan.rows],
        model=DEFAULT_JUDGE_GATEWAY_MODEL,
        client=client,
        max_prompt_tokens=JUDGE_MAX_PROMPT_TOKENS,
        max_new_tokens=JUDGE_MAX_TOKENS,
        max_concurrency=args.max_concurrency,
        retries=0,
        benchmark_provenance=_benchmark_provenance(
            plan,
            max_concurrency=args.max_concurrency,
        ),
    )


def _judgment_rows(
    plan: _JudgePlan,
    batch: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = {
        row["messages_sha256"]: row for row in batch["unique_records"]
    }
    results: list[dict[str, Any]] = []
    for source, verdict_text in zip(
        plan.rows,
        batch["logical_completions"],
        strict=True,
    ):
        record = records[source.messages_sha256]
        results.append(
            {
                "logical_ordinal": source.logical_ordinal,
                "question_ordinal": source.question_ordinal,
                "question_id": source.question_id,
                "arm": source.arm,
                "category": source.category,
                "question_sha256": source.question_sha256,
                "dated_question_sha256": source.dated_question_sha256,
                "gold_answer_sha256": source.gold_answer_sha256,
                "prediction_sha256": source.prediction_sha256,
                "answer_response_journal_sha256": (
                    source.answer_response_journal_sha256
                ),
                "judge_messages_sha256": source.messages_sha256,
                "judge_call_key_sha256": record["call_key_sha256"],
                "judge_request_journal_sha256": record[
                    "request_journal_sha256"
                ],
                "judge_response_journal_sha256": record[
                    "response_journal_sha256"
                ],
                "verdict_sha256": quote_sha256(verdict_text),
                "correct": parse_binary_judge_verdict(verdict_text),
            }
        )
    return results


def _arm_aggregates(
    rows: Sequence[Mapping[str, Any]],
    *,
    question_count: int,
    arms: Sequence[EMFactArm],
) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for arm in arms:
        selected = [row for row in rows if row["arm"] == arm]
        if len(selected) != question_count:
            raise ValueError(f"judge result omitted matched arm {arm}")
        correct = sum(row["correct"] is True for row in selected)
        aggregates.append(
            {
                "arm": arm,
                "questions": question_count,
                "correct": correct,
                "accuracy": correct / question_count,
            }
        )
    return aggregates


def _contains_secret(value: object) -> bool:
    if type(value) is dict:
        return any(
            str(key).casefold() in _FORBIDDEN_SECRET_FIELDS
            or _contains_secret(item)
            for key, item in value.items()
        )
    if type(value) is list:
        return any(_contains_secret(item) for item in value)
    return False


def _stable_batch_projection(batch: Mapping[str, Any]) -> dict[str, Any]:
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


def _judge_artifact(
    *,
    mode: str,
    plan: _JudgePlan,
    batch: FastCompletionBatch,
) -> dict[str, Any]:
    if mode not in {"run", "replay"}:
        raise ValueError("judge artifact mode must be run or replay")
    batch_dump = batch.model_dump()
    rows = _judgment_rows(plan, batch_dump)
    questions = plan.upstream.artifact.question_count
    result = {
        "format": JUDGE_MANIFEST_FORMAT,
        "mode": mode,
        "campaign_binding": _campaign_binding(plan),
        "judge_prompt_population": plan.preflight.model_dump(),
        "completion_batch": batch_dump,
        "question_count": questions,
        "logical_judgment_count": len(rows),
        "unique_judge_completion_count": len(batch.unique_records),
        "judgments": rows,
        "arm_aggregates": _arm_aggregates(
            rows,
            question_count=questions,
            arms=plan.upstream.arms,
        ),
        "gold_loaded_post_upstream_verification": True,
        "explicit_gold_answer_field_persisted": False,
        "judge_completions_may_echo_gold": True,
        "arm_labels_exposed_to_judge": False,
        "zero_state": {
            "contract": ZERO_STATE_CONTRACT,
            "persisted_transformer_token_state": False,
            "retained_transformer_token_state_bytes": 0,
            "external_provider_persistence_certified": False,
        },
    }
    if _contains_secret(result):
        raise ValueError("judge artifact serialized provider credentials")
    return result


def _validate_judge_artifact(
    payload: Mapping[str, Any],
    *,
    expected_mode: str,
    plan: _JudgePlan,
    journal_batch: FastCompletionBatch,
) -> None:
    if type(payload) is not dict or set(payload) != _MANIFEST_FIELDS:
        raise ValueError("judge manifest has a noncanonical shape")
    if (
        payload["format"] != JUDGE_MANIFEST_FORMAT
        or payload["mode"] != expected_mode
        or payload["campaign_binding"] != _campaign_binding(plan)
        or payload["judge_prompt_population"] != plan.preflight.model_dump()
        or payload["gold_loaded_post_upstream_verification"] is not True
        or payload["explicit_gold_answer_field_persisted"] is not False
        or payload["judge_completions_may_echo_gold"] is not True
        or payload["arm_labels_exposed_to_judge"] is not False
    ):
        raise ValueError("judge manifest changed sealed campaign provenance")
    expected = _judge_artifact(
        mode=expected_mode,
        plan=plan,
        batch=journal_batch,
    )
    for field in (
        "question_count",
        "logical_judgment_count",
        "unique_judge_completion_count",
        "judgments",
        "arm_aggregates",
        "zero_state",
    ):
        if payload[field] != expected[field]:
            raise ValueError(f"judge manifest changed verified {field}")
    if _stable_batch_projection(payload["completion_batch"]) != (
        _stable_batch_projection(journal_batch.model_dump())
    ):
        raise ValueError("judge manifest differs from immutable Sol journals")
    usage = payload["completion_batch"]["usage"]
    records = payload["completion_batch"]["unique_records"]
    physical_records = sum(row["physical_call"] is True for row in records)
    checkpoint_records = sum(row["checkpoint_hit"] is True for row in records)
    if (
        usage["physical_calls"] + usage["checkpoint_hits"] != len(records)
        or usage["physical_calls"] != physical_records
        or usage["checkpoint_hits"] != checkpoint_records
        or any(row["physical_call"] == row["checkpoint_hit"] for row in records)
        or (expected_mode == "replay" and usage["physical_calls"] != 0)
        or _contains_secret(payload)
    ):
        raise ValueError("judge manifest changed call disposition or zero-state")


def _make_provider_client(api_key: str, gateway_url: str) -> Any:
    return em_runner._make_provider_client(api_key, gateway_url)


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.enable_provider or args.authorized_provider_calls != 0:
        raise ValueError("preflight forbids provider access and authorization")
    plan = _load_plan(args)
    return {
        "format": PREFLIGHT_FORMAT,
        "campaign_binding": _campaign_binding(plan),
        "logical_prompt_count": plan.preflight.logical_prompt_count,
        "unique_prompt_count": plan.preflight.unique_prompt_count,
        "required_authorized_provider_calls": plan.preflight.unique_prompt_count,
        "maximum_prompt_token_proxy": max(
            row.prompt_token_proxy for row in plan.preflight.ordered_rows
        ),
        "max_prompt_tokens": JUDGE_MAX_PROMPT_TOKENS,
        "max_completion_tokens": JUDGE_MAX_TOKENS,
        "provider_calls": 0,
        "writes": 0,
        "gold_loaded_post_upstream_verification": True,
        "explicit_gold_answer_field_persisted": False,
        "judge_completions_may_echo_gold": True,
        "arm_labels_exposed_to_judge": False,
        "retained_request_token_state_bytes": 0,
    }


def run_judge(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    plan = _load_plan(args)
    required = plan.preflight.unique_prompt_count
    if not args.enable_provider:
        raise ValueError("run phase requires the explicit --enable-provider gate")
    if args.authorized_provider_calls != required:
        raise ValueError(
            "--authorized-provider-calls must exactly equal the preflighted "
            f"unique judge population ({args.authorized_provider_calls} != {required})"
        )
    api_key = os.environ.get(str(args.api_key_env), "").strip()
    if not api_key:
        raise RuntimeError(f"provider API key is empty: {args.api_key_env}")
    client = _make_provider_client(api_key, DEFAULT_GATEWAY_URL)
    runtime = _runtime(plan, args, client=client)
    with runtime:
        batch = runtime.run()
    result = _judge_artifact(mode="run", plan=plan, batch=batch)
    return result, em_runner._publish(_judgments_path(args), result)


def run_replay(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.enable_provider or args.authorized_provider_calls != 0:
        raise ValueError("replay forbids provider access and authorization")
    plan = _load_plan(args)
    source, _source_sha = em_runner._read(_judgments_path(args))
    checkpoints = _judge_checkpoints_path(args)
    if checkpoints.is_symlink() or not checkpoints.is_dir():
        raise ValueError("Sol judge checkpoint directory is missing or invalid")
    runtime = _runtime(plan, args, client=None)
    with runtime:
        batch = runtime.run()
    _validate_judge_artifact(
        source,
        expected_mode="run",
        plan=plan,
        journal_batch=batch,
    )
    result = _judge_artifact(mode="replay", plan=plan, batch=batch)
    return result, em_runner._publish(_judge_replay_path(args), result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("preflight", "run", "replay"), default="preflight"
    )
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-artifact", type=Path)
    parser.add_argument("--judgments", type=Path)
    parser.add_argument("--judge-replay", type=Path)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument(
        "--expected-retrieval-sha256", default=ORIGINAL_1M_RETRIEVAL_SHA256
    )
    parser.add_argument("--source-stage-id", default=DEFAULT_EM_STAGE_ID)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--expected-question-count",
        type=int,
        default=DEFAULT_EXPECTED_QUESTION_COUNT,
    )
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument("--gateway-model", default=DEFAULT_JUDGE_GATEWAY_MODEL)
    parser.add_argument("--caller-model", default=DEFAULT_JUDGE_CALLER_MODEL)
    parser.add_argument("--api-key-env", default="LITELLM_KEY")
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--enable-provider", action="store_true")
    parser.add_argument("--authorized-provider-calls", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.phase == "preflight":
        result: Any = run_preflight(args)
        print(
            "EM-facts semantic-judge preflight passed: "
            f"questions={result['campaign_binding']['question_count']}; "
            f"logical={result['logical_prompt_count']}; "
            f"unique={result['unique_prompt_count']}; "
            f"max_prompt={result['maximum_prompt_token_proxy']}/"
            f"{result['max_prompt_tokens']}; provider_calls=0; writes=0",
            flush=True,
        )
        return 0
    if args.phase == "run":
        result, digest = run_judge(args)
    else:
        result, digest = run_replay(args)
    usage = result["completion_batch"]["usage"]
    aggregates = ", ".join(
        f"{row['arm']}={row['correct']}/{row['questions']}"
        for row in result["arm_aggregates"]
    )
    print(
        f"EM-facts semantic judge {args.phase} published ({digest}): "
        f"{aggregates}; logical={result['logical_judgment_count']}; "
        f"unique={result['unique_judge_completion_count']}; "
        f"physical={usage['physical_calls']}; "
        f"checkpoint_hits={usage['checkpoint_hits']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_UPSTREAM_ROOT",
    "JUDGE_MANIFEST_FORMAT",
    "JUDGE_POLICY_SHA256",
    "PREFLIGHT_FORMAT",
    "build_parser",
    "main",
    "run_judge",
    "run_preflight",
    "run_replay",
]
