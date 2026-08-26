"""Run an independent Sol semantic judge over sealed CAV-link answers.

All answer, replay, and Terra completion journals are verified before benchmark
gold is loaded.  The full paired-arm judge population is then preflighted and
deduplicated before any Sol client exists.  ``replay`` reopens only immutable
judge journals and never receives a provider client.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval import run_fast_1m_cav_link_synthesis as link_runner
from memory_condense.eval._binary_judge_protocol import (
    JUDGE_MAX_TOKENS,
    parse_binary_judge_verdict,
)
from memory_condense.eval.benchmark import build_judge_prompt
from memory_condense.eval.fast_cav_link_synthesis import (
    FAST_CAV_LINK_SYNTHESIS_ARM_IDS,
)
from memory_condense.eval.fast_completion_runtime import (
    FastCompletionBatch,
    FastCompletionRuntime,
    FastPromptPopulation,
    preflight_fast_completion_prompts,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    ORIGINAL_1M_RETRIEVAL_SHA256,
)


PREFLIGHT_FORMAT = "memory-condense-fast-1m-cav-link-sol-judge-preflight-v1"
JUDGE_MANIFEST_FORMAT = "memory-condense-fast-1m-cav-link-sol-judge-v1"
JUDGE_BINDING_FORMAT = "memory-condense-fast-1m-cav-link-sol-judge-binding-v1"
JUDGE_POLICY_FORMAT = "memory-condense-fast-1m-cav-link-sol-judge-policy-v1"
ZERO_STATE_CONTRACT = "stateless-fast-1m-cav-link-sol-judge-boundary-v1"

DEFAULT_UPSTREAM_ROOT = Path(
    "eval_results/longmemeval-1m-fast-cav-link-synthesis-development-"
    "network-authorized-20260823"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_UPSTREAM_ROOT
DEFAULT_RETRIEVAL = link_runner.DEFAULT_RETRIEVAL
DEFAULT_FEATURES = link_runner.DEFAULT_FEATURES
DEFAULT_FEATURES_SHA256 = link_runner.DEFAULT_FEATURES_SHA256
DEFAULT_SPLIT = link_runner.DEFAULT_SPLIT
DEFAULT_GATEWAY_URL = link_runner.DEFAULT_GATEWAY_URL
DEFAULT_JUDGE_GATEWAY_MODEL = "codex_sdk/gpt-5.6-sol"
DEFAULT_JUDGE_CALLER_MODEL = "openai/codex_sdk/gpt-5.6-sol"
LOCKED_ANSWER_GATEWAY_MODEL = "codex_sdk/gpt-5.6-terra"
LOCKED_ANSWER_CALLER_MODEL = "openai/codex_sdk/gpt-5.6-terra"
JUDGE_MAX_PROMPT_TOKENS = 8_000
DEFAULT_EXPECTED_QUESTION_COUNT = 10

JUDGE_POLICY = {
    "format": JUDGE_POLICY_FORMAT,
    "protocol": "binary-semantic-equivalence-v1",
    "judge_prompt_builder": "memory_condense.eval.benchmark.build_judge_prompt",
    "verdict_parser": (
        "memory_condense.eval._binary_judge_protocol.parse_binary_judge_verdict"
    ),
    "answer_model": LOCKED_ANSWER_CALLER_MODEL,
    "judge_model": DEFAULT_JUDGE_CALLER_MODEL,
    "matched_arms": list(FAST_CAV_LINK_SYNTHESIS_ARM_IDS),
    "prompt_token_proxy_cap": JUDGE_MAX_PROMPT_TOKENS,
    "completion_token_cap": JUDGE_MAX_TOKENS,
    "retries": 0,
}
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
        "paired_verdicts",
        "pair_summary",
        "gold_loaded_post_upstream_verification",
        "gold_answer_text_persisted",
        "zero_state",
    }
)


@dataclass(frozen=True, slots=True)
class _VerifiedUpstream:
    experiment: Any
    answers: Mapping[str, Any]
    answer_sha256: str
    replay: Mapping[str, Any]
    replay_sha256: str


@dataclass(frozen=True, slots=True)
class _PlannedJudgment:
    logical_ordinal: int
    question_ordinal: int
    question_id: str
    arm_id: str
    link_exposed: bool
    category: str
    question_sha256: str
    dated_question_sha256: str
    gold_answer_sha256: str
    prediction_sha256: str
    answer_response_sha256: str
    answer_call_key_sha256: str
    answer_request_journal_sha256: str
    answer_response_journal_sha256: str
    messages: tuple[dict[str, str], ...]
    messages_sha256: str


@dataclass(frozen=True, slots=True)
class _JudgePlan:
    upstream: _VerifiedUpstream
    rows: tuple[_PlannedJudgment, ...]
    preflight: FastPromptPopulation
    gold_population_sha256: str


def _answers_path(args: argparse.Namespace) -> Path:
    return Path(args.answers or Path(args.upstream_root) / "answers.json")


def _answer_replay_path(args: argparse.Namespace) -> Path:
    return Path(args.answer_replay or Path(args.upstream_root) / "replay.json")


def _answer_checkpoints_path(args: argparse.Namespace) -> Path:
    return Path(
        args.answer_checkpoints or Path(args.upstream_root) / "completion-calls"
    )


def _judgments_path(args: argparse.Namespace) -> Path:
    return Path(
        args.judgments
        or Path(args.output_root) / "cav-link-semantic-judge-sol.json"
    )


def _judge_replay_path(args: argparse.Namespace) -> Path:
    return Path(
        args.judge_replay
        or Path(args.output_root) / "cav-link-semantic-judge-sol-replay.json"
    )


def _judge_checkpoints_path(args: argparse.Namespace) -> Path:
    return Path(args.output_root) / "cav-link-semantic-judge-sol-calls"


def _validate_args(args: argparse.Namespace) -> None:
    if args.gateway_url != DEFAULT_GATEWAY_URL:
        raise ValueError("semantic judge requires the locked central-dev gateway")
    if args.gateway_model != DEFAULT_JUDGE_GATEWAY_MODEL or (
        args.caller_model != DEFAULT_JUDGE_CALLER_MODEL
    ):
        raise ValueError("semantic judge requires the locked Sol route")
    if type(args.max_concurrency) is not int or args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be a positive integer")
    if type(args.expected_question_count) is not int or (
        args.expected_question_count < 1
    ):
        raise ValueError("--expected-question-count must be positive")
    if args.dataset is None:
        raise ValueError("semantic judging requires --dataset")


def _load_verified_upstream(args: argparse.Namespace) -> _VerifiedUpstream:
    """Verify answer, replay, and Terra journals without touching gold."""

    _validate_args(args)
    experiment = link_runner._load_experiment(args)
    answers, answer_sha = link_runner._read_and_validate_answers(
        experiment,
        _answers_path(args),
        expected_mode="answer",
    )
    replay, replay_sha = link_runner._read_and_validate_answers(
        experiment,
        _answer_replay_path(args),
        expected_mode="replay",
    )
    link_runner._validate_answer_replay_pair(answers, replay)
    checkpoints = _answer_checkpoints_path(args)
    if checkpoints.is_symlink() or not checkpoints.is_dir():
        raise ValueError("Terra answer checkpoint directory is missing or invalid")
    journal_replay = link_runner._replay_journals(
        experiment,
        answers,
        checkpoints,
    )
    if replay["completion_batch"] != journal_replay.model_dump():
        raise ValueError("answer replay differs from immutable Terra journals")
    provenance = answers["completion_batch"]["provenance"]
    benchmark = provenance["benchmark_provenance"]
    if (
        provenance["model"] != LOCKED_ANSWER_GATEWAY_MODEL
        or provenance["retries"] != 0
        or benchmark.get("caller_model_alias") != LOCKED_ANSWER_CALLER_MODEL
        or benchmark.get("gateway_model") != LOCKED_ANSWER_GATEWAY_MODEL
        or benchmark.get("gateway_url") != DEFAULT_GATEWAY_URL
    ):
        raise ValueError("upstream answers are not the locked zero-retry Terra run")
    return _VerifiedUpstream(
        experiment=experiment,
        answers=answers,
        answer_sha256=answer_sha,
        replay=replay,
        replay_sha256=replay_sha,
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
    retrieval_questions = upstream.experiment.retrieval.questions
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

    answer_records = {
        row["messages_sha256"]: row
        for row in upstream.answers["completion_batch"]["unique_records"]
    }
    planned: list[_PlannedJudgment] = []
    gold_projection: list[dict[str, Any]] = []
    for gold in gold_questions:
        gold_projection.append(
            {
                "question_id": gold.question_id,
                "question_sha256": quote_sha256(gold.question),
                "dated_question_sha256": quote_sha256(gold.dated_question),
                "gold_answer_sha256": quote_sha256(gold.answer),
                "category": str(gold.category or "unknown"),
            }
        )
    for source in upstream.answers["answers"]:
        gold = gold_questions[source["question_ordinal"]]
        prediction = source["parsed_response"]["answer"]
        messages = tuple(
            dict(message)
            for message in build_judge_prompt(
                gold.question,
                gold.answer,
                prediction,
            )
        )
        answer_record = answer_records[source["messages_sha256"]]
        planned.append(
            _PlannedJudgment(
                logical_ordinal=source["logical_ordinal"],
                question_ordinal=source["question_ordinal"],
                question_id=source["question_id"],
                arm_id=source["arm_id"],
                link_exposed=source["link_exposed"],
                category=str(gold.category or "unknown"),
                question_sha256=quote_sha256(gold.question),
                dated_question_sha256=quote_sha256(gold.dated_question),
                gold_answer_sha256=quote_sha256(gold.answer),
                prediction_sha256=quote_sha256(prediction),
                answer_response_sha256=source["parsed_response"]["response_sha256"],
                answer_call_key_sha256=answer_record["call_key_sha256"],
                answer_request_journal_sha256=(
                    answer_record["request_journal_sha256"]
                ),
                answer_response_journal_sha256=(
                    answer_record["response_journal_sha256"]
                ),
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
    # Gold is first reachable here, after answer + replay + journals verify.
    gold = _load_gold_population(Path(args.dataset), Path(args.split))
    return _plan_judgments(upstream, gold)


def _campaign_binding(plan: _JudgePlan) -> dict[str, Any]:
    experiment = plan.upstream.experiment
    answer_batch = plan.upstream.answers["completion_batch"]
    return {
        "format": JUDGE_BINDING_FORMAT,
        "answer_manifest_sha256": plan.upstream.answer_sha256,
        "answer_replay_manifest_sha256": plan.upstream.replay_sha256,
        "retrieval_sha256": experiment.retrieval.raw_sha256,
        "feature_manifest_sha256": experiment.feature.raw_sha256,
        "synthesis_population_sha256": experiment.prompts.population_sha256,
        "answer_runtime_identity_sha256": answer_batch["runtime_identity_sha256"],
        "answer_journal_population_sha256": identity_sha256(
            [
                {
                    "call_key_sha256": row["call_key_sha256"],
                    "request_journal_sha256": row["request_journal_sha256"],
                    "response_journal_sha256": row["response_journal_sha256"],
                    "messages_sha256": row["messages_sha256"],
                    "completion_sha256": row["completion_sha256"],
                }
                for row in answer_batch["unique_records"]
            ]
        ),
        "gold_population_sha256": plan.gold_population_sha256,
        "judge_prompt_population_sha256": plan.preflight.prompt_population_sha256,
        "judge_policy_sha256": JUDGE_POLICY_SHA256,
        "answer_model": LOCKED_ANSWER_CALLER_MODEL,
        "judge_model": DEFAULT_JUDGE_CALLER_MODEL,
        "question_count": experiment.retrieval.question_count,
        "logical_judgment_count": len(plan.rows),
        "unique_judge_call_count": plan.preflight.unique_prompt_count,
        "arm_ids": list(FAST_CAV_LINK_SYNTHESIS_ARM_IDS),
        "gold_loaded_post_upstream_verification": True,
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
        "answer_manifest_sha256": plan.upstream.answer_sha256,
        "answer_replay_manifest_sha256": plan.upstream.replay_sha256,
        "judge_prompt_population_sha256": plan.preflight.prompt_population_sha256,
        "gold_population_sha256": plan.gold_population_sha256,
        "gateway_url": DEFAULT_GATEWAY_URL,
        "gateway_model": DEFAULT_JUDGE_GATEWAY_MODEL,
        "caller_model_alias": DEFAULT_JUDGE_CALLER_MODEL,
        "authorized_unique_calls": plan.preflight.unique_prompt_count,
        "logical_prompt_count": plan.preflight.logical_prompt_count,
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
                "arm_id": source.arm_id,
                "link_exposed": source.link_exposed,
                "category": source.category,
                "question_sha256": source.question_sha256,
                "dated_question_sha256": source.dated_question_sha256,
                "gold_answer_sha256": source.gold_answer_sha256,
                "prediction_sha256": source.prediction_sha256,
                "answer_response_sha256": source.answer_response_sha256,
                "answer_call_key_sha256": source.answer_call_key_sha256,
                "answer_request_journal_sha256": (
                    source.answer_request_journal_sha256
                ),
                "answer_response_journal_sha256": (
                    source.answer_response_journal_sha256
                ),
                "judge_messages_sha256": source.messages_sha256,
                "judge_call_key_sha256": record["call_key_sha256"],
                "judge_request_journal_sha256": record["request_journal_sha256"],
                "judge_response_journal_sha256": (
                    record["response_journal_sha256"]
                ),
                "verdict_sha256": quote_sha256(verdict_text),
                "correct": parse_binary_judge_verdict(verdict_text),
            }
        )
    return results


def _paired_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    question_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for arm_id in FAST_CAV_LINK_SYNTHESIS_ARM_IDS:
        selected = [row for row in rows if row["arm_id"] == arm_id]
        if len(selected) != question_count:
            raise ValueError(f"judge result omitted matched arm {arm_id}")
        correct = sum(row["correct"] is True for row in selected)
        aggregates.append(
            {
                "arm_id": arm_id,
                "questions": question_count,
                "correct": correct,
                "accuracy": correct / question_count,
            }
        )
    pairs: list[dict[str, Any]] = []
    outcomes = {
        "both_correct": 0,
        "both_incorrect": 0,
        "linked_only_correct": 0,
        "unlinked_only_correct": 0,
    }
    for question_ordinal in range(question_count):
        selected = [
            row for row in rows if row["question_ordinal"] == question_ordinal
        ]
        if [row["arm_id"] for row in selected] != list(
            FAST_CAV_LINK_SYNTHESIS_ARM_IDS
        ):
            raise ValueError("judge results changed paired-arm order")
        unlinked, linked = selected
        if unlinked["correct"] and linked["correct"]:
            outcome = "both_correct"
        elif not unlinked["correct"] and not linked["correct"]:
            outcome = "both_incorrect"
        elif linked["correct"]:
            outcome = "linked_only_correct"
        else:
            outcome = "unlinked_only_correct"
        outcomes[outcome] += 1
        pairs.append(
            {
                "question_ordinal": question_ordinal,
                "question_id": unlinked["question_id"],
                "unlinked_correct": unlinked["correct"],
                "linked_correct": linked["correct"],
                "outcome": outcome,
            }
        )
    summary = {
        "questions": question_count,
        **outcomes,
        "net_linked_correct_gain": (
            outcomes["linked_only_correct"] - outcomes["unlinked_only_correct"]
        ),
    }
    return aggregates, pairs, summary


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
    questions = plan.upstream.experiment.retrieval.question_count
    aggregates, pairs, summary = _paired_metrics(rows, question_count=questions)
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
        "arm_aggregates": aggregates,
        "paired_verdicts": pairs,
        "pair_summary": summary,
        "gold_loaded_post_upstream_verification": True,
        "gold_answer_text_persisted": False,
        "zero_state": {
            "contract": ZERO_STATE_CONTRACT,
            "persisted_transformer_token_state": False,
            "retained_transformer_token_state_bytes": 0,
            "external_provider_persistence_certified": False,
        },
    }
    if link_runner._contains_secret(result):
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
        or payload["gold_answer_text_persisted"] is not False
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
        "paired_verdicts",
        "pair_summary",
        "zero_state",
    ):
        if payload[field] != expected[field]:
            raise ValueError(f"judge manifest changed verified {field}")
    if link_runner._stable_batch_projection(payload["completion_batch"]) != (
        link_runner._stable_batch_projection(journal_batch.model_dump())
    ):
        raise ValueError("judge manifest differs from immutable Sol journals")
    usage = payload["completion_batch"]["usage"]
    records = payload["completion_batch"]["unique_records"]
    if (
        usage["physical_calls"] + usage["checkpoint_hits"] != len(records)
        or any(
            row["physical_call"] == row["checkpoint_hit"] for row in records
        )
        or (expected_mode == "replay" and usage["physical_calls"] != 0)
        or link_runner._contains_secret(payload)
    ):
        raise ValueError("judge manifest changed call disposition or zero-state")


def _make_provider_client(api_key: str, gateway_url: str) -> Any:
    return link_runner._make_provider_client(api_key, gateway_url)


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
    return result, link_runner._atomic_write_json(_judgments_path(args), result)


def run_replay(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.enable_provider or args.authorized_provider_calls != 0:
        raise ValueError("replay forbids provider access and authorization")
    plan = _load_plan(args)
    source, _source_sha = link_runner._read_canonical_json(_judgments_path(args))
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
    return result, link_runner._atomic_write_json(_judge_replay_path(args), result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("preflight", "run", "replay"), default="preflight"
    )
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--answers", type=Path)
    parser.add_argument("--answer-replay", type=Path)
    parser.add_argument("--answer-checkpoints", type=Path)
    parser.add_argument("--judgments", type=Path)
    parser.add_argument("--judge-replay", type=Path)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument(
        "--expected-retrieval-sha256", default=ORIGINAL_1M_RETRIEVAL_SHA256
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument(
        "--expected-features-sha256", default=DEFAULT_FEATURES_SHA256
    )
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
            "CAV-link semantic-judge preflight passed: "
            f"questions={result['campaign_binding']['question_count']}; "
            f"logical={result['logical_prompt_count']}; "
            f"unique={result['unique_prompt_count']}; "
            f"max_prompt={result['maximum_prompt_token_proxy']}/"
            f"{result['max_prompt_tokens']}; provider_calls=0; writes=0",
            flush=True,
        )
        return 0
    elif args.phase == "run":
        result, digest = run_judge(args)
    else:
        result, digest = run_replay(args)
    usage = result["completion_batch"]["usage"]
    aggregates = ", ".join(
        f"{row['arm_id']}={row['correct']}/{row['questions']}"
        for row in result["arm_aggregates"]
    )
    print(
        f"CAV-link semantic judge {args.phase} published ({digest}): "
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
