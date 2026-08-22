"""One-stage Sol judging for sealed fixed-stage Terra answer artifacts.

The answer artifact is validated against its retrieval input before this
module reads any gold answer.  Only the preregistered direct-episode stage is
eligible, and the complete judge-prompt population is planned and authorized
before the first provider call.  Historical multi-stage synthesis judging is
intentionally a separate protocol.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval._binary_judge_protocol import (
    JUDGE_MAX_TOKENS,
    parse_binary_judge_verdict,
)
from memory_condense.eval.benchmark import build_judge_prompt
from memory_condense.eval.recall_guarded_cumulative_final_answer import (
    FINAL_ANSWER_CAMPAIGN_FORMAT,
    FINAL_ANSWER_FORMAT,
    FINAL_ANSWER_RUNTIME_FORMAT,
    FIXED_STAGE_ID,
    LOCKED_RESPONDER_GATEWAY_MODEL,
    LOCKED_RESPONDER_MODEL,
    RESPONDER_OUTPUT_TOKEN_RESERVE,
    RESPONDER_PROMPT_POLICY_FORMAT,
    RESPONDER_PROMPT_CAP,
    validate_final_answer_artifact,
)
from memory_condense.eval.recall_guarded_cumulative_final_answer_runtime import (
    FINAL_ANSWER_REQUEST_JOURNAL_FORMAT,
    FINAL_ANSWER_RESPONSE_JOURNAL_FORMAT,
)
from memory_condense.eval.recall_guarded_cumulative_provider_synthesis_runtime import (
    CENTRAL_DEV_GATEWAY_URL,
    _gateway_model,
)
from memory_condense.eval.recall_guarded_cumulative_semantic_judge_runtime import (
    DEFAULT_JUDGE_MODEL,
    SEMANTIC_JUDGE_RUNTIME_FORMAT,
)
from memory_condense.eval.reproducibility import implementation_sha256
from memory_condense.ingest.loader import BenchmarkQuestion, BenchmarkSample


FINAL_ANSWER_SEMANTIC_JUDGE_FORMAT = (
    "memory-condense-fixed-stage-final-answer-semantic-judge-score-v1"
)
FINAL_ANSWER_SEMANTIC_JUDGE_CAMPAIGN_FORMAT = (
    "memory-condense-fixed-stage-final-answer-semantic-judge-campaign-v1"
)
LOCKED_JUDGE_MODEL = DEFAULT_JUDGE_MODEL
LOCKED_JUDGE_GATEWAY_MODEL = "codex_sdk/gpt-5.6-sol"
LOCKED_JUDGE_MAX_NEW_TOKENS = JUDGE_MAX_TOKENS
TARGET_ACCURACY = 0.95
MINIMUM_GATE_QUESTIONS = 100

FINAL_ANSWER_SEMANTIC_JUDGE_POLICY = {
    "format": "memory-condense-fixed-stage-semantic-judge-policy-v1",
    "answer_artifact_validator": (
        "memory_condense.eval.recall_guarded_cumulative_final_answer."
        "validate_final_answer_artifact"
    ),
    "judge_prompt_builder": "memory_condense.eval.benchmark.build_judge_prompt",
    "verdict_parser": (
        "memory_condense.eval._binary_judge_protocol."
        "parse_binary_judge_verdict"
    ),
    "question_form": "undated benchmark question",
    "fixed_stage_id": FIXED_STAGE_ID,
    "gate_unit": "one preregistered fixed retrieval stage",
    "target_accuracy": TARGET_ACCURACY,
    "minimum_questions": MINIMUM_GATE_QUESTIONS,
    "responder_model": LOCKED_RESPONDER_MODEL,
    "responder_gateway_model": LOCKED_RESPONDER_GATEWAY_MODEL,
    "responder_runtime_format": FINAL_ANSWER_RUNTIME_FORMAT,
    "responder_prompt_policy_binding": (
        "derived from sealed system messages and verified QA framing"
    ),
    "responder_max_prompt_token_proxy": RESPONDER_PROMPT_CAP,
    "responder_max_new_tokens": RESPONDER_OUTPUT_TOKEN_RESERVE,
    "judge_model": LOCKED_JUDGE_MODEL,
    "judge_gateway_model": LOCKED_JUDGE_GATEWAY_MODEL,
    "judge_runtime_format": SEMANTIC_JUDGE_RUNTIME_FORMAT,
    "judge_max_new_tokens": LOCKED_JUDGE_MAX_NEW_TOKENS,
    "gateway_url": CENTRAL_DEV_GATEWAY_URL,
    "provider_retries": 0,
    "responder_temperature": None,
    "judge_temperature": None,
    "deduplication": "identical canonical question+gold+prediction messages",
    "gold_access": "after complete final-answer artifact validation",
}
FINAL_ANSWER_SEMANTIC_JUDGE_POLICY_SHA256 = identity_sha256(
    FINAL_ANSWER_SEMANTIC_JUDGE_POLICY
)


class FinalAnswerSemanticJudgeRuntime(Protocol):
    identity: Any
    last_journal_record: Mapping[str, Any] | None

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class _BoundQuestion:
    ordinal: int
    source: Mapping[str, Any]
    gold: BenchmarkQuestion


@dataclass(frozen=True, slots=True)
class _PlannedJudgment:
    ordinal: int
    question_id: str
    category: str
    question_sha256: str
    dated_question_sha256: str
    gold_answer_sha256: str
    prediction_sha256: str
    answer_call_key_sha256: str
    answer_response_journal_sha256: str
    messages: tuple[Mapping[str, str], ...]
    messages_sha256: str
    prompt_token_proxy: int


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _runtime_identity(runtime: FinalAnswerSemanticJudgeRuntime) -> dict[str, Any]:
    value = runtime.identity
    if isinstance(value, Mapping):
        return {str(key): child for key, child in value.items()}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        payload = dump()
        if isinstance(payload, Mapping):
            return {str(key): child for key, child in payload.items()}
    raise TypeError("final-answer semantic judge has no mapping identity")


def _attest_responder_runtime(artifact: Mapping[str, Any]) -> None:
    identity = artifact.get("runtime_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("final-answer responder runtime identity is missing")
    expected = {
        "format": FINAL_ANSWER_RUNTIME_FORMAT,
        "gateway_url": CENTRAL_DEV_GATEWAY_URL,
        "caller_model": LOCKED_RESPONDER_MODEL,
        "gateway_model": LOCKED_RESPONDER_GATEWAY_MODEL,
        "default_max_new_tokens": RESPONDER_OUTPUT_TOKEN_RESERVE,
        "max_prompt_token_proxy": RESPONDER_PROMPT_CAP,
        "retries": 0,
        "temperature": None,
        "persisted_request_token_state": False,
        "retained_request_token_state_bytes": 0,
        "external_provider_persistence_certified": False,
    }
    if any(identity.get(name) != value for name, value in expected.items()):
        raise ValueError(
            "fixed-stage judge requires the attested Terra/256/8000/zero-retry "
            "final-answer runtime"
        )
    if _gateway_model(str(identity["caller_model"])) != identity["gateway_model"]:
        raise ValueError("final-answer caller and gateway models conflict")


def _attest_judge_runtime(identity: Mapping[str, Any]) -> None:
    expected = {
        "format": SEMANTIC_JUDGE_RUNTIME_FORMAT,
        "gateway_url": CENTRAL_DEV_GATEWAY_URL,
        "caller_model": LOCKED_JUDGE_MODEL,
        "gateway_model": LOCKED_JUDGE_GATEWAY_MODEL,
        "default_max_new_tokens": LOCKED_JUDGE_MAX_NEW_TOKENS,
        "retries": 0,
        "temperature": None,
    }
    if any(identity.get(name) != value for name, value in expected.items()):
        raise ValueError(
            "fixed-stage campaign requires the attested zero-retry Sol judge"
        )
    if _gateway_model(str(identity["caller_model"])) != identity["gateway_model"]:
        raise ValueError("judge caller and gateway models conflict")


def _validate_answer_rows_before_gold(
    artifact: Mapping[str, Any],
    *,
    retrieval: Mapping[str, Any],
    artifact_sha256: str,
    retrieval_sha256: str,
    sample: BenchmarkSample,
) -> list[_BoundQuestion]:
    """Validate all answer/retrieval state before reading ``question.answer``."""

    # This validator is deliberately the first operation.  In particular,
    # no question answer is hashed, formatted, or otherwise observed above it.
    validate_final_answer_artifact(
        artifact,
        retrieval=retrieval,
        artifact_sha256=artifact_sha256,
        retrieval_sha256=retrieval_sha256,
    )
    _attest_responder_runtime(artifact)
    responder_identity = artifact["runtime_identity"]
    responder_identity_sha256 = _require_sha256(
        artifact.get("runtime_identity_sha256"),
        "responder runtime identity SHA-256",
    )
    if identity_sha256(dict(responder_identity)) != responder_identity_sha256:
        raise ValueError("responder runtime identity seal changed")
    responder_campaign = artifact.get("campaign_binding")
    if not isinstance(responder_campaign, Mapping):
        raise ValueError("responder campaign binding is missing")
    responder_campaign_sha256 = _require_sha256(
        artifact.get("campaign_binding_sha256"),
        "responder campaign binding SHA-256",
    )
    if (
        identity_sha256(dict(responder_campaign))
        != responder_campaign_sha256
        or responder_identity.get("campaign_binding")
        != responder_campaign
        or responder_identity.get("campaign_binding_sha256")
        != responder_campaign_sha256
    ):
        raise ValueError("responder campaign binding seal changed")
    responder_prompt_policy = responder_campaign.get(
        "responder_prompt_policy"
    )
    if (
        not isinstance(responder_prompt_policy, Mapping)
        or responder_prompt_policy.get("format")
        != RESPONDER_PROMPT_POLICY_FORMAT
    ):
        raise ValueError("responder prompt policy identity is missing")
    responder_prompt_policy_sha256 = _require_sha256(
        responder_campaign.get("responder_prompt_policy_sha256"),
        "responder prompt policy SHA-256",
    )
    if (
        identity_sha256(dict(responder_prompt_policy))
        != responder_prompt_policy_sha256
        or artifact.get("responder_prompt_policy_sha256")
        != responder_prompt_policy_sha256
    ):
        raise ValueError("responder prompt policy seal changed")
    if (
        artifact.get("format") != FINAL_ANSWER_FORMAT
        or artifact.get("retrieval_sha256") != retrieval_sha256
        or artifact.get("population_identity_sha256")
        != retrieval.get("population_identity_sha256")
        or artifact.get("fixed_stage_id") != FIXED_STAGE_ID
        or artifact.get("gold_fields_present") is not False
    ):
        raise ValueError("fixed-stage answer/retrieval binding changed")
    rows = artifact.get("questions")
    retrieval_rows = retrieval.get("questions")
    sample_questions = sample.questions
    if (
        not isinstance(rows, list)
        or not isinstance(retrieval_rows, list)
        or len(rows) != len(retrieval_rows)
        or len(rows) != len(sample_questions)
        or artifact.get("question_count") != len(rows)
    ):
        raise ValueError("answer, retrieval, and scoring populations differ")

    bound: list[_BoundQuestion] = []
    seen: set[str] = set()
    for ordinal, (source, retrieval_row, question) in enumerate(
        zip(rows, retrieval_rows, sample_questions, strict=True)
    ):
        if not isinstance(source, Mapping) or not isinstance(
            retrieval_row, Mapping
        ):
            raise ValueError("answer/retrieval question must be an object")
        question_id = question.question_id
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("scoring question ID is missing")
        if question_id in seen:
            raise ValueError("scoring question IDs must be unique")
        seen.add(question_id)
        question_sha = quote_sha256(question.question)
        dated_sha = quote_sha256(question.dated_question)
        if (
            source.get("ordinal") != ordinal
            or retrieval_row.get("ordinal") != ordinal
            or source.get("question_id") != question_id
            or retrieval_row.get("question_id") != question_id
            or source.get("question_sha256") != question_sha
            or retrieval_row.get("question_sha256") != question_sha
            or source.get("dated_question_sha256") != dated_sha
            or retrieval_row.get("dated_question_sha256") != dated_sha
            or source.get("fixed_stage_id") != FIXED_STAGE_ID
            or source.get("prompt_token_cap") != RESPONDER_PROMPT_CAP
            or source.get("output_token_reserve")
            != RESPONDER_OUTPUT_TOKEN_RESERVE
        ):
            raise ValueError("fixed-stage answer order or question binding changed")
        answer = source.get("answer")
        report = source.get("completion_report")
        if not isinstance(answer, Mapping) or not isinstance(report, Mapping):
            raise ValueError("fixed-stage answer provenance is incomplete")
        prediction = answer.get("text")
        if (
            not isinstance(prediction, str)
            or not prediction
            or prediction.strip() != prediction
            or answer.get("sha256") != quote_sha256(prediction)
        ):
            raise ValueError("fixed-stage prediction seal changed")
        prompt_tokens = source.get("prompt_token_proxy")
        if (
            type(prompt_tokens) is not int
            or prompt_tokens < 0
            or prompt_tokens > RESPONDER_PROMPT_CAP
            or report.get("input_token_proxy") != prompt_tokens
            or report.get("max_new_tokens") != RESPONDER_OUTPUT_TOKEN_RESERVE
            or report.get("max_prompt_token_proxy") != RESPONDER_PROMPT_CAP
            or report.get("messages_sha256")
            != source.get("provider_messages_sha256")
            or report.get("completion_sha256") != answer.get("sha256")
            or report.get("retries") != 0
            or report.get("runtime_identity_sha256")
            != responder_identity_sha256
            or report.get("campaign_binding_sha256")
            != responder_campaign_sha256
            or report.get("prompt_population_sha256")
            != responder_identity.get("prompt_population_sha256")
        ):
            raise ValueError("fixed-stage responder request binding changed")
        provider_available = report.get("reported_input_tokens_available")
        provider_tokens = report.get("reported_input_tokens")
        if type(provider_available) is not bool:
            raise ValueError(
                "provider-reported responder input availability changed"
            )
        if provider_available:
            if type(provider_tokens) is not int or provider_tokens < 1:
                raise ValueError(
                    "provider-reported responder input token count changed"
                )
            if provider_tokens > RESPONDER_PROMPT_CAP:
                raise ValueError(
                    "provider-reported responder input exceeds hard cap"
                )
        elif provider_tokens != 0:
            raise ValueError(
                "unavailable provider-reported responder input must be zero"
            )
        for field in (
            "call_key_sha256",
            "request_journal_sha256",
            "response_journal_sha256",
        ):
            _require_sha256(source.get(field), f"answer row {field}")
        if (
            source.get("call_key_sha256") != report.get("call_key_sha256")
            or source.get("request_journal_sha256")
            != report.get("request_journal_sha256")
        ):
            raise ValueError("answer call/report journal binding changed")
        bound.append(_BoundQuestion(ordinal, source, question))
    return bound


def _plan_judgments(
    artifact: Mapping[str, Any],
    *,
    retrieval: Mapping[str, Any],
    artifact_sha256: str,
    retrieval_sha256: str,
    sample: BenchmarkSample,
) -> list[_PlannedJudgment]:
    bound = _validate_answer_rows_before_gold(
        artifact,
        retrieval=retrieval,
        artifact_sha256=artifact_sha256,
        retrieval_sha256=retrieval_sha256,
        sample=sample,
    )
    planned: list[_PlannedJudgment] = []
    # Gold access begins here, after the complete answer artifact, retrieval,
    # runtime, cap, provenance, and non-gold population have passed validation.
    for row in bound:
        question = row.gold
        prediction = str(row.source["answer"]["text"])
        gold_answer = question.answer
        messages = tuple(
            dict(message)
            for message in build_judge_prompt(
                question.question,
                gold_answer,
                prediction,
            )
        )
        planned.append(
            _PlannedJudgment(
                ordinal=row.ordinal,
                question_id=question.question_id,
                category=str(question.category or "unknown"),
                question_sha256=quote_sha256(question.question),
                dated_question_sha256=quote_sha256(question.dated_question),
                gold_answer_sha256=quote_sha256(gold_answer),
                prediction_sha256=quote_sha256(prediction),
                answer_call_key_sha256=str(row.source["call_key_sha256"]),
                answer_response_journal_sha256=str(
                    row.source["response_journal_sha256"]
                ),
                messages=messages,
                messages_sha256=identity_sha256(list(messages)),
                prompt_token_proxy=count_chat_prompt_token_proxy(messages),
            )
        )
    return planned


def _unique_prompts(
    planned: Sequence[_PlannedJudgment],
) -> dict[str, tuple[Mapping[str, str], ...]]:
    unique: dict[str, tuple[Mapping[str, str], ...]] = {}
    for row in planned:
        previous = unique.setdefault(row.messages_sha256, row.messages)
        if previous != row.messages:
            raise RuntimeError("fixed-stage judge message SHA-256 collision")
    return unique


def _gold_population_sha256(planned: Sequence[_PlannedJudgment]) -> str:
    return identity_sha256(
        [
            {
                "ordinal": row.ordinal,
                "question_id": row.question_id,
                "question_sha256": row.question_sha256,
                "dated_question_sha256": row.dated_question_sha256,
                "gold_answer_sha256": row.gold_answer_sha256,
            }
            for row in planned
        ]
    )


def build_final_answer_semantic_judge_campaign_binding(
    artifact: Mapping[str, Any],
    *,
    retrieval: Mapping[str, Any],
    sample: BenchmarkSample,
    artifact_sha256: str,
    retrieval_sha256: str,
    authorized_unique_calls: int,
) -> dict[str, Any]:
    """Preflight the complete fixed-stage judge population without calls."""

    if type(authorized_unique_calls) is not int or authorized_unique_calls < 1:
        raise ValueError("authorized_unique_calls must be a positive integer")
    implementation = implementation_sha256()
    planned = _plan_judgments(
        artifact,
        retrieval=retrieval,
        artifact_sha256=artifact_sha256,
        retrieval_sha256=retrieval_sha256,
        sample=sample,
    )
    unique = _unique_prompts(planned)
    if authorized_unique_calls != len(unique):
        raise ValueError(
            "authorized unique judge-call cap must exactly equal the "
            f"precomputed population ({authorized_unique_calls} != "
            f"{len(unique)})"
        )
    binding = {
        "format": FINAL_ANSWER_SEMANTIC_JUDGE_CAMPAIGN_FORMAT,
        "final_answer_artifact_sha256": artifact_sha256,
        "responder_runtime_identity_sha256": artifact[
            "runtime_identity_sha256"
        ],
        "responder_prompt_policy": dict(
            artifact["campaign_binding"]["responder_prompt_policy"]
        ),
        "responder_prompt_policy_sha256": artifact[
            "responder_prompt_policy_sha256"
        ],
        "retrieval_sha256": retrieval_sha256,
        "population_identity_sha256": artifact["population_identity_sha256"],
        "gold_scoring_population_sha256": _gold_population_sha256(planned),
        "question_count": len(planned),
        "fixed_stage_id": FIXED_STAGE_ID,
        "ordered_judgment_population_sha256": identity_sha256(
            [
                {
                    "ordinal": row.ordinal,
                    "question_id": row.question_id,
                    "question_sha256": row.question_sha256,
                    "gold_answer_sha256": row.gold_answer_sha256,
                    "prediction_sha256": row.prediction_sha256,
                    "judge_messages_sha256": row.messages_sha256,
                    "answer_call_key_sha256": row.answer_call_key_sha256,
                    "answer_response_journal_sha256": (
                        row.answer_response_journal_sha256
                    ),
                }
                for row in planned
            ]
        ),
        "logical_judgment_count": len(planned),
        "unique_judge_prompt_count": len(unique),
        "judge_prompt_population_sha256": identity_sha256(
            [
                {
                    "messages_sha256": digest,
                    "logical_references": sum(
                        row.messages_sha256 == digest for row in planned
                    ),
                }
                for digest in unique
            ]
        ),
        "maximum_judge_prompt_token_proxy": max(
            (row.prompt_token_proxy for row in planned),
            default=0,
        ),
        "authorized_unique_judge_calls": authorized_unique_calls,
        "semantic_judge_policy_sha256": (
            FINAL_ANSWER_SEMANTIC_JUDGE_POLICY_SHA256
        ),
        "semantic_judge_implementation_sha256": implementation,
        "target_accuracy": TARGET_ACCURACY,
        "minimum_questions": MINIMUM_GATE_QUESTIONS,
        "responder_model": LOCKED_RESPONDER_MODEL,
        "responder_max_new_tokens": RESPONDER_OUTPUT_TOKEN_RESERVE,
        "responder_prompt_cap": RESPONDER_PROMPT_CAP,
        "judge_model": LOCKED_JUDGE_MODEL,
        "judge_max_new_tokens": LOCKED_JUDGE_MAX_NEW_TOKENS,
        "provider_retries": 0,
    }
    if implementation_sha256() != implementation:
        raise RuntimeError("judge implementation changed during preflight")
    return binding


def _accuracy_status(correct: int, questions: int) -> dict[str, Any]:
    accuracy = correct / questions if questions else 0.0
    accuracy_met = accuracy >= TARGET_ACCURACY
    population_met = questions >= MINIMUM_GATE_QUESTIONS
    return {
        "questions": questions,
        "correct": correct,
        "incorrect": questions - correct,
        "binary_accuracy": accuracy,
        "target_accuracy": TARGET_ACCURACY,
        "minimum_questions": MINIMUM_GATE_QUESTIONS,
        "minimum_correct_at_observed_population": math.ceil(
            TARGET_ACCURACY * questions
        ),
        "accuracy_threshold_met": accuracy_met,
        "minimum_population_met": population_met,
        "gate_passed": accuracy_met and population_met,
        "status": (
            "pass"
            if accuracy_met and population_met
            else "insufficient_population"
            if not population_met
            else "below_accuracy_target"
        ),
    }


def _immutable_judge_usage(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, int | float]:
    def available(name: str) -> list[int]:
        return [
            int(report[name])
            for report in reports
            if report.get(name + "_available") is True
        ]

    reported_input = available("reported_input_tokens")
    reported_output = available("reported_output_tokens")
    reported_total = available("reported_total_tokens")
    return {
        "unique_journaled_calls": len(reports),
        "reported_input_tokens_available_calls": len(reported_input),
        "reported_input_tokens": sum(reported_input),
        "reported_output_tokens_available_calls": len(reported_output),
        "reported_output_tokens": sum(reported_output),
        "reported_total_tokens_available_calls": len(reported_total),
        "reported_total_tokens": sum(reported_total),
        "input_token_proxy": sum(
            int(report["input_token_proxy"]) for report in reports
        ),
        "output_token_proxy": sum(
            int(report["output_token_proxy"]) for report in reports
        ),
        "elapsed_s": sum(float(report["elapsed_s"]) for report in reports),
        "retries": 0,
    }


def judge_recall_guarded_cumulative_final_answers(
    artifact: Mapping[str, Any],
    *,
    retrieval: Mapping[str, Any],
    sample: BenchmarkSample,
    artifact_sha256: str,
    retrieval_sha256: str,
    runtime: FinalAnswerSemanticJudgeRuntime,
) -> dict[str, Any]:
    """Judge one fixed-stage answer per question and apply the locked gate."""

    implementation = implementation_sha256()
    planned = _plan_judgments(
        artifact,
        retrieval=retrieval,
        artifact_sha256=artifact_sha256,
        retrieval_sha256=retrieval_sha256,
        sample=sample,
    )
    unique = _unique_prompts(planned)
    runtime_identity = _runtime_identity(runtime)
    _attest_judge_runtime(runtime_identity)
    authorized = runtime_identity.get("authorized_unique_calls")
    if type(authorized) is not int or authorized != len(unique):
        raise ValueError(
            "judge runtime authorization must exactly equal the complete "
            "unique prompt population"
        )
    campaign = build_final_answer_semantic_judge_campaign_binding(
        artifact,
        retrieval=retrieval,
        sample=sample,
        artifact_sha256=artifact_sha256,
        retrieval_sha256=retrieval_sha256,
        authorized_unique_calls=authorized,
    )
    runtime_identity_sha256 = identity_sha256(runtime_identity)
    campaign_binding_sha256 = identity_sha256(campaign)
    if (
        runtime_identity.get("campaign_binding") != campaign
        or runtime_identity.get("campaign_binding_sha256")
        != campaign_binding_sha256
    ):
        raise ValueError("Sol runtime belongs to another fixed-stage campaign")

    outcomes: dict[str, dict[str, Any]] = {}
    for messages_sha, messages in unique.items():
        verdict_text = runtime.complete(
            messages,
            max_new_tokens=LOCKED_JUDGE_MAX_NEW_TOKENS,
        )
        verdict = parse_binary_judge_verdict(verdict_text)
        journal = runtime.last_journal_record
        if not isinstance(journal, Mapping):
            raise RuntimeError("Sol runtime omitted journal provenance")
        report = journal.get("completion_report")
        if not isinstance(report, Mapping):
            raise RuntimeError("Sol runtime omitted its completion report")
        call_key = _require_sha256(
            journal.get("call_key_sha256"),
            "judge call key SHA-256",
        )
        request_journal = _require_sha256(
            journal.get("request_journal_sha256"),
            "judge request journal SHA-256",
        )
        response_journal = _require_sha256(
            journal.get("response_journal_sha256"),
            "judge response journal SHA-256",
        )
        if (
            report.get("messages_sha256") != messages_sha
            or report.get("max_new_tokens") != LOCKED_JUDGE_MAX_NEW_TOKENS
            or journal.get("completion_sha256") != quote_sha256(verdict_text)
            or report.get("completion_sha256") != quote_sha256(verdict_text)
            or report.get("call_key_sha256") != call_key
            or report.get("request_journal_sha256") != request_journal
            or report.get("runtime_identity_sha256")
            != runtime_identity_sha256
            or report.get("campaign_binding_sha256")
            != campaign_binding_sha256
            or report.get("retries") != 0
            or (
                "response_journal_sha256" in report
                and report.get("response_journal_sha256")
                != response_journal
            )
        ):
            raise RuntimeError("Sol runtime changed prompt/response binding")
        outcomes[messages_sha] = {
            "correct": verdict,
            "judge_output": verdict_text,
            "judge_output_sha256": quote_sha256(verdict_text),
            "call_key_sha256": call_key,
            "request_journal_sha256": request_journal,
            "response_journal_sha256": response_journal,
            "completion_report": dict(report),
        }

    questions: list[dict[str, Any]] = []
    verdicts: list[bool] = []
    category_values: dict[str, list[bool]] = {}
    for row in planned:
        outcome = outcomes[row.messages_sha256]
        correct = bool(outcome["correct"])
        verdicts.append(correct)
        category_values.setdefault(row.category, []).append(correct)
        questions.append(
            {
                "ordinal": row.ordinal,
                "question_id": row.question_id,
                "category": row.category,
                "question_sha256": row.question_sha256,
                "dated_question_sha256": row.dated_question_sha256,
                "gold_answer_sha256": row.gold_answer_sha256,
                "prediction_sha256": row.prediction_sha256,
                "fixed_stage_id": FIXED_STAGE_ID,
                "answer_call_key_sha256": row.answer_call_key_sha256,
                "answer_response_journal_sha256": (
                    row.answer_response_journal_sha256
                ),
                "judge_messages_sha256": row.messages_sha256,
                "judge_prompt_token_proxy": row.prompt_token_proxy,
                **outcome,
            }
        )
    aggregate = _accuracy_status(sum(verdicts), len(verdicts))
    category_aggregates = [
        {
            "category": category,
            **_accuracy_status(sum(values), len(values)),
        }
        for category, values in sorted(category_values.items())
    ]
    reports = [outcome["completion_report"] for outcome in outcomes.values()]
    result = {
        "format": FINAL_ANSWER_SEMANTIC_JUDGE_FORMAT,
        "final_answer_artifact_sha256": artifact_sha256,
        "responder_runtime_identity_sha256": artifact[
            "runtime_identity_sha256"
        ],
        "responder_prompt_policy": dict(
            campaign["responder_prompt_policy"]
        ),
        "responder_prompt_policy_sha256": artifact[
            "responder_prompt_policy_sha256"
        ],
        "retrieval_sha256": retrieval_sha256,
        "population_identity_sha256": artifact["population_identity_sha256"],
        "gold_scoring_population_sha256": _gold_population_sha256(planned),
        "question_count": len(planned),
        "gold_loaded_posthoc": True,
        "independent_llm_judge": True,
        "fixed_stage_id": FIXED_STAGE_ID,
        "responder_model": LOCKED_RESPONDER_MODEL,
        "judge_model": LOCKED_JUDGE_MODEL,
        "judge_runtime_identity": runtime_identity,
        "judge_runtime_identity_sha256": identity_sha256(runtime_identity),
        "semantic_judge_policy": dict(FINAL_ANSWER_SEMANTIC_JUDGE_POLICY),
        "semantic_judge_policy_sha256": (
            FINAL_ANSWER_SEMANTIC_JUDGE_POLICY_SHA256
        ),
        "semantic_judge_implementation_sha256": implementation,
        "campaign_binding": campaign,
        "campaign_binding_sha256": identity_sha256(campaign),
        "logical_judgment_count": len(planned),
        "unique_judge_prompt_count": len(unique),
        "deduplicated_logical_judgment_count": len(planned) - len(unique),
        "judge_prompt_preflight": {
            "completed_before_provider_calls": True,
            "logical_prompt_count": len(planned),
            "unique_prompt_count": len(unique),
            "maximum_prompt_token_proxy": max(
                (row.prompt_token_proxy for row in planned),
                default=0,
            ),
        },
        "judge_usage": _immutable_judge_usage(reports),
        "questions": questions,
        "category_counts": dict(
            sorted(Counter(row.category for row in planned).items())
        ),
        "category_aggregates": category_aggregates,
        "aggregate": aggregate,
        "target_gate": {
            **aggregate,
            "gate_unit": "one preregistered fixed retrieval stage",
            "fixed_stage_id": FIXED_STAGE_ID,
        },
    }
    if implementation_sha256() != implementation:
        raise RuntimeError("judge implementation changed during scoring")
    return result


__all__ = [
    "FINAL_ANSWER_SEMANTIC_JUDGE_CAMPAIGN_FORMAT",
    "FINAL_ANSWER_SEMANTIC_JUDGE_FORMAT",
    "FINAL_ANSWER_SEMANTIC_JUDGE_POLICY",
    "FINAL_ANSWER_SEMANTIC_JUDGE_POLICY_SHA256",
    "LOCKED_JUDGE_GATEWAY_MODEL",
    "LOCKED_JUDGE_MAX_NEW_TOKENS",
    "LOCKED_JUDGE_MODEL",
    "MINIMUM_GATE_QUESTIONS",
    "TARGET_ACCURACY",
    "build_final_answer_semantic_judge_campaign_binding",
    "judge_recall_guarded_cumulative_final_answers",
]
