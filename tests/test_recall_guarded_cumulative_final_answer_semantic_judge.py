from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from memory_condense.domain._tokenizer import tokenizer_proxy_identity
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval import (
    recall_guarded_cumulative_final_answer_semantic_judge as judge,
)
from memory_condense.eval.recall_guarded_cumulative_final_answer import (
    FINAL_ANSWER_POLICY_SHA256,
)
from memory_condense.ingest.loader import BenchmarkQuestion, BenchmarkSample


_ARTIFACT_SHA = "a" * 64
_RETRIEVAL_SHA = "b" * 64
_POPULATION_SHA = "c" * 64
_IMPLEMENTATION_SHA = "d" * 64


def _responder_identity(
    question_count: int,
    *,
    campaign: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format": judge.FINAL_ANSWER_RUNTIME_FORMAT,
        "gateway_url": judge.CENTRAL_DEV_GATEWAY_URL,
        "caller_model": judge.LOCKED_RESPONDER_MODEL,
        "gateway_model": judge.LOCKED_RESPONDER_GATEWAY_MODEL,
        "default_max_new_tokens": judge.RESPONDER_OUTPUT_TOKEN_RESERVE,
        "max_prompt_token_proxy": judge.RESPONDER_PROMPT_CAP,
        "retries": 0,
        "temperature": None,
        "authorized_unique_calls": question_count,
        "logical_prompt_count": question_count,
        "unique_prompt_count": question_count,
        "prompt_population_sha256": "e" * 64,
        "prompt_token_proxy_identity": tokenizer_proxy_identity(),
        "campaign_binding": dict(campaign),
        "campaign_binding_sha256": identity_sha256(dict(campaign)),
        "request_journal_format": judge.FINAL_ANSWER_REQUEST_JOURNAL_FORMAT,
        "response_journal_format": judge.FINAL_ANSWER_RESPONSE_JOURNAL_FORMAT,
        "persisted_request_token_state": False,
        "retained_request_token_state_bytes": 0,
        "external_provider_persistence_certified": False,
    }


def _inputs(
    count: int,
) -> tuple[dict[str, Any], dict[str, Any], BenchmarkSample]:
    questions = [
        BenchmarkQuestion(
            question_id=f"q{index:03d}",
            question=f"What is value {index}?",
            answer=f"gold-{index}",
            category="even" if index % 2 == 0 else "odd",
            question_date="2026-08-22",
        )
        for index in range(count)
    ]
    retrieval_rows = []
    answer_rows = []
    for ordinal, question in enumerate(questions):
        question_sha = quote_sha256(question.question)
        dated_sha = quote_sha256(question.dated_question)
        call_key = hashlib.sha256(f"answer-call-{ordinal}".encode()).hexdigest()
        request_sha = hashlib.sha256(
            f"answer-request-{ordinal}".encode()
        ).hexdigest()
        response_sha = hashlib.sha256(
            f"answer-response-{ordinal}".encode()
        ).hexdigest()
        prediction = f"prediction-{ordinal}"
        prediction_sha = quote_sha256(prediction)
        message_sha = hashlib.sha256(
            f"answer-messages-{ordinal}".encode()
        ).hexdigest()
        retrieval_rows.append(
            {
                "ordinal": ordinal,
                "question_id": question.question_id,
                "question_sha256": question_sha,
                "dated_question_sha256": dated_sha,
            }
        )
        answer_rows.append(
            {
                "ordinal": ordinal,
                "question_id": question.question_id,
                "question_sha256": question_sha,
                "dated_question_sha256": dated_sha,
                "fixed_stage_id": judge.FIXED_STAGE_ID,
                "prompt_token_proxy": 123,
                "prompt_token_cap": judge.RESPONDER_PROMPT_CAP,
                "output_token_reserve": judge.RESPONDER_OUTPUT_TOKEN_RESERVE,
                "provider_messages_sha256": message_sha,
                "answer": {"text": prediction, "sha256": prediction_sha},
                "call_key_sha256": call_key,
                "request_journal_sha256": request_sha,
                "response_journal_sha256": response_sha,
                "completion_report": {
                    "call_key_sha256": call_key,
                    "request_journal_sha256": request_sha,
                    "messages_sha256": message_sha,
                    "completion_sha256": prediction_sha,
                    "input_token_proxy": 123,
                    "max_prompt_token_proxy": judge.RESPONDER_PROMPT_CAP,
                    "max_new_tokens": judge.RESPONDER_OUTPUT_TOKEN_RESERVE,
                    "reported_input_tokens_available": False,
                    "reported_input_tokens": 0,
                    "retries": 0,
                },
            }
        )
    retrieval = {
        "population_identity_sha256": _POPULATION_SHA,
        "questions": retrieval_rows,
    }
    responder_prompt_policy = {
        "format": judge.RESPONDER_PROMPT_POLICY_FORMAT,
        "ordered_unique_system_content_quote_sha256s": ["f" * 64],
        "qa_user_template_quote_sha256": "1" * 64,
        "prompt_token_proxy_identity": tokenizer_proxy_identity(),
        "responder_output_token_reserve": (
            judge.RESPONDER_OUTPUT_TOKEN_RESERVE
        ),
    }
    responder_campaign = {
        "format": judge.FINAL_ANSWER_CAMPAIGN_FORMAT,
        "question_count": count,
        "prompt_population_sha256": "e" * 64,
        "responder_prompt_policy": responder_prompt_policy,
        "responder_prompt_policy_sha256": identity_sha256(
            responder_prompt_policy
        ),
    }
    responder_identity = _responder_identity(
        count,
        campaign=responder_campaign,
    )
    responder_identity_sha = identity_sha256(responder_identity)
    responder_campaign_sha = identity_sha256(responder_campaign)
    for row in answer_rows:
        report = row["completion_report"]
        report["runtime_identity_sha256"] = responder_identity_sha
        report["campaign_binding_sha256"] = responder_campaign_sha
        report["prompt_population_sha256"] = "e" * 64
    artifact = {
        "format": judge.FINAL_ANSWER_FORMAT,
        "retrieval_sha256": _RETRIEVAL_SHA,
        "population_identity_sha256": _POPULATION_SHA,
        "question_count": count,
        "fixed_stage_id": judge.FIXED_STAGE_ID,
        "gold_fields_present": False,
        "final_answer_policy_sha256": FINAL_ANSWER_POLICY_SHA256,
        "responder_prompt_policy_sha256": responder_campaign[
            "responder_prompt_policy_sha256"
        ],
        "runtime_identity": responder_identity,
        "runtime_identity_sha256": responder_identity_sha,
        "campaign_binding": responder_campaign,
        "campaign_binding_sha256": responder_campaign_sha,
        "questions": answer_rows,
    }
    return artifact, retrieval, BenchmarkSample(
        sample_id="scoring-population",
        questions=questions,
    )


class _FakeJudgeRuntime:
    def __init__(
        self,
        campaign: Mapping[str, Any],
        *,
        authorized: int,
        correct: int,
        model: str = judge.LOCKED_JUDGE_MODEL,
        report_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self.identity = {
            "format": judge.SEMANTIC_JUDGE_RUNTIME_FORMAT,
            "gateway_url": judge.CENTRAL_DEV_GATEWAY_URL,
            "caller_model": model,
            "gateway_model": judge._gateway_model(model),
            "default_max_new_tokens": judge.LOCKED_JUDGE_MAX_NEW_TOKENS,
            "retries": 0,
            "temperature": None,
            "authorized_unique_calls": authorized,
            "campaign_binding": dict(campaign),
            "campaign_binding_sha256": identity_sha256(dict(campaign)),
        }
        self.correct = correct
        self.report_overrides = dict(report_overrides or {})
        self.calls: list[list[dict[str, str]]] = []
        self.last_journal_record: Mapping[str, Any] | None = None

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        assert max_new_tokens == judge.LOCKED_JUDGE_MAX_NEW_TOKENS
        normalized = [dict(row) for row in messages]
        ordinal = len(self.calls)
        self.calls.append(normalized)
        text = (
            "CORRECT: same fact"
            if ordinal < self.correct
            else "INCORRECT: different fact"
        )
        messages_sha = identity_sha256(normalized)
        completion_sha = quote_sha256(text)
        call_key = hashlib.sha256(f"judge-call-{ordinal}".encode()).hexdigest()
        request_sha = hashlib.sha256(
            f"judge-request-{ordinal}".encode()
        ).hexdigest()
        response_sha = hashlib.sha256(
            f"judge-response-{ordinal}".encode()
        ).hexdigest()
        report = {
            "call_key_sha256": call_key,
            "request_journal_sha256": request_sha,
            "response_journal_sha256": response_sha,
            "runtime_identity_sha256": identity_sha256(self.identity),
            "campaign_binding_sha256": self.identity[
                "campaign_binding_sha256"
            ],
            "messages_sha256": messages_sha,
            "completion_sha256": completion_sha,
            "max_new_tokens": judge.LOCKED_JUDGE_MAX_NEW_TOKENS,
            "reported_input_tokens_available": True,
            "reported_input_tokens": 20,
            "reported_output_tokens_available": True,
            "reported_output_tokens": 4,
            "reported_total_tokens_available": True,
            "reported_total_tokens": 24,
            "input_token_proxy": 22,
            "output_token_proxy": 4,
            "elapsed_s": 0.25,
            "retries": 0,
        }
        report.update(self.report_overrides)
        self.last_journal_record = {
            "call_key_sha256": call_key,
            "request_journal_sha256": request_sha,
            "response_journal_sha256": response_sha,
            "completion_sha256": completion_sha,
            "completion_report": report,
        }
        return text


@pytest.fixture(autouse=True)
def _provider_free_artifact_validator(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        judge,
        "validate_final_answer_artifact",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        judge,
        "implementation_sha256",
        lambda: _IMPLEMENTATION_SHA,
    )


@pytest.mark.parametrize(
    ("correct", "passed", "status"),
    [
        (95, True, "pass"),
        (94, False, "below_accuracy_target"),
    ],
)
def test_locked_100_question_gate(
    correct: int,
    passed: bool,
    status: str,
) -> None:
    artifact, retrieval, sample = _inputs(100)
    campaign = judge.build_final_answer_semantic_judge_campaign_binding(
        artifact,
        retrieval=retrieval,
        sample=sample,
        artifact_sha256=_ARTIFACT_SHA,
        retrieval_sha256=_RETRIEVAL_SHA,
        authorized_unique_calls=100,
    )
    runtime = _FakeJudgeRuntime(
        campaign,
        authorized=100,
        correct=correct,
    )

    result = judge.judge_recall_guarded_cumulative_final_answers(
        artifact,
        retrieval=retrieval,
        sample=sample,
        artifact_sha256=_ARTIFACT_SHA,
        retrieval_sha256=_RETRIEVAL_SHA,
        runtime=runtime,
    )

    assert len(runtime.calls) == 100
    assert result["format"] == judge.FINAL_ANSWER_SEMANTIC_JUDGE_FORMAT
    assert result["responder_runtime_identity_sha256"] == artifact[
        "runtime_identity_sha256"
    ]
    assert result["responder_prompt_policy_sha256"] == artifact[
        "responder_prompt_policy_sha256"
    ]
    assert result["responder_prompt_policy"] == artifact["campaign_binding"][
        "responder_prompt_policy"
    ]
    assert result["campaign_binding"][
        "responder_runtime_identity_sha256"
    ] == artifact["runtime_identity_sha256"]
    assert result["campaign_binding"][
        "responder_prompt_policy"
    ] == artifact["campaign_binding"]["responder_prompt_policy"]
    assert result["campaign_binding"][
        "responder_prompt_policy_sha256"
    ] == artifact["responder_prompt_policy_sha256"]
    assert result["fixed_stage_id"] == "direct_episode_additions"
    assert result["logical_judgment_count"] == 100
    assert result["unique_judge_prompt_count"] == 100
    assert result["aggregate"]["correct"] == correct
    assert result["aggregate"]["binary_accuracy"] == correct / 100
    assert result["target_gate"]["gate_passed"] is passed
    assert result["target_gate"]["status"] == status
    assert result["target_gate"]["minimum_population_met"] is True
    assert result["target_gate"]["fixed_stage_id"] == judge.FIXED_STAGE_ID


def test_population_below_100_is_insufficient_even_when_perfect() -> None:
    artifact, retrieval, sample = _inputs(10)
    campaign = judge.build_final_answer_semantic_judge_campaign_binding(
        artifact,
        retrieval=retrieval,
        sample=sample,
        artifact_sha256=_ARTIFACT_SHA,
        retrieval_sha256=_RETRIEVAL_SHA,
        authorized_unique_calls=10,
    )
    runtime = _FakeJudgeRuntime(campaign, authorized=10, correct=10)

    result = judge.judge_recall_guarded_cumulative_final_answers(
        artifact,
        retrieval=retrieval,
        sample=sample,
        artifact_sha256=_ARTIFACT_SHA,
        retrieval_sha256=_RETRIEVAL_SHA,
        runtime=runtime,
    )

    assert result["target_gate"]["binary_accuracy"] == 1.0
    assert result["target_gate"]["minimum_population_met"] is False
    assert result["target_gate"]["gate_passed"] is False
    assert result["target_gate"]["status"] == "insufficient_population"


def test_answer_artifact_validation_precedes_gold_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, retrieval, source_sample = _inputs(1)
    state = {"validated": False}

    def validate(*_args, **_kwargs) -> None:
        state["validated"] = True

    class GuardedQuestion:
        question_id = "q000"
        question = "What is value 0?"
        dated_question = "[Question asked at 2026-08-22]\nWhat is value 0?"
        category = "even"

        @property
        def answer(self) -> str:
            assert state["validated"] is True
            return "gold-0"

    monkeypatch.setattr(judge, "validate_final_answer_artifact", validate)
    guarded_sample = SimpleNamespace(questions=[GuardedQuestion()])

    binding = judge.build_final_answer_semantic_judge_campaign_binding(
        artifact,
        retrieval=retrieval,
        sample=guarded_sample,  # type: ignore[arg-type]
        artifact_sha256=_ARTIFACT_SHA,
        retrieval_sha256=_RETRIEVAL_SHA,
        authorized_unique_calls=1,
    )

    assert state["validated"] is True
    assert binding["question_count"] == len(source_sample.questions)


def test_whole_population_failure_happens_before_any_sol_call() -> None:
    artifact, retrieval, sample = _inputs(3)
    artifact["questions"][-1]["fixed_stage_id"] = (
        "artifact_global_closure_additions"
    )
    runtime = _FakeJudgeRuntime({}, authorized=3, correct=3)

    with pytest.raises(ValueError, match="order or question binding"):
        judge.judge_recall_guarded_cumulative_final_answers(
            artifact,
            retrieval=retrieval,
            sample=sample,
            artifact_sha256=_ARTIFACT_SHA,
            retrieval_sha256=_RETRIEVAL_SHA,
            runtime=runtime,
        )

    assert runtime.calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("caller_model", judge.LOCKED_RESPONDER_MODEL),
        ("gateway_model", judge.LOCKED_RESPONDER_GATEWAY_MODEL),
        ("gateway_url", "https://wrong.invalid/v1"),
        ("format", "wrong-runtime"),
        ("default_max_new_tokens", judge.LOCKED_JUDGE_MAX_NEW_TOKENS + 1),
        ("retries", 1),
        ("temperature", 0.0),
    ],
)
def test_wrong_sol_runtime_is_rejected_before_calls(
    field: str,
    value: Any,
) -> None:
    artifact, retrieval, sample = _inputs(2)
    campaign = judge.build_final_answer_semantic_judge_campaign_binding(
        artifact,
        retrieval=retrieval,
        sample=sample,
        artifact_sha256=_ARTIFACT_SHA,
        retrieval_sha256=_RETRIEVAL_SHA,
        authorized_unique_calls=2,
    )
    runtime = _FakeJudgeRuntime(campaign, authorized=2, correct=2)
    runtime.identity[field] = value

    with pytest.raises(ValueError, match="zero-retry Sol judge"):
        judge.judge_recall_guarded_cumulative_final_answers(
            artifact,
            retrieval=retrieval,
            sample=sample,
            artifact_sha256=_ARTIFACT_SHA,
            retrieval_sha256=_RETRIEVAL_SHA,
            runtime=runtime,
        )

    assert runtime.calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("runtime_identity_sha256", "1" * 64),
        ("campaign_binding_sha256", "2" * 64),
        ("response_journal_sha256", "3" * 64),
    ],
)
def test_sol_completion_report_must_preserve_runtime_and_journal_bindings(
    field: str,
    value: str,
) -> None:
    artifact, retrieval, sample = _inputs(1)
    campaign = judge.build_final_answer_semantic_judge_campaign_binding(
        artifact,
        retrieval=retrieval,
        sample=sample,
        artifact_sha256=_ARTIFACT_SHA,
        retrieval_sha256=_RETRIEVAL_SHA,
        authorized_unique_calls=1,
    )
    runtime = _FakeJudgeRuntime(
        campaign,
        authorized=1,
        correct=1,
        report_overrides={field: value},
    )

    with pytest.raises(RuntimeError, match="prompt/response binding"):
        judge.judge_recall_guarded_cumulative_final_answers(
            artifact,
            retrieval=retrieval,
            sample=sample,
            artifact_sha256=_ARTIFACT_SHA,
            retrieval_sha256=_RETRIEVAL_SHA,
            runtime=runtime,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("caller_model", judge.LOCKED_JUDGE_MODEL),
        ("gateway_model", judge.LOCKED_JUDGE_GATEWAY_MODEL),
        ("default_max_new_tokens", 4_096),
        ("max_prompt_token_proxy", 8_001),
        ("retries", 1),
        ("temperature", 0.0),
        ("persisted_request_token_state", True),
        ("retained_request_token_state_bytes", 1),
        ("external_provider_persistence_certified", True),
    ],
)
def test_wrong_terra_answer_runtime_is_rejected_before_gold_or_calls(
    field: str,
    value: Any,
) -> None:
    artifact, retrieval, sample = _inputs(2)
    artifact["runtime_identity"][field] = value

    with pytest.raises(ValueError, match="Terra/256/8000/zero-retry"):
        judge.build_final_answer_semantic_judge_campaign_binding(
            artifact,
            retrieval=retrieval,
            sample=sample,
            artifact_sha256=_ARTIFACT_SHA,
            retrieval_sha256=_RETRIEVAL_SHA,
            authorized_unique_calls=2,
        )


@pytest.mark.parametrize("authorized", [1, 3, True])
def test_judge_authorization_must_equal_preflight_population(
    authorized: object,
) -> None:
    artifact, retrieval, sample = _inputs(2)

    with pytest.raises(ValueError, match="authorized"):
        judge.build_final_answer_semantic_judge_campaign_binding(
            artifact,
            retrieval=retrieval,
            sample=sample,
            artifact_sha256=_ARTIFACT_SHA,
            retrieval_sha256=_RETRIEVAL_SHA,
            authorized_unique_calls=authorized,  # type: ignore[arg-type]
        )


def test_provider_reported_responder_over_cap_is_rejected_pre_judge() -> None:
    artifact, retrieval, sample = _inputs(1)
    report = artifact["questions"][0]["completion_report"]
    report["reported_input_tokens_available"] = True
    report["reported_input_tokens"] = 8_001

    with pytest.raises(ValueError, match="provider-reported.*exceeds"):
        judge.build_final_answer_semantic_judge_campaign_binding(
            artifact,
            retrieval=retrieval,
            sample=sample,
            artifact_sha256=_ARTIFACT_SHA,
            retrieval_sha256=_RETRIEVAL_SHA,
            authorized_unique_calls=1,
        )


def test_responder_prompt_policy_must_match_validated_answer_campaign() -> None:
    artifact, retrieval, sample = _inputs(1)
    artifact["responder_prompt_policy_sha256"] = "2" * 64

    with pytest.raises(ValueError, match="prompt policy seal changed"):
        judge.build_final_answer_semantic_judge_campaign_binding(
            artifact,
            retrieval=retrieval,
            sample=sample,
            artifact_sha256=_ARTIFACT_SHA,
            retrieval_sha256=_RETRIEVAL_SHA,
            authorized_unique_calls=1,
        )
