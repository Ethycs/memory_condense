from __future__ import annotations

import copy
import hashlib
from typing import Any, Mapping, Sequence

import pytest

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval import recall_guarded_cumulative_semantic_judge as judge
from memory_condense.eval.recall_guarded_cumulative_1m import (
    _canonical_json_bytes,
)
from memory_condense.eval.recall_guarded_cumulative_synthesis import (
    SYNTHESIS_PROMPT_POLICY,
    build_synthesis_messages,
)
from memory_condense.ingest.loader import BenchmarkQuestion, BenchmarkSample


_POPULATION_SHA = "p" * 64
_IMPLEMENTATION_SHA = "i" * 64


def _responder_runtime_identity() -> dict[str, Any]:
    return {
        "format": judge.PROVIDER_RUNTIME_FORMAT,
        "gateway_url": judge.CENTRAL_DEV_GATEWAY_URL,
        "caller_model": judge.LOCKED_RESPONDER_MODEL,
        "gateway_model": judge.LOCKED_RESPONDER_GATEWAY_MODEL,
        "retries": 0,
        "temperature": None,
        "default_max_new_tokens": (
            judge.BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE
        ),
    }


def _sample() -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="sample",
        questions=[
            BenchmarkQuestion(
                question_id="q1",
                question="What color?",
                answer="blue",
                evidence_sources=["source-1"],
            )
        ],
    )


def _stage(stage_id: str, prediction: str) -> dict[str, Any]:
    return {
        "stage_id": stage_id,
        "answer": {"text": prediction},
        "synthesis_mode": "structured_generation",
        "request_policy": {
            "max_new_tokens": judge.BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE
        },
        "prompt_messages_sha256": "0" * 64,
        "completion_report": {
            "call_key_sha256": "c" * 64,
            "input_token_proxy": 0,
            "messages_sha256": "0" * 64,
            "reported_input_tokens_available": False,
            "reported_input_tokens": 0,
            "max_new_tokens": (
                judge.BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE
            ),
        },
    }


def _retrieval() -> dict[str, Any]:
    root = {
        "evidence_id": "root",
        "source_id": "source-root",
        "text": "The palette had several colors.",
    }
    novel = {
        "evidence_id": "novel",
        "source_id": "source-1",
        "text": "The selected color was blue.",
    }
    stages = []
    for index, stage_id in enumerate(("root", *judge.SYNTHESIS_STAGE_IDS)):
        messages = [
            {"role": "system", "content": "Answer only from the evidence."},
            {
                "role": "user",
                "content": (
                    f"Retrieved excerpts for {stage_id}.\n\n"
                    "Question: What color?\nShort answer:"
                ),
            },
        ]
        stages.append(
            {
                "stage_id": stage_id,
                "provider_messages": messages,
                "evidence": [root] if index == 0 else [root, novel],
            }
        )
    return {
        "format": "test-retrieval",
        "population_identity_sha256": _POPULATION_SHA,
        "questions": [
            {
                "ordinal": 0,
                "question_id": "q1",
                "question_sha256": quote_sha256("What color?"),
                "stages": stages,
            }
        ],
    }


def _synthesis() -> dict[str, Any]:
    retrieval = _retrieval()
    synthesis = {
        "format": "test-synthesis",
        "retrieval_sha256": hashlib.sha256(
            _canonical_json_bytes(retrieval)
        ).hexdigest(),
        "population_identity_sha256": _POPULATION_SHA,
        "synthesis_prompt_policy": SYNTHESIS_PROMPT_POLICY,
        "request_policy": {
            "max_new_tokens": judge.BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE
        },
        "runtime_identity": _responder_runtime_identity(),
        "questions": [
            {
                "ordinal": 0,
                "question_id": "q1",
                "question_sha256": quote_sha256("What color?"),
                "stages": [
                    _stage(judge.SYNTHESIS_STAGE_IDS[0], "blue"),
                    _stage(judge.SYNTHESIS_STAGE_IDS[1], "blue"),
                    _stage(judge.SYNTHESIS_STAGE_IDS[2], "red"),
                ],
            }
        ],
    }
    root_ids = {"root"}
    for source_stage, stage in zip(
        retrieval["questions"][0]["stages"][1:],
        synthesis["questions"][0]["stages"],
        strict=True,
    ):
        messages, _aliases, _novel = build_synthesis_messages(
            source_stage,
            root_evidence_ids=root_ids,
            prompt_policy=SYNTHESIS_PROMPT_POLICY,
        )
        messages_sha = identity_sha256(messages)
        stage["prompt_messages_sha256"] = messages_sha
        stage["completion_report"]["messages_sha256"] = messages_sha
        stage["completion_report"][
            "input_token_proxy"
        ] = count_chat_prompt_token_proxy(messages)
    return synthesis


class _FakeRuntime:
    def __init__(
        self,
        binding: Mapping[str, Any],
        *,
        model: str | None = None,
        authorized: int = 2,
    ):
        self.identity = {
            "format": judge.SEMANTIC_JUDGE_RUNTIME_FORMAT,
            "gateway_url": judge.CENTRAL_DEV_GATEWAY_URL,
            "caller_model": model or judge.LOCKED_JUDGE_MODEL,
            "gateway_model": judge._gateway_model(
                model or judge.LOCKED_JUDGE_MODEL
            ),
            "retries": 0,
            "temperature": None,
            "default_max_new_tokens": judge.LOCKED_JUDGE_MAX_NEW_TOKENS,
            "authorized_unique_calls": authorized,
            "campaign_binding": dict(binding),
        }
        self.last_journal_record: Mapping[str, Any] | None = None
        self.calls: list[list[dict[str, str]]] = []

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        normalized = [dict(row) for row in messages]
        self.calls.append(normalized)
        prompt_sha = identity_sha256(normalized)
        text = (
            "INCORRECT: different color"
            if "Predicted answer: red" in normalized[-1]["content"]
            else "CORRECT: same color"
        )
        completion_sha = quote_sha256(text)
        self.last_journal_record = {
            "call_key_sha256": hashlib.sha256(
                prompt_sha.encode("ascii")
            ).hexdigest(),
            "request_journal_sha256": "a" * 64,
            "response_journal_sha256": "b" * 64,
            "completion_sha256": completion_sha,
            "completion_report": {
                "messages_sha256": prompt_sha,
                "completion_sha256": completion_sha,
                "reported_input_tokens_available": True,
                "reported_input_tokens": 20,
                "reported_output_tokens_available": True,
                "reported_output_tokens": 4,
                "input_token_proxy": 22,
                "output_token_proxy": 4,
                "elapsed_s": 0.25,
            },
        }
        return text


@pytest.fixture(autouse=True)
def _minimal_validated_artifact(monkeypatch):
    monkeypatch.setattr(judge, "_validate_assembled_synthesis", lambda _value: None)
    monkeypatch.setattr(judge, "validate_published_retrieval", lambda _value: None)
    monkeypatch.setattr(
        judge,
        "population_identity_sha256",
        lambda _sample: _POPULATION_SHA,
    )
    monkeypatch.setattr(
        judge,
        "implementation_sha256",
        lambda: _IMPLEMENTATION_SHA,
    )


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _retrieval_binding() -> dict[str, Any]:
    retrieval = _retrieval()
    return {
        "retrieval": retrieval,
        "retrieval_sha256": _digest(retrieval),
    }


def test_s1_s3_semantic_judging_deduplicates_and_reports_gate() -> None:
    synthesis = _synthesis()
    sample = _sample()
    synthesis_sha = _digest(synthesis)
    binding = judge.build_semantic_judge_campaign_binding(
        synthesis,
        **_retrieval_binding(),
        sample=sample,
        synthesis_sha256=synthesis_sha,
        authorized_unique_calls=2,
    )
    assert binding["format"] == (
        "memory-condense-recall-guarded-semantic-judge-campaign-v2"
    )
    runtime = _FakeRuntime(binding)

    result = judge.judge_recall_guarded_cumulative_synthesis(
        synthesis,
        **_retrieval_binding(),
        sample=sample,
        synthesis_sha256=synthesis_sha,
        runtime=runtime,
    )

    assert len(runtime.calls) == 2
    assert result["format"] == (
        "memory-condense-recall-guarded-semantic-judge-score-v2"
    )
    assert result["semantic_judge_policy"]["format"] == (
        "memory-condense-independent-binary-semantic-judge-policy-v2"
    )
    assert result["logical_judgment_count"] == 3
    assert result["unique_judge_prompt_count"] == 2
    assert result["deduplicated_logical_judgment_count"] == 1
    assert [row["correct"] for row in result["stage_aggregates"]] == [1, 1, 0]
    assert all(
        row["status"] == "insufficient_population"
        for row in result["stage_aggregates"]
    )
    assert result["pooled_stage_question_accuracy"]["binary_accuracy"] == (
        2 / 3
    )
    assert result["target_gate"] == {
        "target_accuracy": 0.95,
        "minimum_questions_per_stage": 100,
        "gate_unit": "one fixed retrieval/synthesis stage",
        "eligible_stage_count": 0,
        "passing_stage_ids": [],
        "any_stage_passed": False,
        "responder_local_prompt_cap_status": "pass",
        "responder_provider_prompt_cap_status": "unavailable",
        "responder_output_reserve_protocol_eligible": True,
        "status": "not_passed",
    }
    prompt = result["responder_prompt_cap_diagnostics"]
    assert prompt["local_prompt_cap_status"] == "pass"
    assert prompt["all_responder_prompts_proven_within_local_cap"] is True
    assert prompt["provider_prompt_cap_status"] == "unavailable"
    assert prompt["logical_responder_rows"] == 3
    assert "structured_attempt_responder_rows" not in prompt
    assert "complete_responder_request_rows" not in prompt
    assert result["judge_usage"]["unique_journaled_calls"] == 2
    assert result["judge_usage"]["reported_input_tokens"] == 40
    reserve = result["responder_output_reserve_diagnostics"]
    assert reserve["protocol_eligible"] is True
    assert reserve["eligible_effective_answer_request_rows"] == 3


@pytest.mark.parametrize("authorized", [1, 3])
def test_campaign_preflight_requires_exact_unique_call_cap(authorized: int) -> None:
    synthesis = _synthesis()
    with pytest.raises(ValueError, match="exactly equal"):
        judge.build_semantic_judge_campaign_binding(
            synthesis,
            **_retrieval_binding(),
            sample=_sample(),
            synthesis_sha256=_digest(synthesis),
            authorized_unique_calls=authorized,
        )


@pytest.mark.parametrize("prompt_cap", [7_999, 8_000.0])
def test_campaign_preflight_requires_locked_responder_prompt_cap(
    prompt_cap: object,
) -> None:
    synthesis = _synthesis()
    with pytest.raises(ValueError, match="exactly equal.*8000"):
        judge.build_semantic_judge_campaign_binding(
            synthesis,
            **_retrieval_binding(),
            sample=_sample(),
            synthesis_sha256=_digest(synthesis),
            responder_prompt_cap=prompt_cap,  # type: ignore[arg-type]
            authorized_unique_calls=2,
        )


def test_non_sol_judge_model_is_rejected_before_calls() -> None:
    synthesis = _synthesis()
    sample = _sample()
    synthesis_sha = _digest(synthesis)
    binding = judge.build_semantic_judge_campaign_binding(
        synthesis,
        **_retrieval_binding(),
        sample=sample,
        synthesis_sha256=synthesis_sha,
        authorized_unique_calls=2,
    )
    runtime = _FakeRuntime(
        binding,
        model="openai/codex_sdk/gpt-5.6-terra",
    )

    with pytest.raises(ValueError, match="Sol judge route"):
        judge.judge_recall_guarded_cumulative_synthesis(
            synthesis,
            **_retrieval_binding(),
            sample=sample,
            synthesis_sha256=synthesis_sha,
            runtime=runtime,
        )
    assert runtime.calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("caller_model", judge.LOCKED_JUDGE_MODEL),
        ("gateway_model", judge.LOCKED_JUDGE_GATEWAY_MODEL),
        ("gateway_url", "https://wrong-gateway.invalid/v1"),
        ("format", "wrong-responder-runtime"),
        ("retries", 1),
        ("temperature", 0.0),
    ],
)
def test_responder_route_attestation_rejects_conflicts(
    field: str,
    value: Any,
) -> None:
    synthesis = _synthesis()
    synthesis["runtime_identity"][field] = value

    with pytest.raises(ValueError, match="zero-retry Terra responder route"):
        judge.build_semantic_judge_campaign_binding(
            synthesis,
            **_retrieval_binding(),
            sample=_sample(),
            synthesis_sha256=_digest(synthesis),
            authorized_unique_calls=2,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("caller_model", judge.LOCKED_RESPONDER_MODEL),
        ("gateway_model", judge.LOCKED_RESPONDER_GATEWAY_MODEL),
        ("gateway_url", "https://wrong-gateway.invalid/v1"),
        ("format", "wrong-judge-runtime"),
        ("retries", 1),
        ("temperature", 0.0),
        (
            "default_max_new_tokens",
            judge.LOCKED_JUDGE_MAX_NEW_TOKENS + 1,
        ),
    ],
)
def test_judge_route_attestation_rejects_before_calls(
    field: str,
    value: Any,
) -> None:
    synthesis = _synthesis()
    sample = _sample()
    synthesis_sha = _digest(synthesis)
    binding = judge.build_semantic_judge_campaign_binding(
        synthesis,
        **_retrieval_binding(),
        sample=sample,
        synthesis_sha256=synthesis_sha,
        authorized_unique_calls=2,
    )
    runtime = _FakeRuntime(binding)
    runtime.identity[field] = value

    with pytest.raises(ValueError, match="zero-retry Sol judge route"):
        judge.judge_recall_guarded_cumulative_synthesis(
            synthesis,
            **_retrieval_binding(),
            sample=sample,
            synthesis_sha256=synthesis_sha,
            runtime=runtime,
        )
    assert runtime.calls == []


def test_canonical_synthesis_sha_and_population_are_fail_closed() -> None:
    synthesis = _synthesis()
    with pytest.raises(ValueError, match="canonical bytes"):
        judge.build_semantic_judge_campaign_binding(
            synthesis,
            **_retrieval_binding(),
            sample=_sample(),
            synthesis_sha256="0" * 64,
            authorized_unique_calls=2,
        )

    synthesis["population_identity_sha256"] = "x" * 64
    with pytest.raises(ValueError, match="population identities differ"):
        judge.build_semantic_judge_campaign_binding(
            synthesis,
            **_retrieval_binding(),
            sample=_sample(),
            synthesis_sha256=_digest(synthesis),
            authorized_unique_calls=2,
        )


def test_prompt_cap_violation_fails_before_judge_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        judge,
        "count_chat_prompt_token_proxy",
        lambda _messages: 8_001,
    )
    questions = [
        BenchmarkQuestion(
            question_id=f"q{index}",
            question="What color?",
            answer="blue",
        )
        for index in range(100)
    ]
    sample = BenchmarkSample(sample_id="gate", questions=questions)
    retrieval = {
        "format": "test-retrieval",
        "population_identity_sha256": _POPULATION_SHA,
        "questions": [],
    }
    synthesis = {
        "format": "test-synthesis",
        "population_identity_sha256": _POPULATION_SHA,
        "synthesis_prompt_policy": SYNTHESIS_PROMPT_POLICY,
        "request_policy": {
            "max_new_tokens": judge.BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE
        },
        "runtime_identity": _responder_runtime_identity(),
        "questions": [],
    }
    for ordinal, question in enumerate(questions):
        retrieval_question = copy.deepcopy(_retrieval()["questions"][0])
        retrieval_question["ordinal"] = ordinal
        retrieval_question["question_id"] = question.question_id
        retrieval["questions"].append(retrieval_question)
        stages = [
            _stage(stage_id, "blue")
            for stage_id in judge.SYNTHESIS_STAGE_IDS
        ]
        for source_stage, stage in zip(
            retrieval_question["stages"][1:], stages, strict=True
        ):
            messages, _aliases, _novel = build_synthesis_messages(
                source_stage,
                root_evidence_ids={"root"},
                prompt_policy=SYNTHESIS_PROMPT_POLICY,
            )
            messages_sha = identity_sha256(messages)
            stage["prompt_messages_sha256"] = messages_sha
            stage["completion_report"]["messages_sha256"] = messages_sha
            stage["completion_report"]["input_token_proxy"] = 8_001
        synthesis["questions"].append(
            {
                "ordinal": ordinal,
                "question_id": question.question_id,
                "question_sha256": quote_sha256(question.question),
                "stages": stages,
            }
        )
    retrieval_sha = _digest(retrieval)
    synthesis["retrieval_sha256"] = retrieval_sha
    synthesis_sha = _digest(synthesis)
    with pytest.raises(ValueError, match="before judge authorization"):
        judge.build_semantic_judge_campaign_binding(
            synthesis,
            retrieval=retrieval,
            sample=sample,
            synthesis_sha256=synthesis_sha,
            retrieval_sha256=retrieval_sha,
            authorized_unique_calls=1,
        )


def _fallback_with_structured_attempt() -> tuple[
    dict[str, Any], dict[str, Any], list[dict[str, str]]
]:
    retrieval = _retrieval()
    synthesis = _synthesis()
    source_stage = retrieval["questions"][0]["stages"][1]
    source_stage["evidence"] = copy.deepcopy(source_stage["evidence"])
    source_stage["evidence"][1]["text"] += " This is the direct-stage copy."
    stage = synthesis["questions"][0]["stages"][0]
    structured_messages, _aliases, _novel = build_synthesis_messages(
        source_stage,
        root_evidence_ids={"root"},
        prompt_policy=SYNTHESIS_PROMPT_POLICY,
    )
    structured_sha = identity_sha256(structured_messages)
    fallback_messages = [dict(row) for row in source_stage["provider_messages"]]
    fallback_sha = identity_sha256(fallback_messages)
    stage["synthesis_mode"] = "short_answer_with_forced_choice_attribution"
    stage["prompt_messages_sha256"] = fallback_sha
    stage["completion_report"] = {
        "call_key_sha256": "d" * 64,
        "messages_sha256": fallback_sha,
        "input_token_proxy": count_chat_prompt_token_proxy(fallback_messages),
        "reported_input_tokens_available": False,
        "reported_input_tokens": 0,
        "max_new_tokens": judge.BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE,
    }
    stage["structured_attempt"] = {
        "prompt_messages_sha256": structured_sha,
        "completion_report": {
            "call_key_sha256": "e" * 64,
            "messages_sha256": structured_sha,
            "input_token_proxy": count_chat_prompt_token_proxy(
                structured_messages
            ),
            "reported_input_tokens_available": False,
            "reported_input_tokens": 0,
            "max_new_tokens": (
                judge.BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE
            ),
        },
    }
    synthesis["retrieval_sha256"] = _digest(retrieval)
    return synthesis, retrieval, structured_messages


def test_structured_attempt_prompt_is_included_in_cap_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthesis, retrieval, structured_messages = _fallback_with_structured_attempt()
    structured_sha = identity_sha256(structured_messages)
    original_count = count_chat_prompt_token_proxy

    def selective_count(messages) -> int:
        if identity_sha256(list(messages)) == structured_sha:
            return 8_001
        return original_count(messages)

    monkeypatch.setattr(judge, "count_chat_prompt_token_proxy", selective_count)
    attempt = synthesis["questions"][0]["stages"][0]["structured_attempt"]
    attempt["completion_report"]["input_token_proxy"] = 8_001
    diagnostics = judge._responder_prompt_diagnostics(
        synthesis,
        retrieval=retrieval,
        prompt_cap=8_000,
    )

    assert diagnostics["logical_responder_rows"] == 4
    assert diagnostics["structured_attempt_responder_rows"] == 1
    assert diagnostics["complete_responder_request_rows"] == 4
    assert diagnostics["local_prompt_cap_violation_count"] == 1
    attempt_rows = [
        row
        for row in diagnostics["rows"]
        if row.get("request_kind") == "structured_attempt"
    ]
    assert len(attempt_rows) == 1
    assert attempt_rows[0]["local_prompt_cap_compliant"] is False

    retrieval_sha = _digest(retrieval)
    synthesis_sha = _digest(synthesis)
    with pytest.raises(ValueError, match="before judge authorization"):
        judge.build_semantic_judge_campaign_binding(
            synthesis,
            retrieval=retrieval,
            sample=_sample(),
            synthesis_sha256=synthesis_sha,
            retrieval_sha256=retrieval_sha,
            authorized_unique_calls=2,
        )


def test_structured_attempt_missing_report_fails_closed() -> None:
    synthesis, retrieval, _messages = _fallback_with_structured_attempt()
    synthesis["questions"][0]["stages"][0]["structured_attempt"].pop(
        "completion_report"
    )

    with pytest.raises(ValueError, match="completion report is missing"):
        judge._responder_prompt_diagnostics(
            synthesis,
            retrieval=retrieval,
            prompt_cap=8_000,
        )


def test_4096_responder_scores_remain_diagnostic_but_gate_is_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(judge, "MINIMUM_GATE_QUESTIONS", 1)
    synthesis = _synthesis()
    synthesis["runtime_identity"]["default_max_new_tokens"] = 4_096
    synthesis["request_policy"]["max_new_tokens"] = 4_096
    for stage in synthesis["questions"][0]["stages"]:
        stage["request_policy"]["max_new_tokens"] = 4_096
        stage["completion_report"]["max_new_tokens"] = 4_096
    sample = _sample()
    synthesis_sha = _digest(synthesis)
    binding = judge.build_semantic_judge_campaign_binding(
        synthesis,
        **_retrieval_binding(),
        sample=sample,
        synthesis_sha256=synthesis_sha,
        authorized_unique_calls=2,
    )
    assert binding["responder_output_reserve_protocol_eligible"] is False
    runtime = _FakeRuntime(binding)

    result = judge.judge_recall_guarded_cumulative_synthesis(
        synthesis,
        **_retrieval_binding(),
        sample=sample,
        synthesis_sha256=synthesis_sha,
        runtime=runtime,
    )

    assert [row["correct"] for row in result["stage_aggregates"]] == [1, 1, 0]
    assert all(
        row["gate_passed"] is False
        and row["responder_output_reserve_protocol_eligible"] is False
        and row["status"]
        == "protocol_ineligible_responder_output_reserve"
        for row in result["stage_aggregates"]
    )
    reserve = result["responder_output_reserve_diagnostics"]
    assert reserve["required_responder_output_token_reserve"] == 256
    assert reserve["runtime_default_max_new_tokens"] == 4_096
    assert reserve["eligible_effective_answer_request_rows"] == 0
    assert reserve["protocol_eligible"] is False
    assert result["target_gate"]["status"] == "protocol_ineligible"
    assert result["target_gate"]["eligible_stage_count"] == 0
    assert result["target_gate"]["any_stage_passed"] is False
