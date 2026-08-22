from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
)
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval import recall_guarded_cumulative_final_answer as final_answer
from memory_condense.eval._recall_guarded_cumulative_contracts import (
    CausalCoveragePredecessorReceipt,
    CumulativeRetrievalLadder,
    CumulativeRetrievalStageReceipt,
    RecallGuardedCumulativeReceipt,
)
from memory_condense.eval.recall_guarded_cumulative_1m import (
    QUESTION_FORMAT,
    RETRIEVAL_FORMAT,
    STAGE_IDS,
    _canonical_json_bytes,
)
from memory_condense.eval.benchmark import QA_SYSTEM_PROMPT, QA_USER_TEMPLATE
from memory_condense.eval.recall_guarded_cumulative_final_answer import (
    FINAL_ANSWER_FORMAT,
    FINAL_ANSWER_RUNTIME_FORMAT,
    FIXED_STAGE_ID,
    LOCKED_RESPONDER_GATEWAY_MODEL,
    LOCKED_RESPONDER_MODEL,
    RESPONDER_OUTPUT_TOKEN_RESERVE,
    RESPONDER_PROMPT_CAP,
    answer_recall_guarded_cumulative_stage,
    build_final_answer_campaign_binding,
    build_responder_prompt_policy_identity,
    final_answer_prompt_population,
    validate_final_answer_artifact,
)
from memory_condense.eval.recall_guarded_cumulative_final_answer_runtime import (
    RecallGuardedCumulativeFinalAnswerRuntime,
    preflight_final_answer_prompt_population,
)
from memory_condense.eval.recall_guarded_cumulative_provider_synthesis_runtime import (
    CENTRAL_DEV_GATEWAY_URL,
)
from memory_condense.eval.recall_guarded_cumulative_validation_retrieval import (
    VALIDATION_MERGED_RETRIEVAL_FORMAT,
)


_H = "a" * 64
_QUESTION = "Which restaurant did I choose?"


def _messages(
    question: str,
    *,
    filler: str = "",
    system: str = QA_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": QA_USER_TEMPLATE.format(
                context=filler,
                question=question,
            ),
        },
    ]


def _question(
    ordinal: int,
    *,
    population_sha: str,
    store_sha: str,
    implementation_sha: str,
    over_cap: bool = False,
) -> dict[str, Any]:
    question_id = f"question-{ordinal}"
    question = f"{_QUESTION} ({ordinal})"
    filler = "evidence " * 9_000 if over_cap else "I chose Miss Bee Providore."
    messages = _messages(question, filler=filler)
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    prompt_cap = max(RESPONDER_PROMPT_CAP, prompt_tokens)
    evidence_rows = [
        {"evidence_id": "e0", "source_id": "s0", "text": "A shortlist."},
        {
            "evidence_id": "e1",
            "source_id": "s1",
            "text": "I chose Miss Bee Providore.",
        },
    ]
    typed_stages: list[CumulativeRetrievalStageReceipt] = []
    stage_rows: list[dict[str, Any]] = []
    for index, stage_id in enumerate(STAGE_IDS):
        selected = ("e0",) if index == 0 else ("e0", "e1")
        parent = () if index == 0 else typed_stages[-1].selected_evidence_ids
        added = selected if index == 0 else selected[len(parent) :]
        status = "root" if index == 0 else "added" if added else "no_novel_evidence"
        receipt = CumulativeRetrievalStageReceipt(
            stage_id=stage_id,
            matched_controls_sha256=_H,
            method_evidence_sha256=hashlib.sha256(stage_id.encode()).hexdigest(),
            parent_stage_receipt_sha256=(
                None if index == 0 else typed_stages[-1].receipt_sha256
            ),
            parent_evidence_ids=parent,
            selected_evidence_ids=selected,
            added_evidence_ids=added,
            admission_status=status,
            evidence_projection_sha256=hashlib.sha256(
                (stage_id + "projection").encode()
            ).hexdigest(),
            context_sha256=hashlib.sha256((stage_id + "context").encode()).hexdigest(),
            prompt_messages_sha256=identity_sha256(messages),
            context_token_proxy=2,
            max_context_token_proxy=7_000,
            prompt_token_proxy=prompt_tokens,
            max_prompt_token_proxy=prompt_cap,
            responder_output_token_reserve=RESPONDER_OUTPUT_TOKEN_RESERVE,
        )
        typed_stages.append(receipt)
        stage_rows.append(
            {
                "stage_id": stage_id,
                "stage_receipt": asdict(receipt),
                "provider_messages": copy.deepcopy(messages),
                "evidence": copy.deepcopy(evidence_rows[: len(selected)]),
            }
        )
    ladder = CumulativeRetrievalLadder(stages=tuple(typed_stages))
    predecessor = CausalCoveragePredecessorReceipt(
        matched_controls_sha256=_H,
        retrieval_query_sha256=_H,
        prompt_question_sha256=quote_sha256(question),
        retrieval_policy_sha256=_H,
        context_budget_sha256=_H,
        raw_graph_anchor_sequence_sha256=_H,
        raw_graph_chunk_ids=("c0",),
        packed_chunk_ids=("c0",),
        protected_chunk_ids=("c0",),
        direct_protected_chunk_ids=("c0",),
        protected_excerpt_projection_sha256=_H,
        protected_context_sha256=_H,
        selected_anchor_sequence_sha256=_H,
        coverage_selector_report_sha256=_H,
        coverage_candidate_trace_sha256=_H,
        coverage_runtime_certified=True,
        packed_token_counts=(),
        packed_dropped_counts=(),
        prompt_messages_sha256=typed_stages[0].prompt_messages_sha256,
        prompt_token_proxy=prompt_tokens,
        max_prompt_token_proxy=prompt_cap,
        responder_output_token_reserve=RESPONDER_OUTPUT_TOKEN_RESERVE,
    )
    final_receipt = RecallGuardedCumulativeReceipt(
        matched_controls_sha256=_H,
        predecessor_receipt_sha256=predecessor.receipt_sha256,
        direct_expansion_receipt_sha256=_H,
        representative_expansion_receipt_sha256=_H,
        closure_plan_sha256s=(_H, _H, _H),
        novel_projection_receipt_sha256s=(_H, _H, _H),
        addition_packet_receipt_sha256s=(_H, None, None),
        stage_admission_statuses=("added", "no_novel_evidence", "no_novel_evidence"),
        ladder_receipt_sha256=ladder.receipt_sha256,
        representative_runtime_certified=True,
        protected_chunk_ids=("c0",),
        protected_evidence_ids=("e0",),
        added_atom_ids=("e1",),
        added_chunk_ids=("c1",),
        final_chunk_ids=("c0", "c1"),
        final_evidence_ids=("e0", "e1"),
        protected_excerpt_projection_sha256=_H,
        addition_evidence_projection_sha256=_H,
        final_context_sha256=_H,
        prompt_messages_sha256=typed_stages[-1].prompt_messages_sha256,
        context_token_proxy=2,
        max_context_token_proxy=7_000,
        prompt_token_proxy=prompt_tokens,
        max_prompt_token_proxy=prompt_cap,
        responder_output_token_reserve=RESPONDER_OUTPUT_TOKEN_RESERVE,
        prompt_workspace_token_proxy=(
            prompt_tokens + RESPONDER_OUTPUT_TOKEN_RESERVE
        ),
    )
    return {
        "format": QUESTION_FORMAT,
        "population_identity_sha256": population_sha,
        "ordinal": ordinal,
        "question_id": question_id,
        "question_sha256": quote_sha256(question),
        "dated_question_sha256": quote_sha256(question),
        "combined_store_receipt_sha256": store_sha,
        "retrieval_implementation_sha256": implementation_sha,
        "retrieval_receipt": asdict(final_receipt),
        "predecessor_receipt": asdict(predecessor),
        "stage_ids": list(STAGE_IDS),
        "stages": stage_rows,
        "provider_calls": 0,
    }


def _retrieval(*, count: int = 1, over_cap_last: bool = False) -> dict[str, Any]:
    population_sha = "b" * 64
    store_sha = "c" * 64
    implementation_sha = "d" * 64
    questions = [
        _question(
            ordinal,
            population_sha=population_sha,
            store_sha=store_sha,
            implementation_sha=implementation_sha,
            over_cap=over_cap_last and ordinal == count - 1,
        )
        for ordinal in range(count)
    ]
    return {
        "format": RETRIEVAL_FORMAT,
        "gold_fields_present": False,
        "population_identity_sha256": population_sha,
        "retrieval_implementation_sha256": implementation_sha,
        "combined_store_receipt": {"receipt_sha256": store_sha},
        "stage_ids": list(STAGE_IDS),
        "question_count": len(questions),
        "question_part_sha256s": [
            hashlib.sha256(_canonical_json_bytes(row)).hexdigest()
            for row in questions
        ],
        "questions": questions,
    }


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _synthetic_merged_retrieval() -> dict[str, Any]:
    """Small merged-shape fixture; the closed validator is mocked in tests."""

    retrieval = _retrieval(count=2)
    first_store_sha = str(
        retrieval["questions"][0]["combined_store_receipt_sha256"]
    )
    second_store_sha = "9" * 64
    retrieval["format"] = VALIDATION_MERGED_RETRIEVAL_FORMAT
    retrieval.pop("combined_store_receipt")
    retrieval["questions"][1][
        "combined_store_receipt_sha256"
    ] = second_store_sha
    retrieval["question_part_sha256s"] = [
        hashlib.sha256(_canonical_json_bytes(row)).hexdigest()
        for row in retrieval["questions"]
    ]
    retrieval["shards"] = [
        {
            "combined_store_receipt_sha256": first_store_sha,
            "question_count": 1,
        },
        {
            "combined_store_receipt_sha256": second_store_sha,
            "question_count": 1,
        },
    ]
    retrieval["external_reconstruction_receipt_sha256"] = "8" * 64
    return retrieval


class _Runtime:
    def __init__(self, retrieval: Mapping[str, Any], retrieval_sha: str) -> None:
        prompts = final_answer_prompt_population(
            retrieval,
            retrieval_sha256=retrieval_sha,
        )
        unique = len({identity_sha256(list(row)) for row in prompts})
        prompt_population = preflight_final_answer_prompt_population(
            prompts,
            authorized_unique_calls=unique,
        )
        campaign = build_final_answer_campaign_binding(
            retrieval,
            retrieval_sha256=retrieval_sha,
            authorized_unique_calls=unique,
        )
        self.identity = {
            "format": FINAL_ANSWER_RUNTIME_FORMAT,
            "gateway_url": CENTRAL_DEV_GATEWAY_URL,
            "caller_model": LOCKED_RESPONDER_MODEL,
            "gateway_model": LOCKED_RESPONDER_GATEWAY_MODEL,
            "default_max_new_tokens": RESPONDER_OUTPUT_TOKEN_RESERVE,
            "max_prompt_token_proxy": RESPONDER_PROMPT_CAP,
            "retries": 0,
            "temperature": None,
            "authorized_unique_calls": unique,
            "logical_prompt_count": prompt_population.logical_prompt_count,
            "unique_prompt_count": prompt_population.unique_prompt_count,
            "prompt_population_sha256": (
                prompt_population.prompt_population_sha256
            ),
            "prompt_token_proxy_identity": dict(
                prompt_population.prompt_token_proxy_identity
            ),
            "persisted_request_token_state": False,
            "retained_request_token_state_bytes": 0,
            "external_provider_persistence_certified": False,
            "campaign_binding": campaign,
            "campaign_binding_sha256": identity_sha256(campaign),
        }
        self.last_journal_record: Mapping[str, Any] | None = None
        self.calls = 0

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        self.calls += 1
        answer = "Miss Bee Providore"
        messages_sha = identity_sha256(list(messages))
        call_key = hashlib.sha256(messages_sha.encode()).hexdigest()
        runtime_sha = identity_sha256(self.identity)
        campaign_sha = self.identity["campaign_binding_sha256"]
        report = {
            "gateway_url": CENTRAL_DEV_GATEWAY_URL,
            "caller_model": LOCKED_RESPONDER_MODEL,
            "gateway_model": LOCKED_RESPONDER_GATEWAY_MODEL,
            "call_key_sha256": call_key,
            "request_journal_sha256": "e" * 64,
            "messages_sha256": messages_sha,
            "completion_sha256": quote_sha256(answer),
            "runtime_identity_sha256": runtime_sha,
            "campaign_binding_sha256": campaign_sha,
            "prompt_population_sha256": self.identity[
                "prompt_population_sha256"
            ],
            "max_new_tokens": max_new_tokens,
            "max_prompt_token_proxy": RESPONDER_PROMPT_CAP,
            "response_id": "fake-response",
            "response_model": LOCKED_RESPONDER_GATEWAY_MODEL,
            "finish_reason": "stop",
            "reported_usage_available": False,
            "input_token_proxy": count_chat_prompt_token_proxy(messages),
            "output_token_proxy": count_tokens(answer),
            "reported_input_tokens_available": False,
            "reported_input_tokens": 0,
            "reported_output_tokens_available": False,
            "reported_output_tokens": 0,
            "reported_total_tokens_available": False,
            "reported_total_tokens": 0,
            "elapsed_s": 0.1,
            "retries": 0,
            "cache_hit": False,
            "physical_call": True,
            "cumulative_logical_calls": self.calls,
            "cumulative_unique_calls": self.calls,
            "cumulative_physical_calls": self.calls,
            "cumulative_checkpoint_hits": 0,
        }
        self.last_journal_record = {
            "call_key_sha256": call_key,
            "request_journal_sha256": "e" * 64,
            "response_journal_sha256": "f" * 64,
            "completion_sha256": quote_sha256(answer),
            "completion_report": report,
        }
        return answer


def test_fixed_stage_answer_is_gold_blind_sealed_and_validatable() -> None:
    retrieval = _retrieval()
    retrieval_sha = _digest(retrieval)
    runtime = _Runtime(retrieval, retrieval_sha)

    artifact = answer_recall_guarded_cumulative_stage(
        retrieval,
        retrieval_sha256=retrieval_sha,
        runtime=runtime,
    )

    assert artifact["format"] == FINAL_ANSWER_FORMAT
    assert artifact["gold_fields_present"] is False
    assert artifact["fixed_stage_id"] == FIXED_STAGE_ID
    assert artifact["question_count"] == 1
    assert artifact["questions"][0]["answer"]["text"] == "Miss Bee Providore"
    assert artifact["responder_usage"]["unique_journaled_calls"] == 1
    assert artifact["local_transformer_state_receipt"][
        "after_provider_calls_bytes"
    ] == 0
    assert artifact["external_provider_persistence"] == "not_certified"
    assert artifact["responder_prompt_policy_sha256"] == artifact[
        "campaign_binding"
    ]["responder_prompt_policy_sha256"]
    assert identity_sha256(
        artifact["campaign_binding"]["responder_prompt_policy"]
    ) == artifact["responder_prompt_policy_sha256"]
    assert runtime.calls == 1
    validate_final_answer_artifact(
        artifact,
        retrieval=retrieval,
        artifact_sha256=_digest(artifact),
        retrieval_sha256=retrieval_sha,
    )


def test_late_over_cap_prompt_fails_before_first_runtime_call() -> None:
    valid = _retrieval()
    valid_sha = _digest(valid)
    runtime = _Runtime(valid, valid_sha)
    retrieval = _retrieval(count=2, over_cap_last=True)

    with pytest.raises(ValueError, match="frozen responder budget"):
        answer_recall_guarded_cumulative_stage(
            retrieval,
            retrieval_sha256=_digest(retrieval),
            runtime=runtime,
        )

    assert runtime.calls == 0


def test_merged_validation_uses_closed_validator_and_binds_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval = _synthetic_merged_retrieval()
    retrieval_sha = _digest(retrieval)
    validated: list[Mapping[str, Any]] = []
    resolved = (
        {"receipt_sha256": "c" * 64},
        {"receipt_sha256": "9" * 64},
    )
    monkeypatch.setattr(
        final_answer,
        "merged_question_store_receipts",
        lambda value: (validated.append(value), resolved)[1],
    )

    prompts = final_answer_prompt_population(
        retrieval,
        retrieval_sha256=retrieval_sha,
    )
    campaign = build_final_answer_campaign_binding(
        retrieval,
        retrieval_sha256=retrieval_sha,
        authorized_unique_calls=2,
    )

    assert len(prompts) == 2
    assert validated == [retrieval, retrieval]
    assert campaign["external_reconstruction_receipt_sha256"] == "8" * 64


def test_merged_validation_preserves_per_shard_store_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval = _synthetic_merged_retrieval()
    retrieval["questions"][1]["combined_store_receipt_sha256"] = "c" * 64
    retrieval["question_part_sha256s"] = [
        hashlib.sha256(_canonical_json_bytes(row)).hexdigest()
        for row in retrieval["questions"]
    ]
    monkeypatch.setattr(
        final_answer,
        "merged_question_store_receipts",
        lambda _value: (
            {"receipt_sha256": "c" * 64},
            {"receipt_sha256": "9" * 64},
        ),
    )

    with pytest.raises(ValueError, match="cross-binding changed"):
        final_answer_prompt_population(
            retrieval,
            retrieval_sha256=_digest(retrieval),
        )


def test_historical_retrieval_does_not_enter_merged_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval = _retrieval()
    monkeypatch.setattr(
        final_answer,
        "merged_question_store_receipts",
        lambda _value: pytest.fail("historical path entered merged validator"),
    )

    prompts = final_answer_prompt_population(
        retrieval,
        retrieval_sha256=_digest(retrieval),
    )

    assert len(prompts) == 1


def test_responder_prompt_policy_seals_system_but_not_context_or_question() -> None:
    first = build_responder_prompt_policy_identity(
        [_messages("Question one?", filler="Context one.")]
    )
    second = build_responder_prompt_policy_identity(
        [_messages("Question two?", filler="Different context.")]
    )
    historical = build_responder_prompt_policy_identity(
        [
            _messages(
                "Question one?",
                filler="Context one.",
                system="Historical responder instructions.",
            )
        ]
    )

    assert first == second
    assert identity_sha256(first) != identity_sha256(historical)
    assert first["ordered_unique_system_content_quote_sha256s"] == [
        quote_sha256(QA_SYSTEM_PROMPT)
    ]


def test_responder_prompt_policy_rejects_noncanonical_user_framing() -> None:
    messages = _messages("Question one?", filler="Context one.")
    messages[1]["content"] = messages[1]["content"].replace(
        "Short answer:",
        "Answer:",
    )

    with pytest.raises(ValueError, match="QA framing"):
        build_responder_prompt_policy_identity([messages])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("caller_model", "openai/codex_sdk/gpt-5.6-sol"),
        ("default_max_new_tokens", 4_096),
        ("max_prompt_token_proxy", 8_001),
        ("retries", 1),
        ("temperature", 0.0),
        ("persisted_request_token_state", True),
    ],
)
def test_runtime_route_budget_and_state_are_hard_locked(
    field: str,
    value: object,
) -> None:
    retrieval = _retrieval()
    retrieval_sha = _digest(retrieval)
    runtime = _Runtime(retrieval, retrieval_sha)
    runtime.identity[field] = value

    with pytest.raises(ValueError, match="frozen zero-retry Terra runtime"):
        answer_recall_guarded_cumulative_stage(
            retrieval,
            retrieval_sha256=retrieval_sha,
            runtime=runtime,
        )

    assert runtime.calls == 0


def test_artifact_tampering_is_rejected() -> None:
    retrieval = _retrieval()
    retrieval_sha = _digest(retrieval)
    artifact = answer_recall_guarded_cumulative_stage(
        retrieval,
        retrieval_sha256=retrieval_sha,
        runtime=_Runtime(retrieval, retrieval_sha),
    )
    artifact["questions"][0]["fixed_stage_id"] = STAGE_IDS[2]

    with pytest.raises(ValueError, match="question binding changed"):
        validate_final_answer_artifact(
            artifact,
            retrieval=retrieval,
            artifact_sha256=_digest(artifact),
            retrieval_sha256=retrieval_sha,
        )


class _Completions:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> Any:
        self.requests.append(request)
        return SimpleNamespace(
            id="terra-answer",
            model=LOCKED_RESPONDER_GATEWAY_MODEL,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Miss Bee Providore"),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )


class _Client:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_Completions())

    def close(self) -> None:
        return None


def test_live_then_fresh_runtime_replay_produces_identical_artifact(
    tmp_path: Path,
) -> None:
    retrieval = _retrieval()
    retrieval_sha = _digest(retrieval)
    prompts = final_answer_prompt_population(
        retrieval,
        retrieval_sha256=retrieval_sha,
    )
    campaign = build_final_answer_campaign_binding(
        retrieval,
        retrieval_sha256=retrieval_sha,
        authorized_unique_calls=1,
    )
    checkpoint = tmp_path / "final-answer-calls"
    client = _Client()
    with RecallGuardedCumulativeFinalAnswerRuntime(
        checkpoint_dir=checkpoint,
        campaign_binding=campaign,
        prompt_population=prompts,
        authorized_unique_calls=1,
        api_key="ephemeral-test-secret",
        client=client,
    ) as runtime:
        live = answer_recall_guarded_cumulative_stage(
            retrieval,
            retrieval_sha256=retrieval_sha,
            runtime=runtime,
        )
    assert len(client.chat.completions.requests) == 1
    with RecallGuardedCumulativeFinalAnswerRuntime(
        checkpoint_dir=checkpoint,
        campaign_binding=campaign,
        prompt_population=prompts,
        authorized_unique_calls=1,
        replay_only=True,
    ) as runtime:
        replay = answer_recall_guarded_cumulative_stage(
            retrieval,
            retrieval_sha256=retrieval_sha,
            runtime=runtime,
        )
        assert runtime.usage["physical_calls"] == 0
        assert runtime.usage["checkpoint_hits"] == 1
    assert _canonical_json_bytes(live) == _canonical_json_bytes(replay)
