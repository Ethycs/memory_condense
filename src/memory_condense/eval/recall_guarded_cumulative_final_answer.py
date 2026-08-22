"""Gold-blind fixed-stage answers for sealed cumulative retrieval artifacts.

This module is deliberately separate from the experimental S1--S3 synthesis
pipeline.  It sends exactly one preregistered retrieval stage to the frozen
benchmark responder, with a 256-token output allowance, after validating the
complete prompt population and its hard 8,000-token local proxy cap.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, Protocol

from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    tokenizer_proxy_identity,
)
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval._recall_guarded_cumulative_contracts import (
    CausalCoveragePredecessorReceipt,
    CumulativeRetrievalLadder,
    CumulativeRetrievalStageReceipt,
    RecallGuardedCumulativeReceipt,
)
from memory_condense.eval._recall_guarded_cumulative_synthesis_contracts import (
    extract_stage_question,
    validate_published_retrieval,
)
from memory_condense.eval.benchmark import (
    BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE,
    QA_USER_TEMPLATE,
)
from memory_condense.eval.recall_guarded_cumulative_1m import (
    STAGE_IDS,
    _canonical_json_bytes,
)
from memory_condense.eval.recall_guarded_cumulative_final_answer_runtime import (
    FINAL_ANSWER_RUNTIME_FORMAT,
    FinalAnswerCompletionReport,
    LOCKED_FINAL_ANSWER_GATEWAY_MODEL,
    LOCKED_FINAL_ANSWER_MAX_NEW_TOKENS,
    LOCKED_FINAL_ANSWER_MAX_PROMPT_TOKENS,
    LOCKED_FINAL_ANSWER_MODEL,
    preflight_final_answer_prompt_population,
)
from memory_condense.eval.recall_guarded_cumulative_provider_synthesis_runtime import (
    CENTRAL_DEV_GATEWAY_URL,
    _gateway_model,
)
from memory_condense.eval.recall_guarded_cumulative_validation_retrieval import (
    VALIDATION_MERGED_RETRIEVAL_FORMAT,
    merged_question_store_receipts,
)
from memory_condense.eval.reproducibility import implementation_sha256


FINAL_ANSWER_FORMAT = (
    "memory-condense-recall-guarded-fixed-stage-final-answers-v1"
)
FINAL_ANSWER_QUESTION_FORMAT = (
    "memory-condense-recall-guarded-fixed-stage-final-answer-question-v1"
)
FINAL_ANSWER_CAMPAIGN_FORMAT = (
    "memory-condense-recall-guarded-fixed-stage-final-answer-campaign-v1"
)
RESPONDER_PROMPT_POLICY_FORMAT = (
    "memory-condense-sealed-qa-responder-prompt-policy-v1"
)
FIXED_STAGE_ID = "direct_episode_additions"
LOCKED_RESPONDER_MODEL = LOCKED_FINAL_ANSWER_MODEL
LOCKED_RESPONDER_GATEWAY_MODEL = LOCKED_FINAL_ANSWER_GATEWAY_MODEL
RESPONDER_PROMPT_CAP = LOCKED_FINAL_ANSWER_MAX_PROMPT_TOKENS
RESPONDER_OUTPUT_TOKEN_RESERVE = LOCKED_FINAL_ANSWER_MAX_NEW_TOKENS
if RESPONDER_OUTPUT_TOKEN_RESERVE != BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE:
    raise RuntimeError("final-answer runtime and benchmark output reserves differ")

FINAL_ANSWER_POLICY = {
    "format": "memory-condense-fixed-stage-final-answer-policy-v1",
    "input": "sealed retrieval provider_messages",
    "fixed_stage_id": FIXED_STAGE_ID,
    "stage_selection": "preregistered; never selected from answer quality",
    "gold_blind": True,
    "caller_model": LOCKED_RESPONDER_MODEL,
    "gateway_model": LOCKED_RESPONDER_GATEWAY_MODEL,
    "gateway_url": CENTRAL_DEV_GATEWAY_URL,
    "prompt_token_proxy_cap": RESPONDER_PROMPT_CAP,
    "prompt_token_proxy_identity": tokenizer_proxy_identity(),
    "output_token_reserve": RESPONDER_OUTPUT_TOKEN_RESERVE,
    "retries": 0,
    "temperature": None,
    "deduplication": "identical canonical provider messages",
    "persisted_local_transformer_token_state": False,
}
FINAL_ANSWER_POLICY_SHA256 = identity_sha256(FINAL_ANSWER_POLICY)


class FinalAnswerRuntime(Protocol):
    identity: Any
    last_journal_record: Mapping[str, Any] | None

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class _PlannedAnswer:
    ordinal: int
    question_id: str
    question_sha256: str
    dated_question_sha256: str
    question_part_sha256: str
    stage_receipt_sha256: str
    evidence_projection_sha256: str
    messages: tuple[Mapping[str, str], ...]
    messages_sha256: str
    prompt_token_proxy: int


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _canonical_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _runtime_identity(runtime: FinalAnswerRuntime) -> dict[str, Any]:
    value = runtime.identity
    if isinstance(value, Mapping):
        return {str(key): child for key, child in value.items()}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        payload = dump()
        if isinstance(payload, Mapping):
            return {str(key): child for key, child in payload.items()}
    raise TypeError("final-answer runtime has no mapping identity")


def _validated_messages(value: Any) -> tuple[Mapping[str, str], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("selected stage provider messages are missing")
    rows: list[Mapping[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != {"role", "content"}:
            raise ValueError(
                f"selected provider message {index} has a noncanonical shape"
            )
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not role or not isinstance(content, str):
            raise ValueError(
                f"selected provider message {index} requires string fields"
            )
        rows.append({"role": role, "content": content})
    return tuple(rows)


def build_responder_prompt_policy_identity(
    prompt_population: Sequence[Sequence[Mapping[str, str]]],
) -> dict[str, Any]:
    """Seal treatment-independent QA framing and actual system prompts."""

    if not isinstance(prompt_population, Sequence) or not prompt_population:
        raise ValueError("responder prompt population must not be empty")
    context_prefix, remainder = QA_USER_TEMPLATE.split("{context}", 1)
    question_marker, answer_suffix = remainder.split("{question}", 1)
    ordered_system_hashes: list[str] = []
    observed_system_content: dict[str, str] = {}
    for raw_messages in prompt_population:
        messages = _validated_messages(list(raw_messages))
        if (
            len(messages) != 2
            or messages[0].get("role") != "system"
            or messages[1].get("role") != "user"
        ):
            raise ValueError(
                "responder prompt must contain exactly one system and one "
                "user message"
            )
        system_content = str(messages[0]["content"])
        if not system_content:
            raise ValueError("responder system prompt must not be empty")
        system_sha = quote_sha256(system_content)
        previous = observed_system_content.setdefault(
            system_sha,
            system_content,
        )
        if previous != system_content:
            raise RuntimeError("responder system-prompt SHA-256 collision")
        if system_sha not in ordered_system_hashes:
            ordered_system_hashes.append(system_sha)

        user_content = str(messages[1]["content"])
        if not user_content.startswith(context_prefix):
            raise ValueError("responder user prompt changed its context framing")
        framed = user_content[len(context_prefix) :]
        if question_marker not in framed or not framed.endswith(answer_suffix):
            raise ValueError("responder user prompt changed its QA framing")
        context, question_with_suffix = framed.rsplit(question_marker, 1)
        question = question_with_suffix[: -len(answer_suffix)]
        if (
            not context
            or not question
            or question.strip() != question
            or QA_USER_TEMPLATE.format(context=context, question=question)
            != user_content
        ):
            raise ValueError("responder user prompt is not an exact QA template")

    return {
        "format": RESPONDER_PROMPT_POLICY_FORMAT,
        "message_roles": ["system", "user"],
        "ordered_unique_system_content_quote_sha256s": (
            ordered_system_hashes
        ),
        "qa_user_template_quote_sha256": quote_sha256(QA_USER_TEMPLATE),
        "qa_user_framing": {
            "context_prefix": context_prefix,
            "question_marker": question_marker,
            "answer_suffix": answer_suffix,
        },
        "prompt_token_proxy_identity": tokenizer_proxy_identity(),
        "responder_output_token_reserve": RESPONDER_OUTPUT_TOKEN_RESERVE,
    }


def _plan_answers(
    retrieval: Mapping[str, Any],
    *,
    retrieval_sha256: str,
) -> list[_PlannedAnswer]:
    """Validate every sealed prompt before returning any callable work."""

    merged_validation = (
        retrieval.get("format") == VALIDATION_MERGED_RETRIEVAL_FORMAT
    )
    if merged_validation:
        merged_store_receipts = merged_question_store_receipts(retrieval)
    else:
        # Preserve the historical single-store contract unchanged.  Unknown
        # formats deliberately fail through its exact validator.
        validate_published_retrieval(retrieval)
    if _canonical_digest(retrieval) != retrieval_sha256:
        raise ValueError("retrieval SHA-256 does not match canonical bytes")
    _require_sha256(retrieval_sha256, "retrieval SHA-256")
    population_sha = _require_sha256(
        retrieval.get("population_identity_sha256"),
        "retrieval population identity SHA-256",
    )
    retrieval_implementation = _require_sha256(
        retrieval.get("retrieval_implementation_sha256"),
        "retrieval implementation SHA-256",
    )
    questions = retrieval.get("questions")
    part_hashes = retrieval.get("question_part_sha256s")
    if not isinstance(questions, list) or not isinstance(part_hashes, list):
        raise ValueError("retrieval embedded question parts are incomplete")
    if len(part_hashes) != len(questions):
        raise ValueError("retrieval question-part digest population changed")
    if merged_validation:
        _require_sha256(
            retrieval.get("external_reconstruction_receipt_sha256"),
            "external reconstruction receipt SHA-256",
        )
        if len(merged_store_receipts) != len(questions):
            raise ValueError("merged retrieval shard/question counts differ")
        expected_store_receipts: list[str] = []
        for receipt in merged_store_receipts:
            if not isinstance(receipt, Mapping):
                raise ValueError("merged retrieval store receipt is invalid")
            expected_store_receipts.append(
                _require_sha256(
                    receipt.get("receipt_sha256"),
                    "per-shard combined-store receipt SHA-256",
                )
            )
    else:
        store_receipt = retrieval.get("combined_store_receipt")
        if not isinstance(store_receipt, Mapping):
            raise ValueError("retrieval combined-store receipt is missing")
        store_receipt_sha = _require_sha256(
            store_receipt.get("receipt_sha256"),
            "combined-store receipt SHA-256",
        )
        expected_store_receipts = [store_receipt_sha] * len(questions)

    planned: list[_PlannedAnswer] = []
    seen_ids: set[str] = set()
    for ordinal, (question, part_sha, expected_store_receipt) in enumerate(
        zip(questions, part_hashes, expected_store_receipts, strict=True)
    ):
        if not isinstance(question, Mapping):
            raise ValueError("retrieval question must be an object")
        question_id = question.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("retrieval question ID is missing")
        if question_id in seen_ids:
            raise ValueError("retrieval question IDs must be unique")
        seen_ids.add(question_id)
        question_sha = _require_sha256(
            question.get("question_sha256"), "question SHA-256"
        )
        dated_sha = _require_sha256(
            question.get("dated_question_sha256"),
            "dated question SHA-256",
        )
        observed_part_sha = hashlib.sha256(
            _canonical_json_bytes(question)
        ).hexdigest()
        if _require_sha256(part_sha, "question-part SHA-256") != observed_part_sha:
            raise ValueError("retrieval question-part digest changed")
        if (
            question.get("ordinal") != ordinal
            or question.get("population_identity_sha256") != population_sha
            or question.get("combined_store_receipt_sha256")
            != expected_store_receipt
            or question.get("retrieval_implementation_sha256")
            != retrieval_implementation
            or tuple(question.get("stage_ids", ())) != STAGE_IDS
            or question.get("provider_calls") != 0
        ):
            raise ValueError("retrieval question cross-binding changed")
        if any(
            name in question
            for name in ("answer", "gold", "gold_answer", "evidence_sources")
        ):
            raise ValueError("final-answer input contains a gold-bearing field")

        stages = question.get("stages")
        if not isinstance(stages, list) or len(stages) != len(STAGE_IDS):
            raise ValueError("retrieval cumulative stage population changed")
        typed_stages: list[CumulativeRetrievalStageReceipt] = []
        selected_stage: Mapping[str, Any] | None = None
        selected_receipt: CumulativeRetrievalStageReceipt | None = None
        for expected_stage_id, stage in zip(STAGE_IDS, stages, strict=True):
            if not isinstance(stage, Mapping) or stage.get("stage_id") != (
                expected_stage_id
            ):
                raise ValueError("retrieval cumulative stage order changed")
            raw_receipt = stage.get("stage_receipt")
            if not isinstance(raw_receipt, Mapping):
                raise ValueError("retrieval stage receipt is missing")
            typed = CumulativeRetrievalStageReceipt(**dict(raw_receipt))
            typed_stages.append(typed)
            evidence = stage.get("evidence")
            if not isinstance(evidence, list) or tuple(
                item.get("evidence_id")
                if isinstance(item, Mapping)
                else None
                for item in evidence
            ) != typed.selected_evidence_ids:
                raise ValueError("retrieval stage evidence coordinates changed")
            messages = _validated_messages(stage.get("provider_messages"))
            messages_sha = identity_sha256(list(messages))
            prompt_tokens = count_chat_prompt_token_proxy(messages)
            if (
                messages_sha != typed.prompt_messages_sha256
                or prompt_tokens != typed.prompt_token_proxy
            ):
                raise ValueError("retrieval stage prompt seal or token count changed")
            if (
                typed.max_prompt_token_proxy != RESPONDER_PROMPT_CAP
                or typed.responder_output_token_reserve
                != RESPONDER_OUTPUT_TOKEN_RESERVE
                or prompt_tokens > RESPONDER_PROMPT_CAP
            ):
                raise ValueError("retrieval stage violates the frozen responder budget")
            if expected_stage_id == FIXED_STAGE_ID:
                selected_stage = stage
                selected_receipt = typed
        ladder = CumulativeRetrievalLadder(stages=tuple(typed_stages))
        final_receipt = RecallGuardedCumulativeReceipt(
            **dict(question.get("retrieval_receipt", {}))
        )
        predecessor = CausalCoveragePredecessorReceipt(
            **dict(question.get("predecessor_receipt", {}))
        )
        if (
            final_receipt.ladder_receipt_sha256 != ladder.receipt_sha256
            or final_receipt.predecessor_receipt_sha256
            != predecessor.receipt_sha256
            or final_receipt.prompt_messages_sha256
            != typed_stages[-1].prompt_messages_sha256
        ):
            raise ValueError("retrieval question receipts no longer cross-bind")
        assert selected_stage is not None and selected_receipt is not None
        selected_messages = _validated_messages(
            selected_stage.get("provider_messages")
        )
        if quote_sha256(extract_stage_question(selected_stage)) != dated_sha:
            raise ValueError("sealed selected-stage question hash changed")
        planned.append(
            _PlannedAnswer(
                ordinal=ordinal,
                question_id=question_id,
                question_sha256=question_sha,
                dated_question_sha256=dated_sha,
                question_part_sha256=observed_part_sha,
                stage_receipt_sha256=selected_receipt.receipt_sha256,
                evidence_projection_sha256=(
                    selected_receipt.evidence_projection_sha256
                ),
                messages=selected_messages,
                messages_sha256=selected_receipt.prompt_messages_sha256,
                prompt_token_proxy=selected_receipt.prompt_token_proxy,
            )
        )
    if len(planned) != retrieval.get("question_count"):
        raise ValueError("final-answer prompt population is incomplete")
    return planned


def _unique_prompts(
    planned: Sequence[_PlannedAnswer],
) -> dict[str, tuple[Mapping[str, str], ...]]:
    unique: dict[str, tuple[Mapping[str, str], ...]] = {}
    for row in planned:
        previous = unique.setdefault(row.messages_sha256, row.messages)
        if previous != row.messages:
            raise RuntimeError("final-answer message SHA-256 collision")
    return unique


def final_answer_prompt_population(
    retrieval: Mapping[str, Any],
    *,
    retrieval_sha256: str,
) -> tuple[tuple[Mapping[str, str], ...], ...]:
    """Return the completely preflighted ordered provider prompt population."""

    return tuple(
        row.messages
        for row in _plan_answers(
            retrieval,
            retrieval_sha256=retrieval_sha256,
        )
    )


def build_final_answer_campaign_binding(
    retrieval: Mapping[str, Any],
    *,
    retrieval_sha256: str,
    authorized_unique_calls: int,
) -> dict[str, Any]:
    """Return the exact gold-free authorization identity for one campaign."""

    if type(authorized_unique_calls) is not int or authorized_unique_calls < 1:
        raise ValueError("authorized_unique_calls must be a positive integer")
    planned = _plan_answers(
        retrieval,
        retrieval_sha256=retrieval_sha256,
    )
    unique = _unique_prompts(planned)
    responder_prompt_policy = build_responder_prompt_policy_identity(
        [row.messages for row in planned]
    )
    if authorized_unique_calls != len(unique):
        raise ValueError(
            "authorized unique final-answer calls must exactly equal the "
            f"preflight population ({authorized_unique_calls} != {len(unique)})"
        )
    binding = {
        "format": FINAL_ANSWER_CAMPAIGN_FORMAT,
        "retrieval_sha256": retrieval_sha256,
        "population_identity_sha256": retrieval["population_identity_sha256"],
        "question_count": len(planned),
        "ordered_question_population_sha256": identity_sha256(
            [
                {
                    "ordinal": row.ordinal,
                    "question_id": row.question_id,
                    "question_sha256": row.question_sha256,
                    "dated_question_sha256": row.dated_question_sha256,
                    "question_part_sha256": row.question_part_sha256,
                }
                for row in planned
            ]
        ),
        "fixed_stage_id": FIXED_STAGE_ID,
        "selected_stage_population_sha256": identity_sha256(
            [
                {
                    "ordinal": row.ordinal,
                    "question_id": row.question_id,
                    "stage_receipt_sha256": row.stage_receipt_sha256,
                    "evidence_projection_sha256": row.evidence_projection_sha256,
                    "provider_messages_sha256": row.messages_sha256,
                    "prompt_token_proxy": row.prompt_token_proxy,
                }
                for row in planned
            ]
        ),
        "unique_provider_prompt_count": len(unique),
        "provider_prompt_population_sha256": identity_sha256(
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
        "authorized_unique_calls": authorized_unique_calls,
        "final_answer_policy_sha256": FINAL_ANSWER_POLICY_SHA256,
        "responder_prompt_policy": responder_prompt_policy,
        "responder_prompt_policy_sha256": identity_sha256(
            responder_prompt_policy
        ),
        "final_answer_implementation_sha256": implementation_sha256(),
        "caller_model": LOCKED_RESPONDER_MODEL,
        "gateway_model": LOCKED_RESPONDER_GATEWAY_MODEL,
        "gateway_url": CENTRAL_DEV_GATEWAY_URL,
        "prompt_token_proxy_cap": RESPONDER_PROMPT_CAP,
        "prompt_token_proxy_identity": tokenizer_proxy_identity(),
        "output_token_reserve": RESPONDER_OUTPUT_TOKEN_RESERVE,
        "retries": 0,
        "temperature": None,
        "gold_fields_present": False,
        "persisted_local_transformer_token_state": False,
        "persisted_local_transformer_token_state_bytes": 0,
    }
    if retrieval.get("format") == VALIDATION_MERGED_RETRIEVAL_FORMAT:
        binding["external_reconstruction_receipt_sha256"] = _require_sha256(
            retrieval.get("external_reconstruction_receipt_sha256"),
            "external reconstruction receipt SHA-256",
        )
    return binding


def _attest_runtime(identity: Mapping[str, Any]) -> None:
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
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "final-answer campaign requires the frozen zero-retry Terra runtime"
        )
    if _gateway_model(str(identity["caller_model"])) != identity["gateway_model"]:
        raise ValueError("final-answer caller and gateway models conflict")


def _attest_runtime_population(
    identity: Mapping[str, Any],
    planned: Sequence[_PlannedAnswer],
    *,
    authorized_unique_calls: int,
) -> None:
    population = preflight_final_answer_prompt_population(
        [row.messages for row in planned],
        authorized_unique_calls=authorized_unique_calls,
        max_prompt_tokens=RESPONDER_PROMPT_CAP,
    )
    expected = {
        "logical_prompt_count": population.logical_prompt_count,
        "unique_prompt_count": population.unique_prompt_count,
        "prompt_population_sha256": population.prompt_population_sha256,
        "prompt_token_proxy_identity": dict(
            population.prompt_token_proxy_identity
        ),
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("final-answer runtime prompt-population seal changed")


def _immutable_usage(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def _available(name: str) -> list[int]:
        flag = name + "_available"
        return [
            int(report[name])
            for report in reports
            if report.get(flag) is True
        ]

    reported_input = _available("reported_input_tokens")
    reported_output = _available("reported_output_tokens")
    reported_total = _available("reported_total_tokens")
    return {
        "unique_journaled_calls": len(reports),
        "reported_input_tokens_available_calls": len(reported_input),
        "reported_input_tokens": sum(reported_input),
        "reported_output_tokens_available_calls": len(reported_output),
        "reported_output_tokens": sum(reported_output),
        "reported_total_tokens_available_calls": len(reported_total),
        "reported_total_tokens": sum(reported_total),
        "input_token_proxy": sum(int(row["input_token_proxy"]) for row in reports),
        "output_token_proxy": sum(int(row["output_token_proxy"]) for row in reports),
        "elapsed_s": sum(float(row["elapsed_s"]) for row in reports),
        "retries": 0,
    }


def answer_recall_guarded_cumulative_stage(
    retrieval: Mapping[str, Any],
    *,
    retrieval_sha256: str,
    runtime: FinalAnswerRuntime,
) -> dict[str, Any]:
    """Answer the preregistered stage after a whole-population preflight."""

    implementation = implementation_sha256()
    planned = _plan_answers(retrieval, retrieval_sha256=retrieval_sha256)
    unique = _unique_prompts(planned)
    runtime_identity = _runtime_identity(runtime)
    _attest_runtime(runtime_identity)
    authorized = runtime_identity.get("authorized_unique_calls")
    if type(authorized) is not int or authorized != len(unique):
        raise ValueError(
            "final-answer runtime authorization differs from the preflight "
            "unique-prompt population"
        )
    _attest_runtime_population(
        runtime_identity,
        planned,
        authorized_unique_calls=authorized,
    )
    campaign = build_final_answer_campaign_binding(
        retrieval,
        retrieval_sha256=retrieval_sha256,
        authorized_unique_calls=authorized,
    )
    if runtime_identity.get("campaign_binding") != campaign:
        raise ValueError("final-answer runtime belongs to another campaign")
    if runtime_identity.get("campaign_binding_sha256") != identity_sha256(campaign):
        raise ValueError("final-answer runtime campaign digest changed")

    outcomes: dict[str, dict[str, Any]] = {}
    for messages_sha, messages in unique.items():
        completion = runtime.complete(
            messages,
            max_new_tokens=RESPONDER_OUTPUT_TOKEN_RESERVE,
        ).strip()
        if not completion:
            raise RuntimeError("final-answer provider returned an empty answer")
        journal = runtime.last_journal_record
        if not isinstance(journal, Mapping):
            raise RuntimeError("final-answer runtime omitted journal provenance")
        report = journal.get("completion_report")
        if not isinstance(report, Mapping):
            raise RuntimeError("final-answer runtime omitted its completion report")
        if set(report) != {
            field.name for field in fields(FinalAnswerCompletionReport)
        }:
            raise RuntimeError("final-answer completion report fields changed")
        local_tokens = count_chat_prompt_token_proxy(messages)
        if (
            report.get("messages_sha256") != messages_sha
            or report.get("input_token_proxy") != local_tokens
            or report.get("max_new_tokens") != RESPONDER_OUTPUT_TOKEN_RESERVE
            or report.get("max_prompt_token_proxy") != RESPONDER_PROMPT_CAP
            or report.get("prompt_population_sha256")
            != runtime_identity.get("prompt_population_sha256")
            or report.get("runtime_identity_sha256")
            != identity_sha256(runtime_identity)
            or report.get("campaign_binding_sha256")
            != identity_sha256(campaign)
            or report.get("call_key_sha256")
            != journal.get("call_key_sha256")
            or report.get("completion_sha256") != quote_sha256(completion)
            or journal.get("completion_sha256") != quote_sha256(completion)
        ):
            raise RuntimeError("final-answer runtime changed prompt/response binding")
        if local_tokens > RESPONDER_PROMPT_CAP:
            raise RuntimeError("preflighted final-answer prompt exceeds hard cap")
        if (
            report.get("reported_input_tokens_available") is True
            and int(report.get("reported_input_tokens", 0)) > RESPONDER_PROMPT_CAP
        ):
            raise RuntimeError("provider-reported final-answer input exceeds hard cap")
        outcomes[messages_sha] = {
            "answer_text": completion,
            "answer_sha256": quote_sha256(completion),
            "call_key_sha256": journal.get("call_key_sha256"),
            "request_journal_sha256": journal.get("request_journal_sha256"),
            "response_journal_sha256": journal.get("response_journal_sha256"),
            "completion_report": dict(report),
        }
        for label in (
            "call_key_sha256",
            "request_journal_sha256",
            "response_journal_sha256",
        ):
            _require_sha256(outcomes[messages_sha][label], label)

    questions: list[dict[str, Any]] = []
    for row in planned:
        outcome = outcomes[row.messages_sha256]
        questions.append(
            {
                "format": FINAL_ANSWER_QUESTION_FORMAT,
                "ordinal": row.ordinal,
                "question_id": row.question_id,
                "question_sha256": row.question_sha256,
                "dated_question_sha256": row.dated_question_sha256,
                "retrieval_question_part_sha256": row.question_part_sha256,
                "fixed_stage_id": FIXED_STAGE_ID,
                "stage_receipt_sha256": row.stage_receipt_sha256,
                "evidence_projection_sha256": row.evidence_projection_sha256,
                "provider_messages_sha256": row.messages_sha256,
                "prompt_token_proxy": row.prompt_token_proxy,
                "prompt_token_cap": RESPONDER_PROMPT_CAP,
                "output_token_reserve": RESPONDER_OUTPUT_TOKEN_RESERVE,
                "answer": {
                    "text": outcome["answer_text"],
                    "sha256": outcome["answer_sha256"],
                },
                "call_key_sha256": outcome["call_key_sha256"],
                "request_journal_sha256": outcome["request_journal_sha256"],
                "response_journal_sha256": outcome["response_journal_sha256"],
                "completion_report": outcome["completion_report"],
            }
        )
    reports = [outcome["completion_report"] for outcome in outcomes.values()]
    result = {
        "format": FINAL_ANSWER_FORMAT,
        "retrieval_sha256": retrieval_sha256,
        "population_identity_sha256": retrieval["population_identity_sha256"],
        "question_count": len(questions),
        "gold_fields_present": False,
        "fixed_stage_id": FIXED_STAGE_ID,
        "final_answer_policy": dict(FINAL_ANSWER_POLICY),
        "final_answer_policy_sha256": FINAL_ANSWER_POLICY_SHA256,
        "responder_prompt_policy_sha256": campaign[
            "responder_prompt_policy_sha256"
        ],
        "final_answer_implementation_sha256": implementation,
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": identity_sha256(runtime_identity),
        "campaign_binding": campaign,
        "campaign_binding_sha256": identity_sha256(campaign),
        "logical_answer_count": len(planned),
        "unique_provider_prompt_count": len(unique),
        "deduplicated_logical_answer_count": len(planned) - len(unique),
        "prompt_preflight": {
            "completed_before_provider_calls": True,
            "all_prompts_sealed": True,
            "all_local_prompt_proxies_recounted": True,
            "prompt_token_proxy_cap": RESPONDER_PROMPT_CAP,
            "maximum_prompt_token_proxy": max(
                (row.prompt_token_proxy for row in planned), default=0
            ),
            "cap_violation_count": sum(
                row.prompt_token_proxy > RESPONDER_PROMPT_CAP for row in planned
            ),
            "prompt_token_proxy_identity": tokenizer_proxy_identity(),
        },
        "responder_usage": _immutable_usage(reports),
        "local_transformer_state_receipt": {
            "scope": (
                "persisted local token IDs, hidden states, and KV caches; "
                "canonical plaintext request/response journals are provenance"
            ),
            "before_provider_calls_present": False,
            "before_provider_calls_bytes": 0,
            "after_provider_calls_present": False,
            "after_provider_calls_bytes": 0,
            "runtime_owns_local_transformer": False,
            "status": "zero_persisted_local_transformer_token_state",
        },
        "external_provider_persistence": "not_certified",
        "questions": questions,
    }
    if implementation_sha256() != implementation:
        raise RuntimeError("final-answer implementation changed during the run")
    return result


def validate_final_answer_artifact(
    artifact: Mapping[str, Any],
    *,
    retrieval: Mapping[str, Any],
    artifact_sha256: str,
    retrieval_sha256: str,
) -> None:
    """Fail closed on a fixed-stage artifact before post-hoc gold loading."""

    if _canonical_digest(artifact) != artifact_sha256:
        raise ValueError("final-answer SHA-256 does not match canonical bytes")
    expected_top_fields = {
        "format",
        "retrieval_sha256",
        "population_identity_sha256",
        "question_count",
        "gold_fields_present",
        "fixed_stage_id",
        "final_answer_policy",
        "final_answer_policy_sha256",
        "responder_prompt_policy_sha256",
        "final_answer_implementation_sha256",
        "runtime_identity",
        "runtime_identity_sha256",
        "campaign_binding",
        "campaign_binding_sha256",
        "logical_answer_count",
        "unique_provider_prompt_count",
        "deduplicated_logical_answer_count",
        "prompt_preflight",
        "responder_usage",
        "local_transformer_state_receipt",
        "external_provider_persistence",
        "questions",
    }
    if set(artifact) != expected_top_fields:
        raise ValueError("final-answer artifact fields changed")
    planned = _plan_answers(retrieval, retrieval_sha256=retrieval_sha256)
    if (
        artifact.get("format") != FINAL_ANSWER_FORMAT
        or artifact.get("retrieval_sha256") != retrieval_sha256
        or artifact.get("population_identity_sha256")
        != retrieval.get("population_identity_sha256")
        or artifact.get("question_count") != len(planned)
        or artifact.get("gold_fields_present") is not False
        or artifact.get("fixed_stage_id") != FIXED_STAGE_ID
        or artifact.get("final_answer_policy") != FINAL_ANSWER_POLICY
        or artifact.get("final_answer_policy_sha256")
        != FINAL_ANSWER_POLICY_SHA256
    ):
        raise ValueError("final-answer artifact belongs to another protocol")
    runtime_identity = artifact.get("runtime_identity")
    campaign = artifact.get("campaign_binding")
    if not isinstance(runtime_identity, Mapping) or not isinstance(
        campaign, Mapping
    ):
        raise ValueError("final-answer runtime/campaign identity is missing")
    responder_prompt_policy = campaign.get("responder_prompt_policy")
    if not isinstance(responder_prompt_policy, Mapping):
        raise ValueError("responder prompt policy identity is missing")
    responder_prompt_policy_sha256 = _require_sha256(
        campaign.get("responder_prompt_policy_sha256"),
        "responder prompt policy SHA-256",
    )
    if (
        identity_sha256(dict(responder_prompt_policy))
        != responder_prompt_policy_sha256
        or artifact.get("responder_prompt_policy_sha256")
        != responder_prompt_policy_sha256
    ):
        raise ValueError("responder prompt policy seal changed")
    _attest_runtime(runtime_identity)
    if (
        identity_sha256(dict(runtime_identity))
        != artifact.get("runtime_identity_sha256")
        or identity_sha256(dict(campaign))
        != artifact.get("campaign_binding_sha256")
        or runtime_identity.get("campaign_binding") != campaign
    ):
        raise ValueError("final-answer runtime/campaign seal changed")
    expected_campaign = build_final_answer_campaign_binding(
        retrieval,
        retrieval_sha256=retrieval_sha256,
        authorized_unique_calls=len(_unique_prompts(planned)),
    )
    if campaign != expected_campaign:
        raise ValueError("final-answer campaign binding changed")
    _attest_runtime_population(
        runtime_identity,
        planned,
        authorized_unique_calls=len(_unique_prompts(planned)),
    )
    rows = artifact.get("questions")
    if not isinstance(rows, list) or len(rows) != len(planned):
        raise ValueError("final-answer question population is incomplete")
    expected_prompt_preflight = {
        "completed_before_provider_calls": True,
        "all_prompts_sealed": True,
        "all_local_prompt_proxies_recounted": True,
        "prompt_token_proxy_cap": RESPONDER_PROMPT_CAP,
        "maximum_prompt_token_proxy": max(
            (row.prompt_token_proxy for row in planned), default=0
        ),
        "cap_violation_count": 0,
        "prompt_token_proxy_identity": tokenizer_proxy_identity(),
    }
    expected_state = {
        "scope": (
            "persisted local token IDs, hidden states, and KV caches; "
            "canonical plaintext request/response journals are provenance"
        ),
        "before_provider_calls_present": False,
        "before_provider_calls_bytes": 0,
        "after_provider_calls_present": False,
        "after_provider_calls_bytes": 0,
        "runtime_owns_local_transformer": False,
        "status": "zero_persisted_local_transformer_token_state",
    }
    if (
        artifact.get("logical_answer_count") != len(planned)
        or artifact.get("unique_provider_prompt_count")
        != len(_unique_prompts(planned))
        or artifact.get("deduplicated_logical_answer_count")
        != len(planned) - len(_unique_prompts(planned))
        or artifact.get("prompt_preflight") != expected_prompt_preflight
        or artifact.get("local_transformer_state_receipt") != expected_state
        or artifact.get("external_provider_persistence") != "not_certified"
    ):
        raise ValueError("final-answer protocol accounting changed")
    immutable_reports: list[Mapping[str, Any]] = []
    expected_question_fields = {
        "format",
        "ordinal",
        "question_id",
        "question_sha256",
        "dated_question_sha256",
        "retrieval_question_part_sha256",
        "fixed_stage_id",
        "stage_receipt_sha256",
        "evidence_projection_sha256",
        "provider_messages_sha256",
        "prompt_token_proxy",
        "prompt_token_cap",
        "output_token_reserve",
        "answer",
        "call_key_sha256",
        "request_journal_sha256",
        "response_journal_sha256",
        "completion_report",
    }
    for plan, row in zip(planned, rows, strict=True):
        if not isinstance(row, Mapping):
            raise ValueError("final-answer question row must be an object")
        if set(row) != expected_question_fields:
            raise ValueError("final-answer question fields changed")
        answer = row.get("answer")
        report = row.get("completion_report")
        if not isinstance(answer, Mapping) or not isinstance(report, Mapping):
            raise ValueError("final-answer response provenance is incomplete")
        if set(answer) != {"text", "sha256"}:
            raise ValueError("final-answer answer fields changed")
        if set(report) != {
            field.name for field in fields(FinalAnswerCompletionReport)
        }:
            raise ValueError("final-answer completion report fields changed")
        text = answer.get("text")
        if (
            not isinstance(text, str)
            or not text
            or text.strip() != text
        ):
            raise ValueError("final-answer text is empty")
        expected = {
            "format": FINAL_ANSWER_QUESTION_FORMAT,
            "ordinal": plan.ordinal,
            "question_id": plan.question_id,
            "question_sha256": plan.question_sha256,
            "dated_question_sha256": plan.dated_question_sha256,
            "retrieval_question_part_sha256": plan.question_part_sha256,
            "fixed_stage_id": FIXED_STAGE_ID,
            "stage_receipt_sha256": plan.stage_receipt_sha256,
            "evidence_projection_sha256": plan.evidence_projection_sha256,
            "provider_messages_sha256": plan.messages_sha256,
            "prompt_token_proxy": plan.prompt_token_proxy,
            "prompt_token_cap": RESPONDER_PROMPT_CAP,
            "output_token_reserve": RESPONDER_OUTPUT_TOKEN_RESERVE,
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("final-answer question binding changed")
        if (
            answer.get("sha256") != quote_sha256(text.strip())
            or report.get("messages_sha256") != plan.messages_sha256
            or report.get("input_token_proxy") != plan.prompt_token_proxy
            or report.get("max_new_tokens")
            != RESPONDER_OUTPUT_TOKEN_RESERVE
            or row.get("call_key_sha256") != report.get("call_key_sha256")
            or report.get("completion_sha256") != answer.get("sha256")
            or report.get("runtime_identity_sha256")
            != artifact.get("runtime_identity_sha256")
            or report.get("campaign_binding_sha256")
            != artifact.get("campaign_binding_sha256")
            or report.get("prompt_population_sha256")
            != runtime_identity.get("prompt_population_sha256")
            or row.get("request_journal_sha256")
            != report.get("request_journal_sha256")
        ):
            raise ValueError("final-answer journal/report binding changed")
        provider_available = report.get("reported_input_tokens_available")
        provider_tokens = report.get("reported_input_tokens")
        if type(provider_available) is not bool or type(provider_tokens) is not int:
            raise ValueError("provider-reported final-answer usage changed")
        if (
            provider_available
            and not (1 <= provider_tokens <= RESPONDER_PROMPT_CAP)
        ) or (not provider_available and provider_tokens != 0):
            raise ValueError("provider-reported final-answer input exceeds cap")
        if (
            report.get("gateway_url") != CENTRAL_DEV_GATEWAY_URL
            or report.get("caller_model") != LOCKED_RESPONDER_MODEL
            or report.get("gateway_model") != LOCKED_RESPONDER_GATEWAY_MODEL
            or report.get("retries") != 0
            or report.get("cache_hit") is not False
            or report.get("physical_call") is not True
        ):
            raise ValueError("final-answer immutable physical report changed")
        for label in (
            "call_key_sha256",
            "request_journal_sha256",
            "response_journal_sha256",
        ):
            _require_sha256(row.get(label), label)
        immutable_reports.append(report)
    if artifact.get("responder_usage") != _immutable_usage(immutable_reports):
        raise ValueError("final-answer immutable usage accounting changed")


__all__ = [
    "FINAL_ANSWER_CAMPAIGN_FORMAT",
    "FINAL_ANSWER_FORMAT",
    "FINAL_ANSWER_POLICY",
    "FINAL_ANSWER_POLICY_SHA256",
    "FINAL_ANSWER_QUESTION_FORMAT",
    "FINAL_ANSWER_RUNTIME_FORMAT",
    "FIXED_STAGE_ID",
    "LOCKED_RESPONDER_MODEL",
    "RESPONDER_OUTPUT_TOKEN_RESERVE",
    "RESPONDER_PROMPT_POLICY_FORMAT",
    "RESPONDER_PROMPT_CAP",
    "answer_recall_guarded_cumulative_stage",
    "build_final_answer_campaign_binding",
    "build_responder_prompt_policy_identity",
    "final_answer_prompt_population",
    "validate_final_answer_artifact",
]
