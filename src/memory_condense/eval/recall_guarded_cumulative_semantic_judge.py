"""Independent semantic judging for sealed cumulative-synthesis artifacts."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    tokenizer_proxy_identity,
)
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval._binary_judge_protocol import (
    JUDGE_MAX_TOKENS,
    parse_binary_judge_verdict,
)
from memory_condense.eval._recall_guarded_cumulative_synthesis_artifacts import (
    _validate_assembled_synthesis,
)
from memory_condense.eval.benchmark import (
    BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE,
    build_judge_prompt,
)
from memory_condense.eval.recall_guarded_cumulative_1m import (
    _canonical_json_bytes,
    population_identity_sha256,
)
from memory_condense.eval.recall_guarded_cumulative_provider_synthesis_runtime import (
    CENTRAL_DEV_GATEWAY_URL,
    PROVIDER_RUNTIME_FORMAT,
    _gateway_model,
)
from memory_condense.eval.recall_guarded_cumulative_semantic_judge_runtime import (
    SEMANTIC_JUDGE_RUNTIME_FORMAT,
)
from memory_condense.eval.recall_guarded_cumulative_synthesis import (
    SYNTHESIS_PROMPT_POLICY,
    SYNTHESIS_STAGE_IDS,
    build_synthesis_messages,
    validate_published_retrieval,
)
from memory_condense.eval.reproducibility import implementation_sha256
from memory_condense.ingest.loader import BenchmarkQuestion, BenchmarkSample


SEMANTIC_JUDGE_FORMAT = (
    "memory-condense-recall-guarded-semantic-judge-score-v2"
)
SEMANTIC_JUDGE_CAMPAIGN_FORMAT = (
    "memory-condense-recall-guarded-semantic-judge-campaign-v2"
)
TARGET_ACCURACY = 0.95
MINIMUM_GATE_QUESTIONS = 100
DEFAULT_RESPONDER_PROMPT_CAP = 8_000
LOCKED_RESPONDER_MODEL = "openai/codex_sdk/gpt-5.6-terra"
LOCKED_JUDGE_MODEL = "openai/codex_sdk/gpt-5.6-sol"
LOCKED_RESPONDER_GATEWAY_MODEL = "codex_sdk/gpt-5.6-terra"
LOCKED_JUDGE_GATEWAY_MODEL = "codex_sdk/gpt-5.6-sol"
LOCKED_JUDGE_MAX_NEW_TOKENS = JUDGE_MAX_TOKENS
RESPONDER_PROMPT_CAP_SEMANTICS = (
    "complete provider-visible responder chat prompt under the frozen "
    "cl100k_base local proxy; provider input usage is reported separately"
)
SEMANTIC_JUDGE_POLICY = {
    "format": "memory-condense-independent-binary-semantic-judge-policy-v2",
    "prompt_builder": (
        "memory_condense.eval.benchmark.build_judge_prompt"
    ),
    "verdict_parser": (
        "memory_condense.eval._binary_judge_protocol."
        "parse_binary_judge_verdict"
    ),
    "question_form": "undated benchmark question",
    "stage_ids": list(SYNTHESIS_STAGE_IDS),
    "deduplication": (
        "identical canonical question+gold+prediction judge messages"
    ),
    "target_accuracy": TARGET_ACCURACY,
    "minimum_questions_per_stage": MINIMUM_GATE_QUESTIONS,
    "gate_unit": "one fixed retrieval/synthesis stage",
    "locked_responder_model": LOCKED_RESPONDER_MODEL,
    "locked_judge_model": LOCKED_JUDGE_MODEL,
    "locked_responder_gateway_model": LOCKED_RESPONDER_GATEWAY_MODEL,
    "locked_judge_gateway_model": LOCKED_JUDGE_GATEWAY_MODEL,
    "locked_gateway_url": CENTRAL_DEV_GATEWAY_URL,
    "responder_runtime_format": PROVIDER_RUNTIME_FORMAT,
    "judge_runtime_format": SEMANTIC_JUDGE_RUNTIME_FORMAT,
    "provider_retries": 0,
    "responder_temperature": None,
    "judge_temperature": None,
    "judge_max_new_tokens": LOCKED_JUDGE_MAX_NEW_TOKENS,
    "responder_prompt_cap": DEFAULT_RESPONDER_PROMPT_CAP,
    "responder_output_token_reserve": (
        BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE
    ),
    "responder_output_reserve_gate": (
        "runtime+campaign+stage+effective-completion-must-all-equal-reserve"
    ),
    "responder_prompt_cap_proof": (
        "independent-reconstruction-from-sealed-retrieval-v1"
    ),
    "responder_prompt_token_proxy_identity": tokenizer_proxy_identity(),
}
SEMANTIC_JUDGE_POLICY_SHA256 = identity_sha256(SEMANTIC_JUDGE_POLICY)


class SemanticJudgeRuntime(Protocol):
    identity: Any
    last_journal_record: Mapping[str, Any] | None

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class _PlannedJudgment:
    ordinal: int
    question_id: str
    stage_id: str
    question_sha256: str
    gold_answer_sha256: str
    prediction_sha256: str
    messages: tuple[Mapping[str, str], ...]
    messages_sha256: str


def _runtime_identity(runtime: SemanticJudgeRuntime) -> dict[str, Any]:
    value = runtime.identity
    if isinstance(value, Mapping):
        return {str(key): child for key, child in value.items()}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        payload = dump()
        if isinstance(payload, Mapping):
            return {str(key): child for key, child in payload.items()}
    raise TypeError("semantic judge runtime has no mapping identity")


def _canonical_synthesis_digest(synthesis: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(synthesis)).hexdigest()


def _gold_population_sha256(sample: BenchmarkSample) -> str:
    return identity_sha256(
        [
            {
                "question_id": question.question_id,
                "question_sha256": quote_sha256(question.question),
                "gold_answer_sha256": quote_sha256(question.answer),
                "evidence_source_ids": list(question.evidence_sources),
            }
            for question in sample.questions
        ]
    )


def _validate_source_binding(
    synthesis: Mapping[str, Any],
    *,
    retrieval: Mapping[str, Any],
    sample: BenchmarkSample,
    synthesis_sha256: str,
    retrieval_sha256: str,
) -> list[tuple[Mapping[str, Any], BenchmarkQuestion]]:
    _validate_assembled_synthesis(synthesis)
    validate_published_retrieval(retrieval)
    if synthesis_sha256 != _canonical_synthesis_digest(synthesis):
        raise ValueError("synthesis SHA-256 does not match canonical bytes")
    if retrieval_sha256 != _canonical_synthesis_digest(retrieval):
        raise ValueError("retrieval SHA-256 does not match canonical bytes")
    population_sha = population_identity_sha256(sample)
    if (
        synthesis.get("population_identity_sha256") != population_sha
        or retrieval.get("population_identity_sha256") != population_sha
    ):
        raise ValueError("synthesis and benchmark population identities differ")
    if synthesis.get("retrieval_sha256") != retrieval_sha256:
        raise ValueError("synthesis belongs to another retrieval artifact")
    questions = synthesis.get("questions")
    retrieval_questions = retrieval.get("questions")
    if not isinstance(questions, list) or len(questions) != len(sample.questions):
        raise ValueError("synthesis and benchmark question populations differ")
    if not isinstance(retrieval_questions, list) or len(
        retrieval_questions
    ) != len(sample.questions):
        raise ValueError("retrieval and benchmark question populations differ")
    bound: list[tuple[Mapping[str, Any], BenchmarkQuestion]] = []
    for ordinal, (source, retrieval_question, gold) in enumerate(
        zip(questions, retrieval_questions, sample.questions, strict=True)
    ):
        if not isinstance(source, Mapping) or not isinstance(
            retrieval_question, Mapping
        ):
            raise ValueError("synthesis/retrieval question must be an object")
        if (
            source.get("ordinal") != ordinal
            or retrieval_question.get("ordinal") != ordinal
            or source.get("question_id") != gold.question_id
            or retrieval_question.get("question_id") != gold.question_id
            or source.get("question_sha256")
            != retrieval_question.get("question_sha256")
        ):
            raise ValueError("synthesis question order differs from benchmark")
        bound.append((source, gold))
    return bound


def _plan_judgments(
    synthesis: Mapping[str, Any],
    *,
    retrieval: Mapping[str, Any],
    sample: BenchmarkSample,
    synthesis_sha256: str,
    retrieval_sha256: str,
) -> list[_PlannedJudgment]:
    bound = _validate_source_binding(
        synthesis,
        retrieval=retrieval,
        sample=sample,
        synthesis_sha256=synthesis_sha256,
        retrieval_sha256=retrieval_sha256,
    )
    planned: list[_PlannedJudgment] = []
    for ordinal, (source, gold) in enumerate(bound):
        stages = source["stages"]
        for stage in stages:
            stage_id = str(stage["stage_id"])
            if stage_id not in SYNTHESIS_STAGE_IDS:
                raise ValueError("unexpected synthesis stage in judge population")
            prediction = str(stage["answer"]["text"]).strip()
            if not prediction:
                raise ValueError("semantic judge prediction must not be empty")
            messages = tuple(
                dict(row)
                for row in build_judge_prompt(
                    gold.question,
                    gold.answer,
                    prediction,
                )
            )
            planned.append(
                _PlannedJudgment(
                    ordinal=ordinal,
                    question_id=gold.question_id,
                    stage_id=stage_id,
                    question_sha256=quote_sha256(gold.question),
                    gold_answer_sha256=quote_sha256(gold.answer),
                    prediction_sha256=quote_sha256(prediction),
                    messages=messages,
                    messages_sha256=identity_sha256(list(messages)),
                )
            )
    expected = len(sample.questions) * len(SYNTHESIS_STAGE_IDS)
    if len(planned) != expected:
        raise ValueError("semantic judge stage population is incomplete")
    return planned


def _unique_prompts(
    planned: Sequence[_PlannedJudgment],
) -> dict[str, tuple[Mapping[str, str], ...]]:
    unique: dict[str, tuple[Mapping[str, str], ...]] = {}
    for row in planned:
        previous = unique.setdefault(row.messages_sha256, row.messages)
        if previous != row.messages:
            raise RuntimeError("semantic judge message SHA-256 collision")
    return unique


def build_semantic_judge_campaign_binding(
    synthesis: Mapping[str, Any],
    *,
    retrieval: Mapping[str, Any],
    sample: BenchmarkSample,
    synthesis_sha256: str,
    retrieval_sha256: str,
    responder_prompt_cap: int = DEFAULT_RESPONDER_PROMPT_CAP,
    authorized_unique_calls: int,
) -> dict[str, Any]:
    """Build the exact post-hoc, gold-bearing campaign identity."""

    if (
        type(responder_prompt_cap) is not int
        or responder_prompt_cap != DEFAULT_RESPONDER_PROMPT_CAP
    ):
        raise ValueError(
            "responder_prompt_cap must exactly equal the locked 8000-token cap"
        )
    if type(authorized_unique_calls) is not int or authorized_unique_calls < 1:
        raise ValueError("authorized_unique_calls must be a positive integer")
    implementation = implementation_sha256()
    planned = _plan_judgments(
        synthesis,
        retrieval=retrieval,
        sample=sample,
        synthesis_sha256=synthesis_sha256,
        retrieval_sha256=retrieval_sha256,
    )
    unique = _unique_prompts(planned)
    if authorized_unique_calls != len(unique):
        raise ValueError(
            "authorized unique judge-call cap must exactly equal the "
            f"precomputed requirement ({authorized_unique_calls} != {len(unique)})"
        )
    responder_model = _attest_responder_runtime(synthesis)
    prompt_diagnostics = _responder_prompt_diagnostics(
        synthesis,
        retrieval=retrieval,
        prompt_cap=responder_prompt_cap,
    )
    reserve_diagnostics = _responder_output_reserve_diagnostics(synthesis)
    if not prompt_diagnostics[
        "all_responder_prompts_proven_within_local_cap"
    ] or prompt_diagnostics["provider_prompt_cap_violation_count"]:
        raise ValueError(
            "responder prompt-cap proof failed before judge authorization"
        )
    return {
        "format": SEMANTIC_JUDGE_CAMPAIGN_FORMAT,
        "synthesis_sha256": synthesis_sha256,
        "retrieval_sha256": retrieval_sha256,
        "population_identity_sha256": population_identity_sha256(sample),
        "gold_scoring_population_sha256": _gold_population_sha256(sample),
        "question_count": len(sample.questions),
        "stage_ids": list(SYNTHESIS_STAGE_IDS),
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
        "semantic_judge_policy_sha256": SEMANTIC_JUDGE_POLICY_SHA256,
        "semantic_judge_implementation_sha256": implementation,
        "target_accuracy": TARGET_ACCURACY,
        "minimum_questions_per_stage": MINIMUM_GATE_QUESTIONS,
        "responder_prompt_cap": responder_prompt_cap,
        "responder_prompt_cap_semantics": RESPONDER_PROMPT_CAP_SEMANTICS,
        "authorized_unique_judge_calls": authorized_unique_calls,
        "responder_model": LOCKED_RESPONDER_MODEL,
        "judge_model": LOCKED_JUDGE_MODEL,
        "judge_max_new_tokens": LOCKED_JUDGE_MAX_NEW_TOKENS,
        "responder_prompt_cap_diagnostics_sha256": identity_sha256(
            prompt_diagnostics
        ),
        "max_responder_prompt_token_proxy": prompt_diagnostics[
            "max_local_prompt_token_proxy"
        ],
        "responder_prompt_token_proxy_identity": prompt_diagnostics[
            "tokenizer_proxy_identity"
        ],
        "responder_output_reserve_protocol_eligible": reserve_diagnostics[
            "protocol_eligible"
        ],
        "responder_output_reserve_diagnostics_sha256": identity_sha256(
            reserve_diagnostics
        ),
    }


def _attest_responder_runtime(synthesis: Mapping[str, Any]) -> str:
    """Require the exact zero-retry Terra synthesis runtime route."""

    identity = synthesis.get("runtime_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("synthesis responder runtime identity is missing")
    expected = {
        "format": PROVIDER_RUNTIME_FORMAT,
        "gateway_url": CENTRAL_DEV_GATEWAY_URL,
        "caller_model": LOCKED_RESPONDER_MODEL,
        "gateway_model": LOCKED_RESPONDER_GATEWAY_MODEL,
        "retries": 0,
        "temperature": None,
    }
    if any(
        name not in identity or identity[name] != value
        for name, value in expected.items()
    ):
        raise ValueError(
            "semantic campaign requires the attested zero-retry Terra "
            "responder route"
        )
    if _gateway_model(str(identity["caller_model"])) != identity["gateway_model"]:
        raise ValueError("responder caller and gateway models conflict")
    return LOCKED_RESPONDER_MODEL


def _attest_judge_runtime(runtime_identity: Mapping[str, Any]) -> str:
    """Require the exact zero-retry Sol semantic-judge runtime route."""

    expected = {
        "format": SEMANTIC_JUDGE_RUNTIME_FORMAT,
        "gateway_url": CENTRAL_DEV_GATEWAY_URL,
        "caller_model": LOCKED_JUDGE_MODEL,
        "gateway_model": LOCKED_JUDGE_GATEWAY_MODEL,
        "retries": 0,
        "temperature": None,
        "default_max_new_tokens": LOCKED_JUDGE_MAX_NEW_TOKENS,
    }
    if any(
        name not in runtime_identity or runtime_identity[name] != value
        for name, value in expected.items()
    ):
        raise ValueError(
            "semantic campaign requires the attested zero-retry Sol judge route"
        )
    if _gateway_model(str(runtime_identity["caller_model"])) != (
        runtime_identity["gateway_model"]
    ):
        raise ValueError("judge caller and gateway models conflict")
    return LOCKED_JUDGE_MODEL


def _append_responder_prompt_diagnostic(
    rows: list[dict[str, Any]],
    *,
    question_id: str,
    stage_id: str,
    messages: Sequence[Mapping[str, str]],
    declared_messages_sha256: Any,
    report: Any,
    prompt_cap: int,
    request_kind: str | None = None,
) -> None:
    """Recompute and append one responder request or fail closed."""

    messages_sha = identity_sha256(list(messages))
    if messages_sha != declared_messages_sha256:
        raise ValueError(
            "reconstructed responder prompt hash differs from synthesis"
        )
    local_tokens = count_chat_prompt_token_proxy(messages)
    if not isinstance(report, Mapping):
        raise ValueError("synthesis completion report is missing")
    if (
        report.get("input_token_proxy") != local_tokens
        or report.get("messages_sha256") not in {None, messages_sha}
    ):
        raise ValueError(
            "stored responder prompt accounting failed reconstruction"
        )
    provider_flag = report.get("reported_input_tokens_available")
    provider_value = report.get("reported_input_tokens")
    provider_available = (
        provider_flag is True
        and type(provider_value) is int
        and provider_value > 0
    )
    provider_tokens = int(provider_value) if provider_available else 0
    row = {
        "question_id": question_id,
        "stage_id": stage_id,
        "prompt_messages_sha256": messages_sha,
        "synthesis_call_key_sha256": report.get("call_key_sha256"),
        "local_prompt_token_proxy_available": True,
        "local_prompt_token_proxy": local_tokens,
        "local_prompt_cap_compliant": local_tokens <= prompt_cap,
        "local_prompt_proof": (
            "independently_recomputed_from_sealed_retrieval"
        ),
        "provider_input_tokens_available": provider_available,
        "provider_input_tokens": provider_tokens,
        "provider_prompt_cap_compliant": (
            provider_tokens <= prompt_cap if provider_available else None
        ),
    }
    if request_kind is not None:
        row["request_kind"] = request_kind
    rows.append(row)


def _responder_prompt_diagnostics(
    synthesis: Mapping[str, Any],
    *,
    retrieval: Mapping[str, Any],
    prompt_cap: int,
) -> dict[str, Any]:
    policy = synthesis.get("synthesis_prompt_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("synthesis prompt policy is missing")
    retrieval_by_id = {
        str(question["question_id"]): question
        for question in retrieval["questions"]
    }
    rows: list[dict[str, Any]] = []
    structured_attempt_rows = 0
    for question in synthesis["questions"]:
        source_question = retrieval_by_id.get(str(question["question_id"]))
        if not isinstance(source_question, Mapping):
            raise ValueError("synthesis question has no bound retrieval question")
        source_stages = source_question.get("stages")
        if not isinstance(source_stages, list) or len(source_stages) != 4:
            raise ValueError("retrieval question stage population changed")
        root_evidence = source_stages[0].get("evidence")
        if not isinstance(root_evidence, list):
            raise ValueError("retrieval root evidence is missing")
        root_ids = {
            str(row["evidence_id"])
            for row in root_evidence
            if isinstance(row, Mapping)
        }
        for source_stage, stage in zip(
            source_stages[1:], question["stages"], strict=True
        ):
            mode = stage.get("synthesis_mode")
            structured_attempt = stage.get("structured_attempt")
            if structured_attempt is not None and mode != (
                "short_answer_with_forced_choice_attribution"
            ):
                raise ValueError(
                    "structured attempt is attached to a non-fallback stage"
                )
            # Reused stages copy their origin's structured-attempt receipt but
            # do not issue another responder request. Inspect the physical
            # origin exactly once; any violation rejects the whole campaign
            # before judge authorization.
            if (
                structured_attempt is not None
                and stage.get("reused_from_stage_id") is None
            ):
                if not isinstance(structured_attempt, Mapping):
                    raise ValueError("structured attempt must be an object")
                structured_messages, _aliases, _novel = (
                    build_synthesis_messages(
                        source_stage,
                        root_evidence_ids=root_ids,
                        prompt_policy=policy,
                    )
                )
                _append_responder_prompt_diagnostic(
                    rows,
                    question_id=str(question["question_id"]),
                    stage_id=str(stage["stage_id"]),
                    messages=structured_messages,
                    declared_messages_sha256=structured_attempt.get(
                        "prompt_messages_sha256"
                    ),
                    report=structured_attempt.get("completion_report"),
                    prompt_cap=prompt_cap,
                    request_kind="structured_attempt",
                )
                structured_attempt_rows += 1
            if mode == "structured_generation":
                messages, _aliases, _novel = build_synthesis_messages(
                    source_stage,
                    root_evidence_ids=root_ids,
                    prompt_policy=policy,
                )
            elif mode == "short_answer_with_forced_choice_attribution":
                source_messages = source_stage.get("provider_messages")
                if not isinstance(source_messages, list):
                    raise ValueError("retrieval provider messages are missing")
                messages = [dict(row) for row in source_messages]
            else:
                raise ValueError("unknown synthesis mode in responder cap proof")
            _append_responder_prompt_diagnostic(
                rows,
                question_id=str(question["question_id"]),
                stage_id=str(stage["stage_id"]),
                messages=messages,
                declared_messages_sha256=stage.get(
                    "prompt_messages_sha256"
                ),
                report=stage.get("completion_report"),
                prompt_cap=prompt_cap,
            )
    local = [
        row["local_prompt_token_proxy"]
        for row in rows
        if row["local_prompt_token_proxy_available"]
    ]
    provider = [
        row["provider_input_tokens"]
        for row in rows
        if row["provider_input_tokens_available"]
    ]
    local_full = len(local) == len(rows)
    local_violation = any(value > prompt_cap for value in local)
    local_pass = local_full and not local_violation
    provider_full = len(provider) == len(rows)
    provider_violation = any(value > prompt_cap for value in provider)
    provider_pass = provider_full and not provider_violation
    diagnostics = {
        "prompt_cap": prompt_cap,
        "prompt_cap_semantics": RESPONDER_PROMPT_CAP_SEMANTICS,
        "tokenizer_proxy_identity": tokenizer_proxy_identity(),
        "prompt_reconstruction": (
            "sealed-retrieval+embedded-synthesis-policy-v1"
        ),
        "logical_responder_rows": len(rows),
        "local_prompt_proxy_available_rows": len(local),
        "max_local_prompt_token_proxy": max(local, default=None),
        "local_prompt_cap_violation_count": sum(
            value > prompt_cap for value in local
        ),
        "all_responder_prompts_proven_within_local_cap": local_pass,
        "local_prompt_cap_status": (
            "pass"
            if local_pass
            else "fail"
            if local_violation
            else "unavailable"
        ),
        "provider_input_usage_available_rows": len(provider),
        "max_provider_input_tokens": max(provider, default=None),
        "provider_prompt_cap_violation_count": sum(
            value > prompt_cap for value in provider
        ),
        "all_provider_inputs_proven_within_cap": (
            False
            if provider_violation
            else provider_pass
            if provider_full
            else None
        ),
        "provider_prompt_cap_status": (
            "pass"
            if provider_pass
            else "fail"
            if provider_violation
            else "unavailable"
        ),
        "rows": rows,
    }
    # Keep the no-fallback v3 diagnostic projection byte-for-byte stable.
    # Fallback campaigns receive the additional accounting explicitly.
    if structured_attempt_rows:
        diagnostics["structured_attempt_responder_rows"] = (
            structured_attempt_rows
        )
        diagnostics["complete_responder_request_rows"] = len(rows)
    return diagnostics


def _responder_output_reserve_diagnostics(
    synthesis: Mapping[str, Any],
) -> dict[str, Any]:
    """Attest the frozen 256-token effective-answer request allowance."""

    required = BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE
    runtime_identity = synthesis.get("runtime_identity")
    campaign_policy = synthesis.get("request_policy")
    runtime_default = (
        runtime_identity.get("default_max_new_tokens")
        if isinstance(runtime_identity, Mapping)
        else None
    )
    campaign_maximum = (
        campaign_policy.get("max_new_tokens")
        if isinstance(campaign_policy, Mapping)
        else None
    )
    runtime_policy_eligible = (
        type(runtime_default) is int and runtime_default == required
    )
    campaign_policy_eligible = (
        type(campaign_maximum) is int and campaign_maximum == required
    )
    rows: list[dict[str, Any]] = []
    for question in synthesis.get("questions", ()):
        if not isinstance(question, Mapping):
            continue
        for stage in question.get("stages", ()):
            if not isinstance(stage, Mapping):
                continue
            stage_policy = stage.get("request_policy")
            report = stage.get("completion_report")
            stage_maximum = (
                stage_policy.get("max_new_tokens")
                if isinstance(stage_policy, Mapping)
                else None
            )
            effective_maximum = (
                report.get("max_new_tokens")
                if isinstance(report, Mapping)
                else None
            )
            stage_policy_eligible = (
                type(stage_maximum) is int and stage_maximum == required
            )
            effective_request_eligible = (
                type(effective_maximum) is int
                and effective_maximum == required
            )
            rows.append(
                {
                    "question_id": str(question.get("question_id", "")),
                    "stage_id": str(stage.get("stage_id", "")),
                    "required_responder_output_token_reserve": required,
                    "stage_request_policy_max_new_tokens": stage_maximum,
                    "effective_completion_max_new_tokens": effective_maximum,
                    "stage_request_policy_eligible": stage_policy_eligible,
                    "effective_answer_request_eligible": (
                        effective_request_eligible
                    ),
                    "row_protocol_eligible": (
                        stage_policy_eligible and effective_request_eligible
                    ),
                }
            )
    complete_population = len(rows) == (
        int(synthesis.get("question_count", len(synthesis.get("questions", ()))))
        * len(SYNTHESIS_STAGE_IDS)
    )
    rows_eligible = complete_population and all(
        row["row_protocol_eligible"] for row in rows
    )
    protocol_eligible = bool(
        runtime_policy_eligible
        and campaign_policy_eligible
        and rows_eligible
    )
    return {
        "required_responder_output_token_reserve": required,
        "runtime_default_max_new_tokens": runtime_default,
        "runtime_policy_eligible": runtime_policy_eligible,
        "campaign_request_policy_max_new_tokens": campaign_maximum,
        "campaign_request_policy_eligible": campaign_policy_eligible,
        "effective_answer_request_rows": len(rows),
        "complete_stage_population": complete_population,
        "eligible_effective_answer_request_rows": sum(
            row["row_protocol_eligible"] for row in rows
        ),
        "protocol_eligible": protocol_eligible,
        "status": "eligible" if protocol_eligible else "protocol_ineligible",
        "rows": rows,
    }


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


def judge_recall_guarded_cumulative_synthesis(
    synthesis: Mapping[str, Any],
    *,
    retrieval: Mapping[str, Any],
    sample: BenchmarkSample,
    synthesis_sha256: str,
    retrieval_sha256: str,
    runtime: SemanticJudgeRuntime,
    responder_prompt_cap: int = DEFAULT_RESPONDER_PROMPT_CAP,
) -> dict[str, Any]:
    """Judge S1--S3 and return a synthesis/population-bound score artifact."""

    implementation = implementation_sha256()
    planned = _plan_judgments(
        synthesis,
        retrieval=retrieval,
        sample=sample,
        synthesis_sha256=synthesis_sha256,
        retrieval_sha256=retrieval_sha256,
    )
    unique = _unique_prompts(planned)
    runtime_identity = _runtime_identity(runtime)
    authorized = runtime_identity.get("authorized_unique_calls")
    if type(authorized) is not int or authorized != len(unique):
        raise ValueError(
            "judge runtime authorization must exactly equal the unique "
            "prompt population"
        )
    expected_binding = build_semantic_judge_campaign_binding(
        synthesis,
        retrieval=retrieval,
        sample=sample,
        synthesis_sha256=synthesis_sha256,
        retrieval_sha256=retrieval_sha256,
        responder_prompt_cap=responder_prompt_cap,
        authorized_unique_calls=authorized,
    )
    if runtime_identity.get("campaign_binding") != expected_binding:
        raise ValueError("judge runtime belongs to another semantic campaign")
    responder_model = _attest_responder_runtime(synthesis)
    judge_model = _attest_judge_runtime(runtime_identity)

    outcomes: dict[str, dict[str, Any]] = {}
    for messages_sha, messages in unique.items():
        verdict_text = runtime.complete(messages)
        verdict = parse_binary_judge_verdict(verdict_text)
        journal = runtime.last_journal_record
        if not isinstance(journal, Mapping):
            raise RuntimeError("semantic judge runtime omitted journal provenance")
        report = journal.get("completion_report")
        if not isinstance(report, Mapping):
            raise RuntimeError("semantic judge runtime omitted completion report")
        if (
            report.get("messages_sha256") != messages_sha
            or journal.get("completion_sha256")
            != quote_sha256(verdict_text)
        ):
            raise RuntimeError("semantic judge runtime changed prompt/response binding")
        outcomes[messages_sha] = {
            "correct": verdict,
            "judge_output": verdict_text,
            "judge_output_sha256": quote_sha256(verdict_text),
            "call_key_sha256": journal.get("call_key_sha256"),
            "request_journal_sha256": journal.get(
                "request_journal_sha256"
            ),
            "response_journal_sha256": journal.get(
                "response_journal_sha256"
            ),
            "completion_report": dict(report),
        }

    question_rows: list[dict[str, Any]] = []
    by_question: dict[int, dict[str, Any]] = {}
    by_stage: dict[str, list[bool]] = {
        stage_id: [] for stage_id in SYNTHESIS_STAGE_IDS
    }
    for row in planned:
        outcome = outcomes[row.messages_sha256]
        question = by_question.setdefault(
            row.ordinal,
            {
                "ordinal": row.ordinal,
                "question_id": row.question_id,
                "question_sha256": row.question_sha256,
                "gold_answer_sha256": row.gold_answer_sha256,
                "stages": [],
            },
        )
        correct = bool(outcome["correct"])
        by_stage[row.stage_id].append(correct)
        question["stages"].append(
            {
                "stage_id": row.stage_id,
                "prediction_sha256": row.prediction_sha256,
                "judge_messages_sha256": row.messages_sha256,
                **outcome,
            }
        )
    question_rows.extend(by_question[index] for index in sorted(by_question))

    prompt_diagnostics = _responder_prompt_diagnostics(
        synthesis,
        retrieval=retrieval,
        prompt_cap=responder_prompt_cap,
    )
    reserve_diagnostics = _responder_output_reserve_diagnostics(synthesis)
    prompt_rows_by_stage: dict[str, list[Mapping[str, Any]]] = {
        stage_id: [] for stage_id in SYNTHESIS_STAGE_IDS
    }
    for prompt_row in prompt_diagnostics["rows"]:
        prompt_rows_by_stage[str(prompt_row["stage_id"])].append(prompt_row)
    reserve_rows_by_stage: dict[str, list[Mapping[str, Any]]] = {
        stage_id: [] for stage_id in SYNTHESIS_STAGE_IDS
    }
    for reserve_row in reserve_diagnostics["rows"]:
        reserve_rows_by_stage[str(reserve_row["stage_id"])].append(
            reserve_row
        )
    stage_aggregates = []
    for stage_id in SYNTHESIS_STAGE_IDS:
        values = by_stage[stage_id]
        status = _accuracy_status(sum(values), len(values))
        prompt_rows = prompt_rows_by_stage[stage_id]
        local_cap_proven = bool(prompt_rows) and all(
            row["local_prompt_cap_compliant"] is True for row in prompt_rows
        )
        local_violation = any(
            row["local_prompt_cap_compliant"] is False for row in prompt_rows
        )
        provider_violation = any(
            row["provider_prompt_cap_compliant"] is False
            for row in prompt_rows
        )
        prompt_cap_passed = local_cap_proven and not provider_violation
        status["accuracy_and_population_gate_passed"] = status["gate_passed"]
        status["responder_local_prompt_cap_proven"] = local_cap_proven
        status["responder_local_prompt_cap_violation"] = local_violation
        status["available_provider_prompt_cap_violation"] = provider_violation
        status["responder_prompt_cap_gate_passed"] = prompt_cap_passed
        status["gate_passed"] = bool(status["gate_passed"] and prompt_cap_passed)
        if local_violation or provider_violation:
            status["status"] = "responder_prompt_cap_violation"
        elif not local_cap_proven:
            status["status"] = "responder_prompt_cap_not_proven"
        reserve_rows = reserve_rows_by_stage[stage_id]
        output_reserve_eligible = bool(
            reserve_diagnostics["runtime_policy_eligible"]
            and reserve_diagnostics["campaign_request_policy_eligible"]
            and len(reserve_rows) == len(values)
            and all(row["row_protocol_eligible"] for row in reserve_rows)
        )
        status["responder_output_reserve_protocol_eligible"] = (
            output_reserve_eligible
        )
        if not output_reserve_eligible:
            status["gate_passed"] = False
            status["status"] = "protocol_ineligible_responder_output_reserve"
        stage_aggregates.append({"stage_id": stage_id, **status})
    all_values = [value for values in by_stage.values() for value in values]
    passing_stages = [
        row["stage_id"] for row in stage_aggregates if row["gate_passed"]
    ]
    immutable_reports = [
        value["completion_report"] for value in outcomes.values()
    ]
    reported_input = [
        int(report["reported_input_tokens"])
        for report in immutable_reports
        if report.get("reported_input_tokens_available") is True
    ]
    reported_output = [
        int(report["reported_output_tokens"])
        for report in immutable_reports
        if report.get("reported_output_tokens_available") is True
    ]
    reported_total = [
        int(report["reported_total_tokens"])
        for report in immutable_reports
        if report.get("reported_total_tokens_available") is True
    ]
    pooled_accuracy = _accuracy_status(sum(all_values), len(all_values))
    pooled_prompt_cap_passed = bool(
        prompt_diagnostics["all_responder_prompts_proven_within_local_cap"]
        and prompt_diagnostics["provider_prompt_cap_violation_count"] == 0
    )
    pooled_accuracy["accuracy_and_population_gate_passed"] = pooled_accuracy[
        "gate_passed"
    ]
    pooled_accuracy["responder_prompt_cap_gate_passed"] = (
        pooled_prompt_cap_passed
    )
    pooled_accuracy["gate_passed"] = bool(
        pooled_accuracy["gate_passed"] and pooled_prompt_cap_passed
    )
    if not pooled_prompt_cap_passed:
        pooled_accuracy["status"] = (
            "responder_prompt_cap_violation"
            if prompt_diagnostics["local_prompt_cap_violation_count"]
            or prompt_diagnostics["provider_prompt_cap_violation_count"]
            else "responder_prompt_cap_not_proven"
        )
    pooled_accuracy["responder_output_reserve_protocol_eligible"] = (
        reserve_diagnostics["protocol_eligible"]
    )
    if not reserve_diagnostics["protocol_eligible"]:
        pooled_accuracy["gate_passed"] = False
        pooled_accuracy["status"] = (
            "protocol_ineligible_responder_output_reserve"
        )
    result = {
        "format": SEMANTIC_JUDGE_FORMAT,
        "synthesis_sha256": synthesis_sha256,
        "retrieval_sha256": retrieval_sha256,
        "population_identity_sha256": population_identity_sha256(sample),
        "gold_scoring_population_sha256": _gold_population_sha256(sample),
        "question_count": len(sample.questions),
        "stage_ids": list(SYNTHESIS_STAGE_IDS),
        "gold_loaded_posthoc": True,
        "independent_llm_judge": True,
        "responder_model": responder_model,
        "judge_model": judge_model,
        "judge_runtime_identity": runtime_identity,
        "judge_runtime_identity_sha256": identity_sha256(runtime_identity),
        "semantic_judge_policy": dict(SEMANTIC_JUDGE_POLICY),
        "semantic_judge_policy_sha256": SEMANTIC_JUDGE_POLICY_SHA256,
        "semantic_judge_implementation_sha256": implementation,
        "campaign_binding": expected_binding,
        "campaign_binding_sha256": identity_sha256(expected_binding),
        "logical_judgment_count": len(planned),
        "unique_judge_prompt_count": len(unique),
        "deduplicated_logical_judgment_count": len(planned) - len(unique),
        "judge_usage": {
            "unique_journaled_calls": len(immutable_reports),
            "reported_input_tokens_available_calls": len(reported_input),
            "reported_input_tokens": sum(reported_input),
            "reported_output_tokens_available_calls": len(reported_output),
            "reported_output_tokens": sum(reported_output),
            "reported_total_tokens_available_calls": len(reported_total),
            "reported_total_tokens": sum(reported_total),
            "input_token_proxy": sum(
                int(report["input_token_proxy"])
                for report in immutable_reports
            ),
            "output_token_proxy": sum(
                int(report["output_token_proxy"])
                for report in immutable_reports
            ),
            "elapsed_s": sum(
                float(report["elapsed_s"]) for report in immutable_reports
            ),
            "retries": 0,
        },
        "responder_prompt_cap_diagnostics": prompt_diagnostics,
        "responder_output_reserve_diagnostics": reserve_diagnostics,
        "questions": question_rows,
        "stage_aggregates": stage_aggregates,
        "pooled_stage_question_accuracy": pooled_accuracy,
        "target_gate": {
            "target_accuracy": TARGET_ACCURACY,
            "minimum_questions_per_stage": MINIMUM_GATE_QUESTIONS,
            "gate_unit": "one fixed retrieval/synthesis stage",
            "eligible_stage_count": sum(
                bool(
                    row["minimum_population_met"]
                    and row["responder_prompt_cap_gate_passed"]
                    and row[
                        "responder_output_reserve_protocol_eligible"
                    ]
                )
                for row in stage_aggregates
            ),
            "passing_stage_ids": passing_stages,
            "any_stage_passed": bool(passing_stages),
            "responder_local_prompt_cap_status": prompt_diagnostics[
                "local_prompt_cap_status"
            ],
            "responder_provider_prompt_cap_status": prompt_diagnostics[
                "provider_prompt_cap_status"
            ],
            "responder_output_reserve_protocol_eligible": (
                reserve_diagnostics["protocol_eligible"]
            ),
            "status": (
                "protocol_ineligible"
                if not reserve_diagnostics["protocol_eligible"]
                else "pass"
                if passing_stages
                else "not_passed"
            ),
        },
    }
    if implementation_sha256() != implementation:
        raise RuntimeError("semantic judge implementation changed during scoring")
    return result


__all__ = [
    "DEFAULT_RESPONDER_PROMPT_CAP",
    "LOCKED_JUDGE_MODEL",
    "LOCKED_JUDGE_MAX_NEW_TOKENS",
    "LOCKED_RESPONDER_MODEL",
    "MINIMUM_GATE_QUESTIONS",
    "RESPONDER_PROMPT_CAP_SEMANTICS",
    "SEMANTIC_JUDGE_CAMPAIGN_FORMAT",
    "SEMANTIC_JUDGE_FORMAT",
    "SEMANTIC_JUDGE_POLICY",
    "SEMANTIC_JUDGE_POLICY_SHA256",
    "TARGET_ACCURACY",
    "build_semantic_judge_campaign_binding",
    "judge_recall_guarded_cumulative_synthesis",
]
