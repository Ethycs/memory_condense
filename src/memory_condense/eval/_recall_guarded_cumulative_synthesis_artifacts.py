"""Validation and publication of cumulative-synthesis artifacts."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval.benchmark import exact_match
from memory_condense.eval._recall_guarded_cumulative_synthesis_contracts import (
    ANSWER_REUSE_FORMAT,
    ANSWER_REUSE_RULE,
    ANSWERABILITY_BAND_THRESHOLDS,
    EVIDENCE_DENSITY_PER_100_TOKEN_THRESHOLDS,
    EvidenceDensity,
    EvidenceRole,
    SYNTHESIS_FORMAT,
    SYNTHESIS_PROMPT_POLICY,
    SYNTHESIS_QUESTION_FORMAT,
    SYNTHESIS_STAGE_IDS,
    _band,
    _evidence_rows,
    _projection_sha,
    _require_sha256,
    _sha256_text,
    _stage_receipt_sha256,
    _sum_usage,
    _usage_delta,
    _validated_prompt_policy,
    build_synthesis_messages,
    validate_published_retrieval,
)


def _bound_stage(stage: Mapping[str, Any]) -> dict[str, Any]:
    receipt = stage.get("stage_receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("bound stage has no receipt")
    return {
        "stage_id": str(stage.get("stage_id", "")),
        "stage_receipt_sha256": _stage_receipt_sha256(stage),
        "evidence_projection_sha256": _projection_sha(stage),
        "prompt_messages_sha256": _require_sha256(
            receipt.get("prompt_messages_sha256"),
            "bound prompt messages SHA-256",
        ),
        "evidence": _evidence_rows(stage),
    }


def _validate_score_binding(
    row: Mapping[str, Any],
    *,
    evidence: Mapping[str, str],
    policy_sha256: str,
) -> None:
    if (
        row.get("source_id") != evidence["source_id"]
        or row.get("evidence_text_sha256") != _sha256_text(evidence["text"])
    ):
        raise ValueError("episodic score changed its bound evidence")
    for name in (
        "answerability",
        "value_evidence_logit",
        "direct_log_likelihood",
        "indirect_log_likelihood",
        "answerability_per_100_tokens",
    ):
        value = row.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("episodic score contains a non-numeric value")
        if not math.isfinite(float(value)):
            raise ValueError("episodic score contains a non-finite value")
    answerability = float(row["answerability"])
    if not 0.0 <= answerability <= 1.0:
        raise ValueError("episodic answerability must be in [0, 1]")
    tokens = max(1, count_tokens(evidence["text"]))
    density = 100.0 * answerability / tokens
    if row.get("token_count_proxy") != tokens or not math.isclose(
        float(row["answerability_per_100_tokens"]), density,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("episodic token-density accounting changed")
    if row.get("answerability_band") != _band(
        answerability, ANSWERABILITY_BAND_THRESHOLDS
    ):
        raise ValueError("episodic answerability band changed")
    if row.get("evidence_density_band") != _band(
        density, EVIDENCE_DENSITY_PER_100_TOKEN_THRESHOLDS
    ):
        raise ValueError("episodic evidence-density band changed")
    if row.get("evidence_density_policy_sha256") != policy_sha256:
        raise ValueError("episodic evidence-density policy changed")
    if row.get("calibrated") is not False or row.get("inspected") is not True:
        raise ValueError("episodic score changed inspection/calibration status")


def _validate_derived_score(
    row: Mapping[str, Any],
    *,
    text: str,
    policy_sha256: str,
) -> None:
    synthetic_evidence = {
        "source_id": "derived",
        "text": text,
    }
    bound = dict(row)
    bound["source_id"] = "derived"
    bound["evidence_text_sha256"] = _sha256_text(text)
    _validate_score_binding(
        bound,
        evidence=synthetic_evidence,
        policy_sha256=policy_sha256,
    )


def _validate_claim_score_rows(
    stage: Mapping[str, Any],
    *,
    policy_sha256: str,
) -> None:
    claims = stage.get("claims")
    claim_scores = stage.get("claim_scores")
    if not isinstance(claims, list) or not isinstance(claim_scores, list) or [
        row.get("claim_id") if isinstance(row, Mapping) else None
        for row in claim_scores
    ] != [
        row.get("claim_id") if isinstance(row, Mapping) else None
        for row in claims
    ]:
        raise ValueError("synthesis claim-score population changed")
    for claim, score in zip(claims, claim_scores, strict=True):
        assert isinstance(claim, Mapping)
        assert isinstance(score, Mapping)
        answer_value = score.get("answer_value")
        citation_support = score.get("citation_support_proxy")
        if not isinstance(answer_value, Mapping) or not isinstance(
            citation_support, Mapping
        ):
            raise ValueError("synthesis claim score is incomplete")
        _validate_derived_score(
            answer_value,
            text=str(claim["text"]),
            policy_sha256=policy_sha256,
        )
        quotes = "\n".join(
            str(citation["quote"])
            for citation in claim.get("citations", ())
            if isinstance(citation, Mapping)
        )
        _validate_derived_score(
            citation_support,
            text=quotes,
            policy_sha256=policy_sha256,
        )


def _validate_claims_and_labels(
    stage: Mapping[str, Any],
    *,
    bound: Mapping[str, Any],
    root_evidence_ids: set[str],
    allow_unnormalized_abstention: bool = False,
) -> None:
    evidence_rows = bound.get("evidence")
    if not isinstance(evidence_rows, list):
        raise ValueError("bound stage evidence is missing")
    evidence_by_id = {
        str(row["evidence_id"]): row
        for row in evidence_rows
        if isinstance(row, Mapping)
    }
    expected_label_ids = [
        evidence_id
        for evidence_id in evidence_by_id
        if evidence_id not in root_evidence_ids
    ]
    labels = stage.get("evidence_labels")
    if not isinstance(labels, list) or [
        row.get("evidence_id") if isinstance(row, Mapping) else None
        for row in labels
    ] != expected_label_ids:
        raise ValueError("synthesis label population changed")

    claims = stage.get("claims")
    if not isinstance(claims, list):
        raise ValueError("synthesis claims must be a list")
    claim_ids = [
        str(row.get("claim_id", "")) if isinstance(row, Mapping) else ""
        for row in claims
    ]
    if not all(claim_ids) or len(claim_ids) != len(set(claim_ids)):
        raise ValueError("synthesis claim population changed")
    claim_set = set(claim_ids)
    answer = stage.get("answer")
    if not isinstance(answer, Mapping) or not isinstance(answer.get("text"), str):
        raise ValueError("synthesis answer is missing")
    answer_claims = answer.get("claim_ids")
    if not isinstance(answer_claims, list) or len(answer_claims) != len(
        set(answer_claims)
    ) or not set(answer_claims).issubset(claim_set):
        raise ValueError("synthesis answer claim binding changed")
    if exact_match(str(answer["text"]), "I don't know"):
        if (answer_claims or claims) and not allow_unnormalized_abstention:
            raise ValueError("abstention retained active claims")
    elif not answer_claims:
        raise ValueError("non-abstaining synthesis has no answer claim")

    for claim in claims:
        assert isinstance(claim, Mapping)
        if not isinstance(claim.get("text"), str) or not claim["text"]:
            raise ValueError("synthesis claim text is missing")
        citations = claim.get("citations")
        if not isinstance(citations, list) or not citations:
            raise ValueError("synthesis claim has no citations")
        seen_citations: set[tuple[str, str]] = set()
        for citation in citations:
            if not isinstance(citation, Mapping):
                raise ValueError("synthesis citation must be an object")
            evidence_id = str(citation.get("evidence_id", ""))
            evidence = evidence_by_id.get(evidence_id)
            quote = citation.get("quote")
            if evidence is None or not isinstance(quote, str) or not quote:
                raise ValueError("synthesis citation is incomplete")
            if (
                citation.get("source_id") != evidence["source_id"]
                or citation.get("evidence_text_sha256")
                != _sha256_text(str(evidence["text"]))
                or quote not in str(evidence["text"])
                or citation.get("quote_sha256") != quote_sha256(quote)
            ):
                raise ValueError("synthesis citation changed its exact evidence")
            key = (evidence_id, quote)
            if key in seen_citations:
                raise ValueError("synthesis claim contains a duplicate citation")
            seen_citations.add(key)

    for label in labels:
        assert isinstance(label, Mapping)
        evidence = evidence_by_id[str(label["evidence_id"])]
        if (
            label.get("source_id") != evidence["source_id"]
            or label.get("evidence_text_sha256")
            != _sha256_text(str(evidence["text"]))
            or label.get("role") not in EvidenceRole.__args__
            or label.get("density") not in EvidenceDensity.__args__
        ):
            raise ValueError("synthesis evidence label changed its binding")
        supports = label.get("supports_claim_ids")
        if not isinstance(supports, list) or len(supports) != len(set(supports)):
            raise ValueError("synthesis label claim binding changed")
        if not set(supports).issubset(claim_set):
            raise ValueError("synthesis label references an unknown claim")


def _new_episodic_ids(
    bound: Mapping[str, Any],
    *,
    source_bound: Mapping[str, Any],
    root_evidence_ids: set[str],
) -> list[str]:
    source_ids = {
        str(row["evidence_id"])
        for row in source_bound.get("evidence", ())
        if isinstance(row, Mapping)
    }
    return [
        str(row["evidence_id"])
        for row in bound.get("evidence", ())
        if isinstance(row, Mapping)
        and str(row["evidence_id"]) not in root_evidence_ids
        and str(row["evidence_id"]) not in source_ids
    ]


def _labels_are_monotonic_noops(
    stage: Mapping[str, Any],
    evidence_ids: Sequence[str],
) -> bool:
    labels = stage.get("evidence_labels")
    if not isinstance(labels, list):
        return False
    by_id = {
        str(row.get("evidence_id")): row
        for row in labels
        if isinstance(row, Mapping)
    }
    return all(
        evidence_id in by_id
        and by_id[evidence_id].get("density") == "none"
        and by_id[evidence_id].get("role") == "irrelevant"
        and by_id[evidence_id].get("supports_claim_ids") == []
        for evidence_id in evidence_ids
    )


def _effective_reuse_labels(
    source: Mapping[str, Any],
    generated_labels: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Carry prior claim bindings forward and retain only truly new labels."""

    source_labels = source.get("evidence_labels")
    if not isinstance(source_labels, list):
        raise ValueError("monotonic answer reuse source labels are missing")
    source_by_id = {
        str(row.get("evidence_id")): row
        for row in source_labels
        if isinstance(row, Mapping)
    }
    return [
        copy.deepcopy(
            source_by_id.get(str(label.get("evidence_id")), label)
        )
        for label in generated_labels
    ]


def _validate_monotonic_answer_reuse(
    stage: Mapping[str, Any],
    *,
    bound: Mapping[str, Any],
    root_evidence_ids: set[str],
    prior_stages: Mapping[str, Mapping[str, Any]],
    prior_bounds: Mapping[str, Mapping[str, Any]],
    require_when_triggered: bool,
    allow_unnormalized_abstention: bool = False,
) -> None:
    """Validate a claim-consistent override and its discarded generation."""

    receipt = stage.get("monotonic_answer_reuse")
    immediate_stage = list(prior_stages.values())[-1] if prior_stages else None
    immediate_bound = list(prior_bounds.values())[-1] if prior_bounds else None
    immediate_new_ids = (
        []
        if immediate_bound is None
        else _new_episodic_ids(
            bound,
            source_bound=immediate_bound,
            root_evidence_ids=root_evidence_ids,
        )
    )
    trigger = (
        immediate_stage is not None
        and stage.get("reused_from_stage_id") is None
        and _labels_are_monotonic_noops(stage, immediate_new_ids)
    )
    if receipt is None:
        if require_when_triggered and trigger:
            raise ValueError("monotonic answer reuse was omitted for no-op additions")
        return
    if not isinstance(receipt, Mapping):
        raise ValueError("monotonic answer reuse receipt must be an object")
    expected_receipt_fields = {
        "format",
        "rule",
        "from_stage_id",
        "new_evidence_ids",
        "new_evidence_labels_sha256",
        "generated_answer",
        "generated_claims",
        "generated_evidence_labels",
        "generated_answer_sha256",
        "generated_claims_sha256",
        "generated_evidence_labels_sha256",
        "source_answer_sha256",
        "source_claims_sha256",
        "source_evidence_labels_sha256",
        "effective_evidence_labels_sha256",
        "receipt_sha256",
    }
    if set(receipt) != expected_receipt_fields:
        raise ValueError("monotonic answer reuse receipt fields changed")
    source_stage_id = str(receipt.get("from_stage_id", ""))
    source = prior_stages.get(source_stage_id)
    source_bound = prior_bounds.get(source_stage_id)
    if source is None or source_bound is None:
        raise ValueError("monotonic answer reuse source is not an earlier stage")
    if (
        stage.get("reused_from_stage_id") is None
        and immediate_stage is not None
        and source_stage_id != str(immediate_stage.get("stage_id"))
    ):
        raise ValueError(
            "monotonic answer reuse source is not the immediate predecessor"
        )
    new_ids = _new_episodic_ids(
        bound,
        source_bound=source_bound,
        root_evidence_ids=root_evidence_ids,
    )
    generated_answer = receipt.get("generated_answer")
    generated_claims = receipt.get("generated_claims")
    generated_labels = receipt.get("generated_evidence_labels")
    if (
        not isinstance(generated_answer, Mapping)
        or not isinstance(generated_claims, list)
        or not isinstance(generated_labels, list)
        or any(not isinstance(row, Mapping) for row in generated_labels)
    ):
        raise ValueError("monotonic answer reuse omitted generated synthesis")
    generated_by_id = {
        str(row.get("evidence_id")): row for row in generated_labels
    }
    new_labels = [
        generated_by_id[evidence_id]
        for evidence_id in new_ids
        if evidence_id in generated_by_id
    ]
    effective_labels = _effective_reuse_labels(source, generated_labels)
    body = dict(receipt)
    declared_receipt_sha = body.pop("receipt_sha256", None)
    if (
        receipt.get("format") != ANSWER_REUSE_FORMAT
        or receipt.get("rule") != ANSWER_REUSE_RULE
        or receipt.get("new_evidence_ids") != new_ids
        or not _labels_are_monotonic_noops(
            {"evidence_labels": generated_labels}, new_ids
        )
        or receipt.get("new_evidence_labels_sha256")
        != identity_sha256(new_labels)
        or receipt.get("generated_answer_sha256")
        != identity_sha256(generated_answer)
        or receipt.get("generated_claims_sha256")
        != identity_sha256(generated_claims)
        or receipt.get("generated_evidence_labels_sha256")
        != identity_sha256(generated_labels)
        or receipt.get("source_answer_sha256")
        != identity_sha256(source.get("answer"))
        or receipt.get("source_claims_sha256")
        != identity_sha256(source.get("claims"))
        or receipt.get("source_evidence_labels_sha256")
        != identity_sha256(source.get("evidence_labels"))
        or receipt.get("effective_evidence_labels_sha256")
        != identity_sha256(effective_labels)
        or declared_receipt_sha != identity_sha256(body)
    ):
        raise ValueError("monotonic answer reuse receipt changed")
    if (
        stage.get("answer") != source.get("answer")
        or stage.get("claims") != source.get("claims")
        or stage.get("evidence_labels") != effective_labels
    ):
        raise ValueError(
            "monotonic answer reuse changed the prior effective synthesis"
        )
    generated_stage = dict(stage)
    generated_stage["answer"] = generated_answer
    generated_stage["claims"] = generated_claims
    generated_stage["evidence_labels"] = generated_labels
    _validate_claims_and_labels(
        generated_stage,
        bound=bound,
        root_evidence_ids=root_evidence_ids,
        allow_unnormalized_abstention=allow_unnormalized_abstention,
    )


def _validate_question_part(
    source: Mapping[str, Any],
    part: Mapping[str, Any],
    *,
    retrieval_sha256: str,
) -> list[dict[str, Any]]:
    if (
        part.get("format") != SYNTHESIS_QUESTION_FORMAT
        or part.get("retrieval_sha256") != retrieval_sha256
        or part.get("ordinal") != source.get("ordinal")
        or part.get("question_id") != source.get("question_id")
        or part.get("question_sha256") != source.get("question_sha256")
        or part.get("gold_fields_present") is not False
    ):
        raise ValueError("synthesis question part belongs to another retrieval")
    implementation = _require_sha256(
        part.get("synthesis_implementation_sha256"),
        "synthesis implementation SHA-256",
    )
    policy, policy_sha = _validated_prompt_policy(
        part.get("synthesis_prompt_policy"),
        part.get("synthesis_prompt_policy_sha256"),
    )
    runtime_identity = part.get("runtime_identity")
    if not isinstance(runtime_identity, Mapping) or part.get(
        "runtime_identity_sha256"
    ) != identity_sha256(runtime_identity):
        raise ValueError("synthesis runtime identity changed")
    request_policy = part.get("request_policy")
    if not isinstance(request_policy, Mapping) or part.get(
        "request_policy_sha256"
    ) != identity_sha256(request_policy):
        raise ValueError("synthesis request policy changed")

    source_stages = source.get("stages")
    if not isinstance(source_stages, list) or len(source_stages) != 4:
        raise ValueError("retrieval source stages changed")
    bound_stages = [_bound_stage(stage) for stage in source_stages[1:]]
    root_ids = {
        row["evidence_id"] for row in _evidence_rows(source_stages[0])
    }
    final_rows = _evidence_rows(source_stages[-1])
    final_by_id = {row["evidence_id"]: row for row in final_rows}
    expected_scores = [
        row["evidence_id"] for row in final_rows if row["evidence_id"] not in root_ids
    ]
    score_rows = part.get("episodic_evidence_scores")
    if not isinstance(score_rows, list) or [
        row.get("evidence_id") if isinstance(row, Mapping) else None
        for row in score_rows
    ] != expected_scores:
        raise ValueError("episodic score population changed")
    if part.get("episodic_evidence_count") != len(expected_scores):
        raise ValueError("episodic score count changed")
    for score in score_rows:
        assert isinstance(score, Mapping)
        _validate_score_binding(
            score,
            evidence=final_by_id[str(score["evidence_id"])],
            policy_sha256=policy_sha,
        )

    stages = part.get("stages")
    if not isinstance(stages, list) or tuple(
        row.get("stage_id") if isinstance(row, Mapping) else None for row in stages
    ) != SYNTHESIS_STAGE_IDS:
        raise ValueError("synthesis question stages changed")
    origins: dict[str, Mapping[str, Any]] = {}
    prior_stages: dict[str, Mapping[str, Any]] = {}
    prior_bounds: dict[str, Mapping[str, Any]] = {}
    for source_stage, bound, stage in zip(
        source_stages[1:], bound_stages, stages, strict=True
    ):
        assert isinstance(source_stage, Mapping)
        assert isinstance(stage, Mapping)
        messages, _aliases_by_id, novel_aliases = build_synthesis_messages(
            source_stage,
            root_evidence_ids=root_ids,
            prompt_policy=policy,
        )
        source_prompt_sha = identity_sha256(source_stage["provider_messages"])
        structured_prompt_sha = identity_sha256(messages)
        expected_key = identity_sha256(
            {
                "question_sha256": source.get("question_sha256"),
                "evidence_projection_sha256": bound["evidence_projection_sha256"],
                "source_prompt_messages_sha256": source_prompt_sha,
                "structured_prompt_messages_sha256": structured_prompt_sha,
                "runtime_identity_sha256": part["runtime_identity_sha256"],
                "synthesis_prompt_policy_sha256": policy_sha,
                "request_policy_sha256": part["request_policy_sha256"],
            }
        )
        if (
            stage.get("synthesis_key_sha256") != expected_key
            or stage.get("evidence_projection_sha256")
            != bound["evidence_projection_sha256"]
            or stage.get("source_stage_receipt_sha256")
            != bound["stage_receipt_sha256"]
            or stage.get("source_prompt_messages_sha256") != source_prompt_sha
            or stage.get("structured_prompt_messages_sha256")
            != structured_prompt_sha
            or stage.get("runtime_identity_sha256")
            != part["runtime_identity_sha256"]
            or stage.get("synthesis_implementation_sha256") != implementation
            or stage.get("synthesis_prompt_policy_sha256") != policy_sha
            or stage.get("request_policy") != request_policy
            or stage.get("request_policy_sha256")
            != part["request_policy_sha256"]
            or stage.get("episodic_evidence_count") != len(novel_aliases)
        ):
            raise ValueError("synthesis stage identity changed")
        mode = stage.get("synthesis_mode")
        expected_prompt = (
            structured_prompt_sha
            if mode == "structured_generation"
            else source_prompt_sha
            if mode == "short_answer_with_forced_choice_attribution"
            else None
        )
        if expected_prompt is None or stage.get("prompt_messages_sha256") != expected_prompt:
            raise ValueError("synthesis effective prompt binding changed")
        raw = stage.get("raw_completion")
        if not isinstance(raw, str) or stage.get("raw_completion_sha256") != _sha256_text(raw):
            raise ValueError("synthesis raw completion hash changed")
        completion_report = stage.get("completion_report")
        if not isinstance(completion_report, Mapping):
            raise ValueError("synthesis completion report is missing")
        if completion_report.get("completion_sha256") not in {
            None,
            stage["raw_completion_sha256"],
        } or completion_report.get("messages_sha256") not in {
            None,
            expected_prompt,
        }:
            raise ValueError("synthesis completion report changed")
        attempt = stage.get("structured_attempt")
        if attempt is not None:
            if not isinstance(attempt, Mapping):
                raise ValueError("structured attempt must be an object")
            attempt_raw = attempt.get("raw_completion")
            if (
                not isinstance(attempt_raw, str)
                or attempt.get("raw_completion_sha256") != _sha256_text(attempt_raw)
                or attempt.get("prompt_messages_sha256") != structured_prompt_sha
            ):
                raise ValueError("structured attempt binding changed")
        _validate_monotonic_answer_reuse(
            stage,
            bound=bound,
            root_evidence_ids=root_ids,
            prior_stages=prior_stages,
            prior_bounds=prior_bounds,
            require_when_triggered=policy == SYNTHESIS_PROMPT_POLICY,
            allow_unnormalized_abstention=True,
        )
        _validate_claims_and_labels(
            stage,
            bound=bound,
            root_evidence_ids=root_ids,
            allow_unnormalized_abstention=True,
        )
        _validate_claim_score_rows(stage, policy_sha256=policy_sha)
        reused_from = stage.get("reused_from_stage_id")
        if reused_from is None:
            origins[expected_key] = stage
        else:
            origin = origins.get(expected_key)
            if origin is None or origin.get("stage_id") != reused_from:
                raise ValueError("synthesis stage reuse changed its origin")
            for name in (
                "raw_completion",
                "raw_completion_sha256",
                "completion_report",
                "synthesis_mode",
                "structured_attempt",
                "answer",
                "claims",
                "evidence_labels",
                "claim_scores",
                "claim_score_reports",
                "monotonic_answer_reuse",
            ):
                if stage.get(name) != origin.get(name):
                    raise ValueError("reused synthesis output differs from its origin")
        prior_stages[str(stage["stage_id"])] = stage
        prior_bounds[str(stage["stage_id"])] = bound

    usage_delta = part.get("runtime_usage_delta")
    if not isinstance(usage_delta, Mapping):
        raise ValueError("synthesis question has no authoritative usage delta")
    usage_before = part.get("runtime_usage_before")
    usage_after = part.get("runtime_usage_cumulative")
    if (
        not isinstance(usage_before, Mapping)
        or not isinstance(usage_after, Mapping)
        or dict(usage_delta) != _usage_delta(usage_before, usage_after)
    ):
        raise ValueError("synthesis runtime usage delta changed")
    unique_rows = [row for row in stages if row.get("reused_from_stage_id") is None]
    expected_completion_calls = len(unique_rows) + sum(
        row.get("structured_attempt") is not None for row in unique_rows
    )
    if usage_delta.get("completion_calls") != expected_completion_calls:
        raise ValueError("synthesis completion-call accounting changed")
    expected_score_calls = 1
    for row in unique_rows:
        if row.get("attribution_score_report") is not None:
            expected_score_calls += 1
        claim_count = len(row.get("claims", ()))
        if claim_count:
            expected_score_calls += 1 + claim_count
    if usage_delta.get("score_calls") != expected_score_calls:
        raise ValueError("synthesis score-call accounting changed")
    return bound_stages


def assemble_synthesis_artifact(
    retrieval: Mapping[str, Any],
    *,
    retrieval_sha256: str,
    question_parts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Cross-bind independently checkpointed question syntheses."""

    validate_published_retrieval(retrieval)
    questions = retrieval["questions"]
    if len(question_parts) != len(questions):
        raise ValueError("synthesis question part count changed")
    normalized: list[dict[str, Any]] = []
    identity: Mapping[str, Any] | None = None
    runtime_identity_sha: str | None = None
    synthesis_implementation: str | None = None
    synthesis_prompt_policy: Mapping[str, Any] | None = None
    synthesis_prompt_policy_sha: str | None = None
    request_policy: Mapping[str, Any] | None = None
    request_policy_sha: str | None = None
    checkpoint_part_hashes: list[str] = []
    usage_rows: list[Mapping[str, Any]] = []
    for ordinal, (source, part) in enumerate(zip(questions, question_parts, strict=True)):
        if part.get("ordinal") != ordinal:
            raise ValueError("synthesis question ordinal changed")
        bound_stages = _validate_question_part(
            source,
            part,
            retrieval_sha256=retrieval_sha256,
        )
        current_identity = part.get("runtime_identity")
        if not isinstance(current_identity, Mapping):
            raise ValueError("synthesis question part has no runtime identity")
        if identity is None:
            identity = current_identity
        elif dict(identity) != dict(current_identity):
            raise ValueError("synthesis question parts used different runtimes")
        current_runtime_sha = str(part["runtime_identity_sha256"])
        current_implementation = str(part["synthesis_implementation_sha256"])
        current_synthesis_policy = part["synthesis_prompt_policy"]
        current_synthesis_policy_sha = str(
            part["synthesis_prompt_policy_sha256"]
        )
        current_request_policy = part["request_policy"]
        current_request_sha = str(part["request_policy_sha256"])
        if runtime_identity_sha is None:
            runtime_identity_sha = current_runtime_sha
            synthesis_implementation = current_implementation
            synthesis_prompt_policy = current_synthesis_policy
            synthesis_prompt_policy_sha = current_synthesis_policy_sha
            request_policy = current_request_policy
            request_policy_sha = current_request_sha
        elif (
            runtime_identity_sha != current_runtime_sha
            or synthesis_implementation != current_implementation
            or synthesis_prompt_policy != current_synthesis_policy
            or synthesis_prompt_policy_sha != current_synthesis_policy_sha
            or request_policy != current_request_policy
            or request_policy_sha != current_request_sha
        ):
            raise ValueError("synthesis question parts changed campaign identity")
        enriched = copy.deepcopy(dict(part))
        enriched["bound_stages"] = bound_stages
        enriched["root_evidence_ids"] = [
            row["evidence_id"] for row in _evidence_rows(source["stages"][0])
        ]
        normalized.append(enriched)
        checkpoint_part_hashes.append(identity_sha256(dict(part)))
        usage = part.get("runtime_usage_delta")
        assert isinstance(usage, Mapping)
        usage_rows.append(usage)
    unique_calls = sum(
        row.get("reused_from_stage_id") is None
        for part in normalized
        for row in part["stages"]
    )
    artifact = {
        "format": SYNTHESIS_FORMAT,
        "retrieval_sha256": retrieval_sha256,
        "population_identity_sha256": retrieval.get("population_identity_sha256"),
        "runtime_identity": dict(identity or {}),
        "runtime_identity_sha256": runtime_identity_sha,
        "synthesis_implementation_sha256": synthesis_implementation,
        "synthesis_prompt_policy": dict(synthesis_prompt_policy or {}),
        "synthesis_prompt_policy_sha256": synthesis_prompt_policy_sha,
        "request_policy": dict(request_policy or {}),
        "request_policy_sha256": request_policy_sha,
        "gold_fields_present": False,
        "stage_ids": list(SYNTHESIS_STAGE_IDS),
        "question_count": len(normalized),
        "unique_synthesis_calls": unique_calls,
        "checkpoint_question_part_identity_sha256s": checkpoint_part_hashes,
        "authoritative_runtime_usage": _sum_usage(usage_rows),
        "episodic_evidence_count": sum(
            int(part["episodic_evidence_count"]) for part in normalized
        ),
        "questions": normalized,
    }
    artifact["question_identity_sha256s"] = [
        identity_sha256(part) for part in normalized
    ]
    usage = artifact["authoritative_runtime_usage"]
    if usage.get("completion_calls") != sum(
        int(part["runtime_usage_delta"]["completion_calls"])
        for part in normalized
    ) or usage.get("score_calls") != sum(
        int(part["runtime_usage_delta"]["score_calls"])
        for part in normalized
    ):
        raise ValueError("assembled runtime usage is not authoritative")
    return artifact


def normalize_fallback_abstentions(
    synthesis: Mapping[str, Any],
    *,
    source_synthesis_sha256: str,
) -> dict[str, Any]:
    """Publish a gold-free correction for punctuated abstentions.

    Raw completions and their reports remain byte-identical.  Only the
    mechanically generated claim/citation attribution is cleared when SQuAD
    normalization identifies the responder output as ``I don't know``. Reuse
    receipts are refreshed without discarding the model's generated branch.
    Every removed attribution is retained under an explicit audit field.
    """

    _validate_assembled_synthesis(
        synthesis,
        allow_unnormalized_abstention=True,
    )
    normalized = copy.deepcopy(dict(synthesis))
    changed = 0
    generated_changed = 0
    for question in normalized.get("questions", []):
        for stage in question.get("stages", []):
            answer = str(stage.get("answer", {}).get("text", ""))
            if not exact_match(answer, "I don't know"):
                stage["abstention_normalized"] = False
            else:
                old_claims = stage.get("claims", [])
                old_scores = stage.get("claim_scores", [])
                old_reports = stage.get("claim_score_reports", [])
                if old_claims or old_scores or old_reports:
                    discarded = {
                        "claims": old_claims,
                        "claim_scores": old_scores,
                        "claim_score_reports": old_reports,
                    }
                    stage["pre_normalization_abstention_attribution"] = {
                        **discarded,
                        "identity_sha256": identity_sha256(discarded),
                    }
                stage["answer"]["claim_ids"] = []
                stage["claims"] = []
                stage["claim_scores"] = []
                stage["claim_score_reports"] = []
                for label in stage.get("evidence_labels", []):
                    label["supports_claim_ids"] = []
                stage["abstention_normalized"] = True
                changed += 1

            receipt = stage.get("monotonic_answer_reuse")
            if not isinstance(receipt, dict):
                continue
            generated_answer = receipt.get("generated_answer")
            generated_claims = receipt.get("generated_claims")
            generated_labels = receipt.get("generated_evidence_labels")
            if (
                not isinstance(generated_answer, dict)
                or not isinstance(generated_claims, list)
                or not isinstance(generated_labels, list)
            ):
                raise ValueError(
                    "monotonic answer reuse omitted generated synthesis"
                )
            if not exact_match(
                str(generated_answer.get("text", "")), "I don't know"
            ):
                stage["generated_abstention_normalized"] = False
                continue
            discarded_generated = {
                "answer": copy.deepcopy(generated_answer),
                "claims": copy.deepcopy(generated_claims),
                "evidence_labels": copy.deepcopy(generated_labels),
            }
            stage[
                "pre_normalization_generated_abstention_attribution"
            ] = {
                **discarded_generated,
                "identity_sha256": identity_sha256(discarded_generated),
            }
            generated_answer["claim_ids"] = []
            receipt["generated_claims"] = []
            for label in generated_labels:
                if not isinstance(label, dict):
                    raise ValueError(
                        "monotonic generated evidence label is invalid"
                    )
                label["supports_claim_ids"] = []
            stage["generated_abstention_normalized"] = True
            generated_changed += 1

        prior_stages: dict[str, Mapping[str, Any]] = {}
        for stage in question.get("stages", []):
            receipt = stage.get("monotonic_answer_reuse")
            if isinstance(receipt, dict):
                source = prior_stages.get(str(receipt.get("from_stage_id", "")))
                if source is None:
                    raise ValueError(
                        "monotonic answer reuse source is not an earlier stage"
                    )
                generated_answer = receipt["generated_answer"]
                generated_claims = receipt["generated_claims"]
                generated_labels = receipt["generated_evidence_labels"]
                generated_by_id = {
                    str(label["evidence_id"]): label
                    for label in generated_labels
                    if isinstance(label, Mapping)
                }
                new_labels = [
                    generated_by_id[evidence_id]
                    for evidence_id in receipt["new_evidence_ids"]
                ]
                effective_labels = _effective_reuse_labels(
                    source, generated_labels
                )
                stage["evidence_labels"] = effective_labels
                receipt.update(
                    {
                        "new_evidence_labels_sha256": identity_sha256(
                            new_labels
                        ),
                        "generated_answer_sha256": identity_sha256(
                            generated_answer
                        ),
                        "generated_claims_sha256": identity_sha256(
                            generated_claims
                        ),
                        "generated_evidence_labels_sha256": identity_sha256(
                            generated_labels
                        ),
                        "source_answer_sha256": identity_sha256(
                            source.get("answer")
                        ),
                        "source_claims_sha256": identity_sha256(
                            source.get("claims")
                        ),
                        "source_evidence_labels_sha256": identity_sha256(
                            source.get("evidence_labels")
                        ),
                        "effective_evidence_labels_sha256": identity_sha256(
                            effective_labels
                        ),
                    }
                )
                receipt_body = dict(receipt)
                receipt_body.pop("receipt_sha256", None)
                receipt["receipt_sha256"] = identity_sha256(receipt_body)
            prior_stages[str(stage["stage_id"])] = stage
    normalized["normalization"] = {
        "kind": "squad_normalized_abstention_v2",
        "source_synthesis_sha256": source_synthesis_sha256,
        "raw_completions_changed": False,
        "gold_fields_read": False,
        "normalized_stage_rows": changed,
        "normalized_generated_reuse_rows": generated_changed,
    }
    normalized["question_identity_sha256s"] = [
        identity_sha256(question)
        for question in normalized.get("questions", [])
    ]
    _validate_assembled_synthesis(normalized)
    return normalized


def _validate_assembled_synthesis(
    synthesis: Mapping[str, Any],
    *,
    allow_unnormalized_abstention: bool = False,
) -> None:
    if synthesis.get("format") != SYNTHESIS_FORMAT:
        raise ValueError("unexpected synthesis artifact format")
    if synthesis.get("gold_fields_present") is not False:
        raise ValueError("synthesis artifact crossed the gold firewall")
    if tuple(synthesis.get("stage_ids", ())) != SYNTHESIS_STAGE_IDS:
        raise ValueError("synthesis stage population changed")
    _require_sha256(synthesis.get("retrieval_sha256"), "parent retrieval SHA-256")
    _require_sha256(
        synthesis.get("population_identity_sha256"),
        "synthesis population identity SHA-256",
    )
    _require_sha256(
        synthesis.get("synthesis_implementation_sha256"),
        "synthesis implementation SHA-256",
    )
    policy, policy_sha = _validated_prompt_policy(
        synthesis.get("synthesis_prompt_policy"),
        synthesis.get("synthesis_prompt_policy_sha256"),
    )
    runtime_identity = synthesis.get("runtime_identity")
    if not isinstance(runtime_identity, Mapping) or synthesis.get(
        "runtime_identity_sha256"
    ) != identity_sha256(runtime_identity):
        raise ValueError("assembled synthesis runtime identity changed")
    request_policy = synthesis.get("request_policy")
    if not isinstance(request_policy, Mapping) or synthesis.get(
        "request_policy_sha256"
    ) != identity_sha256(request_policy):
        raise ValueError("assembled synthesis request policy changed")
    questions = synthesis.get("questions")
    if not isinstance(questions, list) or synthesis.get("question_count") != len(
        questions
    ):
        raise ValueError("assembled synthesis question population changed")
    if synthesis.get("question_identity_sha256s") != [
        identity_sha256(question) for question in questions
    ]:
        raise ValueError("assembled synthesis question hashes changed")

    usage_rows: list[Mapping[str, Any]] = []
    total_evidence = 0
    unique_calls = 0
    for ordinal, question in enumerate(questions):
        if not isinstance(question, Mapping):
            raise ValueError("assembled synthesis question must be an object")
        if (
            question.get("format") != SYNTHESIS_QUESTION_FORMAT
            or question.get("ordinal") != ordinal
            or question.get("gold_fields_present") is not False
            or question.get("runtime_identity") != runtime_identity
            or question.get("runtime_identity_sha256")
            != synthesis.get("runtime_identity_sha256")
            or question.get("synthesis_implementation_sha256")
            != synthesis.get("synthesis_implementation_sha256")
            or question.get("synthesis_prompt_policy") != policy
            or question.get("synthesis_prompt_policy_sha256") != policy_sha
            or question.get("request_policy") != request_policy
            or question.get("request_policy_sha256")
            != synthesis.get("request_policy_sha256")
        ):
            raise ValueError("assembled synthesis question identity changed")
        bound_stages = question.get("bound_stages")
        stages = question.get("stages")
        root_ids_raw = question.get("root_evidence_ids")
        if (
            not isinstance(bound_stages, list)
            or not isinstance(stages, list)
            or not isinstance(root_ids_raw, list)
            or tuple(row.get("stage_id") for row in bound_stages)
            != SYNTHESIS_STAGE_IDS
            or tuple(row.get("stage_id") for row in stages)
            != SYNTHESIS_STAGE_IDS
        ):
            raise ValueError("assembled synthesis bound stages changed")
        root_ids = {str(value) for value in root_ids_raw}
        if len(root_ids) != len(root_ids_raw):
            raise ValueError("assembled synthesis root evidence changed")
        final_evidence = bound_stages[-1].get("evidence")
        if not isinstance(final_evidence, list):
            raise ValueError("assembled synthesis final evidence is missing")
        final_by_id = {
            str(row["evidence_id"]): row
            for row in final_evidence
            if isinstance(row, Mapping)
        }
        expected_scores = [
            evidence_id for evidence_id in final_by_id if evidence_id not in root_ids
        ]
        score_rows = question.get("episodic_evidence_scores")
        if not isinstance(score_rows, list) or [
            row.get("evidence_id") if isinstance(row, Mapping) else None
            for row in score_rows
        ] != expected_scores:
            raise ValueError("assembled episodic score population changed")
        if question.get("episodic_evidence_count") != len(expected_scores):
            raise ValueError("assembled episodic evidence count changed")
        total_evidence += len(expected_scores)
        for score in score_rows:
            assert isinstance(score, Mapping)
            _validate_score_binding(
                score,
                evidence=final_by_id[str(score["evidence_id"])],
                policy_sha256=policy_sha,
            )
        origins: dict[str, Mapping[str, Any]] = {}
        prior_stages: dict[str, Mapping[str, Any]] = {}
        prior_bounds: dict[str, Mapping[str, Any]] = {}
        for bound, stage in zip(bound_stages, stages, strict=True):
            if not isinstance(bound, Mapping) or not isinstance(stage, Mapping):
                raise ValueError("assembled synthesis stage must be an object")
            if (
                stage.get("evidence_projection_sha256")
                != bound.get("evidence_projection_sha256")
                or stage.get("source_stage_receipt_sha256")
                != bound.get("stage_receipt_sha256")
                or stage.get("source_prompt_messages_sha256")
                != bound.get("prompt_messages_sha256")
                or stage.get("runtime_identity_sha256")
                != synthesis.get("runtime_identity_sha256")
                or stage.get("synthesis_implementation_sha256")
                != synthesis.get("synthesis_implementation_sha256")
                or stage.get("synthesis_prompt_policy_sha256") != policy_sha
                or stage.get("request_policy_sha256")
                != synthesis.get("request_policy_sha256")
            ):
                raise ValueError("assembled synthesis stage binding changed")
            _validate_monotonic_answer_reuse(
                stage,
                bound=bound,
                root_evidence_ids=root_ids,
                prior_stages=prior_stages,
                prior_bounds=prior_bounds,
                require_when_triggered=policy == SYNTHESIS_PROMPT_POLICY,
                allow_unnormalized_abstention=allow_unnormalized_abstention,
            )
            _validate_claims_and_labels(
                stage,
                bound=bound,
                root_evidence_ids=root_ids,
                allow_unnormalized_abstention=allow_unnormalized_abstention,
            )
            raw = stage.get("raw_completion")
            if not isinstance(raw, str) or stage.get(
                "raw_completion_sha256"
            ) != _sha256_text(raw):
                raise ValueError("assembled synthesis raw completion changed")
            report = stage.get("completion_report")
            if not isinstance(report, Mapping) or report.get(
                "completion_sha256"
            ) not in {None, stage["raw_completion_sha256"]} or report.get(
                "messages_sha256"
            ) not in {None, stage.get("prompt_messages_sha256")}:
                raise ValueError("assembled synthesis completion report changed")
            attempt = stage.get("structured_attempt")
            if attempt is not None:
                if not isinstance(attempt, Mapping):
                    raise ValueError("assembled structured attempt changed")
                attempt_raw = attempt.get("raw_completion")
                if not isinstance(attempt_raw, str) or attempt.get(
                    "raw_completion_sha256"
                ) != _sha256_text(attempt_raw):
                    raise ValueError("assembled structured attempt hash changed")
            _validate_claim_score_rows(stage, policy_sha256=policy_sha)
            key = _require_sha256(
                stage.get("synthesis_key_sha256"),
                "synthesis memoization key",
            )
            reused_from = stage.get("reused_from_stage_id")
            if reused_from is None:
                origins[key] = stage
                unique_calls += 1
            else:
                origin = origins.get(key)
                if origin is None or origin.get("stage_id") != reused_from:
                    raise ValueError("assembled synthesis reuse origin changed")
                for name in (
                    "raw_completion",
                    "raw_completion_sha256",
                    "completion_report",
                    "synthesis_mode",
                    "structured_attempt",
                    "answer",
                    "claims",
                    "evidence_labels",
                    "claim_scores",
                    "claim_score_reports",
                    "monotonic_answer_reuse",
                ):
                    if stage.get(name) != origin.get(name):
                        raise ValueError("assembled reused synthesis output changed")
            prior_stages[str(stage["stage_id"])] = stage
            prior_bounds[str(stage["stage_id"])] = bound
        usage_delta = question.get("runtime_usage_delta")
        if not isinstance(usage_delta, Mapping):
            raise ValueError("assembled synthesis usage delta is missing")
        usage_rows.append(usage_delta)
    if synthesis.get("episodic_evidence_count") != total_evidence:
        raise ValueError("assembled total episodic evidence count changed")
    if synthesis.get("unique_synthesis_calls") != unique_calls:
        raise ValueError("assembled unique synthesis-call count changed")
    if synthesis.get("authoritative_runtime_usage") != _sum_usage(usage_rows):
        raise ValueError("assembled authoritative runtime usage changed")
