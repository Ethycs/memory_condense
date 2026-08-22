"""Validate the treatment-side prompt derivation for schema-v3 comparison."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.eval.benchmark import build_judge_prompt
from memory_condense.eval.recall_guarded_cumulative_final_answer import (
    validate_final_answer_artifact,
)

from .compare import (
    PairedComparisonError,
    _canonical_json,
    _mapping,
    _must_equal,
    _text,
    _walk_json,
    canonical_sha256,
)


def verify_treatment_prompt_derivation(
    report: Mapping[str, Any],
    *,
    final_answer_artifact: Mapping[str, Any] | None,
    retrieval: Mapping[str, Any] | None,
    scoring_rows: Sequence[Mapping[str, Any]],
) -> bool:
    """Validate upstream prompts when both referenced artifacts are supplied."""

    if final_answer_artifact is None and retrieval is None:
        return False
    if final_answer_artifact is None or retrieval is None:
        raise PairedComparisonError(
            "fixed-stage prompt derivation requires both the final-answer "
            "artifact and its retrieval"
        )
    artifact = _mapping(
        final_answer_artifact,
        "fixed treatment final-answer artifact",
    )
    retrieval_value = _mapping(retrieval, "fixed treatment retrieval")
    _walk_json(artifact, "fixed treatment final-answer artifact")
    _walk_json(retrieval_value, "fixed treatment retrieval")
    artifact_sha256 = hashlib.sha256(
        (_canonical_json(artifact) + "\n").encode("utf-8")
    ).hexdigest()
    retrieval_sha256 = hashlib.sha256(
        (_canonical_json(retrieval_value) + "\n").encode("utf-8")
    ).hexdigest()
    _must_equal(
        artifact_sha256,
        report["final_answer_artifact_sha256"],
        "fixed treatment final-answer artifact SHA-256",
    )
    _must_equal(
        retrieval_sha256,
        report["retrieval_sha256"],
        "fixed treatment retrieval SHA-256",
    )
    try:
        validate_final_answer_artifact(
            artifact,
            retrieval=retrieval_value,
            artifact_sha256=artifact_sha256,
            retrieval_sha256=retrieval_sha256,
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise PairedComparisonError(
            f"fixed treatment prompt derivation did not validate: {exc}"
        ) from exc
    responder_campaign = _mapping(
        artifact.get("campaign_binding"),
        "fixed treatment final-answer campaign",
    )
    expected = {
        "runtime_identity_sha256": report[
            "responder_runtime_identity_sha256"
        ],
        "responder_prompt_policy_sha256": report[
            "responder_prompt_policy_sha256"
        ],
        "population_identity_sha256": report[
            "population_identity_sha256"
        ],
        "retrieval_sha256": report["retrieval_sha256"],
    }
    for field, expected_value in expected.items():
        _must_equal(
            artifact.get(field),
            expected_value,
            f"fixed treatment final-answer artifact.{field}",
        )
    _must_equal(
        responder_campaign.get("responder_prompt_policy"),
        report["responder_prompt_policy"],
        "fixed treatment derived responder prompt policy",
    )
    artifact_rows = artifact.get("questions")
    score_rows = report.get("questions")
    if not isinstance(artifact_rows, list) or not isinstance(score_rows, list):
        raise PairedComparisonError(
            "fixed treatment answer/score question rows are missing"
        )
    if len(artifact_rows) != len(score_rows):
        raise PairedComparisonError(
            "fixed treatment answer/score question counts differ"
        )
    scoring_by_id = {
        _text(row.get("question_id"), "Mem0 scoring question ID"): row
        for row in scoring_rows
    }
    if len(scoring_by_id) != len(scoring_rows):
        raise PairedComparisonError("Mem0 scoring question IDs must be unique")
    for index, (artifact_row_value, score_row_value) in enumerate(
        zip(artifact_rows, score_rows, strict=True)
    ):
        artifact_row = _mapping(
            artifact_row_value,
            f"fixed treatment final-answer questions[{index}]",
        )
        score_row = _mapping(
            score_row_value,
            f"fixed treatment score questions[{index}]",
        )
        answer = _mapping(
            artifact_row.get("answer"),
            f"fixed treatment final-answer questions[{index}].answer",
        )
        expected_row = {
            "ordinal": artifact_row.get("ordinal"),
            "question_id": artifact_row.get("question_id"),
            "question_sha256": artifact_row.get("question_sha256"),
            "dated_question_sha256": artifact_row.get(
                "dated_question_sha256"
            ),
            "prediction_sha256": answer.get("sha256"),
            "answer_call_key_sha256": artifact_row.get("call_key_sha256"),
            "answer_response_journal_sha256": artifact_row.get(
                "response_journal_sha256"
            ),
        }
        for field, expected_value in expected_row.items():
            _must_equal(
                score_row.get(field),
                expected_value,
                f"fixed treatment answer/score questions[{index}].{field}",
            )
        question_id = _text(
            artifact_row.get("question_id"),
            f"fixed treatment final-answer questions[{index}].question_id",
        )
        scoring_row = scoring_by_id.get(question_id)
        if scoring_row is None:
            raise PairedComparisonError(
                f"Mem0 scoring population omitted treatment question {question_id!r}"
            )
        prediction = _text(
            answer.get("text"),
            f"fixed treatment final-answer questions[{index}].answer.text",
        )
        judge_messages = build_judge_prompt(
            _text(scoring_row.get("question"), "Mem0 scoring question"),
            _text(scoring_row.get("gold_answer"), "Mem0 scoring gold answer"),
            prediction,
        )
        _must_equal(
            score_row.get("judge_messages_sha256"),
            canonical_sha256(judge_messages),
            f"fixed treatment judge questions[{index}].judge_messages_sha256",
        )
        _must_equal(
            score_row.get("judge_prompt_token_proxy"),
            count_chat_prompt_token_proxy(judge_messages),
            f"fixed treatment judge questions[{index}].judge_prompt_token_proxy",
        )
    return True


__all__ = ["verify_treatment_prompt_derivation"]
