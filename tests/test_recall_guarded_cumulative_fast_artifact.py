from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    CAMPAIGN_FORMAT,
    ORIGINAL_1M_RETRIEVAL_SHA256,
    QUESTION_FORMAT,
    RETRIEVAL_FORMAT,
    STAGE_IDS,
    FastArtifactValidationError,
    FastFeatureRow,
    FastRetrievalArtifact,
    load_fast_retrieval_artifact,
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _identity(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)[:-1]).hexdigest()


def _quote(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _evidence(
    evidence_id: str,
    text: str,
    *,
    source_id: str | None = None,
) -> dict[str, str]:
    return {
        "evidence_id": evidence_id,
        "source_id": source_id or f"source-{evidence_id}",
        "text": text,
    }


def _fixture_artifact(
    stage_evidence: list[list[dict[str, str]]] | None = None,
    *,
    retained_request_token_state_bytes: int = 0,
    question_retained_state_bytes: int | None = None,
    raw_question: str = "Which two codes were selected?",
    stage_max_prompt_tokens: list[int] | None = None,
) -> dict[str, object]:
    if question_retained_state_bytes is None:
        question_retained_state_bytes = retained_request_token_state_bytes
    dated_question = (
        "[Question asked at 2026/08/22 (Sat) 12:00]\n" + raw_question
    )
    alpha = _evidence("e-alpha", "  Alpha was selected.\n")
    beta = _evidence("e-beta", "Beta was selected — exactly.")
    # A distinct provenance coordinate with the same feature-bearing text.
    beta_alias = _evidence(
        "e-beta-alias",
        beta["text"],
        source_id="source-beta-alias",
    )
    if stage_evidence is None:
        stage_evidence = [
            [alpha],
            [alpha, beta],
            [alpha, beta, beta_alias],
            [alpha, beta, beta_alias],
        ]
    if stage_max_prompt_tokens is None:
        stage_max_prompt_tokens = [200] * len(STAGE_IDS)

    stages: list[dict[str, object]] = []
    parent_ids: list[str] = []
    parent_receipt_sha: str | None = None
    for index, (stage_id, evidence) in enumerate(
        zip(STAGE_IDS, stage_evidence, strict=True)
    ):
        context = "\n".join(
            f"[{row_index}] {item['text']}"
            for row_index, item in enumerate(evidence, 1)
        )
        messages = [
            {"role": "system", "content": "Answer only from supplied evidence."},
            {
                "role": "user",
                "content": (
                    "Retrieved excerpts from the conversation history:\n"
                    f"{context}\n\nQuestion: {dated_question}\nShort answer:"
                ),
            },
        ]
        selected_ids = [item["evidence_id"] for item in evidence]
        receipt: dict[str, object] = {
            "format": "memory-condense-cumulative-retrieval-stage-v2",
            "stage_id": stage_id,
            "matched_controls_sha256": "1" * 64,
            "method_evidence_sha256": f"{index + 2:064x}",
            "parent_stage_receipt_sha256": parent_receipt_sha,
            "parent_evidence_ids": list(parent_ids),
            "selected_evidence_ids": selected_ids,
            "added_evidence_ids": selected_ids[len(parent_ids) :],
            "admission_status": (
                "root"
                if index == 0
                else "added"
                if len(selected_ids) > len(parent_ids)
                else "no_novel_evidence"
            ),
            "evidence_projection_sha256": _identity(evidence),
            "context_sha256": _quote(context),
            "prompt_messages_sha256": _identity(messages),
            "context_token_proxy": 10 + index,
            "max_context_token_proxy": 100,
            "prompt_token_proxy": 20 + index,
            "max_prompt_token_proxy": stage_max_prompt_tokens[index],
            "responder_output_token_reserve": 32,
        }
        receipt_sha = _identity(receipt)
        receipt["receipt_sha256"] = receipt_sha
        stages.append(
            {
                "stage_id": stage_id,
                "stage_receipt": receipt,
                "provider_messages": messages,
                "evidence": evidence,
            }
        )
        parent_ids = selected_ids
        parent_receipt_sha = receipt_sha

    question_id = "fixture-q"
    question_sha = _quote(raw_question)
    dated_question_sha = _quote(dated_question)
    archived_sample_sha = "9" * 64
    population = {
        "format": "memory-condense-original-1m-development-population-v1",
        "archived_compiled_sample_sha256": archived_sample_sha,
        "transcript_tokens": 1_000_001,
        "turn_count": 5_400,
        "question_count": 1,
        "ordered_question_id_sha256s": [
            _identity({"question_id": question_id})
        ],
        "ordered_question_probe_sha256s": [
            _identity(
                {
                    "question_id": question_id,
                    "question_sha256": question_sha,
                    "dated_question_sha256": dated_question_sha,
                }
            )
        ],
    }
    population_sha = _identity(population)
    retrieval_implementation_sha = "a" * 64
    retrieval_policy_sha = "e" * 64
    compilation_receipt_sha = "8" * 64
    source_store_receipt: dict[str, object] = {
        "format": "memory-condense-fast-source-test-v1",
        "turn_count": 5_400,
    }
    source_store_receipt["receipt_sha256"] = _identity(source_store_receipt)
    combined_store_receipt: dict[str, object] = {
        "format": "memory-condense-recall-guarded-combined-store-v1",
        "turn_count": 5_400,
        "retrieval_policy_sha256": retrieval_policy_sha,
        "compilation_receipt_sha256": compilation_receipt_sha,
        "retained_request_token_state_bytes": retained_request_token_state_bytes,
    }
    combined_store_receipt["receipt_sha256"] = _identity(combined_store_receipt)
    s0_receipt = stages[0]["stage_receipt"]
    assert isinstance(s0_receipt, dict)
    predecessor_receipt: dict[str, object] = {
        "format": "memory-condense-causal-coverage-predecessor-v1",
        "prompt_messages_sha256": s0_receipt["prompt_messages_sha256"],
        "prompt_token_proxy": s0_receipt["prompt_token_proxy"],
        "max_prompt_token_proxy": s0_receipt["max_prompt_token_proxy"],
        "responder_output_token_reserve": s0_receipt[
            "responder_output_token_reserve"
        ],
        "protected_context_sha256": s0_receipt["context_sha256"],
        "retrieval_policy_sha256": retrieval_policy_sha,
        "retained_request_token_state_bytes": question_retained_state_bytes,
    }
    predecessor_receipt["receipt_sha256"] = _identity(predecessor_receipt)
    final_receipt = stages[-1]["stage_receipt"]
    assert isinstance(final_receipt, dict)
    retrieval_receipt: dict[str, object] = {
        "format": "memory-condense-recall-guarded-cumulative-retrieval-v1",
        "predecessor_receipt_sha256": predecessor_receipt["receipt_sha256"],
        "protected_evidence_ids": [
            item["evidence_id"] for item in stage_evidence[0]
        ],
        "final_evidence_ids": [
            item["evidence_id"] for item in stage_evidence[-1]
        ],
        "final_context_sha256": final_receipt["context_sha256"],
        "prompt_messages_sha256": final_receipt["prompt_messages_sha256"],
        "context_token_proxy": final_receipt["context_token_proxy"],
        "max_context_token_proxy": final_receipt["max_context_token_proxy"],
        "prompt_token_proxy": final_receipt["prompt_token_proxy"],
        "max_prompt_token_proxy": final_receipt["max_prompt_token_proxy"],
        "responder_output_token_reserve": final_receipt[
            "responder_output_token_reserve"
        ],
        "matched_controls_sha256": final_receipt["matched_controls_sha256"],
        "stage_admission_statuses": [
            stage["stage_receipt"]["admission_status"]  # type: ignore[index]
            for stage in stages[1:]
        ],
        "retained_request_token_state_bytes": question_retained_state_bytes,
    }
    retrieval_receipt["receipt_sha256"] = _identity(retrieval_receipt)
    question_part: dict[str, object] = {
        "format": QUESTION_FORMAT,
        "population_identity_sha256": population_sha,
        "ordinal": 0,
        "question_id": question_id,
        "question_sha256": question_sha,
        "dated_question_sha256": dated_question_sha,
        "combined_store_receipt_sha256": combined_store_receipt[
            "receipt_sha256"
        ],
        "retrieval_implementation_sha256": retrieval_implementation_sha,
        "retrieval_receipt": retrieval_receipt,
        "predecessor_receipt": predecessor_receipt,
        "stage_ids": list(STAGE_IDS),
        "stages": stages,
        "provider_calls": 0,
    }
    return {
        "format": RETRIEVAL_FORMAT,
        "campaign_format": CAMPAIGN_FORMAT,
        "population_identity": population,
        "population_identity_sha256": population_sha,
        "archived_compiled_sample_sha256": archived_sample_sha,
        "source_store_receipt": source_store_receipt,
        "combined_store_receipt": combined_store_receipt,
        "compilation_receipt_sha256": compilation_receipt_sha,
        "retrieval_implementation_sha256": retrieval_implementation_sha,
        "retrieval_policy_sha256": retrieval_policy_sha,
        "transcript_tokens": 1_000_001,
        "turn_count": 5_400,
        "question_count": 1,
        "question_part_sha256s": [
            hashlib.sha256(_canonical_bytes(question_part)).hexdigest()
        ],
        "questions": [question_part],
        "stage_ids": list(STAGE_IDS),
        "provider_calls": 0,
        "gold_fields_present": False,
    }


def _publish(tmp_path: Path, value: dict[str, object]) -> tuple[Path, str]:
    path = tmp_path / "retrieval.json"
    payload = _canonical_bytes(value)
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    path.with_name(path.name + ".sha256").write_bytes(
        f"{digest}  {path.name}\n".encode("ascii")
    )
    return path, digest


def test_loads_exact_immutable_rows_and_deduplicates_feature_work(tmp_path: Path):
    path, digest = _publish(tmp_path, _fixture_artifact())
    before = {item.name: item.read_bytes() for item in tmp_path.iterdir()}

    artifact = load_fast_retrieval_artifact(path, expected_sha256=digest)

    assert isinstance(artifact, FastRetrievalArtifact)
    assert artifact.raw_sha256 == digest
    assert artifact.stage_ids == STAGE_IDS
    assert artifact.question_count == 1
    assert artifact.logical_feature_row_count == 9
    assert artifact.unique_feature_row_count == 2
    question = artifact.questions[0]
    assert question.question_id == "fixture-q"
    assert question.question == "Which two codes were selected?"
    assert question.dated_question == (
        "[Question asked at 2026/08/22 (Sat) 12:00]\n"
        "Which two codes were selected?"
    )
    assert question.question_parse_receipt.question_form == "dated_header"
    assert question.question_parse_receipt.matching_framing_candidates == 1
    assert question.final_user_message == question.stages[-1].provider_messages[-1]
    assert question.final_user_message.content.endswith(
        f"Question: {question.dated_question}\nShort answer:"
    )

    s0, s1, s2, s3 = question.stages
    assert s0.evidence_ids == ("e-alpha",)
    assert s1.evidence_ids == ("e-alpha", "e-beta")
    assert s2.evidence_ids == ("e-alpha", "e-beta", "e-beta-alias")
    assert s3.evidence_ids == s2.evidence_ids
    assert s0.evidence_ids == s1.evidence_ids[: len(s0.evidence_ids)]
    assert s1.evidence_ids == s2.evidence_ids[: len(s1.evidence_ids)]
    assert s2.evidence_ids == s3.evidence_ids[: len(s2.evidence_ids)]
    assert s0.exact_texts == ("  Alpha was selected.\n",)
    assert s2.source_ids[-1] == "source-beta-alias"
    assert tuple(stage.feature_row_indices for stage in question.stages) == (
        (0,),
        (0, 1),
        (0, 1, 1),
        (0, 1, 1),
    )
    assert tuple(row.evidence_text for row in question.feature_rows) == (
        "  Alpha was selected.\n",
        "Beta was selected — exactly.",
    )
    assert before == {item.name: item.read_bytes() for item in tmp_path.iterdir()}

    with pytest.raises(FrozenInstanceError):
        artifact.raw_sha256 = "0" * 64  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        question.feature_rows[0].evidence_text = "changed"  # type: ignore[misc]
    assert {field.name for field in fields(FastFeatureRow)} == {
        "question",
        "evidence_text",
        "row_sha256",
    }


def test_requires_a_digest_anchor_and_checks_raw_bytes(tmp_path: Path):
    path, digest = _publish(tmp_path, _fixture_artifact())

    with pytest.raises(FastArtifactValidationError, match="SHA-256 mismatch"):
        load_fast_retrieval_artifact(path, expected_sha256="0" * 64)
    with pytest.raises(FastArtifactValidationError, match="expected SHA-256"):
        load_fast_retrieval_artifact(path, verify_sidecar=False)

    artifact = load_fast_retrieval_artifact(
        path,
        expected_sha256=digest,
        verify_sidecar=False,
    )
    assert artifact.raw_sha256 == digest


def test_rejects_an_invalid_digest_sidecar(tmp_path: Path):
    path, digest = _publish(tmp_path, _fixture_artifact())
    path.with_name(path.name + ".sha256").write_text(
        f"{'0' * 64}  {path.name}\n", encoding="ascii", newline="\n"
    )

    with pytest.raises(FastArtifactValidationError, match="sidecar is invalid"):
        load_fast_retrieval_artifact(path, expected_sha256=digest)


def test_rejects_a_non_prefix_cumulative_stage(tmp_path: Path):
    alpha = _evidence("e-alpha", "Alpha")
    beta = _evidence("e-beta", "Beta")
    path, digest = _publish(
        tmp_path,
        _fixture_artifact(
            [
                [alpha],
                [beta, alpha],
                [beta, alpha],
                [beta, alpha],
            ]
        ),
    )

    with pytest.raises(FastArtifactValidationError, match="ordered-prefix"):
        load_fast_retrieval_artifact(path, expected_sha256=digest)


def test_rejects_changed_payload_at_a_repeated_evidence_id(tmp_path: Path):
    alpha = _evidence("e-alpha", "Alpha")
    changed_alpha = _evidence("e-alpha", "Alpha changed")
    beta = _evidence("e-beta", "Beta")
    path, digest = _publish(
        tmp_path,
        _fixture_artifact(
            [
                [alpha],
                [changed_alpha, beta],
                [changed_alpha, beta],
                [changed_alpha, beta],
            ]
        ),
    )

    with pytest.raises(FastArtifactValidationError, match="changed evidence payload"):
        load_fast_retrieval_artifact(path, expected_sha256=digest)


@pytest.mark.parametrize(
    ("artifact_state_bytes", "question_state_bytes", "message"),
    (
        (8, 0, "combined-store receipt persisted"),
        (0, 8, "persisted transformer request state"),
    ),
)
def test_rejects_any_declared_persisted_transformer_request_state(
    tmp_path: Path,
    artifact_state_bytes: int,
    question_state_bytes: int,
    message: str,
):
    path, digest = _publish(
        tmp_path,
        _fixture_artifact(
            retained_request_token_state_bytes=artifact_state_bytes,
            question_retained_state_bytes=question_state_bytes,
        ),
    )

    with pytest.raises(FastArtifactValidationError, match=message):
        load_fast_retrieval_artifact(path, expected_sha256=digest)


def test_question_recovery_uses_both_hashes_when_markers_occur_in_content(
    tmp_path: Path,
):
    raw_question = "Which literal follows?\n\nQuestion: cobalt"
    alpha = _evidence("e-alpha", "Context decoy.\n\nQuestion: amber")
    path, digest = _publish(
        tmp_path,
        _fixture_artifact(
            [[alpha], [alpha], [alpha], [alpha]],
            raw_question=raw_question,
        ),
    )

    artifact = load_fast_retrieval_artifact(path, expected_sha256=digest)
    question = artifact.questions[0]

    assert question.question == raw_question
    assert question.question_parse_receipt.question_marker_occurrences == 3
    assert question.question_parse_receipt.matching_framing_candidates == 1


def test_rejects_a_stage_that_changes_the_hard_prompt_budget(tmp_path: Path):
    path, digest = _publish(
        tmp_path,
        _fixture_artifact(stage_max_prompt_tokens=[200, 201, 201, 201]),
    )

    with pytest.raises(
        FastArtifactValidationError,
        match="changed cumulative controls or hard budgets",
    ):
        load_fast_retrieval_artifact(path, expected_sha256=digest)


def test_accepts_an_explicit_sidecar_path_with_exact_filename_binding(
    tmp_path: Path,
):
    path, digest = _publish(tmp_path, _fixture_artifact())
    sibling = path.with_name(path.name + ".sha256")
    custom = tmp_path / "sealed.digest"
    custom.write_bytes(sibling.read_bytes())
    sibling.unlink()

    artifact = load_fast_retrieval_artifact(
        path,
        expected_sha256=digest,
        sidecar_path=custom,
    )

    assert artifact.raw_sha256 == digest


def test_rejects_non_standard_nonfinite_json_even_when_sidecar_matches(
    tmp_path: Path,
):
    value = _fixture_artifact()
    value["ignored_nonfinite"] = float("nan")
    path, digest = _publish(tmp_path, value)

    with pytest.raises(FastArtifactValidationError, match="non-standard JSON"):
        load_fast_retrieval_artifact(path, expected_sha256=digest)


def test_loads_the_sealed_1m_artifact_without_rebuilding_any_store():
    path = Path(
        "eval_results/longmemeval-1m-recall-guarded-cumulative-"
        "development-20260821/retrieval.json"
    )
    if not path.is_file():
        pytest.skip("sealed local 1M retrieval artifact is not present")
    before = path.read_bytes()

    artifact = load_fast_retrieval_artifact(
        path,
        expected_sha256=ORIGINAL_1M_RETRIEVAL_SHA256,
    )

    assert artifact.raw_sha256 == ORIGINAL_1M_RETRIEVAL_SHA256
    assert artifact.question_count == 10
    assert artifact.transcript_tokens == 1_039_203
    assert artifact.turn_count == 5_400
    assert artifact.logical_feature_row_count == 1_939
    assert artifact.unique_feature_row_count == 530
    assert artifact.retained_request_token_state_bytes == 0
    assert all(
        question.retained_request_token_state_bytes == 0
        for question in artifact.questions
    )
    assert path.read_bytes() == before
