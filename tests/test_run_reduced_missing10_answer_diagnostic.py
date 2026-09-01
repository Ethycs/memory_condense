from __future__ import annotations

from pathlib import Path

import pytest

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256
from tools.matched_eval.artifacts import SealedArtifact, publish_sealed_json
from tools.matched_eval.contracts import assert_gold_blind, identity_sha256
from tools import run_reduced_missing10_answer_diagnostic as diagnostic


def _sha(label: str) -> str:
    return quote_sha256(label)


def _observation(
    *,
    namespace: str,
    method_index: int,
    row_index: int,
    shared: bool = False,
) -> dict[str, object]:
    quote = "Shared exact fact." if shared else f"Method {method_index} fact {row_index}."
    chunk = _sha("shared chunk") if shared else _sha(f"chunk {method_index} {row_index}")
    turn = "shared-turn" if shared else f"turn-{method_index}-{row_index}"
    observation = {
        "candidate_id": _sha(f"candidate {method_index} {row_index} {shared}"),
        "chunk_id": chunk,
        "created_at": "2023-06-01T12:00:00-07:00",
        "discovery_rank": row_index,
        "namespace_id": namespace,
        "observation_sha256": _sha(
            f"observation {method_index} {row_index} {shared}"
        ),
        "quote": quote,
        "quote_sha256": quote_sha256(quote),
        "role": "user",
        "source_id": "source-shared" if shared else f"source-{method_index}",
        "span_end_char": len(quote),
        "span_start_char": 0,
        "token_count": count_tokens(quote),
        "turn_id": turn,
    }
    return observation


def _fixtures() -> tuple[SealedArtifact, SealedArtifact]:
    reduced_rows = []
    parent_by_ordinal: dict[int, dict[str, object]] = {}
    for ordinal in diagnostic.EXPECTED_ORDINALS:
        question_id = f"question-{ordinal}"
        question = f"What happened for memory question {ordinal}?"
        dated = f"[Question asked at 2023/07/01] {question}"
        namespace = _sha(f"namespace {ordinal}")
        methods = []
        for method_index, method_id in enumerate(
            diagnostic.FACT_METHOD_IDS, start=1
        ):
            observations = []
            if ordinal != 72:
                observations = [
                    _observation(
                        namespace=namespace,
                        method_index=method_index,
                        row_index=0,
                        shared=True,
                    ),
                    _observation(
                        namespace=namespace,
                        method_index=method_index,
                        row_index=1,
                    ),
                ]
            methods.append(
                {
                    "callback_selected_candidate_count": len(observations),
                    "callback_selected_candidate_tokens": sum(
                        int(row["token_count"]) for row in observations
                    ),
                    "callback_selected_candidates": observations,
                    "method_id": method_id,
                    "method_receipt_sha256": _sha(
                        f"method {ordinal} {method_id}"
                    ),
                }
            )
        reduced_rows.append(
            {
                "dated_question_sha256": quote_sha256(dated),
                "methods": methods,
                "namespace_id": namespace,
                "ordinal": ordinal,
                "question_id": question_id,
                "question_sha256": quote_sha256(question),
            }
        )
        prediction = f"Parent prediction {ordinal}"
        typed_evidence = {"format": "synthetic-compact-parent", "items": []}
        parent_by_ordinal[ordinal] = {
            "dated_question_sha256": quote_sha256(dated),
            "ordinal": ordinal,
            "parent_prediction": prediction,
            "parent_prediction_sha256": quote_sha256(prediction),
            "provider_projection": {
                "messages_sha256": _sha(f"parent messages {ordinal}"),
                "prompt_token_proxy": 6_900,
                "provider_input": {
                    "dated_question": dated,
                    "typed_evidence": typed_evidence,
                },
            },
            "question_id": question_id,
            "question_sha256": quote_sha256(question),
        }

    reduced_payload = {
        "construction_identity_sha256": (
            "a58fbb31b08d7255b54a4dd48952e3039bc65d9de48af647955303a876c3f623"
        ),
        "format": (
            "memory-condense-reduced-second-read-retrieval-assay-v3-construction"
        ),
        "gold_loaded": False,
        "new_provider_calls": 0,
        "ordinals": list(diagnostic.EXPECTED_ORDINALS),
        "question_count": 10,
        "questions": reduced_rows,
        "retained_transformer_token_state_bytes": 0,
    }
    parent_rows = [
        parent_by_ordinal.get(ordinal, {"ordinal": ordinal})
        for ordinal in range(100)
    ]
    parent_payload = {
        "format": "memory-condense-typed-memory-final-arm-v1-composition-v1",
        "gold_loaded": False,
        "new_provider_calls": 0,
        "questions": parent_rows,
        "retained_transformer_token_state_bytes": 0,
    }
    return (
        SealedArtifact(Path("reduced.json"), "a" * 64, reduced_payload),
        SealedArtifact(Path("parent.json"), "b" * 64, parent_payload),
    )


def test_construct_is_deterministic_gold_blind_and_hard_capped() -> None:
    reduced, parent = _fixtures()
    first = diagnostic.build_construction_payload(reduced, parent)
    second = diagnostic.build_construction_payload(reduced, parent)

    assert first == second
    assert first["provider_ready"] is True
    assert first["gold_loaded"] is False
    assert first["provider_calls"] == 0
    assert first["stage"] == "callback_selected_union_delta_only"
    assert first["maximum_full_chat_plus_output_tokens"] <= 8_000
    assert_gold_blind(first)


def test_dedup_runs_after_four_independent_callback_selections() -> None:
    reduced, parent = _fixtures()
    payload = diagnostic.build_construction_payload(reduced, parent)
    row = payload["questions"][0]
    lanes = row["lane_ledgers"]

    assert [lane["input_selected_count"] for lane in lanes] == [2, 2, 2, 2]
    assert lanes[0]["post_dedup_count"] == 2
    assert [lane["post_dedup_count"] for lane in lanes[1:]] == [1, 1, 1]
    assert not lanes[0]["dedup_exclusions"]
    assert all(len(lane["dedup_exclusions"]) == 1 for lane in lanes[1:])
    assert all(not lane["omitted_observation_sha256s"] for lane in lanes)
    assert all(
        lane["protected_content_token_proxy"]
        <= lane["protected_content_token_cap"]
        for lane in lanes
    )


def test_q72_remains_an_empty_lane_control() -> None:
    reduced, parent = _fixtures()
    payload = diagnostic.build_construction_payload(reduced, parent)
    row = next(value for value in payload["questions"] if value["ordinal"] == 72)

    assert row["allowed_evidence_ids"] == []
    assert all(lane["input_selected_count"] == 0 for lane in row["lane_ledgers"])
    assert row["full_chat_plus_output_tokens"] <= 8_000


def test_frozen_artifact_digest_tamper_is_rejected(tmp_path: Path) -> None:
    artifact, _ = publish_sealed_json(tmp_path / "source.json", {"format": "fixture"})

    with pytest.raises(diagnostic.ReducedMissing10DiagnosticError):
        diagnostic._read_frozen(  # noqa: SLF001
            artifact.path,
            "0" * 64,
            label="fixture",
        )


def test_completion_parser_falls_back_and_validates_evidence_ids() -> None:
    valid = diagnostic._parse_completion(  # noqa: SLF001
        '{"decision":"replace","prediction":"New answer","used_evidence_ids":["F1-001"]}',
        parent_prediction="Parent",
        allowed_evidence_ids=("F1-001",),
    )
    invalid = diagnostic._parse_completion(  # noqa: SLF001
        '{"decision":"replace","prediction":"Bad","used_evidence_ids":["X"]}',
        parent_prediction="Parent",
        allowed_evidence_ids=("F1-001",),
    )

    assert valid == ("New answer", "provider_replace", ("F1-001",), "valid")
    assert invalid[0] == "Parent"
    assert invalid[1] == "invalid_schema_parent_fallback"
