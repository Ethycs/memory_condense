from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools.matched_eval.artifacts import SealedArtifact, publish_sealed_json
from tools.matched_eval.contracts import MatchedEvalContractError, identity_sha256
from tools.matched_eval.specialist_scoped_completion import (
    FORMAT as SCOPED_COMPLETION_FORMAT,
    HARD_COMPLETE_CHAT_TOKEN_CAP,
    MAX_CHAT_PROMPT_TOKENS,
    OUTPUT_TOKEN_RESERVE,
    PROMPT_FORMAT,
    VALIDATION_CONTRACT_FORMAT,
    render_specialist_scoped_prompt,
)
from tools.run_reduced_specialist_answer_v3 import (
    DEFAULT_CONSTRUCTION,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT,
    EXPECTED_ORDINALS,
    EXPECTED_PROVIDER_CALLS,
    FORMAT,
    ReducedSpecialistAnswerV3Error,
    _materialization_projection,
    _preflight_projection,
    _prompt_plan_row,
    _read_construction,
    _validate_preflight,
    build_parser,
)
from tools.run_reduced_specialist_retrieval_assay_v3 import CONSTRUCTION_FORMAT


def _sha(label: str) -> str:
    return quote_sha256(label)


def _semantic_row(label: str) -> dict[str, Any]:
    return {
        "action_concepts": [],
        "completed_action_concepts": [],
        "date": "2026-08-01",
        "entity_terms": ["user", "tea", label],
        "group_terms": [],
        "item_receipt_sha256": _sha(f"item {label}"),
        "numeric_value": None,
        "relation_terms": [],
        "semantic_unit_sha256": _sha(f"semantic {label}"),
        "status": "completed",
        "summary_terms": ["user", "likes", "tea", label],
        "supported_slot_ids": [],
        "unit": None,
    }


def _construction_question(ordinal: int, index: int) -> dict[str, Any]:
    question = f"What tea preference did I record for memory {ordinal}?"
    dated_question = f"[Question asked at 2026/08/28] {question}"
    handle_id = f"H{700001 + index}"
    group_handle = f"G{700001 + index}"
    candidate_id = _sha(f"candidate {ordinal}")
    label = f"blend{ordinal}"
    semantic = _semantic_row(label)
    advisory = {
        "candidate_handle_map": {candidate_id: handle_id},
        "mechanism_id": f"profile_preference_specialist_v3_{ordinal}",
        "purpose": "personalize from one coherent preference cluster",
    }
    validation = {
        "by_handle": {
            handle_id: {
                "semantic_rows": [semantic],
                "status_values": ["completed"],
                "usable_item_receipt_sha256s": [
                    semantic["item_receipt_sha256"]
                ],
            }
        },
        "format": VALIDATION_CONTRACT_FORMAT,
        "include_proposed": False,
        "question_terms": ["what", "tea", "preference", "record", "memory"],
    }
    fitted_input = {
        "dated_question": dated_question,
        "protected_parent_fallback": {
            "label": "fallback_not_evidence",
            "prediction": f"Parent fallback {ordinal}",
            "prediction_sha256": _sha(f"Parent fallback {ordinal}"),
        },
        "response_schema": {
            "decision": "keep_parent|replace",
            "prediction": "nonempty exact text",
            "used_handle_ids": [handle_id],
        },
        "typed_evidence": {
            "handles": [
                {"group_handle": group_handle, "handle_id": handle_id}
            ],
            "items": [],
        },
    }
    provider_input = {
        **fitted_input,
        "specialist_advisories": [advisory],
    }
    prompt = render_specialist_scoped_prompt(provider_input)
    fitted_receipt = _sha(f"fitted {ordinal}")
    terminal_receipt_body = {
        "fitted_prompt_receipt_sha256": fitted_receipt,
        "message_renderer_format": PROMPT_FORMAT,
        "messages_sha256": identity_sha256(list(prompt.messages)),
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "prompt_token_proxy": prompt.prompt_token_proxy,
        "provider_input_sha256": identity_sha256(provider_input),
        "specialist_advisories_sha256": prompt.specialist_advisories_sha256,
        "specialist_prompt_envelope_receipt_sha256": prompt.receipt_sha256,
    }
    body = {
        "dated_question_sha256": _sha(dated_question),
        "fitted_typed_prompt": {
            "allowed_handle_ids": [handle_id],
            "provider_input": fitted_input,
            "receipt_sha256": fitted_receipt,
            "validation_contract": validation,
        },
        "ordinal": ordinal,
        "question_id": f"q{ordinal}",
        "question_sha256": _sha(question),
        "terminal_prompt": {
            "fitted_prompt_receipt_sha256": fitted_receipt,
            "full_chat_plus_output_tokens": (
                prompt.prompt_token_proxy + OUTPUT_TOKEN_RESERVE
            ),
            "hard_prompt_token_cap": HARD_COMPLETE_CHAT_TOKEN_CAP,
            "message_renderer_format": PROMPT_FORMAT,
            "messages_sha256": identity_sha256(list(prompt.messages)),
            "output_token_reserve": OUTPUT_TOKEN_RESERVE,
            "prompt_token_proxy": prompt.prompt_token_proxy,
            "provider_input": provider_input,
            "provider_prompt_count": 0,
            "retained_transformer_token_state_bytes": 0,
            "specialist_advisories_sha256": prompt.specialist_advisories_sha256,
            "specialist_prompt_envelope_receipt_sha256": prompt.receipt_sha256,
            "terminal_prompt_receipt_sha256": identity_sha256(
                terminal_receipt_body
            ),
        },
    }
    return {**body, "question_receipt_sha256": identity_sha256(body)}


def _construction(tmp_path: Path) -> SealedArtifact:
    questions = [
        _construction_question(ordinal, index)
        for index, ordinal in enumerate(EXPECTED_ORDINALS)
    ]
    payload = {
        "format": CONSTRUCTION_FORMAT,
        "gold_loaded": False,
        "new_provider_calls": 0,
        "ordinals": list(EXPECTED_ORDINALS),
        "question_count": EXPECTED_PROVIDER_CALLS,
        "questions": questions,
        "retained_transformer_token_state_bytes": 0,
        "target_labels_loaded": False,
        "target_plan_loaded": False,
    }
    payload["construction_identity_sha256"] = identity_sha256(payload)
    artifact, created = publish_sealed_json(
        tmp_path / "reduced-specialist-construction-v3.json", payload
    )
    assert created
    return artifact


def _preflight_artifact(
    tmp_path: Path,
) -> tuple[SealedArtifact, tuple[dict[str, Any], ...]]:
    construction = _construction(tmp_path)
    source, rows = _read_construction(construction.path, construction.sha256)
    payload = _preflight_projection(
        source,
        rows,
        model=DEFAULT_MODEL,
        gateway_url="https://central-dev.zt:4000/v1",
        max_concurrency=4,
    )
    return (
        SealedArtifact(
            Path("synthetic-preflight-v3.json"),
            identity_sha256(payload),
            payload,
        ),
        rows,
    )


class _FakeBatch:
    def __init__(self, rows: tuple[dict[str, Any], ...]) -> None:
        self.logical_completions = tuple(
            json.dumps(
                {
                    "decision": "keep_parent",
                    "prediction": row["parent_prediction"],
                    "used_handle_ids": [],
                },
                sort_keys=True,
            )
            for row in rows
        )
        self.unique_records = tuple(
            SimpleNamespace(
                call_key_sha256=identity_sha256({"call": index}),
                checkpoint_hit=True,
                completion=completion,
                completion_sha256=quote_sha256(completion),
                messages_sha256=row["messages_sha256"],
                physical_call=False,
                request_journal_sha256=identity_sha256({"request": index}),
                response_journal_sha256=identity_sha256({"response": index}),
            )
            for index, (row, completion) in enumerate(
                zip(rows, self.logical_completions, strict=True)
            )
        )
        self.usage = SimpleNamespace(
            checkpoint_hits=EXPECTED_PROVIDER_CALLS,
            logical_calls=EXPECTED_PROVIDER_CALLS,
            physical_calls=0,
            unique_calls=EXPECTED_PROVIDER_CALLS,
        )

    def model_dump(self) -> dict[str, Any]:
        return {
            "logical_completions": list(self.logical_completions),
            "prompt_population": {},
            "provenance": {},
            "runtime_identity_sha256": identity_sha256({"runtime": "fake-v3"}),
            "unique_records": [vars(row) for row in self.unique_records],
            "usage": vars(self.usage),
        }


def test_v3_paths_protocol_and_required_source_digest_are_distinct() -> None:
    assert DEFAULT_CONSTRUCTION.name == "reduced-specialist-construction-v3.json"
    assert DEFAULT_OUTPUT.name == "reduced-specialist-answer-v3"
    assert FORMAT == "memory-condense-reduced-specialist-terra-answer-v3"
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["preflight"])


def test_preflight_rerenders_and_seals_ten_scoped_prompts(tmp_path: Path) -> None:
    artifact = _construction(tmp_path)
    source, rows = _read_construction(artifact.path, artifact.sha256)
    payload = _preflight_projection(
        source,
        rows,
        model=DEFAULT_MODEL,
        gateway_url="https://central-dev.zt:4000/v1",
        max_concurrency=4,
    )

    assert tuple(row["ordinal"] for row in rows) == EXPECTED_ORDINALS
    assert len({row["messages_sha256"] for row in rows}) == 10
    assert all(row["prompt_token_proxy"] <= MAX_CHAT_PROMPT_TOKENS for row in rows)
    assert payload["prompt_population"]["logical_prompt_count"] == 10
    assert payload["prompt_population"]["unique_prompt_count"] == 10
    assert payload["required_authorized_provider_calls"] == 10
    assert payload["scoped_completion_format"] == SCOPED_COMPLETION_FORMAT
    assert payload["gold_loaded"] is False
    assert payload["provider_calls"] == 0


def test_source_and_terminal_receipt_tamper_fail_closed(tmp_path: Path) -> None:
    artifact = _construction(tmp_path)
    with pytest.raises(
        ReducedSpecialistAnswerV3Error,
        match="construction digest changed",
    ):
        _read_construction(artifact.path, "0" * 64)

    row = _construction_question(EXPECTED_ORDINALS[0], 0)
    terminal = row["terminal_prompt"]
    assert type(terminal) is dict
    terminal["message_renderer_format"] = "tampered-renderer"
    body = dict(row)
    body.pop("question_receipt_sha256")
    row["question_receipt_sha256"] = identity_sha256(body)
    with pytest.raises(
        ReducedSpecialistAnswerV3Error,
        match="scoped prompt seal or hard budget changed",
    ):
        _prompt_plan_row(row, EXPECTED_ORDINALS[0])


@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_loaded_preflight_rejects_nonexact_root_schema(
    tmp_path: Path,
    mutation: str,
) -> None:
    preflight, _rows = _preflight_artifact(tmp_path)
    payload = json.loads(json.dumps(preflight.payload))
    if mutation == "extra":
        payload["unexpected_field"] = "unexpected"
    else:
        payload.pop("gateway_url")
    tampered = SealedArtifact(preflight.path, identity_sha256(payload), payload)

    with pytest.raises(
        ReducedSpecialistAnswerV3Error,
        match="sealed specialist v3 preflight changed",
    ):
        _validate_preflight(tampered)


def test_loaded_preflight_reruns_gold_firewall(tmp_path: Path) -> None:
    preflight, _rows = _preflight_artifact(tmp_path)
    payload = json.loads(json.dumps(preflight.payload))
    payload["reference_answer"] = "must never cross the runtime boundary"
    tampered = SealedArtifact(preflight.path, identity_sha256(payload), payload)

    with pytest.raises(MatchedEvalContractError, match="gold-bearing field"):
        _validate_preflight(tampered)


def test_loaded_preflight_recomputes_complete_envelope_maximum(
    tmp_path: Path,
) -> None:
    preflight, rows = _preflight_artifact(tmp_path)
    expected = max(
        row["prompt_token_proxy"] + OUTPUT_TOKEN_RESERVE for row in rows
    )
    assert preflight.payload["observed_max_complete_envelope_tokens"] == expected
    payload = json.loads(json.dumps(preflight.payload))
    payload["observed_max_complete_envelope_tokens"] = expected - 1
    tampered = SealedArtifact(preflight.path, identity_sha256(payload), payload)

    with pytest.raises(
        ReducedSpecialistAnswerV3Error,
        match="observed complete-envelope maximum changed",
    ):
        _validate_preflight(tampered)


def test_checkpoint_only_materialization_uses_minimal_judge_seam(
    tmp_path: Path,
) -> None:
    artifact = _construction(tmp_path)
    source, rows = _read_construction(artifact.path, artifact.sha256)
    preflight = SimpleNamespace(
        sha256=identity_sha256({"preflight": "fake-v3"}),
        payload={"construction_artifact_sha256": source.sha256},
    )
    payload = _materialization_projection(preflight, rows, _FakeBatch(rows))

    assert payload["format"] == FORMAT
    assert payload["gold_loaded"] is False
    assert payload["physical_provider_calls_during_materialization"] == 0
    assert payload["question_count"] == 10
    assert payload["changed_prediction_count"] == 0
    assert tuple(row["ordinal"] for row in payload["judge_rows"]) == (
        EXPECTED_ORDINALS
    )
    for plan, answer in zip(rows, payload["judge_rows"], strict=True):
        assert set(answer) == {
            "answer_row_sha256",
            "dated_question_sha256",
            "ordinal",
            "prediction",
            "prediction_sha256",
            "question_id",
            "question_sha256",
        }
        body = dict(answer)
        declared = body.pop("answer_row_sha256")
        assert declared == identity_sha256(body)
        assert answer["prediction"] == plan["parent_prediction"]
        assert answer["prediction_sha256"] == quote_sha256(answer["prediction"])
