from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools.matched_eval.artifacts import SealedArtifact, publish_sealed_json
from tools.matched_eval.contracts import MatchedEvalContractError, identity_sha256
from tools.matched_eval.specialist_scoped_completion import (
    HARD_COMPLETE_CHAT_TOKEN_CAP,
    MAX_CHAT_PROMPT_TOKENS,
    OUTPUT_TOKEN_RESERVE,
    PROMPT_FORMAT,
    VALIDATION_CONTRACT_FORMAT,
    render_specialist_scoped_prompt,
)
from tools.matched_eval.typed_memory_final_arm import judge_row_projection
from tools import run_locked_specialist_final_answer as answer


SPECIALIST_ORDINALS = (2, 77)


def _sha(label: str) -> str:
    return quote_sha256(label)


def _semantic_row(
    label: str,
    *,
    numeric: float,
    unit: str,
) -> dict[str, Any]:
    return {
        "action_concepts": ["buy"],
        "completed_action_concepts": ["buy"],
        "date": "2026-08-01",
        "entity_terms": ["user", "feed", label],
        "group_terms": [],
        "item_receipt_sha256": _sha(f"item {label}"),
        "numeric_value": numeric,
        "relation_terms": [],
        "semantic_unit_sha256": _sha(f"semantic {label}"),
        "status": "completed",
        "summary_terms": ["bought", label, str(numeric), unit],
        "supported_slot_ids": [],
        "unit": unit,
    }


def _parent_source(
    ordinal: int,
    *,
    question_id: str,
    question_sha256: str,
    dated_question_sha256: str,
) -> dict[str, Any]:
    prediction = f"Parent fallback {ordinal}"
    parent_body = {
        "changed_from_parent": ordinal % 7 == 0,
        "dated_question_sha256": dated_question_sha256,
        "format": "synthetic-parent-result-row-v1",
        "ordinal": ordinal,
        "parent_prediction_sha256": _sha(f"older parent {ordinal}"),
        "prediction": prediction,
        "prediction_sha256": _sha(prediction),
        "prediction_source": "sealed_typed_final_parent_v1",
        "question_id": question_id,
        "question_sha256": question_sha256,
        "route_id": "numeric_aggregation",
    }
    parent_result = {
        **parent_body,
        "source_row_sha256": identity_sha256(parent_body),
    }
    parent_judge = judge_row_projection(parent_result)
    body = {
        "parent_judge_row": parent_judge,
        "parent_judge_row_sha256": identity_sha256(parent_judge),
        "prediction": prediction,
        "prediction_sha256": _sha(prediction),
        "replay_artifact_sha256": _sha("sealed parent replay"),
        "run_artifact_sha256": _sha("sealed parent run"),
        "source_row_sha256": parent_result["source_row_sha256"],
    }
    return {**body, "receipt_sha256": identity_sha256(body)}


def _specialist_fields(
    ordinal: int,
    *,
    dated_question: str,
    parent_prediction: str,
) -> dict[str, Any]:
    handles = (f"H{700000 + ordinal * 2}", f"H{700001 + ordinal * 2}")
    groups = (f"G{700000 + ordinal * 2}", f"G{700001 + ordinal * 2}")
    candidates = (_sha(f"candidate a {ordinal}"), _sha(f"candidate b {ordinal}"))
    semantics = (
        _semantic_row(f"layer-{ordinal}", numeric=50.0, unit="lb"),
        _semantic_row(f"grain-{ordinal}", numeric=20.0, unit="lb"),
    )
    advisory = {
        "candidate_handle_map": dict(zip(candidates, handles, strict=True)),
        "mechanism_id": "numeric_specialist_v3",
        "operand_groups": [
            {
                "action_class": "buy",
                "candidate_ids": [candidate],
                "entity_key": "feed",
                "operand_values": [value],
                "operation_mode": "sum",
                "source_group_handles": [group],
                "value_basis": "explicit_numeric_mention",
            }
            for candidate, value, group in zip(
                candidates, (50.0, 20.0), groups, strict=True
            )
        ],
        "purpose": "group all compatible numeric operands",
    }
    validation = {
        "by_handle": {
            handle: {
                "semantic_rows": [semantic],
                "status_values": ["completed"],
                "usable_item_receipt_sha256s": [
                    semantic["item_receipt_sha256"]
                ],
            }
            for handle, semantic in zip(handles, semantics, strict=True)
        },
        "format": VALIDATION_CONTRACT_FORMAT,
        "include_proposed": False,
        "question_terms": ["how", "much", "feed", "bought"],
    }
    fitted_input = {
        "dated_question": dated_question,
        "protected_parent_fallback": {
            "label": "fallback_not_evidence",
            "prediction": parent_prediction,
            "prediction_sha256": _sha(parent_prediction),
        },
        "response_schema": {
            "decision": "keep_parent|replace",
            "prediction": "nonempty exact text",
            "used_handle_ids": list(handles),
        },
        "typed_evidence": {
            "handles": [
                {"group_handle": group, "handle_id": handle}
                for handle, group in zip(handles, groups, strict=True)
            ],
            "items": [],
        },
    }
    provider_input = {**fitted_input, "specialist_advisories": [advisory]}
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
    return {
        "additive_composition": {"format": "synthetic-additive-v1"},
        "fitted_typed_prompt": {
            "allowed_handle_ids": list(handles),
            "provider_input": fitted_input,
            "receipt_sha256": fitted_receipt,
            "validation_contract": validation,
        },
        "lane_budget_policy": {"format": "synthetic-lane-budget-v1"},
        "methods": ["numeric_specialist_v3"],
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


def _construction_question(ordinal: int) -> dict[str, Any]:
    question_id = f"q{ordinal:03d}"
    question = f"How much feed was bought in memory {ordinal}?"
    dated_question = f"[Question asked at 2026/08/28] {question}"
    question_sha = _sha(question)
    dated_sha = _sha(dated_question)
    parent = _parent_source(
        ordinal,
        question_id=question_id,
        question_sha256=question_sha,
        dated_question_sha256=dated_sha,
    )
    specialist = ordinal in SPECIALIST_ORDINALS
    body: dict[str, Any] = {
        "applicable_specialist_ids": ["numeric_specialist_v3"] if specialist else [],
        "dated_question_sha256": dated_sha,
        "methods": [],
        "mode": answer.SPECIALIST_MODE if specialist else answer.PARENT_PASSTHROUGH_MODE,
        "namespace_id": f"namespace-{ordinal // 10}",
        "new_provider_calls": 0,
        "ordinal": ordinal,
        "parent_source": parent,
        "question_id": question_id,
        "question_sha256": question_sha,
        "retained_transformer_token_state_bytes": 0,
        "route": {"style": "numeric_aggregation"},
        "terminal_prompt": None,
    }
    if specialist:
        body.update(
            _specialist_fields(
                ordinal,
                dated_question=dated_question,
                parent_prediction=parent["prediction"],
            )
        )
    return {**body, "question_receipt_sha256": identity_sha256(body)}


def _construction(
    tmp_path: Path,
) -> tuple[SealedArtifact, tuple[dict[str, Any], ...], answer.ConstructionLoader]:
    rows = tuple(_construction_question(ordinal) for ordinal in range(100))
    payload = {
        "format": "synthetic-locked-specialist-construction-v1",
        "gold_loaded": False,
        "question_count": 100,
        "questions_sha256": identity_sha256(list(rows)),
    }
    artifact, created = publish_sealed_json(
        tmp_path / "locked-specialist-final-construction-v1.json", payload
    )
    assert created

    def loader(
        path: Path,
        *,
        expected_sha256: str,
    ) -> tuple[SealedArtifact, Sequence[Mapping[str, Any]]]:
        assert path == artifact.path
        assert expected_sha256 == artifact.sha256
        return artifact, tuple(json.loads(json.dumps(rows, sort_keys=True)))

    return artifact, rows, loader


def _plans(
    tmp_path: Path,
) -> tuple[SealedArtifact, tuple[dict[str, Any], ...]]:
    artifact, _rows, loader = _construction(tmp_path)
    return answer.load_answer_plans(
        artifact.path,
        artifact.sha256,
        construction_loader=loader,
    )


def _preflight(
    tmp_path: Path,
) -> tuple[SealedArtifact, SealedArtifact, tuple[dict[str, Any], ...]]:
    construction, plans = _plans(tmp_path)
    payload = answer._preflight_projection(
        construction,
        plans,
        model=answer.DEFAULT_MODEL,
        gateway_url="https://central-dev.zt:4000/v1",
        max_concurrency=4,
    )
    preflight = SealedArtifact(
        Path("synthetic-specialist-preflight-v1.json"),
        identity_sha256(payload),
        payload,
    )
    return construction, preflight, plans


class _FakeBatch:
    def __init__(
        self,
        specialist_rows: Sequence[Mapping[str, Any]],
        completions: Sequence[str] | None = None,
    ) -> None:
        rows = tuple(specialist_rows)
        logical = tuple(
            completions
            or (
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
        )
        self.logical_completions = logical
        self.unique_records = tuple(
            SimpleNamespace(
                call_key_sha256=identity_sha256({"call": index}),
                checkpoint_hit=True,
                completion=completion,
                completion_sha256=_sha(completion),
                messages_sha256=row["messages_sha256"],
                physical_call=False,
                request_journal_sha256=identity_sha256({"request": index}),
                response_journal_sha256=identity_sha256({"response": index}),
            )
            for index, (row, completion) in enumerate(
                zip(rows, logical, strict=True)
            )
        )
        count = len(rows)
        self.usage = SimpleNamespace(
            checkpoint_hits=count,
            logical_calls=count,
            physical_calls=0,
            unique_calls=count,
        )

    def model_dump(self) -> dict[str, Any]:
        return {
            "logical_completions": list(self.logical_completions),
            "prompt_population": {},
            "provenance": {},
            "runtime_identity_sha256": identity_sha256({"runtime": "fake-v1"}),
            "unique_records": [vars(row) for row in self.unique_records],
            "usage": vars(self.usage),
        }


def test_paths_and_public_contract_are_distinct() -> None:
    assert answer.DEFAULT_CONSTRUCTION.name == "locked-specialist-final-construction-v1.json"
    assert answer.DEFAULT_OUTPUT.name == "locked-specialist-final-answer-v1"
    assert answer.RUN_NAME == "locked-specialist-final-answer-v1.json"
    assert answer.REPLAY_NAME == "locked-specialist-final-answer-replay-v1.json"
    assert answer.FORMAT == "memory-condense-locked-specialist-final-terra-answer-v1"
    parser = answer.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["preflight"])


def test_preflight_seals_only_dynamic_unique_specialist_prompts(
    tmp_path: Path,
) -> None:
    construction, plans = _plans(tmp_path)
    payload = answer._preflight_projection(
        construction,
        plans,
        model=answer.DEFAULT_MODEL,
        gateway_url="https://central-dev.zt:4000/v1",
        max_concurrency=4,
    )

    assert len(plans) == 100
    assert payload["question_count"] == 100
    assert payload["specialist_question_count"] == 2
    assert payload["parent_passthrough_count"] == 98
    assert payload["required_authorized_provider_calls"] == 2
    assert tuple(row["ordinal"] for row in payload["physical_prompt_rows"]) == (
        SPECIALIST_ORDINALS
    )
    assert len(payload["parent_passthrough_rows"]) == 98
    assert payload["prompt_population"]["logical_prompt_count"] == 2
    assert payload["prompt_population"]["unique_prompt_count"] == 2
    assert payload["provider_calls"] == 0
    assert payload["gold_loaded"] is False
    assert payload["retained_transformer_token_state_bytes"] == 0
    assert payload["observed_max_complete_envelope_tokens"] <= 8000
    assert all(
        row["prompt_token_proxy"] <= MAX_CHAT_PROMPT_TOKENS
        for row in payload["physical_prompt_rows"]
    )


def test_construction_digest_and_parent_binding_fail_closed(tmp_path: Path) -> None:
    artifact, rows, loader = _construction(tmp_path)
    with pytest.raises(MatchedEvalContractError, match="expected specialist construction"):
        answer.load_answer_plans(
            artifact.path,
            "not-a-sha",
            construction_loader=loader,
        )

    tampered = json.loads(json.dumps(rows))
    raw = tampered[0]
    raw["parent_source"]["prediction"] = "forged parent"
    parent_body = dict(raw["parent_source"])
    parent_body.pop("receipt_sha256")
    raw["parent_source"]["receipt_sha256"] = identity_sha256(parent_body)
    question_body = dict(raw)
    question_body.pop("question_receipt_sha256")
    raw["question_receipt_sha256"] = identity_sha256(question_body)

    def tampered_loader(path: Path, *, expected_sha256: str):
        assert path == artifact.path and expected_sha256 == artifact.sha256
        return artifact, tuple(tampered)

    with pytest.raises(
        answer.LockedSpecialistFinalAnswerError,
        match="parent source seal changed",
    ):
        answer.load_answer_plans(
            artifact.path,
            artifact.sha256,
            construction_loader=tampered_loader,
        )


def test_loaded_preflight_rejects_gold_and_recomputes_complete_max(
    tmp_path: Path,
) -> None:
    _construction_artifact, preflight, plans = _preflight(tmp_path)
    prompts, loaded = answer._validate_preflight(preflight)
    assert len(prompts) == 2
    assert loaded == plans

    gold = json.loads(json.dumps(preflight.payload))
    gold["reference_answer"] = "must not enter the answer stage"
    with pytest.raises(MatchedEvalContractError, match="gold-bearing field"):
        answer._validate_preflight(
            SealedArtifact(preflight.path, identity_sha256(gold), gold)
        )

    wrong_max = json.loads(json.dumps(preflight.payload))
    wrong_max["observed_max_complete_envelope_tokens"] -= 1
    with pytest.raises(
        answer.LockedSpecialistFinalAnswerError,
        match="prompt population changed",
    ):
        answer._validate_preflight(
            SealedArtifact(preflight.path, identity_sha256(wrong_max), wrong_max)
        )


def test_materialization_calls_only_specialists_and_emits_typed_judge_seam(
    tmp_path: Path,
) -> None:
    _construction_artifact, preflight, plans = _preflight(tmp_path)
    specialist = tuple(row for row in plans if row["mode"] == answer.SPECIALIST_MODE)
    completions = (
        json.dumps(
            {
                "decision": "replace",
                "prediction": "70 lb",
                "used_handle_ids": specialist[0]["allowed_handle_ids"],
            },
            separators=(",", ":"),
        ),
        "not valid JSON",
    )
    payload = answer._materialization_projection(
        preflight,
        plans,
        _FakeBatch(specialist, completions),
    )

    assert payload["question_count"] == 100
    assert payload["required_authorized_provider_calls"] == 2
    assert payload["physical_provider_calls_during_materialization"] == 0
    assert payload["parent_passthrough_count"] == 98
    assert payload["specialist_question_count"] == 2
    assert payload["validated_replacement_count"] == 1
    assert payload["invalid_completion_parent_fallback_count"] == 1
    assert payload["changed_prediction_count"] == 1
    assert len(payload["questions"]) == len(payload["judge_rows"]) == 100

    for row, projected in zip(payload["questions"], payload["judge_rows"], strict=True):
        unsigned = dict(row)
        declared = unsigned.pop("source_row_sha256")
        assert declared == identity_sha256(unsigned)
        assert projected == judge_row_projection(row)
        assert row["retained_transformer_token_state_bytes"] == 0

    replacement = payload["questions"][SPECIALIST_ORDINALS[0]]
    assert replacement["prediction"] == "70 lb"
    assert replacement["changed_from_parent"] is True
    assert replacement["solver_valid"] is True
    assert replacement["prediction_source"].endswith("validated_replacement_v1")

    invalid = payload["questions"][SPECIALIST_ORDINALS[1]]
    assert invalid["prediction"] == plans[SPECIALIST_ORDINALS[1]]["parent_prediction"]
    assert invalid["changed_from_parent"] is False
    assert invalid["solver_valid"] is False

    passthrough = payload["questions"][0]
    assert passthrough["prediction"] == plans[0]["parent_prediction"]
    assert passthrough["prediction_sha256"] == plans[0]["parent_prediction_sha256"]
    assert passthrough["changed_from_parent"] is False
    assert passthrough["call_key_sha256"] is None
    assert passthrough["request_journal_sha256"] is None
    assert passthrough["answer_mode"] == answer.PARENT_PASSTHROUGH_MODE


def test_replay_publishes_byte_identical_payload_from_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construction, plans = _plans(tmp_path)
    preflight_payload = answer._preflight_projection(
        construction,
        plans,
        model=answer.DEFAULT_MODEL,
        gateway_url="https://central-dev.zt:4000/v1",
        max_concurrency=4,
    )
    preflight, created = publish_sealed_json(
        tmp_path / answer.PREFLIGHT_NAME,
        preflight_payload,
    )
    assert created
    specialist = tuple(row for row in plans if row["mode"] == answer.SPECIALIST_MODE)
    batch = _FakeBatch(specialist)
    run_payload = answer._materialization_projection(preflight, plans, batch)
    run, created = publish_sealed_json(tmp_path / answer.RUN_NAME, run_payload)
    assert created

    monkeypatch.setattr(
        answer,
        "load_answer_plans",
        lambda path, expected_sha256: (construction, plans),
    )
    monkeypatch.setattr(answer, "_checkpoint_batch", lambda *args, **kwargs: batch)
    args = SimpleNamespace(
        construction=construction.path,
        expected_construction_sha256=construction.sha256,
        expected_preflight_sha256=preflight.sha256,
        expected_run_sha256=run.sha256,
        gateway_url="https://central-dev.zt:4000/v1",
        max_concurrency=4,
        model=answer.DEFAULT_MODEL,
        output_root=tmp_path,
    )
    result = answer.run_replay(args)

    assert result["byte_identical"] is True
    assert result["physical_provider_calls"] == 0
    assert result["run_sha256"] == result["replay_sha256"] == run.sha256
