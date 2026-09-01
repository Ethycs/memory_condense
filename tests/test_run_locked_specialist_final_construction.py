from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools import run_locked_specialist_final_construction as arm
from tools import run_reduced_specialist_retrieval_assay as specialist
from tools.matched_eval.artifacts import SealedArtifact, publish_sealed_json
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval import numeric_operand_specialist as numeric
from tools.matched_eval.specialist_scoped_completion import (
    PROMPT_FORMAT,
    render_specialist_scoped_prompt,
)


def _sha(label: str) -> str:
    return quote_sha256(label)


def _parent_source(ordinal: int) -> dict[str, object]:
    prediction = f"parent prediction {ordinal}"
    question_id = f"q{ordinal:03d}"
    judge = {
        "changed_from_parent": False,
        "dated_question_sha256": _sha(f"dated-{ordinal}"),
        "format": "memory-condense-typed-memory-final-arm-v1-judge-row-v1",
        "ordinal": ordinal,
        "parent_prediction_sha256": _sha(prediction),
        "prediction": prediction,
        "prediction_sha256": _sha(prediction),
        "prediction_source": "typed_final_invalid_keep_parent_v1",
        "question_id": question_id,
        "question_sha256": _sha(f"question-{ordinal}"),
        "route_id": "direct_extract",
        "source_row_sha256": _sha(f"source-{ordinal}"),
    }
    body = {
        "parent_judge_row": judge,
        "parent_judge_row_sha256": identity_sha256(judge),
        "prediction": prediction,
        "prediction_sha256": _sha(prediction),
        "replay_artifact_sha256": arm.EXPECTED_PARENT_REPLAY_SHA256,
        "run_artifact_sha256": arm.EXPECTED_PARENT_RUN_SHA256,
        "source_row_sha256": _sha(f"source-{ordinal}"),
    }
    return {**body, "receipt_sha256": identity_sha256(body)}


def _passthrough_row(ordinal: int) -> dict[str, object]:
    body = {
        "applicable_specialist_ids": [],
        "dated_question_sha256": _sha(f"dated-{ordinal}"),
        "methods": [],
        "mode": "parent_passthrough",
        "namespace_id": _sha(f"namespace-{ordinal % 10}"),
        "new_provider_calls": 0,
        "ordinal": ordinal,
        "parent_source": _parent_source(ordinal),
        "question_id": f"q{ordinal:03d}",
        "question_sha256": _sha(f"question-{ordinal}"),
        "retained_transformer_token_state_bytes": 0,
        "route": {"style": "direct"},
        "terminal_prompt": None,
    }
    return {**body, "question_receipt_sha256": identity_sha256(body)}


def _payload(*, one_specialist: bool = False) -> dict[str, object]:
    questions = [_passthrough_row(ordinal) for ordinal in arm.ORDINALS]
    terminal_tokens: list[int] = []
    if one_specialist:
        row = dict(questions[0])
        row.pop("question_receipt_sha256")
        advisory = {
            "candidate_handle_map": {"candidate-1": "H700001"},
            "mechanism_id": "numeric_operand_closure_v1",
            "numeric_operand_groups": [],
            "purpose": "apply the sealed specialist proof",
        }
        terminal = specialist._terminal_projection(  # noqa: SLF001
            provider_input={
                "dated_question": "[Question asked at 2023/01/01]\nWhat was total?",
                "parent_prediction": "parent prediction 0",
            },
            specialist_advisories=[advisory],
            fitted_prompt_receipt_sha256=_sha("fitted"),
            message_renderer_format=PROMPT_FORMAT,
            prompt_envelope_renderer=render_specialist_scoped_prompt,
        )
        row.update(
            {
                "applicable_specialist_ids": ["numeric_operand_closure_v1"],
                "methods": [{"mechanism_id": "numeric_operand_closure_v1"}],
                "mode": "specialist",
                "terminal_prompt": terminal,
            }
        )
        row["question_receipt_sha256"] = identity_sha256(row)
        questions[0] = row
        terminal_tokens.append(terminal["full_chat_plus_output_tokens"])
    bindings = {
        "parent_composition_artifact_sha256": (
            arm.EXPECTED_PARENT_COMPOSITION_SHA256
        ),
        "parent_full_store_input_artifact_sha256": (
            arm.EXPECTED_PARENT_CLOSURE_SHA256
        ),
        "parent_replay_artifact_sha256": arm.EXPECTED_PARENT_REPLAY_SHA256,
        "parent_run_artifact_sha256": arm.EXPECTED_PARENT_RUN_SHA256,
    }
    receipts = [
        {"database_read_passes": 1, "namespace_id": _sha(f"namespace-{value}")}
        for value in range(10)
    ]
    specialist_count = int(one_specialist)
    payload: dict[str, object] = {
        "bindings": bindings,
        "construction_is_posthoc_outcome_conditioned": False,
        "format": arm.CONSTRUCTION_FORMAT,
        "gold_loaded": False,
        "hard_complete_chat_token_cap": 8_000,
        "max_terminal_complete_envelope_tokens": max(terminal_tokens, default=0),
        "new_provider_calls": 0,
        "ordinals": list(arm.ORDINALS),
        "parent_passthrough_count": 100 - specialist_count,
        "question_count": 100,
        "questions": questions,
        "resident_index_lifecycle": {
            "database_read_passes_per_used_namespace": 1,
            "maximum_simultaneous_namespace_indexes": 1,
            "receipts": receipts,
            "total_database_read_passes": 10,
            "unique_namespace_count": 10,
        },
        "retained_transformer_token_state_bytes": 0,
        "selection_and_routing_frozen_before_target_plan_load": True,
        "specialist_provider_prompt_count": specialist_count,
        "target_labels_loaded": False,
        "target_plan_loaded": False,
    }
    payload["construction_identity_sha256"] = identity_sha256(payload)
    return payload


@pytest.mark.parametrize("one_specialist", [False, True])
def test_validator_accepts_exact_passthrough_and_specialist_modes(
    one_specialist: bool,
) -> None:
    artifact = SealedArtifact(Path("unused.json"), _sha("artifact"), _payload(
        one_specialist=one_specialist
    ))

    rows = arm.validate_construction(artifact)

    assert len(rows) == 100
    assert sum(row["mode"] == "specialist" for row in rows) == int(one_specialist)
    assert all(row["new_provider_calls"] == 0 for row in rows)


def test_passthrough_cannot_smuggle_a_provider_prompt() -> None:
    payload = _payload()
    row = payload["questions"][3]
    row["terminal_prompt"] = {"messages": []}
    body = dict(row)
    body.pop("question_receipt_sha256")
    row["question_receipt_sha256"] = identity_sha256(body)
    unsigned = dict(payload)
    unsigned.pop("construction_identity_sha256")
    payload["construction_identity_sha256"] = identity_sha256(unsigned)

    with pytest.raises(
        arm.LockedSpecialistFinalConstructionError,
        match="passthrough row exposed a provider prompt",
    ):
        arm.validate_construction(
            SealedArtifact(Path("unused.json"), _sha("artifact"), payload)
        )


def test_loader_binds_file_sha_and_accepts_root_path(tmp_path: Path) -> None:
    artifact, _created = publish_sealed_json(
        tmp_path / arm.CONSTRUCTION_NAME, _payload(one_specialist=True)
    )

    loaded, rows = arm.load_verified_construction(
        tmp_path, expected_sha256=artifact.sha256
    )

    assert loaded.sha256 == artifact.sha256
    assert len(rows) == 100


def test_reduced_composer_exposes_backward_compatible_parent_override() -> None:
    parameter = inspect.signature(specialist._composed_question).parameters[  # noqa: SLF001
        "parent_prediction_override"
    ]

    assert parameter.default is None
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_numeric_event_pair_filter_can_leave_a_clean_empty_population() -> None:
    events = numeric._events(  # noqa: SLF001
        (numeric._ActionHit("purchase", 0, 3, "got"),),  # noqa: SLF001
        (numeric._EntityHit("feed", 100, 104),),  # noqa: SLF001
        (SimpleNamespace(start=110, end=112),),
    )

    assert events == ()
