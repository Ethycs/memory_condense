from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools import run_reduced_specialist_retrieval_assay as base
from tools import run_reduced_specialist_retrieval_assay_v3 as v3
from tools.matched_eval.artifacts import SealedArtifact
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.specialist_scoped_completion import (
    PROMPT_FORMAT,
    render_specialist_scoped_prompt,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _construction() -> SealedArtifact:
    rows = []
    for ordinal in base.TARGET_ORDINALS:
        question_id = f"q{ordinal}"
        rows.append(
            {
                "fitted_typed_prompt": {"allowed_handle_ids": ["H700001"]},
                "methods": [
                    {
                        "mechanism_id": "numeric_operand_closure_specialist_v1",
                        "source_ids": [f"{question_id}::answer_{ordinal}"],
                        "source_ids_by_handle": {
                            "H700001": [f"{question_id}::answer_{ordinal}"]
                        },
                    }
                ],
                "ordinal": ordinal,
                "question_id": question_id,
            }
        )
    payload = {
        "format": v3.CONSTRUCTION_FORMAT,
        "gold_loaded": False,
        "new_provider_calls": 0,
        "ordinals": list(base.TARGET_ORDINALS),
        "question_count": len(base.TARGET_ORDINALS),
        "questions": rows,
        "retained_transformer_token_state_bytes": 0,
        "target_labels_loaded": False,
        "target_plan_loaded": False,
    }
    payload["construction_identity_sha256"] = identity_sha256(payload)
    return SealedArtifact(Path("construction-v3.json"), _sha("construction"), payload)


def _target_plan() -> dict[str, object]:
    return {
        "desired_targets": [
            {
                "ordinal": ordinal,
                "question_id": f"q{ordinal}",
                "target_id": f"answer_{ordinal}",
                "target_kind": "source_id",
            }
            for ordinal in base.TARGET_ORDINALS
        ],
        "plan_sha256": _sha("target-plan"),
    }


def test_v3_protocol_and_default_paths_are_distinct_from_v2() -> None:
    assert v3.CONSTRUCTION_FORMAT != base.CONSTRUCTION_FORMAT
    assert v3.AUDIT_FORMAT != base.AUDIT_FORMAT
    assert v3.CONSTRUCTION_NAME != base.CONSTRUCTION_NAME
    assert v3.AUDIT_NAME != base.AUDIT_NAME
    parsed = v3.build_parser().parse_args(["construct"])
    assert parsed.output_root == v3.DEFAULT_OUTPUT_ROOT


def test_v3_audit_uses_explicit_protocol_without_weakening_v2_validation() -> None:
    construction = _construction()
    with pytest.raises(base.ReducedSpecialistAssayError):
        base._validate_construction(construction)  # noqa: SLF001

    audit = base.build_target_audit(
        construction,
        _target_plan(),
        target_plan_file_sha256=_sha("target-plan-file"),
        construction_format=v3.CONSTRUCTION_FORMAT,
        audit_format=v3.AUDIT_FORMAT,
    )

    assert audit["format"] == v3.AUDIT_FORMAT
    assert audit["bindings"]["construction_artifact_sha256"] == construction.sha256
    assert audit["union_source_set_complete_questions"] == len(
        base.TARGET_ORDINALS
    )


def test_v3_terminal_binds_the_specialist_prompt_envelope() -> None:
    provider_input = {"dated_question": "What happened?"}
    advisories = [
        {
            "format": base.SPECIALIST_ADVISORY_FORMAT,
            "handle_ids": ["H700001"],
            "mechanism_id": "synthetic_specialist",
        }
    ]

    terminal = base._terminal_projection(  # noqa: SLF001
        provider_input=provider_input,
        specialist_advisories=advisories,
        fitted_prompt_receipt_sha256=_sha("fitted"),
        message_renderer_format=PROMPT_FORMAT,
        prompt_envelope_renderer=render_specialist_scoped_prompt,
    )
    envelope = render_specialist_scoped_prompt(
        {**provider_input, "specialist_advisories": advisories}
    )

    assert terminal["message_renderer_format"] == PROMPT_FORMAT
    assert (
        terminal["specialist_prompt_envelope_receipt_sha256"]
        == envelope.receipt_sha256
    )
    assert terminal["messages_sha256"] == identity_sha256(list(envelope.messages))
