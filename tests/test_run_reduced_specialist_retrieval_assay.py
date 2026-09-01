from __future__ import annotations

import hashlib
from pathlib import Path

from tools.matched_eval.artifacts import SealedArtifact
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.numeric_operand_specialist import (
    MECHANISM_ID as NUMERIC_MECHANISM_ID,
)
from tools.matched_eval.profile_preference_specialist import (
    MECHANISM_ID as PROFILE_MECHANISM_ID,
)
from tools.matched_eval.temporal_insufficiency_specialist import (
    MECHANISM_ID as TEMPORAL_MECHANISM_ID,
)
from tools.run_reduced_specialist_retrieval_assay import (
    CONSTRUCTION_FORMAT,
    TARGET_ORDINALS,
    applicable_specialist_ids,
    build_target_audit,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _dated(body: str) -> str:
    return f"[Question asked at 2023/05/30 (Tue) 14:24]\n{body}"


def test_question_only_router_partitions_the_reduced_failure_styles() -> None:
    assert applicable_specialist_ids(
        _dated("What is the total weight of the new feed I purchased?")
    ) == (NUMERIC_MECHANISM_ID,)
    assert applicable_specialist_ids(
        _dated("Can you recommend a show or movie for me to watch tonight?")
    ) == (PROFILE_MECHANISM_ID,)
    assert applicable_specialist_ids(
        _dated("What gardening-related activity did I do two weeks ago?")
    ) == (TEMPORAL_MECHANISM_ID,)


def test_compound_numeric_question_adds_the_insufficiency_specialist() -> None:
    assert applicable_specialist_ids(
        _dated("How many plants did I initially plant for tomatoes and chili peppers?")
    ) == (NUMERIC_MECHANISM_ID, TEMPORAL_MECHANISM_ID)


def test_latest_state_contract_routes_to_temporal_specialist() -> None:
    assert applicable_specialist_ids(
        "[Question asked at 2023/05/30 (Tue) 23:20]\n"
        "How much did I spend on a designer handbag?"
    ) == (TEMPORAL_MECHANISM_ID,)


def _construction() -> SealedArtifact:
    questions = []
    for ordinal in TARGET_ORDINALS:
        question_id = f"q{ordinal}"
        method_body = {
            "local_bindings": [
                {"source_id": f"{question_id}::answer_{ordinal}"}
            ],
            "mechanism_id": NUMERIC_MECHANISM_ID,
            "source_ids": [f"{question_id}::answer_{ordinal}"],
            "source_ids_by_handle": {
                "H700001": [f"{question_id}::answer_{ordinal}"]
            },
        }
        questions.append(
            {
                "fitted_typed_prompt": {"allowed_handle_ids": ["H700001"]},
                "methods": [method_body],
                "ordinal": ordinal,
                "question_id": question_id,
            }
        )
    payload = {
        "construction_is_posthoc_outcome_conditioned": True,
        "format": CONSTRUCTION_FORMAT,
        "gold_loaded": False,
        "new_provider_calls": 0,
        "ordinals": list(TARGET_ORDINALS),
        "question_count": len(TARGET_ORDINALS),
        "questions": questions,
        "retained_transformer_token_state_bytes": 0,
        "target_labels_loaded": False,
        "target_plan_loaded": False,
    }
    payload["construction_identity_sha256"] = identity_sha256(payload)
    return SealedArtifact(Path("construction.json"), _sha("construction"), payload)


def _target_plan() -> dict[str, object]:
    return {
        "desired_targets": [
            {
                "ordinal": ordinal,
                "question_id": f"q{ordinal}",
                "target_id": f"answer_{ordinal}",
                "target_kind": "source_id",
            }
            for ordinal in TARGET_ORDINALS
        ],
        "plan_sha256": _sha("target-plan"),
    }


def test_posthoc_audit_counts_history_qualified_local_sources() -> None:
    construction = _construction()
    audit = build_target_audit(
        construction,
        _target_plan(),
        target_plan_file_sha256=_sha("target-plan-file"),
    )

    assert audit["construction_verified_before_target_plan_load"] is True
    assert audit["runtime_use_forbidden"] is True
    assert audit["union_source_set_complete_questions"] == len(TARGET_ORDINALS)
    assert audit["union_source_target_hits"] == len(TARGET_ORDINALS)
    assert audit["method_summary"][NUMERIC_MECHANISM_ID] == {
        "question_count": len(TARGET_ORDINALS),
        "source_set_complete_questions": len(TARGET_ORDINALS),
        "source_target_count": len(TARGET_ORDINALS),
        "source_target_hits": len(TARGET_ORDINALS),
        "terminal_source_set_complete_questions": len(TARGET_ORDINALS),
        "terminal_source_target_hits": len(TARGET_ORDINALS),
    }
