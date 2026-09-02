from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools.matched_eval.artifacts import SealedArtifact, read_sealed_json
from tools.matched_eval.contracts import canonical_json_bytes, identity_sha256
from tools.run_reduced_missing4_v4_construction import (
    AUDIT_NAME,
    CONSTRUCTION_NAME,
    DEFAULT_OUTPUT_ROOT,
    HARD_COMPLETE_CHAT_TOKEN_CAP,
    ReducedMissing4V4Error,
    _target_sources,
    _terminal_projection,
    validate_construction,
)
from tools.run_reduced_specialist_answer_v2 import (
    ReducedSpecialistAnswerV2Error,
    _prompt_plan_row,
)


EXPECTED_CONSTRUCTION_SHA256 = (
    "4328f9334b858909a6511ee7114dd5d3dabf37c45393cf543ea05625fdb4cb43"
)
EXPECTED_AUDIT_SHA256 = (
    "3ddd130db2970c7f576f912423dc5e1fed4d25aa8d88842da1c392f8aef3e96a"
)


def _sha(label: str) -> str:
    return quote_sha256(label)


def _generic_fitted() -> SimpleNamespace:
    dated_question = "[Question asked at 2026/08/28] What did I remember?"
    parent = "The protected fallback."
    story = {
        "group_links": [],
        "incompatible_group_pairs": [],
        "link_overlays": [],
        "link_token_cap": 256,
        "link_token_proxy": 0,
        "omitted_conflict_policy": "clear",
        "policy": "prefer one coherent group",
    }
    provider_input = {
        "dated_question": dated_question,
        "protected_parent_fallback": {
            "label": "fallback_not_evidence",
            "prediction": parent,
            "prediction_sha256": _sha(parent),
        },
        "response_schema": {
            "decision": "keep_parent|replace",
            "prediction": "nonempty exact text",
            "used_handle_ids": ["H500001"],
        },
        "story_coherence": story,
        "typed_evidence": {
            "handles": [
                {"group_handle": "G500001", "handle_id": "H500001"}
            ],
            "items": [],
        },
    }
    return SimpleNamespace(
        allowed_handle_ids=("H500001",),
        preservation_requirements={},
        provider_input=provider_input,
        receipt_sha256=_sha("fitted-v4-row"),
        story_coherence=story,
        validation_contract={},
    )


def _generic_question_row() -> dict[str, object]:
    fitted = _generic_fitted()
    advisory = {
        "generic_frontier_closed": False,
        "mechanism_id": "test_bounded_mechanism_v1",
        "proof_kind": "selected_scope_witness",
        "purpose": "exercise the common answer-v2 row contract",
    }
    terminal = _terminal_projection(
        fitted=fitted,
        specialist_advisories=(advisory,),
    )
    dated_question = fitted.provider_input["dated_question"]
    body: dict[str, object] = {
        "dated_question_sha256": _sha(dated_question),
        "fitted_typed_prompt": {
            "allowed_handle_ids": list(fitted.allowed_handle_ids),
            "preservation_requirements": dict(fitted.preservation_requirements),
            "provider_input": dict(fitted.provider_input),
            "receipt_sha256": fitted.receipt_sha256,
            "story_coherence": dict(fitted.story_coherence),
            "validation_contract": dict(fitted.validation_contract),
        },
        "ordinal": 42,
        "question_id": "synthetic-q42",
        "question_sha256": _sha("What did I remember?"),
        "terminal_prompt": terminal,
    }
    return {**body, "question_receipt_sha256": identity_sha256(body)}


def _published(name: str) -> Path:
    path = DEFAULT_OUTPUT_ROOT / name
    if not path.exists():
        pytest.skip("provider-free v4 runtime artifact is not present")
    return path


def test_all_rows_use_the_generic_answer_v2_prompt_contract() -> None:
    row = _generic_question_row()

    plan = _prompt_plan_row(row, 42)

    assert plan["allowed_handle_ids"] == ["H500001"]
    assert plan["messages_sha256"] == row["terminal_prompt"]["messages_sha256"]
    assert plan["prompt_token_proxy"] + 768 <= HARD_COMPLETE_CHAT_TOKEN_CAP
    assert plan["terminal_prompt_receipt_sha256"] == row["terminal_prompt"][
        "terminal_prompt_receipt_sha256"
    ]


def test_generic_answer_v2_rejects_an_unrecognized_prompt_version() -> None:
    row = _generic_question_row()
    row["terminal_prompt"]["messages_sha256"] = "0" * 64
    unsigned = dict(row)
    unsigned.pop("question_receipt_sha256")
    row["question_receipt_sha256"] = identity_sha256(unsigned)

    with pytest.raises(
        ReducedSpecialistAnswerV2Error,
        match="does not bind one supported renderer",
    ):
        _prompt_plan_row(row, 42)


def test_generic_answer_v2_rejects_non_integer_sealed_token_count() -> None:
    row = _generic_question_row()
    token_count = row["terminal_prompt"]["prompt_token_proxy"]
    row["terminal_prompt"]["prompt_token_proxy"] = float(token_count)
    unsigned = dict(row)
    unsigned.pop("question_receipt_sha256")
    row["question_receipt_sha256"] = identity_sha256(unsigned)

    with pytest.raises(
        ReducedSpecialistAnswerV2Error,
        match="token proxy changed type",
    ):
        _prompt_plan_row(row, 42)


def test_target_sources_freezes_the_six_source_target_population() -> None:
    rows = [
        {"ordinal": 42, "target_kind": "source_id", "target_id": "q42-a"},
        {"ordinal": 42, "target_kind": "source_id", "target_id": "q42-b"},
        {"ordinal": 65, "target_kind": "source_id", "target_id": "q65-a"},
        {"ordinal": 65, "target_kind": "source_id", "target_id": "q65-b"},
        {"ordinal": 74, "target_kind": "source_id", "target_id": "q74-a"},
        {"ordinal": 79, "target_kind": "source_id", "target_id": "q79-a"},
        {"ordinal": 42, "target_kind": "coverage", "target_id": "ignored"},
    ]

    assert _target_sources({"desired_targets": rows}) == {
        42: ("q42-a", "q42-b"),
        65: ("q65-a", "q65-b"),
        74: ("q74-a",),
        79: ("q79-a",),
    }


def test_published_v4_construction_keeps_all_four_scoped_boundaries() -> None:
    artifact = read_sealed_json(_published(CONSTRUCTION_NAME))
    assert artifact.sha256 == EXPECTED_CONSTRUCTION_SHA256

    rows = validate_construction(artifact)
    by_ordinal = {row["ordinal"]: row for row in rows}

    q42 = by_ordinal[42]
    assert q42["operator"]["support_frontier_closed"] is False
    assert q42["operator"]["decision"]["terminal_authorized"] is False

    q65 = by_ordinal[65]
    assert q65["operator"]["execution"]["status"] == "insufficient"
    assert q65["operator"]["generic_frontier_closed"] is False
    assert q65["operator"]["terminal_typed_contribution"]["frontier_mode"] == "bounded"
    assert q65["operator"]["terminal_typed_contribution"]["truncated"] is True
    assert q65["terminal_prompt"]["provider_input"]["specialist_advisories"][0][
        "scope"
    ] == "selected_action_linked_members_only"

    q74 = by_ordinal[74]
    assert q74["fitted_typed_prompt"]["allowed_handle_ids"] == [
        "H950001",
        "H950002",
    ]

    q79 = by_ordinal[79]
    assert q79["operator"]["bundle_population_count"] == 42
    assert q79["operator"]["bundle_selected_count"] == 12
    assert q79["operator"]["winner_handle_id"] == "H900012"
    assert q79["operator"]["global_exhaustiveness_claimed"] is False
    assert all(
        row["terminal_prompt"]["full_chat_plus_output_tokens"]
        <= HARD_COMPLETE_CHAT_TOKEN_CAP
        for row in rows
    )


def test_published_v4_audit_reports_selection_and_terminal_survival() -> None:
    artifact = read_sealed_json(_published(AUDIT_NAME))
    assert artifact.sha256 == EXPECTED_AUDIT_SHA256
    gate = artifact.payload["structural_gate"]
    assert gate["selection_source_target_hits"] == 6
    assert gate["selection_source_target_count"] == 6
    assert gate["terminal_source_target_hits"] == 6
    assert gate["terminal_source_target_count"] == 6
    assert gate["structural_gate_passed"] is True


def test_validator_rejects_q65_generic_frontier_upgrade() -> None:
    source = read_sealed_json(_published(CONSTRUCTION_NAME))
    payload = copy.deepcopy(source.payload)
    q65 = payload["questions"][1]
    operator = q65["operator"]
    operator["generic_frontier_closed"] = True
    operator_body = dict(operator)
    operator_body.pop("operator_receipt_sha256")
    operator["operator_receipt_sha256"] = identity_sha256(operator_body)
    question_body = dict(q65)
    question_body.pop("question_receipt_sha256")
    q65["question_receipt_sha256"] = identity_sha256(question_body)
    construction_body = dict(payload)
    construction_body.pop("construction_identity_sha256")
    payload["construction_identity_sha256"] = identity_sha256(construction_body)
    candidate = SealedArtifact(
        path=source.path,
        sha256=hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        payload=payload,
    )

    with pytest.raises(ReducedMissing4V4Error, match="q65"):
        validate_construction(candidate)
