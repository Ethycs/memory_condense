from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools import run_locked_specialist_final_construction_v2 as arm
from tools.matched_eval.artifacts import SealedArtifact, read_sealed_json
from tools.matched_eval.contracts import canonical_json_bytes, identity_sha256


EXPECTED_CONSTRUCTION_SHA256 = (
    "663d3b34c463c5e28243b8408c17fa431ea7eb9d7720f61b46bb68ba862629fb"
)


@pytest.fixture(scope="module")
def published() -> tuple[
    SealedArtifact,
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    path = arm.DEFAULT_OUTPUT_ROOT / arm.CONSTRUCTION_NAME
    if not path.exists():
        pytest.skip("full-100 v2 runtime construction is not present")
    artifact = read_sealed_json(path)
    assert artifact.sha256 == EXPECTED_CONSTRUCTION_SHA256
    base_artifact, base_rows = arm._load_v1(arm.DEFAULT_V1_CONSTRUCTION)  # noqa: SLF001
    v4_artifact, v4_rows = arm._load_v4(arm.DEFAULT_V4_CONSTRUCTION)  # noqa: SLF001
    rows = arm.validate_construction(
        artifact,
        base_v1=(base_artifact, base_rows),
        source_v4=(v4_artifact, v4_rows),
    )
    return artifact, rows, base_rows, v4_rows


def test_replacement_partition_is_exact_and_disjoint() -> None:
    assert len(arm.LATEST_STATE_ORDINALS) == 10
    assert len(arm.REPAIRED_OPERATOR_ORDINALS) == 3
    assert len(arm.REPLACED_ORDINALS) == 13
    assert len(arm.PRESERVED_ORDINALS) == 87
    assert set(arm.REPLACED_ORDINALS).isdisjoint(arm.PRESERVED_ORDINALS)
    assert set(arm.REPLACED_ORDINALS) | set(arm.PRESERVED_ORDINALS) == set(
        range(100)
    )


def test_typed_latest_state_route_overrides_but_seals_legacy_routing() -> None:
    question_sha = quote_sha256("latest-state question")
    legacy = {"style": "direct_extract", "reason": "direct_fallback"}

    route = arm._fixed_temporal_route(  # noqa: SLF001
        question_sha256=question_sha,
        legacy_route=legacy,
    )

    body = dict(route)
    declared = body.pop("receipt_sha256")
    assert declared == identity_sha256(body)
    assert route["question_sha256"] == question_sha
    assert route["temporal_mode"] == "latest_state"
    assert route["route_basis"] == "typed_operator_spec.temporal_mode"
    assert route["applicable_specialist_ids"] == [
        arm.TEMPORAL_MECHANISM_ID
    ]
    assert route["legacy_route"] == legacy


def test_published_v2_preserves_87_rows_and_promotes_13(
    published: tuple[
        SealedArtifact,
        tuple[dict[str, Any], ...],
        tuple[dict[str, Any], ...],
        tuple[dict[str, Any], ...],
    ],
) -> None:
    artifact, rows, base_rows, _v4_rows = published
    payload = artifact.payload

    assert all(rows[ordinal] == base_rows[ordinal] for ordinal in arm.PRESERVED_ORDINALS)
    assert all(rows[ordinal] != base_rows[ordinal] for ordinal in arm.REPLACED_ORDINALS)
    assert payload["byte_identical_v1_row_count"] == 87
    assert payload["specialist_provider_prompt_count"] == 69
    assert payload["repaired_operator_provider_prompt_count"] == 3
    assert payload["parent_passthrough_count"] == 28
    assert payload["provider_prompt_count"] == 72
    assert payload["max_terminal_complete_envelope_tokens"] == 7475
    assert payload["new_provider_calls"] == 0
    assert payload["retained_transformer_token_state_bytes"] == 0

    scans = payload["replacement_resident_index_lifecycle"]["receipts"]
    assert len(scans) == 7
    assert sorted(ordinal for scan in scans for ordinal in scan["ordinals"]) == list(
        arm.LATEST_STATE_ORDINALS
    )
    assert all(scan["database_read_passes"] == 1 for scan in scans)


def test_latest_state_rows_have_temporal_winners_under_8k(
    published: tuple[
        SealedArtifact,
        tuple[dict[str, Any], ...],
        tuple[dict[str, Any], ...],
        tuple[dict[str, Any], ...],
    ],
) -> None:
    _artifact, rows, _base_rows, _v4_rows = published

    for ordinal in arm.LATEST_STATE_ORDINALS:
        row = rows[ordinal]
        advisories = row["terminal_prompt"]["provider_input"][
            "specialist_advisories"
        ]
        assert row["mode"] == "specialist"
        assert row["applicable_specialist_ids"] == [arm.TEMPORAL_MECHANISM_ID]
        assert row["route"]["temporal_mode"] == "latest_state"
        assert len(advisories) == 1
        assert advisories[0]["mechanism_id"] == arm.TEMPORAL_MECHANISM_ID
        assert advisories[0]["temporal_bundle"]["winner_handle_id"] in row[
            "fitted_typed_prompt"
        ]["allowed_handle_ids"]
        assert row["terminal_prompt"]["full_chat_plus_output_tokens"] <= 8000


def test_repaired_operator_rows_retain_v4_evidence_and_v1_parent(
    published: tuple[
        SealedArtifact,
        tuple[dict[str, Any], ...],
        tuple[dict[str, Any], ...],
        tuple[dict[str, Any], ...],
    ],
) -> None:
    _artifact, rows, base_rows, v4_rows = published
    v4_by_ordinal = {row["ordinal"]: row for row in v4_rows}

    for ordinal in arm.REPAIRED_OPERATOR_ORDINALS:
        row = rows[ordinal]
        source = v4_by_ordinal[ordinal]
        assert row["mode"] == "repaired_operator"
        assert row["parent_source"] == base_rows[ordinal]["parent_source"]
        assert row["operator"] == source["operator"]
        assert row["selection"] == source["selection"]
        assert row["local_provenance"] == source["local_provenance"]
        assert row["methods"] == source["methods"]

    q65 = rows[65]
    assert q65["operator"]["generic_frontier_closed"] is False
    assert q65["operator"]["terminal_typed_contribution"]["frontier_mode"] == "bounded"

    q74 = rows[74]
    q74_source = v4_by_ordinal[74]
    assert q74["v4_terminal_rebased_to_v1_parent"] is True
    assert q74["fitted_typed_prompt"]["allowed_handle_ids"] == [
        "H950001",
        "H950002",
    ]
    assert q74["terminal_prompt"]["provider_input"]["typed_evidence"] == q74_source[
        "terminal_prompt"
    ]["provider_input"]["typed_evidence"]
    assert q74["terminal_prompt"]["provider_input"]["protected_parent_fallback"][
        "prediction"
    ] == base_rows[74]["parent_source"]["prediction"]


def test_validator_rejects_a_mutated_preserved_v1_row(
    published: tuple[
        SealedArtifact,
        tuple[dict[str, Any], ...],
        tuple[dict[str, Any], ...],
        tuple[dict[str, Any], ...],
    ],
) -> None:
    artifact, _rows, base_rows, v4_rows = published
    payload = copy.deepcopy(artifact.payload)
    payload["questions"][0]["question_id"] = "mutated-preserved-row"
    unsigned = dict(payload)
    unsigned.pop("construction_identity_sha256")
    payload["construction_identity_sha256"] = identity_sha256(unsigned)
    candidate = SealedArtifact(
        artifact.path,
        hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        payload,
    )
    base_artifact, _ = arm._load_v1(arm.DEFAULT_V1_CONSTRUCTION)  # noqa: SLF001
    v4_artifact, _ = arm._load_v4(arm.DEFAULT_V4_CONSTRUCTION)  # noqa: SLF001

    with pytest.raises(
        arm.LockedSpecialistFinalConstructionV2Error,
        match="preserved v1 row changed",
    ):
        arm.validate_construction(
            candidate,
            base_v1=(base_artifact, base_rows),
            source_v4=(v4_artifact, v4_rows),
        )
