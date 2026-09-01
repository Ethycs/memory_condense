from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from tools import build_exact11_semantic_atom_manifest as builder
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.contracts import identity_sha256


MANIFEST_SHA256 = (
    "c40bbfc78f07eccbd6b2e489b79f4ad1ba5221dea2aeb707c64ecf84ac514008"
)
MANIFEST_IDENTITY_SHA256 = (
    "f3e8ad4975d953eac16a98003626d7fb3ebc39b4a335e6fcea703e40f487c69c"
)
ATOM_POPULATION_SHA256 = (
    "e2a13b57f44f4b863df22b7d7e906bb6cd74e15c9b895add37bface21907c73c"
)


@pytest.fixture(scope="module")
def rebuilt_manifest() -> dict[str, object]:
    if not builder.DEFAULT_DATASET.is_file():
        pytest.skip("pinned LongMemEval dataset is unavailable")
    return builder.build_manifest(
        builder.DEFAULT_DATASET,
        builder.DEFAULT_TARGET_PLAN,
        builder.DEFAULT_RAW_WITNESS_MANIFEST,
    )


def _reseal_nested_mutation(tmp_path: Path, payload: dict) -> tuple[Path, str]:
    body = {
        key: value
        for key, value in payload.items()
        if key != "manifest_identity_sha256"
    }
    payload["manifest_identity_sha256"] = identity_sha256(body)
    artifact, _ = publish_sealed_json(tmp_path / "changed-atoms.json", payload)
    return artifact.path, artifact.sha256


def test_rebuild_matches_sealed_semantic_atom_manifest(
    rebuilt_manifest: dict[str, object],
) -> None:
    artifact = read_sealed_json(builder.DEFAULT_OUTPUT)

    assert artifact.sha256 == MANIFEST_SHA256
    assert artifact.payload == rebuilt_manifest
    assert artifact.payload["manifest_identity_sha256"] == (
        MANIFEST_IDENTITY_SHA256
    )
    assert artifact.payload["atom_population_sha256"] == ATOM_POPULATION_SHA256
    assert artifact.payload["atom_count"] == 26
    assert artifact.payload["exact_locator_count"] == 36
    assert artifact.payload["raw_witness_assignment_edge_count"] == 34
    assert artifact.payload["raw_witness_count"] == 31
    assert artifact.payload["provider_calls"] == 0
    assert artifact.payload["runtime_use_forbidden"] is True
    assert artifact.payload["terminal_answer_judge_artifacts_loaded"] is False
    assert builder.load_verified_manifest(
        artifact.path, MANIFEST_SHA256
    ).sha256 == MANIFEST_SHA256


def test_atom_declarations_preserve_temporal_singletons_and_no_overclaims(
    rebuilt_manifest: dict[str, object],
) -> None:
    atoms = rebuilt_manifest["atoms"]
    assert isinstance(atoms, list)
    by_key = {(row["ordinal"], row["atom_key"]): row for row in atoms}

    peace = by_key[(53, "plant_peace_lily")]
    assert {
        (row["source_id"], row["session_turn_index"])
        for row in peace["acceptable_evidence_locators"]
    } == {("answer_c2204106_2", 0), ("answer_c2204106_2", 2)}

    art_cube = by_key[(67, "venue_art_cube")]
    assert {
        (row["source_id"], row["session_turn_index"])
        for row in art_cube["acceptable_evidence_locators"]
    } == {
        ("answer_990c8992_2", 0),
        ("answer_990c8992_2", 8),
        ("answer_990c8992_2", 10),
    }
    natural_history = by_key[(67, "venue_natural_history")]
    assert {
        (row["source_id"], row["session_turn_index"])
        for row in natural_history["acceptable_evidence_locators"]
    } == {("answer_990c8992_1", 0)}

    assert by_key[(82, "bike_garmin")]["canonical_claim"] == (
        "The user has a new Garmin bike computer and plans to track rides with it."
    )
    assert by_key[(97, "ubereats_discount")]["canonical_claim"] == (
        "The user received 20 percent off an UberEats order; the source does not "
        "state that it was the first order."
    )
    assert {
        key for ordinal, key in by_key if ordinal == 94
    } == {"baking_class_date", "birthday_cake_date"}


def test_raw_relation_messages_are_associated_but_not_accepted_as_equivalence(
    rebuilt_manifest: dict[str, object],
) -> None:
    atoms = rebuilt_manifest["atoms"]
    assert isinstance(atoms, list)
    raw = read_sealed_json(builder.DEFAULT_RAW_WITNESS_MANIFEST).payload
    raw_by_turn = {
        (
            row["ordinal"],
            row["target_source_id"],
            row["session_turn_index"],
        ): row
        for row in raw["positive_witnesses"]
    }
    by_key = {(row["ordinal"], row["atom_key"]): row for row in atoms}

    for atom_key, raw_turn in (
        ((53, "plant_peace_lily"), (53, "answer_c2204106_3", 0)),
        ((67, "venue_art_cube"), (67, "answer_990c8992_3", 4)),
    ):
        atom = by_key[atom_key]
        relation = raw_by_turn[raw_turn]
        assert relation["witness_kind"] == "relation_link"
        assert relation["witness_receipt_sha256"] in atom[
            "raw_witness_receipt_sha256s"
        ]
        assert relation["content_sha256"] not in {
            locator["content_sha256"]
            for locator in atom["acceptable_evidence_locators"]
        }


def test_manifest_has_full_atom_role_and_raw_assignment_ledgers(
    rebuilt_manifest: dict[str, object],
) -> None:
    atoms = rebuilt_manifest["atoms"]
    assert isinstance(atoms, list)
    assert Counter(row["ordinal"] for row in atoms) == {
        14: 4,
        28: 2,
        40: 3,
        49: 2,
        53: 3,
        54: 1,
        67: 2,
        69: 3,
        82: 2,
        94: 2,
        97: 2,
    }
    assert len(
        {
            receipt
            for atom in atoms
            for receipt in atom["raw_witness_receipt_sha256s"]
        }
    ) == 31
    assert rebuilt_manifest["policy"]["builder_forbidden_inputs"] == [
        "terminal_construction",
        "terminal_replay",
        "answer_artifact",
        "judge_artifact",
        "provider_response",
    ]


def test_resealed_policy_relaxation_fails_closed(tmp_path: Path) -> None:
    payload = deepcopy(read_sealed_json(builder.DEFAULT_OUTPUT).payload)
    payload["policy"]["runtime_routing_use_forbidden"] = False
    policy_body = {
        key: value
        for key, value in payload["policy"].items()
        if key != "receipt_sha256"
    }
    payload["policy"]["receipt_sha256"] = identity_sha256(policy_body)
    path, artifact_sha = _reseal_nested_mutation(tmp_path, payload)

    with pytest.raises(
        builder.Exact11SemanticAtomManifestError,
        match="population, policy, or immutable binding changed",
    ):
        builder.load_verified_manifest(path, artifact_sha)


def test_resealed_noncanonical_locator_date_fails_closed(tmp_path: Path) -> None:
    payload = deepcopy(read_sealed_json(builder.DEFAULT_OUTPUT).payload)
    locator = payload["atoms"][0]["acceptable_evidence_locators"][0]
    locator["source_date_utc"] = locator["source_date_utc"].replace(
        "+00:00", "+01:00"
    )
    locator_body = {
        key: value
        for key, value in locator.items()
        if key != "locator_receipt_sha256"
    }
    locator["locator_receipt_sha256"] = identity_sha256(locator_body)
    atom = payload["atoms"][0]
    atom_body = {
        key: value for key, value in atom.items() if key != "atom_receipt_sha256"
    }
    atom["atom_receipt_sha256"] = identity_sha256(atom_body)
    atom_receipts = [row["atom_receipt_sha256"] for row in payload["atoms"]]
    payload["atom_population_sha256"] = identity_sha256(atom_receipts)
    path, artifact_sha = _reseal_nested_mutation(tmp_path, payload)

    with pytest.raises(
        builder.Exact11SemanticAtomManifestError,
        match="locator schema or authentication changed",
    ):
        builder.load_verified_manifest(path, artifact_sha)


def test_duplicate_same_source_content_cannot_claim_an_exact_turn() -> None:
    declaration = builder.ATOM_DECLARATIONS[0]
    evidence = declaration.acceptable[0]
    message = {"content": "same bytes", "has_answer": True, "role": "user"}
    dataset_by_question = {
        declaration.question_id: {
            "haystack_dates": ["2026/08/29 12:00"],
            "haystack_session_ids": [evidence.source_id],
            "haystack_sessions": [[message, dict(message)]],
        }
    }

    with pytest.raises(
        builder.Exact11SemanticAtomManifestError,
        match="not unique within its source",
    ):
        builder._locator_row(  # noqa: SLF001
            declaration,
            builder.EvidenceDeclaration(evidence.source_id, 0),
            dataset_by_question,
        )
