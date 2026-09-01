from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from memory_condense.domain.integrity import file_sha256
from tools import build_exact11_target_witness_manifest as builder
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.contracts import identity_sha256


@pytest.fixture(scope="module")
def rebuilt_manifest() -> dict[str, object]:
    if not builder.DEFAULT_DATASET.is_file():
        pytest.skip("pinned LongMemEval dataset is unavailable")
    return builder.build_manifest(builder.DEFAULT_DATASET, builder.DEFAULT_TARGET_PLAN)


def test_rebuild_matches_sealed_exact11_witness_manifest(
    rebuilt_manifest: dict[str, object],
) -> None:
    artifact = read_sealed_json(builder.DEFAULT_OUTPUT)

    assert artifact.sha256 == (
        "f6add6368971d9b0b827bc0042c5e2a2e409f26df4f2a30ef18224c34c64bd60"
    )
    assert artifact.payload == rebuilt_manifest
    assert artifact.payload["manifest_identity_sha256"] == identity_sha256(
        {
            key: value
            for key, value in artifact.payload.items()
            if key != "manifest_identity_sha256"
        }
    )
    assert artifact.payload["provider_calls"] == 0
    assert artifact.payload["runtime_use_forbidden"] is True
    assert artifact.payload["analysis_is_posthoc_only"] is True


def test_manifest_separates_answer_atoms_links_and_q67_january_confounder(
    rebuilt_manifest: dict[str, object],
) -> None:
    positive = rebuilt_manifest["positive_witnesses"]
    negative = rebuilt_manifest["negative_witnesses"]
    assert isinstance(positive, list)
    assert isinstance(negative, list)
    kinds = Counter(row["witness_kind"] for row in positive)
    source_keys = {
        (row["ordinal"], row["question_id"], row["target_source_id"])
        for row in positive
    }

    assert len(positive) == 31
    assert kinds == {"answer_atom": 29, "relation_link": 2}
    assert len(source_keys) == 26
    assert len(negative) == 1
    assert all(
        row["witness_receipt_sha256"]
        == identity_sha256(
            {
                key: value
                for key, value in row.items()
                if key != "witness_receipt_sha256"
            }
        )
        for row in (*positive, *negative)
    )

    relation_hashes = {
        row["content_sha256"]
        for row in positive
        if row["witness_kind"] == "relation_link"
    }
    positive_hashes = {row["content_sha256"] for row in positive}
    assert relation_hashes == {
        "af5b78872c00d3220eeb536df70b4f93fa2c9e5d93c784af0a817f0995000c98",
        "a720bd59171b5017431b89e00d76ad14e9424f78ba00970f1385c3a16703e0af",
    }
    assert negative[0]["content_sha256"] == (
        "7763bc0082f3c69f650a4eb75aaf11ac13f9523633f1387f7998333d0973e066"
    )
    assert negative[0]["content_sha256"] not in positive_hashes
    assert {
        row["content_sha256"]
        for row in positive
        if row["ordinal"] == 67 and row["witness_kind"] == "answer_atom"
    } == {
        "c4a277eb04f4d4cc698588a6587e67528363bc6b057d4b67818e3964673c6bae",
        "80a65ca71ef0263e501876cc9ac6175b75ba60d685df0c50677ecf5ae5f27e17",
    }


def test_dataset_hash_and_population_mutations_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changed = tmp_path / "changed.json"
    changed.write_text("[]", encoding="utf-8")
    with pytest.raises(
        builder.Exact11TargetWitnessManifestError,
        match="dataset SHA-256 changed",
    ):
        builder._load_dataset(changed)  # noqa: SLF001

    malformed = tmp_path / "malformed.json"
    malformed.write_text(json.dumps([{}]), encoding="utf-8")
    monkeypatch.setattr(builder, "PINNED_DATASET_SHA256", file_sha256(malformed))
    with pytest.raises(
        builder.Exact11TargetWitnessManifestError,
        match="dataset population changed",
    ):
        builder._load_dataset(malformed)  # noqa: SLF001


def test_target_plan_hash_and_population_mutations_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = read_sealed_json(builder.DEFAULT_TARGET_PLAN).payload
    changed_body = {
        key: value for key, value in original.items() if key != "plan_sha256"
    }
    changed_body["question_count"] = 99
    changed = {
        **changed_body,
        "plan_sha256": identity_sha256(changed_body),
    }
    artifact, _created = publish_sealed_json(tmp_path / "target-plan.json", changed)

    with pytest.raises(
        builder.Exact11TargetWitnessManifestError,
        match="target-owner plan file SHA-256 changed",
    ):
        builder._load_target_plan(artifact.path)  # noqa: SLF001

    monkeypatch.setattr(builder, "PINNED_TARGET_PLAN_FILE_SHA256", artifact.sha256)
    monkeypatch.setattr(
        builder,
        "PINNED_TARGET_PLAN_IDENTITY_SHA256",
        changed["plan_sha256"],
    )
    with pytest.raises(
        builder.Exact11TargetWitnessManifestError,
        match="identity or population changed",
    ):
        builder._load_target_plan(artifact.path)  # noqa: SLF001
