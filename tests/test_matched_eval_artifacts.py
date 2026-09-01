from __future__ import annotations

import json

import pytest

from tools.matched_eval.artifacts import (
    SealedArtifactError,
    publish_sealed_json,
    read_sealed_json,
)


def test_publish_is_canonical_sealed_and_idempotent(tmp_path) -> None:
    path = tmp_path / "artifact.json"
    first, first_created = publish_sealed_json(path, {"z": 1, "a": [2]})
    second, second_created = publish_sealed_json(path, {"a": [2], "z": 1})

    assert first_created is True
    assert second_created is False
    assert first.sha256 == second.sha256
    assert path.read_bytes() == b'{"a":[2],"z":1}\n'
    assert not tuple(tmp_path.glob("*.tmp"))


def test_publish_refuses_to_replace_a_different_artifact(tmp_path) -> None:
    path = tmp_path / "artifact.json"
    publish_sealed_json(path, {"value": 1})

    with pytest.raises(SealedArtifactError, match="refusing"):
        publish_sealed_json(path, {"value": 2})


def test_reader_rejects_noncanonical_json_even_with_a_matching_sidecar(tmp_path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps({"z": 1, "a": 2}, indent=2), encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )

    with pytest.raises(SealedArtifactError, match="canonical"):
        read_sealed_json(path)
