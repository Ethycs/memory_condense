from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from memory_condense.modeling.lfm_extract import (
    LFMCompletion,
    LFMExtractCompletion,
    lfm_checkpoint_identity,
)
from tools.assay_contextual_cards import _publish_no_clobber
from tools.assay_contextual_card_attention import _read_canonical


def _metadata(root: Path) -> None:
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
    ):
        (root / name).write_text(f"fixture:{name}", encoding="utf-8")


def test_checkpoint_identity_binds_single_weight_file(tmp_path: Path) -> None:
    _metadata(tmp_path)
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"single weights")

    identity = lfm_checkpoint_identity(
        tmp_path, model_id="Liquid/test", revision="abc"
    )

    assert identity["model_id"] == "Liquid/test"
    assert identity["revision"] == "abc"
    assert identity["checkpoint_files"]["model.safetensors"] == {
        "bytes": len(b"single weights"),
        "sha256": hashlib.sha256(b"single weights").hexdigest(),
    }


def test_checkpoint_identity_discovers_and_binds_shards(tmp_path: Path) -> None:
    _metadata(tmp_path)
    (tmp_path / "one.safetensors").write_bytes(b"one")
    (tmp_path / "two.safetensors").write_bytes(b"two")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.a": "one.safetensors",
                    "model.b": "two.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    files = lfm_checkpoint_identity(tmp_path)["checkpoint_files"]

    assert set(files) == {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "model.safetensors.index.json",
        "one.safetensors",
        "two.safetensors",
    }


def test_checkpoint_identity_rejects_shard_path_escape(tmp_path: Path) -> None:
    _metadata(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.a": "../outside.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid shard path"):
        lfm_checkpoint_identity(tmp_path)


def test_generic_completion_keeps_extract_compatibility_alias(tmp_path: Path) -> None:
    assert LFMExtractCompletion is LFMCompletion
    with pytest.raises(ValueError, match="dtype"):
        LFMCompletion(tmp_path, dtype="int8")


def test_no_clobber_publisher_preserves_preexisting_sidecar(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"
    sidecar = tmp_path / "artifact.json.sha256"
    sidecar.write_text("preexisting\n", encoding="ascii")

    with pytest.raises(FileExistsError):
        _publish_no_clobber(output, {"value": 1})

    assert not output.exists()
    assert sidecar.read_text(encoding="ascii") == "preexisting\n"


def test_no_clobber_publisher_writes_matching_digest(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"
    digest, sidecar = _publish_no_clobber(output, {"value": 1})

    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    assert sidecar.read_text(encoding="ascii") == f"{digest}  {output.name}\n"

    payload, replayed_digest = _read_canonical(output)
    assert payload == {"value": 1}
    assert replayed_digest == digest


def test_canonical_reader_rejects_crlf_digest_sidecar(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"
    _digest, sidecar = _publish_no_clobber(output, {"value": 1})
    sidecar.write_bytes(sidecar.read_bytes().replace(b"\n", b"\r\n"))

    with pytest.raises(ValueError, match="sidecar"):
        _read_canonical(output)
