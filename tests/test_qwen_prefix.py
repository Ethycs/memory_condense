from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from memory_condense.modeling.qwen_prefix import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    FIRST_SHARD,
    checkpoint_key_is_needed,
    complete_prefix_layers,
    expected_prefix_checkpoint_sha256,
    model_parameter_name,
    verify_prefix_checkpoint,
)


def _write_fake_prefix(root: Path) -> dict[str, str]:
    files: dict[str, bytes] = {
        "config.json": b'{"model_type":"qwen3"}',
        "tokenizer_config.json": b'{"tokenizer_class":"Qwen2Tokenizer"}',
        "tokenizer.json": b'{"version":"1.0"}',
        "vocab.json": b'{"a":0}',
        "merges.txt": b"#version: 0.2\n",
        FIRST_SHARD: b"fake-safe-tensors",
    }
    index = {
        "metadata": {"total_size": len(files[FIRST_SHARD])},
        "weight_map": {
            "model.embed_tokens.weight": FIRST_SHARD,
            "model.layers.0.self_attn.q_proj.weight": FIRST_SHARD,
        },
    }
    files["model.safetensors.index.json"] = json.dumps(
        index,
        sort_keys=True,
    ).encode("utf-8")
    for name, content in files.items():
        (root / name).write_bytes(content)
    return {
        name: hashlib.sha256(content).hexdigest()
        for name, content in files.items()
    }


def test_checkpoint_key_selection_keeps_only_complete_prefix() -> None:
    assert checkpoint_key_is_needed("model.embed_tokens.weight", 7)
    assert checkpoint_key_is_needed("model.layers.0.self_attn.q_proj.weight", 7)
    assert checkpoint_key_is_needed("model.layers.6.mlp.up_proj.weight", 7)
    assert not checkpoint_key_is_needed("model.layers.7.input_layernorm.weight", 7)
    assert not checkpoint_key_is_needed("model.norm.weight", 7)
    assert not checkpoint_key_is_needed("lm_head.weight", 7)


def test_complete_prefix_layers_rejects_a_split_layer() -> None:
    weight_map = {
        "model.layers.0.self_attn.q_proj.weight": "one.safetensors",
        "model.layers.0.self_attn.k_proj.weight": "one.safetensors",
        "model.layers.1.self_attn.q_proj.weight": "one.safetensors",
        "model.layers.1.self_attn.k_proj.weight": "two.safetensors",
        "model.layers.2.self_attn.q_proj.weight": "two.safetensors",
    }

    assert complete_prefix_layers(weight_map, {"one.safetensors"}) == 1
    assert complete_prefix_layers(weight_map, {"one.safetensors", "two.safetensors"}) == 3


def test_model_parameter_name_strips_causal_lm_wrapper() -> None:
    assert (
        model_parameter_name("model.layers.3.self_attn.v_proj.weight")
        == "layers.3.self_attn.v_proj.weight"
    )
    with pytest.raises(ValueError, match="base-model"):
        model_parameter_name("lm_head.weight")


def test_prefix_checkpoint_verifier_binds_all_runtime_files(tmp_path: Path) -> None:
    expected_files = _write_fake_prefix(tmp_path)

    identity = verify_prefix_checkpoint(
        tmp_path,
        layers=1,
        model_id="test/qwen",
        model_revision="revision",
        expected_file_sha256=expected_files,
    )

    assert identity.model_id == "test/qwen"
    assert identity.model_revision == "revision"
    assert len(identity.checkpoint_sha256) == 64
    assert identity.verified_files == (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors.index.json",
        FIRST_SHARD,
    )


def test_prefix_checkpoint_verifier_rejects_tampering(tmp_path: Path) -> None:
    expected_files = _write_fake_prefix(tmp_path)
    (tmp_path / FIRST_SHARD).write_bytes(b"tampered")

    with pytest.raises(ValueError, match=FIRST_SHARD):
        verify_prefix_checkpoint(
            tmp_path,
            layers=1,
            expected_file_sha256=expected_files,
        )


def test_pinned_prefix_manifest_is_layer_scoped() -> None:
    two_layer = expected_prefix_checkpoint_sha256(2)

    assert len(two_layer) == 64
    assert two_layer == expected_prefix_checkpoint_sha256(7)
    assert two_layer != expected_prefix_checkpoint_sha256(8)


def test_qwen_download_tasks_pin_the_checkpoint_revision() -> None:
    pixi = (Path(__file__).parents[1] / "pixi.toml").read_text(encoding="utf-8")
    needle = f"Qwen/Qwen3-8B --revision {DEFAULT_MODEL_REVISION}"

    assert pixi.count(needle) == 2
    assert DEFAULT_MODEL_ID == "Qwen/Qwen3-8B"
