from __future__ import annotations

import pytest

from memory_condense.qwen_prefix import (
    checkpoint_key_is_needed,
    complete_prefix_layers,
    model_parameter_name,
)


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
