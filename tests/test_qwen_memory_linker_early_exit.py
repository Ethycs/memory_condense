from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.associations.qwen_memory_linker import (
    _CoverageAttentionInputReady,
    _capture_coverage_attention_input,
)


class _HookHandle:
    def __init__(self, attention: _StubAttention) -> None:
        self._attention = attention

    def remove(self) -> None:
        self._attention.hook = None
        self._attention.remove_calls += 1


class _StubAttention:
    def __init__(self, layer: int) -> None:
        self.layer = layer
        self.hook: Any = None
        self.forward_calls = 0
        self.remove_calls = 0

    def register_forward_pre_hook(
        self,
        hook: Any,
        *,
        with_kwargs: bool,
    ) -> _HookHandle:
        assert with_kwargs is True
        assert self.hook is None
        self.hook = hook
        return _HookHandle(self)

    def __call__(self, *, hidden_states: str) -> tuple[str, None]:
        if self.hook is not None:
            self.hook(self, (), {"hidden_states": hidden_states})
        self.forward_calls += 1
        return f"{hidden_states}|layer{self.layer}.attention", None


class _StubLayer:
    def __init__(self, layer: int) -> None:
        self.layer = layer
        self.self_attn = _StubAttention(layer)
        self.input_norm_calls = 0
        self.mlp_calls = 0

    def __call__(self, hidden_states: str) -> str:
        self.input_norm_calls += 1
        normalized = f"{hidden_states}|layer{self.layer}.input_norm"
        attended, _ = self.self_attn(hidden_states=normalized)
        self.mlp_calls += 1
        return f"{attended}|layer{self.layer}.mlp"


class _StubPrefixModel:
    def __init__(
        self,
        *,
        stop_before_selected: bool = False,
        error: BaseException | None = None,
    ) -> None:
        self.layers = (_StubLayer(0), _StubLayer(1))
        self.stop_before_selected = stop_before_selected
        self.error = error
        self.final_norm_calls = 0
        self.calls = 0
        self.output_attentions: bool | None = None
        self.output_hidden_states: bool | None = None

    def __call__(
        self,
        *,
        inputs_embeds: str,
        use_cache: bool,
        output_attentions: bool = True,
        output_hidden_states: bool = True,
    ) -> str:
        self.calls += 1
        assert use_cache is False
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        if self.error is not None:
            raise self.error
        hidden = self.layers[0](inputs_embeds)
        if self.stop_before_selected:
            self.final_norm_calls += 1
            return f"{hidden}|final_norm"
        hidden = self.layers[1](hidden)
        self.final_norm_calls += 1
        return f"{hidden}|final_norm"


def _encoder(model: _StubPrefixModel) -> Any:
    return SimpleNamespace(model=model, layers=len(model.layers))


def test_coverage_early_exit_matches_full_forward_attention_input() -> None:
    full_model = _StubPrefixModel()
    captured: dict[str, str] = {}
    full_handle = full_model.layers[1].self_attn.register_forward_pre_hook(
        lambda _module, _args, kwargs: captured.update(
            hidden=kwargs["hidden_states"]
        ),
        with_kwargs=True,
    )
    full_model(inputs_embeds="embedded", use_cache=False)
    full_handle.remove()

    fast_model = _StubPrefixModel()
    actual = _capture_coverage_attention_input(
        _encoder(fast_model),
        {"inputs_embeds": "embedded"},
        layer=1,
    )

    assert actual == captured["hidden"]
    assert full_model.layers[1].self_attn.forward_calls == 1
    assert full_model.layers[1].mlp_calls == 1
    assert full_model.final_norm_calls == 1
    assert fast_model.layers[0].self_attn.forward_calls == 1
    assert fast_model.layers[0].mlp_calls == 1
    assert fast_model.layers[1].input_norm_calls == 1
    assert fast_model.layers[1].self_attn.forward_calls == 0
    assert fast_model.layers[1].mlp_calls == 0
    assert fast_model.final_norm_calls == 0
    assert fast_model.output_attentions is False
    assert fast_model.output_hidden_states is False
    assert fast_model.layers[1].self_attn.hook is None
    assert fast_model.layers[1].self_attn.remove_calls == 1


def test_coverage_early_exit_fails_closed_when_selected_hook_is_skipped() -> None:
    model = _StubPrefixModel(stop_before_selected=True)

    with pytest.raises(RuntimeError, match="early-exit hook did not execute"):
        _capture_coverage_attention_input(
            _encoder(model),
            {"inputs_embeds": "embedded"},
            layer=1,
        )

    assert model.layers[1].self_attn.hook is None
    assert model.layers[1].self_attn.remove_calls == 1


def test_coverage_early_exit_rejects_a_foreign_control_signal() -> None:
    foreign = _CoverageAttentionInputReady()
    model = _StubPrefixModel(error=foreign)

    with pytest.raises(_CoverageAttentionInputReady) as raised:
        _capture_coverage_attention_input(
            _encoder(model),
            {"inputs_embeds": "embedded"},
            layer=1,
        )

    assert raised.value is foreign
    assert model.layers[1].self_attn.hook is None
    assert model.layers[1].self_attn.remove_calls == 1
