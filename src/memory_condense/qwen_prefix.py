"""Live Qwen3 prefix encoder for attention-head memory experiments.

The official Qwen3-8B checkpoint is split into five safetensors shards.  The
first shard contains the token embeddings, all parameters for layers 0--6,
and only part of layer 7.  This module deliberately constructs a seven-layer
model and ignores the incomplete layer-7 tensors, allowing the useful prefix
to run without downloading or loading the rest of the language model.

This is an experimental component.  It does not participate in the existing
``MemoryCondenser`` path yet.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"
DEFAULT_PREFIX_LAYERS = 7
FIRST_SHARD = "model-00001-of-00005.safetensors"

_LAYER_KEY = re.compile(r"^model\.layers\.(\d+)\.")


def _require_torch_stack() -> tuple[Any, ...]:
    """Import the heavyweight experimental dependencies only when requested."""
    try:
        import torch
        import torch.nn.functional as functional
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        from safetensors import safe_open
        from transformers import AutoTokenizer, Qwen3Config, Qwen3Model
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "The Qwen prefix lab requires the Pixi environment. "
            "Run it with `pixi run -e dev qwen-smoke`."
        ) from exc
    return (
        torch,
        functional,
        init_empty_weights,
        set_module_tensor_to_device,
        safe_open,
        AutoTokenizer,
        Qwen3Config,
        Qwen3Model,
        apply_rotary_pos_emb,
    )


def checkpoint_key_is_needed(key: str, layers: int) -> bool:
    """Return whether a full-model checkpoint tensor belongs to this prefix."""
    if key == "model.embed_tokens.weight":
        return True
    match = _LAYER_KEY.match(key)
    return match is not None and int(match.group(1)) < layers


def model_parameter_name(checkpoint_key: str) -> str:
    """Map a ``Qwen3ForCausalLM`` checkpoint key to ``Qwen3Model``."""
    if not checkpoint_key.startswith("model."):
        raise ValueError(f"not a base-model checkpoint key: {checkpoint_key}")
    return checkpoint_key.removeprefix("model.")


def complete_prefix_layers(weight_map: dict[str, str], shards: set[str]) -> int:
    """Count contiguous, complete decoder layers contained in ``shards``.

    Completeness is determined against the repository's full safetensors
    index rather than by assuming a fixed number of tensors per layer.
    """
    expected: dict[int, set[str]] = {}
    present: dict[int, set[str]] = {}
    for key, shard in weight_map.items():
        match = _LAYER_KEY.match(key)
        if match is None:
            continue
        layer = int(match.group(1))
        expected.setdefault(layer, set()).add(key)
        if shard in shards:
            present.setdefault(layer, set()).add(key)

    count = 0
    while expected.get(count) and present.get(count) == expected[count]:
        count += 1
    return count


def inspect_prefix_checkpoint(model_dir: str | Path) -> dict[str, Any]:
    """Describe the locally available Qwen shards and valid prefix length."""
    root = Path(model_dir)
    index_path = root / "model.safetensors.index.json"
    config_path = root / "config.json"
    if not index_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            f"missing Qwen metadata under {root}; download config.json and "
            "model.safetensors.index.json first"
        )

    index = json.loads(index_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    local_shards = {
        shard
        for shard in set(index["weight_map"].values())
        if (root / shard).is_file()
    }
    return {
        "model_type": config.get("model_type"),
        "hidden_size": int(config["hidden_size"]),
        "model_layers": int(config["num_hidden_layers"]),
        "attention_heads": int(config["num_attention_heads"]),
        "key_value_heads": int(config["num_key_value_heads"]),
        "checkpoint_dtype": config.get("torch_dtype"),
        "checkpoint_bytes": int(index.get("metadata", {}).get("total_size", 0)),
        "local_shards": sorted(local_shards),
        "complete_prefix_layers": complete_prefix_layers(
            index["weight_map"], local_shards
        ),
    }


@dataclass(slots=True)
class QwenHeadCapture:
    """Signals captured from one real teacher attention layer."""

    layer: int
    layer_input: Any
    residual: Any
    attention_input: Any
    queries: Any
    keys: Any
    values: Any
    attention: Any | None
    mixed_values: Any
    attention_output: Any
    _o_weight: Any
    head_dim: int

    def output_for_head(self, head: int) -> Any:
        """Return this head's additive contribution after its slice of W_O."""
        torch, functional, *_ = _require_torch_stack()
        heads = self.mixed_values.shape[-1] // self.head_dim
        if not 0 <= head < heads:
            raise IndexError(f"head must be in [0, {heads}), got {head}")
        start = head * self.head_dim
        stop = start + self.head_dim
        with torch.inference_mode():
            return functional.linear(
                self.mixed_values[..., start:stop],
                self._o_weight[:, start:stop],
            )


class Qwen3PrefixEncoder:
    """A coherent prefix of Qwen3-8B loaded directly from selected shards."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        layers: int = DEFAULT_PREFIX_LAYERS,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
        (
            self._torch,
            _,
            init_empty_weights,
            set_module_tensor_to_device,
            safe_open,
            AutoTokenizer,
            Qwen3Config,
            Qwen3Model,
            self._apply_rotary_pos_emb,
        ) = _require_torch_stack()

        self.model_dir = Path(model_dir)
        self.layers = int(layers)
        self.device = self._torch.device(device)
        self.dtype = _torch_dtype(self._torch, dtype)

        inventory = inspect_prefix_checkpoint(self.model_dir)
        available = int(inventory["complete_prefix_layers"])
        if not 1 <= self.layers <= available:
            raise ValueError(
                f"requested {self.layers} layers, but the downloaded shards "
                f"contain only {available} complete prefix layers"
            )
        if self.device.type == "cuda" and not self._torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

        config = Qwen3Config.from_pretrained(self.model_dir, local_files_only=True)
        config.num_hidden_layers = self.layers
        config.use_cache = False
        config._attn_implementation = "eager"
        self.config = config

        with init_empty_weights(include_buffers=False):
            model = Qwen3Model(config)

        loaded: set[str] = set()
        index = json.loads(
            (self.model_dir / "model.safetensors.index.json").read_text(
                encoding="utf-8"
            )
        )
        required_shards = sorted(
            {
                shard
                for key, shard in index["weight_map"].items()
                if checkpoint_key_is_needed(key, self.layers)
            }
        )
        for shard_name in required_shards:
            shard_path = self.model_dir / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(f"required checkpoint shard is missing: {shard_path}")
            with safe_open(shard_path, framework="pt", device="cpu") as checkpoint:
                for key in checkpoint.keys():
                    if not checkpoint_key_is_needed(key, self.layers):
                        continue
                    parameter_name = model_parameter_name(key)
                    set_module_tensor_to_device(
                        model,
                        parameter_name,
                        self.device,
                        value=checkpoint.get_tensor(key),
                        dtype=self.dtype,
                    )
                    loaded.add(parameter_name)

        # The teacher's final norm lives in a later shard.  A prefix memory
        # uses pre-norm layer residuals, but Qwen3Model.forward still executes
        # this module, so materialize its neutral learned weight (all ones).
        set_module_tensor_to_device(
            model,
            "norm.weight",
            self.device,
            value=self._torch.ones(config.hidden_size, dtype=self.dtype),
            dtype=self.dtype,
        )

        missing = [
            name
            for name, parameter in model.named_parameters()
            if parameter.device.type == "meta"
        ]
        if missing:
            raise RuntimeError(f"prefix parameters were not materialized: {missing[:5]}")

        model.to(self.device)
        model.requires_grad_(False)
        model.eval()
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, local_files_only=True
        )
        self.loaded_parameter_names = frozenset(loaded)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def encode(self, text: str) -> Any:
        """Return the actual last-layer residual before the synthetic final norm."""
        captured: dict[str, Any] = {}

        def save_layer(_module: Any, _args: Any, output: Any) -> None:
            captured["residual"] = output

        handle = self.model.layers[-1].register_forward_hook(save_layer)
        try:
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self._torch.inference_mode():
                self.model(**inputs, use_cache=False)
        finally:
            handle.remove()
        return captured["residual"]

    def encode_layers(
        self,
        texts: Sequence[str],
        *,
        layers: Sequence[int] | None = None,
        batch_size: int = 8,
    ) -> dict[int, Any]:
        """Mean-pool pre-norm residuals from several retained layers.

        Returned tensors are CPU ``float32`` arrays of shape
        ``[len(texts), hidden_size]``.  Capturing all candidate layers during
        the same prefix pass makes layer-wise CAV probes inexpensive.
        """
        if not texts:
            return {}
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        selected = tuple(range(self.layers)) if layers is None else tuple(layers)
        if not selected:
            raise ValueError("at least one layer must be selected")
        if len(set(selected)) != len(selected):
            raise ValueError("layers must not contain duplicates")
        invalid = [layer for layer in selected if not 0 <= layer < self.layers]
        if invalid:
            raise IndexError(
                f"layers must be in [0, {self.layers}); invalid values: {invalid}"
            )

        pooled: dict[int, list[Any]] = {layer: [] for layer in selected}
        captured: dict[int, Any] = {}
        handles = []
        for layer in selected:
            def save_layer(
                _module: Any,
                _args: Any,
                output: Any,
                *,
                layer_index: int = layer,
            ) -> None:
                captured[layer_index] = output

            handles.append(self.model.layers[layer].register_forward_hook(save_layer))

        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        try:
            for start in range(0, len(texts), batch_size):
                captured.clear()
                batch = list(texts[start : start + batch_size])
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with self._torch.inference_mode():
                    self.model(**inputs, use_cache=False)
                    for layer in selected:
                        vectors = mean_pool_residual(
                            captured[layer], inputs.get("attention_mask")
                        )
                        pooled[layer].append(vectors.float().cpu())
        finally:
            self.tokenizer.padding_side = original_padding_side
            for handle in handles:
                handle.remove()

        return {
            layer: self._torch.cat(layer_batches, dim=0)
            for layer, layer_batches in pooled.items()
        }

    def capture_layers(
        self,
        text: str,
        *,
        layers: Sequence[int],
    ) -> dict[int, QwenHeadCapture]:
        """Capture complete head signals for several layers in one prefix pass."""
        selected = tuple(int(layer) for layer in layers)
        if not selected:
            raise ValueError("at least one layer must be selected")
        if len(set(selected)) != len(selected):
            raise ValueError("layers must not contain duplicates")
        invalid = [layer for layer in selected if not 0 <= layer < self.layers]
        if invalid:
            raise IndexError(
                f"layers must be in [0, {self.layers}); invalid values: {invalid}"
            )

        saved: dict[int, dict[str, Any]] = {layer: {} for layer in selected}
        handles = []
        for layer in selected:
            decoder = self.model.layers[layer]
            attention = decoder.self_attn

            def save_layer_input(
                _module: Any,
                args: Any,
                *,
                layer_index: int = layer,
            ) -> None:
                saved[layer_index]["layer_input"] = args[0]

            def save_attention_input(
                _module: Any,
                _args: Any,
                kwargs: dict[str, Any],
                *,
                layer_index: int = layer,
            ) -> None:
                saved[layer_index]["attention_input"] = kwargs["hidden_states"]

            def save_mixed_values(
                _module: Any,
                args: Any,
                *,
                layer_index: int = layer,
            ) -> None:
                saved[layer_index]["mixed_values"] = args[0]

            def save_attention_output(
                _module: Any,
                _args: Any,
                _kwargs: Any,
                output: Any,
                *,
                layer_index: int = layer,
            ) -> None:
                saved[layer_index]["attention_output"] = output[0]
                saved[layer_index]["attention"] = output[1]

            def save_residual(
                _module: Any,
                _args: Any,
                output: Any,
                *,
                layer_index: int = layer,
            ) -> None:
                saved[layer_index]["residual"] = output

            handles.extend(
                [
                    decoder.register_forward_pre_hook(save_layer_input),
                    attention.register_forward_pre_hook(
                        save_attention_input, with_kwargs=True
                    ),
                    attention.o_proj.register_forward_pre_hook(save_mixed_values),
                    attention.register_forward_hook(
                        save_attention_output, with_kwargs=True
                    ),
                    decoder.register_forward_hook(save_residual),
                ]
            )

        try:
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            sequence_length = inputs["input_ids"].shape[1]
            position_ids = self._torch.arange(
                sequence_length, device=self.device
            ).unsqueeze(0)
            with self._torch.inference_mode():
                # Do not ask every retained layer to return its LxL attention
                # map. We reconstruct maps only for selected layers below,
                # keeping the transient linker bounded by its own workspace.
                self.model(**inputs, use_cache=False, output_attentions=False)
                captures: dict[int, QwenHeadCapture] = {}
                for layer in selected:
                    decoder = self.model.layers[layer]
                    attention = decoder.self_attn
                    layer_saved = saved[layer]
                    hidden = layer_saved["attention_input"]
                    input_shape = hidden.shape[:-1]
                    head_shape = (*input_shape, -1, attention.head_dim)
                    queries = attention.q_norm(
                        attention.q_proj(hidden).view(head_shape)
                    ).transpose(1, 2)
                    keys = attention.k_norm(
                        attention.k_proj(hidden).view(head_shape)
                    ).transpose(1, 2)
                    values = attention.v_proj(hidden).view(head_shape).transpose(1, 2)
                    cos, sin = self.model.rotary_emb(hidden, position_ids)
                    queries, keys = self._apply_rotary_pos_emb(
                        queries, keys, cos, sin
                    )
                    groups_per_key = (
                        self.config.num_attention_heads
                        // self.config.num_key_value_heads
                    )
                    key_groups = (
                        self._torch.arange(
                            self.config.num_attention_heads,
                            device=self.device,
                        )
                        // groups_per_key
                    )
                    expanded_keys = keys[:, key_groups]
                    logits = self._torch.einsum(
                        "bhqd,bhkd->bhqk", queries, expanded_keys
                    ) * float(attention.scaling)
                    causal_mask = self._torch.triu(
                        self._torch.ones(
                            sequence_length,
                            sequence_length,
                            dtype=self._torch.bool,
                            device=self.device,
                        ),
                        diagonal=1,
                    )
                    logits = logits.masked_fill(
                        causal_mask.view(1, 1, sequence_length, sequence_length),
                        self._torch.finfo(logits.dtype).min,
                    )
                    selected_attention = logits.float().softmax(dim=-1).to(
                        values.dtype
                    )
                    captures[layer] = QwenHeadCapture(
                        layer=layer,
                        layer_input=layer_saved["layer_input"],
                        residual=layer_saved["residual"],
                        attention_input=layer_saved["attention_input"],
                        queries=queries,
                        keys=keys,
                        values=values,
                        attention=selected_attention,
                        mixed_values=layer_saved["mixed_values"],
                        attention_output=layer_saved["attention_output"],
                        _o_weight=attention.o_proj.weight,
                        head_dim=int(attention.head_dim),
                    )
        finally:
            for handle in handles:
                handle.remove()
        return captures

    def capture(self, text: str, *, layer: int | None = None) -> QwenHeadCapture:
        """Capture residual, Q/K/V, attention, and OV signals for one layer."""
        layer = self.layers - 1 if layer is None else int(layer)
        return self.capture_layers(text, layers=(layer,))[layer]


def mean_pool_residual(residual: Any, attention_mask: Any | None = None) -> Any:
    """Pool token residuals into one vector suitable for CAV fitting."""
    if attention_mask is None:
        return residual.mean(dim=1)
    weights = attention_mask.to(device=residual.device, dtype=residual.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1)
    return (residual * weights).sum(dim=1) / denominator


def _torch_dtype(torch: Any, name: str) -> Any:
    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {name!r}; choose {sorted(aliases)}") from exc


def _shape(value: Any | None) -> list[int] | None:
    return None if value is None else list(value.shape)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=DEFAULT_PREFIX_LAYERS)
    parser.add_argument("--capture-layer", type=int)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--text",
        default="A live memory system links related concepts through attention.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch, *_ = _require_torch_stack()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    inventory = inspect_prefix_checkpoint(args.model_dir)
    print("checkpoint", json.dumps(inventory, indent=2))
    encoder = Qwen3PrefixEncoder(
        args.model_dir,
        layers=args.layers,
        device=args.device,
        dtype=args.dtype,
    )
    capture = encoder.capture(args.text, layer=args.capture_layer)
    head_output = capture.output_for_head(args.head)
    report = {
        "device": str(encoder.device),
        "runtime_dtype": str(encoder.dtype),
        "parameters": encoder.parameter_count,
        "capture_layer": capture.layer,
        "residual": _shape(capture.residual),
        "queries": _shape(capture.queries),
        "keys": _shape(capture.keys),
        "values": _shape(capture.values),
        "attention": _shape(capture.attention),
        "attention_output": _shape(capture.attention_output),
        "selected_head_output": _shape(head_output),
    }
    if encoder.device.type == "cuda":
        report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated(encoder.device)
    print("capture", json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
