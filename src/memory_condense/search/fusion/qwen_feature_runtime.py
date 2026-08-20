"""Cold-import-safe runtime attestation for the resident Qwen feature arm."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any


_ENCODER_INSTANCE_FIELDS = frozenset(
    {
        "model_dir",
        "layers",
        "model_id",
        "model_revision",
        "checkpoint_identity",
        "checkpoint_sha256",
        "_torch",
        "_apply_rotary_pos_emb",
        "device",
        "dtype",
        "dtype_name",
        "config",
        "model",
        "tokenizer",
        "loaded_parameter_names",
    }
)
_UNSUPPORTED_RUNTIME_VALUE = object()


@dataclass(frozen=True, slots=True)
class _OwnedRuntimeSnapshot:
    fingerprint: tuple[Any, ...]
    device: str
    execution_dtype: str
    hidden_dim: int


def _canonical_cuda_device(device: Any, torch: Any) -> str:
    normalized = torch.device(device)
    if normalized.type != "cuda" or normalized.index is None:
        raise RuntimeError("resident Qwen parameters require an indexed CUDA device")
    return f"cuda:{normalized.index}"


def _reject_hooks(module: Any) -> None:
    hook_fields = (
        "_forward_pre_hooks",
        "_forward_hooks",
        "_backward_pre_hooks",
        "_backward_hooks",
    )
    if any(getattr(module, name, ()) for name in hook_fields):
        raise RuntimeError("owned Qwen runtime cannot carry foreign execution hooks")


def _reject_global_module_hooks(torch: Any) -> None:
    """Reject process-global PyTorch module hooks at the owned boundary."""

    module_runtime = torch.nn.modules.module
    registry_names = (
        "_global_forward_pre_hooks",
        "_global_forward_hooks",
        "_global_forward_hooks_always_called",
        "_global_forward_hooks_with_kwargs",
        "_global_forward_pre_hooks_with_kwargs",
        "_global_backward_pre_hooks",
        "_global_backward_hooks",
    )
    if any(getattr(module_runtime, name, ()) for name in registry_names):
        raise RuntimeError("owned Qwen runtime cannot run with global module hooks")


def _bounded_runtime_value(value: Any, *, depth: int = 0) -> Any:
    """Project cheap scalar execution metadata without reading tensor bytes."""

    if value is None or type(value) in {bool, int, str}:
        return (type(value).__name__, value)
    if type(value) is float:
        return ("float", repr(value))
    if depth >= 3:
        return _UNSUPPORTED_RUNTIME_VALUE
    if type(value) in {tuple, list} and len(value) <= 256:
        projected = tuple(
            _bounded_runtime_value(item, depth=depth + 1) for item in value
        )
        if all(item is not _UNSUPPORTED_RUNTIME_VALUE for item in projected):
            return (type(value).__name__, projected)
    if type(value) is dict and len(value) <= 256:
        items: list[tuple[Any, Any]] = []
        for key, item in value.items():
            projected_key = _bounded_runtime_value(key, depth=depth + 1)
            projected_item = _bounded_runtime_value(item, depth=depth + 1)
            if (
                projected_key is _UNSUPPORTED_RUNTIME_VALUE
                or projected_item is _UNSUPPORTED_RUNTIME_VALUE
            ):
                return _UNSUPPORTED_RUNTIME_VALUE
            items.append((projected_key, projected_item))
        return ("dict", tuple(sorted(items, key=repr)))
    return _UNSUPPORTED_RUNTIME_VALUE


def _declared_runtime_fields(value: Any, *, include_private: bool) -> tuple[Any, ...]:
    projected: list[tuple[str, Any]] = []
    for name, item in vars(value).items():
        if not include_private and name.startswith("_"):
            continue
        encoded = _bounded_runtime_value(item)
        if encoded is not _UNSUPPORTED_RUNTIME_VALUE:
            projected.append((name, encoded))
    return tuple(sorted(projected))


def _runtime_fingerprint(
    encoder: Any,
    *,
    torch: Any,
    expected_encoder_type: type,
    expected_model_type: type,
    expected_tokenizer_type: type,
    expected_config_type: type,
) -> _OwnedRuntimeSnapshot:
    """Validate and snapshot cheap execution-affecting owned runtime state.

    This deliberately does not read parameter, buffer, tokenizer-backend, or
    other loaded-content bytes. The provider receipt states those exclusions.
    """

    if type(encoder) is not expected_encoder_type:
        raise TypeError("provider requires the exact owned Qwen3PrefixEncoder")
    if set(vars(encoder)) != _ENCODER_INSTANCE_FIELDS:
        raise RuntimeError("Qwen encoder instance fields changed")
    if encoder._torch is not torch:
        raise RuntimeError("Qwen encoder torch runtime changed")
    if type(encoder.model) is not expected_model_type:
        raise TypeError("Qwen encoder model is not the exact owned runtime type")
    if type(encoder.tokenizer) is not expected_tokenizer_type:
        raise TypeError("Qwen encoder tokenizer is not the exact owned runtime type")
    if type(encoder.config) is not expected_config_type:
        raise TypeError("Qwen encoder config is not the exact owned runtime type")
    _reject_global_module_hooks(torch)
    primitive = getattr(encoder, "_encode_selected_layer_final_readout", None)
    if not (
        getattr(primitive, "__self__", None) is encoder
        and getattr(primitive, "__func__", None)
        is expected_encoder_type._encode_selected_layer_final_readout
    ):
        raise RuntimeError("Qwen final-readout primitive was shadowed")
    primitive_function = expected_encoder_type._encode_selected_layer_final_readout
    if (
        getattr(primitive_function, "__module__", None)
        != "memory_condense.modeling.qwen_prefix"
        or getattr(primitive_function, "__qualname__", None)
        != "Qwen3PrefixEncoder._encode_selected_layer_final_readout"
        or inspect.getsourcefile(primitive_function)
        != inspect.getsourcefile(expected_encoder_type)
    ):
        raise RuntimeError("Qwen final-readout primitive lacks owned source identity")
    model = encoder.model
    if getattr(model, "config", None) is not encoder.config:
        raise RuntimeError("Qwen model/config identity changed")
    if model.training or any(module.training for module in model.modules()):
        raise RuntimeError("resident Qwen model must remain in eval mode")
    if getattr(encoder.config, "use_cache", None) is not False:
        raise RuntimeError("resident Qwen config must keep use_cache=False")
    if str(getattr(encoder.config, "_attn_implementation", "")) != "eager":
        raise RuntimeError("resident Qwen provider requires the owned eager attention path")
    if (
        int(getattr(encoder.config, "num_hidden_layers", -1)) != int(encoder.layers)
        or len(model.layers) != int(encoder.layers)
    ):
        raise RuntimeError("Qwen retained-layer metadata disagrees with the live model")

    modules = tuple(model.named_modules())
    for _name, module in modules:
        _reject_hooks(module)
        if "forward" in getattr(module, "__dict__", {}):
            raise RuntimeError("owned Qwen modules cannot shadow forward")
    module_fingerprint = tuple(
        (
            name,
            id(module),
            type(module),
            id(type(module).forward),
            id(getattr(type(module).forward, "__code__", None)),
            _declared_runtime_fields(module, include_private=False),
        )
        for name, module in modules
    )
    parameters = tuple(model.named_parameters())
    if not parameters:
        raise RuntimeError("resident Qwen model has no parameters")
    parameter_devices = {str(parameter.device) for _name, parameter in parameters}
    parameter_dtypes = {str(parameter.dtype) for _name, parameter in parameters}
    if len(parameter_devices) != 1 or len(parameter_dtypes) != 1:
        raise RuntimeError("resident Qwen parameters have mixed device or dtype")
    device = _canonical_cuda_device(parameters[0][1].device, torch)
    if any(parameter.requires_grad for _name, parameter in parameters):
        raise RuntimeError("resident Qwen parameters must have gradients disabled")
    if any(getattr(parameter.device, "type", None) == "meta" for _, parameter in parameters):
        raise RuntimeError("resident Qwen parameters cannot remain on meta")
    if str(parameters[0][1].dtype) != str(encoder.dtype):
        raise RuntimeError("Qwen encoder dtype metadata disagrees with live parameters")
    if f"torch.{str(encoder.dtype_name)}" != str(parameters[0][1].dtype):
        raise RuntimeError("Qwen dtype_name disagrees with live parameters")
    encoder_device = torch.device(encoder.device)
    if encoder_device.type != "cuda":
        raise RuntimeError("resident Qwen encoder metadata must name CUDA")
    if encoder_device.index is None:
        if int(torch.cuda.current_device()) != int(device.split(":", 1)[1]):
            raise RuntimeError("unindexed encoder CUDA device is not the live parameter device")
    elif f"cuda:{encoder_device.index}" != device:
        raise RuntimeError("Qwen encoder device metadata disagrees with live parameters")
    parameter_fingerprint = tuple(
        (
            name,
            id(parameter),
            int(parameter.data_ptr()),
            tuple(int(value) for value in parameter.shape),
            str(parameter.dtype),
            str(parameter.device),
            bool(parameter.requires_grad),
            int(parameter._version),
        )
        for name, parameter in parameters
    )
    buffers = tuple(model.named_buffers())
    if any(getattr(buffer.device, "type", None) != "cuda" for _, buffer in buffers):
        raise RuntimeError("resident Qwen buffers must be CUDA resident")
    if any(_canonical_cuda_device(buffer.device, torch) != device for _, buffer in buffers):
        raise RuntimeError("resident Qwen buffers must share the parameter CUDA device")
    buffer_fingerprint = tuple(
        (
            name,
            id(buffer),
            int(buffer.data_ptr()),
            tuple(int(value) for value in buffer.shape),
            str(buffer.dtype),
            str(buffer.device),
            int(buffer._version),
        )
        for name, buffer in buffers
    )
    tokenizer_state = (
        id(encoder.tokenizer),
        type(encoder.tokenizer),
        id(type(encoder.tokenizer).__call__),
        id(getattr(type(encoder.tokenizer).__call__, "__code__", None)),
        getattr(encoder.tokenizer, "padding_side", None),
        getattr(encoder.tokenizer, "truncation_side", None),
        getattr(encoder.tokenizer, "pad_token_id", None),
        getattr(encoder.tokenizer, "eos_token_id", None),
        getattr(encoder.tokenizer, "bos_token_id", None),
        int(len(encoder.tokenizer)),
        id(getattr(encoder.tokenizer, "backend_tokenizer", None)),
        id(getattr(encoder.tokenizer, "_tokenizer", None)),
    )
    checkpoint = encoder.checkpoint_identity
    checkpoint_state = (
        id(checkpoint),
        type(checkpoint),
        encoder.model_id,
        encoder.model_revision,
        encoder.checkpoint_sha256,
        checkpoint.model_id,
        checkpoint.model_revision,
        checkpoint.checkpoint_sha256,
        tuple(checkpoint.verified_files),
    )
    config_state = (
        id(encoder.config),
        type(encoder.config),
        int(encoder.config.hidden_size),
        int(encoder.config.num_hidden_layers),
        bool(encoder.config.use_cache),
        str(getattr(encoder.config, "_attn_implementation", "")),
        str(encoder.dtype_name),
        _declared_runtime_fields(encoder.config, include_private=True),
    )
    loaded_names = getattr(encoder, "loaded_parameter_names", None)
    if type(loaded_names) is not frozenset or any(type(name) is not str for name in loaded_names):
        raise RuntimeError("Qwen loaded parameter-name identity is malformed")
    known_parameter_names = {name for name, _parameter in parameters}
    if loaded_names != known_parameter_names - {"norm.weight"}:
        raise RuntimeError("Qwen loaded parameter names disagree with the owned loader")
    execution_dtype = next(iter(parameter_dtypes))
    hidden_dim = int(encoder.config.hidden_size)
    fingerprint = (
        id(encoder),
        id(model),
        module_fingerprint,
        parameter_fingerprint,
        buffer_fingerprint,
        tokenizer_state,
        checkpoint_state,
        config_state,
        tuple(sorted(loaded_names)),
        device,
        execution_dtype,
        hidden_dim,
        int(encoder.layers),
        id(primitive_function),
        id(getattr(primitive_function, "__code__", None)),
    )
    return _OwnedRuntimeSnapshot(fingerprint, device, execution_dtype, hidden_dim)


__all__ = [
    "_OwnedRuntimeSnapshot",
    "_reject_global_module_hooks",
    "_runtime_fingerprint",
]
