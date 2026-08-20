"""Canonical identities for transient bounded tensors.

Every digest covers an explicit shape header and contiguous little-endian
float32 bytes.  The returned flat values are a short-lived planner aid; sealed
fusion plans retain only the digest, shape, dtype, and bounded scalar routes.
"""

from __future__ import annotations

import hashlib
import math
import struct
import sys
from dataclasses import dataclass
from typing import Any, Mapping

from memory_condense.domain._discourse_identity import canonical_json, identity_sha256


CANONICAL_TENSOR_DTYPE = "float32-le"


@dataclass(frozen=True, slots=True)
class CanonicalTensor:
    """Transient canonicalization result; never embed this in a sealed plan."""

    shape: tuple[int, ...]
    flat_values: tuple[float, ...]
    tensor_sha256: str
    dtype: str = CANONICAL_TENSOR_DTYPE


def _nested_shape(value: Any, label: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a numeric tensor")
    if not isinstance(value, (tuple, list)):
        return ()
    if not value:
        return (0,)
    child_shapes = tuple(_nested_shape(child, label) for child in value)
    if len(set(child_shapes)) != 1:
        raise ValueError(f"{label} must be rectangular")
    return (len(value), *child_shapes[0])


def tensor_shape(value: Any, *, label: str) -> tuple[int, ...]:
    """Read a tensor-like shape without copying its numerical payload."""

    raw_shape = getattr(value, "shape", None)
    if raw_shape is not None:
        try:
            shape = tuple(int(dimension) for dimension in raw_shape)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label} has an invalid shape") from exc
        if any(dimension < 0 for dimension in shape):
            raise ValueError(f"{label} has an invalid shape")
        return shape
    return _nested_shape(value, label)


def _flatten(value: Any, output: list[Any]) -> None:
    if isinstance(value, (tuple, list)):
        for child in value:
            _flatten(child, output)
        return
    output.append(value)


def canonical_float32_tensor(
    value: Any,
    *,
    label: str,
    retain_values: bool = True,
) -> CanonicalTensor:
    """Copy one tensor-like value to shaped contiguous CPU float32 bytes.

    Torch tensors use a vectorized NumPy byte view.  ``retain_values=False``
    is the digest-only path for feature/residual tensors; bounded attention
    matrices opt in to flat Python values for sparse membership planning.
    """

    materialized = value
    detach = getattr(materialized, "detach", None)
    if callable(detach):
        materialized = detach()
    to_float = getattr(materialized, "float", None)
    if callable(to_float):
        materialized = to_float()
    to_cpu = getattr(materialized, "cpu", None)
    if callable(to_cpu):
        materialized = to_cpu()
    contiguous = getattr(materialized, "contiguous", None)
    if callable(contiguous):
        materialized = contiguous()
    shape = tensor_shape(materialized, label=label)
    if not shape or any(dimension < 1 for dimension in shape):
        raise ValueError(f"{label} must have a non-empty shape")

    to_numpy = getattr(materialized, "numpy", None)
    is_finite = getattr(materialized, "isfinite", None)
    if callable(to_numpy) and callable(is_finite):
        if not bool(is_finite().all().item()):
            raise ValueError(f"{label} must contain finite float32 values")
        array = to_numpy().astype("<f4", copy=False)
        encoded = array.tobytes(order="C")
        normalized = (
            tuple(float(value) for value in array.reshape(-1).tolist())
            if retain_values
            else ()
        )
        header = canonical_json(
            {"dtype": CANONICAL_TENSOR_DTYPE, "shape": list(shape)}
        ).encode("utf-8")
        digest = hashlib.sha256(header + b"\0" + encoded).hexdigest()
        return CanonicalTensor(shape, normalized, digest)

    to_list = getattr(materialized, "tolist", None)
    if callable(to_list):
        materialized = to_list()

    shape = tensor_shape(materialized, label=label)
    if not shape or any(dimension < 1 for dimension in shape):
        raise ValueError(f"{label} must have a non-empty shape")
    raw_values: list[Any] = []
    _flatten(materialized, raw_values)
    expected = math.prod(shape)
    if len(raw_values) != expected:
        raise ValueError(f"{label} values do not match its shape")

    encoded = bytearray()
    normalized: list[float] = []
    for raw in raw_values:
        try:
            value_float = float(raw)
            packed = struct.pack("<f", value_float)
            value_float32 = struct.unpack("<f", packed)[0]
        except (TypeError, ValueError, OverflowError, struct.error) as exc:
            raise ValueError(f"{label} must contain finite float32 values") from exc
        if not math.isfinite(value_float) or not math.isfinite(value_float32):
            raise ValueError(f"{label} must contain finite float32 values")
        encoded.extend(packed)
        normalized.append(value_float32)

    header = canonical_json(
        {"dtype": CANONICAL_TENSOR_DTYPE, "shape": list(shape)}
    ).encode("utf-8")
    digest = hashlib.sha256(header + b"\0" + bytes(encoded)).hexdigest()
    return CanonicalTensor(
        shape,
        tuple(normalized) if retain_values else (),
        digest,
    )


def canonical_state_dict_sha256(state: Mapping[str, Any]) -> str:
    """Hash an operational float32 projection of a parameter mapping.

    This is useful for numerical comparison but is not an exact checkpoint
    identity because it intentionally normalizes original parameter dtypes.
    """

    if not state:
        raise ValueError("router state_dict must be non-empty")
    entries = []
    for name in sorted(state):
        if not str(name).strip():
            raise ValueError("router state_dict names must be non-empty")
        tensor = canonical_float32_tensor(
            state[name],
            label=f"router parameter {name}",
            retain_values=False,
        )
        entries.append(
            {
                "name": str(name),
                "dtype": tensor.dtype,
                "shape": list(tensor.shape),
                "tensor_sha256": tensor.tensor_sha256,
            }
        )
    return identity_sha256(
        {
            "format": "memory-condense-latent-router-state-v1",
            "parameters": entries,
        }
    )


def exact_torch_state_dict_sha256(state: Mapping[str, Any], torch: Any) -> str:
    """Hash exact contiguous loaded tensor bytes, original dtype, shape, and name."""

    if not state:
        raise ValueError("router state_dict must be non-empty")
    entries = []
    for name in sorted(state):
        if not str(name).strip():
            raise ValueError("router state_dict names must be non-empty")
        tensor = state[name].detach().cpu().contiguous()
        shape = tuple(int(value) for value in tensor.shape)
        raw = tensor.view(torch.uint8).reshape(-1).numpy().tobytes(order="C")
        entries.append(
            {
                "name": str(name),
                "dtype": str(tensor.dtype),
                "shape": list(shape),
                "byte_order": sys.byteorder,
                "tensor_bytes_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return identity_sha256(
        {
            "format": "memory-condense-latent-router-loaded-state-v1",
            "parameters": entries,
        }
    )


__all__ = [
    "CANONICAL_TENSOR_DTYPE",
    "CanonicalTensor",
    "canonical_float32_tensor",
    "canonical_state_dict_sha256",
    "exact_torch_state_dict_sha256",
    "tensor_shape",
]
