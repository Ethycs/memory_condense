"""Fixed-bank Pure Attention CAV routing over bounded evidence features.

For a fixed learned CAV bank ``C0`` with K rows and request features ``X``
with N rows, the router performs exactly two rectangular attention passes::

    C0_hat = row_normalize(C0)
    X_hat  = row_normalize(X)
    E      = softmax((C0_hat @ X_hat.T) / tau_extract, dim=N)   # [K, N]
    C1     = E @ X                                               # [K, D]
    C1_hat = row_normalize(C1)
    R      = softmax((X_hat @ C1_hat.T) / tau_reinject, dim=K) # [N, K]
    X1     = X + alpha * (R @ C1)                              # [N, D]

Only KxN and NxK attention matrices are constructed; there is no NxN
operation.  The CAV bank is artifact-derived and resident in the router.  A
runtime receipt binds it without retaining tensors.  Request-derived X, C1,
X1, E, and R are never cached or persisted by the router; the caller owns the
lifetime of the transient ``FixedCAVForward`` result.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from memory_condense.domain._discourse_identity import _sha256, identity_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.domain.sealed import SealedIdentity
from memory_condense.search.fusion.tensor_identity import canonical_float32_tensor


FIXED_CAV_ALGORITHM = "memory-condense-fixed-cav-two-pass-cosine-attention-v1"
FIXED_CAV_MAX_NODES = 64
FIXED_CAV_MAX_CAVS = 16
FIXED_CAV_MAX_HIDDEN_DIM = 4096
FIXED_CAV_MAX_ROUTE_CELLS = 1024

_DTYPES = frozenset(
    {"torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"}
)
_LAYER_KEY = re.compile(r"(?:^|\.)layer_(\d+)$")


def _require_stack() -> tuple[Any, Any]:
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError(
            "FixedCAVRouter requires PyTorch and safetensors from the dev runtime"
        ) from exc
    return torch, safe_open


def _exact_int(value: object, label: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        qualifier = "non-negative" if minimum == 0 else "positive"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return value


def _positive_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite positive number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{label} must be a finite positive number")
    return normalized


def _nonnegative_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite non-negative number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return normalized


def _strings(values: object, label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be a sequence")
    try:
        normalized = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{label} must be a sequence") from exc
    if not normalized:
        raise ValueError(f"{label} must be non-empty")
    if any(type(value) is not str or not value.strip() for value in normalized):
        raise TypeError(f"{label} must contain exact non-empty strings")
    return normalized


def _validate_layer_keys(keys: tuple[str, ...], layer: int) -> None:
    for key in keys:
        match = _LAYER_KEY.search(key)
        if match is None or int(match.group(1)) != layer:
            raise ValueError(
                f"CAV tensor key {key!r} is not aligned to declared layer {layer}"
            )


def _bank_identity_sha256(
    *,
    artifact_file_sha256s: tuple[str, ...],
    ordered_tensor_keys: tuple[str, ...],
    layer: int,
    num_cavs: int,
    hidden_dim: int,
    artifact_dtype: str,
) -> str:
    """Bind the ordered learned source bank independently of runtime knobs."""

    return identity_sha256(
        {
            "format": "memory-condense-fixed-cav-source-bank-v1",
            "artifact_file_sha256s": list(artifact_file_sha256s),
            "ordered_tensor_keys": list(ordered_tensor_keys),
            "layer": layer,
            "num_cavs": num_cavs,
            "hidden_dim": hidden_dim,
            "artifact_dtype": artifact_dtype,
        }
    )


@dataclass(frozen=True, slots=True)
class FixedCAVRuntimeReceipt(SealedIdentity):
    """Frozen, tensor-free identity for one fixed CAV bank and runtime."""

    _SEAL_FIELD = "runtime_sha256"
    _SEAL_MISMATCH = "fixed CAV runtime SHA-256 does not match its contents"

    artifact_file_sha256s: tuple[str, ...]
    ordered_tensor_keys: tuple[str, ...]
    layer: int
    num_cavs: int
    hidden_dim: int
    artifact_dtype: str
    execution_dtype: str
    device: str
    extraction_temperature: float
    reinjection_temperature: float
    alpha: float
    bank_identity_sha256: str
    normalized_cav_bank_sha256: str
    algorithm: str = FIXED_CAV_ALGORITHM
    max_nodes: int = FIXED_CAV_MAX_NODES
    max_cavs: int = FIXED_CAV_MAX_CAVS
    max_hidden_dim: int = FIXED_CAV_MAX_HIDDEN_DIM
    max_route_cells: int = FIXED_CAV_MAX_ROUTE_CELLS
    receipt_retained_tensor_bytes: int = 0
    runtime_sha256: str = ""

    def __post_init__(self) -> None:
        file_hashes = _strings(
            self.artifact_file_sha256s,
            "artifact_file_sha256s",
        )
        file_hashes = tuple(
            _sha256(value, "artifact_file_sha256s") for value in file_hashes
        )
        keys = _strings(self.ordered_tensor_keys, "ordered_tensor_keys")
        object.__setattr__(self, "artifact_file_sha256s", file_hashes)
        object.__setattr__(self, "ordered_tensor_keys", keys)

        layer = _exact_int(self.layer, "layer", minimum=0)
        num_cavs = _exact_int(self.num_cavs, "num_cavs", minimum=1)
        hidden_dim = _exact_int(self.hidden_dim, "hidden_dim", minimum=1)
        if len(file_hashes) != num_cavs or len(keys) != num_cavs:
            raise ValueError("ordered artifact selections disagree with num_cavs")
        if len(set(zip(file_hashes, keys, strict=True))) != num_cavs:
            raise ValueError("ordered CAV artifact selections must be unique")
        _validate_layer_keys(keys, layer)
        if num_cavs > FIXED_CAV_MAX_CAVS:
            raise ValueError("num_cavs exceeds the immutable K policy ceiling")
        if hidden_dim > FIXED_CAV_MAX_HIDDEN_DIM:
            raise ValueError("hidden_dim exceeds the immutable D policy ceiling")

        if type(self.artifact_dtype) is not str or self.artifact_dtype not in _DTYPES:
            raise ValueError("artifact_dtype is unsupported")
        if type(self.execution_dtype) is not str or self.execution_dtype not in _DTYPES:
            raise ValueError("execution_dtype is unsupported")
        if type(self.device) is not str or not self.device.strip():
            raise ValueError("device must be a non-empty string")
        object.__setattr__(
            self,
            "extraction_temperature",
            _positive_float(self.extraction_temperature, "extraction_temperature"),
        )
        object.__setattr__(
            self,
            "reinjection_temperature",
            _positive_float(self.reinjection_temperature, "reinjection_temperature"),
        )
        object.__setattr__(self, "alpha", _nonnegative_float(self.alpha, "alpha"))
        expected_bank_identity = _bank_identity_sha256(
            artifact_file_sha256s=file_hashes,
            ordered_tensor_keys=keys,
            layer=layer,
            num_cavs=num_cavs,
            hidden_dim=hidden_dim,
            artifact_dtype=self.artifact_dtype,
        )
        supplied_bank_identity = _sha256(
            self.bank_identity_sha256,
            "bank_identity_sha256",
        )
        object.__setattr__(self, "bank_identity_sha256", supplied_bank_identity)
        if supplied_bank_identity != expected_bank_identity:
            raise ValueError("bank_identity_sha256 does not match selected artifacts")
        object.__setattr__(
            self,
            "normalized_cav_bank_sha256",
            _sha256(
                self.normalized_cav_bank_sha256,
                "normalized_cav_bank_sha256",
            ),
        )
        if self.algorithm != FIXED_CAV_ALGORITHM:
            raise ValueError("fixed CAV algorithm identity changed")
        expected_policy = (
            FIXED_CAV_MAX_NODES,
            FIXED_CAV_MAX_CAVS,
            FIXED_CAV_MAX_HIDDEN_DIM,
            FIXED_CAV_MAX_ROUTE_CELLS,
        )
        actual_policy = (
            self.max_nodes,
            self.max_cavs,
            self.max_hidden_dim,
            self.max_route_cells,
        )
        if actual_policy != expected_policy:
            raise ValueError("fixed CAV policy ceilings are immutable")
        if (
            type(self.receipt_retained_tensor_bytes) is not int
            or self.receipt_retained_tensor_bytes != 0
        ):
            raise ValueError("fixed CAV receipt cannot retain tensors")
        self._seal()


@dataclass(frozen=True, slots=True)
class FixedCAVForward:
    """Transient request-derived tensors; callers must consume and release."""

    steered_nodes: Any
    extraction_attention: Any
    reinjection_attention: Any


def _normalize_selections(
    selections: Sequence[tuple[str | Path, str]],
) -> tuple[tuple[Path, str], ...]:
    if isinstance(selections, (str, bytes, bytearray)):
        raise TypeError("selections must be ordered (path, key) pairs")
    try:
        values = tuple(selections)
    except TypeError as exc:
        raise TypeError("selections must be ordered (path, key) pairs") from exc
    if not values:
        raise ValueError("at least one CAV selection is required")
    if len(values) > FIXED_CAV_MAX_CAVS:
        raise ValueError("CAV selection count exceeds the immutable K policy ceiling")

    normalized: list[tuple[Path, str]] = []
    for item in values:
        if type(item) is not tuple or len(item) != 2:
            raise TypeError("each CAV selection must be an exact (path, key) tuple")
        raw_path, key = item
        if not isinstance(raw_path, (str, Path)):
            raise TypeError("CAV selection path must be a string or Path")
        if type(key) is not str or not key.strip():
            raise TypeError("CAV selection key must be an exact non-empty string")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"CAV safetensors artifact is missing: {path}")
        normalized.append((path.resolve(), key))
    if len(set(normalized)) != len(normalized):
        raise ValueError("CAV selections must not contain duplicate path/key pairs")
    return tuple(normalized)


def _target_dtype(torch: Any, value: Any | None, fallback: Any) -> Any:
    if value is None:
        return fallback
    aliases = {
        "float16": torch.float16,
        "torch.float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "torch.bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "torch.float32": torch.float32,
        "float64": torch.float64,
        "torch.float64": torch.float64,
    }
    if value in aliases.values():
        return value
    if type(value) is str and value in aliases:
        return aliases[value]
    raise ValueError("dtype must be one of float16, bfloat16, float32, or float64")


class FixedCAVRouter:
    """Parameter-free inference router over one sealed fixed CAV bank."""

    __slots__ = ("_torch", "_cav_queries", "_receipt", "_bank_fingerprint")

    def __init__(
        self,
        *,
        torch: Any,
        normalized_cav_queries: Any,
        receipt: FixedCAVRuntimeReceipt,
    ) -> None:
        if type(receipt) is not FixedCAVRuntimeReceipt:
            raise TypeError("receipt must be an exact FixedCAVRuntimeReceipt")
        receipt._seal()
        if type(normalized_cav_queries) is not torch.Tensor:
            raise TypeError("normalized CAV bank must be an exact torch.Tensor")
        expected_shape = (receipt.num_cavs, receipt.hidden_dim)
        if (
            tuple(int(value) for value in normalized_cav_queries.shape)
            != expected_shape
        ):
            raise ValueError("normalized CAV bank shape disagrees with receipt")
        if str(normalized_cav_queries.dtype) != receipt.execution_dtype:
            raise ValueError("normalized CAV bank dtype disagrees with receipt")
        if str(normalized_cav_queries.device) != receipt.device:
            raise ValueError("normalized CAV bank device disagrees with receipt")
        if bool(normalized_cav_queries.requires_grad) or (
            normalized_cav_queries.grad_fn is not None
        ):
            raise RuntimeError("normalized CAV bank cannot retain autograd state")
        if not bool(torch.isfinite(normalized_cav_queries).all().item()):
            raise ValueError("normalized CAV bank contains non-finite values")
        row_norms = torch.linalg.vector_norm(normalized_cav_queries.float(), dim=1)
        if not bool(torch.allclose(row_norms, torch.ones_like(row_norms), atol=2e-3)):
            raise ValueError("normalized CAV bank rows must have unit norm")

        self._torch = torch
        self._cav_queries = normalized_cav_queries
        self._receipt = receipt
        self._bank_fingerprint = self._runtime_fingerprint()

    @classmethod
    def load(
        cls,
        selections: Sequence[tuple[str | Path, str]],
        *,
        layer: int,
        device: Any = "cpu",
        dtype: Any | None = None,
        extraction_temperature: float = 1.0,
        reinjection_temperature: float = 1.0,
        alpha: float = 1.0,
    ) -> "FixedCAVRouter":
        """Load ordered layer-aligned vectors from content-addressed artifacts."""

        torch, safe_open = _require_stack()
        layer = _exact_int(layer, "layer", minimum=0)
        selected = _normalize_selections(selections)
        keys = tuple(key for _path, key in selected)
        _validate_layer_keys(keys, layer)

        unique_paths = tuple(dict.fromkeys(path for path, _key in selected))
        file_hashes = {path: file_sha256(path) for path in unique_paths}
        vectors: list[Any] = []
        artifact_dtype: str | None = None
        hidden_dim: int | None = None
        for path, key in selected:
            with safe_open(path, framework="pt", device="cpu") as artifact:
                if key not in artifact.keys():
                    raise KeyError(f"missing CAV tensor {key!r} in {path}")
                vector = artifact.get_tensor(key)
            if type(vector) is not torch.Tensor:
                raise TypeError("safetensors returned a foreign tensor type")
            if vector.ndim != 1 or int(vector.shape[0]) < 1:
                raise ValueError(f"CAV tensor {key!r} must have shape [D]")
            width = int(vector.shape[0])
            if width > FIXED_CAV_MAX_HIDDEN_DIM:
                raise ValueError("CAV width exceeds the immutable D policy ceiling")
            if hidden_dim is None:
                hidden_dim = width
            elif width != hidden_dim:
                raise ValueError("selected CAV tensors must share one hidden width")
            source_dtype = str(vector.dtype)
            if source_dtype not in _DTYPES:
                raise ValueError("selected CAV tensor dtype is unsupported")
            if artifact_dtype is None:
                artifact_dtype = source_dtype
            elif source_dtype != artifact_dtype:
                raise ValueError("selected CAV tensors must share one artifact dtype")
            if not bool(torch.isfinite(vector).all().item()):
                raise ValueError(f"CAV tensor {key!r} contains non-finite values")
            if float(torch.linalg.vector_norm(vector.float()).item()) <= 0.0:
                raise ValueError(f"CAV tensor {key!r} must have non-zero norm")
            vectors.append(vector.detach().clone())

        for path in unique_paths:
            if file_sha256(path) != file_hashes[path]:
                raise RuntimeError(f"CAV artifact changed while loading: {path}")
        assert hidden_dim is not None and artifact_dtype is not None
        execution_dtype = _target_dtype(torch, dtype, vectors[0].dtype)
        target_device = torch.device(device)
        bank = None
        normalized = None
        try:
            with torch.no_grad():
                bank = torch.stack(vectors).to(
                    device=target_device,
                    dtype=execution_dtype,
                ).contiguous()
                if not bool(torch.isfinite(bank).all().item()):
                    raise ValueError("CAV bank became non-finite after runtime casting")
                norms = torch.linalg.vector_norm(bank, dim=1, keepdim=True)
                if bool((norms <= 0.0).any().item()):
                    raise ValueError("CAV bank has zero rows after runtime casting")
                normalized = (bank / norms).clone().detach().contiguous()
            normalized.requires_grad_(False)
            receipt = FixedCAVRuntimeReceipt(
                artifact_file_sha256s=tuple(
                    file_hashes[path] for path, _key in selected
                ),
                ordered_tensor_keys=keys,
                layer=layer,
                num_cavs=len(selected),
                hidden_dim=hidden_dim,
                artifact_dtype=artifact_dtype,
                execution_dtype=str(normalized.dtype),
                device=str(normalized.device),
                extraction_temperature=extraction_temperature,
                reinjection_temperature=reinjection_temperature,
                alpha=alpha,
                bank_identity_sha256=_bank_identity_sha256(
                    artifact_file_sha256s=tuple(
                        file_hashes[path] for path, _key in selected
                    ),
                    ordered_tensor_keys=keys,
                    layer=layer,
                    num_cavs=len(selected),
                    hidden_dim=hidden_dim,
                    artifact_dtype=artifact_dtype,
                ),
                normalized_cav_bank_sha256=canonical_float32_tensor(
                    normalized,
                    label="normalized fixed CAV bank",
                    retain_values=False,
                ).tensor_sha256,
            )
            return cls(
                torch=torch,
                normalized_cav_queries=normalized,
                receipt=receipt,
            )
        finally:
            vectors.clear()
            bank = None
            normalized = None

    @property
    def runtime_receipt(self) -> FixedCAVRuntimeReceipt:
        self._assert_runtime()
        return self._receipt

    @property
    def hidden_dim(self) -> int:
        return self._receipt.hidden_dim

    @property
    def max_atoms(self) -> int:
        return self._receipt.max_nodes

    @property
    def runtime_identity_sha256(self) -> str:
        return self.runtime_receipt.runtime_sha256

    @property
    def bank_identity_sha256(self) -> str:
        return self.runtime_receipt.bank_identity_sha256

    @property
    def num_cavs(self) -> int:
        return self._receipt.num_cavs

    @property
    def layer(self) -> int:
        return self._receipt.layer

    def _runtime_fingerprint(self) -> tuple[Any, ...]:
        bank = self._cav_queries
        self._receipt._seal()
        return (
            id(bank),
            int(bank.data_ptr()),
            int(bank._version),
            tuple(int(value) for value in bank.shape),
            tuple(int(value) for value in bank.stride()),
            int(bank.storage_offset()),
            str(bank.layout),
            str(bank.dtype),
            str(bank.device),
            bool(bank.requires_grad),
            bank.grad_fn is None,
            self._receipt.runtime_sha256,
        )

    def _assert_runtime(self) -> None:
        if self._runtime_fingerprint() != self._bank_fingerprint:
            raise RuntimeError("fixed CAV runtime changed after loading")

    def route_one(self, node_features: Any) -> FixedCAVForward:
        """Route one bounded [N,D] request without retaining request tensors."""

        torch = self._torch
        if type(node_features) is not torch.Tensor:
            raise TypeError("node_features must be an exact torch.Tensor")
        if node_features.ndim != 2:
            raise ValueError("node_features must have shape [N, D]")
        node_count, hidden_dim = (int(value) for value in node_features.shape)
        if node_count < 1 or hidden_dim < 1:
            raise ValueError("node_features must have a non-empty shape")
        if node_count > FIXED_CAV_MAX_NODES:
            raise MemoryError("node count exceeds the immutable N policy ceiling")
        if hidden_dim > FIXED_CAV_MAX_HIDDEN_DIM:
            raise MemoryError("hidden width exceeds the immutable D policy ceiling")
        if hidden_dim != self._receipt.hidden_dim:
            raise ValueError("node_features width disagrees with the fixed CAV bank")
        if self._receipt.num_cavs * node_count > FIXED_CAV_MAX_ROUTE_CELLS:
            raise MemoryError("K*N exceeds the immutable route-cell policy ceiling")
        if str(node_features.dtype) != self._receipt.execution_dtype:
            raise ValueError("node_features dtype disagrees with fixed CAV runtime")
        if str(node_features.device) != self._receipt.device:
            raise ValueError("node_features device disagrees with fixed CAV runtime")
        if bool(node_features.requires_grad) or node_features.grad_fn is not None:
            raise RuntimeError("node_features cannot retain an autograd graph")
        if not bool(torch.isfinite(node_features).all().item()):
            raise ValueError("node_features contains non-finite values")

        x_keys = None
        extraction = None
        concepts = None
        concept_keys = None
        reinjection = None
        update = None
        steered = None
        result = None
        try:
            self._assert_runtime()
            with torch.inference_mode():
                node_norms = torch.linalg.vector_norm(
                    node_features,
                    dim=1,
                    keepdim=True,
                )
                if bool((node_norms <= 0.0).any().item()):
                    raise ValueError("node_features rows must have non-zero norm")
                x_keys = node_features / node_norms
                extraction = torch.softmax(
                    (self._cav_queries @ x_keys.transpose(0, 1))
                    / self._receipt.extraction_temperature,
                    dim=1,
                )
                concepts = extraction @ node_features
                concept_norms = torch.linalg.vector_norm(
                    concepts,
                    dim=1,
                    keepdim=True,
                )
                if bool((concept_norms <= 0.0).any().item()):
                    raise ValueError("pooled C1 rows must have non-zero norm")
                concept_keys = concepts / concept_norms
                reinjection = torch.softmax(
                    (x_keys @ concept_keys.transpose(0, 1))
                    / self._receipt.reinjection_temperature,
                    dim=1,
                )
                update = reinjection @ concepts
                steered = node_features + self._receipt.alpha * update
            expected_shapes = (
                (steered, (node_count, hidden_dim), "steered nodes"),
                (
                    extraction,
                    (self._receipt.num_cavs, node_count),
                    "extraction attention",
                ),
                (
                    reinjection,
                    (node_count, self._receipt.num_cavs),
                    "reinjection attention",
                ),
            )
            for value, expected_shape, label in expected_shapes:
                if tuple(int(item) for item in value.shape) != expected_shape:
                    raise RuntimeError(f"{label} has the wrong bounded shape")
                if str(value.dtype) != self._receipt.execution_dtype:
                    raise RuntimeError(f"{label} changed runtime dtype")
                if str(value.device) != self._receipt.device:
                    raise RuntimeError(f"{label} changed runtime device")
                if bool(value.requires_grad) or value.grad_fn is not None:
                    raise RuntimeError(f"{label} retained an autograd graph")
                if not bool(torch.isfinite(value).all().item()):
                    raise ValueError(f"{label} contains non-finite values")
            self._assert_runtime()
            result = FixedCAVForward(
                steered_nodes=steered,
                extraction_attention=extraction,
                reinjection_attention=reinjection,
            )
            return result
        finally:
            node_features = None
            x_keys = None
            extraction = None
            concepts = None
            concept_keys = None
            reinjection = None
            update = None
            steered = None
            result = None


__all__ = [
    "FIXED_CAV_ALGORITHM",
    "FIXED_CAV_MAX_CAVS",
    "FIXED_CAV_MAX_HIDDEN_DIM",
    "FIXED_CAV_MAX_NODES",
    "FIXED_CAV_MAX_ROUTE_CELLS",
    "FixedCAVForward",
    "FixedCAVRouter",
    "FixedCAVRuntimeReceipt",
]
