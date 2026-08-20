"""Private GPU-resident workspace execution for Qwen atom features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memory_condense.search.fusion.qwen_feature_models import (
    QwenAtomFeatureCaps,
    QwenAtomFeatureProviderReceipt,
)


class _QwenFeatureLease:
    """Internal single-consumer holder; it never returns its tensor directly."""

    __slots__ = ("_features", "_torch", "_consumed", "_closed")

    def __init__(self, features: Any, torch: Any) -> None:
        if type(features) is not torch.Tensor:
            raise TypeError("feature lease requires an exact torch.Tensor")
        self._features = features
        self._torch = torch
        self._consumed = False
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def consumed(self) -> bool:
        return self._consumed

    def _discard_once(self, consumer: _DiscardFeatures) -> None:
        if self._closed or self._consumed:
            raise RuntimeError("Qwen feature workspace is already consumed or closed")
        if type(consumer) is not _DiscardFeatures:
            raise TypeError("tranche-A workspace accepts only the exact discard consumer")
        self._features = None
        self._consumed = True
        self._closed = True

    def _close(self) -> None:
        self._features = None
        self._closed = True

    def __copy__(self) -> Any:
        raise TypeError("Qwen feature leases cannot be copied")

    def __deepcopy__(self, _memo: Any) -> Any:
        raise TypeError("Qwen feature leases cannot be deep-copied")

    def __reduce__(self) -> Any:
        raise TypeError("Qwen feature leases cannot be pickled")


class _DiscardFeatures:
    __slots__ = ()


@dataclass(frozen=True, slots=True)
class _FeatureExecutionDiagnostics:
    primary_forward_count: int
    batch_invariance_forward_count: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.primary_forward_count, bool)
            or not isinstance(self.primary_forward_count, int)
            or self.primary_forward_count < 1
            or isinstance(self.batch_invariance_forward_count, bool)
            or not isinstance(self.batch_invariance_forward_count, int)
            or self.batch_invariance_forward_count != 0
        ):
            raise ValueError("feature execution requires primary forwards and zero diagnostics")


def _validate_feature_output(
    value: Any,
    *,
    torch: Any,
    expected_rows: int,
    provider_receipt: QwenAtomFeatureProviderReceipt,
    label: str,
) -> None:
    if type(value) is not torch.Tensor:
        raise TypeError(f"{label} returned a foreign tensor type")
    if tuple(int(item) for item in value.shape) != (
        expected_rows,
        provider_receipt.hidden_dim,
    ):
        raise ValueError(f"{label} has the wrong shape")
    if str(value.device) != provider_receipt.device:
        raise ValueError(f"{label} left the provider CUDA device")
    if str(value.dtype) != provider_receipt.execution_dtype:
        raise ValueError(f"{label} changed execution dtype")
    if bool(value.requires_grad) or value.grad_fn is not None:
        raise RuntimeError(f"{label} retained an autograd graph")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{label} contains non-finite values")


def _execute_feature_batches(
    *,
    encoder: Any,
    torch: Any,
    output_layer: int,
    provider_receipt: QwenAtomFeatureProviderReceipt,
    feature_caps: QwenAtomFeatureCaps,
    rows: tuple[Any, ...],
    batches: tuple[tuple[int, int, int, int], ...],
    gate_token: Any,
    _lease_type: type = _QwenFeatureLease,
    _discard_type: type = _DiscardFeatures,
    _diagnostics_type: type = _FeatureExecutionDiagnostics,
    _validate_output: Any = _validate_feature_output,
) -> _FeatureExecutionDiagnostics:
    """Fill, consume, and unconditionally shed one ``[N,D]`` workspace."""

    feature_tensor = None
    lease: _QwenFeatureLease | None = None
    batch_features = None
    try:
        feature_caps._seal()
        provider_receipt._seal()
        feature_tensor = torch.empty(
            (len(rows), provider_receipt.hidden_dim),
            device=provider_receipt.device,
            dtype=encoder.dtype,
        )
        for start, count, _width, _workspace in batches:
            stop = start + count
            batch_features = encoder._encode_selected_layer_final_readout(
                tuple(row.token_ids for row in rows[start:stop]),
                layer=output_layer,
                _gate_token=gate_token,
            )
            _validate_output(
                batch_features,
                torch=torch,
                expected_rows=count,
                provider_receipt=provider_receipt,
                label="Qwen feature batch",
            )
            feature_tensor[start:stop].copy_(batch_features)
            batch_features = None
        if not bool(torch.isfinite(feature_tensor).all().item()):
            raise ValueError("Qwen atom feature tensor contains non-finite values")
        lease = _lease_type(feature_tensor, torch)
        feature_tensor = None
        lease._discard_once(_discard_type())
        if not lease.closed or not lease.consumed:
            raise RuntimeError("private Qwen feature workspace did not close")
        return _diagnostics_type(len(batches), 0)
    finally:
        batch_features = None
        if lease is not None:
            lease._close()
        lease = None
        feature_tensor = None
        rows = ()
        batches = ()
        gate_token = None


__all__ = [
    "_DiscardFeatures",
    "_FeatureExecutionDiagnostics",
    "_QwenFeatureLease",
    "_execute_feature_batches",
    "_validate_feature_output",
]
