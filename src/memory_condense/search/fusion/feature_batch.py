"""Transient node-feature batch with a sealed, text-free provenance receipt."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.search.fusion.models import (
    DeclaredFeatureExtractorIdentity,
    NodeFeatureReceipt,
)
from memory_condense.search.fusion.tensor_identity import (
    canonical_float32_tensor,
    tensor_shape,
)


@dataclass(frozen=True, slots=True)
class NodeFeatureBatch:
    """Caller-owned values paired with an exact declared provenance receipt."""

    values: Any = field(repr=False, compare=False)
    receipt: NodeFeatureReceipt

    def __post_init__(self) -> None:
        if type(self.receipt) is not NodeFeatureReceipt:
            raise TypeError("receipt must be NodeFeatureReceipt")

    @classmethod
    def create(
        cls,
        values: Any,
        *,
        ordered_atom_ids: Sequence[str],
        query: str,
        extractor: DeclaredFeatureExtractorIdentity,
    ) -> NodeFeatureBatch:
        if type(extractor) is not DeclaredFeatureExtractorIdentity:
            raise TypeError("extractor must be DeclaredFeatureExtractorIdentity")
        atom_ids = tuple(ordered_atom_ids)
        raw_shape = tensor_shape(values, label="node_features")
        if raw_shape != (len(atom_ids), extractor.hidden_dim):
            raise ValueError("node feature shape disagrees with atom IDs or extractor")
        canonical = canonical_float32_tensor(
            values,
            label="node_features",
            retain_values=False,
        )
        receipt = NodeFeatureReceipt(
            extractor=extractor,
            ordered_atom_ids=atom_ids,
            query_sha256=quote_sha256(query),
            tensor_sha256=canonical.tensor_sha256,
            tensor_shape=canonical.shape,
            tensor_dtype=canonical.dtype,
        )
        return cls(values=values, receipt=receipt)

    def detached_snapshot(self) -> Any:
        """Return an isolated same-device copy when the tensor API provides one."""

        value = self.values
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        clone = getattr(value, "clone", None)
        if callable(clone):
            return clone()
        canonical = canonical_float32_tensor(value, label="node_features")
        rows, columns = canonical.shape
        return tuple(
            canonical.flat_values[start : start + columns]
            for start in range(0, rows * columns, columns)
        )


__all__ = ["NodeFeatureBatch"]
