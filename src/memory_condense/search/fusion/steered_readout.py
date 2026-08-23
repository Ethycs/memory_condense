"""Fast, bounded matched readout for latent-steered evidence nodes.

The control and treatment arms differ only in the evidence matrix: the
control scores ``X`` against one query vector ``q`` while the treatment scores
``X1`` against that exact same ``q``.  The returned value is tensor-free and
retains only bounded scalar scores, atom order, dimensions, and identities of
those bounded outputs.  It does not own, release, or identify the caller's
input tensors; their lifetime and upstream provenance remain caller concerns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from memory_condense.domain._discourse_identity import _sha256, identity_sha256
from memory_condense.search.fusion.fixed_cav_router import FixedCAVForward
from memory_condense.search.fusion.latent_router import LatentRouterForward
from memory_condense.search.fusion.tensor_identity import (
    CANONICAL_TENSOR_DTYPE,
    canonical_float32_tensor,
)


_ALGORITHM = "memory-condense-matched-x-x1-cosine-readout-v1"
_POLICY_MAX_OUTPUT_ATOMS = 64
_POLICY_MAX_HIDDEN_DIM = 4096
_SOURCE_DTYPES = frozenset(
    {"torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"}
)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError(
            "matched_steered_readout requires the optional PyTorch runtime"
        ) from exc
    return torch


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _atom_ids(values: object, label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be a sequence of atom identifiers")
    try:
        normalized = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{label} must be a sequence") from exc
    if not normalized:
        raise ValueError(f"{label} must be non-empty")
    if any(type(value) is not str or not value.strip() for value in normalized):
        raise TypeError(f"{label} must contain exact non-empty strings")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must be unique")
    return normalized


def _scores(values: object, label: str) -> tuple[float, ...]:
    try:
        raw = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{label} must be a sequence") from exc
    normalized: list[float] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{label} must contain scalar numbers")
        score = float(value)
        if not math.isfinite(score) or not -1.0 <= score <= 1.0:
            raise ValueError(f"{label} must contain finite values inside [-1, 1]")
        normalized.append(score)
    return tuple(normalized)


def _score_sha256(scores: tuple[float, ...], label: str) -> str:
    return canonical_float32_tensor(scores, label=label).tensor_sha256


def _order_sha256(order: tuple[str, ...]) -> str:
    return identity_sha256(
        {
            "format": "memory-condense-steered-readout-order-v1",
            "atom_ids": list(order),
        }
    )


@dataclass(frozen=True, slots=True)
class MatchedSteeredReadout:
    """Immutable, tensor-free receipt for one matched X-versus-X1 readout."""

    original_atom_order: tuple[str, ...]
    base_scores: tuple[float, ...]
    treatment_scores: tuple[float, ...]
    base_order: tuple[str, ...]
    treatment_order: tuple[str, ...]
    atom_count: int
    hidden_dim: int
    max_output_atoms: int
    max_hidden_dim: int
    source_dtype: str
    base_scores_sha256: str
    treatment_scores_sha256: str
    base_order_sha256: str
    treatment_order_sha256: str
    canonical_dtype: str = CANONICAL_TENSOR_DTYPE
    result_retained_tensor_bytes: int = 0
    readout_sha256: str = ""

    def __post_init__(self) -> None:
        original = _atom_ids(self.original_atom_order, "original_atom_order")
        base_scores = _scores(self.base_scores, "base_scores")
        treatment_scores = _scores(self.treatment_scores, "treatment_scores")
        base_order = _atom_ids(self.base_order, "base_order")
        treatment_order = _atom_ids(self.treatment_order, "treatment_order")
        object.__setattr__(self, "original_atom_order", original)
        object.__setattr__(self, "base_scores", base_scores)
        object.__setattr__(self, "treatment_scores", treatment_scores)
        object.__setattr__(self, "base_order", base_order)
        object.__setattr__(self, "treatment_order", treatment_order)

        atom_count = _positive_int(self.atom_count, "atom_count")
        hidden_dim = _positive_int(self.hidden_dim, "hidden_dim")
        max_output_atoms = _positive_int(self.max_output_atoms, "max_output_atoms")
        max_hidden_dim = _positive_int(self.max_hidden_dim, "max_hidden_dim")
        if max_output_atoms > _POLICY_MAX_OUTPUT_ATOMS:
            raise ValueError("max_output_atoms exceeds the immutable policy ceiling")
        if max_hidden_dim > _POLICY_MAX_HIDDEN_DIM:
            raise ValueError("max_hidden_dim exceeds the immutable policy ceiling")
        if atom_count > max_output_atoms:
            raise ValueError("atom_count exceeds max_output_atoms")
        if hidden_dim > max_hidden_dim:
            raise ValueError("hidden_dim exceeds max_hidden_dim")
        if len(original) != atom_count:
            raise ValueError("original_atom_order disagrees with atom_count")
        if len(base_scores) != atom_count or len(treatment_scores) != atom_count:
            raise ValueError("score count disagrees with atom_count")
        if len(base_order) != atom_count or set(base_order) != set(original):
            raise ValueError("base_order must preserve the exact atom set")
        if len(treatment_order) != atom_count or set(treatment_order) != set(original):
            raise ValueError("treatment_order must preserve the exact atom set")
        if (
            type(self.source_dtype) is not str
            or self.source_dtype not in _SOURCE_DTYPES
        ):
            raise ValueError("source_dtype is not a supported floating-point dtype")
        if self.canonical_dtype != CANONICAL_TENSOR_DTYPE:
            raise ValueError("readout scores require canonical float32-le")
        if (
            type(self.result_retained_tensor_bytes) is not int
            or self.result_retained_tensor_bytes != 0
        ):
            raise ValueError("readout result cannot retain tensor bytes")

        for name in (
            "base_scores_sha256",
            "treatment_scores_sha256",
            "base_order_sha256",
            "treatment_order_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if self.base_scores_sha256 != _score_sha256(base_scores, "base_scores"):
            raise ValueError("base_scores_sha256 does not match base_scores")
        if self.treatment_scores_sha256 != _score_sha256(
            treatment_scores,
            "treatment_scores",
        ):
            raise ValueError(
                "treatment_scores_sha256 does not match treatment_scores"
            )
        if self.base_order_sha256 != _order_sha256(base_order):
            raise ValueError("base_order_sha256 does not match base_order")
        if self.treatment_order_sha256 != _order_sha256(treatment_order):
            raise ValueError("treatment_order_sha256 does not match treatment_order")

        expected_base_order = tuple(
            original[index]
            for index in sorted(
                range(atom_count),
                key=lambda index: (-base_scores[index], index),
            )
        )
        expected_treatment_order = tuple(
            original[index]
            for index in sorted(
                range(atom_count),
                key=lambda index: (-treatment_scores[index], index),
            )
        )
        if base_order != expected_base_order:
            raise ValueError("base_order does not match stable score order")
        if treatment_order != expected_treatment_order:
            raise ValueError("treatment_order does not match stable score order")

        expected_receipt = self._expected_sha256()
        if self.readout_sha256:
            if _sha256(self.readout_sha256, "readout_sha256") != expected_receipt:
                raise ValueError("readout_sha256 does not match readout contents")
        else:
            object.__setattr__(self, "readout_sha256", expected_receipt)

    def _expected_sha256(self) -> str:
        return identity_sha256(
            {
                "format": _ALGORITHM,
                "original_atom_order": list(self.original_atom_order),
                "base_scores": list(self.base_scores),
                "treatment_scores": list(self.treatment_scores),
                "base_order": list(self.base_order),
                "treatment_order": list(self.treatment_order),
                "atom_count": self.atom_count,
                "hidden_dim": self.hidden_dim,
                "max_output_atoms": self.max_output_atoms,
                "max_hidden_dim": self.max_hidden_dim,
                "source_dtype": self.source_dtype,
                "base_scores_sha256": self.base_scores_sha256,
                "treatment_scores_sha256": self.treatment_scores_sha256,
                "base_order_sha256": self.base_order_sha256,
                "treatment_order_sha256": self.treatment_order_sha256,
                "canonical_dtype": self.canonical_dtype,
                "result_retained_tensor_bytes": self.result_retained_tensor_bytes,
            }
        )


def _validate_tensor(
    value: Any,
    *,
    torch: Any,
    label: str,
    expected_shape: tuple[int, ...],
    expected_dtype: Any,
    expected_device: Any,
) -> None:
    if type(value) is not torch.Tensor:
        raise TypeError(f"{label} must be an exact torch.Tensor")
    if tuple(int(dimension) for dimension in value.shape) != expected_shape:
        raise ValueError(f"{label} has the wrong shape")
    if value.dtype != expected_dtype:
        raise ValueError(f"{label} changed source dtype")
    if value.device != expected_device:
        raise ValueError(f"{label} changed source device")
    if not bool(value.is_floating_point()):
        raise TypeError(f"{label} must use a floating-point dtype")
    if bool(value.requires_grad) or value.grad_fn is not None:
        raise RuntimeError(f"{label} retained an autograd graph")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{label} contains non-finite values")


def matched_steered_readout(
    *,
    atom_ids: Sequence[str],
    node_features: Any,
    query_vector: Any,
    routed: LatentRouterForward | FixedCAVForward,
    max_output_atoms: int = 64,
    max_hidden_dim: int = 4096,
) -> MatchedSteeredReadout:
    """Score X and routed X1 against the same q without retaining tensors.

    Stable descending order uses the original atom position as the sole
    tie-breaker.  Extraction and reinjection attention are deliberately not
    read: this readout isolates the effect of ``routed.steered_nodes``.  The
    caller continues to own X, X1, q, and ``routed`` and must release them at
    the appropriate execution seam.
    """

    torch = _require_torch()
    max_output_atoms = _positive_int(max_output_atoms, "max_output_atoms")
    max_hidden_dim = _positive_int(max_hidden_dim, "max_hidden_dim")
    if max_output_atoms > _POLICY_MAX_OUTPUT_ATOMS:
        raise ValueError("max_output_atoms exceeds the immutable policy ceiling")
    if max_hidden_dim > _POLICY_MAX_HIDDEN_DIM:
        raise ValueError("max_hidden_dim exceeds the immutable policy ceiling")
    original = _atom_ids(atom_ids, "atom_ids")
    if type(routed) not in {LatentRouterForward, FixedCAVForward}:
        raise TypeError(
            "routed must be an exact LatentRouterForward or FixedCAVForward"
        )
    if type(node_features) is not torch.Tensor:
        raise TypeError("node_features must be an exact torch.Tensor")
    if node_features.ndim != 2:
        raise ValueError("node_features must have shape [N, D]")
    atom_count, hidden_dim = (int(value) for value in node_features.shape)
    if atom_count < 1 or hidden_dim < 1:
        raise ValueError("node_features must have a non-empty shape")
    if atom_count != len(original):
        raise ValueError("atom_ids disagree with node_features rows")
    if atom_count > max_output_atoms:
        raise MemoryError("node count exceeds max_output_atoms")
    if hidden_dim > max_hidden_dim:
        raise MemoryError("hidden width exceeds max_hidden_dim")
    if not bool(node_features.is_floating_point()):
        raise TypeError("node_features must use a floating-point dtype")
    if str(node_features.dtype) not in _SOURCE_DTYPES:
        raise TypeError("node_features uses an unsupported floating-point dtype")

    steered = None
    base_score_tensor = None
    treatment_score_tensor = None
    base_for_score = None
    treatment_for_score = None
    query_for_score = None
    base_canonical = None
    treatment_canonical = None
    try:
        steered = routed.steered_nodes
        _validate_tensor(
            node_features,
            torch=torch,
            label="node_features",
            expected_shape=(atom_count, hidden_dim),
            expected_dtype=node_features.dtype,
            expected_device=node_features.device,
        )
        _validate_tensor(
            query_vector,
            torch=torch,
            label="query_vector",
            expected_shape=(hidden_dim,),
            expected_dtype=node_features.dtype,
            expected_device=node_features.device,
        )
        _validate_tensor(
            steered,
            torch=torch,
            label="routed.steered_nodes",
            expected_shape=(atom_count, hidden_dim),
            expected_dtype=node_features.dtype,
            expected_device=node_features.device,
        )

        compute_dtype = (
            torch.float64 if node_features.dtype == torch.float64 else torch.float32
        )
        with torch.inference_mode():
            query_for_score = query_vector.to(dtype=compute_dtype)
            base_for_score = node_features.to(dtype=compute_dtype)
            treatment_for_score = steered.to(dtype=compute_dtype)
            query_norm = torch.linalg.vector_norm(query_for_score)
            base_norms = torch.linalg.vector_norm(base_for_score, dim=1)
            treatment_norms = torch.linalg.vector_norm(treatment_for_score, dim=1)
            if float(query_norm.item()) <= 0.0:
                raise ValueError("query_vector must have non-zero norm")
            if bool((base_norms <= 0.0).any().item()):
                raise ValueError("node_features rows must have non-zero norm")
            if bool((treatment_norms <= 0.0).any().item()):
                raise ValueError("routed.steered_nodes rows must have non-zero norm")
            base_score_tensor = torch.nn.functional.cosine_similarity(
                base_for_score,
                query_for_score.unsqueeze(0),
                dim=1,
                eps=0.0,
            ).clamp(-1.0, 1.0)
            treatment_score_tensor = torch.nn.functional.cosine_similarity(
                treatment_for_score,
                query_for_score.unsqueeze(0),
                dim=1,
                eps=0.0,
            ).clamp(-1.0, 1.0)
        if not bool(torch.isfinite(base_score_tensor).all().item()) or not bool(
            torch.isfinite(treatment_score_tensor).all().item()
        ):
            raise ValueError("cosine readout produced non-finite values")

        base_canonical = canonical_float32_tensor(
            base_score_tensor,
            label="base cosine scores",
        )
        treatment_canonical = canonical_float32_tensor(
            treatment_score_tensor,
            label="treatment cosine scores",
        )
        base_scores = base_canonical.flat_values
        treatment_scores = treatment_canonical.flat_values
        base_order = tuple(
            original[index]
            for index in sorted(
                range(atom_count),
                key=lambda index: (-base_scores[index], index),
            )
        )
        treatment_order = tuple(
            original[index]
            for index in sorted(
                range(atom_count),
                key=lambda index: (-treatment_scores[index], index),
            )
        )

        return MatchedSteeredReadout(
            original_atom_order=original,
            base_scores=base_scores,
            treatment_scores=treatment_scores,
            base_order=base_order,
            treatment_order=treatment_order,
            atom_count=atom_count,
            hidden_dim=hidden_dim,
            max_output_atoms=max_output_atoms,
            max_hidden_dim=max_hidden_dim,
            source_dtype=str(node_features.dtype),
            base_scores_sha256=base_canonical.tensor_sha256,
            treatment_scores_sha256=treatment_canonical.tensor_sha256,
            base_order_sha256=_order_sha256(base_order),
            treatment_order_sha256=_order_sha256(treatment_order),
        )
    finally:
        routed = None
        node_features = None
        query_vector = None
        steered = None
        base_score_tensor = None
        treatment_score_tensor = None
        base_for_score = None
        treatment_for_score = None
        query_for_score = None
        base_canonical = None
        treatment_canonical = None


__all__ = ["MatchedSteeredReadout", "matched_steered_readout"]
