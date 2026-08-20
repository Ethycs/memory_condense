"""Exact K-latent, two-pass cross-attention over retrieved evidence nodes.

The router is intentionally optional-PyTorch: importing the fusion package is
lightweight, while constructing this trainable experiment requires the torch
runtime.  Its two attention matrices are K x N for extraction and N x K for
reinjection.  It never constructs an N x N content-attention matrix.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from memory_condense.search.fusion.models import (
    RouterArchitectureReceipt,
    RouterStateReceipt,
    RouterTrainingStatus,
)
from memory_condense.search.fusion.tensor_identity import (
    canonical_state_dict_sha256,
    exact_torch_state_dict_sha256,
)


@dataclass(frozen=True, slots=True)
class LatentRouterForward:
    """Transient differentiable outputs from the exact two-pass router.

    These tensors are training/inference workspace, not a persistence model.
    ``build_evidence_fusion_plan`` immediately reduces them to shaped hashes
    and bounded scalar memberships.
    """

    steered_nodes: Any
    extraction_attention: Any
    reinjection_attention: Any


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError(
            "LatentEvidenceRouter requires the optional PyTorch runtime"
        ) from exc
    return torch, nn


class LatentEvidenceRouter:
    """Trainable facade around two standard multi-head cross-attention blocks.

    ``training_status`` is deliberately conservative.  Fresh parameters are
    ``untrained``.  A caller may mark loaded/trained weights only as
    ``trained_declared``; this core does not certify the training procedure.
    The current loaded parameters are independently bound by an operational
    float32 state identity.  This is not presented as a verified checkpoint
    file or training-procedure receipt.  The inference seal detects supported
    module mutation APIs and captured-reference tampering; arbitrary private
    reflection or process-global PyTorch monkeypatching is outside its claim.
    """

    _ALGORITHM = "memory-condense-k-latent-two-pass-cross-attention-v1"
    __slots__ = (
        "_module",
        "_torch",
        "_architecture_receipt",
        "_training_status",
        "_max_atoms",
        "_max_hidden_dim",
        "_max_route_cells",
        "_sealed_state_receipt",
        "_sealed_runtime_fingerprint",
        "_expected_runtime_structure",
    )

    def __init__(
        self,
        hidden_dim: int,
        *,
        num_latents: int = 16,
        num_heads: int = 4,
        training_status: RouterTrainingStatus = "untrained",
        max_atoms: int = 64,
        max_hidden_dim: int = 4096,
        max_route_cells: int = 1024,
    ) -> None:
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")
        if isinstance(num_latents, bool) or not isinstance(num_latents, int) or num_latents < 1:
            raise ValueError("num_latents must be a positive integer")
        if isinstance(num_heads, bool) or not isinstance(num_heads, int) or num_heads < 1:
            raise ValueError("num_heads must be a positive integer")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if training_status not in {"untrained", "trained_declared"}:
            raise ValueError("training_status must be untrained or trained_declared")
        if isinstance(max_atoms, bool) or not isinstance(max_atoms, int) or max_atoms < 1:
            raise ValueError("max_atoms must be a positive integer")
        if (
            isinstance(max_hidden_dim, bool)
            or not isinstance(max_hidden_dim, int)
            or max_hidden_dim < 1
        ):
            raise ValueError("max_hidden_dim must be a positive integer")
        if hidden_dim > max_hidden_dim:
            raise MemoryError("hidden_dim exceeds router max_hidden_dim")
        if (
            isinstance(max_route_cells, bool)
            or not isinstance(max_route_cells, int)
            or max_route_cells < 1
        ):
            raise ValueError("max_route_cells must be a positive integer")
        if num_latents * max_atoms > max_route_cells:
            raise ValueError("num_latents * max_atoms exceeds max_route_cells")

        torch, nn = _require_torch()

        class _TwoPassModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.latent_concepts = nn.Parameter(
                    torch.empty(num_latents, hidden_dim)
                )
                nn.init.normal_(self.latent_concepts, mean=0.0, std=0.02)
                self.extract_attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    batch_first=True,
                )
                self.reinject_attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    batch_first=True,
                )

            def forward(self, node_features: Any) -> LatentRouterForward:
                if getattr(node_features, "ndim", None) != 3:
                    raise ValueError("node_features must have shape [B, N, D]")
                batch, atom_count, width = tuple(int(v) for v in node_features.shape)
                if batch != 1 or atom_count < 1 or width != hidden_dim:
                    raise ValueError(
                        f"node_features must have shape [1, N, {hidden_dim}] "
                        "with N positive"
                    )
                if atom_count > max_atoms:
                    raise MemoryError("node count exceeds router max_atoms")
                if atom_count * num_latents > max_route_cells:
                    raise MemoryError("K*N exceeds router max_route_cells")
                latents = self.latent_concepts.unsqueeze(0).expand(batch, -1, -1)

                # Extraction: K latent queries read N evidence-node keys/values.
                concepts_updated, extraction = self.extract_attention(
                    query=latents,
                    key=node_features,
                    value=node_features,
                    need_weights=True,
                    average_attn_weights=True,
                )
                # Reinjection: N evidence-node queries read K updated latents.
                reinjected, reinjection = self.reinject_attention(
                    query=node_features,
                    key=concepts_updated,
                    value=concepts_updated,
                    need_weights=True,
                    average_attn_weights=True,
                )
                return LatentRouterForward(
                    steered_nodes=node_features + reinjected,
                    extraction_attention=extraction,
                    reinjection_attention=reinjection,
                )

        self._module = _TwoPassModule()
        self._torch = torch
        self._architecture_receipt = RouterArchitectureReceipt(
            hidden_dim=hidden_dim,
            num_latents=num_latents,
            num_heads=num_heads,
            parameter_count=sum(
                int(parameter.numel()) for parameter in self._module.parameters()
            ),
        )
        self._training_status = training_status
        self._max_atoms = max_atoms
        self._max_hidden_dim = max_hidden_dim
        self._max_route_cells = max_route_cells
        self._sealed_state_receipt: RouterStateReceipt | None = None
        self._sealed_runtime_fingerprint: tuple[Any, ...] = ()
        self._expected_runtime_structure = self._runtime_structure()

    @property
    def module(self) -> Any:
        """Expose the underlying ``nn.Module`` for optimizers and compilers."""

        if self._sealed_state_receipt is not None:
            raise RuntimeError("sealed inference router does not expose its module")
        return self._module

    @property
    def architecture_receipt(self) -> RouterArchitectureReceipt:
        return self._architecture_receipt

    @property
    def architecture_sha256(self) -> str:
        return self._architecture_receipt.architecture_sha256

    @property
    def hidden_dim(self) -> int:
        return self._architecture_receipt.hidden_dim

    @property
    def num_latents(self) -> int:
        return self._architecture_receipt.num_latents

    @property
    def num_heads(self) -> int:
        return self._architecture_receipt.num_heads

    @property
    def training_status(self) -> RouterTrainingStatus:
        return self._training_status

    @property
    def max_atoms(self) -> int:
        return self._max_atoms

    @property
    def max_hidden_dim(self) -> int:
        return self._max_hidden_dim

    @property
    def max_route_cells(self) -> int:
        return self._max_route_cells

    @property
    def state_sha256(self) -> str:
        """Return the cached operational identity of a sealed inference state."""

        return self.state_receipt.operational_float32_sha256

    @property
    def state_receipt(self) -> RouterStateReceipt:
        self._assert_inference_seal()
        assert self._sealed_state_receipt is not None
        return self._sealed_state_receipt

    @staticmethod
    def _reject_runtime_hooks(module: Any) -> None:
        hook_fields = (
            "_forward_pre_hooks",
            "_forward_hooks",
            "_backward_pre_hooks",
            "_backward_hooks",
        )
        if any(getattr(module, name, ()) for name in hook_fields):
            raise RuntimeError("sealed router modules cannot carry execution hooks")

    def _runtime_structure(self) -> tuple[Any, ...]:
        module = self._module
        extract = module.extract_attention
        reinject = module.reinject_attention
        modules = (module, extract, reinject)
        mha_config = tuple(
            (
                attention.embed_dim,
                attention.num_heads,
                attention.kdim,
                attention.vdim,
                attention.dropout,
                attention.batch_first,
                attention.add_zero_attn,
            )
            for attention in (extract, reinject)
        )
        return (
            tuple(
                (
                    id(child),
                    type(child),
                    id(type(child).forward),
                    id(getattr(type(child).forward, "__code__", None)),
                )
                for child in modules
            ),
            mha_config,
        )

    def _runtime_fingerprint(self) -> tuple[Any, ...]:
        module = self._module
        extract = module.extract_attention
        reinject = module.reinject_attention
        modules = (module, extract, reinject)
        for child in modules:
            self._reject_runtime_hooks(child)
            if "forward" in getattr(child, "__dict__", {}):
                raise RuntimeError("sealed router modules cannot shadow forward")
            if child.training:
                raise RuntimeError("sealed router modules must remain in eval mode")
        structure = self._runtime_structure()
        if structure != self._expected_runtime_structure:
            raise RuntimeError("sealed router runtime changed")
        parameters = tuple(
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
            for name, parameter in module.named_parameters()
        )
        buffers = tuple(
            (
                name,
                id(buffer),
                int(buffer.data_ptr()),
                tuple(int(value) for value in buffer.shape),
                str(buffer.dtype),
                str(buffer.device),
                int(buffer._version),
            )
            for name, buffer in module.named_buffers()
        )
        return (
            structure,
            parameters,
            buffers,
        )

    def _assert_inference_seal(self) -> None:
        if self._sealed_state_receipt is None:
            raise RuntimeError("router must be sealed for inference")
        if self._runtime_fingerprint() != self._sealed_runtime_fingerprint:
            raise RuntimeError("sealed router runtime changed")

    def seal_for_inference(
        self,
        *,
        device: Any | None = None,
        dtype: Any | None = None,
    ) -> LatentEvidenceRouter:
        """Isolate, eval, and hash state once for O(parameter-count) sealing.

        The deep copy retires every module/parameter/storage reference exposed
        by the pre-seal training API, including aliases whose raw-byte writes
        do not increment PyTorch version counters.  For large widths, pass the
        final ``device``/``dtype`` here so isolation happens before resident
        placement where possible.  No parameter rehash occurs per packet.
        """

        if self._sealed_state_receipt is not None:
            if device is not None or dtype is not None:
                raise RuntimeError("cannot move or cast a sealed inference router")
            self._assert_inference_seal()
            return self
        module = self._module
        for child in (
            module,
            module.extract_attention,
            module.reinject_attention,
        ):
            self._reject_runtime_hooks(child)
            if "forward" in getattr(child, "__dict__", {}):
                raise RuntimeError("sealed router modules cannot shadow forward")
        if self._runtime_structure() != self._expected_runtime_structure:
            raise RuntimeError("router runtime structure changed before inference seal")

        isolated = copy.deepcopy(module)
        if device is not None or dtype is not None:
            isolated.to(device=device, dtype=dtype)
        isolated.eval()
        self._module = isolated
        self._expected_runtime_structure = self._runtime_structure()
        state = self._module.state_dict()
        parameters = tuple(self._module.parameters())
        receipt = RouterStateReceipt(
            loaded_parameter_bytes_sha256=exact_torch_state_dict_sha256(
                state,
                self._torch,
            ),
            operational_float32_sha256=canonical_state_dict_sha256(state),
            parameter_count=sum(int(parameter.numel()) for parameter in parameters),
            parameter_dtypes=tuple(
                sorted({str(parameter.dtype) for parameter in parameters})
            ),
            training_status=self._training_status,
        )
        fingerprint = self._runtime_fingerprint()
        self._sealed_state_receipt = receipt
        self._sealed_runtime_fingerprint = fingerprint
        return self

    def forward(self, node_features: Any) -> LatentRouterForward:
        if self._sealed_state_receipt is not None:
            raise RuntimeError("sealed inference router only supports route_one")
        return self._module(node_features)

    def route_one(self, node_features: Any) -> LatentRouterForward:
        """Run one bounded packet and return transient two-dimensional tensors."""

        self._assert_inference_seal()
        if type(node_features) is not self._torch.Tensor:
            raise TypeError("node_features must be an exact torch.Tensor")
        if getattr(node_features, "ndim", None) != 2:
            raise ValueError("node_features must have shape [N, D]")
        with self._torch.inference_mode():
            routed = self._module(node_features.unsqueeze(0))
        result = LatentRouterForward(
            steered_nodes=routed.steered_nodes[0],
            extraction_attention=routed.extraction_attention[0],
            reinjection_attention=routed.reinjection_attention[0],
        )
        self._assert_inference_seal()
        return result

    def __call__(self, node_features: Any) -> LatentRouterForward:
        return self.forward(node_features)

    def parameters(self, *args: Any, **kwargs: Any) -> Any:
        if self._sealed_state_receipt is not None:
            raise RuntimeError("sealed inference router does not expose parameters")
        return self._module.parameters(*args, **kwargs)

    def named_parameters(self, *args: Any, **kwargs: Any) -> Any:
        if self._sealed_state_receipt is not None:
            raise RuntimeError("sealed inference router does not expose parameters")
        return self._module.named_parameters(*args, **kwargs)

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        if self._sealed_state_receipt is not None:
            raise RuntimeError("sealed inference router does not expose mutable state")
        return self._module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args: Any, **kwargs: Any) -> Any:
        if self._sealed_state_receipt is not None:
            raise RuntimeError("cannot load state into a sealed inference router")
        return self._module.load_state_dict(*args, **kwargs)

    def train(self, mode: bool = True) -> LatentEvidenceRouter:
        if mode and self._sealed_state_receipt is not None:
            raise RuntimeError("cannot train a sealed inference router")
        self._module.train(mode)
        return self

    def eval(self) -> LatentEvidenceRouter:
        self._module.eval()
        return self

    def to(self, *args: Any, **kwargs: Any) -> LatentEvidenceRouter:
        if self._sealed_state_receipt is not None:
            raise RuntimeError("cannot move or cast a sealed inference router")
        self._module.to(*args, **kwargs)
        return self


__all__ = ["LatentEvidenceRouter", "LatentRouterForward"]
