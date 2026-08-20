from __future__ import annotations

import gc
import math
import weakref
from contextlib import nullcontext
from dataclasses import dataclass, fields, is_dataclass
from types import SimpleNamespace

import pytest

from memory_condense.domain._discourse_identity import canonical_json
from memory_condense.search.fusion import FusionCaps
from memory_condense.search.fusion.latent_router import (
    LatentEvidenceRouter,
    LatentRouterForward,
)
from memory_condense.search.fusion.models import (
    AuthoritativeHyperedge,
    FusionAtomRef,
    RouterArchitectureReceipt,
    RouterStateReceipt,
)
from memory_condense.search.fusion.planning_core import (
    _latent_memberships_and_groups,
    _topology_atom_groups,
)
from memory_condense.search.fusion.qwen_feature_models import (
    QwenAtomFeatureCaps,
    QwenAtomFeatureProviderReceipt,
)
from memory_condense.search.fusion.qwen_matched import (
    _assert_resident_implementation,
    build_qwen_matched_fusion_pair,
)
from memory_condense.search.fusion.resident_executor import (
    _execute_qwen_resident_route,
)
from memory_condense.search.fusion.resident_models import (
    ResidentEvidenceFusionPlan,
    ResidentRouterRuntimeReceipt,
    resident_matched_input_sha256,
)
from memory_condense.search.fusion.tensor_identity import canonical_float32_tensor
from memory_condense.modeling.qwen_prefix import (
    _qwen_prefix_execution_gate,
    _qwen_prefix_gate_state,
)
import memory_condense.search.fusion.planner as planner
import memory_condense.search.fusion.planning_core as planning_core
import memory_condense.search.fusion.qwen_matched as qwen_matched


def _caps(**overrides: object) -> FusionCaps:
    values = {
        "max_atoms": 2,
        "max_latents": 2,
        "max_hidden_dim": 4,
        "max_route_cells": 4,
        "max_topology_links": 1,
        "max_hyperedges": 1,
        "max_groups": 2,
        "max_group_atoms": 2,
        "max_latent_memberships_per_atom": 2,
    }
    values.update(overrides)
    return FusionCaps(**values)


def _feature_caps() -> QwenAtomFeatureCaps:
    return QwenAtomFeatureCaps(
        max_row_tokens=8,
        max_query_tail_tokens=4,
        max_rows_per_forward=2,
        max_workspace_tokens=16,
        max_evidence_characters=32,
        max_query_characters=32,
        batch_invariance_atol=0.0,
        batch_invariance_rtol=0.0,
    )


def _atoms() -> tuple[FusionAtomRef, FusionAtomRef]:
    return (
        FusionAtomRef(
            atom_id="atom:a",
            atom_identity_sha256="a" * 64,
            span_identity_sha256="b" * 64,
            quote_sha256="c" * 64,
        ),
        FusionAtomRef(
            atom_id="atom:b",
            atom_identity_sha256="d" * 64,
            span_identity_sha256="e" * 64,
            quote_sha256="f" * 64,
        ),
    )


def _hyperedges() -> tuple[AuthoritativeHyperedge, ...]:
    return (
        AuthoritativeHyperedge(
            bundle_id="bundle:ab",
            atom_ids=("atom:a", "atom:b"),
            obligation_ids=(),
        ),
    )


def _router_runtime(
    *,
    num_latents: int = 2,
    execution_dtype: str = "torch.float32",
) -> ResidentRouterRuntimeReceipt:
    parameter_count = 168
    architecture = RouterArchitectureReceipt(
        hidden_dim=4,
        num_latents=num_latents,
        num_heads=2,
        parameter_count=parameter_count,
    )
    state = RouterStateReceipt(
        loaded_parameter_bytes_sha256="2" * 64,
        operational_float32_sha256="3" * 64,
        parameter_count=parameter_count,
        parameter_dtypes=(execution_dtype,),
        training_status="untrained",
    )
    return ResidentRouterRuntimeReceipt(
        architecture=architecture,
        state=state,
        device="cuda:0",
        execution_dtype=execution_dtype,
        max_atoms=2,
        max_hidden_dim=4,
        max_route_cells=max(2, num_latents * 2),
    )


def _resident_plans(
    *,
    caps: FusionCaps | None = None,
    runtime: ResidentRouterRuntimeReceipt | None = None,
) -> tuple[ResidentEvidenceFusionPlan, ResidentEvidenceFusionPlan]:
    active_caps = caps or _caps()
    active_runtime = runtime or _router_runtime()
    atoms = _atoms()
    hyperedges = _hyperedges()
    atom_ids = tuple(atom.atom_id for atom in atoms)
    topology_groups, topology_order, degree = _topology_atom_groups(
        atom_ids, hyperedges, active_caps
    )
    extraction = canonical_float32_tensor(
        ((0.8, 0.2), (0.1, 0.9)), label="resident-plan extraction"
    )
    reinjection = canonical_float32_tensor(
        ((0.7, 0.3), (0.2, 0.8)), label="resident-plan reinjection"
    )
    memberships, latent_groups, latent_order = _latent_memberships_and_groups(
        atom_ids,
        extraction,
        reinjection,
        degree,
        active_caps,
        source_dtype=active_runtime.execution_dtype,
    )
    shared = {
        "feature_suboperation_sha256": "4" * 64,
        "matched_input_sha256": "5" * 64,
        "caps": active_caps,
        "atoms": atoms,
        "hyperedges": hyperedges,
    }
    control = ResidentEvidenceFusionPlan(
        mode="topology_only",
        memberships=(),
        groups=topology_groups,
        atom_order=topology_order,
        **shared,
    )
    treatment = ResidentEvidenceFusionPlan(
        mode="latent_router",
        memberships=memberships,
        groups=latent_groups,
        atom_order=latent_order,
        router_runtime=active_runtime,
        extraction_matrix_sha256=extraction.tensor_sha256,
        reinjection_matrix_sha256=reinjection.tensor_sha256,
        extraction_shape=list(extraction.shape),
        reinjection_shape=list(reinjection.shape),
        **shared,
    )
    return control, treatment


def _contains_tensor_or_scalar_subclass(value: object) -> bool:
    if hasattr(value, "shape") and hasattr(value, "device") and hasattr(value, "dtype"):
        return True
    if isinstance(value, (str, int, float, bool)) and type(value) not in {
        str,
        int,
        float,
        bool,
    }:
        return True
    if is_dataclass(value):
        return any(
            _contains_tensor_or_scalar_subclass(getattr(value, item.name))
            for item in fields(value)
        )
    if type(value) is tuple:
        return any(_contains_tensor_or_scalar_subclass(item) for item in value)
    return False


def test_resident_plans_are_deep_frozen_text_free_and_route_joined() -> None:
    control, treatment = _resident_plans()
    assert control.atoms == treatment.atoms
    assert control.hyperedges == treatment.hyperedges
    assert treatment.extraction_shape == (2, 2)
    assert treatment.reinjection_shape == (2, 2)
    assert treatment.plan_retained_request_tensor_bytes == 0
    assert not _contains_tensor_or_scalar_subclass(control)
    assert not _contains_tensor_or_scalar_subclass(treatment)
    encoded = canonical_json(
        {
            "control": control.identity_payload(),
            "treatment": treatment.identity_payload(),
        }
    )
    assert "raw evidence prose" not in encoded
    assert "raw retrieval question" not in encoded


def test_resident_plan_rejects_runtime_outside_its_own_caps() -> None:
    with pytest.raises(ValueError, match="latent count exceeds FusionCaps"):
        _resident_plans(caps=_caps(max_latents=1, max_route_cells=2))


def test_resident_models_reject_scalar_subclass_payloads() -> None:
    class TaggedInt(int):
        pass

    tagged = TaggedInt(2)
    tagged.raw_request_text = "SENSITIVE REQUEST TEXT"
    tagged.tensor = object()
    caps = _caps(max_atoms=tagged)
    with pytest.raises(TypeError, match="unsupported identity value"):
        _resident_plans(caps=caps)

    class TaggedDigest(str):
        pass

    runtime = _router_runtime()
    tagged_digest = TaggedDigest(runtime.runtime_sha256)
    tagged_digest.raw_request_text = "SENSITIVE REQUEST TEXT"
    with pytest.raises(TypeError, match="unsupported identity value"):
        ResidentRouterRuntimeReceipt(
            architecture=runtime.architecture,
            state=runtime.state,
            device=runtime.device,
            execution_dtype=runtime.execution_dtype,
            max_atoms=runtime.max_atoms,
            max_hidden_dim=runtime.max_hidden_dim,
            max_route_cells=runtime.max_route_cells,
            runtime_sha256=tagged_digest,
        )


def test_matched_input_identity_binds_feature_router_and_implementation() -> None:
    runtime = _router_runtime()
    base = resident_matched_input_sha256(
        feature_suboperation_sha256="4" * 64,
        router_runtime_sha256=runtime.runtime_sha256,
        implementation_sha256="5" * 64,
    )
    assert base != resident_matched_input_sha256(
        feature_suboperation_sha256="4" * 64,
        router_runtime_sha256=runtime.runtime_sha256,
        implementation_sha256="6" * 64,
    )


def test_bfloat16_route_normalization_uses_bound_dtype_policy() -> None:
    extraction = canonical_float32_tensor(
        ((0.501953125, 0.5),), label="bf16 extraction"
    )
    reinjection = canonical_float32_tensor(
        ((1.0,), (1.0,)), label="bf16 reinjection"
    )
    caps = _caps(max_latents=1, max_route_cells=2)
    args = (
        ("atom:a", "atom:b"),
        extraction,
        reinjection,
        {"atom:a": 0, "atom:b": 0},
        caps,
    )
    memberships, groups, order = _latent_memberships_and_groups(
        *args, source_dtype="torch.bfloat16"
    )
    assert len(memberships) == 2
    assert len(groups) == 1
    assert set(order) == {"atom:a", "atom:b"}
    with pytest.raises(ValueError, match="softmax-normalized"):
        _latent_memberships_and_groups(*args, source_dtype="torch.float32")


def test_planning_core_is_shared_and_caps_before_adjacency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert planner._topology_atom_groups is planning_core._topology_atom_groups
    assert planner._preflight_topology is planning_core._preflight_topology
    assert planner._latent_memberships_and_groups is planning_core._latent_memberships_and_groups
    caps = FusionCaps(max_topology_links=0)
    edge = AuthoritativeHyperedge(
        bundle_id="bundle:cap",
        atom_ids=("atom:a", "atom:b"),
        obligation_ids=(),
    )
    monkeypatch.setattr(
        planning_core,
        "_adjacency",
        lambda *_args: (_ for _ in ()).throw(AssertionError("adjacency allocated")),
    )
    with pytest.raises(MemoryError, match="co-memberships"):
        planning_core._topology_atom_groups(("atom:a", "atom:b"), (edge,), caps)


def test_shared_topology_golden_covers_ties_disconnected_atoms_and_chunking() -> None:
    atom_ids = ("a", "b", "c", "d", "e")
    hyperedges = (
        AuthoritativeHyperedge("ab", ("a", "b"), ()),
        AuthoritativeHyperedge("bc", ("b", "c"), ()),
    )
    caps = FusionCaps(
        max_atoms=5,
        max_topology_links=2,
        max_hyperedges=2,
        max_groups=5,
        max_group_atoms=2,
    )
    groups, atom_order, degree = _topology_atom_groups(
        atom_ids, hyperedges, caps
    )
    assert tuple(group.atom_ids for group in groups) == (
        ("b", "a"),
        ("c",),
        ("d",),
        ("e",),
    )
    assert atom_order == ("b", "a", "c", "d", "e")
    assert degree == {"a": 1, "b": 2, "c": 1, "d": 0, "e": 0}


def test_sealed_router_keeps_cpu_generic_and_rejects_resident_view() -> None:
    torch = pytest.importorskip("torch")
    router = LatentEvidenceRouter(
        4,
        num_latents=2,
        num_heads=2,
        max_atoms=2,
        max_hidden_dim=4,
        max_route_cells=4,
    ).seal_for_inference()
    assert all(not parameter.requires_grad for parameter in router._module.parameters())
    with pytest.raises(RuntimeError, match="indexed CUDA"):
        _ = router.resident_runtime_receipt
    routed = router.route_one(torch.randn(2, 4))
    assert tuple(routed.steered_nodes.shape) == (2, 4)
    assert tuple(routed.extraction_attention.shape) == (2, 2)
    assert tuple(routed.reinjection_attention.shape) == (2, 2)


def test_sealed_router_rejects_nested_and_global_hooks() -> None:
    torch = pytest.importorskip("torch")
    router = LatentEvidenceRouter(
        4,
        num_latents=2,
        num_heads=2,
        max_atoms=2,
        max_hidden_dim=4,
        max_route_cells=4,
    ).seal_for_inference()
    nested = router._module.extract_attention.out_proj.register_forward_hook(
        lambda _module, _inputs, output: output
    )
    try:
        with pytest.raises(RuntimeError, match="execution hooks"):
            router.route_one(torch.randn(2, 4))
    finally:
        nested.remove()
    global_hook = torch.nn.modules.module.register_module_forward_hook(
        lambda _module, _inputs, output: output
    )
    try:
        with pytest.raises(RuntimeError, match="global module hooks"):
            router.route_one(torch.randn(2, 4))
    finally:
        global_hook.remove()


class _Device:
    def __init__(self, value: str) -> None:
        self.value = value
        self.type, separator, index = value.partition(":")
        self.index = int(index) if separator else None

    def __str__(self) -> str:
        return self.value


class _Scalar:
    def __init__(self, value: bool) -> None:
        self.value = value

    def all(self) -> _Scalar:
        return self

    def item(self) -> bool:
        return self.value


_STRIDED = object()


class _Tensor:
    refs: list[tuple[str, weakref.ReferenceType[_Tensor]]] = []
    host_calls: list[tuple[str, str]] = []
    fail_host_role: str | None = None

    def __init__(
        self,
        values: list[float],
        shape: tuple[int, ...],
        *,
        role: str,
        device: str = "cuda:0",
        dtype: str = "torch.float32",
        requires_grad: bool = False,
        grad_fn: object | None = None,
        root: _Tensor | None = None,
        row_start: int = 0,
    ) -> None:
        self.values = values
        self.shape = shape
        self.role = role
        self.device = _Device(device)
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.layout = _STRIDED
        self.is_meta = False
        self._root = root or self
        self._row_start = row_start
        self._local_version = 0
        type(self).refs.append((role, weakref.ref(self)))

    @classmethod
    def reset(cls) -> None:
        cls.refs = []
        cls.host_calls = []
        cls.fail_host_role = None

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def _version(self) -> int:
        return self._root._local_version

    def data_ptr(self) -> int:
        return id(self._root.values)

    def stride(self) -> tuple[int, ...]:
        result = []
        product = 1
        for dimension in reversed(self.shape):
            result.append(product)
            product *= dimension
        return tuple(reversed(result))

    def storage_offset(self) -> int:
        if len(self._root.shape) < 2:
            return 0
        return self._row_start * math.prod(self._root.shape[1:])

    def is_contiguous(self) -> bool:
        return True

    def is_floating_point(self) -> bool:
        return True

    def __getitem__(self, item: object) -> _Tensor:
        row_width = math.prod(self.shape[1:])
        if isinstance(item, slice):
            start, stop, step = item.indices(self.shape[0])
            if step != 1:
                raise TypeError("fake tensor only supports contiguous row slices")
            return _Tensor(
                self.values[start * row_width : stop * row_width],
                (stop - start, *self.shape[1:]),
                role=self.role,
                device=str(self.device),
                dtype=self.dtype,
                root=self._root,
                row_start=self._row_start + start,
            )
        if item == 0:
            if self.shape[0] != 1:
                raise TypeError("fake integer indexing requires a leading singleton")
            return _Tensor(
                list(self.values),
                self.shape[1:],
                role=self.role,
                device=str(self.device),
                dtype=self.dtype,
                requires_grad=self.requires_grad,
                grad_fn=self.grad_fn,
            )
        raise TypeError("unsupported fake tensor index")

    def copy_(self, other: _Tensor) -> None:
        width = math.prod(self._root.shape[1:])
        start = self._row_start * width
        self._root.values[start : start + len(other.values)] = list(other.values)
        self._root._local_version += 1
        self.values = self._root.values[start : start + len(other.values)]

    def unsqueeze(self, dimension: int) -> _Tensor:
        if dimension != 0:
            raise ValueError("fake tensor only supports leading unsqueeze")
        return _Tensor(
            self.values,
            (1, *self.shape),
            role=self.role,
            device=str(self.device),
            dtype=self.dtype,
            root=self._root,
        )

    def detach(self) -> _Tensor:
        return self

    def float(self) -> _Tensor:
        return _Tensor(
            list(self.values),
            self.shape,
            role=self.role,
            device=str(self.device),
            dtype="torch.float32",
        )

    def cpu(self) -> _Tensor:
        type(self).host_calls.append((self.role, "cpu"))
        if self.role in {"feature", "batch", "steered"}:
            raise AssertionError("full [N,D] host transfer is forbidden")
        if type(self).fail_host_role == self.role:
            raise RuntimeError(f"injected {self.role} canonicalization failure")
        return _Tensor(
            list(self.values),
            self.shape,
            role=self.role,
            device="cpu",
            dtype=self.dtype,
        )

    def contiguous(self) -> _Tensor:
        return self

    def numpy(self):
        import numpy

        type(self).host_calls.append((self.role, "numpy"))
        return numpy.asarray(self.values, dtype="float32").reshape(self.shape)

    def tolist(self):
        type(self).host_calls.append((self.role, "tolist"))
        raise AssertionError("resident route canonicalization must use bounded bytes")

    def isfinite(self) -> _Scalar:
        return _Scalar(all(math.isfinite(value) for value in self.values))


class _FakeTorch:
    Tensor = _Tensor
    strided = _STRIDED
    nn = SimpleNamespace(
        modules=SimpleNamespace(
            module=SimpleNamespace(
                _global_forward_pre_hooks={},
                _global_forward_hooks={},
                _global_forward_hooks_always_called={},
                _global_forward_hooks_with_kwargs={},
                _global_forward_pre_hooks_with_kwargs={},
                _global_backward_pre_hooks={},
                _global_backward_hooks={},
            )
        )
    )

    @staticmethod
    def empty(shape, *, device, dtype):
        return _Tensor(
            [0.0] * math.prod(shape),
            tuple(shape),
            role="feature",
            device=str(device),
            dtype=str(dtype),
        )

    @staticmethod
    def isfinite(value: _Tensor) -> _Scalar:
        return value.isfinite()

    @staticmethod
    def inference_mode():
        return nullcontext()

    @staticmethod
    def autocast(*, device_type, enabled):
        assert device_type == "cuda"
        assert enabled is False
        return nullcontext()

    @staticmethod
    def device(value: object) -> _Device:
        return _Device(str(value))


class _Parameter:
    shape = (1,)
    dtype = "torch.float32"
    device = _Device("cuda:0")
    requires_grad = False
    is_meta = False
    _version = 0

    def data_ptr(self) -> int:
        return id(self)


class _FakeAttention:
    training = False
    embed_dim = 4
    num_heads = 2
    kdim = 4
    vdim = 4
    dropout = 0.0
    batch_first = True
    add_zero_attn = False

    def forward(self, *_args, **_kwargs):
        raise AssertionError("fake attention child is metadata only")


class _FakeRouterModule:
    training = False

    def __init__(self, *, fault: str | None = None) -> None:
        self.extract_attention = _FakeAttention()
        self.reinject_attention = _FakeAttention()
        self.parameter = _Parameter()
        self.fault = fault

    def named_modules(self):
        return (
            ("", self),
            ("extract_attention", self.extract_attention),
            ("reinject_attention", self.reinject_attention),
        )

    def named_parameters(self):
        return (("parameter", self.parameter),)

    def named_buffers(self):
        return ()

    def parameters(self):
        return (self.parameter,)

    def buffers(self):
        return ()

    def modules(self):
        return tuple(module for _name, module in self.named_modules())

    def __call__(self, node_features: _Tensor) -> LatentRouterForward:
        return self.forward(node_features)

    def forward(self, node_features: _Tensor) -> LatentRouterForward:
        try:
            if self.fault == "router":
                raise RuntimeError("injected router failure")
            _, atom_count, hidden_dim = node_features.shape
            steered_shape = (1, atom_count, hidden_dim)
            steered_device = (
                "cuda:1" if self.fault == "steered_device" else "cuda:0"
            )
            steered_dtype = (
                "torch.float16"
                if self.fault == "steered_dtype"
                else "torch.float32"
            )
            if self.fault == "steered_shape":
                steered_shape = (1, atom_count, hidden_dim + 1)
            steered_values = [1.0] * math.prod(steered_shape)
            if self.fault == "steered_nan":
                steered_values[0] = float("nan")
            steered = _Tensor(
                steered_values,
                steered_shape,
                role="steered",
                device=steered_device,
                dtype=steered_dtype,
                requires_grad=self.fault == "steered_grad",
            )
            extraction_values = [0.8, 0.2, 0.1, 0.9]
            if self.fault == "reduction":
                extraction_values = [0.2, 0.2, 0.1, 0.9]
            extraction_shape = (1, 2, atom_count)
            if self.fault == "extraction_shape":
                extraction_shape = (1, 2, atom_count + 1)
                extraction_values = [0.5] * math.prod(extraction_shape)
            extraction_device = (
                "cuda:1" if self.fault == "extraction_device" else "cuda:0"
            )
            extraction_dtype = (
                "torch.float16"
                if self.fault == "extraction_dtype"
                else "torch.float32"
            )
            if self.fault == "extraction_nan":
                extraction_values[0] = float("nan")
            extraction = _Tensor(
                extraction_values,
                extraction_shape,
                role="extraction",
                device=extraction_device,
                dtype=extraction_dtype,
                requires_grad=self.fault == "extraction_grad",
            )
            reinjection_shape = (1, atom_count, 2)
            if self.fault == "reinjection_shape":
                reinjection_shape = (1, atom_count, 3)
            reinjection_values = [0.5] * math.prod(reinjection_shape)
            reinjection = _Tensor(
                reinjection_values,
                reinjection_shape,
                role="reinjection",
                device=(
                    "cuda:1"
                    if self.fault == "reinjection_device"
                    else "cuda:0"
                ),
                dtype=(
                    "torch.float16"
                    if self.fault == "reinjection_dtype"
                    else "torch.float32"
                ),
                requires_grad=self.fault == "reinjection_grad",
            )
            if reinjection_shape == (1, atom_count, 2):
                reinjection.values[:] = [0.7, 0.3, 0.2, 0.8]
            if self.fault == "reinjection_nan":
                reinjection.values[0] = float("nan")
            if self.fault == "storage_mutation":
                node_features._root._local_version += 1
            return LatentRouterForward(steered, extraction, reinjection)
        finally:
            node_features = None


class _FakeEncoder:
    dtype = "torch.float32"

    def __init__(self, *, fault: str | None = None) -> None:
        self.fault = fault
        self.calls: list[tuple[tuple[int, ...], ...]] = []

    def _encode_selected_layer_final_readout(self, rows, *, layer, _gate_token):
        del layer, _gate_token
        exact_rows = tuple(tuple(row) for row in rows)
        self.calls.append(exact_rows)
        if self.fault == "qwen":
            raise RuntimeError("injected Qwen failure")
        tensor = _Tensor(
            [float(sum(row)) for row in exact_rows for _ in range(4)],
            (len(exact_rows), 4),
            role="batch",
        )
        if self.fault == "qwen_nan":
            tensor.values[0] = float("nan")
        return tensor


@dataclass(frozen=True)
class _Row:
    token_ids: tuple[int, ...]


def _provider_receipt(
    *, execution_dtype: str = "torch.float32"
) -> QwenAtomFeatureProviderReceipt:
    return QwenAtomFeatureProviderReceipt(
        implementation_sha256="4" * 64,
        model_id="Qwen3-8B",
        model_revision="main",
        checkpoint_sha256="5" * 64,
        verified_files_sha256="6" * 64,
        tokenizer_identity_sha256="7" * 64,
        retained_layers=2,
        output_layer=1,
        hidden_dim=4,
        device="cuda:0",
        execution_dtype=execution_dtype,
        prompt_template_sha256="8" * 64,
    )


def _fake_router(
    *, fault: str | None = None
) -> tuple[LatentEvidenceRouter, ResidentRouterRuntimeReceipt]:
    router = object.__new__(LatentEvidenceRouter)
    module = _FakeRouterModule(fault=fault)
    runtime = _router_runtime()
    router._module = module
    router._torch = _FakeTorch
    router._architecture_receipt = runtime.architecture
    router._training_status = "untrained"
    router._max_atoms = 2
    router._max_hidden_dim = 4
    router._max_route_cells = 4
    router._sealed_state_receipt = runtime.state
    router._sealed_runtime_fingerprint = ()
    router._expected_runtime_structure = router._runtime_structure()
    router._sealed_runtime_fingerprint = router._runtime_fingerprint(
        state_receipt=runtime.state
    )
    assert router.resident_runtime_receipt == runtime
    return router, runtime


def _run_private_executor(
    *,
    fault: str | None = None,
    encoder: _FakeEncoder | None = None,
    gate_token: object | None = None,
):
    _Tensor.reset()
    if fault in {"extraction", "reinjection"}:
        _Tensor.fail_host_role = fault
    qwen_faults = {"qwen", "qwen_nan"}
    canonical_faults = {"extraction", "reinjection"}
    router_fault = fault if fault not in qwen_faults | canonical_faults else None
    router, runtime = _fake_router(fault=router_fault)
    rows = (_Row((1, 2)), _Row((3, 4)))
    return _execute_qwen_resident_route(
        encoder=encoder or _FakeEncoder(fault=fault if fault in qwen_faults else None),
        torch=_FakeTorch,
        output_layer=1,
        provider_receipt=_provider_receipt(),
        feature_caps=_feature_caps(),
        rows=rows,
        batches=((0, 2, 2, 4),),
        gate_token=gate_token or object(),
        router=router,
        router_runtime=runtime,
        atom_ids=("atom:a", "atom:b"),
        topology_degree={"atom:a": 1, "atom:b": 1},
        caps=_caps(),
    )


def _assert_full_workspace_released() -> None:
    gc.collect()
    full_roles = {"feature", "batch", "steered"}
    assert not [
        role
        for role, reference in _Tensor.refs
        if role in full_roles and reference() is not None
    ]
    assert not [call for call in _Tensor.host_calls if call[0] in full_roles]


def test_private_resident_executor_routes_once_and_only_copies_bounded_matrices() -> None:
    execution = _run_private_executor()
    assert execution.primary_forward_count == 1
    assert execution.router_forward_count == 1
    assert execution.extraction_shape == (2, 2)
    assert execution.reinjection_shape == (2, 2)
    assert len(execution.memberships) == 4
    assert set(execution.atom_order) == {"atom:a", "atom:b"}
    assert _Tensor.host_calls == [
        ("extraction", "cpu"),
        ("extraction", "numpy"),
        ("reinjection", "cpu"),
        ("reinjection", "numpy"),
    ]
    _assert_full_workspace_released()


def test_cuda_bfloat16_executor_copies_only_bounded_route_matrices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the resident integration canary")
    device = torch.device("cuda:0")
    router = LatentEvidenceRouter(
        4,
        num_latents=2,
        num_heads=2,
        max_atoms=2,
        max_hidden_dim=4,
        max_route_cells=4,
    ).seal_for_inference(device=device, dtype=torch.bfloat16)
    runtime = router.resident_runtime_receipt

    class Encoder:
        dtype = torch.bfloat16

        @staticmethod
        def _encode_selected_layer_final_readout(rows, *, layer, _gate_token):
            del layer, _gate_token
            return torch.tensor(
                [[float(sum(row)), 1.0, 2.0, 3.0] for row in rows],
                device=device,
                dtype=torch.bfloat16,
            )

    original_cpu = torch.Tensor.cpu
    copied_shapes: list[tuple[int, ...]] = []

    def cpu_spy(tensor, *args, **kwargs):
        copied_shapes.append(tuple(int(value) for value in tensor.shape))
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu_spy)
    execution = _execute_qwen_resident_route(
        encoder=Encoder(),
        torch=torch,
        output_layer=1,
        provider_receipt=_provider_receipt(
            execution_dtype="torch.bfloat16"
        ),
        feature_caps=_feature_caps(),
        rows=(_Row((1, 2)), _Row((3, 4))),
        batches=((0, 2, 2, 4),),
        gate_token=object(),
        router=router,
        router_runtime=runtime,
        atom_ids=("atom:a", "atom:b"),
        topology_degree={"atom:a": 1, "atom:b": 1},
        caps=_caps(),
    )
    assert execution.extraction_shape == (2, 2)
    assert execution.reinjection_shape == (2, 2)
    assert copied_shapes == [(2, 2), (2, 2)]


@pytest.mark.parametrize(
    "fault,match",
    [
        ("qwen", "injected Qwen failure"),
        ("qwen_nan", "non-finite"),
        ("router", "injected router failure"),
        ("steered_shape", "wrong shape"),
        ("steered_device", "left the resident router device"),
        ("steered_dtype", "changed resident router dtype"),
        ("steered_nan", "non-finite"),
        ("steered_grad", "autograd graph"),
        ("extraction_shape", "wrong shape"),
        ("extraction_device", "left the resident router device"),
        ("extraction_dtype", "changed resident router dtype"),
        ("extraction_nan", "non-finite"),
        ("extraction_grad", "autograd graph"),
        ("reinjection_shape", "wrong shape"),
        ("reinjection_device", "left the resident router device"),
        ("reinjection_dtype", "changed resident router dtype"),
        ("reinjection_nan", "non-finite"),
        ("reinjection_grad", "autograd graph"),
        ("storage_mutation", "mutated or replaced feature storage"),
        ("extraction", "canonicalization failure"),
        ("reinjection", "canonicalization failure"),
        ("reduction", "softmax-normalized"),
    ],
)
def test_private_resident_executor_rejects_and_sheds_failures(
    fault: str,
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError, RuntimeError), match=match) as caught:
        _run_private_executor(fault=fault)
    caught.value.__traceback__ = None
    del caught
    _assert_full_workspace_released()


def test_private_executor_failure_releases_and_reacquires_provider_gate() -> None:
    encoder = _FakeEncoder(fault="qwen_nan")
    with pytest.raises(ValueError, match="non-finite"):
        with _qwen_prefix_execution_gate(encoder) as token:
            _run_private_executor(
                fault="qwen_nan",
                encoder=encoder,
                gate_token=token,
            )
    state = _qwen_prefix_gate_state(encoder)
    assert state.active_token is None
    assert not state.lock.locked()
    with _qwen_prefix_execution_gate(encoder):
        pass


def test_public_builder_rejects_foreign_values_and_owned_class_shadowing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError, match="provider"):
        build_qwen_matched_fusion_pair(
            None,
            None,
            provider=object(),
            router=object(),
            caps=_caps(),
            feature_caps=_feature_caps(),
        )
    monkeypatch.setattr(
        ResidentEvidenceFusionPlan,
        "_validate_mode",
        lambda self: None,
    )
    with pytest.raises(RuntimeError, match="resident fusion .* replaced"):
        _assert_resident_implementation()


def test_public_builder_rejects_stable_router_dependency_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qwen_matched.latent_router_module, "_require_torch", lambda: ())
    with pytest.raises(RuntimeError, match="dependency seams"):
        build_qwen_matched_fusion_pair(
            None,
            None,
            provider=object(),
            router=object(),
            caps=_caps(),
            feature_caps=_feature_caps(),
        )


@pytest.mark.parametrize(
    "attribute,replacement",
    [
        ("_runtime_fingerprint", lambda self, **_kwargs: ()),
        ("state_receipt", property(lambda self: None)),
    ],
)
def test_resident_implementation_rejects_router_class_shadowing(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    replacement: object,
) -> None:
    monkeypatch.setattr(LatentEvidenceRouter, attribute, replacement)
    with pytest.raises(RuntimeError, match="resident fusion .* replaced"):
        _assert_resident_implementation()


def test_resident_implementation_rejects_matched_hash_helper_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qwen_matched,
        "_matched_input_sha256",
        lambda **_kwargs: "0" * 64,
    )
    with pytest.raises(RuntimeError, match="implementation was replaced"):
        _assert_resident_implementation()
