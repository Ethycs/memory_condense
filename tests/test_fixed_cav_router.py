from __future__ import annotations

import gc
import weakref
from dataclasses import fields, is_dataclass, replace

import pytest

from memory_condense.domain.integrity import file_sha256
from memory_condense.search.fusion.fixed_cav_router import (
    FIXED_CAV_ALGORITHM,
    FixedCAVForward,
    FixedCAVRouter,
    FixedCAVRuntimeReceipt,
)


torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")


def _artifacts(tmp_path):
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    safetensors_torch.save_file(
        {"concept_a.layer_2": torch.tensor([1.0, 0.0, 0.0])},
        first,
    )
    safetensors_torch.save_file(
        {"concept_b.layer_2": torch.tensor([0.0, 1.0, 0.0])},
        second,
    )
    return first, second


def _router(tmp_path, **updates) -> FixedCAVRouter:
    first, second = _artifacts(tmp_path)
    values = {
        "selections": (
            (first, "concept_a.layer_2"),
            (second, "concept_b.layer_2"),
        ),
        "layer": 2,
        "device": "cpu",
        "dtype": "float32",
        "extraction_temperature": 0.7,
        "reinjection_temperature": 1.3,
        "alpha": 0.5,
    }
    values.update(updates)
    return FixedCAVRouter.load(**values)


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if is_dataclass(value):
        return any(
            _contains_tensor(getattr(value, item.name)) for item in fields(value)
        )
    if isinstance(value, (tuple, list, dict)):
        children = value.values() if isinstance(value, dict) else value
        return any(_contains_tensor(item) for item in children)
    return False


def test_loader_binds_ordered_artifacts_keys_and_tensor_free_runtime(tmp_path) -> None:
    first, second = _artifacts(tmp_path)
    router = FixedCAVRouter.load(
        (
            (first, "concept_a.layer_2"),
            (second, "concept_b.layer_2"),
        ),
        layer=2,
        extraction_temperature=0.7,
        reinjection_temperature=1.3,
        alpha=0.5,
    )
    receipt = router.runtime_receipt

    assert type(receipt) is FixedCAVRuntimeReceipt
    assert receipt.artifact_file_sha256s == (
        file_sha256(first),
        file_sha256(second),
    )
    assert receipt.ordered_tensor_keys == (
        "concept_a.layer_2",
        "concept_b.layer_2",
    )
    assert (receipt.layer, receipt.num_cavs, receipt.hidden_dim) == (2, 2, 3)
    assert receipt.artifact_dtype == receipt.execution_dtype == "torch.float32"
    assert receipt.algorithm == FIXED_CAV_ALGORITHM
    assert receipt.max_nodes == 64
    assert receipt.max_cavs == 16
    assert receipt.max_hidden_dim == 4096
    assert receipt.max_route_cells == 1024
    assert receipt.receipt_retained_tensor_bytes == 0
    assert not _contains_tensor(receipt)
    assert len(receipt.runtime_sha256) == 64
    assert router.runtime_identity_sha256 == receipt.runtime_sha256
    assert router.bank_identity_sha256 == receipt.bank_identity_sha256
    assert router.max_atoms == receipt.max_nodes == 64

    repeated = FixedCAVRouter.load(
        (
            (first, "concept_a.layer_2"),
            (second, "concept_b.layer_2"),
        ),
        layer=2,
        extraction_temperature=0.7,
        reinjection_temperature=1.3,
        alpha=0.5,
    )
    reversed_bank = FixedCAVRouter.load(
        (
            (second, "concept_b.layer_2"),
            (first, "concept_a.layer_2"),
        ),
        layer=2,
        extraction_temperature=0.7,
        reinjection_temperature=1.3,
        alpha=0.5,
    )
    assert repeated.runtime_receipt == receipt
    assert reversed_bank.runtime_receipt.runtime_sha256 != receipt.runtime_sha256
    with pytest.raises(ValueError, match="runtime SHA-256"):
        replace(receipt, alpha=0.75)


def test_route_matches_exact_two_pass_cosine_equations_without_nn(tmp_path) -> None:
    router = _router(tmp_path)
    nodes = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
        dtype=torch.float32,
    )
    routed = router.route_one(nodes)
    repeated = router.route_one(nodes)

    c0 = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    c0_hat = torch.nn.functional.normalize(c0, dim=1)
    x_hat = torch.nn.functional.normalize(nodes, dim=1)
    expected_e = torch.softmax((c0_hat @ x_hat.T) / 0.7, dim=1)
    c1 = expected_e @ nodes
    c1_hat = torch.nn.functional.normalize(c1, dim=1)
    expected_r = torch.softmax((x_hat @ c1_hat.T) / 1.3, dim=1)
    expected_x1 = nodes + 0.5 * (expected_r @ c1)

    assert type(routed) is FixedCAVForward
    assert tuple(routed.extraction_attention.shape) == (2, 3)
    assert tuple(routed.reinjection_attention.shape) == (3, 2)
    assert tuple(routed.steered_nodes.shape) == (3, 3)
    assert torch.allclose(routed.extraction_attention, expected_e)
    assert torch.allclose(routed.reinjection_attention, expected_r)
    assert torch.allclose(routed.steered_nodes, expected_x1)
    assert torch.equal(routed.extraction_attention, repeated.extraction_attention)
    assert torch.equal(routed.reinjection_attention, repeated.reinjection_attention)
    assert torch.equal(routed.steered_nodes, repeated.steered_nodes)
    assert torch.allclose(routed.extraction_attention.sum(dim=1), torch.ones(2))
    assert torch.allclose(routed.reinjection_attention.sum(dim=1), torch.ones(3))
    assert all(
        not value.requires_grad and value.grad_fn is None
        for value in (
            routed.steered_nodes,
            routed.extraction_attention,
            routed.reinjection_attention,
        )
    )


def test_router_retains_no_request_tensor_and_rejects_autograd(tmp_path) -> None:
    router = _router(tmp_path)
    nodes = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    reference = weakref.ref(nodes)
    routed = router.route_one(nodes)
    del nodes
    gc.collect()

    assert reference() is None
    assert tuple(routed.steered_nodes.shape) == (2, 3)
    with pytest.raises(RuntimeError, match="autograd"):
        router.route_one(torch.ones(2, 3, requires_grad=True))


def test_loader_and_route_fail_closed_on_alignment_shapes_and_policy(tmp_path) -> None:
    first, second = _artifacts(tmp_path)
    with pytest.raises(ValueError, match="aligned"):
        FixedCAVRouter.load(
            ((first, "concept_a.layer_2"),),
            layer=1,
        )
    with pytest.raises(ValueError, match="duplicate"):
        FixedCAVRouter.load(
            (
                (first, "concept_a.layer_2"),
                (first, "concept_a.layer_2"),
            ),
            layer=2,
        )

    too_wide = tmp_path / "too-wide.safetensors"
    safetensors_torch.save_file(
        {"wide.layer_2": torch.ones(4097)},
        too_wide,
    )
    with pytest.raises(ValueError, match="D policy ceiling"):
        FixedCAVRouter.load(((too_wide, "wide.layer_2"),), layer=2)

    many = tmp_path / "many.safetensors"
    safetensors_torch.save_file(
        {f"concept_{index}.layer_2": torch.ones(3) for index in range(17)},
        many,
    )
    with pytest.raises(ValueError, match="K policy ceiling"):
        FixedCAVRouter.load(
            tuple((many, f"concept_{index}.layer_2") for index in range(17)),
            layer=2,
        )

    router = FixedCAVRouter.load(
        (
            (first, "concept_a.layer_2"),
            (second, "concept_b.layer_2"),
        ),
        layer=2,
    )
    with pytest.raises(MemoryError, match="N policy ceiling"):
        router.route_one(torch.ones(65, 3))
    with pytest.raises(ValueError, match="width disagrees"):
        router.route_one(torch.ones(2, 2))
    with pytest.raises(ValueError, match="dtype disagrees"):
        router.route_one(torch.ones(2, 3, dtype=torch.float64))
    with pytest.raises(ValueError, match="non-finite"):
        router.route_one(
            torch.tensor([[float("nan"), 0.0, 0.0], [0.0, 1.0, 0.0]])
        )
    with pytest.raises(ValueError, match="non-zero norm"):
        router.route_one(torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
