from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass, replace

import pytest

from memory_condense.search.fusion.fixed_cav_router import (
    FixedCAVForward,
    FixedCAVRouter,
)
from memory_condense.search.fusion.latent_router import LatentRouterForward
from memory_condense.search.fusion import steered_readout as readout_module
from memory_condense.search.fusion.steered_readout import (
    MatchedSteeredReadout,
    matched_steered_readout,
)


torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")


def _routed(steered_nodes, *, extraction=object(), reinjection=object()):
    return LatentRouterForward(
        steered_nodes=steered_nodes,
        extraction_attention=extraction,
        reinjection_attention=reinjection,
    )


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


def test_x1_alone_changes_treatment_order_while_attention_is_irrelevant() -> None:
    atom_ids = ("atom-a", "atom-b", "atom-c")
    query = torch.tensor([1.0, 0.0])
    nodes = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        dtype=torch.float32,
    )
    irrelevant_extraction = object()
    irrelevant_reinjection = object()
    first = matched_steered_readout(
        atom_ids=atom_ids,
        node_features=nodes,
        query_vector=query,
        routed=_routed(
            nodes.clone(),
            extraction=irrelevant_extraction,
            reinjection=irrelevant_reinjection,
        ),
        max_output_atoms=3,
        max_hidden_dim=2,
    )
    second = matched_steered_readout(
        atom_ids=atom_ids,
        node_features=nodes,
        query_vector=query,
        routed=_routed(
            torch.tensor(
                [[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]],
                dtype=torch.float32,
            ),
            extraction=irrelevant_extraction,
            reinjection=irrelevant_reinjection,
        ),
        max_output_atoms=3,
        max_hidden_dim=2,
    )

    assert first.base_scores == second.base_scores == (1.0, 0.0, -1.0)
    assert first.base_order == second.base_order == atom_ids
    assert first.treatment_order == atom_ids
    assert second.treatment_order == ("atom-b", "atom-a", "atom-c")
    assert first.base_scores_sha256 == second.base_scores_sha256
    assert first.base_order_sha256 == second.base_order_sha256
    assert first.treatment_scores_sha256 != second.treatment_scores_sha256
    assert first.treatment_order_sha256 != second.treatment_order_sha256
    assert first.readout_sha256 != second.readout_sha256


def test_readout_is_bounded_immutable_tensor_free_and_stably_tied() -> None:
    nodes = torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 1.0]])
    result = matched_steered_readout(
        atom_ids=("atom-z", "atom-y", "atom-x"),
        node_features=nodes,
        query_vector=torch.tensor([1.0, 0.0]),
        routed=_routed(nodes.clone()),
        max_output_atoms=3,
        max_hidden_dim=2,
    )

    assert type(result) is MatchedSteeredReadout
    assert result.base_order[:2] == ("atom-z", "atom-y")
    assert result.treatment_order[:2] == ("atom-z", "atom-y")
    assert result.atom_count == len(result.base_scores) == len(result.treatment_scores)
    assert result.atom_count <= result.max_output_atoms
    assert result.hidden_dim <= result.max_hidden_dim
    assert result.result_retained_tensor_bytes == 0
    assert not _contains_tensor(result)
    assert len(result.readout_sha256) == 64
    with pytest.raises(FrozenInstanceError):
        result.atom_count = 99


def test_receipt_is_deterministic_and_rejects_tampering() -> None:
    nodes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    kwargs = {
        "atom_ids": ("atom-a", "atom-b"),
        "node_features": nodes,
        "query_vector": torch.tensor([1.0, 0.0]),
        "routed": _routed(nodes.clone()),
        "max_output_atoms": 2,
        "max_hidden_dim": 2,
    }
    first = matched_steered_readout(**kwargs)
    second = matched_steered_readout(**kwargs)
    assert first == second

    with pytest.raises(ValueError, match="base_scores_sha256"):
        replace(first, base_scores=(0.5, 0.0))
    with pytest.raises(ValueError, match="readout_sha256"):
        replace(first, readout_sha256="0" * 64)


def test_float64_readout_has_no_full_residual_identity_or_cpu_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_shapes: list[tuple[int, ...]] = []
    original_canonicalize = readout_module.canonical_float32_tensor

    def observed_canonicalize(value, **kwargs):
        shape = getattr(value, "shape", (len(value),))
        canonical_shapes.append(tuple(int(item) for item in shape))
        return original_canonicalize(value, **kwargs)

    monkeypatch.setattr(
        readout_module,
        "canonical_float32_tensor",
        observed_canonicalize,
    )
    query = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    first_nodes = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    scaled_nodes = torch.tensor(
        [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
        dtype=torch.float64,
    )
    first = matched_steered_readout(
        atom_ids=("atom-a", "atom-b"),
        node_features=first_nodes,
        query_vector=query,
        routed=_routed(first_nodes.clone()),
        max_output_atoms=2,
        max_hidden_dim=3,
    )
    second = matched_steered_readout(
        atom_ids=("atom-a", "atom-b"),
        node_features=scaled_nodes,
        query_vector=query,
        routed=_routed(scaled_nodes.clone()),
        max_output_atoms=2,
        max_hidden_dim=3,
    )

    assert first == second
    assert first.source_dtype == "torch.float64"
    assert all(shape == (2,) for shape in canonical_shapes)
    field_names = {item.name for item in fields(first)}
    assert "node_features_sha256" not in field_names
    assert "steered_nodes_sha256" not in field_names
    assert "query_vector_sha256" not in field_names


def test_readout_accepts_exact_real_fixed_cav_forward(tmp_path) -> None:
    artifact = tmp_path / "fixed-cav.safetensors"
    safetensors_torch.save_file(
        {"vertical.layer_1": torch.tensor([0.0, 1.0])},
        artifact,
    )
    router = FixedCAVRouter.load(
        ((artifact, "vertical.layer_1"),),
        layer=1,
        alpha=0.75,
    )
    nodes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    routed = router.route_one(nodes)
    result = matched_steered_readout(
        atom_ids=("atom-a", "atom-b"),
        node_features=nodes,
        query_vector=torch.tensor([1.0, 0.0]),
        routed=routed,
        max_output_atoms=2,
        max_hidden_dim=2,
    )

    assert type(routed) is FixedCAVForward
    assert result.base_scores == (1.0, 0.0)
    assert result.treatment_scores != result.base_scores
    assert result.result_retained_tensor_bytes == 0


def test_readout_rejects_shape_dtype_finiteness_norm_and_cap_violations() -> None:
    nodes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    query = torch.tensor([1.0, 0.0])
    routed = _routed(nodes.clone())
    common = {
        "atom_ids": ("atom-a", "atom-b"),
        "node_features": nodes,
        "query_vector": query,
        "routed": routed,
        "max_output_atoms": 2,
        "max_hidden_dim": 2,
    }

    with pytest.raises(ValueError, match=r"\[N, D\]"):
        matched_steered_readout(**{**common, "node_features": nodes.unsqueeze(0)})
    with pytest.raises(ValueError, match="atom_ids disagree"):
        matched_steered_readout(**{**common, "atom_ids": ("atom-a",)})
    with pytest.raises(TypeError, match="sequence of atom identifiers"):
        matched_steered_readout(**{**common, "atom_ids": "ab"})
    with pytest.raises(ValueError, match="query_vector has the wrong shape"):
        matched_steered_readout(**{**common, "query_vector": query.unsqueeze(0)})
    with pytest.raises(ValueError, match="steered_nodes has the wrong shape"):
        matched_steered_readout(**{**common, "routed": _routed(nodes[:1])})
    with pytest.raises(ValueError, match="changed source dtype"):
        matched_steered_readout(
            **{**common, "query_vector": query.to(dtype=torch.float64)}
        )
    with pytest.raises(ValueError, match="non-finite"):
        matched_steered_readout(
            **{
                **common,
                "routed": _routed(torch.tensor([[float("nan"), 0.0], [0.0, 1.0]])),
            }
        )
    with pytest.raises(ValueError, match="non-zero norm"):
        matched_steered_readout(**{**common, "query_vector": torch.zeros(2)})
    with pytest.raises(ValueError, match="non-zero norm"):
        matched_steered_readout(
            **{**common, "routed": _routed(torch.tensor([[0.0, 0.0], [0.0, 1.0]]))}
        )
    with pytest.raises(MemoryError, match="max_output_atoms"):
        matched_steered_readout(**{**common, "max_output_atoms": 1})
    with pytest.raises(MemoryError, match="max_hidden_dim"):
        matched_steered_readout(**{**common, "max_hidden_dim": 1})
    with pytest.raises(ValueError, match="immutable policy ceiling"):
        matched_steered_readout(**{**common, "max_output_atoms": 65})
    with pytest.raises(ValueError, match="immutable policy ceiling"):
        matched_steered_readout(**{**common, "max_hidden_dim": 4097})
