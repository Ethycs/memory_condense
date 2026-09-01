from __future__ import annotations

import copy

import pytest

from tools import analyze_reduced_semantic_target_cells as analysis


def _sha(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _node(
    name: str,
    cell_start: int,
    cell_stop: int,
    *,
    children: list[str] | None = None,
    cosine: float = 1.0,
    specificity: float = 1.0,
    maximum: float = 1.0,
    fixed: str | None = None,
) -> dict[str, object]:
    return {
        "cell_start": cell_start,
        "cell_stop": cell_stop,
        "children": [] if children is None else children,
        "cosine_upper_bound": cosine,
        "fixed_negative_reason": fixed,
        "max_leaf_specificity": maximum,
        "node_receipt_sha256": _sha(name),
        "specificity_gate_available": True,
        "specificity_upper_bound": specificity,
        "tag_gate_available": True,
        "vector_gate_available": True,
    }


def _two_leaf_topology(
    *,
    question_id: str = "q",
    target_ids: tuple[str, str] = ("target-a", "target-b"),
    tokens: int = 100,
) -> dict[str, object]:
    root = _sha(f"{question_id}-root")
    left = _sha(f"{question_id}-left")
    right = _sha(f"{question_id}-right")
    return {
        "cell_order": ["a", "b"],
        "cells": {
            "a": {
                "source_id": f"{question_id}::{target_ids[0]}",
                "token_count": tokens,
            },
            "b": {
                "source_id": f"{question_id}::{target_ids[1]}",
                "token_count": tokens,
            },
        },
        "nodes": {
            root: _node("root-body", 0, 2, children=[left, right]),
            left: _node(
                "left-body",
                0,
                1,
                cosine=0.2,
                specificity=0.2,
            ),
            right: _node(
                "right-body",
                1,
                2,
                cosine=0.8,
                specificity=0.2,
            ),
        },
        "root_node_receipt_sha256": root,
    }


def test_simulate_policy_prunes_only_when_both_gates_are_below_threshold() -> None:
    topology = _two_leaf_topology()

    baseline = analysis.simulate_policy(
        topology, cosine_floor=0.65, specificity_ratio=0.9
    )
    lowered_cosine = analysis.simulate_policy(
        topology, cosine_floor=0.2, specificity_ratio=0.9
    )
    lowered_specificity = analysis.simulate_policy(
        topology, cosine_floor=0.65, specificity_ratio=0.2
    )

    assert baseline["retained_cell_ids"] == ["b"]
    assert baseline["pruned_cell_ids"] == ["a"]
    # Strict inequality means equality at either boundary conservatively MAYs.
    assert lowered_cosine["retained_cell_ids"] == ["a", "b"]
    assert lowered_specificity["retained_cell_ids"] == ["a", "b"]


def test_interval_topology_is_equivalent_to_explicit_coverage_reference() -> None:
    topology = _two_leaf_topology(tokens=137)
    cell_order = topology["cell_order"]
    nodes = topology["nodes"]

    def reference(receipt: str, floor: float, ratio: float) -> tuple[list[str], list[str]]:
        node = nodes[receipt]
        covered = cell_order[node["cell_start"] : node["cell_stop"]]
        if analysis._node_is_negative(  # noqa: SLF001
            node, cosine_floor=floor, specificity_ratio=ratio
        ):
            return [], list(covered)
        if not node["children"]:
            return list(covered), []
        retained: list[str] = []
        pruned: list[str] = []
        for child in node["children"]:
            child_retained, child_pruned = reference(child, floor, ratio)
            retained.extend(child_retained)
            pruned.extend(child_pruned)
        return retained, pruned

    for floor, ratio in ((0.65, 0.9), (0.2, 0.9), (0.65, 0.2), (-1.0, 0.0)):
        expected_retained, expected_pruned = reference(
            topology["root_node_receipt_sha256"], floor, ratio
        )
        actual = analysis.simulate_policy(
            topology, cosine_floor=floor, specificity_ratio=ratio
        )
        assert actual["retained_cell_ids"] == expected_retained
        assert actual["pruned_cell_ids"] == expected_pruned
        assert actual["raw_retained_tokens"] == 137 * len(expected_retained)


def test_interval_topology_shape_does_not_duplicate_cell_coverage_lists() -> None:
    topology = _two_leaf_topology()

    assert topology["cell_order"] == ["a", "b"]
    assert all("covered_cell_ids" not in node for node in topology["nodes"].values())
    assert {
        (node["cell_start"], node["cell_stop"])
        for node in topology["nodes"].values()
    } == {(0, 2), (0, 1), (1, 2)}


def test_simulate_policy_keeps_fixed_negative_independent_of_numeric_gates() -> None:
    topology = _two_leaf_topology()
    left = topology["nodes"][_sha("q-left")]
    left["fixed_negative_reason"] = "required_role_absent"

    result = analysis.simulate_policy(
        topology, cosine_floor=-1.0, specificity_ratio=0.01
    )

    assert result["pruned_cell_ids"] == ["a"]
    assert result["retained_cell_ids"] == ["b"]


def test_bounded_candidate_assay_finds_exact_boundary_with_six_targets() -> None:
    contexts = []
    target_groups = (
        ("q42", ("a42", "b42")),
        ("q65", ("a65", "b65")),
        ("q74", ("a74", "unused74")),
        ("q79", ("a79", "unused79")),
    )
    for ordinal, (question_id, targets) in zip(
        (42, 65, 74, 79), target_groups, strict=True
    ):
        expected = targets if ordinal in {42, 65} else targets[:1]
        contexts.append(
            {
                "expected_target_ids": list(expected),
                "ordinal": ordinal,
                "question_id": question_id,
                "target_cell_ids": ["a", "b"],
                "topology": _two_leaf_topology(
                    question_id=question_id,
                    target_ids=targets,
                    tokens=80,
                ),
            }
        )

    result = analysis._candidate_assay(  # noqa: SLF001
        {512: contexts},
        baseline_floor=0.65,
        baseline_ratio=0.9,
        conservative_terminal_overhead_tokens=1_000,
        hard_complete_token_cap=8_000,
    )

    recommendation = result["recommendation"]
    assert recommendation["target_hits"] == 6
    assert recommendation["full_target_reach"] is True
    assert recommendation["likely_hard_cap_fit"] is True
    assert (
        recommendation["cosine_upper_bound_floor"] == 0.2
        or recommendation["specificity_upper_bound_ratio"] == 0.2
    )


def test_simulated_topology_tamper_fails_closed() -> None:
    topology = _two_leaf_topology()
    tampered = copy.deepcopy(topology)
    tampered["nodes"][_sha("q-root")]["children"] = [_sha("q-left")]

    with pytest.raises(
        analysis.ReducedSemanticTargetDiagnosticError,
        match="changed arity",
    ):
        analysis.simulate_policy(
            tampered, cosine_floor=0.65, specificity_ratio=0.9
        )
