#!/usr/bin/env python3
"""Posthoc target-cell and semantic-policy diagnostic for the sealed missing-4 run.

The construction is fully validated before the target-bearing audit is opened.
The analysis then rebuilds one scoped namespace index at a time, joins exact
target source aliases to immutable cells, and replays the classifier's scalar
gates without provider calls.  Candidate policies are calibration diagnostics,
never runtime inputs or held-out claims.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.persistence.db import Database  # noqa: E402
from tools import run_reduced_second_read_retrieval_assay as reduced_cli  # noqa: E402
from tools import run_reduced_semantic_binary_search_assay as assay  # noqa: E402
from tools.matched_eval import semantic_residual_search as residual  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.full_store_slot_closure import (  # noqa: E402
    build_full_store_window_index,
)
from tools.matched_eval.query_guided_scan import (  # noqa: E402
    cache_namespace_partitions,
)


FORMAT = "memory-condense-reduced-semantic-target-cell-diagnostic-v1"
QUESTION_FORMAT = f"{FORMAT}-question"
TARGET_FORMAT = f"{FORMAT}-target"
CELL_FORMAT = f"{FORMAT}-target-cell"
PRUNER_FORMAT = f"{FORMAT}-pruning-decision"
POLICY_ASSAY_FORMAT = f"{FORMAT}-bounded-policy-assay"
NAME = "reduced-semantic-target-cell-diagnostic-v1.json"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/reduced-semantic-binary-search-missing4-v3"
)
DEFAULT_CONSTRUCTION = DEFAULT_ROOT / assay.CONSTRUCTION_NAME
DEFAULT_AUDIT = DEFAULT_ROOT / assay.AUDIT_NAME
DEFAULT_OUTPUT = DEFAULT_ROOT / NAME
EXPECTED_CONSTRUCTION_SHA256 = (
    "cb6c0e2c66be18039dbb6f246f333d909fd18f40e81231f0fbf167ebc55dfbc8"
)
EXPECTED_AUDIT_SHA256 = (
    "159046c20e22006666efe7662755589521587df1e6758fbaea67d466c48da4a4"
)
DEFAULT_MAX_CELL_TOKEN_CANDIDATES = (2_048, 1_024, 512, 256)
CONSERVATIVE_PROVIDER_MULTIPLIER = 1.45
CONSERVATIVE_PROVIDER_FIXED_TOKENS = 256


class ReducedSemanticTargetDiagnosticError(MatchedEvalContractError):
    """A sealed input, scoped replay, target join, or policy assay changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ReducedSemanticTargetDiagnosticError(message)


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact array")
    return value  # type: ignore[return-value]


def _identity_projection(value: object, label: str) -> dict[str, Any]:
    row = _exact_dict(value, label)
    body = dict(row)
    receipt = require_sha256(body.pop("receipt_sha256", None), label)
    _require(identity_sha256(body) == receipt, f"{label} receipt changed")
    return row


def _load_verified_audit(
    path: Path,
    *,
    expected_sha256: str,
    construction_sha256: str,
) -> tuple[SealedArtifact, tuple[dict[str, Any], ...]]:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == require_sha256(expected_sha256, "expected target audit"),
        "sealed target audit file changed",
    )
    payload = artifact.payload
    rows = tuple(
        _exact_dict(value, "target audit row")
        for value in _exact_list(payload.get("questions"), "target audit rows")
    )
    body = dict(payload)
    declared = require_sha256(
        body.pop("audit_identity_sha256", None), "target audit identity"
    )
    _require(
        identity_sha256(body) == declared
        and payload.get("format") == assay.AUDIT_FORMAT
        and payload.get("construction_artifact_sha256") == construction_sha256
        and payload.get("construction_verified_before_target_plan_load") is True
        and payload.get("audit_is_posthoc_only") is True
        and payload.get("runtime_use_forbidden") is True
        and payload.get("target_labels_loaded") is True
        and payload.get("target_plan_loaded") is True
        and payload.get("new_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and tuple(payload.get("ordinals", ())) == assay.TARGET_ORDINALS
        and len(rows) == assay.QUESTION_COUNT,
        "sealed target audit boundary changed",
    )
    for ordinal, row in zip(assay.TARGET_ORDINALS, rows, strict=True):
        row_body = dict(row)
        row_receipt = require_sha256(
            row_body.pop("audit_row_receipt_sha256", None), "target audit row"
        )
        expected = _exact_list(
            row.get("expected_source_ids"), "expected target sources"
        )
        _require(
            identity_sha256(row_body) == row_receipt
            and row.get("ordinal") == ordinal
            and expected
            and len(expected) == len(set(expected))
            and all(type(value) is str and value for value in expected),
            f"target audit row changed at ordinal {ordinal}",
        )
    return artifact, rows


def _policy_from_projection(value: object) -> residual.SemanticResidualPolicy:
    return residual.semantic_residual_policy_from_projection(value)


def _node_depths(root: Any) -> dict[str, int]:
    depths: dict[str, int] = {}

    def visit(node: Any, depth: int) -> None:
        _require(
            node.receipt_sha256 not in depths,
            "semantic tree repeated a node receipt",
        )
        depths[node.receipt_sha256] = depth
        for child in node.children:
            visit(child, depth + 1)

    visit(root, 0)
    return depths


def _classifier_topology(
    semantic_index: residual.SemanticResidualIndex,
    query: residual.SemanticResidualQuery,
) -> dict[str, Any]:
    classifier = residual._ConservativeResidualClassifier(  # noqa: SLF001
        semantic_index, query
    )
    depths = _node_depths(semantic_index.core_tree.root)
    cell_order = [cell.cell_id for cell in semantic_index.cells]
    cell_position = {cell_id: ordinal for ordinal, cell_id in enumerate(cell_order)}
    _require(
        len(cell_position) == len(cell_order),
        "semantic diagnostic cell order contains duplicates",
    )
    nodes: dict[str, dict[str, Any]] = {}
    for call_ordinal, node in enumerate(semantic_index.core_tree.nodes):
        decision = classifier.classify(
            question=query.dated_question,
            node=node,
            call_ordinal=call_ordinal,
        )
        audit = classifier.audits[-1]
        positions = [cell_position[cell_id] for cell_id in node.cell_ids]
        cell_start = min(positions)
        cell_stop = max(positions) + 1
        _require(
            positions == list(range(cell_start, cell_stop)),
            "semantic tree no longer preserves contiguous cell order",
        )
        nodes[node.receipt_sha256] = {
            "audit_receipt_sha256": audit.receipt_sha256,
            "children": [value.receipt_sha256 for value in node.children],
            "cosine_upper_bound": audit.cosine_upper_bound,
            "cell_start": cell_start,
            "cell_stop": cell_stop,
            "depth": depths[node.receipt_sha256],
            "fixed_negative_reason": (
                audit.reason
                if audit.reason in {"required_role_absent", "exact_literal_absent"}
                else None
            ),
            "max_leaf_specificity": audit.max_leaf_specificity,
            "node_id": node.node_id,
            "node_manifest_receipt_sha256": audit.node_manifest_receipt_sha256,
            "node_receipt_sha256": node.receipt_sha256,
            "specificity_gate_available": audit.specificity_gate_available,
            "specificity_upper_bound": audit.node_specificity_upper_bound,
            "tag_gate_available": audit.tag_gate_available,
            "token_count": node.token_count,
            "vector_gate_available": audit.vector_gate_available,
        }
    cells = {
        cell.cell_id: {
            "cell_id": cell.cell_id,
            "cell_receipt_sha256": cell.receipt_sha256,
            "source_id": cell.source_id,
            "token_count": cell.core_cell.token_count,
        }
        for cell in semantic_index.cells
    }
    return {
        "cell_order": cell_order,
        "cells": cells,
        "nodes": nodes,
        "root_node_receipt_sha256": semantic_index.core_tree.root.receipt_sha256,
    }


def _node_is_negative(
    row: Mapping[str, Any],
    *,
    cosine_floor: float,
    specificity_ratio: float,
) -> bool:
    if row.get("fixed_negative_reason") is not None:
        return True
    cosine = row.get("cosine_upper_bound")
    specificity = row.get("specificity_upper_bound")
    maximum = row.get("max_leaf_specificity")
    return bool(
        row.get("vector_gate_available") is True
        and row.get("tag_gate_available") is True
        and row.get("specificity_gate_available") is True
        and type(cosine) in {int, float}
        and type(specificity) in {int, float}
        and type(maximum) in {int, float}
        and float(cosine) < cosine_floor
        and float(specificity) < specificity_ratio * float(maximum)
    )


def simulate_policy(
    topology: Mapping[str, Any],
    *,
    cosine_floor: float,
    specificity_ratio: float,
) -> dict[str, Any]:
    """Replay exact branch semantics over a compact question topology."""

    nodes = _exact_dict(topology.get("nodes"), "policy topology nodes")
    cells = _exact_dict(topology.get("cells"), "policy topology cells")
    cell_order = _exact_list(topology.get("cell_order"), "policy topology cell order")
    _require(
        len(cell_order) == len(cells)
        and len(cell_order) == len(set(cell_order))
        and set(cell_order) == set(cells),
        "policy topology cell order changed its exact population",
    )
    retained: list[str] = []
    pruned: list[str] = []
    decisions: list[dict[str, Any]] = []

    def visit(receipt: str) -> None:
        node = _exact_dict(nodes.get(receipt), "policy topology node")
        cell_start = node.get("cell_start")
        cell_stop = node.get("cell_stop")
        _require(
            type(cell_start) is int
            and type(cell_stop) is int
            and 0 <= cell_start < cell_stop <= len(cell_order),
            "policy topology node has an invalid cell interval",
        )
        covered = cell_order[cell_start:cell_stop]
        negative = _node_is_negative(
            node,
            cosine_floor=cosine_floor,
            specificity_ratio=specificity_ratio,
        )
        decisions.append(
            {
                "branch_classification": (
                    "definitely_no" if negative else "may_answer"
                ),
                "node_receipt_sha256": receipt,
            }
        )
        if negative:
            pruned.extend(covered)
            return
        children = _exact_list(node.get("children"), "policy topology children")
        if not children:
            _require(len(covered) == 1, "semantic leaf changed coverage")
            retained.extend(covered)
            return
        _require(len(children) == 2, "semantic internal node changed arity")
        for child in children:
            visit(require_sha256(child, "semantic child node"))

    visit(
        require_sha256(
            topology.get("root_node_receipt_sha256"), "semantic topology root"
        )
    )
    _require(
        len(retained) == len(set(retained))
        and len(pruned) == len(set(pruned))
        and set(retained).isdisjoint(pruned)
        and set(retained) | set(pruned) == set(cells),
        "simulated policy lost the exact leaf partition",
    )
    raw_tokens = sum(
        int(_exact_dict(cells[cell_id], "simulated retained cell")["token_count"])
        for cell_id in retained
    )
    return {
        "decision_count": len(decisions),
        "pruned_cell_ids": pruned,
        "raw_retained_tokens": raw_tokens,
        "retained_cell_ids": retained,
    }


def _target_ids_for_cells(
    topology: Mapping[str, Any],
    retained_cell_ids: Sequence[str],
    *,
    expected_target_ids: Sequence[str],
    question_id: str,
) -> tuple[str, ...]:
    cells = _exact_dict(topology.get("cells"), "target topology cells")
    aliases: set[str] = set()
    for cell_id in retained_cell_ids:
        source_id = require_text(
            _exact_dict(cells.get(cell_id), "target retained cell").get("source_id"),
            "target retained source",
        )
        aliases.update(reduced_cli._source_aliases((source_id,), question_id))  # noqa: SLF001
    return tuple(value for value in expected_target_ids if value in aliases)


def _pruning_node_for_cell(
    topology: Mapping[str, Any],
    cell_id: str,
    *,
    cosine_floor: float,
    specificity_ratio: float,
) -> dict[str, Any] | None:
    nodes = _exact_dict(topology.get("nodes"), "pruning topology nodes")
    cell_order = _exact_list(topology.get("cell_order"), "pruning cell order")
    _require(cell_id in cell_order, "target cell is absent from topology order")
    cell_position = cell_order.index(cell_id)
    receipt = require_sha256(
        topology.get("root_node_receipt_sha256"), "pruning topology root"
    )
    while True:
        node = _exact_dict(nodes.get(receipt), "pruning topology node")
        cell_start = node.get("cell_start")
        cell_stop = node.get("cell_stop")
        _require(
            type(cell_start) is int
            and type(cell_stop) is int
            and cell_start <= cell_position < cell_stop,
            "target cell escaped its pruning path",
        )
        if _node_is_negative(
            node,
            cosine_floor=cosine_floor,
            specificity_ratio=specificity_ratio,
        ):
            return node
        children = _exact_list(node.get("children"), "pruning node children")
        if not children:
            return None
        matches: list[object] = []
        for child in children:
            child_node = _exact_dict(nodes.get(child), "pruning child")
            child_start = child_node.get("cell_start")
            child_stop = child_node.get("cell_stop")
            if (
                type(child_start) is int
                and type(child_stop) is int
                and child_start <= cell_position < child_stop
            ):
                matches.append(child)
        _require(len(matches) == 1, "target cell changed its binary path")
        receipt = require_sha256(matches[0], "pruning child receipt")


def _policy_boundaries(
    question_contexts: Sequence[Mapping[str, Any]],
    *,
    baseline_floor: float,
    baseline_ratio: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    floors = {baseline_floor}
    ratios = {baseline_ratio}
    for context in question_contexts:
        topology = _exact_dict(context.get("topology"), "boundary topology")
        for cell_id in context.get("target_cell_ids", ()):  # type: ignore[union-attr]
            pruner = _pruning_node_for_cell(
                topology,
                str(cell_id),
                cosine_floor=baseline_floor,
                specificity_ratio=baseline_ratio,
            )
            if pruner is None:
                continue
            cosine = pruner.get("cosine_upper_bound")
            specificity = pruner.get("specificity_upper_bound")
            maximum = pruner.get("max_leaf_specificity")
            if type(cosine) in {int, float}:
                floors.add(float(cosine))
            if (
                type(specificity) in {int, float}
                and type(maximum) in {int, float}
                and float(maximum) > 0.0
            ):
                ratios.add(float(specificity) / float(maximum))
    return tuple(sorted(floors, reverse=True)), tuple(sorted(ratios, reverse=True))


def _candidate_assay(
    contexts_by_max_cell_tokens: Mapping[
        int, Sequence[Mapping[str, Any]]
    ],
    *,
    baseline_floor: float,
    baseline_ratio: float,
    conservative_terminal_overhead_tokens: int,
    hard_complete_token_cap: int,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for max_cell_tokens, contexts in sorted(
        contexts_by_max_cell_tokens.items(), reverse=True
    ):
        floors, ratios = _policy_boundaries(
            contexts,
            baseline_floor=baseline_floor,
            baseline_ratio=baseline_ratio,
        )
        for floor in floors:
            for ratio in ratios:
                question_rows: list[dict[str, Any]] = []
                target_hits = 0
                for context in contexts:
                    topology = _exact_dict(context.get("topology"), "candidate topology")
                    simulated = simulate_policy(
                        topology,
                        cosine_floor=floor,
                        specificity_ratio=ratio,
                    )
                    expected = tuple(context.get("expected_target_ids", ()))
                    hits = _target_ids_for_cells(
                        topology,
                        simulated["retained_cell_ids"],
                        expected_target_ids=expected,
                        question_id=require_text(
                            context.get("question_id"), "candidate question"
                        ),
                    )
                    provider_estimate = math.ceil(
                        CONSERVATIVE_PROVIDER_MULTIPLIER
                        * int(simulated["raw_retained_tokens"])
                        + CONSERVATIVE_PROVIDER_FIXED_TOKENS
                    )
                    complete_estimate = (
                        provider_estimate + conservative_terminal_overhead_tokens
                    )
                    question_rows.append(
                        {
                            "estimated_complete_envelope_tokens": complete_estimate,
                            "estimated_residual_provider_tokens": provider_estimate,
                            "ordinal": context.get("ordinal"),
                            "raw_retained_tokens": simulated["raw_retained_tokens"],
                            "retained_cell_count": len(
                                simulated["retained_cell_ids"]
                            ),
                            "target_hits": list(hits),
                            "target_total": len(expected),
                        }
                    )
                    target_hits += len(hits)
                likely_fit = all(
                    row["estimated_complete_envelope_tokens"]
                    <= hard_complete_token_cap
                    for row in question_rows
                )
                body = {
                    "cosine_upper_bound_floor": floor,
                    "full_target_reach": target_hits == 6,
                    "likely_hard_cap_fit": likely_fit,
                    "max_cell_tokens": max_cell_tokens,
                    "max_estimated_complete_envelope_tokens": max(
                        row["estimated_complete_envelope_tokens"]
                        for row in question_rows
                    ),
                    "max_raw_retained_tokens": max(
                        row["raw_retained_tokens"] for row in question_rows
                    ),
                    "question_rows": question_rows,
                    "specificity_upper_bound_ratio": ratio,
                    "target_hits": target_hits,
                    "target_total": 6,
                    "total_raw_retained_tokens": sum(
                        row["raw_retained_tokens"] for row in question_rows
                    ),
                }
                candidates.append(
                    {**body, "candidate_receipt_sha256": identity_sha256(body)}
                )
    ranked = sorted(
        candidates,
        key=lambda row: (
            not row["full_target_reach"],
            not row["likely_hard_cap_fit"],
            row["max_estimated_complete_envelope_tokens"],
            row["total_raw_retained_tokens"],
            abs(row["cosine_upper_bound_floor"] - baseline_floor)
            + abs(row["specificity_upper_bound_ratio"] - baseline_ratio),
            -row["max_cell_tokens"],
        ),
    )
    _require(bool(ranked), "bounded policy assay produced no candidates")
    recommendation = ranked[0]
    body = {
        "calibration_is_posthoc_only": True,
        "candidate_count": len(ranked),
        "conservative_provider_fixed_tokens": CONSERVATIVE_PROVIDER_FIXED_TOKENS,
        "conservative_provider_multiplier": CONSERVATIVE_PROVIDER_MULTIPLIER,
        "conservative_terminal_overhead_tokens": (
            conservative_terminal_overhead_tokens
        ),
        "format": POLICY_ASSAY_FORMAT,
        "recommendation": recommendation,
        "recommendation_is_likely_not_exact_fit_validation": True,
        "top_candidates": ranked[: min(12, len(ranked))],
    }
    return {**body, "receipt_sha256": identity_sha256(body)}


def _stored_branch_maps(
    construction_row: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    search = _exact_dict(
        construction_row.get("semantic_residual_search"), "stored semantic search"
    )
    core = _exact_dict(search.get("core_result"), "stored semantic core")
    decisions = tuple(
        _identity_projection(value, "stored semantic decision")
        for value in _exact_list(core.get("decisions"), "stored decisions")
    )
    audits = tuple(
        _identity_projection(value, "stored semantic decision audit")
        for value in _exact_list(search.get("decision_audits"), "stored audits")
    )
    decision_by_node = {
        require_sha256(value.get("node_receipt_sha256"), "decision node"): value
        for value in decisions
    }
    audit_by_decision = {
        require_sha256(value.get("decision_receipt_sha256"), "audit decision"): value
        for value in audits
    }
    _require(
        len(decision_by_node) == len(decisions)
        and len(audit_by_decision) == len(audits)
        and len(decisions) == len(audits),
        "stored branch decision/audit population changed",
    )
    return decision_by_node, audit_by_decision


def _baseline_target_rows(
    construction_row: Mapping[str, Any],
    audit_row: Mapping[str, Any],
    semantic_index: residual.SemanticResidualIndex,
    query: residual.SemanticResidualQuery,
    topology: Mapping[str, Any],
    baseline_policy: residual.SemanticResidualPolicy,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    question_id = require_text(construction_row.get("question_id"), "question ID")
    expected = tuple(
        require_text(value, "expected target")
        for value in _exact_list(
            audit_row.get("expected_source_ids"), "expected targets"
        )
    )
    cells = _exact_dict(topology.get("cells"), "baseline topology cells")
    nodes = _exact_dict(topology.get("nodes"), "baseline topology nodes")
    target_cells: dict[str, list[str]] = {value: [] for value in expected}
    for cell_id, raw_cell in cells.items():
        cell = _exact_dict(raw_cell, "baseline topology cell")
        aliases = reduced_cli._source_aliases(  # noqa: SLF001
            (require_text(cell.get("source_id"), "baseline target source"),),
            question_id,
        )
        for target_id in expected:
            if target_id in aliases:
                target_cells[target_id].append(cell_id)
    _require(
        all(target_cells.values()),
        "a sealed target source is absent from the rebuilt semantic index",
    )
    search = _exact_dict(
        construction_row.get("semantic_residual_search"), "baseline semantic search"
    )
    core = _exact_dict(search.get("core_result"), "baseline semantic core")
    retained = set(
        _exact_list(core.get("retained_leaf_cell_ids"), "baseline retained cells")
    )
    decision_by_node, audit_by_decision = _stored_branch_maps(construction_row)
    node_by_receipt = {
        value.receipt_sha256: value for value in semantic_index.core_tree.nodes
    }
    manifest_by_node = semantic_index.manifest_by_node_receipt
    rows: list[dict[str, Any]] = []
    all_target_cell_ids: list[str] = []
    selected_target_ids: list[str] = []
    for target_id in expected:
        cell_rows: list[dict[str, Any]] = []
        target_selected = False
        for cell_id in target_cells[target_id]:
            all_target_cell_ids.append(cell_id)
            cell = _exact_dict(cells[cell_id], "baseline target cell")
            leaf = next(
                value
                for value in semantic_index.core_tree.nodes
                if value.is_leaf and value.cell_ids == (cell_id,)
            )
            leaf_metric = _exact_dict(
                nodes[leaf.receipt_sha256], "baseline target leaf metric"
            )
            selected = cell_id in retained
            target_selected = target_selected or selected
            pruner = _pruning_node_for_cell(
                topology,
                cell_id,
                cosine_floor=baseline_policy.cosine_upper_bound_floor,
                specificity_ratio=baseline_policy.specificity_upper_bound_ratio,
            )
            pruner_projection: dict[str, Any] | None = None
            if not selected:
                _require(pruner is not None, "pruned target cell has no pruning node")
                node_receipt = require_sha256(
                    pruner.get("node_receipt_sha256"), "target pruning node"
                )
                stored_decision = _exact_dict(
                    decision_by_node.get(node_receipt), "target pruning decision"
                )
                stored_audit = _exact_dict(
                    audit_by_decision.get(stored_decision.get("receipt_sha256")),
                    "target pruning audit",
                )
                rebuilt_node = node_by_receipt[node_receipt]
                rebuilt_manifest = manifest_by_node[node_receipt]
                _require(
                    stored_decision.get("branch_classification") == "definitely_no"
                    and stored_audit.get("reason")
                    in {"dual_gate", "required_role_absent", "exact_literal_absent"}
                    and stored_audit.get("node_manifest_receipt_sha256")
                    == rebuilt_manifest.receipt_sha256
                    and cell_id in rebuilt_node.cell_ids,
                    "stored target pruning decision differs from rebuilt branch",
                )
                pruner_body = {
                    "audit_receipt_sha256": stored_audit.get("receipt_sha256"),
                    "branch_classification": "definitely_no",
                    "call_ordinal": stored_decision.get("call_ordinal"),
                    "cosine_upper_bound": stored_audit.get("cosine_upper_bound"),
                    "covered_leaf_cell_count": len(rebuilt_node.cell_ids),
                    "decision_receipt_sha256": stored_decision.get("receipt_sha256"),
                    "depth": pruner.get("depth"),
                    "format": PRUNER_FORMAT,
                    "max_leaf_specificity": stored_audit.get(
                        "max_leaf_specificity"
                    ),
                    "node_id": rebuilt_node.node_id,
                    "node_receipt_sha256": node_receipt,
                    "reason": stored_audit.get("reason"),
                    "specificity_threshold": stored_audit.get(
                        "specificity_threshold"
                    ),
                    "specificity_upper_bound": stored_audit.get(
                        "node_specificity_upper_bound"
                    ),
                }
                pruner_projection = {
                    **pruner_body,
                    "receipt_sha256": identity_sha256(pruner_body),
                }
            maximum = float(leaf_metric["max_leaf_specificity"])
            specificity = float(leaf_metric["specificity_upper_bound"])
            cell_body = {
                **cell,
                "format": CELL_FORMAT,
                "leaf_cosine_max": leaf_metric.get("cosine_upper_bound"),
                "leaf_node_receipt_sha256": leaf.receipt_sha256,
                "leaf_specificity_fraction_of_question_max": (
                    specificity / maximum if maximum > 0.0 else None
                ),
                "leaf_specificity_idf_sum": specificity,
                "pruning_decision": pruner_projection,
                "selected_by_baseline": selected,
            }
            cell_rows.append(
                {**cell_body, "receipt_sha256": identity_sha256(cell_body)}
            )
        if target_selected:
            selected_target_ids.append(target_id)
        target_body = {
            "cell_count": len(cell_rows),
            "cells": cell_rows,
            "format": TARGET_FORMAT,
            "selected_by_baseline": target_selected,
            "target_id": target_id,
        }
        rows.append(
            {**target_body, "receipt_sha256": identity_sha256(target_body)}
        )
    _require(
        tuple(selected_target_ids)
        == tuple(audit_row.get("selected_source_target_hits", ())),
        "rebuilt target-cell reach differs from sealed posthoc audit",
    )
    return rows, tuple(dict.fromkeys(all_target_cell_ids))


def _question_context(
    *,
    ordinal: int,
    construction_row: Mapping[str, Any],
    audit_row: Mapping[str, Any],
    semantic_index: residual.SemanticResidualIndex,
    vectors: Sequence[Sequence[float]],
    vector_artifact_sha256: str,
) -> tuple[dict[str, Any], residual.SemanticResidualQuery]:
    dated_question = require_text(
        _exact_dict(
            construction_row.get("semantic_query"), "stored semantic query"
        ).get("dated_question"),
        "stored dated question",
    )
    query = residual.compile_semantic_residual_query(
        semantic_index,
        dated_question,
        query_vectors=vectors,
        query_vector_artifact_sha256=vector_artifact_sha256,
    )
    topology = _classifier_topology(semantic_index, query)
    expected = tuple(
        require_text(value, "context target")
        for value in _exact_list(
            audit_row.get("expected_source_ids"), "context targets"
        )
    )
    question_id = require_text(construction_row.get("question_id"), "context question")
    target_cell_ids: list[str] = []
    for cell_id, raw in _exact_dict(topology["cells"], "context cells").items():
        source_id = require_text(
            _exact_dict(raw, "context cell").get("source_id"), "context source"
        )
        aliases = reduced_cli._source_aliases((source_id,), question_id)  # noqa: SLF001
        if any(value in aliases for value in expected):
            target_cell_ids.append(cell_id)
    _require(target_cell_ids, f"candidate topology lost target cells at {ordinal}")
    return (
        {
            "expected_target_ids": list(expected),
            "ordinal": ordinal,
            "question_id": question_id,
            "target_cell_ids": target_cell_ids,
            "topology": topology,
        },
        query,
    )


def build_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    # Runtime/gold firewall: construction validation completes before audit IO.
    construction, construction_rows = assay.load_verified_construction(
        Path(args.construction),
        expected_sha256=args.expected_construction_sha256,
    )
    audit, audit_rows = _load_verified_audit(
        Path(args.audit),
        expected_sha256=args.expected_audit_sha256,
        construction_sha256=construction.sha256,
    )
    construction_by_ordinal = {
        int(value["ordinal"]): value for value in construction_rows
    }
    audit_by_ordinal = {int(value["ordinal"]): value for value in audit_rows}
    vector_sha = require_sha256(
        _exact_dict(
            construction.payload.get("bindings"), "construction bindings"
        ).get("query_vector_artifact_sha256"),
        "construction query vectors",
    )
    vector_artifact, vector_rows = assay.load_query_vectors(
        Path(args.vector_artifact), expected_sha256=vector_sha
    )
    vector_by_ordinal = {
        int(row["ordinal"]): (row, vectors) for row, vectors in vector_rows
    }
    baseline_policy = _policy_from_projection(
        construction.payload.get("semantic_residual_policy")
    )
    max_candidates = tuple(dict.fromkeys(args.max_cell_token_candidates))
    _require(
        max_candidates
        and all(type(value) is int and value > 0 for value in max_candidates)
        and baseline_policy.max_cell_tokens in max_candidates,
        "max-cell candidate set must include the sealed baseline",
    )
    lifecycle = _exact_dict(
        construction.payload.get("resident_index_lifecycle"),
        "construction lifecycle",
    )
    lifecycle_by_namespace = {
        require_sha256(row.get("namespace_id"), "lifecycle namespace"): row
        for raw in _exact_list(lifecycle.get("receipts"), "lifecycle rows")
        for row in [_exact_dict(raw, "lifecycle row")]
    }
    ordinals_by_namespace: dict[str, list[int]] = defaultdict(list)
    for ordinal in assay.TARGET_ORDINALS:
        namespace_id = require_sha256(
            construction_by_ordinal[ordinal].get("namespace_id"),
            "construction namespace",
        )
        ordinals_by_namespace[namespace_id].append(ordinal)
    contexts_by_max: dict[int, list[dict[str, Any]]] = {
        value: [] for value in max_candidates
    }
    baseline_question_rows: dict[int, dict[str, Any]] = {}
    baseline_terminal_overheads = [
        int(row["terminal_prompt"]["full_chat_plus_output_tokens"])
        - int(row["semantic_residual_search"]["attempted_provider_payload_tokens"])
        for row in construction_rows
        if type(row.get("terminal_prompt")) is dict
    ]
    conservative_terminal_overhead = max(baseline_terminal_overheads)
    guided_args = reduced_cli._guided_args(args)  # noqa: SLF001
    lifecycle_rows: list[dict[str, Any]] = []
    for namespace_id in sorted(ordinals_by_namespace):
        scoped = reduced_cli._scoped_guided_context(  # noqa: SLF001
            guided_args, namespace_id
        )
        sealed_lifecycle = lifecycle_by_namespace[namespace_id]
        _require(
            scoped.database_sha256
            == sealed_lifecycle.get("source_database_sha256")
            and scoped.index_sha256 == sealed_lifecycle.get("hnsw_index_sha256")
            and scoped.namespace.combined_store_receipt_sha256
            == sealed_lifecycle.get("source_store_receipt_sha256"),
            "scoped diagnostic store differs from sealed construction",
        )
        with Database(scoped.store_dir / "memory.db", read_only=True) as database:
            cache = cache_namespace_partitions(
                database,
                scoped.namespace,
                source_database_sha256=scoped.database_sha256,
                source_store_receipt_sha256=(
                    scoped.namespace.combined_store_receipt_sha256
                ),
            )
            window_index = build_full_store_window_index(cache)
            source_vectors = residual.load_stored_source_centroid_vectors(
                database, window_index
            )
        _require(
            cache.cache_receipt_sha256
            == sealed_lifecycle.get("cache_receipt_sha256")
            and window_index.receipt_sha256
            == sealed_lifecycle.get("window_index_receipt_sha256")
            and source_vectors.receipt_sha256
            == sealed_lifecycle.get("source_vector_set_receipt_sha256"),
            "rebuilt diagnostic cache/vector set differs from construction",
        )
        max_resident_cells = 0
        for max_cell_tokens in max_candidates:
            policy = residual.SemanticResidualPolicy(
                max_cell_tokens=max_cell_tokens,
                payload_token_cap=baseline_policy.payload_token_cap,
                cosine_upper_bound_floor=(
                    baseline_policy.cosine_upper_bound_floor
                ),
                specificity_upper_bound_ratio=(
                    baseline_policy.specificity_upper_bound_ratio
                ),
                dual_gate_enabled=baseline_policy.dual_gate_enabled,
                classifier_mode=baseline_policy.classifier_mode,
            )
            semantic_index = residual.build_semantic_residual_index(
                window_index, source_vectors, policy=policy
            )
            max_resident_cells = max(max_resident_cells, len(semantic_index.cells))
            if max_cell_tokens == baseline_policy.max_cell_tokens:
                _require(
                    semantic_index.projection()
                    == sealed_lifecycle.get("question_neutral_semantic_index"),
                    "rebuilt baseline semantic index differs from construction",
                )
            for ordinal in ordinals_by_namespace[namespace_id]:
                vector_row, vectors = vector_by_ordinal[ordinal]
                construction_row = construction_by_ordinal[ordinal]
                _require(
                    vector_row.get("row_receipt_sha256")
                    == construction_row.get("query_vector_row_receipt_sha256"),
                    "diagnostic vector row differs from construction",
                )
                context, query = _question_context(
                    ordinal=ordinal,
                    construction_row=construction_row,
                    audit_row=audit_by_ordinal[ordinal],
                    semantic_index=semantic_index,
                    vectors=vectors,
                    vector_artifact_sha256=vector_artifact.sha256,
                )
                contexts_by_max[max_cell_tokens].append(context)
                if max_cell_tokens == baseline_policy.max_cell_tokens:
                    _require(
                        query.projection() == construction_row.get("semantic_query"),
                        "rebuilt baseline query differs from construction",
                    )
                    target_rows, _target_cell_ids = _baseline_target_rows(
                        construction_row,
                        audit_by_ordinal[ordinal],
                        semantic_index,
                        query,
                        context["topology"],
                        baseline_policy,
                    )
                    body = {
                        "expected_target_count": len(target_rows),
                        "format": QUESTION_FORMAT,
                        "namespace_id": namespace_id,
                        "ordinal": ordinal,
                        "question_id": construction_row.get("question_id"),
                        "targets": target_rows,
                    }
                    baseline_question_rows[ordinal] = {
                        **body,
                        "receipt_sha256": identity_sha256(body),
                    }
            del semantic_index
            gc.collect()
        lifecycle_body = {
            "cache_receipt_sha256": cache.cache_receipt_sha256,
            "database_open_passes": 1,
            "max_cell_token_candidate_count": len(max_candidates),
            "maximum_resident_semantic_cell_count": max_resident_cells,
            "maximum_simultaneous_semantic_indexes": 1,
            "namespace_id": namespace_id,
            "source_vector_set_receipt_sha256": source_vectors.receipt_sha256,
            "window_index_receipt_sha256": window_index.receipt_sha256,
        }
        lifecycle_rows.append(
            {
                **lifecycle_body,
                "receipt_sha256": identity_sha256(lifecycle_body),
            }
        )
        del source_vectors, window_index, cache
        gc.collect()
    for contexts in contexts_by_max.values():
        contexts.sort(key=lambda value: int(value["ordinal"]))
        _require(
            tuple(value["ordinal"] for value in contexts)
            == assay.TARGET_ORDINALS,
            "bounded policy context lost the fixed four questions",
        )
    policy_assay = _candidate_assay(
        contexts_by_max,
        baseline_floor=baseline_policy.cosine_upper_bound_floor,
        baseline_ratio=baseline_policy.specificity_upper_bound_ratio,
        conservative_terminal_overhead_tokens=conservative_terminal_overhead,
        hard_complete_token_cap=assay.HARD_COMPLETE_CHAT_TOKEN_CAP,
    )
    questions = [baseline_question_rows[value] for value in assay.TARGET_ORDINALS]
    body: dict[str, Any] = {
        "analysis_is_posthoc_only": True,
        "audit_artifact_sha256": audit.sha256,
        "construction_artifact_sha256": construction.sha256,
        "construction_verified_before_target_audit_load": True,
        "format": FORMAT,
        "gold_loaded": True,
        "max_cell_token_candidates": list(max_candidates),
        "new_provider_calls": 0,
        "ordinals": list(assay.TARGET_ORDINALS),
        "policy_assay": policy_assay,
        "question_count": assay.QUESTION_COUNT,
        "questions": questions,
        "resident_index_lifecycle": {
            "maximum_simultaneous_semantic_indexes": 1,
            "namespace_count": len(lifecycle_rows),
            "rows": lifecycle_rows,
        },
        "retained_transformer_token_state_bytes": 0,
        "runtime_use_forbidden": True,
        "sealed_baseline_policy": baseline_policy.projection(),
        "target_labels_loaded": True,
        "target_source_count": sum(
            row["expected_target_count"] for row in questions
        ),
        "vector_artifact_sha256": vector_artifact.sha256,
    }
    _require(body["target_source_count"] == 6, "diagnostic target count changed")
    payload = {**body, "diagnostic_identity_sha256": identity_sha256(body)}
    validate_diagnostic(payload)
    return payload


def validate_diagnostic(payload: Mapping[str, Any]) -> None:
    row = _exact_dict(payload, "target-cell diagnostic")
    body = dict(row)
    declared = require_sha256(
        body.pop("diagnostic_identity_sha256", None), "target-cell diagnostic"
    )
    questions = _exact_list(row.get("questions"), "diagnostic questions")
    _require(
        identity_sha256(body) == declared
        and row.get("format") == FORMAT
        and row.get("analysis_is_posthoc_only") is True
        and row.get("runtime_use_forbidden") is True
        and row.get("construction_verified_before_target_audit_load") is True
        and row.get("gold_loaded") is True
        and row.get("target_labels_loaded") is True
        and row.get("new_provider_calls") == 0
        and row.get("retained_transformer_token_state_bytes") == 0
        and tuple(row.get("ordinals", ())) == assay.TARGET_ORDINALS
        and len(questions) == assay.QUESTION_COUNT
        and row.get("target_source_count") == 6,
        "target-cell diagnostic boundary changed",
    )
    _identity_projection(row.get("policy_assay"), "bounded policy assay")
    for ordinal, raw_question in zip(assay.TARGET_ORDINALS, questions, strict=True):
        question = _identity_projection(raw_question, "diagnostic question")
        targets = _exact_list(question.get("targets"), "diagnostic targets")
        _require(
            question.get("ordinal") == ordinal
            and question.get("expected_target_count") == len(targets),
            "diagnostic question target population changed",
        )
        for raw_target in targets:
            target = _identity_projection(raw_target, "diagnostic target")
            cells = _exact_list(target.get("cells"), "diagnostic target cells")
            _require(
                target.get("cell_count") == len(cells) and cells,
                "diagnostic target lost its exact cells",
            )
            for raw_cell in cells:
                cell = _identity_projection(raw_cell, "diagnostic target cell")
                if not cell.get("selected_by_baseline"):
                    _identity_projection(
                        cell.get("pruning_decision"), "target pruning decision"
                    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_diagnostic(args)
    target = Path(args.output)
    candidate = SealedArtifact(
        target,
        hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        payload,
    )
    validate_diagnostic(candidate.payload)
    artifact, created = publish_sealed_json(target, payload)
    recommendation = payload["policy_assay"]["recommendation"]
    return {
        "created": created,
        "diagnostic_sha256": artifact.sha256,
        "new_provider_calls": 0,
        "recommendation": {
            key: recommendation[key]
            for key in (
                "cosine_upper_bound_floor",
                "specificity_upper_bound_ratio",
                "max_cell_tokens",
                "target_hits",
                "target_total",
                "likely_hard_cap_fit",
            )
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construction", type=Path, default=DEFAULT_CONSTRUCTION)
    parser.add_argument(
        "--expected-construction-sha256",
        default=EXPECTED_CONSTRUCTION_SHA256,
    )
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--expected-audit-sha256", default=EXPECTED_AUDIT_SHA256)
    parser.add_argument(
        "--vector-artifact", type=Path, default=assay.DEFAULT_VECTOR_ARTIFACT
    )
    parser.add_argument(
        "--max-cell-token-candidates",
        type=int,
        nargs="+",
        default=list(DEFAULT_MAX_CELL_TOKEN_CANDIDATES),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    reduced_cli._add_store_args(parser)  # noqa: SLF001
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
