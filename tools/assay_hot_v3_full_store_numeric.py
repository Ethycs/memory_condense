#!/usr/bin/env python3
"""Provider-free full-store numeric closure over sealed hot-v3 gate rows.

``construct`` authenticates the sealed gate and v3 selection, adapts only rows
routed to ``numeric_full_store``, builds at most one resident index per needed
namespace, and records deterministic results separately from runtime timings.
It never opens benchmark gold or calls a provider.  ``score`` is the only
command that opens references; it performs a post-hoc lexical-exact overlay on
the already sealed construction.

The production hot path should pass a persisted ingest-time
``FullStoreWindowIndex`` to :func:`assay_numeric_row`.  Rebuilding indexes in
this command exists to measure today's cold fallback, not as the target online
architecture.
"""

from __future__ import annotations

import argparse
import copy
import gc
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval._answer_normalization import normalize_answer
from memory_condense.persistence.db import Database
from tools import assay_hot_retrieval_1m as hot
from tools import assay_hot_retrieval_adaptive_full100 as v7
from tools import assay_hot_retrieval_assertion_projection_full100 as v2
from tools import assay_hot_retrieval_full100 as full100
from tools import assay_hot_retrieval_source_seed_hybrid_full100 as v3
from tools import assay_hot_v3_typed_operator_full100 as typed
from tools.matched_eval.contracts import assert_gold_blind, identity_sha256
from tools.matched_eval.full_store_slot_closure import (
    FullStoreWindowIndex,
    build_full_store_window_index,
)
from tools.matched_eval.hot_v3_typed_operator_adapter import adapt_hot_v3_arm
from tools.matched_eval.numeric_operand_specialist import (
    NumericOperandClosureResult,
    scan_numeric_operand_closure,
)
from tools.matched_eval.numeric_policy_frontier_bridge import (
    EXTENDED_SUPPORTED_DOMAINS,
    NumericPolicyFrontierBridgeResult,
    build_operator_first_numeric_frontier,
    operator_first_numeric_frontier_applicable,
)
from tools.matched_eval.operator_first_numeric_policy import (
    OperatorFirstNumericDecision,
    execute_operator_first_numeric_policy,
)
from tools.matched_eval.population import EXPECTED_RETRIEVAL_SHA256
from tools.matched_eval.query_expansion import load_locked_query_expansion_context
from tools.matched_eval.query_guided_scan import cache_namespace_partitions
from tools.matched_eval.typed_operator_executor import ExecutionStatus


FORMAT = "memory-condense-hot-v3-full-store-numeric-construction-v1"
RUNTIME_FORMAT = "memory-condense-hot-v3-full-store-numeric-runtime-v1"
SCORE_FORMAT = "memory-condense-hot-v3-full-store-numeric-score-v1"
ROW_FORMAT = "memory-condense-hot-v3-full-store-numeric-row-v1"
TIMING_FORMAT = "memory-condense-hot-v3-full-store-numeric-row-timing-v1"
POLICY_ID = "fast-v3-gated-full-store-numeric-closed-frontier-v1"

EXPECTED_GATE_CONSTRUCTION_SHA256 = (
    "6b70c3de6031f19128cbcdd04986b28d8b4617a893ed6063469eb3ba4962f86f"
)
EXPECTED_GATE_REPLAY_SHA256 = (
    "756c8d2041d2ed04a0dae370c8f1926543f95d7e4f73a09b8ec3353c9260c79b"
)
EXPECTED_V3_SELECTION_SHA256 = typed.EXPECTED_V3_SELECTION_SHA256
EXPECTED_V2_SELECTION_SHA256 = v3.EXPECTED_V2_SELECTION_SHA256
EXPECTED_V7_SELECTION_SHA256 = v3.EXPECTED_V7_SELECTION_SHA256

DEFAULT_V3_ROOT = v3.DEFAULT_OUTPUT_ROOT
DEFAULT_GATE_ROOT = typed.DEFAULT_OUTPUT_ROOT
DEFAULT_STORE_ROOT = v3.DEFAULT_SOURCE_ROOT
DEFAULT_RETRIEVAL = DEFAULT_STORE_ROOT / "retrieval.json"
DEFAULT_V7_ROOT = typed.DEFAULT_V7_ROOT
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-hot-v3-full-store-numeric-20260906"
)
CONSTRUCTION_NAME = "construction.json"
RUNTIME_NAME = "runtime.json"
SCORE_NAME = "scores.json"
EXPECTED_QUESTION_COUNT = v3.EXPECTED_QUESTION_COUNT


_ROW_IDENTITY_KEYS = (
    "ordinal",
    "shard_offset",
    "local_ordinal",
    "question_id",
    "probe_sha256",
    "retrieval_query_sha256",
    "prompt_question_sha256",
)


Clock = Callable[[], int]


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _elapsed(clock: Clock, started: int) -> int:
    value = clock() - started
    _require(type(value) is int and value >= 0, "runtime clock moved backwards")
    return value


def _percentile95(values: Sequence[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)])


def _result_row(
    *,
    ordinal: int,
    question_id: str,
    dated_question: str,
    local_input_sha256: str,
    applicable: bool,
    index: FullStoreWindowIndex | None,
    specialist: NumericOperandClosureResult | None,
    bridge: NumericPolicyFrontierBridgeResult | None,
    decision: OperatorFirstNumericDecision | None,
) -> dict[str, Any]:
    admissible = bool(
        applicable
        and index is not None
        and bridge is not None
        and decision is not None
        and bridge.applicable
        and bridge.frontier.closed
        and decision.status is ExecutionStatus.SUPPORTED
        and decision.decision == "replace"
        and bool(decision.prediction)
        and decision.policy_input_sha256 == local_input_sha256
        and bridge.frontier.policy_input_sha256 == local_input_sha256
        and decision.relevant_frontier_receipt_sha256
        == bridge.frontier.receipt_sha256
        and bridge.question_sha256 == quote_sha256(dated_question)
        and bridge.window_index_receipt_sha256 == index.receipt_sha256
    )
    body: dict[str, Any] = {
        "admissible_replacement": admissible,
        "applicable": applicable,
        "bridge": None,
        "decision": None,
        "format": ROW_FORMAT,
        "gold_loaded": False,
        "local_operator_input_sha256": local_input_sha256,
        "new_provider_calls": 0,
        "ordinal": ordinal,
        "policy_id": POLICY_ID,
        "question_id": question_id,
        "question_sha256": quote_sha256(dated_question),
        "retained_transformer_token_state_bytes": 0,
        "specialist_receipt_sha256": (
            None if specialist is None else specialist.receipt.receipt_sha256
        ),
        "window_index_receipt_sha256": (
            None if index is None else index.receipt_sha256
        ),
    }
    if bridge is not None:
        body["bridge"] = {
            "applicable": bridge.applicable,
            "census_atom_count": len(bridge.census_atoms),
            "census_material_fact_count": len(
                bridge.census_material_fact_sha256s
            ),
            "census_semantic_key_count": len(
                bridge.census_semantic_key_sha256s
            ),
            "closed": bridge.frontier.closed,
            "physical_content_rows_scanned": bridge.physical_content_rows_scanned,
            "physical_sentence_windows_scanned": (
                bridge.physical_sentence_windows_scanned
            ),
            "provider_material_fact_count": len(
                bridge.provider_material_fact_sha256s
            ),
            "provider_semantic_key_count": len(
                bridge.provider_semantic_key_sha256s
            ),
            "receipt_sha256": bridge.receipt_sha256,
            "represented_material_fact_count": len(
                bridge.represented_material_fact_sha256s
            ),
            "represented_semantic_key_count": len(
                bridge.represented_semantic_key_sha256s
            ),
            "unresolved_candidate_keys": list(bridge.unresolved_candidate_keys),
            "frontier_receipt_sha256": bridge.frontier.receipt_sha256,
        }
    if decision is not None:
        body["decision"] = {
            "decision": decision.decision,
            "mode": decision.mode.value,
            "numeric_result": decision.numeric_result,
            "policy_input_sha256": decision.policy_input_sha256,
            "prediction": decision.prediction,
            "reason": decision.reason,
            "receipt_sha256": decision.receipt_sha256,
            "relevant_frontier_receipt_sha256": (
                decision.relevant_frontier_receipt_sha256
            ),
            "status": decision.status.value,
            "used_handle_ids": list(decision.used_handle_ids),
        }
    assert_gold_blind(body, path="hot_v3_full_store_numeric_row")
    return {**body, "row_receipt_sha256": identity_sha256(body)}


def assay_numeric_row(
    *,
    ordinal: int,
    question_id: str,
    local_operator_input: Mapping[str, Any],
    index: FullStoreWindowIndex | None,
    clock: Clock = time.perf_counter_ns,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute the exact production numeric path for one gate-selected row.

    ``index`` may be ``None`` only when the question grammar is inapplicable.
    The caller must pass ``bundle.local_inventory.operator_input()``; passing
    the capped/provider projection loses local evidence and changes the bound
    policy-input receipt.
    """

    _require(type(ordinal) is int and ordinal >= 0, "ordinal is invalid")
    _require(type(question_id) is str and bool(question_id), "question ID is invalid")
    _require(isinstance(local_operator_input, Mapping), "local input is not a mapping")
    local_input = dict(local_operator_input)
    _require(
        set(local_input) == {"dated_question", "typed_evidence"},
        "numeric assay requires the exact local operator input",
    )
    dated_question = local_input["dated_question"]
    _require(
        type(dated_question) is str and bool(dated_question.strip()),
        "numeric assay question is invalid",
    )
    assert_gold_blind(local_input, path="hot_v3_full_store_numeric_input")
    local_sha = identity_sha256(local_input)
    row_started = clock()
    step = clock()
    applicable = operator_first_numeric_frontier_applicable(
        local_input,
        supported_domains=EXTENDED_SUPPORTED_DOMAINS,
    )
    applicability_ns = _elapsed(clock, step)
    specialist: NumericOperandClosureResult | None = None
    bridge: NumericPolicyFrontierBridgeResult | None = None
    decision: OperatorFirstNumericDecision | None = None
    specialist_ns = bridge_ns = decision_ns = 0
    if applicable:
        _require(
            type(index) is FullStoreWindowIndex,
            "applicable numeric row requires a resident full-store index",
        )
        step = clock()
        specialist = scan_numeric_operand_closure(index, dated_question)
        specialist_ns = _elapsed(clock, step)
        step = clock()
        bridge = build_operator_first_numeric_frontier(
            local_input,
            index=index,
            specialist_result=specialist,
            supported_domains=EXTENDED_SUPPORTED_DOMAINS,
            operator_material_status=True,
        )
        bridge_ns = _elapsed(clock, step)
        step = clock()
        decision = execute_operator_first_numeric_policy(
            local_input,
            relevant_frontier=bridge.frontier,
        )
        decision_ns = _elapsed(clock, step)
    result = _result_row(
        ordinal=ordinal,
        question_id=question_id,
        dated_question=dated_question,
        local_input_sha256=local_sha,
        applicable=applicable,
        index=index,
        specialist=specialist,
        bridge=bridge,
        decision=decision,
    )
    timing = {
        "applicability_ns": applicability_ns,
        "bridge_ns": bridge_ns,
        "decision_ns": decision_ns,
        "format": TIMING_FORMAT,
        "ordinal": ordinal,
        "question_id": question_id,
        "result_row_receipt_sha256": result["row_receipt_sha256"],
        "specialist_ns": specialist_ns,
        "total_ns": _elapsed(clock, row_started),
    }
    return result, timing


def _implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        "tools/assay_hot_v3_full_store_numeric.py",
        "tools/matched_eval/full_store_slot_closure.py",
        "tools/matched_eval/hot_v3_typed_operator_adapter.py",
        "tools/matched_eval/numeric_operand_specialist.py",
        "tools/matched_eval/numeric_policy_frontier_bridge.py",
        "tools/matched_eval/operator_first_numeric_policy.py",
    )
    files = {path: file_sha256(root / path) for path in paths}
    return {
        "files": files,
        "sha256": identity_sha256(
            [{"path": path, "sha256": files[path]} for path in paths]
        ),
    }


def _read_gate(gate_root: Path) -> tuple[dict[str, Any], str, str]:
    construction, digest = hot._read_json_artifact(  # noqa: SLF001
        gate_root / typed.CONSTRUCTION_NAME
    )
    replay, replay_sha = hot._read_json_artifact(  # noqa: SLF001
        gate_root / typed.REPLAY_NAME
    )
    _require(digest == EXPECTED_GATE_CONSTRUCTION_SHA256, "sealed gate changed")
    _require(replay_sha == EXPECTED_GATE_REPLAY_SHA256, "sealed gate replay changed")
    _require(
        construction.get("gold_fields_present") is False
        and construction.get("provider_calls") == 0
        and replay.get("construction_sha256") == digest
        and replay.get("byte_identical") is True,
        "sealed gate crossed its replay firebreak",
    )
    assert_gold_blind(construction, path="hot_v3_numeric_gate")
    return construction, digest, replay_sha


def _gate_rows(gate: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    raw = gate.get("questions")
    _require(isinstance(raw, list), "sealed gate rows changed type")
    rows = {
        int(row["ordinal"]): row
        for row in raw
        if isinstance(row, Mapping)
        and type(row.get("ordinal")) is int
        and isinstance(row.get("next_action"), Mapping)
        and row["next_action"].get("next_action") == "numeric_full_store"
    }
    _require(len(rows) == 32, "numeric gate population changed")
    return rows


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _primary_checkout_root() -> Path:
    root = _repository_root()
    return root.parent.parent if root.parent.name == ".worktrees" else root


def _bound_repository_path(value: object, *, label: str) -> Path:
    """Resolve an authenticated compact-artifact checkout binding."""

    _require(
        type(value) is str and ":" in value,
        f"{label} binding must be checkout-scoped",
    )
    scope, relative_text = value.split(":", 1)
    roots = {
        "current-worktree": _repository_root(),
        "primary-checkout": _primary_checkout_root(),
    }
    root = roots.get(scope)
    relative = Path(relative_text)
    _require(
        root is not None
        and bool(relative_text)
        and not relative.is_absolute()
        and ".." not in relative.parts,
        f"{label} binding escaped its repository checkout",
    )
    target = (root / relative).resolve()
    _require(
        target.is_relative_to(root.resolve()),
        f"{label} binding escaped its repository checkout",
    )
    return target


def _read_fixed_selection(
    path: Path,
    *,
    expected_sha256: str,
    expected_format: str,
    expected_status: str,
    label: str,
) -> tuple[dict[str, Any], str]:
    """Read immutable parent bytes without consulting live implementation IDs."""

    body, digest = hot._read_json_artifact(path)  # noqa: SLF001
    _require(digest == expected_sha256, f"sealed {label} selection changed")
    _require(
        body.get("format") == expected_format
        and body.get("status") == expected_status
        and body.get("gold_fields_present") is False
        and type(body.get("provider_calls")) is int
        and body.get("provider_calls") == 0,
        f"sealed {label} selection crossed its construction firebreak",
    )
    for key in ("judge_calls", "qwen_calls", "responder_calls"):
        if key in body:
            _require(
                type(body[key]) is int and body[key] == 0,
                f"sealed {label} {key} changed",
            )
    assert_gold_blind(body, path=f"sealed_{label}_selection")
    return body, digest


def _locked_rows(
    selection: Mapping[str, Any], *, label: str
) -> dict[int, Mapping[str, Any]]:
    rows = selection.get("questions")
    _require(
        isinstance(rows, list) and len(rows) == EXPECTED_QUESTION_COUNT,
        f"sealed {label} question population changed",
    )
    indexed: dict[int, Mapping[str, Any]] = {}
    question_ids: set[str] = set()
    for expected_ordinal, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"sealed {label} row changed type")
        _require(
            type(row.get("ordinal")) is int
            and row.get("ordinal") == expected_ordinal,
            f"sealed {label} ordinal order changed",
        )
        question_id = row.get("question_id")
        _require(
            type(question_id) is str
            and bool(question_id)
            and question_id not in question_ids,
            f"sealed {label} question identity changed",
        )
        question_ids.add(question_id)
        indexed[expected_ordinal] = row
    return indexed


def _validate_parent_binding_graph(
    compact: Mapping[str, Any],
    v2_selection: Mapping[str, Any],
    v7_selection: Mapping[str, Any],
    *,
    v7_root: Path,
) -> None:
    """Require the fixed parent bytes to describe one transitive lineage."""

    bindings = compact.get("bindings")
    v2_bindings = v2_selection.get("bindings")
    v7_bindings = v7_selection.get("bindings")
    v2_implementation = v2_selection.get("implementation")
    _require(
        isinstance(bindings, Mapping)
        and isinstance(v2_bindings, Mapping)
        and isinstance(v7_bindings, Mapping)
        and isinstance(v2_implementation, Mapping),
        "sealed parent binding graph changed type",
    )
    shared = (
        "compiled_catalog_sha256",
        "population_identity_sha256",
        "probes_sha256",
        "source_root_relative_path",
        "v7_output_relative_path",
        "v7_selection_sha256",
    )
    _require(
        all(bindings.get(key) == v2_bindings.get(key) for key in shared)
        and bindings.get("v2_implementation_sha256")
        == v2_implementation.get("sha256")
        and v2_bindings.get("v7_bindings") == v7_bindings
        and v7_bindings.get("population_identity_sha256")
        == typed.EXPECTED_POPULATION_SHA256
        and _bound_repository_path(
            v2_bindings.get("v7_output_relative_path"),
            label="v2-bound v7 output root",
        )
        == v7_root.resolve(),
        "sealed v3/v2/v7 transitive bindings diverged",
    )


def _sealed_parent_bundle(
    compact: Mapping[str, Any],
    *,
    v2_selection: dict[str, Any],
    v2_selection_sha256: str,
    v7_selection: dict[str, Any],
    v7_selection_sha256: str,
) -> v3._ParentBundle:  # noqa: SLF001
    bindings = compact.get("bindings")
    _require(isinstance(bindings, Mapping), "sealed v3 bindings changed type")
    expected = {
        "v2_selection_sha256": EXPECTED_V2_SELECTION_SHA256,
        "v2_runtime_sha256": v3.EXPECTED_V2_RUNTIME_SHA256,
        "v2_run_manifest_sha256": v3.EXPECTED_V2_RUN_MANIFEST_SHA256,
        "v2_replay_sha256": v3.EXPECTED_V2_REPLAY_SHA256,
        "v7_selection_sha256": EXPECTED_V7_SELECTION_SHA256,
        "population_identity_sha256": typed.EXPECTED_POPULATION_SHA256,
    }
    _require(
        all(bindings.get(key) == value for key, value in expected.items()),
        "sealed v3 parent digest bindings changed",
    )
    source_root = _bound_repository_path(
        bindings.get("source_root_relative_path"), label="source root"
    )
    return v3._ParentBundle(  # noqa: SLF001
        v2_selection=v2_selection,
        v2_selection_sha256=v2_selection_sha256,
        v2_runtime_sha256=str(bindings["v2_runtime_sha256"]),
        v2_run_manifest_sha256=str(bindings["v2_run_manifest_sha256"]),
        v2_replay_sha256=str(bindings["v2_replay_sha256"]),
        v7_selection=v7_selection,
        v7_selection_sha256=v7_selection_sha256,
        parent_root=source_root,
        # The composer does not inspect probe/catalog payloads.  Their exact
        # digests remain transitively authenticated by the fixed v3 bytes.
        probes={},
        probes_sha256=str(bindings["probes_sha256"]),
        catalog={},
        catalog_sha256=str(bindings["compiled_catalog_sha256"]),
    )


def _load_sealed_v3_rows(
    v3_root: Path, *, ordinals: Sequence[int]
) -> tuple[dict[int, dict[str, Any]], str]:
    """Materialize only requested v3 arms from hash-pinned parent bytes.

    This deliberately does not call :func:`v3._load_selection`: that loader
    replays historical implementation identities which are not part of this
    successor assay.  The immutable v3/v2/v7 selection bytes are authenticated
    first; current pure composition is then accepted only when its compact row
    is byte-semantically identical to the sealed v3 row.
    """

    wanted = tuple(sorted(ordinals))
    _require(
        all(
            type(value) is int and 0 <= value < EXPECTED_QUESTION_COUNT
            for value in wanted
        )
        and len(wanted) == len(set(wanted)),
        "requested v3 ordinals changed",
    )
    compact, compact_sha = _read_fixed_selection(
        v3_root / v3.SELECTION_NAME,
        expected_sha256=EXPECTED_V3_SELECTION_SHA256,
        expected_format=v3.SELECTION_FORMAT,
        expected_status="sealed_gold_blind_source_seed_hybrid_full100",
        label="v3",
    )
    bindings = compact.get("bindings")
    _require(isinstance(bindings, Mapping), "sealed v3 bindings changed type")
    v2_root = _bound_repository_path(
        bindings.get("v2_output_relative_path"), label="v2 output root"
    )
    v7_root = _bound_repository_path(
        bindings.get("v7_output_relative_path"), label="v7 output root"
    )
    v2_selection, v2_sha = _read_fixed_selection(
        v2_root / v2.SELECTION_NAME,
        expected_sha256=EXPECTED_V2_SELECTION_SHA256,
        expected_format=v2.SELECTION_FORMAT,
        expected_status="sealed_gold_blind_assertion_projection_full100_assay",
        label="v2",
    )
    v7_selection, v7_sha = _read_fixed_selection(
        v7_root / v7.SELECTION_NAME,
        expected_sha256=EXPECTED_V7_SELECTION_SHA256,
        expected_format=v7.SELECTION_FORMAT,
        expected_status="sealed_gold_blind_locked_full100_adaptive_source_surplus_v1",
        label="v7",
    )
    _validate_parent_binding_graph(
        compact,
        v2_selection,
        v7_selection,
        v7_root=v7_root,
    )
    compact_rows = _locked_rows(compact, label="v3")
    v2_rows = _locked_rows(v2_selection, label="v2")
    v7_rows = _locked_rows(v7_selection, label="v7")
    parent = _sealed_parent_bundle(
        compact,
        v2_selection=v2_selection,
        v2_selection_sha256=v2_sha,
        v7_selection=v7_selection,
        v7_selection_sha256=v7_sha,
    )
    materialized: dict[int, dict[str, Any]] = {}
    for ordinal in wanted:
        compact_row = compact_rows[ordinal]
        v2_row = v2_rows[ordinal]
        v7_row = v7_rows[ordinal]
        _require(
            all(
                compact_row.get(key) == v2_row.get(key) == v7_row.get(key)
                for key in _ROW_IDENTITY_KEYS
            ),
            "sealed v3/v2/v7 row identities diverged",
        )
        fallback_arm = v7_row.get("arms", {}).get("a3_protected_union")
        _require(isinstance(fallback_arm, Mapping), "sealed v7 arm changed type")
        dated_question = typed._extract_dated_question(fallback_arm)  # noqa: SLF001
        _require(
            quote_sha256(dated_question) == compact_row["prompt_question_sha256"],
            "sealed parent dated-question binding changed",
        )
        expected_row, _timing, arm = v3._compose_question(  # noqa: SLF001
            v2_row=v2_row,
            v7_row=v7_row,
            dated_question=dated_question,
            parent=parent,
        )
        _require(
            expected_row == compact_row,
            "current pure composition differs from sealed v3 bytes",
        )
        row = copy.deepcopy(dict(compact_row))
        row["arms"] = {"a3_protected_union": arm}
        materialized[ordinal] = row
    full100._assert_gold_free_rows(list(materialized.values()))  # noqa: SLF001
    return materialized, compact_sha


def _build_resident_index(context: object, namespace: object) -> tuple[FullStoreWindowIndex, dict[str, Any]]:
    namespace_id = str(getattr(namespace, "namespace_id"))
    store = getattr(context, "store_dirs_by_namespace")[namespace_id]
    database_sha = getattr(context, "database_sha256_by_namespace")[namespace_id]
    store_receipt = getattr(namespace, "combined_store_receipt_sha256")
    started = time.perf_counter_ns()
    with Database(store / "memory.db", read_only=True) as database:
        cache_started = time.perf_counter_ns()
        cache = cache_namespace_partitions(
            database,
            namespace,
            source_database_sha256=database_sha,
            source_store_receipt_sha256=store_receipt,
        )
        cache_ns = time.perf_counter_ns() - cache_started
    index_started = time.perf_counter_ns()
    index = build_full_store_window_index(cache)
    index_ns = time.perf_counter_ns() - index_started
    return index, {
        "cache_build_ns": cache_ns,
        "content_row_count": len(index.rows),
        "namespace_id": namespace_id,
        "sentence_window_count": len(index.windows),
        "total_ns": time.perf_counter_ns() - started,
        "window_index_build_ns": index_ns,
        "window_index_receipt_sha256": index.receipt_sha256,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]], timings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    applicable = [row for row in rows if row["applicable"] is True]
    closed = [
        row for row in applicable if row["bridge"] is not None and row["bridge"]["closed"] is True
    ]
    warm = [int(row["total_ns"]) for row in timings if int(row["bridge_ns"]) > 0]
    return {
        "admissible_replacement_count": sum(
            row["admissible_replacement"] is True for row in rows
        ),
        "applicable_count": len(applicable),
        "closed_frontier_count": len(closed),
        "gate_selected_count": len(rows),
        "warm_applicable_mean_ns": (
            None if not warm else statistics.fmean(warm)
        ),
        "warm_applicable_p95_ns": _percentile95(warm),
    }


def construct(
    *,
    v3_root: Path,
    gate_root: Path,
    retrieval_path: Path,
    store_root: Path,
    output_root: Path,
) -> tuple[str, str]:
    """Construct and seal gold-blind results plus non-deterministic timings."""

    for name in (CONSTRUCTION_NAME, RUNTIME_NAME, SCORE_NAME):
        _require(
            not (output_root / name).exists(),
            f"refusing to overwrite existing {name}",
        )
    total_started = time.perf_counter_ns()
    preparation_started = time.perf_counter_ns()
    gate, gate_sha, gate_replay_sha = _read_gate(gate_root)
    selected_gate = _gate_rows(gate)
    source_by_ordinal, selection_sha = _load_sealed_v3_rows(
        v3_root, ordinals=tuple(selected_gate)
    )
    prepared: dict[int, tuple[str, str, Mapping[str, Any]]] = {}
    for ordinal, gate_row in sorted(selected_gate.items()):
        source = source_by_ordinal[ordinal]
        _require(
            source.get("question_id") == gate_row.get("question_id"),
            "gate/v3 question identity changed",
        )
        arm = source["arms"]["a3_protected_union"]
        dated_question = typed._extract_dated_question(arm)  # noqa: SLF001
        bundle = adapt_hot_v3_arm(
            dated_question,
            arm,
            selection_sha256=selection_sha,
            provider_packet=source["provider_packet"],
        )
        # This call is intentional and load-bearing: never substitute the
        # capped provider input here.
        local_input = bundle.local_inventory.operator_input()
        prepared[ordinal] = (
            str(source["question_id"]),
            dated_question,
            local_input,
        )
    preparation_ns = time.perf_counter_ns() - preparation_started

    context_started = time.perf_counter_ns()
    context = load_locked_query_expansion_context(
        retrieval_path,
        store_root=store_root,
        expected_retrieval_sha256=EXPECTED_RETRIEVAL_SHA256,
    )
    context_ns = time.perf_counter_ns() - context_started
    population_by_question = {
        row.source.packet.question_id: row for row in context.population.rows
    }
    by_namespace: dict[str, list[int]] = defaultdict(list)
    preflight_rows: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for ordinal, (question_id, _question, local_input) in prepared.items():
        applicable = operator_first_numeric_frontier_applicable(
            local_input,
            supported_domains=EXTENDED_SUPPORTED_DOMAINS,
        )
        if applicable:
            namespace = population_by_question[question_id].namespace
            by_namespace[namespace.namespace_id].append(ordinal)
        else:
            row, timing = assay_numeric_row(
                ordinal=ordinal,
                question_id=question_id,
                local_operator_input=local_input,
                index=None,
            )
            preflight_rows[ordinal] = (row, timing)

    results = dict(preflight_rows)
    lifecycle: list[dict[str, Any]] = []
    for namespace_id in sorted(by_namespace):
        namespace = population_by_question[
            prepared[by_namespace[namespace_id][0]][0]
        ].namespace
        index, index_timing = _build_resident_index(context, namespace)
        lifecycle.append(index_timing)
        try:
            for ordinal in sorted(by_namespace[namespace_id]):
                question_id, _question, local_input = prepared[ordinal]
                results[ordinal] = assay_numeric_row(
                    ordinal=ordinal,
                    question_id=question_id,
                    local_operator_input=local_input,
                    index=index,
                )
        finally:
            del index
            gc.collect()
    rows = [results[ordinal][0] for ordinal in sorted(results)]
    timings = [results[ordinal][1] for ordinal in sorted(results)]
    aggregate = _aggregate(rows, timings)
    construction_body = {
        "aggregate": aggregate,
        "format": FORMAT,
        "gate_construction_sha256": gate_sha,
        "gate_replay_sha256": gate_replay_sha,
        "gold_loaded": False,
        "implementation": _implementation_identity(),
        "new_provider_calls": 0,
        "policy_id": POLICY_ID,
        "questions": rows,
        "retained_transformer_token_state_bytes": 0,
        "status": "sealed_gold_blind_fast_v3_full_store_numeric",
        "v3_selection_sha256": selection_sha,
    }
    assert_gold_blind(construction_body, path="hot_v3_full_store_numeric")
    construction_sha = hot._atomic_write_json(  # noqa: SLF001
        output_root / CONSTRUCTION_NAME, construction_body
    )
    runtime_body = {
        "authenticated_namespace_count": len(context.store_dirs_by_namespace),
        "construction_sha256": construction_sha,
        "format": RUNTIME_FORMAT,
        "gold_loaded": False,
        "namespace_index_timings": lifecycle,
        "new_provider_calls": 0,
        "preparation_ns": preparation_ns,
        "question_timings": timings,
        "resident_index_namespace_count": len(lifecycle),
        "retained_transformer_token_state_bytes": 0,
        "sealed_store_context_load_ns": context_ns,
        "total_ns": time.perf_counter_ns() - total_started,
    }
    assert_gold_blind(runtime_body, path="hot_v3_full_store_numeric_runtime")
    runtime_sha = hot._atomic_write_json(  # noqa: SLF001
        output_root / RUNTIME_NAME, runtime_body
    )
    print(
        f"Hot-v3 full-store numeric: applicable={aggregate['applicable_count']}/"
        f"{aggregate['gate_selected_count']}; closed={aggregate['closed_frontier_count']}; "
        f"admissible={aggregate['admissible_replacement_count']}; "
        f"construction={construction_sha}; runtime={runtime_sha}",
        flush=True,
    )
    return construction_sha, runtime_sha


def _load_constructed(output_root: Path) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    construction, construction_sha = hot._read_json_artifact(  # noqa: SLF001
        output_root / CONSTRUCTION_NAME
    )
    runtime, runtime_sha = hot._read_json_artifact(  # noqa: SLF001
        output_root / RUNTIME_NAME
    )
    _require(construction.get("format") == FORMAT, "construction format changed")
    _require(runtime.get("format") == RUNTIME_FORMAT, "runtime format changed")
    _require(
        construction.get("gold_loaded") is False
        and construction.get("new_provider_calls") == 0
        and runtime.get("gold_loaded") is False
        and runtime.get("new_provider_calls") == 0
        and runtime.get("construction_sha256") == construction_sha,
        "construction/runtime firebreak changed",
    )
    return construction, construction_sha, runtime, runtime_sha


def _rows_by_ordinal(rows: object, *, expected_count: int, label: str) -> dict[int, Mapping[str, Any]]:
    _require(isinstance(rows, list) and len(rows) == expected_count, f"{label} changed")
    result = {int(row["ordinal"]): row for row in rows if isinstance(row, Mapping)}
    _require(len(result) == expected_count, f"{label} ordinals changed")
    return result


def score(
    *,
    dataset: Path,
    split_manifest: Path,
    output_root: Path,
    v3_score_path: Path,
    v7_judgments_path: Path,
) -> str:
    """Join gold only after construction and report exact replacement delta."""

    construction, construction_sha, _runtime, runtime_sha = _load_constructed(
        output_root
    )
    samples, _identities, population = full100._load_population(  # noqa: SLF001
        dataset, split_manifest
    )
    _require(
        population.get("population_identity_sha256") == typed.EXPECTED_POPULATION_SHA256,
        "score population changed",
    )
    questions = full100._flatten_questions(samples)  # noqa: SLF001
    question_by_ordinal = {index: row for index, row in enumerate(questions)}
    v3_score = typed._read_expected(  # noqa: SLF001
        v3_score_path, typed.EXPECTED_V3_SCORE_SHA256, label="hot-v3 score"
    )
    v7 = typed._read_expected(  # noqa: SLF001
        v7_judgments_path,
        typed.EXPECTED_V7_JUDGMENTS_SHA256,
        label="adaptive-v7 judgments",
    )
    v3_rows = _rows_by_ordinal(v3_score.get("questions"), expected_count=100, label="v3 score")
    v7_rows = _rows_by_ordinal(v7.get("rows"), expected_count=100, label="v7 judgments")
    scored_rows: list[dict[str, Any]] = []
    net_delta = 0
    for result in construction["questions"]:
        ordinal = int(result["ordinal"])
        benchmark = question_by_ordinal[ordinal]
        v7_row = v7_rows[ordinal]
        v3_row = v3_rows[ordinal]
        reference = str(benchmark.answer)
        decision = result.get("decision")
        prediction = "" if decision is None else str(decision["prediction"])
        admissible = result["admissible_replacement"] is True
        replacement_correct = bool(
            admissible
            and normalize_answer(prediction) == normalize_answer(reference)
        )
        baseline_correct = bool(v7_row["correct"])
        delta = (
            int(replacement_correct) - int(baseline_correct)
            if admissible
            else 0
        )
        net_delta += delta
        evidence = v3_row["effective_hybrid"]
        _require(
            result["question_id"] == benchmark.question_id == v7_row["question_id"]
            and result["question_sha256"] == quote_sha256(benchmark.dated_question)
            and v7_row["reference_sha256"] == quote_sha256(reference),
            "post-hoc score row identity changed",
        )
        scored_rows.append(
            {
                "admissible_replacement": admissible,
                "baseline_correct": baseline_correct,
                "exact_delta": delta,
                "ordinal": ordinal,
                "prediction_sha256": quote_sha256(prediction),
                "question_id": benchmark.question_id,
                "reference_sha256": quote_sha256(reference),
                "replacement_lexical_exact": replacement_correct,
                "result_row_receipt_sha256": result["row_receipt_sha256"],
                "v3_all_gold_source_ids_reached": evidence.get(
                    "all_gold_source_ids_reached"
                ),
                "v3_literal_answer": evidence.get("literal_answer"),
            }
        )
    baseline_correct_count = int(v7["correct"])
    body = {
        "baseline_correct_count": baseline_correct_count,
        "construction_sha256": construction_sha,
        "effective_lexical_exact_correct_count": baseline_correct_count + net_delta,
        "format": SCORE_FORMAT,
        "gold_loaded": True,
        "net_exact_gain": net_delta,
        "new_provider_calls": 0,
        "questions": scored_rows,
        "runtime_sha256": runtime_sha,
        "status": "post_construction_gold_join",
        "v3_score_sha256": typed.EXPECTED_V3_SCORE_SHA256,
        "v7_judgments_sha256": typed.EXPECTED_V7_JUDGMENTS_SHA256,
    }
    digest = hot._atomic_write_json(output_root / SCORE_NAME, body)  # noqa: SLF001
    print(
        f"Post-hoc exact overlay: {baseline_correct_count} -> "
        f"{baseline_correct_count + net_delta} ({net_delta:+d}); score={digest}",
        flush=True,
    )
    return digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-root", type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument("--gate-root", type=Path, default=DEFAULT_GATE_ROOT)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("construct")
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--dataset", type=Path, required=True)
    score_parser.add_argument("--split-manifest", type=Path, default=full100.DEFAULT_SPLIT)
    score_parser.add_argument("--v3-score", type=Path, default=DEFAULT_V3_ROOT / v3.SCORE_NAME)
    score_parser.add_argument(
        "--v7-judgments",
        type=Path,
        default=DEFAULT_V7_ROOT / "answer-judgments.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = args.output_root.resolve()
    if args.command == "construct":
        construct(
            v3_root=args.v3_root.resolve(),
            gate_root=args.gate_root.resolve(),
            retrieval_path=args.retrieval.resolve(),
            store_root=args.store_root.resolve(),
            output_root=output_root,
        )
    elif args.command == "score":
        score(
            dataset=args.dataset.resolve(),
            split_manifest=args.split_manifest.resolve(),
            output_root=output_root,
            v3_score_path=args.v3_score.resolve(),
            v7_judgments_path=args.v7_judgments.resolve(),
        )
    else:  # pragma: no cover
        raise AssertionError(f"unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
