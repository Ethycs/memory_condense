#!/usr/bin/env python3
"""Audit and minimally exercise provider-free temporal lanes on hot fast-v3.

The default ``map`` command does not replay successor-sensitive construction
code.  It authenticates immutable sealed artifacts, checks their row receipts,
and rebinds rows only when the question id plus both question digests agree.
Gold-labelled source coverage is an optional, explicitly post-hoc phase.

``live`` rebuilds one namespace index and runs the temporal specialist on one
or more questions in that namespace.  It makes no provider calls and reports
cold index construction separately from repeated warm query latency.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import statistics
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from memory_condense.persistence.db import Database  # noqa: E402
from tools import run_locked_query_guided_scan as guided_scan_cli  # noqa: E402
from tools import run_reduced_second_read_retrieval_assay as second_read_cli  # noqa: E402
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json  # noqa: E402
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.full_store_slot_closure import (  # noqa: E402
    build_full_store_window_index,
)
from tools.matched_eval.query_guided_scan import cache_namespace_partitions  # noqa: E402
from tools.matched_eval.temporal_event_reconciler import (  # noqa: E402
    reconcile_temporal_events,
)
from tools.matched_eval.temporal_insufficiency_specialist import (  # noqa: E402
    adapt_temporal_insufficiency_to_typed_contribution,
    scan_temporal_insufficiency_specialist,
)
from tools.matched_eval.typed_memory_final_arm import (  # noqa: E402
    completion_validation_contract,
)
from tools.matched_eval.typed_operator_adapter import (  # noqa: E402
    merge_typed_evidence_contributions,
)
from tools.matched_eval.typed_operator_executor import execute_typed_operator  # noqa: E402


FORMAT = "memory-condense-hot-v3-temporal-provider-free-probe-v1"
EXPECTED_SELECTION_SHA256 = (
    "0e8027d3150bdf8335ad7a83d1a8bb4a5551312444fea2d5a4e820ece05eb3b7"
)
EXPECTED_FULL_STORE_SHA256 = (
    "044e60f308287dda4d87106646e4cc56f0e96d513b2bfd03a7473da9994ef5c4"
)
EXPECTED_LEGACY_SPECIALIST_SHA256 = (
    "92d01388b4e6ed4c73851dfe47e70c4a1dabda2f5da9b73606853a66cfcd4e86"
)
TEMPORAL_ACTION = "temporal_full_store"


class TemporalProbeError(MatchedEvalContractError):
    """Raised when a sealed input cannot be safely rebound."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise TemporalProbeError(message)


def _verified_receipt(row: Mapping[str, Any], key: str, label: str) -> str:
    body = dict(row)
    declared = require_sha256(body.pop(key, None), label)
    _require(identity_sha256(body) == declared, f"{label} receipt changed")
    return declared


def _rows_by_ordinal(payload: Mapping[str, Any], label: str) -> dict[int, dict[str, Any]]:
    rows = payload.get("questions")
    _require(type(rows) is list and all(type(row) is dict for row in rows), f"{label} rows changed")
    result = {int(row["ordinal"]): dict(row) for row in rows}
    _require(len(result) == len(rows), f"{label} ordinals repeat")
    return result


def _iter_json_array(path: Path, *, chunk_chars: int = 1 << 20) -> Iterator[dict[str, Any]]:
    """Incrementally decode a large top-level JSON array without a 277MB load."""

    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    eof = False
    with path.open("r", encoding="utf-8") as stream:
        while True:
            if not eof:
                piece = stream.read(chunk_chars)
                eof = piece == ""
                buffer += piece
            cursor = 0
            while cursor < len(buffer) and buffer[cursor].isspace():
                cursor += 1
            if not started:
                if cursor == len(buffer) and not eof:
                    buffer = ""
                    continue
                _require(cursor < len(buffer) and buffer[cursor] == "[", "dataset is not a JSON array")
                cursor += 1
                started = True
            while True:
                while cursor < len(buffer) and (buffer[cursor].isspace() or buffer[cursor] == ","):
                    cursor += 1
                if cursor < len(buffer) and buffer[cursor] == "]":
                    return
                if cursor == len(buffer):
                    break
                try:
                    value, stop = decoder.raw_decode(buffer, cursor)
                except json.JSONDecodeError:
                    if eof:
                        raise TemporalProbeError("dataset JSON ended inside a row")
                    break
                _require(type(value) is dict, "dataset row is not an object")
                yield value
                cursor = stop
            buffer = buffer[cursor:]
            if eof:
                _require(not buffer.strip(), "dataset has trailing undecoded bytes")
                return


def _target_rows(dataset: Path, question_ids: set[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _iter_json_array(dataset):
        question_id = row.get("question_id")
        if question_id in question_ids:
            _require(question_id not in result, "dataset question id repeats")
            result[str(question_id)] = row
            if len(result) == len(question_ids):
                break
    _require(set(result) == question_ids, "dataset is missing a gated question")
    return result


def _answer_source_ids(row: Mapping[str, Any]) -> set[str]:
    values = row.get("answer_session_ids")
    _require(type(values) is list and all(type(value) is str and value for value in values), "answer sessions changed")
    return set(values)


def _source_tail(source_id: object) -> str | None:
    if type(source_id) is not str or not source_id:
        return None
    return source_id.split("::", 1)[-1]


def _source_coverage(bindings: Sequence[Mapping[str, Any]], target_ids: set[str]) -> dict[str, Any]:
    reached = sorted(target_ids & {tail for row in bindings if (tail := _source_tail(row.get("source_id")))})
    return {
        "all_target_sources_reached": set(reached) == target_ids,
        "reached_target_source_count": len(reached),
        "reached_target_source_ids": reached,
        "target_source_count": len(target_ids),
    }


def _normalized_text(value: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE))


def _evidence_posthoc(
    candidates: Sequence[Mapping[str, Any]],
    bindings: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    target_ids = _answer_source_ids(target)
    coverage = _source_coverage(bindings, target_ids)
    raw_answer = target.get("answer")
    if type(raw_answer) is str:
        answers = [raw_answer]
    elif type(raw_answer) is list:
        answers = [str(value) for value in raw_answer if value is not None]
    elif raw_answer is None:
        answers = []
    else:
        answers = [str(raw_answer)]
    _require(all(value for value in answers), "reference answer contains an empty value")
    normalized_answers = [_normalized_text(value) for value in answers]
    binding_by_id = {str(row.get("candidate_id")): row for row in bindings}
    matching_quotes = []
    for candidate in candidates:
        binding = binding_by_id.get(str(candidate.get("candidate_id")))
        if binding is None or _source_tail(binding.get("source_id")) not in target_ids:
            continue
        quote = require_text(candidate.get("quote"), "posthoc candidate quote")
        if normalized_answers and all(value in _normalized_text(quote) for value in normalized_answers):
            matching_quotes.append(quote)
    return {
        **coverage,
        "answer_literal_in_target_source_quote": bool(matching_quotes),
        "answer_literal_matching_quote_sha256s": [quote_sha256(value) for value in matching_quotes],
        "reference_answer_sha256s": [quote_sha256(value) for value in answers],
    }


def _validate_local_bindings(method: Mapping[str, Any], *, label: str) -> list[dict[str, Any]]:
    bindings = method.get("local_bindings")
    _require(type(bindings) is list and all(type(row) is dict for row in bindings), f"{label} bindings changed")
    for offset, binding in enumerate(bindings):
        _verified_receipt(binding, "receipt_sha256", f"{label} binding {offset}")
    return [dict(row) for row in bindings]


def _validate_full_store_row(row: Mapping[str, Any], *, ordinal: int) -> list[dict[str, Any]]:
    _verified_receipt(row, "row_receipt_sha256", f"full-store row {ordinal}")
    local = row.get("local_audit")
    provider = row.get("provider_projection")
    _require(type(local) is dict and type(provider) is dict, "full-store projection changed")
    bindings = local.get("bindings")
    candidates = provider.get("candidates")
    _require(
        type(bindings) is list and type(candidates) is list
        and all(type(value) is dict for value in (*bindings, *candidates)),
        "full-store candidate population changed",
    )
    by_id = {str(binding["candidate_id"]): binding for binding in bindings}
    _require(len(by_id) == len(bindings) == len(candidates), "full-store candidate/binding cardinality changed")
    for offset, binding in enumerate(bindings):
        _verified_receipt(binding, "receipt_sha256", f"full-store binding {ordinal}:{offset}")
    for candidate in candidates:
        binding = by_id.get(str(candidate.get("candidate_id")))
        _require(
            binding is not None
            and candidate.get("citation_binding_receipt_sha256") == binding.get("receipt_sha256")
            and candidate.get("quote_sha256") == binding.get("quote_sha256")
            and quote_sha256(require_text(candidate.get("quote"), "full-store quote")) == candidate.get("quote_sha256"),
            f"full-store candidate citation changed at {ordinal}",
        )
    return [dict(row) for row in bindings]


def _temporal_method(row: Mapping[str, Any]) -> dict[str, Any] | None:
    methods = row.get("methods")
    if type(methods) is not list:
        return None
    matches = [value for value in methods if type(value) is dict and value.get("mechanism_id") == "temporal_insufficiency_specialist_v1"]
    _require(len(matches) <= 1, "temporal specialist method repeats")
    return dict(matches[0]) if matches else None


def _without_posthoc(value: Any) -> Any:
    """Remove score-only fields while retaining the authenticated lane map."""

    if type(value) is dict:
        return {
            key: _without_posthoc(item)
            for key, item in value.items()
            if not key.startswith("posthoc_") and key not in {"old95_correct", "v7_correct"}
        }
    if type(value) is list:
        return [_without_posthoc(item) for item in value]
    return value


def _computed_temporal_is_admissible(
    *,
    prediction_source: object,
    operation: object,
    validation_contract: Mapping[str, Any],
    provider_input: Mapping[str, Any],
) -> tuple[bool, str | None]:
    """Apply the production boundary for a model-free reconciler answer.

    The validation contract is authoritative.  The embedded operator spec is
    only a compatibility fallback for older sealed plans.  In particular, a
    duration must not escape into a direct-content question merely because a
    temporal regular expression happened to match it.
    """

    answer_shape = validation_contract.get("answer_shape")
    if type(answer_shape) is not str:
        typed = provider_input.get("typed_evidence")
        operator = typed.get("operator_spec") if type(typed) is dict else None
        answer_shape = operator.get("answer_shape") if type(operator) is dict else None
    allowed_shapes = {
        "direct_duration": frozenset({"duration"}),
        "event_interval": frozenset({"duration"}),
        "event_order": frozenset({"direct", "ordered_list"}),
    }
    admissible = (
        prediction_source == "computed"
        and type(operation) is str
        and type(answer_shape) is str
        and answer_shape in allowed_shapes.get(operation, ())
    )
    return bool(admissible), answer_shape if type(answer_shape) is str else None


def build_map(
    *,
    selection_path: Path,
    compatibility_path: Path,
    scores_path: Path,
    full_store_path: Path,
    legacy_specialist_path: Path,
    legacy_terminal_path: Path | None,
    legacy_terminal_preflight_path: Path | None,
    legacy_terminal_construction_path: Path | None,
    dataset_path: Path | None,
) -> dict[str, Any]:
    selection = read_sealed_json(selection_path)
    compatibility = read_sealed_json(compatibility_path)
    scores = read_sealed_json(scores_path)
    full_store = read_sealed_json(full_store_path)
    legacy = read_sealed_json(legacy_specialist_path)
    _require(selection.sha256 == EXPECTED_SELECTION_SHA256, "sealed fast-v3 selection changed")
    _require(full_store.sha256 == EXPECTED_FULL_STORE_SHA256, "sealed full-store input changed")
    _require(legacy.sha256 == EXPECTED_LEGACY_SPECIALIST_SHA256, "sealed legacy specialist changed")
    _require(compatibility.payload.get("v3_selection_sha256") == selection.sha256, "compatibility lost fast-v3 parent")
    _require(scores.payload.get("construction_sha256") == compatibility.sha256, "scores lost compatibility parent")

    compat = _rows_by_ordinal(compatibility.payload, "compatibility")
    scored = _rows_by_ordinal(scores.payload, "scores")
    full = _rows_by_ordinal(full_store.payload, "full-store")
    legacy_rows = _rows_by_ordinal(legacy.payload, "legacy specialist")
    gate = tuple(
        ordinal for ordinal, row in sorted(compat.items())
        if isinstance(row.get("next_action"), dict)
        and row["next_action"].get("next_action") == TEMPORAL_ACTION
    )
    _require(len(gate) == 38, "temporal gate population changed")

    gold = _target_rows(dataset_path, {str(compat[o]["question_id"]) for o in gate}) if dataset_path else {}
    question_rows: list[dict[str, Any]] = []
    legacy_rebound = 0
    full_all_source = 0
    full_literal = 0
    legacy_all_source = 0
    legacy_literal = 0
    for ordinal in gate:
        fast = compat[ordinal]
        score = scored[ordinal]
        closure = full[ordinal]
        _require(fast.get("question_id") == closure.get("question_id"), f"full-store qid changed at {ordinal}")
        provider = closure.get("provider_projection")
        _require(type(provider) is dict, "full-store provider projection changed")
        dated = require_text(provider.get("dated_question"), "full-store dated question")
        operator = provider.get("operator_spec")
        _require(
            type(operator) is dict
            and quote_sha256(dated) == fast.get("prompt_question_sha256") == operator.get("question_sha256"),
            f"full-store fast-v3 question binding changed at {ordinal}",
        )
        full_bindings = _validate_full_store_row(closure, ordinal=ordinal)
        target = gold.get(str(fast["question_id"]))
        full_coverage = None
        if target is not None:
            full_coverage = _evidence_posthoc(provider.get("candidates", []), full_bindings, target)
            full_all_source += int(full_coverage["all_target_sources_reached"])
            full_literal += int(full_coverage["answer_literal_in_target_source_quote"])

        specialist_summary = None
        old = legacy_rows.get(ordinal)
        if old is not None:
            _verified_receipt(old, "question_receipt_sha256", f"legacy specialist row {ordinal}")
            method = _temporal_method(old)
            if method is not None:
                _verified_receipt(method, "method_receipt_sha256", f"legacy temporal method {ordinal}")
                _require(
                    old.get("question_id") == fast.get("question_id")
                    and old.get("dated_question_sha256") == fast.get("prompt_question_sha256")
                    and old.get("question_sha256") == fast.get("retrieval_query_sha256"),
                    f"legacy temporal row cannot rebind at {ordinal}",
                )
                legacy_rebound += 1
                bindings = _validate_local_bindings(method, label=f"legacy temporal {ordinal}")
                projection = method.get("provider_projection")
                _require(type(projection) is dict, "legacy temporal projection changed")
                bundle = projection.get("temporal_bundle")
                candidates = projection.get("candidates")
                _require(type(candidates) is list, "legacy temporal candidates changed")
                by_candidate = {str(value["candidate_id"]): value for value in candidates}
                coverage = None
                if target is not None:
                    coverage = _evidence_posthoc(candidates, bindings, target)
                    legacy_all_source += int(coverage["all_target_sources_reached"])
                    legacy_literal += int(coverage["answer_literal_in_target_source_quote"])
                ordered = []
                winner = None
                if type(bundle) is dict:
                    for candidate_id in bundle.get("ordered_candidate_ids", []):
                        candidate = by_candidate[str(candidate_id)]
                        ordered.append({"event_date": candidate.get("event_date"), "quote": candidate.get("quote"), "quote_sha256": candidate.get("quote_sha256")})
                    winner_id = bundle.get("winner_candidate_id")
                    if type(winner_id) is str:
                        candidate = by_candidate[winner_id]
                        winner = {"event_date": candidate.get("event_date"), "quote": candidate.get("quote"), "quote_sha256": candidate.get("quote_sha256")}
                specialist_summary = {
                    "artifact_sha256": legacy.sha256,
                    "bundle": None if type(bundle) is not dict else {
                        "ordered": ordered,
                        "population_count": bundle.get("population_count"),
                        "requested_cardinality": bundle.get("requested_cardinality"),
                        "route": bundle.get("route"),
                        "truncated": bundle.get("truncated"),
                        "winner": winner,
                    },
                    "candidate_count": len(candidates),
                    "posthoc_source_coverage": coverage,
                    "rebound": True,
                    "specialist_result_receipt_sha256": method.get("specialist_receipt_sha256"),
                    "terminal_answer_emitted": False,
                }

        question_rows.append({
            "format": f"{FORMAT}-mapped-question-v1",
            "full_store": {
                "artifact_sha256": full_store.sha256,
                "candidate_count": len(provider.get("candidates", [])),
                "posthoc_source_coverage": full_coverage,
                "rebound": True,
                "result_receipt_sha256": closure.get("result_receipt_sha256"),
                "terminal_answer_emitted": False,
            },
            "legacy_temporal_specialist": specialist_summary,
            "old95_correct": score.get("old95_correct"),
            "ordinal": ordinal,
            "posthoc_reference_answer": None if target is None else target.get("answer"),
            "prompt_question_sha256": fast.get("prompt_question_sha256"),
            "question_id": fast.get("question_id"),
            "retrieval_query_sha256": fast.get("retrieval_query_sha256"),
            "v7_correct": score.get("v7_correct"),
        })

    historical_rows: list[dict[str, Any]] = []
    computed = 0
    computed_exact = 0
    computed_gain = 0
    parent_dependent = 0
    if legacy_terminal_path is not None:
        terminal = read_sealed_json(legacy_terminal_path)
        terminal_rows = _rows_by_ordinal(terminal.payload, "legacy terminal")
        for ordinal in gate:
            row = terminal_rows.get(ordinal)
            if row is None or type(row.get("reconciliation")) is not dict:
                continue
            fast = compat[ordinal]
            score = scored[ordinal]
            resolution = row["reconciliation"]
            proof = resolution.get("proof")
            _require(
                row.get("question_id") == fast.get("question_id")
                and row.get("dated_question_sha256") == fast.get("prompt_question_sha256")
                and row.get("question_sha256") == fast.get("retrieval_query_sha256")
                and type(proof) is dict
                and proof.get("question_sha256") == fast.get("prompt_question_sha256"),
                f"legacy terminal question cannot rebind at {ordinal}",
            )
            source = resolution.get("prediction_source")
            exact = resolution.get("prediction_sha256") == score.get("reference_sha256")
            computed += int(source == "computed")
            computed_exact += int(source == "computed" and exact)
            computed_gain += int(source == "computed" and exact and not score.get("v7_correct"))
            parent_dependent += int(source in {"candidate", "parent"})
            historical_rows.append({
                "exact_reference_sha256_match": exact,
                "ordinal": ordinal,
                "prediction": resolution.get("prediction"),
                "prediction_source": source,
                "v7_correct": score.get("v7_correct"),
            })

    compute_only: dict[str, Any] | None = None
    if legacy_terminal_preflight_path is not None or legacy_terminal_construction_path is not None:
        _require(
            legacy_terminal_preflight_path is not None and legacy_terminal_construction_path is not None,
            "compute-only replay requires both preflight and construction",
        )
        preflight = read_sealed_json(legacy_terminal_preflight_path)
        terminal_construction = read_sealed_json(legacy_terminal_construction_path)
        _require(
            preflight.payload.get("construction_artifact_sha256") == terminal_construction.sha256,
            "terminal preflight lost its construction parent",
        )
        plans = {
            int(row["ordinal"]): dict(row)
            for row in preflight.payload.get("physical_prompt_rows", [])
            if type(row) is dict
        }
        construction_rows = _rows_by_ordinal(terminal_construction.payload, "terminal construction")
        replay_rows = []
        attempted = supported = admissible = exact = gain = 0
        for ordinal in gate:
            plan = plans.get(ordinal)
            if plan is None or type(plan.get("provider_input")) is not dict:
                continue
            attempted += 1
            plan_receipt = _verified_receipt(plan, "answer_plan_receipt_sha256", f"answer plan {ordinal}")
            source_row = construction_rows[ordinal]
            construction_receipt = _verified_receipt(
                source_row, "question_receipt_sha256", f"terminal construction row {ordinal}"
            )
            fast = compat[ordinal]
            score = scored[ordinal]
            provider_input = plan["provider_input"]
            dated_question = require_text(provider_input.get("dated_question"), "compute-only dated question")
            _require(
                plan.get("question_id") == source_row.get("question_id") == fast.get("question_id")
                and plan.get("dated_question_sha256") == source_row.get("dated_question_sha256") == fast.get("prompt_question_sha256")
                and plan.get("question_sha256") == source_row.get("question_sha256") == fast.get("retrieval_query_sha256")
                and plan.get("construction_question_receipt_sha256") == construction_receipt
                and quote_sha256(dated_question) == fast.get("prompt_question_sha256"),
                f"compute-only lineage cannot rebind at {ordinal}",
            )
            validation = plan.get("validation_contract")
            allowed = plan.get("allowed_handle_ids")
            _require(
                type(validation) is dict and type(allowed) is list and bool(allowed),
                f"compute-only contract changed at {ordinal}",
            )
            result = reconcile_temporal_events(
                dated_question=dated_question,
                candidate_prediction="__provider_free_compute_only_candidate__",
                parent_prediction="__provider_free_compute_only_parent__",
                provider_input=provider_input,
                validation_contract=validation,
                allowed_handle_ids=tuple(allowed),
                source_receipt_sha256=plan_receipt,
            )
            is_computed = result is not None and result.prediction_source == "computed"
            supported += int(is_computed)
            shape_safe, answer_shape = _computed_temporal_is_admissible(
                prediction_source=None if result is None else result.prediction_source,
                operation=None if result is None else result.operation,
                validation_contract=validation,
                provider_input=provider_input,
            )
            admissible += int(shape_safe)
            is_exact = bool(shape_safe and result is not None and quote_sha256(result.prediction) == score.get("reference_sha256"))
            exact += int(is_exact)
            gain += int(is_exact and not score.get("v7_correct"))
            if result is not None:
                replay_rows.append({
                    "admissible_answer_shape": shape_safe,
                    "answer_shape": answer_shape,
                    "exact_reference_sha256_match": is_exact,
                    "operation": result.operation,
                    "ordinal": ordinal,
                    "prediction": result.prediction,
                    "prediction_source": result.prediction_source,
                    "v7_correct": score.get("v7_correct"),
                })
        compute_only = {
            "admissible_terminal_count": admissible,
            "attempted_plan_count": attempted,
            "construction_sha256": terminal_construction.sha256,
            "exact_reference_count": exact,
            "exact_posthoc_gain_over_v7_count": gain,
            "preflight_sha256": preflight.sha256,
            "rows": replay_rows,
            "supported_before_answer_shape_guard_count": supported,
        }

    construction = {
        "artifact_bindings": {
            "compatibility_sha256": compatibility.sha256,
            "full_store_sha256": full_store.sha256,
            "legacy_specialist_sha256": legacy.sha256,
            "scores_sha256": scores.sha256,
            "selection_sha256": selection.sha256,
            "terminal_construction_sha256": None if compute_only is None else compute_only["construction_sha256"],
            "terminal_preflight_sha256": None if compute_only is None else compute_only["preflight_sha256"],
        },
        "format": f"{FORMAT}-gold-free-construction-v1",
        "full_store_rebound_count": len(gate),
        "gold_loaded": False,
        "integration_contract": {
            "lane_output_types": {
                "full_store_slot_closure": "evidence_and_exact-literal-absence-witness_only",
                "local_temporal_pair": "keep-parent_validation_only",
                "temporal_event_reconciler": "optional_terminal_for_narrow_deterministic_operations",
                "temporal_insufficiency_specialist": "evidence_bundle_only",
            },
            "live_api": [
                "scan_temporal_insufficiency_specialist(index, dated_question)",
                "adapt_temporal_insufficiency_to_typed_contribution(result, handle_start=..., group_start=...)",
                "merge_typed_evidence_contributions(result.operator_spec, contributions)",
                "execute_typed_operator(result.operator_spec, packet)",
                "reconcile_temporal_events(...) for narrow unique-event arithmetic only",
            ],
            "rebind_requires": [
                "sealed artifact digest sidecar",
                "ordinal and question_id identity",
                "dated prompt and retrieval-query digest identity",
                "question/method/local-binding receipt validity",
            ],
            "terminal_admissibility": {
                "bundle_only": "not_terminal",
                "direct_relative_lookup": "requires a unique answer-bearing event on the derived date; current bundle scan does not extract the answer object",
                "latest_or_complete_set": "requires relevant-frontier closure not supplied by bounded scans",
                "local_temporal_pair": "requires and can only retain an existing nonempty parent prediction",
                "reconciler_computed": "requires a declared compatible answer shape, unique directly cited dated operands, and a narrow supported operation",
            },
        },
        "legacy_temporal_bundle_rebound_count": legacy_rebound,
        "physical_provider_calls": 0,
        "questions": [_without_posthoc(row) for row in question_rows],
        "retained_transformer_token_state_bytes": 0,
        "temporal_gate_count": len(gate),
    }
    # The detailed report may contain target labels below, but the construction
    # projection is independently checked before those labels are attached.
    assert_gold_blind(construction, path="hot_v3_temporal_probe_construction")
    return {
        "construction": construction,
        "format": FORMAT,
        "historical_terminal_rebind": {
            "computed_exact_reference_count": computed_exact,
            "computed_posthoc_gain_over_v7_count": computed_gain,
            "computed_terminal_count": computed,
            "note": "Historical proof is question-bound, but direct integration still requires its underlying typed provider-input/validation-contract lineage or a live rebuild.",
            "parent_or_candidate_dependent_count": parent_dependent,
            "rows": historical_rows,
        },
        "authenticated_compute_only_replay": compute_only,
        "posthoc": None if dataset_path is None else {
            "dataset_path": str(dataset_path),
            "full_store_all_target_source_count": full_all_source,
            "full_store_answer_literal_in_target_source_quote_count": full_literal,
            "legacy_temporal_all_target_source_count": legacy_all_source,
            "legacy_temporal_answer_literal_in_target_source_quote_count": legacy_literal,
            "question_count": len(gate),
            "questions": question_rows,
        },
        "provider_calls": 0,
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * quantile + 0.999999)))
    return ordered[index]


def build_live(
    *,
    full_store_path: Path,
    retrieval: Path,
    store_root: Path,
    query_parent_output_root: Path,
    ordinals: Sequence[int],
    warm_runs: int,
) -> dict[str, Any]:
    _require(bool(ordinals) and len(set(ordinals)) == len(ordinals), "live ordinals changed")
    _require(warm_runs > 0, "warm run count must be positive")
    full_store = read_sealed_json(full_store_path)
    _require(full_store.sha256 == EXPECTED_FULL_STORE_SHA256, "sealed full-store input changed")
    full = _rows_by_ordinal(full_store.payload, "full-store")
    rows = [full[int(ordinal)] for ordinal in ordinals]
    namespace_ids = {
        binding.get("namespace_id")
        for row in rows
        for binding in row["local_audit"]["bindings"][:1]
    }
    _require(len(namespace_ids) == 1, "live ordinals must share one namespace")
    namespace_id = require_sha256(next(iter(namespace_ids)), "live namespace")
    args = argparse.Namespace(
        expected_query_parent_preflight_sha256=guided_scan_cli.EXPECTED_PARENT_PREFLIGHT_SHA256,
        expected_retrieval_sha256=guided_scan_cli.EXPECTED_RETRIEVAL_SHA256,
        query_parent_output_root=query_parent_output_root,
        retrieval=retrieval,
        store_root=store_root,
    )
    started = time.perf_counter()
    context = second_read_cli._scoped_guided_context(args, namespace_id)  # noqa: SLF001
    context_seconds = time.perf_counter() - started
    started = time.perf_counter()
    with Database(context.store_dir / "memory.db", read_only=True) as database:
        cache = cache_namespace_partitions(
            database,
            context.namespace,
            source_database_sha256=context.database_sha256,
            source_store_receipt_sha256=context.namespace.combined_store_receipt_sha256,
        )
    cache_seconds = time.perf_counter() - started
    started = time.perf_counter()
    index = build_full_store_window_index(cache)
    index_seconds = time.perf_counter() - started
    sealed_cache = next(
        value for value in full_store.payload["cache_receipts"]
        if value.get("namespace_id") == namespace_id
    )
    _require(
        sealed_cache.get("cache_receipt_sha256") == cache.cache_receipt_sha256
        and sealed_cache.get("window_index_receipt_sha256") == index.receipt_sha256,
        "live index does not match sealed full-store cache",
    )

    output_rows = []
    for ordinal, source in zip(ordinals, rows, strict=True):
        provider = source["provider_projection"]
        dated_question = require_text(provider.get("dated_question"), "live dated question")
        started = time.perf_counter()
        first = scan_temporal_insufficiency_specialist(index, dated_question)
        first_ms = (time.perf_counter() - started) * 1000
        durations = []
        for _ in range(warm_runs):
            started = time.perf_counter()
            replay = scan_temporal_insufficiency_specialist(index, dated_question)
            durations.append((time.perf_counter() - started) * 1000)
            _require(replay.receipt.receipt_sha256 == first.receipt.receipt_sha256, "warm specialist replay changed")

        contribution = adapt_temporal_insufficiency_to_typed_contribution(
            first, handle_start=900_001, group_start=900_001
        )
        packet = merge_typed_evidence_contributions(first.operator_spec, (contribution,))
        execution = execute_typed_operator(first.operator_spec, packet)
        contract = completion_validation_contract(packet, execution, dated_question=dated_question)
        sentinel_candidate = "__provider_free_compute_only_candidate__"
        sentinel_parent = "__provider_free_compute_only_parent__"
        resolution = reconcile_temporal_events(
            dated_question=dated_question,
            candidate_prediction=sentinel_candidate,
            parent_prediction=sentinel_parent,
            provider_input={"typed_evidence": packet.provider_projection()},
            validation_contract=contract,
            allowed_handle_ids=tuple(row.handle_id for row in packet.handles),
            source_receipt_sha256=first.receipt.receipt_sha256,
        )
        computed = resolution if resolution is not None and resolution.prediction_source == "computed" else None
        bundle = first.temporal_bundle
        by_id = {row.candidate_id: row for row in first.candidates}
        output_rows.append({
            "bundle": None if bundle is None else {
                "ordered": [
                    {"event_date": by_id[candidate_id].event_date, "quote": by_id[candidate_id].quote}
                    for candidate_id in bundle.ordered_candidate_ids
                ],
                "population_count": bundle.population_count,
                "requested_cardinality": bundle.requested_cardinality,
                "route": bundle.route,
                "truncated": bundle.truncated,
                "winner": None if bundle.winner_candidate_id is None else {
                    "event_date": by_id[bundle.winner_candidate_id].event_date,
                    "quote": by_id[bundle.winner_candidate_id].quote,
                },
            },
            "candidate_count": len(first.candidates),
            "computed_terminal": None if computed is None else computed.projection(),
            "executor_prediction": execution.prediction,
            "executor_reason": execution.reason,
            "executor_status": execution.status.value,
            "first_query_ms": first_ms,
            "ordinal": int(ordinal),
            "question_id": source.get("question_id"),
            "specialist_receipt_sha256": first.receipt.receipt_sha256,
            "warm_latency_ms": {
                "maximum": max(durations),
                "median": statistics.median(durations),
                "minimum": min(durations),
                "p95": _percentile(durations, 0.95),
                "runs": len(durations),
            },
        })
    result = {
        "cold_latency_seconds": {
            "cache_materialization": cache_seconds,
            "index_construction": index_seconds,
            "sealed_context_validation": context_seconds,
            "total": context_seconds + cache_seconds + index_seconds,
        },
        "format": f"{FORMAT}-live-v1",
        "full_store_artifact_sha256": full_store.sha256,
        "namespace_id": namespace_id,
        "physical_provider_calls": 0,
        "questions": output_rows,
        "retained_transformer_token_state_bytes": 0,
        "window_index_receipt_sha256": index.receipt_sha256,
    }
    del index, cache, context
    gc.collect()
    return result


def _publish_or_print(payload: dict[str, Any], output: Path | None) -> None:
    if output is None:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        print()
        return
    artifact, created = publish_sealed_json(output, payload)
    print(json.dumps({"created": created, "output": str(output), "sha256": artifact.sha256}, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    mapping = commands.add_parser("map")
    mapping.add_argument("--selection", type=Path, required=True)
    mapping.add_argument("--compatibility", type=Path, required=True)
    mapping.add_argument("--scores", type=Path, required=True)
    mapping.add_argument("--full-store-input", type=Path, required=True)
    mapping.add_argument("--legacy-specialist", type=Path, required=True)
    mapping.add_argument("--legacy-terminal", type=Path)
    mapping.add_argument("--legacy-terminal-preflight", type=Path)
    mapping.add_argument("--legacy-terminal-construction", type=Path)
    mapping.add_argument("--dataset", type=Path)
    mapping.add_argument("--output", type=Path)
    live = commands.add_parser("live")
    live.add_argument("--full-store-input", type=Path, required=True)
    live.add_argument("--retrieval", type=Path, required=True)
    live.add_argument("--store-root", type=Path, required=True)
    live.add_argument("--query-parent-output-root", type=Path, required=True)
    live.add_argument("--ordinal", type=int, action="append", required=True)
    live.add_argument("--warm-runs", type=int, default=10)
    live.add_argument("--output", type=Path)
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args()
    if args.command == "map":
        payload = build_map(
            selection_path=args.selection,
            compatibility_path=args.compatibility,
            scores_path=args.scores,
            full_store_path=args.full_store_input,
            legacy_specialist_path=args.legacy_specialist,
            legacy_terminal_path=args.legacy_terminal,
            legacy_terminal_preflight_path=args.legacy_terminal_preflight,
            legacy_terminal_construction_path=args.legacy_terminal_construction,
            dataset_path=args.dataset,
        )
    else:
        payload = build_live(
            full_store_path=args.full_store_input,
            retrieval=args.retrieval,
            store_root=args.store_root,
            query_parent_output_root=args.query_parent_output_root,
            ordinals=tuple(args.ordinal),
            warm_runs=args.warm_runs,
        )
    _publish_or_print(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
