"""Provider-free typed-operator compatibility assay over sealed hot-v3.

``run`` authenticates the sealed source-seed hybrid selection, converts each
effective arm into the common typed evidence contract, and executes the local
closure/consensus/operator stack.  ``replay`` reconstructs those diagnostics
and requires byte identity.  Neither phase opens benchmark gold or provider
responses.  Only ``score`` may attach reference-derived lexical diagnostics
and previously sealed evaluation observations.
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval._answer_normalization import f1_score, normalize_answer
from tools import assay_hot_retrieval_1m as hot
from tools import assay_hot_retrieval_full100 as full100
from tools import assay_hot_retrieval_source_seed_hybrid_full100 as v3
from tools.matched_eval.contracts import assert_gold_blind, identity_sha256
from tools.matched_eval.fast_v3_selective_escalation_gate import (
    AnswerState,
    NextAction,
    OperatorState,
    evaluate_fast_v3_escalation,
    seal_fast_v3_escalation_state,
)
from tools.matched_eval.operator_first_numeric_policy import (
    execute_operator_first_numeric_policy,
)
from tools.matched_eval.typed_operator_executor import (
    ExecutionStatus,
    ExecutorKind,
    build_evidence_consensus,
    build_slot_closure,
    execute_typed_operator,
)


EXPECTED_V3_SELECTION_SHA256 = (
    "0e8027d3150bdf8335ad7a83d1a8bb4a5551312444fea2d5a4e820ece05eb3b7"
)
EXPECTED_V3_SCORE_SHA256 = (
    "c7a009eca11188b2123afc02bbc585d4766157c6fff106d102b155886355a9b8"
)
EXPECTED_V7_JUDGMENTS_SHA256 = (
    "5ca56b60817875546f19afad5c8ee9a017a261fc0b498bd4a580e46a074ad10c"
)
EXPECTED_OLD95_MERGE_SHA256 = (
    "aa210a8bba87897d7fc8e3f4e2a7e71cbcc929fa4eeac6ce5cbf6ef56567c952"
)
EXPECTED_POPULATION_SHA256 = v3.EXPECTED_POPULATION_SHA256
EXPECTED_QUESTION_COUNT = v3.EXPECTED_QUESTION_COUNT

CONSTRUCTION_FORMAT = "memory-condense-hot-v3-typed-operator-full100-compatibility-v3"
REPLAY_FORMAT = "memory-condense-hot-v3-typed-operator-full100-replay-v3"
SCORE_FORMAT = "memory-condense-hot-v3-typed-operator-full100-score-v3"
POLICY_ID = "sealed-hot-v3-common-typed-operator-numeric-overlay-v3"

DEFAULT_V3_ROOT = v3.DEFAULT_OUTPUT_ROOT
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-hot-v3-typed-operator-full100-compatibility-v3-20260906"
)
DEFAULT_V7_ROOT = Path(
    "eval_results/longmemeval-1m-hot-retrieval-adaptive-full100-validation-20260905"
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _primary_checkout_root() -> Path:
    root = _repository_root()
    return root.parent.parent if root.parent.name == ".worktrees" else root


DEFAULT_OLD95_MERGE = _primary_checkout_root() / (
    "eval_results/matched_eval_100/locked-semantic-global-terminal-full100-"
    "terra-answer-v5-r1/policy-v5-r3/differential-sol-judge-v1/merge-v1-r1/"
    "policy-v5-differential-sol-judge-merge-v1.json"
)
CONSTRUCTION_NAME = "compatibility.json"
REPLAY_NAME = "replay.json"
SCORE_NAME = "scores.json"
_CONSTRUCTION_KEYS = frozenset(
    {
        "aggregate",
        "format",
        "gold_fields_present",
        "implementation",
        "policy_id",
        "population_identity_sha256",
        "provider_calls",
        "question_count",
        "question_population_sha256",
        "questions",
        "qwen_calls",
        "retained_transformer_token_state_bytes",
        "status",
        "v3_selection_sha256",
    }
)


def _implementation_identity() -> dict[str, Any]:
    root = _repository_root()
    relative_paths = (
        "tools/assay_hot_v3_typed_operator_full100.py",
        "tools/matched_eval/contracts.py",
        "tools/matched_eval/fast_v3_selective_escalation_gate.py",
        "tools/matched_eval/hot_v3_typed_operator_adapter.py",
        "tools/matched_eval/operator_first_numeric_policy.py",
        "tools/matched_eval/typed_numeric_semantics.py",
        "tools/matched_eval/typed_operator_adapter.py",
        "tools/matched_eval/typed_operator_executor.py",
        "tools/matched_eval/typed_operator_spec.py",
    )
    files = {path: file_sha256(root / path) for path in relative_paths}
    return {
        "files": files,
        "format": "memory-condense-hot-v3-typed-operator-implementation-v3",
        "sha256": identity_sha256(
            [{"path": path, "sha256": files[path]} for path in relative_paths]
        ),
    }


def _projection(value: object, *, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        projected = dict(value)
    else:
        projector = getattr(value, "projection", None)
        if not callable(projector):
            raise TypeError(f"{label} does not expose a projection")
        projected = projector()
    if not isinstance(projected, dict):
        raise TypeError(f"{label} projection must be an object")
    return projected


def _extract_dated_question(arm: Mapping[str, Any]) -> str:
    messages = arm.get("provider_messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or not isinstance(messages[1], Mapping)
        or messages[1].get("role") != "user"
        or not isinstance(messages[1].get("content"), str)
    ):
        raise ValueError("hot-v3 arm omitted the frozen two-message QA prompt")
    content = str(messages[1]["content"])
    marker = "\n\nQuestion: "
    suffix = "\nShort answer:"
    if marker not in content or not content.endswith(suffix):
        raise ValueError("hot-v3 arm changed the frozen QA question envelope")
    question = content.rsplit(marker, 1)[1][: -len(suffix)]
    if not question or question.strip() != question:
        raise ValueError("hot-v3 dated question is not exact non-empty text")
    return question


def _adapt(
    dated_question: str,
    arm: Mapping[str, Any],
    *,
    selection_sha256: str,
    provider_packet: Mapping[str, Any],
) -> object:
    # Local import keeps the lifecycle scaffold importable while the adapter is
    # developed independently; execution still fails closed if it is absent.
    from tools.matched_eval.hot_v3_typed_operator_adapter import adapt_hot_v3_arm

    return adapt_hot_v3_arm(
        dated_question,
        arm,
        selection_sha256=selection_sha256,
        provider_packet=provider_packet,
    )


def _evidence_coverage(packet: object) -> dict[str, Any]:
    handles = tuple(getattr(packet, "handles"))
    bindings = tuple(getattr(packet, "local_bindings"))
    items = tuple(getattr(packet, "items"))
    rejected = tuple(getattr(packet, "rejected_items"))
    frontier = getattr(packet, "frontier")
    binding_by_handle = {row.handle_id: row for row in bindings}
    if tuple(row.handle_id for row in handles) != tuple(binding_by_handle):
        raise ValueError("typed packet lost its one-to-one handle bindings")
    cited_handles = {
        row.handle_id for row in bindings if int(row.citation_char_count) > 0
    }
    included = tuple(row for row in items if row.included)
    cited_items = tuple(
        row for row in included if set(row.handle_ids) & cited_handles
    )
    represented = set(frontier.represented_handle_ids)
    source_groups = {row.source_group_handle for row in handles}
    represented_groups = {
        row.source_group_handle for row in handles if row.handle_id in represented
    }
    return {
        "available_handle_count": len(frontier.available_handle_ids),
        "binding_count": len(bindings),
        "citation_binding_count": len(cited_handles),
        "citation_char_count": sum(int(row.citation_char_count) for row in bindings),
        "all_surviving_included_items_citation_backed": (
            len(cited_items) == len(included)
        ),
        "citation_backed_surviving_included_item_count": len(cited_items),
        "included_item_count": len(included),
        "item_count": len(items),
        "omitted_handle_count": len(frontier.omitted_handle_ids),
        "rejected_item_count": len(rejected),
        "represented_handle_count": len(frontier.represented_handle_ids),
        "represented_source_group_count": len(represented_groups),
        "source_group_count": len(source_groups),
        "unresolved_slot_count": len(frontier.unresolved_slot_ids),
    }


def _numeric_policy_is_admissible(
    *, spec: object, packet: object, decision: object, applicable: bool
) -> bool:
    """Keep bounded closed-world answers out of the effective prediction."""

    return bool(
        applicable
        and getattr(decision, "status") is ExecutionStatus.SUPPORTED
        and (
            not bool(getattr(spec, "requires_complete_frontier"))
            or bool(getattr(getattr(packet, "frontier"), "closed"))
        )
    )


def _diagnostic_row(
    source_row: Mapping[str, Any], *, selection_sha256: str
) -> dict[str, Any]:
    arm_root = source_row.get("arms")
    packet_receipt = source_row.get("provider_packet")
    if (
        not isinstance(arm_root, Mapping)
        or not isinstance(arm_root.get("a3_protected_union"), Mapping)
        or not isinstance(packet_receipt, Mapping)
        or not isinstance(packet_receipt.get("receipt_sha256"), str)
    ):
        raise ValueError("materialized hot-v3 row omitted its effective arm receipt")
    arm = arm_root["a3_protected_union"]
    dated_question = _extract_dated_question(arm)
    if quote_sha256(dated_question) != source_row.get("prompt_question_sha256"):
        raise ValueError("materialized hot-v3 dated question changed")
    bundle = _adapt(
        dated_question,
        arm,
        selection_sha256=selection_sha256,
        provider_packet=packet_receipt,
    )
    spec = getattr(bundle, "operator_spec")
    packet = getattr(bundle, "evidence_packet")
    local_inventory = getattr(bundle, "local_inventory")
    provider_input = getattr(bundle, "provider_input")
    if not isinstance(provider_input, Mapping):
        raise TypeError("hot-v3 typed adapter provider input must be an object")
    audit = _projection(getattr(bundle, "audit"), label="adapter audit")
    bundle_projection = _projection(bundle, label="adapter bundle")
    closure = build_slot_closure(spec, packet)
    consensus = build_evidence_consensus(packet)
    execution = execute_typed_operator(spec, packet)
    numeric_policy = execute_operator_first_numeric_policy(bundle.provider_input)
    local_operator_input = local_inventory.operator_input()
    local_numeric_policy = execute_operator_first_numeric_policy(local_operator_input)
    if execution.executor is ExecutorKind.NONE:
        if execution.status is not ExecutionStatus.NON_DETERMINISTIC:
            raise ValueError("non-applicable typed executor emitted a terminal status")
        applicable = False
    else:
        applicable = True
    numeric_policy_supported = _numeric_policy_is_admissible(
        spec=spec,
        packet=packet,
        decision=numeric_policy,
        applicable=applicable,
    )
    local_numeric_policy_supported = _numeric_policy_is_admissible(
        spec=spec,
        packet=local_inventory,
        decision=local_numeric_policy,
        applicable=applicable,
    )
    if local_numeric_policy_supported:
        effective_status = local_numeric_policy.status
        effective_prediction = local_numeric_policy.prediction
        effective_numeric_result = local_numeric_policy.numeric_result
        effective_handles = local_numeric_policy.used_handle_ids
        effective_receipt = local_numeric_policy.receipt_sha256
        effective_source = "local_operator_first_numeric_policy"
        effective_frontier = local_inventory.frontier
    elif numeric_policy_supported:
        effective_status = numeric_policy.status
        effective_prediction = numeric_policy.prediction
        effective_numeric_result = numeric_policy.numeric_result
        effective_handles = numeric_policy.used_handle_ids
        effective_receipt = numeric_policy.receipt_sha256
        effective_source = "operator_first_numeric_policy"
        effective_frontier = packet.frontier
    else:
        effective_status = execution.status
        effective_prediction = execution.prediction
        effective_numeric_result = execution.numeric_result
        effective_handles = execution.used_handle_ids
        effective_receipt = execution.receipt_sha256
        effective_source = "generic_typed_operator"
        effective_frontier = packet.frontier
    for label, value in (
        ("provider_input", provider_input),
        ("adapter_audit", audit),
        ("adapter_bundle", bundle_projection),
        ("local_inventory", local_inventory.projection()),
        ("local_operator_input", local_operator_input),
    ):
        assert_gold_blind(value, path=label)
    coverage = _evidence_coverage(packet)
    system_handles = set(local_inventory.system_excluded_handle_ids)
    capped_frontier_handles = {
        *packet.frontier.available_handle_ids,
        *packet.frontier.represented_handle_ids,
        *packet.frontier.omitted_handle_ids,
    }
    if system_handles & capped_frontier_handles or any(
        system_handles & set(item.handle_ids) for item in packet.items
    ):
        raise ValueError("system-row evidence entered the provider-visible packet")
    eligible_packed_handles = len(local_inventory.handles) - len(system_handles)
    capacity_omitted = bundle.audit.provider_capacity_omitted_handle_count
    provider_preserved_handles = len(packet.frontier.represented_handle_ids)
    if provider_preserved_handles + capacity_omitted != eligible_packed_handles:
        raise ValueError("provider capacity audit lost its packed-handle partition")
    local_coverage = {
        "capped_handle_count": len(packet.handles),
        "capped_item_count": len(packet.items),
        "capped_operator_token_proxy": packet.provider_payload_token_proxy,
        "local_handle_count": len(local_inventory.handles),
        "local_item_count": len(local_inventory.items),
        "local_operator_token_proxy": local_inventory.local_operator_token_proxy,
        "operator_item_delta": len(local_inventory.items) - len(packet.items),
        "packed_eligible_handle_count": eligible_packed_handles,
        "provider_capacity_omission_rate": (
            0.0
            if eligible_packed_handles == 0
            else capacity_omitted / eligible_packed_handles
        ),
        "provider_capacity_omitted_handle_count": capacity_omitted,
        "provider_packed_handle_preservation_rate": (
            1.0
            if eligible_packed_handles == 0
            else provider_preserved_handles / eligible_packed_handles
        ),
        "provider_preserved_packed_handle_count": provider_preserved_handles,
        "role_filtered_handle_count": len(
            local_inventory.role_filtered_handle_ids
        ),
        "system_excluded_handle_count": len(system_handles),
    }
    operator_state = (
        OperatorState(effective_status.value)
        if applicable
        else OperatorState.NOT_RUN
    )
    unresolved_slots = (
        ()
        if effective_status is ExecutionStatus.SUPPORTED
        else tuple(
            dict.fromkeys((*closure.missing_slot_ids, *closure.conflicted_slot_ids))
        )
    )
    unresolved_slot_set = set(unresolved_slots)
    if spec.required_slots:
        anchor_items = tuple(
            item
            for item in local_inventory.items
            if item.included
            and bool(set(item.supported_slot_ids) & unresolved_slot_set)
        )
        anchor_basis = "included_local_item_supports_unresolved_slot"
    else:
        anchor_items = tuple(
            item for item in local_inventory.items if item.included
        )
        anchor_basis = "included_packed_retrieval_item_no_declared_slots"
    escalation_state = seal_fast_v3_escalation_state(
        provider_packet_sha256=str(packet_receipt["receipt_sha256"]),
        parent_prediction_sha256=None,
        answer_state=AnswerState.NOT_RUN,
        typed_spec_receipt_sha256=spec.receipt_sha256,
        typed_frontier_receipt_sha256=effective_frontier.receipt_sha256,
        frontier_closed=effective_frontier.closed,
        frontier_truncated=effective_frontier.truncated,
        unresolved_slot_ids=unresolved_slots,
        operator_execution_status=operator_state,
        completed_actions=((NextAction.PACKED_OPERATOR,) if applicable else ()),
        available_actions=(
            NextAction.SOURCE_LOCAL_EPISODE,
            NextAction.NUMERIC_FULL_STORE,
            NextAction.PROFILE_FULL_STORE,
            NextAction.TEMPORAL_FULL_STORE,
            NextAction.SEMANTIC_GLOBAL,
        ),
        authenticated_anchor_available=bool(anchor_items),
    )
    next_action = evaluate_fast_v3_escalation(dated_question, escalation_state)
    row = {
        "adapter_audit_sha256": identity_sha256(audit),
        "adapter_bundle_sha256": identity_sha256(bundle_projection),
        "candidate_arbiter_required": not applicable,
        "deterministic_executor_applicable": applicable,
        "evidence_coverage": coverage,
        "evidence_packet": {
            "conflict_policy": packet.conflict_policy.value,
            "frontier": packet.frontier.projection(),
            "output_token_reserve": packet.output_token_reserve,
            "provider_payload_mode": packet.provider_payload_mode.value,
            "provider_payload_token_proxy": packet.provider_payload_token_proxy,
            "receipt_sha256": packet.receipt_sha256,
            "sealed_input_artifact_sha256s": list(
                packet.sealed_input_artifact_sha256s
            ),
        },
        "evidence_consensus": consensus.projection(),
        "effective_packed_execution": {
            "numeric_result": effective_numeric_result,
            "prediction": effective_prediction,
            "receipt_sha256": effective_receipt,
            "source": effective_source,
            "status": effective_status.value,
            "used_handle_ids": list(effective_handles),
        },
        "escalation_state": escalation_state,
        "escalation_anchor": {
            "authenticated_anchor_count": len(anchor_items),
            "basis": anchor_basis,
        },
        "local_inventory": {
            "operator_input_sha256": identity_sha256(local_operator_input),
            "projection_sha256": identity_sha256(local_inventory.projection()),
            "provider_use_forbidden": True,
            "receipt_sha256": local_inventory.receipt_sha256,
        },
        "local_operator_first_numeric": local_numeric_policy.projection(),
        "local_operator_first_numeric_admissible": local_numeric_policy_supported,
        "local_vs_capped": local_coverage,
        "next_action": next_action.projection(),
        "operator_execution": execution.projection(),
        "operator_first_numeric": numeric_policy.projection(),
        "operator_first_numeric_admissible": numeric_policy_supported,
        "operator_spec": spec.projection(),
        "ordinal": int(source_row["ordinal"]),
        "prompt_question_sha256": str(source_row["prompt_question_sha256"]),
        "provider_input_sha256": identity_sha256(dict(provider_input)),
        "provider_packet_receipt_sha256": str(packet_receipt["receipt_sha256"]),
        "question_id": str(source_row["question_id"]),
        "retrieval_query_sha256": str(source_row["retrieval_query_sha256"]),
        "slot_closure": closure.projection(),
    }
    assert_gold_blind(row, path="typed_operator_diagnostic")
    return row


def _count_map(values: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_slots = sum(len(row["operator_spec"]["required_slots"]) for row in rows)
    bound_slots = sum(len(row["slot_closure"]["bound_slot_ids"]) for row in rows)
    missing_slots = sum(len(row["slot_closure"]["missing_slot_ids"]) for row in rows)
    conflicted_slots = sum(
        len(row["slot_closure"]["conflicted_slot_ids"]) for row in rows
    )
    packed_eligible_handles = sum(
        int(row["local_vs_capped"]["packed_eligible_handle_count"])
        for row in rows
    )
    provider_preserved_handles = sum(
        int(row["local_vs_capped"]["provider_preserved_packed_handle_count"])
        for row in rows
    )
    capacity_omitted_handles = sum(
        int(row["local_vs_capped"]["provider_capacity_omitted_handle_count"])
        for row in rows
    )
    by_style: dict[str, dict[str, int]] = {}
    for style in sorted({str(row["operator_spec"]["style"]) for row in rows}):
        group = [row for row in rows if row["operator_spec"]["style"] == style]
        by_style[style] = {
            "candidate_arbiter_required_count": sum(
                row["candidate_arbiter_required"] is True for row in group
            ),
            "all_surviving_included_items_citation_backed_question_count": sum(
                row["evidence_coverage"][
                    "all_surviving_included_items_citation_backed"
                ] is True
                for row in group
            ),
            "closure_sufficient_count": sum(
                row["slot_closure"]["sufficient"] is True for row in group
            ),
            "deterministic_executor_applicable_count": sum(
                row["deterministic_executor_applicable"] is True for row in group
            ),
            "execution_supported_count": sum(
                row["effective_packed_execution"]["status"]
                == ExecutionStatus.SUPPORTED.value
                for row in group
            ),
            "generic_execution_supported_count": sum(
                row["operator_execution"]["status"]
                == ExecutionStatus.SUPPORTED.value
                for row in group
            ),
            "capped_operator_first_numeric_admissible_count": sum(
                row["operator_first_numeric_admissible"] is True
                for row in group
            ),
            "local_operator_first_numeric_admissible_count": sum(
                row["local_operator_first_numeric_admissible"] is True
                for row in group
            ),
            "question_count": len(group),
        }
    generic_statuses = [str(row["operator_execution"]["status"]) for row in rows]
    effective_statuses = [
        str(row["effective_packed_execution"]["status"]) for row in rows
    ]
    return {
        "bounded_frontier_count": sum(
            row["evidence_packet"]["frontier"]["mode"] == "bounded"
            for row in rows
        ),
        "by_execution_reason": _count_map(
            [str(row["operator_execution"]["reason"]) for row in rows]
        ),
        "by_effective_execution_status": _count_map(effective_statuses),
        "by_execution_status": _count_map(generic_statuses),
        "by_executor": _count_map(
            [str(row["operator_execution"]["executor"]) for row in rows]
        ),
        "by_next_action": _count_map(
            [str(row["next_action"]["next_action"]) for row in rows]
        ),
        "by_operator_first_numeric_mode": _count_map(
            [str(row["operator_first_numeric"]["mode"]) for row in rows]
        ),
        "by_operator_first_numeric_reason": _count_map(
            [str(row["operator_first_numeric"]["reason"]) for row in rows]
        ),
        "by_operator_first_numeric_status": _count_map(
            [str(row["operator_first_numeric"]["status"]) for row in rows]
        ),
        "by_local_operator_first_numeric_reason": _count_map(
            [str(row["local_operator_first_numeric"]["reason"]) for row in rows]
        ),
        "by_local_operator_first_numeric_status": _count_map(
            [str(row["local_operator_first_numeric"]["status"]) for row in rows]
        ),
        "by_style": by_style,
        "candidate_arbiter_required_count": sum(
            row["candidate_arbiter_required"] is True for row in rows
        ),
        "all_surviving_included_items_citation_backed_question_count": sum(
            row["evidence_coverage"][
                "all_surviving_included_items_citation_backed"
            ] is True
            for row in rows
        ),
        "closed_frontier_count": sum(
            row["evidence_packet"]["frontier"]["closed"] is True for row in rows
        ),
        "closure_sufficient_count": sum(
            row["slot_closure"]["sufficient"] is True for row in rows
        ),
        "complete_frontier_required_count": sum(
            row["operator_spec"]["requires_complete_frontier"] is True
            for row in rows
        ),
        "conflicted_slot_count": conflicted_slots,
        "deterministic_executor_applicable_count": sum(
            row["deterministic_executor_applicable"] is True for row in rows
        ),
        "execution_supported_count": effective_statuses.count(
            ExecutionStatus.SUPPORTED.value
        ),
        "generic_execution_supported_count": generic_statuses.count(
            ExecutionStatus.SUPPORTED.value
        ),
        "missing_slot_count": missing_slots,
        "nonempty_prediction_count": sum(
            bool(row["effective_packed_execution"]["prediction"]) for row in rows
        ),
        "capped_operator_first_numeric_admissible_count": sum(
            row["operator_first_numeric_admissible"] is True
            for row in rows
        ),
        "capped_operator_first_numeric_raw_supported_count": sum(
            row["operator_first_numeric"]["status"]
            == ExecutionStatus.SUPPORTED.value
            for row in rows
        ),
        "local_operator_first_numeric_admissible_count": sum(
            row["local_operator_first_numeric_admissible"] is True
            for row in rows
        ),
        "local_operator_first_numeric_raw_supported_count": sum(
            row["local_operator_first_numeric"]["status"]
            == ExecutionStatus.SUPPORTED.value
            for row in rows
        ),
        "local_operator_support_gain_over_capped_count": sum(
            row["local_operator_first_numeric_admissible"] is True
            and row["operator_first_numeric_admissible"] is False
            for row in rows
        ),
        "obligation_bound_slot_count": bound_slots,
        "obligation_closure_rate": (
            None if total_slots == 0 else bound_slots / total_slots
        ),
        "obligation_total_slot_count": total_slots,
        "packed_eligible_handle_count": packed_eligible_handles,
        "provider_capacity_omission_rate": (
            0.0
            if packed_eligible_handles == 0
            else capacity_omitted_handles / packed_eligible_handles
        ),
        "provider_capacity_omitted_handle_count": capacity_omitted_handles,
        "provider_packed_handle_preservation_rate": (
            1.0
            if packed_eligible_handles == 0
            else provider_preserved_handles / packed_eligible_handles
        ),
        "provider_preserved_packed_handle_count": provider_preserved_handles,
        "question_count": len(rows),
        "represented_source_group_count": sum(
            int(row["evidence_coverage"]["represented_source_group_count"])
            for row in rows
        ),
        "typed_item_count": sum(
            int(row["evidence_coverage"]["item_count"]) for row in rows
        ),
    }


def _build_body(selection: Mapping[str, Any], selection_sha256: str) -> dict[str, Any]:
    if selection_sha256 != EXPECTED_V3_SELECTION_SHA256:
        raise ValueError("sealed hot-v3 selection changed")
    bindings = selection.get("bindings")
    source_rows = selection.get("questions")
    if (
        not isinstance(bindings, Mapping)
        or bindings.get("population_identity_sha256") != EXPECTED_POPULATION_SHA256
        or not isinstance(source_rows, list)
        or len(source_rows) != EXPECTED_QUESTION_COUNT
    ):
        raise ValueError("sealed hot-v3 population changed")
    rows = [
        _diagnostic_row(row, selection_sha256=selection_sha256)
        for row in source_rows
    ]
    if [row["ordinal"] for row in rows] != list(range(EXPECTED_QUESTION_COUNT)):
        raise ValueError("typed diagnostic population is not in locked ordinal order")
    if len({row["question_id"] for row in rows}) != EXPECTED_QUESTION_COUNT:
        raise ValueError("typed diagnostic question IDs are not unique")
    aggregate = _aggregate(rows)
    implementation = _implementation_identity()
    body = {
        "aggregate": aggregate,
        "format": CONSTRUCTION_FORMAT,
        "gold_fields_present": False,
        "implementation": implementation,
        "policy_id": POLICY_ID,
        "population_identity_sha256": EXPECTED_POPULATION_SHA256,
        "provider_calls": 0,
        "question_count": EXPECTED_QUESTION_COUNT,
        "question_population_sha256": hashlib.sha256(
            hot._canonical_json_bytes(rows)  # noqa: SLF001
        ).hexdigest(),
        "questions": rows,
        "qwen_calls": 0,
        "retained_transformer_token_state_bytes": 0,
        "status": "sealed_gold_blind_provider_free_typed_operator_compatibility",
        "v3_selection_sha256": selection_sha256,
    }
    if (
        aggregate["deterministic_executor_applicable_count"]
        + aggregate["candidate_arbiter_required_count"]
        != EXPECTED_QUESTION_COUNT
        or aggregate["execution_supported_count"]
        != aggregate["nonempty_prediction_count"]
    ):
        raise RuntimeError("typed compatibility aggregate failed its partition gate")
    assert_gold_blind(body, path="typed_operator_compatibility")
    return body


def _load_construction(output_root: Path) -> tuple[dict[str, Any], str]:
    body, digest = hot._read_json_artifact(  # noqa: SLF001
        output_root / CONSTRUCTION_NAME
    )
    rows = body.get("questions")
    if (
        frozenset(body) != _CONSTRUCTION_KEYS
        or body.get("format") != CONSTRUCTION_FORMAT
        or body.get("status")
        != "sealed_gold_blind_provider_free_typed_operator_compatibility"
        or body.get("v3_selection_sha256") != EXPECTED_V3_SELECTION_SHA256
        or body.get("population_identity_sha256") != EXPECTED_POPULATION_SHA256
        or body.get("question_count") != EXPECTED_QUESTION_COUNT
        or body.get("gold_fields_present") is not False
        or body.get("provider_calls") != 0
        or body.get("qwen_calls") != 0
        or body.get("retained_transformer_token_state_bytes") != 0
        or body.get("implementation") != _implementation_identity()
        or not isinstance(rows, list)
        or len(rows) != EXPECTED_QUESTION_COUNT
        or body.get("question_population_sha256")
        != hashlib.sha256(hot._canonical_json_bytes(rows)).hexdigest()  # noqa: SLF001
        or body.get("aggregate") != _aggregate(rows)
    ):
        raise ValueError("typed compatibility construction is invalid")
    assert_gold_blind(body, path="typed_operator_compatibility")
    return body, digest


def run(*, v3_root: Path, output_root: Path) -> str:
    construction_path = output_root / CONSTRUCTION_NAME
    if construction_path.exists():
        _body, digest = _load_construction(output_root)
        return digest
    for name in (REPLAY_NAME, SCORE_NAME):
        if (output_root / name).exists():
            raise FileExistsError("refusing a partial typed-operator output root")
    selection, selection_sha = v3._load_selection(v3_root)  # noqa: SLF001
    body = _build_body(selection, selection_sha)
    digest = hot._atomic_write_json(construction_path, body)  # noqa: SLF001
    _load_construction(output_root)
    print(
        "Hot-v3 typed compatibility published: "
        f"{digest}; applicable={body['aggregate']['deterministic_executor_applicable_count']}"
        f"/{EXPECTED_QUESTION_COUNT}; supported="
        f"{body['aggregate']['execution_supported_count']}/{EXPECTED_QUESTION_COUNT}",
        flush=True,
    )
    return digest


def _expected_replay_body(
    construction: Mapping[str, Any], construction_sha256: str
) -> dict[str, Any]:
    construction_bytes = hot._canonical_json_bytes(construction)  # noqa: SLF001
    if hashlib.sha256(construction_bytes).hexdigest() != construction_sha256:
        raise ValueError("replay construction digest changed")
    question_population_sha256 = construction["question_population_sha256"]
    return {
        "byte_identical": True,
        "construction_sha256": construction_sha256,
        "format": REPLAY_FORMAT,
        "gold_fields_present": False,
        "implementation_sha256": construction["implementation"]["sha256"],
        "population_identity_sha256": EXPECTED_POPULATION_SHA256,
        "provider_calls": 0,
        "question_count": EXPECTED_QUESTION_COUNT,
        "question_population_sha256": question_population_sha256,
        "qwen_calls": 0,
        "replayed_construction_sha256": construction_sha256,
        "replayed_question_population_sha256": question_population_sha256,
        "retained_transformer_token_state_bytes": 0,
        "status": "byte_identical_gold_blind_typed_operator_replay",
        "v3_selection_sha256": EXPECTED_V3_SELECTION_SHA256,
    }


def _load_replay(
    output_root: Path, *, construction_sha256: str
) -> tuple[dict[str, Any], str]:
    construction, actual_construction_sha = _load_construction(output_root)
    if actual_construction_sha != construction_sha256:
        raise ValueError("typed replay belongs to another construction")
    body, digest = hot._read_json_artifact(output_root / REPLAY_NAME)  # noqa: SLF001
    if body != _expected_replay_body(construction, construction_sha256):
        raise ValueError("typed compatibility replay is invalid")
    assert_gold_blind(body, path="typed_operator_replay")
    return body, digest


def replay(*, v3_root: Path, output_root: Path) -> str:
    expected, construction_sha = _load_construction(output_root)
    selection, selection_sha = v3._load_selection(v3_root)  # noqa: SLF001
    replayed = _build_body(selection, selection_sha)
    expected_bytes = hot._canonical_json_bytes(expected)  # noqa: SLF001
    replayed_bytes = hot._canonical_json_bytes(replayed)  # noqa: SLF001
    if expected_bytes != replayed_bytes:
        raise RuntimeError("typed compatibility replay differs from construction")
    if hashlib.sha256(replayed_bytes).hexdigest() != construction_sha:
        raise RuntimeError("typed replay bytes lost the construction digest")
    body = _expected_replay_body(expected, construction_sha)
    digest = hot._atomic_write_json(output_root / REPLAY_NAME, body)  # noqa: SLF001
    _load_replay(output_root, construction_sha256=construction_sha)
    print(f"Hot-v3 typed replay: {digest}; byte_identical=true", flush=True)
    return digest


def _read_expected(path: Path, expected_sha256: str, *, label: str) -> dict[str, Any]:
    body, digest = hot._read_json_artifact(path)  # noqa: SLF001
    if digest != expected_sha256:
        raise ValueError(f"sealed {label} changed")
    return body


def _rows_by_ordinal(
    rows: object, *, label: str
) -> dict[int, Mapping[str, Any]]:
    if not isinstance(rows, list) or len(rows) != EXPECTED_QUESTION_COUNT:
        raise ValueError(f"{label} population changed")
    result: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or type(row.get("ordinal")) is not int:
            raise ValueError(f"{label} row is invalid")
        ordinal = int(row["ordinal"])
        if ordinal in result:
            raise ValueError(f"{label} ordinal repeats")
        result[ordinal] = row
    if set(result) != set(range(EXPECTED_QUESTION_COUNT)):
        raise ValueError(f"{label} ordinals changed")
    return result


def _score_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    supported = [row for row in rows if row["effective_execution_supported"]]
    v7_misses = [row for row in rows if row["v7_correct"] is False]
    by_style: dict[str, dict[str, int]] = {}
    for style in sorted({str(row["style"]) for row in rows}):
        group = [row for row in rows if row["style"] == style]
        by_style[style] = {
            "deterministic_executor_applicable_count": sum(
                row["deterministic_executor_applicable"] is True for row in group
            ),
            "lexical_exact_reference_hit_count": sum(
                row["lexical_exact_reference"] is True for row in group
            ),
            "old95_correct_count": sum(row["old95_correct"] is True for row in group),
            "effective_execution_supported_count": sum(
                row["effective_execution_supported"] is True for row in group
            ),
            "generic_execution_supported_count": sum(
                row["generic_execution_status"] == ExecutionStatus.SUPPORTED.value
                for row in group
            ),
            "local_operator_first_numeric_admissible_count": sum(
                row["local_operator_first_numeric_admissible"] is True
                for row in group
            ),
            "question_count": len(group),
            "v3_all_source_reach_count": sum(
                row["v3_all_gold_source_ids_reached"] is True for row in group
            ),
            "v7_miss_count": sum(row["v7_correct"] is False for row in group),
        }
    return {
        "by_style": by_style,
        "deterministic_executor_applicable_count": sum(
            row["deterministic_executor_applicable"] is True for row in rows
        ),
        "lexical_exact_reference_hit_count": sum(
            row["lexical_exact_reference"] is True for row in rows
        ),
        "mean_lexical_f1_all": statistics.fmean(
            float(row["lexical_f1"]) for row in rows
        ),
        "mean_lexical_f1_supported": (
            None
            if not supported
            else statistics.fmean(float(row["lexical_f1"]) for row in supported)
        ),
        "old95_correct_count": sum(row["old95_correct"] is True for row in rows),
        "effective_execution_supported_count": len(supported),
        "generic_execution_supported_count": sum(
            row["generic_execution_status"] == ExecutionStatus.SUPPORTED.value
            for row in rows
        ),
        "local_operator_first_numeric_admissible_count": sum(
            row["local_operator_first_numeric_admissible"] is True
            for row in rows
        ),
        "question_count": len(rows),
        "v3_all_source_reach_count": sum(
            row["v3_all_gold_source_ids_reached"] is True for row in rows
        ),
        "v3_literal_answer_count": sum(
            row["v3_literal_answer"] is True for row in rows
        ),
        "v7_correct_count": sum(row["v7_correct"] is True for row in rows),
        "v7_miss_count": len(v7_misses),
        "v7_miss_deterministic_applicable_count": sum(
            row["deterministic_executor_applicable"] is True for row in v7_misses
        ),
        "v7_miss_old95_correct_count": sum(
            row["old95_correct"] is True for row in v7_misses
        ),
        "v7_miss_local_operator_first_numeric_admissible_count": sum(
            row["local_operator_first_numeric_admissible"] is True
            for row in v7_misses
        ),
        "v7_miss_source_complete_count": sum(
            row["v3_all_gold_source_ids_reached"] is True for row in v7_misses
        ),
        "v7_miss_supported_execution_count": sum(
            row["effective_execution_supported"] is True for row in v7_misses
        ),
    }


def score(
    *,
    dataset: Path,
    split_manifest: Path,
    output_root: Path,
    v3_score_path: Path,
    v7_judgments_path: Path,
    old95_merge_path: Path,
) -> str:
    construction, construction_sha = _load_construction(output_root)
    _replay, replay_sha = _load_replay(
        output_root, construction_sha256=construction_sha
    )
    samples, _identities, population = full100._load_population(  # noqa: SLF001
        dataset, split_manifest
    )
    if population.get("population_identity_sha256") != EXPECTED_POPULATION_SHA256:
        raise ValueError("post-hoc score population changed")
    questions = full100._flatten_questions(samples)  # noqa: SLF001
    if len(questions) != EXPECTED_QUESTION_COUNT:
        raise ValueError("post-hoc question count changed")
    v3_score = _read_expected(
        v3_score_path, EXPECTED_V3_SCORE_SHA256, label="hot-v3 score"
    )
    v7 = _read_expected(
        v7_judgments_path,
        EXPECTED_V7_JUDGMENTS_SHA256,
        label="adaptive-v7 judgments",
    )
    old95 = _read_expected(
        old95_merge_path, EXPECTED_OLD95_MERGE_SHA256, label="old95 merge"
    )
    if (
        v3_score.get("selection_sha256") != EXPECTED_V3_SELECTION_SHA256
        or v3_score.get("population_identity_sha256") != EXPECTED_POPULATION_SHA256
        or v7.get("population_identity_sha256") != EXPECTED_POPULATION_SHA256
        or v7.get("questions") != EXPECTED_QUESTION_COUNT
        or old95.get("question_count") != EXPECTED_QUESTION_COUNT
    ):
        raise ValueError("post-hoc comparison artifact population changed")
    v3_rows = _rows_by_ordinal(v3_score.get("questions"), label="hot-v3 score")
    v7_rows = _rows_by_ordinal(v7.get("rows"), label="adaptive-v7 judgments")
    old95_rows = _rows_by_ordinal(old95.get("questions"), label="old95 merge")
    rows: list[dict[str, Any]] = []
    for diagnostic, question in zip(
        construction["questions"], questions, strict=True
    ):
        ordinal = int(diagnostic["ordinal"])
        qid = str(diagnostic["question_id"])
        v3_row = v3_rows[ordinal]
        v7_row = v7_rows[ordinal]
        old95_row = old95_rows[ordinal]
        reference = str(question.answer)
        effective = diagnostic["effective_packed_execution"]
        prediction = str(effective["prediction"])
        if (
            qid != question.question_id
            or diagnostic["retrieval_query_sha256"] != quote_sha256(question.question)
            or diagnostic["prompt_question_sha256"]
            != quote_sha256(question.dated_question)
            or any(row.get("question_id") != qid for row in (v3_row, v7_row, old95_row))
            or v7_row.get("question_sha256") != quote_sha256(question.question)
            or old95_row.get("question_sha256") != quote_sha256(question.question)
            or v7_row.get("reference_sha256") != quote_sha256(reference)
            or old95_row.get("reference_sha256") != quote_sha256(reference)
        ):
            raise ValueError("post-hoc row identity differs from sealed construction")
        evidence = v3_row.get("effective_hybrid")
        if not isinstance(evidence, Mapping):
            raise ValueError("hot-v3 score omitted effective hybrid observations")
        supported = effective["status"] == ExecutionStatus.SUPPORTED.value
        rows.append(
            {
                "benchmark_category": str(question.category),
                "deterministic_executor_applicable": diagnostic[
                    "deterministic_executor_applicable"
                ],
                "effective_execution_source": effective["source"],
                "effective_execution_status": effective["status"],
                "effective_execution_supported": supported,
                "generic_execution_status": diagnostic["operator_execution"][
                    "status"
                ],
                "lexical_exact_reference": (
                    supported
                    and normalize_answer(prediction) == normalize_answer(reference)
                ),
                "lexical_f1": f1_score(prediction, reference) if supported else 0.0,
                "local_operator_first_numeric_admissible": diagnostic[
                    "local_operator_first_numeric_admissible"
                ],
                "local_operator_first_numeric_status": diagnostic[
                    "local_operator_first_numeric"
                ]["status"],
                "old95_correct": bool(old95_row["correct"]),
                "operator_first_numeric_admissible": diagnostic[
                    "operator_first_numeric_admissible"
                ],
                "operator_first_numeric_status": diagnostic[
                    "operator_first_numeric"
                ]["status"],
                "ordinal": ordinal,
                "prediction_sha256": quote_sha256(prediction),
                "question_id": qid,
                "reference_sha256": quote_sha256(reference),
                "style": diagnostic["operator_spec"]["style"],
                "v3_all_gold_source_ids_reached": evidence.get(
                    "all_gold_source_ids_reached"
                ),
                "v3_literal_answer": evidence.get("literal_answer"),
                "v7_correct": bool(v7_row["correct"]),
            }
        )
    aggregate = _score_aggregate(rows)
    body = {
        "aggregate": aggregate,
        "construction_sha256": construction_sha,
        "format": SCORE_FORMAT,
        "gold_fields_present": True,
        "implementation_sha256": construction["implementation"]["sha256"],
        "old95_merge_sha256": EXPECTED_OLD95_MERGE_SHA256,
        "population_identity_sha256": EXPECTED_POPULATION_SHA256,
        "provider_calls": 0,
        "question_count": EXPECTED_QUESTION_COUNT,
        "questions": rows,
        "replay_sha256": replay_sha,
        "status": "post_replay_gold_join_typed_operator_compatibility",
        "v3_score_sha256": EXPECTED_V3_SCORE_SHA256,
        "v3_selection_sha256": EXPECTED_V3_SELECTION_SHA256,
        "v7_judgments_sha256": EXPECTED_V7_JUDGMENTS_SHA256,
    }
    digest = hot._atomic_write_json(output_root / SCORE_NAME, body)  # noqa: SLF001
    print(
        f"Hot-v3 typed score: {digest}; supported="
        f"{aggregate['effective_execution_supported_count']}/{EXPECTED_QUESTION_COUNT}; "
        f"exact={aggregate['lexical_exact_reference_hit_count']}/{EXPECTED_QUESTION_COUNT}; "
        f"v7-miss supported={aggregate['v7_miss_supported_execution_count']}/"
        f"{aggregate['v7_miss_count']}",
        flush=True,
    )
    return digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-root", type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run")
    commands.add_parser("replay")
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--dataset", type=Path, required=True)
    score_parser.add_argument("--split-manifest", type=Path, default=full100.DEFAULT_SPLIT)
    score_parser.add_argument(
        "--v3-score", type=Path, default=DEFAULT_V3_ROOT / v3.SCORE_NAME
    )
    score_parser.add_argument(
        "--v7-judgments",
        type=Path,
        default=DEFAULT_V7_ROOT / "answer-judgments.json",
    )
    score_parser.add_argument("--old95-merge", type=Path, default=DEFAULT_OLD95_MERGE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = args.output_root.resolve()
    if args.command == "run":
        run(v3_root=args.v3_root.resolve(), output_root=output_root)
    elif args.command == "replay":
        replay(v3_root=args.v3_root.resolve(), output_root=output_root)
    elif args.command == "score":
        score(
            dataset=args.dataset.resolve(),
            split_manifest=args.split_manifest.resolve(),
            output_root=output_root,
            v3_score_path=args.v3_score.resolve(),
            v7_judgments_path=args.v7_judgments.resolve(),
            old95_merge_path=args.old95_merge.resolve(),
        )
    else:  # pragma: no cover
        raise AssertionError(f"unhandled command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
