"""Provenance-clean temporal fail-open overlay for sealed A1 dispositions.

The generic Terra classifier remains immutable. This module authenticates its
construction/replay and emits a distinct effective-disposition artifact. A
question-derived temporal specialist may change only
``definitely_irrelevant`` to ``unresolved`` for a selected leaf inside the
question's executable target day or lookback window. Original provider
responses are referenced by artifact digest and are never copied or rewritten.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Sequence

from memory_condense.domain.discourse import quote_sha256

from .after_union_fact_closure import (
    CROSS_BOUNDARY_EDGE_FORMAT,
    LEAF_DISPOSITION_FORMAT,
    SELECTED_LEAF_FORMAT,
    SELECTION_FORMAT,
    CrossBoundaryEdge,
    SealedLeafDisposition,
    SelectedHLeaf,
    build_after_union_selection,
)
from .contracts import (
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
    require_text,
)
from .full_store_slot_closure import (
    TemporalTargetMode,
    _parse_datetime,
    _temporal_target,
)
from .typed_operator_spec import compile_typed_operator_spec


A1_FORMAT = "memory-condense-r7-after-union-a1-preflight-v2"
DISPOSITIONS_FORMAT = f"{A1_FORMAT}-sealed-dispositions-v1"
POLICY_ID = "r7-a1-temporal-boundary-fail-open-v2"
EFFECTIVE_DISPOSITIONS_FORMAT = (
    "memory-condense-r7-a1-temporal-fail-open-effective-dispositions-v1"
)
REPORT_FORMAT = "memory-condense-r7-a1-temporal-fail-open-report-v2"
_DATE_LABEL_RE = re.compile(r"^date:(?P<date>\d{4}-\d{2}-\d{2})$")
_DISPOSITIONS = {"relevant", "definitely_irrelevant", "unresolved"}
_POLICY_PROJECTION = {
    "allowed_transition": "definitely_irrelevant_to_unresolved_only",
    "date_source": "authenticated_selected_leaf_boundary_date",
    "format": POLICY_ID,
    "target_source": "dated_question_plus_question_only_operator_spec",
}
POLICY_SHA256 = identity_sha256(_POLICY_PROJECTION)
LEGACY_OVERLAY_MARKER_KEYS = frozenset(
    {
        "base_disposition_artifact_sha256",
        "physical_provider_calls_during_temporal_fail_open",
        "temporal_fail_open_override_count",
        "temporal_fail_open_policy_id",
        "temporal_fail_open_policy_sha256",
    }
)
_A1_QUESTION_FORMAT = f"{A1_FORMAT}-question-v1"
_A1_CLASSIFIER_REQUEST_FORMAT = f"{A1_FORMAT}-classifier-request-v1"
_CLASSIFIER_LIFECYCLE_FORMAT = (
    "memory-condense-r7-after-union-a1-classifier-lifecycle-v1"
)
_BASE_RESPONSE_FORMAT = f"{_CLASSIFIER_LIFECYCLE_FORMAT}-response-row-v1"
_A1_FIREWALL = {
    "benchmark_fields_loaded": False,
    "ordinal_routing_enabled": False,
    "protected_parent_loaded": False,
    "semantic_atom_manifest_loaded": False,
    "source_allowlist_loaded": False,
    "topic_labels_have_exclusion_authority": False,
}
_A1_KEYS = {
    "actionable_compiler_request_count",
    "classifier_payload_class",
    "classifier_request_count",
    "classifier_request_population_sha256",
    "compiler_output_artifact_sha256",
    "compiler_payload_class",
    "compiler_request_count",
    "compiler_workload_status",
    "construction_identity_sha256",
    "construction_status",
    "disposition_artifact_sha256",
    "disposition_classifier_id",
    "expected_question_count",
    "format",
    "gold_loaded",
    "hard_total_token_cap",
    "max_leaves_per_classifier_shard",
    "max_leaves_per_compiler_shard",
    "missing_classifier_call_count",
    "missing_classifier_request_sha256s",
    "missing_compiler_call_count",
    "missing_external_call_count",
    "missing_external_request_sha256s",
    "operator_obligations_closed",
    "output_token_reserve",
    "provider_calls_performed_by_core",
    "question_count",
    "question_population_sha256",
    "questions",
    "retained_transformer_token_state_bytes",
    "runtime_firewall",
    "selected_leaf_count",
    "selected_population_sha256",
    "selected_populations_resolved",
    "source_artifact_sha256",
    "source_replay_artifact_sha256",
    "union_before_exclusion",
}
_A1_QUESTION_KEYS = {
    "actionable_compiler_request_count",
    "classifier_request_count",
    "classifier_request_population_sha256",
    "classifier_requests",
    "compiler_request_count",
    "compiler_request_results",
    "compiler_requests",
    "compiler_workload_status",
    "dated_question",
    "dated_question_sha256",
    "disposition_counts",
    "fact_closure",
    "format",
    "hard_total_token_cap",
    "missing_classifier_request_sha256s",
    "missing_compiler_request_sha256s",
    "operator_execution",
    "operator_obligations",
    "operator_packet",
    "operator_packet_error",
    "operator_spec",
    "output_token_reserve",
    "provider_calls_performed_by_core",
    "question_id",
    "question_receipt_sha256",
    "question_sha256",
    "request_population_sha256",
    "retained_transformer_token_state_bytes",
    "selected_leaf_count",
    "selected_population_sha256",
    "semantic_selection",
    "topic_labels_have_exclusion_authority",
    "union_population_built_before_exclusion",
}
_A1_CLASSIFIER_REQUEST_KEYS = {
    "answer_output_token_reserve",
    "boundary_labels_for_scheduling_only",
    "classifier_output_token_reserve",
    "format",
    "hard_total_token_cap",
    "leaf_handle_ids",
    "messages",
    "payload_class",
    "prompt_token_proxy",
    "question_sha256",
    "request_sha256",
    "selected_union_population_sha256",
    "shard_id",
    "shard_population_sha256",
    "topic_labels_for_scheduling_only",
    "topic_labels_have_exclusion_authority",
}
_DISPOSITION_FIREWALL = {
    "gold_loaded": False,
    "ordinal_routing_enabled": False,
    "protected_parent_loaded": False,
    "reference_loaded": False,
    "semantic_atom_manifest_loaded": False,
    "source_allowlist_loaded": False,
}
_BASE_DISPOSITION_KEYS = {
    "a1_construction_artifact_sha256",
    "a1_replay_artifact_sha256",
    "classifier_id",
    "classifier_request_population_sha256",
    "completion_runtime_identity_sha256",
    "derived_provider_call_count",
    "disposition_population_sha256",
    "format",
    "journal_owner_identity_sha256",
    "lifecycle_format",
    "model",
    "model_prompt_population_sha256",
    "physical_provider_calls_during_materialization",
    "preflight_artifact_sha256",
    "prompt_population_sha256",
    "provider_calls_performed_by_core",
    "question_count",
    "questions",
    "release_authorization_artifact_sha256",
    "response_population_sha256",
    "responses",
    "retained_transformer_token_state_bytes",
    "runtime_firewall",
    "source_artifact_sha256",
    "source_replay_artifact_sha256",
}
_BASE_RESPONSE_KEYS = {
    "call_key_sha256",
    "classifier_output",
    "classifier_output_sha256",
    "dispositions",
    "format",
    "leaf_bindings",
    "messages_sha256",
    "question_sha256",
    "request_journal_sha256",
    "request_sha256",
    "response_journal_sha256",
    "response_row_receipt_sha256",
    "selected_union_population_sha256",
    "source_artifact_sha256",
}
_BASE_QUESTION_KEYS = {
    "classifier_request_sha256s",
    "dispositions",
    "question_sha256",
    "selected_union_population_sha256",
}
_BASE_DISPOSITION_ROW_KEYS = {
    "disposition",
    "handle_id",
    "leaf_receipt_sha256",
}
_LEAF_BINDING_KEYS = {"handle_id", "leaf_receipt_sha256"}


class TemporalFailOpenContractError(ValueError):
    """Raised when a sealed input or one-way temporal-veto invariant changes."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TemporalFailOpenContractError(message)


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)  # type: ignore[arg-type]


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be a list")
    return list(value)  # type: ignore[arg-type]


def _artifact_sha(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _without(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    return {name: child for name, child in value.items() if name != key}


def _verify_receipt(value: Mapping[str, Any], key: str, label: str) -> str:
    receipt = require_sha256(value.get(key), label)
    _require(receipt == identity_sha256(_without(value, key)), f"{label} changed")
    return receipt


def _selected_leaf(raw: object) -> SelectedHLeaf:
    row = _exact_dict(raw, "A1 selected leaf")
    leaf = SelectedHLeaf(
        require_text(row.get("handle_id"), "A1 leaf handle"),
        require_text(row.get("group_handle"), "A1 leaf group"),
        require_text(row.get("text"), "A1 leaf text"),
        require_sha256(row.get("source_receipt_sha256"), "A1 leaf source"),
        tuple(
            require_text(value, "A1 leaf topic label")
            for value in _exact_list(row.get("topic_labels"), "A1 leaf topic labels")
        ),
        tuple(
            require_text(value, "A1 leaf boundary label")
            for value in _exact_list(
                row.get("boundary_labels"), "A1 leaf boundary labels"
            )
        ),
        tuple(
            require_text(value, "A1 leaf edge ID")
            for value in _exact_list(
                row.get("cross_boundary_edge_ids"), "A1 leaf edge IDs"
            )
        ),
        require_sha256(row.get("receipt_sha256"), "A1 leaf"),
    )
    _require(
        row == leaf.projection() and row.get("format") == SELECTED_LEAF_FORMAT,
        "A1 selected leaf differs from its authenticated projection",
    )
    return leaf


def _cross_boundary_edge(raw: object) -> CrossBoundaryEdge:
    row = _exact_dict(raw, "A1 cross-boundary edge")
    edge = CrossBoundaryEdge(
        require_text(row.get("edge_id"), "A1 edge ID"),
        row.get("kind"),  # type: ignore[arg-type]
        require_text(row.get("left_handle_id"), "A1 edge left handle"),
        require_text(row.get("right_handle_id"), "A1 edge right handle"),
        require_text(row.get("relation"), "A1 edge relation"),
        require_sha256(row.get("receipt_sha256"), "A1 edge"),
    )
    _require(
        row == edge.projection() and row.get("format") == CROSS_BOUNDARY_EDGE_FORMAT,
        "A1 cross-boundary edge differs from its authenticated projection",
    )
    return edge


def _sealed_leaf_disposition(raw: object) -> SealedLeafDisposition:
    row = _exact_dict(raw, "A1 provisional leaf disposition")
    disposition = SealedLeafDisposition(
        require_text(row.get("handle_id"), "A1 provisional disposition handle"),
        require_sha256(
            row.get("leaf_receipt_sha256"),
            "A1 provisional disposition leaf",
        ),
        require_sha256(
            row.get("question_sha256"),
            "A1 provisional disposition question",
        ),
        require_text(
            row.get("classifier_id"),
            "A1 provisional disposition classifier",
        ),
        row.get("disposition"),  # type: ignore[arg-type]
        require_sha256(
            row.get("receipt_sha256"),
            "A1 provisional disposition",
        ),
    )
    _require(
        row == disposition.projection()
        and row.get("format") == LEAF_DISPOSITION_FORMAT,
        "A1 provisional disposition differs from its authenticated projection",
    )
    return disposition


def _validate_a1_contract(a1: Mapping[str, Any]) -> tuple[list[Any], list[str]]:
    """Authenticate the complete A1 pre-classification construction contract."""

    _require(
        set(a1) == _A1_KEYS
        and a1.get("format") == A1_FORMAT
        and a1.get("gold_loaded") is False
        and a1.get("union_before_exclusion") is True
        and a1.get("provider_calls_performed_by_core") == 0
        and a1.get("retained_transformer_token_state_bytes") == 0
        and a1.get("construction_status")
        == "preflight_external_classification_then_compilation_required"
        and a1.get("compiler_workload_status")
        == "provisional_fail_open_pending_classifier"
        and a1.get("disposition_artifact_sha256") is None
        and a1.get("compiler_output_artifact_sha256") is None
        and a1.get("selected_populations_resolved") is False
        and a1.get("operator_obligations_closed") is False
        and a1.get("actionable_compiler_request_count") == 0
        and a1.get("classifier_payload_class")
        == "after_union_leaf_relevance_strict_json_v1"
        and a1.get("compiler_payload_class")
        == "typed_fact_compiler_strict_json_v1"
        and a1.get("disposition_classifier_id")
        == "r7-a1-deterministic-all-uncertain-v1"
        and a1.get("hard_total_token_cap") == 8000
        and a1.get("output_token_reserve") == 768
        and type(a1.get("max_leaves_per_classifier_shard")) is int
        and 0 < a1.get("max_leaves_per_classifier_shard") <= 48
        and type(a1.get("max_leaves_per_compiler_shard")) is int
        and 0 < a1.get("max_leaves_per_compiler_shard") <= 8
        and type(a1.get("compiler_request_count")) is int
        and a1.get("compiler_request_count") >= 0
        and a1.get("missing_compiler_call_count") == 0
        and _exact_dict(a1.get("runtime_firewall"), "A1 firewall")
        == _A1_FIREWALL,
        "A1 construction is not the exact gold-blind pre-classification contract",
    )
    _verify_receipt(a1, "construction_identity_sha256", "A1 construction")
    source_sha = require_sha256(a1.get("source_artifact_sha256"), "A1 source")
    source_replay_sha = require_sha256(
        a1.get("source_replay_artifact_sha256"), "A1 source replay"
    )
    _require(source_sha == source_replay_sha, "A1 source construction/replay differ")

    questions = _exact_list(a1.get("questions"), "A1 questions")
    _require(
        type(a1.get("expected_question_count")) is int
        and a1.get("expected_question_count") == len(questions) > 0
        and a1.get("question_count") == len(questions),
        "A1 question population count changed",
    )
    question_receipts: list[str] = []
    selected_population_shas: list[str] = []
    request_sha_population: list[str] = []
    selected_leaf_count = 0
    compiler_request_count = 0
    question_shas: set[str] = set()
    question_ids: set[str] = set()
    for raw_question in questions:
        question = _exact_dict(raw_question, "A1 question")
        _require(
            set(question) == _A1_QUESTION_KEYS,
            "A1 question schema changed",
        )
        question_receipts.append(
            _verify_receipt(question, "question_receipt_sha256", "A1 question")
        )
        dated_question = require_text(question.get("dated_question"), "dated question")
        question_sha = require_sha256(question.get("question_sha256"), "A1 question")
        question_id = require_text(question.get("question_id"), "A1 question ID")
        _require(
            question.get("format") == _A1_QUESTION_FORMAT
            and quote_sha256(dated_question) == question_sha
            and question.get("dated_question_sha256") == question_sha
            and question.get("union_population_built_before_exclusion") is True
            and question.get("provider_calls_performed_by_core") == 0
            and question.get("retained_transformer_token_state_bytes") == 0
            and question.get("hard_total_token_cap") == 8000
            and question.get("output_token_reserve") == 768
            and question.get("topic_labels_have_exclusion_authority") is False
            and question.get("actionable_compiler_request_count") == 0
            and question.get("compiler_workload_status")
            == "provisional_fail_open_preview"
            and question.get("operator_execution") is None
            and question.get("operator_packet") is None
            and question.get("operator_packet_error") == "no_compiled_facts"
            and question_sha not in question_shas
            and question_id not in question_ids,
            "A1 question/date/union contract changed",
        )
        question_shas.add(question_sha)
        question_ids.add(question_id)

        semantic = _exact_dict(
            question.get("semantic_selection"), "A1 semantic selection"
        )
        _verify_receipt(semantic, "receipt_sha256", "A1 semantic selection")
        _require(
            semantic.get("format") == SELECTION_FORMAT
            and semantic.get("gold_loaded") is False
            and semantic.get("question_sha256") == question_sha
            and semantic.get("provider_calls_performed_by_core") == 0
            and semantic.get("retained_transformer_token_state_bytes") == 0,
            "A1 semantic selection contract changed",
        )
        leaves = tuple(
            _selected_leaf(row)
            for row in _exact_list(semantic.get("leaves"), "A1 selected leaves")
        )
        edges = tuple(
            _cross_boundary_edge(row)
            for row in _exact_list(
                semantic.get("cross_boundary_edges"), "A1 cross-boundary edges"
            )
        )
        provisional_dispositions = tuple(
            _sealed_leaf_disposition(row)
            for row in _exact_list(
                semantic.get("dispositions"),
                "A1 provisional leaf dispositions",
            )
        )
        replayed_selection = build_after_union_selection(
            dated_question,
            leaves,
            provisional_dispositions,
            cross_boundary_edges=edges,
        )
        _require(
            semantic == replayed_selection.projection(),
            "A1 semantic selection differs from deterministic replay",
        )
        leaf_ids = tuple(leaf.handle_id for leaf in leaves)
        edge_ids = {edge.edge_id for edge in edges}
        _require(
            len(set(leaf_ids)) == len(leaves)
            and all(
                set(leaf.cross_boundary_edge_ids) <= edge_ids for leaf in leaves
            )
            and all(set(edge.handle_ids) <= set(leaf_ids) for edge in edges),
            "A1 semantic leaf/edge population changed",
        )
        selected_sha = require_sha256(
            question.get("selected_population_sha256"), "A1 selected population"
        )
        _require(
            selected_sha == identity_sha256([leaf.projection() for leaf in leaves])
            and question.get("selected_leaf_count") == len(leaves),
            "A1 selected population receipt changed",
        )
        selected_leaf_count += len(leaves)
        selected_population_shas.append(selected_sha)

        requests = [
            _exact_dict(row, "A1 classifier request")
            for row in _exact_list(
                question.get("classifier_requests"), "A1 classifier requests"
            )
        ]
        request_shas: list[str] = []
        request_leaf_ids: list[str] = []
        for request in requests:
            _require(
                set(request) == _A1_CLASSIFIER_REQUEST_KEYS
                and request.get("answer_output_token_reserve") == 768
                and request.get("classifier_output_token_reserve") == 1024
                and request.get("hard_total_token_cap") == 8000
                and request.get("payload_class")
                == "after_union_leaf_relevance_strict_json_v1"
                and request.get("topic_labels_have_exclusion_authority")
                is False
                and type(request.get("prompt_token_proxy")) is int
                and 0 <= request.get("prompt_token_proxy") <= 6976
                and bool(require_text(request.get("shard_id"), "A1 shard ID")),
                "A1 classifier request schema/policy changed",
            )
            for label_key in (
                "boundary_labels_for_scheduling_only",
                "topic_labels_for_scheduling_only",
            ):
                labels = [
                    require_text(value, f"A1 classifier {label_key}")
                    for value in _exact_list(
                        request.get(label_key), f"A1 classifier {label_key}"
                    )
                ]
                _require(
                    len(labels) == len(set(labels)),
                    f"A1 classifier {label_key} repeats",
                )
            request_sha = _verify_receipt(
                request, "request_sha256", "A1 classifier request"
            )
            handles = [
                require_text(value, "A1 classifier request handle")
                for value in _exact_list(
                    request.get("leaf_handle_ids"), "A1 classifier request handles"
                )
            ]
            _require(
                request.get("format") == _A1_CLASSIFIER_REQUEST_FORMAT
                and request.get("question_sha256") == question_sha
                and request.get("selected_union_population_sha256") == selected_sha
                and request.get("shard_population_sha256")
                == identity_sha256(handles),
                "A1 classifier request binding changed",
            )
            request_shas.append(request_sha)
            request_leaf_ids.extend(handles)
        _require(
            request_leaf_ids == list(leaf_ids)
            and len(request_shas) == len(set(request_shas))
            and question.get("classifier_request_count") == len(requests)
            and question.get("classifier_request_population_sha256")
            == identity_sha256(request_shas),
            "A1 question classifier population changed",
        )
        _require(
            question.get("missing_classifier_request_sha256s")
            == request_shas
            and question.get("missing_compiler_request_sha256s") == []
            and question.get("disposition_counts")
            == {
                "definitely_irrelevant": 0,
                "relevant": 0,
                "uncertain": len(leaves),
            },
            "A1 provisional question status changed",
        )
        compiler_requests = [
            _exact_dict(row, "A1 compiler request")
            for row in _exact_list(
                question.get("compiler_requests"), "A1 compiler requests"
            )
        ]
        compiler_results = _exact_list(
            question.get("compiler_request_results"),
            "A1 compiler request results",
        )
        compiler_shas = [
            require_sha256(row.get("request_sha256"), "A1 compiler request")
            for row in compiler_requests
        ]
        _require(
            question.get("compiler_request_count") == len(compiler_requests)
            == len(compiler_results)
            and question.get("request_population_sha256")
            == identity_sha256(compiler_shas),
            "A1 compiler preview population changed",
        )
        _exact_dict(question.get("fact_closure"), "A1 fact closure")
        _exact_list(question.get("operator_obligations"), "A1 obligations")
        _exact_dict(question.get("operator_spec"), "A1 operator spec")
        compiler_request_count += len(compiler_requests)
        request_sha_population.extend(request_shas)

    _require(
        a1.get("question_population_sha256") == identity_sha256(question_receipts)
        and a1.get("selected_leaf_count") == selected_leaf_count
        and a1.get("selected_population_sha256")
        == identity_sha256(selected_population_shas)
        and len(request_sha_population) == len(set(request_sha_population))
        and a1.get("classifier_request_count") == len(request_sha_population)
        and a1.get("classifier_request_population_sha256")
        == identity_sha256(sorted(request_sha_population))
        and a1.get("missing_classifier_call_count") == len(request_sha_population)
        and a1.get("missing_classifier_request_sha256s") == request_sha_population,
        "A1 global population/receipt contract changed",
    )
    _require(
        a1.get("compiler_request_count") == compiler_request_count
        and a1.get("missing_external_call_count") == len(request_sha_population)
        and a1.get("missing_external_request_sha256s") == request_sha_population,
        "A1 external-work population changed",
    )
    return questions, request_sha_population


def _leaf_dates(leaf: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for raw in _exact_list(leaf.get("boundary_labels"), "A1 leaf boundary labels"):
        label = require_text(raw, "A1 leaf boundary label")
        match = _DATE_LABEL_RE.fullmatch(label)
        if match is not None:
            values.append(date.fromisoformat(match.group("date")).isoformat())
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True, slots=True)
class _Protection:
    mode: str
    target_date: str | None
    asked_date: str | None
    lookback_days: int | None
    derivation: str

    @property
    def applicable(self) -> bool:
        return self.mode != "none"

    def protects(self, leaf_dates: Sequence[str]) -> bool:
        if self.mode == "exact_day":
            return self.target_date in set(leaf_dates)
        if self.mode == "lookback_window":
            _require(
                self.asked_date is not None and self.lookback_days is not None,
                "lookback protection lost its executable bounds",
            )
            asked = date.fromisoformat(self.asked_date)
            return any(
                0 <= (asked - date.fromisoformat(value)).days <= self.lookback_days
                for value in leaf_dates
            )
        return False

    def projection(self) -> dict[str, Any]:
        return {
            "asked_date": self.asked_date,
            "derivation": self.derivation,
            "lookback_days": self.lookback_days,
            "mode": self.mode,
            "target_date": self.target_date,
        }


def _protection(dated_question: str) -> _Protection:
    spec = compile_typed_operator_spec(dated_question)
    target = _temporal_target(dated_question, spec)
    if target.mode is TemporalTargetMode.EXACT_DAY:
        _require(target.target_date is not None, "exact temporal target lost its date")
        return _Protection(
            "exact_day", target.target_date, None, None, target.derivation
        )
    if target.mode is TemporalTargetMode.LOOKBACK_WINDOW:
        asked = _parse_datetime(target.asked_at)
        _require(
            asked is not None and target.lookback_days is not None,
            "lookback temporal target lost its executable bounds",
        )
        return _Protection(
            "lookback_window",
            None,
            asked.date().isoformat(),
            target.lookback_days,
            target.derivation,
        )
    return _Protection("none", None, None, None, target.derivation)


def _validate_base_response_rows(
    dispositions: Mapping[str, Any],
    request_sha_population: Sequence[str],
    a1_questions: Sequence[object],
) -> Mapping[str, tuple[dict[str, Any], ...]]:
    request_contracts: dict[str, dict[str, Any]] = {}
    for raw_question in a1_questions:
        question = _exact_dict(raw_question, "A1 question")
        question_sha = require_sha256(
            question.get("question_sha256"), "A1 question"
        )
        selected_sha = require_sha256(
            question.get("selected_population_sha256"),
            "A1 selected population",
        )
        semantic = _exact_dict(
            question.get("semantic_selection"), "A1 semantic selection"
        )
        leaves_by_handle = {
            require_text(leaf.get("handle_id"), "A1 leaf handle"): leaf
            for leaf in (
                _exact_dict(row, "A1 selected leaf")
                for row in _exact_list(
                    semantic.get("leaves"), "A1 selected leaves"
                )
            )
        }
        for raw_request in _exact_list(
            question.get("classifier_requests"), "A1 classifier requests"
        ):
            request = _exact_dict(raw_request, "A1 classifier request")
            request_sha = require_sha256(
                request.get("request_sha256"), "A1 classifier request"
            )
            handles = [
                require_text(value, "A1 classifier request handle")
                for value in _exact_list(
                    request.get("leaf_handle_ids"),
                    "A1 classifier request handles",
                )
            ]
            messages = _exact_list(
                request.get("messages"), "A1 classifier request messages"
            )
            _require(
                len(messages) == 2
                and [
                    _exact_dict(row, "A1 classifier request message").get(
                        "role"
                    )
                    for row in messages
                ]
                == ["system", "user"]
                and all(
                    set(_exact_dict(row, "A1 classifier request message"))
                    == {"content", "role"}
                    and bool(
                        require_text(
                            _exact_dict(
                                row, "A1 classifier request message"
                            ).get("content"),
                            "A1 classifier request message content",
                        )
                    )
                    for row in messages
                ),
                "A1 classifier request message envelope changed",
            )
            request_contracts[request_sha] = {
                "leaf_bindings": [
                    {
                        "handle_id": handle,
                        "leaf_receipt_sha256": require_sha256(
                            leaves_by_handle[handle].get("receipt_sha256"),
                            "A1 selected leaf",
                        ),
                    }
                    for handle in handles
                ],
                "messages_sha256": identity_sha256(messages),
                "question_sha256": question_sha,
                "selected_union_population_sha256": selected_sha,
            }
    _require(
        set(request_contracts) == set(request_sha_population),
        "A1 response request contract population changed",
    )
    response_rows = _exact_list(dispositions.get("responses"), "base responses")
    receipts = [
        require_sha256(
            _exact_dict(row, "base response").get("response_row_receipt_sha256"),
            "base response receipt",
        )
        for row in response_rows
    ]
    _require(
        len(response_rows) == len(request_sha_population)
        and dispositions.get("response_population_sha256")
        == identity_sha256(receipts),
        "base response population changed",
    )
    result: dict[str, tuple[dict[str, Any], ...]] = {}
    for raw in response_rows:
        response = _exact_dict(raw, "base response")
        _require(
            set(response) == _BASE_RESPONSE_KEYS
            and response.get("format") == _BASE_RESPONSE_FORMAT,
            "base response schema changed",
        )
        _verify_receipt(response, "response_row_receipt_sha256", "base response")
        request_sha = require_sha256(response.get("request_sha256"), "base response request")
        _require(
            request_sha in request_sha_population and request_sha not in result,
            "base response request population changed",
        )
        request_contract = request_contracts[request_sha]
        leaf_bindings = [
            _exact_dict(row, "base response leaf binding")
            for row in _exact_list(
                response.get("leaf_bindings"), "base response leaf bindings"
            )
        ]
        _require(
            all(set(row) == _LEAF_BINDING_KEYS for row in leaf_bindings)
            and leaf_bindings == request_contract["leaf_bindings"]
            and response.get("messages_sha256")
            == request_contract["messages_sha256"]
            and response.get("question_sha256")
            == request_contract["question_sha256"]
            and response.get("selected_union_population_sha256")
            == request_contract["selected_union_population_sha256"]
            and response.get("source_artifact_sha256")
            == dispositions.get("source_artifact_sha256"),
            "base response request/question/source binding changed",
        )
        for key in (
            "call_key_sha256",
            "messages_sha256",
            "request_journal_sha256",
            "response_journal_sha256",
        ):
            require_sha256(response.get(key), f"base response {key}")
        completion = require_text(response.get("classifier_output"), "classifier output")
        _require(
            response.get("classifier_output_sha256") == quote_sha256(completion),
            "base classifier output digest changed",
        )
        try:
            parsed = json.loads(completion)
        except json.JSONDecodeError as exc:
            raise TemporalFailOpenContractError(
                "base classifier output is not strict JSON"
            ) from exc
        parsed_output = _exact_dict(parsed, "base classifier output")
        _require(
            set(parsed_output) == {"leaf_dispositions"},
            "base classifier output schema changed",
        )
        parsed_rows = _exact_list(
            parsed_output.get("leaf_dispositions"),
            "base classifier output dispositions",
        )
        _require(
            all(
                set(_exact_dict(row, "base classifier disposition"))
                == {"disposition", "handle_id"}
                for row in parsed_rows
            ),
            "base classifier disposition schema changed",
        )
        sealed_rows = tuple(
            _exact_dict(row, "base response disposition")
            for row in _exact_list(response.get("dispositions"), "base response dispositions")
        )
        _require(
            all(set(row) == _BASE_DISPOSITION_ROW_KEYS for row in sealed_rows),
            "base sealed disposition schema changed",
        )
        _require(
            [
                {
                    "disposition": row.get("disposition"),
                    "handle_id": row.get("handle_id"),
                }
                for row in sealed_rows
            ]
            == parsed_rows,
            "base response dispositions differ from provider output",
        )
        for row in sealed_rows:
            _require(
                row.get("disposition") in _DISPOSITIONS,
                "base response disposition enum changed",
            )
            require_text(row.get("handle_id"), "base response handle")
            require_sha256(row.get("leaf_receipt_sha256"), "base response leaf")
        result[request_sha] = sealed_rows
    _require(
        set(result) == set(request_sha_population),
        "base responses do not exactly cover classifier requests",
    )
    return result


def build_temporal_fail_open_artifacts(
    a1_payload: Mapping[str, Any],
    a1_artifact_sha256: str,
    a1_replay_payload: Mapping[str, Any],
    a1_replay_artifact_sha256: str,
    disposition_payload: Mapping[str, Any],
    disposition_artifact_sha256: str,
    disposition_replay_payload: Mapping[str, Any],
    disposition_replay_artifact_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a distinct effective-disposition overlay and audit report."""

    a1 = _exact_dict(a1_payload, "A1 construction")
    a1_replay = _exact_dict(a1_replay_payload, "A1 replay")
    dispositions = _exact_dict(disposition_payload, "base dispositions")
    disposition_replay = _exact_dict(
        disposition_replay_payload, "base disposition replay"
    )
    for payload, supplied, label in (
        (a1, a1_artifact_sha256, "A1 construction"),
        (a1_replay, a1_replay_artifact_sha256, "A1 replay"),
        (dispositions, disposition_artifact_sha256, "base dispositions"),
        (
            disposition_replay,
            disposition_replay_artifact_sha256,
            "base disposition replay",
        ),
    ):
        _require(
            require_sha256(supplied, label) == _artifact_sha(payload),
            f"{label} digest differs from its payload",
        )
    _require(
        a1_artifact_sha256 == a1_replay_artifact_sha256 and a1 == a1_replay,
        "A1 construction and replay are not byte-identical",
    )
    _require(
        disposition_artifact_sha256 == disposition_replay_artifact_sha256
        and dispositions == disposition_replay,
        "base disposition construction and replay are not byte-identical",
    )
    a1_questions, request_sha_population = _validate_a1_contract(a1)
    _require(
        set(dispositions) == _BASE_DISPOSITION_KEYS
        and dispositions.get("format") == DISPOSITIONS_FORMAT
        and dispositions.get("lifecycle_format")
        == _CLASSIFIER_LIFECYCLE_FORMAT
        and dispositions.get("a1_construction_artifact_sha256")
        == a1_artifact_sha256
        and dispositions.get("a1_replay_artifact_sha256")
        == a1_replay_artifact_sha256
        and dispositions.get("source_artifact_sha256")
        == a1.get("source_artifact_sha256")
        and dispositions.get("source_replay_artifact_sha256")
        == a1.get("source_replay_artifact_sha256")
        and dispositions.get("provider_calls_performed_by_core") == 0
        and dispositions.get("physical_provider_calls_during_materialization") == 0
        and dispositions.get("retained_transformer_token_state_bytes") == 0
        and _exact_dict(dispositions.get("runtime_firewall"), "base firewall")
        == _DISPOSITION_FIREWALL,
        "base dispositions are not exactly bound to the sealed A1 pair",
    )
    for key in (
        "completion_runtime_identity_sha256",
        "journal_owner_identity_sha256",
        "model_prompt_population_sha256",
        "preflight_artifact_sha256",
        "prompt_population_sha256",
        "release_authorization_artifact_sha256",
    ):
        require_sha256(dispositions.get(key), f"base dispositions {key}")
    require_text(dispositions.get("classifier_id"), "base classifier ID")
    require_text(dispositions.get("model"), "base classifier model")
    assert_gold_blind(a1, path="r7_a1_temporal_fail_open_a1")
    assert_gold_blind(dispositions, path="r7_a1_temporal_fail_open_base")

    base_questions = _exact_list(dispositions.get("questions"), "base questions")
    _require(
        len(a1_questions) == len(base_questions) > 0
        and a1.get("question_count") == len(a1_questions)
        and dispositions.get("question_count") == len(base_questions)
        and dispositions.get("disposition_population_sha256")
        == identity_sha256(base_questions),
        "A1/base question or disposition population changed",
    )
    _require(
        dispositions.get("classifier_request_population_sha256")
        == a1.get("classifier_request_population_sha256")
        and dispositions.get("derived_provider_call_count")
        == len(request_sha_population),
        "classifier request population changed",
    )
    response_by_request = _validate_base_response_rows(
        dispositions, request_sha_population, a1_questions
    )

    effective_questions: list[dict[str, Any]] = []
    report_questions: list[dict[str, Any]] = []
    changed_total = 0
    protected_total = 0
    for raw_a1_question, raw_base_question in zip(
        a1_questions, base_questions, strict=True
    ):
        a1_question = _exact_dict(raw_a1_question, "A1 question")
        base_question = _exact_dict(raw_base_question, "base question")
        _require(
            set(base_question) == _BASE_QUESTION_KEYS,
            "base disposition question schema changed",
        )
        question_sha = require_sha256(a1_question.get("question_sha256"), "A1 question")
        dated_question = require_text(a1_question.get("dated_question"), "dated question")
        _require(
            question_sha == quote_sha256(dated_question)
            and base_question.get("question_sha256") == question_sha,
            "base disposition question order or identity changed",
        )
        semantic = _exact_dict(
            a1_question.get("semantic_selection"), "A1 semantic selection"
        )
        _require(
            semantic.get("gold_loaded") is False,
            "A1 semantic selection loaded gold",
        )
        _verify_receipt(semantic, "receipt_sha256", "A1 semantic selection")
        leaves = [
            _exact_dict(row, "A1 selected leaf")
            for row in _exact_list(semantic.get("leaves"), "A1 selected leaves")
        ]
        for leaf in leaves:
            _verify_receipt(leaf, "receipt_sha256", "A1 selected leaf")
        _require(
            a1_question.get("selected_leaf_count") == len(leaves)
            and a1_question.get("selected_population_sha256")
            == identity_sha256(leaves)
            and base_question.get("selected_union_population_sha256")
            == a1_question.get("selected_population_sha256"),
            "selected union population changed",
        )
        request_shas = [
            require_sha256(
                _exact_dict(row, "A1 classifier request").get("request_sha256"),
                "A1 classifier request",
            )
            for row in _exact_list(
                a1_question.get("classifier_requests"), "A1 classifier requests"
            )
        ]
        _require(
            base_question.get("classifier_request_sha256s") == request_shas,
            "base question classifier request order changed",
        )
        response_rows = tuple(
            row for request_sha in request_shas for row in response_by_request[request_sha]
        )
        base_rows = tuple(
            _exact_dict(row, "base question disposition")
            for row in _exact_list(
                base_question.get("dispositions"), "base question dispositions"
            )
        )
        _require(
            base_rows == response_rows and len(base_rows) == len(leaves),
            "base question dispositions differ from authenticated responses",
        )
        protection = _protection(dated_question)
        effective_rows: list[dict[str, Any]] = []
        overrides: list[dict[str, Any]] = []
        for leaf, base_row in zip(leaves, base_rows, strict=True):
            handle = require_text(leaf.get("handle_id"), "A1 leaf handle")
            leaf_receipt = require_sha256(leaf.get("receipt_sha256"), "A1 leaf")
            base_disposition = base_row.get("disposition")
            _require(
                base_disposition in _DISPOSITIONS
                and base_row.get("handle_id") == handle
                and base_row.get("leaf_receipt_sha256") == leaf_receipt,
                "base disposition enum/order/leaf binding changed",
            )
            leaf_dates = _leaf_dates(leaf)
            protected = protection.protects(leaf_dates)
            protected_total += int(protected)
            effective_disposition = base_disposition
            reason = "unchanged"
            if protected and base_disposition == "definitely_irrelevant":
                effective_disposition = "unresolved"
                reason = "question_derived_temporal_target_match"
                changed_total += 1
            transition_body = {
                "base_disposition": base_disposition,
                "effective_disposition": effective_disposition,
                "handle_id": handle,
                "leaf_receipt_sha256": leaf_receipt,
                "reason": reason,
            }
            transition = {
                **transition_body,
                "transition_receipt_sha256": identity_sha256(transition_body),
            }
            effective_rows.append(transition)
            if reason != "unchanged":
                overrides.append(
                    {
                        "handle_id": handle,
                        "leaf_date_population_sha256": identity_sha256(list(leaf_dates)),
                        "transition_receipt_sha256": transition[
                            "transition_receipt_sha256"
                        ],
                    }
                )
        effective_question_body = {
            "classifier_request_sha256s": request_shas,
            "effective_dispositions": effective_rows,
            "effective_disposition_population_sha256": identity_sha256(effective_rows),
            "question_sha256": question_sha,
            "selected_union_population_sha256": a1_question[
                "selected_population_sha256"
            ],
            "temporal_protection_receipt_sha256": identity_sha256(
                protection.projection()
            ),
        }
        effective_questions.append(
            {
                **effective_question_body,
                "question_effective_disposition_receipt_sha256": identity_sha256(
                    effective_question_body
                ),
            }
        )
        report_questions.append(
            {
                "format": f"{REPORT_FORMAT}-question-v1",
                "override_count": len(overrides),
                "overrides": overrides,
                "protected_selected_leaf_count": sum(
                    protection.protects(_leaf_dates(leaf)) for leaf in leaves
                ),
                "question_sha256": question_sha,
                "temporal_protection_receipt_sha256": identity_sha256(
                    protection.projection()
                ),
                "temporal_target_applicable": protection.applicable,
            }
        )

    base_classifier_id = require_text(
        dispositions.get("classifier_id"), "base classifier ID"
    )
    effective_classifier_id = f"{base_classifier_id}+{POLICY_ID}"
    effective = {
        "a1_construction_artifact_sha256": a1_artifact_sha256,
        "a1_replay_artifact_sha256": a1_replay_artifact_sha256,
        "base_classifier_id": base_classifier_id,
        "base_disposition_artifact_sha256": disposition_artifact_sha256,
        "base_disposition_replay_artifact_sha256": disposition_replay_artifact_sha256,
        "effective_classifier_id": effective_classifier_id,
        "effective_disposition_population_sha256": identity_sha256(
            effective_questions
        ),
        "format": EFFECTIVE_DISPOSITIONS_FORMAT,
        "physical_provider_calls": 0,
        "policy_id": POLICY_ID,
        "policy_sha256": POLICY_SHA256,
        "provider_calls_performed_by_core": 0,
        "question_count": len(effective_questions),
        "questions": effective_questions,
        "retained_transformer_token_state_bytes": 0,
        "runtime_firewall": dict(_DISPOSITION_FIREWALL),
        "source_artifact_sha256": dispositions["source_artifact_sha256"],
        "source_replay_artifact_sha256": dispositions[
            "source_replay_artifact_sha256"
        ],
        "temporal_fail_open_override_count": changed_total,
    }
    assert_gold_blind(effective, path="r7_a1_temporal_fail_open_effective")
    effective_sha = _artifact_sha(effective)
    report = {
        "a1_construction_artifact_sha256": a1_artifact_sha256,
        "a1_replay_artifact_sha256": a1_replay_artifact_sha256,
        "base_disposition_artifact_sha256": disposition_artifact_sha256,
        "base_disposition_replay_artifact_sha256": disposition_replay_artifact_sha256,
        "effective_disposition_artifact_sha256": effective_sha,
        "format": REPORT_FORMAT,
        "gold_loaded": False,
        "override_count": changed_total,
        "physical_provider_calls": 0,
        "policy_id": POLICY_ID,
        "policy_sha256": effective["policy_sha256"],
        "protected_selected_leaf_count": protected_total,
        "provider_calls_performed_by_core": 0,
        "question_count": len(report_questions),
        "question_population_sha256": identity_sha256(
            [row["question_sha256"] for row in report_questions]
        ),
        "questions": report_questions,
        "retained_transformer_token_state_bytes": 0,
        "runtime_firewall": dict(_DISPOSITION_FIREWALL),
        "transition_policy": "definitely_irrelevant_to_unresolved_only",
    }
    report["report_identity_sha256"] = identity_sha256(report)
    assert_gold_blind(report, path="r7_a1_temporal_fail_open_report")
    return effective, report


def validate_temporal_fail_open_effective_artifact(
    effective_payload: Mapping[str, Any],
    effective_artifact_sha256: str,
    effective_replay_payload: Mapping[str, Any],
    effective_replay_artifact_sha256: str,
    a1_payload: Mapping[str, Any],
    a1_artifact_sha256: str,
    a1_replay_payload: Mapping[str, Any],
    a1_replay_artifact_sha256: str,
    base_disposition_payload: Mapping[str, Any],
    base_disposition_artifact_sha256: str,
    base_disposition_replay_payload: Mapping[str, Any],
    base_disposition_replay_artifact_sha256: str,
) -> dict[str, Any]:
    """Re-derive and authenticate one effective overlay from all sealed parents."""

    effective = _exact_dict(effective_payload, "effective dispositions")
    effective_replay = _exact_dict(
        effective_replay_payload, "effective disposition replay"
    )
    for payload, supplied, label in (
        (effective, effective_artifact_sha256, "effective dispositions"),
        (
            effective_replay,
            effective_replay_artifact_sha256,
            "effective disposition replay",
        ),
    ):
        _require(
            require_sha256(supplied, label) == _artifact_sha(payload),
            f"{label} digest differs from its payload",
        )
    _require(
        effective_artifact_sha256 == effective_replay_artifact_sha256
        and effective == effective_replay,
        "effective disposition construction and replay are not byte-identical",
    )
    expected, _report = build_temporal_fail_open_artifacts(
        a1_payload,
        a1_artifact_sha256,
        a1_replay_payload,
        a1_replay_artifact_sha256,
        base_disposition_payload,
        base_disposition_artifact_sha256,
        base_disposition_replay_payload,
        base_disposition_replay_artifact_sha256,
    )
    _require(
        effective == expected and effective_artifact_sha256 == _artifact_sha(expected),
        "effective dispositions differ from the re-derived temporal policy",
    )
    return effective


__all__ = [
    "EFFECTIVE_DISPOSITIONS_FORMAT",
    "LEGACY_OVERLAY_MARKER_KEYS",
    "POLICY_ID",
    "POLICY_SHA256",
    "REPORT_FORMAT",
    "TemporalFailOpenContractError",
    "build_temporal_fail_open_artifacts",
    "validate_temporal_fail_open_effective_artifact",
]
