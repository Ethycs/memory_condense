#!/usr/bin/env python3
"""Gold-blind confirmation terminal-policy boundary for arbitrary populations.

This module owns no retriever and no provider.  Its production path consumes
the exact authenticated answer plan emitted by the frozen semantic-global
terminal v5 compiler, preserving that compiler's P/R/L/G selection, budgets,
deduplication, linked backfill, and rendered provider messages byte-for-byte.

The older candidate-backend path remains only as a synthetic mechanism assay.
It authenticates the frozen confirmation treatment, a complete sealed
parent-answer population, and then uses two narrow dependency-injection seams:

* the repository's question-local semantic residual eligibility gate decides
  whether a terminal attempt is applicable; and
* a :class:`TerminalCandidateBackend` returns four receipt-bound P/R/L/G
  candidate planes for an eligible question.

The output is a set of no-clobber namespace checkpoints plus a deterministic
merged preflight.  Eligible rows with retained evidence contain the exact
normalized Terra messages an executor would submit.  Ineligible rows are
explicit byte-preserving parent passthroughs; empty or over-budget terminal
rows are explicit parent fallbacks.  There is intentionally no provider
client, provider flag, retry loop, authorization command, or benchmark reader.

The boundary uses a conservative stdlib UTF-8 byte upper bound as its default
token proxy.  A production caller may inject a different deterministic counter
with a sealed identity, but the counter remains outside this module so loading
the boundary never loads a heavyweight tokenizer.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

from tools.confirmation_contracts import (
    RuntimePolicy,
    SealedJson,
    _decode_treatment,
    _verify_preflight,
    publish_sealed_json,
    read_runtime_policy,
    read_sealed_json,
)
from tools.confirmation_s0_prompt_preflight import (
    FrozenTerraRuntime,
    PREFLIGHT_PROVIDER_INPUT_FORMAT,
    _runtime_from_policy,
)
from tools.matched_eval.contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.confirmation_semantic_helpers import validate_answer_plan
from tools.matched_eval.semantic_global_terminal_adapter import (
    SemanticGlobalTerminalPolicy,
)
from tools.matched_eval.semantic_residual_eligibility import (
    SemanticResidualEligibilityDecision,
    SemanticResidualEligibilityPolicy,
    evaluate_semantic_residual_eligibility,
)
from tools.matched_eval.typed_memory_final_arm import render_final_messages
from tools.confirmation_canonical import (
    canonical_sha256,
    exact_keys,
    require_int,
    require_list,
    require_mapping,
)


PARENT_POPULATION_FORMAT = "memory-condense-confirmation-terminal-parent-population-v1"
PARENT_ROW_FORMAT = f"{PARENT_POPULATION_FORMAT}-row-v1"
ELIGIBILITY_INPUT_FORMAT = f"{PARENT_POPULATION_FORMAT}-eligibility-input-v1"
CANDIDATE_FORMAT = "memory-condense-confirmation-terminal-candidate-v1"
PLANE_FORMAT = "memory-condense-confirmation-terminal-candidate-plane-v1"
PLANES_FORMAT = "memory-condense-confirmation-terminal-candidate-planes-v1"
CHECKPOINT_FORMAT = "memory-condense-confirmation-terminal-policy-checkpoint-v1"
PLAN_ROW_FORMAT = "memory-condense-confirmation-terminal-policy-row-v1"
MERGED_FORMAT = "memory-condense-confirmation-terminal-policy-preflight-v1"
PROVIDER_PAYLOAD_FORMAT = "memory-condense-confirmation-terminal-payload-v1"
V5_PLAN_EXPORT_FORMAT = "memory-condense-confirmation-terminal-v5-plan-export-v1"
V5_PLAN_EXPORT_ROW_FORMAT = f"{V5_PLAN_EXPORT_FORMAT}-row-v1"
V5_CHECKPOINT_FORMAT = "memory-condense-confirmation-terminal-v5-checkpoint-v1"
V5_COMPILATION_FORMAT = "memory-condense-semantic-global-terminal-compilation-v5"
V5_ANSWER_PLAN_FORMAT = "memory-condense-reduced-semantic-global-terminal-assay-v2-answer-plan-v2"
V5_ROUTE_ID = "semantic-global-terminal-terra-answer-v2"

# This order and these independent minima/caps are the population-neutral
# surface shared with semantic_global_terminal_adapter.SemanticGlobalTerminalPolicy.
# We intentionally do not import that live-object module: doing so would load
# the retrieval/tokenizer graph at this firebreak.
PLANE_ORDER = ("P", "R", "L", "G")
PLANE_BUDGETS: Mapping[str, tuple[int, int, int]] = MappingProxyType(
    {
        "P": (16, 1_400, 1),
        "R": (16, 1_600, 1),
        "L": (16, 1_600, 1),
        "G": (24, 2_400, 1),
    }
)
HANDLE_START: Mapping[str, int] = MappingProxyType(
    {"P": 100_000, "R": 200_000, "L": 300_000, "G": 400_000}
)
TERMINAL_SYSTEM_PROMPT = (
    "Answer the dated question using only the supplied P/R/L/G memory evidence. "
    "Resolve conflicts by specificity and time. If the evidence cannot improve "
    "the protected parent prediction, return it unchanged. Return strict JSON "
    'with keys decision ("keep_parent" or "replace"), prediction, and '
    "used_handle_ids."
)

_PARENT_KEYS = {
    "format",
    "status",
    "gold_loaded",
    "physical_provider_calls",
    "policy_manifest_sha256",
    "treatment_file_sha256",
    "treatment_preflight_sha256",
    "question_count",
    "ordered_question_ids_sha256",
    "rows",
    "artifact_identity_sha256",
}
_PARENT_ROW_KEYS = {
    "format",
    "question_id",
    "namespace_id",
    "namespace_receipt_sha256",
    "question",
    "question_sha256",
    "dated_question",
    "dated_question_sha256",
    "parent_prediction",
    "parent_prediction_sha256",
    "source_row_receipt_sha256",
    "eligibility_input",
    "row_receipt_sha256",
}
_ELIGIBILITY_KEYS = {
    "format",
    "answer_row",
    "construction_row",
    "prior_answer_row",
    "reconciliation_row",
    "receipt_sha256",
}
_CHECKPOINT_KEYS = {
    "format",
    "status",
    "gold_loaded",
    "physical_provider_calls",
    "bindings",
    "backend_identity_sha256",
    "token_counter_identity_sha256",
    "namespace_id",
    "namespace_receipt_sha256",
    "question_count",
    "ordered_parent_row_receipts_sha256",
    "rows",
    "checkpoint_receipt_sha256",
}
_MERGED_KEYS = {
    "format",
    "status",
    "gold_loaded",
    "physical_provider_calls",
    "provider_execution_available",
    "authorization_released",
    "bindings",
    "runtime",
    "plane_policy",
    "population",
    "execution",
    "namespace_checkpoints",
    "ordered_rows",
    "preflight_identity_sha256",
}
_BINDING_KEYS = {
    "policy_manifest_sha256",
    "treatment_file_sha256",
    "treatment_preflight_sha256",
    "parent_population_sha256",
}
_V5_EXPORT_KEYS = {
    "format",
    "status",
    "gold_loaded",
    "physical_provider_calls",
    "bindings",
    "terminal_compilation_format",
    "route_id",
    "eligible_question_count",
    "ordered_parent_row_receipts_sha256",
    "rows",
    "artifact_identity_sha256",
}
_V5_EXPORT_ROW_KEYS = {
    "format",
    "question_id",
    "namespace_id",
    "namespace_receipt_sha256",
    "parent_row_receipt_sha256",
    "source_row_receipt_sha256",
    "source_question_assay",
    "source_question_assay_receipt_sha256",
    "answer_plan_receipt_sha256",
    "terminal_compilation_receipt_sha256",
    "provider_input_sha256",
    "messages_sha256",
    "row_receipt_sha256",
}
_V5_CHECKPOINT_KEYS = {
    "format",
    "status",
    "gold_loaded",
    "physical_provider_calls",
    "bindings",
    "plan_adapter_identity_sha256",
    "prompt_verifier_identity_sha256",
    "namespace_id",
    "namespace_receipt_sha256",
    "question_count",
    "ordered_parent_row_receipts_sha256",
    "rows",
    "checkpoint_receipt_sha256",
}
_ROUTING_FORBIDDEN_KEYS = frozenset(
    {
        "allowlist",
        "eligible_ordinals",
        "eligible_question_ids",
        "miss_ordinals",
        "ordinal",
        "ordinals",
        "target_ordinals",
        "target_question_ids",
        "validation_ordinals",
        "validation_question_ids",
        "whitelist",
    }
)


class ConfirmationTerminalBoundaryError(ValueError):
    """A terminal policy input, candidate plane, or replay failed closed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationTerminalBoundaryError(message)


def _text(value: object, label: str) -> str:
    try:
        return require_text(value, label)  # type: ignore[arg-type]
    except ValueError as exc:
        raise ConfirmationTerminalBoundaryError(str(exc)) from exc


def _sha(value: object, label: str) -> str:
    try:
        return require_sha256(value, label)  # type: ignore[arg-type]
    except ValueError as exc:
        raise ConfirmationTerminalBoundaryError(str(exc)) from exc


def _self_seal(value: Mapping[str, Any], *, key: str, label: str) -> str:
    declared = _sha(value.get(key), f"{label} {key}")
    body = dict(value)
    body.pop(key, None)
    _require(identity_sha256(body) == declared, f"{label} self-seal differs")
    return declared


def _assert_no_population_routing(value: object, path: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            _require(
                key not in _ROUTING_FORBIDDEN_KEYS,
                f"population-specific routing field is forbidden: {path}.{raw_key}",
            )
            _assert_no_population_routing(child, f"{path}.{raw_key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _assert_no_population_routing(child, f"{path}[{index}]")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _messages_sha256(messages: Sequence[Mapping[str, str]]) -> str:
    return identity_sha256([dict(row) for row in messages])


@dataclass(frozen=True, slots=True)
class TokenCounter:
    """A deterministic prompt counter whose identity is sealed in every plan."""

    identity_sha256: str
    count_text: Callable[[str], int]
    count_messages: Callable[[Sequence[Mapping[str, str]]], int]

    def __post_init__(self) -> None:
        _sha(self.identity_sha256, "token counter identity")
        _require(callable(self.count_text) and callable(self.count_messages), "token counter changed")


def _utf8_count(text: str) -> int:
    return len(text.encode("utf-8"))


def _utf8_message_count(messages: Sequence[Mapping[str, str]]) -> int:
    # A byte is a conservative upper bound for ordinary byte-token encodings;
    # explicit framing keeps the contract safe without loading a tokenizer.
    return sum(_utf8_count(str(row["content"])) + 8 for row in messages) + 8


UTF8_UPPER_BOUND_COUNTER = TokenCounter(
    identity_sha256=identity_sha256(
        {
            "format": "memory-condense-stdlib-utf8-byte-token-upper-bound-v1",
            "bytes_per_content_unit": 1,
            "framing_units_per_message": 8,
            "fixed_framing_units": 8,
        }
    ),
    count_text=_utf8_count,
    count_messages=_utf8_message_count,
)


@dataclass(frozen=True, slots=True)
class TerminalParentRow:
    question_id: str
    namespace_id: str
    namespace_receipt_sha256: str
    question: str
    dated_question: str
    parent_prediction: str
    source_row_receipt_sha256: str
    answer_row: Mapping[str, Any]
    construction_row: Mapping[str, Any]
    prior_answer_row: Mapping[str, Any] | None
    reconciliation_row: Mapping[str, Any] | None
    eligibility_input_receipt_sha256: str
    row_receipt_sha256: str


@dataclass(frozen=True, slots=True)
class ConfirmationTerminalInputs:
    policy: RuntimePolicy
    treatment: SealedJson
    treatment_preflight: SealedJson
    parent_population: SealedJson
    ordered_question_ids_sha256: str
    rows: tuple[TerminalParentRow, ...]
    namespaces: tuple[tuple[str, str, tuple[str, ...]], ...]
    runtime: FrozenTerraRuntime


@dataclass(frozen=True, slots=True)
class ConfirmationTerminalV5PlanExport:
    """Externally sealed, exact answer plans emitted by the frozen v5 compiler."""

    artifact: SealedJson
    rows_by_parent_receipt: Mapping[str, Mapping[str, Any]]
    adapter_identity_sha256: str


@dataclass(frozen=True, slots=True)
class TerminalCandidateRequest:
    """Question-local backend view; population IDs and positions are absent."""

    policy_manifest_sha256: str
    treatment_preflight_sha256: str
    parent_population_sha256: str
    parent_row_receipt_sha256: str
    source_row_receipt_sha256: str
    namespace_id: str
    namespace_receipt_sha256: str
    question: str
    dated_question: str
    parent_prediction: str
    eligibility_receipt_sha256: str


@dataclass(frozen=True, slots=True)
class TerminalCandidate:
    plane: str
    namespace_id: str
    parent_row_receipt_sha256: str
    source_binding_sha256: str
    source_group_handle: str
    text: str
    priority: tuple[int, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(self.plane in PLANE_ORDER, "terminal candidate plane changed")
        _text(self.namespace_id, "candidate namespace")
        _sha(self.parent_row_receipt_sha256, "candidate parent row")
        _sha(self.source_binding_sha256, "candidate source binding")
        _require(
            re.fullmatch(r"G[0-9]{3,9}", self.source_group_handle) is not None,
            "candidate source group is not opaque",
        )
        _text(self.text, "candidate evidence text")
        _require(
            type(self.priority) is tuple
            and bool(self.priority)
            and len(self.priority) <= 16
            and all(type(value) is int for value in self.priority),
            "candidate priority must be a bounded integer tuple",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "terminal candidate changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "format": CANDIDATE_FORMAT,
            "plane": self.plane,
            "namespace_id": self.namespace_id,
            "parent_row_receipt_sha256": self.parent_row_receipt_sha256,
            "source_binding_sha256": self.source_binding_sha256,
            "source_group_handle": self.source_group_handle,
            "text": self.text,
            "text_sha256": hashlib.sha256(self.text.encode("utf-8")).hexdigest(),
            "priority": list(self.priority),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class TerminalCandidatePlane:
    plane: str
    candidates: tuple[TerminalCandidate, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(self.plane in PLANE_ORDER, "candidate plane changed")
        _require(
            type(self.candidates) is tuple
            and all(
                type(row) is TerminalCandidate and row.plane == self.plane
                for row in self.candidates
            ),
            "candidate plane population changed",
        )
        receipts = tuple(row.receipt_sha256 for row in self.candidates)
        _require(len(receipts) == len(set(receipts)), "candidate plane repeats a row")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "candidate plane changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "format": PLANE_FORMAT,
            "plane": self.plane,
            "candidates": [row.projection() for row in self.candidates],
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class TerminalCandidatePlanes:
    backend_identity_sha256: str
    policy_manifest_sha256: str
    parent_row_receipt_sha256: str
    namespace_id: str
    namespace_receipt_sha256: str
    planes: tuple[TerminalCandidatePlane, ...]
    physical_provider_calls: int = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.backend_identity_sha256, "candidate backend"),
            (self.policy_manifest_sha256, "candidate policy"),
            (self.parent_row_receipt_sha256, "candidate parent row"),
            (self.namespace_receipt_sha256, "candidate namespace receipt"),
        ):
            _sha(value, label)
        _text(self.namespace_id, "candidate namespace ID")
        _require(
            type(self.planes) is tuple
            and tuple(row.plane for row in self.planes) == PLANE_ORDER,
            "candidate planes lost exact P/R/L/G order",
        )
        all_receipts = tuple(
            row.receipt_sha256 for plane in self.planes for row in plane.candidates
        )
        _require(
            len(all_receipts) == len(set(all_receipts)),
            "candidate receipt crossed planes",
        )
        _require(self.physical_provider_calls == 0, "candidate backend used a provider")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "candidate planes changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="confirmation_terminal_candidates")
        _assert_no_population_routing(self.projection(), "confirmation_terminal_candidates")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "format": PLANES_FORMAT,
            "backend_identity_sha256": self.backend_identity_sha256,
            "policy_manifest_sha256": self.policy_manifest_sha256,
            "parent_row_receipt_sha256": self.parent_row_receipt_sha256,
            "namespace_id": self.namespace_id,
            "namespace_receipt_sha256": self.namespace_receipt_sha256,
            "planes": [row.projection() for row in self.planes],
            "physical_provider_calls": self.physical_provider_calls,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


class TerminalCandidateBackend(Protocol):
    """Closed, provider-free source of exact question-local P/R/L/G planes."""

    @property
    def identity_sha256(self) -> str: ...

    @property
    def policy_manifest_sha256(self) -> str: ...

    def candidate_planes(self, request: TerminalCandidateRequest) -> TerminalCandidatePlanes: ...


@dataclass(frozen=True, slots=True)
class CallableSemanticGlobalTerminalAdapter:
    """Synthetic candidate-export seam retained for boundary-level assays.

    This seam does not claim semantic-global terminal v5 equivalence.  The
    production path is ``compile_confirmation_terminal_v5_plan_export`` plus
    the v5 execution/merge functions below, which ingest the compiler's exact
    authenticated plan rather than reconstructing candidate selection.
    """

    policy_manifest_sha256: str
    exporter_identity_sha256: str
    exporter: Callable[[TerminalCandidateRequest], TerminalCandidatePlanes]

    def __post_init__(self) -> None:
        _sha(self.policy_manifest_sha256, "production adapter policy")
        _sha(self.exporter_identity_sha256, "production exporter identity")
        _require(callable(self.exporter), "production exporter is not callable")

    @property
    def identity_sha256(self) -> str:
        return identity_sha256(
            {
                "format": "memory-condense-confirmation-semantic-global-export-adapter-v1",
                "policy_manifest_sha256": self.policy_manifest_sha256,
                "exporter_identity_sha256": self.exporter_identity_sha256,
            }
        )

    def candidate_planes(self, request: TerminalCandidateRequest) -> TerminalCandidatePlanes:
        result = self.exporter(request)
        _require(type(result) is TerminalCandidatePlanes, "production exporter changed type")
        return result


@dataclass(frozen=True, slots=True)
class ConfirmationTerminalExecution:
    checkpoint_paths: tuple[Path, ...]
    checkpoint_sha256s: tuple[str, ...]
    backend_identity_sha256: str
    token_counter_identity_sha256: str
    created_count: int
    reused_count: int
    physical_provider_calls: int = 0


def _namespace_schedule(preflight: SealedJson) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    result: list[tuple[str, str, tuple[str, ...]]] = []
    seen: set[str] = set()
    for index, raw in enumerate(require_list(preflight.payload.get("namespaces"), "preflight namespaces")):
        row = require_mapping(raw, f"preflight namespace {index}")
        namespace_id = _text(row.get("namespace_id"), f"namespace {index} ID")
        receipt = _sha(row.get("namespace_receipt_sha256"), f"namespace {index} receipt")
        raw_ids = require_list(row.get("question_ids"), f"namespace {index} question IDs")
        ids = tuple(_text(value, f"namespace {index} question ID") for value in raw_ids)
        _require(len(ids) == len(set(ids)) and not (seen & set(ids)), "preflight namespaces overlap")
        seen.update(ids)
        result.append((namespace_id, receipt, ids))
    return tuple(result)


def _decode_parent_row(
    raw: object,
    *,
    index: int,
    expected_question_id: str,
    expected_question: str,
    expected_dated_question: str,
    expected_namespace: tuple[str, str],
) -> TerminalParentRow:
    label = f"terminal parent row {index}"
    row = require_mapping(raw, label)
    exact_keys(row, _PARENT_ROW_KEYS, label)
    _require(row["format"] == PARENT_ROW_FORMAT, f"{label} format changed")
    question_id = _text(row["question_id"], f"{label} question ID")
    question = _text(row["question"], f"{label} question")
    dated_question = _text(row["dated_question"], f"{label} dated question")
    namespace_id = _text(row["namespace_id"], f"{label} namespace")
    namespace_receipt = _sha(row["namespace_receipt_sha256"], f"{label} namespace receipt")
    parent_prediction = _text(row["parent_prediction"], f"{label} prediction")
    _require(question_id == expected_question_id, f"{label} is missing or reordered")
    _require(question == expected_question and dated_question == expected_dated_question, f"{label} question changed")
    _require((namespace_id, namespace_receipt) == expected_namespace, f"{label} escaped its namespace")
    _require(row["question_sha256"] == hashlib.sha256(question.encode()).hexdigest(), f"{label} question hash changed")
    _require(row["dated_question_sha256"] == hashlib.sha256(dated_question.encode()).hexdigest(), f"{label} dated-question hash changed")
    _require(row["parent_prediction_sha256"] == hashlib.sha256(parent_prediction.encode()).hexdigest(), f"{label} prediction hash changed")
    source_receipt = _sha(row["source_row_receipt_sha256"], f"{label} source receipt")
    eligibility = require_mapping(row["eligibility_input"], f"{label} eligibility input")
    exact_keys(eligibility, _ELIGIBILITY_KEYS, f"{label} eligibility input")
    _require(eligibility["format"] == ELIGIBILITY_INPUT_FORMAT, f"{label} eligibility format changed")
    eligibility_receipt = _self_seal(eligibility, key="receipt_sha256", label=f"{label} eligibility input")
    answer_row = require_mapping(eligibility["answer_row"], f"{label} answer row")
    construction_row = require_mapping(eligibility["construction_row"], f"{label} construction row")
    prior_raw = eligibility["prior_answer_row"]
    reconciliation_raw = eligibility["reconciliation_row"]
    prior = None if prior_raw is None else require_mapping(prior_raw, f"{label} prior answer")
    reconciliation = None if reconciliation_raw is None else require_mapping(reconciliation_raw, f"{label} reconciliation")
    _require(answer_row.get("prediction") == parent_prediction, f"{label} eligibility answer differs from parent")
    assert_gold_blind(eligibility, path=f"confirmation_terminal_parent.rows[{index}].eligibility")
    _assert_no_population_routing(eligibility, f"confirmation_terminal_parent.rows[{index}].eligibility")
    row_receipt = _self_seal(row, key="row_receipt_sha256", label=label)
    return TerminalParentRow(
        question_id=question_id,
        namespace_id=namespace_id,
        namespace_receipt_sha256=namespace_receipt,
        question=question,
        dated_question=dated_question,
        parent_prediction=parent_prediction,
        source_row_receipt_sha256=source_receipt,
        answer_row=MappingProxyType(dict(answer_row)),
        construction_row=MappingProxyType(dict(construction_row)),
        prior_answer_row=None if prior is None else MappingProxyType(dict(prior)),
        reconciliation_row=None if reconciliation is None else MappingProxyType(dict(reconciliation)),
        eligibility_input_receipt_sha256=eligibility_receipt,
        row_receipt_sha256=row_receipt,
    )


def load_confirmation_terminal_inputs(
    *,
    runtime_policy_path: str | Path,
    expected_runtime_policy_sha256: str,
    treatment_input_path: str | Path,
    expected_treatment_input_sha256: str,
    treatment_preflight_path: str | Path,
    expected_treatment_preflight_sha256: str,
    parent_population_path: str | Path,
    expected_parent_population_sha256: str,
) -> ConfirmationTerminalInputs:
    """Authenticate all label-free ancestors before terminal policy runs."""

    treatment_artifact = read_sealed_json(treatment_input_path, expected_sha256=expected_treatment_input_sha256, label="label-free confirmation treatment")
    treatment, _ = _decode_treatment(treatment_artifact)
    policy = read_runtime_policy(
        runtime_policy_path,
        expected_runtime_policy_sha256=expected_runtime_policy_sha256,
        treatment=treatment,
    )
    preflight = read_sealed_json(treatment_preflight_path, expected_sha256=expected_treatment_preflight_sha256, label="label-free confirmation preflight")
    _verify_preflight(preflight, treatment)
    parent = read_sealed_json(parent_population_path, expected_sha256=expected_parent_population_sha256, label="sealed terminal parent population")
    value = parent.payload
    exact_keys(value, _PARENT_KEYS, "terminal parent population")
    _require(value["format"] == PARENT_POPULATION_FORMAT and value["status"] == "complete", "terminal parent population is not complete")
    _require(value["gold_loaded"] is False and value["physical_provider_calls"] == 0, "terminal parent population crossed a firebreak")
    _require(value["policy_manifest_sha256"] == policy.sha256, "terminal parent binds another policy")
    _require(value["treatment_file_sha256"] == treatment_artifact.sha256, "terminal parent binds another treatment")
    _require(value["treatment_preflight_sha256"] == preflight.sha256, "terminal parent binds another preflight")
    count = require_int(value["question_count"], "terminal parent question count", minimum=1)
    _require(count == len(treatment.samples), "terminal parent population is incomplete")
    order = _sha(value["ordered_question_ids_sha256"], "terminal parent ordered IDs")
    _require(order == treatment.ordered_question_ids_sha256, "terminal parent binds another order")
    _self_seal(value, key="artifact_identity_sha256", label="terminal parent population")
    assert_gold_blind(value, path="confirmation_terminal_parent")
    schedule = _namespace_schedule(preflight)
    membership = {
        question_id: (namespace_id, receipt)
        for namespace_id, receipt, ids in schedule
        for question_id in ids
    }
    question_ids = tuple(sample.sample_id for sample in treatment.samples)
    _require(set(membership) == set(question_ids), "preflight namespace population changed")
    raw_rows = require_list(value["rows"], "terminal parent rows")
    _require(len(raw_rows) == count, "terminal parent has missing rows")
    rows: list[TerminalParentRow] = []
    for index, (raw, sample) in enumerate(zip(raw_rows, treatment.samples, strict=True)):
        _require(len(sample.questions) == 1, "treatment sample question count changed")
        question = sample.questions[0]
        rows.append(
            _decode_parent_row(
                raw,
                index=index,
                expected_question_id=sample.sample_id,
                expected_question=question.question,
                expected_dated_question=question.dated_question,
                expected_namespace=membership[sample.sample_id],
            )
        )
    observed = tuple(row.question_id for row in rows)
    _require(observed == question_ids and len(set(observed)) == count, "terminal parent rows repeat or reorder IDs")
    _require(canonical_sha256(list(observed)) == order, "terminal parent row-order root differs")
    return ConfirmationTerminalInputs(
        policy=policy,
        treatment=treatment_artifact,
        treatment_preflight=preflight,
        parent_population=parent,
        ordered_question_ids_sha256=order,
        rows=tuple(rows),
        namespaces=schedule,
        runtime=_runtime_from_policy(policy),
    )


def _eligibility(row: TerminalParentRow) -> SemanticResidualEligibilityDecision:
    try:
        return evaluate_semantic_residual_eligibility(
            row.answer_row,
            row.construction_row,
            prior_answer_row=row.prior_answer_row,
            reconciliation_row=row.reconciliation_row,
            policy=SemanticResidualEligibilityPolicy(),
        )
    except MatchedEvalContractError as exc:
        raise ConfirmationTerminalBoundaryError(f"terminal eligibility failed: {exc}") from exc


def _validate_frozen_v5_question_plan(
    parent: TerminalParentRow,
    source_index: int,
    raw_question: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[dict[str, str], ...]]:
    """Authenticate the exact answer plan produced by the frozen v5 compiler."""

    question = require_mapping(raw_question, "frozen terminal-v5 question assay")
    question_receipt = _self_seal(
        question,
        key="question_assay_receipt_sha256",
        label="frozen terminal-v5 question assay",
    )
    try:
        plan = validate_answer_plan(
            require_mapping(
                question.get("terminal_answer_plan"),
                "frozen terminal-v5 answer plan",
            ),
            question,
        )
        expected_policy = SemanticGlobalTerminalPolicy().projection()
    except (MatchedEvalContractError, ValueError) as exc:
        raise ConfirmationTerminalBoundaryError(
            f"frozen terminal-v5 answer plan failed authentication: {exc}"
        ) from exc
    compilation = require_mapping(
        plan.get("terminal_compilation"),
        "frozen terminal-v5 compilation",
    )
    selections = require_list(
        compilation.get("plane_selections"),
        "frozen terminal-v5 plane selections",
    )
    budget_by_plane = {
        str(item["plane"]): item
        for item in require_list(
            expected_policy["plane_budgets"],
            "frozen terminal-v5 policy budgets",
        )
    }
    _require(
        plan.get("format") == V5_ANSWER_PLAN_FORMAT
        and plan.get("route_id") == V5_ROUTE_ID
        and compilation.get("format") == V5_COMPILATION_FORMAT
        and compilation.get("policy") == expected_policy
        and "post_dedup_backfill" in compilation,
        "terminal plan is not the exact linked-and-backfilled v5 policy",
    )
    _require(
        len(selections) == len(PLANE_ORDER)
        and tuple(item.get("plane") for item in selections) == PLANE_ORDER
        and all(
            item.get("max_items") == budget_by_plane[str(item.get("plane"))]["max_items"]
            and item.get("evidence_token_cap")
            == budget_by_plane[str(item.get("plane"))]["evidence_token_cap"]
            and item.get("minimum_items")
            == budget_by_plane[str(item.get("plane"))]["minimum_items"]
            for item in selections
        ),
        "terminal-v5 plane order or independent budgets changed",
    )
    _require(
        question.get("ordinal") == source_index
        and question.get("question_id") == parent.question_id
        and question.get("namespace_id") == parent.namespace_id
        and question.get("question_sha256")
        == hashlib.sha256(parent.question.encode()).hexdigest()
        and question.get("dated_question_sha256")
        == hashlib.sha256(parent.dated_question.encode()).hexdigest()
        and plan.get("question_id") == parent.question_id
        and plan.get("question_sha256")
        == hashlib.sha256(parent.question.encode()).hexdigest()
        and plan.get("dated_question") == parent.dated_question
        and plan.get("dated_question_sha256")
        == hashlib.sha256(parent.dated_question.encode()).hexdigest()
        and plan.get("parent_prediction") == parent.parent_prediction
        and plan.get("parent_prediction_sha256")
        == hashlib.sha256(parent.parent_prediction.encode()).hexdigest()
        and question_receipt == question.get("question_assay_receipt_sha256"),
        "terminal-v5 plan escaped its exact parent row",
    )
    provider_input = require_mapping(
        plan.get("provider_input"),
        "frozen terminal-v5 typed provider input",
    )
    messages = tuple(dict(item) for item in render_final_messages(provider_input))
    _require(
        identity_sha256([dict(item) for item in messages])
        == plan.get("messages_sha256"),
        "terminal-v5 typed prompt bytes changed",
    )
    assert_gold_blind(question, path="confirmation_terminal_v5_question_plan")
    return dict(plan), messages


def compile_confirmation_terminal_v5_plan_export(
    inputs: ConfirmationTerminalInputs,
    *,
    frozen_question_assays: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind exact frozen-compiler v5 plans to content-derived eligible parents."""

    _require(type(inputs) is ConfirmationTerminalInputs, "terminal input boundary changed")
    eligible = tuple(
        (index, row)
        for index, row in enumerate(inputs.rows)
        if _eligibility(row).eligible
    )
    questions = tuple(frozen_question_assays)
    _require(
        len(questions) == len(eligible)
        and all(type(question) is dict for question in questions),
        "frozen terminal-v5 plan population differs from derived eligibility",
    )
    rows: list[dict[str, Any]] = []
    for (source_index, parent), question in zip(eligible, questions, strict=True):
        plan, _messages = _validate_frozen_v5_question_plan(
            parent,
            source_index,
            question,
        )
        body = {
            "format": V5_PLAN_EXPORT_ROW_FORMAT,
            "question_id": parent.question_id,
            "namespace_id": parent.namespace_id,
            "namespace_receipt_sha256": parent.namespace_receipt_sha256,
            "parent_row_receipt_sha256": parent.row_receipt_sha256,
            "source_row_receipt_sha256": parent.source_row_receipt_sha256,
            "source_question_assay": dict(question),
            "source_question_assay_receipt_sha256": question[
                "question_assay_receipt_sha256"
            ],
            "answer_plan_receipt_sha256": plan["answer_plan_receipt_sha256"],
            "terminal_compilation_receipt_sha256": plan[
                "terminal_compilation_receipt_sha256"
            ],
            "provider_input_sha256": plan["provider_input_sha256"],
            "messages_sha256": plan["messages_sha256"],
        }
        rows.append({**body, "row_receipt_sha256": identity_sha256(body)})
    bindings = {
        **_bindings(inputs),
        "frozen_compiler_answer_plan_format": V5_ANSWER_PLAN_FORMAT,
    }
    body = {
        "format": V5_PLAN_EXPORT_FORMAT,
        "status": "complete",
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "bindings": bindings,
        "terminal_compilation_format": V5_COMPILATION_FORMAT,
        "route_id": V5_ROUTE_ID,
        "eligible_question_count": len(rows),
        "ordered_parent_row_receipts_sha256": identity_sha256(
            [row.row_receipt_sha256 for _index, row in eligible]
        ),
        "rows": rows,
    }
    assert_gold_blind(body, path="confirmation_terminal_v5_plan_export")
    return {**body, "artifact_identity_sha256": identity_sha256(body)}


def publish_confirmation_terminal_v5_plan_export(
    inputs: ConfirmationTerminalInputs,
    *,
    frozen_question_assays: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> tuple[SealedJson, bool]:
    return publish_sealed_json(
        output_path,
        compile_confirmation_terminal_v5_plan_export(
            inputs,
            frozen_question_assays=frozen_question_assays,
        ),
    )


def load_confirmation_terminal_v5_plan_export(
    inputs: ConfirmationTerminalInputs,
    *,
    path: str | Path,
    expected_sha256: str,
) -> ConfirmationTerminalV5PlanExport:
    artifact = read_sealed_json(
        path,
        expected_sha256=expected_sha256,
        label="sealed confirmation terminal-v5 plan export",
    )
    value = artifact.payload
    exact_keys(value, _V5_EXPORT_KEYS, "confirmation terminal-v5 plan export")
    rows = require_list(value.get("rows"), "confirmation terminal-v5 export rows")
    for index, raw in enumerate(rows):
        row = require_mapping(raw, f"confirmation terminal-v5 export row {index}")
        exact_keys(row, _V5_EXPORT_ROW_KEYS, f"confirmation terminal-v5 export row {index}")
        _self_seal(row, key="row_receipt_sha256", label=f"confirmation terminal-v5 export row {index}")
    rebuilt = compile_confirmation_terminal_v5_plan_export(
        inputs,
        frozen_question_assays=[
            require_mapping(row["source_question_assay"], "source question assay")
            for row in rows
        ],
    )
    _require(value == rebuilt, "confirmation terminal-v5 plan export differs from replay")
    _self_seal(value, key="artifact_identity_sha256", label="confirmation terminal-v5 plan export")
    by_parent = {
        str(row["parent_row_receipt_sha256"]): MappingProxyType(dict(row))
        for row in rows
    }
    _require(len(by_parent) == len(rows), "confirmation terminal-v5 export repeats a parent")
    adapter_identity = identity_sha256(
        {
            "format": "memory-condense-confirmation-frozen-terminal-v5-plan-adapter-v1",
            "plan_export_sha256": artifact.sha256,
            "terminal_compilation_format": V5_COMPILATION_FORMAT,
            "selection_reimplemented": False,
            "typed_prompt_reencoded": False,
        }
    )
    return ConfirmationTerminalV5PlanExport(
        artifact=artifact,
        rows_by_parent_receipt=MappingProxyType(by_parent),
        adapter_identity_sha256=adapter_identity,
    )


def _validate_candidate_planes(
    result: TerminalCandidatePlanes,
    *,
    backend: TerminalCandidateBackend,
    request: TerminalCandidateRequest,
) -> None:
    _require(type(result) is TerminalCandidatePlanes, "candidate backend changed result type")
    _require(result.backend_identity_sha256 == backend.identity_sha256, "candidate result binds another backend")
    _require(result.policy_manifest_sha256 == backend.policy_manifest_sha256 == request.policy_manifest_sha256, "candidate backend binds another policy")
    _require(result.parent_row_receipt_sha256 == request.parent_row_receipt_sha256, "candidate result binds another parent row")
    _require((result.namespace_id, result.namespace_receipt_sha256) == (request.namespace_id, request.namespace_receipt_sha256), "candidate result escaped its namespace")
    for plane in result.planes:
        for candidate in plane.candidates:
            _require(candidate.namespace_id == request.namespace_id and candidate.parent_row_receipt_sha256 == request.parent_row_receipt_sha256, "candidate escaped its question-local namespace")


def _select_candidates(
    result: TerminalCandidatePlanes,
    *,
    token_counter: TokenCounter,
) -> tuple[tuple[TerminalCandidate, ...], list[dict[str, Any]], dict[str, Any]]:
    selected_by_plane: dict[str, list[TerminalCandidate]] = {}
    audits: list[dict[str, Any]] = []
    for plane_row in result.planes:
        maximum, cap, minimum = PLANE_BUDGETS[plane_row.plane]
        ordered = sorted(
            plane_row.candidates,
            key=lambda row: (tuple(-value for value in row.priority), row.receipt_sha256),
        )
        selected: list[TerminalCandidate] = []
        skipped: list[str] = []
        used = 0
        for candidate in ordered:
            amount = token_counter.count_text(candidate.text)
            _require(type(amount) is int and amount > 0, "candidate token proxy changed")
            if len(selected) >= maximum or used + amount > cap:
                skipped.append(candidate.receipt_sha256)
                continue
            selected.append(candidate)
            used += amount
        _require(not ordered or len(selected) >= minimum, f"{plane_row.plane} candidates cannot satisfy their independent minimum")
        selected_by_plane[plane_row.plane] = selected
        audit_body = {
            "format": "memory-condense-confirmation-terminal-plane-selection-v1",
            "plane": plane_row.plane,
            "candidate_plane_receipt_sha256": plane_row.receipt_sha256,
            "candidate_receipt_sha256s": [row.receipt_sha256 for row in ordered],
            "selected_candidate_receipt_sha256s": [row.receipt_sha256 for row in selected],
            "skipped_candidate_receipt_sha256s": skipped,
            "max_items": maximum,
            "evidence_token_proxy_cap": cap,
            "minimum_items": minimum,
            "selected_token_proxy": used,
            "selection_budget_non_borrowable": True,
            "skip_oversized_and_continue": True,
        }
        audits.append({**audit_body, "receipt_sha256": identity_sha256(audit_body)})
    retained: list[TerminalCandidate] = []
    duplicate_rows: list[dict[str, str]] = []
    owner_by_text: dict[str, TerminalCandidate] = {}
    for plane in PLANE_ORDER:
        for candidate in selected_by_plane[plane]:
            span = hashlib.sha256(candidate.text.encode("utf-8")).hexdigest()
            owner = owner_by_text.get(span)
            if owner is None:
                owner_by_text[span] = candidate
                retained.append(candidate)
            else:
                duplicate_rows.append(
                    {
                        "duplicate_candidate_receipt_sha256": candidate.receipt_sha256,
                        "owner_candidate_receipt_sha256": owner.receipt_sha256,
                        "text_sha256": span,
                    }
                )
    dedup_body = {
        "format": "memory-condense-confirmation-terminal-post-selection-dedup-v1",
        "dedup_after_all_plane_selection": True,
        "retention_order": list(PLANE_ORDER),
        "retained_candidate_receipt_sha256s": [row.receipt_sha256 for row in retained],
        "duplicates": duplicate_rows,
    }
    return tuple(retained), audits, {**dedup_body, "receipt_sha256": identity_sha256(dedup_body)}


def _provider_messages(
    row: TerminalParentRow,
    retained: Sequence[TerminalCandidate],
) -> tuple[dict[str, str], ...]:
    by_plane: dict[str, list[dict[str, str]]] = {plane: [] for plane in PLANE_ORDER}
    counters = {plane: HANDLE_START[plane] for plane in PLANE_ORDER}
    for candidate in retained:
        handle = f"H{counters[candidate.plane]:06d}"
        counters[candidate.plane] += 1
        by_plane[candidate.plane].append(
            {
                "handle_id": handle,
                "source_group_handle": candidate.source_group_handle,
                "text": candidate.text,
            }
        )
    payload = {
        "format": PROVIDER_PAYLOAD_FORMAT,
        "dated_question": row.dated_question,
        "protected_parent_fallback": {
            "prediction": row.parent_prediction,
            "prediction_sha256": hashlib.sha256(row.parent_prediction.encode()).hexdigest(),
        },
        "evidence_planes": [
            {"plane": plane, "evidence": by_plane[plane]} for plane in PLANE_ORDER
        ],
        "response_schema": {
            "decision": "keep_parent|replace",
            "prediction": "nonempty exact text",
            "used_handle_ids": ["H100000"],
        },
    }
    messages = (
        {"role": "system", "content": TERMINAL_SYSTEM_PROMPT},
        {"role": "user", "content": _canonical_json(payload)},
    )
    return tuple(dict(message) for message in messages)


def _provider_input(messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    body = {
        "format": PREFLIGHT_PROVIDER_INPUT_FORMAT,
        "messages": [dict(row) for row in messages],
        "messages_sha256": _messages_sha256(messages),
    }
    return {**body, "provider_input_receipt_sha256": identity_sha256(body)}


def _compile_row(
    inputs: ConfirmationTerminalInputs,
    row: TerminalParentRow,
    *,
    backend: TerminalCandidateBackend,
    token_counter: TokenCounter,
) -> dict[str, Any]:
    decision = _eligibility(row)
    candidate_result: TerminalCandidatePlanes | None = None
    selections: list[dict[str, Any]] = []
    dedup: dict[str, Any] | None = None
    provider: dict[str, Any] | None = None
    retained: tuple[TerminalCandidate, ...] = ()
    if decision.eligible:
        request = TerminalCandidateRequest(
            policy_manifest_sha256=inputs.policy.sha256,
            treatment_preflight_sha256=inputs.treatment_preflight.sha256,
            parent_population_sha256=inputs.parent_population.sha256,
            parent_row_receipt_sha256=row.row_receipt_sha256,
            source_row_receipt_sha256=row.source_row_receipt_sha256,
            namespace_id=row.namespace_id,
            namespace_receipt_sha256=row.namespace_receipt_sha256,
            question=row.question,
            dated_question=row.dated_question,
            parent_prediction=row.parent_prediction,
            eligibility_receipt_sha256=decision.receipt_sha256,
        )
        candidate_result = backend.candidate_planes(request)
        _validate_candidate_planes(candidate_result, backend=backend, request=request)
        retained, selections, dedup = _select_candidates(candidate_result, token_counter=token_counter)
        if retained:
            messages = _provider_messages(row, retained)
            prompt_proxy = token_counter.count_messages(messages)
            _require(type(prompt_proxy) is int and prompt_proxy > 0, "terminal prompt token proxy changed")
            if prompt_proxy <= inputs.runtime.input_token_cap:
                provider = _provider_input(messages)
                disposition = "terminal_provider_required"
            else:
                disposition = "parent_fallback_prompt_over_budget"
        else:
            disposition = "parent_fallback_no_terminal_evidence"
    else:
        disposition = "parent_passthrough"
    would_call = provider is not None
    body = {
        "format": PLAN_ROW_FORMAT,
        "question_id": row.question_id,
        "namespace_id": row.namespace_id,
        "namespace_receipt_sha256": row.namespace_receipt_sha256,
        "parent_population_row_receipt_sha256": row.row_receipt_sha256,
        "source_parent_row_receipt_sha256": row.source_row_receipt_sha256,
        "eligibility": decision.projection(),
        "disposition": disposition,
        "parent_prediction": row.parent_prediction,
        "parent_prediction_sha256": hashlib.sha256(row.parent_prediction.encode()).hexdigest(),
        "fallback_policy": "byte_exact_parent_on_inapplicable_empty_over_budget_or_invalid_completion-v1",
        "candidate_planes": None if candidate_result is None else candidate_result.projection(),
        "plane_selections": selections,
        "post_selection_dedup": dedup,
        "retained_candidate_receipt_sha256s": [candidate.receipt_sha256 for candidate in retained],
        "provider_input": provider,
        "prompt_token_proxy": None if provider is None else token_counter.count_messages(provider["messages"]),
        "would_call": would_call,
        "physical_provider_calls": 0,
    }
    assert_gold_blind(body, path="confirmation_terminal_policy_row")
    return {**body, "row_receipt_sha256": identity_sha256(body)}


def _bindings(inputs: ConfirmationTerminalInputs) -> dict[str, str]:
    return {
        "policy_manifest_sha256": inputs.policy.sha256,
        "treatment_file_sha256": inputs.treatment.sha256,
        "treatment_preflight_sha256": inputs.treatment_preflight.sha256,
        "parent_population_sha256": inputs.parent_population.sha256,
    }


def _checkpoint_payload(
    inputs: ConfirmationTerminalInputs,
    *,
    namespace_id: str,
    namespace_receipt_sha256: str,
    question_ids: tuple[str, ...],
    rows_by_id: Mapping[str, TerminalParentRow],
    backend: TerminalCandidateBackend,
    token_counter: TokenCounter,
) -> dict[str, Any]:
    rows = [
        _compile_row(inputs, rows_by_id[question_id], backend=backend, token_counter=token_counter)
        for question_id in question_ids
    ]
    body = {
        "format": CHECKPOINT_FORMAT,
        "status": "compiled",
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "bindings": _bindings(inputs),
        "backend_identity_sha256": backend.identity_sha256,
        "token_counter_identity_sha256": token_counter.identity_sha256,
        "namespace_id": namespace_id,
        "namespace_receipt_sha256": namespace_receipt_sha256,
        "question_count": len(rows),
        "ordered_parent_row_receipts_sha256": identity_sha256(
            [rows_by_id[question_id].row_receipt_sha256 for question_id in question_ids]
        ),
        "rows": rows,
    }
    assert_gold_blind(body, path="confirmation_terminal_checkpoint")
    return {**body, "checkpoint_receipt_sha256": identity_sha256(body)}


def execute_confirmation_terminal_policy(
    inputs: ConfirmationTerminalInputs,
    *,
    backend: TerminalCandidateBackend,
    output_root: str | Path,
    token_counter: TokenCounter = UTF8_UPPER_BOUND_COUNTER,
) -> ConfirmationTerminalExecution:
    """Compile/reuse one provider-free checkpoint for every namespace."""

    _require(type(inputs) is ConfirmationTerminalInputs, "terminal input boundary changed")
    _sha(backend.identity_sha256, "terminal backend identity")
    _require(backend.policy_manifest_sha256 == inputs.policy.sha256, "terminal backend binds another frozen policy")
    _require(type(token_counter) is TokenCounter, "terminal token counter changed")
    rows_by_id = {row.question_id: row for row in inputs.rows}
    root = Path(output_root) / "checkpoints"
    paths: list[Path] = []
    digests: list[str] = []
    created_count = 0
    reused_count = 0
    for namespace_id, namespace_receipt, question_ids in inputs.namespaces:
        payload = _checkpoint_payload(
            inputs,
            namespace_id=namespace_id,
            namespace_receipt_sha256=namespace_receipt,
            question_ids=question_ids,
            rows_by_id=rows_by_id,
            backend=backend,
            token_counter=token_counter,
        )
        key = identity_sha256({"namespace_id": namespace_id, "namespace_receipt_sha256": namespace_receipt})
        artifact, created = publish_sealed_json(root / f"{key}.json", payload)
        paths.append(artifact.path)
        digests.append(artifact.sha256)
        created_count += int(created)
        reused_count += int(not created)
    return ConfirmationTerminalExecution(
        checkpoint_paths=tuple(paths),
        checkpoint_sha256s=tuple(digests),
        backend_identity_sha256=backend.identity_sha256,
        token_counter_identity_sha256=token_counter.identity_sha256,
        created_count=created_count,
        reused_count=reused_count,
    )


def _v5_prompt_verifier_identity() -> str:
    return identity_sha256(
        {
            "format": "memory-condense-confirmation-terminal-v5-prompt-verifier-v1",
            "answer_plan_format": V5_ANSWER_PLAN_FORMAT,
            "terminal_compilation_format": V5_COMPILATION_FORMAT,
            "route_id": V5_ROUTE_ID,
        }
    )


def _compile_frozen_v5_row(
    parent: TerminalParentRow,
    *,
    source_index: int,
    export_row: Mapping[str, Any] | None,
) -> dict[str, Any]:
    decision = _eligibility(parent)
    provider: dict[str, Any] | None = None
    typed_provider: dict[str, Any] | None = None
    frozen_plan: dict[str, Any] | None = None
    plane_selections: list[Any] = []
    post_selection_dedup: dict[str, Any] | None = None
    post_dedup_backfill: dict[str, Any] | None = None
    retained_receipts: list[Any] = []
    prompt_tokens: int | None = None
    if decision.eligible:
        _require(export_row is not None, "eligible parent is missing its frozen terminal-v5 plan")
        question = require_mapping(
            export_row.get("source_question_assay"),
            "frozen terminal-v5 source question",
        )
        plan, messages = _validate_frozen_v5_question_plan(
            parent,
            source_index,
            question,
        )
        _require(
            export_row.get("parent_row_receipt_sha256") == parent.row_receipt_sha256
            and export_row.get("source_row_receipt_sha256")
            == parent.source_row_receipt_sha256,
            "terminal-v5 export row escaped its parent binding",
        )
        compilation = require_mapping(
            plan["terminal_compilation"],
            "frozen terminal-v5 compilation",
        )
        typed_provider = dict(
            require_mapping(plan["provider_input"], "frozen terminal-v5 provider input")
        )
        provider = _provider_input(messages)
        _require(
            provider["messages_sha256"] == plan["messages_sha256"],
            "terminal-v5 provider message bytes changed at confirmation boundary",
        )
        prompt_tokens = require_int(
            plan["prompt_token_proxy"],
            "frozen terminal-v5 prompt token proxy",
            minimum=1,
        )
        plane_selections = list(
            require_list(compilation["plane_selections"], "terminal-v5 plane selections")
        )
        post_selection_dedup = dict(
            require_mapping(
                compilation["post_selection_dedup"],
                "terminal-v5 post-selection dedup",
            )
        )
        post_dedup_backfill = dict(
            require_mapping(
                compilation["post_dedup_backfill"],
                "terminal-v5 post-dedup backfill",
            )
        )
        retained_receipts = list(
            require_list(
                post_dedup_backfill["final_retained_candidate_receipt_sha256s"],
                "terminal-v5 final retained candidates",
            )
        )
        frozen_plan = {
            "format": V5_PLAN_EXPORT_ROW_FORMAT,
            "source_export_row_receipt_sha256": export_row["row_receipt_sha256"],
            "source_question_assay_receipt_sha256": export_row[
                "source_question_assay_receipt_sha256"
            ],
            "answer_plan_receipt_sha256": plan["answer_plan_receipt_sha256"],
            "terminal_compilation_format": V5_COMPILATION_FORMAT,
            "terminal_compilation_receipt_sha256": plan[
                "terminal_compilation_receipt_sha256"
            ],
            "provider_input_sha256": plan["provider_input_sha256"],
            "messages_sha256": plan["messages_sha256"],
            "selection_reimplemented": False,
            "typed_prompt_reencoded": False,
        }
        disposition = "terminal_provider_required"
    else:
        _require(export_row is None, "ineligible parent received a terminal-v5 plan")
        disposition = "parent_passthrough"
    body = {
        "format": PLAN_ROW_FORMAT,
        "question_id": parent.question_id,
        "namespace_id": parent.namespace_id,
        "namespace_receipt_sha256": parent.namespace_receipt_sha256,
        "parent_population_row_receipt_sha256": parent.row_receipt_sha256,
        "source_parent_row_receipt_sha256": parent.source_row_receipt_sha256,
        "eligibility": decision.projection(),
        "disposition": disposition,
        "parent_prediction": parent.parent_prediction,
        "parent_prediction_sha256": hashlib.sha256(parent.parent_prediction.encode()).hexdigest(),
        "fallback_policy": "byte_exact_parent_on_inapplicable_or_invalid-completion-v1",
        "frozen_terminal_v5_plan": frozen_plan,
        "typed_provider_input": typed_provider,
        "candidate_planes": None,
        "plane_selections": plane_selections,
        "post_selection_dedup": post_selection_dedup,
        "post_dedup_backfill": post_dedup_backfill,
        "retained_candidate_receipt_sha256s": retained_receipts,
        "provider_input": provider,
        "prompt_token_proxy": prompt_tokens,
        "would_call": provider is not None,
        "physical_provider_calls": 0,
    }
    assert_gold_blind(body, path="confirmation_terminal_v5_policy_row")
    return {**body, "row_receipt_sha256": identity_sha256(body)}


def _v5_bindings(
    inputs: ConfirmationTerminalInputs,
    plan_export: ConfirmationTerminalV5PlanExport,
) -> dict[str, str]:
    return {
        **_bindings(inputs),
        "terminal_v5_plan_export_sha256": plan_export.artifact.sha256,
    }


def _v5_checkpoint_payload(
    inputs: ConfirmationTerminalInputs,
    plan_export: ConfirmationTerminalV5PlanExport,
    *,
    namespace_id: str,
    namespace_receipt_sha256: str,
    question_ids: tuple[str, ...],
) -> dict[str, Any]:
    parent_by_id = {row.question_id: row for row in inputs.rows}
    source_index_by_id = {
        row.question_id: index for index, row in enumerate(inputs.rows)
    }
    rows = [
        _compile_frozen_v5_row(
            parent_by_id[question_id],
            source_index=source_index_by_id[question_id],
            export_row=plan_export.rows_by_parent_receipt.get(
                parent_by_id[question_id].row_receipt_sha256
            ),
        )
        for question_id in question_ids
    ]
    body = {
        "format": V5_CHECKPOINT_FORMAT,
        "status": "compiled",
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "bindings": _v5_bindings(inputs, plan_export),
        "plan_adapter_identity_sha256": plan_export.adapter_identity_sha256,
        "prompt_verifier_identity_sha256": _v5_prompt_verifier_identity(),
        "namespace_id": namespace_id,
        "namespace_receipt_sha256": namespace_receipt_sha256,
        "question_count": len(rows),
        "ordered_parent_row_receipts_sha256": identity_sha256(
            [parent_by_id[question_id].row_receipt_sha256 for question_id in question_ids]
        ),
        "rows": rows,
    }
    return {**body, "checkpoint_receipt_sha256": identity_sha256(body)}


def execute_confirmation_terminal_v5_policy(
    inputs: ConfirmationTerminalInputs,
    *,
    plan_export: ConfirmationTerminalV5PlanExport,
    output_root: str | Path,
) -> ConfirmationTerminalExecution:
    """Compile checkpoints from exact frozen-v5 plans without reselecting evidence."""

    _require(type(inputs) is ConfirmationTerminalInputs, "terminal input boundary changed")
    _require(
        type(plan_export) is ConfirmationTerminalV5PlanExport,
        "terminal-v5 plan export boundary changed",
    )
    root = Path(output_root) / "v5-checkpoints"
    paths: list[Path] = []
    digests: list[str] = []
    created_count = 0
    for namespace_id, namespace_receipt, question_ids in inputs.namespaces:
        payload = _v5_checkpoint_payload(
            inputs,
            plan_export,
            namespace_id=namespace_id,
            namespace_receipt_sha256=namespace_receipt,
            question_ids=question_ids,
        )
        key = identity_sha256(
            {"namespace_id": namespace_id, "namespace_receipt_sha256": namespace_receipt}
        )
        artifact, created = publish_sealed_json(root / f"{key}.json", payload)
        paths.append(artifact.path)
        digests.append(artifact.sha256)
        created_count += int(created)
    return ConfirmationTerminalExecution(
        checkpoint_paths=tuple(paths),
        checkpoint_sha256s=tuple(digests),
        backend_identity_sha256=plan_export.adapter_identity_sha256,
        token_counter_identity_sha256=_v5_prompt_verifier_identity(),
        created_count=created_count,
        reused_count=len(paths) - created_count,
    )


def _validate_checkpoint(
    artifact: SealedJson,
    *,
    inputs: ConfirmationTerminalInputs,
    expected_namespace: tuple[str, str, tuple[str, ...]],
    execution: ConfirmationTerminalExecution,
) -> tuple[dict[str, Any], ...]:
    value = artifact.payload
    exact_keys(value, _CHECKPOINT_KEYS, "terminal checkpoint")
    _require(value["format"] == CHECKPOINT_FORMAT and value["status"] == "compiled", "terminal checkpoint format/status changed")
    _require(value["gold_loaded"] is False and value["physical_provider_calls"] == 0, "terminal checkpoint crossed a firebreak")
    bindings = require_mapping(value["bindings"], "terminal checkpoint bindings")
    exact_keys(bindings, _BINDING_KEYS, "terminal checkpoint bindings")
    _require(dict(bindings) == _bindings(inputs), "terminal checkpoint binds another input")
    _require(value["backend_identity_sha256"] == execution.backend_identity_sha256, "terminal checkpoint backend changed")
    _require(value["token_counter_identity_sha256"] == execution.token_counter_identity_sha256, "terminal checkpoint token counter changed")
    namespace_id, namespace_receipt, question_ids = expected_namespace
    _require((value["namespace_id"], value["namespace_receipt_sha256"]) == (namespace_id, namespace_receipt), "terminal checkpoint escaped its namespace")
    rows = tuple(dict(row) for row in require_list(value["rows"], "terminal checkpoint rows") if type(row) is dict)
    _require(len(rows) == len(question_ids) == value["question_count"], "terminal checkpoint population is incomplete")
    _require(tuple(row.get("question_id") for row in rows) == question_ids, "terminal checkpoint rows are missing or reordered")
    parent_by_id = {row.question_id: row for row in inputs.rows}
    _require(
        value["ordered_parent_row_receipts_sha256"]
        == identity_sha256([parent_by_id[question_id].row_receipt_sha256 for question_id in question_ids]),
        "terminal checkpoint parent population changed",
    )
    _self_seal(value, key="checkpoint_receipt_sha256", label="terminal checkpoint")
    for index, row in enumerate(rows):
        _self_seal(row, key="row_receipt_sha256", label=f"terminal checkpoint row {index}")
        _require(row.get("physical_provider_calls") == 0, "terminal row reports provider calls")
        provider = row.get("provider_input")
        if provider is not None:
            provider_map = require_mapping(provider, f"terminal row {index} provider input")
            _self_seal(provider_map, key="provider_input_receipt_sha256", label=f"terminal row {index} provider input")
            messages = require_list(provider_map.get("messages"), f"terminal row {index} messages")
            _require(provider_map.get("messages_sha256") == identity_sha256(messages), "terminal provider messages changed")
    return rows


def compile_confirmation_terminal_merge(
    inputs: ConfirmationTerminalInputs,
    *,
    execution: ConfirmationTerminalExecution,
) -> dict[str, Any]:
    """Merge sealed checkpoints in preflight namespace order, not path order."""

    _require(execution.physical_provider_calls == 0, "terminal execution used providers")
    _require(len(execution.checkpoint_paths) == len(execution.checkpoint_sha256s) == len(inputs.namespaces), "terminal checkpoint population changed")
    by_namespace: dict[str, SealedJson] = {}
    for path, digest in zip(execution.checkpoint_paths, execution.checkpoint_sha256s, strict=True):
        sealed = read_sealed_json(path, expected_sha256=digest, label="terminal namespace checkpoint")
        receipt = _sha(sealed.payload.get("namespace_receipt_sha256"), "terminal checkpoint namespace receipt")
        _require(receipt not in by_namespace, "terminal namespace checkpoint repeats")
        by_namespace[receipt] = sealed
    _require(set(by_namespace) == {row[1] for row in inputs.namespaces}, "terminal namespace checkpoint set differs")
    ordered_rows: list[dict[str, Any]] = []
    refs: list[dict[str, Any]] = []
    for namespace in inputs.namespaces:
        artifact = by_namespace[namespace[1]]
        rows = _validate_checkpoint(artifact, inputs=inputs, expected_namespace=namespace, execution=execution)
        ordered_rows.extend(rows)
        refs.append(
            {
                "namespace_id": namespace[0],
                "namespace_receipt_sha256": namespace[1],
                "checkpoint_sha256": artifact.sha256,
                "checkpoint_receipt_sha256": artifact.payload["checkpoint_receipt_sha256"],
                "question_count": len(rows),
            }
        )
    ids = tuple(row["question_id"] for row in ordered_rows)
    expected_ids = tuple(row.question_id for row in inputs.rows)
    _require(ids == expected_ids, "terminal merge is incomplete or reordered")
    would_calls = sum(row["would_call"] is True for row in ordered_rows)
    passthroughs = sum(row["disposition"] == "parent_passthrough" for row in ordered_rows)
    fallbacks = sum(str(row["disposition"]).startswith("parent_fallback_") for row in ordered_rows)
    body = {
        "format": MERGED_FORMAT,
        "status": "compiled",
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "provider_execution_available": False,
        "authorization_released": False,
        "bindings": _bindings(inputs),
        "runtime": inputs.runtime.projection(),
        "plane_policy": {
            "plane_order": list(PLANE_ORDER),
            "budgets": [
                {
                    "plane": plane,
                    "max_items": PLANE_BUDGETS[plane][0],
                    "evidence_token_proxy_cap": PLANE_BUDGETS[plane][1],
                    "minimum_items": PLANE_BUDGETS[plane][2],
                }
                for plane in PLANE_ORDER
            ],
            "independent_non_borrowable": True,
            "dedup_after_selection": True,
            "receipt_sha256": identity_sha256(
                {"plane_order": list(PLANE_ORDER), "budgets": [list(PLANE_BUDGETS[p]) for p in PLANE_ORDER]}
            ),
        },
        "population": {
            "question_count": len(ordered_rows),
            "ordered_question_ids_sha256": inputs.ordered_question_ids_sha256,
            "namespace_count": len(refs),
        },
        "execution": {
            "logical_terminal_prompt_count": would_calls,
            "would_call_count": would_calls,
            "would_call_count_status": "exact",
            "parent_passthrough_count": passthroughs,
            "parent_fallback_count": fallbacks,
            "physical_provider_calls": 0,
            "provider_execution_available": False,
            "retry_count": inputs.runtime.retry_count,
            "model": inputs.runtime.model,
            "input_token_cap": inputs.runtime.input_token_cap,
            "output_token_reserve": inputs.runtime.output_token_reserve,
            "hard_complete_chat_token_cap": inputs.runtime.hard_complete_chat_token_cap,
            "backend_identity_sha256": execution.backend_identity_sha256,
            "token_counter_identity_sha256": execution.token_counter_identity_sha256,
        },
        "namespace_checkpoints": refs,
        "ordered_rows": ordered_rows,
    }
    assert_gold_blind(body, path="confirmation_terminal_preflight")
    return {**body, "preflight_identity_sha256": identity_sha256(body)}


def _frozen_v5_policy_projection() -> dict[str, Any]:
    from tools.matched_eval.semantic_global_terminal_adapter import (
        SemanticGlobalTerminalPolicy,
    )

    return dict(SemanticGlobalTerminalPolicy().projection())


def compile_confirmation_terminal_v5_merge(
    inputs: ConfirmationTerminalInputs,
    *,
    plan_export: ConfirmationTerminalV5PlanExport,
    execution: ConfirmationTerminalExecution,
) -> dict[str, Any]:
    """Merge exact-v5 checkpoints without reranking or rebuilding typed prompts."""

    _require(execution.physical_provider_calls == 0, "terminal-v5 execution used providers")
    _require(
        execution.backend_identity_sha256 == plan_export.adapter_identity_sha256
        and execution.token_counter_identity_sha256 == _v5_prompt_verifier_identity(),
        "terminal-v5 execution identity changed",
    )
    _require(
        len(execution.checkpoint_paths)
        == len(execution.checkpoint_sha256s)
        == len(inputs.namespaces),
        "terminal-v5 checkpoint population changed",
    )
    by_namespace: dict[str, SealedJson] = {}
    for path, digest in zip(
        execution.checkpoint_paths,
        execution.checkpoint_sha256s,
        strict=True,
    ):
        sealed = read_sealed_json(
            path,
            expected_sha256=digest,
            label="terminal-v5 namespace checkpoint",
        )
        exact_keys(sealed.payload, _V5_CHECKPOINT_KEYS, "terminal-v5 namespace checkpoint")
        receipt = _sha(
            sealed.payload.get("namespace_receipt_sha256"),
            "terminal-v5 checkpoint namespace receipt",
        )
        _require(receipt not in by_namespace, "terminal-v5 checkpoint repeats a namespace")
        by_namespace[receipt] = sealed
    _require(
        set(by_namespace) == {row[1] for row in inputs.namespaces},
        "terminal-v5 checkpoint namespace set differs",
    )
    ordered_rows: list[dict[str, Any]] = []
    refs: list[dict[str, Any]] = []
    for namespace_id, namespace_receipt, question_ids in inputs.namespaces:
        artifact = by_namespace[namespace_receipt]
        expected = _v5_checkpoint_payload(
            inputs,
            plan_export,
            namespace_id=namespace_id,
            namespace_receipt_sha256=namespace_receipt,
            question_ids=question_ids,
        )
        _require(
            artifact.payload == expected,
            "terminal-v5 checkpoint differs from exact frozen-plan replay",
        )
        rows = require_list(artifact.payload["rows"], "terminal-v5 checkpoint rows")
        ordered_rows.extend(dict(row) for row in rows)
        refs.append(
            {
                "namespace_id": namespace_id,
                "namespace_receipt_sha256": namespace_receipt,
                "checkpoint_sha256": artifact.sha256,
                "checkpoint_receipt_sha256": artifact.payload[
                    "checkpoint_receipt_sha256"
                ],
                "question_count": len(rows),
            }
        )
    _require(
        tuple(row["question_id"] for row in ordered_rows)
        == tuple(row.question_id for row in inputs.rows),
        "terminal-v5 merge is incomplete or reordered",
    )
    would_calls = sum(row["would_call"] is True for row in ordered_rows)
    passthroughs = len(ordered_rows) - would_calls
    _require(
        would_calls == len(plan_export.rows_by_parent_receipt),
        "terminal-v5 exact plan population changed during merge",
    )
    policy_body = {
        "authority": "authenticated-frozen-semantic-global-terminal-v5-plan",
        "terminal_compilation_format": V5_COMPILATION_FORMAT,
        "terminal_policy": _frozen_v5_policy_projection(),
        "selection_reimplemented": False,
        "typed_prompt_reencoded": False,
    }
    body = {
        "format": MERGED_FORMAT,
        "status": "compiled",
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "provider_execution_available": False,
        "authorization_released": False,
        "bindings": _v5_bindings(inputs, plan_export),
        "runtime": inputs.runtime.projection(),
        "plane_policy": {
            **policy_body,
            "receipt_sha256": identity_sha256(policy_body),
        },
        "population": {
            "question_count": len(ordered_rows),
            "ordered_question_ids_sha256": inputs.ordered_question_ids_sha256,
            "namespace_count": len(refs),
        },
        "execution": {
            "logical_terminal_prompt_count": would_calls,
            "would_call_count": would_calls,
            "would_call_count_status": "exact",
            "parent_passthrough_count": passthroughs,
            "parent_fallback_count": 0,
            "physical_provider_calls": 0,
            "provider_execution_available": False,
            "retry_count": inputs.runtime.retry_count,
            "model": inputs.runtime.model,
            "input_token_cap": inputs.runtime.input_token_cap,
            "output_token_reserve": inputs.runtime.output_token_reserve,
            "hard_complete_chat_token_cap": inputs.runtime.hard_complete_chat_token_cap,
            "backend_identity_sha256": execution.backend_identity_sha256,
            "token_counter_identity_sha256": execution.token_counter_identity_sha256,
            "selection_reimplemented": False,
            "typed_prompt_reencoded": False,
        },
        "namespace_checkpoints": refs,
        "ordered_rows": ordered_rows,
    }
    assert_gold_blind(body, path="confirmation_terminal_v5_preflight")
    return {**body, "preflight_identity_sha256": identity_sha256(body)}


def publish_confirmation_terminal_v5_merge(
    inputs: ConfirmationTerminalInputs,
    *,
    plan_export: ConfirmationTerminalV5PlanExport,
    execution: ConfirmationTerminalExecution,
    output_path: str | Path,
) -> tuple[SealedJson, bool]:
    return publish_sealed_json(
        output_path,
        compile_confirmation_terminal_v5_merge(
            inputs,
            plan_export=plan_export,
            execution=execution,
        ),
    )


def replay_confirmation_terminal_v5_policy(
    inputs: ConfirmationTerminalInputs,
    *,
    plan_export: ConfirmationTerminalV5PlanExport,
    checkpoint_root: str | Path,
    source_preflight_path: str | Path,
    expected_source_preflight_sha256: str,
    replay_output_path: str | Path,
) -> tuple[SealedJson, bool]:
    source = read_sealed_json(
        source_preflight_path,
        expected_sha256=expected_source_preflight_sha256,
        label="terminal-v5 policy preflight",
    )
    exact_keys(source.payload, _MERGED_KEYS, "terminal-v5 policy preflight")
    _self_seal(
        source.payload,
        key="preflight_identity_sha256",
        label="terminal-v5 policy preflight",
    )
    execution = execute_confirmation_terminal_v5_policy(
        inputs,
        plan_export=plan_export,
        output_root=checkpoint_root,
    )
    replayed = compile_confirmation_terminal_v5_merge(
        inputs,
        plan_export=plan_export,
        execution=execution,
    )
    _require(replayed == source.payload, "terminal-v5 policy replay differs")
    artifact, created = publish_sealed_json(replay_output_path, replayed)
    _require(artifact.sha256 == source.sha256, "terminal-v5 policy replay seal differs")
    return artifact, created


def publish_confirmation_terminal_merge(
    inputs: ConfirmationTerminalInputs,
    *,
    execution: ConfirmationTerminalExecution,
    output_path: str | Path,
) -> tuple[SealedJson, bool]:
    return publish_sealed_json(output_path, compile_confirmation_terminal_merge(inputs, execution=execution))


def replay_confirmation_terminal_policy(
    inputs: ConfirmationTerminalInputs,
    *,
    backend: TerminalCandidateBackend,
    checkpoint_root: str | Path,
    source_preflight_path: str | Path,
    expected_source_preflight_sha256: str,
    replay_output_path: str | Path,
    token_counter: TokenCounter = UTF8_UPPER_BOUND_COUNTER,
) -> tuple[SealedJson, bool]:
    """Re-run local policy and require a byte-identical merged preflight."""

    source = read_sealed_json(source_preflight_path, expected_sha256=expected_source_preflight_sha256, label="terminal policy preflight")
    exact_keys(source.payload, _MERGED_KEYS, "terminal policy preflight")
    _self_seal(source.payload, key="preflight_identity_sha256", label="terminal policy preflight")
    execution = execute_confirmation_terminal_policy(inputs, backend=backend, output_root=checkpoint_root, token_counter=token_counter)
    replayed = compile_confirmation_terminal_merge(inputs, execution=execution)
    _require(replayed == source.payload, "terminal policy replay differs")
    artifact, created = publish_sealed_json(replay_output_path, replayed)
    _require(artifact.sha256 == source.sha256, "terminal policy replay seal differs")
    return artifact, created


__all__ = [
    "CANDIDATE_FORMAT",
    "CHECKPOINT_FORMAT",
    "CallableSemanticGlobalTerminalAdapter",
    "ConfirmationTerminalBoundaryError",
    "ConfirmationTerminalExecution",
    "ConfirmationTerminalInputs",
    "ConfirmationTerminalV5PlanExport",
    "ELIGIBILITY_INPUT_FORMAT",
    "MERGED_FORMAT",
    "PARENT_POPULATION_FORMAT",
    "PARENT_ROW_FORMAT",
    "PLANE_BUDGETS",
    "PLANE_ORDER",
    "PLANES_FORMAT",
    "PLAN_ROW_FORMAT",
    "V5_ANSWER_PLAN_FORMAT",
    "V5_CHECKPOINT_FORMAT",
    "V5_COMPILATION_FORMAT",
    "V5_PLAN_EXPORT_FORMAT",
    "V5_PLAN_EXPORT_ROW_FORMAT",
    "V5_ROUTE_ID",
    "TerminalCandidate",
    "TerminalCandidateBackend",
    "TerminalCandidatePlane",
    "TerminalCandidatePlanes",
    "TerminalCandidateRequest",
    "TokenCounter",
    "UTF8_UPPER_BOUND_COUNTER",
    "compile_confirmation_terminal_merge",
    "compile_confirmation_terminal_v5_merge",
    "compile_confirmation_terminal_v5_plan_export",
    "execute_confirmation_terminal_policy",
    "execute_confirmation_terminal_v5_policy",
    "load_confirmation_terminal_v5_plan_export",
    "load_confirmation_terminal_inputs",
    "publish_confirmation_terminal_merge",
    "publish_confirmation_terminal_v5_merge",
    "publish_confirmation_terminal_v5_plan_export",
    "replay_confirmation_terminal_policy",
    "replay_confirmation_terminal_v5_policy",
]
