#!/usr/bin/env python3
"""Promote the frozen P/R/L/G terminal policy to locked validation100.

The work population is derived exclusively from the authenticated R7
eligibility gate.  The CLI deliberately exposes no ordinal selector.  Eligible
rows run through the existing one-open-per-namespace V7 builder and terminal
compiler; noneligible rows preserve their exact sealed V3 prediction.

This module is provider-free.  Construction and replay retain no transformer
token state and publish a complete ordered 100-row artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from tools import run_locked_semantic_residual_construction_v4 as r7_cli  # noqa: E402
from tools import run_locked_specialist_final_reconcile_v3 as v3_cli  # noqa: E402
from tools import run_reduced_semantic_global_completion_assay as v7_cli  # noqa: E402
from tools import run_reduced_semantic_global_terminal_assay as terminal_cli  # noqa: E402
from tools import run_reduced_source_group_reinjection_assay as v6_cli  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.semantic_global_completion import (  # noqa: E402
    SemanticGlobalCompletionPolicy,
)
from tools.matched_eval import semantic_residual_search as residual  # noqa: E402
from tools.matched_eval.semantic_global_terminal_adapter import (  # noqa: E402
    SemanticGlobalTerminalPolicy,
    TerminalSealedSources,
)
from tools.matched_eval.source_group_reinjection import (  # noqa: E402
    SourceGroupReinjectionPolicy,
)


FORMAT = "memory-condense-locked-semantic-global-terminal-full100-construction-v1"
ROW_FORMAT = f"{FORMAT}-question-row-v1"
NAMESPACE_FORMAT = f"{FORMAT}-namespace-receipt-v1"
SOURCE_BINDINGS_FORMAT = f"{FORMAT}-source-bindings-v1"
POLICY_BINDINGS_FORMAT = f"{FORMAT}-policy-bindings-v1"
POPULATION_FORMAT = f"{FORMAT}-gate-derived-population-v1"
RESIDENT_EXECUTION_FORMAT = f"{FORMAT}-resident-execution-v1"
SIDECAR_FORMAT = f"{FORMAT}-namespace-sidecar-v1"
COMPACT_PLAN_FORMAT = f"{FORMAT}-compact-answer-plan-v1"

CONSTRUCTION_NAME = "semantic-global-terminal-full100-construction-v1.json"
REPLAY_NAME = "semantic-global-terminal-full100-construction-replay-v1.json"
SIDECAR_DIR_NAME = "semantic-global-terminal-full100-namespace-sidecars-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-semantic-global-terminal-full100-v1"
)
DEFAULT_OUTPUT_ROOT_BY_MODE = {
    terminal_cli.TERMINAL_COMPILATION_MODE_V2: DEFAULT_OUTPUT_ROOT,
    terminal_cli.TERMINAL_COMPILATION_MODE_V3: REPOSITORY_ROOT
    / "eval_results/matched_eval_100/locked-semantic-global-terminal-full100-v3-r1",
    terminal_cli.TERMINAL_COMPILATION_MODE_V4: REPOSITORY_ROOT
    / "eval_results/matched_eval_100/locked-semantic-global-terminal-full100-v4-r1",
    terminal_cli.TERMINAL_COMPILATION_MODE_V5: REPOSITORY_ROOT
    / "eval_results/matched_eval_100/locked-semantic-global-terminal-full100-v5-r1",
}

QUESTION_COUNT = 100
ELIGIBLE_COUNT = 68
PASSTHROUGH_COUNT = QUESTION_COUNT - ELIGIBLE_COUNT
TERMINAL_MODE = "terminal_plan"
PASSTHROUGH_MODE = "v3_passthrough"
POPULATION_DERIVATION = "sealed_gate_eligibility_true_and_r7_residual_synthesis_v1"


class LockedSemanticGlobalTerminalFull100Error(MatchedEvalContractError):
    """A source, gate-derived population, terminal row, or replay changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedSemanticGlobalTerminalFull100Error(message)


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact array")
    return value  # type: ignore[return-value]


def _exact_int(value: object, label: str) -> int:
    _require(type(value) is int, f"{label} must be an exact integer")
    return value  # type: ignore[return-value]


def _with_receipt(
    body: Mapping[str, Any], key: str = "receipt_sha256"
) -> dict[str, Any]:
    return {**dict(body), key: identity_sha256(body)}


def _validate_receipt(value: object, *, key: str, label: str) -> dict[str, Any]:
    row = _exact_dict(value, label)
    declared = require_sha256(row.get(key), label)
    body = {name: child for name, child in row.items() if name != key}
    _require(declared == identity_sha256(body), f"{label} receipt changed")
    return row


@dataclass(frozen=True, slots=True)
class _SourceArtifacts:
    gate: SealedArtifact
    r7: SealedArtifact
    vectors: SealedArtifact
    vector_replay: SealedArtifact
    parent: SealedArtifact
    gate_rows: tuple[dict[str, Any], ...]
    r7_rows: tuple[dict[str, Any], ...]
    parent_rows: tuple[dict[str, Any], ...]
    residual_policy: residual.SemanticResidualPolicy


@dataclass(frozen=True, slots=True)
class Full100ConstructionBundle:
    """Compact manifest plus full resident audits stored once per namespace."""

    manifest: dict[str, Any]
    sidecars: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class VerifiedFull100Construction:
    """Strict authenticated view, including full exact11 sidecar plans."""

    construction: SealedArtifact
    replay: SealedArtifact
    provider_plans: tuple[dict[str, Any], ...]
    passthroughs: tuple[dict[str, Any], ...]
    exact11_terminal_plans: tuple[dict[str, Any], ...]
    residual_policy: residual.SemanticResidualPolicy

    def legacy_tuple(self) -> tuple[
        SealedArtifact,
        SealedArtifact,
        tuple[dict[str, Any], ...],
        tuple[dict[str, Any], ...],
    ]:
        return (
            self.construction,
            self.replay,
            self.provider_plans,
            self.passthroughs,
        )


def _artifact_identity(payload: Mapping[str, Any], key: str, label: str) -> None:
    declared = require_sha256(payload.get(key), label)
    body = {name: value for name, value in payload.items() if name != key}
    _require(declared == identity_sha256(body), f"{label} identity changed")


def _validate_source_artifacts(
    gate: SealedArtifact,
    r7: SealedArtifact,
    vectors: SealedArtifact,
    vector_replay: SealedArtifact,
    parent: SealedArtifact,
) -> _SourceArtifacts:
    gate_rows = tuple(
        _exact_dict(row, "gate question")
        for row in _exact_list(gate.payload.get("questions"), "gate questions")
    )
    r7_rows = tuple(
        _exact_dict(row, "R7 question")
        for row in _exact_list(r7.payload.get("questions"), "R7 questions")
    )
    parent_rows = tuple(
        _exact_dict(row, "V3 parent question")
        for row in _exact_list(parent.payload.get("questions"), "V3 parent questions")
    )
    gate_bindings = _exact_dict(gate.payload.get("bindings"), "gate bindings")
    r7_bindings = _exact_dict(r7.payload.get("bindings"), "R7 bindings")
    vector_rows = _exact_list(vectors.payload.get("rows"), "query-vector rows")
    try:
        residual_policy = residual.semantic_residual_policy_from_projection(
            r7.payload.get("residual_search_policy")
        )
    except residual.SemanticResidualSearchError as exc:
        raise LockedSemanticGlobalTerminalFull100Error(
            "R7 residual search policy authentication failed"
        ) from exc
    _require(
        gate.payload.get("format") == r7_cli.GATE_FORMAT
        and r7.payload.get("format") == r7_cli.CONSTRUCTION_FORMAT
        and vectors.payload.get("format") == r7_cli.VECTOR_FORMAT
        and vector_replay.payload.get("format") == r7_cli.VECTOR_FORMAT
        and parent.payload.get("format") == v3_cli.FORMAT
        and gate.payload.get("question_count") == QUESTION_COUNT
        and r7.payload.get("question_count") == QUESTION_COUNT
        and parent.payload.get("question_count") == QUESTION_COUNT
        and len(gate_rows) == len(r7_rows) == len(parent_rows) == QUESTION_COUNT
        and vectors.sha256 == vector_replay.sha256
        and vectors.payload == vector_replay.payload
        and vectors.payload.get("question_count") == ELIGIBLE_COUNT
        and len(vector_rows) == ELIGIBLE_COUNT
        and gate_bindings.get("answer_artifact_sha256") == parent.sha256
        and r7_bindings.get("gate_artifact_sha256") == gate.sha256
        and r7_bindings.get("query_vector_artifact_sha256") == vectors.sha256
        and r7_bindings.get("query_vector_replay_artifact_sha256")
        == vector_replay.sha256,
        "full100 source artifact roots or populations changed",
    )
    _artifact_identity(gate.payload, "gate_identity_sha256", "gate")
    _artifact_identity(r7.payload, "construction_identity_sha256", "R7 construction")
    _artifact_identity(vectors.payload, "vector_identity_sha256", "query vectors")
    return _SourceArtifacts(
        gate=gate,
        r7=r7,
        vectors=vectors,
        vector_replay=vector_replay,
        parent=parent,
        gate_rows=gate_rows,
        r7_rows=r7_rows,
        parent_rows=parent_rows,
        residual_policy=residual_policy,
    )


def _load_build_sources(args: argparse.Namespace) -> _SourceArtifacts:
    gate, gate_sources = r7_cli._load_verified_gate(args)  # noqa: SLF001
    r7 = v6_cli._verified_r7_construction(args)  # noqa: SLF001
    vectors, _values = r7_cli._load_vectors(  # noqa: SLF001
        Path(args.vectors), str(args.expected_vector_sha256), gate
    )
    vector_replay, _replayed = r7_cli._load_vectors(  # noqa: SLF001
        Path(args.vector_replay), str(args.expected_vector_sha256), gate
    )
    parent = gate_sources[2]
    _require(
        type(parent) is SealedArtifact
        and parent.sha256 == require_sha256(
            str(args.expected_answer_sha256), "V3 parent answer"
        ),
        "verified gate escaped its V3 parent answer",
    )
    return _validate_source_artifacts(gate, r7, vectors, vector_replay, parent)


def _read_expected(path: str | Path, expected_sha256: str, label: str) -> SealedArtifact:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == require_sha256(expected_sha256, label),
        f"{label} artifact changed",
    )
    return artifact


def _derived_eligible_ordinals(sources: _SourceArtifacts) -> tuple[int, ...]:
    eligible: list[int] = []
    vector_ordinals = {
        _exact_int(_exact_dict(row, "query-vector row").get("ordinal"), "vector ordinal")
        for row in _exact_list(sources.vectors.payload.get("rows"), "query-vector rows")
    }
    for ordinal, (gate_row, r7_row, parent_row) in enumerate(
        zip(sources.gate_rows, sources.r7_rows, sources.parent_rows, strict=True)
    ):
        gate_receipt = _validate_receipt(
            gate_row, key="gate_row_receipt_sha256", label=f"gate row {ordinal}"
        )
        eligibility = _validate_receipt(
            gate_receipt.get("eligibility"),
            key="receipt_sha256",
            label=f"eligibility row {ordinal}",
        )
        r7_receipt = _validate_receipt(
            r7_row,
            key="question_receipt_sha256",
            label=f"R7 question {ordinal}",
        )
        prediction = require_text(
            parent_row.get("prediction"), f"V3 prediction {ordinal}"
        )
        is_eligible = eligibility.get("eligible")
        _require(
            type(is_eligible) is bool
            and gate_row.get("ordinal") == r7_receipt.get("ordinal") == ordinal
            and parent_row.get("ordinal") == ordinal
            and gate_row.get("question_id")
            == r7_receipt.get("question_id")
            == parent_row.get("question_id")
            and gate_row.get("question_sha256")
            == r7_receipt.get("question_sha256")
            == parent_row.get("question_sha256")
            and gate_row.get("dated_question_sha256")
            == r7_receipt.get("dated_question_sha256")
            == parent_row.get("dated_question_sha256")
            and gate_row.get("current_prediction") == prediction
            and gate_row.get("current_prediction_sha256")
            == parent_row.get("prediction_sha256")
            == quote_sha256(prediction)
            and gate_row.get("source_answer_row_sha256")
            == identity_sha256(parent_row)
            and r7_receipt.get("mode")
            == ("residual_synthesis" if is_eligible else "not_eligible")
            and ((ordinal in vector_ordinals) is is_eligible),
            f"gate/R7/V3 derivation changed at ordinal {ordinal}",
        )
        if is_eligible:
            eligible.append(ordinal)
    derived = tuple(eligible)
    _require(
        len(derived) == ELIGIBLE_COUNT
        and sources.gate.payload.get("eligible_count") == ELIGIBLE_COUNT
        and sources.gate.payload.get("eligible_ordinals") == list(derived)
        and vector_ordinals == set(derived),
        "sealed gate-derived eligible population changed",
    )
    return derived


def _source_bindings(
    sources: _SourceArtifacts,
    sealed_sources: TerminalSealedSources,
) -> dict[str, Any]:
    body = {
        "format": SOURCE_BINDINGS_FORMAT,
        "gate_artifact_sha256": sources.gate.sha256,
        "gate_identity_sha256": sources.gate.payload["gate_identity_sha256"],
        "gate_source_bindings": sources.gate.payload["bindings"],
        "parent_answer_artifact_sha256": sources.parent.sha256,
        "query_vector_artifact_sha256": sources.vectors.sha256,
        "query_vector_replay_artifact_sha256": sources.vector_replay.sha256,
        "r7_construction_artifact_sha256": sources.r7.sha256,
        "r7_construction_identity_sha256": sources.r7.payload[
            "construction_identity_sha256"
        ],
        "r7_source_bindings": sources.r7.payload["bindings"],
        "terminal_sealed_sources": sealed_sources.projection(),
    }
    return _with_receipt(body)


def _policy_bindings(
    sources: _SourceArtifacts,
    terminalized: Mapping[str, Any],
    terminal_policy: SemanticGlobalTerminalPolicy,
    terminal_mode: str = terminal_cli.TERMINAL_COMPILATION_MODE_V2,
) -> dict[str, Any]:
    _require(
        terminalized.get("local_policy")
        == SourceGroupReinjectionPolicy().projection()
        and terminalized.get("global_policy")
        == SemanticGlobalCompletionPolicy().projection(),
        "resident execution escaped the frozen V6/V7 policy",
    )
    body: dict[str, Any] = {
        "eligibility_policy": sources.gate.payload["eligibility_policy"],
        "format": POLICY_BINDINGS_FORMAT,
        "global_policy": terminalized["global_policy"],
        "local_policy": terminalized["local_policy"],
        "residual_search_policy": sources.r7.payload["residual_search_policy"],
        "terminal_policy": terminal_policy.projection(),
    }
    if terminal_mode != terminal_cli.TERMINAL_COMPILATION_MODE_V2:
        body["terminal_compilation_format"] = (
            terminal_cli.TERMINAL_COMPILATION_FORMAT_BY_MODE[terminal_mode]
        )
    return _with_receipt(body)


def _validate_terminal_answer_plan(
    raw: Mapping[str, Any], question: Mapping[str, Any]
) -> dict[str, Any]:
    return terminal_cli._validate_answer_plan(raw, question)  # noqa: SLF001


def _validate_resident_question(
    question: Mapping[str, Any], gate_row: Mapping[str, Any]
) -> dict[str, Any]:
    row = _validate_receipt(
        question,
        key="question_assay_receipt_sha256",
        label="resident terminal question",
    )
    _require(
        row.get("ordinal") == gate_row.get("ordinal")
        and row.get("question_id") == gate_row.get("question_id")
        and row.get("question_sha256") == gate_row.get("question_sha256")
        and row.get("dated_question_sha256")
        == gate_row.get("dated_question_sha256")
        and row.get("namespace_id") == gate_row.get("namespace_id")
        and row.get("new_provider_calls") == 0
        and row.get("retained_transformer_token_state_bytes") == 0,
        "resident terminal question escaped its gate identity",
    )
    plan = _validate_terminal_answer_plan(
        _exact_dict(row.get("terminal_answer_plan"), "terminal answer plan"), row
    )
    _require(
        plan.get("parent_prediction") == gate_row.get("current_prediction")
        and plan.get("parent_prediction_sha256")
        == gate_row.get("current_prediction_sha256"),
        "terminal answer plan escaped the exact V3 parent",
    )
    return row


def _require_r7_question_reexecution(
    question: Mapping[str, Any],
    r7_row: Mapping[str, Any],
) -> None:
    """Bind one resident V7 row to the exact authenticated R7 question row."""

    ordinal = _exact_int(question.get("ordinal"), "resident terminal ordinal")
    expected = require_sha256(
        r7_row.get("question_receipt_sha256"),
        f"R7 question {ordinal}",
    )
    _require(
        r7_row.get("ordinal") == ordinal
        and question.get("r7_exact_question_rebuilt") is True
        and question.get("r7_question_receipt_sha256") == expected,
        f"resident terminal question {ordinal} lost exact R7 reexecution binding",
    )


def _compact_answer_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Keep provider-executable fields while moving the full audit to a sidecar."""

    full = _exact_dict(plan, "terminal answer plan")
    _exact_dict(full.get("terminal_compilation"), "terminal compilation")
    provider_plan = {
        key: value for key, value in full.items() if key != "terminal_compilation"
    }
    body = {
        "format": COMPACT_PLAN_FORMAT,
        "full_answer_plan_receipt_sha256": require_sha256(
            full.get("answer_plan_receipt_sha256"), "terminal answer plan"
        ),
        "provider_plan": provider_plan,
        "provider_plan_sha256": identity_sha256(provider_plan),
        "terminal_compilation_receipt_sha256": require_sha256(
            full.get("terminal_compilation_receipt_sha256"),
            "terminal compilation",
        ),
    }
    return _with_receipt(body, "compact_plan_receipt_sha256")


def _question_row(
    *,
    ordinal: int,
    gate_row: Mapping[str, Any],
    parent_row: Mapping[str, Any],
    terminal_question: Mapping[str, Any] | None,
    terminal_sidecar_sha256: str | None,
) -> dict[str, Any]:
    eligible = bool(_exact_dict(gate_row.get("eligibility"), "eligibility")["eligible"])
    prediction = require_text(parent_row.get("prediction"), "parent prediction")
    _require(
        eligible
        == (terminal_question is not None)
        == (terminal_sidecar_sha256 is not None),
        "terminal work row differs from gate-derived eligibility",
    )
    compact_plan: dict[str, Any] | None = None
    question_receipt: str | None = None
    if terminal_question is not None:
        validated = _validate_resident_question(terminal_question, gate_row)
        compact_plan = _compact_answer_plan(
            _exact_dict(validated.get("terminal_answer_plan"), "terminal answer plan")
        )
        question_receipt = require_sha256(
            validated.get("question_assay_receipt_sha256"),
            "resident terminal question",
        )
        require_sha256(terminal_sidecar_sha256, "terminal namespace sidecar")
    body = {
        "dated_question_sha256": gate_row["dated_question_sha256"],
        "eligibility_receipt_sha256": gate_row["eligibility"]["receipt_sha256"],
        "format": ROW_FORMAT,
        "gate_row_receipt_sha256": gate_row["gate_row_receipt_sha256"],
        "mode": TERMINAL_MODE if eligible else PASSTHROUGH_MODE,
        "namespace_id": gate_row["namespace_id"],
        "new_provider_calls": 0,
        "ordinal": ordinal,
        "parent_answer_row_sha256": identity_sha256(parent_row),
        "parent_prediction": prediction,
        "parent_prediction_sha256": quote_sha256(prediction),
        "passthrough_prediction": None if eligible else prediction,
        "question_id": gate_row["question_id"],
        "question_sha256": gate_row["question_sha256"],
        "retained_transformer_token_state_bytes": 0,
        "terminal_answer_plan": compact_plan,
        "terminal_question_receipt_sha256": question_receipt,
        "terminal_sidecar_sha256": terminal_sidecar_sha256,
    }
    return _with_receipt(body, "question_construction_receipt_sha256")


def _resident_namespace_receipts(
    terminalized: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for raw in _exact_list(
        terminalized.get("namespace_receipts"), "resident namespace receipts"
    ):
        row = _validate_receipt(
            raw,
            key="namespace_assay_receipt_sha256",
            label="resident namespace receipt",
        )
        namespace_id = require_text(row.get("namespace_id"), "resident namespace")
        _require(namespace_id not in output, "resident namespace repeated")
        output[namespace_id] = row
    return output


def _sidecar_artifact_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _namespace_sidecars(
    *,
    terminal_questions: Sequence[Mapping[str, Any]],
    resident_by_namespace: Mapping[str, Mapping[str, Any]],
    source_bindings_receipt_sha256: str,
    policy_bindings_receipt_sha256: str,
) -> tuple[tuple[dict[str, Any], ...], dict[str, str]]:
    by_namespace: dict[str, list[dict[str, Any]]] = {}
    for raw in terminal_questions:
        row = _exact_dict(raw, "resident terminal question")
        namespace_id = require_text(row.get("namespace_id"), "resident namespace")
        by_namespace.setdefault(namespace_id, []).append(row)
    _require(
        set(by_namespace) == set(resident_by_namespace),
        "resident namespace population changed before sidecar sealing",
    )
    sidecars: list[dict[str, Any]] = []
    sha_by_namespace: dict[str, str] = {}
    for namespace_id in sorted(by_namespace):
        questions = by_namespace[namespace_id]
        resident = _exact_dict(
            resident_by_namespace[namespace_id], "resident namespace receipt"
        )
        question_receipts = [
            require_sha256(
                row.get("question_assay_receipt_sha256"),
                "resident terminal question",
            )
            for row in questions
        ]
        _require(
            resident.get("question_assay_receipt_sha256s") == question_receipts,
            "resident namespace question receipts changed",
        )
        body = {
            "format": SIDECAR_FORMAT,
            "namespace_id": namespace_id,
            "new_provider_calls": 0,
            "ordinals": [
                _exact_int(row.get("ordinal"), "resident terminal ordinal")
                for row in questions
            ],
            "policy_bindings_receipt_sha256": require_sha256(
                policy_bindings_receipt_sha256, "policy bindings"
            ),
            "question_assay_receipt_sha256s": question_receipts,
            "question_count": len(questions),
            "questions": [dict(row) for row in questions],
            "resident_namespace_receipt": dict(resident),
            "retained_transformer_token_state_bytes": 0,
            "source_bindings_receipt_sha256": require_sha256(
                source_bindings_receipt_sha256, "source bindings"
            ),
        }
        sidecar = {
            **body,
            "sidecar_identity_sha256": identity_sha256(body),
        }
        digest = _sidecar_artifact_sha256(sidecar)
        sidecars.append(sidecar)
        sha_by_namespace[namespace_id] = digest
    return tuple(sidecars), sha_by_namespace


def _namespace_rows(
    questions: Sequence[Mapping[str, Any]],
    resident_by_namespace: Mapping[str, Mapping[str, Any]],
    sidecar_sha_by_namespace: Mapping[str, str],
) -> list[dict[str, Any]]:
    by_namespace: dict[str, list[Mapping[str, Any]]] = {}
    for row in questions:
        by_namespace.setdefault(str(row["namespace_id"]), []).append(row)
    output: list[dict[str, Any]] = []
    for namespace_id in sorted(by_namespace):
        rows = by_namespace[namespace_id]
        terminal_rows = [row for row in rows if row["mode"] == TERMINAL_MODE]
        passthrough_rows = [row for row in rows if row["mode"] == PASSTHROUGH_MODE]
        resident = resident_by_namespace.get(namespace_id)
        _require(
            (resident is not None)
            == (namespace_id in sidecar_sha_by_namespace)
            == bool(terminal_rows),
            "resident namespace presence differs from eligible population",
        )
        if resident is not None:
            _require(
                resident.get("question_assay_receipt_sha256s")
                == [
                    row["terminal_question_receipt_sha256"]
                    for row in terminal_rows
                ],
                "resident namespace question receipts changed",
            )
        body = {
            "eligible_ordinals": [row["ordinal"] for row in terminal_rows],
            "format": NAMESPACE_FORMAT,
            "namespace_id": namespace_id,
            "passthrough_ordinals": [row["ordinal"] for row in passthrough_rows],
            "question_construction_receipt_sha256s": [
                row["question_construction_receipt_sha256"] for row in rows
            ],
            "resident_namespace_receipt_sha256": (
                None
                if resident is None
                else resident["namespace_assay_receipt_sha256"]
            ),
            "terminal_sidecar_sha256": sidecar_sha_by_namespace.get(namespace_id),
        }
        output.append(_with_receipt(body, "namespace_receipt_sha256"))
    _require(
        set(resident_by_namespace)
        == set(sidecar_sha_by_namespace)
        <= set(by_namespace),
        "resident namespace escaped the full100 population",
    )
    return output


def _population_receipt(
    sources: _SourceArtifacts, questions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    eligible = [row for row in questions if row["mode"] == TERMINAL_MODE]
    passthrough = [row for row in questions if row["mode"] == PASSTHROUGH_MODE]
    body = {
        "derivation": POPULATION_DERIVATION,
        "eligible_count": len(eligible),
        "eligible_ordinals": [row["ordinal"] for row in eligible],
        "format": POPULATION_FORMAT,
        "gate_artifact_sha256": sources.gate.sha256,
        "gate_row_receipt_sha256s": [
            row["gate_row_receipt_sha256"] for row in questions
        ],
        "ordinal_list_used_as_policy": False,
        "passthrough_count": len(passthrough),
        "question_count": len(questions),
    }
    return _with_receipt(body)


def _resident_execution_receipt(
    terminalized: Mapping[str, Any],
    terminal_questions: Sequence[Mapping[str, Any]],
    resident_namespaces: Mapping[str, Mapping[str, Any]],
    sidecar_sha_by_namespace: Mapping[str, str],
) -> dict[str, Any]:
    r7_bindings = _exact_dict(terminalized.get("r7_bindings"), "resident R7 bindings")
    body = {
        "eligible_question_assay_receipt_sha256s": [
            row["question_assay_receipt_sha256"] for row in terminal_questions
        ],
        "format": RESIDENT_EXECUTION_FORMAT,
        "global_policy": terminalized["global_policy"],
        "local_policy": terminalized["local_policy"],
        "namespace_assay_receipt_sha256s": [
            resident_namespaces[key]["namespace_assay_receipt_sha256"]
            for key in sorted(resident_namespaces)
        ],
        "new_provider_calls": terminalized["new_provider_calls"],
        "production_ordinal_routing_enabled": terminalized[
            "production_ordinal_routing_enabled"
        ],
        "question_count": len(terminal_questions),
        "r7_bindings": r7_bindings,
        "resident_construction_identity_sha256": require_sha256(
            terminalized.get("construction_identity_sha256"),
            "resident construction",
        ),
        "retained_transformer_token_state_bytes": terminalized[
            "retained_transformer_token_state_bytes"
        ],
        "source_indexes_rebuilt_not_serialized": terminalized[
            "source_indexes_rebuilt_not_serialized"
        ],
        "v6_v7_single_resident_index_pass": terminalized[
            "v6_v7_single_resident_index_pass"
        ],
        "v7_replay_count": terminalized["v7_replay_count"],
        "namespace_sidecar_sha256s": [
            sidecar_sha_by_namespace[key]
            for key in sorted(sidecar_sha_by_namespace)
        ],
    }
    return _with_receipt(body)


def _compose_payload(
    *,
    sources: _SourceArtifacts,
    terminalized: Mapping[str, Any],
    terminal_policy: SemanticGlobalTerminalPolicy,
    terminal_mode: str = terminal_cli.TERMINAL_COMPILATION_MODE_V2,
) -> Full100ConstructionBundle:
    terminalized_body = {
        key: value
        for key, value in terminalized.items()
        if key != "construction_identity_sha256"
    }
    eligible_ordinals = _derived_eligible_ordinals(sources)
    terminal_questions = tuple(
        _exact_dict(row, "resident terminal question")
        for row in _exact_list(
            terminalized.get("questions"), "resident terminal questions"
        )
    )
    for question in terminal_questions:
        ordinal = _exact_int(
            question.get("ordinal"), "resident terminal ordinal"
        )
        _require(
            0 <= ordinal < len(sources.r7_rows),
            "resident terminal ordinal escaped the R7 population",
        )
        _require_r7_question_reexecution(question, sources.r7_rows[ordinal])
    expected_compilation_format = (
        terminal_cli.TERMINAL_COMPILATION_FORMAT_BY_MODE[terminal_mode]
    )
    resident_compilation_formats = {
        require_text(
            _exact_dict(
                _exact_dict(
                    row.get("terminal_answer_plan"),
                    "resident terminal answer plan",
                ).get("terminal_compilation"),
                "resident terminal compilation",
            ).get("format"),
            "resident terminal compilation format",
        )
        for row in terminal_questions
    }
    _require(
        terminalized.get("format") == v7_cli.FORMAT
        and require_sha256(
            terminalized.get("construction_identity_sha256"),
            "resident construction",
        )
        == identity_sha256(terminalized_body)
        and terminalized.get("question_count") == ELIGIBLE_COUNT
        and terminalized.get("v7_replay_count") == ELIGIBLE_COUNT
        and terminalized.get("new_provider_calls") == 0
        and terminalized.get("retained_transformer_token_state_bytes") == 0
        and terminalized.get("production_ordinal_routing_enabled") is False
        and tuple(row.get("ordinal") for row in terminal_questions)
        == eligible_ordinals
        and terminalized.get("r7_bindings")
        == {
            "construction_artifact_sha256": sources.r7.sha256,
            "gate_artifact_sha256": sources.gate.sha256,
            "query_vector_artifact_sha256": sources.vectors.sha256,
            "query_vector_replay_artifact_sha256": sources.vector_replay.sha256,
        }
        and resident_compilation_formats == {expected_compilation_format},
        "resident terminal execution population changed",
    )
    terminal_by_ordinal = {
        _exact_int(row.get("ordinal"), "resident terminal ordinal"): row
        for row in terminal_questions
    }
    _require(
        len(terminal_by_ordinal) == ELIGIBLE_COUNT,
        "resident terminal ordinal repeated",
    )
    resident_namespaces = _resident_namespace_receipts(terminalized)
    sealed_sources = TerminalSealedSources(
        protected_owner_artifact_sha256=sources.r7.sha256,
        residual_artifact_sha256=sources.r7.sha256,
        parent_artifact_sha256=sources.gate.sha256,
    )
    source_bindings = _source_bindings(sources, sealed_sources)
    policy_bindings = _policy_bindings(
        sources, terminalized, terminal_policy, terminal_mode
    )
    sidecars, sidecar_sha_by_namespace = _namespace_sidecars(
        terminal_questions=terminal_questions,
        resident_by_namespace=resident_namespaces,
        source_bindings_receipt_sha256=source_bindings["receipt_sha256"],
        policy_bindings_receipt_sha256=policy_bindings["receipt_sha256"],
    )
    questions = [
        _question_row(
            ordinal=ordinal,
            gate_row=sources.gate_rows[ordinal],
            parent_row=sources.parent_rows[ordinal],
            terminal_question=terminal_by_ordinal.get(ordinal),
            terminal_sidecar_sha256=(
                sidecar_sha_by_namespace.get(str(sources.gate_rows[ordinal]["namespace_id"]))
                if ordinal in terminal_by_ordinal
                else None
            ),
        )
        for ordinal in range(QUESTION_COUNT)
    ]
    namespaces = _namespace_rows(
        questions, resident_namespaces, sidecar_sha_by_namespace
    )
    population = _population_receipt(sources, questions)
    resident = _resident_execution_receipt(
        terminalized,
        terminal_questions,
        resident_namespaces,
        sidecar_sha_by_namespace,
    )
    body = {
        "eligible_count": ELIGIBLE_COUNT,
        "format": FORMAT,
        "gate_derived_population": population,
        "gold_loaded": False,
        "namespace_count": len(namespaces),
        "namespace_receipts": namespaces,
        "new_provider_calls": 0,
        "ordinal_cli_routing_available": False,
        "passthrough_count": PASSTHROUGH_COUNT,
        "policy_bindings": policy_bindings,
        "production_ordinal_routing_enabled": False,
        "question_count": QUESTION_COUNT,
        "questions": questions,
        "resident_execution": resident,
        "retained_transformer_token_state_bytes": 0,
        "source_artifact_bindings": source_bindings,
        "terminal_namespace_sidecar_count": len(sidecars),
        "terminal_namespace_sidecar_sha256s": [
            sidecar_sha_by_namespace[key]
            for key in sorted(sidecar_sha_by_namespace)
        ],
        "terminal_answer_plan_count": ELIGIBLE_COUNT,
    }
    if terminal_mode != terminal_cli.TERMINAL_COMPILATION_MODE_V2:
        body["terminal_compilation_format"] = (
            terminal_cli.TERMINAL_COMPILATION_FORMAT_BY_MODE[terminal_mode]
        )
    assert_gold_blind(body, path="semantic_global_terminal_full100_construction")
    manifest = {**body, "construction_identity_sha256": identity_sha256(body)}
    _require(
        "local_audit" not in json.dumps(manifest, ensure_ascii=False),
        "resident local audit escaped into compact full100 manifest",
    )
    return Full100ConstructionBundle(manifest=manifest, sidecars=sidecars)


def build_construction_bundle(args: argparse.Namespace) -> Full100ConstructionBundle:
    sources = _load_build_sources(args)
    eligible_ordinals = _derived_eligible_ordinals(sources)
    sealed_sources = TerminalSealedSources(
        protected_owner_artifact_sha256=sources.r7.sha256,
        residual_artifact_sha256=sources.r7.sha256,
        parent_artifact_sha256=sources.gate.sha256,
    )
    terminal_policy = SemanticGlobalTerminalPolicy()
    terminal_mode = terminal_cli.terminal_compilation_mode(args)
    resident_args = argparse.Namespace(**vars(args))
    # This is the sole ordinal handoff.  It is derived from the authenticated
    # gate above and is not caller- or CLI-controlled.
    resident_args.ordinals = eligible_ordinals
    terminalized = v7_cli.build_assay(
        resident_args,
        terminal_compiler=partial(
            terminal_cli._compile_answer_plan_core,  # noqa: SLF001
            sealed_sources=sealed_sources,
            policy=terminal_policy,
            terminal_mode=terminal_mode,
        ),
    )
    return _compose_payload(
        sources=sources,
        terminalized=terminalized,
        terminal_policy=terminal_policy,
        terminal_mode=terminal_mode,
    )


def build_construction(args: argparse.Namespace) -> dict[str, Any]:
    """Build only the compact manifest; publication uses the bundle API."""

    return build_construction_bundle(args).manifest


def _validate_bound_projection(
    payload: Mapping[str, Any],
    sources: _SourceArtifacts,
    sidecar_root: str | Path,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    """Authenticate the compact manifest against exact full namespace sidecars."""

    body = {
        name: value
        for name, value in payload.items()
        if name != "construction_identity_sha256"
    }
    _require(
        require_sha256(
            payload.get("construction_identity_sha256"), "full100 construction"
        )
        == identity_sha256(body),
        "full100 construction identity changed",
    )
    expected_sealed_sources = TerminalSealedSources(
        protected_owner_artifact_sha256=sources.r7.sha256,
        residual_artifact_sha256=sources.r7.sha256,
        parent_artifact_sha256=sources.gate.sha256,
    )
    declared_compilation_format = payload.get("terminal_compilation_format")
    if declared_compilation_format is None:
        terminal_mode = terminal_cli.TERMINAL_COMPILATION_MODE_V2
        expected_compilation_format = terminal_cli.TERMINAL_COMPILATION_FORMAT
    else:
        expected_compilation_format = require_text(
            declared_compilation_format, "full100 terminal compilation format"
        )
        terminal_mode = terminal_cli.terminal_compilation_mode_for_format(
            expected_compilation_format
        )
        _require(
            terminal_mode != terminal_cli.TERMINAL_COMPILATION_MODE_V2,
            "full100 frozen v2 must retain its historical projection",
        )
    expected_source_bindings = _source_bindings(sources, expected_sealed_sources)
    expected_policy_body: dict[str, Any] = {
        "eligibility_policy": sources.gate.payload["eligibility_policy"],
        "format": POLICY_BINDINGS_FORMAT,
        "global_policy": SemanticGlobalCompletionPolicy().projection(),
        "local_policy": SourceGroupReinjectionPolicy().projection(),
        "residual_search_policy": sources.r7.payload["residual_search_policy"],
        "terminal_policy": SemanticGlobalTerminalPolicy().projection(),
    }
    if terminal_mode != terminal_cli.TERMINAL_COMPILATION_MODE_V2:
        expected_policy_body["terminal_compilation_format"] = (
            expected_compilation_format
        )
    expected_policy_bindings = _with_receipt(expected_policy_body)
    _require(
        payload.get("source_artifact_bindings") == expected_source_bindings
        and payload.get("policy_bindings") == expected_policy_bindings,
        "full100 construction roots, policies, or provenance changed",
    )
    derived = _derived_eligible_ordinals(sources)
    derived_set = set(derived)
    raw_namespace_rows = _exact_list(
        payload.get("namespace_receipts"), "full100 namespace receipts"
    )
    resident_by_namespace: dict[str, dict[str, Any]] = {}
    full_question_by_ordinal: dict[int, dict[str, Any]] = {}
    sidecar_sha_by_namespace: dict[str, str] = {}
    sidecar_payload_by_namespace: dict[str, dict[str, Any]] = {}
    root = Path(sidecar_root)
    for raw_namespace in raw_namespace_rows:
        namespace = _validate_receipt(
            raw_namespace,
            key="namespace_receipt_sha256",
            label="full100 namespace receipt",
        )
        namespace_id = require_text(
            namespace.get("namespace_id"), "full100 namespace"
        )
        _require(
            namespace_id not in sidecar_payload_by_namespace,
            "full100 namespace repeated",
        )
        raw_sidecar_sha = namespace.get("terminal_sidecar_sha256")
        if raw_sidecar_sha is None:
            continue
        sidecar_sha = require_sha256(raw_sidecar_sha, "terminal namespace sidecar")
        try:
            sidecar = _read_expected(
                root / SIDECAR_DIR_NAME / f"{sidecar_sha}.json",
                sidecar_sha,
                f"terminal namespace sidecar {namespace_id}",
            )
        except MatchedEvalContractError as exc:
            raise LockedSemanticGlobalTerminalFull100Error(
                f"terminal namespace sidecar is unavailable or unauthenticated: {namespace_id}"
            ) from exc
        sidecar_payload = sidecar.payload
        sidecar_body = {
            key: value
            for key, value in sidecar_payload.items()
            if key != "sidecar_identity_sha256"
        }
        _require(
            set(sidecar_payload)
            == {
                "format",
                "namespace_id",
                "new_provider_calls",
                "ordinals",
                "policy_bindings_receipt_sha256",
                "question_assay_receipt_sha256s",
                "question_count",
                "questions",
                "resident_namespace_receipt",
                "retained_transformer_token_state_bytes",
                "sidecar_identity_sha256",
                "source_bindings_receipt_sha256",
            }
            and require_sha256(
                sidecar_payload.get("sidecar_identity_sha256"),
                "terminal namespace sidecar identity",
            )
            == identity_sha256(sidecar_body)
            and sidecar_payload.get("format") == SIDECAR_FORMAT
            and sidecar_payload.get("namespace_id") == namespace_id
            and sidecar_payload.get("source_bindings_receipt_sha256")
            == expected_source_bindings["receipt_sha256"]
            and sidecar_payload.get("policy_bindings_receipt_sha256")
            == expected_policy_bindings["receipt_sha256"]
            and sidecar_payload.get("new_provider_calls") == 0
            and sidecar_payload.get("retained_transformer_token_state_bytes") == 0,
            "terminal namespace sidecar identity/provenance changed",
        )
        resident = _validate_receipt(
            sidecar_payload.get("resident_namespace_receipt"),
            key="namespace_assay_receipt_sha256",
            label="resident namespace receipt",
        )
        raw_full_questions = _exact_list(
            sidecar_payload.get("questions"), "terminal sidecar questions"
        )
        ordinals: list[int] = []
        receipts: list[str] = []
        for raw_question in raw_full_questions:
            raw_row = _exact_dict(raw_question, "resident terminal question")
            ordinal = _exact_int(
                raw_row.get("ordinal"), "resident terminal ordinal"
            )
            _require(
                ordinal in derived_set
                and ordinal not in full_question_by_ordinal,
                "terminal sidecar population repeated or escaped the gate",
            )
            validated = _validate_resident_question(
                raw_row, sources.gate_rows[ordinal]
            )
            _require_r7_question_reexecution(
                validated, sources.r7_rows[ordinal]
            )
            _require(
                validated.get("namespace_id") == namespace_id,
                "terminal sidecar question escaped its namespace",
            )
            plan = _exact_dict(
                validated.get("terminal_answer_plan"), "terminal answer plan"
            )
            compilation = _exact_dict(
                plan.get("terminal_compilation"), "terminal compilation"
            )
            _require(
                plan.get("source_artifact_bindings")
                == expected_sealed_sources.projection()
                and compilation.get("policy")
                == expected_policy_bindings["terminal_policy"]
                and compilation.get("format") == expected_compilation_format,
                "terminal sidecar plan escaped frozen source/policy bindings",
            )
            full_question_by_ordinal[ordinal] = validated
            ordinals.append(ordinal)
            receipts.append(validated["question_assay_receipt_sha256"])
        expected_sidecar_body = {
            "format": SIDECAR_FORMAT,
            "namespace_id": namespace_id,
            "new_provider_calls": 0,
            "ordinals": ordinals,
            "policy_bindings_receipt_sha256": expected_policy_bindings[
                "receipt_sha256"
            ],
            "question_assay_receipt_sha256s": receipts,
            "question_count": len(raw_full_questions),
            "questions": raw_full_questions,
            "resident_namespace_receipt": resident,
            "retained_transformer_token_state_bytes": 0,
            "source_bindings_receipt_sha256": expected_source_bindings[
                "receipt_sha256"
            ],
        }
        _require(
            sidecar_payload
            == {
                **expected_sidecar_body,
                "sidecar_identity_sha256": identity_sha256(expected_sidecar_body),
            }
            and resident.get("namespace_id") == namespace_id
            and resident.get("question_assay_receipt_sha256s") == receipts
            and namespace.get("resident_namespace_receipt_sha256")
            == resident.get("namespace_assay_receipt_sha256"),
            "terminal namespace sidecar contents changed",
        )
        assert_gold_blind(
            sidecar_payload,
            path=f"verified_semantic_global_terminal_full100_sidecar.{namespace_id}",
        )
        resident_by_namespace[namespace_id] = resident
        sidecar_sha_by_namespace[namespace_id] = sidecar_sha
        sidecar_payload_by_namespace[namespace_id] = sidecar_payload

    _require(
        len(full_question_by_ordinal) == ELIGIBLE_COUNT
        and set(full_question_by_ordinal) == derived_set,
        "terminal sidecars do not exactly cover the gate-derived population",
    )
    raw_questions = _exact_list(payload.get("questions"), "full100 questions")
    _require(len(raw_questions) == QUESTION_COUNT, "full100 question population changed")
    questions: list[dict[str, Any]] = []
    provider_plans: list[dict[str, Any]] = []
    passthroughs: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(raw_questions):
        row = _validate_receipt(
            raw,
            key="question_construction_receipt_sha256",
            label=f"full100 question {ordinal}",
        )
        full_question = full_question_by_ordinal.get(ordinal)
        expected = _question_row(
            ordinal=ordinal,
            gate_row=sources.gate_rows[ordinal],
            parent_row=sources.parent_rows[ordinal],
            terminal_question=full_question,
            terminal_sidecar_sha256=(
                sidecar_sha_by_namespace.get(
                    str(sources.gate_rows[ordinal]["namespace_id"])
                )
                if full_question is not None
                else None
            ),
        )
        _require(
            row.get("ordinal") == ordinal,
            f"full100 question identity/mode changed at ordinal {ordinal}",
        )
        if full_question is None:
            _require(
                row.get("passthrough_prediction")
                == sources.parent_rows[ordinal].get("prediction"),
                "V3 passthrough prediction changed",
            )
        _require(row == expected, f"full100 question identity/mode changed at ordinal {ordinal}")
        if full_question is None:
            passthroughs.append(row)
        else:
            compact = _exact_dict(
                row.get("terminal_answer_plan"), "compact terminal answer plan"
            )
            provider_plans.append(
                _exact_dict(compact.get("provider_plan"), "terminal provider plan")
            )
        questions.append(row)

    expected_namespaces = _namespace_rows(
        questions, resident_by_namespace, sidecar_sha_by_namespace
    )
    resident_questions = [full_question_by_ordinal[ordinal] for ordinal in derived]
    resident_namespace_rows = [
        resident_by_namespace[key] for key in sorted(resident_by_namespace)
    ]
    expected_r7_bindings = {
        "construction_artifact_sha256": sources.r7.sha256,
        "gate_artifact_sha256": sources.gate.sha256,
        "query_vector_artifact_sha256": sources.vectors.sha256,
        "query_vector_replay_artifact_sha256": sources.vector_replay.sha256,
    }
    resident_body = {
        "diagnostic_population_explicitly_supplied": True,
        "format": v7_cli.FORMAT,
        "global_policy": expected_policy_bindings["global_policy"],
        "gold_loaded": False,
        "local_policy": expected_policy_bindings["local_policy"],
        "namespace_receipts": resident_namespace_rows,
        "new_provider_calls": 0,
        "production_ordinal_routing_enabled": False,
        "question_count": ELIGIBLE_COUNT,
        "questions": resident_questions,
        "r7_bindings": expected_r7_bindings,
        "retained_transformer_token_state_bytes": 0,
        "source_indexes_rebuilt_not_serialized": True,
        "v6_v7_single_resident_index_pass": True,
        "v7_replay_count": ELIGIBLE_COUNT,
    }
    reconstructed_resident = {
        **resident_body,
        "construction_identity_sha256": identity_sha256(resident_body),
    }
    expected_resident_execution = _resident_execution_receipt(
        reconstructed_resident,
        resident_questions,
        resident_by_namespace,
        sidecar_sha_by_namespace,
    )
    expected_population = _population_receipt(sources, questions)
    expected_manifest_body = {
        "eligible_count": ELIGIBLE_COUNT,
        "format": FORMAT,
        "gate_derived_population": expected_population,
        "gold_loaded": False,
        "namespace_count": len(expected_namespaces),
        "namespace_receipts": expected_namespaces,
        "new_provider_calls": 0,
        "ordinal_cli_routing_available": False,
        "passthrough_count": PASSTHROUGH_COUNT,
        "policy_bindings": expected_policy_bindings,
        "production_ordinal_routing_enabled": False,
        "question_count": QUESTION_COUNT,
        "questions": questions,
        "resident_execution": expected_resident_execution,
        "retained_transformer_token_state_bytes": 0,
        "source_artifact_bindings": expected_source_bindings,
        "terminal_namespace_sidecar_count": len(sidecar_sha_by_namespace),
        "terminal_namespace_sidecar_sha256s": [
            sidecar_sha_by_namespace[key]
            for key in sorted(sidecar_sha_by_namespace)
        ],
        "terminal_answer_plan_count": ELIGIBLE_COUNT,
    }
    if terminal_mode != terminal_cli.TERMINAL_COMPILATION_MODE_V2:
        expected_manifest_body["terminal_compilation_format"] = (
            expected_compilation_format
        )
    _require(
        payload
        == {
            **expected_manifest_body,
            "construction_identity_sha256": identity_sha256(expected_manifest_body),
        }
        and len(provider_plans) == ELIGIBLE_COUNT
        and len(passthroughs) == PASSTHROUGH_COUNT,
        "full100 manifest, population, sidecars, or resident replay changed",
    )
    assert_gold_blind(payload, path="verified_semantic_global_terminal_full100")
    _require(
        set(terminal_cli.EXACT_ORDINALS) <= set(full_question_by_ordinal),
        "full100 exact11 terminal plan population is incomplete",
    )
    exact11_terminal_plans = tuple(
        _exact_dict(
            full_question_by_ordinal[ordinal].get("terminal_answer_plan"),
            f"full100 exact11 terminal plan {ordinal}",
        )
        for ordinal in terminal_cli.EXACT_ORDINALS
    )
    _require(
        tuple(plan.get("ordinal") for plan in exact11_terminal_plans)
        == terminal_cli.EXACT_ORDINALS,
        "full100 exact11 terminal plan projection changed",
    )
    return (
        tuple(provider_plans),
        tuple(passthroughs),
        exact11_terminal_plans,
    )


def load_verified_full100_construction_detailed(
    output_root: str | Path,
    expected_construction_sha256: str,
    expected_replay_sha256: str,
    *,
    gate_path: str | Path = r7_cli.DEFAULT_GATE,
    expected_gate_sha256: str = v6_cli.EXPECTED_R7_GATE_SHA256,
    r7_path: str | Path = v6_cli.DEFAULT_R7_CONSTRUCTION,
    expected_r7_sha256: str = v6_cli.EXPECTED_R7_CONSTRUCTION_SHA256,
    vectors_path: str | Path = r7_cli.DEFAULT_VECTORS,
    vector_replay_path: str | Path = r7_cli.DEFAULT_VECTOR_REPLAY,
    expected_vector_sha256: str = v6_cli.EXPECTED_R7_VECTOR_SHA256,
    parent_path: str | Path = r7_cli.DEFAULT_ANSWER,
    expected_parent_sha256: str = r7_cli.EXPECTED_ANSWER_SHA256,
) -> VerifiedFull100Construction:
    """Load every authenticated source and expose exact11 full sidecar plans."""

    root = Path(output_root)
    construction = _read_expected(
        root / CONSTRUCTION_NAME,
        expected_construction_sha256,
        "full100 construction",
    )
    replay = _read_expected(
        root / REPLAY_NAME, expected_replay_sha256, "full100 replay"
    )
    _require(
        construction.sha256 == replay.sha256
        and construction.payload == replay.payload,
        "full100 construction/replay are not byte-identical",
    )
    gate = _read_expected(gate_path, expected_gate_sha256, "R7 gate")
    r7 = _read_expected(r7_path, expected_r7_sha256, "R7 construction")
    vectors = _read_expected(
        vectors_path, expected_vector_sha256, "R7 query vectors"
    )
    vector_replay = _read_expected(
        vector_replay_path, expected_vector_sha256, "R7 query-vector replay"
    )
    parent = _read_expected(parent_path, expected_parent_sha256, "V3 parent")
    sources = _validate_source_artifacts(gate, r7, vectors, vector_replay, parent)
    provider_plans, passthroughs, exact11_terminal_plans = _validate_bound_projection(
        construction.payload, sources, root
    )
    return VerifiedFull100Construction(
        construction=construction,
        replay=replay,
        provider_plans=provider_plans,
        passthroughs=passthroughs,
        exact11_terminal_plans=exact11_terminal_plans,
        residual_policy=sources.residual_policy,
    )


def load_verified_full100_construction(
    output_root: str | Path,
    expected_construction_sha256: str,
    expected_replay_sha256: str,
    *,
    gate_path: str | Path = r7_cli.DEFAULT_GATE,
    expected_gate_sha256: str = v6_cli.EXPECTED_R7_GATE_SHA256,
    r7_path: str | Path = v6_cli.DEFAULT_R7_CONSTRUCTION,
    expected_r7_sha256: str = v6_cli.EXPECTED_R7_CONSTRUCTION_SHA256,
    vectors_path: str | Path = r7_cli.DEFAULT_VECTORS,
    vector_replay_path: str | Path = r7_cli.DEFAULT_VECTOR_REPLAY,
    expected_vector_sha256: str = v6_cli.EXPECTED_R7_VECTOR_SHA256,
    parent_path: str | Path = r7_cli.DEFAULT_ANSWER,
    expected_parent_sha256: str = r7_cli.EXPECTED_ANSWER_SHA256,
) -> tuple[
    SealedArtifact,
    SealedArtifact,
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    """Preserve the original public four-tuple API."""

    return load_verified_full100_construction_detailed(
        output_root,
        expected_construction_sha256,
        expected_replay_sha256,
        gate_path=gate_path,
        expected_gate_sha256=expected_gate_sha256,
        r7_path=r7_path,
        expected_r7_sha256=expected_r7_sha256,
        vectors_path=vectors_path,
        vector_replay_path=vector_replay_path,
        expected_vector_sha256=expected_vector_sha256,
        parent_path=parent_path,
        expected_parent_sha256=expected_parent_sha256,
    ).legacy_tuple()


def run_construct(args: argparse.Namespace) -> dict[str, Any]:
    bundle = build_construction_bundle(args)
    payload = bundle.manifest
    output_root = output_root_for_args(args)
    sidecar_created_count = 0
    for sidecar_payload in bundle.sidecars:
        expected_sha256 = _sidecar_artifact_sha256(sidecar_payload)
        sidecar, created = publish_sealed_json(
            output_root
            / SIDECAR_DIR_NAME
            / f"{expected_sha256}.json",
            sidecar_payload,
        )
        _require(
            sidecar.sha256 == expected_sha256,
            "terminal namespace sidecar publication changed bytes",
        )
        sidecar_created_count += int(created)
    artifact, created = publish_sealed_json(
        output_root / CONSTRUCTION_NAME, payload
    )
    return {
        "construction_sha256": artifact.sha256,
        "created": created,
        "eligible_count": payload["eligible_count"],
        "new_provider_calls": 0,
        "passthrough_count": payload["passthrough_count"],
        "question_count": payload["question_count"],
        "retained_transformer_token_state_bytes": 0,
        "sidecar_count": len(bundle.sidecars),
        "sidecar_created_count": sidecar_created_count,
    }


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    bundle = build_construction_bundle(args)
    rebuilt = bundle.manifest
    output_root = output_root_for_args(args)
    for sidecar_payload in bundle.sidecars:
        expected_sha256 = _sidecar_artifact_sha256(sidecar_payload)
        sidecar = _read_expected(
            output_root
            / SIDECAR_DIR_NAME
            / f"{expected_sha256}.json",
            expected_sha256,
            "terminal namespace sidecar replay",
        )
        _require(
            sidecar.payload == sidecar_payload,
            "terminal namespace sidecar differs from exact resident replay",
        )
    construction = read_sealed_json(output_root / CONSTRUCTION_NAME)
    _require(
        construction.sha256
        == require_sha256(
            str(args.expected_construction_output_sha256), "full100 construction"
        )
        and construction.payload == rebuilt,
        "full100 construction differs from exact resident replay",
    )
    replay, _created = publish_sealed_json(
        output_root / REPLAY_NAME, rebuilt
    )
    _require(
        replay.sha256 == construction.sha256,
        "full100 replay changed construction bytes",
    )
    return {
        "byte_identical": True,
        "construction_sha256": construction.sha256,
        "new_provider_calls": 0,
        "replay_sha256": replay.sha256,
        "retained_transformer_token_state_bytes": 0,
        "sidecar_count": len(bundle.sidecars),
    }


def _add_resident_args(parser: argparse.ArgumentParser) -> None:
    # Reuse the established source/store/policy options, but intentionally do
    # not call V6's `_add_args`: that helper exposes `--ordinals`.
    r7_cli._add_sources(parser)  # noqa: SLF001
    r7_cli._add_budget(parser)  # noqa: SLF001
    v6_cli.reduced_cli._add_store_args(parser)  # noqa: SLF001
    v6_cli.semantic_cli._add_policy_args(parser)  # noqa: SLF001
    parser.set_defaults(
        output_root=DEFAULT_OUTPUT_ROOT,
        expected_gate_sha256=v6_cli.EXPECTED_R7_GATE_SHA256,
    )
    parser.add_argument(
        "--r7-construction", type=Path, default=v6_cli.DEFAULT_R7_CONSTRUCTION
    )
    parser.add_argument(
        "--expected-r7-construction-sha256",
        default=v6_cli.EXPECTED_R7_CONSTRUCTION_SHA256,
    )
    parser.add_argument("--vectors", type=Path, default=r7_cli.DEFAULT_VECTORS)
    parser.add_argument(
        "--vector-replay", type=Path, default=r7_cli.DEFAULT_VECTOR_REPLAY
    )
    parser.add_argument(
        "--expected-vector-sha256", default=v6_cli.EXPECTED_R7_VECTOR_SHA256
    )
    parser.add_argument(
        "--protected-owner-token-cap",
        type=int,
        default=r7_cli.DEFAULT_PROTECTED_OWNER_TOKEN_CAP,
    )
    local = SourceGroupReinjectionPolicy()
    global_policy = SemanticGlobalCompletionPolicy()
    global_lanes = {row.lane_id: row for row in global_policy.lane_budgets}
    parser.set_defaults(
        local_payload_token_cap=local.local_payload_token_cap,
        max_selected_segments=local.max_selected_segments,
        base_segments_per_group=local.base_segments_per_group,
        max_query_term_obligations=local.max_query_term_obligations,
        source_neighbor_radius=local.source_neighbor_radius,
        max_source_neighbors_per_anchor=local.max_source_neighbors_per_anchor,
        max_episode_segments_per_seed=local.max_episode_segments_per_seed,
        global_payload_token_cap=global_policy.global_payload_token_cap,
        global_max_node_visits=global_policy.max_node_visits,
        global_max_retained_leaf_cells=global_policy.max_retained_leaf_cells,
        global_source_neighbor_radius=global_policy.source_neighbor_radius,
        global_max_hydrated_segments=global_policy.max_hydrated_segments,
        global_max_entity_obligations=global_policy.max_entity_obligations,
        dense_max_segments=global_lanes["dense"].max_selected_segments,
        dense_token_cap=global_lanes["dense"].pre_dedup_token_cap,
        sparse_max_segments=global_lanes["sparse"].max_selected_segments,
        sparse_token_cap=global_lanes["sparse"].pre_dedup_token_cap,
        personal_temporal_max_segments=(
            global_lanes["personal_temporal"].max_selected_segments
        ),
        personal_temporal_token_cap=(
            global_lanes["personal_temporal"].pre_dedup_token_cap
        ),
        diversity_max_segments=(
            global_lanes["source_date_diversity"].max_selected_segments
        ),
        diversity_token_cap=(
            global_lanes["source_date_diversity"].pre_dedup_token_cap
        ),
    )
    parser.add_argument("--episode-artifact-id")
    parser.add_argument(
        "--auto-resolve-episode-artifact",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-episode-anchors", type=int, default=8)
    parser.add_argument("--previous-episodes", type=int, default=1)
    parser.add_argument("--next-episodes", type=int, default=1)
    parser.add_argument("--max-episode-seeds", type=int, default=24)
    parser.add_argument("--max-episode-direct-fallbacks", type=int, default=16)
    parser.add_argument(
        "--terminal-compilation-mode",
        choices=terminal_cli.TERMINAL_COMPILATION_MODES,
        default=terminal_cli.TERMINAL_COMPILATION_MODE_V2,
    )


def output_root_for_args(args: argparse.Namespace) -> Path:
    """Route successor modes away from the frozen full100 v2 default root."""

    mode = terminal_cli.terminal_compilation_mode(args)
    configured = Path(args.output_root)
    if configured == DEFAULT_OUTPUT_ROOT:
        return DEFAULT_OUTPUT_ROOT_BY_MODE[mode]
    return configured


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    construct = commands.add_parser("construct")
    _add_resident_args(construct)
    replay = commands.add_parser("replay")
    _add_resident_args(replay)
    replay.add_argument("--expected-construction-output-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_construct(args) if args.command == "construct" else run_replay(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "COMPACT_PLAN_FORMAT",
    "CONSTRUCTION_NAME",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_OUTPUT_ROOT_BY_MODE",
    "ELIGIBLE_COUNT",
    "FORMAT",
    "Full100ConstructionBundle",
    "VerifiedFull100Construction",
    "LockedSemanticGlobalTerminalFull100Error",
    "PASSTHROUGH_COUNT",
    "PASSTHROUGH_MODE",
    "QUESTION_COUNT",
    "REPLAY_NAME",
    "SIDECAR_DIR_NAME",
    "SIDECAR_FORMAT",
    "TERMINAL_MODE",
    "build_construction",
    "build_construction_bundle",
    "build_parser",
    "load_verified_full100_construction",
    "load_verified_full100_construction_detailed",
    "main",
    "output_root_for_args",
    "run_construct",
    "run_replay",
]
