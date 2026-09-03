#!/usr/bin/env python3
"""Provider-free reconciliation over the sealed locked specialist V2 answers.

This is a new answer lifecycle, not a mutation of the V2 experiment.  Its only
answer-bearing inputs are the exact V2 preflight, terminal run, and byte-
identical replay.  It re-executes three gold-blind deterministic lanes over the
already sealed provider projections and composes them in this fixed order:

1. question-bound temporal reconciliation on temporal routes;
2. sealed numeric reconciliation on numeric routes;
3. cross-plane protection of a stronger parent on validated replacements;
4. byte-preserved V2 prediction fallback.

No completion provider is reachable from this module.  Every lane audit, row,
and top-level policy is receipt-bound; every materialized question is self-
hashed; the V2 8,000-token envelope is preserved; and transformer token state
is never retained.  ``audit`` is read-only.  ``materialize`` requires caller-
supplied frozen lane-population receipts so an unfrozen assay cannot silently
be promoted to the production V3 artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from tools import run_locked_specialist_final_answer_v2 as answer_v2  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.cross_plane_authority import (  # noqa: E402
    CrossPlaneAuthorityProtection,
    protect_parent_from_cross_plane_authority,
)
from tools.matched_eval.numeric_evidence_reconciler import (  # noqa: E402
    NumericEvidenceReconciliationReceipt,
)
from tools.matched_eval.numeric_evidence_reconciler_v2 import (  # noqa: E402
    NumericEvidenceReconciliationV2Receipt,
    reconcile_sealed_numeric_evidence_v2,
)
from tools.matched_eval.temporal_event_reconciler import (  # noqa: E402
    TemporalEventResolution,
    reconcile_temporal_events,
)
from tools.matched_eval.typed_memory_final_arm import (  # noqa: E402
    judge_row_projection,
)
from tools.matched_eval.typed_numeric_semantics import (  # noqa: E402
    NumericQualifier,
    numeric_mentions,
)


FORMAT = "memory-condense-locked-specialist-final-reconciliation-v3"
RESULT_ROW_FORMAT = f"{FORMAT}-result-row-v1"
LANE_STATUS_FORMAT = f"{FORMAT}-lane-status-row-v1"
POLICY_FORMAT = f"{FORMAT}-composition-policy-v1"
AUDIT_FORMAT = f"{FORMAT}-lane-audit-v1"

RUN_NAME = "locked-specialist-final-reconciliation-v3.json"
REPLAY_NAME = "locked-specialist-final-reconciliation-replay-v3.json"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V2_ROOT = answer_v2.DEFAULT_OUTPUT
DEFAULT_PREFLIGHT = DEFAULT_V2_ROOT / answer_v2.PREFLIGHT_NAME
DEFAULT_V2_RUN = DEFAULT_V2_ROOT / answer_v2.RUN_NAME
DEFAULT_V2_REPLAY = DEFAULT_V2_ROOT / answer_v2.REPLAY_NAME
DEFAULT_OUTPUT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-specialist-final-reconciliation-v3"
)

EXPECTED_V2_PREFLIGHT_SHA256 = (
    "61371cd58b239a07f493ea4c116908a7f72e252cb503c0a5210f30c7f66ad413"
)
EXPECTED_V2_RUN_SHA256 = (
    "8fddda61fd5834c7af55d868fe942b2522eb9a65e3aa2437ac8f1f5da7f9dac3"
)
EXPECTED_V2_REPLAY_SHA256 = EXPECTED_V2_RUN_SHA256

EXPECTED_QUESTION_COUNT = 100
EXPECTED_PHYSICAL_PROMPT_COUNT = 72
HARD_COMPLETE_CHAT_TOKEN_CAP = answer_v2.HARD_COMPLETE_CHAT_TOKEN_CAP
OUTPUT_TOKEN_RESERVE = answer_v2.OUTPUT_TOKEN_RESERVE

TEMPORAL_ROUTE_IDS = frozenset({"temporal_timeline"})
NUMERIC_ROUTE_IDS = frozenset({"numeric_reduce"})
COMPOSITION_ORDER = (
    "question_bound_temporal",
    "sealed_numeric",
    "cross_plane_parent_protection",
    "v2_fallback",
)
_DISTINCT_IDENTITY_RE = re.compile(
    r"\b(?:different|distinct|unique)\b", re.IGNORECASE
)
_RECURRING_FREQUENCY_RE = re.compile(
    r"\b(?:per\s+(?:day|week|month|year)|each\s+(?:day|week|month|year)|"
    r"weekly|monthly|typical\s+week)\b",
    re.IGNORECASE,
)


class LockedSpecialistFinalReconciliationV3Error(MatchedEvalContractError):
    """Raised when a V3 source, lane, or replay contract changes."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedSpecialistFinalReconciliationV3Error(message)


@dataclass(frozen=True, slots=True)
class _SourceBundle:
    preflight: SealedArtifact
    run: SealedArtifact
    replay: SealedArtifact
    plans: tuple[dict[str, Any], ...]
    rows: tuple[dict[str, Any], ...]
    providers_by_ordinal: dict[int, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class _LaneAudit:
    lane: str
    status_rows: tuple[dict[str, Any], ...]
    resolved_rows: tuple[dict[str, Any], ...]
    resolutions_by_ordinal: Mapping[int, Any]

    @property
    def status_population_sha256(self) -> str:
        return identity_sha256(list(self.status_rows))

    @property
    def resolved_population_sha256(self) -> str:
        return identity_sha256(list(self.resolved_rows))

    def projection(self) -> dict[str, Any]:
        body = {
            "format": AUDIT_FORMAT,
            "lane": self.lane,
            "provider_calls": 0,
            "resolved_count": len(self.resolved_rows),
            "resolved_population_sha256": self.resolved_population_sha256,
            "retained_transformer_token_state_bytes": 0,
            "status_population_sha256": self.status_population_sha256,
            "status_rows": list(self.status_rows),
        }
        return {**body, "receipt_sha256": identity_sha256(body)}


@dataclass(frozen=True, slots=True)
class _LaneAudits:
    temporal: _LaneAudit
    numeric: _LaneAudit
    authority: _LaneAudit


def _question_row_is_self_hashed(row: Mapping[str, Any]) -> bool:
    body = dict(row)
    declared = body.pop("source_row_sha256", None)
    return declared == identity_sha256(body)


def _strict_provider_input(plan: Mapping[str, Any]) -> dict[str, Any]:
    messages = plan.get("messages")
    _require(
        type(messages) is list
        and bool(messages)
        and plan.get("messages_sha256") == identity_sha256(messages),
        f"V2 messages changed at ordinal {plan.get('ordinal')}",
    )
    assert type(messages) is list
    final = messages[-1]
    _require(
        type(final) is dict
        and final.get("role") == "user"
        and type(final.get("content")) is str,
        f"V2 provider envelope changed at ordinal {plan.get('ordinal')}",
    )
    try:
        value = json.loads(final["content"])
    except json.JSONDecodeError as exc:
        raise LockedSpecialistFinalReconciliationV3Error(
            f"V2 provider input is not strict JSON at ordinal {plan.get('ordinal')}"
        ) from exc
    _require(
        type(value) is dict
        and quote_sha256(value.get("dated_question"))
        == plan.get("dated_question_sha256"),
        f"V2 provider input escaped its dated question at {plan.get('ordinal')}",
    )
    transform = plan.get("adapter_prompt_transform")
    if type(transform) is dict and transform.get("provider_input_sha256") is not None:
        _require(
            transform["provider_input_sha256"] == identity_sha256(value),
            f"V2 transformed provider input changed at {plan.get('ordinal')}",
        )
    assert_gold_blind(value, path=f"v3_provider_input_{plan.get('ordinal')}")
    return dict(value)


def _load_verified_sources(
    *,
    preflight_path: Path,
    run_path: Path,
    replay_path: Path,
    expected_preflight_sha256: str,
    expected_run_sha256: str,
    expected_replay_sha256: str,
) -> _SourceBundle:
    for value, label in (
        (expected_preflight_sha256, "expected V2 preflight"),
        (expected_run_sha256, "expected V2 run"),
        (expected_replay_sha256, "expected V2 replay"),
    ):
        require_sha256(value, label)
    preflight = read_sealed_json(preflight_path)
    run = read_sealed_json(run_path)
    replay = read_sealed_json(replay_path)
    _require(
        preflight.sha256 == expected_preflight_sha256
        and run.sha256 == expected_run_sha256
        and replay.sha256 == expected_replay_sha256,
        "V2 source artifact SHA binding changed",
    )
    _require(
        run.sha256 == replay.sha256 and run.payload == replay.payload,
        "V2 run and replay are not byte-identical",
    )
    _require(
        preflight.payload.get("format") == answer_v2.PREFLIGHT_FORMAT
        and run.payload.get("format") == answer_v2.FORMAT
        and run.payload.get("preflight_artifact_sha256") == preflight.sha256
        and run.payload.get("gold_loaded") is False
        and replay.payload.get("gold_loaded") is False
        and run.payload.get("physical_provider_calls_during_materialization") == 0
        and run.payload.get("retained_transformer_token_state_bytes") == 0,
        "V2 answer boundary changed",
    )
    _require(
        preflight.payload.get("gold_loaded") is False
        and preflight.payload.get("provider_calls") == 0
        and preflight.payload.get("retained_transformer_token_state_bytes") == 0
        and preflight.payload.get("hard_complete_chat_token_cap")
        == HARD_COMPLETE_CHAT_TOKEN_CAP
        and preflight.payload.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
        and type(preflight.payload.get("observed_max_complete_envelope_tokens"))
        is int
        and preflight.payload["observed_max_complete_envelope_tokens"]
        <= HARD_COMPLETE_CHAT_TOKEN_CAP,
        "V2 hard prompt-token boundary changed",
    )

    _prompts, raw_plans = answer_v2.validate_preflight_artifact(preflight)
    plans = tuple(dict(row) for row in raw_plans)
    questions = run.payload.get("questions")
    judge_rows = run.payload.get("judge_rows")
    _require(
        len(plans) == EXPECTED_QUESTION_COUNT
        and type(questions) is list
        and type(judge_rows) is list
        and len(questions) == len(judge_rows) == EXPECTED_QUESTION_COUNT,
        "V2 full-100 population changed",
    )
    assert type(questions) is list and type(judge_rows) is list
    rows = tuple(dict(row) for row in questions)
    _require(
        tuple(int(row.get("ordinal", -1)) for row in plans)
        == tuple(range(EXPECTED_QUESTION_COUNT))
        == tuple(int(row.get("ordinal", -1)) for row in rows),
        "V2 answer population order changed",
    )
    providers: dict[int, dict[str, Any]] = {}
    physical_count = 0
    for plan, row, judge in zip(plans, rows, judge_rows, strict=True):
        ordinal = int(plan["ordinal"])
        _require(
            _question_row_is_self_hashed(row)
            and judge == judge_row_projection(row)
            and row.get("question_id") == plan.get("question_id")
            and row.get("question_sha256") == plan.get("question_sha256")
            and row.get("dated_question_sha256")
            == plan.get("dated_question_sha256")
            and row.get("parent_prediction_sha256")
            == plan.get("parent_prediction_sha256")
            and quote_sha256(row.get("prediction")) == row.get("prediction_sha256")
            and row.get("retained_transformer_token_state_bytes") == 0,
            f"V2 row/plan binding changed at ordinal {ordinal}",
        )
        if plan.get("mode") == answer_v2.SPECIALIST_MODE:
            provider = _strict_provider_input(plan)
            _require(
                type(plan.get("prompt_token_proxy")) is int
                and plan["prompt_token_proxy"] + OUTPUT_TOKEN_RESERVE
                <= HARD_COMPLETE_CHAT_TOKEN_CAP,
                f"V2 prompt escaped 8k at ordinal {ordinal}",
            )
            providers[ordinal] = provider
            physical_count += 1
    _require(
        physical_count == EXPECTED_PHYSICAL_PROMPT_COUNT
        and run.payload.get("provider_question_count")
        == EXPECTED_PHYSICAL_PROMPT_COUNT,
        "V2 physical prompt population changed",
    )
    assert_gold_blind(run.payload, path="v3_verified_v2_run")
    return _SourceBundle(preflight, run, replay, plans, rows, providers)


def _base_lane_status(
    *,
    lane: str,
    plan: Mapping[str, Any],
    source_row: Mapping[str, Any],
    provider_input: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "answer_plan_receipt_sha256": plan["answer_plan_receipt_sha256"],
        "format": LANE_STATUS_FORMAT,
        "lane": lane,
        "messages_sha256": plan["messages_sha256"],
        "ordinal": plan["ordinal"],
        "provider_input_sha256": identity_sha256(provider_input),
        "question_id": plan["question_id"],
        "route_id": plan["route_id"],
        "source_answer_row_sha256": source_row["source_row_sha256"],
    }


def _sealed_status_row(body: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(body)
    unsigned.update(
        {
            "gold_loaded": False,
            "provider_calls": 0,
            "retained_transformer_token_state_bytes": 0,
        }
    )
    assert_gold_blind(unsigned, path="v3_lane_status")
    return {**unsigned, "receipt_sha256": identity_sha256(unsigned)}


def _audit_temporal(bundle: _SourceBundle) -> _LaneAudit:
    statuses: list[dict[str, Any]] = []
    resolved: list[dict[str, Any]] = []
    resolutions: dict[int, TemporalEventResolution] = {}
    for plan, source_row in zip(bundle.plans, bundle.rows, strict=True):
        ordinal = int(plan["ordinal"])
        provider = bundle.providers_by_ordinal.get(ordinal)
        if provider is None:
            continue
        result = reconcile_temporal_events(
            dated_question=provider["dated_question"],
            candidate_prediction=source_row["prediction"],
            parent_prediction=plan["parent_prediction"],
            provider_input=provider,
            validation_contract=plan["validation_contract"],
            allowed_handle_ids=plan["allowed_handle_ids"],
            source_receipt_sha256=plan["answer_plan_receipt_sha256"],
        )
        route_eligible = plan["route_id"] in TEMPORAL_ROUTE_IDS
        base = _base_lane_status(
            lane="temporal", plan=plan, source_row=source_row,
            provider_input=provider,
        )
        if result is None:
            status = _sealed_status_row(
                {
                    **base,
                    "composition_eligible": False,
                    "reason": "no_question_bound_temporal_resolution",
                    "resolution": None,
                    "status": "unresolved",
                }
            )
        else:
            projection = result.projection()
            status = _sealed_status_row(
                {
                    **base,
                    "composition_eligible": route_eligible,
                    "reason": (
                        "resolved_on_temporal_route"
                        if route_eligible
                        else "resolved_but_route_ineligible"
                    ),
                    "resolution": projection,
                    "status": "resolved",
                }
            )
            summary = {
                "answer_plan_receipt_sha256": plan["answer_plan_receipt_sha256"],
                "chosen": result.prediction_source,
                "computed": result.proof["computed"],
                "handles": list(result.proof_handle_ids),
                "ordinal": ordinal,
                "proof_type": result.operation,
                "provider_input_sha256": identity_sha256(provider),
                "question_id": plan["question_id"],
                "receipt": result.receipt_sha256,
                "resolved_prediction_sha256": quote_sha256(result.prediction),
                "route_eligible": route_eligible,
                "source_answer_row_sha256": source_row["source_row_sha256"],
            }
            resolved.append(summary)
            if route_eligible:
                resolutions[ordinal] = result
        statuses.append(status)
    _require(
        len(statuses) == len(bundle.providers_by_ordinal),
        "temporal audit population changed",
    )
    return _LaneAudit("temporal", tuple(statuses), tuple(resolved), resolutions)


def _audit_numeric(bundle: _SourceBundle) -> _LaneAudit:
    statuses: list[dict[str, Any]] = []
    resolved: list[dict[str, Any]] = []
    resolutions: dict[int, NumericEvidenceReconciliationV2Receipt] = {}
    for plan, source_row in zip(bundle.plans, bundle.rows, strict=True):
        ordinal = int(plan["ordinal"])
        provider = bundle.providers_by_ordinal.get(ordinal)
        if provider is None:
            continue
        provider_sha = identity_sha256(provider)
        receipt = reconcile_sealed_numeric_evidence_v2(
            provider, sealed_provider_input_sha256=provider_sha
        )
        route_eligible = plan["route_id"] in NUMERIC_ROUTE_IDS
        composition_eligible = receipt.supported and route_eligible
        projection = receipt.projection()
        base = _base_lane_status(
            lane="numeric", plan=plan, source_row=source_row,
            provider_input=provider,
        )
        status = _sealed_status_row(
            {
                **base,
                "composition_eligible": composition_eligible,
                "reason": (
                    receipt.reason
                    if route_eligible or not receipt.supported
                    else "supported_but_route_ineligible"
                ),
                "resolution": projection,
                "status": receipt.status.value,
            }
        )
        if receipt.supported:
            resolved.append(
                {
                    "answer_plan_receipt_sha256": plan[
                        "answer_plan_receipt_sha256"
                    ],
                    "mode": receipt.mode.value,
                    "numeric_result": receipt.numeric_result,
                    "ordinal": ordinal,
                    "provider_input_sha256": provider_sha,
                    "question_id": plan["question_id"],
                    "receipt": receipt.receipt_sha256,
                    "route_eligible": route_eligible,
                    "source_answer_row_sha256": source_row["source_row_sha256"],
                    "unit": receipt.unit,
                    "used_handle_ids": list(receipt.used_handle_ids),
                }
            )
            if route_eligible:
                resolutions[ordinal] = receipt
        statuses.append(status)
    _require(
        len(statuses) == len(bundle.providers_by_ordinal),
        "numeric audit population changed",
    )
    return _LaneAudit("numeric", tuple(statuses), tuple(resolved), resolutions)


def _authority_composition_eligibility(
    resolution: CrossPlaneAuthorityProtection,
    *,
    dated_question: str,
) -> tuple[bool, str]:
    """Return a generic, evidence-derived gate for raw authority proofs.

    Exact totals and explicit durations are complete claims.  A bounded
    cardinality proof is weaker: it is admitted only as a real composite of at
    least two independently receipt-bound operands.  Distinct/unique counts
    require an explicit identity/dedup proof that the v1 authority projection
    does not carry, and recurring totals require a closed recurrence operator;
    both therefore remain with V2 rather than being guessed here.
    """

    if resolution.basis in {
        "exact_current_total",
        "explicit_duration",
        "exact_declared_total",
    }:
        return True, "complete_cross_plane_authority"
    _require(
        resolution.basis == "bounded_cardinality_lower_bound",
        "unknown authority basis",
    )
    question = dated_question.rsplit("]\n", 1)[-1]
    evidence = resolution.proof.get("parent_evidence")
    if type(evidence) is not list or len(evidence) < 2:
        return False, "bounded_cardinality_requires_composite_operands"
    if _DISTINCT_IDENTITY_RE.search(question):
        return False, "distinct_cardinality_requires_identity_dedup_proof"
    if _RECURRING_FREQUENCY_RE.search(question):
        return False, "recurring_cardinality_requires_frequency_closure"
    receipt_ids = [
        value
        for row in evidence
        if type(row) is dict
        for value in row.get("contract_item_receipt_sha256s", [])
    ]
    if (
        len(receipt_ids) < len(evidence)
        or len(receipt_ids) != len(set(receipt_ids))
        or any(type(value) is not str for value in receipt_ids)
    ):
        return False, "bounded_cardinality_operands_not_independent"
    return True, "independent_composite_cardinality_lower_bound"


def _audit_authority(bundle: _SourceBundle) -> _LaneAudit:
    statuses: list[dict[str, Any]] = []
    resolved: list[dict[str, Any]] = []
    resolutions: dict[int, CrossPlaneAuthorityProtection] = {}
    for plan, source_row in zip(bundle.plans, bundle.rows, strict=True):
        ordinal = int(plan["ordinal"])
        provider = bundle.providers_by_ordinal.get(ordinal)
        if provider is None:
            continue
        base = _base_lane_status(
            lane="authority", plan=plan, source_row=source_row,
            provider_input=provider,
        )
        required = (
            source_row.get("decision") == "replace"
            and type(source_row.get("proof_kind")) is str
            and type(source_row.get("completion_receipt_sha256")) is str
            and type(source_row.get("specialist_scope_receipt_sha256")) is str
            and type(source_row.get("used_handle_ids")) is list
            and bool(source_row["used_handle_ids"])
        )
        result: CrossPlaneAuthorityProtection | None = None
        if required:
            result = protect_parent_from_cross_plane_authority(
                dated_question=provider["dated_question"],
                parent_prediction=plan["parent_prediction"],
                replacement_prediction=source_row["prediction"],
                replacement_used_handle_ids=source_row["used_handle_ids"],
                replacement_proof_kind=source_row["proof_kind"],
                provider_input=provider,
                validation_contract=plan["validation_contract"],
                allowed_handle_ids=plan["allowed_handle_ids"],
                answer_plan_receipt_sha256=plan["answer_plan_receipt_sha256"],
                base_scope_receipt_sha256=source_row[
                    "specialist_scope_receipt_sha256"
                ],
                source_completion_sha256=source_row[
                    "completion_receipt_sha256"
                ],
            )
        if result is None:
            status = _sealed_status_row(
                {
                    **base,
                    "composition_eligible": False,
                    "reason": (
                        "no_cross_plane_parent_protection"
                        if required
                        else "v2_row_is_not_a_proven_replacement"
                    ),
                    "resolution": None,
                    "status": "unresolved",
                }
            )
        else:
            eligible, reason = _authority_composition_eligibility(
                result, dated_question=provider["dated_question"]
            )
            projection = result.projection()
            status = _sealed_status_row(
                {
                    **base,
                    "composition_eligible": eligible,
                    "reason": reason,
                    "resolution": projection,
                    "status": "resolved",
                }
            )
            resolved.append(
                {
                    "basis": result.basis,
                    "composition_eligible": eligible,
                    "eligibility_reason": reason,
                    "ordinal": ordinal,
                    "parent_support_handle_ids": list(
                        result.parent_support_handle_ids
                    ),
                    "provider_input_sha256": identity_sha256(provider),
                    "question_id": plan["question_id"],
                    "receipt": result.receipt_sha256,
                    "replacement_handle_ids": list(
                        result.replacement_handle_ids
                    ),
                    "source_answer_row_sha256": source_row["source_row_sha256"],
                }
            )
            if eligible:
                resolutions[ordinal] = result
        statuses.append(status)
    _require(
        len(statuses) == len(bundle.providers_by_ordinal),
        "authority audit population changed",
    )
    return _LaneAudit("authority", tuple(statuses), tuple(resolved), resolutions)


def _build_lane_audits(bundle: _SourceBundle) -> _LaneAudits:
    return _LaneAudits(
        temporal=_audit_temporal(bundle),
        numeric=_audit_numeric(bundle),
        authority=_audit_authority(bundle),
    )


def _same_number(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)


def _numeric_prediction_matches(
    prediction: str,
    *,
    dated_question: str,
    receipt: NumericEvidenceReconciliationReceipt
    | NumericEvidenceReconciliationV2Receipt,
) -> bool:
    if receipt.boolean_result is not None:
        normalized = re.sub(r"[^a-z]+", " ", prediction.casefold()).strip()
        if receipt.boolean_result:
            return normalized in {"yes", "yes it is", "yes it did", "true"}
        return normalized in {"no", "no it is not", "no it did not", "false"}
    mentions = numeric_mentions(prediction, question=dated_question)
    return (
        len(mentions) == 1
        and mentions[0].qualifier is NumericQualifier.EXACT
        and receipt.numeric_result is not None
        and _same_number(mentions[0].value, receipt.numeric_result)
    )


def _format_scalar(value: float) -> str:
    _require(math.isfinite(value), "numeric result is not finite")
    return str(int(value)) if value.is_integer() else format(value, ".12g")


def _pluralize_unit(unit: str) -> str:
    if unit.endswith(("species", "series")):
        return unit
    if unit.endswith(("s", "x", "z", "ch", "sh")):
        return f"{unit}es"
    if unit.endswith("y") and len(unit) > 1 and unit[-2] not in "aeiou":
        return f"{unit[:-1]}ies"
    return f"{unit}s"


def _render_numeric_prediction(
    *,
    candidate_prediction: str,
    parent_prediction: str,
    dated_question: str,
    receipt: NumericEvidenceReconciliationReceipt
    | NumericEvidenceReconciliationV2Receipt,
) -> tuple[str, str]:
    _require(receipt.supported, "cannot render an unsupported numeric receipt")
    for prediction, source in (
        (candidate_prediction, "candidate"),
        (parent_prediction, "parent"),
    ):
        if _numeric_prediction_matches(
            prediction, dated_question=dated_question, receipt=receipt
        ):
            return prediction, source
    if receipt.boolean_result is not None:
        return ("Yes." if receipt.boolean_result else "No."), "computed"
    assert receipt.numeric_result is not None
    scalar = _format_scalar(receipt.numeric_result)
    unit = receipt.unit
    if unit == "$":
        prediction = f"${scalar}"
    elif unit == "%":
        prediction = f"{scalar}%"
    elif unit and "/" in unit:
        numerator, denominator = unit.split("/", 1)
        numerator = numerator.replace("_", " ")
        suffix = (
            numerator
            if receipt.numeric_result == 1
            else _pluralize_unit(numerator)
        )
        prediction = f"{scalar} {suffix} per {denominator}"
    elif unit:
        suffix = unit if receipt.numeric_result == 1 else _pluralize_unit(unit)
        prediction = f"{scalar} {suffix}"
    else:
        prediction = scalar
    return prediction, "computed"


def _result_row(
    *,
    bundle: _SourceBundle,
    plan: Mapping[str, Any],
    source_row: Mapping[str, Any],
    prediction: str,
    prediction_source: str,
    lane: str,
    resolution_projection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    require_text(prediction, "V3 prediction")
    body = {
        "answer_plan_receipt_sha256": plan["answer_plan_receipt_sha256"],
        "changed_from_parent": prediction != plan["parent_prediction"],
        "changed_from_v2": prediction != source_row["prediction"],
        "construction_question_receipt_sha256": plan[
            "construction_question_receipt_sha256"
        ],
        "dated_question_sha256": plan["dated_question_sha256"],
        "decision_lane": lane,
        "format": RESULT_ROW_FORMAT,
        "gold_loaded": False,
        "ordinal": plan["ordinal"],
        "parent_prediction_sha256": plan["parent_prediction_sha256"],
        "physical_provider_calls": 0,
        "prediction": prediction,
        "prediction_sha256": quote_sha256(prediction),
        "prediction_source": prediction_source,
        "question_id": plan["question_id"],
        "question_sha256": plan["question_sha256"],
        "reconciliation": None if resolution_projection is None else dict(
            resolution_projection
        ),
        "retained_transformer_token_state_bytes": 0,
        "route_id": plan["route_id"],
        "v2_prediction_sha256": source_row["prediction_sha256"],
        "v2_preflight_artifact_sha256": bundle.preflight.sha256,
        "v2_replay_artifact_sha256": bundle.replay.sha256,
        "v2_source_row_sha256": source_row["source_row_sha256"],
        "v2_run_artifact_sha256": bundle.run.sha256,
    }
    assert_gold_blind(body, path=f"v3_result_{plan['ordinal']}")
    return {**body, "source_row_sha256": identity_sha256(body)}


def _compose_rows(
    bundle: _SourceBundle,
    audits: _LaneAudits,
) -> tuple[dict[str, Any], ...]:
    results: list[dict[str, Any]] = []
    for plan, source_row in zip(bundle.plans, bundle.rows, strict=True):
        ordinal = int(plan["ordinal"])
        provider = bundle.providers_by_ordinal.get(ordinal)
        temporal = audits.temporal.resolutions_by_ordinal.get(ordinal)
        numeric = audits.numeric.resolutions_by_ordinal.get(ordinal)
        authority = audits.authority.resolutions_by_ordinal.get(ordinal)
        if temporal is not None:
            assert isinstance(temporal, TemporalEventResolution)
            result = _result_row(
                bundle=bundle,
                plan=plan,
                source_row=source_row,
                prediction=temporal.prediction,
                prediction_source=(
                    "locked_v3_temporal_" + temporal.prediction_source
                ),
                lane="question_bound_temporal",
                resolution_projection=temporal.projection(),
            )
        elif numeric is not None:
            _require(
                isinstance(
                    numeric,
                    (
                        NumericEvidenceReconciliationReceipt,
                        NumericEvidenceReconciliationV2Receipt,
                    ),
                ),
                "numeric resolution changed type",
            )
            _require(provider is not None, "numeric provider input disappeared")
            prediction, source = _render_numeric_prediction(
                candidate_prediction=source_row["prediction"],
                parent_prediction=plan["parent_prediction"],
                dated_question=provider["dated_question"],
                receipt=numeric,
            )
            result = _result_row(
                bundle=bundle,
                plan=plan,
                source_row=source_row,
                prediction=prediction,
                prediction_source=f"locked_v3_numeric_{source}",
                lane="sealed_numeric",
                resolution_projection=numeric.projection(),
            )
        elif authority is not None:
            assert isinstance(authority, CrossPlaneAuthorityProtection)
            result = _result_row(
                bundle=bundle,
                plan=plan,
                source_row=source_row,
                prediction=authority.prediction,
                prediction_source="locked_v3_cross_plane_protected_parent",
                lane="cross_plane_parent_protection",
                resolution_projection=authority.projection(),
            )
        else:
            result = _result_row(
                bundle=bundle,
                plan=plan,
                source_row=source_row,
                prediction=source_row["prediction"],
                prediction_source="locked_v3_v2_fallback",
                lane="v2_fallback",
                resolution_projection=None,
            )
        results.append(result)
    _require(
        tuple(row["ordinal"] for row in results)
        == tuple(range(len(bundle.rows)))
        and all(_question_row_is_self_hashed(row) for row in results),
        "V3 result population changed",
    )
    return tuple(results)


def _policy_projection(
    bundle: _SourceBundle,
    audits: _LaneAudits,
    *,
    expected_temporal_status_population_sha256: str,
    expected_numeric_status_population_sha256: str,
    expected_authority_status_population_sha256: str,
) -> dict[str, Any]:
    expected = {
        "authority": expected_authority_status_population_sha256,
        "numeric": expected_numeric_status_population_sha256,
        "temporal": expected_temporal_status_population_sha256,
    }
    observed = {
        "authority": audits.authority.status_population_sha256,
        "numeric": audits.numeric.status_population_sha256,
        "temporal": audits.temporal.status_population_sha256,
    }
    for lane, value in expected.items():
        require_sha256(value, f"expected frozen {lane} full-72 status population")
        _require(
            value == observed[lane],
            f"{lane} full-72 status population differs from its frozen receipt",
        )
    for value, label in (
        (bundle.preflight.sha256, "V2 preflight source"),
        (bundle.run.sha256, "V2 run source"),
        (bundle.replay.sha256, "V2 replay source"),
    ):
        require_sha256(value, label)
    body = {
        "authority_composition_gate": {
            "bounded_composite_minimum_operand_count": 2,
            "distinct_identity_requires_explicit_dedup_proof": True,
            "recurring_frequency_requires_closure": True,
        },
        "composition_order": list(COMPOSITION_ORDER),
        "format": POLICY_FORMAT,
        "frozen_full72_status_population_sha256s": expected,
        "gold_loaded": False,
        "hard_complete_chat_token_cap": HARD_COMPLETE_CHAT_TOKEN_CAP,
        "local_lane_audit_receipts": {
            "authority": audits.authority.projection()["receipt_sha256"],
            "numeric": audits.numeric.projection()["receipt_sha256"],
            "temporal": audits.temporal.projection()["receipt_sha256"],
        },
        "local_lane_resolved_population_sha256s": {
            "authority": audits.authority.resolved_population_sha256,
            "numeric": audits.numeric.resolved_population_sha256,
            "temporal": audits.temporal.resolved_population_sha256,
        },
        "local_lane_status_population_sha256s": {
            "authority": audits.authority.status_population_sha256,
            "numeric": audits.numeric.status_population_sha256,
            "temporal": audits.temporal.status_population_sha256,
        },
        "physical_provider_calls": 0,
        "retained_transformer_token_state_bytes": 0,
        "v2_preflight_artifact_sha256": bundle.preflight.sha256,
        "v2_replay_artifact_sha256": bundle.replay.sha256,
        "v2_run_artifact_sha256": bundle.run.sha256,
    }
    assert_gold_blind(body, path="v3_composition_policy")
    return {**body, "receipt_sha256": identity_sha256(body)}


def _materialization_projection(
    bundle: _SourceBundle,
    audits: _LaneAudits,
    *,
    expected_temporal_status_population_sha256: str,
    expected_numeric_status_population_sha256: str,
    expected_authority_status_population_sha256: str,
) -> dict[str, Any]:
    policy = _policy_projection(
        bundle,
        audits,
        expected_temporal_status_population_sha256=(
            expected_temporal_status_population_sha256
        ),
        expected_numeric_status_population_sha256=(
            expected_numeric_status_population_sha256
        ),
        expected_authority_status_population_sha256=(
            expected_authority_status_population_sha256
        ),
    )
    questions = _compose_rows(bundle, audits)
    judge_rows = [judge_row_projection(row) for row in questions]
    payload = {
        "changed_from_v2_count": sum(row["changed_from_v2"] for row in questions),
        "composition_policy": policy,
        "format": FORMAT,
        "gold_loaded": False,
        "hard_complete_chat_token_cap": HARD_COMPLETE_CHAT_TOKEN_CAP,
        "judge_rows": judge_rows,
        "lane_audits": {
            "authority": audits.authority.projection(),
            "numeric": audits.numeric.projection(),
            "temporal": audits.temporal.projection(),
        },
        "max_chat_prompt_tokens": bundle.preflight.payload[
            "max_chat_prompt_tokens"
        ],
        "observed_max_complete_envelope_tokens": bundle.preflight.payload[
            "observed_max_complete_envelope_tokens"
        ],
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "physical_provider_calls_during_materialization": 0,
        "question_count": EXPECTED_QUESTION_COUNT,
        "questions": list(questions),
        "retained_transformer_token_state_bytes": 0,
        "v2_preflight_artifact_sha256": bundle.preflight.sha256,
        "v2_replay_artifact_sha256": bundle.replay.sha256,
        "v2_run_artifact_sha256": bundle.run.sha256,
    }
    _require(
        len(payload["judge_rows"]) == EXPECTED_QUESTION_COUNT
        and payload["observed_max_complete_envelope_tokens"]
        <= HARD_COMPLETE_CHAT_TOKEN_CAP,
        "V3 output population or token cap changed",
    )
    assert_gold_blind(payload, path="locked_specialist_final_reconciliation_v3")
    return payload


def _load_from_args(args: argparse.Namespace) -> _SourceBundle:
    return _load_verified_sources(
        preflight_path=Path(args.preflight),
        run_path=Path(args.v2_run),
        replay_path=Path(args.v2_replay),
        expected_preflight_sha256=str(args.expected_preflight_sha256),
        expected_run_sha256=str(args.expected_v2_run_sha256),
        expected_replay_sha256=str(args.expected_v2_replay_sha256),
    )


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    bundle = _load_from_args(args)
    audits = _build_lane_audits(bundle)
    result = {
        "authority": {
            "eligible_count": len(audits.authority.resolutions_by_ordinal),
            "resolved_count": len(audits.authority.resolved_rows),
            "resolved_ordinals": [
                row["ordinal"] for row in audits.authority.resolved_rows
            ],
            "resolved_population_sha256": (
                audits.authority.resolved_population_sha256
            ),
            "status_population_sha256": audits.authority.status_population_sha256,
        },
        "gold_loaded": False,
        "numeric": {
            "eligible_count": len(audits.numeric.resolutions_by_ordinal),
            "resolved_count": len(audits.numeric.resolved_rows),
            "resolved_ordinals": [row["ordinal"] for row in audits.numeric.resolved_rows],
            "resolved_population_sha256": audits.numeric.resolved_population_sha256,
            "status_population_sha256": audits.numeric.status_population_sha256,
        },
        "physical_provider_calls": 0,
        "retained_transformer_token_state_bytes": 0,
        "temporal": {
            "eligible_count": len(audits.temporal.resolutions_by_ordinal),
            "resolved_count": len(audits.temporal.resolved_rows),
            "resolved_ordinals": [
                row["ordinal"] for row in audits.temporal.resolved_rows
            ],
            "resolved_population_sha256": (
                audits.temporal.resolved_population_sha256
            ),
            "status_population_sha256": audits.temporal.status_population_sha256,
        },
        "v2_preflight_artifact_sha256": bundle.preflight.sha256,
        "v2_replay_artifact_sha256": bundle.replay.sha256,
        "v2_run_artifact_sha256": bundle.run.sha256,
    }
    assert_gold_blind(result, path="v3_read_only_audit")
    return result


def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    bundle = _load_from_args(args)
    audits = _build_lane_audits(bundle)
    payload = _materialization_projection(
        bundle,
        audits,
        expected_temporal_status_population_sha256=str(
            args.expected_temporal_status_population_sha256
        ),
        expected_numeric_status_population_sha256=str(
            args.expected_numeric_status_population_sha256
        ),
        expected_authority_status_population_sha256=str(
            args.expected_authority_status_population_sha256
        ),
    )
    artifact, created = publish_sealed_json(Path(args.output_root) / RUN_NAME, payload)
    return {
        "changed_from_v2_count": payload["changed_from_v2_count"],
        "physical_provider_calls": 0,
        "question_count": EXPECTED_QUESTION_COUNT,
        "run_sha256": artifact.sha256,
        "terminal_run_replayed": not created,
    }


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    bundle = _load_from_args(args)
    audits = _build_lane_audits(bundle)
    rebuilt = _materialization_projection(
        bundle,
        audits,
        expected_temporal_status_population_sha256=str(
            args.expected_temporal_status_population_sha256
        ),
        expected_numeric_status_population_sha256=str(
            args.expected_numeric_status_population_sha256
        ),
        expected_authority_status_population_sha256=str(
            args.expected_authority_status_population_sha256
        ),
    )
    terminal = read_sealed_json(Path(args.output_root) / RUN_NAME)
    _require(
        terminal.sha256
        == require_sha256(args.expected_v3_run_sha256, "expected V3 run")
        and terminal.payload == rebuilt,
        "V3 terminal run differs from provider-free reconstruction",
    )
    replay, _created = publish_sealed_json(
        Path(args.output_root) / REPLAY_NAME, terminal.payload
    )
    _require(
        replay.sha256 == terminal.sha256,
        "V3 replay is not byte-identical",
    )
    return {
        "byte_identical": True,
        "physical_provider_calls": 0,
        "replay_sha256": replay.sha256,
        "run_sha256": terminal.sha256,
    }


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--v2-run", type=Path, default=DEFAULT_V2_RUN)
    parser.add_argument("--v2-replay", type=Path, default=DEFAULT_V2_REPLAY)
    parser.add_argument(
        "--expected-preflight-sha256", default=EXPECTED_V2_PREFLIGHT_SHA256
    )
    parser.add_argument(
        "--expected-v2-run-sha256", default=EXPECTED_V2_RUN_SHA256
    )
    parser.add_argument(
        "--expected-v2-replay-sha256", default=EXPECTED_V2_REPLAY_SHA256
    )


def _add_frozen_lane_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--expected-temporal-status-population-sha256", required=True
    )
    parser.add_argument(
        "--expected-numeric-status-population-sha256", required=True
    )
    parser.add_argument(
        "--expected-authority-status-population-sha256", required=True
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="read-only full-72 lane audit")
    _add_source_args(audit)
    materialize = commands.add_parser(
        "materialize", help="seal provider-free V3 after lane populations freeze"
    )
    _add_source_args(materialize)
    _add_frozen_lane_args(materialize)
    materialize.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    replay = commands.add_parser(
        "replay", help="prove byte-identical provider-free V3 replay"
    )
    _add_source_args(replay)
    _add_frozen_lane_args(replay)
    replay.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    replay.add_argument("--expected-v3-run-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        result = run_audit(args)
    elif args.command == "materialize":
        result = run_materialize(args)
    else:
        result = run_replay(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_PREFLIGHT",
    "DEFAULT_V2_REPLAY",
    "DEFAULT_V2_RUN",
    "EXPECTED_V2_PREFLIGHT_SHA256",
    "EXPECTED_V2_REPLAY_SHA256",
    "EXPECTED_V2_RUN_SHA256",
    "FORMAT",
    "LockedSpecialistFinalReconciliationV3Error",
    "REPLAY_NAME",
    "RUN_NAME",
    "build_parser",
    "main",
    "run_audit",
    "run_materialize",
    "run_replay",
]
