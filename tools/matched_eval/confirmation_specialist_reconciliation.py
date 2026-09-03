"""Pure deterministic specialist-v3 reconciliation core.

The confirmation prediction path uses these receipt-bound lanes without
importing the historical runner or any of its validation source lifecycle.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from memory_condense.domain.discourse import quote_sha256
from tools.matched_eval.artifacts import SealedArtifact
from tools.matched_eval.contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.cross_plane_authority import (
    CrossPlaneAuthorityProtection,
    protect_parent_from_cross_plane_authority,
)
from tools.matched_eval.numeric_evidence_reconciler import (
    NumericEvidenceReconciliationReceipt,
)
from tools.matched_eval.numeric_evidence_reconciler_v2 import (
    NumericEvidenceReconciliationV2Receipt,
    reconcile_sealed_numeric_evidence_v2,
)
from tools.matched_eval.temporal_event_reconciler import (
    TemporalEventResolution,
    reconcile_temporal_events,
)
from tools.matched_eval.typed_numeric_semantics import (
    NumericQualifier,
    numeric_mentions,
)

FORMAT = "memory-condense-locked-specialist-final-reconciliation-v3"
RESULT_ROW_FORMAT = f"{FORMAT}-result-row-v1"
LANE_STATUS_FORMAT = f"{FORMAT}-lane-status-row-v1"
AUDIT_FORMAT = f"{FORMAT}-lane-audit-v1"
TEMPORAL_ROUTE_IDS = frozenset({"temporal_timeline"})
NUMERIC_ROUTE_IDS = frozenset({"numeric_reduce"})
COMPOSITION_ORDER = (
    "question_bound_temporal",
    "sealed_numeric",
    "cross_plane_parent_protection",
    "v2_fallback",
)
_DISTINCT_IDENTITY_RE = re.compile(r"\b(?:different|distinct|unique)\b", re.IGNORECASE)
_RECURRING_FREQUENCY_RE = re.compile(
    r"\b(?:per\s+(?:day|week|month|year)|each\s+(?:day|week|month|year)|"
    r"weekly|monthly|typical\s+week)\b",
    re.IGNORECASE,
)


class LockedSpecialistFinalReconciliationV3Error(MatchedEvalContractError):
    """A deterministic specialist reconciliation invariant changed."""


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




SourceBundle = _SourceBundle
LaneAudits = _LaneAudits
build_lane_audits = _build_lane_audits
compose_rows = _compose_rows

__all__ = [
    "COMPOSITION_ORDER",
    "LaneAudits",
    "SourceBundle",
    "build_lane_audits",
    "compose_rows",
]
