#!/usr/bin/env python3
"""Run the provider-free specialist assay on the frozen unresolved ten.

The construction command is deliberately small and gold-blind.  It streams
one immutable 1M namespace at a time, routes from dated question text only,
lets every applicable specialist select under its own budget, deduplicates
only after selection, and fits the resulting typed union into the hard 8k
answer envelope.  No provider is called.

The audit command first verifies that sealed construction and only then opens
the post-hoc target-owner plan.  Its source-reach metrics therefore cannot
affect routing, retrieval, ranking, composition, or fitting.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.domain._tokenizer import (  # noqa: E402
    count_chat_prompt_token_proxy,
)
from tools import run_reduced_second_read_retrieval_assay as reduced_cli  # noqa: E402
from tools._routed_repair_routing import (  # noqa: E402
    RoutedRepairStyle,
    route_question,
)
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
from tools.matched_eval.numeric_operand_specialist import (  # noqa: E402
    MECHANISM_ID as NUMERIC_MECHANISM_ID,
    NumericOperandClosureResult,
    adapt_numeric_operand_closure_to_typed_contribution,
    scan_numeric_operand_closure,
)
from tools.matched_eval.profile_preference_specialist import (  # noqa: E402
    MECHANISM_ID as PROFILE_MECHANISM_ID,
    ProfilePreferenceResult,
    adapt_profile_preference_to_typed_contribution,
    select_profile_preference_evidence,
)
from tools.matched_eval.protected_parent_contribution import (  # noqa: E402
    MECHANISM_ID as PROTECTED_PARENT_MECHANISM_ID,
    ProtectedParentContributionSet,
    rehydrate_protected_parent_contributions,
)
from tools.matched_eval.prompt_tick_contracts import (  # noqa: E402
    CallBudget,
    LaneBudget,
)
from tools.matched_eval.specialist_scoped_completion import (  # noqa: E402
    SPECIALIST_ADVISORY_FORMAT,
)
from tools.matched_eval.temporal_insufficiency_specialist import (  # noqa: E402
    BundleRole,
    MECHANISM_ID as TEMPORAL_MECHANISM_ID,
    SpecialistRoute,
    TemporalInsufficiencyResult,
    adapt_temporal_insufficiency_to_typed_contribution,
    scan_temporal_insufficiency_specialist,
)
from tools.matched_eval.typed_additive_composer import (  # noqa: E402
    compose_additive_typed_evidence,
)
from tools.matched_eval.typed_lane_allocator import (  # noqa: E402
    lane_content_token_proxy,
)
from tools.matched_eval.typed_memory_final_arm import (  # noqa: E402
    LOCAL_RETENTION_PRIORITY_WIDTH,
    fit_typed_final_prompt,
    render_final_messages,
)
from tools.matched_eval.typed_operator_adapter import (  # noqa: E402
    EvidenceStatus,
    ProviderPayloadMode,
    TypedEvidenceContribution,
)
from tools.matched_eval.typed_operator_spec import (  # noqa: E402
    TemporalMode,
    TypedOperatorSpec,
    compile_typed_operator_spec,
)


FORMAT = "memory-condense-reduced-specialist-retrieval-assay-v2"
CONSTRUCTION_FORMAT = f"{FORMAT}-construction"
AUDIT_FORMAT = f"{FORMAT}-posthoc-target-audit"
CONSTRUCTION_NAME = "reduced-specialist-construction-v2.json"
AUDIT_NAME = "reduced-specialist-target-audit-v2.json"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TARGET_ORDINALS = reduced_cli.TARGET_ORDINALS
QUESTION_COUNT = len(TARGET_ORDINALS)
HARD_COMPLETE_CHAT_TOKEN_CAP = 8_000
OUTPUT_TOKEN_RESERVE = 768

DEFAULT_FROZEN_REDUCED = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/reduced-second-read-missing10-v3/"
    "reduced-second-read-construction-v3.json"
)
EXPECTED_FROZEN_REDUCED_SHA256 = (
    "c1f3aeae910c072196e5d9550e5ddd723cb9df14fd79e9c4e0420dd611e013db"
)
DEFAULT_SOURCE_ROOT = reduced_cli.DEFAULT_SOURCE_ROOT
DEFAULT_TARGET_PLAN = reduced_cli.DEFAULT_TARGET_PLAN
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/reduced-specialist-missing10-v2"
)

MECHANISM_HANDLE_START = {
    # The cumulative parent stack owns prefixes zero through six.  Specialist
    # layers are appended after it and therefore use the only three remaining
    # six-digit opaque-ID partitions.
    NUMERIC_MECHANISM_ID: 700_001,
    PROFILE_MECHANISM_ID: 800_001,
    TEMPORAL_MECHANISM_ID: 900_001,
}
MECHANISM_PRIORITY = {
    NUMERIC_MECHANISM_ID: 20,
    PROFILE_MECHANISM_ID: 10,
    TEMPORAL_MECHANISM_ID: 30,
}
PROTECTED_PARENT_LANE_ID = "protected-parent"
SPECIALIST_LANE_ID = {
    NUMERIC_MECHANISM_ID: "numeric-operand",
    PROFILE_MECHANISM_ID: "profile-preference",
    TEMPORAL_MECHANISM_ID: "temporal-insufficiency",
}
NOMINAL_LANE_TOKEN_CAP = {
    PROTECTED_PARENT_LANE_ID: 1_200,
    SPECIALIST_LANE_ID[NUMERIC_MECHANISM_ID]: 1_400,
    SPECIALIST_LANE_ID[PROFILE_MECHANISM_ID]: 1_200,
    SPECIALIST_LANE_ID[TEMPORAL_MECHANISM_ID]: 1_400,
}


class ReducedSpecialistAssayError(MatchedEvalContractError):
    """A frozen input, gold firewall, specialist, or hard cap changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ReducedSpecialistAssayError(message)


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact array")
    return value


def _frozen_exact10(path: Path) -> SealedArtifact:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == EXPECTED_FROZEN_REDUCED_SHA256
        and artifact.payload.get("format")
        == "memory-condense-reduced-second-read-retrieval-assay-v3-construction"
        and artifact.payload.get("gold_loaded") is False
        and artifact.payload.get("target_labels_loaded") is False
        and artifact.payload.get("target_plan_loaded") is False
        and artifact.payload.get("new_provider_calls") == 0
        and artifact.payload.get("retained_transformer_token_state_bytes") == 0
        and tuple(artifact.payload.get("ordinals", ())) == TARGET_ORDINALS,
        "frozen exact-ten construction changed",
    )
    return artifact


def _frozen_question_rows(artifact: SealedArtifact) -> tuple[dict[str, Any], ...]:
    values = _exact_list(artifact.payload.get("questions"), "frozen questions")
    rows = tuple(_exact_dict(row, "frozen question") for row in values)
    _require(
        len(rows) == QUESTION_COUNT
        and tuple(row.get("ordinal") for row in rows) == TARGET_ORDINALS
        and len({row.get("question_id") for row in rows}) == QUESTION_COUNT,
        "frozen exact-ten question population changed",
    )
    return rows


def _question_inputs(
    composition_row: Mapping[str, Any],
) -> tuple[str, str, str]:
    provider = _exact_dict(
        composition_row.get("provider_projection"), "composition provider"
    )
    provider_input = _exact_dict(provider.get("provider_input"), "provider input")
    dated_question = require_text(
        provider_input.get("dated_question"), "specialist dated question"
    )
    parent_prediction = require_text(
        composition_row.get("parent_prediction"), "specialist parent prediction"
    )
    question_id = require_text(
        composition_row.get("question_id"), "specialist question ID"
    )
    return dated_question, parent_prediction, question_id


def _requires_numeric_absence(spec: TypedOperatorSpec) -> bool:
    return bool(
        spec.requires_all_slots
        and sum(slot.requires_numeric for slot in spec.required_slots) >= 2
    )


def applicable_specialist_ids(
    dated_question: str,
) -> tuple[str, ...]:
    """Return the gold-blind specialist route for one dated question."""

    route = route_question(dated_question)
    spec = compile_typed_operator_spec(dated_question)
    methods: list[str] = []
    if route.style is RoutedRepairStyle.NUMERIC_REDUCE:
        methods.append(NUMERIC_MECHANISM_ID)
    if route.style is RoutedRepairStyle.SYNTHESIZE:
        methods.append(PROFILE_MECHANISM_ID)
    if (
        route.style is RoutedRepairStyle.TIMELINE
        or spec.temporal_mode is not TemporalMode.NONE
        or _requires_numeric_absence(spec)
    ):
        methods.append(TEMPORAL_MECHANISM_ID)
    return tuple(methods)


@dataclass(frozen=True, slots=True)
class _SpecialistRun:
    mechanism_id: str
    result: object
    contribution: TypedEvidenceContribution
    local_bindings: tuple[Any, ...]
    provider_projection: Mapping[str, Any]
    local_projection: Mapping[str, Any]
    specialist_receipt_sha256: str


def _numeric_run(index: Any, dated_question: str, spec: TypedOperatorSpec) -> _SpecialistRun:
    result = scan_numeric_operand_closure(
        index, dated_question, operator_spec=spec
    )
    contribution = adapt_numeric_operand_closure_to_typed_contribution(
        result,
        handle_start=MECHANISM_HANDLE_START[NUMERIC_MECHANISM_ID],
        group_start=MECHANISM_HANDLE_START[NUMERIC_MECHANISM_ID],
    )
    return _SpecialistRun(
        NUMERIC_MECHANISM_ID,
        result,
        contribution,
        result.local_bindings,
        result.provider_projection(),
        result.local_audit_projection(),
        result.receipt.receipt_sha256,
    )


def _profile_run(index: Any, dated_question: str) -> _SpecialistRun:
    result = select_profile_preference_evidence(index, dated_question)
    contribution = adapt_profile_preference_to_typed_contribution(
        result,
        handle_start=MECHANISM_HANDLE_START[PROFILE_MECHANISM_ID],
        group_start=MECHANISM_HANDLE_START[PROFILE_MECHANISM_ID],
    )
    return _SpecialistRun(
        PROFILE_MECHANISM_ID,
        result,
        contribution,
        result.local_bindings,
        result.provider_projection(),
        result.local_projection(),
        result.receipt_sha256,
    )


def _temporal_run(index: Any, dated_question: str) -> _SpecialistRun:
    result = scan_temporal_insufficiency_specialist(index, dated_question)
    contribution = adapt_temporal_insufficiency_to_typed_contribution(
        result,
        handle_start=MECHANISM_HANDLE_START[TEMPORAL_MECHANISM_ID],
        group_start=MECHANISM_HANDLE_START[TEMPORAL_MECHANISM_ID],
    )
    return _SpecialistRun(
        TEMPORAL_MECHANISM_ID,
        result,
        contribution,
        result.local_bindings,
        result.provider_projection(),
        result.local_audit_projection(),
        result.receipt.receipt_sha256,
    )


def _run_specialists(index: Any, dated_question: str) -> tuple[_SpecialistRun, ...]:
    spec = compile_typed_operator_spec(dated_question)
    runs: list[_SpecialistRun] = []
    for mechanism_id in applicable_specialist_ids(dated_question):
        if mechanism_id == NUMERIC_MECHANISM_ID:
            runs.append(_numeric_run(index, dated_question, spec))
        elif mechanism_id == PROFILE_MECHANISM_ID:
            runs.append(_profile_run(index, dated_question))
        else:
            runs.append(_temporal_run(index, dated_question))
    _require(
        tuple(row.mechanism_id for row in runs)
        == applicable_specialist_ids(dated_question),
        "specialist invocation order changed",
    )
    return tuple(runs)


def _span_keys(runs: Sequence[_SpecialistRun]) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for run in runs:
        _require(
            len(run.contribution.bindings) == len(run.local_bindings),
            "specialist adapter lost local binding order",
        )
        for typed, local in zip(
            run.contribution.bindings, run.local_bindings, strict=True
        ):
            result[typed.handle_id] = (
                identity_sha256(
                    {
                        "chunk_id": local.span.chunk_id,
                        "end_char": local.span.end_char,
                        "format": (
                            "memory-condense-locked-typed-memory-final-arm-v1-"
                            "canonical-coordinate-span-v1"
                        ),
                        "namespace_id": local.namespace_id,
                        "quote_sha256": local.quote_sha256,
                        "source_id": local.source_id,
                        "start_char": local.span.start_char,
                    }
                ),
            )
    return result


def _priority(*values: int) -> tuple[int, ...]:
    _require(
        len(values) <= LOCAL_RETENTION_PRIORITY_WIDTH,
        "specialist local priority changed width",
    )
    return (*values, *((0,) * (LOCAL_RETENTION_PRIORITY_WIDTH - len(values))))


def _specialist_local_priorities(
    runs: Sequence[_SpecialistRun],
) -> dict[str, tuple[int, ...]]:
    """Keep question-derived operator members ahead of generic neighbors."""

    priorities: dict[str, tuple[int, ...]] = {}
    for run in runs:
        handle_by_candidate = {
            local.candidate_id: binding.handle_id
            for binding, local in zip(
                run.contribution.bindings, run.local_bindings, strict=True
            )
        }
        if type(run.result) is TemporalInsufficiencyResult:
            bundle = run.result.temporal_bundle
            bundle_ids = (
                set() if bundle is None else set(bundle.ordered_candidate_ids)
            )
            winner = None if bundle is None else bundle.winner_candidate_id
            predecessor = (
                None if bundle is None else bundle.predecessor_candidate_id
            )
            for ordinal, candidate in enumerate(run.result.candidates):
                critical_role = candidate.bundle_role in {
                    BundleRole.WINNER,
                    BundleRole.PREDECESSOR,
                    BundleRole.ORDERED_OPERAND,
                    BundleRole.SLOT_SUPPORT,
                }
                priorities[handle_by_candidate[candidate.candidate_id]] = _priority(
                    1_000_000,
                    int(candidate.candidate_id in {winner, predecessor}),
                    int(critical_role),
                    int(candidate.candidate_id in bundle_ids),
                    int(candidate.candidate_id == winner),
                    int(candidate.candidate_id == predecessor),
                    -ordinal,
                )
        elif type(run.result) is NumericOperandClosureResult:
            group_rank: dict[str, int] = {}
            for group_ordinal, group in enumerate(run.result.operand_groups):
                for candidate_id in group.candidate_ids:
                    group_rank.setdefault(candidate_id, group_ordinal)
            for ordinal, candidate in enumerate(run.result.candidates):
                grouped = candidate.candidate_id in group_rank
                priorities[handle_by_candidate[candidate.candidate_id]] = _priority(
                    900_000,
                    int(grouped),
                    -group_rank.get(candidate.candidate_id, 1_000_000),
                    -ordinal,
                )
        elif type(run.result) is ProfilePreferenceResult:
            for ordinal, candidate in enumerate(run.result.candidates):
                priorities[handle_by_candidate[candidate.candidate_id]] = _priority(
                    800_000,
                    -ordinal,
                )
    return priorities


def _method_projection(run: _SpecialistRun) -> dict[str, Any]:
    candidate_handle_rows = [
        {
            "candidate_id": local.candidate_id,
            "handle_id": binding.handle_id,
        }
        for binding, local in zip(
            run.contribution.bindings, run.local_bindings, strict=True
        )
    ]
    body = {
        "accepted_typed_item_count": len(
            run.contribution.parsed.accepted_items
        ),
        "candidate_handle_rows": candidate_handle_rows,
        "local_binding_count": len(run.local_bindings),
        "local_bindings": [row.projection() for row in run.local_bindings],
        "local_projection_sha256": identity_sha256(run.local_projection),
        "mechanism_id": run.mechanism_id,
        "new_provider_calls": 0,
        "provider_projection": dict(run.provider_projection),
        "provider_projection_sha256": identity_sha256(run.provider_projection),
        "retained_transformer_token_state_bytes": 0,
        "source_ids": list(
            dict.fromkeys(local.source_id for local in run.local_bindings)
        ),
        "source_ids_by_handle": {
            binding.handle_id: [local.source_id]
            for binding, local in zip(
                run.contribution.bindings, run.local_bindings, strict=True
            )
        },
        "specialist_receipt_sha256": run.specialist_receipt_sha256,
        "typed_contribution": run.contribution.projection(),
    }
    return {**body, "method_receipt_sha256": identity_sha256(body)}


def _parent_method_projection(
    parent: ProtectedParentContributionSet,
) -> dict[str, Any]:
    source_ids_by_handle = {
        row.handle_id: list(row.source_ids)
        for row in parent.audit.source_provenance
    }
    body = {
        "accepted_typed_item_count": sum(
            len(row.parsed.accepted_items) for row in parent.contributions
        ),
        "component_mechanism_ids": [
            row.mechanism_id for row in parent.contributions
        ],
        "local_binding_count": sum(
            len(row.bindings) for row in parent.contributions
        ),
        "local_bindings": [
            provenance.cloned_binding.projection()
            for provenance in parent.audit.source_provenance
        ],
        "mechanism_id": PROTECTED_PARENT_MECHANISM_ID,
        "new_provider_calls": 0,
        "protected_parent": parent.projection(),
        "protected_parent_audit": parent.audit.projection(),
        "retained_transformer_token_state_bytes": 0,
        "source_ids": list(parent.source_ids),
        "source_ids_by_handle": source_ids_by_handle,
        "typed_contributions": [
            row.projection() for row in parent.contributions
        ],
    }
    return {**body, "method_receipt_sha256": identity_sha256(body)}


def _usable_typed_item(item: Any, spec: TypedOperatorSpec) -> bool:
    return bool(
        item.included
        and not item.content_conflict
        and item.status is not EvidenceStatus.CANCELLED
        and (
            item.status is not EvidenceStatus.PROPOSED
            or spec.include_proposed
        )
    )


def _bounded_lane_budget(
    lane_id: str,
    contributions: Sequence[TypedEvidenceContribution],
    spec: TypedOperatorSpec,
) -> LaneBudget:
    contributions = tuple(contributions)
    _require(bool(contributions), "specialist lane cannot be empty")
    bindings = tuple(
        binding for contribution in contributions for binding in contribution.bindings
    )
    minimum_items = tuple(
        next(
            (
                item
                for item in contribution.parsed.accepted_items
                if _usable_typed_item(item, spec)
            ),
            None,
        )
        for contribution in contributions
    )
    minimum_items = tuple(item for item in minimum_items if item is not None)
    minimum_proxy = lane_content_token_proxy(minimum_items, bindings)
    # Allocation may reorder items by local operator priority.  BPE token
    # counts are mildly order-sensitive at JSON boundaries, so the declared
    # cap must remain the nominal envelope even when the source-order proxy is
    # a few tokens smaller.
    cap = max(NOMINAL_LANE_TOKEN_CAP[lane_id], minimum_proxy)
    return LaneBudget(
        lane_id=lane_id,
        final_content_token_cap=cap,
        preparation=CallBudget(
            HARD_COMPLETE_CHAT_TOKEN_CAP,
            OUTPUT_TOKEN_RESERVE,
            0,
        ),
    )


def _compact_absence_advisory(
    result: TemporalInsufficiencyResult,
    candidate_handle_map: Mapping[str, str],
) -> dict[str, Any] | None:
    if SpecialistRoute.NUMERIC_SLOT_INSUFFICIENCY not in result.routes:
        return None
    certificate = result.absence_certificate
    if not certificate.applicable:
        return None
    return {
        "applicable": certificate.applicable,
        "every_exact_entity_posting_scanned": (
            certificate.every_exact_entity_posting_scanned
        ),
        "every_scoped_source_row_scanned": (
            certificate.every_scoped_source_row_scanned
        ),
        "may_conclude_operator_insufficient": (
            certificate.may_conclude_operator_insufficient
        ),
        "physical_content_rows_scanned": certificate.physical_content_rows_scanned,
        "physical_sentence_windows_scanned": (
            certificate.physical_sentence_windows_scanned
        ),
        "provider_instruction": certificate.provider_instruction,
        "scope_definition": certificate.scope_definition,
        "scoped_content_row_count": certificate.scoped_content_row_count,
        "scoped_source_count": certificate.scoped_source_count,
        "semantic_absence_may_be_inferred": False,
        "slot_coverage": [
            {
                "entity_assertion_source_count": row.entity_assertion_source_count,
                "entity_assertion_window_count": row.entity_assertion_window_count,
                "exact_entity_terms": list(row.exact_entity_terms),
                "explicit_numeric_assertion_source_count": (
                    row.explicit_numeric_assertion_source_count
                ),
                "explicit_numeric_assertion_window_count": (
                    row.explicit_numeric_assertion_window_count
                ),
                "explicit_numeric_operand_missing": (
                    row.explicit_numeric_operand_missing
                ),
                "scope_has_grounded_predicate_assertion": (
                    row.scope_has_grounded_predicate_assertion
                ),
                "selected_supporting_handle_ids": [
                    candidate_handle_map[candidate_id]
                    for candidate_id in row.selected_supporting_candidate_ids
                    if candidate_id in candidate_handle_map
                ],
                "slot_label": row.slot_label,
            }
            for row in certificate.slot_coverage
        ],
    }


def _specialist_advisories(
    runs: Sequence[_SpecialistRun],
    allowed_handle_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Project specialist semantics only over handles that survived fitting."""

    allowed = set(allowed_handle_ids)
    rows: list[dict[str, Any]] = []
    for run in runs:
        handles = {
            local.candidate_id: binding.handle_id
            for binding, local in zip(
                run.contribution.bindings, run.local_bindings, strict=True
            )
            if binding.handle_id in allowed
        }
        if type(run.result) is NumericOperandClosureResult:
            if any(
                group.operation_mode not in {"count", "sum"}
                for group in run.result.operand_groups
            ):
                # The scoped deterministic proof currently supports only
                # additive reductions.  Comparisons remain in the ordinary
                # parent evidence, but must not be mislabeled as a scoped
                # numeric proof.
                continue
            groups: list[dict[str, Any]] = []
            owned_candidates: set[str] = set()
            candidate_ownership_repeats = False
            group_by_candidate = {
                local.candidate_id: binding.source_group_handle
                for binding, local in zip(
                    run.contribution.bindings,
                    run.local_bindings,
                    strict=True,
                )
                if binding.handle_id in allowed
            }
            all_operand_groups_represented = True
            for group in run.result.operand_groups:
                candidate_ids = tuple(
                    candidate_id
                    for candidate_id in group.candidate_ids
                    if candidate_id in handles
                )
                if not candidate_ids:
                    # A deterministic reduction is valid only over the exact
                    # operand-group population sealed by the specialist.  If
                    # fitting removes every witness for even one group, a
                    # surviving subset must not be re-described as the whole
                    # reduction universe.
                    all_operand_groups_represented = False
                    break
                # The scoped completion proof cites each opaque evidence
                # handle at most once.  A source window can contain several
                # numeric mentions and therefore appear in several closure
                # groups, but projecting that shape would let one handle
                # authorize multiple operands.  Keep the specialist
                # fail-closed and omit the whole numeric advisory; the caller
                # will preserve the sealed parent instead of publishing an
                # incomplete or double-countable reduction.
                if set(candidate_ids) & owned_candidates:
                    candidate_ownership_repeats = True
                    break
                owned_candidates.update(candidate_ids)
                projection = {
                    "action_class": group.action_class,
                    "entity_key": group.entity_key,
                    "handle_ids": [
                        handles[candidate_id] for candidate_id in candidate_ids
                    ],
                    "operand_values": list(group.operand_values),
                    "operation_mode": group.operation_mode,
                    "source_group_handles": list(
                        dict.fromkeys(
                            group_by_candidate[candidate_id]
                            for candidate_id in candidate_ids
                        )
                    ),
                    "value_basis": group.value_basis,
                }
                groups.append(projection)
            if (
                candidate_ownership_repeats
                or not all_operand_groups_represented
                or not handles
                or len(groups) != len(run.result.operand_groups)
            ):
                continue
            rows.append(
                {
                    "format": SPECIALIST_ADVISORY_FORMAT,
                    "handle_ids": list(handles.values()),
                    "mechanism_id": run.mechanism_id,
                    "operand_groups": groups,
                    "purpose": "group distinct numeric event operands before reduction",
                }
            )
        elif type(run.result) is TemporalInsufficiencyResult:
            if not handles:
                # An absence certificate without a surviving citation cannot
                # satisfy the scoped compiler's nonempty, exact provenance
                # boundary.  Preserve the parent instead of emitting a
                # handle-free advisory.
                continue
            bundle: dict[str, Any] | None = None
            if run.result.temporal_bundle is not None:
                source_bundle = run.result.temporal_bundle
                ordered = tuple(
                    candidate_id
                    for candidate_id in source_bundle.ordered_candidate_ids
                    if candidate_id in handles
                )
                if ordered:
                    winner = (
                        source_bundle.winner_candidate_id
                        if source_bundle.winner_candidate_id in handles
                        else None
                    )
                    predecessor = (
                        source_bundle.predecessor_candidate_id
                        if source_bundle.predecessor_candidate_id in handles
                        else None
                    )
                    bundle = {
                        "ordered_handle_ids": [handles[value] for value in ordered],
                        "original_population_count": source_bundle.population_count,
                        "predecessor_handle_id": (
                            None if predecessor is None else handles[predecessor]
                        ),
                        "query_time": source_bundle.query_time,
                        "requested_cardinality": source_bundle.requested_cardinality,
                        "route": source_bundle.route,
                        "target_date": source_bundle.target_date,
                        "terminal_selection_truncated": (
                            len(ordered) < len(source_bundle.ordered_candidate_ids)
                            or source_bundle.truncated
                        ),
                        "winner_handle_id": (
                            None if winner is None else handles[winner]
                        ),
                    }
            absence = _compact_absence_advisory(run.result, handles)
            if bundle is None and absence is None:
                continue
            rows.append(
                {
                    "absence_certificate": absence,
                    "format": SPECIALIST_ADVISORY_FORMAT,
                    "handle_ids": list(handles.values()),
                    "mechanism_id": run.mechanism_id,
                    "purpose": "apply participant/entity constraints before temporal selection",
                    "temporal_bundle": bundle,
                }
            )
        elif type(run.result) is ProfilePreferenceResult:
            if not handles:
                continue
            rows.append(
                {
                    "format": SPECIALIST_ADVISORY_FORMAT,
                    "handle_ids": list(handles.values()),
                    "mechanism_id": run.mechanism_id,
                    "purpose": "personalize from one coherent first-person preference cluster",
                }
            )
    return rows


def _terminal_projection(
    *,
    provider_input: Mapping[str, Any],
    specialist_advisories: Sequence[Mapping[str, Any]],
    fitted_prompt_receipt_sha256: str,
    message_renderer: Callable[
        [Mapping[str, Any]], Sequence[Mapping[str, str]]
    ] = render_final_messages,
    message_renderer_format: str | None = None,
    prompt_envelope_renderer: Callable[[Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Seal and recount the actual terminal provider payload."""

    fitted_receipt = require_sha256(
        fitted_prompt_receipt_sha256, "fitted terminal prompt receipt"
    )
    advisories = [dict(row) for row in specialist_advisories]
    terminal_input = {
        **dict(provider_input),
        "specialist_advisories": advisories,
    }
    prompt_envelope_receipt: str | None = None
    if prompt_envelope_renderer is None:
        messages = tuple(dict(row) for row in message_renderer(terminal_input))
    else:
        envelope = prompt_envelope_renderer(terminal_input)
        messages = tuple(dict(row) for row in envelope.messages)
        prompt_envelope_receipt = require_sha256(
            envelope.receipt_sha256,
            "specialist prompt envelope",
        )
        envelope_projection = envelope.projection()
        _require(
            type(envelope_projection) is dict
            and envelope_projection.get("provider_input_sha256")
            == identity_sha256(terminal_input)
            and envelope_projection.get("specialist_advisories_sha256")
            == identity_sha256(advisories)
            and envelope_projection.get("messages_sha256")
            == identity_sha256(list(messages)),
            "specialist prompt envelope diverged from terminal input",
        )
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    _require(
        prompt_tokens + OUTPUT_TOKEN_RESERVE <= HARD_COMPLETE_CHAT_TOKEN_CAP,
        "specialist advisory escaped the hard 8k answer envelope",
    )
    advisory_sha = identity_sha256(advisories)
    messages_sha = identity_sha256(list(messages))
    receipt_body = {
        "fitted_prompt_receipt_sha256": fitted_receipt,
        "messages_sha256": messages_sha,
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "prompt_token_proxy": prompt_tokens,
        "provider_input_sha256": identity_sha256(terminal_input),
        "specialist_advisories_sha256": advisory_sha,
    }
    if message_renderer_format is not None:
        receipt_body["message_renderer_format"] = require_text(
            message_renderer_format, "specialist message renderer format"
        )
    if prompt_envelope_receipt is not None:
        receipt_body["specialist_prompt_envelope_receipt_sha256"] = (
            prompt_envelope_receipt
        )
    result = {
        "fitted_prompt_receipt_sha256": fitted_receipt,
        "full_chat_plus_output_tokens": prompt_tokens + OUTPUT_TOKEN_RESERVE,
        "hard_prompt_token_cap": HARD_COMPLETE_CHAT_TOKEN_CAP,
        "messages_sha256": messages_sha,
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "prompt_token_proxy": prompt_tokens,
        "provider_input": terminal_input,
        "provider_prompt_count": 0,
        "retained_transformer_token_state_bytes": 0,
        "specialist_advisories_sha256": advisory_sha,
        "terminal_prompt_receipt_sha256": identity_sha256(receipt_body),
    }
    if message_renderer_format is not None:
        result["message_renderer_format"] = receipt_body[
            "message_renderer_format"
        ]
    if prompt_envelope_receipt is not None:
        result["specialist_prompt_envelope_receipt_sha256"] = (
            prompt_envelope_receipt
        )
    return result


def _composed_question(
    *,
    ordinal: int,
    index: Any,
    composition_row: Mapping[str, Any],
    parent_composition_artifact_sha256: str,
    frozen_row: Mapping[str, Any],
    parent_prediction_override: str | None = None,
    terminal_message_renderer: Callable[
        [Mapping[str, Any]], Sequence[Mapping[str, str]]
    ] = render_final_messages,
    terminal_message_renderer_format: str | None = None,
    terminal_prompt_envelope_renderer: (
        Callable[[Mapping[str, Any]], Any] | None
    ) = None,
) -> dict[str, Any]:
    dated_question, parent_prediction, question_id = _question_inputs(
        composition_row
    )
    if parent_prediction_override is not None:
        parent_prediction = require_text(
            parent_prediction_override,
            "specialist parent prediction override",
        )
    _require(
        frozen_row.get("ordinal") == ordinal
        and frozen_row.get("question_id") == question_id
        and frozen_row.get("namespace_id") == index.cache.namespace_id,
        "specialist question/index binding changed",
    )
    route = route_question(dated_question)
    spec = compile_typed_operator_spec(dated_question)
    parent = rehydrate_protected_parent_contributions(
        composition_row,
        spec,
        parent_composition_artifact_sha256,
    )
    runs = _run_specialists(index, dated_question)
    methods = [
        _parent_method_projection(parent),
        *(_method_projection(run) for run in runs),
    ]
    specialist_contributions = tuple(
        run.contribution
        for run in runs
        if run.contribution.parsed.accepted_items
    )
    contributions = (*parent.contributions, *specialist_contributions)
    lane_groups: dict[str, tuple[TypedEvidenceContribution, ...]] = {
        PROTECTED_PARENT_LANE_ID: parent.contributions,
    }
    for contribution in specialist_contributions:
        lane_groups[SPECIALIST_LANE_ID[contribution.mechanism_id]] = (
            contribution,
        )
    lane_budgets = tuple(
        _bounded_lane_budget(lane_id, values, spec)
        for lane_id, values in lane_groups.items()
    )
    lane_by_mechanism = {
        contribution.mechanism_id: lane_id
        for lane_id, values in lane_groups.items()
        for contribution in values
    }
    dedup_priorities = {
        contribution.mechanism_id: (
            100
            if contribution in parent.contributions
            else MECHANISM_PRIORITY[contribution.mechanism_id]
        )
        for contribution in contributions
    }
    fair_priorities = {
        contribution.mechanism_id: (
            0
            if contribution in parent.contributions
            else MECHANISM_PRIORITY[contribution.mechanism_id]
        )
        for contribution in contributions
    }
    known_handles = {
        binding.handle_id
        for contribution in contributions
        for binding in contribution.bindings
    }
    exact_span_keys = dict(parent.exact_span_keys_by_handle)
    exact_span_keys.update(
        {
            handle: value
            for handle, value in _span_keys(runs).items()
            if handle in known_handles
        }
    )
    local_priorities = dict(parent.local_selection_priority_by_handle)
    local_priorities.update(
        {
            handle: value
            for handle, value in _specialist_local_priorities(runs).items()
            if handle in known_handles
        }
    )
    composition = compose_additive_typed_evidence(
        spec,
        contributions,
        lane_budgets=lane_budgets,
        lane_by_mechanism=lane_by_mechanism,
        dedup_owner_priority_by_mechanism=dedup_priorities,
        exact_span_keys_by_handle=exact_span_keys,
        local_selection_priority_by_handle=local_priorities,
        fair_merge_priority_by_mechanism=fair_priorities,
        provider_payload_mode=ProviderPayloadMode.COMPACT_FINAL_V2,
    )
    fitted = fit_typed_final_prompt(
        dated_question=dated_question,
        parent_prediction=parent_prediction,
        packet=composition.packet,
        mechanism_by_handle=composition.mechanism_by_handle,
        local_retention_priority_by_handle=(
            composition.retained_local_priority_by_handle
        ),
        minimum_usable_items_per_mechanism=1,
        protected_item_receipt_sha256s=(
            composition.protected_item_receipt_sha256s
        ),
        protection_source_receipt_sha256=(
            composition.protection_source_receipt_sha256
        ),
    )
    terminal_projection = _terminal_projection(
        provider_input=fitted.provider_input,
        specialist_advisories=_specialist_advisories(
            runs, fitted.allowed_handle_ids
        ),
        fitted_prompt_receipt_sha256=fitted.receipt_sha256,
        message_renderer=terminal_message_renderer,
        message_renderer_format=terminal_message_renderer_format,
        prompt_envelope_renderer=terminal_prompt_envelope_renderer,
    )
    body = {
        "additive_composition": composition.projection(),
        "additive_composition_local_audit": {
            "dropped_binding_projections": [
                dict(value) for value in composition.dropped_binding_projections
            ],
            "fair_merge": dict(composition.fair_merge_audit),
            "minimum_allocation": composition.minimum_allocation.projection(),
            "post_selection_dedup": dict(
                composition.post_selection_dedup_audit
            ),
            "shared_lane_surplus_fill": dict(composition.surplus_fill_audit),
        },
        "applicable_specialist_ids": list(applicable_specialist_ids(dated_question)),
        "dated_question_sha256": composition_row.get("dated_question_sha256"),
        "fitted_typed_prompt": fitted.projection(),
        "lane_budget_policy": {
            "lane_budgets": [
                {
                    "final_content_token_cap": row.final_content_token_cap,
                    "lane_id": row.lane_id,
                    "preparation_call_cap": row.preparation.call_cap,
                }
                for row in lane_budgets
            ],
            "lane_by_mechanism": lane_by_mechanism,
            "nominal_lane_token_caps": dict(NOMINAL_LANE_TOKEN_CAP),
            "separate_non_borrowable_minima_then_shared_surplus": True,
        },
        "methods": methods,
        "namespace_id": index.cache.namespace_id,
        "new_provider_calls": 0,
        "ordinal": ordinal,
        "question_id": question_id,
        "question_sha256": composition_row.get("question_sha256"),
        "retained_transformer_token_state_bytes": 0,
        "route": route.identity_payload(),
        "terminal_prompt": terminal_projection,
    }
    assert_gold_blind(body, path="reduced_specialist_question")
    return {**body, "question_receipt_sha256": identity_sha256(body)}


def build_construction(
    args: argparse.Namespace,
    *,
    construction_format: str = CONSTRUCTION_FORMAT,
    terminal_message_renderer: Callable[
        [Mapping[str, Any]], Sequence[Mapping[str, str]]
    ] = render_final_messages,
    terminal_message_renderer_format: str | None = None,
    terminal_prompt_envelope_renderer: (
        Callable[[Mapping[str, Any]], Any] | None
    ) = None,
) -> dict[str, Any]:
    frozen = _frozen_exact10(Path(args.frozen_reduced))
    frozen_rows = _frozen_question_rows(frozen)
    frozen_by_ordinal = {int(row["ordinal"]): row for row in frozen_rows}
    groups = reduced_cli._namespace_ordinal_groups(frozen_rows)  # noqa: SLF001
    composition, closure, composition_rows, _closure_rows = (
        reduced_cli._read_source_artifacts(Path(args.source_root))  # noqa: SLF001
    )
    questions_by_ordinal: dict[int, dict[str, Any]] = {}
    lifecycle: list[dict[str, Any]] = []
    for namespace_id, ordinals in groups:
        _target_rows, index, receipt = reduced_cli._scoped_resident_index(  # noqa: SLF001
            args,
            namespace_id=namespace_id,
            ordinals=ordinals,
            composition_rows=composition_rows,
            closure=closure,
        )
        for ordinal in ordinals:
            questions_by_ordinal[ordinal] = _composed_question(
                ordinal=ordinal,
                index=index,
                composition_row=composition_rows[ordinal],
                parent_composition_artifact_sha256=composition.sha256,
                frozen_row=frozen_by_ordinal[ordinal],
                terminal_message_renderer=terminal_message_renderer,
                terminal_message_renderer_format=(
                    terminal_message_renderer_format
                ),
                terminal_prompt_envelope_renderer=(
                    terminal_prompt_envelope_renderer
                ),
            )
        lifecycle.append(receipt)
        del index
        gc.collect()
    _require(
        set(questions_by_ordinal) == set(TARGET_ORDINALS),
        "streamed specialist construction lost a question",
    )
    questions = [questions_by_ordinal[value] for value in TARGET_ORDINALS]
    payload: dict[str, Any] = {
        "bindings": {
            "frozen_exact10_construction_sha256": frozen.sha256,
            "parent_composition_sha256": composition.sha256,
            "parent_full_store_input_sha256": closure.sha256,
        },
        "construction_is_posthoc_outcome_conditioned": True,
        "format": require_text(
            construction_format, "specialist construction format"
        ),
        "gold_loaded": False,
        "hard_complete_chat_token_cap": HARD_COMPLETE_CHAT_TOKEN_CAP,
        "new_provider_calls": 0,
        "ordinals": list(TARGET_ORDINALS),
        "question_count": QUESTION_COUNT,
        "questions": questions,
        "resident_index_lifecycle": {
            "database_read_passes_per_used_namespace": 1,
            "maximum_simultaneous_namespace_indexes": 1,
            "receipts": sorted(lifecycle, key=lambda row: row["namespace_id"]),
            "unique_namespace_count": len(lifecycle),
        },
        "retained_transformer_token_state_bytes": 0,
        "selection_and_routing_frozen_before_target_plan_load": True,
        "target_labels_loaded": False,
        "target_plan_loaded": False,
    }
    assert_gold_blind(payload, path="reduced_specialist_construction")
    payload["construction_identity_sha256"] = identity_sha256(payload)
    return payload


def _validate_construction(
    artifact: SealedArtifact,
    *,
    construction_format: str = CONSTRUCTION_FORMAT,
) -> tuple[dict[str, Any], ...]:
    payload = artifact.payload
    rows = _exact_list(payload.get("questions"), "specialist questions")
    _require(
        payload.get("format")
        == require_text(construction_format, "specialist construction format")
        and tuple(payload.get("ordinals", ())) == TARGET_ORDINALS
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("gold_loaded") is False
        and payload.get("target_labels_loaded") is False
        and payload.get("target_plan_loaded") is False
        and payload.get("new_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and len(rows) == QUESTION_COUNT,
        "specialist construction boundary changed",
    )
    unsigned = dict(payload)
    declared = require_sha256(
        unsigned.pop("construction_identity_sha256", None),
        "specialist construction identity",
    )
    _require(identity_sha256(unsigned) == declared, "construction identity changed")
    result = tuple(_exact_dict(row, "specialist question") for row in rows)
    _require(
        tuple(row.get("ordinal") for row in result) == TARGET_ORDINALS,
        "specialist question order changed",
    )
    assert_gold_blind(payload, path="validated_reduced_specialist_construction")
    return result


def _binding_aliases(method: Mapping[str, Any], question_id: str) -> set[str]:
    sources = [
        require_text(value, "method local source")
        for value in _exact_list(method.get("source_ids"), "method source IDs")
    ]
    return reduced_cli._source_aliases(sources, question_id)  # noqa: SLF001


def _terminal_binding_aliases(
    method: Mapping[str, Any],
    question_id: str,
    allowed_handle_ids: set[str],
) -> set[str]:
    raw = _exact_dict(
        method.get("source_ids_by_handle"), "method sources by handle"
    )
    sources: list[str] = []
    for handle, values in raw.items():
        require_text(handle, "method source handle")
        if handle not in allowed_handle_ids:
            continue
        sources.extend(
            require_text(value, "terminal method source")
            for value in _exact_list(values, "terminal method sources")
        )
    return reduced_cli._source_aliases(sources, question_id)  # noqa: SLF001


def build_target_audit(
    construction: SealedArtifact,
    target_plan: Mapping[str, Any],
    *,
    target_plan_file_sha256: str,
    construction_format: str = CONSTRUCTION_FORMAT,
    audit_format: str = AUDIT_FORMAT,
) -> dict[str, Any]:
    questions = _validate_construction(
        construction, construction_format=construction_format
    )
    audited: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "question_count": 0,
            "source_target_count": 0,
            "source_target_hits": 0,
            "source_set_complete_questions": 0,
            "terminal_source_target_hits": 0,
            "terminal_source_set_complete_questions": 0,
        }
    )
    for row in questions:
        ordinal = int(row["ordinal"])
        question_id = require_text(row.get("question_id"), "audit question")
        expected, relation_required, coverage_required = (
            reduced_cli._expected_sources(  # noqa: SLF001
                target_plan, ordinal, question_id
            )
        )
        methods = tuple(
            _exact_dict(value, "specialist method")
            for value in _exact_list(row.get("methods"), "specialist methods")
        )
        method_rows: list[dict[str, Any]] = []
        union_aliases: set[str] = set()
        terminal_union_aliases: set[str] = set()
        fitted = _exact_dict(
            row.get("fitted_typed_prompt"), "audited fitted prompt"
        )
        allowed_handles = set(
            require_text(value, "audited allowed handle")
            for value in _exact_list(
                fitted.get("allowed_handle_ids"), "audited allowed handles"
            )
        )
        for method in methods:
            mechanism_id = require_text(
                method.get("mechanism_id"), "audit mechanism"
            )
            aliases = _binding_aliases(method, question_id)
            terminal_aliases = _terminal_binding_aliases(
                method, question_id, allowed_handles
            )
            union_aliases.update(aliases)
            terminal_union_aliases.update(terminal_aliases)
            hits = [source for source in expected if source in aliases]
            terminal_hits = [
                source for source in expected if source in terminal_aliases
            ]
            complete = len(hits) == len(expected)
            terminal_complete = len(terminal_hits) == len(expected)
            method_rows.append(
                {
                    "mechanism_id": mechanism_id,
                    "reached_source_ids": hits,
                    "source_set_complete": complete,
                    "source_target_count": len(expected),
                    "source_target_hits": len(hits),
                    "terminal_reached_source_ids": terminal_hits,
                    "terminal_source_set_complete": terminal_complete,
                    "terminal_source_target_hits": len(terminal_hits),
                }
            )
            summary = aggregate[mechanism_id]
            summary["question_count"] += 1
            summary["source_target_count"] += len(expected)
            summary["source_target_hits"] += len(hits)
            summary["source_set_complete_questions"] += int(complete)
            summary["terminal_source_target_hits"] += len(terminal_hits)
            summary["terminal_source_set_complete_questions"] += int(
                terminal_complete
            )
        union_hits = [source for source in expected if source in union_aliases]
        terminal_union_hits = [
            source for source in expected if source in terminal_union_aliases
        ]
        terminal = row.get("terminal_prompt")
        audited.append(
            {
                "coverage_check_required": coverage_required,
                "expected_source_ids": list(expected),
                "methods": method_rows,
                "ordinal": ordinal,
                "provider_ready": type(terminal) is dict,
                "question_id": question_id,
                "relation_required": relation_required,
                "terminal_full_chat_plus_output_tokens": (
                    None
                    if type(terminal) is not dict
                    else terminal.get("full_chat_plus_output_tokens")
                ),
                "terminal_union_reached_source_ids": terminal_union_hits,
                "terminal_union_source_set_complete": (
                    len(terminal_union_hits) == len(expected)
                ),
                "terminal_union_source_target_hits": len(terminal_union_hits),
                "union_reached_source_ids": union_hits,
                "union_source_set_complete": len(union_hits) == len(expected),
                "union_source_target_hits": len(union_hits),
            }
        )
    payload: dict[str, Any] = {
        "bindings": {
            "construction_artifact_sha256": construction.sha256,
            "construction_identity_sha256": construction.payload[
                "construction_identity_sha256"
            ],
            "target_plan_file_sha256": target_plan_file_sha256,
            "target_plan_identity_sha256": target_plan["plan_sha256"],
        },
        "construction_verified_before_target_plan_load": True,
        "format": require_text(audit_format, "specialist audit format"),
        "method_summary": dict(sorted(aggregate.items())),
        "new_provider_calls": 0,
        "ordinals": list(TARGET_ORDINALS),
        "posthoc_target_labels_loaded": True,
        "question_count": QUESTION_COUNT,
        "questions": audited,
        "retained_transformer_token_state_bytes": 0,
        "runtime_use_forbidden": True,
        "terminal_union_source_set_complete_questions": sum(
            row["terminal_union_source_set_complete"] for row in audited
        ),
        "terminal_union_source_target_hits": sum(
            row["terminal_union_source_target_hits"] for row in audited
        ),
        "union_source_set_complete_questions": sum(
            row["union_source_set_complete"] for row in audited
        ),
        "union_source_target_count": sum(
            len(row["expected_source_ids"]) for row in audited
        ),
        "union_source_target_hits": sum(
            row["union_source_target_hits"] for row in audited
        ),
    }
    payload["audit_identity_sha256"] = identity_sha256(payload)
    return payload


def run_construct(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_construction(args)
    artifact, created = publish_sealed_json(
        Path(args.output_root) / CONSTRUCTION_NAME, payload
    )
    return {
        "construction_sha256": artifact.sha256,
        "created": created,
        "new_provider_calls": 0,
        "question_count": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    construction = read_sealed_json(Path(args.construction))
    _require(
        construction.sha256
        == require_sha256(
            args.expected_construction_sha256,
            "expected specialist construction",
        ),
        "specialist construction artifact changed",
    )
    _validate_construction(construction)
    target_plan, plan_sha = reduced_cli._read_target_plan(  # noqa: SLF001
        Path(args.target_plan)
    )
    payload = build_target_audit(
        construction,
        target_plan,
        target_plan_file_sha256=plan_sha,
    )
    artifact, created = publish_sealed_json(Path(args.output), payload)
    return {
        "audit_sha256": artifact.sha256,
        "created": created,
        "new_provider_calls": 0,
        "question_count": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "terminal_union_source_set_complete_questions": payload[
            "terminal_union_source_set_complete_questions"
        ],
        "terminal_union_source_target_hits": payload[
            "terminal_union_source_target_hits"
        ],
        "union_source_set_complete_questions": payload[
            "union_source_set_complete_questions"
        ],
        "union_source_target_count": payload["union_source_target_count"],
        "union_source_target_hits": payload["union_source_target_hits"],
    }


def _add_store_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--retrieval", type=Path, default=reduced_cli.guided_scan_cli.DEFAULT_RETRIEVAL
    )
    parser.add_argument(
        "--store-root", type=Path, default=reduced_cli.guided_scan_cli.DEFAULT_STORE_ROOT
    )
    parser.add_argument(
        "--query-parent-output-root",
        type=Path,
        default=reduced_cli.guided_scan_cli.DEFAULT_PARENT_OUTPUT,
    )
    parser.add_argument(
        "--expected-retrieval-sha256",
        default=reduced_cli.guided_scan_cli.EXPECTED_RETRIEVAL_SHA256,
    )
    parser.add_argument(
        "--expected-query-parent-preflight-sha256",
        default=reduced_cli.guided_scan_cli.EXPECTED_PARENT_PREFLIGHT_SHA256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    construct = commands.add_parser("construct")
    construct.add_argument(
        "--frozen-reduced", type=Path, default=DEFAULT_FROZEN_REDUCED
    )
    construct.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    construct.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    _add_store_args(construct)
    audit = commands.add_parser("audit")
    audit.add_argument(
        "--construction",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / CONSTRUCTION_NAME,
    )
    audit.add_argument("--expected-construction-sha256", required=True)
    audit.add_argument("--target-plan", type=Path, default=DEFAULT_TARGET_PLAN)
    audit.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / AUDIT_NAME)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_construct(args) if args.command == "construct" else run_audit(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_FORMAT",
    "AUDIT_NAME",
    "CONSTRUCTION_FORMAT",
    "CONSTRUCTION_NAME",
    "ReducedSpecialistAssayError",
    "SPECIALIST_ADVISORY_FORMAT",
    "applicable_specialist_ids",
    "build_target_audit",
    "main",
]
