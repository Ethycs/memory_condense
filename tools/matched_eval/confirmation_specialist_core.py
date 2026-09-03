"""Pure specialist construction core used by confirmation prediction.

This is the question-local mechanism surface of the historical specialist
assay without its validation population, artifact readers, audit command, or
CLI imports.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from tools._routed_repair_routing import RoutedRepairStyle, route_question
from tools.matched_eval.contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.numeric_operand_specialist import (
    MECHANISM_ID as NUMERIC_MECHANISM_ID,
    NumericOperandClosureResult,
    adapt_numeric_operand_closure_to_typed_contribution,
    scan_numeric_operand_closure,
)
from tools.matched_eval.profile_preference_specialist import (
    MECHANISM_ID as PROFILE_MECHANISM_ID,
    ProfilePreferenceResult,
    adapt_profile_preference_to_typed_contribution,
    select_profile_preference_evidence,
)
from tools.matched_eval.protected_parent_contribution import (
    MECHANISM_ID as PROTECTED_PARENT_MECHANISM_ID,
    ProtectedParentContributionSet,
    rehydrate_protected_parent_contributions,
)
from tools.matched_eval.prompt_tick_contracts import CallBudget, LaneBudget
from tools.matched_eval.specialist_scoped_completion import SPECIALIST_ADVISORY_FORMAT
from tools.matched_eval.temporal_insufficiency_specialist import (
    BundleRole,
    MECHANISM_ID as TEMPORAL_MECHANISM_ID,
    SpecialistRoute,
    TemporalInsufficiencyResult,
    adapt_temporal_insufficiency_to_typed_contribution,
    scan_temporal_insufficiency_specialist,
)
from tools.matched_eval.typed_additive_composer import (
    LEGACY_COMPOSITION_MODE,
    compose_additive_typed_evidence,
)
from tools.matched_eval.typed_lane_allocator import lane_content_token_proxy
from tools.matched_eval.typed_memory_final_arm import (
    LOCAL_RETENTION_PRIORITY_WIDTH,
    fit_typed_final_prompt,
    render_final_messages,
)
from tools.matched_eval.typed_operator_adapter import (
    EvidenceStatus,
    ProviderPayloadMode,
    TypedEvidenceContribution,
)
from tools.matched_eval.typed_operator_spec import (
    TemporalMode,
    TypedOperatorSpec,
    compile_typed_operator_spec,
)

HARD_COMPLETE_CHAT_TOKEN_CAP = 8_000
OUTPUT_TOKEN_RESERVE = 768
MECHANISM_HANDLE_START = {
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
    """A pure specialist route, selection, composition, or fit changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ReducedSpecialistAssayError(message)


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact array")
    return value  # type: ignore[return-value]

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
    typed_composition_mode: str = LEGACY_COMPOSITION_MODE,
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
        provider_payload_mode=(
            ProviderPayloadMode.COMPACT_FINAL
            if typed_composition_mode == LEGACY_COMPOSITION_MODE
            else ProviderPayloadMode.COMPACT_FINAL_V2
        ),
        composition_mode=typed_composition_mode,
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




TYPED_ANSWER_FORMAT = "memory-condense-reduced-specialist-terra-answer-v2"


def handle_groups(
    provider_input: Mapping[str, Any],
    allowed_handle_ids: Sequence[str],
) -> dict[str, str]:
    """Authenticate the fitted opaque handle-to-group population."""

    typed = provider_input.get("typed_evidence")
    _require(type(typed) is dict, "terminal typed evidence is missing")
    assert type(typed) is dict
    handles = typed.get("handles")
    _require(type(handles) is list, "terminal typed handles changed type")
    groups: dict[str, str] = {}
    for raw in handles:
        _require(
            type(raw) is dict
            and set(raw) >= {"handle_id", "group_handle"}
            and type(raw.get("handle_id")) is str
            and bool(raw["handle_id"])
            and type(raw.get("group_handle")) is str
            and bool(raw["group_handle"]),
            "terminal handle/group row changed schema",
        )
        assert type(raw) is dict
        handle = str(raw["handle_id"])
        _require(handle not in groups, "terminal handle/group rows repeat")
        groups[handle] = str(raw["group_handle"])
    _require(
        tuple(allowed_handle_ids)
        and len(tuple(allowed_handle_ids)) == len(set(allowed_handle_ids))
        and set(groups) == set(allowed_handle_ids),
        "fitted allowed handles differ from the terminal handle/group bindings",
    )
    return groups


__all__ = [
    "TYPED_ANSWER_FORMAT",
    "_composed_question",
    "_question_inputs",
    "applicable_specialist_ids",
    "handle_groups",
    "route_question",
]
