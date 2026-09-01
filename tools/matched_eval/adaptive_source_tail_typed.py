"""Provider-free seam from the sealed source-tail wave to typed evidence.

The tail mapper has already validated exact quotes and performed neither
post-map deduplication nor final fact packing.  This module completes those two
local stages, retains their receipts, and exposes only opaque H/G handles to
the common typed layer.  It never opens a memory store, loads gold, or invokes
a model/provider.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from memory_condense.domain.discourse import quote_sha256

from tools import run_locked_adaptive_source_map as source_cli
from tools._routed_repair_routing import RoutedRepairStyle

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .source_history_fact_union import (
    DirectEvidenceExclusion,
    DirectEvidenceRef,
    FactUnionEnvelope,
    LaneAdmission,
    PostMapFactUnion,
    UnionFact,
    build_post_map_fact_union,
    pack_fact_union_envelope,
)
from .source_history_mapper_live import SourceMapperMaterialization
from .typed_operator_adapter import (
    EvidenceHandleBinding,
    EvidenceOrigin,
    EvidenceStatus,
    FrontierMode,
    NumericRole,
    ParsedTypedItems,
    ProvenanceGrade,
    TypedEvidenceContribution,
    TypedItemKind,
    ValueAuthority,
    conservative_numeric_value,
    parse_typed_items,
)
from .typed_operator_spec import TypedOperatorSpec


ROW_FORMAT = "memory-condense-adaptive-source-tail-typed-row-v1"
MECHANISM_ID = "adaptive_source_tail_wave_2_recovery_v1"
DIRECT_POINTER_MECHANISM_ID = (
    "adaptive_source_tail_wave_2_recovery_v1_direct_pointer"
)
_MAX_OPAQUE_ORDINAL = 999_999
_HANDLE_RE = re.compile(r"^H[0-9]{3,6}$")
_GROUP_RE = re.compile(r"^G[0-9]{3,6}$")
_DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}-\d{2}(?:-\d{2})?\b|"
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+(?:19|20)\d{2}\b",
    re.IGNORECASE,
)


def _require(ok: object, message: str) -> None:
    if not ok:
        raise MatchedEvalContractError(message)


def _ordered_work_identity(
    question: source_cli.FastMaterializationQuestionPlan,
) -> tuple[str, str, str]:
    work = question.mapping_plan.work_items
    _require(bool(work), "tail typed question has no map work")
    identities = {
        (row.question_id, row.dated_question_sha256, row.route_receipt_sha256)
        for row in work
    }
    _require(
        len(identities) == 1 and next(iter(identities))[0] == question.question_id,
        "tail typed question work identity changed",
    )
    return next(iter(identities))


@dataclass(frozen=True, slots=True)
class TailFactUnionRow:
    """One exact tail question/materialization/post-map-union binding."""

    ordinal: int
    question_id: str
    dated_question_sha256: str
    route_receipt_sha256: str
    question_plan: source_cli.FastMaterializationQuestionPlan
    mapper_materialization: SourceMapperMaterialization
    fact_union: PostMapFactUnion
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    gold_loaded: Literal[False] = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(type(self.ordinal) is int and self.ordinal >= 0, "tail row ordinal changed")
        require_text(self.question_id, "tail row question ID")
        require_sha256(self.dated_question_sha256, "tail row dated question")
        require_sha256(self.route_receipt_sha256, "tail row route")
        _require(
            type(self.question_plan) is source_cli.FastMaterializationQuestionPlan,
            "tail row question plan changed type",
        )
        _require(
            type(self.mapper_materialization) is SourceMapperMaterialization,
            "tail row materialization changed type",
        )
        _require(type(self.fact_union) is PostMapFactUnion, "tail row fact union changed type")
        work_question_id, dated_sha, route_sha = _ordered_work_identity(self.question_plan)
        _require(
            self.ordinal == self.question_plan.ordinal
            and self.question_id == self.question_plan.question_id == work_question_id
            and self.dated_question_sha256 == dated_sha
            and self.route_receipt_sha256 == route_sha,
            "tail row escaped its exact question identity",
        )
        _require(
            self.mapper_materialization.preflight_receipt_sha256
            == self.question_plan.mapper_preflight.receipt_sha256
            and self.mapper_materialization.mapping_plan_receipt_sha256
            == self.question_plan.mapping_plan.receipt_sha256
            and self.mapper_materialization.hydration_plan_receipt_sha256
            == self.question_plan.hydration_plan.receipt_sha256
            and self.mapper_materialization.provider_calls_during_materialization == 0
            and self.mapper_materialization.retained_transformer_token_state_bytes == 0,
            "tail row mapper lineage or zero-state firewall changed",
        )
        _require(
            self.fact_union.parent == self.question_plan.hydration_plan.parent
            and self.fact_union.hydration_plan_receipt_sha256
            == self.question_plan.hydration_plan.receipt_sha256
            and self.fact_union.map_batch_receipt_sha256s
            == tuple(row.receipt_sha256 for row in self.mapper_materialization.batches),
            "tail row post-map union escaped its mapper batches",
        )
        _require(
            self.provider_prompt_count == 0
            and self.retained_transformer_token_state_bytes == 0
            and self.gold_loaded is False,
            "tail row must remain provider-free, gold-blind, and zero-state",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise MatchedEvalContractError("tail typed row receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="adaptive_source_tail_typed_row")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "dated_question_sha256": self.dated_question_sha256,
            "fact_union_receipt_sha256": self.fact_union.receipt_sha256,
            "format": ROW_FORMAT,
            "gold_loaded": False,
            "mapper_materialization_receipt_sha256": (
                self.mapper_materialization.receipt_sha256
            ),
            "ordinal": self.ordinal,
            "provider_prompt_count": 0,
            "question_id": self.question_id,
            "question_plan_hydration_receipt_sha256": (
                self.question_plan.hydration_plan.receipt_sha256
            ),
            "retained_transformer_token_state_bytes": 0,
            "route_receipt_sha256": self.route_receipt_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def build_tail_post_map_fact_unions(
    questions: tuple[source_cli.FastMaterializationQuestionPlan, ...],
    materializations: tuple[SourceMapperMaterialization, ...],
) -> tuple[TailFactUnionRow, ...]:
    """Build exactly one post-map union for every loaded tail question."""

    _require(
        type(questions) is tuple
        and bool(questions)
        and all(type(row) is source_cli.FastMaterializationQuestionPlan for row in questions),
        "tail typed questions changed type",
    )
    _require(
        type(materializations) is tuple
        and len(materializations) == len(questions)
        and all(type(row) is SourceMapperMaterialization for row in materializations),
        "tail typed materialization population changed",
    )
    _require(
        len({row.question_id for row in questions}) == len(questions)
        and len({row.ordinal for row in questions}) == len(questions),
        "tail typed question identities repeat",
    )
    result: list[TailFactUnionRow] = []
    for question, materialization in zip(questions, materializations, strict=True):
        _require(
            materialization.preflight_receipt_sha256
            == question.mapper_preflight.receipt_sha256
            and materialization.mapping_plan_receipt_sha256
            == question.mapping_plan.receipt_sha256
            and materialization.hydration_plan_receipt_sha256
            == question.hydration_plan.receipt_sha256,
            "tail typed materialization escaped its ordered question plan",
        )
        fact_union = build_post_map_fact_union(
            question.hydration_plan,
            batches=materialization.batches,
            direct_evidence=question.direct_evidence,
        )
        _question_id, dated_sha, route_sha = _ordered_work_identity(question)
        result.append(
            TailFactUnionRow(
                question.ordinal,
                question.question_id,
                dated_sha,
                route_sha,
                question,
                materialization,
                fact_union,
            )
        )
    return tuple(result)


def _opaque_group_keys(admissions: tuple[LaneAdmission, ...]) -> tuple[str, ...]:
    """Return local source co-membership digests, never question-prefix affinity."""

    return tuple(
        identity_sha256(
            {
                "origin_source_locators": sorted(
                    {
                        identity_sha256(
                            {
                                "namespace_id": origin.namespace_id,
                                "source_id": origin.source_id,
                            }
                        )
                        for origin in admission.union_fact.origins
                    }
                )
            }
        )
        for admission in admissions
    )


def _allocate_groups(keys: tuple[str, ...], *, start: int) -> tuple[str, ...]:
    _require(
        type(start) is int and 1 <= start <= _MAX_OPAQUE_ORDINAL,
        "tail opaque group allocation start is invalid",
    )
    by_key: dict[str, str] = {}
    groups: list[str] = []
    for key in keys:
        require_sha256(key, "tail opaque group key")
        if key not in by_key:
            ordinal = start + len(by_key)
            _require(ordinal <= _MAX_OPAQUE_ORDINAL, "tail opaque group allocation overflow")
            by_key[key] = f"G{ordinal:03d}"
        groups.append(by_key[key])
    return tuple(groups)


def _numeric_value(summary: str) -> float | None:
    # ISO dates contain numeric-looking ``-MM`` and ``-DD`` fragments.  The
    # shared conservative extractor removes complete date spans and accepts an
    # operand only when exactly one non-date number remains.
    return conservative_numeric_value(summary)


def _numeric_role(summary: str, numeric_value: float | None) -> NumericRole:
    if numeric_value is None:
        return NumericRole.NONE
    if re.search(r"\b(?:baseline|started|initial(?:ly)?)\b", summary, re.IGNORECASE):
        return NumericRole.BASELINE
    if re.search(
        r"\b(?:ended|ending|current|now|reached|grew to)\b",
        summary,
        re.IGNORECASE,
    ):
        return NumericRole.END
    if re.search(
        r"\b(?:increase|gain|grew by|decrease|loss|delta)\b",
        summary,
        re.IGNORECASE,
    ):
        return NumericRole.DELTA
    return NumericRole.OPERAND


def _status(summary: str, event_status: str | None) -> EvidenceStatus:
    combined = " ".join(row for row in (event_status, summary) if row)
    if re.search(r"\b(?:cancelled|canceled|abandoned)\b", combined, re.IGNORECASE):
        return EvidenceStatus.CANCELLED
    if re.search(
        r"\b(?:needs?\s+to|still\s+needs?|not\s+yet|awaiting|pending)\b",
        combined,
        re.IGNORECASE,
    ):
        return EvidenceStatus.CURRENT
    if re.search(r"\b(?:plan|planned|proposed|intend)\b", combined, re.IGNORECASE):
        return EvidenceStatus.PROPOSED
    if re.search(r"\b(?:current|currently|now|latest)\b", combined, re.IGNORECASE):
        return EvidenceStatus.CURRENT
    if re.search(
        r"\b(?:completed|finished|did|bought|paid|spent|went|visited)\b",
        combined,
        re.IGNORECASE,
    ):
        return EvidenceStatus.COMPLETED
    return EvidenceStatus.UNKNOWN


def _kind(spec: TypedOperatorSpec) -> TypedItemKind:
    return {
        RoutedRepairStyle.NUMERIC_REDUCE: TypedItemKind.OPERAND,
        RoutedRepairStyle.SET_JOIN: TypedItemKind.MEMBER,
        RoutedRepairStyle.TIMELINE: TypedItemKind.EVENT,
        RoutedRepairStyle.STATE_CHAIN: TypedItemKind.STATE,
        RoutedRepairStyle.EXTRACT: TypedItemKind.DIRECT,
        RoutedRepairStyle.SYNTHESIZE: TypedItemKind.CLAIM,
    }[spec.style]


def _raw_union_fact_item(
    spec: TypedOperatorSpec,
    fact: UnionFact,
    handle_id: str,
) -> dict[str, Any]:
    summary = " | ".join(fact.fact_variants)
    numeric = _numeric_value(summary)
    event = fact.event_tuple
    date_match = _DATE_RE.search(summary)
    raw: dict[str, Any] = {
        "handle_ids": [handle_id],
        "included": True,
        "kind": _kind(spec).value,
        "numeric_role": _numeric_role(summary, numeric).value,
        "status": _status(summary, None if event is None else event.status).value,
        "summary": summary,
        "value_authority": ValueAuthority.EXPLICIT.value,
    }
    if numeric is not None:
        raw["numeric_value"] = numeric
    if event is not None:
        raw["entity_key"] = event.subject
        raw["relation"] = event.predicate
        if event.event_time.casefold() not in {"unknown", "unspecified", "none"}:
            raw["date"] = event.event_time
    elif date_match is not None:
        raw["date"] = date_match.group(0)
    return raw


def _raw_direct_pointer_item(
    spec: TypedOperatorSpec,
    fact: UnionFact,
    refs: tuple[DirectEvidenceRef, ...],
    handle_id: str,
) -> dict[str, Any]:
    """Render compact typed semantics without repacking excluded raw text.

    The exact fact variants and direct citations remain in the prompt-external
    local audit.  The provider lane receives only structured event coordinates
    plus role/status/scalar fields needed to interpret the pointer.
    """

    raw_summary = " | ".join(fact.fact_variants)
    numeric = _numeric_value(raw_summary)
    role = _numeric_role(raw_summary, numeric)
    event = fact.event_tuple
    if event is None:
        ref_events = {
            identity_sha256(ref.event_tuple.projection()): ref.event_tuple
            for ref in refs
            if ref.event_tuple is not None
        }
        if len(ref_events) == 1:
            event = next(iter(ref_events.values()))
    status = _status(
        raw_summary,
        None if event is None else event.status,
    )
    # These are exact compact mapped claims, not the protected raw citation.
    # Keep whole variants: the outer typed packet/lane allocator owns budget
    # admission and must drop a whole semantic item rather than truncating a
    # selected fact inside the item.
    mapped_claims = [" ".join(variant.split()) for variant in fact.fact_variants]
    _require(bool(mapped_claims), "direct pointer lost its mapped semantic claim")
    # Keep the summary claim-only.  Status, scalar role, event coordinates and
    # DIRECT_POINTER provenance are already separate typed fields/opaque handle
    # metadata.  Mixing those metadata words into the claim can make a literal
    # status value such as ``unknown`` look like factual insufficiency.
    summary = " | ".join(mapped_claims)
    raw: dict[str, Any] = {
        "handle_ids": [handle_id],
        "included": True,
        "kind": _kind(spec).value,
        "numeric_role": role.value,
        "status": status.value,
        "summary": summary,
        "value_authority": ValueAuthority.DERIVED.value,
    }
    if numeric is not None:
        raw["numeric_value"] = numeric
    if event is not None:
        raw["entity_key"] = event.subject
        raw["relation"] = event.predicate
        if event.event_time.casefold() not in {"unknown", "unspecified", "none"}:
            raw["date"] = event.event_time
    return raw


def _raw_item(
    spec: TypedOperatorSpec,
    admission: LaneAdmission,
    handle_id: str,
) -> dict[str, Any]:
    return _raw_union_fact_item(spec, admission.union_fact, handle_id)


def _binding_locator(
    row: TailFactUnionRow,
    envelope: FactUnionEnvelope,
    admission: LaneAdmission,
) -> dict[str, Any]:
    fact = admission.union_fact
    return {
        "admission_receipt_sha256": admission.receipt_sha256,
        "fact_union_envelope_receipt_sha256": envelope.receipt_sha256,
        "fact_union_receipt_sha256": row.fact_union.receipt_sha256,
        "fact_variant_sha256s": [quote_sha256(value) for value in fact.fact_variants],
        "origin_projection_sha256s": [
            identity_sha256(origin.projection()) for origin in fact.origins
        ],
        "origin_receipt_sha256s": [
            origin.mapped_item_receipt_sha256 for origin in fact.origins
        ],
        "tail_typed_row_receipt_sha256": row.receipt_sha256,
        "union_fact_id": fact.union_fact_id,
        "union_fact_receipt_sha256": fact.receipt_sha256,
    }


def _evidence_binding(
    *,
    row: TailFactUnionRow,
    envelope: FactUnionEnvelope,
    admission: LaneAdmission,
    artifact_sha256: str,
    handle_id: str,
    group_handle: str,
) -> EvidenceHandleBinding:
    _require(_HANDLE_RE.fullmatch(handle_id) is not None, "tail handle is not opaque")
    _require(_GROUP_RE.fullmatch(group_handle) is not None, "tail group is not opaque")
    fact = admission.union_fact
    summary = " | ".join(fact.fact_variants)
    citation = json.dumps(
        [origin.quote for origin in fact.origins],
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return EvidenceHandleBinding(
        handle_id,
        EvidenceOrigin.SOURCE_FACT,
        ProvenanceGrade.EXACT_FACT_UNION,
        group_handle,
        artifact_sha256,
        envelope.receipt_sha256,
        admission.receipt_sha256,
        quote_sha256(summary),
        quote_sha256(citation),
        len(citation),
        identity_sha256(_binding_locator(row, envelope, admission)),
    )


def _direct_pointer_rows(
    row: TailFactUnionRow,
) -> tuple[
    tuple[
        DirectEvidenceExclusion,
        UnionFact,
        tuple[DirectEvidenceRef, ...],
    ],
    ...,
]:
    union_by_id = {
        fact.union_fact_id: fact
        for fact in row.fact_union.union_facts_before_direct_exclusion
    }
    retained_ids = {fact.union_fact_id for fact in row.fact_union.retained_facts}
    direct_by_id = {
        direct.evidence_id: direct for direct in row.question_plan.direct_evidence
    }
    _require(
        len(union_by_id) == len(row.fact_union.union_facts_before_direct_exclusion)
        and len(direct_by_id) == len(row.question_plan.direct_evidence),
        "tail direct-pointer source identities repeat",
    )
    result: list[
        tuple[
            DirectEvidenceExclusion,
            UnionFact,
            tuple[DirectEvidenceRef, ...],
        ]
    ] = []
    for exclusion in row.fact_union.direct_exclusions:
        fact = union_by_id.get(exclusion.union_fact_id)
        refs = tuple(
            direct_by_id.get(evidence_id)
            for evidence_id in exclusion.matching_direct_evidence_ids
        )
        _require(
            fact is not None
            and fact.union_fact_id not in retained_ids
            and bool(refs)
            and all(type(ref) is DirectEvidenceRef for ref in refs),
            "tail direct exclusion escaped its fact/direct-evidence binding",
        )
        result.append(
            (
                exclusion,
                fact,
                tuple(ref for ref in refs if type(ref) is DirectEvidenceRef),
            )
        )
    _require(
        len({row[0].union_fact_id for row in result}) == len(result),
        "tail direct exclusions repeat a union fact",
    )
    return tuple(result)


def _direct_pointer_group_keys(
    rows: tuple[
        tuple[
            DirectEvidenceExclusion,
            UnionFact,
            tuple[DirectEvidenceRef, ...],
        ],
        ...,
    ],
) -> tuple[str, ...]:
    return tuple(
        identity_sha256(
            {
                "protected_direct_source_locators": sorted(
                    {
                        identity_sha256(
                            {
                                "namespace_id": ref.namespace_id,
                                "source_id": ref.source_id,
                            }
                        )
                        for ref in refs
                    }
                )
            }
        )
        for _exclusion, _fact, refs in rows
    )


def _direct_pointer_locator(
    row: TailFactUnionRow,
    exclusion: DirectEvidenceExclusion,
    fact: UnionFact,
    refs: tuple[DirectEvidenceRef, ...],
) -> dict[str, Any]:
    return {
        "direct_evidence_projection_sha256": (
            row.fact_union.parent.direct_evidence_projection_sha256
        ),
        "direct_evidence_receipt_sha256s": [
            ref.evidence_receipt_sha256 for ref in refs
        ],
        "direct_evidence_ref_projection_sha256s": [
            identity_sha256(ref.projection()) for ref in refs
        ],
        "direct_exclusion_receipt_sha256": exclusion.receipt_sha256,
        "fact_variant_sha256s": [quote_sha256(value) for value in fact.fact_variants],
        "match_modes": list(exclusion.match_modes),
        "matching_direct_evidence_ids": list(exclusion.matching_direct_evidence_ids),
        "origin_projection_sha256s": [
            identity_sha256(origin.projection()) for origin in fact.origins
        ],
        "origin_receipt_sha256s": [
            origin.mapped_item_receipt_sha256 for origin in fact.origins
        ],
        "tail_typed_row_receipt_sha256": row.receipt_sha256,
        "union_fact_id": fact.union_fact_id,
        "union_fact_receipt_sha256": fact.receipt_sha256,
    }


def _direct_pointer_equivalence_locator(
    fact: UnionFact,
    refs: tuple[DirectEvidenceRef, ...],
) -> dict[str, Any]:
    """Stable exact-span/action seed shared across base/tail treatments."""

    return {
        "direct_span_dedup_key_sha256s": sorted(
            {ref.dedup_key_sha256 for ref in refs}
        ),
        "mapped_fact_payload_sha256": quote_sha256(
            " | ".join(fact.fact_variants)
        ),
    }


def _direct_pointer_binding(
    *,
    row: TailFactUnionRow,
    exclusion: DirectEvidenceExclusion,
    fact: UnionFact,
    refs: tuple[DirectEvidenceRef, ...],
    artifact_sha256: str,
    handle_id: str,
    group_handle: str,
) -> EvidenceHandleBinding:
    summary = " | ".join(fact.fact_variants)
    return EvidenceHandleBinding(
        handle_id,
        EvidenceOrigin.DIRECT_POINTER,
        ProvenanceGrade.DIRECT_POINTER,
        group_handle,
        artifact_sha256,
        row.fact_union.receipt_sha256,
        exclusion.receipt_sha256,
        quote_sha256(summary),
        # The protected direct citation already occupies the earlier evidence
        # lane.  A semantic pointer carries no second copy of that raw text.
        quote_sha256(""),
        0,
        identity_sha256(_direct_pointer_equivalence_locator(fact, refs)),
    )


def adapt_tail_fact_union_contribution(
    operator_spec: TypedOperatorSpec,
    row: TailFactUnionRow,
    *,
    materialization_artifact_sha256: str,
    parent_prompt_token_proxy: int,
    handle_start: int,
    group_start: int,
    mechanism_id: str = MECHANISM_ID,
) -> TypedEvidenceContribution:
    """Adapt admitted retained facts with caller-owned global H/G ranges.

    No relevance filter or second dedup pass is applied here.  Consequently,
    every lane admission—including duplicated-looking or noisy evidence—keeps
    its own handle, while multiple facts from the same local source group share
    only an opaque G handle.  Completeness always remains ``BOUNDED``.
    """

    if type(operator_spec) is not TypedOperatorSpec:
        raise TypeError("operator_spec must be exact")
    if type(row) is not TailFactUnionRow:
        raise TypeError("tail row must be exact")
    require_sha256(materialization_artifact_sha256, "tail materialization artifact")
    require_text(mechanism_id, "tail typed mechanism")
    _require(
        operator_spec.question_sha256 == row.dated_question_sha256
        and operator_spec.route_receipt_sha256 == row.route_receipt_sha256,
        "tail typed contribution escaped its question/route binding",
    )
    _require(
        type(handle_start) is int and 1 <= handle_start <= _MAX_OPAQUE_ORDINAL,
        "tail opaque handle allocation start is invalid",
    )
    envelope = pack_fact_union_envelope(
        row.fact_union,
        parent_prompt_token_proxy=parent_prompt_token_proxy,
    )
    admissions = tuple(
        admission
        for lane_pack in envelope.lane_packs
        for admission in lane_pack.admissions
    )
    retained_by_id = {fact.union_fact_id: fact for fact in row.fact_union.retained_facts}
    _require(
        len(retained_by_id) == len(row.fact_union.retained_facts)
        and all(
            admission.union_fact is retained_by_id.get(admission.union_fact.union_fact_id)
            for admission in admissions
        ),
        "tail typed admission escaped retained post-map facts",
    )
    if admissions:
        _require(
            handle_start + len(admissions) - 1 <= _MAX_OPAQUE_ORDINAL,
            "tail opaque handle allocation overflow",
        )
    group_handles = _allocate_groups(_opaque_group_keys(admissions), start=group_start)
    bindings: list[EvidenceHandleBinding] = []
    raw_items: list[dict[str, Any]] = []
    for offset, (admission, group_handle) in enumerate(
        zip(admissions, group_handles, strict=True)
    ):
        handle_id = f"H{handle_start + offset:03d}"
        binding = _evidence_binding(
            row=row,
            envelope=envelope,
            admission=admission,
            artifact_sha256=materialization_artifact_sha256,
            handle_id=handle_id,
            group_handle=group_handle,
        )
        bindings.append(binding)
        raw_items.append(_raw_item(operator_spec, admission, handle_id))
    parsed = parse_typed_items(
        raw_items,
        operator_spec=operator_spec,
        bindings=tuple(bindings),
    )
    # Bind the parser result back to the exact admitted fact population without
    # changing its accepted/rejected items.
    parsed = ParsedTypedItems(
        parsed.accepted_items,
        parsed.rejected_items,
        identity_sha256(
            {
                "admission_receipt_sha256s": [
                    admission.receipt_sha256 for admission in admissions
                ],
                "fact_union_envelope_receipt_sha256": envelope.receipt_sha256,
                "format": f"{ROW_FORMAT}-typed-parse",
                "parser_receipt_sha256": parsed.parse_receipt_sha256,
                "tail_typed_row_receipt_sha256": row.receipt_sha256,
            }
        ),
    )
    truncated = bool(
        row.fact_union.pending_window_ids
        or any(pack.not_admitted_union_fact_ids for pack in envelope.lane_packs)
    )
    return TypedEvidenceContribution(
        mechanism_id,
        tuple(bindings),
        parsed,
        materialization_artifact_sha256,
        FrontierMode.BOUNDED,
        truncated,
    )


def adapt_tail_direct_pointer_contribution(
    operator_spec: TypedOperatorSpec,
    row: TailFactUnionRow,
    *,
    materialization_artifact_sha256: str,
    handle_start: int,
    group_start: int,
    mechanism_id: str = DIRECT_POINTER_MECHANISM_ID,
) -> TypedEvidenceContribution:
    """Preserve excluded fact semantics as pointers to protected direct evidence.

    Post-selection dedup remains authoritative: these are never reintroduced as
    ``SOURCE_FACT`` bindings and their exact raw citations are not copied into a
    second prompt lane.  Each pointer instead binds the excluded union-fact and
    exclusion receipts to the already protected direct-evidence receipts.
    """

    if type(operator_spec) is not TypedOperatorSpec:
        raise TypeError("operator_spec must be exact")
    if type(row) is not TailFactUnionRow:
        raise TypeError("tail row must be exact")
    require_sha256(materialization_artifact_sha256, "tail materialization artifact")
    require_text(mechanism_id, "tail direct-pointer mechanism")
    _require(
        operator_spec.question_sha256 == row.dated_question_sha256
        and operator_spec.route_receipt_sha256 == row.route_receipt_sha256,
        "tail direct-pointer contribution escaped its question/route binding",
    )
    _require(
        type(handle_start) is int and 1 <= handle_start <= _MAX_OPAQUE_ORDINAL,
        "tail direct-pointer handle allocation start is invalid",
    )
    pointer_rows = _direct_pointer_rows(row)
    if pointer_rows:
        _require(
            handle_start + len(pointer_rows) - 1 <= _MAX_OPAQUE_ORDINAL,
            "tail direct-pointer handle allocation overflow",
        )
    groups = _allocate_groups(
        _direct_pointer_group_keys(pointer_rows), start=group_start
    )
    bindings: list[EvidenceHandleBinding] = []
    raw_items: list[dict[str, Any]] = []
    for offset, ((exclusion, fact, refs), group_handle) in enumerate(
        zip(pointer_rows, groups, strict=True)
    ):
        handle_id = f"H{handle_start + offset:03d}"
        bindings.append(
            _direct_pointer_binding(
                row=row,
                exclusion=exclusion,
                fact=fact,
                refs=refs,
                artifact_sha256=materialization_artifact_sha256,
                handle_id=handle_id,
                group_handle=group_handle,
            )
        )
        raw_items.append(
            _raw_direct_pointer_item(
                operator_spec,
                fact,
                refs,
                handle_id,
            )
        )
    parsed = parse_typed_items(
        raw_items,
        operator_spec=operator_spec,
        bindings=tuple(bindings),
    )
    parsed = ParsedTypedItems(
        parsed.accepted_items,
        parsed.rejected_items,
        identity_sha256(
            {
                "direct_exclusion_receipt_sha256s": [
                    exclusion.receipt_sha256
                    for exclusion, _fact, _refs in pointer_rows
                ],
                "format": f"{ROW_FORMAT}-direct-pointer-parse",
                "parser_receipt_sha256": parsed.parse_receipt_sha256,
                "tail_typed_row_receipt_sha256": row.receipt_sha256,
            }
        ),
    )
    return TypedEvidenceContribution(
        mechanism_id,
        tuple(bindings),
        parsed,
        materialization_artifact_sha256,
        FrontierMode.BOUNDED,
        bool(row.fact_union.pending_window_ids),
    )


def adapt_tail_question_contributions(
    operator_spec: TypedOperatorSpec,
    row: TailFactUnionRow,
    *,
    materialization_artifact_sha256: str,
    parent_prompt_token_proxy: int,
    source_handle_start: int,
    source_group_start: int,
    pointer_handle_start: int,
    pointer_group_start: int,
    source_mechanism_id: str = MECHANISM_ID,
    pointer_mechanism_id: str = DIRECT_POINTER_MECHANISM_ID,
) -> tuple[TypedEvidenceContribution, TypedEvidenceContribution]:
    """Return separately accountable retained-fact and direct-pointer arms."""

    source = adapt_tail_fact_union_contribution(
        operator_spec,
        row,
        materialization_artifact_sha256=materialization_artifact_sha256,
        parent_prompt_token_proxy=parent_prompt_token_proxy,
        handle_start=source_handle_start,
        group_start=source_group_start,
        mechanism_id=source_mechanism_id,
    )
    pointer = adapt_tail_direct_pointer_contribution(
        operator_spec,
        row,
        materialization_artifact_sha256=materialization_artifact_sha256,
        handle_start=pointer_handle_start,
        group_start=pointer_group_start,
        mechanism_id=pointer_mechanism_id,
    )
    _require(
        not (
            {binding.handle_id for binding in source.bindings}
            & {binding.handle_id for binding in pointer.bindings}
        )
        and not (
            {binding.source_group_handle for binding in source.bindings}
            & {binding.source_group_handle for binding in pointer.bindings}
        ),
        "tail retained-fact and direct-pointer allocation ranges collide",
    )
    return source, pointer


__all__ = [
    "DIRECT_POINTER_MECHANISM_ID",
    "MECHANISM_ID",
    "ROW_FORMAT",
    "TailFactUnionRow",
    "adapt_tail_fact_union_contribution",
    "adapt_tail_direct_pointer_contribution",
    "adapt_tail_question_contributions",
    "build_tail_post_map_fact_unions",
]
