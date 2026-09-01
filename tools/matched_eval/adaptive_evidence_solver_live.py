"""Provider-free contract for the adaptive map-plus-source final solver.

This is deliberately a sibling of :mod:`query_evidence_map_solver_v2_live`.
The latter remains the immutable baseline-V2 population.  This module inserts
an optional, already validated post-map source-fact plane after the terminal
V2 evidence map and before one final answer decision.  It performs no I/O and
never calls a provider; prompt execution is an external step whose exact text
is ingested through the completion-plane contract below.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal, Mapping, Sequence

from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
)
from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval.fast_completion_runtime import (
    FastPromptPopulation,
    preflight_fast_completion_prompts,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    FastProviderMessage,
)
from tools._routed_repair_routing import RoutedRepairStyle

from . import live
from .contracts import (
    MatchedEvalContractError,
    StageDisposition,
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
    require_text,
)
from .query_evidence_map_solver_v2_live import (
    ELIGIBLE_STYLES,
    MAP_ITEM_FORMAT,
    MAX_PROMPT_TOKENS,
    PRESERVED_STYLES,
    SOLVER_OUTPUT_TOKEN_RESERVE,
    EvidenceMapPlan,
    EvidenceMapPlanRow,
    ValidatedMapItem,
    VerifiedEvidenceMapPlane,
    VerifiedEvidenceMapRow,
)
from .query_operator_refinement_live import _plain_messages, _route_projection
from .source_history_fact_union import (
    EXTERNAL_LINK_OVERLAY_TOKEN_RESERVE,
    FINAL_PROMPT_TOKEN_CAP,
    FORMAT as SOURCE_FACT_FORMAT,
    LANE_ORDER,
    LANE_TOKEN_BUDGETS,
    OUTPUT_TOKEN_RESERVE as SOURCE_FACT_OUTPUT_TOKEN_RESERVE,
    DirectEvidenceExclusion,
    FactLane,
    FactOrigin,
    FactUnionEnvelope,
    LaneAdmission,
    LanePack,
    ParentIdentity,
    PostMapFactUnion,
    UnionFact,
    compact_fact_prompt_projection,
    pack_fact_union_envelope,
)


FORMAT = "memory-condense-adaptive-evidence-solver-v3"
ARM_LABEL = "S0_PLUS_VALIDATED_MAP_PLUS_POST_MAP_SOURCE_FACT_SOLVER_V3"
PLAN_ID = "matched_adaptive_evidence_solver_v3"
RENDERER_ID = "matched_adaptive_evidence_solver_v3"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight"


class AdaptiveEvidenceSolverError(MatchedEvalContractError):
    """A terminal parent, fact plane, prompt, or completion changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise AdaptiveEvidenceSolverError(message)


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _seal(kind: str, body: Mapping[str, Any]) -> str:
    value = {"format": f"{FORMAT}-{kind}", **body}
    assert_gold_blind(value, path="adaptive_evidence_solver")
    return identity_sha256(value)


def _source_seal(kind: str, body: Mapping[str, Any]) -> str:
    return identity_sha256({"format": f"{SOURCE_FACT_FORMAT}-{kind}", **body})


def _exact_zero(value: object, label: str) -> None:
    _require(type(value) is int and value == 0, f"{label} must be exact zero")


_OPERATOR_WORK: dict[RoutedRepairStyle, tuple[str, ...]] = {
    RoutedRepairStyle.EXTRACT: (
        "Enumerate every supplied candidate value that could directly answer the "
        "question.",
        "Compare entity, event, role, date, and requested field; disambiguate "
        "similar candidates before selecting one.",
        "Replace only when one source-backed value uniquely satisfies the full "
        "question; otherwise preserve the parent.",
    ),
    RoutedRepairStyle.NUMERIC_REDUCE: (
        "Enumerate every potentially eligible operand with value, unit, sign, "
        "entity or event, date, status, and inclusion reason.",
        "Apply question constraints first, deduplicate repeated events only after "
        "selection, then perform the requested count, arithmetic, or comparison.",
        "Do not replace from a partial operand list or combine incompatible units; "
        "preserve the parent when the complete operand frontier is not established.",
    ),
    RoutedRepairStyle.SET_JOIN: (
        "Enumerate every candidate member and every explicit inclusion, exclusion, "
        "correction, role, date, and status constraint.",
        "Apply constraints before deduplicating identical selected members, then "
        "check the requested cardinality and ordering.",
        "Do not replace from a subset of the member frontier; preserve the parent "
        "when completeness cannot be established.",
    ),
    RoutedRepairStyle.SYNTHESIZE: (
        "Enumerate every relevant claim separately with subject, aspect, polarity, "
        "date, status, attribution, conflict, and supersession.",
        "Compare all claims bearing on the requested synthesis and reconcile only "
        "explicit conflicts or corrections; do not cherry-pick one supporting fact.",
        "Replace only with a synthesis supported by the compared source-backed "
        "claims; preserve the parent when material conflicts remain unresolved.",
    ),
    RoutedRepairStyle.TIMELINE: (
        "Enumerate every potentially relevant event with date or order, subject, "
        "event status, and explicit start, end, cancellation, or completion role.",
        "Order the events as of the question timestamp, keeping proposals, attempts, "
        "cancellations, and completions distinct before selecting or calculating.",
        "Do not replace from a partial timeline or missing boundary event; preserve "
        "the parent when the required event frontier is incomplete.",
    ),
    RoutedRepairStyle.STATE_CHAIN: (
        "Enumerate every relevant dated state, revision, correction, and "
        "supersession, preserving old and new values.",
        "Order the chain as of the question timestamp and apply only completed "
        "changes; a proposal or attempt does not establish a new state.",
        "The existing parent may already be the resolved state: preserve it unless "
        "the complete source-backed chain establishes a different requested state.",
    ),
}


def _operator_policy_projection(style: RoutedRepairStyle) -> dict[str, Any]:
    """Return evidence-independent operator instructions selected by route only."""

    _require(type(style) is RoutedRepairStyle, "adaptive solver route style changed")
    _require(
        set(_OPERATOR_WORK) == set(RoutedRepairStyle),
        "adaptive solver operator policy does not cover every route",
    )
    return {
        "complete_frontier_rule": (
            "When QUESTION_ONLY_ROUTE_JSON.modifiers.requires_complete_frontier "
            "is true, replacement requires checking every plausible supplied "
            "operand, member, event, or claim. Absence of evidence is not an "
            "exclusion. If completeness cannot be established, preserve the parent."
        ),
        "operator": style.value,
        "question_only": True,
        "silent_work": list(_OPERATOR_WORK[style]),
    }


def _source_fact_line(alias: str, fact: UnionFact) -> str:
    return _json(compact_fact_prompt_projection(alias, fact))


def _validate_origin(origin: FactOrigin, *, namespace_id: str) -> None:
    _require(type(origin) is FactOrigin, "source-fact origin changed type")
    _require(type(origin.lane) is FactLane, "source-fact origin changed lane")
    for value, label in (
        (origin.selection_id, "source-fact selection ID"),
        (origin.window_id, "source-fact window ID"),
        (origin.source_id, "source-fact source ID"),
        (origin.chunk_id, "source-fact chunk ID"),
        (origin.mapper_item_id, "source-fact mapper item ID"),
        (origin.source_created_at, "source-fact observed date"),
        (origin.source_role, "source-fact source role"),
    ):
        require_text(value, label)
    _require(origin.namespace_id == namespace_id, "source fact escaped its namespace")
    require_sha256(origin.namespace_id, "source-fact namespace")
    require_sha256(origin.quote_sha256, "source-fact quote")
    require_sha256(origin.mapped_item_receipt_sha256, "source-fact mapped item")
    _require(
        type(origin.quote) is str
        and bool(origin.quote)
        and quote_sha256(origin.quote) == origin.quote_sha256
        and type(origin.quote_start_char) is int
        and type(origin.quote_end_char) is int
        and origin.quote_start_char >= 0
        and origin.quote_end_char - origin.quote_start_char == len(origin.quote),
        "source-fact exact quote coordinates changed",
    )


def _validate_union_fact(fact: UnionFact, *, namespace_id: str) -> None:
    _require(type(fact) is UnionFact, "source union fact changed type")
    for value, label in (
        (fact.union_fact_id, "union fact ID"),
        (fact.dedup_key_sha256, "union dedup key"),
        (fact.receipt_sha256, "union fact receipt"),
    ):
        require_sha256(value, label)
    _require(
        type(fact.dedup_projection) is dict
        and identity_sha256(fact.dedup_projection) == fact.dedup_key_sha256,
        "source fact dedup projection changed",
    )
    _require(
        type(fact.fact_variants) is tuple
        and bool(fact.fact_variants)
        and len(set(fact.fact_variants)) == len(fact.fact_variants),
        "source fact variants are empty or repeated",
    )
    for value in fact.fact_variants:
        require_text(value, "source fact variant")
    _require(
        type(fact.origins) is tuple
        and bool(fact.origins)
        and len({row.mapped_item_receipt_sha256 for row in fact.origins})
        == len(fact.origins),
        "source fact origins are empty or repeated",
    )
    for origin in fact.origins:
        _validate_origin(origin, namespace_id=namespace_id)
    expected_owner = min(
        (origin.lane for origin in fact.origins), key=LANE_ORDER.index
    )
    _require(fact.owner_lane is expected_owner, "source fact owner lane changed")
    if fact.event_tuple is None:
        first = fact.origins[0]
        expected_dedup: dict[str, Any] = {
            "chunk_id": first.chunk_id,
            "kind": "source_chunk_quote",
            "namespace_id": first.namespace_id,
            "quote_sha256": first.quote_sha256,
            "source_id": first.source_id,
        }
        _require(
            all(
                origin.namespace_id == first.namespace_id
                and origin.source_id == first.source_id
                and origin.chunk_id == first.chunk_id
                and origin.quote_sha256 == first.quote_sha256
                for origin in fact.origins
            ),
            "quote-deduped source fact merged unequal coordinates",
        )
    else:
        expected_dedup = {
            "event_tuple": fact.event_tuple.projection(),
            "kind": "full_event_tuple",
        }
    _require(fact.dedup_projection == expected_dedup, "source fact dedup kind changed")
    _require(
        fact.union_fact_id
        == _source_seal("union-fact-id", {"dedup_key_sha256": fact.dedup_key_sha256}),
        "source union fact ID seal changed",
    )
    body = {
        "dedup_key_sha256": fact.dedup_key_sha256,
        "dedup_projection": fact.dedup_projection,
        "fact_variants": list(fact.fact_variants),
        "origins": [row.projection() for row in fact.origins],
        "owner_lane": fact.owner_lane.value,
        "union_fact_id": fact.union_fact_id,
    }
    _require(
        fact.receipt_sha256 == _source_seal("union-fact", body),
        "source union fact receipt changed",
    )


def _validate_parent_binding(
    parent: ParentIdentity,
    map_plan: EvidenceMapPlan,
    map_row: EvidenceMapPlanRow,
) -> None:
    _require(type(parent) is ParentIdentity, "source-fact parent changed type")
    source_snapshot = map_plan.direct_plan.adapter_population.source_population.snapshot
    _require(
        parent.population_identity_sha256
        == source_snapshot.population_identity_sha256
        and parent.question_order_sha256 == source_snapshot.question_order_sha256
        and parent.snapshot_id == source_snapshot.snapshot_id
        and parent.parent_packet_id == map_row.packet_id,
        "source facts escaped their exact question/map parent",
    )
    require_sha256(parent.namespace_id, "source-fact parent namespace")
    require_sha256(parent.parent_stage_receipt_sha256, "source-fact parent stage")
    require_sha256(
        parent.direct_evidence_projection_sha256,
        "source-fact direct evidence projection",
    )


def _validate_fact_union(
    fact_union: PostMapFactUnion,
    map_plan: EvidenceMapPlan,
    map_row: EvidenceMapPlanRow,
) -> None:
    _require(type(fact_union) is PostMapFactUnion, "fact union changed type")
    _validate_parent_binding(fact_union.parent, map_plan, map_row)
    require_sha256(fact_union.hydration_plan_receipt_sha256, "hydration plan")
    for values, label in (
        (fact_union.completed_window_ids, "completed source windows"),
        (fact_union.pending_window_ids, "pending source windows"),
        (fact_union.map_batch_receipt_sha256s, "source map batches"),
    ):
        _require(type(values) is tuple, f"{label} changed type")
        for value in values:
            require_sha256(value, label)
        _require(len(set(values)) == len(values), f"{label} repeat")
    _require(
        not set(fact_union.completed_window_ids) & set(fact_union.pending_window_ids)
        and len(fact_union.completed_window_ids)
        == len(fact_union.map_batch_receipt_sha256s),
        "source window lifecycle changed",
    )
    _require(
        type(fact_union.accepted_before_dedup_count) is int
        and fact_union.accepted_before_dedup_count >= 0
        and type(fact_union.rejected_item_count) is int
        and fact_union.rejected_item_count >= 0,
        "source map counts changed",
    )
    before = fact_union.union_facts_before_direct_exclusion
    retained = fact_union.retained_facts
    exclusions = fact_union.direct_exclusions
    _require(
        type(before) is tuple
        and type(retained) is tuple
        and type(exclusions) is tuple
        and all(type(row) is DirectEvidenceExclusion for row in exclusions),
        "post-map union collections changed type",
    )
    for fact in before:
        _validate_union_fact(fact, namespace_id=fact_union.parent.namespace_id)
    ids = tuple(row.union_fact_id for row in before)
    dedup = tuple(row.dedup_key_sha256 for row in before)
    _require(
        len(set(ids)) == len(ids)
        and len(set(dedup)) == len(dedup)
        and fact_union.accepted_before_dedup_count >= len(before),
        "post-map source facts were not exactly deduplicated",
    )
    excluded_ids: list[str] = []
    for exclusion in exclusions:
        require_sha256(exclusion.union_fact_id, "direct exclusion union fact")
        _require(
            exclusion.union_fact_id in set(ids)
            and type(exclusion.matching_direct_evidence_ids) is tuple
            and bool(exclusion.matching_direct_evidence_ids)
            and len(set(exclusion.matching_direct_evidence_ids))
            == len(exclusion.matching_direct_evidence_ids),
            "post-map direct exclusion changed",
        )
        _require(
            type(exclusion.match_modes) is tuple
            and bool(exclusion.match_modes)
            and len(set(exclusion.match_modes)) == len(exclusion.match_modes)
            and all(
                value
                in {
                    "exact_event_tuple",
                    "legacy_exact_quote_hash",
                    "same_chunk_exact_or_contained_quote",
                    "same_chunk_strict_substring",
                }
                for value in exclusion.match_modes
            ),
            "post-map direct exclusion modes changed",
        )
        for value in exclusion.matching_direct_evidence_ids:
            require_text(value, "matching direct evidence ID")
        expected = _source_seal(
            "direct-exclusion",
            {
                "match_modes": list(exclusion.match_modes),
                "matching_direct_evidence_ids": list(
                    exclusion.matching_direct_evidence_ids
                ),
                "operation_position": "after_map_and_post_map_chunk_dedup",
                "union_fact_id": exclusion.union_fact_id,
            },
        )
        _require(
            exclusion.receipt_sha256 == expected,
            "post-map direct exclusion receipt changed",
        )
        excluded_ids.append(exclusion.union_fact_id)
    _require(len(set(excluded_ids)) == len(excluded_ids), "direct exclusion repeats")
    expected_retained = tuple(row for row in before if row.union_fact_id not in excluded_ids)
    _require(
        retained == expected_retained,
        "source facts were not excluded strictly after post-map dedup",
    )
    body = {
        "accepted_before_dedup_count": fact_union.accepted_before_dedup_count,
        "completed_window_ids": list(fact_union.completed_window_ids),
        "direct_exclusions": [row.receipt_sha256 for row in exclusions],
        "hydration_plan_receipt_sha256": fact_union.hydration_plan_receipt_sha256,
        "map_batch_receipt_sha256s": list(fact_union.map_batch_receipt_sha256s),
        "operation_order": [
            "hydrate_without_dedup",
            "map_validate_individually",
            "source_history_post_map_exact_chunk_dedup",
            "sequential_em_fact_union",
            "direct_evidence_exact_or_same_chunk_child_exclusion",
            "non_borrowing_lane_pack",
        ],
        "parent_identity_sha256": fact_union.parent.identity_sha256,
        "pending_window_ids": list(fact_union.pending_window_ids),
        "rejected_item_count": fact_union.rejected_item_count,
        "retained_union_fact_ids": [row.union_fact_id for row in retained],
        "union_fact_ids": list(ids),
    }
    _require(
        fact_union.receipt_sha256 == _source_seal("post-map-union", body),
        "post-map source fact union receipt changed",
    )


def _validate_lane_pack(pack: LanePack, *, namespace_id: str) -> None:
    _require(type(pack) is LanePack, "source lane pack changed type")
    _require(
        type(pack.lane) is FactLane
        and pack.token_cap == LANE_TOKEN_BUDGETS[pack.lane]
        and type(pack.non_borrowing) is bool
        and pack.non_borrowing is True,
        "source lane budget/borrowing contract changed",
    )
    candidates = pack.candidate_union_fact_ids
    omitted = pack.not_admitted_union_fact_ids
    _require(
        type(candidates) is tuple
        and type(omitted) is tuple
        and len(set(candidates)) == len(candidates)
        and len(set(omitted)) == len(omitted),
        "source lane candidate IDs changed",
    )
    admissions = pack.admissions
    _require(
        type(admissions) is tuple
        and all(type(row) is LaneAdmission for row in admissions),
        "source lane admissions changed type",
    )
    prefix = pack.lane.value[0].upper()
    _require(
        tuple(row.alias for row in admissions)
        == tuple(f"{prefix}{index:03d}" for index in range(1, len(admissions) + 1)),
        "source fact evidence aliases changed",
    )
    admitted_ids: list[str] = []
    lines: list[str] = []
    for admission in admissions:
        _validate_union_fact(admission.union_fact, namespace_id=namespace_id)
        _require(
            admission.union_fact.owner_lane is pack.lane,
            "source fact entered the wrong lane",
        )
        line = _source_fact_line(admission.alias, admission.union_fact)
        expected = _source_seal(
            "lane-admission",
            {
                "alias": admission.alias,
                "rendered_line_sha256": quote_sha256(line),
                "union_fact_id": admission.union_fact.union_fact_id,
                "union_fact_receipt_sha256": admission.union_fact.receipt_sha256,
            },
        )
        _require(
            admission.rendered_line == line and admission.receipt_sha256 == expected,
            "source fact lane admission changed",
        )
        admitted_ids.append(admission.union_fact.union_fact_id)
        lines.append(line)
    _require(
        not set(admitted_ids) & set(omitted)
        and tuple(value for value in candidates if value in set(admitted_ids))
        == tuple(admitted_ids)
        and tuple(value for value in candidates if value not in set(admitted_ids))
        == omitted,
        "source lane admissions do not partition candidates",
    )
    header = f"[{pack.lane.value.upper()}_FACTS]"
    block = "" if not lines else header + "\n" + "\n".join(lines)
    _require(
        pack.rendered_block == block
        and pack.tokens_used == count_tokens(block)
        and pack.tokens_used <= pack.token_cap,
        "source lane rendered token envelope changed",
    )
    body = {
        "admissions": [row.receipt_sha256 for row in admissions],
        "candidate_union_fact_ids": list(candidates),
        "lane": pack.lane.value,
        "non_borrowing": True,
        "not_admitted_union_fact_ids": list(omitted),
        "rendered_block_sha256": quote_sha256(block),
        "token_cap": pack.token_cap,
        "tokens_used": pack.tokens_used,
    }
    _require(pack.receipt_sha256 == _source_seal("lane-pack", body), "lane pack seal changed")


def _validate_fact_envelope(
    envelope: FactUnionEnvelope,
    map_plan: EvidenceMapPlan,
    map_row: EvidenceMapPlanRow,
    *,
    parent_prompt_token_proxy: int,
) -> None:
    _require(type(envelope) is FactUnionEnvelope, "fact envelope changed type")
    _validate_parent_binding(envelope.parent, map_plan, map_row)
    _require(
        envelope.parent_prompt_token_proxy == parent_prompt_token_proxy
        and envelope.external_link_overlay_token_reserve
        == EXTERNAL_LINK_OVERLAY_TOKEN_RESERVE
        and envelope.output_token_reserve == SOURCE_FACT_OUTPUT_TOKEN_RESERVE
        and envelope.hard_prompt_token_cap == FINAL_PROMPT_TOKEN_CAP
        and envelope.hard_prompt_token_cap == MAX_PROMPT_TOKENS,
        "packed source facts target a different final renderer envelope",
    )
    _exact_zero(
        envelope.retained_transformer_token_state_bytes,
        "fact envelope retained transformer state",
    )
    _require(
        type(envelope.lane_packs) is tuple
        and tuple(row.lane for row in envelope.lane_packs) == LANE_ORDER,
        "fact envelope lane order changed",
    )
    for pack in envelope.lane_packs:
        _validate_lane_pack(pack, namespace_id=envelope.parent.namespace_id)
    rendered = "\n\n".join(
        row.rendered_block for row in envelope.lane_packs if row.rendered_block
    )
    fact_tokens = count_tokens(rendered)
    final = (
        envelope.parent_prompt_token_proxy
        + fact_tokens
        + envelope.external_link_overlay_token_reserve
        + envelope.output_token_reserve
    )
    _require(
        envelope.rendered_fact_union == rendered
        and envelope.fact_union_token_proxy == fact_tokens
        and envelope.final_envelope_token_proxy == final
        and final <= envelope.hard_prompt_token_cap,
        "fact envelope rendered token accounting changed",
    )
    body = {
        "fact_union_receipt_sha256": envelope.fact_union_receipt_sha256,
        "fact_union_token_proxy": fact_tokens,
        "external_link_overlay_token_reserve": EXTERNAL_LINK_OVERLAY_TOKEN_RESERVE,
        "final_envelope_token_proxy": final,
        "hard_prompt_token_cap": FINAL_PROMPT_TOKEN_CAP,
        "lane_pack_receipt_sha256s": [
            row.receipt_sha256 for row in envelope.lane_packs
        ],
        "lane_token_budgets": {
            lane.value: LANE_TOKEN_BUDGETS[lane] for lane in LANE_ORDER
        },
        "output_token_reserve": SOURCE_FACT_OUTPUT_TOKEN_RESERVE,
        "parent_identity_sha256": envelope.parent.identity_sha256,
        "parent_prompt_token_proxy": parent_prompt_token_proxy,
        "rendered_fact_union_sha256": quote_sha256(rendered),
        "retained_transformer_token_state_bytes": 0,
    }
    _require(
        envelope.receipt_sha256 == _source_seal("envelope", body),
        "fact envelope receipt changed",
    )


def _validate_map_item(item: ValidatedMapItem) -> None:
    _require(type(item) is ValidatedMapItem, "validated map item changed type")
    body = {
        "alias": item.alias,
        "candidate": item.candidate,
        "citation": item.citation,
        "citation_match": item.citation_match,
        "format": MAP_ITEM_FORMAT,
        "item_id": item.item_id,
        "kind": item.kind,
        "source_index": item.source_index,
    }
    _require(
        item.item_sha256 == identity_sha256(body),
        "validated map item receipt changed",
    )


def _validate_terminal_map(
    map_plan: EvidenceMapPlan,
    map_plane: VerifiedEvidenceMapPlane,
) -> None:
    _require(
        map_plane.run_sha256 == map_plane.replay_sha256
        and map_plane.parent_plane is map_plan.direct_plane
        and map_plane.parent_answer_run_sha256 == map_plan.direct_plane.run_sha256
        and map_plane.adapter_population_id
        == map_plan.direct_plan.adapter_population.population_id
        and map_plane.retrieval_sha256
        == map_plan.direct_plan.adapter_population.source_population.retrieval_sha256
        and map_plane.snapshot_id == map_plan.snapshot.snapshot_id
        and len(map_plane.rows) == len(map_plan.rows),
        "adaptive solver requires the exact terminal V2 map plane",
    )
    runtime = live._thaw_json(map_plane.runtime_ledger)
    _require(
        sha256(canonical_json_bytes(runtime)).hexdigest()
        == map_plane.runtime_ledger_sha256,
        "terminal V2 map runtime seal changed",
    )


def _map_evidence_projection(row: VerifiedEvidenceMapRow) -> dict[str, Any]:
    return {
        "map_parse_receipt_sha256": row.map_parse_receipt_sha256,
        "map_source_row_sha256": row.source_row_sha256,
        "items": [
            {
                "candidate": item.candidate,
                "citation": item.citation,
                "evidence_id": item.item_id,
                "kind": item.kind,
                "map_item_sha256": item.item_sha256,
                "source_alias": item.alias,
            }
            for item in row.accepted_items
        ],
    }


def _source_evidence_projection(
    envelope: FactUnionEnvelope | None,
) -> dict[str, Any]:
    if envelope is None:
        return {
            "activated": False,
            "items": [],
        }
    items: list[dict[str, Any]] = []
    for pack in envelope.lane_packs:
        for admission in pack.admissions:
            fact = admission.union_fact
            items.append(compact_fact_prompt_projection(admission.alias, fact))
    return {
        "activated": True,
        "items": items,
    }


def _render_messages(
    planned: EvidenceMapPlanRow,
    mapped: VerifiedEvidenceMapRow,
    envelope: FactUnionEnvelope | None,
) -> tuple[FastProviderMessage, ...]:
    system = (
        "Solve the dated question using only VALIDATED_EVIDENCE_JSON. The direct "
        "parent prediction is a fallback candidate for assessment, not evidence. "
        "Map items and source facts have distinct exact evidence_id values and "
        "must not be conflated. Default to keep_parent. A map item may help "
        "assess the parent, but cannot authorize replacement by itself. Return "
        "one strict JSON object with exactly "
        "decision, prediction, used_evidence_ids and no markdown. decision is "
        "keep_parent, replace, or insufficient. keep_parent requires prediction "
        "byte-for-byte equal to the direct parent. replace requires a nonempty "
        "concise prediction, unique supplied evidence IDs, and at least one "
        "source-fact evidence ID. "
        "insufficient requires empty prediction and empty used_evidence_ids."
    )
    packet = planned.direct_plan_row.adapter.source.packet
    source_projection = _source_evidence_projection(envelope)
    source_ids = tuple(
        str(item["evidence_id"]) for item in source_projection["items"]
    )
    schema_example = (
        {
            "decision": "replace",
            "prediction": "concise prediction",
            "used_evidence_ids": [source_ids[0]],
        }
        if source_ids
        else {
            "decision": "keep_parent",
            "prediction": planned.direct_answer_row.prediction,
            "used_evidence_ids": [],
        }
    )
    user = (
        "QUESTION_BINDING_JSON:\n"
        + _json(
            {
                "dated_question_sha256": packet.dated_question_sha256,
                "question_id": packet.question_id,
                "question_sha256": packet.question_sha256,
            }
        )
        + "\n\nDATED_QUESTION_JSON:\n"
        + _json(packet.dated_question)
        + "\n\nQUESTION_ONLY_ROUTE_JSON:\n"
        + _json(_route_projection(planned.route))
        + "\n\nQUESTION_ONLY_OPERATOR_POLICY_JSON:\n"
        + _json(_operator_policy_projection(planned.route.style))
        + "\n\nVALIDATED_EVIDENCE_JSON:\n"
        + _json(
            {
                "map": _map_evidence_projection(mapped),
                "source_facts": source_projection,
            }
        )
        + "\n\nDIRECT_PARENT_FALLBACK_FOR_ASSESSMENT_JSON:\n"
        + _json(
            {
                "label": "fallback_for_assessment_not_evidence",
                "prediction": planned.direct_answer_row.prediction,
                "prediction_sha256": planned.direct_answer_row.prediction_sha256,
            }
        )
        + "\n\nSTRICT_SCHEMA_JSON:\n"
        + _json(schema_example)
        + "\n\nDECISION_JSON:"
    )
    return (
        FastProviderMessage(role="system", content=system),
        FastProviderMessage(role="user", content=user),
    )


@dataclass(frozen=True, slots=True)
class AdaptiveEvidenceSolverPlanRow:
    map_plan_row: EvidenceMapPlanRow
    map_row: VerifiedEvidenceMapRow
    fact_union: PostMapFactUnion | None
    fact_envelope: FactUnionEnvelope | None
    source_fact_validation_mode: str
    allowed_map_item_ids: tuple[str, ...]
    allowed_source_fact_ids: tuple[str, ...]
    messages: tuple[FastProviderMessage, ...] | None
    messages_sha256: str | None
    prompt_id: str | None
    prompt_token_proxy: int | None
    packet_id: str
    receipt_sha256: str
    disposition: StageDisposition
    reason: str
    retained_transformer_token_state_bytes: Literal[0] = 0

    @property
    def ordinal(self) -> int:
        return self.map_plan_row.ordinal

    @property
    def question_id(self) -> str:
        return self.map_row.question_id

    @property
    def submitted(self) -> bool:
        return self.messages is not None

    @property
    def activated(self) -> bool:
        return self.fact_envelope is not None

    @property
    def allowed_evidence_ids(self) -> tuple[str, ...]:
        return self.allowed_map_item_ids + self.allowed_source_fact_ids


@dataclass(frozen=True, slots=True)
class AdaptiveEvidenceSolverPlan:
    map_plan: EvidenceMapPlan
    map_plane: VerifiedEvidenceMapPlane
    rows: tuple[AdaptiveEvidenceSolverPlanRow, ...]
    prompt_population: FastPromptPopulation | None
    max_prompt_tokens: int
    output_token_reserve: int
    plan_identity_sha256: str
    retained_transformer_token_state_bytes: Literal[0] = 0

    @property
    def submitted_rows(self) -> tuple[AdaptiveEvidenceSolverPlanRow, ...]:
        return tuple(row for row in self.rows if row.submitted)

    @property
    def required_calls(self) -> int:
        return len(self.submitted_rows)


def _normalize_fact_inputs(
    values: Mapping[str, Any] | None,
    *,
    cls: type,
    label: str,
) -> dict[str, Any]:
    if values is None:
        return {}
    _require(isinstance(values, Mapping), f"{label} must be a question mapping")
    result: dict[str, Any] = {}
    for key, value in values.items():
        require_text(key, f"{label} question ID")
        _require(type(value) is cls, f"{label} values must be exact {cls.__name__}")
        _require(key not in result, f"{label} question IDs repeat")
        result[key] = value
    return result


def build_adaptive_evidence_solver_plan(
    map_plan: EvidenceMapPlan,
    map_plane: VerifiedEvidenceMapPlane,
    *,
    source_fact_unions: Mapping[str, PostMapFactUnion] | None = None,
    packed_fact_envelopes: Mapping[str, FactUnionEnvelope] | None = None,
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
    output_token_reserve: int = SOLVER_OUTPUT_TOKEN_RESERVE,
) -> AdaptiveEvidenceSolverPlan:
    """Build one distinct final prompt per V2-eligible or source-activated row."""

    if type(map_plan) is not EvidenceMapPlan:
        raise TypeError("map_plan must be an exact EvidenceMapPlan")
    if type(map_plane) is not VerifiedEvidenceMapPlane:
        raise TypeError("map_plane must be an exact VerifiedEvidenceMapPlane")
    _require(
        type(max_prompt_tokens) is int
        and max_prompt_tokens == MAX_PROMPT_TOKENS
        and type(output_token_reserve) is int
        and output_token_reserve == SOLVER_OUTPUT_TOKEN_RESERVE,
        "adaptive solver token envelope changed",
    )
    _validate_terminal_map(map_plan, map_plane)
    unions = _normalize_fact_inputs(
        source_fact_unions, cls=PostMapFactUnion, label="source fact unions"
    )
    envelopes = _normalize_fact_inputs(
        packed_fact_envelopes,
        cls=FactUnionEnvelope,
        label="packed fact envelopes",
    )
    question_ids = tuple(row.question_id for row in map_plane.rows)
    _require(
        len(set(question_ids)) == len(question_ids)
        and set(unions) | set(envelopes) <= set(question_ids),
        "source facts escaped or ambiguously matched the V2 question population",
    )
    rows: list[AdaptiveEvidenceSolverPlanRow] = []
    prompts: list[tuple[dict[str, str], ...]] = []
    for planned, mapped in zip(map_plan.rows, map_plane.rows, strict=True):
        packet = planned.direct_plan_row.adapter.source.packet
        _require(
            mapped.ordinal == planned.ordinal
            and mapped.question_id == packet.question_id
            and mapped.question_sha256 == packet.question_sha256
            and mapped.dated_question_sha256 == packet.dated_question_sha256
            and mapped.route_id == planned.route.style.value
            and mapped.map_plan_row_receipt_sha256 == planned.receipt_sha256
            and mapped.direct_parent_prediction_sha256
            == planned.direct_answer_row.prediction_sha256
            and planned.route.question_sha256 == packet.dated_question_sha256,
            f"adaptive solver map/question/route binding changed at ordinal {planned.ordinal}",
        )
        for item in mapped.accepted_items:
            _validate_map_item(item)
        map_ids = tuple(item.item_id for item in mapped.accepted_items)
        _require(len(set(map_ids)) == len(map_ids), "validated map evidence IDs repeat")

        fact_union = unions.get(mapped.question_id)
        supplied_envelope = envelopes.get(mapped.question_id)
        if fact_union is not None:
            _validate_fact_union(fact_union, map_plan, planned)

        base_messages = _render_messages(planned, mapped, None)
        base_tokens = count_chat_prompt_token_proxy(_plain_messages(base_messages))
        if fact_union is not None:
            generated = pack_fact_union_envelope(
                fact_union, parent_prompt_token_proxy=base_tokens
            )
            if supplied_envelope is not None:
                _require(
                    supplied_envelope == generated,
                    "supplied fact envelope differs from deterministic post-map packing",
                )
            envelope = generated
            validation_mode = "post_map_union_repacked"
        elif supplied_envelope is not None:
            envelope = supplied_envelope
            validation_mode = "sealed_fact_envelope_revalidated"
        else:
            envelope = None
            validation_mode = "no_source_gate_activation"
        if envelope is not None:
            _validate_fact_envelope(
                envelope,
                map_plan,
                planned,
                parent_prompt_token_proxy=base_tokens,
            )
            if fact_union is not None:
                _require(
                    envelope.fact_union_receipt_sha256 == fact_union.receipt_sha256,
                    "fact envelope lost its post-map union binding",
                )
        source_ids = tuple(
            admission.alias
            for pack in (() if envelope is None else envelope.lane_packs)
            for admission in pack.admissions
        )
        _require(
            len(set(source_ids)) == len(source_ids)
            and not set(source_ids) & set(map_ids),
            "map and source evidence IDs are not distinct",
        )

        # A validated map item can assess the parent but cannot authorize a
        # replacement.  Consequently, a final provider call is actionable only
        # when deterministic packing admitted at least one source-fact alias.
        # Empty/map-only rows preserve the exact direct parent provider-free.
        should_submit = bool(source_ids)
        _require(
            planned.route.style in ELIGIBLE_STYLES | PRESERVED_STYLES,
            "adaptive solver encountered an unknown question-only route",
        )
        packet_id = _seal(
            "packet",
            {
                "fact_envelope_receipt_sha256": (
                    None if envelope is None else envelope.receipt_sha256
                ),
                "map_item_sha256s": [
                    item.item_sha256 for item in mapped.accepted_items
                ],
                "map_source_row_sha256": mapped.source_row_sha256,
                "parent_map_packet_id": planned.packet_id,
                "source_fact_ids": list(source_ids),
            },
        )
        if should_submit:
            messages = _render_messages(planned, mapped, envelope)
            plain = _plain_messages(messages)
            prompt_tokens = count_chat_prompt_token_proxy(plain)
            _require(
                prompt_tokens + output_token_reserve <= max_prompt_tokens,
                "adaptive final prompt plus fixed output reserve exceeds 8k at "
                f"ordinal {planned.ordinal}: {prompt_tokens}+{output_token_reserve}",
            )
            messages_sha = identity_sha256(list(plain))
            prompt_id = _seal(
                "prompt-id",
                {
                    "fact_envelope_receipt_sha256": (
                        None if envelope is None else envelope.receipt_sha256
                    ),
                    "map_parse_receipt_sha256": mapped.map_parse_receipt_sha256,
                    "messages_sha256": messages_sha,
                    "question_id": mapped.question_id,
                    "renderer_id": RENDERER_ID,
                },
            )
            reason = (
                "eligible_question_with_validated_map_and_source_facts"
                if envelope is not None
                else "eligible_question_with_validated_map"
            )
            body = {
                "allowed_map_item_ids": list(map_ids),
                "allowed_source_fact_ids": list(source_ids),
                "disposition": StageDisposition.ADDED.value,
                "fact_envelope_receipt_sha256": (
                    None if envelope is None else envelope.receipt_sha256
                ),
                "map_source_row_sha256": mapped.source_row_sha256,
                "messages_sha256": messages_sha,
                "output_token_reserve": output_token_reserve,
                "packet_id": packet_id,
                "prompt_id": prompt_id,
                "prompt_token_proxy": prompt_tokens,
                "reason": reason,
                "source_fact_validation_mode": validation_mode,
            }
            row = AdaptiveEvidenceSolverPlanRow(
                planned,
                mapped,
                fact_union,
                envelope,
                validation_mode,
                map_ids,
                source_ids,
                messages,
                messages_sha,
                prompt_id,
                prompt_tokens,
                packet_id,
                _seal("plan-row", body),
                StageDisposition.ADDED,
                reason,
            )
            prompts.append(plain)
        else:
            reason = (
                "source_gate_activated_without_admitted_source_fact"
                if envelope is not None
                else "no_admitted_source_fact"
            )
            body = {
                "allowed_map_item_ids": list(map_ids),
                "allowed_source_fact_ids": [],
                "disposition": StageDisposition.NO_OP.value,
                "fact_envelope_receipt_sha256": (
                    None if envelope is None else envelope.receipt_sha256
                ),
                "map_source_row_sha256": mapped.source_row_sha256,
                "packet_id": packet_id,
                "reason": reason,
                "source_fact_validation_mode": validation_mode,
            }
            row = AdaptiveEvidenceSolverPlanRow(
                planned,
                mapped,
                fact_union,
                envelope,
                validation_mode,
                map_ids,
                (),
                None,
                None,
                None,
                None,
                packet_id,
                _seal("plan-row", body),
                StageDisposition.NO_OP,
                reason,
            )
        rows.append(row)
    population = (
        preflight_fast_completion_prompts(
            prompts, max_prompt_tokens=max_prompt_tokens
        )
        if prompts
        else None
    )
    if population is not None:
        _require(
            population.logical_prompt_count
            == population.unique_prompt_count
            == len(prompts),
            "adaptive final prompts are not a one-to-one question population",
        )
    body = {
        "arm_label": ARM_LABEL,
        "map_plan_identity_sha256": map_plan.plan_identity_sha256,
        "map_run_sha256": map_plane.run_sha256,
        "map_runtime_ledger_sha256": map_plane.runtime_ledger_sha256,
        "max_prompt_tokens": max_prompt_tokens,
        "output_token_reserve": output_token_reserve,
        "plan_id": PLAN_ID,
        "renderer_id": RENDERER_ID,
        "row_receipt_sha256s": [row.receipt_sha256 for row in rows],
    }
    assert_gold_blind(body, path="adaptive_evidence_solver_plan")
    return AdaptiveEvidenceSolverPlan(
        map_plan,
        map_plane,
        tuple(rows),
        population,
        max_prompt_tokens,
        output_token_reserve,
        _seal("plan", body),
    )


@dataclass(frozen=True, slots=True)
class AdaptiveEvidenceSolverPreflight:
    plan_identity_sha256: str
    map_run_sha256: str
    map_replay_sha256: str
    map_runtime_ledger_sha256: str
    prompt_population_sha256: str
    required_authorized_provider_calls: int
    ordered_row_receipt_sha256s: tuple[str, ...]
    submitted_prompt_ids: tuple[str, ...]
    observed_max_prompt_token_proxy: int
    max_prompt_tokens: int
    output_token_reserve: int
    receipt_sha256: str
    provider_calls_executed: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0

    def projection(self) -> dict[str, Any]:
        return {
            "arm_label": ARM_LABEL,
            "format": PREFLIGHT_FORMAT,
            "gold_loaded": False,
            "hard_prompt_token_cap": self.max_prompt_tokens,
            "map_replay_sha256": self.map_replay_sha256,
            "map_run_sha256": self.map_run_sha256,
            "map_runtime_ledger_sha256": self.map_runtime_ledger_sha256,
            "observed_max_prompt_token_proxy": self.observed_max_prompt_token_proxy,
            "ordered_row_receipt_sha256s": list(
                self.ordered_row_receipt_sha256s
            ),
            "output_token_reserve": self.output_token_reserve,
            "plan_id": PLAN_ID,
            "plan_identity_sha256": self.plan_identity_sha256,
            "prompt_population_sha256": self.prompt_population_sha256,
            "provider_calls_executed": self.provider_calls_executed,
            "renderer_id": RENDERER_ID,
            "required_authorized_provider_calls": (
                self.required_authorized_provider_calls
            ),
            "retained_transformer_token_state_bytes": (
                self.retained_transformer_token_state_bytes
            ),
            "submitted_prompt_ids": list(self.submitted_prompt_ids),
        }


def preflight_adaptive_evidence_solver(
    plan: AdaptiveEvidenceSolverPlan,
) -> AdaptiveEvidenceSolverPreflight:
    """Seal the full prompt/call population without executing it."""

    if type(plan) is not AdaptiveEvidenceSolverPlan:
        raise TypeError("plan must be an exact AdaptiveEvidenceSolverPlan")
    _validate_terminal_map(plan.map_plan, plan.map_plane)
    _exact_zero(
        plan.retained_transformer_token_state_bytes,
        "adaptive plan retained transformer state",
    )
    population_sha = (
        _seal("empty-prompt-population", {"max_prompt_tokens": plan.max_prompt_tokens})
        if plan.prompt_population is None
        else plan.prompt_population.prompt_population_sha256
    )
    body = {
        "map_replay_sha256": plan.map_plane.replay_sha256,
        "map_run_sha256": plan.map_plane.run_sha256,
        "map_runtime_ledger_sha256": plan.map_plane.runtime_ledger_sha256,
        "observed_max_prompt_token_proxy": max(
            (row.prompt_token_proxy or 0 for row in plan.submitted_rows),
            default=0,
        ),
        "ordered_row_receipt_sha256s": [row.receipt_sha256 for row in plan.rows],
        "output_token_reserve": plan.output_token_reserve,
        "plan_identity_sha256": plan.plan_identity_sha256,
        "prompt_population_sha256": population_sha,
        "required_authorized_provider_calls": plan.required_calls,
        "submitted_prompt_ids": [
            require_sha256(row.prompt_id, "adaptive prompt ID")
            for row in plan.submitted_rows
        ],
    }
    receipt = _seal("preflight", body)
    result = AdaptiveEvidenceSolverPreflight(
        plan.plan_identity_sha256,
        plan.map_plane.run_sha256,
        plan.map_plane.replay_sha256,
        plan.map_plane.runtime_ledger_sha256,
        population_sha,
        plan.required_calls,
        tuple(row.receipt_sha256 for row in plan.rows),
        tuple(row.prompt_id for row in plan.submitted_rows if row.prompt_id is not None),
        max((row.prompt_token_proxy or 0 for row in plan.submitted_rows), default=0),
        plan.max_prompt_tokens,
        plan.output_token_reserve,
        receipt,
    )
    assert_gold_blind(result.projection(), path="adaptive_solver_preflight")
    return result


@dataclass(frozen=True, slots=True)
class AdaptiveSolverCompletion:
    ordinal: int
    question_id: str
    prompt_id: str
    messages_sha256: str
    completion: str
    completion_sha256: str
    request_receipt_sha256: str
    response_receipt_sha256: str
    receipt_sha256: str
    retained_transformer_token_state_bytes: Literal[0] = 0


@dataclass(frozen=True, slots=True)
class AdaptiveSolverCompletionPlane:
    plan_identity_sha256: str
    preflight_receipt_sha256: str
    rows: tuple[AdaptiveSolverCompletion, ...]
    receipt_sha256: str
    provider_calls_executed_by_ingest: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0


def capture_adaptive_solver_completions(
    plan: AdaptiveEvidenceSolverPlan,
    preflight: AdaptiveEvidenceSolverPreflight,
    completions_by_question: Mapping[str, str],
) -> AdaptiveSolverCompletionPlane:
    """Ingest externally obtained exact completions; this function calls nobody."""

    if type(plan) is not AdaptiveEvidenceSolverPlan:
        raise TypeError("plan must be an exact AdaptiveEvidenceSolverPlan")
    if type(preflight) is not AdaptiveEvidenceSolverPreflight:
        raise TypeError("preflight must be an exact AdaptiveEvidenceSolverPreflight")
    _require(
        preflight.plan_identity_sha256 == plan.plan_identity_sha256
        and preflight.receipt_sha256
        == preflight_adaptive_evidence_solver(plan).receipt_sha256,
        "completion ingest preflight differs from the exact plan",
    )
    _require(
        isinstance(completions_by_question, Mapping)
        and set(completions_by_question)
        == {row.question_id for row in plan.submitted_rows},
        "completion population must exactly equal submitted adaptive questions",
    )
    rows: list[AdaptiveSolverCompletion] = []
    for planned in plan.submitted_rows:
        completion = completions_by_question[planned.question_id]
        _require(type(completion) is str, "completion must be exact text")
        assert planned.prompt_id is not None and planned.messages_sha256 is not None
        completion_sha = quote_sha256(completion)
        request = _seal(
            "completion-request",
            {
                "messages_sha256": planned.messages_sha256,
                "ordinal": planned.ordinal,
                "prompt_id": planned.prompt_id,
                "question_id": planned.question_id,
            },
        )
        response = _seal(
            "completion-response",
            {
                "completion_sha256": completion_sha,
                "request_receipt_sha256": request,
            },
        )
        body = {
            "completion_sha256": completion_sha,
            "messages_sha256": planned.messages_sha256,
            "ordinal": planned.ordinal,
            "prompt_id": planned.prompt_id,
            "question_id": planned.question_id,
            "request_receipt_sha256": request,
            "response_receipt_sha256": response,
            "retained_transformer_token_state_bytes": 0,
        }
        rows.append(
            AdaptiveSolverCompletion(
                planned.ordinal,
                planned.question_id,
                planned.prompt_id,
                planned.messages_sha256,
                completion,
                completion_sha,
                request,
                response,
                _seal("completion", body),
            )
        )
    body = {
        "completion_receipt_sha256s": [row.receipt_sha256 for row in rows],
        "plan_identity_sha256": plan.plan_identity_sha256,
        "preflight_receipt_sha256": preflight.receipt_sha256,
        "provider_calls_executed_by_ingest": 0,
        "retained_transformer_token_state_bytes": 0,
    }
    return AdaptiveSolverCompletionPlane(
        plan.plan_identity_sha256,
        preflight.receipt_sha256,
        tuple(rows),
        _seal("completion-plane", body),
    )


@dataclass(frozen=True, slots=True)
class ParsedAdaptiveSolverDecision:
    valid: bool
    decision: str
    prediction: str
    used_evidence_ids: tuple[str, ...]
    error_code: str
    receipt_sha256: str


def _invalid_decision(code: str) -> ParsedAdaptiveSolverDecision:
    return ParsedAdaptiveSolverDecision(
        False,
        "invalid",
        "",
        (),
        code,
        _seal("invalid-parse", {"error_code": code}),
    )


def parse_adaptive_solver_completion(
    completion: str,
    *,
    allowed_evidence_ids: Sequence[str],
    replacement_evidence_ids: Sequence[str],
    parent_prediction: str,
) -> ParsedAdaptiveSolverDecision:
    """Validate the exact three-field decision and its evidence lifecycle."""

    if type(completion) is not str:
        raise TypeError("completion must be exact text")
    if type(parent_prediction) is not str or not parent_prediction:
        raise TypeError("parent_prediction must be nonempty exact text")
    allowed = tuple(allowed_evidence_ids)
    replacement = tuple(replacement_evidence_ids)
    _require(
        all(type(value) is str and bool(value) for value in allowed)
        and len(set(allowed)) == len(allowed),
        "allowed adaptive evidence IDs repeat or changed type",
    )
    _require(
        all(type(value) is str and bool(value) for value in replacement)
        and len(set(replacement)) == len(replacement)
        and set(replacement) <= set(allowed),
        "replacement evidence IDs repeat or escape allowed evidence",
    )
    try:
        raw = json.loads(
            completion,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (json.JSONDecodeError, ValueError):
        return _invalid_decision("invalid_json")
    if type(raw) is not dict or set(raw) != {
        "decision",
        "prediction",
        "used_evidence_ids",
    }:
        return _invalid_decision("root_schema")
    decision, prediction, used = (
        raw["decision"],
        raw["prediction"],
        raw["used_evidence_ids"],
    )
    if (
        type(decision) is not str
        or type(prediction) is not str
        or type(used) is not list
        or any(type(value) is not str for value in used)
        or len(set(used)) != len(used)
    ):
        return _invalid_decision("values")
    if any(value not in set(allowed) for value in used):
        return _invalid_decision("unknown_evidence_id")
    if decision == "replace":
        if not prediction or prediction.strip() != prediction or not used:
            return _invalid_decision("replace_contract")
        if not set(used) & set(replacement):
            return _invalid_decision("replace_requires_source_fact")
    elif decision == "keep_parent":
        if prediction != parent_prediction:
            return _invalid_decision("keep_parent_contract")
    elif decision == "insufficient":
        if prediction != "" or used:
            return _invalid_decision("insufficient_contract")
    else:
        return _invalid_decision("decision")
    body = {
        "decision": decision,
        "prediction_sha256": quote_sha256(prediction),
        "used_evidence_ids": list(used),
    }
    return ParsedAdaptiveSolverDecision(
        True,
        decision,
        prediction,
        tuple(used),
        "none",
        _seal("parse", body),
    )


@dataclass(frozen=True, slots=True)
class AdaptiveEvidenceSolverResultRow:
    ordinal: int
    question_id: str
    question_sha256: str
    dated_question_sha256: str
    prediction: str
    prediction_sha256: str
    prediction_source: str
    parent_prediction_sha256: str
    changed_from_parent: bool
    solver_valid: bool
    solver_decision: str
    solver_used_evidence_ids: tuple[str, ...]
    solver_used_map_item_ids: tuple[str, ...]
    solver_used_source_fact_ids: tuple[str, ...]
    solver_parse_receipt_sha256: str | None
    plan_row_receipt_sha256: str
    completion_receipt_sha256: str | None
    receipt_sha256: str
    provider_calls_executed_by_materializer: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0


@dataclass(frozen=True, slots=True)
class AdaptiveEvidenceSolverRun:
    plan_identity_sha256: str
    preflight_receipt_sha256: str
    completion_plane_receipt_sha256: str
    rows: tuple[AdaptiveEvidenceSolverResultRow, ...]
    receipt_sha256: str
    provider_calls_executed_by_materializer: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0


def _validate_completion_plane(
    plan: AdaptiveEvidenceSolverPlan,
    preflight: AdaptiveEvidenceSolverPreflight,
    plane: AdaptiveSolverCompletionPlane,
) -> None:
    _require(type(plane) is AdaptiveSolverCompletionPlane, "completion plane changed type")
    _exact_zero(plane.provider_calls_executed_by_ingest, "completion ingest calls")
    _exact_zero(
        plane.retained_transformer_token_state_bytes,
        "completion plane retained transformer state",
    )
    _require(
        plane.plan_identity_sha256 == plan.plan_identity_sha256
        and plane.preflight_receipt_sha256 == preflight.receipt_sha256
        and len(plane.rows) == len(plan.submitted_rows),
        "completion plane escaped its plan/preflight",
    )
    for planned, completion in zip(plan.submitted_rows, plane.rows, strict=True):
        _require(type(completion) is AdaptiveSolverCompletion, "completion row changed type")
        _exact_zero(
            completion.retained_transformer_token_state_bytes,
            "completion retained transformer state",
        )
        assert planned.prompt_id is not None and planned.messages_sha256 is not None
        _require(
            completion.ordinal == planned.ordinal
            and completion.question_id == planned.question_id
            and completion.prompt_id == planned.prompt_id
            and completion.messages_sha256 == planned.messages_sha256
            and completion.completion_sha256 == quote_sha256(completion.completion),
            "completion row changed prompt/response binding",
        )
        request = _seal(
            "completion-request",
            {
                "messages_sha256": planned.messages_sha256,
                "ordinal": planned.ordinal,
                "prompt_id": planned.prompt_id,
                "question_id": planned.question_id,
            },
        )
        response = _seal(
            "completion-response",
            {
                "completion_sha256": completion.completion_sha256,
                "request_receipt_sha256": request,
            },
        )
        body = {
            "completion_sha256": completion.completion_sha256,
            "messages_sha256": planned.messages_sha256,
            "ordinal": planned.ordinal,
            "prompt_id": planned.prompt_id,
            "question_id": planned.question_id,
            "request_receipt_sha256": request,
            "response_receipt_sha256": response,
            "retained_transformer_token_state_bytes": 0,
        }
        _require(
            completion.request_receipt_sha256 == request
            and completion.response_receipt_sha256 == response
            and completion.receipt_sha256 == _seal("completion", body),
            "completion lifecycle receipt changed",
        )
    body = {
        "completion_receipt_sha256s": [row.receipt_sha256 for row in plane.rows],
        "plan_identity_sha256": plan.plan_identity_sha256,
        "preflight_receipt_sha256": preflight.receipt_sha256,
        "provider_calls_executed_by_ingest": 0,
        "retained_transformer_token_state_bytes": 0,
    }
    _require(
        plane.receipt_sha256 == _seal("completion-plane", body),
        "completion plane receipt changed",
    )


def materialize_adaptive_evidence_solver(
    plan: AdaptiveEvidenceSolverPlan,
    preflight: AdaptiveEvidenceSolverPreflight,
    completions: AdaptiveSolverCompletionPlane,
) -> AdaptiveEvidenceSolverRun:
    """Materialize decisions with byte-exact parent fallback and no I/O."""

    if type(plan) is not AdaptiveEvidenceSolverPlan:
        raise TypeError("plan must be an exact AdaptiveEvidenceSolverPlan")
    if type(preflight) is not AdaptiveEvidenceSolverPreflight:
        raise TypeError("preflight must be an exact AdaptiveEvidenceSolverPreflight")
    _require(
        preflight.receipt_sha256
        == preflight_adaptive_evidence_solver(plan).receipt_sha256,
        "materializer preflight changed",
    )
    _validate_completion_plane(plan, preflight, completions)
    completion_by_ordinal = {row.ordinal: row for row in completions.rows}
    rows: list[AdaptiveEvidenceSolverResultRow] = []
    for planned in plan.rows:
        parent = planned.map_plan_row.direct_answer_row
        completion = completion_by_ordinal.get(planned.ordinal)
        if planned.submitted:
            _require(completion is not None, "submitted row lost its completion")
            assert completion is not None
            parsed = parse_adaptive_solver_completion(
                completion.completion,
                allowed_evidence_ids=planned.allowed_evidence_ids,
                replacement_evidence_ids=planned.allowed_source_fact_ids,
                parent_prediction=parent.prediction,
            )
            if parsed.valid and parsed.decision == "replace":
                prediction = parsed.prediction
                source = "adaptive_validated_evidence_replacement_v3"
            elif parsed.valid and parsed.decision == "keep_parent":
                prediction = parent.prediction
                source = "adaptive_validated_evidence_keep_parent_v3"
            else:
                prediction = parent.prediction
                source = "sealed_direct_query_fallback"
            decision = parsed.decision
            valid = parsed.valid
            used = parsed.used_evidence_ids
            parse_receipt: str | None = parsed.receipt_sha256
            completion_receipt: str | None = completion.receipt_sha256
        else:
            _require(completion is None, "no-op row acquired a completion")
            prediction = parent.prediction
            source = "sealed_direct_query_fallback"
            decision = "not_submitted"
            valid = False
            used = ()
            parse_receipt = None
            completion_receipt = None
        prediction_sha = quote_sha256(prediction)
        used_map = tuple(value for value in used if value in set(planned.allowed_map_item_ids))
        used_source = tuple(
            value for value in used if value in set(planned.allowed_source_fact_ids)
        )
        _require(
            len(used_map) + len(used_source) == len(used),
            "materialized decision cites evidence outside its exact planes",
        )
        body = {
            "changed_from_parent": prediction_sha != parent.prediction_sha256,
            "completion_receipt_sha256": completion_receipt,
            "dated_question_sha256": planned.map_row.dated_question_sha256,
            "ordinal": planned.ordinal,
            "parent_prediction_sha256": parent.prediction_sha256,
            "plan_row_receipt_sha256": planned.receipt_sha256,
            "prediction_sha256": prediction_sha,
            "prediction_source": source,
            "question_id": planned.question_id,
            "question_sha256": planned.map_row.question_sha256,
            "solver_decision": decision,
            "solver_parse_receipt_sha256": parse_receipt,
            "solver_used_evidence_ids": list(used),
            "solver_used_map_item_ids": list(used_map),
            "solver_used_source_fact_ids": list(used_source),
            "solver_valid": valid,
        }
        rows.append(
            AdaptiveEvidenceSolverResultRow(
                planned.ordinal,
                planned.question_id,
                planned.map_row.question_sha256,
                planned.map_row.dated_question_sha256,
                prediction,
                prediction_sha,
                source,
                parent.prediction_sha256,
                prediction_sha != parent.prediction_sha256,
                valid,
                decision,
                used,
                used_map,
                used_source,
                parse_receipt,
                planned.receipt_sha256,
                completion_receipt,
                _seal("result-row", body),
            )
        )
    body = {
        "completion_plane_receipt_sha256": completions.receipt_sha256,
        "plan_identity_sha256": plan.plan_identity_sha256,
        "preflight_receipt_sha256": preflight.receipt_sha256,
        "provider_calls_executed_by_materializer": 0,
        "result_row_receipt_sha256s": [row.receipt_sha256 for row in rows],
        "retained_transformer_token_state_bytes": 0,
    }
    return AdaptiveEvidenceSolverRun(
        plan.plan_identity_sha256,
        preflight.receipt_sha256,
        completions.receipt_sha256,
        tuple(rows),
        _seal("run", body),
    )


@dataclass(frozen=True, slots=True)
class VerifiedAdaptiveEvidenceSolverPlane:
    run_receipt_sha256: str
    replay_receipt_sha256: str
    plan_identity_sha256: str
    preflight_receipt_sha256: str
    completion_plane_receipt_sha256: str
    rows: tuple[AdaptiveEvidenceSolverResultRow, ...]
    receipt_sha256: str
    provider_calls_executed_by_replay: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0


def replay_adaptive_evidence_solver(
    plan: AdaptiveEvidenceSolverPlan,
    preflight: AdaptiveEvidenceSolverPreflight,
    completions: AdaptiveSolverCompletionPlane,
    run: AdaptiveEvidenceSolverRun,
) -> VerifiedAdaptiveEvidenceSolverPlane:
    """Recompute the full materialization and require byte-identical receipts."""

    if type(run) is not AdaptiveEvidenceSolverRun:
        raise TypeError("run must be an exact AdaptiveEvidenceSolverRun")
    replay = materialize_adaptive_evidence_solver(plan, preflight, completions)
    _require(run == replay, "adaptive solver run differs from deterministic replay")
    body = {
        "completion_plane_receipt_sha256": completions.receipt_sha256,
        "plan_identity_sha256": plan.plan_identity_sha256,
        "preflight_receipt_sha256": preflight.receipt_sha256,
        "provider_calls_executed_by_replay": 0,
        "replay_receipt_sha256": replay.receipt_sha256,
        "retained_transformer_token_state_bytes": 0,
        "run_receipt_sha256": run.receipt_sha256,
    }
    return VerifiedAdaptiveEvidenceSolverPlane(
        run.receipt_sha256,
        replay.receipt_sha256,
        plan.plan_identity_sha256,
        preflight.receipt_sha256,
        completions.receipt_sha256,
        run.rows,
        _seal("verified-plane", body),
    )


__all__ = [
    "ARM_LABEL",
    "FORMAT",
    "PLAN_ID",
    "PREFLIGHT_FORMAT",
    "RENDERER_ID",
    "AdaptiveEvidenceSolverError",
    "AdaptiveEvidenceSolverPlan",
    "AdaptiveEvidenceSolverPlanRow",
    "AdaptiveEvidenceSolverPreflight",
    "AdaptiveEvidenceSolverResultRow",
    "AdaptiveEvidenceSolverRun",
    "AdaptiveSolverCompletion",
    "AdaptiveSolverCompletionPlane",
    "ParsedAdaptiveSolverDecision",
    "VerifiedAdaptiveEvidenceSolverPlane",
    "build_adaptive_evidence_solver_plan",
    "capture_adaptive_solver_completions",
    "materialize_adaptive_evidence_solver",
    "parse_adaptive_solver_completion",
    "preflight_adaptive_evidence_solver",
    "replay_adaptive_evidence_solver",
]
