#!/usr/bin/env python3
"""Build v7 compact, fact-reserved episodic packets over the sealed locked100.

This is an additive successor to the sealed v6/r3 assay.  It deliberately leaves
that implementation byte-identical so historical artifacts remain authentic.
The successor reuses v6's immutable candidate scoring and integrity helpers, but
selects specialist and physical lanes independently, performs exact-ID dedup only
after both selections, reserves a compact fact slice before optional episode
expansion, and keeps opaque provenance in a sealed audit manifest rather than the
provider-visible prompt.  This program loads no benchmark labels and performs no
provider I/O.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import re
import sys
import threading
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.integrity import file_sha256
from tools import assay_hot_retrieval_1m as hot
from tools import assay_hot_v6_spine_episode_fact_ledger_full100 as legacy
from tools.matched_eval import hot_v6_query_fact_ledger as fact_ledger_contract
from tools.matched_eval.contracts import assert_gold_blind, identity_sha256
from tools.matched_eval.hot_v3_user_led_envelope_shadow import (
    UserLedEnvelopeShadowIndex,
)
from tools.matched_eval.hot_v6_query_fact_ledger import (
    FactLedgerSlice,
    QueryFactLedger,
    QueryFactLedgerError,
    compile_query_fact_ledger,
    select_and_render_query_fact_ledger,
)
from tools.matched_eval.hot_v6_typed_reducer_advisory import (
    compile_hot_v6_typed_reducer_advisory,
)
from tools.matched_eval.typed_numeric_semantics import (
    NumericDimension,
    expected_numeric_dimension,
)
from tools.matched_eval.typed_operator_spec import (
    TemporalMode,
    TypedOperatorSpec,
    normalized_terms,
)


FORMAT = "memory-condense-hot-v7-spine-episode-fact-reserved-selection-v6"
ROW_FORMAT = f"{FORMAT}-row-v1"
SELECTION_NAME = legacy.SELECTION_NAME
REPLAY_NAME = legacy.REPLAY_NAME
EXPECTED_QUESTION_COUNT = legacy.EXPECTED_QUESTION_COUNT
EXPECTED_POPULATION_SHA256 = legacy.EXPECTED_POPULATION_SHA256
MAX_CONTEXT_TOKENS = legacy.MAX_CONTEXT_TOKENS
MAX_WORKSPACE_TOKENS = legacy.MAX_WORKSPACE_TOKENS
OUTPUT_TOKEN_RESERVE = legacy.OUTPUT_TOKEN_RESERVE
MAX_EPISODES = legacy.MAX_EPISODES
MAX_EPISODES_PER_SOURCE = legacy.MAX_EPISODES_PER_SOURCE
MAX_EPISODE_RAW_CHUNKS = legacy.MAX_EPISODE_RAW_CHUNKS
MAX_EPISODE_RAW_TOKENS = legacy.MAX_EPISODE_RAW_TOKENS
MAX_FACTS = legacy.MAX_FACTS
MAX_FACT_TOKENS = legacy.MAX_FACT_TOKENS
INITIAL_FACT_SELECTION_AUDIT_TOKENS = 2_048
MAX_NUMERIC_COMPLETION_SOURCES_PER_SLOT = 1
MAX_NUMERIC_COMPLETION_ROWS_PER_SLOT_SOURCE = 1
MAX_NUMERIC_COMPLETION_ENVELOPES = 4
MAX_NUMERIC_COMPLETION_TURNS = 8
LEXICAL_LANE_EPISODES = legacy.LEXICAL_LANE_EPISODES
ANCHOR_LANE_EPISODES = legacy.ANCHOR_LANE_EPISODES
SLOT_LANE_EPISODES = legacy.SLOT_LANE_EPISODES
TYPED_LANE_EPISODES = legacy.TYPED_LANE_EPISODES
PHYSICAL_OWNER_LANE_EPISODES = legacy.PHYSICAL_OWNER_LANE_EPISODES
PHYSICAL_TRANSITION_LANE_EPISODES = legacy.PHYSICAL_TRANSITION_LANE_EPISODES
EPISODE_LANE_TOKEN_BUDGET = legacy.EPISODE_LANE_TOKEN_BUDGET
EPISODE_LANE_CHUNK_BUDGET = legacy.EPISODE_LANE_CHUNK_BUDGET
PHYSICAL_OWNER_LANE_TOKEN_BUDGET = legacy.PHYSICAL_OWNER_LANE_TOKEN_BUDGET
PHYSICAL_OWNER_LANE_CHUNK_BUDGET = legacy.PHYSICAL_OWNER_LANE_CHUNK_BUDGET
PHYSICAL_TRANSITION_LANE_TOKEN_BUDGET = (
    legacy.PHYSICAL_TRANSITION_LANE_TOKEN_BUDGET
)
PHYSICAL_TRANSITION_LANE_CHUNK_BUDGET = (
    legacy.PHYSICAL_TRANSITION_LANE_CHUNK_BUDGET
)

DEFAULT_SOURCE_ROOT = legacy.DEFAULT_SOURCE_ROOT
DEFAULT_STORE_ROOT = legacy.DEFAULT_STORE_ROOT
DEFAULT_RETRIEVAL = legacy.DEFAULT_RETRIEVAL
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-hot-v7-spine-episode-fact-reserved-full100-20260908-r9"
)

USER_TEMPLATE = (
    "Retrieved memory. <G#> blocks are protected global evidence. <E#> blocks "
    "are isolated single-source user-led exchanges; local <A# owner=...> rows "
    "belong to the cited user lead. <REF G#> reuses exact global evidence. "
    "<F#> entries are exact quote facts whose backs= label cites rendered raw "
    "evidence. Full opaque provenance remains in the sealed audit manifest. "
    "Do not merge blocks or sources.\n{context}\n\n"
    "Question: {question}\nShort answer:"
)

_LEGACY_SCOPE_LOCK = threading.RLock()
_LEGACY_IMPLEMENTATION_IDENTITY = legacy._implementation_identity  # noqa: SLF001
_SPECIALIST_LANES = (
    "required_slot",
    "typed_operation",
    "query_user_lead",
    "source_local_anchor",
)
_PHYSICAL_LANES = ("physical_anchor_owner", "physical_anchor_transition")
_NUMERIC_RELATION_UNIT_TERMS = {
    NumericDimension.COUNT: frozenset({"count", "each", "many", "number", "total"}),
    NumericDimension.CURRENCY: frozenset(
        {
            "accommodation",
            "cost",
            "each",
            "hotel",
            "night",
            "nightly",
            "pay",
            "paid",
            "per",
            "price",
            "room",
            "spend",
            "spent",
            "total",
        }
    ),
    NumericDimension.DURATION: frozenset(
        {"day", "hour", "long", "minute", "month", "second", "since", "week", "year"}
    ),
    NumericDimension.MEASURE: frozenset(
        {
            "feet",
            "foot",
            "gram",
            "inch",
            "kg",
            "kilogram",
            "kilometer",
            "km",
            "lb",
            "meter",
            "mile",
            "ounce",
            "pound",
        }
    ),
    NumericDimension.PERCENTAGE: frozenset(
        {"discount", "percent", "percentage", "rate"}
    ),
    NumericDimension.GENERIC: frozenset(
        {"cost", "each", "number", "pay", "per", "price", "total"}
    ),
}


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ValueError(message)


@contextmanager
def _legacy_scope(**updates: object):
    """Temporarily parameterize the immutable v6 selector under one process lock."""

    with _LEGACY_SCOPE_LOCK:
        previous = {name: getattr(legacy, name) for name in updates}
        try:
            for name, value in updates.items():
                setattr(legacy, name, value)
            yield
        finally:
            for name, value in previous.items():
                setattr(legacy, name, value)


def _implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    legacy_identity = _LEGACY_IMPLEMENTATION_IDENTITY()
    files = {
        **legacy_identity["files"],
        "tools/assay_hot_v7_spine_episode_fact_reserved_full100.py": file_sha256(
            root / "tools/assay_hot_v7_spine_episode_fact_reserved_full100.py"
        ),
    }
    for path in (
        "tools/matched_eval/hot_v6_typed_reducer.py",
        "tools/matched_eval/hot_v6_typed_reducer_advisory.py",
        "tools/matched_eval/numeric_evidence_reconciler.py",
        "tools/matched_eval/numeric_evidence_reconciler_v2.py",
        "tools/matched_eval/operator_first_numeric_policy.py",
        "tools/matched_eval/temporal_event_reconciler.py",
        "tools/matched_eval/typed_memory_final_arm.py",
        "tools/matched_eval/typed_operator_adapter.py",
        "tools/matched_eval/typed_operator_executor.py",
    ):
        files[path] = file_sha256(root / path)
    return {
        "files": files,
        "format": "memory-condense-hot-v7-spine-episode-implementation-v4",
        "legacy_v6_implementation_sha256": legacy_identity["sha256"],
        "sha256": identity_sha256(
            [{"path": path, "sha256": files[path]} for path in sorted(files)]
        ),
    }


def _transition_episode_cap(spec: TypedOperatorSpec) -> int:
    """Use broad adjacency only when the typed obligation needs a frontier."""

    complex_obligation = (
        spec.temporal_mode is not TemporalMode.NONE
        or (spec.requires_all_slots and bool(spec.required_slots))
        or spec.requires_complete_frontier
        or len(spec.required_slots) > 1
    )
    requested = 4 if complex_obligation else 1 if spec.personalization_required else 2
    return min(PHYSICAL_TRANSITION_LANE_EPISODES, requested)


def _pass_decisions(
    selection: Mapping[str, Any], *, selection_pass: str
) -> list[dict[str, Any]]:
    return [
        {**copy.deepcopy(decision), "selection_pass": selection_pass}
        for decision in selection["lane_decisions"]
    ]


def _merge_group_metadata(
    retained: dict[str, Any], duplicate: Mapping[str, Any]
) -> None:
    retained_lanes = retained["rank_audit"]["selected_lanes"]
    for lane in duplicate["rank_audit"]["selected_lanes"]:
        if lane not in retained_lanes:
            retained_lanes.append(lane)
    retained_links = retained.setdefault("physical_anchor_links", [])
    for link in duplicate.get("physical_anchor_links", []):
        if link not in retained_links:
            retained_links.append(copy.deepcopy(link))


def _merge_episode_passes(
    specialist: Mapping[str, Any],
    physical: Mapping[str, Any],
    *,
    effective_transition_cap: int,
) -> dict[str, Any]:
    """Reserve the bounded specialist union, then spend remainder on linking."""

    _require(
        specialist["candidate_ranking_sha256"]
        == physical["candidate_ranking_sha256"],
        "independent episode passes changed their candidate population",
    )
    retained: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    by_source: Counter[str] = Counter()
    used_chunks = used_tokens = 0
    union_decisions: list[dict[str, Any]] = []

    for selection_pass, selection in (
        ("specialist_reserved", specialist),
        ("physical_remainder", physical),
    ):
        for source_group in selection["selected_groups"]:
            envelope_id = str(source_group["envelope_id"])
            group = copy.deepcopy(dict(source_group))
            group.pop("group_sha256", None)
            if envelope_id in retained:
                _merge_group_metadata(retained[envelope_id], group)
                union_decisions.append(
                    {
                        "decision": "selected_then_exact_envelope_dedup",
                        "envelope_id": envelope_id,
                        "selection_pass": selection_pass,
                    }
                )
                continue
            chunk_count = len(group["rows"])
            token_count = int(group["token_count"])
            source_id = str(group["source_id"])
            reason = None
            if len(retained) >= MAX_EPISODES:
                reason = "rejected_union_episode_cap"
            elif by_source[source_id] >= MAX_EPISODES_PER_SOURCE:
                reason = "rejected_union_source_episode_cap"
            elif (
                used_chunks + chunk_count > MAX_EPISODE_RAW_CHUNKS
                or used_tokens + token_count > MAX_EPISODE_RAW_TOKENS
            ):
                reason = "rejected_union_episode_raw_budget"
            if reason is not None:
                union_decisions.append(
                    {
                        "decision": reason,
                        "envelope_id": envelope_id,
                        "selection_pass": selection_pass,
                    }
                )
                continue
            retained[envelope_id] = group
            order.append(envelope_id)
            by_source[source_id] += 1
            used_chunks += chunk_count
            used_tokens += token_count

    selected_groups: list[dict[str, Any]] = []
    for envelope_id in order:
        group = retained[envelope_id]
        selected_groups.append({**group, "group_sha256": identity_sha256(group)})

    lane_rankings = [
        copy.deepcopy(row)
        for selection, names in (
            (specialist, set(_SPECIALIST_LANES)),
            (physical, set(_PHYSICAL_LANES)),
        )
        for row in selection["lane_rankings"]
        if str(row["lane"]).split(":", 1)[0] in names
    ]
    lane_used_chunks = {
        **specialist["lane_used_chunks"],
        **physical["lane_used_chunks"],
    }
    lane_used_tokens = {
        **specialist["lane_used_tokens"],
        **physical["lane_used_tokens"],
    }
    body = {
        key: copy.deepcopy(specialist[key])
        for key in (
            "active_source_ids",
            "candidate_envelope_count",
            "candidate_ranking_sha256",
            "candidate_index_binding",
            "source_anchor_populations_sha256",
            "sources_without_user_led_episode",
            "typed_spec_sha256",
        )
    }
    body.update(
        {
            "lane_budgets": {
                "episode_chunk_cap_each": EPISODE_LANE_CHUNK_BUDGET,
                "episode_token_cap_each": EPISODE_LANE_TOKEN_BUDGET,
                "physical_owner_chunk_cap": PHYSICAL_OWNER_LANE_CHUNK_BUDGET,
                "physical_owner_episode_cap": PHYSICAL_OWNER_LANE_EPISODES,
                "physical_owner_token_cap": PHYSICAL_OWNER_LANE_TOKEN_BUDGET,
                "physical_transition_chunk_cap": (
                    PHYSICAL_TRANSITION_LANE_CHUNK_BUDGET
                ),
                "physical_transition_episode_hard_cap": (
                    PHYSICAL_TRANSITION_LANE_EPISODES
                ),
                "physical_transition_episode_cap": effective_transition_cap,
                "physical_transition_token_cap": (
                    PHYSICAL_TRANSITION_LANE_TOKEN_BUDGET
                ),
                "query_user_lead_episode_cap": LEXICAL_LANE_EPISODES,
                "required_slot_total_episode_cap": SLOT_LANE_EPISODES,
                "source_local_anchor_episode_cap": ANCHOR_LANE_EPISODES,
                "typed_operation_episode_cap": TYPED_LANE_EPISODES,
                "union_policy": "specialist_reserved_then_physical_remainder",
            },
            "lane_decisions": [
                *_pass_decisions(specialist, selection_pass="specialist_reserved"),
                *_pass_decisions(physical, selection_pass="physical_remainder"),
                *union_decisions,
            ],
            "lane_proposal_count": sum(
                int(row["candidate_count"]) for row in lane_rankings
            ),
            "lane_rankings": lane_rankings,
            "lane_used_chunks": lane_used_chunks,
            "lane_used_tokens": lane_used_tokens,
            "physical_anchor_diagnostics": copy.deepcopy(
                physical["physical_anchor_diagnostics"]
            ),
            "physical_anchor_local_choices": copy.deepcopy(
                physical["physical_anchor_local_choices"]
            ),
            "physical_anchor_rankings": copy.deepcopy(
                physical["physical_anchor_rankings"]
            ),
            "provider_calls": 0,
            "selected_chunk_count": used_chunks,
            "selected_episode_count": len(selected_groups),
            "selected_groups": selected_groups,
            "selected_raw_token_count": used_tokens,
            "selection_pass_receipts": {
                "physical_remainder": physical["receipt_sha256"],
                "specialist_reserved": specialist["receipt_sha256"],
            },
        }
    )
    result = {**body, "receipt_sha256": identity_sha256(body)}
    assert_gold_blind(result, path="hot_v7_spine_episode_selection")
    return result


def select_spine_episodes(
    index: UserLedEnvelopeShadowIndex,
    *,
    active_source_ids: Sequence[str],
    dated_question: str,
    parent_packed_rows: Sequence[Mapping[str, Any]] = (),
    typed_spec: TypedOperatorSpec | None = None,
) -> dict[str, Any]:
    """Select specialist lanes independently before bounded physical linking."""

    spec = typed_spec or legacy.compile_typed_operator_spec(dated_question)
    with _legacy_scope(
        PHYSICAL_OWNER_LANE_EPISODES=0,
        PHYSICAL_TRANSITION_LANE_EPISODES=0,
    ):
        specialist = legacy.select_spine_episodes(
            index,
            active_source_ids=active_source_ids,
            dated_question=dated_question,
            parent_packed_rows=parent_packed_rows,
            typed_spec=spec,
        )
    transition_cap = _transition_episode_cap(spec)
    with _legacy_scope(
        ANCHOR_LANE_EPISODES=0,
        LEXICAL_LANE_EPISODES=0,
        SLOT_LANE_EPISODES=0,
        TYPED_LANE_EPISODES=0,
        PHYSICAL_TRANSITION_LANE_EPISODES=transition_cap,
    ):
        physical = legacy.select_spine_episodes(
            index,
            active_source_ids=active_source_ids,
            dated_question=dated_question,
            parent_packed_rows=parent_packed_rows,
            typed_spec=spec,
        )
    return _merge_episode_passes(
        specialist,
        physical,
        effective_transition_cap=transition_cap,
    )


def _global_fact_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "created_at": row.get("created_at"),
            "evidence_id": str(row["evidence_id"]),
            "raw_text": str(row["raw_text"]),
            "raw_text_sha256": str(
                row.get("raw_text_sha256") or quote_sha256(str(row["raw_text"]))
            ),
            "role": str(row["role"]),
            "source_id": str(row["source_id"]),
        }
        for row in rows
    ]


def _candidate_fact_rows(
    index: UserLedEnvelopeShadowIndex, active_source_ids: Sequence[str]
) -> list[dict[str, Any]]:
    """Project every exact cache row in already activated sources with ownership."""

    active = set(active_source_ids)
    envelope_by_id = {
        str(envelope.envelope_id): envelope for envelope in index.envelopes
    }
    output: list[dict[str, Any]] = []
    for row in sorted(
        index.row_by_chunk_id.values(), key=lambda value: (value.ordinal, value.chunk_id)
    ):
        if str(row.source_id) not in active:
            continue
        envelope_id = index.envelope_by_chunk_id.get(row.chunk_id)
        envelope = envelope_by_id.get(str(envelope_id))
        user_lead = None
        if envelope is not None:
            user_lead = next(
                (
                    chunk_id
                    for chunk_id, turn_id in zip(
                        envelope.chunk_ids, envelope.chunk_turn_ids, strict=True
                    )
                    if turn_id == envelope.opener_turn_id
                    and index.row_by_chunk_id[chunk_id].role == "user"
                ),
                None,
            )
        output.append(
            {
                "chunk_id": row.chunk_id,
                "created_at": row.created_at,
                "envelope_id": str(envelope_id) if envelope is not None else None,
                "evidence_id": row.chunk_id,
                "exchange_id": str(envelope_id) if envelope is not None else None,
                "raw_text": row.text,
                "raw_text_sha256": row.text_sha256,
                "role": row.role,
                "source_id": row.source_id,
                "user_lead_evidence_id": user_lead,
            }
        )
    return output


def _fact_input_rows(
    global_rows: Sequence[Mapping[str, Any]],
    episode_groups: Sequence[Mapping[str, Any]],
    *,
    index: UserLedEnvelopeShadowIndex,
    active_source_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Prepare independent selected G/E and activated-source candidate lanes."""

    global_facts = _global_fact_rows(global_rows)
    episode_facts = legacy._episode_fact_rows(episode_groups)  # noqa: SLF001
    candidate_facts = _candidate_fact_rows(index, active_source_ids)
    candidate_by_id = {
        str(row["evidence_id"]): row for row in candidate_facts
    }
    retained: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    bindings: list[dict[str, Any]] = []
    representation_exclusions: list[dict[str, Any]] = []

    for global_row in global_facts:
        evidence_id = str(global_row["evidence_id"])
        physical = candidate_by_id.get(evidence_id)
        if physical is not None and (
            global_row["raw_text"] != physical["raw_text"]
            or global_row["raw_text_sha256"] != physical["raw_text_sha256"]
        ):
            # The sealed episode manifest separately proves whether this is a
            # bounded G projection.  The fact compiler requires one exact byte
            # representation per evidence ID, so the full cache row owns facts.
            representation_exclusions.append(
                {
                    "evidence_id": evidence_id,
                    "excluded_lane": "global_projection",
                    "excluded_raw_text_sha256": global_row["raw_text_sha256"],
                    "retained_lane": "activated_source_candidate",
                    "retained_raw_text_sha256": physical["raw_text_sha256"],
                }
            )
            continue
        selected_row = copy.deepcopy(physical or global_row)
        retained[evidence_id] = selected_row
        order.append(evidence_id)

    for row in episode_facts:
        evidence_id = str(row["evidence_id"])
        previous = retained.get(evidence_id)
        if previous is None:
            retained[evidence_id] = copy.deepcopy(row)
            order.append(evidence_id)
            continue
        bindings.append(
            {
                "evidence_id": evidence_id,
                "excluded_lane": "global",
                "excluded_raw_text_sha256": str(previous["raw_text_sha256"]),
                "retained_lane": "episode",
                "retained_raw_text_sha256": str(row["raw_text_sha256"]),
                "same_exact_bytes": (
                    previous["raw_text"] == row["raw_text"]
                    and previous["raw_text_sha256"] == row["raw_text_sha256"]
                ),
            }
        )
        # The episode representation retains exchange/user-lead ownership.  Raw
        # G remains rendered; this replacement affects only downstream facts.
        retained[evidence_id] = copy.deepcopy(row)
    selected_rows = [retained[evidence_id] for evidence_id in order]
    body = {
        "absence_authority": "none_active_source_scan_does_not_close_frontier",
        "activated_source_candidate_count": len(candidate_facts),
        "activated_source_population_sha256": identity_sha256(candidate_facts),
        "dedup_bindings": bindings,
        "dedup_policy": "post_independent_selection_exact_evidence_id",
        "episode_input_count": len(episode_facts),
        "global_input_count": len(global_facts),
        "representation_exclusions": representation_exclusions,
        "selected_post_dedup_count": len(selected_rows),
        "selected_post_dedup_population_sha256": identity_sha256(selected_rows),
    }
    return (
        selected_rows,
        candidate_facts,
        {**body, "receipt_sha256": identity_sha256(body)},
    )


def _slot_term_hits(slot: object, text: str) -> tuple[str, ...]:
    present = set(normalized_terms(text))
    return tuple(term for term in slot.match_terms if term in present)


def _per_unit_terms(question: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            normalized
            for match in re.finditer(
                r"\bper\s+([A-Za-z][A-Za-z0-9'’-]*)", question, re.IGNORECASE
            )
            for normalized in normalized_terms(match.group(1))
        )
    )


def _numeric_completion_selection(
    dated_question: str,
    ledger: QueryFactLedger,
    episode_selection: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    index: UserLedEnvelopeShadowIndex,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Select one same-source raw numeric witness for each unresolved slot.

    A selected required-slot episode is the source anchor.  Completion never
    mutates the ledger binding or claims frontier closure: it only exposes one
    exact, same-source numeric quote whose dimension and query-derived
    relation/unit terms match.  Per-unit questions require that unit in the
    very same quote as the numeric operand.
    """

    unresolved = set(ledger.unresolved_slot_ids)
    numeric_slots = tuple(
        slot
        for slot in ledger.operator_spec.required_slots
        if slot.slot_id in unresolved and slot.requires_numeric
    )
    dimension = expected_numeric_dimension(
        operator_spec=ledger.operator_spec,
        question=dated_question,
    )
    question_terms = normalized_terms(dated_question)
    relation_terms = tuple(
        term
        for term in question_terms
        if term in _NUMERIC_RELATION_UNIT_TERMS[dimension]
    )
    unit_terms = _per_unit_terms(dated_question)
    candidate_by_id = {
        str(row["evidence_id"]): row for row in candidate_rows
    }
    facts_by_source: dict[str, list[object]] = {}
    for fact in ledger.facts:
        facts_by_source.setdefault(str(fact.backing_source_id), []).append(fact)

    anchor_ids: list[str] = []
    operand_ids: list[str] = []
    decisions: list[dict[str, Any]] = []
    for slot in numeric_slots:
        lane = f"required_slot:{slot.slot_id}"
        anchor_candidates: list[tuple[int, int, int, str, str, str]] = []
        group_by_id = {
            str(group["envelope_id"]): group
            for group in episode_selection["selected_groups"]
        }
        lane_positions: dict[str, int] = {}
        for decision in episode_selection["lane_decisions"]:
            if (
                decision.get("lane") != lane
                or not str(decision.get("decision", "")).startswith("selected")
            ):
                continue
            envelope_id = str(decision["envelope_id"])
            lane_positions.setdefault(envelope_id, int(decision["lane_rank"]))
        for envelope_id, lane_rank in sorted(
            lane_positions.items(), key=lambda item: (item[1], item[0])
        ):
            group = group_by_id.get(envelope_id)
            if group is None:
                continue
            anchor_row = max(
                group["rows"],
                key=lambda row: (
                    legacy._compatible_hits(  # noqa: SLF001
                        slot.match_terms, normalized_terms(str(row["text"]))
                    ),
                    int(str(row["role"]) == "user"),
                    len(_slot_term_hits(slot, str(row["text"]))),
                    -int(row["ordinal"]),
                    str(row["chunk_id"]),
                ),
            )
            compatible_hits = legacy._compatible_hits(  # noqa: SLF001
                slot.match_terms, normalized_terms(str(anchor_row["text"]))
            )
            anchor_candidates.append(
                (
                    lane_rank,
                    -compatible_hits,
                    -int(str(anchor_row["role"]) == "user"),
                    str(group["source_id"]),
                        envelope_id,
                    str(anchor_row["chunk_id"]),
                )
            )
        ordered_anchors = sorted(anchor_candidates)
        source_anchors: list[tuple[str, str, str]] = []
        seen_sources: set[str] = set()
        for _rank, _role, _hits, source_id, envelope_id, evidence_id in ordered_anchors:
            if source_id in seen_sources:
                continue
            seen_sources.add(source_id)
            source_anchors.append((source_id, envelope_id, evidence_id))

        slot_decision: dict[str, Any] = {
            "anchor_candidate_count": len(anchor_candidates),
            "anchor_population_sha256": identity_sha256(
                [list(value) for value in ordered_anchors]
            ),
            "decision": "no_selected_required_slot_anchor",
            "relation_terms": list(relation_terms),
            "slot_id": slot.slot_id,
            "slot_label": slot.label,
            "unit_terms": list(unit_terms),
        }
        if not source_anchors:
            decisions.append(slot_decision)
            continue

        source_results: list[
            tuple[
                tuple[int, ...],
                str,
                str,
                str,
                list[tuple[str, tuple[tuple[int, ...], object]]],
                Counter[str],
            ]
        ] = []
        for source_position, (
            source_id,
            anchor_envelope_id,
            anchor_evidence_id,
        ) in enumerate(source_anchors):
            qualified: dict[str, tuple[tuple[int, ...], object]] = {}
            rejected: Counter[str] = Counter()
            for fact in facts_by_source.get(source_id, []):
                if any(
                    bound_slot_id != slot.slot_id
                    for bound_slot_id in fact.bound_slot_ids
                ):
                    rejected["different_required_slot"] += 1
                    continue
                if str(fact.source_role) != "user":
                    rejected["non_user"] += 1
                    continue
                if fact.status.value not in {"asserted", "completed"}:
                    rejected["non_final_status"] += 1
                    continue
                if not any(
                    operand.dimension == dimension.value
                    for operand in fact.numeric_operands
                ):
                    rejected["numeric_dimension"] += 1
                    continue
                quote_terms = set(normalized_terms(fact.exact_quote))
                relation_hits = tuple(
                    term for term in relation_terms if term in quote_terms
                )
                unit_hits = tuple(
                    term
                    for term in unit_terms
                    if term in quote_terms
                    or (term == "night" and "nightly" in quote_terms)
                )
                if (unit_terms and not unit_hits) or (
                    not unit_terms and not relation_hits
                ):
                    rejected["query_relation_or_unit"] += 1
                    continue
                row = candidate_by_id.get(str(fact.backing_evidence_id))
                if row is None:
                    rejected["candidate_row_missing"] += 1
                    continue
                ordinal = int(
                    index.row_by_chunk_id[str(fact.backing_evidence_id)].ordinal
                )
                score = (
                    int(bool(unit_hits)),
                    len(unit_hits),
                    len(relation_hits),
                    int(fact.status.value == "completed"),
                    len(fact.query_term_hits),
                    int(fact.relevance_score),
                    -ordinal,
                )
                prior = qualified.get(str(fact.backing_evidence_id))
                if prior is None or score > prior[0]:
                    qualified[str(fact.backing_evidence_id)] = (score, fact)
            ordered = sorted(
                qualified.items(),
                key=lambda item: (item[1][0], item[0]),
                reverse=True,
            )
            if ordered:
                source_score = (*ordered[0][1][0], -source_position)
                source_results.append(
                    (
                        source_score,
                        source_id,
                        anchor_envelope_id,
                        anchor_evidence_id,
                        ordered,
                        rejected,
                    )
                )
        if not source_results:
            source_id, anchor_envelope_id, anchor_evidence_id = source_anchors[0]
            slot_decision.update(
                {
                    "anchor_envelope_id": anchor_envelope_id,
                    "anchor_evidence_id": anchor_evidence_id,
                    "anchor_source_id": source_id,
                    "decision": "no_source_local_numeric_relation_match",
                    "qualified_candidate_count": 0,
                    "qualified_population_sha256": identity_sha256([]),
                    "rejection_counts": {},
                    "scanned_anchor_source_count": len(source_anchors),
                    "selected_operand_evidence_ids": [],
                }
            )
            decisions.append(slot_decision)
            continue
        (
            _source_score,
            source_id,
            anchor_envelope_id,
            anchor_evidence_id,
            ordered,
            rejected,
        ) = source_results[0]
        chosen = ordered[:MAX_NUMERIC_COMPLETION_ROWS_PER_SLOT_SOURCE]
        anchor_ids.append(anchor_evidence_id)
        operand_ids.extend(evidence_id for evidence_id, _value in chosen)
        slot_decision.update(
            {
                "anchor_envelope_id": anchor_envelope_id,
                "anchor_evidence_id": anchor_evidence_id,
                "anchor_source_id": source_id,
                "decision": (
                    "selected_source_local_numeric_raw"
                    if chosen
                    else "no_source_local_numeric_relation_match"
                ),
                "qualified_candidate_count": len(ordered),
                "qualified_population_sha256": identity_sha256(
                    [
                        {
                            "evidence_id": evidence_id,
                            "fact_id": fact.fact_id,
                            "score": list(score),
                        }
                        for evidence_id, (score, fact) in ordered
                    ]
                ),
                "rejection_counts": {
                    key: rejected[key] for key in sorted(rejected)
                },
                "scanned_anchor_source_count": len(source_anchors),
                "selected_operand_evidence_ids": [
                    evidence_id for evidence_id, _value in chosen
                ],
            }
        )
        decisions.append(slot_decision)

    anchors = list(dict.fromkeys(anchor_ids))
    operands = list(dict.fromkeys(operand_ids))
    body = {
        "absence_authority": "none_completion_does_not_close_frontier",
        "anchor_evidence_ids": anchors,
        "decisions": decisions,
        "expected_numeric_dimension": dimension.value,
        "frontier_closed": False,
        "gold_loaded": False,
        "max_completion_envelopes": MAX_NUMERIC_COMPLETION_ENVELOPES,
        "max_completion_rows_per_slot_source": (
            MAX_NUMERIC_COMPLETION_ROWS_PER_SLOT_SOURCE
        ),
        "max_completion_sources_per_slot": (
            MAX_NUMERIC_COMPLETION_SOURCES_PER_SLOT
        ),
        "max_completion_turns": MAX_NUMERIC_COMPLETION_TURNS,
        "operand_evidence_ids": operands,
        "policy": "selected_required_slot_anchor_same_source_numeric_unit",
        "provider_calls": 0,
        "slot_bindings_added": 0,
        "unresolved_numeric_slot_ids": [slot.slot_id for slot in numeric_slots],
    }
    return anchors, operands, body


def _project_cache_row(
    row: object,
    *,
    envelope_id: str,
    opener_turn_id: str,
    user_lead_evidence_id: str,
) -> dict[str, Any]:
    return {
        "chunk_id": row.chunk_id,
        "created_at": row.created_at,
        "envelope_id": envelope_id,
        "opener_user_chunk_id": user_lead_evidence_id,
        "opener_user_turn_id": opener_turn_id,
        "ordinal": row.ordinal,
        "role": row.role,
        "source_id": row.source_id,
        "text": row.text,
        "text_sha256": row.text_sha256,
        "token_count": row.token_count,
        "turn_id": row.turn_id,
    }


def _finalize_episode_group(group: Mapping[str, Any]) -> dict[str, Any]:
    body = copy.deepcopy(dict(group))
    body.pop("group_sha256", None)
    body["rows"].sort(key=lambda row: (int(row["ordinal"]), str(row["chunk_id"])))
    body["row_count"] = len(body["rows"])
    body["token_count"] = sum(int(row["token_count"]) for row in body["rows"])
    body["turn_ids"] = list(
        dict.fromkeys(str(row["turn_id"]) for row in body["rows"])
    )
    return {**body, "group_sha256": identity_sha256(body)}


def _merge_episode_groups(
    retained: Mapping[str, Any], expansion: Mapping[str, Any]
) -> dict[str, Any]:
    """Union exact rows for one envelope while preserving required rows."""

    _require(
        str(retained["envelope_id"]) == str(expansion["envelope_id"])
        and str(retained["source_id"]) == str(expansion["source_id"])
        and str(retained["opener_user_chunk_id"])
        == str(expansion["opener_user_chunk_id"]),
        "episode expansion crossed an exact owner boundary",
    )
    body = copy.deepcopy(dict(retained))
    body.pop("group_sha256", None)
    by_id = {str(row["chunk_id"]): row for row in body["rows"]}
    for source in expansion["rows"]:
        chunk_id = str(source["chunk_id"])
        prior = by_id.get(chunk_id)
        if prior is not None:
            _require(
                str(prior["text_sha256"]) == str(source["text_sha256"])
                and str(prior["text"]) == str(source["text"])
                and str(prior["role"]) == str(source["role"]),
                "episode expansion changed exact row bytes",
            )
            continue
        row = copy.deepcopy(dict(source))
        body["rows"].append(row)
        by_id[chunk_id] = row
    _merge_group_metadata(body, expansion)
    reasons = body.setdefault("truncation_reasons", [])
    for reason in expansion.get("truncation_reasons", []):
        if reason not in reasons:
            reasons.append(str(reason))
    return _finalize_episode_group(body)


def _merge_episode_group_sets(
    retained: Sequence[Mapping[str, Any]],
    additions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {
        str(group["envelope_id"]): copy.deepcopy(dict(group)) for group in retained
    }
    order = [str(group["envelope_id"]) for group in retained]
    for addition in additions:
        envelope_id = str(addition["envelope_id"])
        if envelope_id in by_id:
            by_id[envelope_id] = _merge_episode_groups(by_id[envelope_id], addition)
        else:
            by_id[envelope_id] = copy.deepcopy(dict(addition))
            order.append(envelope_id)
    return [by_id[envelope_id] for envelope_id in order]


def _hydrate_numeric_completion_groups(
    anchor_evidence_ids: Sequence[str],
    operand_evidence_ids: Sequence[str],
    *,
    index: UserLedEnvelopeShadowIndex,
    global_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Hydrate exact opener+target rows for one-hop numeric completion."""

    global_by_id = {str(row["evidence_id"]): row for row in global_rows}
    envelope_by_id = {
        str(envelope.envelope_id): envelope for envelope in index.envelopes
    }
    kind_by_id = {
        **{str(evidence_id): "slot_anchor" for evidence_id in anchor_evidence_ids},
        **{str(evidence_id): "numeric_operand" for evidence_id in operand_evidence_ids},
    }
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    bindings: list[dict[str, str]] = []
    represented_global: list[str] = []
    unresolved: list[str] = []
    for evidence_id in dict.fromkeys(
        [*map(str, anchor_evidence_ids), *map(str, operand_evidence_ids)]
    ):
        backing = index.row_by_chunk_id.get(evidence_id)
        if backing is None:
            unresolved.append(evidence_id)
            continue
        global_row = global_by_id.get(evidence_id)
        if (
            global_row is not None
            and str(global_row["raw_text"]) == backing.text
            and str(global_row["raw_text_sha256"]) == backing.text_sha256
        ):
            represented_global.append(evidence_id)
            bindings.append(
                {
                    "binding_kind": kind_by_id[evidence_id],
                    "evidence_id": evidence_id,
                    "representation": "protected_global",
                }
            )
            continue
        envelope_id = index.envelope_by_chunk_id.get(evidence_id)
        envelope = envelope_by_id.get(str(envelope_id))
        if envelope is None:
            unresolved.append(evidence_id)
            continue
        opener_rows = [
            index.row_by_chunk_id[chunk_id]
            for chunk_id, turn_id in zip(
                envelope.chunk_ids, envelope.chunk_turn_ids, strict=True
            )
            if turn_id == envelope.opener_turn_id
            and index.row_by_chunk_id[chunk_id].role == "user"
        ]
        if not opener_rows:
            unresolved.append(evidence_id)
            continue
        lead_id = opener_rows[0].chunk_id
        group = groups.get(str(envelope.envelope_id))
        if group is None:
            group = {
                "envelope_id": str(envelope.envelope_id),
                "opener_user_chunk_id": lead_id,
                "opener_user_turn_id": str(envelope.opener_turn_id),
                "physical_anchor_links": [],
                "rank_audit": {
                    "numeric_completion_only": True,
                    "selected_lanes": [
                        "unresolved_numeric_source_local_completion"
                    ],
                },
                "rows": [],
                "source_id": str(envelope.source_id),
                "truncation_reasons": [
                    "numeric_completion_minimal_owner_hydration"
                ],
            }
            groups[str(envelope.envelope_id)] = group
            order.append(str(envelope.envelope_id))
        existing = {str(row["chunk_id"]) for row in group["rows"]}
        for row in [opener_rows[0], backing]:
            if row.chunk_id in existing:
                continue
            group["rows"].append(
                _project_cache_row(
                    row,
                    envelope_id=str(envelope.envelope_id),
                    opener_turn_id=str(envelope.opener_turn_id),
                    user_lead_evidence_id=lead_id,
                )
            )
            existing.add(row.chunk_id)
        bindings.append(
            {
                "binding_kind": kind_by_id[evidence_id],
                "envelope_id": str(envelope.envelope_id),
                "evidence_id": evidence_id,
                "representation": "minimal_episode_raw",
            }
        )

    hydrated = [_finalize_episode_group(groups[envelope_id]) for envelope_id in order]
    admitted: list[dict[str, Any]] = []
    used_turns: set[str] = set()
    cap_rejections: list[dict[str, Any]] = []
    for group in hydrated:
        trial_turns = used_turns | set(map(str, group["turn_ids"]))
        if (
            len(admitted) >= MAX_NUMERIC_COMPLETION_ENVELOPES
            or len(trial_turns) > MAX_NUMERIC_COMPLETION_TURNS
        ):
            cap_rejections.append(
                {
                    "envelope_id": str(group["envelope_id"]),
                    "reason": "numeric_completion_owner_bound",
                }
            )
            continue
        admitted.append(group)
        used_turns = trial_turns
    admitted_envelopes = {str(group["envelope_id"]) for group in admitted}
    admitted_ids = {
        evidence_id
        for evidence_id in kind_by_id
        if evidence_id in represented_global
        or str(index.envelope_by_chunk_id.get(evidence_id)) in admitted_envelopes
    }
    body = {
        "admitted_evidence_ids": sorted(admitted_ids),
        "bindings": bindings,
        "cap_rejections": cap_rejections,
        "hydration_policy": "minimal_opener_plus_exact_numeric_target",
        "represented_global_evidence_ids": represented_global,
        "unresolved_evidence_ids": list(dict.fromkeys(unresolved)),
    }
    return admitted, {**body, "receipt_sha256": identity_sha256(body)}


def _hydrate_fact_backing_groups(
    episode_groups: Sequence[Mapping[str, Any]],
    facts: Sequence[Mapping[str, Any]],
    *,
    index: UserLedEnvelopeShadowIndex,
    global_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add the exact user-led owner neighborhood for candidate-only facts."""

    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for source in episode_groups:
        group = copy.deepcopy(dict(source))
        group.pop("group_sha256", None)
        envelope_id = str(group["envelope_id"])
        groups[envelope_id] = group
        order.append(envelope_id)
    global_by_id = {str(row["evidence_id"]): row for row in global_rows}
    envelope_by_id = {
        str(envelope.envelope_id): envelope for envelope in index.envelopes
    }
    hydrated: list[dict[str, Any]] = []
    unresolved: list[str] = []

    for fact in facts:
        backing_id = str(fact["backing_evidence_id"])
        quote = str(fact["exact_quote"])
        global_row = global_by_id.get(backing_id)
        if global_row is not None and quote in str(global_row["raw_text"]):
            continue
        if any(
            str(row["chunk_id"]) == backing_id and quote in str(row["text"])
            for group in groups.values()
            for row in group["rows"]
        ):
            continue
        envelope_id = fact.get("envelope_id") or index.envelope_by_chunk_id.get(
            backing_id
        )
        envelope = envelope_by_id.get(str(envelope_id))
        backing = index.row_by_chunk_id.get(backing_id)
        if envelope is None or backing is None or quote not in backing.text:
            unresolved.append(backing_id)
            continue
        opener_rows = [
            index.row_by_chunk_id[chunk_id]
            for chunk_id, turn_id in zip(
                envelope.chunk_ids, envelope.chunk_turn_ids, strict=True
            )
            if turn_id == envelope.opener_turn_id
            and index.row_by_chunk_id[chunk_id].role == "user"
        ]
        if not opener_rows:
            unresolved.append(backing_id)
            continue
        lead_id = opener_rows[0].chunk_id
        required_rows = [opener_rows[0]]
        if backing.chunk_id not in {row.chunk_id for row in required_rows}:
            required_rows.append(backing)
        group = groups.get(str(envelope.envelope_id))
        if group is None:
            group = {
                "envelope_id": str(envelope.envelope_id),
                "opener_user_chunk_id": lead_id,
                "opener_user_turn_id": str(envelope.opener_turn_id),
                "physical_anchor_links": [],
                "rank_audit": {
                    "fact_backing_only": True,
                    "selected_lanes": ["activated_source_fact_backing"],
                },
                "rows": [],
                "source_id": str(envelope.source_id),
                "truncation_reasons": ["fact_backing_minimal_hydration"],
            }
            groups[str(envelope.envelope_id)] = group
            order.append(str(envelope.envelope_id))
        else:
            lanes = group["rank_audit"]["selected_lanes"]
            if "activated_source_fact_backing" not in lanes:
                lanes.append("activated_source_fact_backing")
        existing_ids = {str(row["chunk_id"]) for row in group["rows"]}
        for row in required_rows:
            if row.chunk_id not in existing_ids:
                group["rows"].append(
                    _project_cache_row(
                        row,
                        envelope_id=str(envelope.envelope_id),
                        opener_turn_id=str(envelope.opener_turn_id),
                        user_lead_evidence_id=lead_id,
                    )
                )
                existing_ids.add(row.chunk_id)
        hydrated.append(
            {
                "backing_evidence_id": backing_id,
                "envelope_id": str(envelope.envelope_id),
                "fact_id": str(fact["fact_id"]),
            }
        )

    output: list[dict[str, Any]] = []
    for envelope_id in order:
        group = groups[envelope_id]
        group["rows"].sort(
            key=lambda row: (int(row["ordinal"]), str(row["chunk_id"]))
        )
        group["row_count"] = len(group["rows"])
        group["token_count"] = sum(int(row["token_count"]) for row in group["rows"])
        group["turn_ids"] = list(
            dict.fromkeys(str(row["turn_id"]) for row in group["rows"])
        )
        output.append({**group, "group_sha256": identity_sha256(group)})
    body = {
        "hydrated_bindings": hydrated,
        "hydration_policy": "selected_candidate_fact_exact_owner_neighborhood",
        "unresolved_backing_evidence_ids": list(dict.fromkeys(unresolved)),
    }
    return output, {**body, "receipt_sha256": identity_sha256(body)}


def _provider_label_index(
    parent_rows: Sequence[Mapping[str, Any]],
    episode_groups: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    labels = {
        str(row["evidence_id"]): f"G{position}"
        for position, row in enumerate(parent_rows, 1)
        if row.get("evidence_id") is not None
    }
    for block, group in enumerate(episode_groups, 1):
        raw_rows = group.get("raw_rows", group.get("rows", []))
        raw_by_id = {str(row["chunk_id"]): row for row in raw_rows}
        refs_by_id = {
            str(reference["episode_chunk_id"]): reference
            for reference in group.get("global_refs", [])
        }
        manifest_rows = group.get(
            "manifest_rows",
            [
                {"chunk_id": str(row["chunk_id"]), "representation": "episode_raw"}
                for row in raw_rows
            ],
        )
        for row_number, manifest_row in enumerate(manifest_rows, 1):
            chunk_id = str(manifest_row["chunk_id"])
            if manifest_row["representation"] == "global_ref":
                global_id = str(refs_by_id[chunk_id]["global_evidence_id"])
                if global_id in labels:
                    labels.setdefault(chunk_id, labels[global_id])
                continue
            role = str(raw_by_id[chunk_id]["role"])
            marker = "U" if role == "user" else "A" if role == "assistant" else "X"
            labels[chunk_id] = f"E{block}.{marker}{row_number}"
    return labels


def _render_fact_section(
    facts: Sequence[Mapping[str, Any]], labels: Mapping[str, str]
) -> str:
    if not facts:
        return ""
    lines = ["<FACTS exact_quotes raw_authoritative>"]
    for number, fact in enumerate(facts, 1):
        backing_id = str(fact["backing_evidence_id"])
        metadata = [
            f"backs={labels.get(backing_id, 'RAW')}",
            f"role={str(fact['source_role'])[:1].upper()}",
            f"status={fact['status']}",
        ]
        if fact.get("time_mentions"):
            metadata.append("time=" + "|".join(fact["time_mentions"]))
        if fact.get("action_concepts"):
            metadata.append("actions=" + "|".join(fact["action_concepts"]))
        operands = fact.get("numeric_operands", [])
        if operands:
            metadata.append("nums=" + "|".join(str(row["surface"]) for row in operands))
        created = fact.get("source_created_at")
        dated = f" [{created}]" if created else ""
        lines.append(
            f"<F{number} {' '.join(metadata)}>{dated} {fact['exact_quote']}"
        )
    return "\n".join(lines)


def _render_numeric_completion_section(
    selection: Mapping[str, Any], labels: Mapping[str, str]
) -> tuple[str, list[dict[str, str]]]:
    lines = ["<SOURCE_LOCAL_CANDIDATES unbound>"]
    bindings: list[dict[str, str]] = []
    for decision in selection["decisions"]:
        if decision["decision"] != "selected_source_local_numeric_raw":
            continue
        anchor_id = str(decision["anchor_evidence_id"])
        anchor_label = labels.get(anchor_id)
        for operand_id in decision["selected_operand_evidence_ids"]:
            operand_label = labels.get(str(operand_id))
            if anchor_label is None or operand_label is None:
                continue
            relation = "|".join(decision["unit_terms"] or decision["relation_terms"])
            binding = {
                "anchor_evidence_id": anchor_id,
                "anchor_provider_label": anchor_label,
                "backing_evidence_id": str(operand_id),
                "backing_provider_label": operand_label,
                "relation": relation,
                "slot_id": str(decision["slot_id"]),
            }
            bindings.append(binding)
            lines.append(
                f"<SC{len(bindings)} anchor={anchor_label} backs={operand_label} "
                f"relation={relation} dimension={selection['expected_numeric_dimension']} "
                "status=unbound_same_source>"
            )
    if not bindings:
        return "", []
    lines.append(
        "These are same-source raw candidates only; no unresolved slot is bound "
        "and absence is not established."
    )
    return "\n".join(lines), bindings


def _provider_overlay_accounting(
    global_rows: Sequence[Mapping[str, Any]],
    episode_manifests: Sequence[Mapping[str, Any]],
    facts: Sequence[Mapping[str, Any]],
    *,
    advisory_text: str,
    completion_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Render and count one exact provider overlay against one manifest set.

    Provider labels are a function of the final ordered manifest population.
    Admission and final rendering must therefore share this single accounting
    path; counting a filtered manifest view can under-reserve the hard cap.
    """

    labels = _provider_label_index(global_rows, episode_manifests)
    fact_section = _render_fact_section(facts, labels)
    completion_text, completion_bindings = (
        _render_numeric_completion_section(completion_selection, labels)
        if completion_selection is not None
        else ("", [])
    )
    fact_advisory_text = "\n\n".join(
        value for value in (fact_section, advisory_text) if value
    )
    provider_tail = "\n\n".join(
        value for value in (advisory_text, completion_text) if value
    )
    provider_text = "\n\n".join(
        value
        for value in (fact_section, advisory_text, completion_text)
        if value
    )
    return {
        "completion_bindings": completion_bindings,
        "completion_text": completion_text,
        "fact_advisory_text": fact_advisory_text,
        "fact_advisory_token_count": count_tokens(fact_advisory_text),
        "fact_section": fact_section,
        "fact_section_token_count": count_tokens(fact_section),
        "labels": labels,
        "provider_tail": provider_tail,
        "provider_text": provider_text,
        "total_token_count": count_tokens(provider_text),
    }


def _represented_raw_evidence_ids(
    global_rows: Sequence[Mapping[str, Any]],
    episode_manifests: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Return the exact ordered raw-address population visible in one packet."""

    return tuple(
        dict.fromkeys(
            [str(row["evidence_id"]) for row in global_rows]
            + [
                str(row["chunk_id"])
                for manifest in episode_manifests
                for row in manifest["raw_rows"]
            ]
            + [
                str(reference["episode_chunk_id"])
                for manifest in episode_manifests
                for reference in manifest["global_refs"]
            ]
        )
    )


def _compile_ledger_advisory(
    dated_question: str,
    ledger: QueryFactLedger,
    selected_slice: FactLedgerSlice | None,
    *,
    represented_backing_evidence_ids: Sequence[str],
) -> tuple[str, dict[str, Any]]:
    """Compile a supported typed reduction over the final visible fact slice."""

    if selected_slice is None:
        body = {
            "format": f"{FORMAT}-typed-reducer-advisory-empty-v1",
            "gold_loaded": False,
            "ledger_receipt_sha256": ledger.receipt_sha256,
            "provider_advisory": {
                "emitted": False,
                "support_fact_labels": [],
                "text": "",
                "text_sha256": quote_sha256(""),
            },
            "provider_calls": 0,
            "question_sha256": quote_sha256(dated_question),
            "reason": "no_selected_fact_slice",
            "represented_backing_evidence_ids": list(
                represented_backing_evidence_ids
            ),
            "retained_transformer_token_state_bytes": 0,
            "selected_slice_receipt_sha256": None,
            "status": "not_run",
        }
        assert_gold_blind(body, path="hot_v7_empty_typed_reducer_advisory")
        return "", body
    result = compile_hot_v6_typed_reducer_advisory(
        dated_question,
        ledger,
        selected_slice,
        represented_backing_evidence_ids=represented_backing_evidence_ids,
    )
    return result.provider_advisory_text, result.audit_projection


def _render_context(
    parent_rows: Sequence[Mapping[str, Any]],
    episode_groups: Sequence[Mapping[str, Any]],
    facts: Sequence[Mapping[str, Any]],
    *,
    advisory_text: str = "",
) -> str:
    sections: list[str] = []
    labels = _provider_label_index(parent_rows, episode_groups)
    for block, row in enumerate(parent_rows, 1):
        sections.append(f"<G{block}>\n{legacy._evidence_line(row)}")  # noqa: SLF001
    for block, group in enumerate(episode_groups, 1):
        lines = [f"<E{block}>"]
        raw_rows = group.get("raw_rows", group.get("rows", []))
        raw_by_id = {str(row["chunk_id"]): row for row in raw_rows}
        refs_by_id = {
            str(reference["episode_chunk_id"]): reference
            for reference in group.get("global_refs", [])
        }
        collisions_by_id = {
            str(collision["episode_evidence_id"]): collision
            for collision in group.get("representation_collisions", [])
        }
        manifest_rows = group.get(
            "manifest_rows",
            [
                {"chunk_id": str(row["chunk_id"]), "representation": "episode_raw"}
                for row in raw_rows
            ],
        )
        opener_id = str(group.get("opener_user_chunk_id", ""))
        opener = labels.get(opener_id, "U1").split(".")[-1]
        for row_number, manifest_row in enumerate(manifest_rows, 1):
            chunk_id = str(manifest_row["chunk_id"])
            if manifest_row["representation"] == "global_ref":
                global_id = str(refs_by_id[chunk_id]["global_evidence_id"])
                _require(global_id in labels, "episode global provider citation missing")
                lines.append(f"<REF {labels[global_id]}>")
                continue
            row = raw_by_id[chunk_id]
            role = str(row["role"])
            marker = "U" if role == "user" else "A" if role == "assistant" else "X"
            local = f"{marker}{row_number}"
            attributes = f" owner={opener}" if role == "assistant" else ""
            if manifest_row["representation"] == "episode_collision_raw":
                collision = collisions_by_id.get(chunk_id)
                _require(collision is not None, "episode collision binding missing")
                attributes += (
                    f" projection_of={collision['global_citation']}"
                    f" collision=C{manifest_row['collision_ordinal']}"
                )
            lines.append(
                f"<{local}{attributes}> {legacy._evidence_line(row)}"  # noqa: SLF001
            )
        sections.append("\n".join(lines))
    fact_section = _render_fact_section(facts, labels)
    if fact_section:
        sections.append(fact_section)
    if advisory_text:
        sections.append(advisory_text)
    return "\n\n".join(sections)


def _provider_provenance_manifest(
    global_citations: Mapping[str, Any],
    episode_manifests: Sequence[Mapping[str, Any]],
    facts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    global_entries = global_citations["entries"]
    global_labels = {
        str(entry["evidence_id"]): str(entry["citation"]) for entry in global_entries
    }
    episodes: list[dict[str, Any]] = []
    backing_labels = dict(global_labels)
    for block, manifest in enumerate(episode_manifests, 1):
        raw_by_id = {str(row["chunk_id"]): row for row in manifest["raw_rows"]}
        refs_by_id = {
            str(row["episode_chunk_id"]): row for row in manifest["global_refs"]
        }
        rows: list[dict[str, Any]] = []
        for row_number, manifest_row in enumerate(manifest["manifest_rows"], 1):
            chunk_id = str(manifest_row["chunk_id"])
            if manifest_row["representation"] == "global_ref":
                global_id = str(refs_by_id[chunk_id]["global_evidence_id"])
                provider_label = global_labels[global_id]
                role = next(
                    str(entry["role"])
                    for entry in global_entries
                    if str(entry["evidence_id"]) == global_id
                )
            else:
                raw = raw_by_id[chunk_id]
                role = str(raw["role"])
                marker = "U" if role == "user" else "A" if role == "assistant" else "X"
                provider_label = f"E{block}.{marker}{row_number}"
                backing_labels[chunk_id] = provider_label
            rows.append(
                {
                    "evidence_id": chunk_id,
                    "provider_label": provider_label,
                    "representation": str(manifest_row["representation"]),
                    "role": role,
                }
            )
        episodes.append(
            {
                "envelope_id": str(manifest["envelope_id"]),
                "provider_block": f"E{block}",
                "rows": rows,
                "source_id": str(manifest["source_id"]),
            }
        )
    fact_entries = [
        {
            "backing_evidence_id": str(fact["backing_evidence_id"]),
            "backing_provider_label": backing_labels.get(
                str(fact["backing_evidence_id"]), "RAW"
            ),
            "fact_id": str(fact["fact_id"]),
            "provider_label": f"F{number}",
        }
        for number, fact in enumerate(facts, 1)
    ]
    body = {
        "episodes": episodes,
        "facts": fact_entries,
        "format": f"{FORMAT}-provider-provenance-v1",
        "global_citation_receipt_sha256": global_citations["receipt_sha256"],
        "policy": "compact_provider_labels_full_ids_audit_only",
    }
    return {**body, "receipt_sha256": identity_sha256(body)}


def _prompt(question: str, context: str) -> tuple[list[dict[str, str]], int, int]:
    messages = [
        {"role": "system", "content": legacy.OPERATION_AWARE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(context=context, question=question),
        },
    ]
    context_tokens = count_tokens(context)
    workspace_tokens = hot.count_chat_prompt_token_proxy(messages) + OUTPUT_TOKEN_RESERVE
    return messages, context_tokens, workspace_tokens


def _fits(dated_question: str, context: str) -> tuple[bool, int, int]:
    _messages, context_tokens, workspace_tokens = _prompt(dated_question, context)
    return (
        context_tokens <= MAX_CONTEXT_TOKENS
        and workspace_tokens <= MAX_WORKSPACE_TOKENS,
        context_tokens,
        workspace_tokens,
    )


def _fact_projection(fact_slice: object | None) -> list[dict[str, Any]]:
    return (
        [fact.projection() for fact in fact_slice.facts]
        if fact_slice is not None
        else []
    )


def _mandatory_fact_count(ledger: QueryFactLedger) -> int:
    """Return the sealed v6 slot/origin/type-lane coverage cardinality."""

    return len(fact_ledger_contract._coverage_fact_ids(ledger))  # noqa: SLF001


def _select_audit_fact_slice(
    ledger: QueryFactLedger,
    *,
    max_facts: int,
    initial_tokens: int = INITIAL_FACT_SELECTION_AUDIT_TOKENS,
) -> tuple[FactLedgerSlice, int]:
    """Encode an exact fact-count slice without imposing a provider budget.

    The verbose v6 audit renderer carries full hashes and is intentionally much
    larger than the compact provider representation.  Grow its private bound
    only until the requested number of facts is representable.  Provider
    admission is decided later from the compact fact/advisory bytes.
    """

    _require(
        type(max_facts) is int and 1 <= max_facts <= MAX_FACTS,
        "audit fact-count request escaped the sealed fact cap",
    )
    _require(
        type(initial_tokens) is int and initial_tokens >= 1,
        "audit token floor must be a positive exact integer",
    )
    mandatory_count = _mandatory_fact_count(ledger)
    if mandatory_count > max_facts:
        raise QueryFactLedgerError(
            "query-fact render fact bound cannot preserve slot/lane coverage"
        )
    target_count = min(max_facts, len(ledger.facts))
    audit_tokens = initial_tokens
    while True:
        try:
            selected = select_and_render_query_fact_ledger(
                ledger,
                max_facts=max_facts,
                max_tokens=audit_tokens,
            )
        except QueryFactLedgerError as exc:
            if "token bound cannot preserve slot/lane coverage" not in str(exc):
                raise
        else:
            if len(selected.facts) == target_count:
                return selected, audit_tokens
        audit_tokens *= 2


def _fact_selection_audit(
    ledger: QueryFactLedger,
    *,
    selected_slice: FactLedgerSlice | None,
    status: str,
    provider_token_count: int,
) -> dict[str, Any]:
    selected = (
        legacy._compact_fact_selection(ledger, selected_slice=selected_slice)  # noqa: SLF001
        if selected_slice is not None
        else legacy._empty_fact_selection(ledger, status=status)  # noqa: SLF001
    )
    body = dict(selected)
    body.pop("receipt_sha256", None)
    audit_rendered_token_count = int(body["selected_fact_token_count"])
    body.update(
        {
            "audit_rendered_token_count": audit_rendered_token_count,
            "audit_selection_initial_token_floor": (
                INITIAL_FACT_SELECTION_AUDIT_TOKENS
            ),
            "audit_selection_token_budget": (
                selected_slice.max_tokens if selected_slice is not None else None
            ),
            "internal_audit_decoupled_from_provider_admission": True,
            "fact_token_budget": MAX_FACT_TOKENS,
            "provider_rendering": "compact_local_labels",
            "provider_visible_token_count": provider_token_count,
            "selected_fact_token_count": provider_token_count,
        }
    )
    return {**body, "receipt_sha256": identity_sha256(body)}


def _fact_required_envelopes(
    facts: Sequence[Mapping[str, Any]],
    manifests: Sequence[Mapping[str, Any]],
) -> set[str]:
    backing_ids = {str(fact["backing_evidence_id"]) for fact in facts}
    return {
        str(manifest["envelope_id"])
        for manifest in manifests
        if any(str(row["chunk_id"]) in backing_ids for row in manifest["raw_rows"])
    }


def _episode_manifests(
    groups: Sequence[Mapping[str, Any]],
    *,
    global_by_id: Mapping[str, Mapping[str, Any]],
    global_citations: Mapping[str, str],
) -> tuple[list[dict[str, Any]], str | None]:
    manifests: list[dict[str, Any]] = []
    for group in groups:
        manifest, error = legacy._episode_manifest(  # noqa: SLF001
            group, global_by_id, global_citations=global_citations
        )
        if error is not None:
            return [], error
        _require(manifest is not None, "episode manifest unexpectedly absent")
        manifests.append(manifest)
    return manifests, None


def _sealed_audit(body: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = copy.deepcopy(dict(body))
    return {**unsigned, "receipt_sha256": identity_sha256(unsigned)}


def _compose_arm(
    source_arm: Mapping[str, Any],
    *,
    dated_question: str,
    index: UserLedEnvelopeShadowIndex,
    candidate_binding: Mapping[str, Any],
) -> dict[str, Any]:
    parent = [copy.deepcopy(row) for row in source_arm["packed_evidence"]]
    active = legacy._active_source_ids(parent)  # noqa: SLF001
    candidates = legacy._candidate_rows(index, active)  # noqa: SLF001
    conserved, reason = legacy._conserves_parent_sources(  # noqa: SLF001
        parent, candidates=candidates, active_source_ids=active
    )
    if not conserved:
        return legacy._exact_operation_a_fallback(  # noqa: SLF001
            source_arm, reason, candidate_binding=candidate_binding
        )
    spec = legacy.compile_typed_operator_spec(dated_question)
    episode_selection = select_spine_episodes(
        index,
        active_source_ids=active,
        dated_question=dated_question,
        parent_packed_rows=parent,
        typed_spec=spec,
    )
    global_rows, global_omitted, _legacy_dedup = legacy._select_global_raw(  # noqa: SLF001
        parent
    )
    operation_a_reference = legacy._operation_a_reference(source_arm)  # noqa: SLF001
    _require(not global_omitted, "monotone parent lane omitted a sealed row")
    global_by_id = {str(row["evidence_id"]): row for row in global_rows}
    global_citation_manifest = legacy._global_citation_manifest(global_rows)  # noqa: SLF001
    global_citations = {
        str(entry["evidence_id"]): str(entry["citation"])
        for entry in global_citation_manifest["entries"]
    }

    base_groups = list(episode_selection["selected_groups"])
    base_manifests, manifest_error = _episode_manifests(
        base_groups,
        global_by_id=global_by_id,
        global_citations=global_citations,
    )
    if manifest_error is not None:
        return legacy._exact_operation_a_fallback(  # noqa: SLF001
            source_arm, manifest_error, candidate_binding=candidate_binding
        )
    selected_fact_rows, candidate_fact_rows, fact_input_selection = _fact_input_rows(
        global_rows,
        base_groups,
        index=index,
        active_source_ids=active,
    )
    ledger = compile_query_fact_ledger(
        dated_question,
        selected_fact_rows,
        candidate_rows=candidate_fact_rows,
    )
    completion_anchor_ids, completion_operand_ids, completion_selection = (
        _numeric_completion_selection(
            dated_question,
            ledger,
            episode_selection,
            candidate_fact_rows,
            index=index,
        )
    )
    completion_groups, completion_hydration = _hydrate_numeric_completion_groups(
        completion_anchor_ids,
        completion_operand_ids,
        index=index,
        global_rows=global_rows,
    )
    groups = base_groups
    manifests = base_manifests
    fact_backing_hydration = _sealed_audit(
        {
            "hydrated_bindings": [],
            "hydration_policy": "selected_candidate_fact_exact_owner_neighborhood",
            "unresolved_backing_evidence_ids": [],
        }
    )
    fact_slice: FactLedgerSlice | None = None
    mandatory_fact_count = _mandatory_fact_count(ledger)
    initial_fact_count = min(MAX_FACTS, len(ledger.facts))
    _require(initial_fact_count >= 1, "compiled fact ledger unexpectedly empty")
    fact_status = (
        "mandatory_coverage_exceeds_fact_count_cap"
        if mandatory_fact_count > initial_fact_count
        else "mandatory_provider_coverage_did_not_fit"
    )
    fact_provider_tokens = 0
    advisory_text = ""
    advisory_body: dict[str, Any] | None = None
    required_envelopes: set[str] = set()
    provider_admission_attempts: list[dict[str, Any]] = []
    audit_token_floor = INITIAL_FACT_SELECTION_AUDIT_TOKENS
    requested_fact_counts = (
        range(initial_fact_count, mandatory_fact_count - 1, -1)
        if mandatory_fact_count <= initial_fact_count
        else ()
    )
    for requested_fact_count in requested_fact_counts:
        candidate_slice, audit_token_budget = _select_audit_fact_slice(
            ledger,
            max_facts=requested_fact_count,
            initial_tokens=audit_token_floor,
        )
        audit_token_floor = max(audit_token_floor, audit_token_budget)
        candidate_facts = _fact_projection(candidate_slice)
        attempt: dict[str, Any] = {
            "audit_rendered_token_count": candidate_slice.rendered_token_count,
            "audit_token_budget": audit_token_budget,
            "mandatory_fact_count": mandatory_fact_count,
            "requested_fact_count": requested_fact_count,
            "selected_fact_count": len(candidate_facts),
        }
        candidate_groups, candidate_hydration = _hydrate_fact_backing_groups(
            (),
            candidate_facts,
            index=index,
            global_rows=global_rows,
        )
        candidate_manifests, backing_manifest_error = _episode_manifests(
            candidate_groups,
            global_by_id=global_by_id,
            global_citations=global_citations,
        )
        unresolved_backing = candidate_hydration[
            "unresolved_backing_evidence_ids"
        ]
        if backing_manifest_error is not None or unresolved_backing:
            fact_status = (
                f"fact_backing_manifest_invalid:{backing_manifest_error}"
                if backing_manifest_error is not None
                else "candidate_fact_backing_unavailable"
            )
            attempt.update(
                {
                    "decision": fact_status,
                    "unresolved_backing_evidence_count": len(unresolved_backing),
                }
            )
            provider_admission_attempts.append(attempt)
            continue
        candidate_required = _fact_required_envelopes(
            candidate_facts, candidate_manifests
        )
        candidate_represented_ids = _represented_raw_evidence_ids(
            global_rows, candidate_manifests
        )
        candidate_backing_ids = {
            str(fact["backing_evidence_id"]) for fact in candidate_facts
        }
        if not candidate_backing_ids <= set(candidate_represented_ids):
            fact_status = "candidate_fact_backing_not_rendered"
            attempt["decision"] = fact_status
            provider_admission_attempts.append(attempt)
            continue
        labels = _provider_label_index(global_rows, candidate_manifests)
        if not candidate_backing_ids <= set(labels):
            fact_status = "candidate_fact_backing_has_no_provider_label"
            attempt["decision"] = fact_status
            provider_admission_attempts.append(attempt)
            continue
        candidate_advisory, candidate_advisory_body = _compile_ledger_advisory(
            dated_question,
            ledger,
            candidate_slice,
            represented_backing_evidence_ids=candidate_represented_ids,
        )
        overlay = _provider_overlay_accounting(
            global_rows,
            candidate_manifests,
            candidate_facts,
            advisory_text=candidate_advisory,
        )
        fact_section_tokens = int(overlay["fact_section_token_count"])
        advisory_tokens = count_tokens(candidate_advisory)
        provider_tokens = int(overlay["fact_advisory_token_count"])
        context = _render_context(
            global_rows,
            candidate_manifests,
            candidate_facts,
            advisory_text=candidate_advisory,
        )
        fits, context_tokens, workspace_tokens = _fits(dated_question, context)
        attempt.update(
            {
                "advisory_token_count": advisory_tokens,
                "context_token_count": context_tokens,
                "fact_section_token_count": fact_section_tokens,
                "provider_overlay_token_count": provider_tokens,
                "provider_overlay_within_cap": provider_tokens <= MAX_FACT_TOKENS,
                "prompt_within_caps": fits,
                "workspace_token_count": workspace_tokens,
            }
        )
        if fits and provider_tokens <= MAX_FACT_TOKENS:
            attempt["decision"] = "admitted"
            provider_admission_attempts.append(attempt)
            fact_slice = candidate_slice
            fact_status = "selected_with_mandatory_coverage"
            fact_provider_tokens = provider_tokens
            required_envelopes = candidate_required
            advisory_text = candidate_advisory
            advisory_body = candidate_advisory_body
            groups = candidate_groups
            manifests = candidate_manifests
            fact_backing_hydration = candidate_hydration
            break
        if requested_fact_count != mandatory_fact_count:
            attempt["decision"] = "provider_fact_count_backoff"
        elif provider_tokens > MAX_FACT_TOKENS and fits:
            attempt["decision"] = "mandatory_provider_coverage_did_not_fit"
        elif provider_tokens <= MAX_FACT_TOKENS and not fits:
            attempt["decision"] = "mandatory_fact_backing_prompt_did_not_fit"
        else:
            attempt["decision"] = "mandatory_fact_packet_did_not_fit"
        provider_admission_attempts.append(attempt)
        fact_status = str(attempt["decision"])

    facts = _fact_projection(fact_slice)
    if advisory_body is None:
        advisory_text, advisory_body = _compile_ledger_advisory(
            dated_question,
            ledger,
            None,
            represented_backing_evidence_ids=_represented_raw_evidence_ids(
                global_rows, ()
            ),
        )
    required_groups = groups if fact_slice is not None else []
    required_manifests = manifests if fact_slice is not None else []
    admitted_group_by_id = {
        str(group["envelope_id"]): copy.deepcopy(dict(group))
        for group in required_groups
    }
    admitted_manifest_by_id = {
        str(manifest["envelope_id"]): copy.deepcopy(dict(manifest))
        for manifest in required_manifests
    }
    admitted_order = [str(group["envelope_id"]) for group in required_groups]
    rejected_manifests: list[dict[str, str]] = []
    completion_text = ""
    completion_bindings: list[dict[str, str]] = []
    completion_admission: dict[str, Any] = {
        "decision": "not_applicable",
        "prompt_rejection_reason": None,
    }
    if completion_operand_ids:
        trial_groups = _merge_episode_group_sets(
            list(admitted_group_by_id.values()), completion_groups
        )
        trial_manifests, completion_manifest_error = _episode_manifests(
            trial_groups,
            global_by_id=global_by_id,
            global_citations=global_citations,
        )
        expected_completion_ids = set(
            map(str, [*completion_anchor_ids, *completion_operand_ids])
        )
        represented_completion_ids = set(
            _represented_raw_evidence_ids(global_rows, trial_manifests)
        )
        trial_overlay = _provider_overlay_accounting(
            global_rows,
            trial_manifests,
            facts,
            advisory_text=advisory_text,
            completion_selection=completion_selection,
        )
        trial_completion_text = str(trial_overlay["completion_text"])
        trial_bindings = list(trial_overlay["completion_bindings"])
        trial_fact_provider_tokens = int(
            trial_overlay["fact_advisory_token_count"]
        )
        trial_provider_overlay_tokens = int(trial_overlay["total_token_count"])
        trial_tail = str(trial_overlay["provider_tail"])
        trial_context = _render_context(
            global_rows,
            trial_manifests,
            facts,
            advisory_text=trial_tail,
        )
        trial_fits, trial_context_tokens, trial_workspace_tokens = _fits(
            dated_question, trial_context
        )
        complete_representation = (
            completion_manifest_error is None
            and expected_completion_ids <= represented_completion_ids
            and len(trial_bindings) == len(completion_operand_ids)
        )
        completion_admission = {
            "complete_representation": complete_representation,
            "context_token_count": trial_context_tokens,
            "decision": "rejected_completion_representation",
            "expected_evidence_ids": sorted(expected_completion_ids),
            "manifest_error": completion_manifest_error,
            "prompt_rejection_reason": None,
            "fact_advisory_token_count": trial_fact_provider_tokens,
            "provider_overlay_token_count": trial_provider_overlay_tokens,
            "provider_overlay_within_cap": (
                trial_fact_provider_tokens <= MAX_FACT_TOKENS
                and trial_provider_overlay_tokens <= MAX_FACT_TOKENS
            ),
            "represented_evidence_ids": sorted(
                expected_completion_ids & represented_completion_ids
            ),
            "workspace_token_count": trial_workspace_tokens,
        }
        if (
            complete_representation
            and trial_fits
            and trial_fact_provider_tokens <= MAX_FACT_TOKENS
            and trial_provider_overlay_tokens <= MAX_FACT_TOKENS
        ):
            completion_admission["decision"] = "admitted_before_optional_episodes"
            for group, manifest in zip(trial_groups, trial_manifests, strict=True):
                envelope_id = str(group["envelope_id"])
                if envelope_id not in admitted_group_by_id:
                    admitted_order.append(envelope_id)
                admitted_group_by_id[envelope_id] = group
                admitted_manifest_by_id[envelope_id] = manifest
            completion_text = trial_completion_text
            completion_bindings = trial_bindings
        elif complete_representation and not trial_fits:
            completion_admission["decision"] = "rejected_completion_prompt_cap"
            completion_admission["prompt_rejection_reason"] = (
                "hard_prompt_cap_after_mandatory_facts"
            )
        elif complete_representation:
            completion_admission["decision"] = "rejected_completion_provider_cap"
            completion_admission["prompt_rejection_reason"] = (
                "hard_provider_overlay_cap_after_mandatory_facts"
            )

        if (
            complete_representation
            and completion_admission["decision"].startswith("rejected_completion_")
            and fact_slice is not None
            and len(facts) > mandatory_fact_count
        ):
            completion_backoff_attempts: list[dict[str, Any]] = []
            for requested_fact_count in range(
                len(facts) - 1, mandatory_fact_count - 1, -1
            ):
                candidate_slice, audit_token_budget = _select_audit_fact_slice(
                    ledger,
                    max_facts=requested_fact_count,
                    initial_tokens=audit_token_floor,
                )
                audit_token_floor = max(audit_token_floor, audit_token_budget)
                candidate_facts = _fact_projection(candidate_slice)
                attempt: dict[str, Any] = {
                    "audit_rendered_token_count": candidate_slice.rendered_token_count,
                    "audit_token_budget": audit_token_budget,
                    "mandatory_fact_count": mandatory_fact_count,
                    "requested_fact_count": requested_fact_count,
                    "selected_fact_count": len(candidate_facts),
                }
                candidate_fact_groups, candidate_hydration = (
                    _hydrate_fact_backing_groups(
                        (),
                        candidate_facts,
                        index=index,
                        global_rows=global_rows,
                    )
                )
                candidate_groups = _merge_episode_group_sets(
                    candidate_fact_groups, completion_groups
                )
                candidate_manifests, candidate_manifest_error = _episode_manifests(
                    candidate_groups,
                    global_by_id=global_by_id,
                    global_citations=global_citations,
                )
                unresolved_backing = candidate_hydration[
                    "unresolved_backing_evidence_ids"
                ]
                if candidate_manifest_error is not None or unresolved_backing:
                    attempt["decision"] = (
                        f"fact_backing_manifest_invalid:{candidate_manifest_error}"
                        if candidate_manifest_error is not None
                        else "candidate_fact_backing_unavailable"
                    )
                    completion_backoff_attempts.append(attempt)
                    continue
                represented = set(
                    _represented_raw_evidence_ids(global_rows, candidate_manifests)
                )
                backing_ids = {
                    str(fact["backing_evidence_id"]) for fact in candidate_facts
                }
                if not backing_ids <= represented:
                    attempt["decision"] = "candidate_fact_backing_not_rendered"
                    completion_backoff_attempts.append(attempt)
                    continue
                candidate_advisory, candidate_advisory_body = (
                    _compile_ledger_advisory(
                        dated_question,
                        ledger,
                        candidate_slice,
                        represented_backing_evidence_ids=tuple(sorted(represented)),
                    )
                )
                candidate_overlay = _provider_overlay_accounting(
                    global_rows,
                    candidate_manifests,
                    candidate_facts,
                    advisory_text=candidate_advisory,
                    completion_selection=completion_selection,
                )
                candidate_completion_text = str(
                    candidate_overlay["completion_text"]
                )
                candidate_bindings = list(
                    candidate_overlay["completion_bindings"]
                )
                if (
                    not expected_completion_ids <= represented
                    or len(candidate_bindings) != len(completion_operand_ids)
                ):
                    attempt["decision"] = "completion_backing_not_rendered"
                    completion_backoff_attempts.append(attempt)
                    continue
                candidate_fact_section = str(candidate_overlay["fact_section"])
                candidate_fact_provider_tokens = int(
                    candidate_overlay["fact_advisory_token_count"]
                )
                candidate_provider_tokens = int(
                    candidate_overlay["total_token_count"]
                )
                candidate_tail = str(candidate_overlay["provider_tail"])
                candidate_context = _render_context(
                    global_rows,
                    candidate_manifests,
                    candidate_facts,
                    advisory_text=candidate_tail,
                )
                candidate_fits, candidate_context_tokens, candidate_workspace_tokens = (
                    _fits(dated_question, candidate_context)
                )
                attempt.update(
                    {
                        "advisory_token_count": count_tokens(candidate_advisory),
                        "context_token_count": candidate_context_tokens,
                        "fact_section_token_count": count_tokens(
                            candidate_fact_section
                        ),
                        "numeric_completion_token_count": count_tokens(
                            candidate_completion_text
                        ),
                        "prompt_within_caps": candidate_fits,
                        "provider_overlay_token_count": candidate_provider_tokens,
                        "provider_overlay_within_cap": (
                            candidate_fact_provider_tokens <= MAX_FACT_TOKENS
                            and candidate_provider_tokens <= MAX_FACT_TOKENS
                        ),
                        "workspace_token_count": candidate_workspace_tokens,
                    }
                )
                if (
                    candidate_fits
                    and candidate_fact_provider_tokens <= MAX_FACT_TOKENS
                    and candidate_provider_tokens <= MAX_FACT_TOKENS
                ):
                    attempt["decision"] = "admitted"
                    completion_backoff_attempts.append(attempt)
                    provider_admission_attempts[-1]["decision"] = (
                        "provider_fact_count_backoff_for_numeric_completion"
                    )
                    provider_admission_attempts[-1][
                        "numeric_completion_overlay_token_count"
                    ] = trial_provider_overlay_tokens
                    provider_admission_attempts.extend(completion_backoff_attempts)
                    fact_slice = candidate_slice
                    facts = candidate_facts
                    groups = candidate_groups
                    manifests = candidate_manifests
                    fact_backing_hydration = candidate_hydration
                    required_envelopes = {
                        str(group["envelope_id"]) for group in candidate_groups
                    }
                    advisory_text = candidate_advisory
                    advisory_body = candidate_advisory_body
                    fact_provider_tokens = candidate_fact_provider_tokens
                    admitted_group_by_id = {
                        str(group["envelope_id"]): group
                        for group in candidate_groups
                    }
                    admitted_manifest_by_id = {
                        str(manifest["envelope_id"]): manifest
                        for manifest in candidate_manifests
                    }
                    admitted_order = [
                        str(group["envelope_id"]) for group in candidate_groups
                    ]
                    completion_text = candidate_completion_text
                    completion_bindings = candidate_bindings
                    completion_admission.update(
                        {
                            "context_token_count": candidate_context_tokens,
                            "decision": "admitted_after_optional_fact_backoff",
                            "fact_advisory_token_count": (
                                candidate_fact_provider_tokens
                            ),
                            "prompt_rejection_reason": None,
                            "provider_overlay_token_count": (
                                candidate_provider_tokens
                            ),
                            "provider_overlay_within_cap": True,
                            "workspace_token_count": candidate_workspace_tokens,
                        }
                    )
                    break
                attempt["decision"] = (
                    "provider_fact_count_backoff_for_numeric_completion"
                    if requested_fact_count != mandatory_fact_count
                    else "mandatory_packet_plus_completion_did_not_fit"
                )
                completion_backoff_attempts.append(attempt)
            completion_admission["fact_backoff_attempts"] = (
                completion_backoff_attempts
            )

    optional_admitted = 0
    provider_overlay = _provider_overlay_accounting(
        global_rows,
        [admitted_manifest_by_id[value] for value in admitted_order],
        facts,
        advisory_text=advisory_text,
        completion_selection=(completion_selection if completion_text else None),
    )
    provider_tail = str(provider_overlay["provider_tail"])
    completion_text = str(provider_overlay["completion_text"])
    completion_bindings = list(provider_overlay["completion_bindings"])
    for group in base_groups:
        envelope_id = str(group["envelope_id"])
        if envelope_id in admitted_group_by_id:
            rejected_manifests.append(
                {
                    "envelope_id": envelope_id,
                    "reason": "overlapping_optional_expansion_deferred",
                }
            )
            continue
        manifest, manifest_error = _episode_manifests(
            [group],
            global_by_id=global_by_id,
            global_citations=global_citations,
        )
        if manifest_error is not None:
            rejected_manifests.append(
                {
                    "envelope_id": envelope_id,
                    "reason": f"optional_manifest_invalid:{manifest_error}",
                }
            )
            continue
        trial_manifests = [
            *[admitted_manifest_by_id[value] for value in admitted_order],
            manifest[0],
        ]
        trial_overlay = _provider_overlay_accounting(
            global_rows,
            trial_manifests,
            facts,
            advisory_text=advisory_text,
            completion_selection=(
                completion_selection if completion_text else None
            ),
        )
        trial_provider_tokens = int(trial_overlay["total_token_count"])
        trial_fact_provider_tokens = int(
            trial_overlay["fact_advisory_token_count"]
        )
        trial_provider_within_cap = (
            trial_fact_provider_tokens <= MAX_FACT_TOKENS
            and trial_provider_tokens <= MAX_FACT_TOKENS
        )
        trial_provider_tail = str(trial_overlay["provider_tail"])
        context = _render_context(
            global_rows,
            trial_manifests,
            facts,
            advisory_text=trial_provider_tail,
        )
        fits, _context_tokens, _workspace_tokens = _fits(dated_question, context)
        if fits and trial_provider_within_cap:
            admitted_group_by_id[envelope_id] = copy.deepcopy(dict(group))
            admitted_manifest_by_id[envelope_id] = manifest[0]
            admitted_order.append(envelope_id)
            optional_admitted += 1
            provider_overlay = trial_overlay
            provider_tail = trial_provider_tail
            completion_text = str(trial_overlay["completion_text"])
            completion_bindings = list(trial_overlay["completion_bindings"])
        else:
            rejected_manifests.append(
                {
                    "envelope_id": envelope_id,
                    "reason": (
                        "hard_provider_overlay_cap_after_fact_reserve"
                        if not trial_provider_within_cap
                        else "hard_prompt_cap_after_fact_reserve"
                    ),
                }
            )
    admitted_groups = [admitted_group_by_id[value] for value in admitted_order]
    admitted_manifests = [admitted_manifest_by_id[value] for value in admitted_order]
    if not admitted_manifests and fact_slice is None and not completion_text:
        return legacy._exact_operation_a_fallback(  # noqa: SLF001
            source_arm,
            "no_admissible_user_led_episode_or_fact",
            candidate_binding=candidate_binding,
        )

    represented_ids = set(
        _represented_raw_evidence_ids(global_rows, admitted_manifests)
    )
    if any(str(fact["backing_evidence_id"]) not in represented_ids for fact in facts):
        return legacy._exact_operation_a_fallback(  # noqa: SLF001
            source_arm,
            "fact_backing_raw_not_rendered",
            candidate_binding=candidate_binding,
        )

    final_overlay = _provider_overlay_accounting(
        global_rows,
        admitted_manifests,
        facts,
        advisory_text=advisory_text,
        completion_selection=(completion_selection if completion_text else None),
    )
    final_fact_section = str(final_overlay["fact_section"])
    fact_provider_tokens = int(final_overlay["fact_advisory_token_count"])
    provider_overlay_tokens = int(final_overlay["total_token_count"])
    provider_tail = str(final_overlay["provider_tail"])
    completion_text = str(final_overlay["completion_text"])
    completion_bindings = list(final_overlay["completion_bindings"])
    context = _render_context(
        global_rows, admitted_manifests, facts, advisory_text=provider_tail
    )
    messages, context_tokens, workspace_tokens = _prompt(dated_question, context)
    _require(
        context_tokens <= MAX_CONTEXT_TOKENS
        and workspace_tokens <= MAX_WORKSPACE_TOKENS,
        "admitted fact-reserved packet exceeded its hard prompt cap",
    )
    payload = hot._canonical_json_bytes({"messages": messages})  # noqa: SLF001
    _require(
        fact_provider_tokens <= MAX_FACT_TOKENS
        and provider_overlay_tokens <= MAX_FACT_TOKENS,
        (
            "final compact fact/advisory overlay exceeded its reserved budget: "
            f"fact_advisory={fact_provider_tokens} "
            f"total={provider_overlay_tokens} cap={MAX_FACT_TOKENS} "
            f"question_sha256={quote_sha256(dated_question)}"
        ),
    )
    rendered_raw_ids = [
        str(row["chunk_id"])
        for manifest in admitted_manifests
        for row in manifest["raw_rows"]
    ]
    selected_raw_ids = [
        str(row["chunk_id"]) for group in admitted_groups for row in group["rows"]
    ]
    raw_dedup = [
        {
            "address": str(reference["episode_chunk_id"]),
            "excluded_episode_chunk_id": str(reference["episode_chunk_id"]),
            "retained_global_evidence_id": str(reference["global_evidence_id"]),
        }
        for manifest in admitted_manifests
        for reference in manifest["global_refs"]
    ]
    fact_ledger = _fact_selection_audit(
        ledger,
        selected_slice=fact_slice,
        status=fact_status,
        provider_token_count=fact_provider_tokens,
    )
    fact_reservation = _sealed_audit(
        {
            "admitted_optional_episode_count": optional_admitted,
            "fact_provider_token_cap": MAX_FACT_TOKENS,
            "fact_provider_token_count": fact_provider_tokens,
            "initial_provider_fact_count": initial_fact_count,
            "internal_audit_encoding_controls_provider_admission": False,
            "mandatory_fact_count": mandatory_fact_count,
            "mandatory_fact_hydration_policy": "empty_then_minimal_opener_plus_backing",
            "provider_admission_attempts": provider_admission_attempts,
            "provider_admission_policy": (
                "descending_fact_count_preserve_mandatory_coverage"
            ),
            "provider_admission_attempt_count": len(provider_admission_attempts),
            "provider_fact_count_backoff_steps": (
                initial_fact_count - len(facts)
                if fact_slice is not None
                else max(0, initial_fact_count - mandatory_fact_count)
            ),
            "provider_selected_fact_count": len(facts),
            "provider_total_overlay_token_count": provider_overlay_tokens,
            "required_envelope_ids": sorted(required_envelopes),
            "reserved_before_optional_episode_expansion": True,
            "selection_status": fact_status,
        }
    )
    provider_provenance = _provider_provenance_manifest(
        global_citation_manifest, admitted_manifests, facts
    )
    fact_advisory = _sealed_audit(
        {
            "provider_text_sha256": quote_sha256(advisory_text),
            "provider_token_count": count_tokens(advisory_text),
            "typed_reducer_audit": advisory_body,
        }
    )
    numeric_slot_completion = _sealed_audit(
        {
            **completion_selection,
            "admission": completion_admission,
            "hydration": completion_hydration,
            "provider_bindings": completion_bindings,
            "provider_text": completion_text,
            "provider_text_sha256": quote_sha256(completion_text),
            "provider_token_count": count_tokens(completion_text),
            "unresolved_slot_ids_after": list(ledger.unresolved_slot_ids),
            "unresolved_slot_ids_before": list(ledger.unresolved_slot_ids),
        }
    )
    return {
        "active_source_ids": list(active),
        "candidate_universe_binding": copy.deepcopy(dict(candidate_binding)),
        "context_token_proxy": context_tokens,
        "dedup_stage": "post_independent_selection_exact_evidence_id",
        "episode_selection": episode_selection,
        "episode_manifests": admitted_manifests,
        "episode_manifest_rejections": rejected_manifests,
        "fact_budget_reservation": fact_reservation,
        "fact_backing_hydration": fact_backing_hydration,
        "fact_advisory": fact_advisory,
        "fact_input_selection": fact_input_selection,
        "fact_ledger": fact_ledger,
        "fallback_reason": None,
        "global_citation_manifest": global_citation_manifest,
        "global_raw_omitted_evidence_ids": global_omitted,
        "global_raw_selected_evidence_ids": [
            str(row["evidence_id"]) for row in global_rows
        ],
        "mode": "spine_indexed_episodic_fact_ledger",
        "numeric_slot_completion": numeric_slot_completion,
        "operation_a_reference": operation_a_reference,
        "overlay_context_token_delta": (
            context_tokens - int(operation_a_reference["context_token_proxy"])
        ),
        "overlay_workspace_token_delta": (
            workspace_tokens
            - int(operation_a_reference["prompt_workspace_token_proxy"])
        ),
        "packed_chunk_ids": [
            *[row["evidence_id"] for row in global_rows],
            *rendered_raw_ids,
        ],
        "parent_all_rendered": True,
        "parent_population_sha256": identity_sha256(parent),
        "parent_rows_protected": True,
        "prompt_token_proxy": hot.count_chat_prompt_token_proxy(messages),
        "prompt_workspace_token_proxy": workspace_tokens,
        "provider_messages": messages,
        "provider_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "provider_payload_utf8_bytes": len(payload),
        "provider_provenance_manifest": provider_provenance,
        "raw_collision_bindings": [
            copy.deepcopy(collision)
            for manifest in admitted_manifests
            for collision in manifest["representation_collisions"]
        ],
        "raw_dedup_bindings": raw_dedup,
        "raw_selected_chunk_ids": selected_raw_ids,
        "rendered_fact_ids": [str(fact["fact_id"]) for fact in facts],
        "rendered_parent_evidence_ids": [
            str(row["evidence_id"]) for row in global_rows
        ],
        "rendered_partitions": {
            "episode_global_ref_count": sum(
                len(manifest["global_refs"]) for manifest in admitted_manifests
            ),
            "episode_manifest_count": len(admitted_manifests),
            "episode_raw_chunk_count": len(rendered_raw_ids),
            "episode_representation_collision_count": sum(
                len(manifest["representation_collisions"])
                for manifest in admitted_manifests
            ),
            "fact_count": len(facts),
            "global_parent_count": len(global_rows),
        },
        "rendered_raw_chunk_ids": rendered_raw_ids,
        "source_conservation_validated": True,
    }


def _project_selection(
    construction: Mapping[str, Any],
    *,
    construction_sha256: str,
    runtime_sha256: str,
    replay_sha256: str,
    retrieval_path: Path,
    store_root: Path,
) -> dict[str, Any]:
    with _legacy_scope(
        DEFAULT_OUTPUT_ROOT=DEFAULT_OUTPUT_ROOT,
        FORMAT=FORMAT,
        ROW_FORMAT=ROW_FORMAT,
        USER_TEMPLATE=USER_TEMPLATE,
        _compose_arm=_compose_arm,
        _implementation_identity=_implementation_identity,
    ):
        selection = legacy._project_selection(  # noqa: SLF001
            construction,
            construction_sha256=construction_sha256,
            runtime_sha256=runtime_sha256,
            replay_sha256=replay_sha256,
            retrieval_path=retrieval_path,
            store_root=store_root,
        )
    selection["budget"].update(
        {
            "fact_reservation_order": "before_optional_episode_expansion",
            "fact_selection_audit_initial_token_floor": (
                INITIAL_FACT_SELECTION_AUDIT_TOKENS
            ),
            "fact_selection_audit_policy": (
                "grow_to_requested_fact_count_not_provider_coupled"
            ),
            "provider_fact_admission_policy": (
                "descending_fact_count_preserve_mandatory_coverage"
            ),
            "numeric_completion_envelope_cap": MAX_NUMERIC_COMPLETION_ENVELOPES,
            "numeric_completion_policy": (
                "unresolved_slot_selected_anchor_same_source_raw"
            ),
            "numeric_completion_rows_per_slot_source": (
                MAX_NUMERIC_COMPLETION_ROWS_PER_SLOT_SOURCE
            ),
            "numeric_completion_turn_cap": MAX_NUMERIC_COMPLETION_TURNS,
            "physical_transition_lane_policy": "typed_dynamic_1_2_4",
            "provider_provenance_policy": "compact_labels_sealed_audit_mapping",
            "specialist_union_policy": "independent_reserved_selection",
        }
    )
    assert_gold_blind(selection, path="hot_v7_spine_episode_fact_selection")
    return selection


def _validate_successor_extensions(selection: Mapping[str, Any]) -> None:
    for ordinal, row in enumerate(selection["questions"]):
        arm = row["arms"]["a3_protected_union"]
        if arm["mode"] != "spine_indexed_episodic_fact_ledger":
            continue
        for name in (
            "fact_advisory",
            "fact_backing_hydration",
            "fact_budget_reservation",
            "fact_input_selection",
            "numeric_slot_completion",
            "provider_provenance_manifest",
        ):
            body = arm.get(name)
            _require(type(body) is dict, f"v7 {name} missing at {ordinal}")
            unsigned = dict(body)
            receipt = unsigned.pop("receipt_sha256", None)
            _require(
                receipt == identity_sha256(unsigned),
                f"v7 {name} receipt changed at {ordinal}",
            )
        reservation = arm["fact_budget_reservation"]
        _require(
            reservation["reserved_before_optional_episode_expansion"] is True
            and reservation["fact_provider_token_count"] <= MAX_FACT_TOKENS,
            f"v7 fact reserve changed at {ordinal}",
        )
        _require(
            reservation.get("provider_total_overlay_token_count", 0)
            <= MAX_FACT_TOKENS
            and reservation.get("mandatory_fact_hydration_policy")
            == "empty_then_minimal_opener_plus_backing",
            f"v7 compact overlay/minimal backing changed at {ordinal}",
        )
        attempts = reservation.get("provider_admission_attempts")
        _require(
            reservation.get("internal_audit_encoding_controls_provider_admission")
            is False
            and reservation.get("provider_admission_policy")
            == "descending_fact_count_preserve_mandatory_coverage"
            and type(attempts) is list
            and all(type(attempt) is dict for attempt in attempts),
            f"v7 provider fact admission audit changed at {ordinal}",
        )
        requested_counts = [
            int(attempt["requested_fact_count"]) for attempt in attempts
        ]
        audit_token_budgets = [
            int(attempt["audit_token_budget"]) for attempt in attempts
        ]
        _require(
            requested_counts
            == list(
                range(
                    reservation["initial_provider_fact_count"],
                    reservation["initial_provider_fact_count"]
                    - len(requested_counts),
                    -1,
                )
            )
            and reservation.get("provider_admission_attempt_count")
            == len(attempts)
            and all(
                int(attempt["selected_fact_count"])
                == int(attempt["requested_fact_count"])
                and int(attempt["selected_fact_count"])
                >= reservation["mandatory_fact_count"]
                and int(attempt["audit_rendered_token_count"])
                <= int(attempt["audit_token_budget"])
                for attempt in attempts
            )
            and all(
                budget >= INITIAL_FACT_SELECTION_AUDIT_TOKENS
                for budget in audit_token_budgets
            )
            and all(
                later >= earlier
                for earlier, later in zip(
                    audit_token_budgets, audit_token_budgets[1:], strict=False
                )
            ),
            f"v7 provider fact count backoff changed at {ordinal}",
        )
        if reservation["selection_status"] == "selected_with_mandatory_coverage":
            _require(
                bool(attempts)
                and attempts[-1].get("decision") == "admitted"
                and reservation["provider_selected_fact_count"]
                == attempts[-1]["selected_fact_count"]
                and reservation["provider_fact_count_backoff_steps"]
                == reservation["initial_provider_fact_count"]
                - reservation["provider_selected_fact_count"]
                and arm["fact_ledger"]["selected_fact_count"]
                == reservation["provider_selected_fact_count"]
                and arm["fact_ledger"]["audit_selection_token_budget"]
                == attempts[-1]["audit_token_budget"]
                and arm["fact_ledger"]["audit_rendered_token_count"]
                == attempts[-1]["audit_rendered_token_count"],
                f"v7 admitted fact attempt changed at {ordinal}",
            )
        elif (
            reservation["selection_status"]
            == "mandatory_coverage_exceeds_fact_count_cap"
        ):
            _require(
                reservation["mandatory_fact_count"]
                > reservation["initial_provider_fact_count"]
                and not attempts
                and reservation["provider_selected_fact_count"] == 0
                and reservation["provider_fact_count_backoff_steps"] == 0,
                f"v7 structural fact-count failure changed at {ordinal}",
            )
        else:
            _require(
                bool(attempts)
                and attempts[-1]["requested_fact_count"]
                == reservation["mandatory_fact_count"]
                and attempts[-1]["selected_fact_count"]
                == reservation["mandatory_fact_count"]
                and attempts[-1].get("decision")
                not in {"admitted", "provider_fact_count_backoff"}
                and reservation["selection_status"]
                == attempts[-1].get("decision")
                and reservation["provider_selected_fact_count"] == 0
                and reservation["provider_fact_count_backoff_steps"]
                == reservation["initial_provider_fact_count"]
                - reservation["mandatory_fact_count"],
                f"v7 minimum mandatory fact attempt changed at {ordinal}",
            )
        advisory_envelope = arm["fact_advisory"]
        typed_audit = advisory_envelope.get("typed_reducer_audit")
        _require(
            type(typed_audit) is dict,
            f"v7 typed reducer audit missing at {ordinal}",
        )
        assert_gold_blind(typed_audit, path=f"hot_v7_typed_reducer_audit[{ordinal}]")
        provider_advisory = typed_audit.get("provider_advisory")
        _require(
            type(provider_advisory) is dict
            and type(provider_advisory.get("text")) is str
            and type(provider_advisory.get("emitted")) is bool,
            f"v7 typed reducer provider advisory changed at {ordinal}",
        )
        advisory_text = provider_advisory["text"]
        _require(
            provider_advisory["emitted"] == bool(advisory_text)
            and provider_advisory.get("text_sha256") == quote_sha256(advisory_text)
            and advisory_envelope["provider_text_sha256"]
            == quote_sha256(advisory_text)
            and advisory_envelope["provider_token_count"]
            == count_tokens(advisory_text),
            f"v7 typed reducer advisory accounting changed at {ordinal}",
        )
        reduction = typed_audit.get("reduction")
        supported = type(reduction) is dict and reduction.get("status") == "supported"
        _require(
            bool(advisory_text) == supported,
            f"v7 unsupported typed reduction emitted text at {ordinal}",
        )
        if advisory_text:
            _require(
                advisory_text in arm["provider_messages"][1]["content"],
                f"v7 typed reducer advisory missing from prompt at {ordinal}",
            )
        forbidden_ids = {
            *typed_audit.get("represented_backing_evidence_ids", []),
            *(
                binding.get("backing_evidence_id")
                for binding in typed_audit.get("local_fact_bindings", [])
            ),
            *(
                binding.get("fact_id")
                for binding in typed_audit.get("local_fact_bindings", [])
            ),
        }
        _require(
            not any(
                type(value) is str and value and value in advisory_text
                for value in forbidden_ids
            ),
            f"v7 typed reducer advisory exposed raw IDs at {ordinal}",
        )
        completion = arm["numeric_slot_completion"]
        completion_text = completion.get("provider_text")
        _require(
            type(completion_text) is str
            and completion.get("frontier_closed") is False
            and completion.get("slot_bindings_added") == 0
            and completion.get("unresolved_slot_ids_before")
            == arm["fact_ledger"]["unresolved_slot_ids"]
            and completion.get("unresolved_slot_ids_after")
            == arm["fact_ledger"]["unresolved_slot_ids"]
            and completion.get("provider_text_sha256")
            == quote_sha256(completion_text)
            and completion.get("provider_token_count")
            == count_tokens(completion_text),
            f"v7 numeric completion closure/accounting changed at {ordinal}",
        )
        if completion_text:
            _require(
                completion_text in arm["provider_messages"][1]["content"],
                f"v7 numeric completion text missing at {ordinal}",
            )
        completion_ids = {
            str(binding[key])
            for binding in completion.get("provider_bindings", [])
            for key in ("anchor_evidence_id", "backing_evidence_id")
        }
        _require(
            not any(value in completion_text for value in completion_ids),
            f"v7 numeric completion exposed opaque IDs at {ordinal}",
        )
        expected_provenance = _provider_provenance_manifest(
            arm["global_citation_manifest"],
            arm["episode_manifests"],
            arm["fact_ledger"]["selected_facts"],
        )
        _require(
            arm["provider_provenance_manifest"] == expected_provenance,
            f"v7 provider provenance mapping changed at {ordinal}",
        )


def _load_selection(output_root: Path) -> tuple[dict[str, Any], str]:
    with _legacy_scope(
        DEFAULT_OUTPUT_ROOT=DEFAULT_OUTPUT_ROOT,
        FORMAT=FORMAT,
        ROW_FORMAT=ROW_FORMAT,
        USER_TEMPLATE=USER_TEMPLATE,
        _implementation_identity=_implementation_identity,
    ):
        selection, digest = legacy._load_selection(output_root)  # noqa: SLF001
    _validate_successor_extensions(selection)
    return selection, digest


def load_selection(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[dict[str, Any], str]:
    return _load_selection(output_root.resolve())


def materialize(
    *,
    source_root: Path,
    retrieval_path: Path,
    store_root: Path,
    output_root: Path,
) -> str:
    _require(not output_root.exists(), "output root must be unique and absent")
    construction, construction_sha, runtime_sha, replay_sha = (
        legacy.source_assay._load_source(source_root)  # noqa: SLF001
    )
    selection = _project_selection(
        construction,
        construction_sha256=construction_sha,
        runtime_sha256=runtime_sha,
        replay_sha256=replay_sha,
        retrieval_path=retrieval_path,
        store_root=store_root,
    )
    digest = hot._atomic_write_json(output_root / SELECTION_NAME, selection)  # noqa: SLF001
    print(
        "Spine-episodic fact-reserved v7: "
        f"{selection['question_count']} prompts; "
        f"episode_raw={selection['aggregate']['rendered_episode_raw_chunk_count']}; "
        f"facts={selection['aggregate']['rendered_fact_count']}; selection={digest}",
        flush=True,
    )
    return digest


def replay(
    *,
    source_root: Path,
    retrieval_path: Path,
    store_root: Path,
    output_root: Path,
) -> str:
    sealed, sealed_sha = hot._read_json_artifact(output_root / SELECTION_NAME)  # noqa: SLF001
    _require(not (output_root / REPLAY_NAME).exists(), "refusing to overwrite replay")
    construction, construction_sha, runtime_sha, replay_sha = (
        legacy.source_assay._load_source(source_root)  # noqa: SLF001
    )
    rebuilt = _project_selection(
        construction,
        construction_sha256=construction_sha,
        runtime_sha256=runtime_sha,
        replay_sha256=replay_sha,
        retrieval_path=retrieval_path,
        store_root=store_root,
    )
    _require(rebuilt == sealed, "v7 fact-reserved selection replay changed")
    body = {
        "canonical_semantic_identity": True,
        "format": f"{FORMAT}-semantic-replay-v1",
        "provider_calls": 0,
        "question_count": len(rebuilt["questions"]),
        "row_receipts_sha256": identity_sha256(
            [row["row_receipt_sha256"] for row in rebuilt["questions"]]
        ),
        "selection_sha256": sealed_sha,
        "source_construction_sha256": construction_sha,
        "source_replay_sha256": replay_sha,
        "source_runtime_sha256": runtime_sha,
    }
    artifact = {**body, "replay_receipt_sha256": identity_sha256(body)}
    assert_gold_blind(artifact, path="hot_v7_spine_episode_replay")
    digest = hot._atomic_write_json(output_root / REPLAY_NAME, artifact)  # noqa: SLF001
    print(f"Spine-episodic v7 replay: 100/100 exact; replay={digest}", flush=True)
    return digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("materialize")
    commands.add_parser("replay")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "materialize":
        materialize(
            source_root=args.source_root.resolve(),
            retrieval_path=args.retrieval.resolve(),
            store_root=args.store_root.resolve(),
            output_root=args.output_root.resolve(),
        )
    elif args.command == "replay":
        replay(
            source_root=args.source_root.resolve(),
            retrieval_path=args.retrieval.resolve(),
            store_root=args.store_root.resolve(),
            output_root=args.output_root.resolve(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "FORMAT",
    "_load_selection",
    "load_selection",
    "materialize",
    "replay",
    "select_spine_episodes",
]
