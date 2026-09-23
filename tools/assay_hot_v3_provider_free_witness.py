#!/usr/bin/env python3
"""Gold-blind hot-v3 witness/link successor assay.

``construct`` authenticates immutable hot-v3 parent bytes, builds each
needed resident namespace index exactly once, and composes three independently
budgeted local lanes ahead of the parent packet. It never opens benchmark
answers or calls a model/provider. ``score`` is a separate post-hoc join.

Exact evidence-occurrence deduplication occurs only after every lane has
selected; excerpt occurrences cannot impersonate their backing full chunks.
The authoritative representative of every source present in the sealed parent
packet must survive the 7k-context/8k-workspace prefix. Otherwise the exact
sealed parent arm is reused byte-semantically.
"""

from __future__ import annotations

import argparse
import copy
import gc
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.search.packing.ranked_prefix_prompt import (
    RankedPrefixPromptPack,
    pack_ranked_prefix_prompt,
)
from tools import assay_hot_retrieval_1m as hot
from tools import assay_hot_retrieval_full100 as full100
from tools import assay_hot_retrieval_source_seed_hybrid_full100 as v3
from tools import assay_hot_v3_typed_operator_full100 as typed
from tools.assay_hot_v3_full_store_numeric import (
    _build_resident_index,
    _load_sealed_v3_rows,
)
from tools.matched_eval.contracts import assert_gold_blind, identity_sha256
from tools.matched_eval.hot_typed_witness import (
    build_hot_typed_witness_index,
    query_hot_typed_witnesses,
)
from tools.matched_eval.hot_v3_activated_turn_links import (
    build_hot_v3_activated_turn_link_index,
    select_hot_v3_activated_turn_links,
)
from tools.matched_eval.population import EXPECTED_RETRIEVAL_SHA256
from tools.matched_eval import profile_preference_specialist as profile_lane
from tools.matched_eval.profile_preference_specialist import (
    select_profile_preference_evidence,
)
from tools.matched_eval.query_expansion import load_locked_query_expansion_context


FORMAT = "memory-condense-hot-v3-provider-free-witness-construction-v2"
RUNTIME_FORMAT = "memory-condense-hot-v3-provider-free-witness-runtime-v2"
SCORE_FORMAT = "memory-condense-hot-v3-provider-free-witness-score-v2"
ROW_FORMAT = "memory-condense-hot-v3-provider-free-witness-row-v2"
COMPOSITION_FORMAT = "memory-condense-hot-v3-provider-free-composition-v2"
TIMING_FORMAT = "memory-condense-hot-v3-provider-free-witness-timing-v2"
POLICY_ID = "hot-v3-profile-typed-witness-activated-turn-successor-v2"
EXCERPT_OCCURRENCE_FORMAT = "memory-condense-hot-v3-excerpt-occurrence-v1"
LEGACY_PARENT_PROJECTION_OCCURRENCE_FORMAT = (
    "memory-condense-hot-v3-legacy-parent-projection-occurrence-v1"
)

MAX_CONTEXT_TOKENS = v3.MAX_CONTEXT_TOKENS
MAX_PROMPT_TOKENS = v3.MAX_PROMPT_TOKENS
OUTPUT_TOKEN_RESERVE = v3.OUTPUT_TOKEN_RESERVE
EXPECTED_QUESTION_COUNT = v3.EXPECTED_QUESTION_COUNT
PARENT_FALLBACK_SEED_LIMIT = 2

DEFAULT_V3_ROOT = v3.DEFAULT_OUTPUT_ROOT
DEFAULT_STORE_ROOT = v3.DEFAULT_SOURCE_ROOT
DEFAULT_RETRIEVAL = DEFAULT_STORE_ROOT / "retrieval.json"
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-hot-v3-provider-free-witness-20260906"
)
CONSTRUCTION_NAME = "construction.json"
RUNTIME_NAME = "runtime.json"
SCORE_NAME = "scores.json"
REPLAY_NAME = "replay.json"
REPLAY_FORMAT = "memory-condense-hot-v3-provider-free-witness-replay-v2"

LANE_ORDER = (
    "profile_preference",
    "typed_witness",
    "activated_turn_links",
)

Clock = Callable[[], int]
TypedResultResolver = Callable[[object, str, object], object]


@dataclass(frozen=True, slots=True)
class AssayVariant:
    """Artifact identity for a compatible typed-lane implementation."""

    construction_format: str = FORMAT
    runtime_format: str = RUNTIME_FORMAT
    score_format: str = SCORE_FORMAT
    row_format: str = ROW_FORMAT
    composition_format: str = COMPOSITION_FORMAT
    timing_format: str = TIMING_FORMAT
    replay_format: str = REPLAY_FORMAT
    policy_id: str = POLICY_ID
    construction_status: str = "sealed_gold_blind_hot_v3_provider_free_witness"
    implementation_format: str = "memory-condense-hot-v3-witness-implementation-v2"
    implementation_extra_paths: tuple[str, ...] = ()
    repair_legacy_parent_projection_collisions: bool = False

    def __post_init__(self) -> None:
        values = (
            self.construction_format,
            self.runtime_format,
            self.score_format,
            self.row_format,
            self.composition_format,
            self.timing_format,
            self.replay_format,
            self.policy_id,
            self.construction_status,
            self.implementation_format,
        )
        _require(
            all(type(value) is str and value for value in values)
            and type(self.implementation_extra_paths) is tuple
            and len(self.implementation_extra_paths)
            == len(set(self.implementation_extra_paths))
            and all(
                type(value) is str and value
                for value in self.implementation_extra_paths
            )
            and type(self.repair_legacy_parent_projection_collisions) is bool,
            "assay variant identity changed",
        )


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ValueError(message)


BASE_VARIANT = AssayVariant()


def _elapsed(clock: Clock, started: int) -> int:
    value = clock() - started
    _require(type(value) is int and value >= 0, "runtime clock moved backwards")
    return value


def _normalize_ordinals(ordinals: Sequence[int]) -> tuple[int, ...]:
    result = tuple(sorted(ordinals))
    _require(
        result
        and all(
            type(value) is int and 0 <= value < EXPECTED_QUESTION_COUNT
            for value in result
        )
        and len(result) == len(set(result)),
        "requested ordinals must be unique exact integers in the locked population",
    )
    return result


def _parse_ordinals(value: str) -> tuple[int, ...]:
    text = value.strip()
    if text.casefold() in {"all", "full100"}:
        return tuple(range(EXPECTED_QUESTION_COUNT))
    _require(bool(text), "ordinal selection is empty")
    try:
        parsed = tuple(int(part.strip()) for part in text.split(","))
    except ValueError as exc:
        raise ValueError(
            "ordinals must be 'all' or comma-separated integers"
        ) from exc
    return _normalize_ordinals(parsed)


def _percentile95(values: Sequence[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)])


def _raw_evidence_row(
    *,
    chunk_id: object,
    turn_id: object,
    source_id: object,
    role: object,
    created_at: object,
    raw_text: object,
    route: str,
    excerpt: bool = False,
) -> dict[str, Any]:
    values = {
        "chunk_id": chunk_id,
        "turn_id": turn_id,
        "source_id": source_id,
        "role": role,
        "created_at": created_at,
        "raw_text": raw_text,
    }
    _require(
        all(type(value) is str and value for value in values.values()),
        "successor evidence lost exact raw provenance",
    )
    provenance = " | ".join((str(created_at), str(role)))
    rendered = f"[{provenance}] {raw_text}"
    raw_text_sha256 = quote_sha256(str(raw_text))
    rendered_text_sha256 = quote_sha256(rendered)
    physical_chunk_id = str(chunk_id)
    evidence_id = physical_chunk_id
    if excerpt:
        evidence_id = identity_sha256(
            {
                "backing_chunk_id": physical_chunk_id,
                "created_at": str(created_at),
                "format": EXCERPT_OCCURRENCE_FORMAT,
                "raw_text_sha256": raw_text_sha256,
                "rendered_text_sha256": rendered_text_sha256,
                "role": str(role),
                "source_id": str(source_id),
            }
        )
    row = {
        "evidence_id": evidence_id,
        "chunk_id": evidence_id,
        "turn_id": str(turn_id),
        "source_id": str(source_id),
        "role": str(role),
        "created_at": str(created_at),
        "route": route,
        "score": 1.0,
        "raw_text": str(raw_text),
        "raw_text_sha256": raw_text_sha256,
        "rendered_text": rendered,
        "rendered_text_sha256": rendered_text_sha256,
    }
    if excerpt:
        row["backing_chunk_id"] = physical_chunk_id
        row["excerpt_occurrence"] = True
    return row


def _profile_rows(result: object) -> tuple[dict[str, Any], ...]:
    audit = getattr(result, "audit")
    if getattr(audit, "status") != "selected":
        return ()
    candidates = tuple(getattr(result, "candidates"))
    bindings = tuple(getattr(result, "local_bindings"))
    _require(len(candidates) == len(bindings), "profile candidates lost bindings")
    rows: list[dict[str, Any]] = []
    for candidate, binding in zip(candidates, bindings, strict=True):
        span = getattr(binding, "span")
        rows.append(
            _raw_evidence_row(
                chunk_id=getattr(span, "chunk_id"),
                turn_id=getattr(span, "turn_id") or getattr(span, "chunk_id"),
                source_id=getattr(binding, "source_id"),
                role=getattr(candidate, "role"),
                created_at=getattr(candidate, "created_at"),
                raw_text=getattr(candidate, "quote"),
                route="hot_v3_profile_preference",
                excerpt=True,
            )
        )
    return tuple(rows)


def _typed_witness_rows(result: object) -> tuple[dict[str, Any], ...]:
    # The query is called with no protected IDs, preserving its independent
    # pre-dedup selection until the successor composer owns collision policy.
    witnesses = tuple(getattr(result, "selected_before_dedup"))
    rows: list[dict[str, Any]] = []
    for witness in witnesses:
        span = getattr(witness, "span")
        rows.append(
            _raw_evidence_row(
                chunk_id=getattr(span, "chunk_id"),
                turn_id=getattr(span, "turn_id") or getattr(span, "chunk_id"),
                source_id=getattr(witness, "source_id"),
                role=getattr(span, "role") or "user",
                created_at=(
                    getattr(span, "created_at")
                    or "1970-01-01T00:00:00+00:00"
                ),
                raw_text=getattr(witness, "quote"),
                route="hot_v3_typed_witness",
                excerpt=True,
            )
        )
    return tuple(rows)


def _activated_rows(result: object) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for chunk in tuple(getattr(result, "selected_before_dedup")):
        span = getattr(chunk, "span")
        rows.append(
            _raw_evidence_row(
                chunk_id=getattr(chunk, "chunk_id"),
                turn_id=getattr(span, "turn_id") or getattr(chunk, "chunk_id"),
                source_id=getattr(chunk, "source_id"),
                role=getattr(chunk, "role"),
                created_at=getattr(chunk, "created_at"),
                raw_text=getattr(chunk, "raw_text"),
                route="hot_v3_activated_turn_links",
            )
        )
    return tuple(rows)


def _result_receipt(result: object, *, profile: bool = False) -> str:
    receipt = getattr(result, "receipt_sha256", None) if profile else None
    if receipt is None:
        receipt = getattr(getattr(result, "receipt"), "receipt_sha256")
    _require(type(receipt) is str and len(receipt) == 64, "lane receipt changed")
    return receipt


def _parent_sources(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(row["source_id"]) for row in rows))


def _evidence_occurrence_sha256(row: Mapping[str, Any]) -> str:
    """Bind exact content to fields shared by parent and live-store rows.

    The compact sealed-v3 evidence schema intentionally omits ``turn_id``.
    Physical ``chunk_id``/``backing_chunk_id`` plus exact content and source
    coordinates are the common occurrence authority; including a live-only
    turn field would make the same raw chunk appear novel after hydration.
    """

    raw_text = str(row["raw_text"])
    rendered_text = str(row["rendered_text"])
    _require(
        row.get("evidence_id") == row.get("chunk_id")
        and row.get("raw_text_sha256") == quote_sha256(raw_text)
        and row.get("rendered_text_sha256") == quote_sha256(rendered_text)
        and rendered_text.endswith(raw_text),
        "successor evidence changed its raw occurrence binding",
    )
    return identity_sha256(
        {
            "backing_chunk_id": str(
                row.get("backing_chunk_id", row["chunk_id"])
            ),
            "created_at": str(row["created_at"]),
            "raw_text_sha256": str(row["raw_text_sha256"]),
            "rendered_text_sha256": str(row["rendered_text_sha256"]),
            "role": str(row["role"]),
            "source_id": str(row["source_id"]),
        }
    )


def _repair_legacy_parent_projection_collisions(
    parent: Sequence[Mapping[str, Any]],
    lane_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, str], ...]]:
    """Give legacy assertion excerpts an identity distinct from their raw chunk.

    Historical v2/v3 packets stored an assertion quote under its backing
    physical chunk ID.  A later activated-neighbor lane can therefore hydrate
    the authenticated full raw chunk under the same ID with different bytes.
    Both are valid evidence occurrences; neither may impersonate or erase the
    other.  Repair only this sealed legacy route and only after the independent
    activated lane has selected the colliding full chunk.
    """

    activated_occurrences: dict[str, set[str]] = {}
    for row in lane_rows["activated_turn_links"]:
        if (
            row.get("excerpt_occurrence") is True
            or row.get("route") != "hot_v3_activated_turn_links"
        ):
            continue
        activated_occurrences.setdefault(str(row["chunk_id"]), set()).add(
            _evidence_occurrence_sha256(row)
        )

    repaired: list[dict[str, Any]] = []
    audit: list[dict[str, str]] = []
    for raw in parent:
        row = copy.deepcopy(dict(raw))
        backing_chunk_id = str(row["chunk_id"])
        candidates = activated_occurrences.get(backing_chunk_id)
        parent_occurrence = _evidence_occurrence_sha256(row)
        collision = candidates is not None and parent_occurrence not in candidates
        if not collision:
            repaired.append(row)
            continue

        assertion_receipt = row.get("assertion_fact_receipt_sha256")
        _require(
            row.get("route") == "activated_assertion_projection"
            and type(assertion_receipt) is str
            and len(assertion_receipt) == 64
            and all(character in "0123456789abcdef" for character in assertion_receipt),
            "non-excerpt successor changed bytes for a parent chunk ID",
        )
        occurrence_id = identity_sha256(
            {
                "assertion_fact_receipt_sha256": str(
                    row["assertion_fact_receipt_sha256"]
                ),
                "backing_chunk_id": backing_chunk_id,
                "created_at": str(row["created_at"]),
                "format": LEGACY_PARENT_PROJECTION_OCCURRENCE_FORMAT,
                "raw_text_sha256": str(row["raw_text_sha256"]),
                "rendered_text_sha256": str(row["rendered_text_sha256"]),
                "role": str(row["role"]),
                "source_id": str(row["source_id"]),
            }
        )
        row["backing_chunk_id"] = backing_chunk_id
        row["chunk_id"] = occurrence_id
        row["evidence_id"] = occurrence_id
        row["excerpt_occurrence"] = True
        row["legacy_parent_projection_identity_repaired"] = True
        repaired.append(row)
        audit.append(
            {
                "assertion_fact_receipt_sha256": str(
                    row["assertion_fact_receipt_sha256"]
                ),
                "backing_chunk_id": backing_chunk_id,
                "legacy_parent_occurrence_sha256": parent_occurrence,
                "repaired_occurrence_id": occurrence_id,
            }
        )
    return tuple(repaired), tuple(audit)


def _validate_parent_chunk_claims(
    lane_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    parent: Sequence[Mapping[str, Any]],
) -> None:
    parent_occurrences: dict[str, set[str]] = {}
    parent_rows_by_chunk: dict[str, list[Mapping[str, Any]]] = {}
    for row in parent:
        chunk_id = str(row["chunk_id"])
        parent_occurrences.setdefault(chunk_id, set()).add(
            _evidence_occurrence_sha256(row)
        )
        parent_rows_by_chunk.setdefault(chunk_id, []).append(row)
    for lane, rows in lane_rows.items():
        for row in rows:
            occurrence = _evidence_occurrence_sha256(row)
            if row.get("excerpt_occurrence") is True:
                backing = row.get("backing_chunk_id")
                _require(
                    type(backing) is str
                    and bool(backing)
                    and str(row["chunk_id"]) != backing,
                    "specialist excerpt lost its distinct occurrence ID",
                )
                continue
            chunk_id = str(row["chunk_id"])
            expected = parent_occurrences.get(chunk_id)
            if expected is not None and occurrence not in expected:
                fields = (
                    "created_at",
                    "raw_text_sha256",
                    "rendered_text_sha256",
                    "role",
                    "source_id",
                )
                actual = {field: row.get(field) for field in fields}
                parents = [
                    {field: parent_row.get(field) for field in fields}
                    for parent_row in parent_rows_by_chunk[chunk_id]
                ]
                _require(
                    False,
                    "non-excerpt successor changed bytes for a parent chunk ID: "
                    f"lane={lane}, chunk_id={chunk_id}, "
                    f"successor={actual!r}, parents={parents!r}",
                )


def _dedup_successor_lanes(
    lane_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, str]],
    dict[str, list[str]],
    dict[str, tuple[str, str]],
]:
    _require(tuple(lane_rows) == LANE_ORDER, "successor lane order changed")
    retained: list[dict[str, Any]] = []
    duplicates: list[dict[str, str]] = []
    owner_by_occurrence: dict[str, tuple[str, str]] = {}
    retained_by_lane = {lane: [] for lane in LANE_ORDER}
    for lane in LANE_ORDER:
        for raw in lane_rows[lane]:
            row = copy.deepcopy(dict(raw))
            chunk_id = str(row.get("chunk_id", ""))
            _require(bool(chunk_id), "successor lane emitted an empty chunk ID")
            occurrence = _evidence_occurrence_sha256(row)
            owner = owner_by_occurrence.get(occurrence)
            if owner is not None:
                duplicates.append(
                    {
                        "chunk_id": chunk_id,
                        "evidence_occurrence_sha256": occurrence,
                        "excluded_lane": lane,
                        "retained_lane": owner[0],
                        "retained_chunk_id": owner[1],
                    }
                )
                continue
            owner_by_occurrence[occurrence] = (lane, chunk_id)
            retained_by_lane[lane].append(chunk_id)
            retained.append(row)
    return retained, duplicates, retained_by_lane, owner_by_occurrence


def _pack_ranked_raw_evidence(
    evidence: Sequence[dict[str, Any]],
    *,
    prompt_question: str,
    max_context_tokens: int,
    max_prompt_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pack the exact legacy arm schema with logarithmic prefix probes."""

    _require(
        OUTPUT_TOKEN_RESERVE == hot.RESPONDER_OUTPUT_TOKEN_RESERVE,
        "successor output reserve diverged from the provider arm contract",
    )
    candidates = tuple(evidence)
    pack: RankedPrefixPromptPack[
        dict[str, Any], list[dict[str, str]]
    ] = pack_ranked_prefix_prompt(
        candidates,
        count_context_tokens=lambda rows: hot._context_token_proxy(  # noqa: SLF001
            [str(row["rendered_text"]) for row in rows]
        ),
        render_prompt=lambda rows: hot.build_qa_prompt(
            prompt_question,
            [str(row["rendered_text"]) for row in rows],
        ),
        count_prompt_tokens=hot.count_chat_prompt_token_proxy,
        max_context_tokens=max_context_tokens,
        max_prompt_tokens=max_prompt_tokens,
        output_token_reserve=OUTPUT_TOKEN_RESERVE,
    )
    serialized = hot._canonical_json_bytes(  # noqa: SLF001
        {"messages": pack.rendered_prompt}
    )
    envelope = hot._ProviderPromptEnvelope(  # noqa: SLF001
        packed_count=pack.packed_count,
        context_token_proxy=pack.context_token_count,
        prompt_token_proxy=pack.prompt_token_count,
        provider_messages=pack.rendered_prompt,
        serialized=serialized,
        pack_and_prompt_render_count_ns=0,
        serialize_ns=0,
        provider_ready_at_ns=0,
    )
    semantic = hot._raw_packet_semantic(candidates, envelope)  # noqa: SLF001
    return semantic, pack.audit.projection()


def compose_successor_arm(
    *,
    parent_arm: Mapping[str, Any],
    dated_question: str,
    lane_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    lane_receipts: Mapping[str, str],
    clock: Clock = time.perf_counter_ns,
    timing_sink: dict[str, int] | None = None,
    variant: AssayVariant = BASE_VARIANT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compose selected local lanes, preserving the exact parent on failure."""

    compose_started = clock()
    _require(tuple(lane_receipts) == LANE_ORDER, "lane receipt order changed")
    _require(tuple(lane_rows) == LANE_ORDER, "successor lane order changed")
    hot._validate_arm_payload(  # noqa: SLF001
        parent_arm,
        prompt_question=dated_question,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
    )
    sealed_parent = tuple(copy.deepcopy(list(parent_arm["packed_evidence"])))
    if variant.repair_legacy_parent_projection_collisions:
        parent, legacy_projection_repairs = (
            _repair_legacy_parent_projection_collisions(
                sealed_parent,
                lane_rows,
            )
        )
    else:
        parent = sealed_parent
        legacy_projection_repairs = ()
    _require(parent, "sealed parent packet became empty")
    required_sources = _parent_sources(parent)
    required_representatives = {
        source_id: next(
            row for row in parent if str(row["source_id"]) == source_id
        )
        for source_id in required_sources
    }
    _validate_parent_chunk_claims(lane_rows, parent)
    (
        successor,
        duplicates,
        retained_by_lane,
        owner_by_occurrence,
    ) = _dedup_successor_lanes(lane_rows)

    # Specialist excerpts have occurrence IDs distinct from their backing raw
    # chunks. Only exact occurrence/content equality may displace a parent row.
    parent_unique: list[dict[str, Any]] = []
    for raw in parent:
        row = copy.deepcopy(dict(raw))
        chunk_id = str(row["chunk_id"])
        occurrence = _evidence_occurrence_sha256(row)
        owner = owner_by_occurrence.get(occurrence)
        if owner is not None:
            duplicates.append(
                {
                    "chunk_id": chunk_id,
                    "evidence_occurrence_sha256": occurrence,
                    "excluded_lane": "sealed_v3_parent",
                    "retained_lane": owner[0],
                    "retained_chunk_id": owner[1],
                }
            )
            continue
        owner_by_occurrence[occurrence] = ("sealed_v3_parent", chunk_id)
        parent_unique.append(row)

    protected_parent: list[dict[str, Any]] = []
    protected_ids: list[str] = []
    protected_occurrences: list[str] = []
    for source_id in required_sources:
        occurrence = _evidence_occurrence_sha256(
            required_representatives[source_id]
        )
        owner = owner_by_occurrence.get(occurrence)
        _require(
            owner is not None,
            "parent source lost its authoritative representative during dedup",
        )
        protected_ids.append(owner[1])
        protected_occurrences.append(occurrence)
        if owner[0] == "sealed_v3_parent":
            protected_parent.append(
                next(
                    row
                    for row in parent_unique
                    if _evidence_occurrence_sha256(row) == occurrence
                )
            )

    protected_set = {
        _evidence_occurrence_sha256(row) for row in protected_parent
    }
    parent_remainder = [
        row
        for row in parent_unique
        if _evidence_occurrence_sha256(row) not in protected_set
    ]
    ranked = [*successor, *protected_parent, *parent_remainder]
    attempted, packing_audit = _pack_ranked_raw_evidence(
        ranked,
        prompt_question=dated_question,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
    )
    packed_sources = {
        str(row["source_id"]) for row in attempted["packed_evidence"]
    }
    missing_sources = tuple(
        source_id
        for source_id in required_sources
        if source_id not in packed_sources
    )
    packed_occurrences = {
        _evidence_occurrence_sha256(row)
        for row in attempted["packed_evidence"]
    }
    missing_representative_sources = tuple(
        source_id
        for source_id, occurrence in zip(
            required_sources, protected_occurrences, strict=True
        )
        if occurrence not in packed_occurrences
    )
    source_conserved = not missing_sources and not missing_representative_sources
    effective = (
        attempted if source_conserved else copy.deepcopy(dict(parent_arm))
    )
    mode = "successor" if source_conserved else "exact_parent_fallback"
    if mode == "exact_parent_fallback":
        _require(
            effective == dict(parent_arm),
            "source-conservation fallback changed the sealed parent",
        )
    provider_ready_at = clock()
    hot._validate_arm_payload(  # noqa: SLF001
        effective,
        prompt_question=dated_question,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
    )
    body = {
        "candidate_chunk_ids": [str(row["chunk_id"]) for row in ranked],
        "context_token_cap": MAX_CONTEXT_TOKENS,
        "dedup_exclusions": duplicates,
        "effective_provider_payload_sha256": effective[
            "provider_payload_sha256"
        ],
        "format": variant.composition_format,
        "lane_receipt_sha256s": dict(lane_receipts),
        "lane_retained_after_dedup_ids": retained_by_lane,
        "lane_selected_before_dedup_ids": {
            lane: [str(row["chunk_id"]) for row in lane_rows[lane]]
            for lane in LANE_ORDER
        },
        "legacy_parent_projection_collision_repair_enabled": (
            variant.repair_legacy_parent_projection_collisions
        ),
        "legacy_parent_projection_identity_repairs": list(
            legacy_projection_repairs
        ),
        "missing_parent_source_id_sha256s": [
            quote_sha256(value) for value in missing_sources
        ],
        "missing_parent_representative_source_id_sha256s": [
            quote_sha256(value) for value in missing_representative_sources
        ],
        "mode": mode,
        "new_lane_owns_exact_parent_occurrence_duplicates": True,
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "packing_audit": packing_audit,
        "packing_audit_sha256": identity_sha256(packing_audit),
        "packed_chunk_ids": list(effective["packed_chunk_ids"]),
        "parent_provider_payload_sha256": parent_arm[
            "provider_payload_sha256"
        ],
        "parent_source_count": len(required_sources),
        "parent_source_id_sha256s": [
            quote_sha256(value) for value in required_sources
        ],
        "post_selection_exact_evidence_occurrence_dedup": True,
        "prompt_workspace_token_cap": MAX_PROMPT_TOKENS,
        "protected_parent_representative_chunk_ids": protected_ids,
        "protected_parent_representative_occurrence_sha256s": (
            protected_occurrences
        ),
        "parent_representative_conservation_passed": not (
            missing_representative_sources
        ),
        "source_conservation_passed": source_conserved,
    }
    assert_gold_blind(body, path="hot_v3_witness_composition")
    audit = {**body, "receipt_sha256": identity_sha256(body)}
    finished_at = clock()
    if timing_sink is not None:
        prompt_ready_ns = provider_ready_at - compose_started
        post_ready_ns = finished_at - provider_ready_at
        _require(
            prompt_ready_ns >= 0 and post_ready_ns >= 0,
            "runtime clock moved backwards inside composition",
        )
        timing_sink.update(
            {
                "post_ready_validation_audit_ns": post_ready_ns,
                "prompt_ready_ns": prompt_ready_ns,
                "total_ns": finished_at - compose_started,
            }
        )
    return effective, audit


@dataclass(frozen=True, slots=True)
class RuntimeHooks:
    """Injectable lifecycle boundary used by focused provider-free tests."""

    load_parent: Callable[..., tuple[dict[int, dict[str, Any]], str]]
    load_context: Callable[..., object]
    build_full_index: Callable[..., tuple[object, Mapping[str, Any]]]
    profile_applicable: Callable[[str], bool]
    select_profile: Callable[..., object]
    build_typed_index: Callable[[object], object]
    query_typed: Callable[..., object]
    build_link_index: Callable[[object], object]
    query_links: Callable[..., object]
    # Optional alternative implementation of the existing typed lane.  The
    # resolver receives the resident full-store index, dated question, and
    # already-sealed unrestricted result.  Returning that same result leaves
    # v2 byte-semantic; returning a compatible result replaces rather than
    # appends typed evidence.
    resolve_typed: TypedResultResolver | None = None


def _default_hooks() -> RuntimeHooks:
    return RuntimeHooks(
        load_parent=_load_sealed_v3_rows,
        load_context=load_locked_query_expansion_context,
        build_full_index=_build_resident_index,
        profile_applicable=_profile_applicable,
        select_profile=select_profile_preference_evidence,
        build_typed_index=build_hot_typed_witness_index,
        query_typed=query_hot_typed_witnesses,
        build_link_index=build_hot_v3_activated_turn_link_index,
        query_links=select_hot_v3_activated_turn_links,
    )


@dataclass(frozen=True, slots=True)
class _SkippedProfileAudit:
    status: str = "not_applicable"


@dataclass(frozen=True, slots=True)
class _SkippedProfileResult:
    audit: _SkippedProfileAudit
    candidates: tuple[()] = ()
    local_bindings: tuple[()] = ()
    receipt_sha256: str = ""


def _profile_applicable(dated_question: str) -> bool:
    """Mirror the specialist's question-only entrance gate without a scan."""

    _asked_at, body = profile_lane._question_parts(dated_question)  # noqa: SLF001
    terms = frozenset(profile_lane.indexed_surface_terms(body))
    return bool(
        profile_lane._RECOMMENDATION_RE.search(body)  # noqa: SLF001
        and profile_lane._recognized_domains(terms)  # noqa: SLF001
    )


def _skipped_profile(dated_question: str) -> _SkippedProfileResult:
    body = {
        "mechanism_id": profile_lane.MECHANISM_ID,
        "question_sha256": quote_sha256(dated_question),
        "status": "not_applicable",
    }
    return _SkippedProfileResult(
        audit=_SkippedProfileAudit(),
        receipt_sha256=identity_sha256(body),
    )


def _implementation_identity(
    variant: AssayVariant = BASE_VARIANT,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    base_paths = (
        "src/memory_condense/domain/_tokenizer.py",
        "src/memory_condense/domain/discourse.py",
        "src/memory_condense/domain/integrity.py",
        "src/memory_condense/eval/_retrieval_qa_prompt.py",
        "src/memory_condense/search/packing/ranked_prefix_prompt.py",
        "src/memory_condense/search/source_neighborhood.py",
        "tools/assay_hot_retrieval_1m.py",
        "tools/assay_hot_retrieval_full100.py",
        "tools/assay_hot_retrieval_source_seed_hybrid_full100.py",
        "tools/assay_hot_v3_provider_free_witness.py",
        "tools/assay_hot_v3_full_store_numeric.py",
        "tools/assay_hot_v3_typed_operator_full100.py",
        "tools/matched_eval/contracts.py",
        "tools/matched_eval/profile_preference_specialist.py",
        "tools/matched_eval/hot_typed_witness.py",
        "tools/matched_eval/hot_v3_activated_turn_links.py",
        "tools/matched_eval/population.py",
        "tools/matched_eval/query_expansion.py",
    )
    paths = (*base_paths, *variant.implementation_extra_paths)
    _require(len(paths) == len(set(paths)), "implementation path repeated")
    files = {path: file_sha256(root / path) for path in paths}
    return {
        "format": variant.implementation_format,
        "files": files,
        "sha256": identity_sha256(
            [{"path": path, "sha256": files[path]} for path in paths]
        ),
    }


def _lane_status(result: object, *, profile: bool = False) -> str:
    if profile:
        return str(getattr(getattr(result, "audit"), "status"))
    return str(getattr(result, "status", "selected"))


def _question_row(
    *,
    source: Mapping[str, Any],
    dated_question: str,
    effective_arm: Mapping[str, Any],
    composition: Mapping[str, Any],
    lane_statuses: Mapping[str, str],
    variant: AssayVariant = BASE_VARIANT,
) -> dict[str, Any]:
    compact_parent = {
        key: copy.deepcopy(value)
        for key, value in source.items()
        if key != "arms"
    }
    provider_packet = compact_parent.get("provider_packet")
    _require(
        isinstance(provider_packet, Mapping)
        and type(provider_packet.get("receipt_sha256")) is str
        and len(provider_packet["receipt_sha256"]) == 64,
        "authenticated parent provider-packet receipt changed",
    )
    body = {
        "composition": copy.deepcopy(dict(composition)),
        "effective_arm": copy.deepcopy(dict(effective_arm)),
        "format": variant.row_format,
        "gold_loaded": False,
        "model_calls": 0,
        "new_provider_calls": 0,
        "ordinal": int(source["ordinal"]),
        "parent_compact_row_sha256": identity_sha256(compact_parent),
        "parent_provider_packet_receipt_sha256": provider_packet[
            "receipt_sha256"
        ],
        "policy_id": variant.policy_id,
        "prompt_question_sha256": quote_sha256(dated_question),
        "question_id": str(source["question_id"]),
        "retained_transformer_token_state_bytes": 0,
        "statuses": dict(lane_statuses),
    }
    assert_gold_blind(body, path="hot_v3_witness_row")
    return {**body, "row_receipt_sha256": identity_sha256(body)}


def _physical_ids(index: object) -> frozenset[str]:
    return frozenset(
        str(row.chunk_id) for row in tuple(getattr(index, "rows"))
    )


def _profile_chunk_ids(result: object) -> tuple[str, ...]:
    if getattr(getattr(result, "audit"), "status") != "selected":
        return ()
    return tuple(
        str(getattr(getattr(binding, "span"), "chunk_id"))
        for binding in tuple(getattr(result, "local_bindings"))
    )


def _typed_chunk_ids(result: object) -> tuple[str, ...]:
    return tuple(
        str(getattr(getattr(row, "span"), "chunk_id"))
        for row in tuple(getattr(result, "selected_before_dedup"))
    )


def _select_link_seeds(
    *,
    profile_result: object,
    typed_result: object,
    parent_arm: Mapping[str, Any],
    physical_ids: frozenset[str],
) -> tuple[str, ...]:
    specialist = tuple(
        dict.fromkeys(
            (
                *_profile_chunk_ids(profile_result),
                *_typed_chunk_ids(typed_result),
            )
        )
    )
    physical_specialist = tuple(
        value for value in specialist if value in physical_ids
    )
    if physical_specialist:
        return physical_specialist
    physical_parent = tuple(
        value
        for value in parent_arm["packed_chunk_ids"]
        if value in physical_ids
    )
    return physical_parent[:PARENT_FALLBACK_SEED_LIMIT]


def _assay_one(
    *,
    source: Mapping[str, Any],
    index: object,
    typed_index: object,
    link_index: object,
    hooks: RuntimeHooks,
    clock: Clock,
    variant: AssayVariant = BASE_VARIANT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent_arm = source["arms"]["a3_protected_union"]
    dated_question = typed._extract_dated_question(parent_arm)  # noqa: SLF001
    started = clock()

    step = clock()
    profile_applicable = hooks.profile_applicable(dated_question)
    profile_applicability_ns = _elapsed(clock, step)
    profile_ns = 0
    if profile_applicable:
        step = clock()
        profile = hooks.select_profile(index, dated_question)
        profile_ns = _elapsed(clock, step)
    else:
        profile = _skipped_profile(dated_question)

    step = clock()
    baseline_witness = hooks.query_typed(
        typed_index,
        dated_question,
        protected_chunk_ids=(),
    )
    witness = (
        baseline_witness
        if hooks.resolve_typed is None
        else hooks.resolve_typed(index, dated_question, baseline_witness)
    )
    typed_ns = _elapsed(clock, step)

    physical = _physical_ids(index)
    seed_ids = _select_link_seeds(
        profile_result=profile,
        typed_result=witness,
        parent_arm=parent_arm,
        physical_ids=physical,
    )
    # The link index accepts opaque parent IDs, but seeds must be physical.
    # Passing the complete parent sequence lets its receipt prove that an
    # activated duplicate owns the collision without materializing metadata.
    parent_ids = tuple(parent_arm["packed_chunk_ids"])
    step = clock()
    links = hooks.query_links(
        link_index,
        seed_ids,
        parent_chunk_ids=parent_ids,
    )
    link_ns = _elapsed(clock, step)

    lane_rows = {
        "profile_preference": _profile_rows(profile),
        "typed_witness": _typed_witness_rows(witness),
        "activated_turn_links": _activated_rows(links),
    }
    lane_receipts = {
        "profile_preference": _result_receipt(profile, profile=True),
        "typed_witness": _result_receipt(witness),
        "activated_turn_links": _result_receipt(links),
    }
    compose_timing: dict[str, int] = {}
    effective, composition = compose_successor_arm(
        parent_arm=parent_arm,
        dated_question=dated_question,
        lane_rows=lane_rows,
        lane_receipts=lane_receipts,
        clock=clock,
        timing_sink=compose_timing,
        variant=variant,
    )
    row = _question_row(
        source=source,
        dated_question=dated_question,
        effective_arm=effective,
        composition=composition,
        lane_statuses={
            "profile_preference": _lane_status(profile, profile=True),
            "typed_witness": _lane_status(witness),
            "activated_turn_links": _lane_status(links),
        },
        variant=variant,
    )
    timing = {
        "activated_turn_links_ns": link_ns,
        "compose_ns": compose_timing["total_ns"],
        "compose_post_ready_validation_audit_ns": compose_timing[
            "post_ready_validation_audit_ns"
        ],
        "compose_prompt_ready_ns": compose_timing["prompt_ready_ns"],
        "format": variant.timing_format,
        "ordinal": int(source["ordinal"]),
        "profile_applicability_ns": profile_applicability_ns,
        "profile_preference_ns": profile_ns,
        "question_id": str(source["question_id"]),
        "result_row_receipt_sha256": row["row_receipt_sha256"],
        "total_ns": _elapsed(clock, started),
        "typed_witness_ns": typed_ns,
    }
    return row, timing


def _semantic_aggregate(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "activated_successor_count": sum(
            row["composition"]["mode"] == "successor" for row in rows
        ),
        "exact_parent_fallback_count": sum(
            row["composition"]["mode"] == "exact_parent_fallback"
            for row in rows
        ),
        "profile_selected_count": sum(
            row["statuses"]["profile_preference"] == "selected"
            for row in rows
        ),
        "question_count": len(rows),
        "typed_witness_applicable_count": sum(
            row["statuses"]["typed_witness"] != "not_applicable"
            for row in rows
        ),
    }


def _runtime_aggregate(
    rows: Sequence[Mapping[str, Any]],
    timings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    warm = [int(row["total_ns"]) for row in timings]
    prompt_ready = [int(row["compose_prompt_ready_ns"]) for row in timings]
    post_ready = [
        int(row["compose_post_ready_validation_audit_ns"])
        for row in timings
    ]
    return {
        **_semantic_aggregate(rows),
        "compose_post_ready_validation_audit_mean_ns": (
            statistics.fmean(post_ready) if post_ready else None
        ),
        "compose_post_ready_validation_audit_p95_ns": _percentile95(
            post_ready
        ),
        "compose_prompt_ready_mean_ns": (
            statistics.fmean(prompt_ready) if prompt_ready else None
        ),
        "compose_prompt_ready_p95_ns": _percentile95(prompt_ready),
        "warm_mean_ns": statistics.fmean(warm) if warm else None,
        "warm_p95_ns": _percentile95(warm),
    }


def construct(
    *,
    v3_root: Path,
    retrieval_path: Path,
    store_root: Path,
    output_root: Path,
    ordinals: Sequence[int] = tuple(range(EXPECTED_QUESTION_COUNT)),
    hooks: RuntimeHooks | None = None,
    clock: Clock = time.perf_counter_ns,
    variant: AssayVariant = BASE_VARIANT,
) -> tuple[str, str]:
    """Build a sealed gold-blind successor selection and separate timings."""

    wanted = _normalize_ordinals(ordinals)
    for name in (CONSTRUCTION_NAME, RUNTIME_NAME, SCORE_NAME):
        _require(
            not (output_root / name).exists(),
            f"refusing to overwrite {name}",
        )
    runtime = hooks or _default_hooks()
    total_started = clock()

    step = clock()
    source_by_ordinal, selection_sha = runtime.load_parent(
        v3_root, ordinals=wanted
    )
    parent_load_ns = _elapsed(clock, step)
    _require(
        tuple(sorted(source_by_ordinal)) == wanted,
        "parent ordinal set changed",
    )

    step = clock()
    context = runtime.load_context(
        retrieval_path,
        store_root=store_root,
        expected_retrieval_sha256=EXPECTED_RETRIEVAL_SHA256,
    )
    context_load_ns = _elapsed(clock, step)
    population_by_question = {
        row.source.packet.question_id: row for row in context.population.rows
    }
    by_namespace: dict[str, list[int]] = {}
    namespace_objects: dict[str, object] = {}
    for ordinal in wanted:
        source = source_by_ordinal[ordinal]
        question_id = str(source["question_id"])
        population_row = population_by_question.get(question_id)
        _require(
            population_row is not None,
            "parent question absent from store context",
        )
        namespace = population_row.namespace
        namespace_id = str(namespace.namespace_id)
        by_namespace.setdefault(namespace_id, []).append(ordinal)
        namespace_objects[namespace_id] = namespace

    results: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    lifecycle: list[dict[str, Any]] = []
    for namespace_id in sorted(by_namespace):
        namespace = namespace_objects[namespace_id]
        full_started = clock()
        index, base_timing = runtime.build_full_index(context, namespace)
        full_ns = _elapsed(clock, full_started)
        step = clock()
        typed_index = runtime.build_typed_index(index)
        typed_build_ns = _elapsed(clock, step)
        step = clock()
        link_index = runtime.build_link_index(index)
        link_build_ns = _elapsed(clock, step)
        lifecycle.append(
            {
                **dict(base_timing),
                "full_index_setup_ns": full_ns,
                "link_index_build_ns": link_build_ns,
                "namespace_id": namespace_id,
                "question_count": len(by_namespace[namespace_id]),
                "typed_index_build_ns": typed_build_ns,
            }
        )
        try:
            for ordinal in by_namespace[namespace_id]:
                try:
                    results[ordinal] = _assay_one(
                        source=source_by_ordinal[ordinal],
                        index=index,
                        typed_index=typed_index,
                        link_index=link_index,
                        hooks=runtime,
                        clock=clock,
                        variant=variant,
                    )
                except BaseException as exc:
                    exc.add_note(
                        "hot-v3 assay failed for "
                        f"ordinal={ordinal}, namespace_id={namespace_id}"
                    )
                    raise
        finally:
            del link_index, typed_index, index
            gc.collect()

    rows = [results[ordinal][0] for ordinal in wanted]
    timings = [results[ordinal][1] for ordinal in wanted]
    construction_body = {
        "aggregate": _semantic_aggregate(rows),
        "format": variant.construction_format,
        "gold_loaded": False,
        "implementation": _implementation_identity(variant),
        "model_calls": 0,
        "new_provider_calls": 0,
        "ordinals": list(wanted),
        "policy_id": variant.policy_id,
        "question_count": len(rows),
        "questions": rows,
        "retained_transformer_token_state_bytes": 0,
        "status": variant.construction_status,
        "v3_selection_sha256": selection_sha,
    }
    assert_gold_blind(
        construction_body, path="hot_v3_provider_free_witness"
    )
    construction_sha = hot._atomic_write_json(  # noqa: SLF001
        output_root / CONSTRUCTION_NAME, construction_body
    )
    runtime_body = {
        "cold_setup": {
            "namespace_index_timings": lifecycle,
            "sealed_parent_load_ns": parent_load_ns,
            "sealed_store_context_load_ns": context_load_ns,
        },
        "construction_sha256": construction_sha,
        "format": variant.runtime_format,
        "gold_loaded": False,
        "model_calls": 0,
        "new_provider_calls": 0,
        "question_timings": timings,
        "resident_namespace_count": len(lifecycle),
        "retained_transformer_token_state_bytes": 0,
        "total_ns": _elapsed(clock, total_started),
        "warm_aggregate": _runtime_aggregate(rows, timings),
    }
    assert_gold_blind(
        runtime_body, path="hot_v3_provider_free_witness_runtime"
    )
    runtime_sha = hot._atomic_write_json(  # noqa: SLF001
        output_root / RUNTIME_NAME, runtime_body
    )
    print(
        f"Hot-v3 witness: {len(rows)} rows; successor="
        f"{construction_body['aggregate']['activated_successor_count']}; "
        f"fallback="
        f"{construction_body['aggregate']['exact_parent_fallback_count']}; "
        f"construction={construction_sha}; runtime={runtime_sha}",
        flush=True,
    )
    return construction_sha, runtime_sha


def _load_constructed(
    output_root: Path,
    variant: AssayVariant = BASE_VARIANT,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    construction, construction_sha = hot._read_json_artifact(  # noqa: SLF001
        output_root / CONSTRUCTION_NAME
    )
    runtime, runtime_sha = hot._read_json_artifact(  # noqa: SLF001
        output_root / RUNTIME_NAME
    )
    _require(
        construction.get("format") == variant.construction_format,
        "construction format changed",
    )
    _require(
        runtime.get("format") == variant.runtime_format,
        "runtime format changed",
    )
    _require(
        construction.get("gold_loaded") is False
        and construction.get("new_provider_calls") == 0
        and construction.get("model_calls") == 0
        and runtime.get("gold_loaded") is False
        and runtime.get("new_provider_calls") == 0
        and runtime.get("model_calls") == 0
        and runtime.get("construction_sha256") == construction_sha,
        "construction/runtime firebreak changed",
    )
    aggregate = construction.get("aggregate")
    _require(
        isinstance(aggregate, Mapping)
        and not any(str(key).endswith("_ns") for key in aggregate),
        "canonical construction contains runtime timing data",
    )
    assert_gold_blind(construction, path="loaded_hot_v3_witness")
    return construction, construction_sha, runtime, runtime_sha


def replay(
    *,
    v3_root: Path,
    output_root: Path,
    parent_loader: Callable[
        ..., tuple[dict[int, dict[str, Any]], str]
    ] = _load_sealed_v3_rows,
    variant: AssayVariant = BASE_VARIANT,
) -> str:
    """Reconstruct sealed effective provider bytes without gold or calls."""

    _require(
        not (output_root / REPLAY_NAME).exists(),
        "refusing to overwrite replay",
    )
    construction, construction_sha, _runtime, _runtime_sha = (
        _load_constructed(output_root, variant)
    )
    _require(
        construction.get("implementation") == _implementation_identity(variant),
        "replay implementation differs from the sealed construction",
    )
    wanted = _normalize_ordinals(tuple(construction["ordinals"]))
    parent_rows, parent_sha = parent_loader(v3_root, ordinals=wanted)
    _require(
        parent_sha == construction["v3_selection_sha256"],
        "replay parent selection changed",
    )
    constructed_by_ordinal = {
        int(row["ordinal"]): row for row in construction["questions"]
    }
    replayed: list[dict[str, Any]] = []
    for ordinal in wanted:
        selected = constructed_by_ordinal[ordinal]
        sealed = dict(selected)
        row_receipt = sealed.pop("row_receipt_sha256")
        _require(
            row_receipt == identity_sha256(sealed),
            "replay row receipt changed",
        )
        parent = parent_rows[ordinal]
        compact_parent = {
            key: copy.deepcopy(value)
            for key, value in parent.items()
            if key != "arms"
        }
        _require(
            identity_sha256(compact_parent)
            == selected["parent_compact_row_sha256"],
            "replay parent row identity changed",
        )
        parent_arm = parent["arms"]["a3_protected_union"]
        dated_question = typed._extract_dated_question(  # noqa: SLF001
            parent_arm
        )
        _require(
            quote_sha256(dated_question)
            == selected["prompt_question_sha256"],
            "replay prompt question changed",
        )
        effective = selected["effective_arm"]
        reconstructed, packing_audit = _pack_ranked_raw_evidence(
            tuple(copy.deepcopy(effective["selected_evidence"])),
            prompt_question=dated_question,
            max_context_tokens=MAX_CONTEXT_TOKENS,
            max_prompt_tokens=MAX_PROMPT_TOKENS,
        )
        _require(
            reconstructed == effective,
            "replay failed to reconstruct exact effective provider bytes",
        )
        replayed.append(
            {
                "arm_identity_sha256": identity_sha256(reconstructed),
                "ordinal": ordinal,
                "packing_audit": packing_audit,
                "provider_payload_sha256": reconstructed[
                    "provider_payload_sha256"
                ],
                "question_id": selected["question_id"],
                "row_receipt_sha256": row_receipt,
            }
        )
    body = {
        "construction_sha256": construction_sha,
        "format": variant.replay_format,
        "gold_loaded": False,
        "model_calls": 0,
        "new_provider_calls": 0,
        "question_count": len(replayed),
        "questions": replayed,
        "status": "exact_semantic_bytes_reconstructed",
        "v3_selection_sha256": parent_sha,
    }
    assert_gold_blind(body, path="hot_v3_provider_free_witness_replay")
    digest = hot._atomic_write_json(  # noqa: SLF001
        output_root / REPLAY_NAME, body
    )
    print(
        f"Hot-v3 witness replay: {len(replayed)} exact arms; replay={digest}",
        flush=True,
    )
    return digest


def score(
    *,
    dataset: Path,
    split_manifest: Path,
    v3_root: Path,
    output_root: Path,
    population_loader: Callable[
        ..., tuple[object, object, Mapping[str, Any]]
    ] = full100._load_population,  # noqa: SLF001
    question_flattener: Callable[[object], Sequence[object]] = (
        full100._flatten_questions  # noqa: SLF001
    ),
    parent_loader: Callable[
        ..., tuple[dict[int, dict[str, Any]], str]
    ] = _load_sealed_v3_rows,
    variant: AssayVariant = BASE_VARIANT,
) -> str:
    """Open gold only here and compare successor vs authenticated parent."""

    _require(
        not (output_root / SCORE_NAME).exists(),
        "refusing to overwrite score",
    )
    construction, construction_sha, _runtime, runtime_sha = (
        _load_constructed(output_root, variant)
    )
    wanted = _normalize_ordinals(tuple(construction["ordinals"]))
    samples, _identities, population = population_loader(
        dataset, split_manifest
    )
    _require(
        population.get("population_identity_sha256")
        == typed.EXPECTED_POPULATION_SHA256,
        "score population identity changed",
    )
    questions = tuple(question_flattener(samples))
    _require(
        len(questions) == EXPECTED_QUESTION_COUNT,
        "score population changed",
    )
    parent_rows, parent_sha = parent_loader(v3_root, ordinals=wanted)
    _require(
        parent_sha == construction["v3_selection_sha256"],
        "post-hoc parent selection changed",
    )
    constructed_by_ordinal = {
        int(row["ordinal"]): row for row in construction["questions"]
    }
    _require(
        tuple(sorted(constructed_by_ordinal)) == wanted,
        "constructed score ordinal set changed",
    )
    rows: list[dict[str, Any]] = []
    for ordinal in wanted:
        selected = constructed_by_ordinal[ordinal]
        benchmark = questions[ordinal]
        _require(
            selected["question_id"] == benchmark.question_id
            and selected["prompt_question_sha256"]
            == quote_sha256(benchmark.dated_question),
            "post-hoc benchmark identity changed",
        )
        sealed = dict(selected)
        row_receipt = sealed.pop("row_receipt_sha256")
        _require(
            row_receipt == identity_sha256(sealed),
            "constructed row receipt changed",
        )
        parent_arm = parent_rows[ordinal]["arms"]["a3_protected_union"]
        successor_score = v3._score_arm(  # noqa: SLF001
            selected["effective_arm"], benchmark
        )
        parent_score = v3._score_arm(parent_arm, benchmark)  # noqa: SLF001
        rows.append(
            {
                "effective": successor_score,
                "ordinal": ordinal,
                "parent": parent_score,
                "question_id": benchmark.question_id,
                "row_receipt_sha256": row_receipt,
            }
        )
    body = {
        "aggregate": {
            "effective_all_gold_source_reach": sum(
                row["effective"]["all_gold_source_ids_reached"] is True
                for row in rows
            ),
            "effective_literal_answer_hits": sum(
                row["effective"]["literal_answer"] is True for row in rows
            ),
            "parent_all_gold_source_reach": sum(
                row["parent"]["all_gold_source_ids_reached"] is True
                for row in rows
            ),
            "parent_literal_answer_hits": sum(
                row["parent"]["literal_answer"] is True for row in rows
            ),
            "question_count": len(rows),
        },
        "construction_sha256": construction_sha,
        "format": variant.score_format,
        "gold_loaded": True,
        "new_provider_calls": 0,
        "population_identity_sha256": population[
            "population_identity_sha256"
        ],
        "questions": rows,
        "runtime_sha256": runtime_sha,
        "status": "post_construction_gold_join",
        "v3_selection_sha256": parent_sha,
    }
    digest = hot._atomic_write_json(  # noqa: SLF001
        output_root / SCORE_NAME, body
    )
    print(
        "Hot-v3 witness post-hoc score: literal="
        f"{body['aggregate']['effective_literal_answer_hits']}/{len(rows)}; "
        f"parent={body['aggregate']['parent_literal_answer_hits']}; "
        f"score={digest}",
        flush=True,
    )
    return digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-root", type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    construct_parser = commands.add_parser("construct")
    construct_parser.add_argument(
        "--ordinals",
        default="all",
        help="'all'/'full100' or comma-separated locked ordinals",
    )
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--dataset", type=Path, required=True)
    score_parser.add_argument("--split-manifest", type=Path, required=True)
    commands.add_parser("replay")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = args.output_root.resolve()
    if args.command == "construct":
        construct(
            v3_root=args.v3_root.resolve(),
            retrieval_path=args.retrieval.resolve(),
            store_root=args.store_root.resolve(),
            output_root=output_root,
            ordinals=_parse_ordinals(args.ordinals),
        )
    elif args.command == "score":
        score(
            dataset=args.dataset.resolve(),
            split_manifest=args.split_manifest.resolve(),
            v3_root=args.v3_root.resolve(),
            output_root=output_root,
        )
    elif args.command == "replay":
        replay(
            v3_root=args.v3_root.resolve(),
            output_root=output_root,
        )
    else:  # pragma: no cover
        raise AssertionError(f"unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPOSITION_FORMAT",
    "CONSTRUCTION_NAME",
    "FORMAT",
    "LANE_ORDER",
    "RuntimeHooks",
    "REPLAY_NAME",
    "SCORE_NAME",
    "compose_successor_arm",
    "construct",
    "replay",
    "score",
]
