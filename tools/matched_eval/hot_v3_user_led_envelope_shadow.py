"""Provider-free user-led envelope selection over a frozen namespace cache.

This adapter exists for historical, immutable evaluation stores which predate
the durable conversation-envelope journal.  It applies the same deterministic
user-opener boundary rule to authenticated ``CachedContentRow`` objects.  A
question is never accepted: only already-packed, byte-bound parent evidence
may become an anchor.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import EvidenceSpan, quote_sha256
from memory_condense.search.episodes.user_led import UserLedEpisodeBuilder

from .contracts import MatchedEvalContractError, assert_gold_blind, identity_sha256
from .query_guided_scan import CachedContentRow, NamespacePartitionCache


MECHANISM_ID = "hot-v4-user-led-envelope-shadow-v1"
INDEX_FORMAT = "memory-condense-hot-v4-user-led-envelope-shadow-index-v1"
SELECTION_FORMAT = "memory-condense-hot-v4-user-led-envelope-shadow-selection-v1"
COMPOSITION_FORMAT = "memory-condense-hot-v4-user-led-envelope-shadow-composition-v1"
COMPANION_ROUTE = "hot_v4_user_led_envelope_companion"


class UserLedEnvelopeShadowError(MatchedEvalContractError):
    """A frozen-cache, parent, receipt, or atomic-packing invariant changed."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise UserLedEnvelopeShadowError(message)


def _sha(value: Mapping[str, Any]) -> str:
    return identity_sha256(dict(value))


def _physical_id(row: Mapping[str, Any]) -> str:
    value = row.get("backing_chunk_id", row.get("chunk_id"))
    _require(type(value) is str and bool(value), "parent physical chunk ID changed")
    return str(value)


def _parent_footprint(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for position, raw in enumerate(rows):
        _require(isinstance(raw, Mapping), "packed parent evidence must be mappings")
        row = dict(raw)
        logical_id = row.get("chunk_id")
        _require(type(logical_id) is str and bool(logical_id), "parent chunk ID changed")
        result.append(
            {
                "logical_chunk_id": logical_id,
                "physical_chunk_id": _physical_id(row),
                "position": position,
                "row_sha256": _sha(row),
            }
        )
    return tuple(result)


def _row_span(row: CachedContentRow) -> EvidenceSpan:
    return EvidenceSpan(
        chunk_id=row.chunk_id,
        start_char=0,
        end_char=len(row.text),
        quote_sha256=row.text_sha256,
        ordinal=row.ordinal,
        source_id=row.source_id,
        turn_start_char=row.turn_start_char,
        turn_id=row.turn_id,
        role=row.role,
        created_at=row.created_at,
    )


@dataclass(frozen=True, slots=True)
class UserLedEnvelopeShadowBudget:
    """Independent pre-packing bounds for envelope companions."""

    max_envelopes: int = 4
    max_turns_per_envelope: int = 8
    max_companion_chunks: int = 16
    max_companion_tokens: int = 800

    def __post_init__(self) -> None:
        for name in (
            "max_envelopes",
            "max_turns_per_envelope",
            "max_companion_chunks",
            "max_companion_tokens",
        ):
            value = getattr(self, name)
            minimum = 1 if name in {"max_envelopes", "max_turns_per_envelope"} else 0
            _require(
                type(value) is int and value >= minimum,
                f"{name} is outside its supported bound",
            )

    def projection(self) -> dict[str, int]:
        return {
            "max_envelopes": self.max_envelopes,
            "max_turns_per_envelope": self.max_turns_per_envelope,
            "max_companion_chunks": self.max_companion_chunks,
            "max_companion_tokens": self.max_companion_tokens,
        }


@dataclass(frozen=True, slots=True)
class UserLedEnvelopeShadowEnvelope:
    envelope_id: str
    source_id: str
    exchange_kind: str
    opener_turn_id: str | None
    turn_ids: tuple[str, ...]
    turn_ordinals: tuple[int, ...]
    chunk_ids: tuple[str, ...]
    chunk_turn_ids: tuple[str, ...]
    chunk_token_counts: tuple[int, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(bool(self.envelope_id) and bool(self.source_id), "envelope identity changed")
        _require(self.exchange_kind in {"user_led", "orphan_prelude"}, "exchange kind changed")
        _require(
            len(self.turn_ids) == len(self.turn_ordinals)
            and len(self.chunk_ids) == len(self.chunk_turn_ids)
            and len(self.chunk_ids) == len(self.chunk_token_counts)
            and bool(self.chunk_ids),
            "envelope coordinates changed",
        )
        _require(len(set(self.chunk_ids)) == len(self.chunk_ids), "envelope chunks repeat")
        _require(
            tuple(dict.fromkeys(self.chunk_turn_ids)) == self.turn_ids,
            "envelope turn/chunk order changed",
        )
        if self.exchange_kind == "user_led":
            _require(
                self.opener_turn_id == self.turn_ids[0],
                "user-led envelope opener changed",
            )
        else:
            _require(self.opener_turn_id is None, "orphan prelude acquired an opener")
        expected = identity_sha256(self.projection(include_receipt=False))
        _require(not self.receipt_sha256 or self.receipt_sha256 == expected, "envelope receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "chunk_ids": list(self.chunk_ids),
            "chunk_token_counts": list(self.chunk_token_counts),
            "chunk_turn_ids": list(self.chunk_turn_ids),
            "envelope_id": self.envelope_id,
            "exchange_kind": self.exchange_kind,
            "opener_turn_id": self.opener_turn_id,
            "source_id": self.source_id,
            "turn_ids": list(self.turn_ids),
            "turn_ordinals": list(self.turn_ordinals),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class UserLedEnvelopeShadowIndex:
    namespace_id: str
    cache_receipt_sha256: str
    envelopes: tuple[UserLedEnvelopeShadowEnvelope, ...]
    row_by_chunk_id: Mapping[str, CachedContentRow] = field(repr=False, compare=False)
    envelope_by_chunk_id: Mapping[str, str] = field(repr=False, compare=False)
    _envelope_by_id: Mapping[str, UserLedEnvelopeShadowEnvelope] = field(
        init=False,
        repr=False,
        compare=False,
    )
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(len(self.namespace_id) == 64, "namespace ID changed")
        _require(len(self.cache_receipt_sha256) == 64, "cache receipt changed")
        rows = dict(self.row_by_chunk_id)
        membership = dict(self.envelope_by_chunk_id)
        _require(set(rows) == set(membership), "envelope index lost cached rows")
        _require(len(set(membership.values())) <= len(self.envelopes), "unknown envelope membership")
        envelopes_by_id = {row.envelope_id: row for row in self.envelopes}
        _require(
            len(envelopes_by_id) == len(self.envelopes),
            "envelope identities repeated",
        )
        known = set(envelopes_by_id)
        _require(set(membership.values()).issubset(known), "chunk points outside envelope index")
        object.__setattr__(self, "row_by_chunk_id", MappingProxyType(rows))
        object.__setattr__(self, "envelope_by_chunk_id", MappingProxyType(membership))
        object.__setattr__(
            self,
            "_envelope_by_id",
            MappingProxyType(envelopes_by_id),
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        _require(not self.receipt_sha256 or self.receipt_sha256 == expected, "index receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    @property
    def envelope_by_id(self) -> Mapping[str, UserLedEnvelopeShadowEnvelope]:
        return self._envelope_by_id

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "cache_receipt_sha256": self.cache_receipt_sha256,
            "chunk_count": len(self.row_by_chunk_id),
            "envelope_count": len(self.envelopes),
            "envelopes_sha256": identity_sha256(
                [row.projection() for row in self.envelopes]
            ),
            "format": INDEX_FORMAT,
            "gold_loaded": False,
            "mechanism_id": MECHANISM_ID,
            "namespace_id": self.namespace_id,
            "provider_calls": 0,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        assert_gold_blind(value, path="user_led_envelope_shadow_index")
        return value


def build_user_led_envelope_shadow_index(
    cache: NamespacePartitionCache,
) -> UserLedEnvelopeShadowIndex:
    """Build a deterministic sidecar without writing the frozen store."""

    _require(type(cache) is NamespacePartitionCache, "expected an exact namespace cache")
    rows = tuple(row for values in cache.rows_by_partition.values() for row in values)
    by_source: dict[str, list[CachedContentRow]] = {}
    row_by_id: dict[str, CachedContentRow] = {}
    for row in rows:
        _require(type(row) is CachedContentRow, "cache row type changed")
        _require(row.chunk_id not in row_by_id, "cached chunk ID repeated")
        row_by_id[row.chunk_id] = row
        by_source.setdefault(row.source_id, []).append(row)

    envelopes: list[UserLedEnvelopeShadowEnvelope] = []
    membership: dict[str, str] = {}
    builder = UserLedEpisodeBuilder()
    artifact_id = f"user-led-shadow-{cache.namespace_id[:24]}"
    for source_id in sorted(by_source):
        source_rows = sorted(
            by_source[source_id],
            key=lambda row: (
                row.ordinal,
                row.turn_start_char,
                row.turn_end_char,
                row.chunk_id,
            ),
        )
        built = builder.build(
            source_id=source_id,
            artifact_id=artifact_id,
            spans=tuple(_row_span(row) for row in source_rows),
        )
        for shard in built.shards:
            # No sharding was requested, so each shard is one complete exchange.
            _require(shard.shard_count == 1, "historical envelope unexpectedly sharded")
            spans = shard.retrieval_evidence
            turn_ids = tuple(dict.fromkeys(str(span.turn_id) for span in spans))
            ordinal_by_turn = {str(span.turn_id): span.ordinal for span in spans}
            envelope = UserLedEnvelopeShadowEnvelope(
                envelope_id=shard.exchange_id,
                source_id=source_id,
                exchange_kind=shard.exchange_kind,
                opener_turn_id=(
                    None
                    if shard.exchange_kind == "orphan_prelude"
                    else str(shard.lead_evidence[0].turn_id)
                ),
                turn_ids=turn_ids,
                turn_ordinals=tuple(ordinal_by_turn[turn_id] for turn_id in turn_ids),
                chunk_ids=tuple(span.chunk_id for span in spans),
                chunk_turn_ids=tuple(str(span.turn_id) for span in spans),
                chunk_token_counts=tuple(row_by_id[span.chunk_id].token_count for span in spans),
            )
            envelopes.append(envelope)
            for chunk_id in envelope.chunk_ids:
                _require(chunk_id not in membership, "chunk belongs to multiple envelopes")
                membership[chunk_id] = envelope.envelope_id
    envelopes.sort(key=lambda row: (row.turn_ordinals[0], row.source_id, row.envelope_id))
    return UserLedEnvelopeShadowIndex(
        namespace_id=cache.namespace_id,
        cache_receipt_sha256=cache.cache_receipt_sha256,
        envelopes=tuple(envelopes),
        row_by_chunk_id=row_by_id,
        envelope_by_chunk_id=membership,
    )


def _exact_physical_parent(
    raw: Mapping[str, Any], row: CachedContentRow
) -> bool:
    logical_id = raw.get("chunk_id")
    evidence_id = raw.get("evidence_id")
    physical_id = raw.get("backing_chunk_id", logical_id)
    raw_text = raw.get("raw_text")
    rendered_text = raw.get("rendered_text")
    if (
        type(logical_id) is not str
        or not logical_id
        or evidence_id != logical_id
        or physical_id != row.chunk_id
        or type(raw_text) is not str
        or not raw_text
        or type(rendered_text) is not str
        or not rendered_text
    ):
        return False
    common_provenance_matches = (
        raw.get("turn_id") == row.turn_id
        and raw.get("source_id") == row.source_id
        and raw.get("role") == row.role
        and raw.get("created_at") == row.created_at
        and raw.get("raw_text_sha256") == quote_sha256(raw_text)
        and raw.get("rendered_text_sha256")
        == quote_sha256(rendered_text)
        and rendered_text.endswith(raw_text)
    )
    if not common_provenance_matches:
        return False
    if "backing_chunk_id" not in raw:
        return logical_id == row.chunk_id and raw_text == row.text
    return (
        logical_id != row.chunk_id
        and raw.get("excerpt_occurrence") is True
        and raw_text in row.text
    )


@dataclass(frozen=True, slots=True)
class UserLedEnvelopeShadowGroup:
    envelope_id: str
    source_id: str
    opener_turn_id: str
    parent_logical_chunk_ids: tuple[str, ...]
    anchor_chunk_ids: tuple[str, ...]
    selected_turn_ids: tuple[str, ...]
    selected_turn_ordinals: tuple[int, ...]
    ordered_chunk_ids: tuple[str, ...]
    companion_chunk_ids: tuple[str, ...]
    companion_token_count: int
    truncation_reasons: tuple[str, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(bool(self.parent_logical_chunk_ids), "group has no parent anchors")
        _require(bool(self.companion_chunk_ids), "empty companion groups are not selected")
        _require(set(self.anchor_chunk_ids).issubset(self.ordered_chunk_ids), "anchor escaped group")
        _require(set(self.companion_chunk_ids).issubset(self.ordered_chunk_ids), "companion escaped group")
        _require(self.selected_turn_ids[0] == self.opener_turn_id, "group opener is not first")
        expected = identity_sha256(self.projection(include_receipt=False))
        _require(not self.receipt_sha256 or self.receipt_sha256 == expected, "group receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "anchor_chunk_ids": list(self.anchor_chunk_ids),
            "companion_chunk_ids": list(self.companion_chunk_ids),
            "companion_token_count": self.companion_token_count,
            "envelope_id": self.envelope_id,
            "opener_turn_id": self.opener_turn_id,
            "ordered_chunk_ids": list(self.ordered_chunk_ids),
            "parent_logical_chunk_ids": list(self.parent_logical_chunk_ids),
            "selected_turn_ids": list(self.selected_turn_ids),
            "selected_turn_ordinals": list(self.selected_turn_ordinals),
            "source_id": self.source_id,
            "truncation_reasons": list(self.truncation_reasons),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class UserLedEnvelopeShadowSelection:
    index_receipt_sha256: str
    parent_footprint: tuple[dict[str, Any], ...]
    budget: UserLedEnvelopeShadowBudget
    groups: tuple[UserLedEnvelopeShadowGroup, ...]
    diagnostics: tuple[dict[str, Any], ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(len(self.index_receipt_sha256) == 64, "selection index receipt changed")
        _require(len(self.groups) <= self.budget.max_envelopes, "envelope cap exceeded")
        _require(
            sum(len(row.companion_chunk_ids) for row in self.groups)
            <= self.budget.max_companion_chunks,
            "companion chunk cap exceeded",
        )
        _require(
            sum(row.companion_token_count for row in self.groups)
            <= self.budget.max_companion_tokens,
            "companion token cap exceeded",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        _require(not self.receipt_sha256 or self.receipt_sha256 == expected, "selection receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "budget": self.budget.projection(),
            "diagnostics": [dict(row) for row in self.diagnostics],
            "format": SELECTION_FORMAT,
            "gold_loaded": False,
            "groups": [row.projection() for row in self.groups],
            "index_receipt_sha256": self.index_receipt_sha256,
            "parent_footprint": [dict(row) for row in self.parent_footprint],
            "provider_calls": 0,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        assert_gold_blind(value, path="user_led_envelope_shadow_selection")
        return value


def select_user_led_envelope_shadow(
    index: UserLedEnvelopeShadowIndex,
    packed_parent_evidence: Sequence[Mapping[str, Any]],
    *,
    budget: UserLedEnvelopeShadowBudget = UserLedEnvelopeShadowBudget(),
) -> UserLedEnvelopeShadowSelection:
    """Select bounded companions using parent anchors, never the question."""

    parent = tuple(dict(row) for row in packed_parent_evidence)
    footprint = _parent_footprint(parent)
    parent_physical_ids = {_physical_id(row) for row in parent}
    envelopes = index.envelope_by_id
    grouped: dict[str, list[tuple[int, Mapping[str, Any], CachedContentRow]]] = {}
    diagnostics: list[dict[str, Any]] = []
    for position, raw in enumerate(parent):
        logical_id = str(raw["chunk_id"])
        physical_id = _physical_id(raw)
        cached = index.row_by_chunk_id.get(physical_id)
        if cached is None:
            diagnostics.append(
                {
                    "logical_chunk_id": logical_id,
                    "physical_chunk_id": physical_id,
                    "reason": "parent_chunk_not_physical",
                }
            )
            continue
        if not _exact_physical_parent(raw, cached):
            diagnostics.append(
                {
                    "logical_chunk_id": logical_id,
                    "physical_chunk_id": physical_id,
                    "reason": "parent_chunk_mismatch",
                }
            )
            continue
        envelope_id = index.envelope_by_chunk_id[physical_id]
        envelope = envelopes[envelope_id]
        if envelope.exchange_kind != "user_led":
            diagnostics.append(
                {
                    "logical_chunk_id": logical_id,
                    "physical_chunk_id": physical_id,
                    "reason": "no_user_opener",
                }
            )
            continue
        grouped.setdefault(envelope_id, []).append((position, raw, cached))

    ordered_envelopes = sorted(grouped, key=lambda key: grouped[key][0][0])
    groups: list[UserLedEnvelopeShadowGroup] = []
    used_chunks = 0
    used_tokens = 0
    for envelope_id in ordered_envelopes:
        envelope = envelopes[envelope_id]
        anchor_rows = grouped[envelope_id]
        anchor_turn_ids = tuple(dict.fromkeys(row.turn_id for _, _, row in anchor_rows))
        required_turns = tuple(dict.fromkeys((str(envelope.opener_turn_id), *anchor_turn_ids)))
        if len(required_turns) > budget.max_turns_per_envelope:
            diagnostics.append({"envelope_id": envelope_id, "reason": "mandatory_turn_bound"})
            continue
        chunks_by_turn = {
            turn_id: tuple(
                chunk_id
                for chunk_id, owner in zip(envelope.chunk_ids, envelope.chunk_turn_ids, strict=True)
                if owner == turn_id
            )
            for turn_id in envelope.turn_ids
        }
        token_by_chunk = dict(zip(envelope.chunk_ids, envelope.chunk_token_counts, strict=True))
        required_ids = tuple(chunk for turn in required_turns for chunk in chunks_by_turn[turn])
        required_companions = tuple(chunk for chunk in required_ids if chunk not in parent_physical_ids)
        required_tokens = sum(token_by_chunk[chunk] for chunk in required_companions)
        if (
            used_chunks + len(required_companions) > budget.max_companion_chunks
            or used_tokens + required_tokens > budget.max_companion_tokens
        ):
            diagnostics.append({"envelope_id": envelope_id, "reason": "mandatory_companion_bound"})
            continue

        ordinal_by_turn = dict(zip(envelope.turn_ids, envelope.turn_ordinals, strict=True))
        anchor_ordinals = tuple(ordinal_by_turn[turn_id] for turn_id in anchor_turn_ids)
        candidates = sorted(
            (turn_id for turn_id in envelope.turn_ids if turn_id not in required_turns),
            key=lambda turn_id: (
                min(abs(ordinal_by_turn[turn_id] - value) for value in anchor_ordinals),
                ordinal_by_turn[turn_id],
                turn_id,
            ),
        )
        selected_turns = list(required_turns)
        local_companions = list(required_companions)
        local_tokens = required_tokens
        skipped_chunks = skipped_tokens = False
        for turn_id in candidates:
            if len(selected_turns) >= budget.max_turns_per_envelope:
                break
            additions = [chunk for chunk in chunks_by_turn[turn_id] if chunk not in parent_physical_ids]
            addition_tokens = sum(token_by_chunk[chunk] for chunk in additions)
            if used_chunks + len(local_companions) + len(additions) > budget.max_companion_chunks:
                skipped_chunks = True
                continue
            if used_tokens + local_tokens + addition_tokens > budget.max_companion_tokens:
                skipped_tokens = True
                continue
            selected_turns.append(turn_id)
            local_companions.extend(additions)
            local_tokens += addition_tokens
        if not local_companions:
            diagnostics.append({"envelope_id": envelope_id, "reason": "no_novel_companion"})
            continue
        selected_turns = sorted(set(selected_turns), key=lambda turn: (ordinal_by_turn[turn], turn))
        ordered_ids = tuple(chunk for turn in selected_turns for chunk in chunks_by_turn[turn])
        ordered_ids = tuple(chunk for chunk in ordered_ids if chunk in parent_physical_ids or chunk in local_companions)
        reasons: list[str] = []
        if len(selected_turns) < len(envelope.turn_ids):
            reasons.append("turn_bound")
        if skipped_chunks:
            reasons.append("chunk_bound")
        if skipped_tokens:
            reasons.append("token_bound")
        if len(groups) >= budget.max_envelopes:
            diagnostics.append(
                {
                    "envelope_id": envelope_id,
                    "logical_chunk_ids": [
                        str(raw["chunk_id"]) for _, raw, _ in anchor_rows
                    ],
                    "reason": "max_envelopes",
                }
            )
            continue
        groups.append(
            UserLedEnvelopeShadowGroup(
                envelope_id=envelope_id,
                source_id=envelope.source_id,
                opener_turn_id=str(envelope.opener_turn_id),
                parent_logical_chunk_ids=tuple(str(raw["chunk_id"]) for _, raw, _ in anchor_rows),
                anchor_chunk_ids=tuple(row.chunk_id for _, _, row in anchor_rows),
                selected_turn_ids=tuple(selected_turns),
                selected_turn_ordinals=tuple(ordinal_by_turn[turn] for turn in selected_turns),
                ordered_chunk_ids=ordered_ids,
                companion_chunk_ids=tuple(local_companions),
                companion_token_count=local_tokens,
                truncation_reasons=tuple(reasons),
            )
        )
        used_chunks += len(local_companions)
        used_tokens += local_tokens

    return UserLedEnvelopeShadowSelection(
        index_receipt_sha256=index.receipt_sha256,
        parent_footprint=footprint,
        budget=budget,
        groups=tuple(groups),
        diagnostics=tuple(diagnostics),
    )


def _companion_row(row: CachedContentRow) -> dict[str, Any]:
    rendered = f"[{row.created_at} | {row.role}] {row.text}"
    return {
        "evidence_id": row.chunk_id,
        "chunk_id": row.chunk_id,
        "turn_id": row.turn_id,
        "source_id": row.source_id,
        "role": row.role,
        "created_at": row.created_at,
        "route": COMPANION_ROUTE,
        "score": 0.0,
        "raw_text": row.text,
        "raw_text_sha256": row.text_sha256,
        "rendered_text": rendered,
        "rendered_text_sha256": quote_sha256(rendered),
    }


@dataclass(frozen=True, slots=True)
class UserLedEnvelopeShadowComposition:
    selection_receipt_sha256: str
    parent_footprint: tuple[dict[str, Any], ...]
    selected_group_ids: tuple[str, ...]
    admitted_group_ids: tuple[str, ...]
    rejected_group_ids: tuple[str, ...]
    selected_companion_chunk_ids: tuple[str, ...]
    admitted_companion_chunk_ids: tuple[str, ...]
    selected_companion_token_count: int
    admitted_companion_token_count: int
    context_token_count: int
    prompt_workspace_token_count: int
    max_context_tokens: int
    max_prompt_tokens: int
    packed_evidence: tuple[dict[str, Any], ...] = field(repr=False, compare=False)
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(set(self.admitted_group_ids).isdisjoint(self.rejected_group_ids), "group decision overlaps")
        _require(
            set(self.admitted_group_ids) | set(self.rejected_group_ids)
            == set(self.selected_group_ids),
            "group decision is incomplete",
        )
        _require(self.context_token_count <= self.max_context_tokens, "context cap exceeded")
        _require(self.prompt_workspace_token_count <= self.max_prompt_tokens, "workspace cap exceeded")
        expected = identity_sha256(self.projection(include_receipt=False))
        _require(not self.receipt_sha256 or self.receipt_sha256 == expected, "composition receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "admitted_companion_chunk_ids": list(self.admitted_companion_chunk_ids),
            "admitted_companion_token_count": self.admitted_companion_token_count,
            "admitted_group_ids": list(self.admitted_group_ids),
            "context_token_count": self.context_token_count,
            "format": COMPOSITION_FORMAT,
            "gold_loaded": False,
            "max_context_tokens": self.max_context_tokens,
            "max_prompt_tokens": self.max_prompt_tokens,
            "packed_evidence_count": len(self.packed_evidence),
            "packed_evidence_sha256": identity_sha256(list(self.packed_evidence)),
            "parent_footprint": [dict(row) for row in self.parent_footprint],
            "prompt_workspace_token_count": self.prompt_workspace_token_count,
            "provider_calls": 0,
            "rejected_group_ids": list(self.rejected_group_ids),
            "selected_companion_chunk_ids": list(self.selected_companion_chunk_ids),
            "selected_companion_token_count": self.selected_companion_token_count,
            "selected_group_ids": list(self.selected_group_ids),
            "selection_receipt_sha256": self.selection_receipt_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        assert_gold_blind(value, path="user_led_envelope_shadow_composition")
        return value


def compose_user_led_envelope_shadow(
    index: UserLedEnvelopeShadowIndex,
    selection: UserLedEnvelopeShadowSelection,
    packed_parent_evidence: Sequence[Mapping[str, Any]],
    *,
    measure_packet: Callable[[Sequence[Mapping[str, Any]]], tuple[int, int]],
    max_context_tokens: int,
    max_prompt_tokens: int,
) -> UserLedEnvelopeShadowComposition:
    """Admit selected groups atomically without evicting a parent occurrence."""

    _require(selection.index_receipt_sha256 == index.receipt_sha256, "selection index changed")
    parent = tuple(copy.deepcopy(dict(row)) for row in packed_parent_evidence)
    footprint = _parent_footprint(parent)
    _require(footprint == selection.parent_footprint, "composition parent changed")
    _require(type(max_context_tokens) is int and max_context_tokens >= 0, "context cap changed")
    _require(type(max_prompt_tokens) is int and max_prompt_tokens >= 0, "prompt cap changed")
    parent_hashes = Counter(_sha(row) for row in parent)
    parent_physical = {_physical_id(row) for row in parent}
    current = list(parent)
    initial_measure = measure_packet(current)
    _require(
        type(initial_measure) is tuple
        and len(initial_measure) == 2
        and all(type(value) is int and value >= 0 for value in initial_measure),
        "packet measurement changed",
    )
    _require(
        initial_measure[0] <= max_context_tokens and initial_measure[1] <= max_prompt_tokens,
        "sealed parent exceeds the requested caps",
    )
    current_measure = initial_measure
    admitted: list[str] = []
    rejected: list[str] = []
    admitted_companions: list[str] = []
    for group in selection.groups:
        logical_set = set(group.parent_logical_chunk_ids)
        indexes = [position for position, row in enumerate(current) if str(row["chunk_id"]) in logical_set]
        _require(len(indexes) == len(logical_set), "group parent membership changed")
        insertion = min(indexes)
        parent_by_physical: dict[str, list[dict[str, Any]]] = {}
        for row in current:
            if str(row["chunk_id"]) not in logical_set:
                continue
            parent_by_physical.setdefault(_physical_id(row), []).append(row)
        ordered: list[dict[str, Any]] = []
        local_companions: list[str] = []
        for chunk_id in group.ordered_chunk_ids:
            if chunk_id in parent_by_physical:
                ordered.extend(parent_by_physical[chunk_id])
            elif chunk_id in parent_physical:
                # A synthetic/external parent already owns this physical chunk.
                continue
            else:
                _require(chunk_id in group.companion_chunk_ids, "group contains an unselected row")
                cached = index.row_by_chunk_id.get(chunk_id)
                _require(cached is not None and cached.source_id == group.source_id, "companion provenance changed")
                ordered.append(_companion_row(cached))
                local_companions.append(chunk_id)
        remainder = [row for row in current if str(row["chunk_id"]) not in logical_set]
        proposed = [*remainder[:insertion], *ordered, *remainder[insertion:]]
        measured = measure_packet(proposed)
        _require(
            type(measured) is tuple
            and len(measured) == 2
            and all(type(value) is int and value >= 0 for value in measured),
            "packet measurement changed",
        )
        if measured[0] > max_context_tokens or measured[1] > max_prompt_tokens:
            rejected.append(group.envelope_id)
            continue
        current = proposed
        current_measure = measured
        admitted.append(group.envelope_id)
        admitted_companions.extend(local_companions)

    observed_parent_hashes = Counter(
        _sha(row) for row in current if str(row.get("route")) != COMPANION_ROUTE
    )
    _require(observed_parent_hashes == parent_hashes, "atomic composition evicted or changed a parent")
    selected_companions = tuple(
        chunk_id for group in selection.groups for chunk_id in group.companion_chunk_ids
    )
    return UserLedEnvelopeShadowComposition(
        selection_receipt_sha256=selection.receipt_sha256,
        parent_footprint=footprint,
        selected_group_ids=tuple(group.envelope_id for group in selection.groups),
        admitted_group_ids=tuple(admitted),
        rejected_group_ids=tuple(rejected),
        selected_companion_chunk_ids=selected_companions,
        admitted_companion_chunk_ids=tuple(admitted_companions),
        selected_companion_token_count=sum(
            index.row_by_chunk_id[chunk_id].token_count for chunk_id in selected_companions
        ),
        admitted_companion_token_count=sum(
            index.row_by_chunk_id[chunk_id].token_count for chunk_id in admitted_companions
        ),
        context_token_count=current_measure[0],
        prompt_workspace_token_count=current_measure[1],
        max_context_tokens=max_context_tokens,
        max_prompt_tokens=max_prompt_tokens,
        packed_evidence=tuple(current),
    )


__all__ = [
    "COMPANION_ROUTE",
    "UserLedEnvelopeShadowBudget",
    "UserLedEnvelopeShadowComposition",
    "UserLedEnvelopeShadowError",
    "UserLedEnvelopeShadowIndex",
    "UserLedEnvelopeShadowSelection",
    "build_user_led_envelope_shadow_index",
    "compose_user_led_envelope_shadow",
    "select_user_led_envelope_shadow",
]
