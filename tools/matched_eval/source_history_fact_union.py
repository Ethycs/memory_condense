"""Provider-free selected-source hydration and post-map fact union.

The factories below form one gold-blind lifecycle.  Source selections are
hydrated independently (including repeated selections), mapper facts are
salvaged with exact quotes, and only then are facts deduplicated, stripped of
direct-evidence copies, and packed into fixed non-borrowing lanes.  This module
does not execute a provider or retain transformer token state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

from memory_condense.application.discourse_sources import (
    SourceChunkStream,
    scan_discourse_source_chunks,
)
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import parse_source_metadata

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)


FORMAT = "memory-condense-source-history-fact-union-v2"
DEFAULT_HISTORY_WINDOW_TOKEN_CAP = 6_200
FINAL_PROMPT_TOKEN_CAP = 8_000
OUTPUT_TOKEN_RESERVE = 768
EXTERNAL_LINK_OVERLAY_TOKEN_RESERVE = 256


class FactLane(str, Enum):
    DIRECT = "direct"
    PARTITION = "partition"
    GUIDED = "guided"
    EM = "em"


LANE_ORDER = (
    FactLane.DIRECT,
    FactLane.PARTITION,
    FactLane.GUIDED,
    FactLane.EM,
)
LANE_TOKEN_BUDGETS: Mapping[FactLane, int] = MappingProxyType(
    {
        FactLane.DIRECT: 384,
        FactLane.PARTITION: 192,
        FactLane.GUIDED: 192,
        FactLane.EM: 256,
    }
)
MAX_PARENT_PROMPT_TOKENS = (
    FINAL_PROMPT_TOKEN_CAP
    - OUTPUT_TOKEN_RESERVE
    - EXTERNAL_LINK_OVERLAY_TOKEN_RESERVE
    - sum(LANE_TOKEN_BUDGETS.values())
)


class SourceHistoryFactUnionError(MatchedEvalContractError):
    """Raised when a hydration, provenance, or budget invariant changes."""


class FrozenSourceMembershipLike(Protocol):
    source_id: str
    content_chunk_ids: tuple[str, ...]
    metadata_chunk_ids: tuple[str, ...]
    stream_sha256: str


def _require(ok: object, message: str) -> None:
    if not ok:
        raise SourceHistoryFactUnionError(message)


def _integer(value: object, label: str, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"{label} must be >= {minimum}")
    return value  # type: ignore[return-value]


def _tuple(value: object, label: str) -> tuple[Any, ...]:
    _require(type(value) is tuple, f"{label} must be an immutable tuple")
    return value  # type: ignore[return-value]


def _seal(kind: str, body: Mapping[str, Any]) -> str:
    value = {"format": f"{FORMAT}-{kind}", **body}
    assert_gold_blind(value, path="source_history_fact_union")
    return identity_sha256(value)


def _raw_sha(value: object) -> str:
    try:
        return identity_sha256(value)
    except (TypeError, ValueError):
        return quote_sha256(repr(value))


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True, slots=True)
class ParentIdentity:
    population_identity_sha256: str
    question_order_sha256: str
    snapshot_id: str
    namespace_id: str
    parent_packet_id: str
    parent_stage_receipt_sha256: str
    direct_evidence_projection_sha256: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.population_identity_sha256, "population identity"),
            (self.question_order_sha256, "question order"),
            (self.snapshot_id, "snapshot ID"),
            (self.namespace_id, "namespace ID"),
            (self.parent_packet_id, "parent packet ID"),
            (self.parent_stage_receipt_sha256, "parent stage receipt"),
            (self.direct_evidence_projection_sha256, "direct evidence projection"),
        ):
            require_sha256(value, label)

    def projection(self) -> dict[str, str]:
        return {
            "direct_evidence_projection_sha256": self.direct_evidence_projection_sha256,
            "format": f"{FORMAT}-parent",
            "namespace_id": self.namespace_id,
            "parent_packet_id": self.parent_packet_id,
            "parent_stage_receipt_sha256": self.parent_stage_receipt_sha256,
            "population_identity_sha256": self.population_identity_sha256,
            "question_order_sha256": self.question_order_sha256,
            "snapshot_id": self.snapshot_id,
        }

    @property
    def identity_sha256(self) -> str:
        return identity_sha256(self.projection())


@dataclass(frozen=True, slots=True)
class FrozenHistoryChunk:
    source_id: str
    chunk_id: str
    turn_id: str
    turn_ordinal: int
    role: str
    created_at: str
    start_char: int
    end_char: int
    text: str
    token_count: int
    turn_text_sha256: str
    metadata_chunk: bool

    def projection(self, *, include_text: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "created_at": self.created_at,
            "end_char": self.end_char,
            "metadata_chunk": self.metadata_chunk,
            "role": self.role,
            "source_id": self.source_id,
            "start_char": self.start_char,
            "text_sha256": quote_sha256(self.text),
            "token_count": self.token_count,
            "turn_id": self.turn_id,
            "turn_ordinal": self.turn_ordinal,
            "turn_text_sha256": self.turn_text_sha256,
        }
        if include_text:
            value["text"] = self.text
        return value

    @property
    def chunk_receipt_sha256(self) -> str:
        return _seal("chunk", self.projection(include_text=True))


@dataclass(frozen=True, slots=True)
class HydratedSourceHistory:
    namespace_id: str
    source_id: str
    content_chunk_ids: tuple[str, ...]
    metadata_chunk_ids: tuple[str, ...]
    stream_sha256: str
    membership_projection_sha256: str
    chunks: tuple[FrozenHistoryChunk, ...]
    store_bytes_revalidated: bool
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class SourceSelection:
    selection_id: str
    lane: FactLane
    namespace_id: str
    source_id: str
    rank: int
    selector_receipt_sha256: str

    def __post_init__(self) -> None:
        require_text(self.selection_id, "source selection ID")
        _require(type(self.lane) is FactLane, "selection lane must be canonical")
        require_sha256(self.namespace_id, "selected source namespace ID")
        require_text(self.source_id, "selected source ID")
        _integer(self.rank, "source selection rank")
        require_sha256(self.selector_receipt_sha256, "source selector receipt")

    def projection(self) -> dict[str, Any]:
        return {
            "lane": self.lane.value,
            "namespace_id": self.namespace_id,
            "rank": self.rank,
            "selection_id": self.selection_id,
            "selector_receipt_sha256": self.selector_receipt_sha256,
            "source_id": self.source_id,
        }


@dataclass(frozen=True, slots=True)
class SourceHistoryWindow:
    parent_identity_sha256: str
    selection: SourceSelection
    history_receipt_sha256: str
    window_ordinal: int
    chunks: tuple[FrozenHistoryChunk, ...]
    content_token_proxy: int
    token_cap: int
    window_id: str
    receipt_sha256: str

    def mapping_payload(self) -> dict[str, Any]:
        value = {
            "chunks": [row.projection(include_text=True) for row in self.chunks],
            "format": f"{FORMAT}-mapping-input",
            "frozen_chunk_boundaries": True,
            "lane": self.selection.lane.value,
            "namespace_id": self.selection.namespace_id,
            "parent_identity_sha256": self.parent_identity_sha256,
            "selection_id": self.selection.selection_id,
            "source_id": self.selection.source_id,
            "window_id": self.window_id,
            "window_receipt_sha256": self.receipt_sha256,
        }
        assert_gold_blind(value, path="source_history_mapping_input")
        return value

    @property
    def mapping_payload_sha256(self) -> str:
        return identity_sha256(self.mapping_payload())


@dataclass(frozen=True, slots=True)
class SourceHistoryHydrationPlan:
    parent: ParentIdentity
    selections: tuple[SourceSelection, ...]
    histories: tuple[HydratedSourceHistory, ...]
    windows: tuple[SourceHistoryWindow, ...]
    max_window_tokens: int
    receipt_sha256: str

    @property
    def pending_window_ids(self) -> tuple[str, ...]:
        return tuple(row.window_id for row in self.windows)


def _membership_projection(membership: FrozenSourceMembershipLike) -> dict[str, Any]:
    source_id = require_text(membership.source_id, "frozen source ID")
    content = _tuple(membership.content_chunk_ids, "content chunk IDs")
    metadata = _tuple(membership.metadata_chunk_ids, "metadata chunk IDs")
    ids = content + metadata
    _require(
        ids and all(type(value) is str and value for value in ids),
        "frozen source requires exact non-empty chunk IDs",
    )
    _require(len(set(ids)) == len(ids), "frozen source chunk IDs repeat")
    return {
        "content_chunk_ids": list(content),
        "metadata_chunk_ids": list(metadata),
        "source_id": source_id,
        "stream_sha256": require_sha256(
            membership.stream_sha256, "frozen stream SHA-256"
        ),
    }


def _matching_stream(
    streams: Sequence[SourceChunkStream], membership: FrozenSourceMembershipLike
) -> SourceChunkStream:
    matches = tuple(row for row in streams if row.source_id == membership.source_id)
    _require(len(matches) == 1, "frozen source is absent or duplicated in store scan")
    stream = matches[0]
    _require(
        (
            stream.content_chunk_ids,
            stream.metadata_chunk_ids,
            stream.stream_sha256,
        )
        == (
            membership.content_chunk_ids,
            membership.metadata_chunk_ids,
            membership.stream_sha256,
        ),
        "frozen source membership differs from scan_discourse_source_chunks",
    )
    return stream


def hydrate_source_histories(
    database: Database,
    memberships: tuple[FrozenSourceMembershipLike, ...],
    *,
    namespace_id: str,
    revalidate_store_bytes: Callable[[], None],
) -> tuple[HydratedSourceHistory, ...]:
    """Hydrate selected memberships with one scan and one namespace read."""

    if type(database) is not Database:
        raise TypeError("database must be an exact Database")
    _require(database.read_only, "source hydration requires a read-only Database")
    _tuple(memberships, "frozen source memberships")
    _require(bool(memberships), "batch hydration requires selected memberships")
    namespace_id = require_sha256(namespace_id, "source-history namespace ID")
    _require(callable(revalidate_store_bytes), "store-byte revalidator is required")
    revalidate_store_bytes()
    projections = tuple(_membership_projection(row) for row in memberships)
    _require(
        len({row["source_id"] for row in projections}) == len(projections),
        "batch hydration memberships repeat a namespaced source",
    )
    scanned = scan_discourse_source_chunks(database)
    streams = tuple(_matching_stream(scanned, row) for row in memberships)
    expected_by_source = {
        row.source_id: set(row.content_chunk_ids + row.metadata_chunk_ids)
        for row in streams
    }
    chunks_by_source: dict[str, list[FrozenHistoryChunk]] = {
        row.source_id: [] for row in streams
    }
    rows = database.execute(
        "SELECT c.chunk_id,c.turn_id,c.text,c.start_char,c.end_char,c.token_count,"
        "t.text,t.source_id,t.role,t.created_at,t.ordinal "
        "FROM chunks c JOIN turns t ON t.turn_id=c.turn_id "
        "ORDER BY t.ordinal,c.rowid"
    )
    for row in rows:
        chunk_id, turn_id = str(row[0]), str(row[1])
        source_id = str(row[7] or turn_id).strip()
        if source_id not in chunks_by_source:
            continue
        _require(
            chunk_id in expected_by_source[source_id],
            "store scan and hydration chunk sets differ",
        )
        text, start, end, turn_text = str(row[2]), int(row[3]), int(row[4]), str(row[6])
        _require(
            0 <= start < end <= len(turn_text) and turn_text[start:end] == text,
            "stored chunk no longer matches its owning turn coordinates",
        )
        tokens = int(row[5])
        _require(tokens == count_tokens(text), "stored chunk token count changed")
        chunks_by_source[source_id].append(
            FrozenHistoryChunk(
                source_id=source_id,
                chunk_id=chunk_id,
                turn_id=turn_id,
                turn_ordinal=int(row[10]),
                role=require_text(str(row[8]), "history role"),
                created_at=require_text(str(row[9]), "history creation time"),
                start_char=start,
                end_char=end,
                text=text,
                token_count=tokens,
                turn_text_sha256=quote_sha256(turn_text),
                metadata_chunk=parse_source_metadata(text) is not None,
            )
        )
    for stream in streams:
        chunks = chunks_by_source[stream.source_id]
        expected = expected_by_source[stream.source_id]
        _require(
            len(chunks) == len(expected)
            and {row.chunk_id for row in chunks} == expected,
            "source hydration did not recover the complete frozen stream",
        )
        _require(
            tuple(row.chunk_id for row in chunks if not row.metadata_chunk)
            == stream.content_chunk_ids
            and tuple(row.chunk_id for row in chunks if row.metadata_chunk)
            == stream.metadata_chunk_ids,
            "hydrated chronology changed frozen membership",
        )
    revalidate_store_bytes()
    result: list[HydratedSourceHistory] = []
    for stream, projection in zip(streams, projections, strict=True):
        chunks = tuple(chunks_by_source[stream.source_id])
        membership_sha = identity_sha256(projection)
        body = {
            "chunk_receipt_sha256s": [row.chunk_receipt_sha256 for row in chunks],
            "content_chunk_ids": list(stream.content_chunk_ids),
            "database_read_only": True,
            "membership_projection_sha256": membership_sha,
            "metadata_chunk_ids": list(stream.metadata_chunk_ids),
            "namespace_id": namespace_id,
            "source_id": stream.source_id,
            "store_bytes_revalidated_before_after": True,
            "stream_sha256": stream.stream_sha256,
            "validated_against_scan_discourse_source_chunks": True,
        }
        result.append(
            HydratedSourceHistory(
                namespace_id,
                stream.source_id,
                stream.content_chunk_ids,
                stream.metadata_chunk_ids,
                stream.stream_sha256,
                membership_sha,
                chunks,
                True,
                _seal("hydrated-source", body),
            )
        )
    return tuple(result)


def hydrate_source_history(
    database: Database,
    membership: FrozenSourceMembershipLike,
    *,
    namespace_id: str,
    revalidate_store_bytes: Callable[[], None],
) -> HydratedSourceHistory:
    """Compatibility wrapper for a single selected membership."""

    return hydrate_source_histories(
        database,
        (membership,),
        namespace_id=namespace_id,
        revalidate_store_bytes=revalidate_store_bytes,
    )[0]


def window_source_history(
    parent: ParentIdentity,
    selection: SourceSelection,
    history: HydratedSourceHistory,
    *,
    max_window_tokens: int = DEFAULT_HISTORY_WINDOW_TOKEN_CAP,
) -> tuple[SourceHistoryWindow, ...]:
    """Greedily window one selected source without splitting a frozen chunk."""

    if not all(
        (
            type(parent) is ParentIdentity,
            type(selection) is SourceSelection,
            type(history) is HydratedSourceHistory,
        )
    ):
        raise TypeError("window inputs must use exact lifecycle types")
    cap = _integer(max_window_tokens, "history window cap", 1)
    _require(
        (selection.namespace_id, selection.source_id)
        == (parent.namespace_id, history.source_id)
        and history.namespace_id == parent.namespace_id,
        "selected source escaped its parent namespace",
    )
    groups: list[tuple[FrozenHistoryChunk, ...]] = []
    current: list[FrozenHistoryChunk] = []
    used = 0
    mapping_chunks = tuple(row for row in history.chunks if not row.metadata_chunk)
    _require(mapping_chunks, "selected source has no factual content chunks")
    for chunk in mapping_chunks:
        _require(
            chunk.token_count <= cap,
            "a frozen chunk exceeds the window cap and cannot be split",
        )
        if current and used + chunk.token_count > cap:
            groups.append(tuple(current))
            current, used = [], 0
        current.append(chunk)
        used += chunk.token_count
    if current:
        groups.append(tuple(current))
    result: list[SourceHistoryWindow] = []
    for ordinal, chunks in enumerate(groups):
        body = {
            "chunk_receipt_sha256s": [row.chunk_receipt_sha256 for row in chunks],
            "content_token_proxy": sum(row.token_count for row in chunks),
            "frozen_chunk_boundaries": True,
            "history_receipt_sha256": history.receipt_sha256,
            "parent_identity_sha256": parent.identity_sha256,
            "selection": selection.projection(),
            "token_cap": cap,
            "window_ordinal": ordinal,
        }
        window_id = _seal("window-id", body)
        result.append(
            SourceHistoryWindow(
                parent_identity_sha256=parent.identity_sha256,
                selection=selection,
                history_receipt_sha256=history.receipt_sha256,
                window_ordinal=ordinal,
                chunks=chunks,
                content_token_proxy=body["content_token_proxy"],
                token_cap=cap,
                window_id=window_id,
                receipt_sha256=_seal("window", {**body, "window_id": window_id}),
            )
        )
    return tuple(result)


def plan_source_history_hydration(
    parent: ParentIdentity,
    *,
    selections: tuple[SourceSelection, ...],
    histories: tuple[HydratedSourceHistory, ...],
    max_window_tokens: int = DEFAULT_HISTORY_WINDOW_TOKEN_CAP,
) -> SourceHistoryHydrationPlan:
    """Preserve every method selection as an independent map lifecycle."""

    if type(parent) is not ParentIdentity:
        raise TypeError("parent must be an exact ParentIdentity")
    _tuple(selections, "source selections")
    _tuple(histories, "hydrated histories")
    _require(
        all(type(row) is SourceSelection for row in selections)
        and all(type(row) is HydratedSourceHistory for row in histories),
        "hydration inputs changed type",
    )
    _require(
        len({row.selection_id for row in selections}) == len(selections)
        and len({(row.lane, row.rank) for row in selections}) == len(selections),
        "selection IDs or per-lane ranks repeat",
    )
    _require(
        all(row.namespace_id == parent.namespace_id for row in selections + histories),
        "hydration inputs escaped the parent namespace",
    )
    by_source = {(row.namespace_id, row.source_id): row for row in histories}
    selected = {(row.namespace_id, row.source_id) for row in selections}
    _require(
        len(by_source) == len(histories) and set(by_source) == selected,
        "hydrated histories must exactly cover namespaced selected sources",
    )
    cap = _integer(max_window_tokens, "history window cap", 1)
    windows = tuple(
        window
        for selection in selections
        for window in window_source_history(
            parent,
            selection,
            by_source[(selection.namespace_id, selection.source_id)],
            max_window_tokens=cap,
        )
    )
    body = {
        "history_receipt_sha256s": [row.receipt_sha256 for row in histories],
        "max_window_tokens": cap,
        "no_dedup_before_mapping": True,
        "parent_identity_sha256": parent.identity_sha256,
        "selections": [row.projection() for row in selections],
        "window_receipt_sha256s": [row.receipt_sha256 for row in windows],
    }
    return SourceHistoryHydrationPlan(
        parent, selections, histories, windows, cap, _seal("hydration-plan", body)
    )


_EVENT_KEYS = frozenset(
    {"event_time", "object", "polarity", "predicate", "status", "subject"}
)
_FACT_KEYS = frozenset(
    {
        "chunk_id",
        "event_tuple",
        "fact",
        "mapper_item_id",
        "quote",
        "quote_end_char",
        "quote_sha256",
        "quote_start_char",
        "source_id",
    }
)


@dataclass(frozen=True, slots=True)
class EventTuple:
    subject: str
    predicate: str
    object_value: str
    event_time: str
    polarity: str
    status: str

    def __post_init__(self) -> None:
        for value in (
            self.subject,
            self.predicate,
            self.object_value,
            self.event_time,
            self.polarity,
            self.status,
        ):
            require_text(value, "full event coordinate")

    def projection(self) -> dict[str, str]:
        return {
            "event_time": self.event_time,
            "object": self.object_value,
            "polarity": self.polarity,
            "predicate": self.predicate,
            "status": self.status,
            "subject": self.subject,
        }


def _event(value: object) -> EventTuple | None:
    if value is None:
        return None
    _require(type(value) is dict and set(value) == _EVENT_KEYS, "event_tuple_schema")
    assert type(value) is dict
    return EventTuple(
        value["subject"],
        value["predicate"],
        value["object"],
        value["event_time"],
        value["polarity"],
        value["status"],
    )


def _dedup(
    namespace_id: str,
    source_id: str,
    quote_sha: str,
    event: EventTuple | None,
    *,
    chunk_id: str | None = None,
) -> dict[str, Any]:
    if event is not None:
        return {"event_tuple": event.projection(), "kind": "full_event_tuple"}
    if chunk_id is not None:
        return {
            "chunk_id": chunk_id,
            "kind": "source_chunk_quote",
            "namespace_id": namespace_id,
            "quote_sha256": quote_sha,
            "source_id": source_id,
        }
    return {
        "kind": "source_quote",
        "namespace_id": namespace_id,
        "quote_sha256": quote_sha,
        "source_id": source_id,
    }


@dataclass(frozen=True, slots=True)
class ValidatedMappedFact:
    mapper_item_id: str
    source_index: int
    lane: FactLane
    selection_id: str
    namespace_id: str
    source_id: str
    window_id: str
    chunk_id: str
    fact: str
    fact_token_proxy: int
    quote: str
    quote_sha256: str
    quote_start_char: int
    quote_end_char: int
    source_created_at: str
    source_role: str
    event_tuple: EventTuple | None
    item_receipt_sha256: str

    @property
    def dedup_projection(self) -> dict[str, Any]:
        return _dedup(
            self.namespace_id,
            self.source_id,
            self.quote_sha256,
            self.event_tuple,
            chunk_id=self.chunk_id,
        )

    @property
    def dedup_key_sha256(self) -> str:
        return identity_sha256(self.dedup_projection)


@dataclass(frozen=True, slots=True)
class RejectedMappedFact:
    source_index: int
    window_id: str
    reason: str
    raw_item_sha256: str
    rejection_receipt_sha256: str


@dataclass(frozen=True, slots=True)
class MappedFactBatch:
    parent_identity_sha256: str
    hydration_plan_receipt_sha256: str
    window_id: str
    window_receipt_sha256: str
    source_item_count: int
    accepted: tuple[ValidatedMappedFact, ...]
    rejected: tuple[RejectedMappedFact, ...]
    receipt_sha256: str


def _reject(
    window: SourceHistoryWindow, index: int, reason: str, raw: object
) -> RejectedMappedFact:
    raw_sha = _raw_sha(raw)
    body = {
        "raw_item_sha256": raw_sha,
        "reason": reason,
        "source_index": index,
        "window_id": window.window_id,
    }
    return RejectedMappedFact(index, window.window_id, reason, raw_sha, _seal("map-reject", body))


def _parse_fact(
    window: SourceHistoryWindow, index: int, raw: object
) -> ValidatedMappedFact:
    _require(type(raw) is dict and set(raw) == _FACT_KEYS, "item_schema")
    assert type(raw) is dict
    item_id = require_text(raw["mapper_item_id"], "mapper item ID")
    fact = require_text(raw["fact"], "mapped fact")
    source_id = require_text(raw["source_id"], "mapped source ID")
    chunk_id = require_text(raw["chunk_id"], "mapped chunk ID")
    quote = require_text(raw["quote"], "exact mapped quote")
    quote_sha = require_sha256(raw["quote_sha256"], "mapped quote SHA-256")
    start = _integer(raw["quote_start_char"], "mapped quote start")
    end = _integer(raw["quote_end_char"], "mapped quote end", 1)
    event = _event(raw["event_tuple"])
    _require(source_id == window.selection.source_id, "source_mismatch")
    chunks = tuple(row for row in window.chunks if row.chunk_id == chunk_id)
    _require(len(chunks) == 1, "unknown_chunk")
    chunk = chunks[0]
    _require(0 <= start < end <= len(chunk.text), "quote_coordinates")
    _require(chunk.text[start:end] == quote, "quote_not_exact")
    _require(quote_sha == quote_sha256(quote), "quote_sha256_mismatch")
    body = {
        "chunk_id": chunk_id,
        "event_tuple": None if event is None else event.projection(),
        "fact": fact,
        "fact_token_proxy": count_tokens(fact),
        "lane": window.selection.lane.value,
        "mapper_item_id": item_id,
        "namespace_id": window.selection.namespace_id,
        "quote": quote,
        "quote_end_char": end,
        "quote_sha256": quote_sha,
        "quote_start_char": start,
        "selection_id": window.selection.selection_id,
        "source_created_at": chunk.created_at,
        "source_id": source_id,
        "source_index": index,
        "source_role": chunk.role,
        "window_id": window.window_id,
    }
    return ValidatedMappedFact(
        item_id,
        index,
        window.selection.lane,
        window.selection.selection_id,
        window.selection.namespace_id,
        source_id,
        window.window_id,
        chunk_id,
        fact,
        count_tokens(fact),
        quote,
        quote_sha,
        start,
        end,
        chunk.created_at,
        chunk.role,
        event,
        _seal("mapped-item", body),
    )


def validate_mapped_facts(
    plan: SourceHistoryHydrationPlan,
    window: SourceHistoryWindow,
    raw_items: tuple[object, ...],
) -> MappedFactBatch:
    """Salvage each exact cited fact independently; do not deduplicate it."""

    if type(plan) is not SourceHistoryHydrationPlan or type(window) is not SourceHistoryWindow:
        raise TypeError("map validation requires exact plan/window types")
    _tuple(raw_items, "raw mapped facts")
    matches = tuple(row for row in plan.windows if row.window_id == window.window_id)
    _require(
        len(matches) == 1
        and matches[0] == window
        and window.parent_identity_sha256 == plan.parent.identity_sha256,
        "map window escaped its immutable hydration plan",
    )
    accepted: list[ValidatedMappedFact] = []
    rejected: list[RejectedMappedFact] = []
    item_ids: set[str] = set()
    for index, raw in enumerate(raw_items):
        try:
            item = _parse_fact(window, index, raw)
            _require(item.mapper_item_id not in item_ids, "duplicate_mapper_item_id")
        except (MatchedEvalContractError, TypeError, KeyError) as exc:
            rejected.append(_reject(window, index, str(exc) or type(exc).__name__, raw))
            continue
        item_ids.add(item.mapper_item_id)
        accepted.append(item)
    body = {
        "accepted": [row.item_receipt_sha256 for row in accepted],
        "hydration_plan_receipt_sha256": plan.receipt_sha256,
        "parent_identity_sha256": plan.parent.identity_sha256,
        "rejected": [row.rejection_receipt_sha256 for row in rejected],
        "salvage_valid_items_individually": True,
        "source_item_count": len(raw_items),
        "window_id": window.window_id,
        "window_receipt_sha256": window.receipt_sha256,
    }
    return MappedFactBatch(
        plan.parent.identity_sha256,
        plan.receipt_sha256,
        window.window_id,
        window.receipt_sha256,
        len(raw_items),
        tuple(accepted),
        tuple(rejected),
        _seal("map-batch", body),
    )


def validate_mapper_completion(
    plan: SourceHistoryHydrationPlan,
    window: SourceHistoryWindow,
    completion: str,
) -> MappedFactBatch:
    """Strictly decode ``{"facts": [...]}`` and delegate item salvage."""

    if type(completion) is not str:
        raise TypeError("mapper completion must be exact text")
    try:
        value = json.loads(
            completion,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (json.JSONDecodeError, ValueError):
        value = None
    if type(value) is dict and set(value) == {"facts"} and type(value["facts"]) is list:
        return validate_mapped_facts(plan, window, tuple(value["facts"]))
    # Preserve the same batch contract for a root-level failure.
    rejected = _reject(window, -1, "root_schema_or_invalid_json", completion)
    body = {
        "accepted": [],
        "hydration_plan_receipt_sha256": plan.receipt_sha256,
        "parent_identity_sha256": plan.parent.identity_sha256,
        "rejected": [rejected.rejection_receipt_sha256],
        "salvage_valid_items_individually": True,
        "source_item_count": 1,
        "window_id": window.window_id,
        "window_receipt_sha256": window.receipt_sha256,
    }
    return MappedFactBatch(
        plan.parent.identity_sha256,
        plan.receipt_sha256,
        window.window_id,
        window.receipt_sha256,
        1,
        (),
        (rejected,),
        _seal("map-batch", body),
    )


@dataclass(frozen=True, slots=True)
class DirectEvidenceRef:
    evidence_id: str
    namespace_id: str
    source_id: str
    quote_sha256: str
    evidence_receipt_sha256: str
    event_tuple: EventTuple | None = None
    text: str | None = None

    def __post_init__(self) -> None:
        require_text(self.evidence_id, "direct evidence ID")
        require_sha256(self.namespace_id, "direct evidence namespace ID")
        require_text(self.source_id, "direct evidence source ID")
        require_sha256(self.quote_sha256, "direct evidence quote SHA-256")
        require_sha256(self.evidence_receipt_sha256, "direct evidence receipt")
        _require(
            self.event_tuple is None or type(self.event_tuple) is EventTuple,
            "direct event tuple changed type",
        )
        _require(
            self.text is None
            or (
                type(self.text) is str
                and bool(self.text)
                and quote_sha256(self.text) == self.quote_sha256
            ),
            "direct exposed text changed its exact SHA-256 binding",
        )

    def projection(self) -> dict[str, Any]:
        return {
            "event_tuple": None if self.event_tuple is None else self.event_tuple.projection(),
            "evidence_id": self.evidence_id,
            "evidence_receipt_sha256": self.evidence_receipt_sha256,
            "exposed_text_sha256": (
                None if self.text is None else quote_sha256(self.text)
            ),
            "namespace_id": self.namespace_id,
            "quote_sha256": self.quote_sha256,
            "source_id": self.source_id,
        }

    @property
    def dedup_key_sha256(self) -> str:
        return identity_sha256(
            _dedup(self.namespace_id, self.source_id, self.quote_sha256, self.event_tuple)
        )


def direct_evidence_projection_sha256(
    evidence: tuple[DirectEvidenceRef, ...],
) -> str:
    _tuple(evidence, "direct evidence")
    _require(
        all(type(row) is DirectEvidenceRef for row in evidence)
        and len({row.evidence_id for row in evidence}) == len(evidence),
        "direct evidence must have exact types and unique IDs",
    )
    return identity_sha256([row.projection() for row in evidence])


@dataclass(frozen=True, slots=True)
class FactOrigin:
    lane: FactLane
    selection_id: str
    window_id: str
    namespace_id: str
    source_id: str
    chunk_id: str
    quote: str
    quote_sha256: str
    quote_start_char: int
    quote_end_char: int
    source_created_at: str
    source_role: str
    mapper_item_id: str
    mapped_item_receipt_sha256: str

    def projection(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "lane": self.lane.value,
            "mapped_item_receipt_sha256": self.mapped_item_receipt_sha256,
            "mapper_item_id": self.mapper_item_id,
            "namespace_id": self.namespace_id,
            "quote": self.quote,
            "quote_end_char": self.quote_end_char,
            "quote_sha256": self.quote_sha256,
            "quote_start_char": self.quote_start_char,
            "selection_id": self.selection_id,
            "source_created_at": self.source_created_at,
            "source_id": self.source_id,
            "source_role": self.source_role,
            "window_id": self.window_id,
        }


@dataclass(frozen=True, slots=True)
class UnionFact:
    union_fact_id: str
    dedup_key_sha256: str
    dedup_projection: Mapping[str, Any]
    fact_variants: tuple[str, ...]
    event_tuple: EventTuple | None
    origins: tuple[FactOrigin, ...]
    owner_lane: FactLane
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class DirectEvidenceExclusion:
    union_fact_id: str
    matching_direct_evidence_ids: tuple[str, ...]
    match_modes: tuple[str, ...]
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class PostMapFactUnion:
    parent: ParentIdentity
    hydration_plan_receipt_sha256: str
    completed_window_ids: tuple[str, ...]
    pending_window_ids: tuple[str, ...]
    map_batch_receipt_sha256s: tuple[str, ...]
    accepted_before_dedup_count: int
    rejected_item_count: int
    union_facts_before_direct_exclusion: tuple[UnionFact, ...]
    direct_exclusions: tuple[DirectEvidenceExclusion, ...]
    retained_facts: tuple[UnionFact, ...]
    receipt_sha256: str


def _origin(fact: ValidatedMappedFact) -> FactOrigin:
    return FactOrigin(
        fact.lane,
        fact.selection_id,
        fact.window_id,
        fact.namespace_id,
        fact.source_id,
        fact.chunk_id,
        fact.quote,
        fact.quote_sha256,
        fact.quote_start_char,
        fact.quote_end_char,
        fact.source_created_at,
        fact.source_role,
        fact.mapper_item_id,
        fact.item_receipt_sha256,
    )


def _union_group(facts: Sequence[ValidatedMappedFact]) -> UnionFact:
    key = facts[0].dedup_key_sha256
    _require(all(row.dedup_key_sha256 == key for row in facts), "dedup group changed key")
    variants = tuple(dict.fromkeys(row.fact for row in facts))
    origin_by_receipt: dict[str, FactOrigin] = {}
    for row in facts:
        origin_by_receipt.setdefault(row.item_receipt_sha256, _origin(row))
    origins = tuple(origin_by_receipt.values())
    owner = min((row.lane for row in origins), key=LANE_ORDER.index)
    union_id = _seal("union-fact-id", {"dedup_key_sha256": key})
    body = {
        "dedup_key_sha256": key,
        "dedup_projection": facts[0].dedup_projection,
        "fact_variants": list(variants),
        "origins": [row.projection() for row in origins],
        "owner_lane": owner.value,
        "union_fact_id": union_id,
    }
    return UnionFact(
        union_id,
        key,
        facts[0].dedup_projection,
        variants,
        facts[0].event_tuple,
        origins,
        owner,
        _seal("union-fact", body),
    )


def _direct_exclusion_match(
    plan: SourceHistoryHydrationPlan,
    fact: UnionFact,
    direct_evidence: Sequence[DirectEvidenceRef],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return a complete, ordered coverage proof for one post-map union fact.

    Legacy refs without exposed text retain exact event/quote-hash behavior.
    Refs carrying exposed text may additionally cover an exact mapped quote,
    including a strict substring, only when that text resolves to one unique
    frozen chunk in the same source and that exact chunk owns the mapped quote.
    A multi-origin union is excluded only when every origin is covered.
    """

    all_origins = {row.mapped_item_receipt_sha256 for row in fact.origins}
    covered: set[str] = set()
    evidence_modes: list[tuple[str, str]] = []
    chunks = tuple(
        chunk
        for history in plan.histories
        for chunk in history.chunks
        if not chunk.metadata_chunk
    )
    for direct in direct_evidence:
        if (
            direct.event_tuple is not None
            and fact.event_tuple is not None
            and direct.event_tuple.projection() == fact.event_tuple.projection()
        ):
            covered.update(all_origins)
            evidence_modes.append((direct.evidence_id, "exact_event_tuple"))
            continue
        if direct.text is None:
            hits = {
                origin.mapped_item_receipt_sha256
                for origin in fact.origins
                if origin.namespace_id == direct.namespace_id
                and origin.source_id == direct.source_id
                and origin.quote_sha256 == direct.quote_sha256
            }
            if hits:
                covered.update(hits)
                evidence_modes.append(
                    (direct.evidence_id, "legacy_exact_quote_hash")
                )
            continue
        containing_chunks = tuple(
            chunk
            for chunk in chunks
            if chunk.source_id == direct.source_id and direct.text in chunk.text
        )
        # Ambiguous repeated chunk text cannot prove a provenance parent.
        if len(containing_chunks) != 1:
            continue
        direct_chunk = containing_chunks[0]
        hits = {
            origin.mapped_item_receipt_sha256
            for origin in fact.origins
            if origin.namespace_id == direct.namespace_id
            and origin.source_id == direct.source_id
            and origin.chunk_id == direct_chunk.chunk_id
            and origin.quote in direct.text
        }
        if not hits:
            continue
        covered.update(hits)
        hit_origins = tuple(
            origin
            for origin in fact.origins
            if origin.mapped_item_receipt_sha256 in hits
        )
        mode = (
            "same_chunk_strict_substring"
            if all(origin.quote != direct.text for origin in hit_origins)
            else "same_chunk_exact_or_contained_quote"
        )
        evidence_modes.append((direct.evidence_id, mode))
    if covered != all_origins:
        return (), ()
    matches = tuple(dict.fromkeys(evidence_id for evidence_id, _mode in evidence_modes))
    modes = tuple(dict.fromkeys(mode for _evidence_id, mode in evidence_modes))
    return matches, modes


def build_post_map_fact_union(
    plan: SourceHistoryHydrationPlan,
    *,
    batches: tuple[MappedFactBatch, ...] = (),
    direct_evidence: tuple[DirectEvidenceRef, ...] = (),
) -> PostMapFactUnion:
    """Post-map exact dedup, provenance merge, then direct-evidence exclusion."""

    if type(plan) is not SourceHistoryHydrationPlan:
        raise TypeError("plan must be an exact hydration plan")
    _tuple(batches, "mapped batches")
    _tuple(direct_evidence, "direct evidence")
    _require(
        all(type(row) is MappedFactBatch for row in batches)
        and all(type(row) is DirectEvidenceRef for row in direct_evidence),
        "union inputs changed type",
    )
    _require(
        all(row.namespace_id == plan.parent.namespace_id for row in direct_evidence)
        and direct_evidence_projection_sha256(direct_evidence)
        == plan.parent.direct_evidence_projection_sha256,
        "direct evidence differs from immutable namespaced parent projection",
    )
    windows = {row.window_id: row for row in plan.windows}
    by_window: dict[str, MappedFactBatch] = {}
    for batch in batches:
        _require(batch.window_id not in by_window, "map window has multiple batches")
        _require(
            batch.window_id in windows
            and batch.parent_identity_sha256 == plan.parent.identity_sha256
            and batch.hydration_plan_receipt_sha256 == plan.receipt_sha256
            and batch.window_receipt_sha256 == windows[batch.window_id].receipt_sha256,
            "map batch escaped its immutable plan binding",
        )
        by_window[batch.window_id] = batch
    ordered = tuple(by_window[row.window_id] for row in plan.windows if row.window_id in by_window)
    accepted = tuple(item for batch in ordered for item in batch.accepted)
    # Source-history facts establish the union first.  EM is a later
    # representation stage and may merge provenance into that union, but it
    # cannot silently become the canonical earlier source fact.
    source_items = tuple(row for row in accepted if row.lane is not FactLane.EM)
    em_items = tuple(row for row in accepted if row.lane is FactLane.EM)
    groups: dict[str, list[ValidatedMappedFact]] = {}
    for item in (*source_items, *em_items):
        groups.setdefault(item.dedup_key_sha256, []).append(item)
    unioned = tuple(_union_group(group) for group in groups.values())
    exclusions: list[DirectEvidenceExclusion] = []
    retained: list[UnionFact] = []
    for fact in unioned:
        matches, modes = _direct_exclusion_match(plan, fact, direct_evidence)
        if not matches:
            retained.append(fact)
            continue
        body = {
            "match_modes": list(modes),
            "matching_direct_evidence_ids": list(matches),
            "operation_position": "after_map_and_post_map_chunk_dedup",
            "union_fact_id": fact.union_fact_id,
        }
        exclusions.append(
            DirectEvidenceExclusion(
                fact.union_fact_id,
                matches,
                modes,
                _seal("direct-exclusion", body),
            )
        )
    completed = tuple(row.window_id for row in plan.windows if row.window_id in by_window)
    pending = tuple(row.window_id for row in plan.windows if row.window_id not in by_window)
    batch_receipts = tuple(row.receipt_sha256 for row in ordered)
    body = {
        "accepted_before_dedup_count": len(accepted),
        "completed_window_ids": list(completed),
        "direct_exclusions": [row.receipt_sha256 for row in exclusions],
        "hydration_plan_receipt_sha256": plan.receipt_sha256,
        "map_batch_receipt_sha256s": list(batch_receipts),
        "operation_order": [
            "hydrate_without_dedup",
            "map_validate_individually",
            "source_history_post_map_exact_chunk_dedup",
            "sequential_em_fact_union",
            "direct_evidence_exact_or_same_chunk_child_exclusion",
            "non_borrowing_lane_pack",
        ],
        "parent_identity_sha256": plan.parent.identity_sha256,
        "pending_window_ids": list(pending),
        "rejected_item_count": sum(len(row.rejected) for row in ordered),
        "retained_union_fact_ids": [row.union_fact_id for row in retained],
        "union_fact_ids": [row.union_fact_id for row in unioned],
    }
    return PostMapFactUnion(
        plan.parent,
        plan.receipt_sha256,
        completed,
        pending,
        batch_receipts,
        len(accepted),
        sum(len(row.rejected) for row in ordered),
        unioned,
        tuple(exclusions),
        tuple(retained),
        _seal("post-map-union", body),
    )


@dataclass(frozen=True, slots=True)
class LaneAdmission:
    alias: str
    union_fact: UnionFact
    rendered_line: str
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class LanePack:
    lane: FactLane
    token_cap: int
    candidate_union_fact_ids: tuple[str, ...]
    admissions: tuple[LaneAdmission, ...]
    not_admitted_union_fact_ids: tuple[str, ...]
    rendered_block: str
    tokens_used: int
    non_borrowing: bool
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class FactUnionEnvelope:
    parent: ParentIdentity
    fact_union_receipt_sha256: str
    parent_prompt_token_proxy: int
    lane_packs: tuple[LanePack, ...]
    rendered_fact_union: str
    fact_union_token_proxy: int
    external_link_overlay_token_reserve: int
    output_token_reserve: int
    final_envelope_token_proxy: int
    hard_prompt_token_cap: int
    retained_transformer_token_state_bytes: int
    receipt_sha256: str

    def __post_init__(self) -> None:
        _require(
            type(self.retained_transformer_token_state_bytes) is int
            and self.retained_transformer_token_state_bytes == 0,
            "envelope retained transformer token state",
        )


def compact_fact_prompt_projection(alias: str, fact: UnionFact) -> dict[str, Any]:
    """Project only answer-relevant fact content behind a stable local alias.

    Exact quotes, IDs, hashes, namespaces, chunks, windows, and mapper
    coordinates remain transitively sealed by ``LaneAdmission`` ->
    ``UnionFact`` receipts and are deliberately not model-visible.
    """

    require_text(alias, "compact fact evidence alias")
    if type(fact) is not UnionFact:
        raise TypeError("compact prompt fact must be an exact UnionFact")
    contexts: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for origin in fact.origins:
        key = (origin.source_created_at, origin.source_role)
        if key not in seen:
            seen.add(key)
            contexts.append(
                {"date": origin.source_created_at, "role": origin.source_role}
            )
    value: dict[str, Any] = {
        "contexts": contexts,
        "evidence_id": alias,
        "facts": list(fact.fact_variants),
    }
    if fact.event_tuple is not None:
        value["event"] = fact.event_tuple.projection()
    return value


def _fact_line(alias: str, fact: UnionFact) -> str:
    return _json(compact_fact_prompt_projection(alias, fact))


def _pack_lane(lane: FactLane, facts: Sequence[UnionFact]) -> LanePack:
    cap, header, prefix = LANE_TOKEN_BUDGETS[lane], f"[{lane.value.upper()}_FACTS]", lane.value[0].upper()
    admitted: list[LaneAdmission] = []
    omitted: list[str] = []
    lines: list[str] = []
    for fact in facts:
        alias = f"{prefix}{len(admitted) + 1:03d}"
        line = _fact_line(alias, fact)
        if count_tokens(header + "\n" + "\n".join((*lines, line))) > cap:
            omitted.append(fact.union_fact_id)
            continue
        body = {
            "alias": alias,
            "rendered_line_sha256": quote_sha256(line),
            "union_fact_id": fact.union_fact_id,
            "union_fact_receipt_sha256": fact.receipt_sha256,
        }
        admitted.append(LaneAdmission(alias, fact, line, _seal("lane-admission", body)))
        lines.append(line)
    block = "" if not lines else header + "\n" + "\n".join(lines)
    used = count_tokens(block)
    body = {
        "admissions": [row.receipt_sha256 for row in admitted],
        "candidate_union_fact_ids": [row.union_fact_id for row in facts],
        "lane": lane.value,
        "non_borrowing": True,
        "not_admitted_union_fact_ids": omitted,
        "rendered_block_sha256": quote_sha256(block),
        "token_cap": cap,
        "tokens_used": used,
    }
    return LanePack(
        lane,
        cap,
        tuple(row.union_fact_id for row in facts),
        tuple(admitted),
        tuple(omitted),
        block,
        used,
        True,
        _seal("lane-pack", body),
    )


def pack_fact_union_envelope(
    fact_union: PostMapFactUnion,
    *,
    parent_prompt_token_proxy: int,
) -> FactUnionEnvelope:
    """Pack four fixed lanes and prove parent + facts + 768 reserve <= 8k."""

    if type(fact_union) is not PostMapFactUnion:
        raise TypeError("fact_union must be an exact PostMapFactUnion")
    parent_tokens = _integer(parent_prompt_token_proxy, "parent prompt tokens")
    _require(
        parent_tokens <= MAX_PARENT_PROMPT_TOKENS,
        "parent prompt exceeds capacity reserved for fixed fact lanes",
    )
    packs = tuple(
        _pack_lane(
            lane,
            tuple(row for row in fact_union.retained_facts if row.owner_lane is lane),
        )
        for lane in LANE_ORDER
    )
    rendered = "\n\n".join(row.rendered_block for row in packs if row.rendered_block)
    fact_tokens = count_tokens(rendered)
    final_tokens = (
        parent_tokens
        + fact_tokens
        + EXTERNAL_LINK_OVERLAY_TOKEN_RESERVE
        + OUTPUT_TOKEN_RESERVE
    )
    _require(final_tokens <= FINAL_PROMPT_TOKEN_CAP, "final 8k envelope overflow")
    body = {
        "fact_union_receipt_sha256": fact_union.receipt_sha256,
        "fact_union_token_proxy": fact_tokens,
        "external_link_overlay_token_reserve": EXTERNAL_LINK_OVERLAY_TOKEN_RESERVE,
        "final_envelope_token_proxy": final_tokens,
        "hard_prompt_token_cap": FINAL_PROMPT_TOKEN_CAP,
        "lane_pack_receipt_sha256s": [row.receipt_sha256 for row in packs],
        "lane_token_budgets": {lane.value: LANE_TOKEN_BUDGETS[lane] for lane in LANE_ORDER},
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "parent_identity_sha256": fact_union.parent.identity_sha256,
        "parent_prompt_token_proxy": parent_tokens,
        "rendered_fact_union_sha256": quote_sha256(rendered),
        "retained_transformer_token_state_bytes": 0,
    }
    return FactUnionEnvelope(
        fact_union.parent,
        fact_union.receipt_sha256,
        parent_tokens,
        packs,
        rendered,
        fact_tokens,
        EXTERNAL_LINK_OVERLAY_TOKEN_RESERVE,
        OUTPUT_TOKEN_RESERVE,
        final_tokens,
        FINAL_PROMPT_TOKEN_CAP,
        0,
        _seal("envelope", body),
    )


__all__ = [
    "DEFAULT_HISTORY_WINDOW_TOKEN_CAP",
    "EXTERNAL_LINK_OVERLAY_TOKEN_RESERVE",
    "DirectEvidenceExclusion",
    "DirectEvidenceRef",
    "EventTuple",
    "FINAL_PROMPT_TOKEN_CAP",
    "FactLane",
    "FactOrigin",
    "FactUnionEnvelope",
    "FrozenHistoryChunk",
    "FrozenSourceMembershipLike",
    "HydratedSourceHistory",
    "LANE_ORDER",
    "LANE_TOKEN_BUDGETS",
    "LaneAdmission",
    "LanePack",
    "MAX_PARENT_PROMPT_TOKENS",
    "MappedFactBatch",
    "OUTPUT_TOKEN_RESERVE",
    "ParentIdentity",
    "PostMapFactUnion",
    "RejectedMappedFact",
    "SourceHistoryFactUnionError",
    "SourceHistoryHydrationPlan",
    "SourceHistoryWindow",
    "SourceSelection",
    "UnionFact",
    "ValidatedMappedFact",
    "build_post_map_fact_union",
    "compact_fact_prompt_projection",
    "direct_evidence_projection_sha256",
    "hydrate_source_histories",
    "hydrate_source_history",
    "pack_fact_union_envelope",
    "plan_source_history_hydration",
    "validate_mapped_facts",
    "validate_mapper_completion",
    "window_source_history",
]
