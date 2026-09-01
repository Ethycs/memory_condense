"""Provider-free exhaustive scan guided by sealed query-expansion routes.

The parent query-expansion run already contains one global top-four partition
route for every materialized query.  This child never calls that provider or
retriever again.  It verifies the parent preflight/run/replay/runtime bytes,
combines the sealed routes by deterministic rank-weighted vote, and scans an
immutable in-memory catalog of the selected partitions.

Each frozen SQLite store is read with one ordered query per execution.  Rows
are cached by partition and reused by all ten questions bound to that store.
Question IDs remain provenance only: neither IDs nor known source prefixes are
accepted by the routing or scan functions.  Exact S0 dedup happens only after
the independent 2,400-token selection has been fixed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import EvidenceSpan, make_atom_id, quote_sha256
from memory_condense.persistence.db import TURN_SOURCE_ID_SQL, Database
from memory_condense.search.indexes.lexical import tokenize

from .artifacts import SealedArtifact, publish_sealed_json, read_sealed_json
from .contracts import (
    MatchedEvalContractError,
    StageDisposition,
    StageTrace,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .ledger import RuntimeLedgerEntry, build_runtime_ledger
from .partition_scan import _SENTENCE_RE, _bounded_excerpt, _partition
from .query_expansion import (
    ARM_LABEL as PARENT_ARM_LABEL,
    ENTIRE_STORE_SCOPE,
    PARTITION_ROUTE,
    ROUTING_RECEIPT_FORMAT,
    STAGE_ID as PARENT_STAGE_ID,
    FrozenSourceNamespace,
    LockedQueryExpansionContext,
    PartitionRoutingReceipt,
    QueryExpansionPopulation,
)
from .query_expansion_repack_v2 import (
    VerifiedQueryExpansionParent,
    verify_query_expansion_parent,
)


ARM_LABEL = "S0_PLUS_QUERY_GUIDED_EXHAUSTIVE_SCAN_V1"
PLAN_ID = "matched_s0_query_guided_exhaustive_scan_v1"
STAGE_ID = "query_guided_exhaustive_partition_additions_v1"
MECHANISM_ID = "sealed_query_vote_partition6_cached_exact_span_v1"
RENDERER_ID = "query_guided_exact_span_payload_v1"

RUN_FORMAT = "memory-condense-query-guided-exhaustive-scan-run-v1"
ROW_FORMAT = "memory-condense-query-guided-exhaustive-scan-row-v1"
CACHE_FORMAT = "memory-condense-query-guided-partition-cache-v1"
VOTE_FORMAT = "memory-condense-query-guided-partition-vote-v1"

RUN_NAME = "query-guided-scan-v1-run.json"
RUN_REPLAY_NAME = "query-guided-scan-v1-run-replay.json"
RUNTIME_LEDGER_NAME = "runtime-ledger.json"
RUNTIME_LEDGER_REPLAY_NAME = "runtime-ledger-replay.json"

RANK_WEIGHTS = (4, 3, 2, 1)
DEFAULT_PARTITION_SLOTS = 6
DEFAULT_TOKEN_CAP = 2_400
DEFAULT_MAX_EXCERPT_TOKENS = 96
DEFAULT_MAX_SPANS_PER_SOURCE = 2
DEFAULT_COVERAGE_NUMERATOR = 4
DEFAULT_COVERAGE_DENOMINATOR = 5


class QueryGuidedScanError(MatchedEvalContractError):
    """Raised when a sealed parent, store cache, or scan lifecycle changes."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise QueryGuidedScanError(message)


def _ordered_ids(value: object, label: str) -> tuple[str, ...]:
    _require(type(value) is list, f"{label} must be an exact array")
    rows = tuple(value)
    _require(
        all(type(row) is str and row and row.strip() == row for row in rows),
        f"{label} must contain exact non-empty IDs",
    )
    _require(len(rows) == len(set(rows)), f"{label} must be ordered and unique")
    return rows


@dataclass(frozen=True, slots=True)
class QueryGuidedScanBudget:
    """A non-borrowing route, span, and evidence budget."""

    partition_slots: int = DEFAULT_PARTITION_SLOTS
    candidate_token_cap: int = DEFAULT_TOKEN_CAP
    max_excerpt_tokens: int = DEFAULT_MAX_EXCERPT_TOKENS
    max_spans_per_source: int = DEFAULT_MAX_SPANS_PER_SOURCE
    coverage_reserve_numerator: int = DEFAULT_COVERAGE_NUMERATOR
    coverage_reserve_denominator: int = DEFAULT_COVERAGE_DENOMINATOR

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            _require(
                type(value) is int and value > 0,
                f"{name} must be a positive exact integer",
            )
        _require(
            self.partition_slots == DEFAULT_PARTITION_SLOTS,
            "the locked scan requires exactly six partitions",
        )
        _require(
            self.candidate_token_cap == DEFAULT_TOKEN_CAP,
            "the locked scan requires its separate 2,400-token cap",
        )
        _require(
            self.max_spans_per_source == DEFAULT_MAX_SPANS_PER_SOURCE,
            "the locked scan permits at most two spans per source",
        )
        _require(
            self.coverage_reserve_numerator
            < self.coverage_reserve_denominator,
            "coverage reserve must leave a positive enrichment slice",
        )

    @property
    def coverage_token_reserve(self) -> int:
        return (
            self.candidate_token_cap * self.coverage_reserve_numerator
            // self.coverage_reserve_denominator
        )

    def projection(self) -> dict[str, int]:
        return {
            "candidate_token_cap": self.candidate_token_cap,
            "coverage_reserve_denominator": self.coverage_reserve_denominator,
            "coverage_reserve_numerator": self.coverage_reserve_numerator,
            "coverage_token_reserve": self.coverage_token_reserve,
            "max_excerpt_tokens": self.max_excerpt_tokens,
            "max_spans_per_source": self.max_spans_per_source,
            "partition_slots": self.partition_slots,
        }

    @property
    def budget_id(self) -> str:
        return identity_sha256(
            {
                "arm_label": ARM_LABEL,
                "non_borrowing": True,
                "rank_weights": list(RANK_WEIGHTS),
                **self.projection(),
            }
        )


@dataclass(frozen=True, slots=True)
class PartitionVote:
    partition_id: str
    vote_score: int
    receipt_hit_count: int
    best_parent_rank: int
    first_observation_index: int
    inventory_rank: int

    def __post_init__(self) -> None:
        require_text(self.partition_id, "partition vote ID")
        for value, label in (
            (self.vote_score, "partition vote score"),
            (self.receipt_hit_count, "partition receipt-hit count"),
            (self.best_parent_rank, "partition best parent rank"),
            (self.first_observation_index, "partition first observation"),
            (self.inventory_rank, "partition inventory rank"),
        ):
            _require(type(value) is int and value >= 0, f"{label} changed")

    def projection(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PartitionVotePlan:
    selected_partitions: tuple[str, ...]
    ranking: tuple[PartitionVote, ...]
    parent_receipt_sha256s: tuple[str, ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        _require(
            type(self.selected_partitions) is tuple
            and len(self.selected_partitions) == DEFAULT_PARTITION_SLOTS
            and len(set(self.selected_partitions)) == len(self.selected_partitions),
            "partition vote must select exactly six unique partitions",
        )
        _require(
            type(self.ranking) is tuple
            and tuple(row.partition_id for row in self.ranking[:6])
            == self.selected_partitions,
            "partition vote ranking changed",
        )
        _require(
            type(self.parent_receipt_sha256s) is tuple
            and self.parent_receipt_sha256s,
            "partition vote requires parent route receipts",
        )
        for value in self.parent_receipt_sha256s:
            require_sha256(value, "parent route receipt")
        require_sha256(self.receipt_sha256, "partition vote receipt")
        _require(
            self.receipt_sha256 == identity_sha256(self._body()),
            "partition vote receipt changed",
        )

    def _body(self) -> dict[str, Any]:
        return {
            "fill_policy": "ranked_votes_then_frozen_inventory_order",
            "format": VOTE_FORMAT,
            "known_history_filter_used": False,
            "parent_receipt_sha256s": list(self.parent_receipt_sha256s),
            "partition_slots": DEFAULT_PARTITION_SLOTS,
            "question_id_filter_used": False,
            "rank_weights": list(RANK_WEIGHTS),
            "ranking": [row.projection() for row in self.ranking],
            "selected_partitions": list(self.selected_partitions),
            "source_prefix_filter_used": False,
        }

    def projection(self) -> dict[str, Any]:
        return {**self._body(), "receipt_sha256": self.receipt_sha256}


def _project_parent_route(
    raw: Mapping[str, Any],
    *,
    query: str,
    namespace: FrozenSourceNamespace,
) -> PartitionRoutingReceipt:
    """Rehydrate and byte-check one sealed parent routing receipt."""

    selected = _ordered_ids(raw.get("selected_partitions"), "route partitions")
    receipt = PartitionRoutingReceipt(
        query_sha256=str(raw.get("query_sha256", "")),
        namespace_id=str(raw.get("namespace_id", "")),
        selected_partitions=selected,
        partition_inventory_total=raw.get("partition_inventory_total"),
        routed_source_count=raw.get("routed_source_count"),
        active_partition_scan_status=str(
            raw.get("active_partition_scan_status", "")
        ),
        active_partition_scan_contract=str(
            raw.get("active_partition_scan_contract", "")
        ),
        active_partition_exhaustive=raw.get("active_partition_exhaustive"),
        receipt_sha256=str(raw.get("receipt_sha256", "")),
    )
    _require(receipt.projection() == dict(raw), "parent route projection changed")
    _require(
        receipt.query_sha256 == quote_sha256(query)
        and receipt.namespace_id == namespace.namespace_id,
        "parent route changed its query or namespace binding",
    )
    _require(
        receipt.partition_inventory_total == len(namespace.partition_ids)
        and set(receipt.selected_partitions) <= set(namespace.partition_ids),
        "parent route escaped the frozen partition inventory",
    )
    return receipt


def aggregate_partition_votes(
    materialized_queries: Sequence[str],
    raw_routing_receipts: Sequence[Mapping[str, Any]],
    *,
    namespace: FrozenSourceNamespace,
    partition_slots: int = DEFAULT_PARTITION_SLOTS,
) -> PartitionVotePlan:
    """Combine sealed top-four routes without a question/source identifier."""

    _require(
        partition_slots == DEFAULT_PARTITION_SLOTS,
        "locked partition vote slot count changed",
    )
    queries = tuple(materialized_queries)
    raws = tuple(raw_routing_receipts)
    _require(
        queries
        and len(queries) == len(raws)
        and len(namespace.partition_ids) >= partition_slots,
        "partition vote population changed",
    )
    _require(
        all(type(value) is str and value and value.strip() == value for value in queries),
        "materialized queries must be exact non-empty text",
    )
    receipts = tuple(
        _project_parent_route(raw, query=query, namespace=namespace)
        for query, raw in zip(queries, raws, strict=True)
    )
    inventory_rank = {
        partition: index for index, partition in enumerate(namespace.partition_ids)
    }
    scores = {partition: 0 for partition in namespace.partition_ids}
    hits = {partition: 0 for partition in namespace.partition_ids}
    best = {partition: len(RANK_WEIGHTS) for partition in namespace.partition_ids}
    first = {
        partition: len(receipts) * len(RANK_WEIGHTS) + inventory_rank[partition]
        for partition in namespace.partition_ids
    }
    for receipt_index, receipt in enumerate(receipts):
        for rank, partition in enumerate(receipt.selected_partitions):
            scores[partition] += RANK_WEIGHTS[rank]
            hits[partition] += 1
            best[partition] = min(best[partition], rank)
            first[partition] = min(
                first[partition], receipt_index * len(RANK_WEIGHTS) + rank
            )
    ranking = tuple(
        PartitionVote(
            partition_id=partition,
            vote_score=scores[partition],
            receipt_hit_count=hits[partition],
            best_parent_rank=best[partition],
            first_observation_index=first[partition],
            inventory_rank=inventory_rank[partition],
        )
        for partition in sorted(
            namespace.partition_ids,
            key=lambda value: (
                -scores[value],
                -hits[value],
                best[value],
                first[value],
                inventory_rank[value],
                value,
            ),
        )
    )
    body = {
        "fill_policy": "ranked_votes_then_frozen_inventory_order",
        "format": VOTE_FORMAT,
        "known_history_filter_used": False,
        "parent_receipt_sha256s": [row.receipt_sha256 for row in receipts],
        "partition_slots": partition_slots,
        "question_id_filter_used": False,
        "rank_weights": list(RANK_WEIGHTS),
        "ranking": [row.projection() for row in ranking],
        "selected_partitions": [
            row.partition_id for row in ranking[:partition_slots]
        ],
        "source_prefix_filter_used": False,
    }
    assert_gold_blind(body, path="query_guided_partition_vote")
    return PartitionVotePlan(
        selected_partitions=tuple(body["selected_partitions"]),
        ranking=ranking,
        parent_receipt_sha256s=tuple(body["parent_receipt_sha256s"]),
        receipt_sha256=identity_sha256(body),
    )


@dataclass(frozen=True, slots=True)
class CachedSentenceWindow:
    start_char: int
    end_char: int
    text_sha256: str
    token_count: int
    terms: frozenset[str]

    def __post_init__(self) -> None:
        _require(
            type(self.start_char) is int
            and type(self.end_char) is int
            and 0 <= self.start_char < self.end_char,
            "cached sentence coordinates changed",
        )
        require_sha256(self.text_sha256, "cached sentence text")
        _require(
            type(self.token_count) is int
            and self.token_count > 0
            and type(self.terms) is frozenset,
            "cached sentence surface changed",
        )


def _sentence_windows(text: str) -> tuple[CachedSentenceWindow, ...]:
    coordinates: list[tuple[int, int]] = []
    for match in _SENTENCE_RE.finditer(text):
        start, end = match.span()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            coordinates.append((start, end))
    if not coordinates:
        coordinates.append((0, len(text)))
    return tuple(
        CachedSentenceWindow(
            start_char=start,
            end_char=end,
            text_sha256=quote_sha256(text[start:end]),
            token_count=count_tokens(text[start:end]),
            terms=frozenset(tokenize(text[start:end])),
        )
        for start, end in coordinates
    )


@dataclass(frozen=True, slots=True)
class CachedContentRow:
    namespace_id: str
    partition_id: str
    chunk_id: str
    turn_id: str
    source_id: str
    role: str
    created_at: str
    ordinal: int
    turn_start_char: int
    turn_end_char: int
    text: str
    text_sha256: str
    token_count: int
    sentence_windows: tuple[CachedSentenceWindow, ...]

    def __post_init__(self) -> None:
        for value, label in (
            (self.partition_id, "cached partition"),
            (self.chunk_id, "cached chunk"),
            (self.turn_id, "cached turn"),
            (self.source_id, "cached source"),
            (self.role, "cached role"),
            (self.created_at, "cached timestamp"),
            (self.text, "cached text"),
        ):
            require_text(value, label)
        require_sha256(self.namespace_id, "cached namespace")
        require_sha256(self.text_sha256, "cached text")
        _require(_partition(self.source_id) == self.partition_id, "cached partition changed")
        _require(
            type(self.ordinal) is int
            and type(self.turn_start_char) is int
            and type(self.turn_end_char) is int
            and self.ordinal >= 0
            and 0 <= self.turn_start_char < self.turn_end_char,
            "cached coordinates changed",
        )
        _require(
            self.turn_end_char - self.turn_start_char == len(self.text),
            "cached chunk/turn character coordinates changed",
        )
        _require(
            self.text_sha256 == quote_sha256(self.text)
            and self.token_count == count_tokens(self.text),
            "cached content bytes changed",
        )
        _require(
            type(self.sentence_windows) is tuple
            and self.sentence_windows == _sentence_windows(self.text),
            "cached sentence windows changed",
        )

    def receipt_projection(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "created_at": self.created_at,
            "ordinal": self.ordinal,
            "partition_id": self.partition_id,
            "role": self.role,
            "source_id": self.source_id,
            "text_sha256": self.text_sha256,
            "token_count": self.token_count,
            "turn_end_char": self.turn_end_char,
            "turn_id": self.turn_id,
            "turn_start_char": self.turn_start_char,
        }


@dataclass(frozen=True, slots=True)
class NamespacePartitionCache:
    namespace_id: str
    source_database_sha256: str
    source_store_receipt_sha256: str
    physical_store_row_count: int
    metadata_row_count: int
    rows_by_partition: Mapping[str, tuple[CachedContentRow, ...]]
    cache_receipt_sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.namespace_id, "cache namespace")
        require_sha256(self.source_database_sha256, "cache database")
        require_sha256(self.source_store_receipt_sha256, "cache store receipt")
        require_sha256(self.cache_receipt_sha256, "cache receipt")
        _require(
            type(self.physical_store_row_count) is int
            and type(self.metadata_row_count) is int
            and self.physical_store_row_count >= self.metadata_row_count >= 0,
            "cache scan counts changed",
        )
        _require(
            isinstance(self.rows_by_partition, Mapping),
            "partition cache must be an immutable mapping",
        )
        rows = tuple(
            row for values in self.rows_by_partition.values() for row in values
        )
        _require(
            len({row.chunk_id for row in rows}) == len(rows)
            and all(row.namespace_id == self.namespace_id for row in rows),
            "partition cache row inventory changed",
        )
        _require(
            self.cache_receipt_sha256 == identity_sha256(self._body()),
            "partition cache receipt changed",
        )

    @property
    def content_row_count(self) -> int:
        return sum(len(rows) for rows in self.rows_by_partition.values())

    def _body(self) -> dict[str, Any]:
        partitions = []
        for partition_id, rows in self.rows_by_partition.items():
            row_receipts = [row.receipt_projection() for row in rows]
            partitions.append(
                {
                    "content_row_count": len(rows),
                    "content_rows_sha256": identity_sha256(row_receipts),
                    "partition_id": partition_id,
                }
            )
        return {
            "cache_immutable": True,
            "content_row_count": self.content_row_count,
            "database_read_passes": 1,
            "format": CACHE_FORMAT,
            "metadata_row_count": self.metadata_row_count,
            "namespace_id": self.namespace_id,
            "partitions": partitions,
            "physical_store_row_count": self.physical_store_row_count,
            "source_database_sha256": self.source_database_sha256,
            "source_store_receipt_sha256": self.source_store_receipt_sha256,
        }

    def projection(self) -> dict[str, Any]:
        return {**self._body(), "cache_receipt_sha256": self.cache_receipt_sha256}


def cache_namespace_partitions(
    database: Database,
    namespace: FrozenSourceNamespace,
    *,
    source_database_sha256: str,
    source_store_receipt_sha256: str,
) -> NamespacePartitionCache:
    """Read a frozen store once and cache every exact content row by partition."""

    _require(
        type(database) is Database and database.read_only,
        "query-guided caches require an exact read-only Database",
    )
    require_sha256(source_database_sha256, "cache database")
    require_sha256(source_store_receipt_sha256, "cache store receipt")
    _require(
        source_store_receipt_sha256 == namespace.combined_store_receipt_sha256,
        "cache store receipt differs from its sealed namespace",
    )
    mutable: dict[str, list[CachedContentRow]] = {
        partition: [] for partition in namespace.partition_ids
    }
    observed: set[str] = set()
    physical_count = 0
    metadata_count = 0
    query = (
        "SELECT c.chunk_id, c.turn_id, c.text, c.start_char, c.end_char, "
        "c.token_count, "
        + TURN_SOURCE_ID_SQL
        + ", t.role, t.created_at, t.ordinal "
        "FROM chunks AS c JOIN turns AS t ON t.turn_id = c.turn_id "
        "ORDER BY t.ordinal, c.rowid"
    )
    # This is deliberately the only SQLite SELECT in the cache builder.
    for raw in database.execute(query):
        physical_count += 1
        chunk_id = str(raw[0])
        source_id = str(raw[6])
        _require(chunk_id not in observed, "store chunk identity repeated")
        observed.add(chunk_id)
        _require(
            namespace.chunk_to_source.get(chunk_id) == source_id,
            "store row escaped its sealed source membership",
        )
        if chunk_id in namespace.metadata_chunk_ids:
            metadata_count += 1
            continue
        partition_id = _partition(source_id)
        _require(partition_id in mutable, "content row escaped partition inventory")
        mutable[partition_id].append(
            CachedContentRow(
                namespace_id=namespace.namespace_id,
                partition_id=partition_id,
                chunk_id=chunk_id,
                turn_id=str(raw[1]),
                source_id=source_id,
                role=str(raw[7]),
                created_at=str(raw[8]),
                ordinal=int(raw[9]),
                turn_start_char=int(raw[3]),
                turn_end_char=int(raw[4]),
                text=str(raw[2]),
                text_sha256=quote_sha256(str(raw[2])),
                token_count=int(raw[5]),
                sentence_windows=_sentence_windows(str(raw[2])),
            )
        )
    _require(
        observed == set(namespace.chunk_to_source),
        "store chunk inventory changed from the sealed namespace",
    )
    frozen = MappingProxyType(
        {partition: tuple(rows) for partition, rows in mutable.items()}
    )
    partitions = []
    for partition_id, rows in frozen.items():
        row_receipts = [row.receipt_projection() for row in rows]
        partitions.append(
            {
                "content_row_count": len(rows),
                "content_rows_sha256": identity_sha256(row_receipts),
                "partition_id": partition_id,
            }
        )
    body = {
        "cache_immutable": True,
        "content_row_count": sum(len(rows) for rows in frozen.values()),
        "database_read_passes": 1,
        "format": CACHE_FORMAT,
        "metadata_row_count": metadata_count,
        "namespace_id": namespace.namespace_id,
        "partitions": partitions,
        "physical_store_row_count": physical_count,
        "source_database_sha256": source_database_sha256,
        "source_store_receipt_sha256": source_store_receipt_sha256,
    }
    return NamespacePartitionCache(
        namespace_id=namespace.namespace_id,
        source_database_sha256=source_database_sha256,
        source_store_receipt_sha256=source_store_receipt_sha256,
        physical_store_row_count=physical_count,
        metadata_row_count=metadata_count,
        rows_by_partition=frozen,
        cache_receipt_sha256=identity_sha256(body),
    )


@dataclass(frozen=True, slots=True)
class QueryGuidedCandidate:
    evidence_id: str
    atom_id: str
    source_id: str
    partition_id: str
    text: str
    token_count: int
    span: EvidenceSpan
    best_query_index: int
    best_query_sha256: str
    overlap_term_count: int
    matching_query_count: int
    aggregate_overlap_count: int
    query_coverage: float
    excerpt_density: float
    exact_phrase_match: bool
    source_rank: int
    span_rank: int

    def __post_init__(self) -> None:
        require_sha256(self.evidence_id, "query-guided evidence")
        require_text(self.atom_id, "query-guided atom")
        require_sha256(self.best_query_sha256, "best query")
        require_text(self.source_id, "query-guided source")
        require_text(self.partition_id, "query-guided partition")
        require_text(self.text, "query-guided text")
        _require(_partition(self.source_id) == self.partition_id, "candidate partition changed")
        _require(self.span.source_id == self.source_id, "candidate span source changed")
        _require(make_atom_id(self.span) == self.atom_id, "candidate atom changed")
        _require(
            quote_sha256(self.text) == self.span.quote_sha256
            and count_tokens(self.text) == self.token_count,
            "candidate exact text changed",
        )
        _require(
            self.evidence_id
            == identity_sha256({"atom_id": self.atom_id, "mechanism_id": MECHANISM_ID}),
            "candidate evidence identity changed",
        )
        for value, label in (
            (self.best_query_index, "best query index"),
            (self.overlap_term_count, "overlap count"),
            (self.matching_query_count, "matching query count"),
            (self.aggregate_overlap_count, "aggregate overlap"),
            (self.source_rank, "source rank"),
            (self.span_rank, "span rank"),
        ):
            _require(type(value) is int and value >= 0, f"{label} changed")
        _require(
            self.span_rank < DEFAULT_MAX_SPANS_PER_SOURCE,
            "candidate span rank exceeds the locked bound",
        )
        _require(
            all(
                math.isfinite(value) and 0.0 <= value <= 1.0
                for value in (self.query_coverage, self.excerpt_density)
            ),
            "candidate surface scores changed",
        )
        _require(type(self.exact_phrase_match) is bool, "phrase-match flag changed")

    def projection(self) -> dict[str, Any]:
        return {
            "aggregate_overlap_count": self.aggregate_overlap_count,
            "atom_id": self.atom_id,
            "best_query_index": self.best_query_index,
            "best_query_sha256": self.best_query_sha256,
            "evidence_id": self.evidence_id,
            "exact_phrase_match": self.exact_phrase_match,
            "excerpt_density": self.excerpt_density,
            "matching_query_count": self.matching_query_count,
            "overlap_term_count": self.overlap_term_count,
            "partition_id": self.partition_id,
            "query_coverage": self.query_coverage,
            "source_id": self.source_id,
            "source_rank": self.source_rank,
            "span": self.span.identity_payload(),
            "span_rank": self.span_rank,
            "text": self.text,
            "text_sha256": quote_sha256(self.text),
            "token_count": self.token_count,
        }


@dataclass(frozen=True, slots=True)
class _SpanDraft:
    row: CachedContentRow
    start_char: int
    end_char: int
    excerpt: str
    best_query_index: int
    best_query_sha256: str
    overlap_term_count: int
    matching_query_count: int
    aggregate_overlap_count: int
    query_coverage: float
    excerpt_density: float
    exact_phrase_match: bool

    @property
    def score(self) -> tuple[Any, ...]:
        return (
            self.query_coverage,
            self.overlap_term_count,
            self.matching_query_count,
            self.aggregate_overlap_count,
            self.excerpt_density,
            int(self.exact_phrase_match),
            -self.row.ordinal,
            self.row.chunk_id,
        )


def _score_content_row(
    row: CachedContentRow,
    query_surfaces: Sequence[str],
    query_term_sets: Sequence[frozenset[str]],
    *,
    max_excerpt_tokens: int,
) -> _SpanDraft:
    scored: list[tuple[tuple[Any, ...], tuple[Any, ...]]] = []
    matching = 0
    aggregate = 0
    folded_text = row.text.casefold()
    for query_index, (query, terms) in enumerate(
        zip(query_surfaces, query_term_sets, strict=True)
    ):
        window = max(
            row.sentence_windows,
            key=lambda value: (
                len(terms & value.terms),
                len(terms & value.terms) / max(len(value.terms), 1),
                -value.start_char,
            ),
        )
        overlap = len(terms & window.terms)
        matching += bool(overlap)
        aggregate += overlap
        coverage = overlap / max(len(terms), 1)
        density = overlap / max(len(window.terms), 1)
        exact = query.casefold() in folded_text
        key = (coverage, overlap, density, int(exact), -query_index)
        scored.append(
            (
                key,
                (
                    query_index,
                    quote_sha256(query),
                    window,
                    overlap,
                    coverage,
                    density,
                    exact,
                ),
            )
    )
    _require(scored, "query-guided scoring requires query surfaces")
    _key, best = max(scored, key=lambda value: value[0])
    window = best[2]
    _require(type(window) is CachedSentenceWindow, "best sentence window changed")
    if window.token_count <= max_excerpt_tokens:
        start = window.start_char
        end = window.end_char
        excerpt = row.text[start:end]
    else:
        start, end, excerpt = _bounded_excerpt(
            row.text,
            query_term_sets[int(best[0])],
            max_tokens=max_excerpt_tokens,
        )
    excerpt_terms = frozenset(tokenize(excerpt))
    best_terms = query_term_sets[int(best[0])]
    best_overlap = len(best_terms & excerpt_terms)
    return _SpanDraft(
        row=row,
        best_query_index=int(best[0]),
        best_query_sha256=str(best[1]),
        start_char=start,
        end_char=end,
        excerpt=excerpt,
        overlap_term_count=best_overlap,
        query_coverage=best_overlap / max(len(best_terms), 1),
        excerpt_density=best_overlap / max(len(excerpt_terms), 1),
        exact_phrase_match=bool(best[6]),
        matching_query_count=matching,
        aggregate_overlap_count=aggregate,
    )


def score_query_guided_candidates(
    cache: NamespacePartitionCache,
    *,
    selected_partitions: Sequence[str],
    query_surfaces: Sequence[str],
    budget: QueryGuidedScanBudget = QueryGuidedScanBudget(),
) -> tuple[QueryGuidedCandidate, ...]:
    """Score every cached content row and keep two exact spans per source."""

    selected = tuple(selected_partitions)
    _require(
        len(selected) == budget.partition_slots
        and len(set(selected)) == len(selected)
        and set(selected) <= set(cache.rows_by_partition),
        "scored partition selection changed",
    )
    surfaces = tuple(query_surfaces)
    _require(
        surfaces
        and all(type(value) is str and value and value.strip() == value for value in surfaces),
        "query surfaces must be exact non-empty text",
    )
    term_sets = tuple(frozenset(tokenize(value)) for value in surfaces)
    _require(any(term_sets), "all query surfaces lost their lexical terms")
    by_source: dict[str, list[_SpanDraft]] = {}
    for partition in selected:
        for row in cache.rows_by_partition[partition]:
            draft = _score_content_row(
                row,
                surfaces,
                term_sets,
                max_excerpt_tokens=budget.max_excerpt_tokens,
            )
            by_source.setdefault(row.source_id, []).append(draft)

    retained: dict[str, tuple[_SpanDraft, ...]] = {}
    for source_id, rows in by_source.items():
        ordered = sorted(rows, key=lambda value: value.score, reverse=True)
        unique: list[_SpanDraft] = []
        seen: set[tuple[str, int, int]] = set()
        for row in ordered:
            coordinate = (row.row.chunk_id, row.start_char, row.end_char)
            if coordinate in seen:
                continue
            seen.add(coordinate)
            unique.append(row)
            if len(unique) == budget.max_spans_per_source:
                break
        retained[source_id] = tuple(unique)

    partition_sources: dict[str, tuple[str, ...]] = {}
    source_ranks: dict[str, int] = {}
    for partition in selected:
        sources = [value for value in retained if _partition(value) == partition]
        sources.sort(key=lambda value: retained[value][0].score, reverse=True)
        partition_sources[partition] = tuple(sources)
        source_ranks.update({source: rank for rank, source in enumerate(sources)})

    def candidate(source_id: str, span_rank: int) -> QueryGuidedCandidate:
        draft = retained[source_id][span_rank]
        span = EvidenceSpan(
            chunk_id=draft.row.chunk_id,
            start_char=draft.start_char,
            end_char=draft.end_char,
            quote_sha256=quote_sha256(draft.excerpt),
            ordinal=draft.row.ordinal,
            source_id=source_id,
            turn_start_char=draft.row.turn_start_char,
            turn_id=draft.row.turn_id,
            role=draft.row.role,
            created_at=draft.row.created_at,
        )
        atom_id = make_atom_id(span)
        return QueryGuidedCandidate(
            evidence_id=identity_sha256(
                {"atom_id": atom_id, "mechanism_id": MECHANISM_ID}
            ),
            atom_id=atom_id,
            source_id=source_id,
            partition_id=_partition(source_id),
            text=draft.excerpt,
            token_count=count_tokens(draft.excerpt),
            span=span,
            best_query_index=draft.best_query_index,
            best_query_sha256=draft.best_query_sha256,
            overlap_term_count=draft.overlap_term_count,
            matching_query_count=draft.matching_query_count,
            aggregate_overlap_count=draft.aggregate_overlap_count,
            query_coverage=draft.query_coverage,
            excerpt_density=draft.excerpt_density,
            exact_phrase_match=draft.exact_phrase_match,
            source_rank=source_ranks[source_id],
            span_rank=span_rank,
        )

    # Primary candidates are partition-round-robin before every secondary.
    # This ordering makes every later bounded selection an auditable ordered
    # subsequence of the complete candidate population.
    output: list[QueryGuidedCandidate] = []
    for span_rank in range(budget.max_spans_per_source):
        width = max(
            (len(partition_sources[value]) for value in selected), default=0
        )
        for source_rank in range(width):
            for partition in selected:
                sources = partition_sources[partition]
                if source_rank >= len(sources):
                    continue
                source_id = sources[source_rank]
                if len(retained[source_id]) > span_rank:
                    output.append(candidate(source_id, span_rank))
    ids = tuple(row.evidence_id for row in output)
    _require(len(ids) == len(set(ids)), "query-guided candidate identity repeated")
    return tuple(output)


@dataclass(frozen=True, slots=True)
class BalancedSelection:
    selected_ids: tuple[str, ...]
    phase_by_id: Mapping[str, str]
    selected_token_count: int
    coverage_tokens: int
    enrichment_tokens: int
    reclaim_tokens: int


def select_balanced_candidates(
    candidates: Sequence[QueryGuidedCandidate],
    *,
    budget: QueryGuidedScanBudget = QueryGuidedScanBudget(),
) -> BalancedSelection:
    """Partition-round-robin source coverage, then optional second spans."""

    rows = tuple(candidates)
    ids = tuple(row.evidence_id for row in rows)
    _require(len(ids) == len(set(ids)), "balanced selection candidates repeated")
    phase: dict[str, str] = {}
    selected: set[str] = set()
    coverage_used = 0
    primaries = tuple(row for row in rows if row.span_rank == 0)
    secondaries = tuple(row for row in rows if row.span_rank > 0)
    for row in primaries:
        if coverage_used + row.token_count <= budget.coverage_token_reserve:
            selected.add(row.evidence_id)
            phase[row.evidence_id] = "partition_source_coverage"
            coverage_used += row.token_count

    covered_sources = {
        row.source_id for row in primaries if row.evidence_id in selected
    }
    total = coverage_used
    enrichment_used = 0
    for row in secondaries:
        if row.source_id not in covered_sources:
            continue
        if total + row.token_count <= budget.candidate_token_cap:
            selected.add(row.evidence_id)
            phase[row.evidence_id] = "optional_second_span"
            enrichment_used += row.token_count
            total += row.token_count

    reclaim_used = 0
    # Reclaim unused space in candidate order.  Primaries appear before
    # secondaries, so uncovered sources remain preferred.
    for row in rows:
        if row.evidence_id in selected:
            continue
        if row.span_rank > 0 and row.source_id not in covered_sources:
            continue
        if total + row.token_count <= budget.candidate_token_cap:
            selected.add(row.evidence_id)
            phase[row.evidence_id] = "ordered_reclaim"
            reclaim_used += row.token_count
            total += row.token_count
            if row.span_rank == 0:
                covered_sources.add(row.source_id)
    ordered = tuple(row.evidence_id for row in rows if row.evidence_id in selected)
    _require(
        total <= budget.candidate_token_cap
        and total == sum(row.token_count for row in rows if row.evidence_id in selected),
        "balanced selection token accounting changed",
    )
    return BalancedSelection(
        selected_ids=ordered,
        phase_by_id=MappingProxyType(phase),
        selected_token_count=total,
        coverage_tokens=coverage_used,
        enrichment_tokens=enrichment_used,
        reclaim_tokens=reclaim_used,
    )


def _s0_coordinates(prompt: Any) -> dict[tuple[str, str], str]:
    coordinates: dict[tuple[str, str], str] = {}
    for evidence in prompt.source.packet.protected_evidence:
        coordinates.setdefault(
            (evidence.source_id, quote_sha256(evidence.text)), evidence.evidence_id
        )
    return coordinates


def _construct_row(
    prompt: Any,
    raw_parent: Mapping[str, Any],
    cache: NamespacePartitionCache,
    *,
    budget: QueryGuidedScanBudget,
) -> dict[str, Any]:
    materialized = tuple(raw_parent.get("materialized_queries", ()))
    raw_routes = raw_parent.get("routing_receipts")
    _require(
        type(raw_parent.get("receipt_sha256")) is str
        and type(raw_routes) is list
        and all(type(row) is dict for row in raw_routes),
        "parent row routes changed",
    )
    vote = aggregate_partition_votes(
        materialized,
        tuple(raw_routes),
        namespace=prompt.namespace,
        partition_slots=budget.partition_slots,
    )
    query_surfaces = (*materialized, prompt.source.packet.dated_question)
    candidates = score_query_guided_candidates(
        cache,
        selected_partitions=vote.selected_partitions,
        query_surfaces=query_surfaces,
        budget=budget,
    )
    selection = select_balanced_candidates(candidates, budget=budget)
    by_id = {row.evidence_id: row for row in candidates}
    s0 = _s0_coordinates(prompt)
    aliases: list[tuple[str, str]] = []
    admitted: list[str] = []
    for evidence_id in selection.selected_ids:
        candidate = by_id[evidence_id]
        alias = s0.get((candidate.source_id, quote_sha256(candidate.text)))
        if alias is None:
            admitted.append(evidence_id)
        else:
            aliases.append((evidence_id, alias))
    excluded = tuple(value for value, _alias in aliases)
    admitted_ids = tuple(admitted)
    tokens_used = sum(by_id[value].token_count for value in admitted_ids)
    trace = StageTrace(
        candidate_ids=tuple(by_id),
        selected_before_dedup_ids=selection.selected_ids,
        dedup_excluded_ids=excluded,
        admitted_ids=admitted_ids,
        token_cap=budget.candidate_token_cap,
        tokens_used=tokens_used,
        disposition=(StageDisposition.ADDED if admitted_ids else StageDisposition.NO_OP),
        reason=(
            "query_guided_exact_spans_admitted"
            if admitted_ids
            else "selection_empty_or_exactly_deduped"
        ),
    )
    selected_by_partition = {
        partition: sum(
            by_id[value].partition_id == partition for value in selection.selected_ids
        )
        for partition in vote.selected_partitions
    }
    admitted_by_partition = {
        partition: sum(
            by_id[value].partition_id == partition for value in admitted_ids
        )
        for partition in vote.selected_partitions
    }
    unsigned: dict[str, Any] = {
        "admitted_candidate_ids": list(admitted_ids),
        "admitted_candidates": [by_id[value].projection() for value in admitted_ids],
        "budget_id": budget.budget_id,
        "cache_receipt_sha256": cache.cache_receipt_sha256,
        "candidate_ids": list(by_id),
        "candidates": [row.projection() for row in candidates],
        "candidate_token_cap": budget.candidate_token_cap,
        "coverage_tokens_used": selection.coverage_tokens,
        "dated_question_sha256": prompt.source.packet.dated_question_sha256,
        "dedup_alias_bindings": [list(value) for value in aliases],
        "dedup_excluded_candidate_ids": list(excluded),
        "dedup_timing": "after_bounded_selection",
        "disposition": trace.disposition.value,
        "enrichment_tokens_used": selection.enrichment_tokens,
        "format": ROW_FORMAT,
        "gold_loaded": False,
        "known_history_filter_used": False,
        "logical_scanned_content_row_count": sum(
            len(cache.rows_by_partition[value]) for value in vote.selected_partitions
        ),
        "materialized_queries": list(materialized),
        "namespace_id": prompt.namespace.namespace_id,
        "not_admitted_candidate_ids": [],
        "ordinal": prompt.source.ordinal,
        "parent_packet_id": prompt.source.packet.packet_id,
        "parent_query_expansion_row_receipt_sha256": raw_parent["receipt_sha256"],
        "parent_routing_receipts": list(raw_routes),
        "parent_routing_receipts_sha256": identity_sha256(raw_routes),
        "partition_admitted_counts": admitted_by_partition,
        "partition_selected_counts": selected_by_partition,
        "partition_vote": vote.projection(),
        "provider_calls": 0,
        "question_id": prompt.source.packet.question_id,
        "question_id_filter_used": False,
        "question_sha256": prompt.source.packet.question_sha256,
        "query_surface_sha256s": [quote_sha256(value) for value in query_surfaces],
        "reason": trace.reason,
        "reclaim_tokens_used": selection.reclaim_tokens,
        "retained_transformer_token_state_bytes": 0,
        "selected_before_dedup_candidate_ids": list(selection.selected_ids),
        "selected_before_dedup_token_count": selection.selected_token_count,
        "selection_phase_by_candidate_id": {
            value: selection.phase_by_id[value] for value in selection.selected_ids
        },
        "source_prefix_filter_used": False,
        "stage_id": STAGE_ID,
        "tokens_used": tokens_used,
    }
    assert_gold_blind(unsigned, path="query_guided_scan_row")
    return {**unsigned, "receipt_sha256": identity_sha256(unsigned)}


def _build_payload(
    context: LockedQueryExpansionContext,
    parent: VerifiedQueryExpansionParent,
    *,
    budget: QueryGuidedScanBudget,
) -> dict[str, Any]:
    population = context.population
    _require(
        parent.population.preflight_projection() == population.preflight_projection(),
        "store-backed population differs from the verified parent",
    )
    raw_parent_rows = parent.run.payload.get("questions")
    _require(type(raw_parent_rows) is list, "parent question rows changed")
    bound: dict[str, list[tuple[Any, Mapping[str, Any]]]] = {}
    for prompt, raw in zip(population.rows, raw_parent_rows, strict=True):
        _require(type(raw) is dict, "parent question row changed")
        bound.setdefault(prompt.namespace.namespace_id, []).append((prompt, raw))

    output_rows: list[dict[str, Any] | None] = [None] * len(population.rows)
    cache_receipts: list[dict[str, Any]] = []
    for namespace in population.namespaces:
        namespace_id = namespace.namespace_id
        store = context.store_dirs_by_namespace[namespace_id]
        with Database(store / "memory.db", read_only=True) as database:
            cache = cache_namespace_partitions(
                database,
                namespace,
                source_database_sha256=context.database_sha256_by_namespace[
                    namespace_id
                ],
                source_store_receipt_sha256=(
                    namespace.combined_store_receipt_sha256
                ),
            )
        cache_receipts.append(cache.projection())
        for prompt, raw in bound.get(namespace_id, ()):
            output_rows[prompt.source.ordinal] = _construct_row(
                prompt, raw, cache, budget=budget
            )
    _require(all(row is not None for row in output_rows), "scan omitted questions")
    questions = [row for row in output_rows if row is not None]
    aggregate = {
        "admitted_candidate_count": sum(
            len(row["admitted_candidate_ids"]) for row in questions
        ),
        "candidate_count": sum(len(row["candidate_ids"]) for row in questions),
        "dedup_excluded_candidate_count": sum(
            len(row["dedup_excluded_candidate_ids"]) for row in questions
        ),
        "logical_scanned_content_row_memberships": sum(
            int(row["logical_scanned_content_row_count"]) for row in questions
        ),
        "maximum_tokens_used": max(int(row["tokens_used"]) for row in questions),
        "selected_candidate_count": sum(
            len(row["selected_before_dedup_candidate_ids"]) for row in questions
        ),
        "selected_second_span_count": sum(
            sum(candidate["span_rank"] > 0 for candidate in row["admitted_candidates"])
            for row in questions
        ),
        "total_tokens_used": sum(int(row["tokens_used"]) for row in questions),
    }
    historical_calls = int(parent.run.payload.get("provider_unique_calls", -1))
    _require(
        historical_calls == population.prompt_population.unique_prompt_count,
        "parent provider population changed",
    )
    payload: dict[str, Any] = {
        "aggregate": aggregate,
        "arm_label": ARM_LABEL,
        "budget": budget.projection(),
        "budget_id": budget.budget_id,
        "cache_policy": "one_ordered_sqlite_read_then_immutable_partition_tuples",
        "format": RUN_FORMAT,
        "gold_loaded": False,
        "historical_parent_provider_calls": historical_calls,
        "known_history_filter_used": False,
        "namespace_cache_receipts": cache_receipts,
        "new_provider_calls": 0,
        "parent_bindings": {
            "preflight_sha256": parent.preflight.sha256,
            "run_sha256": parent.run.sha256,
            "runtime_ledger_sha256": parent.runtime_ledger.sha256,
        },
        "partition_aggregation": "sealed_route_rank_weighted_vote_4_3_2_1",
        "physical_database_read_passes": len(cache_receipts),
        "plan_id": PLAN_ID,
        "provider_calls": 0,
        "question_count": len(questions),
        "questions": questions,
        "retained_transformer_token_state_bytes": 0,
        "routing_retrieval_rerun": False,
        "source_population_id": population.source_population.population_id,
        "source_prefix_filter_used": False,
    }
    assert_gold_blind(payload, path="query_guided_scan_run")
    return payload


def _runtime_entries(
    population: QueryExpansionPopulation,
    run: SealedArtifact,
) -> tuple[RuntimeLedgerEntry, ...]:
    rows = run.payload.get("questions")
    _require(
        type(rows) is list and len(rows) == len(population.rows),
        "scan runtime population changed",
    )
    entries: list[RuntimeLedgerEntry] = []
    for prompt, raw in zip(population.rows, rows, strict=True):
        _require(type(raw) is dict, "scan runtime row changed")
        receipt_sha = require_sha256(raw.get("receipt_sha256"), "scan row receipt")
        unsigned = dict(raw)
        unsigned.pop("receipt_sha256")
        _require(identity_sha256(unsigned) == receipt_sha, "scan row receipt changed")
        candidates = tuple(raw.get("candidate_ids", ()))
        selected = tuple(raw.get("selected_before_dedup_candidate_ids", ()))
        excluded = tuple(raw.get("dedup_excluded_candidate_ids", ()))
        not_admitted = tuple(raw.get("not_admitted_candidate_ids", ()))
        admitted = tuple(raw.get("admitted_candidate_ids", ()))
        delta_sha = identity_sha256(
            {
                "admitted_candidate_ids": list(admitted),
                "dedup_excluded_candidate_ids": list(excluded),
                "selected_before_dedup_candidate_ids": list(selected),
                "stage_id": STAGE_ID,
            }
        )
        packet_sha = identity_sha256(
            {
                "admitted_candidate_ids": list(admitted),
                "parent_query_expansion_row_receipt_sha256": raw[
                    "parent_query_expansion_row_receipt_sha256"
                ],
                "stage_id": STAGE_ID,
            }
        )
        entries.append(
            RuntimeLedgerEntry(
                event_type="stage",
                ordinal=prompt.source.ordinal,
                question_id=prompt.source.packet.question_id,
                question_sha256=prompt.source.packet.question_sha256,
                arm_label=ARM_LABEL,
                parent_arm_label=PARENT_ARM_LABEL,
                stage_id=STAGE_ID,
                parent_stage_id=PARENT_STAGE_ID,
                mechanism_id=MECHANISM_ID,
                delta_kind="membership",
                renderer_id=RENDERER_ID,
                legacy_renderer=False,
                disposition=StageDisposition(str(raw["disposition"])),
                candidate_ids=candidates,
                selected_before_dedup_ids=selected,
                dedup_excluded_ids=excluded,
                not_admitted_ids=not_admitted,
                admitted_ids=admitted,
                token_cap=int(raw["candidate_token_cap"]),
                tokens_used=int(raw["tokens_used"]),
                reported_tokens_used=int(raw["tokens_used"]),
                local_model_calls=0,
                provider_calls=0,
                provider_prompt_cap=0,
                provider_prompt_reserved=0,
                global_provider_prompt_cap=0,
                historical_provider_calls=1,
                parent_packet_sha256=str(
                    raw["parent_query_expansion_row_receipt_sha256"]
                ),
                packet_sha256=packet_sha,
                delta_sha256=delta_sha,
                stage_receipt_sha256=receipt_sha,
                source_row_sha256=identity_sha256(dict(raw)),
                reason=str(raw["reason"]),
            )
        )
    return tuple(entries)


def _ledger_payload(
    population: QueryExpansionPopulation,
    run: SealedArtifact,
    parent: VerifiedQueryExpansionParent,
) -> dict[str, Any]:
    return build_runtime_ledger(
        snapshot_id=population.source_population.snapshot.snapshot_id,
        plan_id=PLAN_ID,
        entries=_runtime_entries(population, run),
        source_artifacts=(
            {"role": "sealed_retrieval", "sha256": population.source_population.retrieval_sha256},
            {"role": "parent_query_preflight", "sha256": parent.preflight.sha256},
            {"role": "parent_query_run", "sha256": parent.run.sha256},
            {"role": "parent_query_runtime", "sha256": parent.runtime_ledger.sha256},
            {"role": "query_guided_scan_run", "sha256": run.sha256},
        ),
    )


@dataclass(frozen=True, slots=True)
class QueryGuidedScanResult:
    run_artifact: SealedArtifact
    runtime_ledger_artifact: SealedArtifact
    physical_provider_calls: int = 0
    retained_transformer_token_state_bytes: int = 0


def materialize_query_guided_scan(
    context: LockedQueryExpansionContext,
    *,
    parent_output_root: str | Path,
    output_root: str | Path,
    expected_parent_preflight_sha256: str,
    expected_parent_run_sha256: str,
    expected_parent_runtime_ledger_sha256: str,
    budget: QueryGuidedScanBudget = QueryGuidedScanBudget(),
) -> QueryGuidedScanResult:
    """Seal the provider-free cached exhaustive scan."""

    output = Path(output_root)
    _require(not (output / RUN_NAME).exists(), "scan run exists; use replay")
    context.revalidate_store_bytes()
    parent = verify_query_expansion_parent(
        context.population,
        parent_output_root=parent_output_root,
        expected_preflight_sha256=expected_parent_preflight_sha256,
        expected_run_sha256=expected_parent_run_sha256,
        expected_runtime_ledger_sha256=expected_parent_runtime_ledger_sha256,
    )
    payload = _build_payload(context, parent, budget=budget)
    context.revalidate_store_bytes()
    run, _created = publish_sealed_json(output / RUN_NAME, payload)
    ledger, _created = publish_sealed_json(
        output / RUNTIME_LEDGER_NAME,
        _ledger_payload(context.population, run, parent),
    )
    return QueryGuidedScanResult(run, ledger)


def replay_query_guided_scan(
    context: LockedQueryExpansionContext,
    *,
    parent_output_root: str | Path,
    output_root: str | Path,
    expected_parent_preflight_sha256: str,
    expected_parent_run_sha256: str,
    expected_parent_runtime_ledger_sha256: str,
    expected_run_sha256: str,
    budget: QueryGuidedScanBudget = QueryGuidedScanBudget(),
) -> QueryGuidedScanResult:
    """Rebuild the cache/scan and require byte-identical run and ledger bytes."""

    output = Path(output_root)
    expected = require_sha256(expected_run_sha256, "expected query-guided scan")
    source = read_sealed_json(output / RUN_NAME)
    _require(source.sha256 == expected, "sealed query-guided scan changed")
    context.revalidate_store_bytes()
    parent = verify_query_expansion_parent(
        context.population,
        parent_output_root=parent_output_root,
        expected_preflight_sha256=expected_parent_preflight_sha256,
        expected_run_sha256=expected_parent_run_sha256,
        expected_runtime_ledger_sha256=expected_parent_runtime_ledger_sha256,
    )
    rebuilt = _build_payload(context, parent, budget=budget)
    context.revalidate_store_bytes()
    _require(rebuilt == source.payload, "query-guided scan replay changed")
    replay, _created = publish_sealed_json(output / RUN_REPLAY_NAME, rebuilt)
    _require(replay.sha256 == source.sha256, "query-guided run/replay seals differ")

    source_ledger = read_sealed_json(output / RUNTIME_LEDGER_NAME)
    rebuilt_ledger = _ledger_payload(context.population, source, parent)
    _require(
        rebuilt_ledger == source_ledger.payload,
        "query-guided runtime ledger changed",
    )
    ledger_replay, _created = publish_sealed_json(
        output / RUNTIME_LEDGER_REPLAY_NAME, rebuilt_ledger
    )
    _require(
        ledger_replay.sha256 == source_ledger.sha256,
        "query-guided runtime/replay seals differ",
    )
    return QueryGuidedScanResult(source, source_ledger)


__all__ = [
    "ARM_LABEL",
    "BalancedSelection",
    "CachedContentRow",
    "NamespacePartitionCache",
    "QueryGuidedCandidate",
    "QueryGuidedScanBudget",
    "QueryGuidedScanError",
    "QueryGuidedScanResult",
    "RUN_NAME",
    "RUN_REPLAY_NAME",
    "RUNTIME_LEDGER_NAME",
    "RUNTIME_LEDGER_REPLAY_NAME",
    "aggregate_partition_votes",
    "cache_namespace_partitions",
    "materialize_query_guided_scan",
    "replay_query_guided_scan",
    "score_query_guided_candidates",
    "select_balanced_candidates",
]
