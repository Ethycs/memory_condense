"""Gold-blind matched-eval lane over the incremental conversation graph.

Cold construction appends every exact :class:`CachedContentRow` from an
already authenticated :class:`FullStoreWindowIndex` once.  Query ticks use a
dated question to steer a bounded graph walk from explicit physical seed
chunks.  Seeds are activation inputs only: this lane emits and charges tokens
for novel graph-reached raw chunks, then leaves cross-lane deduplication to a
later composer.

The implementation is deliberately provider-free.  It accepts no question
ID, reference, answer, prediction, model, or provider client.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import EvidenceSpan, quote_sha256
from memory_condense.search.incremental_conversation_graph import (
    ConversationGraphChunk,
    ConversationGraphStats,
    GraphAppendReceipt,
    GraphEvidence,
    GraphTransition,
    GraphTraversalPolicy,
    IncrementalConversationGraph,
    PhraseExtractionPolicy,
    PhraseOccurrence,
)

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .full_store_slot_closure import FullStoreWindowIndex
from .query_guided_scan import CachedContentRow


MECHANISM_ID = "hot_incremental_conversation_graph_lane_v1"
DEFAULT_EVIDENCE_TOKEN_CAP = 1_200
INDEX_FORMAT = "memory-condense-hot-incremental-graph-index-v1"
PATH_STEP_FORMAT = "memory-condense-hot-incremental-graph-path-step-v1"
EVIDENCE_FORMAT = "memory-condense-hot-incremental-graph-evidence-v1"
RECEIPT_FORMAT = "memory-condense-hot-incremental-graph-receipt-v1"
RESULT_FORMAT = "memory-condense-hot-incremental-graph-result-v1"
SEED_BINDING_FORMAT = "memory-condense-hot-incremental-graph-seed-binding-v1"

GraphLaneStatus = Literal[
    "no_seed",
    "no_novel_graph_evidence",
    "novel_graph_evidence_all_over_budget",
    "novel_graph_evidence_selected",
]

_DATED_QUESTION_RE = re.compile(
    r"^\[Question asked at [^\]\r\n]+\]\s*(?P<body>\S[\s\S]*)$",
    re.IGNORECASE,
)


class HotIncrementalGraphLaneError(MatchedEvalContractError):
    """Raised when the graph snapshot, path, or independent budget changes."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise HotIncrementalGraphLaneError(message)


def _ordered_unique(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(values)
    _require(
        all(type(value) is str and value and value.strip() == value for value in result)
        and len(result) == len(set(result)),
        f"{label} must be ordered unique exact text",
    )
    return result


def _stats_projection(stats: ConversationGraphStats) -> dict[str, int]:
    return asdict(stats)


def _graph_chunk(row: CachedContentRow) -> ConversationGraphChunk:
    return ConversationGraphChunk(
        chunk_id=row.chunk_id,
        source_id=row.source_id,
        turn_id=row.turn_id,
        ordinal=row.ordinal,
        role=row.role,
        text=row.text,
        start_char=row.turn_start_char,
        end_char=row.turn_end_char,
        created_at=row.created_at,
        text_sha256=row.text_sha256,
    )


@dataclass(frozen=True, slots=True)
class HotIncrementalGraphBudget:
    """Independent graph expansion and raw-evidence bounds."""

    evidence_token_cap: Literal[1200] = DEFAULT_EVIDENCE_TOKEN_CAP
    max_seed_chunks: int = 16
    max_hops: int = 2
    max_degree: int = 12
    max_frontier: int = 48
    max_candidates: int = 48
    max_query_phrases: int = 32
    max_phrases_per_node: int = 32
    max_phrase_postings: int = 64
    hop_decay: float = 0.72
    sequence_weight: float = 0.58

    def __post_init__(self) -> None:
        _require(
            type(self.evidence_token_cap) is int
            and self.evidence_token_cap == DEFAULT_EVIDENCE_TOKEN_CAP,
            "graph lane must retain its independent 1,200-token cap",
        )
        for name in (
            "max_seed_chunks",
            "max_degree",
            "max_frontier",
            "max_candidates",
            "max_query_phrases",
            "max_phrases_per_node",
            "max_phrase_postings",
        ):
            _require(
                type(getattr(self, name)) is int and getattr(self, name) > 0,
                f"{name} must be a positive exact integer",
            )
        _require(
            type(self.max_hops) is int and 0 <= self.max_hops <= 2,
            "max_hops must be an exact integer in [0, 2]",
        )
        for name in ("hop_decay", "sequence_weight"):
            value = float(getattr(self, name))
            _require(
                math.isfinite(value) and 0.0 < value <= 1.0,
                f"{name} must be finite and in (0, 1]",
            )

    def graph_policy(self, *, seed_count: int) -> GraphTraversalPolicy:
        return GraphTraversalPolicy(
            max_hops=self.max_hops,
            max_degree=self.max_degree,
            max_frontier=self.max_frontier,
            # The core's max_results is already exclusive of explicit seeds.
            max_results=self.max_candidates,
            max_seed_chunks=self.max_seed_chunks,
            max_query_phrases=self.max_query_phrases,
            max_phrases_per_node=self.max_phrases_per_node,
            max_phrase_postings=self.max_phrase_postings,
            hop_decay=self.hop_decay,
            sequence_weight=self.sequence_weight,
        )

    def projection(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "budget_scope": "novel_graph_evidence_before_global_dedup",
            "seed_activation_inputs_charge_tokens": False,
            "source_diversity_policy": "round_robin_first_seen_graph_rank",
        }

    @property
    def budget_id(self) -> str:
        return identity_sha256(
            {"mechanism_id": MECHANISM_ID, "budget": self.projection()}
        )


@dataclass(frozen=True, slots=True)
class HotIncrementalGraphSeedBinding:
    """Physical activation seeds bound to one sealed upstream retrieval.

    The graph lane does not accept a bare seed list.  The complete upstream
    selected-ID inventory is sealed here, and graph seeds must be an ordered
    subsequence of that inventory.
    """

    upstream_retrieval_receipt_sha256: str
    upstream_selected_chunk_ids: tuple[str, ...]
    seed_chunk_ids: tuple[str, ...]
    upstream_selected_chunk_ids_sha256: str = ""
    seed_chunk_ids_sha256: str = ""
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(
            self.upstream_retrieval_receipt_sha256,
            "upstream retrieval receipt",
        )
        upstream = _ordered_unique(
            self.upstream_selected_chunk_ids,
            "upstream selected chunk IDs",
        )
        seeds = _ordered_unique(self.seed_chunk_ids, "graph seed chunks")
        _require(
            _is_subsequence(seeds, upstream),
            "graph seeds are not an ordered subset of upstream retrieval",
        )
        upstream_digest = identity_sha256(list(upstream))
        seed_digest = identity_sha256(list(seeds))
        if self.upstream_selected_chunk_ids_sha256:
            _require(
                self.upstream_selected_chunk_ids_sha256 == upstream_digest,
                "upstream selected-ID digest changed",
            )
        if self.seed_chunk_ids_sha256:
            _require(
                self.seed_chunk_ids_sha256 == seed_digest,
                "graph seed-ID digest changed",
            )
        object.__setattr__(
            self,
            "upstream_selected_chunk_ids_sha256",
            upstream_digest,
        )
        object.__setattr__(self, "seed_chunk_ids_sha256", seed_digest)
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "graph seed binding changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="hot_incremental_graph_seed_binding")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "format": SEED_BINDING_FORMAT,
            "seed_chunk_ids": list(self.seed_chunk_ids),
            "seed_chunk_ids_sha256": self.seed_chunk_ids_sha256,
            "upstream_retrieval_receipt_sha256": (
                self.upstream_retrieval_receipt_sha256
            ),
            "upstream_selected_chunk_ids": list(
                self.upstream_selected_chunk_ids
            ),
            "upstream_selected_chunk_ids_sha256": (
                self.upstream_selected_chunk_ids_sha256
            ),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class HotIncrementalGraphIndex:
    """Full-store rows bound to one incrementally reconstructed graph image."""

    parent: FullStoreWindowIndex
    graph: IncrementalConversationGraph = field(repr=False, compare=False)
    rows_by_chunk_id: Mapping[str, CachedContentRow] = field(repr=False)
    append_receipts: Mapping[str, GraphAppendReceipt] = field(repr=False)
    graph_stats: ConversationGraphStats
    append_inventory_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(type(self.parent) is FullStoreWindowIndex, "full-store index changed")
        _require(
            type(self.graph) is IncrementalConversationGraph,
            "incremental graph implementation changed",
        )
        rows = dict(self.rows_by_chunk_id)
        receipts = dict(self.append_receipts)
        expected_rows = tuple(self.parent.rows)
        _require(
            len(rows) == len(expected_rows)
            and tuple(rows) == tuple(row.chunk_id for row in expected_rows)
            and all(rows.get(row.chunk_id) == row for row in expected_rows),
            "graph index row inventory changed",
        )
        _require(
            tuple(receipts) == tuple(rows)
            and all(
                type(receipts[chunk_id]) is GraphAppendReceipt
                and receipts[chunk_id].chunk_id == chunk_id
                for chunk_id in rows
            ),
            "graph append receipt inventory changed",
        )
        _require(
            tuple(receipt.revision for receipt in receipts.values())
            == tuple(range(1, len(receipts) + 1)),
            "graph cold reconstruction did not append every row exactly once",
        )
        _require(
            type(self.graph_stats) is ConversationGraphStats
            and self.graph.stats() == self.graph_stats
            and self.graph_stats.chunk_count == len(rows)
            and self.graph_stats.revision == len(rows),
            "graph stats differ from the full-store snapshot",
        )
        for row in expected_rows:
            _require(
                self.graph.chunk(row.chunk_id) == _graph_chunk(row),
                "graph physical chunk differs from its cached source row",
            )
        inventory = [
            {
                "append_receipt_sha256": receipts[row.chunk_id].receipt_sha256,
                "chunk_id": row.chunk_id,
                "source_row_receipt_sha256": identity_sha256(
                    row.receipt_projection()
                ),
            }
            for row in expected_rows
        ]
        _require(
            self.append_inventory_sha256 == identity_sha256(inventory),
            "graph append inventory receipt changed",
        )
        object.__setattr__(self, "rows_by_chunk_id", MappingProxyType(rows))
        object.__setattr__(self, "append_receipts", MappingProxyType(receipts))
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "graph index receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="hot_incremental_graph_index")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "append_inventory_sha256": self.append_inventory_sha256,
            "cache_receipt_sha256": self.parent.cache.cache_receipt_sha256,
            "format": INDEX_FORMAT,
            "full_store_index_receipt_sha256": self.parent.receipt_sha256,
            "gold_loaded": False,
            "graph_stats": _stats_projection(self.graph_stats),
            "incremental_append_attempt_count": len(self.rows_by_chunk_id),
            "incremental_append_created_count": len(self.rows_by_chunk_id),
            "incremental_append_retry_count": 0,
            "model_calls": 0,
            "new_provider_calls": 0,
            "phrase_extraction_policy_sha256": (
                self.graph.extraction_policy.policy_sha256
            ),
            "physical_content_row_count": len(self.rows_by_chunk_id),
            "retained_transformer_token_state_bytes": 0,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def build_hot_incremental_graph_index(
    parent: FullStoreWindowIndex,
    /,
    *,
    extraction_policy: PhraseExtractionPolicy | None = None,
) -> HotIncrementalGraphIndex:
    """Cold-reconstruct a graph by appending every exact parent row once."""

    _require(type(parent) is FullStoreWindowIndex, "full-store index changed")
    if extraction_policy is not None:
        _require(
            type(extraction_policy) is PhraseExtractionPolicy,
            "phrase extraction policy changed",
        )
    graph = IncrementalConversationGraph(extraction_policy=extraction_policy)
    rows: dict[str, CachedContentRow] = {}
    receipts: dict[str, GraphAppendReceipt] = {}
    inventory: list[dict[str, str]] = []
    for row in parent.rows:
        _require(row.chunk_id not in rows, "full-store row repeats a physical chunk")
        append = graph.append_chunk(_graph_chunk(row))
        _require(append.created, "cold graph append unexpectedly became a retry")
        rows[row.chunk_id] = row
        receipts[row.chunk_id] = append.receipt
        inventory.append(
            {
                "append_receipt_sha256": append.receipt.receipt_sha256,
                "chunk_id": row.chunk_id,
                "source_row_receipt_sha256": identity_sha256(
                    row.receipt_projection()
                ),
            }
        )
    return HotIncrementalGraphIndex(
        parent=parent,
        graph=graph,
        rows_by_chunk_id=rows,
        append_receipts=receipts,
        graph_stats=graph.stats(),
        append_inventory_sha256=identity_sha256(inventory),
    )


@dataclass(frozen=True, slots=True)
class GraphOccurrenceRef:
    occurrence_id: str
    phrase_key: str
    chunk_id: str
    source_id: str
    turn_id: str
    start_char: int
    end_char: int
    quote: str
    quote_sha256: str
    token_count: int

    @classmethod
    def from_core(cls, occurrence: PhraseOccurrence) -> GraphOccurrenceRef:
        return cls(**asdict(occurrence))

    def __post_init__(self) -> None:
        for value, label in (
            (self.occurrence_id, "graph occurrence"),
            (self.phrase_key, "graph occurrence phrase"),
            (self.chunk_id, "graph occurrence chunk"),
            (self.source_id, "graph occurrence source"),
            (self.turn_id, "graph occurrence turn"),
            (self.quote, "graph occurrence quote"),
        ):
            require_text(value, label)
        require_sha256(self.occurrence_id, "graph occurrence ID")
        require_sha256(self.quote_sha256, "graph occurrence quote")
        _require(
            type(self.start_char) is int
            and type(self.end_char) is int
            and 0 <= self.start_char < self.end_char
            and self.quote_sha256 == quote_sha256(self.quote)
            and type(self.token_count) is int
            and self.token_count > 0,
            "graph occurrence coordinates or exact quote changed",
        )

    def projection(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HotIncrementalGraphPathStep:
    source_chunk_id: str
    target_chunk_id: str
    relation: str
    weight: float
    shared_phrases: tuple[str, ...]
    sequence_direction: str | None
    source_occurrence: GraphOccurrenceRef | None
    target_occurrence: GraphOccurrenceRef | None
    source_append_receipt_sha256: str
    target_append_receipt_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_text(self.source_chunk_id, "graph path source")
        require_text(self.target_chunk_id, "graph path target")
        _require(
            self.source_chunk_id != self.target_chunk_id
            and self.relation in {"same_source_sequence", "shared_phrase"}
            and math.isfinite(float(self.weight))
            and 0.0 < self.weight <= 1.0,
            "graph path edge identity changed",
        )
        shared = _ordered_unique(self.shared_phrases, "shared graph phrases")
        require_sha256(self.source_append_receipt_sha256, "path source append")
        require_sha256(self.target_append_receipt_sha256, "path target append")
        if self.relation == "shared_phrase":
            _require(
                bool(shared)
                and type(self.source_occurrence) is GraphOccurrenceRef
                and type(self.target_occurrence) is GraphOccurrenceRef
                and self.sequence_direction is None
                and self.source_occurrence.chunk_id == self.source_chunk_id
                and self.target_occurrence.chunk_id == self.target_chunk_id
                and self.source_occurrence.phrase_key == self.target_occurrence.phrase_key
                and self.source_occurrence.phrase_key in shared,
                "shared-phrase path lost its exact occurrence bridge",
            )
        else:
            _require(
                not shared
                and self.source_occurrence is None
                and self.target_occurrence is None
                and self.sequence_direction in {"previous", "next"},
                "sequence path acquired phrase evidence or lost direction",
            )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "graph path receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "format": PATH_STEP_FORMAT,
            "relation": self.relation,
            "sequence_direction": self.sequence_direction,
            "shared_phrases": list(self.shared_phrases),
            "source_append_receipt_sha256": self.source_append_receipt_sha256,
            "source_chunk_id": self.source_chunk_id,
            "source_occurrence": (
                None
                if self.source_occurrence is None
                else self.source_occurrence.projection()
            ),
            "target_append_receipt_sha256": self.target_append_receipt_sha256,
            "target_chunk_id": self.target_chunk_id,
            "target_occurrence": (
                None
                if self.target_occurrence is None
                else self.target_occurrence.projection()
            ),
            "weight": self.weight,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _validate_occurrence_binding(
    occurrence: PhraseOccurrence,
    row: CachedContentRow,
) -> None:
    local_start = occurrence.start_char - row.turn_start_char
    local_end = occurrence.end_char - row.turn_start_char
    _require(
        occurrence.chunk_id == row.chunk_id
        and occurrence.source_id == row.source_id
        and occurrence.turn_id == row.turn_id
        and 0 <= local_start < local_end <= len(row.text)
        and row.text[local_start:local_end] == occurrence.quote
        and quote_sha256(occurrence.quote) == occurrence.quote_sha256,
        "graph occurrence is not an exact cached-row substring",
    )


def _path_step(
    index: HotIncrementalGraphIndex,
    edge: GraphTransition,
) -> HotIncrementalGraphPathStep:
    rows = index.rows_by_chunk_id
    _require(
        edge.source_chunk_id in rows and edge.target_chunk_id in rows,
        "graph path escaped the full-store row inventory",
    )
    if edge.source_occurrence is not None:
        _validate_occurrence_binding(
            edge.source_occurrence,
            rows[edge.source_chunk_id],
        )
    if edge.target_occurrence is not None:
        _validate_occurrence_binding(
            edge.target_occurrence,
            rows[edge.target_chunk_id],
        )
    return HotIncrementalGraphPathStep(
        source_chunk_id=edge.source_chunk_id,
        target_chunk_id=edge.target_chunk_id,
        relation=edge.relation,
        weight=edge.weight,
        shared_phrases=edge.shared_phrases,
        sequence_direction=edge.sequence_direction,
        source_occurrence=(
            None
            if edge.source_occurrence is None
            else GraphOccurrenceRef.from_core(edge.source_occurrence)
        ),
        target_occurrence=(
            None
            if edge.target_occurrence is None
            else GraphOccurrenceRef.from_core(edge.target_occurrence)
        ),
        source_append_receipt_sha256=index.append_receipts[
            edge.source_chunk_id
        ].receipt_sha256,
        target_append_receipt_sha256=index.append_receipts[
            edge.target_chunk_id
        ].receipt_sha256,
    )


def _materialize_path(
    index: HotIncrementalGraphIndex,
    evidence: GraphEvidence,
) -> tuple[HotIncrementalGraphPathStep, ...]:
    path = tuple(_path_step(index, edge) for edge in evidence.path)
    _require(
        len(path) == evidence.hop
        and bool(path)
        and path[-1].target_chunk_id == evidence.chunk_id
        and all(
            left.target_chunk_id == right.source_chunk_id
            for left, right in zip(path, path[1:])
        ),
        "graph evidence path is not a complete physical chain",
    )
    return path


@dataclass(frozen=True, slots=True)
class HotIncrementalGraphEvidence:
    """One complete raw CachedContentRow reached beyond an activation seed."""

    chunk_id: str
    source_id: str
    turn_id: str
    role: str
    created_at: str
    ordinal: int
    turn_start_char: int
    turn_end_char: int
    raw_text: str
    token_count: int
    score: float
    graph_hop: int
    supporting_seed_chunk_ids: tuple[str, ...]
    matched_question_phrases: tuple[str, ...]
    path: tuple[HotIncrementalGraphPathStep, ...]
    span: EvidenceSpan
    path_receipt_sha256: str
    append_receipt_sha256: str
    index_receipt_sha256: str
    cache_receipt_sha256: str
    full_store_index_receipt_sha256: str
    source_row_receipt_sha256: str
    span_receipt_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.chunk_id, "graph evidence chunk"),
            (self.source_id, "graph evidence source"),
            (self.turn_id, "graph evidence turn"),
            (self.role, "graph evidence role"),
            (self.created_at, "graph evidence timestamp"),
            (self.raw_text, "graph evidence raw text"),
        ):
            require_text(value, label)
        seeds = _ordered_unique(
            self.supporting_seed_chunk_ids,
            "graph supporting seed chunks",
        )
        _ordered_unique(
            self.matched_question_phrases,
            "matched graph question phrases",
        )
        for value, label in (
            (self.path_receipt_sha256, "graph path"),
            (self.append_receipt_sha256, "graph evidence append"),
            (self.index_receipt_sha256, "graph index"),
            (self.cache_receipt_sha256, "graph cache"),
            (self.full_store_index_receipt_sha256, "graph full-store index"),
            (self.source_row_receipt_sha256, "graph source row"),
            (self.span_receipt_sha256, "graph evidence span"),
        ):
            require_sha256(value, label)
        _require(
            type(self.ordinal) is int
            and type(self.turn_start_char) is int
            and type(self.turn_end_char) is int
            and self.ordinal >= 0
            and 0 <= self.turn_start_char < self.turn_end_char
            and self.turn_end_char - self.turn_start_char == len(self.raw_text)
            and type(self.token_count) is int
            and self.token_count == count_tokens(self.raw_text)
            and math.isfinite(float(self.score))
            and 0.0 < self.score <= 1.0
            and type(self.graph_hop) is int
            and 1 <= self.graph_hop <= 2,
            "graph evidence coordinates, score, hop, or token count changed",
        )
        _require(
            type(self.span) is EvidenceSpan
            and self.span.chunk_id == self.chunk_id
            and self.span.source_id == self.source_id
            and self.span.turn_id == self.turn_id
            and self.span.role == self.role
            and self.span.created_at == self.created_at
            and self.span.ordinal == self.ordinal
            and self.span.turn_start_char == self.turn_start_char
            and self.span.start_char == 0
            and self.span.end_char == len(self.raw_text)
            and self.span.quote_sha256 == quote_sha256(self.raw_text)
            and self.span_receipt_sha256 == identity_sha256(
                self.span.identity_payload()
            ),
            "graph evidence lost its complete exact raw span",
        )
        _require(
            len(self.path) == self.graph_hop
            and bool(self.path)
            and seeds == (self.path[0].source_chunk_id,)
            and self.path[-1].target_chunk_id == self.chunk_id
            and all(
                left.target_chunk_id == right.source_chunk_id
                for left, right in zip(self.path, self.path[1:])
            )
            and self.path_receipt_sha256
            == identity_sha256([step.receipt_sha256 for step in self.path]),
            "graph evidence path receipts changed",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "graph evidence receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "append_receipt_sha256": self.append_receipt_sha256,
            "cache_receipt_sha256": self.cache_receipt_sha256,
            "chunk_id": self.chunk_id,
            "created_at": self.created_at,
            "format": EVIDENCE_FORMAT,
            "full_store_index_receipt_sha256": self.full_store_index_receipt_sha256,
            "graph_hop": self.graph_hop,
            "index_receipt_sha256": self.index_receipt_sha256,
            "matched_question_phrases": list(self.matched_question_phrases),
            "ordinal": self.ordinal,
            "path": [step.projection() for step in self.path],
            "path_receipt_sha256": self.path_receipt_sha256,
            "raw_text": self.raw_text,
            "raw_text_sha256": quote_sha256(self.raw_text),
            "role": self.role,
            "score": self.score,
            "source_id": self.source_id,
            "source_row_receipt_sha256": self.source_row_receipt_sha256,
            "span": self.span.identity_payload(),
            "span_receipt_sha256": self.span_receipt_sha256,
            "supporting_seed_chunk_ids": list(self.supporting_seed_chunk_ids),
            "token_count": self.token_count,
            "turn_end_char": self.turn_end_char,
            "turn_id": self.turn_id,
            "turn_start_char": self.turn_start_char,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _materialize_evidence(
    index: HotIncrementalGraphIndex,
    evidence: GraphEvidence,
    path: tuple[HotIncrementalGraphPathStep, ...],
) -> HotIncrementalGraphEvidence:
    row = index.rows_by_chunk_id[evidence.chunk_id]
    _require(
        evidence.chunk == _graph_chunk(row),
        "graph result physical chunk differs from cached source evidence",
    )
    span = EvidenceSpan(
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
    return HotIncrementalGraphEvidence(
        chunk_id=row.chunk_id,
        source_id=row.source_id,
        turn_id=row.turn_id,
        role=row.role,
        created_at=row.created_at,
        ordinal=row.ordinal,
        turn_start_char=row.turn_start_char,
        turn_end_char=row.turn_end_char,
        raw_text=row.text,
        token_count=row.token_count,
        score=evidence.score,
        graph_hop=evidence.hop,
        supporting_seed_chunk_ids=evidence.supporting_seed_chunk_ids,
        matched_question_phrases=evidence.matched_query_phrases,
        path=path,
        span=span,
        path_receipt_sha256=identity_sha256(
            [step.receipt_sha256 for step in path]
        ),
        append_receipt_sha256=index.append_receipts[row.chunk_id].receipt_sha256,
        index_receipt_sha256=index.receipt_sha256,
        cache_receipt_sha256=index.parent.cache.cache_receipt_sha256,
        full_store_index_receipt_sha256=index.parent.receipt_sha256,
        source_row_receipt_sha256=identity_sha256(row.receipt_projection()),
        span_receipt_sha256=identity_sha256(span.identity_payload()),
    )


def _source_round_robin(
    evidence: Sequence[GraphEvidence],
    rows: Mapping[str, CachedContentRow],
) -> tuple[GraphEvidence, ...]:
    grouped: dict[str, list[GraphEvidence]] = {}
    for row in evidence:
        grouped.setdefault(rows[row.chunk_id].source_id, []).append(row)
    source_order = tuple(grouped)
    output: list[GraphEvidence] = []
    depth = 0
    while True:
        emitted = False
        for source_id in source_order:
            values = grouped[source_id]
            if depth < len(values):
                output.append(values[depth])
                emitted = True
        if not emitted:
            return tuple(output)
        depth += 1


def _core_transition_projection(edge: GraphTransition) -> dict[str, Any]:
    return {
        "relation": edge.relation,
        "sequence_direction": edge.sequence_direction,
        "shared_phrases": list(edge.shared_phrases),
        "source_chunk_id": edge.source_chunk_id,
        "source_occurrence_id": (
            None if edge.source_occurrence is None else edge.source_occurrence.occurrence_id
        ),
        "target_chunk_id": edge.target_chunk_id,
        "target_occurrence_id": (
            None if edge.target_occurrence is None else edge.target_occurrence.occurrence_id
        ),
        "weight": edge.weight,
    }


@dataclass(frozen=True, slots=True)
class HotIncrementalGraphReceipt:
    status: GraphLaneStatus
    index_receipt_sha256: str
    budget_id: str
    dated_question_sha256: str
    question_body_sha256: str
    seed_binding: HotIncrementalGraphSeedBinding
    seed_chunk_ids: tuple[str, ...]
    seed_chunk_ids_sha256: str
    seed_append_receipt_sha256s: tuple[str, ...]
    seed_activation_input_tokens: int
    graph_search_policy_sha256: str
    graph_search_receipt_sha256: str
    graph_ranked_novel_ids: tuple[str, ...]
    candidate_population_ids: tuple[str, ...]
    candidate_source_ids: tuple[str, ...]
    candidate_path_receipt_sha256s: tuple[str, ...]
    selected_before_dedup_ids: tuple[str, ...]
    selected_evidence_receipt_sha256s: tuple[str, ...]
    selected_before_dedup_tokens: int
    budget_excluded_ids: tuple[str, ...]
    selection_truncated: bool
    independent_lane_token_cap: Literal[1200] = DEFAULT_EVIDENCE_TOKEN_CAP
    graph_max_hops: int = 2
    selection_before_global_dedup: Literal[True] = True
    seed_activation_inputs_charge_tokens: Literal[False] = False
    seed_activation_inputs_emitted: Literal[False] = False
    source_diversity_policy: Literal[
        "round_robin_first_seen_graph_rank"
    ] = "round_robin_first_seen_graph_rank"
    refill_after_global_dedup: Literal[False] = False
    new_provider_calls: Literal[0] = 0
    model_calls: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    gold_loaded: Literal[False] = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            self.status
            in {
                "no_seed",
                "no_novel_graph_evidence",
                "novel_graph_evidence_all_over_budget",
                "novel_graph_evidence_selected",
            },
            "graph lane status changed",
        )
        for value, label in (
            (self.index_receipt_sha256, "graph lane index"),
            (self.budget_id, "graph lane budget"),
            (self.dated_question_sha256, "dated graph question"),
            (self.question_body_sha256, "graph question body"),
            (self.seed_chunk_ids_sha256, "graph seed population"),
            (self.graph_search_policy_sha256, "graph search policy"),
            (self.graph_search_receipt_sha256, "graph search"),
        ):
            require_sha256(value, label)
        seeds = _ordered_unique(self.seed_chunk_ids, "graph seed chunks")
        _require(
            type(self.seed_binding) is HotIncrementalGraphSeedBinding
            and seeds == self.seed_binding.seed_chunk_ids
            and self.seed_chunk_ids_sha256
            == self.seed_binding.seed_chunk_ids_sha256,
            "graph receipt seeds differ from their upstream binding",
        )
        graph_ranked = _ordered_unique(
            self.graph_ranked_novel_ids,
            "graph-ranked novel chunks",
        )
        candidates = _ordered_unique(
            self.candidate_population_ids,
            "diversified graph candidates",
        )
        _require(
            len(self.seed_append_receipt_sha256s) == len(seeds)
            and all(
                require_sha256(value, "graph seed append")
                for value in self.seed_append_receipt_sha256s
            )
            and self.seed_chunk_ids_sha256 == identity_sha256(list(seeds)),
            "graph seed receipt inventory changed",
        )
        _require(
            set(candidates) == set(graph_ranked)
            and len(candidates) == len(self.candidate_source_ids)
            and all(type(value) is str and value for value in self.candidate_source_ids)
            and set(candidates).isdisjoint(seeds)
            and len(self.candidate_path_receipt_sha256s) == len(candidates)
            and all(
                require_sha256(value, "candidate graph path")
                for value in self.candidate_path_receipt_sha256s
            ),
            "graph candidate population or source-diversity permutation changed",
        )
        selected = _ordered_unique(
            self.selected_before_dedup_ids,
            "selected graph chunks",
        )
        excluded = _ordered_unique(
            self.budget_excluded_ids,
            "budget-excluded graph chunks",
        )
        _require(
            set(selected).isdisjoint(excluded)
            and set(selected) | set(excluded) == set(candidates)
            and _is_subsequence(selected, candidates)
            and _is_subsequence(excluded, candidates)
            and len(self.selected_evidence_receipt_sha256s) == len(selected)
            and all(
                require_sha256(value, "selected graph evidence")
                for value in self.selected_evidence_receipt_sha256s
            ),
            "graph selection does not partition its frozen candidate population",
        )
        _require(
            type(self.seed_activation_input_tokens) is int
            and self.seed_activation_input_tokens >= 0
            and type(self.selected_before_dedup_tokens) is int
            and 0 <= self.selected_before_dedup_tokens
            <= self.independent_lane_token_cap
            and self.selection_truncated == bool(excluded),
            "graph lane token accounting changed",
        )
        expected_status: GraphLaneStatus
        if not seeds:
            expected_status = "no_seed"
        elif not candidates:
            expected_status = "no_novel_graph_evidence"
        elif not selected:
            expected_status = "novel_graph_evidence_all_over_budget"
        else:
            expected_status = "novel_graph_evidence_selected"
        _require(self.status == expected_status, "graph lane status is inconsistent")
        _require(
            self.independent_lane_token_cap == DEFAULT_EVIDENCE_TOKEN_CAP
            and type(self.graph_max_hops) is int
            and 0 <= self.graph_max_hops <= 2
            and self.selection_before_global_dedup is True
            and self.seed_activation_inputs_charge_tokens is False
            and self.seed_activation_inputs_emitted is False
            and self.source_diversity_policy
            == "round_robin_first_seen_graph_rank"
            and self.refill_after_global_dedup is False
            and self.new_provider_calls == self.model_calls == 0
            and self.retained_transformer_token_state_bytes == 0
            and self.gold_loaded is False,
            "graph lane zero-call, seed, or dedup policy changed",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "graph lane receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="hot_incremental_graph_receipt")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "budget_excluded_ids": list(self.budget_excluded_ids),
            "budget_id": self.budget_id,
            "candidate_path_receipt_sha256s": list(
                self.candidate_path_receipt_sha256s
            ),
            "candidate_population_ids": list(self.candidate_population_ids),
            "candidate_source_ids": list(self.candidate_source_ids),
            "dated_question_sha256": self.dated_question_sha256,
            "format": RECEIPT_FORMAT,
            "gold_loaded": False,
            "graph_max_hops": self.graph_max_hops,
            "graph_ranked_novel_ids": list(self.graph_ranked_novel_ids),
            "graph_search_policy_sha256": self.graph_search_policy_sha256,
            "graph_search_receipt_sha256": self.graph_search_receipt_sha256,
            "independent_lane_token_cap": self.independent_lane_token_cap,
            "index_receipt_sha256": self.index_receipt_sha256,
            "model_calls": 0,
            "new_provider_calls": 0,
            "question_body_sha256": self.question_body_sha256,
            "refill_after_global_dedup": False,
            "retained_transformer_token_state_bytes": 0,
            "seed_activation_input_tokens": self.seed_activation_input_tokens,
            "seed_activation_inputs_charge_tokens": False,
            "seed_activation_inputs_emitted": False,
            "seed_binding": self.seed_binding.projection(),
            "seed_append_receipt_sha256s": list(
                self.seed_append_receipt_sha256s
            ),
            "seed_chunk_ids": list(self.seed_chunk_ids),
            "seed_chunk_ids_sha256": self.seed_chunk_ids_sha256,
            "selected_before_dedup_ids": list(self.selected_before_dedup_ids),
            "selected_before_dedup_tokens": self.selected_before_dedup_tokens,
            "selected_evidence_receipt_sha256s": list(
                self.selected_evidence_receipt_sha256s
            ),
            "selection_before_global_dedup": True,
            "selection_truncated": self.selection_truncated,
            "source_diversity_policy": self.source_diversity_policy,
            "status": self.status,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _is_subsequence(values: Sequence[str], parent: Sequence[str]) -> bool:
    iterator = iter(parent)
    return all(any(candidate == value for candidate in iterator) for value in values)


@dataclass(frozen=True, slots=True)
class HotIncrementalGraphResult:
    selected_before_dedup: tuple[HotIncrementalGraphEvidence, ...]
    receipt: HotIncrementalGraphReceipt
    budget: HotIncrementalGraphBudget

    def __post_init__(self) -> None:
        _require(
            type(self.selected_before_dedup) is tuple
            and all(
                type(row) is HotIncrementalGraphEvidence
                for row in self.selected_before_dedup
            )
            and type(self.receipt) is HotIncrementalGraphReceipt
            and type(self.budget) is HotIncrementalGraphBudget,
            "graph lane result types changed",
        )
        _require(
            tuple(row.chunk_id for row in self.selected_before_dedup)
            == self.receipt.selected_before_dedup_ids
            and tuple(row.receipt_sha256 for row in self.selected_before_dedup)
            == self.receipt.selected_evidence_receipt_sha256s
            and sum(row.token_count for row in self.selected_before_dedup)
            == self.receipt.selected_before_dedup_tokens
            <= self.budget.evidence_token_cap
            and self.receipt.budget_id == self.budget.budget_id,
            "graph lane result differs from its receipt or independent budget",
        )

    @property
    def status(self) -> GraphLaneStatus:
        return self.receipt.status

    @property
    def selected_before_dedup_ids(self) -> tuple[str, ...]:
        return self.receipt.selected_before_dedup_ids

    def audit_projection(self) -> dict[str, Any]:
        value = {
            "format": RESULT_FORMAT,
            "receipt": self.receipt.projection(),
            "selected_before_dedup": [
                row.projection() for row in self.selected_before_dedup
            ],
        }
        assert_gold_blind(value, path="hot_incremental_graph_result")
        return value


def _search_receipt(
    result: Any,
    *,
    question_body_sha256: str,
) -> str:
    projection = {
        "derive_question_seeds": result.question_seed_derivation_enabled,
        "evidence": [
            {
                "chunk_id": row.chunk_id,
                "hop": row.hop,
                "matched_query_phrases": list(row.matched_query_phrases),
                "path": [_core_transition_projection(edge) for edge in row.path],
                "score": row.score,
                "supporting_seed_chunk_ids": list(row.supporting_seed_chunk_ids),
            }
            for row in result.evidence
        ],
        "graph_revision": result.graph_revision,
        "query_phrase_keys": list(result.query_phrase_keys),
        "question_body_sha256": question_body_sha256,
        "seed_chunk_ids": list(result.seed_chunk_ids),
        "traversal_policy_sha256": result.traversal_policy_sha256,
    }
    assert_gold_blind(projection, path="hot_incremental_graph_search")
    return identity_sha256(projection)


def query_hot_incremental_graph(
    index: HotIncrementalGraphIndex,
    dated_question: str,
    seed_binding: HotIncrementalGraphSeedBinding,
    /,
    *,
    budget: HotIncrementalGraphBudget = HotIncrementalGraphBudget(),
) -> HotIncrementalGraphResult:
    """Select novel graph-reached raw chunks under an independent budget."""

    _require(type(index) is HotIncrementalGraphIndex, "graph index changed")
    _require(type(budget) is HotIncrementalGraphBudget, "graph budget changed")
    _require(
        type(seed_binding) is HotIncrementalGraphSeedBinding,
        "graph lane requires a sealed upstream seed binding",
    )
    require_text(dated_question, "dated graph question")
    match = _DATED_QUESTION_RE.fullmatch(dated_question)
    _require(match is not None, "graph lane requires a dated question header")
    assert match is not None
    question_body = match.group("body")
    seeds = seed_binding.seed_chunk_ids
    _require(
        len(seeds) <= budget.max_seed_chunks,
        "graph seed population exceeds its hard bound",
    )
    missing = next(
        (chunk_id for chunk_id in seeds if chunk_id not in index.rows_by_chunk_id),
        None,
    )
    _require(missing is None, f"graph seed is absent from full store: {missing}")
    _require(
        index.graph.stats() == index.graph_stats,
        "live graph changed after the matched-eval index was sealed",
    )

    graph_policy = budget.graph_policy(seed_count=len(seeds))
    core = index.graph.search(
        question_body,
        seed_chunk_ids=seeds,
        derive_question_seeds=False,
        policy=graph_policy,
    )
    _require(
        core.graph_revision == index.graph_stats.revision
        and core.question_seed_derivation_enabled is False
        and set(core.seed_chunk_ids) == set(seeds),
        "graph search introduced an unsealed activation seed or revision",
    )
    seed_set = set(seeds)
    novel_ranked = tuple(
        row for row in core.evidence if row.chunk_id not in seed_set
    )
    _require(
        len(novel_ranked) <= budget.max_candidates
        and all(
            1 <= row.hop <= budget.max_hops
            and row.path
            and row.path[0].source_chunk_id in seed_set
            for row in novel_ranked
        ),
        "novel candidate escaped the bounded explicit-seed graph walk",
    )
    candidates = _source_round_robin(novel_ranked, index.rows_by_chunk_id)
    paths = {
        row.chunk_id: _materialize_path(index, row) for row in candidates
    }

    selected_core: list[GraphEvidence] = []
    excluded_ids: list[str] = []
    used_tokens = 0
    for candidate in candidates:
        tokens = index.rows_by_chunk_id[candidate.chunk_id].token_count
        if used_tokens + tokens > budget.evidence_token_cap:
            excluded_ids.append(candidate.chunk_id)
            continue
        selected_core.append(candidate)
        used_tokens += tokens
    selected = tuple(
        _materialize_evidence(index, row, paths[row.chunk_id])
        for row in selected_core
    )

    if not seeds:
        status: GraphLaneStatus = "no_seed"
    elif not candidates:
        status = "no_novel_graph_evidence"
    elif not selected:
        status = "novel_graph_evidence_all_over_budget"
    else:
        status = "novel_graph_evidence_selected"
    question_body_digest = quote_sha256(question_body)
    receipt = HotIncrementalGraphReceipt(
        status=status,
        index_receipt_sha256=index.receipt_sha256,
        budget_id=budget.budget_id,
        dated_question_sha256=quote_sha256(dated_question),
        question_body_sha256=question_body_digest,
        seed_binding=seed_binding,
        seed_chunk_ids=seeds,
        seed_chunk_ids_sha256=identity_sha256(list(seeds)),
        seed_append_receipt_sha256s=tuple(
            index.append_receipts[value].receipt_sha256 for value in seeds
        ),
        seed_activation_input_tokens=sum(
            index.rows_by_chunk_id[value].token_count for value in seeds
        ),
        graph_search_policy_sha256=graph_policy.policy_sha256,
        graph_search_receipt_sha256=_search_receipt(
            core,
            question_body_sha256=question_body_digest,
        ),
        graph_ranked_novel_ids=tuple(row.chunk_id for row in novel_ranked),
        candidate_population_ids=tuple(row.chunk_id for row in candidates),
        candidate_source_ids=tuple(
            index.rows_by_chunk_id[row.chunk_id].source_id for row in candidates
        ),
        candidate_path_receipt_sha256s=tuple(
            identity_sha256([step.receipt_sha256 for step in paths[row.chunk_id]])
            for row in candidates
        ),
        selected_before_dedup_ids=tuple(row.chunk_id for row in selected),
        selected_evidence_receipt_sha256s=tuple(
            row.receipt_sha256 for row in selected
        ),
        selected_before_dedup_tokens=used_tokens,
        budget_excluded_ids=tuple(excluded_ids),
        selection_truncated=bool(excluded_ids),
        graph_max_hops=budget.max_hops,
    )
    return HotIncrementalGraphResult(
        selected_before_dedup=selected,
        receipt=receipt,
        budget=budget,
    )


__all__ = [
    "DEFAULT_EVIDENCE_TOKEN_CAP",
    "GraphLaneStatus",
    "GraphOccurrenceRef",
    "HotIncrementalGraphBudget",
    "HotIncrementalGraphEvidence",
    "HotIncrementalGraphIndex",
    "HotIncrementalGraphLaneError",
    "HotIncrementalGraphPathStep",
    "HotIncrementalGraphReceipt",
    "HotIncrementalGraphResult",
    "HotIncrementalGraphSeedBinding",
    "MECHANISM_ID",
    "build_hot_incremental_graph_index",
    "query_hot_incremental_graph",
]
