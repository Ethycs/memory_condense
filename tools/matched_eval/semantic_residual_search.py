"""Provider-free terminal semantic residual retrieval.

This module is the final retrieval layer after specialist selection.  It
turns every immutable source history in a :class:`FullStoreWindowIndex` into
one or more adjacent bounded semantic cells, orders those cells with a
deterministic semantic bisection, and delegates uncertainty-preserving tree
traversal to :mod:`tools.matched_eval.semantic_binary_search`.

The classifier implemented here is deliberately conservative.  A branch is
``definitely_no`` only when a complete manifest proves a hard contradiction,
or when a declared dual gate proves both a low vector upper bound and a low
query-specific IDF/specificity upper bound.  Missing vectors or manifests are
``may_answer``.  Both uncertain children are traversed.  The complete retained
population is ranked before protected-evidence deduplication and then greedily
packed under the exact terminal evidence-plane budget.  Every unpacked novel
survivor remains explicit in the open packing frontier.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import EvidenceSpan, quote_sha256
from memory_condense.persistence.db import (
    INDEXED_CHUNK_SQL,
    Database,
)

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .full_store_slot_closure import (
    FullStoreWindowIndex,
    LocalCitationBinding,
    indexed_surface_terms,
)
from .semantic_binary_search import (
    SemanticBinarySearchResult,
    SemanticBranchDecision,
    SemanticCell,
    SemanticSearchNode,
    SemanticSearchTree,
    ProvenanceBinding,
    build_semantic_search_tree,
    make_branch_decision,
    semantic_binary_search,
    validate_semantic_binary_search_result,
)
from .typed_action_semantics import canonical_action_concepts
from .typed_numeric_semantics import numeric_mentions, single_numeric_mention
from .typed_operator_adapter import (
    ITEM_FORMAT,
    EvidenceHandleBinding,
    EvidenceOrigin,
    EvidenceStatus,
    FrontierMode,
    NumericRole,
    ParsedTypedItems,
    ProvenanceGrade,
    TypedEvidenceContribution,
    TypedEvidenceItem,
    TypedItemKind,
    parse_typed_items,
)
from .typed_operator_spec import (
    AnswerShape,
    TypedOperatorSpec,
    compile_typed_operator_spec,
)


MECHANISM_ID = "semantic_residual_terminal_branch_and_bound_v3"
TYPED_ADAPTER_MECHANISM_ID = (
    "semantic_residual_terminal_typed_adapter_packing_bounded_v1"
)
POLICY_FORMAT = "memory-condense-semantic-residual-terminal-policy-v2"
VECTOR_SET_FORMAT = "memory-condense-semantic-residual-source-vectors-v1"
CHUNK_VECTOR_SET_FORMAT = "memory-condense-semantic-residual-stored-chunk-vectors-v2"
SEGMENT_FORMAT = "memory-condense-semantic-residual-exact-segment-v1"
CELL_FORMAT = "memory-condense-semantic-residual-source-cell-v2"
NODE_MANIFEST_FORMAT = "memory-condense-semantic-residual-node-manifest-v2"
INDEX_FORMAT = "memory-condense-semantic-residual-terminal-index-v2"
QUERY_FORMAT = "memory-condense-semantic-residual-query-policy-v2"
DECISION_AUDIT_FORMAT = "memory-condense-semantic-residual-decision-audit-v2"
EVIDENCE_FORMAT = "memory-condense-semantic-residual-exact-evidence-v1"
RESULT_FORMAT = "memory-condense-semantic-residual-terminal-result-v3"
FRONTIER_FORMAT = "memory-condense-semantic-residual-classified-frontier-v2"
DUPLICATE_FORMAT = "memory-condense-semantic-residual-protected-duplicate-v1"
SEMANTIC_SEED_ALGORITHM = "deterministic-two-sweep-cosine-v1"
BOUNDED_PACKING_ALGORITHM = (
    "sealed-relevance-rank-source-temporal-diversity-"
    "post-dedup-exact-greedy-v1"
)
SOURCE_GROUP_ALLOCATION_FORMAT = (
    "memory-condense-semantic-residual-source-group-allocation-v1"
)

DEFAULT_MAX_CELL_TOKENS = 2_048
DEFAULT_PAYLOAD_TOKEN_CAP = 5_500
DEFAULT_COSINE_UPPER_BOUND_FLOOR = 0.05
DEFAULT_SPECIFICITY_UPPER_BOUND_RATIO = 0.75
_VECTOR_EPSILON = 1e-12
_BOUND_EPSILON = 1e-10
_DATED_RE = re.compile(r"^\[Question asked at (?P<asked_at>.+?)\]\s*", re.I | re.S)
_EXACT_LITERAL_RE = re.compile(
    r"\bexact\s+(?:term|phrase)\s+[\"“](?P<literal>[^\"”]{1,160})[\"”]",
    re.I,
)


class SemanticResidualSearchError(MatchedEvalContractError):
    """A residual tree, manifest, decision, hydration, or replay changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise SemanticResidualSearchError(message)


def _receipt(value: Mapping[str, object], declared: str, label: str) -> str:
    expected = identity_sha256(value)
    if declared:
        _require(require_sha256(declared, label) == expected, f"{label} changed")
    return expected


def _ordered_unique(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    _require(
        type(values) is tuple
        and all(type(value) is str and value for value in values)
        and len(set(values)) == len(values),
        f"{label} must be ordered unique text",
    )
    return values


def _finite_number(value: object, label: str) -> float:
    _require(type(value) in {int, float}, f"{label} must be an exact number")
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _normalize_vector(
    value: Sequence[float] | np.ndarray | None,
    *,
    label: str,
) -> tuple[float, ...] | None:
    if value is None:
        return None
    _require(
        not isinstance(value, (str, bytes, bytearray)),
        f"{label} changed vector type",
    )
    raw = tuple(_finite_number(item, label) for item in value)
    _require(bool(raw), f"{label} cannot be empty")
    norm = math.sqrt(math.fsum(item * item for item in raw))
    if norm <= _VECTOR_EPSILON:
        return None
    return tuple(item / norm for item in raw)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    _require(len(left) == len(right), "semantic vector dimensions changed")
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    _require(len(left) == len(right), "semantic vector dimensions changed")
    return math.sqrt(
        math.fsum((a - b) * (a - b) for a, b in zip(left, right, strict=True))
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def semantic_residual_source_identity_receipt(source_id: str) -> str:
    """Return the stable, prompt-external identity receipt for one source."""

    require_text(source_id, "semantic residual source identity")
    return identity_sha256(
        {
            "format": SOURCE_GROUP_ALLOCATION_FORMAT,
            "source_id": source_id,
        }
    )


def semantic_residual_source_group_map(
    source_ids: Sequence[str],
) -> Mapping[str, str]:
    """Allocate stable six-digit opaque groups over an exact source universe.

    The initial slot is derived from the source-identity receipt.  Collisions
    are resolved by deterministic open addressing in sorted source order.  A
    terminal renderer must pass the same full retained-source universe, so a
    changing visible R/P subset can never renumber an already packed row.
    """

    _require(
        not isinstance(source_ids, (str, bytes, bytearray)),
        "semantic residual group source population changed type",
    )
    ordered = tuple(sorted(set(source_ids)))
    _require(
        len(ordered) == len(tuple(source_ids))
        and all(type(value) is str and value for value in ordered),
        "semantic residual group source population must be ordered unique text",
    )
    slot_count = 1_000_000
    _require(
        len(ordered) <= slot_count,
        "semantic residual opaque group namespace exhausted",
    )
    occupied: dict[int, str] = {}
    allocated: dict[str, str] = {}
    for source_id in ordered:
        receipt = semantic_residual_source_identity_receipt(source_id)
        initial = int(receipt[:16], 16) % slot_count
        for displacement in range(slot_count):
            slot = (initial + displacement) % slot_count
            if slot not in occupied:
                occupied[slot] = source_id
                allocated[source_id] = f"G{slot:06d}"
                break
        else:  # pragma: no cover - guarded by the population bound above
            raise SemanticResidualSearchError(
                "semantic residual opaque group allocation failed"
            )
    _require(
        len(set(allocated.values())) == len(allocated),
        "semantic residual opaque group collision escaped allocation",
    )
    return MappingProxyType(allocated)


@dataclass(frozen=True, slots=True)
class SemanticResidualPolicy:
    """Frozen, query-independent limits for the terminal residual layer."""

    max_cell_tokens: int = DEFAULT_MAX_CELL_TOKENS
    payload_token_cap: int = DEFAULT_PAYLOAD_TOKEN_CAP
    cosine_upper_bound_floor: float = DEFAULT_COSINE_UPPER_BOUND_FLOOR
    specificity_upper_bound_ratio: float = DEFAULT_SPECIFICITY_UPPER_BOUND_RATIO
    dual_gate_enabled: bool = True
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            type(self.max_cell_tokens) is int and self.max_cell_tokens > 0,
            "semantic residual max-cell budget changed",
        )
        _require(
            type(self.payload_token_cap) is int and self.payload_token_cap > 0,
            "semantic residual payload budget changed",
        )
        floor = _finite_number(
            self.cosine_upper_bound_floor, "semantic cosine floor"
        )
        _require(-1.0 <= floor <= 1.0, "semantic cosine floor escaped [-1, 1]")
        ratio = _finite_number(
            self.specificity_upper_bound_ratio, "semantic specificity ratio"
        )
        _require(0.0 < ratio <= 1.0, "semantic specificity ratio escaped (0, 1]")
        _require(type(self.dual_gate_enabled) is bool, "dual gate flag changed")
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "semantic residual policy receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="semantic_residual_policy")

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "cosine_upper_bound_floor": self.cosine_upper_bound_floor,
            "dual_gate_enabled": self.dual_gate_enabled,
            "format": POLICY_FORMAT,
            "gold_loaded": False,
            "max_cell_tokens": self.max_cell_tokens,
            "mechanism_id": MECHANISM_ID,
            "new_provider_calls": 0,
            "payload_token_cap": self.payload_token_cap,
            "bounded_packing_algorithm": BOUNDED_PACKING_ALGORITHM,
            "retained_transformer_token_state_bytes": 0,
            "specificity_upper_bound_ratio": self.specificity_upper_bound_ratio,
            "terminal_after_specialist_selection": True,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class SourceCentroidVectorSet:
    """Normalized source centroids read from an already immutable store."""

    window_index_receipt_sha256: str
    source_vector_artifact_sha256: str
    vector_dimension: int
    vectors: Mapping[str, tuple[float, ...] | None]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.window_index_receipt_sha256, "source-vector index")
        require_sha256(self.source_vector_artifact_sha256, "source-vector artifact")
        _require(
            type(self.vector_dimension) is int and self.vector_dimension >= 0,
            "source-vector dimension changed",
        )
        _require(isinstance(self.vectors, Mapping) and self.vectors, "source vectors changed")
        normalized: dict[str, tuple[float, ...] | None] = {}
        for source_id, vector in sorted(self.vectors.items()):
            require_text(source_id, "source-vector source")
            if vector is None:
                normalized[source_id] = None
                continue
            _require(
                type(vector) is tuple
                and len(vector) == self.vector_dimension
                and self.vector_dimension > 0
                and all(type(value) is float and math.isfinite(value) for value in vector),
                "source centroid changed normalized coordinates",
            )
            norm = math.sqrt(math.fsum(value * value for value in vector))
            _require(abs(norm - 1.0) <= 1e-8, "source centroid is not normalized")
            normalized[source_id] = vector
        _require(
            self.vector_dimension > 0 or all(value is None for value in normalized.values()),
            "zero-dimensional source-vector set retained coordinates",
        )
        object.__setattr__(self, "vectors", MappingProxyType(normalized))
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "source centroid vector-set receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="semantic_source_vectors")

    @property
    def missing_source_ids(self) -> tuple[str, ...]:
        return tuple(key for key, value in self.vectors.items() if value is None)

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "format": VECTOR_SET_FORMAT,
            "gold_loaded": False,
            "missing_source_ids": list(self.missing_source_ids),
            "new_provider_calls": 0,
            "retained_transformer_token_state_bytes": 0,
            "source_count": len(self.vectors),
            "source_vector_artifact_sha256": self.source_vector_artifact_sha256,
            "vectors": [
                {"source_id": source_id, "vector": None if vector is None else list(vector)}
                for source_id, vector in self.vectors.items()
            ],
            "vector_dimension": self.vector_dimension,
            "vectorized_source_count": len(self.vectors) - len(self.missing_source_ids),
            "window_index_receipt_sha256": self.window_index_receipt_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class StoredChunkVectorSet:
    """Normalized stored chunk embeddings with exact missing coverage.

    Cells derive their own centroid from the unique chunks they contain.  A
    missing entry therefore invalidates only overlapping cells, never every
    sibling from the same coherent source history.
    """

    window_index_receipt_sha256: str
    chunk_vector_artifact_sha256: str
    vector_dimension: int
    vectors: Mapping[str, tuple[float, ...] | None]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.window_index_receipt_sha256, "chunk-vector index")
        require_sha256(self.chunk_vector_artifact_sha256, "chunk-vector artifact")
        _require(
            type(self.vector_dimension) is int and self.vector_dimension >= 0,
            "chunk-vector dimension changed",
        )
        _require(isinstance(self.vectors, Mapping) and self.vectors, "chunk vectors changed")
        normalized: dict[str, tuple[float, ...] | None] = {}
        for chunk_id, vector in sorted(self.vectors.items()):
            require_text(chunk_id, "chunk-vector chunk")
            if vector is None:
                normalized[chunk_id] = None
                continue
            _require(
                type(vector) is tuple
                and len(vector) == self.vector_dimension
                and self.vector_dimension > 0
                and all(type(value) is float and math.isfinite(value) for value in vector),
                "stored chunk vector changed normalized coordinates",
            )
            norm = math.sqrt(math.fsum(value * value for value in vector))
            _require(abs(norm - 1.0) <= 1e-8, "stored chunk vector is not normalized")
            normalized[chunk_id] = vector
        _require(
            self.vector_dimension > 0 or all(value is None for value in normalized.values()),
            "zero-dimensional chunk-vector set retained coordinates",
        )
        object.__setattr__(self, "vectors", MappingProxyType(normalized))
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "stored chunk vector-set receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="semantic_stored_chunk_vectors")

    @property
    def missing_chunk_ids(self) -> tuple[str, ...]:
        return tuple(key for key, value in self.vectors.items() if value is None)

    @property
    def source_vector_artifact_sha256(self) -> str:
        """Compatibility alias for callers that only seal the artifact."""

        return self.chunk_vector_artifact_sha256

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "chunk_count": len(self.vectors),
            "chunk_vector_artifact_sha256": self.chunk_vector_artifact_sha256,
            "format": CHUNK_VECTOR_SET_FORMAT,
            "gold_loaded": False,
            "missing_chunk_ids": list(self.missing_chunk_ids),
            "new_provider_calls": 0,
            "retained_transformer_token_state_bytes": 0,
            "vector_dimension": self.vector_dimension,
            "vector_granularity": "stored_chunk_overlap_cell_centroid_v1",
            "vectorized_chunk_count": len(self.vectors) - len(self.missing_chunk_ids),
            "vectors": [
                {"chunk_id": chunk_id, "vector": None if vector is None else list(vector)}
                for chunk_id, vector in self.vectors.items()
            ],
            "window_index_receipt_sha256": self.window_index_receipt_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def source_centroid_vector_set(
    index: FullStoreWindowIndex,
    source_centroid_vectors: Mapping[str, Sequence[float] | np.ndarray | None],
    /,
    *,
    source_vector_artifact_sha256: str | None = None,
) -> SourceCentroidVectorSet:
    """Normalize caller-supplied stored centroids and bind exact coverage."""

    _require(type(index) is FullStoreWindowIndex, "source vectors require exact index")
    _require(isinstance(source_centroid_vectors, Mapping), "source vectors require a mapping")
    source_ids = tuple(sorted({row.source_id for row in index.rows}))
    _require(
        set(source_centroid_vectors) <= set(source_ids),
        "source-vector mapping contains a source outside the index",
    )
    normalized: dict[str, tuple[float, ...] | None] = {}
    dimension = 0
    for source_id in source_ids:
        vector = _normalize_vector(
            source_centroid_vectors.get(source_id), label="stored source centroid"
        )
        if vector is not None:
            if dimension == 0:
                dimension = len(vector)
            _require(len(vector) == dimension, "stored source centroids changed dimension")
        normalized[source_id] = vector
    inventory = {
        "source_ids": list(source_ids),
        "vectors": [
            {"source_id": key, "vector": None if value is None else list(value)}
            for key, value in normalized.items()
        ],
        "window_index_receipt_sha256": index.receipt_sha256,
    }
    artifact = (
        identity_sha256(inventory)
        if source_vector_artifact_sha256 is None
        else require_sha256(source_vector_artifact_sha256, "source vector artifact")
    )
    return SourceCentroidVectorSet(
        index.receipt_sha256,
        artifact,
        dimension,
        MappingProxyType(normalized),
    )


def load_stored_chunk_vectors(
    database: Database,
    index: FullStoreWindowIndex,
    /,
) -> StoredChunkVectorSet:
    """Read immutable chunk embeddings for later exact cell-local centroids."""

    _require(
        type(database) is Database and database.read_only,
        "stored chunk-vector loading requires an exact read-only database",
    )
    _require(type(index) is FullStoreWindowIndex, "stored chunk vectors require exact index")
    rows = database.execute(
        "SELECT c.chunk_id, c.embedding "
        "FROM chunks AS c JOIN turns AS t ON t.turn_id = c.turn_id "
        f"WHERE {INDEXED_CHUNK_SQL} "
        "ORDER BY c.chunk_id"
    ).fetchall()
    indexed_chunk_ids = tuple(sorted({row.chunk_id for row in index.rows}))
    indexed_chunk_set = set(indexed_chunk_ids)
    raw_by_chunk: dict[str, bytes] = {}
    parsed_by_chunk: dict[str, np.ndarray] = {}
    dimension_counts: dict[int, int] = {}
    for raw_chunk_id, raw_blob in rows:
        chunk_id = str(raw_chunk_id)
        if chunk_id not in indexed_chunk_set:
            continue
        _require(chunk_id not in raw_by_chunk, "stored chunk vector repeated a chunk")
        blob = bytes(raw_blob)
        raw_by_chunk[chunk_id] = blob
        try:
            vector = np.frombuffer(blob, dtype=np.float32).astype(np.float64)
        except ValueError:
            continue
        if (
            vector.ndim != 1
            or vector.size == 0
            or not bool(np.isfinite(vector).all())
        ):
            continue
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= _VECTOR_EPSILON:
            continue
        dimension = int(vector.size)
        parsed_by_chunk[chunk_id] = vector / norm
        dimension_counts[dimension] = dimension_counts.get(dimension, 0) + 1
    dimension = (
        min(
            dimension_counts,
            key=lambda value: (-dimension_counts[value], value),
        )
        if dimension_counts
        else 0
    )
    vectors: dict[str, tuple[float, ...] | None] = {}
    for chunk_id in indexed_chunk_ids:
        vector = parsed_by_chunk.get(chunk_id)
        vectors[chunk_id] = (
            None
            if vector is None or int(vector.size) != dimension
            else tuple(float(item) for item in vector)
        )
    artifact = identity_sha256(
        {
            "cache_receipt_sha256": index.cache.cache_receipt_sha256,
            "chunk_embedding_rows_read": len(rows),
            "chunk_embedding_rows": [
                {
                    "chunk_id": chunk_id,
                    "embedding_blob_sha256": hashlib.sha256(blob).hexdigest(),
                    "embedding_byte_count": len(blob),
                }
                for chunk_id, blob in sorted(raw_by_chunk.items())
            ],
            "source_database_sha256": index.cache.source_database_sha256,
            "stored_chunk_vectors": [
                {"chunk_id": key, "vector": None if value is None else list(value)}
                for key, value in vectors.items()
            ],
            "window_index_receipt_sha256": index.receipt_sha256,
        }
    )
    return StoredChunkVectorSet(
        index.receipt_sha256,
        artifact,
        dimension,
        MappingProxyType(vectors),
    )


def load_stored_source_centroid_vectors(
    database: Database,
    index: FullStoreWindowIndex,
    /,
) -> StoredChunkVectorSet:
    """Compatibility entry point; v2 returns stored chunk-granularity vectors."""

    return load_stored_chunk_vectors(database, index)


@dataclass(frozen=True, slots=True)
class ExactCellSegment:
    """One exact contiguous quote inside an authoritative cached row."""

    source_id: str
    partition_id: str
    span: EvidenceSpan
    quote: str
    quote_sha256: str
    token_count: int
    role: str
    created_at: str
    surface_terms: tuple[str, ...]
    action_concepts: tuple[str, ...]
    event_dates: tuple[str, ...]
    has_undated_window: bool
    contains_numeric_value: bool
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_text(self.source_id, "semantic segment source")
        require_text(self.partition_id, "semantic segment partition")
        _require(type(self.span) is EvidenceSpan, "semantic segment span changed")
        _require(
            self.span.source_id == self.source_id,
            "semantic segment span changed source",
        )
        require_text(self.quote, "semantic segment quote")
        require_sha256(self.quote_sha256, "semantic segment quote")
        _require(
            self.quote_sha256 == quote_sha256(self.quote)
            and self.span.quote_sha256 == self.quote_sha256
            and type(self.token_count) is int
            and self.token_count == count_tokens(self.quote),
            "semantic segment exact quote changed",
        )
        require_text(self.role, "semantic segment role")
        require_text(self.created_at, "semantic segment created-at")
        for values, label in (
            (self.surface_terms, "semantic segment terms"),
            (self.action_concepts, "semantic segment actions"),
            (self.event_dates, "semantic segment dates"),
        ):
            _ordered_unique(values, label)
        _require(
            type(self.has_undated_window) is bool
            and type(self.contains_numeric_value) is bool,
            "semantic segment flags changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "semantic segment receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "action_concepts": list(self.action_concepts),
            "contains_numeric_value": self.contains_numeric_value,
            "created_at": self.created_at,
            "event_dates": list(self.event_dates),
            "format": SEGMENT_FORMAT,
            "has_undated_window": self.has_undated_window,
            "partition_id": self.partition_id,
            "quote": self.quote,
            "quote_sha256": self.quote_sha256,
            "role": self.role,
            "source_id": self.source_id,
            "span": self.span.identity_payload(),
            "surface_terms": list(self.surface_terms),
            "token_count": self.token_count,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class SemanticResidualCell:
    """One bounded adjacent piece of an otherwise coherent source history."""

    core_cell: SemanticCell
    source_id: str
    source_history_receipt_sha256: str
    source_cell_ordinal: int
    source_cell_count: int
    segments: tuple[ExactCellSegment, ...]
    normalized_source_centroid: tuple[float, ...] | None
    vector_derivation: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(type(self.core_cell) is SemanticCell, "residual core cell changed")
        require_text(self.source_id, "residual cell source")
        require_sha256(self.source_history_receipt_sha256, "source history receipt")
        _require(
            type(self.source_cell_ordinal) is int
            and type(self.source_cell_count) is int
            and 0 <= self.source_cell_ordinal < self.source_cell_count,
            "source cell ordinal changed",
        )
        _require(
            type(self.segments) is tuple
            and self.segments
            and all(type(row) is ExactCellSegment for row in self.segments)
            and all(row.source_id == self.source_id for row in self.segments),
            "residual cell segments changed",
        )
        expected_text = "\n".join(row.quote for row in self.segments)
        _require(
            self.core_cell.text == expected_text
            and len(self.core_cell.provenance) == 1
            and self.core_cell.provenance[0].source_id == self.source_id
            and self.core_cell.provenance[0].source_receipt_sha256
            == self.source_history_receipt_sha256,
            "residual cell lost source coherence",
        )
        if self.normalized_source_centroid is not None:
            _require(
                type(self.normalized_source_centroid) is tuple
                and self.normalized_source_centroid,
                "residual cell vector changed",
            )
            norm = math.sqrt(
                math.fsum(value * value for value in self.normalized_source_centroid)
            )
            _require(abs(norm - 1.0) <= 1e-8, "residual cell vector is not normalized")
        _require(
            self.vector_derivation
            in {
                "stored_unique_overlapping_chunk_centroid_v2",
                "legacy_explicit_source_centroid_compatibility_v1",
            },
            "residual cell vector derivation changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "semantic residual cell receipt",
            ),
        )

    @property
    def cell_id(self) -> str:
        return self.core_cell.cell_id

    @property
    def surface_terms(self) -> tuple[str, ...]:
        return tuple(sorted({term for row in self.segments for term in row.surface_terms}))

    @property
    def action_concepts(self) -> tuple[str, ...]:
        return tuple(sorted({term for row in self.segments for term in row.action_concepts}))

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "core_cell": self.core_cell.projection(),
            "format": CELL_FORMAT,
            "normalized_source_centroid": (
                None
                if self.normalized_source_centroid is None
                else list(self.normalized_source_centroid)
            ),
            "segments": [row.projection() for row in self.segments],
            "source_cell_count": self.source_cell_count,
            "source_cell_ordinal": self.source_cell_ordinal,
            "source_history_receipt_sha256": self.source_history_receipt_sha256,
            "source_id": self.source_id,
            "vector_derivation": self.vector_derivation,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _bounded_text_slices(text: str, max_tokens: int) -> tuple[tuple[int, int], ...]:
    """Partition exact text into adjacent nonempty slices under ``max_tokens``."""

    require_text(text, "semantic source row text")
    _require(type(max_tokens) is int and max_tokens > 0, "slice budget changed")
    if count_tokens(text) <= max_tokens:
        return ((0, len(text)),)
    output: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        low = start + 1
        high = len(text)
        best = start
        while low <= high:
            midpoint = (low + high) // 2
            if count_tokens(text[start:midpoint]) <= max_tokens:
                best = midpoint
                low = midpoint + 1
            else:
                high = midpoint - 1
        _require(best > start, "one source character exceeds the token cell budget")
        if best < len(text) and (
            text[best - 1].isspace() or text[best].isspace()
        ):
            # Keep exact adjacent coverage while preventing an exact quote
            # from gaining synthetic leading/trailing whitespace at a split.
            # Moving the boundary backward cannot violate the token cap.
            candidate = best
            while candidate > start + 1 and (
                text[candidate - 1].isspace() or text[candidate].isspace()
            ):
                candidate -= 1
            if candidate > start:
                best = candidate
        output.append((start, best))
        start = best
    _require(
        output[0][0] == 0
        and output[-1][1] == len(text)
        and all(left[1] == right[0] for left, right in zip(output, output[1:], strict=False))
        and "".join(text[start:end] for start, end in output) == text
        and all(count_tokens(text[start:end]) <= max_tokens for start, end in output),
        "bounded source slicing lost exact adjacent coverage",
    )
    return tuple(output)


def _segments_by_source(
    index: FullStoreWindowIndex,
    *,
    max_tokens: int,
) -> dict[str, tuple[ExactCellSegment, ...]]:
    windows_by_chunk: dict[str, list[Any]] = defaultdict(list)
    for window in index.windows:
        windows_by_chunk[window.row.chunk_id].append(window)
    rows_by_source: dict[str, list[Any]] = defaultdict(list)
    for row in index.rows:
        rows_by_source[row.source_id].append(row)
    result: dict[str, tuple[ExactCellSegment, ...]] = {}
    for source_id, unordered in sorted(rows_by_source.items()):
        rows = sorted(
            unordered,
            key=lambda row: (
                row.ordinal,
                row.turn_start_char,
                row.turn_end_char,
                row.chunk_id,
            ),
        )
        segments: list[ExactCellSegment] = []
        for row in rows:
            for start, end in _bounded_text_slices(row.text, max_tokens):
                quote = row.text[start:end]
                overlaps = tuple(
                    window
                    for window in windows_by_chunk[row.chunk_id]
                    if window.start_char < end and start < window.end_char
                )
                _require(bool(overlaps), "exact source slice escaped sentence windows")
                event_dates = tuple(
                    sorted(
                        {
                            window.event_date
                            for window in overlaps
                            if window.event_date is not None
                            and window.event_date_basis == "explicit_text_date"
                        }
                    )
                )
                span = EvidenceSpan(
                    chunk_id=row.chunk_id,
                    start_char=start,
                    end_char=end,
                    quote_sha256=quote_sha256(quote),
                    ordinal=row.ordinal,
                    source_id=row.source_id,
                    turn_start_char=row.turn_start_char,
                    turn_id=row.turn_id,
                    role=row.role,
                    created_at=row.created_at,
                )
                segments.append(
                    ExactCellSegment(
                        source_id=row.source_id,
                        partition_id=row.partition_id,
                        span=span,
                        quote=quote,
                        quote_sha256=quote_sha256(quote),
                        token_count=count_tokens(quote),
                        role=row.role,
                        created_at=row.created_at,
                        surface_terms=tuple(sorted(set(indexed_surface_terms(quote)))),
                        action_concepts=canonical_action_concepts(quote),
                        event_dates=event_dates,
                        # ``row_created_at`` remains separately available as
                        # derived chronology; it is not a textual event date.
                        has_undated_window=any(
                            window.event_date_basis != "explicit_text_date"
                            for window in overlaps
                        ),
                        contains_numeric_value=bool(numeric_mentions(quote)),
                    )
                )
        _require(bool(segments), "source history produced no exact segments")
        result[source_id] = tuple(segments)
    return result


def _source_history_receipt(source_id: str, segments: Sequence[ExactCellSegment]) -> str:
    return identity_sha256(
        {
            "format": f"{CELL_FORMAT}-source-history",
            "ordered_segment_receipt_sha256s": [row.receipt_sha256 for row in segments],
            "source_id": source_id,
        }
    )


def _source_cells(
    index: FullStoreWindowIndex,
    vectors: SourceCentroidVectorSet | StoredChunkVectorSet,
    policy: SemanticResidualPolicy,
) -> tuple[SemanticResidualCell, ...]:
    segments_by_source = _segments_by_source(index, max_tokens=policy.max_cell_tokens)
    cells: list[SemanticResidualCell] = []
    for source_id, segments in segments_by_source.items():
        groups: list[tuple[ExactCellSegment, ...]] = []
        current: list[ExactCellSegment] = []
        for segment in segments:
            proposed = (*current, segment)
            rendered = "\n".join(row.quote for row in proposed)
            if current and count_tokens(rendered) > policy.max_cell_tokens:
                groups.append(tuple(current))
                current = [segment]
            else:
                current.append(segment)
        if current:
            groups.append(tuple(current))
        _require(
            groups
            and tuple(row for group in groups for row in group) == segments
            and all(
                count_tokens("\n".join(row.quote for row in group))
                <= policy.max_cell_tokens
                for group in groups
            ),
            "source-history cell split changed adjacency or budget",
        )
        source_receipt = _source_history_receipt(source_id, segments)
        coherence_id = identity_sha256(
            {
                "namespace_id": index.cache.namespace_id,
                "source_history_receipt_sha256": source_receipt,
            }
        )
        for ordinal, group in enumerate(groups):
            cell_id = "C" + identity_sha256(
                {
                    "coherence_id": coherence_id,
                    "ordered_segment_receipt_sha256s": [
                        row.receipt_sha256 for row in group
                    ],
                    "source_cell_ordinal": ordinal,
                }
            )[:24]
            text = "\n".join(row.quote for row in group)
            core = SemanticCell(
                cell_id,
                coherence_id,
                text,
                count_tokens(text),
                (ProvenanceBinding(source_id, source_receipt),),
            )
            if type(vectors) is StoredChunkVectorSet:
                chunk_ids = tuple(
                    dict.fromkeys(segment.span.chunk_id for segment in group)
                )
                chunk_vectors = tuple(vectors.vectors[chunk_id] for chunk_id in chunk_ids)
                if not chunk_vectors or any(value is None for value in chunk_vectors):
                    cell_centroid = None
                else:
                    present = tuple(value for value in chunk_vectors if value is not None)
                    _require(
                        all(len(value) == vectors.vector_dimension for value in present),
                        "cell-local chunk embeddings changed common dimension",
                    )
                    mean = tuple(
                        math.fsum(value[position] for value in present) / len(present)
                        for position in range(vectors.vector_dimension)
                    )
                    cell_centroid = _normalize_vector(
                        mean, label="stored cell-local chunk centroid"
                    )
                vector_derivation = "stored_unique_overlapping_chunk_centroid_v2"
            else:
                cell_centroid = vectors.vectors[source_id]
                vector_derivation = "legacy_explicit_source_centroid_compatibility_v1"
            cells.append(
                SemanticResidualCell(
                    core,
                    source_id,
                    source_receipt,
                    ordinal,
                    len(groups),
                    group,
                    cell_centroid,
                    vector_derivation,
                )
            )
    _require(
        cells
        and len({row.cell_id for row in cells}) == len(cells),
        "semantic source cells changed identity",
    )
    return tuple(cells)


def _farthest_seed_pair(
    values: Sequence[SemanticResidualCell],
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    """Return deterministic approximate farthest seeds in O(n*d).

    The previous exhaustive Python pair loop was O(n^2*d).  At roughly five
    hundred 1024-dimensional source histories per namespace it dominated the
    otherwise provider-free path.  Two cosine sweeps retain a semantic axis,
    have deterministic ID tie-breaking, and are named in the sealed policy.
    """

    available = tuple(sorted(
        (row for row in values if row.normalized_source_centroid is not None),
        key=lambda row: row.cell_id,
    ))
    if len(available) < 2:
        return None
    matrix = np.asarray(
        [row.normalized_source_centroid for row in available], dtype=np.float64
    )
    first_index = int(np.argmin(matrix @ matrix[0]))
    second_scores = matrix @ matrix[first_index]
    second_scores[first_index] = math.inf
    second_index = int(np.argmin(second_scores))
    left = available[first_index]
    right = available[second_index]
    assert left.normalized_source_centroid is not None
    assert right.normalized_source_centroid is not None
    if right.cell_id < left.cell_id:
        left, right = right, left
    return left.normalized_source_centroid, right.normalized_source_centroid


def _farthest_seed_pair_scalar_reference(
    values: Sequence[SemanticResidualCell],
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    """Slow small-fixture oracle for the vectorized two-sweep implementation."""

    available = tuple(sorted(
        (row for row in values if row.normalized_source_centroid is not None),
        key=lambda row: row.cell_id,
    ))
    if len(available) < 2:
        return None
    anchor = available[0].normalized_source_centroid
    assert anchor is not None
    first = min(
        available,
        key=lambda row: (_dot(row.normalized_source_centroid or (), anchor), row.cell_id),
    )
    first_vector = first.normalized_source_centroid
    assert first_vector is not None
    second = min(
        (row for row in available if row.cell_id != first.cell_id),
        key=lambda row: (
            _dot(row.normalized_source_centroid or (), first_vector),
            row.cell_id,
        ),
    )
    left, right = sorted((first, second), key=lambda row: row.cell_id)
    assert left.normalized_source_centroid is not None
    assert right.normalized_source_centroid is not None
    return left.normalized_source_centroid, right.normalized_source_centroid


def _semantic_leaf_order(
    cells: Sequence[SemanticResidualCell],
) -> tuple[SemanticResidualCell, ...]:
    """Recursively bisect equal-sized semantic halves by farthest-seed axis."""

    values = tuple(cells)
    if len(values) <= 1:
        return values
    pair = _farthest_seed_pair(values)
    if pair is None:
        ranked = tuple(sorted(values, key=lambda row: row.cell_id))
    else:
        left, right = pair
        raw_axis = tuple(a - b for a, b in zip(left, right, strict=True))
        axis = _normalize_vector(raw_axis, label="semantic bisection axis")
        if axis is None:
            ranked = tuple(sorted(values, key=lambda row: row.cell_id))
        else:
            ranked = tuple(
                sorted(
                    values,
                    key=lambda row: (
                        row.normalized_source_centroid is None,
                        0.0
                        if row.normalized_source_centroid is None
                        else _dot(row.normalized_source_centroid, axis),
                        row.cell_id,
                    ),
                )
            )
    midpoint = len(ranked) // 2
    return (
        *_semantic_leaf_order(ranked[:midpoint]),
        *_semantic_leaf_order(ranked[midpoint:]),
    )


@dataclass(frozen=True, slots=True)
class SemanticNodeManifest:
    """Complete symbolic inventory and certified vector ball for one node."""

    node_receipt_sha256: str
    cell_receipt_sha256s: tuple[str, ...]
    normalized_centroid: tuple[float, ...] | None
    radius: float | None
    vector_complete: bool
    tag_manifest_complete: bool
    roles: tuple[str, ...]
    created_at_min: str
    created_at_max: str
    event_dates: tuple[str, ...]
    has_undated_window: bool
    contains_numeric_value: bool
    surface_terms: tuple[str, ...]
    action_concepts: tuple[str, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.node_receipt_sha256, "node manifest node")
        _ordered_unique(self.cell_receipt_sha256s, "node manifest cells")
        _require(bool(self.cell_receipt_sha256s), "node manifest lost cells")
        for value in self.cell_receipt_sha256s:
            require_sha256(value, "node manifest cell")
        _require(
            type(self.vector_complete) is bool
            and type(self.tag_manifest_complete) is bool,
            "node manifest completeness flags changed",
        )
        if self.normalized_centroid is None:
            _require(self.radius is None and not self.vector_complete, "missing centroid claimed coverage")
        else:
            _require(
                type(self.normalized_centroid) is tuple
                and self.normalized_centroid
                and self.radius is not None,
                "node centroid/radius changed",
            )
            norm = math.sqrt(math.fsum(value * value for value in self.normalized_centroid))
            _require(abs(norm - 1.0) <= 1e-8, "node centroid is not normalized")
            radius = _finite_number(self.radius, "node vector radius")
            _require(0.0 <= radius <= 2.0 + 1e-8, "node vector radius changed")
        for values, label in (
            (self.roles, "node manifest roles"),
            (self.event_dates, "node manifest dates"),
            (self.surface_terms, "node manifest terms"),
            (self.action_concepts, "node manifest actions"),
        ):
            _ordered_unique(values, label)
        require_text(self.created_at_min, "node created-at minimum")
        require_text(self.created_at_max, "node created-at maximum")
        _require(
            self.created_at_min <= self.created_at_max
            and type(self.has_undated_window) is bool
            and type(self.contains_numeric_value) is bool,
            "node manifest dates/flags changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "semantic node manifest receipt",
            ),
        )

    def cosine_upper_bound(
        self, query_vectors: Sequence[tuple[float, ...]]
    ) -> float | None:
        if not self.vector_complete or self.normalized_centroid is None or self.radius is None:
            return None
        vectors = tuple(query_vectors)
        if not vectors:
            return None
        if any(len(row) != len(self.normalized_centroid) for row in vectors):
            return None
        return max(
            min(1.0, _dot(vector, self.normalized_centroid) + self.radius + _BOUND_EPSILON)
            for vector in vectors
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "action_concepts": list(self.action_concepts),
            "cell_receipt_sha256s": list(self.cell_receipt_sha256s),
            "contains_numeric_value": self.contains_numeric_value,
            "created_at_max": self.created_at_max,
            "created_at_min": self.created_at_min,
            "event_dates": list(self.event_dates),
            "format": NODE_MANIFEST_FORMAT,
            "has_undated_window": self.has_undated_window,
            "node_receipt_sha256": self.node_receipt_sha256,
            "normalized_centroid": (
                None if self.normalized_centroid is None else list(self.normalized_centroid)
            ),
            "radius": self.radius,
            "roles": list(self.roles),
            "surface_terms": list(self.surface_terms),
            "tag_manifest_complete": self.tag_manifest_complete,
            "vector_complete": self.vector_complete,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _node_manifest(
    node: SemanticSearchNode,
    by_cell_id: Mapping[str, SemanticResidualCell],
) -> SemanticNodeManifest:
    cells = tuple(by_cell_id[row.cell_id] for row in node.cells)
    vectors = tuple(row.normalized_source_centroid for row in cells)
    complete = bool(vectors) and all(row is not None for row in vectors)
    centroid: tuple[float, ...] | None = None
    radius: float | None = None
    if complete:
        present = tuple(row for row in vectors if row is not None)
        dimension = len(present[0])
        _require(all(len(row) == dimension for row in present), "node vectors changed dimension")
        mean = tuple(
            math.fsum(row[index] for row in present) / len(present)
            for index in range(dimension)
        )
        centroid = _normalize_vector(mean, label="node mean centroid")
        if centroid is None:
            centroid = present[0]
        radius = max(_distance(vector, centroid) for vector in present)
    segments = tuple(segment for cell in cells for segment in cell.segments)
    return SemanticNodeManifest(
        node_receipt_sha256=node.receipt_sha256,
        cell_receipt_sha256s=tuple(row.receipt_sha256 for row in cells),
        normalized_centroid=centroid,
        radius=radius,
        vector_complete=complete,
        tag_manifest_complete=True,
        roles=tuple(sorted({row.role for row in segments})),
        created_at_min=min(row.created_at for row in segments),
        created_at_max=max(row.created_at for row in segments),
        event_dates=tuple(sorted({date for row in segments for date in row.event_dates})),
        has_undated_window=any(row.has_undated_window for row in segments),
        contains_numeric_value=any(row.contains_numeric_value for row in segments),
        surface_terms=tuple(sorted({term for row in segments for term in row.surface_terms})),
        action_concepts=tuple(sorted({term for row in segments for term in row.action_concepts})),
    )


@dataclass(frozen=True, slots=True)
class SemanticResidualIndex:
    """Shared question-neutral tree and compact receipt inventory per namespace."""

    window_index_receipt_sha256: str
    namespace_id: str
    cache_receipt_sha256: str
    source_database_sha256: str
    source_store_receipt_sha256: str
    policy: SemanticResidualPolicy
    source_vectors: SourceCentroidVectorSet | StoredChunkVectorSet
    cells: tuple[SemanticResidualCell, ...]
    core_tree: SemanticSearchTree
    node_manifests: tuple[SemanticNodeManifest, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.window_index_receipt_sha256, "residual index window index"),
            (self.namespace_id, "residual index namespace"),
            (self.cache_receipt_sha256, "residual index cache"),
            (self.source_database_sha256, "residual index database"),
            (self.source_store_receipt_sha256, "residual index store"),
        ):
            require_sha256(value, label)
        _require(type(self.policy) is SemanticResidualPolicy, "residual index policy changed")
        _require(
            type(self.source_vectors) in {SourceCentroidVectorSet, StoredChunkVectorSet},
            "residual index vectors changed",
        )
        _require(
            type(self.cells) is tuple
            and self.cells
            and all(type(row) is SemanticResidualCell for row in self.cells)
            and type(self.core_tree) is SemanticSearchTree
            and self.core_tree.cells == tuple(row.core_cell for row in self.cells),
            "residual index cell/tree population changed",
        )
        _require(
            type(self.node_manifests) is tuple
            and len(self.node_manifests) == len(self.core_tree.nodes)
            and all(type(row) is SemanticNodeManifest for row in self.node_manifests)
            and tuple(row.node_receipt_sha256 for row in self.node_manifests)
            == tuple(row.receipt_sha256 for row in self.core_tree.nodes),
            "residual index node-manifest population changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "semantic residual index receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="semantic_residual_index")

    @property
    def cell_by_id(self) -> Mapping[str, SemanticResidualCell]:
        return MappingProxyType({row.cell_id: row for row in self.cells})

    @property
    def manifest_by_node_receipt(self) -> Mapping[str, SemanticNodeManifest]:
        return MappingProxyType(
            {row.node_receipt_sha256: row for row in self.node_manifests}
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        """Compact shared projection; exact corpus text remains in the store."""

        value: dict[str, object] = {
            "cache_receipt_sha256": self.cache_receipt_sha256,
            "cell_count": len(self.cells),
            "cell_receipt_sha256s": [row.receipt_sha256 for row in self.cells],
            "core_tree_receipt_sha256": self.core_tree.receipt_sha256,
            "format": INDEX_FORMAT,
            "gold_loaded": False,
            "new_provider_calls": 0,
            "namespace_id": self.namespace_id,
            "node_manifest_receipt_sha256s": [
                row.receipt_sha256 for row in self.node_manifests
            ],
            "ordered_leaf_cell_ids": [row.cell_id for row in self.cells],
            "policy": self.policy.projection(),
            "retained_transformer_token_state_bytes": 0,
            "segment_count": sum(len(row.segments) for row in self.cells),
            "source_database_sha256": self.source_database_sha256,
            "source_store_receipt_sha256": self.source_store_receipt_sha256,
            "source_vector_set_receipt_sha256": self.source_vectors.receipt_sha256,
            "source_vector_artifact_sha256": (
                self.source_vectors.source_vector_artifact_sha256
            ),
            "vector_granularity": (
                "stored_chunk_cell_centroid_v2"
                if type(self.source_vectors) is StoredChunkVectorSet
                else "legacy_source_centroid_compatibility_v1"
            ),
            "semantic_seed_algorithm": SEMANTIC_SEED_ALGORITHM,
            "terminal_after_specialist_selection": True,
            "window_index_receipt_sha256": self.window_index_receipt_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value

    def local_manifest_projection(self) -> dict[str, object]:
        """Optional shared audit material, never duplicated per question."""

        return {
            "cells": [row.projection() for row in self.cells],
            "compact_index_receipt_sha256": self.receipt_sha256,
            "core_tree": self.core_tree.projection(),
            "node_manifests": [row.projection() for row in self.node_manifests],
            "source_vectors": self.source_vectors.projection(),
        }


def build_semantic_residual_index(
    index: FullStoreWindowIndex,
    source_centroid_vectors: (
        SourceCentroidVectorSet
        | StoredChunkVectorSet
        | Mapping[str, Sequence[float] | np.ndarray | None]
    ),
    /,
    *,
    policy: SemanticResidualPolicy = SemanticResidualPolicy(),
    source_vector_artifact_sha256: str | None = None,
) -> SemanticResidualIndex:
    """Build one shared full-population semantic tree for a namespace."""

    _require(type(index) is FullStoreWindowIndex, "semantic residual requires exact index")
    _require(type(policy) is SemanticResidualPolicy, "semantic residual policy changed")
    if type(source_centroid_vectors) in {SourceCentroidVectorSet, StoredChunkVectorSet}:
        vector_set = source_centroid_vectors
        _require(
            source_vector_artifact_sha256 is None
            and vector_set.window_index_receipt_sha256 == index.receipt_sha256,
            "source-vector set escaped the full-store index",
        )
    else:
        vector_set = source_centroid_vector_set(
            index,
            source_centroid_vectors,
            source_vector_artifact_sha256=source_vector_artifact_sha256,
        )
    if type(vector_set) is StoredChunkVectorSet:
        _require(
            set(vector_set.vectors) == {row.chunk_id for row in index.rows},
            "stored chunk vectors lost full chunk coverage",
        )
    else:
        _require(
            set(vector_set.vectors) == {row.source_id for row in index.rows},
            "source vectors lost full source coverage",
        )
    unordered = _source_cells(index, vector_set, policy)
    cells = _semantic_leaf_order(unordered)
    _require(
        set(row.receipt_sha256 for row in cells)
        == set(row.receipt_sha256 for row in unordered)
        and len(cells) == len(unordered),
        "semantic leaf ordering lost a full-store cell",
    )
    tree = build_semantic_search_tree(tuple(row.core_cell for row in cells))
    by_id = {row.cell_id: row for row in cells}
    manifests = tuple(_node_manifest(node, by_id) for node in tree.nodes)
    return SemanticResidualIndex(
        window_index_receipt_sha256=index.receipt_sha256,
        namespace_id=index.cache.namespace_id,
        cache_receipt_sha256=index.cache.cache_receipt_sha256,
        source_database_sha256=index.cache.source_database_sha256,
        source_store_receipt_sha256=index.cache.source_store_receipt_sha256,
        policy=policy,
        source_vectors=vector_set,
        cells=cells,
        core_tree=tree,
        node_manifests=manifests,
    )


def _query_facets_from_spec(
    dated_question: str,
    operator_spec: TypedOperatorSpec,
) -> tuple[str, ...]:
    body = _DATED_RE.sub("", dated_question).strip()
    candidates: list[str] = [body]
    candidates.extend(
        f"{slot.label}: {' '.join(slot.match_terms)}"
        for slot in operator_spec.required_slots
    )
    actions = canonical_action_concepts(body)
    if actions:
        candidates.append("actions: " + " ".join(actions))
    return tuple(dict.fromkeys(value for value in candidates if value))


def semantic_residual_query_facets(dated_question: str, /) -> tuple[str, ...]:
    """Return the exact question-only facet texts that callers should embed."""

    require_text(dated_question, "semantic residual dated question")
    spec = compile_typed_operator_spec(dated_question)
    return _query_facets_from_spec(dated_question, spec)


@dataclass(frozen=True, slots=True)
class SemanticResidualQuery:
    """Sealed question-only policy and optional frozen facet embeddings."""

    residual_index_receipt_sha256: str
    dated_question: str
    operator_spec: TypedOperatorSpec
    facet_texts: tuple[str, ...]
    query_terms: tuple[str, ...]
    slot_terms: tuple[str, ...]
    action_concepts: tuple[str, ...]
    exact_literals: tuple[str, ...]
    query_vectors: tuple[tuple[float, ...] | None, ...]
    query_vector_dimension: int
    query_vector_complete: bool
    query_vector_artifact_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.residual_index_receipt_sha256, "residual query index")
        require_text(self.dated_question, "semantic residual dated question")
        _require(
            type(self.operator_spec) is TypedOperatorSpec
            and self.operator_spec.question_sha256 == quote_sha256(self.dated_question),
            "semantic residual operator spec escaped its question",
        )
        _ordered_unique(self.facet_texts, "semantic query facets")
        _require(
            self.facet_texts
            == _query_facets_from_spec(self.dated_question, self.operator_spec),
            "semantic residual facet texts changed",
        )
        for values, label in (
            (self.query_terms, "semantic query terms"),
            (self.slot_terms, "semantic slot terms"),
            (self.action_concepts, "semantic query actions"),
            (self.exact_literals, "semantic exact literals"),
        ):
            _ordered_unique(values, label)
        _require(
            type(self.query_vectors) is tuple
            and len(self.query_vectors) == len(self.facet_texts)
            and type(self.query_vector_dimension) is int
            and self.query_vector_dimension >= 0,
            "semantic query vector inventory changed",
        )
        for vector in self.query_vectors:
            if vector is None:
                continue
            _require(
                type(vector) is tuple
                and len(vector) == self.query_vector_dimension
                and all(type(value) is float and math.isfinite(value) for value in vector),
                "semantic query vector changed",
            )
            norm = math.sqrt(math.fsum(value * value for value in vector))
            _require(abs(norm - 1.0) <= 1e-8, "semantic query vector is not normalized")
        expected_complete = bool(self.query_vectors) and all(
            vector is not None for vector in self.query_vectors
        )
        _require(
            type(self.query_vector_complete) is bool
            and self.query_vector_complete == expected_complete
            and (
                self.query_vector_dimension > 0
                if self.query_vector_complete
                else True
            ),
            "semantic query vector completeness changed",
        )
        require_sha256(self.query_vector_artifact_sha256, "semantic query vector artifact")
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "semantic residual query receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="semantic_residual_query")

    @property
    def present_query_vectors(self) -> tuple[tuple[float, ...], ...]:
        return tuple(vector for vector in self.query_vectors if vector is not None)

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "action_concepts": list(self.action_concepts),
            "dated_question": self.dated_question,
            "exact_literals": list(self.exact_literals),
            "facet_texts": list(self.facet_texts),
            "format": QUERY_FORMAT,
            "gold_loaded": False,
            "new_provider_calls": 0,
            "operator_spec_receipt_sha256": self.operator_spec.receipt_sha256,
            "query_terms": list(self.query_terms),
            "query_vector_artifact_sha256": self.query_vector_artifact_sha256,
            "query_vector_complete": self.query_vector_complete,
            "query_vector_dimension": self.query_vector_dimension,
            "query_vector_sha256s": [
                None if vector is None else identity_sha256({"vector": list(vector)})
                for vector in self.query_vectors
            ],
            "residual_index_receipt_sha256": self.residual_index_receipt_sha256,
            "retained_transformer_token_state_bytes": 0,
            "slot_terms": list(self.slot_terms),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value

    def local_vector_projection(self) -> dict[str, object]:
        """Exact query-vector artifact, stored once outside result projections."""

        return {
            "facet_texts": list(self.facet_texts),
            "query_vector_artifact_sha256": self.query_vector_artifact_sha256,
            "vectors": [None if row is None else list(row) for row in self.query_vectors],
        }


def compile_semantic_residual_query(
    residual_index: SemanticResidualIndex,
    dated_question: str,
    /,
    *,
    query_vectors: Sequence[Sequence[float] | np.ndarray | None] = (),
    query_vector_artifact_sha256: str | None = None,
) -> SemanticResidualQuery:
    """Compile a gold-blind query; empty/missing vectors conservatively stay MAY."""

    _require(type(residual_index) is SemanticResidualIndex, "semantic query index changed")
    require_text(dated_question, "semantic residual dated question")
    spec = compile_typed_operator_spec(dated_question)
    facets = _query_facets_from_spec(dated_question, spec)
    supplied = tuple(query_vectors)
    if supplied:
        _require(len(supplied) == len(facets), "semantic query vectors lost facet alignment")
        normalized = tuple(
            _normalize_vector(vector, label="semantic query facet vector")
            for vector in supplied
        )
    else:
        normalized = tuple(None for _ in facets)
    present = tuple(row for row in normalized if row is not None)
    dimension = len(present[0]) if present else 0
    _require(
        all(len(row) == dimension for row in present),
        "semantic query facet vectors changed dimension",
    )
    _require(
        not present
        or residual_index.source_vectors.vector_dimension in {0, dimension},
        "semantic query/source vector dimensions differ",
    )
    vector_inventory = {
        "facet_texts": list(facets),
        "residual_index_receipt_sha256": residual_index.receipt_sha256,
        "vectors": [None if row is None else list(row) for row in normalized],
    }
    artifact = (
        identity_sha256(vector_inventory)
        if query_vector_artifact_sha256 is None
        else require_sha256(query_vector_artifact_sha256, "query vector artifact")
    )
    body = _DATED_RE.sub("", dated_question).strip()
    terms = tuple(sorted(set(indexed_surface_terms(body))))
    slot_terms = tuple(
        sorted({term for slot in spec.required_slots for term in slot.match_terms})
    )
    actions = canonical_action_concepts(body)
    exact_literals = tuple(
        sorted({match.group("literal").strip().casefold() for match in _EXACT_LITERAL_RE.finditer(body)})
    )
    return SemanticResidualQuery(
        residual_index_receipt_sha256=residual_index.receipt_sha256,
        dated_question=dated_question,
        operator_spec=spec,
        facet_texts=facets,
        query_terms=terms,
        slot_terms=slot_terms,
        action_concepts=actions,
        exact_literals=exact_literals,
        query_vectors=normalized,
        query_vector_dimension=dimension,
        query_vector_complete=bool(normalized) and all(row is not None for row in normalized),
        query_vector_artifact_sha256=artifact,
    )


@dataclass(frozen=True, slots=True)
class SemanticResidualDecisionAudit:
    decision_receipt_sha256: str
    node_manifest_receipt_sha256: str
    query_receipt_sha256: str
    reason: str
    cosine_upper_bound: float | None
    intersecting_surface_terms: tuple[str, ...]
    intersecting_action_concepts: tuple[str, ...]
    missing_required_role: str | None
    absent_exact_literals: tuple[str, ...]
    vector_gate_available: bool
    tag_gate_available: bool
    query_specificity_receipt_sha256: str
    node_specificity_upper_bound: float | None
    max_leaf_specificity: float
    specificity_threshold: float
    specificity_gate_available: bool
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.decision_receipt_sha256, "residual audit decision"),
            (self.node_manifest_receipt_sha256, "residual audit manifest"),
            (self.query_receipt_sha256, "residual audit query"),
            (self.query_specificity_receipt_sha256, "residual query specificity"),
        ):
            require_sha256(value, label)
        _require(
            self.reason in {
                "required_role_absent",
                "exact_literal_absent",
                "dual_gate",
                "may_answer",
            },
            "semantic residual audit reason changed",
        )
        if self.cosine_upper_bound is not None:
            _finite_number(self.cosine_upper_bound, "residual cosine upper bound")
        if self.node_specificity_upper_bound is not None:
            _require(
                _finite_number(
                    self.node_specificity_upper_bound,
                    "residual node specificity upper bound",
                )
                >= 0.0,
                "residual node specificity upper bound is negative",
            )
        _require(
            _finite_number(self.max_leaf_specificity, "residual max leaf specificity")
            >= 0.0
            and _finite_number(self.specificity_threshold, "residual specificity threshold")
            >= 0.0,
            "residual specificity scores changed",
        )
        for values, label in (
            (self.intersecting_surface_terms, "residual intersecting terms"),
            (self.intersecting_action_concepts, "residual intersecting actions"),
            (self.absent_exact_literals, "residual absent literals"),
        ):
            _ordered_unique(values, label)
        if self.missing_required_role is not None:
            _require(
                self.missing_required_role in {"user", "assistant"},
                "residual missing role changed",
            )
        _require(
            type(self.vector_gate_available) is bool
            and type(self.tag_gate_available) is bool,
            "residual gate availability changed",
        )
        _require(
            type(self.specificity_gate_available) is bool,
            "residual specificity gate availability changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "semantic residual decision audit receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "absent_exact_literals": list(self.absent_exact_literals),
            "cosine_upper_bound": self.cosine_upper_bound,
            "decision_receipt_sha256": self.decision_receipt_sha256,
            "format": DECISION_AUDIT_FORMAT,
            "intersecting_action_concepts": list(self.intersecting_action_concepts),
            "intersecting_surface_terms": list(self.intersecting_surface_terms),
            "missing_required_role": self.missing_required_role,
            "max_leaf_specificity": self.max_leaf_specificity,
            "node_manifest_receipt_sha256": self.node_manifest_receipt_sha256,
            "node_specificity_upper_bound": self.node_specificity_upper_bound,
            "query_receipt_sha256": self.query_receipt_sha256,
            "query_specificity_receipt_sha256": self.query_specificity_receipt_sha256,
            "reason": self.reason,
            "specificity_gate_available": self.specificity_gate_available,
            "specificity_threshold": self.specificity_threshold,
            "tag_gate_available": self.tag_gate_available,
            "vector_gate_available": self.vector_gate_available,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


class _ConservativeResidualClassifier:
    def __init__(self, index: SemanticResidualIndex, query: SemanticResidualQuery) -> None:
        self.index = index
        self.query = query
        query_surface = tuple(sorted(set(query.query_terms) | set(query.slot_terms)))
        query_actions = tuple(sorted(set(query.action_concepts)))
        leaf_count = len(index.cells)
        self.term_specificity = {
            term: math.log(
                (leaf_count + 1)
                / (
                    sum(term in set(cell.surface_terms) for cell in index.cells)
                    + 1
                )
            )
            for term in query_surface
        }
        self.action_specificity = {
            action: math.log(
                (leaf_count + 1)
                / (
                    sum(action in set(cell.action_concepts) for cell in index.cells)
                    + 1
                )
            )
            for action in query_actions
        }
        leaf_scores = tuple(
            math.fsum(
                (
                    *(
                        weight
                        for term, weight in self.term_specificity.items()
                        if term in set(cell.surface_terms)
                    ),
                    *(
                        weight
                        for action, weight in self.action_specificity.items()
                        if action in set(cell.action_concepts)
                    ),
                )
            )
            for cell in index.cells
        )
        self.max_leaf_specificity = max(leaf_scores, default=0.0)
        self.specificity_threshold = (
            index.policy.specificity_upper_bound_ratio
            * self.max_leaf_specificity
        )
        specificity_body = {
            "format": f"{DECISION_AUDIT_FORMAT}-query-specificity-v1",
            "leaf_cell_count": leaf_count,
            "max_leaf_specificity": self.max_leaf_specificity,
            "query_receipt_sha256": query.receipt_sha256,
            "residual_index_receipt_sha256": index.receipt_sha256,
            "signal_rows": [
                {
                    "idf": weight,
                    "kind": "surface_term",
                    "leaf_cell_df": sum(
                        term in set(cell.surface_terms) for cell in index.cells
                    ),
                    "signal": term,
                }
                for term, weight in self.term_specificity.items()
            ]
            + [
                {
                    "idf": weight,
                    "kind": "action_concept",
                    "leaf_cell_df": sum(
                        action in set(cell.action_concepts) for cell in index.cells
                    ),
                    "signal": action,
                }
                for action, weight in self.action_specificity.items()
            ],
            "specificity_threshold": self.specificity_threshold,
            "specificity_upper_bound_ratio": (
                index.policy.specificity_upper_bound_ratio
            ),
        }
        self.query_specificity_receipt_sha256 = identity_sha256(specificity_body)
        self.classifier_id = (
            f"{MECHANISM_ID}:conservative:{query.receipt_sha256[:16]}:"
            f"{self.query_specificity_receipt_sha256[:16]}"
        )
        self.audits: list[SemanticResidualDecisionAudit] = []

    def classify(
        self,
        *,
        question: str,
        node: SemanticSearchNode,
        call_ordinal: int,
    ) -> SemanticBranchDecision:
        _require(question == self.query.dated_question, "classifier question changed")
        manifest = self.index.manifest_by_node_receipt[node.receipt_sha256]
        required_role = self.query.operator_spec.required_evidence_role
        missing_role = (
            required_role
            if required_role is not None and required_role not in manifest.roles
            else None
        )
        if self.query.exact_literals:
            node_text = "\n".join(cell.text for cell in node.cells).casefold()
            absent_literals = tuple(
                literal for literal in self.query.exact_literals if literal not in node_text
            )
        else:
            absent_literals = ()
        query_surface = set(self.query.query_terms) | set(self.query.slot_terms)
        term_overlap = tuple(sorted(query_surface & set(manifest.surface_terms)))
        action_overlap = tuple(
            sorted(set(self.query.action_concepts) & set(manifest.action_concepts))
        )
        upper = manifest.cosine_upper_bound(self.query.present_query_vectors)
        vector_available = bool(
            self.query.query_vector_complete
            and manifest.vector_complete
            and upper is not None
        )
        tag_available = manifest.tag_manifest_complete
        specificity_upper = (
            math.fsum(
                (
                    *(
                        self.term_specificity[term]
                        for term in term_overlap
                    ),
                    *(
                        self.action_specificity[action]
                        for action in action_overlap
                    ),
                )
            )
            if tag_available
            else None
        )
        specificity_available = bool(
            tag_available
            and (self.term_specificity or self.action_specificity)
            and specificity_upper is not None
        )
        if missing_role is not None:
            classification: Literal["definitely_no", "may_answer"] = "definitely_no"
            reason = "required_role_absent"
        elif absent_literals:
            classification = "definitely_no"
            reason = "exact_literal_absent"
        elif (
            self.index.policy.dual_gate_enabled
            and vector_available
            and tag_available
            and upper is not None
            and upper < self.index.policy.cosine_upper_bound_floor
            and specificity_available
            and specificity_upper is not None
            and specificity_upper < self.specificity_threshold
        ):
            classification = "definitely_no"
            reason = "dual_gate"
        else:
            classification = "may_answer"
            reason = "may_answer"
        decision = make_branch_decision(
            classifier_id=self.classifier_id,
            question_sha256=quote_sha256(question),
            node_receipt_sha256=node.receipt_sha256,
            call_ordinal=call_ordinal,
            branch_classification=classification,
        )
        self.audits.append(
            SemanticResidualDecisionAudit(
                decision_receipt_sha256=decision.receipt_sha256,
                node_manifest_receipt_sha256=manifest.receipt_sha256,
                query_receipt_sha256=self.query.receipt_sha256,
                reason=reason,
                cosine_upper_bound=upper,
                intersecting_surface_terms=term_overlap,
                intersecting_action_concepts=action_overlap,
                missing_required_role=missing_role,
                absent_exact_literals=absent_literals,
                vector_gate_available=vector_available,
                tag_gate_available=tag_available,
                query_specificity_receipt_sha256=(
                    self.query_specificity_receipt_sha256
                ),
                node_specificity_upper_bound=specificity_upper,
                max_leaf_specificity=self.max_leaf_specificity,
                specificity_threshold=self.specificity_threshold,
                specificity_gate_available=specificity_available,
            )
        )
        return decision


@dataclass(frozen=True, slots=True)
class SemanticResidualEvidence:
    """Provider-visible exact survivor with prompt-external local provenance."""

    candidate_id: str
    cell_id: str
    segment_receipt_sha256: str
    source_group_handle: str
    quote: str
    quote_sha256: str
    token_count: int
    role: str
    created_at: str
    event_dates: tuple[str, ...]
    contains_numeric_value: bool
    matched_query_terms: tuple[str, ...]
    matched_action_concepts: tuple[str, ...]
    citation_binding_receipt_sha256: str
    packing_protection: Literal["must_include"] = "must_include"
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.candidate_id, "semantic residual candidate")
        require_text(self.cell_id, "semantic residual evidence cell")
        require_sha256(self.segment_receipt_sha256, "semantic residual segment")
        _require(
            re.fullmatch(r"G[0-9]{4,6}", self.source_group_handle) is not None,
            "semantic residual source group changed",
        )
        require_text(self.quote, "semantic residual quote")
        require_sha256(self.quote_sha256, "semantic residual quote")
        _require(
            self.quote_sha256 == quote_sha256(self.quote)
            and type(self.token_count) is int
            and self.token_count == count_tokens(self.quote),
            "semantic residual evidence lost exact quote bytes",
        )
        require_text(self.role, "semantic residual evidence role")
        require_text(self.created_at, "semantic residual evidence created-at")
        for values, label in (
            (self.event_dates, "semantic residual evidence dates"),
            (self.matched_query_terms, "semantic residual matched terms"),
            (self.matched_action_concepts, "semantic residual matched actions"),
        ):
            _ordered_unique(values, label)
        _require(
            type(self.contains_numeric_value) is bool
            and self.packing_protection == "must_include",
            "semantic residual evidence flags changed",
        )
        require_sha256(
            self.citation_binding_receipt_sha256,
            "semantic residual citation binding",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "semantic residual evidence receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "candidate_id": self.candidate_id,
            "cell_id": self.cell_id,
            "citation_binding_receipt_sha256": self.citation_binding_receipt_sha256,
            "contains_numeric_value": self.contains_numeric_value,
            "created_at": self.created_at,
            "event_dates": list(self.event_dates),
            "format": EVIDENCE_FORMAT,
            "matched_action_concepts": list(self.matched_action_concepts),
            "matched_query_terms": list(self.matched_query_terms),
            "packing_protection": self.packing_protection,
            "quote": self.quote,
            "quote_sha256": self.quote_sha256,
            "role": self.role,
            "segment_receipt_sha256": self.segment_receipt_sha256,
            "source_group_handle": self.source_group_handle,
            "token_count": self.token_count,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class SemanticResidualProtectedDuplicate:
    """A selected survivor omitted only because an immutable owner has it."""

    cell_id: str
    segment_receipt_sha256: str
    span_identity_sha256: str
    protected_candidate_id: str
    protected_binding_receipt_sha256: str
    reason: Literal["exact_span_already_in_protected_evidence"] = (
        "exact_span_already_in_protected_evidence"
    )
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_text(self.cell_id, "semantic duplicate cell")
        for value, label in (
            (self.segment_receipt_sha256, "semantic duplicate segment"),
            (self.span_identity_sha256, "semantic duplicate span"),
            (self.protected_candidate_id, "semantic duplicate protected candidate"),
            (self.protected_binding_receipt_sha256, "semantic duplicate owner"),
        ):
            require_sha256(value, label)
        _require(
            self.reason == "exact_span_already_in_protected_evidence",
            "semantic duplicate reason changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "semantic residual duplicate receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "cell_id": self.cell_id,
            "format": DUPLICATE_FORMAT,
            "protected_binding_receipt_sha256": self.protected_binding_receipt_sha256,
            "protected_candidate_id": self.protected_candidate_id,
            "reason": self.reason,
            "segment_receipt_sha256": self.segment_receipt_sha256,
            "span_identity_sha256": self.span_identity_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class SemanticResidualAttemptedSelection:
    """Prompt-external provenance retained even when payload packing fails."""

    cell_id: str
    source_id: str
    segment_receipt_sha256: str
    disposition: Literal["novel", "protected_exact_duplicate"]
    candidate_id: str
    local_binding_receipt_sha256: str
    evidence_receipt_sha256: str | None
    protected_duplicate_receipt_sha256: str | None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_text(self.cell_id, "attempted semantic cell")
        require_text(self.source_id, "attempted semantic source")
        for value, label in (
            (self.segment_receipt_sha256, "attempted semantic segment"),
            (self.candidate_id, "attempted semantic candidate"),
            (self.local_binding_receipt_sha256, "attempted semantic binding"),
        ):
            require_sha256(value, label)
        _require(
            self.disposition in {"novel", "protected_exact_duplicate"},
            "attempted semantic disposition changed",
        )
        if self.disposition == "novel":
            require_sha256(self.evidence_receipt_sha256, "attempted semantic evidence")
            _require(
                self.protected_duplicate_receipt_sha256 is None,
                "novel attempted selection carries a duplicate receipt",
            )
        else:
            require_sha256(
                self.protected_duplicate_receipt_sha256,
                "attempted protected duplicate",
            )
            _require(
                self.evidence_receipt_sha256 is None,
                "protected attempted selection carries residual evidence",
            )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "attempted semantic selection receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "candidate_id": self.candidate_id,
            "cell_id": self.cell_id,
            "disposition": self.disposition,
            "evidence_receipt_sha256": self.evidence_receipt_sha256,
            "format": f"{RESULT_FORMAT}-attempted-selection-v1",
            "local_binding_receipt_sha256": self.local_binding_receipt_sha256,
            "protected_duplicate_receipt_sha256": (
                self.protected_duplicate_receipt_sha256
            ),
            "segment_receipt_sha256": self.segment_receipt_sha256,
            "source_id": self.source_id,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class ClassifiedResidualFrontierReceipt:
    """Complete leaf partition plus exact survivor packing/owner coverage."""

    residual_index_receipt_sha256: str
    core_result_receipt_sha256: str
    retained_leaf_cell_ids: tuple[str, ...]
    certified_negative_leaf_cell_ids: tuple[str, ...]
    retained_segment_receipt_sha256s: tuple[str, ...]
    packed_segment_receipt_sha256s: tuple[str, ...]
    protected_duplicate_segment_receipt_sha256s: tuple[str, ...]
    protected_duplicate_audit_receipt_sha256s: tuple[str, ...]
    unresolved_segment_receipt_sha256s: tuple[str, ...]
    classified_leaf_count: int
    complete_leaf_partition: Literal[True] = True
    all_novel_survivors_protected: bool = False
    closed: bool = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.residual_index_receipt_sha256, "classified frontier index")
        require_sha256(self.core_result_receipt_sha256, "classified frontier core")
        for values, label in (
            (self.retained_leaf_cell_ids, "classified retained cells"),
            (self.certified_negative_leaf_cell_ids, "classified negative cells"),
            (self.retained_segment_receipt_sha256s, "classified retained segments"),
            (self.packed_segment_receipt_sha256s, "classified packed segments"),
            (
                self.protected_duplicate_segment_receipt_sha256s,
                "classified duplicate segments",
            ),
            (
                self.protected_duplicate_audit_receipt_sha256s,
                "classified duplicate audits",
            ),
            (self.unresolved_segment_receipt_sha256s, "classified unresolved segments"),
        ):
            _ordered_unique(values, label)
        _require(
            type(self.classified_leaf_count) is int
            and self.classified_leaf_count > 0
            and self.classified_leaf_count
            == len(self.retained_leaf_cell_ids)
            + len(self.certified_negative_leaf_cell_ids)
            and set(self.retained_leaf_cell_ids).isdisjoint(
                self.certified_negative_leaf_cell_ids
            ),
            "classified frontier lost its leaf partition",
        )
        retained = set(self.retained_segment_receipt_sha256s)
        packed = set(self.packed_segment_receipt_sha256s)
        duplicates = set(self.protected_duplicate_segment_receipt_sha256s)
        unresolved = set(self.unresolved_segment_receipt_sha256s)
        _require(
            packed.isdisjoint(duplicates)
            and packed.isdisjoint(unresolved)
            and duplicates.isdisjoint(unresolved)
            and packed | duplicates | unresolved == retained,
            "classified survivor coverage changed",
        )
        expected_closed = not unresolved
        # Vacuous-empty and all-duplicate frontiers have no novel survivor
        # outside protected ownership.  Packed R rows are novel by definition,
        # so packing all of them closes packing without making them protected.
        expected_all_novel_protected = not packed and not unresolved
        _require(
            self.complete_leaf_partition is True
            and type(self.all_novel_survivors_protected) is bool
            and self.all_novel_survivors_protected
            == expected_all_novel_protected
            and type(self.closed) is bool
            and self.closed == expected_closed,
            "classified frontier closure is not justified",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "classified residual frontier receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "all_novel_survivors_protected": self.all_novel_survivors_protected,
            "certified_negative_leaf_cell_ids": list(
                self.certified_negative_leaf_cell_ids
            ),
            "classified_leaf_count": self.classified_leaf_count,
            "closed": self.closed,
            "complete_leaf_partition": True,
            "core_result_receipt_sha256": self.core_result_receipt_sha256,
            "format": FRONTIER_FORMAT,
            "packed_segment_receipt_sha256s": list(
                self.packed_segment_receipt_sha256s
            ),
            "protected_duplicate_audit_receipt_sha256s": list(
                self.protected_duplicate_audit_receipt_sha256s
            ),
            "protected_duplicate_segment_receipt_sha256s": list(
                self.protected_duplicate_segment_receipt_sha256s
            ),
            "residual_index_receipt_sha256": self.residual_index_receipt_sha256,
            "retained_leaf_cell_ids": list(self.retained_leaf_cell_ids),
            "retained_segment_receipt_sha256s": list(
                self.retained_segment_receipt_sha256s
            ),
            "unresolved_segment_receipt_sha256s": list(
                self.unresolved_segment_receipt_sha256s
            ),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _span_identity_sha256(span: EvidenceSpan) -> str:
    return identity_sha256(span.identity_payload())


def _provider_payload(
    dated_question: str,
    evidence: Sequence[SemanticResidualEvidence],
    frontier: ClassifiedResidualFrontierReceipt,
    *,
    fallback_reason: str,
) -> dict[str, object]:
    return {
        "dated_question": dated_question,
        "evidence": [
            {
                "date": (
                    row.event_dates[0]
                    if len(row.event_dates) == 1
                    else row.created_at
                ),
                "date_basis": (
                    "textual_event_date"
                    if len(row.event_dates) == 1
                    else "source_created_at"
                ),
                "evidence_handle": f"R{index:04d}",
                "quote": row.quote,
                "role": row.role,
                "source_group_handle": row.source_group_handle,
            }
            for index, row in enumerate(evidence, start=1)
        ],
        "format": f"{RESULT_FORMAT}-provider-payload",
        "residual_frontier": {
            "all_novel_survivors_protected": (
                frontier.all_novel_survivors_protected
            ),
            "closed": frontier.closed,
            "complete_leaf_partition": frontier.complete_leaf_partition,
            "fallback_reason": fallback_reason,
            "receipt_sha256": frontier.receipt_sha256,
        },
    }


@dataclass(frozen=True, slots=True)
class SemanticResidualSearchResult:
    """Compact per-question result; only retained novel quotes carry text."""

    residual_index_receipt_sha256: str
    query: SemanticResidualQuery
    core_result: SemanticBinarySearchResult
    decision_audits: tuple[SemanticResidualDecisionAudit, ...]
    protected_evidence_population_receipt_sha256: str
    protected_duplicates: tuple[SemanticResidualProtectedDuplicate, ...]
    attempted_selection: tuple[SemanticResidualAttemptedSelection, ...]
    evidence: tuple[SemanticResidualEvidence, ...]
    local_bindings: tuple[LocalCitationBinding, ...]
    classified_frontier: ClassifiedResidualFrontierReceipt
    attempted_evidence_count: int
    attempted_provider_payload_tokens: int
    provider_payload_tokens: int
    packed_residual_evidence_tokens: int
    packed_residual_evidence_sha256: str
    residual_evidence_token_cap: int
    fallback_required: bool
    fallback_reason: Literal["none", "zero_packable_novel_evidence"]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.residual_index_receipt_sha256, "residual result index")
        _require(
            type(self.query) is SemanticResidualQuery
            and self.query.residual_index_receipt_sha256
            == self.residual_index_receipt_sha256,
            "residual result query escaped its index",
        )
        _require(type(self.core_result) is SemanticBinarySearchResult, "residual core changed")
        _require(
            type(self.decision_audits) is tuple
            and all(type(row) is SemanticResidualDecisionAudit for row in self.decision_audits)
            and tuple(row.decision_receipt_sha256 for row in self.decision_audits)
            == tuple(row.receipt_sha256 for row in self.core_result.decisions),
            "residual result decision audit changed",
        )
        require_sha256(
            self.protected_evidence_population_receipt_sha256,
            "protected evidence population",
        )
        for values, expected, label in (
            (
                self.protected_duplicates,
                SemanticResidualProtectedDuplicate,
                "residual protected duplicates",
            ),
            (
                self.attempted_selection,
                SemanticResidualAttemptedSelection,
                "residual attempted selection",
            ),
            (self.evidence, SemanticResidualEvidence, "residual evidence"),
            (self.local_bindings, LocalCitationBinding, "residual local bindings"),
        ):
            _require(
                type(values) is tuple and all(type(row) is expected for row in values),
                f"{label} changed type",
            )
        _require(
            tuple(row.candidate_id for row in self.evidence)
            == tuple(row.candidate_id for row in self.local_bindings)
            and all(
                evidence.citation_binding_receipt_sha256 == binding.receipt_sha256
                and evidence.source_group_handle == binding.source_group_handle
                and evidence.quote_sha256 == binding.quote_sha256
                for evidence, binding in zip(self.evidence, self.local_bindings, strict=True)
            ),
            "residual evidence lost LocalCitationBinding alignment",
        )
        _require(
            type(self.classified_frontier) is ClassifiedResidualFrontierReceipt
            and self.classified_frontier.residual_index_receipt_sha256
            == self.residual_index_receipt_sha256
            and self.classified_frontier.core_result_receipt_sha256
            == self.core_result.receipt_sha256,
            "residual result changed classified frontier",
        )
        _require(
            tuple(row.segment_receipt_sha256 for row in self.attempted_selection)
            == self.classified_frontier.retained_segment_receipt_sha256s,
            "residual attempted selection lost retained-segment order",
        )
        for value, label in (
            (self.attempted_evidence_count, "attempted evidence count"),
            (self.attempted_provider_payload_tokens, "attempted provider tokens"),
            (self.provider_payload_tokens, "provider payload tokens"),
            (
                self.packed_residual_evidence_tokens,
                "packed residual evidence tokens",
            ),
            (self.residual_evidence_token_cap, "residual evidence token cap"),
        ):
            _require(type(value) is int and value >= 0, f"residual {label} changed")
        _require(
            sum(row.disposition == "novel" for row in self.attempted_selection)
            == self.attempted_evidence_count,
            "residual attempted evidence count changed",
        )
        _require(
            tuple(row.segment_receipt_sha256 for row in self.evidence)
            == self.classified_frontier.packed_segment_receipt_sha256s
            and tuple(
                row.segment_receipt_sha256 for row in self.protected_duplicates
            )
            == self.classified_frontier.protected_duplicate_segment_receipt_sha256s,
            "residual packed/protected frontier population changed",
        )
        attempted_novel = tuple(
            row
            for row in self.attempted_selection
            if row.disposition == "novel"
        )
        attempted_protected = tuple(
            row
            for row in self.attempted_selection
            if row.disposition == "protected_exact_duplicate"
        )
        _require(
            {
                row.segment_receipt_sha256 for row in attempted_novel
            }
            == set(self.classified_frontier.packed_segment_receipt_sha256s)
            | set(self.classified_frontier.unresolved_segment_receipt_sha256s)
            and tuple(
                row.segment_receipt_sha256 for row in attempted_protected
            )
            == tuple(
                row.segment_receipt_sha256
                for row in self.protected_duplicates
            ),
            "residual attempted dispositions changed frontier partition",
        )
        attempted_by_segment = {
            row.segment_receipt_sha256: row for row in self.attempted_selection
        }
        _require(
            all(
                attempted_by_segment[evidence.segment_receipt_sha256].disposition
                == "novel"
                and attempted_by_segment[
                    evidence.segment_receipt_sha256
                ].candidate_id
                == evidence.candidate_id
                and attempted_by_segment[
                    evidence.segment_receipt_sha256
                ].local_binding_receipt_sha256
                == binding.receipt_sha256
                and attempted_by_segment[
                    evidence.segment_receipt_sha256
                ].evidence_receipt_sha256
                == evidence.receipt_sha256
                for evidence, binding in zip(
                    self.evidence, self.local_bindings, strict=True
                )
            )
            and all(
                attempted.protected_duplicate_receipt_sha256
                == duplicate.receipt_sha256
                and attempted.candidate_id == duplicate.protected_candidate_id
                and attempted.local_binding_receipt_sha256
                == duplicate.protected_binding_receipt_sha256
                for attempted, duplicate in zip(
                    attempted_protected,
                    self.protected_duplicates,
                    strict=True,
                )
            ),
            "residual attempted disposition lost its exact row ownership",
        )
        _require(
            type(self.fallback_required) is bool
            and self.fallback_required
            == (self.attempted_evidence_count > 0 and not self.evidence)
            and self.fallback_reason
            == (
                "zero_packable_novel_evidence"
                if self.fallback_required
                else "none"
            ),
            "residual fallback/bounded-packing contract changed",
        )
        if self.fallback_required:
            _require(
                not self.evidence and not self.local_bindings,
                "residual fallback silently sent a top-k subset",
            )
        else:
            _require(
                len(self.evidence) <= self.attempted_evidence_count,
                "bounded residual result gained a novel survivor",
            )
        _require(
            self.packed_residual_evidence_tokens
            == _terminal_residual_evidence_tokens(self.evidence)
            <= self.residual_evidence_token_cap
            and self.residual_evidence_token_cap > 0,
            "residual exact evidence-plane token accounting changed",
        )
        _require(
            require_sha256(
                self.packed_residual_evidence_sha256,
                "packed residual evidence plane",
            )
            == semantic_residual_terminal_evidence_sha256(self.evidence),
            "residual exact evidence-plane serialization changed",
        )
        expected_payload_tokens = count_tokens(_canonical_json(self.provider_projection()))
        _require(
            expected_payload_tokens == self.provider_payload_tokens,
            "residual provider payload token accounting changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "semantic residual result receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="semantic_residual_result")
        assert_gold_blind(self.provider_projection(), path="semantic_residual_provider")

    @property
    def frontier_closed(self) -> bool:
        """Legacy alias for packing closure, not semantic support closure."""

        return self.classified_frontier.closed

    @property
    def packing_frontier_closed(self) -> bool:
        """Whether every retained MAY segment was packed or visibly protected."""

        return self.classified_frontier.closed

    def provider_projection(self) -> dict[str, object]:
        return _provider_payload(
            self.query.dated_question,
            self.evidence,
            self.classified_frontier,
            fallback_reason=self.fallback_reason,
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "attempted_evidence_count": self.attempted_evidence_count,
            "attempted_provider_payload_tokens": self.attempted_provider_payload_tokens,
            "attempted_selection_receipt_sha256": identity_sha256(
                {
                    "format": f"{RESULT_FORMAT}-attempted-selection-population-v1",
                    "row_receipt_sha256s": [
                        row.receipt_sha256 for row in self.attempted_selection
                    ],
                }
            ),
            "classified_frontier": self.classified_frontier.projection(),
            "core_result": self.core_result.projection(),
            "decision_audits": [row.projection() for row in self.decision_audits],
            "dedup_after_semantic_selection": True,
            "evidence": [row.projection() for row in self.evidence],
            "fallback_reason": self.fallback_reason,
            "fallback_required": self.fallback_required,
            "format": RESULT_FORMAT,
            "gold_loaded": False,
            "local_binding_receipt_sha256s": [
                row.receipt_sha256 for row in self.local_bindings
            ],
            "new_provider_calls": 0,
            "protected_evidence_mutated": False,
            "protected_evidence_population_receipt_sha256": (
                self.protected_evidence_population_receipt_sha256
            ),
            "protected_duplicates": [
                row.projection() for row in self.protected_duplicates
            ],
            "provider_payload_tokens": self.provider_payload_tokens,
            "packed_residual_evidence_tokens": (
                self.packed_residual_evidence_tokens
            ),
            "packed_residual_evidence_sha256": (
                self.packed_residual_evidence_sha256
            ),
            "residual_evidence_token_cap": self.residual_evidence_token_cap,
            "bounded_packing_algorithm": BOUNDED_PACKING_ALGORITHM,
            "query_receipt_sha256": self.query.receipt_sha256,
            "query_vector_artifact_sha256": self.query.query_vector_artifact_sha256,
            "residual_index_receipt_sha256": self.residual_index_receipt_sha256,
            "retained_transformer_token_state_bytes": 0,
            "searched_complete_memory_population": True,
            "terminal_after_specialist_selection": True,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value

    def local_audit_projection(self) -> dict[str, object]:
        return {
            "attempted_selection": [
                row.projection() for row in self.attempted_selection
            ],
            "classified_frontier": self.classified_frontier.projection(),
            "compact_result_receipt_sha256": self.receipt_sha256,
            "local_bindings": [row.projection() for row in self.local_bindings],
            "protected_duplicates": [
                row.projection() for row in self.protected_duplicates
            ],
            "query": self.query.projection(),
        }


def _protected_evidence_inventory(
    residual_index: SemanticResidualIndex,
    protected_evidence: Sequence[LocalCitationBinding],
) -> tuple[tuple[LocalCitationBinding, ...], Mapping[str, LocalCitationBinding], str]:
    bindings = tuple(protected_evidence)
    _require(
        all(type(row) is LocalCitationBinding for row in bindings),
        "protected residual evidence requires exact LocalCitationBinding owners",
    )
    for row in bindings:
        _require(
            row.namespace_id == residual_index.namespace_id
            and row.cache_receipt_sha256 == residual_index.cache_receipt_sha256
            and row.source_database_sha256 == residual_index.source_database_sha256
            and row.source_store_receipt_sha256 == residual_index.source_store_receipt_sha256,
            "protected residual owner escaped the immutable source lineage",
        )
    ordered = tuple(sorted(bindings, key=lambda row: row.receipt_sha256))
    by_span: dict[str, LocalCitationBinding] = {}
    for row in ordered:
        by_span.setdefault(_span_identity_sha256(row.span), row)
    population_receipt = identity_sha256(
        {
            "format": f"{DUPLICATE_FORMAT}-protected-population",
            "local_binding_receipt_sha256s": [row.receipt_sha256 for row in ordered],
            "residual_index_receipt_sha256": residual_index.receipt_sha256,
        }
    )
    return ordered, MappingProxyType(by_span), population_receipt


def semantic_residual_protected_evidence_population_receipt(
    residual_index: SemanticResidualIndex,
    protected_evidence: Sequence[LocalCitationBinding],
) -> str:
    """Seal the exact protected-owner population used by post-selection dedup."""

    return _protected_evidence_inventory(
        residual_index,
        protected_evidence,
    )[2]


def semantic_residual_terminal_evidence_rows(
    evidence: Sequence[SemanticResidualEvidence],
) -> tuple[dict[str, object], ...]:
    """Render the exact provider-visible R rows in their sealed order."""

    _require(
        not isinstance(evidence, (str, bytes, bytearray))
        and all(type(row) is SemanticResidualEvidence for row in evidence),
        "semantic residual terminal evidence changed type",
    )
    return tuple(
        {
            "created_at": row.created_at,
            "event_dates": list(row.event_dates),
            "evidence_handle": f"R{ordinal:04d}",
            "quote": row.quote,
            "role": row.role,
            "source_group_handle": row.source_group_handle,
        }
        for ordinal, row in enumerate(evidence, start=1)
    )


def semantic_residual_terminal_evidence_projection(
    evidence: Sequence[SemanticResidualEvidence],
) -> dict[str, object]:
    """Exact terminal residual plane used by the non-borrowable 2,400 cap."""

    return {
        "residual_evidence": list(
            semantic_residual_terminal_evidence_rows(evidence)
        )
    }


def semantic_residual_terminal_evidence_sha256(
    evidence: Sequence[SemanticResidualEvidence],
) -> str:
    return hashlib.sha256(
        _canonical_json(
            semantic_residual_terminal_evidence_projection(evidence)
        ).encode("utf-8")
    ).hexdigest()


def _terminal_residual_evidence_tokens(
    evidence: Sequence[SemanticResidualEvidence],
) -> int:
    return count_tokens(
        _canonical_json(semantic_residual_terminal_evidence_projection(evidence))
    )


def _created_at_timestamp(value: str) -> float:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _rank_retained_pairs(
    pairs: Sequence[tuple[SemanticResidualCell, ExactCellSegment]],
    *,
    classifier: _ConservativeResidualClassifier,
    query: SemanticResidualQuery,
) -> tuple[tuple[SemanticResidualCell, ExactCellSegment], ...]:
    """Rank the complete selected set before protected-evidence deduplication."""

    query_terms = set(query.query_terms) | set(query.slot_terms)
    query_actions = set(query.action_concepts)
    exact_literals = tuple(value.casefold() for value in query.exact_literals)
    numeric_needed = bool(
        query.operator_spec.answer_shape in {AnswerShape.NUMBER, AnswerShape.DURATION}
        or any(slot.requires_numeric for slot in query.operator_spec.required_slots)
    )
    temporal_needed = query.operator_spec.temporal_mode.value != "none"
    maximum_specificity = max(classifier.max_leaf_specificity, 1e-12)
    entries: list[dict[str, object]] = []
    for cell, segment in pairs:
        terms = set(segment.surface_terms)
        actions = set(segment.action_concepts)
        semantic = (
            max(
                (
                    _dot(cell.normalized_source_centroid, vector)
                    for vector in query.present_query_vectors
                ),
                default=-1.0,
            )
            if cell.normalized_source_centroid is not None
            else -1.0
        )
        semantic_unit = (semantic + 1.0) / 2.0
        specificity = math.fsum(
            (
                *(
                    classifier.term_specificity[term]
                    for term in query_terms & terms
                    if term in classifier.term_specificity
                ),
                *(
                    classifier.action_specificity[action]
                    for action in query_actions & actions
                    if action in classifier.action_specificity
                ),
            )
        )
        specificity_unit = min(1.0, specificity / maximum_specificity)
        slot_hits = sum(
            len(set(slot.match_terms) & terms) >= slot.minimum_match_term_count
            for slot in query.operator_spec.required_slots
        )
        slot_unit = (
            slot_hits / len(query.operator_spec.required_slots)
            if query.operator_spec.required_slots
            else 0.0
        )
        quote_folded = segment.quote.casefold()
        literal_hits = sum(value in quote_folded for value in exact_literals)
        literal_unit = (
            literal_hits / len(exact_literals) if exact_literals else 0.0
        )
        numeric_affinity = float(numeric_needed and segment.contains_numeric_value)
        temporal_affinity = float(temporal_needed and bool(segment.event_dates))
        base_score = math.fsum(
            (
                0.45 * semantic_unit,
                0.25 * specificity_unit,
                0.15 * slot_unit,
                0.10 * literal_unit,
                0.025 * numeric_affinity,
                0.025 * temporal_affinity,
            )
        )
        entries.append(
            {
                "base_score": base_score,
                "cell": cell,
                "created_at_timestamp": _created_at_timestamp(segment.created_at),
                "date_key": (
                    segment.event_dates[0]
                    if segment.event_dates
                    else segment.created_at[:10]
                ),
                "segment": segment,
                "semantic": semantic,
            }
        )
    baseline = sorted(
        entries,
        key=lambda row: (
            -round(float(row["base_score"]), 12),
            -round(float(row["semantic"]), 12),
            int(row["segment"].token_count),  # type: ignore[union-attr]
            str(row["segment"].receipt_sha256),  # type: ignore[union-attr]
        ),
    )
    source_seen: dict[str, int] = defaultdict(int)
    date_seen: dict[str, int] = defaultdict(int)
    for row in baseline:
        cell = row["cell"]
        assert type(cell) is SemanticResidualCell
        date_key = str(row["date_key"])
        row["source_repeat_ordinal"] = source_seen[cell.source_id]
        row["date_repeat_ordinal"] = date_seen[date_key]
        source_seen[cell.source_id] += 1
        date_seen[date_key] += 1
        row["diversified_score"] = float(row["base_score"]) - (
            0.01 * min(int(row["source_repeat_ordinal"]), 10)
            + 0.005 * min(int(row["date_repeat_ordinal"]), 10)
        )
    ranked = sorted(
        baseline,
        key=lambda row: (
            -round(float(row["diversified_score"]), 12),
            -round(float(row["base_score"]), 12),
            (
                -float(row["created_at_timestamp"])
                if temporal_needed
                else 0.0
            ),
            int(row["segment"].token_count),  # type: ignore[union-attr]
            str(row["segment"].receipt_sha256),  # type: ignore[union-attr]
        ),
    )
    return tuple((row["cell"], row["segment"]) for row in ranked)  # type: ignore[misc]


def _frontier(
    residual_index: SemanticResidualIndex,
    core: SemanticBinarySearchResult,
    retained_segment_receipts: tuple[str, ...],
    packed_segment_receipts: tuple[str, ...],
    duplicates: tuple[SemanticResidualProtectedDuplicate, ...],
    unresolved_segment_receipts: tuple[str, ...],
) -> ClassifiedResidualFrontierReceipt:
    return ClassifiedResidualFrontierReceipt(
        residual_index_receipt_sha256=residual_index.receipt_sha256,
        core_result_receipt_sha256=core.receipt_sha256,
        retained_leaf_cell_ids=core.retained_leaf_cell_ids,
        certified_negative_leaf_cell_ids=core.pruned_leaf_cell_ids,
        retained_segment_receipt_sha256s=retained_segment_receipts,
        packed_segment_receipt_sha256s=packed_segment_receipts,
        protected_duplicate_segment_receipt_sha256s=tuple(
            row.segment_receipt_sha256 for row in duplicates
        ),
        protected_duplicate_audit_receipt_sha256s=tuple(
            row.receipt_sha256 for row in duplicates
        ),
        unresolved_segment_receipt_sha256s=unresolved_segment_receipts,
        classified_leaf_count=len(residual_index.cells),
        all_novel_survivors_protected=(
            not packed_segment_receipts and not unresolved_segment_receipts
        ),
        closed=not unresolved_segment_receipts,
    )


def search_semantic_residual(
    residual_index: SemanticResidualIndex,
    query: SemanticResidualQuery,
    /,
    *,
    protected_evidence: Sequence[LocalCitationBinding] = (),
) -> SemanticResidualSearchResult:
    """Run the terminal tree over all cells, then exact-dedup and pack once."""

    _require(type(residual_index) is SemanticResidualIndex, "residual search index changed")
    _require(
        type(query) is SemanticResidualQuery
        and query.residual_index_receipt_sha256 == residual_index.receipt_sha256,
        "residual search query escaped its index",
    )
    classifier = _ConservativeResidualClassifier(residual_index, query)
    # This terminal classifier is local and deterministic, so descend every
    # uncertain internal node.  A per-node text-fit stop is not compositional:
    # several individually fitting survivor branches can overflow their union,
    # and it prevents the dual gate from pruning irrelevant descendants.
    fit_predicate = lambda node: False
    core = semantic_binary_search(
        residual_index.core_tree,
        query.dated_question,
        classifier,
        fit_predicate=fit_predicate,
        fit_policy_id=(
            f"semantic-residual-full-descent:{residual_index.policy.receipt_sha256}"
        ),
    )
    validate_semantic_binary_search_result(
        residual_index.core_tree,
        query.dated_question,
        core,
        fit_predicate=fit_predicate,
    )
    _require(
        len(classifier.audits) == core.classifier_calls,
        "residual classifier lost an audit",
    )
    _ordered_protected, protected_by_span, protected_population_receipt = (
        _protected_evidence_inventory(residual_index, protected_evidence)
    )
    by_cell = residual_index.cell_by_id
    selected_pairs = tuple(
        (by_cell[cell_id], segment)
        for cell_id in core.retained_leaf_cell_ids
        for segment in by_cell[cell_id].segments
    )
    retained_pairs = _rank_retained_pairs(
        selected_pairs,
        classifier=classifier,
        query=query,
    )
    retained_segment_receipts = tuple(
        segment.receipt_sha256 for _cell, segment in retained_pairs
    )
    source_ids = tuple(sorted({cell.source_id for cell, _segment in retained_pairs}))
    groups = semantic_residual_source_group_map(source_ids)
    duplicates: list[SemanticResidualProtectedDuplicate] = []
    attempted_selection: list[SemanticResidualAttemptedSelection] = []
    evidence_rows: list[SemanticResidualEvidence] = []
    local_rows: list[LocalCitationBinding] = []
    query_terms = set(query.query_terms) | set(query.slot_terms)
    query_actions = set(query.action_concepts)
    for cell, segment in retained_pairs:
        span_sha = _span_identity_sha256(segment.span)
        owner = protected_by_span.get(span_sha)
        if owner is not None:
            duplicate = SemanticResidualProtectedDuplicate(
                cell_id=cell.cell_id,
                segment_receipt_sha256=segment.receipt_sha256,
                span_identity_sha256=span_sha,
                protected_candidate_id=owner.candidate_id,
                protected_binding_receipt_sha256=owner.receipt_sha256,
            )
            duplicates.append(duplicate)
            attempted_selection.append(
                SemanticResidualAttemptedSelection(
                    cell_id=cell.cell_id,
                    source_id=cell.source_id,
                    segment_receipt_sha256=segment.receipt_sha256,
                    disposition="protected_exact_duplicate",
                    candidate_id=owner.candidate_id,
                    local_binding_receipt_sha256=owner.receipt_sha256,
                    evidence_receipt_sha256=None,
                    protected_duplicate_receipt_sha256=duplicate.receipt_sha256,
                )
            )
            continue
        candidate_id = identity_sha256(
            {
                "format": f"{EVIDENCE_FORMAT}-candidate",
                "residual_index_receipt_sha256": residual_index.receipt_sha256,
                "segment_receipt_sha256": segment.receipt_sha256,
            }
        )
        group = groups[cell.source_id]
        local = LocalCitationBinding(
            candidate_id=candidate_id,
            source_group_handle=group,
            namespace_id=residual_index.namespace_id,
            cache_receipt_sha256=residual_index.cache_receipt_sha256,
            source_database_sha256=residual_index.source_database_sha256,
            source_store_receipt_sha256=residual_index.source_store_receipt_sha256,
            source_id=segment.source_id,
            partition_id=segment.partition_id,
            span=segment.span,
            quote_sha256=segment.quote_sha256,
        )
        evidence = SemanticResidualEvidence(
            candidate_id=candidate_id,
            cell_id=cell.cell_id,
            segment_receipt_sha256=segment.receipt_sha256,
            source_group_handle=group,
            quote=segment.quote,
            quote_sha256=segment.quote_sha256,
            token_count=segment.token_count,
            role=segment.role,
            created_at=segment.created_at,
            event_dates=segment.event_dates,
            contains_numeric_value=segment.contains_numeric_value,
            matched_query_terms=tuple(
                sorted(query_terms & set(segment.surface_terms))
            ),
            matched_action_concepts=tuple(
                sorted(query_actions & set(segment.action_concepts))
            ),
            citation_binding_receipt_sha256=local.receipt_sha256,
        )
        evidence_rows.append(evidence)
        local_rows.append(local)
        attempted_selection.append(
            SemanticResidualAttemptedSelection(
                cell_id=cell.cell_id,
                source_id=cell.source_id,
                segment_receipt_sha256=segment.receipt_sha256,
                disposition="novel",
                candidate_id=candidate_id,
                local_binding_receipt_sha256=local.receipt_sha256,
                evidence_receipt_sha256=evidence.receipt_sha256,
                protected_duplicate_receipt_sha256=None,
            )
        )
    frozen_duplicates = tuple(duplicates)
    attempted_evidence = tuple(evidence_rows)
    attempted_locals = tuple(local_rows)
    attempted_tokens = _terminal_residual_evidence_tokens(attempted_evidence)
    packed_evidence: list[SemanticResidualEvidence] = []
    packed_locals: list[LocalCitationBinding] = []
    unresolved_segment_receipts: list[str] = []
    for row, binding in zip(attempted_evidence, attempted_locals, strict=True):
        candidate = (*packed_evidence, row)
        if (
            _terminal_residual_evidence_tokens(candidate)
            <= residual_index.policy.payload_token_cap
        ):
            packed_evidence.append(row)
            packed_locals.append(binding)
        else:
            # Skip an oversized candidate and continue: a later, shorter exact
            # survivor may still fit the independent residual plane.
            unresolved_segment_receipts.append(row.segment_receipt_sha256)
    evidence = tuple(packed_evidence)
    local_bindings = tuple(packed_locals)
    packed_segment_receipts = tuple(
        row.segment_receipt_sha256 for row in evidence
    )
    frontier = _frontier(
        residual_index,
        core,
        retained_segment_receipts,
        packed_segment_receipts,
        frozen_duplicates,
        tuple(unresolved_segment_receipts),
    )
    fallback_required = bool(attempted_evidence) and not evidence
    fallback_reason: Literal["none", "zero_packable_novel_evidence"] = (
        "zero_packable_novel_evidence" if fallback_required else "none"
    )
    packed_residual_evidence_tokens = _terminal_residual_evidence_tokens(evidence)
    packed_residual_evidence_sha256 = (
        semantic_residual_terminal_evidence_sha256(evidence)
    )
    provider_payload_tokens = count_tokens(
        _canonical_json(
            _provider_payload(
                query.dated_question,
                evidence,
                frontier,
                fallback_reason=fallback_reason,
            )
        )
    )
    return SemanticResidualSearchResult(
        residual_index_receipt_sha256=residual_index.receipt_sha256,
        query=query,
        core_result=core,
        decision_audits=tuple(classifier.audits),
        protected_evidence_population_receipt_sha256=protected_population_receipt,
        protected_duplicates=frozen_duplicates,
        attempted_selection=tuple(attempted_selection),
        evidence=evidence,
        local_bindings=local_bindings,
        classified_frontier=frontier,
        attempted_evidence_count=len(attempted_evidence),
        attempted_provider_payload_tokens=attempted_tokens,
        provider_payload_tokens=provider_payload_tokens,
        packed_residual_evidence_tokens=packed_residual_evidence_tokens,
        packed_residual_evidence_sha256=packed_residual_evidence_sha256,
        residual_evidence_token_cap=residual_index.policy.payload_token_cap,
        fallback_required=fallback_required,
        fallback_reason=fallback_reason,
    )


def validate_semantic_residual_search(
    residual_index: SemanticResidualIndex,
    query: SemanticResidualQuery,
    result: SemanticResidualSearchResult,
    /,
    *,
    protected_evidence: Sequence[LocalCitationBinding] = (),
) -> SemanticResidualSearchResult:
    """Recompute from sealed local inputs and require exact byte projection."""

    _require(type(result) is SemanticResidualSearchResult, "residual result changed type")
    replayed = search_semantic_residual(
        residual_index,
        query,
        protected_evidence=protected_evidence,
    )
    _require(
        replayed.receipt_sha256 == result.receipt_sha256
        and replayed.projection() == result.projection()
        and replayed.local_audit_projection() == result.local_audit_projection(),
        "semantic residual replay differs from sealed result",
    )
    return result


def replay_semantic_residual_search(
    residual_index: SemanticResidualIndex,
    query: SemanticResidualQuery,
    sealed_result: SemanticResidualSearchResult,
    /,
    *,
    protected_evidence: Sequence[LocalCitationBinding] = (),
) -> SemanticResidualSearchResult:
    """Return a fresh deterministic replay after strict sealed-result validation."""

    validate_semantic_residual_search(
        residual_index,
        query,
        sealed_result,
        protected_evidence=protected_evidence,
    )
    return search_semantic_residual(
        residual_index,
        query,
        protected_evidence=protected_evidence,
    )


def validate_semantic_residual_search_projection(
    residual_index: SemanticResidualIndex,
    query: SemanticResidualQuery,
    projection: Mapping[str, object],
    /,
    *,
    protected_evidence: Sequence[LocalCitationBinding] = (),
) -> SemanticResidualSearchResult:
    """Strict JSON projection loader: reconstruct from store/index, then compare."""

    _require(type(projection) is dict, "semantic residual projection changed schema")
    replayed = search_semantic_residual(
        residual_index,
        query,
        protected_evidence=protected_evidence,
    )
    _require(
        replayed.projection() == projection,
        "stored semantic residual projection differs from deterministic replay",
    )
    return replayed


load_semantic_residual_search_projection = validate_semantic_residual_search_projection


def adapt_semantic_residual_to_typed_contribution(
    result: SemanticResidualSearchResult,
    /,
    *,
    handle_start: int,
    group_start: int = 1,
) -> TypedEvidenceContribution:
    """Adapt the terminal residual delta without overstating support closure.

    ``ClassifiedResidualFrontierReceipt.closed`` is a packing property: every
    retained MAY segment was packed here or has a visible protected owner.  A
    leaf pruned by the semantic dual gate is not a typed/hard proof that the
    leaf contains no answer support.  Consequently packing closure maps only
    to ``BOUNDED``.  An unpacked/fallback result maps to ``OPEN`` and remains
    truncated.  This adapter never emits ``EXHAUSTIVE``; that mode requires a
    separate support-closure certificate covering every pruned leaf.
    """

    _require(
        type(result) is SemanticResidualSearchResult,
        "typed residual adapter requires an exact result",
    )
    for value, label in (
        (handle_start, "residual handle start"),
        (group_start, "residual group start"),
    ):
        _require(type(value) is int and value >= 1, f"{label} must be positive")
    _require(
        handle_start + len(result.evidence) - 1 <= 999_999,
        "residual handle range exceeds opaque contract",
    )
    local_groups = tuple(
        dict.fromkeys(row.source_group_handle for row in result.evidence)
    )
    _require(
        group_start + len(local_groups) - 1 <= 999_999,
        "residual group range exceeds opaque contract",
    )
    global_groups = {
        local: f"G{group_start + index:03d}"
        for index, local in enumerate(local_groups)
    }
    bindings: list[EvidenceHandleBinding] = []
    raw_items: list[dict[str, object]] = []
    spec = result.query.operator_spec
    for index, (evidence, local) in enumerate(
        zip(result.evidence, result.local_bindings, strict=True)
    ):
        handle_id = f"H{handle_start + index:03d}"
        binding = EvidenceHandleBinding(
            handle_id=handle_id,
            origin=EvidenceOrigin.MAP,
            provenance_grade=ProvenanceGrade.EXACT_CITATION,
            source_group_handle=global_groups[evidence.source_group_handle],
            sealed_artifact_sha256=result.receipt_sha256,
            parent_receipt_sha256=result.classified_frontier.receipt_sha256,
            evidence_receipt_sha256=evidence.receipt_sha256,
            payload_sha256=identity_sha256(evidence.projection()),
            citation_sha256=evidence.quote_sha256,
            citation_char_count=len(evidence.quote),
            local_source_locator_sha256=local.receipt_sha256,
        )
        mention = single_numeric_mention(
            evidence.quote,
            operator_spec=spec,
            question=result.query.dated_question,
        )
        if mention is not None:
            kind = TypedItemKind.OPERAND.value
        elif spec.temporal_mode.value != "none":
            kind = TypedItemKind.EVENT.value
        elif spec.answer_shape is AnswerShape.SET_LIST:
            kind = TypedItemKind.MEMBER.value
        elif spec.style.value == "state_chain":
            kind = TypedItemKind.STATE.value
        elif spec.answer_shape is AnswerShape.SYNTHESIS:
            kind = TypedItemKind.CLAIM.value
        else:
            kind = TypedItemKind.DIRECT.value
        summary = evidence.quote
        _require(bool(summary.strip()), "residual exact quote has no typed content")
        raw: dict[str, object] = {
            "handle_ids": [handle_id],
            "included": True,
            "kind": kind,
            "numeric_role": (
                NumericRole.OPERAND.value if mention is not None else NumericRole.NONE.value
            ),
            "specificity_terms": [],
            "summary": summary,
            "value_authority": "explicit",
        }
        if len(evidence.event_dates) == 1:
            evidence_date = evidence.event_dates[0]
            date_basis = "textual_event_date"
            raw["value_authority"] = "explicit"
        else:
            evidence_date = evidence.created_at
            date_basis = "source_created_at"
            raw["value_authority"] = "derived"
        raw["date"] = evidence_date
        if evidence.role in {"user", "assistant"}:
            raw["relation"] = f"authored_by_{evidence.role};date_basis={date_basis}"
        else:
            raw["relation"] = f"date_basis={date_basis}"
        if mention is not None:
            raw["numeric_value"] = mention.value
            raw["numeric_qualifier"] = mention.qualifier.value
            if mention.unit is not None:
                raw["unit"] = mention.unit
        bindings.append(binding)
        raw_items.append(raw)
    frozen_bindings = tuple(bindings)
    # The common parser rejects surrounding whitespace as an input-schema
    # convenience.  Validate the semantic fields through that parser, then
    # restore the selected segment's byte-exact citation in the immutable
    # typed item.  No provider-visible summary is stripped or rewritten.
    parser_items = [
        {**row, "summary": str(row["summary"]).strip()} for row in raw_items
    ]
    parsed = parse_typed_items(
        parser_items,
        operator_spec=spec,
        bindings=frozen_bindings,
    )
    if not parsed.rejected_items:
        exact_items: list[TypedEvidenceItem] = []
        for source_index, (item, evidence) in enumerate(
            zip(parsed.accepted_items, result.evidence, strict=True)
        ):
            item_id = identity_sha256(
                {
                    "format": ITEM_FORMAT,
                    "handle_ids": list(item.handle_ids),
                    "source_index": source_index,
                    "summary_sha256": quote_sha256(evidence.quote),
                    "supported_slot_ids": list(item.supported_slot_ids),
                }
            )
            exact_items.append(
                TypedEvidenceItem(
                    item_id=item_id,
                    handle_ids=item.handle_ids,
                    kind=item.kind,
                    summary=evidence.quote,
                    entity_key=item.entity_key,
                    group_key=item.group_key,
                    numeric_value=item.numeric_value,
                    numeric_role=item.numeric_role,
                    numeric_qualifier=item.numeric_qualifier,
                    unit=item.unit,
                    date=item.date,
                    status=item.status,
                    relation=item.relation,
                    participant_count=item.participant_count,
                    value_authority=item.value_authority,
                    included=item.included,
                    supported_slot_ids=item.supported_slot_ids,
                    content_coherence=item.content_coherence,
                    content_conflict=item.content_conflict,
                    conflict_receipt_sha256=item.conflict_receipt_sha256,
                    specificity_terms=item.specificity_terms,
                    personalization_anchors=item.personalization_anchors,
                )
            )
        parsed = ParsedTypedItems(
            accepted_items=tuple(exact_items),
            rejected_items=(),
            parse_receipt_sha256=identity_sha256(
                {
                    "accepted_item_receipt_sha256s": [
                        row.receipt_sha256 for row in exact_items
                    ],
                    "format": f"{ITEM_FORMAT}-parse",
                    "rejected_item_receipt_sha256s": [],
                }
            ),
        )
    represented = {
        handle_id
        for item in parsed.accepted_items
        for handle_id in item.handle_ids
    }
    _require(
        not parsed.rejected_items
        and represented == {row.handle_id for row in frozen_bindings},
        "terminal residual adapter failed to represent a packed MAY survivor",
    )
    packing_closed = result.packing_frontier_closed
    frontier_mode = FrontierMode.BOUNDED if packing_closed else FrontierMode.OPEN
    contribution = TypedEvidenceContribution(
        mechanism_id=TYPED_ADAPTER_MECHANISM_ID,
        bindings=frozen_bindings,
        parsed=parsed,
        sealed_artifact_sha256=result.receipt_sha256,
        frontier_mode=frontier_mode,
        truncated=not packing_closed,
    )
    _require(
        contribution.frontier_mode
        is (FrontierMode.BOUNDED if packing_closed else FrontierMode.OPEN)
        and contribution.truncated == (not packing_closed)
        and contribution.frontier_mode is not FrontierMode.EXHAUSTIVE,
        "typed residual adapter overstated packing closure as support closure",
    )
    return contribution


__all__ = [
    "ClassifiedResidualFrontierReceipt",
    "ExactCellSegment",
    "MECHANISM_ID",
    "BOUNDED_PACKING_ALGORITHM",
    "SOURCE_GROUP_ALLOCATION_FORMAT",
    "TYPED_ADAPTER_MECHANISM_ID",
    "SEMANTIC_SEED_ALGORITHM",
    "SemanticNodeManifest",
    "SemanticResidualCell",
    "SemanticResidualDecisionAudit",
    "SemanticResidualEvidence",
    "SemanticResidualAttemptedSelection",
    "SemanticResidualIndex",
    "SemanticResidualPolicy",
    "SemanticResidualProtectedDuplicate",
    "SemanticResidualQuery",
    "SemanticResidualSearchError",
    "SemanticResidualSearchResult",
    "SourceCentroidVectorSet",
    "StoredChunkVectorSet",
    "adapt_semantic_residual_to_typed_contribution",
    "build_semantic_residual_index",
    "compile_semantic_residual_query",
    "load_semantic_residual_search_projection",
    "load_stored_chunk_vectors",
    "load_stored_source_centroid_vectors",
    "replay_semantic_residual_search",
    "search_semantic_residual",
    "semantic_residual_source_group_map",
    "semantic_residual_source_identity_receipt",
    "semantic_residual_protected_evidence_population_receipt",
    "semantic_residual_terminal_evidence_projection",
    "semantic_residual_terminal_evidence_rows",
    "semantic_residual_terminal_evidence_sha256",
    "semantic_residual_query_facets",
    "source_centroid_vector_set",
    "validate_semantic_residual_search",
    "validate_semantic_residual_search_projection",
]
