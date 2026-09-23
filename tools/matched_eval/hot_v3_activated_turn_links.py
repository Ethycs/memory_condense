"""Provider-free raw turn-neighborhood lane for the sealed hot-v3 path.

The lane starts from chunk IDs already selected by another retriever.  Those
physical chunks are activation coordinates, not evidence emitted by this
lane.  The resident full-store rows and query-independent
``SourceNeighborhoodIndex`` resolve complete raw chunks in the immediate
same-source predecessor and successor turns.  Only those linked neighbors are
packed, in deterministic neighborhood order, under the separate 1,200-token
cap.  A large activation seed therefore cannot starve its own neighbor.

Cross-lane deduplication is deliberately later: the activated selection is
frozen before parent IDs are inspected, exact chunk-ID collisions are owned by
the activated lane, and neither lane is refilled.  No semantic deduplication,
provider call, gold label, or answer is involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import EvidenceSpan, quote_sha256
from memory_condense.search.source_neighborhood import (
    LinkDirection,
    SourceChunkMetadata,
    SourceNeighborhoodIndex,
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


MECHANISM_ID = "hot_v3_activated_turn_links_v2"
DEFAULT_TOKEN_CAP = 1_200
INDEX_FORMAT = "memory-condense-hot-v3-activated-turn-link-index-v2"
LINK_FORMAT = "memory-condense-hot-v3-activated-turn-link-v2"
CHUNK_FORMAT = "memory-condense-hot-v3-activated-turn-chunk-v2"
RECEIPT_FORMAT = "memory-condense-hot-v3-activated-turn-links-receipt-v2"
RESULT_FORMAT = "memory-condense-hot-v3-activated-turn-links-result-v2"

ChunkOrigin = Literal["neighbor"]


class HotV3ActivatedTurnLinkError(MatchedEvalContractError):
    """Raised when resident topology, exact chunks, or lane policy changes."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise HotV3ActivatedTurnLinkError(message)


def _ordered_unique(values: Sequence[str], label: str) -> tuple[str, ...]:
    _require(not isinstance(values, (str, bytes)), f"{label} must be a sequence")
    result = tuple(values)
    _require(
        all(type(value) is str and value and value.strip() == value for value in result),
        f"{label} must contain exact non-empty text",
    )
    _require(len(result) == len(set(result)), f"{label} must be ordered unique")
    return result


def _is_subsequence(values: Sequence[str], parent: Sequence[str]) -> bool:
    parent_iter = iter(parent)
    return all(any(candidate == value for candidate in parent_iter) for value in values)


def _metadata(row: CachedContentRow) -> SourceChunkMetadata:
    return SourceChunkMetadata(
        chunk_id=row.chunk_id,
        source_id=row.source_id,
        turn_id=row.turn_id,
        ordinal=row.ordinal,
        start_char=row.turn_start_char,
    )


def _metadata_projection(row: CachedContentRow) -> dict[str, Any]:
    metadata = _metadata(row)
    return {
        "chunk_id": metadata.chunk_id,
        "ordinal": metadata.ordinal,
        "source_id": metadata.source_id,
        "start_char": metadata.start_char,
        "turn_id": metadata.turn_id,
    }


@dataclass(frozen=True, slots=True)
class ActivatedTurnLinkBudget:
    """Locked independent raw-chunk allowance for this additive lane."""

    token_cap: int = DEFAULT_TOKEN_CAP

    def __post_init__(self) -> None:
        _require(
            type(self.token_cap) is int and self.token_cap == DEFAULT_TOKEN_CAP,
            "activated turn-link budget must remain the independent 1,200-token cap",
        )

    def projection(self) -> dict[str, Any]:
        return {
            "budget_scope": (
                "linked_neighbor_evidence_only_before_global_chunk_dedup"
            ),
            "seed_activation_inputs_charge_tokens": False,
            "token_cap": self.token_cap,
        }

    @property
    def budget_id(self) -> str:
        return identity_sha256(
            {"budget": self.projection(), "mechanism_id": MECHANISM_ID}
        )


@dataclass(frozen=True, slots=True)
class HotV3ActivatedTurnLinkIndex:
    """Full-store raw rows paired with their immutable source topology."""

    parent: FullStoreWindowIndex
    source_neighborhood: SourceNeighborhoodIndex
    rows_by_chunk_id: Mapping[str, CachedContentRow] = field(repr=False)
    neighborhood_inventory_sha256: str = ""
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(type(self.parent) is FullStoreWindowIndex, "full-store index changed")
        _require(
            type(self.source_neighborhood) is SourceNeighborhoodIndex,
            "source-neighborhood index changed",
        )
        _require(isinstance(self.rows_by_chunk_id, Mapping), "row inventory changed")
        rows = tuple(self.parent.rows)
        by_id = dict(self.rows_by_chunk_id)
        _require(
            len(by_id) == len(rows)
            and all(
                type(key) is str
                and type(row) is CachedContentRow
                and key == row.chunk_id
                and by_id.get(row.chunk_id) == row
                for key, row in by_id.items()
            )
            and all(by_id.get(row.chunk_id) == row for row in rows),
            "row inventory lost its exact full-store binding",
        )
        metadata = [_metadata_projection(row) for row in rows]
        expected_inventory = identity_sha256(metadata)
        if self.neighborhood_inventory_sha256:
            _require(
                self.neighborhood_inventory_sha256 == expected_inventory,
                "source-neighborhood metadata inventory changed",
            )
        _require(
            self.source_neighborhood.chunk_count == len(rows)
            and self.source_neighborhood.turn_count == len({row.turn_id for row in rows})
            and self.source_neighborhood.source_count
            == len({row.source_id for row in rows}),
            "source-neighborhood cardinality differs from full-store rows",
        )
        object.__setattr__(self, "rows_by_chunk_id", MappingProxyType(by_id))
        object.__setattr__(
            self, "neighborhood_inventory_sha256", expected_inventory
        )
        expected_receipt = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected_receipt, "turn-link index changed")
        object.__setattr__(self, "receipt_sha256", expected_receipt)
        assert_gold_blind(self.projection(), path="hot_v3_activated_turn_link_index")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "cache_receipt_sha256": self.parent.cache.cache_receipt_sha256,
            "chunk_count": len(self.rows_by_chunk_id),
            "format": INDEX_FORMAT,
            "full_store_index_receipt_sha256": self.parent.receipt_sha256,
            "gold_loaded": False,
            "model_calls": 0,
            "neighborhood_inventory_sha256": self.neighborhood_inventory_sha256,
            "new_provider_calls": 0,
            "retained_transformer_token_state_bytes": 0,
            "source_count": self.source_neighborhood.source_count,
            "turn_count": self.source_neighborhood.turn_count,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def build_hot_v3_activated_turn_link_index(
    parent: FullStoreWindowIndex,
    /,
    *,
    source_neighborhood_index: SourceNeighborhoodIndex | None = None,
) -> HotV3ActivatedTurnLinkIndex:
    """Bind resident full-store rows to a reusable local-turn topology.

    Callers that already compiled ``SourceNeighborhoodIndex`` may pass it to
    avoid rebuilding topology.  Query-time seed and neighbor coordinates are
    still checked against the full-store rows before any chunk is admitted.
    """

    _require(type(parent) is FullStoreWindowIndex, "full-store index changed")
    rows = tuple(parent.rows)
    neighborhood = source_neighborhood_index or SourceNeighborhoodIndex(
        tuple(_metadata(row) for row in rows)
    )
    return HotV3ActivatedTurnLinkIndex(
        parent=parent,
        source_neighborhood=neighborhood,
        rows_by_chunk_id=MappingProxyType({row.chunk_id: row for row in rows}),
        neighborhood_inventory_sha256=identity_sha256(
            [_metadata_projection(row) for row in rows]
        ),
    )


@dataclass(frozen=True, slots=True)
class ActivatedTurnLink:
    """Content-addressed edge from one seed to one neighboring raw chunk."""

    seed_chunk_id: str
    linked_chunk_id: str
    source_id: str
    direction: LinkDirection
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.seed_chunk_id, "turn-link seed"),
            (self.linked_chunk_id, "turn-link neighbor"),
            (self.source_id, "turn-link source"),
        ):
            require_text(value, label)
        _require(
            self.seed_chunk_id != self.linked_chunk_id,
            "turn-link edge points to its seed",
        )
        _require(
            self.direction in {"predecessor_turn", "successor_turn"},
            "turn-link direction changed",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "turn-link edge changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "direction": self.direction,
            "format": LINK_FORMAT,
            "linked_chunk_id": self.linked_chunk_id,
            "seed_chunk_id": self.seed_chunk_id,
            "source_id": self.source_id,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class ActivatedTurnChunk:
    """One exact full raw chunk with source and span receipts intact."""

    chunk_id: str
    source_id: str
    role: str
    created_at: str
    raw_text: str
    token_count: int
    span: EvidenceSpan
    origin: ChunkOrigin
    neighborhood_link_receipt_sha256s: tuple[str, ...]
    namespace_id: str
    cache_receipt_sha256: str
    full_store_index_receipt_sha256: str
    source_row_receipt_sha256: str
    span_receipt_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.chunk_id, "activated chunk"),
            (self.source_id, "activated source"),
            (self.role, "activated role"),
            (self.created_at, "activated created-at"),
            (self.raw_text, "activated raw text"),
        ):
            require_text(value, label)
        for value, label in (
            (self.namespace_id, "activated namespace"),
            (self.cache_receipt_sha256, "activated cache"),
            (self.full_store_index_receipt_sha256, "activated full-store index"),
            (self.source_row_receipt_sha256, "activated source row"),
            (self.span_receipt_sha256, "activated span"),
        ):
            require_sha256(value, label)
        links = _ordered_unique(
            self.neighborhood_link_receipt_sha256s,
            "activated neighborhood-link receipts",
        )
        for receipt in links:
            require_sha256(receipt, "activated neighborhood link")
        _require(
            type(self.span) is EvidenceSpan
            and self.span.chunk_id == self.chunk_id
            and self.span.source_id == self.source_id
            and self.span.role == self.role
            and self.span.created_at == self.created_at
            and self.span.start_char == 0
            and self.span.end_char == len(self.raw_text)
            and self.span.quote_sha256 == quote_sha256(self.raw_text),
            "activated raw chunk lost its exact span",
        )
        _require(
            self.span_receipt_sha256 == identity_sha256(self.span.identity_payload()),
            "activated span receipt changed",
        )
        _require(
            type(self.token_count) is int
            and self.token_count == count_tokens(self.raw_text)
            and self.token_count > 0,
            "activated chunk token count changed",
        )
        _require(self.origin == "neighbor", "chunk origin changed")
        _require(
            bool(links),
            "every activated output chunk must carry a neighborhood path",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "activated chunk changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "cache_receipt_sha256": self.cache_receipt_sha256,
            "chunk_id": self.chunk_id,
            "created_at": self.created_at,
            "format": CHUNK_FORMAT,
            "full_store_index_receipt_sha256": self.full_store_index_receipt_sha256,
            "namespace_id": self.namespace_id,
            "neighborhood_link_receipt_sha256s": list(
                self.neighborhood_link_receipt_sha256s
            ),
            "origin": self.origin,
            "raw_text": self.raw_text,
            "raw_text_sha256": quote_sha256(self.raw_text),
            "role": self.role,
            "source_id": self.source_id,
            "source_row_receipt_sha256": self.source_row_receipt_sha256,
            "span": self.span.identity_payload(),
            "span_receipt_sha256": self.span_receipt_sha256,
            "token_count": self.token_count,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _materialize(
    index: HotV3ActivatedTurnLinkIndex,
    row: CachedContentRow,
    *,
    origin: ChunkOrigin,
    link_receipts: Sequence[str] = (),
) -> ActivatedTurnChunk:
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
    return ActivatedTurnChunk(
        chunk_id=row.chunk_id,
        source_id=row.source_id,
        role=row.role,
        created_at=row.created_at,
        raw_text=row.text,
        token_count=row.token_count,
        span=span,
        origin=origin,
        neighborhood_link_receipt_sha256s=tuple(link_receipts),
        namespace_id=row.namespace_id,
        cache_receipt_sha256=index.parent.cache.cache_receipt_sha256,
        full_store_index_receipt_sha256=index.parent.receipt_sha256,
        source_row_receipt_sha256=identity_sha256(row.receipt_projection()),
        span_receipt_sha256=identity_sha256(span.identity_payload()),
    )


@dataclass(frozen=True, slots=True)
class ActivatedTurnLinkReceipt:
    """Auditable activation, neighbor selection, and no-refill proof."""

    index_receipt_sha256: str
    budget_id: str
    seed_chunk_ids: tuple[str, ...]
    seed_chunk_ids_sha256: str
    seed_activation_input_tokens: int
    candidate_population_ids: tuple[str, ...]
    candidate_neighbor_ids_sha256: str
    neighborhood_link_receipt_sha256s: tuple[str, ...]
    neighborhood_paths_sha256: str
    selected_before_dedup_ids: tuple[str, ...]
    selected_before_dedup_tokens: int
    budget_excluded_ids: tuple[str, ...]
    parent_input_ids: tuple[str, ...]
    parent_input_ids_sha256: str
    exact_parent_duplicate_ids: tuple[str, ...]
    activated_retained_after_dedup_ids: tuple[str, ...]
    parent_retained_after_dedup_ids: tuple[str, ...]
    unmaterialized_parent_retained_ids: tuple[str, ...]
    retained_after_dedup_ids: tuple[str, ...]
    selection_truncated: bool
    independent_lane_token_cap: Literal[1200] = DEFAULT_TOKEN_CAP
    selection_before_parent_dedup: Literal[True] = True
    selection_before_global_dedup: Literal[True] = True
    activated_lane_wins_exact_chunk_collision: Literal[True] = True
    refill_after_parent_dedup: Literal[False] = False
    exact_chunk_id_dedup_only: Literal[True] = True
    seed_activation_inputs_charge_tokens: Literal[False] = False
    seed_activation_inputs_emitted: Literal[False] = False
    new_provider_calls: Literal[0] = 0
    model_calls: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    gold_loaded: Literal[False] = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.index_receipt_sha256, "activated turn-link index")
        require_sha256(self.budget_id, "activated turn-link budget")
        require_sha256(
            self.candidate_neighbor_ids_sha256,
            "activated candidate-neighbor population",
        )
        require_sha256(self.neighborhood_paths_sha256, "activated paths")
        seeds = _ordered_unique(self.seed_chunk_ids, "seed chunk IDs")
        candidates = _ordered_unique(
            self.candidate_population_ids, "candidate population IDs"
        )
        links = _ordered_unique(
            self.neighborhood_link_receipt_sha256s,
            "neighborhood link receipts",
        )
        for value in links:
            require_sha256(value, "neighborhood link receipt")
        selected = _ordered_unique(
            self.selected_before_dedup_ids, "selected-before-dedup IDs"
        )
        excluded = _ordered_unique(self.budget_excluded_ids, "budget-excluded IDs")
        parent = _ordered_unique(self.parent_input_ids, "parent input IDs")
        duplicates = _ordered_unique(
            self.exact_parent_duplicate_ids, "exact parent duplicate IDs"
        )
        activated = _ordered_unique(
            self.activated_retained_after_dedup_ids,
            "activated retained-after-dedup IDs",
        )
        parent_retained = _ordered_unique(
            self.parent_retained_after_dedup_ids,
            "parent retained-after-dedup IDs",
        )
        unmaterialized_parent = _ordered_unique(
            self.unmaterialized_parent_retained_ids,
            "unmaterialized parent retained IDs",
        )
        retained = _ordered_unique(
            self.retained_after_dedup_ids, "retained-after-dedup IDs"
        )
        _require(
            self.seed_chunk_ids_sha256 == identity_sha256(list(seeds))
            and self.parent_input_ids_sha256 == identity_sha256(list(parent)),
            "turn-link input population receipt changed",
        )
        _require(
            self.candidate_neighbor_ids_sha256 == identity_sha256(list(candidates))
            and self.neighborhood_paths_sha256 == identity_sha256(list(links))
            and set(candidates).isdisjoint(seeds),
            "activation seeds escaped into the neighbor candidate population",
        )
        _require(
            set(selected).isdisjoint(excluded)
            and set(selected) | set(excluded) == set(candidates)
            and _is_subsequence(selected, candidates)
            and _is_subsequence(excluded, candidates),
            "frozen selection does not partition the candidate population",
        )
        _require(
            tuple(value for value in parent if value in set(selected)) == duplicates
            and tuple(value for value in parent if value not in set(selected))
            == parent_retained,
            "exact parent-chunk ownership transfer changed",
        )
        _require(
            activated == selected
            and retained == (*activated, *parent_retained)
            and len(retained) == len(set(retained)),
            "activated-lane-wins retained order changed",
        )
        _require(
            set(unmaterialized_parent) <= set(parent_retained)
            and _is_subsequence(unmaterialized_parent, parent_retained),
            "unmaterialized parent audit escaped retained parent IDs",
        )
        _require(
            type(self.seed_activation_input_tokens) is int
            and self.seed_activation_input_tokens >= 0
            and type(self.selected_before_dedup_tokens) is int
            and 0 <= self.selected_before_dedup_tokens
            <= self.independent_lane_token_cap,
            "turn-link token accounting changed",
        )
        _require(
            type(self.selection_truncated) is bool
            and self.selection_truncated == bool(excluded),
            "turn-link truncation flag changed",
        )
        _require(
            self.independent_lane_token_cap == DEFAULT_TOKEN_CAP
            and self.selection_before_parent_dedup is True
            and self.selection_before_global_dedup is True
            and self.activated_lane_wins_exact_chunk_collision is True
            and self.refill_after_parent_dedup is False
            and self.exact_chunk_id_dedup_only is True
            and self.seed_activation_inputs_charge_tokens is False
            and self.seed_activation_inputs_emitted is False
            and self.new_provider_calls == self.model_calls == 0
            and self.retained_transformer_token_state_bytes == 0
            and self.gold_loaded is False,
            "turn-link zero-call or ownership policy changed",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "turn-link receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="hot_v3_activated_turn_link_receipt")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "activated_lane_wins_exact_chunk_collision": True,
            "activated_retained_after_dedup_ids": list(
                self.activated_retained_after_dedup_ids
            ),
            "budget_excluded_ids": list(self.budget_excluded_ids),
            "budget_id": self.budget_id,
            "candidate_neighbor_ids_sha256": self.candidate_neighbor_ids_sha256,
            "candidate_population_ids": list(self.candidate_population_ids),
            "exact_chunk_id_dedup_only": True,
            "exact_parent_duplicate_ids": list(self.exact_parent_duplicate_ids),
            "format": RECEIPT_FORMAT,
            "gold_loaded": False,
            "independent_lane_token_cap": self.independent_lane_token_cap,
            "index_receipt_sha256": self.index_receipt_sha256,
            "model_calls": 0,
            "neighborhood_link_receipt_sha256s": list(
                self.neighborhood_link_receipt_sha256s
            ),
            "neighborhood_paths_sha256": self.neighborhood_paths_sha256,
            "new_provider_calls": 0,
            "parent_input_ids": list(self.parent_input_ids),
            "parent_input_ids_sha256": self.parent_input_ids_sha256,
            "parent_retained_after_dedup_ids": list(
                self.parent_retained_after_dedup_ids
            ),
            "refill_after_parent_dedup": False,
            "retained_after_dedup_ids": list(self.retained_after_dedup_ids),
            "retained_transformer_token_state_bytes": 0,
            "seed_chunk_ids": list(self.seed_chunk_ids),
            "seed_chunk_ids_sha256": self.seed_chunk_ids_sha256,
            "seed_activation_input_tokens": self.seed_activation_input_tokens,
            "seed_activation_inputs_charge_tokens": False,
            "seed_activation_inputs_emitted": False,
            "selected_before_dedup_ids": list(self.selected_before_dedup_ids),
            "selected_before_dedup_tokens": self.selected_before_dedup_tokens,
            "selection_before_global_dedup": True,
            "selection_before_parent_dedup": True,
            "selection_truncated": self.selection_truncated,
            "unmaterialized_parent_retained_ids": list(
                self.unmaterialized_parent_retained_ids
            ),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class ActivatedTurnLinkResult:
    """Raw activated chunks plus an ID-level parent-union merge recipe."""

    links: tuple[ActivatedTurnLink, ...]
    selected_before_dedup: tuple[ActivatedTurnChunk, ...]
    receipt: ActivatedTurnLinkReceipt
    budget: ActivatedTurnLinkBudget

    def __post_init__(self) -> None:
        for values, expected, label in (
            (self.links, ActivatedTurnLink, "turn links"),
            (self.selected_before_dedup, ActivatedTurnChunk, "selected chunks"),
        ):
            _require(
                type(values) is tuple and all(type(value) is expected for value in values),
                f"{label} changed",
            )
        selected_ids = tuple(row.chunk_id for row in self.selected_before_dedup)
        _require(
            tuple(row.receipt_sha256 for row in self.links)
            == self.receipt.neighborhood_link_receipt_sha256s
            and selected_ids == self.receipt.selected_before_dedup_ids
            and selected_ids == self.receipt.activated_retained_after_dedup_ids,
            "turn-link result differs from its receipt",
        )
        _require(
            all(row.origin == "neighbor" for row in self.selected_before_dedup),
            "turn-link result ownership changed",
        )
        _require(
            sum(row.token_count for row in self.selected_before_dedup)
            == self.receipt.selected_before_dedup_tokens
            <= self.budget.token_cap
            and self.receipt.budget_id == self.budget.budget_id,
            "turn-link result token accounting changed",
        )

    @property
    def activated_retained_after_dedup(self) -> tuple[ActivatedTurnChunk, ...]:
        """The activated owner side; exact parent duplicates never remove it."""

        return self.selected_before_dedup

    @property
    def selected_before_dedup_ids(self) -> tuple[str, ...]:
        return self.receipt.selected_before_dedup_ids

    @property
    def retained_after_dedup_ids(self) -> tuple[str, ...]:
        """Activated IDs followed by opaque or physical surviving parent IDs."""

        return self.receipt.retained_after_dedup_ids

    def audit_projection(self) -> dict[str, Any]:
        value = {
            "format": RESULT_FORMAT,
            "links": [row.projection() for row in self.links],
            "receipt": self.receipt.projection(),
            "selected_before_dedup": [
                row.projection() for row in self.selected_before_dedup
            ],
        }
        assert_gold_blind(value, path="hot_v3_activated_turn_link_result")
        return value


def _validate_neighborhood_metadata(
    metadata: SourceChunkMetadata,
    row: CachedContentRow,
) -> None:
    _require(metadata == _metadata(row), "source-neighborhood coordinate changed")


def select_hot_v3_activated_turn_links(
    index: HotV3ActivatedTurnLinkIndex,
    seed_chunk_ids: Sequence[str],
    /,
    *,
    parent_chunk_ids: Sequence[str] = (),
    budget: ActivatedTurnLinkBudget = ActivatedTurnLinkBudget(),
) -> ActivatedTurnLinkResult:
    """Activate from seeds, select immediate raw neighbors, then dedup parent.

    Seeds must be physical full-store chunks, but are coordinates only: they
    are neither emitted nor charged to this lane's evidence cap.  Parent IDs
    are intentionally inspected only after the independent neighbor selection
    is fixed.  Exact neighbor collisions are retained on the activated side;
    the returned combined order is activated neighbors followed by novel
    parent chunks.  No vacancy is refilled after this ownership transfer.
    """

    _require(type(index) is HotV3ActivatedTurnLinkIndex, "turn-link index changed")
    _require(type(budget) is ActivatedTurnLinkBudget, "turn-link budget changed")
    seeds = _ordered_unique(seed_chunk_ids, "seed chunk IDs")
    parent_ids = _ordered_unique(parent_chunk_ids, "parent chunk IDs")
    rows = index.rows_by_chunk_id
    missing_seed = next((value for value in seeds if value not in rows), None)
    _require(missing_seed is None, f"seed chunk is absent from full store: {missing_seed}")

    try:
        neighborhood = index.source_neighborhood.neighbors(seeds)
    except ValueError as exc:
        raise HotV3ActivatedTurnLinkError(
            "source-neighborhood index cannot resolve a full-store seed"
        ) from exc
    for group in neighborhood.seed_groups:
        for metadata in group.seeds:
            _validate_neighborhood_metadata(metadata, rows[metadata.chunk_id])
    for metadata in neighborhood.candidates:
        _require(
            metadata.chunk_id in rows,
            "source-neighborhood candidate is absent from full-store rows",
        )
        _validate_neighborhood_metadata(metadata, rows[metadata.chunk_id])

    links = tuple(
        ActivatedTurnLink(
            seed_chunk_id=row.seed_chunk_id,
            linked_chunk_id=row.linked_chunk_id,
            source_id=row.source_id,
            direction=row.direction,
        )
        for row in neighborhood.links
    )
    link_receipts_by_chunk: dict[str, list[str]] = {}
    for link in links:
        seed = rows[link.seed_chunk_id]
        linked = rows[link.linked_chunk_id]
        _require(
            seed.source_id == linked.source_id == link.source_id,
            "source-neighborhood link crossed sources",
        )
        link_receipts_by_chunk.setdefault(link.linked_chunk_id, []).append(
            link.receipt_sha256
        )

    candidate_ids = tuple(neighborhood.candidate_chunk_ids)
    _require(
        len(candidate_ids) == len(set(candidate_ids)),
        "neighborhood candidate population repeats a chunk",
    )
    _require(
        set(candidate_ids).isdisjoint(seeds),
        "activation seed escaped into neighbor candidate population",
    )
    selected_ids: list[str] = []
    budget_excluded_ids: list[str] = []
    used_tokens = 0
    for chunk_id in candidate_ids:
        row = rows[chunk_id]
        if used_tokens + row.token_count > budget.token_cap:
            budget_excluded_ids.append(chunk_id)
            continue
        selected_ids.append(chunk_id)
        used_tokens += row.token_count

    selected = tuple(
        _materialize(
            index,
            rows[chunk_id],
            origin="neighbor",
            link_receipts=link_receipts_by_chunk[chunk_id],
        )
        for chunk_id in selected_ids
    )

    # This is deliberately the first read of parent membership.  Activated
    # chunks own exact collisions, and neither side is refilled afterward.
    selected_set = set(selected_ids)
    duplicates = tuple(value for value in parent_ids if value in selected_set)
    parent_retained_ids = tuple(
        value for value in parent_ids if value not in selected_set
    )
    unmaterialized_parent_ids = tuple(
        value for value in parent_retained_ids if value not in rows
    )
    retained_ids = (*selected_ids, *parent_retained_ids)
    receipt = ActivatedTurnLinkReceipt(
        index_receipt_sha256=index.receipt_sha256,
        budget_id=budget.budget_id,
        seed_chunk_ids=seeds,
        seed_chunk_ids_sha256=identity_sha256(list(seeds)),
        seed_activation_input_tokens=sum(rows[value].token_count for value in seeds),
        candidate_population_ids=tuple(candidate_ids),
        candidate_neighbor_ids_sha256=identity_sha256(list(candidate_ids)),
        neighborhood_link_receipt_sha256s=tuple(
            row.receipt_sha256 for row in links
        ),
        neighborhood_paths_sha256=identity_sha256(
            [row.receipt_sha256 for row in links]
        ),
        selected_before_dedup_ids=tuple(selected_ids),
        selected_before_dedup_tokens=used_tokens,
        budget_excluded_ids=tuple(budget_excluded_ids),
        parent_input_ids=parent_ids,
        parent_input_ids_sha256=identity_sha256(list(parent_ids)),
        exact_parent_duplicate_ids=duplicates,
        activated_retained_after_dedup_ids=tuple(selected_ids),
        parent_retained_after_dedup_ids=parent_retained_ids,
        unmaterialized_parent_retained_ids=unmaterialized_parent_ids,
        retained_after_dedup_ids=tuple(retained_ids),
        selection_truncated=bool(budget_excluded_ids),
    )
    return ActivatedTurnLinkResult(
        links=links,
        selected_before_dedup=selected,
        receipt=receipt,
        budget=budget,
    )


__all__ = [
    "ActivatedTurnChunk",
    "ActivatedTurnLink",
    "ActivatedTurnLinkBudget",
    "ActivatedTurnLinkReceipt",
    "ActivatedTurnLinkResult",
    "DEFAULT_TOKEN_CAP",
    "HotV3ActivatedTurnLinkError",
    "HotV3ActivatedTurnLinkIndex",
    "MECHANISM_ID",
    "build_hot_v3_activated_turn_link_index",
    "select_hot_v3_activated_turn_links",
]
