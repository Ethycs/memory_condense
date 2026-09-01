"""Provider-free semantic and fact closure over an exact selected H population.

This module is deliberately downstream of retrieval.  It accepts the exact
post-selection H-leaf population, consumes already sealed relevance decisions,
and uses :mod:`semantic_binary_search` only as a fail-closed accounting core.
Internal nodes are always ``may_answer``; a leaf is pruned only when its sealed
disposition is ``definitely_irrelevant``.

The retained leaves are then reconciled with sharded, precomputed atomic facts
whose provenance is carried by :class:`typed_fact_compiler.CompiledTypedFact`.
Every selected leaf must have exactly one terminal outcome: one or more facts,
``definitely_irrelevant``, or ``unresolved``.  Shards are merged without model
IO, exact citations are checked against the selected leaf bytes, and duplicate
event/member facts are grouped by a structured fingerprint rather than story
group membership or surface-text equality.

Only selected-population resolution and operator-obligation coverage are
computed here.  Neither is promoted to full-store semantic closure.  The API
has no parent prediction, benchmark answer, ordinal routing, source allowlist,
or semantic-atom-manifest input.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .semantic_binary_search import (
    ProvenanceBinding,
    SemanticBinarySearchResult,
    SemanticBranchDecision,
    SemanticCell,
    SemanticSearchNode,
    SemanticSearchTree,
    build_semantic_search_tree,
    make_branch_decision,
    replay_semantic_binary_search,
    semantic_binary_search,
)
from .typed_fact_compiler import CompiledTypedFact, TypedFactCitation


SELECTED_LEAF_FORMAT = "memory-condense-after-union-selected-h-leaf-v1"
CROSS_BOUNDARY_EDGE_FORMAT = "memory-condense-after-union-cross-boundary-edge-v1"
LEAF_DISPOSITION_FORMAT = "memory-condense-after-union-leaf-disposition-v1"
SELECTION_FORMAT = "memory-condense-after-union-semantic-selection-v1"
OBLIGATION_FORMAT = "memory-condense-after-union-operator-obligation-v1"
ATOMIC_FACT_FORMAT = "memory-condense-after-union-structured-atomic-fact-v1"
LEAF_FACT_OUTCOME_FORMAT = "memory-condense-after-union-leaf-fact-outcome-v1"
FACT_SHARD_FORMAT = "memory-condense-after-union-fact-outcome-shard-v1"
FINGERPRINT_FORMAT = "memory-condense-after-union-event-member-fingerprint-v1"
MERGED_FACT_FORMAT = "memory-condense-after-union-merged-fact-v1"
POPULATION_COVERAGE_FORMAT = (
    "memory-condense-after-union-selected-population-coverage-v1"
)
OBLIGATION_ROW_FORMAT = "memory-condense-after-union-obligation-coverage-row-v1"
OBLIGATION_COVERAGE_FORMAT = "memory-condense-after-union-obligation-coverage-v1"
CLOSURE_FORMAT = "memory-condense-after-union-fact-closure-v1"
CLASSIFIER_ADAPTER_ID = "after-union-sealed-leaf-disposition-adapter-v1"
FIT_POLICY_ID = "after-union-exact-leaf-descent-v1"

_H_RE = re.compile(r"^H[0-9]{3,6}$")
_G_RE = re.compile(r"^G[0-9]{3,6}$")

LeafRelevance = Literal["relevant", "definitely_irrelevant", "uncertain"]
LeafFactDisposition = Literal["facts", "definitely_irrelevant", "unresolved"]
ObligationKind = Literal[
    "direct",
    "member",
    "event",
    "operand",
    "endpoint",
    "qualifier",
    "comparison_side",
]
CrossBoundaryEdgeKind = Literal["entity", "event", "temporal"]


class AfterUnionFactClosureError(MatchedEvalContractError):
    """A selected leaf, sealed decision, fact, shard, or receipt changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise AfterUnionFactClosureError(message)


def _exact_tuple(value: object, label: str) -> tuple:
    _require(type(value) is tuple, f"{label} must be an exact tuple")
    return value


def _ordered_unique(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    _exact_tuple(values, label)
    _require(
        all(type(value) is str and bool(value) and value.strip() == value for value in values)
        and len(set(values)) == len(values),
        f"{label} must be ordered unique exact text",
    )
    return values


def _receipt(projection: Mapping[str, object], declared: str, label: str) -> str:
    expected = identity_sha256(projection)
    if declared:
        _require(require_sha256(declared, label) == expected, f"{label} changed")
    return expected


def _key(value: str | None) -> str:
    return "" if value is None else " ".join(value.casefold().split())


def _normalized_unique(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    _ordered_unique(values, label)
    normalized = tuple(_key(value) for value in values)
    _require(
        all(normalized) and len(set(normalized)) == len(normalized),
        f"{label} repeat after normalization",
    )
    return normalized


def _citation_identity(citation: TypedFactCitation) -> str:
    """Return a stable receipt for the compiler citation's public projection."""

    _require(type(citation) is TypedFactCitation, "citation identity type changed")
    return identity_sha256(citation.projection())


@dataclass(frozen=True, slots=True)
class CrossBoundaryEdge:
    """One explicit entity/event/time bridge between selected topic regions."""

    edge_id: str
    kind: CrossBoundaryEdgeKind
    left_handle_id: str
    right_handle_id: str
    relation: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_text(self.edge_id, "cross-boundary edge ID")
        _require(
            self.kind in {"entity", "event", "temporal"},
            "cross-boundary edge kind changed",
        )
        _require(
            type(self.left_handle_id) is str
            and _H_RE.fullmatch(self.left_handle_id)
            and type(self.right_handle_id) is str
            and _H_RE.fullmatch(self.right_handle_id)
            and self.left_handle_id != self.right_handle_id,
            "cross-boundary edge endpoints changed",
        )
        require_text(self.relation, "cross-boundary edge relation")
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "cross-boundary edge receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="after_union_cross_boundary_edge")

    @property
    def handle_ids(self) -> tuple[str, str]:
        return (self.left_handle_id, self.right_handle_id)

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "edge_id": self.edge_id,
            "format": CROSS_BOUNDARY_EDGE_FORMAT,
            "kind": self.kind,
            "left_handle_id": self.left_handle_id,
            "relation": self.relation,
            "right_handle_id": self.right_handle_id,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class SelectedHLeaf:
    """One exact post-selection H item and its immutable source receipt."""

    handle_id: str
    group_handle: str
    text: str
    source_receipt_sha256: str
    topic_labels: tuple[str, ...] = ()
    boundary_labels: tuple[str, ...] = ()
    cross_boundary_edge_ids: tuple[str, ...] = ()
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(type(self.handle_id) is str and _H_RE.fullmatch(self.handle_id), "selected H handle changed")
        _require(type(self.group_handle) is str and _G_RE.fullmatch(self.group_handle), "selected G handle changed")
        require_text(self.text, "selected leaf text")
        require_sha256(self.source_receipt_sha256, "selected leaf source receipt")
        _normalized_unique(self.topic_labels, "selected leaf topic labels")
        _normalized_unique(self.boundary_labels, "selected leaf boundary labels")
        _ordered_unique(
            self.cross_boundary_edge_ids,
            "selected leaf cross-boundary edge IDs",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "selected leaf receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="after_union_selected_leaf")

    def semantic_cell(self) -> SemanticCell:
        binding = ProvenanceBinding(self.handle_id, self.source_receipt_sha256)
        return SemanticCell(
            self.handle_id,
            self.group_handle,
            self.text,
            count_tokens(self.text),
            (binding,),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "format": SELECTED_LEAF_FORMAT,
            "group_handle": self.group_handle,
            "handle_id": self.handle_id,
            "source_receipt_sha256": self.source_receipt_sha256,
            "text": self.text,
            "token_count": count_tokens(self.text),
            "topic_labels": list(self.topic_labels),
            "boundary_labels": list(self.boundary_labels),
            "cross_boundary_edge_ids": list(self.cross_boundary_edge_ids),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class SealedLeafDisposition:
    """One precomputed R/I/U decision bound to a question and exact leaf."""

    handle_id: str
    leaf_receipt_sha256: str
    question_sha256: str
    classifier_id: str
    disposition: LeafRelevance
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(type(self.handle_id) is str and _H_RE.fullmatch(self.handle_id), "disposition H handle changed")
        require_sha256(self.leaf_receipt_sha256, "disposition leaf receipt")
        require_sha256(self.question_sha256, "disposition question")
        require_text(self.classifier_id, "disposition classifier ID")
        _require(
            self.disposition in {"relevant", "definitely_irrelevant", "uncertain"},
            "leaf disposition must be relevant, definitely_irrelevant, or uncertain",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "leaf disposition receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="after_union_leaf_disposition")

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "classifier_id": self.classifier_id,
            "disposition": self.disposition,
            "format": LEAF_DISPOSITION_FORMAT,
            "handle_id": self.handle_id,
            "leaf_receipt_sha256": self.leaf_receipt_sha256,
            "question_sha256": self.question_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


class _SealedLeafClassifier:
    """Adapt sealed leaf decisions to the binary search's two-way boundary."""

    classifier_id = CLASSIFIER_ADAPTER_ID

    def __init__(
        self,
        question_sha256: str,
        dispositions: Mapping[str, SealedLeafDisposition],
    ) -> None:
        self._question_sha256 = question_sha256
        self._dispositions = dict(dispositions)

    def classify(
        self,
        *,
        question: str,
        node: SemanticSearchNode,
        call_ordinal: int,
    ) -> SemanticBranchDecision:
        _require(
            quote_sha256(question) == self._question_sha256,
            "semantic classifier question changed",
        )
        branch = "may_answer"
        if node.is_leaf:
            disposition = self._dispositions[node.cells[0].cell_id]
            branch = (
                "definitely_no"
                if disposition.disposition == "definitely_irrelevant"
                else "may_answer"
            )
        return make_branch_decision(
            classifier_id=self.classifier_id,
            question_sha256=self._question_sha256,
            node_receipt_sha256=node.receipt_sha256,
            call_ordinal=call_ordinal,
            branch_classification=branch,
        )


def _never_fit(_node: SemanticSearchNode) -> bool:
    return False


@dataclass(frozen=True, slots=True)
class AfterUnionSelection:
    """Exact R/I/U population plus its replayable binary-search partition."""

    question_sha256: str
    leaves: tuple[SelectedHLeaf, ...]
    dispositions: tuple[SealedLeafDisposition, ...]
    cross_boundary_edges: tuple[CrossBoundaryEdge, ...]
    semantic_tree: SemanticSearchTree
    semantic_result: SemanticBinarySearchResult
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.question_sha256, "after-union selection question")
        _require(
            type(self.leaves) is tuple
            and bool(self.leaves)
            and all(type(row) is SelectedHLeaf for row in self.leaves)
            and len({row.handle_id for row in self.leaves}) == len(self.leaves),
            "after-union selection leaves changed",
        )
        _require(
            type(self.dispositions) is tuple
            and len(self.dispositions) == len(self.leaves)
            and all(type(row) is SealedLeafDisposition for row in self.dispositions)
            and tuple(row.handle_id for row in self.dispositions)
            == tuple(row.handle_id for row in self.leaves),
            "after-union disposition population changed",
        )
        _require(
            type(self.cross_boundary_edges) is tuple
            and all(type(row) is CrossBoundaryEdge for row in self.cross_boundary_edges)
            and len({row.edge_id for row in self.cross_boundary_edges})
            == len(self.cross_boundary_edges),
            "after-union cross-boundary edge population changed",
        )
        leaves_by_handle = {row.handle_id: row for row in self.leaves}
        edges_by_id = {row.edge_id: row for row in self.cross_boundary_edges}
        _require(
            all(set(row.handle_ids) <= set(leaves_by_handle) for row in self.cross_boundary_edges),
            "cross-boundary edge escaped the selected population",
        )
        for leaf in self.leaves:
            expected = {
                edge.edge_id
                for edge in self.cross_boundary_edges
                if leaf.handle_id in edge.handle_ids
            }
            _require(
                set(leaf.cross_boundary_edge_ids) == expected
                and set(leaf.cross_boundary_edge_ids) <= set(edges_by_id),
                "selected leaf cross-boundary edge descriptor changed",
            )
        _require(
            type(self.semantic_tree) is SemanticSearchTree
            and type(self.semantic_result) is SemanticBinarySearchResult
            and tuple(row.cell_id for row in self.semantic_tree.cells)
            == tuple(row.handle_id for row in self.leaves)
            and self.semantic_result.tree_receipt_sha256
            == self.semantic_tree.receipt_sha256
            and self.semantic_result.question_sha256 == self.question_sha256,
            "after-union semantic tree/result binding changed",
        )
        expected_pruned = tuple(
            row.handle_id
            for row in self.dispositions
            if row.disposition == "definitely_irrelevant"
        )
        expected_retained = tuple(
            row.handle_id
            for row in self.dispositions
            if row.disposition != "definitely_irrelevant"
        )
        _require(
            self.semantic_result.pruned_leaf_cell_ids == expected_pruned
            and self.semantic_result.retained_leaf_cell_ids == expected_retained,
            "after-union search pruned something other than definitely irrelevant leaves",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "after-union selection receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="after_union_selection")

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "dispositions": [row.projection() for row in self.dispositions],
            "cross_boundary_edges": [
                row.projection() for row in self.cross_boundary_edges
            ],
            "format": SELECTION_FORMAT,
            "gold_loaded": False,
            "leaves": [row.projection() for row in self.leaves],
            "provider_calls_performed_by_core": 0,
            "question_sha256": self.question_sha256,
            "retained_transformer_token_state_bytes": 0,
            "semantic_result": self.semantic_result.projection(),
            "semantic_tree_receipt_sha256": self.semantic_tree.receipt_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def build_after_union_selection(
    question: str,
    leaves: Sequence[SelectedHLeaf],
    dispositions: Sequence[SealedLeafDisposition],
    *,
    cross_boundary_edges: Sequence[CrossBoundaryEdge] = (),
) -> AfterUnionSelection:
    """Build an exact leaf partition from sealed R/I/U decisions without IO."""

    require_text(question, "after-union question")
    leaf_rows = tuple(leaves)
    disposition_rows = tuple(dispositions)
    edge_rows = tuple(cross_boundary_edges)
    _require(
        bool(leaf_rows) and all(type(row) is SelectedHLeaf for row in leaf_rows),
        "after-union selection requires exact selected H leaves",
    )
    _require(
        all(type(row) is SealedLeafDisposition for row in disposition_rows),
        "after-union selection requires sealed leaf dispositions",
    )
    _require(
        all(type(row) is CrossBoundaryEdge for row in edge_rows),
        "after-union selection requires exact cross-boundary edges",
    )
    handles = tuple(row.handle_id for row in leaf_rows)
    _require(len(set(handles)) == len(handles), "selected H leaf population repeats")
    by_handle = {row.handle_id: row for row in disposition_rows}
    _require(
        len(by_handle) == len(disposition_rows)
        and set(by_handle) == set(handles),
        "sealed dispositions must cover the exact selected H population",
    )
    question_sha = quote_sha256(question)
    ordered_dispositions = tuple(by_handle[handle] for handle in handles)
    classifier_ids = {row.classifier_id for row in ordered_dispositions}
    _require(len(classifier_ids) == 1, "sealed dispositions changed classifier population")
    for leaf, disposition in zip(leaf_rows, ordered_dispositions, strict=True):
        _require(
            disposition.leaf_receipt_sha256 == leaf.receipt_sha256
            and disposition.question_sha256 == question_sha,
            "sealed disposition escaped its question or selected leaf",
        )
    tree = build_semantic_search_tree(tuple(row.semantic_cell() for row in leaf_rows))
    classifier = _SealedLeafClassifier(
        question_sha,
        {row.handle_id: row for row in ordered_dispositions},
    )
    semantic_result = semantic_binary_search(
        tree,
        question,
        classifier,
        fit_predicate=_never_fit,
        fit_policy_id=FIT_POLICY_ID,
    )
    replay_semantic_binary_search(
        tree,
        question,
        semantic_result,
        fit_predicate=_never_fit,
    )
    return AfterUnionSelection(
        question_sha,
        leaf_rows,
        ordered_dispositions,
        edge_rows,
        tree,
        semantic_result,
    )


def replay_after_union_selection(
    question: str,
    sealed: AfterUnionSelection,
) -> AfterUnionSelection:
    """Rebuild a sealed selection and require a byte-identical projection."""

    _require(type(sealed) is AfterUnionSelection, "sealed selection changed type")
    replayed = build_after_union_selection(
        question,
        sealed.leaves,
        sealed.dispositions,
        cross_boundary_edges=sealed.cross_boundary_edges,
    )
    _require(
        replayed.projection() == sealed.projection()
        and replayed.receipt_sha256 == sealed.receipt_sha256,
        "after-union selection replay differs from sealed result",
    )
    return replayed


@dataclass(frozen=True, slots=True)
class OperatorObligation:
    """One question/operator-derived fact requirement; never benchmark-derived."""

    obligation_id: str
    kind: ObligationKind
    description: str
    required: bool = True
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_text(self.obligation_id, "operator obligation ID")
        _require(
            self.kind
            in {
                "direct",
                "member",
                "event",
                "operand",
                "endpoint",
                "qualifier",
                "comparison_side",
            },
            "operator obligation kind changed",
        )
        require_text(self.description, "operator obligation description")
        _require(type(self.required) is bool, "operator obligation required flag changed")
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "operator obligation receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="after_union_operator_obligation")

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "description": self.description,
            "format": OBLIGATION_FORMAT,
            "kind": self.kind,
            "obligation_id": self.obligation_id,
            "required": self.required,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class StructuredAtomicFact:
    """One compiler-validated exact-cited fact plus event/member structure."""

    leaf_handle_id: str
    compiled_fact: CompiledTypedFact
    predicate: str
    member_key: str | None = None
    event_time: str | None = None
    source_time: str | None = None
    qualifiers: tuple[str, ...] = ()
    obligation_ids: tuple[str, ...] = ()
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(type(self.leaf_handle_id) is str and _H_RE.fullmatch(self.leaf_handle_id), "atomic fact H handle changed")
        _require(type(self.compiled_fact) is CompiledTypedFact, "atomic fact lost compiled fact provenance")
        require_text(self.predicate, "atomic fact predicate")
        for value, label in (
            (self.member_key, "member key"),
            (self.event_time, "event time"),
            (self.source_time, "source time"),
        ):
            if value is not None:
                require_text(value, f"atomic fact {label}")
        _normalized_unique(self.qualifiers, "atomic fact qualifiers")
        _ordered_unique(self.obligation_ids, "atomic fact obligation IDs")
        _require(
            set(self.compiled_fact.slot_ids) <= set(self.obligation_ids),
            "atomic fact lost a compiler-supported slot obligation",
        )
        _require(
            self.leaf_handle_id in self.compiled_fact.handle_ids,
            "atomic fact must cite its owning selected leaf",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "structured atomic fact receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="after_union_structured_atomic_fact")

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "compiled_fact": self.compiled_fact.projection(),
            "event_time": self.event_time,
            "format": ATOMIC_FACT_FORMAT,
            "leaf_handle_id": self.leaf_handle_id,
            "member_key": self.member_key,
            "obligation_ids": list(self.obligation_ids),
            "predicate": self.predicate,
            "qualifiers": list(self.qualifiers),
            "source_time": self.source_time,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class LeafFactOutcome:
    """Exactly one terminal compiler outcome for one selected leaf."""

    handle_id: str
    leaf_receipt_sha256: str
    leaf_disposition_receipt_sha256: str
    disposition: LeafFactDisposition
    facts: tuple[StructuredAtomicFact, ...] = ()
    unresolved_obligation_ids: tuple[str, ...] = ()
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(type(self.handle_id) is str and _H_RE.fullmatch(self.handle_id), "leaf fact outcome H handle changed")
        require_sha256(self.leaf_receipt_sha256, "leaf fact outcome leaf")
        require_sha256(
            self.leaf_disposition_receipt_sha256,
            "leaf fact outcome disposition",
        )
        _require(
            self.disposition in {"facts", "definitely_irrelevant", "unresolved"},
            "leaf fact outcome disposition changed",
        )
        _require(
            type(self.facts) is tuple
            and all(type(row) is StructuredAtomicFact for row in self.facts)
            and len({row.receipt_sha256 for row in self.facts}) == len(self.facts),
            "leaf fact outcome fact population changed",
        )
        _ordered_unique(
            self.unresolved_obligation_ids,
            "leaf fact outcome unresolved obligation IDs",
        )
        if self.disposition == "facts":
            _require(
                bool(self.facts)
                and not self.unresolved_obligation_ids
                and all(row.leaf_handle_id == self.handle_id for row in self.facts),
                "facts outcome requires one or more facts owned by its leaf",
            )
        else:
            _require(not self.facts, "non-fact leaf outcome cannot carry facts")
            if self.disposition == "definitely_irrelevant":
                _require(
                    not self.unresolved_obligation_ids,
                    "definitely irrelevant leaf cannot carry unresolved obligations",
                )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "leaf fact outcome receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="after_union_leaf_fact_outcome")

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "disposition": self.disposition,
            "facts": [row.projection() for row in self.facts],
            "format": LEAF_FACT_OUTCOME_FORMAT,
            "handle_id": self.handle_id,
            "leaf_disposition_receipt_sha256": (
                self.leaf_disposition_receipt_sha256
            ),
            "leaf_receipt_sha256": self.leaf_receipt_sha256,
            "unresolved_obligation_ids": list(self.unresolved_obligation_ids),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class FactOutcomeShard:
    """A sealed bounded compiler shard, consumed without provider IO."""

    shard_id: str
    selection_receipt_sha256: str
    outcomes: tuple[LeafFactOutcome, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_text(self.shard_id, "fact outcome shard ID")
        require_sha256(self.selection_receipt_sha256, "fact shard selection")
        _require(
            type(self.outcomes) is tuple
            and bool(self.outcomes)
            and all(type(row) is LeafFactOutcome for row in self.outcomes)
            and len({row.handle_id for row in self.outcomes}) == len(self.outcomes),
            "fact shard outcome population changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "fact outcome shard receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="after_union_fact_shard")

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "format": FACT_SHARD_FORMAT,
            "outcomes": [row.projection() for row in self.outcomes],
            "selection_receipt_sha256": self.selection_receipt_sha256,
            "shard_id": self.shard_id,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class EventMemberFingerprint:
    """Structured event/member identity used for cross-shard deduplication."""

    kind: str
    entity_key: str
    predicate_key: str
    member_key: str
    temporal_identity: str
    status: str
    numeric_value: float | None
    unit_key: str
    qualifier_keys: tuple[str, ...]
    fallback_fact_key: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_text(self.kind, "event/member fingerprint kind")
        require_text(self.predicate_key, "event/member fingerprint predicate")
        _require(
            all(
                type(value) is str and value == _key(value)
                for value in (
                    self.entity_key,
                    self.predicate_key,
                    self.member_key,
                    self.temporal_identity,
                    self.status,
                    self.unit_key,
                    self.fallback_fact_key,
                )
            ),
            "event/member fingerprint keys are not canonical",
        )
        _require(
            type(self.qualifier_keys) is tuple
            and tuple(sorted(set(self.qualifier_keys))) == self.qualifier_keys
            and all(type(row) is str and bool(row) and row == _key(row) for row in self.qualifier_keys),
            "event/member fingerprint qualifiers changed",
        )
        _require(
            self.numeric_value is None
            or (
                type(self.numeric_value) in {int, float}
                and math.isfinite(float(self.numeric_value))
            ),
            "event/member fingerprint numeric value changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "event/member fingerprint receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "entity_key": self.entity_key,
            "fallback_fact_key": self.fallback_fact_key,
            "format": FINGERPRINT_FORMAT,
            "kind": self.kind,
            "member_key": self.member_key,
            "numeric_value": self.numeric_value,
            "predicate_key": self.predicate_key,
            "qualifier_keys": list(self.qualifier_keys),
            "status": self.status,
            "temporal_identity": self.temporal_identity,
            "unit_key": self.unit_key,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def event_member_fingerprint(fact: StructuredAtomicFact) -> EventMemberFingerprint:
    """Return the deterministic structured identity for one atomic fact."""

    _require(type(fact) is StructuredAtomicFact, "fingerprint requires an exact atomic fact")
    compiled = fact.compiled_fact
    entity_key = _key(compiled.entity)
    member_key = _key(fact.member_key)
    temporal_identity = _key(fact.event_time or compiled.date or fact.source_time)
    unit_key = _key(compiled.unit)
    status = _key(compiled.status)
    qualifiers = tuple(sorted(_normalized_unique(fact.qualifiers, "fingerprint qualifiers")))
    structured = bool(
        entity_key
        or member_key
        or temporal_identity
        or compiled.numeric_value is not None
        or unit_key
        or status
        or qualifiers
    )
    return EventMemberFingerprint(
        compiled.kind,
        entity_key,
        _key(fact.predicate),
        member_key,
        temporal_identity,
        status,
        compiled.numeric_value,
        unit_key,
        qualifiers,
        "" if structured else _key(compiled.text),
    )


@dataclass(frozen=True, slots=True)
class MergedStructuredFact:
    """All exact-cited atomic facts sharing one event/member fingerprint."""

    fingerprint: EventMemberFingerprint
    facts: tuple[StructuredAtomicFact, ...]
    leaf_handle_ids: tuple[str, ...]
    citations: tuple[TypedFactCitation, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(type(self.fingerprint) is EventMemberFingerprint, "merged fact fingerprint changed")
        _require(
            type(self.facts) is tuple
            and bool(self.facts)
            and all(type(row) is StructuredAtomicFact for row in self.facts)
            and tuple(sorted(row.receipt_sha256 for row in self.facts))
            == tuple(row.receipt_sha256 for row in self.facts)
            and all(
                event_member_fingerprint(row).receipt_sha256
                == self.fingerprint.receipt_sha256
                for row in self.facts
            ),
            "merged fact population or fingerprint changed",
        )
        _ordered_unique(self.leaf_handle_ids, "merged fact leaf handles")
        _require(
            set(self.leaf_handle_ids)
            == {row.leaf_handle_id for row in self.facts},
            "merged fact leaf ownership changed",
        )
        _require(
            type(self.citations) is tuple
            and bool(self.citations)
            and all(type(row) is TypedFactCitation for row in self.citations)
            and tuple(sorted(_citation_identity(row) for row in self.citations))
            == tuple(_citation_identity(row) for row in self.citations)
            and len({_citation_identity(row) for row in self.citations})
            == len(self.citations),
            "merged fact citation population changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "merged structured fact receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="after_union_merged_fact")

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "citations": [row.projection() for row in self.citations],
            "facts": [row.projection() for row in self.facts],
            "fingerprint": self.fingerprint.projection(),
            "format": MERGED_FACT_FORMAT,
            "leaf_handle_ids": list(self.leaf_handle_ids),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class SelectedPopulationCoverage:
    """Exact terminal outcomes for the selected population only."""

    selected_leaf_ids: tuple[str, ...]
    relevant_leaf_ids: tuple[str, ...]
    uncertain_leaf_ids: tuple[str, ...]
    fact_leaf_ids: tuple[str, ...]
    definitely_irrelevant_leaf_ids: tuple[str, ...]
    unresolved_leaf_ids: tuple[str, ...]
    exact_outcome_coverage: bool
    selected_population_resolved: bool
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for values, label in (
            (self.selected_leaf_ids, "selected population leaf IDs"),
            (self.relevant_leaf_ids, "selected relevant leaf IDs"),
            (self.uncertain_leaf_ids, "selected uncertain leaf IDs"),
            (self.fact_leaf_ids, "selected fact leaf IDs"),
            (
                self.definitely_irrelevant_leaf_ids,
                "selected definitely irrelevant leaf IDs",
            ),
            (self.unresolved_leaf_ids, "selected unresolved leaf IDs"),
        ):
            _ordered_unique(values, label)
        selected = set(self.selected_leaf_ids)
        relevance_parts = (
            set(self.relevant_leaf_ids),
            set(self.uncertain_leaf_ids),
            set(self.definitely_irrelevant_leaf_ids),
        )
        outcome_parts = (
            set(self.fact_leaf_ids),
            set(self.definitely_irrelevant_leaf_ids),
            set(self.unresolved_leaf_ids),
        )
        _require(
            not any(left & right for index, left in enumerate(relevance_parts) for right in relevance_parts[index + 1 :])
            and set().union(*relevance_parts) == selected
            and not any(left & right for index, left in enumerate(outcome_parts) for right in outcome_parts[index + 1 :])
            and set().union(*outcome_parts) == selected,
            "selected population coverage lost a partition",
        )
        _require(
            self.exact_outcome_coverage is True
            and self.selected_population_resolved == (not self.unresolved_leaf_ids),
            "selected population closure flags changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "selected population coverage receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="after_union_population_coverage")

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "definitely_irrelevant_leaf_ids": list(
                self.definitely_irrelevant_leaf_ids
            ),
            "exact_outcome_coverage": self.exact_outcome_coverage,
            "fact_leaf_ids": list(self.fact_leaf_ids),
            "format": POPULATION_COVERAGE_FORMAT,
            "relevant_leaf_ids": list(self.relevant_leaf_ids),
            "selected_leaf_ids": list(self.selected_leaf_ids),
            "selected_population_resolved": self.selected_population_resolved,
            "uncertain_leaf_ids": list(self.uncertain_leaf_ids),
            "unresolved_leaf_ids": list(self.unresolved_leaf_ids),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class ObligationCoverageRow:
    obligation: OperatorObligation
    supporting_fact_fingerprint_sha256s: tuple[str, ...]
    unresolved_leaf_ids: tuple[str, ...]
    covered: bool
    closed_within_selected_population: bool
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(type(self.obligation) is OperatorObligation, "obligation coverage obligation changed")
        _require(
            type(self.supporting_fact_fingerprint_sha256s) is tuple
            and len(set(self.supporting_fact_fingerprint_sha256s))
            == len(self.supporting_fact_fingerprint_sha256s),
            "obligation supporting fact population changed",
        )
        for value in self.supporting_fact_fingerprint_sha256s:
            require_sha256(value, "obligation supporting fact fingerprint")
        _ordered_unique(self.unresolved_leaf_ids, "obligation unresolved leaf IDs")
        _require(
            self.covered == bool(self.supporting_fact_fingerprint_sha256s)
            and self.closed_within_selected_population
            == (self.covered and not self.unresolved_leaf_ids),
            "obligation coverage flags changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "obligation coverage row receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="after_union_obligation_row")

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "closed_within_selected_population": (
                self.closed_within_selected_population
            ),
            "covered": self.covered,
            "format": OBLIGATION_ROW_FORMAT,
            "obligation": self.obligation.projection(),
            "supporting_fact_fingerprint_sha256s": list(
                self.supporting_fact_fingerprint_sha256s
            ),
            "unresolved_leaf_ids": list(self.unresolved_leaf_ids),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class OperatorObligationCoverage:
    rows: tuple[ObligationCoverageRow, ...]
    missing_required_obligation_ids: tuple[str, ...]
    unresolved_required_obligation_ids: tuple[str, ...]
    required_obligations_closed_within_selected_population: bool
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            type(self.rows) is tuple
            and all(type(row) is ObligationCoverageRow for row in self.rows)
            and len({row.obligation.obligation_id for row in self.rows})
            == len(self.rows),
            "operator obligation coverage rows changed",
        )
        _ordered_unique(
            self.missing_required_obligation_ids,
            "missing required obligation IDs",
        )
        _ordered_unique(
            self.unresolved_required_obligation_ids,
            "unresolved required obligation IDs",
        )
        missing = tuple(
            row.obligation.obligation_id
            for row in self.rows
            if row.obligation.required
            and not row.covered
            and not row.unresolved_leaf_ids
        )
        unresolved = tuple(
            row.obligation.obligation_id
            for row in self.rows
            if row.obligation.required and bool(row.unresolved_leaf_ids)
        )
        required_closed = all(
            row.closed_within_selected_population
            for row in self.rows
            if row.obligation.required
        )
        _require(
            self.missing_required_obligation_ids == missing
            and self.unresolved_required_obligation_ids == unresolved
            and self.required_obligations_closed_within_selected_population
            == required_closed,
            "operator obligation closure accounting changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "operator obligation coverage receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="after_union_obligation_coverage")

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "format": OBLIGATION_COVERAGE_FORMAT,
            "missing_required_obligation_ids": list(
                self.missing_required_obligation_ids
            ),
            "required_obligations_closed_within_selected_population": (
                self.required_obligations_closed_within_selected_population
            ),
            "rows": [row.projection() for row in self.rows],
            "unresolved_required_obligation_ids": list(
                self.unresolved_required_obligation_ids
            ),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class AfterUnionFactClosure:
    """Merged facts plus separate selected-population/obligation receipts."""

    question_sha256: str
    selection_receipt_sha256: str
    shard_receipt_sha256s: tuple[str, ...]
    leaf_outcomes: tuple[LeafFactOutcome, ...]
    merged_facts: tuple[MergedStructuredFact, ...]
    selected_population_coverage: SelectedPopulationCoverage
    operator_obligation_coverage: OperatorObligationCoverage
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.question_sha256, "after-union closure question")
        require_sha256(self.selection_receipt_sha256, "after-union closure selection")
        _require(
            type(self.shard_receipt_sha256s) is tuple
            and tuple(sorted(set(self.shard_receipt_sha256s)))
            == self.shard_receipt_sha256s,
            "after-union closure shard receipts changed",
        )
        for value in self.shard_receipt_sha256s:
            require_sha256(value, "after-union closure shard")
        _require(
            type(self.leaf_outcomes) is tuple
            and all(type(row) is LeafFactOutcome for row in self.leaf_outcomes)
            and tuple(row.handle_id for row in self.leaf_outcomes)
            == self.selected_population_coverage.selected_leaf_ids,
            "after-union closure leaf outcomes changed",
        )
        _require(
            type(self.merged_facts) is tuple
            and all(type(row) is MergedStructuredFact for row in self.merged_facts)
            and type(self.selected_population_coverage)
            is SelectedPopulationCoverage
            and type(self.operator_obligation_coverage)
            is OperatorObligationCoverage,
            "after-union closure fact or coverage types changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "after-union fact closure receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="after_union_fact_closure")

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "format": CLOSURE_FORMAT,
            "full_store_support_closure_available": False,
            "gold_loaded": False,
            "leaf_outcomes": [row.projection() for row in self.leaf_outcomes],
            "merged_facts": [row.projection() for row in self.merged_facts],
            "operator_obligation_coverage": (
                self.operator_obligation_coverage.projection()
            ),
            "provider_calls_performed_by_core": 0,
            "question_sha256": self.question_sha256,
            "retained_transformer_token_state_bytes": 0,
            "selected_population_coverage": (
                self.selected_population_coverage.projection()
            ),
            "selection_receipt_sha256": self.selection_receipt_sha256,
            "shard_receipt_sha256s": list(self.shard_receipt_sha256s),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _validate_atomic_fact(
    fact: StructuredAtomicFact,
    *,
    leaves: Mapping[str, SelectedHLeaf],
    dispositions: Mapping[str, SealedLeafDisposition],
    obligation_ids: frozenset[str],
) -> None:
    _require(
        set(fact.obligation_ids) <= obligation_ids,
        "atomic fact names an unknown operator obligation",
    )
    for citation in fact.compiled_fact.citations:
        leaf = leaves.get(citation.handle_id)
        _require(leaf is not None, "atomic fact cites a leaf outside the selected population")
        assert leaf is not None
        _require(
            dispositions[citation.handle_id].disposition
            != "definitely_irrelevant",
            "atomic fact cites a pruned definitely irrelevant leaf",
        )
        _require(
            citation.group_handle == leaf.group_handle
            and citation.quote in leaf.text
            and citation.source_summary_sha256 == quote_sha256(leaf.text),
            "atomic fact citation is not exact selected-leaf evidence",
        )


def _merge_structured_facts(
    facts: Sequence[StructuredAtomicFact],
    leaf_order: Mapping[str, int],
) -> tuple[MergedStructuredFact, ...]:
    grouped: dict[str, tuple[EventMemberFingerprint, list[StructuredAtomicFact]]] = {}
    for fact in facts:
        fingerprint = event_member_fingerprint(fact)
        existing = grouped.get(fingerprint.receipt_sha256)
        if existing is None:
            grouped[fingerprint.receipt_sha256] = (fingerprint, [fact])
        else:
            _require(
                existing[0].projection() == fingerprint.projection(),
                "event/member fingerprint receipt collision",
            )
            existing[1].append(fact)
    merged: list[MergedStructuredFact] = []
    for fingerprint, rows in grouped.values():
        ordered_facts = tuple(sorted(rows, key=lambda row: row.receipt_sha256))
        handles = tuple(
            sorted(
                {row.leaf_handle_id for row in ordered_facts},
                key=leaf_order.__getitem__,
            )
        )
        citations_by_receipt = {
            _citation_identity(citation): citation
            for row in ordered_facts
            for citation in row.compiled_fact.citations
        }
        citations = tuple(
            citations_by_receipt[value]
            for value in sorted(citations_by_receipt)
        )
        merged.append(MergedStructuredFact(fingerprint, ordered_facts, handles, citations))
    return tuple(
        sorted(
            merged,
            key=lambda row: (
                min(leaf_order[value] for value in row.leaf_handle_ids),
                row.fingerprint.receipt_sha256,
            ),
        )
    )


def merge_after_union_fact_shards(
    selection: AfterUnionSelection,
    obligations: Sequence[OperatorObligation],
    shards: Sequence[FactOutcomeShard],
) -> AfterUnionFactClosure:
    """Merge exact fact shards and compute two non-overstated coverage ledgers."""

    _require(type(selection) is AfterUnionSelection, "after-union selection changed type")
    obligation_rows = tuple(obligations)
    shard_rows = tuple(shards)
    _require(
        all(type(row) is OperatorObligation for row in obligation_rows)
        and len({row.obligation_id for row in obligation_rows})
        == len(obligation_rows),
        "operator obligation population changed",
    )
    _require(
        bool(shard_rows)
        and all(type(row) is FactOutcomeShard for row in shard_rows)
        and len({row.shard_id for row in shard_rows}) == len(shard_rows)
        and all(
            row.selection_receipt_sha256 == selection.receipt_sha256
            for row in shard_rows
        ),
        "fact shards escaped the selected population",
    )
    flattened = tuple(outcome for shard in shard_rows for outcome in shard.outcomes)
    outcomes_by_handle = {row.handle_id: row for row in flattened}
    selected_ids = tuple(row.handle_id for row in selection.leaves)
    _require(
        len(outcomes_by_handle) == len(flattened)
        and set(outcomes_by_handle) == set(selected_ids),
        "fact outcome shards must cover the exact selected leaf population once",
    )
    ordered_outcomes = tuple(outcomes_by_handle[value] for value in selected_ids)
    leaves = {row.handle_id: row for row in selection.leaves}
    dispositions = {row.handle_id: row for row in selection.dispositions}
    known_obligations = frozenset(row.obligation_id for row in obligation_rows)
    atomic_facts: list[StructuredAtomicFact] = []
    for outcome in ordered_outcomes:
        leaf = leaves[outcome.handle_id]
        disposition = dispositions[outcome.handle_id]
        _require(
            outcome.leaf_receipt_sha256 == leaf.receipt_sha256
            and outcome.leaf_disposition_receipt_sha256
            == disposition.receipt_sha256,
            "leaf fact outcome escaped its selected leaf or disposition",
        )
        if disposition.disposition == "definitely_irrelevant":
            _require(
                outcome.disposition == "definitely_irrelevant",
                "only a sealed definitely irrelevant leaf may take that outcome",
            )
        else:
            _require(
                outcome.disposition in {"facts", "unresolved"},
                "retained relevant/uncertain leaf requires facts or unresolved",
            )
        _require(
            set(outcome.unresolved_obligation_ids) <= known_obligations,
            "unresolved leaf names an unknown operator obligation",
        )
        for fact in outcome.facts:
            _validate_atomic_fact(
                fact,
                leaves=leaves,
                dispositions=dispositions,
                obligation_ids=known_obligations,
            )
            atomic_facts.append(fact)
    leaf_order = {value: index for index, value in enumerate(selected_ids)}
    merged_facts = _merge_structured_facts(atomic_facts, leaf_order)

    relevant_ids = tuple(
        row.handle_id
        for row in selection.dispositions
        if row.disposition == "relevant"
    )
    uncertain_ids = tuple(
        row.handle_id
        for row in selection.dispositions
        if row.disposition == "uncertain"
    )
    fact_ids = tuple(
        row.handle_id for row in ordered_outcomes if row.disposition == "facts"
    )
    irrelevant_ids = tuple(
        row.handle_id
        for row in ordered_outcomes
        if row.disposition == "definitely_irrelevant"
    )
    unresolved_ids = tuple(
        row.handle_id
        for row in ordered_outcomes
        if row.disposition == "unresolved"
    )
    population_coverage = SelectedPopulationCoverage(
        selected_ids,
        relevant_ids,
        uncertain_ids,
        fact_ids,
        irrelevant_ids,
        unresolved_ids,
        True,
        not unresolved_ids,
    )

    coverage_rows: list[ObligationCoverageRow] = []
    for obligation in obligation_rows:
        supporting = tuple(
            row.fingerprint.receipt_sha256
            for row in merged_facts
            if any(
                obligation.obligation_id in fact.obligation_ids
                for fact in row.facts
            )
        )
        unresolved = tuple(
            row.handle_id
            for row in ordered_outcomes
            if row.disposition == "unresolved"
            and (
                not row.unresolved_obligation_ids
                or obligation.obligation_id in row.unresolved_obligation_ids
            )
        )
        coverage_rows.append(
            ObligationCoverageRow(
                obligation,
                supporting,
                unresolved,
                bool(supporting),
                bool(supporting) and not unresolved,
            )
        )
    row_tuple = tuple(coverage_rows)
    missing_required = tuple(
        row.obligation.obligation_id
        for row in row_tuple
        if row.obligation.required and not row.covered and not row.unresolved_leaf_ids
    )
    unresolved_required = tuple(
        row.obligation.obligation_id
        for row in row_tuple
        if row.obligation.required and bool(row.unresolved_leaf_ids)
    )
    obligation_coverage = OperatorObligationCoverage(
        row_tuple,
        missing_required,
        unresolved_required,
        all(
            row.closed_within_selected_population
            for row in row_tuple
            if row.obligation.required
        ),
    )
    return AfterUnionFactClosure(
        selection.question_sha256,
        selection.receipt_sha256,
        tuple(sorted(row.receipt_sha256 for row in shard_rows)),
        ordered_outcomes,
        merged_facts,
        population_coverage,
        obligation_coverage,
    )


def replay_after_union_fact_closure(
    selection: AfterUnionSelection,
    obligations: Sequence[OperatorObligation],
    shards: Sequence[FactOutcomeShard],
    sealed: AfterUnionFactClosure,
) -> AfterUnionFactClosure:
    """Re-merge sealed inputs and require byte-identical coverage receipts."""

    _require(type(sealed) is AfterUnionFactClosure, "sealed fact closure changed type")
    replayed = merge_after_union_fact_shards(selection, obligations, shards)
    _require(
        replayed.projection() == sealed.projection()
        and replayed.receipt_sha256 == sealed.receipt_sha256,
        "after-union fact closure replay differs from sealed result",
    )
    return replayed


__all__ = [
    "AfterUnionFactClosure",
    "AfterUnionFactClosureError",
    "AfterUnionSelection",
    "CrossBoundaryEdge",
    "CrossBoundaryEdgeKind",
    "EventMemberFingerprint",
    "FactOutcomeShard",
    "LeafFactDisposition",
    "LeafFactOutcome",
    "LeafRelevance",
    "MergedStructuredFact",
    "ObligationCoverageRow",
    "ObligationKind",
    "OperatorObligation",
    "OperatorObligationCoverage",
    "SealedLeafDisposition",
    "SelectedHLeaf",
    "SelectedPopulationCoverage",
    "StructuredAtomicFact",
    "build_after_union_selection",
    "event_member_fingerprint",
    "merge_after_union_fact_shards",
    "replay_after_union_fact_closure",
    "replay_after_union_selection",
]
