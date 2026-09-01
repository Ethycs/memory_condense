"""Deterministic semantic branch-and-bound over provenance-bound memory cells.

This module owns no model client.  A caller injects a classifier which must
return one of two fail-closed branch classifications: ``definitely_no`` or
``may_answer``.  Only the former may prune.  A ``may_answer`` internal node
either stops because the caller's fit predicate accepts the complete node, or
recurses into both children.  Thus uncertainty is represented by
``may_answer`` and can never silently discard a branch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)


CELL_FORMAT = "memory-condense-semantic-search-cell-v1"
NODE_FORMAT = "memory-condense-semantic-search-node-v1"
TREE_FORMAT = "memory-condense-semantic-search-tree-v1"
DECISION_FORMAT = "memory-condense-semantic-search-branch-decision-v1"
VISIT_FORMAT = "memory-condense-semantic-search-branch-visit-v1"
LEAF_OUTCOME_FORMAT = "memory-condense-semantic-search-leaf-outcome-v1"
RESULT_FORMAT = "memory-condense-semantic-binary-search-result-v1"

BranchClassification = Literal["definitely_no", "may_answer"]
VisitAction = Literal["pruned", "retained_leaf", "retained_fit", "expanded"]
LeafDisposition = Literal["pruned", "retained"]
FitPredicate = Callable[["SemanticSearchNode"], bool]


class SemanticBinarySearchError(MatchedEvalContractError):
    """A tree, decision, traversal, coverage, or replay invariant changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise SemanticBinarySearchError(message)


def _exact_tuple(value: object, label: str) -> tuple:
    _require(type(value) is tuple, f"{label} must be an exact tuple")
    return value


def _receipt(projection: Mapping[str, object], declared: str, label: str) -> str:
    computed = identity_sha256(projection)
    if declared:
        _require(
            require_sha256(declared, label) == computed,
            f"{label} changed",
        )
    return computed


@dataclass(frozen=True, slots=True)
class ProvenanceBinding:
    """One immutable source receipt bound into a semantic cell."""

    source_id: str
    source_receipt_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_text(self.source_id, "semantic provenance source ID")
        require_sha256(
            self.source_receipt_sha256, "semantic provenance source receipt"
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "semantic provenance receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "source_id": self.source_id,
            "source_receipt_sha256": self.source_receipt_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class SemanticCell:
    """A coherent text cell with one or more exact provenance bindings."""

    cell_id: str
    coherence_id: str
    text: str
    token_count: int
    provenance: tuple[ProvenanceBinding, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_text(self.cell_id, "semantic cell ID")
        require_text(self.coherence_id, "semantic cell coherence ID")
        _require(type(self.text) is str and bool(self.text), "semantic cell text changed")
        _require(
            type(self.token_count) is int
            and self.token_count >= 0
            and self.token_count == count_tokens(self.text),
            "semantic cell token count changed",
        )
        provenance = _exact_tuple(self.provenance, "semantic cell provenance")
        _require(
            bool(provenance)
            and all(type(row) is ProvenanceBinding for row in provenance)
            and len({row.source_id for row in provenance}) == len(provenance),
            "semantic cell requires ordered unique provenance bindings",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "semantic cell receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "cell_id": self.cell_id,
            "coherence_id": self.coherence_id,
            "format": CELL_FORMAT,
            "provenance": [row.projection() for row in self.provenance],
            "text": self.text,
            "token_count": self.token_count,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class SemanticSearchNode:
    """One immutable contiguous node in the balanced pair tree."""

    span_start: int
    span_end: int
    cells: tuple[SemanticCell, ...]
    left: "SemanticSearchNode | None" = None
    right: "SemanticSearchNode | None" = None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        cells = _exact_tuple(self.cells, "semantic node cells")
        _require(
            type(self.span_start) is int
            and type(self.span_end) is int
            and 0 <= self.span_start < self.span_end
            and len(cells) == self.span_end - self.span_start
            and all(type(row) is SemanticCell for row in cells)
            and len({row.cell_id for row in cells}) == len(cells),
            "semantic node span/cell population changed",
        )
        leaf = len(cells) == 1
        if leaf:
            _require(
                self.left is None and self.right is None,
                "semantic leaf cannot have children",
            )
        else:
            _require(
                type(self.left) is SemanticSearchNode
                and type(self.right) is SemanticSearchNode,
                "semantic internal node requires an exact child pair",
            )
            assert self.left is not None and self.right is not None
            _require(
                self.left.span_start == self.span_start
                and self.left.span_end == self.right.span_start
                and self.right.span_end == self.span_end
                and self.left.cells + self.right.cells == cells
                and abs(len(self.left.cells) - len(self.right.cells)) <= 1,
                "semantic node children changed balanced contiguous coverage",
            )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "semantic node receipt",
            ),
        )

    @property
    def node_id(self) -> str:
        return f"N{self.span_start:06d}-{self.span_end:06d}"

    @property
    def token_count(self) -> int:
        return sum(row.token_count for row in self.cells)

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return tuple(row.cell_id for row in self.cells)

    @property
    def is_leaf(self) -> bool:
        return len(self.cells) == 1

    @property
    def children(self) -> tuple["SemanticSearchNode", ...]:
        if self.is_leaf:
            return ()
        assert self.left is not None and self.right is not None
        return (self.left, self.right)

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "cell_ids": list(self.cell_ids),
            "cell_receipt_sha256s": [row.receipt_sha256 for row in self.cells],
            "child_node_receipt_sha256s": [
                row.receipt_sha256 for row in self.children
            ],
            "format": NODE_FORMAT,
            "is_leaf": self.is_leaf,
            "node_id": self.node_id,
            "span_end": self.span_end,
            "span_start": self.span_start,
            "token_count": self.token_count,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _preorder_nodes(node: SemanticSearchNode) -> tuple[SemanticSearchNode, ...]:
    return (node, *(child for row in node.children for child in _preorder_nodes(row)))


@dataclass(frozen=True, slots=True)
class SemanticSearchTree:
    """The complete deterministic tree and exact ordered leaf population."""

    cells: tuple[SemanticCell, ...]
    root: SemanticSearchNode
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        cells = _exact_tuple(self.cells, "semantic tree cells")
        _require(
            bool(cells)
            and all(type(row) is SemanticCell for row in cells)
            and len({row.cell_id for row in cells}) == len(cells)
            and type(self.root) is SemanticSearchNode
            and self.root.span_start == 0
            and self.root.span_end == len(cells)
            and self.root.cells == cells,
            "semantic tree root/cell population changed",
        )
        nodes = _preorder_nodes(self.root)
        _require(
            len(nodes) == 2 * len(cells) - 1
            and tuple(row.cells[0] for row in nodes if row.is_leaf) == cells
            and len({row.receipt_sha256 for row in nodes}) == len(nodes),
            "semantic tree lost exact ordered binary leaf coverage",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "semantic tree receipt",
            ),
        )

    @property
    def nodes(self) -> tuple[SemanticSearchNode, ...]:
        return _preorder_nodes(self.root)

    @property
    def token_count(self) -> int:
        return sum(row.token_count for row in self.cells)

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "cells": [row.projection() for row in self.cells],
            "format": TREE_FORMAT,
            "node_count": len(self.nodes),
            "nodes": [row.projection() for row in self.nodes],
            "ordered_leaf_cell_ids": [row.cell_id for row in self.cells],
            "root_node_receipt_sha256": self.root.receipt_sha256,
            "token_count": self.token_count,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _build_node(
    cells: tuple[SemanticCell, ...], start: int, end: int
) -> SemanticSearchNode:
    selected = cells[start:end]
    if end - start == 1:
        return SemanticSearchNode(start, end, selected)
    midpoint = start + (end - start) // 2
    return SemanticSearchNode(
        start,
        end,
        selected,
        _build_node(cells, start, midpoint),
        _build_node(cells, midpoint, end),
    )


def build_semantic_search_tree(
    cells: Sequence[SemanticCell],
) -> SemanticSearchTree:
    """Build a balanced contiguous pair tree without reordering any cell."""

    values = tuple(cells)
    _require(
        bool(values) and all(type(row) is SemanticCell for row in values),
        "semantic search requires at least one exact cell",
    )
    return SemanticSearchTree(values, _build_node(values, 0, len(values)))


def validate_semantic_search_tree(tree: SemanticSearchTree) -> None:
    _require(type(tree) is SemanticSearchTree, "semantic tree changed type")
    for cell in tree.cells:
        for binding in cell.provenance:
            _require(
                identity_sha256(binding.projection(include_receipt=False))
                == binding.receipt_sha256,
                "semantic provenance receipt changed after construction",
            )
        _require(
            identity_sha256(cell.projection(include_receipt=False))
            == cell.receipt_sha256,
            "semantic cell receipt changed after construction",
        )
    for node in reversed(tree.nodes):
        _require(
            identity_sha256(node.projection(include_receipt=False))
            == node.receipt_sha256,
            "semantic node receipt changed after construction",
        )
    _require(
        identity_sha256(tree.projection(include_receipt=False))
        == tree.receipt_sha256,
        "semantic tree receipt changed after construction",
    )


@dataclass(frozen=True, slots=True)
class SemanticBranchDecision:
    """One sealed classifier response for one exact tree node."""

    classifier_id: str
    question_sha256: str
    node_receipt_sha256: str
    call_ordinal: int
    branch_classification: BranchClassification
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_text(self.classifier_id, "semantic classifier ID")
        require_sha256(self.question_sha256, "semantic decision question")
        require_sha256(self.node_receipt_sha256, "semantic decision node")
        _require(
            type(self.call_ordinal) is int and self.call_ordinal >= 0,
            "semantic decision call ordinal changed",
        )
        _require(
            type(self.branch_classification) is str
            and self.branch_classification in {"definitely_no", "may_answer"},
            "semantic branch classification must be definitely_no or may_answer",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "semantic branch decision receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "branch_classification": self.branch_classification,
            "call_ordinal": self.call_ordinal,
            "classifier_id": self.classifier_id,
            "format": DECISION_FORMAT,
            "node_receipt_sha256": self.node_receipt_sha256,
            "question_sha256": self.question_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def make_branch_decision(
    *,
    classifier_id: str,
    question_sha256: str,
    node_receipt_sha256: str,
    call_ordinal: int,
    branch_classification: BranchClassification,
) -> SemanticBranchDecision:
    return SemanticBranchDecision(
        classifier_id,
        question_sha256,
        node_receipt_sha256,
        call_ordinal,
        branch_classification,
    )


@runtime_checkable
class SemanticBranchClassifier(Protocol):
    """Injectable classifier boundary; uncertainty must return may_answer."""

    classifier_id: str

    def classify(
        self,
        *,
        question: str,
        node: SemanticSearchNode,
        call_ordinal: int,
    ) -> SemanticBranchDecision: ...


@dataclass(frozen=True, slots=True)
class BranchVisit:
    node_receipt_sha256: str
    decision_receipt_sha256: str
    branch_classification: BranchClassification
    action: VisitAction
    fit_evaluated: bool
    fit_matched: bool
    covered_leaf_cell_ids: tuple[str, ...]
    child_node_receipt_sha256s: tuple[str, ...] = ()
    child_visit_receipt_sha256s: tuple[str, ...] = ()
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.node_receipt_sha256, "semantic visit node")
        require_sha256(self.decision_receipt_sha256, "semantic visit decision")
        _require(
            type(self.branch_classification) is str
            and self.branch_classification in {"definitely_no", "may_answer"}
            and type(self.action) is str
            and self.action
            in {"pruned", "retained_leaf", "retained_fit", "expanded"}
            and type(self.fit_evaluated) is bool
            and type(self.fit_matched) is bool,
            "semantic visit classification/action changed",
        )
        covered = _exact_tuple(
            self.covered_leaf_cell_ids, "semantic visit covered leaves"
        )
        child_nodes = _exact_tuple(
            self.child_node_receipt_sha256s, "semantic visit child nodes"
        )
        child_visits = _exact_tuple(
            self.child_visit_receipt_sha256s, "semantic visit child visits"
        )
        _require(
            bool(covered)
            and all(type(value) is str and bool(value) for value in covered)
            and len(set(covered)) == len(covered)
            and len(child_nodes) == len(child_visits)
            and len(child_nodes) in {0, 2},
            "semantic visit leaf/child coverage changed",
        )
        for value in (*child_nodes, *child_visits):
            require_sha256(value, "semantic visit child receipt")
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "semantic branch visit receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "action": self.action,
            "branch_classification": self.branch_classification,
            "child_node_receipt_sha256s": list(
                self.child_node_receipt_sha256s
            ),
            "child_visit_receipt_sha256s": list(
                self.child_visit_receipt_sha256s
            ),
            "covered_leaf_cell_ids": list(self.covered_leaf_cell_ids),
            "decision_receipt_sha256": self.decision_receipt_sha256,
            "fit_evaluated": self.fit_evaluated,
            "fit_matched": self.fit_matched,
            "format": VISIT_FORMAT,
            "node_receipt_sha256": self.node_receipt_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class LeafOutcome:
    leaf_ordinal: int
    cell_id: str
    disposition: LeafDisposition
    deciding_node_receipt_sha256: str
    decision_receipt_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            type(self.leaf_ordinal) is int and self.leaf_ordinal >= 0,
            "semantic leaf ordinal changed",
        )
        require_text(self.cell_id, "semantic outcome cell ID")
        _require(
            type(self.disposition) is str
            and self.disposition in {"pruned", "retained"},
            "semantic leaf disposition changed",
        )
        require_sha256(self.deciding_node_receipt_sha256, "deciding node")
        require_sha256(self.decision_receipt_sha256, "deciding decision")
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "semantic leaf outcome receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "cell_id": self.cell_id,
            "deciding_node_receipt_sha256": self.deciding_node_receipt_sha256,
            "decision_receipt_sha256": self.decision_receipt_sha256,
            "disposition": self.disposition,
            "format": LEAF_OUTCOME_FORMAT,
            "leaf_ordinal": self.leaf_ordinal,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class SemanticBinarySearchResult:
    tree_receipt_sha256: str
    question_sha256: str
    question_token_count: int
    classifier_id: str
    fit_policy_id: str
    decisions: tuple[SemanticBranchDecision, ...]
    visits: tuple[BranchVisit, ...]
    leaf_outcomes: tuple[LeafOutcome, ...]
    retained_leaf_cell_ids: tuple[str, ...]
    pruned_leaf_cell_ids: tuple[str, ...]
    retained_token_count: int
    pruned_token_count: int
    classified_node_token_count: int
    classifier_calls: int
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.tree_receipt_sha256, "semantic result tree")
        require_sha256(self.question_sha256, "semantic result question")
        require_text(self.classifier_id, "semantic result classifier")
        require_text(self.fit_policy_id, "semantic result fit policy")
        for values, label, expected in (
            (self.decisions, "semantic result decisions", SemanticBranchDecision),
            (self.visits, "semantic result visits", BranchVisit),
            (self.leaf_outcomes, "semantic result outcomes", LeafOutcome),
        ):
            _require(
                type(values) is tuple
                and all(type(row) is expected for row in values),
                f"{label} changed type",
            )
        for values, label in (
            (self.retained_leaf_cell_ids, "retained leaf IDs"),
            (self.pruned_leaf_cell_ids, "pruned leaf IDs"),
        ):
            _exact_tuple(values, label)
        for value, label in (
            (self.question_token_count, "question tokens"),
            (self.retained_token_count, "retained tokens"),
            (self.pruned_token_count, "pruned tokens"),
            (self.classified_node_token_count, "classified node tokens"),
            (self.classifier_calls, "classifier calls"),
        ):
            _require(type(value) is int and value >= 0, f"semantic {label} changed")
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "semantic binary search result receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "classified_node_token_count": self.classified_node_token_count,
            "classifier_calls": self.classifier_calls,
            "classifier_id": self.classifier_id,
            "decisions": [row.projection() for row in self.decisions],
            "fit_policy_id": self.fit_policy_id,
            "format": RESULT_FORMAT,
            "gold_loaded": False,
            "leaf_outcomes": [row.projection() for row in self.leaf_outcomes],
            "provider_calls_performed_by_core": 0,
            "pruned_leaf_cell_ids": list(self.pruned_leaf_cell_ids),
            "pruned_token_count": self.pruned_token_count,
            "question_sha256": self.question_sha256,
            "question_token_count": self.question_token_count,
            "retained_leaf_cell_ids": list(self.retained_leaf_cell_ids),
            "retained_token_count": self.retained_token_count,
            "retained_transformer_token_state_bytes": 0,
            "tree_receipt_sha256": self.tree_receipt_sha256,
            "visits": [row.projection() for row in self.visits],
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _validate_decision(
    decision: object,
    *,
    classifier_id: str,
    question_sha256: str,
    node: SemanticSearchNode,
    call_ordinal: int,
) -> SemanticBranchDecision:
    _require(
        type(decision) is SemanticBranchDecision,
        "classifier returned a malformed semantic branch decision",
    )
    assert type(decision) is SemanticBranchDecision
    _require(
        identity_sha256(decision.projection(include_receipt=False))
        == decision.receipt_sha256
        and decision.classifier_id == classifier_id
        and decision.question_sha256 == question_sha256
        and decision.node_receipt_sha256 == node.receipt_sha256
        and decision.call_ordinal == call_ordinal
        and decision.branch_classification in {"definitely_no", "may_answer"},
        "classifier decision receipt or request binding changed",
    )
    return decision


def semantic_binary_search(
    tree: SemanticSearchTree,
    question: str,
    classifier: SemanticBranchClassifier,
    *,
    fit_predicate: FitPredicate,
    fit_policy_id: str,
) -> SemanticBinarySearchResult:
    """Classify, prune, and retain an exact leaf partition without provider IO."""

    validate_semantic_search_tree(tree)
    require_text(question, "semantic search question")
    _require(callable(fit_predicate), "semantic fit predicate must be callable")
    classifier_id = require_text(
        getattr(classifier, "classifier_id", None), "semantic classifier ID"
    )
    fit_policy_id = require_text(fit_policy_id, "semantic fit policy ID")
    question_sha = quote_sha256(question)
    decisions: list[SemanticBranchDecision] = []

    def walk(
        node: SemanticSearchNode,
    ) -> tuple[BranchVisit, tuple[BranchVisit, ...], tuple[LeafOutcome, ...]]:
        call_ordinal = len(decisions)
        decision = _validate_decision(
            classifier.classify(
                question=question,
                node=node,
                call_ordinal=call_ordinal,
            ),
            classifier_id=classifier_id,
            question_sha256=question_sha,
            node=node,
            call_ordinal=call_ordinal,
        )
        decisions.append(decision)
        if decision.branch_classification == "definitely_no":
            action: VisitAction = "pruned"
            fit_evaluated = fit_matched = False
            child_nodes: tuple[str, ...] = ()
            child_visits: tuple[str, ...] = ()
            nested_visits: tuple[BranchVisit, ...] = ()
            outcomes = tuple(
                LeafOutcome(
                    cell_ordinal,
                    cell.cell_id,
                    "pruned",
                    node.receipt_sha256,
                    decision.receipt_sha256,
                )
                for cell_ordinal, cell in zip(
                    range(node.span_start, node.span_end), node.cells, strict=True
                )
            )
        elif node.is_leaf:
            action = "retained_leaf"
            fit_evaluated = fit_matched = False
            child_nodes = child_visits = ()
            nested_visits = ()
            outcomes = (
                LeafOutcome(
                    node.span_start,
                    node.cells[0].cell_id,
                    "retained",
                    node.receipt_sha256,
                    decision.receipt_sha256,
                ),
            )
        else:
            fit_value = fit_predicate(node)
            _require(type(fit_value) is bool, "semantic fit predicate changed type")
            fit_evaluated = True
            fit_matched = fit_value
            if fit_value:
                action = "retained_fit"
                child_nodes = child_visits = ()
                nested_visits = ()
                outcomes = tuple(
                    LeafOutcome(
                        cell_ordinal,
                        cell.cell_id,
                        "retained",
                        node.receipt_sha256,
                        decision.receipt_sha256,
                    )
                    for cell_ordinal, cell in zip(
                        range(node.span_start, node.span_end), node.cells, strict=True
                    )
                )
            else:
                action = "expanded"
                left, right = node.children
                left_visit, left_visits, left_outcomes = walk(left)
                right_visit, right_visits, right_outcomes = walk(right)
                child_nodes = (left.receipt_sha256, right.receipt_sha256)
                child_visits = (
                    left_visit.receipt_sha256,
                    right_visit.receipt_sha256,
                )
                nested_visits = (*left_visits, *right_visits)
                outcomes = (*left_outcomes, *right_outcomes)
        visit = BranchVisit(
            node.receipt_sha256,
            decision.receipt_sha256,
            decision.branch_classification,
            action,
            fit_evaluated,
            fit_matched,
            node.cell_ids,
            child_nodes,
            child_visits,
        )
        return visit, (visit, *nested_visits), outcomes

    _root_visit, visits, outcomes = walk(tree.root)
    retained = tuple(
        row.cell_id for row in outcomes if row.disposition == "retained"
    )
    pruned = tuple(row.cell_id for row in outcomes if row.disposition == "pruned")
    cells_by_id = {row.cell_id: row for row in tree.cells}
    nodes_by_receipt = {row.receipt_sha256: row for row in tree.nodes}
    result = SemanticBinarySearchResult(
        tree.receipt_sha256,
        question_sha,
        count_tokens(question),
        classifier_id,
        fit_policy_id,
        tuple(decisions),
        visits,
        outcomes,
        retained,
        pruned,
        sum(cells_by_id[value].token_count for value in retained),
        sum(cells_by_id[value].token_count for value in pruned),
        sum(nodes_by_receipt[row.node_receipt_sha256].token_count for row in visits),
        len(decisions),
    )
    validate_semantic_binary_search_result(
        tree,
        question,
        result,
        fit_predicate=fit_predicate,
    )
    return result


def validate_semantic_binary_search_result(
    tree: SemanticSearchTree,
    question: str,
    result: SemanticBinarySearchResult,
    *,
    fit_predicate: FitPredicate | None = None,
) -> None:
    """Fail closed on altered decisions, missing may branches, or leaf loss."""

    validate_semantic_search_tree(tree)
    _require(type(result) is SemanticBinarySearchResult, "semantic result changed type")
    require_text(question, "semantic validation question")
    if fit_predicate is not None:
        _require(callable(fit_predicate), "semantic validation fit predicate changed")
    _require(
        result.tree_receipt_sha256 == tree.receipt_sha256
        and result.question_sha256 == quote_sha256(question)
        and result.question_token_count == count_tokens(question)
        and result.classifier_calls == len(result.decisions) == len(result.visits)
        and result.classifier_calls > 0
        and identity_sha256(result.projection(include_receipt=False))
        == result.receipt_sha256,
        "semantic result envelope or receipt changed",
    )
    nodes = {row.receipt_sha256: row for row in tree.nodes}
    _require(len(nodes) == len(tree.nodes), "semantic tree node receipts repeat")
    visit_index = 0
    generated: list[LeafOutcome] = []

    def consume(node: SemanticSearchNode) -> BranchVisit:
        nonlocal visit_index
        _require(
            visit_index < len(result.visits),
            "may-answer traversal dropped a required branch",
        )
        visit = result.visits[visit_index]
        decision = result.decisions[visit_index]
        current = visit_index
        visit_index += 1
        _validate_decision(
            decision,
            classifier_id=result.classifier_id,
            question_sha256=result.question_sha256,
            node=node,
            call_ordinal=current,
        )
        _require(
            identity_sha256(visit.projection(include_receipt=False))
            == visit.receipt_sha256
            and visit.node_receipt_sha256 == node.receipt_sha256
            and visit.decision_receipt_sha256 == decision.receipt_sha256
            and visit.branch_classification == decision.branch_classification
            and visit.covered_leaf_cell_ids == node.cell_ids,
            "semantic visit receipt or node coverage changed",
        )
        if decision.branch_classification == "definitely_no":
            _require(
                visit.action == "pruned"
                and not visit.fit_evaluated
                and not visit.fit_matched
                and not visit.child_node_receipt_sha256s
                and not visit.child_visit_receipt_sha256s,
                "only definitely_no may prune a semantic branch",
            )
            generated.extend(
                LeafOutcome(
                    ordinal,
                    cell.cell_id,
                    "pruned",
                    node.receipt_sha256,
                    decision.receipt_sha256,
                )
                for ordinal, cell in zip(
                    range(node.span_start, node.span_end), node.cells, strict=True
                )
            )
            return visit
        if node.is_leaf:
            _require(
                visit.action == "retained_leaf"
                and not visit.fit_evaluated
                and not visit.fit_matched
                and not visit.child_node_receipt_sha256s
                and not visit.child_visit_receipt_sha256s,
                "may-answer leaf must be retained",
            )
            generated.append(
                LeafOutcome(
                    node.span_start,
                    node.cells[0].cell_id,
                    "retained",
                    node.receipt_sha256,
                    decision.receipt_sha256,
                )
            )
            return visit
        _require(visit.fit_evaluated, "may-answer internal node skipped fit policy")
        if fit_predicate is not None:
            fit_value = fit_predicate(node)
            _require(
                type(fit_value) is bool and fit_value == visit.fit_matched,
                "sealed fit decision differs from caller fit predicate",
            )
        if visit.fit_matched:
            _require(
                visit.action == "retained_fit"
                and not visit.child_node_receipt_sha256s
                and not visit.child_visit_receipt_sha256s,
                "fit-stopped may-answer node changed action",
            )
            generated.extend(
                LeafOutcome(
                    ordinal,
                    cell.cell_id,
                    "retained",
                    node.receipt_sha256,
                    decision.receipt_sha256,
                )
                for ordinal, cell in zip(
                    range(node.span_start, node.span_end), node.cells, strict=True
                )
            )
            return visit
        _require(
            visit.action == "expanded"
            and visit.child_node_receipt_sha256s
            == tuple(row.receipt_sha256 for row in node.children),
            "non-fitting may-answer node failed to retain both branches",
        )
        child_visits = tuple(consume(child) for child in node.children)
        _require(
            visit.child_visit_receipt_sha256s
            == tuple(row.receipt_sha256 for row in child_visits),
            "may-answer child visit receipts changed or a branch was dropped",
        )
        return visit

    consume(tree.root)
    _require(
        visit_index == len(result.visits)
        and tuple(generated) == result.leaf_outcomes,
        "semantic traversal has unreachable visits or altered leaf outcomes",
    )
    ordered_ids = tuple(row.cell_id for row in tree.cells)
    outcome_ids = tuple(row.cell_id for row in result.leaf_outcomes)
    retained = tuple(
        row.cell_id for row in result.leaf_outcomes if row.disposition == "retained"
    )
    pruned = tuple(
        row.cell_id for row in result.leaf_outcomes if row.disposition == "pruned"
    )
    cells = {row.cell_id: row for row in tree.cells}
    _require(
        outcome_ids == ordered_ids
        and tuple(row.leaf_ordinal for row in result.leaf_outcomes)
        == tuple(range(len(tree.cells)))
        and retained == result.retained_leaf_cell_ids
        and pruned == result.pruned_leaf_cell_ids
        and set(retained).isdisjoint(pruned)
        and set(retained) | set(pruned) == set(ordered_ids)
        and result.retained_token_count
        == sum(cells[value].token_count for value in retained)
        and result.pruned_token_count
        == sum(cells[value].token_count for value in pruned)
        and result.retained_token_count + result.pruned_token_count
        == tree.token_count
        and result.classified_node_token_count
        == sum(nodes[row.node_receipt_sha256].token_count for row in result.visits),
        "semantic result lost exact ordered leaf/token coverage",
    )
    assert_gold_blind(result.projection(), path="semantic_binary_search_result")


class _ReplayClassifier:
    def __init__(
        self, classifier_id: str, decisions: tuple[SemanticBranchDecision, ...]
    ) -> None:
        self.classifier_id = classifier_id
        self._decisions = decisions
        self.consumed = 0

    def classify(
        self,
        *,
        question: str,
        node: SemanticSearchNode,
        call_ordinal: int,
    ) -> SemanticBranchDecision:
        del question, node
        _require(
            call_ordinal == self.consumed < len(self._decisions),
            "semantic replay requested an unsealed classifier call",
        )
        decision = self._decisions[self.consumed]
        self.consumed += 1
        return decision


def replay_semantic_binary_search(
    tree: SemanticSearchTree,
    question: str,
    sealed_result: SemanticBinarySearchResult,
    *,
    fit_predicate: FitPredicate,
) -> SemanticBinarySearchResult:
    """Replay a result from only its sealed decisions and require byte identity."""

    validate_semantic_binary_search_result(
        tree, question, sealed_result, fit_predicate=fit_predicate
    )
    classifier = _ReplayClassifier(
        sealed_result.classifier_id, sealed_result.decisions
    )
    replayed = semantic_binary_search(
        tree,
        question,
        classifier,
        fit_predicate=fit_predicate,
        fit_policy_id=sealed_result.fit_policy_id,
    )
    _require(
        classifier.consumed == len(sealed_result.decisions)
        and replayed.projection() == sealed_result.projection()
        and replayed.receipt_sha256 == sealed_result.receipt_sha256,
        "semantic binary-search replay differs from sealed result",
    )
    return replayed


__all__ = [
    "BranchClassification",
    "BranchVisit",
    "FitPredicate",
    "LeafDisposition",
    "LeafOutcome",
    "ProvenanceBinding",
    "SemanticBinarySearchError",
    "SemanticBinarySearchResult",
    "SemanticBranchClassifier",
    "SemanticBranchDecision",
    "SemanticCell",
    "SemanticSearchNode",
    "SemanticSearchTree",
    "build_semantic_search_tree",
    "make_branch_decision",
    "replay_semantic_binary_search",
    "semantic_binary_search",
    "validate_semantic_binary_search_result",
    "validate_semantic_search_tree",
]
