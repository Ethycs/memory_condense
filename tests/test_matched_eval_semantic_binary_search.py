from __future__ import annotations

from dataclasses import replace

import pytest

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256
from tools.matched_eval.semantic_binary_search import (
    ProvenanceBinding,
    SemanticBinarySearchError,
    SemanticBranchDecision,
    SemanticCell,
    SemanticSearchNode,
    build_semantic_search_tree,
    make_branch_decision,
    replay_semantic_binary_search,
    semantic_binary_search,
    validate_semantic_binary_search_result,
)


QUESTION = "Which memory cell can establish the requested fact?"
QUESTION_SHA = quote_sha256(QUESTION)


def _binding(source_id: str) -> ProvenanceBinding:
    return ProvenanceBinding(source_id, quote_sha256(f"receipt:{source_id}"))


def _cell(index: int, *, multi_source: bool = False) -> SemanticCell:
    text = f"Memory cell {index} contains coherent evidence for event {index}."
    provenance = [_binding(f"source-{index}")]
    if multi_source:
        provenance.append(_binding(f"source-{index}-corroborating"))
    return SemanticCell(
        f"cell-{index}",
        f"coherence-{index}",
        text,
        count_tokens(text),
        tuple(provenance),
    )


def _tree(count: int = 4):
    return build_semantic_search_tree(
        tuple(_cell(index, multi_source=index == 0) for index in range(count))
    )


class _Classifier:
    classifier_id = "test-semantic-classifier-v1"

    def __init__(self, plan=None) -> None:
        self.plan = {} if plan is None else dict(plan)
        self.calls: list[tuple[str, ...]] = []

    def classify(
        self,
        *,
        question: str,
        node: SemanticSearchNode,
        call_ordinal: int,
    ) -> SemanticBranchDecision:
        assert question == QUESTION
        self.calls.append(node.cell_ids)
        return make_branch_decision(
            classifier_id=self.classifier_id,
            question_sha256=QUESTION_SHA,
            node_receipt_sha256=node.receipt_sha256,
            call_ordinal=call_ordinal,
            branch_classification=self.plan.get(node.cell_ids, "may_answer"),
        )


def _never_fits(_node: SemanticSearchNode) -> bool:
    return False


def test_balanced_pair_tree_preserves_multisource_leaf_order_and_replays() -> None:
    tree = _tree(5)
    rebuilt = _tree(5)

    assert tree.root.left is not None
    assert tree.root.right is not None
    assert len(tree.root.left.cells) == 2
    assert len(tree.root.right.cells) == 3
    assert tuple(row.cell_id for row in tree.cells) == tuple(
        f"cell-{value}" for value in range(5)
    )
    assert len(tree.cells[0].provenance) == 2
    assert tree.projection() == rebuilt.projection()
    assert tree.receipt_sha256 == rebuilt.receipt_sha256


def test_uncertainty_may_answer_recurses_both_branches_to_every_leaf() -> None:
    tree = _tree()
    classifier = _Classifier()

    result = semantic_binary_search(
        tree,
        QUESTION,
        classifier,
        fit_predicate=_never_fits,
        fit_policy_id="never-fit-v1",
    )

    assert result.classifier_calls == 7
    assert len(classifier.calls) == 7
    assert result.pruned_leaf_cell_ids == ()
    assert result.retained_leaf_cell_ids == (
        "cell-0",
        "cell-1",
        "cell-2",
        "cell-3",
    )
    assert tuple(row.leaf_ordinal for row in result.leaf_outcomes) == (0, 1, 2, 3)
    assert all(row.disposition == "retained" for row in result.leaf_outcomes)
    assert result.projection()["gold_loaded"] is False
    assert result.projection()["provider_calls_performed_by_core"] == 0
    assert result.projection()["retained_transformer_token_state_bytes"] == 0


def test_definitely_no_prunes_only_its_exact_subtree_and_fit_retains_other() -> None:
    tree = _tree()
    left_ids = tree.root.left.cell_ids
    right_ids = tree.root.right.cell_ids
    classifier = _Classifier({left_ids: "definitely_no"})

    result = semantic_binary_search(
        tree,
        QUESTION,
        classifier,
        fit_predicate=lambda node: node.cell_ids == right_ids,
        fit_policy_id="right-subtree-fits-v1",
    )

    assert result.classifier_calls == 3
    assert classifier.calls == [tree.root.cell_ids, left_ids, right_ids]
    assert result.pruned_leaf_cell_ids == left_ids
    assert result.retained_leaf_cell_ids == right_ids
    assert result.pruned_token_count + result.retained_token_count == tree.token_count
    assert [row.action for row in result.visits] == [
        "expanded",
        "pruned",
        "retained_fit",
    ]


def test_fit_stop_retains_entire_may_answer_root_with_one_call() -> None:
    tree = _tree()
    classifier = _Classifier()

    result = semantic_binary_search(
        tree,
        QUESTION,
        classifier,
        fit_predicate=lambda _node: True,
        fit_policy_id="root-fits-v1",
    )

    assert result.classifier_calls == 1
    assert result.visits[0].action == "retained_fit"
    assert result.retained_leaf_cell_ids == tree.root.cell_ids
    assert result.pruned_leaf_cell_ids == ()


class _MalformedClassifier:
    classifier_id = "malformed-classifier-v1"

    def classify(self, **_kwargs):
        return {"branch_classification": "may_answer"}


def test_malformed_classifier_response_fails_closed() -> None:
    with pytest.raises(
        SemanticBinarySearchError,
        match="malformed semantic branch decision",
    ):
        semantic_binary_search(
            _tree(1),
            QUESTION,
            _MalformedClassifier(),
            fit_predicate=_never_fits,
            fit_policy_id="never-fit-v1",
        )


def test_uncertain_is_not_a_third_verdict_and_must_be_encoded_may_answer() -> None:
    with pytest.raises(
        SemanticBinarySearchError,
        match="must be definitely_no or may_answer",
    ):
        SemanticBranchDecision(
            "classifier-v1",
            QUESTION_SHA,
            quote_sha256("node"),
            0,
            "uncertain",  # type: ignore[arg-type]
        )


class _TamperedClassifier(_Classifier):
    def classify(self, **kwargs) -> SemanticBranchDecision:
        decision = super().classify(**kwargs)
        object.__setattr__(decision, "receipt_sha256", "0" * 64)
        return decision


def test_tampered_classifier_decision_fails_closed() -> None:
    with pytest.raises(
        SemanticBinarySearchError,
        match="decision receipt or request binding changed",
    ):
        semantic_binary_search(
            _tree(1),
            QUESTION,
            _TamperedClassifier(),
            fit_predicate=_never_fits,
            fit_policy_id="never-fit-v1",
        )


def test_dropped_may_answer_children_fail_closed_even_with_resealed_visits() -> None:
    tree = _tree()
    result = semantic_binary_search(
        tree,
        QUESTION,
        _Classifier(),
        fit_predicate=_never_fits,
        fit_policy_id="never-fit-v1",
    )
    forged_root = replace(
        result.visits[0],
        action="retained_fit",
        fit_matched=True,
        child_node_receipt_sha256s=(),
        child_visit_receipt_sha256s=(),
        receipt_sha256="",
    )
    forged = replace(
        result,
        visits=(forged_root, *result.visits[1:]),
        receipt_sha256="",
    )

    with pytest.raises(
        SemanticBinarySearchError,
        match="fit decision differs|unreachable visits",
    ):
        validate_semantic_binary_search_result(
            tree,
            QUESTION,
            forged,
            fit_predicate=_never_fits,
        )


def test_sealed_decisions_replay_to_byte_identical_result_without_classifier() -> None:
    tree = _tree()
    left_ids = tree.root.left.cell_ids
    classifier = _Classifier({left_ids: "definitely_no"})
    result = semantic_binary_search(
        tree,
        QUESTION,
        classifier,
        fit_predicate=_never_fits,
        fit_policy_id="never-fit-v1",
    )

    replayed = replay_semantic_binary_search(
        tree,
        QUESTION,
        result,
        fit_predicate=_never_fits,
    )

    assert replayed.projection() == result.projection()
    assert replayed.receipt_sha256 == result.receipt_sha256
