from __future__ import annotations

import copy
from dataclasses import asdict, fields
from inspect import signature
import pickle

import pytest

import tools.matched_eval.semantic_binary_search as search
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256
from tools.matched_eval.contracts import identity_sha256


def _tree() -> search.SemanticSearchTree:
    texts = (
        "alpha memory",
        "beta memory evidence",
        "gamma memory evidence record",
    )
    cells = tuple(
        search.SemanticCell(
            f"cell-{index}",
            f"coherence-{index}",
            text,
            count_tokens(text),
            (
                search.ProvenanceBinding(
                    f"source-{index}",
                    quote_sha256(f"receipt:source-{index}"),
                ),
            ),
        )
        for index, text in enumerate(texts)
    )
    return search.build_semantic_search_tree(cells)


def test_caches_preserve_constructor_equality_and_projected_bytes() -> None:
    tree = _tree()
    rebuilt = _tree()

    assert tuple(signature(search.SemanticSearchNode).parameters) == (
        "span_start",
        "span_end",
        "cells",
        "left",
        "right",
        "receipt_sha256",
    )
    assert tuple(signature(search.SemanticSearchTree).parameters) == (
        "cells",
        "root",
        "receipt_sha256",
    )
    field_names = {
        value.name
        for value in (
            *fields(search.SemanticSearchNode),
            *fields(search.SemanticSearchTree),
        )
    }
    assert not {
        "_nodes_cache",
        "_token_count_cache",
    } & field_names
    serialized = asdict(tree)
    assert "_nodes_cache" not in serialized
    assert "_token_count_cache" not in serialized
    assert "_token_count_cache" not in serialized["root"]
    assert "cells" not in serialized["root"]

    assert tree == rebuilt
    assert hash(tree) == hash(rebuilt)
    assert "_cache" not in repr(tree)
    assert tree.receipt_sha256 == (
        "5318e7a93e25f2be543db8aa9918e7b2f06a7b2917280e7ac457996c9ae77fd9"
    )
    assert identity_sha256(tree.projection()) == (
        "341d907fb48e4f03d58c3c9486d65713e897f11ab514f525812ad1f675b32b9e"
    )
    assert tuple(node.node_id for node in tree.nodes) == (
        "N000000-000003",
        "N000000-000001",
        "N000001-000003",
        "N000001-000002",
        "N000002-000003",
    )
    assert tuple(node.token_count for node in tree.nodes) == (9, 2, 7, 3, 4)


def test_repeated_tree_access_reuses_one_preorder_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _tree()
    nodes = tree.nodes
    projection = tree.projection()

    def fail_preorder(
        _node: search.SemanticSearchNode,
    ) -> tuple[search.SemanticSearchNode, ...]:
        raise AssertionError("preorder population was recomputed")

    monkeypatch.setattr(search, "_preorder_nodes", fail_preorder)

    assert tree.nodes is nodes
    assert tree.nodes is nodes
    assert tree.projection() == projection
    search.validate_semantic_search_tree(tree)


def test_every_node_reuses_one_shared_leaf_population() -> None:
    tree = _tree()

    assert all(node._cell_population is tree.cells for node in tree.nodes)
    assert len({id(node._cell_population) for node in tree.nodes}) == 1
    assert tuple(tuple(row.cell_id for row in node.iter_cells()) for node in tree.nodes) == (
        ("cell-0", "cell-1", "cell-2"),
        ("cell-0",),
        ("cell-1", "cell-2"),
        ("cell-1",),
        ("cell-2",),
    )
    # The compatibility property remains an exact tuple, but it is a view
    # materialized on demand rather than retained on every node.
    assert type(tree.root.cells) is tuple
    assert tree.root.cells == tree.cells


def test_repeated_token_access_uses_non_projected_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _tree()
    projection = tree.projection()
    node_token_counts = tuple(node.token_count for node in tree.nodes)

    def fail_sum(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("descendant token counts were recomputed")

    monkeypatch.setattr(search, "sum", fail_sum, raising=False)

    assert tree.token_count == 9
    assert tree.token_count == 9
    assert tuple(node.token_count for node in tree.nodes) == node_token_counts
    assert tuple(node.projection()["token_count"] for node in tree.nodes) == (
        9,
        2,
        7,
        3,
        4,
    )
    assert tree.projection() == projection
    search.validate_semantic_search_tree(tree)


def test_tampered_internal_token_cache_still_fails_validation() -> None:
    tree = _tree()
    object.__setattr__(tree.root, "_token_count_cache", tree.root.token_count + 1)

    with pytest.raises(search.SemanticBinarySearchError, match="node receipt changed"):
        search.validate_semantic_search_tree(tree)


def test_copy_and_pickle_keep_internal_caches_usable() -> None:
    tree = _tree()

    assert copy.copy(tree) is tree
    assert copy.deepcopy(tree) is tree
    assert copy.copy(tree.root) is tree.root
    assert copy.deepcopy(tree.root) is tree.root

    restored = pickle.loads(pickle.dumps(tree))
    assert restored == tree
    assert restored is not tree
    assert restored.nodes == tree.nodes
    assert restored.token_count == tree.token_count
    assert restored.projection() == tree.projection()
    assert all(node._cell_population is restored.cells for node in restored.nodes)
    search.validate_semantic_search_tree(restored)
