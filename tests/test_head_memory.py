from __future__ import annotations

import torch

from memory_condense.head_memory import (
    AssociativeMemoryCandidate,
    CAVLinkIndex,
    HeadAssociationGraph,
    HeadKVStore,
    HeadMemoryItem,
    _rank_association_walk,
    compose_associative_candidates,
)


def _item(
    episode_id: str,
    key: list[list[float]],
    value: list[list[float]],
    signature: list[float],
    *,
    importance: float = 0.0,
    pinned: bool = False,
    residual: list[float] | None = None,
) -> HeadMemoryItem:
    return HeadMemoryItem(
        episode_id=episode_id,
        text=episode_id,
        keys=torch.tensor(key),
        values=torch.tensor(value),
        cav_signature=torch.tensor(signature),
        residual=None if residual is None else torch.tensor(residual),
        importance=importance,
        pinned=pinned,
    )


def test_gqa_addressing_retrieves_and_mixes_live_values() -> None:
    store = HeadKVStore(
        query_heads=4,
        key_value_heads=2,
        head_dim=2,
        device=torch.device("cpu"),
        head_vote_k=2,
    )
    store.write(
        _item("x", [[1.0, 0.0], [0.0, 1.0]], [[10.0, 0.0], [0.0, 20.0]], [1.0])
    )
    store.write(
        _item("y", [[-1.0, 0.0], [0.0, -1.0]], [[-10.0, 0.0], [0.0, -20.0]], [-1.0])
    )
    queries = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )

    result = store.address(queries, top_k=1, scaling=1.0)

    assert result.indices.tolist() == [0]
    assert result.head_weights.shape == (4, 1, 2)
    assert result.mixed_values.shape == (1, 4, 2)
    assert torch.all(result.head_weights[:, :, 0] > result.head_weights[:, :, 1])
    assert torch.all(result.mixed_values[0, :2, 0] > 0)
    assert torch.all(result.mixed_values[0, 2:, 1] > 0)


def test_cav_gating_can_break_a_qk_tie() -> None:
    store = HeadKVStore(
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
        device=torch.device("cpu"),
    )
    store.write(_item("near", [[1.0, 0.0]], [[1.0, 0.0]], [2.0]))
    store.write(_item("far", [[1.0, 0.0]], [[0.0, 1.0]], [-2.0]))

    result = store.address(
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        top_k=1,
        scaling=1.0,
        cav_signature=torch.tensor([1.5]),
        cav_weight=1.0,
    )

    assert result.indices.tolist() == [0]


def test_address_can_limit_live_attention_to_candidate_items() -> None:
    store = HeadKVStore(
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
        device=torch.device("cpu"),
    )
    store.write(_item("excluded", [[1.0, 0.0]], [[9.0, 0.0]], [0.0]))
    store.write(_item("candidate_a", [[0.0, 1.0]], [[0.0, 2.0]], [0.0]))
    store.write(_item("candidate_b", [[0.0, -1.0]], [[0.0, -2.0]], [0.0]))

    result = store.address(
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        top_k=2,
        scaling=1.0,
        candidate_indices=[1, 2],
    )

    assert set(result.indices.tolist()) == {1, 2}
    assert result.head_weights.shape[-1] == 2
    assert result.slot_ranges[0] == (0, 0)


def test_pruning_preserves_pins_and_prefers_accessed_items() -> None:
    store = HeadKVStore(
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
        device=torch.device("cpu"),
    )
    store.write(_item("old", [[1.0, 0.0]], [[1.0, 0.0]], [0.0]))
    store.write(_item("used", [[0.0, 1.0]], [[0.0, 1.0]], [0.0]))
    store.write(
        _item("pinned", [[-1.0, 0.0]], [[-1.0, 0.0]], [0.0], pinned=True)
    )
    store.touch(torch.tensor([1]))

    removed = store.prune(2, age_half_life=1.0)

    assert removed == ["old"]
    assert {item.episode_id for item in store.items} == {"used", "pinned"}


def test_pruning_uses_decayed_qk_mass_and_ov_transport() -> None:
    store = HeadKVStore(
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
        device=torch.device("cpu"),
    )
    store.write(_item("head_used", [[1.0, 0.0]], [[1.0, 0.0]], [0.0]))
    store.write(_item("unread", [[0.0, 1.0]], [[0.0, 1.0]], [0.0]))
    store.touch(
        torch.tensor([0]),
        attention_mass=torch.tensor([4.0]),
        ov_transport=torch.tensor([3.0]),
    )

    removed = store.prune(1, age_half_life=100.0)

    assert removed == ["unread"]
    survivor = store.items[0]
    assert survivor.qk_attention_mass == 4.0
    assert survivor.ov_transport == 3.0


def test_episode_index_lookup_is_rebuilt_after_pruning() -> None:
    store = HeadKVStore(
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
        device=torch.device("cpu"),
    )
    store.write(
        _item("first", [[1.0, 0.0]], [[1.0, 0.0]], [0.0], pinned=True)
    )
    store.write(
        _item(
            "remove",
            [[0.0, 1.0]],
            [[0.0, 1.0]],
            [0.0],
            importance=-100.0,
        )
    )
    store.write(_item("last", [[-1.0, 0.0]], [[-1.0, 0.0]], [0.0]))

    assert store.indices_for_episode_ids(["last", "first", "last"]) == [2, 0]

    assert store.prune(2) == ["remove"]
    assert store.indices_for_episode_ids(["last", "missing", "first"]) == [1, 0]


def test_residual_entry_and_positive_cav_prior_are_separate_scores() -> None:
    store = HeadKVStore(
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
        device=torch.device("cpu"),
    )
    store.write(
        _item(
            "semantic",
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [-2.0],
            residual=[1.0, 0.0],
        )
    )
    store.write(
        _item(
            "typed",
            [[0.0, 1.0]],
            [[0.0, 1.0]],
            [2.0],
            residual=[0.8, 0.2],
        )
    )

    direct = store.residual_scores(torch.tensor([1.0, 0.0]))
    gated = store.residual_scores(
        torch.tensor([1.0, 0.0]), cav_weight=1.0, cav_mode="positive"
    )

    assert direct.argmax().item() == 0
    assert gated.argmax().item() == 1


def test_association_layer_residuals_can_be_scored_without_a_second_store() -> None:
    store = HeadKVStore(
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
        device=torch.device("cpu"),
    )
    store.write(
        _item(
            "entry",
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [0.0],
            residual=[1.0, 0.0],
        )
    )
    store.write(
        _item(
            "association",
            [[0.0, 1.0]],
            [[0.0, 1.0]],
            [0.0],
            residual=[0.0, 1.0],
        )
    )
    store.items[0].association_residual = torch.tensor([0.0, 1.0])
    store.items[1].association_residual = torch.tensor([1.0, 0.0])

    entry_scores = store.residual_scores(torch.tensor([1.0, 0.0]))
    association_scores = store.residual_scores(
        torch.tensor([1.0, 0.0]), association_layer=True
    )

    assert entry_scores.argmax().item() == 0
    assert association_scores.argmax().item() == 1


def test_association_graph_combines_repeated_head_evidence() -> None:
    graph = HeadAssociationGraph()
    graph.add("new", "old", torch.tensor([0.1, 0.9, 0.2, 0.8]))
    graph.add("new", "old", torch.tensor([0.3, 0.7, 0.4, 0.6]))

    edge = graph.neighbors("new")[0]

    assert graph.edge_count == 2  # forward and reverse
    assert edge.destination_id == "old"
    assert edge.evidence_count == 2
    assert torch.allclose(edge.head_weights, torch.tensor([0.2, 0.8, 0.3, 0.7]))


def test_association_graph_removes_pruned_sources_and_destinations() -> None:
    graph = HeadAssociationGraph()
    graph.add("a", "b", torch.tensor([0.8, 0.2]))
    graph.add("b", "c", torch.tensor([0.7, 0.3]))

    removed_edges = graph.remove_episode_ids(["b"])

    assert removed_edges == 4
    assert graph.edge_count == 0
    assert graph.neighbors("a") == ()
    assert graph.neighbors("b") == ()


def test_association_graph_bounds_degree_with_qk_and_ov_utility() -> None:
    graph = HeadAssociationGraph()
    graph.add(
        "source",
        "weak",
        torch.tensor([0.1, 0.1]),
        reverse=False,
        ov_transport=0.0,
    )
    graph.add(
        "source",
        "useful",
        torch.tensor([0.2, 0.2]),
        reverse=False,
        ov_transport=2.0,
    )

    assert graph.prune_neighbors(1) == 1
    edge = graph.neighbors("source")[0]
    assert edge.destination_id == "useful"
    assert edge.ov_transport == 2.0


def test_composition_recycles_only_duplicate_anchor_slots() -> None:
    anchors = [
        AssociativeMemoryCandidate("a", "same fact"),
        AssociativeMemoryCandidate("a-copy", "  SAME   fact  "),
        AssociativeMemoryCandidate("b", "second fact"),
        AssociativeMemoryCandidate("b-copy", "second fact"),
    ]
    qk = [
        AssociativeMemoryCandidate("q-duplicate", "same fact", route="qk"),
        AssociativeMemoryCandidate("q", "QK neighbor", route="qk"),
    ]
    residual = [
        AssociativeMemoryCandidate("a", "different text", route="residual"),
        AssociativeMemoryCandidate("r", "residual candidate", route="residual"),
    ]

    result = compose_associative_candidates(
        anchors,
        qk_neighbors=qk,
        residual_candidates=residual,
        top_k=4,
        qk_reserve=1,
    )

    assert [candidate.episode_id for candidate in result.candidates] == [
        "a",
        "b",
        "q",
        "r",
    ]
    assert result.duplicates_removed == 2
    assert result.qk_added == 1
    assert result.residual_added == 1


def test_composition_never_displaces_unique_direct_anchors() -> None:
    anchors = [
        AssociativeMemoryCandidate("a", "first"),
        AssociativeMemoryCandidate("b", "second"),
    ]

    result = compose_associative_candidates(
        anchors,
        qk_neighbors=[AssociativeMemoryCandidate("q", "neighbor", route="qk")],
        residual_candidates=[
            AssociativeMemoryCandidate("r", "residual", route="residual")
        ],
        top_k=2,
    )

    assert [candidate.episode_id for candidate in result.candidates] == ["a", "b"]
    assert result.qk_added == 0
    assert result.residual_added == 0


def test_composition_can_reserve_a_fixed_association_budget() -> None:
    anchors = [
        AssociativeMemoryCandidate("a", "first"),
        AssociativeMemoryCandidate("b", "second"),
        AssociativeMemoryCandidate("c", "third"),
    ]

    result = compose_associative_candidates(
        anchors,
        qk_neighbors=[AssociativeMemoryCandidate("q", "neighbor", route="qk")],
        residual_candidates=[
            AssociativeMemoryCandidate("r", "concept", route="cav")
        ],
        top_k=3,
        qk_reserve=1,
        association_slots=2,
    )

    assert [candidate.episode_id for candidate in result.candidates] == [
        "a",
        "q",
        "r",
    ]
    assert result.anchors_displaced == 2


def test_reserved_association_slots_backfill_with_direct_anchors() -> None:
    anchors = [
        AssociativeMemoryCandidate("a", "first"),
        AssociativeMemoryCandidate("b", "second"),
        AssociativeMemoryCandidate("c", "third"),
    ]

    result = compose_associative_candidates(
        anchors,
        top_k=3,
        association_slots=2,
    )

    assert [candidate.episode_id for candidate in result.candidates] == [
        "a",
        "b",
        "c",
    ]
    assert result.anchors_displaced == 0


def test_cav_link_index_links_episodes_and_concepts_compactly() -> None:
    index = CAVLinkIndex(("retrieval", "pruning", "unrelated"))
    index.add("seed", (2.0, 1.0, -1.0))
    index.add("related", (1.5, 0.5, -2.0))
    index.add("partial", (0.5, -1.0, -1.0))
    index.add("noise", (-1.0, -1.0, 2.0))

    neighbors = index.neighbors(["seed"], top_k=3)

    assert [hit.episode_id for hit in neighbors] == ["related", "partial"]
    assert neighbors[0].shared_concepts == ("retrieval", "pruning")
    assert index.concept_neighbors("retrieval")[0] == ("pruning", 2)
    assert index.signature_bytes == 4 * 3 * 4

    assert index.remove(["related", "missing"]) == 1
    assert index.episode_count == 3
    assert index.concept_neighbors("retrieval")[0] == ("pruning", 1)


def test_association_graph_selects_heads_that_recover_known_links() -> None:
    graph = HeadAssociationGraph()
    graph.add("anchor_a", "fact_a", torch.tensor([0.9, 0.1, 0.8, 0.2]))
    graph.add("anchor_a", "noise_a", torch.tensor([0.1, 0.9, 0.2, 0.8]))
    graph.add("anchor_b", "fact_b", torch.tensor([0.8, 0.2, 0.7, 0.3]))
    graph.add("anchor_b", "noise_b", torch.tensor([0.2, 0.8, 0.3, 0.7]))

    result = graph.calibrate_heads(
        [("anchor_a", "fact_a"), ("anchor_b", "fact_b")], keep=2
    )

    assert set(result["selected_heads"]) == {0, 2}
    assert graph.neighbors("anchor_a")[0].destination_id == "fact_a"


def test_association_calibration_tolerates_an_empty_graph() -> None:
    graph = HeadAssociationGraph()

    result = graph.calibrate_heads([("missing", "also_missing")])

    assert result == {"selected_heads": [], "head_mrr": []}
    assert graph.selected_heads == ()


def test_association_walk_reranks_and_reinforces_a_seeded_destination() -> None:
    graph = HeadAssociationGraph()
    graph.add("fact", "anchor", torch.tensor([0.4, 0.4]))
    graph.calibrate_heads([("anchor", "fact")], keep=2)

    ranked, hops = _rank_association_walk(
        graph,
        [("anchor", 0.8), ("fact", 0.7)],
        top_k=2,
        hops=1,
    )

    assert [episode_id for episode_id, _ in ranked] == ["fact", "anchor"]
    assert hops == (("anchor", "fact"), ())


def test_uncalibrated_association_walk_preserves_semantic_seed_order() -> None:
    graph = HeadAssociationGraph()
    graph.add("fact", "anchor", torch.tensor([0.9, 0.9]))

    ranked, _ = _rank_association_walk(
        graph,
        [("anchor", 0.8), ("distractor", 0.7)],
        top_k=3,
        hops=1,
    )

    assert [episode_id for episode_id, _ in ranked] == [
        "anchor",
        "distractor",
        "fact",
    ]
