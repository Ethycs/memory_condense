from __future__ import annotations

import inspect

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools.matched_eval.after_union_fact_closure import (
    AfterUnionFactClosureError,
    CrossBoundaryEdge,
    FactOutcomeShard,
    LeafFactOutcome,
    OperatorObligation,
    SealedLeafDisposition,
    SelectedHLeaf,
    StructuredAtomicFact,
    build_after_union_selection,
    merge_after_union_fact_shards,
    replay_after_union_fact_closure,
    replay_after_union_selection,
)
from tools.matched_eval.typed_fact_compiler import (
    CompiledTypedFact,
    TypedFactCitation,
)


QUESTION = "Which purchases and events across my memories satisfy this request?"
QUESTION_SHA = quote_sha256(QUESTION)
CLASSIFIER_ID = "sealed-test-r-i-u-v1"


def _leaf(
    handle: str,
    group: str,
    text: str,
    *,
    topics: tuple[str, ...] = (),
    boundaries: tuple[str, ...] = (),
    edge_ids: tuple[str, ...] = (),
) -> SelectedHLeaf:
    return SelectedHLeaf(
        handle,
        group,
        text,
        quote_sha256(text),
        topic_labels=topics,
        boundary_labels=boundaries,
        cross_boundary_edge_ids=edge_ids,
    )


def _disposition(
    leaf: SelectedHLeaf,
    value: str,
) -> SealedLeafDisposition:
    return SealedLeafDisposition(
        leaf.handle_id,
        leaf.receipt_sha256,
        QUESTION_SHA,
        CLASSIFIER_ID,
        value,  # type: ignore[arg-type]
    )


def _atomic_fact(
    leaf: SelectedHLeaf,
    *,
    fact_id: str,
    quote: str,
    kind: str,
    entity: str | None,
    predicate: str,
    member_key: str | None,
    obligation_ids: tuple[str, ...],
    event_time: str | None = None,
    source_time: str | None = None,
    status: str | None = "completed",
    numeric_value: float | None = None,
    unit: str | None = None,
    qualifiers: tuple[str, ...] = (),
    source_index: int = 0,
) -> StructuredAtomicFact:
    citation = TypedFactCitation(
        leaf.handle_id,
        leaf.group_handle,
        quote,
        quote_sha256(quote),
        quote_sha256(leaf.text),
        source_index,
    )
    compiled = CompiledTypedFact(
        fact_id,
        quote,
        kind,
        entity,
        numeric_value,
        unit,
        event_time,
        status,
        obligation_ids,
        (citation,),
        (),
        1.0,
        source_index,
    )
    return StructuredAtomicFact(
        leaf.handle_id,
        compiled,
        predicate,
        member_key,
        event_time,
        source_time,
        qualifiers,
        obligation_ids,
    )


def _outcome(
    leaf: SelectedHLeaf,
    disposition: SealedLeafDisposition,
    value: str,
    *,
    facts: tuple[StructuredAtomicFact, ...] = (),
    unresolved: tuple[str, ...] = (),
) -> LeafFactOutcome:
    return LeafFactOutcome(
        leaf.handle_id,
        leaf.receipt_sha256,
        disposition.receipt_sha256,
        value,  # type: ignore[arg-type]
        facts,
        unresolved,
    )


def test_sealed_r_i_u_drives_full_leaf_descent_and_only_i_prunes() -> None:
    leaves = (
        _leaf("H001", "G001", "A relevant memory."),
        _leaf("H002", "G002", "A sealed irrelevant memory."),
        _leaf("H003", "G003", "An uncertain memory."),
    )
    dispositions = (
        _disposition(leaves[2], "uncertain"),
        _disposition(leaves[0], "relevant"),
        _disposition(leaves[1], "definitely_irrelevant"),
    )

    selection = build_after_union_selection(QUESTION, leaves, dispositions)

    assert selection.semantic_result.classifier_calls == 5
    assert selection.semantic_result.retained_leaf_cell_ids == ("H001", "H003")
    assert selection.semantic_result.pruned_leaf_cell_ids == ("H002",)
    assert tuple(row.handle_id for row in selection.dispositions) == (
        "H001",
        "H002",
        "H003",
    )
    assert selection.projection()["provider_calls_performed_by_core"] == 0
    assert replay_after_union_selection(QUESTION, selection).projection() == (
        selection.projection()
    )


def test_multitopic_boundary_metadata_never_blocks_cross_group_composition() -> None:
    edge = CrossBoundaryEdge(
        "E-trip-concert",
        "event",
        "H001",
        "H002",
        "same_trip_concert_purchase",
    )
    leaves = (
        _leaf(
            "H001",
            "G001",
            "I bought the concert ticket for 90 USD on 2026-06-10.",
            topics=("music", "travel"),
            boundaries=("event", "purchase"),
            edge_ids=(edge.edge_id,),
        ),
        _leaf(
            "H002",
            "G002",
            "I booked the trip hotel for 140 USD on 2026-06-10.",
            topics=("travel", "finance"),
            boundaries=("purchase", "lodging"),
            edge_ids=(edge.edge_id,),
        ),
        _leaf(
            "H003",
            "G003",
            "I attended an unrelated cooking class.",
            topics=("cooking", "events"),
        ),
    )
    dispositions = (
        _disposition(leaves[0], "relevant"),
        _disposition(leaves[1], "relevant"),
        _disposition(leaves[2], "definitely_irrelevant"),
    )
    selection = build_after_union_selection(
        QUESTION,
        leaves,
        dispositions,
        cross_boundary_edges=(edge,),
    )
    ticket = _atomic_fact(
        leaves[0],
        fact_id="F001",
        quote="bought the concert ticket for 90 USD on 2026-06-10",
        kind="operand",
        entity="concert ticket",
        predicate="purchase",
        member_key="concert ticket",
        obligation_ids=("concert-cost",),
        event_time="2026-06-10",
        numeric_value=90.0,
        unit="USD",
    )
    hotel = _atomic_fact(
        leaves[1],
        fact_id="F002",
        quote="booked the trip hotel for 140 USD on 2026-06-10",
        kind="operand",
        entity="trip hotel",
        predicate="purchase",
        member_key="trip hotel",
        obligation_ids=("hotel-cost",),
        event_time="2026-06-10",
        numeric_value=140.0,
        unit="USD",
    )
    shards = (
        FactOutcomeShard(
            "music-shard",
            selection.receipt_sha256,
            (
                _outcome(leaves[0], dispositions[0], "facts", facts=(ticket,)),
                _outcome(leaves[2], dispositions[2], "definitely_irrelevant"),
            ),
        ),
        FactOutcomeShard(
            "travel-finance-shard",
            selection.receipt_sha256,
            (_outcome(leaves[1], dispositions[1], "facts", facts=(hotel,)),),
        ),
    )
    obligations = (
        OperatorObligation("concert-cost", "operand", "Concert purchase cost"),
        OperatorObligation("hotel-cost", "operand", "Hotel purchase cost"),
    )

    closure = merge_after_union_fact_shards(selection, obligations, shards)

    assert selection.semantic_result.retained_leaf_cell_ids == ("H001", "H002")
    assert selection.semantic_result.pruned_leaf_cell_ids == ("H003",)
    assert [row.group_handle for row in selection.leaves[:2]] == ["G001", "G002"]
    assert {handle for row in closure.merged_facts for handle in row.leaf_handle_ids} == {
        "H001",
        "H002",
    }
    assert closure.selected_population_coverage.selected_population_resolved is True
    assert (
        closure.operator_obligation_coverage
        .required_obligations_closed_within_selected_population
        is True
    )


def test_multiple_facts_per_handle_and_unresolved_coverage_are_explicit() -> None:
    leaves = (
        _leaf(
            "H001",
            "G001",
            "I picked up a peace lily. I also bought a succulent.",
        ),
        _leaf("H002", "G002", "I bought a kettle."),
        _leaf("H003", "G003", "I picked up some live plants."),
    )
    dispositions = (
        _disposition(leaves[0], "relevant"),
        _disposition(leaves[1], "definitely_irrelevant"),
        _disposition(leaves[2], "uncertain"),
    )
    selection = build_after_union_selection(QUESTION, leaves, dispositions)
    peace_lily = _atomic_fact(
        leaves[0],
        fact_id="F001",
        quote="picked up a peace lily",
        kind="member",
        entity="peace lily",
        predicate="acquire",
        member_key="peace lily",
        obligation_ids=("plant-members",),
    )
    succulent = _atomic_fact(
        leaves[0],
        fact_id="F002",
        quote="bought a succulent",
        kind="member",
        entity="succulent",
        predicate="acquire",
        member_key="succulent",
        obligation_ids=("plant-members",),
        source_index=1,
    )
    shard_a = FactOutcomeShard(
        "plants-resolved",
        selection.receipt_sha256,
        (
            _outcome(
                leaves[0],
                dispositions[0],
                "facts",
                facts=(peace_lily, succulent),
            ),
            _outcome(leaves[1], dispositions[1], "definitely_irrelevant"),
        ),
    )
    shard_b = FactOutcomeShard(
        "plants-open",
        selection.receipt_sha256,
        (
            _outcome(
                leaves[2],
                dispositions[2],
                "unresolved",
                unresolved=("plant-members",),
            ),
        ),
    )
    obligations = (
        OperatorObligation(
            "plant-members",
            "member",
            "Every distinct acquired plant member",
        ),
    )

    left = merge_after_union_fact_shards(
        selection,
        obligations,
        (shard_a, shard_b),
    )
    right = merge_after_union_fact_shards(
        selection,
        obligations,
        (shard_b, shard_a),
    )

    assert left.projection() == right.projection()
    assert left.receipt_sha256 == right.receipt_sha256
    assert len(left.leaf_outcomes[0].facts) == 2
    assert len(left.merged_facts) == 2
    assert left.selected_population_coverage.exact_outcome_coverage is True
    assert left.selected_population_coverage.unresolved_leaf_ids == ("H003",)
    assert left.selected_population_coverage.selected_population_resolved is False
    coverage = left.operator_obligation_coverage.rows[0]
    assert coverage.covered is True
    assert coverage.unresolved_leaf_ids == ("H003",)
    assert coverage.closed_within_selected_population is False
    assert replay_after_union_fact_closure(
        selection,
        obligations,
        (shard_b, shard_a),
        left,
    ).projection() == left.projection()


def test_structured_fingerprint_dedups_same_event_across_unlinked_groups() -> None:
    leaves = (
        _leaf(
            "H001",
            "G001",
            "On June 4 I completed servicing my road bike.",
        ),
        _leaf(
            "H002",
            "G002",
            "The road-bike service was completed on June 4.",
        ),
    )
    dispositions = tuple(_disposition(row, "relevant") for row in leaves)
    selection = build_after_union_selection(QUESTION, leaves, dispositions)
    first = _atomic_fact(
        leaves[0],
        fact_id="F001",
        quote="completed servicing my road bike",
        kind="event",
        entity="road bike",
        predicate="service",
        member_key="road bike",
        obligation_ids=("bike-services",),
        event_time="2026-06-04",
        source_time="2026-06-05",
    )
    second = _atomic_fact(
        leaves[1],
        fact_id="F002",
        quote="road-bike service was completed",
        kind="event",
        entity="road bike",
        predicate="service",
        member_key="road bike",
        obligation_ids=("bike-services",),
        event_time="2026-06-04",
        source_time="2026-06-07",
        source_index=1,
    )
    shards = (
        FactOutcomeShard(
            "first",
            selection.receipt_sha256,
            (_outcome(leaves[0], dispositions[0], "facts", facts=(first,)),),
        ),
        FactOutcomeShard(
            "second",
            selection.receipt_sha256,
            (_outcome(leaves[1], dispositions[1], "facts", facts=(second,)),),
        ),
    )
    obligations = (
        OperatorObligation("bike-services", "event", "Distinct bike services"),
    )

    closure = merge_after_union_fact_shards(selection, obligations, shards)

    assert len(closure.merged_facts) == 1
    assert closure.merged_facts[0].leaf_handle_ids == ("H001", "H002")
    assert len(closure.merged_facts[0].citations) == 2
    assert closure.merged_facts[0].fingerprint.temporal_identity == "2026-06-04"
    assert (
        closure.operator_obligation_coverage
        .required_obligations_closed_within_selected_population
        is True
    )


def test_missing_leaf_outcome_and_nonexact_citation_fail_closed() -> None:
    leaf = _leaf("H001", "G001", "I bought the exact blue lamp.")
    disposition = _disposition(leaf, "relevant")
    selection = build_after_union_selection(QUESTION, (leaf,), (disposition,))
    obligation = OperatorObligation("lamp", "member", "Purchased lamp")

    with pytest.raises(
        AfterUnionFactClosureError,
        match="cover the exact selected leaf population",
    ):
        merge_after_union_fact_shards(
            selection,
            (obligation,),
            (
                FactOutcomeShard(
                    "wrong-population",
                    selection.receipt_sha256,
                    (
                        LeafFactOutcome(
                            "H999",
                            quote_sha256("leaf"),
                            quote_sha256("disposition"),
                            "unresolved",
                            (),
                            ("lamp",),
                        ),
                    ),
                ),
            ),
        )

    bad_citation = TypedFactCitation(
        "H001",
        "G001",
        "quote absent from selected leaf",
        quote_sha256("quote absent from selected leaf"),
        quote_sha256(leaf.text),
        0,
    )
    compiled = CompiledTypedFact(
        "F001",
        "quote absent from selected leaf",
        "member",
        "blue lamp",
        None,
        None,
        None,
        "completed",
        ("lamp",),
        (bad_citation,),
        (),
        1.0,
        0,
    )
    bad_fact = StructuredAtomicFact(
        "H001",
        compiled,
        "purchase",
        "blue lamp",
        obligation_ids=("lamp",),
    )
    bad_outcome = _outcome(
        leaf,
        disposition,
        "facts",
        facts=(bad_fact,),
    )

    with pytest.raises(
        AfterUnionFactClosureError,
        match="not exact selected-leaf evidence",
    ):
        merge_after_union_fact_shards(
            selection,
            (obligation,),
            (
                FactOutcomeShard(
                    "bad-citation",
                    selection.receipt_sha256,
                    (bad_outcome,),
                ),
            ),
        )


def test_public_core_has_no_parent_benchmark_or_allowlist_inputs() -> None:
    for function in (build_after_union_selection, merge_after_union_fact_shards):
        parameters = set(inspect.signature(function).parameters)
        assert parameters.isdisjoint(
            {
                "parent",
                "parent_prediction",
                "gold",
                "reference",
                "ordinal",
                "source_allowlist",
                "semantic_atom_manifest",
            }
        )

    leaf = _leaf("H001", "G001", "One exact memory.")
    selection = build_after_union_selection(
        QUESTION,
        (leaf,),
        (_disposition(leaf, "uncertain"),),
    )
    projection = selection.projection()
    assert projection["gold_loaded"] is False
    assert projection["provider_calls_performed_by_core"] == 0
