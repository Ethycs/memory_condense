from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from memory_condense.application.discourse_sources import scan_discourse_source_chunks
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.schemas import Chunk
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.search.indexes.lexical import LexicalIndex

from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.full_store_slot_closure import build_full_store_window_index
from tools.matched_eval.hot_typed_witness import (
    HotTypedWitnessBudget,
    HotTypedWitnessError,
    assess_ordered_list_ambiguity,
    build_hot_typed_witness_index,
    hot_typed_witness_applicable,
    query_hot_typed_witnesses,
)
from tools.matched_eval.query_expansion import FrozenSourceNamespace
from tools.matched_eval.query_guided_scan import cache_namespace_partitions
from tools.matched_eval.typed_operator_spec import compile_typed_operator_spec


ASKED = datetime(2026, 8, 27, 12, tzinfo=timezone.utc)


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _build_index(
    path: Path,
    rows: list[tuple[str, str, datetime] | tuple[str, str, datetime, str]],
):
    database = Database(path)
    transcript = TranscriptStore(database)
    lexical = LexicalIndex(database)
    for number, raw in enumerate(rows):
        source_id, text, created_at, *role = raw
        turn = transcript.append(
            role[0] if role else "user",
            text,
            source_id=source_id,
            created_at=created_at,
        )
        lexical.add_chunks(
            [
                Chunk(
                    chunk_id=f"chunk-{number}",
                    turn_id=turn.turn_id,
                    text=text,
                    start_char=0,
                    end_char=len(text),
                    token_count=count_tokens(text),
                )
            ]
        )
    streams = scan_discourse_source_chunks(database)
    database.close()
    store_receipt = _sha("store")
    namespace = FrozenSourceNamespace.from_source_streams(
        snapshot_id=_sha("snapshot"),
        combined_store_receipt_sha256=store_receipt,
        source_streams=streams,
    )
    with Database(path, read_only=True) as readonly:
        cache = cache_namespace_partitions(
            readonly,
            namespace,
            source_database_sha256=_sha("database"),
            source_store_receipt_sha256=store_receipt,
        )
    return build_hot_typed_witness_index(build_full_store_window_index(cache))


def _question(body: str) -> str:
    return f"[Question asked at 2026/08/27 12:00] {body}"


def test_index_precomputes_clause_scoped_completed_and_planned_actions(
    tmp_path: Path,
) -> None:
    index = _build_index(
        tmp_path / "clauses.db",
        [
            (
                "travel::story",
                "I plan to visit Oslo; I visited Rome with my friend.",
                ASKED - timedelta(days=2),
            )
        ],
    )

    planned = tuple(index.windows[value].quote for value in index.planned_action_postings["visit"])
    completed = tuple(
        index.windows[value].quote
        for value in index.completed_action_postings["visit"]
    )

    assert planned == ("I plan to visit Oslo",)
    assert completed == ("I visited Rome with my friend.",)
    assert index.projection()["new_provider_calls"] == 0
    assert index.projection()["model_calls"] == 0
    assert index.projection()["retained_transformer_token_state_bytes"] == 0


def test_latest_visit_with_participant_rejects_planned_only_clause(
    tmp_path: Path,
) -> None:
    index = _build_index(
        tmp_path / "participant.db",
        [
            (
                "travel::old",
                "I visited Madrid with my friend.",
                ASKED - timedelta(days=20),
            ),
            (
                "travel::new",
                "I plan to visit Oslo; I visited Rome with my friend.",
                ASKED - timedelta(days=2),
            ),
        ],
    )

    result = query_hot_typed_witnesses(
        index, _question("What was the last place I visited with my friend?")
    )

    assert result.witnesses[0].quote == "I visited Rome with my friend."
    assert result.witnesses[0].completed_action_concepts == ("visit",)
    assert result.witnesses[0].matched_participant_terms == ("friend",)
    assert all("plan to visit Oslo" not in row.quote for row in result.witnesses)


def test_exact_day_action_postings_recover_lexically_unknown_object(
    tmp_path: Path,
) -> None:
    index = _build_index(
        tmp_path / "exact-day.db",
        [
            (
                "appliance::purchase",
                "I bought a smoker for the patio.",
                ASKED - timedelta(days=10),
            ),
            (
                "calendar::noise",
                "I reviewed the quarterly calendar.",
                ASKED - timedelta(days=10),
            ),
            (
                "assistant::echo",
                "You bought a smoker for the patio.",
                ASKED - timedelta(days=10),
                "assistant",
            ),
        ],
    )

    result = query_hot_typed_witnesses(
        index, _question("What did I acquire ten days ago?")
    )

    target = next(row for row in result.witnesses if "smoker" in row.quote)
    assert target.completed_action_concepts == ("acquire",)
    assert target.event_date == "2026-08-17"
    assert target.temporal_distance_days == 0
    assert "exact_temporal_target" in target.selection_axes
    assert result.receipt.effective_evidence_role == "user"
    assert all(row.span.role == "user" for row in result.witnesses)


def test_participation_family_links_completed_took_part_and_participated(
    tmp_path: Path,
) -> None:
    index = _build_index(
        tmp_path / "participation.db",
        [
            (
                "sport::spring",
                "I completed the Spring Sprint Triathlon.",
                ASKED - timedelta(days=90),
            ),
            (
                "sport::summer",
                "I took part in the Midsummer 5K Run.",
                ASKED - timedelta(days=60),
            ),
            (
                "sport::autumn",
                "I participated in the company charity soccer tournament.",
                ASKED - timedelta(days=30),
            ),
        ],
    )

    result = query_hot_typed_witnesses(
        index, _question("In what order did I participate in sporting events?")
    )

    assert len(result.witnesses) == 3
    assert all(
        "participate" in row.completed_action_concepts for row in result.witnesses
    )
    assert result.applicable is True
    assert result.status == "applicable_witnesses_available"


def test_specific_multi_action_query_keeps_completed_clauses_not_proposal(
    tmp_path: Path,
) -> None:
    index = _build_index(
        tmp_path / "multi-action.db",
        [
            (
                "chores::one",
                "I bought a chair; I assembled a desk; I sold a table; "
                "I fixed a cabinet; I plan to buy a sofa.",
                ASKED - timedelta(days=1),
            )
        ],
    )

    result = query_hot_typed_witnesses(
        index,
        _question(
            "How many furniture items did I buy, assemble, sell, or fix?"
        ),
    )
    quotes = {row.quote for row in result.witnesses}

    assert {
        "I bought a chair",
        "I assembled a desk",
        "I sold a table",
        "I fixed a cabinet",
    } <= quotes
    assert all("plan to buy" not in quote for quote in quotes)
    assert result.receipt.generic_completed_action_query is True


def test_temporal_order_recovers_completed_travel_events_across_sources(
    tmp_path: Path,
) -> None:
    index = _build_index(
        tmp_path / "travel-order.db",
        [
            ("trip::peru", "I went hiking in Peru.", ASKED - timedelta(days=30)),
            (
                "trip::utah",
                "I took a road trip through Utah.",
                ASKED - timedelta(days=20),
            ),
            (
                "trip::yosemite",
                "I camped in Yosemite.",
                ASKED - timedelta(days=12),
            ),
            ("trip::lima", "I plan to travel to Lima.", ASKED - timedelta(days=1)),
        ],
    )

    result = query_hot_typed_witnesses(
        index,
        _question(
            "In what order did I take trips involving hiking, a road trip, "
            "and camping?"
        ),
    )
    quotes = {row.quote for row in result.witnesses}

    assert {
        "I went hiking in Peru.",
        "I took a road trip through Utah.",
        "I camped in Yosemite.",
    } <= quotes
    assert "I plan to travel to Lima." not in quotes
    assert all("travel" in row.completed_action_concepts for row in result.witnesses)


@pytest.mark.parametrize(("count_text", "expected"), (("three", 3), ("12", 12)))
def test_ordered_list_compiles_explicit_cardinality_not_window_duration(
    count_text: str,
    expected: int,
) -> None:
    spec = compile_typed_operator_spec(
        _question(
            f"What is the order of the {count_text} trips I took in the "
            "past three months, from earliest to latest?"
        )
    )

    assert spec.cardinality == expected
    assert spec.temporal_window_days == 93


def test_source_scoped_ordered_travel_recovers_phrasal_completed_events(
    tmp_path: Path,
) -> None:
    targets = {
        "trip::ridge",
        "trip::coast",
        "trip::forest",
    }
    index = _build_index(
        tmp_path / "phrasal-travel.db",
        [
            (
                "trip::ridge",
                "I just got back from a day hike to Cedar Ridge with my family.",
                ASKED - timedelta(days=80),
            ),
            (
                "trip::coast",
                "I went on a road trip through the coast with friends.",
                ASKED - timedelta(days=50),
            ),
            (
                "trip::forest",
                "I started my solo camping trip in the national forest.",
                ASKED - timedelta(days=20),
            ),
            ("noise::lunch", "I got back from lunch.", ASKED - timedelta(days=15)),
            ("noise::desk", "I returned from my desk.", ASKED - timedelta(days=10)),
            ("noise::work", "I started a project.", ASKED - timedelta(days=5)),
        ],
    )
    question = _question(
        "What is the order of the three trips I took in the past three "
        "months, from earliest to latest?"
    )

    result = query_hot_typed_witnesses(
        index,
        question,
        eligible_source_ids=targets,
    )
    travel_quotes = {
        index.windows[position].quote
        for position in index.completed_action_postings["travel"]
    }

    assert result.operator_spec.cardinality == 3
    assert {row.source_id for row in result.witnesses} == targets
    assert len(result.witnesses) == 3
    assert all(row.span.role == "user" for row in result.witnesses)
    assert all(row.event_date is not None for row in result.witnesses)
    assert all("travel" in row.completed_action_concepts for row in result.witnesses)
    assert "I got back from lunch." not in travel_quotes
    assert "I returned from my desk." not in travel_quotes
    assert "I started a project." not in travel_quotes
    scoped_decision = assess_ordered_list_ambiguity(result)
    assert scoped_decision.escalate is False
    assert scoped_decision.reason == "scoped_query_not_eligible"


def test_ordered_list_ambiguity_escalates_only_above_twice_requested_count(
    tmp_path: Path,
) -> None:
    destinations = (
        "Cedar Ridge",
        "North Beach",
        "Pine Valley",
        "Lake Basin",
        "Red Canyon",
        "South Coast",
        "Willow Park",
    )
    broad_index = _build_index(
        tmp_path / "ambiguous-travel.db",
        [
            (
                f"trip::{position}",
                f"I took a trip to {destination}.",
                ASKED - timedelta(days=85 - position * 10),
            )
            for position, destination in enumerate(destinations)
        ],
    )
    travel_question = _question(
        "What is the order of the three trips I took in the past three "
        "months, from earliest to latest?"
    )

    broad_result = query_hot_typed_witnesses(broad_index, travel_question)
    broad_decision = assess_ordered_list_ambiguity(broad_result)

    assert broad_decision.requested_cardinality == 3
    assert broad_decision.ambiguity_threshold == 6
    assert broad_decision.distinct_selected_source_count == 7
    assert broad_decision.candidate_population_count == 7
    assert broad_decision.trigger_axes == (
        "distinct_selected_sources",
        "candidate_population",
    )
    assert broad_decision.escalate is True
    assert broad_decision.reason == "ordered_list_ambiguity_exceeds_threshold"

    narrow_index = _build_index(
        tmp_path / "bounded-sports.db",
        [
            (
                "sport::spring",
                "I completed the Spring Sprint Triathlon.",
                ASKED - timedelta(days=25),
            ),
            (
                "sport::summer",
                "I took part in the Midsummer 5K Run.",
                ASKED - timedelta(days=12),
            ),
        ],
    )
    sports_result = query_hot_typed_witnesses(
        narrow_index,
        _question(
            "What is the order of the three sports events I participated in "
            "during the past month, from earliest to latest?"
        ),
    )
    sports_decision = assess_ordered_list_ambiguity(sports_result)

    assert sports_decision.requested_cardinality == 3
    assert sports_decision.distinct_selected_source_count == 2
    assert sports_decision.candidate_population_count == 2
    assert sports_decision.trigger_axes == ()
    assert sports_decision.escalate is False
    assert sports_decision.reason == "ambiguity_threshold_not_exceeded"


def test_date_only_business_milestone_uses_typed_action_and_exact_provenance(
    tmp_path: Path,
) -> None:
    index = _build_index(
        tmp_path / "milestone.db",
        [
            (
                "business::studio",
                "My design studio opened on 2026-04-03.",
                datetime(2026, 4, 3, tzinfo=timezone.utc),
            ),
            (
                "business::noise",
                "I sketched a logo on 2026-04-04.",
                datetime(2026, 4, 4, tzinfo=timezone.utc),
            ),
        ],
    )

    result = query_hot_typed_witnesses(
        index, _question("On what date did my business studio open?")
    )
    target = next(row for row in result.witnesses if "studio opened" in row.quote)
    source_row = next(row for row in index.parent.rows if row.chunk_id == target.span.chunk_id)

    assert target.completed_action_concepts == ("business_milestone",)
    assert target.event_date == "2026-04-03"
    assert source_row.text[target.span.start_char : target.span.end_char] == target.quote
    assert target.indexed_window_receipt_sha256
    assert target.receipt_sha256


def test_business_milestone_links_signed_contract_and_launched_website(
    tmp_path: Path,
) -> None:
    index = _build_index(
        tmp_path / "business-order.db",
        [
            (
                "business::contract",
                "I signed the company contract on 2026-03-01.",
                datetime(2026, 3, 1, tzinfo=timezone.utc),
            ),
            (
                "business::website",
                "I launched the company website on 2026-03-12.",
                datetime(2026, 3, 12, tzinfo=timezone.utc),
            ),
        ],
    )

    result = query_hot_typed_witnesses(
        index,
        _question(
            "Which business milestone came first, signing the contract or "
            "launching the website?"
        ),
    )

    assert {row.event_date for row in result.witnesses} == {
        "2026-03-01",
        "2026-03-12",
    }
    assert all(
        "business_milestone" in row.completed_action_concepts
        for row in result.witnesses
    )


def test_colloquial_relative_business_milestone_reserves_bounded_frontier(
    tmp_path: Path,
) -> None:
    index = _build_index(
        tmp_path / "relative-business.db",
        [
            (
                "business::website",
                "I launched my website and drafted a business plan.",
                datetime(2026, 7, 10, tzinfo=timezone.utc),
            ),
            (
                "wellness::exact-noise",
                "I started taking yoga classes.",
                datetime(2026, 7, 30, tzinfo=timezone.utc),
            ),
            (
                "business::contract",
                "I signed a contract with my first client today.",
                datetime(2026, 7, 31, tzinfo=timezone.utc),
            ),
            (
                "assistant::echo",
                "I signed a contract with my first client today.",
                datetime(2026, 7, 30, tzinfo=timezone.utc),
                "assistant",
            ),
            (
                "friend::decoy",
                "My friend signed a contract with her first client, and I congratulated her.",
                datetime(2026, 7, 31, tzinfo=timezone.utc),
            ),
            (
                "business::outside",
                "I opened my business today.",
                datetime(2026, 8, 2, tzinfo=timezone.utc),
            ),
        ],
    )

    result = query_hot_typed_witnesses(
        index,
        _question(
            "What was the significant buisiness milestone I mentioned four weeks ago?"
        ),
    )

    assert result.receipt.query_action_concepts == ("business_milestone",)
    assert result.receipt.candidate_strategy == "relative_business_milestone_frontier"
    assert result.receipt.candidate_population_count == 2
    assert [row.quote for row in result.witnesses] == [
        "I signed a contract with my first client today.",
        "I launched my website and drafted a business plan.",
    ]
    assert all(row.span.role == "user" for row in result.witnesses)


def test_same_resident_postings_support_global_and_eligible_source_scope(
    tmp_path: Path,
) -> None:
    index = _build_index(
        tmp_path / "source-scope.db",
        [
            ("museum::a", "I visited the Prado with my friend.", ASKED - timedelta(days=4)),
            ("museum::b", "I visited the Tate with my friend.", ASKED - timedelta(days=2)),
        ],
    )
    question = _question("Which museums did I visit with my friend?")

    global_result = query_hot_typed_witnesses(index, question)
    local_result = query_hot_typed_witnesses(
        index, question, eligible_source_ids={"museum::a"}
    )

    assert {row.source_id for row in global_result.witnesses} == {
        "museum::a",
        "museum::b",
    }
    assert {row.source_id for row in local_result.witnesses} == {"museum::a"}
    assert local_result.receipt.scope_mode == "eligible_sources"
    assert local_result.receipt.eligible_source_count == 1
    assert local_result.receipt.matched_eligible_source_count == 1
    assert local_result.receipt.index_receipt_sha256 == global_result.receipt.index_receipt_sha256


def test_eligible_source_visit_bridge_returns_exact_origin_chunk(
    tmp_path: Path,
) -> None:
    index = _build_index(
        tmp_path / "source-visit.db",
        [
            (
                "gallery::story",
                "I visited The Art Cube on 2/15.",
                datetime(2026, 2, 15, tzinfo=timezone.utc),
            ),
            (
                "gallery::story",
                "The opening reception had several abstract paintings.",
                datetime(2026, 2, 16, tzinfo=timezone.utc),
            ),
            (
                "gallery::other",
                "I visited another museum in February.",
                datetime(2026, 2, 20, tzinfo=timezone.utc),
            ),
        ],
    )
    question = _question("Which gallery did I visit in February?")

    result = query_hot_typed_witnesses(
        index,
        question,
        eligible_source_ids={"gallery::story"},
    )

    assert hot_typed_witness_applicable(question) is True
    assert result.origin_chunk_ids == ("chunk-0",)
    assert result.witnesses[0].quote == "I visited The Art Cube on 2/15."
    assert result.witnesses[0].source_id == "gallery::story"


def test_protected_dedup_occurs_after_budget_selection_and_does_not_refill(
    tmp_path: Path,
) -> None:
    index = _build_index(
        tmp_path / "dedup.db",
        [
            ("buy::one", "I bought a smoker.", ASKED - timedelta(days=10)),
            ("buy::two", "I purchased a grill.", ASKED - timedelta(days=10)),
        ],
    )
    question = _question("What did I acquire ten days ago?")
    budget = HotTypedWitnessBudget(max_candidates=1, max_candidates_per_source=1)
    control = query_hot_typed_witnesses(index, question, budget=budget)
    protected_chunk = control.selected_before_dedup[0].span.chunk_id

    result = query_hot_typed_witnesses(
        index,
        question,
        protected_chunk_ids=(protected_chunk,),
        budget=budget,
    )

    assert result.receipt.candidate_population_count == 2
    assert result.receipt.selected_before_dedup_ids == control.receipt.selected_before_dedup_ids
    assert result.receipt.dedup_excluded_ids == result.receipt.selected_before_dedup_ids
    assert result.witnesses == ()
    assert result.receipt.refill_after_protected_dedup is False
    assert result.receipt.selected_before_dedup_tokens <= 1_200
    projection = json.dumps(result.audit_projection(), sort_keys=True)
    assert '"new_provider_calls": 0' in projection
    assert '"model_calls": 0' in projection
    assert '"retained_transformer_token_state_bytes": 0' in projection


def test_scope_and_protected_inputs_fail_closed_on_ambiguous_text(
    tmp_path: Path,
) -> None:
    index = _build_index(
        tmp_path / "invalid.db",
        [("travel::one", "I visited Rome.", ASKED)],
    )

    with pytest.raises(HotTypedWitnessError, match="collection, not text"):
        query_hot_typed_witnesses(
            index,
            _question("Where did I visit?"),
            eligible_source_ids="travel::one",
        )
    with pytest.raises(HotTypedWitnessError, match="ordered and unique"):
        query_hot_typed_witnesses(
            index,
            _question("Where did I visit?"),
            protected_chunk_ids=("chunk-0", "chunk-0"),
        )
