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
from tools.matched_eval.contracts import canonical_json_bytes, identity_sha256
from tools.matched_eval.full_store_slot_closure import build_full_store_window_index
from tools.matched_eval.query_expansion import FrozenSourceNamespace
from tools.matched_eval.query_guided_scan import cache_namespace_partitions
from tools.matched_eval.semantic_global_completion import (
    SemanticGlobalCompletionError,
    SemanticGlobalCompletionPolicy,
    compile_semantic_global_completion_request,
    replay_semantic_global_completion,
    search_semantic_global_completion,
    validate_semantic_global_completion,
    validate_semantic_global_completion_projection,
)
from tools.matched_eval.semantic_residual_search import (
    SemanticResidualPolicy,
    build_semantic_residual_index,
    compile_semantic_residual_query,
)


BASE = datetime(2026, 3, 1, tzinfo=timezone.utc)


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _build(
    tmp_path: Path,
    name: str,
    rows: list[tuple[str, str, datetime, str]],
    vectors: dict[str, list[float] | None],
    *,
    max_cell_tokens: int = 160,
):
    path = tmp_path / f"{name}.db"
    database = Database(path)
    transcript = TranscriptStore(database)
    lexical = LexicalIndex(database)
    for ordinal, (source_id, text, created_at, role) in enumerate(rows):
        turn = transcript.append(
            role,
            text,
            source_id=source_id,
            created_at=created_at,
        )
        lexical.add_chunks(
            [
                Chunk(
                    chunk_id=f"chunk-{ordinal}",
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
    store_receipt = _sha(f"store:{name}")
    namespace = FrozenSourceNamespace.from_source_streams(
        snapshot_id=_sha(f"snapshot:{name}"),
        combined_store_receipt_sha256=store_receipt,
        source_streams=streams,
    )
    with Database(path, read_only=True) as readonly:
        cache = cache_namespace_partitions(
            readonly,
            namespace,
            source_database_sha256=_sha(f"database:{name}"),
            source_store_receipt_sha256=store_receipt,
        )
    window = build_full_store_window_index(cache)
    return build_semantic_residual_index(
        window,
        vectors,
        policy=SemanticResidualPolicy(
            max_cell_tokens=max_cell_tokens,
            payload_token_cap=2_000,
            dual_gate_enabled=False,
        ),
    )


def _query(
    index,
    body: str,
    vector: list[float] | None = None,
    *,
    asked_at: str = "2026/03/25 18:26",
):
    dated = f"[Question asked at {asked_at}] {body}"
    if vector is None:
        return compile_semantic_residual_query(index, dated)
    # Every exact question-only facet is embedded independently in production;
    # a repeated synthetic vector is sufficient for this local mechanism test.
    from tools.matched_eval.semantic_residual_search import semantic_residual_query_facets

    facets = semantic_residual_query_facets(dated)
    return compile_semantic_residual_query(
        index,
        dated,
        query_vectors=[vector for _ in facets],
    )


def _request(query, **kwargs):
    return compile_semantic_global_completion_request(
        query,
        prior_needs_global_search=True,
        **kwargs,
    )


def _search(index, query, *, policy=None, protected=()):
    return search_semantic_global_completion(
        index,
        query,
        _request(query),
        policy=policy or SemanticGlobalCompletionPolicy(),
        protected_evidence=protected,
    )


def test_dense_bridge_recovers_appliance_smoker_and_exact_relative_date(
    tmp_path: Path,
) -> None:
    smoker = "I bought a Traeger smoker on March 15, 2026."
    index = _build(
        tmp_path,
        "smoker-bridge",
        [
            ("pantry", "I reorganized the pantry shelves.", BASE, "user"),
            ("smoker", smoker, BASE + timedelta(days=14), "user"),
            ("music", "You could try a new jazz playlist.", BASE, "assistant"),
        ],
        {"pantry": [0.0, 1.0], "smoker": [1.0, 0.0], "music": [-1.0, 0.0]},
    )
    query = _query(
        index,
        "What kitchen appliance did I buy 10 days ago?",
        [1.0, 0.0],
    )

    result = _search(index, query)

    assert any(row.quote == smoker for row in result.evidence)
    smoker_row = next(row for row in result.evidence if row.quote == smoker)
    date_obligations = [row for row in result.obligations if row.kind == "date"]
    assert len(date_obligations) == 1
    assert date_obligations[0].obligation_id in smoker_row.supported_obligation_ids
    assert result.provider_payload_tokens <= result.policy.global_payload_token_cap
    assert result.projection()["new_provider_calls"] == 0
    assert result.projection()["retained_transformer_token_state_bytes"] == 0


def test_production_weekday_date_and_completed_event_witness_recover_real_smoker_shape(
    tmp_path: Path,
) -> None:
    target_date = datetime(2023, 3, 15, 5, 6, tzinfo=timezone.utc)
    smoker = (
        "I'm looking for some new BBQ sauce recipes to try out. "
        "By the way, I just got a smoker today and I'm excited to experiment."
    )
    distractor_date = datetime(2023, 3, 24, 12, 0, tzinfo=timezone.utc)
    distractors = [
        (
            f"distractor-{ordinal:03d}",
            f"Kitchen appliance candidate {ordinal}: I might buy one later.",
            distractor_date,
            "user",
        )
        for ordinal in range(210)
    ]
    rows = [*distractors, ("smoker", smoker, target_date, "user")]
    vectors = {source_id: [1.0, 0.0] for source_id, *_rest in distractors}
    vectors["smoker"] = [-1.0, 0.0]
    index = _build(tmp_path, "production-weekday-smoker", rows, vectors)
    query = _query(
        index,
        "What kitchen appliance did I buy 10 days ago?",
        [1.0, 0.0],
        asked_at="2023/03/25 (Sat) 18:04",
    )
    result = _search(
        index,
        query,
        policy=SemanticGlobalCompletionPolicy(
            max_node_visits=64,
            max_retained_leaf_cells=1,
            max_hydrated_segments=32,
        ),
    )

    date_obligation = next(row for row in result.obligations if row.kind == "date")
    assert (date_obligation.target_date_start, date_obligation.target_date_end) == (
        "2023-03-15",
        "2023-03-15",
    )
    smoker_candidate = next(row for row in result.candidates if row.quote == smoker)
    assert dict(smoker_candidate.lane_scores)["personal_temporal"] >= 48.0
    assert smoker_candidate.cell_id in result.tree_frontier.retained_leaf_cell_ids
    assert any(row.quote == smoker for row in result.evidence)
    assert result.tree_frontier.search_closed is False
    assert result.tree_frontier.unresolved_leaf_cell_ids


def test_brought_home_completed_acquisition_prioritizes_low_dense_recent_plant(
    tmp_path: Path,
) -> None:
    target_date = datetime(2023, 5, 24, 12, 30, tzinfo=timezone.utc)
    peace_lily = (
        "I'm having some issues with my peace lily; it has been losing leaves "
        "since I brought it home."
    )
    distractors = [
        (
            f"plant-distractor-{ordinal:03d}",
            f"Plant candidate {ordinal}: I might acquire one later.",
            datetime(2023, 5, 30, tzinfo=timezone.utc),
            "user",
        )
        for ordinal in range(210)
    ]
    rows = [*distractors, ("peace-lily", peace_lily, target_date, "user")]
    vectors = {source_id: [1.0, 0.0] for source_id, *_rest in distractors}
    vectors["peace-lily"] = [-1.0, 0.0]
    index = _build(tmp_path, "brought-home-plant", rows, vectors)
    query = _query(
        index,
        "Which plant did I acquire last month?",
        [1.0, 0.0],
        asked_at="2023/06/02 (Fri) 09:00",
    )
    result = _search(
        index,
        query,
        policy=SemanticGlobalCompletionPolicy(
            max_node_visits=64,
            max_retained_leaf_cells=1,
            max_hydrated_segments=32,
        ),
    )

    plant_candidate = next(row for row in result.candidates if row.quote == peace_lily)
    assert dict(plant_candidate.lane_scores)["personal_temporal"] >= 48.0
    assert plant_candidate.cell_id in result.tree_frontier.retained_leaf_cell_ids
    assert any(row.quote == peace_lily for row in result.evidence)
    assert result.tree_frontier.definitely_no_leaf_cell_ids == ()
    assert result.tree_frontier.unresolved_leaf_cell_ids


def test_planned_service_witness_requires_question_include_proposed(
    tmp_path: Path,
) -> None:
    planned = (
        "I'm looking into getting a new tire for my commuter bike. "
        "I think it is time to replace it this month, before April comes."
    )
    index = _build(
        tmp_path,
        "planned-replacement",
        [("commuter", planned, datetime(2023, 3, 20, tzinfo=timezone.utc), "user")],
        {"commuter": [1.0, 0.0]},
    )
    proposed_query = _query(
        index,
        "Which bike did I plan to service in March?",
        [1.0, 0.0],
        asked_at="2023/03/25 (Sat) 18:04",
    )
    completed_only_query = _query(
        index,
        "Which bike did I service in March?",
        [1.0, 0.0],
        asked_at="2023/03/25 (Sat) 18:04",
    )

    proposed = _search(index, proposed_query)
    completed_only = _search(index, completed_only_query)
    proposed_candidate = next(row for row in proposed.candidates if row.quote == planned)
    completed_only_candidate = next(
        row for row in completed_only.candidates if row.quote == planned
    )

    assert proposed_query.operator_spec.include_proposed is True
    assert completed_only_query.operator_spec.include_proposed is False
    assert dict(proposed_candidate.lane_scores)["personal_temporal"] >= 48.0
    assert dict(completed_only_candidate.lane_scores)["personal_temporal"] < 48.0
    assert any(row.quote == planned for row in proposed.evidence)


def test_inline_month_day_and_took_person_to_venue_prioritize_completed_visit(
    tmp_path: Path,
) -> None:
    museum = (
        "I'm looking for some art supply stores in the city. By the way, "
        "I took my niece to the Natural History Museum on 2/8 and she loved "
        "the dinosaur exhibit!"
    )
    distractors = [
        (
            f"museum-distractor-{ordinal:03d}",
            f"Museum candidate {ordinal}: I might visit one later.",
            datetime(2023, 2, 15, tzinfo=timezone.utc),
            "user",
        )
        for ordinal in range(210)
    ]
    rows = [
        *distractors,
        ("natural-history", museum, datetime(2023, 3, 3, 15, 10, tzinfo=timezone.utc), "user"),
    ]
    vectors = {source_id: [1.0, 0.0] for source_id, *_rest in distractors}
    vectors["natural-history"] = [-1.0, 0.0]
    index = _build(tmp_path, "inline-month-day-museum", rows, vectors)
    query = _query(
        index,
        "How many museums did I visit in February?",
        [1.0, 0.0],
        asked_at="2023/03/25 (Sat) 18:04",
    )
    result = _search(
        index,
        query,
        policy=SemanticGlobalCompletionPolicy(
            max_node_visits=64,
            max_retained_leaf_cells=1,
            max_hydrated_segments=32,
        ),
    )

    museum_candidate = next(row for row in result.candidates if row.quote == museum)
    assert museum_candidate.event_dates == ("2023-02-08",)
    assert dict(museum_candidate.lane_scores)["personal_temporal"] >= 48.0
    assert museum_candidate.cell_id in result.tree_frontier.retained_leaf_cell_ids
    assert any(row.quote == museum for row in result.evidence)
    assert result.tree_frontier.definitely_no_leaf_cell_ids == ()
    assert result.tree_frontier.unresolved_leaf_cell_ids


def test_sparse_action_and_date_lanes_keep_low_rank_second_count_operand(
    tmp_path: Path,
) -> None:
    serviced = "I serviced my road bike on March 2, 2026."
    planned = "I planned to service my commuter bike on March 18, 2026."
    rows = [
        (f"distractor-{i:02d}", f"I organized garden box {i}.", BASE, "user")
        for i in range(24)
    ] + [
        ("road", serviced, BASE + timedelta(days=1), "user"),
        ("commuter", planned, BASE + timedelta(days=17), "user"),
    ]
    vectors = {source_id: [0.0, 1.0] for source_id, *_rest in rows}
    vectors["road"] = [0.6, 0.8]
    vectors["commuter"] = [0.2, 0.98]
    index = _build(tmp_path, "second-operand", rows, vectors)
    query = _query(
        index,
        "How many bikes did I service or plan to service in March?",
        [1.0, 0.0],
    )

    result = _search(index, query)

    quotes = {row.quote for row in result.evidence}
    assert {serviced, planned} <= quotes
    sources = {
        binding.source_id
        for evidence, binding in zip(result.evidence, result.local_bindings, strict=True)
        if evidence.quote in {serviced, planned}
    }
    assert sources == {"road", "commuter"}
    sparse = next(row for row in result.lane_receipts if row.lane_id == "sparse")
    assert len(sparse.selected_segment_receipt_sha256s) >= 2


def test_same_source_user_fact_outranks_generic_assistant_tail(tmp_path: Path) -> None:
    generic = "You could consider jewelry such as rings, necklaces, or bracelets."
    fact = "I acquired a sapphire ring at the market yesterday."
    index = _build(
        tmp_path,
        "generic-vs-fact",
        [
            ("history", generic, BASE, "assistant"),
            ("history", fact, BASE + timedelta(days=1), "user"),
        ],
        {"history": [1.0, 0.0]},
    )
    result = _search(
        index,
        _query(index, "What jewelry did I acquire?", [1.0, 0.0]),
    )

    assert any(row.quote == fact for row in result.evidence)
    fact_attempt = next(
        row for row in result.attempted_selection if row.candidate_id == next(
            candidate.candidate_id for candidate in result.candidates if candidate.quote == fact
        )
    )
    generic_candidate = next(row for row in result.candidates if row.quote == generic)
    fact_candidate = next(row for row in result.candidates if row.quote == fact)
    assert dict(fact_candidate.lane_scores)["personal_temporal"] > dict(
        generic_candidate.lane_scores
    )["personal_temporal"]
    assert fact_attempt.selection_rank < next(
        (
            row.selection_rank
            for row in result.attempted_selection
            if row.candidate_id == generic_candidate.candidate_id
        ),
        10**6,
    )


def test_source_date_diversity_lane_is_independent_and_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        ("alpha", f"I visited gallery Alpha exhibit {i} in March.", BASE + timedelta(days=i), "user")
        for i in range(4)
    ] + [
        ("beta", "I visited gallery Beta in March.", BASE + timedelta(days=6), "user"),
        ("gamma", "I visited gallery Gamma in February.", BASE - timedelta(days=3), "user"),
    ]
    index = _build(
        tmp_path,
        "diversity",
        rows,
        {"alpha": [1.0, 0.0], "beta": [1.0, 0.0], "gamma": [1.0, 0.0]},
        max_cell_tokens=30,
    )
    query = _query(index, "Which galleries did I visit?", [1.0, 0.0])
    result = _search(index, query)
    diversity = next(
        row for row in result.lane_receipts if row.lane_id == "source_date_diversity"
    )
    candidate_by_receipt = {row.receipt_sha256: row for row in result.candidates}
    selected_sources = [
        candidate_by_receipt[value].source_id
        for value in diversity.selected_candidate_receipt_sha256s
    ]

    assert {"alpha", "beta", "gamma"} <= set(selected_sources)
    assert len(set(selected_sources[:3])) == 3

    search_calls = 0

    def counted_search(*args, **kwargs):
        nonlocal search_calls
        search_calls += 1
        return search_semantic_global_completion(*args, **kwargs)

    monkeypatch.setattr(
        "tools.matched_eval.semantic_global_completion.search_semantic_global_completion",
        counted_search,
    )
    replayed = replay_semantic_global_completion(
        index,
        query,
        _request(query),
        result,
    )
    assert search_calls == 1
    assert replayed is not result
    assert canonical_json_bytes(replayed.projection()) == canonical_json_bytes(
        result.projection()
    )


def test_low_scores_never_prune_and_unexpanded_cells_remain_unresolved(
    tmp_path: Path,
) -> None:
    rows = [
        (f"source-{i}", f"Memory branch {i} contains unrelated notes.", BASE, "user")
        for i in range(12)
    ]
    index = _build(
        tmp_path,
        "unresolved",
        rows,
        {source_id: [-1.0, 0.0] for source_id, *_rest in rows},
    )
    query = _query(index, "What was the heliotrope access code?", [1.0, 0.0])
    policy = SemanticGlobalCompletionPolicy(max_node_visits=1)
    result = _search(index, query, policy=policy)

    assert result.tree_frontier.definitely_no_leaf_cell_ids == ()
    assert set(result.tree_frontier.unresolved_leaf_cell_ids) == {
        row.cell_id for row in index.cells
    }
    assert result.tree_frontier.unexpanded_node_receipt_sha256s
    assert result.tree_frontier.search_closed is False
    assert result.closure.needs_further_global_search is True
    assert all(row.action != "definitely_no" for row in result.tree_frontier.visits)


def test_multiple_exact_literals_in_separate_branches_are_not_conjunctively_pruned(
    tmp_path: Path,
) -> None:
    alpha = "The exact term \"alpha\" appeared in my first note."
    beta = "The exact term \"beta\" appeared in my second note."
    unrelated = "This branch contains neither requested code word."
    index = _build(
        tmp_path,
        "split-literals",
        [
            ("alpha-source", alpha, BASE, "user"),
            ("beta-source", beta, BASE, "user"),
            ("other-source", unrelated, BASE, "user"),
        ],
        {
            "alpha-source": [1.0, 0.0],
            "beta-source": [1.0, 0.0],
            "other-source": [-1.0, 0.0],
        },
    )
    query = _query(
        index,
        'Which notes used exact term "alpha" and exact term "beta"?',
        [1.0, 0.0],
    )
    result = _search(index, query)

    retained_quotes = {row.quote for row in result.evidence}
    assert {alpha, beta} <= retained_quotes
    pruned_sources = {
        index.cell_by_id[cell_id].source_id
        for cell_id in result.tree_frontier.definitely_no_leaf_cell_ids
    }
    assert "alpha-source" not in pruned_sources
    assert "beta-source" not in pruned_sources


def test_all_lanes_select_before_exact_owner_dedup_and_reinject_owner(
    tmp_path: Path,
) -> None:
    owned = "I bought the smoker on March 15, 2026."
    other = "I serviced the grill on March 16, 2026."
    index = _build(
        tmp_path,
        "owner-dedup",
        [
            ("owned", owned, BASE + timedelta(days=14), "user"),
            ("other", other, BASE + timedelta(days=15), "user"),
        ],
        {"owned": [1.0, 0.0], "other": [0.8, 0.2]},
    )
    query = _query(index, "What did I buy or service in March?", [1.0, 0.0])
    first = _search(index, query)
    owned_binding = next(
        binding
        for evidence, binding in zip(first.evidence, first.local_bindings, strict=True)
        if evidence.quote == owned
    )
    deduped = _search(index, query, protected=(owned_binding,))

    duplicate = next(row for row in deduped.protected_duplicates if row.protected_candidate_id == owned_binding.candidate_id)
    attempt = next(
        row
        for row in deduped.attempted_selection
        if row.segment_receipt_sha256 == duplicate.segment_receipt_sha256
    )
    assert duplicate.protected_binding_receipt_sha256 == owned_binding.receipt_sha256
    assert attempt.disposition == "protected_exact_duplicate"
    assert attempt.selected_lane_ids
    assert all(row.quote != owned for row in deduped.evidence)
    assert deduped.projection()["selected_before_em_and_protected_dedup"] is True


def test_skip_continue_packing_keeps_short_row_after_oversized_witness(
    tmp_path: Path,
) -> None:
    long = "I bought the heliotrope appliance. " + " ".join(
        f"detail-{i:04d}" for i in range(700)
    )
    short = "I bought the backup appliance, a compact smoker."
    index = _build(
        tmp_path,
        "skip-continue",
        [
            ("long", long, BASE, "user"),
            ("short", short, BASE, "user"),
        ],
        {"long": [1.0, 0.0], "short": [0.9, 0.1]},
        max_cell_tokens=2_048,
    )
    query = _query(index, "What heliotrope appliance did I buy?", [1.0, 0.0])
    policy = SemanticGlobalCompletionPolicy(global_payload_token_cap=1_600)
    result = _search(index, query, policy=policy)

    assert any(row.quote == short for row in result.evidence)
    long_segments = {
        row.segment_receipt_sha256
        for row in result.candidates
        if row.source_id == "long"
    }
    assert long_segments & set(
        result.closure.budget_unpacked_segment_receipt_sha256s
    )
    assert result.provider_payload_tokens <= 1_600


def test_multicause_synthesis_admits_independent_user_events(tmp_path: Path) -> None:
    drivetrain = "My Sunday group rides improved after I replaced the chain and cassette."
    garmin = "My Sunday group rides also improved after I added a Garmin bike computer."
    index = _build(
        tmp_path,
        "multi-cause",
        [
            ("drivetrain", drivetrain, BASE, "user"),
            ("computer", garmin, BASE + timedelta(days=2), "user"),
            ("generic", "You could improve group rides with regular practice.", BASE, "assistant"),
        ],
        {
            "drivetrain": [1.0, 0.0],
            "computer": [0.95, 0.05],
            "generic": [0.8, 0.2],
        },
    )
    result = _search(
        index,
        _query(index, "Why did my Sunday group rides improve?", [1.0, 0.0]),
    )

    assert {drivetrain, garmin} <= {row.quote for row in result.evidence}
    assert len(
        {
            binding.source_id
            for evidence, binding in zip(result.evidence, result.local_bindings, strict=True)
            if evidence.quote in {drivetrain, garmin}
        }
    ) == 2


def test_projection_mutation_fails_and_byte_exact_replay_passes(tmp_path: Path) -> None:
    fact = "I bought a smoker on March 15, 2026."
    index = _build(
        tmp_path,
        "mutation",
        [("fact", fact, BASE + timedelta(days=14), "user")],
        {"fact": [1.0, 0.0]},
    )
    query = _query(index, "What appliance did I buy 10 days ago?", [1.0, 0.0])
    request = _request(query)
    result = search_semantic_global_completion(index, query, request)

    validate_semantic_global_completion(index, query, request, result)
    projection = json.loads(json.dumps(result.projection()))
    loaded = validate_semantic_global_completion_projection(
        index,
        query,
        request,
        projection,
    )
    assert loaded.receipt_sha256 == result.receipt_sha256
    projection["evidence"][0]["quote"] = "tampered"
    with pytest.raises(
        SemanticGlobalCompletionError,
        match="stored global completion projection differs",
    ):
        validate_semantic_global_completion_projection(
            index,
            query,
            request,
            projection,
        )


def test_request_is_generic_and_rejects_unknown_slot() -> None:
    # The public compiler is exercised with a real query in all other tests;
    # inspect its schema there instead of inventing a benchmark identifier.
    assert "ordinal" not in json.dumps(
        compile_semantic_global_completion_request.__annotations__, default=str
    ).casefold()
