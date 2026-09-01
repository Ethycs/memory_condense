from __future__ import annotations

import inspect
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from memory_condense.application.discourse_sources import scan_discourse_source_chunks
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.schemas import Chunk
from memory_condense.persistence.db import Database
from memory_condense.persistence.discourse_store import DiscourseStore
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.search.indexes.lexical import LexicalIndex

from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.full_store_slot_closure import (
    ExactTermAbsenceStatus,
    FullStoreSlotClosureBudget,
    TemporalTargetMode,
    adapt_full_store_slot_closure_to_typed_contribution,
    build_full_store_window_index,
    scan_full_store_slot_closure,
    scan_full_store_slot_closures,
)
from tools.matched_eval.query_expansion import FrozenSourceNamespace
from tools.matched_eval.query_guided_scan import (
    cache_namespace_partitions,
    score_query_guided_candidates,
)
from tools.matched_eval.typed_operator_spec import SlotKind
from tools.matched_eval.typed_operator_adapter import (
    EvidenceOrigin,
    FrontierMode,
    ProvenanceGrade,
    merge_typed_evidence_contributions,
)


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _write_cache(
    path: Path,
    rows: list[tuple[str, str, datetime]],
):
    database = Database(path)
    transcript = TranscriptStore(database)
    lexical = LexicalIndex(database)
    for index, (source_id, text, created_at) in enumerate(rows):
        turn = transcript.append(
            "user", text, source_id=source_id, created_at=created_at
        )
        lexical.add_chunks(
            [
                Chunk(
                    chunk_id=f"chunk-{index}",
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
    store_receipt = _sha("combined-store")
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
    return cache


def test_all_partition_scan_recalls_fact_outside_top_six_without_prefix_route(
    tmp_path: Path,
) -> None:
    base = datetime(2026, 8, 1, tzinfo=timezone.utc)
    rows = [
        (f"p{index}::noise", f"Ordinary filler memory number {index}.", base)
        for index in range(7)
    ]
    rows.append(
        (
            "p7::remote",
            "The obsidian beacon verification code was QZ-741.",
            base + timedelta(days=1),
        )
    )
    path = tmp_path / "cross-prefix.db"
    cache = _write_cache(path, rows)
    question = (
        "[Question asked at 2026/08/27 12:00] "
        "What was the verification code on the obsidian beacon?"
    )

    top_six = score_query_guided_candidates(
        cache,
        selected_partitions=tuple(cache.rows_by_partition)[:6],
        query_surfaces=(question,),
    )
    result = scan_full_store_slot_closure(cache, question)

    assert all(row.partition_id != "p7" for row in top_six)
    target_index = next(
        index for index, row in enumerate(result.candidates) if "QZ-741" in row.quote
    )
    binding = result.local_bindings[target_index]
    assert binding.partition_id == "p7"
    assert binding.source_id == "p7::remote"
    assert result.receipt.physical_content_rows_scanned == len(rows)
    assert result.receipt.physical_partition_count == 8
    assert result.receipt.physical_scan_exhaustive is True
    assert result.receipt.semantic_completeness_status == "not_claimed"
    assert result.receipt.question_id_filter_used is False
    assert result.receipt.known_source_prefix_filter_used is False
    assert result.receipt.partition_routing_used is False

    provider_json = json.dumps(result.provider_projection(), sort_keys=True)
    assert '"partition_id"' not in provider_json
    assert '"source_id"' not in provider_json
    assert "p7::remote" not in provider_json
    with Database(path, read_only=True) as database:
        assert DiscourseStore(database).hydrate_span(binding.span) == (
            result.candidates[target_index].quote
        )


def test_two_numeric_operands_receive_independent_exact_citations(
    tmp_path: Path,
) -> None:
    base = datetime(2026, 8, 10, tzinfo=timezone.utc)
    cache = _write_cache(
        tmp_path / "operands.db",
        [
            ("garden-a::one", "I initially planted 6 tomato plants.", base),
            (
                "garden-z::two",
                "I initially planted 4 chili pepper plants.",
                base + timedelta(days=2),
            ),
            ("noise::three", "The ceramic vase was moved upstairs.", base),
        ],
    )
    question = (
        "[Question asked at 2026/08/27 12:00] "
        "How many plants did I initially plant for tomatoes and chili peppers?"
    )

    result = scan_full_store_slot_closure(cache, question)

    operand_slots = tuple(
        slot.slot_id
        for slot in result.operator_spec.required_slots
        if slot.kind is SlotKind.OPERAND
    )
    supported = {
        slot_id for row in result.candidates for slot_id in row.supported_slot_ids
    }
    assert len(operand_slots) == 2
    assert set(operand_slots) <= supported
    assert any("6 tomato" in row.quote for row in result.candidates)
    assert any("4 chili" in row.quote for row in result.candidates)
    assert {
        binding.partition_id
        for row, binding in zip(
            result.candidates, result.local_bindings, strict=True
        )
        if set(row.supported_slot_ids) & set(operand_slots)
    } == {"garden-a", "garden-z"}
    assert result.receipt.selected_evidence_tokens <= 2_400
    assert (
        result.receipt.provider_payload_tokens
        + result.budget.protocol_token_reserve
        + result.budget.output_token_reserve
        <= result.budget.hard_prompt_token_cap
    )


def test_temporal_target_day_admits_semantically_distant_activity(
    tmp_path: Path,
) -> None:
    asked = datetime(2026, 8, 27, 12, tzinfo=timezone.utc)
    cache = _write_cache(
        tmp_path / "temporal.db",
        [
            (
                "old::target",
                "I planted rosemary and mint beside the porch.",
                asked - timedelta(days=14),
            ),
            (
                "recent::noise",
                "I replaced a bicycle cable after work.",
                asked - timedelta(days=2),
            ),
        ],
    )
    question = (
        "[Question asked at 2026/08/27 12:00] "
        "What gardening-related activity did I do two weeks ago?"
    )

    result = scan_full_store_slot_closure(cache, question)

    target = next(row for row in result.candidates if "rosemary" in row.quote)
    assert result.temporal_target.mode is TemporalTargetMode.EXACT_DAY
    assert result.temporal_target.target_date == "2026-08-13"
    assert target.temporal_distance_days == 0
    assert "question_derived_temporal_target" in target.selection_axes


def test_q42_and_q72_insufficiency_shapes_remain_semantically_unresolved(
    tmp_path: Path,
) -> None:
    base = datetime(2026, 6, 1, tzinfo=timezone.utc)
    cache = _write_cache(
        tmp_path / "insufficiency.db",
        [
            (
                "memo-a::poster",
                "I presented a poster for my undergrad project at an annual symposium.",
                base,
            ),
            (
                "memo-b::doctor",
                "I saved Dr. Johnson's office phone number in my contacts.",
                base + timedelta(days=1),
            ),
        ],
    )
    questions = (
        (
            "[Question asked at 2026/08/27 12:00] "
            "At which university did I present a poster for my undergrad project?"
        ),
        (
            "[Question asked at 2026/08/27 12:00] "
            "How often do I see Dr. Johnson?"
        ),
    )

    results = tuple(scan_full_store_slot_closure(cache, row) for row in questions)

    assert any("poster" in row.quote for row in results[0].candidates)
    assert any("Johnson" in row.quote for row in results[1].candidates), (
        [row.quote for row in results[1].candidates],
        results[1].receipt.candidate_population_count,
        results[1].receipt.role_rejected_candidate_count,
        results[1].operator_spec.required_evidence_role,
    )
    for result in results:
        assert result.absence_witness.status is ExactTermAbsenceStatus.UNRESOLVED
        assert result.absence_witness.may_assert_exact_literal_absence is False
        assert result.absence_witness.semantic_absence_may_be_inferred is False
        assert result.receipt.semantic_completeness_status == "not_claimed"
        assert result.receipt.evidence_status != "narrow_exact_literal_absence_only"


def test_absence_is_closed_only_for_explicitly_quoted_exact_term(
    tmp_path: Path,
) -> None:
    base = datetime(2026, 8, 1, tzinfo=timezone.utc)
    cache = _write_cache(
        tmp_path / "literal.db",
        [("p0::one", "I mentioned the saffron arch yesterday.", base)],
    )

    ordinary = scan_full_store_slot_closure(
        cache,
        "[Question asked at 2026/08/27 12:00] Did I mention a cobalt gate?",
    )
    exact = scan_full_store_slot_closure(
        cache,
        (
            "[Question asked at 2026/08/27 12:00] "
            'Did I ever mention the exact phrase "cobalt gate"?'
        ),
    )

    assert ordinary.absence_witness.status is ExactTermAbsenceStatus.UNRESOLVED
    assert ordinary.absence_witness.may_assert_exact_literal_absence is False
    assert exact.absence_witness.status is ExactTermAbsenceStatus.LITERAL_ABSENT
    assert exact.absence_witness.may_assert_exact_literal_absence is True
    assert exact.absence_witness.semantic_absence_may_be_inferred is False
    assert exact.absence_witness.physical_content_rows_scanned == 1


def test_public_scan_api_has_no_identifier_route_or_provider_inputs() -> None:
    signature = inspect.signature(scan_full_store_slot_closure)

    assert tuple(signature.parameters) == ("cache", "dated_question", "budget")
    forbidden = {
        "question_id",
        "source_id",
        "source_prefix",
        "known_source",
        "selected_partitions",
        "reference",
        "prediction",
        "provider",
        "client",
    }
    assert forbidden.isdisjoint(signature.parameters)
    assert signature.parameters["cache"].kind is inspect.Parameter.POSITIONAL_ONLY
    assert (
        signature.parameters["dated_question"].kind
        is inspect.Parameter.POSITIONAL_ONLY
    )
    assert (
        signature.parameters["budget"].default
        == FullStoreSlotClosureBudget()
    )


def test_one_window_index_is_reused_across_prompt_ticks(tmp_path: Path) -> None:
    base = datetime(2026, 8, 1, tzinfo=timezone.utc)
    cache = _write_cache(
        tmp_path / "shared-index.db",
        [
            (
                f"partition-{index}::memory",
                f"The archive contains zephyrtoken{index}.",
                base + timedelta(days=index),
            )
            for index in range(20)
        ],
    )
    questions = tuple(
        f"[Question asked at 2026/08/27 12:00] What was zephyrtoken{index}?"
        for index in range(10)
    )

    index = build_full_store_window_index(cache)
    direct = tuple(
        scan_full_store_slot_closure(index, question) for question in questions
    )
    batch = scan_full_store_slot_closures(cache, questions)

    assert len({row.receipt.window_index_receipt_sha256 for row in direct}) == 1
    assert len({row.receipt.window_index_receipt_sha256 for row in batch}) == 1
    assert all(
        row.receipt.window_index_reuse_mode == "reused_prebuilt_index"
        and row.receipt.query_tick_full_physical_rescan is False
        and row.receipt.physical_content_rows_scanned == 20
        and row.receipt.query_candidate_windows_considered < 20
        for row in (*direct, *batch)
    )
    assert index.projection()["window_index_build_passes"] == 1
    assert index.projection()["all_content_rows_indexed"] is True


def test_content_source_coherence_is_preserved_before_diverse_filler(
    tmp_path: Path,
) -> None:
    base = datetime(2026, 8, 1, tzinfo=timezone.utc)
    cache = _write_cache(
        tmp_path / "coherence.db",
        [
            ("story::shared", "The amber project began in Lisbon.", base),
            (
                "story::shared",
                "The amber project concluded beside the Oslo harbor.",
                base + timedelta(days=2),
            ),
            ("filler-a::one", "An amber bowl sat on a shelf.", base),
            ("filler-b::two", "A project notebook had blank pages.", base),
        ],
    )
    question = (
        "[Question asked at 2026/08/27 12:00] "
        "What connected the amber project in Lisbon and Oslo?"
    )

    result = scan_full_store_slot_closure(cache, question)
    story = [
        row
        for row, binding in zip(
            result.candidates, result.local_bindings, strict=True
        )
        if binding.source_id == "story::shared"
    ]

    assert len(story) == 2
    assert len({row.source_group_handle for row in story}) == 1
    assert all("content_source_coherence" in row.selection_axes for row in story)


def test_content_coherence_budget_retains_four_same_source_events(
    tmp_path: Path,
) -> None:
    base = datetime(2026, 8, 1, tzinfo=timezone.utc)
    cache = _write_cache(
        tmp_path / "four-event-coherence.db",
        [
            ("long-story::events", "The amber launch began at dawn.", base),
            (
                "long-story::events",
                "The amber bridge opened after lunch.",
                base + timedelta(days=1),
            ),
            (
                "long-story::events",
                "The amber gallery welcomed us next.",
                base + timedelta(days=2),
            ),
            (
                "long-story::events",
                "The amber harbor ceremony finished the sequence.",
                base + timedelta(days=3),
            ),
            ("filler::one", "An amber cup was on the counter.", base),
        ],
    )
    question = (
        "[Question asked at 2026/08/27 12:00] "
        "What was the sequence of the amber launch, bridge, gallery, and harbor?"
    )

    result = scan_full_store_slot_closure(cache, question)
    story = [
        row
        for row, binding in zip(
            result.candidates, result.local_bindings, strict=True
        )
        if binding.source_id == "long-story::events"
    ]

    assert result.budget.max_candidates_per_source >= 6
    assert result.budget.source_coherence_candidate_reserve >= 8
    assert len(story) == 4
    assert len({row.source_group_handle for row in story}) == 1
    assert all("content_source_coherence" in row.selection_axes for row in story)


def test_bounded_typed_contribution_keeps_exact_pointer_provenance_opaque(
    tmp_path: Path,
) -> None:
    base = datetime(2026, 8, 10, tzinfo=timezone.utc)
    cache = _write_cache(
        tmp_path / "typed-contribution.db",
        [
            ("secret-a::one", "I initially planted 6 tomato plants.", base),
            (
                "secret-b::two",
                "I initially planted 4 chili pepper plants.",
                base + timedelta(days=2),
            ),
        ],
    )
    question = (
        "[Question asked at 2026/08/27 12:00] "
        "How many plants did I initially plant for tomatoes and chili peppers?"
    )
    result = scan_full_store_slot_closure(cache, question)

    contribution = adapt_full_store_slot_closure_to_typed_contribution(
        result, handle_start=120, group_start=220
    )
    packet = merge_typed_evidence_contributions(
        result.operator_spec, (contribution,)
    )

    assert contribution.frontier_mode is FrontierMode.BOUNDED
    assert packet.frontier.mode is FrontierMode.BOUNDED
    assert packet.frontier.closed is False
    assert result.receipt.physical_scan_exhaustive is True
    assert [row.handle_id for row in contribution.bindings] == [
        f"H{120 + index:03d}" for index in range(len(result.candidates))
    ]
    assert all(
        row.origin is EvidenceOrigin.DIRECT_POINTER
        and row.provenance_grade is ProvenanceGrade.DIRECT_POINTER
        and row.evidence_receipt_sha256 == local.receipt_sha256
        for row, local in zip(
            contribution.bindings, result.local_bindings, strict=True
        )
    )
    assert all(row.source_group_handle >= "G220" for row in contribution.bindings)
    numeric = {
        item.numeric_value
        for item in contribution.parsed.accepted_items
        if item.numeric_value is not None
    }
    assert {4.0, 6.0} <= numeric
    assert all(
        item.date is not None
        and item.status.value == "completed"
        and item.relation == "authored_by_user"
        for item in contribution.parsed.accepted_items
    )
    provider_json = json.dumps(packet.provider_projection(), sort_keys=True)
    assert '"partition_id"' not in provider_json
    assert '"source_id"' not in provider_json
    assert "secret-a::one" not in provider_json
    assert "secret-b::two" not in provider_json


def test_source_story_group_does_not_collapse_two_numeric_comparison_sides(
    tmp_path: Path,
) -> None:
    base = datetime(2026, 8, 1, tzinfo=timezone.utc)
    cache = _write_cache(
        tmp_path / "comparison-groups.db",
        [
            ("same-story::trip", "I spent 500 dollars in Hawaii.", base),
            (
                "same-story::trip",
                "I spent 300 dollars in Tokyo.",
                base + timedelta(days=1),
            ),
        ],
    )
    question = (
        "[Question asked at 2026/08/27 12:00] "
        "how much more did I spend in Hawaii compared to Tokyo?"
    )
    result = scan_full_store_slot_closure(cache, question)

    contribution = adapt_full_store_slot_closure_to_typed_contribution(
        result, handle_start=300, group_start=400
    )
    items = tuple(
        row for row in contribution.parsed.accepted_items if row.numeric_value is not None
    )

    assert len(items) == 2
    assert len({row.source_group_handle for row in contribution.bindings}) == 1
    assert {row.entity_key for row in items} == {"Hawaii", "Tokyo"}
    assert all(row.group_key is None for row in items)


def test_q28_q53_numeric_candidate_flags_reject_date_and_duration(
    tmp_path: Path,
) -> None:
    base = datetime(2023, 3, 1, tzinfo=timezone.utc)
    cache = _write_cache(
        tmp_path / "numeric-hygiene.db",
        [
            (
                "bike-story::one",
                "I serviced my road bike on March 2.",
                base,
            ),
            (
                "plant-story::one",
                "The aquarium plants need a 31-day treatment window.",
                base + timedelta(days=1),
            ),
        ],
    )
    q28 = scan_full_store_slot_closure(
        cache,
        "[Question asked at 2023/03/20 (Mon) 23:57] "
        "How many bikes did I service or plan to service in March?",
    )
    q53 = scan_full_store_slot_closure(
        cache,
        "[Question asked at 2023/05/30 (Tue) 21:51] "
        "How many plants did I acquire in the last month?",
    )
    assert any("March 2" in row.quote for row in q28.candidates)
    assert all(
        not row.contains_numeric_value
        for row in q28.candidates
        if "March 2" in row.quote
    )
    assert any("31-day" in row.quote for row in q53.candidates)
    assert all(
        not row.contains_numeric_value
        for row in q53.candidates
        if "31-day" in row.quote
    )


def test_q75_rank_cannot_cover_price_slot_and_qualifiers_survive_adapter(
    tmp_path: Path,
) -> None:
    base = datetime(2023, 5, 1, tzinfo=timezone.utc)
    cache = _write_cache(
        tmp_path / "qualified-comparison.db",
        [
            ("noise::hawaii", "Top 5 Hawaii experiences.", base),
            (
                "trip::hawaii",
                "I spent over $300 per night in Hawaii.",
                base + timedelta(days=1),
            ),
            (
                "trip::tokyo",
                "I stayed in a hostel in Tokyo that cost around $30 per night.",
                base + timedelta(days=2),
            ),
        ],
    )
    question = (
        "[Question asked at 2023/05/30 (Tue) 22:16] "
        "How much more did I spend on accommodations per night in Hawaii "
        "compared to Tokyo?"
    )
    result = scan_full_store_slot_closure(cache, question)
    ranked = next(row for row in result.candidates if "Top 5" in row.quote)
    assert ranked.contains_numeric_value is False
    assert ranked.supported_slot_ids == ()
    assert not any(
        axis.startswith("required_slot:") for axis in ranked.selection_axes
    )

    contribution = adapt_full_store_slot_closure_to_typed_contribution(
        result,
        handle_start=700,
        group_start=800,
    )
    values = {
        item.numeric_value: (item.numeric_qualifier.value, item.unit)
        for item in contribution.parsed.accepted_items
        if item.numeric_value is not None
    }
    assert values == {
        300.0: ("lower_bound", "$"),
        30.0: ("approximate", "$"),
    }


def test_query_overlap_is_not_promoted_to_q8_style_specificity_certificate(
    tmp_path: Path,
) -> None:
    base = datetime(2026, 8, 1, tzinfo=timezone.utc)
    cache = _write_cache(
        tmp_path / "specificity.db",
        [
            (
                "fruit-story::one",
                "I bought Honeycrisp apples at the market.",
                base,
            )
        ],
    )
    question = (
        "[Question asked at 2026/08/27 12:00] "
        "What specific type of apples did I buy?"
    )
    result = scan_full_store_slot_closure(cache, question)

    contribution = adapt_full_store_slot_closure_to_typed_contribution(
        result, handle_start=500, group_start=600
    )

    assert result.operator_spec.specificity_required is True
    assert any(row.matched_query_terms for row in result.candidates)
    assert contribution.parsed.accepted_items
    assert all(
        row.specificity_terms == ()
        for row in contribution.parsed.accepted_items
    )


def test_earlier_relative_offsets_compile_exact_target_days_for_tail_shapes(
    tmp_path: Path,
) -> None:
    asked = datetime(2026, 8, 27, 12, tzinfo=timezone.utc)
    cases = (
        ("two weeks earlier", 14, "violet-marker"),
        ("ten days earlier", 10, "cobalt-marker"),
        ("four weeks earlier", 28, "saffron-marker"),
    )
    cache = _write_cache(
        tmp_path / "earlier-offsets.db",
        [
            (
                f"offset-{days}::memory",
                f"I recorded {marker} during the activity.",
                asked - timedelta(days=days),
            )
            for _phrase, days, marker in cases
        ],
    )
    index = build_full_store_window_index(cache)

    for phrase, days, marker in cases:
        result = scan_full_store_slot_closure(
            index,
            (
                "[Question asked at 2026/08/27 12:00] "
                f"What activity did I record {phrase}?"
            ),
        )
        target = next(row for row in result.candidates if marker in row.quote)
        assert result.temporal_target.mode is TemporalTargetMode.EXACT_DAY
        assert result.temporal_target.target_date == (
            asked.date() - timedelta(days=days)
        ).isoformat()
        assert target.temporal_distance_days == 0
        assert "question_derived_temporal_target" in target.selection_axes
