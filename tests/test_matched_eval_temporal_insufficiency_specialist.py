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
from tools.matched_eval.full_store_slot_closure import build_full_store_window_index
from tools.matched_eval.query_expansion import FrozenSourceNamespace
from tools.matched_eval.query_guided_scan import cache_namespace_partitions
from tools.matched_eval.temporal_insufficiency_specialist import (
    BundleRole,
    SpecialistRoute,
    TemporalInsufficiencyBudget,
    adapt_temporal_insufficiency_to_typed_contribution,
    scan_temporal_insufficiency_specialist,
)
from tools.matched_eval.typed_operator_adapter import (
    EvidenceOrigin,
    EvidenceStatus,
    FrontierMode,
    ProvenanceGrade,
)


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _write_index(
    path: Path,
    rows: list[tuple[str, str, datetime]],
):
    database = Database(path)
    transcript = TranscriptStore(database)
    lexical = LexicalIndex(database)
    for offset, (source_id, text, created_at) in enumerate(rows):
        turn = transcript.append(
            "user", text, source_id=source_id, created_at=created_at
        )
        lexical.add_chunks(
            [
                Chunk(
                    chunk_id=f"chunk-{offset}",
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
    return build_full_store_window_index(cache)


def _sources(result) -> set[str]:
    return {row.source_id for row in result.local_bindings}


def test_public_api_has_no_identifier_gold_or_provider_route() -> None:
    signature = inspect.signature(scan_temporal_insufficiency_specialist)

    assert tuple(signature.parameters) == ("index", "dated_question", "budget")
    assert signature.parameters["index"].kind is inspect.Parameter.POSITIONAL_ONLY
    assert (
        signature.parameters["dated_question"].kind
        is inspect.Parameter.POSITIONAL_ONLY
    )
    assert signature.parameters["budget"].default == TemporalInsufficiencyBudget()
    forbidden = {
        "question_id",
        "source_id",
        "source_prefix",
        "partition_id",
        "reference",
        "prediction",
        "provider",
        "client",
    }
    assert forbidden.isdisjoint(signature.parameters)


def test_relative_day_reserves_exact_winner_and_domain_predecessor(
    tmp_path: Path,
) -> None:
    index = _write_index(
        tmp_path / "relative.db",
        [
            (
                "garden-old::session",
                "I attended a gardening workshop where I learned crop rotation.",
                datetime(2023, 4, 15, tzinfo=timezone.utc),
            ),
            (
                "garden-target::session",
                "I just planted 12 new tomato saplings today and I am excited.",
                datetime(2023, 4, 21, tzinfo=timezone.utc),
            ),
            (
                "garden-target::session",
                (
                    "I've been using a gardening app to track weather and soil "
                    "moisture, which helped me plan gardening activities."
                ),
                datetime(2023, 4, 21, tzinfo=timezone.utc),
            ),
            (
                "noise::session",
                "I replaced my bicycle cable today.",
                datetime(2023, 4, 21, tzinfo=timezone.utc),
            ),
        ],
    )
    question = (
        "[Question asked at 2023/05/05 (Fri) 16:42]\n"
        "What gardening-related activity did I do two weeks ago?"
    )

    result = scan_temporal_insufficiency_specialist(index, question)

    assert result.routes == (SpecialistRoute.TEMPORAL_RELATIVE,)
    assert _sources(result) == {"garden-old::session", "garden-target::session"}
    assert result.temporal_bundle is not None
    winner = next(
        row
        for row in result.candidates
        if row.candidate_id == result.temporal_bundle.winner_candidate_id
    )
    predecessor = next(
        row
        for row in result.candidates
        if row.candidate_id == result.temporal_bundle.predecessor_candidate_id
    )
    assert winner.event_date == "2023-04-21"
    assert "12 new tomato" in winner.quote
    assert winner.bundle_role is BundleRole.WINNER
    assert predecessor.event_date == "2023-04-15"
    assert predecessor.bundle_role is BundleRole.PREDECESSOR


def test_elapsed_route_applies_singular_participant_before_recency(
    tmp_path: Path,
) -> None:
    index = _write_index(
        tmp_path / "elapsed.db",
        [
            (
                "science::session",
                (
                    "I went on a behind-the-scenes tour of the Science Museum "
                    "with a friend who is a chemistry professor."
                ),
                datetime(2022, 10, 22, tzinfo=timezone.utc),
            ),
            (
                "lecture::session",
                "I attended a lecture at the History Museum.",
                datetime(2023, 1, 11, tzinfo=timezone.utc),
            ),
            (
                "dad::session",
                "I took a guided tour at the Natural History Museum with my dad.",
                datetime(2023, 2, 18, tzinfo=timezone.utc),
            ),
            (
                "plural::session",
                "I just visited the local art museum with friends.",
                datetime(2023, 3, 16, tzinfo=timezone.utc),
            ),
        ],
    )
    question = (
        "[Question asked at 2023/03/25 (Sat) 17:18]\n"
        "How many months have passed since I last visited a museum with a friend?"
    )

    result = scan_temporal_insufficiency_specialist(index, question)

    assert result.routes == (SpecialistRoute.TEMPORAL_INTERVAL,)
    assert result.temporal_bundle is not None
    winner = next(
        row
        for row in result.candidates
        if row.candidate_id == result.temporal_bundle.winner_candidate_id
    )
    assert winner.event_date == "2022-10-22"
    assert "with a friend" in winner.quote
    assert "singular_participant_relation_support" in winner.selection_axes
    assert "implicit_query_time_end_anchor" in winner.selection_axes
    assert result.temporal_bundle.predecessor_candidate_id is None
    # Only bounded, later entity near-misses remain visible.  They explain why
    # recency alone cannot replace the singular-friend boundary.
    assert _sources(result) == {
        "science::session",
        "lecture::session",
        "dad::session",
        "plural::session",
    }
    assert len(result.candidates) == 4
    assert all(
        "interval_entity_constraint_comparator" in row.selection_axes
        for row in result.candidates
        if row.candidate_id != result.temporal_bundle.winner_candidate_id
    )


def test_order_route_expands_sports_events_beyond_literal_query_stems(
    tmp_path: Path,
) -> None:
    index = _write_index(
        tmp_path / "sports-order.db",
        [
            (
                "triathlon::session",
                "I completed a city triathlon today after a long bicycle leg.",
                datetime(2024, 6, 3, tzinfo=timezone.utc),
            ),
            (
                "run::session",
                "I finished a midsummer 5K run today with a personal best.",
                datetime(2024, 6, 11, tzinfo=timezone.utc),
            ),
            (
                "soccer::session",
                "I participated in our annual charity soccer tournament today.",
                datetime(2024, 6, 18, tzinfo=timezone.utc),
            ),
            (
                "lexical-noise::session",
                "I wondered whether the waterfront is open during the week.",
                datetime(2024, 6, 8, tzinfo=timezone.utc),
            ),
            (
                "spectator::session",
                "I watched a soccer match at a neighborhood sports bar.",
                datetime(2024, 6, 15, tzinfo=timezone.utc),
            ),
            (
                "unrelated-event::session",
                "I attended a high-performance driving education event.",
                datetime(2024, 6, 20, tzinfo=timezone.utc),
            ),
        ],
    )
    question = (
        "[Question asked at 2024/07/01 (Mon) 12:00]\n"
        "What is the order of the three sports events I participated in "
        "during the past month, from earliest to latest?"
    )

    result = scan_temporal_insufficiency_specialist(index, question)

    assert result.routes == (SpecialistRoute.TEMPORAL_ORDER,)
    assert result.temporal_bundle is not None
    assert [row.event_date for row in result.candidates] == [
        "2024-06-03",
        "2024-06-11",
        "2024-06-18",
    ]
    assert _sources(result) == {
        "triathlon::session",
        "run::session",
        "soccer::session",
    }
    quotes = "\n".join(row.quote for row in result.candidates)
    assert "triathlon" in quotes
    assert "5K run" in quotes
    assert "soccer tournament" in quotes
    assert "waterfront" not in quotes
    assert "sports bar" not in quotes


def test_order_route_prefers_same_source_completed_event_over_adjacent_discussion(
    tmp_path: Path,
) -> None:
    index = _write_index(
        tmp_path / "sports-source-representative.db",
        [
            (
                "triathlon::session",
                "I completed a city triathlon today after a long bicycle leg.",
                datetime(2024, 6, 3, tzinfo=timezone.utc),
            ),
            (
                "run::session",
                "I finished a midsummer 5K run today with a personal best.",
                datetime(2024, 6, 11, tzinfo=timezone.utc),
            ),
            (
                "soccer::session",
                (
                    "I participate in our annual charity soccer tournament "
                    "today, and I want to take care of myself."
                ),
                datetime(2024, 6, 18, tzinfo=timezone.utc),
            ),
            (
                "soccer::session",
                (
                    "I'm learning about hydration for athletes. Can you explain "
                    "how I can stay hydrated during the soccer tournament?"
                ),
                datetime(2024, 6, 18, tzinfo=timezone.utc),
            ),
            (
                "future::session",
                "I want to participate in a community soccer tournament next week.",
                datetime(2024, 6, 20, tzinfo=timezone.utc),
            ),
        ],
    )
    question = (
        "[Question asked at 2024/07/01 (Mon) 12:00]\n"
        "What is the order of the three sports events I participated in "
        "during the past month, from earliest to latest?"
    )

    result = scan_temporal_insufficiency_specialist(index, question)

    assert [row.event_date for row in result.candidates] == [
        "2024-06-03",
        "2024-06-11",
        "2024-06-18",
    ]
    soccer = next(row for row in result.candidates if row.event_date == "2024-06-18")
    assert "I participate in our annual charity soccer tournament" in soccer.quote
    assert "hydration" not in soccer.quote
    assert "want to participate" not in "\n".join(
        row.quote for row in result.candidates
    )
    assert "concrete_completed_event_surface" in soccer.selection_axes


def test_order_route_keeps_all_three_requested_operands_chronologically(
    tmp_path: Path,
) -> None:
    index = _write_index(
        tmp_path / "order.db",
        [
            (
                "muir::session",
                "I took a day hike to Muir Woods with my family.",
                datetime(2023, 3, 10, tzinfo=timezone.utc),
            ),
            (
                "muir::session",
                "I packed my Eastern Sierra backpack and bear canister.",
                datetime(2023, 3, 10, tzinfo=timezone.utc),
            ),
            (
                "bigsur::session",
                "I drove on a trip through Big Sur and Monterey.",
                datetime(2023, 4, 20, tzinfo=timezone.utc),
            ),
            (
                "bigsur::session",
                "My Eastern Sierra backpack needed a better bear canister.",
                datetime(2023, 4, 20, tzinfo=timezone.utc),
            ),
            (
                "yosemite::session",
                "I went on a camping trip to Yosemite.",
                datetime(2023, 5, 15, tzinfo=timezone.utc),
            ),
            (
                "yosemite::session",
                "I reused the Eastern Sierra backpack and bear canister.",
                datetime(2023, 5, 15, tzinfo=timezone.utc),
            ),
            (
                "unlinked-new-york::session",
                "I recently took a quick business trip to New York City.",
                datetime(2023, 5, 29, tzinfo=timezone.utc),
            ),
            (
                "unlinked-yellowstone::session",
                "I drove 2500 miles on my Yellowstone road trip.",
                datetime(2023, 5, 25, tzinfo=timezone.utc),
            ),
            (
                "old::session",
                "I traveled to Yellowstone on a winter vacation.",
                datetime(2022, 12, 1, tzinfo=timezone.utc),
            ),
        ],
    )
    question = (
        "[Question asked at 2023/06/01 (Thu) 03:56]\n"
        "What is the order of the three trips I took in the past three months, "
        "from earliest to latest?"
    )

    result = scan_temporal_insufficiency_specialist(index, question)

    assert result.routes == (SpecialistRoute.TEMPORAL_ORDER,)
    assert result.temporal_bundle is not None
    assert result.temporal_bundle.requested_cardinality == 3
    ordered = [
        next(row for row in result.candidates if row.candidate_id == candidate_id)
        for candidate_id in result.temporal_bundle.ordered_candidate_ids
    ]
    assert len(ordered) == 3
    assert [row.event_date for row in ordered] == [
        "2023-03-10",
        "2023-04-20",
        "2023-05-15",
    ]
    assert all(row.bundle_role is BundleRole.ORDERED_OPERAND for row in ordered)


def test_business_typo_near_target_day_beats_earlier_comparator(
    tmp_path: Path,
) -> None:
    index = _write_index(
        tmp_path / "business.db",
        [
            (
                "launch::session",
                "I just launched my website and created a business plan outline.",
                datetime(2023, 2, 10, tzinfo=timezone.utc),
            ),
            (
                "client::session",
                "I just signed a contract with my first client today.",
                datetime(2023, 3, 1, tzinfo=timezone.utc),
            ),
            (
                "noise::session",
                "I just did an online grocery order and spent around $60 today.",
                datetime(2023, 2, 28, tzinfo=timezone.utc),
            ),
        ],
    )
    question = (
        "[Question asked at 2023/03/28 (Tue) 20:35]\n"
        "What was the significant buisiness milestone I mentioned four weeks ago?"
    )

    result = scan_temporal_insufficiency_specialist(index, question)

    assert result.temporal_bundle is not None
    winner = next(
        row
        for row in result.candidates
        if row.candidate_id == result.temporal_bundle.winner_candidate_id
    )
    assert winner.event_date == "2023-03-01"
    assert "first client" in winner.quote
    predecessor = next(
        row
        for row in result.candidates
        if row.candidate_id == result.temporal_bundle.predecessor_candidate_id
    )
    assert "launched my website" in predecessor.quote
    assert {"launch::session", "client::session"} <= _sources(result)

    contribution = adapt_temporal_insufficiency_to_typed_contribution(
        result, handle_start=730_001, group_start=740_001
    )
    status_by_summary = {
        item.summary: item.status for item in contribution.parsed.accepted_items
    }
    assert status_by_summary[winner.quote] is EvidenceStatus.COMPLETED
    assert status_by_summary[predecessor.quote] is EvidenceStatus.COMPLETED


def test_scoped_numeric_absence_ignores_unlinked_chili_number(
    tmp_path: Path,
) -> None:
    index = _write_index(
        tmp_path / "absence.db",
        [
            (
                "tomato::session",
                "I planted 5 tomato plants initially, and they are producing.",
                datetime(2023, 5, 22, tzinfo=timezone.utc),
            ),
            (
                "cucumber::session",
                "I have got 3 cucumber plants growing in the same garden.",
                datetime(2023, 5, 29, tzinfo=timezone.utc),
            ),
            (
                "unlinked::session",
                "I cooked chili for 8 people at a winter party.",
                datetime(2023, 1, 2, tzinfo=timezone.utc),
            ),
        ],
    )
    question = (
        "[Question asked at 2023/05/30 (Tue) 14:24]\n"
        "How many plants did I initially plant for tomatoes and chili peppers?"
    )

    result = scan_temporal_insufficiency_specialist(index, question)

    assert result.routes == (SpecialistRoute.NUMERIC_SLOT_INSUFFICIENCY,)
    certificate = result.absence_certificate
    assert certificate.every_exact_entity_posting_scanned is True
    assert certificate.every_scoped_source_row_scanned is True
    assert certificate.scoped_source_count == 2
    assert certificate.may_conclude_operator_insufficient is True
    assert certificate.semantic_absence_may_be_inferred is False
    by_label = {row.slot_label: row for row in certificate.slot_coverage}
    assert by_label["tomatoes"].explicit_numeric_assertion_source_count == 1
    assert by_label["tomatoes"].explicit_numeric_operand_missing is False
    assert by_label["chili peppers"].entity_assertion_window_count == 0
    assert by_label["chili peppers"].explicit_numeric_operand_missing is True
    assert "Do not infer or copy a count" in (certificate.provider_instruction or "")
    assert _sources(result) == {"tomato::session", "cucumber::session"}


def test_scoped_numeric_certificate_withholds_insufficiency_when_operand_exists(
    tmp_path: Path,
) -> None:
    index = _write_index(
        tmp_path / "covered.db",
        [
            (
                "tomato::session",
                "I planted 5 tomato plants initially.",
                datetime(2023, 5, 22, tzinfo=timezone.utc),
            ),
            (
                "chili::session",
                "I planted 4 chili pepper plants initially.",
                datetime(2023, 5, 25, tzinfo=timezone.utc),
            ),
        ],
    )
    question = (
        "[Question asked at 2023/05/30 (Tue) 14:24]\n"
        "How many plants did I initially plant for tomatoes and chili peppers?"
    )

    result = scan_temporal_insufficiency_specialist(index, question)

    assert result.absence_certificate.may_conclude_operator_insufficient is False
    assert result.absence_certificate.provider_instruction is None
    assert all(
        not row.explicit_numeric_operand_missing
        for row in result.absence_certificate.slot_coverage
    )


def test_projection_is_opaque_replayable_bounded_and_exactly_provenanced(
    tmp_path: Path,
) -> None:
    path = tmp_path / "provenance.db"
    index = _write_index(
        path,
        [
            (
                "secret-target::session",
                "I just planted 12 new tomato saplings today.",
                datetime(2023, 4, 21, tzinfo=timezone.utc),
            )
        ],
    )
    question = (
        "[Question asked at 2023/05/05 (Fri) 16:42]\n"
        "What gardening-related activity did I do two weeks ago?"
    )

    first = scan_temporal_insufficiency_specialist(index, question)
    second = scan_temporal_insufficiency_specialist(index, question)

    assert first.provider_projection() == second.provider_projection()
    assert first.local_audit_projection() == second.local_audit_projection()
    provider_json = json.dumps(first.provider_projection(), sort_keys=True)
    assert "secret-target" not in provider_json
    assert '"source_id"' not in provider_json
    assert '"partition_id"' not in provider_json
    assert first.receipt.new_provider_calls == 0
    assert first.receipt.retained_transformer_token_state_bytes == 0
    assert first.receipt.gold_loaded is False
    assert first.receipt.selected_evidence_tokens <= first.budget.evidence_token_cap
    assert (
        first.receipt.provider_payload_tokens
        + first.budget.output_token_reserve
        + first.budget.protocol_token_reserve
        <= first.budget.hard_prompt_token_cap
    )
    binding = first.local_bindings[0]
    with Database(path, read_only=True) as database:
        assert DiscourseStore(database).hydrate_span(binding.span) == first.candidates[0].quote

    contribution = adapt_temporal_insufficiency_to_typed_contribution(
        first, handle_start=710_001, group_start=720_001
    )
    assert contribution.frontier_mode is FrontierMode.BOUNDED
    assert not contribution.parsed.rejected_items
    assert all(
        row.origin is EvidenceOrigin.DIRECT_POINTER
        and row.provenance_grade is ProvenanceGrade.DIRECT_POINTER
        and row.evidence_receipt_sha256 == local.receipt_sha256
        for row, local in zip(
            contribution.bindings, first.local_bindings, strict=True
        )
    )
