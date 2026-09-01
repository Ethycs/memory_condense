from __future__ import annotations

import inspect
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from memory_condense.application.discourse_sources import scan_discourse_source_chunks
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.schemas import Chunk
from memory_condense.persistence.db import Database
from memory_condense.persistence.discourse_store import DiscourseStore
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.search.indexes.lexical import LexicalIndex

from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.full_store_slot_closure import build_full_store_window_index
from tools.matched_eval.numeric_operand_specialist import (
    NumericOperandBudget,
    NumericOperandSpecialistError,
    adapt_numeric_operand_closure_to_typed_contribution,
    scan_numeric_operand_closure,
)
from tools.matched_eval.query_expansion import FrozenSourceNamespace
from tools.matched_eval.query_guided_scan import cache_namespace_partitions
from tools.matched_eval.typed_operator_spec import compile_typed_operator_spec


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _write_index(
    path: Path,
    rows: list[tuple[str, str, datetime] | tuple[str, str, datetime, str]],
):
    database = Database(path)
    transcript = TranscriptStore(database)
    lexical = LexicalIndex(database)
    for index, raw in enumerate(rows):
        source_id, text, created_at = raw[:3]
        role = raw[3] if len(raw) == 4 else "user"
        turn = transcript.append(
            role, text, source_id=source_id, created_at=created_at
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
    store_receipt = _sha(f"store:{path.name}")
    namespace = FrozenSourceNamespace.from_source_streams(
        snapshot_id=_sha(f"snapshot:{path.name}"),
        combined_store_receipt_sha256=store_receipt,
        source_streams=streams,
    )
    with Database(path, read_only=True) as readonly:
        cache = cache_namespace_partitions(
            readonly,
            namespace,
            source_database_sha256=_sha(f"database:{path.name}"),
            source_store_receipt_sha256=store_receipt,
        )
    return build_full_store_window_index(cache)


def test_total_feed_weight_reserves_both_measure_operands_with_exact_citations(
    tmp_path: Path,
) -> None:
    asked = datetime(2023, 5, 30, 12, 23, tzinfo=timezone.utc)
    index = _write_index(
        tmp_path / "feed-operands.db",
        [
            (
                "farm-a::layer",
                "I'm tracking the layer feed I recently purchased. "
                "I got a 50-pound batch for the chickens.",
                asked - timedelta(days=7),
            ),
            (
                "farm-b::scratch",
                "I also bought 20 pounds of organic scratch grains for my chickens recently.",
                asked - timedelta(hours=10),
            ),
            (
                "farm-c::noise",
                "I spent $120 and got a 10% discount at the farm store.",
                asked - timedelta(days=2),
            ),
            (
                "farm-d::assistant",
                "I purchased 900 pounds of feed in this hypothetical example.",
                asked - timedelta(days=1),
                "assistant",
            ),
        ],
    )
    question = (
        "[Question asked at 2023/05/30 (Tue) 12:23] "
        "What is the total weight of the new feed I purchased in the past two months?"
    )

    result = scan_numeric_operand_closure(index, question)

    assert result.operation_mode == "sum"
    assert result.expected_numeric_dimension == "measure"
    assert sorted(
        value for group in result.operand_groups for value in group.operand_values
    ) == [20.0, 50.0]
    assert {group.entity_key for group in result.operand_groups} == {
        "layer_feed",
        "scratch_grain",
    }
    assert sum(group.operand_values[0] for group in result.operand_groups) == 70
    assert result.receipt.all_plausible_operand_groups_reserved is True
    assert result.receipt.physical_content_rows_scanned == 4
    assert result.receipt.physical_scan_exhaustive is True
    assert result.receipt.new_provider_calls == 0
    assert result.receipt.retained_transformer_token_state_bytes == 0
    assert {binding.source_id for binding in result.local_bindings} == {
        "farm-a::layer",
        "farm-b::scratch",
    }
    with Database(tmp_path / "feed-operands.db", read_only=True) as database:
        store = DiscourseStore(database)
        for candidate, binding in zip(
            result.candidates, result.local_bindings, strict=True
        ):
            assert store.hydrate_span(binding.span) == candidate.quote
    provider_json = json.dumps(result.provider_projection(), sort_keys=True)
    assert '"source_id"' not in provider_json
    assert '"partition_id"' not in provider_json
    assert "farm-a::layer" not in provider_json
    assert (
        result.receipt.provider_payload_tokens
        + result.budget.output_token_reserve
        + result.budget.protocol_token_reserve
        <= result.budget.hard_prompt_token_cap
    )
    contribution = adapt_numeric_operand_closure_to_typed_contribution(
        result,
        handle_start=600_001,
        group_start=600_001,
    )
    assert sorted(
        row.numeric_value for row in contribution.parsed.accepted_items
    ) == [20.0, 50.0]
    assert {row.unit for row in contribution.parsed.accepted_items} == {"lb"}
    assert all(
        row.value_authority.value == "explicit"
        and row.numeric_role.value == "operand"
        for row in contribution.parsed.accepted_items
    )
    assert not contribution.parsed.rejected_items


def test_furniture_count_groups_repeated_purchase_but_keeps_four_operations(
    tmp_path: Path,
) -> None:
    asked = datetime(2023, 5, 30, 18, 48, tzinfo=timezone.utc)
    index = _write_index(
        tmp_path / "furniture-operands.db",
        [
            (
                "home-a::one",
                "I just got a new coffee table from West Elm about three weeks ago.",
                asked - timedelta(days=21),
            ),
            (
                "home-b::two",
                "I just got a new coffee table and rearranged my living room. "
                "Last week I finally ordered a new mattress from Casper.",
                asked - timedelta(days=14),
            ),
            (
                "home-c::three",
                "I finally assembled that IKEA bookshelf for my home office about two months ago.",
                asked - timedelta(days=60),
            ),
            (
                "home-d::four",
                "I just got a new coffee table and rearranged the furniture.",
                asked - timedelta(days=16),
            ),
            (
                "home-d::four",
                "I finally got around to fixing the wobbly leg on my kitchen table last weekend.",
                asked - timedelta(days=7),
            ),
            (
                "home-e::proposal",
                "I'm thinking I might buy a desk next year.",
                asked - timedelta(days=1),
            ),
            (
                "home-f::idiom",
                "My parents might like a bed and breakfast near the vineyard "
                "where my cousin got married.",
                asked - timedelta(days=5),
            ),
            (
                "home-g::background",
                "I recently got Max new dog food, so I should wash his blankets "
                "and beds before he uses the new bed.",
                asked - timedelta(days=3),
            ),
        ],
    )
    question = (
        "[Question asked at 2023/05/30 (Tue) 18:48] "
        "How many pieces of furniture did I buy, assemble, sell, or fix in the past few months?"
    )

    result = scan_numeric_operand_closure(index, question)

    assert result.operation_mode == "count"
    assert {
        (group.action_class, group.entity_key) for group in result.operand_groups
    } == {
        ("buy", "coffee_table"),
        ("buy", "mattress"),
        ("assemble", "bookshelf"),
        ("fix", "kitchen_table"),
    }
    assert sum(group.operand_values[0] for group in result.operand_groups) == 4
    coffee = next(
        group for group in result.operand_groups if group.entity_key == "coffee_table"
    )
    assert coffee.value_basis == "implicit_distinct_event_unit"
    assert len(coffee.source_group_handles) == 3
    assert len(coffee.candidate_ids) == 3
    assert result.receipt.multi_mention_operand_group_count == 1
    assert result.receipt.all_plausible_operand_groups_reserved is True
    assert all("proposal" not in row.source_id for row in result.local_bindings)

    contribution = adapt_numeric_operand_closure_to_typed_contribution(
        result,
        handle_start=700_001,
        group_start=700_001,
    )
    assert len(contribution.bindings) == len(result.candidates)
    assert len(contribution.parsed.accepted_items) == 4
    assert not contribution.parsed.rejected_items
    coffee_item = next(
        row
        for row in contribution.parsed.accepted_items
        if row.entity_key == "coffee_table"
    )
    assert coffee_item.numeric_value == 1
    assert coffee_item.numeric_role.value == "operand"
    assert coffee_item.value_authority.value == "derived"
    assert len(coffee_item.handle_ids) == 3
    assert {
        row.group_key for row in contribution.parsed.accepted_items
    } == {row.operand_group_id for row in result.operand_groups}
    assert all(
        typed.evidence_receipt_sha256 == local.receipt_sha256
        and typed.citation_sha256 == candidate.quote_sha256
        and typed.local_source_locator_sha256 == local.receipt_sha256
        for typed, local, candidate in zip(
            contribution.bindings,
            result.local_bindings,
            result.candidates,
            strict=True,
        )
    )
    assert contribution.provider_prompt_count == 0
    assert contribution.retained_transformer_token_state_bytes == 0


def test_lanes_select_independently_then_exact_span_dedup_and_replay(
    tmp_path: Path,
) -> None:
    asked = datetime(2026, 8, 27, 12, tzinfo=timezone.utc)
    index = _write_index(
        tmp_path / "lane-dedup.db",
        [
            (
                "seeded::feed",
                "I purchased a 12-pound bag of layer feed yesterday.",
                asked - timedelta(days=1),
            )
        ],
    )
    question = (
        "[Question asked at 2026/08/27 12:00] "
        "What was the total weight of feed I purchased recently?"
    )

    first = scan_numeric_operand_closure(
        index,
        question,
        seed_source_ids=("seeded::feed",),
        seed_history_ids=("feed",),
    )
    second = scan_numeric_operand_closure(
        index,
        question,
        seed_source_ids=("seeded::feed",),
        seed_history_ids=("feed",),
    )

    assert first.receipt.independent_lane_selected_occurrence_count >= 5
    assert first.receipt.post_selection_exact_span_count == 1
    assert first.receipt.exact_span_duplicate_count >= 4
    assert first.receipt.exact_span_dedup_stage == (
        "after_independent_lane_selection"
    )
    assert set(first.candidates[0].selection_lanes) >= {
        "numeric_operands",
        "event_operands",
        "source_diverse_operands",
        "seeded_operand_closure",
        "action:buy",
    }
    assert first.provider_projection() == second.provider_projection()
    assert first.local_audit_projection() == second.local_audit_projection()
    assert first.receipt.receipt_sha256 == second.receipt.receipt_sha256


def test_tight_cap_reports_unreserved_group_without_claiming_completeness(
    tmp_path: Path,
) -> None:
    asked = datetime(2026, 8, 27, 12, tzinfo=timezone.utc)
    index = _write_index(
        tmp_path / "bounded.db",
        [
            (
                "room-a::one",
                "I bought a coffee table yesterday.",
                asked - timedelta(days=1),
            ),
            (
                "room-b::two",
                "I assembled a bookshelf yesterday.",
                asked - timedelta(days=1),
            ),
        ],
    )
    question = (
        "[Question asked at 2026/08/27 12:00] "
        "How many pieces of furniture did I buy or assemble?"
    )
    budget = NumericOperandBudget(
        evidence_token_cap=32,
        max_candidates=1,
        max_candidates_per_lane=1,
        lane_token_cap=32,
        max_candidates_per_source=1,
        max_window_sentences=1,
        max_excerpt_tokens=32,
        max_operand_groups=1,
    )

    result = scan_numeric_operand_closure(index, question, budget=budget)

    assert len(result.candidates) == 1
    assert len(result.operand_groups) == 1
    assert result.receipt.plausible_operand_group_count == 2
    assert result.receipt.all_plausible_operand_groups_reserved is False
    assert result.receipt.selection_truncated is True
    assert result.receipt.semantic_completeness_status == "not_claimed"


def test_contract_has_no_outcome_or_provider_inputs_and_rejects_wrong_spec(
    tmp_path: Path,
) -> None:
    parameters = inspect.signature(scan_numeric_operand_closure).parameters
    assert {
        "question_id",
        "reference",
        "prediction",
        "provider",
        "expected_source_ids",
    }.isdisjoint(parameters)
    index = _write_index(
        tmp_path / "wrong-spec.db",
        [
            (
                "room::one",
                "I bought a coffee table.",
                datetime(2026, 8, 1, tzinfo=timezone.utc),
            )
        ],
    )
    question = (
        "[Question asked at 2026/08/27 12:00] "
        "How many pieces of furniture did I buy?"
    )
    wrong = compile_typed_operator_spec(
        "[Question asked at 2026/08/27 12:00] How many apples did I buy?"
    )

    with pytest.raises(
        NumericOperandSpecialistError,
        match="different question",
    ):
        scan_numeric_operand_closure(index, question, operator_spec=wrong)
