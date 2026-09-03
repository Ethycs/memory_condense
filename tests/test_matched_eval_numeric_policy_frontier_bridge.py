from __future__ import annotations

from dataclasses import replace
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
from tools.matched_eval.numeric_operand_specialist import scan_numeric_operand_closure
from tools.matched_eval.numeric_policy_frontier_bridge import (
    EXTENDED_SUPPORTED_DOMAINS,
    NumericPolicyFrontierBridgeError,
    build_operator_first_numeric_frontier,
    operator_first_numeric_frontier_applicable,
)
from tools.matched_eval import numeric_policy_frontier_bridge as bridge_module
from tools.matched_eval.operator_first_numeric_policy import (
    execute_operator_first_numeric_policy,
)
from tools.matched_eval.query_expansion import FrozenSourceNamespace
from tools.matched_eval.query_guided_scan import cache_namespace_partitions
from tools.matched_eval.typed_operator_executor import ExecutionStatus


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _write_index(
    path: Path,
    rows: list[tuple[str, str, datetime] | tuple[str, str, datetime, str]],
):
    database = Database(path)
    transcript = TranscriptStore(database)
    lexical = LexicalIndex(database)
    for offset, raw in enumerate(rows):
        source_id, text, created_at = raw[:3]
        role = raw[3] if len(raw) == 4 else "user"
        turn = transcript.append(
            role,
            text,
            source_id=source_id,
            created_at=created_at,
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


def _item(
    handle: str,
    summary: str,
    *,
    date: str,
    status: str = "completed",
    relation: str = "authored_by_user;date_basis=source_created_at",
) -> dict[str, object]:
    return {
        "content_coherence": "match",
        "date": date,
        "handle_ids": [handle],
        "included": True,
        "kind": "direct",
        "relation": relation,
        "status": status,
        "summary": summary,
        "supported_slot_ids": [],
        "value_authority": "explicit",
    }


def _provider(
    question: str,
    items: list[dict[str, object]],
    *,
    include_proposed: bool = False,
) -> dict[str, object]:
    handles = tuple(
        dict.fromkeys(
            handle
            for item in items
            for handle in item["handle_ids"]  # type: ignore[index]
        )
    )
    return {
        "dated_question": question,
        "protected_parent_fallback": {
            "label": "fallback_not_evidence",
            "prediction": "not consulted",
        },
        "typed_evidence": {
            "conflict_policy": "quarantine",
            "format": "test-compact-typed-evidence-v1",
            "frontier": {
                "closed": False,
                "mode": "open",
                "truncated": True,
            },
            "handles": [
                {
                    "group_handle": f"G{offset:03d}",
                    "handle_id": handle,
                    "origin": "map",
                }
                for offset, handle in enumerate(handles, start=1)
            ],
            "items": items,
            "operator_spec": {
                "answer_shape": "number",
                "comparison_mode": "none",
                "include_proposed": include_proposed,
                "operation": "count_or_aggregate",
                "query_timestamp": question.split("]", 1)[0].removeprefix(
                    "[Question asked at "
                ),
                "required_slots": [],
                "requires_complete_frontier": True,
                "style": "numeric_reduce",
                "temporal_window_days": None,
            },
        },
    }


def _plant_fixture(tmp_path: Path):
    asked = datetime(2023, 5, 30, 21, 51, tzinfo=timezone.utc)
    peace = "I bought the peace lily and a succulent plant two weeks ago."
    snake = "My snake plant, which I got from my sister last month, needs repotting."
    index = _write_index(
        tmp_path / "plants.db",
        [
            ("garden-a::peace", peace, asked - timedelta(days=2)),
            ("garden-b::snake", snake, asked - timedelta(days=1)),
            (
                "garden-c::fern",
                "I watered my fern yesterday.",
                asked - timedelta(days=1),
            ),
            (
                "garden-d::assistant",
                "I bought an orchid yesterday in this hypothetical example.",
                asked - timedelta(days=1),
                "assistant",
            ),
        ],
    )
    question = (
        "[Question asked at 2023/05/30 (Tue) 21:51]\n"
        "How many plants did I acquire in the last month?"
    )
    specialist = scan_numeric_operand_closure(index, question)
    items = [
        _item("H001", peace, date=(asked - timedelta(days=2)).isoformat()),
        _item(
            "H002",
            snake,
            date=(asked - timedelta(days=1)).isoformat(),
            status="unknown",
        ),
    ]
    return index, question, specialist, items


def test_full_store_census_closes_only_after_exact_bidirectional_mapping(
    tmp_path: Path,
) -> None:
    index, question, specialist, items = _plant_fixture(tmp_path)
    provider = _provider(question, items)

    bridge = build_operator_first_numeric_frontier(
        provider,
        index=index,
        specialist_result=specialist,
    )
    decision = execute_operator_first_numeric_policy(
        provider,
        relevant_frontier=bridge.frontier,
    )

    assert operator_first_numeric_frontier_applicable(provider) is True
    assert bridge.closed is True
    assert bridge.unresolved_candidate_keys == ()
    assert bridge.physical_content_rows_scanned == len(index.rows)
    assert bridge.physical_sentence_windows_scanned == len(index.windows)
    assert len(bridge.census_atoms) == 3
    assert len(bridge.represented_semantic_key_sha256s) == 3
    assert decision.status is ExecutionStatus.SUPPORTED
    assert decision.prediction == "3"
    assert decision.used_handle_ids == ("H001", "H002")


def test_full_content_row_recovers_multisentence_commuter_bike_coreference(
    tmp_path: Path,
) -> None:
    asked = datetime(2023, 3, 20, 23, 57, tzinfo=timezone.utc)
    road = "I got my road bike serviced on March 10th."
    commuter = (
        "My commuter bike has a worn rear tire. "
        "It's time to replace it this month."
    )
    index = _write_index(
        tmp_path / "multisentence-bikes.db",
        [
            ("garage-a::road", road, asked - timedelta(days=8)),
            ("garage-b::commuter", commuter, asked),
        ],
    )
    question = (
        "[Question asked at 2023/03/20 (Mon) 23:57]\n"
        "How many bikes did I service or plan to service in March?"
    )
    provider = _provider(
        question,
        [
            _item(
                "H001",
                road,
                date=(asked - timedelta(days=8)).isoformat(),
            ),
            _item(
                "H002",
                commuter,
                date=asked.isoformat(),
                status="proposed",
            ),
        ],
        include_proposed=True,
    )
    specialist = scan_numeric_operand_closure(index, question)

    bridge = build_operator_first_numeric_frontier(
        provider,
        index=index,
        specialist_result=specialist,
    )
    decision = execute_operator_first_numeric_policy(
        provider,
        relevant_frontier=bridge.frontier,
    )

    assert bridge.closed is True
    assert {row.entity_key for row in bridge.census_atoms} == {
        "road_bike",
        "commuter_bike",
    }
    assert decision.prediction == "2"


def test_full_content_row_preserves_original_and_replacement_boot_roles(
    tmp_path: Path,
) -> None:
    asked = datetime(2023, 2, 15, 23, 50, tzinfo=timezone.utc)
    blazer = "I still need to pick up my dry cleaning for the navy blue blazer."
    boots = (
        "I need to return some boots because they were too small. "
        "I exchanged them for a larger size, but I haven't had a chance "
        "to pick them up yet."
    )
    index = _write_index(
        tmp_path / "multisentence-boots.db",
        [
            ("store-a::blazer", blazer, asked - timedelta(days=1)),
            ("store-b::boots", boots, asked - timedelta(hours=2)),
        ],
    )
    question = (
        "[Question asked at 2023/02/15 (Wed) 23:50]\n"
        "How many items of clothing do I need to pick up or return from a store?"
    )
    provider = _provider(
        question,
        [
            _item(
                "H001",
                blazer,
                date=(asked - timedelta(days=1)).isoformat(),
                status="current",
            ),
            _item(
                "H002",
                boots,
                date=(asked - timedelta(hours=2)).isoformat(),
                status="current",
            ),
        ],
    )
    specialist = scan_numeric_operand_closure(index, question)

    bridge = build_operator_first_numeric_frontier(
        provider,
        index=index,
        specialist_result=specialist,
    )
    decision = execute_operator_first_numeric_policy(
        provider,
        relevant_frontier=bridge.frontier,
    )

    assert bridge.closed is True
    assert {(row.action_key, row.entity_key) for row in bridge.census_atoms} == {
        ("pickup", "navy_blue_blazer"),
        ("return", "original_boot"),
        ("pickup", "replacement_boot"),
    }
    assert decision.prediction == "3"


def test_omitted_full_store_operand_keeps_frontier_open(tmp_path: Path) -> None:
    index, question, specialist, items = _plant_fixture(tmp_path)
    provider = _provider(question, items[:1])

    bridge = build_operator_first_numeric_frontier(
        provider,
        index=index,
        specialist_result=specialist,
    )
    decision = execute_operator_first_numeric_policy(
        provider,
        relevant_frontier=bridge.frontier,
    )

    assert bridge.closed is False
    assert any(value.startswith("missing:") for value in bridge.unresolved_candidate_keys)
    assert decision.status is ExecutionStatus.INSUFFICIENT
    assert decision.reason == "relevant_candidate_frontier_not_closed"


def test_provider_only_candidate_keeps_frontier_open(tmp_path: Path) -> None:
    index, question, specialist, items = _plant_fixture(tmp_path)
    items.append(
        _item(
            "H003",
            "I bought an orchid yesterday.",
            date="2023-05-29T21:51:00+00:00",
        )
    )
    provider = _provider(question, items)

    bridge = build_operator_first_numeric_frontier(
        provider,
        index=index,
        specialist_result=specialist,
    )

    assert bridge.closed is False
    assert any(
        value.startswith("provider_only:")
        for value in bridge.unresolved_candidate_keys
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "unknown"),
        ("date", "2023-04-01T00:00:00+00:00"),
        ("relation", "authored_by_assistant;date_basis=source_created_at"),
        ("content_coherence", "conflict"),
    ],
)
def test_material_status_date_role_and_conflict_mismatch_fail_closed(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    index, question, specialist, items = _plant_fixture(tmp_path)
    changed = [dict(row) for row in items]
    changed[0][field] = value

    bridge = build_operator_first_numeric_frontier(
        _provider(question, changed),
        index=index,
        specialist_result=specialist,
    )

    assert bridge.closed is False
    assert bridge.unresolved_candidate_keys


def test_operator_material_profile_collapses_only_admitted_status_variants(
    tmp_path: Path,
) -> None:
    index, question, specialist, items = _plant_fixture(tmp_path)
    changed = [dict(row) for row in items]
    changed[0]["status"] = "unknown"
    provider = _provider(question, changed)

    strict = build_operator_first_numeric_frontier(
        provider,
        index=index,
        specialist_result=specialist,
    )
    operator_material = build_operator_first_numeric_frontier(
        provider,
        index=index,
        specialist_result=specialist,
        operator_material_status=True,
    )

    assert strict.closed is False
    assert operator_material.closed is True
    assert {row.status for row in operator_material.census_atoms} == {
        "operator_eligible"
    }
    assert (
        execute_operator_first_numeric_policy(
            provider,
            relevant_frontier=operator_material.frontier,
        ).prediction
        == "3"
    )


def test_numeric_unit_and_contribution_material_mismatch_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index, question, specialist, items = _plant_fixture(tmp_path)
    provider = _provider(question, items)
    original_compile = bridge_module.compile_operator_first_numeric_candidates

    def altered_compile(value):
        compilation = original_compile(value)
        if value is not provider or not compilation.candidate_atoms:
            return compilation
        altered = replace(
            compilation.candidate_atoms[0],
            numeric_value=9.0,
            unit="items",
            contribution_value=2.0,
            receipt_sha256="",
        )
        return replace(
            compilation,
            candidate_atoms=(altered, *compilation.candidate_atoms[1:]),
            receipt_sha256="",
        )

    monkeypatch.setattr(
        bridge_module,
        "compile_operator_first_numeric_candidates",
        altered_compile,
    )

    bridge = build_operator_first_numeric_frontier(
        provider,
        index=index,
        specialist_result=specialist,
    )

    assert bridge.closed is False
    assert any(
        value.startswith(("missing_fact:", "provider_only_fact:"))
        for value in bridge.unresolved_candidate_keys
    )


def test_empty_supported_census_does_not_claim_zero_count_closure(
    tmp_path: Path,
) -> None:
    asked = datetime(2023, 5, 30, 21, 51, tzinfo=timezone.utc)
    index = _write_index(
        tmp_path / "empty-plants.db",
        [
            (
                "garden-a::water",
                "I watered my fern yesterday.",
                asked - timedelta(days=1),
            )
        ],
    )
    question = (
        "[Question asked at 2023/05/30 (Tue) 21:51]\n"
        "How many plants did I acquire in the last month?"
    )
    provider = _provider(question, [])
    specialist = scan_numeric_operand_closure(index, question)

    bridge = build_operator_first_numeric_frontier(
        provider,
        index=index,
        specialist_result=specialist,
    )

    assert operator_first_numeric_frontier_applicable(provider) is True
    assert bridge.closed is False
    assert bridge.unresolved_candidate_keys == ("empty_candidate_census",)


def test_legacy_specialist_selection_truncation_is_audit_only(
    tmp_path: Path,
) -> None:
    index, question, _default_specialist, items = _plant_fixture(tmp_path)
    scanned = scan_numeric_operand_closure(index, question)
    specialist = replace(
        scanned,
        receipt=replace(
            scanned.receipt,
            selection_truncated=True,
            receipt_sha256="",
        ),
    )
    assert specialist.receipt.selection_truncated is True

    bridge = build_operator_first_numeric_frontier(
        _provider(question, items),
        index=index,
        specialist_result=specialist,
    )

    assert bridge.specialist_selection_truncated is True
    assert bridge.closed is True
    assert bridge.frontier.selection_truncated is False


def test_domain_whitelist_leaves_jewelry_out_of_scope() -> None:
    question = (
        "[Question asked at 2023/05/30 (Tue) 15:43]\n"
        "How many pieces of jewelry did I acquire in the last two months?"
    )
    provider = _provider(
        question,
        [_item("H001", "I bought a silver necklace.", date="2023-05-29")],
    )

    assert operator_first_numeric_frontier_applicable(provider) is False
    assert (
        operator_first_numeric_frontier_applicable(
            provider,
            supported_domains=EXTENDED_SUPPORTED_DOMAINS,
        )
        is True
    )


def test_bridge_rejects_specialist_from_another_question(tmp_path: Path) -> None:
    index, question, _specialist, items = _plant_fixture(tmp_path)
    wrong = scan_numeric_operand_closure(
        index,
        "[Question asked at 2023/05/30 (Tue) 21:51] "
        "How many bikes did I acquire in the last month?",
    )

    with pytest.raises(
        NumericPolicyFrontierBridgeError,
        match="another question",
    ):
        build_operator_first_numeric_frontier(
            _provider(question, items),
            index=index,
            specialist_result=wrong,
        )


def test_bridge_replay_is_byte_stable_and_parent_independent(tmp_path: Path) -> None:
    index, question, specialist, items = _plant_fixture(tmp_path)
    first_provider = _provider(question, items)
    second_provider = _provider(question, items)
    second_provider["protected_parent_fallback"] = {
        "label": "fallback_not_evidence",
        "prediction": "different parent",
    }

    first = build_operator_first_numeric_frontier(
        first_provider,
        index=index,
        specialist_result=specialist,
    )
    second = build_operator_first_numeric_frontier(
        second_provider,
        index=index,
        specialist_result=specialist,
    )

    assert first.projection() == second.projection()
    assert first.receipt_sha256 == second.receipt_sha256
    assert first.provider_prompt_count == 0
    assert first.gold_loaded is False
