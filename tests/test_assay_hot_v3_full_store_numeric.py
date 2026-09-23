from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.application.discourse_sources import scan_discourse_source_chunks
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.schemas import Chunk
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.search.indexes.lexical import LexicalIndex
from tools import assay_hot_v3_full_store_numeric as assay
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.full_store_slot_closure import build_full_store_window_index
from tools.matched_eval.query_expansion import FrozenSourceNamespace
from tools.matched_eval.query_guided_scan import cache_namespace_partitions


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _write_index(path: Path, rows: list[tuple[str, str, datetime]]):
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


def _item(handle: str, summary: str, *, created_at: datetime) -> dict[str, object]:
    return {
        "content_coherence": "match",
        "date": created_at.isoformat(),
        "handle_ids": [handle],
        "included": True,
        "kind": "direct",
        "relation": "authored_by_user;date_basis=source_created_at",
        "status": "current",
        "summary": summary,
        "supported_slot_ids": [],
        "value_authority": "explicit",
    }


def _local_input(question: str, items: list[dict[str, object]]) -> dict[str, object]:
    handles = tuple(
        dict.fromkeys(
            handle
            for item in items
            for handle in item["handle_ids"]  # type: ignore[index]
        )
    )
    return {
        "dated_question": question,
        "typed_evidence": {
            "conflict_policy": "quarantine",
            "format": "memory-condense-hot-v3-local-operator-evidence-v1",
            "frontier": {"closed": False, "mode": "bounded", "truncated": True},
            "gold_loaded": False,
            "handles": [
                {
                    "group_handle": f"G{offset:03d}",
                    "handle_id": handle,
                    "origin": "direct_pointer",
                }
                for offset, handle in enumerate(handles, start=1)
            ],
            "items": items,
            "local_only": True,
            "operator_spec": {
                "answer_shape": "number",
                "comparison_mode": "none",
                "include_proposed": False,
                "operation": "count_or_aggregate",
                "query_timestamp": question.split("]", 1)[0].removeprefix(
                    "[Question asked at "
                ),
                "required_slots": [],
                "requires_complete_frontier": True,
                "style": "numeric_reduce",
                "temporal_window_days": None,
            },
            "provider_prompt_count": 0,
            "provider_use_forbidden": True,
            "retained_transformer_token_state_bytes": 0,
        },
    }


def test_closed_frontier_is_the_only_admissible_replacement(tmp_path: Path) -> None:
    asked = datetime(2023, 2, 15, 23, 50, tzinfo=timezone.utc)
    blazer = "I still need to pick up my dry cleaning for the navy blue blazer."
    boots = (
        "I need to return some boots because they were too small. "
        "I exchanged them for a larger size, but I haven't had a chance "
        "to pick them up yet."
    )
    index = _write_index(
        tmp_path / "clothes.db",
        [
            ("store-a::blazer", blazer, asked - timedelta(days=1)),
            ("store-b::boots", boots, asked - timedelta(hours=2)),
        ],
    )
    question = (
        "[Question asked at 2023/02/15 (Wed) 23:50]\n"
        "How many items of clothing do I need to pick up or return from a store?"
    )
    local_input = _local_input(
        question,
        [
            _item("H001", blazer, created_at=asked - timedelta(days=1)),
            _item("H002", boots, created_at=asked - timedelta(hours=2)),
        ],
    )
    ticks = iter(range(0, 100, 10)).__next__

    row, timing = assay.assay_numeric_row(
        ordinal=69,
        question_id="q69",
        local_operator_input=local_input,
        index=index,
        clock=ticks,
    )

    assert row["applicable"] is True
    assert row["bridge"]["closed"] is True
    assert row["decision"]["status"] == "supported"
    assert row["decision"]["prediction"] == "3"
    assert row["admissible_replacement"] is True
    assert row["decision"]["policy_input_sha256"] == identity_sha256(local_input)
    assert timing == {
        "applicability_ns": 10,
        "bridge_ns": 10,
        "decision_ns": 10,
        "format": assay.TIMING_FORMAT,
        "ordinal": 69,
        "question_id": "q69",
        "result_row_receipt_sha256": row["row_receipt_sha256"],
        "specialist_ns": 10,
        "total_ns": 90,
    }


def test_inapplicable_gate_row_never_requires_or_scans_an_index() -> None:
    question = (
        "[Question asked at 2023/05/30 (Tue) 21:51]\n"
        "Where did I leave my keys?"
    )
    local_input = _local_input(question, [])
    ticks = iter((0, 10, 20, 30)).__next__

    row, timing = assay.assay_numeric_row(
        ordinal=1,
        question_id="q1",
        local_operator_input=local_input,
        index=None,
        clock=ticks,
    )

    assert row["applicable"] is False
    assert row["bridge"] is None
    assert row["decision"] is None
    assert row["admissible_replacement"] is False
    assert row["gold_loaded"] is False
    assert row["new_provider_calls"] == 0
    assert timing["specialist_ns"] == timing["bridge_ns"] == 0
    assert timing["decision_ns"] == 0
    assert timing["total_ns"] == 30


def test_source_clock_anchor_matches_full_store_relative_event_fact(
    tmp_path: Path,
) -> None:
    asked = datetime(2023, 5, 30, 21, 51, tzinfo=timezone.utc)
    source_clock = datetime(2023, 5, 20, 10, 0, tzinfo=timezone.utc)
    text = "I bought a peace lily two weeks ago."
    index = _write_index(
        tmp_path / "relative-plant.db",
        [("garden::peace", text, source_clock)],
    )
    question = (
        "[Question asked at 2023/05/30 (Tue) 21:51]\n"
        "How many plants did I acquire in the last month?"
    )
    item = _item("H001", text, created_at=asked)
    item.pop("date")
    item["temporal_anchor"] = source_clock.isoformat()

    row, _timing = assay.assay_numeric_row(
        ordinal=53,
        question_id="q53",
        local_operator_input=_local_input(question, [item]),
        index=index,
    )

    assert row["bridge"]["closed"] is True
    assert row["bridge"]["unresolved_candidate_keys"] == []
    assert row["decision"]["prediction"] == "1"
    assert row["admissible_replacement"] is True


def test_aggregate_reports_warm_latency_separately_from_gate_population() -> None:
    rows = [
        {"applicable": False, "admissible_replacement": False, "bridge": None},
        {
            "applicable": True,
            "admissible_replacement": True,
            "bridge": {"closed": True},
        },
        {
            "applicable": True,
            "admissible_replacement": False,
            "bridge": {"closed": False},
        },
    ]
    timings = [
        {"total_ns": 3, "bridge_ns": 0},
        {"total_ns": 10, "bridge_ns": 4},
        {"total_ns": 30, "bridge_ns": 20},
    ]

    result = assay._aggregate(rows, timings)  # noqa: SLF001

    assert result == {
        "admissible_replacement_count": 1,
        "applicable_count": 2,
        "closed_frontier_count": 1,
        "gate_selected_count": 3,
        "warm_applicable_mean_ns": 20.0,
        "warm_applicable_p95_ns": 30.0,
    }


def test_post_hoc_score_is_a_separate_gold_join(
    tmp_path: Path, monkeypatch
) -> None:
    question = (
        "[Question asked at 2023/02/15 (Wed) 23:50]\n"
        "How many items do I need to pick up or return?"
    )
    result_body = {
        "admissible_replacement": True,
        "applicable": True,
        "bridge": {"closed": True},
        "decision": {"prediction": "3", "status": "supported"},
        "format": assay.ROW_FORMAT,
        "gold_loaded": False,
        "local_operator_input_sha256": "a" * 64,
        "new_provider_calls": 0,
        "ordinal": 0,
        "policy_id": assay.POLICY_ID,
        "question_id": "q0",
        "question_sha256": quote_sha256(question),
        "retained_transformer_token_state_bytes": 0,
        "specialist_receipt_sha256": "b" * 64,
        "window_index_receipt_sha256": "c" * 64,
    }
    result = {
        **result_body,
        "row_receipt_sha256": identity_sha256(result_body),
    }
    output_root = tmp_path / "assay"
    construction_sha = assay.hot._atomic_write_json(  # noqa: SLF001
        output_root / assay.CONSTRUCTION_NAME,
        {
            "format": assay.FORMAT,
            "gold_loaded": False,
            "new_provider_calls": 0,
            "questions": [result],
        },
    )
    assay.hot._atomic_write_json(  # noqa: SLF001
        output_root / assay.RUNTIME_NAME,
        {
            "construction_sha256": construction_sha,
            "format": assay.RUNTIME_FORMAT,
            "gold_loaded": False,
            "new_provider_calls": 0,
        },
    )
    benchmark = SimpleNamespace(
        answer="3", question_id="q0", dated_question=question
    )
    monkeypatch.setattr(
        assay.full100,
        "_load_population",
        lambda *_args: (
            [object()],
            object(),
            {"population_identity_sha256": assay.typed.EXPECTED_POPULATION_SHA256},
        ),
    )
    monkeypatch.setattr(
        assay.full100, "_flatten_questions", lambda _samples: [benchmark]
    )
    v3_rows = [
        {
            "effective_hybrid": {
                "all_gold_source_ids_reached": True,
                "literal_answer": True,
            },
            "ordinal": ordinal,
        }
        for ordinal in range(100)
    ]
    v7_rows = [
        {
            "correct": False if ordinal == 0 else ordinal < 70,
            "ordinal": ordinal,
            "question_id": "q0" if ordinal == 0 else f"q{ordinal}",
            "reference_sha256": (
                quote_sha256("3") if ordinal == 0 else "d" * 64
            ),
        }
        for ordinal in range(100)
    ]

    def fake_read(path: Path, _expected: str, *, label: str):
        del path
        return (
            {"questions": v3_rows}
            if label == "hot-v3 score"
            else {"correct": 70, "rows": v7_rows}
        )

    monkeypatch.setattr(assay.typed, "_read_expected", fake_read)

    assay.score(
        dataset=tmp_path / "gold.json",
        split_manifest=tmp_path / "split.json",
        output_root=output_root,
        v3_score_path=tmp_path / "v3.json",
        v7_judgments_path=tmp_path / "v7.json",
    )
    scored, _digest = assay.hot._read_json_artifact(  # noqa: SLF001
        output_root / assay.SCORE_NAME
    )

    assert scored["gold_loaded"] is True
    assert scored["baseline_correct_count"] == 70
    assert scored["net_exact_gain"] == 1
    assert scored["effective_lexical_exact_correct_count"] == 71


def test_sealed_parent_reader_uses_fixed_bytes_not_legacy_loaders(
    tmp_path: Path, monkeypatch
) -> None:
    question = (
        "[Question asked at 2023/03/03 (Fri) 23:25]\n"
        "How many different museums or galleries did I visit in February?"
    )
    identity = {
        "local_ordinal": 0,
        "ordinal": 0,
        "probe_sha256": _sha("probe"),
        "prompt_question_sha256": quote_sha256(question),
        "question_id": "q0",
        "retrieval_query_sha256": _sha("retrieval"),
        "shard_offset": 0,
    }
    fallback_arm = {
        "provider_messages": [
            {"content": "sealed evidence", "role": "system"},
            {
                "content": f"Use memory.\n\nQuestion: {question}\nShort answer:",
                "role": "user",
            },
        ]
    }
    v7_bindings = {
        "parent_bindings": {"population_identity_sha256": assay.typed.EXPECTED_POPULATION_SHA256},
        "parent_output_relative_path": "unused",
        "parent_selection_sha256": _sha("parent-v6"),
        "population_identity_sha256": assay.typed.EXPECTED_POPULATION_SHA256,
    }
    v7_root = tmp_path / "v7"
    v7_body = {
        "bindings": v7_bindings,
        "format": assay.v7.SELECTION_FORMAT,
        "gold_fields_present": False,
        "provider_calls": 0,
        "questions": [{**identity, "arms": {"a3_protected_union": fallback_arm}}],
        "status": "sealed_gold_blind_locked_full100_adaptive_source_surplus_v1",
    }
    v7_sha = assay.hot._atomic_write_json(  # noqa: SLF001
        v7_root / assay.v7.SELECTION_NAME, v7_body
    )
    v2_bindings = {
        "compiled_catalog_sha256": _sha("catalog"),
        "population_identity_sha256": assay.typed.EXPECTED_POPULATION_SHA256,
        "probes_sha256": _sha("probes"),
        "source_root_relative_path": "primary-checkout:unused-source",
        "v7_bindings": copy.deepcopy(v7_bindings),
        "v7_output_relative_path": "current-worktree:unused-v7",
        "v7_selection_sha256": v7_sha,
    }
    v2_root = tmp_path / "v2"
    v2_implementation_sha = _sha("v2-implementation")
    v2_body = {
        "bindings": v2_bindings,
        "format": assay.v2.SELECTION_FORMAT,
        "gold_fields_present": False,
        "implementation": {"sha256": v2_implementation_sha},
        "provider_calls": 0,
        "questions": [copy.deepcopy(identity)],
        "status": "sealed_gold_blind_assertion_projection_full100_assay",
    }
    v2_sha = assay.hot._atomic_write_json(  # noqa: SLF001
        v2_root / assay.v2.SELECTION_NAME, v2_body
    )
    compact_row = {**identity, "provider_packet": {"sealed": True}}
    compact_bindings = {
        **{
            key: value
            for key, value in v2_bindings.items()
            if key
            in {
                "compiled_catalog_sha256",
                "population_identity_sha256",
                "probes_sha256",
                "source_root_relative_path",
                "v7_output_relative_path",
                "v7_selection_sha256",
            }
        },
        "v2_implementation_sha256": v2_implementation_sha,
        "v2_output_relative_path": "current-worktree:unused-v2",
        "v2_replay_sha256": assay.v3.EXPECTED_V2_REPLAY_SHA256,
        "v2_run_manifest_sha256": assay.v3.EXPECTED_V2_RUN_MANIFEST_SHA256,
        "v2_runtime_sha256": assay.v3.EXPECTED_V2_RUNTIME_SHA256,
        "v2_selection_sha256": v2_sha,
    }
    v3_root = tmp_path / "v3"
    v3_body = {
        "bindings": compact_bindings,
        "format": assay.v3.SELECTION_FORMAT,
        "gold_fields_present": False,
        "provider_calls": 0,
        "questions": [compact_row],
        "status": "sealed_gold_blind_source_seed_hybrid_full100",
    }
    v3_sha = assay.hot._atomic_write_json(  # noqa: SLF001
        v3_root / assay.v3.SELECTION_NAME, v3_body
    )
    monkeypatch.setattr(assay, "EXPECTED_QUESTION_COUNT", 1)
    monkeypatch.setattr(assay, "EXPECTED_V3_SELECTION_SHA256", v3_sha)
    monkeypatch.setattr(assay, "EXPECTED_V2_SELECTION_SHA256", v2_sha)
    monkeypatch.setattr(assay, "EXPECTED_V7_SELECTION_SHA256", v7_sha)

    roots = {
        "source root": tmp_path / "source",
        "v2 output root": v2_root,
        "v7 output root": v7_root,
        "v2-bound v7 output root": v7_root,
    }
    monkeypatch.setattr(
        assay,
        "_bound_repository_path",
        lambda _value, *, label: roots[label].resolve(),
    )
    for module in (assay.v3, assay.v2, assay.v7):
        monkeypatch.setattr(
            module,
            "_load_selection",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("legacy loader must not run")
            ),
            raising=False,
        )
    composed: list[int] = []

    def compose(*, v2_row, v7_row, dated_question, parent):
        assert parent.v2_selection == v2_body
        assert parent.v7_selection == v7_body
        assert v7_row["arms"]["a3_protected_union"] == fallback_arm
        assert dated_question == question
        composed.append(v2_row["ordinal"])
        return copy.deepcopy(compact_row), {}, copy.deepcopy(fallback_arm)

    monkeypatch.setattr(assay.v3, "_compose_question", compose)

    rows, digest = assay._load_sealed_v3_rows(  # noqa: SLF001
        v3_root, ordinals=(0,)
    )

    assert digest == v3_sha
    assert composed == [0]
    assert "arms" not in v3_body["questions"][0]
    assert rows[0]["arms"]["a3_protected_union"] == fallback_arm

    monkeypatch.setattr(
        assay.v3,
        "_compose_question",
        lambda **_kwargs: (
            {**compact_row, "question_id": "changed"},
            {},
            copy.deepcopy(fallback_arm),
        ),
    )
    with pytest.raises(ValueError, match="differs from sealed v3 bytes"):
        assay._load_sealed_v3_rows(v3_root, ordinals=(0,))  # noqa: SLF001


def test_fixed_selection_reader_rejects_an_unpinned_digest(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    assay.hot._atomic_write_json(  # noqa: SLF001
        path,
        {
            "format": "fixture",
            "gold_fields_present": False,
            "provider_calls": 0,
            "status": "sealed",
        },
    )

    with pytest.raises(ValueError, match="sealed fixture selection changed"):
        assay._read_fixed_selection(  # noqa: SLF001
            path,
            expected_sha256="0" * 64,
            expected_format="fixture",
            expected_status="sealed",
            label="fixture",
        )


def test_locked_rows_rejects_boolean_ordinals(monkeypatch) -> None:
    monkeypatch.setattr(assay, "EXPECTED_QUESTION_COUNT", 1)

    with pytest.raises(ValueError, match="ordinal order changed"):
        assay._locked_rows(  # noqa: SLF001
            {"questions": [{"ordinal": False, "question_id": "q0"}]},
            label="fixture",
        )


@pytest.mark.parametrize(
    "binding",
    ("current-worktree:../escape", "unknown-scope:artifact"),
)
def test_parent_binding_cannot_escape_a_checkout(binding: str) -> None:
    with pytest.raises(ValueError, match="escaped its repository checkout"):
        assay._bound_repository_path(binding, label="fixture")  # noqa: SLF001
