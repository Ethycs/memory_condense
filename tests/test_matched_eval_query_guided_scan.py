from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.application.discourse_sources import scan_discourse_source_chunks
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.schemas import Chunk
from memory_condense.persistence.db import Database
from memory_condense.persistence.discourse_store import DiscourseStore
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.search.indexes.lexical import LexicalIndex

from tools import run_locked_query_guided_scan as scan_cli
from tools.matched_eval.contracts import EvidenceItem, MemoryPacket, identity_sha256
from tools.matched_eval.query_expansion import (
    FrozenSourceMembership,
    FrozenSourceNamespace,
    PartitionRoutingReceipt,
)
from tools.matched_eval.query_guided_scan import (
    QueryGuidedScanBudget,
    QueryGuidedScanError,
    _construct_row,
    aggregate_partition_votes,
    cache_namespace_partitions,
    score_query_guided_candidates,
    select_balanced_candidates,
)


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _namespace() -> FrozenSourceNamespace:
    return FrozenSourceNamespace(
        snapshot_id=_sha("snapshot"),
        combined_store_receipt_sha256=_sha("store"),
        sources=tuple(
            FrozenSourceMembership(
                source_id=f"p{index}::source",
                content_chunk_ids=(f"content-{index}",),
                metadata_chunk_ids=(f"metadata-{index}",),
                stream_sha256=_sha(f"stream-{index}"),
            )
            for index in range(6)
        ),
    )


def test_partition_vote_aggregates_parent_ranks_and_fills_exact_six() -> None:
    namespace = _namespace()
    queries = ("orchid launch", "blue needle")
    receipts = (
        PartitionRoutingReceipt.create(
            query=queries[0],
            namespace=namespace,
            selected_partitions=("p1", "p2", "p3", "p4"),
            routed_source_count=6,
        ),
        PartitionRoutingReceipt.create(
            query=queries[1],
            namespace=namespace,
            selected_partitions=("p2", "p1", "p5", "p0"),
            routed_source_count=6,
        ),
    )

    plan = aggregate_partition_votes(
        queries,
        tuple(row.projection() for row in receipts),
        namespace=namespace,
    )

    assert plan.selected_partitions == ("p1", "p2", "p3", "p5", "p4", "p0")
    assert plan.ranking[0].vote_score == 7
    assert plan.ranking[1].vote_score == 7
    assert plan.parent_receipt_sha256s == tuple(
        row.receipt_sha256 for row in receipts
    )


def test_partition_vote_rejects_query_receipt_mismatch() -> None:
    namespace = _namespace()
    receipt = PartitionRoutingReceipt.create(
        query="sealed query",
        namespace=namespace,
        selected_partitions=("p0", "p1", "p2", "p3"),
        routed_source_count=6,
    )
    with pytest.raises(QueryGuidedScanError, match="query or namespace"):
        aggregate_partition_votes(
            ("different query",), (receipt.projection(),), namespace=namespace
        )


def _write_store(path: Path) -> tuple[Path, FrozenSourceNamespace]:
    db = Database(path)
    transcript = TranscriptStore(db)
    lexical = LexicalIndex(db)
    base = datetime(2026, 8, 20, tzinfo=timezone.utc)
    rows = [
        ("p0::wanted", "The orchid launch needle was cobalt blue."),
        ("p0::wanted", "Later the orchid needle moved beside the telescope."),
        ("p1::noise", "A ceramic bowl was placed on the desk."),
        ("p2::noise", "The city train arrived before noon."),
        ("p3::noise", "A bicycle needed a replacement chain."),
        ("p4::noise", "The weather was calm and ordinary."),
        ("p5::noise", "A bedside lamp used a warm bulb."),
    ]
    for index, (source_id, text) in enumerate(rows):
        turn = transcript.append(
            "user",
            text,
            source_id=source_id,
            created_at=base + timedelta(minutes=index),
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
    streams = scan_discourse_source_chunks(db)
    db.close()
    namespace = FrozenSourceNamespace.from_source_streams(
        snapshot_id=_sha("snapshot"),
        combined_store_receipt_sha256=_sha("store"),
        source_streams=streams,
    )
    return path, namespace


def test_store_is_cached_once_and_candidates_keep_hydratable_exact_spans(
    tmp_path: Path,
) -> None:
    path, namespace = _write_store(tmp_path / "memory.db")
    with Database(path, read_only=True) as database:
        cache = cache_namespace_partitions(
            database,
            namespace,
            source_database_sha256=_sha("database"),
            source_store_receipt_sha256=_sha("store"),
        )
        candidates = score_query_guided_candidates(
            cache,
            selected_partitions=namespace.partition_ids,
            query_surfaces=(
                "orchid launch needle",
                "[Question asked at 2026/08/27]\nWhat color was the orchid needle?",
            ),
        )
        store = DiscourseStore(database)
        assert all(store.hydrate_span(row.span) == row.text for row in candidates)

    wanted = [row for row in candidates if row.source_id == "p0::wanted"]
    assert cache.projection()["database_read_passes"] == 1
    assert cache.content_row_count == 7
    assert len(wanted) == 2
    assert [row.span_rank for row in wanted] == [0, 1]
    assert wanted[0].overlap_term_count > 0


def test_selection_balances_sources_then_allows_second_spans(tmp_path: Path) -> None:
    path, namespace = _write_store(tmp_path / "memory.db")
    with Database(path, read_only=True) as database:
        cache = cache_namespace_partitions(
            database,
            namespace,
            source_database_sha256=_sha("database"),
            source_store_receipt_sha256=_sha("store"),
        )
    candidates = score_query_guided_candidates(
        cache,
        selected_partitions=namespace.partition_ids,
        query_surfaces=("orchid needle",),
    )
    selection = select_balanced_candidates(candidates)
    by_id = {row.evidence_id: row for row in candidates}

    assert selection.selected_token_count <= 2_400
    assert {by_id[value].partition_id for value in selection.selected_ids} == set(
        namespace.partition_ids
    )
    assert any(by_id[value].span_rank == 1 for value in selection.selected_ids)


def test_exact_s0_dedup_occurs_after_query_guided_selection(tmp_path: Path) -> None:
    path, namespace = _write_store(tmp_path / "memory.db")
    with Database(path, read_only=True) as database:
        cache = cache_namespace_partitions(
            database,
            namespace,
            source_database_sha256=_sha("database"),
            source_store_receipt_sha256=_sha("store"),
        )
    question = "What color was the orchid launch needle?"
    dated = "[Question asked at 2026/08/27 (Thu) 10:00]\n" + question
    exact = "The orchid launch needle was cobalt blue."
    protected = EvidenceItem(
        evidence_id="protected-exact",
        source_id="p0::wanted",
        text=exact,
        token_count=count_tokens(exact),
    )
    packet = MemoryPacket(
        question_id="question-id-is-provenance-only",
        question_sha256=quote_sha256(question),
        dated_question=dated,
        dated_question_sha256=quote_sha256(dated),
        stage_id="causal_graph_coverage_predecessor",
        protected_evidence=(protected,),
    )
    query = "orchid launch needle color"
    route = PartitionRoutingReceipt.create(
        query=query,
        namespace=namespace,
        selected_partitions=namespace.partition_ids[:4],
        routed_source_count=len(namespace.sources),
    )
    prompt = SimpleNamespace(
        source=SimpleNamespace(ordinal=0, packet=packet), namespace=namespace
    )
    raw_parent = {
        "materialized_queries": [query],
        "receipt_sha256": _sha("parent-row"),
        "routing_receipts": [route.projection()],
    }

    row = _construct_row(
        prompt, raw_parent, cache, budget=QueryGuidedScanBudget()
    )

    assert row["dedup_timing"] == "after_bounded_selection"
    assert row["dedup_excluded_candidate_ids"]
    excluded = row["dedup_excluded_candidate_ids"][0]
    assert excluded in row["selected_before_dedup_candidate_ids"]
    assert [excluded, protected.evidence_id] in row["dedup_alias_bindings"]
    assert excluded not in row["admitted_candidate_ids"]


def test_locked_runner_exposes_no_provider_or_retrieval_rerun_switches() -> None:
    materialize = scan_cli._parser().parse_args(["materialize"])
    replay = scan_cli._parser().parse_args(
        ["replay", "--expected-run-sha256", "b" * 64]
    )
    for args in (materialize, replay):
        assert not any(
            hasattr(args, value)
            for value in (
                "enable_provider",
                "authorized_provider_calls",
                "api_key_env",
                "policy",
                "qwen_prefix",
                "device",
            )
        )


def test_summary_reports_provider_free_cache_and_population_counts() -> None:
    artifact = SimpleNamespace(
        path=SimpleNamespace(as_posix=lambda: "run.json"),
        sha256="c" * 64,
        payload={
            "aggregate": {
                "admitted_candidate_count": 3,
                "candidate_count": 9,
                "dedup_excluded_candidate_count": 1,
                "logical_scanned_content_row_memberships": 20,
                "maximum_tokens_used": 22,
                "selected_candidate_count": 4,
                "selected_second_span_count": 1,
                "total_tokens_used": 22,
            },
            "physical_database_read_passes": 1,
            "questions": [{}],
        },
    )
    ledger = SimpleNamespace(
        path=SimpleNamespace(as_posix=lambda: "ledger.json"), sha256="d" * 64
    )
    result = SimpleNamespace(run_artifact=artifact, runtime_ledger_artifact=ledger)

    summary = scan_cli._summary(
        result, command="materialize", elapsed_seconds=1.23456
    )

    assert summary["new_provider_calls"] == 0
    assert summary["physical_database_read_passes"] == 1
    assert summary["retained_transformer_token_state_bytes"] == 0
    assert summary["elapsed_seconds"] == 1.235
