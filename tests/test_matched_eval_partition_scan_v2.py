from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.schemas import Chunk
from memory_condense.persistence.db import Database
from memory_condense.persistence.discourse_store import DiscourseStore
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.search.indexes.lexical import LexicalIndex
from tools.matched_eval.artifacts import publish_sealed_json
from tools.matched_eval.contracts import (
    ArtifactRef,
    EvaluationMemorySnapshot,
    EvidenceItem,
    MemoryPacket,
    identity_sha256,
)
from tools.matched_eval.partition_scan import construct_partition_scan_question
from tools.matched_eval.partition_scan_v2 import (
    PartitionScanV2Generation,
    PartitionScanV2MembershipAdapter,
    construct_partition_scan_v2_question,
    load_partition_scan_v2_generation,
    partition_scan_v2_arm_plan,
    partition_token_quotas,
)


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _packet(*, protected: tuple[EvidenceItem, ...] = ()) -> MemoryPacket:
    question = "Which note mentions the orchid launch needle?"
    dated = "[Question asked at 2026/08/27 (Thu) 10:00]\n" + question
    return MemoryPacket(
        question_id="question-prefix-is-not-a-partition",
        question_sha256=quote_sha256(question),
        dated_question=dated,
        dated_question_sha256=quote_sha256(dated),
        stage_id="causal_graph_coverage_predecessor",
        protected_evidence=protected,
    )


def _write_store(path: Path, rows: list[tuple[str, str]]) -> Path:
    db = Database(path)
    transcript = TranscriptStore(db)
    index = LexicalIndex(db)
    base = datetime(2026, 8, 20, tzinfo=timezone.utc)
    for number, (source_id, text) in enumerate(rows):
        turn = transcript.append(
            "user",
            text,
            source_id=source_id,
            created_at=base + timedelta(minutes=number),
        )
        index.add_chunks(
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
    db.close()
    return path


def _snapshot() -> EvaluationMemorySnapshot:
    return EvaluationMemorySnapshot(
        population_identity_sha256=_sha("population"),
        question_order_sha256=_sha("order"),
        source_artifacts=(ArtifactRef(role="sealed_retrieval", sha256=_sha("retrieval")),),
    )


def test_partition_token_quotas_are_rank_weighted_and_conserve_cap() -> None:
    assert partition_token_quotas(0) == ()
    assert partition_token_quotas(1) == (2_048,)
    assert partition_token_quotas(2) == (1_366, 682)
    assert partition_token_quotas(3) == (1_171, 585, 292)
    assert partition_token_quotas(4) == (1_024, 512, 256, 256)


def test_v2_partition_quota_recovers_source_starved_by_v1_global_fill(
    tmp_path: Path,
) -> None:
    padding = " ".join(f"filler{number}" for number in range(38))
    rows = [
        (f"alpha::source-{number:03d}", f"orchid launch needle {padding} alpha{number}")
        for number in range(55)
    ]
    rows.extend(
        [
            ("beta::wanted", f"orchid launch needle {padding} decisive"),
            ("gamma::one", "orchid launch needle gamma"),
            ("delta::one", "orchid launch needle delta"),
        ]
    )
    path = _write_store(tmp_path / "memory.db", rows)
    packet = _packet()
    with Database(path, read_only=True) as db:
        v1 = construct_partition_scan_question(
            db,
            ordinal=0,
            shard_offset=0,
            packet=packet,
            eligible=True,
            source_database_sha256=_sha("database"),
            source_store_receipt_sha256=_sha("store"),
        )
        v2 = construct_partition_scan_v2_question(
            db,
            ordinal=0,
            shard_offset=0,
            packet=packet,
            eligible=True,
            source_database_sha256=_sha("database"),
            source_store_receipt_sha256=_sha("store"),
        )

    v1_by_id = {row.evidence_id: row for row in v1.candidates}
    v2_by_id = {row.evidence_id: row for row in v2.candidates}
    assert "beta" in v2.selected_partitions
    assert not any(
        v1_by_id[value].source_id == "beta::wanted"
        for value in v1.trace.selected_before_dedup_ids
    )
    assert any(
        v2_by_id[value].source_id == "beta::wanted"
        for value in v2.trace.selected_before_dedup_ids
    )
    assert sum(row.selected_token_count for row in v2.partition_allocations) <= 2_048
    assert tuple(row.token_quota for row in v2.partition_allocations) == (
        1_024,
        512,
        256,
        256,
    )
    assert tuple(row.coverage_token_reserve for row in v2.partition_allocations) == (
        983,
        491,
        245,
        245,
    )


def test_v2_keeps_two_exact_query_centred_spans_and_dedups_after_selection(
    tmp_path: Path,
) -> None:
    first = "The orchid launch needle was cobalt blue."
    second = "A later orchid launch needle was stored beside the telescope."
    path = _write_store(
        tmp_path / "memory.db",
        [
            ("beta::wanted", first),
            ("beta::wanted", second),
            ("gamma::noise", "The weather was calm."),
        ],
    )
    protected = EvidenceItem(
        evidence_id="protected-first",
        source_id="beta::wanted",
        text=first,
        token_count=count_tokens(first),
    )
    with Database(path, read_only=True) as db:
        row = construct_partition_scan_v2_question(
            db,
            ordinal=0,
            shard_offset=0,
            packet=_packet(protected=(protected,)),
            eligible=True,
            source_database_sha256=_sha("database"),
            source_store_receipt_sha256=_sha("store"),
        )
        wanted = [
            candidate
            for candidate in row.candidates
            if candidate.source_id == "beta::wanted"
        ]
        assert [candidate.span_rank for candidate in wanted] == [0, 1]
        assert all(
            DiscourseStore(db).hydrate_span(candidate.span) == candidate.text
            for candidate in wanted
        )

    selected = set(row.trace.selected_before_dedup_ids)
    assert all(candidate.evidence_id in selected for candidate in wanted)
    assert len(row.trace.dedup_excluded_ids) == 1
    assert (row.trace.dedup_excluded_ids[0], protected.evidence_id) in row.dedup_alias_bindings
    assert len(
        [candidate for candidate in wanted if candidate.evidence_id in row.trace.admitted_ids]
    ) == 1


def test_v2_generation_round_trip_and_membership_adapter(tmp_path: Path) -> None:
    path = _write_store(
        tmp_path / "memory.db",
        [("beta::wanted", "The orchid launch needle is seven.")],
    )
    packet = _packet()
    with Database(path, read_only=True) as db:
        question = construct_partition_scan_v2_question(
            db,
            ordinal=0,
            shard_offset=0,
            packet=packet,
            eligible=True,
            source_database_sha256=_sha("database"),
            source_store_receipt_sha256=_sha("store"),
        )
    snapshot = _snapshot()
    eligibility_sha = _sha("eligibility")
    generation = PartitionScanV2Generation(
        retrieval_sha256=_sha("retrieval"),
        eligibility_manifest_sha256=eligibility_sha,
        population_identity_sha256=snapshot.population_identity_sha256,
        questions=(question,),
    )
    artifact, _ = publish_sealed_json(tmp_path / "generation.json", generation.projection())
    population = SimpleNamespace(
        retrieval_sha256=_sha("retrieval"),
        snapshot=snapshot,
        rows=(SimpleNamespace(packet=packet),),
    )
    loaded = load_partition_scan_v2_generation(
        str(artifact.path),
        expected_generation_sha256=artifact.sha256,
        population=population,
        expected_eligibility_manifest_sha256=eligibility_sha,
    )
    delta = PartitionScanV2MembershipAdapter(loaded).propose(
        snapshot=snapshot,
        packet=packet,
        stage=partition_scan_v2_arm_plan().stages[0],
    )

    assert tuple(row.evidence_id for row in delta.additions) == question.trace.admitted_ids
    assert delta.trace.provider_prompt_count == 0
    assert loaded.projection() == generation.projection()


def test_v2_ineligible_question_is_a_zero_scan_noop(tmp_path: Path) -> None:
    path = _write_store(
        tmp_path / "memory.db",
        [("beta::wanted", "The orchid launch needle is seven.")],
    )
    with Database(path, read_only=True) as db:
        row = construct_partition_scan_v2_question(
            db,
            ordinal=0,
            shard_offset=0,
            packet=_packet(),
            eligible=False,
            source_database_sha256=_sha("database"),
            source_store_receipt_sha256=_sha("store"),
        )
    assert row.selected_partitions == ()
    assert row.partition_allocations == ()
    assert row.trace.candidate_ids == ()
    assert row.trace.reason == "question_only_route_ineligible"
