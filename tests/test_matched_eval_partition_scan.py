from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

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
from tools.matched_eval.partition_scan import (
    PartitionScanGeneration,
    PartitionScanMembershipAdapter,
    construct_partition_scan_question,
    load_partition_scan_generation,
    partition_scan_arm_plan,
)


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _packet(*, protected: tuple[EvidenceItem, ...] = ()) -> MemoryPacket:
    question = "Which history says the orchid launch code is seven?"
    dated = "[Question asked at 2026/08/27 (Thu) 10:00]\n" + question
    return MemoryPacket(
        question_id="alpha-question",
        question_sha256=quote_sha256(question),
        dated_question=dated,
        dated_question_sha256=quote_sha256(dated),
        stage_id="causal_graph_coverage_predecessor",
        protected_evidence=protected,
    )


def _store(tmp_path: Path) -> Path:
    path = tmp_path / "memory.db"
    db = Database(path)
    transcript = TranscriptStore(db)
    index = LexicalIndex(db)
    rows = (
        ("alpha::noise", "The weather was ordinary and grey."),
        ("beta::wanted", "The orchid launch code is seven."),
        ("gamma::noise", "A ceramic bowl sits on the table."),
        ("delta::noise", "The train arrives before noon."),
        ("epsilon::noise", "A bicycle needs a replacement chain."),
    )
    for number, (source_id, text) in enumerate(rows):
        turn = transcript.append(
            "user",
            text,
            source_id=source_id,
            created_at=datetime(2026, 8, 27, number, tzinfo=timezone.utc),
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


def test_partition_scan_uses_semantic_partitions_not_question_id_prefix(
    tmp_path: Path,
) -> None:
    path = _store(tmp_path)
    packet = _packet()
    with Database(path, read_only=True) as db:
        row = construct_partition_scan_question(
            db,
            ordinal=0,
            shard_offset=0,
            packet=packet,
            eligible=True,
            source_database_sha256=_sha("database"),
            source_store_receipt_sha256=_sha("store"),
        )

    # The question ID starts with alpha, but the lexical evidence lives in the
    # beta partition.  Runtime question-ID prefix filtering would miss it.
    assert row.selected_partitions[0] == "beta"
    assert any(candidate.source_id == "beta::wanted" for candidate in row.candidates)
    assert row.scanned_row_count == 4
    assert row.scanned_source_count == 4
    assert row.trace.tokens_used <= 2_048
    wanted = next(candidate for candidate in row.candidates if candidate.source_id == "beta::wanted")
    with Database(path, read_only=True) as db:
        assert DiscourseStore(db).hydrate_span(wanted.span) == wanted.text


def test_partition_scan_selects_then_dedups_against_protected_s0(tmp_path: Path) -> None:
    path = _store(tmp_path)
    protected_text = "The orchid launch code is seven."
    protected = EvidenceItem(
        evidence_id="protected-beta",
        source_id="beta::wanted",
        text=protected_text,
        token_count=count_tokens(protected_text),
    )
    packet = _packet(protected=(protected,))
    with Database(path, read_only=True) as db:
        row = construct_partition_scan_question(
            db,
            ordinal=0,
            shard_offset=0,
            packet=packet,
            eligible=True,
            source_database_sha256=_sha("database"),
            source_store_receipt_sha256=_sha("store"),
        )

    wanted = next(candidate for candidate in row.candidates if candidate.source_id == "beta::wanted")
    assert wanted.evidence_id in row.trace.selected_before_dedup_ids
    assert wanted.evidence_id in row.trace.dedup_excluded_ids
    assert (wanted.evidence_id, protected.evidence_id) in row.dedup_alias_bindings
    assert wanted.evidence_id not in row.trace.admitted_ids


def test_generation_round_trip_and_membership_adapter(tmp_path: Path) -> None:
    path = _store(tmp_path)
    packet = _packet()
    with Database(path, read_only=True) as db:
        question = construct_partition_scan_question(
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
    generation = PartitionScanGeneration(
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
    loaded = load_partition_scan_generation(
        str(artifact.path),
        expected_generation_sha256=artifact.sha256,
        population=population,
        expected_eligibility_manifest_sha256=eligibility_sha,
    )
    delta = PartitionScanMembershipAdapter(loaded).propose(
        snapshot=snapshot,
        packet=packet,
        stage=partition_scan_arm_plan().stages[0],
    )

    assert tuple(row.evidence_id for row in delta.additions) == question.trace.admitted_ids
    assert delta.trace.provider_prompt_count == 0
    assert loaded.projection() == generation.projection()


def test_ineligible_question_is_a_zero_scan_noop(tmp_path: Path) -> None:
    path = _store(tmp_path)
    with Database(path, read_only=True) as db:
        row = construct_partition_scan_question(
            db,
            ordinal=0,
            shard_offset=0,
            packet=_packet(),
            eligible=False,
            source_database_sha256=_sha("database"),
            source_store_receipt_sha256=_sha("store"),
        )
    assert row.selected_partitions == ()
    assert row.scanned_row_count == 0
    assert row.trace.candidate_ids == ()
    assert row.trace.reason == "question_only_route_ineligible"
