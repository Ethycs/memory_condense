from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.application.discourse_sources import scan_discourse_source_chunks
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.domain.schemas import Chunk
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.search.indexes.lexical import LexicalIndex
from tests.test_matched_eval_query_map_source_gate_adapter import (
    _map_plan as _query_map_plan,
    _map_plane as _query_map_plane,
    _plan as _query_plan,
)
from tools.matched_eval.contracts import (
    ArtifactRef,
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
)
from tools.matched_eval.locked_source_gate_adapter import (
    DIRECT_STREAM_PROFILE_REPACK_V2,
    DIRECT_STREAM_PROFILE_V1,
    LockedSourceGateActivationInput,
    LockedSourceGateAdapterError,
    LockedSourceGatePins,
    LockedSourceHydrationInput,
    VerifiedLockedSourceGateRow,
    _direct_refs,
    _repack_direct_frontier,
    build_locked_source_gate_adapter,
    locked_activation_input_from_query_map_adapter,
    project_locked_lane_source_stream,
)
from tools.matched_eval.query_expansion import FrozenSourceNamespace
from tools.matched_eval.query_map_source_gate_adapter import adapt_query_map_solver_v2
from tools.matched_eval.source_gate_controller import (
    LaneSourceBudget,
    ObligationKind,
    QuestionObligation,
    SourceGatePolicy,
    start_source_gate,
)
from tools.matched_eval.source_history_fact_union import (
    DirectEvidenceRef,
    FactLane,
    hydrate_source_histories,
)


def _sha(value: str) -> str:
    return quote_sha256(value)


def _store(path: Path) -> tuple[FrozenSourceNamespace, str, str]:
    database = Database(path / "memory.db")
    transcript, lexical = TranscriptStore(database), LexicalIndex(database)
    for ordinal, source_id in enumerate(("shared", "direct-only", "partition-only", "guided-only")):
        text = f"Alpha stored {source_id} evidence in blue."
        turn = transcript.append(
            "user", text, source_id=source_id, turn_id=f"turn-{ordinal}",
            created_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        )
        lexical.add_chunks((Chunk(
            chunk_id=f"chunk-{ordinal}", turn_id=turn.turn_id, text=text,
            start_char=0, end_char=len(text), token_count=count_tokens(text),
        ),))
    streams = scan_discourse_source_chunks(database)
    database.close()
    (path / "hnsw_index.bin").write_bytes(b"sealed-test-index")
    namespace = FrozenSourceNamespace.from_source_streams(
        snapshot_id=_sha("snapshot"),
        combined_store_receipt_sha256=_sha("combined-store"),
        source_streams=streams,
    )
    return namespace, file_sha256(path / "memory.db"), file_sha256(path / "hnsw_index.bin")


def _policy() -> SourceGatePolicy:
    return SourceGatePolicy(
        "adapter-test-v1",
        (
            LaneSourceBudget(FactLane.DIRECT, 2, 2, 1),
            LaneSourceBudget(FactLane.PARTITION, 2, 2, 1),
            LaneSourceBudget(FactLane.GUIDED, 2, 2, 1),
        ),
        global_unique_source_cap=6,
        max_physical_map_calls=8,
        max_rounds=2,
    )


def _fixture(tmp_path: Path):
    namespace, database_sha, index_sha = _store(tmp_path)
    direct = project_locked_lane_source_stream(
        FactLane.DIRECT, ("shared", "shared", "direct-only"),
        row_receipt=_sha("direct-row"), selected_ids=("d0", "d1", "d2"),
        artifact_sha256=_sha("direct-artifact"),
    )
    partition = project_locked_lane_source_stream(
        FactLane.PARTITION, ("partition-only", "shared"),
        row_receipt=_sha("partition-row"), selected_ids=("p0", "p1"),
        artifact_sha256=_sha("partition-artifact"),
    )
    guided = project_locked_lane_source_stream(
        FactLane.GUIDED, ("shared", "guided-only"),
        row_receipt=_sha("guided-row"), selected_ids=("g0", "g1"),
        artifact_sha256=_sha("guided-artifact"),
    )
    question = "What color did Alpha store?"
    direct_ref = DirectEvidenceRef(
        "root-evidence", namespace.namespace_id, "shared", _sha("root quote"),
        _sha("root-evidence-receipt"),
    )
    row = VerifiedLockedSourceGateRow(
        0, "question-0", _sha(question), question, _sha(question),
        _sha("source-packet"), _sha("population"), _sha("question-order"),
        namespace.snapshot_id, namespace, (direct_ref,), (direct, partition, guided),
        tmp_path, database_sha, index_sha,
    )
    obligation = QuestionObligation(
        ObligationKind.SUPPORT, ("alpha", "blue"), 2,
    )
    activation = LockedSourceGateActivationInput(
        "question-0", row.source_packet_id, _sha("map-packet"), 11, _sha("upstream-plan"),
        _sha("upstream-frontier"), (obligation.obligation_id,), (obligation,),
    )
    artifacts = (
        ArtifactRef("direct_query_run", _sha("direct-artifact"), "direct.json"),
        ArtifactRef("partition_r96_generation", _sha("partition-artifact"), "partition.json"),
        ArtifactRef("guided_run", _sha("guided-artifact"), "guided.json"),
    )
    return row, activation, artifacts


def test_adapter_preserves_method_ranks_then_reuses_one_physical_source(tmp_path: Path) -> None:
    row, activation, artifacts = _fixture(tmp_path)
    population = build_locked_source_gate_adapter(
        (row,), (activation,), source_artifacts=artifacts, policy=_policy(),
    )
    question = population.questions[0]
    candidates = question.plan.candidates
    assert tuple((item.lane, item.source_id, item.rank) for item in candidates) == (
        (FactLane.DIRECT, "shared", 0),
        (FactLane.DIRECT, "direct-only", 1),
        (FactLane.PARTITION, "partition-only", 0),
        (FactLane.PARTITION, "shared", 1),
        (FactLane.GUIDED, "shared", 0),
        (FactLane.GUIDED, "guided-only", 1),
    )
    memberships = {item.source_id: item for item in row.namespace.sources}
    assert all(
        item.membership_projection_sha256 == identity_sha256(memberships[item.source_id].projection())
        and item.stream_sha256 == memberships[item.source_id].stream_sha256
        for item in candidates
    )
    assert question.plan.activation.receipt_sha256
    assert question.source_packet_id == row.source_packet_id
    assert question.plan.parent.parent_packet_id == activation.map_packet_id
    assert question.plan.activation.parent_packet_id == activation.map_packet_id
    assert question.source_packet_id != question.plan.parent.parent_packet_id
    assert question.activation_input_receipt_sha256 == activation.receipt_sha256
    assert population.receipt_sha256
    assert question.plan.activation.unresolved_obligation_ids == (activation.unresolved_obligations[0].obligation_id,)
    assert question.plan.eligible_frontier.exhaustive is False
    assert_gold_blind(question.plan.projection(), path="adapter_test.plan")

    round_plan = start_source_gate(question.plan)
    assert len(round_plan.selections) == 6  # logical lane credit is retained
    hydration = question.hydration_input(round_plan)
    assert tuple(item.source_id for item in hydration.memberships) == (
        "shared", "direct-only", "partition-only", "guided-only",
    )  # one physical hydration per namespaced source
    database = hydration.open_read_only_database()
    try:
        histories = hydrate_source_histories(
            database, hydration.memberships, namespace_id=hydration.namespace_id,
            revalidate_store_bytes=hydration.revalidate_store_bytes,
        )
    finally:
        database.close()
    assert tuple(item.source_id for item in histories) == tuple(item.source_id for item in hydration.memberships)


def test_lane_projection_collapses_only_after_span_selection() -> None:
    stream = project_locked_lane_source_stream(
        FactLane.DIRECT, ("s2", "s1", "s2", "s3", "s1"),
        row_receipt=_sha("row"), selected_ids=("e0", "e1", "e2", "e3", "e4"),
        artifact_sha256=_sha("artifact"),
    )
    assert stream.source_ids == ("s2", "s1", "s3")
    assert stream.receipt_sha256 == project_locked_lane_source_stream(
        FactLane.DIRECT, ("s2", "s1", "s2", "s3", "s1"),
        row_receipt=_sha("row"), selected_ids=("e0", "e1", "e2", "e3", "e4"),
        artifact_sha256=_sha("artifact"),
    ).receipt_sha256


def test_repack_direct_profile_defaults_are_exact_and_opt_in() -> None:
    pins = LockedSourceGatePins()

    assert pins.query_repack_run_sha256 == (
        "960c8192ff8b97b599f37ac067f79036f4403bd8dfb8cb8532c13b309dea7c47"
    )
    assert pins.query_repack_runtime_sha256 == (
        "99d4df790f80b95da521fe1ffd5eddb7d7c041f082fc34a386977ee7db9cedd3"
    )
    assert DIRECT_STREAM_PROFILE_V1 != DIRECT_STREAM_PROFILE_REPACK_V2


def test_locked_direct_refs_bind_exact_exposed_text_without_projecting_it(
    tmp_path: Path,
) -> None:
    namespace, _database_sha, _index_sha = _store(tmp_path)
    evidence = SimpleNamespace(
        evidence_id="direct-blue",
        source_id="shared",
        text="Alpha stored shared evidence in blue.",
    )
    direct_row = SimpleNamespace(
        source=SimpleNamespace(
            packet=SimpleNamespace(protected_evidence=(evidence,))
        ),
        admitted_delta=(),
    )

    (direct,) = _direct_refs(namespace, direct_row)

    assert direct.text == evidence.text
    assert direct.quote_sha256 == quote_sha256(evidence.text)
    assert direct.projection()["exposed_text_sha256"] == quote_sha256(evidence.text)
    assert evidence.text not in str(direct.projection())


def test_adapter_fails_closed_on_missing_membership_and_store_tamper(tmp_path: Path) -> None:
    row, activation, artifacts = _fixture(tmp_path)
    escaped = project_locked_lane_source_stream(
        FactLane.GUIDED, ("not-in-namespace",), row_receipt=_sha("escaped-row"),
        selected_ids=("escaped",), artifact_sha256=_sha("guided-artifact"),
    )
    with pytest.raises(LockedSourceGateAdapterError, match="lacks sealed namespaced membership"):
        replace(row, lane_streams=row.lane_streams[:2] + (escaped,))

    question = build_locked_source_gate_adapter(
        (row,), (activation,), source_artifacts=artifacts, policy=_policy(),
    ).questions[0]
    hydration = question.hydration_input(start_source_gate(question.plan))
    assert isinstance(hydration, LockedSourceHydrationInput)
    (tmp_path / "hnsw_index.bin").write_bytes(b"tampered")
    with pytest.raises(LockedSourceGateAdapterError, match="store index changed"):
        hydration.revalidate_store_bytes()


def test_adapter_requires_a_gold_blind_unresolved_activation(tmp_path: Path) -> None:
    row, activation, artifacts = _fixture(tmp_path)
    with pytest.raises(LockedSourceGateAdapterError, match="only built for unresolved obligations"):
        replace(activation, unresolved_obligations=())
    with pytest.raises(MatchedEvalContractError, match="SHA-256"):
        replace(activation, obligation_ids=("gold-answer",))
    with pytest.raises(LockedSourceGateAdapterError, match="source packet escaped"):
        build_locked_source_gate_adapter(
            (row,), (replace(activation, source_packet_id=_sha("wrong-parent")),),
            source_artifacts=artifacts, policy=_policy(),
        )
    population = build_locked_source_gate_adapter(
        (row,), (activation,), source_artifacts=artifacts, policy=_policy(),
    )
    assert population.receipt_sha256 == build_locked_source_gate_adapter(
        (row,), (activation,), source_artifacts=artifacts, policy=_policy(),
    ).receipt_sha256


def test_direct_stream_profiles_are_explicit_and_receipt_distinct(
    tmp_path: Path,
) -> None:
    row, activation, artifacts = _fixture(tmp_path)
    v1 = build_locked_source_gate_adapter(
        (row,),
        (activation,),
        source_artifacts=artifacts,
        policy=_policy(),
    )
    repack = build_locked_source_gate_adapter(
        (row,),
        (activation,),
        source_artifacts=artifacts,
        policy=_policy(),
        direct_stream_profile=DIRECT_STREAM_PROFILE_REPACK_V2,
    )

    assert v1.direct_stream_profile == DIRECT_STREAM_PROFILE_V1
    assert repack.direct_stream_profile == DIRECT_STREAM_PROFILE_REPACK_V2
    assert (
        v1.direct_stream_profile_receipt_sha256
        != repack.direct_stream_profile_receipt_sha256
    )
    assert v1.receipt_sha256 != repack.receipt_sha256


def test_repack_direct_frontier_reads_selected_ids_before_source_collapse() -> None:
    namespace_id = _sha("namespace")
    packet = SimpleNamespace(
        question_id="question-0",
        question_sha256=_sha("question"),
        dated_question_sha256=_sha("dated-question"),
        packet_id=_sha("source-packet"),
    )
    prompt = SimpleNamespace(
        source=SimpleNamespace(packet=packet),
        namespace=SimpleNamespace(namespace_id=namespace_id),
    )
    candidate_ids = (_sha("c0"), _sha("c1"), _sha("c2"))
    selected_ids = (candidate_ids[1], candidate_ids[0], candidate_ids[2])
    parent_receipt = _sha("parent-row")
    unsigned = {
        "candidate_ids": list(candidate_ids),
        "candidate_metadata": [
            {
                "candidate_id": candidate_ids[0],
                "namespace_id": namespace_id,
                "source_id": "source-a",
            },
            {
                "candidate_id": candidate_ids[1],
                "namespace_id": namespace_id,
                "source_id": "source-b",
            },
            {
                "candidate_id": candidate_ids[2],
                "namespace_id": namespace_id,
                "source_id": "source-b",
            },
        ],
        "dated_question_sha256": packet.dated_question_sha256,
        "format": "memory-condense-query-expansion-repack-row-v2",
        "namespace_id": namespace_id,
        "ordinal": 0,
        "parent_packet_id": packet.packet_id,
        "parent_row_receipt_sha256": parent_receipt,
        "provider_calls": 0,
        "question_id": packet.question_id,
        "question_sha256": packet.question_sha256,
        "retrieval_rerun": False,
        "retained_transformer_token_state_bytes": 0,
        "selected_before_dedup_candidate_ids": list(selected_ids),
        "source_membership_coverage": {
            "repack_selected_source_count": 2,
            "repack_selected_source_ids": ["source-b", "source-a"],
        },
    }
    raw = {**unsigned, "receipt_sha256": identity_sha256(unsigned)}

    observed_ids, observed_sources, row_receipt = _repack_direct_frontier(
        raw,
        ordinal=0,
        prompt=prompt,
        parent_row_receipt_sha256=parent_receipt,
    )

    assert observed_ids == selected_ids
    assert observed_sources == ("source-b", "source-a", "source-b")
    assert row_receipt == identity_sha256(unsigned)

    tampered = dict(raw)
    tampered["selected_before_dedup_candidate_ids"] = [_sha("escaped")]
    tampered_unsigned = dict(tampered)
    tampered_unsigned.pop("receipt_sha256")
    tampered["receipt_sha256"] = identity_sha256(tampered_unsigned)
    with pytest.raises(
        LockedSourceGateAdapterError,
        match="escaped sealed candidate IDs",
    ):
        _repack_direct_frontier(
            tampered,
            ordinal=0,
            prompt=prompt,
            parent_row_receipt_sha256=parent_receipt,
        )


def test_post_map_activation_joins_locked_sources_without_collapsing_packets(
    tmp_path: Path,
) -> None:
    query_run, map_plan = _query_map_plan(tmp_path / "query", _query_plan())
    adapted = adapt_query_map_solver_v2(
        query_run,
        map_plan,
        _query_map_plane(map_plan, ()),
    ).rows[0]
    assert adapted.activation is not None
    activation = locked_activation_input_from_query_map_adapter(
        adapted,
        as_of_turn=11,
    )

    store_root = tmp_path / "store"
    store_root.mkdir()
    locked_row, _old_activation, artifacts = _fixture(store_root)
    packet = map_plan.rows[0].direct_plan_row.adapter.source.packet
    locked_row = replace(
        locked_row,
        question_id=packet.question_id,
        question_sha256=packet.question_sha256,
        dated_question=packet.dated_question,
        dated_question_sha256=packet.dated_question_sha256,
        source_packet_id=packet.packet_id,
    )
    population = build_locked_source_gate_adapter(
        (locked_row,),
        (activation,),
        source_artifacts=artifacts,
        policy=_policy(),
    )
    question = population.questions[0]

    assert activation.source_packet_id == packet.packet_id
    assert activation.map_packet_id == map_plan.rows[0].packet_id
    assert activation.source_packet_id != activation.map_packet_id
    assert question.packet_binding_projection() == {
        "activation_input_receipt_sha256": activation.receipt_sha256,
        "map_packet_id": map_plan.rows[0].packet_id,
        "source_packet_id": packet.packet_id,
    }
    assert question.plan.parent.parent_packet_id == map_plan.rows[0].packet_id
    assert question.plan.activation.parent_packet_id == map_plan.rows[0].packet_id
