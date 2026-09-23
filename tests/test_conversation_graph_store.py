"""Durability and lifecycle contracts for the post-index T1g worker."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.application.ingest_workflow import (
    _IDLE_GRAPH_RETRY_MAX_TURNS,
)
from memory_condense.domain.schemas import Chunk
from memory_condense.persistence.conversation_graph_store import (
    GRAPH_BOOTSTRAP_DEFAULT_MAX_TURNS,
    GRAPH_BOOTSTRAP_HARD_MAX_TURNS,
    ConversationGraphStore,
    EMPTY_GRAPH_CHECKPOINT_SHA256,
)
from memory_condense.search.incremental_conversation_graph import (
    StoryAffinityIndexPolicy,
)


class _Embedder:
    dim = 8

    def embed_query(self, _query: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        vector[0] = 1.0
        return vector

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        vector = self.embed_query("").tolist()
        return [chunk.model_copy(update={"embedding": vector}) for chunk in chunks]


def _condenser(path) -> MemoryCondenser:
    return MemoryCondenser(
        data_dir=path,
        embedder=_Embedder(),
        auto_extract=False,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    )


def _ingest_status(condenser: MemoryCondenser, turn_id: str) -> str:
    return str(
        condenser._db.execute(
            "SELECT status FROM pending_ingests WHERE turn_id = ?",
            (turn_id,),
        ).fetchone()[0]
    )


def test_t0_claim_is_atomic_but_does_not_compile_graph_evidence(tmp_path) -> None:
    with _condenser(tmp_path / "t0") as condenser:
        turn, chunks = condenser.capture(
            "user",
            "Cedar-Clinic signed the Northwind Labs contract.",
            turn_id="captured-turn",
        )

        job = condenser._db.execute(
            "SELECT g.status, g.ingest_manifest_sha256, p.manifest_sha256, "
            "p.status FROM pending_graph_compilations AS g "
            "JOIN pending_ingests AS p ON p.turn_id = g.turn_id "
            "WHERE g.artifact_id = ? AND g.turn_id = ?",
            (condenser._conversation_graphs.artifact_id, turn.turn_id),
        ).fetchone()
        state = condenser._db.execute(
            "SELECT revision, chunk_count, occurrence_count, "
            "checkpoint_sha256 FROM conversation_graph_state "
            "WHERE artifact_id = ?",
            (condenser._conversation_graphs.artifact_id,),
        ).fetchone()

        assert chunks
        assert job[0] == "pending"
        assert job[1] == job[2]
        assert job[3] == "pending"
        assert state == (0, 0, 0, EMPTY_GRAPH_CHECKPOINT_SHA256)
        assert condenser._db.execute(
            "SELECT COUNT(*) FROM conversation_graph_chunks"
        ).fetchone() == (0,)
        assert condenser._db.execute(
            "SELECT COUNT(*) FROM conversation_phrase_occurrences"
        ).fetchone() == (0,)

        # A graph worker cannot move ahead of ordinary searchability.
        assert condenser.drain_pending_graph_compilations() == []
        assert condenser._conversation_graphs.attempt_count(turn.turn_id) == 0


def test_single_ingest_publishes_hashed_delta_after_t1(tmp_path) -> None:
    with _condenser(tmp_path / "ready") as condenser:
        turn, chunks = condenser.ingest(
            "user",
            "Cedar-Clinic signed the Northwind Labs contract.",
            source_id="session-17",
            turn_id="ready-turn",
        )

        assert chunks
        assert _ingest_status(condenser, turn.turn_id) == "indexed"
        assert condenser._conversation_graphs.status(turn.turn_id) == "ready"
        receipt = condenser._conversation_graphs._terminal_receipt(turn.turn_id)
        assert receipt is not None
        assert receipt.chunk_count == len(chunks)
        assert receipt.occurrence_count > 0
        assert receipt.first_revision == 1
        assert receipt.last_revision == len(chunks)
        assert receipt.checkpoint_sha256 != EMPTY_GRAPH_CHECKPOINT_SHA256

        row = condenser._db.execute(
            "SELECT delta_sha256, parent_checkpoint_sha256, checkpoint_sha256, "
            "text_sha256, runtime_receipt_sha256 "
            "FROM conversation_graph_chunks WHERE artifact_id = ?",
            (condenser._conversation_graphs.artifact_id,),
        ).fetchone()
        assert row[1] == EMPTY_GRAPH_CHECKPOINT_SHA256
        assert all(len(str(value)) == 64 for value in row)
        assert "text" not in {
            str(value[1])
            for value in condenser._db.execute(
                "PRAGMA table_info(conversation_graph_chunks)"
            ).fetchall()
        }

        graph = condenser.conversation_graph()
        assert graph.stats().chunk_count == len(chunks)
        occurrence = graph.occurrences("cedar clinic")[0]
        assert occurrence.chunk_id == chunks[0].chunk_id
        assert occurrence.quote == "Cedar-Clinic"


def test_graph_failure_cannot_roll_back_searchable_t1_and_retries_after_restart(
    tmp_path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "fail-open"
    condenser = _condenser(data_dir)
    original_prepare = condenser._conversation_graphs._prepare_turn

    def fail_prepare(_turn_id: str):
        raise RuntimeError("synthetic graph compiler failure")

    monkeypatch.setattr(
        condenser._conversation_graphs,
        "_prepare_turn",
        fail_prepare,
    )
    turn, chunks = condenser.ingest(
        "user",
        "The durable saffron witness remains searchable.",
        turn_id="fail-open-turn",
    )

    assert chunks
    assert _ingest_status(condenser, turn.turn_id) == "indexed"
    assert condenser._conversation_graphs.status(turn.turn_id) == "pending"
    assert condenser._conversation_graphs.attempt_count(turn.turn_id) == 1
    assert condenser.search_hybrid("saffron witness", k=1)[0].chunk.chunk_id == (
        chunks[0].chunk_id
    )
    assert condenser._db.execute(
        "SELECT COUNT(*) FROM conversation_graph_chunks"
    ).fetchone() == (0,)
    monkeypatch.setattr(
        condenser._conversation_graphs,
        "_prepare_turn",
        original_prepare,
    )
    condenser.close()

    # The pending receipt, not process memory, owns retry after a restart.
    with _condenser(data_dir) as restarted:
        completed = restarted.drain_pending_graph_compilations(max_turns=1)
        assert len(completed) == 1
        assert completed[0].created is True
        assert completed[0].receipt.status == "ready"
        assert restarted._conversation_graphs.attempt_count(turn.turn_id) == 1
        assert restarted.conversation_graph().chunk(chunks[0].chunk_id) is not None


def test_restart_hydrates_persisted_occurrences_without_reextracting_corpus(
    tmp_path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "restart"
    with _condenser(data_dir) as condenser:
        _turn, chunks = condenser.ingest(
            "user",
            "Project Juniper approved the cobalt telescope.",
            turn_id="restart-turn",
        )
        expected_chunk_id = chunks[0].chunk_id
        expected_revision = condenser.conversation_graph().stats().revision

    def forbidden_extract(*_args, **_kwargs):
        raise AssertionError("restart must consume persisted phrase deltas")

    monkeypatch.setattr(
        "memory_condense.search.incremental_conversation_graph."
        "extract_canonical_phrases",
        forbidden_extract,
    )
    with _condenser(data_dir) as restarted:
        graph = restarted.conversation_graph()
        assert graph.stats().revision == expected_revision
        assert graph.chunk(expected_chunk_id) is not None
        assert graph.occurrences("project juniper")[0].chunk_id == expected_chunk_id
        assert "juniper" in graph.story_source_terms("restart-turn")
        assert graph.story_term_sources("juniper") == ("restart-turn",)
        assert graph.stats().story_term_membership_count > 0
        assert graph.stats().story_evidence_chunk_count == 1


def test_suffix_fixed_point_turn_compiles_and_restores_after_restart(tmp_path) -> None:
    data_dir = tmp_path / "suffix-fixed-point"
    with _condenser(data_dir) as condenser:
        turn, chunks = condenser.ingest(
            "user",
            "Cedar Archive closed after the gates were raised.",
            source_id="closed-session",
            turn_id="closed-turn",
        )

        assert condenser._conversation_graphs.status(turn.turn_id) == "ready"
        assert condenser._conversation_graphs.attempt_count(turn.turn_id) == 0
        assert chunks
        assert {"closed", "raised"} <= set(
            condenser.conversation_graph().story_source_terms("closed-session")
        )

    with _condenser(data_dir) as restarted:
        graph = restarted.conversation_graph()
        assert graph.chunk(chunks[0].chunk_id) is not None
        assert {"closed", "raised"} <= set(
            graph.story_source_terms("closed-session")
        )


def test_loaded_resident_adopts_only_newer_persisted_delta(
    tmp_path,
    monkeypatch,
) -> None:
    with _condenser(tmp_path / "delta-sync") as condenser:
        _first_turn, first_chunks = condenser.ingest(
            "user",
            "Project Juniper opened Cedar Observatory.",
            source_id="session-a",
            turn_id="delta-first",
        )
        graph = condenser.conversation_graph()
        assert graph.stats().revision == 1

        _second_turn, second_chunks = condenser.ingest(
            "user",
            "Northwind Labs repaired the cobalt telescope.",
            source_id="session-b",
            turn_id="delta-second",
        )

        def forbidden_extract(*_args, **_kwargs):
            raise AssertionError("resident sync must hydrate only persisted deltas")

        monkeypatch.setattr(
            "memory_condense.search.incremental_conversation_graph."
            "extract_canonical_phrases",
            forbidden_extract,
        )
        synced = condenser.conversation_graph()

        assert synced is graph
        assert synced.stats().revision == 2
        assert synced.chunk(first_chunks[0].chunk_id) is not None
        assert synced.chunk(second_chunks[0].chunk_id) is not None


def test_resident_sync_pins_target_across_concurrent_advance(
    tmp_path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "pinned-sync"
    with _condenser(data_dir) as producer:
        _turn, first_chunks = producer.ingest(
            "user",
            "Project Juniper opened Cedar Observatory.",
            source_id="session-a",
            turn_id="pinned-first",
        )

    writer = _condenser(data_dir)
    reader = _condenser(data_dir)
    second_chunks: list[Chunk] = []
    try:
        resident = reader.conversation_graph()
        assert resident.stats().revision == 1
        original_execute = reader._db.execute
        advanced = False

        class _AdvanceAfterEmptyPage:
            def __init__(self, cursor) -> None:
                self._cursor = cursor

            def fetchall(self):
                nonlocal advanced
                rows = self._cursor.fetchall()
                if not rows and not advanced:
                    advanced = True
                    _turn, chunks = writer.ingest(
                        "user",
                        "Northwind Labs repaired the cobalt telescope.",
                        source_id="session-b",
                        turn_id="pinned-second",
                    )
                    second_chunks.extend(chunks)
                return rows

        def execute_with_advance(sql: str, params: tuple = ()):
            cursor = original_execute(sql, params)
            if (
                "FROM conversation_graph_chunks AS g " in sql
                and "g.append_revision <= ?" in sql
            ):
                return _AdvanceAfterEmptyPage(cursor)
            return cursor

        monkeypatch.setattr(reader._db, "execute", execute_with_advance)
        pinned = reader.conversation_graph()

        assert advanced is True
        assert pinned is resident
        assert pinned.stats().revision == 1
        assert pinned.chunk(second_chunks[0].chunk_id) is None

        caught_up = reader.conversation_graph()
        assert caught_up is resident
        assert caught_up.stats().revision == 2
        assert caught_up.chunk(first_chunks[0].chunk_id) is not None
        assert caught_up.chunk(second_chunks[0].chunk_id) is not None
    finally:
        # The writer closes last so its current ordinary ANN image wins over
        # the reader's intentionally pinned pre-advance image.
        reader.close()
        writer.close()


def test_idempotent_worker_retry_does_not_republish_deltas(tmp_path) -> None:
    with _condenser(tmp_path / "idempotent") as condenser:
        turn, _chunks = condenser.ingest(
            "assistant",
            "Northwind Labs approved Project Juniper.",
            turn_id="idempotent-turn",
        )
        before = condenser._db.execute(
            "SELECT revision, chunk_count, occurrence_count, checkpoint_sha256 "
            "FROM conversation_graph_state WHERE artifact_id = ?",
            (condenser._conversation_graphs.artifact_id,),
        ).fetchone()

        assert condenser.drain_pending_graph_compilations(
            turn_ids=(turn.turn_id,)
        ) == []
        after = condenser._db.execute(
            "SELECT revision, chunk_count, occurrence_count, checkpoint_sha256 "
            "FROM conversation_graph_state WHERE artifact_id = ?",
            (condenser._conversation_graphs.artifact_id,),
        ).fetchone()
        assert after == before


def test_idle_t1_drain_retries_an_indexed_graph_backlog(tmp_path, monkeypatch) -> None:
    with _condenser(tmp_path / "idle-retry") as condenser:
        original_prepare = condenser._conversation_graphs._prepare_turn

        def fail_prepare(_turn_id: str):
            raise RuntimeError("one graph failure")

        monkeypatch.setattr(
            condenser._conversation_graphs,
            "_prepare_turn",
            fail_prepare,
        )
        turn, _chunks = condenser.ingest(
            "user",
            "The violet archive references Cedar Observatory.",
            turn_id="idle-retry-turn",
        )
        monkeypatch.setattr(
            condenser._conversation_graphs,
            "_prepare_turn",
            original_prepare,
        )

        assert condenser.pending_ingest_count() == 0
        assert condenser._conversation_graphs.status(turn.turn_id) == "pending"
        assert condenser.drain_pending_ingests(max_manifests=1) == []
        assert condenser._conversation_graphs.status(turn.turn_id) == "ready"


def test_idle_t1_drain_uses_a_finite_default_graph_retry_bound(
    tmp_path,
    monkeypatch,
) -> None:
    with _condenser(tmp_path / "idle-default-bound") as condenser:
        observed: list[tuple[int | None, object]] = []

        def record_graph_drain(*, max_turns, turn_ids=None):
            observed.append((max_turns, turn_ids))
            return []

        monkeypatch.setattr(
            condenser,
            "_drain_pending_graph_fail_open",
            record_graph_drain,
        )

        assert condenser.drain_pending_ingests() == []
        assert condenser.drain_pending_ingests(max_manifests=5) == []
        assert observed == [
            (_IDLE_GRAPH_RETRY_MAX_TURNS, None),
            (5, None),
        ]


def test_conversation_graph_state_cannot_be_deleted(tmp_path) -> None:
    with _condenser(tmp_path / "durable-state") as condenser:
        condenser.ingest(
            "user",
            "Cedar Observatory retained its cobalt ledger.",
            turn_id="durable-state-turn",
        )

        with pytest.raises(sqlite3.IntegrityError, match="state is durable"):
            condenser._db.execute(
                "DELETE FROM conversation_graph_state WHERE artifact_id = ?",
                (condenser._conversation_graphs.artifact_id,),
            )
        condenser._db.connection.rollback()

        assert condenser.conversation_graph().stats().revision == 1


def test_explicit_bootstrap_is_resumable_idempotent_and_policy_scoped(
    tmp_path,
    monkeypatch,
) -> None:
    with _condenser(tmp_path / "bootstrap-policy") as condenser:
        ingested = condenser.ingest_many(
            [
                (
                    "user",
                    f"Project {name} retained marker {position}.",
                    f"session-{position}",
                    None,
                    f"bootstrap-{position}",
                )
                for position, name in enumerate(
                    ("Juniper", "Cedar", "Northwind"),
                    start=1,
                )
            ]
        )
        turn_ids = tuple(turn.turn_id for turn, _chunks in ingested)
        current_artifact = condenser._conversation_graphs.artifact_id

        first = condenser.bootstrap_conversation_graph(max_turns=2)
        second = condenser.bootstrap_conversation_graph(max_turns=2)
        replay = condenser.bootstrap_conversation_graph(max_turns=2)

        assert first.artifact_id == current_artifact
        assert first.selected_turn_ids == turn_ids[:2]
        assert first.claimed_turn_ids == ()
        assert first.completed_turn_ids == turn_ids[:2]
        assert first.pending_turn_ids == ()
        assert first.remaining_turn_count == 1
        assert first.unsupported_indexed_turn_count == 0
        assert second.selected_turn_ids == turn_ids[2:]
        assert second.completed_turn_ids == turn_ids[2:]
        assert second.remaining_turn_count == 0
        assert replay.selected_turn_ids == ()
        assert condenser.conversation_graph().stats().revision == 3

        changed_store = ConversationGraphStore(
            condenser._db,
            story_index_policy=StoryAffinityIndexPolicy(
                max_sources_per_term=63,
            ),
        )
        condenser._conversation_graphs = changed_store
        assert changed_store.artifact_id != current_artifact

        from memory_condense.search import incremental_conversation_graph

        original_extract = incremental_conversation_graph.extract_canonical_phrases

        def fail_extract(*_args, **_kwargs):
            raise RuntimeError("synthetic post-claim extraction failure")

        monkeypatch.setattr(
            incremental_conversation_graph,
            "extract_canonical_phrases",
            fail_extract,
        )
        failed = condenser.bootstrap_conversation_graph(max_turns=2)
        assert failed.selected_turn_ids == turn_ids[:2]
        assert failed.claimed_turn_ids == turn_ids[:2]
        assert failed.completed_turn_ids == ()
        assert failed.pending_turn_ids == turn_ids[:2]
        assert failed.remaining_turn_count == 3

        monkeypatch.setattr(
            incremental_conversation_graph,
            "extract_canonical_phrases",
            original_extract,
        )
        resumed = condenser.bootstrap_conversation_graph(max_turns=2)
        tail = condenser.bootstrap_conversation_graph(max_turns=2)
        changed_replay = condenser.bootstrap_conversation_graph(max_turns=2)

        assert resumed.selected_turn_ids == turn_ids[:2]
        assert resumed.claimed_turn_ids == ()
        assert resumed.completed_turn_ids == turn_ids[:2]
        assert resumed.remaining_turn_count == 1
        assert tail.selected_turn_ids == turn_ids[2:]
        assert tail.claimed_turn_ids == turn_ids[2:]
        assert tail.completed_turn_ids == turn_ids[2:]
        assert tail.remaining_turn_count == 0
        assert changed_replay.selected_turn_ids == ()
        assert condenser.conversation_graph().stats().revision == 3


def test_bootstrap_default_limit_is_finite(tmp_path) -> None:
    with _condenser(tmp_path / "bootstrap-default-limit") as condenser:
        ingested = condenser.ingest_many(
            [
                (
                    "user",
                    f"Bounded bootstrap evidence marker {position}.",
                    f"bounded-session-{position}",
                    None,
                    f"bounded-turn-{position:02d}",
                )
                for position in range(GRAPH_BOOTSTRAP_DEFAULT_MAX_TURNS + 1)
            ]
        )
        turn_ids = tuple(turn.turn_id for turn, _chunks in ingested)

        result = condenser.bootstrap_conversation_graph()

        assert result.selected_turn_ids == turn_ids[
            :GRAPH_BOOTSTRAP_DEFAULT_MAX_TURNS
        ]
        assert len(result.completed_turn_ids) == GRAPH_BOOTSTRAP_DEFAULT_MAX_TURNS
        assert result.remaining_turn_count == 1


def test_bootstrap_reports_indexed_turn_without_manifest_as_unsupported(
    tmp_path,
) -> None:
    with _condenser(tmp_path / "bootstrap-unsupported") as condenser:
        legacy = condenser._transcript.append(
            "user",
            "Legacy cobalt evidence has no replay manifest.",
            turn_id="legacy-without-manifest",
        )
        source_chunks = condenser._chunker.chunk_turn(
            legacy.turn_id,
            legacy.text,
        )
        indexed_chunks = condenser._embedder.embed_chunks(source_chunks)
        condenser._retriever.add_chunks(indexed_chunks)

        result = condenser.bootstrap_conversation_graph()

        assert result.selected_turn_ids == ()
        assert result.claimed_turn_ids == ()
        assert result.completed_turn_ids == ()
        assert result.remaining_turn_count == 0
        assert result.unsupported_indexed_turn_count == 1
        assert condenser._conversation_graphs.status(legacy.turn_id) is None
        assert condenser.conversation_graph().stats().revision == 0


@pytest.mark.parametrize(
    "value",
    [0, -1, True, 1.5, GRAPH_BOOTSTRAP_HARD_MAX_TURNS + 1],
)
def test_bootstrap_rejects_unbounded_or_invalid_page_sizes(tmp_path, value) -> None:
    with _condenser(tmp_path / f"bootstrap-bound-{value}") as condenser:
        with pytest.raises(ValueError, match="integer from 1 through"):
            condenser.bootstrap_conversation_graph(max_turns=value)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_graph_worker_rejects_invalid_bounds(tmp_path, value) -> None:
    with _condenser(tmp_path / f"bound-{value}") as condenser:
        with pytest.raises(ValueError, match="positive integer"):
            condenser.drain_pending_graph_compilations(max_turns=value)
