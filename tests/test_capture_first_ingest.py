"""Focused coverage for capture-first and bounded pending-ingest drains."""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.application.ingest_workflow import (
    DeferredCorrectionTargetStaleError,
)
from memory_condense.domain.schemas import (
    Chunk,
    CreateOp,
    DeleteOp,
    MemoryOps,
    MemoryStatus,
    MemoryType,
    Provenance,
    UpdateOp,
)
from memory_condense.ingest.extractor import (
    ExtractionUnavailableError,
    LLMExtractor,
)
from memory_condense.persistence.db import CURRENT_SCHEMA_VERSION


class RecordingEmbedder:
    """Deterministic provider double with observable batch boundaries."""

    dim = 8

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.before_embed = None

    def embed_query(self, _query: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        vector[0] = 1.0
        return vector

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        if self.before_embed is not None:
            self.before_embed(chunks)
        self.calls.append(tuple(chunk.turn_id for chunk in chunks))
        vector = self.embed_query("").tolist()
        return [chunk.model_copy(update={"embedding": vector}) for chunk in chunks]


class FailingEmbedder(RecordingEmbedder):
    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        raise RuntimeError("synthetic provider outage")


def _condenser(tmp_path, embedder: RecordingEmbedder) -> MemoryCondenser:
    return MemoryCondenser(
        data_dir=tmp_path,
        embedder=embedder,
        auto_extract=False,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    )


def test_capture_publishes_turn_and_manifest_without_provider_or_index(tmp_path):
    embedder = RecordingEmbedder()
    with _condenser(tmp_path / "capture", embedder) as condenser:
        before_meta = dict(
            condenser._db.execute(
                "SELECT key, value FROM meta WHERE key IN "
                "('chunk_index_revision', 'next_hnsw_label')"
            ).fetchall()
        )
        turn, chunks = condenser.capture(
            "user",
            "The capture-first amber fact is durable.",
            turn_id="captured-turn",
        )

        assert turn.turn_id == "captured-turn"
        assert chunks and all(chunk.embedding is None for chunk in chunks)
        assert embedder.calls == []
        assert condenser.transcript.count() == 1
        assert condenser.pending_ingest_count() == 1
        assert condenser._pending_ingests.get(turn.turn_id) is not None
        assert condenser._db.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone() == (0,)
        assert dict(
            condenser._db.execute(
                "SELECT key, value FROM meta WHERE key IN "
                "('chunk_index_revision', 'next_hnsw_label')"
            ).fetchall()
        ) == before_meta
        assert condenser.retriever._index.get_current_count() == 0


def test_capture_many_prefetches_receipts_with_set_queries(tmp_path):
    embedder = RecordingEmbedder()
    records = [
        (
            "user",
            f"Set-oriented capture value {index}.",
            "source",
            None,
            f"set-capture-{index:03d}",
        )
        for index in range(100)
    ]
    with _condenser(tmp_path / "set-capture", embedder) as condenser:
        statements: list[str] = []
        condenser._db.connection.set_trace_callback(statements.append)
        try:
            condenser.capture_many(records)
        finally:
            condenser._db.connection.set_trace_callback(None)

        receipt_reads = [
            statement
            for statement in statements
            if re.match(r"^\s*SELECT\b", statement, re.IGNORECASE)
            and "FROM pending_ingests" in statement
        ]
        # One prefetch plus the bounded claim/re-read checks; never one lookup
        # per explicit turn ID.
        assert len(receipt_reads) <= 4
        assert condenser.pending_ingest_count() == len(records)


def test_indexed_capture_replay_checks_presence_without_loading_vectors(tmp_path):
    embedder = RecordingEmbedder()
    records = [
        (
            "user",
            f"Indexed replay value {index}.",
            "source",
            None,
            f"indexed-replay-{index:03d}",
        )
        for index in range(32)
    ]
    with _condenser(tmp_path / "indexed-replay", embedder) as condenser:
        condenser.ingest_many(records)
        statements: list[str] = []
        condenser._db.connection.set_trace_callback(statements.append)
        try:
            replayed = condenser.capture_many(records)
        finally:
            condenser._db.connection.set_trace_callback(None)

        chunk_reads = [
            " ".join(statement.upper().split())
            for statement in statements
            if re.match(r"^\s*SELECT\b", statement, re.IGNORECASE)
            and "FROM CHUNKS" in statement.upper()
        ]
        assert len(replayed) == len(records)
        assert chunk_reads
        assert all("EMBEDDING IS NOT NULL" in query for query in chunk_reads)
        assert all(
            "TOKEN_COUNT, EMBEDDING, HNSW_LABEL" not in query
            for query in chunk_reads
        )


def test_provider_failure_survives_restart_with_exact_generated_ids(tmp_path):
    data_dir = tmp_path / "provider-restart"
    with _condenser(data_dir, FailingEmbedder()) as interrupted:
        with pytest.raises(RuntimeError, match="synthetic provider outage"):
            interrupted.ingest("user", "The generated ochre turn survives.")
        manifest = interrupted._pending_ingests.list_pending()[0]
        expected_ids = [chunk.chunk_id for chunk in manifest.chunks]
        assert expected_ids

    with _condenser(data_dir, RecordingEmbedder()) as recovered:
        rows = recovered.drain_pending_ingests()

        assert [chunk.chunk_id for chunk in rows[0][1]] == expected_ids
        assert recovered.pending_ingest_count() == 0


def test_generated_capture_retry_adopts_receipt_across_chunker_change(tmp_path):
    data_dir = tmp_path / "generated-retry"
    source_embedder = RecordingEmbedder()
    text = "Alpha one two three. Beta four five six. Gamma seven eight nine."
    with MemoryCondenser(
        data_dir=data_dir,
        embedder=source_embedder,
        auto_extract=False,
        chunker_min_tokens=1,
        chunker_max_tokens=5,
    ) as captured:
        turn, source_chunks = captured.capture("user", text)
        source_topology = [
            (chunk.chunk_id, chunk.start_char, chunk.end_char)
            for chunk in source_chunks
        ]
        assert len(source_topology) > 1

    retry_embedder = RecordingEmbedder()
    with MemoryCondenser(
        data_dir=data_dir,
        embedder=retry_embedder,
        auto_extract=False,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as retried:
        with pytest.raises(ValueError, match="different content"):
            retried.ingest("user", text + " Changed.", turn_id=turn.turn_id)
        assert retry_embedder.calls == []
        assert retried.pending_ingest_count() == 1

        completed_turn, completed_chunks = retried.ingest(
            "user",
            text,
            turn_id=turn.turn_id,
        )

        assert completed_turn == turn
        assert [
            (chunk.chunk_id, chunk.start_char, chunk.end_char)
            for chunk in completed_chunks
        ] == source_topology
        assert all(chunk.embedding is not None for chunk in completed_chunks)
        assert len(retry_embedder.calls) == 1
        assert retried.pending_ingest_count() == 0


def test_synchronous_ingest_embeds_only_after_capture_is_durable(tmp_path):
    embedder = RecordingEmbedder()
    data_dir = tmp_path / "sync-order"
    with _condenser(data_dir, embedder) as condenser:
        observed: list[tuple[int, int, int]] = []

        def observe_capture(_chunks):
            assert not condenser._db.connection.in_transaction
            with sqlite3.connect(data_dir / "memory.db") as independent:
                observed.append(
                    (
                        independent.execute(
                            "SELECT COUNT(*) FROM turns"
                        ).fetchone()[0],
                        independent.execute(
                            "SELECT COUNT(*) FROM pending_ingests "
                            "WHERE status = 'pending'"
                        ).fetchone()[0],
                        independent.execute(
                            "SELECT COUNT(*) FROM chunks"
                        ).fetchone()[0],
                    )
                )

        embedder.before_embed = observe_capture
        completed = condenser.ingest_many(
            [
                ("user", "First durable turn.", "source", None, "turn-1"),
                ("assistant", "Second durable turn.", "source", None, "turn-2"),
            ]
        )

        assert observed == [(2, 2, 0)]
        assert len(embedder.calls) == 1
        assert all(
            chunk.embedding is not None
            for _turn, chunks in completed
            for chunk in chunks
        )
        assert condenser.pending_ingest_count() == 0


def test_synchronous_ingest_does_not_drain_unrelated_backlog(tmp_path):
    embedder = RecordingEmbedder()
    with _condenser(tmp_path / "owned-drain", embedder) as condenser:
        condenser.capture(
            "user",
            "The older violet capture stays queued.",
            turn_id="backlog-turn",
        )

        turn, chunks = condenser.ingest(
            "assistant",
            "The new green capture completes synchronously.",
            turn_id="synchronous-turn",
        )

        assert turn.turn_id == "synchronous-turn"
        assert chunks and all(chunk.embedding is not None for chunk in chunks)
        assert embedder.calls == [("synchronous-turn",)]
        assert condenser.pending_ingest_count() == 1
        assert [
            manifest.turn_id
            for manifest in condenser._pending_ingests.list_pending()
        ] == ["backlog-turn"]


def test_exact_indexed_retry_reuses_durable_vectors_without_reembedding(tmp_path):
    embedder = RecordingEmbedder()
    with _condenser(tmp_path / "indexed-retry", embedder) as condenser:
        _turn, original = condenser.ingest(
            "user",
            "The durable coral vector is already searchable.",
            turn_id="indexed-retry-turn",
        )
        assert len(embedder.calls) == 1

        _retried_turn, retried = condenser.ingest(
            "user",
            "The durable coral vector is already searchable.",
            turn_id="indexed-retry-turn",
        )

        assert len(embedder.calls) == 1
        assert [chunk.chunk_id for chunk in retried] == [
            chunk.chunk_id for chunk in original
        ]
        assert [chunk.embedding for chunk in retried] == [
            chunk.embedding for chunk in original
        ]


def test_bounded_drain_batches_whole_manifests_and_guarantees_progress(tmp_path):
    embedder = RecordingEmbedder()
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    records = [
        (
            "user",
            f"Pending turn {index} has several tokens to embed.",
            "source",
            base_time + timedelta(seconds=index),
            f"pending-{index}",
        )
        for index in range(4)
    ]
    with _condenser(tmp_path / "bounded-drain", embedder) as condenser:
        captured = condenser.capture_many(records)
        assert condenser.pending_ingest_count() == 4
        assert embedder.calls == []
        stats = condenser.pending_ingest_stats()
        assert stats["manifest_count"] == 4
        assert stats["chunk_count"] == sum(len(chunks) for _, chunks in captured)
        assert stats["token_count"] == sum(
            chunk.token_count for _turn, chunks in captured for chunk in chunks
        )
        assert isinstance(stats["oldest_age_seconds"], float)

        first = condenser.drain_pending_ingests(max_manifests=2)
        assert [turn.turn_id for turn, _chunks in first] == [
            "pending-0",
            "pending-1",
        ]
        assert condenser.pending_ingest_count() == 2
        assert len(embedder.calls) == 1
        assert len(embedder.calls[0]) == sum(len(chunks) for _, chunks in first)

        # The next manifest is larger than both work bounds, but admitting it
        # whole prevents a permanently blocked queue head.
        second = condenser.drain_pending_ingests(max_chunks=1, max_tokens=1)
        assert [turn.turn_id for turn, _chunks in second] == ["pending-2"]
        assert (
            sum(
                chunk.token_count
                for _turn, chunks in second
                for chunk in chunks
            )
            > 1
        )
        assert condenser.pending_ingest_count() == 1

        final = condenser.recover_pending_ingests(max_tokens=1)
        assert [turn.turn_id for turn, _chunks in final] == ["pending-3"]
        assert condenser.pending_ingest_count() == 0
        assert len(embedder.calls) == 3
        assert condenser.pending_ingest_stats() == {
            "manifest_count": 0,
            "chunk_count": 0,
            "token_count": 0,
            "oldest_age_seconds": None,
            "failed_count": 0,
            "oldest_error_kind": None,
        }
        assert all(
            chunk.embedding is None
            for _turn, chunks in captured
            for chunk in chunks
        )


def test_bounded_drain_fifo_uses_turn_ordinal_when_receipt_times_tie(tmp_path):
    embedder = RecordingEmbedder()
    records = [
        ("user", f"Ordered value {turn_id}.", "source", None, turn_id)
        for turn_id in ("turn-z", "turn-a", "turn-m")
    ]
    with _condenser(tmp_path / "ordinal-fifo", embedder) as condenser:
        condenser.capture_many(records)

        first = condenser.drain_pending_ingests(max_manifests=1)

        # One batched claim deliberately gives all three receipts the same
        # creation timestamp. FIFO is transcript ordinal, never lexical ID.
        assert [turn.turn_id for turn, _chunks in first] == ["turn-z"]


def test_chunk_bound_admits_one_oversized_multichunk_manifest(tmp_path):
    embedder = RecordingEmbedder()
    with MemoryCondenser(
        data_dir=tmp_path / "multichunk-overflow",
        embedder=embedder,
        auto_extract=False,
        chunker_min_tokens=1,
        chunker_max_tokens=5,
    ) as condenser:
        _turn, source_chunks = condenser.capture(
            "user",
            "Alpha one two three. Beta four five six. Gamma seven eight nine.",
            turn_id="multichunk-turn",
        )
        assert len(source_chunks) > 1

        drained = condenser.drain_pending_ingests(max_chunks=1)

        assert len(drained) == 1
        assert len(drained[0][1]) == len(source_chunks)
        assert condenser.pending_ingest_count() == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_manifests": 0},
        {"max_chunks": -1},
        {"max_tokens": True},
    ],
)
def test_drain_rejects_nonpositive_or_boolean_bounds(tmp_path, kwargs):
    embedder = RecordingEmbedder()
    with _condenser(tmp_path / "invalid-bound", embedder) as condenser:
        with pytest.raises(ValueError, match="positive integer"):
            condenser.drain_pending_ingests(**kwargs)


def test_capture_many_rejects_conflicts_before_any_publication(tmp_path):
    embedder = RecordingEmbedder()
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    with _condenser(tmp_path / "capture-conflict", embedder) as condenser:
        with pytest.raises(ValueError, match="duplicate turn_id"):
            condenser.capture_many(
                [
                    ("user", "First body.", "source", timestamp, "same-turn"),
                    ("user", "Conflicting body.", "source", timestamp, "same-turn"),
                ]
            )

        assert condenser.transcript.count() == 0
        assert condenser.pending_ingest_count() == 0
        assert embedder.calls == []


def test_drain_preserves_auto_extract_semantics(tmp_path):
    embedder = RecordingEmbedder()
    with MemoryCondenser(
        data_dir=tmp_path / "auto-extract-drain",
        embedder=embedder,
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.capture(
            "user",
            "I prefer dark mode in every application.",
            turn_id="preference-turn",
        )

        drained = condenser.drain_pending_ingests()

        assert [turn.turn_id for turn, _chunks in drained] == ["preference-turn"]
        assert condenser.pending_ingest_count() == 0
        assert len(condenser.memory.list_items()) == 1


def test_deferred_t1_drain_leaves_t2_recoverable_without_reembedding(tmp_path):
    embedder = RecordingEmbedder()
    with MemoryCondenser(
        data_dir=tmp_path / "deferred-enrichment",
        embedder=embedder,
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.capture(
            "user",
            "I prefer dark mode in every application.",
            turn_id="deferred-preference",
        )

        indexed = condenser.drain_pending_ingests(enrich=False)

        assert [turn.turn_id for turn, _chunks in indexed] == [
            "deferred-preference"
        ]
        assert condenser.pending_ingest_count() == 0
        assert condenser.pending_enrichment_stats()["turn_count"] == 1
        assert condenser.pending_enrichment_stats()["ready_count"] == 1
        assert condenser.memory.list_items() == []
        embedding_calls = list(embedder.calls)

        enriched = condenser.drain_pending_enrichments(max_turns=1)

        assert [turn.turn_id for turn, _chunks in enriched] == [
            "deferred-preference"
        ]
        assert embedder.calls == embedding_calls
        assert condenser.pending_enrichment_stats() == {
            "turn_count": 0,
            "ready_count": 0,
            "oldest_age_seconds": None,
            "failed_count": 0,
            "oldest_error_kind": None,
            "discarded_legacy_count": 0,
            "deferred_correction_count": 0,
            "legacy_quarantined_count": 0,
        }
        assert len(condenser.memory.list_items()) == 1


def test_legacy_recover_alias_never_runs_automatic_extraction(tmp_path):
    with MemoryCondenser(
        data_dir=tmp_path / "legacy-recover",
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.capture(
            "user",
            "I prefer the recover alias to remain base-only.",
            turn_id="legacy-recover-turn",
        )

        recovered = condenser.recover_pending_ingests()

        assert [turn.turn_id for turn, _chunks in recovered] == [
            "legacy-recover-turn"
        ]
        assert condenser.pending_ingest_count() == 0
        assert condenser.pending_enrichment_count() == 1
        assert condenser.memory.list_items() == []


def test_deferred_enrichment_seals_without_reading_retired_evidence(
    tmp_path, monkeypatch
):
    embedder = RecordingEmbedder()
    with MemoryCondenser(
        data_dir=tmp_path / "retired-enrichment",
        embedder=embedder,
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.capture(
            "user",
            "I prefer the retired ultraviolet setting.",
            turn_id="retired-enrichment-turn",
        )
        indexed = condenser.drain_pending_ingests(enrich=False)
        assert len(indexed) == 1 and len(indexed[0][1]) == 1
        assert condenser.retriever.delete_chunk(indexed[0][1][0].chunk_id)

        def reject_extraction(_turns, _chunks=None):
            raise AssertionError("retired evidence reached the extractor")

        monkeypatch.setattr(condenser._extractor, "extract", reject_extraction)
        enriched = condenser.drain_pending_enrichments()

        assert enriched == [(indexed[0][0], [])]
        assert condenser.pending_enrichment_count() == 0
        assert condenser.memory.list_items() == []


def test_exact_ingest_retry_cannot_enrich_retired_evidence(tmp_path, monkeypatch):
    embedder = RecordingEmbedder()
    text = "I prefer the retired silver setting."
    with MemoryCondenser(
        data_dir=tmp_path / "retired-ingest-retry",
        embedder=embedder,
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.capture("user", text, turn_id="retired-retry-turn")
        indexed = condenser.drain_pending_ingests(enrich=False)
        assert condenser.retriever.delete_chunk(indexed[0][1][0].chunk_id)
        embedding_calls = list(embedder.calls)

        def reject_extraction(_turns, _chunks=None):
            raise AssertionError("retired retry evidence reached the extractor")

        monkeypatch.setattr(condenser._extractor, "extract", reject_extraction)
        _turn, retried = condenser.ingest(
            "user", text, turn_id="retired-retry-turn"
        )

        assert embedder.calls == embedding_calls
        assert len(retried) == 1 and retried[0].embedding is None
        assert condenser.pending_enrichment_count() == 0
        assert condenser.memory.list_items() == []


def test_deferred_enrichment_limits_partial_turn_text_to_live_chunks(
    tmp_path, monkeypatch
):
    with MemoryCondenser(
        data_dir=tmp_path / "partially-retired-enrichment",
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=6,
    ) as condenser:
        condenser.capture(
            "user",
            "I prefer red tea. I prefer blue coffee.",
            turn_id="partially-retired-turn",
        )
        indexed = condenser.drain_pending_ingests(enrich=False)
        source_chunks = indexed[0][1]
        assert len(source_chunks) > 1
        retired = source_chunks[0]
        assert condenser.retriever.delete_chunk(retired.chunk_id)
        observed: list[tuple[str, list[str]]] = []

        original_extract = condenser._extractor.extract

        def observe_extraction(turns, chunks=None):
            observed.append(
                (turns[0].text, [chunk.text for chunk in chunks or []])
            )
            return original_extract(turns, chunks)

        monkeypatch.setattr(condenser._extractor, "extract", observe_extraction)
        enriched = condenser.drain_pending_enrichments()

        live_texts = [chunk.text for chunk in source_chunks[1:]]
        assert observed == [("\n".join(live_texts), live_texts)]
        assert retired.text not in observed[0][0]
        assert [chunk.text for chunk in enriched[0][1]] == live_texts
        assert condenser.pending_enrichment_count() == 0


def test_drain_rejects_nonboolean_enrichment_mode(tmp_path):
    with _condenser(
        tmp_path / "invalid-enrichment-mode", RecordingEmbedder()
    ) as condenser:
        with pytest.raises(ValueError, match="enrich must be a boolean"):
            condenser.drain_pending_ingests(enrich=1)


def test_auto_extract_drain_claims_missing_receipt_before_provider(tmp_path):
    data_dir = tmp_path / "drain-adopts-enrichment"
    with MemoryCondenser(
        data_dir=data_dir,
        embedder=RecordingEmbedder(),
        auto_extract=False,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as captured:
        captured.capture(
            "user",
            "I prefer cyan windows in every application.",
            turn_id="cross-config-turn",
        )
        assert captured.pending_enrichment_count() == 0

        embedder = RecordingEmbedder()
    with MemoryCondenser(
        data_dir=data_dir,
        embedder=embedder,
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as drained:
        observed: list[tuple[str, str, int]] = []

        def observe_receipt(_chunks):
            assert not drained._db.connection.in_transaction
            with sqlite3.connect(data_dir / "memory.db") as independent:
                observed.append(
                    independent.execute(
                        "SELECT e.status, p.status, "
                        "(SELECT COUNT(*) FROM chunks) "
                        "FROM pending_enrichments AS e "
                        "JOIN pending_ingests AS p ON p.turn_id = e.turn_id "
                        "WHERE e.turn_id = 'cross-config-turn'"
                    ).fetchone()
                )

        embedder.before_embed = observe_receipt
        completed = drained.drain_pending_ingests()

        assert [turn.turn_id for turn, _chunks in completed] == [
            "cross-config-turn"
        ]
        assert observed == [("pending", "pending", 0)]
        assert drained.pending_ingest_count() == 0
        assert drained.pending_enrichment_count() == 0
        assert drained._db.schema_version == CURRENT_SCHEMA_VERSION
        assert len(drained.memory.list_items()) == 1


def test_drain_finalizes_enrichment_when_extractor_emits_no_ops(tmp_path):
    embedder = RecordingEmbedder()
    with MemoryCondenser(
        data_dir=tmp_path / "empty-enrichment",
        embedder=embedder,
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.capture("assistant", "A neutral acknowledgement.", turn_id="no-op")
        assert condenser.pending_enrichment_count() == 1

        condenser.drain_pending_ingests()

        assert condenser.pending_enrichment_count() == 0
        assert condenser.memory.list_items() == []


@pytest.mark.parametrize("failure", ["transport", "invalid"])
def test_durable_llm_failure_stays_pending_until_valid_noop(tmp_path, failure):
    attempts = 0

    def complete(_system, _user):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            if failure == "transport":
                raise RuntimeError("temporary provider outage")
            return "invalid-json"
        return (
            '{"create": [], "update": [], "supersede": [], '
            '"delete": [], "pin": []}'
        )

    with MemoryCondenser(
        data_dir=tmp_path / f"durable-llm-{failure}",
        embedder=RecordingEmbedder(),
        extractor=LLMExtractor(complete),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.capture(
            "user",
            "This neutral statement has enough words.",
            turn_id=f"durable-llm-{failure}",
        )
        condenser.drain_pending_ingests(enrich=False)

        with pytest.raises(ExtractionUnavailableError):
            condenser.drain_pending_enrichments()
        assert condenser.pending_enrichment_count() == 1
        assert condenser.memory.list_items() == []

        completed = condenser.drain_pending_enrichments()
        assert [turn.turn_id for turn, _chunks in completed] == [
            f"durable-llm-{failure}"
        ]
        assert condenser.pending_enrichment_count() == 0
        assert attempts == 2


def test_drain_extraction_failure_is_recoverable_after_restart_without_embedding(
    tmp_path, monkeypatch
):
    embedder = RecordingEmbedder()
    data_dir = tmp_path / "auto-extract-retry"
    with MemoryCondenser(
        data_dir=data_dir,
        embedder=embedder,
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.capture(
            "user",
            "I prefer dark mode in every application.",
            turn_id="retry-preference-turn",
        )

        def fail_extract(_turns, _chunks=None):
            raise RuntimeError("synthetic extraction outage")

        monkeypatch.setattr(condenser._extractor, "extract", fail_extract)
        with pytest.raises(RuntimeError, match="synthetic extraction outage"):
            condenser.drain_pending_ingests()

        # Index completion has the same boundary as synchronous ingest: an
        # extraction error does not roll back already-searchable chunks.
        assert condenser.pending_ingest_count() == 0
        assert condenser.pending_enrichment_count() == 1
        assert condenser.search_hybrid("dark mode", k=1)
        assert condenser.memory.list_items() == []

    recovered_embedder = RecordingEmbedder()
    with MemoryCondenser(
        data_dir=data_dir,
        embedder=recovered_embedder,
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as recovered:
        rows = recovered.drain_pending_enrichments()
        assert [turn.turn_id for turn, _chunks in rows] == ["retry-preference-turn"]
        assert recovered_embedder.calls == []
        assert recovered.pending_enrichment_count() == 0
        assert len(recovered.memory.list_items()) == 1


def test_first_enrichment_failure_leaves_it_and_unattempted_turns_pending(
    tmp_path, monkeypatch
):
    embedder = RecordingEmbedder()
    with MemoryCondenser(
        data_dir=tmp_path / "multi-enrichment-failure",
        embedder=embedder,
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.capture_many(
            [
                ("user", "I prefer dark mode.", "s", None, "turn-one"),
                ("user", "We decided to use SQLite.", "s", None, "turn-two"),
            ]
        )
        original_extract = condenser._extractor.extract
        attempts: list[str] = []

        def fail_first(turns, chunks=None):
            attempts.append(turns[0].turn_id)
            raise RuntimeError("first extraction fails")

        monkeypatch.setattr(condenser._extractor, "extract", fail_first)
        with pytest.raises(RuntimeError, match="first extraction fails"):
            condenser.drain_pending_ingests()

        assert attempts == ["turn-one"]
        assert condenser.pending_ingest_count() == 0
        assert condenser.pending_enrichment_count() == 2

        replayed: list[str] = []

        def record_extract(turns, chunks=None):
            replayed.append(turns[0].turn_id)
            return original_extract(turns, chunks)

        monkeypatch.setattr(condenser._extractor, "extract", record_extract)
        first = condenser.drain_pending_enrichments(max_turns=1)
        assert [turn.turn_id for turn, _chunks in first] == ["turn-one"]
        assert replayed == ["turn-one"]
        assert condenser.pending_enrichment_count() == 1

        condenser.drain_pending_enrichments()
        assert replayed == ["turn-one", "turn-two"]
        assert condenser.pending_enrichment_count() == 0


def test_staged_enrichment_apply_and_receipt_seal_are_one_transaction(
    tmp_path, monkeypatch
):
    with MemoryCondenser(
        data_dir=tmp_path / "atomic-enrichment",
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.capture(
            "user", "I prefer atomic blue settings.", turn_id="atomic-turn"
        )
        condenser.drain_pending_ingests(enrich=False)

        extraction_calls = 0
        original_extract = condenser._extractor.extract

        def counted_extract(turns, chunks=None):
            nonlocal extraction_calls
            extraction_calls += 1
            return original_extract(turns, chunks)

        monkeypatch.setattr(condenser._extractor, "extract", counted_extract)
        original_finalize = condenser._pending_enrichments.finalize

        def fail_after_receipt_update(turn_id):
            original_finalize(turn_id)
            raise RuntimeError("synthetic finalize crash")

        monkeypatch.setattr(
            condenser._pending_enrichments,
            "finalize",
            fail_after_receipt_update,
        )
        with pytest.raises(RuntimeError, match="synthetic finalize crash"):
            condenser.drain_pending_enrichments()

        assert condenser.memory.count() == 0
        assert condenser.pending_enrichment_count() == 1
        assert condenser._pending_enrichments.staged_result("atomic-turn") is not None
        assert extraction_calls == 1

        monkeypatch.setattr(
            condenser._pending_enrichments, "finalize", original_finalize
        )
        condenser.drain_pending_enrichments()

        assert condenser.memory.count() == 1
        assert condenser.pending_enrichment_count() == 0
        assert extraction_calls == 1
        replay_state = condenser._db.execute(
            "SELECT staged_ops_json, staged_chunk_ids_json, "
            "length(staged_result_sha256) FROM pending_enrichment_state "
            "WHERE turn_id = 'atomic-turn'"
        ).fetchone()
        assert replay_state == (None, None, 64)


def test_staged_enrichment_survives_restart_before_apply(tmp_path, monkeypatch):
    data_dir = tmp_path / "stage-before-apply"
    with MemoryCondenser(
        data_dir=data_dir,
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.capture(
            "user", "I prefer restart-safe green settings.", turn_id="staged-turn"
        )
        [(turn, chunks)] = condenser.drain_pending_ingests(enrich=False)
        proposed = condenser._extractor.extract([turn], chunks)
        accepted = condenser._validator.validate_for_enrichment(
            proposed, turn, chunks
        ).accepted
        condenser._pending_enrichments.stage_ops(
            turn.turn_id, accepted, [chunk.chunk_id for chunk in chunks]
        )

    with MemoryCondenser(
        data_dir=data_dir,
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as recovered:
        monkeypatch.setattr(
            recovered._extractor,
            "extract",
            lambda _turns, _chunks=None: (_ for _ in ()).throw(
                AssertionError("staged result should avoid re-extraction")
            ),
        )
        completed = recovered.drain_pending_enrichments()
        assert [turn.turn_id for turn, _chunks in completed] == ["staged-turn"]
        assert recovered.memory.count() == 1


def test_t2_provider_work_happens_before_the_sqlite_writer_lock(tmp_path):
    class TransactionAwareEmbedder(RecordingEmbedder):
        def __init__(self) -> None:
            super().__init__()
            self.condenser = None
            self.memory_query_calls = 0

        def embed_query(self, query: str) -> np.ndarray:
            if query:
                assert self.condenser is not None
                assert not self.condenser._db.connection.in_transaction
                self.memory_query_calls += 1
            return super().embed_query(query)

    embedder = TransactionAwareEmbedder()
    with MemoryCondenser(
        data_dir=tmp_path / "provider-before-lock",
        embedder=embedder,
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        embedder.condenser = condenser
        condenser.capture(
            "user", "I prefer provider-safe teal settings.", turn_id="provider-turn"
        )
        condenser.drain_pending_ingests(enrich=False)
        condenser.drain_pending_enrichments()

        assert embedder.memory_query_calls == 1
        assert condenser.memory.count(status=MemoryStatus.ACTIVE) == 1


def test_retirement_after_staging_turns_replay_into_a_noop(tmp_path):
    with MemoryCondenser(
        data_dir=tmp_path / "retire-after-stage",
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.capture(
            "user", "I prefer evidence-safe violet settings.", turn_id="retire-stage"
        )
        [(turn, chunks)] = condenser.drain_pending_ingests(enrich=False)
        proposed = condenser._extractor.extract([turn], chunks)
        accepted = condenser._validator.validate_for_enrichment(
            proposed, turn, chunks
        ).accepted
        condenser._pending_enrichments.stage_ops(
            turn.turn_id, accepted, [chunk.chunk_id for chunk in chunks]
        )
        assert condenser.retriever.delete_chunk(chunks[0].chunk_id)

        condenser.drain_pending_enrichments()

        assert condenser.memory.count() == 0
        assert condenser.pending_enrichment_count() == 0


def test_partial_retirement_after_staging_keeps_still_grounded_operation(tmp_path):
    with MemoryCondenser(
        data_dir=tmp_path / "partial-retire-after-stage",
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=6,
    ) as condenser:
        condenser.capture(
            "user",
            "I prefer retired amber tea. I prefer durable blue coffee.",
            turn_id="partial-stage",
        )
        [(turn, chunks)] = condenser.drain_pending_ingests(enrich=False)
        assert len(chunks) > 1
        ops = MemoryOps(
            create=[
                CreateOp(
                    type=MemoryType.PREFERENCE,
                    content=f"fact from {chunk.chunk_id}",
                    provenance=[
                        Provenance(
                            turn_id=turn.turn_id,
                            chunk_id=chunk.chunk_id,
                            quote=chunk.text,
                        )
                    ],
                )
                for chunk in (chunks[0], chunks[-1])
            ]
            + [
                CreateOp(
                    type=MemoryType.DECISION,
                    content="fact with redundant citations",
                    provenance=[
                        Provenance(
                            turn_id=turn.turn_id,
                            chunk_id=chunk.chunk_id,
                            quote=chunk.text,
                        )
                        for chunk in (chunks[0], chunks[-1])
                    ],
                )
            ]
        )
        accepted = condenser._validator.validate_for_enrichment(
            ops, turn, chunks
        ).accepted
        condenser._pending_enrichments.stage_ops(
            turn.turn_id, accepted, [chunk.chunk_id for chunk in chunks]
        )
        assert condenser.retriever.delete_chunk(chunks[0].chunk_id)

        condenser.drain_pending_enrichments()

        active = {item.content: item for item in condenser.memory.list_items()}
        assert set(active) == {
            f"fact from {chunks[-1].chunk_id}",
            "fact with redundant citations",
        }
        assert [
            citation.chunk_id
            for citation in active["fact with redundant citations"].provenance
        ] == [chunks[-1].chunk_id]


def test_old_staged_create_cannot_resurrect_later_explicit_retirement(tmp_path):
    with MemoryCondenser(
        data_dir=tmp_path / "source-ordered-stage",
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.capture(
            "user", "I prefer source-ordered slate mode.", turn_id="older-source"
        )
        [(turn, chunks)] = condenser.drain_pending_ingests(enrich=False)
        proposed = condenser._extractor.extract([turn], chunks)
        accepted = condenser._validator.validate_for_enrichment(
            proposed, turn, chunks
        ).accepted
        condenser._pending_enrichments.stage_ops(
            turn.turn_id, accepted, [chunk.chunk_id for chunk in chunks]
        )
        explicit = condenser.memory.create(accepted.create[0])
        assert condenser.memory.delete(DeleteOp(mem_id=explicit.mem_id))

        condenser.drain_pending_enrichments()

        assert condenser.memory.count(status=MemoryStatus.ACTIVE) == 0
        assert condenser.memory.get(explicit.mem_id).status is MemoryStatus.DELETED

        # A genuinely later source turn may reassert the exact same fact.
        condenser.capture(
            "user", "I prefer source-ordered slate mode.", turn_id="later-source"
        )
        condenser.drain_pending_ingests(enrich=False)
        condenser.drain_pending_enrichments()
        assert condenser.memory.count(status=MemoryStatus.ACTIVE) == 1


def test_deferred_noncreate_operation_fails_closed_and_remains_pending(
    tmp_path, monkeypatch
):
    with MemoryCondenser(
        data_dir=tmp_path / "forbidden-t2-update",
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.capture(
            "user", "I prefer a stable base memory.", turn_id="base-source"
        )
        [(base_turn, base_chunks)] = condenser.drain_pending_ingests(enrich=False)
        base_ops = condenser._extractor.extract([base_turn], base_chunks)
        base = condenser.memory.create(base_ops.create[0])
        condenser._pending_enrichments.finalize(base_turn.turn_id)
        condenser._db.commit()

        condenser.capture(
            "user", "I prefer a forbidden deferred update.", turn_id="update-source"
        )
        condenser.drain_pending_ingests(enrich=False)
        malicious = MemoryOps(
            update=[UpdateOp(mem_id=base.mem_id, content="mutated by T2")]
        )
        monkeypatch.setattr(
            condenser._extractor,
            "extract",
            lambda _turns, _chunks=None: malicious,
        )

        with pytest.raises(ExtractionUnavailableError, match="forbidden"):
            condenser.drain_pending_enrichments()

        assert condenser.pending_enrichment_count() == 1
        assert condenser.memory.get(base.mem_id).content != "mutated by T2"


def test_unbound_deferred_correction_is_queued_then_explicitly_resolved(tmp_path):
    with MemoryCondenser(
        data_dir=tmp_path / "unbound-correction",
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        predecessor_turn = condenser.transcript.append(
            "user", "The deadline was Wednesday."
        )
        predecessor = condenser.memory.create(
            CreateOp(
                type=MemoryType.DECISION,
                content="The deadline was Wednesday.",
                provenance=[
                    Provenance(
                        turn_id=predecessor_turn.turn_id,
                        quote="The deadline was Wednesday.",
                    )
                ],
            )
        )
        condenser.capture(
            "user", "Actually the deadline is Thursday.", turn_id="correction-source"
        )
        condenser.drain_pending_ingests(enrich=False)
        condenser.drain_pending_enrichments()

        assert condenser.pending_enrichment_count() == 0
        assert condenser.memory.count(status=MemoryStatus.ACTIVE) == 1
        assert condenser.search_hybrid("deadline Thursday", k=1)
        staged = condenser._pending_enrichments.staged_result("correction-source")
        assert staged is None
        queued = condenser.pending_deferred_corrections()
        assert [entry["turn_id"] for entry in queued] == ["correction-source"]
        correction_id = str(queued[0]["correction_id"])
        assert condenser.pending_enrichment_stats()[
            "deferred_correction_count"
        ] == 1

        successor = condenser.resolve_deferred_correction(
            correction_id, predecessor.mem_id
        )

        assert successor.type is MemoryType.CORRECTION
        assert successor.supersedes == predecessor.mem_id
        assert condenser.memory.get(predecessor.mem_id).status is (
            MemoryStatus.SUPERSEDED
        )
        assert condenser.pending_deferred_corrections() == []
        assert condenser.pending_enrichment_stats()[
            "deferred_correction_count"
        ] == 0
        assert condenser._db.execute(
            "SELECT status, operation_sha256 IS NOT NULL, target_mem_id, "
            "successor_mem_id FROM pending_corrections "
            "WHERE correction_id = ?",
            (correction_id,),
        ).fetchone() == (
            "resolved",
            1,
            predecessor.mem_id,
            successor.mem_id,
        )


def test_identical_corrections_keep_distinct_ordered_receipts(tmp_path, monkeypatch):
    with MemoryCondenser(
        data_dir=tmp_path / "duplicate-corrections",
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        text = "Actually the deadline is Thursday."
        condenser.capture("user", text, turn_id="duplicate-corrections")
        condenser.drain_pending_ingests(enrich=False)
        correction = CreateOp(
            type=MemoryType.CORRECTION,
            content="The deadline is Thursday.",
            provenance=[
                Provenance(
                    turn_id="duplicate-corrections",
                    quote="deadline is Thursday",
                )
            ],
        )
        monkeypatch.setattr(
            condenser._extractor,
            "extract",
            lambda _turns, _chunks=None: MemoryOps(
                create=[correction, correction.model_copy()]
            ),
        )

        condenser.drain_pending_enrichments()

        queued = condenser.pending_deferred_corrections()
        assert [entry["operation_index"] for entry in queued] == [0, 1]
        correction_ids = [str(entry["correction_id"]) for entry in queued]
        assert len(set(correction_ids)) == 2
        with pytest.raises(sqlite3.IntegrityError, match="terminal disposition"):
            condenser._db.execute(
                "UPDATE pending_corrections SET status = 'dismissed', "
                "operation_index = 9, decided_at = ?, reason = 'tamper' "
                "WHERE correction_id = ?",
                (datetime.now(timezone.utc).isoformat(), correction_ids[0]),
            )
        condenser._db.connection.rollback()
        assert all(
            condenser.dismiss_deferred_correction(correction_id)
            for correction_id in correction_ids
        )


def test_retired_staged_correction_finalizes_without_poisoning_t2(tmp_path):
    with MemoryCondenser(
        data_dir=tmp_path / "retired-correction",
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.capture(
            "user", "Actually the deadline is Thursday.", turn_id="retired-correction"
        )
        [(turn, chunks)] = condenser.drain_pending_ingests(enrich=False)
        ops = condenser._validator.validate_for_enrichment(
            condenser._extractor.extract([turn], chunks),
            turn,
            chunks,
            allow_unbound_corrections=True,
        ).accepted
        condenser._pending_enrichments.stage_ops(
            turn.turn_id, ops, [chunk.chunk_id for chunk in chunks]
        )
        assert condenser.retriever.delete_chunk(chunks[0].chunk_id)

        condenser.drain_pending_enrichments()

        assert condenser.pending_enrichment_count() == 0
        assert condenser.pending_deferred_corrections() == []
        assert condenser.memory.list_items() == []


def test_partial_retirement_queues_only_still_grounded_correction(tmp_path):
    with MemoryCondenser(
        data_dir=tmp_path / "partially-retired-corrections",
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=6,
    ) as condenser:
        condenser.capture(
            "user",
            "Actually amber is retired. Actually blue is durable.",
            turn_id="partial-corrections",
        )
        [(turn, chunks)] = condenser.drain_pending_ingests(enrich=False)
        assert len(chunks) > 1
        ops = MemoryOps(
            create=[
                CreateOp(
                    type=MemoryType.CORRECTION,
                    content=f"correction from {chunk.chunk_id}",
                    provenance=[
                        Provenance(
                            turn_id=turn.turn_id,
                            chunk_id=chunk.chunk_id,
                            quote=chunk.text,
                        )
                    ],
                )
                for chunk in (chunks[0], chunks[-1])
            ]
        )
        accepted = condenser._validator.validate_for_enrichment(
            ops, turn, chunks, allow_unbound_corrections=True
        ).accepted
        condenser._pending_enrichments.stage_ops(
            turn.turn_id, accepted, [chunk.chunk_id for chunk in chunks]
        )
        assert condenser.retriever.delete_chunk(chunks[0].chunk_id)

        condenser.drain_pending_enrichments()

        [queued] = condenser.pending_deferred_corrections()
        assert queued["correction"].content == f"correction from {chunks[-1].chunk_id}"
        assert condenser.dismiss_deferred_correction(str(queued["correction_id"]))


def test_deferred_correction_rejects_target_changed_during_embedding(tmp_path):
    class TargetMutatingEmbedder(RecordingEmbedder):
        def __init__(self):
            super().__init__()
            self.on_query = None

        def embed_query(self, query):
            callback, self.on_query = self.on_query, None
            if callback is not None:
                callback()
            return super().embed_query(query)

    embedder = TargetMutatingEmbedder()
    with MemoryCondenser(
        data_dir=tmp_path / "correction-target-race",
        embedder=embedder,
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        source = condenser.transcript.append("user", "The deadline was Wednesday.")
        predecessor = condenser.memory.create(
            CreateOp(
                type=MemoryType.DECISION,
                content="The deadline was Wednesday.",
                provenance=[
                    Provenance(
                        turn_id=source.turn_id,
                        quote="The deadline was Wednesday.",
                    )
                ],
            ),
            embedding=np.zeros(embedder.dim, dtype=np.float32),
        )
        condenser.capture(
            "user", "Actually the deadline is Thursday.", turn_id="racing-correction"
        )
        condenser.drain_pending_ingests(enrich=False)
        condenser.drain_pending_enrichments()
        embedder.on_query = lambda: condenser.memory.update(
            UpdateOp(mem_id=predecessor.mem_id, details="changed after review")
        )

        [queued] = condenser.pending_deferred_corrections()
        with pytest.raises(
            DeferredCorrectionTargetStaleError, match="changed after operator review"
        ):
            condenser.resolve_deferred_correction(
                str(queued["correction_id"]), predecessor.mem_id
            )

        assert condenser.memory.get(predecessor.mem_id).status is MemoryStatus.ACTIVE
        assert len(condenser.pending_deferred_corrections()) == 1
        assert condenser.dismiss_deferred_correction(str(queued["correction_id"]))


def test_deferred_correction_rejects_semantic_noop_before_embedding(
    tmp_path, monkeypatch
):
    class QueryCountingEmbedder(RecordingEmbedder):
        def __init__(self):
            super().__init__()
            self.query_calls = 0

        def embed_query(self, query):
            self.query_calls += 1
            return super().embed_query(query)

    embedder = QueryCountingEmbedder()
    with MemoryCondenser(
        data_dir=tmp_path / "correction-noop",
        embedder=embedder,
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        source = condenser.transcript.append("user", "The deadline was Wednesday.")
        target = condenser.memory.create(
            CreateOp(
                type=MemoryType.DECISION,
                content="The deadline was Wednesday.",
                provenance=[
                    Provenance(
                        turn_id=source.turn_id,
                        quote="The deadline was Wednesday.",
                    )
                ],
            ),
            embedding=np.zeros(embedder.dim, dtype=np.float32),
        )
        condenser.capture(
            "user",
            "Actually the deadline was Wednesday.",
            turn_id="noop-correction",
        )
        condenser.drain_pending_ingests(enrich=False)
        monkeypatch.setattr(
            condenser._extractor,
            "extract",
            lambda _turns, _chunks=None: MemoryOps(
                create=[
                    CreateOp(
                        type=MemoryType.CORRECTION,
                        content=" the   deadline was wednesday. ",
                        provenance=[
                            Provenance(
                                turn_id="noop-correction",
                                quote="deadline was Wednesday",
                            )
                        ],
                    )
                ]
            ),
        )
        condenser.drain_pending_enrichments()
        [queued] = condenser.pending_deferred_corrections()
        calls_before_resolution = embedder.query_calls

        with pytest.raises(
            DeferredCorrectionTargetStaleError,
            match="would not change the selected target",
        ):
            condenser.resolve_deferred_correction(
                str(queued["correction_id"]), target.mem_id
            )

        assert embedder.query_calls == calls_before_resolution
        assert condenser.memory.get(target.mem_id).status is MemoryStatus.ACTIVE
        assert len(condenser.pending_deferred_corrections()) == 1


def test_deferred_correction_rejects_active_replacement_collision(
    tmp_path, monkeypatch
):
    with MemoryCondenser(
        data_dir=tmp_path / "correction-collision",
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        old_source = condenser.transcript.append(
            "user", "The deadline was Wednesday."
        )
        target = condenser.memory.create(
            CreateOp(
                type=MemoryType.DECISION,
                content="The deadline was Wednesday.",
                provenance=[
                    Provenance(
                        turn_id=old_source.turn_id,
                        quote="The deadline was Wednesday.",
                    )
                ],
            )
        )
        collision_source = condenser.transcript.append(
            "user", "The deadline is already Thursday."
        )
        collision = condenser.memory.create(
            CreateOp(
                type=MemoryType.CORRECTION,
                content="The deadline is Thursday.",
                provenance=[
                    Provenance(
                        turn_id=collision_source.turn_id,
                        quote="deadline is already Thursday",
                    )
                ],
            )
        )
        condenser.capture(
            "user", "Actually the deadline is Thursday.", turn_id="collision-correction"
        )
        condenser.drain_pending_ingests(enrich=False)
        monkeypatch.setattr(
            condenser._extractor,
            "extract",
            lambda _turns, _chunks=None: MemoryOps(
                create=[
                    CreateOp(
                        type=MemoryType.CORRECTION,
                        content="The deadline is Thursday.",
                        provenance=[
                            Provenance(
                                turn_id="collision-correction",
                                quote="deadline is Thursday",
                            )
                        ],
                    )
                ]
            ),
        )
        condenser.drain_pending_enrichments()
        [queued] = condenser.pending_deferred_corrections()

        with pytest.raises(
            DeferredCorrectionTargetStaleError,
            match="replacement already exists as active memory",
        ):
            condenser.resolve_deferred_correction(
                str(queued["correction_id"]), target.mem_id
            )

        assert condenser.memory.get(target.mem_id).status is MemoryStatus.ACTIVE
        assert condenser.memory.get(collision.mem_id).status is MemoryStatus.ACTIVE
        assert len(condenser.pending_deferred_corrections()) == 1


def test_correction_resolution_rolls_back_memory_and_receipt_on_failure(
    tmp_path, monkeypatch
):
    with MemoryCondenser(
        data_dir=tmp_path / "correction-resolution-rollback",
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        old_source = condenser.transcript.append(
            "user", "The deadline was Wednesday."
        )
        target = condenser.memory.create(
            CreateOp(
                type=MemoryType.DECISION,
                content="The deadline was Wednesday.",
                provenance=[
                    Provenance(
                        turn_id=old_source.turn_id,
                        quote="The deadline was Wednesday.",
                    )
                ],
            )
        )
        condenser.capture(
            "user", "Actually the deadline is Thursday.", turn_id="rollback-correction"
        )
        condenser.drain_pending_ingests(enrich=False)
        monkeypatch.setattr(
            condenser._extractor,
            "extract",
            lambda _turns, _chunks=None: MemoryOps(
                create=[
                    CreateOp(
                        type=MemoryType.CORRECTION,
                        content="The deadline is Thursday.",
                        provenance=[
                            Provenance(
                                turn_id="rollback-correction",
                                quote="deadline is Thursday",
                            )
                        ],
                    )
                ]
            ),
        )
        condenser.drain_pending_enrichments()
        [queued] = condenser.pending_deferred_corrections()
        original_resolve = condenser._pending_enrichments.resolve_correction

        def fail_after_resolution(*args, **kwargs):
            original_resolve(*args, **kwargs)
            raise RuntimeError("synthetic post-resolution failure")

        monkeypatch.setattr(
            condenser._pending_enrichments,
            "resolve_correction",
            fail_after_resolution,
        )

        with pytest.raises(RuntimeError, match="post-resolution"):
            condenser.resolve_deferred_correction(
                str(queued["correction_id"]), target.mem_id
            )

        assert condenser.memory.get(target.mem_id).status is MemoryStatus.ACTIVE
        assert condenser.memory.count() == 1
        assert [
            entry["correction_id"]
            for entry in condenser.pending_deferred_corrections()
        ] == [queued["correction_id"]]


def test_correction_does_not_block_safe_fact_and_can_be_dismissed(
    tmp_path, monkeypatch
):
    with MemoryCondenser(
        data_dir=tmp_path / "mixed-correction",
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        text = "Actually the deadline is Thursday. I prefer blue reports."
        condenser.capture("user", text, turn_id="mixed-correction")
        condenser.drain_pending_ingests(enrich=False)
        monkeypatch.setattr(
            condenser._extractor,
            "extract",
            lambda _turns, _chunks=None: MemoryOps(
                create=[
                    CreateOp(
                        type=MemoryType.CORRECTION,
                        content="The deadline is Thursday.",
                        provenance=[
                            Provenance(
                                turn_id="mixed-correction",
                                quote="deadline is Thursday",
                            )
                        ],
                    ),
                    CreateOp(
                        type=MemoryType.PREFERENCE,
                        content="I prefer blue reports.",
                        provenance=[
                            Provenance(
                                turn_id="mixed-correction",
                                quote="I prefer blue reports.",
                            )
                        ],
                    ),
                ]
            ),
        )

        condenser.drain_pending_enrichments()

        assert [item.content for item in condenser.memory.list_items()] == [
            "I prefer blue reports."
        ]
        assert len(condenser.pending_deferred_corrections()) == 1
        [queued] = condenser.pending_deferred_corrections()
        [(chunk_id,)] = condenser._db.execute(
            "SELECT chunk_id FROM chunks WHERE turn_id = 'mixed-correction'"
        ).fetchall()
        assert condenser.retriever.delete_chunk(chunk_id)
        assert condenser.dismiss_deferred_correction(str(queued["correction_id"]))
        assert condenser.pending_deferred_corrections() == []
        assert condenser._db.execute(
            "SELECT status, target_mem_id, successor_mem_id "
            "FROM pending_corrections "
            "WHERE turn_id = 'mixed-correction'"
        ).fetchone() == ("dismissed", None, None)
        assert [item.content for item in condenser.memory.list_items()] == [
            "I prefer blue reports."
        ]


def test_legacy_custom_extractor_declares_success_through_extract(tmp_path):
    class LegacyExtractor:
        def extract(self, turns, chunks=None):
            return MemoryOps(
                create=[
                    CreateOp(
                        type=MemoryType.PREFERENCE,
                        content="legacy extractor fact",
                        provenance=[
                            Provenance(
                                turn_id=turns[0].turn_id,
                                quote="legacy extractor fact",
                            )
                        ],
                    )
                ]
            )

    with MemoryCondenser(
        data_dir=tmp_path / "legacy-extractor",
        embedder=RecordingEmbedder(),
        extractor=LegacyExtractor(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.capture(
            "user", "A legacy extractor fact is present.", turn_id="legacy-source"
        )
        condenser.drain_pending_ingests(enrich=False)
        condenser.drain_pending_enrichments()

        assert [item.content for item in condenser.memory.list_items()] == [
            "legacy extractor fact"
        ]


@pytest.mark.parametrize("cite_chunk", [True, False])
def test_enrichment_rejects_provenance_outside_its_live_view(
    tmp_path, monkeypatch, cite_chunk
):
    with MemoryCondenser(
        data_dir=tmp_path / f"retired-provenance-{cite_chunk}",
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=6,
    ) as condenser:
        condenser.capture(
            "user",
            "I prefer retired amber tea. I prefer live blue coffee.",
            turn_id="view-boundary",
        )
        [(_turn, chunks)] = condenser.drain_pending_ingests(enrich=False)
        assert len(chunks) > 1
        retired = chunks[0]
        assert condenser.retriever.delete_chunk(retired.chunk_id)
        malicious = MemoryOps(
            create=[
                CreateOp(
                    type=MemoryType.PREFERENCE,
                    content="A memory sourced from retired evidence",
                    provenance=[
                        Provenance(
                            turn_id="view-boundary",
                            chunk_id=retired.chunk_id if cite_chunk else None,
                            quote=retired.text,
                        )
                    ],
                )
            ]
        )
        monkeypatch.setattr(
            condenser._extractor,
            "extract",
            lambda _turns, _chunks=None: malicious,
        )

        condenser.drain_pending_enrichments()

        assert condenser.memory.count() == 0
        assert condenser.pending_enrichment_count() == 0


def test_failed_enrichment_yields_to_fresh_work_after_bounded_retry(
    tmp_path, monkeypatch
):
    with MemoryCondenser(
        data_dir=tmp_path / "fair-enrichment-retry",
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.capture_many(
            [
                ("user", "I prefer poison red mode.", "s", None, "poison"),
                ("user", "I prefer healthy blue mode.", "s", None, "healthy"),
            ]
        )
        condenser.drain_pending_ingests(enrich=False)
        original_extract = condenser._extractor.extract

        def fail_poison(turns, chunks=None):
            if turns[0].turn_id == "poison":
                raise RuntimeError("secret payload must not persist")
            return original_extract(turns, chunks)

        monkeypatch.setattr(condenser._extractor, "extract", fail_poison)
        with pytest.raises(RuntimeError):
            condenser.drain_pending_enrichments(max_turns=1)
        with pytest.raises(RuntimeError):
            condenser.drain_pending_enrichments(max_turns=1)

        completed = condenser.drain_pending_enrichments(max_turns=1)
        assert [turn.turn_id for turn, _chunks in completed] == ["healthy"]
        stats = condenser.pending_enrichment_stats()
        assert stats["failed_count"] == 1
        assert stats["oldest_error_kind"] == "RuntimeError"
        assert "secret" not in repr(stats)


def test_delayed_enrichment_retry_is_not_counted_ready(tmp_path):
    with MemoryCondenser(
        data_dir=tmp_path / "delayed-ready-count",
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.capture(
            "user", "I prefer a delayed retry counter.", turn_id="delayed-retry"
        )
        condenser.drain_pending_ingests(enrich=False)
        condenser._pending_enrichments.record_failure(
            "delayed-retry", RuntimeError("first failure")
        )
        condenser._pending_enrichments.record_failure(
            "delayed-retry", RuntimeError("second failure")
        )

        stats = condenser.pending_enrichment_stats()
        assert stats["turn_count"] == 1
        assert stats["ready_count"] == 0
        assert stats["failed_count"] == 1


def test_first_staged_winner_clears_pre_stage_retry_cooldown(tmp_path):
    with MemoryCondenser(
        data_dir=tmp_path / "stage-clears-cooldown",
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.capture(
            "user", "I prefer a staged retry winner.", turn_id="staged-retry"
        )
        [(turn, chunks)] = condenser.drain_pending_ingests(enrich=False)
        condenser._pending_enrichments.record_failure(
            turn.turn_id, RuntimeError("first provider failure")
        )
        condenser._pending_enrichments.record_failure(
            turn.turn_id, RuntimeError("second provider failure")
        )
        assert condenser.pending_enrichment_stats()["ready_count"] == 0

        assert condenser._pending_enrichments.stage_ops(
            turn.turn_id,
            MemoryOps(),
            [chunk.chunk_id for chunk in chunks],
        ) is not None

        assert condenser._db.execute(
            "SELECT attempt_count, last_attempt_at, next_attempt_at, "
            "last_error_kind FROM pending_enrichment_state WHERE turn_id = ?",
            (turn.turn_id,),
        ).fetchone() == (0, None, None, None)
        assert condenser._pending_enrichments.pending_turn_ids() == [turn.turn_id]
        assert condenser.pending_enrichment_stats()["ready_count"] == 1


def test_stale_provider_failure_cannot_hide_concurrent_staged_winner(tmp_path):
    with MemoryCondenser(
        data_dir=tmp_path / "stale-provider-loser",
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.capture(
            "user", "I prefer a concurrent staging winner.", turn_id="stage-race"
        )
        [(turn, chunks)] = condenser.drain_pending_ingests(enrich=False)
        assert condenser._pending_enrichments.stage_ops(
            turn.turn_id,
            MemoryOps(),
            [chunk.chunk_id for chunk in chunks],
        ) is not None

        recorded = condenser._pending_enrichments.record_failure(
            turn.turn_id,
            RuntimeError("stale provider loser"),
            ignore_if_staged=True,
        )

        assert recorded is False
        assert condenser._db.execute(
            "SELECT attempt_count, last_attempt_at, next_attempt_at, "
            "last_error_kind FROM pending_enrichment_state WHERE turn_id = ?",
            (turn.turn_id,),
        ).fetchone() == (0, None, None, None)
        assert condenser._pending_enrichments.pending_turn_ids() == [turn.turn_id]


def test_enrichment_falls_through_when_selected_retry_class_was_emptied(
    tmp_path, monkeypatch
):
    with MemoryCondenser(
        data_dir=tmp_path / "enrichment-class-race",
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.capture(
            "user", "I prefer a fresh fallback fact.", turn_id="fresh-t2"
        )
        condenser.drain_pending_ingests(enrich=False)
        monkeypatch.setattr(
            condenser._pending_enrichments,
            "choose_retry_class",
            lambda _now: True,
        )

        completed = condenser.drain_pending_enrichments(max_turns=1)

        assert [turn.turn_id for turn, _chunks in completed] == ["fresh-t2"]
        assert condenser.pending_enrichment_count() == 0


def test_pre_v15_pending_enrichment_is_quarantined_but_t1_stays_searchable(
    tmp_path,
):
    with MemoryCondenser(
        data_dir=tmp_path / "legacy-enrichment-quarantine",
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.capture(
            "user", "I prefer a legacy searchable fact.", turn_id="legacy-t2"
        )
        condenser.drain_pending_ingests(enrich=False)
        # Reproduce a v14 default-ordinal receipt at the v15 migration boundary.
        condenser._db.execute(
            "UPDATE turns SET ordinal = 0 WHERE turn_id = 'legacy-t2'"
        )
        condenser._db.execute(
            "UPDATE meta SET value = '0' "
            "WHERE key = 'v15_legacy_retirement_boundary'"
        )
        condenser._db.execute(
            "INSERT INTO pending_enrichment_legacy_quarantine "
            "(turn_id, migration_boundary) VALUES ('legacy-t2', 0)"
        )
        condenser._db.commit()

        assert condenser.drain_pending_enrichments() == []

        assert condenser.search_hybrid("legacy searchable", k=1)
        stats = condenser.pending_enrichment_stats()
        assert stats["ready_count"] == 0
        assert stats["failed_count"] == 0
        assert stats["legacy_quarantined_count"] == 1
        assert condenser.pending_legacy_enrichments() == [
            {
                "turn_id": "legacy-t2",
                "source_ordinal": 0,
                "migration_boundary": 0,
            }
        ]
        assert condenser.discard_legacy_pending_enrichment("legacy-t2")
        assert condenser.pending_enrichment_count() == 0
        assert condenser.pending_enrichment_stats()["discarded_legacy_count"] == 1
        assert condenser._db.execute(
            "SELECT disposition, reason FROM pending_enrichment_dispositions "
            "WHERE turn_id = 'legacy-t2'"
        ).fetchone() == (
            "discarded_legacy",
            "pre-v15 source-order ambiguity; T1 evidence retained",
        )


def test_legacy_enrichment_is_not_discardable_until_t1_is_searchable(tmp_path):
    with MemoryCondenser(
        data_dir=tmp_path / "legacy-enrichment-before-t1",
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.capture(
            "user", "I prefer a legacy pending fact.", turn_id="legacy-before-t1"
        )
        manifest = condenser._pending_ingests.get("legacy-before-t1")
        assert manifest is not None
        condenser._db.connection.execute("BEGIN IMMEDIATE")
        condenser._pending_enrichments.claim(manifest)
        condenser._db.connection.commit()
        condenser._db.execute(
            "UPDATE turns SET ordinal = 0 WHERE turn_id = 'legacy-before-t1'"
        )
        condenser._db.execute(
            "UPDATE meta SET value = '0' "
            "WHERE key = 'v15_legacy_retirement_boundary'"
        )
        condenser._db.execute(
            "INSERT INTO pending_enrichment_legacy_quarantine "
            "(turn_id, migration_boundary) VALUES ('legacy-before-t1', 0)"
        )
        condenser._db.commit()

        assert condenser.pending_legacy_enrichments() == []
        assert not condenser.discard_legacy_pending_enrichment("legacy-before-t1")

        condenser.drain_pending_ingests(enrich=False)

        assert [entry["turn_id"] for entry in condenser.pending_legacy_enrichments()] == [
            "legacy-before-t1"
        ]
        assert condenser.discard_legacy_pending_enrichment("legacy-before-t1")


def test_valid_t2_runs_behind_multiple_legacy_quarantines(tmp_path):
    with MemoryCondenser(
        data_dir=tmp_path / "legacy-enrichment-head-of-line",
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.capture_many(
            [
                ("user", "I prefer legacy amber.", None, None, "legacy-a"),
                ("user", "I prefer legacy green.", None, None, "legacy-b"),
                ("user", "I prefer current blue.", None, None, "current-c"),
            ]
        )
        condenser.drain_pending_ingests(enrich=False)
        condenser._db.execute(
            "UPDATE meta SET value = '1' "
            "WHERE key = 'v15_legacy_retirement_boundary'"
        )
        condenser._db.executemany(
            "INSERT INTO pending_enrichment_legacy_quarantine "
            "(turn_id, migration_boundary) VALUES (?, 1)",
            [("legacy-a",), ("legacy-b",)],
        )
        condenser._db.commit()

        completed = condenser.drain_pending_enrichments(max_turns=1)

        assert [turn.turn_id for turn, _chunks in completed] == ["current-c"]
        assert condenser.pending_enrichment_count() == 2
        assert condenser.pending_enrichment_stats()["legacy_quarantined_count"] == 2
        assert {item.content for item in condenser.memory.list_items()} == {
            "I prefer current blue."
        }


def test_post_migration_t2_claim_for_old_turn_is_quarantined(tmp_path):
    data_dir = tmp_path / "post-migration-old-turn-claim"
    with MemoryCondenser(
        data_dir=data_dir,
        embedder=RecordingEmbedder(),
        auto_extract=False,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.capture(
            "user", "I prefer safely adopted violet.", turn_id="safe-old-turn"
        )
        # The source existed at the recorded boundary, but its T2 receipt did
        # not. This models v15 adopting a v13/auto_extract=False pending T1.
        condenser._db.execute(
            "UPDATE meta SET value = '1' "
            "WHERE key = 'v15_legacy_retirement_boundary'"
        )
        condenser._db.commit()

    with MemoryCondenser(
        data_dir=data_dir,
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        completed = condenser.drain_pending_ingests(enrich=True)

        assert [turn.turn_id for turn, _chunks in completed] == ["safe-old-turn"]
        assert condenser.pending_enrichment_count() == 1
        assert condenser.pending_enrichment_stats()["failed_count"] == 0
        assert [entry["turn_id"] for entry in condenser.pending_legacy_enrichments()] == [
            "safe-old-turn"
        ]
        assert condenser.memory.list_items() == []
        assert condenser.discard_legacy_pending_enrichment("safe-old-turn")


def test_exact_ingest_retry_of_old_t1_is_quarantined_without_failure(tmp_path):
    data_dir = tmp_path / "post-migration-old-turn-ingest"
    text = "I prefer safely retried indigo."
    with MemoryCondenser(
        data_dir=data_dir,
        embedder=RecordingEmbedder(),
        auto_extract=False,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.ingest("user", text, turn_id="safe-old-ingest")
        condenser._db.execute(
            "UPDATE meta SET value = '1' "
            "WHERE key = 'v15_legacy_retirement_boundary'"
        )
        condenser._db.commit()

    with MemoryCondenser(
        data_dir=data_dir,
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        turn, chunks = condenser.ingest(
            "user", text, turn_id="safe-old-ingest"
        )

        assert turn.turn_id == "safe-old-ingest"
        assert chunks
        assert condenser.pending_enrichment_stats()["failed_count"] == 0
        assert [entry["turn_id"] for entry in condenser.pending_legacy_enrichments()] == [
            "safe-old-ingest"
        ]
        assert condenser.memory.list_items() == []


def test_legacy_disposition_and_finalize_roll_back_together(tmp_path, monkeypatch):
    with MemoryCondenser(
        data_dir=tmp_path / "legacy-disposition-rollback",
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.capture(
            "user", "I prefer retained legacy cyan.", turn_id="rollback-legacy"
        )
        condenser.drain_pending_ingests(enrich=False)
        condenser._db.execute(
            "INSERT INTO pending_enrichment_legacy_quarantine "
            "(turn_id, migration_boundary) VALUES ('rollback-legacy', 0)"
        )
        condenser._db.commit()
        original_finalize = condenser._pending_enrichments.finalize

        def fail_after_finalize(turn_id):
            original_finalize(turn_id)
            raise RuntimeError("synthetic post-disposition failure")

        monkeypatch.setattr(
            condenser._pending_enrichments, "finalize", fail_after_finalize
        )

        with pytest.raises(RuntimeError, match="post-disposition"):
            condenser.discard_legacy_pending_enrichment("rollback-legacy")

        assert condenser._pending_enrichments.status("rollback-legacy") == "pending"
        assert condenser._db.execute(
            "SELECT COUNT(*) FROM pending_enrichment_dispositions "
            "WHERE turn_id = 'rollback-legacy'"
        ).fetchone() == (0,)
        assert condenser.search_hybrid("legacy cyan", k=1)


def test_stage_ops_returns_locked_winner_without_post_commit_reread(
    tmp_path, monkeypatch
):
    with MemoryCondenser(
        data_dir=tmp_path / "stage-locked-winner",
        embedder=RecordingEmbedder(),
        auto_extract=True,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    ) as condenser:
        condenser.capture(
            "user", "I prefer a locked staging winner.", turn_id="stage-winner"
        )
        [(turn, chunks)] = condenser.drain_pending_ingests(enrich=False)
        ops = condenser._validator.validate_for_enrichment(
            condenser._extractor.extract([turn], chunks), turn, chunks
        ).accepted
        monkeypatch.setattr(
            condenser._pending_enrichments,
            "staged_result",
            lambda _turn_id: (_ for _ in ()).throw(
                AssertionError("stage_ops must not re-read after commit")
            ),
        )

        winner = condenser._pending_enrichments.stage_ops(
            turn.turn_id, ops, [chunk.chunk_id for chunk in chunks]
        )

        assert winner is not None
        assert winner[0] == ops


def test_failed_ingest_batch_isolates_poison_and_does_not_block_later_turns(
    tmp_path,
):
    class SelectiveEmbedder(RecordingEmbedder):
        def embed_chunks(self, chunks):
            self.calls.append(tuple(chunk.turn_id for chunk in chunks))
            if any(chunk.turn_id == "poison" for chunk in chunks):
                raise RuntimeError("secret source payload must not persist")
            vector = self.embed_query("").tolist()
            return [
                chunk.model_copy(update={"embedding": vector}) for chunk in chunks
            ]

    embedder = SelectiveEmbedder()
    with _condenser(tmp_path / "fair-ingest-retry", embedder) as condenser:
        condenser.capture_many(
            [
                ("user", "Poison ingest source text.", "s", None, "poison"),
                ("user", "Collateral ingest source text.", "s", None, "collateral"),
                ("user", "Fresh ingest source text.", "s", None, "fresh"),
            ]
        )
        with pytest.raises(RuntimeError):
            condenser.drain_pending_ingests(max_manifests=2)
        with pytest.raises(RuntimeError):
            condenser.drain_pending_ingests(max_manifests=2)

        fresh = condenser.drain_pending_ingests(max_manifests=2)
        with pytest.raises(RuntimeError):
            condenser.drain_pending_ingests(max_manifests=2)
        collateral = condenser.drain_pending_ingests(max_manifests=2)

        assert [turn.turn_id for turn, _chunks in collateral] == ["collateral"]
        assert [turn.turn_id for turn, _chunks in fresh] == ["fresh"]
        assert condenser.pending_ingest_count() == 1
        stats = condenser.pending_ingest_stats()
        assert stats["failed_count"] == 1
        assert stats["oldest_error_kind"] == "RuntimeError"
        assert "secret" not in repr(stats)
        persisted = condenser._db.execute(
            "SELECT last_error_kind FROM pending_ingest_attempts "
            "WHERE turn_id = 'poison'"
        ).fetchone()
        assert persisted == ("RuntimeError",)


def test_transient_batch_failure_retries_the_original_cohort_once(tmp_path):
    class TransientEmbedder(RecordingEmbedder):
        def __init__(self):
            super().__init__()
            self.batch_calls = []

        def embed_chunks(self, chunks):
            self.batch_calls.append(tuple(chunk.turn_id for chunk in chunks))
            if len(self.batch_calls) == 1:
                raise RuntimeError("transient provider outage")
            vector = self.embed_query("").tolist()
            return [
                chunk.model_copy(update={"embedding": vector}) for chunk in chunks
            ]

    embedder = TransientEmbedder()
    with _condenser(tmp_path / "transient-ingest-cohort", embedder) as condenser:
        condenser.capture_many(
            [
                ("user", f"Transient turn {index}.", "s", None, f"turn-{index}")
                for index in range(4)
            ]
        )

        with pytest.raises(RuntimeError, match="transient provider"):
            condenser.drain_pending_ingests(max_manifests=4)
        completed = condenser.drain_pending_ingests(max_manifests=4)

        expected = tuple(f"turn-{index}" for index in range(4))
        assert [turn.turn_id for turn, _chunks in completed] == list(expected)
        assert embedder.batch_calls == [expected, expected]
        assert condenser.pending_ingest_count() == 0


def test_ingest_falls_through_when_selected_retry_class_was_emptied(
    tmp_path, monkeypatch
):
    with _condenser(
        tmp_path / "ingest-class-race", RecordingEmbedder()
    ) as condenser:
        condenser.capture("user", "Fresh T1 fallback text.", turn_id="fresh-t1")
        monkeypatch.setattr(
            condenser._pending_ingests,
            "choose_retry_class",
            lambda _now: True,
        )

        completed = condenser.drain_pending_ingests(max_manifests=1)

        assert [turn.turn_id for turn, _chunks in completed] == ["fresh-t1"]
        assert condenser.pending_ingest_count() == 0
