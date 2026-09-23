"""Stateful ingestion and indexed-signature workflows for the condenser."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Sequence

import numpy as np

from memory_condense.associations.association_store import AssociationArtifact
from memory_condense.domain.schemas import (
    Chunk,
    CreateOp,
    MemoryItem,
    MemoryOps,
    MemoryStatus,
    MemoryType,
    RetrievalResult,
    SupersedeOp,
    Turn,
    content_key,
)
from memory_condense.ingest.extractor import ExtractionUnavailableError
from memory_condense.ingest.transcript_source import TranscriptFile
from memory_condense.persistence.conversation_graph_store import (
    GRAPH_BOOTSTRAP_DEFAULT_MAX_TURNS,
    GraphBootstrapResult,
    GraphCompilationResult,
)
from memory_condense.persistence.conversation_envelope_store import (
    CONVERSATION_ENVELOPE_DEFAULT_MAX_TURNS,
    CONVERSATION_ENVELOPE_HARD_MAX_TURNS,
    ConversationEnvelopeAssignment,
    ConversationEnvelopeBootstrapResult,
    ConversationEnvelopeEvent,
)
from memory_condense.persistence.db import INDEXED_CHUNK_SQL
from memory_condense.persistence.pending_ingest_store import (
    PendingIngestAlreadyIndexedError,
    PendingIngestManifest,
)
from memory_condense.persistence.memory_store import PreparedMemoryPlanStaleError
from memory_condense.search.incremental_conversation_graph import (
    IncrementalConversationGraph,
)


_IngestRecord = (
    tuple[str, str, str | None]
    | tuple[str, str, str | None, datetime | None]
    | tuple[str, str, str | None, datetime | None, str]
)
_NormalizedIngestRecord = tuple[
    str,
    str,
    str | None,
    datetime | None,
    str | None,
]
_IngestResult = tuple[Turn, list[Chunk]]
_IDLE_GRAPH_RETRY_MAX_TURNS = 32
_IDLE_ENVELOPE_RETRY_MAX_TURNS = 32


class DeferredCorrectionTargetStaleError(RuntimeError):
    """An explicitly selected correction target is no longer active."""


class AmbiguousLegacyEnrichmentError(ExtractionUnavailableError):
    """Pre-v15 pending T2 work has unknowable memory-mutation ordering."""


def _correction_target_revision(item: MemoryItem) -> str:
    """Hash the semantic target state an operator reviewed before binding."""
    payload = json.dumps(
        {
            "content": item.content,
            "details": item.details,
            "mem_id": item.mem_id,
            "provenance": sorted(
                (
                    citation.turn_id,
                    citation.chunk_id or "",
                    citation.quote,
                )
                for citation in item.provenance
            ),
            "status": item.status.value,
            "supersedes": item.supersedes,
            "type": item.type.value,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bind_explicit_chunk_ids(
    turn_id: str,
    chunks: Sequence[Chunk],
) -> list[Chunk]:
    """Derive stable chunk IDs from an explicit turn and exact source slice."""

    output: list[Chunk] = []
    for chunk in chunks:
        body = json.dumps(
            {
                "format": "memory-condense-explicit-chunk-id-v1",
                "turn_id": turn_id,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "text": chunk.text,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        chunk_id = hashlib.sha256(body.encode("utf-8")).hexdigest()
        output.append(chunk.model_copy(update={"chunk_id": chunk_id}))
    return output


def _validate_embedded_chunks(
    source_chunks: Sequence[Chunk],
    embedded_chunks: Sequence[Chunk],
) -> list[Chunk]:
    """Accept exactly one derived vector/lexical result per source chunk."""
    expected: dict[str, Chunk] = {}
    for chunk in source_chunks:
        if chunk.chunk_id in expected:
            raise ValueError("source chunks contain a duplicate chunk identity")
        expected[chunk.chunk_id] = chunk

    actual: dict[str, Chunk] = {}
    try:
        candidates = list(embedded_chunks)
    except TypeError as exc:
        raise ValueError("embedder did not return a chunk sequence") from exc
    for chunk in candidates:
        if not isinstance(chunk, Chunk):
            raise ValueError("embedder returned a non-chunk value")
        if chunk.chunk_id in actual:
            raise ValueError("embedder returned a duplicate chunk identity")
        actual[chunk.chunk_id] = chunk

    if set(actual) != set(expected):
        raise ValueError("embedder changed the complete chunk identity set")

    output: list[Chunk] = []
    for chunk_id, source in expected.items():
        embedded = actual[chunk_id]
        if embedded.embedding is None:
            raise ValueError("embedder returned a chunk without an embedding")
        if source.model_copy(
            update={
                "embedding": embedded.embedding,
                "lexical_weights": embedded.lexical_weights,
            }
        ) != embedded:
            raise ValueError("embedder changed a chunk source field")
        output.append(embedded)
    return output


def _embed_source_chunks(embedder: object, chunks: Sequence[Chunk]) -> list[Chunk]:
    """Snapshot source ownership before invoking an external provider."""
    source_snapshot = tuple(chunk.model_copy(deep=True) for chunk in chunks)
    embed_chunks = getattr(embedder, "embed_chunks")
    return _validate_embedded_chunks(
        source_snapshot,
        embed_chunks(
            [chunk.model_copy(deep=True) for chunk in source_snapshot]
        ),
    )


class IngestWorkflowMixin:
    """Internal workflow methods composed by ``MemoryCondenser``."""

    def ingest(
        self,
        role: str,
        text: str,
        *,
        source_id: str | None = None,
        created_at: datetime | None = None,
        turn_id: str | None = None,
    ) -> tuple[Turn, list[Chunk]]:
        """Ingest a single conversation turn.

        Stores the turn, chunks and embeds the text, indexes the chunks for
        both dense and lexical retrieval, and — when ``auto_extract`` is on —
        proposes memory items, validates their provenance, and applies the
        surviving ops. Capture is committed before provider work begins, so an
        embedding or extraction exception can leave a durable turn for an
        idempotent retry with the same explicit identity.
        """
        captured, manifests, ingest_status = self._capture_records(
            [(role, text, source_id, created_at, turn_id)]
        )
        completed = self._complete_captured_ingests(
            captured,
            manifests,
            ingest_status,
        )
        turn, chunks = completed[0]

        # Envelope assignment is deterministic derived metadata.  It runs only
        # after T1 has committed and therefore cannot make raw/searchable state
        # unavailable if its independent worker fails.
        self._drain_pending_envelopes_fail_open(
            max_turns=1,
            turn_ids=(turn.turn_id,),
        )

        # T1 has committed before this best-effort stage begins. A graph
        # compiler failure remains a replayable T1g receipt and cannot make
        # the raw turn or either ordinary index unavailable.
        self._drain_pending_graph_fail_open(
            max_turns=1,
            turn_ids=(turn.turn_id,),
        )

        if self._auto_extract:
            self._enrich_captured_turn(turn)

        return turn, chunks

    def capture(
        self,
        role: str,
        text: str,
        *,
        source_id: str | None = None,
        created_at: datetime | None = None,
        turn_id: str | None = None,
    ) -> tuple[Turn, list[Chunk]]:
        """Durably capture one turn and its exact chunk topology.

        No embedding provider, search index, or automatic memory extraction is
        touched. Non-empty turns remain in the pending ingest journal until a
        synchronous ingest retry or :meth:`drain_pending_ingests` completes
        their indexes. Returned chunks are source chunks and therefore have no
        derived embedding.
        """
        captured, _manifests, _ingest_status = self._capture_records(
            [(role, text, source_id, created_at, turn_id)]
        )
        return captured[0]

    def ingest_many(
        self,
        turns: Sequence[_IngestRecord],
    ) -> list[tuple[Turn, list[Chunk]]]:
        """Ingest a turn batch with one embedding/index update.

        This is the fast path for document and benchmark loading. Transcript
        order and source provenance remain exact, but all chunks are embedded
        together so a 30-turn session does not launch 30 tiny model forwards.

        Automatic memory extraction remains strictly turn-causal and therefore
        uses :meth:`ingest` sequentially. The batched path is used when
        ``auto_extract=False``, which is already the retrieval-evaluation and
        corpus-indexing configuration.
        """
        if self._db.read_only:
            raise sqlite3.OperationalError("attempt to write a readonly database")
        records, explicit_times = self._normalize_ingest_records(turns)
        if self._auto_extract:
            return [
                self.ingest(
                    role,
                    text,
                    source_id=source_id,
                    created_at=(
                        explicit_times[turn_id]
                        if turn_id in explicit_times and created_at is None
                        else created_at
                    ),
                    turn_id=turn_id,
                )
                for role, text, source_id, created_at, turn_id in records
            ]

        captured, manifests, ingest_status = self._capture_normalized_records(
            records,
            explicit_times,
        )
        completed = self._complete_captured_ingests(
            captured,
            manifests,
            ingest_status,
        )
        turn_ids = tuple(dict.fromkeys(turn.turn_id for turn, _chunks in completed))
        if turn_ids:
            self._drain_pending_envelopes_fail_open(
                max_turns=min(
                    len(turn_ids), CONVERSATION_ENVELOPE_HARD_MAX_TURNS
                ),
                turn_ids=turn_ids,
            )
        return completed

    def capture_many(
        self,
        turns: Sequence[_IngestRecord],
    ) -> list[tuple[Turn, list[Chunk]]]:
        """Atomically capture an ordered turn batch without indexing it.

        As with :meth:`capture`, automatic extraction is deferred until the
        captured turns are drained.
        """
        captured, _manifests, _ingest_status = self._capture_records(turns)
        return captured

    def _normalize_ingest_records(
        self,
        turns: Sequence[_IngestRecord],
    ) -> tuple[list[_NormalizedIngestRecord], dict[str, datetime]]:
        """Normalize batch identities and resolve authoritative timestamps."""
        records: list[_NormalizedIngestRecord] = []
        for record in turns:
            if not 3 <= len(record) <= 5:
                raise ValueError(
                    "ingest records need role, text, source, optional time, "
                    "and optional explicit turn ID"
                )
            role, text, source_id, created_at, turn_id = (
                *record,
                *(None,) * (5 - len(record)),
            )
            if turn_id is not None:
                turn_id = str(turn_id).strip()
                if not turn_id:
                    raise ValueError("explicit turn IDs must be non-empty")
            if created_at is not None:
                created_at = (
                    created_at.replace(tzinfo=timezone.utc)
                    if created_at.tzinfo is None
                    else created_at.astimezone(timezone.utc)
                )
            records.append((role, text, source_id, created_at, turn_id))

        # An omitted timestamp is a wildcard for an explicit identity, just
        # as it is in sequential ``append``/``publish_turn`` retries. Resolve
        # every mixed batch group to its one explicit timestamp before any
        # ``Turn`` generates a default value. Two genuinely explicit values
        # still conflict and fail before embedding or publication.
        explicit_times: dict[str, datetime] = {}
        for _role, _text, _source_id, created_at, explicit_id in records:
            if explicit_id is None or created_at is None:
                continue
            previous_time = explicit_times.setdefault(explicit_id, created_at)
            if previous_time != created_at:
                raise ValueError(
                    "batch contains duplicate turn_id with different content"
                )
        return records, explicit_times

    def _capture_records(
        self,
        turns: Sequence[_IngestRecord],
    ) -> tuple[
        list[_IngestResult],
        dict[str, PendingIngestManifest],
        dict[str, str],
    ]:
        """Normalize and durably capture one caller-owned batch."""
        if self._db.read_only:
            raise sqlite3.OperationalError("attempt to write a readonly database")
        records, explicit_times = self._normalize_ingest_records(turns)
        return self._capture_normalized_records(records, explicit_times)

    def _capture_normalized_records(
        self,
        records: Sequence[_NormalizedIngestRecord],
        explicit_times: dict[str, datetime],
    ) -> tuple[
        list[_IngestResult],
        dict[str, PendingIngestManifest],
        dict[str, str],
    ]:
        """Publish turns and manifests only after all batch validation passes."""

        staged: list[tuple[Turn, list[Chunk]]] = []
        publication_requests: list[tuple[Turn, bool]] = []
        flat_chunks: list[Chunk] = []
        staged_explicit_turns: dict[str, tuple[Turn, bool]] = {}
        existing_manifests = self._pending_ingests.get_many(
            [
                turn_id
                for _role, _text, _source_id, _created_at, turn_id in records
                if turn_id is not None
            ]
        )
        for role, text, source_id, created_at, turn_id in records:
            created_at_was_explicit = created_at is not None
            effective_created_at = (
                explicit_times.get(turn_id, created_at)
                if turn_id is not None
                else created_at
            )
            turn = self._transcript.stage(
                role,
                text,
                source_id=source_id,
                created_at=effective_created_at,
                turn_id=turn_id,
            )
            if turn_id is not None:
                previous = staged_explicit_turns.get(turn_id)
                if previous is None:
                    staged_explicit_turns[turn_id] = (
                        turn,
                        not created_at_was_explicit,
                    )
                elif (
                    not created_at_was_explicit
                    and previous[1]
                    and previous[0].role == turn.role
                    and previous[0].text == turn.text
                    and previous[0].source_id == turn.source_id
                ):
                    # ``Turn`` generates created_at when it is omitted. Reuse
                    # the first generated value so an exact repeated explicit
                    # identity stays idempotent inside this one batch.
                    turn = previous[0]
            # Once a turn has a durable receipt, that receipt—not today's
            # chunker configuration—is the authoritative replay topology.
            # This also makes a generated-ID capture retryable through the
            # ordinary API: its original chunks used generated IDs, whereas a
            # later call necessarily supplies the returned turn ID explicitly.
            durable_manifest = (
                existing_manifests.get(turn.turn_id)
                if turn_id is not None
                else None
            )
            if durable_manifest is not None:
                chunks = durable_manifest.reconstruct(turn)
            else:
                chunks = self._chunker.chunk_turn(turn.turn_id, text)
                if turn_id is not None:
                    chunks = _bind_explicit_chunk_ids(turn.turn_id, chunks)
            staged.append((turn, chunks))
            publication_requests.append((turn, created_at_was_explicit))
            flat_chunks.extend(chunks)

        self._validate_staged_turns([turn for turn, _chunks in staged])
        # Exact repeated explicit turns are idempotent at the transcript layer,
        # but indexing both copies would allocate two HNSW labels for one
        # deterministic chunk ID.  Collapse only byte-identical chunk models;
        # a conflicting identity is an input error, not a dedup opportunity.
        self._unique_chunks(flat_chunks)
        manifests: dict[str, PendingIngestManifest] = {}
        for turn, chunks in staged:
            manifest = PendingIngestManifest.build(turn, chunks)
            previous = manifests.setdefault(turn.turn_id, manifest)
            if previous != manifest:
                raise ValueError(
                    "batch contains duplicate turn_id with different chunks"
                )

        published, ingest_status = self._publish_staged_turns(
            publication_requests,
            manifests,
        )
        staged = [(published[turn.turn_id], chunks) for turn, chunks in staged]
        return staged, manifests, ingest_status

    def _unique_chunks(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Collapse exact duplicates while rejecting identity collisions."""
        unique: dict[str, Chunk] = {}
        for chunk in chunks:
            previous = unique.setdefault(chunk.chunk_id, chunk)
            if previous != chunk:
                raise ValueError(
                    "batch contains duplicate chunk_id with different content"
                )
        return list(unique.values())

    def _complete_captured_ingests(
        self,
        captured: Sequence[_IngestResult],
        manifests: dict[str, PendingIngestManifest],
        ingest_status: dict[str, str],
    ) -> list[_IngestResult]:
        """Embed one captured batch and seal only its pending manifests."""
        flat_chunks = self._unique_chunks(
            [chunk for _turn, chunks in captured for chunk in chunks]
        )
        pending_turn_ids = {
            turn_id
            for turn_id, status in ingest_status.items()
            if status == "pending"
        }
        pending_source_chunks = [
            chunk for chunk in flat_chunks if chunk.turn_id in pending_turn_ids
        ]
        pending_manifests = [
            manifest
            for turn_id, manifest in manifests.items()
            if ingest_status[turn_id] == "pending"
        ]
        try:
            embedded = (
                _embed_source_chunks(self._embedder, pending_source_chunks)
                if pending_source_chunks
                else []
            )
            if pending_manifests:
                self._index_pending_manifests(embedded, pending_manifests)
        except BaseException as exc:
            self._pending_ingests.record_failure(
                [manifest.turn_id for manifest in pending_manifests], exc
            )
            raise

        by_turn: dict[str, list[Chunk]] = {}
        for chunk in embedded:
            by_turn.setdefault(chunk.turn_id, []).append(chunk)
        durable_by_turn: dict[str, list[Chunk]] = {}
        output: list[_IngestResult] = []
        for turn, _chunks in captured:
            if ingest_status[turn.turn_id] == "indexed":
                if turn.turn_id not in durable_by_turn:
                    durable_by_turn[turn.turn_id] = self._indexed_turn_chunks(
                        turn.turn_id
                    )
                chunks = durable_by_turn[turn.turn_id]
            else:
                chunks = by_turn.get(turn.turn_id, [])
            output.append((turn, chunks))
        return output

    def _validate_staged_turns(self, turns: Sequence[Turn]) -> None:
        """Reject conflicting identities before a batch publishes anything."""
        by_id: dict[str, Turn] = {}
        for turn in turns:
            previous = by_id.setdefault(turn.turn_id, turn)
            if previous != turn:
                raise ValueError(
                    "batch contains duplicate turn_id with different content"
                )

    def _publish_staged_turns(
        self,
        requests: Sequence[tuple[Turn, bool]],
        manifests: dict[str, PendingIngestManifest],
    ) -> tuple[dict[str, Turn], dict[str, str]]:
        """Atomically publish canonical turns and their replay manifests."""
        published: dict[str, Turn] = {}
        ingest_status: dict[str, str] = {}
        connection = self._db.connection
        try:
            connection.execute("BEGIN IMMEDIATE")
            for turn, created_at_was_explicit in requests:
                stored, _inserted = self._transcript.publish_turn(
                    turn,
                    compare_created_at=created_at_was_explicit,
                    commit=False,
                )
                published[turn.turn_id] = stored
            manifest_batch = tuple(manifests.values())
            ingest_status.update(
                self._pending_ingests.claim_many(manifest_batch)
            )
            if self._auto_extract:
                self._pending_enrichments.claim_many(manifest_batch)
            # T0 records only this manifest-bound obligation. Phrase
            # extraction and graph publication are forbidden in this
            # transcript/pending-ingest transaction.
            self._conversation_graphs.claim_many(manifest_batch)
            # The envelope claim similarly stores coordinates and authority
            # only. Boundary assignment runs later and stores no raw text.
            self._conversation_envelopes.claim_many(tuple(published))
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        return published, ingest_status

    def _index_pending_manifests(
        self,
        embedded: Sequence[Chunk],
        manifests: Sequence[PendingIngestManifest],
    ) -> None:
        """Complete pending receipts, filtering raced completions finitely."""
        remaining: dict[str, PendingIngestManifest] = {}
        for manifest in manifests:
            previous = remaining.setdefault(manifest.turn_id, manifest)
            if previous != manifest:
                raise ValueError("conflicting pending manifests in one index pass")
        while remaining:
            attempt = tuple(remaining.values())
            attempt_turn_ids = set(remaining)
            attempt_chunks = [
                chunk
                for chunk in embedded
                if chunk.turn_id in attempt_turn_ids
            ]
            try:
                if attempt_chunks:
                    self._retriever.add_chunks(
                        attempt_chunks,
                        finalize=lambda attempt=attempt: (
                            self._pending_ingests.finalize(attempt)
                        ),
                    )
                else:
                    connection = self._db.connection
                    try:
                        connection.execute("BEGIN IMMEDIATE")
                        self._pending_ingests.finalize(attempt)
                        connection.commit()
                    except BaseException:
                        connection.rollback()
                        raise
            except PendingIngestAlreadyIndexedError as exc:
                previous_count = len(remaining)
                for turn_id in exc.turn_ids:
                    remaining.pop(turn_id, None)
                if len(remaining) == previous_count:
                    raise RuntimeError(
                        "indexed-race signal did not identify an attempted receipt"
                    ) from exc
            else:
                return

    def pending_ingest_count(self) -> int:
        """Return replayable turn publications awaiting complete indexing."""
        return self._pending_ingests.count()

    def pending_graph_compilation_count(
        self,
        *,
        indexed_only: bool = False,
    ) -> int:
        """Return durable T1g obligations, optionally restricted to T1-ready."""

        return self._conversation_graphs.pending_count(indexed_only=indexed_only)

    def pending_conversation_envelope_count(self) -> int:
        """Return current-policy envelope assignments awaiting publication."""

        return self._conversation_envelopes.pending_count()

    def drain_pending_conversation_envelopes(
        self,
        *,
        max_turns: int = CONVERSATION_ENVELOPE_DEFAULT_MAX_TURNS,
        turn_ids: Sequence[str] | None = None,
    ) -> list[ConversationEnvelopeAssignment]:
        """Publish one finite, fail-open-safe envelope-assignment page."""

        if self._db.read_only:
            raise sqlite3.OperationalError("attempt to write a readonly database")
        return self._conversation_envelopes.drain_pending(
            max_turns=max_turns,
            turn_ids=turn_ids,
        )

    def bootstrap_conversation_envelopes(
        self,
        *,
        max_turns: int = CONVERSATION_ENVELOPE_DEFAULT_MAX_TURNS,
    ) -> ConversationEnvelopeBootstrapResult:
        """Claim and publish one finite page from durable ingest receipts."""

        if self._db.read_only:
            raise sqlite3.OperationalError("attempt to write a readonly database")
        return self._conversation_envelopes.bootstrap(max_turns=max_turns)

    def conversation_envelope_for_turn(
        self, turn_id: str
    ) -> ConversationEnvelopeEvent | None:
        """Return the current-policy immutable envelope event for one turn."""

        return self._conversation_envelopes.event_for_turn(turn_id)

    def conversation_envelope_assignment(
        self, turn_id: str
    ) -> ConversationEnvelopeAssignment | None:
        """Return a terminal receipt, including a no-anchor diagnostic."""

        return self._conversation_envelopes.assignment(turn_id)

    def conversation_envelope_members(
        self, envelope_id: str
    ) -> tuple[ConversationEnvelopeEvent, ...]:
        """Return an envelope's events in durable turn order."""

        return self._conversation_envelopes.events_for_envelope(envelope_id)

    def current_conversation_envelope(
        self, source_id: str
    ) -> ConversationEnvelopeEvent | None:
        """Return the latest user-opened envelope for a stable source."""

        return self._conversation_envelopes.current_envelope(source_id)

    def _drain_pending_envelopes_fail_open(
        self,
        *,
        max_turns: int,
        turn_ids: Sequence[str] | None = None,
    ) -> list[ConversationEnvelopeAssignment]:
        """Protect completed T1 state from every envelope-worker failure."""

        try:
            return self.drain_pending_conversation_envelopes(
                max_turns=max_turns,
                turn_ids=turn_ids,
            )
        except Exception as error:
            self.last_conversation_envelope_error = (
                f"{type(error).__module__}.{type(error).__qualname__}"
            )
            return []

    def drain_pending_graph_compilations(
        self,
        *,
        max_turns: int | None = None,
        turn_ids: Sequence[str] | None = None,
    ) -> list[GraphCompilationResult]:
        """Run bounded, fail-open graph compilation after ordinary indexing."""

        if self._db.read_only:
            raise sqlite3.OperationalError("attempt to write a readonly database")
        return self._conversation_graphs.drain_pending(
            max_turns=max_turns,
            turn_ids=turn_ids,
        )

    def bootstrap_conversation_graph(
        self,
        *,
        max_turns: int = GRAPH_BOOTSTRAP_DEFAULT_MAX_TURNS,
    ) -> GraphBootstrapResult:
        """Backfill one bounded active-policy page from sealed T1 receipts."""

        if self._db.read_only:
            raise sqlite3.OperationalError("attempt to write a readonly database")
        return self._conversation_graphs.bootstrap_indexed_turns(
            max_turns=max_turns
        )

    def _drain_pending_graph_fail_open(
        self,
        *,
        max_turns: int | None,
        turn_ids: Sequence[str] | None = None,
    ) -> list[GraphCompilationResult]:
        """Protect a completed T1 boundary from every derived-graph failure."""

        try:
            return self.drain_pending_graph_compilations(
                max_turns=max_turns,
                turn_ids=turn_ids,
            )
        except Exception as error:
            self.last_graph_compilation_error = (
                f"{type(error).__module__}.{type(error).__qualname__}"
            )
            return []

    def conversation_graph(self) -> IncrementalConversationGraph:
        """Load the resident graph once, then adopt only later durable deltas."""

        return self._conversation_graphs.graph()

    def pending_ingest_stats(self) -> dict[str, int | float | str | None]:
        """Project bounded service telemetry from durable pending receipts."""
        row = self._db.execute(
            "SELECT COUNT(DISTINCT p.turn_id), COUNT(r.chunk_id), "
            "COALESCE(SUM(r.token_count), 0), MIN(p.created_at), "
            "(SELECT COUNT(*) FROM pending_ingest_attempts AS a "
            " JOIN pending_ingests AS failed ON failed.turn_id = a.turn_id "
            " WHERE failed.status = 'pending'), "
            "(SELECT a.last_error_kind FROM pending_ingest_attempts AS a "
            " JOIN pending_ingests AS failed ON failed.turn_id = a.turn_id "
            " WHERE failed.status = 'pending' "
            " ORDER BY failed.created_at, failed.turn_id LIMIT 1) "
            "FROM pending_ingests AS p "
            "LEFT JOIN ingest_chunk_reservations AS r "
            "ON r.turn_id = p.turn_id WHERE p.status = 'pending'"
        ).fetchone()
        manifest_count = int(row[0])
        oldest_at = (
            datetime.fromisoformat(str(row[3]))
            if row[3] is not None
            else None
        )
        oldest_age_seconds = (
            max(
                0.0,
                (
                    datetime.now(timezone.utc)
                    - (
                        oldest_at.replace(tzinfo=timezone.utc)
                        if oldest_at.tzinfo is None
                        else oldest_at.astimezone(timezone.utc)
                    )
                ).total_seconds(),
            )
            if oldest_at is not None
            else None
        )
        return {
            "manifest_count": manifest_count,
            "chunk_count": int(row[1]),
            "token_count": int(row[2]),
            "oldest_age_seconds": oldest_age_seconds,
            "failed_count": int(row[4]),
            "oldest_error_kind": None if row[5] is None else str(row[5]),
        }

    def _pending_manifest_batch(
        self,
        *,
        max_manifests: int | None,
        max_chunks: int | None,
        max_tokens: int | None,
    ) -> list[PendingIngestManifest]:
        """Stream only enough ordered receipt rows to fill one drain batch."""
        now = datetime.now(timezone.utc).isoformat()
        retry_class = self._pending_ingests.choose_retry_class(now)
        if retry_class is None:
            return []

        def select(
            selected_retry_class: bool,
        ) -> tuple[
            list[PendingIngestManifest], tuple[str, BaseException] | None
        ]:
            statement = (
                "SELECT p.turn_id, p.manifest_sha256, p.manifest_json, "
                "COALESCE(a.attempt_count, 0), a.last_attempt_at "
                "FROM pending_ingests AS p "
                "JOIN turns AS t ON t.turn_id = p.turn_id "
                "LEFT JOIN pending_ingest_attempts AS a "
                "ON a.turn_id = p.turn_id "
                "WHERE p.status = 'pending' "
            )
            if selected_retry_class:
                statement += (
                    "AND a.attempt_count > 0 AND a.next_attempt_at <= ? "
                )
                parameters: tuple[object, ...] = (now,)
                statement += "ORDER BY a.next_attempt_at, a.last_attempt_at, "
                statement += "t.ordinal, p.turn_id"
            else:
                statement += "AND a.turn_id IS NULL "
                parameters = ()
                statement += "ORDER BY t.ordinal, p.turn_id"
            if max_manifests is not None:
                statement += " LIMIT ?"
                parameters += (max_manifests,)

            manifests: list[PendingIngestManifest] = []
            selected_chunks = 0
            selected_tokens = 0
            selection_error: tuple[str, BaseException] | None = None
            cursor = self._db.execute(statement, parameters)
            try:
                retry_cohort: tuple[int, str] | None = None
                retry_singleton = False
                for (
                    turn_id,
                    manifest_sha256,
                    manifest_json,
                    attempt_count,
                    last_attempt_at,
                ) in cursor:
                    attempts = int(attempt_count)
                    candidate_cohort = (attempts, str(last_attempt_at))
                    if manifests and (
                        retry_singleton
                        or (
                            retry_cohort is not None
                            and candidate_cohort != retry_cohort
                        )
                    ):
                        break
                    try:
                        manifest = PendingIngestManifest.from_json(str(manifest_json))
                        if (
                            manifest.turn_id != str(turn_id)
                            or manifest.sha256 != str(manifest_sha256)
                        ):
                            raise ValueError(
                                "pending ingest manifest receipt is inconsistent"
                            )
                    except BaseException as exc:
                        selection_error = (str(turn_id), exc)
                        break
                    manifest_chunks = len(manifest.chunks)
                    manifest_tokens = sum(
                        chunk.token_count for chunk in manifest.chunks
                    )
                    exceeds_work_bound = (
                        max_chunks is not None
                        and selected_chunks + manifest_chunks > max_chunks
                    ) or (
                        max_tokens is not None
                        and selected_tokens + manifest_tokens > max_tokens
                    )
                    if manifests and exceeds_work_bound:
                        break
                    manifests.append(manifest)
                    if selected_retry_class and len(manifests) == 1:
                        # One whole failed provider batch gets one whole-batch
                        # retry. A second failure dissolves it into singletons
                        # so one poison turn cannot monopolize recovery.
                        if attempts == 1:
                            retry_cohort = candidate_cohort
                        else:
                            retry_singleton = True
                    selected_chunks += manifest_chunks
                    selected_tokens += manifest_tokens
            finally:
                cursor.close()
            return manifests, selection_error

        manifests, selection_error = select(retry_class)
        if not manifests and selection_error is None:
            # The selected class can become empty after the fairness toggle is
            # committed. Do useful work from the opposite eligible class in
            # this tick rather than surfacing a false-empty queue.
            manifests, selection_error = select(not retry_class)
        if selection_error is not None:
            failed_turn_id, error = selection_error
            self._pending_ingests.record_failure([failed_turn_id], error)
            raise error
        return manifests

    def drain_pending_ingests(
        self,
        *,
        max_manifests: int | None = None,
        max_chunks: int | None = None,
        max_tokens: int | None = None,
        enrich: bool = True,
    ) -> list[tuple[Turn, list[Chunk]]]:
        """Complete one bounded batch from the durable pending journal.

        A manifest is indivisible because its receipt seals the complete chunk
        topology. Bounds therefore stop before the next manifest, while the
        first pending manifest is always admitted even when it alone exceeds a
        chunk or token bound. This guarantees finite forward progress without
        partially publishing a turn. When ``auto_extract`` is enabled, memory
        extraction follows indexing in captured turn order unless ``enrich``
        is false. The latter lets a scheduler run T1 and bounded T2 recovery as
        separate stages without re-embedding. Enrichment receipts are still
        claimed before T1 so deferred work remains durable.

        Concurrent independent helpers are safe but at-least-once: they can
        select and embed the same pending manifest before one finalizes it.
        Calls sharing one condenser instance must be serialized by the caller.
        """
        if self._db.read_only:
            raise sqlite3.OperationalError("attempt to write a readonly database")
        if type(enrich) is not bool:
            raise ValueError("enrich must be a boolean")
        for name, value in (
            ("max_manifests", max_manifests),
            ("max_chunks", max_chunks),
            ("max_tokens", max_tokens),
        ):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")

        manifests = self._pending_manifest_batch(
            max_manifests=max_manifests,
            max_chunks=max_chunks,
            max_tokens=max_tokens,
        )

        if not manifests:
            self._drain_pending_envelopes_fail_open(
                max_turns=min(
                    _IDLE_ENVELOPE_RETRY_MAX_TURNS
                    if max_manifests is None
                    else max_manifests,
                    CONVERSATION_ENVELOPE_HARD_MAX_TURNS,
                )
            )
            # An earlier graph failure can outlive the T1 queue. Give the
            # independent derived stage a bounded retry on otherwise-idle
            # completion ticks.
            self._drain_pending_graph_fail_open(
                max_turns=(
                    _IDLE_GRAPH_RETRY_MAX_TURNS
                    if max_manifests is None
                    else max_manifests
                )
            )
            return []

        if self._auto_extract:
            # The draining instance explicitly elects automatic extraction.
            # Persist that obligation before embedding/indexing so a provider
            # or process failure cannot leave a searchable turn with no
            # recoverable enrichment receipt. This adopts v13-era and
            # auto_extract=False captures without guessing during migration.
            connection = self._db.connection
            try:
                connection.execute("BEGIN IMMEDIATE")
                self._pending_enrichments.claim_many(manifests)
                connection.commit()
            except BaseException as exc:
                connection.rollback()
                self._pending_ingests.record_failure(
                    [manifest.turn_id for manifest in manifests], exc
                )
                raise

        staged: list[tuple[Turn, list[Chunk]]] = []
        for manifest in manifests:
            try:
                turn = self._transcript.get_turn(manifest.turn_id)
                if turn is None:
                    raise RuntimeError("pending ingest references an unknown turn")
                chunks = manifest.reconstruct(turn)
            except BaseException as exc:
                self._pending_ingests.record_failure([manifest.turn_id], exc)
                raise
            staged.append((turn, chunks))

        by_turn = {manifest.turn_id: manifest for manifest in manifests}
        completed = self._complete_captured_ingests(
            staged,
            by_turn,
            {manifest.turn_id: "pending" for manifest in manifests},
        )
        self._drain_pending_envelopes_fail_open(
            max_turns=min(len(completed), CONVERSATION_ENVELOPE_HARD_MAX_TURNS),
            turn_ids=tuple(turn.turn_id for turn, _chunks in completed),
        )
        self._drain_pending_graph_fail_open(
            max_turns=len(completed),
            turn_ids=tuple(turn.turn_id for turn, _chunks in completed),
        )
        if self._auto_extract and enrich:
            for turn, _chunks in completed:
                self._enrich_captured_turn(turn)
        return completed

    def pending_enrichment_count(self) -> int:
        """Return turns with an outstanding automatic-extraction obligation.

        This includes source captures still waiting for base indexing as well
        as indexed turns ready for :meth:`drain_pending_enrichments`.
        """
        return self._pending_enrichments.count()

    def pending_enrichment_stats(self) -> dict[str, int | float | str | None]:
        """Project total/ready T2 obligations and the oldest durable age."""

        sampled_at = datetime.now(timezone.utc)
        row = self._db.execute(
            "SELECT COUNT(*), "
            "COALESCE(SUM(CASE WHEN p.status = 'indexed' "
            " AND NOT EXISTS (SELECT 1 FROM "
            " pending_enrichment_legacy_quarantine AS q "
            " WHERE q.turn_id = e.turn_id) "
            " AND (s.next_attempt_at IS NULL OR s.next_attempt_at <= ?) "
            " THEN 1 ELSE 0 END), 0), "
            "MIN(e.created_at), "
            "COALESCE(SUM(CASE WHEN s.attempt_count > 0 THEN 1 ELSE 0 END), 0), "
            "(SELECT state.last_error_kind FROM pending_enrichment_state AS state "
            " JOIN pending_enrichments AS failed ON failed.turn_id = state.turn_id "
            " WHERE failed.status = 'pending' AND state.attempt_count > 0 "
            " ORDER BY failed.created_at, failed.turn_id LIMIT 1), "
            "(SELECT COUNT(*) FROM pending_enrichment_dispositions "
            " WHERE disposition = 'discarded_legacy'), "
            "(SELECT COUNT(*) FROM pending_corrections "
            " WHERE status = 'pending'), "
            "(SELECT COUNT(*) FROM pending_enrichment_legacy_quarantine AS q "
            " JOIN pending_enrichments AS legacy ON legacy.turn_id = q.turn_id "
            " JOIN pending_ingests AS legacy_ingest "
            " ON legacy_ingest.turn_id = legacy.turn_id "
            " WHERE legacy.status = 'pending' "
            " AND legacy_ingest.status = 'indexed' "
            ") "
            "FROM pending_enrichments AS e "
            "JOIN pending_ingests AS p ON p.turn_id = e.turn_id "
            "JOIN turns AS t ON t.turn_id = e.turn_id "
            "LEFT JOIN pending_enrichment_state AS s ON s.turn_id = e.turn_id "
            "WHERE e.status = 'pending'",
            (sampled_at.isoformat(),),
        ).fetchone()
        oldest_at = (
            datetime.fromisoformat(str(row[2]))
            if row[2] is not None
            else None
        )
        oldest_age_seconds = (
            max(
                0.0,
                (
                    sampled_at
                    - (
                        oldest_at.replace(tzinfo=timezone.utc)
                        if oldest_at.tzinfo is None
                        else oldest_at.astimezone(timezone.utc)
                    )
                ).total_seconds(),
            )
            if oldest_at is not None
            else None
        )
        return {
            "turn_count": int(row[0]),
            "ready_count": int(row[1]),
            "oldest_age_seconds": oldest_age_seconds,
            "failed_count": int(row[3]),
            "oldest_error_kind": None if row[4] is None else str(row[4]),
            "discarded_legacy_count": int(row[5]),
            "deferred_correction_count": int(row[6]),
            "legacy_quarantined_count": int(row[7]),
        }

    def pending_deferred_corrections(self) -> list[dict[str, object]]:
        """List independent correction receipts and current evidence liveness."""
        pending: list[dict[str, object]] = []
        for record in self._pending_enrichments.list_corrections():
            turn = self._transcript.get_turn(record.turn_id)
            grounded = False
            if turn is not None:
                report = self._validator.validate_for_enrichment(
                    MemoryOps(create=[record.operation]),
                    turn,
                    self._live_enrichment_chunks(record.turn_id),
                    allow_unbound_corrections=True,
                )
                grounded = bool(report.accepted.create)
            pending.append(
                {
                    "correction_id": record.correction_id,
                    "turn_id": record.turn_id,
                    "operation_index": record.operation_index,
                    "correction": record.operation,
                    "grounded": grounded,
                }
            )
        return pending

    def _require_applicable_correction_target(
        self, correction: CreateOp, target: MemoryItem
    ) -> None:
        """Reject a no-op or an already-canonical replacement identity."""
        target_identity = content_key(target.type, target.content)
        if content_key(target.type, correction.content) == target_identity:
            raise DeferredCorrectionTargetStaleError(
                "correction would not change the selected target"
            )
        collision = self._memory.find_by_content(
            correction.type, correction.content
        )
        if collision is not None and collision.mem_id != target.mem_id:
            raise DeferredCorrectionTargetStaleError(
                "correction replacement already exists as active memory: "
                f"{collision.mem_id}"
            )

    def resolve_deferred_correction(
        self, correction_id: str, target_mem_id: str
    ) -> MemoryItem:
        """Atomically bind one grounded correction to one reviewed target."""
        record = self._pending_enrichments.get_correction(correction_id)
        if record is None or record.status != "pending":
            raise ValueError("correction is not pending")
        target = self._memory.get(target_mem_id)
        if target is None or target.status is not MemoryStatus.ACTIVE:
            raise DeferredCorrectionTargetStaleError(
                f"correction target is not active: {target_mem_id}"
            )
        expected_content_hash = content_key(target.type, target.content)
        expected_revision = _correction_target_revision(target)
        turn = self._transcript.get_turn(record.turn_id)
        if turn is None:
            raise RuntimeError("deferred correction source turn is missing")

        for _attempt in range(4):
            current_chunks = self._live_enrichment_chunks(record.turn_id)
            applicable = self._validator.validate_for_enrichment(
                MemoryOps(create=[record.operation]),
                turn,
                current_chunks,
                allow_unbound_corrections=True,
            ).accepted.create
            if len(applicable) != 1:
                raise RuntimeError(
                    "deferred correction is no longer grounded in live evidence"
                )
            correction = applicable[0]
            self._require_applicable_correction_target(correction, target)
            [vector] = self._memory.prepare_supersede_embeddings([correction])
            connection = self._db.connection
            try:
                connection.execute("BEGIN IMMEDIATE")
                self._pending_enrichments.require_pending_correction(
                    correction_id, record.operation_sha256
                )
                locked_chunks = self._live_enrichment_chunks(record.turn_id)
                if [chunk.chunk_id for chunk in locked_chunks] != [
                    chunk.chunk_id for chunk in current_chunks
                ]:
                    connection.rollback()
                    continue
                locked_target = self._memory.get(target_mem_id)
                if (
                    locked_target is None
                    or locked_target.status is not MemoryStatus.ACTIVE
                    or _correction_target_revision(locked_target)
                    != expected_revision
                ):
                    raise DeferredCorrectionTargetStaleError(
                        "correction target changed after operator review: "
                        f"{target_mem_id}"
                    )
                self._require_applicable_correction_target(
                    correction, locked_target
                )
                successor = self._memory.supersede(
                    SupersedeOp(
                        mem_id=target_mem_id,
                        replacement=correction,
                    ),
                    _commit=False,
                    _resolved_embedding=vector,
                    _expected_content_hash=expected_content_hash,
                )
                if successor is None:
                    raise DeferredCorrectionTargetStaleError(
                        f"correction target is not active: {target_mem_id}"
                    )
                self._pending_enrichments.resolve_correction(
                    correction_id,
                    target_mem_id=target_mem_id,
                    successor_mem_id=successor.mem_id,
                )
                connection.commit()
                return self._memory.get(successor.mem_id) or successor
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise
        raise RuntimeError("live correction evidence changed repeatedly")

    def dismiss_deferred_correction(self, correction_id: str) -> bool:
        """Explicitly dismiss one correction; evidence may already be retired."""
        record = self._pending_enrichments.get_correction(correction_id)
        if record is None or record.status != "pending":
            return False
        connection = self._db.connection
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._pending_enrichments.require_pending_correction(
                correction_id, record.operation_sha256
            )
            self._pending_enrichments.dismiss_correction(correction_id)
            connection.commit()
            return True
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise

    def unbound_legacy_retirements(self) -> list[MemoryItem]:
        """List legacy retired identities needing a source-order binding."""
        return self._memory.unbound_legacy_retirements()

    def bind_legacy_retirement(
        self, mem_id: str, *, retired_at_turn: int
    ) -> bool:
        """Bind one legacy retirement within the recorded migration boundary."""
        return self._memory.bind_legacy_retirement(
            mem_id, retired_at_turn=retired_at_turn
        )

    def pending_legacy_enrichments(self) -> list[dict[str, object]]:
        """Discover pre-v15 T2 receipts quarantined from automatic replay."""
        rows = self._db.execute(
            "SELECT e.turn_id, t.ordinal, q.migration_boundary "
            "FROM pending_enrichments AS e "
            "JOIN turns AS t ON t.turn_id = e.turn_id "
            "JOIN pending_ingests AS p ON p.turn_id = e.turn_id "
            "JOIN pending_enrichment_legacy_quarantine AS q "
            "ON q.turn_id = e.turn_id "
            "WHERE e.status = 'pending' AND p.status = 'indexed' "
            "ORDER BY t.ordinal, e.turn_id"
        ).fetchall()
        return [
            {
                "turn_id": str(turn_id),
                "source_ordinal": int(ordinal),
                "migration_boundary": int(migration_boundary),
            }
            for turn_id, ordinal, migration_boundary in rows
        ]

    def discard_legacy_pending_enrichment(self, turn_id: str) -> bool:
        """Explicitly retire unsafe legacy T2 work; its T1 chunks stay searchable."""
        row = self._db.execute(
            "SELECT t.ordinal, e.status, p.status FROM turns AS t "
            "JOIN pending_enrichments AS e ON e.turn_id = t.turn_id "
            "JOIN pending_ingests AS p ON p.turn_id = t.turn_id "
            "JOIN pending_enrichment_legacy_quarantine AS q "
            "ON q.turn_id = t.turn_id WHERE t.turn_id = ?",
            (turn_id,),
        ).fetchone()
        if (
            row is None
            or str(row[1]) != "pending"
            or str(row[2]) != "indexed"
        ):
            return False
        connection = self._db.connection
        try:
            connection.execute("BEGIN IMMEDIATE")
            locked = self._db.execute(
                "SELECT t.ordinal FROM turns AS t "
                "JOIN pending_enrichments AS e ON e.turn_id = t.turn_id "
                "JOIN pending_ingests AS p ON p.turn_id = t.turn_id "
                "JOIN pending_enrichment_legacy_quarantine AS q "
                "ON q.turn_id = t.turn_id "
                "WHERE t.turn_id = ? AND e.status = 'pending' "
                "AND p.status = 'indexed'",
                (turn_id,),
            ).fetchone()
            if locked is None:
                connection.rollback()
                return False
            self._pending_enrichments.record_legacy_disposition(turn_id)
            self._pending_enrichments.finalize(turn_id)
            if self._pending_enrichments.staged_result(turn_id) is not None:
                self._pending_enrichments.clear_staged_result(turn_id)
            connection.commit()
            return True
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise

    def _indexed_turn_chunks(self, turn_id: str) -> list[Chunk]:
        """Hydrate one indexed turn without invoking the embedding provider."""
        rows = self._db.execute(
            "SELECT chunk_id, text, start_char, end_char, token_count, embedding, "
            "lexical_weights FROM chunks WHERE turn_id = ? "
            "ORDER BY start_char, end_char, chunk_id",
            (turn_id,),
        ).fetchall()
        return [
            Chunk(
                chunk_id=str(row[0]),
                turn_id=turn_id,
                text=str(row[1]),
                start_char=int(row[2]),
                end_char=int(row[3]),
                token_count=int(row[4]),
                embedding=(
                    np.frombuffer(row[5], dtype=np.float32).tolist()
                    if row[5] is not None
                    else None
                ),
                lexical_weights=(json.loads(str(row[6])) if row[6] else None),
            )
            for row in rows
        ]

    def _enrichment_turn_chunks(self, turn_id: str) -> tuple[list[Chunk], bool]:
        """Hydrate only live chunks and report whether any were retired."""
        rows = self._db.execute(
            "SELECT chunk_id, text, start_char, end_char, token_count, embedding, "
            "lexical_weights, hnsw_label IS NOT NULL, term_count IS NOT NULL "
            "FROM chunks WHERE turn_id = ? "
            "ORDER BY start_char, end_char, chunk_id",
            (turn_id,),
        ).fetchall()
        chunks: list[Chunk] = []
        retired = False
        for row in rows:
            if row[5] is None or not bool(row[7]) or not bool(row[8]):
                retired = True
                continue
            chunks.append(
                Chunk(
                    chunk_id=str(row[0]),
                    turn_id=turn_id,
                    text=str(row[1]),
                    start_char=int(row[2]),
                    end_char=int(row[3]),
                    token_count=int(row[4]),
                    embedding=np.frombuffer(row[5], dtype=np.float32).tolist(),
                    lexical_weights=(
                        json.loads(str(row[6])) if row[6] else None
                    ),
                )
            )
        return chunks, retired

    def _enrich_one(self, turn: Turn, chunks: list[Chunk]) -> None:
        """Stage one deterministic result, replay it, and seal its receipt."""
        status = self._pending_enrichments.status(turn.turn_id)
        if status is None:
            raise RuntimeError("automatic extraction has no enrichment receipt")
        if status == "enriched":
            return
        pre_stage_provider_work = False
        try:
            source_row = self._db.execute(
                "SELECT ordinal FROM turns WHERE turn_id = ?",
                (turn.turn_id,),
            ).fetchone()
            if source_row is None:
                raise RuntimeError("automatic extraction source turn is missing")
            source_ordinal = int(source_row[0])
            if self._pending_enrichments.is_legacy_quarantined(turn.turn_id):
                raise AmbiguousLegacyEnrichmentError(
                    "pre-v15 pending enrichment requires explicit disposal; "
                    "its indexed T1 evidence remains searchable"
                )
            staged = self._pending_enrichments.staged_result(turn.turn_id)
            if staged is None:
                pre_stage_provider_work = True
                if chunks:
                    proposed = self._extract_memory_ops_for_enrichment(
                        [turn], chunks
                    )
                    ops = self._validator.validate_for_enrichment(
                        proposed,
                        turn,
                        chunks,
                        allow_unbound_corrections=True,
                    ).accepted
                else:
                    ops = MemoryOps()
                staged = self._pending_enrichments.stage_ops(
                    turn.turn_id,
                    ops,
                    [chunk.chunk_id for chunk in chunks],
                )
                pre_stage_provider_work = False
                if staged is None:
                    return
            ops, _staged_chunk_ids = staged

            # Embedding is provider work and must never hold SQLite's writer
            # lock. Validate against a fresh live view and resolve vectors
            # first; after BEGIN IMMEDIATE, retry the plan if retirement won
            # the race. Chunk retirement is monotonic, so this loop converges.
            for _attempt in range(4):
                current_chunks = self._live_enrichment_chunks(turn.turn_id)
                applicable = self._validator.validate_for_enrichment(
                    ops,
                    turn,
                    current_chunks,
                    allow_unbound_corrections=True,
                ).accepted
                corrections = tuple(
                    op
                    for op in applicable.create
                    if op.type is MemoryType.CORRECTION
                )
                publishable = MemoryOps(
                    create=[
                        op
                        for op in applicable.create
                        if op.type is not MemoryType.CORRECTION
                    ]
                )
                prepared = self._memory.prepare_create_embeddings(
                    publishable, source_ordinal=source_ordinal
                )

                connection = self._db.connection
                connection.execute("BEGIN IMMEDIATE")
                # Recheck only after owning the write lock. A concurrent helper
                # may have applied this same staged result and sealed the receipt
                # while this helper was resolving vectors.
                locked_status = self._pending_enrichments.status(turn.turn_id)
                if locked_status == "enriched":
                    connection.commit()
                    return
                if locked_status != "pending":
                    raise RuntimeError("automatic extraction receipt became invalid")
                locked_chunks = self._live_enrichment_chunks(turn.turn_id)
                if [chunk.chunk_id for chunk in locked_chunks] != [
                    chunk.chunk_id for chunk in current_chunks
                ]:
                    connection.rollback()
                    continue
                try:
                    self._memory.apply(
                        publishable,
                        _commit=False,
                        _prepared_embeddings=prepared,
                    )
                except PreparedMemoryPlanStaleError:
                    connection.rollback()
                    continue
                self._pending_enrichments.queue_corrections(
                    turn.turn_id, corrections
                )
                self._pending_enrichments.finalize(turn.turn_id)
                # The per-operation correction queue now owns every unresolved
                # reversal, so the transient whole-turn replay payload can be
                # cleared after every successful T2 completion.
                self._pending_enrichments.clear_staged_result(turn.turn_id)
                connection.commit()
                return
            raise RuntimeError("live enrichment evidence changed repeatedly")
        except BaseException as exc:
            if self._db.connection.in_transaction:
                self._db.connection.rollback()
            self._pending_enrichments.record_failure(
                turn.turn_id,
                exc,
                ignore_if_staged=pre_stage_provider_work,
            )
            raise

    def _enrich_captured_turn(self, turn: Turn) -> list[Chunk]:
        """Build the live T2 view, recording any pre-extraction poison."""
        # A receipt adopted from a pre-v15 source is intentionally
        # quarantined, not failed. T1 is complete and searchable; an operator
        # must explicitly dispose of ambiguous T2 work. Centralizing this
        # guard covers synchronous ingest retries and both drain entrypoints.
        if self._pending_enrichments.is_legacy_quarantined(turn.turn_id):
            return []
        try:
            enrichment_turn, chunks = self._enrichment_view(turn)
        except BaseException as exc:
            self._pending_enrichments.record_failure(turn.turn_id, exc)
            raise
        self._enrich_one(enrichment_turn, chunks)
        return chunks

    def _live_enrichment_chunks(self, turn_id: str) -> list[Chunk]:
        """Read the T2 evidence view without hydrating vector BLOBs."""
        return [
            Chunk(
                chunk_id=str(row[0]),
                turn_id=turn_id,
                text=str(row[1]),
                start_char=int(row[2]),
                end_char=int(row[3]),
                token_count=int(row[4]),
            )
            for row in self._db.execute(
                "SELECT chunk_id, text, start_char, end_char, token_count "
                "FROM chunks WHERE turn_id = ? "
                "AND embedding IS NOT NULL AND hnsw_label IS NOT NULL "
                "AND term_count IS NOT NULL ORDER BY chunk_id",
                (turn_id,),
            ).fetchall()
        ]

    def _enrichment_view(self, turn: Turn) -> tuple[Turn, list[Chunk]]:
        """Confine extraction to the turn's currently live indexed chunks."""
        chunks, retired = self._enrichment_turn_chunks(turn.turn_id)
        if not retired:
            return turn, chunks
        return (
            turn.model_copy(
                update={"text": "\n".join(chunk.text for chunk in chunks)}
            ),
            chunks,
        )

    def drain_pending_enrichments(
        self, *, max_turns: int | None = None
    ) -> list[tuple[Turn, list[Chunk]]]:
        """Enrich indexed turns one at a time in ordinal order.

        Extraction is at-least-once: a process can apply memory operations and
        fail before sealing its receipt. Deferred extraction observes the
        drain-time current-turn/heat state, not synchronous causal visibility.
        A failure stops the pass, leaving that turn and every unattempted turn
        pending for restart recovery without re-embedding source chunks.
        """
        if self._db.read_only:
            raise sqlite3.OperationalError("attempt to write a readonly database")
        if max_turns is not None and (type(max_turns) is not int or max_turns < 1):
            raise ValueError("max_turns must be a positive integer or None")
        completed: list[tuple[Turn, list[Chunk]]] = []
        for turn_id in self._pending_enrichments.pending_turn_ids(
            max_turns=max_turns
        ):
            try:
                turn = self._transcript.get_turn(turn_id)
                if turn is None:
                    raise RuntimeError(
                        "pending enrichment references an unknown turn"
                    )
            except BaseException as exc:
                self._pending_enrichments.record_failure(turn_id, exc)
                raise
            chunks = self._enrich_captured_turn(turn)
            completed.append((turn, chunks))
        return completed

    def recover_pending_ingests(
        self,
        *,
        max_manifests: int | None = None,
        max_chunks: int | None = None,
        max_tokens: int | None = None,
    ) -> list[tuple[Turn, list[Chunk]]]:
        """Replay one bounded base-index batch without changing legacy semantics."""
        return self.drain_pending_ingests(
            max_manifests=max_manifests,
            max_chunks=max_chunks,
            max_tokens=max_tokens,
            enrich=False,
        )

    def ingest_transcript(
        self,
        transcript: "TranscriptFile",
        *,
        only_pending: bool = True,
    ) -> dict[str, object]:
        """Ingest a vendor chat export, re-reading only what changed.

        ``transcript`` owns the byte index; this refreshes it, ingests the
        conversations that are new or edited since the last call, and returns
        a summary. With ``only_pending=False`` every conversation is ingested
        regardless of the delta, which is the correct choice for a fresh store.

        Message IDs become turn IDs and conversation IDs become source IDs, so
        re-ingesting an edited conversation replays the same identities rather
        than duplicating the history under fresh ones.
        """

        delta = transcript.refresh()
        index = transcript.index
        if index is None:
            raise RuntimeError("transcript refresh completed without an index")
        spans = delta.pending if only_pending else index.spans
        messages = list(transcript.iter_messages(spans))
        records = [message.as_ingest_record() for message in messages]
        ingested = self.ingest_many(records) if records else []
        return {
            "path": str(index.path),
            "layout": index.layout,
            "sha256": index.sha256,
            "byte_size": index.byte_size,
            "status": delta.status,
            "conversations_indexed": len(index.spans),
            "conversations_ingested": len(spans),
            "messages_ingested": len(ingested),
            "removed_conversations": list(delta.removed),
        }

    def compile_cav_signatures(
        self,
        linker: object,
        artifact: AssociationArtifact,
        chunks: Sequence[Chunk | RetrievalResult],
        *,
        batch_size: int = 8,
        overwrite: bool = False,
    ) -> dict[str, int]:
        """Compile event/concept memberships into bounded durable scalars.

        The Qwen prefix acts only as a write-time teacher.  No residual,
        attention matrix, token sequence, or K/V cache is stored; each chunk
        contributes exactly one float32 value per named concept.
        """

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        bank = getattr(linker, "cav_bank", None)
        compile_many = getattr(linker, "signatures", None)
        if bank is None or compile_many is None:
            raise ValueError("linker must expose a CAV bank and batched signatures")
        if tuple(bank.names) != artifact.concept_names:
            raise ValueError("linker and artifact concept names do not match")
        if int(bank.layer) != artifact.cav_layer:
            raise ValueError("linker and artifact CAV layers do not match")
        self._associations.register_artifact(artifact)

        unique: dict[str, Chunk] = {}
        for value in chunks:
            chunk = value.chunk if isinstance(value, RetrievalResult) else value
            unique.setdefault(chunk.chunk_id, chunk)
        pending = [
            chunk
            for chunk in unique.values()
            if overwrite
            or not self._associations.has_signature(
                chunk.chunk_id, artifact.artifact_id
            )
        ]
        span_texts: list[str] = []
        span_owners: list[str] = []
        for chunk in pending:
            spans = self._chunker.conceptual_spans(chunk.text)
            for span in spans or [chunk.text]:
                span_texts.append(span)
                span_owners.append(chunk.chunk_id)
        span_signatures = compile_many(
            span_texts,
            batch_size=batch_size,
        )
        if len(span_signatures) != len(span_texts):
            raise ValueError("linker returned a misaligned signature batch")
        pooled: dict[str, tuple[float, ...]] = {}
        for chunk_id, signature in zip(
            span_owners, span_signatures, strict=True
        ):
            values = tuple(float(value) for value in signature)
            previous = pooled.get(chunk_id)
            pooled[chunk_id] = (
                values
                if previous is None
                else tuple(
                    max(left, right)
                    for left, right in zip(previous, values, strict=True)
                )
            )
        written = self._associations.put_signatures(
            artifact.artifact_id,
            [
                (chunk.chunk_id, pooled[chunk.chunk_id])
                for chunk in pending
            ],
        )
        return {
            "requested": len(unique),
            "compiled": written,
            "reused": len(unique) - written,
            "compiled_spans": len(span_texts),
            "signature_width": len(artifact.concept_names),
            # Canonical invariant: no request-derived token IDs, Q/K/V,
            # attention maps, residuals, or generation K/V survive the pass.
            # Reusable checkpoint weights/tokenizer assets are not request
            # state and are deliberately outside this metric.
            "retained_request_token_state_bytes": 0,
            # Compatibility alias retained for old reports.
            "retained_token_state_bytes": 0,
        }

    def compile_indexed_cav_signatures(
        self,
        linker: object,
        artifact: AssociationArtifact,
        *,
        batch_size: int = 8,
        overwrite: bool = False,
        roles: Sequence[str] = ("user", "assistant", "system"),
    ) -> dict[str, int]:
        """Compile every active indexed chunk without hydrating embeddings."""

        selected_roles = tuple(dict.fromkeys(str(role) for role in roles))
        invalid_roles = set(selected_roles) - {"user", "assistant", "system"}
        if not selected_roles or invalid_roles:
            raise ValueError("roles must contain valid transcript roles")
        placeholders = ",".join("?" for _ in selected_roles)
        rows = self._db.execute(
            "SELECT c.chunk_id, c.turn_id, c.text, c.start_char, c.end_char, "
            "c.token_count FROM chunks AS c "
            "JOIN turns AS t ON t.turn_id = c.turn_id "
            f"WHERE {INDEXED_CHUNK_SQL} "
            f"AND t.role IN ({placeholders}) ORDER BY c.hnsw_label",
            selected_roles,
        ).fetchall()
        chunks = [
            Chunk(
                chunk_id=str(row[0]),
                turn_id=str(row[1]),
                text=str(row[2]),
                start_char=int(row[3]),
                end_char=int(row[4]),
                token_count=int(row[5]),
            )
            for row in rows
        ]
        return self.compile_cav_signatures(
            linker,
            artifact,
            chunks,
            batch_size=batch_size,
            overwrite=overwrite,
        )

    def extract_memory(
        self, turns: list[Turn], chunks: list[Chunk] | None = None
    ) -> dict[str, int]:
        """Propose, validate, and apply memory ops for the given turns.

        Ops whose provenance cannot be verified against the transcript are
        rejected — an LLM cannot write a memory it did not quote.
        """
        ops = self._extractor.extract(turns, chunks)
        return self._apply_extracted_memory_ops(ops)

    def _extract_memory_ops_for_enrichment(
        self, turns: list[Turn], chunks: list[Chunk]
    ) -> MemoryOps:
        """Propose ops through an extractor's failure-signaling path."""
        extract_durable = getattr(
            self._extractor, "extract_durable_for_enrichment", None
        ) or getattr(self._extractor, "extract_durable", None)
        # Legacy/custom Extractor implementations declare their returned
        # MemoryOps successful. Provider adapters that can fail ambiguously
        # should implement DurableExtractor so transport/schema failure raises
        # instead of masquerading as an empty result.
        ops = (
            self._extractor.extract(turns, chunks)
            if extract_durable is None
            else extract_durable(turns, chunks)
        )
        if not isinstance(ops, MemoryOps):
            raise TypeError("extractor must return MemoryOps")
        if ops.update or ops.supersede or ops.delete or ops.pin:
            raise ExtractionUnavailableError(
                "deferred enrichment returned forbidden non-create operations"
            )
        return ops

    def _apply_extracted_memory_ops(self, ops: MemoryOps) -> dict[str, int]:
        """Validate and apply one provider-independent operation result."""
        if ops.is_empty():
            return {}
        report = self._validator.validate(ops)
        return self._memory.apply(report)

__all__ = ["IngestWorkflowMixin"]
