"""Stateful ingestion and indexed-signature workflows for the condenser."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Sequence

from memory_condense.associations.association_store import AssociationArtifact
from memory_condense.domain.schemas import Chunk, RetrievalResult, Turn
from memory_condense.ingest.transcript_source import TranscriptFile
from memory_condense.persistence.db import INDEXED_CHUNK_SQL
from memory_condense.persistence.pending_ingest_store import (
    PendingIngestAlreadyIndexedError,
    PendingIngestManifest,
)


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
        surviving ops.
        """
        if self._db.read_only:
            raise sqlite3.OperationalError("attempt to write a readonly database")
        turn = self._transcript.stage(
            role,
            text,
            source_id=source_id,
            created_at=created_at,
            turn_id=turn_id,
        )
        chunks = self._chunker.chunk_turn(turn.turn_id, text)
        if turn_id is not None:
            chunks = _bind_explicit_chunk_ids(turn.turn_id, chunks)
        manifest = PendingIngestManifest.build(turn, chunks)
        if chunks:
            chunks = _embed_source_chunks(self._embedder, chunks)

        published, ingest_status = self._publish_staged_turns(
            [(turn, created_at is not None)],
            {turn.turn_id: manifest},
        )
        turn = published[turn.turn_id]
        if chunks and ingest_status[turn.turn_id] == "pending":
            self._index_pending_manifests(
                chunks,
                [manifest],
            )

        if self._auto_extract:
            self.extract_memory([turn], chunks)

        return turn, chunks

    def ingest_many(
        self,
        turns: Sequence[
            tuple[str, str, str | None]
            | tuple[str, str, str | None, datetime | None]
            | tuple[str, str, str | None, datetime | None, str]
        ],
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
        records: list[
            tuple[str, str, str | None, datetime | None, str | None]
        ] = []
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

        staged: list[tuple[Turn, list[Chunk]]] = []
        publication_requests: list[tuple[Turn, bool]] = []
        flat_chunks: list[Chunk] = []
        staged_explicit_turns: dict[str, tuple[Turn, bool]] = {}
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
        unique_chunks: dict[str, Chunk] = {}
        for chunk in flat_chunks:
            previous = unique_chunks.setdefault(chunk.chunk_id, chunk)
            if previous != chunk:
                raise ValueError(
                    "batch contains duplicate chunk_id with different content"
                )
        flat_chunks = list(unique_chunks.values())
        manifests: dict[str, PendingIngestManifest] = {}
        for turn, chunks in staged:
            manifest = PendingIngestManifest.build(turn, chunks)
            previous = manifests.setdefault(turn.turn_id, manifest)
            if previous != manifest:
                raise ValueError(
                    "batch contains duplicate turn_id with different chunks"
                )

        if not flat_chunks:
            published, _ingest_status = self._publish_staged_turns(
                publication_requests,
                manifests,
            )
            staged = [(published[turn.turn_id], chunks) for turn, chunks in staged]
            return staged

        embedded = _embed_source_chunks(self._embedder, flat_chunks)
        published, ingest_status = self._publish_staged_turns(
            publication_requests,
            manifests,
        )
        staged = [(published[turn.turn_id], chunks) for turn, chunks in staged]
        pending_manifests = [
            manifest
            for turn_id, manifest in manifests.items()
            if ingest_status[turn_id] == "pending"
        ]
        if pending_manifests:
            self._index_pending_manifests(embedded, pending_manifests)
        by_turn: dict[str, list[Chunk]] = {}
        for chunk in embedded:
            by_turn.setdefault(chunk.turn_id, []).append(chunk)
        return [(turn, by_turn.get(turn.turn_id, [])) for turn, _ in staged]

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
                ingest_status[turn.turn_id] = self._pending_ingests.claim(
                    manifests[turn.turn_id]
                )
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

    def recover_pending_ingests(self) -> list[tuple[Turn, list[Chunk]]]:
        """Replay every durable pending manifest through the normal indexes."""
        if self._db.read_only:
            raise sqlite3.OperationalError("attempt to write a readonly database")
        manifests = self._pending_ingests.list_pending()
        if not manifests:
            return []

        staged: list[tuple[Turn, list[Chunk]]] = []
        flat_chunks: list[Chunk] = []
        unique_chunks: dict[str, Chunk] = {}
        for manifest in manifests:
            turn = self._transcript.get_turn(manifest.turn_id)
            if turn is None:
                raise RuntimeError("pending ingest references an unknown turn")
            chunks = manifest.reconstruct(turn)
            staged.append((turn, chunks))
            for chunk in chunks:
                previous = unique_chunks.setdefault(chunk.chunk_id, chunk)
                if previous != chunk:
                    raise ValueError(
                        "pending ingests contain a conflicting chunk identity"
                    )
        flat_chunks = list(unique_chunks.values())

        if flat_chunks:
            embedded = _embed_source_chunks(self._embedder, flat_chunks)
            self._index_pending_manifests(embedded, manifests)
        else:
            self._index_pending_manifests([], manifests)
            embedded = []

        by_turn: dict[str, list[Chunk]] = {}
        for chunk in embedded:
            by_turn.setdefault(chunk.turn_id, []).append(chunk)
        return [(turn, by_turn.get(turn.turn_id, [])) for turn, _chunks in staged]

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
        if ops.is_empty():
            return {}
        report = self._validator.validate(ops)
        return self._memory.apply(report)

__all__ = ["IngestWorkflowMixin"]
