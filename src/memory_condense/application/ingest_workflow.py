"""Stateful ingestion and indexed-signature workflows for the condenser."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from memory_condense.associations.association_store import AssociationArtifact
from memory_condense.domain.schemas import Chunk, RetrievalResult, Turn


class IngestWorkflowMixin:
    """Internal workflow methods composed by ``MemoryCondenser``."""

    def ingest(
        self,
        role: str,
        text: str,
        *,
        source_id: str | None = None,
        created_at: datetime | None = None,
    ) -> tuple[Turn, list[Chunk]]:
        """Ingest a single conversation turn.

        Stores the turn, chunks and embeds the text, indexes the chunks for
        both dense and lexical retrieval, and — when ``auto_extract`` is on —
        proposes memory items, validates their provenance, and applies the
        surviving ops.
        """
        turn = self._transcript.append(
            role,
            text,
            source_id=source_id,
            created_at=created_at,
        )
        chunks = self._chunker.chunk_turn(turn.turn_id, text)

        if chunks:
            chunks = self._embedder.embed_chunks(chunks)
            self._retriever.add_chunks(chunks)

        if self._auto_extract:
            self.extract_memory([turn], chunks)

        return turn, chunks

    def ingest_many(
        self,
        turns: Sequence[
            tuple[str, str, str | None]
            | tuple[str, str, str | None, datetime | None]
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
        records: list[tuple[str, str, str | None, datetime | None]] = []
        for record in turns:
            if len(record) == 3:
                role, text, source_id = record
                records.append((role, text, source_id, None))
            elif len(record) == 4:
                role, text, source_id, created_at = record
                records.append((role, text, source_id, created_at))
            else:  # pragma: no cover - static tuple union, runtime guard
                raise ValueError(
                    "ingest records need role, text, source, and optional time"
                )
        if self._auto_extract:
            return [
                self.ingest(
                    role,
                    text,
                    source_id=source_id,
                    created_at=created_at,
                )
                for role, text, source_id, created_at in records
            ]

        staged: list[tuple[Turn, list[Chunk]]] = []
        flat_chunks: list[Chunk] = []
        for role, text, source_id, created_at in records:
            turn = self._transcript.append(
                role,
                text,
                source_id=source_id,
                created_at=created_at,
            )
            chunks = self._chunker.chunk_turn(turn.turn_id, text)
            staged.append((turn, chunks))
            flat_chunks.extend(chunks)

        if not flat_chunks:
            return staged

        embedded = self._embedder.embed_chunks(flat_chunks)
        self._retriever.add_chunks(embedded)
        by_turn: dict[str, list[Chunk]] = {}
        for chunk in embedded:
            by_turn.setdefault(chunk.turn_id, []).append(chunk)
        return [(turn, by_turn.get(turn.turn_id, [])) for turn, _ in staged]

    def compile_cav_signatures(
        self,
        linker: object,
        artifact: AssociationArtifact,
        chunks: Sequence[Chunk | RetrievalResult],
        *,
        batch_size: int = 8,
        overwrite: bool = False,
        conceptual_spans: bool = True,
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
            or self._associations.get_signature(
                chunk.chunk_id, artifact.artifact_id
            )
            is None
        ]
        span_texts: list[str] = []
        span_owners: list[str] = []
        for chunk in pending:
            spans = (
                self._chunker.conceptual_spans(chunk.text)
                if conceptual_spans
                else [chunk.text]
            )
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
            "WHERE c.embedding IS NOT NULL AND c.hnsw_label IS NOT NULL "
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
