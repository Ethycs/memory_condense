"""Durable T1g journal and append deltas for the conversation graph."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Sequence

from memory_condense.domain._discourse_identity import (
    identity_sha256,
    quote_sha256,
)
from memory_condense.domain.schemas import Chunk, Turn
from memory_condense.persistence.db import Database
from memory_condense.persistence.pending_ingest_store import PendingIngestManifest
from memory_condense.search.incremental_conversation_graph import (
    ConversationGraphAppendDelta,
    ConversationGraphChunk,
    GraphAppendReceipt,
    IncrementalConversationGraph,
    PhraseExtractionPolicy,
    PhraseOccurrence,
    StoryAffinityIndexPolicy,
    StoryTermMembership,
)


GRAPH_ARTIFACT_FORMAT = "memory-condense-conversation-graph-v1"
EMPTY_GRAPH_CHECKPOINT_SHA256 = "0" * 64
GRAPH_BOOTSTRAP_DEFAULT_MAX_TURNS = 32
GRAPH_BOOTSTRAP_HARD_MAX_TURNS = 512
_RESTORE_PAGE_SIZE = 128
_PERSISTENCE_POLICY_SHA256 = identity_sha256(
    {
        "schema": "conversation-graph-persistence-policy-v1",
        "checkpoint": "sha256-parent-delta-chain",
        "evidence_text": "hydrate-from-authoritative-turn-and-chunk",
        "occurrence_order": "compiler-order",
        "resident_restore": "validated-append-delta-without-extraction",
        "sequence_coordinate": ["source_id", "turn_ordinal", "start_char"],
        "t0": "manifest-bound-journal-claim-only",
        "t1g": "post-index-short-sqlite-transaction",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _failure_kind(error: BaseException) -> str:
    kind = f"{type(error).__module__}.{type(error).__qualname__}"
    return kind[:255]


def _state_sha256(
    *,
    artifact_id: str,
    revision: int,
    chunk_count: int,
    occurrence_count: int,
    story_term_membership_count: int,
    story_evidence_chunk_count: int,
    ready_turn_count: int,
    checkpoint_sha256: str,
) -> str:
    return identity_sha256(
        {
            "schema": "conversation-graph-state-v1",
            "artifact_id": artifact_id,
            "revision": revision,
            "chunk_count": chunk_count,
            "occurrence_count": occurrence_count,
            "story_term_membership_count": story_term_membership_count,
            "story_evidence_chunk_count": story_evidence_chunk_count,
            "ready_turn_count": ready_turn_count,
            "checkpoint_sha256": checkpoint_sha256,
        }
    )


@dataclass(frozen=True, slots=True)
class GraphCompilationReceipt:
    """Terminal, manifest-bound receipt for one deterministic T1g turn."""

    artifact_id: str
    turn_id: str
    ingest_manifest_sha256: str
    status: str
    first_revision: int | None
    last_revision: int | None
    chunk_count: int
    occurrence_count: int
    checkpoint_sha256: str
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class GraphCompilationResult:
    """One worker result; concurrent/replayed completion is not recreated."""

    created: bool
    receipt: GraphCompilationReceipt


@dataclass(frozen=True, slots=True)
class GraphBootstrapResult:
    """One bounded current-policy bootstrap page and its remaining work."""

    artifact_id: str
    selected_turn_ids: tuple[str, ...]
    claimed_turn_ids: tuple[str, ...]
    completed_turn_ids: tuple[str, ...]
    pending_turn_ids: tuple[str, ...]
    remaining_turn_count: int
    unsupported_indexed_turn_count: int


@dataclass(frozen=True, slots=True)
class _PreparedChunk:
    chunk: ConversationGraphChunk
    occurrences: tuple[PhraseOccurrence, ...]
    story_candidates: tuple[tuple[str, PhraseOccurrence], ...]


@dataclass(frozen=True, slots=True)
class _PreparedTurn:
    turn_id: str
    manifest_sha256: str
    chunks: tuple[_PreparedChunk, ...]


class ConversationGraphStore:
    """Persist and restore one versioned incremental conversation graph.

    T0 calls :meth:`claim_many` inside the transcript/manifest transaction.
    It inserts no evidence-derived rows.  The worker compiles only receipts
    whose normal ingest is already ``indexed`` and commits a whole turn in a
    separate transaction.  Failures remain pending and never affect T1.
    """

    def __init__(
        self,
        db: Database,
        *,
        extraction_policy: PhraseExtractionPolicy | None = None,
        story_index_policy: StoryAffinityIndexPolicy | None = None,
    ) -> None:
        self._db = db
        self._extraction_policy = extraction_policy or PhraseExtractionPolicy()
        self._story_index_policy = story_index_policy or StoryAffinityIndexPolicy()
        self._artifact_id = identity_sha256(
            {
                "schema": "conversation-graph-artifact-identity-v1",
                "format": GRAPH_ARTIFACT_FORMAT,
                "extraction_policy_sha256": (
                    self._extraction_policy.policy_sha256
                ),
                "story_index_policy_sha256": (
                    self._story_index_policy.policy_sha256
                ),
                "persistence_policy_sha256": _PERSISTENCE_POLICY_SHA256,
            }
        )
        self._resident: IncrementalConversationGraph | None = None
        self._observed_revision = 0
        self._observed_checkpoint_sha256 = EMPTY_GRAPH_CHECKPOINT_SHA256
        self._resident_lock = RLock()
        self.last_failures: dict[str, str] = {}

    @property
    def artifact_id(self) -> str:
        return self._artifact_id

    @property
    def extraction_policy(self) -> PhraseExtractionPolicy:
        return self._extraction_policy

    @property
    def story_index_policy(self) -> StoryAffinityIndexPolicy:
        return self._story_index_policy

    def _ensure_artifact(self) -> None:
        if not self._db.connection.in_transaction:
            raise RuntimeError("graph artifact creation requires caller transaction")
        now = _utc_now()
        self._db.execute(
            "INSERT INTO graph_artifacts "
            "(artifact_id, format, extraction_policy_sha256, "
            " story_index_policy_sha256, persistence_policy_sha256, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(artifact_id) DO NOTHING",
            (
                self._artifact_id,
                GRAPH_ARTIFACT_FORMAT,
                self._extraction_policy.policy_sha256,
                self._story_index_policy.policy_sha256,
                _PERSISTENCE_POLICY_SHA256,
                now,
            ),
        )
        initial_state_sha256 = _state_sha256(
            artifact_id=self._artifact_id,
            revision=0,
            chunk_count=0,
            occurrence_count=0,
            story_term_membership_count=0,
            story_evidence_chunk_count=0,
            ready_turn_count=0,
            checkpoint_sha256=EMPTY_GRAPH_CHECKPOINT_SHA256,
        )
        self._db.execute(
            "INSERT INTO conversation_graph_state "
            "(artifact_id, revision, chunk_count, occurrence_count, "
            " story_term_membership_count, story_evidence_chunk_count, "
            " ready_turn_count, checkpoint_sha256, state_sha256, updated_at) "
            "VALUES (?, 0, 0, 0, 0, 0, 0, ?, ?, ?) "
            "ON CONFLICT(artifact_id) DO NOTHING",
            (
                self._artifact_id,
                EMPTY_GRAPH_CHECKPOINT_SHA256,
                initial_state_sha256,
                now,
            ),
        )

    def claim_many(
        self,
        manifests: Sequence[PendingIngestManifest],
    ) -> dict[str, str]:
        """Claim manifest-bound T1g work in the caller's T0 transaction."""

        by_turn: dict[str, PendingIngestManifest] = {}
        for manifest in manifests:
            previous = by_turn.setdefault(manifest.turn_id, manifest)
            if previous != manifest:
                raise ValueError("conflicting graph manifests in one claim")
        if not by_turn:
            return {}
        if not self._db.connection.in_transaction:
            raise RuntimeError("claim_many requires an active caller transaction")
        self._ensure_artifact()
        now = _utc_now()
        self._db.connection.executemany(
            "INSERT INTO pending_graph_compilations "
            "(artifact_id, turn_id, ingest_manifest_sha256, status, created_at) "
            "VALUES (?, ?, ?, 'pending', ?) "
            "ON CONFLICT(artifact_id, turn_id) DO NOTHING",
            [
                (self._artifact_id, manifest.turn_id, manifest.sha256, now)
                for manifest in by_turn.values()
            ],
        )
        placeholders = ",".join("?" for _ in by_turn)
        rows = self._db.execute(
            "SELECT turn_id, ingest_manifest_sha256, status "
            "FROM pending_graph_compilations "
            "WHERE artifact_id = ? "
            f"AND turn_id IN ({placeholders})",
            (self._artifact_id, *by_turn),
        ).fetchall()
        found = {
            str(turn_id): (str(manifest_sha256), str(status))
            for turn_id, manifest_sha256, status in rows
        }
        output: dict[str, str] = {}
        for turn_id, manifest in by_turn.items():
            row = found.get(turn_id)
            if row is None or row[0] != manifest.sha256:
                raise ValueError("turn already has a different graph manifest")
            output[turn_id] = row[1]
        return output

    def status(self, turn_id: str) -> str | None:
        row = self._db.execute(
            "SELECT status FROM pending_graph_compilations "
            "WHERE artifact_id = ? AND turn_id = ?",
            (self._artifact_id, turn_id),
        ).fetchone()
        return None if row is None else str(row[0])

    def pending_count(self, *, indexed_only: bool = False) -> int:
        sql = (
            "SELECT COUNT(*) FROM pending_graph_compilations AS g "
            "JOIN pending_ingests AS p ON p.turn_id = g.turn_id "
            "WHERE g.artifact_id = ? AND g.status = 'pending'"
        )
        if indexed_only:
            sql += " AND p.status = 'indexed'"
        return int(self._db.execute(sql, (self._artifact_id,)).fetchone()[0])

    def attempt_count(self, turn_id: str) -> int | None:
        row = self._db.execute(
            "SELECT attempt_count FROM pending_graph_compilations "
            "WHERE artifact_id = ? AND turn_id = ?",
            (self._artifact_id, turn_id),
        ).fetchone()
        return None if row is None else int(row[0])

    def _claim_bootstrap_page(
        self,
        *,
        max_turns: int,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Claim one manifest-backed page without compiling graph evidence."""

        connection = self._db.connection
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = self._db.execute(
                "SELECT p.turn_id, p.manifest_sha256, p.manifest_json, "
                "g.ingest_manifest_sha256, g.status "
                "FROM pending_ingests AS p "
                "JOIN turns AS t ON t.turn_id = p.turn_id "
                "LEFT JOIN pending_graph_compilations AS g "
                "ON g.artifact_id = ? AND g.turn_id = p.turn_id "
                "WHERE p.status = 'indexed' "
                "AND (g.turn_id IS NULL OR g.status = 'pending') "
                "ORDER BY t.ordinal, p.turn_id LIMIT ?",
                (self._artifact_id, max_turns),
            ).fetchall()
            selected: list[str] = []
            missing: list[PendingIngestManifest] = []
            for (
                turn_id,
                manifest_sha256,
                manifest_json,
                graph_manifest_sha256,
                graph_status,
            ) in rows:
                normalized_turn_id = str(turn_id)
                manifest = PendingIngestManifest.from_json(str(manifest_json))
                if (
                    manifest.turn_id != normalized_turn_id
                    or manifest.sha256 != str(manifest_sha256)
                ):
                    raise ValueError(
                        "bootstrap pending ingest manifest receipt is inconsistent"
                    )
                selected.append(normalized_turn_id)
                if graph_status is None:
                    missing.append(manifest)
                elif (
                    str(graph_status) != "pending"
                    or str(graph_manifest_sha256) != manifest.sha256
                ):
                    raise ValueError(
                        "bootstrap graph job and ingest manifest disagree"
                    )
            if missing:
                statuses = self.claim_many(missing)
                if any(statuses[row.turn_id] != "pending" for row in missing):
                    raise RuntimeError("bootstrap claim did not create pending work")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        return tuple(selected), tuple(row.turn_id for row in missing)

    def _bootstrap_remaining_count(self) -> int:
        return int(
            self._db.execute(
                "SELECT COUNT(*) FROM pending_ingests AS p "
                "LEFT JOIN pending_graph_compilations AS g "
                "ON g.artifact_id = ? AND g.turn_id = p.turn_id "
                "WHERE p.status = 'indexed' "
                "AND (g.turn_id IS NULL OR g.status = 'pending')",
                (self._artifact_id,),
            ).fetchone()[0]
        )

    def _unsupported_indexed_turn_count(self) -> int:
        """Count complete legacy T1 turns that have no replay manifest."""

        return int(
            self._db.execute(
                "SELECT COUNT(*) FROM turns AS t "
                "WHERE NOT EXISTS ("
                " SELECT 1 FROM pending_ingests AS p WHERE p.turn_id = t.turn_id"
                ") AND EXISTS ("
                " SELECT 1 FROM chunks AS c WHERE c.turn_id = t.turn_id"
                ") AND NOT EXISTS ("
                " SELECT 1 FROM chunks AS c WHERE c.turn_id = t.turn_id "
                " AND (c.embedding IS NULL OR c.hnsw_label IS NULL "
                "      OR c.term_count IS NULL)"
                ")"
            ).fetchone()[0]
        )

    def bootstrap_indexed_turns(
        self,
        *,
        max_turns: int = GRAPH_BOOTSTRAP_DEFAULT_MAX_TURNS,
    ) -> GraphBootstrapResult:
        """Boundedly bootstrap the active policy from sealed indexed turns.

        Existing pending jobs are included so a failed page is resumable.
        Missing jobs are claimed together in a short transaction; phrase
        extraction happens only afterward and only for the exact selected
        turn IDs. Complete legacy turns without a pending-ingest manifest are
        counted as unsupported rather than reconstructed heuristically.
        """

        if (
            type(max_turns) is not int
            or max_turns < 1
            or max_turns > GRAPH_BOOTSTRAP_HARD_MAX_TURNS
        ):
            raise ValueError(
                "max_turns must be an integer from 1 through "
                f"{GRAPH_BOOTSTRAP_HARD_MAX_TURNS}"
            )
        selected, claimed = self._claim_bootstrap_page(max_turns=max_turns)
        if selected:
            self.drain_pending(max_turns=len(selected), turn_ids=selected)
        status_by_turn = {turn_id: self.status(turn_id) for turn_id in selected}
        if any(status is None for status in status_by_turn.values()):
            raise RuntimeError("bootstrap selected work lost its durable graph job")
        completed = tuple(
            turn_id
            for turn_id in selected
            if status_by_turn[turn_id] in {"ready", "no_output"}
        )
        pending = tuple(
            turn_id
            for turn_id in selected
            if status_by_turn[turn_id] == "pending"
        )
        return GraphBootstrapResult(
            artifact_id=self._artifact_id,
            selected_turn_ids=selected,
            claimed_turn_ids=claimed,
            completed_turn_ids=completed,
            pending_turn_ids=pending,
            remaining_turn_count=self._bootstrap_remaining_count(),
            unsupported_indexed_turn_count=(
                self._unsupported_indexed_turn_count()
            ),
        )

    def _pending_indexed_turn_ids(
        self,
        *,
        max_turns: int | None,
        turn_ids: Sequence[str] | None,
    ) -> list[str]:
        params: list[object] = [self._artifact_id]
        sql = (
            "SELECT g.turn_id FROM pending_graph_compilations AS g "
            "JOIN pending_ingests AS p ON p.turn_id = g.turn_id "
            "JOIN turns AS t ON t.turn_id = g.turn_id "
            "WHERE g.artifact_id = ? AND g.status = 'pending' "
            "AND p.status = 'indexed' "
        )
        if turn_ids is not None:
            normalized = tuple(dict.fromkeys(str(value) for value in turn_ids))
            if not normalized:
                return []
            sql += "AND g.turn_id IN (" + ",".join("?" for _ in normalized) + ") "
            params.extend(normalized)
        sql += "ORDER BY (g.attempt_count > 0), t.ordinal, g.turn_id"
        if max_turns is not None:
            sql += " LIMIT ?"
            params.append(max_turns)
        return [
            str(row[0]) for row in self._db.execute(sql, tuple(params)).fetchall()
        ]

    def _prepare_turn(self, turn_id: str) -> _PreparedTurn | None:
        row = self._db.execute(
            "SELECT g.ingest_manifest_sha256, g.status, p.status, "
            "p.manifest_sha256, p.manifest_json, "
            "t.role, t.text, t.source_id, t.created_at, t.ordinal "
            "FROM pending_graph_compilations AS g "
            "JOIN pending_ingests AS p ON p.turn_id = g.turn_id "
            "JOIN turns AS t ON t.turn_id = g.turn_id "
            "WHERE g.artifact_id = ? AND g.turn_id = ?",
            (self._artifact_id, turn_id),
        ).fetchone()
        if row is None or str(row[1]) != "pending":
            return None
        if str(row[2]) != "indexed":
            return None
        manifest = PendingIngestManifest.from_json(str(row[4]))
        if (
            manifest.turn_id != turn_id
            or manifest.sha256 != str(row[0])
            or manifest.sha256 != str(row[3])
        ):
            raise ValueError("graph job and ingest manifest identities disagree")
        turn = Turn(
            turn_id=turn_id,
            role=str(row[5]),
            text=str(row[6]),
            source_id=None if row[7] is None else str(row[7]),
            created_at=str(row[8]),
        )
        ordinal = int(row[9])
        source_chunks = manifest.reconstruct(turn)
        if source_chunks:
            placeholders = ",".join("?" for _ in source_chunks)
            durable_rows = self._db.execute(
                "SELECT chunk_id, turn_id, text, start_char, end_char, "
                "token_count, embedding IS NOT NULL, hnsw_label IS NOT NULL, "
                "term_count IS NOT NULL FROM chunks "
                f"WHERE chunk_id IN ({placeholders})",
                tuple(chunk.chunk_id for chunk in source_chunks),
            ).fetchall()
            durable = {str(value[0]): value for value in durable_rows}
        else:
            durable = {}
        prepared_chunks: list[_PreparedChunk] = []
        for chunk in sorted(
            source_chunks,
            key=lambda value: (value.start_char, value.end_char, value.chunk_id),
        ):
            durable_row = durable.get(chunk.chunk_id)
            expected = (
                chunk.chunk_id,
                turn_id,
                chunk.text,
                chunk.start_char,
                chunk.end_char,
                chunk.token_count,
                1,
                1,
                1,
            )
            if durable_row is None or tuple(durable_row) != expected:
                raise ValueError("graph compilation requires a complete T1 chunk")
            graph_chunk = ConversationGraphChunk.from_chunk(
                chunk,
                turn,
                ordinal=ordinal,
            )
            # Compile each immutable physical delta independently. Persistent
            # first-seen caps are applied under the T1g writer lock below.
            compiler = IncrementalConversationGraph(
                extraction_policy=self._extraction_policy,
                story_index_policy=self._story_index_policy,
            )
            compiler.append_chunk(graph_chunk)
            compiled_delta = compiler.append_delta(graph_chunk.chunk_id)
            prepared_chunks.append(
                _PreparedChunk(
                    chunk=graph_chunk,
                    occurrences=compiled_delta.occurrences,
                    story_candidates=tuple(
                        (membership.term, membership.occurrence)
                        for membership in (
                            compiled_delta.new_story_term_memberships
                        )
                    ),
                )
            )
        return _PreparedTurn(
            turn_id=turn_id,
            manifest_sha256=manifest.sha256,
            chunks=tuple(prepared_chunks),
        )

    @staticmethod
    def _runtime_receipt(
        *,
        artifact_revision: int,
        chunk: ConversationGraphChunk,
        occurrences: Sequence[PhraseOccurrence],
        extraction_policy_sha256: str,
        story_index_policy_sha256: str,
        new_story_terms: Sequence[tuple[str, PhraseOccurrence]],
        story_evidence_chunk_retained: bool,
        predecessor_chunk_id: str | None,
        successor_chunk_id: str | None,
    ) -> GraphAppendReceipt:
        phrase_keys = tuple(sorted({row.phrase_key for row in occurrences}))
        payload = {
            "schema": "incremental-conversation-graph-append-v1",
            "revision": artifact_revision,
            "chunk": chunk.identity_payload(),
            "extraction_policy_sha256": extraction_policy_sha256,
            "story_index_policy_sha256": story_index_policy_sha256,
            "occurrence_ids": [row.occurrence_id for row in occurrences],
            "phrase_keys": list(phrase_keys),
            "new_story_terms": [
                {"occurrence_id": occurrence.occurrence_id, "term": term}
                for term, occurrence in new_story_terms
            ],
            "story_evidence_chunk_retained": story_evidence_chunk_retained,
            "predecessor_chunk_id": predecessor_chunk_id,
            "successor_chunk_id": successor_chunk_id,
        }
        return GraphAppendReceipt(
            revision=artifact_revision,
            chunk_id=chunk.chunk_id,
            chunk_identity_sha256=identity_sha256(chunk.identity_payload()),
            extraction_policy_sha256=extraction_policy_sha256,
            story_index_policy_sha256=story_index_policy_sha256,
            occurrence_count=len(occurrences),
            phrase_key_count=len(phrase_keys),
            new_story_term_membership_count=len(new_story_terms),
            story_evidence_chunk_retained=story_evidence_chunk_retained,
            predecessor_chunk_id=predecessor_chunk_id,
            successor_chunk_id=successor_chunk_id,
            receipt_sha256=identity_sha256(payload),
        )

    def _neighbor_ids(
        self,
        chunk: ConversationGraphChunk,
    ) -> tuple[str | None, str | None]:
        coordinates = (
            self._artifact_id,
            chunk.source_id,
            chunk.ordinal,
            chunk.start_char,
            chunk.chunk_id,
        )
        predecessor = self._db.execute(
            "SELECT chunk_id FROM conversation_graph_chunks "
            "WHERE artifact_id = ? AND source_id = ? AND "
            "(turn_ordinal, start_char, chunk_id) < (?, ?, ?) "
            "ORDER BY turn_ordinal DESC, start_char DESC, chunk_id DESC LIMIT 1",
            coordinates,
        ).fetchone()
        successor = self._db.execute(
            "SELECT chunk_id FROM conversation_graph_chunks "
            "WHERE artifact_id = ? AND source_id = ? AND "
            "(turn_ordinal, start_char, chunk_id) > (?, ?, ?) "
            "ORDER BY turn_ordinal, start_char, chunk_id LIMIT 1",
            coordinates,
        ).fetchone()
        return (
            None if predecessor is None else str(predecessor[0]),
            None if successor is None else str(successor[0]),
        )

    def _terminal_receipt(self, turn_id: str) -> GraphCompilationReceipt | None:
        row = self._db.execute(
            "SELECT ingest_manifest_sha256, status, first_revision, "
            "last_revision, chunk_count, occurrence_count, "
            "checkpoint_sha256, receipt_sha256 "
            "FROM pending_graph_compilations "
            "WHERE artifact_id = ? AND turn_id = ?",
            (self._artifact_id, turn_id),
        ).fetchone()
        if row is None or str(row[1]) == "pending":
            return None
        receipt = GraphCompilationReceipt(
            artifact_id=self._artifact_id,
            turn_id=turn_id,
            ingest_manifest_sha256=str(row[0]),
            status=str(row[1]),
            first_revision=None if row[2] is None else int(row[2]),
            last_revision=None if row[3] is None else int(row[3]),
            chunk_count=int(row[4]),
            occurrence_count=int(row[5]),
            checkpoint_sha256=str(row[6]),
            receipt_sha256=str(row[7]),
        )
        delta_rows = self._db.execute(
            "SELECT delta_sha256 FROM conversation_graph_chunks "
            "WHERE artifact_id = ? AND turn_id = ? ORDER BY append_revision",
            (self._artifact_id, turn_id),
        ).fetchall()
        expected = identity_sha256(
            {
                "schema": "conversation-graph-turn-receipt-v1",
                "artifact_id": self._artifact_id,
                "turn_id": turn_id,
                "ingest_manifest_sha256": receipt.ingest_manifest_sha256,
                "status": receipt.status,
                "first_revision": receipt.first_revision,
                "last_revision": receipt.last_revision,
                "chunk_count": receipt.chunk_count,
                "occurrence_count": receipt.occurrence_count,
                "delta_sha256s": [str(value[0]) for value in delta_rows],
                "checkpoint_sha256": receipt.checkpoint_sha256,
            }
        )
        if expected != receipt.receipt_sha256:
            raise ValueError("graph terminal receipt hash does not match")
        return receipt

    def _publish_prepared(self, prepared: _PreparedTurn) -> GraphCompilationResult:
        connection = self._db.connection
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = self._terminal_receipt(prepared.turn_id)
            if existing is not None:
                connection.commit()
                return GraphCompilationResult(False, existing)
            job = self._db.execute(
                "SELECT g.status, g.ingest_manifest_sha256, p.status "
                "FROM pending_graph_compilations AS g "
                "JOIN pending_ingests AS p ON p.turn_id = g.turn_id "
                "WHERE g.artifact_id = ? AND g.turn_id = ?",
                (self._artifact_id, prepared.turn_id),
            ).fetchone()
            if (
                job is None
                or str(job[0]) != "pending"
                or str(job[1]) != prepared.manifest_sha256
                or str(job[2]) != "indexed"
            ):
                raise RuntimeError("graph compilation lost its indexed CAS")
            state = self._db.execute(
                "SELECT revision, chunk_count, occurrence_count, "
                "story_term_membership_count, story_evidence_chunk_count, "
                "ready_turn_count, checkpoint_sha256 FROM conversation_graph_state "
                "WHERE artifact_id = ?",
                (self._artifact_id,),
            ).fetchone()
            if state is None:
                raise RuntimeError("graph artifact state is missing")
            revision = int(state[0])
            chunk_count = int(state[1])
            occurrence_count = int(state[2])
            story_term_membership_count = int(state[3])
            story_evidence_chunk_count = int(state[4])
            ready_turn_count = int(state[5])
            checkpoint_sha256 = str(state[6])
            first_revision: int | None = None
            delta_sha256s: list[str] = []
            for prepared_chunk in prepared.chunks:
                chunk = prepared_chunk.chunk
                if self._db.execute(
                    "SELECT 1 FROM conversation_graph_chunks "
                    "WHERE artifact_id = ? AND chunk_id = ?",
                    (self._artifact_id, chunk.chunk_id),
                ).fetchone() is not None:
                    raise RuntimeError("pending graph job already owns a chunk delta")
                predecessor, successor = self._neighbor_ids(chunk)
                existing_source_terms = {
                    str(value[0])
                    for value in self._db.execute(
                        "SELECT story_term FROM conversation_story_term_memberships "
                        "WHERE artifact_id = ? AND source_id = ?",
                        (self._artifact_id, chunk.source_id),
                    ).fetchall()
                }
                candidate_terms = tuple(
                    dict.fromkeys(
                        term
                        for term, _occurrence in prepared_chunk.story_candidates
                    )
                )
                term_source_counts: dict[str, int] = {}
                if candidate_terms:
                    term_placeholders = ",".join("?" for _ in candidate_terms)
                    term_source_counts = {
                        str(term): int(count)
                        for term, count in self._db.execute(
                            "SELECT story_term, COUNT(*) FROM "
                            "conversation_story_term_memberships "
                            "WHERE artifact_id = ? "
                            f"AND story_term IN ({term_placeholders}) "
                            "GROUP BY story_term",
                            (self._artifact_id, *candidate_terms),
                        ).fetchall()
                    }
                remaining_source_slots = (
                    self._story_index_policy.max_terms_per_source
                    - len(existing_source_terms)
                )
                new_story_terms: list[tuple[str, PhraseOccurrence]] = []
                for term, occurrence in prepared_chunk.story_candidates:
                    if remaining_source_slots <= 0:
                        break
                    if term in existing_source_terms:
                        continue
                    if term_source_counts.get(term, 0) >= (
                        self._story_index_policy.max_sources_per_term
                    ):
                        continue
                    existing_source_terms.add(term)
                    term_source_counts[term] = term_source_counts.get(term, 0) + 1
                    new_story_terms.append((term, occurrence))
                    remaining_source_slots -= 1
                retained_source_evidence = int(
                    self._db.execute(
                        "SELECT COUNT(*) FROM conversation_graph_chunks "
                        "WHERE artifact_id = ? AND source_id = ? "
                        "AND story_evidence_chunk_retained = 1",
                        (self._artifact_id, chunk.source_id),
                    ).fetchone()[0]
                )
                story_evidence_chunk_retained = (
                    chunk.role == "user"
                    and retained_source_evidence
                    < self._story_index_policy.max_user_chunks_per_source
                )
                revision += 1
                if first_revision is None:
                    first_revision = revision
                runtime_receipt = self._runtime_receipt(
                    artifact_revision=revision,
                    chunk=chunk,
                    occurrences=prepared_chunk.occurrences,
                    extraction_policy_sha256=(
                        self._extraction_policy.policy_sha256
                    ),
                    story_index_policy_sha256=(
                        self._story_index_policy.policy_sha256
                    ),
                    new_story_terms=new_story_terms,
                    story_evidence_chunk_retained=story_evidence_chunk_retained,
                    predecessor_chunk_id=predecessor,
                    successor_chunk_id=successor,
                )
                phrase_keys = tuple(
                    sorted({row.phrase_key for row in prepared_chunk.occurrences})
                )
                delta_sha256 = identity_sha256(
                    {
                        "schema": "conversation-graph-chunk-delta-v1",
                        "artifact_id": self._artifact_id,
                        "append_revision": revision,
                        "chunk_identity_sha256": (
                            runtime_receipt.chunk_identity_sha256
                        ),
                        "extraction_policy_sha256": (
                            self._extraction_policy.policy_sha256
                        ),
                        "occurrence_ids": [
                            row.occurrence_id
                            for row in prepared_chunk.occurrences
                        ],
                        "phrase_keys": list(phrase_keys),
                        "story_index_policy_sha256": (
                            self._story_index_policy.policy_sha256
                        ),
                        "new_story_terms": [
                            {
                                "occurrence_id": occurrence.occurrence_id,
                                "term": term,
                            }
                            for term, occurrence in new_story_terms
                        ],
                        "story_evidence_chunk_retained": (
                            story_evidence_chunk_retained
                        ),
                        "predecessor_chunk_id": predecessor,
                        "successor_chunk_id": successor,
                        "runtime_receipt_sha256": (
                            runtime_receipt.receipt_sha256
                        ),
                    }
                )
                next_checkpoint = identity_sha256(
                    {
                        "schema": "conversation-graph-checkpoint-v1",
                        "artifact_id": self._artifact_id,
                        "revision": revision,
                        "parent_checkpoint_sha256": checkpoint_sha256,
                        "delta_sha256": delta_sha256,
                    }
                )
                self._db.execute(
                    "INSERT INTO conversation_graph_chunks "
                    "(artifact_id, chunk_id, turn_id, source_id, turn_ordinal, "
                    " role, start_char, end_char, created_at, text_sha256, "
                    " chunk_identity_sha256, append_revision, "
                    " predecessor_chunk_id, successor_chunk_id, occurrence_count, "
                    " phrase_key_count, new_story_term_membership_count, "
                    " story_evidence_chunk_retained, runtime_receipt_sha256, "
                    " delta_sha256, "
                    " parent_checkpoint_sha256, checkpoint_sha256) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                    "?, ?, ?, ?, ?, ?)",
                    (
                        self._artifact_id,
                        chunk.chunk_id,
                        chunk.turn_id,
                        chunk.source_id,
                        chunk.ordinal,
                        chunk.role,
                        chunk.start_char,
                        chunk.end_char,
                        chunk.created_at,
                        chunk.text_sha256,
                        runtime_receipt.chunk_identity_sha256,
                        revision,
                        predecessor,
                        successor,
                        len(prepared_chunk.occurrences),
                        len(phrase_keys),
                        len(new_story_terms),
                        int(story_evidence_chunk_retained),
                        runtime_receipt.receipt_sha256,
                        delta_sha256,
                        checkpoint_sha256,
                        next_checkpoint,
                    ),
                )
                self._db.connection.executemany(
                    "INSERT INTO conversation_phrase_occurrences "
                    "(artifact_id, occurrence_id, chunk_id, occurrence_ordinal, "
                    " canonical_key, start_char, end_char, quote_sha256, "
                    " token_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            self._artifact_id,
                            occurrence.occurrence_id,
                            chunk.chunk_id,
                            position,
                            occurrence.phrase_key,
                            occurrence.start_char,
                            occurrence.end_char,
                            occurrence.quote_sha256,
                            occurrence.token_count,
                        )
                        for position, occurrence in enumerate(
                            prepared_chunk.occurrences
                        )
                    ],
                )
                self._db.connection.executemany(
                    "INSERT INTO conversation_story_term_memberships "
                    "(artifact_id, source_id, story_term, occurrence_id, "
                    " append_revision, membership_ordinal) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (
                            self._artifact_id,
                            chunk.source_id,
                            term,
                            occurrence.occurrence_id,
                            revision,
                            position,
                        )
                        for position, (term, occurrence) in enumerate(
                            new_story_terms
                        )
                    ],
                )
                delta_sha256s.append(delta_sha256)
                checkpoint_sha256 = next_checkpoint
                chunk_count += 1
                occurrence_count += len(prepared_chunk.occurrences)
                story_term_membership_count += len(new_story_terms)
                story_evidence_chunk_count += int(
                    story_evidence_chunk_retained
                )
            status = "ready" if prepared.chunks else "no_output"
            ready_turn_count += 1
            state_sha256 = _state_sha256(
                artifact_id=self._artifact_id,
                revision=revision,
                chunk_count=chunk_count,
                occurrence_count=occurrence_count,
                story_term_membership_count=story_term_membership_count,
                story_evidence_chunk_count=story_evidence_chunk_count,
                ready_turn_count=ready_turn_count,
                checkpoint_sha256=checkpoint_sha256,
            )
            completed_at = _utc_now()
            updated = self._db.execute(
                "UPDATE conversation_graph_state SET revision = ?, "
                "chunk_count = ?, occurrence_count = ?, "
                "story_term_membership_count = ?, story_evidence_chunk_count = ?, "
                "ready_turn_count = ?, "
                "checkpoint_sha256 = ?, state_sha256 = ?, updated_at = ? "
                "WHERE artifact_id = ?",
                (
                    revision,
                    chunk_count,
                    occurrence_count,
                    story_term_membership_count,
                    story_evidence_chunk_count,
                    ready_turn_count,
                    checkpoint_sha256,
                    state_sha256,
                    completed_at,
                    self._artifact_id,
                ),
            ).rowcount
            if updated != 1:
                raise RuntimeError("graph state CAS did not advance")
            last_revision = revision if prepared.chunks else None
            turn_receipt_sha256 = identity_sha256(
                {
                    "schema": "conversation-graph-turn-receipt-v1",
                    "artifact_id": self._artifact_id,
                    "turn_id": prepared.turn_id,
                    "ingest_manifest_sha256": prepared.manifest_sha256,
                    "status": status,
                    "first_revision": first_revision,
                    "last_revision": last_revision,
                    "chunk_count": len(prepared.chunks),
                    "occurrence_count": sum(
                        len(value.occurrences) for value in prepared.chunks
                    ),
                    "delta_sha256s": delta_sha256s,
                    "checkpoint_sha256": checkpoint_sha256,
                }
            )
            terminal_count = self._db.execute(
                "UPDATE pending_graph_compilations SET status = ?, "
                "completed_at = ?, first_revision = ?, last_revision = ?, "
                "chunk_count = ?, occurrence_count = ?, checkpoint_sha256 = ?, "
                "receipt_sha256 = ? WHERE artifact_id = ? AND turn_id = ? "
                "AND status = 'pending' AND ingest_manifest_sha256 = ?",
                (
                    status,
                    completed_at,
                    first_revision,
                    last_revision,
                    len(prepared.chunks),
                    sum(len(value.occurrences) for value in prepared.chunks),
                    checkpoint_sha256,
                    turn_receipt_sha256,
                    self._artifact_id,
                    prepared.turn_id,
                    prepared.manifest_sha256,
                ),
            ).rowcount
            if terminal_count != 1:
                raise RuntimeError("graph terminal receipt CAS did not advance")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        receipt = self._terminal_receipt(prepared.turn_id)
        if receipt is None:
            raise RuntimeError("committed graph receipt is missing")
        return GraphCompilationResult(True, receipt)

    def _record_failure(self, turn_id: str, error: BaseException) -> None:
        connection = self._db.connection
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._db.execute(
                "UPDATE pending_graph_compilations "
                "SET attempt_count = attempt_count + 1, last_attempt_at = ?, "
                "last_error_kind = ? "
                "WHERE artifact_id = ? AND turn_id = ? AND status = 'pending'",
                (
                    _utc_now(),
                    _failure_kind(error),
                    self._artifact_id,
                    turn_id,
                ),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    def drain_pending(
        self,
        *,
        max_turns: int | None = None,
        turn_ids: Sequence[str] | None = None,
    ) -> list[GraphCompilationResult]:
        """Run bounded post-index work, recording and isolating every failure."""

        if max_turns is not None and (
            isinstance(max_turns, bool)
            or not isinstance(max_turns, int)
            or max_turns < 1
        ):
            raise ValueError("max_turns must be a positive integer or None")
        selected = self._pending_indexed_turn_ids(
            max_turns=max_turns,
            turn_ids=turn_ids,
        )
        output: list[GraphCompilationResult] = []
        failures: dict[str, str] = {}
        for turn_id in selected:
            try:
                prepared = self._prepare_turn(turn_id)
                if prepared is None:
                    continue
                output.append(self._publish_prepared(prepared))
            except Exception as error:
                failures[turn_id] = _failure_kind(error)
                try:
                    self._record_failure(turn_id, error)
                except Exception as receipt_error:
                    failures[turn_id] += f";receipt={_failure_kind(receipt_error)}"
        self.last_failures = failures
        return output

    def _restore_delta(
        self,
        graph: IncrementalConversationGraph,
        *,
        chunk: ConversationGraphChunk,
        occurrences: tuple[PhraseOccurrence, ...],
        new_story_terms: tuple[tuple[str, PhraseOccurrence], ...],
        story_evidence_chunk_retained: bool,
        receipt: GraphAppendReceipt,
    ) -> None:
        """Hydrate one authenticated delta without rerunning phrase extraction."""
        graph.restore_append_delta(
            ConversationGraphAppendDelta(
                chunk=chunk,
                occurrences=occurrences,
                new_story_term_memberships=tuple(
                    StoryTermMembership(term=term, occurrence=occurrence)
                    for term, occurrence in new_story_terms
                ),
                story_evidence_chunk_retained=story_evidence_chunk_retained,
                receipt=receipt,
            )
        )

    def _sync_resident_unlocked(self) -> IncrementalConversationGraph:
        if self._resident is None:
            self._resident = IncrementalConversationGraph(
                extraction_policy=self._extraction_policy,
                story_index_policy=self._story_index_policy,
            )
            self._observed_revision = 0
            self._observed_checkpoint_sha256 = EMPTY_GRAPH_CHECKPOINT_SHA256
        graph = self._resident

        # Pin one committed state image before reading any append rows. Without
        # this upper bound, a busy writer can make the paging loop chase a
        # moving tail indefinitely; a commit between the final empty page and
        # the former trailing state read also caused a false checkpoint
        # mismatch. Graph artifacts and chunk deltas are immutable, so this
        # copied target remains a valid restore boundary while later commits
        # wait for the next call.
        target_row = self._db.execute(
            "SELECT s.revision, s.chunk_count, s.occurrence_count, "
            "s.story_term_membership_count, s.story_evidence_chunk_count, "
            "s.ready_turn_count, s.checkpoint_sha256, s.state_sha256 "
            "FROM graph_artifacts AS a "
            "LEFT JOIN conversation_graph_state AS s "
            "ON s.artifact_id = a.artifact_id WHERE a.artifact_id = ?",
            (self._artifact_id,),
        ).fetchone()
        if target_row is None:
            if self._observed_revision != 0 or graph.stats().chunk_count != 0:
                raise ValueError("resident graph has no durable artifact")
            return graph
        if target_row[0] is None:
            raise ValueError("durable graph artifact has no state")
        target_revision = int(target_row[0])
        target_checkpoint_sha256 = str(target_row[6])
        if target_revision < self._observed_revision:
            raise ValueError("durable graph state moved behind its resident")
        while True:
            rows = self._db.execute(
                "SELECT g.chunk_id, g.turn_id, g.source_id, g.turn_ordinal, "
                "g.role, g.start_char, g.end_char, g.created_at, g.text_sha256, "
                "g.chunk_identity_sha256, g.append_revision, "
                "g.predecessor_chunk_id, g.successor_chunk_id, "
                "g.occurrence_count, g.phrase_key_count, "
                "g.new_story_term_membership_count, "
                "g.story_evidence_chunk_retained, "
                "g.runtime_receipt_sha256, g.delta_sha256, "
                "g.parent_checkpoint_sha256, g.checkpoint_sha256, "
                "c.text, c.token_count, t.text "
                "FROM conversation_graph_chunks AS g "
                "JOIN chunks AS c ON c.chunk_id = g.chunk_id "
                "JOIN turns AS t ON t.turn_id = g.turn_id "
                "WHERE g.artifact_id = ? AND g.append_revision > ? "
                "AND g.append_revision <= ? "
                "ORDER BY g.append_revision LIMIT ?",
                (
                    self._artifact_id,
                    self._observed_revision,
                    target_revision,
                    _RESTORE_PAGE_SIZE,
                ),
            ).fetchall()
            if not rows:
                break
            chunk_ids = [str(row[0]) for row in rows]
            placeholders = ",".join("?" for _ in chunk_ids)
            occurrence_rows = self._db.execute(
                "SELECT occurrence_id, chunk_id, occurrence_ordinal, "
                "canonical_key, start_char, end_char, quote_sha256, token_count "
                "FROM conversation_phrase_occurrences "
                "WHERE artifact_id = ? "
                f"AND chunk_id IN ({placeholders}) "
                "ORDER BY chunk_id, occurrence_ordinal",
                (self._artifact_id, *chunk_ids),
            ).fetchall()
            occurrences_by_chunk: dict[str, list[tuple[object, ...]]] = {}
            for occurrence_row in occurrence_rows:
                occurrences_by_chunk.setdefault(
                    str(occurrence_row[1]), []
                ).append(tuple(occurrence_row))
            story_rows = self._db.execute(
                "SELECT o.chunk_id, m.story_term, m.occurrence_id, "
                "m.membership_ordinal FROM conversation_story_term_memberships AS m "
                "JOIN conversation_phrase_occurrences AS o "
                "ON o.artifact_id = m.artifact_id "
                "AND o.occurrence_id = m.occurrence_id "
                "WHERE m.artifact_id = ? "
                f"AND o.chunk_id IN ({placeholders}) "
                "ORDER BY o.chunk_id, m.membership_ordinal",
                (self._artifact_id, *chunk_ids),
            ).fetchall()
            story_by_chunk: dict[str, list[tuple[str, str, int]]] = {}
            for story_chunk_id, story_term, occurrence_id, position in story_rows:
                story_by_chunk.setdefault(str(story_chunk_id), []).append(
                    (str(story_term), str(occurrence_id), int(position))
                )
            for row in rows:
                (
                    chunk_id,
                    turn_id,
                    source_id,
                    ordinal,
                    role,
                    start_char,
                    end_char,
                    created_at,
                    text_sha256,
                    chunk_identity_sha256,
                    append_revision,
                    predecessor,
                    successor,
                    stored_occurrence_count,
                    phrase_key_count,
                    stored_story_membership_count,
                    story_evidence_chunk_retained,
                    runtime_receipt_sha256,
                    delta_sha256,
                    parent_checkpoint_sha256,
                    checkpoint_sha256,
                    chunk_text,
                    token_count,
                    turn_text,
                ) = row
                turn = Turn(
                    turn_id=str(turn_id),
                    role=str(role),
                    text=str(turn_text),
                    source_id=str(source_id),
                    created_at=str(created_at),
                )
                chunk_model = Chunk(
                    chunk_id=str(chunk_id),
                    turn_id=str(turn_id),
                    text=str(chunk_text),
                    start_char=int(start_char),
                    end_char=int(end_char),
                    token_count=int(token_count),
                )
                graph_chunk = ConversationGraphChunk.from_chunk(
                    chunk_model,
                    turn,
                    ordinal=int(ordinal),
                )
                if graph_chunk.text_sha256 != str(text_sha256):
                    raise ValueError("persisted graph chunk text hash does not match")
                if identity_sha256(graph_chunk.identity_payload()) != str(
                    chunk_identity_sha256
                ):
                    raise ValueError("persisted graph chunk identity does not match")
                restored_occurrences: list[PhraseOccurrence] = []
                for expected_position, occurrence_row in enumerate(
                    occurrences_by_chunk.get(str(chunk_id), [])
                ):
                    (
                        occurrence_id,
                        occurrence_chunk_id,
                        occurrence_position,
                        canonical_key,
                        occurrence_start,
                        occurrence_end,
                        occurrence_quote_sha256,
                        occurrence_token_count,
                    ) = occurrence_row
                    if int(occurrence_position) != expected_position:
                        raise ValueError("graph occurrence order is not contiguous")
                    absolute_start = int(occurrence_start)
                    absolute_end = int(occurrence_end)
                    if not (
                        graph_chunk.start_char <= absolute_start < absolute_end
                        <= graph_chunk.end_char
                    ):
                        raise ValueError("graph occurrence escaped its source chunk")
                    quote = turn.text[absolute_start:absolute_end]
                    if quote_sha256(quote) != str(occurrence_quote_sha256):
                        raise ValueError("graph occurrence quote hash does not match")
                    expected_occurrence_id = identity_sha256(
                        {
                            "schema": "conversation-phrase-occurrence-v1",
                            "chunk_id": str(occurrence_chunk_id),
                            "phrase_key": str(canonical_key),
                            "start_char": absolute_start,
                            "end_char": absolute_end,
                            "quote_sha256": str(occurrence_quote_sha256),
                        }
                    )
                    if expected_occurrence_id != str(occurrence_id):
                        raise ValueError("graph occurrence identity does not match")
                    restored_occurrences.append(
                        PhraseOccurrence(
                            occurrence_id=str(occurrence_id),
                            phrase_key=str(canonical_key),
                            chunk_id=str(chunk_id),
                            source_id=str(source_id),
                            turn_id=str(turn_id),
                            start_char=absolute_start,
                            end_char=absolute_end,
                            quote=quote,
                            quote_sha256=str(occurrence_quote_sha256),
                            token_count=int(occurrence_token_count),
                        )
                    )
                occurrences = tuple(restored_occurrences)
                occurrence_by_id = {
                    value.occurrence_id: value for value in occurrences
                }
                restored_story_terms: list[tuple[str, PhraseOccurrence]] = []
                for expected_position, (
                    story_term,
                    story_occurrence_id,
                    story_position,
                ) in enumerate(story_by_chunk.get(str(chunk_id), [])):
                    if story_position != expected_position:
                        raise ValueError(
                            "graph story membership order is not contiguous"
                        )
                    story_occurrence = occurrence_by_id.get(story_occurrence_id)
                    if story_occurrence is None:
                        raise ValueError("graph story membership lost its occurrence")
                    restored_story_terms.append((story_term, story_occurrence))
                new_story_terms = tuple(restored_story_terms)
                phrase_keys = tuple(sorted({value.phrase_key for value in occurrences}))
                if (
                    len(occurrences) != int(stored_occurrence_count)
                    or len(phrase_keys) != int(phrase_key_count)
                    or len(new_story_terms) != int(stored_story_membership_count)
                ):
                    raise ValueError("graph delta occurrence counts do not match")
                receipt = self._runtime_receipt(
                    artifact_revision=int(append_revision),
                    chunk=graph_chunk,
                    occurrences=occurrences,
                    extraction_policy_sha256=(
                        self._extraction_policy.policy_sha256
                    ),
                    story_index_policy_sha256=(
                        self._story_index_policy.policy_sha256
                    ),
                    new_story_terms=new_story_terms,
                    story_evidence_chunk_retained=bool(
                        story_evidence_chunk_retained
                    ),
                    predecessor_chunk_id=(
                        None if predecessor is None else str(predecessor)
                    ),
                    successor_chunk_id=None if successor is None else str(successor),
                )
                if receipt.receipt_sha256 != str(runtime_receipt_sha256):
                    raise ValueError("graph runtime receipt hash does not match")
                expected_delta_sha256 = identity_sha256(
                    {
                        "schema": "conversation-graph-chunk-delta-v1",
                        "artifact_id": self._artifact_id,
                        "append_revision": int(append_revision),
                        "chunk_identity_sha256": receipt.chunk_identity_sha256,
                        "extraction_policy_sha256": (
                            self._extraction_policy.policy_sha256
                        ),
                        "occurrence_ids": [
                            value.occurrence_id for value in occurrences
                        ],
                        "phrase_keys": list(phrase_keys),
                        "story_index_policy_sha256": (
                            self._story_index_policy.policy_sha256
                        ),
                        "new_story_terms": [
                            {
                                "occurrence_id": occurrence.occurrence_id,
                                "term": term,
                            }
                            for term, occurrence in new_story_terms
                        ],
                        "story_evidence_chunk_retained": bool(
                            story_evidence_chunk_retained
                        ),
                        "predecessor_chunk_id": receipt.predecessor_chunk_id,
                        "successor_chunk_id": receipt.successor_chunk_id,
                        "runtime_receipt_sha256": receipt.receipt_sha256,
                    }
                )
                if expected_delta_sha256 != str(delta_sha256):
                    raise ValueError("graph delta hash does not match")
                if str(parent_checkpoint_sha256) != self._observed_checkpoint_sha256:
                    raise ValueError("graph checkpoint parent is not contiguous")
                expected_checkpoint_sha256 = identity_sha256(
                    {
                        "schema": "conversation-graph-checkpoint-v1",
                        "artifact_id": self._artifact_id,
                        "revision": int(append_revision),
                        "parent_checkpoint_sha256": str(parent_checkpoint_sha256),
                        "delta_sha256": str(delta_sha256),
                    }
                )
                if expected_checkpoint_sha256 != str(checkpoint_sha256):
                    raise ValueError("graph checkpoint hash does not match")
                self._restore_delta(
                    graph,
                    chunk=graph_chunk,
                    occurrences=occurrences,
                    new_story_terms=new_story_terms,
                    story_evidence_chunk_retained=bool(
                        story_evidence_chunk_retained
                    ),
                    receipt=receipt,
                )
                self._observed_revision = int(append_revision)
                self._observed_checkpoint_sha256 = str(checkpoint_sha256)
        state = target_row
        if (
            target_revision != self._observed_revision
            or target_checkpoint_sha256 != self._observed_checkpoint_sha256
        ):
            raise ValueError("resident graph does not match its durable checkpoint")
        stats = graph.stats()
        expected_state_sha256 = _state_sha256(
            artifact_id=self._artifact_id,
            revision=target_revision,
            chunk_count=int(state[1]),
            occurrence_count=int(state[2]),
            story_term_membership_count=int(state[3]),
            story_evidence_chunk_count=int(state[4]),
            ready_turn_count=int(state[5]),
            checkpoint_sha256=target_checkpoint_sha256,
        )
        if expected_state_sha256 != str(state[7]) or (
            stats.chunk_count != int(state[1])
            or stats.occurrence_count != int(state[2])
            or stats.story_term_membership_count != int(state[3])
            or stats.story_evidence_chunk_count != int(state[4])
        ):
            raise ValueError("resident graph state hash or counters do not match")
        return graph

    def graph(self) -> IncrementalConversationGraph:
        """Return the resident graph after adopting only unseen durable deltas."""

        with self._resident_lock:
            try:
                return self._sync_resident_unlocked()
            except BaseException:
                self._resident = None
                self._observed_revision = 0
                self._observed_checkpoint_sha256 = (
                    EMPTY_GRAPH_CHECKPOINT_SHA256
                )
                raise


__all__ = [
    "ConversationGraphStore",
    "EMPTY_GRAPH_CHECKPOINT_SHA256",
    "GRAPH_BOOTSTRAP_DEFAULT_MAX_TURNS",
    "GRAPH_BOOTSTRAP_HARD_MAX_TURNS",
    "GRAPH_ARTIFACT_FORMAT",
    "GraphBootstrapResult",
    "GraphCompilationReceipt",
    "GraphCompilationResult",
]
