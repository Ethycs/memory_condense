"""Durable journal for automatic memory extraction after indexing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Sequence

from memory_condense.domain.schemas import CreateOp, MemoryOps, MemoryType
from memory_condense.persistence.db import Database
from memory_condense.persistence.pending_ingest_store import (
    PendingIngestManifest,
    _failure_kind,
    _retry_at,
)

_SQLITE_PARAMETER_BUDGET = 500


@dataclass(frozen=True, slots=True)
class PendingCorrection:
    """One durable correction operation and its terminal audit mapping."""

    correction_id: str
    turn_id: str
    operation_index: int
    operation: CreateOp
    operation_sha256: str
    status: str
    created_at: str
    decided_at: str | None
    target_mem_id: str | None
    successor_mem_id: str | None
    reason: str | None


class PendingEnrichmentStore:
    """Per-turn, at-least-once enrichment receipts bound to ingest manifests."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def claim(self, manifest: PendingIngestManifest) -> str:
        return self.claim_many((manifest,))[manifest.turn_id]

    def claim_many(
        self,
        manifests: Sequence[PendingIngestManifest],
    ) -> dict[str, str]:
        """Atomically adopt manifest-bound receipts in bounded SQL batches."""
        by_turn: dict[str, PendingIngestManifest] = {}
        for manifest in manifests:
            previous = by_turn.setdefault(manifest.turn_id, manifest)
            if previous != manifest:
                raise ValueError("conflicting enrichment manifests in one claim")
        if not by_turn:
            return {}
        if not self._db.connection.in_transaction:
            raise RuntimeError("claim_many requires an active caller transaction")

        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (manifest.turn_id, manifest.sha256, now)
            for manifest in by_turn.values()
        ]
        row_width = 3
        batch_size = _SQLITE_PARAMETER_BUDGET // row_width
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            placeholders = ",".join("(?, ?, 'pending', ?, NULL)" for _ in batch)
            parameters = tuple(value for row in batch for value in row)
            self._db.execute(
                "INSERT INTO pending_enrichments "
                "(turn_id, ingest_manifest_sha256, status, created_at, enriched_at) "
                f"VALUES {placeholders} ON CONFLICT(turn_id) DO NOTHING",
                parameters,
            )

        # The v15 AFTER INSERT trigger classifies any exact receipt adopted
        # from a pre-migration source. Keeping that invariant in SQLite also
        # covers direct writers without duplicating a second scan here.
        turn_ids = list(by_turn)
        records: dict[str, tuple[str, str]] = {}
        for start in range(0, len(turn_ids), _SQLITE_PARAMETER_BUDGET):
            batch = turn_ids[start : start + _SQLITE_PARAMETER_BUDGET]
            placeholders = ",".join("?" for _ in batch)
            receipt_rows = self._db.execute(
                "SELECT turn_id, ingest_manifest_sha256, status "
                "FROM pending_enrichments "
                f"WHERE turn_id IN ({placeholders})",
                tuple(batch),
            ).fetchall()
            records.update(
                (str(turn_id), (str(manifest_sha256), str(status)))
                for turn_id, manifest_sha256, status in receipt_rows
            )

        statuses: dict[str, str] = {}
        for turn_id, manifest in by_turn.items():
            record = records.get(turn_id)
            if record is None or record[0] != manifest.sha256:
                raise ValueError("turn already has a different enrichment manifest")
            if record[1] not in {"pending", "enriched"}:
                raise ValueError("enrichment receipt has an invalid status")
            statuses[turn_id] = record[1]
        return statuses

    def count(self) -> int:
        return int(
            self._db.execute(
                "SELECT COUNT(*) FROM pending_enrichments WHERE status = 'pending'"
            ).fetchone()[0]
        )

    def status(self, turn_id: str) -> str | None:
        row = self._db.execute(
            "SELECT status FROM pending_enrichments WHERE turn_id = ?", (turn_id,)
        ).fetchone()
        return None if row is None else str(row[0])

    def pending_turn_ids(self, *, max_turns: int | None = None) -> list[str]:
        now = datetime.now(timezone.utc).isoformat()
        retry_class = self.choose_retry_class(now)
        if retry_class is None:
            return []

        def select(selected_retry_class: bool) -> list[str]:
            sql = (
                "SELECT e.turn_id FROM pending_enrichments AS e "
                "JOIN turns AS t ON t.turn_id = e.turn_id "
                "JOIN pending_ingests AS p ON p.turn_id = e.turn_id "
                "LEFT JOIN pending_enrichment_state AS s "
                "ON s.turn_id = e.turn_id "
                "WHERE e.status = 'pending' AND p.status = 'indexed' "
            )
            sql += (
                "AND NOT EXISTS (SELECT 1 FROM "
                "pending_enrichment_legacy_quarantine AS q "
                "WHERE q.turn_id = e.turn_id) "
            )
            if selected_retry_class:
                sql += "AND s.attempt_count > 0 AND s.next_attempt_at <= ? "
                params: tuple[object, ...] = (now,)
                sql += "ORDER BY s.next_attempt_at, s.last_attempt_at, "
                sql += "t.ordinal, e.turn_id"
            else:
                sql += "AND (s.turn_id IS NULL OR s.attempt_count = 0) "
                params = ()
                sql += "ORDER BY t.ordinal, e.turn_id"
            if max_turns is not None:
                sql += " LIMIT ?"
                params += (max_turns,)
            return [
                str(row[0]) for row in self._db.execute(sql, params).fetchall()
            ]

        selected = select(retry_class)
        # Another helper can finish the chosen class after the durable fairness
        # toggle commits. Fall through to the opposite eligible class in the
        # same tick instead of reporting a false-empty queue.
        return selected or select(not retry_class)

    def choose_retry_class(self, now: str) -> bool | None:
        """Alternate retry/fresh classes when both are currently eligible."""
        connection = self._db.connection
        if connection.in_transaction:
            raise RuntimeError(
                "choose_retry_class requires no active caller transaction"
            )
        try:
            connection.execute("BEGIN IMMEDIATE")
            base = (
                " FROM pending_enrichments AS e "
                "JOIN pending_ingests AS p ON p.turn_id = e.turn_id "
                "JOIN turns AS t ON t.turn_id = e.turn_id "
                "LEFT JOIN pending_enrichment_state AS s ON s.turn_id = e.turn_id "
                "WHERE e.status = 'pending' AND p.status = 'indexed' "
            )
            base += (
                "AND NOT EXISTS (SELECT 1 FROM "
                "pending_enrichment_legacy_quarantine AS q "
                "WHERE q.turn_id = e.turn_id) "
            )
            base += "AND "
            fresh = bool(
                connection.execute(
                    "SELECT EXISTS(SELECT 1" + base
                    + "(s.turn_id IS NULL OR s.attempt_count = 0))",
                ).fetchone()[0]
            )
            retry = bool(
                connection.execute(
                    "SELECT EXISTS(SELECT 1" + base
                    + "s.attempt_count > 0 AND s.next_attempt_at <= ?)",
                    (now,),
                ).fetchone()[0]
            )
            if fresh and retry:
                row = connection.execute(
                    "SELECT prefer_retry FROM pending_work_schedule "
                    "WHERE stage = 'enrichment'"
                ).fetchone()
                if row is None:
                    raise RuntimeError("enrichment retry schedule is missing")
                choice = bool(row[0])
                connection.execute(
                    "UPDATE pending_work_schedule "
                    "SET prefer_retry = 1 - prefer_retry "
                    "WHERE stage = 'enrichment'"
                )
            elif retry:
                choice = True
            elif fresh:
                choice = False
            else:
                choice = None
            connection.commit()
            return choice
        except BaseException:
            connection.rollback()
            raise

    def is_legacy_quarantined(self, turn_id: str) -> bool:
        """Return whether this exact receipt has ambiguous pre-v15 history."""
        return self._db.execute(
            "SELECT 1 FROM pending_enrichment_legacy_quarantine "
            "WHERE turn_id = ?",
            (turn_id,),
        ).fetchone() is not None

    def staged_result(
        self, turn_id: str
    ) -> tuple[MemoryOps, tuple[str, ...]] | None:
        """Return immutable validated ops and their live-evidence identities."""
        row = self._db.execute(
            "SELECT staged_ops_json, staged_chunk_ids_json, staged_result_sha256 "
            "FROM pending_enrichment_state "
            "WHERE turn_id = ?",
            (turn_id,),
        ).fetchone()
        if row is None or row[0] is None or row[1] is None:
            return None
        return self._decode_staged_result(row)

    def staged_digest(self, turn_id: str) -> str | None:
        """Return the immutable staged-result identity, including after clear."""
        row = self._db.execute(
            "SELECT staged_result_sha256 FROM pending_enrichment_state "
            "WHERE turn_id = ?",
            (turn_id,),
        ).fetchone()
        return None if row is None or row[0] is None else str(row[0])

    def require_staged_result(self, turn_id: str, expected_sha256: str) -> None:
        """CAS precondition for an operator consuming a finalized result."""
        if not self._db.connection.in_transaction:
            raise RuntimeError("staged-result CAS requires caller transaction")
        row = self._db.execute(
            "SELECT 1 FROM pending_enrichment_state AS s "
            "JOIN pending_enrichments AS e ON e.turn_id = s.turn_id "
            "WHERE s.turn_id = ? AND e.status = 'enriched' "
            "AND s.staged_ops_json IS NOT NULL "
            "AND s.staged_result_sha256 = ?",
            (turn_id, expected_sha256),
        ).fetchone()
        if row is None:
            raise RuntimeError("deferred correction was already consumed")

    def consume_staged_result(self, turn_id: str, expected_sha256: str) -> None:
        """Clear exactly one finalized staged payload under a caller lock."""
        self.require_staged_result(turn_id, expected_sha256)
        updated = self._db.execute(
            "UPDATE pending_enrichment_state "
            "SET staged_ops_json = NULL, staged_chunk_ids_json = NULL "
            "WHERE turn_id = ? AND staged_ops_json IS NOT NULL "
            "AND staged_result_sha256 = ?",
            (turn_id, expected_sha256),
        ).rowcount
        if updated != 1:
            raise RuntimeError("deferred correction was already consumed")

    @staticmethod
    def _decode_staged_result(
        row: Sequence[object],
    ) -> tuple[MemoryOps, tuple[str, ...]]:
        """Validate one replay row already captured from SQLite."""
        try:
            payload = json.loads(str(row[0]))
            chunk_ids = json.loads(str(row[1]))
            if not isinstance(payload, dict) or set(payload) != {
                "create",
                "update",
                "supersede",
                "delete",
                "pin",
            }:
                raise ValueError
            if (
                not isinstance(chunk_ids, list)
                or any(not isinstance(value, str) or not value for value in chunk_ids)
                or chunk_ids != sorted(set(chunk_ids))
            ):
                raise ValueError
            expected_sha256 = hashlib.sha256(
                (str(row[0]) + "\0" + str(row[1])).encode("utf-8")
            ).hexdigest()
            if row[2] != expected_sha256:
                raise ValueError
            return MemoryOps.model_validate(payload), tuple(chunk_ids)
        except Exception as exc:
            raise ValueError("staged enrichment result is invalid") from exc

    def stage_ops(
        self, turn_id: str, ops: MemoryOps, chunk_ids: Sequence[str]
    ) -> tuple[MemoryOps, tuple[str, ...]] | None:
        """Persist the first validated result and return that canonical winner.

        Concurrent model calls may disagree.  The first committed value owns
        the receipt; every helper subsequently replays that exact value.
        """
        canonical = json.dumps(
            ops.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        canonical_chunk_ids = json.dumps(
            sorted(set(chunk_ids)),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        result_sha256 = hashlib.sha256(
            (canonical + "\0" + canonical_chunk_ids).encode("utf-8")
        ).hexdigest()
        connection = self._db.connection
        if connection.in_transaction:
            raise RuntimeError("stage_ops requires no active caller transaction")
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO pending_enrichment_state "
                "(turn_id, attempt_count, last_attempt_at, next_attempt_at, "
                " last_error_kind, staged_ops_json, staged_chunk_ids_json, "
                " staged_result_sha256) "
                "SELECT ?, 0, NULL, NULL, NULL, ?, ?, ? WHERE EXISTS "
                "(SELECT 1 FROM pending_enrichments "
                " WHERE turn_id = ? AND status = 'pending') "
                "ON CONFLICT(turn_id) DO UPDATE SET "
                "attempt_count = 0, "
                "last_attempt_at = NULL, "
                "next_attempt_at = NULL, "
                "last_error_kind = NULL, "
                "staged_ops_json = excluded.staged_ops_json, "
                "staged_chunk_ids_json = excluded.staged_chunk_ids_json, "
                "staged_result_sha256 = excluded.staged_result_sha256 "
                "WHERE pending_enrichment_state.staged_ops_json IS NULL",
                (
                    turn_id,
                    canonical,
                    canonical_chunk_ids,
                    result_sha256,
                    turn_id,
                ),
            )
            row = connection.execute(
                "SELECT staged_ops_json, staged_chunk_ids_json, "
                "staged_result_sha256 "
                "FROM pending_enrichment_state "
                "WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            if row is None or row[0] is None or row[1] is None:
                status_row = connection.execute(
                    "SELECT status FROM pending_enrichments WHERE turn_id = ?",
                    (turn_id,),
                ).fetchone()
                if status_row is not None and str(status_row[0]) == "enriched":
                    connection.commit()
                    return None
                raise RuntimeError(
                    "cannot stage operations without a pending receipt"
                )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        # Decode the row captured while this transaction still owned the
        # writer lock. A concurrent helper may finalize and clear the payload
        # immediately after our commit; a post-commit re-read would then turn
        # successful concurrent completion into a false failure.
        return self._decode_staged_result(row)

    def record_failure(
        self,
        turn_id: str,
        error: BaseException,
        *,
        ignore_if_staged: bool = False,
    ) -> bool:
        """Persist one current T2 failure while leaving its receipt pending.

        A provider attempt that began before another helper staged the
        canonical result is stale. ``ignore_if_staged`` prevents that loser
        from hiding the winner behind a new cooldown. Failures after staging
        still advance normal bounded backoff.
        """
        kind = _failure_kind(error)
        now_value = datetime.now(timezone.utc)
        connection = self._db.connection
        if connection.in_transaction:
            connection.rollback()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT COALESCE(s.attempt_count, 0), "
                "COALESCE(s.staged_ops_json IS NOT NULL, 0) "
                "FROM pending_enrichments AS e "
                "LEFT JOIN pending_enrichment_state AS s "
                "ON s.turn_id = e.turn_id "
                "WHERE e.turn_id = ? AND e.status = 'pending'",
                (turn_id,),
            ).fetchone()
            if row is None or (ignore_if_staged and bool(row[1])):
                connection.commit()
                return False
            previous_attempts = int(row[0])
            connection.execute(
                "INSERT INTO pending_enrichment_state "
                "(turn_id, attempt_count, last_attempt_at, next_attempt_at, "
                " last_error_kind, staged_ops_json, staged_chunk_ids_json, "
                " staged_result_sha256) "
                "VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL) "
                "ON CONFLICT(turn_id) DO UPDATE SET "
                "attempt_count = excluded.attempt_count, "
                "last_attempt_at = excluded.last_attempt_at, "
                "next_attempt_at = excluded.next_attempt_at, "
                "last_error_kind = excluded.last_error_kind",
                (
                    turn_id,
                    previous_attempts + 1,
                    now_value.isoformat(),
                    _retry_at(previous_attempts, now_value),
                    kind,
                ),
            )
            connection.commit()
            return True
        except BaseException:
            connection.rollback()
            raise

    def clear_staged_result(self, turn_id: str) -> None:
        """Erase replay payload after success while retaining its digest."""
        if not self._db.connection.in_transaction:
            raise RuntimeError("clear_staged_result requires caller transaction")
        updated = self._db.execute(
            "UPDATE pending_enrichment_state "
            "SET staged_ops_json = NULL, staged_chunk_ids_json = NULL "
            "WHERE turn_id = ? AND staged_ops_json IS NOT NULL",
            (turn_id,),
        ).rowcount
        if updated == 0:
            row = self._db.execute(
                "SELECT staged_result_sha256 FROM pending_enrichment_state "
                "WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            if row is None or row[0] is None:
                raise RuntimeError("enrichment replay state has no staged digest")

    def finalize(self, turn_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        updated = self._db.execute(
            "UPDATE pending_enrichments SET status = 'enriched', enriched_at = ? "
            "WHERE turn_id = ? AND status = 'pending'",
            (now, turn_id),
        ).rowcount
        if updated == 0:
            row = self._db.execute(
                "SELECT status FROM pending_enrichments WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("enrichment completion has no receipt")
            if str(row[0]) != "enriched":
                raise RuntimeError("enrichment receipt has an invalid status")

    def record_legacy_disposition(self, turn_id: str) -> None:
        """Seal an explicit legacy discard beside, not as, T2 success."""
        if not self._db.connection.in_transaction:
            raise RuntimeError("legacy disposition requires caller transaction")
        self._db.execute(
            "INSERT INTO pending_enrichment_dispositions "
            "(turn_id, disposition, decided_at, reason) "
            "VALUES (?, 'discarded_legacy', ?, ?)",
            (
                turn_id,
                datetime.now(timezone.utc).isoformat(),
                "pre-v15 source-order ambiguity; T1 evidence retained",
            ),
        )

    @staticmethod
    def _correction_payload(operation: CreateOp) -> tuple[str, str]:
        canonical = json.dumps(
            operation.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _correction_id(
        turn_id: str, operation_index: int, canonical: str
    ) -> str:
        return hashlib.sha256(
            (turn_id + "\0" + str(operation_index) + "\0" + canonical).encode(
                "utf-8"
            )
        ).hexdigest()

    @classmethod
    def _decode_correction(cls, row: Sequence[object]) -> PendingCorrection:
        operation_index = int(row[2])
        canonical = str(row[3])
        operation = CreateOp.model_validate(json.loads(canonical))
        expected_canonical, expected_sha256 = cls._correction_payload(operation)
        correction_id = str(row[0])
        turn_id = str(row[1])
        if (
            operation_index < 0
            or operation.type is not MemoryType.CORRECTION
            or canonical != expected_canonical
            or str(row[4]) != expected_sha256
            or correction_id
            != cls._correction_id(turn_id, operation_index, canonical)
        ):
            raise ValueError("pending correction receipt is inconsistent")
        return PendingCorrection(
            correction_id=correction_id,
            turn_id=turn_id,
            operation_index=operation_index,
            operation=operation,
            operation_sha256=expected_sha256,
            status=str(row[5]),
            created_at=str(row[6]),
            decided_at=None if row[7] is None else str(row[7]),
            target_mem_id=None if row[8] is None else str(row[8]),
            successor_mem_id=None if row[9] is None else str(row[9]),
            reason=None if row[10] is None else str(row[10]),
        )

    def queue_corrections(
        self, turn_id: str, corrections: Sequence[CreateOp]
    ) -> tuple[PendingCorrection, ...]:
        """Publish grounded corrections inside the T2 completion transaction."""
        if not self._db.connection.in_transaction:
            raise RuntimeError("correction publication requires caller transaction")
        now = datetime.now(timezone.utc).isoformat()
        for operation_index, operation in enumerate(corrections):
            if operation.type is not MemoryType.CORRECTION:
                raise ValueError("correction queue accepts only Correction creates")
            canonical, operation_sha256 = self._correction_payload(operation)
            correction_id = self._correction_id(
                turn_id, operation_index, canonical
            )
            self._db.execute(
                "INSERT INTO pending_corrections "
                "(correction_id, turn_id, operation_index, operation_json, "
                " operation_sha256, "
                " status, created_at, decided_at, target_mem_id, "
                " successor_mem_id, reason) "
                "SELECT ?, ?, ?, ?, ?, 'pending', ?, NULL, NULL, NULL, NULL "
                "WHERE EXISTS (SELECT 1 FROM pending_enrichments "
                " WHERE turn_id = ? AND status = 'pending') "
                "ON CONFLICT(correction_id) DO NOTHING",
                (
                    correction_id,
                    turn_id,
                    operation_index,
                    canonical,
                    operation_sha256,
                    now,
                    turn_id,
                ),
            )
        return tuple(
            correction
            for correction in self.list_corrections(turn_id=turn_id)
            if correction.status == "pending"
        )

    def list_corrections(
        self, *, status: str | None = "pending", turn_id: str | None = None
    ) -> list[PendingCorrection]:
        if status not in {None, "pending", "resolved", "dismissed"}:
            raise ValueError("invalid correction status")
        clauses: list[str] = []
        params: list[object] = []
        if status is not None:
            clauses.append("c.status = ?")
            params.append(status)
        if turn_id is not None:
            clauses.append("c.turn_id = ?")
            params.append(turn_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._db.execute(
            "SELECT c.correction_id, c.turn_id, c.operation_index, "
            "c.operation_json, c.operation_sha256, c.status, c.created_at, "
            "c.decided_at, c.target_mem_id, c.successor_mem_id, c.reason "
            "FROM pending_corrections AS c "
            "JOIN turns AS t ON t.turn_id = c.turn_id"
            + where
            + " ORDER BY t.ordinal, c.operation_index, c.correction_id",
            tuple(params),
        ).fetchall()
        return [self._decode_correction(row) for row in rows]

    def get_correction(self, correction_id: str) -> PendingCorrection | None:
        rows = self._db.execute(
            "SELECT correction_id, turn_id, operation_index, operation_json, "
            "operation_sha256, status, created_at, decided_at, target_mem_id, "
            "successor_mem_id, reason FROM pending_corrections "
            "WHERE correction_id = ?",
            (correction_id,),
        ).fetchall()
        return None if not rows else self._decode_correction(rows[0])

    def require_pending_correction(
        self, correction_id: str, expected_sha256: str
    ) -> None:
        if not self._db.connection.in_transaction:
            raise RuntimeError("correction CAS requires caller transaction")
        row = self._db.execute(
            "SELECT 1 FROM pending_corrections WHERE correction_id = ? "
            "AND operation_sha256 = ? AND status = 'pending'",
            (correction_id, expected_sha256),
        ).fetchone()
        if row is None:
            raise RuntimeError("deferred correction was already decided")

    def resolve_correction(
        self, correction_id: str, *, target_mem_id: str, successor_mem_id: str
    ) -> None:
        if not self._db.connection.in_transaction:
            raise RuntimeError("correction resolution requires caller transaction")
        updated = self._db.execute(
            "UPDATE pending_corrections SET status = 'resolved', decided_at = ?, "
            "target_mem_id = ?, successor_mem_id = ?, reason = ? "
            "WHERE correction_id = ? AND status = 'pending'",
            (
                datetime.now(timezone.utc).isoformat(),
                target_mem_id,
                successor_mem_id,
                "operator bound grounded correction to reviewed target",
                correction_id,
            ),
        ).rowcount
        if updated != 1:
            raise RuntimeError("deferred correction was already decided")

    def dismiss_correction(self, correction_id: str) -> None:
        if not self._db.connection.in_transaction:
            raise RuntimeError("correction dismissal requires caller transaction")
        updated = self._db.execute(
            "UPDATE pending_corrections SET status = 'dismissed', decided_at = ?, "
            "reason = ? WHERE correction_id = ? AND status = 'pending'",
            (
                datetime.now(timezone.utc).isoformat(),
                "operator explicitly dismissed deferred correction",
                correction_id,
            ),
        ).rowcount
        if updated != 1:
            raise RuntimeError("deferred correction was already decided")

    def pending_correction_count(self) -> int:
        return int(
            self._db.execute(
                "SELECT COUNT(*) FROM pending_corrections "
                "WHERE status = 'pending'"
            ).fetchone()[0]
        )


__all__ = ["PendingCorrection", "PendingEnrichmentStore"]
