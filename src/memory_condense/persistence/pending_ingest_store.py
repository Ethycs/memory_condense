"""Durable, provider-free journal for incomplete turn-to-index publication."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

from memory_condense.domain.schemas import Chunk, Turn
from memory_condense.persistence.db import Database


_MANIFEST_FORMAT = "memory-condense-pending-ingest-v1"
_SQLITE_PARAMETER_BUDGET = 500
_MAX_FAILURE_KIND_CHARS = 80


def _failure_kind(error: BaseException) -> str:
    """Return a bounded class label without persisting exception payloads."""
    raw = type(error).__name__
    safe = "".join(
        character if character.isascii() and character.isalnum() else "_"
        for character in raw
    ).strip("_")
    return (safe or "Exception")[:_MAX_FAILURE_KIND_CHARS]


def _retry_at(previous_attempts: int, now: datetime) -> str:
    """Retry once immediately, then apply bounded exponential backoff."""
    delay_seconds = (
        0.0
        if previous_attempts == 0
        else min(60.0, 1.0 * (2 ** min(previous_attempts - 1, 6)))
    )
    return (now + timedelta(seconds=delay_seconds)).isoformat()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class PendingIngestAlreadyIndexedError(RuntimeError):
    """A stale helper attempted to complete a receipt already sealed indexed."""

    def __init__(self, turn_ids: Sequence[str]) -> None:
        self.turn_ids = tuple(sorted(set(turn_ids)))
        super().__init__(
            "pending ingest receipts are already indexed: "
            + ", ".join(self.turn_ids)
        )


def backfill_legacy_ingest_receipts(conn: sqlite3.Connection) -> None:
    """Seal pre-v13 chunk topologies and their global ID reservations."""
    rows = conn.execute(
        "SELECT turn_id, chunk_id, text, start_char, end_char, token_count "
        "FROM chunks ORDER BY turn_id, start_char, end_char, chunk_id"
    ).fetchall()
    grouped: dict[str, list[tuple]] = {}
    for row in rows:
        grouped.setdefault(str(row[0]), []).append(row)
    now = datetime.now(timezone.utc).isoformat()
    receipts: list[tuple[str, str, str, str, str, str]] = []
    reservations: list[tuple[str, str, int, int, int, str]] = []
    for turn_id, chunk_rows in grouped.items():
        manifest = PendingIngestManifest(
            turn_id=turn_id,
            chunks=tuple(
                PendingChunkManifest(
                    chunk_id=str(row[1]),
                    start_char=int(row[3]),
                    end_char=int(row[4]),
                    token_count=int(row[5]),
                    text_sha256=_sha256_text(str(row[2])),
                )
                for row in chunk_rows
            ),
        )
        # V12 has no claim distinguishing interrupted work from intentional
        # lexical-only or retired state. Never infer replay permission.
        receipts.append(
            (
                turn_id,
                manifest.sha256,
                manifest.canonical_json,
                "indexed",
                now,
                now,
            )
        )
        reservations.extend(
            (
                row.chunk_id,
                turn_id,
                row.start_char,
                row.end_char,
                row.token_count,
                row.text_sha256,
            )
            for row in manifest.chunks
        )
    conn.executemany(
        "INSERT INTO pending_ingests "
        "(turn_id, manifest_sha256, manifest_json, status, created_at, indexed_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        receipts,
    )
    conn.executemany(
        "INSERT INTO ingest_chunk_reservations "
        "(chunk_id, turn_id, start_char, end_char, token_count, text_sha256) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        reservations,
    )


@dataclass(frozen=True)
class PendingChunkManifest:
    """Text-free identity needed to replay one exact derived chunk."""

    chunk_id: str
    start_char: int
    end_char: int
    token_count: int
    text_sha256: str

    def payload(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "end_char": self.end_char,
            "start_char": self.start_char,
            "text_sha256": self.text_sha256,
            "token_count": self.token_count,
        }


@dataclass(frozen=True)
class PendingIngestManifest:
    """Canonical replay receipt for one turn's complete chunk population."""

    turn_id: str
    chunks: tuple[PendingChunkManifest, ...]

    @classmethod
    def build(
        cls,
        turn: Turn,
        chunks: Sequence[Chunk],
    ) -> PendingIngestManifest:
        rows: list[PendingChunkManifest] = []
        seen: set[str] = set()
        for chunk in sorted(
            chunks,
            key=lambda value: (
                value.start_char,
                value.end_char,
                value.chunk_id,
            ),
        ):
            if chunk.turn_id != turn.turn_id:
                raise ValueError("pending chunk belongs to a different turn")
            if chunk.chunk_id in seen:
                raise ValueError("pending manifest contains a duplicate chunk_id")
            seen.add(chunk.chunk_id)
            if (
                chunk.start_char < 0
                or chunk.end_char <= chunk.start_char
                or chunk.end_char > len(turn.text)
                or turn.text[chunk.start_char : chunk.end_char] != chunk.text
            ):
                raise ValueError("pending chunk does not match its turn span")
            if chunk.token_count < 0:
                raise ValueError("pending chunk token_count must be non-negative")
            rows.append(
                PendingChunkManifest(
                    chunk_id=chunk.chunk_id,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    token_count=chunk.token_count,
                    text_sha256=_sha256_text(chunk.text),
                )
            )
        return cls(turn_id=turn.turn_id, chunks=tuple(rows))

    @classmethod
    def from_json(cls, value: str) -> PendingIngestManifest:
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("pending ingest manifest is not valid JSON") from exc
        if not isinstance(payload, dict) or set(payload) != {
            "chunks",
            "format",
            "turn_id",
        }:
            raise ValueError("pending ingest manifest has an invalid shape")
        if payload["format"] != _MANIFEST_FORMAT:
            raise ValueError("pending ingest manifest has an unknown format")
        turn_id = payload["turn_id"]
        raw_chunks = payload["chunks"]
        if not isinstance(turn_id, str) or not turn_id:
            raise ValueError("pending ingest manifest has an invalid turn_id")
        if not isinstance(raw_chunks, list):
            raise ValueError("pending ingest manifest chunks must be a list")

        chunks: list[PendingChunkManifest] = []
        seen: set[str] = set()
        for raw in raw_chunks:
            if not isinstance(raw, dict) or set(raw) != {
                "chunk_id",
                "end_char",
                "start_char",
                "text_sha256",
                "token_count",
            }:
                raise ValueError("pending chunk manifest has an invalid shape")
            chunk_id = raw["chunk_id"]
            text_sha256 = raw["text_sha256"]
            start_char = raw["start_char"]
            end_char = raw["end_char"]
            token_count = raw["token_count"]
            if not isinstance(chunk_id, str) or not chunk_id or chunk_id in seen:
                raise ValueError("pending chunk manifest has an invalid chunk_id")
            if (
                type(start_char) is not int
                or type(end_char) is not int
                or type(token_count) is not int
                or start_char < 0
                or end_char <= start_char
                or token_count < 0
            ):
                raise ValueError("pending chunk manifest has invalid coordinates")
            if (
                not isinstance(text_sha256, str)
                or len(text_sha256) != 64
                or any(character not in "0123456789abcdef" for character in text_sha256)
            ):
                raise ValueError("pending chunk manifest has an invalid text hash")
            seen.add(chunk_id)
            chunks.append(
                PendingChunkManifest(
                    chunk_id=chunk_id,
                    start_char=start_char,
                    end_char=end_char,
                    token_count=token_count,
                    text_sha256=text_sha256,
                )
            )
        manifest = cls(turn_id=turn_id, chunks=tuple(chunks))
        if manifest.canonical_json != value:
            raise ValueError("pending ingest manifest is not canonical")
        return manifest

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            {
                "chunks": [chunk.payload() for chunk in self.chunks],
                "format": _MANIFEST_FORMAT,
                "turn_id": self.turn_id,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @property
    def sha256(self) -> str:
        return _sha256_text(self.canonical_json)

    def reconstruct(self, turn: Turn) -> list[Chunk]:
        """Rebuild source chunks without retaining text in the journal."""
        if turn.turn_id != self.turn_id:
            raise ValueError("pending ingest manifest belongs to a different turn")
        output: list[Chunk] = []
        for row in self.chunks:
            if row.end_char > len(turn.text):
                raise ValueError("pending chunk span exceeds its turn")
            text = turn.text[row.start_char : row.end_char]
            if _sha256_text(text) != row.text_sha256:
                raise ValueError("pending chunk hash does not match its turn span")
            output.append(
                Chunk(
                    chunk_id=row.chunk_id,
                    turn_id=turn.turn_id,
                    text=text,
                    start_char=row.start_char,
                    end_char=row.end_char,
                    token_count=row.token_count,
                )
            )
        return output


class PendingIngestStore:
    """Shared manifests: compatible writers help instead of competing."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def claim(self, manifest: PendingIngestManifest) -> str:
        """Insert/adopt an exact manifest and return its durable status.

        For compatibility, a caller without a transaction gets one opened and
        still owns the eventual commit. Bulk :meth:`claim_many` requires its
        caller to establish the transaction explicitly.
        """
        connection = self._db.connection
        started = not connection.in_transaction
        if started:
            connection.execute("BEGIN IMMEDIATE")
        try:
            return self.claim_many((manifest,))[manifest.turn_id]
        except BaseException:
            if started:
                connection.rollback()
            raise

    def claim_many(
        self,
        manifests: Sequence[PendingIngestManifest],
    ) -> dict[str, str]:
        """Atomically adopt exact manifests and return status by turn ID.

        The caller owns the surrounding write transaction. Reads, receipt
        inserts, and globally unique chunk reservations are set-oriented and
        split only to remain below SQLite's conservative parameter budget.
        """
        by_turn: dict[str, PendingIngestManifest] = {}
        for manifest in manifests:
            previous = by_turn.setdefault(manifest.turn_id, manifest)
            if previous != manifest:
                raise ValueError("conflicting pending manifests in one claim")
        if not by_turn:
            return {}
        if not self._db.connection.in_transaction:
            raise RuntimeError("claim_many requires an active caller transaction")

        turn_ids = list(by_turn)
        records = self._get_records(turn_ids)
        durable = self._durable_manifests(turn_ids)
        now = datetime.now(timezone.utc).isoformat()
        pending_rows: list[tuple[object, ...]] = []

        for turn_id, manifest in by_turn.items():
            record = records.get(turn_id)
            durable_record = durable.get(turn_id)
            if record is not None:
                if record[0] != manifest:
                    raise ValueError(
                        "turn already has a different pending chunk manifest"
                    )
                if durable_record is not None and durable_record[0] != manifest:
                    raise ValueError(
                        "turn durable chunk topology no longer matches its receipt"
                    )
                if (
                    durable_record is None
                    and record[1] == "indexed"
                    and manifest.chunks
                ):
                    raise ValueError(
                        "indexed ingest receipt has no durable chunk topology"
                    )
                continue

            if durable_record is not None and durable_record[0] != manifest:
                raise ValueError(
                    "turn already has a different durable chunk topology"
                )
            already_indexed = durable_record is not None and durable_record[1]
            status = (
                "indexed" if not manifest.chunks or already_indexed else "pending"
            )
            pending_rows.append(
                (
                    turn_id,
                    manifest.sha256,
                    manifest.canonical_json,
                    status,
                    now,
                    now if status == "indexed" else None,
                )
            )

        self._insert_rows(
            "INSERT INTO pending_ingests "
            "(turn_id, manifest_sha256, manifest_json, status, created_at, indexed_at) "
            "VALUES ",
            "(?, ?, ?, ?, ?, ?)",
            pending_rows,
            " ON CONFLICT(turn_id) DO NOTHING",
        )

        # Re-read every row even when it existed before this call. That keeps
        # exact-manifest adoption fail closed if this connection is ever used
        # without SQLite's expected BEGIN IMMEDIATE writer serialization.
        records = self._get_records(turn_ids)
        for turn_id, manifest in by_turn.items():
            record = records.get(turn_id)
            if record is None or record[0] != manifest:
                raise ValueError(
                    "turn already has a different pending chunk manifest"
                )

        self._claim_reservations_many(tuple(by_turn.values()))
        return {turn_id: records[turn_id][1] for turn_id in by_turn}

    def get(self, turn_id: str) -> PendingIngestManifest | None:
        return self.get_many((turn_id,)).get(turn_id)

    def get_many(
        self,
        turn_ids: Sequence[str],
    ) -> dict[str, PendingIngestManifest]:
        """Load existing sealed topologies with bounded set queries."""
        return {
            turn_id: record[0]
            for turn_id, record in self._get_records(turn_ids).items()
        }

    def validate_chunk_membership(
        self,
        chunks: Sequence[Chunk],
        *,
        allow_indexed_rebuild: bool = False,
    ) -> None:
        """Reject unowned chunks and terminal indexed-member reactivation."""
        by_turn: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            by_turn.setdefault(chunk.turn_id, []).append(chunk)
        if not by_turn:
            return

        receipt_status: dict[str, str] = {}
        turn_ids = list(by_turn)
        for start in range(0, len(turn_ids), 500):
            batch = turn_ids[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = self._db.execute(
                "SELECT turn_id, status FROM pending_ingests "
                f"WHERE turn_id IN ({placeholders})",
                tuple(batch),
            ).fetchall()
            receipt_status.update((str(row[0]), str(row[1])) for row in rows)

        reservations: dict[str, tuple] = {}
        chunk_ids = list(dict.fromkeys(chunk.chunk_id for chunk in chunks))
        for start in range(0, len(chunk_ids), 500):
            batch = chunk_ids[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = self._db.execute(
                "SELECT chunk_id, turn_id, start_char, end_char, token_count, "
                "text_sha256 FROM ingest_chunk_reservations "
                f"WHERE chunk_id IN ({placeholders})",
                tuple(batch),
            ).fetchall()
            reservations.update((str(row[0]), row) for row in rows)

        durable_state: dict[str, tuple] = {}
        for start in range(0, len(chunk_ids), 500):
            batch = chunk_ids[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = self._db.execute(
                "SELECT chunk_id, embedding IS NOT NULL, "
                "hnsw_label IS NOT NULL, term_count IS NOT NULL FROM chunks "
                f"WHERE chunk_id IN ({placeholders})",
                tuple(batch),
            ).fetchall()
            durable_state.update((str(row[0]), row) for row in rows)

        terminal_turns: set[str] = set()
        for turn_id, turn_chunks in by_turn.items():
            for chunk in turn_chunks:
                reservation = reservations.get(chunk.chunk_id)
                if reservation is None:
                    if turn_id not in receipt_status:
                        # Raw TranscriptStore turns intentionally have no
                        # receipt and remain valid first-time index inputs.
                        continue
                    raise ValueError(
                        "chunk is not a member of its turn ingest manifest"
                    )
                if (
                    str(reservation[1]) != turn_id
                    or chunk.start_char != int(reservation[2])
                    or chunk.end_char != int(reservation[3])
                    or chunk.token_count != int(reservation[4])
                    or _sha256_text(chunk.text) != str(reservation[5])
                ):
                    raise ValueError(
                        "chunk source fields do not match its global reservation"
                    )
                if (
                    not allow_indexed_rebuild
                    and receipt_status.get(turn_id) == "indexed"
                ):
                    state = durable_state.get(chunk.chunk_id)
                    if state is None or not all(bool(value) for value in state[1:]):
                        terminal_turns.add(turn_id)
        if terminal_turns:
            raise PendingIngestAlreadyIndexedError(tuple(terminal_turns))

    def _claim_reservations_many(
        self,
        manifests: Sequence[PendingIngestManifest],
    ) -> None:
        """Insert or verify globally unique ownership for many manifests."""
        expected: dict[str, tuple[str, PendingChunkManifest]] = {}
        expected_by_turn: dict[str, set[str]] = {
            manifest.turn_id: set() for manifest in manifests
        }
        reservation_rows: list[tuple[object, ...]] = []
        for manifest in manifests:
            turn_chunk_ids = expected_by_turn[manifest.turn_id]
            for row in manifest.chunks:
                owner = expected.get(row.chunk_id)
                if owner is not None:
                    raise ValueError(
                        "pending manifests reuse one chunk identity"
                    )
                expected[row.chunk_id] = (manifest.turn_id, row)
                turn_chunk_ids.add(row.chunk_id)
                reservation_rows.append(
                    (
                        row.chunk_id,
                        manifest.turn_id,
                        row.start_char,
                        row.end_char,
                        row.token_count,
                        row.text_sha256,
                    )
                )

        self._insert_rows(
            "INSERT INTO ingest_chunk_reservations "
            "(chunk_id, turn_id, start_char, end_char, token_count, text_sha256) "
            "VALUES ",
            "(?, ?, ?, ?, ?, ?)",
            reservation_rows,
            " ON CONFLICT(chunk_id) DO NOTHING",
        )

        actual: dict[str, tuple] = {}
        chunk_ids = list(expected)
        for batch in self._batches(chunk_ids):
            placeholders = ",".join("?" for _ in batch)
            rows = self._db.execute(
                "SELECT chunk_id, turn_id, start_char, end_char, token_count, "
                "text_sha256 FROM ingest_chunk_reservations "
                f"WHERE chunk_id IN ({placeholders})",
                tuple(batch),
            ).fetchall()
            actual.update((str(row[0]), row) for row in rows)

        for chunk_id, (turn_id, expected_row) in expected.items():
            row = actual.get(chunk_id)
            if row is None or str(row[1]) != turn_id:
                raise ValueError(
                    "chunk identity is reserved by a different ingest manifest"
                )
            if (
                int(row[2]) != expected_row.start_char
                or int(row[3]) != expected_row.end_char
                or int(row[4]) != expected_row.token_count
                or str(row[5]) != expected_row.text_sha256
            ):
                raise ValueError("ingest chunk reservation is inconsistent")

        owned_by_turn = {turn_id: set() for turn_id in expected_by_turn}
        for batch in self._batches(list(expected_by_turn)):
            placeholders = ",".join("?" for _ in batch)
            rows = self._db.execute(
                "SELECT turn_id, chunk_id FROM ingest_chunk_reservations "
                f"WHERE turn_id IN ({placeholders})",
                tuple(batch),
            ).fetchall()
            for turn_id, chunk_id in rows:
                owned_by_turn[str(turn_id)].add(str(chunk_id))
        if owned_by_turn != expected_by_turn:
            raise ValueError("turn has a different reserved chunk topology")

    @staticmethod
    def _batches(values: Sequence[str]) -> list[Sequence[str]]:
        return [
            values[start : start + _SQLITE_PARAMETER_BUDGET]
            for start in range(0, len(values), _SQLITE_PARAMETER_BUDGET)
        ]

    def _insert_rows(
        self,
        prefix: str,
        row_placeholder: str,
        rows: Sequence[tuple[object, ...]],
        suffix: str,
    ) -> None:
        if not rows:
            return
        row_width = len(rows[0])
        if row_width <= 0 or any(len(row) != row_width for row in rows):
            raise ValueError("batched SQL rows must have one fixed positive width")
        batch_size = max(1, _SQLITE_PARAMETER_BUDGET // row_width)
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            placeholders = ",".join(row_placeholder for _ in batch)
            parameters = tuple(value for row in batch for value in row)
            self._db.execute(prefix + placeholders + suffix, parameters)

    def _get_records(
        self,
        turn_ids: Sequence[str],
    ) -> dict[str, tuple[PendingIngestManifest, str]]:
        records: dict[str, tuple[PendingIngestManifest, str]] = {}
        for batch in self._batches(list(dict.fromkeys(turn_ids))):
            placeholders = ",".join("?" for _ in batch)
            rows = self._db.execute(
                "SELECT turn_id, manifest_sha256, manifest_json, status "
                "FROM pending_ingests "
                f"WHERE turn_id IN ({placeholders})",
                tuple(batch),
            ).fetchall()
            for turn_id, manifest_sha256, manifest_json, status_value in rows:
                turn_id_value = str(turn_id)
                manifest = PendingIngestManifest.from_json(str(manifest_json))
                if (
                    manifest.turn_id != turn_id_value
                    or manifest.sha256 != str(manifest_sha256)
                ):
                    raise ValueError(
                        "pending ingest manifest receipt is inconsistent"
                    )
                status = str(status_value)
                if status not in {"pending", "indexed"}:
                    raise ValueError("pending ingest manifest has an invalid status")
                records[turn_id_value] = (manifest, status)
        return records

    def _get_record(
        self, turn_id: str
    ) -> tuple[PendingIngestManifest, str] | None:
        return self._get_records((turn_id,)).get(turn_id)

    def _durable_manifests(
        self,
        turn_ids: Sequence[str],
    ) -> dict[str, tuple[PendingIngestManifest, bool]]:
        grouped: dict[str, list[tuple]] = {}
        for batch in self._batches(list(dict.fromkeys(turn_ids))):
            placeholders = ",".join("?" for _ in batch)
            rows = self._db.execute(
                "SELECT turn_id, chunk_id, text, start_char, end_char, token_count, "
                "embedding IS NOT NULL, hnsw_label IS NOT NULL, "
                "term_count IS NOT NULL FROM chunks "
                f"WHERE turn_id IN ({placeholders}) "
                "ORDER BY turn_id, start_char, end_char, chunk_id",
                tuple(batch),
            ).fetchall()
            for row in rows:
                grouped.setdefault(str(row[0]), []).append(row)

        output: dict[str, tuple[PendingIngestManifest, bool]] = {}
        for turn_id, rows in grouped.items():
            rows.sort(key=lambda row: (int(row[3]), int(row[4]), str(row[1])))
            manifest = PendingIngestManifest(
                turn_id=turn_id,
                chunks=tuple(
                    PendingChunkManifest(
                        chunk_id=str(row[1]),
                        start_char=int(row[3]),
                        end_char=int(row[4]),
                        token_count=int(row[5]),
                        text_sha256=_sha256_text(str(row[2])),
                    )
                    for row in rows
                ),
            )
            complete = all(
                bool(row[6]) and bool(row[7]) and bool(row[8])
                for row in rows
            )
            output[turn_id] = (manifest, complete)
        return output

    def _durable_manifest(
        self, turn_id: str
    ) -> tuple[PendingIngestManifest, bool] | None:
        return self._durable_manifests((turn_id,)).get(turn_id)

    def list_pending(self) -> list[PendingIngestManifest]:
        rows = self._db.execute(
            "SELECT turn_id, manifest_sha256, manifest_json "
            "FROM pending_ingests WHERE status = 'pending' "
            "ORDER BY created_at, turn_id"
        ).fetchall()
        output: list[PendingIngestManifest] = []
        for turn_id, manifest_sha256, manifest_json in rows:
            manifest = PendingIngestManifest.from_json(str(manifest_json))
            if (
                manifest.turn_id != str(turn_id)
                or manifest.sha256 != str(manifest_sha256)
            ):
                raise ValueError("pending ingest manifest receipt is inconsistent")
            output.append(manifest)
        return output

    def count(self) -> int:
        return int(
            self._db.execute(
                "SELECT COUNT(*) FROM pending_ingests WHERE status = 'pending'"
            ).fetchone()[0]
        )

    def choose_retry_class(self, now: str) -> bool | None:
        """Alternate retry/fresh classes when both are currently eligible."""
        connection = self._db.connection
        if connection.in_transaction:
            raise RuntimeError(
                "choose_retry_class requires no active caller transaction"
            )
        try:
            connection.execute("BEGIN IMMEDIATE")
            fresh = bool(
                connection.execute(
                    "SELECT EXISTS(SELECT 1 FROM pending_ingests AS p "
                    "LEFT JOIN pending_ingest_attempts AS a "
                    "ON a.turn_id = p.turn_id "
                    "WHERE p.status = 'pending' AND a.turn_id IS NULL)"
                ).fetchone()[0]
            )
            retry = bool(
                connection.execute(
                    "SELECT EXISTS(SELECT 1 FROM pending_ingests AS p "
                    "JOIN pending_ingest_attempts AS a ON a.turn_id = p.turn_id "
                    "WHERE p.status = 'pending' AND a.next_attempt_at <= ?)",
                    (now,),
                ).fetchone()[0]
            )
            if fresh and retry:
                row = connection.execute(
                    "SELECT prefer_retry FROM pending_work_schedule "
                    "WHERE stage = 'ingest'"
                ).fetchone()
                if row is None:
                    raise RuntimeError("ingest retry schedule is missing")
                choice = bool(row[0])
                connection.execute(
                    "UPDATE pending_work_schedule "
                    "SET prefer_retry = 1 - prefer_retry "
                    "WHERE stage = 'ingest'"
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

    def record_failure(
        self, turn_ids: Sequence[str], error: BaseException
    ) -> None:
        """Persist one failed T1 attempt without consuming its receipt.

        Failed rows sort behind never-attempted work.  This gives later turns
        a chance to become searchable while retaining every failed manifest
        for inspection and deterministic repair.
        """
        unique_ids = tuple(dict.fromkeys(str(turn_id) for turn_id in turn_ids))
        if not unique_ids:
            return
        kind = _failure_kind(error)
        now_value = datetime.now(timezone.utc)
        connection = self._db.connection
        if connection.in_transaction:
            connection.rollback()
        try:
            connection.execute("BEGIN IMMEDIATE")
            previous: dict[str, int] = {}
            for start in range(0, len(unique_ids), _SQLITE_PARAMETER_BUDGET):
                batch = unique_ids[start : start + _SQLITE_PARAMETER_BUDGET]
                placeholders = ",".join("?" for _ in batch)
                previous.update(
                    (str(row[0]), int(row[1]))
                    for row in connection.execute(
                        "SELECT turn_id, attempt_count "
                        "FROM pending_ingest_attempts "
                        f"WHERE turn_id IN ({placeholders})",
                        tuple(batch),
                    ).fetchall()
                )
            dissolving_failed_cohort = len(unique_ids) > 1 and all(
                previous.get(turn_id, 0) == 1 for turn_id in unique_ids
            )
            connection.executemany(
                "INSERT INTO pending_ingest_attempts "
                "(turn_id, attempt_count, last_attempt_at, next_attempt_at, "
                " last_error_kind) "
                "SELECT ?, ?, ?, ?, ? WHERE EXISTS "
                "(SELECT 1 FROM pending_ingests "
                " WHERE turn_id = ? AND status = 'pending') "
                "ON CONFLICT(turn_id) DO UPDATE SET "
                "attempt_count = excluded.attempt_count, "
                "last_attempt_at = excluded.last_attempt_at, "
                "next_attempt_at = excluded.next_attempt_at, "
                "last_error_kind = excluded.last_error_kind",
                [
                    (
                        turn_id,
                        previous.get(turn_id, 0) + 1,
                        now_value.isoformat(),
                        (
                            now_value.isoformat()
                            if dissolving_failed_cohort
                            else _retry_at(previous.get(turn_id, 0), now_value)
                        ),
                        kind,
                        turn_id,
                    )
                    for turn_id in unique_ids
                ],
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    def finalize(self, manifests: Sequence[PendingIngestManifest]) -> None:
        """Prove complete durable indexing, then seal receipts before commit."""
        by_turn: dict[str, PendingIngestManifest] = {}
        for manifest in manifests:
            previous = by_turn.setdefault(manifest.turn_id, manifest)
            if previous != manifest:
                raise ValueError("conflicting pending manifests in one finalizer")

        records = self._get_records(tuple(by_turn))
        already_indexed: list[str] = []
        for manifest in by_turn.values():
            record = records.get(manifest.turn_id)
            if record is None:
                raise RuntimeError("pending ingest completion has no manifest receipt")
            if record[0] != manifest:
                raise ValueError("turn pending manifest changed before completion")
            if record[1] != "pending":
                already_indexed.append(manifest.turn_id)
        if already_indexed:
            # The caller owns BEGIN IMMEDIATE. Raising here rolls back any
            # dense/BM25 writes it staged before entering this finalizer, so a
            # stale helper cannot resurrect a later retirement.
            raise PendingIngestAlreadyIndexedError(already_indexed)

        expected: dict[str, tuple[str, PendingChunkManifest]] = {}
        expected_by_turn: dict[str, set[str]] = {}
        for manifest in by_turn.values():
            turn_ids = expected_by_turn.setdefault(manifest.turn_id, set())
            for row in manifest.chunks:
                if row.chunk_id in expected:
                    raise ValueError(
                        "pending manifests reuse one chunk identity"
                    )
                expected[row.chunk_id] = (manifest.turn_id, row)
                turn_ids.add(row.chunk_id)

        durable_by_turn = {turn_id: set() for turn_id in expected_by_turn}
        turn_ids = list(expected_by_turn)
        for start in range(0, len(turn_ids), 500):
            batch = turn_ids[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = self._db.execute(
                "SELECT turn_id, chunk_id FROM chunks "
                f"WHERE turn_id IN ({placeholders})",
                tuple(batch),
            ).fetchall()
            for turn_id, chunk_id in rows:
                durable_by_turn[str(turn_id)].add(str(chunk_id))
        if durable_by_turn != expected_by_turn:
            raise RuntimeError(
                "pending ingest durable topology differs from its manifest"
            )

        durable: dict[str, tuple] = {}
        chunk_ids = list(expected)
        for start in range(0, len(chunk_ids), 500):
            batch = chunk_ids[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = self._db.execute(
                "SELECT chunk_id, turn_id, text, start_char, end_char, token_count, "
                "embedding, hnsw_label, term_count FROM chunks "
                f"WHERE chunk_id IN ({placeholders})",
                tuple(batch),
            ).fetchall()
            durable.update((str(row[0]), row) for row in rows)

        for chunk_id, (turn_id, expected_row) in expected.items():
            row = durable.get(chunk_id)
            if row is None or (
                str(row[1]) != turn_id
                or int(row[3]) != expected_row.start_char
                or int(row[4]) != expected_row.end_char
                or int(row[5]) != expected_row.token_count
                or _sha256_text(str(row[2])) != expected_row.text_sha256
                or row[6] is None
                or row[7] is None
                or row[8] is None
            ):
                raise RuntimeError(
                    "pending ingest cannot complete before every chunk is indexed"
                )

        indexed_at = datetime.now(timezone.utc).isoformat()
        manifest_rows = [
            (manifest.turn_id, manifest.sha256) for manifest in by_turn.values()
        ]
        batch_size = max(1, (_SQLITE_PARAMETER_BUDGET - 1) // 2)
        for start in range(0, len(manifest_rows), batch_size):
            batch = manifest_rows[start : start + batch_size]
            placeholders = ",".join("(?, ?)" for _ in batch)
            parameters = (indexed_at,) + tuple(
                value for row in batch for value in row
            )
            updated = self._db.execute(
                "UPDATE pending_ingests SET status = 'indexed', "
                "indexed_at = COALESCE(indexed_at, ?) "
                "WHERE status = 'pending' "
                f"AND (turn_id, manifest_sha256) IN (VALUES {placeholders})",
                parameters,
            ).rowcount
            if updated != len(batch):
                raise RuntimeError(
                    "pending ingest completion lost its manifest receipt"
                )


__all__ = [
    "PendingChunkManifest",
    "PendingIngestAlreadyIndexedError",
    "PendingIngestManifest",
    "PendingIngestStore",
    "backfill_legacy_ingest_receipts",
]
