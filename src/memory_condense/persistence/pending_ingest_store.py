"""Durable, provider-free journal for incomplete turn-to-index publication."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from memory_condense.domain.schemas import Chunk, Turn
from memory_condense.persistence.db import Database


_MANIFEST_FORMAT = "memory-condense-pending-ingest-v1"


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
        """Insert/adopt an exact manifest and return its durable status."""
        now = datetime.now(timezone.utc).isoformat()
        record = self._get_record(manifest.turn_id)
        if record is not None:
            if record[0] != manifest:
                raise ValueError(
                    "turn already has a different pending chunk manifest"
                )
            self._claim_reservations(manifest)
            durable = self._durable_manifest(manifest.turn_id)
            if durable is not None and durable[0] != manifest:
                raise ValueError(
                    "turn durable chunk topology no longer matches its receipt"
                )
            if durable is None and record[1] == "indexed" and manifest.chunks:
                raise ValueError(
                    "indexed ingest receipt has no durable chunk topology"
                )
            return record[1]

        durable = self._durable_manifest(manifest.turn_id)
        if durable is not None and durable[0] != manifest:
            raise ValueError("turn already has a different durable chunk topology")
        already_indexed = durable is not None and durable[1]
        status = (
            "indexed"
            if not manifest.chunks or already_indexed
            else "pending"
        )
        self._db.execute(
            "INSERT INTO pending_ingests "
            "(turn_id, manifest_sha256, manifest_json, status, created_at, indexed_at) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(turn_id) DO NOTHING",
            (
                manifest.turn_id,
                manifest.sha256,
                manifest.canonical_json,
                status,
                now,
                now if status == "indexed" else None,
            ),
        )
        record = self._get_record(manifest.turn_id)
        if record is None or record[0] != manifest:
            raise ValueError("turn already has a different pending chunk manifest")
        self._claim_reservations(manifest)
        return record[1]

    def get(self, turn_id: str) -> PendingIngestManifest | None:
        record = self._get_record(turn_id)
        return record[0] if record is not None else None

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
                "SELECT chunk_id, embedding, hnsw_label, term_count FROM chunks "
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
                    if state is None or any(value is None for value in state[1:]):
                        terminal_turns.add(turn_id)
        if terminal_turns:
            raise PendingIngestAlreadyIndexedError(tuple(terminal_turns))

    def _claim_reservations(self, manifest: PendingIngestManifest) -> None:
        """Insert or verify one manifest's globally unique chunk ownership."""
        self._db.executemany(
            "INSERT INTO ingest_chunk_reservations "
            "(chunk_id, turn_id, start_char, end_char, token_count, text_sha256) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(chunk_id) DO NOTHING",
            [
                (
                    row.chunk_id,
                    manifest.turn_id,
                    row.start_char,
                    row.end_char,
                    row.token_count,
                    row.text_sha256,
                )
                for row in manifest.chunks
            ],
        )

        expected = {row.chunk_id: row for row in manifest.chunks}
        actual: dict[str, tuple] = {}
        chunk_ids = list(expected)
        for start in range(0, len(chunk_ids), 500):
            batch = chunk_ids[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = self._db.execute(
                "SELECT chunk_id, turn_id, start_char, end_char, token_count, "
                "text_sha256 FROM ingest_chunk_reservations "
                f"WHERE chunk_id IN ({placeholders})",
                tuple(batch),
            ).fetchall()
            actual.update((str(row[0]), row) for row in rows)

        for chunk_id, expected_row in expected.items():
            row = actual.get(chunk_id)
            if row is None or str(row[1]) != manifest.turn_id:
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

        owned_ids = {
            str(row[0])
            for row in self._db.execute(
                "SELECT chunk_id FROM ingest_chunk_reservations WHERE turn_id = ?",
                (manifest.turn_id,),
            ).fetchall()
        }
        if owned_ids != set(expected):
            raise ValueError("turn has a different reserved chunk topology")

    def _get_record(
        self, turn_id: str
    ) -> tuple[PendingIngestManifest, str] | None:
        row = self._db.execute(
            "SELECT manifest_sha256, manifest_json, status FROM pending_ingests "
            "WHERE turn_id = ?",
            (turn_id,),
        ).fetchone()
        if row is None:
            return None
        manifest = PendingIngestManifest.from_json(str(row[1]))
        if manifest.turn_id != turn_id or manifest.sha256 != str(row[0]):
            raise ValueError("pending ingest manifest receipt is inconsistent")
        status = str(row[2])
        if status not in {"pending", "indexed"}:
            raise ValueError("pending ingest manifest has an invalid status")
        return manifest, status

    def _durable_manifest(
        self, turn_id: str
    ) -> tuple[PendingIngestManifest, bool] | None:
        rows = self._db.execute(
            "SELECT chunk_id, text, start_char, end_char, token_count, "
            "embedding, hnsw_label, term_count FROM chunks "
            "WHERE turn_id = ? ORDER BY start_char, end_char, chunk_id",
            (turn_id,),
        ).fetchall()
        if not rows:
            return None
        manifest = PendingIngestManifest(
            turn_id=turn_id,
            chunks=tuple(
                PendingChunkManifest(
                    chunk_id=str(row[0]),
                    start_char=int(row[2]),
                    end_char=int(row[3]),
                    token_count=int(row[4]),
                    text_sha256=_sha256_text(str(row[1])),
                )
                for row in rows
            ),
        )
        complete = all(
            row[5] is not None and row[6] is not None and row[7] is not None
            for row in rows
        )
        return manifest, complete

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

    def finalize(self, manifests: Sequence[PendingIngestManifest]) -> None:
        """Prove complete durable indexing, then seal receipts before commit."""
        by_turn: dict[str, PendingIngestManifest] = {}
        for manifest in manifests:
            previous = by_turn.setdefault(manifest.turn_id, manifest)
            if previous != manifest:
                raise ValueError("conflicting pending manifests in one finalizer")

        already_indexed: list[str] = []
        for manifest in by_turn.values():
            record = self._get_record(manifest.turn_id)
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
        for manifest in by_turn.values():
            updated = self._db.execute(
                "UPDATE pending_ingests SET status = 'indexed', "
                "indexed_at = COALESCE(indexed_at, ?) "
                "WHERE turn_id = ? AND manifest_sha256 = ?",
                (indexed_at, manifest.turn_id, manifest.sha256),
            ).rowcount
            if updated != 1:
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
