from __future__ import annotations

from memory_condense.db import Database
from memory_condense.schemas import Turn


class TranscriptStore:
    """Append-only store for conversation transcript turns."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def append(self, role: str, text: str) -> Turn:
        """Create and persist a new turn. Returns the Turn with generated ID."""
        turn = Turn(role=role, text=text)
        self._db.execute(
            "INSERT INTO turns (turn_id, role, text, created_at) VALUES (?, ?, ?, ?)",
            (turn.turn_id, turn.role, turn.text, turn.created_at.isoformat()),
        )
        self._db.commit()
        return turn

    def get_turn(self, turn_id: str) -> Turn | None:
        """Retrieve a single turn by ID."""
        cur = self._db.execute(
            "SELECT turn_id, role, text, created_at FROM turns WHERE turn_id = ?",
            (turn_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_turn(row)

    def get_recent(self, n: int = 20) -> list[Turn]:
        """Return the N most recent turns, ordered oldest-first."""
        cur = self._db.execute(
            "SELECT turn_id, role, text, created_at FROM turns "
            "ORDER BY created_at DESC LIMIT ?",
            (n,),
        )
        rows = cur.fetchall()
        return [self._row_to_turn(r) for r in reversed(rows)]

    def get_all(self) -> list[Turn]:
        """Return all turns, ordered by created_at."""
        cur = self._db.execute(
            "SELECT turn_id, role, text, created_at FROM turns ORDER BY created_at"
        )
        return [self._row_to_turn(r) for r in cur.fetchall()]

    def count(self) -> int:
        """Return total number of stored turns."""
        cur = self._db.execute("SELECT COUNT(*) FROM turns")
        return cur.fetchone()[0]

    @staticmethod
    def _row_to_turn(row: tuple) -> Turn:
        return Turn(
            turn_id=row[0],
            role=row[1],
            text=row[2],
            created_at=row[3],
        )
