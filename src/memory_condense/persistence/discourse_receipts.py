"""Content-bound snapshot receipts for the persisted discourse graph."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol, Sequence

from memory_condense.domain.discourse import (
    DiscourseSnapshot,
    canonical_json,
)
from memory_condense.persistence.db import CURRENT_SCHEMA_VERSION


class DiscourseSnapshotError(ValueError):
    """A graph high-water receipt is corrupt, missing, or no longer current."""


class _Executor(Protocol):
    def execute(self, sql: str, params: tuple = ()) -> Any: ...


_SOURCE_ROW_STREAMS = (
    (
        "turns",
        "SELECT turn_id, role, text, source_id, created_at, ordinal "
        "FROM turns ORDER BY ordinal, turn_id",
    ),
    (
        "chunks",
        "SELECT chunk_id, turn_id, text, start_char, end_char, token_count "
        "FROM chunks ORDER BY chunk_id",
    ),
)

_GRAPH_ROW_STREAMS = (
    ("discourse_artifacts", "SELECT * FROM discourse_artifacts ORDER BY artifact_id"),
    ("episodes", "SELECT * FROM episodes ORDER BY episode_id"),
    (
        "episode_evidence",
        "SELECT * FROM episode_evidence ORDER BY episode_id, evidence_order",
    ),
    (
        "episode_representatives",
        "SELECT * FROM episode_representatives ORDER BY episode_id, rank",
    ),
    ("discourse_units", "SELECT * FROM discourse_units ORDER BY unit_id"),
    (
        "discourse_unit_evidence",
        "SELECT * FROM discourse_unit_evidence ORDER BY unit_id, evidence_order",
    ),
    (
        "discourse_relations",
        "SELECT * FROM discourse_relations ORDER BY relation_id",
    ),
    (
        "discourse_relation_members",
        "SELECT * FROM discourse_relation_members "
        "ORDER BY relation_id, member_order",
    ),
    (
        "discourse_relation_evidence",
        "SELECT * FROM discourse_relation_evidence "
        "ORDER BY relation_id, evidence_order",
    ),
    (
        "discourse_artifact_coverage",
        "SELECT * FROM discourse_artifact_coverage "
        "ORDER BY artifact_id, chunk_id, coverage_kind, chunk_identity_sha256",
    ),
    (
        "discourse_artifact_coverage_receipts",
        "SELECT * FROM discourse_artifact_coverage_receipts "
        "ORDER BY artifact_id, coverage_kind, source_revision",
    ),
)


def _row_stream_sha256(
    executor: _Executor,
    streams: Sequence[tuple[str, str]],
) -> str:
    """Hash a typed canonical row stream without materializing the corpus."""

    digest = hashlib.sha256()
    for table, query in streams:
        digest.update(canonical_json({"table": table}).encode("utf-8"))
        digest.update(b"\n")
        cursor = executor.execute(query)
        while True:
            rows = cursor.fetchmany(512)
            if not rows:
                break
            for row in rows:
                digest.update(
                    canonical_json({"row": list(row)}).encode("utf-8")
                )
                digest.update(b"\n")
    return digest.hexdigest()


def discourse_content_digests(executor: _Executor) -> tuple[str, str]:
    """Return authoritative source and immutable graph content roots."""

    return (
        _row_stream_sha256(executor, _SOURCE_ROW_STREAMS),
        _row_stream_sha256(executor, _GRAPH_ROW_STREAMS),
    )


class DiscourseReceiptMixin:
    """Mixin implementing immutable, content-bound graph snapshots."""

    _db: _Executor

    def _revision_state(self) -> tuple[int, int]:
        row = self._db.execute(
            "SELECT source_revision, graph_content_revision "
            "FROM discourse_revision_state WHERE singleton = 1"
        ).fetchone()
        if row is None:
            raise DiscourseSnapshotError("discourse revision state is missing")
        return int(row[0]), int(row[1])

    def _digest_cache_is_safe(self) -> bool:
        """Whether revision counters cannot still be restored by rollback."""

        connection = getattr(self._db, "connection", self._db)
        return getattr(connection, "in_transaction", None) is False

    def _current_high_water(
        self,
    ) -> tuple[int, int, tuple[str, ...], int, int, str, str]:
        max_turn = int(
            self._db.execute(
                "SELECT COALESCE(MAX(ordinal), 0) FROM turns"
            ).fetchone()[0]
        )
        chunk_count = int(
            self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        )
        artifact_ids = tuple(
            row[0]
            for row in self._db.execute(
                "SELECT artifact_id FROM discourse_artifacts ORDER BY artifact_id"
            ).fetchall()
        )
        source_revision, graph_content_revision = self._revision_state()
        digest_key = (source_revision, graph_content_revision)
        cache_is_safe = self._digest_cache_is_safe()
        cached = getattr(self, "_discourse_digest_cache", None)
        if cache_is_safe and cached is not None and cached[0] == digest_key:
            source_content, graph_content = cached[1]
        else:
            source_content, graph_content = discourse_content_digests(self._db)
            if cache_is_safe:
                self._discourse_digest_cache = (
                    digest_key,
                    (source_content, graph_content),
                )
        return (
            max_turn,
            chunk_count,
            artifact_ids,
            source_revision,
            graph_content_revision,
            source_content,
            graph_content,
        )

    def _snapshot_from_row(self, row: Sequence[Any]) -> DiscourseSnapshot:
        try:
            artifact_ids = json.loads(row[4])
        except (TypeError, json.JSONDecodeError) as exc:
            raise DiscourseSnapshotError(
                "snapshot artifact IDs are invalid JSON"
            ) from exc
        if (
            not isinstance(artifact_ids, list)
            or canonical_json(artifact_ids) != row[4]
            or artifact_ids != sorted(set(artifact_ids))
            or any(
                not isinstance(item, str) or not item.strip()
                for item in artifact_ids
            )
        ):
            raise DiscourseSnapshotError("snapshot artifact IDs are not canonical")
        try:
            return DiscourseSnapshot(
                max_turn_ordinal=int(row[1]),
                chunk_count=int(row[2]),
                graph_revision=int(row[0]),
                schema_version=int(row[3]),
                artifact_ids=tuple(artifact_ids),
                source_revision=int(row[6]),
                graph_content_revision=int(row[7]),
                source_content_sha256=row[8],
                graph_content_sha256=row[9],
                snapshot_sha256=row[5],
            )
        except ValueError as exc:
            raise DiscourseSnapshotError(
                "snapshot receipt identity is corrupt"
            ) from exc

    def _latest_stored_snapshot(self) -> DiscourseSnapshot | None:
        row = self._db.execute(
            "SELECT graph_revision, max_turn_ordinal, chunk_count, "
            "schema_version, artifact_ids, snapshot_sha256, source_revision, "
            "graph_content_revision, source_content_sha256, "
            "graph_content_sha256 FROM discourse_graph_revisions "
            "ORDER BY graph_revision DESC LIMIT 1"
        ).fetchone()
        return None if row is None else self._snapshot_from_row(row)

    def _live_snapshot(self) -> DiscourseSnapshot:
        stored = self._latest_stored_snapshot()
        (
            max_turn,
            chunk_count,
            artifacts,
            source_revision,
            graph_content_revision,
            source_content,
            graph_content,
        ) = self._current_high_water()
        return DiscourseSnapshot(
            max_turn_ordinal=max_turn,
            chunk_count=chunk_count,
            graph_revision=0 if stored is None else stored.graph_revision,
            schema_version=CURRENT_SCHEMA_VERSION,
            artifact_ids=artifacts,
            source_revision=source_revision,
            graph_content_revision=graph_content_revision,
            source_content_sha256=source_content,
            graph_content_sha256=graph_content,
        )

    def _append_snapshot(self) -> DiscourseSnapshot:
        revision = int(
            self._db.execute(
                "SELECT COALESCE(MAX(graph_revision), 0) + 1 "
                "FROM discourse_graph_revisions"
            ).fetchone()[0]
        )
        (
            max_turn,
            chunk_count,
            artifact_ids,
            source_revision,
            graph_content_revision,
            source_content,
            graph_content,
        ) = self._current_high_water()
        snapshot = DiscourseSnapshot(
            max_turn_ordinal=max_turn,
            chunk_count=chunk_count,
            graph_revision=revision,
            schema_version=CURRENT_SCHEMA_VERSION,
            artifact_ids=artifact_ids,
            source_revision=source_revision,
            graph_content_revision=graph_content_revision,
            source_content_sha256=source_content,
            graph_content_sha256=graph_content,
        )
        self._db.execute(
            "INSERT INTO discourse_graph_revisions "
            "(graph_revision, max_turn_ordinal, chunk_count, schema_version, "
            "artifact_ids, snapshot_sha256, source_revision, "
            "graph_content_revision, source_content_sha256, "
            "graph_content_sha256) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                snapshot.graph_revision,
                snapshot.max_turn_ordinal,
                snapshot.chunk_count,
                snapshot.schema_version,
                canonical_json(list(snapshot.artifact_ids)),
                snapshot.snapshot_sha256,
                snapshot.source_revision,
                snapshot.graph_content_revision,
                snapshot.source_content_sha256,
                snapshot.graph_content_sha256,
            ),
        )
        return snapshot

    def snapshot(self, graph_revision: int | None = None) -> DiscourseSnapshot:
        """Read and verify one immutable publication high-water receipt."""

        if graph_revision is None:
            return self._live_snapshot()
        row = self._db.execute(
            "SELECT graph_revision, max_turn_ordinal, chunk_count, "
            "schema_version, artifact_ids, snapshot_sha256, source_revision, "
            "graph_content_revision, source_content_sha256, "
            "graph_content_sha256 FROM discourse_graph_revisions "
            "WHERE graph_revision = ?",
            (int(graph_revision),),
        ).fetchone()
        if row is None:
            if graph_revision != 0:
                raise KeyError(f"unknown discourse graph revision: {graph_revision}")
            return self._live_snapshot()
        snapshot = self._snapshot_from_row(row)
        current = self._current_high_water()
        if (
            current[0] < snapshot.max_turn_ordinal
            or current[1] < snapshot.chunk_count
            or current[3] < snapshot.source_revision
            or current[4] < snapshot.graph_content_revision
        ):
            raise DiscourseSnapshotError("authoritative high-water moved backwards")
        if not set(snapshot.artifact_ids).issubset(current[2]):
            raise DiscourseSnapshotError("a snapshot artifact no longer exists")
        return snapshot

    def latest_snapshot(self) -> DiscourseSnapshot:
        return self.snapshot()

    def validate_snapshot(
        self,
        snapshot: DiscourseSnapshot,
        *,
        require_current: bool = True,
    ) -> bool:
        live = self.latest_snapshot()
        if snapshot == live:
            return True
        if require_current:
            raise DiscourseSnapshotError("snapshot is not current")
        if snapshot.graph_revision == 0:
            raise DiscourseSnapshotError("snapshot does not match current source state")
        stored = self.snapshot(snapshot.graph_revision)
        if stored != snapshot:
            raise DiscourseSnapshotError("snapshot does not match its stored receipt")
        return True


__all__ = [
    "DiscourseReceiptMixin",
    "DiscourseSnapshotError",
    "discourse_content_digests",
]
