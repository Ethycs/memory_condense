"""Source-grounded persistence for episodic discourse closure.

The repository stores graph structure and exact source coordinates, never a
generated evidence string or request-derived transformer state.  Every
evidence span is revalidated against both its chunk and authoritative turn on
write *and* read, so a graph row cannot silently outlive or misquote its
source.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Sequence, TypeVar

from memory_condense.domain.discourse import (
    ArtifactCoverageReceipt,
    DiscourseArtifact,
    DiscourseRelation,
    DiscourseSnapshot,
    DiscourseUnit,
    Episode,
    EpisodeRepresentative,
    EvidenceSpan,
    RelationMember,
    canonical_json,
    evidence_span_sort_key,
    identity_sha256,
)
from memory_condense.persistence.db import Database
from memory_condense.persistence.discourse_evidence import (
    DiscourseIdentityError,
    EvidenceStoreMixin,
    SourceEvidenceError,
    _safe_metadata,
    _strict_json_object,
    _unique,
)
from memory_condense.persistence.discourse_receipts import (
    DiscourseReceiptMixin,
    DiscourseSnapshotError,
)


_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class ArtifactCoverageMark:
    chunk_id: str
    coverage_kind: str
    status: str

    def __post_init__(self) -> None:
        if not str(self.chunk_id).strip():
            raise ValueError("coverage chunk_id must be non-empty")
        if self.coverage_kind not in {"episode", "discourse"}:
            raise ValueError("coverage_kind must be episode or discourse")
        if self.status not in {"annotated", "no_output"}:
            raise ValueError("coverage status must be annotated or no_output")


class DiscourseStore(EvidenceStoreMixin, DiscourseReceiptMixin):
    """SQLite repository for immutable episodic and discourse graph records."""

    def __init__(self, db: Database) -> None:
        self._db = db

    @contextmanager
    def _publication(self) -> Iterator[None]:
        """One atomic unit even when composed inside an outer transaction."""

        self._db.execute("SAVEPOINT discourse_graph_publication")
        try:
            yield
        except BaseException:
            self._db.execute("ROLLBACK TO discourse_graph_publication")
            self._db.execute("RELEASE discourse_graph_publication")
            raise
        else:
            self._db.execute("RELEASE discourse_graph_publication")

    @staticmethod
    def _dedupe_by_id(
        values: Sequence[_T], id_name: str, *, label: str
    ) -> tuple[_T, ...]:
        unique: dict[str, _T] = {}
        for value in values:
            identity = str(getattr(value, id_name))
            prior = unique.get(identity)
            if prior is not None and prior != value:
                raise DiscourseIdentityError(
                    f"batch reuses {label} ID {identity!r} with different contents"
                )
            unique[identity] = value
        return tuple(unique.values())

    def publish(
        self,
        artifact: DiscourseArtifact,
        *,
        episodes: Sequence[Episode] = (),
        representatives: Sequence[EpisodeRepresentative] = (),
        units: Sequence[DiscourseUnit] = (),
        relations: Sequence[DiscourseRelation] = (),
        coverage: Sequence[ArtifactCoverageMark] = (),
    ) -> DiscourseSnapshot:
        """Atomically publish one deterministic graph batch and its receipt.

        Replaying an identical batch is a no-op and returns the existing latest
        high-water receipt.  Reusing any stable ID for different contents rolls
        back the whole batch and raises :class:`DiscourseIdentityError`.
        """

        episodes = self._dedupe_by_id(episodes, "episode_id", label="episode")
        representatives = tuple(
            dict.fromkeys(representatives)
        )
        units = self._dedupe_by_id(units, "unit_id", label="unit")
        relations = self._dedupe_by_id(relations, "relation_id", label="relation")
        coverage_by_coordinate: dict[tuple[str, str], ArtifactCoverageMark] = {}
        for mark in coverage:
            if not isinstance(mark, ArtifactCoverageMark):
                raise TypeError("coverage must contain ArtifactCoverageMark values")
            coordinate = (mark.chunk_id, mark.coverage_kind)
            prior = coverage_by_coordinate.get(coordinate)
            if prior is not None and prior != mark:
                raise DiscourseIdentityError(
                    f"coverage coordinate {coordinate!r} has conflicting statuses"
                )
            coverage_by_coordinate[coordinate] = mark
        if any(item.artifact_id != artifact.artifact_id for item in episodes):
            raise ValueError("every episode in a publication must use its artifact")
        if any(item.artifact_id != artifact.artifact_id for item in units):
            raise ValueError("every unit in a publication must use its artifact")
        if any(item.artifact_id != artifact.artifact_id for item in relations):
            raise ValueError("every relation in a publication must use its artifact")

        changed = False
        with self._publication():
            changed |= self._insert_artifact(artifact)
            for episode in sorted(
                episodes,
                key=lambda item: (item.source_id, item.sequence_no, item.episode_id),
            ):
                changed |= self._insert_episode(episode)
            for source_id in sorted({item.source_id for item in episodes}):
                self._validate_source_episode_order(artifact.artifact_id, source_id)
            for representative in sorted(
                representatives, key=lambda item: (item.episode_id, item.rank)
            ):
                changed |= self._insert_representative(
                    representative, required_artifact=artifact.artifact_id
                )
            for unit in sorted(units, key=lambda item: item.unit_id):
                changed |= self._insert_unit(unit)
            for relation in sorted(relations, key=lambda item: item.relation_id):
                changed |= self._insert_relation(relation)
            for mark in sorted(
                coverage_by_coordinate.values(),
                key=lambda item: (item.coverage_kind, item.chunk_id),
            ):
                changed |= self._insert_coverage(artifact.artifact_id, mark)
            if changed or self._latest_stored_snapshot() != self._live_snapshot():
                snapshot = self._append_snapshot()
            else:
                snapshot = self.latest_snapshot()
        return snapshot

    def put_artifact(self, artifact: DiscourseArtifact) -> DiscourseSnapshot:
        return self.publish(artifact)

    def register_artifact(self, artifact: DiscourseArtifact) -> DiscourseSnapshot:
        """Compatibility spelling for stores that register artifact identities."""

        return self.put_artifact(artifact)

    def publish_batch(
        self,
        artifact: DiscourseArtifact,
        *,
        episodes: Sequence[Episode] = (),
        representatives: Sequence[EpisodeRepresentative] = (),
        units: Sequence[DiscourseUnit] = (),
        relations: Sequence[DiscourseRelation] = (),
        coverage: Sequence[ArtifactCoverageMark] = (),
    ) -> DiscourseSnapshot:
        """Explicit batch spelling of :meth:`publish`."""

        return self.publish(
            artifact,
            episodes=episodes,
            representatives=representatives,
            units=units,
            relations=relations,
            coverage=coverage,
        )

    def put_episodes(
        self,
        artifact: DiscourseArtifact,
        episodes: Sequence[Episode],
        *,
        representatives: Sequence[EpisodeRepresentative] = (),
    ) -> DiscourseSnapshot:
        return self.publish(
            artifact, episodes=episodes, representatives=representatives
        )

    def put_units(
        self,
        artifact: DiscourseArtifact,
        units: Sequence[DiscourseUnit],
    ) -> DiscourseSnapshot:
        return self.publish(artifact, units=units)

    def put_relations(
        self,
        artifact: DiscourseArtifact,
        relations: Sequence[DiscourseRelation],
    ) -> DiscourseSnapshot:
        return self.publish(artifact, relations=relations)

    def _insert_artifact(self, artifact: DiscourseArtifact) -> bool:
        metadata = _safe_metadata(
            artifact.metadata,
            label="artifact metadata",
            owner="artifact",
        )
        stored = self.get_artifact(artifact.artifact_id)
        if stored is not None:
            if stored != artifact:
                raise DiscourseIdentityError(
                    f"artifact_id {artifact.artifact_id!r} already has another identity"
                )
            return False
        self._db.execute(
            "INSERT INTO discourse_artifacts "
            "(artifact_id, kind, implementation_sha256, policy_sha256, model_id, "
            "model_revision, checkpoint_sha256, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                artifact.artifact_id,
                artifact.kind,
                artifact.implementation_sha256,
                artifact.policy_sha256,
                artifact.model_id,
                artifact.model_revision,
                artifact.checkpoint_sha256,
                metadata,
            ),
        )
        return True

    def _insert_episode(self, episode: Episode) -> bool:
        stored = self.get_episode(episode.episode_id)
        if stored is not None:
            if stored != episode:
                raise DiscourseIdentityError(
                    f"episode_id {episode.episode_id!r} already has another identity"
                )
            return False
        coordinate = self._db.execute(
            "SELECT episode_id FROM episodes WHERE artifact_id = ? AND source_id = ? "
            "AND sequence_no = ?",
            (episode.artifact_id, episode.source_id, episode.sequence_no),
        ).fetchone()
        receipt_owner = self._db.execute(
            "SELECT episode_id FROM episodes WHERE receipt_sha256 = ?",
            (episode.receipt_sha256,),
        ).fetchone()
        if coordinate is not None or receipt_owner is not None:
            owner = coordinate[0] if coordinate is not None else receipt_owner[0]
            raise DiscourseIdentityError(
                f"episode identity coordinate is already owned by {owner!r}"
            )
        for span in episode.evidence:
            self._validate_span(span, required_source=episode.source_id)
        self._db.execute(
            "INSERT INTO episodes "
            "(episode_id, artifact_id, source_id, sequence_no, first_ordinal, "
            "last_ordinal, boundary_method, initial_boundary, refined_boundary, "
            "boundary_score, boundary_threshold, receipt_sha256) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                episode.episode_id,
                episode.artifact_id,
                episode.source_id,
                episode.sequence_no,
                episode.first_ordinal,
                episode.last_ordinal,
                episode.boundary_method,
                episode.initial_boundary,
                episode.refined_boundary,
                episode.boundary_score,
                episode.boundary_threshold,
                episode.receipt_sha256,
            ),
        )
        self._insert_evidence(
            "episode_evidence",
            "episode_id",
            episode.episode_id,
            episode.evidence,
            required_source=episode.source_id,
        )
        return True

    def _validate_source_episode_order(self, artifact_id: str, source_id: str) -> None:
        rows = self._db.execute(
            "SELECT episode_id, sequence_no "
            "FROM episodes WHERE artifact_id = ? AND source_id = ? "
            "ORDER BY sequence_no, episode_id",
            (artifact_id, source_id),
        ).fetchall()
        for left, right in zip(rows, rows[1:]):
            left_episode = self.get_episode(str(left[0]))
            right_episode = self.get_episode(str(right[0]))
            if left_episode is None or right_episode is None:
                raise DiscourseIdentityError("episode order references a missing row")
            left_span = left_episode.evidence[-1]
            right_span = right_episode.evidence[0]
            overlaps_same_turn = (
                left_span.ordinal == right_span.ordinal
                and (left_span.source_id or source_id)
                == (right_span.source_id or source_id)
                and left_span.turn_start_char + left_span.end_char
                > right_span.turn_start_char + right_span.start_char
            )
            if (
                int(right[1]) != int(left[1]) + 1
                or evidence_span_sort_key(left_span)
                >= evidence_span_sort_key(right_span)
                or overlaps_same_turn
            ):
                raise ValueError(
                    f"episodes {left[0]!r} and {right[0]!r} overlap or violate "
                    "source-local sequence order"
                )

    def _insert_representative(
        self,
        representative: EpisodeRepresentative,
        *,
        required_artifact: str,
    ) -> bool:
        row = self._db.execute(
            "SELECT r.chunk_id, r.vector_identity_sha256, e.artifact_id "
            "FROM episode_representatives AS r "
            "JOIN episodes AS e ON e.episode_id = r.episode_id "
            "WHERE r.episode_id = ? AND r.rank = ?",
            (representative.episode_id, representative.rank),
        ).fetchone()
        if row is not None:
            stored = EpisodeRepresentative(
                episode_id=representative.episode_id,
                chunk_id=row[0],
                rank=representative.rank,
                vector_identity_sha256=row[1],
            )
            if stored != representative or row[2] != required_artifact:
                raise DiscourseIdentityError(
                    "episode representative rank already has another identity"
                )
            self._source_row(stored.chunk_id)
            return False
        episode_row = self._db.execute(
            "SELECT artifact_id FROM episodes WHERE episode_id = ?",
            (representative.episode_id,),
        ).fetchone()
        if episode_row is None:
            raise KeyError(f"unknown episode: {representative.episode_id}")
        if episode_row[0] != required_artifact:
            raise ValueError("representative episode belongs to another artifact")
        occupied_chunk = self._db.execute(
            "SELECT rank, vector_identity_sha256 FROM episode_representatives "
            "WHERE episode_id = ? AND chunk_id = ?",
            (representative.episode_id, representative.chunk_id),
        ).fetchone()
        if occupied_chunk is not None:
            raise DiscourseIdentityError(
                "episode representative chunk already has another rank or identity"
            )
        if self._db.execute(
            "SELECT 1 FROM episode_evidence WHERE episode_id = ? AND chunk_id = ?",
            (representative.episode_id, representative.chunk_id),
        ).fetchone() is None:
            raise ValueError("an episode representative must cite an episode chunk")
        self._source_row(representative.chunk_id)
        self._db.execute(
            "INSERT INTO episode_representatives "
            "(episode_id, chunk_id, rank, vector_identity_sha256) VALUES (?, ?, ?, ?)",
            (
                representative.episode_id,
                representative.chunk_id,
                representative.rank,
                representative.vector_identity_sha256,
            ),
        )
        return True

    def _insert_unit(self, unit: DiscourseUnit) -> bool:
        metadata = _safe_metadata(
            unit.metadata,
            label="unit metadata",
            owner="unit",
        )
        stored = self.get_unit(unit.unit_id)
        if stored is not None:
            if stored != unit:
                raise DiscourseIdentityError(
                    f"unit_id {unit.unit_id!r} already has another identity"
                )
            return False
        for span in unit.evidence:
            self._validate_span(span)
        self._db.execute(
            "INSERT INTO discourse_units "
            "(unit_id, artifact_id, kind, canonical_key, asserted_ordinal, "
            "confidence, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                unit.unit_id,
                unit.artifact_id,
                unit.kind,
                unit.canonical_key,
                unit.asserted_ordinal,
                unit.confidence,
                metadata,
            ),
        )
        self._insert_evidence(
            "discourse_unit_evidence",
            "unit_id",
            unit.unit_id,
            unit.evidence,
        )
        return True

    def _insert_relation(self, relation: DiscourseRelation) -> bool:
        metadata = _safe_metadata(
            relation.metadata,
            label="relation metadata",
            owner="relation",
        )
        stored = self.get_relation(relation.relation_id)
        if stored is not None:
            if stored != relation:
                raise DiscourseIdentityError(
                    f"relation_id {relation.relation_id!r} already has another identity"
                )
            return False
        unit_artifacts: set[str] = set()
        for member in relation.members:
            row = self._db.execute(
                "SELECT artifact_id FROM discourse_units WHERE unit_id = ?",
                (member.unit_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown discourse unit: {member.unit_id}")
            unit_artifacts.add(str(row[0]))
        if unit_artifacts != {relation.artifact_id}:
            raise ValueError("a relation cannot connect units from another artifact")
        for span in relation.evidence:
            self._validate_span(span)
        self._db.execute(
            "INSERT INTO discourse_relations "
            "(relation_id, artifact_id, relation_type, confidence, created_ordinal, "
            "metadata) VALUES (?, ?, ?, ?, ?, ?)",
            (
                relation.relation_id,
                relation.artifact_id,
                relation.relation_type,
                relation.confidence,
                relation.created_ordinal,
                metadata,
            ),
        )
        self._db.executemany(
            "INSERT INTO discourse_relation_members "
            "(relation_id, member_order, unit_id, role, ordinal, weight) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    relation.relation_id,
                    position,
                    member.unit_id,
                    member.role,
                    member.ordinal,
                    member.weight,
                )
                for position, member in enumerate(relation.members)
            ],
        )
        self._insert_evidence(
            "discourse_relation_evidence",
            "relation_id",
            relation.relation_id,
            relation.evidence,
        )
        return True

    def _insert_coverage(
        self,
        artifact_id: str,
        mark: ArtifactCoverageMark,
    ) -> bool:
        source = self._source_row(mark.chunk_id)
        if mark.coverage_kind == "episode":
            has_output = self._db.execute(
                "SELECT 1 FROM episode_evidence AS ee "
                "JOIN episodes AS e ON e.episode_id = ee.episode_id "
                "WHERE e.artifact_id = ? AND ee.chunk_id = ? LIMIT 1",
                (artifact_id, mark.chunk_id),
            ).fetchone() is not None
        else:
            has_output = self._db.execute(
                "SELECT 1 FROM discourse_unit_evidence AS ue "
                "JOIN discourse_units AS u ON u.unit_id = ue.unit_id "
                "WHERE u.artifact_id = ? AND ue.chunk_id = ? "
                "UNION ALL SELECT 1 FROM discourse_relation_evidence AS re "
                "JOIN discourse_relations AS r ON r.relation_id = re.relation_id "
                "WHERE r.artifact_id = ? AND re.chunk_id = ? LIMIT 1",
                (artifact_id, mark.chunk_id, artifact_id, mark.chunk_id),
            ).fetchone() is not None
        expected_status = "annotated" if has_output else "no_output"
        if mark.status != expected_status:
            raise ValueError(
                "coverage status contradicts the artifact's persisted outputs"
            )
        source_revision = self._revision_state()[0]
        body = {
            "artifact_id": artifact_id,
            "chunk_id": mark.chunk_id,
            "coverage_kind": mark.coverage_kind,
            "source_revision": source_revision,
            "chunk_identity_sha256": source.identity_sha256,
            "status": mark.status,
        }
        receipt = identity_sha256(body)
        row = self._db.execute(
            "SELECT source_revision, status, receipt_sha256 "
            "FROM discourse_artifact_coverage WHERE artifact_id = ? "
            "AND chunk_id = ? AND coverage_kind = ? "
            "AND chunk_identity_sha256 = ?",
            (
                artifact_id,
                mark.chunk_id,
                mark.coverage_kind,
                source.identity_sha256,
            ),
        ).fetchone()
        if row is not None:
            if row[1] != mark.status:
                raise DiscourseIdentityError(
                    "coverage status changed under one artifact and source identity"
                )
            stored_body = dict(body)
            stored_body["source_revision"] = int(row[0])
            if identity_sha256(stored_body) != row[2]:
                raise DiscourseIdentityError("stored coverage receipt is corrupt")
            return False
        self._db.execute(
            "INSERT INTO discourse_artifact_coverage "
            "(artifact_id, chunk_id, coverage_kind, source_revision, "
            "chunk_identity_sha256, status, receipt_sha256) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                artifact_id,
                mark.chunk_id,
                mark.coverage_kind,
                source_revision,
                source.identity_sha256,
                mark.status,
                receipt,
            ),
        )
        return True

    def get_artifact(self, artifact_id: str) -> DiscourseArtifact | None:
        row = self._db.execute(
            "SELECT artifact_id, kind, implementation_sha256, policy_sha256, "
            "model_id, model_revision, checkpoint_sha256, metadata "
            "FROM discourse_artifacts WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        if row is None:
            return None
        metadata = _strict_json_object(row[7], label="artifact metadata")
        _safe_metadata(metadata, label="artifact metadata", owner="artifact")
        return DiscourseArtifact(
            artifact_id=row[0],
            kind=row[1],
            implementation_sha256=row[2],
            policy_sha256=row[3],
            model_id=row[4],
            model_revision=row[5],
            checkpoint_sha256=row[6],
            metadata=metadata,
        )

    def get_episode(self, episode_id: str) -> Episode | None:
        row = self._db.execute(
            "SELECT episode_id, artifact_id, source_id, sequence_no, first_ordinal, "
            "last_ordinal, boundary_method, initial_boundary, refined_boundary, "
            "boundary_score, boundary_threshold, receipt_sha256 "
            "FROM episodes WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        if row is None:
            return None
        evidence = self._read_evidence(
            "episode_evidence",
            "episode_id",
            episode_id,
            required_source=row[2],
        )
        return Episode(
            episode_id=row[0],
            artifact_id=row[1],
            source_id=row[2],
            sequence_no=int(row[3]),
            first_ordinal=int(row[4]),
            last_ordinal=int(row[5]),
            evidence=evidence,
            boundary_method=row[6],
            initial_boundary=None if row[7] is None else int(row[7]),
            refined_boundary=None if row[8] is None else int(row[8]),
            boundary_score=None if row[9] is None else float(row[9]),
            boundary_threshold=None if row[10] is None else float(row[10]),
            receipt_sha256=row[11],
        )

    def get_representatives(
        self, episode_id: str
    ) -> tuple[EpisodeRepresentative, ...]:
        rows = self._db.execute(
            "SELECT chunk_id, rank, vector_identity_sha256 "
            "FROM episode_representatives WHERE episode_id = ? ORDER BY rank",
            (episode_id,),
        ).fetchall()
        values = tuple(
            EpisodeRepresentative(
                episode_id=episode_id,
                chunk_id=row[0],
                rank=int(row[1]),
                vector_identity_sha256=row[2],
            )
            for row in rows
        )
        evidence_chunks = {
            row[0]
            for row in self._db.execute(
                "SELECT chunk_id FROM episode_evidence WHERE episode_id = ?",
                (episode_id,),
            ).fetchall()
        }
        for value in values:
            self._source_row(value.chunk_id)
            if value.chunk_id not in evidence_chunks:
                raise DiscourseIdentityError(
                    "stored episode representative is outside its episode"
                )
        return values

    def get_unit(self, unit_id: str) -> DiscourseUnit | None:
        row = self._db.execute(
            "SELECT unit_id, artifact_id, kind, canonical_key, asserted_ordinal, "
            "confidence, metadata FROM discourse_units WHERE unit_id = ?",
            (unit_id,),
        ).fetchone()
        if row is None:
            return None
        metadata = _strict_json_object(row[6], label="unit metadata")
        _safe_metadata(metadata, label="unit metadata", owner="unit")
        return DiscourseUnit(
            unit_id=row[0],
            artifact_id=row[1],
            kind=row[2],
            canonical_key=row[3],
            asserted_ordinal=int(row[4]),
            confidence=float(row[5]),
            evidence=self._read_evidence(
                "discourse_unit_evidence", "unit_id", unit_id
            ),
            metadata=metadata,
        )

    def get_relation(self, relation_id: str) -> DiscourseRelation | None:
        row = self._db.execute(
            "SELECT relation_id, artifact_id, relation_type, confidence, "
            "created_ordinal, metadata FROM discourse_relations WHERE relation_id = ?",
            (relation_id,),
        ).fetchone()
        if row is None:
            return None
        member_rows = self._db.execute(
            "SELECT member_order, unit_id, role, ordinal, weight "
            "FROM discourse_relation_members WHERE relation_id = ? "
            "ORDER BY member_order",
            (relation_id,),
        ).fetchall()
        if [int(item[0]) for item in member_rows] != list(range(len(member_rows))):
            raise DiscourseIdentityError(
                f"stored member order for {relation_id!r} is not contiguous"
            )
        members = tuple(
            RelationMember(
                unit_id=item[1],
                role=item[2],
                ordinal=int(item[3]),
                weight=float(item[4]),
            )
            for item in member_rows
        )
        for member in members:
            unit_row = self._db.execute(
                "SELECT artifact_id FROM discourse_units WHERE unit_id = ?",
                (member.unit_id,),
            ).fetchone()
            if unit_row is None or unit_row[0] != row[1]:
                raise DiscourseIdentityError(
                    f"relation {relation_id!r} has a cross-artifact member"
                )
        metadata = _strict_json_object(row[5], label="relation metadata")
        _safe_metadata(metadata, label="relation metadata", owner="relation")
        return DiscourseRelation(
            relation_id=row[0],
            artifact_id=row[1],
            relation_type=row[2],
            members=members,
            evidence=self._read_evidence(
                "discourse_relation_evidence", "relation_id", relation_id
            ),
            confidence=float(row[3]),
            created_ordinal=int(row[4]),
            metadata=metadata,
        )

    def coverage_for_chunks(
        self,
        artifact_id: str,
        chunk_ids: Sequence[str],
        *,
        coverage_kind: str = "discourse",
    ) -> dict[str, str]:
        """Return fresh per-chunk annotation statuses in first-input order."""

        if coverage_kind not in {"episode", "discourse"}:
            raise ValueError("coverage_kind must be episode or discourse")
        result: dict[str, str] = {}
        for chunk_id in _unique(str(item) for item in chunk_ids):
            source = self._source_row(chunk_id)
            row = self._db.execute(
                "SELECT source_revision, status, receipt_sha256 "
                "FROM discourse_artifact_coverage WHERE artifact_id = ? "
                "AND chunk_id = ? AND coverage_kind = ? "
                "AND chunk_identity_sha256 = ?",
                (
                    artifact_id,
                    chunk_id,
                    coverage_kind,
                    source.identity_sha256,
                ),
            ).fetchone()
            if row is None:
                continue
            body = {
                "artifact_id": artifact_id,
                "chunk_id": chunk_id,
                "coverage_kind": coverage_kind,
                "source_revision": int(row[0]),
                "chunk_identity_sha256": source.identity_sha256,
                "status": row[1],
            }
            if identity_sha256(body) != row[2]:
                raise DiscourseIdentityError("stored coverage receipt is corrupt")
            result[chunk_id] = str(row[1])
        return result

    def finalize_artifact_coverage(
        self,
        artifact_id: str,
        *,
        coverage_kind: str = "discourse",
    ) -> ArtifactCoverageReceipt:
        """Publish an O(1)-read receipt proving full current-corpus coverage."""

        if coverage_kind not in {"episode", "discourse"}:
            raise ValueError("coverage_kind must be episode or discourse")
        if self.get_artifact(artifact_id) is None:
            raise KeyError(f"unknown discourse artifact: {artifact_id}")
        with self._publication():
            # Acquire SQLite's writer reservation before reading.  The source
            # revision, row identities, stage marks, and receipt insertion are
            # consequently one transaction snapshot across connections.
            self._db.execute(
                "UPDATE meta SET value = value WHERE key = 'schema_version'"
            )
            source_revision = self._revision_state()[0]
            chunk_ids = tuple(
                str(row[0])
                for row in self._db.execute(
                    "SELECT chunk_id FROM chunks ORDER BY chunk_id"
                ).fetchall()
            )
            coverage_rows = self.verified_artifact_coverage_rows(
                artifact_id,
                chunk_ids,
                coverage_kind,
            )
            receipt = ArtifactCoverageReceipt(
                artifact_id=artifact_id,
                coverage_kind=coverage_kind,
                source_revision=source_revision,
                chunk_count=len(chunk_ids),
                coverage_sha256=identity_sha256(coverage_rows),
                turn_coverage_sha256=(
                    self.authoritative_turn_coverage_sha256()
                ),
            )
            if self._revision_state()[0] != source_revision:
                raise DiscourseSnapshotError(
                    "source changed while finalizing artifact coverage"
                )
            if coverage_rows != self.verified_artifact_coverage_rows(
                artifact_id,
                chunk_ids,
                coverage_kind,
            ):
                raise DiscourseSnapshotError(
                    "coverage changed while finalizing artifact coverage"
                )
            existing = self.artifact_coverage(
                artifact_id,
                coverage_kind=coverage_kind,
            )
            if existing is not None:
                if existing != receipt:
                    raise DiscourseIdentityError(
                        "current artifact coverage already has another receipt"
                    )
                return existing
            self._db.execute(
                "INSERT INTO discourse_artifact_coverage_receipts "
                "(artifact_id, coverage_kind, source_revision, chunk_count, "
                "coverage_sha256, turn_coverage_sha256, receipt_sha256) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    receipt.artifact_id,
                    receipt.coverage_kind,
                    receipt.source_revision,
                    receipt.chunk_count,
                    receipt.coverage_sha256,
                    receipt.turn_coverage_sha256,
                    receipt.receipt_sha256,
                ),
            )
            self._append_snapshot()
        return receipt

    def artifact_coverage(
        self,
        artifact_id: str,
        coverage_kind: str = "discourse",
    ) -> ArtifactCoverageReceipt | None:
        """Return exhaustive coverage only at the current source revision."""

        if coverage_kind not in {"episode", "discourse"}:
            raise ValueError("coverage_kind must be episode or discourse")
        source_revision = self._revision_state()[0]
        row = self._db.execute(
            "SELECT chunk_count, coverage_sha256, turn_coverage_sha256, "
            "receipt_sha256 "
            "FROM discourse_artifact_coverage_receipts WHERE artifact_id = ? "
            "AND coverage_kind = ? AND source_revision = ?",
            (artifact_id, coverage_kind, source_revision),
        ).fetchone()
        if row is None:
            return None
        receipt = ArtifactCoverageReceipt(
            artifact_id=artifact_id,
            coverage_kind=coverage_kind,
            source_revision=source_revision,
            chunk_count=int(row[0]),
            coverage_sha256=str(row[1]),
            turn_coverage_sha256=str(row[2]),
            receipt_sha256=str(row[3]),
        )
        current_count = int(
            self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        )
        if receipt.chunk_count != current_count:
            raise DiscourseIdentityError("coverage receipt chunk count is stale")
        return receipt

    def units_for_artifact(
        self,
        artifact_id: str,
        *,
        limit: int | None = None,
    ) -> tuple[DiscourseUnit, ...]:
        """Return a deterministically bounded artifact-wide unit scan."""

        limit = self._bounded_limit(limit)
        sql = (
            "SELECT unit_id FROM discourse_units WHERE artifact_id = ? "
            "ORDER BY asserted_ordinal DESC, unit_id"
        )
        params: list[Any] = [artifact_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        values = tuple(
            self.get_unit(str(row[0]))
            for row in self._db.execute(sql, tuple(params)).fetchall()
        )
        return tuple(item for item in values if item is not None)

    @staticmethod
    def _bounded_limit(limit: int | None) -> int | None:
        if limit is None:
            return None
        if limit < 0:
            raise ValueError("limit must be non-negative")
        return int(limit)

    def episodes_for_source(
        self,
        artifact_id: str,
        source_id: str,
        *,
        start_sequence: int | None = None,
        end_sequence: int | None = None,
        limit: int | None = None,
    ) -> tuple[Episode, ...]:
        limit = self._bounded_limit(limit)
        self._validate_source_episode_order(artifact_id, source_id)
        clauses = ["artifact_id = ?", "source_id = ?"]
        params: list[Any] = [artifact_id, source_id]
        if start_sequence is not None:
            clauses.append("sequence_no >= ?")
            params.append(int(start_sequence))
        if end_sequence is not None:
            clauses.append("sequence_no <= ?")
            params.append(int(end_sequence))
        sql = (
            "SELECT episode_id FROM episodes WHERE "
            + " AND ".join(clauses)
            + " ORDER BY sequence_no, episode_id"
        )
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        ids = [row[0] for row in self._db.execute(sql, tuple(params)).fetchall()]
        episodes = tuple(self.get_episode(item) for item in ids)
        return tuple(item for item in episodes if item is not None)

    def adjacent_episodes(
        self,
        episode_id: str,
        *,
        radius: int = 1,
        include_self: bool = False,
    ) -> tuple[Episode, ...]:
        if radius < 0:
            raise ValueError("radius must be non-negative")
        seed = self.get_episode(episode_id)
        if seed is None:
            raise KeyError(f"unknown episode: {episode_id}")
        if radius == 0:
            return (seed,) if include_self else ()
        prior = self._db.execute(
            "SELECT episode_id FROM episodes WHERE artifact_id = ? AND source_id = ? "
            "AND sequence_no < ? ORDER BY sequence_no DESC, episode_id DESC LIMIT ?",
            (seed.artifact_id, seed.source_id, seed.sequence_no, radius),
        ).fetchall()
        following = self._db.execute(
            "SELECT episode_id FROM episodes WHERE artifact_id = ? AND source_id = ? "
            "AND sequence_no > ? ORDER BY sequence_no, episode_id LIMIT ?",
            (seed.artifact_id, seed.source_id, seed.sequence_no, radius),
        ).fetchall()
        ordered_ids = [row[0] for row in reversed(prior)]
        if include_self:
            ordered_ids.append(seed.episode_id)
        ordered_ids.extend(row[0] for row in following)
        values = tuple(self.get_episode(item) for item in ordered_ids)
        return tuple(item for item in values if item is not None)

    def episode_ids_for_chunks(
        self,
        chunk_ids: Sequence[str],
        *,
        artifact_id: str | None = None,
    ) -> dict[str, str]:
        """Map each matched input chunk to its first deterministic episode.

        Mapping insertion order follows first occurrence in ``chunk_ids``;
        missing chunks are omitted.  Supplying an artifact removes ambiguity
        when multiple immutable graph interpretations cover the same chunk.
        """

        selected = _unique(str(item) for item in chunk_ids)
        found: dict[str, str] = {}
        for offset in range(0, len(selected), 400):
            batch = selected[offset : offset + 400]
            if not batch:
                continue
            placeholders = ",".join("?" for _ in batch)
            params: list[Any] = list(batch)
            where = f"ee.chunk_id IN ({placeholders})"
            if artifact_id is not None:
                where += " AND e.artifact_id = ?"
                params.append(artifact_id)
            rows = self._db.execute(
                "SELECT ee.chunk_id, e.episode_id FROM episode_evidence AS ee "
                "JOIN episodes AS e ON e.episode_id = ee.episode_id WHERE "
                + where
                + " ORDER BY ee.chunk_id, e.artifact_id, e.source_id, "
                "e.sequence_no, e.episode_id",
                tuple(params),
            ).fetchall()
            for chunk_id, found_episode_id in rows:
                found.setdefault(chunk_id, found_episode_id)
        result = {
            chunk_id: found[chunk_id] for chunk_id in selected if chunk_id in found
        }
        for chunk_id, episode_id in result.items():
            episode = self.get_episode(episode_id)
            if episode is None or chunk_id not in {
                span.chunk_id for span in episode.evidence
            }:
                raise DiscourseIdentityError(
                    f"chunk mapping for {chunk_id!r} is not source-grounded"
                )
        return result

    def episodes_for_chunks(
        self,
        chunk_ids: Sequence[str],
        *,
        artifact_id: str | None = None,
    ) -> dict[str, Episode]:
        ids = self.episode_ids_for_chunks(chunk_ids, artifact_id=artifact_id)
        hydrated: dict[str, Episode] = {}
        for chunk_id, episode_id in ids.items():
            value = self.get_episode(episode_id)
            if value is not None:
                hydrated[chunk_id] = value
        return hydrated

    def units_for_chunks(
        self,
        chunk_ids: Sequence[str],
        *,
        artifact_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[DiscourseUnit, ...]:
        limit = self._bounded_limit(limit)
        selected = _unique(str(item) for item in chunk_ids)
        candidates: dict[str, int] = {}
        for offset in range(0, len(selected), 400):
            batch = selected[offset : offset + 400]
            if not batch:
                continue
            placeholders = ",".join("?" for _ in batch)
            params: list[Any] = list(batch)
            where = f"ue.chunk_id IN ({placeholders})"
            if artifact_id is not None:
                where += " AND u.artifact_id = ?"
                params.append(artifact_id)
            sql = (
                "SELECT DISTINCT u.unit_id, u.asserted_ordinal "
                "FROM discourse_unit_evidence AS ue "
                "JOIN discourse_units AS u ON u.unit_id = ue.unit_id WHERE "
                + where
                + " ORDER BY u.asserted_ordinal DESC, u.unit_id"
            )
            if limit is not None:
                sql += " LIMIT ?"
                params.append(limit)
            rows = self._db.execute(sql, tuple(params)).fetchall()
            candidates.update((str(row[0]), int(row[1])) for row in rows)
            if limit is not None and len(candidates) > limit:
                candidates = dict(
                    sorted(
                        candidates.items(),
                        key=lambda item: (-item[1], item[0]),
                    )[:limit]
                )
        ids = [
            unit_id
            for unit_id, _ in sorted(
                candidates.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        values = tuple(self.get_unit(item) for item in ids)
        return tuple(item for item in values if item is not None)

    def incident_relations(
        self,
        unit_ids: Sequence[str],
        *,
        artifact_id: str | None = None,
        max_degree: int,
    ) -> dict[str, tuple[DiscourseRelation, ...]]:
        """Return at most ``max_degree`` deterministic incident edges per unit."""

        if max_degree < 0:
            raise ValueError("max_degree must be non-negative")
        result: dict[str, tuple[DiscourseRelation, ...]] = {}
        for unit_id in _unique(str(item) for item in unit_ids):
            params: list[Any] = [unit_id]
            where = "m.unit_id = ?"
            if artifact_id is not None:
                where += " AND r.artifact_id = ?"
                params.append(artifact_id)
            params.append(max_degree)
            rows = self._db.execute(
                "SELECT r.relation_id FROM discourse_relation_members AS m "
                "JOIN discourse_relations AS r ON r.relation_id = m.relation_id "
                "WHERE "
                + where
                + " ORDER BY r.confidence DESC, r.created_ordinal DESC, "
                "r.relation_id LIMIT ?",
                tuple(params),
            ).fetchall()
            values = tuple(self.get_relation(row[0]) for row in rows)
            result[unit_id] = tuple(item for item in values if item is not None)
        return result

    def relations_incident_to(
        self,
        unit_id: str,
        *,
        artifact_id: str | None = None,
        max_degree: int,
    ) -> tuple[DiscourseRelation, ...]:
        return self.incident_relations(
            (unit_id,), artifact_id=artifact_id, max_degree=max_degree
        )[unit_id]

    def incident_units(
        self,
        unit_ids: Sequence[str],
        *,
        artifact_id: str | None = None,
        max_degree: int,
    ) -> dict[str, tuple[DiscourseUnit, ...]]:
        relations = self.incident_relations(
            unit_ids, artifact_id=artifact_id, max_degree=max_degree
        )
        result: dict[str, tuple[DiscourseUnit, ...]] = {}
        for unit_id, incident in relations.items():
            peer_ids: list[str] = []
            for relation in incident:
                peer_ids.extend(
                    member.unit_id
                    for member in relation.members
                    if member.unit_id != unit_id
                )
            peers = tuple(self.get_unit(item) for item in dict.fromkeys(peer_ids))
            result[unit_id] = tuple(
                item for item in peers if item is not None
            )[:max_degree]
        return result

    def relations_for_chunks(
        self,
        chunk_ids: Sequence[str],
        *,
        artifact_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[DiscourseRelation, ...]:
        limit = self._bounded_limit(limit)
        selected = _unique(str(item) for item in chunk_ids)
        candidates: dict[str, tuple[float, int]] = {}
        for offset in range(0, len(selected), 400):
            batch = selected[offset : offset + 400]
            if not batch:
                continue
            placeholders = ",".join("?" for _ in batch)
            params: list[Any] = list(batch)
            where = f"re.chunk_id IN ({placeholders})"
            if artifact_id is not None:
                where += " AND r.artifact_id = ?"
                params.append(artifact_id)
            sql = (
                "SELECT DISTINCT r.relation_id, r.confidence, r.created_ordinal "
                "FROM discourse_relation_evidence AS re "
                "JOIN discourse_relations AS r ON r.relation_id = re.relation_id "
                "WHERE "
                + where
                + " ORDER BY r.confidence DESC, r.created_ordinal DESC, r.relation_id"
            )
            if limit is not None:
                sql += " LIMIT ?"
                params.append(limit)
            rows = self._db.execute(sql, tuple(params)).fetchall()
            candidates.update(
                (str(row[0]), (float(row[1]), int(row[2]))) for row in rows
            )
            if limit is not None and len(candidates) > limit:
                candidates = dict(
                    sorted(
                        candidates.items(),
                        key=lambda item: (-item[1][0], -item[1][1], item[0]),
                    )[:limit]
                )
        ids = [
            relation_id
            for relation_id, _ in sorted(
                candidates.items(),
                key=lambda item: (-item[1][0], -item[1][1], item[0]),
            )
        ]
        values = tuple(self.get_relation(item) for item in ids)
        return tuple(item for item in values if item is not None)

    def stats(self) -> dict[str, int]:
        counts = {}
        for label, table in (
            ("artifacts", "discourse_artifacts"),
            ("episodes", "episodes"),
            ("episode_evidence_spans", "episode_evidence"),
            ("representatives", "episode_representatives"),
            ("units", "discourse_units"),
            ("relations", "discourse_relations"),
            ("relation_members", "discourse_relation_members"),
            ("coverage_rows", "discourse_artifact_coverage"),
            ("coverage_receipts", "discourse_artifact_coverage_receipts"),
            ("revisions", "discourse_graph_revisions"),
        ):
            counts[label] = int(
                self._db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
        graph_tables = (
            "discourse_artifacts",
            "episodes",
            "episode_evidence",
            "episode_representatives",
            "discourse_units",
            "discourse_unit_evidence",
            "discourse_relations",
            "discourse_relation_members",
            "discourse_relation_evidence",
            "discourse_artifact_coverage",
            "discourse_artifact_coverage_receipts",
            "discourse_graph_revisions",
        )
        unsafe_fragments = (
            "activation",
            "attention",
            "cache",
            "generated",
            "hidden_state",
            "prompt",
            "request_token",
            "response_text",
            "token_ids",
        )
        columns = {
            str(row[1]).casefold()
            for table in graph_tables
            for row in self._db.execute(f"PRAGMA table_info({table})").fetchall()
        }
        unsafe = {
            column
            for column in columns
            if any(fragment in column for fragment in unsafe_fragments)
        }
        if unsafe:
            raise DiscourseIdentityError(
                f"discourse schema can retain request state: {sorted(unsafe)}"
            )
        for table, owner in (
            ("discourse_artifacts", "artifact"),
            ("discourse_units", "unit"),
            ("discourse_relations", "relation"),
        ):
            for (raw,) in self._db.execute(f"SELECT metadata FROM {table}"):
                metadata = _strict_json_object(raw, label=f"{owner} metadata")
                _safe_metadata(
                    metadata,
                    label=f"{owner} metadata",
                    owner=owner,
                )
        counts["retained_request_token_state_bytes"] = 0
        counts["retained_token_state_bytes"] = 0
        return counts


__all__ = [
    "ArtifactCoverageMark",
    "DiscourseIdentityError",
    "DiscourseSnapshotError",
    "DiscourseStore",
    "SourceEvidenceError",
]
