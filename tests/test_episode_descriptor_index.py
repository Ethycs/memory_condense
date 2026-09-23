from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from memory_condense.domain.discourse import (
    DiscourseArtifact,
    Episode,
    EpisodeRepresentative,
    EvidenceSpan,
    identity_sha256,
    quote_sha256,
)
from memory_condense.persistence.db import Database
from memory_condense.persistence.discourse_store import DiscourseStore
from memory_condense.search.episodes.descriptor_index import (
    EpisodeDescriptorIndex,
    EpisodeDescriptorIndexUnavailable,
)


ARTIFACT_ID = "disc-descriptor-test"
EMBEDDING_IDENTITY = {
    "model_id": "fixture-embedding",
    "revision": "1",
    "dimension": 2,
}
CREATED_AT = "2026-09-04T00:00:00+00:00"


def _artifact() -> DiscourseArtifact:
    return DiscourseArtifact(
        artifact_id=ARTIFACT_ID,
        kind="episode-descriptor-fixture",
        implementation_sha256="a" * 64,
        policy_sha256="b" * 64,
        model_id="fixture",
        model_revision="1",
        checkpoint_sha256="c" * 64,
    )


def _vector_identity(chunk_id: str, vector: Sequence[float]) -> str:
    values = np.asarray(vector, dtype=np.float32)
    return identity_sha256(
        {
            "method": "ordinary_embedding",
            "chunk_id": chunk_id,
            "vector": [
                0.0 if float(value) == 0.0 else float(value)
                for value in values
            ],
        }
    )


def _lexical_identity(chunk_id: str) -> str:
    text = f"source text for {chunk_id}"
    return identity_sha256(
        {
            "method": "lexical_control",
            "chunk_id": chunk_id,
            "text_sha256": identity_sha256(text),
        }
    )


def _publish_graph(
    path: Path,
    specs: Sequence[
        tuple[
            str,
            str,
            int,
            Sequence[tuple[str, Sequence[float], int, str | None]],
        ]
    ],
) -> tuple[Database, str]:
    db = Database(path)
    episodes: list[Episode] = []
    representatives: list[EpisodeRepresentative] = []
    ordinal = 1
    for episode_id, source_id, sequence_no, rep_specs in specs:
        spans: list[EvidenceSpan] = []
        for chunk_id, vector, rank, identity_override in rep_specs:
            text = f"source text for {chunk_id}"
            turn_id = f"turn-{chunk_id}"
            raw = np.asarray(vector, dtype=np.float32)
            db.execute(
                "INSERT INTO turns "
                "(turn_id, role, text, source_id, created_at, ordinal) "
                "VALUES (?, 'user', ?, ?, ?, ?)",
                (turn_id, text, source_id, CREATED_AT, ordinal),
            )
            db.execute(
                "INSERT INTO chunks "
                "(chunk_id, turn_id, text, start_char, end_char, token_count, "
                "embedding) VALUES (?, ?, ?, 0, ?, 4, ?)",
                (chunk_id, turn_id, text, len(text), raw.tobytes()),
            )
            spans.append(
                EvidenceSpan(
                    chunk_id=chunk_id,
                    start_char=0,
                    end_char=len(text),
                    quote_sha256=quote_sha256(text),
                    ordinal=ordinal,
                    source_id=source_id,
                    turn_id=turn_id,
                    role="user",
                    created_at=CREATED_AT,
                )
            )
            representatives.append(
                EpisodeRepresentative(
                    episode_id=episode_id,
                    chunk_id=chunk_id,
                    rank=rank,
                    vector_identity_sha256=(
                        identity_override
                        if identity_override is not None
                        else _vector_identity(chunk_id, vector)
                    ),
                )
            )
            ordinal += 1
        episodes.append(
            Episode(
                episode_id=episode_id,
                artifact_id=ARTIFACT_ID,
                source_id=source_id,
                sequence_no=sequence_no,
                first_ordinal=spans[0].ordinal,
                last_ordinal=spans[-1].ordinal,
                evidence=tuple(spans),
                boundary_method="fixture",
            )
        )
    snapshot = DiscourseStore(db).publish(
        _artifact(),
        episodes=tuple(episodes),
        representatives=tuple(representatives),
    )
    db.commit()
    return db, snapshot.snapshot_sha256


def _standard_graph(path: Path) -> tuple[Database, str]:
    return _publish_graph(
        path,
        (
            (
                "ep-a",
                "source-a",
                0,
                (
                    ("chunk-a0", (0.0, 1.0), 0, None),
                    ("chunk-a1", (1.0, 0.0), 1, None),
                ),
            ),
            (
                "ep-b",
                "source-a",
                1,
                (("chunk-b0", (0.8, 0.6), 0, None),),
            ),
            (
                "ep-c",
                "source-b",
                0,
                (("chunk-c0", (0.8, 0.6), 0, None),),
            ),
        ),
    )


def test_builds_once_with_one_ordered_join_and_scores_max_cosine(
    tmp_path,
    monkeypatch,
) -> None:
    db, snapshot_sha256 = _standard_graph(tmp_path / "memory.db")
    joined_queries: list[str] = []
    original_execute = db.execute

    def recording_execute(sql: str, params: tuple = ()):
        if "FROM discourse_artifacts AS a" in sql:
            joined_queries.append(sql)
        return original_execute(sql, params)

    monkeypatch.setattr(db, "execute", recording_execute)
    index = EpisodeDescriptorIndex(db, dimension=2)
    try:
        first = index.ensure(
            ARTIFACT_ID,
            snapshot_sha256=snapshot_sha256,
            embedding_identity=EMBEDDING_IDENTITY,
        )
        second = index.ensure(
            ARTIFACT_ID,
            snapshot_sha256=snapshot_sha256,
            embedding_identity=EMBEDDING_IDENTITY,
        )

        assert first.rebuilt is True
        assert second.rebuilt is False
        assert first.receipt == second.receipt
        assert len(joined_queries) == 1
        assert "ORDER BY" in joined_queries[0]
        assert first.receipt.episode_count == 3
        assert first.receipt.representative_count == 4
        assert index.resident_bytes == 4 * 2 * np.dtype(np.float32).itemsize

        first_rep_only = index.score(
            (1.0, 0.0),
            ("ep-a", "ep-b"),
            max_representatives=1,
            embedding_identity=EMBEDDING_IDENTITY,
        )
        both_reps = index.score(
            (1.0, 0.0),
            ("ep-a", "ep-b"),
            max_representatives=2,
            embedding_identity=EMBEDDING_IDENTITY,
        )
        assert tuple(item.episode_id for item in first_rep_only.scores) == (
            "ep-b",
            "ep-a",
        )
        assert tuple(item.episode_id for item in both_reps.scores) == (
            "ep-a",
            "ep-b",
        )
        assert both_reps.scores[0].score == pytest.approx(1.0)
    finally:
        db.close()


def test_lexically_selected_representative_can_use_its_durable_vector(
    tmp_path,
) -> None:
    chunk_id = "chunk-lexical"
    db, snapshot_sha256 = _publish_graph(
        tmp_path / "memory.db",
        (
            (
                "ep-lexical",
                "source-a",
                0,
                ((chunk_id, (0.0, 2.0), 0, _lexical_identity(chunk_id)),),
            ),
        ),
    )
    try:
        index = EpisodeDescriptorIndex(db, dimension=2)
        built = index.ensure(
            ARTIFACT_ID,
            snapshot_sha256=snapshot_sha256,
            embedding_identity=EMBEDDING_IDENTITY,
        )
        scored = index.score(
            (0.0, 1.0),
            ("ep-lexical",),
            max_representatives=1,
            embedding_identity=EMBEDDING_IDENTITY,
        )

        assert built.receipt.episode_count == 1
        assert scored.scores[0].episode_id == "ep-lexical"
        assert scored.scores[0].score == pytest.approx(1.0)
    finally:
        db.close()


def test_equal_scores_keep_the_offered_episode_order(tmp_path) -> None:
    db, snapshot_sha256 = _standard_graph(tmp_path / "memory.db")
    index = EpisodeDescriptorIndex(db, dimension=2)
    try:
        index.ensure(
            ARTIFACT_ID,
            snapshot_sha256=snapshot_sha256,
            embedding_identity=EMBEDDING_IDENTITY,
        )
        result = index.score(
            (1.0, 0.0),
            ("ep-c", "ep-b"),
            max_representatives=1,
            embedding_identity=EMBEDDING_IDENTITY,
        )
        assert tuple(item.episode_id for item in result.scores) == (
            "ep-c",
            "ep-b",
        )
        assert result.scores[0].score == result.scores[1].score
    finally:
        db.close()


def test_receipts_exclude_text_vectors_and_nondeterministic_timings(tmp_path) -> None:
    source_text = "source text for chunk-a0"
    db, snapshot_sha256 = _standard_graph(tmp_path / "memory.db")
    index = EpisodeDescriptorIndex(db, dimension=2)
    try:
        ensured = index.ensure(
            ARTIFACT_ID,
            snapshot_sha256=snapshot_sha256,
            embedding_identity=EMBEDDING_IDENTITY,
        )
        scored = index.score(
            (1.0, 0.0),
            ("ep-a", "ep-b"),
            max_representatives=2,
            embedding_identity=EMBEDDING_IDENTITY,
        )
        catalog_identity = json.dumps(ensured.receipt.identity_payload())
        score_identity = json.dumps(scored.receipt.identity_payload())
        for encoded in (catalog_identity, score_identity):
            assert source_text not in encoded
            assert '"vector"' not in encoded
            assert "elapsed_ms" not in encoded
        assert ensured.timing.elapsed_ms >= 0.0
        assert scored.timing.elapsed_ms >= 0.0
        assert (
            scored.timing.semantic_receipt_sha256
            == scored.receipt.receipt_sha256
        )
    finally:
        db.close()


def test_read_only_database_build_and_score_do_not_mutate_store(tmp_path) -> None:
    path = tmp_path / "memory.db"
    writable, snapshot_sha256 = _standard_graph(path)
    writable.close()
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    read_only = Database(path, read_only=True)
    index = EpisodeDescriptorIndex(read_only, dimension=2)
    try:
        ensured = index.ensure(
            ARTIFACT_ID,
            snapshot_sha256=snapshot_sha256,
            embedding_identity=EMBEDDING_IDENTITY,
        )
        scored = index.score(
            (1.0, 0.0),
            ("ep-a", "ep-b"),
            max_representatives=2,
            embedding_identity=EMBEDDING_IDENTITY,
        )
        assert ensured.receipt.representative_count == 4
        assert scored.scores[0].episode_id == "ep-a"
    finally:
        read_only.close()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize(
    ("vector", "identity_override", "expected_reason"),
    (
        ((1.0, 0.0), "f" * 64, "vector_identity_mismatch"),
        ((0.0, 0.0), None, "zero_representative_vector"),
        ((1.0, 0.0, 0.0), None, "dimension_mismatch"),
        ((float("nan"), 0.0), "f" * 64, "invalid_representative_vector"),
    ),
)
def test_corrupt_or_incompatible_vectors_are_typed_failures(
    tmp_path,
    vector,
    identity_override,
    expected_reason,
) -> None:
    db, snapshot_sha256 = _publish_graph(
        tmp_path / "memory.db",
        (
            (
                "ep-a",
                "source-a",
                0,
                (("chunk-a", vector, 0, identity_override),),
            ),
        ),
    )
    try:
        with pytest.raises(EpisodeDescriptorIndexUnavailable) as caught:
            EpisodeDescriptorIndex(db, dimension=2).ensure(
                ARTIFACT_ID,
                snapshot_sha256=snapshot_sha256,
                embedding_identity=EMBEDDING_IDENTITY,
            )
        assert caught.value.reason == expected_reason
    finally:
        db.close()


def test_noncontiguous_representative_ranks_fail_safely(tmp_path) -> None:
    db, snapshot_sha256 = _publish_graph(
        tmp_path / "memory.db",
        (
            (
                "ep-a",
                "source-a",
                0,
                (("chunk-a", (1.0, 0.0), 1, None),),
            ),
        ),
    )
    try:
        with pytest.raises(EpisodeDescriptorIndexUnavailable) as caught:
            EpisodeDescriptorIndex(db, dimension=2).ensure(
                ARTIFACT_ID,
                snapshot_sha256=snapshot_sha256,
                embedding_identity=EMBEDDING_IDENTITY,
            )
        assert caught.value.reason == "invalid_representative_rank"
    finally:
        db.close()


def test_representative_outside_episode_fails_safely(tmp_path) -> None:
    db, snapshot_sha256 = _publish_graph(
        tmp_path / "memory.db",
        (
            (
                "ep-a",
                "source-a",
                0,
                (("chunk-a", (1.0, 0.0), 0, None),),
            ),
            (
                "ep-b",
                "source-a",
                1,
                (("chunk-b", (0.0, 1.0), 0, None),),
            ),
        ),
    )
    try:
        db.execute("DROP TRIGGER trg_episode_representatives_no_update")
        db.execute(
            "UPDATE episode_representatives SET chunk_id = 'chunk-b' "
            "WHERE episode_id = 'ep-a'"
        )
        db.commit()
        with pytest.raises(EpisodeDescriptorIndexUnavailable) as caught:
            EpisodeDescriptorIndex(db, dimension=2).ensure(
                ARTIFACT_ID,
                snapshot_sha256=snapshot_sha256,
                embedding_identity=EMBEDDING_IDENTITY,
            )
        assert caught.value.reason == "representative_outside_episode"
    finally:
        db.close()


def test_revision_change_invalidates_scores_and_release_is_terminal(tmp_path) -> None:
    db, snapshot_sha256 = _standard_graph(tmp_path / "memory.db")
    index = EpisodeDescriptorIndex(db, dimension=2)
    try:
        index.ensure(
            ARTIFACT_ID,
            snapshot_sha256=snapshot_sha256,
            embedding_identity=EMBEDDING_IDENTITY,
        )
        db.execute(
            "INSERT OR REPLACE INTO meta (key, value) "
            "VALUES ('chunk_index_revision', '1')"
        )
        db.commit()
        with pytest.raises(EpisodeDescriptorIndexUnavailable) as stale:
            index.score(
                (1.0, 0.0),
                ("ep-a",),
                max_representatives=1,
                embedding_identity=EMBEDDING_IDENTITY,
            )
        assert stale.value.reason == "stale_catalog"

        index.invalidate()
        assert index.built is False
        rebuilt = index.ensure(
            ARTIFACT_ID,
            snapshot_sha256=snapshot_sha256,
            embedding_identity=EMBEDDING_IDENTITY,
        )
        assert rebuilt.rebuilt is True
        index.release()
        assert index.built is False
        assert index.resident_bytes == 0
        with pytest.raises(EpisodeDescriptorIndexUnavailable) as released:
            index.ensure(
                ARTIFACT_ID,
                snapshot_sha256=snapshot_sha256,
                embedding_identity=EMBEDDING_IDENTITY,
            )
        assert released.value.reason == "released"
    finally:
        db.close()


def test_missing_episode_or_wrong_embedding_identity_never_returns_partial_scores(
    tmp_path,
) -> None:
    db, snapshot_sha256 = _standard_graph(tmp_path / "memory.db")
    index = EpisodeDescriptorIndex(db, dimension=2)
    try:
        index.ensure(
            ARTIFACT_ID,
            snapshot_sha256=snapshot_sha256,
            embedding_identity=EMBEDDING_IDENTITY,
        )
        with pytest.raises(EpisodeDescriptorIndexUnavailable) as missing:
            index.score(
                (1.0, 0.0),
                ("ep-a", "missing"),
                max_representatives=1,
                embedding_identity=EMBEDDING_IDENTITY,
            )
        assert missing.value.reason == "missing_episode"

        foreign_identity = dict(EMBEDDING_IDENTITY, revision="2")
        with pytest.raises(EpisodeDescriptorIndexUnavailable) as foreign:
            index.score(
                (1.0, 0.0),
                ("ep-a",),
                max_representatives=1,
                embedding_identity=foreign_identity,
            )
        assert foreign.value.reason == "embedding_identity_mismatch"
    finally:
        db.close()
