from __future__ import annotations

import sqlite3
from dataclasses import replace

import pytest

from memory_condense.domain.discourse import (
    DiscourseArtifact,
    DiscourseRelation,
    DiscourseUnit,
    Episode,
    EpisodeRepresentative,
    EvidenceSpan,
    RelationMember,
    quote_sha256,
)
from memory_condense.domain.discourse_routing import DiscourseUnitRoute
from memory_condense.persistence.db import Database
from memory_condense.persistence.discourse_store import (
    ArtifactCoverageMark,
    DiscourseIdentityError,
    DiscourseSnapshotError,
    DiscourseStore,
    SourceEvidenceError,
)
from memory_condense.persistence.discourse_receipts import (
    discourse_content_digests,
)


def _artifact(artifact_id: str = "disc-test") -> DiscourseArtifact:
    return DiscourseArtifact(
        artifact_id=artifact_id,
        kind="deterministic-boundaries",
        implementation_sha256="a" * 64,
        policy_sha256="b" * 64,
        model_id="local-boundary-scorer",
        model_revision="1",
        checkpoint_sha256="c" * 64,
        metadata={"tokenizer_id": "static-tokenizer", "window": 8},
    )


def _span(
    chunk_id: str,
    text: str,
    ordinal: int,
    source_id: str,
    *,
    start: int = 0,
    end: int | None = None,
) -> EvidenceSpan:
    stop = len(text) if end is None else end
    return EvidenceSpan(
        chunk_id=chunk_id,
        start_char=start,
        end_char=stop,
        quote_sha256=quote_sha256(text[start:stop]),
        ordinal=ordinal,
        source_id=source_id,
        turn_id=f"t{ordinal}",
        role="user",
        created_at="2026-08-18T00:00:00+00:00",
    )


@pytest.fixture
def graph(tmp_path):
    path = tmp_path / "discourse.db"
    db = Database(path)
    rows = (
        ("t1", "c1", "Setup uses SQLite.", "alpha", 1),
        ("t2", "c2", "Experiment produced 42.", "alpha", 2),
        ("t3", "c3", "Decision changed to Postgres.", "alpha", 3),
        ("t4", "c4", "Distant constraint is offline.", "beta", 4),
    )
    for turn_id, chunk_id, text, source_id, ordinal in rows:
        db.execute(
            "INSERT INTO turns "
            "(turn_id, role, text, source_id, created_at, ordinal) "
            "VALUES (?, 'user', ?, ?, '2026-08-18T00:00:00+00:00', ?)",
            (turn_id, text, source_id, ordinal),
        )
        db.execute(
            "INSERT INTO chunks "
            "(chunk_id, turn_id, text, start_char, end_char, token_count) "
            "VALUES (?, ?, ?, 0, ?, 4)",
            (chunk_id, turn_id, text, len(text)),
        )
    db.commit()
    try:
        yield path, db, DiscourseStore(db), {row[1]: row[2] for row in rows}
    finally:
        db.close()


def _episode(
    episode_id: str,
    chunk_id: str,
    text: str,
    ordinal: int,
    sequence_no: int,
    *,
    source_id: str = "alpha",
) -> Episode:
    return Episode(
        episode_id=episode_id,
        artifact_id="disc-test",
        source_id=source_id,
        sequence_no=sequence_no,
        first_ordinal=ordinal,
        last_ordinal=ordinal,
        evidence=(_span(chunk_id, text, ordinal, source_id),),
        boundary_method="source-window",
        initial_boundary=ordinal,
        refined_boundary=ordinal,
        boundary_score=0.75,
        boundary_threshold=0.5,
    )


def _unit(
    unit_id: str,
    kind: str,
    chunk_id: str,
    text: str,
    ordinal: int,
    source_id: str = "alpha",
) -> DiscourseUnit:
    return DiscourseUnit(
        unit_id=unit_id,
        artifact_id="disc-test",
        kind=kind,
        canonical_key=unit_id,
        asserted_ordinal=ordinal,
        confidence=0.9,
        evidence=(_span(chunk_id, text, ordinal, source_id),),
        metadata={"route": kind},
    )


def _relation(
    relation_id: str,
    relation_type: str,
    left: DiscourseUnit,
    right: DiscourseUnit,
    *,
    confidence: float,
) -> DiscourseRelation:
    return DiscourseRelation(
        relation_id=relation_id,
        artifact_id="disc-test",
        relation_type=relation_type,
        members=(
            RelationMember(left.unit_id, "subject", 0),
            RelationMember(right.unit_id, "object", 1),
        ),
        evidence=(*left.evidence, *right.evidence),
        confidence=confidence,
        created_ordinal=max(left.asserted_ordinal, right.asserted_ordinal),
        metadata={"linker": "manual-test"},
    )


def test_atomic_publication_roundtrips_every_contract(graph):
    _, _, store, texts = graph
    artifact = _artifact()
    episode = _episode("ep-1", "c1", texts["c1"], 1, 0)
    representative = EpisodeRepresentative("ep-1", "c1", 0, "d" * 64)
    setup = _unit("u-setup", "setup", "c1", texts["c1"], 1)
    result = _unit("u-result", "result", "c2", texts["c2"], 2)
    relation = _relation("r-supports", "supports", setup, result, confidence=0.8)

    receipt = store.publish(
        artifact,
        episodes=(episode,),
        representatives=(representative,),
        units=(setup, result),
        relations=(relation,),
    )

    assert store.get_artifact(artifact.artifact_id) == artifact
    assert store.get_episode(episode.episode_id) == episode
    assert store.get_representatives(episode.episode_id) == (representative,)
    assert store.get_unit(setup.unit_id) == setup
    assert store.get_relation(relation.relation_id) == relation
    assert store.snapshot(receipt.graph_revision) == receipt
    assert receipt.max_turn_ordinal == 4
    assert receipt.chunk_count == 4
    assert store.stats()["retained_request_token_state_bytes"] == 0


def test_artifact_unit_routes_stream_without_hydrating_evidence(graph, monkeypatch):
    _, _, store, texts = graph
    older = _unit("u-old", "state", "c1", texts["c1"], 1)
    newer = _unit("u-new", "claim", "c3", texts["c3"], 3)
    other_artifact = _artifact("other-artifact")
    other = replace(
        _unit("u-other", "claim", "c2", texts["c2"], 2),
        artifact_id=other_artifact.artifact_id,
    )
    store.publish(_artifact(), units=(older, newer))
    store.publish(other_artifact, units=(other,))

    monkeypatch.setattr(
        store,
        "get_unit",
        lambda _unit_id: (_ for _ in ()).throw(AssertionError("hydrated")),
    )
    routes = tuple(store.iter_unit_routes_for_artifact("disc-test"))

    assert routes == (
        DiscourseUnitRoute.from_unit(newer),
        DiscourseUnitRoute.from_unit(older),
    )


def test_identical_replay_is_idempotent_and_does_not_advance_revision(graph):
    _, _, store, texts = graph
    artifact = _artifact()
    episode = _episode("ep-1", "c1", texts["c1"], 1, 0)
    first = store.publish(artifact, episodes=(episode,))
    second = store.publish(artifact, episodes=(episode, episode))
    assert first == second
    assert store.stats()["revisions"] == 1


def test_identity_mismatch_is_rejected_without_new_revision(graph):
    _, _, store, texts = graph
    artifact = _artifact()
    episode = _episode("ep-1", "c1", texts["c1"], 1, 0)
    first = store.publish(artifact, episodes=(episode,))
    conflicting = Episode(
        episode_id=episode.episode_id,
        artifact_id=episode.artifact_id,
        source_id=episode.source_id,
        sequence_no=episode.sequence_no,
        first_ordinal=episode.first_ordinal,
        last_ordinal=episode.last_ordinal,
        evidence=episode.evidence,
        boundary_method="different-method",
    )
    with pytest.raises(DiscourseIdentityError, match="another identity"):
        store.publish(artifact, episodes=(conflicting,))
    assert store.latest_snapshot() == first
    assert store.get_episode("ep-1") == episode


def test_late_batch_failure_rolls_back_earlier_rows(graph):
    _, _, store, texts = graph
    artifact = _artifact()
    store.publish(artifact)
    new_unit = _unit("u-new", "setup", "c1", texts["c1"], 1)
    missing = _unit("u-missing", "result", "c2", texts["c2"], 2)
    invalid_relation = _relation(
        "r-invalid", "supports", new_unit, missing, confidence=0.7
    )
    before = store.latest_snapshot()
    with pytest.raises(KeyError, match="u-missing"):
        store.publish(artifact, units=(new_unit,), relations=(invalid_relation,))
    assert store.get_unit("u-new") is None
    assert store.get_relation("r-invalid") is None
    assert store.latest_snapshot() == before


def test_bad_hash_or_source_is_rejected_on_write(graph):
    _, _, store, texts = graph
    artifact = _artifact()
    bad = EvidenceSpan("c1", 0, len(texts["c1"]), "e" * 64, 1, "alpha")
    episode = Episode(
        episode_id="ep-bad",
        artifact_id=artifact.artifact_id,
        source_id="alpha",
        sequence_no=0,
        first_ordinal=1,
        last_ordinal=1,
        evidence=(bad,),
        boundary_method="test",
    )
    with pytest.raises(SourceEvidenceError, match="quote hash"):
        store.publish(artifact, episodes=(episode,))
    assert store.get_artifact(artifact.artifact_id) is None


def test_source_is_revalidated_on_every_read(graph):
    _, db, store, texts = graph
    episode = _episode("ep-1", "c1", texts["c1"], 1, 0)
    store.publish(_artifact(), episodes=(episode,))
    db.execute("UPDATE turns SET text = 'tampered source' WHERE turn_id = 't1'")
    db.commit()
    with pytest.raises(SourceEvidenceError, match="chunk 'c1'"):
        store.get_episode("ep-1")


def test_full_chunk_fail_open_evidence_is_verified_and_input_ordered(graph):
    _, _, store, texts = graph
    spans = store.evidence_for_chunks(("c3", "c1", "c3"))
    assert [item.chunk_id for item in spans] == ["c3", "c1"]
    assert [store.hydrate_span(item) for item in spans] == [texts["c3"], texts["c1"]]
    assert [item.ordinal for item in spans] == [3, 1]


def test_same_turn_chunks_persist_authoritative_source_order_not_id_order(graph):
    _, db, store, _ = graph
    turn_text = "first|second"
    db.execute(
        "INSERT INTO turns "
        "(turn_id, role, text, source_id, created_at, ordinal) "
        "VALUES ('t5', 'user', ?, 'gamma', '2026-08-18', 5)",
        (turn_text,),
    )
    # IDs deliberately sort opposite to their authoritative source positions.
    db.execute(
        "INSERT INTO chunks "
        "(chunk_id, turn_id, text, start_char, end_char, token_count) "
        "VALUES ('z-first', 't5', 'first', 0, 5, 1)"
    )
    db.execute(
        "INSERT INTO chunks "
        "(chunk_id, turn_id, text, start_char, end_char, token_count) "
        "VALUES ('a-second', 't5', 'second', 6, 12, 1)"
    )
    db.commit()

    spans = store.evidence_for_chunks(("z-first", "a-second"))
    assert [item.turn_start_char for item in spans] == [0, 6]
    episode = Episode(
        episode_id="ep-same-turn",
        artifact_id="disc-test",
        source_id="gamma",
        sequence_no=0,
        first_ordinal=5,
        last_ordinal=5,
        evidence=spans,
        boundary_method="source-order-test",
    )
    store.publish(_artifact(), episodes=(episode,))
    assert store.get_episode(episode.episode_id) == episode

    wrong_position = EvidenceSpan(
        chunk_id="a-second",
        start_char=0,
        end_char=6,
        quote_sha256=quote_sha256("second"),
        ordinal=5,
        source_id="gamma",
        # Compatibility default zero is not valid for this nonzero-start chunk.
    )
    with pytest.raises(SourceEvidenceError, match="turn start"):
        store.hydrate_span(wrong_position)


def test_source_order_adjacency_and_chunk_mapping_are_deterministic(graph):
    _, _, store, texts = graph
    episodes = (
        _episode("ep-0", "c1", texts["c1"], 1, 0),
        _episode("ep-1", "c2", texts["c2"], 2, 1),
        _episode("ep-2", "c3", texts["c3"], 3, 2),
    )
    store.publish(_artifact(), episodes=tuple(reversed(episodes)))
    assert [item.episode_id for item in store.episodes_for_source("disc-test", "alpha")] == [
        "ep-0",
        "ep-1",
        "ep-2",
    ]
    assert [item.episode_id for item in store.adjacent_episodes("ep-1")] == [
        "ep-0",
        "ep-2",
    ]
    assert list(store.episode_ids_for_chunks(("c3", "missing", "c1")).items()) == [
        ("c3", "ep-2"),
        ("c1", "ep-0"),
    ]


def test_source_local_episode_order_violation_rolls_back(graph):
    _, _, store, texts = graph
    later = _episode("ep-later", "c3", texts["c3"], 3, 0)
    earlier = _episode("ep-earlier", "c1", texts["c1"], 1, 1)
    with pytest.raises(ValueError, match="source-local sequence order"):
        store.publish(_artifact(), episodes=(later, earlier))
    assert store.get_artifact("disc-test") is None
    assert store.get_episode("ep-later") is None


def test_representative_must_be_inside_its_episode(graph):
    _, _, store, texts = graph
    episode = _episode("ep-1", "c1", texts["c1"], 1, 0)
    outside = EpisodeRepresentative("ep-1", "c2", 0, "d" * 64)
    with pytest.raises(ValueError, match="must cite an episode chunk"):
        store.publish(
            _artifact(), episodes=(episode,), representatives=(outside,)
        )
    assert store.get_episode("ep-1") is None


def test_incident_queries_apply_per_unit_degree_cap(graph):
    _, _, store, texts = graph
    setup = _unit("u-setup", "setup", "c1", texts["c1"], 1)
    result = _unit("u-result", "result", "c2", texts["c2"], 2)
    decision = _unit("u-decision", "decision", "c3", texts["c3"], 3)
    support = _relation("r-support", "supports", setup, result, confidence=0.6)
    revision = _relation("r-revision", "revises", result, decision, confidence=0.9)
    store.publish(
        _artifact(),
        units=(setup, result, decision),
        relations=(support, revision),
    )
    assert store.incident_relations(("u-result",), max_degree=1)["u-result"] == (
        revision,
    )
    assert store.units_for_chunks(("c2",)) == (result,)
    assert store.relations_for_chunks(("c1", "c3")) == (revision, support)


def test_relations_cannot_cross_artifact_identity(graph):
    _, _, store, texts = graph
    first = _unit("u-first", "setup", "c1", texts["c1"], 1)
    store.publish(_artifact(), units=(first,))
    other_artifact = _artifact("disc-other")
    other = DiscourseUnit(
        unit_id="u-other",
        artifact_id="disc-other",
        kind="result",
        canonical_key="other",
        asserted_ordinal=2,
        confidence=0.8,
        evidence=(_span("c2", texts["c2"], 2, "alpha"),),
    )
    store.publish(other_artifact, units=(other,))
    cross = DiscourseRelation(
        relation_id="r-cross",
        artifact_id="disc-test",
        relation_type="supports",
        members=(
            RelationMember("u-first", "subject", 0),
            RelationMember("u-other", "object", 1),
        ),
        evidence=(*first.evidence, *other.evidence),
        confidence=0.8,
        created_ordinal=2,
    )
    with pytest.raises(ValueError, match="another artifact"):
        store.publish(_artifact(), relations=(cross,))


@pytest.mark.parametrize(
    "metadata",
    [
        {"token_ids": [1, 2]},
        {"nested": {"activations": [0.1]}},
        {"generated_text": "unsupported summary"},
        {"past_key_values": [[1.0]]},
    ],
)
def test_request_shaped_metadata_is_never_persisted(graph, metadata):
    _, _, store, _ = graph
    artifact = DiscourseArtifact(
        artifact_id="disc-unsafe",
        kind="test",
        implementation_sha256="a" * 64,
        policy_sha256="b" * 64,
        metadata=metadata,
    )
    with pytest.raises(ValueError, match="cannot persist request-derived"):
        store.publish(artifact)
    assert store.get_artifact("disc-unsafe") is None


def test_graph_schema_has_no_evidence_text_or_transformer_state_columns(graph):
    _, db, _, _ = graph
    tables = (
        "discourse_artifacts",
        "episodes",
        "episode_evidence",
        "episode_representatives",
        "discourse_units",
        "discourse_unit_evidence",
        "discourse_relations",
        "discourse_relation_members",
        "discourse_relation_evidence",
        "discourse_graph_revisions",
    )
    columns = {
        row[1]
        for table in tables
        for row in db.execute(f"PRAGMA table_info({table})").fetchall()
    }
    assert not {
        "text",
        "quote",
        "token_ids",
        "activations",
        "hidden_states",
        "kv_cache",
    } & columns


def test_snapshot_receipt_is_immutable_and_detects_stale_high_water(graph):
    _, db, store, _ = graph
    receipt = store.publish(_artifact())
    assert store.validate_snapshot(receipt)
    db.execute(
        "INSERT INTO turns "
        "(turn_id, role, text, source_id, created_at, ordinal) "
        "VALUES ('t5', 'user', 'new turn', 'alpha', '2026-08-18', 5)"
    )
    db.commit()
    assert store.snapshot(receipt.graph_revision) == receipt
    with pytest.raises(DiscourseSnapshotError, match="not current"):
        store.validate_snapshot(receipt)
    assert store.validate_snapshot(receipt, require_current=False)


def test_snapshot_receipt_rejects_update_and_delete_at_storage_boundary(graph):
    _, db, store, _ = graph
    receipt = store.publish(_artifact())
    with pytest.raises(sqlite3.IntegrityError, match="receipts are immutable"):
        db.execute(
            "UPDATE discourse_graph_revisions SET chunk_count = 0 "
            "WHERE graph_revision = ?",
            (receipt.graph_revision,),
        )
    db.connection.rollback()
    with pytest.raises(sqlite3.IntegrityError, match="receipts are immutable"):
        db.execute(
            "DELETE FROM discourse_graph_revisions WHERE graph_revision = ?",
            (receipt.graph_revision,),
        )
    db.connection.rollback()
    assert store.snapshot() == receipt


def test_every_published_graph_row_family_is_update_immutable(graph):
    _, db, store, texts = graph
    episode = _episode("ep-1", "c1", texts["c1"], 1, 0)
    representative = EpisodeRepresentative("ep-1", "c1", 0, "d" * 64)
    left = _unit("u-left", "setup", "c1", texts["c1"], 1)
    right = _unit("u-right", "result", "c2", texts["c2"], 2)
    relation = _relation("r-edge", "supports", left, right, confidence=0.8)
    receipt = store.publish(
        _artifact(),
        episodes=(episode,),
        representatives=(representative,),
        units=(left, right),
        relations=(relation,),
    )
    db.commit()
    for table, column in (
        ("discourse_artifacts", "kind"),
        ("episodes", "boundary_method"),
        ("episode_evidence", "quote_sha256"),
        ("episode_representatives", "rank"),
        ("discourse_units", "canonical_key"),
        ("discourse_unit_evidence", "quote_sha256"),
        ("discourse_relations", "relation_type"),
        ("discourse_relation_members", "role"),
        ("discourse_relation_evidence", "quote_sha256"),
    ):
        with pytest.raises(sqlite3.IntegrityError, match="rows are immutable"):
            db.execute(f"UPDATE {table} SET {column} = {column}")
        db.connection.rollback()
    assert store.validate_snapshot(receipt)


def test_read_only_store_can_verify_but_not_publish(graph):
    path, db, store, _ = graph
    receipt = store.publish(_artifact())
    db.close()
    with Database(path, read_only=True) as read_db:
        read_store = DiscourseStore(read_db)
        assert read_store.get_artifact("disc-test") == _artifact()
        assert read_store.snapshot() == receipt
        with pytest.raises(sqlite3.OperationalError):
            read_store.publish(_artifact("disc-new"))


def test_multispan_graph_identity_is_canonical_and_member_ordinals_are_dense():
    first = _span("z-first", "first", 1, "alpha")
    second = _span("a-second", "second", 2, "alpha")
    left = DiscourseUnit(
        unit_id="u-left",
        artifact_id="disc-test",
        kind="claim",
        canonical_key="left",
        asserted_ordinal=2,
        confidence=1.0,
        evidence=(second, first),
    )
    right = DiscourseUnit(
        unit_id="u-right",
        artifact_id="disc-test",
        kind="claim",
        canonical_key="right",
        asserted_ordinal=2,
        confidence=1.0,
        evidence=(second,),
    )
    relation = DiscourseRelation(
        relation_id="r-canonical",
        artifact_id="disc-test",
        relation_type="supports",
        members=(
            RelationMember(right.unit_id, "object", 1),
            RelationMember(left.unit_id, "subject", 0),
        ),
        evidence=(second, first),
        confidence=1.0,
        created_ordinal=2,
    )

    assert left.evidence == (first, second)
    assert relation.evidence == (first, second)
    assert tuple(member.ordinal for member in relation.members) == (0, 1)
    with pytest.raises(ValueError, match="contiguous from zero"):
        DiscourseRelation(
            relation_id="r-gapped",
            artifact_id="disc-test",
            relation_type="supports",
            members=(
                RelationMember("u-left", "subject", 0),
                RelationMember("u-right", "object", 2),
            ),
            evidence=(first, second),
            confidence=1.0,
            created_ordinal=2,
        )


def test_new_publication_rejects_evidence_without_authoritative_source(graph):
    _, _, store, texts = graph
    span = replace(_span("c1", texts["c1"], 1, "alpha"), source_id=None)
    unit = DiscourseUnit(
        unit_id="u-missing-source",
        artifact_id="disc-test",
        kind="claim",
        canonical_key="missing source",
        asserted_ordinal=1,
        confidence=1.0,
        evidence=(span,),
    )
    with pytest.raises(SourceEvidenceError, match="provenance.*incomplete"):
        store.publish(_artifact(), units=(unit,))


def test_same_count_source_mutation_invalidates_content_bound_snapshot(graph):
    _, db, store, _ = graph
    before = store.publish(_artifact())
    db.execute(
        "UPDATE turns SET text = 'Setup uses DuckDB.' WHERE turn_id = 't1'"
    )
    db.execute(
        "UPDATE chunks SET text = 'Setup uses DuckDB.' WHERE chunk_id = 'c1'"
    )
    db.commit()
    after = store.snapshot()
    assert after.chunk_count == before.chunk_count
    assert after.max_turn_ordinal == before.max_turn_ordinal
    assert after.source_revision > before.source_revision
    assert after.source_content_sha256 != before.source_content_sha256
    assert after.snapshot_sha256 != before.snapshot_sha256
    with pytest.raises(DiscourseSnapshotError, match="not current"):
        store.validate_snapshot(before)


def test_rolled_back_outer_transaction_cannot_poison_content_root_cache(graph):
    _, db, store, _ = graph
    first = store.publish(_artifact("disc-a"))
    db.commit()
    assert store.snapshot() == first

    db.execute("BEGIN")
    failed = store.publish(_artifact("disc-b"))
    db.connection.rollback()
    assert store.snapshot() == first

    committed = store.publish(_artifact("disc-c"))
    _, authoritative_graph_root = discourse_content_digests(db)
    assert committed.graph_content_sha256 == authoritative_graph_root
    assert committed.graph_content_sha256 != failed.graph_content_sha256
    assert store.snapshot() == committed


def test_same_shape_databases_with_different_source_bytes_never_hash_identically(
    tmp_path,
):
    snapshots = []
    for name, text in (("first", "source-A"), ("second", "source-B")):
        with Database(tmp_path / f"{name}.db") as db:
            db.execute(
                "INSERT INTO turns "
                "(turn_id, role, text, source_id, created_at, ordinal) "
                "VALUES ('t1', 'user', ?, 'thread', '2026-08-18', 1)",
                (text,),
            )
            db.execute(
                "INSERT INTO chunks "
                "(chunk_id, turn_id, text, start_char, end_char, token_count) "
                "VALUES ('c1', 't1', ?, 0, 8, 1)",
                (text,),
            )
            db.commit()
            snapshots.append(DiscourseStore(db).publish(_artifact()))
    assert snapshots[0].source_revision == snapshots[1].source_revision
    assert snapshots[0].chunk_count == snapshots[1].chunk_count
    assert snapshots[0].artifact_ids == snapshots[1].artifact_ids
    assert snapshots[0].source_content_sha256 != snapshots[1].source_content_sha256
    assert snapshots[0].snapshot_sha256 != snapshots[1].snapshot_sha256


def test_finalized_coverage_is_whole_corpus_and_stales_on_source_growth(graph):
    _, db, store, _ = graph
    marks = tuple(
        ArtifactCoverageMark(f"c{index}", "discourse", "no_output")
        for index in range(1, 5)
    )
    store.publish(_artifact(), coverage=marks)
    receipt = store.finalize_artifact_coverage("disc-test")
    assert receipt.chunk_count == 4
    assert store.artifact_coverage("disc-test") == receipt
    db.execute(
        "INSERT INTO turns "
        "(turn_id, role, text, source_id, created_at, ordinal) "
        "VALUES ('t5', 'user', 'new', 'alpha', '2026-08-18', 5)"
    )
    db.execute(
        "INSERT INTO chunks "
        "(chunk_id, turn_id, text, start_char, end_char, token_count) "
        "VALUES ('c5', 't5', 'new', 0, 3, 1)"
    )
    db.commit()
    assert store.artifact_coverage("disc-test") is None


@pytest.mark.parametrize(
    ("turn_text", "chunk_text"),
    (("hidden binding constraint", None), ("visible hidden", "visible")),
)
def test_coverage_finalization_rejects_uninspectable_turn_bytes(
    tmp_path,
    turn_text,
    chunk_text,
):
    with Database(tmp_path / f"coverage-{chunk_text or 'orphan'}.db") as db:
        db.execute(
            "INSERT INTO turns "
            "(turn_id, role, text, source_id, created_at, ordinal) "
            "VALUES ('t1', 'user', ?, 'thread', '2026-08-18', 1)",
            (turn_text,),
        )
        marks = ()
        if chunk_text is not None:
            db.execute(
                "INSERT INTO chunks "
                "(chunk_id, turn_id, text, start_char, end_char, token_count) "
                "VALUES ('c1', 't1', ?, 0, ?, 1)",
                (chunk_text, len(chunk_text)),
            )
            marks = (ArtifactCoverageMark("c1", "discourse", "no_output"),)
        db.commit()
        store = DiscourseStore(db)
        store.publish(_artifact(), coverage=marks)
        with pytest.raises(SourceEvidenceError, match="uncovered non-whitespace"):
            store.finalize_artifact_coverage("disc-test")


def test_coverage_finalization_blocks_cross_connection_source_toctou(graph):
    path, db, store, _ = graph
    store.publish(
        _artifact(),
        coverage=tuple(
            ArtifactCoverageMark(f"c{index}", "discourse", "no_output")
            for index in range(1, 5)
        ),
    )

    class _MutatingStore(DiscourseStore):
        attempted = False
        mutation_blocked = False

        def coverage_for_chunks(self, *args, **kwargs):
            result = super().coverage_for_chunks(*args, **kwargs)
            if not self.attempted:
                self.attempted = True
                external = sqlite3.connect(str(path), timeout=0.0)
                try:
                    external.execute(
                        "UPDATE turns SET text = 'Setup uses DuckDB.' "
                        "WHERE turn_id = 't1'"
                    )
                    external.commit()
                except sqlite3.OperationalError as exc:
                    self.mutation_blocked = "locked" in str(exc).lower()
                    external.rollback()
                finally:
                    external.close()
            return result

    finalizer = _MutatingStore(db)
    receipt = finalizer.finalize_artifact_coverage("disc-test")
    assert finalizer.mutation_blocked
    assert finalizer.artifact_coverage("disc-test") == receipt
