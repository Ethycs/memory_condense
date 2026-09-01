from __future__ import annotations

from argparse import Namespace

import pytest

from memory_condense.domain.discourse import (
    DiscourseArtifact,
    Episode,
    EvidenceSpan,
    quote_sha256,
)
from memory_condense.persistence.db import Database
from memory_condense.persistence.discourse_store import DiscourseStore
from tools.matched_eval.contracts import identity_sha256
from tools.run_reduced_source_group_reinjection_assay import (
    ReducedSourceGroupReinjectionAssayError,
    _resolve_episode_artifact,
)


def _args(*, auto: bool = False, explicit: str | None = None) -> Namespace:
    return Namespace(
        auto_resolve_episode_artifact=auto,
        episode_artifact_id=explicit,
        max_episode_anchors=8,
        previous_episodes=1,
        next_episodes=1,
        max_episode_seeds=24,
        max_episode_direct_fallbacks=16,
    )


def _publish_episode(
    database: Database,
    *,
    artifact: DiscourseArtifact,
    ordinal: int,
) -> Episode:
    source_id = "history-a"
    turn_id = f"turn-{ordinal}"
    chunk_id = f"chunk-{ordinal}"
    text = f"Exact episode assertion {ordinal}."
    created_at = "2026-08-29T00:00:00+00:00"
    database.execute(
        "INSERT INTO turns "
        "(turn_id, role, text, source_id, created_at, ordinal) "
        "VALUES (?, 'user', ?, ?, ?, ?)",
        (turn_id, text, source_id, created_at, ordinal),
    )
    database.execute(
        "INSERT INTO chunks "
        "(chunk_id, turn_id, text, start_char, end_char, token_count) "
        "VALUES (?, ?, ?, 0, ?, 4)",
        (chunk_id, turn_id, text, len(text)),
    )
    database.commit()
    span = EvidenceSpan(
        chunk_id=chunk_id,
        start_char=0,
        end_char=len(text),
        quote_sha256=quote_sha256(text),
        ordinal=ordinal,
        source_id=source_id,
        turn_id=turn_id,
        role="user",
        created_at=created_at,
    )
    episode = Episode(
        episode_id=f"episode-{ordinal}",
        artifact_id=artifact.artifact_id,
        source_id=source_id,
        sequence_no=0,
        first_ordinal=ordinal,
        last_ordinal=ordinal,
        evidence=(span,),
        boundary_method="fixed_interval",
    )
    DiscourseStore(database).publish(artifact, episodes=(episode,))
    return episode


def _artifact(kind: str, marker: str) -> DiscourseArtifact:
    return DiscourseArtifact.create(
        kind=kind,
        implementation_sha256=marker * 64,
        policy={"marker": marker},
    )


def test_episode_auto_resolution_is_namespace_local_and_receipt_bound(tmp_path) -> None:
    with Database(tmp_path / "auto.db") as database:
        artifact = _artifact("longmemeval-diffuse-fixed_interval", "a")
        _publish_episode(database, artifact=artifact, ordinal=1)

        store, policy, binding = _resolve_episode_artifact(
            database,
            _args(auto=True),
        )

        assert type(store) is DiscourseStore
        assert policy is not None and policy.artifact_id == artifact.artifact_id
        assert binding["artifact_id"] == artifact.artifact_id
        assert binding["artifact_receipt_sha256"] == identity_sha256(
            artifact.identity_payload()
        )
        assert binding["episode_count"] == 1
        assert binding["resolution_mode"] == (
            "authenticated_namespace_fixed_interval_auto"
        )


def test_episode_auto_resolution_fails_closed_on_zero_or_ambiguity(tmp_path) -> None:
    with Database(tmp_path / "zero.db") as database:
        with pytest.raises(
            ReducedSourceGroupReinjectionAssayError,
            match="exactly one populated fixed-interval artifact",
        ):
            _resolve_episode_artifact(database, _args(auto=True))

    with Database(tmp_path / "ambiguous.db") as database:
        first = _artifact("fixed_interval", "b")
        second = _artifact("longmemeval-diffuse-fixed_interval", "c")
        _publish_episode(database, artifact=first, ordinal=1)
        _publish_episode(database, artifact=second, ordinal=2)
        with pytest.raises(
            ReducedSourceGroupReinjectionAssayError,
            match="exactly one populated fixed-interval artifact",
        ):
            _resolve_episode_artifact(database, _args(auto=True))


def test_explicit_episode_artifact_is_preserved_for_control_fixtures(tmp_path) -> None:
    with Database(tmp_path / "explicit.db") as database:
        artifact = _artifact("fixture-boundaries", "d")
        _publish_episode(database, artifact=artifact, ordinal=1)

        _store, policy, binding = _resolve_episode_artifact(
            database,
            _args(explicit=artifact.artifact_id),
        )

        assert policy is not None and policy.artifact_id == artifact.artifact_id
        assert binding["artifact_kind"] == "fixture-boundaries"
        assert binding["resolution_mode"] == "explicit_artifact_id"
