from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from memory_condense.domain.discourse import (
    DiscourseArtifact,
    EvidenceSpan,
    quote_sha256,
)
from memory_condense.persistence.db import Database
from memory_condense.persistence.discourse_store import DiscourseStore
from memory_condense.search.episodes import (
    UserLedEpisodeBuilder,
    build_user_led_episodes,
)


ARTIFACT = "disc-user-led-test"
SOURCE = "conversation-a"


def _span(
    ordinal: int,
    role: str | None,
    *,
    part: int = 0,
    source_id: str = SOURCE,
    turn_id: str | None = None,
) -> EvidenceSpan:
    text = f"{ordinal}:{role}:{part}:{source_id}"
    return EvidenceSpan(
        chunk_id=f"chunk-{source_id}-{ordinal}-{part}",
        start_char=0,
        end_char=len(text),
        quote_sha256=quote_sha256(text),
        ordinal=ordinal,
        source_id=source_id,
        turn_start_char=part * 100,
        turn_id=turn_id if turn_id is not None else f"turn-{ordinal}",
        role=role,
        created_at="2026-09-07T00:00:00+00:00",
    )


def _owned(result) -> tuple[EvidenceSpan, ...]:
    return tuple(span for shard in result.shards for span in shard.owned_evidence)


def test_user_turns_lead_exact_exchanges_and_machine_roles_attach() -> None:
    prelude = _span(0, "system")
    lead_a = (_span(1, "user", part=0), _span(1, "user", part=1))
    answer = _span(2, "assistant")
    tool_like_system = _span(3, "system")
    lead_b = _span(4, "user")
    answer_b = _span(5, "assistant")
    spans = (prelude, *lead_a, answer, tool_like_system, lead_b, answer_b)

    result = build_user_led_episodes(
        source_id=SOURCE,
        artifact_id=ARTIFACT,
        spans=spans,
        sequence_start=7,
    )

    assert _owned(result) == spans
    assert [shard.exchange_kind for shard in result.shards] == [
        "orphan_prelude",
        "user_led",
        "user_led",
    ]
    assert result.shards[0].episode.evidence == (prelude,)
    assert result.shards[1].lead_evidence == lead_a
    assert result.shards[1].episode.evidence == (
        *lead_a,
        answer,
        tool_like_system,
    )
    assert result.shards[2].episode.evidence == (lead_b, answer_b)
    assert [episode.sequence_no for episode in result.episodes] == [7, 8, 9]
    assert len(result.exchange_ids) == 3


def test_build_is_deterministic_and_receipts_reject_tampering() -> None:
    spans = (_span(0, "user"), _span(1, "assistant"))
    builder = UserLedEpisodeBuilder()

    first = builder.build(source_id=SOURCE, artifact_id=ARTIFACT, spans=spans)
    replay = builder.build(source_id=SOURCE, artifact_id=ARTIFACT, spans=spans)

    assert first == replay
    assert first.receipt_sha256 == replay.receipt_sha256
    assert first.episodes[0].receipt_sha256 == replay.episodes[0].receipt_sha256
    with pytest.raises(ValueError, match="stable lead anchor"):
        replace(first.shards[0], exchange_id="exchange-tampered")
    with pytest.raises(FrozenInstanceError):
        first.shards[0].shard_index = 9  # type: ignore[misc]


def test_consecutive_user_turns_create_distinct_microepisodes() -> None:
    first_user = _span(0, "user")
    second_user = _span(1, "user")
    answer = _span(2, "assistant")

    result = build_user_led_episodes(
        source_id=SOURCE,
        artifact_id=ARTIFACT,
        spans=(first_user, second_user, answer),
    )

    assert len(result.shards) == 2
    assert result.shards[0].episode.evidence == (first_user,)
    assert result.shards[1].episode.evidence == (second_user, answer)
    assert result.shards[0].exchange_id != result.shards[1].exchange_id


def test_orphan_prelude_never_attaches_forward_to_first_user() -> None:
    orphan_assistant = _span(0, "assistant")
    orphan_system = _span(1, "system")
    user = _span(2, "user")

    result = UserLedEpisodeBuilder().build(
        source_id=SOURCE,
        artifact_id=ARTIFACT,
        spans=(orphan_assistant, orphan_system, user),
    )

    prelude, user_exchange = result.shards
    assert prelude.exchange_kind == "orphan_prelude"
    assert prelude.episode.boundary_method == "orphan_prelude"
    assert prelude.episode.evidence == (orphan_assistant, orphan_system)
    assert user_exchange.episode.evidence == (user,)


def test_long_exchange_shards_with_one_stable_id_and_retained_whole_lead() -> None:
    lead = (_span(0, "user", part=0), _span(0, "user", part=1))
    attached = tuple(
        _span(index, "assistant" if index % 2 else "system")
        for index in range(1, 6)
    )
    spans = (*lead, *attached)

    result = UserLedEpisodeBuilder(max_spans_per_shard=4).build(
        source_id=SOURCE,
        artifact_id=ARTIFACT,
        spans=spans,
    )

    assert len(result.shards) == 3
    assert len({shard.exchange_id for shard in result.shards}) == 1
    assert [shard.shard_index for shard in result.shards] == [0, 1, 2]
    assert all(shard.shard_count == 3 for shard in result.shards)
    assert result.shards[0].owned_evidence == (*lead, *attached[:2])
    assert result.shards[0].retained_lead_evidence == ()
    assert all(
        shard.retained_lead_evidence == lead for shard in result.shards[1:]
    )
    assert all(shard.retrieval_evidence[:2] == lead for shard in result.shards)
    assert all(len(shard.retrieval_evidence) <= 4 for shard in result.shards)
    assert result.shards[1].episode.evidence == attached[2:4]
    assert result.shards[2].episode.evidence == attached[4:]
    # Ownership, unlike retained context, is a strict exactly-once partition.
    assert _owned(result) == spans
    assert len({_item.quote_sha256 for _item in _owned(result)}) == len(spans)


def test_prelude_sharding_is_explicit_and_partitions_without_a_fake_lead() -> None:
    spans = tuple(_span(index, "system") for index in range(5))

    result = UserLedEpisodeBuilder(max_spans_per_shard=2).build(
        source_id=SOURCE,
        artifact_id=ARTIFACT,
        spans=spans,
    )

    assert len(result.shards) == 3
    assert len({shard.exchange_id for shard in result.shards}) == 1
    assert all(shard.exchange_kind == "orphan_prelude" for shard in result.shards)
    assert all(not shard.lead_evidence for shard in result.shards)
    assert _owned(result) == spans


def test_sharded_owned_episodes_persist_in_monotonic_source_order(tmp_path) -> None:
    spans = (
        _span(0, "user"),
        _span(1, "assistant"),
        _span(2, "system"),
        _span(3, "assistant"),
    )
    result = UserLedEpisodeBuilder(max_spans_per_shard=2).build(
        source_id=SOURCE,
        artifact_id=ARTIFACT,
        spans=spans,
    )
    assert [episode.evidence for episode in result.episodes] == [
        spans[:2],
        spans[2:3],
        spans[3:],
    ]
    assert result.shards[1].retrieval_evidence == (spans[0], spans[2])

    db = Database(tmp_path / "user-led.db")
    try:
        for span in spans:
            text = f"{span.ordinal}:{span.role}:0:{SOURCE}"
            db.execute(
                "INSERT INTO turns "
                "(turn_id, role, text, source_id, created_at, ordinal) "
                "VALUES (?, ?, ?, ?, '2026-09-07T00:00:00+00:00', ?)",
                (span.turn_id, span.role, text, SOURCE, span.ordinal),
            )
            db.execute(
                "INSERT INTO chunks "
                "(chunk_id, turn_id, text, start_char, end_char, token_count) "
                "VALUES (?, ?, ?, 0, ?, 1)",
                (span.chunk_id, span.turn_id, text, len(text)),
            )
        db.commit()
        store = DiscourseStore(db)
        artifact = DiscourseArtifact(
            artifact_id=ARTIFACT,
            kind="user-led-episodes",
            implementation_sha256="a" * 64,
            policy_sha256="b" * 64,
        )

        store.publish(artifact, episodes=result.episodes)

        assert store.episodes_for_source(ARTIFACT, SOURCE) == result.episodes
        assert len(
            store.episode_ids_for_chunks(
                tuple(span.chunk_id for span in spans),
                artifact_id=ARTIFACT,
            )
        ) == len(spans)
    finally:
        db.close()


@pytest.mark.parametrize(
    ("spans", "message"),
    [
        (
            (_span(0, "user"), _span(1, "assistant", source_id="other")),
            "explicit source_id",
        ),
        (
            (_span(1, "assistant"), _span(0, "user")),
            "source order",
        ),
        (
            (_span(0, None),),
            "explicit user, assistant, or system role",
        ),
    ],
)
def test_source_order_and_explicit_role_are_required(spans, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        UserLedEpisodeBuilder().build(
            source_id=SOURCE,
            artifact_id=ARTIFACT,
            spans=spans,
        )


def test_turn_metadata_conflicts_are_rejected() -> None:
    mixed_roles = (
        _span(0, "user", part=0),
        _span(0, "assistant", part=1),
    )
    with pytest.raises(ValueError, match="share one role"):
        UserLedEpisodeBuilder().build(
            source_id=SOURCE,
            artifact_id=ARTIFACT,
            spans=mixed_roles,
        )

    reused_turn_id = (
        _span(0, "user", turn_id="reused"),
        _span(1, "assistant", turn_id="reused"),
    )
    with pytest.raises(ValueError, match="multiple ordinals"):
        UserLedEpisodeBuilder().build(
            source_id=SOURCE,
            artifact_id=ARTIFACT,
            spans=reused_turn_id,
        )


def test_shard_bound_never_splits_or_crowds_out_the_user_lead() -> None:
    lead = (_span(0, "user", part=0), _span(0, "user", part=1))
    with pytest.raises(ValueError, match="cannot split"):
        UserLedEpisodeBuilder(max_spans_per_shard=1).build(
            source_id=SOURCE,
            artifact_id=ARTIFACT,
            spans=lead,
        )
    with pytest.raises(ValueError, match="no room"):
        UserLedEpisodeBuilder(max_spans_per_shard=2).build(
            source_id=SOURCE,
            artifact_id=ARTIFACT,
            spans=(*lead, _span(1, "assistant")),
        )


def test_empty_input_has_a_deterministic_empty_receipt() -> None:
    first = build_user_led_episodes(
        source_id=SOURCE,
        artifact_id=ARTIFACT,
        spans=(),
    )
    replay = build_user_led_episodes(
        source_id=SOURCE,
        artifact_id=ARTIFACT,
        spans=(),
    )
    assert first == replay
    assert first.shards == ()
    assert first.episodes == ()
