from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from memory_condense.application.discourse_sources import scan_discourse_source_chunks
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import Episode
from memory_condense.domain.schemas import Chunk
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.search.episodes.retrieval import EpisodeRetrievalPolicy
from memory_condense.search.indexes.lexical import LexicalIndex
from tools.matched_eval.contracts import canonical_json_bytes, identity_sha256
from tools.matched_eval.full_store_slot_closure import (
    LocalCitationBinding,
    build_full_store_window_index,
)
from tools.matched_eval.query_expansion import FrozenSourceNamespace
from tools.matched_eval.query_guided_scan import cache_namespace_partitions
from tools.matched_eval.semantic_residual_search import (
    SemanticResidualPolicy,
    build_semantic_residual_index,
    compile_semantic_residual_query,
    semantic_residual_source_group_map,
)
from tools.matched_eval.source_group_reinjection import (
    SourceGroupReinjectionError,
    SourceGroupReinjectionPolicy,
    authenticate_source_group_selection,
    replay_source_group_reinjection,
    search_source_group_reinjection,
    validate_source_group_reinjection,
    validate_source_group_selection,
)


BASE = datetime(2026, 8, 1, tzinfo=timezone.utc)


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _build(
    tmp_path: Path,
    name: str,
    rows: list[tuple[str, str, datetime, str]],
    *,
    max_cell_tokens: int = 512,
):
    path = tmp_path / f"{name}.db"
    database = Database(path)
    transcript = TranscriptStore(database)
    lexical = LexicalIndex(database)
    for index, (source_id, text, created_at, role) in enumerate(rows):
        turn = transcript.append(
            role,
            text,
            source_id=source_id,
            created_at=created_at,
        )
        lexical.add_chunks(
            [
                Chunk(
                    chunk_id=f"chunk-{index}",
                    turn_id=turn.turn_id,
                    text=text,
                    start_char=0,
                    end_char=len(text),
                    token_count=count_tokens(text),
                )
            ]
        )
    streams = scan_discourse_source_chunks(database)
    database.close()
    store_receipt = _sha(f"store:{name}")
    namespace = FrozenSourceNamespace.from_source_streams(
        snapshot_id=_sha(f"snapshot:{name}"),
        combined_store_receipt_sha256=store_receipt,
        source_streams=streams,
    )
    with Database(path, read_only=True) as readonly:
        cache = cache_namespace_partitions(
            readonly,
            namespace,
            source_database_sha256=_sha(f"database:{name}"),
            source_store_receipt_sha256=store_receipt,
        )
    window_index = build_full_store_window_index(cache)
    vectors = {row.source_id: None for row in window_index.rows}
    return build_semantic_residual_index(
        window_index,
        vectors,
        policy=SemanticResidualPolicy(
            max_cell_tokens=max_cell_tokens,
            payload_token_cap=2_000,
        ),
    )


def _segment(index, quote: str):
    matches = [
        (cell, segment)
        for cell in index.cells
        for segment in cell.segments
        if segment.quote == quote
    ]
    assert len(matches) == 1
    return matches[0]


def _binding(index, quote: str) -> LocalCitationBinding:
    cell, segment = _segment(index, quote)
    universe = tuple(sorted({row.source_id for row in index.cells}))
    groups = semantic_residual_source_group_map(universe)
    return LocalCitationBinding(
        candidate_id=_sha(f"candidate:{quote}"),
        source_group_handle=groups[cell.source_id],
        namespace_id=index.namespace_id,
        cache_receipt_sha256=index.cache_receipt_sha256,
        source_database_sha256=index.source_database_sha256,
        source_store_receipt_sha256=index.source_store_receipt_sha256,
        source_id=segment.source_id,
        partition_id=segment.partition_id,
        span=segment.span,
        quote_sha256=segment.quote_sha256,
    )


def _selection(index, handles: dict[str, LocalCitationBinding]):
    universe = tuple(sorted({row.source_id for row in index.cells}))
    return authenticate_source_group_selection(
        index,
        handles,
        group_universe_source_ids=universe,
    )


def _query(index, body: str):
    return compile_semantic_residual_query(
        index,
        f"[Question asked at 2026/08/29 12:00] {body}",
    )


def test_user_fact_outranks_same_source_generic_assistant_text(tmp_path: Path) -> None:
    generic = (
        "You could consider jewelry that you acquired, including rings, "
        "necklaces, or bracelets."
    )
    fact = "I acquired a sapphire ring at the market yesterday."
    index = _build(
        tmp_path,
        "generic-vs-fact",
        [
            ("history-a", generic, BASE, "assistant"),
            ("history-a", fact, BASE + timedelta(days=1), "user"),
        ],
    )
    anchor = _binding(index, generic)
    selection = _selection(index, {"R0001": anchor})
    query = _query(index, "What jewelry did I acquire?")

    result = search_source_group_reinjection(
        index,
        query,
        selection,
        policy=SourceGroupReinjectionPolicy(
            base_segments_per_group=1,
            source_neighbor_radius=0,
            max_source_neighbors_per_anchor=0,
        ),
    )

    assert any(row.quote == fact for row in result.evidence)
    fact_attempt = next(
        row for row in result.attempted_selection if row.segment_receipt_sha256 == _segment(index, fact)[1].receipt_sha256
    )
    generic_attempt = next(
        row for row in result.attempted_selection if row.segment_receipt_sha256 == _segment(index, generic)[1].receipt_sha256
    )
    assert dict(fact_attempt.score_components)["factual_assertion"] == 1.0
    assert dict(generic_attempt.score_components)["generic_advice_penalty"] == -1.0
    assert generic_attempt.disposition == "protected_exact_duplicate"


def test_declarative_fact_before_question_keeps_factual_priority(
    tmp_path: Path,
) -> None:
    generic = (
        "You could consider a Garmin bike computer or another fitness tracker "
        "for general cycling metrics."
    )
    mixed = (
        "I bought a new Garmin bike computer yesterday. "
        "Can you tell me how to configure it?"
    )
    index = _build(
        tmp_path,
        "mixed-fact-question",
        [
            ("history-a", generic, BASE, "assistant"),
            ("history-a", mixed, BASE + timedelta(days=1), "user"),
        ],
    )
    selection = _selection(index, {"R0001": _binding(index, generic)})
    result = search_source_group_reinjection(
        index,
        _query(index, "Which Garmin bike computer did I buy yesterday?"),
        selection,
        policy=SourceGroupReinjectionPolicy(
            base_segments_per_group=2,
            source_neighbor_radius=0,
            max_source_neighbors_per_anchor=0,
        ),
    )

    mixed_attempt = next(
        row
        for row in result.attempted_selection
        if row.segment_receipt_sha256 == _segment(index, mixed)[1].receipt_sha256
    )
    generic_attempt = next(
        row
        for row in result.attempted_selection
        if row.segment_receipt_sha256 == _segment(index, generic)[1].receipt_sha256
    )
    assert dict(mixed_attempt.score_components)["factual_assertion"] == 1.0
    assert dict(generic_attempt.score_components)["factual_assertion"] == 0.0
    assert mixed_attempt.disposition == "packed_novel"
    assert generic_attempt.disposition == "protected_exact_duplicate"
    assert any(row.quote == mixed for row in result.evidence)


def test_source_local_neighbor_expansion_is_bounded_and_receipt_bound(
    tmp_path: Path,
) -> None:
    fact = "I visited the Natural History Museum with Priya in February."
    anchor_text = "You could write a short museum trip recap."
    far = "I later reorganized the pantry shelves."
    index = _build(
        tmp_path,
        "source-neighbor",
        [
            ("history-a", fact, BASE, "user"),
            ("history-a", anchor_text, BASE + timedelta(hours=1), "assistant"),
            ("history-a", far, BASE + timedelta(hours=2), "user"),
        ],
    )
    anchor = _binding(index, anchor_text)
    selection = _selection(index, {"P0001": anchor})
    result = search_source_group_reinjection(
        index,
        _query(index, "Which museum did I visit in February?"),
        selection,
        policy=SourceGroupReinjectionPolicy(
            base_segments_per_group=1,
            source_neighbor_radius=1,
            max_source_neighbors_per_anchor=1,
        ),
    )

    fact_attempt = next(
        row for row in result.attempted_selection if row.segment_receipt_sha256 == _segment(index, fact)[1].receipt_sha256
    )
    assert any(route.startswith("source_local_previous:distance=1") for route in fact_attempt.selection_routes)
    assert not any(
        route.startswith("source_local_next")
        for row in result.attempted_selection
        for route in row.selection_routes
    )
    selected_handle = selection.selected_handles[0]
    assert selected_handle.anchor_cell_receipt_sha256 == _segment(index, anchor_text)[0].receipt_sha256
    assert selected_handle.anchor_segment_receipt_sha256 == _segment(index, anchor_text)[1].receipt_sha256


class _EpisodeStore:
    def __init__(self, episodes: tuple[Episode, ...]):
        self.by_id = {row.episode_id: row for row in episodes}
        self.by_chunk = {
            span.chunk_id: row.episode_id
            for row in episodes
            for span in row.evidence
        }

    def episode_ids_for_chunks(self, chunk_ids, *, artifact_id=None):
        return {
            chunk_id: self.by_chunk[chunk_id]
            for chunk_id in chunk_ids
            if chunk_id in self.by_chunk
            and (
                artifact_id is None
                or self.by_id[self.by_chunk[chunk_id]].artifact_id == artifact_id
            )
        }

    def get_episode(self, episode_id):
        return self.by_id.get(episode_id)

    def adjacent_episodes(self, episode_id, *, radius=1, include_self=False):
        anchor = self.by_id[episode_id]
        return tuple(
            row
            for row in self.by_id.values()
            if row.source_id == anchor.source_id
            and abs(row.sequence_no - anchor.sequence_no) <= radius
            and (include_self or row.episode_id != episode_id)
        )

    def episodes_for_source(self, artifact_id, source_id, *, limit=None):
        rows = tuple(
            sorted(
                (
                    row
                    for row in self.by_id.values()
                    if row.artifact_id == artifact_id and row.source_id == source_id
                ),
                key=lambda row: (row.sequence_no, row.episode_id),
            )
        )
        return rows if limit is None else rows[:limit]


def test_existing_episode_primitive_reinjects_exact_neighbor_spans(tmp_path: Path) -> None:
    anchor_text = "I started planning the weekend outing."
    bridge = "We discussed a few unrelated logistics."
    fact = "I visited the Art Cube gallery with Mina."
    index = _build(
        tmp_path,
        "episode-neighbor",
        [
            ("history-a", anchor_text, BASE, "user"),
            ("history-a", bridge, BASE + timedelta(hours=1), "assistant"),
            ("history-a", fact, BASE + timedelta(hours=2), "user"),
        ],
    )
    anchor_segment = _segment(index, anchor_text)[1]
    fact_segment = _segment(index, fact)[1]
    artifact_id = "episodes-v1"
    episodes = (
        Episode(
            episode_id="episode-0",
            artifact_id=artifact_id,
            source_id="history-a",
            sequence_no=0,
            first_ordinal=anchor_segment.span.ordinal,
            last_ordinal=anchor_segment.span.ordinal,
            evidence=(anchor_segment.span,),
            boundary_method="fixture",
        ),
        Episode(
            episode_id="episode-1",
            artifact_id=artifact_id,
            source_id="history-a",
            sequence_no=1,
            first_ordinal=fact_segment.span.ordinal,
            last_ordinal=fact_segment.span.ordinal,
            evidence=(fact_segment.span,),
            boundary_method="fixture",
        ),
    )
    store = _EpisodeStore(episodes)
    selection = _selection(index, {"R0001": _binding(index, anchor_text)})
    result = search_source_group_reinjection(
        index,
        _query(index, "Which gallery did I visit?"),
        selection,
        policy=SourceGroupReinjectionPolicy(
            base_segments_per_group=1,
            source_neighbor_radius=0,
            max_source_neighbors_per_anchor=0,
        ),
        episode_lookup=store,
        episode_policy=EpisodeRetrievalPolicy(
            artifact_id=artifact_id,
            max_anchor_episodes=1,
            previous_episodes=0,
            next_episodes=1,
            max_episode_seeds=2,
            max_direct_fallbacks=1,
        ),
    )

    fact_attempt = next(
        row for row in result.attempted_selection if row.segment_receipt_sha256 == fact_segment.receipt_sha256
    )
    assert any(route.startswith("episode_next:episode=episode-1") for route in fact_attempt.selection_routes)
    assert result.episode_plan is not None
    assert result.episode_population_receipt_sha256 is not None
    assert any(row.quote == fact for row in result.evidence)


def test_noisy_source_lane_cannot_starve_direct_episode_fact(tmp_path: Path) -> None:
    noisy_anchor = "Let's review the Sunday cycling performance archive."
    noisy = [
        (
            f"I measured Sunday cycling performance value {index} after "
            f"adjusting ordinary training setting {index}."
        )
        for index in range(8)
    ]
    episode_anchor = "Let's revisit the equipment used for the Sunday ride."
    episode_generic = (
        "During a Sunday bike ride, improving performance metrics can require "
        "general calibration advice and several ordinary equipment checks."
    )
    target = (
        "I installed a new Garmin bike computer yesterday, which improved "
        "my Sunday ride metrics. Can you help me calibrate it?"
    )
    index = _build(
        tmp_path,
        "stratified-direct-episode",
        [
            ("history-noisy", noisy_anchor, BASE, "assistant"),
            *(
                (
                    "history-noisy",
                    text,
                    BASE + timedelta(minutes=position + 1),
                    "user",
                )
                for position, text in enumerate(noisy)
            ),
            ("history-target", episode_anchor, BASE, "assistant"),
            (
                "history-target",
                episode_generic,
                BASE + timedelta(minutes=1),
                "assistant",
            ),
            ("history-target", target, BASE + timedelta(minutes=2), "user"),
        ],
    )
    anchor_segment = _segment(index, episode_anchor)[1]
    generic_segment = _segment(index, episode_generic)[1]
    target_segment = _segment(index, target)[1]
    artifact_id = "stratified-episodes-v1"
    store = _EpisodeStore(
        (
            Episode(
                episode_id="episode-target",
                artifact_id=artifact_id,
                source_id="history-target",
                    sequence_no=0,
                    first_ordinal=anchor_segment.span.ordinal,
                    last_ordinal=target_segment.span.ordinal,
                    evidence=(
                        anchor_segment.span,
                        generic_segment.span,
                        target_segment.span,
                    ),
                boundary_method="fixture",
            ),
        )
    )
    selection = _selection(
        index,
        {
            "R0001": _binding(index, noisy_anchor),
            "R0002": _binding(index, episode_anchor),
        },
    )
    query = _query(index, "Why did my Sunday ride metrics improve?")
    policy = SourceGroupReinjectionPolicy(
        local_payload_token_cap=180,
        max_selected_segments=3,
        base_segments_per_group=3,
        source_neighbor_radius=1,
        max_source_neighbors_per_anchor=1,
        max_episode_segments_per_seed=1,
    )
    episode_policy = EpisodeRetrievalPolicy(
        artifact_id=artifact_id,
        max_anchor_episodes=1,
        previous_episodes=0,
        next_episodes=0,
        max_episode_seeds=1,
        max_direct_fallbacks=1,
    )

    result = search_source_group_reinjection(
        index,
        query,
        selection,
        policy=policy,
        episode_lookup=store,
        episode_policy=episode_policy,
    )
    replayed = replay_source_group_reinjection(
        index,
        query,
        selection,
        result,
        episode_lookup=store,
        episode_policy=episode_policy,
    )

    target_attempt = next(
        row
        for row in result.attempted_selection
        if row.segment_receipt_sha256 == target_segment.receipt_sha256
    )
    assert target_attempt.selection_rank == 2
    assert target_attempt.disposition == "packed_novel"
    assert any(
        route.startswith("episode_direct:episode=episode-target")
        for route in target_attempt.selection_routes
    )
    assert any(row.quote == target for row in result.evidence)
    assert result.frontier.selection_truncated is True
    assert result.packed_local_evidence_tokens <= policy.local_payload_token_cap
    assert result.projection()["new_provider_calls"] == 0
    assert result.projection()["retained_transformer_token_state_bytes"] == 0
    assert replayed.receipt_sha256 == result.receipt_sha256
    assert canonical_json_bytes(replayed.projection()) == canonical_json_bytes(
        result.projection()
    )


def test_stable_ties_and_replay_are_byte_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor_text = "Let's review the two vase purchases."
    rows = [
        ("history-a", anchor_text, BASE, "assistant"),
        ("history-a", "I bought an amber vase.", BASE, "user"),
        ("history-a", "I bought a cobalt vase.", BASE, "user"),
    ]
    index = _build(tmp_path, "stable-ties", rows)
    selection = _selection(index, {"R0001": _binding(index, anchor_text)})
    query = _query(index, "Which vases did I buy?")
    policy = SourceGroupReinjectionPolicy(
        base_segments_per_group=3,
        source_neighbor_radius=0,
        max_source_neighbors_per_anchor=0,
    )
    first = search_source_group_reinjection(index, query, selection, policy=policy)
    second = search_source_group_reinjection(index, query, selection, policy=policy)
    validate_source_group_reinjection(index, query, selection, first)

    search_calls = 0

    def counted_search(*args, **kwargs):
        nonlocal search_calls
        search_calls += 1
        return search_source_group_reinjection(*args, **kwargs)

    monkeypatch.setattr(
        "tools.matched_eval.source_group_reinjection.search_source_group_reinjection",
        counted_search,
    )
    replayed = replay_source_group_reinjection(index, query, selection, first)

    assert search_calls == 1
    assert replayed is not first
    assert first.receipt_sha256 == second.receipt_sha256 == replayed.receipt_sha256
    assert canonical_json_bytes(first.projection()) == canonical_json_bytes(
        replayed.projection()
    )
    assert tuple(row.segment_receipt_sha256 for row in first.attempted_selection) == tuple(
        row.segment_receipt_sha256 for row in second.attempted_selection
    )


def test_post_selection_dedup_reinjects_exact_visible_owner(tmp_path: Path) -> None:
    anchor_text = "Let's review your appliance notes."
    fact = "I bought a pellet smoker ten days ago."
    index = _build(
        tmp_path,
        "owner-closure",
        [
            ("history-a", anchor_text, BASE, "assistant"),
            ("history-a", fact, BASE + timedelta(days=1), "user"),
        ],
    )
    anchor = _binding(index, anchor_text)
    owner = _binding(index, fact)
    selection = _selection(index, {"R0001": anchor})
    protected = {"R0001": anchor, "P0007": owner}
    result = search_source_group_reinjection(
        index,
        _query(index, "Which appliance did I buy ten days ago?"),
        selection,
        protected_handle_bindings=protected,
    )

    duplicate = next(
        row for row in result.protected_duplicates if row.segment_receipt_sha256 == _segment(index, fact)[1].receipt_sha256
    )
    assert duplicate.protected_evidence_handle == "P0007"
    assert duplicate.protected_binding_receipt_sha256 == owner.receipt_sha256
    assert all(row.quote != fact for row in result.evidence)
    attempt = next(
        row for row in result.attempted_selection if row.segment_receipt_sha256 == duplicate.segment_receipt_sha256
    )
    assert attempt.disposition == "protected_exact_duplicate"
    assert result.projection()["selected_before_protected_dedup"] is True


def test_terminal_group_rebase_preserves_original_binding_receipt(
    tmp_path: Path,
) -> None:
    fact = "I stored the amber compass in the upper cabinet."
    index = _build(
        tmp_path,
        "terminal-group-rebase",
        [
            ("history-a", fact, BASE, "user"),
            ("history-b", "The garden gate was inspected.", BASE, "assistant"),
        ],
    )
    original = _binding(index, fact)
    prior_plane = replace(
        original,
        source_group_handle="G9999",
        receipt_sha256="",
    )
    terminal_group = semantic_residual_source_group_map(
        tuple(sorted({row.source_id for row in index.cells}))
    )[original.source_id]

    with pytest.raises(
        SourceGroupReinjectionError,
        match="selected handle G mapping",
    ):
        authenticate_source_group_selection(
            index,
            {"P0001": prior_plane},
            group_universe_source_ids=tuple(
                sorted({row.source_id for row in index.cells})
            ),
        )

    selection = authenticate_source_group_selection(
        index,
        {"P0001": prior_plane},
        group_universe_source_ids=tuple(
            sorted({row.source_id for row in index.cells})
        ),
        selected_handle_groups={"P0001": terminal_group},
    )
    selected = selection.selected_handles[0]
    assert selected.source_group_handle == terminal_group
    assert selected.local_binding.receipt_sha256 == prior_plane.receipt_sha256
    assert selected.local_binding.source_group_handle == "G9999"
    validate_source_group_selection(index, selection)


def test_nonborrowable_cap_skips_oversized_row_and_continues(tmp_path: Path) -> None:
    anchor_text = "Let's review the access notes."
    huge = (
        "I recorded the heliotrope access code LIME-42 with the exact door detail. "
        + " ".join(f"heliotrope-detail-{index}" for index in range(260))
    )
    short = "The backup code was BLUE-7."
    index = _build(
        tmp_path,
        "skip-continue",
        [
            ("history-a", anchor_text, BASE, "assistant"),
            ("history-a", huge, BASE + timedelta(hours=1), "user"),
            ("history-a", short, BASE + timedelta(hours=2), "user"),
        ],
        max_cell_tokens=2_000,
    )
    selection = _selection(index, {"R0001": _binding(index, anchor_text)})
    cap = 105
    result = search_source_group_reinjection(
        index,
        _query(index, "What heliotrope access code did I record?"),
        selection,
        policy=SourceGroupReinjectionPolicy(
            local_payload_token_cap=cap,
            base_segments_per_group=3,
            source_neighbor_radius=0,
            max_source_neighbors_per_anchor=0,
        ),
    )

    huge_attempt = next(
        row for row in result.attempted_selection if row.segment_receipt_sha256 == _segment(index, huge)[1].receipt_sha256
    )
    short_attempt = next(
        row for row in result.attempted_selection if row.segment_receipt_sha256 == _segment(index, short)[1].receipt_sha256
    )
    assert huge_attempt.disposition == "budget_unpacked"
    assert short_attempt.disposition == "packed_novel"
    assert any(row.quote == short for row in result.evidence)
    assert result.packed_local_evidence_tokens <= cap
    assert result.attempted_local_evidence_tokens > cap
    assert result.frontier.packing_closed is False
    assert result.frontier.needs_global_search is True


def test_mutation_rejection_and_frozen_contract(tmp_path: Path) -> None:
    anchor_text = "I bought a cedar desk."
    index = _build(
        tmp_path,
        "mutation",
        [("history-a", anchor_text, BASE, "user")],
    )
    selection = _selection(index, {"R0001": _binding(index, anchor_text)})
    query = _query(index, "Which desk did I buy?")
    result = search_source_group_reinjection(index, query, selection)

    with pytest.raises(SourceGroupReinjectionError, match="result receipt changed"):
        replace(result, receipt_sha256="0" * 64)
    with pytest.raises(SourceGroupReinjectionError):
        replace(
            selection.group_rows[0],
            source_group_handle="G999999",
        )
    with pytest.raises(FrozenInstanceError):
        result.provider_payload_tokens = 0  # type: ignore[misc]
    validate_source_group_selection(index, selection)
