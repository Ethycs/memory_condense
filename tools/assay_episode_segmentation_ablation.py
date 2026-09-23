#!/usr/bin/env python3
"""Small matched, provider-free assay for conversational episode boundaries."""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import (
    Episode,
    EvidenceSpan,
    identity_sha256,
    quote_sha256,
)
from memory_condense.search.episodes import EpisodeBuilder
from memory_condense.search.episodes.boundaries import (
    AdaptiveBoundaryDetector,
    CohesionBoundaryRefiner,
)
from memory_condense.search.episodes.representatives import (
    select_episode_representatives,
)
from memory_condense.search.episodes.surprise import (
    LexicalEmbeddingChangeScorer,
    lexical_cosine,
    score_surprise_sequence,
)
from memory_condense.search.indexes.lexical import BM25_B, BM25_K1, tokenize
from tools.matched_eval.artifacts import (
    publish_sealed_json,
    read_sealed_json,
)


FORMAT = "memory-condense-episode-segmentation-ablation-v2"
SELECTION_NAME = "selection.json"
EVALUATION_NAME = "evaluation.json"
RUNTIME_NAME = "runtime.json"
REPLAY_NAME = "replay.json"
OVERLAY_NAME = "macro-overlay.json"
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/episode-segmentation-ablation-v2-20260907"
)
ARMS = ("surprise", "user_micro", "hybrid_overlay")
REPRESENTATIVE_LIMIT = 2
MAX_ANCHORS = 2
NEIGHBOR_RADIUS = 1
MAX_EPISODES = 4
MAX_CHUNKS = 12
MAX_PROMPT_TOKENS = 512
WARM_REPETITIONS = 25
SCALAR_MICRO_CAP = 2
FRONTIER_MICRO_CAP = 6


@dataclass(frozen=True, slots=True)
class AssayChunk:
    chunk_id: str
    source_id: str
    turn_id: str
    ordinal: int
    role: str
    text: str

    def span(self) -> EvidenceSpan:
        return EvidenceSpan(
            chunk_id=self.chunk_id,
            start_char=0,
            end_char=len(self.text),
            quote_sha256=quote_sha256(self.text),
            ordinal=self.ordinal,
            source_id=self.source_id,
            turn_id=self.turn_id,
            role=self.role,
            created_at=f"2026-01-{self.ordinal + 1:02d}T12:00:00+00:00",
        )


@dataclass(frozen=True, slots=True)
class AssayQuestion:
    question_id: str
    query_type: str
    text: str

    def projection(self) -> dict[str, str]:
        return {
            "question_id": self.question_id,
            "query_type": self.query_type,
            "text": self.text,
            "query_sha256": quote_sha256(self.text),
        }


@dataclass(frozen=True, slots=True)
class AssayGold:
    question_id: str
    target_chunk_ids: tuple[str, ...]
    answer_bearing_chunk_ids: tuple[str, ...]
    accepted_roles: tuple[str, ...]
    required_neighbor_pairs: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ArmIndex:
    arm: str
    episodes: tuple[Episode, ...]
    representatives: Mapping[str, tuple[str, ...]]
    build_ns: int
    representative_build_ns: int
    macro_memberships: tuple[MacroMembership, ...] = ()


@dataclass(frozen=True, slots=True)
class MacroMembership:
    group_id: str
    source_id: str
    member_episode_ids: tuple[str, ...]

    def projection(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "member_episode_ids": list(self.member_episode_ids),
            "source_id": self.source_id,
        }


def _corpus() -> tuple[AssayChunk, ...]:
    rows: tuple[tuple[str, str, str, str], ...] = (
        ("travel", "tr-u-muir", "user", "In March I took a family trip to Muir Woods."),
        ("travel", "tr-a-muir", "assistant", "Muir Woods was the first of those trips."),
        ("travel", "tr-u-big-sur", "user", "In April I took a road trip through Big Sur and Monterey."),
        ("travel", "tr-a-big-sur", "assistant", "Big Sur was the second trip, after Muir Woods."),
        ("travel", "tr-u-yosemite", "user", "In May I went on a camping trip to Yosemite."),
        ("travel", "tr-a-yosemite", "assistant", "Yosemite completed the three-trip sequence."),
        ("travel", "tr-u-pack", "user", "My hiking backpack is bright orange."),
        ("travel", "tr-a-pack", "assistant", "I noted the orange hiking backpack."),
        ("profile", "pf-system", "system", "Maintain exact ownership and correction order."),
        ("profile", "pf-u-mara", "user", "My emergency contact was Mara."),
        ("profile", "pf-a-mara", "assistant", "Mara is listed as your emergency contact."),
        ("profile", "pf-u-niko", "user", "Correction: my emergency contact is now Niko."),
        ("profile", "pf-a-niko", "assistant", "Niko replaces Mara as the emergency contact."),
        ("profile", "pf-u-tea", "user", "For breakfast I prefer jasmine tea."),
        ("profile", "pf-a-tea", "assistant", "Jasmine tea is your breakfast preference."),
        ("profile", "pf-u-amber", "user", "For the Solstice project, the launch codeword was amber."),
        ("profile", "pf-a-amber", "assistant", "The Solstice launch codeword is amber."),
        ("profile", "pf-u-cobalt", "user", "Actually, cobalt replaces amber as the Solstice launch codeword."),
        ("profile", "pf-a-cobalt", "assistant", "The latest Solstice codeword is cobalt."),
        ("support", "sp-u-receipt", "user", "Where did I say I stored the warranty receipt?"),
        ("support", "sp-a-receipt", "assistant", "You stored the warranty receipt in the blue desk drawer."),
        ("support", "sp-u-serial", "user", "Does that receipt include the device serial number?"),
        ("support", "sp-a-serial", "assistant", "Yes, the receipt includes serial number ZX-41."),
        ("support", "sp-u-dana", "user", "My planning meeting with Dana is on Tuesday."),
        ("support", "sp-a-dana", "assistant", "The Dana planning meeting is Tuesday."),
    )
    return tuple(
        AssayChunk(
            chunk_id=turn_id,
            source_id=source_id,
            turn_id=turn_id,
            ordinal=ordinal,
            role=role,
            text=text,
        )
        for ordinal, (source_id, turn_id, role, text) in enumerate(rows)
    )


def _questions() -> tuple[AssayQuestion, ...]:
    return (
        AssayQuestion("q-order", "ordered_multi_event", "What was the order of my Muir Woods, Big Sur, and Yosemite trips?"),
        AssayQuestion("q-contact", "latest_correction", "Who is my current emergency contact?"),
        AssayQuestion("q-code", "topic_return_correction", "What is the latest Solstice launch codeword?"),
        AssayQuestion("q-tea", "direct_user_fact", "What tea do I prefer for breakfast?"),
        AssayQuestion("q-receipt", "assistant_pair_closure", "Where is the warranty receipt stored?"),
        AssayQuestion("q-serial", "followup_pair_closure", "What serial number is on the warranty receipt?"),
        AssayQuestion("q-meeting", "role_owned_schedule", "When is my planning meeting with Dana?"),
        AssayQuestion("q-cross", "cross_source_enumeration", "Which trip places and Dana meeting day did I mention?"),
    )


def _gold() -> tuple[AssayGold, ...]:
    return (
        AssayGold("q-order", ("tr-u-muir", "tr-u-big-sur", "tr-u-yosemite"), ("tr-u-muir", "tr-a-muir", "tr-u-big-sur", "tr-a-big-sur", "tr-u-yosemite", "tr-a-yosemite"), ("user",)),
        AssayGold("q-contact", ("pf-u-niko",), ("pf-u-mara", "pf-a-mara", "pf-u-niko", "pf-a-niko"), ("user",)),
        AssayGold("q-code", ("pf-u-cobalt",), ("pf-u-amber", "pf-a-amber", "pf-u-cobalt", "pf-a-cobalt"), ("user",)),
        AssayGold("q-tea", ("pf-u-tea",), ("pf-u-tea", "pf-a-tea"), ("user",)),
        AssayGold("q-receipt", ("sp-a-receipt",), ("sp-a-receipt",), ("assistant",), (("sp-u-receipt", "sp-a-receipt"),)),
        AssayGold("q-serial", ("sp-a-serial",), ("sp-a-serial",), ("assistant",), (("sp-u-serial", "sp-a-serial"),)),
        AssayGold("q-meeting", ("sp-u-dana",), ("sp-u-dana", "sp-a-dana"), ("user",)),
        AssayGold("q-cross", ("tr-u-muir", "tr-u-big-sur", "tr-u-yosemite", "sp-u-dana"), ("tr-u-muir", "tr-a-muir", "tr-u-big-sur", "tr-a-big-sur", "tr-u-yosemite", "tr-a-yosemite", "sp-u-dana", "sp-a-dana"), ("user",)),
    )


def _source_rows(chunks: Sequence[AssayChunk]) -> tuple[tuple[str, tuple[AssayChunk, ...]], ...]:
    source_ids = tuple(dict.fromkeys(row.source_id for row in chunks))
    return tuple(
        (source_id, tuple(row for row in chunks if row.source_id == source_id))
        for source_id in source_ids
    )


def _build_surprise(chunks: Sequence[AssayChunk], artifact_id: str) -> tuple[Episode, ...]:
    episodes: list[Episode] = []
    for source_id, rows in _source_rows(chunks):
        result = EpisodeBuilder().build(
            source_id=source_id,
            artifact_id=artifact_id,
            spans=tuple(row.span() for row in rows),
            texts=tuple(row.text for row in rows),
        )
        episodes.extend(result.episodes)
    return tuple(episodes)


def _build_user_led(chunks: Sequence[AssayChunk], artifact_id: str) -> tuple[Episode, ...]:
    """Narrow adapter for the production user-led segmenter."""

    from memory_condense.search.episodes.user_led import UserLedEpisodeBuilder

    episodes: list[Episode] = []
    for source_id, rows in _source_rows(chunks):
        result = UserLedEpisodeBuilder().build(
            source_id=source_id,
            artifact_id=artifact_id,
            spans=tuple(row.span() for row in rows),
        )
        episodes.extend(result.episodes)
    return tuple(episodes)


def _macro_groups(
    microepisodes: Sequence[Episode],
    text_by_chunk: Mapping[str, str],
) -> tuple[tuple[Episode, ...], ...]:
    rows = tuple(microepisodes)
    if not rows:
        return ()
    texts = tuple(
        "\n".join(text_by_chunk[span.chunk_id] for span in episode.evidence)
        for episode in rows
    )
    scores = score_surprise_sequence(LexicalEmbeddingChangeScorer(), texts)
    initial = AdaptiveBoundaryDetector(
        window_size=3,
        gamma=1.0,
        min_history=2,
    ).detect(scores)
    matrix = tuple(
        tuple(lexical_cosine(left, right) for right in texts)
        for left in texts
    )
    refined = CohesionBoundaryRefiner(
        window=1,
        max_nodes=8,
        max_degree=3,
    ).refine(initial, item_count=len(rows), similarities=matrix)
    wanted = {item.position for item in refined}
    boundaries: list[int] = []
    start = 0
    while len(rows) - start > 4:
        choices = sorted(value for value in wanted if start < value <= start + 4)
        boundary = choices[0] if choices else start + 4
        boundaries.append(boundary)
        start = boundary
    boundaries.extend(sorted(value for value in wanted if start < value < len(rows)))
    cuts = tuple(sorted(set(boundaries)))
    starts = (0, *cuts)
    ends = (*cuts, len(rows))
    return tuple(tuple(rows[start:end]) for start, end in zip(starts, ends, strict=True))


def _build_hybrid(
    chunks: Sequence[AssayChunk],
    artifact_id: str,
) -> tuple[tuple[Episode, ...], tuple[MacroMembership, ...]]:
    microepisodes = _build_user_led(chunks, artifact_id)
    text_by_chunk = {row.chunk_id: row.text for row in chunks}
    memberships: list[MacroMembership] = []
    for source_id, _rows in _source_rows(chunks):
        source_micro = tuple(row for row in microepisodes if row.source_id == source_id)
        for group in _macro_groups(source_micro, text_by_chunk):
            member_ids = tuple(row.episode_id for row in group)
            memberships.append(
                MacroMembership(
                    group_id="macro-" + identity_sha256(
                        {
                            "format": FORMAT,
                            "source_id": source_id,
                            "member_episode_ids": list(member_ids),
                        }
                    )[:24],
                    source_id=source_id,
                    member_episode_ids=member_ids,
                )
            )
    return microepisodes, tuple(memberships)


def _build_arm(arm: str, chunks: Sequence[AssayChunk]) -> ArmIndex:
    artifact_id = identity_sha256({"format": FORMAT, "arm": arm})
    started = time.perf_counter_ns()
    memberships: tuple[MacroMembership, ...] = ()
    if arm == "surprise":
        episodes = _build_surprise(chunks, artifact_id)
    elif arm == "user_micro":
        episodes = _build_user_led(chunks, artifact_id)
    elif arm == "hybrid_overlay":
        episodes, memberships = _build_hybrid(chunks, artifact_id)
    else:
        raise ValueError(f"unknown arm: {arm}")
    build_ns = time.perf_counter_ns() - started
    text_by_chunk = {row.chunk_id: row.text for row in chunks}
    started = time.perf_counter_ns()
    representatives = {
        episode.episode_id: tuple(
            row.chunk_id
            for row in select_episode_representatives(
                episode,
                limit=REPRESENTATIVE_LIMIT,
                texts=text_by_chunk,
            )
        )
        for episode in episodes
    }
    representative_build_ns = time.perf_counter_ns() - started
    return ArmIndex(
        arm,
        episodes,
        representatives,
        build_ns,
        representative_build_ns,
        memberships,
    )


def _bm25_scores(
    query: str,
    chunks: Sequence[AssayChunk],
) -> dict[str, float]:
    searchable = tuple(row for row in chunks if row.role in {"user", "assistant"})
    documents = {
        row.chunk_id: Counter(tokenize(row.text))
        for row in searchable
    }
    query_terms = tuple(tokenize(query))
    document_count = len(documents)
    average_length = statistics.fmean(sum(row.values()) for row in documents.values())
    frequencies = {
        term: sum(term in document for document in documents.values())
        for term in set(query_terms)
    }
    scores: dict[str, float] = {}
    for chunk_id, document in documents.items():
        length = sum(document.values())
        score = 0.0
        for term in query_terms:
            frequency = document.get(term, 0)
            if not frequency:
                continue
            inverse_frequency = math.log(
                1.0
                + (document_count - frequencies[term] + 0.5)
                / (frequencies[term] + 0.5)
            )
            denominator = frequency + BM25_K1 * (
                1.0 - BM25_B + BM25_B * length / average_length
            )
            score += (
                inverse_frequency
                * frequency
                * (BM25_K1 + 1.0)
                / denominator
            )
        scores[chunk_id] = score
    return scores


def _frontier_query(question: AssayQuestion) -> bool:
    body = question.text.casefold()
    return bool(
        re.search(r"\b(order|ordered|earliest|latest)\b.*\btrips?\b", body)
        or re.search(r"\b(list|name)\s+all\b", body)
        or (body.startswith("which ") and " and " in body)
    )


def _overlay_payload(index: ArmIndex) -> dict[str, Any]:
    if index.arm != "hybrid_overlay":
        raise ValueError("macro overlay requested for another arm")
    episode_ids = tuple(row.episode_id for row in index.episodes)
    members = tuple(
        episode_id
        for group in index.macro_memberships
        for episode_id in group.member_episode_ids
    )
    if len(members) != len(set(members)) or set(members) != set(episode_ids):
        raise ValueError("macro overlay must partition all preserved microepisodes")
    links = [
        {
            "left_group_id": left.group_id,
            "right_group_id": right.group_id,
            "source_id": left.source_id,
        }
        for left, right in zip(
            index.macro_memberships,
            index.macro_memberships[1:],
        )
        if left.source_id == right.source_id
    ]
    return {
        "arm": index.arm,
        "format": FORMAT + "-macro-overlay",
        "gold_loaded": False,
        "groups": [row.projection() for row in index.macro_memberships],
        "links": links,
        "microepisode_receipts": {
            row.episode_id: row.receipt_sha256 for row in index.episodes
        },
    }


def _pack(
    admitted: Sequence[Episode],
    chunks: Sequence[AssayChunk],
    *,
    protected_chunk_ids: Sequence[str] = (),
) -> tuple[list[str], int]:
    by_chunk = {row.chunk_id: row for row in chunks}
    ordered_ids = tuple(protected_chunk_ids) + tuple(
        span.chunk_id for episode in admitted for span in episode.evidence
    )
    selected: list[str] = []
    token_count = 0
    for chunk_id in ordered_ids:
        if chunk_id in selected or len(selected) >= MAX_CHUNKS:
            continue
        row = by_chunk[chunk_id]
        cost = count_tokens(f"[{row.role}] {row.text}")
        if token_count + cost > MAX_PROMPT_TOKENS:
            continue
        selected.append(chunk_id)
        token_count += cost
    return selected, token_count


def _pack_overlay_microepisodes(
    admitted: Sequence[Episode],
    chunks: Sequence[AssayChunk],
    *,
    raw_anchor_chunk_ids: Sequence[str],
) -> tuple[list[str], int]:
    """Pack complete micros, preserving their user-led evidence order."""

    by_chunk = {row.chunk_id: row for row in chunks}
    anchor_ids = set(raw_anchor_chunk_ids)
    anchor_episode_ids = {
        episode.episode_id
        for episode in admitted
        if any(span.chunk_id in anchor_ids for span in episode.evidence)
    }
    selected: list[str] = []
    token_count = 0
    for episode in admitted:
        additions = [
            span.chunk_id
            for span in episode.evidence
            if span.chunk_id not in selected
        ]
        addition_tokens = sum(
            count_tokens(f"[{by_chunk[value].role}] {by_chunk[value].text}")
            for value in additions
        )
        fits = (
            len(selected) + len(additions) <= MAX_CHUNKS
            and token_count + addition_tokens <= MAX_PROMPT_TOKENS
        )
        if not fits:
            if episode.episode_id in anchor_episode_ids:
                raise RuntimeError("raw anchor microepisode exceeds the output budget")
            continue
        selected.extend(additions)
        token_count += addition_tokens
    if not anchor_ids <= set(selected):
        raise RuntimeError("raw anchor disappeared from its complete microepisode")
    return selected, token_count


def _query_representative(
    index: ArmIndex,
    question: AssayQuestion,
    chunks: Sequence[AssayChunk],
) -> dict[str, Any]:
    by_chunk = {row.chunk_id: row for row in chunks}
    ranked = sorted(
        index.episodes,
        key=lambda episode: (
            -max(
                (
                    lexical_cosine(question.text, by_chunk[chunk_id].text)
                    for chunk_id in index.representatives[episode.episode_id]
                ),
                default=0.0,
            ),
            episode.source_id,
            episode.sequence_no,
            episode.episode_id,
        ),
    )
    anchors = ranked[:MAX_ANCHORS]
    by_coordinate = {
        (episode.source_id, episode.sequence_no): episode
        for episode in index.episodes
    }
    admitted: list[Episode] = []
    seen: set[str] = set()
    for anchor in anchors:
        offsets = (0, *range(-1, -NEIGHBOR_RADIUS - 1, -1), *range(1, NEIGHBOR_RADIUS + 1))
        for offset in offsets:
            candidate = by_coordinate.get(
                (anchor.source_id, anchor.sequence_no + offset)
            )
            if candidate is None or candidate.episode_id in seen:
                continue
            if len(admitted) >= MAX_EPISODES:
                break
            seen.add(candidate.episode_id)
            admitted.append(candidate)
    selected, token_count = _pack(admitted, chunks)
    return {
        "question_id": question.question_id,
        "raw_anchor_chunk_ids": [],
        "selected_chunk_ids": selected,
        "selected_episode_ids": [row.episode_id for row in admitted],
        "prompt_tokens": token_count,
    }


def _query_overlay(
    index: ArmIndex,
    question: AssayQuestion,
    chunks: Sequence[AssayChunk],
) -> dict[str, Any]:
    scores = _bm25_scores(question.text, chunks)
    by_chunk = {row.chunk_id: row for row in chunks}
    owner = {
        span.chunk_id: episode
        for episode in index.episodes
        for span in episode.evidence
    }
    raw_anchor_rows: list[AssayChunk] = []
    raw_anchor_episode_ids: set[str] = set()
    for row in sorted(
        (
            row
            for row in chunks
            if row.chunk_id in scores and scores[row.chunk_id] > 0.0
        ),
        key=lambda row: (-scores[row.chunk_id], row.ordinal, row.chunk_id),
    ):
        episode_id = owner[row.chunk_id].episode_id
        if episode_id in raw_anchor_episode_ids:
            continue
        raw_anchor_rows.append(row)
        raw_anchor_episode_ids.add(episode_id)
        if len(raw_anchor_rows) >= MAX_ANCHORS:
            break
    raw_anchors = tuple(row.chunk_id for row in raw_anchor_rows)
    admitted: list[Episode] = []
    seen: set[str] = set()
    for chunk_id in raw_anchors:
        episode = owner[chunk_id]
        if episode.episode_id not in seen:
            admitted.append(episode)
            seen.add(episode.episode_id)

    frontier = _frontier_query(question)
    micro_cap = FRONTIER_MICRO_CAP if frontier else SCALAR_MICRO_CAP
    if frontier:
        episode_by_id = {row.episode_id: row for row in index.episodes}
        group_by_member = {
            episode_id: group
            for group in index.macro_memberships
            for episode_id in group.member_episode_ids
        }
        group_position = {
            group.group_id: position
            for position, group in enumerate(index.macro_memberships)
        }
        activated_group_ids = {
            group_by_member[episode.episode_id].group_id
            for episode in admitted
        }
        traversed_group_ids = set(activated_group_ids)
        for group_id in tuple(activated_group_ids):
            position = group_position[group_id]
            source_id = index.macro_memberships[position].source_id
            for neighbor_position in (position - 1, position + 1):
                if (
                    0 <= neighbor_position < len(index.macro_memberships)
                    and index.macro_memberships[neighbor_position].source_id
                    == source_id
                ):
                    traversed_group_ids.add(
                        index.macro_memberships[neighbor_position].group_id
                    )
        expansion_ids = {
            member_id
            for group in index.macro_memberships
            if group.group_id in traversed_group_ids
            for member_id in group.member_episode_ids
        }
        candidates = sorted(
            (
                episode_by_id[episode_id]
                for episode_id in expansion_ids
                if episode_id not in seen
            ),
            key=lambda episode: (
                -max(scores.get(span.chunk_id, 0.0) for span in episode.evidence),
                episode.source_id,
                episode.sequence_no,
                episode.episode_id,
            ),
        )
        for episode in candidates:
            if len(admitted) >= micro_cap:
                break
            admitted.append(episode)
            seen.add(episode.episode_id)

    selected, token_count = _pack_overlay_microepisodes(
        admitted,
        chunks,
        raw_anchor_chunk_ids=raw_anchors,
    )
    return {
        "frontier_expansion": frontier,
        "microepisode_cap": micro_cap,
        "prompt_tokens": token_count,
        "question_id": question.question_id,
        "raw_anchor_chunk_ids": list(raw_anchors),
        "selected_chunk_ids": selected,
        "selected_episode_ids": [row.episode_id for row in admitted],
    }


def _query(
    index: ArmIndex,
    question: AssayQuestion,
    chunks: Sequence[AssayChunk],
) -> dict[str, Any]:
    if index.arm == "hybrid_overlay":
        return _query_overlay(index, question, chunks)
    return _query_representative(index, question, chunks)


def _percentile95(values: Sequence[int]) -> int:
    ordered = sorted(values)
    return ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]


def construct(output_root: Path) -> tuple[str, str]:
    chunks = _corpus()
    questions = _questions()
    indexes = {arm: _build_arm(arm, chunks) for arm in ARMS}
    overlay, _ = publish_sealed_json(
        output_root / OVERLAY_NAME,
        _overlay_payload(indexes["hybrid_overlay"]),
    )
    selections: dict[str, list[dict[str, Any]]] = {}
    query_timings: dict[str, list[int]] = {}
    cold_query_timings: dict[str, list[int]] = {}
    for arm in ARMS:
        rows: list[dict[str, Any]] = []
        cold: list[int] = []
        for question in questions:
            started = time.perf_counter_ns()
            rows.append(_query(indexes[arm], question, chunks))
            cold.append(time.perf_counter_ns() - started)
        selections[arm] = rows
        cold_query_timings[arm] = cold
        samples: list[int] = []
        for _repeat in range(WARM_REPETITIONS):
            for question in questions:
                started = time.perf_counter_ns()
                _query(indexes[arm], question, chunks)
                samples.append(time.perf_counter_ns() - started)
        query_timings[arm] = samples
    corpus_projection = [
        {
            "chunk_id": row.chunk_id,
            "source_id": row.source_id,
            "turn_id": row.turn_id,
            "ordinal": row.ordinal,
            "role": row.role,
            "text_sha256": quote_sha256(row.text),
        }
        for row in chunks
    ]
    config = {
        "max_anchors": MAX_ANCHORS,
        "max_chunks": MAX_CHUNKS,
        "max_episodes": MAX_EPISODES,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "neighbor_radius": NEIGHBOR_RADIUS,
        "representative_limit": REPRESENTATIVE_LIMIT,
        "scalar_microepisode_cap": SCALAR_MICRO_CAP,
        "frontier_microepisode_cap": FRONTIER_MICRO_CAP,
        "warm_repetitions": WARM_REPETITIONS,
    }
    selection_payload = {
        "arms": selections,
        "config": config,
        "corpus_sha256": identity_sha256(corpus_projection),
        "format": FORMAT + "-selection",
        "gold_loaded": False,
        "model_calls": 0,
        "macro_overlay_sha256": overlay.sha256,
        "new_provider_calls": 0,
        "questions": [row.projection() for row in questions],
    }
    selection, _ = publish_sealed_json(
        output_root / SELECTION_NAME,
        selection_payload,
    )
    runtime_payload = {
        "arms": {
            arm: {
                "episode_build_ns": indexes[arm].build_ns,
                "episode_count": len(indexes[arm].episodes),
                "macro_group_count": len(indexes[arm].macro_memberships),
                "max_episode_chunks": max(
                    len(row.evidence) for row in indexes[arm].episodes
                ),
                "query_samples": len(query_timings[arm]),
                "query_first_pass_median_ns": int(
                    statistics.median(cold_query_timings[arm])
                ),
                "query_first_pass_p95_ns": _percentile95(cold_query_timings[arm]),
                "query_warm_median_ns": int(statistics.median(query_timings[arm])),
                "query_warm_p95_ns": _percentile95(query_timings[arm]),
                "representative_build_ns": indexes[arm].representative_build_ns,
            }
            for arm in ARMS
        },
        "format": FORMAT + "-runtime",
        "gold_loaded": False,
        "selection_sha256": selection.sha256,
        "timing_method": (
            "episode_build_and_representative_build_are_cold; query_first_pass_"
            "is_one_pass_over_each_query_on_the_built_index; query_warm_repeats_"
            "that_same_index"
        ),
    }
    runtime, _ = publish_sealed_json(output_root / RUNTIME_NAME, runtime_payload)
    return selection.sha256, runtime.sha256


def evaluate(output_root: Path) -> str:
    selection = read_sealed_json(output_root / SELECTION_NAME)
    chunks = _corpus()
    by_chunk = {row.chunk_id: row for row in chunks}
    gold_by_question = {row.question_id: row for row in _gold()}
    arm_metrics: dict[str, Any] = {}
    for arm in ARMS:
        metrics: list[dict[str, Any]] = []
        for selected in selection.payload["arms"][arm]:
            gold = gold_by_question[selected["question_id"]]
            selected_ids = set(selected["selected_chunk_ids"])
            target_ids = set(gold.target_chunk_ids)
            target_sources = {by_chunk[value].source_id for value in target_ids}
            selected_sources = {by_chunk[value].source_id for value in selected_ids}
            answer_bearing = selected_ids & set(gold.answer_bearing_chunk_ids)
            correct_role = {
                value
                for value in answer_bearing
                if by_chunk[value].role in gold.accepted_roles
            }
            closed = sum(
                left in selected_ids and right in selected_ids
                for left, right in gold.required_neighbor_pairs
            )
            per_source_target_recall = [
                len(
                    selected_ids
                    & {
                        value
                        for value in target_ids
                        if by_chunk[value].source_id == source_id
                    }
                )
                / sum(
                    by_chunk[value].source_id == source_id for value in target_ids
                )
                for source_id in sorted(target_sources)
            ]
            metrics.append(
                {
                    "evidence_recall": len(selected_ids & target_ids) / len(target_ids),
                    "neighbor_closure": (
                        closed / len(gold.required_neighbor_pairs)
                        if gold.required_neighbor_pairs
                        else 1.0
                    ),
                    "prompt_tokens": selected["prompt_tokens"],
                    "question_id": gold.question_id,
                    "role_correctness": (
                        len(correct_role) / len(answer_bearing)
                        if answer_bearing
                        else 0.0
                    ),
                    "source_recall": len(selected_sources & target_sources) / len(target_sources),
                    "target_source_evidence_recall": statistics.fmean(
                        per_source_target_recall
                    ),
                }
            )
        arm_metrics[arm] = {
            "means": {
                name: statistics.fmean(row[name] for row in metrics)
                for name in (
                    "evidence_recall",
                    "source_recall",
                    "target_source_evidence_recall",
                    "role_correctness",
                    "neighbor_closure",
                    "prompt_tokens",
                )
            },
            "questions": metrics,
        }
    artifact, _ = publish_sealed_json(
        output_root / EVALUATION_NAME,
        {
            "arms": arm_metrics,
            "format": FORMAT + "-evaluation",
            "gold_loaded_postseal": True,
            "selection_sha256": selection.sha256,
        },
    )
    return artifact.sha256


def replay(output_root: Path) -> str:
    expected = read_sealed_json(output_root / SELECTION_NAME)
    expected_overlay = read_sealed_json(output_root / OVERLAY_NAME)
    chunks = _corpus()
    questions = _questions()
    indexes = {arm: _build_arm(arm, chunks) for arm in ARMS}
    actual_overlay = _overlay_payload(indexes["hybrid_overlay"])
    if actual_overlay != expected_overlay.payload:
        raise RuntimeError("episode macro overlay replay changed")
    if expected.payload.get("macro_overlay_sha256") != expected_overlay.sha256:
        raise RuntimeError("episode selection lost its macro overlay binding")
    actual = {
        arm: [_query(indexes[arm], question, chunks) for question in questions]
        for arm in ARMS
    }
    if actual != expected.payload["arms"]:
        raise RuntimeError("episode segmentation selection replay changed")
    artifact, _ = publish_sealed_json(
        output_root / REPLAY_NAME,
        {
            "format": FORMAT + "-replay",
            "gold_loaded": False,
            "replay_equal": True,
            "selection_sha256": expected.sha256,
        },
    )
    return artifact.sha256


def run(output_root: Path) -> tuple[str, str, str, str]:
    selection, runtime = construct(output_root)
    evaluation = evaluate(output_root)
    replay_sha = replay(output_root)
    return selection, runtime, evaluation, replay_sha


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    for value in run(args.output_root.resolve()):
        print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
