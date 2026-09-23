from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.application.discourse_sources import scan_discourse_source_chunks
from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.schemas import Chunk
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.search.incremental_conversation_graph import (
    ConversationGraphChunk,
)
from memory_condense.search.indexes.lexical import LexicalIndex

from tools import assay_hot_v3_ordered_story_replacement as graph_assay
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.full_store_slot_closure import build_full_store_window_index
from tools.matched_eval.hot_incremental_graph_lane import (
    build_hot_incremental_graph_index,
)
from tools.matched_eval.hot_incremental_ordered_story import (
    HotIncrementalOrderedStoryError,
    HotIncrementalOrderedStoryReplacementResult,
    replace_ambiguous_ordered_typed_witnesses_from_graph,
)
from tools.matched_eval.hot_ordered_story_replacement import (
    HotOrderedStoryReplacementResult,
    replace_ambiguous_ordered_typed_witnesses,
)
from tools.matched_eval.hot_typed_witness import (
    assess_ordered_list_ambiguity,
    build_hot_typed_witness_index,
    query_hot_typed_witnesses,
)
from tools.matched_eval.query_expansion import FrozenSourceNamespace
from tools.matched_eval.query_guided_scan import cache_namespace_partitions
from tools.matched_eval.temporal_insufficiency_specialist import (
    scan_temporal_insufficiency_specialist,
)


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _index(
    path: Path,
    rows: list[tuple[str, str, datetime]],
):
    database = Database(path)
    transcript = TranscriptStore(database)
    lexical = LexicalIndex(database)
    for offset, (source_id, text, created_at) in enumerate(rows):
        turn = transcript.append(
            "user", text, source_id=source_id, created_at=created_at
        )
        lexical.add_chunks(
            [
                Chunk(
                    chunk_id=f"chunk-{offset}",
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
    store_receipt = _sha("combined-store")
    namespace = FrozenSourceNamespace.from_source_streams(
        snapshot_id=_sha("snapshot"),
        combined_store_receipt_sha256=store_receipt,
        source_streams=streams,
    )
    with Database(path, read_only=True) as readonly:
        cache = cache_namespace_partitions(
            readonly,
            namespace,
            source_database_sha256=_sha("database"),
            source_store_receipt_sha256=store_receipt,
        )
    return build_full_store_window_index(cache)


QUESTION = (
    "[Question asked at 2023/06/01 (Thu) 03:56]\n"
    "What is the order of the three trips I took in the past three months, "
    "from earliest to latest?"
)

INELIGIBLE_QUESTION = (
    "[Question asked at 2023/06/01 (Thu) 03:56]\n"
    "Where did I leave my backpack?"
)


def _population_context(
    namespace_id: str,
    *dated_questions: str,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    namespace = SimpleNamespace(namespace_id=namespace_id)
    rows = tuple(
        SimpleNamespace(
            namespace=namespace,
            source=SimpleNamespace(
                packet=SimpleNamespace(
                    dated_question=dated_question,
                    dated_question_sha256=quote_sha256(dated_question),
                )
            ),
        )
        for dated_question in dated_questions
    )
    return SimpleNamespace(population=SimpleNamespace(rows=rows)), namespace


def _story_rows(*, decoys: bool) -> list[tuple[str, str, datetime]]:
    rows = [
        (
            "muir::session",
            "I took a day hike to Muir Woods with my family.",
            datetime(2023, 3, 10, tzinfo=timezone.utc),
        ),
        (
            "muir::session",
            "I packed my Eastern Sierra backpack and bear canister.",
            datetime(2023, 3, 10, tzinfo=timezone.utc),
        ),
        (
            "bigsur::session",
            "I took a road trip through Big Sur and Monterey.",
            datetime(2023, 4, 20, tzinfo=timezone.utc),
        ),
        (
            "bigsur::session",
            "My Eastern Sierra backpack needed a better bear canister.",
            datetime(2023, 4, 20, tzinfo=timezone.utc),
        ),
        (
            "yosemite::session",
            "I went on a camping trip to Yosemite.",
            datetime(2023, 5, 15, tzinfo=timezone.utc),
        ),
        (
            "yosemite::session",
            "I reused the Eastern Sierra backpack and bear canister.",
            datetime(2023, 5, 15, tzinfo=timezone.utc),
        ),
    ]
    if decoys:
        rows.extend(
            [
                (
                    "new-york::session",
                    "I recently took a quick business trip to New York City.",
                    datetime(2023, 5, 29, tzinfo=timezone.utc),
                ),
                (
                    "yellowstone::session",
                    "I went on a 2500-mile Yellowstone road trip.",
                    datetime(2023, 5, 25, tzinfo=timezone.utc),
                ),
                (
                    "columbia::session",
                    "I hiked around Columbia River Gorge on a weekend trip.",
                    datetime(2023, 5, 22, tzinfo=timezone.utc),
                ),
                (
                    "whistler::session",
                    "I went to Whistler for a fast ski trip.",
                    datetime(2023, 5, 20, tzinfo=timezone.utc),
                ),
                (
                    "costa-rica::session",
                    "I went on a short surfing trip to Costa Rica.",
                    datetime(2023, 5, 18, tzinfo=timezone.utc),
                ),
                (
                    "british-columbia::session",
                    "I went on a cycling trip through British Columbia.",
                    datetime(2023, 5, 17, tzinfo=timezone.utc),
                ),
                (
                    "old::session",
                    "I traveled to Yellowstone on a winter vacation.",
                    datetime(2022, 12, 1, tzinfo=timezone.utc),
                ),
            ]
        )
    return rows


def _baseline(index):
    return query_hot_typed_witnesses(
        build_hot_typed_witness_index(index),
        QUESTION,
        protected_chunk_ids=(),
    )


def test_false_ambiguity_gate_returns_exact_baseline_without_scanning(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path / "compact.db", _story_rows(decoys=False))
    baseline = _baseline(index)
    assert assess_ordered_list_ambiguity(baseline).escalate is False

    def forbidden_scan(*_args):
        raise AssertionError("specialist ran behind a false gate")

    resolved = replace_ambiguous_ordered_typed_witnesses(
        index,
        QUESTION,
        baseline,
        specialist_scanner=forbidden_scan,
    )

    assert resolved is baseline


def test_q86_like_ambiguity_replaces_broad_typed_decoys_with_linked_story(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path / "ambiguous.db", _story_rows(decoys=True))
    baseline = _baseline(index)
    decision = assess_ordered_list_ambiguity(baseline)
    assert decision.escalate is True
    assert decision.requested_cardinality == 3

    resolved = replace_ambiguous_ordered_typed_witnesses(
        index,
        QUESTION,
        baseline,
    )

    assert isinstance(resolved, HotOrderedStoryReplacementResult)
    assert [row.event_date for row in resolved.witnesses] == [
        "2023-03-10",
        "2023-04-20",
        "2023-05-15",
    ]
    rendered = "\n".join(row.quote for row in resolved.witnesses)
    assert "Muir Woods" in rendered
    assert "Big Sur" in rendered
    assert "Yosemite" in rendered
    assert "New York" not in rendered
    assert "Yellowstone" not in rendered
    assert "Columbia" not in rendered
    assert "Whistler" not in rendered
    assert "Costa Rica" not in rendered
    assert "British Columbia" not in rendered
    assert resolved.receipt.baseline_appended is False


def test_q86_like_graph_replacement_is_exact_upstream_typed_subset(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path / "graph-ambiguous.db", _story_rows(decoys=True))
    graph_index = build_hot_incremental_graph_index(index)
    baseline = _baseline(index)

    resolved = replace_ambiguous_ordered_typed_witnesses_from_graph(
        graph_index,
        QUESTION,
        baseline,
    )

    assert isinstance(resolved, HotIncrementalOrderedStoryReplacementResult)
    assert resolved.receipt.ordered_event_times_utc == (
        "2023-03-10T00:00:00+00:00",
        "2023-04-20T00:00:00+00:00",
        "2023-05-15T00:00:00+00:00",
    )
    assert [row.source_id for row in resolved.witnesses] == [
        "muir::session",
        "bigsur::session",
        "yosemite::session",
    ]
    rendered = "\n".join(row.quote for row in resolved.witnesses)
    assert "Muir Woods" in rendered
    assert "Big Sur" in rendered
    assert "Yosemite" in rendered
    assert "New York" not in rendered
    assert "Yellowstone" not in rendered
    upstream_by_chunk = {
        row.span.chunk_id: row.candidate_id
        for row in baseline.selected_before_dedup
    }
    assert all(
        row.span.chunk_id in upstream_by_chunk
        and upstream_by_chunk[row.span.chunk_id]
        in row.upstream_typed_candidate_ids
        for row in resolved.witnesses
    )
    audit = resolved.audit_projection()
    assert audit["receipt"]["baseline_appended"] is False
    assert audit["receipt"]["replacement_is_upstream_subset"] is True
    assert audit["receipt"]["new_provider_calls"] == 0
    assert audit["receipt"]["model_calls"] == 0
    assert audit["receipt"]["gold_loaded"] is False


def test_graph_replacement_returns_identical_baseline_on_semantic_abstention(
    tmp_path: Path,
) -> None:
    rows = [
        (
            f"source-{number}",
            f"I took a trip to Destination {number}.",
            datetime(2023, 5, number + 1, tzinfo=timezone.utc),
        )
        for number in range(8)
    ]
    index = _index(tmp_path / "graph-disconnected.db", rows)
    graph_index = build_hot_incremental_graph_index(index)
    baseline = _baseline(index)
    assert assess_ordered_list_ambiguity(baseline).escalate is True

    resolved = replace_ambiguous_ordered_typed_witnesses_from_graph(
        graph_index,
        QUESTION,
        baseline,
    )

    assert resolved is baseline


def test_graph_replacement_does_not_hide_a_changed_sealed_graph(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path / "graph-changed.db", _story_rows(decoys=True))
    graph_index = build_hot_incremental_graph_index(index)
    baseline = _baseline(index)
    extra_text = "I took a separate trip after the graph was sealed."
    graph_index.graph.append_chunk(
        ConversationGraphChunk(
            chunk_id="post-seal",
            source_id="post-seal-source",
            turn_id="post-seal-turn",
            ordinal=10_000,
            role="user",
            text=extra_text,
            start_char=0,
            end_char=len(extra_text),
            created_at="2023-05-31T00:00:00+00:00",
        )
    )

    with pytest.raises(HotIncrementalOrderedStoryError, match="changed after"):
        replace_ambiguous_ordered_typed_witnesses_from_graph(
            graph_index,
            QUESTION,
            baseline,
        )


def test_ordered_story_variant_builds_one_graph_for_repeated_namespace_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full = _index(tmp_path / "graph-lifecycle.db", _story_rows(decoys=True))
    builds = {"full": 0, "graph": 0}
    actual_graph_builder = graph_assay.build_hot_incremental_graph_index

    def fake_full_builder(_context, _namespace):
        builds["full"] += 1
        return full, {"base_index_build_ns": 17}

    def counted_graph_builder(parent):
        builds["graph"] += 1
        return actual_graph_builder(parent)

    monkeypatch.setattr(
        graph_assay.base,
        "_build_resident_index",
        fake_full_builder,
    )
    monkeypatch.setattr(
        graph_assay,
        "build_hot_incremental_graph_index",
        counted_graph_builder,
    )
    hooks = graph_assay._runtime_hooks()
    context, namespace = _population_context(_sha("eligible-namespace"), QUESTION)
    namespace_index, timing = hooks.build_full_index(context, namespace)
    typed_index = hooks.build_typed_index(namespace_index)
    baseline = hooks.query_typed(
        typed_index,
        QUESTION,
        protected_chunk_ids=(),
    )

    first = hooks.resolve_typed(namespace_index, QUESTION, baseline)
    second = hooks.resolve_typed(namespace_index, QUESTION, baseline)

    assert builds == {"full": 1, "graph": 1}
    assert namespace_index.full is full
    assert namespace_index.graph is not None
    assert namespace_index.graph.parent is full
    assert namespace_index.rows == full.rows
    assert namespace_index.eligible_dated_question_sha256s == (
        quote_sha256(QUESTION),
    )
    assert namespace_index.graph_build_reason == (
        "eligible_ordered_temporal_questions"
    )
    assert timing["graph_index_build_count"] == 1
    assert timing["graph_index_build_ns"] > 0
    assert timing["graph_index_receipt_sha256"] == (
        namespace_index.graph.receipt_sha256
    )
    assert isinstance(first, HotIncrementalOrderedStoryReplacementResult)
    assert second == first


def test_ordered_story_variant_skips_graph_for_ineligible_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full = _index(tmp_path / "graph-skipped.db", _story_rows(decoys=True))
    builds = {"full": 0, "graph": 0}

    def fake_full_builder(_context, _namespace):
        builds["full"] += 1
        return full, {"base_index_build_ns": 17}

    def forbidden_graph_builder(_parent):
        builds["graph"] += 1
        raise AssertionError("ineligible namespace built a graph")

    monkeypatch.setattr(
        graph_assay.base,
        "_build_resident_index",
        fake_full_builder,
    )
    monkeypatch.setattr(
        graph_assay,
        "build_hot_incremental_graph_index",
        forbidden_graph_builder,
    )
    hooks = graph_assay._runtime_hooks()
    context, namespace = _population_context(
        _sha("ineligible-namespace"),
        INELIGIBLE_QUESTION,
    )
    namespace_index, timing = hooks.build_full_index(context, namespace)
    typed_index = hooks.build_typed_index(namespace_index)
    baseline = hooks.query_typed(
        typed_index,
        INELIGIBLE_QUESTION,
        protected_chunk_ids=(),
    )

    resolved = hooks.resolve_typed(
        namespace_index,
        INELIGIBLE_QUESTION,
        baseline,
    )

    assert resolved is baseline
    assert builds == {"full": 1, "graph": 0}
    assert namespace_index.graph is None
    assert namespace_index.eligible_dated_question_sha256s == ()
    assert namespace_index.graph_build_reason == (
        "no_eligible_ordered_temporal_questions"
    )
    assert timing["graph_index_build_count"] == 0
    assert timing["graph_index_build_ns"] == 0
    assert timing["graph_index_receipt_sha256"] is None
    assert timing["graph_index_stats"] is None
    assert namespace_index.projection()["graph_build_count"] == 0
    assert namespace_index.projection()["graph_index_receipt_sha256"] is None


def test_ordered_story_variant_returns_baseline_for_ineligible_question_in_graph_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full = _index(tmp_path / "graph-mixed.db", _story_rows(decoys=True))
    monkeypatch.setattr(
        graph_assay.base,
        "_build_resident_index",
        lambda _context, _namespace: (full, {}),
    )
    hooks = graph_assay._runtime_hooks()
    context, namespace = _population_context(
        _sha("mixed-namespace"),
        QUESTION,
        INELIGIBLE_QUESTION,
    )
    namespace_index, _timing = hooks.build_full_index(context, namespace)
    typed_index = hooks.build_typed_index(namespace_index)
    baseline = hooks.query_typed(
        typed_index,
        INELIGIBLE_QUESTION,
        protected_chunk_ids=(),
    )

    assert namespace_index.graph is not None
    assert hooks.resolve_typed(
        namespace_index,
        INELIGIBLE_QUESTION,
        baseline,
    ) is baseline


def test_incomplete_first_person_proof_fails_open_to_exact_baseline(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path / "invalid.db", _story_rows(decoys=True))
    baseline = _baseline(index)
    specialist = scan_temporal_insufficiency_specialist(index, QUESTION)
    first = specialist.temporal_bundle.ordered_candidate_ids[0]
    damaged = replace(
        specialist,
        candidates=tuple(
            replace(row, first_person_assertion=False)
            if row.candidate_id == first
            else row
            for row in specialist.candidates
        ),
    )

    resolved = replace_ambiguous_ordered_typed_witnesses(
        index,
        QUESTION,
        baseline,
        specialist_scanner=lambda *_args: damaged,
    )

    assert resolved is baseline
