from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.schemas import Turn
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import (
    RawSectionSpan,
    SectionSummary,
    summarize_section,
    summarize_user_sections,
)


def _turn(identity, text, *, role="user", source="source-a"):
    return Turn(turn_id=identity, text=text, role=role, source_id=source,
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc))


def _section(identity, turns, summary):
    return summarize_section(identity, turns, summarize=lambda _rows: summary,
                             summarizer_identity="fixture-v1")


def test_summary_routes_paraphrase_then_hydrates_whole_exact_raw_section_only():
    raw = [
        _turn("u1", "Where is it?\r\nPlease keep the original receipt."),
        _turn("a1", "  In drawer №2.\nSerial: α-07  \n", role="assistant"),
        _turn("u2", "Unrelated raw words: warranty location receipt serial."),
    ]
    sections = [_section("relevant", raw[:2], "Bicycle warranty paperwork storage location"),
                _section("decoy", raw[2:], "Cooking recipes")]
    index = SectionSummaryIndex(sections)
    loaded = []
    def load(identity):
        loaded.append(identity)
        return next(turn for turn in raw if turn.turn_id == identity)
    plan = index.route("Where are the bicycle warranty documents?", max_sections=1)
    assert loaded == []
    assert [r.section.section_id for r in plan.routes] == ["relevant"]
    result = hydrate_section_plan(plan, load_turn=load)
    assert loaded == ["u1", "a1"]
    assert [row.text for row in result.sections[0].evidence] == [turn.text for turn in raw[:2]]
    context = result.render_context()
    assert all(turn.text in context for turn in raw[:2])
    assert "Bicycle warranty paperwork" not in context
    assert "source-a" not in context
    assert "assistant" in context
    assert result.context_token_count == count_tokens(context)
    assert not result.requires_raw_fallback
    assert not plan.frontier_closed


def test_raw_words_and_opaque_source_ids_do_not_influence_summary_routing():
    section = _section("warranty", [_turn("u1", "warranty", source="warranty")], "Cooking")
    plan = SectionSummaryIndex([section]).route("warranty")
    assert plan.routes == ()
    result = hydrate_section_plan(plan, load_turn=lambda _id: pytest.fail("raw scan on summary miss"))
    assert result.requires_raw_fallback
    assert result.raw_turn_read_count == 0


def test_user_sections_keep_source_order_machine_payloads_and_orphan_preludes():
    turns = [
        _turn("orphan", "prelude", role="system"),
        _turn("a", "first lead"),
        _turn("foreign", "other lead", source="source-b"),
        _turn("b", "first response", role="assistant"),
        _turn("c", "next lead"),
    ]
    calls = []
    def summarize(rows):
        calls.append(tuple(row.turn_id for row in rows))
        return "topic"
    summaries = summarize_user_sections(turns, summarize=summarize, summarizer_identity="test")
    assert calls == [("orphan",), ("a", "b"), ("foreign",), ("c",)]
    index = SectionSummaryIndex(summaries)
    index.route("topic")
    assert len(calls) == 4  # No query-time summarization.
    assert {section.source_id for section in summaries} == {"source-a", "source-b"}


def test_explicit_large_topical_section_can_span_multiple_user_episodes():
    turns = [_turn("u1", "initial topic"), _turn("a1", "detail", role="assistant"),
             _turn("u2", "follow-up"), _turn("a2", "correction", role="assistant")]
    section = _section("macro", turns, "topic progression")
    result = hydrate_section_plan(SectionSummaryIndex([section]).route("progression"),
                                  load_turn={turn.turn_id: turn for turn in turns}.get)
    assert len(result.sections[0].evidence) == 4


def test_persisted_summary_index_reopens_without_raw_text_or_model():
    section = _section("section", [_turn("u", "UNIQUE RAW SECRET")], "Summary topic")
    index = SectionSummaryIndex([section])
    serialized = index.to_json()
    assert "UNIQUE RAW SECRET" not in serialized
    restored = SectionSummaryIndex.from_json(serialized)
    assert restored.to_json() == serialized
    assert restored.route("topic") == index.route("topic")
    with pytest.raises(AttributeError, match="immutable"):
        restored.sections = ()
    payload = json.loads(serialized)
    payload["sections"][0]["spans"][0]["end_char"] += 1
    with pytest.raises(ValueError, match="receipt"):
        SectionSummaryIndex.from_json(json.dumps(payload))


@pytest.mark.parametrize("mutation", ["text", "source", "role", "date", "missing"])
def test_stale_or_missing_member_rejects_the_whole_section(mutation):
    turns = [_turn("u", "lead"), _turn("a", "detail", role="assistant")]
    index = SectionSummaryIndex([_section("section", turns, "topic")])
    records = {turn.turn_id: turn for turn in turns}
    if mutation == "missing":
        records.pop("a")
    else:
        field, value = {
            "text": ("text", "changed"), "source": ("source_id", "foreign"),
            "role": ("role", "user"), "date": ("created_at", datetime(2026, 2, 1, tzinfo=timezone.utc)),
        }[mutation]
        records["a"] = records["a"].model_copy(update={field: value})
    result = hydrate_section_plan(index.route("topic"), load_turn=records.get)
    assert result.sections == ()
    assert result.render_context() == ""
    assert result.requires_raw_fallback
    assert len(result.diagnostics) == 1


def test_exact_character_section_preserves_whitespace_and_rejects_changed_parent():
    turn = _turn("u", "outside\n  exact τext\r\nmore outside")
    start, end = 8, 22
    span = RawSectionSpan.from_turn(turn, start_char=start, end_char=end)
    section = SectionSummary("slice", "source-a", "chosen phrase", (span,), "test")
    plan = SectionSummaryIndex([section]).route("phrase")
    result = hydrate_section_plan(plan, load_turn=lambda _id: turn)
    assert result.sections[0].evidence[0].text == turn.text[start:end]
    changed = turn.model_copy(update={"text": "OUTSIDE" + turn.text[7:]})
    assert hydrate_section_plan(plan, load_turn=lambda _id: changed).sections == ()


def test_summary_source_scope_is_exact_and_ties_are_deterministic():
    sections = [_section(identity, [_turn(identity, "raw", source=source)], "same topic")
                for identity, source in (("b", "prefix::b"), ("a", "prefix::a"))]
    index = SectionSummaryIndex(sections)
    assert [r.section.section_id for r in index.route("topic").routes] == ["a", "b"]
    plan = index.route("topic", eligible_source_ids=["prefix::b"])
    assert [r.section.section_id for r in plan.routes] == ["b"]
    assert index.route("topic", eligible_source_ids=["prefix"]).routes == ()


def test_complete_raw_sections_are_atomic_under_both_budgets():
    turns = [_turn("u", "first raw row"), _turn("a", "second raw row", role="assistant")]
    plan = SectionSummaryIndex([_section("s", turns, "topic")]).route("topic")
    records = {turn.turn_id: turn for turn in turns}
    complete = hydrate_section_plan(plan, load_turn=records.get)
    partial = hydrate_section_plan(plan, load_turn=records.get,
                                    max_context_tokens=complete.context_token_count - 1)
    assert not partial.sections and partial.requires_raw_fallback
    assert partial.diagnostics[0].reason == "context_budget"
    limited = hydrate_section_plan(plan, load_turn=lambda _id: pytest.fail("over-budget I/O"),
                                   max_raw_spans=1)
    assert limited.raw_turn_read_count == 0
    assert limited.diagnostics[0].reason == "raw_span_budget"


def test_independent_sections_survive_a_failed_or_oversize_section():
    turns = [_turn("a", "many " * 200), _turn("b", "short")]
    sections = [_section("a", turns[:1], "topic"), _section("b", turns[1:], "topic")]
    reads = []
    def load(identity):
        reads.append(identity)
        return turns[1]
    result = hydrate_section_plan(SectionSummaryIndex(sections).route("topic"),
                                  load_turn=load, max_context_tokens=100)
    assert reads == ["b"]
    assert [s.section.section_id for s in result.sections] == ["b"]
    assert result.requires_raw_fallback


def test_bad_summary_and_cross_source_inputs_are_rejected_before_compilation():
    with pytest.raises(ValueError, match="cross source"):
        _section("s", [_turn("a", "one"), _turn("b", "two", source="b")], "topic")
    with pytest.raises(ValueError, match="token budget"):
        summarize_section("s", [_turn("a", "raw")], summarize=lambda _: "long " * 200,
                          summarizer_identity="test", max_summary_tokens=8)
    with pytest.raises(ValueError, match="contiguous"):
        span = RawSectionSpan.from_turn(_turn("a", "one two"), end_char=3)
        SectionSummary("s", "source-a", "topic", (span, span), "test")


class _Embedder:
    dim = 8
    def embed_query(self, _query):
        vector = np.zeros(self.dim, dtype=np.float32)
        vector[0] = 1
        return vector
    def embed_chunks(self, chunks):
        return [chunk.model_copy(update={"embedding": self.embed_query("").tolist()}) for chunk in chunks]


def test_condenser_reads_selected_sections_from_durable_transcript_after_restart(tmp_path):
    kwargs = dict(data_dir=tmp_path / "store", embedder=_Embedder(), auto_extract=False,
                  chunker_min_tokens=1, chunker_max_tokens=50)
    with MemoryCondenser(**kwargs) as condenser:
        (user, _), (assistant, _) = condenser.ingest_many([
            ("user", "Where is it?", "session"),
            ("assistant", "  In drawer №2.\nKeep this spacing. ", "session"),
        ])
        section = _section("s", [user, assistant], "Warranty storage location")
        saved = SectionSummaryIndex([section]).to_json()
    with MemoryCondenser(**kwargs) as condenser:
        result = condenser.search_summary_sections("warranty", SectionSummaryIndex.from_json(saved))
        assert result.raw_turn_read_count == 2
        assert result.sections[0].evidence[1].text == assistant.text
        assert "Warranty storage location" not in result.render_context()
