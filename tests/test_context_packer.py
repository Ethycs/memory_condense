from __future__ import annotations

import pytest

from memory_condense._tokenizer import count_tokens, truncate_to_tokens
from memory_condense.context_packer import ContextBudget, ContextPacker
from memory_condense.schemas import (
    Chunk,
    MemoryItem,
    MemoryResult,
    MemoryStatus,
    MemoryType,
    PinState,
    RetrievalResult,
)


def _memory(content: str, **kwargs) -> MemoryItem:
    defaults = dict(type=MemoryType.DECISION, content=content)
    defaults.update(kwargs)
    return MemoryItem(**defaults)


def _result(text: str) -> RetrievalResult:
    chunk = Chunk(
        turn_id="t1",
        text=text,
        start_char=0,
        end_char=len(text),
        token_count=count_tokens(text),
    )
    return RetrievalResult(chunk=chunk, score=1.0)


class TestTruncateToTokens:
    def test_short_text_unchanged(self):
        assert truncate_to_tokens("hello world", 100) == "hello world"

    def test_long_text_is_cut(self):
        text = "word " * 500
        out = truncate_to_tokens(text, 10)
        assert count_tokens(out) <= 10
        assert len(out) < len(text)

    def test_zero_budget_returns_empty(self):
        assert truncate_to_tokens("anything", 0) == ""

    def test_literal_special_token_text_is_counted_as_corpus_data(self):
        text = "A chat export literally contains <|endoftext|> here."
        assert count_tokens(text) > 0
        assert truncate_to_tokens(text, 100) == text


class TestMemoryHeader:
    def test_empty_memories_produce_no_header(self):
        packed = ContextPacker().pack(memories=[])
        assert packed.memory_header == ""
        assert not any(m["content"].startswith("Relevant memory") for m in packed.messages)

    def test_bullets_are_typed(self):
        packed = ContextPacker().pack(memories=[_memory("use SQLite for storage")])
        assert "[Decision] use SQLite for storage" in packed.memory_header

    def test_details_appended_in_parens(self):
        item = _memory("use SQLite", details="WAL mode")
        packed = ContextPacker().pack(memories=[item])
        assert "(WAL mode)" in packed.memory_header

    def test_pinned_items_are_marked(self):
        item = _memory("never log secrets", pin=PinState.USER)
        packed = ContextPacker().pack(memories=[item])
        assert "]*" in packed.memory_header

    def test_superseded_items_are_excluded(self):
        stale = _memory("old decision", status=MemoryStatus.SUPERSEDED)
        packed = ContextPacker().pack(memories=[stale])
        assert packed.memory_header == ""

    def test_accepts_memory_results(self):
        wrapped = MemoryResult(item=_memory("wrapped item"), score=1.0)
        packed = ContextPacker().pack(memories=[wrapped])
        assert "wrapped item" in packed.memory_header

    def test_records_only_memory_ids_that_reach_the_header(self):
        first = _memory("first compact decision")
        oversized = _memory("detail " * 100)
        packed = ContextPacker(ContextBudget(memory_header_tokens=20)).pack(
            memories=[first, oversized]
        )

        assert packed.memory_ids == [first.mem_id]
        assert packed.dropped["memories"] == 1

    def test_header_respects_budget_and_counts_drops(self):
        budget = ContextBudget(memory_header_tokens=40)
        items = [_memory(f"decision number {i} " + "detail " * 20) for i in range(10)]
        packed = ContextPacker(budget).pack(memories=items)
        assert packed.token_counts["memory_header"] <= 40
        assert packed.dropped["memories"] > 0


class TestRecentTurns:
    def test_turns_kept_in_chronological_order(self):
        turns = [("user", "first"), ("assistant", "second"), ("user", "third")]
        packed = ContextPacker().pack(recent_turns=turns)
        assert packed.recent_turns == turns

    def test_oldest_turns_dropped_first(self):
        budget = ContextBudget(recent_window_tokens=20)
        turns = [("user", "old " * 30), ("user", "recent")]
        packed = ContextPacker(budget).pack(recent_turns=turns)
        assert packed.recent_turns == [("user", "recent")]
        assert packed.dropped["recent_turns"] == 1

    def test_budget_is_never_exceeded(self):
        budget = ContextBudget(recent_window_tokens=50)
        turns = [("user", "filler " * 40) for _ in range(10)]
        packed = ContextPacker(budget).pack(recent_turns=turns)
        assert packed.token_counts["recent_turns"] <= 50


class TestExpansions:
    def test_default_count_allows_all_ten_retrieval_candidates(self):
        results = [_result(f"short excerpt {i}") for i in range(10)]
        packed = ContextPacker().pack(expansions=results)

        assert len(packed.expansions) == 10
        assert packed.dropped["expansions"] == 0

    def test_expansions_capped_by_count(self):
        budget = ContextBudget(max_expansions=2)
        results = [_result(f"excerpt {i}") for i in range(5)]
        packed = ContextPacker(budget).pack(expansions=results)
        assert len(packed.expansions) == 2
        assert packed.dropped["expansions"] == 3

    def test_each_expansion_truncated(self):
        budget = ContextBudget(max_expansion_tokens=10)
        packed = ContextPacker(budget).pack(expansions=[_result("word " * 200)])
        assert count_tokens(packed.expansions[0]) <= 15  # 10 + index marker

    def test_final_expansion_uses_the_remaining_aggregate_budget(self):
        budget = ContextBudget(
            expansion_tokens=50,
            max_expansions=10,
            max_expansion_tokens=30,
        )
        packed = ContextPacker(budget).pack(
            expansions=[_result("alpha " * 100), _result("beta " * 100)]
        )

        assert len(packed.expansions) == 2
        assert packed.token_counts["expansions"] <= 50
        assert count_tokens(packed.expansions[1]) < count_tokens(
            packed.expansions[0]
        )

    def test_expansions_are_numbered(self):
        packed = ContextPacker().pack(expansions=[_result("alpha"), _result("beta")])
        assert packed.expansions[0].startswith("[1]")
        assert packed.expansions[1].startswith("[2]")


class TestMessageAssembly:
    def test_section_order(self):
        packed = ContextPacker().pack(
            system_prompt="You are helpful.",
            memories=[_memory("prefers dark mode", type=MemoryType.PREFERENCE)],
            recent_turns=[("user", "hi"), ("assistant", "hello")],
            expansions=[_result("supporting text")],
            user_text="what do I prefer?",
        )
        roles = [m["role"] for m in packed.messages]
        contents = [m["content"] for m in packed.messages]

        assert contents[0] == "You are helpful."
        assert contents[1].startswith("Relevant memory:")
        assert roles[2:4] == ["user", "assistant"]
        assert contents[4].startswith("Supporting excerpts:")
        assert packed.messages[-1] == {"role": "user", "content": "what do I prefer?"}

    def test_empty_sections_are_omitted(self):
        packed = ContextPacker().pack(user_text="hello")
        assert packed.messages == [{"role": "user", "content": "hello"}]

    def test_total_tokens_sums_sections(self):
        packed = ContextPacker().pack(
            system_prompt="sys", recent_turns=[("user", "hi")], user_text="q"
        )
        assert packed.total_tokens == sum(packed.token_counts.values())

    def test_budget_total_is_the_ceiling(self):
        budget = ContextBudget()
        packed = ContextPacker(budget).pack(
            memories=[_memory("m " * 500) for _ in range(50)],
            recent_turns=[("user", "t " * 500) for _ in range(50)],
            expansions=[_result("e " * 500) for _ in range(50)],
        )
        packed_body = (
            packed.token_counts["memory_header"]
            + packed.token_counts["recent_turns"]
            + packed.token_counts["expansions"]
        )
        assert packed_body <= budget.total()
