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

    def test_consolidation_candidates_are_additive_to_direct_slots(self):
        direct = [_result("direct alpha"), _result("direct beta")]
        learned = _result("learned gamma").model_copy(
            update={"route": "live_consolidation"}
        )
        budget = ContextBudget(
            expansion_tokens=100,
            max_expansions=2,
            max_consolidation_expansions=1,
        )

        packed = ContextPacker(budget).pack(expansions=[*direct, learned])

        assert packed.expansion_chunk_ids == [
            direct[0].chunk.chunk_id,
            direct[1].chunk.chunk_id,
            learned.chunk.chunk_id,
        ]

    def test_budget_aware_order_keeps_short_high_utility_evidence(self):
        long = _result("long " * 20).model_copy(update={"score": 0.9})
        short_a = _result("short alpha").model_copy(update={"score": 0.8})
        short_b = _result("short beta").model_copy(update={"score": 0.7})
        budget = ContextBudget(
            expansion_tokens=18,
            max_expansions=3,
            budget_aware_expansions=True,
        )

        packed = ContextPacker(budget).pack(
            expansions=[long, short_a, short_b]
        )

        assert short_a.chunk.chunk_id in packed.expansion_chunk_ids
        assert short_b.chunk.chunk_id in packed.expansion_chunk_ids
        assert long.chunk.chunk_id not in packed.expansion_chunk_ids

    def test_source_diverse_budget_aware_order_penalizes_repeated_source(self):
        source_a_first = _result("alpha first").model_copy(
            update={"score": 1.0, "memory_source_id": "source-a"}
        )
        source_a_second = _result("alpha second").model_copy(
            update={"score": 0.9, "memory_source_id": "source-a"}
        )
        source_b = _result("beta evidence").model_copy(
            update={"score": 0.6, "memory_source_id": "source-b"}
        )
        budget = ContextBudget(
            expansion_tokens=30,
            max_expansions=2,
            budget_aware_expansions=True,
            source_diverse_expansions=True,
        )

        packed = ContextPacker(budget).pack(
            expansions=[source_a_first, source_a_second, source_b]
        )

        assert packed.expansion_chunk_ids == [
            source_a_first.chunk.chunk_id,
            source_b.chunk.chunk_id,
        ]

    def test_query_aware_sentence_packing_keeps_best_matches_in_source_order(self):
        result = _result(
            "The garden is quiet today. "
            "My deployment region is westus3. "
            "The build uses twelve workers. "
            "Nothing else changed."
        )
        budget = ContextBudget(
            expansion_tokens=100,
            query_aware_sentence_expansions=True,
            max_sentences_per_expansion=2,
        )

        packed = ContextPacker(budget).pack(
            expansions=[result],
            user_text="Which deployment region and how many workers did I use?",
        )

        excerpt = packed.expansions[0]
        assert "westus3" in excerpt
        assert "twelve workers" in excerpt
        assert "garden" not in excerpt
        assert "Nothing else" not in excerpt
        assert packed.expansion_chunk_ids == [result.chunk.chunk_id]

    def test_query_aware_sentence_packing_preserves_dense_only_hit(self):
        original = "A semantically relevant sentence. Another supporting sentence. Third."
        budget = ContextBudget(query_aware_sentence_expansions=True)

        packed = ContextPacker(budget).pack(
            expansions=[_result(original)],
            user_text="unmatched vocabulary",
        )

        assert original in packed.expansions[0]

    def test_information_gain_packing_stops_before_low_signal_noise(self):
        relevant = _result("The cerulean launch code is seven.").model_copy(
            update={"score": 0.2, "memory_source_id": "source-relevant"}
        )
        noise = _result("Garden furniture and ordinary weather.").model_copy(
            update={"score": 0.001, "memory_source_id": "source-noise"}
        )
        budget = ContextBudget(
            expansion_tokens=100,
            information_gain_expansions=True,
            min_information_gain_per_token=0.005,
        )

        packed = ContextPacker(budget).pack(
            expansions=[noise, relevant],
            user_text="What is the cerulean launch code?",
        )

        assert packed.expansion_chunk_ids == [relevant.chunk.chunk_id]
        assert packed.token_counts["expansions"] < budget.expansion_tokens

    def test_information_gain_packing_retains_semantic_only_signal(self):
        semantic = _result("A paraphrased but useful passage.").model_copy(
            update={"score": 0.9}
        )
        budget = ContextBudget(
            information_gain_expansions=True,
            min_information_gain_per_token=0.005,
        )

        packed = ContextPacker(budget).pack(
            expansions=[semantic],
            user_text="unmatched query vocabulary",
        )

        assert packed.expansion_chunk_ids == [semantic.chunk.chunk_id]

    def test_information_gain_zero_threshold_preserves_retrieval_order(self):
        first = _result("First ranked evidence.").model_copy(
            update={"score": 0.1, "memory_source_id": "source-a"}
        )
        second = _result("Second ranked evidence.").model_copy(
            update={"score": 0.9, "memory_source_id": "source-b"}
        )
        budget = ContextBudget(
            expansion_tokens=100,
            information_gain_expansions=True,
            min_information_gain_per_token=0.0,
        )

        packed = ContextPacker(budget).pack(
            expansions=[first, second],
            user_text="evidence",
        )

        assert packed.expansion_chunk_ids == [
            first.chunk.chunk_id,
            second.chunk.chunk_id,
        ]

    def test_information_gain_retains_more_for_multi_fact_query(self):
        marginal = _result("A distinct supporting detail appears here.").model_copy(
            update={"score": 0.05, "memory_source_id": "source-a"}
        )
        budget = ContextBudget(
            expansion_tokens=100,
            information_gain_expansions=True,
            min_information_gain_per_token=0.0075,
        )
        packer = ContextPacker(budget)

        singleton = packer.pack(
            expansions=[marginal],
            user_text="What happened?",
        )
        multi_fact = packer.pack(
            expansions=[marginal],
            user_text="List all items in order.",
        )

        assert singleton.expansion_chunk_ids == []
        assert multi_fact.expansion_chunk_ids == [marginal.chunk.chunk_id]

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

    def test_records_ids_only_for_expansions_that_reach_the_prompt(self):
        results = [_result(f"excerpt {index}") for index in range(3)]
        packed = ContextPacker(ContextBudget(max_expansions=2)).pack(
            expansions=results
        )
        assert packed.expansion_chunk_ids == [
            results[0].chunk.chunk_id,
            results[1].chunk.chunk_id,
        ]


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
