from __future__ import annotations

from types import SimpleNamespace

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
    Turn,
)


def _memory(content: str, **kwargs) -> MemoryItem:
    defaults = dict(type=MemoryType.DECISION, content=content)
    defaults.update(kwargs)
    return MemoryItem(**defaults)


def _result(text: str, *, source_id: str | None = None) -> RetrievalResult:
    chunk = Chunk(
        turn_id="t1",
        text=text,
        start_char=0,
        end_char=len(text),
        token_count=count_tokens(text),
    )
    return RetrievalResult(
        chunk=chunk,
        score=1.0,
        memory_source_id=source_id,
    )


def _reserved_museum_results(
    count: int = 6,
) -> tuple[list[RetrievalResult], dict[str, str], list[str]]:
    results: list[RetrievalResult] = []
    source_metadata: dict[str, str] = {}
    timestamps: list[str] = []
    for index in range(count):
        source_id = f"museum-session-{index}"
        body = (
            f"Museum visit {index} identifies venue {index}. "
            + "Additional raw evidence supplies distinguishing context. " * 8
        )
        result = _result(body, source_id=source_id)
        results.append(
            result.model_copy(
                update={
                    "turn": Turn(
                        turn_id=result.chunk.turn_id,
                        source_id=source_id,
                        role="user",
                        text=body,
                    )
                }
            )
        )
        timestamp = (
            f"2024/07/{index + 1:02d} (Mon) 08:30 "
            "Pacific Daylight Time archival session"
        )
        timestamps.append(timestamp)
        source_metadata[source_id] = (
            f"[{source_id} took place at {timestamp}]"
        )
    return results, source_metadata, timestamps


class _AllReservedSelector:
    requires_baseline_ranking = True
    requires_complete_frontier = True

    def __init__(self) -> None:
        self.last_candidate_trace = []

    def select(self, query, values, **kwargs):
        del query, kwargs
        self.last_candidate_trace = [
            {
                "chunk_id": value.chunk.chunk_id,
                "group_id": f"event-{index}",
                "group_role": "representative",
                "coverage_reserved": True,
            }
            for index, value in enumerate(values)
        ]
        return list(values)


class _FixedClosureSelector:
    """Deterministic test double for the destructive closure proof."""

    requires_baseline_ranking = True
    requires_complete_frontier = True
    strict = False

    def __init__(
        self,
        reserved: list[RetrievalResult],
        *,
        report_updates: dict | None = None,
        trace_case: str = "",
        output_case: str = "",
        allow_selected_scope_fixed_k_closure: bool = False,
    ) -> None:
        self._reserved = list(reserved)
        self._reserved_ids = {
            result.chunk.chunk_id for result in reserved
        }
        self._report_updates = report_updates or {}
        self._trace_case = trace_case
        self._output_case = output_case
        self.allow_selected_scope_fixed_k_closure = bool(
            allow_selected_scope_fixed_k_closure
        )
        self.last_candidate_trace = []
        self.last_report = None

    def select(self, query, values, **kwargs):
        del query, kwargs
        rows = []
        reserved_index = 0
        for value in values:
            chunk_id = value.chunk.chunk_id
            reserved = chunk_id in self._reserved_ids
            if reserved:
                group_id = f"event-{reserved_index}"
                reserved_index += 1
            else:
                group_id = f"support-{len(rows)}"
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "group_id": group_id,
                    "group_role": "representative" if reserved else "support",
                    "coverage_reserved": reserved,
                    "reservation_basis": (
                        "canonical_fixed_frontier" if reserved else None
                    ),
                    "role_match": True,
                    "temporal_in_scope": True,
                }
            )
        reserved_rows = [row for row in rows if row["coverage_reserved"]]
        if self._trace_case == "neural_basis":
            reserved_rows[0]["reservation_basis"] = "neural_credible"
        elif self._trace_case == "duplicate_group":
            reserved_rows[1]["group_id"] = reserved_rows[0]["group_id"]
        elif self._trace_case == "explicit_temporal_false":
            reserved_rows[0]["temporal_in_scope"] = False
        elif self._trace_case == "explicit_role_false":
            reserved_rows[0]["role_match"] = False
        self.last_candidate_trace = rows

        by_id = {value.chunk.chunk_id: value for value in values}
        returned = [
            by_id[result.chunk.chunk_id]
            for result in self._reserved
            if result.chunk.chunk_id in by_id
        ]
        returned.extend(
            value
            for value in values
            if value.chunk.chunk_id not in self._reserved_ids
        )
        if self._output_case == "injected":
            returned.append(_result("untrusted injection", source_id="attacker"))
        elif self._output_case == "omit_reserved":
            returned = [
                value
                for value in returned
                if value.chunk.chunk_id != reserved_rows[0]["chunk_id"]
            ]
        report = {
            "operator": "fixed_cardinality",
            "cardinality": len(self._reserved_ids),
            "requires_completeness": True,
            "input_candidates": len(values),
            "inspected_candidates": len(values),
            "classified_candidates": len(values),
            "output_candidates": len(returned),
            "selection_status": "applied",
            "bypass_reason": "",
            "fallback_reason": "",
            "score_provider_fallback": "",
            "quantifier": "fixed_cardinality",
            "ordering": "ascending",
            "frontier_candidates": len(values),
            "frontier_attempted": len(values),
            "frontier_uninspected": 0,
            "routed_frontier_exhaustive": True,
            "active_partition_total": len(values),
            "active_partition_inspected": len(values),
            "active_partition_exhaustive": True,
            "active_partition_sources_total": max(1, len(self._reserved_ids)),
            "active_partition_structural_rows": len(self._reserved_ids),
            "active_partition_structural_hypotheses": len(self._reserved_ids),
            "active_partition_scan_contract": (
                "canonical_venue_episode_aligned_v1"
            ),
            "active_partition_semantically_complete": True,
            "partition_scope_kind": "global",
            "partition_inventory_total": 3,
            "selected_partition_count": 3,
            "partition_scope_exhaustive": True,
            "selected_scope_structurally_complete": True,
            "global_semantic_complete": True,
            "allow_selected_scope_fixed_k_closure": (
                self.allow_selected_scope_fixed_k_closure
            ),
            "active_partition_candidates_truncated": 0,
            "active_partition_structural_overflow": 0,
            "reserved_representatives": len(self._reserved_ids),
            "structural_eligible_clusters": len(self._reserved_ids),
            "structural_reserved_representatives": len(self._reserved_ids),
            "cardinality_deficit": 0,
        }
        report.update(self._report_updates)
        # A new immutable-style report object per call is part of the proof
        # that this diagnostic belongs to the current query.
        self.last_report = SimpleNamespace(**report)
        return returned


def _fixed_closure_candidates():
    reserved = [
        _result(
            f"Museum {index} is the exact visited venue. "
            + "The raw turn contains complete identifying evidence. " * 3,
            source_id=f"museum-{index}",
        )
        for index in range(3)
    ]
    distractors = [
        _result(
            "Modern Art Gallery was only a recommendation, not a visit.",
            source_id="distractor",
        ),
        _result(
            "A duplicate recap mentions Museum 1 without new evidence.",
            source_id="recap",
        ),
    ]
    source_metadata = {
        f"museum-{index}": (
            f"[museum-{index} took place at 2024/07/{index + 1:02d} "
            "(Mon) 08:30 Pacific Daylight Time]"
        )
        for index in range(3)
    }
    source_metadata.update(
        {
            "distractor": (
                "[distractor took place at 2024/07/04 (Thu) 08:30 "
                "Pacific Daylight Time]"
            ),
            "recap": (
                "[recap took place at 2024/07/05 (Fri) 08:30 "
                "Pacific Daylight Time]"
            ),
        }
    )
    return reserved, distractors, source_metadata


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
        singleton_reason = packer.last_expansion_trace[0]["cutoff_reason"]
        multi_fact = packer.pack(
            expansions=[marginal],
            user_text="List all items in order.",
        )

        assert singleton.expansion_chunk_ids == []
        assert singleton_reason == "preselector_information_gain_filtered"
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

    def test_source_timestamp_is_bound_to_companion_evidence(self):
        budget = ContextBudget(source_metadata_expansions=True)
        packed = ContextPacker(budget).pack(
            expansions=[
                _result(
                    "[session-a took place at 2023/06/28 (Wed) 20:26]",
                    source_id="session-a",
                ),
                _result("I now have 1,300 followers.", source_id="session-a"),
            ]
        )

        assert packed.expansions == [
            "[1 @ 2023/06/28 (Wed) 20:26] I now have 1,300 followers."
        ]

    def test_persisted_source_metadata_binds_without_timestamp_candidate(self):
        budget = ContextBudget(source_metadata_expansions=True)
        packed = ContextPacker(budget).pack(
            expansions=[
                _result("I now have 1,300 followers.", source_id="session-a")
            ],
            source_metadata={
                "session-a": "[session-a took place at 2023/06/28 (Wed) 20:26]"
            },
        )

        assert packed.expansions == [
            "[1 @ 2023/06/28 (Wed) 20:26] I now have 1,300 followers."
        ]

    def test_source_provenance_preserves_speaker_role(self):
        result = _result(
            "I take classes at Serenity Yoga.", source_id="session-a"
        ).model_copy(
            update={
                "turn": Turn(
                    turn_id="t1",
                    source_id="session-a",
                    role="user",
                    text="I take classes at Serenity Yoga.",
                )
            }
        )
        packed = ContextPacker(
            ContextBudget(source_metadata_expansions=True)
        ).pack(
            expansions=[result],
            source_metadata={
                "session-a": "[session-a took place at 2023/06/28 (Wed) 20:26]"
            },
        )

        assert packed.expansions == [
            "[1 @ 2023/06/28 (Wed) 20:26 | user] "
            "I take classes at Serenity Yoga."
        ]

    def test_unbound_timestamp_remains_available_for_date_questions(self):
        budget = ContextBudget(source_metadata_expansions=True)
        timestamp = "[session-a took place at 2023/06/28 (Wed) 20:26]"
        packed = ContextPacker(budget).pack(
            expansions=[_result(timestamp, source_id="session-a")]
        )

        assert timestamp in packed.expansions[0]

    def test_persisted_orphan_timestamp_does_not_compete_with_evidence(self):
        budget = ContextBudget(
            source_metadata_expansions=True,
            information_gain_expansions=True,
            min_information_gain_per_token=0.0,
        )
        timestamp = "[session-a took place at 2023/06/28 (Wed) 20:26]"
        packed = ContextPacker(budget).pack(
            expansions=[
                _result(timestamp, source_id="session-a"),
                _result("I take yoga classes at Serenity Yoga.", source_id="session-b"),
            ],
            source_metadata={"session-a": timestamp},
        )

        assert packed.expansions == [
            "[1] I take yoga classes at Serenity Yoga."
        ]

    def test_selected_timestamp_promotes_content_from_candidate_pool(self):
        packer = ContextPacker(ContextBudget(source_metadata_expansions=True))
        timestamp = _result(
            "[session-a took place at 2023/06/28 (Wed) 20:26]",
            source_id="session-a",
        )
        content = _result(
            "I now have 1,300 followers.",
            source_id="session-a",
        )

        metadata, evidence = packer._bind_source_metadata(
            [timestamp],
            candidate_pool=[timestamp, content],
        )

        assert metadata == {"session-a": "2023/06/28 (Wed) 20:26"}
        assert evidence == [content]

    def test_source_metadata_respects_the_expansion_token_ceiling(self):
        budget = ContextBudget(
            expansion_tokens=25,
            max_expansion_tokens=20,
            source_metadata_expansions=True,
        )
        packed = ContextPacker(budget).pack(
            expansions=[
                _result(
                    "[session-a took place at 2023/06/28 (Wed) 20:26]",
                    source_id="session-a",
                ),
                _result("detail " * 100, source_id="session-a"),
            ]
        )

        assert packed.token_counts["expansions"] <= 25

    def test_records_ids_only_for_expansions_that_reach_the_prompt(self):
        results = [_result(f"excerpt {index}") for index in range(3)]
        packed = ContextPacker(ContextBudget(max_expansions=2)).pack(
            expansions=results
        )
        assert packed.expansion_chunk_ids == [
            results[0].chunk.chunk_id,
            results[1].chunk.chunk_id,
        ]

    def test_trace_exposes_selector_group_ranks_and_packing_cutoff(self):
        results = [
            _result(f"museum visit {index}", source_id=f"event-{index}")
            for index in range(3)
        ]

        class TracedSelector:
            requires_baseline_ranking = False

            def __init__(self):
                self.last_candidate_trace = []

            def select(
                self,
                query,
                candidates,
                *,
                max_results=None,
                source_timestamps=None,
            ):
                del query, max_results, source_timestamps
                self.last_candidate_trace = [
                    {
                        "chunk_id": candidate.chunk.chunk_id,
                        "source_id": candidate.memory_source_id,
                        "selector_input_rank": rank,
                        "group_id": "event-1",
                        "group_role": (
                            "representative" if rank == 2 else "support"
                        ),
                        "representative_chunk_id": candidates[1].chunk.chunk_id,
                        "merge_similarity": 0.91,
                        "merge_threshold": 0.85,
                        "reservation_basis": (
                            "canonical_fixed_frontier" if rank == 2 else None
                        ),
                    }
                    for rank, candidate in enumerate(candidates, start=1)
                ]
                return [candidates[1], candidates[0], candidates[2]]

        selector = TracedSelector()
        packer = ContextPacker(
            ContextBudget(max_expansions=1),
            expansion_selector=selector,
        )

        packed = packer.pack(expansions=results, user_text="Which museums?")

        assert packed.expansion_chunk_ids == [results[1].chunk.chunk_id]
        trace = {
            row["chunk_id"]: row for row in packer.last_expansion_trace
        }
        expected = {
            "chunk_id": results[0].chunk.chunk_id,
            "source_id": "event-0",
            "route": "",
            "original_rank": 1,
            "selector_input_rank": 1,
            "post_selector_rank": 2,
            "packed_rank": None,
            "cutoff_reason": "direct_count_cap",
            "chunk_tokens": results[0].chunk.token_count,
            "rendered_tokens": None,
            "cumulative_tokens": None,
            "group_id": "event-1",
            "group_role": "support",
            "representative_chunk_id": results[1].chunk.chunk_id,
            "merge_similarity": 0.91,
            "merge_threshold": 0.85,
            "qk_score": None,
            "ov_transport": None,
            "prefix_utility": None,
        }
        assert {
            key: trace[results[0].chunk.chunk_id][key] for key in expected
        } == expected
        representative = trace[results[1].chunk.chunk_id]
        assert representative["original_rank"] == 2
        assert representative["post_selector_rank"] == 1
        assert representative["packed_rank"] == 1
        assert representative["cutoff_reason"] == "packed"
        assert representative["group_role"] == "representative"
        assert representative["reservation_basis"] == (
            "canonical_fixed_frontier"
        )
        assert all("text" not in row for row in packer.last_expansion_trace)

    def test_trace_uses_first_route_rank_and_unique_selector_rank(self):
        first = _result("first", source_id="source-a")
        second = _result("second", source_id="source-b")

        class DeduplicatingSelector:
            requires_baseline_ranking = False
            last_candidate_trace = []

            def select(self, query, candidates, **kwargs):
                del query, kwargs
                return [candidates[0], candidates[1]]

        packer = ContextPacker(expansion_selector=DeduplicatingSelector())

        packer.pack(expansions=[first, second, first])

        trace = {
            row["chunk_id"]: row for row in packer.last_expansion_trace
        }
        assert trace[first.chunk.chunk_id]["original_rank"] == 1
        assert trace[first.chunk.chunk_id]["selector_input_rank"] == 1
        assert trace[second.chunk.chunk_id]["original_rank"] == 2
        assert trace[second.chunk.chunk_id]["selector_input_rank"] == 2

    def test_coverage_reservation_packs_every_credible_representative_first(self):
        first = _result("alpha " * 100, source_id="source-a")
        support = _result("duplicate support " * 100, source_id="source-a")
        second = _result("beta " * 100, source_id="source-b")

        class CoverageSelector:
            requires_baseline_ranking = True
            requires_complete_frontier = True

            def __init__(self):
                self.last_candidate_trace = []

            def select(self, query, candidates, **kwargs):
                del query, kwargs
                by_id = {
                    candidate.chunk.chunk_id: candidate for candidate in candidates
                }
                self.last_candidate_trace = [
                    {
                        "chunk_id": support.chunk.chunk_id,
                        "group_id": "event-1",
                        "group_role": "support",
                        "coverage_reserved": False,
                    },
                    {
                        "chunk_id": first.chunk.chunk_id,
                        "group_id": "event-1",
                        "group_role": "representative",
                        "coverage_reserved": True,
                    },
                    {
                        "chunk_id": second.chunk.chunk_id,
                        "group_id": "event-2",
                        "group_role": "representative",
                        "coverage_reserved": True,
                    },
                ]
                # Deliberately interleave support; the packer must enforce the
                # reservation contract rather than trust incidental ordering.
                return [
                    by_id[support.chunk.chunk_id],
                    by_id[first.chunk.chunk_id],
                    by_id[second.chunk.chunk_id],
                ]

        class NoPrefilterPacker(ContextPacker):
            def _information_gain_order(self, expansions, *, query):
                raise AssertionError(
                    "complete-frontier selector must run before IG filtering"
                )

        packer = NoPrefilterPacker(
            ContextBudget(
                expansion_tokens=64,
                max_expansion_tokens=20,
                max_expansions=1,
                information_gain_expansions=True,
            ),
            expansion_selector=CoverageSelector(),
        )

        packed = packer.pack(
            expansions=[first, support, second],
            user_text="List all events",
        )

        assert packed.expansion_chunk_ids == [
            first.chunk.chunk_id,
            second.chunk.chunk_id,
        ]
        assert packed.token_counts["expansions"] <= 64
        trace = {
            row["chunk_id"]: row for row in packer.last_expansion_trace
        }
        assert trace[first.chunk.chunk_id]["coverage_pack_rank"] == 1
        assert trace[second.chunk.chunk_id]["coverage_pack_rank"] == 2
        assert trace[first.chunk.chunk_id]["coverage_reservation_feasible"] is True
        assert trace[second.chunk.chunk_id]["cutoff_reason"] == "packed"
        assert trace[support.chunk.chunk_id]["cutoff_reason"] == (
            "direct_count_cap"
        )

    def test_performance_reservation_uses_raw_primary_not_pruned_later_recap(self):
        primary = _result(
            "I attended a music festival in Brooklyn with close friends and "
            "heard several favorite indie bands. Concert chronology notes "
            "cover the past two months in earliest order.",
            source_id="brooklyn-source",
        )
        later = _result(
            "Concerts musical events attended past two months chronological "
            "order earliest notes. I later recapped attending the music "
            "festival in Brooklyn.",
            source_id="brooklyn-source",
        )
        query = (
            "List all concerts and musical events I attended in the past two "
            "months in chronological order"
        )

        class PerformanceSelector:
            requires_baseline_ranking = True
            requires_complete_frontier = True

            def __init__(self):
                self.last_candidate_trace = []

            def select(self, _query, candidates, **_kwargs):
                by_id = {
                    candidate.chunk.chunk_id: candidate
                    for candidate in candidates
                }
                self.last_candidate_trace = [
                    {
                        "chunk_id": later.chunk.chunk_id,
                        "group_role": "support",
                        "coverage_reserved": False,
                    },
                    {
                        "chunk_id": primary.chunk.chunk_id,
                        "group_role": "representative",
                        "coverage_reserved": True,
                        "reservation_basis": "direct_performance_frontier",
                    },
                ]
                return [
                    by_id[later.chunk.chunk_id],
                    by_id[primary.chunk.chunk_id],
                ]

        packer = ContextPacker(
            ContextBudget(
                expansion_tokens=160,
                max_expansion_tokens=40,
                max_expansions=2,
                min_coverage_expansion_tokens=24,
                query_aware_sentence_expansions=True,
                max_sentences_per_expansion=1,
            ),
            expansion_selector=PerformanceSelector(),
        )
        assert "Brooklyn" not in packer._prepare_expansion_text(
            later.chunk.text,
            query,
        )

        packed = packer.pack(
            expansions=[later, primary],
            user_text=query,
        )

        assert packed.expansion_chunk_ids[0] == primary.chunk.chunk_id
        assert "Brooklyn" in packed.expansions[0]
        if later.chunk.chunk_id in packed.expansion_chunk_ids:
            later_index = packed.expansion_chunk_ids.index(later.chunk.chunk_id)
            assert "Brooklyn" not in packed.expansions[later_index]
        trace = {
            row["chunk_id"]: row for row in packer.last_expansion_trace
        }
        assert trace[primary.chunk.chunk_id]["coverage_reservation_active"] is True
        assert trace[primary.chunk.chunk_id]["reservation_basis"] == (
            "direct_performance_frontier"
        )

    def test_six_reservations_keep_raw_body_floor_after_long_provenance(self):
        candidates, source_metadata, timestamps = _reserved_museum_results()
        floor = 24
        minimum_snippets = [
            truncate_to_tokens(candidate.chunk.text.strip(), floor)
            for candidate in candidates
        ]
        minimum_budget = count_tokens("Supporting excerpts:") + sum(
            count_tokens(
                f"[{ordinal} @ {timestamp} | user] " + snippet
            )
            + 1
            for ordinal, (timestamp, snippet) in enumerate(
                zip(timestamps, minimum_snippets, strict=True),
                start=1,
            )
        )
        packer = ContextPacker(
            ContextBudget(
                expansion_tokens=minimum_budget,
                max_expansion_tokens=40,
                min_coverage_expansion_tokens=floor,
                max_expansions=1,
                query_aware_sentence_expansions=True,
                max_sentences_per_expansion=1,
                source_metadata_expansions=True,
            ),
            expansion_selector=_AllReservedSelector(),
        )
        query = "List all museum visits"

        # This is the regression precondition: ordinary sentence packing is
        # intentionally shorter than the reservation's useful-content floor.
        assert all(
            count_tokens(
                packer._prepare_expansion_text(candidate.chunk.text, query)
            )
            < floor
            for candidate in candidates
        )

        packed = packer.pack(
            expansions=candidates,
            user_text=query,
            source_metadata=source_metadata,
        )

        assert packed.expansion_chunk_ids == [
            candidate.chunk.chunk_id for candidate in candidates
        ]
        assert packed.token_counts["expansions"] <= minimum_budget
        assert all(
            packed.expansion_source_token_counts[f"museum-session-{index}"]
            >= floor
            for index in range(6)
        )
        trace = {
            row["chunk_id"]: row for row in packer.last_expansion_trace
        }
        assert all(
            trace[candidate.chunk.chunk_id]["content_tokens"] >= floor
            and trace[candidate.chunk.chunk_id]["rendered_tokens"]
            > trace[candidate.chunk.chunk_id]["content_tokens"]
            and trace[candidate.chunk.chunk_id]["coverage_reservation_active"]
            is True
            for candidate in candidates
        )

    def test_long_provenance_reservations_degrade_to_same_feasible_prefix(self):
        candidates, source_metadata, timestamps = _reserved_museum_results()
        floor = 24
        minimum_snippets = [
            truncate_to_tokens(candidate.chunk.text.strip(), floor)
            for candidate in candidates
        ]
        # Fund exactly three raw-body floors.  A fourth reservation must not
        # steal content from those earlier representatives.
        prefix_budget = count_tokens("Supporting excerpts:") + sum(
            count_tokens(
                f"[{ordinal} @ {timestamp} | user] " + snippet
            )
            + 1
            for ordinal, (timestamp, snippet) in enumerate(
                zip(timestamps[:3], minimum_snippets[:3], strict=True),
                start=1,
            )
        )

        def run_once():
            packer = ContextPacker(
                ContextBudget(
                    expansion_tokens=prefix_budget,
                    max_expansion_tokens=40,
                    min_coverage_expansion_tokens=floor,
                    max_expansions=1,
                    query_aware_sentence_expansions=True,
                    max_sentences_per_expansion=1,
                    source_metadata_expansions=True,
                ),
                expansion_selector=_AllReservedSelector(),
            )
            packed = packer.pack(
                expansions=candidates,
                user_text="List all museum visits",
                source_metadata=source_metadata,
            )
            trace = {
                row["chunk_id"]: row for row in packer.last_expansion_trace
            }
            active = [
                candidate.chunk.chunk_id
                for candidate in candidates
                if trace[candidate.chunk.chunk_id][
                    "coverage_reservation_active"
                ]
            ]
            degraded = [
                candidate.chunk.chunk_id
                for candidate in candidates
                if trace[candidate.chunk.chunk_id][
                    "coverage_reservation_degraded"
                ]
            ]
            return packer, packed, active, degraded

        first = run_once()
        second = run_once()
        expected_active = [
            candidate.chunk.chunk_id for candidate in candidates[:3]
        ]
        expected_degraded = [
            candidate.chunk.chunk_id for candidate in candidates[3:]
        ]
        assert first[2:] == second[2:]
        assert first[2] == expected_active
        assert first[3] == expected_degraded
        assert first[1].expansion_chunk_ids == expected_active
        assert first[1].token_counts["expansions"] <= prefix_budget
        first_trace = {
            row["chunk_id"]: row
            for row in first[0].last_expansion_trace
        }
        assert all(
            first_trace[chunk_id]["content_tokens"] == floor
            for chunk_id in expected_active
        )

    def test_typed_fixed_full_body_coverage_closes_distractor_tail(self):
        reserved, distractors, source_metadata = _fixed_closure_candidates()
        selector = _FixedClosureSelector(reserved)
        packer = ContextPacker(
            ContextBudget(
                expansion_tokens=800,
                max_expansion_tokens=160,
                max_expansions=10,
                source_metadata_expansions=True,
                query_aware_sentence_expansions=True,
                max_sentences_per_expansion=1,
            ),
            expansion_selector=selector,
        )

        packed = packer.pack(
            # The routed frontier is deliberately not chronological. The
            # selector's trusted output order, not trace iteration order,
            # determines the exact closed packet.
            expansions=[
                reserved[2],
                distractors[0],
                reserved[0],
                reserved[1],
                distractors[1],
            ],
            user_text=(
                "What is the order of the three museums I visited from "
                "earliest to latest?"
            ),
            source_metadata=source_metadata,
        )

        assert packed.expansion_chunk_ids == [
            result.chunk.chunk_id for result in reserved
        ]
        trace = {
            row["chunk_id"]: row for row in packer.last_expansion_trace
        }
        assert all(
            trace[result.chunk.chunk_id]["cutoff_reason"] == "packed"
            and trace[result.chunk.chunk_id][
                "post_coverage_closure_applied"
            ]
            is True
            and trace[result.chunk.chunk_id]["content_tokens"]
            == count_tokens(result.chunk.text.strip())
            for result in reserved
        )
        assert all(
            trace[result.chunk.chunk_id]["cutoff_reason"]
            == "post_coverage_closed"
            and trace[result.chunk.chunk_id]["post_coverage_closed"] is True
            and trace[result.chunk.chunk_id]["packed_rank"] is None
            for result in distractors
        )
        assert packer.last_closure_report["applied"] is True
        assert packer.last_closure_report["closure_scope"] == "global_semantic"
        assert (
            packer.last_closure_report["closure_global_recall_guaranteed"]
            is True
        )
        assert packer.last_closure_report["partition_scope_kind"] == "global"
        assert all(
            row["closure_scope"] == "global_semantic"
            and row["closure_global_recall_guaranteed"] is True
            for row in trace.values()
        )

    def test_selected_partition_scope_stays_fail_open_without_policy_opt_in(self):
        reserved, distractors, source_metadata = _fixed_closure_candidates()
        selector = _FixedClosureSelector(
            reserved,
            report_updates={
                "partition_scope_kind": "approximate_top_k",
                "partition_inventory_total": 40,
                "selected_partition_count": 4,
                "partition_scope_exhaustive": False,
                "global_semantic_complete": False,
            },
        )
        packer = ContextPacker(
            ContextBudget(
                expansion_tokens=800,
                max_expansion_tokens=160,
                max_expansions=10,
                source_metadata_expansions=True,
            ),
            expansion_selector=selector,
        )

        packed = packer.pack(
            expansions=[*reserved, *distractors],
            user_text=(
                "What is the order of the three museums I visited from "
                "earliest to latest?"
            ),
            source_metadata=source_metadata,
        )

        assert {result.chunk.chunk_id for result in distractors}.issubset(
            packed.expansion_chunk_ids
        )
        assert packer.last_closure_report["applied"] is False
        assert all(
            row["closure_scope"] == ""
            and row["closure_global_recall_guaranteed"] is False
            for row in packer.last_expansion_trace
        )

    def test_selected_partition_scope_opt_in_closes_without_global_claim(self):
        reserved, distractors, source_metadata = _fixed_closure_candidates()
        selector = _FixedClosureSelector(
            reserved,
            report_updates={
                "partition_scope_kind": "approximate_top_k",
                "partition_inventory_total": 40,
                "selected_partition_count": 4,
                "partition_scope_exhaustive": False,
                "global_semantic_complete": False,
            },
            allow_selected_scope_fixed_k_closure=True,
        )
        packer = ContextPacker(
            ContextBudget(
                expansion_tokens=800,
                max_expansion_tokens=160,
                max_expansions=10,
                source_metadata_expansions=True,
            ),
            expansion_selector=selector,
        )

        packed = packer.pack(
            expansions=[*reserved, *distractors],
            user_text=(
                "What is the order of the three museums I visited from "
                "earliest to latest?"
            ),
            source_metadata=source_metadata,
        )

        assert packed.expansion_chunk_ids == [
            result.chunk.chunk_id for result in reserved
        ]
        assert packer.last_closure_report["applied"] is True
        assert (
            packer.last_closure_report["closure_scope"]
            == "selected_scope_policy"
        )
        assert (
            packer.last_closure_report["closure_global_recall_guaranteed"]
            is False
        )
        assert (
            packer.last_closure_report["partition_scope_kind"]
            == "approximate_top_k"
        )
        assert all(
            row["closure_scope"] == "selected_scope_policy"
            and row["closure_global_recall_guaranteed"] is False
            for row in packer.last_expansion_trace
        )

    @pytest.mark.parametrize(
        ("case", "report_updates", "trace_case", "output_case"),
        [
            ("not_applied", {"selection_status": "fallback"}, "", ""),
            ("not_fixed", {"quantifier": "all"}, "", ""),
            (
                "routed_incomplete",
                {"routed_frontier_exhaustive": False},
                "",
                "",
            ),
            ("uninspected", {"frontier_uninspected": 1}, "", ""),
            (
                "active_incomplete",
                {"active_partition_exhaustive": False},
                "",
                "",
            ),
            (
                "active_unknown",
                {
                    "active_partition_total": None,
                    "active_partition_inspected": None,
                    "active_partition_exhaustive": None,
                },
                "",
                "",
            ),
            (
                "semantic_incomplete",
                {"active_partition_semantically_complete": False},
                "",
                "",
            ),
            (
                "selected_scope_incomplete",
                {"selected_scope_structurally_complete": False},
                "",
                "",
            ),
            (
                "global_semantic_unproven",
                {"global_semantic_complete": None},
                "",
                "",
            ),
            (
                "partition_scope_count_mismatch",
                {"selected_partition_count": 2},
                "",
                "",
            ),
            (
                "partition_scope_inventory_missing",
                {"partition_inventory_total": None},
                "",
                "",
            ),
            (
                "scan_contract_missing",
                {"active_partition_scan_contract": ""},
                "",
                "",
            ),
            (
                "structural_hypothesis_mismatch",
                {"active_partition_structural_hypotheses": 2},
                "",
                "",
            ),
            (
                "structural_rows_too_small",
                {"active_partition_structural_rows": 2},
                "",
                "",
            ),
            (
                "active_sources_missing",
                {"active_partition_sources_total": 0},
                "",
                "",
            ),
            (
                "structural_overflow",
                {
                    "structural_eligible_clusters": 4,
                    "active_partition_structural_overflow": 1,
                    "active_partition_semantically_complete": False,
                },
                "",
                "",
            ),
            (
                "scan_truncated",
                {
                    "active_partition_candidates_truncated": 1,
                    "active_partition_semantically_complete": False,
                },
                "",
                "",
            ),
            (
                "active_count_mismatch",
                {
                    "active_partition_total": 6,
                    "active_partition_inspected": 5,
                },
                "",
                "",
            ),
            ("cardinality_deficit", {"cardinality_deficit": 1}, "", ""),
            (
                "structural_count_mismatch",
                {"structural_reserved_representatives": 2},
                "",
                "",
            ),
            ("untyped_basis", {}, "neural_basis", ""),
            ("duplicate_structural_group", {}, "duplicate_group", ""),
            ("temporal_rejection", {}, "explicit_temporal_false", ""),
            ("role_rejection", {}, "explicit_role_false", ""),
            (
                "provider_fallback",
                {"score_provider_fallback": "provider unavailable"},
                "",
                "",
            ),
            ("rejected_selector_output", {}, "", "injected"),
            ("omitted_reservation", {}, "", "omit_reserved"),
            ("truncated_raw_body", {}, "", ""),
            ("inactive_reservation", {}, "", ""),
            ("invalid_timestamp", {}, "", ""),
            ("out_of_order_timestamp", {}, "", ""),
        ],
    )
    def test_post_coverage_closure_fails_open_on_any_unproven_gate(
        self,
        case,
        report_updates,
        trace_case,
        output_case,
    ):
        reserved, distractors, source_metadata = _fixed_closure_candidates()
        selector = _FixedClosureSelector(
            reserved,
            report_updates=report_updates,
            trace_case=trace_case,
            output_case=output_case,
        )
        max_expansion_tokens = 160
        expansion_tokens = 800
        if case == "truncated_raw_body":
            max_expansion_tokens = 24
        elif case == "inactive_reservation":
            expansion_tokens = 35
        elif case == "invalid_timestamp":
            source_metadata["museum-1"] = (
                "[museum-1 took place at sometime last summer]"
            )
        elif case == "out_of_order_timestamp":
            source_metadata["museum-1"] = (
                "[museum-1 took place at 2024/07/09 (Tue) 08:30 "
                "Pacific Daylight Time]"
            )
        packer = ContextPacker(
            ContextBudget(
                expansion_tokens=expansion_tokens,
                max_expansion_tokens=max_expansion_tokens,
                max_expansions=10,
                source_metadata_expansions=True,
            ),
            expansion_selector=selector,
        )

        packer.pack(
            expansions=[*reserved, *distractors],
            user_text=(
                "What is the order of the three museums I visited from "
                "earliest to latest?"
            ),
            source_metadata=source_metadata,
        )

        trace = packer.last_expansion_trace
        assert trace
        assert all(
            row["post_coverage_closure_applied"] is False for row in trace
        )
        assert not any(
            row["cutoff_reason"] == "post_coverage_closed" for row in trace
        )

    def test_complete_frontier_bypass_is_query_dependent(self):
        candidates = [
            _result(f"candidate {index}", source_id=f"source-{index}")
            for index in range(3)
        ]
        observed: list[list[str]] = []

        class QueryAwareSelector:
            requires_baseline_ranking = True
            requires_complete_frontier = True

            def __init__(self):
                self.last_candidate_trace = []

            @staticmethod
            def requires_complete_frontier_for(query):
                return query.startswith("List all")

            def select(self, query, values, **kwargs):
                del query, kwargs
                observed.append([value.chunk.chunk_id for value in values])
                return list(values)

        class FilteringPacker(ContextPacker):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.ig_calls = 0

            def _information_gain_order(self, expansions, *, query):
                del query
                self.ig_calls += 1
                return list(expansions[-1:])

        packer = FilteringPacker(
            ContextBudget(information_gain_expansions=True),
            expansion_selector=QueryAwareSelector(),
        )

        packer.pack(expansions=candidates, user_text="Where did I go?")
        packer.pack(expansions=candidates, user_text="List all places I visited")

        assert packer.ig_calls == 1
        assert observed == [
            [candidates[-1].chunk.chunk_id],
            [candidate.chunk.chunk_id for candidate in candidates],
        ]

    def test_derived_duration_suppresses_only_proven_conflicting_recap(self):
        def user_result(
            chunk_id: str,
            text: str,
            source_id: str,
        ) -> RetrievalResult:
            result = _result(text, source_id=source_id)
            turn = Turn(
                turn_id=f"turn-{chunk_id}",
                source_id=source_id,
                role="user",
                text=text,
            )
            return result.model_copy(
                update={
                    "chunk": result.chunk.model_copy(
                        update={
                            "chunk_id": chunk_id,
                            "turn_id": turn.turn_id,
                        }
                    ),
                    "turn": turn,
                }
            )

        onset = user_result(
            "onset",
            "I just started attending the workshop today.",
            "onset-source",
        )
        recap = user_result(
            "recap",
            "I've been attending the workshop for about 6 weeks now.",
            "recap-source",
        )
        endpoint = user_result(
            "endpoint",
            "I bought my workshop equipment today.",
            "endpoint-source",
        )
        candidates = [onset, recap, endpoint]
        source_metadata = {
            "onset-source": (
                "[onset-source took place at 2024/01/01 (Mon) 09:00]"
            ),
            "recap-source": (
                "[recap-source took place at 2024/01/01 (Mon) 12:00]"
            ),
            "endpoint-source": (
                "[endpoint-source took place at 2024/01/22 (Mon) 09:00]"
            ),
        }
        packer = ContextPacker(
            ContextBudget(
                expansion_tokens=1000,
                max_expansions=10,
                source_metadata_expansions=True,
            )
        )

        packed = packer.pack(
            expansions=candidates,
            user_text=(
                "How many weeks had I been attending the workshop when I "
                "bought my equipment?"
            ),
            source_metadata=source_metadata,
        )

        assert packed.expansion_chunk_ids == ["onset", "endpoint"]
        assert packed.token_counts["expansions"] <= 1000
        trace = {
            row["chunk_id"]: row for row in packer.last_expansion_trace
        }
        assert trace["recap"]["cutoff_reason"] == (
            "temporal_conflict_suppressed"
        )
        assert trace["recap"]["temporal_conflict_action"] == "suppressed"
        assert trace["recap"]["temporal_conflict_basis"] == (
            "approximate_duration_conflicts_with_explicit_onset"
        )
        assert trace["recap"]["temporal_onset_chunk_id"] == "onset"
        assert trace["recap"]["temporal_endpoint_chunk_id"] == "endpoint"
        assert trace["onset"]["cutoff_reason"] == "packed"
        assert trace["endpoint"]["cutoff_reason"] == "packed"

    def test_explicit_active_partition_counts_reach_capable_selector(self):
        candidate = _result("one routed member", source_id="source-a")
        observed = {}

        class PartitionAwareSelector:
            requires_baseline_ranking = False
            last_candidate_trace = []

            def select(
                self,
                query,
                values,
                *,
                source_timestamps=None,
                active_partition_total=None,
                active_partition_inspected=None,
            ):
                del query, source_timestamps
                observed["total"] = active_partition_total
                observed["inspected"] = active_partition_inspected
                return list(values)

        ContextPacker(expansion_selector=PartitionAwareSelector()).pack(
            expansions=[candidate],
            user_text="List all members",
            active_partition_total=3,
            active_partition_inspected=1,
        )

        assert observed == {"total": 3, "inspected": 1}

    def test_typed_active_partition_scan_reaches_capable_selector(self):
        candidate = _result("one typed member", source_id="source-a")
        observed = {}

        class ScanAwareSelector:
            requires_baseline_ranking = False
            last_candidate_trace = []

            def select(
                self,
                query,
                values,
                *,
                source_timestamps=None,
                active_partition_total=None,
                active_partition_inspected=None,
                active_partition_scan=None,
            ):
                del query, source_timestamps
                observed.update(
                    total=active_partition_total,
                    inspected=active_partition_inspected,
                    scan=dict(active_partition_scan or {}),
                )
                return list(values)

        scan = {
            "active_partition_total": 3163,
            "active_partition_inspected": 3163,
            "active_partition_exhaustive": True,
            "active_partition_structural_hypotheses": 6,
            "active_partition_scan_contract": "canonical_primary_event_v1",
            "active_partition_semantically_complete": True,
        }
        ContextPacker(expansion_selector=ScanAwareSelector()).pack(
            expansions=[candidate],
            user_text="Name the six museums I visited",
            active_partition_scan=scan,
        )

        assert observed == {
            "total": 3163,
            "inspected": 3163,
            "scan": scan,
        }

    def test_selector_cannot_inject_or_replace_evidence_and_omissions_fail_open(self):
        first = _result("trusted first", source_id="source-a")
        second = _result("trusted second", source_id="source-b")
        third = _result("trusted third", source_id="source-c")
        fabricated = _result("FABRICATED PAYLOAD", source_id="attacker")
        replacement = first.model_copy(update={"score": 999.0})

        class UnsafeSelector:
            requires_baseline_ranking = False
            strict = False
            last_candidate_trace = []

            def select(self, query, candidates, **kwargs):
                del query, kwargs
                # Only ``second`` is an exact admitted object. ``third`` is
                # omitted and must be appended fail-open after the valid lead.
                return [
                    second,
                    fabricated,
                    replacement,
                    second,
                ]

        packer = ContextPacker(
            ContextBudget(expansion_tokens=100, max_expansions=3),
            expansion_selector=UnsafeSelector(),
        )

        packed = packer.pack(
            expansions=[first, second, third],
            user_text="List all trusted facts",
        )

        assert packed.expansion_chunk_ids == [
            second.chunk.chunk_id,
            first.chunk.chunk_id,
            third.chunk.chunk_id,
        ]
        assert "FABRICATED PAYLOAD" not in "\n".join(
            message["content"] for message in packed.messages
        )
        trace = {
            row["chunk_id"]: row for row in packer.last_expansion_trace
        }
        assert trace[first.chunk.chunk_id]["selector_output_rejection"] == (
            "selector_replacement_rejected"
        )
        assert trace[fabricated.chunk.chunk_id]["cutoff_reason"] == (
            "selector_injected_rejected"
        )
        assert trace[fabricated.chunk.chunk_id]["packed_rank"] is None

    def test_strict_selector_rejects_fabricated_evidence(self):
        trusted = _result("trusted", source_id="source-a")
        fabricated = _result("fabricated", source_id="attacker")

        class StrictUnsafeSelector:
            requires_baseline_ranking = False
            strict = True
            last_candidate_trace = []

            def select(self, query, candidates, **kwargs):
                del query, candidates, kwargs
                return [fabricated]

        packer = ContextPacker(expansion_selector=StrictUnsafeSelector())

        with pytest.raises(ValueError, match="unsafe selector output"):
            packer.pack(expansions=[trusted], user_text="List all facts")

    def test_many_coverage_reservations_degrade_to_a_feasible_prefix(self):
        candidates = [
            _result(
                (f"Event {index} has an explicit identifying answer value. " * 8),
                source_id=f"source-{index}",
            )
            for index in range(12)
        ]

        class ManyCoverageSelector:
            requires_baseline_ranking = True
            requires_complete_frontier = True

            def __init__(self):
                self.last_candidate_trace = []

            def select(self, query, values, **kwargs):
                del query, kwargs
                self.last_candidate_trace = [
                    {
                        "chunk_id": value.chunk.chunk_id,
                        "group_id": f"event-{index}",
                        "group_role": "representative",
                        "coverage_reserved": True,
                    }
                    for index, value in enumerate(values)
                ]
                return list(values)

        packer = ContextPacker(
            ContextBudget(
                expansion_tokens=90,
                max_expansion_tokens=40,
                min_coverage_expansion_tokens=24,
                max_expansions=2,
            ),
            expansion_selector=ManyCoverageSelector(),
        )

        packed = packer.pack(
            expansions=candidates,
            user_text="List all events",
        )

        trace = {
            row["chunk_id"]: row for row in packer.last_expansion_trace
        }
        active = [
            candidate
            for candidate in candidates
            if trace[candidate.chunk.chunk_id].get("coverage_reservation_active")
        ]
        degraded = [
            candidate
            for candidate in candidates
            if trace[candidate.chunk.chunk_id].get("coverage_reservation_degraded")
        ]
        assert 0 < len(active) < len(candidates)
        assert degraded
        assert packed.expansion_chunk_ids == [
            candidate.chunk.chunk_id for candidate in active
        ]
        assert all(
            trace[candidate.chunk.chunk_id]["coverage_content_cap"] >= 24
            and trace[candidate.chunk.chunk_id]["cutoff_reason"] == "packed"
            for candidate in active
        )
        assert all(
            trace[candidate.chunk.chunk_id]["coverage_reservation_feasible"]
            is False
            for candidate in degraded
        )

    def test_infeasible_reservation_degrades_without_zero_evidence_break(self):
        candidates = [
            _result("explicit answer value " * 20, source_id=f"source-{index}")
            for index in range(4)
        ]

        class CoverageSelector:
            requires_baseline_ranking = True
            requires_complete_frontier = True

            def __init__(self):
                self.last_candidate_trace = []

            def select(self, query, values, **kwargs):
                del query, kwargs
                self.last_candidate_trace = [
                    {
                        "chunk_id": value.chunk.chunk_id,
                        "group_role": "representative",
                        "coverage_reserved": True,
                    }
                    for value in values
                ]
                return list(values)

        packer = ContextPacker(
            ContextBudget(
                expansion_tokens=28,
                max_expansion_tokens=40,
                min_coverage_expansion_tokens=24,
                max_expansions=1,
            ),
            expansion_selector=CoverageSelector(),
        )

        packed = packer.pack(expansions=candidates, user_text="List all events")

        assert len(packed.expansion_chunk_ids) == 1
        first_trace = next(
            row
            for row in packer.last_expansion_trace
            if row["chunk_id"] == candidates[0].chunk.chunk_id
        )
        assert first_trace["coverage_reservation_active"] is False
        assert first_trace["cutoff_reason"] == "packed"


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
