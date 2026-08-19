from __future__ import annotations

from memory_condense.associations.head_memory_models import (
    MemoryLinkHit,
    NestedMemoryInspection,
)
from memory_condense.search.selectors.qwen_rerank import QwenCandidateReranker, _qk_ov_cav_utility
from memory_condense.domain.schemas import Chunk, RetrievalResult, Turn


def _result(index: int, *, source_id: str = "source-a") -> RetrievalResult:
    turn = Turn(
        turn_id=f"turn-{index}",
        role="user",
        text=f"candidate {index}",
        source_id=source_id,
    )
    chunk = Chunk(
        chunk_id=f"chunk-{index}",
        turn_id=turn.turn_id,
        text=f"candidate {index} " * 20,
        start_char=0,
        end_char=20,
        token_count=40,
    )
    return RetrievalResult(
        chunk=chunk,
        turn=turn,
        score=1.0 - index / 20.0,
        route="hybrid_source_local",
    )


class FakeNestedLinker:
    max_candidates = 8

    def __init__(self) -> None:
        self.query = ""
        self.groups = []

    def inspect_nested(self, query, groups, *, beam_per_group, top_k, score_mode):
        self.query = query
        self.groups = [list(group) for group in groups]
        assert beam_per_group == 2
        assert top_k == 2
        assert score_mode == "qk_ov"
        # Deliberately select the weakest scalar candidates. This makes the
        # Qwen reserve visible without pretending the fake is a good model.
        candidates = [candidate for group in self.groups for candidate in group]
        hits = tuple(
            MemoryLinkHit(
                episode_id=candidate.episode_id,
                qk_score=float(rank + 1),
                ov_transport=float(rank + 2),
                head_weights=(0.5,),
            )
            for rank, candidate in enumerate(reversed(candidates[-2:]))
        )
        return NestedMemoryInspection(
            hits=hits,
            passes=5,
            max_workspace_candidates=4,
            max_workspace_tokens=321,
            total_candidate_inspections=12,
        )


def test_qwen_reranker_protects_scalar_prefix_and_uses_bounded_reserve():
    linker = FakeNestedLinker()
    reranker = QwenCandidateReranker(
        linker,
        candidate_pool=6,
        qwen_slots=2,
        group_size=2,
        beam_per_group=2,
        candidate_tokens=5,
        query_tokens=4,
    )

    selected = reranker.rerank("one two three four five", [_result(i) for i in range(8)], top_k=4)

    assert [result.chunk.chunk_id for result in selected[:2]] == ["chunk-0", "chunk-1"]
    assert [result.chunk.chunk_id for result in selected[2:]] == ["chunk-5", "chunk-4"]
    assert all(result.route == "qwen_rerank" for result in selected[2:])
    assert len(linker.groups) == 2
    assert max(len(group) for group in linker.groups) == 2
    assert reranker.last_report is not None
    assert reranker.last_report.input_candidates == 6
    assert reranker.last_report.inspected_candidates == 4
    assert reranker.last_report.qwen_candidates_added == 2
    assert reranker.last_report.max_workspace_tokens == 321
    assert reranker.last_report.retained_transformer_state_bytes == 0


def test_qwen_reranker_empty_input_does_not_invoke_linker():
    linker = FakeNestedLinker()
    reranker = QwenCandidateReranker(linker)

    assert reranker.rerank("query", [], top_k=4) == []
    assert linker.groups == []
    assert reranker.last_report is not None
    assert reranker.last_report.output_candidates == 0


def test_qwen_utility_rewards_positive_event_concept_margin():
    plain = MemoryLinkHit(
        episode_id="plain",
        qk_score=0.2,
        ov_transport=0.3,
        head_weights=(1.0,),
    )
    event = MemoryLinkHit(
        episode_id="event",
        qk_score=0.2,
        ov_transport=0.3,
        head_weights=(1.0,),
        metadata={"cav_signature": (0.8,)},
    )

    assert _qk_ov_cav_utility(event) > _qk_ov_cav_utility(plain)


def test_qwen_reranker_can_limit_enumeration_candidates_to_unique_sources():
    linker = FakeNestedLinker()
    reranker = QwenCandidateReranker(
        linker,
        candidate_pool=6,
        qwen_slots=2,
        group_size=2,
        beam_per_group=2,
    )
    candidates = [
        _result(0, source_id="event-a"),
        _result(1, source_id="event-a"),
        _result(2, source_id="event-b"),
        _result(3, source_id="event-c"),
        _result(4, source_id="event-d"),
        _result(5, source_id="event-e"),
    ]

    selected = reranker.rerank(
        "Which events happened?",
        candidates,
        top_k=4,
        unique_sources=True,
    )

    sources = [result.turn.source_id for result in selected]
    assert len(sources) == len(set(sources)) == 4
    assert reranker.last_report is not None
    assert reranker.last_report.input_candidates == 5


def test_qwen_attention_selects_seeds_without_replacing_context():
    linker = FakeNestedLinker()
    reranker = QwenCandidateReranker(
        linker,
        candidate_pool=6,
        qwen_slots=2,
        group_size=2,
        beam_per_group=2,
    )

    seeds = reranker.select("query", [_result(i) for i in range(8)], top_k=2)

    assert [seed.chunk.chunk_id for seed in seeds] == ["chunk-5", "chunk-4"]
    assert all(seed.route == "qwen_attention_seed" for seed in seeds)
    assert reranker.last_report is not None
    assert reranker.last_report.protected_candidates == 0
    assert reranker.last_report.inspected_candidates == 6
