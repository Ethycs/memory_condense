"""Test the eval runner with mocked LLM calls."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from memory_condense.eval.runner import replay_conversation
from memory_condense.eval.schemas import ChunkerConfig, EvalConfig, RetrievalConfig

RESPONDER_PROMPT_TOKENS = 100
RESPONDER_COMPLETION_TOKENS = 20
JUDGE_PROMPT_TOKENS = 50
JUDGE_COMPLETION_TOKENS = 10


def _mock_completion(content: str, prompt_tokens: int, completion_tokens: int):
    mock_choice = MagicMock()
    mock_choice.message.content = content
    return MagicMock(
        choices=[mock_choice],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
    )


def _mock_responder_completion(**kwargs):
    """Create a mock litellm.completion response for the responder."""
    return _mock_completion(
        "Mock generated response",
        RESPONDER_PROMPT_TOKENS,
        RESPONDER_COMPLETION_TOKENS,
    )


def _mock_judge_completion(**kwargs):
    """Create a mock litellm.completion response for the judge."""
    return _mock_completion(
        json.dumps({"score": 4, "reasoning": "Good match"}),
        JUDGE_PROMPT_TOKENS,
        JUDGE_COMPLETION_TOKENS,
    )


@pytest.mark.slow
@patch("memory_condense.eval.responder.litellm")
@patch("memory_condense.eval.judge.litellm")
def test_replay_conversation_basic(
    mock_judge_litellm, mock_resp_litellm, tmp_path
):
    mock_resp_litellm.completion.side_effect = _mock_responder_completion
    mock_judge_litellm.completion.side_effect = _mock_judge_completion

    turns = [
        ("user", "Hello, my name is Alex."),
        ("assistant", "Nice to meet you, Alex!"),
        ("user", "What is Python?"),
        ("assistant", "Python is a programming language."),
    ]

    config = EvalConfig(
        chunker=ChunkerConfig(min_tokens=5, max_tokens=50),
        retrieval=RetrievalConfig(k=3, ef_search=50),
    )

    result = replay_conversation(
        filename="test.txt",
        turns=turns,
        config=config,
        data_dir=tmp_path / "data",
    )

    assert result.filename == "test.txt"
    assert result.num_turns == 4
    # Two user turns -> two scored results
    assert len(result.turn_results) == 2
    assert all(tr.score == 4 for tr in result.turn_results)
    assert result.mean_score == 4.0


@pytest.mark.slow
@patch("memory_condense.eval.responder.litellm")
@patch("memory_condense.eval.judge.litellm")
def test_replay_conversation_records_usage(
    mock_judge_litellm, mock_resp_litellm, tmp_path
):
    mock_resp_litellm.completion.side_effect = _mock_responder_completion
    mock_judge_litellm.completion.side_effect = _mock_judge_completion

    turns = [
        ("user", "Hello, my name is Alex."),
        ("assistant", "Nice to meet you, Alex!"),
        ("user", "What is Python?"),
        ("assistant", "Python is a programming language."),
    ]

    config = EvalConfig(
        chunker=ChunkerConfig(min_tokens=5, max_tokens=50),
        retrieval=RetrievalConfig(k=3, ef_search=50),
    )

    result = replay_conversation("test.txt", turns, config, tmp_path / "data")

    n = len(result.turn_results)
    assert n == 2

    for tr in result.turn_results:
        assert tr.responder_usage.input_tokens == RESPONDER_PROMPT_TOKENS
        assert tr.responder_usage.output_tokens == RESPONDER_COMPLETION_TOKENS
        assert tr.judge_usage.input_tokens == JUDGE_PROMPT_TOKENS
        assert tr.judge_usage.output_tokens == JUDGE_COMPLETION_TOKENS
        assert tr.responder_usage.calls == 1
        assert tr.judge_usage.calls == 1
        assert tr.context_tokens > 0
        assert tr.retrieval_s >= 0.0

    # Conversation-level aggregation
    assert result.usage.calls == 2 * n
    assert result.usage.input_tokens == n * (
        RESPONDER_PROMPT_TOKENS + JUDGE_PROMPT_TOKENS
    )
    assert result.usage.output_tokens == n * (
        RESPONDER_COMPLETION_TOKENS + JUDGE_COMPLETION_TOKENS
    )
    assert result.usage.total_tokens == n * (
        RESPONDER_PROMPT_TOKENS
        + JUDGE_PROMPT_TOKENS
        + RESPONDER_COMPLETION_TOKENS
        + JUDGE_COMPLETION_TOKENS
    )

    # Context grows with conversation depth (memory + recent window)
    assert result.turn_results[1].context_tokens > result.turn_results[0].context_tokens


@pytest.mark.slow
@patch("memory_condense.eval.responder.litellm")
@patch("memory_condense.eval.judge.litellm")
def test_replay_teacher_forces_actual_response(
    mock_judge_litellm, mock_resp_litellm, tmp_path
):
    """The *actual* assistant turn is ingested, never the generated one."""
    mock_resp_litellm.completion.side_effect = _mock_responder_completion
    mock_judge_litellm.completion.side_effect = _mock_judge_completion

    turns = [
        ("user", "Hello, my name is Alex."),
        ("assistant", "Nice to meet you, Alex!"),
        ("user", "What did I say my name was?"),
        ("assistant", "You said your name is Alex."),
    ]

    config = EvalConfig(
        chunker=ChunkerConfig(min_tokens=5, max_tokens=50),
        retrieval=RetrievalConfig(k=3, ef_search=50),
    )

    replay_conversation("test.txt", turns, config, tmp_path / "data")

    # The second responder call sees the recorded assistant turn in its
    # recent-conversation window, not "Mock generated response".
    second_call = mock_resp_litellm.completion.call_args_list[1]
    contents = [m["content"] for m in second_call.kwargs["messages"]]
    assert "Nice to meet you, Alex!" in contents
    assert "Mock generated response" not in contents


@pytest.mark.slow
@patch("memory_condense.eval.responder.litellm")
@patch("memory_condense.eval.judge.litellm")
def test_replay_handles_leading_assistant(
    mock_judge_litellm, mock_resp_litellm, tmp_path
):
    """Test conversations that start with an assistant turn."""
    mock_resp_litellm.completion.side_effect = _mock_responder_completion
    mock_judge_litellm.completion.side_effect = _mock_judge_completion

    turns = [
        ("assistant", "Welcome! How can I help?"),
        ("user", "Tell me about embeddings."),
        ("assistant", "Embeddings are vector representations."),
    ]

    config = EvalConfig(
        chunker=ChunkerConfig(min_tokens=5, max_tokens=50),
        retrieval=RetrievalConfig(k=3),
    )

    result = replay_conversation("test.txt", turns, config, tmp_path / "data")

    # Only one user turn with a following assistant response
    assert len(result.turn_results) == 1


# ---------------------------------------------------------------------------
# Retrieval-mode dispatch. These patch MemoryCondenser itself so no embedding
# model is loaded — they assert only which search method the runner reaches for.
# ---------------------------------------------------------------------------

TWO_TURNS = [
    ("user", "Hello, my name is Alex."),
    ("assistant", "Nice to meet you, Alex!"),
    ("user", "What did I say my name was?"),
    ("assistant", "You said Alex."),
]


def _run_with_mock_condenser(config, tmp_path):
    with patch("memory_condense.eval.runner.MemoryCondenser") as mock_cls, patch(
        "memory_condense.eval.responder.litellm"
    ) as mock_resp, patch("memory_condense.eval.judge.litellm") as mock_judge:
        mock_resp.completion.side_effect = _mock_responder_completion
        mock_judge.completion.side_effect = _mock_judge_completion

        mc = mock_cls.return_value.__enter__.return_value
        mc.search.return_value = []
        mc.search_hybrid.return_value = []

        replay_conversation("t.txt", TWO_TURNS, config, tmp_path / "data")
        return mc


def test_dense_is_the_default_retrieval_path(tmp_path):
    config = EvalConfig(retrieval=RetrievalConfig(k=3))
    assert config.retrieval.hybrid is False

    mc = _run_with_mock_condenser(config, tmp_path)

    assert mc.search.called
    assert not mc.search_hybrid.called


def test_hybrid_flag_switches_retrieval_path(tmp_path):
    config = EvalConfig(retrieval=RetrievalConfig(k=3, hybrid=True, alpha=0.4))

    mc = _run_with_mock_condenser(config, tmp_path)

    assert mc.search_hybrid.called
    assert not mc.search.called
    assert mc.search_hybrid.call_args.kwargs["alpha"] == 0.4


def test_hybrid_run_filenames_are_distinct(tmp_path):
    """A hybrid run must not overwrite the dense run it is being compared to."""
    from memory_condense.eval.report import save_run_result
    from memory_condense.eval.schemas import EvalRunResult

    def _result(hybrid: bool) -> EvalRunResult:
        return EvalRunResult(
            config=EvalConfig(retrieval=RetrievalConfig(k=10, hybrid=hybrid)),
            conversations=[],
            aggregate_mean_score=0.0,
            aggregate_recall_at_4=0.0,
            run_timestamp="2026-08-14T00:00:00+00:00",
        )

    dense = save_run_result(_result(False), tmp_path)
    hybrid = save_run_result(_result(True), tmp_path)

    assert "hybrid" in hybrid.name
    assert "hybrid" not in dense.name


# ---------------------------------------------------------------------------
# Memory mode — the arm that makes build_context / ContextPacker / rank_score
# measurable at all. Before it existed both eval paths called mc.search
# directly and exercised none of them.
# ---------------------------------------------------------------------------


def _run_memory_mode(config, tmp_path):
    from memory_condense.domain.schemas import PackedContext

    with patch("memory_condense.eval.runner.MemoryCondenser") as mock_cls, patch(
        "memory_condense.eval.responder.litellm"
    ) as mock_resp, patch("memory_condense.eval.judge.litellm") as mock_judge:
        mock_resp.completion.side_effect = _mock_responder_completion
        mock_judge.completion.side_effect = _mock_judge_completion

        mc = mock_cls.return_value.__enter__.return_value
        mc.heat_counts.return_value = {"HOT": 1, "WARM": 0, "COLD": 0}
        mc.build_context.return_value = PackedContext(
            messages=[{"role": "user", "content": "packed"}],
            memory_header="Relevant memory:\n- [Decision] a\n- [Decision] b",
            memory_ids=["a", "b"],
            expansions=["excerpt one"],
            token_counts={"memory_header": 12, "user_text": 3},
            dropped={"memories": 7},
        )
        replay_conversation("t.txt", TWO_TURNS, config, tmp_path / "data")
        return mock_cls, mc


def test_memory_mode_uses_build_context_not_search(tmp_path):
    config = EvalConfig(retrieval=RetrievalConfig(k=3, mode="memory"))

    _, mc = _run_memory_mode(config, tmp_path)

    assert mc.build_context.called
    assert not mc.search.called
    assert not mc.search_hybrid.called


def test_memory_mode_passes_the_configured_recent_window(tmp_path):
    """build_context defaults to 8 while the chunk arms use recent_window (4).

    Letting the default through would hand the memory arm a 2x larger recent
    window, and the measured delta would be recency rather than memory.
    """
    config = EvalConfig(retrieval=RetrievalConfig(k=3, mode="memory"), recent_window=4)

    _, mc = _run_memory_mode(config, tmp_path)

    assert mc.build_context.call_args.kwargs["recent_turns"] == 4


def test_memory_mode_matches_expansion_count_to_k(tmp_path):
    """Otherwise this compares '3 excerpts + header' against '10 chunks'."""
    config = EvalConfig(retrieval=RetrievalConfig(k=10, mode="memory"))

    mock_cls, mc = _run_memory_mode(config, tmp_path)

    assert mc.build_context.call_args.kwargs["k_expansions"] == 10
    assert mock_cls.call_args.kwargs["budget"].max_expansions == 10


def test_memory_mode_uses_hybrid_expansions_by_default(tmp_path):
    config = EvalConfig(retrieval=RetrievalConfig(k=10, mode="memory"))

    _, mc = _run_memory_mode(config, tmp_path)

    assert mc.build_context.call_args.kwargs["hybrid"] is True


def test_memory_mode_turns_extraction_on(tmp_path):
    """In memory mode the extractor is the thing under test, not dead cost."""
    config = EvalConfig(retrieval=RetrievalConfig(k=3, mode="memory"))
    mock_cls, _ = _run_memory_mode(config, tmp_path)
    assert mock_cls.call_args.kwargs["auto_extract"] is True


def test_dense_mode_leaves_extraction_off(tmp_path):
    config = EvalConfig(retrieval=RetrievalConfig(k=3))
    with patch("memory_condense.eval.runner.MemoryCondenser") as mock_cls, patch(
        "memory_condense.eval.responder.litellm"
    ) as mock_resp, patch("memory_condense.eval.judge.litellm") as mock_judge:
        mock_resp.completion.side_effect = _mock_responder_completion
        mock_judge.completion.side_effect = _mock_judge_completion
        mock_cls.return_value.__enter__.return_value.search.return_value = []
        replay_conversation("t.txt", TWO_TURNS, config, tmp_path / "data")

    assert mock_cls.call_args.kwargs["auto_extract"] is False


def test_memory_mode_records_the_header_drop_count(tmp_path):
    """The per-turn measurement behind the header-budget finding."""
    config = EvalConfig(retrieval=RetrievalConfig(k=3, mode="memory"))

    with patch("memory_condense.eval.runner.MemoryCondenser") as mock_cls, patch(
        "memory_condense.eval.responder.litellm"
    ) as mock_resp, patch("memory_condense.eval.judge.litellm") as mock_judge:
        from memory_condense.domain.schemas import PackedContext

        mock_resp.completion.side_effect = _mock_responder_completion
        mock_judge.completion.side_effect = _mock_judge_completion
        mc = mock_cls.return_value.__enter__.return_value
        mc.heat_counts.return_value = {"HOT": 2, "WARM": 1, "COLD": 0}
        mc.build_context.return_value = PackedContext(
            messages=[{"role": "user", "content": "packed"}],
            memory_header="Relevant memory:\n- [Decision] a\n- [Decision] b",
            memory_ids=["a", "b"],
            token_counts={"memory_header": 12},
            dropped={"memories": 7},
        )
        result = replay_conversation("t.txt", TWO_TURNS, config, tmp_path / "data")

    scored = result.turn_results[-1]
    assert scored.memories_dropped == 7
    assert scored.memory_items_packed == 2
    assert scored.heat_counts == {"HOT": 2, "WARM": 1, "COLD": 0}
    assert scored.context_tokens == 12


def test_memory_run_filenames_are_distinct_from_dense(tmp_path):
    """The two arms of the decision run must not differ only by timestamp."""
    from memory_condense.eval.report import save_run_result
    from memory_condense.eval.schemas import EvalRunResult

    def _result(mode: str) -> EvalRunResult:
        return EvalRunResult(
            config=EvalConfig(retrieval=RetrievalConfig(k=10, mode=mode)),
            conversations=[],
            aggregate_mean_score=0.0,
            aggregate_recall_at_4=0.0,
            run_timestamp="2026-08-15T00:00:00+00:00",
        )

    dense = save_run_result(_result("dense"), tmp_path)
    memory = save_run_result(_result("memory"), tmp_path)

    assert "memory" in memory.name
    assert "dense" in dense.name
    assert dense.name != memory.name
