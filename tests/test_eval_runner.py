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
