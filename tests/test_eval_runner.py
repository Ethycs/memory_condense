"""Test the eval runner with mocked LLM calls."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from memory_condense.eval.runner import replay_conversation
from memory_condense.eval.schemas import ChunkerConfig, EvalConfig, RetrievalConfig


def _mock_responder_completion(**kwargs):
    """Create a mock litellm.completion response for the responder."""
    mock_choice = MagicMock()
    mock_choice.message.content = "Mock generated response"
    return MagicMock(choices=[mock_choice])


def _mock_judge_completion(**kwargs):
    """Create a mock litellm.completion response for the judge."""
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(
        {"score": 4, "reasoning": "Good match"}
    )
    return MagicMock(choices=[mock_choice])


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
