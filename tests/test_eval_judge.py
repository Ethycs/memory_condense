import inspect
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from memory_condense.eval.judge import (
    JUDGE_MAX_TOKENS,
    judge_response,
    judge_response_with_usage,
)


def _mock_response(content: str, prompt_tokens: int = 0, completion_tokens: int = 0):
    mock_choice = MagicMock()
    mock_choice.message.content = content
    return MagicMock(
        choices=[mock_choice],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
    )


@patch("memory_condense.eval.judge.litellm")
def test_judge_response_parses_score(mock_litellm):
    mock_litellm.completion.return_value = _mock_response(
        json.dumps({"score": 4, "reasoning": "Good match with minor differences"})
    )

    score, reasoning = judge_response(
        user_text="What is X?",
        actual_response="X is a thing.",
        generated_response="X is something.",
        model="gpt-4o-mini",
    )

    assert score == 4
    assert "Good match" in reasoning
    mock_litellm.completion.assert_called_once()


@patch("memory_condense.eval.judge.litellm")
def test_judge_clamps_score(mock_litellm):
    mock_litellm.completion.return_value = _mock_response(
        json.dumps({"score": 10, "reasoning": "Perfect"})
    )

    score, _ = judge_response("q", "a", "a")
    assert score == 5  # clamped to max


@patch("memory_condense.eval.judge.litellm")
def test_judge_handles_bad_json(mock_litellm):
    mock_litellm.completion.return_value = _mock_response("not valid json")

    score, reasoning = judge_response("q", "a", "a")
    assert score == 1
    assert "Failed to parse" in reasoning


@patch("memory_condense.eval.judge.litellm")
def test_judge_does_not_pass_temperature(mock_litellm):
    """Regression: Claude Sonnet 5 rejects non-default sampling params (400)."""
    mock_litellm.completion.return_value = _mock_response(
        json.dumps({"score": 3, "reasoning": "ok"})
    )

    judge_response("q", "a", "b")

    kwargs = mock_litellm.completion.call_args.kwargs
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs
    # And the signature no longer exposes one to accidentally re-plumb.
    assert "temperature" not in inspect.signature(judge_response).parameters
    assert (
        "temperature" not in inspect.signature(judge_response_with_usage).parameters
    )


@patch("memory_condense.eval.judge.litellm")
def test_judge_default_model_and_max_tokens(mock_litellm):
    """Sonnet 5 thinks adaptively; max_tokens caps thinking + text together."""
    mock_litellm.completion.return_value = _mock_response(
        json.dumps({"score": 5, "reasoning": "great"})
    )

    judge_response("q", "a", "b")

    kwargs = mock_litellm.completion.call_args.kwargs
    assert kwargs["model"] == "anthropic/claude-sonnet-5"
    assert kwargs["max_tokens"] == 1024
    assert JUDGE_MAX_TOKENS == 1024
    assert kwargs["num_retries"] == 5


@patch("memory_condense.eval.judge.litellm")
def test_judge_response_with_usage(mock_litellm):
    mock_litellm.completion.return_value = _mock_response(
        json.dumps({"score": 4, "reasoning": "fine"}),
        prompt_tokens=321,
        completion_tokens=45,
    )

    score, reasoning, usage = judge_response_with_usage("q", "a", "b")

    assert score == 4
    assert reasoning == "fine"
    assert usage.input_tokens == 321
    assert usage.output_tokens == 45
    assert usage.total_tokens == 366
    assert usage.calls == 1
    assert usage.elapsed_s >= 0.0
