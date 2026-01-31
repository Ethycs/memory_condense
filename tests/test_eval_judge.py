import json
from unittest.mock import MagicMock, patch

from memory_condense.eval.judge import judge_response


@patch("memory_condense.eval.judge.litellm")
def test_judge_response_parses_score(mock_litellm):
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(
        {"score": 4, "reasoning": "Good match with minor differences"}
    )
    mock_litellm.completion.return_value = MagicMock(choices=[mock_choice])

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
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({"score": 10, "reasoning": "Perfect"})
    mock_litellm.completion.return_value = MagicMock(choices=[mock_choice])

    score, _ = judge_response("q", "a", "a")
    assert score == 5  # clamped to max


@patch("memory_condense.eval.judge.litellm")
def test_judge_handles_bad_json(mock_litellm):
    mock_choice = MagicMock()
    mock_choice.message.content = "not valid json"
    mock_litellm.completion.return_value = MagicMock(choices=[mock_choice])

    score, reasoning = judge_response("q", "a", "a")
    assert score == 1
    assert "Failed to parse" in reasoning
