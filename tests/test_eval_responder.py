from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from memory_condense.eval.responder import (
    build_prompt,
    generate_response,
    generate_response_with_usage,
)
from memory_condense.schemas import Chunk, RetrievalResult


def _mock_response(content: str, prompt_tokens: int = 0, completion_tokens: int = 0):
    mock_choice = MagicMock()
    mock_choice.message.content = content
    return MagicMock(
        choices=[mock_choice],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
    )


def test_build_prompt_no_context():
    messages = build_prompt("Hello", retrieved=[], recent_turns=[])
    assert len(messages) == 2  # system + user
    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == "Hello"


def test_build_prompt_with_chunks():
    chunk = Chunk(
        turn_id="t1", text="relevant info", start_char=0, end_char=13, token_count=2
    )
    result = RetrievalResult(chunk=chunk, score=0.9)
    messages = build_prompt("Question?", retrieved=[result], recent_turns=[])
    assert len(messages) == 3  # system + memory + user
    assert "relevant info" in messages[1]["content"]


def test_build_prompt_with_recent_turns():
    messages = build_prompt(
        "Follow-up",
        retrieved=[],
        recent_turns=[("user", "earlier"), ("assistant", "response")],
    )
    assert len(messages) == 4  # system + user turn + assistant turn + current user
    assert messages[1]["content"] == "earlier"
    assert messages[2]["content"] == "response"


@patch("memory_condense.eval.responder.litellm")
def test_generate_response_calls_litellm(mock_litellm):
    mock_litellm.completion.return_value = _mock_response("Generated response text")

    result = generate_response(
        user_text="What is Python?",
        retrieved=[],
        recent_turns=[],
        model="gpt-4o-mini",
    )

    assert result == "Generated response text"
    mock_litellm.completion.assert_called_once()
    call_kwargs = mock_litellm.completion.call_args
    assert call_kwargs.kwargs["model"] == "gpt-4o-mini"


@patch("memory_condense.eval.responder.litellm")
def test_generate_response_default_model_and_sampling(mock_litellm):
    """Haiku 4.5 replaces retired 3.5 Haiku and still accepts sampling params."""
    mock_litellm.completion.return_value = _mock_response("hi")

    generate_response(user_text="q", retrieved=[], recent_turns=[])

    kwargs = mock_litellm.completion.call_args.kwargs
    assert kwargs["model"] == "anthropic/claude-haiku-4-5"
    assert kwargs["temperature"] == 0.3
    assert kwargs["max_tokens"] == 1024
    assert kwargs["num_retries"] == 5


@patch("memory_condense.eval.responder.litellm")
def test_generate_response_with_usage(mock_litellm):
    mock_litellm.completion.return_value = _mock_response(
        "Generated response text", prompt_tokens=200, completion_tokens=30
    )

    text, usage = generate_response_with_usage(
        user_text="What is Python?",
        retrieved=[],
        recent_turns=[],
        model="gpt-4o-mini",
    )

    assert text == "Generated response text"
    assert usage.input_tokens == 200
    assert usage.output_tokens == 30
    assert usage.total_tokens == 230
    assert usage.calls == 1
    assert usage.elapsed_s >= 0.0


@patch("memory_condense.eval.responder.litellm")
def test_generate_response_tolerates_missing_usage(mock_litellm):
    """Usage fields vary by provider — missing usage must not blow up."""
    mock_choice = MagicMock()
    mock_choice.message.content = "text"
    mock_litellm.completion.return_value = MagicMock(choices=[mock_choice])

    text, usage = generate_response_with_usage("q", [], [])

    assert text == "text"
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.calls == 1
