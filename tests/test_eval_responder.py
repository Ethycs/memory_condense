from unittest.mock import MagicMock, patch

from memory_condense.eval.responder import build_prompt, generate_response
from memory_condense.schemas import Chunk, RetrievalResult


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
    mock_choice = MagicMock()
    mock_choice.message.content = "Generated response text"
    mock_litellm.completion.return_value = MagicMock(choices=[mock_choice])

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
