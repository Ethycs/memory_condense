import pytest

from memory_condense._tokenizer import (
    CHAT_FRAMING_TOKENS_FIXED,
    CHAT_FRAMING_TOKENS_PER_MESSAGE,
    count_chat_prompt_token_proxy,
    count_tokens,
    tokenizer_proxy_identity,
)
from memory_condense.chunker import Chunker


@pytest.fixture
def chunker():
    return Chunker(min_tokens=10, max_tokens=50)


def test_empty_text(chunker):
    assert chunker.chunk_turn("t1", "") == []
    assert chunker.chunk_turn("t1", "   ") == []


def test_single_short_sentence(chunker):
    chunks = chunker.chunk_turn("t1", "Hello world.")
    assert len(chunks) == 1
    assert chunks[0].turn_id == "t1"
    assert chunks[0].text.strip() == "Hello world."


def test_multiple_sentences_merge():
    # Use small token limits to force merging behavior
    chunker = Chunker(min_tokens=5, max_tokens=30)
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    chunks = chunker.chunk_turn("t1", text)
    # Should have at least 1 chunk
    assert len(chunks) >= 1
    # All text should be covered
    all_text = " ".join(c.text for c in chunks)
    for word in ["First", "Second", "Third", "Fourth"]:
        assert word in all_text


def test_chunk_offsets():
    chunker = Chunker(min_tokens=5, max_tokens=200)
    text = "Hello world. This is a test."
    chunks = chunker.chunk_turn("t1", text)
    for chunk in chunks:
        assert chunk.start_char >= 0
        assert chunk.end_char <= len(text) + 1  # allow for minor offset
        assert chunk.start_char < chunk.end_char


def test_token_count_populated():
    chunker = Chunker(min_tokens=5, max_tokens=200)
    text = "This is a simple sentence with several words in it."
    chunks = chunker.chunk_turn("t1", text)
    assert len(chunks) == 1
    assert chunks[0].token_count == count_tokens(chunks[0].text)


def test_chat_prompt_proxy_adds_explicit_framing_and_binds_vocabulary():
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Where?"},
    ]
    expected = (
        sum(count_tokens(message["content"]) for message in messages)
        + CHAT_FRAMING_TOKENS_PER_MESSAGE * len(messages)
        + CHAT_FRAMING_TOKENS_FIXED
    )

    assert count_chat_prompt_token_proxy(messages) == expected
    identity = tokenizer_proxy_identity()
    assert identity["schema"] == "memory-condense-prompt-token-proxy-v1"
    assert identity["encoding"] == "cl100k_base"
    assert len(str(identity["vocabulary_sha256"])) == 64
    assert identity["chat_framing_tokens_per_message"] == 8
    assert identity["chat_framing_tokens_fixed"] == 8


def test_conceptual_spans_separate_plan_from_completed_event(chunker):
    text = (
        "I'm planning to buy concert merchandise. By the way, I just got "
        "back from an amazing Billie Eilish concert today."
    )

    assert chunker.conceptual_spans(text) == [
        "I'm planning to buy concert merchandise.",
        "I just got back from an amazing Billie Eilish concert today.",
    ]


def test_long_text_splits():
    # Force splitting with tight limits
    chunker = Chunker(min_tokens=3, max_tokens=15)
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "A wonderful serenity has taken possession of my entire soul. "
        "I am so happy my dear friend so absorbed in the exquisite sense. "
        "Like these sweet mornings of spring which I enjoy with my whole heart."
    )
    chunks = chunker.chunk_turn("t1", text)
    assert len(chunks) > 1
    # Every persisted count and boundary is exact.
    for chunk in chunks:
        assert chunk.token_count == count_tokens(chunk.text)
        assert chunk.token_count <= 15


def test_hard_split_preserves_unicode_and_exact_maximum():
    chunker = Chunker(min_tokens=2, max_tokens=10)
    text = "🙂" * 40

    chunks = chunker.chunk_turn("t1", text)

    assert "".join(chunk.text for chunk in chunks) == text
    assert all(chunk.token_count == count_tokens(chunk.text) for chunk in chunks)
    assert all(chunk.token_count <= 10 for chunk in chunks)


@pytest.mark.parametrize(
    ("min_tokens", "max_tokens"),
    [(0, 10), (10, 9)],
)
def test_invalid_token_bounds_are_rejected(min_tokens, max_tokens):
    with pytest.raises(ValueError):
        Chunker(min_tokens=min_tokens, max_tokens=max_tokens)
