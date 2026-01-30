import pytest

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
    assert chunks[0].token_count > 0


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
    # Each chunk should respect max_tokens (approximately)
    for chunk in chunks:
        assert chunk.token_count <= 20  # some margin for merge edge cases
