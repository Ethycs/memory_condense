"""End-to-end integration test: ingest -> chunk -> embed -> retrieve.

Requires bge-m3 model. Marked as slow.
"""

import pytest

from memory_condense import MemoryCondenser


@pytest.mark.slow
def test_ingest_and_search(tmp_dir):
    with MemoryCondenser(data_dir=tmp_dir / "integration") as mc:
        mc.ingest("user", "My name is Alex and I work on memory systems.")
        mc.ingest(
            "assistant",
            "Nice to meet you, Alex! Memory systems are fascinating.",
        )
        mc.ingest("user", "I prefer Python and SQLite for storage.")

        results = mc.search("What is the user's name?", k=3)
        assert len(results) > 0
        # The chunk mentioning "Alex" should rank high
        top_texts = " ".join(r.chunk.text for r in results[:2])
        assert "Alex" in top_texts


@pytest.mark.slow
def test_ingest_empty_text(tmp_dir):
    with MemoryCondenser(data_dir=tmp_dir / "empty") as mc:
        turn, chunks = mc.ingest("user", "")
        assert turn.text == ""
        assert chunks == []


@pytest.mark.slow
def test_search_empty_index(tmp_dir):
    with MemoryCondenser(data_dir=tmp_dir / "empty_search") as mc:
        results = mc.search("anything", k=5)
        assert results == []


@pytest.mark.slow
def test_transcript_access(tmp_dir):
    with MemoryCondenser(data_dir=tmp_dir / "transcript") as mc:
        mc.ingest("user", "hello")
        mc.ingest("assistant", "hi there")

        assert mc.transcript.count() == 2
        recent = mc.transcript.get_recent(1)
        assert recent[0].text == "hi there"
