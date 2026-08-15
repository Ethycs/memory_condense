"""Facade wiring tests.

These use a deterministic fake embedder so the whole pipeline — chunk, embed,
index, extract, validate, rank, pack — is exercised without downloading bge-m3.
The real-model path is covered by tests/test_integration.py behind `slow`.
"""

from __future__ import annotations

import re
import zlib

import numpy as np
import pytest

from memory_condense.condenser import MemoryCondenser
from memory_condense.context_packer import ContextBudget
from memory_condense.schemas import (
    Chunk,
    MemoryOps,
    MemoryStatus,
    MemoryType,
    PinState,
    Provenance,
    CreateOp,
    PinOp,
)


class FakeEmbedder:
    """Bag-of-words hashing embedder — deterministic across processes."""

    def __init__(self, dim: int = 32) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self._dim, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            v[zlib.crc32(token.encode()) % self._dim] += 1.0
        if not v.any():
            v[0] = 1.0
        return v

    def embed_query(self, query: str) -> np.ndarray:
        return self._vec(query)

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        return [
            c.model_copy(update={"embedding": self._vec(c.text).tolist()})
            for c in chunks
        ]


CONVERSATION = [
    ("user", "I prefer Python for this project. We decided to use SQLite for storage."),
    ("assistant", "Good call, SQLite in WAL mode never blocks readers."),
    ("user", "The index must never exceed one gigabyte on disk."),
]


@pytest.fixture
def mc(tmp_path):
    condenser = MemoryCondenser(
        data_dir=tmp_path / "data",
        embedder=FakeEmbedder(),
    )
    yield condenser
    condenser.close()


@pytest.fixture
def populated(mc):
    for role, text in CONVERSATION:
        mc.ingest(role, text)
    return mc


class TestIngest:
    def test_returns_turn_and_chunks(self, mc):
        turn, chunks = mc.ingest("user", "I prefer dark mode in all my apps.")
        assert turn.role == "user"
        assert chunks
        assert all(c.turn_id == turn.turn_id for c in chunks)

    def test_chunks_are_embedded_and_indexed(self, mc):
        _, chunks = mc.ingest("user", "SQLite is the storage layer.")
        assert all(c.embedding is not None for c in chunks)
        assert mc.search("storage layer", k=3)

    def test_transcript_is_appended(self, populated):
        assert populated.transcript.count() == len(CONVERSATION)

    def test_memory_extracted_automatically(self, populated):
        items = populated.memory.list_items()
        assert items, "rule-based extractor should have produced memory items"

    def test_every_memory_carries_provenance(self, populated):
        for item in populated.memory.list_items():
            assert item.provenance, f"{item.mem_id} has no provenance"
            for prov in item.provenance:
                turn = populated.transcript.get_turn(prov.turn_id)
                assert turn is not None
                assert prov.quote.strip() in turn.text

    def test_auto_extract_can_be_disabled(self, tmp_path):
        with MemoryCondenser(
            data_dir=tmp_path / "d", embedder=FakeEmbedder(), auto_extract=False
        ) as quiet:
            quiet.ingest("user", "We decided to use SQLite for storage.")
            assert quiet.memory.list_items() == []

    def test_empty_text_is_safe(self, mc):
        turn, chunks = mc.ingest("user", "   ")
        assert chunks == []


class TestExtractionSafety:
    def test_fabricated_provenance_is_rejected(self, populated):
        fake = CreateOp(
            type=MemoryType.DECISION,
            content="We decided to rewrite everything in Rust.",
            provenance=[Provenance(turn_id="nope", quote="rewrite in Rust")],
        )
        report = populated.validator.validate(MemoryOps(create=[fake]))
        assert not report.ok
        assert report.accepted.is_empty()

    def test_misquoted_provenance_is_rejected(self, populated):
        real_turn = populated.transcript.get_all()[0]
        fake = CreateOp(
            type=MemoryType.DECISION,
            content="We decided to use Postgres.",
            provenance=[
                Provenance(turn_id=real_turn.turn_id, quote="we decided to use Postgres")
            ],
        )
        report = populated.validator.validate(MemoryOps(create=[fake]))
        assert not report.ok


class TestRetrieval:
    def test_dense_search_returns_scored_chunks(self, populated):
        results = populated.search("storage", k=3)
        assert results
        assert all(r.score is not None for r in results)

    def test_hybrid_search_populates_both_components(self, populated):
        results = populated.search_hybrid("SQLite storage", k=3)
        assert results
        assert all(r.dense_score is not None for r in results)
        assert all(r.lexical_score is not None for r in results)

    def test_hybrid_is_sorted_descending(self, populated):
        scores = [r.score for r in populated.search_hybrid("SQLite", k=5)]
        assert scores == sorted(scores, reverse=True)

    def test_recall_memories_returns_ranked_items(self, populated):
        results = populated.recall_memories("what storage did we pick?", k=5)
        assert results
        assert all(r.item.status is MemoryStatus.ACTIVE for r in results)
        assert [r.score for r in results] == sorted(
            [r.score for r in results], reverse=True
        )


class TestContextAssembly:
    def test_returns_packed_context(self, populated):
        ctx = populated.build_context("What storage are we using?")
        assert ctx.messages
        assert ctx.messages[-1]["role"] == "user"
        assert ctx.messages[-1]["content"] == "What storage are we using?"

    def test_system_prompt_comes_first(self, populated):
        ctx = populated.build_context("q", system_prompt="You are helpful.")
        assert ctx.messages[0] == {"role": "system", "content": "You are helpful."}

    def test_memory_header_present_when_memories_exist(self, populated):
        ctx = populated.build_context("What did we decide?")
        assert ctx.memory_header.startswith("Relevant memory:")

    def test_budget_is_respected(self, tmp_path):
        budget = ContextBudget(
            recent_window_tokens=30, memory_header_tokens=30, expansion_tokens=30
        )
        with MemoryCondenser(
            data_dir=tmp_path / "d", embedder=FakeEmbedder(), budget=budget
        ) as small:
            for role, text in CONVERSATION * 4:
                small.ingest(role, text)
            ctx = small.build_context("what did we decide?")
            assert ctx.token_counts["memory_header"] <= 30
            assert ctx.token_counts["recent_turns"] <= 30
            assert ctx.token_counts["expansions"] <= 30

    def test_zero_k_produces_no_memory_or_expansions(self, populated):
        ctx = populated.build_context("q", k_memories=0, k_expansions=0)
        assert ctx.memory_header == ""
        assert ctx.expansions == []


class TestLifecycle:
    def test_heat_counts_reported(self, populated):
        counts = populated.heat_counts()
        assert set(counts) <= {"HOT", "WARM", "COLD"}
        assert sum(counts.values()) == len(populated.memory.list_items())

    def test_pinning_survives_reopen(self, tmp_path):
        data_dir = tmp_path / "persist"
        with MemoryCondenser(data_dir=data_dir, embedder=FakeEmbedder()) as first:
            first.ingest("user", "We decided to use SQLite for storage.")
            item = first.memory.list_items()[0]
            first.memory.pin(PinOp(mem_id=item.mem_id, pin=PinState.USER))
            mem_id = item.mem_id

        with MemoryCondenser(data_dir=data_dir, embedder=FakeEmbedder()) as second:
            reloaded = second.memory.get(mem_id)
            assert reloaded is not None
            assert reloaded.pin is PinState.USER

    def test_transcript_and_index_persist(self, tmp_path):
        data_dir = tmp_path / "persist2"
        with MemoryCondenser(data_dir=data_dir, embedder=FakeEmbedder()) as first:
            for role, text in CONVERSATION:
                first.ingest(role, text)

        with MemoryCondenser(data_dir=data_dir, embedder=FakeEmbedder()) as second:
            assert second.transcript.count() == len(CONVERSATION)
            assert second.search("storage", k=3)

    def test_context_manager_closes_cleanly(self, tmp_path):
        with MemoryCondenser(data_dir=tmp_path / "d", embedder=FakeEmbedder()) as c:
            c.ingest("user", "hello there")
        assert (tmp_path / "d" / "memory.db").exists()
