"""MCP server tests.

The tool functions are exercised directly against a temporary store with a
fake embedder, so nothing here downloads bge-m3 or opens a socket. The MCP
plumbing itself (FastMCP) is not re-tested — only our tool bodies and the
data-directory resolution.
"""

from __future__ import annotations

import asyncio

import pytest

from memory_condense import mcp_server
from tests.test_condenser import FakeEmbedder


@pytest.fixture
def server(tmp_path, monkeypatch):
    """Point the server at a temp store backed by a fake embedder."""
    from memory_condense.condenser import MemoryCondenser

    monkeypatch.setenv("MEMORY_CONDENSE_DATA_DIR", str(tmp_path / "store"))
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)

    condenser = MemoryCondenser(
        data_dir=tmp_path / "store", embedder=FakeEmbedder()
    )
    monkeypatch.setattr(mcp_server, "_condenser", condenser)
    yield mcp_server
    condenser.close()
    monkeypatch.setattr(mcp_server, "_condenser", None)


class TestDataDirResolution:
    def test_explicit_env_var_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MEMORY_CONDENSE_DATA_DIR", str(tmp_path / "explicit"))
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path / "project"))
        assert mcp_server._data_dir() == tmp_path / "explicit"

    def test_falls_back_to_project_dir(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MEMORY_CONDENSE_DATA_DIR", raising=False)
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path / "project"))
        assert mcp_server._data_dir() == tmp_path / "project" / ".memory_condense"

    def test_falls_back_to_cwd(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MEMORY_CONDENSE_DATA_DIR", raising=False)
        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        assert mcp_server._data_dir() == tmp_path / ".memory_condense"


class TestRemember:
    def test_stores_and_reports_id(self, server):
        out = server.remember("We decided to use SQLite for storage.")
        assert "Remembered" in out
        assert "[Decision]" in out
        assert server._condense().memory.count() == 1

    def test_type_is_case_insensitive(self, server):
        out = server.remember("User prefers dark mode.", type="preference")
        assert "[Preference]" in out

    def test_unknown_type_falls_back_to_entity(self, server):
        out = server.remember("Postgres is the prod database.", type="nonsense")
        assert "[Entity]" in out

    def test_empty_content_is_rejected(self, server):
        assert "Nothing stored" in server.remember("   ")
        assert server._condense().memory.count() == 0

    def test_pin_flag_pins(self, server):
        out = server.remember("Never log secrets.", type="Constraint", pin=True)
        assert "PINNED" in out

    def test_stored_memory_has_real_provenance(self, server):
        server.remember("We decided to use hnswlib.")
        item = server._condense().memory.list_items()[0]
        assert item.provenance
        turn = server._condense().transcript.get_turn(item.provenance[0].turn_id)
        assert turn is not None
        assert item.provenance[0].quote in turn.text


class TestRecall:
    def test_empty_store_says_so(self, server):
        assert "No memories stored yet" in server.recall("anything")

    def test_returns_stored_memory(self, server):
        server.remember("We decided to use SQLite for storage.")
        out = server.recall("what database are we using?")
        assert "SQLite" in out
        assert "score=" in out

    def test_limit_is_respected(self, server):
        for i in range(5):
            server.remember(f"Decision number {i} about the build system.")
        out = server.recall("build system", limit=2)
        assert out.startswith("2 memories")


class TestSearch:
    def test_empty_store_says_so(self, server):
        assert "Nothing ingested yet" in server.search("anything")

    def test_finds_ingested_text(self, server):
        server.ingest("The retrieval index uses hnswlib with cosine distance.")
        out = server.search("hnswlib")
        assert "hnswlib" in out

    def test_hybrid_reports_both_components(self, server):
        server.ingest("The retrieval index uses hnswlib with cosine distance.")
        out = server.search("hnswlib", hybrid=True)
        assert "dense" in out and "lexical" in out

    def test_dense_only_omits_components(self, server):
        server.ingest("The retrieval index uses hnswlib with cosine distance.")
        out = server.search("hnswlib", hybrid=False)
        assert "lexical" not in out


class TestIngest:
    def test_reports_chunks(self, server):
        out = server.ingest("We decided to use SQLite. It must never block writes.")
        assert "chunk(s)" in out
        assert server._condense().transcript.count() == 1

    def test_empty_text_is_rejected(self, server):
        assert "Nothing ingested" in server.ingest("  ")

    def test_bad_role_falls_back_to_user(self, server):
        out = server.ingest("Some content here.", role="wizard")
        assert "user turn" in out


class TestStatsPinForget:
    def test_stats_reports_counts(self, server):
        server.remember("We decided to use SQLite for storage.")
        out = server.memory_stats()
        assert "Active memories: 1" in out
        assert "Heat:" in out

    def test_pin_by_short_id(self, server):
        server.remember("We decided to use SQLite for storage.")
        item = server._condense().memory.list_items()[0]
        out = server.pin_memory(item.mem_id[:8])
        assert "Pinned" in out and "PINNED" in out

    def test_unpin(self, server):
        server.remember("x is y.", pin=True)
        item = server._condense().memory.list_items()[0]
        assert "Unpinned" in server.pin_memory(item.mem_id[:8], pinned=False)

    def test_unknown_id_is_reported_not_raised(self, server):
        assert "No memory matches" in server.pin_memory("deadbeef")
        assert "No memory matches" in server.forget("deadbeef")

    def test_forget_soft_deletes(self, server):
        server.remember("Temporary decision to revisit.")
        item = server._condense().memory.list_items()[0]
        assert "Forgot" in server.forget(item.mem_id[:8])
        assert server._condense().memory.list_items() == []
        # The row survives — provenance must stay walkable.
        assert server._condense().memory.get(item.mem_id) is not None


class TestToolRegistration:
    """`list_tools` is async; driven with asyncio.run to avoid a plugin dependency."""

    @staticmethod
    def _tools():
        return asyncio.run(mcp_server.mcp.list_tools())

    def test_all_tools_are_registered(self):
        assert {t.name for t in self._tools()} == {
            "remember",
            "recall",
            "search",
            "ingest",
            "memory_stats",
            "pin_memory",
            "forget",
        }

    def test_every_tool_has_a_description(self):
        for tool in self._tools():
            assert tool.description and len(tool.description) > 40, tool.name

    def test_descriptions_say_when_to_call(self):
        """Prescriptive descriptions measurably improve tool triggering."""
        by_name = {t.name: (t.description or "") for t in self._tools()}
        for name in ("remember", "recall", "search"):
            assert "Call this" in by_name[name] or "Use this" in by_name[name], name

    def test_inputs_are_schema_typed(self):
        by_name = {t.name: t for t in self._tools()}
        schema = by_name["remember"].inputSchema
        assert schema["properties"]["content"]["type"] == "string"
        assert schema["properties"]["pin"]["type"] == "boolean"
        assert schema["required"] == ["content"]
