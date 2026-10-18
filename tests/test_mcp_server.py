"""MCP server tests.

The tool functions are exercised directly against a temporary store with a
fake embedder, so nothing here downloads bge-m3 or opens a socket. The MCP
plumbing itself (FastMCP) is not re-tested — only our tool bodies and the
data-directory resolution.
"""

from __future__ import annotations

import asyncio

import pytest

from memory_condense.interfaces import mcp_server
from tests.test_condenser import FakeEmbedder


@pytest.fixture
def server(tmp_path, monkeypatch):
    """Point the server at a temp store backed by a fake embedder."""
    from memory_condense.application.condenser import MemoryCondenser

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

    def test_walks_up_to_the_project_root(self, monkeypatch, tmp_path):
        """A subdirectory shares the project's store instead of starting its own.

        This used to trust `Path.cwd()`, so a server launched from `src/` kept
        a different store than one launched from the repository root.
        """
        monkeypatch.delenv("MEMORY_CONDENSE_DATA_DIR", raising=False)
        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        (tmp_path / ".git").mkdir()
        deep = tmp_path / "src" / "pkg" / "sub"
        deep.mkdir(parents=True)
        monkeypatch.chdir(deep)
        assert mcp_server._data_dir() == tmp_path / ".memory_condense"

    def test_no_project_falls_back_to_the_session_home(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MEMORY_CONDENSE_DATA_DIR", raising=False)
        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        monkeypatch.chdir(tmp_path)  # a bare directory: no .git, no pyproject.toml
        from memory_condense.runtime.config import user_directory

        assert mcp_server._data_dir() == user_directory() / "data"

    def test_an_existing_local_store_is_adopted_not_orphaned(self, monkeypatch, tmp_path):
        """The old `<cwd>/.memory_condense` default may hold real memories."""
        monkeypatch.delenv("MEMORY_CONDENSE_DATA_DIR", raising=False)
        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        (tmp_path / ".memory_condense").mkdir()
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

    def test_stats_labels_whose_store_it_is(self, server):
        """The proxy keeps a different store; the label must not let that blur."""
        lines = server.memory_stats().splitlines()
        assert lines[0].startswith("Store: ")
        assert "this MCP server's own store" in lines[1]
        assert "recall" in lines[1] and "forget" in lines[1]

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


class TestProvenanceSourcing:
    """`remember` must prefer a real transcript turn over a self-quoting one."""

    #: Chosen to match none of `extractor.RULES` — it has no "we decided",
    #: "must", "i prefer" or other cue word. Otherwise `ingest`'s auto-extraction
    #: creates this memory first and the subsequent `remember` reports a merge
    #: instead of exercising the witnessed-provenance path this class is about.
    FACT = "The beta ships on Friday."

    def test_witnessed_when_the_text_was_actually_said(self, server):
        server._condense().ingest("user", f"{self.FACT} No exceptions.")
        before = server._condense().transcript.count()

        out = server.remember(self.FACT)

        assert "witnessed in the transcript" in out
        # No new turn: the memory cites what the user really said.
        assert server._condense().transcript.count() == before

    def test_asserted_when_nothing_in_the_transcript_says_it(self, server):
        out = server.remember("The user works in Pacific time.", "Entity")

        assert "asserted" in out
        assert server._condense().transcript.count() == 1

    def test_the_cited_turn_really_contains_the_quote(self, server):
        server._condense().ingest("user", f"{self.FACT} No exceptions.")
        server.remember(self.FACT)

        item = server._condense().memory.list_items()[0]
        assert item.provenance, "memory stored with no provenance"
        source = item.provenance[0]
        assert server._condense().validator.quote_matches(source.turn_id, source.quote)
        # The cited turn is the user's own, not one manufactured for the memory.
        assert "No exceptions." in server._condense().transcript.get_turn(
            source.turn_id
        ).text

    def test_wildcards_in_a_fact_do_not_match_unrelated_turns(self, server):
        """A LIKE lookup that forgets to escape would call this witnessed."""
        server._condense().ingest("user", "Coverage is above ninety percent.")

        out = server.remember("Coverage is 100% _guaranteed_.")

        assert "asserted" in out


class TestSupersede:
    def test_links_the_replacement_to_what_it_replaced(self, server):
        server.remember("The beta ships on Friday.", "Decision")
        old = server._condense().memory.list_items()[0]

        out = server.supersede(old.mem_id[:8], "The beta ships on Monday.")

        assert "Superseded" in out
        rows = {i.mem_id: i for i in server._condense().memory.list_items(status=None)}
        assert rows[old.mem_id].status.value == "superseded"
        new = next(i for i in rows.values() if i.supersedes == old.mem_id)
        assert new.content == "The beta ships on Monday."
        assert new.status.value == "active"

    def test_replacement_inherits_the_old_type_by_default(self, server):
        server.remember("Tabs, not spaces.", "Preference")
        old = server._condense().memory.list_items()[0]

        server.supersede(old.mem_id[:8], "Spaces, not tabs.")

        new = next(
            i
            for i in server._condense().memory.list_items()
            if i.supersedes == old.mem_id
        )
        assert new.type.value == "Preference"

    def test_explicit_type_overrides_it(self, server):
        server.remember("Tabs, not spaces.", "Preference")
        old = server._condense().memory.list_items()[0]

        server.supersede(old.mem_id[:8], "Spaces are mandatory.", "Constraint")

        new = next(
            i
            for i in server._condense().memory.list_items()
            if i.supersedes == old.mem_id
        )
        assert new.type.value == "Constraint"

    def test_superseded_items_stop_being_recalled(self, server):
        server.remember("The beta ships on Friday.", "Decision")
        old = server._condense().memory.list_items()[0]
        server.supersede(old.mem_id[:8], "The beta ships on Monday.")

        out = server.recall("when does the beta ship")

        assert "Monday" in out
        assert "Friday" not in out

    def test_unknown_id_is_reported_not_raised(self, server):
        assert "No memory matches" in server.supersede("deadbeef", "Anything.")

    def test_empty_content_stores_nothing(self, server):
        server.remember("The beta ships on Friday.", "Decision")
        old = server._condense().memory.list_items()[0]

        assert "Nothing superseded" in server.supersede(old.mem_id[:8], "   ")
        assert server._condense().memory.count() == 1


class TestDedupThroughMCP:
    def test_re_remembering_a_known_fact_is_a_no_op(self, server):
        server.remember("The beta ships on Friday.", "Decision")
        out = server.remember("The beta ships on Friday.", "Decision")

        assert "Already remembered" in out
        assert server._condense().memory.count() == 1

    def test_a_repeat_assertion_appends_no_transcript_turn(self, server):
        server.remember("The beta ships on Friday.", "Decision")
        before = server._condense().transcript.count()

        server.remember("the beta ships on   FRIDAY.", "Decision")

        assert server._condense().transcript.count() == before

    def test_a_repeat_assertion_can_still_pin(self, server):
        server.remember("The beta ships on Friday.", "Decision")
        out = server.remember("The beta ships on Friday.", "Decision", pin=True)

        assert "PINNED" in out
        assert server._condense().memory.list_items()[0].is_pinned

    def test_the_same_text_under_a_different_type_is_kept_separately(self, server):
        server.remember("The beta ships on Friday.", "Decision")
        out = server.remember("The beta ships on Friday.", "Constraint")

        assert "Already remembered" not in out
        assert server._condense().memory.count() == 2


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
            "supersede",
            "forget",
            "memory_snapshot_save",
            "memory_snapshot_restore",
            "memory_reset",
            "memory_load_transcript",
        }

    def test_the_old_ambiguous_names_are_gone(self):
        """`memory_load` read as "load content in" reached a destructive tool."""
        assert {"memory_save", "memory_load"}.isdisjoint({t.name for t in self._tools()})

    def test_the_destructive_tool_says_so_in_its_first_line(self):
        by_name = {t.name: (t.description or "") for t in self._tools()}
        first = by_name["memory_snapshot_restore"].splitlines()[0]
        assert "DESTRUCTIVE" in first
        assert "discard" in first

    def test_control_tools_name_the_store_they_act_on(self):
        """The control tools act on the proxy's store, not this server's."""
        by_name = {t.name: (t.description or "") for t in self._tools()}
        for name in ("memory_snapshot_save", "memory_snapshot_restore",
                     "memory_reset", "memory_load_transcript"):
            assert "proxy" in by_name[name], name
        for name in ("memory_reset", "memory_snapshot_save"):
            assert "separate" in by_name[name] or "different store" in by_name[name], name

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
