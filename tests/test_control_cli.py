"""`memory-condense ctl` and the MCP control tools, fully offline.

`status` is rendered from documents written with ``write_status`` into a tmp
dir; the mutating commands and the MCP tools talk to a tiny stdlib HTTP
server standing in for the proxy's control routes and recording what it was
sent. No real proxy, store, or model is involved.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from memory_condense.control import cli as ctl
from memory_condense.control.status import empty_status, write_status
from memory_condense.interfaces import mcp_server


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ingesting_doc() -> dict:
    doc = empty_status(mode="augment", pid=4242)
    doc["state"] = "ingesting"
    doc["queue"].update({"capacity": 256, "in_flight": 3, "retained_bytes": 1048576,
                         "jobs": [{"id": "bulk-feedface", "kind": "transcript", "session": "feedface",
                                   "position": 1, "bytes": 1024 * 1024 * 5}]})
    doc["current"] = {"job": "bulk-01a08337", "kind": "transcript",
                      "records_done": 2944, "records_total": 5677,
                      "bytes_done": 134217728, "bytes_total": 258895902,
                      "tokens_done": 612000, "started_at": "2026-10-13T08:39:00Z", "eta_s": 98}
    doc["rates"] = {"tokens_per_s_recent": 6200, "records_per_s_recent": 31.0,
                    "avg_s_per_1k_tokens": 0.16, "window_s": 30}
    doc["totals"] = {"tokens_ingested": 1210000, "records_ingested": 18231, "sessions": 3,
                     "store_bytes": 48100000}
    doc["serving"] = {"requests": 41, "rewritten": 37, "static_tokens_last": 1104000,
                      "served_tokens_last": 48200, "saved_tokens_total": 38900000}
    doc["store"] = {"path": "/data/memory.sqlite", "snapshot_dir": "/data/snapshots",
                    "last_snapshot": "2026-10-13T08-40-00.sqlite",
                    "snapshots": ["2026-10-13T08-40-00.sqlite", "before-reset.sqlite"]}
    return doc


@pytest.fixture
def status_file(tmp_path):
    path = tmp_path / "control" / "status.json"
    write_status(path, _ingesting_doc())
    return path


class _Handler(BaseHTTPRequestHandler):
    """Records every POST; answers like the proxy's control routes would."""

    calls: list  # set per server instance via the server object

    def log_message(self, *_):  # keep pytest output clean
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.calls.append((self.path, body))
        if self.path == "/_control/load" and body.get("name") == "missing":
            status, reply = 404, {"error": "no such snapshot: missing"}
        elif self.path == "/_control/reset" and not body.get("confirm"):
            status, reply = 400, {"error": "confirm required"}
        else:
            status, reply = 200, {"ok": True, "route": self.path, "echo": body}
        payload = json.dumps(reply).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def fake_proxy():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.calls = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def closed_port():
    """A URL nothing listens on."""
    probe = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = probe.server_address[1]
    probe.server_close()
    return f"http://127.0.0.1:{port}"


# ---------------------------------------------------------------------------
# ctl status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_renders_summary_and_table(self, status_file, capsys):
        assert ctl.main(["status", "--status", str(status_file)]) == 0
        out = capsys.readouterr().out
        lines = out.splitlines()
        assert lines[0].startswith("mem ▸ ingest 52% (ETA 1m38s)")
        assert "1.21M tok" in lines[0]
        assert "serving 48k of 1.10M" in lines[0]
        assert "ingesting (augment, pid 4242)" in out
        assert "capacity 256 · in-flight 3" in out
        assert "jobs 1" in out and "bulk-feedface" in out
        assert "2944/5677 records (52%)" in out
        assert "128.0MB/246.9MB" in out
        assert "612k tok" in out and "ETA 1m38s" in out
        assert "6k tok/s · 31.0 rec/s · 0.16 s/1k tok" in out
        assert "18231 records · 3 sessions" in out
        assert "41 requests · 37 rewritten" in out
        assert "/data/memory.sqlite" in out
        assert "before-reset.sqlite" in out
        assert "STALE" not in out

    def test_idle_document(self, tmp_path, capsys):
        path = tmp_path / "status.json"
        write_status(path, empty_status())
        assert ctl.main(["status", "--status", str(path)]) == 0
        out = capsys.readouterr().out
        assert out.splitlines()[0] == "mem ▸ 0 tok"
        rows = {line.split(None, 1)[0]: line.split(None, 1)[1] for line in out.splitlines()[1:] if line.strip()}
        assert rows["current"] == "idle"
        assert rows["snapshots"] == "none"
        assert rows["state"].startswith("idle (observe, pid ")

    def test_marks_stale_after_ten_seconds(self, tmp_path, capsys):
        path = tmp_path / "status.json"
        write_status(path, _ingesting_doc())
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 125))
        path.write_text(json.dumps(doc), encoding="utf-8")

        assert ctl.main(["status", "--status", str(path)]) == 0
        out = capsys.readouterr().out
        assert "stale 2m05s" in out.splitlines()[0]
        assert "STALE" in out

    def test_json_output_adds_stale_flag(self, status_file, capsys):
        assert ctl.main(["status", "--json", "--status", str(status_file)]) == 0
        doc = json.loads(capsys.readouterr().out)
        assert doc["schema"] == 1
        assert doc["state"] == "ingesting"
        assert doc["stale"] is False
        assert doc["age_seconds"] >= 0

    def test_exit_3_when_missing(self, tmp_path, capsys, monkeypatch):
        assert ctl.main(["status", "--status", str(tmp_path / "nope.json")]) == 3
        assert "proxy not running or no store here" in capsys.readouterr().err

    def test_exit_3_when_nothing_resolves(self, tmp_path, capsys, monkeypatch):
        for var in ("MEMORY_CONDENSE_STATUS", "MEMORY_CONDENSE_DATA_DIR", "CLAUDE_PROJECT_DIR"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(ctl, "locate_status_file", lambda *a, **k: None)
        assert ctl.main(["status"]) == 3
        assert "proxy not running" in capsys.readouterr().err

    def test_locates_via_env_when_no_flag(self, status_file, monkeypatch, capsys):
        monkeypatch.setenv("MEMORY_CONDENSE_STATUS", str(status_file))
        assert ctl.main(["status"]) == 0
        assert "ingest 52%" in capsys.readouterr().out

    def test_unreadable_schema_is_exit_3(self, tmp_path, capsys):
        path = tmp_path / "status.json"
        path.write_text('{"schema": 99}', encoding="utf-8")
        assert ctl.main(["status", "--status", str(path)]) == 3
        assert "unreadable" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# ctl save / load / reset / load-transcript
# ---------------------------------------------------------------------------


class TestMutatingCommands:
    def test_save_without_name_posts_empty_object(self, fake_proxy, capsys):
        assert ctl.main(["save", "--proxy", fake_proxy.base_url]) == 0
        assert fake_proxy.calls == [("/_control/save", {})]
        assert json.loads(capsys.readouterr().out)["ok"] is True

    def test_save_with_name(self, fake_proxy, capsys):
        assert ctl.main(["save", "before-refactor", "--proxy", fake_proxy.base_url]) == 0
        assert fake_proxy.calls == [("/_control/save", {"name": "before-refactor"})]

    def test_load(self, fake_proxy, capsys):
        assert ctl.main(["load", "before-refactor", "--proxy", fake_proxy.base_url]) == 0
        assert fake_proxy.calls == [("/_control/load", {"name": "before-refactor"})]
        assert json.loads(capsys.readouterr().out)["echo"] == {"name": "before-refactor"}

    def test_refused_route_prints_body_and_exits_1(self, fake_proxy, capsys):
        assert ctl.main(["load", "missing", "--proxy", fake_proxy.base_url]) == 1
        captured = capsys.readouterr()
        assert "HTTP 404" in captured.err
        assert json.loads(captured.out) == {"error": "no such snapshot: missing"}

    def test_reset_refuses_without_yes(self, fake_proxy, capsys):
        assert ctl.main(["reset", "--proxy", fake_proxy.base_url]) == 2
        assert "--yes" in capsys.readouterr().err
        assert fake_proxy.calls == []

    def test_reset_with_yes_sends_confirm(self, fake_proxy, capsys):
        assert ctl.main(["reset", "--yes", "--proxy", fake_proxy.base_url]) == 0
        assert fake_proxy.calls == [("/_control/reset", {"confirm": True})]

    def test_load_transcript(self, fake_proxy, capsys):
        assert ctl.main(["load-transcript", "01a08337", "--proxy", fake_proxy.base_url]) == 0
        assert fake_proxy.calls == [("/_control/load-transcript", {"session": "01a08337"})]

    def test_proxy_url_from_env(self, fake_proxy, monkeypatch, capsys):
        monkeypatch.setenv("MEMORY_CONDENSE_PROXY", fake_proxy.base_url + "/")
        assert ctl.main(["save"]) == 0
        assert fake_proxy.calls == [("/_control/save", {})]

    def test_default_proxy_url(self, monkeypatch):
        monkeypatch.delenv("MEMORY_CONDENSE_PROXY", raising=False)
        assert ctl.default_proxy_url() == "http://127.0.0.1:8787"

    def test_connection_error_hint(self, closed_port, capsys):
        assert ctl.main(["save", "--proxy", closed_port]) == 1
        err = capsys.readouterr().err
        assert "not reachable" in err and "memory-condense proxy" in err
        assert err.count("\n") == 1


class TestTopLevelDispatch:
    def test_ctl_is_routed_from_installed_command(self, status_file, capsys):
        from memory_condense.cli import main as root

        assert root(["ctl", "status", "--status", str(status_file)]) == 0
        assert "ingest 52%" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------


class TestMCPControlTools:
    @pytest.fixture(autouse=True)
    def _point_at_fake(self, fake_proxy, monkeypatch):
        monkeypatch.setenv("MEMORY_CONDENSE_PROXY", fake_proxy.base_url)
        self.proxy = fake_proxy

    def test_memory_save(self):
        out = mcp_server.memory_save()
        assert self.proxy.calls == [("/_control/save", {})]
        assert json.loads(out)["ok"] is True

    def test_memory_save_named(self):
        mcp_server.memory_save("milestone")
        assert self.proxy.calls == [("/_control/save", {"name": "milestone"})]

    def test_memory_load(self):
        out = mcp_server.memory_load("milestone")
        assert self.proxy.calls == [("/_control/load", {"name": "milestone"})]
        assert json.loads(out)["echo"] == {"name": "milestone"}

    def test_memory_load_requires_name(self):
        assert "required" in mcp_server.memory_load("  ")
        assert self.proxy.calls == []

    def test_memory_load_refusal_is_text_not_exception(self):
        out = mcp_server.memory_load("missing")
        assert "refused" in out and "no such snapshot" in out

    def test_memory_reset_refuses_without_confirm(self):
        out = mcp_server.memory_reset()
        assert "refused" in out.lower()
        assert "confirm" in out
        assert self.proxy.calls == []

    def test_memory_reset_with_confirm(self):
        mcp_server.memory_reset(confirm=True)
        assert self.proxy.calls == [("/_control/reset", {"confirm": True})]

    def test_memory_load_transcript(self):
        mcp_server.memory_load_transcript("01a08337")
        assert self.proxy.calls == [("/_control/load-transcript", {"session": "01a08337"})]

    def test_memory_reset_docstring_demands_confirmation(self):
        assert "explicit" in mcp_server.memory_reset.__doc__
        assert "confirm=True" in mcp_server.memory_reset.__doc__


class TestMCPProxyDown:
    def test_friendly_text_when_proxy_is_down(self, closed_port, monkeypatch):
        monkeypatch.setenv("MEMORY_CONDENSE_PROXY", closed_port)
        out = mcp_server.memory_save()
        assert "not reachable" in out
        assert "nothing was changed" in out


class TestMemoryStatsAppendsStatusLine:
    @pytest.fixture
    def server(self, tmp_path, monkeypatch):
        from memory_condense.application.condenser import MemoryCondenser
        from tests.test_condenser import FakeEmbedder

        store = tmp_path / "store"
        monkeypatch.setenv("MEMORY_CONDENSE_DATA_DIR", str(store))
        monkeypatch.delenv("MEMORY_CONDENSE_STATUS", raising=False)
        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        condenser = MemoryCondenser(data_dir=store, embedder=FakeEmbedder())
        monkeypatch.setattr(mcp_server, "_condenser", condenser)
        yield store
        condenser.close()
        monkeypatch.setattr(mcp_server, "_condenser", None)

    def test_appends_one_line_when_status_exists(self, server):
        write_status(server / "control" / "status.json", _ingesting_doc())
        out = mcp_server.memory_stats()
        lines = out.splitlines()
        assert lines[0].startswith("Store: ")
        assert "Active memories: 0" in out
        assert lines[-1] == "Proxy: mem ▸ ingest 52% (ETA 1m38s) · 1.21M tok · q:1+3 · serving 48k of 1.10M"

    def test_silent_when_no_status(self, server, monkeypatch):
        monkeypatch.setattr(mcp_server, "locate_status_file", lambda *a, **k: None)
        assert "Proxy:" not in mcp_server.memory_stats()
