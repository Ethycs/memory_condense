"""Control core: snapshots, the swappable store, bulk loading, the publisher, routes.

Everything here is offline: temp data dirs, a real temp SQLite for
``VACUUM INTO``, a real :class:`CaptureQueue` with list/no-op sinks, and a
mock upstream behind Starlette's TestClient.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

from memory_condense.control.bulk import BulkLoader, RateWindow, exchange_text, resolve_transcript
from memory_condense.control.publisher import StatusPublisher
from memory_condense.control.snapshots import CondenserHandle, SnapshotStore, validate_snapshot_name
from memory_condense.control.status import empty_status, one_line, read_status, status_path_for
from memory_condense.interfaces.proxy_scheduler import CaptureQueue
from memory_condense.interfaces.proxy_server import ProxyConfig, build_app, capture_health


# -- helpers -----------------------------------------------------------------

def _wait(predicate, timeout: float = 15.0, step: float = 0.02) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(step)
    raise AssertionError("condition not met before timeout")


def _make_db(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE IF NOT EXISTS turns (id INTEGER PRIMARY KEY, text TEXT NOT NULL)")
        conn.executemany("INSERT INTO turns (text) VALUES (?)", [(row,) for row in rows])
        conn.commit()
    finally:
        conn.close()


def _rows(path: Path) -> list[str]:
    conn = sqlite3.connect(str(path))
    try:
        return [row[0] for row in conn.execute("SELECT text FROM turns ORDER BY id")]
    finally:
        conn.close()


def _claude_record(kind: str, content, **extra) -> str:
    record = {"type": kind, "uuid": f"{kind}-{len(json.dumps(content))}", "timestamp": "2026-10-13T08:00:00Z",
              "message": {"role": kind, "content": content}}
    record.update(extra)
    return json.dumps(record)


def _write_claude_transcript(path: Path, exchanges: int) -> list[tuple[str, str]]:
    """``exchanges`` user/assistant pairs plus the noise a real transcript carries."""
    lines: list[str] = []
    pairs: list[tuple[str, str]] = []
    lines.append(json.dumps({"type": "summary", "summary": "irrelevant"}))
    lines.append(_claude_record("user", "meta prompt", isMeta=True))
    for i in range(exchanges):
        user = f"question number {i} with enough words to count"
        reply = f"answer number {i} spelled out in prose"
        pairs.append((user, reply))
        lines.append(_claude_record("user", user))
        lines.append(_claude_record("assistant", [{"type": "text", "text": reply}]))
        lines.append(_claude_record("assistant", [{"type": "tool_use", "id": f"t{i}", "name": "Read", "input": {}}]))
        lines.append(_claude_record("user", [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "x" * 50}]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return pairs


class FakeCondenser:
    """What the sink and the handle need: ``capture_many``, ``close``, a db path."""

    instances: list["FakeCondenser"] = []

    def __init__(self, data_dir: Path) -> None:
        self.database_path = data_dir / "memory.db"
        data_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.database_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("CREATE TABLE IF NOT EXISTS turns (id INTEGER PRIMARY KEY, text TEXT NOT NULL)")
        self.conn.commit()
        self.closed = False
        self.captured: list[tuple] = []
        FakeCondenser.instances.append(self)

    def capture_many(self, records):
        if self.closed:
            raise RuntimeError("closed condenser")
        self.captured.extend(records)
        self.conn.executemany("INSERT INTO turns (text) VALUES (?)", [(r[1],) for r in records])
        self.conn.commit()
        return []

    def close(self) -> None:
        self.closed = True
        self.conn.close()


class _QueueOnLoop:
    """A real CaptureQueue running on a background event loop thread."""

    def __init__(self, sink, **kwargs) -> None:
        self.loop = asyncio.new_event_loop()
        self.queue = CaptureQueue(sink, **kwargs)
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)

    def __enter__(self) -> "_QueueOnLoop":
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._start(), self.loop).result(5)
        return self

    async def _start(self) -> None:
        self.queue.start()

    def __exit__(self, *_exc) -> None:
        asyncio.run_coroutine_threadsafe(self.queue.stop(), self.loop).result(30)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(5)
        self.loop.close()


# -- snapshots ---------------------------------------------------------------

class TestSnapshotStore:
    def test_save_is_a_consistent_copy_with_a_record(self, tmp_path):
        store = SnapshotStore(tmp_path)
        _make_db(store.db_path, ["a", "b"])
        record = store.save()
        assert record["file"].endswith(".sqlite")
        assert Path(record["path"]).is_file()
        assert record["bytes"] > 0
        assert _rows(Path(record["path"])) == ["a", "b"]
        assert [r["name"] for r in store.list()] == [record["name"]]
        assert store.latest()["name"] == record["name"]
        assert not [p for p in store.snapshot_dir.iterdir() if p.name.startswith(".")], "no temp files left"

    def test_named_save_refuses_overwrite_and_validates_names(self, tmp_path):
        store = SnapshotStore(tmp_path)
        _make_db(store.db_path, ["a"])
        record = store.save("base")
        assert record["name"] == "base"
        assert store.path_for("base") == Path(record["path"])
        assert store.path_for("base.sqlite") == Path(record["path"])
        with pytest.raises(FileExistsError):
            store.save("base")
        for bad in ("", "../x", "a/b", "a\\b", ".hidden", "a..b", "x" * 200, "sp ace"):
            with pytest.raises(ValueError):
                validate_snapshot_name(bad)
            with pytest.raises(ValueError):
                store.path_for(bad)

    def test_save_without_a_store_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SnapshotStore(tmp_path).save()

    def test_restore_and_clear_replace_the_closed_store_and_drop_sidecars(self, tmp_path):
        store = SnapshotStore(tmp_path)
        _make_db(store.db_path, ["a"])
        store.save("base")
        _make_db(store.db_path, ["b"])
        assert _rows(store.db_path) == ["a", "b"]
        Path(f"{store.db_path}-wal").write_bytes(b"stale")
        Path(f"{store.db_path}-shm").write_bytes(b"stale")
        store.index_path.write_bytes(b"stale ann image")

        store.restore("base")
        assert _rows(store.db_path) == ["a"]
        assert not Path(f"{store.db_path}-wal").exists()
        assert not Path(f"{store.db_path}-shm").exists()
        assert not store.index_path.exists()
        with pytest.raises(FileNotFoundError):
            store.restore("missing")

        store.clear()
        assert not store.db_path.exists()


class TestCondenserHandle:
    def test_sink_runs_under_lock_and_swap_reopens_a_new_generation(self, tmp_path):
        FakeCondenser.instances.clear()
        handle = CondenserHandle(lambda: FakeCondenser(tmp_path), data_dir=tmp_path)
        assert handle.closed
        handle.open()
        first = handle.condenser
        assert handle.generation == 1
        assert handle.db_path == tmp_path / "memory.db"

        from memory_condense.interfaces.proxy_capture import ExchangeCapture
        from memory_condense.ingest.transcripts import TranscriptMessage

        capture = ExchangeCapture(
            provider="anthropic", conversation_id="c", model=None,
            prompt=(TranscriptMessage(role="user", text="hi", conversation_id="c", message_id="m1", ordinal=0),),
            reply_text="hello", streamed=False, request_sha256="a" * 64, prompt_tokens_estimate=1,
        )
        sink = handle.capture_sink()
        assert sink(capture) is True
        assert [r[1] for r in first.captured] == ["hi", "hello"]

        snapshots = SnapshotStore(tmp_path)
        snapshots.save("base")
        sink(capture)
        assert _rows(handle.db_path) == ["hi", "hello", "hi", "hello"]

        handle.swap(lambda: snapshots.restore("base"))
        assert first.closed
        assert handle.generation == 2
        assert handle.condenser is not first
        assert _rows(handle.db_path) == ["hi", "hello"]
        sink(capture)  # the same sink now writes into the reopened store
        assert handle.condenser.captured and first.captured != handle.condenser.captured

        handle.close()
        assert handle.closed
        with pytest.raises(RuntimeError):
            sink(capture)

    def test_failed_reopen_leaves_the_handle_closed(self, tmp_path):
        calls = {"n": 0}

        def factory():
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("disk gone")
            return FakeCondenser(tmp_path)

        handle = CondenserHandle(factory, data_dir=tmp_path)
        handle.open()
        with pytest.raises(OSError):
            handle.swap(lambda: None)
        assert handle.closed
        with pytest.raises(RuntimeError):
            handle.condenser


# -- bulk --------------------------------------------------------------------

class TestRateWindow:
    def test_rates_are_recent_window_not_lifetime(self):
        window = RateWindow(window_s=30)
        assert window.rates(now=0.0)["tokens_per_s_recent"] == 0.0
        assert window.eta_s(1000, now=0.0) is None
        window.add(0.0, 10_000, 10)  # a fast burst long ago
        window.add(100.0, 100, 1)
        window.add(110.0, 100, 1)
        rates = window.rates(now=120.0)
        assert rates["window_s"] == 30
        assert rates["tokens_per_s_recent"] == pytest.approx(200 / 20, rel=1e-3)
        assert rates["records_per_s_recent"] == pytest.approx(2 / 20, rel=1e-3)
        assert rates["avg_s_per_1k_tokens"] == pytest.approx(100.0, rel=1e-3)
        assert window.eta_s(1000, now=120.0) == 100
        assert window.rates(now=200.0)["tokens_per_s_recent"] == 0.0  # stalled: nothing recent


class TestExchangeText:
    def test_claude_and_codex_shapes(self):
        assert exchange_text(json.loads(_claude_record("user", "hello"))) == ("user", "hello")
        assert exchange_text(json.loads(_claude_record("assistant", [{"type": "text", "text": "a"}, {"type": "tool_use"}]))) == ("assistant", "a")
        assert exchange_text(json.loads(_claude_record("user", [{"type": "tool_result", "content": "x"}]))) is None
        assert exchange_text(json.loads(_claude_record("user", "meta", isMeta=True))) is None
        codex = {"type": "response_item", "payload": {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "codex q"}]}}
        assert exchange_text(codex) == ("user", "codex q")
        assert exchange_text({"type": "response_item", "payload": {"type": "function_call"}}) is None
        assert exchange_text({"type": "summary"}) is None
        assert exchange_text("not a record") is None

    def test_resolve_transcript_takes_paths_without_discovery(self, tmp_path):
        path = tmp_path / "01a08337-abcd.jsonl"
        _write_claude_transcript(path, 1)
        resolved, session_id, kind = resolve_transcript(str(path))
        assert Path(resolved) == path and session_id == "01a08337-abcd" and kind == "claude"
        codex = tmp_path / "rollout-2026-10-13T08-00-00-0199b2c4-1111-2222-3333-444444444444.jsonl"
        codex.write_text(json.dumps({"type": "session_meta", "payload": {}}) + "\n", encoding="utf-8")
        _, session_id, kind = resolve_transcript(str(codex))
        assert kind == "codex" and session_id == "0199b2c4-1111-2222-3333-444444444444"


class TestBulkLoader:
    def test_job_feeds_exchanges_through_the_queue_and_reports_progress(self, tmp_path):
        path = tmp_path / "01a08337-session.jsonl"
        pairs = _write_claude_transcript(path, 5)
        total_lines = len(path.read_bytes().splitlines())
        sunk = []
        events: list[str] = []
        with _QueueOnLoop(sunk.append, maxsize=4) as running:
            loader = BulkLoader(running.queue, capacity=4, loop=running.loop, listener=events.append)
            job = loader.submit(str(path))
            assert job.id == "bulk-01a08337"
            assert job.session == "01a08337-session"
            _wait(lambda: job.finished)
            _wait(lambda: running.queue.ingested == len(pairs))
            loader.stop()

        assert job.state == "done", job.error
        assert job.records_total == total_lines
        assert job.records_done == total_lines
        assert job.bytes_total == path.stat().st_size
        assert job.bytes_done == job.bytes_total
        assert job.tokens_done == sum(len(line) // 4 for line in path.read_bytes().splitlines())
        assert job.captures_emitted == len(pairs)
        assert [(c.prompt[0].text, c.reply_text) for c in sunk] == pairs
        assert {c.conversation_id for c in sunk} == {"transcript:01a08337-session"}
        assert all(c.provider == "anthropic" for c in sunk)
        assert "state" in events and "progress" in events

        snapshot = loader.snapshot()
        assert snapshot["state"] == "idle" and snapshot["current"] is None
        assert snapshot["totals"] == {"tokens_ingested": job.tokens_done, "records_ingested": total_lines, "sessions": 1}
        assert snapshot["last_job"]["state"] == "done"
        assert snapshot["jobs"] == []

    def test_jobs_are_fifo_with_positions_and_cancellable(self, tmp_path):
        first = tmp_path / "aaaa1111.jsonl"
        second = tmp_path / "bbbb2222.jsonl"
        _write_claude_transcript(first, 40)
        _write_claude_transcript(second, 2)
        release = threading.Event()

        def slow_sink(_capture):
            release.wait(5)

        with _QueueOnLoop(slow_sink, maxsize=1, shutdown_timeout_seconds=30) as running:
            loader = BulkLoader(running.queue, capacity=1, loop=running.loop)
            job1 = loader.submit(str(first))
            job2 = loader.submit(str(second))
            _wait(lambda: job1.state == "running")
            snapshot = loader.snapshot()
            assert snapshot["state"] == "ingesting"
            assert snapshot["current"]["job"] == job1.id
            assert snapshot["current"]["records_total"] == len(first.read_bytes().splitlines())
            assert [(j["id"], j["position"]) for j in snapshot["jobs"]] == [(job2.id, 1)]

            assert loader.cancel(job1.id) is True
            _wait(lambda: job1.finished)
            assert job1.state == "cancelled"
            assert job1.records_done < job1.records_total
            release.set()
            _wait(lambda: job2.finished)
            assert job2.state == "done", job2.error
            assert loader.cancel("bulk-nope") is False
            loader.stop()
        assert [j.id for j in loader.history()] == [job1.id, job2.id]

    def test_unknown_session_is_rejected_at_submit(self, tmp_path):
        with _QueueOnLoop(lambda c: None) as running:
            loader = BulkLoader(running.queue, capacity=4, loop=running.loop)
            with pytest.raises(FileNotFoundError):
                loader.submit(str(tmp_path / "does-not-exist-anywhere-0000.jsonl"))
            loader.stop()


# -- publisher ---------------------------------------------------------------

class _Counters:
    state = "running"
    worker_quiesced = False
    in_flight = 2
    depth = 1
    retained_bytes = 4096
    rejected_saturated = 0
    capture_terminal_failures = 0
    ingested = 7


class TestStatusPublisher:
    def test_document_is_complete_and_written_atomically(self, tmp_path):
        counters = _Counters()
        publisher = StatusPublisher(
            mode="observe", captures=counters, capacity=256,
            health=lambda: capture_health(counters), data_dir=tmp_path,
            snapshots=SnapshotStore(tmp_path), pid=4242,
        )
        _make_db(tmp_path / "memory.db", ["a"])
        SnapshotStore(tmp_path).save("base")
        publisher.record_serving(1_000_000, 50_000, True)
        publisher.record_serving(1_100_000, 48_200, True)
        publisher.record_capture(300, "conv-1")
        publisher.record_capture(200, "conv-1")
        doc = publisher.publish()

        assert set(doc) >= set(empty_status())
        assert doc["pid"] == 4242 and doc["mode"] == "observe"
        assert doc["health"] == {"status": "ok", "reasons": []}
        assert doc["queue"]["capacity"] == 256 and doc["queue"]["in_flight"] == 2
        assert doc["queue"]["retained_bytes"] == 4096 and doc["queue"]["jobs"] == []
        assert doc["state"] == "ingesting"  # captures in flight
        assert doc["serving"] == {
            "requests": 2, "rewritten": 2, "static_tokens_last": 1_100_000,
            "served_tokens_last": 48_200, "saved_tokens_total": 950_000 + 1_051_800,
        }
        assert doc["totals"]["records_ingested"] == 7
        assert doc["totals"]["tokens_ingested"] == 500
        assert doc["totals"]["sessions"] == 1
        assert doc["totals"]["store_bytes"] > 0
        assert doc["store"]["snapshots"] == ["base.sqlite"] and doc["store"]["last_snapshot"] == "base.sqlite"
        assert doc["store"]["path"] == str(tmp_path / "memory.db")

        on_disk = read_status(status_path_for(tmp_path))
        assert on_disk is not None and on_disk["serving"] == doc["serving"]
        assert not [p for p in (tmp_path / "control").iterdir() if p.suffix == ".tmp"]
        assert "serving 48k of 1.10M" in one_line(on_disk)

    def test_state_override_and_health_degradation(self, tmp_path):
        counters = _Counters()
        counters.in_flight = 0
        counters.depth = 0
        publisher = StatusPublisher(mode="observe", captures=counters, capacity=8,
                                    health=lambda: capture_health(counters), data_dir=tmp_path)
        assert publisher.publish()["state"] == "idle"
        assert publisher.set_state("swapping")["state"] == "swapping"
        assert read_status(status_path_for(tmp_path))["state"] == "swapping"
        doc = publisher.set_state("error", error="boom")
        assert doc["state"] == "error" and doc["error"] == "boom"
        assert publisher.set_state(None)["state"] == "idle"
        counters.rejected_saturated = 3
        doc = publisher.publish()
        assert doc["health"] == {"status": "degraded", "reasons": ["capture_admission_loss"]}
        counters.state = "failed"
        doc = publisher.publish()
        assert doc["health"]["status"] == "failed" and doc["state"] == "error"

    def test_ticker_writes_while_dirty_and_stops(self, tmp_path):
        counters = _Counters()
        publisher = StatusPublisher(mode="observe", captures=counters, capacity=8,
                                    health=lambda: capture_health(counters), data_dir=tmp_path, tick_s=0.05)
        publisher.start()
        assert publisher.writes == 1
        publisher.record_serving(10, 10, False)
        _wait(lambda: publisher.writes >= 2, timeout=3)
        writes = publisher.writes
        time.sleep(0.2)
        assert publisher.writes == writes, "idle publisher does not rewrite"
        publisher.stop()
        assert read_status(status_path_for(tmp_path))["serving"]["requests"] == 1


# -- routes ------------------------------------------------------------------

def _upstream(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"content": [{"type": "text", "text": "upstream answer"}]})


def _app(tmp_path, *, sink=None, store=None, **config):
    return build_app(
        config=ProxyConfig(**config),
        sink=sink,
        client=httpx.AsyncClient(transport=httpx.MockTransport(_upstream)),
        data_dir=tmp_path,
        store=store,
    )


def _chat(client: TestClient, text: str = "hi") -> None:
    response = client.post(
        "/v1/messages",
        json={"model": "m", "messages": [{"role": "user", "content": text}]},
        headers={"x-api-key": "k", "x-memory-conversation-id": "conv-a"},
    )
    assert response.status_code == 200


class TestControlRoutes:
    def test_status_route_publishes_the_document_and_serving_receipts(self, tmp_path):
        captured = []
        app = _app(tmp_path, sink=captured.append)
        with TestClient(app) as client:
            status_file = status_path_for(tmp_path)
            assert read_status(status_file)["state"] == "idle", "written at startup"
            _chat(client, "first question")
            _wait(lambda: app.state.captures.ingested == 1)
            doc = client.get("/_control/status").json()
            assert doc["schema"] == 1 and doc["mode"] == "observe" and doc["pid"] == os.getpid()
            assert doc["serving"]["requests"] == 1 and doc["serving"]["rewritten"] == 0
            assert doc["serving"]["static_tokens_last"] == doc["serving"]["served_tokens_last"]
            assert doc["totals"]["records_ingested"] == 1
            assert doc["totals"]["tokens_ingested"] == (len("first question") + len("upstream answer")) // 4
            assert doc["totals"]["sessions"] == 1
            assert doc["queue"]["capacity"] == 256
            assert doc["health"]["status"] == client.get("/_memory/health").json()["status"]
            assert read_status(status_file)["serving"]["requests"] == 1
            assert doc["store"]["snapshot_dir"] == str(tmp_path / "snapshots")
        assert read_status(status_file) is not None

    def test_health_route_is_unchanged_and_shared_with_status(self, tmp_path):
        app = _app(tmp_path)
        with TestClient(app) as client:
            health = client.get("/_memory/health").json()
            doc = client.get("/_control/status").json()
            assert health["status"] == "ok" and health["health_reasons"] == []
            assert doc["health"] == {"status": health["status"], "reasons": health["health_reasons"]}

    def test_save_requires_a_store_file_and_validates_names(self, tmp_path):
        app = _app(tmp_path)
        with TestClient(app) as client:
            assert client.post("/_control/save", json={}).status_code == 404
            _make_db(tmp_path / "memory.db", ["a"])
            record = client.post("/_control/save", json={"name": "base"}).json()
            assert record["name"] == "base" and (tmp_path / "snapshots" / "base.sqlite").is_file()
            assert client.post("/_control/save", json={"name": "base"}).status_code == 409
            assert client.post("/_control/save", json={"name": "../x"}).status_code == 400
            assert client.post("/_control/save", json={"name": 7}).status_code == 400
            auto = client.post("/_control/save", content=b"not json").json()
            assert auto["name"] != "base"
            doc = client.get("/_control/status").json()
            assert set(doc["store"]["snapshots"]) == {"base.sqlite", auto["file"]}

    def test_load_and_reset_need_a_swappable_store(self, tmp_path):
        app = _app(tmp_path)
        with TestClient(app) as client:
            assert client.post("/_control/load", json={"name": "x"}).status_code == 501
            assert client.post("/_control/reset", json={"confirm": True}).status_code == 501
            assert client.post("/_control/reset", json={}).status_code == 400
            assert client.post("/_control/reset", json={"confirm": "yes"}).status_code == 400

    def test_load_swaps_the_store_under_the_sink_lock(self, tmp_path):
        FakeCondenser.instances.clear()
        store = CondenserHandle(lambda: FakeCondenser(tmp_path), data_dir=tmp_path)
        store.open()
        app = _app(tmp_path, sink=store.capture_sink(), store=store)
        with TestClient(app) as client:
            _chat(client, "before snapshot")
            _wait(lambda: app.state.captures.ingested == 1)
            assert client.post("/_control/save", json={"name": "base"}).status_code == 200
            _chat(client, "after snapshot")
            _wait(lambda: app.state.captures.ingested == 2)
            assert _rows(tmp_path / "memory.db") == ["before snapshot", "upstream answer", "after snapshot", "upstream answer"]

            assert client.post("/_control/load", json={}).status_code == 400
            assert client.post("/_control/load", json={"name": "a/b"}).status_code == 400
            assert client.post("/_control/load", json={"name": "missing"}).status_code == 404

            reply = client.post("/_control/load", json={"name": "base"})
            assert reply.status_code == 200, reply.text
            assert reply.json()["loaded"] == "base.sqlite" and reply.json()["generation"] == 2
            assert FakeCondenser.instances[0].closed and not store.closed
            assert _rows(tmp_path / "memory.db") == ["before snapshot", "upstream answer"]
            doc = client.get("/_control/status").json()
            assert doc["state"] == "idle" and doc["error"] is None

            _chat(client, "after load")  # captures land in the reopened store
            _wait(lambda: app.state.captures.ingested == 3)
            assert _rows(tmp_path / "memory.db") == ["before snapshot", "upstream answer", "after load", "upstream answer"]
            assert app.state.captures.capture_terminal_failures == 0

            reply = client.post("/_control/reset", json={"confirm": True})
            assert reply.status_code == 200, reply.text
            body = reply.json()
            assert body["reset"] is True and body["generation"] == 3
            assert body["backup"]["file"] in {p.name for p in (tmp_path / "snapshots").iterdir()}
            assert _rows(Path(body["backup"]["path"])) == ["before snapshot", "upstream answer", "after load", "upstream answer"]
            assert _rows(tmp_path / "memory.db") == []
            _chat(client, "fresh")
            _wait(lambda: app.state.captures.ingested == 4)
            assert _rows(tmp_path / "memory.db") == ["fresh", "upstream answer"]
            doc = client.get("/_control/status").json()
            assert doc["state"] == "idle" and len(doc["store"]["snapshots"]) == 2
        store.close()

    def test_load_transcript_enqueues_a_bulk_job_on_the_live_queue(self, tmp_path):
        transcript = tmp_path / "sessions" / "01a08337-feed.jsonl"
        pairs = _write_claude_transcript(transcript, 6)
        captured = []
        app = _app(tmp_path, sink=captured.append)
        with TestClient(app) as client:
            missing = client.post("/_control/load-transcript", json={"session": "no-such-session-000"})
            assert missing.status_code == 404
            assert client.post("/_control/load-transcript", json={}).status_code == 400
            reply = client.post("/_control/load-transcript", json={"session": str(transcript)})
            assert reply.status_code == 202, reply.text
            job = reply.json()["job"]
            assert job["id"] == "bulk-01a08337" and job["kind"] == "transcript" and job["position"] == 1
            assert job["bytes"] == transcript.stat().st_size

            def finished():
                doc = client.get("/_control/status").json()
                return doc["current"] is None and (doc.get("last_job") or {}).get("job") == job["id"]

            _wait(finished)
            _wait(lambda: app.state.captures.ingested == len(pairs))
            doc = client.get("/_control/status").json()
            assert doc["state"] == "idle" and doc["queue"]["jobs"] == []
            assert doc["last_job"]["state"] == "done" and doc["last_job"]["records_done"] == doc["last_job"]["records_total"]
            assert doc["totals"]["sessions"] == 1
            assert doc["totals"]["records_ingested"] == len(pairs)
            assert doc["totals"]["tokens_ingested"] == doc["last_job"]["tokens_done"]
            assert doc["rates"]["window_s"] == 30
            assert [(c.prompt[0].text, c.reply_text) for c in captured] == pairs
            assert read_status(status_path_for(tmp_path))["last_job"]["state"] == "done"

    def test_augment_mode_reports_status_but_cannot_swap_or_bulk_load(self, tmp_path):
        class FakeMemory:
            closed = False

            def status(self):
                return {"active_sessions": 1, "pending_events": 0, "pending_feedback": 0,
                        "failed_sessions": 1, "inline_memory": True}

            def close(self):
                self.closed = True

        memory = FakeMemory()
        app = build_app(
            config=ProxyConfig(mode="augment"), memory=memory,
            client=httpx.AsyncClient(transport=httpx.MockTransport(_upstream)), data_dir=tmp_path,
        )
        with TestClient(app) as client:
            doc = client.get("/_control/status").json()
            assert doc["mode"] == "augment"
            assert doc["health"] == {"status": "degraded", "reasons": ["memory_sessions_failed"]}
            assert client.get("/_memory/health").json()["status"] == "degraded"
            assert doc["memory"]["active_sessions"] == 1
            for route, body in (("save", {}), ("load", {"name": "x"}), ("reset", {"confirm": True}),
                                ("load-transcript", {"session": "x"})):
                assert client.post(f"/_control/{route}", json=body).status_code == 501, route
        assert memory.closed
