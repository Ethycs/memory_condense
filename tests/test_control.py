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
from types import SimpleNamespace
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

from memory_condense.control.bulk import (
    BulkLoader,
    RateWindow,
    exchange_text,
    ingested_tokens,
    resolve_transcript,
)
from memory_condense.control.publisher import StatusPublisher
from memory_condense.control.snapshots import (
    CondenserHandle,
    SessionSnapshotStore,
    SnapshotStore,
    tree_bytes,
    validate_snapshot_name,
)
from memory_condense.control.status import (
    candidate_status_files,
    empty_status,
    one_line,
    read_status,
    status_path_for,
)
from memory_condense.interfaces.proxy_receipts import ServingReceipt, prompt_tokens
from memory_condense.interfaces.proxy_scheduler import CaptureQueue
from memory_condense.interfaces.proxy_server import (
    ProxyConfig,
    build_app,
    capture_health,
    default_data_dir,
    memory_health,
)


# -- helpers -----------------------------------------------------------------

def _receipt(static, served, rewritten, *, usage=None, mode="augment") -> ServingReceipt:
    """A serving receipt with the numbers a test cares about.

    ``StatusPublisher.record_serving`` takes receipts, not loose positionals:
    the receipt has already decided savings and which served figure wins.
    """
    return ServingReceipt(
        format="anthropic", mode=mode, rewritten=rewritten, reason="test",
        static_tokens=static, served_tokens=served, usage_input_tokens=usage,
    )


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


def _session_db(root: Path, namespace: str, rows: list[str]) -> Path:
    """One conversation's store tree, with the sidecars and junk a real one has."""
    path = root / "sessions" / namespace / "chat" / "chat-events.sqlite"
    _make_db(path, rows)
    (path.parent / "chat-events.sqlite-wal").write_bytes(b"x" * 32)
    (path.parent / "chat-owner.lock").write_bytes(b"")
    (root / "sessions" / namespace / "actor.json").write_text(
        json.dumps({"case_id": namespace}), encoding="utf-8"
    )
    return path


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


@dataclass
class _Wire:
    """What ``proxy_server`` reads off a ``MemoryProxy`` reply."""

    status: int
    headers: dict
    body: bytes


class FakeSessions:
    """A ``MemoryProxy`` stand-in: the registry contract the control plane uses.

    Real augment sessions are ``ChatSession`` trees under
    ``<data-dir>/sessions/<ns>`` and cost model loads to open. What the control
    routes and the status publisher need from the proxy is ``status``,
    ``quiesce``, ``swap``, ``generation``, ``closed``, ``directory`` and
    ``close`` -- over stores that are genuinely closed while the files move.
    Here one "session" is one SQLite file.
    ``tests/test_proxy_memory.py`` exercises the real ``MemoryProxy.swap``.
    """

    def __init__(self, directory, *, failed: int = 0, serve=None) -> None:
        self.directory = Path(directory)
        self.lock = threading.RLock()
        self.closed = False
        self.generation = 0
        self.failed = failed
        self.open_namespaces: list[str] = []
        self.closed_stores: list[list[str]] = []
        #: ``serve(body) -> bytes | None`` decides what goes upstream for the
        #: visible answer: a rewritten body, the client's own bytes
        #: (passthrough), or ``None`` for an idempotent replay that sends
        #: nothing. The default echoes the client's body.
        self.serve = serve
        self.sent: list[bytes] = []
        self.summaries: list[bytes] = []

    def admit(self, namespace: str, rows: list[str]) -> Path:
        with self.lock:
            path = self.directory / "sessions" / namespace / "chat-events.sqlite"
            _make_db(path, rows)
            if namespace not in self.open_namespaces:
                self.open_namespaces.append(namespace)
            return path

    def exchange(self, body, provider, headers, send, summary_send=None, fmt=None):
        """The shape ``proxy_server`` drives: ``send(sent_bytes)`` per attempt.

        Like the real ``MemoryProxy``, this never returns the body it put on
        the wire -- it only hands it to ``send``, which is exactly how the
        proxy learns what was served. Summaries go through ``summary_send``
        and are separate calls, not the client's answer.
        """
        if summary_send is not None:
            self.summaries.append(b'{"summary": true}')
            summary_send(b'{"summary": true}')
        served = body if self.serve is None else self.serve(body)
        if served is None:
            # Nothing reached the upstream: an already-complete request being
            # replayed from the ledger.
            return _Wire(200, {"content-type": "application/json"}, b'{"replayed": true}')
        self.sent.append(served)
        return send(served)

    def status(self):
        # Deliberately lock-free, like MemoryProxy.status: a status request
        # must not queue behind a swap or a cold start.
        return {"active_sessions": len(self.open_namespaces), "pending_events": 0,
                "pending_feedback": 0, "failed_sessions": self.failed, "inline_memory": True}

    @contextmanager
    def quiesce(self):
        with self.lock:
            if self.closed:
                raise RuntimeError("memory proxy is closing")
            yield list(self.open_namespaces)

    def swap(self, mutate):
        with self.quiesce() as namespaces:
            self.closed_stores.append(namespaces)
            self.open_namespaces.clear()
            self.generation += 1
            return mutate()

    def close(self):
        self.closed = True


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


class TestSessionSnapshotStore:
    """Augment mode's store is a tree of per-conversation stores, not one file."""

    def test_save_mirrors_the_tree_with_a_manifest_and_skips_sidecars(self, tmp_path):
        store = SessionSnapshotStore(tmp_path)
        assert store.list() == [] and store.latest() is None and not store.has_store()
        with pytest.raises(FileNotFoundError):
            store.save()

        _session_db(tmp_path, "ns-a", ["a1", "a2"])
        _session_db(tmp_path, "ns-b", ["b1"])
        assert store.has_store()

        record = store.save("base")
        assert record["name"] == record["file"] == "base"
        assert record["bytes"] > 0 and record["file_count"] >= 5
        root = tmp_path / "snapshots" / "base"
        manifest = store.manifest("base")
        assert manifest["schema"] == 1 and manifest["kind"] == "sessions"
        assert manifest["namespaces"] == ["ns-a", "ns-b"]
        kinds = {entry["path"]: entry["kind"] for entry in manifest["files"]}
        assert kinds["ns-a/chat/chat-events.sqlite"] == "vacuum"
        assert kinds["ns-a/actor.json"] == "copy"
        # The WAL belongs to the live database and the owner lock to whoever
        # opens the session next; neither is part of a snapshot.
        assert not [path for path in kinds if path.endswith(("-wal", "-shm", ".lock"))]
        assert _rows(root / "sessions" / "ns-a" / "chat" / "chat-events.sqlite") == ["a1", "a2"]
        assert _rows(root / "sessions" / "ns-b" / "chat" / "chat-events.sqlite") == ["b1"]

        assert [r["file"] for r in store.list()] == ["base"]
        assert store.latest()["file"] == "base"
        assert store.has_snapshot("base") and not store.has_snapshot("other")
        with pytest.raises(FileExistsError):
            store.save("base")
        with pytest.raises(ValueError):
            store.save("../escape")
        auto = store.save()
        assert auto["file"] != "base"
        assert {r["file"] for r in store.list()} == {"base", auto["file"]}

        # A half-written snapshot and a foreign directory are not snapshots.
        (tmp_path / "snapshots" / ".partial.1.tmp").mkdir()
        (tmp_path / "snapshots" / "no-manifest").mkdir()
        assert {r["file"] for r in store.list()} == {"base", auto["file"]}

        # `db_path` is the tree, and it sizes the tree for the status document.
        assert str(store.db_path) == str(tmp_path / "sessions")
        assert store.db_path.is_dir() and store.db_path.name == "sessions"
        assert store.db_path.stat().st_size == tree_bytes(tmp_path / "sessions") > 0

    def test_restore_and_clear_replace_the_closed_tree(self, tmp_path):
        store = SessionSnapshotStore(tmp_path)
        live = _session_db(tmp_path, "ns-a", ["a1"])
        store.save("base")

        _make_db(live, ["a2"])
        _session_db(tmp_path, "ns-c", ["c1"])
        assert _rows(live) == ["a1", "a2"]

        with pytest.raises(FileNotFoundError):
            store.restore("missing")
        assert store.restore("base") == tmp_path / "sessions"
        assert _rows(live) == ["a1"]
        assert not (tmp_path / "sessions" / "ns-c").exists()
        assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")], "no staging left behind"

        store.clear()
        assert not (tmp_path / "sessions").exists() and not store.has_store()
        assert [r["file"] for r in store.list()] == ["base"], "snapshots survive a clear"
        store.clear()  # idempotent


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

    def test_a_failed_mutation_reopens_instead_of_wedging_the_handle(self, tmp_path):
        """A mutation that failed left the files alone; ingestion must survive.

        A restore whose ``os.replace`` hits a Windows sharing violation, or a
        pre-reset backup that cannot be written, are the common failures and
        neither touches the store. Staying closed would turn them into a dead
        proxy with no recovery short of a restart.
        """
        FakeCondenser.instances.clear()
        handle = CondenserHandle(lambda: FakeCondenser(tmp_path), data_dir=tmp_path)
        handle.open()
        first = handle.condenser

        def boom():
            raise PermissionError("the file is used by another process")

        with pytest.raises(PermissionError):
            handle.swap(boom)
        assert not handle.closed and handle.generation == 2
        assert handle.condenser is not first and first.closed

    def test_a_failed_mutation_whose_reopen_also_fails_reports_the_mutation(self, tmp_path):
        """The one unrecoverable case still ends closed, and still names the cause."""
        calls = {"n": 0}

        def factory():
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("disk gone")
            return FakeCondenser(tmp_path)

        handle = CondenserHandle(factory, data_dir=tmp_path)
        handle.open()
        with pytest.raises(PermissionError):
            handle.swap(lambda: (_ for _ in ()).throw(PermissionError("in use")))
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
        # Two figures, two units. `tokens_done` is file progress over every
        # record -- tool calls, tool results, the summary and the meta prompt
        # included -- and is what the rate window and the ETA are measured
        # against. `tokens_ingested` is what the captures actually carry, the
        # same count the live proxy path reports, and is strictly smaller
        # here because this transcript is mostly noise.
        assert job.tokens_ingested == sum(ingested_tokens(c) for c in sunk)
        assert 0 < job.tokens_ingested < job.tokens_done
        assert job.captures_emitted == len(pairs)
        assert [(c.prompt[0].text, c.reply_text) for c in sunk] == pairs
        assert {c.conversation_id for c in sunk} == {"transcript:01a08337-session"}
        assert all(c.provider == "anthropic" for c in sunk)
        assert "state" in events and "progress" in events

        snapshot = loader.snapshot()
        assert snapshot["state"] == "idle" and snapshot["current"] is None
        assert snapshot["totals"] == {"tokens_ingested": job.tokens_ingested,
                                      "records_ingested": total_lines, "sessions": 1}
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
        publisher.record_serving(_receipt(1_000_000, 50_000, True))
        publisher.record_serving(_receipt(1_100_000, 48_200, True))
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
            "levels": {"local": 0, "handoff": 0, "passthrough": 0},
            "passthrough_forced": 0, "passthrough_degraded": 0, "index_deferred": 0,
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
        publisher.record_serving(_receipt(10, 10, False, mode="observe"))
        _wait(lambda: publisher.writes >= 2, timeout=3)
        writes = publisher.writes
        time.sleep(0.2)
        assert publisher.writes == writes, "idle publisher does not rewrite"
        publisher.stop()
        assert read_status(status_path_for(tmp_path))["serving"]["requests"] == 1

    def test_the_receipt_decides_the_served_figure_and_the_savings(self, tmp_path):
        """The publisher accumulates; it does not recompute (finding 3)."""
        counters = _Counters()
        publisher = StatusPublisher(mode="augment", captures=counters, capacity=8,
                                    health=lambda: ("ok", []), data_dir=tmp_path)
        # The upstream's own count beats the estimate on both the published
        # served figure and the savings.
        publisher.record_serving(_receipt(1_000, 400, True, usage=250))
        serving = publisher.publish()["serving"]
        assert serving["served_tokens_last"] == 250
        assert serving["saved_tokens_total"] == 750

        # An estimate that undercounts what the upstream then reports does not
        # subtract from the total.
        publisher.record_serving(_receipt(1_000, 400, True, usage=1_500))
        serving = publisher.publish()["serving"]
        assert serving["served_tokens_last"] == 1_500
        assert serving["saved_tokens_total"] == 750

        # An unknown served side is unknown, not zero.
        publisher.record_serving(_receipt(1_000, None, False))
        serving = publisher.publish()["serving"]
        assert serving["served_tokens_last"] is None
        assert serving["requests"] == 3 and serving["rewritten"] == 2
        assert serving["saved_tokens_total"] == 750

    def test_publishes_are_ordered_and_never_block_the_request_path(self, tmp_path):
        """Finding 9a: a document assembled earlier cannot land on top.

        The old code built ``document()`` outside the lock the writer then
        took, so a publisher that assembled first could write *after* one that
        assembled later and leave the file stale until the next tick.
        """
        counters = _Counters()
        counters.in_flight = counters.depth = 0
        publisher = StatusPublisher(mode="observe", captures=counters, capacity=8,
                                    health=lambda: ("ok", []), data_dir=tmp_path)
        publisher.publish("seed")
        entered, release = threading.Event(), threading.Event()
        real = publisher.document

        def document_then_stall():
            doc = real()  # assembled now, written only once released
            if not entered.is_set():
                entered.set()
                assert release.wait(15)
            return doc

        publisher.document = document_then_stall
        first = threading.Thread(target=lambda: publisher.publish("first"), daemon=True)
        first.start()
        assert entered.wait(15)

        # The inference path does not queue behind a publish in progress:
        # `record_serving` takes the counter lock, not the publish lock.
        publisher.record_serving(_receipt(1_000, 10, True, mode="observe"))
        assert publisher.last_document["serving"]["requests"] == 0, "nothing written yet"

        second = threading.Thread(target=lambda: publisher.publish("second"), daemon=True)
        second.start()
        time.sleep(0.2)
        assert publisher.writes == 1, "the second publisher waits for the first"
        release.set()
        first.join(15)
        second.join(15)

        assert publisher.writes == 3
        # The older document was written first and the newer one after it, so
        # what a reader finds is never behind what the publisher knows.
        assert publisher.last_document["serving"]["requests"] == 1
        assert read_status(status_path_for(tmp_path))["serving"]["requests"] == 1


class TestStatusLocation:
    def test_the_default_data_dir_is_in_the_status_resolution_order(self, monkeypatch, tmp_path):
        """Finding 9b: a plain run must be findable by the statusline."""
        for var in ("MEMORY_CONDENSE_STATUS", "MEMORY_CONDENSE_DATA_DIR", "MEMORY_DATA_DIR"):
            monkeypatch.delenv(var, raising=False)

        # No environment at all: the bare default is absolute and is the last
        # candidate -- the same path the installed command passes explicitly.
        bare = default_data_dir()
        assert Path(bare).is_absolute()
        assert status_path_for(bare) in candidate_status_files()

        # The canonical variable every reader already keys off now steers the
        # writer too, so candidate 2 resolves.
        monkeypatch.setenv("MEMORY_CONDENSE_DATA_DIR", str(tmp_path / "chosen"))
        assert Path(default_data_dir()) == tmp_path / "chosen"
        assert status_path_for(default_data_dir()) in candidate_status_files()

        # The proxy's own older variable still works, second in line.
        monkeypatch.setenv("MEMORY_DATA_DIR", str(tmp_path / "legacy"))
        assert Path(default_data_dir()) == tmp_path / "chosen"
        monkeypatch.delenv("MEMORY_CONDENSE_DATA_DIR")
        assert Path(default_data_dir()) == tmp_path / "legacy"


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

    def test_a_failed_request_leaves_no_receipt_behind(self, tmp_path):
        """A receipt describes what happened, so a 502 is not a served request."""
        def broken(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route to upstream")

        app = build_app(
            config=ProxyConfig(), client=httpx.AsyncClient(transport=httpx.MockTransport(broken)),
            data_dir=tmp_path,
        )
        with TestClient(app) as client:
            reply = client.post("/v1/messages", json={"model": "m", "messages": [
                {"role": "user", "content": "hi"}]}, headers={"x-api-key": "k"})
            assert reply.status_code == 502
            doc = client.get("/_control/status").json()
            assert doc["serving"] == empty_status()["serving"]

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
            # Finding 9c: the headline total counts the text that was
            # ingested, not the bytes of the file it came out of, so the same
            # content loaded in bulk and arriving live contributes the same
            # number. The file figure is still reported, under `last_job`,
            # because that is what the percentage and the ETA need.
            assert doc["totals"]["tokens_ingested"] == sum(ingested_tokens(c) for c in captured)
            assert doc["totals"]["tokens_ingested"] == doc["last_job"]["tokens_ingested"]
            assert doc["last_job"]["tokens_done"] > doc["totals"]["tokens_ingested"]
            assert doc["rates"]["window_s"] == 30
            assert [(c.prompt[0].text, c.reply_text) for c in captured] == pairs
            assert read_status(status_path_for(tmp_path))["last_job"]["state"] == "done"

    def test_a_failed_mutation_leaves_the_store_serving_and_the_state_idle(self, tmp_path):
        """Finding 4 at the route level: no wedged store, no wedged status."""
        FakeCondenser.instances.clear()
        store = CondenserHandle(lambda: FakeCondenser(tmp_path), data_dir=tmp_path)
        store.open()
        app = _app(tmp_path, sink=store.capture_sink(), store=store)
        with TestClient(app) as client:
            _chat(client, "before")
            _wait(lambda: app.state.captures.ingested == 1)
            assert client.post("/_control/save", json={"name": "base"}).status_code == 200

            snapshots = app.state.snapshots
            restore = snapshots.restore
            snapshots.restore = lambda name: (_ for _ in ()).throw(
                PermissionError("the file is used by another process")
            )
            reply = client.post("/_control/load", json={"name": "base"})
            assert reply.status_code == 500
            assert reply.json()["error"]["type"] == "swap_failed"
            assert reply.json()["error"]["store_live"] is True
            assert not store.closed

            doc = client.get("/_control/status").json()
            assert doc["state"] == "idle" and doc["error"] is None, "not stuck in error"
            _chat(client, "after the failed load")  # ingestion survived
            _wait(lambda: app.state.captures.ingested == 2)
            assert _rows(tmp_path / "memory.db") == [
                "before", "upstream answer", "after the failed load", "upstream answer",
            ]

            snapshots.restore = restore
            assert client.post("/_control/load", json={"name": "base"}).status_code == 200
            assert _rows(tmp_path / "memory.db") == ["before", "upstream answer"]
        store.close()

    def test_augment_mode_saves_loads_and_resets_but_cannot_bulk_load(self, tmp_path):
        memory = FakeSessions(tmp_path, failed=1)
        memory.admit("ns-a", ["first turn"])
        app = build_app(
            config=ProxyConfig(mode="augment"), memory=memory,
            client=httpx.AsyncClient(transport=httpx.MockTransport(_upstream)), data_dir=tmp_path,
        )
        live = tmp_path / "sessions" / "ns-a" / "chat-events.sqlite"
        with TestClient(app) as client:
            doc = client.get("/_control/status").json()
            assert doc["mode"] == "augment"
            assert doc["health"] == {"status": "degraded", "reasons": ["memory_sessions_failed"]}
            assert client.get("/_memory/health").json()["status"] == "degraded"
            assert doc["memory"]["active_sessions"] == 1
            # "the store" in augment mode is the session tree, sized recursively.
            assert doc["store"]["path"] == str(tmp_path / "sessions")
            assert doc["store"]["snapshot_dir"] == str(tmp_path / "snapshots")
            assert doc["totals"]["store_bytes"] == tree_bytes(tmp_path / "sessions") > 0

            record = client.post("/_control/save", json={"name": "base"}).json()
            assert record["file"] == "base" and record["name"] == "base"
            assert client.post("/_control/save", json={"name": "base"}).status_code == 409
            assert client.post("/_control/save", json={"name": "../x"}).status_code == 400
            doc = client.get("/_control/status").json()
            assert doc["store"]["snapshots"] == ["base"] and doc["store"]["last_snapshot"] == "base"

            # Move the tree on: a new conversation and a new turn in the old one.
            memory.admit("ns-b", ["stray turn"])
            _make_db(live, ["second turn"])
            assert _rows(live) == ["first turn", "second turn"]

            assert client.post("/_control/load", json={}).status_code == 400
            assert client.post("/_control/load", json={"name": "a/b"}).status_code == 400
            assert client.post("/_control/load", json={"name": "missing"}).status_code == 404
            reply = client.post("/_control/load", json={"name": "base"})
            assert reply.status_code == 200, reply.text
            assert reply.json() == {"loaded": "base", "store": str(tmp_path / "sessions"),
                                    "generation": 1}
            assert _rows(live) == ["first turn"], "the snapshot's rows came back"
            assert not (tmp_path / "sessions" / "ns-b").exists()
            assert memory.closed_stores == [["ns-a", "ns-b"]], "every store closed for the swap"
            doc = client.get("/_control/status").json()
            assert doc["state"] == "idle" and doc["error"] is None

            body = client.post("/_control/reset", json={"confirm": True}).json()
            assert body["reset"] is True and body["generation"] == 2
            backup = Path(body["backup"]["path"]) / "sessions" / "ns-a" / "chat-events.sqlite"
            assert _rows(backup) == ["first turn"], "reset backs the tree up first"
            assert not (tmp_path / "sessions").exists()
            assert client.post("/_control/reset", json={}).status_code == 400
            assert client.post("/_control/reset", json={"confirm": "yes"}).status_code == 400
            assert len(client.get("/_control/status").json()["store"]["snapshots"]) == 2

            # Bulk loading rides the observe-mode capture queue, which augment
            # mode never starts, so it stays 501 -- and says why.
            reply = client.post("/_control/load-transcript", json={"session": "x"})
            assert reply.status_code == 501
            assert "observe-mode only" in reply.json()["error"]["message"]
        assert memory.closed

    def test_a_swap_is_visible_as_swapping_and_holds_off_the_next_one(self, tmp_path):
        memory = FakeSessions(tmp_path)
        memory.admit("ns-a", ["first turn"])
        app = build_app(
            config=ProxyConfig(mode="augment"), memory=memory,
            client=httpx.AsyncClient(transport=httpx.MockTransport(_upstream)), data_dir=tmp_path,
        )
        with TestClient(app) as client:
            assert client.post("/_control/save", json={"name": "base"}).status_code == 200
            gate, entered = threading.Event(), threading.Event()
            snapshots = app.state.snapshots
            restore = snapshots.restore

            def slow(name):
                entered.set()
                assert gate.wait(15)
                return restore(name)

            snapshots.restore = slow
            worker = threading.Thread(target=lambda: client.post("/_control/load", json={"name": "base"}))
            worker.start()
            try:
                assert entered.wait(15)
                assert read_status(status_path_for(tmp_path))["state"] == "swapping"
                assert memory.lock.acquire(timeout=0.2) is False, "admissions are blocked"
            finally:
                gate.set()
                worker.join(30)
            assert read_status(status_path_for(tmp_path))["state"] == "idle"
        assert memory.closed


# -- serving receipts --------------------------------------------------------

#: A client prompt with real resent history: what augment mode exists to trim.
_HISTORY = {
    "model": "m",
    "system": "You are terse.",
    "messages": [
        *[
            message
            for i in range(6)
            for message in (
                {"role": "user", "content": f"question {i} " * 20},
                {"role": "assistant", "content": f"answer {i} " * 20},
            )
        ],
        {"role": "user", "content": "the latest question"},
    ],
}
#: The estimator reads text, not framing, so this does not depend on which
#: JSON encoder put the body on the wire.
_HISTORY_TOKENS = prompt_tokens("anthropic", json.dumps(_HISTORY).encode("utf-8"))
_LATEST_ONLY = {"model": "m", "messages": _HISTORY["messages"][-1:]}


def _upstream_with_usage(request: httpx.Request) -> httpx.Response:
    """The upstream's own prompt count, which beats any estimate."""
    return httpx.Response(200, json={
        "content": [{"type": "text", "text": "upstream answer"}],
        "usage": {"input_tokens": 120, "cache_read_input_tokens": 30, "output_tokens": 3},
    })


def _streaming_upstream_with_usage(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=(
            b'event: message_start\ndata: {"type":"message_start","message":{"id":"m1",'
            b'"usage":{"input_tokens":100,"cache_read_input_tokens":50}}}\n\n'
            b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
            b'"delta":{"type":"text_delta","text":"streamed answer"}}\n\n'
            b'event: message_delta\ndata: {"type":"message_delta",'
            b'"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}\n\n'
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        ),
    )


def _post(client: TestClient, payload: dict):
    return client.post("/v1/messages", json=payload, headers={"x-api-key": "k"})


class TestServingReceipts:
    """Finding 3: both modes publish ``serving`` through ``proxy_receipts``.

    The guarantees checked here are the ones the module was written for and
    the wired path used to break: one estimator across both modes, a served
    figure that exists in augment mode, ``rewritten`` that means a rewrite,
    the upstream's own ``usage`` preferred over any estimate, and a receipt
    recorded only once the request has actually been served.
    """

    def test_observe_publishes_the_upstream_count_on_both_sides(self, tmp_path):
        app = build_app(
            config=ProxyConfig(), sink=lambda capture: None,
            client=httpx.AsyncClient(transport=httpx.MockTransport(_upstream_with_usage)),
            data_dir=tmp_path,
        )
        with TestClient(app) as client:
            assert _post(client, _HISTORY).status_code == 200
            doc = client.get("/_control/status").json()
        serving = doc["serving"]
        assert serving["requests"] == 1 and serving["rewritten"] == 0
        assert _HISTORY_TOKENS > 150, "the estimate and the real count differ"
        assert serving["static_tokens_last"] == serving["served_tokens_last"] == 150
        assert serving["saved_tokens_total"] == 0, "forwarding verbatim saves nothing"

    def test_observe_reads_usage_out_of_a_stream_it_already_tees(self, tmp_path):
        captured: list = []
        app = build_app(
            config=ProxyConfig(), sink=captured.append,
            client=httpx.AsyncClient(transport=httpx.MockTransport(_streaming_upstream_with_usage)),
            data_dir=tmp_path,
        )
        with TestClient(app) as client:
            reply = _post(client, dict(_HISTORY, stream=True))
            assert reply.status_code == 200
            assert b"streamed answer" in reply.content, "the stream reaches the client intact"
            _wait(lambda: app.state.captures.ingested == 1)
            serving = client.get("/_control/status").json()["serving"]
        assert serving["requests"] == 1
        assert serving["static_tokens_last"] == serving["served_tokens_last"] == 150
        assert [capture.reply_text for capture in captured] == ["streamed answer"]

    def test_observe_falls_back_to_the_estimate_without_a_usage_report(self, tmp_path):
        app = _app(tmp_path, sink=lambda capture: None)
        with TestClient(app) as client:
            assert _post(client, _HISTORY).status_code == 200
            serving = client.get("/_control/status").json()["serving"]
        assert serving["static_tokens_last"] == serving["served_tokens_last"] == _HISTORY_TOKENS
        assert serving["saved_tokens_total"] == 0

    def _augment(self, tmp_path, memory, upstream=_upstream_with_usage):
        return build_app(
            config=ProxyConfig(mode="augment"), memory=memory, data_dir=tmp_path,
            client=httpx.AsyncClient(transport=httpx.MockTransport(upstream)),
        )

    def test_augment_measures_the_rewrite_it_actually_served(self, tmp_path):
        """The mode whose whole point is serving less than the client sent."""
        served = json.dumps(_LATEST_ONLY).encode("utf-8")
        memory = FakeSessions(tmp_path, serve=lambda body: served)
        app = self._augment(tmp_path, memory)
        with TestClient(app) as client:
            assert _post(client, _HISTORY).status_code == 200
            doc = client.get("/_control/status").json()
        assert memory.sent == [served]
        serving = doc["serving"]
        assert serving["requests"] == 1 and serving["rewritten"] == 1
        # Static is the client's prompt by the shared estimator, not
        # len(body)//4 over tool definitions and JSON framing.
        assert serving["static_tokens_last"] == _HISTORY_TOKENS
        # Served is the upstream's own count of what went up, not None.
        assert serving["served_tokens_last"] == 150
        assert serving["saved_tokens_total"] == _HISTORY_TOKENS - 150 > 0
        # And the segment the statusline renders now appears in augment mode.
        assert f"serving 150 of {_HISTORY_TOKENS}" in one_line(doc)

    def test_augment_estimates_the_served_body_when_the_upstream_is_silent(self, tmp_path):
        served = json.dumps(_LATEST_ONLY).encode("utf-8")
        memory = FakeSessions(tmp_path, serve=lambda body: served)
        app = self._augment(tmp_path, memory, upstream=_upstream)
        with TestClient(app) as client:
            assert _post(client, _HISTORY).status_code == 200
            serving = client.get("/_control/status").json()["serving"]
        assert serving["static_tokens_last"] == _HISTORY_TOKENS
        assert serving["served_tokens_last"] == prompt_tokens("anthropic", served)
        assert serving["saved_tokens_total"] == _HISTORY_TOKENS - serving["served_tokens_last"]

    def test_an_augment_passthrough_is_not_counted_as_a_rewrite(self, tmp_path):
        """The pre-send guard's sanctioned degraded mode sends client bytes."""
        memory = FakeSessions(tmp_path, serve=lambda body: body)
        app = self._augment(tmp_path, memory)
        with TestClient(app) as client:
            assert _post(client, _HISTORY).status_code == 200
            serving = client.get("/_control/status").json()["serving"]
        assert memory.sent, "the client's own bytes went upstream"
        assert serving["requests"] == 1 and serving["rewritten"] == 0
        assert serving["static_tokens_last"] == serving["served_tokens_last"] == 150
        assert serving["saved_tokens_total"] == 0

    def test_a_replay_that_sends_nothing_leaves_the_served_side_unknown(self, tmp_path):
        memory = FakeSessions(tmp_path, serve=lambda body: None)
        app = self._augment(tmp_path, memory)
        with TestClient(app) as client:
            assert _post(client, _HISTORY).status_code == 200
            serving = client.get("/_control/status").json()["serving"]
        assert memory.sent == []
        assert serving["requests"] == 1 and serving["rewritten"] == 0
        assert serving["static_tokens_last"] == _HISTORY_TOKENS
        assert serving["served_tokens_last"] is None, "unknown, never zero"
        assert serving["saved_tokens_total"] == 0

    def test_an_internal_summary_is_not_the_clients_request(self, tmp_path):
        """A summary generation must not become the served receipt."""
        served = json.dumps(_LATEST_ONLY).encode("utf-8")
        memory = FakeSessions(tmp_path, serve=lambda body: served)
        app = self._augment(tmp_path, memory)
        with TestClient(app) as client:
            assert _post(client, _HISTORY).status_code == 200
            serving = client.get("/_control/status").json()["serving"]
        assert memory.summaries, "a summary call did happen"
        assert serving["requests"] == 1, "one receipt, for the visible answer"
        assert serving["static_tokens_last"] == _HISTORY_TOKENS
        assert serving["served_tokens_last"] == 150

    def test_a_failed_augment_request_is_not_a_served_request(self, tmp_path):
        """Finding 3's placement bug: the receipt used to precede the attempt."""
        def refuse(body):
            raise RuntimeError("the memory service fell over")

        memory = FakeSessions(tmp_path, serve=refuse)
        app = self._augment(tmp_path, memory)
        with TestClient(app) as client:
            assert _post(client, _HISTORY).status_code == 503
            doc = client.get("/_control/status").json()
        assert doc["serving"] == empty_status()["serving"]
        assert "serving" not in one_line(doc)


class TestSessionCapacityIsVisible:
    """An operator must see a full session registry before it refuses a request.

    `MemoryProxy.entry` raises 429 when `len(self.entries)` reaches
    `max_sessions`, and it removes no session to make room. Before this,
    `status()` published `active_sessions` but never the limit, and
    `memory_health` only flagged `failed_sessions` -- so health read `ok` right
    up to a user-facing 429, and no surface could warn. Package A of
    `01 - Design/05 - Local Model Handoff and Concurrent Linking.md`.
    """

    def test_health_collects_every_reason_rather_than_the_first(self):
        assert memory_health({}) == ("ok", [])
        assert memory_health({"failed_sessions": 1}) == ("degraded", ["memory_sessions_failed"])
        assert memory_health({"sessions_at_capacity": True}) == (
            "degraded", ["memory_sessions_at_capacity"])
        status, reasons = memory_health({"failed_sessions": 1, "sessions_at_capacity": True})
        assert status == "degraded"
        assert reasons == ["memory_sessions_failed", "memory_sessions_at_capacity"]

    def test_a_conversation_still_open_is_not_a_failure(self):
        """At capacity is degraded, not failed: open conversations keep working."""
        assert memory_health({"sessions_at_capacity": True})[0] == "degraded"

    def test_status_publishes_the_limit_and_the_count_that_gates_admission(self, tmp_path):
        """Built on a real MemoryProxy: `status` grows, and a duck-type rots."""
        from memory_condense.interfaces.proxy_memory import MemoryProxy, SessionEntry

        def session(_namespace, _directory):
            raise AssertionError("this test never admits a real session")

        memory = MemoryProxy(tmp_path, session, inline_memory=False, max_sessions=2)
        assert memory.status()["max_sessions"] == 2
        assert memory.status()["sessions_at_capacity"] is False

        for n in range(2):
            entry = SessionEntry(SimpleNamespace(
                status=lambda: dict(pending_events=0, pending_feedback=0,
                                    last_error=None, preparation_jobs=0)))
            # The registry `entry` gates on, and the mirror `status` reads.
            memory.entries[f"ns{n}"] = entry
            memory._live[f"ns{n}"] = entry
            snapshot = memory.status()
            assert snapshot["admitted_sessions"] == n + 1
            assert snapshot["sessions_at_capacity"] is (n + 1 >= 2)

        # The signal must follow the count that `entry` enforces, not the count
        # of sessions whose status happened to be readable.
        assert memory_health(memory.status())[1] == ["memory_sessions_at_capacity"]
