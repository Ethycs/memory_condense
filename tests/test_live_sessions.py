import json
import os
import time

import pytest

from memory_condense.live_sessions import discovery, handles
from memory_condense.live_sessions.discovery import SessionKind, discover_sessions
from memory_condense.live_sessions.handles import (
    AgentProcess,
    HandleSnapshot,
    HandleTracker,
    OpenHandle,
)
from memory_condense.live_sessions.mmap_view import MappedSession
from memory_condense.live_sessions.virtual_fs import Mirror, VirtualSession


def _append(path, data: bytes) -> None:
    with open(path, "ab") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


# -- mmap_view -----------------------------------------------------------------

def test_mapped_session_follows_growth_and_hides_torn_line(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_bytes(b"")
    with MappedSession(path) as m:
        assert m.size == 0 and list(m.records()) == []

        _append(path, b'{"n":1}\n{"n":2}')  # second record has no newline yet
        recs = list(m.records())
        assert [r.json()["n"] for r in recs] == [1]
        resume = m.last_offset()
        assert resume == len(b'{"n":1}\n')

        _append(path, b"\n")
        recs = list(m.records(resume))
        assert [r.json()["n"] for r in recs] == [2]
        assert m.last_offset() == m.size


def test_mapped_session_follow_stops_and_resumes_offsets(tmp_path):
    path = tmp_path / "s.jsonl"
    _append(path, b'{"n":1}\n')
    seen = []
    with MappedSession(path) as m:
        for rec in m.follow(0, poll=0.01, stop=lambda: len(seen) >= 2):
            seen.append(rec.json()["n"])
            if len(seen) == 1:
                _append(path, b'{"n":2}\n')  # lands after the current map; follow remaps
        assert m.last_offset() == m.size
    assert seen == [1, 2]


def test_last_offset_is_valid_when_consumer_stops_early(tmp_path):
    path = tmp_path / "s.jsonl"
    _append(path, b'{"n":1}\n{"n":2}\n{"n":3}\n')
    with MappedSession(path) as m:
        it = m.records()
        assert next(it).json()["n"] == 1
        resume = m.last_offset()  # consumer breaks out here
        assert [r.json()["n"] for r in m.records(resume)] == [2, 3]


# -- handles / tracker ------------------------------------------------------------

def test_tracker_diffs_snapshots_into_events(monkeypatch):
    a = OpenHandle(1, "claude.exe", handles.normalize("a.jsonl"))
    b = OpenHandle(2, "node.exe", handles.normalize("b.jsonl"))
    frames = iter([
        HandleSnapshot(at=1.0, handles=frozenset({a})),
        HandleSnapshot(at=2.0, handles=frozenset({a, b})),
        HandleSnapshot(at=3.0, handles=frozenset({b})),
    ])
    monkeypatch.setattr(handles, "snapshot", lambda **kw: next(frames))
    t = HandleTracker()
    assert [(e.kind, e.handle.pid) for e in t.poll()] == [("opened", 1)]
    assert [(e.kind, e.handle.pid) for e in t.poll()] == [("opened", 2)]
    assert [(e.kind, e.handle.pid) for e in t.poll()] == [("closed", 1)]
    assert t.live_paths() == {b.path}
    assert t.owners("b.jsonl") == {2}


# -- discovery -------------------------------------------------------------------

@pytest.fixture
def fake_roots(tmp_path, monkeypatch):
    claude = tmp_path / "claude" / "projects" / "F--work-proj-x"  # encode(r"F:\work\proj_x")
    codex = tmp_path / "codex" / "sessions"
    claude.mkdir(parents=True)
    codex.mkdir(parents=True)
    monkeypatch.setattr(
        discovery,
        "SESSION_ROOTS",
        {
            SessionKind.CLAUDE: (str(claude.parent),),
            SessionKind.CODEX: (str(codex),),
        },
    )
    return claude, codex


def test_discovery_liveness_from_handles_then_recency(fake_roots):
    claude, codex = fake_roots
    live = claude / "11111111-aaaa-bbbb-cccc-222222222222.jsonl"
    recent = codex / "rollout-2026-09-22T10-00-00-33333333-aaaa-bbbb-cccc-444444444444.jsonl"
    stale = claude / "55555555-aaaa-bbbb-cccc-666666666666.jsonl"
    for p in (live, recent, stale):
        p.write_bytes(b'{"type":"x"}\n')
    old = time.time() - 3600
    os.utime(stale, (old, old))

    snap = HandleSnapshot(
        at=time.time(),
        handles=frozenset({OpenHandle(4242, "claude.exe", handles.normalize(str(live)))}),
    )
    refs = {r.session_id: r for r in discover_sessions(snap=snap, live_window=60)}
    assert refs["11111111-aaaa-bbbb-cccc-222222222222"].liveness == "handle"
    assert refs["11111111-aaaa-bbbb-cccc-222222222222"].pids == (4242,)
    assert refs["11111111-aaaa-bbbb-cccc-222222222222"].project == "F--work-proj-x"
    assert refs["33333333-aaaa-bbbb-cccc-444444444444"].liveness == "recent"
    assert refs["33333333-aaaa-bbbb-cccc-444444444444"].kind is SessionKind.CODEX
    assert refs["55555555-aaaa-bbbb-cccc-666666666666"].liveness == "idle"


def test_discovery_attributes_claude_sessions_by_process_cwd(fake_roots):
    """Claude holds no handle; a claude.exe launched in the project cwd owns the fresh transcript."""
    claude, _ = fake_roots
    fresh = claude / "aaaaaaaa-aaaa-bbbb-cccc-222222222222.jsonl"
    stale = claude / "bbbbbbbb-aaaa-bbbb-cccc-222222222222.jsonl"
    for p in (fresh, stale):
        p.write_bytes(b'{"type":"x"}\n')
    old = time.time() - 3600
    os.utime(stale, (old, old))
    snap = HandleSnapshot(
        at=time.time(),
        handles=frozenset(),
        processes=(
            AgentProcess(pid=555, name="claude.exe", cwd=r"f:\work\proj_x"),   # case differs
            AgentProcess(pid=556, name="claude.exe", cwd=r"F:\elsewhere"),
            AgentProcess(pid=557, name="node.exe", cwd=None),
        ),
    )
    refs = {r.session_id: r for r in discover_sessions(snap=snap, live_window=60)}
    assert refs["aaaaaaaa-aaaa-bbbb-cccc-222222222222"].liveness == "process"
    assert refs["aaaaaaaa-aaaa-bbbb-cccc-222222222222"].pids == (555,)
    assert refs["bbbbbbbb-aaaa-bbbb-cccc-222222222222"].liveness == "idle"  # fresh required


# -- virtual_fs -------------------------------------------------------------------

def test_virtual_session_renders_through_transform(fake_roots):
    claude, _ = fake_roots
    path = claude / "77777777-aaaa-bbbb-cccc-888888888888.jsonl"
    path.write_bytes(b'{"type":"user","text":"hi"}\n{"type":"progress"}\n')
    ref = discover_sessions(use_handles=False)[0]
    drop_progress = lambda recs: [r for r in recs if r["type"] != "progress"]
    with VirtualSession(ref, drop_progress) as vs:
        out = [json.loads(l) for l in vs.render().splitlines()]
    assert out == [{"type": "user", "text": "hi"}]


def test_passthrough_render_is_byte_exact_for_complete_records(fake_roots):
    claude, _ = fake_roots
    path = claude / "cccccccc-aaaa-bbbb-cccc-888888888888.jsonl"
    body = '{"type":"user","text":"café  spaced","n":1.50}\n{"k": [1, 2]}\n'.encode("utf-8")
    path.write_bytes(body + b'{"torn":')  # trailing torn record must be excluded
    ref = discover_sessions(use_handles=False)[0]
    with VirtualSession(ref) as vs:
        assert vs.render() == body


def test_mirror_serves_handle_backed_sessions_and_retires_closed(fake_roots, tmp_path, monkeypatch):
    claude, _ = fake_roots
    path = claude / "99999999-aaaa-bbbb-cccc-000000000000.jsonl"
    path.write_bytes(b'{"type":"user"}\n')
    key = handles.normalize(str(path))
    frames = iter([
        HandleSnapshot(at=1.0, handles=frozenset({OpenHandle(7, "claude.exe", key)})),
        HandleSnapshot(at=2.0, handles=frozenset()),
    ])
    monkeypatch.setattr(handles, "snapshot", lambda **kw: next(frames))

    mirror = Mirror(tmp_path / "mirror")
    first = mirror.sync()
    served = mirror.root / "claude" / "F--work-proj-x" / "99999999-aaaa-bbbb-cccc-000000000000.jsonl"
    assert first["written"] == [str(served)] and served.read_bytes() == b'{"type":"user"}\n'

    second = mirror.sync()  # handle closed -> retired, not served on recency
    assert second["written"] == [] and second["retired"] == [str(served)]
    assert not served.exists()
