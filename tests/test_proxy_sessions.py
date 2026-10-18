import json
import os
import time
from types import SimpleNamespace

from memory_condense.interfaces import proxy_sessions as ps


def _anthropic(first: str, user_id: str | None = None) -> bytes:
    body = {"model": "m", "messages": [{"role": "user", "content": first}, {"role": "assistant", "content": "ok"}]}
    if user_id:
        body["metadata"] = {"user_id": user_id}
    return json.dumps(body).encode()


def _responses(first: str) -> bytes:
    return json.dumps({"model": "m", "input": [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": first}]},
    ]}).encode()


def test_explicit_header_wins():
    cid, src = ps.resolve_conversation_id({"X-Memory-Conversation-Id": " thread-1 "}, _anthropic("hi"))
    assert (cid, src) == ("thread-1", ps.SOURCE_HEADER)


def test_codex_session_id_header_is_the_client_native_id():
    cid, src = ps.resolve_conversation_id({"session_id": "01a0cc59-280e-71d2-8335-f3a6ccce0aa3"}, _responses("fix it"))
    assert cid == "session:01a0cc59-280e-71d2-8335-f3a6ccce0aa3" and src == ps.SOURCE_CLIENT


def test_claude_metadata_user_id_session_is_parsed_when_present():
    uid = "user_ab12_account_7f0c1b2e-1111-2222-3333-444444444444_session_8F7F7561-E2AF-4BAB-8F13-8FAB1E2D71BB"
    cid, src = ps.resolve_conversation_id({}, _anthropic("hi", uid))
    assert cid == "session:8f7f7561-e2af-4bab-8f13-8fab1e2d71bb" and src == ps.SOURCE_CLIENT


def test_digest_fallback_is_stable_and_turn_local():
    body = _anthropic("hello")
    a = ps.resolve_conversation_id({}, body)
    b = ps.resolve_conversation_id({}, body)
    c = ps.resolve_conversation_id({}, _anthropic("other"))
    assert a == b and a[1] == ps.SOURCE_DIGEST and a[0].startswith("proxy:") and a[0] != c[0]


def test_head_text_across_formats_and_garbage():
    assert ps.head_text(json.loads(_anthropic("  Fix the  login bug "))) == "Fix the  login bug"
    assert ps.head_text(json.loads(_responses("Fix it"))) == "Fix it"
    assert ps.head_text({"messages": "nope"}) == "" and ps.head_text(None) == ""
    assert ps.resolve_conversation_id({}, b"not json")[1] == ps.SOURCE_DIGEST


def _write_transcript(path, first_user: str, kind: str = "claude") -> None:
    if kind == "claude":
        recs = [
            {"type": "mode", "mode": "normal"},
            {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": first_user}]}},
            {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}},
        ]
    else:
        recs = [
            {"type": "session_meta", "payload": {}},
            {"type": "response_item", "payload": {"type": "message", "role": "user",
                                                  "content": [{"type": "input_text", "text": first_user}]}},
        ]
    path.write_bytes(b"".join(json.dumps(r).encode() + b"\n" for r in recs))


def test_first_user_text_of_transcript_handles_both_clients(tmp_path):
    c = tmp_path / "c.jsonl"
    x = tmp_path / "x.jsonl"
    _write_transcript(c, "Claude first")
    _write_transcript(x, "Codex first", kind="codex")
    assert ps.first_user_text_of_transcript(str(c)) == "Claude first"
    assert ps.first_user_text_of_transcript(str(x)) == "Codex first"


def _ref(path, session_id, liveness, mtime):
    """A SessionRef stand-in whose on-disk mtime matches what the scan saw.

    ``locate`` re-stats a candidate rather than trusting the scan's frozen
    mtime, so the file has to carry the age the ref claims.
    """
    os.utime(path, (mtime, mtime))
    return SimpleNamespace(key=str(path).lower(), path=str(path), session_id=session_id, liveness=liveness, mtime=mtime)


def test_locator_attributes_request_to_live_transcript(tmp_path, monkeypatch):
    t = tmp_path / "s1.jsonl"
    _write_transcript(t, "Build the widget")
    refs = [_ref(t, "11111111-aaaa-bbbb-cccc-222222222222", "process", time.time())]
    monkeypatch.setattr("memory_condense.live_sessions.live_sessions", lambda **kw: refs)
    loc = ps.TranscriptLocator()
    loc.refresh(wait=True)
    cid, src = ps.resolve_conversation_id({}, _anthropic("Build   the widget"), locator=loc)
    assert cid == "transcript:11111111-aaaa-bbbb-cccc-222222222222" and src == ps.SOURCE_TRANSCRIPT


def test_locator_tiebreak_prefers_owned_then_clearly_newest_else_none(tmp_path, monkeypatch):
    a, b, c = (tmp_path / f"{n}.jsonl" for n in "abc")
    for p in (a, b, c):
        _write_transcript(p, "same prompt")
    now = time.time()
    # owned beats recency-only
    refs = [_ref(a, "A", "recent", now), _ref(b, "B", "handle", now - 100)]
    monkeypatch.setattr("memory_condense.live_sessions.live_sessions", lambda **kw: refs)
    loc = ps.TranscriptLocator()
    loc.refresh(wait=True)
    assert loc.locate("same prompt").session_id == "B"
    # two owned, clearly different ages -> newest; same age -> ambiguous -> None
    refs2 = [_ref(a, "A", "process", now), _ref(b, "B", "process", now - 60)]
    refs3 = [_ref(a, "A", "process", now), _ref(c, "C", "process", now - 1)]
    for expected, rs in ((("A"), refs2), ((None), refs3)):
        monkeypatch.setattr("memory_condense.live_sessions.live_sessions", lambda rs=rs, **kw: rs)
        loc = ps.TranscriptLocator()
        loc.refresh(wait=True)
        got = loc.locate("same prompt")
        assert (got.session_id if got else None) == expected


def test_locator_never_blocks_or_raises_when_scan_fails(monkeypatch):
    def boom(**kw):
        raise RuntimeError("psutil exploded")

    monkeypatch.setattr("memory_condense.live_sessions.live_sessions", boom)
    loc = ps.TranscriptLocator()
    loc.refresh(wait=True)
    assert loc.locate("anything") is None
    cid, src = ps.resolve_conversation_id({}, _anthropic("anything"), locator=loc)
    assert src == ps.SOURCE_DIGEST


def test_locator_evicts_sessions_that_have_exited(tmp_path, monkeypatch):
    """A finished transcript stops being a candidate.

    Otherwise a new conversation opening with the same line as a finished one
    is captured under the dead session's id, merging the two.
    """
    old, new = tmp_path / "old.jsonl", tmp_path / "new.jsonl"
    for p in (old, new):
        _write_transcript(p, "same opening line")
    now = time.time()
    live = [_ref(old, "OLD", "handle", now)]
    monkeypatch.setattr("memory_condense.live_sessions.live_sessions", lambda **kw: live)
    loc = ps.TranscriptLocator()
    loc.refresh(wait=True)
    assert loc.locate("same opening line").session_id == "OLD"
    live[:] = [_ref(new, "NEW", "handle", now)]  # the old session exited
    loc._built_at = 0.0  # ttl elapsed -> the proxy's background scan runs again
    loc.refresh(wait=True)
    assert loc.locate("same opening line").session_id == "NEW"
    live[:] = []  # nothing live at all -> no attribution, caller uses the digest
    loc._built_at = 0.0
    loc.refresh(wait=True)
    assert loc.locate("same opening line") is None


def test_stale_handle_liveness_does_not_outrank_a_live_session(tmp_path, monkeypatch):
    """Past max_age the scan's liveness tier is no longer evidence."""
    dead, alive = tmp_path / "dead.jsonl", tmp_path / "alive.jsonl"
    for p in (dead, alive):
        _write_transcript(p, "shared opening")
    now = time.time()
    # At scan time both were live and the one that has since exited held a handle,
    # which would beat the other's recency-only tier forever if it were cached.
    refs = [_ref(dead, "DEAD", "handle", now), _ref(alive, "ALIVE", "recent", now)]
    monkeypatch.setattr("memory_condense.live_sessions.live_sessions", lambda **kw: refs)
    loc = ps.TranscriptLocator(max_age=0.0)  # the index is immediately past max_age
    loc.refresh(wait=True)
    os.utime(dead, (now - 600, now - 600))  # dead went cold; alive keeps being appended
    assert loc.locate("shared opening").session_id == "ALIVE"
    os.utime(alive, (now - 600, now - 600))  # both cold -> nothing is attributable
    assert loc.locate("shared opening") is None


def test_live_session_keeps_one_conversation_id_across_turns(tmp_path, monkeypatch):
    """Expiry must not make a still-live session flap between id and digest."""
    t = tmp_path / "live.jsonl"
    _write_transcript(t, "keep me together")
    sid = "77777777-aaaa-bbbb-cccc-888888888888"
    refs = [_ref(t, sid, "handle", time.time())]
    monkeypatch.setattr("memory_condense.live_sessions.live_sessions", lambda **kw: refs)
    loc = ps.TranscriptLocator()
    loc.refresh(wait=True)
    ids = set()
    for _turn in range(4):
        with open(t, "ab") as fh:  # the client appends this turn to its transcript
            fh.write(b'{"type": "user", "message": {"role": "user", "content": "more"}}\n')
        refs[:] = [_ref(t, sid, "handle", time.time())]
        loc._built_at = 0.0
        loc.refresh(wait=True)  # the rescan a long-lived proxy does between turns
        cid, src = ps.resolve_conversation_id({}, _anthropic("keep me together"), locator=loc)
        assert src == ps.SOURCE_TRANSCRIPT
        ids.add(cid)
    ids.add(ps.resolve_conversation_id({}, _anthropic("keep me together"), locator=loc)[0])
    ids.add(ps.resolve_conversation_id({}, _anthropic("keep me together"), locator=loc)[0])
    assert ids == {f"transcript:{sid}"}


def test_rescan_prunes_its_caches_and_reads_each_transcript_once(tmp_path, monkeypatch):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write_transcript(a, "alpha")
    _write_transcript(b, "beta")
    reads: list[str] = []
    real = ps.first_user_text_of_transcript
    monkeypatch.setattr(ps, "first_user_text_of_transcript",
                        lambda path, **kw: (reads.append(path), real(path, **kw))[1])
    live = [_ref(a, "A", "handle", time.time())]
    monkeypatch.setattr("memory_condense.live_sessions.live_sessions", lambda **kw: live)
    loc = ps.TranscriptLocator()
    loc.refresh(wait=True)
    live[:] = [_ref(b, "B", "handle", time.time())]
    loc._built_at = 0.0
    loc.refresh(wait=True)
    assert loc.locate("alpha") is None and loc.locate("beta").session_id == "B"
    assert set(loc._seen) == {str(b).lower()}  # bounded by the live set, not by history
    assert reads == [str(a), str(b)]  # the first-message read is cached across scans


def test_client_native_id_folds_a_bare_uuid_header_to_one_id():
    uid = "user_ab12_session_8F7F7561-E2AF-4BAB-8F13-8FAB1E2D71BB"
    from_header = ps.client_native_id({"session_id": "8F7F7561-E2AF-4BAB-8F13-8FAB1E2D71BB"}, None)
    from_metadata = ps.client_native_id({}, {"metadata": {"user_id": uid}})
    assert from_header == from_metadata == "session:8f7f7561-e2af-4bab-8f13-8fab1e2d71bb"
    # an opaque id may be case-sensitive, so it crosses verbatim
    assert ps.client_native_id({"session_id": "AbC-123"}, None) == "session:AbC-123"
