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
