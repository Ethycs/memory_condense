"""Clients without x-memory-conversation-id (Claude Code, Codex) are served and threaded."""

import json

import httpx
import pytest
from starlette.testclient import TestClient

from memory_condense.interfaces.proxy_memory import WireResponse
from memory_condense.interfaces.proxy_server import ProxyConfig, build_app


def _anthropic_body(text: str) -> bytes:
    return json.dumps({"model": "claude-x", "max_tokens": 8,
                       "messages": [{"role": "user", "content": text}]}).encode()


def _responses_body(text: str) -> bytes:
    return json.dumps({"model": "gpt-x", "input": [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}]}).encode()


def _upstream(seen: list[httpx.Request]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/responses"):
            return httpx.Response(200, json={"id": "r1", "output": [
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
                "usage": {"input_tokens": 3, "output_tokens": 1}})
        return httpx.Response(200, json={"id": "m1", "type": "message", "role": "assistant",
                                         "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                                         "usage": {"input_tokens": 3, "output_tokens": 1}})
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# -- observe: turns thread by the client's own id ------------------------------------------

def test_observe_threads_codex_turns_by_session_id_header():
    captured = []
    seen: list[httpx.Request] = []
    app = build_app(config=ProxyConfig(mode="observe", capture_compact_prompts=False),
                    sink=lambda c: captured.append(c) or True, client=_upstream(seen))
    with TestClient(app) as client:
        for text in ("first", "second"):
            r = client.post("/v1/responses", content=_responses_body(text),
                            headers={"content-type": "application/json",
                                     "session_id": "01a0cc59-280e-71d2-8335-f3a6ccce0aa3"})
            assert r.status_code == 200
    assert len(captured) == 2
    assert {c.conversation_id for c in captured} == {"session:01a0cc59-280e-71d2-8335-f3a6ccce0aa3"}
    # the client's id never leaks to the upstream as a memory header
    assert all("x-memory-conversation-id" not in req.headers for req in seen)


def test_observe_without_any_id_stays_turn_local():
    captured = []
    app = build_app(config=ProxyConfig(mode="observe", capture_compact_prompts=False),
                    sink=lambda c: captured.append(c) or True, client=_upstream([]))
    with TestClient(app) as client:
        for text in ("first", "second"):
            client.post("/v1/messages", content=_anthropic_body(text), headers={"content-type": "application/json"})
    ids = [c.conversation_id for c in captured]
    assert len(ids) == 2 and ids[0] != ids[1] and all(i.startswith("proxy:") for i in ids)


# -- augment: no 400 without the header -----------------------------------------------------------

class _FakeMemory:
    """Records what the memory service was handed; answers like the upstream."""

    inline_memory = False

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def exchange(self, body, provider, headers, send, summary_send=None, fmt=None):
        self.calls.append({"headers": dict(headers), "provider": provider, "fmt": fmt})
        reply = send(body)
        return WireResponse(reply.status, reply.headers, reply.body)

    def status(self):
        return {"active_sessions": 1, "pending_events": 0, "pending_feedback": 0, "failed_sessions": 0}

    def close(self):
        pass


@pytest.mark.parametrize("path,body,headers,expected", [
    ("/v1/messages", _anthropic_body("hi"), {}, None),  # digest fallback
    ("/v1/responses", _responses_body("hi"), {"session_id": "abc-123"}, "session:abc-123"),
    ("/v1/messages", _anthropic_body("hi"), {"x-memory-conversation-id": "explicit"}, "explicit"),
])
def test_augment_serves_headerless_clients_with_a_resolved_id(path, body, headers, expected):
    memory = _FakeMemory()
    seen: list[httpx.Request] = []
    app = build_app(config=ProxyConfig(mode="augment"), memory=memory, client=_upstream(seen))
    with TestClient(app) as client:
        r = client.post(path, content=body, headers={"content-type": "application/json", **headers})
    assert r.status_code == 200, r.text
    handed = {k.lower(): v for k, v in memory.calls[-1]["headers"].items()}
    got = handed.get("x-memory-conversation-id")
    if expected is None:
        assert got and got.startswith("proxy:")
    else:
        assert got == expected
    assert all("x-memory-conversation-id" not in req.headers for req in seen)


def test_chatgpt_codex_backend_path_drops_v1():
    from memory_condense.interfaces.proxy_server import build_app as _b  # noqa: F401
    seen: list[httpx.Request] = []
    config = ProxyConfig(mode="observe", capture_compact_prompts=False,
                         upstreams={"anthropic": "https://api.anthropic.com",
                                    "openai": "https://chatgpt.com/backend-api/codex"})
    app = build_app(config=config, sink=lambda c: True, client=_upstream(seen))
    with TestClient(app) as client:
        client.post("/v1/responses", content=_responses_body("hi"), headers={"content-type": "application/json"})
    assert str(seen[-1].url) == "https://chatgpt.com/backend-api/codex/responses"
