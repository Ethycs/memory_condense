import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from memory_condense.context_proxy import units as U
from memory_condense.context_proxy.policy import PassthroughPolicy, RecencyWindowPolicy, apply
from memory_condense.context_proxy.server import ProxyConfig, Upstream, make_server
from memory_condense.context_proxy.wire import rewrite


# -- fixtures: conversations -----------------------------------------------------------

def _tool_turn(i: int, big: int = 0) -> list[dict]:
    return [
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hm", "signature": f"sig{i}"},
            {"type": "tool_use", "id": f"tu{i}", "name": "Read", "input": {"path": f"f{i}.py"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"tu{i}", "content": "x" * big},
        ]},
    ]


def anthropic_messages(n_tools: int = 6, big: int = 2000) -> list[dict]:
    msgs = [{"role": "user", "content": "Fix the login bug"}]
    for i in range(n_tools):
        msgs += _tool_turn(i, big)
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": f"step {i} done"}]})
        msgs.append({"role": "user", "content": f"continue {i}"})
    msgs.append({"role": "user", "content": "now what?"})
    return msgs


def responses_input(n_tools: int = 6, big: int = 2000) -> list[dict]:
    items = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "<environment_context>cwd</environment_context>"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Fix the login bug"}]},
    ]
    for i in range(n_tools):
        items += [
            {"type": "reasoning", "id": f"rs{i}", "summary": [], "encrypted_content": "..."},
            {"type": "function_call", "call_id": f"c{i}", "name": "shell", "arguments": "{}"},
            {"type": "function_call_output", "call_id": f"c{i}", "output": "y" * big},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": f"step {i}"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": f"continue {i}"}]},
        ]
    return items


# -- units ---------------------------------------------------------------------------------

def test_anthropic_grouping_keeps_tool_pairs_and_thinking_together():
    units = U.group_anthropic(anthropic_messages(2))
    kinds = [u.kind for u in units]
    assert kinds == ["user", "exchange", "assistant", "user", "exchange", "assistant", "user", "user"]
    ex = units[1]
    assert ex.items[0]["role"] == "assistant" and ex.items[1]["role"] == "user"
    assert ex.tool_names == ["Read"]
    assert units[0].user_text == "Fix the login bug"


def test_responses_grouping_attaches_reasoning_and_outputs():
    units = U.group_responses(responses_input(2))
    assert [u.kind for u in units][:5] == ["user", "user", "exchange", "assistant", "user"]
    ex = units[2]
    assert [i["type"] for i in ex.items] == ["reasoning", "function_call", "function_call_output"]
    assert U.head_length(units) == 2


# -- policy + rewrite ------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", ["anthropic", "responses"])
def test_recency_policy_trims_middle_and_preserves_pairing(fmt):
    conv = anthropic_messages() if fmt == "anthropic" else responses_input()
    path = "/v1/messages" if fmt == "anthropic" else "/responses"
    key = "messages" if fmt == "anthropic" else "input"
    body = {"model": "m", key: conv}
    new, report = rewrite(path, body, RecencyWindowPolicy(budget_tokens=3000))
    assert report.rewritten and new is not None
    out = new[key]
    assert U.check(fmt, out) == []
    assert report.tokens_after < report.tokens_before
    # head and tail survive; the memory block sits right after the head
    head = U.head_length(U.group(fmt, conv))
    assert out[:head] == conv[:head]
    assert out[-1] == conv[-1]
    memory = out[head]
    text = memory["content"][0]["text"]
    assert "elided" in text and "User asked: continue 0" in text and "Read" in text or "shell" in text


def test_rewrite_is_noop_within_budget_and_for_server_side_state():
    body = {"model": "m", "messages": anthropic_messages(1, big=10)}
    new, report = rewrite("/v1/messages", body, RecencyWindowPolicy(budget_tokens=1_000_000))
    assert new is None and not report.rewritten and report.reason == "within budget"
    new, report = rewrite("/responses", {"input": responses_input(), "previous_response_id": "r1"},
                          RecencyWindowPolicy(budget_tokens=10))
    assert new is None and "server-side" in report.reason
    new, report = rewrite("/v1/models", {"messages": []}, PassthroughPolicy())
    assert new is None and report.format is None


def test_memory_system_seam_receives_request_and_session_handle():
    seen = {}

    def condense(req):
        seen["req"] = req
        return f"MEMORY BLOCK for {req.fingerprint} ({len(req.dropped)} dropped)"

    handle = object()
    units = U.group_anthropic(anthropic_messages())
    plan = RecencyWindowPolicy(budget_tokens=3000, condense=condense).plan(units, "anthropic", handle)
    req = seen["req"]
    assert req.session is handle and req.fmt == "anthropic"
    assert req.head_text == "Fix the login bug" and len(req.fingerprint) == 16
    assert plan.memory_text.startswith("MEMORY BLOCK for " + req.fingerprint)
    assert len(req.dropped) + len(req.kept) == len(units)


def test_apply_never_splits_a_unit():
    units = U.group_anthropic(anthropic_messages(3))
    plan = RecencyWindowPolicy(budget_tokens=2500).plan(units, "anthropic")
    out = apply("anthropic", units, plan)
    ids_use = [b["id"] for m in out if m["role"] == "assistant" for b in m["content"] if b.get("type") == "tool_use"]
    ids_res = [b["tool_use_id"] for m in out if m["role"] == "user" and isinstance(m["content"], list)
               for b in m["content"] if b.get("type") == "tool_result"]
    assert sorted(ids_use) == sorted(ids_res)


# -- end-to-end through a fake streaming upstream --------------------------------------------

class _FakeUpstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    received: list[dict] = []

    def log_message(self, *a):  # noqa: D401
        pass

    def do_GET(self):
        payload = b'{"data":[{"id":"m1"}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers["content-length"]))
        _FakeUpstream.received.append({"path": self.path, "headers": dict(self.headers), "body": json.loads(raw)})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("anthropic-ratelimit-unified-status", "allowed")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for i in range(3):
            chunk = f"event: delta\ndata: {{\"i\":{i}}}\n\n".encode()
            self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
            self.wfile.flush()
            time.sleep(0.05)
        self.wfile.write(b"0\r\n\r\n")


@pytest.fixture
def stack(tmp_path):
    up = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    cfg = ProxyConfig(
        upstream=Upstream.parse(f"http://127.0.0.1:{up.server_port}/base"),
        policy=RecencyWindowPolicy(budget_tokens=3000),
        ledger_path=tmp_path / "ledger.jsonl",
        quiet=True,
    )
    px = make_server("127.0.0.1", 0, cfg)
    threading.Thread(target=px.serve_forever, daemon=True).start()
    _FakeUpstream.received.clear()
    yield px.server_port, tmp_path / "ledger.jsonl"
    px.shutdown()
    up.shutdown()


def test_proxy_rewrites_forwards_headers_and_streams(stack):
    port, ledger = stack
    body = json.dumps({"model": "claude-x", "stream": True, "messages": anthropic_messages()}).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", "/v1/messages", body=body, headers={
        "Content-Type": "application/json",
        "anthropic-beta": "oauth-2025-04-20,context-1m-2025-08-07",
        "anthropic-version": "2023-06-01",
        "Authorization": "Bearer sk-ant-oat01-secret",
    })
    resp = conn.getresponse()
    assert resp.status == 200
    assert resp.getheader("content-type") == "text/event-stream"
    assert resp.getheader("anthropic-ratelimit-unified-status") == "allowed"
    first = resp.read1(4096)  # arrives before the upstream finishes
    assert first.startswith(b"event: delta")
    rest = resp.read()
    assert (first + rest).count(b"event: delta") == 3

    up = _FakeUpstream.received[-1]
    assert up["path"] == "/base/v1/messages"
    assert up["headers"]["anthropic-beta"] == "oauth-2025-04-20,context-1m-2025-08-07"
    assert up["headers"]["Authorization"] == "Bearer sk-ant-oat01-secret"
    assert "Host" in up["headers"] and up["headers"]["Host"].startswith("127.0.0.1")
    sent = up["body"]["messages"]
    assert U.check_anthropic(sent) == [] and len(sent) < len(json.loads(body)["messages"])
    assert up["body"]["stream"] is True and up["body"]["model"] == "claude-x"

    deadline = time.time() + 5  # the ledger is written after the stream closes
    while not ledger.exists() and time.time() < deadline:
        time.sleep(0.02)
    entry = json.loads(ledger.read_text().splitlines()[-1])
    assert entry["report"]["rewritten"] and entry["status"] == 200

    conn.request("GET", "/v1/models")  # keep-alive + passthrough of a non-inference route
    r2 = conn.getresponse()
    assert r2.status == 200 and json.loads(r2.read())["data"][0]["id"] == "m1"
