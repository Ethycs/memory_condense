"""A local model-API proxy that serves the model a condensed conversation.

The client (Claude Code via ``ANTHROPIC_BASE_URL``, Codex via a
``model_providers`` entry) keeps and displays its own full transcript. This
proxy rewrites only the conversation array of inference requests, forwards
every header the upstream needs verbatim, and streams the response back
byte-for-byte -- so from the client's side nothing changed except the token
usage the API reports, which is what keeps native auto-compaction from firing.

Stdlib only. ``HTTPResponse.read1`` is used for the upstream stream so SSE
chunks are relayed as they arrive rather than after a buffer fills.
"""

from __future__ import annotations

import gzip
import http.client
import json
import sys
import threading
import time
import zlib
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .policy import ContextPolicy, PassthroughPolicy
from .wire import Report, rewrite

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "host", "content-length",
}


@dataclass
class Upstream:
    scheme: str
    host: str
    port: int
    base_path: str

    @classmethod
    def parse(cls, url: str) -> "Upstream":
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"upstream must be an http(s) URL, got {url!r}")
        port = parts.port or (443 if parts.scheme == "https" else 80)
        return cls(parts.scheme, parts.hostname, port, parts.path.rstrip("/"))

    def connect(self, timeout: float) -> http.client.HTTPConnection:
        if self.scheme == "https":
            return http.client.HTTPSConnection(self.host, self.port, timeout=timeout)
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout)


@dataclass
class ProxyConfig:
    upstream: Upstream
    policy: ContextPolicy = field(default_factory=PassthroughPolicy)
    ledger_path: Path | None = None
    upstream_timeout: float = 600.0
    quiet: bool = False
    locator: object | None = None  # bridge.SessionLocator, joins requests to on-disk originals


class _Ledger:
    def __init__(self, path: Path | None, quiet: bool) -> None:
        self.path = path
        self.quiet = quiet
        self._lock = threading.Lock()

    def record(self, entry: dict) -> None:
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            if self.path is not None:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            if not self.quiet:
                r = entry.get("report") or {}
                mark = "REWRITTEN" if r.get("rewritten") else "passthrough"
                units = f"{r.get('units_kept', '-')}/{r.get('units_total', '-')}"
                print(
                    f"{entry['method']} {entry['path']} -> {entry['status']} "
                    f"{entry['elapsed_ms']}ms  {mark}  units {units}  "
                    f"tokens ~{r.get('tokens_before', '-')}->~{r.get('tokens_after', '-')}  {r.get('reason', '')}",
                    file=sys.stderr,
                    flush=True,
                )


def _decode_body(raw: bytes, encoding: str | None) -> bytes:
    if not encoding:
        return raw
    enc = encoding.lower()
    if enc == "gzip":
        return gzip.decompress(raw)
    if enc == "deflate":
        return zlib.decompress(raw)
    return raw


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    config: ProxyConfig  # set by make_server
    ledger: _Ledger

    def log_message(self, fmt: str, *args) -> None:  # silence default access log
        pass

    def do_GET(self) -> None:
        self._proxy()

    def do_POST(self) -> None:
        self._proxy()

    def do_PUT(self) -> None:
        self._proxy()

    def do_DELETE(self) -> None:
        self._proxy()

    # -- request side ---------------------------------------------------------
    def _read_request_body(self) -> bytes:
        if self.headers.get("transfer-encoding", "").lower() == "chunked":
            chunks = []
            while True:
                size = int(self.rfile.readline().strip().split(b";")[0] or b"0", 16)
                if size == 0:
                    self.rfile.readline()
                    break
                chunks.append(self.rfile.read(size))
                self.rfile.readline()
            return b"".join(chunks)
        length = int(self.headers.get("content-length") or 0)
        return self.rfile.read(length) if length else b""

    def _maybe_rewrite(self, raw: bytes) -> tuple[bytes, dict[str, str], Report | None]:
        """Returns (body to send, header overrides, report)."""
        if self.command != "POST" or not raw:
            return raw, {}, None
        ctype = self.headers.get("content-type", "")
        if "json" not in ctype:
            return raw, {}, None
        try:
            body = json.loads(_decode_body(raw, self.headers.get("content-encoding")))
        except (ValueError, OSError, zlib.error):
            return raw, {}, None
        if not isinstance(body, dict):
            return raw, {}, None
        new_body, report = rewrite(self.path, body, self.config.policy, self.config.locator)
        if new_body is None:
            return raw, {}, report
        data = json.dumps(new_body, ensure_ascii=False).encode("utf-8")
        return data, {"content-encoding": None}, report  # None -> drop header

    def _proxy(self) -> None:
        started = time.monotonic()
        raw = self._read_request_body()
        body, overrides, report = self._maybe_rewrite(raw)
        headers: dict[str, str] = {}
        for name, value in self.headers.items():
            if name.lower() in HOP_BY_HOP:
                continue
            headers[name] = value
        for name, value in overrides.items():
            for existing in [k for k in headers if k.lower() == name]:
                del headers[existing]
            if value is not None:
                headers[name] = value
        headers["Content-Length"] = str(len(body))
        target = self.config.upstream.base_path + self.path
        status = 502
        try:
            conn = self.config.upstream.connect(self.config.upstream_timeout)
            try:
                conn.request(self.command, target, body=body if body else None, headers=headers)
                resp = conn.getresponse()
                status = resp.status
                self._relay(resp)
            finally:
                conn.close()
        except (OSError, http.client.HTTPException) as exc:
            self._error(502, f"upstream unreachable: {exc}")
        self.ledger.record(
            {
                "ts": time.time(),
                "method": self.command,
                "path": self.path,
                "status": status,
                "elapsed_ms": int((time.monotonic() - started) * 1000),
                "request_bytes": len(raw),
                "sent_bytes": len(body),
                "report": report.as_dict() if report else None,
            }
        )

    # -- response side --------------------------------------------------------
    def _relay(self, resp: http.client.HTTPResponse) -> None:
        self.send_response(resp.status, resp.reason)
        content_length = None
        for name, value in resp.getheaders():
            low = name.lower()
            if low in HOP_BY_HOP and low != "content-length":
                continue
            if low == "content-length":
                content_length = value
                continue
            self.send_header(name, value)
        chunked = content_length is None
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")
        else:
            self.send_header("Content-Length", content_length)
        self.end_headers()
        try:
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                if chunked:
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                else:
                    self.wfile.write(chunk)
                self.wfile.flush()
            if chunked:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _error(self, status: int, message: str) -> None:
        payload = json.dumps({"type": "error", "error": {"type": "proxy_error", "message": message}}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def make_server(host: str, port: int, config: ProxyConfig) -> ThreadingHTTPServer:
    handler = type("BoundProxyHandler", (ProxyHandler,), {
        "config": config,
        "ledger": _Ledger(config.ledger_path, config.quiet),
    })
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server
