# Capturing conversations: the provider proxy and transcript import

**Status:** implemented and test-covered (`tests/test_proxy.py`,
`tests/test_transcripts.py`); observe-only — the proxy does not yet rewrite
prompts

**Date:** 2026-08-23

Memory needs conversations. There are two ways to get them, and this project
now has both: a proxy that sees live traffic, and an importer for the exports
you already have.

## 1. The proxy

`memory_condense.interfaces.proxy_server` speaks the Anthropic and OpenAI wire
protocols. A client points its base URL at the proxy; the proxy forwards each
request upstream unchanged and captures the exchange on the way past. Memory
becomes a property of the transport, so no client needs to integrate anything.

```bash
python -m memory_condense.interfaces.proxy_server --port 8787 --data-dir data

ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude
OPENAI_BASE_URL=http://127.0.0.1:8787 python my_openai_script.py
```

Captured endpoints are `/v1/messages` (Anthropic) and `/v1/chat/completions`
(OpenAI), streaming or not. Every other path is proxied untouched, so model
listings, embeddings, and files keep working through the same base URL.
`GET /_memory/health` reports mode and capture counters.

No new dependencies: `starlette`, `httpx`, and `uvicorn` already ship with the
`mcp`/`litellm` stack.

### Three rules the implementation enforces

**Capture never breaks the call.** Body parsing, the capture queue, and the
ingest sink are each wrapped. A throwing sink, a full queue, or an
unparseable body is counted and dropped while the client's response proceeds
untouched. A proxy in the critical path must fail open.

**Observe before augment.** The default `observe` mode forwards request bytes
verbatim, so installing the proxy cannot change any answer. `ProxyConfig(mode=
"augment")` currently *raises* rather than silently behaving like observe —
prompt rewriting is the project's point, but it must arrive as a measured,
opt-in change, not a silent one.

**Credentials pass through and are never stored.** The caller's `x-api-key` /
`authorization` headers are copied upstream and redacted from any receipt view
(`redacted_headers`).

The proxy strips `accept-encoding` from forwarded requests and
`content-encoding` from relayed responses, so bodies cross it uncompressed and
are never re-encoded. That costs some bandwidth between proxy and upstream and
removes a whole class of corruption risk; the client still sees correct bytes
either way.

### What gets ingested

Only the newest turn pair. A chat client resends the whole history on every
call; re-ingesting all of it would duplicate the conversation on each
exchange. `ExchangeCapture.ingest_records` therefore emits the last user turn
plus the assistant reply.

Conversation identity comes from an `x-memory-conversation-id` (or
`x-conversation-id`) request header when the client supplies one, and
otherwise from the request digest. Without that header each exchange stands
alone — correct, but unthreaded. Clients that can set one should.

Streaming replies are reassembled by `StreamAccumulator`, which is fed a copy
of each forwarded chunk. It tolerates chunk boundaries mid-frame and truncated
streams, returning whatever text arrived.

## 2. Transcript import

`memory_condense.ingest.transcript_source.TranscriptFile` reads exports you
already have: ChatGPT account exports (mapping trees), Claude account exports
(`chat_messages`), Anthropic Messages bodies, and JSONL forms of each.

```python
from memory_condense.application.condenser import MemoryCondenser
from memory_condense.ingest.transcript_source import TranscriptFile

transcript = TranscriptFile("~/Downloads/conversations.json")
summary = condenser.ingest_transcript(transcript)   # ingests everything new
```

The file is memory-mapped and indexed by byte range, so conversations decode
one at a time rather than the whole export becoming one Python object. The
index scanner tracks JSON string state, so braces, brackets, and escaped
quotes inside message text do not shift element boundaries.

ChatGPT stores edits and regenerations as sibling branches; the parser
reconstructs the longest root-to-leaf path, the conventional reading of "the
conversation as last seen".

### Change is handled by re-indexing, not by writing through the mapping

The mapping is opened **read-only** deliberately. Every corpus source pins a
`sha256` and every evidence span pins a `quote_sha256`; editing bytes beneath
stored spans would silently invalidate provenance for everything already
ingested from that file. Live transcripts are therefore append-mostly:

| `refresh()` status | Meaning | Pending work |
| --- | --- | --- |
| `new` | first index of this path | every conversation |
| `unchanged` | identical size and digest | none |
| `appended` | existing conversations intact, new ones added | only the new ones |
| `rewritten` | at least one existing conversation changed or was removed | only changed ones |

`ingest_transcript` ingests only the pending set, so a growing export costs
work proportional to what changed. Message IDs become turn IDs and
conversation IDs become source IDs, so re-ingesting an edited conversation
replays the same identities instead of duplicating history under fresh ones.

## 3. What this does not do yet

- **No prompt rewriting.** The cost win — swapping bulk history for a
  retrieved packet — is the `augment` mode that is currently refused. It needs
  a matched evaluation before it can be trusted in front of real traffic.
- **No tool-call capture.** `tool_use` / `tool_result` blocks are forwarded
  correctly but contribute no ingestable prose today.
- **No multi-user isolation.** One proxy process writes into one store. Serving
  several users requires a store per conversation namespace.
