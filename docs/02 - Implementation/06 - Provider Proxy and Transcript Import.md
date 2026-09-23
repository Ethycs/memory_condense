# Capturing conversations: the provider proxy and transcript import

**Status:** implemented and test-covered (`tests/test_proxy.py`,
`tests/test_proxy_scheduler.py`, `tests/test_transcripts.py`); observe-only —
the proxy does not yet rewrite prompts

**Date:** 2026-08-23
**Updated:** 2026-09-04

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
`GET /_memory/health` reports mode, queue depth/age/retained bytes, admission
outcomes, sink retries and terminal failures, stage-specific completion
outcomes, worker state, and an age-aware snapshot of durable T1/T2 backlog.
The snapshot is sampled only by the serialized worker, so the HTTP event loop
never shares its SQLite connection. Its top-level `status` is `ok`,
`degraded` after capture-admission or terminal-capture loss, or `failed` when
the capture worker fails; `health_reasons` provides stable machine-readable
causes. Failed worker health returns HTTP 503, while degraded service remains
HTTP 200.

No new dependencies: `starlette`, `httpx`, and `uvicorn` already ship with the
`mcp`/`litellm` stack.

### Four rules the implementation enforces

**Capture never breaks the call, but admission is explicit.** Body parsing,
the capture queue, and the ingest sink are each wrapped. Production HTTP uses
the awaited `CaptureQueue.offer_async()` API: a full queue first applies
bounded backpressure; once its waiter, time, count, or retained-payload-byte
budget is exhausted the new capture is rejected and counted while the
client's response proceeds untouched. Reservations include producer waiters,
queued captures, in-flight batches, and captures sleeping for a sink retry.
Async producers enter one ordered admission lane, so a later producer cannot
bypass an earlier producer already waiting for capacity.
The byte figure is deterministic UTF-8 payload accounting, not a claim about
exact Python RSS. The compatibility `CaptureQueue.offer()` API remains a
synchronous, immediate, fail-open offer and returns whether admission
succeeded. An admitted capture is never evicted to make room for a newer one.
Queue admission is not durability: only the SQLite turn/manifest commit is.

**Capture and completion are separate service tiers.** The focused scheduler
in `memory_condense.interfaces.proxy_scheduler` owns one worker and greedily
coalesces admitted exchanges. It captures each exchange in FIFO order without
invoking a model. A sink exception retries the same FIFO item with capped
exponential backoff. Retries are bounded; exhaustion increments
`captures_terminal_failures`, releases that item, and lets later work proceed.
This prevents one poison capture from wedging shutdown, but it also means the
counter represents explicit capture loss. There is no crash-safe dead-letter
replay until a future disk spool exists.

Whenever the RAM queue is empty, a serialized completion tick attempts bounded
T1 base-index work with enrichment disabled. T1 retains priority while it is
making progress, but a separately bounded T2 attempt is allowed when T1 fails,
makes no progress, or is observed empty. That exception matters: an unrelated
poison T1 manifest cannot permanently strand an already-indexed T2 receipt.
Transient callback, T1, T2, snapshot, and no-progress failures retry after
capped exponential backoff without waiting for another capture. Captures
arriving during backoff win immediately. Under sustained capture traffic, a
configurable capture-batch quota forces one maintenance tick, preventing T1
starvation. A low-frequency idle poll also rechecks the durable journals after
an empty snapshot, so receipts committed by another process are eventually
serviced. The poll waits on the capture queue with a timeout: a prompt or STOP
wakes it immediately and wins the scheduling boundary. A FIFO stop sentinel is
placed after all accepted work and forces exactly one final bounded maintenance
tick, even during backoff.

The durable tiers are deliberately distinct:

- **T0, captured:** the turn, text-free exact chunk manifest, global chunk-ID
  reservations, and (when requested) T2 obligation commit together before any
  embedder, native index, or extractor call. Queue admission alone is not T0.
- **T1, searchable:** one bounded whole-manifest batch reconstructs the sealed
  chunks and proves dense, HNSW, and BM25 population in the transaction that
  advances `pending -> indexed`. The first over-budget manifest is still
  admitted, so bounds cannot deadlock a large turn.
- **T2, enriched:** already-indexed live evidence is extracted, validated,
  canonically staged with a digest, and applied before `pending -> enriched`.
  It never re-embeds the source chunks. A successfully parsed empty operation
  set is a real no-op; provider, JSON, or schema failure on the durable
  extractor path raises and leaves the receipt retryable.

Fresh work and due retries alternate when both are eligible. T1 gives a failed
bounded cohort one same-cohort retry, then isolates its members into singleton
attempts so a poison manifest cannot consume every slot. T2 is selected one
turn at a time. Attempt history, error kind, next eligible time, and the first
canonical staged T2 result are durable, while retry delays are bounded.

Deferred T2 has create-only authority. Safe creates can publish and the parent
receipt can finish even when the result also contains reversals: every grounded
`Correction` becomes an independent durable operation record instead of an
active memory. An operator later resolves one against a reviewed active target
(with live-evidence, revision, no-op, and collision checks repeated under the
writer lock) or dismisses it. Both terminal decisions retain the immutable
original operation for audit.

Source-order retirement receipts prevent delayed T2 from recreating an identity
that an equal-or-later turn updated, deleted, superseded, or deduplicated. v15
cannot infer that chronology for pre-v15 terminal memory rows. Operators can
bind a known legacy retirement ordinal once, within the migration boundary.
Likewise, each T2 receipt that was pending during migration—or is later claimed
for a source turn at or before that boundary—is exactly and durably quarantined
from automatic replay. `pending_legacy_enrichments()` lists operator-actionable
quarantined receipts once T1 is indexed; `discard_legacy_pending_enrichment()`
records a distinct `discarded_legacy` disposition only after T1 is indexed,
leaving its searchable chunks intact.

Schema v15 is an explicit stop-the-world upgrade. SQLite writer-fence triggers
make an already-open pre-v15 connection fail closed on its next covered write,
but they do not make rolling mixed-version operation safe. Stop all writers,
migrate once, and restart every process before accepting traffic. This restart
is also required for the process-local native HNSW graph: a SQLite trigger
cannot reconcile index state held by an older live process.

Queue size, waiter count, retained-byte budget, offer/shutdown timeouts, worker
batch size, maintenance quota/backoff, sink retry/backoff, T1
manifest/chunk/token bounds, idle polling, and the T2 turn bound are explicit
CLI/`ProxyConfig` settings. `ProxyConfig` and programmatic `build_app()` keep
full prompts by default for arbitrary custom sinks. The production CLI instead
compacts each capture to the last user turn before the streaming closure and
queue; `--retain-full-capture-prompt` opts out. The original full-prompt token
estimate and request digest survive compaction.

Shutdown closes admission before placing the sentinel and reports a timeout
if its deadline expires. That deadline bounds the scheduler await and state
transition, not hard process exit: Python cannot kill a synchronous function
already running in a default-executor thread, and interpreter teardown may
wait for that thread. The condenser is deliberately left open while such a
worker remains live, avoiding a use-after-close race. A truly bounded exit for
an untrusted or potentially hung sink/embedder requires a subprocess worker
that the parent can terminate; this proxy does not yet provide that boundary.

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
plus the assistant reply. The user identity is the conversation ID plus the
full request SHA-256. The assistant identity additionally includes the full
reply-text SHA-256. Exact retries are idempotent, while edited requests at the
same conversation position and stochastic replies cannot collide with an
earlier turn merely because a parser assigned the same positional ID.

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
