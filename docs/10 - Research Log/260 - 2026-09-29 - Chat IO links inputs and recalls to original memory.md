# Chat I/O links inputs and recalls to original memory

**Status:** Implemented; local lifecycle and integration tests pass. Model quality and latency have not been remeasured.
**Date:** 2026-09-29
**Applies to:** The shared chat interface and newly prepared engineering/research runs.

**Follow-up:** [Log 261](261%20-%202026-09-29%20-%20Single%20million-token%20Chat%20IO%20regression.md)
records the subsequent 1.115M-token, 100-question regression: 94/100, with durable
I/O links and Hebbian updates verified. It uses a fixed historical retrieval
snapshot and does not measure continuous native hierarchy refresh.

The application now captures conversation events at the model I/O boundary. A
request enters the durable session before recall or generation. The returned
packet is captured with its triggering input ID and exact original source
addresses. The completed model response is captured before it is returned to the
caller, with links to that input and packet. Tool results use the same journal
and retain the assistant call ID.

```text
input event -> durable capture -> native recall -> saved packet -> model
     |                              |                              |
     +------- input ID -------------+------ packet ID ------------ output
                                    |
                         exact original source spans
                                    |
                     existing Hebbian co-access graph
```

`ChatIO.exchange` owns that sequence. `ChatIO.invoke` wraps the generation
boundary when an application already assembled its current working window;
the engineering runner uses it for both arms. `ChatSession` owns admission,
indexing, packets, and links. The live JSONL interface and evaluation adapter
share these classes. This replaces the previous new-run policy of waiting for
recall, working-window eviction, or task completion to ingest new I/O.

All completed user, assistant, and tool events follow the same admission rules:
stable IDs, chronological order, exact text, original role, and capture time.
Identical retries deduplicate; conflicting reuse fails atomically. A session
directory permits one active writer. Background compilation coalesces arrivals
and uses the existing raw-summary, summary-only Qwen, and vector caches. Recall
waits for acknowledged input to be indexed; indexing errors leave the journal
intact and prevent an answer from quietly using stale memory.

Each packet stores its query, input event ID, rendered evidence, and exact span
receipts (source/turn IDs, character bounds, hashes, and section IDs). Readers
see source IDs in the evidence labels. `recalls_for_input` follows these links
without scanning the transcript. A recalled copy retains its immediate address,
the earlier packet ID, and flattened pointers to the original sources.

When the I/O wrapper completes a model exchange using a nonempty packet, it
records packet use through the existing `observe_context_access` path. Original
delivered spans map to durable chunk IDs; the triggering input supplies one node
within the existing 16-node bound. Existing rank discount, decay, degree limits,
and event deduplication apply. Recall copies and evidence marked as derived do
not strengthen their own associations. No new weighting or importance policy
was introduced. This measures co-access, not answer correctness. Merely returning
a standalone recall does not imply use; that caller can report an outcome.

The new binding retains the native index's existing per-session storage timestamp
convention. Actual capture times remain in the chat journal. The native router
still requires one occurrence timestamp per source; this change does not add
per-event historical time filtering to that router.

The callable boundary is in `application/chat_io.py`; durable links are in
`application/chat_session.py`; exact hydration and graph bridging are in
`application/chat_native.py`. The JSONL adapter is `interfaces/chat.py`. With a
fresh prepared run and its gateway worker active, the existing compiler/reader
binding can be launched from the repository root:

```powershell
$env:PYTHONPATH='src;.'
.pixi/envs/dev/python.exe -m tools.engineering_research_chat --run <new-run-directory> --session <session-directory> --actor <sealed-actor-json>
```

Example input lines (reuse IDs only for retries of the same operation):

```json
{"operation":"exchange","event_id":"u1","request_id":"r1","text":"Continue our implementation."}
{"operation":"recalls","event_id":"u1"}
{"operation":"flush"}
```

The exchange reply includes the response and saved packet with source pointers.
Lower-level `message`, `recall`, `feedback`, and `status` operations use the same
session. Clients assemble a streamed message before submitting it; this interface
does not persist each partial token. Native backend operations currently retain
the evaluation binding's separate-process installation/reopen behavior, so its
continuous-update latency needs measurement and optimization before a live SLO
can be claimed.

Validation: **84 focused tests passed.** Coverage includes live-versus-replay event identity, all message roles, ingestion
during compilation, failure recovery, atomic conflicts, writer/session isolation,
close/reopen, exact native hydration, source pointers, recall-copy ancestry,
co-access of input and original chunks, and retry after a graph commit. Matched
runner tests verify capture before generation/tool execution and both control
and memory arms using the same I/O wrapper. Models/vectors are deterministic test
doubles; transcript storage, native snapshots/hydration, and the Hebbian graph
are real application components. These are lifecycle checks, not new accuracy
or speed measurements.

Old sealed run plans preserve their historical ingestion policy and results.
New plans enable `chat_ingestion`. The legacy observe-only provider proxy and
file watcher remain separate utilities; this change does not attach arbitrary
external chat clients, including the current coding-agent conversation, to the
new interface automatically. Native cap-8 routing itself does not yet consume
the learned graph; this work connects I/O and original-memory co-access to the
existing graph without introducing a new routing policy.

## Follow-up: every input and output advances live memory

The turn stream is append-only: the compiler's application ingestion writes only
the suffix beyond the stored chronological prefix. Both the current user input
and the completed assistant output enter that stream, as do tool results and
recall receipts. The next recall waits for all prior captured events to reach
the native index. The fixed historical snapshot in Log 261 was a benchmark
boundary, not the live adapter's ingestion policy.

The live adapter now explicitly validates the installed and reopened prefix hash,
turn count, and native snapshot identity. It rejects rewinds, edits to an
acknowledged prefix, and stale recall receipts. A verified identical prefix is a
no-op. An unsuccessful refresh does not advance the acknowledged prefix.

Additional regression checks recall new facts from the user, assistant, and tool
turns without a manual flush, then recall them again after reopening. Compiler
checks prove that adding an input and then an output preserves earlier atomic
summaries and sends only the new content for raw summarization. **29 focused
tests passed** across chat lifecycle, native integration, and engineering live
tests. These use model doubles; they are not another real-model accuracy or
latency measurement.

Fast-forward currently describes raw-turn append and reuse of cached model work.
The compiler still traverses the summary population and republishes a complete
native hierarchy. This follow-up does not claim a fully incremental tree update
or validate its continuous-update latency at one million tokens.
