# Capturing conversations: the provider proxy and transcript import

**Status:** observe capture and opt-in native memory implemented; covered by
`tests/test_proxy.py`, `tests/test_proxy_memory.py`,
`tests/test_proxy_scheduler.py`, and the shared chat lifecycle tests.

**Date:** 2026-08-23
**Updated:** 2026-10-08

Memory needs conversations. There are two ways to get them, and this project
now has both: a proxy that sees live traffic, and an importer for the exports
you already have.

## 1. The proxy

`memory_condense.interfaces.proxy_server` speaks the Anthropic and OpenAI wire
protocols. Its default `observe` mode forwards requests unchanged and captures
exchanges. Opt-in `augment` mode replaces older history with recalled evidence
using the evaluated resident ChatSession backend. Both modes use the client's
own provider credential and configured upstream URL.

### Native memory beta (`--mode augment`)

The proxy is distributed as version 0.2.2 with an installed `memory-condense`
command and a packaged resident runtime. It no longer imports research tools
or reads model paths relative to the checkout. See the root
[installation guide](../../README.md) for the Pixi build, standalone installer,
model setup, diagnostics, and client example. The native runtime currently
targets Windows x64 and Linux x86_64 with CUDA and Python 3.11–3.12.
Linux additionally requires glibc 2.38 or newer, the floor set by the pinned
llama.cpp CUDA 12.8 build, which is the lowest Linux CUDA build upstream
publishes at tag b11272.

After installation:

```powershell
& "$env:LOCALAPPDATA\memory_condense\proxy\memory-condense.cmd" proxy `
  --openai-base-url https://central-dev.zt:4000/v1
```

Set the client's OpenAI-compatible base URL to `http://127.0.0.1:8787/v1`.
For Anthropic Messages, use `http://127.0.0.1:8787` and configure the upstream
with `--anthropic-base-url`. OS certificate trust remains enabled, including
for the private gateway. Supply these headers on every memory request:

| Header | Value |
| --- | --- |
| `Authorization` or `x-api-key` | The client's provider credential |
| `x-memory-conversation-id` | Stable identifier for this conversation |
| `x-memory-request-id` | Unique identifier for this generation; reuse for an identical retry |

The conversation header is required (`x-conversation-id` is also accepted).
Request IDs default to the exact request-body digest when omitted. Reusing a
request ID with different body bytes returns 409. A completed retry returns
the saved public response without another generation, including after restart.
Edited branches need a new conversation ID. Clients can send their full
history, a matching suffix, or only the new input/tool result.

The lifecycle is input capture, recall, generation, output capture, then
background preparation and ordered publication. System/developer instructions
remain in their original policy channels. Native tools and tool results remain
provider-shaped. The default recent window is twelve exchanges and 8,192 tokens;
unindexed input is retained until publication. An initial import larger than
that budget waits for indexing before dropping history. Cold model startup and
that initial import can therefore delay service.

The resident stack shares local Qwen attention, FastEmbed BGE on CPU, and the
local Llama fallback summarizer across conversations. Ordinary text requests
ask the selected reader for the visible answer and hidden exchange summaries in
one generation. Valid summaries are reused by ingestion; original source spans
and recall links survive into the existing Hebbian co-access learning. A
completed response records use of a packet, not a guarantee of answer accuracy.
Tools, structured-output schemas, custom stop sequences, and log probabilities
retain their native output contract and use background summarization instead.
Native ingestion stores exact raw IO and stable source chunks for learning,
then indexes the routing summaries. It does not embed or maintain the legacy
dense/lexical index for new raw turns or delivered recall copies. Native capture
receipts are separate from the legacy ingest journal: capture alone does not
claim that either search index has been published. Learning accepts unembedded
chunks only after the native snapshot covers their exact original sources.

Warm publication reuses authenticated raw prefixes and checks changed span
partitions and new turns. Full checks remain on cold admission and recovery;
compact index membership and complete snapshot receipts still bind the result.
Older full snapshots migrate to the incremental store during startup. Feedback
is applied after each committed prefix, including while more ingestion is ready.
The runtime writes completed and failed sync timings to
`ingestion-timings.jsonl` in each conversation directory.

`--no-inline-memory` selects that behavior for every request. Inline requests
reserve 512 additional output tokens (4,096 + 512 when no limit was provided).

**Local summaries with API escalation (0.2.2):** valid local
Llama extraction and merge results stay local. Incomplete generations, invalid
or missing source extracts, local transport errors, and an unavailable Llama
runtime escalate the failed job to the conversation's current provider/model.
The API must return exact quotations from each input fragment; invalid API
results leave indexing pending instead of publishing replacement prefixes.
The prompt explicitly includes metadata/recall fragments in its coverage
requirement. These checks establish source ownership and completion, not
semantic completeness or answer correctness.

Qwen attention and BGE embeddings remain local with their existing identities.
Merge escalation sends the user spine and existing summaries; raw extraction
escalation sends its source fragments. Normal answer generation still requests
inline summaries in the same call. Escalation adds a separate paid background
call only when a local summary job fails, or local summarization is unavailable.
As with other ingestion, a backlog can delay a freshness barrier or final flush.
`--no-memory-summary-api-fallback` restores the previous local-only compiler
behavior. With escalation enabled, missing Llama assets do not prevent proxy
startup; the required Qwen/BGE assets still must be present.

Provider credentials stay in memory, scoped to the existing credential/provider/
conversation namespace. Summary calls use the same provider and model without
the visible request's tool definitions, response schema or idempotency headers.
The HTTP transport stays available to compiler workers after the reply and
through final shutdown drain. No credentials are written to summary receipts.

Accepted repairs are cached by source, typed request, policy and provider/model
under `sessions/<namespace>/summary-fallback/<job digest>/`. Request, reservation,
response and acceptance/error files retain the outcome. A job gets at most one
API attempt per policy/model identity: invalid results and unknown transport
outcomes are retained without automatic resend. Review the receipt before
retrying; there is no automatic repair/reset command yet. Selecting another
model permits a new attempt. After restarting, send a normal chat request to
rebind the provider/model before flushing pending summary work; an unbound flush
cannot silently restore local replacement-prefix behavior. Existing compiled
summaries are not automatically rebuilt by this policy.

Live proxy escalation uses the versioned `semantic-summary-grounding-v1`
validator namespace. A summary may paraphrase its fragment when pinned BGE-M3
maps it to a short exact support span and deterministic gates preserve
identifiers, quantities, named entities, negation, and completion status. The
stored support remains an exact substring with the original raw pointer. The
policy payload and SHA-256 appear in the summary request receipt and proxy
health response. Changing any verifier rule or threshold creates another cache
namespace; it does not overwrite outcomes admitted by an earlier policy.

**Streaming limitation:** augment mode buffers the upstream response through
validation and durable capture. It returns compatible SSE with hidden summaries
removed, but does not provide incremental first-token delivery. Tool SSE is
returned with its original framing after capture. Observe mode still streams
as it arrives. Truncated or malformed responses do not become successful memory.
Only a confirmed terminal empty completion can retry, at most twice by default;
timeouts and unknown provider outcomes are retained and are not resent.

`GET /_memory/health` reports active sessions, pending events/feedback, and
indexing failures in augment mode. `summary_routing` additionally counts local
acceptances/failures, API calls/acceptances/failures and cache hits for the current
process. `POST /_memory/flush` drains one conversation;
send its same conversation and credential headers, plus
`x-memory-provider: anthropic` for an Anthropic conversation (default: `openai`).
Normal shutdown drains all sessions and releases the resident models. The
default active-session limit is eight; `--memory-max-sessions` changes it.
Restarting releases active slots without deleting stored conversations.

Each credential/provider/conversation combination gets a separate hashed
namespace, journal, and index. Rotating credentials selects a new namespace.
Raw I/O, the actual served payload, and provider attempt bodies are persisted
locally for recovery and diagnosis. Credential headers and cookies are excluded
from saved responses. This is a local beta service, not a hosted authentication
or tenant-management service.

Supported memory endpoints are `/v1/chat/completions` and `/v1/messages` with
text and native tools. `/v1/responses` and multimodal memory requests return an
explicit error. The proxy relays other endpoints but does not add memory to them.
The Python `proxy` extra declares the resident runtime dependencies. The Pixi
installer combines the built conda package with its pinned PyPI additions.
`memory-condense setup` downloads pinned assets; `setup --reuse PATH` registers
an existing asset cache after verification, and `doctor` checks the environment.

### Dialogue-wide retrieval module

Version 0.2.2 enables the module in
`memory_condense.application.dialogue_retrieval`.
The proxy defaults to `--memory-retrieval-policy auto`: a shared
CPU BGE classifier chooses cap-8 or hierarchy plus exact neighborhood evidence.
Explicit `cap8`, `dialogue` (raw neighborhoods only), and
`hierarchy_neighborhoods` modes bypass the classifier.

The module ranks stored leaf summaries, reserves chronology coverage, balances
speaker roles and neighboring turns, and optionally prioritizes closing assistant
responses. Qwen's stored attention boundaries remain the topology. Routing
does not read raw transcript text or call Qwen or a reader model. Original raw
spans are hydrated after selection and delivered to the configured answer model.
An assistant closing response is evidence of what it said, not independent
verification that its claim was correct.

The default profile reproduces the evaluated closing-response policy:

```json
{
  "version": 1,
  "episode_limit": 12,
  "chronology_bands": 4,
  "closing_responses": true,
  "neighbor_turns": 1,
  "max_context_tokens": 6144,
  "max_raw_spans": 128
}
```

Save a profile outside the installed application directory, then start:

```powershell
memory-condense proxy --memory-retrieval-policy dialogue `
  --memory-dialogue-config C:\memory-config\dialogue.json `
  --openai-base-url https://your-gateway.example/v1
```

Unknown settings and invalid types fail before model startup. Neighborhood count
is 1–256; chronology bands are 1–the neighborhood count. Neighbor distance is
0–4 turns within a selected neighborhood. Raw-span budget is 1–4,096 and context
budget is 1–65,536 tokens. The token cap applies to the hydrator's rendered
evidence; provider prompts, memory envelopes, recent exchanges and the answer
have additional tokens. These ranges bound configuration, not measured quality
or capacity guarantees. The profile is immutable for a running service; restart
after tuning. No new package build is needed for a profile-only change.

Each published memory prepares its leaf-summary vectors with the shared BGE
encoder and existing content-addressed disk cache. A first enable may embed leaf
summaries once; unchanged summaries reuse vectors across appends and restarts.
Recall embeds its query once. It never compiles a new summary index or loads a
new model. In automatic mode, an uncertain first classification can embed one
additional input containing the preceding six user/assistant messages, bounded
to a 256-token suffix. Current input, tool messages, and internal recall copies
are excluded from that context. Remaining uncertainty and questions exceeding
the classifier's 512-token budget use cap-8. Encoder identity errors still fail
validation; they are not disguised as ordinary uncertainty.
A preparation failure leaves concurrent recalls on the prior complete
snapshot. The ordinary background ingestion and exact-source Hebbian learning
path stays in use, including ancestry for previously recalled copies.

`/_memory/health` exposes `retrieval.mode` and the validated profile/hash.
Recall journal events also retain the effective policy, selected neighborhood
IDs, source-bound index/route hashes, and actual packing budgets/counts under
`metadata._chat.retrieval`. Original span references are still stored alongside
them. Provider messages receive evidence, not these internal diagnostics.

For Python embedding, pass a `DialogueRetrievalPolicy(...)` as `dialogue_policy`
and choose `retrieval_policy='auto'`, `'dialogue'`, or
`'hierarchy_neighborhoods'`, or `'decision_history'` on `NativeProxySessions` or `ResidentNativeBackend`.
The session factory defaults to auto; the lower-level backend retains its
existing cap-8 default unless configured. A caller that already owns
an authenticated native memory can call `memory.configure_dialogue(policy,
episode_matrix, embedding_identity=...)` before publishing it. Matrix rows must
match the native hierarchy's leaf order, be normalized FP32, and use the live
query encoder's identity. The module itself has no model or database dependency
beyond the existing native search components.

The combined tactic adds authenticated stored hierarchy summaries, capped at
24,576 rendered tokens, in addition to the profile's raw-evidence budget.
It sends the entire eligible hierarchy when it fits. Otherwise it prioritizes
selected neighborhoods and ancestors, omits whole summaries, and reports the
included/omitted counts. Summary labels and pointers are distinct from exact
raw `<MEMORY>` passages. Summary-only claims are not independent verified
evidence and do not create Hebbian learning credit; delivered raw spans use the
existing original-source learning path. Receipts retain summary IDs, hashes,
source/turn pointers, classifier decisions, effective tactic, and packet size.

Roll back by restarting with `--memory-retrieval-policy cap8` and removing
`--memory-dialogue-config`. Existing conversations and their learning remain
usable. Completed request IDs retain their saved responses even after a policy
change; use fresh request IDs for tuning comparisons. The first integration
replay matched all 15 saved closing-policy evidence packets on the 775,741-token
engineering memory. This proves integration parity, not a new accuracy result;
the [research record](../10%20-%20Research%20Log/263%20-%202026-10-06%20-%20Dialogue-wide%20neighborhood%20retrieval.md)
contains the quality results and their limits.

### Query-driven decision and fix history

Automatic routing includes `decision_history` for requests to reconstruct past
decisions, attempted fixes, outcomes, corrections and unresolved work. A sealed
21,659-byte binary head shares the existing CPU BGE query vector with the scope
classifier; it adds no transformer, query embedding or provider request. The
head proposes history intent, not a factual interpretation of the archive.
Ordinary fact lookups and requests to perform new work keep the existing tactic.
The prior focused/overview/uncertain head and its contextual fallback still run.

The history tactic uses the same hierarchy/neighborhood selection, source/date
scope and exact raw hydration. It adds D labels on delivered passages and a
bounded annotation table (at most 2,048 additional tokens). The table records
mention time, original speaker attribution, exchange membership and structural
`follows` links. It reports annotation omissions. Those links never imply
causation, successful verification, effective dates or supersession. The reader
is guided to distinguish proposals, attempts, reported implementation, tests and
releases, and to cite both sides of any inferred correction.

Existing `MemoryStore` supersession links are captured with the published
snapshot. A link appears only when independent exact passages containing both
support quotes were delivered. Missing endpoints suppress the link. Native
summary-only archives often have no typed revision records: the reader can
explain an explicitly supported change, but this tactic does not extract or
persist a new fact ledger. Scope remains the current conversation namespace;
cross-conversation project membership is a separate feature.

Original references, recall ancestry, request replay and Hebbian learning retain
their existing behavior. Annotations do not create independent evidence or
learning nodes. The durable recall receipt includes the selected head identity,
annotation coverage and revision links under `metadata._chat.retrieval`.

Force the tactic with `--memory-retrieval-policy decision_history`. Use
`hierarchy_neighborhoods` for the same broad retrieval without these annotations,
or `cap8` for focused recall. Each is an explicit alternative to automatic BGE
selection; existing saved responses still replay unchanged.

The second classifier confirmation got 51/52 fresh authored cases correct
(25/26 history requests, 26/26 negatives); none of 200 saved factual query
vectors selected the new tactic. These are intent results, not answer accuracy.
The first confirmation was 32/40 and was explicitly promoted to development
before the second fit. See the [research record](../10%20-%20Research%20Log/264%20-%202026-10-08%20-%20BGE-only%20tactic%20prioritization.md).

### ADR evidence and fresh-chat handoff conventions

Broad hierarchy packets and decision-history packets use the ADR and developer-guide
structure from [Ethycs/claude-chat-decompose](https://github.com/Ethycs/claude-chat-decompose/tree/c103a71b6846027395165b8c3d9070d3bbad762c).
The adaptation is packaged in `application.continuity`; its upstream MIT notice ships
in `runtime/data/chat-decompose-LICENSE.txt`. It does not require the workflow's
Claude agents, transcript splitting scripts, or another provider call at runtime.

Ask, for example: “Prepare a fresh-chat handoff: current design and progress,
decisions and why, superseded approaches, unresolved problems, and next actions.”
The existing BGE scope/history heads select the retrieval tactic. The reader then
uses the requested format in the ordinary answer call:

- **ADRs:** context, decision, consequences, alternatives and sources; decision
  status and source date; `PIVOT`, `LOCK-IN`, `SCOPE-CUT`, or unknown when unsupported.
- **Handoff:** purpose, supported current design/progress, rationale, discarded
  alternatives, open work and next actions, with a portable source map.
- **Revision history:** citations to both sides of a supported change, with
  chronology kept separate from supersession and unresolved conflicts retained.

An initial real-archive check routed the explicit handoff and ADR prompts broadly,
but “What does the next engineer need to know to continue this project?” fell back
to focused recall. Handoff intent is not a separately trained classifier class.
For an explicit handoff session, `--memory-retrieval-policy decision_history`
forces the broad evidence and conventions. These three authored checks are not a
general routing-accuracy estimate.

Unlike the source template's initial `Accepted` default, chat proposals remain
proposals unless acceptance is supported. Accepted design is also separate from
reported implementation, observed testing and release. Conversation dates are
mention dates, not automatically effective dates. A handoff describes the retrieved
snapshot, not guaranteed present-day state.

`metadata._chat.retrieval.continuity` journals the format version, pinned workflow,
index/hierarchy hashes, routing date, eligible source IDs, coverage and citation
bindings. Exact `D` labels bind to existing raw reference indices and span hashes;
summary `H` labels bind to hierarchy reference indices and section hashes. Labels
are packet-local: a portable answer is instructed to map used labels back to original
source/turn IDs and character ranges, or source/turn ranges for summaries. The
contract adds a small prompt overhead, reported separately as `token_count`;
the packet total includes it. Retrieval, exact raw spans and learning credit are
unchanged. Focused cap-8 recall does not receive the contract.

These are reader conventions with an authenticated evidence manifest. The runtime
does not enforce generated prose correctness or persist generated ADRs as a verified
decision ledger. It does not run the workflow's complete-archive coverage verifier.
A new conversation still has its own namespace: copy the generated handoff into it;
automatic project-wide sharing across conversations remains separate work.

The initial live Sol smoke used the existing 775,741-token engineering archive,
one 28,045-token packet, all 51 eligible hierarchy summaries and 11 exact spans.
The conventions themselves added 395 tokens. All 11 labels cited by the answer
resolved to delivered evidence. The answer distinguished an older clean commit
from later uncommitted work, but shortened several source IDs in its prose map.
The evaluator's `handoff.json` therefore includes a deterministic full-address
map copied from the journal bindings. A portable client should render addresses
from that manifest instead of relying on the model to reproduce hashes. Generated
decision-status claims still require evidence review; this one smoke is not an
answer-accuracy benchmark. Evidence: `eval_results/continuity-handoff-20261008-r2`.

### Observe mode

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

The observe scheduler and capture guarantees below remain unchanged.

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
verbatim. `ProxyConfig(mode="augment")` requires an explicit memory service
when building the app programmatically; the CLI supplies `NativeProxySessions`.

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

## 3. Local tactic prioritization (2026-10-08)

The checkout includes a BGE-only classifier for deciding whether to
prioritize focused recall (`cap8`) or dialogue-wide retrieval
(`hierarchy_neighborhoods`). An underspecified request returns
`preferred_tactic: null` and `abstained: true`. The standalone command proposes
a tactic. The proxy's default auto mode dispatches it as described above,
including contextual reclassification and a cap-8 fallback for abstention.

With the existing local assets installed:

```powershell
memory-condense route --query "What have we tried across this project?" --assets-dir .cache
memory-condense route --query "Is this working?" --recent "We are reviewing the project's experiments and original goals." --assets-dir .cache
memory-condense route --jsonl --assets-dir .cache
```

JSONL mode accepts one object per line, for example
`{"query":"Is this working?","recent":["We are checking one function's return value."]}`.
It keeps one CPU BGE-M3 encoder resident for all requests. Invalid requests
produce an error object without ending that stream. No models are downloaded by
this command; use the existing setup command first. A single-query invocation
pays model startup each time.

Applications already holding a `LocalRuntime` can call
`runtime.tactic_router().prioritize(question, recent=recent_messages)`. The
classifier shares `runtime.embedding()`; it starts no Qwen, Llama, or API reader.
Its packaged linear head adds about 64 KB. The classifier requires the exact
FastEmbed embedding identity used in training; it rejects a different encoder
instead of silently applying incompatible weights.

Pass recent messages or summaries oldest first. The classifier keeps up to six
nonempty entries within a 256-token suffix, with a separate 512-token question
budget. It does not read the memory archive or hydrate raw evidence. Softmax
scores are rankings, not calibrated correctness probabilities. The selected head
abstains through a learned `uncertain` class; no additional confidence threshold
was selected. Direct API callers must handle abstention explicitly; the proxy
does this through focused recall.

For context-free questions, `prioritize` can reuse an existing query embedding
through `input_vector={"vector": vector, "text_sha256": quote_sha256(text),
"embedding_identity": summary_embedding_identity(encoder)}`. Here `text` must
exactly equal the first result of `render_input(question, recent)`. A contextual
decision requires the embedding of that complete rendered input. Reuse avoids
another encoder call; passing a bare vector is rejected.

Measured CPU embedding cost was 0.131 seconds mean and 0.215 seconds p95, plus
about 0.093 milliseconds for the head. The confirmation results and remaining
errors are recorded in [Research Log 264](../10%20-%20Research%20Log/264%20-%202026-10-08%20-%20BGE-only%20tactic%20prioritization.md).

### Optional native backend experiments

`--memory-experiment-config JSON` enables bounded learned-association, heat,
Qwen/CAV summary attention, temporary head-memory and Qwen-weighted IO learning
experiments in `augment` mode. These reuse the resident model stack and preserve
the baseline packet under a separate additions budget. The default is off.
See [Native Experimental Modules](07%20-%20Native%20Experimental%20Modules.md)
for the module inventory, profiles, provenance rules and measured limits.

## 4. Validation and remaining limits

Focused tests exercise both provider formats, streaming and nonstreaming
responses, hidden summaries, history replacement, tool calls/results, exact
retry replay after reopening, concurrent duplicates, crash boundaries,
credential namespace separation, malformed responses, and source-linked learning.
These use the real HTTP handler and ChatSession with a lightweight backend.
The final affected suite passed **181 tests** across the proxy, scheduler,
context handling, shared chat lifecycle, native backend, and embedding scheduler.
CLI help, module compilation, and `git diff --check` also passed.

The real-model probe in `eval_results/proxy-native-20261002-r2` completed two
Sol requests with the local resident backend. The second recalled the older
readiness contract after the imported chat was indexed. The tool request then
returned a gateway 404; replaying the same served payload directly returned the
same 404. A separate Haiku tool preflight was blocked by upstream API credit.
Consequently this is evidence for live text recall, not a completed live tool
cycle. No new 1M accuracy score is claimed for this HTTP integration.
The separate-process `audit.json` matched all **39** journal events to stored
transcript text, verified **40** original source pointers, reopened the native
index, and found the single applicable learning update persisted with zero
pending events or feedback. Both successful replies supplied accepted inline
summary pairs. The failed request was retained without a successful learning
update. Provider traffic used real HTTPS; inbound HTTP ran through Starlette's
test client, and no external listener was deployed by the probe.

Remaining limits are live tool-route availability on the configured gateway,
buffered streaming in memory mode, Windows/CUDA native runtime support, and
Responses API/multimodal support. Observe mode retains its original newest-pair
capture and single-store behavior; native tool capture and separate conversation
stores are augment features.

### Installable package verification (2026-10-02)

`pixi build --output-dir dist` produced the 0.2.0 conda artifact. The standalone
PowerShell installer installed it and its dependencies into a separate Pixi
environment. Setup reused the existing cache, verified all 69 pinned asset
files, and passed the installed `doctor` check with the RTX 2070 SUPER.
The resident code now lives under `memory_condense.runtime`; research tools
import that implementation through compatibility modules.

The installed HTTP acceptance run in
`eval_results/proxy-installed-smoke-20261002-r1/report.json` used real local
Qwen/BGE/Llama models and a controlled loopback provider. All four requests
completed, including a native tool-call/result cycle. Older evidence was
recalled after indexing, an identical retry made no extra provider request,
the final pending-event/feedback counts were zero, and model shutdown completed.
The test imported the installed package from its environment and rejected
imports of `tools` and `tests`. It reused the existing local model/runtime
cache; it was not a new model download or an answer-quality/latency benchmark.
After the final artifact refresh, a separate installed-process audit matched
the packaged runtime files to source, reopened all 44 stored events, checked
53 original source pointers, and verified three persisted learning updates.
FastEmbed and nvCOMP also imported directly from the installed environment.
Evidence is in that run's `package-persistence-audit.json`.

The affected regression suite passed 191 tests. The final packaging suite
passed 10 tests, including one added cold-download/cache-reuse test (192 distinct
tests across both runs). Download tests use controlled responses and exercise
pinned revisions, archive verification, cache reuse, and traversal rejection.
The release ZIP includes the conda artifact, installer, instructions, and
checksums; it has not been uploaded to a public package registry.
