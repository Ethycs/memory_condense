# Bounded engineering session memory evaluation

Status: stopped incomplete on 2026-09-22; first coding task did not progress. Both workers closed.

The user clarified that the target is completing a real engineering session with
memory replacing accumulated context. The prior eight-prompt replay retained an
unbounded current-turn tool history and never ingested those observations. Its
coding result therefore did not establish this stronger target.

## Frozen successor scope

Recorded continuation root: `eval_results/native-spine-engineering-session-20260917-r4`.
Plan SHA-256: `9855f565f5cda4e8fa337400ea48fe663dde5b1be06e5b885edfb8ebbfbd3063`.

Use the same eight original user prompts, 214-message historical seed, and fresh
checkout at `03105c423bf19b7a2b03f29a6b55e527f17ab2db`. No original future assistant
solutions or the previous candidate patch enter the coding agent's checkout.
This is one feature episode, not a complete-project or 1M-token coding benchmark.
Pre-episode tool dumps remain absent from the historical seed; all newly generated
actions and observations are captured without truncating their stored text.

Before every model action, new events enter ordinary `MemoryCondenser.ingest_many`,
compiled atomic summaries, the user-spine attention hierarchy, and persisted native
and parent summary indexes. The application closes and reopens before retrieval.
Exact selected raw sections are hydrated only after summary routing. Assistant and
tool observations are eligible evidence, without the QA-only user-role projection.

The active continuation has a hard 24,576-token local prompt estimate limit. It contains
system/tool instructions, the current user prompt, up to 3,072 tokens of semantic
memory, up to 2,048 tokens from the most recent eight user leads, up to 1,024 tokens
from the last completed assistant reply, and a 12,288-token preview of the latest tool
observation, and up to 1,536 tokens from six recent work receipts. The recent user
leads, last reply, and work receipts are recovered through exact memory
hydration. Earlier raw tool history is never appended as a growing chat transcript.
The complete latest observation is stored before its preview is sent. The agent can
request additional memory retrieval or read further pages from the working files.

The fixed actor limit is 80 responses per user prompt. A response can batch up to
eight bounded file tools. File changes and offline tests run under the ordinary
sandbox. A separate authorized gateway worker performs generation only and cannot
execute model-produced code. Its Qwen requests are reconstructed from typed summary
requests before dispatch. Raw summarization and coding use Sol. Qwen receives only
summaries. Existing content-addressed summary/attention caches may be reused.

FP32 BGE embeddings use the GPU; verified weights move to CPU before any subsequent
summary generation or Qwen attention stage. Local Qwen attention is released before
gateway summary merging. Explicit recall returns at most 2,304 tokens of semantic
evidence without repeating the already supplied reserved user/reply blocks.
Per-action ingestion, reopened retrieval,
and model generation have separate measurements; short-answer QA latency does not
describe this coding workload.

## Verification and grading

Seventeen offline harness boundary checks passed before r3 preparation, covering request
budgets, exact retention of a user correction despite 100 intervening tool events,
Unicode preview bounds, real file search, path confinement, role separation, and the
existing chronology/Qwen boundaries. The additional checks verify that work receipts
contain actual observed outcomes without code payloads, and are hydrated exactly
from stored memory.

The preliminary r1 run is preserved, including its frozen controller. CPU seed
ingestion took 591.6 seconds; its first retrieval took 1.013 seconds and the first
request used 4,739 tokens. It performed one memory lookup and no candidate code
changes. It was stopped before implementation to stage GPU residency and avoid
repeating reserved context in explicit recall. Its completed generation requests
remain recorded; both r1 workers have exited. No engineering outcome is claimed
for that preliminary diagnostic.

The r2 run used staged GPU embeddings and reduced initial ingestion to 27.27 seconds.
It completed the first two prompts but repeated memory lookups and decay-file reads
during the third prompt. The run was stopped after twelve tool actions in that
prompt, before any code change, instead of consuming the full 80-response ceiling.
Its plan, frozen controller, requests, responses, observations and stop diagnosis
remain under the r2 root. Both r2 workers have exited. It did not complete the
engineering episode and is retained as a negative working-memory result.

The r3 successor adds a deterministic compact receipt after each executed action.
Receipts record the actual commands, file pages, observed status and source event
IDs/hashes. They contain no hand-authored solution or acceptance hints. Each receipt
is itself ingested, summarized, indexed and recovered through exact memory hydration.
The system instructions tell the actor to use these receipts to retain progress and
to match discussion versus implementation scope. This is a development repair, not
a single-variable causal comparison. Every model request remains capped at 12,288
tokens; the semantic evidence allocation was reduced to accommodate recent work.

The r3 run completed prompts 1–3, getting past the discussion loop, and began the
implementation prompt. File-read batches returned 6,517–10,312 tokens, but the
3,072-token immediate-observation preview repeatedly hid requested files. The agent
alternated read batches without editing. Recent work receipts were actually present
in the served packets (1,098–1,498 tokens in the inspected coding actions), so this
was not failure to persist/retrieve those receipts. The current-tool allocation was
too small for the requested multi-file reads.

At prompt index 3, action 7, r3 also stopped on strict raw-summary support quotation
validation after three recorded attempts. The r4 parser permits only mechanically
escaped or outer-quote-corrected model-selected support that literally occurs in
the same source fragment. Unsupported selections can be dropped only when exact
model-selected support remains. It cannot invent replacement quotations; summary
text is unchanged. All three failed source responses passed the original strict
validator after this repair. Each accepted repair is separately journaled.

The r4 continuation retains the three completed prompts, every generated event,
every completed action and gateway record from r3. It does not ask those user prompts
again. The candidate checkout still exactly matched its starting files. The new
24,576-token total cap provides a 12,288-token current-observation slot; earlier
history still enters through bounded memory retrieval. Nineteen offline boundary
checks passed before continuation. R3's frozen controller and original artifacts
remain intact, and both r3 workers have exited. This is a documented development
continuation with a changed working-context allocation, not an unchanged-policy
benchmark result.

During r4, a separately bound adapter restored the historical replay reader's
240-line limit (the new harness had clamped requested 190–220-line reads to 120).
The reader now reports the exact returned line range. `read-tool-adapter.json`
records the adapter's hash, updated system description and the event prefix at
activation; each subsequent answer request binds that artifact. The frozen base
controller, completed actions, memory contents, context budget and hidden checks
are unchanged. Twenty harness checks passed before resuming with this adapter.
Controller entry point: `tools/native_spine_engineering_read240.py`.

The prior independent behavior checks were copied and hashed before any generation.
They remain outside the agent's accessible checkout. Regression checks and those
behavior checks will evaluate the candidate as generated. Historical API checks are
a separate diagnostic because naming/default differences can fail them without
establishing a behavioral defect. These checks were informed by the earlier replay;
they are a development evaluation, not a new blind benchmark.

The final auditor reconstructs each request's exact chronological memory prefix,
checks that all previous generated action/tool pairs were ingested, verifies the
fixed request cap, and reconstructs each hydrated span from the permitted raw event.
No full-context model control or another 100-question campaign is started.

Implementation: `tools/native_spine_engineering_session.py`.
Audit/report: `tools/report_native_spine_engineering_session.py`.
Boundary checks: `tests/test_native_spine_engineering_session.py`.

## September 22 continuation

The prior process stopped during raw summarization at prompt index 3, action 18,
with an acknowledged gateway InternalServerError. It had completed the first three
prompts and 18 coding-prompt actions without changing the candidate. No coding
answer was in flight or lost. A sealed reconciliation receipt records the checkpoint.

The latest context adapter preserves one actual assistant action/tool-observation
pair in normal chat roles. Older actions still enter only through bounded memory.
It first served action 16. Action 15 had already been acknowledged by the gateway;
it was recovered and its read-only tools executed once, without another model call.
The separately bound tool-cycle adapter preserves the frozen base controller.

Resume uses `tools/native_spine_engineering_transport_resume.py`. It permits at most
two retries of an acknowledged server or rate-limit error. Each retry retains the
original failed response and binds the identical generation payload under a new
request identity. Successes and unacknowledged requests are not resent. The actor
context policy is unchanged. Twenty-five offline boundary checks passed before
resuming. This remains an in-progress development evaluation, with no successful
engineering outcome claimed.

The server retry succeeded. Action 18 was another read-only batch, leaving the
candidate unchanged. The controller was stopped before publishing action 19; the
gateway finished its current summary response and exited. All 70 generated events
and all acknowledged requests remain. The checkpoint had no pending actor response
or partial action artifact.

A declared working-state continuation now requests a short actor-authored note
inside each action, retained in the bounded immediate assistant/tool pair and also
ingested with that action. The note contains conclusions and next steps, not a
human-supplied solution. Current tool observations are displayed after decoding the
known JSON wrappers; stored observations and source text are unchanged. Three
sample batches lost 16–17% of tokens to those wrappers. The overall 24,576-token
cap and 12,288-token tool preview remain fixed. This addresses action continuity and
presentation together; it is not an attention ablation or a controlled comparison.
Three additional offline checks verify exact code/literal-escape display and note
retention within the cap. Entry point: `tools/native_spine_engineering_working_state.py`.

## Working-evidence routing diagnosis

The readable/tool-note continuation also repeated reads through coding action 22.
Its audited partial report records three completed prompts out of eight, 28 model
responses (23 in the coding prompt), 334,248 stored raw tokens and zero changed
candidate files. It is a negative incomplete engineering result. The report,
requests, observations, and empty candidate patch remain preserved.

Inspection found a concrete packet-construction fault. Direct summary ranking
included recent code observations, but `protected_direct=0` let hierarchy context
precede them. Older conversation filled the semantic budget, while recent user
instructions and activity metadata were also supplied in separate reservations.
The current user correction was present; newer working evidence was crowded out.
An offline comparison on action 22 reordered only the existing summary-selected
addresses and recovered newer code at the same 3,072-token budget, with zero API
calls. The comparison is retained as `working-retrieval-offline-comparison.json`.

The successor serving policy prioritizes direct matches, omits sections already
supplied and action/activity metadata, and defers the current live observation
without dropping it (its preview may be incomplete). Semantic evidence receives
6,144 tokens within the unchanged 24,576-token total cap. All routes still refer to
exact original sections selected through summaries. Routing uses summary addresses;
hydration reconstructs the selected raw text. `section_working_context.py` contains the reusable ordering policy;
`native_spine_engineering_working_retrieval.py` applies it to this episode. Thirty-two
offline checks passed. The live continuation is explicitly capped at five new
model actions before checking for actual edits or test progress. It starts at
action 23 and retains every earlier completed action; no user prompt is repeated.

## Stopped five-action result

The five-action gate was reached automatically before another ingestion or coding
request. All five responses were read/search batches: 34 reads and six searches,
zero edits and zero test runs. The episode remains at three of eight completed
prompts. The normal controller exited at its declared gate and the gateway closed;
no requests are unacknowledged. No further live expansion is justified by this result.

Across the retained episode the final audit verified 132 packets and 887 exact raw
spans for 33 model responses. The candidate still matches its starting checkout.
Stored history is 419,436 tokens, largely repeated tool reads; it is not evidence of
successful engineering over a 1M-token history. The final three generated events
are retained in the event journal but were not ingested after the stopping gate;
every actual answer used its complete preceding ingested prefix.

The five-action phase used 88 new generation calls (five actor responses, 22 raw
summarizations and 61 Qwen summary merges). Mean ingestion/publication was 152.88 s
per action, retrieval 1.84 s, generation 19.70 s, and answer input 20,107 tokens.
This synchronous per-tool ingestion design does not meet chat-like interaction
latency. Thirty-two offline checks passed; engineering acceptance checks were not
run because no code was produced. Do not conflate harness checks with coding success.

The routing repair is verified, but it did not resolve the read loop. The next
working hypothesis is that the actor needs a realistic bounded working context
and incremental ingestion. The one-pair live context and close/reopen after every
tool batch were harness choices, not user requirements. This observation does not
prove that Qwen attention or the answer model is intrinsically unsuitable. A new
controlled continuation should address the working-context lifecycle, rather than
spend more calls on this unchanged loop.

Readable result: `eval_results/native-spine-engineering-session-20260917-r4/working-retrieval-gate-result.md`.
Full audited result: `working-retrieval-checkpoint-report.json` in that directory,
SHA-256 `b4ab64ac5b7b638bbde2a477147f746258439ba3c3911a8d5ea16220c8ed908b`.
The earlier `partial-report.json` and `partial-candidate.patch` remain untouched.
