# Chronos temporal event lane integration plan

**Date:** 2026-10-16  
**Status:** Completed 500-question source-date replay: 59.8% to 66.2%; production timestamp boundary still needs repair

**Correction found during accuracy measurement:** native span `created_at` is
an archive export anchor in the resident/compiler path, not the original
message time. Rendering that value as `mentioned_at` is incorrect. The first
500-question replay scored 299/500, unchanged from control. A separate replay
bound dates to the original dataset turns and scored **331/500 (+6.4 percentage
points)**, without changing retrieved text or rebuilding histories. Do not
promote the export-anchor envelope as a source
date fix. The production timestamp boundary and legacy-store handling still
need correction; the event ledger remains unimplemented.

## Motivation

The completed 500-question LongMemEval-S run scored 299/500, but its failures
expose two defects that could affect the temporal result: recall used the
latest indexed event as the question-time anchor instead of the dataset's
actual `question_date`, and ordinary cap-eight memory envelopes omitted source
timestamps. Temporal accuracy fell from 71.7% when this substituted anchor was
within one day of the true question date to 23.3% when it differed by at least
one day. This association does not establish how much either repair will
improve accuracy; that requires a matched comparison.

Chronos supplies the most relevant published repair pattern: preserve raw
conversation turns while selectively extracting temporally grounded events
into a second calendar. Each event has a subject, verb and object; an absolute
or approximate datetime range; and lexical aliases. Query-time retrieval uses
both the event and turn calendars, including date filters and exact lexical
search. Chronos reports 92.6% with GPT-4o and 95.6% with Claude Opus 4.6 on
LongMemEval-S. Its 116-question ablation shows the event calendar is especially
important for the weaker reader, while dynamic prompting contributes no
overall gain with the stronger reader.

Sources:

- [Chronos paper and architecture](https://arxiv.org/html/2603.16862v1#S3)
- [Chronos event extraction](https://arxiv.org/html/2603.16862v1#S3.SS1)
- [Chronos retrieval and agent tools](https://arxiv.org/html/2603.16862v1#S3.SS3)
- [Chronos ablations](https://arxiv.org/html/2603.16862v1#S4.SS4)

## First implementation slice and artifact audit

The live `ChatSession` now forwards the triggering input's durable timestamp
to both resident recall paths, including recall against an older published
snapshot while ingestion is queued. Standalone native calls accept `asked_at`
and default to current UTC. The diagnostic subprocess binds its request and
cache identity to the same clock. LongMemEval callers explicitly supply the
dataset's question date. Missing benchmark dates fail before ingestion;
invalid explicit native dates fail rather than silently using the wall clock.

Every exact evidence envelope now includes its stored `mentioned_at`. The
packet also exposes the question-time header used by routing and records the
bound question in its retrieval receipt. Mention time is not an extracted
event date or a fact-validity interval.

Regression testing exposed an integration dependency: copied-recall summary
reuse previously accepted only undated envelopes. Its parser now authenticates
the new question clock against the recall receipt and the mention clock against
the sealed source span, while retaining support for legacy envelopes. Exact
recall copies continue to reuse their original summaries without generation.
Changed source text, dates, identities, or arbitrary added preambles fail reuse.

`tools/audit_chronos_packet_time.py` replays only metadata on the 500 saved r27
packets. It checks sealed artifacts and source-span hashes, uses explicit span
lengths so literal memory delimiters inside quotations remain safe, and writes
a separate report plus three example packets. No histories, summaries or
embeddings were rebuilt, and no reader or judge was called.

| Packet audit | Saved r27 | Dated replay |
| --- | ---: | ---: |
| Questions | 500 | 500 |
| Exact evidence spans | 9,772 | 9,772, unchanged |
| Spans displaying source timestamps | 79 | 9,772 |
| Total packet tokens (`cl100k_base`) | 1,224,212 | 1,446,958 |
| Average packet tokens | 2,448.4 | 2,893.9 |

The additional metadata costs **445.5 tokens per packet on average (+18.2%)**.
This measures packet text, not full prompt usage or latency. Existing candidate
selection and source bodies are unchanged in this audit. Corrected request-time
routing is implemented but was not rerun against these histories. This audit
alone produces no answer score; the completed accuracy comparisons are below.

Artifacts:

- `eval_results/chronos-time-packet-audit-20261016-r1/report.json`
- `eval_results/chronos-time-packet-audit-20261016-r1/example-000.json`
- original control: `eval_results/longmemeval-s-native-release-sol-20261014-r27`

The comparisons below isolate reader-only metadata changes on the frozen
packets. Corrected-clock rerouting over existing indexes and the event calendar
remain separate, unmeasured additions. None of these comparisons requires
regenerating the original histories or summaries.

Validation: **96 focused tests passed in 34.41 seconds** across request clocks,
chat/session lifecycle, batch queues, streaming, native hydration, resident
publication/reopen, continuity and summary reuse. Report:
`eval_results/chronos-time-validation-20261016-r1.xml`. Models were test doubles;
the resident fixture also stubs Torch's thread/cache configuration and fixes
the request clock for reproducible reopen comparisons. Fresh system-temp
directories avoided the unusually slow workspace test I/O.

An expanded continuity/decision-history/experimental run passed 98 tests and
failed one unrelated existing proxy CLI mock: `serve()` does not accept the
`locator` keyword introduced by the separate pending proxy changes. That
failure is outside this clock/envelope change; the wider suite is not claimed
green. A captured stack also confirmed slow PyTorch module-file imports during
an unrelated attention-hook test, independently of any model weight load.

Audit reproduction (uses the saved campaign and produces a separate output):

```powershell
.\.pixi\envs\dev\python.exe -m tools.audit_chronos_packet_time --source eval_results/longmemeval-s-native-release-sol-20261014-r27 --output eval_results/chronos-time-packet-audit-20261016-r1
```

### Accuracy measurement and timestamp provenance correction

The timestamp-only comparison uses **all 500 saved r27 packets**, preserving
successes as well as failures.
The control is the saved 299/500 result. Reader and judge remain
`codex_sdk/gpt-5.6-sol`, reasoning `none`, with the original prompts and output
limits. Exact duplicate predictions reuse their original verdict; changed
predictions are judged against the same reference and binary rubric. Report
paired gains, losses, accuracy-point change and uncertainty, including category
breakdowns. This comparison does not measure corrected-clock rerouting.

Runner: `tools/evaluate_chronos_packet_time.py`. Completed diagnostic artifacts:
`eval_results/chronos-time-accuracy-20261016-r2`. The sealed plan bounds the
population and concurrency to six requests. It starts from existing packets;
there are no new history ingestions, embeddings, or local model loads.

The first attempt stalled importing OpenAI SDK modules from disk. The replay
now uses the existing proxy HTTP/SSE decoder with identical wire request
settings; the transport contract and paired-score calculations have four
passing tests. Twelve completed canary answers from the first attempt were
adopted, including four recovered from saved responses; two interrupted
response-sidecar publications were checked against their already sealed SSE
streams. `adopted-canary.json` records the provenance. All twelve canary
predictions exactly match their controls. No canary answer was regenerated.

The completed export-anchor replay scored **299/500 (59.8%) versus 299/500**:
32 recoveries and 32 regressions. Temporal questions changed from 60/133 to
66/133, while knowledge-update questions fell from 57/78 to 45/78. Every
changed-answer control was regraded with the same current judge (186 calls);
four control verdicts changed, balancing to zero overall. This is no accuracy
improvement, and it is not a valid experiment on true mention timestamps.

Inspection traced the problem to `Compiler._atoms_batch`, subprocess `install`
and `ResidentNativeBackend._stamp`: they stamp live native turns with the
source export time even though the durable chat events retain their own dates.
The original metadata audit checked integrity against those stored spans; it
did not establish that their dates matched the source conversations.

`tools/evaluate_chronos_source_time.py` now verifies **every retrieved raw span**
against its original dataset turn: turn ID, source ID, role, whole-turn hash and
span hash. Only then does it annotate the envelope with the original session
date. This corrects **9,570/9,772** displayed dates across 499 packets. One
unchanged packet answer and all 186 completed control rechecks are reused.
The corrected artifacts are in
`eval_results/chronos-source-time-accuracy-20261016-r1`. Packet token counts
remain 1,446,958 total: +445.5 per packet versus baseline. Seven replay,
source-integrity, transport and paired-statistic tests passed in 1.99 seconds.

The corrected comparison completed all 500 questions. It changed no passage
membership, ordering or body text, and made no new summaries, embeddings or
local model calls. It does not measure corrected-clock routing, an event index,
or a production timestamp migration.

| All 500 questions, same Sol reader and rubric | Correct | Accuracy | Change from control |
| --- | ---: | ---: | ---: |
| Saved r27 control | 299 | 59.8% | — |
| Expose native export anchors as mention dates (rejected) | 299 | 59.8% | 0.0 pp |
| Attach verified original session dates | **331** | **66.2%** | **+6.4 pp** |

The corrected-date arm has **54 recoveries and 22 regressions**, a net 32
additional correct answers and **15.9% fewer errors**. The paired normal 95%
interval for the accuracy difference is **+3.03 to +9.77 percentage points**;
the exact two-sided McNemar p-value is **0.000313**. These statistics describe
this matched population, not an unseen-data replication or the complete Chronos
architecture.

| Category | Questions | Control correct | Correct dates | Change |
| --- | ---: | ---: | ---: | ---: |
| Temporal reasoning | 133 | 60 (45.1%) | **90 (67.7%)** | **+22.56 pp** |
| Knowledge update | 78 | 57 (73.1%) | 58 (74.4%) | +1.28 pp |
| Multi-session | 133 | 63 (47.4%) | 65 (48.9%) | +1.50 pp |
| Single-session user | 70 | 65 (92.9%) | 66 (94.3%) | +1.43 pp |
| Single-session assistant | 56 | 46 (82.1%) | 46 (82.1%) | 0.00 pp |
| Single-session preference | 30 | 8 (26.7%) | 6 (20.0%) | -6.67 pp |

Temporal reasoning supplies **30 of the 32 net recoveries**, with 34 gains and
four losses. Its paired 95% interval is +14.32 to +30.79 points, with exact
McNemar p = 6.04e-7. The other category changes are not individually conclusive;
the preference regressions remain in the result. Dates repair many elapsed-time
answers but do not by themselves solve changing-fact selection, aggregation or
the existing preference-answer contract.

All **179 changed predictions** have paired control checks: 133 reused checks
from the diagnostic arm and 46 new checks. The remaining 321 identical answers
share their original verdict in both arms. Four control verdicts differ from
the historical grades, balancing to zero within the multi-session category;
the audited totals, gain/loss counts and category scores above are unchanged.
Original reader controls were reused, not regenerated contemporaneously.

One reader request (ordinal 322) timed out after 120 seconds. Its request,
error and reservation were preserved, then one documented retry recovered it.
All 368 completed judgments were retained when resuming. The corrected campaign
made 500 reader requests including that failure, adopted one identical-prompt
answer, made 224 judge requests (178 new-answer grades and 46 control checks),
and finished with 500 sealed answers and judgments. See
`transport-recovery-001.json` and `completion-audit.json`. No end-to-end latency
improvement is claimed from this API replay.

Final corrected artifacts:

- `eval_results/chronos-source-time-accuracy-20261016-r1/report-final.json`
- `eval_results/chronos-source-time-accuracy-20261016-r1/report-control-rechecked.json`
- `eval_results/chronos-source-time-accuracy-20261016-r1/audit.json`
- `eval_results/chronos-source-time-accuracy-20261016-r1/completion-audit.json`

Reproduce the sealed reports without new model calls:

```powershell
.\.pixi\envs\dev\python.exe -m tools.evaluate_chronos_source_time report
.\.pixi\envs\dev\python.exe -m tools.regrade_chronos_control report --root eval_results/chronos-source-time-accuracy-20261016-r1
```

The production repair should use the already available committed `ChatEvent`
for each hydrated turn to supply mention time. Verify its role/source and exact
text against the native span before annotating it. Keep the existing storage
anchor separate from this semantic timestamp so old sealed indexes need not be
silently rewritten. Copied-recall summary reuse must authenticate the date
against that trusted journal binding too; merely trusting a copied envelope's
metadata would lose the current provenance guarantee. Test different original
dates across sources, cold reopen, stale published snapshots and recall reuse.
This is a metadata-boundary repair; no extra model is required.

## Design to add

### 1. Carry real question time through recall

Add an `asked_at` value to the recall boundary. Interactive recall defaults it
to the triggering input event's timestamp; benchmark callers supply the
dataset question date. Remove the resident backend's substitution of the
latest indexed event at 23:59.

Every hydrated evidence envelope exposes `mentioned_at`. Temporal event
evidence additionally exposes `event_start`, `event_end`, `time_precision`,
`valid_from` and `valid_until`. Mention time remains source metadata; event time
describes the world occurrence; validity describes when a changing assertion
was true.

### 2. Add a sparse, source-bound temporal event ledger

Use the existing ingestion/summarization work to emit a second artifact only
for dated occurrences and state transitions:

```text
event_id
subject, predicate, object
event_start, event_end, time_precision
mentioned_at, ingested_at
valid_from, valid_until
aliases[]
supersedes_event_id
source_span_ids[]
```

Event records are routing and computation artifacts, never answer authority.
Every event must bind to exact source spans. The reader continues to receive
the original hydrated text after selection. Extraction should share the
existing background ingest call where possible so the lane does not add a
second model load or blocking request.

Ambiguous expressions become ranges anchored to mention time. For example,
`last month` maps to the prior calendar month, while `recently` records an
explicit wider interval and its coarse precision. Updates retain both values:
the new event points to its predecessor and closes the predecessor's validity
interval.

### 3. Extend local tactic routing into a query plan

Reuse the resident BGE encoder and small classifier head to emit:

```text
route: turns | events | both
entities: [...]
time_window: [start, end] | none
operation: lookup | latest | before_after | count | duration | compare
evidence_requirement: one | all | sequence
```

Deterministic parsing handles explicit dates and simple relative expressions.
The frontier reader is the fallback for ambiguous plans. An always-on
query-time prompt-generation call is not justified because Chronos's stronger
reader received no aggregate benefit from dynamic prompting.

### 4. Retrieve both lanes and hydrate one source of truth

Run the current summary hierarchy/BGE/BM25/neighborhood path and event lookup
in parallel. Event lookup combines BGE or alias search with indexed datetime
filters. Merge candidates by source-bound identity, use the existing packer and
closure receipts, then hydrate exact raw spans.

The event lane adds operations the current hierarchy cannot express cleanly:

- retrieve all events within a date interval;
- order events by event time rather than mention order;
- follow a changing fact's supersession chain;
- distinguish a planned future event from a past occurrence; and
- expand an event hit to its source turn and adjacent conversational context.

### 5. Compute temporal and aggregation answers deterministically

Before reader synthesis, execute bounded operators over selected event rows:

- count distinct source-bound occurrences;
- select the latest valid value;
- calculate date differences and durations;
- filter before, after and within intervals;
- compare earlier and later states; and
- perform set union/intersection for multi-session enumeration.

Bind every computed result to the contributing source spans in the packet
receipt. This addresses a residual class Chronos itself reports: counting and
arithmetic errors after successful retrieval.

### 6. Make iterative retrieval conditional on closure

When the closure engine reports an unmet obligation after the initial packet,
allow one targeted second round: event search, exact lexical search, neighbor
expansion, or a bridge query containing a newly identified entity. If the
obligation remains unsatisfied, expose the insufficiency signal to the reader
instead of inviting fabrication. This recovers Chronos's useful agentic
behavior without an open-ended ReAct loop on every turn.

## Components to reuse

| Required function | Existing component |
| --- | --- |
| Local tactic classification | BGE tactic and history-intent heads |
| Semantic routing | Summary hierarchy and BGE index |
| Exact lexical retrieval | Additive BM25 route |
| Conversational expansion | Hierarchy neighborhoods and source ordering |
| Changing-fact lineage | Durable memory supersession links |
| Evidence completeness | Closure obligations, witnesses and receipts |
| Factual authority | Exact raw-section hydration |
| Learning | Successful-packet Hebbian feedback |
| Background work | Resident ingest queue and publication snapshots |

No external document or vector database is required for the first version.
The event ledger can use the existing persistence boundary and embedding
runtime.

## Implementation order and gates

1. **Request-clock implementation tested; mention-date boundary needs repair:**
   correct `asked_at` propagation is present. Rendering native `created_at`
   exposed export anchors rather than original message times; fix that boundary
   and authenticate legacy source dates before promoting dated envelopes.
2. Add event schema, source-span validation and deterministic date utilities.
3. Emit events from the existing ingest path without an additional blocking
   model call.
4. Add event/date/operation outputs to the local query planner.
5. Add parallel event retrieval, merge and exact hydration.
6. Add deterministic temporal and set operators.
7. Add closure-triggered one-round retry and abstention.

Each stage requires a provider-free source-integrity test. Accuracy promotion
requires a frozen matched LongMemEval-S comparison using the same reader,
question timestamps, prompts and judges, followed by LongMemEval-M or another
untouched population. Report accuracy, evidence coverage, packet tokens,
recall latency, ingestion lag and incremental storage. Preserve the current
route as the control throughout.

## Expected payoff

The primary targets are temporal reasoning and multi-session aggregation, the
two weak genuine categories in the latest run. Knowledge-update questions
should also benefit from validity intervals and explicit revision chains.
Preference failures in the current run mostly require a corrected answer and
judge contract; the temporal event lane is not their primary repair.

