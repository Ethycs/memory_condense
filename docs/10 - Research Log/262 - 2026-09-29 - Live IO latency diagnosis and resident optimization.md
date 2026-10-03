# Live IO latency diagnosis and resident optimization

**Status:** Streaming passes 24/24 replies; maximum memory lag three exchanges inside a twelve-exchange recent window. Mean reply 4.83 s; final drain 7.87 s. Earlier twelve-exchange sub-50 target remains unmet.
**Date:** 2026-09-30 (includes the twelve-exchange full-cycle optimization)
**Related:** [Chat IO regression, Log 261](261%20-%202026-09-29%20-%20Single%20million-token%20Chat%20IO%20regression.md)

## Finding

The previous five-second chat result did not include continuous native summary
and hierarchy refresh. Running that complete path on the existing 1,115,343-token
history revealed substantial live ingestion overhead. Cheap raw append/indexing
does not imply cheap summary compilation and snapshot publication.

The user stopped the clean re-ingest control in favor of optimization. It stopped
after 4,352 turns. **No completed million-token clean-versus-append equivalence
result exists.** The small fixture comparison does pass.

## Measured live path

The subprocess adapter launched fresh application processes for install, reopen
verification, recall, and Hebbian feedback. It repeatedly loaded model weights,
reconstructed summary indexes, validated the historical prefix, and serialized a
complete snapshot around each small update.

The replacement keeps one application, compiler, embedding model and attention
cache on ChatSession's existing single worker. It retains authenticated native
and parent indexes after durable publication. BGE and the Qwen prefix alternate
GPU residency; their verified weights remain in host RAM. Qwen still sees only
summaries. BGE summary embeddings remain FP32; the local six-layer Qwen prefix
remains FP16. Neither precision nor retrieval policy changed.

Every input, output, tool result, recall receipt and feedback event still enters
the unified journal and full ingestion path. Recall waits for the captured
prefix. Source pointers and Hebbian access IDs are preserved. Initial admission
and restart validate persisted indexes; warm operations use the admitted objects.
A valid older snapshot can be rebuilt after interruption left raw ingestion ahead
of native publication. Corrupted or mismatched persisted pairs fail closed.

| Measurement | Subprocess adapter | Resident adapter |
|---|---:|---:|
| First exchange, complete answer | 205.01 s | 78.64 s |
| First exchange, indexing/learning drained | 366.83 s | 116.68 s |
| Follow-up, complete answer | 103.43 s | 31.72 s |
| Follow-up, indexing/learning drained | 265.77 s | 79.27 s |

These are two analogous synthetic continuations on copies of the same real
million-token history, not an accuracy benchmark or latency distribution. New
deployment facts and summaries were generated. The optimized follow-up ran after
an unexpected worker exit; recovery took 41.97 s separately, then a different
fresh question was ingested for the measured exchange. The exit produced no Python
traceback and its cause remains undiagnosed. The replacement worker completed.
Do not describe this as an uninterrupted two-exchange reliability run.

The follow-up correctly recovered the requested cluster, the actual generated
migration ID, and the tool's check count. Its packet included all three original
event IDs. All 5,441 captured events were indexed, with no pending feedback or
reported errors at completion. Separate-process verification returned an identical
packet and identical native/parent receipts. Cold bootstrap was 25.07 s; restart
plus first query was 39.98 s. These setup costs are separate from the table.

The run used two reader calls, seven raw-summary calls, and four summary-only
merge calls. It reused historical compilation and performed **zero full-history
re-ingestions**. No claim about the previous 94/100 score changes follows from
this bounded probe.

## Remaining costs and additional changes

Resident recall itself measured 0.21–0.28 s; Hebbian feedback took 0.02–0.19 s.
The measured full refresh still spent roughly 6.2–7.0 s publishing complete
indexes, plus raw-summary and hierarchical-merge calls. Summarizing copied recall
packets adds work. Ordinary warm raw ingestion was around 0.25–1.69 s in the
follow-up; cold model setup was much larger.

The generation worker also serialized background summarization and reader calls.
After the live probe, it was changed to allow one reader call and one compiler
call concurrently, with reservation/budget checks owned by a single dispatcher.
A blocking-provider test proves the reader can finish while a summary is pending
and that shutdown retains the in-flight response. No additional live generation
was run after this change, so its latency benefit is **not** included in the table.

Additional serialization changes use direct runtime mapping checks, fast handling
of JSON scalars, reuse an existing immutable leaf-only lexical index, and serialize
each publication payload once for both hashing and storage. After these changes,
CPU-only republication of the completed 5,441-turn store took 5.35 s and produced
identical native and parent receipts. This measures publication only, with loaded
indexes; it is not end-to-end ingestion timing.

The next substantial optimization is incremental snapshot publication and avoiding
new raw-summary generation for recall text whose original summaries are already
available through authenticated source pointers. Fresh raw-summary latency must
also fall before synchronous full ingestion can meet five seconds. The current
implementation is faster, but is not ready to claim that target.

## Verification and artifacts

The final targeted run had **90 passing tests and one pre-existing failure**.
The failure is
`test_discourse_store.py::test_coverage_finalization_blocks_cross_connection_source_toctou`.
It reproduces against an isolated archive of untouched HEAD sources and the HEAD
test. Its injected `coverage_for_chunks` hook is no longer called by the current
finalization implementation. This unrelated test was not changed.

New tests cover resident-versus-clean snapshot equality, identical packets after
cold reopen, append-only admission, model/application instance reuse, absence of
warm disk reconstruction, failed-publication freshness guards and restart repair,
and independent reader/summary scheduling.

- Baseline live measurements and canceled control:
  `eval_results/chat-io-live-20260929-r1/`
- Optimized live results and restart checks:
  `eval_results/chat-io-optimization-20260929-r1/report.json`
- Recovery detail: `recovery.json`, `resume-probe.log`, `resume_probe.py` in that run.
- Post-probe publication equivalence: `serialization-check.json`.
- Baseline reproduction: `head-baseline-test.log`, `check_baseline.py`.
- New implementation: `tools/engineering_research_resident.py`; live `open_chat`
  now selects it. `ProcessNativeBackend` retains the old diagnostic control.

All probe and gateway workers were closed. No full-context generation, new large
battery, model-precision change, or historical accuracy rerun was performed.

## Incremental indexes and persistent model placement

Follow-up work now updates the live index and its durable section rows instead
of reconstructing the historical postings and rewriting both snapshot files.
`SectionSummaryIndex.updated` reuses unchanged term postings and canonical
section serialization. Postings use stable section IDs, so a new section sorting
before an existing section does not renumber its entries. Corpus statistics and
hierarchy validation still cover the complete new population. Native v1 receipt
hashes remain identical to a full rebuild; this is not an O(1) whole pipeline.

Parent projection v2 binds each address to its original root receipt, instead of
including the whole hierarchy hash in every parent ID. Unchanged parent vectors
are reused by exact original-root identity, including during migration from the
old snapshot. Old parent files and their readers remain supported. New parent
receipts intentionally use the v2 format.

The resident adapter publishes through `install_native_spine_incremental` into
`native-spine-live-v1.sqlite`. SQLite WAL transactions write only changed section
and vector rows and commit native/parent manifests together. Initial migration
populates this store once and preserves the legacy files. Cold admission checks
every stored row, vector, full manifest, and original transcript binding. Existing
live stores must continue using incremental publication; legacy whole-snapshot
writers reject them rather than leaving two competing current snapshots.

One saved 1,125,710-token population advanced from 5,437 to 5,438 turns:

| Component | Measured result |
|---|---:|
| Full construction of the two combined indexes | 2.377 s |
| Incremental update of those same indexes | 0.264 s |
| Complete cached compilation, including that update | 0.312 s |
| Stable parent projection update | 0.040 s |
| Incremental native + parent publication | 2.556 s |
| Legacy native-only full publication control | 5.078 s |

The warm commit upserted one atomic section, two hierarchy sections, and one
parent, and deleted the superseded parent. The native receipt matched the legacy
full build. The complete new manifest matched a freshly constructed section
store, and five matched routing probes returned identical route receipts and
source addresses. These probes used representative stored summary vectors to
isolate implementation parity; they are not fresh user-question accuracy scores.
A separate process reconstructed identical native and parent receipts. This was
cached compilation/publication, with no generation calls, vector recomputation,
or raw-history re-ingestion. It does not measure a new end-to-end answer latency.

The model placement test keeps the Qwen vocabulary embedding table in CPU RAM
and transfers only selected token vectors. Qwen's six FP16 transformer layers
and BGE's FP32 weights stay on the GPU. Models load and verify during startup;
the adapter uses the prior staged policy if startup GPU headroom is insufficient.
The attention execution/cache identity binds host-embedding placement.

On the local RTX 2070 SUPER, the warm test measured 0.079 s for attention and
0.046 s for the accompanying summary embedding batch, with no whole-model moves.
The first retained attention pass took 0.349 s. Both passes matched the staged
attention scores within 1e-6 and returned byte-identical BGE vectors. Peak Torch
allocation was 4.630 GB, with 2.757 GB reported free by CUDA after the pass.
Model precision was unchanged. These bounded passes do not establish a p95
latency or memory guarantee for every future desktop workload.

Validation now includes 107 distinct focused passing tests across the relevant
index, publication, routing, model, and chat modules. New cases cover changed-row
writes, stale-writer rejection, transaction rollback, raw/summary/vector/manifest
corruption, parent-ID stability, both historical facades, and fresh-build/cold
reopen equality. The earlier unrelated discourse-store failure was not retested.

Artifacts:

- `eval_results/incremental-native-update-20260929-r2/report.json`
- `eval_results/incremental-native-update-20260929-r2/restart.json`
- `eval_results/resident-model-placement-20260929-r2/report.json`
- Reproduction: `tools/evaluate_incremental_native_update.py` and
  `tools/probe_resident_model_placement.py` (use fresh result directories).

Whole-transcript validation, matrix assembly, and complete content-hash scanning
still contribute to publication. Fresh raw-summary generation is also unchanged
by these two optimizations. A full warm interactive run is needed before quoting
a replacement for the earlier 31.72-second answer time.

### Complete-turn integration check

`tools/evaluate_chat_io_complete_turn.py` prepares one fresh follow-up on a copy
of the existing 5,441-event memory. It measures the ordinary ChatIO exchange and
then flushes recall receipts, the assistant output, feedback, and learning.
Historical raw summaries are reused; startup warmup prohibits gateway calls.

The first startup exposed a sealed-cache integration error: the resident adapter
tried to replace `method.json` both after implementation changes and after GPU
placement selection. Resident attention now uses content-addressed method files,
preserving existing records and binding the selected placement. Fourteen focused
tests pass, including reopening the same cache under both placement choices.

The corrected `chat-io-complete-turn-20260929-r2` completed local admission and
warmup in 56.192 s, including initial migration. Both models selected retained GPU
placement. All 5,441 events were indexed with no pending feedback or errors.
No generation calls or history re-ingestion occurred during startup. Following
explicit user approval of the gateway and payload, the prepared worker completed
one measured turn. The approval wait was outside the timing window.

| Measurement | Previous resident follow-up | Complete-turn integration |
|---|---:|---:|
| Input preparation | 26.149 s | 12.701 s |
| Recall | 0.207 s | 0.324 s |
| Answer generation | 4.783 s | 3.103 s |
| Answer delivered | 31.720 s | 16.782 s |
| Full cycle, including answer and learning | 79.273 s | 49.410 s |
| Input-side Other compilation | 11.159 s | 2.134 s |
| Input-side publication | 6.185 s | 2.441 s |

The new run began with 1,127,820 tokens and 5,441 events, and finished with
1,130,052 tokens and 5,445 events. The answer exactly matched the expected
cluster, migration identifier, and tool check count. The packet included the
original input, assistant output, and tool result. Input, recall receipt,
assistant output, and feedback were all indexed, Hebbian learning completed,
and no events, feedback, or errors remained pending. The original store's files
were unchanged. This is one synthetic follow-up on a saved real-size memory,
not a new accuracy campaign or an identical-request latency control.

Three raw-summary calls took 26.374 s in total; two summary-only Qwen merge calls
took 4.567 s. Other compilation across all three refreshes took 4.129 s, and
publication took 7.432 s. The reader overlaps background processing, so its
duration must not be added again to the full-cycle components. After the visible
answer, finishing ingestion and learning took another 32.629 s.

For the input refresh, local compilation outside the atom-summary stage took
0.723 s, including 0.471 s of attention and a 0.210 s combined-index update.
The broader 2.134 s Other compilation remainder also includes summary handling
and gateway coordination. Thus the earlier roughly 0.4 s component estimate
was not the measured end-to-end remainder. There were no model reloads or
whole-model transfers during the turn; Qwen processed summaries only.

Artifacts: `eval_results/chat-io-complete-turn-20260929-r2/report.json`,
`exchange.json`, `bootstrap.json`, and `worker-wait.json`. Both workers exited
successfully after this single turn. The result improves visible answer latency
by 47% and full-cycle latency by 38% relative to the earlier analogous probe;
it does not yet meet the five-second interactive target.

### Full-cycle diagnosis

The three serial memory refreshes were input (12.701 s), recalled packet
(22.560 s), and assistant output plus feedback (13.278 s). Recall took 0.324 s
and the Hebbian update 0.133 s; journal/coordination overhead accounts for the
remaining wall time. Answer generation overlapped the recalled-packet refresh.
The visible answer arrived at 16.782 s, leaving 32.629 s of pending work.

The 7,657-character recalled packet was fragmented and summarized as new system
text, costing 15.168 s in Sol generation alone. Its raw/chunk ingestion took
3.792 s and publication 2.403 s. The short structured success receipt required
another 4.117 s Sol call. The assistant's exact answer already had a raw-summary
cache entry in the source run, which this turn reused. A novel answer's summary
generation cost was therefore not measured here.

Across the full cycle: raw summaries 26.374 s, Qwen summary merges 4.567 s,
other compilation 4.129 s, raw/chunk ingestion 4.884 s, summary embeddings
0.578 s, publication 7.432 s, recall 0.324 s, Hebbian learning 0.133 s, and
coordination remainder 0.990 s. These sum to 49.410 s without double-counting
the overlapping reader. `full-cycle-breakdown.json` binds this accounting to
the saved report; this analysis made no new model calls.

The strongest next optimization candidate is reusing authenticated source
summaries for recalled copies and compiling known structured feedback directly,
inside the existing IO ingestion path. Exact span coverage and original pointers
must remain validated, and partial spans without reusable summaries must retain
the existing summarization fallback. This is a proposed optimization, not an
implemented or measured saving. Publication also still scans the complete
snapshot despite writing only changed rows. Learning itself is already cheap.

### Six-exchange recent context and ingestion queue

The live chat binding now defaults to `batch_exchanges=6`, counting six complete
user–assistant exchanges. Additional assistant/tool messages under the same user
input remain in that exchange. Every event is still committed immediately to the
chat journal, including recall receipts and success feedback. Expensive native
summary/hierarchy ingestion runs once per completed batch, with feedback captured
before the ordinary ChatIO exchange schedules its batch.

Recall still executes for every new question against the committed memory prefix.
The reader receives that evidence plus exact recent journal events: the last six
exchanges and the current unanswered input, including tool results. Uncommitted
unanswered inputs are retained even if repeated failures make the window larger.
Internal recall and feedback copies are excluded from the recent window to avoid
recursive growth. This is an exchange-count window, not a strict token cap.
`RecallPacket.recent_events` preserves the event/session addresses and captured
text; `context_text` composes archive evidence with that recent conversation.
Packet retries restore the originally delivered window verbatim. Recall-receipt
events store archive evidence plus pointers to recent events, without duplicating
the whole rolling window in each receipt.

An ingestion-state row records the committed prefix and an in-flight target.
Restart restores the committed archive without eagerly ingesting a partial queue;
an interrupted batch resumes its recorded target before new work. This covers a
native commit succeeding before the queue acknowledgement. Learning only runs
after the input, recalled-source references, and success receipt are indexed.
Explicit `flush()` and normal close force the remaining partial batch to commit.
In this initial version, batch failures remained visible and blocked archive
reads until recovery. The nonblocking revision below replaces that behavior for
the resident live backend.

The live JSONL binding and newly prepared engineering runs use six exchanges.
The low-level ChatSession default and the existing single-turn latency controls
retain eager ingestion (`batch_exchanges=0`); old sealed engineering plans retain
their earlier policy. CLI override: `--batch-exchanges 6` (default), or `0` for an
eager diagnostic. Reader adapters use `packet.context_text`; clients reading the
JSON packet directly must include its `recent_events` alongside `text`.

Validation: 47 focused tests pass across chat batching, the shared IO service,
native hydration/learning, the resident backend, seed reuse, gateway, and
engineering lifecycle. Twelve consecutive JSONL exchanges issue twelve recalls
and two ingestion batches after initial archive admission. Native-memory tests
verify current facts are visible before ingestion, all events commit together,
and the six deferred Hebbian updates reach original sources. Other cases cover
partial queues across restart, retries, tools, empty archives, failed batches,
lost commit acknowledgements, and migration of old packet journals.

No new model-backed latency or accuracy result is claimed for batching. The
backend still serializes ingestion and recall on its worker, so a question
arriving during a batch can wait for that batch to finish. Batch generation also
remains subject to the compiler's existing fragment and token limits; six
exchanges do not imply a single provider request.

### Nonblocking recall and short-text admission

The first model-backed twelve-exchange attempt is preserved at
`eval_results/chat-io-batch12-20260929-r2`. Its first six answers were correct,
but the first ingestion batch took 218.501 seconds and held up turn seven.
The user stopped this design and requested nonblocking ingestion plus a
summarization bypass for text shorter than 1.5 times the summarization limit.
This is a partial baseline, not a completed twelve-answer result.

The resident backend now publishes a complete recall view only after a
successful native-index commit. A separate reader uses that immutable routing
index, the matching event prefix, and its own read-only SQLite connection for
exact raw hydration. It does not read the writer's uncommitted transaction or
claim the newer transcript is already indexed. The existing eager diagnostic
API still rejects an advanced raw transcript without an updated index.

Initial archive admission remains a startup barrier. Subsequent compilation,
generation, publication and learning run on the writer while questions use the
last committed archive plus the durable recent tail. If a batch lags or fails,
all uncommitted messages remain in that tail even beyond six exchanges. This
preserves current facts at the cost of a temporarily larger answer prompt.
Batch errors remain visible in status. Explicit flush and close still drain
pending ingestion and learning; shutdown waits for active readers first.

BGE and Qwen retain their existing residency policy. A shared lock covers actual
model execution and staging, not remote generation or the whole compilation.
Queries can therefore experience brief GPU contention, but do not wait for an
entire ingestion batch. Low-memory staging parks Qwen before releasing that
lock so a concurrent reader can safely resume BGE.

The raw summarizer's hard output limit is 128 tokens. New raw fragments below
192 tokens retain their exact text, original span and content-bound receipt,
with an explicit `verbatim-short-v1` mode and no fabricated model request.
Exactly 192 tokens and longer still use the existing raw summarizer. Historical
cached receipts remain unchanged. Hierarchy merges retain their channel limits;
their typed inputs can now include short text verbatim. Accordingly, the new
run does not claim that all Qwen inputs are paraphrased summaries.

Validation: 52 focused tests pass. Tests hold ingestion open while later
questions complete, preserve an aged-out but uncommitted fact, read the prior
native snapshot during raw append, switch to the new snapshot after publication,
recover after a failed batch, and verify the strict 191/192-token boundary.
The existing fast-forward versus clean-build and cold-reopen comparisons also
pass. The revised model-backed run is
`eval_results/chat-io-batch12-20260929-r3`, using the original saved 1.130M-token
memory, twelve exchanges, two expected six-exchange batches, no inter-turn
flushes or delays, and a final drain. Post-drain hydration is checked separately
without another answer-model call.

The revised run completed with every lifecycle check passing. All twelve
answers were correct: four record acknowledgements and eight recall questions,
including a correction from Nimbus release r17 to r18. This is a bounded
continuation test, not a new broad accuracy benchmark. All 48 new journal events
were indexed in two batches of 24; all twelve successful recalls reached the
learning path. A separate local post-drain recall hydrated the original Larch
input and its exact cluster/rollback values. There were no pending events,
feedback or errors, and the original copied-from store was unchanged.

| Measured work, excluding startup | Time |
| --- | ---: |
| Twelve answers, first input to last answer | 65.929 s |
| Mean / median answer | 5.462 / 5.427 s |
| Maximum answer | 7.397 s |
| Raw-summary generation, six calls | 588.569 s |
| Hierarchy-summary generation, 27 calls | 65.403 s |
| Total summarization generation | 653.972 s |
| First / second ingestion batch | 309.521 / 411.750 s |
| Full cycle including ingestion and learning | 757.573 s |
| Drain remaining after the last answer | 691.611 s |

Startup was 41.026 s. The answer model averaged 4.693 s in generation, and mean
answer input was 3,703 tokens. Turns seven through twelve all finished while
the first batch was running; five began and ended wholly inside that batch,
while turn seven began just before the batch started. The archive grew from
5,445 events / 1,130,052 tokens to 5,493 events / 1,158,014 tokens. No historical
re-ingestion or new history construction was performed. Models were Sol for
answers/raw summaries and qwen3-8b for hierarchy merges; raw Sol calls explicitly
requested no reasoning. Both local models remained resident with no model reloads
during either measured batch.

The bypass created 34 distinct verbatim short-fragment receipts. Long evidence
packets still dominated raw summarization. Calls already return multiple
summaries: the first batch grouped its seven generated fragments as 3, 3 and 1
under the existing eight-fragment / 6,300-token packing limits. These calls run
sequentially on the compiler lane. The second batch also required three raw calls.
No background-latency improvement over the interrupted run is demonstrated here;
the first batch was slower despite skipping short-fragment generation.

The throughput criterion is not met: for all twelve exchanges, summary generation
was 9.92 times answer wall time. For just the first six exchanges, answers took
35.119 s versus 269.825 s of summary generation and 309.521 s of full ingestion.
Nonblocking recall removes that wait from the visible answer path, but the pending
tail grows under a conversation with no pauses. Sustained operation requires the
whole batch, including indexing and learning, to finish within the arrival time
of another six exchanges. User think time would extend that interval; this run
intentionally inserted none. Concurrent independent compilation requests and
reuse of authenticated summaries for recalled copies remain unimplemented
optimization candidates.

Artifacts: `report.json`, `latency-breakdown.json`, `timeline.json`, twelve
`turn-*.json` records and `post-drain-recall.json` under
`eval_results/chat-io-batch12-20260929-r3`. The interrupted run's
`stopped-run-assessment.json` records its incomplete baseline. Both model workers
have exited.

### Research: accelerating summarization

Research checked against primary documentation on 2026-09-29. This research
section audits the completed run and proposes experiments; the implementation
and evaluation follow-up below records the subsequently authorized changes.
The reproducible local evidence is saved in
`eval_results/chat-io-batch12-20260929-r3/summarization-research-audit.json`.

All six raw-generation requests contained recalled evidence copies. Their
thirteen fragments account for 20,592 proxy content tokens and all 588.569 s of
raw generation, approximately 90% of total summary generation time. New user
and assistant text in this particular test needed no expensive raw generation.
All 173 delivered reference occurrences match an exact atomic-summary span in
the **initial** archive, before these twelve exchanges. This includes full span
identity, source address and content hash, not approximate text similarity.

The leading change is therefore provenance-aware reuse during the same IO
admission path. Persist the delivered packet and its event/input/original-source
links, authenticate its source spans, and reuse their existing summary artifacts
for compilation. Preserve original speaker/time attribution and the distinction
between a recall copy and independent evidence. Newly clipped, modified, missing
or unsupported spans still need their own valid representation. Avoid expanding
every packet into another large collection of duplicate generated summaries;
retain compact receipt metadata and links to the existing semantic records.
This is a proposed representation change, requiring retrieval and learning
regression checks. Exact matches establish reuse eligibility, not a measured
latency saving or proof that the replacement representation is sufficient.

| Counterfactual for these twelve exchanges | Summary generation | Complete batch service |
| --- | ---: | ---: |
| Observed sequential compiler | 653.972 s | 721.271 s |
| Three simultaneous raw requests per batch, all other work unchanged | 377.047 s | 444.346 s |
| Remove all repeated raw generation, all other work unchanged | 65.403 s | 132.702 s |

These are arithmetic bounds under stated assumptions, not benchmarks. For
concurrency, the two raw phases become their slowest individual calls,
118.658 s and 192.985 s. This assumes capacity with no added contention and
retains the measured merge work. Reuse excludes replacement/validation overhead
and assumes unchanged merges, which may not hold. Even the reuse estimate leaves
indexing and merging above the 65.929 s answer interval. Raw/chunk ingestion
alone consumed 42.119 s; authenticated reuse of duplicate chunk work is another
candidate once the new representation is settled.

The current compiler loops over raw batches synchronously, and the gateway has
one compiler lane shared by raw and merge jobs. Increasing a thread count alone
would not remove both bottlenecks. A bounded pool of three independent jobs
needs matching gateway capacity and a reserved reader lane. Independent
exchanges and sibling hierarchy merges can also overlap; parents must await
their actual children, and identical request hashes should share one in-flight
result. OpenAI's [latency guidance](https://developers.openai.com/api/docs/guides/latency-optimization)
supports parallelizing independent operations, smaller models and reducing
generated output. [llama.cpp's server design](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README-dev.md)
describes serving multiple slots in one inference batch. Several summaries in
one JSON response still share one autoregressive output stream; that packing
does not itself create independent decoding streams.

| Candidate | Proposed role | Evidence and limitation |
| --- | --- | --- |
| LFM2-2.6B-Transcript, Q4_K_M resident runtime | First local candidate for new long raw text | A transcript specialist with an [official quantized release](https://huggingface.co/LiquidAI/LFM2-2.6B-Transcript-GGUF). Our older native run produced three valid cards in 6.507, 7.687 and 6.713 s, using about 5.22 GB peak allocated GPU memory. Those prompts were only 391–462 model tokens; these are not matched long-packet timings. |
| LFM2.5-1.2B-Instruct | Smaller CPU/quantized speed challenger | Its [model card](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) recommends extraction and RAG and supplies GGUF deployment options. The vendor's hardware-specific throughput figures do not predict this machine's performance. Engineering identifiers, corrections and code/tool observations need a matched quality check. |
| Qwen3-4B-Instruct-2507 | Optional replacement for hierarchy **summary merges** | The [official card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) specifies a 4B model with non-thinking-only output. This is a smaller generation candidate; it does not establish equivalent routing summaries or require changing the six-layer attention scorer. |
| GPT-5.4 nano, reasoning none, direct stateless API | Hosted speed/quality control | [Official documentation](https://developers.openai.com/api/docs/models/gpt-5.4-nano) identifies extraction/high-volume work and supports none as the default reasoning effort. Its availability through the approved gateway and performance on these packets have not been established. |

Liquid's [Transcript benchmark](https://www.liquid.ai/blog/the-future-of-meeting-summarization-local-fast-private-and-fully-secure)
reports 10K input / 1K output in 16 s using its AMD/llama.cpp setup. That is useful
motivation for a quantized-runtime test, not an RTX 2070 SUPER latency promise.
The original [local assay](105%20-%202026-09-05%20-%20Source-local%20contextual%20cards%20with%20LFM2%20Transcript.md)
already explains why its native FP16 memory usage differs from the optimized
vendor configuration. Keep the chosen summarizer resident; test quantized CPU
or separate serving capacity before adding a second full generator beside the
existing GPU-resident reader and attention models.

Two runtime optimizations are secondary. [Prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
can reuse common prompt processing, but it does not eliminate output decoding
or reuse the finished summary. Put stable instructions first and changing text
after them. [Speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
can accelerate token generation in suitable memory-bound workloads; gains depend
on model/runtime/hardware and traffic. Liquid's [296M DSpark drafter](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-DSpark)
reports roughly 2x decoding speedup for LFM2.5-1.2B in SGLang on tested hardware,
but requires a compatible build and another model. It is a later experiment,
after choosing and validating the target summarizer.

The present Sol route returned only 1,695 proxy output tokens across 588.569 s.
It reports zero usage, and raw calls were not streamed, so the artifacts cannot
separate adapter queue time, prefill, hidden reasoning and decoding. The outgoing
request explicitly used reasoning none; zero usage cannot prove how the route
executed it. A direct-endpoint control should log first-token time, output rate
and actual usage. Simply lowering the 4,096-token ceiling does not remove time
spent producing the already short observed responses. Prompt/format changes must
retain bounded exact support, source attribution, uncertainty and correction
handling.

Existing probes also narrow the work. The current Qwen no-thinking request form
returned four valid summaries averaging 2.402 s; alternate chat-template/JSON
parameter variants returned HTTP 500 rather than usable summaries. Repeating
those flags is not an established optimization. Local attention consumed only
1.408 s across both batches, so attention-head pruning cannot materially address
the hundreds of seconds spent in raw generation.

Recommended bounded evaluation order: first replay the twelve saved packets
through authenticated reuse and verify their source/learning receipts. Then
compare concurrency one versus three on the six existing raw requests and a
small fixed set of representative merge requests. Compare the winning serving
configuration with Transcript Q4_K_M and one smaller/direct-API control on those
same inputs, adding a few genuinely new long engineering turns to exercise the
fallback. Record end-to-end service time, first-token time, validation failures,
exact entities/numbers/status/corrections and unsupported assertions. Finally
repeat one twelve-exchange continuation against the existing memory. The goal
is complete six-exchange ingestion and learning within the next six exchanges'
arrival interval, with a bounded pending tail and unchanged recall behavior;
another population of million-token histories is unnecessary for this decision.

## Authorized follow-up: source reuse and bounded concurrency

After "Proceed in order", the compiler gained an authenticated recall-copy
path. It verifies every packet label, source-span receipt, exact copied-text
hash, delimiter and trailing byte against existing atomic summary records.
The raw recall event remains stored in full. Its compact observation descriptor
binds the existing source-summary receipts; facts continue to route through
those original summaries instead of another paraphrase of the same evidence.
This does not promote a recall copy into an independent user assertion.
Partial, altered or unavailable spans use ordinary compilation. Historical
cached summaries retain their identities. Earlier live sources are compiled
before dependent receipts, making cold and incremental compilation agree.

The saved replay at `eval_results/recall-summary-reuse-20260929-r2/report.json`
covers all 48 new events without any raw model call. Ten newly compiled packets
reuse 145 exact source references; two packets preserve historical cached
summaries with 28 references. The earlier r1 replay incorrectly expected all
twelve packets to take the new path and is retained as a failed assertion;
the corrected report explicitly distinguishes reuse from preserved caches.
This distinction does not change the zero-generation result.

Independent raw batches and independent user-led exchanges now run with at most
three workers. The gateway reserves a separate reader slot and still validates
and reserves budgets serially before dispatch. Identical merge keys share one
in-flight result. Transcript order is preserved, attached summaries await their
user spine, and parent hierarchy compilation remains dependency ordered.
The concurrency setting is recorded in the new evaluation plan.

A six-call summary-only probe compared the same three saved Qwen merge requests:
sequential service took **9.095 s**, concurrency three **7.100 s**; both produced
three structurally valid responses. This small probe does not prove semantic
equivalence. It is saved at `eval_results/summary-concurrency-20260929-r1`.
The expensive old six-request raw-generation comparison was unnecessary once
the saved replay showed that all six requests disappear under reuse. The local
candidate assay still uses their exact payloads to test genuinely required raw
generation separately. Regression tests exercise three concurrent raw batches,
ordered output, single-flight merges, and reader admission while all compiler
slots are blocked.

The combined live run `eval_results/chat-io-batch12-20260929-r4` reused the same
initial 1,130,052-token archive, twelve questions and reader model as r3. It
finished with 1,158,014 tokens and 5,493 events. Every lifecycle check passed:
12/12 answers, both 24-event batches, all twelve learning updates, exact new
original-source hydration after draining, no pending events or errors, and an
unchanged source store. Four answers began and ended entirely during ingestion.
These twelve synthetic continuation questions are a bounded regression, not a
replacement for the earlier hundred-question or engineering-quality benchmarks.

| Same twelve-exchange continuation, startup excluded | r3 baseline | Reuse + concurrency r4 |
| --- | ---: | ---: |
| Correct answers | 12/12 | 12/12 |
| Mean answer latency | 5.462 s | 5.908 s |
| Answer window through last response | 65.930 s | 71.264 s |
| Raw model calls | 6 | 0 |
| Raw model service | 588.569 s | 0 s |
| Merge model calls | 27 | 28 |
| Sum of merge call durations | 65.403 s | 96.520 s |
| Atom/exchange/hierarchy stage wall time | 668.824 s | 56.736 s |
| Raw/chunk indexing | 42.119 s | 26.230 s |
| Complete batch service | 721.271 s | 93.561 s |
| Full cycle including learning | 757.573 s | 125.261 s |

Full-cycle latency fell **83.47%**. The two new batches took 40.429 and
53.133 seconds. Merge call durations overlap under concurrency; their sum must
not be compared directly to elapsed answer time. Measured summary stages now
fit below the aggregate answer interval, but complete maintenance still exceeds
it. The remaining drain after the last answer was 53.884 seconds. Therefore
summary generation passes the aggregate `t_summary < t_answers` comparison on
this workload, while sustained ingestion-plus-learning throughput remains an
open optimization. No waiting between user turns was added to make it pass.

Model aliases and gateway capacity were unchanged. The reader averaged
4.950 seconds of provider time, compared with 4.693 seconds before, and one
merge call took 12.733 seconds. This is one measurement per configuration;
shared-provider variability and changed merge contents prevent attributing
every timing difference to concurrency. The original raw workload removal is
directly verified. `comparison.json` and `tools/assess_summary_acceleration.py`
record the timing boundaries and reproducible comparison.

## Resident small-model checks, in the requested order

Official Q4_K_M files for LFM2-2.6B-Transcript and LFM2.5-1.2B-Instruct were
downloaded into the workspace cache and checked against Hugging Face LFS SHA-256
values. The official llama.cpp Windows CPU build b11272 was checked against its
release digest. No installed application or shared gateway configuration was
changed. Both models ran in an isolated loopback server on CPU, with six threads,
so they did not compete for the existing Qwen/BGE GPU allocation. The server
stayed resident across each model's requests and was closed afterwards.

The initial Transcript test used three concurrent slots on the six saved raw
requests plus new engineering input. It was stopped as unpromising after the
first completed response took 181.070 seconds, returned four incorrectly
labelled atoms for one fragment, and invented support including a migration
completion. Shared CPU prefill also delayed a short request behind long ones.
Only one response completed; the others are incomplete, not scored failures
or a completed seven-request benchmark. The stop receipt and server trace are
under `eval_results/small-summarizer-20260929-r1/transcript`.

The bounded follow-up tested Transcript first, then the smaller model, on the
same four exact fragment texts: three new synthetic engineering observations
of 274–280 tokens, and one saved 2,048-token raw fragment. It used one CPU slot,
JSON schema constraints for cardinality and labels, and a 512-token output cap.
Transcript received its documented meeting format and recommended temperature
0.3; the smaller instruct model received the existing raw-summary prompt at
temperature zero. Therefore this is a practical candidate screen with disclosed
configuration differences, not a strict same-prompt model ablation. Transcript
format guidance is in its [official model card](https://huggingface.co/LiquidAI/LFM2-2.6B-Transcript).

| Resident CPU candidate | Valid after existing quote repair | Mean, three new fragments | Saved long fragment | Peak working set |
| --- | ---: | ---: | ---: | ---: |
| LFM2-2.6B-Transcript Q4_K_M | 3/4 | 18.232 s | 42.896 s | 3.04 GB |
| LFM2.5-1.2B-Instruct Q4_K_M | 4/4 | 4.519 s | 14.561 s | 1.48 GB |

Neither candidate is promoted. Transcript's long-fragment summary wrongly
assigns **boreal-mig to Arden**, even though the input explicitly pairs Arden's
saffron-842 cluster with **arden-mig**, and Boreal's cobalt-731 with boreal-mig.
This is a factual association error despite structural validation passing.
Its first new engineering fragment also fails bounded exact-support validation.
The current quote repair may discard unsupported quotations when an exact
model-selected quote survives; that does not certify every summary claim.

The smaller model returned generic labels such as "Pytest results summary".
Across the three new cases, only **1 of 15 explicitly checked identifiers and
numbers** appears in its routing summaries. Several missing details occur in
its support quotations, but the routing index embeds the summary text, making
those omissions relevant. It also drops the new release correction from the
first summary. This is a fidelity-screen result, not a measured loss of
downstream recall accuracy. Its shorter output partly explains the speed;
the timing is not evidence of equal quality at lower latency.

GPT-5.4 nano is not advertised by the authorized gateway, and no direct OpenAI
credential is configured, so the hosted control made zero generation calls.
The existing Sol raw-summary route and Qwen merge/attention models remain in
use. The verified runtime improvements are source reuse and bounded compiler
concurrency. Future small-model work would require better summary fidelity,
then a matched retrieval test, before changing the live route.

Artifacts and reproduction:

- `tools/replay_recall_summary_reuse.py`: saved-workload reuse and raw coverage.
- `tools/probe_summary_concurrency.py`: six-call summary-only concurrency probe.
- `tools/evaluate_chat_io_batch12.py`: the completed r4 live lifecycle test.
- `tools/fetch_summary_benchmark_assets.py`: pinned downloads with hash checks.
- `tools/probe_small_summarizers.py`: bounded resident candidate assays.
- `tools/assess_small_summarizers.py`: source-bound fidelity findings and timing.
- `eval_results/small-summarizer-20260929-r1/assessment.json`: completed model
  decisions, plus the explicitly stopped initial configuration.

Final focused validation: **92 tests passed** in 32.52 seconds, covering raw
summary bypass, authenticated reuse and fallback, clean/incremental equality,
parallel receipt order, three-slot scheduling, identical-request sharing, live
publication, chat queuing, exact hydration, original-source learning, and the
user-spine/parent hierarchy. `git diff --check` passed. All task-owned model and
gateway workers are closed; no additional million-token histories were built.

## Twelve-exchange sub-50-second attempt, 2026-09-30

The user confirmed that the target means all twelve exchanges, both six-exchange
ingestion batches, and all learning, and authorized testing a faster reader.
Every run below clones the same saved 1,130,052-token archive and adds only the
48 continuation events. None rebuilds a million-token dialogue. The final drain
is included; cold admission/model startup remains separately reported, as in r4.
No delays are inserted between questions. These are the existing four record
acknowledgements and eight recall questions, including the Nimbus correction.

Reader screening used three unchanged saved packets per route. Sol and Luna
passed 3/3, averaging 5.360 and 4.921 seconds respectively. The direct Haiku route
failed with an insufficient-credit response; both Phi aliases returned server
errors. Those are unavailable-provider results, not accuracy failures. Haiku
through `claude_code/claude-haiku-4-5` returned all three correct fact objects,
in 5.847, 2.520 and 4.230 seconds, but wrapped them in Markdown JSON fences.
The saved probes are `eval_results/sub50-reader-probe-20260930-r1` and `-r2`.
Error rows' placeholder durations in those probe reports are not measured
response latencies. The faster-reader choice is an evaluation override; the
default reader has not been globally switched on this small quality sample.

The initial live Haiku run r5 had 12/12 correct fact objects, but its original
strict JSON decoder counted only 2/12. That report is preserved unchanged.
`answer-format-assessment.json` records the separate format diagnosis. Later
runs accept JSON or exactly one complete Markdown JSON fence, rejecting extra
prose or multiple blocks. No answer value, source packet, or captured provider
output is edited. Fenced answers are counted separately. This is a disclosed
format accommodation, not recovery of missing or incorrect facts.

Retained implementation changes:

- Exact summary reuse can preserve role/date attribution explicitly. Only
  trusted outputs produced by that summarizer instance are flattened; arbitrary
  source text resembling the wrapper is never interpreted as provenance.
  Historical cached generations retain their receipts.
- Parent nodes retain a 512-token allowance for exact reuse. Fresh lossy merges
  produce at most 128 tokens, leaving room for subsequent exact concatenation.
  The first fresh merge uses the existing 48-word instruction, followed by a
  24-word repair if validation fails. This avoids spending a long generation
  merely to discover that a token-only instruction was ignored. Existing merge
  caches are read before applying the new generation policy.
- Raw ingestion and summary compilation overlap. Both must succeed before
  native publication or acknowledgement; a compiler failure leaves the previous
  published prefix available and the captured suffix recoverable.
- Independent hierarchy nodes compile with at most three workers after all
  attention cuts are fixed. A parent waits for its children and its attached
  channel waits for its user spine. Output order and receipt construction are
  independent of task completion order. Default library execution remains serial.
- Background embedding uses batches of eight and yields to waiting queries.
  A bounded 1,024-entry cache reuses vectors for identical text under the exact
  encoder identity, preserving every chunk's own source ID and coordinates.
  Returned vectors cannot mutate the cached values.
- Each published routing snapshot owns a read-only map of its committed, frozen
  raw turns. Exact hydration and span validation use that same authenticated
  prefix while a writer advances the database. This removed the observed long
  recall stalls during writes. Raw text is still hydrated only after routing;
  the snapshot does not become extra Qwen input.
- Gateway request/response polling intervals are shorter. Compiler capacity
  remains three slots with a reader slot reserved; no provider retries or budget
  bypass were introduced.

The r6 embedding change alone did not eliminate every stall: r7 still spent
10.650 seconds in one recall with only 0.036 seconds of aggregate query model-lock
wait. Moving published hydration off the mutable database was therefore a
separate correction, not evidence that all previous delays came from the GPU.

Storage experiments were not retained. Profiling only the 48 saved new events,
with their already computed vectors, found SQLite commit/checkpoint work
dominating raw indexing. Separating checkpoints retained `synchronous=FULL` and
waited for each complete checkpoint before acknowledgement. This follows
[SQLite's documented separate-checkpoint mechanism](https://www.sqlite.org/wal.html),
while retaining [full WAL commit synchronization](https://www.sqlite.org/pragma.html#pragma_synchronous).
The real r9 cycle nevertheless took 71.132 seconds: the first batch's raw append
took 12.837 seconds and its checkpoint another 12.550 seconds; compilation had
already finished in 10.098 seconds. The second checkpoint took 6.217 seconds.
There was insufficient overlap to improve elapsed time, so the runtime change
was removed. A 64 MiB SQLite page-cache probe likewise showed no useful gain
(8.871 and 3.847 seconds versus 8.645 and 3.952 in the original offline profile).
SQLite durability and automatic checkpoint defaults remain unchanged.

The profile artifacts are under `eval_results/sub50-raw-profile-20260930-r1`,
`-r2`, and `-r3`. These are diagnostic timings with profiler overhead and reused
embeddings, not substitute full-cycle results. `tools/profile_sub50_raw_ingest.py`
reproduces the bounded comparison without model calls.

### Final measurement and limits

| Configuration | Full cycle | Mean answer | Final drain | Correct facts | Merge calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| r4, previous Sol baseline | 125.261 s | 5.908 s | 53.884 s | 12/12 | 28 |
| r5, Haiku and overlap | 79.426 s | 5.043 s | 18.351 s | 12/12* | 16 |
| r6, flattened reuse and embedding scheduling | 76.328 s | 3.255 s | 36.850 s | 12/12 | 9 |
| r7, shorter generated parents | 72.613 s | 4.464 s | 18.517 s | 12/12 | 10 |
| r8, snapshot hydration and parallel hierarchy | 67.527 s | 3.494 s | 25.106 s | 12/12 | 8 |
| r9, concise first pass and checkpoint experiment | 71.132 s | 4.126 s | 21.147 s | 12/12 | 7 |
| **r10, retained configuration, normal checkpoints** | **64.539 s** | **3.951 s** | **16.608 s** | **12/12** | **7** |

*r5's original strict-format score remains 2/12; the factual-format distinction
is described above. Each row is one timing sample, not a controlled isolation
of every change. Reader route latency varied between runs, and earlier batch
publication changes subsequent recall packets and their recent-tail lengths.
All runs retain their original requests, responses, plans, source hashes, and
event journals. The final reader model differs from the r4 baseline.

The retained configuration reduces measured full-cycle latency by **48.48%**.
All twelve answers completed in 47.931 seconds. Mean answer time was 3.951
seconds and maximum 7.143 seconds; the slowest answer contained a 6.577-second
reader call. Thus this run also does not establish a five-second worst-case
answer guarantee. Startup took 41.344 seconds and is excluded from the 64.539
second full-cycle measurement, consistently with the earlier twelve-turn runs.

Both 24-event ingestion batches completed in 14.059 and 16.290 seconds. Across
both, summary compilation took 22.006 seconds, raw ingestion 12.574 seconds,
summary embedding/projection 2.471 seconds, and publication 5.119 seconds.
Compilation overlaps raw ingestion, so those figures are not additive wall time.
The final batch's critical path is approximately 11.994 seconds of compilation
(9.183 in hierarchy), 1.333 seconds of summary embedding/projection, and 2.704
seconds of publication, plus prefix/acknowledgement overhead. Its 6.133-second
raw ingestion fits inside compilation. The final run's remaining drain therefore
comes primarily from hierarchy compilation and publication, although the earlier
runs demonstrate substantial storage latency variability.

All lifecycle checks passed: 12 recalls, 12 learning updates, exactly two batches,
all 48 I/O events captured and indexed, original source pointers and exact new
facts hydrated after draining, no pending events/feedback/errors, and unchanged
source files. The final archive contains 5,493 events and 1,155,997 tokens. It is
slightly smaller than earlier variants because the first published batch becomes
available before the last questions, changing their delivered recall packets.
No captured I/O is dropped. Eight answers used the disclosed JSON-fence decoder.

The under-50 target remains **14.539 seconds away** on this final sample.
With the same 16.608-second drain, the twelve answers would have to finish in
33.392 seconds (about 2.78 seconds per exchange including overhead). With the
same answer window, only 2.069 seconds would remain for the complete final batch.
The next meaningful changes therefore concern the reader route and preparation
of the final batch; more polling tweaks will not close this measured gap.

This result supports faster live maintenance and a promising faster-reader
candidate on these exact questions. It does not establish unchanged broad
engineering quality or a new 95% accuracy result. More aggressive parent
compression also merits a larger recall evaluation before a general quality
claim. Sol remains the default reader outside this explicit evaluation override.

Final validation: **103 tests passed in 32.48 seconds**, including authenticated
recall-copy reuse/fallback, source-coordinate-preserving vector reuse, query
scheduling, historical merge preservation, parallel hierarchy receipt equality,
compilation failure recovery, cold reopen equality, snapshot hydration, chat
queuing, and original-source learning. `git diff --check` passed. Every task-owned
live/gateway worker completed and closed.

Final artifacts:

- `eval_results/chat-io-batch12-20260930-r10/report.json`: final timing and checks.
- `eval_results/sub50-cycle-20260930-final-assessment.json`: all measured variants,
  including the original strict r5 score and separately decoded fact accuracy.
- `tools/assess_sub50_cycle.py`: sealed-report comparison, no model calls.
- `tools/evaluate_chat_io_batch12.py`: prepare with a fresh `--root` and
  `--reader-model claude_code/claude-haiku-4-5`, then run its `live` and `gateway`
  phases against that same root. Existing run journals must not be overwritten.

## Last push: prepare summaries while the six-exchange queue fills

The next optimization moves reusable compilation into the answer interval.
`ChatSession` now accepts optional preparation thresholds, and the resident
six-exchange binding uses exchanges **three and five**. The same single writer
prepares summaries and attention/hierarchy caches for already durable, completed
journal prefixes. This does not append the raw memory store, publish an index,
acknowledge ingestion, or apply learning. All of those still occur at the
six-exchange boundary. Each question continues to recall the committed snapshot
plus its exact uncommitted conversation tail.

Preparation excludes the active exchange, including a captured assistant answer
whose feedback has not yet been saved. Duplicate queued requests reuse the
prepared prefix, and lagging work selects the latest eligible threshold. A
preparation failure leaves the journal and published memory intact; ordinary
sync/flush can recover it. Preparation caches survive through their existing
content-bound receipts, so restart does not require trusting an uncommitted
native index. Backends without preparation retain their existing behavior.

A related queue fix allows a completed six-exchange prefix to publish while
the next exchange is open. Previously the capture guard could defer that ready
batch until another narrow gap between answers. The unfinished exchange remains
excluded from both publication and preparation. Tests cover closed-prefix
boundaries, no early raw write/publication/learning, failure recovery, and
equality after reopening.

The first live preparation run, **r11**, finished in **52.274 seconds**, with
12/12 correct answers, seven merge calls, zero raw calls, both 24-event batches,
and all twelve learning updates. Four preparation jobs occupied 15.417 seconds
inside the measured cycle. Batch service fell from r10's 14.059/16.290 seconds
to 10.646/11.715; the final drain fell from 16.608 to 12.009 seconds. Mean answer
time was 3.303 seconds, maximum 4.544, and every lifecycle check passed. The
reader also ran faster than r10 (2.751 versus 3.431 seconds per call), so the
whole 12.265-second improvement cannot be attributed to preparation alone.

The subsequent token-count cache retains exact counts keyed by complete short
text and encoding name. Entries longer than 4,096 characters bypass it; token
arrays are not cached. Literal tokenizer control strings remain ordinary source
text. Source hashes, gap-free coverage checks, exact hydration, and all manifest
validation continue unchanged. The first 4,096-entry version, **r12**, completed
in 53.047 seconds with 12/12 correct answers and all checks passing. It recorded
11,016 cache hits but 17,386 misses; mean reader time rose to 2.840 seconds.

An offline replay on the same saved, authenticated snapshot compared capacities
without model calls. Increasing the bound from 4,096 to **8,192 entries** changed
publication from **3.028 to 1.990 seconds**. At 4,096 entries it had 4,319 hits
and 6,679 misses; at 8,192 it had 10,998 hits and zero misses. Both reconstructed
the exact same manifest. The larger bounded cache is retained. This microcheck
supports avoiding repeated tokenization, not an isolated claim about total chat
latency. Artifacts are `eval_results/publication-token-cache-20260930-r1` and
`tools/profile_publication_token_cache.py`.

The first 8,192-entry live run, **r13**, passed all correctness/lifecycle checks
but regressed to **62.642 seconds**. Publication improved to 1.553/1.014 seconds
and the cache recorded 26,738 hits versus 1,336 misses, while the first raw batch
took 16.584 seconds. That delay pushed preparation late: an 8.479-second partial
compile continued after the final batch became ready and held up its publication.
The faster validation was real, but this run did not meet the latency target.

The final scheduling correction makes preparation cooperatively yield to a
complete batch. Before each fresh raw-summary or merge generation, it checks
whether ingestion is ready (or the session is closing). It preserves completed
cached work and allows an existing provider call to finish without cancellation
or resubmission. A superseded preparation is not an ingestion error, and ordinary
sync then finishes the complete prefix. Tests verify that no new model call
starts after yielding, the captured queue remains intact, and normal sync resumes.

### Final measurements and storage diagnosis

All five last-push runs used the same twelve continuation questions, the same
saved 1.13M-token starting archive, and the authorized Haiku reader. Every run
scored **12/12** (eight recalls and four acknowledgements) and passed all
ingestion, original-source hydration, and learning checks. No history was
regenerated or re-ingested. These are individual timing samples with shared
provider variability, not repeated measurements establishing a latency guarantee.

| Run | Change tested | Full cycle | Mean answer | Final drain |
| --- | --- | ---: | ---: | ---: |
| r11 | Preparation at exchanges three and five | 52.274 s | 3.303 s | 12.009 s |
| r12 | Exact token cache, 4,096 entries; completed-prefix scheduling | 53.047 s | 3.393 s | 11.758 s |
| r13 | Exact token cache, 8,192 entries | 62.642 s | 3.518 s | 19.898 s |
| r14 | Preparation yields to ready full batches | 68.851 s | 3.327 s | 28.007 s |
| r15 | Same code; C: active memory and chat journal | 54.926 s | 3.680 s | 10.290 s |

r14 exposed storage as a separate bottleneck: raw indexing took **27.915 and
13.466 seconds**, while compilation took 4.230 and 11.880 seconds. Its first
batch spent much longer writing raw events than preparing summaries. A bounded
48-event replay with previously computed vectors reproduced expensive F: writes.
Changing checkpoint timing or using a 64 MiB SQLite page cache did not establish
a useful full-cycle improvement; neither change was retained. FULL synchronous
durability and the normal WAL checkpoint policy remain in place.

The same no-model raw replay in a fresh C: temporary directory took **1.680 and
1.019 seconds** for its two batches, including profiler overhead. It reused
vectors, so these are storage diagnostic timings, not live ingestion timings.
The full r15 experiment then copied the saved starting memory and journal to
`C:\Users\Keytone\AppData\Local\Temp\memory-condense-batch12-20260930-r15`.
Only the active memory and journal paths changed; generation artifacts and
compiler caches remained on F:. The original project store was unchanged.
`--runtime-root` makes this evaluation placement explicit in both plan and report.

With C: active storage, r15 raw ingestion took **4.106 and 3.269 seconds**.
Its two complete batches took **7.720 and 9.957 seconds**, including summary
compilation, embedding, publication, and acknowledgements. The second batch was
dominated by **7.302 seconds of compilation**, including 6.341 seconds in the
hierarchy, followed by 0.977 seconds of embedding/projection and 1.334 seconds
of publication. Raw ingestion overlapped compilation; those durations must not
be added as elapsed time. Four preparation jobs occupied 15.377 seconds during
the answer interval. The exact token cache recorded 26,872 hits and 1,426 misses.

All twelve answers finished in **44.635 seconds**. Mean answer latency was
**3.680 seconds**, maximum **4.951 seconds**. The reader itself averaged 3.063
seconds; mean delivered prompt length was 3,259 token proxies. Final ingestion
and learning drained for another **10.290 seconds**, making the complete cycle
**54.926 seconds**. Startup took 46.171 seconds and is excluded consistently
with the earlier cycle measurements. All twelve answers were below five seconds
in this sample; that does not establish a worst-case guarantee.

The latest full cycle is **56.15% shorter** than the 125.261-second baseline,
but that comparison includes a reader-model change and active storage placement.
The best observed full cycle remains r11's 52.274 seconds. **No measured run met
the under-50 target.** Final hierarchy generation remains the largest component
of r15's drain. Further progress needs a reduction in that dependent work or in
reader latency; repeating unchanged runs would not establish an improvement.

r15 saved all 48 new events, ending at **5,493 events and 1,154,493 tokens**.
It issued twelve reader calls, seven merge calls, and zero raw-summary calls.
Both batches and all twelve Hebbian updates completed, with no pending events,
feedback, or errors. These results remain a bounded continuation and do not
replace the broader recall or engineering-quality evaluations. Sol remains the
default reader outside the explicit evaluation override.

Final regression validation: **135 tests passed in 44.61 seconds**, covering
preparation, cooperative yielding, completed-prefix publication, durability and
recovery, exact token-count caching, authenticated reuse, resident retrieval,
hierarchy compilation, and original-source learning. The timing runs and their
gateway workers all completed and closed.

A separate process then reopened r15 with no model calls and read-only database
access. All 5,493 raw event IDs, roles, and texts matched the captured events;
the chat journal matched exactly; native and parent snapshot receipts matched
the final live recall. The committed position covered every event, all twelve
new feedback records were applied, and both databases passed their integrity
check. The closed C: runtime was copied to the run's `live` directory on F: for
archival after timing; all eleven file hashes matched. That artifact copy is
outside the measured cycle, whose active C: store was already durable.
`git diff --check` passed.

Last-push artifacts:

- `eval_results/chat-io-batch12-20260930-r11` through `r15`: immutable plans,
  requests, responses, event journals, timings, and checks for each variant.
- `eval_results/sub50-cycle-20260930-last-push-assessment.json`: comparison of all
  completed runs, including explicit runtime storage placement.
- `eval_results/sub50-raw-profile-20260930-r4-local-temp`: archived C: no-model
  storage diagnostic report and profiles.
- `tools/verify_chat_cycle_reopen.py`: model-free, read-only verification of the
  saved raw events, journal, snapshot receipts, and applied feedback.
- `eval_results/chat-io-batch12-20260930-r15/runtime-reopen-check.json` and
  `runtime-archive.json`: cold-reopen checks and byte-identical archive receipt.

## Streaming preparation with a twelve-exchange recent window

The next design separates recent context from ingestion scheduling. The resident
chat binding now defaults to streaming preparation and retains twelve completed
exchanges plus the current input. The context budget is 8,192 token proxies for
the rendered recent-event data. It trims whole committed exchanges first;
uncommitted source text remains visible even if that temporarily exceeds the
budget. This is a retention target, not a hard cap that silently discards input.
Explicit `batch_exchanges` selects the previous batch/eager path for comparisons.

After each exchange closes, up to three independent jobs prepare its atomic
and user-spine/attached-context summaries. Raw atom cache admission is serialized
to protect sealed files and their checksum sidecars; independent raw request
batches and exchange-summary generation retain their existing concurrency.
Workers own no application database and cannot publish or apply learning.
Preparation completions wake the single writer. It commits only the contiguous
ready prefix, coalescing already-complete results without waiting for a fixed
exchange count. A slow earlier job cannot be bypassed by a later result. There
are at most three admitted preparation jobs; excess input stays in the durable
journal rather than creating an unbounded executor queue.

The hierarchy is an incrementally extended forest. Each completed group of
eight exchanges receives the existing Qwen user-spine attention cuts and
bounded parent summaries once. Those sealed group sections survive later
appends unchanged. The open group remains routable through its typed exchange
summaries with exact raw spans; flush/close builds the final partial hierarchy.
Higher parents do not merge across these sealed eight-exchange groups. This
changes the live grouping policy relative to rebuilding a whole live-source
tree, and needs broader recall evaluation before claiming quality equivalence.

Publication still waits for raw indexing, required summaries, embeddings, and
source validation. Learning follows the committed prefix. Failed publication
retains completed preparation results for retry. Failed preparation leaves
the journal intact; explicit flush retries it. Restart authenticates the saved
snapshot and journal checkpoint; uncommitted work can be prepared again from
durable source events. A cold-reopened snapshot becomes the next immutable base.

The initial regression suite passed **140 tests in 54.22 seconds**. New tests
exercise out-of-order completion, the three-job bound, no premature publication
or learning, preparation/publication failure recovery, separate context and
ingestion settings, preservation of uncommitted text under the token budget,
sealed-root stability, exact cold-reopen equality, and append after reopening.

The live evaluator is `tools/evaluate_chat_io_stream.py`. It reuses the saved
1.13M-token archive, asks the original twelve questions followed by twelve
additional continuation checks, and performs no inter-turn flush or delay.
It records the number of completed exchanges missing from the recalled snapshot
on each question. The main scheduling check is whether that lag ever exceeds
the twelve-exchange recent window. Final drain and all learning remain included
in total cycle time; startup is reported separately.

### Streaming live result, 2026-09-30

`eval_results/chat-io-stream24-20260930-r1` completed **24/24 correct answers**,
comprising four record acknowledgements and twenty recall answers. All lifecycle
checks passed. It used the same authorized Haiku reader and C: active-storage
placement as r15, with one existing **1,130,052-token** starting archive. There
were no new history builds, inter-turn delays, or inter-turn flushes.

| Measurement | Streaming result |
| --- | ---: |
| Maximum completed exchanges absent from a recalled snapshot | **3** |
| Configured recent exchange window | **12**, plus current input |
| Correct answers | **24/24** |
| Mean / maximum answer latency | **4.834 / 9.383 s** |
| Answers below five seconds | 16/24 |
| Reader provider mean | 3.950 s |
| Mean delivered prompt | 4,633 token proxies |
| First twelve replies, without a midpoint drain | 56.728 s |
| All twenty-four replies | 117.686 s |
| Final ingestion and learning drain | **7.868 s** |
| Complete twenty-four-exchange cycle | **125.554 s** |
| Startup, excluded from the cycle | 140.680 s |
| Provider calls | 24 reader, 6 merge, 0 raw-summary |

The queue kept up in this sample: memory was at most three completed exchanges
behind on any recall, well inside the twelve-exchange retention window. Its
lag did not grow across the second half of the conversation. There were 24
exchange preparations and 23 ordered publications; one publication coalesced
two already-prepared exchanges. Three complete eight-exchange hierarchy groups
were sealed, with no open group left at the final drain.

Preparation jobs occupied 12.876 seconds in aggregate. Publication operations,
which include raw indexing and hierarchy/embedding work, occupied 74.937 seconds
on the writer. Within those operations, raw indexing totaled 21.014 seconds,
compilation 18.445 seconds, and snapshot publication 28.272 seconds. Concurrent
stages overlap, so these figures are not additive elapsed cycle time. Publishing
more frequently incurs more total validation/publication work than the prior
two-batch design, but the measured writer still stayed ahead of context eviction.

This is evidence for the scheduling design, **not an established reply-speed
improvement**. The previous twelve-question r15 sample averaged 3.680 seconds
per answer versus 4.834 here. Its provider mean was 3.063 versus 3.950 seconds,
and mean delivered prompt length was 3,259 versus 4,633 tokens. The larger window
initially included more historical source text; provider latency also varied.
These are unmatched timing samples, so the difference cannot be assigned solely
to context size or scheduling. The first twelve replies in this run did not
beat the prior twelve-reply window, and no under-50 full-cycle claim follows.

All 96 new input/output/recall/feedback events were indexed, ending at **5,541
events and 1,172,412 tokens**. All 24 learning updates completed. Post-drain
recall hydrated the exact original new Project Larch input and both recorded
facts. There were no pending events, feedback, or errors, and the original
source memory files remained unchanged. A separate read-only process verified
every persisted event ID/role/text against the captured journal, exact native
and parent receipt equality, all applied feedback, and database integrity.
The closed C: runtime was then archived under the run's `live` directory on F:;
all eleven file hashes matched. Archive export was outside the measured cycle.

Two final concurrency/IO hardening changes followed the frozen live run:
standalone late tool observations now enter preparation without waiting for
another user exchange, and merge-cache readers share the writer's per-request
lock so they cannot observe an artifact before its checksum sidecar is sealed.
Those cases have focused regression tests; the live timing above predates these
two changes. Final validation passed **142 tests in 51.95 seconds** and
`git diff --check`. All live/gateway workers closed.

Artifacts and entry points:

- `eval_results/chat-io-stream24-20260930-r1/report.json`: checks, per-turn lag,
  answer latency, prompts, and all backend phase timings.
- `eval_results/chat-io-stream24-20260930-r1/runtime-reopen-check.json` and
  `runtime-archive.json`: cold-reopen and archive verification.
- `tools/evaluate_chat_io_stream.py`: bounded, frozen 24-exchange live evaluator.
- `tools/engineering_research_chat.py`: streaming by default; explicit
  `--batch-exchanges 6` retains batching and `--batch-exchanges 0` is eager.
- `tests/test_chat_streaming.py`: ordered, bounded preparation and durable IO.

### Local Llama 3.2 3B speed and fidelity screen, 2026-09-30

The requested local trial used **Llama 3.2 3B Instruct Q4_K_M** on this machine's
**RTX 2070 SUPER, 8 GB**, with llama.cpp b11272 CUDA 12.4, all layers requested
on GPU, one slot, a 16,384-token context, flash attention, and Q8_0 KV cache.
The model and both runtime archives passed pinned SHA-256 checks. Model source:
[Bartowski's quantization](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/tree/5ab33fa94d1d04e903623ae72c95d1696f09f9e8)
of [Meta's instruction model](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).
Runtime source: [official llama.cpp b11272 release](https://github.com/ggml-org/llama.cpp/releases/tag/b11272).

The probe replayed all **24 reader packets and six summary requests** saved by
the streaming run above. It preserved each message and output limit, used
temperature zero, and made no retries or prompt repairs. Prefix caching was
disabled; all 30 responses reported zero cached prompt tokens. One separate
warmup brought the total to 31 local generations. No histories were rebuilt,
and no memory stores or production model defaults changed.

| Measurement | Local Llama | Historical control on the same packets |
| --- | ---: | ---: |
| Reader answers correct | **23/24 (95.8%)** | Haiku: 24/24 |
| Recall answers, excluding four acknowledgements | **19/20** | Haiku: 20/20 |
| Mean reader request latency | **1.681 s** | Haiku: 3.950 s |
| Median / maximum reader latency | **1.528 / 3.113 s** | — |
| Mean reader time to first content | **1.483 s** | — |
| Reader requests below five seconds | **24/24** | — |
| Reader prompt processing | **3,171 tokens/s** | — |
| Reader generation after first token | **96.4 tokens/s** | — |
| Mean summary request latency | **0.634 s** | Qwen: 2.559 s |
| Summary generation after first token | **113.5 tokens/s** | — |
| Summary responses passing existing parser | **5/6** | — |

Llama processed a mean 4,644 model-native input tokens per reader request.
Model/server startup took **51.136 s**, and the first tiny generation took
another **47.608 s**: **98.744 s** before the measured resident replay.
Those costs are excluded from the request table and must be paid on a cold
start. Whole-device usage was about 5.7 GB during the run and 2.6 GB after the
temporary server stopped; these are device snapshots, not isolated allocation
or peak measurements.

The reader's only miss was turn six: it returned `{"recorded": true}` instead of
the requested Nimbus region and release, even though the exact values were in
the packet. This is a task-following failure on delivered evidence. The reader
latency was about 2.35 times lower than the historical Haiku mean, but the
hosted control was recorded earlier and the resident memory pipeline was not
running during Llama's replay. This is a reader screen on repeated synthetic
facts, not a new 100-question score, engineering-quality result, or full-cycle
latency measurement.

Summary speed did not establish usable summary quality:

- `26-merge.json` stopped with incomplete JSON, missing its closing brace.
- `27-merge.json` reported **20 checks passed**, while the source said **three**.
  Twenty was a recall-section count in another event. It also lost entity
  attribution among the projects.
- `30-merge.json` assigned Larch's `juniper-641` / `larch-mig-604` to **Arden**;
  the source explicitly supplied Arden's `saffron-842` / `arden-mig`.
- `25-merge.json` and `29-merge.json` were valid JSON but replaced exact Larch
  identifiers with generic field names.

**Decision:** retain Llama as a fast local reader candidate for a broader saved
packet and engineering test. Do not promote these summary outputs as a Qwen
replacement. GPU co-residency with the memory attention/embedding models still
needs measurement before claiming a local full-pipeline improvement.

Artifacts: `eval_results/llama32-local-20260930-r1/assessment.json` records the
failure findings; `corrected-report.json` contains the final timing aggregates;
each response and exact source request is retained. The original sealed report
is preserved: the derived report corrects decode throughput to exclude the
first token, which llama.cpp accounts for in prompt processing, and removes an
inapplicable reader-correctness field from summary statistics. No calls were
repeated for that reporting correction. `tools/probe_llama32_local.py` is the
replay entry point. The local server closed and released its GPU allocation.

### Unused Qwen coverage weights, 2026-09-30

The current summary chunker loads six Qwen3-8B blocks and reads attention layer
5, using zero-based numbering. Blocks 0–4 execute their attention and MLPs to
produce layer 5's input. Coverage then stops at layer 5's attention pre-hook,
after input normalization, and explicitly computes its Q/K/V/O readout. Layer
5's MLP, its post-attention norm, and the synthetic model-final norm are never
executed by this path, although their weights occupy GPU memory. Earlier MLPs
and layer 5's value/output projections remain necessary for the current signal.
K/V caching and gradients are already disabled; there is no generation head.

A local probe replayed three saved-summary batches containing 17 candidates,
with three measured repetitions per batch in each arm. Hooks rejected any
baseline execution of the supposedly unused modules. The candidate replaced
only those modules with parameter-free guards that raise if executed. **Every
QK score, OV magnitude, head weight, transport-vector element, candidate order,
and workspace counter matched exactly.** No precision or earlier-layer weights
changed.

| Allocation, decimal GB | Baseline | Unused weights removed |
| --- | ---: | ---: |
| Qwen GPU parameter bytes | 2.315 | 2.013 |
| Qwen process peak allocated bytes | 2.495 | 2.193 |
| CPU embedding table bytes | 1.245 | 1.245 |
| Mean measured coverage call | 106.54 ms | 106.53 ms |

The measured saving was **302,006,272 bytes**, about **302 MB / 288 MiB**:
301,989,888 bytes in the unused MLP and 8,192 bytes in each unused norm. That
is approximately 13% of Qwen's GPU parameter allocation. The probe did not load
BGE or Llama, so its peak is not the complete memory-system footprint. Latency
was unchanged, consistent with removing resident weights that were already
skipped during execution. Production construction and defaults remain unchanged;
any integration must explicitly restrict pruning to coverage use because the
generic encoder's residual/capture APIs can execute these modules.

The earlier [association layer sweep](03%20-%202026-08-16%20-%20Live%20Qwen%20head%20memory%20smokes.md)
favored layer 1 for selected-head association links, while layer 5 performed
better for residual/CAV entry. The [two-layer answer pilot](15%20-%202026-08-18%20-%20Policy-locked%201M-context%20answer%20pilot.md)
scored 10/10 development questions under its older retrieval policy. These are
grounds for testing two layers, not evidence that two layers preserve today's
summary chunking. The first two complete blocks contain 771,785,728 FP16 bytes;
removing their analogous unused final MLP and post-attention norm would leave
469,787,648 GPU weight bytes, plus the unchanged CPU embedding table. These
two-layer figures are parameter counts, not measured runtime or quality results.

Evidence: `eval_results/qwen-unused-memory-20260930-r2/report.json` and its
sealed before/after outputs; entry point `tools/probe_qwen_unused_memory.py`.
The first attempt stopped during artifact serialization before pruning; r2
corrected the output container and completed. The three existing early-exit
regression tests pass. No provider calls, history rebuilds, answer evaluations,
or production changes were made. The probe process exited and released GPU
memory. Llama/Qwen co-residency remains unmeasured.

### Lossless GPU weight window, 2026-09-30

The user clarified that the requested weight window must use **lossless
compression**, preserving the current FP16 values. An initial quantization
probe was stopped and its experimental driver removed. No quantization was
integrated. NVIDIA nvCOMP 5.3.0 was downloaded into an isolated cache, with both
official wheel hashes verified; the project environment was not modified.
[nvCOMP](https://docs.nvidia.com/cuda/nvcomp/) provides GPU lossless codecs, and
its [Python API](https://docs.nvidia.com/cuda/nvcomp/py_api.html) permits reusable
decompression configurations and decoding into existing device buffers.

The completed probe covers all **39 used linear matrices**: the five full
blocks and the final attention readout. Norms and CPU token embeddings retain
their existing representation. ANS-compressed buffers remain on GPU. Each
linear operation restores its matrix, executes the same FP16 GEMM, and releases
the expanded matrix. The largest matrix is **100,663,296 bytes**. The optional
byte-plane layout separates the two bytes of each FP16 value before compression
and reverses that permutation after decoding; it changes no bits. FP32 attention
reductions remain unchanged.

| Matched three-batch measurement | Pruned FP16 baseline | ANS plain | ANS byte planes |
| --- | ---: | ---: | ---: |
| Linear weight storage, decimal GB | 2.013 | 1.625 | **1.429** |
| Live allocated GPU memory, decimal GB | 2.022 | 1.640 | **1.439** |
| Peak tracked allocation, decimal GB | 2.192 | 1.873 | **1.738** |
| Mean coverage call | 96.78 ms | 119.48 ms | **151.91 ms** |
| Restored matrices byte-exact | baseline | 39/39 | **39/39** |
| Linker outputs exactly equal | baseline | 17/17 | **17/17** |

The byte-plane arm saves **584,633,582 weight bytes (29.0%)** and **454,041,600
peak allocation bytes (20.7%)** against the already-pruned baseline, adding
**55.13 ms** per measured call. That is **1.34 GiB live / 1.62 GiB peak**.
All candidate orders, scores, head weights, transport vectors, and workspace
counters match. Three saved batches and three repetitions per arm were used;
no histories, answers, or summaries were regenerated. Compression preparation
and byte checks are excluded from warm call timing. CPU copies exist solely
as experimental controls between arms; forwards transfer no weight matrices
from the host. The existing CPU embedding lookup is unchanged.

Two allocation details matter. nvCOMP's DLPack export exposes backing capacity,
not just compressed length, so compressed buffers are compacted once to their
actual byte length. Compression also needs more scratch than decoding: the r2
probe uses separate codecs and releases the setup codec before measurement.
The r1 probe retained that workspace and therefore understated the saving.
nvCOMP array allocations use a Torch-backed allocator for tracking. Native
codec allocations may remain separate, so reports also include device-free
snapshots. The CUDA allocator reserved about 2.00 GB after the byte-plane arm;
reserved memory is distinct from the 1.74 GB peak live allocation.

Combined residency is still an estimate. Qwen's measured 1.62 GiB peak plus
Llama's earlier roughly 3.0 GiB incremental device use gives about **4.6 GiB**.
GPU-resident BGE adds roughly 2.1 GiB of weights, taking the model budget to
about **6.7 GiB before desktop and other overhead**. Keeping BGE permanently on
CPU is a candidate placement, but neither that configuration nor simultaneous
Llama/Qwen residency was exercised here. No production loader or defaults changed.

Artifacts: `eval_results/qwen-lossless-window-20260930-r2/report.json`, paired
sealed output files, and `tools/probe_qwen_lossless_window.py`. All restored
weights and linker outputs passed exact comparisons, sealed artifacts verified,
and `git diff --check` passed. The probe exited and released its GPU allocations.

### BGE permanently on CPU: measured latency tradeoff, 2026-09-30

The same pinned BGE-M3 checkpoint was replayed in FP32 on the Ryzen 9 3900X
and RTX 2070 SUPER. Inputs were 24 saved live-IO queries, 17 summaries in three
batches of 4–8, and 16 raw chunks in two batches of eight. Each workload ran
twice after warmup: 48 query calls, six summary batches, and four chunk batches.
No histories were rebuilt and no provider calls were made.

| Mean warm embedding time | GPU | Tuned CPU | Added time |
| --- | ---: | ---: | ---: |
| Single recall query | 0.037 s | **0.341 s** | **0.304 s** |
| Summary batch, 4–8 texts | 0.113 s | **1.795 s** | **1.683 s** |
| Raw chunk batch, eight texts | 0.304 s | **5.843 s** | **5.539 s** |

CPU query p95 was 0.460 s and maximum was 0.546 s; the longest eight-chunk
batch took 6.087 s. BGE token lengths were 26–53 for queries, 36–210 for
summaries, and 11–351 for chunks. These are component measurements excluding
model loading and queue wait, not end-to-end chat or answer-accuracy results.
The completed GPU control from r1 was reused after verifying identical input
texts, sealed metadata, and vector hashes.

CPU thread configuration materially changes the result. This PyTorch build
uses a native ATen thread pool: setting four ATen threads left MKL at one
effective thread. The initial attempt to change ATen counts after computation
was stopped when the runtime rejected those changes. A fresh probe kept ATen
at four and tested 1, 4, 8, and 12 MKL threads, checking effective counts with
`torch.__config__.parallel_info()`. It selected 12 using the recorded objective
of four query calls plus one eight-summary batch. That pilot improved the
eight-summary batch from 14.298 s with one MKL thread to 3.350 s with twelve.
The override uses
[MKL_Set_Num_Threads_Local](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-0/mkl-set-num-threads-local.html),
which applies to the calling thread; production integration must configure
the actual embedding worker. No model weights or numerical precision changed.

All 57 compared vectors were finite. Minimum CPU/GPU cosine agreement was
0.999999999989 and maximum absolute difference was 5.67e-7. Outputs are
numerically very close, not bit-identical; this does not establish full-index
ranking or answer equivalence. CPU placement removes **2,271,019,008 bytes
(2.115 GiB)** of BGE weights from the GPU. The CPU arm had zero Torch GPU
allocation and about 2.35 GB process RSS.

Foreground recall uses one query embedding, making the isolated 0.304 s
increase promising. Background ingestion is the concern: the shared model
lock can hold recall behind an approximately six-second chunk batch. Smaller
CPU batches of one, two, and four long chunks averaged **0.799, 1.475, and
2.753 seconds** respectively. Smaller batches plus verified foreground queue
priority should be tested before adopting CPU placement. Queue contention,
sustained ingest throughput, and simultaneous Llama/Qwen execution were not
measured here.

Production remains unchanged. The staged embedding loader currently forces
CUDA and stored embedding identity includes device placement. Supporting CPU
against existing GPU-built indexes requires an explicit compatibility decision;
the probe neither bypassed that check nor rebuilt indexes.

Evidence: `eval_results/bge-cpu-placement-20260930-r2/report.json`, the sealed
GPU control in `eval_results/bge-cpu-placement-20260930-r1`, and their hashed
vector files. `tools/probe_bge_cpu_placement.py` reproduces the experiment;
`--baseline` reuses a compatible sealed GPU control. The r2 directory preserves
the exact measured driver. Artifact integrity and vector hashes were verified.
The benchmark process exited and released its allocations.

### Recall admission and CPU ingestion scheduling repair, 2026-09-30

The requested 1M-token evaluation was paused before launch to repair ingestion
scheduling. The old scheduler checked for waiting queries before acquiring a
separate model lock. Several background jobs could pass that check and queue
on the lock before a recall arrived, defeating the intended foreground priority.

`SharedEmbedding` now uses a reentrant admission gate with foreground and
background FIFO queues under one condition. A running operation completes;
waiting recalls then precede all waiting background jobs, including earlier
arrivals. Nested model operations retain ownership and exceptions release the
gate. Qwen's existing shared model operations use the same admission gate.
CPU and unresolved-device background batches default to **one text per call**;
explicit CUDA placement retains batches of eight. Summary embedding and raw
chunk embedding both yield between batches. Concurrent chunk-cache access is
also synchronized without holding that cache lock during model execution.

This changes scheduling, not the configured encoder identity, weights, reader,
or placement defaults. CPU BGE placement and lossless Qwen compression remain
separate experimental configurations. Small batches bound the amount of work
ahead of recall, not its wall time for arbitrary text lengths. Sustained
foreground saturation can still delay ingestion; the recent-context window
and backlog need end-to-end measurement.

The real CPU contention probe reused the earlier saved inputs: twelve queries,
sixteen chunks in two background jobs, and seventeen summaries in a third.
All four calling threads were warmed and configured with twelve MKL threads
and four ATen threads before measurement. No generation or history rebuild
occurred.

| Measured CPU contention result | Value |
| --- | ---: |
| Mean / maximum query queue wait | **0.563 / 0.883 s** |
| Mean / maximum query embedding including wait | **0.922 / 1.355 s** |
| Longest raw chunk / summary model call | **0.887 / 0.740 s** |
| All twelve query calls finished | **11.077 s** |
| All 33 background texts and twelve queries drained | **24.487 s** |
| GPU allocation in the probe | **0 bytes** |

The twelve query vectors exactly match the saved CPU control. Chunk and summary
vectors remain numerically equivalent: minimum cosine 0.9999999999966 and
maximum absolute error 2.80e-7. Changed batch padding can change floating-point
rounding. All work completed with no dropped texts. These are component
contention results, not answer latency or a sustained chat throughput result;
the earlier approximately six-second eight-chunk call is not a matched old
scheduler contention control.

Before the fix, all eleven existing scheduling/resident tests passed after
redirecting pytest temporary storage into the workspace; the initial default
temporary directory was inaccessible. After the fix, **26 focused tests pass**,
covering already-queued background jobs, real foreground admission, exception
release, reentrancy, CPU/GPU batch order, source ownership, resident publication,
and streaming persistence. `git diff --check` also passes.

Evidence: `eval_results/bge-recall-schedule-20260930-r1/report.json` and its
sealed implementation/input plan; entry point `tools/probe_bge_recall_schedule.py`.
The probe exited successfully. The 1M/100-question evaluation has not started.

### FastEmbed CPU BGE-M3 comparison, 2026-09-30

The requested FastEmbed comparison is complete. FastEmbed 0.8.1 has no built-in
BGE-M3 entry, so its custom-model API loads a pinned
[BGE-M3 ONNX export](https://huggingface.co/onnx-community/bge-m3-ONNX)
at revision `25b9af8e87a38eb120cfe87125383677b9cd309e`. The graph contains
389 FP32 initializers and no quantization operators. CLS pooling, unit
normalization, and the 8,192-token tokenizer limit match the measured PyTorch
behavior. Model and tokenizer file hashes are sealed. Packages are isolated
under `.cache/experiments/fastembed/package`; the main environment is unchanged.

The same 24 queries, 17 summaries, and 16 raw chunks from the preceding CPU
experiment were replayed with the same repetition counts. A small 4/8/12-thread
pilot selected twelve ONNX Runtime threads on the recorded latency objective.

| Mean CPU model time | PyTorch, tuned MKL | FastEmbed / ONNX Runtime |
| --- | ---: | ---: |
| Single query, 48 calls | 0.341 s | **0.125 s** |
| Summary batch, 4–8 texts, six calls | 1.795 s | **1.481 s** |
| Eight raw chunks, four calls | 5.843 s | **5.130 s** |

Query embedding is **2.72 times as fast** in this component experiment.
Query p95 is 0.174 s. All 57 vectors agree with the pinned PyTorch CPU
control to minimum cosine 0.9999999999919 and maximum absolute difference
4.70e-7. This is numerical agreement, not bit equality or an accuracy result.
Four individual chunk calls averaged 0.487 s, with a 0.623 s maximum; their
population differs from the full chunk batches. Small background batches
remain necessary to keep a long batch from blocking foreground recall.

`tools/fastembed_bge.py` validates this sealed admission and exposes the real
ONNX backend identity. An explicit opt-in policy permits querying the earlier
pinned FP32 BGE CPU/CUDA indexes. Historical source receipts retain their
original provenance, while new receipts name the FastEmbed backend and its
compatibility evidence. The default legacy application loader is unchanged;
the new combined local runtime selects FastEmbed. No history was rebuilt and
no provider inference calls were made in the component comparison.

Evidence: `eval_results/fastembed-bge-cpu-20260930-r1/report.json`,
`admission.json`, the hashed vector file, and `tools/probe_fastembed_bge_cpu.py`.
The combined local runtime and compatibility/scheduling changes pass **34
focused tests**. The following end-to-end evaluation must establish behavior
under concurrent ingestion, Qwen attention, and local generation separately.

The combined runner is `tools/evaluate_chat_io_local100.py`. It reuses one
authenticated 1,115,343-token seed and continuously ingests new IO into a C:
working copy. BGE stays on CPU, compressed Qwen and Llama stay on GPU, and
generation uses only a loopback llama.cpp server. Historical seed summaries
retain their earlier model provenance; this is not a fresh local rebuild of
the million-token source. New local summaries use source-checked extracts,
with exact-prefix fallbacks recorded separately. Extract validity does not
establish that the chosen quote contains the most useful information.

Two interrupted attempts are retained transparently. `chat-io-local100-20260930-r1`
failed before answering because the benchmark adapter omitted the streaming
published-event count. Its regression test now covers that contract.
`chat-io-local100-20260930-r2` stopped after 22 answers when the benchmark's v7
packet envelope was found to bypass authenticated recall-summary reuse. The
corrected adapter stores the ordinary canonical exact-source envelope, while
the reader receives the v7 projection of those same delivered spans. All 22
saved packets passed an offline source-validated alias replay. Local extract
parsing also accepts JSON arrays and fenced JSON while retaining exact-source
checks, and generation limits scale with the number of input fragments.
Neither interruption was selected using accuracy grades. The complete attempt
is recorded separately under `chat-io-local100-20260930-r3`.

The r3 evaluation completed all 100 answers, continuous native hierarchy
publication, and all 100 learning updates, using only local inference.

| Combined local runtime result | Measured value |
| --- | ---: |
| Cold admission and model warmup, excluded below | 136.68 s |
| Mean / median reply | **4.928 / 5.070 s** |
| Reply p95 / maximum | 7.046 / 20.900 s |
| Replies below five seconds | 48 / 100 |
| Answer window, including artifact overhead | 502.64 s |
| Final ingestion and learning drain | **369.40 s** |
| Full measured cycle | **872.03 s** |
| Unpublished completed exchanges at the end of answering | 47 |
| Expected support quotes present | **100 / 100 packets** |
| Packet text identical to the earlier PyTorch/Sol run | **98 / 100** |

The test meets a five-second **mean reply** target narrowly; it does not meet
that limit consistently or establish that ingestion keeps pace. New IO is
retained in recent context while publication lags, so the reader's context
grows beyond twelve exchanges during this stress test. Mean reader input was
7,529 model tokens. The largest reply spike was reader prompt processing,
with negligible GPU admission wait, not CPU query embedding.

BGE's 101 query calls, including warmup, used 11.28 s of compute and 15.25 s
of queue wait; maximum query wait was 0.550 s. Raw embedding used **574.68 s**
for 1,570 uncached chunks; summary embedding used 43.89 s. Thirteen Qwen
attention calls took **6.05 s** total. Llama handled 100 answers and 55 summary
merges; there were no raw summarizer calls after authenticated recall reuse
was restored. Eighty-four merge-fragment fallbacks retained exact source
prefixes. Those fallbacks preserve provenance, not necessarily useful coverage.
The figures overlap and must not be added as disjoint wall-clock stages.

The main remaining ingestion cost is re-indexing delivered recall copies.
They account for **260,468 of 268,217 new body tokens (97.1%)** and **1,434 of
1,734 new raw chunks (82.7%)**. The journal must retain these delivered packets,
but authenticated original-source pointers provide a candidate route to reuse
existing index entries. That optimization is not implemented by this experiment.

The separate-process audit recovered all **5,821 events**, including 400 new
IO/recall/feedback events, 100 packets, 1,627 exact original-source pointers,
and 100 Hebbian updates. Both native and parent receipts reconstruct exactly;
the original million-token source remains unchanged. There were no outstanding
events or feedback after the drain, and no OOM or per-turn model transfers.

Reader quality does **not** preserve the earlier 94/100 result. The local
Llama grader reported 34/100 but made obvious semantic-equivalence errors.
Codex's review of all fixed question/reference/prediction triples found **64
adequate answers, 20 incomplete, and 16 incorrect or nonanswers**. Under that
stated rubric the local grader falsely rejected 31 adequate answers and
accepted one incomplete answer. This is an assistant reference review, not
an independent human adjudication or the earlier Sol grader. It must not
replace historical campaign scores. Every packet contained the required
support, so these observed answer failures occur after evidence delivery.
Examples include omitted multi-part facts, an earlier word-limit correction,
and answers copied from unrelated recent topics. The changed reader model
and appended recent-context prompt are both confounded; no ablation establishes
their individual contributions. FastEmbed is supported by this result, while
the local Llama reader configuration is not ready to replace the prior reader.

Evidence in the r3 directory includes `report.json`, `cycle.json`,
`reference-review.json`, `retrieval-comparison.json`, `new-io-cost.json`, and
`reopen-audit.json`. All 447 measured source files are retained under
`implementation/` with their original hashes. Subsequent cleanup fixes release
the compressed Qwen weights when a runtime closes, without changing inference.
Repeated real-device checks retained exact pre/post-compression score receipts.
Native allocator buffers still retain 8.125 MiB per runtime lifecycle until
process exit; full allocator teardown is not claimed. See
`eval_results/local-runtime-close-20260930-r5/assessment.json`. Earlier cleanup
checks are retained: two rejected probes added a field to the strictly bound
encoder identity; the callback now belongs to the runtime wrapper instead.
The focused integration regression suite passes **46 tests**.

### Shared Qwen prefix with a quantized generative continuation: feasibility

The proposed replacement for the separate Llama reader is architecturally
possible, but has not been implemented or benchmarked. The current local
reader is Llama 3.2 **3B**, and the pinned Qwen3-8B has **36 layers**. Attention
routing can stop at layer 5; generation must restore the omitted layer-5 MLP
and normalization, run layers 6–35, and apply the final normalization and output
head. Later quantization cannot retroactively affect the earlier attention
signal when the prefix input, weights, and computation are unchanged.

The installed llama.cpp quantizer supports per-tensor dtype overrides through
`--tensor-type` and `--tensor-type-file`, permitting FP16 prefix weights with
a Q4_K_M remainder. Its
[quantizer documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
and [Qwen's GGUF model card](https://huggingface.co/Qwen/Qwen3-8B-GGUF)
support that format choice. This does not provide a ready bridge from our
PyTorch/nvCOMP attention prefix to a llama.cpp decoder; shared weights and
intermediate attention access need runtime integration and score validation.

Local safetensors headers contain 1,157,678,592 parameters in the full six-layer
prefix (**2.156 GiB at FP16**), 5,788,392,960 in the remaining 30 layers, and
622,329,856 each in the embedding table and output head. A pure Q4_K storage
lower bound for the tail is 3.032 GiB; Q4_K_M adds some higher-precision tensors.
Keeping the embedding table on CPU and extending the measured lossless prefix
compression gives a rough **5–6 GiB GPU weight budget**, before KV cache,
workspace and desktop use. Fit on this 8GB GPU is therefore not established;
shorter context or some CPU placement may be needed. Non-thinking generation
is supported by [Qwen](https://huggingface.co/Qwen/Qwen3-8B).

This proposal shares model weights. Summary-routing inputs and reader prompts
differ, so their activations are not generally reusable. Every generated token
still traverses the complete model. Quantization should reduce weight traffic
relative to full-precision Qwen, but speed relative to the current 3B reader,
answer quality, and the effect on attention require a measured packet replay.
It also leaves the separately measured CPU raw-ingestion bottleneck to solve.

### Inline frontier answer and exchange summaries (2026-09-30)

The live JSONL binding now requests one structured generation from its configured
reader: `answer`, followed by `memory.user` and `memory.assistant`. Each memory
channel contains a routing summary capped at 128 tokens and one to four exact
support quotes from its own source. The prompt preserves requests, proposals,
reported outcomes, negation, and uncertainty; assistant statements do not become
user assertions. Quote validation establishes source binding, not factual truth
or semantic entailment of every summary claim.

`ChatIO` returns only the decoded answer and existing provider metrics. It saves
the full generated envelope internally alongside the exact user/assistant text,
source hashes, input/output event IDs, recall packet IDs, and model/request
provenance. An acknowledged retry returns the saved visible response without
another provider call. Internal summaries do not enter the recent-chat text or
raw hydration. Ordinary recall learning still records co-access, not correctness.

The compiler consumes valid inline summaries for whole turns that fit its
existing 2,048-token raw fragment boundary. This also reuses the frontier summary
for short turns. Larger turns retain normal fragment summarization so retrieval
granularity is preserved. Later hierarchical merges still run as needed; this
change removes eligible raw-summary requests, not every compilation operation.
Already published user turns stay unchanged under eager ingestion. A valid
answer with missing, oversized, or incorrectly quoted memory is retained and
uses ordinary ingestion. Truncated or ambiguous response envelopes fail explicitly.

The default is enabled in `tools/engineering_research_chat.py`; the diagnostic
switch is `--no-inline-memory`. Other readers use `generate_inline` explicitly,
so existing benchmark runners and their frozen results are not silently changed.
The current gateway/JSONL transport buffers the entire response. Answer-first
token streaming has not been implemented, and the shared output cap must cover
both answer and summary. No visible latency improvement is claimed.

The one-call Sol check is stored in
`eval_results/inline-memory-sol-smoke-20260930-r1/report.json`. It requested a
reversible staging deployment plan without execution. Sol returned the plan and
two accepted summaries in **17.63 seconds**, using **412 input / 283 output
tokens**, with no reasoning tokens. Both summaries became source-bound atoms
without an additional raw-summary generation. This is one synthetic protocol
check, not an accuracy benchmark or a controlled timing comparison.

The probe initially wrote its atom list in the wrong artifact envelope. Offline
finalization reused the saved generation and wrote `compiled-atoms.json` and
`report.json`; the failed `atoms.json` is retained for audit and is not an
authoritative result. No provider retry occurred. Focused tests cover hidden
output, cached retries, fallback behavior, source/role binding, independent
caches, fragment boundaries, resident publication, original-text hydration,
and exact cold reopening.

#### One 1M memory, 100 inline Sol answers: completed evaluation

`eval_results/chat-io-inline-sol100-20260930-r1/` contains the completed run.
It reused one authenticated **1,115,343-token** history and its 100 questions;
no history was re-ingested. Sol (`codex_sdk/gpt-5.6-sol`, reasoning disabled)
generated the answer and hidden summaries in each of 100 calls. Grading began
after all answers and the final ingestion drain were sealed, using 100 separate
Sol calls at concurrency three. A single gateway warmup was additional.
Local compilation retained FastEmbed BGE-M3 on CPU, the losslessly compressed
six-layer Qwen attention prefix, and Llama extractive hierarchy merges.

| Measure | Result |
| --- | ---: |
| Original automatic grade | 91/100; no invalid grades |
| Required support present in packet | 100/100 |
| Accepted inline summary pairs | 99/100; one quote-validation fallback |
| Inline summaries actually reused | 198: 99 user, 99 assistant |
| Mean / median / p95 reply | 12.010 / 11.216 / 17.826 s |
| Maximum reply / replies under five seconds | 27.037 s / 0 |
| Mean reader request | 10.485 s |
| Mean input / combined answer-and-memory output | 3,784.02 / 126.52 tokens |
| Answer window / final drain | 1,223.502 / 18.100 s |
| Complete measured cycle | 1,241.602 s |
| Cold startup, excluded from cycle | 222.216 s |
| Maximum ingestion backlog | Three completed exchanges |
| Separate raw-summary calls / hierarchy merge calls | 0 / 15 |

Assistant source review covered every rejected answer. Five rejections called
additional facts unsupported even though the delivered user statements explicitly
contained them: stand-up practice (ordinal 18), abstract-art events (22), nutrition
requests (45), a proposed painting hobby (62), and Narnia's personification (67).
Two rejections demanded details the question did not request: the photo inside
the locket (21), and the physics-summary subject matter when the question asked
only for format and word count (34). These seven corrections give **98/100
adjusted**. The original **91/100** is retained unchanged. This is an assistant
review of misses, not independent human adjudication or a regrade of passing
answers. The purchase sentence's "last week" scope (53) remains ambiguous and
uncredited. The road-bike answer (65) omitted the user's existing Strava app.
`source-review.json` binds every conclusion to exact quotes in the saved packet.

Separate-process reopening reconstructed both index receipts and verified
**5,821 events, 100 packets, 1,627 original-source pointers, and 100 Hebbian
updates**, with zero pending ingestion or feedback. The original source store
was unchanged. All 198 inline atoms were found in the compiler cache. The sole
invalid summary pair used ordinary ingestion; it did not cause another answer
generation. The local server stopped after grading and GPU use returned to
about 2.5 GiB including the desktop.

The gateway's mean first-content time was **10.484 s**, nearly identical to its
10.485 s total reader time. Output is effectively buffered upstream; changing
the chat frontend alone cannot provide answer-first delivery through this route.
This configuration does not meet the five-second reply target. Its shorter
final drain must not be attributed solely to inline summaries: slower answers
also give background ingestion more time to catch up.

Compared with the earlier 94/100 Sol run, the question population is identical
and **98/100 raw evidence packets match exactly**. That earlier run lacked live
native hierarchy refresh and the recent-chat tail, so this is not a controlled
inline-on/off comparison. Historical as-of questions exclude the newly generated
I/O from retrieval. The run validates answer regression and ingestion of inline
summaries; it does not measure their future recall quality or establish that a
model understands its own summaries better.

Evidence: `report.json`, `assessment.json`, `source-review.json`,
`historical-comparison.json`, `reopen-audit.json`, and `verification.json`.
All **450 frozen implementation files** were archived, and **1,615 sealed JSON
artifacts** verified before publication of the verification record itself.

## October 1 larger-battery stop and gateway diagnosis

**Status:** Investigation complete; campaign remains stopped; upstream cause
unconfirmed. The ten existing 1M histories were queued for 100 questions each,
followed by 15 matched engineering tasks (30 actor arms). The engineering bundle
contains the ten original engineering checkpoints and five additional checkpoints
from five further archive families. All fifteen source cutoffs and rubric anchors
were audited before engineering execution; those tasks have not run.

`eval_results/inline-battery-20261001-r1/` stopped in history 01 after **87
completed answers**. The next request, ordinal 87 / question 88, returned empty
content, `finish_reason=stop`, no usage and no first-content timestamp after
6.069 seconds. The inline parser then reported a JSON-envelope error. No grading
had begun: neither an 87% accuracy score nor a completed 1,000-question result
exists. The user-requested operational stop prevented the remaining histories
and engineering tasks from starting.

The failed prompt was 3,386 project-proxy tokens with a 1,536-token output
budget. It asked about the planned Arkansas River fishing trip; the previous
completed inline run answered the same question successfully. Separate diagnostics
preserved the exact failed prompt and did not replace any benchmark answer:

| Diagnostic | Result | Request time |
| --- | --- | ---: |
| Exact request, streaming | Answer and both summaries accepted | 8.785 s |
| Exact request, non-streaming | Answer and both summaries accepted | 6.106 s |
| Simple inline-summary control | Answer and both summaries accepted | 6.406 s |
| Exact request, streaming again | Answer and both summaries accepted | 6.489 s |

All four HTTP responses were 200. For every probe, SDK-assembled text exactly
matched the saved raw response. All three exact-request replays answered
"guided float trip" and "brown trout." The non-streaming route returned zero
input and output usage despite producing text; switching transports is therefore
not an established reliability or accounting repair.

The older September 25 engineering battery already contains the same
empty-content/stop/no-usage pattern: **22 E07 memory replies and 16 E07
full-context replies**, among 123 actor calls overall. These precede the inline
summary protocol. Together with the successful exact replays, this points toward
an intermittent no-answer failure on the gateway/provider route. It does not
identify the internal failing component or establish an empty-response rate.
The original failed response did not retain raw chunks or the gateway request ID,
so refusal/tool-only output or an upstream adapter defect cannot be retrospectively
distinguished with certainty.

A separate-process partial-lifecycle audit verified **5,772 journal events and
stored turns**, all **87 completed answers**, **88 recall packets**, **1,418 exact
original-source pointers**, and **87 applied Hebbian updates**, with no pending
feedback. Both index receipts loaded. The failed input, recall and error are
durable; there is no invented assistant output for question 88. Of the completed
answers, 85 had accepted inline pairs and two used ordinary-summary fallback.
The local server shut down and GPU usage returned to about 2.4 GiB.

The local reporting repair distinguishes empty completed responses from invalid
JSON, unfinished output, refusal, reasoning-only output and tool-only output.
Both evaluation gateway adapters now retain parsed stream chunks and allowlisted
gateway correlation headers. Successful visible text is unchanged, sensitive
response headers are excluded, and **no automatic retry was added**. Targeted
validation passed **34 tests**; `git diff --check` passed. The stopped campaign's
original artifacts and archived implementation remain intact. A continuation
must explicitly record the new implementation and retain the original failure;
the diagnostic successes are not replacement benchmark answers.

Evidence: [diagnosis and limitations](../../eval_results/inline-gateway-diagnosis-20261001-r1/assessment.json),
[partial lifecycle audit](../../eval_results/inline-gateway-diagnosis-20261001-r1/partial-lifecycle-audit.json),
and the four per-probe `wire.json` / `result.json` records in that directory.

### Three-history continuation

At the user's request, `eval_results/inline-three-20261001-r1` targets three
completed 1M histories, with 100 questions and an independent persistence audit
per history. History 1 clones the stopped store and journal, retains all 87
completed answers byte for byte, and generates only the remaining 13. The
failed question's original recall/error remain durable; its continuation uses a
new recall linked to the existing input. Its audit therefore expects 101 packets,
100 successful learning updates, and two extra operational events. Histories 2
and 3 use their existing ingested stores and run all 100 questions. There is no
historical reingestion.

The frozen policy allows two additional attempts only after a confirmed terminal
empty response. Each attempt retains its request, response, stream and error;
successful recovery records bind the failed and successful requests. Refusal,
reasoning-only, unfinished and ambiguous outcomes do not take this retry path.
Reply timings include retry waits and attempts. The initial failed request and
the interruption remain separately visible; history 1's cycle timing measures
only its continuation, while its answer distribution includes the retained 87.
Automated accuracy and provider reliability are reported separately. Diagnostic
replays are excluded from benchmark answers.

The continuation and bounded-retry checks, together with inline-memory and stop
gate regression checks, passed **37 tests**. The campaign completed all three
histories, grading, and separate-process persistence audits:

| History | Raw body tokens | Automated correct | Required support complete | Mean reply | Accepted inline pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1, resumed | 1,115,343 | 94/100 | 100/100 | 10.91 s | 98/100 |
| 2 | 1,111,235 | 94/100 | 100/100 | 10.04 s | 100/100 |
| 3 | 1,151,461 | 88/100 | 98/100 | 10.21 s | 100/100 |
| Total | Three separate histories | **276/300 (92%)** | **298/300 (99.3%)** | **10.39 s** | **298/300** |

These are raw automated grades, with zero invalid grades and no manual score
corrections. Of the 24 rejected answers, 22 had all recorded support quotes in
their delivered packet. Complete support does not determine whether those
rejections are reader failures or grader errors. The two support gaps were both
in history 3: the requested cabinet-official quiz format (question 56) and the
farm-finance budget template (question 60). Both answers were rejected.

All 300 learning updates persisted. Audits reconstructed both native and parent
index receipts, checked every journal event against stored turns, verified
4,923 exact original-source pointers across 301 packets, and confirmed that all
original histories were unchanged. The extra packet is the retained failed
attempt in history 1. All 87 retained answers remained byte-identical, and all
local model servers shut down after completion.

Histories 2 and 3 took 1,035.63 and 1,046.56 seconds for their full answer,
ingestion and learning cycles, including final drains of 17.67 and 13.19 seconds.
These cycle times exclude startup and grading. History 1's remaining 13 answers
and final drain took 160.38 seconds, including a 19.05-second drain; it has no
single uninterrupted full-cycle measurement. Mean reply still exceeds the
five-second target.

There were **zero new provider errors** across 213 new answer requests, 300 judge
requests and three gateway preflights. The bounded retry path was not needed in
this campaign. The original empty response remains preserved; this successful
continuation does not establish that the upstream intermittent fault is fixed.
The exact evaluated implementation and test files are archived with the run.
These three tests validate the QA lifecycle; the paused engineering battery is
separate.

Evidence: [sealed final verification](../../eval_results/inline-three-20261001-r1/final-verification.json),
[campaign completion](../../eval_results/inline-three-20261001-r1/complete.json), and
[implementation snapshot](../../eval_results/inline-three-20261001-r1/implementation-snapshot.json).

## Production native-ingestion correction (2026-10-03 UTC)

The native proxy had retained the ordinary raw dense/lexical ingestion path even
though its recall uses summary indexes. Delivered recall copies therefore caused
new raw embeddings. Incremental SQLite publication also repeated full raw-source
validation and token counting. These were downstream integration problems; shared
answer/summary generation did not remove them.

`ResidentNativeBackend` now calls `capture_native_many`. Exact raw turns and their
deterministic learning chunks commit together with compact native capture
receipts, without raw embedding, HNSW updates, lexical postings, or legacy pending
ingest obligations. A capture receipt is not a searchable publication receipt.
Legacy ingestion remains available through its original API. Existing source
chunks and graph identities are preserved.

Warm publication reuses immutable, authenticated source prefixes, incrementally
extends the full transcript hash, and validates changed span partitions and new
turns. SQLite source-revision triggers invalidate the raw-turn cache on external
edits. Unchanged summary/vector rows retain their admitted hashes. Cold admission
and recovery still reconstruct and validate the complete persisted snapshot.
Whole-index membership checks, dense-array construction for changed snapshots,
and complete canonical snapshot hashing remain; this is not a claim that every
publication operation is constant-time.

Older full snapshots migrate to the incremental store during cold admission,
before interactive updates. The chat writer reads only new journal rows after its
committed prefix. Learning now runs after each successful publication, even when
more preparation is ready. Its existing graph accepts unembedded native chunks
only when the published native index covers their exact raw sources. Capture
alone does not authorize learning. Successful and failed sync phase timings are
saved to each conversation's `ingestion-timings.jsonl`.

Verification:

- **239 distinct tests passed** across the expanded proxy/lifecycle/package suite
  and the final targeted checks. New checks cover atomic capture, conflicting
  retries, missing chunk topology, rejection of altered raw addresses, avoidance
  of old-prefix tokenization/reloads/raw embedding, ongoing learning under a
  backlog, and preserved diagnostics after failed publication. The capture,
  incremental-store, native-chat, and consolidation tests were added to CI.
- The **29 recorded exchanges** from the stopped 2M run were replayed through the
  production resident writer on a private clone. The source starts at
  **2,226,578 tokens / 10,666 events** and ends at 10,782 events. All 29 Hebbian
  updates survived, with **zero raw chunk embeddings**. The final incremental
  native and parent manifests matched a fresh publication exactly and passed
  full cold reopen.
- In the final replay, preparation, ingestion, publication, and learning averaged
  **3.659 seconds per exchange**, with a **7.314-second maximum**. Cold admission,
  including one-time migration, was **61.912 seconds**, reported separately. A
  concurrent Pixi package build overlapped early exchanges, so these timings are
  not an isolated throughput measurement.
- The replay reused authenticated recorded summaries, attention results, and
  vectors and made **zero new model calls**. It isolates the repaired downstream
  work. It does not establish fresh answer accuracy or complete live-model
  throughput at 2M, 5M, or 10M tokens.
- A rebuilt Pixi `.conda` artifact also passed the real HTTP proxy probe using
  the existing dependency environment and real local Qwen, BGE, and Llama. The
  package payload matched the tested source files and imported no research
  modules. Four controlled provider calls covered inline replies, older-context
  recall, a native tool/result cycle, and duplicate replay without another
  provider request. Final health had zero pending events, pending feedback, or
  failed sessions. After shutdown, a cold audit verified **44 events, 53 exact
  source pointers, and three learning updates**. All 73 stored source chunks had
  **zero raw embeddings and zero HNSW labels**, with no legacy ingest rows. All
  models stopped. This uses a controlled provider, not fresh answer-quality
  grading or a fresh dependency-install test.

Evidence: [final replay](../../eval_results/native-ingest-replay-20261003-r3/report.json),
[phase timings](../../eval_results/native-ingest-replay-20261003-r3/backend-timings.json),
[expanded regression results](../../eval_results/native-ingest-ci-20261003.xml),
[final targeted results](../../eval_results/native-ingest-final-20261003.xml),
[packaged HTTP check](../../eval_results/native-ingest-package-20261003/probe/report.json),
[packaged cold-reopen audit](../../eval_results/native-ingest-package-20261003/probe/reopen-audit.json).
The replay report records the exact tested source-file hashes. The original
stress-test artifacts were not modified.

## Live 2M / 200-question rerun (2026-10-03 UTC)

The corrected run completed **186/200 answers correct (93.0%, automated Sol
grading, no manual score adjustments)**. Complete reference support quotes were
delivered in **199/200 packets (99.5%)**. Thirteen of the fourteen graded misses
had complete support in the packet; the remaining miss lacked full support. This
does not classify those thirteen as reader errors rather than grader errors.

This is one combined memory containing **2,226,578 stored source tokens**
(2,210,122 unique role/text tokens), with 200 fresh accepted answers. Historical
question dates remain unchanged: eligible content varies by date, with a minimum
of 1,115,343 tokens. Authenticated compiled source summaries were reused; this
does not measure cold summarization of 2M new tokens.

- Mean reply: **11.747 s**, median **10.679 s**, p95 **18.364 s**. Mean reader
  call: **10.069 s**. These cover successful exchanges, including in-exchange
  retries, and exclude startup, stopped unsuccessful exchanges, restarts, and
  grading. There is no uninterrupted full-cycle timing claim.
- Across all three corrected-run segments, **205 warm sync updates averaged
  3.368 s** (maximum 10.633 s). **200 distinct learning updates averaged 0.207 s**.
  The maximum completed-exchange backlog was **3**, accounting for recovery
  events. Final drain was **5.613 s**, with **zero pending events or feedback**.
- **199 inline summary pairs accepted, one fallback**. All **3,609 new raw
  chunks** had **zero raw embeddings and zero HNSW labels**.
- Separate-process cold reopening verified **11,470 events**, **202 recall
  packets** (including two failed attempts), **3,391 exact original-source
  pointers**, and **200 persisted Hebbian updates**. Native and parent receipts
  reconstructed exactly. The repaired source remained unchanged. Models stopped.

Two defects were exposed and kept visible in the evidence:

1. The original combined benchmark sorted its atomic sections by ID but simply
   concatenated source vector matrices. **All 10,746 vectors were consequently
   attached to the wrong sections.** The first rerun's 61/200 score is marked
   invalid as an estimate of system accuracy. The original 1M stores were intact.
   `run_memory_scale_stress.py` now aligns vectors by authenticated section
   identity. A fresh corrected copy preserved raw data, questions, and references
   byte-for-byte, made no model calls, and independently verified **zero remaining
   vector mismatches**. No raw history was reingested.
2. The gateway returned a completed but malformed inline JSON envelope after
   138 answers, then again after 147. Recovery preserved all completed answers
   and failed IO. The evaluation harness now supports repeated recovery and up
   to two retries for malformed envelopes, without inspecting answer quality.
   Another malformed envelope at question 148 was recovered in-process. There
   were **203 actor calls for 200 accepted answers, three malformed envelopes,
   and two process resumptions**. This retry change is in the evaluation harness;
   production proxy envelope recovery was not changed by this test.

Seven targeted tests passed for vector identity/routing, invalid source bindings,
bounded protocol retries, unchanged prompts, and repeated recovery preserving
answers and failed packets. The repaired ingestion path kept pace with the
reader in the corrected 2M run. The score remains below the 95% accuracy target;
5M and 10M have not been rerun.

Evidence: [verified cross-segment summary](../../eval_results/scale-2m-rerun-20261003-r4/verified-summary.json),
[all 200 grades and latency](../../eval_results/scale-2m-rerun-20261003-r4/report.json),
[cold-reopen audit](../../eval_results/scale-2m-rerun-20261003-r4/reopen-audit.json),
[vector alignment audit](../../eval_results/scale-2m-source-aligned-20261003-r1/vector-alignment-audit.json),
[invalidated first rerun](../../eval_results/scale-2m-rerun-20261003-r1/accuracy-invalidated.json).
