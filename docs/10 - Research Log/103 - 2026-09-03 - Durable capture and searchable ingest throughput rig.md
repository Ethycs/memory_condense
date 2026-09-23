# Durable capture and searchable ingest throughput rig

**Date:** 2026-09-03

**Status:** implementation and controlled local-disk fake-model assay measured;
real BGE-M3 performance remains unmeasured.

## The three-tier service objective

“Stored faster than generation” has three different boundaries. They must not
be collapsed into one whole-program timer.

| Tier | Boundary | Initial service objective |
| --- | --- | --- |
| T0 durable capture | completed exchange through committed turn and pending-manifest publication | p95 at most 250 ms per capture batch and zero model calls; proxy admission/loss is a separate measurement |
| T1 base searchable | capture start through dense/lexical index publication | at least 500 chunk-token proxies/s and at least 2x the configured offered generation rate, every measured turn no more than `max(2 s, turn_tokens / 500)`, final pending depth zero |
| T2 enriched | searchable base through optional Qwen/CAV/Hebbian link publication | asynchronous; p95 age at most 10 s or before the next dependent turn, never blocks or rolls back T0/T1 |

The numerical thresholds are initial engineering gates, not measured production
claims. The rig evaluates two latency constraints independently: the initial
service deadline `max(2 s, turn_tokens / 500)` and the offered-rate generation
interval `turn_tokens / offered_tokens_per_second`. Its portable gates require
sustained searchable throughput of at least twice the offered rate and
capture-to-searchable lag shorter than that generation interval. Queue age and
loss require a separate proxy load test.

## Historical measurements that bound the target

- The original CPU BGE-M3 baseline indexed 297,767 chunk-token proxies in 937
  seconds: about 318 tokens/s, 3.11 chunks/s, and 2.58 turns/s. See Research
  Log 00 and its `cc_bench.log`.
- The frozen v3 CUDA cache preparation covered 10,441,617 transcript-token
  proxies, 54,246 turns, and 79,915 chunks in 5,810.5 seconds: about 1,797
  tokens/s, 9.34 turns/s, and 13.75 chunks/s. This includes more than the base
  append but benefits from large batches; it is not a single-exchange latency
  distribution. See Research Log 16.
- The first 1.039M cold build took 609.3 seconds, about 1,705 transcript-token
  proxies/s. Its cached ten-query run took 35.6 seconds. See Research Log 10.
- Delayed causal Qwen consolidation measured 324.35 seconds for 483 bounded
  events, or 0.672 seconds/event, after a 2.77-second prefix load. See Research
  Log 09. An earlier compilation measured 0.718 seconds/chunk. See Research
  Log 04.
- Retrospective CAV-span work compiled 6,450 spans from 2,478 user chunks and
  ran ten retrievals in about 134 seconds. See Research Log 14. That backfill
  is not a live per-turn latency result.

The later 35-to-60-minute million-token jobs are historical replay apparatus,
not online ingest measurements. They additionally embed historical queries,
simulate chronological retrieval, build causal graphs, run query stages, and
seal audit artifacts. They must not set the live write-path SLA.

Provider journals give a useful interval comparison but not a decoding-rate
claim. The 68 Terra v5 calls have approximately 13.45-second median,
21.89-second p95, and 32.11-second maximum full-call latency; their outputs are
short structured answers. The repository still has no controlled provider
tokens-per-second measurement.

## Implemented assay

[`tools/performance_rig/ingest_throughput.py`](../../tools/performance_rig/ingest_throughput.py)
uses the production API names directly:

1. deterministic turns are handed to `capture_many()` in configured batches;
2. capture latency, throughput, pending depth, and embedding-call count are
   recorded at the durable boundary;
3. `drain_pending_ingests()` is called with optional manifest, chunk, and token
   bounds until the durable journal is empty;
4. drain-batch p50/p95/maximum latency and both end-to-end and drain-only
   throughput ratios are reported, with 100 tokens/s as the default projection;
   the 2x headroom gate uses end-to-end T0-to-T1 throughput, while the
   drain-only ratio is diagnostic;
5. every returned turn receives a capture-to-searchable latency; a pure
   lexical marker probe and a public dense query check both base indexes without
   treating approximate top-k recall as a write failure; dense verification
   adds exactly one query-model call after T1, while lexical verification adds
   none; and
6. the JSON report evaluates core T0/T1 independently, marks proxy loss and T2
   unmeasured, and therefore makes no end-to-end queue-loss claim.

The harness captures the configured population first and then drains it. This
is intentionally a finite burst/backlog assay: early turns pay the queue wait,
and the T1 throughput gate uses capture-start through final searchable
publication rather than the faster drain-only denominator. A fresh or empty
`--data-dir` is required; the rig never deletes or silently reuses a store.
The headroom projection is not an asynchronous load test: it compares measured
finite-burst end-to-end service rate with one explicit offered-rate assumption.
The drain-only ratio is retained only to isolate the searchable worker's service
capacity once the complete burst is durably captured.

The default `fake` mode hashes a low-vocabulary workload into 64-dimensional
stable vectors and requires no model or network. It exercises the SQLite,
chunking, BM25, HNSW, and orchestration paths, but it does not bound their cost
for real 1024-dimensional, high-vocabulary traffic. `real` mode instantiates
the repository's `EmbeddingService` with explicit model, device, and batch-size
controls. A counting wrapper proves that `capture_many()` did not invoke either
embedder and accounts separately for the one public dense query call made after
T1 publication.

Quick deterministic run:

```powershell
pixi run --frozen -e dev python tools/performance_rig/ingest_throughput.py `
  --embedder fake --turns 64 --tokens-per-turn 500 `
  --capture-batch-size 2 --drain-max-manifests 16
```

Controlled-host real-model run:

```powershell
pixi run --frozen -e dev python tools/performance_rig/ingest_throughput.py `
  --embedder real --device cuda --turns 64 --tokens-per-turn 500 `
  --output-json C:\Users\Keytone\Downloads\memory-condense-rig\ingest-speed.json
```

The real run should be repeated across turn sizes 32, 128, 500, 2,000, and
8,000 tokens; resident-corpus sizes 10k, 100k, and 1M tokens; and offered rates
20, 50, 100, and 200 tokens/s. Cold model load and warm service runs must be
reported separately. A ten-minute sustained test plus a 64-exchange burst is
the minimum useful queue/backpressure assay.

## Production pipeline implemented

The observe proxy now uses the same boundaries instead of synchronously
ingesting one exchange at a time:

1. a bounded admission queue never evicts accepted work; saturation applies a
   bounded wait and then records an explicit rejection without changing the
   provider response;
2. one serialized scheduler greedily captures already-admitted exchanges in
   FIFO batches, with a separate transaction per exchange for failure
   isolation and no embedding call at T0;
3. after capture priority is satisfied, a bounded T1 tick embeds/indexes a
   group of whole manifests with `enrich=False`; it repeats while it makes
   progress and no newer capture is waiting;
4. a separately bounded T2 tick recovers durable enrichment receipts from
   SQLite without re-embedding (default one turn per tick). It may run after T1
   progress and when T1 makes no progress or fails, so a persistent T1 backlog
   cannot starve T2; and
5. startup, idle, and FIFO-sentinel shutdown all service old durable work.

T1 and T2 failures have separate counters. A T2 outage cannot stop a
multi-tick T1 drain. The worker publishes an immutable backlog snapshot after
each tick; health reads that snapshot rather than sharing the SQLite connection
from the event-loop thread. Snapshot age and oldest-work ages continue to
advance between samples. Queue admission, durable capture, T1 completion, T2
completion, and shutdown timeout are therefore observable as distinct events.

Receipt claiming was also changed from repeated per-turn scans to bounded set
operations. A regression crosses SQLite's conservative 500-parameter boundary
with 501 turns and verifies constant-per-parameter-batch reads. Exact indexed
retries now hydrate the already-durable vectors rather than invoking the
embedder again. Two safe inner-loop copies were removed as well: chunk merging
reuses its exact running token count, and SQLite serialization reuses the
already-created float32 vector.

## Historical integrated fake-model result

The final `r9` 64-turn assay used 500-token turns, two-turn durable
capture batches, 16-manifest drain batches, and a projected offered rate of
100 tokens/s. It covered 32,768 chunk-token proxies in 192 chunks. Results:

- durable capture p50 was 47.8 ms, p95 was 57.6 ms, and maximum was 59.0 ms;
- capture made zero embedding calls and left all 64 manifests recoverable;
- four drain batches had 49.4 ms p50 and 60.3 ms p95/maximum latency;
- capture-to-searchable p95 was 1.560 s and maximum was 1.619 s;
- end-to-end searchable throughput was 18,405 chunk-token proxies/s;
- drain-only throughput was 160,825 chunk-token proxies/s, a projected 1,608x
  headroom over 100 tokens/s; and
- pending depth returned to zero, while lexical and native ANN verification
  both passed.

All measured core-capture and T1 gates passed, including the independent
two-second service deadline and five-second offered-generation interval.
Relative to the same `r6` assay before set-based journal claims, total durable
capture fell from 3.871 s to 1.577 s (2.46x faster), capture p95 fell from
228.9 ms to 57.6 ms, and capture-to-searchable p95 fell from 3.770 s to
1.560 s. End-to-end searchable throughput rose from 7,684 to 18,405
chunk-token proxies/s (2.40x).

Proxy admission/loss was not measured by this core rig. These values are a
deterministic fake-model functional measurement taken while another evaluation
occupied the host. They exercise SQLite, chunking, BM25, HNSW, and
orchestration, but the synthetic 64-dimensional, low-vocabulary workload does
not bound those costs for production traffic and does not replace the pending
warm CUDA BGE-M3 run.

Reproduction receipt:

```text
base revision: cf77ffbf48182a096ed9eeef42128cdf238beeb6
report: eval_results/ingest-speed-smoke-20260903-r9.json
sha256: 6B001BFBD7EE7FB8063BB67FFFE1147A19CACAA9437D6A07708F2209C6C3A593
```

## 2026-09-04 controlled local-disk v3 result

The final `r15` v3 assay ran from exact source tree
`ab1bb696061733f83cfac07d46cbe1c3e83091cb` on a controlled local-disk store.
It used the same 64 turns at 500 configured tokens, two-turn capture batches,
16-manifest T1 batches, and 100 offered generation tokens/s. The workload
contained 32,768 chunk-token proxies across 192 chunks.

- T0 durable capture p50 was 46.6717 ms, p95 was 48.7288 ms, and maximum was
  57.2458 ms. All T0 embedding-call counters remained zero.
- Capture took 1.5003940 s. The bounded T1 drain took 0.1897792 s, making the
  measured T0-to-T1 phase 1.6901880 s.
- T1 capture-to-searchable lag was 0.8940063 s p50, 1.4880827 s p95, and
  1.5454253 s maximum.
- End-to-end searchable throughput was 19,387.1924 chunk-token proxies/s, or
  193.8719x the offered rate. This is the gated headroom result.
- Drain-only throughput was 172,663.8114 chunk-token proxies/s, or 1,726.6381x
  the offered rate. This is a diagnostic denominator, not the SLA gate.
- The public dense query and lexical verification both passed. The dense check
  made exactly one post-T1 query embedding call; T0 made none.
- T2 remained deliberately deferred: all 64 enrichment receipts were ready,
  with zero failures. Every applicable T0/T1 SLA gate passed.

The `r11` and `r12` diagnostics placed SQLite on the synchronized repository
path and fell to approximately 3,600--4,000 end-to-end chunk-token
proxies/s. The `r13` profile attributed 6.061 s of cumulative pipeline time to
SQLite commit and 2.129 s to chunking. Moving only the store to controlled
local disk restored `r14`/`r15` throughput. That comparison strongly implicates
storage synchronization/fsync as the cause of the synchronized-path regression
and does not support a full Rust rewrite as the next optimization lever. The
post-move `r15` run was not profiled, so its current bottleneck remains unknown.

Reproduction receipt:

```text
source tree: ab1bb696061733f83cfac07d46cbe1c3e83091cb
report: eval_results/ingest-speed-smoke-20260904-r15.json
sha256: D3145EDC8B0031C19120DB700F5837AE81C573FF80F71E6A0EF4EC7313214D41
```

## Verification

- The integrated ingest, journal, schema, chunker, proxy, lexical/ANN, and rig
  selection passed 206 tests.
- The complete condenser module plus capture-first regressions passed 158
  tests, including generated-ID restart, changed-chunker replay, cross-version
  enrichment adoption, no-reembed indexed retries, and retired evidence.
- The broader non-model suite cannot be represented as a clean isolated-
  worktree run: five locked-evaluation fixture modules require sealed files
  that exist only under the active primary worktree. They were not copied or
  bypassed. The affected production surfaces above were tested directly.

## Claim boundary

This change creates the measurement surface and the priority-aware production
scheduler. It does not establish the proxy's sustained admission/loss rate,
that CUDA throughput remains flat under concurrency, or that T2 enrichment
keeps up with multiple live sessions. A real result requires the
controlled-host matrix, exact machine/model identity, and capture queue plus
durable-backlog observations from the same run.
