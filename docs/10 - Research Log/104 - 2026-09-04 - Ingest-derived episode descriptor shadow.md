# Ingest-derived episode descriptor shadow

**Date:** 2026-09-04

**Status:** startup catalog and shadow instrumentation implemented and locally
tested; the sealed 1M preparation gate passed, while the Qwen winner replay is
still pending. Live candidate narrowing is not enabled.

## Question

Can query-independent episode work move to ingestion or startup so retrieval
does not repeatedly validate and normalize the same representative vectors?

## Decision

Yes. Compile the memory-side keys once, retain raw evidence as the authority,
and leave only query-dependent scoring, ambiguity resolution, closure, and
packing on the read path.

The first implementation targets the expensive S2 episode tournament. It
builds a transient exact descriptor catalog from existing episode
representatives and their durable chunk embeddings. The catalog is prepared
before the query-time Qwen runtimes load. Each query then embeds the question,
scores the resident matrix once, max-pools the bounded representatives for each
episode, and proposes a small episode frontier.

Preparation is currently an explicit opt-in compilation/startup call. It is
not automatic work in the T0 capture, T1 searchable-ingest, or T2 enrichment
tiers.

This is deliberately a shadow treatment. The proposal is measured, but the
complete population still enters the unchanged source-local Qwen tournament.
Consequently this change cannot improve live latency yet, and it cannot reduce
recall through pruning. It establishes the evidence needed to decide whether
live narrowing is safe.

## Baseline and scale

The fresh 1M-token offset-000 ten-question retrieval measured 203.960 seconds
median, 248.351 seconds mean, and 461.362 seconds maximum per question. The S2
representative stage can offer up to 256 episodes and dominates the read path
with sequential local-Qwen inspections.

The sealed combined store inspected for this work contains 1,119 episodes and
2,238 representative vectors. A contiguous `2238 x 1024` float32 matrix is
9,166,848 bytes, or 8.74 MiB. The same store currently contains zero
`chunk_cav_signatures`; therefore CAV is not available as an honest first-stage
shortlist for this fixture. CAV routing can be evaluated later only after an
actual, content-bound compilation exists.

Controlled local measurements separated compilation from the hot path:

| Operation | Result |
| --- | ---: |
| Sealed one-time validation and normalization of all 2,238 rows | 6.515 s |
| Warm exact matrix score | 0.246 ms median, 0.299 ms p95 |
| Resident descriptor matrix | 8.74 MiB |

The matrix timing is a local synthetic exact-matvec measurement, not an
end-to-end retrieval result. An earlier prototype validation measured about
3.26 seconds; the stricter sealed run, which also verified lexical-control
representative identities, measured 6.515 seconds. Neither number includes
question embedding, Qwen loading or inspection, closure, packing, or answer
generation.

The provider-free preparation artifact is
[`offset000-prepare.json`](data/2026-09-04-ingest-derived-episode-descriptor-shadow/offset000-prepare.json),
SHA-256 `8900ca76734d4f221656c11c8f6f8813829ae29397f3be54ce121b0c2519fd40`.
It binds the sealed historical retrieval, physical combined store, source
embedding identity, compilation, validation policy, and descriptor catalog.

## Primary-LLM boundary cost

Retrieval latency is only one term. The final cost is

`retrieval + hydration/packing + serialization + network RTT + bytes/bandwidth + prompt tokens/prefill throughput + output tokens/decode throughput`.

The sealed offset-000 final S3 packets make the boundary concrete. Across its
ten questions, the provider messages contain 6,929--7,341 prompt-token proxies
(median 7,298), including 6,607--6,996 context-token proxies (median 6,975.5).
They contain 33--73 selected evidence items (median 60) and serialize to
21,821--26,171 UTF-8 bytes (median 24,261).

The wire bytes are small: 24,261 bytes takes about 19 ms at 10 Mbit/s or 194 ms
at 1 Mbit/s, before RTT and queueing. Once retrieval approaches seconds rather
than minutes, primary-model prefill is the larger boundary term: its lower
bound is approximately `7,298 / effective_prefill_tokens_per_second`, while a
fully used 256-token answer reserve costs
`256 / effective_decode_tokens_per_second`. These are arithmetic projections,
not measured provider latencies.

Historical non-streaming Terra calls varied from a 4.30-second median in the
ten-question development journals to 13.45 seconds median and 21.89 seconds
p95 in the later v5 campaign. Neither artifact separates queueing, cache,
prefill, reasoning, or decode time, so that variation cannot be attributed to
evidence length. Streaming TTFT and gateway phase metrics are required before
optimizing the provider boundary against a specific bottleneck.

Therefore ingestion should also compile compact, provenance-bound evidence
cards: atomic claim, entities, event time, source/span IDs, and a short exact
quote. Point questions can send cards plus a few raw excerpts; ambiguous,
reasoning, and completeness questions retain a wider raw-chunk fallback. A
sealed token-budget ablation must prove that this reduces primary-model prefill
without lowering answer accuracy. A text-only remote API cannot consume local
CAV or K/V state; a co-located custom primary model could additionally reuse
pre-tokenized chunks or content-addressed K/V blocks.

## As-built path

`EpisodeDescriptorIndex` performs one ordered SQLite join over discourse
artifacts, episodes, representatives, evidence membership, and chunks. Before
admission it verifies:

- artifact, episode, source, and sequence identity;
- representative ordering, uniqueness, and membership in episode evidence;
- vector presence, width, finiteness, and non-zero norm; and
- the exact durable representative-selection identity (`ordinary_embedding`
  or `lexical_control`) and, separately, the source-store embedding identity
  used for descriptor scoring.

It normalizes admitted vectors once and binds the catalog to the discourse
snapshot, embedding identity, graph-content revision, and chunk-index revision.
It writes no schema or sidecar. Successful episode publication invalidates the
resident catalog after commit, and condenser shutdown releases it.

The prefilter accepts either that verified resident score result or strictly
validated durable vectors. It emits text- and vector-free receipts, stable
score ordering, timings, proposal size, the full Qwen winners, and proposal
containment of those winners. Completeness queries bypass narrowing. Missing
query vectors, incomplete descriptors, identity mismatches, stale revisions,
and corrupt rows all preserve the full candidate population.

The dedicated assay exposes shadow mode; the canonical resumable S0--S3 runner
is intentionally unchanged because its checkpoint schema does not bind
diagnostic policy. An explicit guard rejects `apply` mode in the live
representative path until promotion. The S2-only assay reuses sealed historical
S0 protected anchors and can replay Qwen winner containment without rebuilding
the corpus, running downstream closure, calling a provider, or exposing gold
answers.

## Authority boundary

Descriptors are routing keys, not memories. The append-only transcript, source
chunks, evidence spans, and their durable provenance remain the factual
authority. A selected descriptor points back to those rows; it does not replace
them in the final LLM payload. If the derived plane cannot prove that binding,
retrieval fails open.

This preserves the intended division of work:

- the current opt-in startup compiler computes stable episode descriptors,
  identities, normalization, offsets, and receipts; future background ingest
  or sleep-time work may compute content-bound links;
- retrieval computes the question vector, cheap descriptor match, bounded
  ambiguity resolution, graph/temporal closure, and final packing; and
- answer synthesis receives hydrated raw evidence with provenance, not only a
  compressed descriptor.

## Promotion gates

Live narrowing remains prohibited until all of these gates pass:

1. Full-Qwen winner recall at proposal cap 16 is `1.0` on the sealed offset-000
   assay and then on the locked 100-question population.
2. Explicitly protected and lexical-protected anchors have 100% retention, with
   zero descriptor-caused evidence drops.
3. Completeness-sensitive questions always retain the exhaustive population;
   corrupt, stale, or incomplete catalogs always fail open.
4. Proposal-to-full-population ratio is at most `0.25`, with `0.125` preferred,
   and at least four of seven eligible point queries can actually narrow.
5. Warm prefilter p95 is at most 50 ms.
6. With narrowing applied in an isolated treatment, S2 Qwen latency reaches
   p50 at most 1 second and p95 at most 2 seconds, or demonstrates at least a
   4x speedup without a recall regression.

The order matters: prove containment in shadow, replay the sealed population,
then enable an isolated apply treatment. End-to-end accuracy and latency are
reported only after that treatment, never inferred from matrix timing.

## Verification

At final validation, the descriptor, prefilter, representative-retrieval,
discourse, and cumulative-retrieval core group passed 77 tests. Substituting
the 1M CLI runner suite for the cumulative core passed 69 tests. A separate
condenser/lifecycle/1M-runner group passed 168 tests with one skip. These groups
overlap and their counts must not be added as though they were unique tests.

No provider, LiteLLM, responder, or judge calls were made for this
implementation or its unit tests.
