# Reduced exact-ten treatment matrix and streamed memory control

**Date:** 2026-08-28

**Status:** sealed provider-free structural assay and unpublished streamed
control complete; no responder or judge calls; no answer score

## Question tested

The preceding reduced assays placed most remaining loss before final packing,
at bounded callback selection, and showed that a fact-derived reread must be
composed over its parent rather than evaluated as a replacement. This assay
tested two narrower hypotheses on the same ten questions that were still
missing source targets:

1. **memory-pressure hypothesis:** holding all seven approximately
   one-million-token namespace indexes resident at once might change retrieval
   output relative to processing one namespace per child process; and
2. **mechanism hypothesis:** coverage-aware callback selection and CAV-style
   cited-parent provenance reinjection might each repair the fact reread, and
   their composition might be monotonic.

The population contains 23 labelled source targets across exact ordinals
`7, 31, 36, 43, 61, 72, 77, 81, 86, 93`. Those questions address seven
independently ingested namespaces. Their physical content indexes contain
7,208,302 tokens in aggregate, approximately one million tokens per namespace.
This is seven separate memory stores, not one 7.2-million-token prompt.

## Experimental firewalls

- Construction was gold blind. Questions and frozen parent packets were
  available to retrieval, while target source labels were joined only after
  the construction artifact was sealed.
- The first four v2 methods were replayed under the same contracts. Four fact
  treatments then formed a `2 x 2` matrix whose bits are coverage-aware
  callback selection and cited-parent provenance reinjection: `00`, `10`,
  `01`, and `11`.
- Every method/question delta retained the common caps of 12 selected
  candidates and 1,536 selected evidence tokens. The complete prompt cap
  remained 8,000 tokens.
- Exact selected spans were deduplicated only after selection. A span already
  present in the frozen parent remained represented as parent coverage rather
  than being counted as a missing delta.
- The resident run made zero provider calls and retained zero transformer
  token-state bytes. The streamed replay also made zero provider calls,
  retained zero transformer token-state bytes, and was forbidden from
  publishing a replacement artifact.
- A combined 69-test gate covered action proof terms, active reconstruction,
  full-store scanning, fact reconstruction, coverage packing, the assay
  runner, and streamed replay behavior before the result was accepted.

Nine of ten fact packets activated the applicable fact treatments. The packet
for ordinal 72 remained fail-closed as invalid, so neither coverage selection
nor provenance reinjection was allowed to turn it into answer evidence.

## Sealed resident artifacts

| Artifact | Identity |
| --- | --- |
| v3 gold-free construction SHA-256 | `c1f3aeae910c072196e5d9550e5ddd723cb9df14fd79e9c4e0420dd611e013db` |
| construction semantic identity | `a58fbb31b08d7255b54a4dd48952e3039bc65d9de48af647955303a876c3f623` |
| v3 post-hoc target audit SHA-256 | `488008b2a80ebb4fbb18de11caf161e1adc93c66700c28c9f4f0933e7685e626` |

The construction is
`eval_results/matched_eval_100/reduced-second-read-missing10-v3/reduced-second-read-construction-v3.json`;
the target audit is the adjacent
`reduced-second-read-target-audit-v3.json`.

The all-resident construction took 362.640 seconds. Peak measured working set
was 1,885,270,016 bytes, approximately 1.756 GiB.

## Seven-method structural result

The stage counts below are source-target hits out of 23. `Final fit` is the
isolated method delta under its fixed cap. `Parent union` is a post-selection
structural union with the frozen parent, not a terminally repacked prompt.
`Complete` counts questions whose entire labelled source set is present in
that structural union.

| Method | Scanner population | Bounded callback | Hydrated prefit | Final fit | Parent union | Complete |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy active | 23/23 | 4/23 | 4/23 | 4/23 | 10/23 | 0/10 |
| wider passive | 23/23 | 4/23 | 4/23 | 4/23 | 10/23 | 0/10 |
| selected source/turn | 10/23 | 10/23 | 10/23 | 5/23 | 10/23 | 0/10 |
| fact `00`: baseline | 20/23 | 10/23 | 8/23 | 8/23 | **14/23** | **3/10** |
| fact `10`: coverage only | 20/23 | **13/23** | **12/23** | 8/23 | 13/23 | 2/10 |
| fact `01`: provenance only | **21/23** | 11/23 | 10/23 | 6/23 | 11/23 | 1/10 |
| fact `11`: coverage plus provenance | **21/23** | **13/23** | **12/23** | 5/23 | 13/23 | 1/10 |

The switches genuinely activated: on the nine valid packets, coverage changed
callback membership and order, while provenance changed scanner population
and the downstream memberships. The result is nevertheless non-monotonic.
Coverage protects more targets through callback and prefit, but its final
delta does not exceed `00` and its parent union is one target lower.
Provenance exposes one additional target in the population, yet replaces
useful downstream candidates. Combining both switches retains the callback
and prefit gains but falls to five final-fit targets.

This is a replacement/competition failure, not evidence that the new signals
are inert. The useful candidates selected by one mechanism are allowed to
displace the protected contribution of another inside a shared delta lane.
The matrix therefore does not support promoting `10`, `01`, or `11` over the
baseline fact reread.

Every method has **0/10 provider-ready structural unions**. No terminal fair
repack was performed, and several raw parent-plus-delta unions overflow the
8,000-token envelope. Even a union whose raw bound happens to fit is not
certified provider-ready under this experiment's terminal policy.

A post-hoc audit of the evidence union across all four fact treatments reaches
17/23 target sources and 5/10 complete source sets. The arm outputs themselves
are gold blind; target labels are used only to measure this union. It remains a
diagnostic upper bound because the combined evidence has not been constructed,
deduplicated, fairly repacked, or fitted under the terminal prompt cap. It is
not a provider-ready treatment or an answer-accuracy result.

## Streamed memory-pressure control

The construction was replayed with one child process per namespace, allowing
each namespace index to be released before the next was loaded. The streamed
control achieved exact equality at every comparison level:

- 7/7 namespace receipts;
- 10/10 question receipts;
- 70/70 method/question receipts; and
- the same canonical construction bytes and artifact SHA-256 as the resident
  run.

Aggregate indexed content remained 7,208,302 tokens, but the largest resident
namespace contained 1,033,517 tokens. Streaming therefore reduced the
simultaneously resident index-token count by 85.66%. Its maximum worker peak
was 882,741,248 bytes, approximately 0.822 GiB. Elapsed time increased to
532.274 seconds because namespace isolation traded process startup and replay
overhead for lower peak residency.

The streamed run had `publication=false`, made no provider calls, and retained
zero transformer token-state bytes. Exact byte equality means the
process-memory layout changed speed and peak memory, but did not change
retrieval recall on this population.

## Conclusion

The reduced workload separates the two suspected causes cleanly:

1. **memory management affects operating cost, not the measured recall.** A
   child-per-namespace dataflow roughly halves measured peak working set and
   reduces simultaneous index residency by 85.66%, but reproduces the resident
   retrieval result exactly; and
2. **the remaining failure is architectural composition.** Coverage selection
   and CAV provenance each alter the intended stage, but allowing them to
   replace candidates inside one shared lane makes the stack non-monotonic.

The next treatment should preserve the baseline fact contribution and assign
separate protected budgets or lanes to coverage and CAV/provenance additions.
Selection should occur within each mechanism before exact-span deduplication,
then the parent and all deltas should pass through one protected-minimum fair
merge and terminal hard fitter. This tests genuinely additive composition
without hiding useful evidence through early cross-method competition.

No responder or judge campaign is promoted from this assay. A provider run is
not warranted until the composed structural packet is terminally repacked,
fits the common 8,000-token envelope, and improves the protected baseline
without a source-coverage regression.
