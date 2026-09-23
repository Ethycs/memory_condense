# Activated-source assertion projection v2 provider-free result

Date: 2026-09-06

Status: sealed provider-free projection assay on the previously analysis-used
locked validation100 fixture; promising density and composition result, but no
new semantic-answer accuracy measurement and no untouched confirmation

## Result

Activated-source assertion projection v2 turns the neighborhood behind the
sources already activated by adaptive v7 into a compact sequence of exact
assertion spans. The run was entirely deterministic and provider-free. It
made zero Qwen, responder, judge, or other provider calls, sealed selection
before gold scoring, and reproduced the selection byte-for-byte.

As a standalone evidence arm, projection is denser but not yet a safe
replacement for v7 raw evidence:

| Evidence diagnostic | V7 raw fallback | Assertion projection | Change |
|---|---:|---:|---:|
| Complete gold source-ID reach | **93/100** | 81/100 | -12 |
| Mean gold source-ID recall | **0.955000** | 0.885833 | -0.069167 |
| Literal-answer containment | **54/100** | 51/100 | -3 |
| Mean best evidence F1 | 0.099386 | **0.109749** | +0.010363 |

These are **evidence diagnostics**, not semantic answer accuracy. No model was
asked to answer the projected packets, and no semantic judge evaluated those
answers. In particular, literal containment is a surface oracle and complete
gold source-ID reach is a labeled-source oracle; neither establishes what a
downstream LLM would infer from the packet.

The useful result is therefore not “projection beats v7.” It is that a packet
using roughly 2.5k context-token proxies retains almost all of v7's literal
containment and improves mean best-F1, while leaving substantial room under
the 7k context cap for protected raw evidence. A provider-free composition
probe confirms that the two formats are complementary.

## Population and experimental boundary

| Item | Value |
|---|---|
| Population | 100 questions across 10 independent approximately-1M-token stores |
| Population identity SHA-256 | `9b8ad9337cfece1306358d0e03682a977f1b289a14b6ff7bfe40c90e6e2cb246` |
| Dataset SHA-256 | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| Split SHA-256 | `8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4` |
| V7 selection SHA-256 | `867a4439af1c369c3f702491045392b8c64c5e2b3b4216c93fd973ada1b6df20` |
| Projection policy | `activated-source-assertion-projection-assay-v2` |
| Provider/Qwen/responder/judge calls | 0/0/0/0 |

This is the same locked validation100 population used repeatedly for
development and failure analysis. The sealed v2 selection is valid as a
gold-blind comparative artifact, but its scored result is **not an untouched
confirmation result**. The source-seed composition probe below is even more
limited: it is a post-hoc, provider-free diagnostic on that locked population
and was not published as its own sealed run/replay bundle.

## Projection method

For each dated question, v2:

1. derives active sources as exact, opaque source IDs from the first
   occurrence of each source in the sealed v7 packed evidence;
2. scans those sources with one parameterized exact `IN` query;
3. splits source chunks into exact quote spans and filters questions,
   proposals, and unsafe fragments while retaining declarative facts,
   categorical values, completed actions, and question-relevant adjacent
   assertion blocks;
4. ranks facts in typed role/operator lanes without using gold or a provider;
5. deduplicates by exact occurrence identity only after lane selection, with
   same-lane and token-tail refill; and
6. applies the sealed binary ranked-prefix packer.

The projection core has a 2,400-token fact budget and a 40-chunk ceiling. The
outer packet retains the existing 7,000 context-token and 8,000 prompt-
workspace-token caps. Exact source spans, role, time, and chunk receipts are
kept so the compact artifact can reconstruct and authenticate the exact
provider-bound messages.

V2 deliberately keeps its projection arm and SHA-pinned v7 fallback arm
separate. `route_adoption` is `undecided`; the assay does not silently replace
or concatenate the production evidence path.

## Packet shape

Across 100 questions, projection selected and packed 2,366 facts and dropped
none at the outer cap.

| Measure | Value |
|---|---:|
| Facts per packet, mean | 23.66 |
| Facts per packet, min / p50 / p95 / max | 10 / 25 / 30 / 35 |
| Context-token proxy, min / p50 / mean / p95 / max | 2,419 / 2,489 / 2,483.98 / 2,512 / 2,535 |
| Prompt-workspace proxy, min / p50 / mean / p95 / max | 3,008 / 3,070 / 3,068.63 / 3,099 / 3,114 |
| Active sources per question, min / p50 / mean / p95 / max | 20 / 38 / 37.22 / 46 / 48 |
| Scanned chunks per question, min / p50 / mean / p95 / max | 391 / 722.5 / 714.97 / 902 / 953 |
| Route modes, user / mixed / assistant | 87 / 7 / 6 |

The active-source count explains the main structural loss: an average of
37.22 active sources competes for an average of only 23.66 projected facts.
A global typed-lane budget cannot represent every active source, even when
the source is available. This is a local-to-global admission defect, not
evidence that assertion extraction itself has failed.

## Failure taxonomy

The stored fixture ordinal is zero-based.

### 1. Inherited address/frontier failures

V7 itself lacks at least one labeled source for ordinals 7, 36, 54, 61, 77,
86, and 93. Because projection is exactly confined to v7-activated sources,
it cannot invent the missing global address. These seven cases still need a
broader frontier, global specialist, recursive search, or another address-
discovery mechanism.

### 2. Projection admission and source-coverage failures

Projection has no complete-source gain over v7 and loses complete-source
status on 12 cases that v7 covers: 14, 19, 21, 31, 34, 42, 43, 53, 65, 76,
81, and 87. The source was available to the scan, but the global fact budget
and lane competition failed to preserve a representative from every needed
source. Ordinal 19 is also a projection-only literal hit, demonstrating that
answer containment can improve even while a separate labeled source is lost.

The direct repair is a protected, source-balanced seed lane ahead of projected
facts, not a larger undifferentiated projection budget.

### 3. Exact-span density versus surface containment

Projection gains literal containment at ordinals 19, 62, and 97, while v7
alone contains the literal answer at 3, 22, 23, 32, 55, and 89. Both arms hit
48 cases and both miss 43. These disagreements establish complementarity:
projection can expose decisive facts buried inside a raw chunk, while raw
evidence preserves short or context-dependent spans that the projection
filter or budget can omit.

The 43 joint literal misses are not automatically retrieval failures. Some
answers require comparison, temporal ordering, aggregation, or paraphrastic
synthesis and may be answerable without containing the reference string.
Only a separately authorized answer-and-judge run can measure that behavior.

## Incremental latency

The complete provider-free run took 227.537 seconds for 100 questions. The
per-question timing scope is only the activated-source scan, assertion
projection, binary packing, serialization, and associated validation work.
It excludes v7 retrieval, initial shard/database setup, provider RTT, model
prefill, and answer decoding.

| Incremental stage | Mean | p50 | p95 | Max |
|---|---:|---:|---:|---:|
| Active-source scan | 44.429 ms | 45.141 ms | 50.573 ms | 52.112 ms |
| Assertion projection | 2,072.658 ms | 2,100.452 ms | 2,682.343 ms | 2,943.835 ms |
| Binary pack | 4.286 ms | 4.444 ms | 5.121 ms | 5.519 ms |
| Serialization | 0.140 ms | 0.141 ms | 0.177 ms | 0.194 ms |
| Incremental question total | 2,240.777 ms | 2,277.504 ms | 2,905.576 ms | 3,153.351 ms |

Assertion projection accounts for approximately 92.49% of mean incremental
question time. Search and packing are no longer the bottleneck. Clause
splitting, assertion classification, stable feature computation, and token
costs are largely query-independent and should move to ingest or a resident
cache. The arithmetic sum of median scan, pack, and serialization is about
49.7 ms, but that is only a lower-bound target: a real cached path still has
query-dependent ranking, lookup, validation, and end-to-end orchestration.

## Provider-free source-seed hybrid probe

A post-hoc composition probe tested whether a small protected raw prefix can
conserve source coverage while projection supplies denser evidence. The
`source_seed_projection` order is:

1. the first v7 raw chunk from each distinct v7 source;
2. all projected facts;
3. the complete v7 raw sequence; then
4. exact chunk-ID deduplication followed by the same binary 7k/8k pack.

Deduplication happens after the arms have selected their candidates, so one
arm cannot suppress another arm's search or ranking. Gold is absent from
construction and used only for the diagnostics shown below.

| Composition | Complete source reach | Literal hits | Mean best F1 | Mean packed | Mean context proxy |
|---|---:|---:|---:|---:|---:|
| V7 raw | 93/100 | 54/100 | 0.099386 | 52.86 | 6,512.87 |
| Projection first, then raw | 92/100 | 55/100 | **0.126007** | 57.74 | 6,884.01 |
| Raw first, then projection | 93/100 | 56/100 | 0.102794 | 57.79 | 6,934.49 |
| One raw source seed, projection, then raw | **93/100** | **56/100** | 0.113612 | 59.22 | 6,873.57 |
| Source seeds + 600 projected-token prefix + raw | 93/100 | 56/100 | 0.108595 | 55.37 | 6,626.12 |
| Source seeds + 1,000 projected-token prefix + raw | 93/100 | 56/100 | 0.114287 | 57.53 | 6,721.46 |
| Source seeds + 1,400 projected-token prefix + raw | 93/100 | 56/100 | **0.115009** | 59.05 | 6,803.94 |
| Source seeds + 1,800 projected-token prefix + raw | 93/100 | 56/100 | 0.114122 | 59.14 | 6,852.54 |

Projection-first maximizes this surface F1 diagnostic but loses v7 source
coverage at ordinal 21 and trades three literal gains for two literal losses.
Raw-first is monotone for these diagnostics, but puts the dense facts too late
to improve F1 substantially. Source-seed-first preserves all 93 v7 complete-
source packets, adds literal hits at ordinals 62 and 97 with no literal loss,
and raises mean best-F1 by 0.014226, about 14.3% relative to v7.

The 1,400-token projected-prefix variant is the strongest efficiency point in
this small diagnostic table: it retains 93/56, reaches the highest
source-preserving F1, and uses less mean context than the unlimited
source-seed projection order. That budget was selected after observing this
analysis-used population, however, so it is a candidate for a registered v3
policy, not a validated optimum.

## Artifact receipts

Canonical v2 root:
`eval_results/longmemeval-1m-hot-retrieval-assertion-projection-v2-full100-validation-20260906`

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `selection.json` | 5,452,417 | `cdb0c591912892fe7f2296f2a17e5bd8000d97d10da8c48da4bce26dab007e57` |
| `runtime.json` | 22,137 | `e75dad54d65d292890f2a6577ca12dc71d3e021611b818ac2ce1a791d523592b` |
| `run_manifest.json` | 776 | `c5772024b6adb8c56b17e98bc2c2d8ae7f10b904c05010cbdaa66d2dde1ccc69` |
| `replay.json` | 943 | `c61037081506aa1613142dd12cf78a9a607e6cbca8d39bfd3b6f425b67664afd` |
| `scores.json` | 58,472 | `4447149939a289b9dc84ac86902aa57de2fba7b6a9abdd03a6978dfa5195aebe` |

The v2 implementation identity is
`800d6c4fa9522949a1a2122eb1d186d14c6d81f57cbc586f5c9674609c4f2bdd`.
The compact selection is 4.483x smaller than the v7 selection and retains
zero provider request-token state bytes. The valid run publishes
`run_manifest.json` last, after authenticating the selection and runtime
sidecars, so a manifest marks a complete recoverable bundle rather than a
partially published run.

The focused core and assay suite passed 58 tests in 29.10 seconds. The replay
was byte-identical and reported zero provider and Qwen calls.

## Decision

Do not replace v7 raw evidence with standalone projection. Promote the
architecture implied by the source-seed probe to a separately sealed v3
candidate: protect one deterministic raw representative per activated source,
spend a separately bounded budget on dense exact assertion spans, append the
remaining v7 raw evidence, and deduplicate only after each mechanism has made
its selection.

Move query-independent assertion work to ingest/cache before claiming a
real-time path. Then seal and replay the composition provider-free. Only after
that should a newly authorized Terra/Sol run measure semantic accuracy. Final
acceptance still requires a fresh untouched confirmation population; neither
the v2 evidence metrics nor the post-hoc hybrid probe satisfy that bar.
