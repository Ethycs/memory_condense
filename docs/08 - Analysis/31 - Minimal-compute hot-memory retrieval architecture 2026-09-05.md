# Minimal-compute hot-memory retrieval architecture

Date: 2026-09-05

Status: implemented through the sealed source-seed assertion hybrid v3;
provider-free evidence reaches 93/100 complete sources, 56/100 literal
answers, and 0.115009 mean best F1, while semantic evaluation remains pending

## Decision

The v6 development implementation confirms the decision: remove local Qwen
inspection from the normal 1M cumulative evidence-retrieval route rather than
making that inspection incrementally faster. The ordinary product
`search_hybrid` route already avoids Qwen; the completed provider-free assay
also stops the benchmark/composed route from placing a Qwen tournament on
every question's critical path.

The memory should remain exact and addressable in the common durable store.
During T2 enrichment, it should compile several small, query-independent
*addresses* for the same raw evidence: lexical postings, dense chunk and
evidence-atom vectors, episode/source descriptors, typed time fields, and
graph/CAV pointers. At query time the system should:

1. parse lexical and structured cues on CPU while computing **one** current
   query embedding;
2. score the hot address planes with postings lookups and exact matrix
   products;
3. preserve a separate budget for every specialist and union their selected
   IDs;
4. deduplicate only after selection, refill vacated slots, and hydrate the
   exact raw evidence;
5. send that bounded evidence to the primary answer LLM; and
6. invoke a more expensive late-interaction or semantic-block path only when a
   computable coverage/sufficiency check remains open.

In compact form:

```text
T2 ingest (off critical path)
raw/provenance
  ├─ BM25 postings and exact literals
  ├─ dense addresses: chunk + atom/fact + episode/source/card
  ├─ typed time/entity/number fields
  └─ graph/CAV/EM pointers to raw source spans

one prompt tick
query ─┬─ CPU lexical + typed lookup ───────────┐
       └─ one resident query encoder ─ exact scans
                                                ↓
                    protected per-lane union → post-selection dedup/refill
                                                ↓
                                      exact raw hydration
                                                ↓
                         sufficient: emit to primary LLM
                         open: bounded late interaction / wider raw emit
```

This preserves the project's intended layered memory design. CAV, Hebbian
heat, episodes, contextual cards, and typed closure remain complementary
routes into the same evidence store; none becomes a replacement factual
authority.

## Current measured result: v6 windowed linked retrieval

The implemented fast path was measured on the pinned original development
concatenation: 1,039,203 transcript-token proxies, 5,400 turns, 7,895 chunks,
and ten development questions. One warmup rotation was followed by 20 cyclic
rotations, producing 200 measured prompt-to-serialized-evidence samples. The
artifact root is
`eval_results/longmemeval-1m-hot-retrieval-linked-windowed-development-20260905`.

| Measure | v6 windowed linked result |
| --- | ---: |
| Warm local A3 latency | 52.58465 ms p50 / 71.4195 ms p95 |
| Questions reaching every labeled source | 10/10 |
| Questions containing the literal answer | 5/10 |
| Answer-value component recall | 1.000 on the two applicable list questions (11/11 components) |
| Mean / maximum packed chunks | 26.3 / 39 |
| Mean / maximum context proxy | 3,475.7 / 4,460 tokens |
| Maximum prompt-workspace proxy | 5,041 tokens |
| Packing drops | 0 |
| Query-critical Qwen / provider calls | 0 / 0 |
| Query-independent sidecar compile | 1.6724382 s, 4,720.65 chunks/s |
| Gold-blind replay | byte-identical |
| Separate sealed Terra answer + Sol judge | 10/10 |

The four independently selected lanes retain fixed budgets: resident BM25-8,
exact dense-8, source-neighborhood-8, and temporal-event-24. Their winners are
unioned, deduplicated, and refilled; no lane loses a slot merely because
another lane selected the same chunk. The final product is a bounded packet of
authenticated raw chunks ready for the primary answer LLM.

V6 adds one general rule before lane selection. If a dated question states an
explicit lookback, calendar arithmetic derives an inclusive evidence window
and applies it to **every ranked lane before that lane selects its winners**.
The concert question's two-calendar-month window is
2023-02-22 19:31 through 2023-04-22 19:31 in source-local wall-clock time.
Questions without an explicit dated lookback fail open and retain their full
ranked lanes. This is an admissibility constraint, not a relevance reranker.

The source-neighborhood lane still repairs the concrete Serenity Yoga
local-to-global orphan first demonstrated in v5: BM25 and dense found a generic
yoga-app response in the correct source, while same-source adjacent-turn
linking recovered the immediately following user turn naming Serenity Yoga at
neighborhood rank 8. The temporal-event lane has the complementary job of
enumerating completed first-person events across distributed dated turns. The
new calendar window prevents those event candidates, or candidates from any
other lane, from leaking outside an explicit question horizon.

The source/literal/component rows are evidence metrics, not answer accuracy. A
labeled source can contain many turns, and literal-answer presence does not
test whether the answer model performs sorting, subtraction, or synthesis
correctly. The v6 retrieval boundary includes the dated prompt, raw hydration,
packing, rendering, and canonical serialization; it excludes model load/setup,
Qwen, provider RTT, provider prefill, and answer decoding.

### Current end-to-end answer result: v6

A separate gold-free responder pass sent the A3 raw packets directly to
`codex_sdk/gpt-5.6-terra`, sealed the ten predictions, and only then submitted
question, reference, and sealed prediction to `codex_sdk/gpt-5.6-sol`. The Sol
judge marked 10/10 correct. Recorded per-call provider latency was about 8.075
seconds p50 for Terra and 9.305 seconds p50 for Sol; neither is included in the
71.4195 ms retrieval p95.

The v5 ablation remains diagnostically important. It reached the same 10/10
labeled sources, 5/10 literal answers, and 1.000 applicable component recall,
but its unwindowed raw packet caused Terra to return the five correct concerts
plus an out-of-window Killers decoy. Sol therefore scored v5 at 9/10. The v6
calendar-window rule removed that decoy, reduced maximum context from 4,797 to
4,460 tokens and maximum workspace from 5,386 to 5,041 tokens, improved p95
from 74.7484 to 71.4195 ms, and lifted the sealed answer result to 10/10.

Those two scores are development evidence only. The following frozen transfer
assay supplies the broader comparative result; untouched generalization still
remains untested.

## Full100 transfer result: speed holds, terminal accuracy does not

The policy-frozen successor assay in [Research Log
109](../10%20-%20Research%20Log/109%20-%202026-09-05%20-%20Hot%20raw-chunk%20locked100%20comparative%20result.md)
applied the unchanged v6 controls to ten independent approximately-1M-token
namespaces and 100 questions. The fixture has already been used in prior
analysis, so this is comparative validation rather than untouched
confirmation.

| Measure | Dev10 | Locked validation100 |
|---|---:|---:|
| Warm provider-ready retrieval p50 | 52.58465 ms | 51.93795 ms |
| Warm provider-ready retrieval p95 | 71.4195 ms | 72.5515 ms |
| All-required-source reach | 10/10 | 72/100 |
| Literal-answer containment | 5/10 | 51/100 |
| Sealed Terra/Sol semantic accuracy | 10/10 | **66/100** |
| Maximum context/workspace proxies | 4,460 / 5,041 | 4,420 / 4,995 |

The latency result transfers almost exactly; the accuracy result does not.
The full100 retrieval and byte-identical replay still make zero Qwen and zero
provider calls, and every selected chunk fits. Of the 34 wrong answers, 22
lack at least one labeled source and 12 fail despite nominal complete-source
reach. Crucially, the retained 96-deep frontier reaches all labeled sources
for 95/100: only five are true frontier misses, while 17 lose already-found
sources during fixed top-eight lane admission. Multi-session questions expose
the main retrieval gap (11/27 complete-source packets), while preference
questions expose the main post-retrieval gap (5/6 complete-source packets but
only 1/6 correct). Five nominally source-complete failures also omit decisive
within-source chunks, so source-ID reach is not a sufficiency certificate.

This changes the architectural verdict in one important way. The hot raw
union is a strong common first stage, not a sufficient terminal policy. It
beats the historical fixed-S1 semantic baseline by ten points while roughly
halving responder prompt tokens, but trails the materially heavier
policy-v5-r3 frontier by 29 points. The next design should keep the
approximately-72-ms parent and first repair source-balanced adaptive admission
over candidates it already finds. Ingest-derived atom/span, episode/source,
temporal, and operator-aware closure should then activate only for remaining
open obligations. Reinstating a local-Qwen tournament for every prompt is not
supported by this result.

## Adaptive v7 result: admission repairs most source loss

[Research Log
110](../10%20-%20Research%20Log/110%20-%202026-09-06%20-%20Adaptive%20source-balanced%20hot%20full100%20result.md)
records the completed successor. V7 authenticates the frozen v6 selection,
protects its packed A3 evidence as an exact prefix, and admits 32 additional
chunks from the sealed BM25, exact-dense, and temporal frontiers. Opaque
source groups receive one representative before deterministic round-robin
surplus; no new query encoding, search, Qwen, or provider call occurs in the
retrieval plane.

| Measure | Frozen v6 | Adaptive v7 |
|---|---:|---:|
| Complete labeled-source packets | 72/100 | **93/100** |
| Literal-answer containment | 51/100 | **54/100** |
| Mean labeled-source recall | 0.835833 | **0.955000** |
| Terra/Sol semantic accuracy | 66/100 | **70/100** |

All 21 complete-source changes are gains. Of the 17 answer-wrong admission
misses whose sources were already in the v6 wide frontier, v7 closes 15;
ordinals 7 and 36 remain. The five genuine frontier misses remain ordinals
54, 61, 77, 86, and 93. This confirms that fixed final admission—not 1M-token
address search—was the dominant source-recall loss. It also narrows the next
retrieval work to two distinct problems: residual policy choice over known
candidates and new address discovery for absent candidates.

Semantic accuracy changes through eight wins and four losses. Because v7
never removes parent evidence, those losses are answer-model
non-monotonicity under a larger raw packet, not retrieval eviction. The next
accuracy layer should therefore improve evidence density or apply a bounded
obligation-aware answer/check policy while retaining the source-complete v7
packet as protected state.

The sealed v7 adaptive increment measures 22.858 ms p50 but 356.965 ms p95.
That tail is an implementation defect: 44 overflowing packets trigger linear
prefix recounts whose cumulative text work is quadratic. Source selection and
hydration remain single-digit-millisecond p95 stages. Sealed v8 now reproduces
all 100 provider-bound payloads byte-for-byte with 56 complete-prefix fast
paths and 44 binary fallbacks. Packing falls from the sealed v7 artifact's
129.835349-ms mean and 346.8103-ms p95 to 30.229117-ms mean and 67.8294-ms
p95: 4.295x mean/total and 5.113x p95 speedups. The v7 baseline is a
non-contemporaneous same-machine artifact, and the measurement covers packing
only, not integrated retrieval or provider latency. Provider calls remain
zero; retrieval and answer accuracy are unchanged by exact byte equivalence.

## Source-seed assertion hybrid v3: preserve reach, add density

The sealed v3 successor composes two specialists instead of asking either one
to replace the other. Adaptive v7 is the **raw address-and-reach specialist**:
its source-balanced admission finds the broad set of relevant opaque sources
and retains authenticated raw chunks. Assertion projection v2 is the
**evidence-density specialist**: it converts activated source neighborhoods
into compact, query-relevant fact-like assertions. Projection is an addressable
compression aid, not factual authority; the raw store remains authoritative.

The standalone projection result made the tradeoff visible. It raised mean
best F1 from v7's 0.099386 to 0.109749, but complete-source reach fell from
93/100 to 81/100. V3 therefore uses the following source-preserving order:

```text
independently selected v7 raw + independently selected v2 assertions
  -> protect the first v7 raw chunk for every exact opaque source
  -> admit a bounded 1,400-token assertion prefix
  -> append the remaining v7 raw chunks
  -> deduplicate exact chunk IDs after both arms have selected
  -> binary ranked-prefix pack under the existing hard caps
  -> emit the exact sealed v7 raw packet if the source-seed gate fails
```

The order is intentional. Deduplicating before either specialist selects would
let overlap consume or suppress that specialist's budget. Exact raw source
seeds precede assertions so an assertion from the same source cannot
impersonate the protected raw row. Binary packing retains the v8 logarithmic
prefix search. If the final packet cannot preserve every required seed, v3
fails safe to the exact pre-rendered v7 packet, provided that fallback itself
satisfies the same context, workspace, and output-reserve caps.

| Provider-free evidence measure | Adaptive v7 raw | Source-seed hybrid v3 |
|---|---:|---:|
| Complete labeled-source packets | 93/100 | **93/100** |
| Literal-answer containment | 54/100 | **56/100** |
| Mean labeled-source recall | 0.955000 | **0.955000** |
| Mean best F1 | 0.099386 | **0.115009** |

These are retrieval-evidence diagnostics, not semantic answer accuracy. V3
made zero provider, judge, and Qwen calls; it has not inherited v7's 70/100
Terra/Sol score. Its measured incremental composition boundary averages
141.25 ms with 172.52 ms p95. That timing starts from the already sealed v7
raw and v2 projection parents and ends at provider-ready v3 bytes; it excludes
parent retrieval, assertion-projection computation, and all provider work.

The sealed artifact root is
`eval_results/longmemeval-1m-hot-retrieval-source-seed-hybrid-v3-full100-validation-20260906`.

| V3 receipt | SHA-256 |
|---|---|
| Implementation | `a1024b6d20010bad6427b48bbca58dd08be42bc62bbd55e04aedf12cdd6cc1c7` |
| Selection | `0e8027d3150bdf8335ad7a83d1a8bb4a5551312444fea2d5a4e820ece05eb3b7` |
| Runtime | `c99a3683b4faa0ff0d309be3ede4af6319dbcdc6fc68793a684092e2612f8082` |
| Run manifest | `b01db952af11a44de008bd4c5ef4d6f9f23faa36e537cd6e7fb7888738e43147` |
| Replay | `deb0446486c0990507d7a295f5bfdbd5d1d25ef80ce474fbafb1c282dd23085c` |
| Retrieval score | `c7a009eca11188b2123afc02bbc585d4766157c6fff106d102b155886355a9b8` |

The source gate is deliberately narrower than a full evidence guarantee. It
protects one first raw chunk per exact source represented by v7; it does not
preserve every answer-bearing raw chunk later in that source's ranked
remainder, discover a source absent from v7, or prove that an assertion retains
every needed qualifier. The raw and projection arms must also share one
authoritative chunk-ID namespace. Without that identity contract,
post-selection deduplication, raw hydration, and source-seed accounting are
not trustworthy.

## Why this is the decision

### The measured bottleneck is not the small scorer

[Research Log 104](../10%20-%20Research%20Log/104%20-%202026-09-04%20-%20Ingest-derived%20episode%20descriptor%20shadow.md)
measured the current cumulative 1M-token retrieval at 203.960 seconds median,
248.351 seconds mean, and 461.362 seconds maximum per question. Sequential S2
local-Qwen inspections dominate that path.

The same experiment measured a resident exact score over 2,238 normalized
1,024-dimensional episode rows at 0.246 ms median and 0.299 ms p95. Its 6.515
second validation/normalization cost is one-time construction work, not an
inherent per-query cost.

[Research Log 106](../10%20-%20Research%20Log/106%20-%202026-09-05%20-%20QKOV%20to%20MiniLM%20distillation%20cascade.md)
measured one warm six-layer Qwen pass at 99.53 ms median and 118.83 ms p95,
after a 129.26 second cold load. The nested tournament requires many such
passes. In contrast,
[Research Log 107](../10%20-%20Research%20Log/107%20-%202026-09-05%20-%20Evidence-supervised%20MiniLM%20accuracy%20experiment.md)
measured the trained MiniLM score of 96 candidates at 61.14 ms median and 75.62
ms p95, but it regressed exact annotated-turn hit@8 from 155/200 to 127/200.

Even making that 61 ms scorer five times faster would save only about 49 ms,
or 0.024% of a 204-second median. It matters only if it **eliminates** the Qwen
tournament. Adding another fast ranker in front of an always-executed Qwen
fallback cannot produce the required speedup.

### The lifecycle change is now measured

The v6 windowed linked run measured 52.58465 ms p50 and 71.4195 ms p95 from a dated
question prompt to canonical serialized raw-evidence messages. Numerically,
the p50 is about 3,879 times below the historical 203.960-second median. That
ratio is deliberately an apples-to-oranges lifecycle comparison: the old
number includes the higher-layer local-Qwen tournament, while v6 terminates at
provider-ready bytes and includes no answer-model call. It proves that the
multi-minute local inspection was architectural overhead, not an unavoidable
cost of searching 1M tokens.

The bounded v6 packets stayed below 4,460 context-token proxies and 5,041
prompt-workspace-token proxies, with zero packing drops. Provider RTT, prefill,
reasoning, and answer decoding remain outside this retrieval artifact. The
separate Terra/Sol pass establishes 10/10 development answer accuracy, but its
provider time is not folded into the local-retrieval number. Once local
retrieval is below 100 ms, that external answer envelope is likely to dominate
and must continue to be reported separately.

### New resident query-encoder diagnostic

A 2026-09-05 offline spot check loaded the repository's pinned BGE-M3
SentenceTransformer on the local RTX 2070 Super, synchronized CUDA around each
call, and timed one representative question for 30 sequential warm
repetitions:

| Measure | Result |
| --- | ---: |
| Verified cold load | 42,458.88 ms |
| First query encode | 1,276.19 ms |
| Warm query encode | 36.24 ms p50 / 82.00 ms p95 / 125.90 ms max |
| Warm mean | 43.68 ms |
| Peak CUDA allocated / reserved | 2,175.69 / 2,188.00 MiB |

This was a one-query engineering diagnostic, not a sealed benchmark artifact;
it excludes parsing, BM25, matrix scans, hydration, packing, and serialization.
The later v6 assay supplied the missing formal measurement: all ten probes,
20 cyclic rotations, 200 samples, 71.4195 ms p95, and separate cold/setup
accounting. The encoder must remain resident, but the complete local retrieval
path now passes both the 200 ms promotion gate and the 100 ms stretch gate.

The capture side does not require this query cost. [Research Log
103](../10%20-%20Research%20Log/103%20-%202026-09-03%20-%20Durable%20capture%20and%20searchable%20ingest%20throughput%20rig.md)
measured T0 durable capture at 46.6717 ms p50 and T1 capture-to-searchable lag
at 0.8940 seconds p50 / 1.4881 seconds p95 in its controlled local-disk
fake-model run. Slower semantic compilation can remain deferred T2 work.

## First-principles constraints

1. **Raw text is the authority; compiled representations are addresses.**
   Every route must end in authenticated source IDs or spans that can be
   hydrated. A summary, CAV, recurrent state, or KV cache cannot silently
   replace literal evidence.
2. **Pay corpus cost once.** Chunk, atom, episode, source, temporal, and link
   representations should be computed when memory changes, not for every
   question.
3. **Pay query semantics once.** A single query forward should feed all dense
   and, where available, learned-sparse/multi-vector scoring heads.
4. **Search independently, compose monotonically.** A specialist gets its own
   budget. Selected evidence is unioned before exclusion; EM duplicates are
   removed after selection, and vacated capacity is refilled.
5. **Do not approximate a small search prematurely.** There are 7,895 current
   exact-span chunks in the measured development 1M-token store ([Research Log
   22](../10%20-%20Research%20Log/22%20-%202026-08-21%20-%20Recall-guarded%20cumulative%20retrieval.md)).
   An exact flat scan is simpler and safer than ANN at this scale.
6. **A gate must remove more work than it adds.** Confidence logic is valuable
   only when it prevents an expensive stage. It should not sit in front of a
   fallback that always runs.
7. **Let the answer model reason.** This system ultimately supplies evidence
   to a primary LLM. A second local LLM should not repeatedly synthesize the
   same evidence unless it measurably reduces total answer-bound work.

These constraints extend the dual-plane conclusion in [Analysis
17](17%20-%20Local-global%20memory%20connectivity%20technique%20assay%202026-08-27.md):
use compact routing state to find exact discrete/raw evidence. They also
preserve the soft-boundary, union-before-exclusion rule in [Analysis
23](23%20-%20Soft%20topical%20boundaries%20for%20long-chat%20memory%20retrieval%20-%20literature%20review%202026-08-30.md).

## What can actually stay hot

The first row is the completed v6 measurement. The remaining rows are
calculated capacity figures for possible extensions. Binary units are used for
resident memory.

| Representation | Work/footprint at the measured scale | Interpretation |
| --- | ---: | --- |
| Completed v6 compiled sidecars | 35,174,675 bytes; resident BM25 numeric arrays 2,801,532 bytes; source-neighborhood metadata about 1,153,664 bytes | Measured, query-independent, and raw-text-free; the dense matrix dominates the artifact |
| One dense chunk vector | 7,895 x 1,024 float32 = 30.84 MiB; one exact query scan = 8.08M MACs | Keep hot; ANN is unnecessary initially |
| Existing episode descriptor catalog | 2,238 x 1,024 float32 = 8.74 MiB | Already measured at 0.246 ms median |
| Eight dense addresses per chunk | 123.36 MiB fp16; 64.68M MACs for one pooled query | Feasible way to represent several facts/spans without a new query model |
| Eight 1,024d latent keys **and values** per chunk | 246.72 MiB fp16; about 129.35M MACs for pooled QK+AV | Feasible as a later bounded attention sidecar, not needed for the first assay |
| Generic ColBERT-style 128d token vectors | 244.14 MiB fp16 for 1M tokens, before offsets/index data | Feasible, but it is a separate model/projection from the current checkpoint |
| Native BGE-M3 1,024d token vectors | 1.91 GiB fp16 for 1M tokens | Too costly to score exhaustively; shortlist first |
| Full Qwen3-8B bf16 KV | 137.33 GiB for 1M tokens | Impossible on the 8 GiB RTX 2070 Super |

The Qwen KV arithmetic is:

```text
2 (K,V) x 36 layers x 8 KV heads x 128 head dimension
x 2 bytes x 1,000,000 tokens = 147.456 GB = 137.33 GiB
```

Six layers still require 22.89 GiB and two layers require 7.63 GiB before model
weights and runtime buffers. Qwen's local configuration also caps native
position length at 40,960 tokens; that cache alone would be 5.625 GiB.

Even assuming a hypothetical full cache, each new query/decode token performs
roughly 294.9B QK+AV MACs across 36 layers and streams 147.456 GB of KV. At the
card's nominal 448 GB/s bandwidth, the bandwidth-only lower bound is about
329 ms per token, before kernels, indexing, weights, or host transfer. Keeping
the 1M raw corpus and compact addresses hot is realistic. Keeping its full
Qwen attention state hot is not.

The measured process RSS was about 1.57 GiB and resident CUDA allocation was
about 2.12 GiB, largely model/runtime state rather than index data. Cold setup
was 32.206 seconds, including 18.224 seconds for checkpoint verification,
model load, and first forward, plus 11.574 seconds to construct the resident
BM25 arrays. Those are lifecycle/setup costs and are excluded from the warm
query samples. Separately, the query-independent v6 sidecar compilation took
1.672 seconds at 4,720.65 chunks/s. A production service must keep both encoder
and indices resident; recreating them per prompt would erase the latency gain.

## Literature verdict by mechanism

### 1. Exact sparse plus single-vector dense retrieval: immediate GO

BM25 remains the cheapest exact/literal/date/name rescue plane, while a dense
chunk vector catches paraphrase. Dense passage retrieval is specifically
designed to encode documents once and pay one query encoder plus maximum-inner
product search at runtime ([DPR](https://aclanthology.org/2020.emnlp-main.550/)).
At 7,895 rows, exact search avoids the recall and operational costs of HNSW or
IVF/PQ.

The important extension is *not* a new reranker. It is to compile more
addressable units with the already authenticated dense encoder:

- exact evidence atoms/facts and their raw-span IDs;
- source and episode descriptors;
- source-local contextual cards;
- entity/alias and temporal projections; and
- a small number of sentence/span addresses for multi-fact chunks.

All of those can share the same pooled query vector and one or a few flat
matrices. This is the lowest-risk way to move semantic work from retrieval to
ingestion.

The existing last-N contextual-card prototype fits this design only as
deferred T2 indexing. [Research Log
105](../10%20-%20Research%20Log/105%20-%202026-09-05%20-%20Source-local%20contextual%20cards%20with%20LFM2%20Transcript.md)
measured resident LFM2-Transcript generation at 6.507--7.687 seconds per tiny
synthetic card, while its current Qwen card search remained 60.45 ms median
even after source gating. Compile cards asynchronously, embed each accepted
card once, score the card addresses with the common query vector, and hydrate
their raw last-N source spans. Do not generate a card or run a second attention
model during the prompt tick, and do not treat a summary as factual authority.

### 2. Learned sparse retrieval: GO for an isolated sidecar assay

[SPLADE](https://arxiv.org/abs/2107.05720) learns sparse vocabulary expansion
while retaining inverted-index execution. An efficiency study reports variants
within 4 ms of BM25 under its test conditions, with less than a 10% MRR@10
reduction relative to the studied state-of-the-art single-stage neural
rankers ([efficiency study](https://arxiv.org/abs/2207.03834)). Those numbers
do not transfer automatically to this corpus, but the mechanism is a strong
fit for aliases and paraphrases that dense pooling or exact terms miss.

[BGE-M3](https://arxiv.org/abs/2402.03216) is particularly attractive because
one backbone supports dense, learned-sparse, and multi-vector retrieval. Its
[official implementation](https://github.com/FlagOpen/FlagEmbedding/blob/master/research/BGE_M3/modeling.py)
computes those heads from one hidden-state pass.

There is a local blocker: this repository's current `EmbeddingService` wraps
the pinned checkpoint through SentenceTransformer and exposes dense vectors
only. The inspected local snapshot does not include the separate trained
`sparse_linear.pt` and `colbert_linear.pt` head files expected by the
FlagEmbedding inference implementation. Therefore this arm requires a
separately authenticated complete artifact/backend. It must not be presented
as merely enabling two booleans on the existing wrapper.

### 3. Late interaction: GO only after a cheap shortlist

[ColBERTv2](https://arxiv.org/abs/2112.01488) keeps token-granular document
representations and applies query-token MaxSim, reducing its late-interaction
footprint by 6--10x through residual compression.
[PLAID](https://arxiv.org/abs/2205.09707) adds centroid pruning and reports up
to 7x GPU and 45x CPU speedups over vanilla ColBERTv2. More recent engines
include [WARP](https://arxiv.org/abs/2501.17788), which reports 3x lower
latency than the official PLAID engine, and
[MUVERA](https://arxiv.org/abs/2405.19504), which maps multi-vector similarity
to fixed-dimensional MIPS proxies and reports 2--5x fewer candidates at
similar recall in its experiments.

Those systems target collections far larger than 7,895 chunks. Building their
full indexing machinery first would be apparatus-heavy. The first useful test
is:

1. use the exact lexical+dense+specialist union to obtain 32--96 chunks;
2. compute MaxSim only over that shortlist;
3. add late-interaction winners without displacing protected evidence; and
4. measure whether exact annotated-turn recall rises enough to justify the
   query head and storage.

For 96 average-size chunks and a 32-token query, a separate 128d ColBERT
representation costs about 49.6M MACs and reads about 2.96 MiB of vectors.
Native 1,024d BGE-M3 multi-vectors cost about 396.7M MACs and 23.65 MiB.
Either is practical after shortlisting; neither should scan all 1M tokens by
default.

### 4. Per-chunk latent/CAV blocks: HOLD behind the simpler assay

The project's CAV idea is directionally consistent with learned landmark and
index-branch systems: a compact block representation routes a query to raw
memory. [Landmark Attention](https://arxiv.org/abs/2305.16300) trains landmark
tokens to choose context blocks, and the newer
[REFRAG](https://arxiv.org/abs/2509.01092) compresses RAG chunks, selects
important ones, and selectively expands raw representations.

These papers also expose the distinction that matters here. Landmark Attention
changes/fine-tunes the model. REFRAG's primary study requires substantial
encoder/projector/decoder training rather than a drop-in index; its ICLR-2026
submission reports roughly 4,300 H100 GPU-hours across reconstruction,
continued pretraining, and selective-expansion training
([paper](https://openreview.net/pdf?id=uOi0MHNrwo)). A local CAV block can be a
cheap **routing** experiment, but it cannot inherit those accuracy claims and
cannot replace the cited raw evidence.

The smallest safe local form is a versioned matrix of several query-independent
keys per source/episode/card, followed by raw hydration. Test it only if
multi-address dense and learned sparse retrieval leave a demonstrable
local-to-global gap.

### 5. KV caching: useful after retrieval, not as retrieval

[Prompt Cache](https://arxiv.org/abs/2311.04934) reports large TTFT gains by
reusing fixed, position-correct prompt modules.
[RAGCache](https://arxiv.org/abs/2404.12457) caches retrieved knowledge states
in a GPU/host hierarchy and reports up to 4x TTFT improvement.

Arbitrary cached chunks are not directly composable. Their hidden states omit
cross-attention from preceding chunks.
[CacheBlend](https://arxiv.org/abs/2405.16444) addresses this by selectively
recomputing tokens and reports 2.2--3.3x TTFT improvements;
[Cache-Craft](https://arxiv.org/abs/2502.15734) similarly manages reusable
chunk caches and reports 2x end-to-end latency reduction over prefix caching.
These optimize the **answer-model prefill after chunks are selected**. They do
not identify the chunks.

They are also unavailable across the current text-only LiteLLM boundary: a
local KV tensor or CAV cannot be sent through an ordinary provider prompt.
The interoperable product is the hydrated text/fact payload. Revisit chunk KV
caching only if the primary answer model becomes self-hosted and the serving
runtime exposes compatible cache handles.

### 6. Sparse decoder attention: real, but the wrong first intervention

Sparse attention now clearly works when the model and runtime are designed for
it:

- [MInference](https://arxiv.org/abs/2407.02490) reports up to 10x faster 1M
  prefill, but its own reference point is roughly 30 minutes for an 8B model on
  one A100;
- [Quest](https://arxiv.org/abs/2406.10774) selects query-dependent KV pages
  and reports up to 2.23x self-attention and 7.03x overall speedups;
- [RetrievalAttention](https://arxiv.org/abs/2409.10516) indexes KV on CPU and
  accesses 1--3%, but its cited 8B demonstration is 128K on a 24 GiB RTX 4090
  plus host memory at 0.188 seconds per generated token;
- training-free [Sketch&Walk](https://arxiv.org/abs/2602.07397) reports up to
  6x at 20% attention density with custom kernels;
- model-native [MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392)
  reports 28.4x less per-token attention compute and 14.2x/7.6x
  prefill/decode speedups at 1M on H800; and
- [LongCat Sparse Attention](https://arxiv.org/abs/2608.01662) uses
  streaming-aware, cross-layer, and hierarchical indexing in models trained
  natively at 1M context.

These results answer “can sparse attention work?” with **yes**. They do not
make it the right retrofit here. Every approach still requires model KV,
per-layer query/index work, specialized kernels, large host/device memory, a
different trained model, or some combination. Even a 1% slice of this Qwen
cache is about 1.47 GB per query token before weights and indexes. External
retrieval over a 31 MiB dense matrix is orders of magnitude cheaper.

### 7. Recurrent and state-space memory: research-only

[Mamba](https://arxiv.org/abs/2312.00752),
[Recurrent Memory Transformer](https://arxiv.org/abs/2207.06881), and
[Infini-attention](https://arxiv.org/abs/2404.07143) can make online cost
independent or nearly independent of history length. They also replace or
adapt the answer model and compress history into lossy state. That state does
not preserve a guaranteed, inspectable address for every raw fact. It is a
useful auxiliary predictor, not the sole memory plane for this system.

## Implemented query-critical path

### Fast path

1. Preserve two question forms: the plain question drives retrieval, while the
   dated presentation prompt is retained for the final responder. Embedding
   the date header in the search query was tested and regressed ordinary
   retrieval, so it is not part of v6.
2. Parse temporal/event cues on CPU while one resident BGE-M3 query encode runs
   for exact dense scoring.
3. For a dated question with an explicit lookback, subtract calendar months
   from the question timestamp to construct an inclusive, source-local
   wall-clock evidence window. Otherwise leave the window inactive.
4. Search resident exact BM25 and the 7,895-row normalized dense matrix with
   independent budgets of eight chunks each. Apply any active window to each
   ranked candidate tail before selecting either lane's eight winners.
5. For applicable first/last/earliest/latest event-set questions, enumerate
   first-person completed-event candidates with a protected budget of 24,
   subject to the same pre-selection window.
6. Union the admissible base anchors, then walk immediate predecessor/successor turns
   within each source, round-robin across anchors, for eight source-neighborhood
   winners. Turn ordinals define chronology; `start_char` defines within-turn
   chunk order. Filter linked candidates through the active window before the
   neighborhood lane selects.
7. Union BM25-8, dense-8, source-neighborhood-8, and temporal-event-24 in that
   protected order. Deduplicate exact chunk IDs only after each lane selects,
   and refill from each lane's retained tail so overlap cannot reduce its
   contribution.
8. Hydrate authenticated raw chunks, pack under 7,000 context and 8,000
   workspace token-proxy caps, render the dated responder prompt, and
   serialize the provider-ready messages.

The source-neighborhood and temporal-event lanes solve different topology
problems. Neighborhood linking crosses a local turn boundary after a base
retriever has found the right conversational region. Event enumeration joins
distributed, dated observations whose individual chunks are locally weak but
whose set answers an ordering question. Calendar-window filtering enforces an
explicit query constraint consistently across them and the base lanes. The
retrievers remain additive specialists, not re-rankers allowed to evict
lexical or dense evidence.

Adjacent-prompt heat/slew should be an additive continuity lane: prior selected
IDs can receive bounded heat, but the current query is still encoded. That
keeps the n+1 behavior cheap without allowing stale attention to suppress a
topic change.

### Open-frontier path

If the evidence packet lacks required operands, dates, entity bindings, or a
closed source frontier:

1. widen exact dense/lexical retrieval and multi-branch source/episode
   traversal;
2. run late interaction over at most 96 already shortlisted chunks;
3. hydrate the newly selected raw spans and re-check the typed obligation; and
4. if still open, spend remaining prompt budget on the best unresolved raw
   chunks and let the primary LLM synthesize.

The default fallback is **more raw evidence**, not a second local generation
pass. The primary LLM can reason over the bounded raw packet directly. Qwen
inspection remains an experimental diagnostic until it both rescues unique
evidence and has a resident execution environment that does not impose the
measured 129-second cold load.

The previously proposed **semantic binary search** belongs in this
open-frontier branch, not in the first-stage 7,895-row scan. A hard one-child
descent can orphan a relevant topic at an imperfect boundary. The safe form
scores query-independent child descriptors with the same query vector, retains
a small beam plus every child inside a calibrated margin, and recurses until it
can hydrate raw leaves or prove the typed frontier closed. At this corpus size
it is justified by global-to-local closure, not by flat-search speed; it becomes
a scaling optimization only when the address population is large enough that
profiling shows the exact scan is material.

## Staged assay

The v6 artifact tests the mechanisms separately before recombination. The two
specialists retain the historical A2 prefix but have distinct arm IDs in the
receipt.

| Arm | Implemented mechanism | Packed result on dev10 |
| --- | --- | ---: |
| `a0_bm25` | resident exact BM25, budget 8 | 6/10 all-source; 2/10 literal |
| `a1_exact_dense` | exact dense chunk scan, budget 8 | 6/10 all-source; 4/10 literal |
| `a2_source_neighborhood` | same-source adjacent-turn linking, budget 8 | 8/10 all-source; 3/10 literal |
| `a2_temporal_event` | first-person completed-event enumeration, budget 24 when applicable | 2/10 all-source overall; 1.000 component recall on both applicable list questions |
| `a3_protected_union` | protected post-selection union and refill of all four lanes | 10/10 all-source; 5/10 literal; 1.000 applicable component recall |

The A3 union measured 3,475.7 mean and 4,460 maximum context-token proxies,
with 5,041 maximum prompt-workspace-token proxies and no dropped chunks. It
packed 26.3 chunks on average and at most 39, down from v5's maximum of 45.
The museum question sets the v6 context/workspace maxima; the windowed concert
packet is 32 chunks and 3,903 context-token proxies. The exact Serenity Yoga
successor remains packed. This is the implemented minimum-compute path.
Candidate extensions remain staged behind it:

The locked100 v3 evaluator deliberately keeps the terminal arm ID
`a3_protected_union`. Selecting the `source-seed-hybrid-v3` profile changes
which authenticated policy materializes that terminal packet; it does not
change the provider-facing schema or silently blend v3 into a historical v6
or v7 selection.

| Future arm | Added mechanism | Purpose |
| --- | --- | --- |
| A4 | multiple dense atom/span addresses | test whether ingest-side granularity repairs pooled-chunk loss |
| A5 | learned sparse sidecar | alias/paraphrase contribution; requires authenticated head artifact |
| A6 | shortlisted late interaction | fine-grained residual contribution |
| A7 | composed uncertainty gate | demonstrate that the expensive stage is skipped safely |

No arm may replace protected selected IDs. Deduplication occurs only after
selection and refills from retained tails. Future mechanisms should likewise
be assayed in isolation, then added only when they contribute unique evidence
without regression.

### Population discipline

- Use the already analysis-exposed evidence-supervised 200-question material
  and the original offset-000 1M development concatenation for mechanism
  development.
- Do not open validation100 or confirmation200 for this latency research.
- Reuse sealed historical retrieval artifacts when comparing against the
  204-second path; do not rerun it merely to reproduce a known cost.
- Gold/reference data may be joined only after query outputs are sealed. It
  never enters the runtime index, question classifier, or prompt.

The exact first 1M assay population is the original development concatenation
from Research Log 22: 1,039,203 transcript-token proxies, 5,400 turns, 7,895
current chunks, population SHA-256
`fa9a06ebd103d87086943cfa94091bdf607fe07874bc871e465aad409b85ca18`,
and exact-span source receipt
`92c764d7fabfbeef9d068fc52210148eb44b4613530d987f2c5856baeda5bb45`.
Its historical S0 is the minimum evidence control: 10/10 all-source reach,
1.000 mean answer-component recall, 5/10 literal answers, and 2,127.4 mean /
2,332 maximum context-token proxies.

### Executable artifact boundary

The implementation is the provider-free
`tools/assay_hot_retrieval_1m.py` tool. Its five process-separated commands
must be executed in this order:

1. `export-probes` reads the pinned development dataset and split once, then
   publishes only gold-free retrieval and dated-prompt questions.
2. `compile-base` verifies the exact source receipt and builds the ordered,
   normalized 7,895 x 1,024 float32 matrix plus a text-free chunk/source/turn
   coordinate manifest. It accepts no questions or gold.
3. `run` accepts only the compiled sidecar, source store, and sealed gold-free
   probe bundle. It writes deterministic `selection.json` semantics and a
   separately bound `runtime.json`; clock samples never enter the selection
   digest.
4. `replay` reruns only gold-blind selection and requires byte-identical
   semantic output.
5. `score` reopens the raw dataset only after the selection and replay are
   sealed, verifies their SHA bindings, and joins labels into `scores.json`.

The completed v6 receipts are:

| Artifact | SHA-256 |
| --- | --- |
| Gold-free probes | `58b46ac89044780b02f21a3ce4fa8896caa8ab68027fb42dc69935a12d8409f7` |
| Compiled manifest | `0ff8dc76830d8ba52ee5acc78121ffede8313c450a78891d39379b904abbdca1` |
| Implementation | `5863bbbde0f8ebc277e23bfd4188c3c4aec2d64a57cc03272b07708097809859` |
| Semantic selection | `cafe769331b36a2500b43d012360a775669e9ffdc4519bf6a115f83c767cba06` |
| Runtime | `6e6f0ab33a4b0a0d2c58f01a39d125ded3c9240af80571f91d0f4685e99be441` |
| Replay | `2f7679bf1f8bb3788d663db9636fba916a7c1a33157d8365e2a95f0f06b50b8e` |
| Retrieval score | `3e3329004d54d9961e04fbef2f170db695a972f0400b236c4a0bbffac4c9d1d5` |
| Sealed Terra answers | `b423e297c9ac916a08d729ef3c542af8d02e5f204ddfa9bb78beb530f65fa435` |
| Sol judgments | `cc5638c2a0c80263a4f1b198902e54c88d8b38232791caf5852519f474249b08` |

Replay reproduced the selection bytes exactly. The wide top-96
lexical/dense lists are retained only as an unrendered containment ceiling.
Genuine learned-sparse, multi-vector, and atom/span-address outputs remain
separate future arms so their gains and costs cannot be hidden inside the base
result.

### Measured timing trace and retained contract

The 200-sample v6 trace isolates the principal online stages:

| Stage | p50 | p95 |
| --- | ---: | ---: |
| Query encode | 38.552 ms | 56.574 ms |
| Parallel retrieval wall | 42.320 ms | 61.603 ms |
| Resident BM25 | 2.623 ms | 5.646 ms |
| Exact dense scan | 3.317 ms | 3.965 ms |
| Source-neighborhood traversal | 0.478 ms | 1.023 ms |
| Temporal-event search | 0.001 ms | 4.673 ms |
| Raw hydration | 2.985 ms | 4.818 ms |
| Pack, render, and token count | 6.446 ms | 8.328 ms |
| Full provider-ready boundary | 52.585 ms | 71.420 ms |

For every future arm, continue recording separately:

- query tokenize/parse;
- query encoder cold load and resident forward;
- BM25;
- each exact matrix scan;
- typed/temporal/graph/CAV traversal;
- union, dedup, refill, and closure;
- late interaction when invoked;
- raw hydration;
- packing and serialization;
- candidate and token counts;
- Qwen/pass/provider call counts; and
- CPU RSS, persistent index bytes, peak VRAM, and T2 build cost.

Cold-start, warm single-query, and steady batched measurements must not be
mixed. Local retrieval ends when the provider-ready bytes exist; provider RTT,
TTFT, and completion latency are a separate envelope.

### Promotion gates

The first fast path is a GO only if all of the following hold on the
development assay:

1. warm local retrieval p95 is at most 200 ms for initial promotion, with
   100 ms as the real-time stretch target;
2. the normal path makes zero Qwen and zero provider calls;
3. protected BM25 and prior composed evidence IDs are never evicted;
4. exact annotated-turn, source, and all-required-source reach do not regress
   versus A3;
5. every addition reports unique recovered targets before and after packing;
6. the provider envelope remains within the existing 8,000-token cap;
7. persistent derived index data remains at most 512 MiB for A0--A5, unless a
   measured accuracy gain justifies a separately reported exception;
8. T0/T1 publication is unchanged; all new corpus work is resumable T2; and
9. receipts bind corpus, model/head artifact, index parameters, source code,
   and ordered outputs.

A6 late interaction must additionally show that its unique evidence rescues
justify its p95 cost. A7 must report trigger rate and expected latency, not
only latency conditional on taking or skipping the branch.

V6 passes every gate measurable on this development projection: both latency
thresholds, 10/10 all-source reach, 5/10 literal reach, 1.000 applicable
component recall, context/workspace caps, protected-BM25 survival, the 512 MiB
sidecar limit, zero drops, zero Qwen/provider calls, byte-identical replay, and
10/10 on the separate sealed responder/judge pass.
Formal promotion remains unavailable because this pinned dev1M projection
does not expose the exact annotated-turn non-regression metric. That is an
evaluation-availability caveat, not evidence that the measured retrieval gate
failed.

## Exact code seam

The ordinary dense/BM25 path is already model-light:

- `modeling/embedding.py::EmbeddingService` owns the single dense query
  encode;
- `search/indexes/hybrid_queries.py::hybrid_query` independently
  normalizes dense and BM25 candidates, unions IDs deterministically, and
  hydrates only its final top-k;
- `search/indexes/lexical.py::LexicalIndex.search` is the exact lexical
  plane; its persisted `lexical_weights` are ordinary term frequencies,
  not BGE-M3 learned sparse output;
- `search/episodes/descriptor_index.py::EpisodeDescriptorIndex.score` is
  already the desired resident exact-matrix pattern, but remains opt-in and
  shadow-only; and
- `search/packing/context_packer.py::ContextPacker.pack` already emits
  hydrated raw evidence.

The implementation leaves the validated cumulative runner unchanged and
builds the assay beside it:

1. `search/hot_lexical.py::ResidentBM25Index` compiles the durable lexical
   postings into exact resident Okapi-BM25 arrays.
2. `search/hot_retrieval.py::ExactDenseAddressIndex` owns the read-only
   normalized dense matrix and deterministic exact scan.
3. `search/source_neighborhood.py::SourceNeighborhoodIndex` compiles
   source/turn/ordinal/start-character coordinates once, then returns bounded
   adjacent-turn links without rescanning metadata per query.
4. `search/temporal_enumeration.py` plans implicit temporal-event-set queries,
   compiles query-independent role/event features, and resolves explicit dated
   lookbacks with calendar-month arithmetic. The assay filters each ranked
   lane against that inclusive window before selection.
5. `search/post_selection_lane_union.py::post_selection_lane_union` protects
   each route's budget, unions only after selection, deduplicates exact chunk
   IDs, and refills from the same lane.
6. `tools/assay_hot_retrieval_1m.py` binds those planes, hydrates exact raw
   text from the durable store, packs it through the existing context contract,
   and emits canonical provider-ready messages.
7. `search/adaptive_source_admission.py::source_balanced_surplus_admission`
   preserves a sealed parent prefix, bounds per-source lane influence, admits
   uncovered source representatives first, and fills surplus by deterministic
   source round robin.
8. `tools/assay_hot_retrieval_adaptive_full100.py` applies that selector to
   the authenticated v6 frontiers without rerunning search or opening gold.
9. `search/activated_assertion_projection.py` projects activated source
   neighborhoods into compact, provenance-bearing assertions; its output is a
   density specialist rather than a replacement raw authority.
10. `search/source_preserving_hybrid.py` protects one first v7 raw chunk per
    exact source, composes the bounded assertion prefix and raw remainder,
    deduplicates exact IDs after selection, binary-packs the result, and
    reuses the exact raw fallback if its source-seed gate cannot pass.
11. `tools/assay_hot_retrieval_source_seed_hybrid_full100.py` binds the sealed
    v7 and v2 parents and seals the provider-free v3 run, replay, and score.
12. `tools/evaluate_hot_retrieval_full100.py` exposes the explicit
    `source-seed-hybrid-v3` selection profile while preserving
    `a3_protected_union` as the terminal provider arm.

The historical locked 1M route loads Qwen through
`eval/recall_guarded_cumulative_1m.py::_load_shared_qwen`; S2 then
reaches `QwenMemoryLinker.inspect_nested` through representative episode
retrieval. V6 never attaches that shared selector. The measured development
store also has no episode, relation, artifact, or CAV rows, so those specialists
cannot honestly be credited in this run; an absent lane must fail open rather
than narrowing the packet.

## Implementation status and next order

1. **Completed: provider-free timing and receipts.** V6 separates semantic
   selection from runtime noise, rotates all ten probes over 200 samples, and
   proves byte-identical gold-blind replay.
2. **Completed: minimum-compute A3.** One dense query embedding, resident BM25,
   source-neighborhood linking, temporal-event enumeration, protected budgets,
   post-selection dedup/refill, raw hydration, and provider-ready serialization
   now run without Qwen.
3. **Completed: general temporal admissibility.** Explicit dated lookbacks now
   produce inclusive calendar windows applied to every ranked lane before
   selection. The v5 Killers decoy is absent from v6.
4. **Completed: measure answer accuracy separately.** The sealed v6 raw packets
   produced 10/10 under Terra response and Sol judging, without folding
   provider latency or gold into the retrieval artifact.
5. **Completed: locked full100 transfer and adaptive admission.** Frozen v6
   scores 66/100 with 72 complete-source packets; additive v7 scores 70/100
   with 93 complete-source packets and byte-identical retrieval replay.
6. **Completed: sealed binary prefix packing as v8.** All 100 v7 packet
   payloads are byte-identical while the quadratic overflow recount is
   replaced by logarithmic prefix search; packing p95 falls 5.113x against
   the non-contemporaneous same-machine v7 artifact.
7. **Completed: assertion-density projection v2.** The provider-free arm
   establishes a denser fact-like representation, while its 81/100
   complete-source result shows why it cannot replace raw reach.
8. **Completed: source-seed hybrid v3.** Protected raw source representatives,
   a 1,400-token projection prefix, the raw remainder, post-selection exact-ID
   deduplication, binary packing, and an exact raw fallback retain 93/100
   source reach while improving literal containment to 56/100 and mean best F1
   to 0.115009. No semantic result is claimed.
9. **Measured but rejected: provider-free typed/turn successor.** It raises
   complete source-ID reach from 93/100 to 97/100, but an adversarial audit
   found that derivative clauses/windows reused physical chunk IDs and could
   displace authoritative parent text. Five questions regress best-evidence
   F1, so the artifact is a useful structural diagnostic, not a promoted
   parent or a semantic-accuracy result. Repair physical occurrence identity,
   link-seed budget waste, and linear packing before rerunning.
10. **Next: repair and reseal the provider-free successor.** Hydrate exact raw
   chunks from specialist addresses, preserve parent authority on conflicting
   IDs, treat link seeds as activation rather than output evidence, and use
   binary ranked-prefix packing. Then rerun the residual and full100
   provider-free assays before requesting any semantic calls.
11. **Build the bounded conversational phrase graph only on the repaired
   parent.** Use the append-at-ingest, two-hop, independently budgeted design
   in Analysis 32; first target the remaining graph-eligible connectivity
   failures rather than replacing the hot lanes.
12. **Persist resident setup.** Load the query encoder, resident BM25,
   dense matrix, and source-neighborhood coordinates once per service process;
   measure startup and steady-state query latency as separate envelopes.
13. **Build another address arm only for a measured residual.** If validation
   failures reveal missing evidence rather than reasoning failures, compile
   atom/fact/span dense addresses with the pinned encoder before downloading a
   new model.
14. **Inventory and authenticate BGE-M3 multifunction heads.** If still needed,
   add the complete FlagEmbedding artifact as a versioned sidecar and assay
   learned sparse separately.
15. **Add shortlist-only late interaction only when justified.** Start with a
   simple exact implementation. Add PLAID, WARP, MUVERA, JAX, Numba, Rust, or
   custom CUDA only if profiling proves that arithmetic/kernel is the new
   bottleneck.
16. **Try compact CAV/latent blocks only for a measured residual class.**
17. **Optimize final-model prefill last.** Prefix/chunk KV reuse or a
   sparse-native answer model matters only after retrieval is bounded and only
   when the serving boundary can consume its state.

## What this can and cannot guarantee

The architecture can guarantee bounded online work for fixed limits, exact
provenance hydration, non-eviction of the subset named by each protection gate,
post-selection deduplication, deterministic ordering, and complete receipts.
In v3 the source-seed gate guarantees that every represented exact source
retains its first protected raw row or that the system returns the exact valid
v7 raw fallback.

It cannot mathematically guarantee 95% semantic recall under a bounded prompt:
every learned address can rank a relevant item below its cap, every summary
can omit a fact, and some questions require synthesis rather than discovery.
The practical guarantee is explicitly scoped fail-open composition: no compact
method may silently remove evidence named by its protection gate. In v3 that
protected set is the first raw representative per exact source, not the whole
v7 raw remainder. An unresolved typed obligation widens raw retrieval instead
of pretending that a compressed state is complete.

That prediction is now supported on dev1M: replacing the sequential Qwen
tournament with the resident linked union removed nearly all of the historical
local latency while retaining 10/10 labeled-source reach, restoring 5/10
literal reach, closing both parseable multi-event component sets, and staying
inside the prompt caps with no drops. The v6 calendar-window constraint also
removed the observed temporal decoy and produced 10/10 on the sealed Terra/Sol
development pass. The broader comparative run now shows the complementary
limit: v7 recovers 93/100 complete-source packets but only 70/100 semantic
answers. What remains unproven is exact annotated-turn non-regression on a
projection that exposes it, generalization beyond the analysis-used
validation questions, and a lightweight answer policy that uses the added
evidence monotonically. Byte-equivalent v8 binary packing is now sealed. The
source-seed assertion hybrid now preserves v7's 93/100 complete-source reach
while improving provider-free density and literal containment, but its source
gate cannot preserve every answer-bearing row in the raw remainder. The
immediate experiments are a separately authorized v3 semantic lifecycle, an
integrated resident-path latency measurement, and separate treatment of the
two residual admission misses, five frontier misses, and answer-policy
failures—without putting a second local generator back onto every retrieval
critical path.
